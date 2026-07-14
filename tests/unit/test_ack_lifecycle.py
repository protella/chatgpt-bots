"""F38 — the 👀 as a CLAIM ON WORK, not an acknowledgment of receipt.

The rule, in the user's words: "Other human teammates don't drop eyes and then do nothing,
that's misleading. If it adds it, it needs to do something, or go back and remove it."

So: the eye goes on when real, slow work actually starts — never on a prediction that work is
coming, never on a tool call that's about to be rejected — and comes back off if the turn ends
up producing nothing the user can see.

The three things codex called most likely to bite, each pinned here:
  1. treating Slack's `already_reacted` as ownership (and then removing someone else's 👀)
  2. a stale lease removing a reaction a CONCURRENT turn has since made its own
  3. claiming work from the generic tool hook, which fires before validation
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import config
from message_processor.turn_runtime import TurnRuntime
from message_processor.handlers.text import _claims_work, _reaction_committed
from slack_client.messaging import SlackMessagingMixin
from slack_client.channel_pulse import ChannelPulse


def _msg(channel="C1", ts="100.0"):
    from base_client import Message
    return Message(text="build me a deck", user_id="U1", channel_id=channel,
                   thread_id=ts, metadata={"ts": ts})


class _Host(SlackMessagingMixin):
    """A real messaging mixin over a fake Slack client, so the guard/lease code is exercised
    for real rather than mocked away."""

    def __init__(self, pulse=None, add_error=None, remove_error=None):
        self.added, self.removed = [], []

        async def reactions_add(channel, name, timestamp):
            if add_error:
                raise add_error
            self.added.append((channel, name, timestamp))

        async def reactions_remove(channel, name, timestamp):
            if remove_error:
                raise remove_error
            self.removed.append((channel, name, timestamp))

        self.app = SimpleNamespace(client=SimpleNamespace(
            reactions_add=reactions_add, reactions_remove=reactions_remove))
        self.channel_pulse = pulse

    def log_debug(self, *a, **k):
        pass

    log_info = log_warning = log_error = log_debug


def _slack_error(code):
    from slack_sdk.errors import SlackApiError
    return SlackApiError("boom", response={"error": code})


# --------------------------------------------------------------- ownership: the lease

@pytest.mark.asyncio
async def test_a_genuine_add_yields_a_lease(monkeypatch):
    monkeypatch.setattr(config, "enable_reactions", True)
    host = _Host()
    result, lease = await host._reserve_and_react_owned("C1", "100.0", "eyes")
    assert result["ok"] is True
    assert lease is not None and lease["emoji"] == "eyes"
    assert host.added == [("C1", "eyes", "100.0")]


@pytest.mark.asyncio
async def test_already_reacted_yields_no_lease(monkeypatch):
    """THE trap. Slack says already_reacted when the reaction is already on the message — the
    emoji is present, so the call succeeds, but WE did not put it there this time. A lease
    here would let this turn remove a reaction a previous turn deliberately placed."""
    monkeypatch.setattr(config, "enable_reactions", True)
    host = _Host(add_error=_slack_error("already_reacted"))
    result, lease = await host._reserve_and_react_owned("C1", "100.0", "eyes")
    assert result["ok"] is True          # the emoji IS there — the caller's intent is met
    assert result.get("idempotent") is True
    assert lease is None                 # ...but it is not ours to take back


@pytest.mark.asyncio
async def test_a_failed_add_yields_no_lease(monkeypatch):
    monkeypatch.setattr(config, "enable_reactions", True)
    host = _Host(add_error=_slack_error("invalid_name"))
    result, lease = await host._reserve_and_react_owned("C1", "100.0", "eyes")
    assert result["ok"] is False
    assert lease is None


@pytest.mark.asyncio
async def test_duplicate_call_in_the_same_turn_yields_no_second_lease(monkeypatch):
    monkeypatch.setattr(config, "enable_reactions", True)
    host = _Host()
    _r1, lease1 = await host._reserve_and_react_owned("C1", "100.0", "eyes")
    _r2, lease2 = await host._reserve_and_react_owned("C1", "100.0", "eyes")
    assert lease1 is not None
    assert lease2 is None                # committed slot: idempotent, and only one owner
    assert len(host.added) == 1


# --------------------------------------------------------------- removal

@pytest.mark.asyncio
async def test_removing_an_owned_reaction_cleans_slack_guard_and_pulse(monkeypatch):
    monkeypatch.setattr(config, "enable_reactions", True)
    pulse = ChannelPulse(size=10)
    pulse.record("C1", ts="100.0", thread_ts=None, user_id="U1",
                 display_name="Peter", sender_type="human", text="build me a deck", is_bot=False)
    host = _Host(pulse=pulse)

    _result, lease = await host._reserve_and_react_owned("C1", "100.0", "eyes")
    # The synthetic "[reacted :eyes: ...]" history entry is in the ring...
    assert any("reacted :eyes:" in (e.get("text") or "")
               for e in pulse._buffers["C1"])

    assert await host.remove_owned_reaction(lease) is True
    assert host.removed == [("C1", "eyes", "100.0")]
    # ...and gone again. Leave it and the classifier reads a phantom reaction on the next
    # message and reasons from a thing that is no longer on screen.
    assert not any("reacted :eyes:" in (e.get("text") or "")
                   for e in pulse._buffers["C1"])
    # The F6 slot is freed too, so a later deliberate re-add can land.
    assert host._reaction_guard.get(("C1", "100.0")) in (None, {})


@pytest.mark.asyncio
async def test_an_evicted_claim_cannot_remove_the_reaction_it_no_longer_owns(monkeypatch):
    """The race that killed the first design, driven through the REAL code path.

    Ownership used to live in a map beside the guard, and the argument was "if a concurrent
    turn re-adds the emoji it overwrites the owner token, so a stale lease can't fire". That
    argument is wrong, because Slack's `already_reacted` is silent about WHO reacted:

        1. turn A adds 👀 and records itself as owner
        2. A's guard entry is evicted (the LRU is bounded)
        3. turn B reserves the same emoji: no slot, so it calls Slack
        4. Slack says already_reacted — A's 👀 is still up there — so B gets no lease
        5. ...and B never overwrote A's ownership record, because it never had one to write
        6. A ends silently, its token still "matches", and it rips the 👀 out from under B

    Ownership now lives IN the slot, so eviction destroys the claim: A can no longer prove the
    reaction is its own and declines to touch it. Losing the right to clean up is the safe
    failure. Removing someone else's reaction is not.
    """
    monkeypatch.setattr(config, "enable_reactions", True)
    host = _Host()
    _r, lease_a = await host._reserve_and_react_owned("C1", "100.0", "eyes")
    assert lease_a is not None

    # (2) A's guard entry is evicted.
    host._reaction_guard.pop(("C1", "100.0"))
    host._reaction_guard_ts.pop(("C1", "100.0"), None)

    # (3)+(4) turn B reserves the same emoji and Slack reports it's already there.
    host_b_add_error = _slack_error("already_reacted")

    async def reactions_add(channel, name, timestamp):
        raise host_b_add_error
    host.app.client.reactions_add = reactions_add

    result_b, lease_b = await host._reserve_and_react_owned("C1", "100.0", "eyes")
    assert result_b["ok"] is True and lease_b is None   # present, but not B's to remove

    # (6) A goes silent and tries to clean up. It must NOT strip the reaction.
    assert await host.remove_owned_reaction(lease_a) is False
    assert host.removed == []


@pytest.mark.asyncio
async def test_a_lease_from_a_different_turn_cannot_remove_this_ones(monkeypatch):
    """Same emoji, same message, a genuinely different owner token → refuse."""
    monkeypatch.setattr(config, "enable_reactions", True)
    host = _Host()
    _r, lease = await host._reserve_and_react_owned("C1", "100.0", "eyes")
    forged = dict(lease, token="not-the-owners-token")
    assert await host.remove_owned_reaction(forged) is False
    assert host.removed == []


@pytest.mark.asyncio
async def test_a_permanent_reaction_does_not_leave_an_unevictable_claim(monkeypatch):
    """`_reserve_and_react` is for reactions nobody takes back (a gate verdict, the model's
    own react tool). It must SETTLE the lease it discards — an owned slot is pinned against
    eviction, so leaking one would slowly fill the guard with entries that can never age out,
    and could push a live work-claim out of it."""
    monkeypatch.setattr(config, "enable_reactions", True)
    host = _Host()
    result = await host._reserve_and_react("C1", "100.0", "tada")
    assert result["ok"] is True
    slot = host._reaction_guard[("C1", "100.0")]["tada"]
    assert slot is True                  # committed, unowned, evictable — not a live claim


@pytest.mark.asyncio
async def test_settling_a_lease_keeps_the_reaction(monkeypatch):
    monkeypatch.setattr(config, "enable_reactions", True)
    host = _Host()
    _r, lease = await host._reserve_and_react_owned("C1", "100.0", "eyes")
    host.settle_reaction_lease(lease)
    assert host.removed == []
    # Ownership released, so a stale retry can't come back later and remove it.
    assert await host.remove_owned_reaction(lease) is False


@pytest.mark.asyncio
async def test_a_failed_removal_leaves_the_bookkeeping_intact(monkeypatch):
    """If Slack refuses the remove, the reaction may well still be on the message. Better a
    stale 👀 than a guard that thinks a slot is free when the emoji is still there."""
    monkeypatch.setattr(config, "enable_reactions", True)
    host = _Host(remove_error=_slack_error("internal_error"))
    _r, lease = await host._reserve_and_react_owned("C1", "100.0", "eyes")
    assert await host.remove_owned_reaction(lease) is False
    assert host._reaction_guard[("C1", "100.0")]["eyes"] is True   # still committed


@pytest.mark.asyncio
async def test_no_reaction_counts_as_a_successful_removal(monkeypatch):
    """Someone else already took it off. The goal state — "the emoji is not there" — holds."""
    monkeypatch.setattr(config, "enable_reactions", True)
    host = _Host(remove_error=_slack_error("no_reaction"))
    _r, lease = await host._reserve_and_react_owned("C1", "100.0", "eyes")
    assert await host.remove_owned_reaction(lease) is True


# --------------------------------------------------------------- claim_work

@pytest.mark.asyncio
async def test_claim_work_is_idempotent_across_many_tools(monkeypatch):
    """A turn that searches the web, runs code AND calls MCP places exactly one 👀."""
    monkeypatch.setattr(config, "enable_reactions", True)
    monkeypatch.setattr(config, "enable_ack_reaction", True, raising=False)
    monkeypatch.setattr(config, "ack_reaction_emoji", "eyes", raising=False)
    host = _Host()
    turn = TurnRuntime(silence_capable=True, progress_enabled=False)
    msg = _msg()
    for _ in range(5):
        await turn.claim_work(host, msg)
    assert len(host.added) == 1
    assert turn.ack_lease is not None


@pytest.mark.asyncio
async def test_claim_work_is_silent_when_slack_fails(monkeypatch):
    monkeypatch.setattr(config, "enable_reactions", True)
    monkeypatch.setattr(config, "enable_ack_reaction", True, raising=False)
    client = MagicMock()
    client._reserve_and_react_owned = AsyncMock(side_effect=RuntimeError("slack down"))
    turn = TurnRuntime()
    await turn.claim_work(client, _msg())      # must not raise — an emoji never fails a turn
    assert turn.ack_lease is None


@pytest.mark.asyncio
async def test_claim_work_respects_the_feature_flag(monkeypatch):
    monkeypatch.setattr(config, "enable_ack_reaction", False, raising=False)
    client = MagicMock()
    client._reserve_and_react_owned = AsyncMock()
    turn = TurnRuntime()
    await turn.claim_work(client, _msg())
    client._reserve_and_react_owned.assert_not_awaited()


# --------------------------------------------------------------- which events claim

def test_only_slow_hosted_tools_claim_work():
    # Real work, genuinely slow: claim.
    assert _claims_work("web_search", "started")
    assert _claims_work("file_search", "started")
    assert _claims_work("code_interpreter", "started")
    assert _claims_work("image_generation", "started")
    assert _claims_work("mcp:datassential", "calling")
    assert _claims_work("mcp", "calling")


def test_plumbing_and_unvalidated_calls_never_claim_work():
    # MCP *discovery* runs before the model has decided to call anything.
    assert not _claims_work("mcp", "discovering_tools")
    assert not _claims_work("mcp", "tools_discovered")
    # Completion is not a start.
    assert not _claims_work("web_search", "completed")
    # THE trap codex caught: local tool events fire the instant a call is DISPATCHED —
    # before its arguments are validated and before a duplicate background job is rejected.
    # Claiming here would flash an eye on a call that never actually happened. Slow local
    # tools claim from inside their own executors instead, once they know they'll do the work.
    assert not _claims_work("local:start_background_job", "started")
    assert not _claims_work("local:read_document", "started")
    assert not _claims_work("local:save_memory", "started")
    assert not _claims_work("local:react_to_message", "started")


# --------------------------------------------------------------- settle: did we do the thing?

def _settle_client():
    client = MagicMock()
    client.settle_reaction_lease = MagicMock()
    client.remove_owned_reaction = AsyncMock(return_value=True)
    return client


@pytest.mark.asyncio
async def test_produced_output_keeps_the_eye():
    client = _settle_client()
    turn = TurnRuntime(ack_lease={"token": "t"})
    await turn.settle_ack(client, produced_output=True)
    client.settle_reaction_lease.assert_called_once()
    client.remove_owned_reaction.assert_not_awaited()


@pytest.mark.asyncio
async def test_producing_nothing_takes_the_eye_back():
    client = _settle_client()
    turn = TurnRuntime(ack_lease={"token": "t"})
    await turn.settle_ack(client, produced_output=False)
    client.remove_owned_reaction.assert_awaited_once()
    client.settle_reaction_lease.assert_not_called()


def test_an_interrupted_turn_retracts_its_claim():
    """The fail-closed path posts "I got cut off partway through that answer" into the one
    surface it has, so `posted` is True — but the turn claimed work and delivered none of it.
    Reading `posted` alone would keep the 👀 on a turn that visibly failed."""
    from main import ChatBotV2
    from base_client import Response

    interrupted = Response(type="text", content="",
                           metadata={"streamed": True, "posted": True, "interrupted": True})
    turn = TurnRuntime()
    assert ChatBotV2._produced_visible_output(interrupted, turn) is False

    # ...unless a tool already delivered something real, which the interruption doesn't undo.
    turn.visible_action_committed = True
    assert ChatBotV2._produced_visible_output(interrupted, turn) is True


@pytest.mark.asyncio
async def test_settle_without_a_claim_does_nothing():
    client = _settle_client()
    turn = TurnRuntime(ack_lease=None)     # no slow tool ever ran: there is no eye
    await turn.settle_ack(client, produced_output=False)
    client.remove_owned_reaction.assert_not_awaited()
    client.settle_reaction_lease.assert_not_called()


def test_a_deliberate_response_reaction_counts_as_output():
    # A no_reply turn may react instead of replying. That reaction IS the answer.
    assert _reaction_committed([{"name": "react_to_message", "ok": True}])
    assert not _reaction_committed([{"name": "react_to_message", "ok": False}])
    assert not _reaction_committed([{"name": "save_memory", "ok": True}])
    assert not _reaction_committed([])


@pytest.mark.asyncio
async def test_a_cancelled_turn_still_completes_its_removal(monkeypatch):
    """A turn cancelled mid-removal must not strand the slot in `removing` — a live claim is
    PINNED against eviction, so a run of cancelled turns would grow the supposedly bounded
    guard without limit, and the 👀 would be left up with nobody able to prove they own it.

    What makes this safe is that the removal is its OWN task and the caller only shields a
    wait on it: cancelling the turn detaches the waiter and leaves the removal running."""
    import asyncio

    monkeypatch.setattr(config, "enable_reactions", True)
    started = asyncio.Event()

    async def slow_remove(channel, name, timestamp):
        started.set()
        await asyncio.sleep(0.05)
        host.removed.append((channel, name, timestamp))

    host = _Host()
    host.app.client.reactions_remove = slow_remove
    _r, lease = await host._reserve_and_react_owned("C1", "100.0", "eyes")

    task = asyncio.ensure_future(host.remove_owned_reaction(lease))
    await started.wait()
    task.cancel()                      # the turn is torn down mid-removal
    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.sleep(0.15)           # the removal task lives on and finishes the job
    assert host.removed == [("C1", "eyes", "100.0")]
    assert host._reaction_guard.get(("C1", "100.0")) in (None, {}), \
        "the slot is stranded in `removing` — it will never be evicted"


@pytest.mark.asyncio
async def test_even_a_killed_removal_task_settles_the_slot(monkeypatch):
    """Defense in depth for the one case the shield can't cover: the removal TASK itself being
    cancelled (loop teardown). `except Exception` would miss it — CancelledError is a
    BaseException — so the state transition lives in a `finally`. The reaction may survive,
    but the slot must not stay `removing`, because `removing` is unevictable."""
    import asyncio

    monkeypatch.setattr(config, "enable_reactions", True)
    host = _Host()

    async def hangs(channel, name, timestamp):
        await asyncio.sleep(30)

    host.app.client.reactions_remove = hangs
    _r, lease = await host._reserve_and_react_owned("C1", "100.0", "eyes")

    # Exactly what remove_owned_reaction sets up, then the task gets killed under it.
    inner = asyncio.ensure_future(host._run_reaction_removal(
        "C1", "100.0", "eyes", lease["token"], lease))
    host._reaction_guard[("C1", "100.0")]["eyes"] = {
        "token": lease["token"], host._REMOVING: inner}
    await asyncio.sleep(0)
    inner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await inner

    slot = host._reaction_guard[("C1", "100.0")]["eyes"]
    assert slot is True, "slot left in `removing` — pinned forever, never evictable"


@pytest.mark.asyncio
async def test_a_reservation_during_removal_waits_for_the_outcome(monkeypatch):
    """A `removing` slot must not answer "the emoji is there".

    If it did, the model's react_to_message would get ok=True without calling Slack, and then
    the removal would land and delete the emoji — producing a reaction-only reply whose
    reaction does not exist. The reserver waits for the real outcome and re-adds."""
    import asyncio

    monkeypatch.setattr(config, "enable_reactions", True)
    host = _Host()

    async def slow_remove(channel, name, timestamp):
        await asyncio.sleep(0.05)
        host.removed.append((channel, name, timestamp))

    host.app.client.reactions_remove = slow_remove
    _r, lease = await host._reserve_and_react_owned("C1", "100.0", "eyes")
    assert len(host.added) == 1

    remover = asyncio.ensure_future(host.remove_owned_reaction(lease))
    await asyncio.sleep(0)              # let the removal publish its `removing` slot
    result, lease2 = await host._reserve_and_react_owned("C1", "100.0", "eyes")
    await remover

    # The reservation waited, saw the emoji really go, and put it back — for real.
    assert result["ok"] is True
    assert lease2 is not None, "the second reserver must own the reaction it re-added"
    assert len(host.added) == 2, "it reported success without ever calling Slack"


@pytest.mark.asyncio
async def test_a_failed_removal_leaves_a_reservation_reporting_present(monkeypatch):
    """The other half: if the remove FAILS, the emoji is still up there, so a reserver that
    waited on it should get idempotent success rather than adding a duplicate."""
    monkeypatch.setattr(config, "enable_reactions", True)
    host = _Host(remove_error=_slack_error("internal_error"))
    _r, lease = await host._reserve_and_react_owned("C1", "100.0", "eyes")

    import asyncio
    remover = asyncio.ensure_future(host.remove_owned_reaction(lease))
    await asyncio.sleep(0)
    result, lease2 = await host._reserve_and_react_owned("C1", "100.0", "eyes")
    assert await remover is False

    assert result["ok"] is True and result.get("idempotent") is True
    assert lease2 is None
    assert len(host.added) == 1        # no duplicate add


@pytest.mark.asyncio
async def test_a_churning_slot_never_returns_a_malformed_result(monkeypatch):
    """A reserver can be overtaken repeatedly — it waits out one removal, and by the time it
    looks again the emoji has been re-added and is being removed again. The old fixed two-pass
    retry could exhaust its passes on exactly this and fall through returning `(None, None)`,
    which the react tool then subscripts (`result["ok"]` → TypeError).

    Rather than choreograph an exact generation count — the scheduler decides who wins, and an
    earlier version of this test quietly proved nothing because the waiting reserver claimed
    the slot before generation 2 could start — hammer the same slot with concurrent claims and
    removals and assert the invariant that actually matters: every reservation returns a
    well-formed result, never None.
    """
    import asyncio

    monkeypatch.setattr(config, "enable_reactions", True)
    host = _Host()

    async def slow_remove(channel, name, timestamp):
        await asyncio.sleep(0.01)
        host.removed.append((channel, name, timestamp))

    host.app.client.reactions_remove = slow_remove

    async def claim_and_drop():
        result, lease = await host._reserve_and_react_owned("C1", "100.0", "eyes")
        assert result is not None, "the retry loop fell through and returned nothing"
        assert isinstance(result, dict) and "ok" in result
        if lease is not None:
            await host.remove_owned_reaction(lease)
        return result

    results = await asyncio.gather(*(claim_and_drop() for _ in range(6)))
    for r in results:
        assert r["ok"] is True or r["error"] in ("reaction_busy", "reaction_failed")


@pytest.mark.asyncio
async def test_an_add_waiter_rechecks_before_reporting_success(monkeypatch):
    """B waits on C's in-flight add. C's add succeeds and C gets the lease — then C decides it
    produced nothing and starts removing. If B trusted the add future's `True`, it would report
    a reaction that is on its way out (and for react_to_message, promise one that won't exist).
    B must re-read the slot after waking."""
    import asyncio

    monkeypatch.setattr(config, "enable_reactions", True)
    host = _Host()
    gate = asyncio.Event()

    async def slow_add(channel, name, timestamp):
        await gate.wait()
        host.added.append((channel, name, timestamp))

    host.app.client.reactions_add = slow_add

    c = asyncio.ensure_future(host._reserve_and_react_owned("C1", "100.0", "eyes"))
    await asyncio.sleep(0)
    b = asyncio.ensure_future(host._reserve_and_react_owned("C1", "100.0", "eyes"))
    await asyncio.sleep(0)

    gate.set()                       # C's add lands; B is still parked on the future
    _rc, lease_c = await c
    assert lease_c is not None
    await host.remove_owned_reaction(lease_c)   # C produced nothing, takes it back

    result_b, _lease_b = await b
    assert result_b is not None
    # Whatever B answers, it must not be a bare "yes, it's there" derived from the stale add.
    if result_b.get("ok"):
        slot = host._reaction_guard.get(("C1", "100.0"), {}).get("eyes")
        assert slot is not None, "B reported the emoji present after it was removed"


@pytest.mark.asyncio
async def test_a_removal_wait_that_times_out_is_reported_busy(monkeypatch):
    """Timing out on someone else's removal is NOT evidence the reaction survived — the task
    is still running and may yet succeed. Guessing "still present" is the same lie, one step
    later. Say so instead."""
    import asyncio

    monkeypatch.setattr(config, "enable_reactions", True)
    monkeypatch.setattr(config, "tool_call_timeout", 0.05, raising=False)
    host = _Host()

    async def very_slow_remove(channel, name, timestamp):
        await asyncio.sleep(5)

    host.app.client.reactions_remove = very_slow_remove
    _r, lease = await host._reserve_and_react_owned("C1", "100.0", "eyes")
    remover = asyncio.ensure_future(host.remove_owned_reaction(lease))
    await asyncio.sleep(0)

    result, lease2 = await host._reserve_and_react_owned("C1", "100.0", "eyes")
    assert result["ok"] is False and result["error"] == "reaction_busy"
    assert lease2 is None

    remover.cancel()
    with pytest.raises(asyncio.CancelledError):
        await remover


@pytest.mark.asyncio
async def test_the_claim_cannot_stall_the_work_it_announces(monkeypatch):
    """The 👀 is placed from inside the tool callback, so for a hosted tool the Responses
    event loop is waiting on it. A wedged Slack call must not hold up the web search."""
    import asyncio

    monkeypatch.setattr(config, "enable_reactions", True)
    monkeypatch.setattr(config, "enable_ack_reaction", True, raising=False)
    monkeypatch.setattr(config, "tool_call_timeout", 0.05, raising=False)

    async def _never_returns(*a, **k):
        await asyncio.sleep(30)

    client = MagicMock()
    client._reserve_and_react_owned = _never_returns

    turn = TurnRuntime()
    await asyncio.wait_for(turn.claim_work(client, _msg()), timeout=2)  # bounded, not hung
    assert turn.ack_lease is None
