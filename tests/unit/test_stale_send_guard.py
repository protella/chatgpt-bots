"""The stale-send guard: don't answer a message the conversation has already moved past.

Every race here is driven by an asyncio.Event or a Barrier, never a sleep. A sleep-based race
test is a coin flip that passes on a fast machine and fails in CI, and the thing under test is
precisely an ordering — so the ordering has to be forced, not hoped for.

What these hold to:

1. SCOPE. A same-sender top-level fast-follow stales the first turn; two different people's
   top-level questions never collide; a reply under a root stales the turn answering that root.
2. MONOTONICITY. Out-of-order execution cannot lower a watermark, and an edit (which keeps its
   original ts) never supersedes itself.
3. LIFETIME. A newer turn that finishes first does not erase the watermark an older, still-open
   turn is about to read — the entry lives while any lease in its scope does.
4. ADMISSION. Only messages that actually reached a turn raise the watermark. Anything dropped
   before dispatch owns no successor, and letting it raise the mark would silence a valid answer.
5. NO SLACK CALL. A suppressed turn makes zero API calls — the check runs BEFORE the mutation,
   which is the only thing that makes this a guard rather than an apology.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from message_processor.stale_send_guard import (COMMITTED, PENDING, SUPPRESSED,
                                                ConversationWatermarks,
                                                StaleSendSuppressed, TurnSendLease, is_newer,
                                                primary_scope_key, scopes_for, ts_key)


def _msg(ts, *, channel="C1", thread=None, sender="U1"):
    return SimpleNamespace(channel_id=channel, thread_id=thread or ts,
                           metadata={"ts": ts, "sender_id": sender})


# --------------------------------------------------------------------------- the comparator

def test_slack_timestamps_compare_numerically_never_lexically():
    """'9.0' is older than '10.0'. Compared as strings it is not, and that is a whole class of
    bug in a system whose every ordering question is a Slack ts."""
    assert ts_key("9.0") < ts_key("10.0")
    assert is_newer("10.0", "9.0") and not is_newer("9.0", "10.0")
    assert is_newer("100.000002", "100.000001")


def test_a_missing_or_malformed_ts_never_wins():
    assert not is_newer(None, "1.0")
    assert is_newer("1.0", None)          # anything beats nothing
    assert not is_newer("garbage", "1.0")  # unparseable sorts to zero, never supersedes


def test_an_edit_does_not_supersede_itself():
    """An edit re-dispatch keeps the ORIGINAL ts, so a strictly-newer test leaves it alone —
    edit supersession stays the edit path's job."""
    assert not is_newer("100.0", "100.0")


# ------------------------------------------------------------------------------- the scopes

def test_a_top_level_message_is_watched_by_both_scopes():
    scopes = scopes_for("C1", "100.0", "100.0", "U1")
    assert scopes == (("thread", "C1", "100.0"), ("top", "C1", "U1"))


def test_a_thread_reply_is_watched_only_by_its_thread():
    scopes = scopes_for("C1", "101.0", "100.0", "U1")
    assert scopes == (("thread", "C1", "100.0"),)


def test_an_unattributed_message_gets_no_top_scope():
    """Bucketing strangers under "unknown" would let one person's message silence another's
    answer. The thread scope still applies — that one is keyed by the conversation, not by who
    happened to send it."""
    scopes = scopes_for("C1", "100.0", "100.0", None)
    assert scopes == (("thread", "C1", "100.0"),)


def test_the_participation_key_keeps_its_own_unknown_bucket():
    """The two rules differ on purpose. Collapsing unattributed messages in the burst logic
    merges a reply; omitting the scope in the guard prevents a wrongful silence. Same question,
    opposite consequence, so the shared helper preserves the participation table exactly."""
    assert primary_scope_key("C1", "100.0", "100.0", None) == "C1|top|unknown"
    assert primary_scope_key("C1", "100.0", "100.0", "U1") == "C1|top|U1"
    assert primary_scope_key("C1", "101.0", "100.0", "U1") == "C1|100.0"


# ------------------------------------------------------------------------- collision rules

def test_a_same_sender_fast_follow_stales_the_first_turn():
    marks = ConversationWatermarks()
    first = marks.begin_turn(_msg("100.0"))
    marks.begin_turn(_msg("101.0"))              # same person, straight after

    with pytest.raises(StaleSendSuppressed) as caught:
        first.authorize("final_post")
    assert caught.value.observed_latest_ts == "101.0"
    assert caught.value.scope[0] == "top"
    assert first.state == SUPPRESSED


def test_two_different_people_never_collide():
    """Both questions are real and both deserve an answer. Keying the top-level scope per
    sender is the whole reason this is safe to run in a busy channel."""
    marks = ConversationWatermarks()
    alice = marks.begin_turn(_msg("100.0", sender="U_ALICE"))
    bob = marks.begin_turn(_msg("101.0", sender="U_BOB"))

    alice.authorize("final_post")     # no raise
    bob.authorize("final_post")
    assert alice.state == PENDING and bob.state == PENDING


def test_a_reply_under_a_root_stales_the_turn_answering_that_root():
    """Cross-author on purpose: the reply lands in the same thread with the full history, so
    the answer that was being written for the root is now the wrong shape."""
    marks = ConversationWatermarks()
    root_turn = marks.begin_turn(_msg("100.0", sender="U_ALICE"))
    marks.begin_turn(_msg("100.5", thread="100.0", sender="U_BOB"))

    with pytest.raises(StaleSendSuppressed) as caught:
        root_turn.authorize("final_post")
    assert caught.value.scope[0] == "thread"


def test_a_reply_in_a_different_thread_is_not_a_collision():
    marks = ConversationWatermarks()
    first = marks.begin_turn(_msg("100.0", thread="50.0", sender="U1"))
    marks.begin_turn(_msg("101.0", thread="60.0", sender="U2"))
    first.authorize("final_post")     # different conversations entirely


def test_a_third_message_supersedes_both_earlier_turns():
    marks = ConversationWatermarks()
    first = marks.begin_turn(_msg("100.0"))
    second = marks.begin_turn(_msg("101.0"))
    marks.begin_turn(_msg("102.0"))

    for lease in (first, second):
        with pytest.raises(StaleSendSuppressed):
            lease.authorize("final_post")


# --------------------------------------------------------------------------- monotonicity

@pytest.mark.asyncio
async def test_out_of_order_execution_cannot_lower_the_watermark():
    """Two turns admitted in one order and RESUMED in the other. The barrier forces exactly the
    interleaving that a sleep-based test would only sometimes produce."""
    marks = ConversationWatermarks()
    both_admitted = asyncio.Barrier(2)
    results = {}

    async def turn(ts, name):
        lease = marks.begin_turn(_msg(ts))
        await both_admitted.wait()          # neither proceeds until both are on the record
        try:
            lease.authorize("final_post")
            results[name] = "permitted"
        except StaleSendSuppressed:
            results[name] = "suppressed"
        return lease

    # The NEWER turn resumes first; the older one must still find the higher watermark.
    newer, older = await asyncio.gather(turn("101.0", "newer"), turn("100.0", "older"))
    assert results == {"newer": "permitted", "older": "suppressed"}
    assert newer.state == PENDING and older.state == SUPPRESSED


def test_an_older_admission_never_lowers_an_existing_mark():
    marks = ConversationWatermarks()
    marks.begin_turn(_msg("101.0"))
    late = marks.begin_turn(_msg("100.0"))    # delivered late, genuinely older
    scope = ("top", "C1", "U1")
    assert marks.latest_for(scope) == "101.0"
    with pytest.raises(StaleSendSuppressed):
        late.authorize("final_post")


# ------------------------------------------------------------------------------- lifetime

def test_a_newer_turn_finishing_first_does_not_erase_the_watermark():
    """THE LIFETIME RULE. If the entry vanished when the newer turn closed, the older turn —
    still writing — would look up its scope, find nothing, and post the answer the newer message
    superseded. Entries live while any lease in the scope does."""
    marks = ConversationWatermarks()
    older = marks.begin_turn(_msg("100.0"))
    newer = marks.begin_turn(_msg("101.0"))

    newer.close()                              # the successor finishes first
    assert marks.tracked_scopes == 2           # …and the older lease still holds the scopes

    with pytest.raises(StaleSendSuppressed):
        older.authorize("final_post")

    older.close()
    assert marks.tracked_scopes == 0           # bounded by concurrency, with no timer anywhere


def test_closing_is_idempotent_and_releases_exactly_once():
    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg("100.0"))
    other = marks.begin_turn(_msg("100.0", sender="U2"))
    lease.close()
    lease.close()                              # a double close must not free somebody else's hold
    assert marks.tracked_scopes >= 1
    other.close()
    assert marks.tracked_scopes == 0


# ------------------------------------------------------------------------- what this turn owns

def test_absorbing_a_batched_source_keeps_the_turn_current():
    """A drained queue hands its newest message to the successor as the trigger. Having actually
    answered it, the turn is not stale for it."""
    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg("102.0"))    # the drain's trigger IS the newest queued message
    lease.advance_last_seen("102.0")
    lease.authorize("final_post")


def test_the_ownership_ceiling_refuses_a_source_from_a_later_turn():
    """An older turn must not absorb a newer same-scope message through a concurrent history
    rebuild: that message already woke a successor, and both turns would answer it."""
    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg("100.0"))
    assert lease.owns("100.0") and lease.owns("99.0")
    assert not lease.owns("101.0")


# ------------------------------------------ a channel request is not one conversation

def _channel_items(*messages):
    """The stream items a channel request carries, off the real serializer.

    The assembler forwards each `message_items` entry verbatim — role, content, metadata — so
    these are the items the lease actually reads on a channel turn.
    """
    from tests.unit.channel_turn_harness import build_stream

    stream = build_stream(messages)
    return [{"role": item.role, "content": item.content, "metadata": dict(item.metadata)}
            for item in stream.message_items]


def _turn_answering_root_100():
    """A turn answering root 100.0 that owns the window up to 110.0 and has so far accounted
    only for the root.

    The two marks are independent by design — the ceiling says what this turn OWNS, `last_seen_ts`
    what it has ACCOUNTED FOR — and they are set apart here so the SCOPE rule is what gets
    measured. Equal, the ownership ceiling would refuse every candidate on its own and the scoping
    would never run.
    """
    lease = TurnSendLease(scopes=scopes_for("C1", "100.0", "100.0", "U1"),
                          last_seen_ts="100.0", ceiling_ts="110.0")
    return SimpleNamespace(send_lease=lease), _msg("100.0")


def test_a_channel_request_never_advances_the_lease_across_threads():
    """THE SCOPE FIX. A channel request contains the whole channel, so "the newest role:user ts
    in it" is somebody else's conversation. Absorbing that would tell the guard this turn had
    accounted for a message it is not answering — and the guard would then wave through an answer
    a real newer reply should have suppressed."""
    from message_processor.handlers.text import (advance_channel_lease_to_request,
                                                 advance_lease_to_request)
    from tests.unit.channel_turn_harness import normalized

    items = _channel_items(
        normalized("100.0", "the question", sender_id="U1"),
        normalized("104.0", "a reply under it", sender_id="U2", thread_root_ts="100.0"),
        normalized("102.0", "somebody else's thread", sender_id="U9"),
        normalized("105.0", "…and its reply", sender_id="U9", thread_root_ts="102.0"),
        normalized("111.0", "a reply a LATER turn owns", sender_id="U2", thread_root_ts="100.0"),
    )

    turn, message = _turn_answering_root_100()
    advance_channel_lease_to_request(turn, items, message)
    assert turn.send_lease.last_seen_ts == "104.0", (
        "only the reply in this turn's own thread scope, and nothing above the ceiling")

    dm_turn, _ = _turn_answering_root_100()
    advance_lease_to_request(dm_turn, items, message)
    assert dm_turn.send_lease.last_seen_ts == "105.0", (
        "precondition: the DM helper reads the newest owned ts anywhere in the request, which is "
        "exactly why a channel turn cannot use it")


def test_a_channel_lease_still_absorbs_its_own_senders_top_level_burst():
    """The other half of the scope rule, unchanged: one person's rapid second top-level question
    is the same moment, and a stranger's unrelated one is not."""
    from message_processor.handlers.text import advance_channel_lease_to_request
    from tests.unit.channel_turn_harness import normalized

    items = _channel_items(
        normalized("100.0", "the question", sender_id="U1"),
        normalized("106.0", "same person, straight after", sender_id="U1"),
        normalized("107.0", "a stranger's own question", sender_id="U9"),
    )

    turn, message = _turn_answering_root_100()
    advance_channel_lease_to_request(turn, items, message)
    assert turn.send_lease.last_seen_ts == "106.0"


def test_advancing_the_mark_is_monotonic_across_retries():
    """Retries and fallbacks rebuild the request and recompute this. It must never slide back —
    a regression would re-expose a turn to a message it had already accounted for."""
    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg("100.0"))
    lease.advance_last_seen("100.0")
    lease.advance_last_seen("99.0")
    assert lease.last_seen_ts == "100.0"


# ------------------------------------------------------------- committed / suppressed states

def test_once_committed_everything_that_follows_is_allowed():
    """A split reply checks before its FIRST chunk only. Suppressing the tail would leave half
    an answer in the room, which is worse than a late one."""
    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg("100.0"))
    lease.authorize("final_post")
    lease.commit()
    marks.begin_turn(_msg("101.0"))            # newer traffic arrives mid-delivery
    lease.authorize("final_post")              # the remaining chunks still go
    assert lease.state == COMMITTED


def test_a_suppressed_turn_never_gets_a_second_opinion():
    """Trying a different surface is not new information. A turn refused once is refused for
    everything visible after it."""
    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg("100.0"))
    marks.begin_turn(_msg("101.0"))
    with pytest.raises(StaleSendSuppressed):
        lease.authorize("native_start")
    with pytest.raises(StaleSendSuppressed):
        lease.authorize("legacy_seed")
    assert lease.state == SUPPRESSED


def test_a_failed_send_leaves_the_lease_pending_so_a_retry_is_rechecked():
    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg("100.0"))
    lease.authorize("final_post")              # permitted; Slack then fails, so no commit
    assert lease.state == PENDING
    marks.begin_turn(_msg("101.0"))
    with pytest.raises(StaleSendSuppressed):
        lease.authorize("final_post")          # the retry is checked again, and now refused


# ------------------------------------------------------------------ zero Slack calls, no error

@pytest.mark.asyncio
async def test_a_suppressed_send_makes_no_slack_call_at_all():
    """The point of checking BEFORE the mutation. If this ever posted first and apologized
    after, the feature would be worse than not having it."""
    from slack_client.messaging import SlackMessagingMixin

    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg("100.0"))
    marks.begin_turn(_msg("101.0"))

    host = MagicMock()
    host.app.client.chat_postMessage = AsyncMock()
    host.send_message = SlackMessagingMixin.send_message.__get__(host)
    with pytest.raises(StaleSendSuppressed):
        await host.send_message("C1", "100.0", "the stale answer", lease=lease)
    host.app.client.chat_postMessage.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_suppressed_seed_makes_no_slack_call():
    from slack_client.messaging import SlackMessagingMixin

    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg("100.0"))
    marks.begin_turn(_msg("101.0"))

    host = MagicMock()
    host.app.client.chat_postMessage = AsyncMock()
    host.send_message_get_ts = SlackMessagingMixin.send_message_get_ts.__get__(host)
    with pytest.raises(StaleSendSuppressed):
        await host.send_message_get_ts("C1", "100.0", "seed", lease=lease)
    host.app.client.chat_postMessage.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_suppressed_placeholder_edit_makes_no_slack_call():
    from slack_client.messaging import SlackMessagingMixin

    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg("100.0"))
    marks.begin_turn(_msg("101.0"))

    host = MagicMock()
    host.app.client.chat_update = AsyncMock()
    host.update_message_streaming = SlackMessagingMixin.update_message_streaming.__get__(host)
    with pytest.raises(StaleSendSuppressed):
        await host.update_message_streaming("C1", "T1", "answer text", lease=lease)
    host.app.client.chat_update.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_suppressed_native_start_never_opens_the_stream():
    from slack_client.messaging import NativeStreamSession

    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg("100.0"))
    marks.begin_turn(_msg("101.0"))

    client = MagicMock()
    client.chat_startStream = AsyncMock()
    session = NativeStreamSession(client, "C1", "100.0", team_id="T", user_id="U1")
    with pytest.raises(StaleSendSuppressed):
        await session.start("hello", lease=lease)
    client.chat_startStream.assert_not_awaited()
    assert session.active is False


def test_the_suppression_is_not_an_api_error():
    """It must not be catchable as, or convertible into, a transport failure — a turn that files
    it as one would retry a send we deliberately never made."""
    from slack_sdk.errors import SlackApiError
    assert not issubclass(StaleSendSuppressed, SlackApiError)
    err = StaleSendSuppressed(scope=("top", "C1", "U1"), last_seen_ts="100.0",
                              observed_latest_ts="101.0", surface="final_post")
    assert err.surface == "final_post" and err.observed_latest_ts == "101.0"


# ------------------------------------------------------------------------------- admission

@pytest.mark.asyncio
async def test_only_messages_that_reach_a_turn_raise_the_watermark():
    """ADMISSION IS THE DEFINITION. The lease opens on handle_message's first line, so the set
    of messages that raise the mark is exactly the set that reached a turn. A message dropped
    before dispatch — our own post, a lifecycle subtype, a participation-off channel, the
    app_mention duplicate of a message event — owns no successor, and letting it raise the mark
    would silence an answer that nothing is going to replace."""
    from base_client import Message
    from main import ChatBotV2

    app = ChatBotV2.__new__(ChatBotV2)
    app.processor = MagicMock()
    app.processor.thread_manager = None
    app.processor.process_message = AsyncMock(return_value=None)
    app._run_participation_gate = AsyncMock(return_value=None)
    client = MagicMock()
    client.send_thinking_indicator = AsyncMock(return_value=None)
    client.delete_message = AsyncMock()

    scope = ("top", "C1", "U1")
    assert app.watermarks.latest_for(scope) is None      # nothing dropped has touched it

    await app.handle_message(
        Message(text="q", user_id="U1", channel_id="C1", thread_id="100.0",
                metadata={"ts": "100.0", "sender_id": "U1", "gate_required": True}), client)

    # The turn ran and released; with no lease left open the entry is gone, which is the
    # bounded-by-concurrency rule doing its job.
    assert app.watermarks.tracked_scopes == 0


@pytest.mark.asyncio
async def test_a_turn_holds_its_scope_for_exactly_its_own_lifetime():
    """Barrier-driven: the older turn is INSIDE handle_message when the newer one is admitted,
    which is the only interleaving where the guard has anything to do."""
    from base_client import Message
    from main import ChatBotV2

    app = ChatBotV2.__new__(ChatBotV2)
    app.processor = MagicMock()
    app.processor.thread_manager = None
    admitted = asyncio.Event()
    observed = {}

    async def _slow_turn(message, client, thinking_id=None, turn=None):
        admitted.set()                      # the older turn is now in flight
        await newer_admitted.wait()         # …and waits for the newer message to arrive
        observed["latest"] = turn.send_lease.observed_latest()[0]
        return None

    newer_admitted = asyncio.Event()
    app.processor.process_message = AsyncMock(side_effect=_slow_turn)
    client = MagicMock()
    client.send_thinking_indicator = AsyncMock(return_value=None)
    client.delete_message = AsyncMock()

    def _m(ts):
        return Message(text="q", user_id="U1", channel_id="C1", thread_id=ts,
                       metadata={"ts": ts, "sender_id": "U1"})

    async def _newer():
        await admitted.wait()
        lease = app.watermarks.begin_turn(_m("101.0"))
        newer_admitted.set()
        return lease

    older_task = asyncio.create_task(app.handle_message(_m("100.0"), client))
    newer_lease = await _newer()
    await older_task

    assert observed["latest"] == "101.0", "the older turn saw the newer arrival"
    newer_lease.close()
    assert app.watermarks.tracked_scopes == 0


# ------------------------------------------------------- what the ledger says about a refusal

def test_a_refused_turn_with_nothing_else_visible_is_stale_suppressed():
    """Not a silence — nobody chose it, and there is no silence reason to record. Not
    delivery_failed — Slack was never called, so nothing failed."""
    from base_client import Message
    from main import ChatBotV2
    from message_processor.turn_runtime import TurnRuntime

    message = Message(text="q", user_id="U1", channel_id="C1", thread_id="100.0",
                      metadata={"ts": "100.0"})
    assert ChatBotV2._stale_terminal_kind(None, TurnRuntime(), message) == "stale_suppressed"


def test_a_refused_turn_that_still_reacted_keeps_the_louder_label():
    from base_client import Message
    from main import ChatBotV2
    from message_processor.turn_runtime import TurnRuntime

    message = Message(text="q", user_id="U1", channel_id="C1", thread_id="100.0",
                      metadata={"ts": "100.0"})
    reacted = TurnRuntime(reaction_committed=True)
    assert ChatBotV2._stale_terminal_kind(None, reacted, message) == "reaction_only"

    # Re-baselined: the second half of this test used to assert that a GATE-placed emoji also
    # counted as the louder fact. The gate places nothing in the room now — the metadata stamp it
    # read (`participation_reaction_emoji`) has no producer — so a stale-refused turn with no
    # reaction of its own really did leave the room empty, and labelling it `reaction_only` on the
    # strength of a stamp nobody writes would report an emoji that is not there.
    stale_stamp = Message(text="q", user_id="U1", channel_id="C1", thread_id="100.0",
                          metadata={"ts": "100.0", "participation_reaction_emoji": "eyes"})
    assert ChatBotV2._stale_terminal_kind(None, TurnRuntime(), stale_stamp) == "stale_suppressed"


def test_a_refused_turn_with_a_detached_surface_is_detached():
    """A background job posted its own card. The turn is not silent, whatever happened to the
    words that would have accompanied it."""
    from base_client import Message
    from main import ChatBotV2
    from message_processor.turn_runtime import TurnRuntime

    message = Message(text="q", user_id="U1", channel_id="C1", thread_id="100.0",
                      metadata={"ts": "100.0"})
    detached = TurnRuntime(visible_action_committed=True)
    assert ChatBotV2._stale_terminal_kind(None, detached, message) == "detached"


def test_the_new_kind_is_in_the_vocabulary_and_the_contract_moved():
    from message_processor import participation_telemetry as pt
    assert "stale_suppressed" in pt.KINDS
    # v8 = the single-stream events on top of the binary gate. The version moves whenever the
    # ledger's field set or event cardinality changes, and this assertion exists so a field change
    # cannot ship without someone bumping it deliberately.
    assert pt.CONTRACT_VERSION == 8
    assert pt.GATE_CONTRACT == "binary-v1"


# ==================================== the buffered path is the one that most needs guarding

@pytest.mark.asyncio
async def test_a_buffered_turn_makes_no_slack_call_when_superseded(monkeypatch):
    """THE PROMISE. A silence-capable turn buffers its whole answer so ONE check can span the
    entire model call. That was empty while the final post omitted the lease: the turn held
    every word, then posted them anyway."""
    from config import config as cfg
    from tests.unit.test_reply_surface import (FakeOpenAI, FakeSlack, _message,
                                               _processor, _run, _thread_state)
    from message_processor.turn_runtime import TurnRuntime

    monkeypatch.setattr(cfg, "enable_no_reply_tool", True, raising=False)
    marks = ConversationWatermarks()
    msg, state = _message(silence_capable=True), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=False)
    turn.send_lease = marks.begin_turn(msg)
    assert turn.silence_capable, "this test is about the buffered route"

    slack = FakeSlack(native=True)
    processor = _processor(FakeOpenAI(["the answer nobody ", "is waiting for any more"]))

    # Same THREAD scope as the turn under test (test_reply_surface's message carries no
    # sender_id, so the top scope is correctly omitted for it).
    marks.begin_turn(_msg("11.0", thread="10.0", sender=None))

    with pytest.raises(StaleSendSuppressed):
        await _run(processor, slack, msg, state, turn)

    assert slack.posts == [], f"a buffered turn posted while superseded: {slack.calls}"
    assert slack.edits == [] and slack.streams == []
    assert turn.send_lease.state == SUPPRESSED


@pytest.mark.asyncio
async def test_a_buffered_turn_that_is_still_current_posts_normally(monkeypatch):
    """The other direction, so the guard cannot pass by suppressing everything."""
    from config import config as cfg
    from tests.unit.test_reply_surface import (FakeOpenAI, FakeSlack, _message,
                                               _processor, _run, _thread_state)
    from message_processor.turn_runtime import TurnRuntime

    monkeypatch.setattr(cfg, "enable_no_reply_tool", True, raising=False)
    marks = ConversationWatermarks()
    msg, state = _message(silence_capable=True), _thread_state()
    turn = TurnRuntime.for_message(msg, channel_post_allowed=False)
    turn.send_lease = marks.begin_turn(msg)

    slack = FakeSlack(native=True)
    processor = _processor(FakeOpenAI(["here you go"]))
    resp = await _run(processor, slack, msg, state, turn)

    assert len(slack.posts) == 1
    assert resp.metadata.get("posted") is True
    assert turn.send_lease.state == COMMITTED


def test_the_guard_mode_is_recorded_not_re_derived():
    """An addressed CHANNEL reply buffers too — it has no native path — and re-deriving the
    mode from `silence_capable` reported it as start_only, which is the opposite of what the
    guard did for it."""
    from message_processor.turn_runtime import TurnRuntime
    turn = TurnRuntime()
    assert turn.guard_mode is None            # nothing claimed before the turn binds
    turn.guard_mode = "buffered"
    assert turn.guard_mode == "buffered"


# ============================================ a suppression is never a failure, anywhere

@pytest.mark.asyncio
async def test_tool_dispatch_re_raises_instead_of_reporting_a_broken_tool():
    """post_to_thread declining because the room moved on is control flow. Reported as
    `execution_error` the model reads a broken tool and tries again — causing the very post the
    guard just refused."""
    from tool_registry import ToolContext, ToolRegistry

    async def _refuses(ctx, args):
        raise StaleSendSuppressed(scope=("top", "C1", "U1"), last_seen_ts="100.0",
                                  observed_latest_ts="101.0", surface="post_to_thread")

    registry = ToolRegistry()
    registry.register({"type": "function", "name": "post_to_thread", "parameters": {}}, _refuses)
    with pytest.raises(StaleSendSuppressed):
        await registry.dispatch(ToolContext(channel_id="C1", thread_ts="100.0"),
                                "post_to_thread", "{}")


@pytest.mark.asyncio
async def test_an_ordinary_tool_error_is_still_reported_not_raised():
    """The re-raise must be surgical: a genuine tool bug still becomes a result the model can
    read, exactly as before."""
    from tool_registry import ToolContext, ToolRegistry

    async def _boom(ctx, args):
        raise RuntimeError("the tool is broken")

    registry = ToolRegistry()
    registry.register({"type": "function", "name": "whatever", "parameters": {}}, _boom)
    result = await registry.dispatch(ToolContext(channel_id="C1", thread_ts="100.0"),
                                     "whatever", "{}")
    assert result["ok"] is False and result["error"] == "execution_error"


@pytest.mark.asyncio
async def test_process_message_never_turns_a_suppression_into_an_error_response():
    """It would put "something went wrong" in a channel where nothing did — INSTEAD of the
    answer the guard correctly withheld."""
    from base_client import Message
    from message_processor.base import MessageProcessor

    class _Proc:
        process_message = MessageProcessor.process_message

        def __init__(self):
            from thread_manager import AsyncThreadStateManager
            self.thread_manager = AsyncThreadStateManager(db=None)
            self.db = None

        def log_info(self, *a, **k): pass
        def log_debug(self, *a, **k): pass
        def log_warning(self, *a, **k): pass
        def log_error(self, *a, **k): pass

        async def _dispatch_pending_batch(self, *a, **k): return None
        async def _notify_drain_failure(self, *a, **k): return None

    proc = _Proc()

    async def _raise(*a, **k):
        raise StaleSendSuppressed(scope=("top", "C1", "U1"), last_seen_ts="100.0",
                                  observed_latest_ts="101.0", surface="final_post")

    # Either step-2 entry point — a channel turn creates its state, a DM rebuilds one — and the
    # suppression has to come back out of process_message untouched.
    proc._get_or_rebuild_thread_state = _raise
    proc.get_or_create_channel_thread_state = _raise
    message = Message(text="q", user_id="U1", channel_id="C1", thread_id="100.0",
                      metadata={"ts": "100.0"})
    with pytest.raises(StaleSendSuppressed):
        await proc.process_message(message, client=MagicMock(), thinking_id=None)


# ================================================== the notices are words, not chrome

@pytest.mark.asyncio
@pytest.mark.parametrize("superseded", [False, True])
async def test_a_prior_timeout_notice_is_guarded_like_any_first_surface(superseded):
    """It promises "picking up from here" and the work it promises takes time. A newer message
    during that window must not leave the promise standing alone with no answer behind it."""
    import contextlib
    from base_client import Message
    from message_processor.base import MessageProcessor
    from message_processor.turn_runtime import TurnRuntime

    class _Proc:
        process_message = MessageProcessor.process_message
        # The real notice helper, so this still exercises the send it is about.
        _post_prior_timeout_notice = MessageProcessor._post_prior_timeout_notice

        def __init__(self):
            from thread_manager import AsyncThreadStateManager
            self.thread_manager = AsyncThreadStateManager(db=None)
            self.db = None

        def log_info(self, *a, **k): pass
        def log_debug(self, *a, **k): pass
        def log_warning(self, *a, **k): pass
        def log_error(self, *a, **k): pass

    marks = ConversationWatermarks()
    message = Message(text="q", user_id="U1", channel_id="C1", thread_id="100.0",
                      metadata={"ts": "100.0", "sender_id": "U1"})
    turn = TurnRuntime()
    turn.send_lease = marks.begin_turn(message)
    if superseded:
        marks.begin_turn(_msg("101.0"))

    sent = []

    async def _send(channel_id=None, text=None, thread_id=None, lease=None, surface=None,
                    **kw):
        if lease is not None:
            lease.authorize(surface or "final_post")
        sent.append(text)
        return "99.9"

    client = MagicMock()
    client.send_message = AsyncMock(side_effect=_send)

    async def _state(*a, **k):
        return SimpleNamespace(had_timeout=True, messages=[], channel_id="C1",
                               thread_ts="100.0", root_author=("U1", "human"),
                               config_overrides={}, current_model="gpt-5.6-sol",
                               participants={}, has_trimmed_messages=False)

    proc = _Proc()
    # A channel turn creates its state instead of rebuilding one; both entry points answer with
    # the timed-out thread so the notice is reached whichever way step 2 goes.
    proc._get_or_rebuild_thread_state = _state
    proc.get_or_create_channel_thread_state = _state
    # r2-11: on a channel turn the notice now waits for the window and the admission, so the two
    # steps in front of it have to succeed for this test to reach the thing it is about.
    proc._build_channel_turn_stream = AsyncMock(return_value=None)
    proc._admit_channel_request = AsyncMock()
    proc._process_attachments = AsyncMock(return_value=([], [], []))
    proc._build_user_content = MagicMock(return_value="q")
    with contextlib.suppress(Exception):
        await proc.process_message(message, client=client, thinking_id=None, turn=turn)

    assert sent == ([] if superseded else ["⚠️ Heads up — my last answer in this thread never "
                                           "finished. Picking up from here."])


# ============================================================ commit at the first landed piece

@pytest.mark.asyncio
async def test_a_split_reply_commits_when_its_FIRST_chunk_lands():
    """Committing only after the whole loop left a window where a suppression could be raised
    against a reply that was already half in the room."""
    from slack_client.messaging import SlackMessagingMixin

    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg("100.0"))
    states = []

    host = MagicMock()
    host.MAX_MESSAGE_LENGTH = 40
    host.format_text = lambda t: t
    host._compose_reply_with_footer = MagicMock(return_value=None)
    host._record_receipt = AsyncMock()
    host._split_message = lambda t: ["first chunk", "second chunk", "third chunk"]

    async def _post(**kw):
        states.append(lease.state)          # the state as each chunk goes out
        return {"ts": f"ts.{len(states)}"}

    host.app.client.chat_postMessage = AsyncMock(side_effect=_post)
    host.send_message = SlackMessagingMixin.send_message.__get__(host)

    await host.send_message("C1", "100.0", "x" * 200, lease=lease)

    assert states[0] == PENDING, "the first chunk is the one the guard actually checks"
    assert states[1:] == [COMMITTED] * (len(states) - 1), (
        "every chunk after the first must find the lease already committed")


@pytest.mark.asyncio
async def test_a_single_message_send_commits_so_later_pieces_are_never_refused():
    """A footer, an artifact or a post_to_thread after the answer is up must not be suppressed —
    the room has already seen the turn speak."""
    from slack_client.messaging import SlackMessagingMixin

    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg("100.0"))

    host = MagicMock()
    host.MAX_MESSAGE_LENGTH = 4000
    host.format_text = lambda t: t
    host._compose_reply_with_footer = MagicMock(return_value=None)
    host._record_receipt = AsyncMock()
    host.app.client.chat_postMessage = AsyncMock(return_value={"ts": "99.9"})
    host.send_message = SlackMessagingMixin.send_message.__get__(host)

    await host.send_message("C1", "100.0", "short answer", lease=lease)
    assert lease.state == COMMITTED

    marks.begin_turn(_msg("101.0"))         # newer traffic arrives after the answer is up
    lease.authorize("post_to_thread")       # …and the follow-up piece still goes


# ============================================ round 3: the paths that still swallowed it

@pytest.mark.asyncio
async def test_a_split_retry_reauthorizes_before_its_second_attempt():
    """The first attempt's check says nothing about the second. A failed chunk waits — ample
    time for a newer message — and the retry used to post without asking again."""
    from slack_client.messaging import SlackMessagingMixin
    from slack_sdk.errors import SlackApiError

    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg("100.0"))
    attempts = []

    async def _post(**kw):
        attempts.append(len(attempts) + 1)
        if len(attempts) == 1:
            # The first attempt fails, and the conversation moves on during the backoff.
            marks.begin_turn(_msg("101.0"))
            raise SlackApiError("rate limited", response=MagicMock(headers={}))
        return {"ts": "99.9"}

    host = MagicMock()
    host.MAX_MESSAGE_LENGTH = 40
    host.format_text = lambda t: t
    host._compose_reply_with_footer = MagicMock(return_value=None)
    host._record_receipt = AsyncMock()
    host._split_message = lambda t: ["chunk one", "chunk two"]
    host.app.client.chat_postMessage = AsyncMock(side_effect=_post)
    host.send_message = SlackMessagingMixin.send_message.__get__(host)

    with pytest.raises(StaleSendSuppressed):
        await host.send_message("C1", "100.0", "x" * 200, lease=lease)
    assert attempts == [1], "the retry must be refused, not posted"


@pytest.mark.asyncio
async def test_the_truncation_notice_is_refused_when_nothing_landed():
    """With zero chunks delivered it would be the turn's first and only visible words."""
    from slack_client.messaging import SlackMessagingMixin
    from slack_sdk.errors import SlackApiError

    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg("100.0"))
    posts = []

    async def _post(**kw):
        posts.append(kw.get("text"))
        marks.begin_turn(_msg("101.0"))       # superseded during the very first attempt
        raise SlackApiError("nope", response=MagicMock(headers={}))

    host = MagicMock()
    host.MAX_MESSAGE_LENGTH = 40
    host.format_text = lambda t: t
    host._compose_reply_with_footer = MagicMock(return_value=None)
    host._record_receipt = AsyncMock()
    host._split_message = lambda t: ["chunk one", "chunk two"]
    host.app.client.chat_postMessage = AsyncMock(side_effect=_post)
    host.send_message = SlackMessagingMixin.send_message.__get__(host)

    with pytest.raises(StaleSendSuppressed):
        await host.send_message("C1", "100.0", "x" * 200, lease=lease)
    assert not any("cut off" in (t or "") for t in posts), (
        "the truncation notice must not be the only thing a superseded turn says")


@pytest.mark.asyncio
async def test_a_committed_split_still_posts_its_remaining_chunks_and_notice():
    """The other direction: once a chunk has landed the reader is owed the rest, and owed an
    explanation if the rest fails. A committed lease short-circuits every later check."""
    from slack_client.messaging import SlackMessagingMixin
    from slack_sdk.errors import SlackApiError

    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg("100.0"))
    posts = []

    async def _post(**kw):
        posts.append(kw.get("text"))
        if len(posts) == 1:
            marks.begin_turn(_msg("101.0"))   # newer traffic AFTER the first chunk landed
            return {"ts": "99.9"}
        if len(posts) < 4:
            raise SlackApiError("nope", response=MagicMock(headers={}))
        return {"ts": "99.10"}

    host = MagicMock()
    host.MAX_MESSAGE_LENGTH = 40
    host.format_text = lambda t: t
    host._compose_reply_with_footer = MagicMock(return_value=None)
    host._record_receipt = AsyncMock()
    host._split_message = lambda t: ["chunk one", "chunk two"]
    host.app.client.chat_postMessage = AsyncMock(side_effect=_post)
    host.send_message = SlackMessagingMixin.send_message.__get__(host)

    await host.send_message("C1", "100.0", "x" * 200, lease=lease)
    assert lease.state == COMMITTED
    assert any("cut off" in (t or "") for t in posts), (
        "a half-delivered reply owes the reader the truncation notice")


# --------------------------------------------------------------- the sweep, as a real check

DELIVERY_FILES = ["main.py", "message_processor/base.py", "message_processor/handlers/text.py",
                  "slack_client/messaging.py", "streaming/native_sink.py", "tool_registry.py"]

# Functions that MUTATE Slack. A call to one of these without a lease cannot be refused, so on
# any path where a lease may still be pending it is an unguarded first-visible write waiting to
# happen. Deliberately includes `delete_message`: taking a surface down is chrome, but it is
# still a mutation the audit should account for rather than skip silently.
TRANSPORT = {"send_message", "send_message_get_ts", "update_message_streaming",
             "update_message", "delete_message",
             "chat_postMessage", "chat_update", "chat_startStream", "chat_delete"}

# The escape hatch, deliberately narrow: a trailing `# unleased-ok: <why>` on the call's first
# line. It exists so an intentional unguarded write is a decision somebody wrote down, not an
# omission nobody noticed.
ALLOW_MARKER = "# unleased-ok:"


def _delivery_sources():
    import pathlib as _p
    repo = _p.Path(__file__).resolve().parents[2]
    return {name: (repo / name).read_text() for name in DELIVERY_FILES}


def _propagating_functions(sources):
    """Functions that can raise StaleSendSuppressed, derived from the call graph rather than
    listed by hand — a hand-list is a second copy of the truth and goes stale the first time
    somebody adds a wrapper.

    Seed: a function that obtains a lease FOR ITSELF — reads `send_lease`, or calls
    `.authorize()` — and therefore raises whatever its caller does. A function that merely takes
    an OPTIONAL `lease` parameter is deliberately NOT a seed: called without one it cannot
    raise, so treating it as a propagator flags every unrelated same-named method in the repo
    (`start` alone has three definitions, two of which are streams and one of which is a socket).

    Then iterate: a function that CALLS a propagating function propagates too, to a fixpoint."""
    import ast

    bodies, calls, lease_parameterized = {}, {}, set()
    for name, src in sources.items():
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body = ast.unparse(node)
            bodies[(name, node.name)] = body
            if any(a.arg == "lease" for a in list(node.args.args) + list(node.args.kwonlyargs)):
                lease_parameterized.add((name, node.name))
            called = set()
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    func = inner.func
                    called.add(func.attr if isinstance(func, ast.Attribute)
                               else getattr(func, "id", ""))
            calls[(name, node.name)] = called

    # …and a function that takes a `lease` PARAMETER is excluded even if it calls `.authorize()`:
    # it raises only when a caller hands it one, which the try-body check catches directly via
    # `lease=`. Without this the transport functions themselves become seeds and every chrome
    # caller of `update_message_streaming` is flagged for a call that cannot raise.
    propagating = {key for key, body in bodies.items()
                   if ("send_lease" in body or ".authorize(" in body)
                   and key not in lease_parameterized}
    changed = True
    while changed:
        changed = False
        names = {key[1] for key in propagating}
        for key, called in calls.items():
            if key not in propagating and (called & names):
                propagating.add(key)
                changed = True
    return {key[1] for key in propagating}


# Identifiers that hold a lease, for the positional-argument check.
LEASE_NAMES = {"lease", "send_lease", "_lease"}


def test_no_delivery_boundary_can_swallow_a_suppression():
    """Every `except` that can observe StaleSendSuppressed must re-raise it.

    WHAT THIS DOES AND DOES NOT PROVE. It finds try-blocks that hand a lease to a call (by
    keyword or position), call `.authorize()`, or call a function that obtains a lease FOR
    ITSELF — and it walks the call graph transitively for that last class. It does NOT trace a
    lease through a chain of intermediate wrappers that each take one optionally and forward it:
    that needs data flow, and a half-working version would give a false sense of proof, which is
    worse than a narrow honest one. In practice the delivery paths call the transport directly
    or through one self-leasing handler, so the gap is small — but it is a gap, and this
    paragraph is the reason nobody should read more assurance into a green run than that.

    Three things it establishes rather than assumes:

    * WHICH try bodies can raise — derived from the call graph (functions that pass a lease, and
      transitively their callers), not from a hand-maintained list of substrings.
    * Python's first-matching-handler rule, so a generic `except Exception` shadowed by an
      earlier `except StaleSendSuppressed: raise` is correctly treated as unreachable.
    * That a `StaleSendSuppressed` handler actually RE-RAISES. One that logs and continues is
      worse than a generic handler, because it looks deliberate."""
    import ast

    sources = _delivery_sources()
    propagating = _propagating_functions(sources)
    catches = {"Exception", "BaseException", "StaleSendSuppressed", "bare"}
    gaps = []

    for name, src in sources.items():
        lines = src.splitlines()
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Try):
                continue
            body = "\n".join(lines[node.body[0].lineno - 1: node.body[-1].end_lineno])
            body_calls, hands_lease = set(), False
            for stmt in node.body:
                for c in ast.walk(stmt):
                    if not isinstance(c, ast.Call):
                        continue
                    body_calls.add(c.func.attr if isinstance(c.func, ast.Attribute)
                                   else getattr(c.func, "id", ""))
                    # ARGUMENT-SENSITIVE: a lease handed over positionally raises exactly as one
                    # passed by keyword, and `lease=lease` inside a wrapper is invisible to a
                    # substring test of the enclosing block.
                    if any(kw.arg == "lease" for kw in c.keywords):
                        hands_lease = True
                    # …but only for a callee that could refuse. A local named `lease` is not
                    # necessarily a SEND lease — a turn also holds a reaction lease — and
                    # reading the name alone flagged non-delivery calls as delivery boundaries.
                    callee = (c.func.attr if isinstance(c.func, ast.Attribute)
                              else getattr(c.func, "id", ""))
                    if (callee in TRANSPORT or callee in propagating) and any(
                            isinstance(a, ast.Name) and a.id in LEASE_NAMES for a in c.args):
                        hands_lease = True
            can_raise = hands_lease or ".authorize(" in body or bool(body_calls & propagating)
            if not can_raise:
                continue
            shadowed = False
            for handler in node.handlers:
                label = ast.unparse(handler.type) if handler.type else "bare"
                if label not in catches:
                    continue
                handler_body = ast.unparse(handler)
                reraises = any(isinstance(n, ast.Raise) and n.exc is None
                               for n in ast.walk(handler))
                # A StaleSendSuppressed handler is safe if it re-raises OR if it is the one
                # place that legitimately CONSUMES the suppression — the terminal handler, which
                # records it in the ledger. Verified by looking for that record, not by
                # exempting a line number.
                records = ("stale_send(" in handler_body
                           or "finish_attempt(" in handler_body)
                if label == "StaleSendSuppressed":
                    if not (reraises or records):
                        gaps.append(f"{name}:{handler.lineno} except StaleSendSuppressed "
                                    f"neither re-raises nor records it")
                    shadowed = True      # either way it consumed the exception deliberately
                    continue
                if shadowed:
                    continue
                if reraises:
                    shadowed = True
                    continue
                gaps.append(f"{name}:{handler.lineno} except {label}")
    assert gaps == [], (
        "these handlers can observe StaleSendSuppressed without re-raising: " + "; ".join(gaps))


def test_every_unleased_slack_mutation_is_a_written_down_decision():
    """A transport call with no `lease=` CANNOT be refused. That is right for chrome and for
    writes into a surface the turn already committed — and wrong for anything that could be the
    turn's FIRST visible content, which is how three interruption notices slipped through a
    review that classified them as "edits an existing surface".

    So every unleased mutation in the delivery files has to carry `# unleased-ok: <why>`. The
    marker does not make a call safe; it makes the claim explicit and reviewable, and it turns
    the next omission into a test failure instead of a review round."""
    import ast

    unmarked = []
    for name, src in _delivery_sources().items():
        lines = src.splitlines()
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in TRANSPORT:
                continue
            if any(kw.arg == "lease" for kw in node.keywords):
                continue
            call_src = "\n".join(lines[node.lineno - 1:node.end_lineno])
            if ALLOW_MARKER in call_src:
                continue
            unmarked.append(f"{name}:{node.lineno} {node.func.attr}()")
    assert unmarked == [], (
        "unleased Slack mutations with no written-down justification "
        f"({ALLOW_MARKER} <why>): " + "; ".join(unmarked))


# ============================================ round 6: the terminal notices around the guard

@pytest.mark.asyncio
async def test_a_landed_error_notice_commits_the_lease():
    """A terminal notice that reached the room means the turn HAS spoken. Left pending, a later
    visible piece of the same turn could still be refused after the reader saw something."""
    from slack_client.messaging import SlackMessagingMixin

    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg("100.0"))

    host = MagicMock()
    host.app.client.chat_update = AsyncMock(return_value={"ok": True})
    host.update_message = SlackMessagingMixin.update_message.__get__(host)

    assert await host.update_message("C1", "T1", "⚠️ something went wrong", lease=lease) is True
    assert lease.state == COMMITTED


@pytest.mark.asyncio
async def test_a_superseded_error_notice_is_never_written():
    from slack_client.messaging import SlackMessagingMixin

    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg("100.0"))
    marks.begin_turn(_msg("101.0"))

    host = MagicMock()
    host.app.client.chat_update = AsyncMock()
    host.update_message = SlackMessagingMixin.update_message.__get__(host)

    with pytest.raises(StaleSendSuppressed):
        await host.update_message("C1", "T1", "⚠️ something went wrong", lease=lease)
    host.app.client.chat_update.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("superseded", [False, True])
async def test_handle_error_carries_the_lease_end_to_end(superseded):
    """On a turn with no thinking surface — every silence-capable buffered turn — the error
    notice is the room's FIRST and only word from us, so it is guarded like any first surface."""
    from base_client import BaseClient

    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg("100.0"))
    if superseded:
        marks.begin_turn(_msg("101.0"))

    sent = []

    async def _send(channel_id, thread_id, text, blocks=None, meta_out=None, lease=None,
                    receipts=None, receipt_kind=None):
        if lease is not None:
            lease.authorize("error_notice")
        sent.append(text)
        return "99.9"

    # A stand-in rather than a subclass: BaseClient is abstract, and the contract under test is
    # `handle_error` forwarding the lease — not the fifteen transports it also declares.
    client = SimpleNamespace(name="test", send_message_async=_send,
                             format_error_message=lambda e: e,
                             log_error=lambda *a, **k: None,
                             log_warning=lambda *a, **k: None)
    if superseded:
        with pytest.raises(StaleSendSuppressed):
            await BaseClient.handle_error(client, "C1", "100.0", "boom", lease=lease)
    else:
        await BaseClient.handle_error(client, "C1", "100.0", "boom", lease=lease)
    assert sent == ([] if superseded else ["boom"])


@pytest.mark.asyncio
@pytest.mark.parametrize("superseded", [False, True])
async def test_the_timeout_notice_is_awaited_with_the_lease_not_scheduled(superseded):
    """It used to route through `_update_status`, which SCHEDULES a detached task — and a
    suppression raised inside one can never reach the turn that caused it. This notice is
    terminal, so it is awaited directly with the lease."""
    import contextlib
    from base_client import Message
    from message_processor.base import MessageProcessor
    from message_processor.turn_runtime import TurnRuntime

    class _Proc:
        process_message = MessageProcessor.process_message

        def __init__(self):
            from thread_manager import AsyncThreadStateManager
            self.thread_manager = AsyncThreadStateManager(db=None)
            self.db = None

        def log_info(self, *a, **k): pass
        def log_debug(self, *a, **k): pass
        def log_warning(self, *a, **k): pass
        def log_error(self, *a, **k): pass

        async def _dispatch_pending_batch(self, *a, **k): return None

    marks = ConversationWatermarks()
    message = Message(text="q", user_id="U1", channel_id="C1", thread_id="100.0",
                      metadata={"ts": "100.0", "sender_id": "U1"})
    turn = TurnRuntime()
    turn.send_lease = marks.begin_turn(message)
    if superseded:
        marks.begin_turn(_msg("101.0"))

    edits = []

    async def _update(channel_id, message_id, text, lease=None, surface=None,
                      receipts=None, receipt_kind=None):
        if lease is not None:
            lease.authorize(surface or "error_notice")
        edits.append(text)
        return True

    client = MagicMock()
    client.update_message = AsyncMock(side_effect=_update)

    async def _state(*a, **k):
        raise TimeoutError("the model took too long")

    proc = _Proc()
    # Both step-2 entry points time out, so the notice under test is the TIMEOUT one whichever
    # way the surface routes — not a generic error notice standing in for it.
    proc._get_or_rebuild_thread_state = _state
    proc.get_or_create_channel_thread_state = _state
    with contextlib.suppress(Exception):
        await proc.process_message(message, client=client, thinking_id="T1", turn=turn)

    assert (edits == []) is superseded, (
        "a superseded turn must write no timeout notice; a current one must write exactly one")


# ============================================================ rider E: the suppression's evidence
#
# A refused turn is refused ONCE and then refused again at every later surface it tries, and those
# later refusals used to carry nothing: fourteen copies of "newer message None in None supersedes
# this turn", which names neither the message that superseded it nor the scope. The fact does not
# change after the first refusal, so it is remembered.

def test_a_re_authorization_rethrows_the_original_evidence():
    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg("100.0"))
    marks.begin_turn(_msg("101.0"))          # the room moves on

    with pytest.raises(StaleSendSuppressed) as first:
        lease.authorize("native_start")
    with pytest.raises(StaleSendSuppressed) as second:
        lease.authorize("footer")

    assert first.value.observed_latest_ts == "101.0"
    assert second.value.observed_latest_ts == "101.0", "the same fact, not a blank one"
    assert second.value.scope == first.value.scope
    assert second.value.surface == "footer"          # …only the surface differs
    assert "None in None" not in str(second.value)


def test_the_evidence_survives_the_superseding_turn_closing_its_lease():
    """The watermark entry is gone by the time the later surfaces try — which is exactly when the
    old code re-derived it and got nothing."""
    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg("100.0"))
    newer = marks.begin_turn(_msg("101.0"))

    with pytest.raises(StaleSendSuppressed):
        lease.authorize("native_start")
    newer.close()

    with pytest.raises(StaleSendSuppressed) as again:
        lease.authorize("artifact")
    assert again.value.observed_latest_ts == "101.0"


# ---------------------------------------------------- rider E: no streaming layer swallows it

def _stream_host():
    from unittest.mock import AsyncMock as _AsyncMock

    host = MagicMock()
    host.logged = {"warning": [], "error": []}
    host.log_info = host.log_debug = lambda *a, **k: None
    host.log_warning = lambda msg, *a, **k: host.logged["warning"].append(str(msg))
    host.log_error = lambda msg, *a, **k: host.logged["error"].append(str(msg))
    host._safe_api_call = _AsyncMock(return_value=SimpleNamespace())
    return host


def _events(kind):
    """A finite stream (pitfall #6) shaped for the layer under test."""
    if kind == "event":
        class _Hostile:
            @property
            def type(self):
                raise StaleSendSuppressed(scope=("thread", "C1", "10.0"), last_seen_ts="10.0",
                                          observed_latest_ts="11.0", surface="probe")

        return [_Hostile()]
    if kind == "delta":
        return [SimpleNamespace(type="response.output_text.delta", delta="hello")]
    return [SimpleNamespace(type="response.output_text.delta", delta="hello"),
            SimpleNamespace(type="response.completed",
                            response=SimpleNamespace(usage=SimpleNamespace(
                                input_tokens=1, output_tokens=1)))]


def _callback(kind):
    """Raises the suppression at the layer this case is about."""
    def _cb(chunk):
        if kind == "delta" and chunk is not None:
            raise StaleSendSuppressed(scope=("thread", "C1", "10.0"), last_seen_ts="10.0",
                                      observed_latest_ts="11.0", surface="stream_delta")
        if kind == "completion" and chunk is None:
            raise StaleSendSuppressed(scope=("thread", "C1", "10.0"), last_seen_ts="10.0",
                                      observed_latest_ts="11.0", surface="stream_flush")
    return _cb


@pytest.mark.parametrize("with_tools", [False, True])
@pytest.mark.parametrize("layer", ["delta", "completion", "event"])
@pytest.mark.asyncio
async def test_no_streaming_layer_swallows_a_suppression(layer, with_tools):
    """SIX catch sites — a delta callback, a completion flush and a per-event catch, in each of the
    two streaming implementations — every one of which used to log the suppression as an ordinary
    callback/event error and carry on. The turn then streamed to a room that had moved on, and the
    fourteen uninformative WARNINGs the P2 battery saw were these layers each catching it in turn.

    Everything ELSE those handlers catch keeps today's swallow; that is the next test."""
    from openai_client.api import responses as R

    host = _stream_host()

    async def _iter(response, op):
        for event in _events(layer):
            yield event

    host._safe_stream_iteration = _iter

    call = (R.create_streaming_response_with_tools if with_tools
            else R.create_streaming_response)
    kwargs = {"tools": [{"type": "web_search"}]} if with_tools else {}

    with pytest.raises(StaleSendSuppressed) as raised:
        await call(host, messages=[{"role": "user", "content": "hi"}],
                   stream_callback=_callback(layer), model="gpt-5.6-sol", **kwargs)

    # The evidence reaches the top intact, and the outer handler files it as a suppression rather
    # than as an API error with a stack trace under a turn where nothing went wrong.
    assert raised.value.observed_latest_ts == "11.0"
    assert any("ended without posting" in line for line in host.logged["warning"])
    assert host.logged["error"] == []


@pytest.mark.parametrize("with_tools", [False, True])
@pytest.mark.asyncio
async def test_an_ordinary_callback_error_is_still_swallowed(with_tools):
    """The mutation of the test above. A broken buffer must not end the stream — only a
    suppression does."""
    from openai_client.api import responses as R

    host = _stream_host()

    async def _iter(response, op):
        for event in _events("completion"):
            yield event

    host._safe_stream_iteration = _iter

    def _cb(chunk):
        raise RuntimeError("the buffer fell over")

    call = (R.create_streaming_response_with_tools if with_tools
            else R.create_streaming_response)
    kwargs = {"tools": [{"type": "web_search"}]} if with_tools else {}

    text = await call(host, messages=[{"role": "user", "content": "hi"}],
                      stream_callback=_cb, model="gpt-5.6-sol", **kwargs)

    assert text == "hello", "the stream completed; the callback's failure was its own"
    assert any("callback error" in line for line in host.logged["warning"])
