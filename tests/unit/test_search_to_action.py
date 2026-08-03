"""§2g/§5 — a root a TOOL RESULT proved this turn becomes a legal post_to_thread target.

Three questions, and the file is organized around them:

1. **WHAT DOES A SEARCH RESULT PROVE?** A result carries a REPLY's ts, and a reply's ts is not a
   thread root. **CHANNEL_SEARCH_REBUILD §S1 changed where the root comes from and nothing else
   about this file's subject.** A channel turn no longer queries `assistant.search.context` — it
   runs the bot-token in-channel scan — so the root now comes DIRECTLY off the normalized
   message's `thread_ts`, the one source S6 names, and the permalink derivation that used to do
   the work is neither consulted nor reachable on this surface. The claims below are the same
   claims, restated in the terms of the backend that actually serves a channel: a reply proves
   its root, a top-level message proves nothing, a root in another channel cannot arrive at all,
   and a timestamp the normalizer cannot read takes its whole message out of the result.
2. **WHAT MAY ENROLL?** Only the three named tools, only on `ok`, only in THIS channel, only
   with same-workspace provenance in hand, and only as the LAST step before the result is
   returned — so a result the model never saw has widened nothing.
3. **WHAT DOES ENROLLMENT BUY?** A post into the found thread, from the NEXT round onwards, with
   the same tool-name provenance record every other reply already gets (§5.4a).

RECORDED LIVE PAYLOAD (`_RECORDED_HIT`): captured 2026-07-31 from `assistant.search.context` in
workspace `T0320A41P`, plus a seeded two-message thread in `C0BKX77NU66` whose permalinks were
read back with `chat.getPermalink`. What the live API actually returns is written up beside the
fixture, because it decides which of §2g's two derivation sources ever fires.
"""
import asyncio
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import config
from message_processor.turn_runtime import DEST_KIND_POST_TO_THREAD, TurnRuntime
from slack_client.history_tool import SlackHistoryToolMixin
from slack_client.messaging import SlackMessagingMixin
from slack_client.search_tool import SlackSearchToolMixin
from tests.unit import channel_turn_harness as harness
from tool_registry import ToolContext, serialize_tool_result

CHANNEL = "C1"
OTHER_CHANNEL = "C2"
TEAM = "T1"
OTHER_TEAM = "T9"

# The thread this channel's stream never rendered — far below the periphery floor, and the whole
# point of the wave: unreachable until a tool result proves it exists.
OLD_ROOT = "1000.000100"
OLD_REPLY = "1000.000200"


# --------------------------------------------------------------- the recorded live payload
#
# WHAT SLACK ACTUALLY RETURNS (probe, 2026-07-31, workspace T0320A41P). An
# `assistant.search.context` hit carries exactly these keys:
#
#     author_name, author_email, author_user_id, team_id, channel_id, channel_name,
#     message_ts, content, is_author_bot, permalink   (+ reply_count on a thread ROOT)
#
# THERE IS NO `thread_ts` KEY. §2g's first derivation source never fires on a live payload; it
# is defensive, and the PERMALINK is the source that does the work. The permalink shape §2g
# assumes is exactly what Slack mints — verified against a seeded thread in C0BKX77NU66:
#
#     reply:            …/archives/C0BKX77NU66/p1785553079009269?thread_ts=1785553078.882309&cid=C0BKX77NU66
#     root WITH replies:…/archives/C0BKX77NU66/p1785553078882309?thread_ts=1785553078.882309&cid=C0BKX77NU66
#     top-level, no replies: …/archives/C092X66TU3G/p1777986290119489      (no query string at all)
#
# So a hit on a message with no thread carries no query parameters and enrolls nothing — the
# normal case — while both a reply and a replied-to root carry `thread_ts` AND a matching `cid`.
# Author identity is replaced with a synthetic user below: the shape is the fixture, the people
# are not.
_RECORDED_HIT = {
    "author_name": "Test User",
    "author_email": "test.user@example.com",
    "author_user_id": "U01",
    "team_id": TEAM,
    "channel_id": CHANNEL,
    "channel_name": "chatgpt-bot-test",
    "message_ts": OLD_REPLY,
    "content": "the rollout is blocked on the cert renewal",
    "is_author_bot": False,
    "permalink": (f"https://example.slack.com/archives/{CHANNEL}/p1000000200"
                  f"?thread_ts={OLD_ROOT}&cid={CHANNEL}"),
}


def hit(**overrides):
    """One search hit in the recorded shape, with fields overridden or removed (`None` on a key
    the recorded payload has removes it, so a test can say "this one carries no team id")."""
    m = dict(_RECORDED_HIT)
    for key, value in overrides.items():
        if value is None and key in m:
            del m[key]
        else:
            m[key] = value
    return m


def reply_hit(**overrides):
    """The recorded payload AS A THREAD REPLY — `thread_ts` present.

    The live assistant payload carries no `thread_ts` key at all (see the probe above), which is
    why the retired backend had to mine the permalink for one. The scan has exactly one source,
    so a channel case that needs a root must SAY it is a reply rather than imply it with a link.
    """
    return hit(**{"thread_ts": OLD_ROOT, **overrides})


def permalink(channel=CHANNEL, root=OLD_ROOT, cid=CHANNEL, ts="1000000200"):
    url = f"https://example.slack.com/archives/{channel}/p{ts}"
    params = []
    if root is not None:
        params.append(f"thread_ts={root}")
    if cid is not None:
        params.append(f"cid={cid}")
    return f"{url}?{'&'.join(params)}" if params else url


# ------------------------------------------------------------------------------- the hosts

class _Host(SlackSearchToolMixin, SlackHistoryToolMixin):
    """The two mixins on ONE object, as production composes them on the SlackBot — so search
    reaches the CANONICAL `_delivery_allowed` and `_bot_team_id` rather than a lookalike."""

    def __init__(self, team=TEAM):
        self.self_team_id = team
        self.app = MagicMock()
        for name in ("log_info", "log_debug", "log_warning", "log_error"):
            setattr(self, name, MagicMock())


def as_scanned(m):
    """One recorded HIT → the Slack message the in-channel scan actually reads.

    The hit shape stays this file's vocabulary — the cases are written in it, and the DM surface
    still receives exactly these bytes — but a channel turn's results now come off
    `conversations.history`, so the fixture is translated rather than replaced. `thread_ts` is
    carried across UNCHANGED and is the only root source the scan has; a hit's permalink rides
    along untouched precisely so the tests can prove it is never consulted.
    """
    scanned = {"ts": m.get("message_ts"), "text": m.get("content", ""),
               "user": m.get("author_user_id")}
    if "team_id" in m:
        scanned["team"] = m["team_id"]
    if "context_team_id" in m:
        # A SECOND, DIFFERENT stamp is how a scanned message contradicts itself about its
        # workspace — the shape the delivery rule refuses to classify.
        scanned["context_team_id"] = m["context_team_id"]
    if "thread_ts" in m:
        scanned["thread_ts"] = m["thread_ts"]
    if "permalink" in m:
        scanned["permalink"] = m["permalink"]
    return scanned


def _search_host(messages, *, team=TEAM, scanned=None):
    """The REAL `execute_search_tool` on a CHANNEL surface — so, the REAL in-channel scan.

    The fakes are the Slack transport and the two membership lookups the delivery rule makes.
    `_authorize_channel_read` is stubbed for the same reason `_history_host` stubs it: this file
    is about what a result AUTHORIZES, and the channel-read gate has its own file
    (test_channel_scope_guard.py) where it is driven for real. `_delivery_allowed` is NOT stubbed
    — the canonical rule stays in the path, which is half of what §2g rests on.

    `api_call` stays wired so the DM cases in this file still exercise the assistant backend.
    """
    host = _Host(team)
    host._destination_forces_current_only = AsyncMock(return_value=False)
    host._source_is_public = AsyncMock(return_value=True)
    host._authorize_channel_read = AsyncMock(return_value=("ALLOW", "current_channel"))
    host.resolve_usernames = AsyncMock(return_value={})
    host.app.client.api_call = AsyncMock(
        return_value={"results": {"messages": list(messages)}})
    page = [as_scanned(m) for m in messages] if scanned is None else list(scanned)
    host.app.client.conversations_history = AsyncMock(
        return_value={"ok": True, "messages": page})
    # Any root the page names is fetched; none of these fixtures put anything under one.
    host.app.client.conversations_replies = AsyncMock(return_value={"ok": True, "messages": []})
    host.app.client.chat_getPermalink = AsyncMock(
        return_value={"ok": True, "permalink": _RECORDED_HIT["permalink"]})
    return host


def _history_host(result):
    """The REAL dispatcher + enrollment seam over a canned `fetch_history_tool` result."""
    host = _Host()
    host._authorize_channel_read = AsyncMock(return_value=("ALLOW", "current_channel"))
    host.fetch_history_tool = AsyncMock(return_value=result)
    host.get_message_permalink_tool = AsyncMock(return_value=result)
    host.fetch_channel_info_tool = AsyncMock(return_value=result)
    host.fetch_pinned_messages_tool = AsyncMock(return_value=result)
    return host


def _post_host():
    """The REAL `execute_post_to_thread` with `send_message` mocked — the authorization decision
    is what these tests read, and it is made before anything is sent."""
    host = MagicMock()
    host.execute_post_to_thread = SlackMessagingMixin.execute_post_to_thread.__get__(host)
    host.send_message = AsyncMock(return_value="900.0")
    return host


def _turn(channel=CHANNEL):
    """A channel turn with a REAL serialized stream that renders ONE thread — and never the old
    root, which sits far below the floor."""
    turn = TurnRuntime()
    harness.pin_channel_turn(turn, channel_id=channel, messages=[
        harness.normalized("9000.0", "a recent question", channel_id=channel),
        harness.normalized("9001.0", "a recent answer", channel_id=channel,
                           thread_root_ts="9000.0"),
    ], trigger_ts="9000.0", origin_thread_ts="9000.0")
    return turn


def _ctx(turn, *, channel=CHANNEL, is_dm=False, thread="9000.0"):
    return ToolContext(channel_id=channel, thread_ts=thread, trigger_ts=thread,
                       action_token="xoxa-test", is_dm=is_dm, client=MagicMock(),
                       db=MagicMock(), turn=turn)


def _registry(name, executor, *, timeout=30.0):
    from tool_registry import ToolRegistry

    registry = ToolRegistry()
    registry.register({"type": "function", "name": name, "parameters": {}}, executor,
                      timeout=timeout)
    return registry


async def _search(host, ctx, args=None, *, timeout=30.0):
    """Run the search THROUGH THE REGISTRY, which is what commits authority.

    §2g is a two-phase act: the executor stages a claim, and `ToolRegistry` grants it only at
    the moment it selects that result for the model. Calling the executor directly would
    exercise half the mechanism and prove nothing about what actually widens the allowlist —
    which is the whole question these tests are asking."""
    out = await _registry("search_slack", host.execute_search_tool,
                          timeout=timeout).dispatch_all(
        ctx, [{"name": "search_slack", "call_id": "s",
               "arguments": args if args is not None else {"query": "cert renewal"}}])
    return out[0]


async def _history(host, ctx, name, args):
    """The history dispatcher, through the registry, for the same reason."""
    async def run(call_ctx, call_args):
        return await host.dispatch_history_tool_call(name, call_args, ctx=call_ctx)

    out = await _registry(name, run).dispatch_all(
        ctx, [{"name": name, "call_id": "h", "arguments": args}])
    return out[0]


def _trusted(turn):
    """What the NEXT round's ToolContext would be stamped with — the production seam, not a
    hand-built set."""
    from message_processor.handlers.text import _trusted_thread_roots
    return _trusted_thread_roots(turn)


# =============================================== 1. WHAT DOES A SEARCH RESULT PROVE?

@pytest.mark.asyncio
async def test_a_scanned_reply_enrolls_the_root_it_names():
    """T78, on the backend that now serves a channel: a scanned reply's own `thread_ts` becomes
    the entry's `thread_ts` and the turn's discovered root.

    The provenance is ASSERTED PRESENT rather than assumed — a fixture that quietly lost its
    team stamp would enroll nothing under §2g, and the test would be describing a mechanism it
    had disabled."""
    assert _RECORDED_HIT["team_id"] == TEAM, "the recorded payload must carry same-team provenance"
    turn = _turn()
    host = _search_host([hit(thread_ts=OLD_ROOT)])

    out = await _search(host, _ctx(turn))

    assert out["ok"] is True
    assert out["results"][0]["thread_ts"] == OLD_ROOT
    assert turn.discovered_thread_roots == frozenset({OLD_ROOT})


@pytest.mark.asyncio
async def test_a_top_level_result_enrolls_nothing():
    """T79. No `thread_ts` on the message — exactly what Slack sends for a top-level message
    nobody replied to — so there is no thread to make a target."""
    turn = _turn()
    host = _search_host([hit(permalink=permalink(root=None, cid=None))])

    out = await _search(host, _ctx(turn))

    assert out["ok"] is True
    assert out["results"][0]["thread_ts"] is None
    assert turn.discovered_thread_roots == frozenset()


@pytest.mark.asyncio
@pytest.mark.parametrize("link", [
    permalink(root=OLD_ROOT),                               # a link that names a root
    permalink(channel=OTHER_CHANNEL, cid=OTHER_CHANNEL),    # …and one pointing somewhere else
], ids=["names-a-root", "names-another-channel"])
async def test_a_permalink_on_the_payload_is_never_a_root_source(link):
    """T89/T90 restated: ONE SOURCE, and the permalink is not it.

    The retired assistant backend had to derive a root from the permalink's `thread_ts`
    parameter, and had to prove the link agreed with the hit about which channel it was before
    trusting it. The scan reads `thread_ts` off the message and nothing else, so a link that
    names a root — or names another conversation entirely — authorizes nothing. Pinned rather
    than assumed: a future backend that started parsing links again would fail here.
    """
    turn = _turn()
    host = _search_host([hit(permalink=link)])

    out = await _search(host, _ctx(turn))

    assert out["results"][0]["thread_ts"] is None
    assert turn.discovered_thread_roots == frozenset()


@pytest.mark.asyncio
@pytest.mark.parametrize("root", ["", "abc", {"ts": OLD_ROOT}],
                         ids=["empty", "not-a-number", "dict"])
async def test_a_malformed_root_timestamp_enrolls_nothing(root):
    """T92, and the scan is STRICTER than the path it replaces.

    A `thread_ts` the normalizer cannot place in time does not merely fail to enroll: the whole
    MESSAGE is unreadable (`secondary_ts` raises, so `_normalize_scanned` declines it), and a
    message the scan cannot place is one it must not report. Nothing is returned, nothing is
    enrolled, and nothing raises.
    """
    turn = _turn()
    host = _search_host([hit(thread_ts=root, permalink=permalink(root=None, cid=None))])

    out = await _search(host, _ctx(turn))

    assert out["ok"] is True
    assert out["count"] == 0
    assert turn.discovered_thread_roots == frozenset()


@pytest.mark.asyncio
async def test_an_absent_root_is_absent_and_a_numeric_one_is_refused_at_the_boundary():
    """The two `thread_ts` shapes the malformed case leaves out.

    `None` is ABSENT — Slack's own way of saying "not in a thread" — and enrolls nothing.

    A JSON NUMBER IS REFUSED AT THE SCAN BOUNDARY (codex review #4). `secondary_ts` stringifies
    whatever it is handed, so a number would otherwise arrive looking like a perfectly good root
    and flow straight into `stage_discovered_root`, widening where this turn may post — and a
    float cannot even be relied on to preserve a timestamp's identity. The shared normalizer's
    wider behaviour is deliberately left alone; this tool refuses the malformed TYPE itself,
    drops the message, and reports the gap in its coverage rather than certifying a channel it
    did not entirely read.
    """
    turn = _turn()
    absent = _search_host([hit(thread_ts=None, permalink=permalink(root=None, cid=None))])
    out = await _search(absent, _ctx(turn))
    assert out["results"][0]["thread_ts"] is None
    assert turn.discovered_thread_roots == frozenset()

    numeric_turn = _turn()
    numeric = _search_host([hit(thread_ts=1000.0001, permalink=permalink(root=None, cid=None))])
    out = await _search(numeric, _ctx(numeric_turn))
    assert out["count"] == 0, "a non-string root takes its whole message out of the result"
    assert out["coverage"]["complete"] is False
    assert out["coverage"]["stopped_reason"] == "history_data_invalid"
    assert numeric_turn.discovered_thread_roots == frozenset()


# =============================================== 2. WHAT MAY ENROLL?

@pytest.mark.asyncio
async def test_an_unstamped_scanned_reply_enrolls_its_root():
    """T91, FIRST HALF — the S6 addendum, tested against the ruling rather than against the
    retired backend's answer (codex review #7).

    A scanned message carrying no team stamp at all is treated as OURS: it came back from the
    authorized current channel on our own bot token, and our own posts carry no `team`, so the
    opposite rule would make everything the bot ever said unenrollable. The fixture is a REPLY —
    a case with no direct `thread_ts` could not enroll whatever its provenance said, and would
    pass for the wrong reason.
    """
    turn = _turn()
    unstamped = reply_hit(team_id=None)
    assert "team_id" not in unstamped, "the fixture must carry no workspace stamp at all"
    host = _search_host([unstamped])

    out = await _search(host, _ctx(turn))

    assert out["results"][0]["thread_ts"] == OLD_ROOT
    assert turn.discovered_thread_roots == frozenset({OLD_ROOT})


@pytest.mark.asyncio
@pytest.mark.parametrize("provenance", [
    {"team_id": TEAM, "context_team_id": OTHER_TEAM},    # contradictory: two distinct stamps
    {"team_id": OTHER_TEAM},                             # another workspace
], ids=["contradictory", "foreign"])
async def test_contradictory_or_foreign_team_ids_enroll_nothing(provenance):
    """T91, SECOND HALF. Driven through the REAL executor, so it proves the check happens while
    the parsed provenance is still in hand — the shaped entry has thrown it away by then. Both
    fixtures are REPLIES with a direct root, so the refusal is about the workspace and nothing
    else; a message that stamps two workspaces cannot be classified, and one that stamps another
    workspace is not ours to make a target of."""
    turn = _turn()
    host = _search_host([reply_hit(**provenance)])

    out = await _search(host, _ctx(turn))

    assert out["count"] == 0, "not deliverable, so not returned either"
    assert turn.discovered_thread_roots == frozenset()


@pytest.mark.asyncio
async def test_a_cross_channel_root_never_becomes_a_target():
    """T81. `post_to_thread` posts into `ctx.channel_id`, so a root in ANOTHER channel is
    awareness and never a target — enrolling one would authorize a post into a channel the turn
    is not in.

    THE RULE NOW HOLDS TWICE, and both halves are asserted. The scan reads the authorized
    current channel only, so a foreign root cannot ARRIVE: a result naming another conversation
    still comes back as this one, and `stage_discovered_root`'s channel is `ctx.channel_id` by
    construction. And the post side — the half that actually protects the other channel —
    refuses a root nothing proved, exactly as before.
    """
    turn = _turn()
    host = _search_host([hit(channel_id=OTHER_CHANNEL,
                             permalink=permalink(channel=OTHER_CHANNEL, cid=OTHER_CHANNEL))])

    out = await _search(host, _ctx(turn))

    assert [r["channel"] for r in out["results"]] == [CHANNEL], (
        "the payload named another conversation; the result is the trusted current one")
    assert turn.discovered_thread_roots == frozenset(), "no root was proved for this channel"
    poster = _post_host()
    refusal = await poster.execute_post_to_thread(
        SimpleNamespace(channel_id=CHANNEL, thread_ts="9000.0", trigger_ts="9000.0",
                        trusted_thread_roots=_trusted(turn), turn=turn),
        {"thread_ts": OLD_ROOT, "text": "over there"})
    assert refusal["error"] == "unknown_thread"
    poster.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_context_with_no_turn_enrolls_nothing():
    """T93. A DM ctx and a channel ctx with no turn each enroll nothing and are not errors —
    there is no turn to widen. A hand-built context that cannot say WHEN it is asking is a
    different case and is refused outright (codex review #6): the trigger fence is what keeps a
    scan out of the present, and a scan with no fence is not one this backend will run."""
    for ctx in (_ctx(None, is_dm=True), _ctx(None)):
        host = _search_host([_RECORDED_HIT])
        out = await _search(host, ctx)
        assert out["ok"] is True and out["count"] == 1

    triggerless = SimpleNamespace(channel_id=CHANNEL, action_token="xoxa-test", is_dm=False)
    refused = await _search(_search_host([_RECORDED_HIT]), triggerless)
    assert refused["ok"] is False and refused["error"] == "search_unavailable"


@pytest.mark.asyncio
async def test_a_result_that_never_reaches_the_model_enrolls_nothing():
    """T94. Staging is the LAST step before the return, so a result the model never saw can never
    have widened authorization. Three ways to not reach the model: a shaping error, an aborted
    name resolution, and a validation refusal.

    Each is driven through the REGISTRY, which is what the model's answer actually comes from —
    and note it does not re-raise: a tool bug becomes `{"ok": False, "error": "execution_error"}`
    so a broken tool degrades the answer rather than killing the response. What matters here is
    that the answer the model receives, whatever shape it takes, granted nothing."""
    turn = _turn()

    # A shaping error — the entry build itself raises, so no claim is ever staged. (The seam
    # moved with the backend: the scan shapes its entries around the team stamp rather than
    # around a permalink-derived root.)
    host = _search_host([reply_hit()])
    host._scan_team_ids = MagicMock(side_effect=RuntimeError("shaping failed"))
    out = await _search(host, _ctx(turn))
    assert out["ok"] is False and out["error"] == "execution_error"
    assert turn.discovered_thread_roots == frozenset()

    # An aborted name resolution — the resolver does not merely fail, it takes the call down.
    host = _search_host([reply_hit()])
    host.resolve_usernames = AsyncMock(side_effect=BaseException("resolution aborted"))
    with pytest.raises(BaseException, match="resolution aborted"):
        await _search(host, _ctx(turn))
    assert turn.discovered_thread_roots == frozenset()

    # A validation refusal — the executor returns before it ever reaches the API.
    host = _search_host([reply_hit()])
    out = await _search(host, _ctx(turn), {"query": "   "})
    assert out["ok"] is False
    assert turn.discovered_thread_roots == frozenset()


@pytest.mark.asyncio
async def test_a_call_that_already_timed_out_enrolls_nothing():
    """T94, the LIFECYCLE half, driven through the REAL registry.

    A tool execution is SHIELDED, so it keeps running after the waiter gives up — deliberately,
    because cancelling a tool mid-effect is how a posted message loses the receipt that makes it
    ours. The consequence this closes: a slow search finishes and tries to enroll long after the
    model was handed `{error: timeout}` for that same call. The model was shown a timeout, and a
    timeout proves no thread.

    THE ORDERING IS THE TEST. The executor stages its claim and then keeps running; the registry
    picks the timeout. So the claim is made BEFORE the answer is chosen, which is exactly the
    window a clock-reading gate could not close — it would sample the deadline while the result
    was still undecided, and a deadline that expired in the remaining microseconds would leave a
    root enrolled for a call the model was told had failed. Here there is no window: the commit
    is on the branch that selects the result, and the timeout branch is a different one.
    """
    turn = _turn()
    host = _search_host([reply_hit()])
    staged: List[Any] = []
    real = host.execute_search_tool

    async def crawling(ctx, args):
        result = await real(ctx, args)          # stages its claim here…
        staged.extend(getattr(getattr(ctx, "tool_flight", None), "staged_roots", None) or ())
        await asyncio.sleep(0.15)               # …and is still running when the waiter gives up
        return result

    dispatched = await _registry("search_slack", crawling, timeout=0.02).dispatch_all(
        _ctx(turn), [{"name": "search_slack", "call_id": "s",
                      "arguments": {"query": "cert renewal"}}])

    assert dispatched[0]["error"] == "timeout", "the model was told the call did not finish"
    assert [r.root_ts for r in staged] == [OLD_ROOT], (
        "the claim WAS staged — this test is about what happens to a staged claim, and a run "
        "that staged nothing would pass for the wrong reason")
    await asyncio.sleep(0.25)      # let the shielded execution finish, as production lets it
    assert turn.discovered_thread_roots == frozenset()


@pytest.mark.asyncio
async def test_a_duplicate_caller_commits_the_claims_the_original_waiter_abandoned():
    """The claims belong to the EXECUTION, not to whoever happened to ask for it first.

    One flight can have several waiters: a duplicate dispatch of the same call id joins the
    flight rather than doing the work twice, and only the waiter that CREATED it has a per-call
    context. So when the original waiter goes away — cancelled here, and a timeout is the other
    way — and a duplicate is the one that ultimately RECEIVES the completed result, that
    duplicate is the caller whose answer the model reads. It has to be able to commit what the
    execution proved, or the model is shown a result that granted nothing.
    """
    turn = _turn()
    host = _search_host([reply_hit()])
    real = host.execute_search_tool

    async def unhurried(ctx, args):
        await asyncio.sleep(0.05)
        return await real(ctx, args)

    registry = _registry("search_slack", unhurried)
    args = {"query": "cert renewal"}

    # The original waiter opens the flight, then is cancelled while the work runs on.
    first = asyncio.create_task(registry.dispatch(_ctx(turn), "search_slack", args, "dup"))
    await asyncio.sleep(0.01)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert turn.discovered_thread_roots == frozenset(), "nothing committed on the way out"

    # A duplicate joins the SAME flight and receives the completed result.
    second = await registry.dispatch(_ctx(turn), "search_slack", args, "dup")

    assert second["ok"] is True and second["results"][0]["thread_ts"] == OLD_ROOT
    assert turn.discovered_thread_roots == frozenset({OLD_ROOT})


@pytest.mark.asyncio
async def test_a_flight_every_waiter_times_out_on_commits_nothing():
    """The inverse, and the reason the rule is about the SELECTED result rather than the flight
    reaching one: two waiters on one flight, both give up, the execution finishes anyway. Nobody
    was shown it, so nobody may act on it."""
    turn = _turn()
    host = _search_host([reply_hit()])
    real = host.execute_search_tool

    async def crawling(ctx, args):
        result = await real(ctx, args)
        await asyncio.sleep(0.2)
        return result

    registry = _registry("search_slack", crawling, timeout=0.02)
    args = {"query": "cert renewal"}
    both = await asyncio.gather(
        registry.dispatch(_ctx(turn), "search_slack", args, "slow"),
        registry.dispatch(_ctx(turn), "search_slack", args, "slow"))

    assert [r["error"] for r in both] == ["timeout", "timeout"]
    await asyncio.sleep(0.3)
    assert turn.discovered_thread_roots == frozenset()


@pytest.mark.asyncio
async def test_a_root_cut_off_by_truncation_enrolls_nothing(monkeypatch):
    """T94, the TRUNCATION half. The function output the model reads is clipped to
    TOOL_RESULT_MAX_CHARS, so the tail of a long result never reaches it. Enrollment runs against
    the SERIALIZED, TRUNCATED bytes: an early hit still enrolls, and a hit past the cut — which
    the model cannot see and cannot have chosen — does not.

    Driven through the REAL `execute_search_tool` and the REAL `serialize_tool_result`, with only
    the cap lowered, so it is the production clip that decides.

    THE CAP IS MEASURED, NOT GUESSED. A hand-picked number was a fixture that only worked while
    the payload's bytes stayed exactly as long as they were on the day it was written — and the
    payload has since grown a coverage block. The first pass reads where the second root's own
    field STARTS and cuts there, so the case keeps testing the clip rather than an old byte
    count. Ranking is newest-first, so the surviving entry is deliberately the NEWER one."""
    late_root = "1000.000900"
    kept = reply_hit(message_ts="1000.000950")
    clipped_away = hit(message_ts="1000.000200", thread_ts=late_root,
                       content="cert renewal " + "x" * 400)

    probe_turn = _turn()
    probe = await _search(_search_host([kept, clipped_away]), _ctx(probe_turn))
    assert [e["thread_ts"] for e in probe["results"]] == [OLD_ROOT, late_root], (
        "both roots are in the result — the difference is only what the model gets to read")
    cut = serialize_tool_result(probe).index(f'"thread_ts": "{late_root}"')

    turn = _turn()
    monkeypatch.setattr(config, "tool_result_max_chars", cut)
    out = await _search(_search_host([kept, clipped_away]), _ctx(turn))

    assert [e["thread_ts"] for e in out["results"]] == [OLD_ROOT, late_root]
    assert turn.discovered_thread_roots == frozenset({OLD_ROOT})


@pytest.mark.asyncio
async def test_a_truncated_root_quoted_in_an_earlier_message_enrolls_nothing(monkeypatch):
    """T94, THE COLLISION. The surviving check is about a root's OWN STRUCTURED FIELD, not about
    its digits being somewhere in the bytes.

    An earlier hit's message text QUOTES the later hit's root — the most ordinary thing in the
    world, someone pasting a timestamp — and the later hit is then clipped away entirely. A
    search for the digits finds them (in the first hit's `text`) and would authorize a thread the
    model was never shown, sourced from prose. That is the precise thing §2g exists to refuse,
    and it is why the check reads the `"thread_ts": "…"` PAIR: JSON escapes the quotes inside a
    string value, so no message text can forge one."""
    late_root = "1000.000900"
    kept = reply_hit(message_ts="1000.000950",
                     content=f"cert renewal — see the thread at {late_root} for the rest")
    clipped_away = hit(message_ts="1000.000200", thread_ts=late_root,
                       content="cert renewal " + "y" * 400)

    probe_turn = _turn()
    probe = await _search(_search_host([kept, clipped_away]), _ctx(probe_turn))
    cut = serialize_tool_result(probe).index(f'"thread_ts": "{late_root}"')

    turn = _turn()
    monkeypatch.setattr(config, "tool_result_max_chars", cut)
    out = await _search(_search_host([kept, clipped_away]), _ctx(turn))

    clipped = serialize_tool_result(out)
    assert late_root in clipped, "the digits ARE in what the model read — as somebody's prose"
    assert f'"thread_ts": "{late_root}"' not in clipped, "…but its own field never arrived"
    assert turn.discovered_thread_roots == frozenset({OLD_ROOT})


@pytest.mark.asyncio
async def test_a_failed_tool_result_enrolls_nothing():
    """T83. `ok: False` from all three tools. A refusal proves no thread."""
    turn = _turn()
    failed = {"ok": False, "error": "not_accessible", "channel": CHANNEL,
              "thread_ts": OLD_ROOT,
              "messages": [{"ts": OLD_ROOT, "reply_count": 3}]}
    host = _history_host(failed)
    for name in ("fetch_thread_messages", "fetch_channel_history"):
        await _history(host, _ctx(turn), name, {"channel_id": CHANNEL,
                                                     "thread_ts": OLD_ROOT})
    search = _search_host([reply_hit()])
    # The scan's own refusal shape: the channel-read gate says no, so the tool returns the
    # generic `not_accessible` payload and never fetches a page. (A Slack FETCH failure is a
    # different thing entirely — that comes back `ok: True` with an honest incomplete coverage
    # block, which is the point of §S7 and is covered in test_search_tool.py.)
    search._authorize_channel_read = AsyncMock(return_value=("DENY", "bot_not_member"))
    out = await _search(search, _ctx(turn))

    assert out["ok"] is False
    assert turn.discovered_thread_roots == frozenset()


@pytest.mark.asyncio
async def test_a_successful_fetch_thread_messages_enrolls_its_root():
    """T95. The positive case, end to end: an authorized read enrolls the root it returned, and
    the REAL executor then accepts a post there."""
    turn = _turn()
    host = _history_host({"ok": True, "channel": CHANNEL, "thread_ts": OLD_ROOT,
                          "messages": [{"ts": OLD_ROOT, "text": "the infra sync"}]})

    await _history(host, _ctx(turn), "fetch_thread_messages", {"channel_id": CHANNEL, "thread_ts": OLD_ROOT})

    assert turn.discovered_thread_roots == frozenset({OLD_ROOT})
    poster = _post_host()
    out = await poster.execute_post_to_thread(
        SimpleNamespace(channel_id=CHANNEL, thread_ts="9000.0", trigger_ts="9000.0",
                        trusted_thread_roots=_trusted(turn), turn=turn),
        {"thread_ts": OLD_ROOT, "text": "answered over there"})
    assert out["ok"] is True and out["thread_ts"] == OLD_ROOT


@pytest.mark.asyncio
async def test_fetch_channel_history_enrolls_only_roots_with_replies():
    """T84. A message with no thread is not a target, so only entries whose `reply_count` says a
    thread hangs off them enroll."""
    turn = _turn()
    host = _history_host({"ok": True, "channel": CHANNEL, "thread_ts": None, "messages": [
        {"ts": "1000.000001", "reply_count": 3},     # a real thread
        {"ts": "1000.000002"},                       # a bare top-level message
        {"ts": "1000.000003", "reply_count": 0},
        {"ts": "1000.000004", "reply_count": False},
        {"ts": "1000.000005", "reply_count": True},  # True is an int — and not a reply count
    ]})

    await _history(host, _ctx(turn), "fetch_channel_history", {"channel_id": CHANNEL})

    assert turn.discovered_thread_roots == frozenset({"1000.000001"})


@pytest.mark.asyncio
@pytest.mark.parametrize("replies,enrolls", [
    (0, False), (False, False), (True, False), ("3", False), (3.0, False), (None, False),
    ("__missing__", False), (3, True),
], ids=["zero", "false", "true", "string", "float", "null", "missing", "three"])
async def test_the_reply_count_predicate_is_exact(replies, enrolls):
    """T97. `isinstance(v, int) and not isinstance(v, bool) and v > 0`, and nothing looser.
    `True` is the case that catches a bare truthiness test, because `True` is an `int`."""
    turn = _turn()
    entry = {"ts": OLD_ROOT}
    if replies != "__missing__":
        entry["reply_count"] = replies
    host = _history_host({"ok": True, "channel": CHANNEL, "thread_ts": None, "messages": [entry]})

    await _history(host, _ctx(turn), "fetch_channel_history", {"channel_id": CHANNEL})

    assert (turn.discovered_thread_roots == frozenset({OLD_ROOT})) is enrolls


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["get_message_permalink", "fetch_channel_info",
                                  "fetch_pinned_messages", "resolve_channel_name"])
async def test_only_the_three_named_tools_enroll(name):
    """T96. The source allowlist. Each of these returns a result carrying a plausible
    `thread_ts`, and none of them was asked for a thread root.

    The set itself is pinned EXACTLY, not sampled: a fourth source added later would slip past
    any list of rejected names this test happened to enumerate."""
    from message_processor.turn_runtime import DISCOVERY_SOURCES

    assert DISCOVERY_SOURCES == frozenset({
        "fetch_thread_messages", "fetch_channel_history", "search_slack"})

    turn = _turn()
    host = _history_host({"ok": True, "channel": CHANNEL, "thread_ts": OLD_ROOT,
                          "messages": [{"ts": OLD_ROOT, "reply_count": 3}]})

    await _history(host, _ctx(turn), name, {"channel_id": CHANNEL, "message_ts": OLD_ROOT})

    assert turn.discovered_thread_roots == frozenset()
    assert turn.enroll_discovered_root(channel_id=CHANNEL, root_ts=OLD_ROOT, source=name) is False


# =============================================== 3. WHAT DOES ENROLLMENT BUY?

@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    from database import DatabaseManager
    db = DatabaseManager(platform="slack")
    yield db
    db.conn.close()


def _handler_host(loop_result, db, *, streaming=True):
    """The REAL handler — EITHER twin (§5.4a covers both) — with the REAL provenance writer and
    its REAL task registry.

    `_persist_tool_provenance` and `_schedule_async_call` are the production methods on purpose:
    stubbing either would assert the test's own scaffolding, and a fixture cannot catch a MISSING
    PRODUCTION WRITER — which is exactly what was missing before §5.4a."""
    from message_processor.handlers.text import TextHandlerMixin
    from message_processor.utilities import MessageUtilitiesMixin

    host = MagicMock()
    method = (TextHandlerMixin._handle_streaming_text_response if streaming
              else TextHandlerMixin._handle_text_response)
    host.handler = method.__get__(host)
    for cls, name in ((MessageUtilitiesMixin, "_persist_tool_provenance"),
                      (MessageUtilitiesMixin, "_schedule_async_call"),
                      (MessageUtilitiesMixin, "drain_background_tasks")):
        setattr(host, name, getattr(cls, name).__get__(host))
    host._background_tasks = set()
    host.db = db
    host._is_reaction_only = MagicMock(return_value=False)
    host.mcp_manager = MagicMock()

    async def _passthru(m, *a, **k):
        return m

    async def _none(*a, **k):
        return None

    async def _empty(*a, **k):
        return ""

    host._inject_image_analyses = _passthru
    host._pre_trim_messages_for_api = _passthru
    host._build_channel_info = _empty
    host._drop_dead_containers = _none
    host._resolve_ci_container = _none
    host._prepare_sandbox_tools = _none
    host._cleanup_silent_stream = AsyncMock()
    host._get_system_prompt = MagicMock(return_value="sys")
    host._build_suffix_context = MagicMock(return_value="")
    host._build_participant_roster = MagicMock(return_value="")
    host._build_tools_array = MagicMock(return_value=[{"type": "function", "name": "t"}])
    host._materialize_request_tools = MagicMock(
        return_value=(MagicMock(), {"model": "m"}, True, None))
    host._build_tool_context = MagicMock(return_value=SimpleNamespace(
        background_job_started=False, sandbox_image_assets=[], mounted_files=[]))
    host._add_message_with_token_management = MagicMock()
    host.openai_client = MagicMock()
    host.openai_client.create_streaming_response_with_tool_loop = AsyncMock(
        return_value=loop_result)
    host.openai_client.create_text_response_with_tool_loop = AsyncMock(return_value=loop_result)
    return host


async def _drive_handler(host, turn, *, streaming=True):
    from unittest.mock import patch

    from base_client import Message

    message = Message(text="what was the cert blocker?", user_id="U1", channel_id=CHANNEL,
                      thread_id="9000.0", metadata={"ts": "9000.0"})
    thread_state = SimpleNamespace(
        messages=[{"role": "user", "content": "hi"}], channel_id=CHANNEL, thread_ts="9000.0",
        current_model="gpt-5.6-sol", config_overrides={}, has_summary_head=False,
        channel_directives=None, record_usage=MagicMock(), last_usage=None, participants={})

    async def fake_config(**kw):
        return {"model": "gpt-5.6-sol", "temperature": 1.0, "max_tokens": 100,
                "enable_streaming": streaming, "enable_code_interpreter": False}

    client = MagicMock()
    client.name = "Slack"
    client.supports_streaming = MagicMock(return_value=True)
    client.supports_native_streaming = MagicMock(return_value=False)
    client.get_streaming_config = MagicMock(
        return_value={"update_interval": 0.0, "buffer_size": 1, "min_interval": 0.0})
    with patch.object(config, "get_thread_config_async", side_effect=fake_config):
        return await host.handler("what was the cert blocker?", thread_state, client, message,
                                  thinking_id=None, turn=turn)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False], ids=["streaming", "non-streaming"])
async def test_search_authorizes_a_post_to_an_old_root(temp_db, monkeypatch, streaming):
    """T80. The whole wave in one drive, on BOTH handler twins (§5.4a covers both).

    (i) AUTHORIZATION, through the REAL `ToolRegistry.dispatch_all` — ROUNDS, not a hand-called
        resolver, because the round boundary is the mechanism: a root far below the periphery
        floor, which this turn's stream never labelled, is refused in the round before the
        search and accepted in the round after it.
    (ii) PROVENANCE (§5.4a). The same turn's words go into that thread, so the handler takes the
        words-elsewhere early return — and it must still leave a `message_tool_usage` row naming
        `search_slack`, keyed on the DESTINATION post rather than on the turn's own trigger.
    (iii) THE COMMITTED DISCRIMINATOR. A second post_to_thread destination that Slack accepted
        but the turn never committed to gets NO row: with parallel effects, writing both would
        attribute this turn's tools to a message the turn did not stand behind.

    The test SYNCHRONISES WITH THE REAL WRITER rather than mocking it away — the persist is
    scheduled through `_schedule_async_call`, so the rows are read only after
    `drain_background_tasks()`, the same registry the production shutdown path drains. Asserting
    immediately would race the writer.
    """
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    monkeypatch.setattr(config, "enable_tool_provenance", True)
    monkeypatch.setattr(config, "enable_tool_result_memory", False)

    from tool_registry import ToolRegistry

    turn = _turn()
    search = _search_host([reply_hit()])
    poster = _post_host()
    registry = ToolRegistry()
    registry.register({"type": "function", "name": "search_slack", "parameters": {}},
                      search.execute_search_tool)
    registry.register({"type": "function", "name": "post_to_thread", "parameters": {}},
                      poster.execute_post_to_thread)

    # ONE context for the whole turn, exactly as `_build_tool_context` makes it — every round
    # below reuses it, which is why the re-stamp has to happen at the round boundary.
    ctx = _ctx(turn)
    ctx.trusted_thread_roots = _trusted(turn)
    post_call = {"name": "post_to_thread", "call_id": "p",
                 "arguments": {"thread_ts": OLD_ROOT, "text": "answered over there"}}

    # (i) ROUND 1 — the old root is not in the stream, and the executor says so.
    before = await registry.dispatch_all(ctx, [dict(post_call, call_id="p1")])
    assert before[0]["error"] == "unknown_thread"

    # ROUND 2 — the search returns the hit that proves the thread exists.
    found = await registry.dispatch_all(ctx, [
        {"name": "search_slack", "call_id": "s", "arguments": {"query": "cert renewal"}}])
    assert found[0]["ok"] is True

    # ROUND 3 — the same root, the same executor, the same context object: now authorized.
    after = await registry.dispatch_all(ctx, [dict(post_call, call_id="p3")])
    assert after[0]["ok"] is True

    # (ii)+(iii) the turn's own words landed over there, and one surface was never committed.
    committed_ts, observed_only_ts = "9100.0", "9200.0"
    turn.visible_action_committed = True
    turn.mark_destination_committed(channel_id=CHANNEL, first_ts=committed_ts,
                                    kind=DEST_KIND_POST_TO_THREAD, thread_root_ts=OLD_ROOT,
                                    text="answered over there")
    turn.note_destination_observed(channel_id=CHANNEL, first_ts=observed_only_ts,
                                   kind=DEST_KIND_POST_TO_THREAD, thread_root_ts=OLD_ROOT)

    host = _handler_host({"text": "", "tools_used": [], "local_tool_calls": [
        {"name": "search_slack", "ok": True, "gist": "cert renewal"},
        {"name": "post_to_thread", "ok": True, "gist": OLD_ROOT},
    ]}, temp_db, streaming=streaming)
    response = await _drive_handler(host, turn, streaming=streaming)

    assert response.content == "" and response.metadata["posted"] is False
    if streaming:
        # Only the streamed twin has a placeholder to take down; the other never opened one.
        host._cleanup_silent_stream.assert_awaited()

    await host.drain_background_tasks()
    rows = await temp_db.get_thread_tool_usage_async(f"{CHANNEL}:{OLD_ROOT}")

    assert committed_ts in rows, "the committed cross-thread post carries its provenance"
    # NAMES **AND GISTS** — the pinned scope of §5.4a. A names-only regression would still let
    # an operator see that a search ran and never what it searched for, which is most of the
    # answer to "where did this come from?".
    assert {"tool_name": "search_slack", "gist": "cert renewal"} in rows[committed_ts]
    assert {"tool_name": "post_to_thread", "gist": OLD_ROOT} in rows[committed_ts]
    assert observed_only_ts not in rows, "an observed-only surface is not this turn's word"
    assert "9000.0" not in rows, "keyed on the destination post, never on the turn's trigger"

@pytest.mark.asyncio
async def test_same_round_enrollment_does_not_authorize_the_same_round_post(monkeypatch):
    """T82. A `search_slack` and a `post_to_thread` dispatched in ONE round: the post is REFUSED.

    The executor's own comment forbids the alternative — a set widened at the moment of posting
    would make "authorized" mean "the model named it in the same breath". `dispatch_all`
    re-stamps the allowlist at its TOP, before any of the round's calls run, so this property is
    ORDERING rather than luck: whatever the round enrolls lands after the stamp it would have had
    to beat, and is available from the next round. The model searches, then posts.

    THE SIBLING POST IS HELD UNTIL THE SEARCH HAS COMMITTED, and that ordering is the test.
    Left to the event loop the post resolves first, and then it is refused for an uninteresting
    reason — the root did not exist yet — which would let the real defect through unnoticed. So
    the post waits for the commit and is refused with the root ALREADY ENROLLED and visible on
    the turn. That is the hard case: everything the forbidden check would read is sitting there,
    and the answer is still no, because the set this round authorizes against was resolved before
    the round began.

    ON THE MUTATIONS. §5.7's "stamp per-call rather than per-round" still does NOT fail here, and
    the code is not wrong: `asyncio.gather` reaches both per-call stamps before the search's API
    await resolves, so per-call and per-round are observationally identical at that seam. The
    mutation that IS reachable — and that the ordering above exists to keep reachable — is
    `execute_post_to_thread` unioning `ctx.turn.discovered_thread_roots` at CHECK time instead of
    reading the set the round resolved. Anyone re-verifying W3 should mutate that line.
    """
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    from tool_registry import ToolRegistry

    turn = _turn()
    search = _search_host([reply_hit()])
    poster = _post_host()
    saw_committed = []

    async def post_after_the_search_commits(call_ctx, args):
        for _ in range(400):
            if turn.discovered_thread_roots:
                break
            await asyncio.sleep(0.005)
        saw_committed.append(set(turn.discovered_thread_roots))
        return await poster.execute_post_to_thread(call_ctx, args)

    registry = ToolRegistry()
    registry.register({"type": "function", "name": "search_slack", "parameters": {}},
                      search.execute_search_tool)
    registry.register({"type": "function", "name": "post_to_thread", "parameters": {}},
                      post_after_the_search_commits)

    ctx = _ctx(turn)
    ctx.trusted_thread_roots = _trusted(turn)      # stamped for the round, as production does
    assert OLD_ROOT not in ctx.trusted_thread_roots

    results = await registry.dispatch_all(ctx, [
        {"name": "search_slack", "arguments": {"query": "cert renewal"}, "call_id": "c1"},
        {"name": "post_to_thread",
         "arguments": {"thread_ts": OLD_ROOT, "text": "answered over there"}, "call_id": "c2"},
    ])

    assert results[0]["ok"] is True, "the search itself ran and returned its hit"
    assert saw_committed == [{OLD_ROOT}], (
        "the post ran AFTER the commit — without that this test refuses for the wrong reason "
        "and stops guarding anything")
    assert results[1]["error"] == "unknown_thread"
    poster.send_message.assert_not_awaited()
    # …and the round did its job: the root IS enrolled, ready for the round after this one.
    assert turn.discovered_thread_roots == frozenset({OLD_ROOT})
    assert OLD_ROOT in _trusted(turn)


@pytest.mark.asyncio
async def test_a_broken_stream_still_denies_all_but_the_discovered_set():
    """T85. FXP3B finding 3 preserved: a stream that is PRESENT but cannot say what it showed
    answers with the EMPTY set, never None. The discovered roots union onto that empty set — they
    were proved by a tool result, not by the broken stream — and nothing else passes."""
    turn = _turn()
    host = _search_host([reply_hit()])
    await _search(host, _ctx(turn))
    turn.channel_stream = MagicMock()   # present, unreadable

    trusted = _trusted(turn)

    assert trusted == frozenset({OLD_ROOT})
    poster = _post_host()
    for target, expected in ((OLD_ROOT, True), ("9000.0", False)):
        out = await poster.execute_post_to_thread(
            SimpleNamespace(channel_id=CHANNEL, thread_ts="8000.0", trigger_ts="8000.0",
                            trusted_thread_roots=trusted, turn=turn),
            {"thread_ts": target, "text": "hello"})
        assert out["ok"] is expected
