"""post_to_thread as a FOREIGN-TARGET protocol (spec §14), not just a second place to post.

Under full channel visibility the model can see every thread in the room, so "post over there"
stopped being a rare hand-off and became something it can decide on its own. That makes three
questions load-bearing, and this file is organized around them:

1. **MAY IT?** The target must be a thread this turn's stream actually SHOWED the model — the
   frozen trusted-root set, checked BEFORE anything is sent. Not a live Slack lookup: a lookup
   answers "does that thread exist", which is a different question from "is that a thread you were
   given". A DM (no stream) keeps the legacy behaviour verbatim, and an EMPTY set still enforces.
2. **DID IT LAND, AND IS IT OURS?** The post and the receipt that claims it are ONE critical
   section (the effect lease), booked against the TARGET root rather than the turn's own thread,
   and the destination record is committed with what Slack actually took.
3. **WHAT DOES THE ORIGIN SEE?** Nothing, by design. A delivered cross-thread post means empty
   prose here is a valid ending — not the bare-empty contract violation that shape used to mean —
   and neither handler may apologize for it or persist it as this exchange.

The refusal rails from F23 stay: current-thread double-post, empty text, missing ts, disabled,
never raises. The per-thread mute mechanism was removed, so the tool posts with NO mute lookup.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import prompts
from config import config
from markdown_converter import MarkdownConverter
from message_processor.turn_runtime import (DEST_KIND_POST_TO_THREAD, POST_TO_THREAD_TOOL,
                                           TurnRuntime)
from slack_client.formatting.text import SlackFormattingMixin
from slack_client.messaging import SlackMessagingMixin
from tests.unit import channel_turn_harness as harness
from tool_registry import SURFACE_CHANNEL, ToolContext, ToolRegistry


def _ctx(channel="C1", thread="root.1", trigger="msg.1", *, trusted=None, turn=None):
    # The per-thread mute mechanism was removed: post_to_thread no longer consults any mute
    # state. The db still exposes is_thread_muted_async so a test can prove it is NEVER called.
    db = MagicMock()
    db.is_thread_muted_async = AsyncMock(return_value=True)
    return ToolContext(channel_id=channel, thread_ts=thread, trigger_ts=trigger,
                       client=MagicMock(), db=db, trusted_thread_roots=trusted, turn=turn)


def _light_host():
    """Host with the executor bound and send_message MOCKED (refusal / routing tests)."""
    s = MagicMock()
    s.execute_post_to_thread = SlackMessagingMixin.execute_post_to_thread.__get__(s)
    s.send_message = AsyncMock(return_value="900.0")
    return s


def _real_send_host():
    """Host with the REAL send_message + format_text so markdown conversion and the
    receipt record actually run (proves the standard messaging path is used)."""
    s = MagicMock()
    s.execute_post_to_thread = SlackMessagingMixin.execute_post_to_thread.__get__(s)
    s.send_message = SlackMessagingMixin.send_message.__get__(s)
    s.format_text = SlackFormattingMixin.format_text.__get__(s)
    s._encode_mentions = lambda t: t
    s.markdown_converter = MarkdownConverter(platform="slack")
    s._record_receipt = SlackMessagingMixin._record_receipt.__get__(s)
    s._compose_reply_with_footer = SlackMessagingMixin._compose_reply_with_footer.__get__(s)
    s._SECTION_TEXT_LIMIT = SlackMessagingMixin._SECTION_TEXT_LIMIT
    s._FOOTER_INLINE_MAX = SlackMessagingMixin._FOOTER_INLINE_MAX
    s.MAX_MESSAGE_LENGTH = 3900
    s.app.client.chat_postMessage = AsyncMock(return_value={"ts": "900.0"})
    return s


# ------------------------------------------------------------------- schema, both surfaces

def _schema(surface="dm"):
    host = MagicMock()
    host._POST_TO_THREAD_DESCRIPTION = SlackMessagingMixin._POST_TO_THREAD_DESCRIPTION
    host._POST_TO_THREAD_TARGET_DESCRIPTION = (
        SlackMessagingMixin._POST_TO_THREAD_TARGET_DESCRIPTION)
    host._post_to_thread_schema = SlackMessagingMixin._post_to_thread_schema
    getter = (SlackMessagingMixin.get_post_to_thread_channel_schema if surface == "channel"
              else SlackMessagingMixin.get_post_to_thread_tool_schema)
    return getter.__get__(host)()


@pytest.mark.parametrize("surface", ["dm", "channel"])
def test_schema_shape_and_required_params(surface):
    schema = _schema(surface)
    assert schema["name"] == "post_to_thread" == POST_TO_THREAD_TOOL
    props = schema["parameters"]["properties"]
    assert set(schema["parameters"]["required"]) == {"thread_ts", "text"}
    assert "thread_ts" in props and "text" in props
    # current-channel-only: no channel_id parameter is exposed
    assert "channel_id" not in props


def test_the_two_surfaces_now_describe_the_tool_differently():
    """RETIRED TRIPWIRE (P3b). This asserted the two schemas were still IDENTICAL — P3a's proof that
    landing the read moved no bytes, and true only while the channel constants were empty. The
    invariant that outlives it: the channel surface carries its own words on BOTH the description and
    the target parameter, and the DM surface is pinned to the legacy constants byte for byte. The
    channel wording itself is pinned in test_channel_restraint_prompts.py."""
    channel, dm = _schema("channel"), _schema("dm")
    assert channel != dm
    assert channel["description"] == prompts.CHANNEL_POST_TO_THREAD_DESCRIPTION
    assert (channel["parameters"]["properties"]["thread_ts"]["description"]
            == prompts.CHANNEL_POST_TO_THREAD_TARGET_DESCRIPTION)
    assert dm["description"] == SlackMessagingMixin._POST_TO_THREAD_DESCRIPTION
    assert (dm["parameters"]["properties"]["thread_ts"]["description"]
            == SlackMessagingMixin._POST_TO_THREAD_TARGET_DESCRIPTION)

    with patch.object(prompts, "CHANNEL_POST_TO_THREAD_DESCRIPTION", "Post it once, over there."), \
         patch.object(prompts, "CHANNEL_POST_TO_THREAD_TARGET_DESCRIPTION", "A stream label."):
        channel, dm = _schema("channel"), _schema("dm")
    assert channel["description"] == "Post it once, over there."
    assert channel["parameters"]["properties"]["thread_ts"]["description"] == "A stream label."
    assert dm["description"] == SlackMessagingMixin._POST_TO_THREAD_DESCRIPTION


def test_the_registry_hands_the_channel_surface_its_own_schema():
    """The selection is the registry's, not the executor's: one registration, two schemas."""
    host = MagicMock()
    host._POST_TO_THREAD_DESCRIPTION = SlackMessagingMixin._POST_TO_THREAD_DESCRIPTION
    host._POST_TO_THREAD_TARGET_DESCRIPTION = (
        SlackMessagingMixin._POST_TO_THREAD_TARGET_DESCRIPTION)
    host._post_to_thread_schema = SlackMessagingMixin._post_to_thread_schema
    reg = ToolRegistry()
    reg.register(SlackMessagingMixin.get_post_to_thread_tool_schema.__get__(host)(),
                 AsyncMock(),
                 channel_schema=SlackMessagingMixin.get_post_to_thread_channel_schema.__get__(host))

    with patch.object(prompts, "CHANNEL_POST_TO_THREAD_DESCRIPTION", "channel words"):
        dm_schemas = {s["name"]: s for s in reg.schemas()}
        channel_schemas = {s["name"]: s for s in reg.schemas(surface=SURFACE_CHANNEL)}

    assert "post_to_thread" in dm_schemas and "post_to_thread" in channel_schemas
    assert channel_schemas["post_to_thread"]["description"] == "channel words"
    assert dm_schemas["post_to_thread"]["description"] != "channel words"


def test_the_registered_channel_schema_is_wired_on_the_real_bot():
    """A schema nothing registers is a schema nobody reads."""
    import inspect

    from slack_client.base import SlackBot

    src = inspect.getsource(SlackBot._build_tool_registry)
    assert "channel_schema=self.get_post_to_thread_channel_schema" in src


# ------------------------------------------------------------------- 1. MAY IT? authorization

def _stream_with_two_threads():
    """A real serialized stream: two rooted threads plus a top-level message with no replies."""
    return harness.build_stream([
        harness.normalized("10.0", "the first question"),
        harness.normalized("11.0", "an answer", thread_root_ts="10.0"),
        harness.normalized("20.0", "another question"),
        harness.normalized("21.0", "another answer", thread_root_ts="20.0"),
        harness.normalized("30.0", "a top-level line nobody replied to"),
    ], channel_id="C1")


def test_the_trusted_roots_are_exactly_the_threads_the_stream_labelled():
    stream = _stream_with_two_threads()
    assert stream.trusted_thread_roots == frozenset({"10.0", "20.0"})
    # …and the set matches what the model can READ: the rendered `thread=` labels.
    rendered = "\n".join(item.content for item in stream.message_items)
    for root in stream.trusted_thread_roots:
        assert f"thread={root}" in rendered
    # A top-level message with no replies carries no thread label, so it is not a target. That is
    # the schema's promise kept, not an oversight — P4's root anchors widen it at PIN time.
    assert "30.0" not in stream.trusted_thread_roots


@pytest.mark.asyncio
async def test_an_untrusted_target_is_refused_before_anything_is_sent(monkeypatch):
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()
    turn = TurnRuntime()
    ctx = _ctx(thread="10.0", trigger="10.0",
               trusted=_stream_with_two_threads().trusted_thread_roots, turn=turn)

    out = await host.execute_post_to_thread(ctx, {"thread_ts": "99.9", "text": "hello"})

    assert out["ok"] is False and out["error"] == "unknown_thread"
    host.send_message.assert_not_awaited()
    assert turn.destinations == [] and turn.visible_action_committed is False
    # The refusal points at where a real id comes from, so the model can recover.
    assert "stream" in out["message"]


@pytest.mark.asyncio
async def test_an_invented_root_is_still_refused(monkeypatch):
    """T86. The bait case, unchanged by W3 — now with a POPULATED discovered set present.

    Widening the allowlist from tool results is the one thing that could have turned "the model
    named a ts" into an authorization, so the refusal is re-proved in the state where it would
    have: a turn that really did discover a root this turn, asked to post at a different one it
    made up."""
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    from message_processor.handlers.text import _trusted_thread_roots
    from tests.unit import channel_turn_harness as h

    turn = TurnRuntime()
    h.pin_channel_turn(turn, stream=_stream_with_two_threads(), channel_id="C1")
    assert turn.enroll_discovered_root(channel_id="C1", root_ts="40.0",
                                       source="search_slack") is True
    trusted = _trusted_thread_roots(turn)
    assert trusted == frozenset({"10.0", "20.0", "40.0"}), "the discovered root really is in"

    host = _light_host()
    out = await host.execute_post_to_thread(
        _ctx(thread="10.0", trigger="10.0", trusted=trusted, turn=turn),
        {"thread_ts": "99.9", "text": "hello"})

    assert out["ok"] is False and out["error"] == "unknown_thread"
    host.send_message.assert_not_awaited()
    assert turn.destinations == [] and turn.visible_action_committed is False
    # The refusal now names the way through, so a model that guessed can recover by opening it.
    assert "fetch_thread_messages" in out["message"]


@pytest.mark.asyncio
async def test_a_trusted_target_is_authorized_and_posted(monkeypatch):
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()
    ctx = _ctx(thread="10.0", trigger="10.0",
               trusted=_stream_with_two_threads().trusted_thread_roots)

    out = await host.execute_post_to_thread(ctx, {"thread_ts": "20.0", "text": "over here"})

    assert out["ok"] is True and out["thread_ts"] == "20.0"
    assert host.send_message.await_args.args == ("C1", "20.0", "over here")


@pytest.mark.asyncio
async def test_an_empty_trusted_set_still_enforces(monkeypatch):
    """A channel turn whose stream rendered no thread has no thread to post into. Empty is a
    frozenset, not None — the difference between "nothing was shown" and "no stream exists"."""
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()

    out = await host.execute_post_to_thread(_ctx(trusted=frozenset()),
                                            {"thread_ts": "OTHER.9", "text": "hello"})

    assert out["error"] == "unknown_thread"
    host.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_dm_turn_carries_no_allowlist_and_posts_exactly_as_before(monkeypatch):
    """The DM executor regression. There is no stream in a DM, so `trusted_thread_roots` is None
    and the tool behaves verbatim — including for a target no stream could have shown it."""
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()
    ctx = _ctx(channel="D1", thread="root.1")
    assert ctx.trusted_thread_roots is None

    out = await host.execute_post_to_thread(ctx, {"thread_ts": "OTHER.9", "text": "hello"})

    assert out["ok"] is True and out["posted_ts"] == "900.0"
    assert host.send_message.await_args.args == ("D1", "OTHER.9", "hello")


def test_the_handler_reads_the_allowlist_off_the_pinned_stream():
    """The seam: whatever the turn pinned is what the executor authorizes against.

    None is the widest answer there is — it means "no stream exists", and the executor keeps its
    legacy behaviour — so it stays exclusive to turns that genuinely have no stream. A stream that
    is PRESENT but cannot say what it showed denies EVERY target instead. It used to widen to
    None, which made a malformed stream the most permissive one in the system."""
    from message_processor.handlers.text import _trusted_thread_roots

    turn = TurnRuntime()
    assert _trusted_thread_roots(turn) is None, "no stream: legacy behaviour"
    turn.channel_stream = _stream_with_two_threads()
    assert _trusted_thread_roots(turn) == frozenset({"10.0", "20.0"})
    turn.channel_stream = MagicMock()
    assert _trusted_thread_roots(turn) == frozenset(), "a present-but-unreadable stream denies all"
    turn.channel_stream = SimpleNamespace(trusted_thread_roots=None)
    assert _trusted_thread_roots(turn) == frozenset(), "and so does one with no roots at all"


@pytest.mark.asyncio
async def test_a_channel_turn_whose_stream_is_unreadable_posts_nowhere(monkeypatch):
    """The half that matters: the deny-all set reaches the executor and every target is refused —
    including one the model could have read off a stream that rendered correctly."""
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()

    out = await host.execute_post_to_thread(_ctx(trusted=frozenset()),
                                            {"thread_ts": "10.0", "text": "hello"})

    assert out["error"] == "unknown_thread"
    host.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_current_thread_is_refused_before_the_allowlist_is_consulted(monkeypatch):
    """same_thread comes first, and its message is the useful one: reply normally. The origin root
    IS in the trusted set (the stream showed it), so this rail is what keeps a double-post out."""
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()
    ctx = _ctx(thread="10.0", trigger="10.0",
               trusted=_stream_with_two_threads().trusted_thread_roots)

    out = await host.execute_post_to_thread(ctx, {"thread_ts": "10.0", "text": "hello"})

    assert out["error"] == "same_thread"
    host.send_message.assert_not_awaited()


# --------------------------------------------------- 2. DID IT LAND, AND IS IT OURS?

@pytest.mark.asyncio
async def test_posts_markdown_converted_text_to_target(monkeypatch):
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _real_send_host()
    out = await host.execute_post_to_thread(
        _ctx(thread="root.1", trigger="msg.1"),
        {"thread_ts": "OTHER.9", "text": "Answer: **done**"},
    )
    assert out["ok"] is True and out["thread_ts"] == "OTHER.9"
    call = host.app.client.chat_postMessage.await_args
    assert call.kwargs["thread_ts"] == "OTHER.9"
    # markdown converted to Slack mrkdwn (**bold** -> *bold*)
    assert "*done*" in call.kwargs["text"]


@pytest.mark.asyncio
async def test_records_the_post_against_the_target_thread(monkeypatch):
    """Our own post is booked against the thread it LANDED in, not the turn's own conversation.
    Re-baselined from the retired channel-pulse record onto the receipt that replaced it — the
    root attribution is the fact that mattered, and it still does."""
    from message_processor import outbound_receipts

    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    booked = {}

    async def _record(**kwargs):
        booked.update(kwargs)

    monkeypatch.setattr(outbound_receipts, "record_transport_post", _record)
    host = _real_send_host()
    await host.execute_post_to_thread(
        _ctx(thread="root.1"), {"thread_ts": "OTHER.9", "text": "hi there"}
    )
    assert booked["thread_root_ts"] == "OTHER.9" and booked["message_ts"] == "900.0"


@pytest.mark.asyncio
async def test_records_the_destination_on_the_turn(monkeypatch):
    """A cross-thread post is one of the five surfaces a turn's own words can land on, and the one
    that lands somewhere other than the conversation it was asked in. turn_outcome reports observed
    destinations and memory extraction reads committed ones, so a post the ledger cannot see is a
    turn that visibly spoke and is filed as having said nothing.

    Accept is finalization here — the post is written once and never edited — so the record must
    come out COMMITTED with its chars, keyed on the TARGET root."""
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()
    turn = TurnRuntime()
    ctx = _ctx(thread="root.1", turn=turn)
    out = await host.execute_post_to_thread(
        ctx, {"thread_ts": "OTHER.9", "text": "Answered over in the other thread."})
    assert out["ok"] is True
    assert len(turn.destinations) == 1
    record = turn.destinations[0]
    assert record.kind == DEST_KIND_POST_TO_THREAD == "post_to_thread"
    assert record.channel_id == "C1"
    assert record.thread_root_ts == "OTHER.9"     # where it landed, not root.1
    assert record.first_ts == "900.0"
    assert record.committed is True
    assert record.chars == len("Answered over in the other thread.")
    assert turn.committed_destinations == [record]
    assert record.as_payload()["state"] == "committed"


@pytest.mark.asyncio
async def test_a_refused_or_failed_post_records_no_destination(monkeypatch):
    """Nothing landed, so there is nowhere to record — a destination for a post that never
    happened would read as a delivery in the ledger."""
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    for args, send in ((
            {"thread_ts": "root.1", "text": "hello"}, AsyncMock(return_value="900.0")), (
            {"thread_ts": "OTHER.9", "text": "hello"}, AsyncMock(return_value=None)), (
            {"thread_ts": "OTHER.9", "text": "hello"},
            AsyncMock(side_effect=RuntimeError("slack down")))):
        host = _light_host()
        host.send_message = send
        turn = TurnRuntime()
        ctx = _ctx(thread="root.1", turn=turn)
        out = await host.execute_post_to_thread(ctx, args)
        assert out["ok"] is False
        assert turn.destinations == []
        assert turn.visible_action_committed is False


@pytest.mark.asyncio
async def test_a_broken_destination_ledger_still_delivers(monkeypatch):
    """Bookkeeping is downstream of the words. A post that Slack accepted is a success whatever
    the ledger does with it."""
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()
    turn = MagicMock()
    turn.note_destination_observed.side_effect = RuntimeError("ledger broken")
    ctx = _ctx(turn=turn)
    out = await host.execute_post_to_thread(
        ctx, {"thread_ts": "OTHER.9", "text": "hello"})
    assert out["ok"] is True and out["posted_ts"] == "900.0"


@pytest.mark.asyncio
async def test_routes_through_send_message(monkeypatch):
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()
    out = await host.execute_post_to_thread(
        _ctx(), {"thread_ts": "OTHER.9", "text": "hello"}
    )
    assert out["ok"] is True
    # The lease rides every guarded send now; the destination and text are what this
    # test is about.
    assert host.send_message.await_args.args == ("C1", "OTHER.9", "hello")


@pytest.mark.asyncio
async def test_the_post_runs_under_an_effect_lease(monkeypatch):
    """The post and the receipt that claims it are one critical section, so settlement cannot run
    between them (the full ordering is pinned in test_tool_flights.py)."""
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    turn = TurnRuntime()
    seen = []

    async def _send(*a, **k):
        seen.append(turn.held_effect_leases)
        return "900.0"

    host = _light_host()
    host.send_message = _send
    await host.execute_post_to_thread(_ctx(turn=turn), {"thread_ts": "OTHER.9", "text": "hi"})

    assert seen == [("post_to_thread",)]
    assert turn.held_effect_leases == (), "released when the effect settles"


@pytest.mark.asyncio
async def test_a_revoked_turn_posts_nothing_and_says_so(monkeypatch):
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()
    turn = TurnRuntime()
    turn.revoke_effects("straggler refused cancellation")

    out = await host.execute_post_to_thread(_ctx(turn=turn),
                                            {"thread_ts": "OTHER.9", "text": "hi"})

    assert out["ok"] is False and out["error"] == "turn_cancelled"
    assert "nothing was posted" in out["message"]
    host.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_suppression_is_re_raised_not_reported_as_a_failed_post(monkeypatch):
    """The room moved on before this landed. Nothing was attempted, so telling the model the post
    failed would invite a retry of something we deliberately did not do."""
    from message_processor.stale_send_guard import StaleSendSuppressed

    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()
    host.send_message = AsyncMock(side_effect=StaleSendSuppressed(
        scope=("thread", "C1", "10.0"), last_seen_ts="10.0", observed_latest_ts="12.0",
        surface="post_to_thread"))
    turn = TurnRuntime()

    with pytest.raises(StaleSendSuppressed):
        await host.execute_post_to_thread(_ctx(turn=turn), {"thread_ts": "OTHER.9", "text": "x"})

    assert turn.destinations == [] and turn.visible_action_committed is False


# ------------------------------------------------------------------- refusal rails

@pytest.mark.asyncio
async def test_posts_with_no_mute_lookup(monkeypatch):
    # Even against a target the old code would have refused as "muted", the tool now posts and
    # NEVER consults is_thread_muted_async — the mute mechanism is gone.
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()
    ctx = _ctx()
    out = await host.execute_post_to_thread(
        ctx, {"thread_ts": "OTHER.9", "text": "hello"}
    )
    assert out["ok"] is True and out["posted_ts"] == "900.0"
    ctx.db.is_thread_muted_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_trigger_ts_also_counts_as_current(monkeypatch):
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()
    out = await host.execute_post_to_thread(
        _ctx(thread="root.1", trigger="msg.1"), {"thread_ts": "msg.1", "text": "hello"}
    )
    assert out["ok"] is False and out["error"] == "same_thread"


@pytest.mark.asyncio
async def test_empty_text_refused(monkeypatch):
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()
    out = await host.execute_post_to_thread(
        _ctx(), {"thread_ts": "OTHER.9", "text": "   "}
    )
    assert out["ok"] is False and out["error"] == "empty_text"


@pytest.mark.asyncio
async def test_missing_thread_ts_refused(monkeypatch):
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()
    out = await host.execute_post_to_thread(_ctx(), {"text": "hello"})
    assert out["ok"] is False and out["error"] == "missing_thread_ts"


@pytest.mark.asyncio
async def test_disabled_refused(monkeypatch):
    monkeypatch.setattr(config, "enable_post_to_thread_tool", False)
    host = _light_host()
    out = await host.execute_post_to_thread(
        _ctx(), {"thread_ts": "OTHER.9", "text": "hello"}
    )
    assert out["ok"] is False and out["error"] == "disabled"


@pytest.mark.asyncio
async def test_never_raises_on_slack_failure(monkeypatch):
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()
    host.send_message = AsyncMock(side_effect=RuntimeError("slack down"))
    out = await host.execute_post_to_thread(
        _ctx(), {"thread_ts": "OTHER.9", "text": "hello"}
    )
    assert out["ok"] is False and out["error"] == "post_failed"


@pytest.mark.asyncio
async def test_send_returns_none_is_failure(monkeypatch):
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()
    host.send_message = AsyncMock(return_value=None)
    out = await host.execute_post_to_thread(
        _ctx(), {"thread_ts": "OTHER.9", "text": "hello"}
    )
    assert out["ok"] is False and out["error"] == "post_failed"


@pytest.mark.asyncio
async def test_the_turn_outcome_row_names_the_foreign_destination(monkeypatch):
    """§2d: the ledger's only account of a cross-thread post. The turn's KIND stays in the existing
    vocabulary (`detached`) and the cross-thread identity lives in destinations[].kind — a new
    outcome kind would have split the turn population against itself."""
    import json
    import pathlib

    from message_processor import participation_telemetry as pt

    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()
    turn = TurnRuntime()
    turn.stream_build_present = True
    turn.H = "99.0"
    await host.execute_post_to_thread(_ctx(thread="10.0", turn=turn),
                                      {"thread_ts": "OTHER.9", "text": "answered over there"})

    pt.initialize()
    assert pt.emit_turn_outcome(turn, channel_id="C1", trigger_ts="10.0", kind="detached") is True
    pt._drain()
    rows = [json.loads(line) for line in
            (pathlib.Path(config.log_directory) / pt.LOG_NAME).read_text().splitlines()
            if line.strip()]
    row = [r for r in rows if r["event"] == "turn_outcome"][-1]

    assert row["kind"] == "detached"
    assert row["destinations"] == [{"channel_id": "C1", "thread_root_ts": "OTHER.9",
                                   "first_ts": "900.0", "state": "committed",
                                   "chars": len("answered over there"),
                                   "kind": DEST_KIND_POST_TO_THREAD}]
    assert "text" not in row["destinations"][0]


# ------------------------------------------- 3. WHAT DOES THE ORIGIN SEE? (nothing)

def _completion_host(loop_result, *, streaming=False):
    """The REAL handler on a minimal host, driven to the point where empty prose is decided."""
    from message_processor.handlers.text import TextHandlerMixin

    host = MagicMock()
    method = (TextHandlerMixin._handle_streaming_text_response if streaming
              else TextHandlerMixin._handle_text_response)
    host.handler = method.__get__(host)
    host._is_reaction_only = MagicMock(return_value=False)
    host.db = None
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
    host._schedule_async_call = MagicMock()
    host.openai_client = MagicMock()
    host.openai_client.create_text_response_with_tool_loop = AsyncMock(return_value=loop_result)
    host.openai_client.create_streaming_response_with_tool_loop = AsyncMock(
        return_value=loop_result)
    return host


async def _drive(host, turn, *, streaming=False):
    from base_client import Message

    message = Message(text="hi", user_id="U1", channel_id="C1", thread_id="10.0",
                      metadata={"ts": "10.0"})
    thread_state = SimpleNamespace(
        messages=[{"role": "user", "content": "hi"}], channel_id="C1", thread_ts="10.0",
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
        return await host.handler("hi", thread_state, client, message, thinking_id=None,
                                  turn=turn)


def _logged(mock_method):
    return " ".join(str(call.args[0]) for call in mock_method.call_args_list if call.args)


@pytest.mark.asyncio
async def test_the_non_streaming_origin_says_nothing_and_is_not_a_glitch():
    """Empty prose after a delivered cross-thread post is a VALID terminal. Read as the bare-empty
    contract violation it looks like, the turn is warned about and filed as `empty` — while the
    room can read the answer in the other thread."""
    turn = TurnRuntime()
    turn.visible_action_committed = True     # the post landed; Slack is the authority
    host = _completion_host({"text": "", "tools_used": [],
                             "local_tool_calls": [{"name": "post_to_thread", "ok": True}]})

    response = await _drive(host, turn)

    assert response.content == "" and response.metadata["posted"] is False
    assert "Empty non-streaming response" not in _logged(host.log_warning)
    assert "landed elsewhere" in _logged(host.log_info)
    # main.py classifies this as a visible action rather than a silence or an empty turn.
    from main import ChatBotV2
    assert ChatBotV2._classify_visible_action(response, turn) == "detached"


@pytest.mark.asyncio
async def test_a_genuinely_empty_non_streaming_turn_is_still_a_warning():
    """The mutation of the test above: with nothing delivered anywhere, the bare-empty branch is
    exactly right and must stay."""
    turn = TurnRuntime()
    host = _completion_host({"text": "", "tools_used": [], "local_tool_calls": []})

    response = await _drive(host, turn)

    assert response.content == ""
    assert "Empty non-streaming response" in _logged(host.log_warning)
    from main import ChatBotV2
    assert ChatBotV2._classify_visible_action(response, turn) == "empty"


@pytest.mark.asyncio
async def test_the_streamed_origin_tears_its_surface_down_instead_of_apologizing():
    """The finalizer's apology ("I couldn't generate a response") would land directly above an
    answer the room can already read in another thread. So the turn returns before it, and the
    placeholder it never committed to comes down."""
    turn = TurnRuntime()
    turn.visible_action_committed = True
    host = _completion_host({"text": "", "tools_used": [],
                             "local_tool_calls": [{"name": "post_to_thread", "ok": True}]},
                            streaming=True)

    response = await _drive(host, turn, streaming=True)

    assert response.content == "" and response.metadata["posted"] is False
    assert "apologize" not in (response.content or "")
    host._cleanup_silent_stream.assert_awaited()
    assert host._cleanup_silent_stream.await_args.args[-1] == "words-elsewhere"


# ------------------------------------------------------------------- provenance + guidance

def test_provenance_line_includes_post_to_thread():
    from message_processor.tool_provenance import build_provenance, render_used_tools_annotation
    tools = build_provenance([{"name": "post_to_thread", "ok": True, "gist": ""}], [])
    line = render_used_tools_annotation(tools)
    assert "post_to_thread" in line


def test_local_tools_guidance_has_post_to_thread_bullet():
    assert "post_to_thread" in prompts.LOCAL_TOOLS_GUIDANCE
