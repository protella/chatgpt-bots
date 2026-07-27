"""F52 — an EDIT to a recent human message can also drive a reply.

Today a message_changed event reconciles the pulse and re-offers content to ambient memory but
NEVER drives a reply: "@mention the bot, then edit to add the question" gets silence, and a
meaningful edit to an already-answered message never gets a correction. F52 adds that — behind
ENABLE_EDIT_TRIGGERED_REPLIES — through a chain of zero-cost pre-gates and two routes: a mention
ADDED by the edit takes the addressed wake path (Slack fires no app_mention for edits); every
other channel edit goes through the participation engine's full typo-vs-meaning judgment.

These exercise the real decision code in SlackMessageEventsMixin + ParticipationEngine with
stubbed I/O, asserting both the anti-annoyance guarantees (unfurl / identical / own-message /
stale / flag-off cost nothing) and the two routing branches.

Commit 6 changed WHO owns the edit context, not whether it survives. It is keyed by
(channel, ts, MARKER) and rides the gate's typed SourceMessage instead of being spliced into a
prose "[EDIT]" block, because an edit keeps its ORIGINAL Slack timestamp: (channel, ts) alone could
not tell the edit's own re-evaluation from the stale original attempt it superseded, and whichever
ran first popped the context. The original could therefore arrive holding the edit's before/after
text, conclude it WAS the edit, and in doing so skip the supersession check meant to silence it.
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from base_client import Message
from config import config
from message_processor.participation import ParticipationEngine
from slack_client.event_handlers.message_events import SlackMessageEventsMixin
from slack_client.utilities import SlackUtilitiesMixin


class _Bot(SlackMessageEventsMixin, SlackUtilitiesMixin):
    def log_debug(self, *a, **k):
        pass

    def log_info(self, *a, **k):
        pass

    def log_warning(self, *a, **k):
        pass

    def log_error(self, *a, **k):
        pass


async def _fake_event_to_message(event, client):
    attachments = []
    for f in event.get("files", []) or []:
        mimetype = f.get("mimetype", "")
        attachments.append({
            "type": "image" if mimetype.startswith("image/") else "file",
            "url": f.get("url_private"),
            "id": f.get("id"),
            "name": f.get("name"),
            "mimetype": mimetype,
        })
    return Message(
        text=event.get("text", ""),
        user_id=event.get("user"),
        channel_id=event.get("channel"),
        thread_id=event.get("thread_ts") or event.get("ts"),
        attachments=attachments,
        metadata={"ts": event.get("ts")},
    )


def _make_bot():
    bot = _Bot.__new__(_Bot)
    bot.bot_user_id = "UBOT"
    bot.bot_id = "BBOT"
    bot.app_id = None
    bot.message_handler = AsyncMock()
    bot.app = MagicMock()
    bot.app.client = MagicMock()
    bot.channel_pulse = MagicMock()
    bot.channel_pulse.ensure_backfill = AsyncMock()
    bot._event_to_message = _fake_event_to_message
    bot._get_channel_settings = AsyncMock(return_value=None)
    bot._thread_participation = AsyncMock(return_value=(False, 1, 0))
    return bot


def _recent_ts(age_seconds: float = 5.0) -> str:
    return f"{time.time() - age_seconds:.6f}"


def _changed(*, old="please review", new="please review the numbers", user="UHUMAN",
             channel="C1", msg_ts=None, thread_ts=None, edited_ts=None, bot_id=None):
    """A message_changed event. `old`/`new` are previous/current text; msg_ts is the ORIGINAL ts."""
    msg_ts = msg_ts or _recent_ts()
    inner = {"type": "message", "user": user, "ts": msg_ts, "text": new,
             "edited": {"user": user, "ts": edited_ts or _recent_ts(1.0)}}
    prev = {"type": "message", "user": user, "ts": msg_ts, "text": old}
    if thread_ts:
        inner["thread_ts"] = thread_ts
        prev["thread_ts"] = thread_ts
    if bot_id:
        inner["bot_id"] = bot_id
        inner["subtype"] = "bot_message"
        prev["bot_id"] = bot_id
    return {"subtype": "message_changed", "channel": channel, "ts": _recent_ts(0.5),
            "message": inner, "previous_message": prev}


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setattr(config, "enable_edit_triggered_replies", True, raising=False)
    monkeypatch.setattr(config, "enable_channel_listening", True, raising=False)
    monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
    monkeypatch.setattr(config, "edit_reply_window_minutes", 60, raising=False)
    monkeypatch.setattr(config, "participation_debounce_seconds", 0.02, raising=False)
    monkeypatch.setattr(config, "bot_name_aliases", ["ChatGPT"], raising=False)


def _capture_schedule(bot):
    """Replace the runner + scheduler so _maybe_edit_triggered_reply's decision is observable
    WITHOUT spawning a task. Returns the list that receives one entry per scheduled reply."""
    scheduled = []
    bot._run_edit_triggered_reply = MagicMock(return_value="CORO")
    bot._schedule_edit_reply = lambda coro: scheduled.append(coro)
    return scheduled


# ----------------------------------------------------------------- zero-cost pre-gate matrix

def test_flag_off_does_nothing(monkeypatch):
    monkeypatch.setattr(config, "enable_edit_triggered_replies", False, raising=False)
    bot = _make_bot()
    scheduled = _capture_schedule(bot)
    bot._maybe_edit_triggered_reply(_changed(), bot.app.client)
    assert scheduled == []
    bot._run_edit_triggered_reply.assert_not_called()


def test_identical_normalized_text_edit_costs_nothing(flag_on):
    bot = _make_bot()
    scheduled = _capture_schedule(bot)
    # An unfurl / attachment change fires message_changed with byte-identical text (bar whitespace).
    bot._maybe_edit_triggered_reply(
        _changed(old="hello   world", new="hello world"), bot.app.client)
    assert scheduled == []


def test_own_message_edit_never_triggers(flag_on):
    bot = _make_bot()
    scheduled = _capture_schedule(bot)
    # The bot's own streamed chat.update edits arrive as subtype bot_message / own.
    bot._maybe_edit_triggered_reply(
        _changed(user="UBOT", old="thinking", new="here is the answer"), bot.app.client)
    assert scheduled == []


def test_other_bot_message_edit_never_triggers(flag_on):
    bot = _make_bot()
    scheduled = _capture_schedule(bot)
    bot._maybe_edit_triggered_reply(
        _changed(user="UOTHER", bot_id="B999", old="build queued", new="build passed"),
        bot.app.client)
    assert scheduled == []


def test_edit_older_than_window_never_triggers(flag_on):
    bot = _make_bot()
    scheduled = _capture_schedule(bot)
    old_ts = f"{time.time() - 3600 * 3:.6f}"  # 3 hours old, window is 60 min
    bot._maybe_edit_triggered_reply(
        _changed(msg_ts=old_ts, old="q", new="a much longer question now"), bot.app.client)
    assert scheduled == []


def test_ambient_edit_requires_channel_listening(flag_on, monkeypatch):
    monkeypatch.setattr(config, "enable_channel_listening", False, raising=False)
    bot = _make_bot()
    scheduled = _capture_schedule(bot)
    # No mention, not a DM → engine branch, which a new message wouldn't reach with listening off.
    bot._maybe_edit_triggered_reply(_changed(), bot.app.client)
    assert scheduled == []


def test_meaningful_channel_edit_schedules(flag_on):
    bot = _make_bot()
    scheduled = _capture_schedule(bot)
    bot._maybe_edit_triggered_reply(_changed(), bot.app.client)
    assert len(scheduled) == 1
    args = bot._run_edit_triggered_reply.call_args.args
    # (event, client, channel_id, msg_ts, old_text, new_text, is_dm, mention_added)
    assert args[2] == "C1"
    assert args[6] is False  # is_dm
    assert args[7] is False  # mention_added


def test_mention_added_by_edit_schedules_even_with_listening_off(flag_on, monkeypatch):
    monkeypatch.setattr(config, "enable_channel_listening", False, raising=False)
    bot = _make_bot()
    scheduled = _capture_schedule(bot)
    # Forgot to tag the bot; the edit ADDS the @mention. app_mention never fires for an edit.
    bot._maybe_edit_triggered_reply(
        _changed(old="what's the weather", new="<@UBOT> what's the weather"), bot.app.client)
    assert len(scheduled) == 1
    assert bot._run_edit_triggered_reply.call_args.args[7] is True  # mention_added


def test_mention_present_before_and_after_is_not_mention_added(flag_on):
    bot = _make_bot()
    scheduled = _capture_schedule(bot)
    bot._maybe_edit_triggered_reply(
        _changed(old="<@UBOT> hi", new="<@UBOT> what's the weather"), bot.app.client)
    assert len(scheduled) == 1
    # Mention was already there → engine branch (typo-vs-meaning), NOT the addressed shortcut.
    assert bot._run_edit_triggered_reply.call_args.args[7] is False


def test_dm_edit_schedules_as_addressed(flag_on, monkeypatch):
    monkeypatch.setattr(config, "enable_channel_listening", False, raising=False)
    bot = _make_bot()
    scheduled = _capture_schedule(bot)
    bot._maybe_edit_triggered_reply(
        _changed(channel="D1", old="hi", new="what's the weather"), bot.app.client)
    assert len(scheduled) == 1
    assert bot._run_edit_triggered_reply.call_args.args[6] is True  # is_dm


# ----------------------------------------------------------------- routing (both branches)

@pytest.mark.asyncio
async def test_mention_added_routes_to_addressed_path(flag_on):
    bot = _make_bot()
    bot._handle_slack_message = AsyncMock()
    bot._dispatch_edit_to_engine = AsyncMock()
    event = _changed(old="what's up", new="<@UBOT> what's up")
    await bot._run_edit_triggered_reply(
        event, bot.app.client, "C1", event["message"]["ts"],
        "what's up", "<@UBOT> what's up", is_dm=False, mention_added=True)
    bot._dispatch_edit_to_engine.assert_not_called()
    bot._handle_slack_message.assert_awaited_once()
    synth, _client = bot._handle_slack_message.await_args.args[0], bot._handle_slack_message.await_args.args[1]
    assert bot._handle_slack_message.await_args.kwargs["wake_source"] == "app_mention"
    # Synthetic FRESH event: no message_changed subtype, edited text at the ORIGINAL ts.
    assert "subtype" not in synth
    assert synth["ts"] == event["message"]["ts"]
    assert synth["text"] == "<@UBOT> what's up"
    assert synth["channel"] == "C1"


@pytest.mark.asyncio
async def test_dm_edit_routes_to_dm_addressed_path(flag_on):
    bot = _make_bot()
    bot._handle_slack_message = AsyncMock()
    event = _changed(channel="D1", old="hi", new="what's the weather")
    await bot._run_edit_triggered_reply(
        event, bot.app.client, "D1", event["message"]["ts"],
        "hi", "what's the weather", is_dm=True, mention_added=False)
    bot._handle_slack_message.assert_awaited_once()
    assert bot._handle_slack_message.await_args.kwargs["wake_source"] == "dm"


@pytest.mark.asyncio
async def test_ambient_edit_routes_to_engine(flag_on):
    bot = _make_bot()
    bot._handle_slack_message = AsyncMock()
    bot._dispatch_edit_to_engine = AsyncMock()
    event = _changed()
    await bot._run_edit_triggered_reply(
        event, bot.app.client, "C1", event["message"]["ts"],
        "please review", "please review the numbers", is_dm=False, mention_added=False)
    bot._handle_slack_message.assert_not_called()
    bot._dispatch_edit_to_engine.assert_awaited_once()


# ----------------------------------------------------------------- edit-burst debounce collapse

@pytest.mark.asyncio
async def test_edit_burst_on_one_message_collapses(flag_on):
    bot = _make_bot()
    dispatched = []

    async def _record_engine(client, synthetic, channel_id, msg_ts, old_text, new_text,
                             marker=None):
        dispatched.append(new_text)

    bot._dispatch_edit_to_engine = _record_engine
    msg_ts = _recent_ts()
    # Two rapid edits of the SAME message (same msg_ts, DIFFERENT edit markers).
    e1 = _changed(msg_ts=msg_ts, old="draft", new="draft v1", edited_ts=_recent_ts(2.0))
    e2 = _changed(msg_ts=msg_ts, old="draft", new="draft final", edited_ts=_recent_ts(0.1))
    t1 = asyncio.create_task(bot._run_edit_triggered_reply(
        e1, bot.app.client, "C1", msg_ts, "draft", "draft v1", False, False))
    await asyncio.sleep(0.005)
    t2 = asyncio.create_task(bot._run_edit_triggered_reply(
        e2, bot.app.client, "C1", msg_ts, "draft", "draft final", False, False))
    await asyncio.gather(t1, t2)
    # Only the NEWEST edit in the burst survives.
    assert dispatched == ["draft final"]


# ----------------------------------------------------------------- engine dispatch details

@pytest.mark.asyncio
async def test_engine_dispatch_stashes_context_and_marks_check(flag_on):
    bot = _make_bot()
    bot._get_channel_settings = AsyncMock(return_value={"participation_level": "on"})
    bot._thread_participation = AsyncMock(return_value=(True, 1, 0))  # bot already in thread
    msg_ts = _recent_ts()
    synthetic = {"channel": "C1", "ts": msg_ts, "user": "UHUMAN", "text": "please review the numbers"}
    await bot._dispatch_edit_to_engine(
        bot.app.client, synthetic, "C1", msg_ts, "please review", "please review the numbers",
        marker="E1")
    bot.message_handler.assert_awaited_once()
    msg = bot.message_handler.await_args.args[0]
    assert msg.metadata["gate_required"] is True
    assert msg.metadata["participation_level"] == "on"
    # Stashed for evaluate() to read, keyed (channel, ts, MARKER) — the marker is what makes the
    # context THIS attempt's rather than "whoever pops it first".
    ctx = bot._edit_reply_ctx_map[f"C1|{msg_ts}|E1"]
    assert ctx["old_text"] == "please review"
    assert ctx["already_replied"] is True
    # And the dispatched message carries the same marker, so only its attempt can claim the context.
    assert msg.metadata["edit_reply_marker"] == "E1"


@pytest.mark.asyncio
async def test_engine_dispatch_silent_when_mentions_only_no_name(flag_on):
    bot = _make_bot()
    bot._get_channel_settings = AsyncMock(return_value={"participation_level": "mentions_only"})
    msg_ts = _recent_ts()
    synthetic = {"channel": "C1", "ts": msg_ts, "user": "UHUMAN", "text": "just some chatter"}
    await bot._dispatch_edit_to_engine(
        bot.app.client, synthetic, "C1", msg_ts, "just chatter", "just some chatter")
    bot.message_handler.assert_not_called()  # a new ambient message wouldn't respond here either


@pytest.mark.asyncio
async def test_engine_dispatch_silent_when_participation_off(flag_on):
    bot = _make_bot()
    bot._get_channel_settings = AsyncMock(return_value={"participation_level": "off"})
    msg_ts = _recent_ts()
    synthetic = {"channel": "C1", "ts": msg_ts, "user": "UHUMAN", "text": "please review the numbers"}
    await bot._dispatch_edit_to_engine(
        bot.app.client, synthetic, "C1", msg_ts, "please review", "please review the numbers")
    bot.message_handler.assert_not_called()


# ----------------------------------------------------------------- engine sees the edit context

class _RecordingClient:
    """A facade carrying a stashed edit context (as the real SlackBot would) + the wake classifier.

    In production the facade and the OpenAI client are two objects; the engine takes the classifier
    from its constructor and the edit store from the `client=` it is handed. This harness plays both
    so one object can assert on both halves."""

    def __init__(self, wake=True):
        self._wake = wake
        self.calls = 0
        self.last_sources = ()
        self._edit_reply_ctx_map = {}

    async def classify_wake(self, *, sources, channel_steering_text=None):
        self.calls += 1
        self.last_sources = tuple(sources)
        return self._wake


@pytest.mark.asyncio
async def test_the_edit_rides_the_source_record_the_gate_judges(monkeypatch):
    """Re-baselined: the edit's before/after text is INTRINSIC source data now, not a prose block.

    It used to be spliced into the classifier's text as an "[EDIT] …" note, which is why this test
    once asserted on the classifier's prompt string. The gate takes typed SourceMessage records, so
    the edit travels as `source.edit` and the rendering is the renderer's business (asserted in
    test_wake_classifier.py). What has to hold HERE is that the engine finds the stashed context,
    attaches it to the record it judges, and consumes it."""
    monkeypatch.setattr(config, "participation_debounce_seconds", 0.0, raising=False)
    client = _RecordingClient(wake=True)
    client._edit_reply_ctx_map["C1|100.1|E1"] = {
        "old_text": "please review", "new_text": "please review the Q3 numbers",
        "already_replied": True,
    }
    engine = ParticipationEngine(client)
    evaluation = await engine.evaluate(
        channel_id="C1", ts="100.1", text="please review the Q3 numbers",
        client=client, edit_marker="E1")
    assert evaluation.decision.wake is True
    assert client.calls == 1
    source = client.last_sources[-1]
    assert source.edit == {"old_text": "please review",
                           "new_text": "please review the Q3 numbers",
                           "already_replied": True}
    # The message's own text is untouched — the edit is metadata about it, not a rewrite of it.
    assert source.text == "please review the Q3 numbers"
    # Consumed (popped): a second evaluation of the same edit falls back to a plain judgment rather
    # than replaying stale before-text.
    assert "C1|100.1|E1" not in client._edit_reply_ctx_map


@pytest.mark.asyncio
async def test_the_original_attempt_cannot_consume_the_edits_context(monkeypatch):
    """The ownership bug this keying exists to prevent, asserted directly.

    The stale ORIGINAL attempt carries no marker. Under the old (channel, ts) key it could pop the
    edit's context, read itself as the edit — which also suppressed the supersession check meant to
    silence it — and answer twice. It must get None, and must leave the context sitting there for
    the attempt that owns it."""
    monkeypatch.setattr(config, "participation_debounce_seconds", 0.0, raising=False)
    client = _RecordingClient(wake=True)
    client._edit_reply_ctx_map["C1|100.1|E1"] = {
        "old_text": "review", "new_text": "review the Q3 numbers", "already_replied": False}

    # The direct helper: no marker at all, and a WRONG marker, both get nothing.
    assert ParticipationEngine._take_edit_context(client, "C1", "100.1") is None
    assert ParticipationEngine._take_edit_context(client, "C1", "100.1", "E0") is None
    assert "C1|100.1|E1" in client._edit_reply_ctx_map      # still the edit's to claim

    # And end to end: the marker-less attempt judges a plain message (no edit on the record)...
    engine = ParticipationEngine(client)
    await engine.evaluate(channel_id="C1", ts="100.1", text="review the Q3 numbers", client=client)
    assert client.last_sources[-1].edit is None
    # ...while the edit's own attempt, carrying the marker, still gets its context.
    await engine.evaluate(channel_id="C1", ts="100.1", text="review the Q3 numbers",
                          client=client, edit_marker="E1")
    assert client.last_sources[-1].edit["old_text"] == "review"


@pytest.mark.asyncio
async def test_typo_edit_one_eval_and_no_wake(monkeypatch):
    # A typo fix is exactly the case the gate is allowed to sleep through, and it costs ONE call.
    monkeypatch.setattr(config, "participation_debounce_seconds", 0.0, raising=False)
    client = _RecordingClient(wake=False)
    client._edit_reply_ctx_map["C1|100.1|E1"] = {
        "old_text": "the wether is nice", "new_text": "the weather is nice",
        "already_replied": False,
    }
    engine = ParticipationEngine(client)
    evaluation = await engine.evaluate(
        channel_id="C1", ts="100.1", text="the weather is nice", client=client, edit_marker="E1")
    assert client.calls == 1
    assert evaluation.decision.wake is False
    assert evaluation.decline_cause is None      # a real decision, not a decline


@pytest.mark.asyncio
async def test_ordinary_message_untouched_by_edit_plumbing(monkeypatch):
    """A non-edit message has no stashed context and no marker, so its source record is a plain
    message — nothing about the ordinary path changes."""
    monkeypatch.setattr(config, "participation_debounce_seconds", 0.0, raising=False)
    client = _RecordingClient()
    engine = ParticipationEngine(client)
    await engine.evaluate(channel_id="C1", ts="9.9", text="hello team", client=client)
    source = client.last_sources[-1]
    assert source.text == "hello team"
    assert source.edit is None


@pytest.mark.asyncio
async def test_a_mock_client_yields_no_edit_context_at_all(monkeypatch):
    """A bare MagicMock answers every attribute with another truthy mock, so a truthiness test
    on the store handed back a MOCK edit context: the gate's source record silently grew an edit
    block and edit supersession was suppressed. Whole suites then asserted things about a prompt
    production never sends. The store has to be an actual mapping."""
    from unittest.mock import MagicMock

    monkeypatch.setattr(config, "participation_debounce_seconds", 0.0, raising=False)
    mock_client = MagicMock()
    # Even WITH a marker — the marker is necessary, not sufficient.
    assert ParticipationEngine._take_edit_context(mock_client, "C1", "9.9", "E1") is None

    recorder = _RecordingClient()
    # The classifier lives on the openai_client; the FACADE is the mock, exactly as in the
    # fixtures that carry one.
    engine = ParticipationEngine(recorder)
    await engine.evaluate(channel_id="C1", ts="9.9", text="hello team", client=mock_client,
                          edit_marker="E1")
    assert recorder.last_sources[-1].text == "hello team"   # unpolluted
    assert recorder.last_sources[-1].edit is None


# ----------------------------------------------------------------- what the model is told

def test_the_edit_rubric_lives_in_the_rendered_source_not_the_prompt():
    """Re-baselined: the typo-vs-meaning rubric moved from prompt prose to the rendered message.

    The rich prompt carried a paragraph about "[EDIT]" notes, typos and corrections; the ten-line
    binary prompt carries no rubric at all, because a rubric is per-message and the prompt is not.
    The instruction now rides the source block itself, which is also the only place that KNOWS
    whether the assistant already replied."""
    from openai_client.api.responses import _render_wake_source
    from message_processor.participation import SourceMessage

    replied = _render_wake_source(SourceMessage(
        ts="100.1", text="please review the Q3 numbers", sender_name="Peter",
        sender_type="human", edit={"old_text": "please review", "already_replied": True}),
        index=0, total=1)
    assert "was EDITED" in replied
    assert '"please review"' in replied           # the before-text, so a typo fix is visible as one
    assert "already replied" in replied
    assert "only if the edit changes what is being asked" in replied

    unreplied = _render_wake_source(SourceMessage(
        ts="100.1", text="please review the Q3 numbers", sender_name="Peter",
        sender_type="human", edit={"old_text": "please review", "already_replied": False}),
        index=0, total=1)
    assert "has not replied to it yet" in unreplied

    # An edit that ADDED text where there was none must still say so rather than quoting nothing.
    from_nothing = _render_wake_source(SourceMessage(
        ts="100.1", text="now with a question", sender_name="Peter", sender_type="human",
        edit={"old_text": "", "already_replied": False}), index=0, total=1)
    assert "it had no text before" in from_nothing


# --------------------------------------------------- Bug A: engine supersession (double-answer)

@pytest.mark.asyncio
async def test_edit_supersedes_original_in_flight_evaluation(monkeypatch):
    """The ORIGINAL (pre-edit) message is mid-debounce when an edit supersedes it: its evaluation
    must return None (no stale respond), exactly as a newer burst arrival would cause."""
    monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
    client = _RecordingClient(wake=True)
    engine = ParticipationEngine(client)
    # Kick off the original's evaluation, then supersede it mid-debounce (as the edit path does).
    task = asyncio.create_task(engine.evaluate(
        channel_id="C1", ts="100.1", text="does anyone remember the WAL default?",
        sender_id="UHUMAN", client=client))
    await asyncio.sleep(0.005)
    engine.supersede("C1", "100.1", thread_root=None, sender_id="UHUMAN")
    evaluation = await task
    assert evaluation.decision is None           # superseded — no second answer
    assert evaluation.decline_cause == "edit_superseded"
    assert client.calls == 0          # the classifier was never even consulted


@pytest.mark.asyncio
async def test_edits_own_reevaluation_survives_supersession(monkeypatch):
    """The edit's OWN fresh evaluation carries edit context and must NOT be dropped by the
    supersession mark (only the context-free stale original is)."""
    monkeypatch.setattr(config, "participation_debounce_seconds", 0.0, raising=False)
    client = _RecordingClient(wake=True)
    client._edit_reply_ctx_map["C1|100.1|E1"] = {
        "old_text": "review", "new_text": "review the Q3 numbers", "already_replied": False}
    engine = ParticipationEngine(client)
    engine.supersede("C1", "100.1", thread_root=None, sender_id="UHUMAN")
    evaluation = await engine.evaluate(
        channel_id="C1", ts="100.1", text="review the Q3 numbers",
        sender_id="UHUMAN", client=client, edit_marker="E1")
    assert evaluation.decision.wake is True   # the edit's own eval answers
    assert client.calls == 1
    # Structural, not a coincidence of pop ordering: the exemption is "this attempt HAS edit
    # context", and the context is unreachable without the marker.
    assert evaluation.decline_cause is None


def test_maybe_edit_calls_supersede(flag_on):
    """_maybe_edit_triggered_reply supersedes the original's engine evaluation on the facade's
    wired engine, keyed by the message's (channel, ts, sender)."""
    bot = _make_bot()
    _capture_schedule(bot)
    engine = MagicMock()
    bot.processor = MagicMock()
    bot.processor.participation_engine = engine
    bot._maybe_edit_triggered_reply(
        _changed(channel="C1", user="UHUMAN", old="q", new="a longer question now"),
        bot.app.client)
    engine.supersede.assert_called_once()
    kwargs = engine.supersede.call_args.kwargs
    assert engine.supersede.call_args.args[0] == "C1"
    assert kwargs.get("sender_id") == "UHUMAN"


# --------------------------------------------------- Bug A: mention-added duplicate suppression

@pytest.mark.asyncio
async def test_mention_added_skips_synthetic_when_app_mention_already_seen(flag_on):
    """When Slack already delivered a genuine app_mention for the edited ts, the synthetic
    addressed dispatch is a duplicate and is skipped."""
    bot = _make_bot()
    bot._handle_slack_message = AsyncMock()
    bot._note_app_mention_seen("C1", "100.1")
    await bot._run_edit_triggered_reply(
        _changed(channel="C1", msg_ts="100.1", old="what's up", new="<@UBOT> what's up"),
        bot.app.client, "C1", "100.1", "what's up", "<@UBOT> what's up",
        is_dm=False, mention_added=True)
    bot._handle_slack_message.assert_not_called()      # Slack's app_mention covers it


@pytest.mark.asyncio
async def test_mention_added_dispatches_when_no_app_mention_seen(flag_on):
    """Fallback preserved: with no genuine app_mention seen, the synthetic dispatch still fires."""
    bot = _make_bot()
    bot._handle_slack_message = AsyncMock()
    await bot._run_edit_triggered_reply(
        _changed(channel="C1", msg_ts="100.1", old="what's up", new="<@UBOT> what's up"),
        bot.app.client, "C1", "100.1", "what's up", "<@UBOT> what's up",
        is_dm=False, mention_added=True)
    bot._handle_slack_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_engine_edit_dispatch_registers_marker(flag_on):
    """The engine-branch edit registers its ts with the surviving marker AND stamps the dispatched
    message, so the queue drain keeps the edit's own dispatch."""
    bot = _make_bot()
    bot._get_channel_settings = AsyncMock(return_value={"participation_level": "on"})
    await bot._run_edit_triggered_reply(
        _changed(channel="C1", msg_ts="100.1", old="review", new="review the Q3 numbers"),
        bot.app.client, "C1", "100.1", "review", "review the Q3 numbers",
        is_dm=False, mention_added=False)
    marker = bot.edit_dispatch_marker("C1", "100.1")
    assert marker is not None
    dispatched = bot.message_handler.await_args.args[0]
    assert dispatched.metadata.get("edit_reply_marker") == marker
