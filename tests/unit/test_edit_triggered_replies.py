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
"""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from base_client import Message
from config import config
from message_processor.participation import ParticipationEngine
from prompts import PARTICIPATION_SYSTEM_PROMPT
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
    bot._get_channel_settings = AsyncMock(return_value={"participation_level": "judicious"})
    bot._thread_participation = AsyncMock(return_value=(True, 1, 0))  # bot already in thread
    msg_ts = _recent_ts()
    synthetic = {"channel": "C1", "ts": msg_ts, "user": "UHUMAN", "text": "please review the numbers"}
    await bot._dispatch_edit_to_engine(
        bot.app.client, synthetic, "C1", msg_ts, "please review", "please review the numbers")
    bot.message_handler.assert_awaited_once()
    msg = bot.message_handler.await_args.args[0]
    assert msg.metadata["participation_check"] is True
    assert msg.metadata["participation_level"] == "judicious"
    # Edit context stashed on the facade for evaluate() to read, keyed by (channel, ts).
    ctx = bot._edit_reply_ctx_map[f"C1|{msg_ts}"]
    assert ctx["old_text"] == "please review"
    assert ctx["already_replied"] is True


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
    """A facade carrying a stashed edit context (as the real SlackBot would) + a classifier."""

    def __init__(self, verdict):
        self._verdict = verdict
        self.calls = 0
        self.last_text = None
        self._edit_reply_ctx_map = {}

    async def classify_participation(self, text, signals=None):
        self.calls += 1
        self.last_text = text
        return self._verdict


@pytest.mark.asyncio
async def test_engine_folds_edit_block_into_classifier_prompt(monkeypatch):
    monkeypatch.setattr(config, "participation_debounce_seconds", 0.0, raising=False)
    client = _RecordingClient({"action": "respond"})
    client._edit_reply_ctx_map["C1|100.1"] = {
        "old_text": "please review", "new_text": "please review the Q3 numbers",
        "already_replied": True,
    }
    engine = ParticipationEngine(client)
    verdict = await engine.evaluate(
        channel_id="C1", ts="100.1", text="please review the Q3 numbers", client=client)
    assert verdict.action == "respond"
    assert client.calls == 1
    # The classifier saw the [EDIT] block with the old text + already-replied note; the verdict's
    # own text stays untouched for the responder.
    assert "[EDIT]" in client.last_text
    assert "please review" in client.last_text
    assert "already replied" in client.last_text
    # Consumed: a re-eval of the same ts falls back to a plain judgment.
    assert "C1|100.1" not in client._edit_reply_ctx_map


@pytest.mark.asyncio
async def test_typo_edit_one_eval_ignore_stays_silent(monkeypatch):
    monkeypatch.setattr(config, "participation_debounce_seconds", 0.0, raising=False)
    client = _RecordingClient({"action": "ignore"})
    client._edit_reply_ctx_map["C1|100.1"] = {
        "old_text": "the wether is nice", "new_text": "the weather is nice",
        "already_replied": False,
    }
    engine = ParticipationEngine(client)
    verdict = await engine.evaluate(
        channel_id="C1", ts="100.1", text="the weather is nice", client=client)
    assert client.calls == 1        # at most ONE engine evaluation
    assert verdict.action == "ignore"  # a typo fix stays silent


@pytest.mark.asyncio
async def test_ordinary_message_untouched_by_edit_plumbing(monkeypatch):
    """A non-edit message has no stashed context, so the classifier text is byte-for-byte the
    message text — nothing about the ordinary path changes."""
    monkeypatch.setattr(config, "participation_debounce_seconds", 0.0, raising=False)
    client = _RecordingClient({"action": "ignore"})
    engine = ParticipationEngine(client)
    await engine.evaluate(channel_id="C1", ts="9.9", text="hello team", client=client)
    assert client.last_text == "hello team"


# ----------------------------------------------------------------- prompt content

def test_participation_prompt_carries_edit_rubric():
    assert "[EDIT]" in PARTICIPATION_SYSTEM_PROMPT
    lower = PARTICIPATION_SYSTEM_PROMPT.lower()
    assert "edited message" in lower
    assert "typo" in lower
    assert "correction" in lower


# --------------------------------------------------- Bug A: engine supersession (double-answer)

@pytest.mark.asyncio
async def test_edit_supersedes_original_in_flight_evaluation(monkeypatch):
    """The ORIGINAL (pre-edit) message is mid-debounce when an edit supersedes it: its evaluation
    must return None (no stale respond), exactly as a newer burst arrival would cause."""
    monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
    client = _RecordingClient({"action": "respond"})
    engine = ParticipationEngine(client)
    # Kick off the original's evaluation, then supersede it mid-debounce (as the edit path does).
    task = asyncio.create_task(engine.evaluate(
        channel_id="C1", ts="100.1", text="does anyone remember the WAL default?",
        sender_id="UHUMAN", client=client))
    await asyncio.sleep(0.005)
    engine.supersede("C1", "100.1", thread_root=None, sender_id="UHUMAN")
    verdict = await task
    assert verdict is None            # superseded — no second answer
    assert client.calls == 0          # the classifier was never even consulted


@pytest.mark.asyncio
async def test_edits_own_reevaluation_survives_supersession(monkeypatch):
    """The edit's OWN fresh evaluation carries edit context and must NOT be dropped by the
    supersession mark (only the context-free stale original is)."""
    monkeypatch.setattr(config, "participation_debounce_seconds", 0.0, raising=False)
    client = _RecordingClient({"action": "respond"})
    client._edit_reply_ctx_map["C1|100.1"] = {
        "old_text": "review", "new_text": "review the Q3 numbers", "already_replied": False}
    engine = ParticipationEngine(client)
    engine.supersede("C1", "100.1", thread_root=None, sender_id="UHUMAN")
    verdict = await engine.evaluate(
        channel_id="C1", ts="100.1", text="review the Q3 numbers",
        sender_id="UHUMAN", client=client)
    assert verdict.action == "respond"   # the edit's own eval answers
    assert client.calls == 1


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
    bot._get_channel_settings = AsyncMock(return_value={"participation_level": "judicious"})
    await bot._run_edit_triggered_reply(
        _changed(channel="C1", msg_ts="100.1", old="review", new="review the Q3 numbers"),
        bot.app.client, "C1", "100.1", "review", "review the Q3 numbers",
        is_dm=False, mention_added=False)
    marker = bot.edit_dispatch_marker("C1", "100.1")
    assert marker is not None
    dispatched = bot.message_handler.await_args.args[0]
    assert dispatched.metadata.get("edit_reply_marker") == marker
