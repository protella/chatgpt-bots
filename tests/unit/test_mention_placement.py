"""AREA B — @mention/DM replies honor top-level placement.

Two layers:

1. message_events (_handle_slack_message): the MENTION path now resolves reply_in_channel like
   the channel path (row explicit True/False wins; None/absent → config.reply_in_channel_default)
   and stamps message.metadata["reply_in_channel"] = True when truthy — inside the non-D block,
   before the other_bot dispatch. DMs are untouched; the self-sender loop guard still short-circuits.

2. main.py (handle_message): with reply_in_channel now settable on a mention turn, place_in_channel
   can be True → the reply posts top-level, BUT code-interpreter artifacts (B2) and rescued sandbox
   images (B2) STILL thread off message.thread_id, never top-level; and an engine "thread" verdict
   still overrides the setting.

Real decision code, stubbed I/O — no network/DB.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from base_client import Message
from config import config
from message_processor.participation import ParticipationVerdict
from slack_client.event_handlers.message_events import SlackMessageEventsMixin


# ============================================================ layer 1: the stamp

class _Bot(SlackMessageEventsMixin):
    def log_debug(self, *a, **k):
        pass

    def log_info(self, *a, **k):
        pass

    def log_warning(self, *a, **k):
        pass

    def log_error(self, *a, **k):
        pass


def _make_bot(cs, sender_type="human"):
    bot = _Bot.__new__(_Bot)
    bot.bot_user_id = "UBOT"
    bot.bot_id = "BBOT"
    bot.app_id = None
    bot.message_handler = AsyncMock()
    bot.app = MagicMock()
    bot.app.client = MagicMock()
    bot.channel_pulse = None  # skip the pulse feed/backfill branch
    bot.db = MagicMock()
    # Existing user with completed settings → no onboarding modal; message dispatches.
    bot.db.get_user_preferences_async = AsyncMock(return_value={"settings_completed": True})
    bot._get_channel_settings = AsyncMock(return_value=cs)
    bot.classify_sender = lambda e: sender_type
    bot._post_settings_button_if_new_thread = AsyncMock()
    bot._maybe_set_assistant_thread_title = AsyncMock()

    async def _e2m(event, client):
        return Message(
            text=event.get("text", ""), user_id=event.get("user"),
            channel_id=event.get("channel"),
            thread_id=event.get("thread_ts") or event.get("ts"),
            attachments=[], metadata={"ts": event.get("ts")})

    bot._event_to_message = _e2m
    return bot


def _evt(**kw):
    e = {"channel": "C1", "ts": "100.1", "user": "UHUMAN", "text": "<@UBOT> hi"}
    e.update(kw)
    return e


@pytest.fixture(autouse=True)
def _judicious(monkeypatch):
    # Keep participation off the "off" rail (the app_mention path drops @mentions when off).
    monkeypatch.setattr(config, "channel_response_mode", "auto_respond", raising=False)
    monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
    monkeypatch.setattr(config, "enable_channel_listening", False, raising=False)


def _dispatched(bot):
    return bot.message_handler.await_args.args[0]


@pytest.mark.asyncio
async def test_mention_row_true_stamps_true(monkeypatch):
    monkeypatch.setattr(config, "reply_in_channel_default", False, raising=False)  # default OFF…
    bot = _make_bot({"response_mode": "auto_respond", "reply_in_channel": True})   # …row wins
    await bot._handle_slack_message(_evt(), bot.app.client, wake_source="app_mention")
    assert _dispatched(bot).metadata.get("reply_in_channel") is True


@pytest.mark.asyncio
async def test_mention_row_false_never_stamps(monkeypatch):
    monkeypatch.setattr(config, "reply_in_channel_default", True, raising=False)   # default ON…
    bot = _make_bot({"response_mode": "auto_respond", "reply_in_channel": False})  # …explicit False wins
    await bot._handle_slack_message(_evt(), bot.app.client, wake_source="app_mention")
    assert _dispatched(bot).metadata.get("reply_in_channel") is not True


@pytest.mark.asyncio
async def test_mention_row_null_inherits_default_true(monkeypatch):
    monkeypatch.setattr(config, "reply_in_channel_default", True, raising=False)
    bot = _make_bot({"response_mode": "auto_respond", "reply_in_channel": None})
    await bot._handle_slack_message(_evt(), bot.app.client, wake_source="app_mention")
    assert _dispatched(bot).metadata.get("reply_in_channel") is True


@pytest.mark.asyncio
async def test_mention_no_row_inherits_default(monkeypatch):
    monkeypatch.setattr(config, "reply_in_channel_default", True, raising=False)
    bot = _make_bot(None)   # no channel settings row at all
    await bot._handle_slack_message(_evt(), bot.app.client, wake_source="app_mention")
    assert _dispatched(bot).metadata.get("reply_in_channel") is True


@pytest.mark.asyncio
async def test_mention_no_row_default_off_stays_unstamped(monkeypatch):
    monkeypatch.setattr(config, "reply_in_channel_default", False, raising=False)
    bot = _make_bot(None)
    await bot._handle_slack_message(_evt(), bot.app.client, wake_source="app_mention")
    assert _dispatched(bot).metadata.get("reply_in_channel") is not True


@pytest.mark.asyncio
async def test_dm_mention_never_stamps(monkeypatch):
    # DM path skips the whole non-D channel block → reply_in_channel is never resolved/stamped.
    monkeypatch.setattr(config, "reply_in_channel_default", True, raising=False)
    bot = _make_bot({"reply_in_channel": True})  # even a truthy row can't reach a DM
    await bot._handle_slack_message(
        _evt(channel="D123", text="hi"), bot.app.client, wake_source="dm")
    assert _dispatched(bot).metadata.get("reply_in_channel") is not True


@pytest.mark.asyncio
async def test_other_bot_top_level_mention_stamps_and_dispatches(monkeypatch):
    # NEW: a peer bot's top-level @mention gets the stamp too, THEN dispatches early (before
    # onboarding). The sender/self guards are retained (see the self-sender test below).
    monkeypatch.setattr(config, "reply_in_channel_default", True, raising=False)
    bot = _make_bot({"response_mode": "auto_respond", "reply_in_channel": True},
                    sender_type="other_bot")
    await bot._handle_slack_message(
        _evt(user="UCLAUDE"), bot.app.client, wake_source="app_mention")
    bot.message_handler.assert_awaited_once()
    assert _dispatched(bot).metadata.get("reply_in_channel") is True


@pytest.mark.asyncio
async def test_self_sender_short_circuits_without_dispatch(monkeypatch):
    # Loop guard: our own message returns before stamping OR dispatching.
    monkeypatch.setattr(config, "reply_in_channel_default", True, raising=False)
    bot = _make_bot({"reply_in_channel": True}, sender_type="self")
    await bot._handle_slack_message(_evt(), bot.app.client, wake_source="app_mention")
    bot.message_handler.assert_not_awaited()


# ============================================================ layer 2: main.py placement effect

def _place_app(resp_meta=None, content="answer"):
    from main import ChatBotV2
    app = ChatBotV2.__new__(ChatBotV2)
    app.participation_engine = None
    app.processor = MagicMock()
    resp = MagicMock()
    resp.type = "text"
    resp.content = content
    resp.metadata = dict(resp_meta or {})
    app.processor.process_message = AsyncMock(return_value=resp)
    app.processor.thread_manager = MagicMock(spec=[])  # no in-flight/upload-latch attrs
    client = MagicMock()
    client.channel_pulse = None
    client.send_thinking_indicator = AsyncMock(return_value="think.1")
    client.delete_message = AsyncMock()
    client.send_message = AsyncMock()
    client.format_text = lambda t: t
    client.maybe_post_response_footer = AsyncMock()
    client.clear_assistant_status = AsyncMock()
    return app, client, resp


def _mention_msg(meta, thread_id="10.0", channel_id="C1"):
    m = {"ts": "10.0"}
    m.update(meta)
    return Message(text="q", user_id="U1", channel_id=channel_id,
                   thread_id=thread_id, metadata=m)


@pytest.mark.asyncio
async def test_top_level_mention_posts_top_level():
    app, client, _ = _place_app()
    await app.handle_message(_mention_msg({"reply_in_channel": True}), client)
    assert client.send_message.await_args.args[1] is None   # top-level (post_thread_id None)


@pytest.mark.asyncio
async def test_in_thread_mention_still_threads():
    app, client, _ = _place_app()
    # A mention INSIDE a thread (ts != thread_id) is never a top-level trigger.
    await app.handle_message(
        _mention_msg({"ts": "11.0", "reply_in_channel": True}, thread_id="10.0"), client)
    assert client.send_message.await_args.args[1] == "10.0"


@pytest.mark.asyncio
async def test_dm_mention_stays_threaded():
    app, client, _ = _place_app()
    await app.handle_message(
        _mention_msg({"reply_in_channel": True}, channel_id="D123"), client)
    assert client.send_message.await_args.args[1] == "10.0"   # DMs never move top-level


@pytest.mark.asyncio
async def test_artifacts_thread_under_top_level_mention(monkeypatch):
    # B2: the reply lands top-level, but the code-interpreter artifact still threads off
    # message.thread_id (post_thread_id is None on a top-level reply).
    app, client, _ = _place_app(resp_meta={"artifact_containers": ["cont_1"]})
    published = AsyncMock(return_value=["file_1"])
    monkeypatch.setattr("message_processor.artifacts.publish_artifacts", published)
    await app.handle_message(_mention_msg({"reply_in_channel": True}), client)
    assert client.send_message.await_args.args[1] is None          # reply top-level
    assert published.await_args.kwargs["thread_id"] == "10.0"      # artifact threads
    assert published.await_args.kwargs["thread_key"] == "C1:10.0"


@pytest.mark.asyncio
async def test_sandbox_rescue_threads_under_top_level_mention():
    # B2: rescued sandbox images are handed message.thread_id, never the None post_thread_id.
    app, client, _ = _place_app()
    app._rescue_sandbox_images = AsyncMock(return_value=0)
    await app.handle_message(_mention_msg({"reply_in_channel": True}), client)
    assert client.send_message.await_args.args[1] is None          # reply top-level
    assert app._rescue_sandbox_images.await_args.args[3] == "10.0"  # 4th positional arg


@pytest.mark.asyncio
async def test_thread_verdict_overrides_setting_and_artifacts_still_thread(monkeypatch):
    # The engine's "thread" placement verdict overrides reply_in_channel (an ALLOWANCE, not a
    # mandate); the reply threads, and the artifact threads off the same root either way.
    app, client, _ = _place_app(resp_meta={"artifact_containers": ["cont_1"]})
    app.participation_engine = MagicMock()
    verdict = ParticipationVerdict(action="respond", emoji="", placement="thread",
                                   reason="worth a thread")
    app._run_participation_gate = AsyncMock(return_value=verdict)
    published = AsyncMock(return_value=["file_1"])
    monkeypatch.setattr("message_processor.artifacts.publish_artifacts", published)
    await app.handle_message(
        _mention_msg({"reply_in_channel": True, "participation_check": True}), client)
    assert client.send_message.await_args.args[1] == "10.0"        # threaded (verdict wins)
    assert published.await_args.kwargs["thread_id"] == "10.0"
