"""Participation-backoff redesign (Layer 1) — the NULL-inherit reply-placement fix.

Covers reply_in_channel resolution: a row with an EXPLICIT False forces threads-only; a row with a
NULL/inherit reply_in_channel falls back to the global default (the bug: the old `elif` hung off
`if cs:` so a NULL row never reached the default); no row at all also inherits the default.

(The per-thread mute mechanism this file once also covered — the message_events pre-gate and the
post_to_thread cross-thread rail — was removed; those tests are gone with it.)

Real decision code, stubbed I/O; no network/DB.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from base_client import Message
from config import config
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


def _make_bot(cs):
    bot = _Bot.__new__(_Bot)
    bot.bot_user_id = "UBOT"
    bot.bot_id = "BBOT"
    bot.app_id = None
    bot.message_handler = AsyncMock()
    bot.app = MagicMock()
    bot.app.client = MagicMock()
    bot.channel_pulse = None  # skip pulse backfill in the dispatch path
    bot.db = MagicMock()

    async def _cs(channel_id):
        return cs

    bot._get_channel_settings = _cs

    async def _fake_event_to_message(event, client):
        return Message(
            text=event.get("text", ""), user_id=event.get("user"),
            channel_id=event.get("channel"),
            thread_id=event.get("thread_ts") or event.get("ts"),
            attachments=[], metadata={"ts": event.get("ts")})

    bot._event_to_message = _fake_event_to_message
    return bot


def _evt(**kw):
    e = {"channel": "C1", "ts": "100.1", "user": "UHUMAN", "text": "anyone know q3 numbers?",
         "channel_type": "channel"}
    e.update(kw)
    return e


@pytest.fixture
def judicious(monkeypatch):
    # auto_respond → judicious: unaddressed messages reach the engine (participation_check).
    monkeypatch.setattr(config, "channel_response_mode", "auto_respond", raising=False)
    monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
    monkeypatch.setattr(config, "bot_name_aliases", ["ChatGPT", "ChatGPT-Dev"], raising=False)


# ----------------------------------------------------------------- reply_in_channel inherit

@pytest.mark.asyncio
async def test_null_reply_in_channel_inherits_default(judicious, monkeypatch):
    monkeypatch.setattr(config, "reply_in_channel_default", True, raising=False)
    # a row EXISTS but reply_in_channel is None (inherit) — the old `elif` never reached the
    # default here and silently forced threads-only.
    bot = _make_bot({"response_mode": "auto_respond", "reply_in_channel": None})
    await bot._handle_channel_message(_evt(text="lunch anyone?"), bot.app.client)
    msg = bot.message_handler.await_args.args[0]
    assert msg.metadata.get("reply_in_channel") is True


@pytest.mark.asyncio
async def test_explicit_false_reply_in_channel_forces_threads(judicious, monkeypatch):
    monkeypatch.setattr(config, "reply_in_channel_default", True, raising=False)
    bot = _make_bot({"response_mode": "auto_respond", "reply_in_channel": False})
    await bot._handle_channel_message(_evt(text="lunch anyone?"), bot.app.client)
    msg = bot.message_handler.await_args.args[0]
    assert msg.metadata.get("reply_in_channel") is not True


@pytest.mark.asyncio
async def test_no_row_inherits_default(judicious, monkeypatch):
    monkeypatch.setattr(config, "reply_in_channel_default", True, raising=False)
    bot = _make_bot(None)
    await bot._handle_channel_message(_evt(text="lunch anyone?"), bot.app.client)
    msg = bot.message_handler.await_args.args[0]
    assert msg.metadata.get("reply_in_channel") is True


@pytest.mark.asyncio
async def test_no_row_default_off_stays_threads(judicious, monkeypatch):
    monkeypatch.setattr(config, "reply_in_channel_default", False, raising=False)
    bot = _make_bot(None)
    await bot._handle_channel_message(_evt(text="lunch anyone?"), bot.app.client)
    msg = bot.message_handler.await_args.args[0]
    assert msg.metadata.get("reply_in_channel") is not True
