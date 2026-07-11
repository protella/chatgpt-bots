"""Phase 7 — per-channel config & response modes.

Covers: channel_settings CRUD (sync + async), mode resolution (DB override vs global
fallback), directive injection into the response system prompt, reply_in_channel placement,
and the bot-sender onboarding bypass. All with stubbed I/O — no live bot, no legacy suite.
"""
from __future__ import annotations

import sqlite3
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from base_client import Message
from config import config
from database import DatabaseManager
from message_processor.utilities import MessageUtilitiesMixin
from slack_client.event_handlers.message_events import SlackMessageEventsMixin
from slack_client.utilities import SlackUtilitiesMixin


# --------------------------------------------------------------------------- DB CRUD

class TestChannelSettingsDB:
    @pytest.fixture
    def temp_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("os.makedirs"):
                db = DatabaseManager("test")
                db.db_path = f"{tmpdir}/test.db"
                if getattr(db, "conn", None):
                    db.conn.close()
                db.conn = sqlite3.connect(db.db_path, check_same_thread=False, isolation_level=None)
                db.conn.row_factory = sqlite3.Row
                db.conn.execute("PRAGMA journal_mode=WAL")
                db.init_schema()
                yield db
                if getattr(db, "conn", None):
                    db.conn.close()

    def test_get_unset_returns_none(self, temp_db):
        assert temp_db.get_channel_settings("C_NONE") is None

    def test_set_then_get(self, temp_db):
        temp_db.set_channel_settings("C1", response_mode="auto_respond",
                                     directives="only jump in on deploy failures",
                                     reply_in_channel=True, updated_by="U1")
        row = temp_db.get_channel_settings("C1")
        assert row["response_mode"] == "auto_respond"
        assert row["directives"] == "only jump in on deploy failures"
        assert row["reply_in_channel"] is True  # stored as int, returned as bool
        assert row["updated_by"] == "U1"

    def test_partial_update_keeps_other_fields(self, temp_db):
        temp_db.set_channel_settings("C2", response_mode="auto_respond", directives="rule A")
        # Update only directives — mode must be preserved.
        temp_db.set_channel_settings("C2", directives="rule B")
        row = temp_db.get_channel_settings("C2")
        assert row["response_mode"] == "auto_respond"
        assert row["directives"] == "rule B"

    async def test_async_roundtrip(self, temp_db):
        await temp_db.set_channel_settings_async("C3", response_mode="off", directives="stay quiet")
        row = await temp_db.get_channel_settings_async("C3")
        assert row["response_mode"] == "off"
        assert row["directives"] == "stay quiet"
        assert row["reply_in_channel"] is False

    async def test_async_get_unset_none(self, temp_db):
        assert await temp_db.get_channel_settings_async("C_MISSING") is None


# --------------------------------------------------------------------------- mixin harness

class _Bot(SlackMessageEventsMixin, SlackUtilitiesMixin):
    def log_debug(self, *a, **k): pass
    def log_info(self, *a, **k): pass
    def log_warning(self, *a, **k): pass
    def log_error(self, *a, **k): pass


def _make_bot():
    bot = _Bot.__new__(_Bot)
    bot.bot_user_id = "UBOT"
    bot.bot_id = "BBOT"
    bot.app_id = None
    bot.message_handler = AsyncMock()
    bot.app = MagicMock()
    bot.app.client = MagicMock()
    bot.db = MagicMock()
    bot.db.get_channel_settings_async = AsyncMock(return_value=None)

    async def _fake_event_to_message(event, client):
        return Message(
            text=event.get("text", ""),
            user_id=event.get("user"),
            channel_id=event.get("channel"),
            thread_id=event.get("thread_ts") or event.get("ts"),
            metadata={"ts": event.get("ts")},
        )

    bot._event_to_message = _fake_event_to_message
    return bot


def _evt(**kw):
    e = {"channel": "C1", "ts": "100.1", "user": "UHUMAN", "text": "hello there", "channel_type": "channel"}
    e.update(kw)
    return e


# --------------------------------------------------------------------------- mode resolution

def test_resolve_mode_fallback_and_override():
    bot = _make_bot()
    assert bot._resolve_mode(None) == config.channel_response_mode  # global default ('tag_only')
    assert bot._resolve_mode({"response_mode": "auto_respond"}) == "auto_respond"
    assert bot._resolve_mode({"response_mode": "OFF"}) == "off"  # normalized


async def test_get_channel_response_mode_db_vs_global():
    bot = _make_bot()
    bot.db.get_channel_settings_async = AsyncMock(return_value={"response_mode": "auto_respond"})
    assert await bot._get_channel_response_mode("C1") == "auto_respond"
    bot.db.get_channel_settings_async = AsyncMock(return_value=None)
    assert await bot._get_channel_response_mode("C1") == config.channel_response_mode


async def test_dm_has_no_channel_settings():
    bot = _make_bot()
    bot.db.get_channel_settings_async = AsyncMock(return_value={"response_mode": "auto_respond"})
    # DMs (channel id starts with 'D') never carry channel settings → global default.
    assert await bot._get_channel_settings("D123") is None


# --------------------------------------------------------------------------- directive surfacing

async def test_channel_message_surfaces_directives_and_reply_in_channel():
    bot = _make_bot()
    bot._get_channel_settings = AsyncMock(return_value={
        "response_mode": "tag_only", "directives": "only deploys", "reply_in_channel": True,
    })
    # Addressed by name so it dispatches even in tag_only.
    await bot._handle_channel_message(_evt(text="ChatGPT status?"), bot.app.client)
    assert bot.message_handler.await_count == 1
    msg = bot.message_handler.await_args.args[0]
    assert msg.metadata.get("channel_directives") == "only deploys"
    assert msg.metadata.get("reply_in_channel") is True


def test_system_prompt_includes_channel_directives():
    proc = MessageUtilitiesMixin.__new__(type("P", (MessageUtilitiesMixin,), {}))
    client = MagicMock()
    client.name = "slack"
    with_directive = proc._get_system_prompt(client, channel_directives="stay quiet unless tagged")
    assert "CHANNEL GROUND RULES" in with_directive
    assert "stay quiet unless tagged" in with_directive
    without = proc._get_system_prompt(client)
    assert "CHANNEL GROUND RULES" not in without


# --------------------------------------------------------------------------- bot onboarding bypass

async def test_bot_sender_bypasses_onboarding():
    bot = _make_bot()
    bot.db.get_user_preferences_async = AsyncMock(return_value={"settings_completed": True})
    bot._post_settings_button_if_new_thread = AsyncMock()
    evt = _evt(user="UOTHER", text="hi from another bot")
    evt["bot_id"] = "BOTHER"  # other bot
    await bot._handle_slack_message(evt, bot.app.client)
    assert bot.message_handler.await_count == 1
    bot.db.get_user_preferences_async.assert_not_awaited()  # onboarding skipped


async def test_human_sender_still_reaches_onboarding():
    bot = _make_bot()
    bot.db.get_user_preferences_async = AsyncMock(return_value={"settings_completed": True})
    bot._post_settings_button_if_new_thread = AsyncMock()
    await bot._handle_slack_message(_evt(user="UHUMAN"), bot.app.client)
    bot.db.get_user_preferences_async.assert_awaited()  # human goes through onboarding gate
    assert bot.message_handler.await_count == 1


# --------------------------------------------------------------------------- off gates @mentions

async def test_off_level_drops_app_mention():
    """The modal promises "Off — never respond here, even when @mentioned": an explicit
    @mention wake in a participation-off channel must be dropped before dispatch
    (otherwise off collapses into mentions_only)."""
    bot = _make_bot()
    bot._get_channel_settings = AsyncMock(return_value={"participation_level": "off"})
    await bot._handle_slack_message(
        _evt(text="<@UBOT> testing"), bot.app.client, wake_source="app_mention")
    bot.message_handler.assert_not_awaited()


async def test_mentions_only_level_still_answers_app_mention():
    bot = _make_bot()
    bot.db.get_user_preferences_async = AsyncMock(return_value={"settings_completed": True})
    bot._post_settings_button_if_new_thread = AsyncMock()
    bot._get_channel_settings = AsyncMock(return_value={"participation_level": "mentions_only"})
    await bot._handle_slack_message(
        _evt(text="<@UBOT> testing"), bot.app.client, wake_source="app_mention")
    assert bot.message_handler.await_count == 1


async def test_off_level_does_not_gate_dm():
    """DMs have no channel settings — the off gate is app_mention + channel only."""
    bot = _make_bot()
    bot.db.get_user_preferences_async = AsyncMock(return_value={"settings_completed": True})
    bot._post_settings_button_if_new_thread = AsyncMock()
    bot._maybe_set_assistant_thread_title = AsyncMock()  # DM path touches the assistant surface
    await bot._handle_slack_message(
        _evt(channel="D123", text="hi"), bot.app.client, wake_source="dm")
    assert bot.message_handler.await_count == 1
