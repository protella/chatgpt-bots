"""Phase 5/6 + 2.5 — channel-listening decision logic, reply placement, and bot-in-roster.

These exercise the real decision code in SlackMessageEventsMixin with stubbed I/O, so they
assert the SAFE-by-default behavior the keystone is supposed to ship.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from base_client import Message
from config import config
from message_processor.utilities import build_roster_text
from slack_client.event_handlers.message_events import SlackMessageEventsMixin
from slack_client.utilities import SlackUtilitiesMixin


class _Bot(SlackMessageEventsMixin, SlackUtilitiesMixin):
    """Minimal harness exposing the real channel-decision logic with stubbed logging/I/O."""

    def log_debug(self, *a, **k):
        pass

    def log_info(self, *a, **k):
        pass

    def log_warning(self, *a, **k):
        pass

    def log_error(self, *a, **k):
        pass


def _make_bot():
    bot = _Bot.__new__(_Bot)
    bot.bot_user_id = "UBOT"
    bot.bot_id = "BBOT"
    bot.app_id = None
    bot.message_handler = AsyncMock()
    bot.app = MagicMock()
    bot.app.client = MagicMock()

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


@pytest.fixture
def tag_only(monkeypatch):
    monkeypatch.setattr(config, "channel_response_mode", "tag_only", raising=False)
    monkeypatch.setattr(config, "bot_name_aliases", ["ChatGPT", "ChatGPT-Dev"], raising=False)


@pytest.mark.asyncio
async def test_own_message_by_user_id_short_circuits(tag_only):
    bot = _make_bot()
    await bot._handle_channel_message(_evt(user="UBOT", text="ChatGPT hi"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_own_message_by_bot_id_short_circuits(tag_only):
    bot = _make_bot()
    await bot._handle_channel_message(_evt(bot_id="BBOT", user=None, text="ChatGPT hi"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_subtype_skipped(tag_only):
    bot = _make_bot()
    await bot._handle_channel_message(_evt(subtype="channel_join", text="ChatGPT hi"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_off_mode_never_responds(monkeypatch):
    monkeypatch.setattr(config, "channel_response_mode", "off", raising=False)
    monkeypatch.setattr(config, "bot_name_aliases", ["ChatGPT"], raising=False)
    bot = _make_bot()
    await bot._handle_channel_message(_evt(text="ChatGPT help me"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_tag_only_unaddressed_ignored(tag_only):
    bot = _make_bot()
    await bot._handle_channel_message(_evt(text="lunch anyone?"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_tag_only_name_addressed_responds(tag_only):
    bot = _make_bot()
    await bot._handle_channel_message(_evt(text="ChatGPT, what's the weather?"), bot.app.client)
    bot.message_handler.assert_called_once()
    msg = bot.message_handler.call_args[0][0]
    assert msg.metadata.get("channel_listen") is True
    assert msg.metadata.get("wake_classify") is not True  # addressed → no classifier needed


@pytest.mark.asyncio
async def test_explicit_mention_is_deduped(tag_only):
    # An <@UBOT> mention is already delivered via the app_mention event; channel path must skip.
    bot = _make_bot()
    await bot._handle_channel_message(_evt(text="<@UBOT> hello"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_auto_respond_sets_wake_classify_for_unaddressed(monkeypatch):
    monkeypatch.setattr(config, "channel_response_mode", "auto_respond", raising=False)
    monkeypatch.setattr(config, "bot_name_aliases", ["ChatGPT"], raising=False)
    bot = _make_bot()
    await bot._handle_channel_message(_evt(text="anyone know the q3 numbers?"), bot.app.client)
    bot.message_handler.assert_called_once()
    msg = bot.message_handler.call_args[0][0]
    assert msg.metadata.get("wake_classify") is True


@pytest.mark.asyncio
async def test_thread_reply_one_on_one_responds(tag_only):
    bot = _make_bot()
    bot._thread_participation = AsyncMock(return_value=(True, 1))
    await bot._handle_channel_message(_evt(text="and what about friday?", thread_ts="50.0", ts="60.0"), bot.app.client)
    bot.message_handler.assert_called_once()


@pytest.mark.asyncio
async def test_thread_reply_multiparty_unaddressed_ignored(tag_only):
    bot = _make_bot()
    bot._thread_participation = AsyncMock(return_value=(True, 3))
    await bot._handle_channel_message(_evt(text="sounds good to me", thread_ts="50.0", ts="60.0"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_reply_placed_in_thread(tag_only):
    bot = _make_bot()
    await bot._handle_channel_message(_evt(text="ChatGPT ping", ts="77.7"), bot.app.client)
    msg = bot.message_handler.call_args[0][0]
    assert msg.thread_id == "77.7"  # top-level wake → reply in a thread rooted at the message


def test_text_mentions_bot_name_whole_word(tag_only):
    bot = _make_bot()
    assert bot._text_mentions_bot_name("hey ChatGPT can you help")
    assert bot._text_mentions_bot_name("CHATGPT-DEV go")
    assert not bot._text_mentions_bot_name("the chatgptithon event")  # whole-word match only
    assert not bot._text_mentions_bot_name("no name here")


@pytest.mark.asyncio
async def test_thread_participation_counts_humans_and_bot(tag_only):
    bot = _make_bot()
    bot.app.client.conversations_replies = AsyncMock(return_value={"messages": [
        {"user": "UBOT", "bot_id": "BBOT"},  # self
        {"user": "UHUMAN1"},
        {"user": "UHUMAN2"},
        {"user": "UHUMAN1"},  # dup human
    ]})
    bot_present, humans = await bot._thread_participation("C1", "50.0")
    assert bot_present is True
    assert humans == 2


@pytest.mark.asyncio
async def test_thread_participation_handles_api_error(tag_only):
    bot = _make_bot()
    bot.app.client.conversations_replies = AsyncMock(side_effect=RuntimeError("boom"))
    assert await bot._thread_participation("C1", "50.0") == (False, 0)


def test_default_config_is_safe():
    # OUT OF THE BOX: the bot must not auto-listen, and the default channel mode is tag_only.
    assert config.enable_channel_listening is False
    assert config.channel_response_mode == "tag_only"


def test_bot_with_real_user_id_lands_in_roster():
    # Phase 2.5: another bot that posts with a real user_id can be tagged via the roster.
    txt = build_roster_text({"U123": "Peter", "U999": "Claude"}, user_cache={}, bot_user_id="UBOT")
    assert "<@U999>" in txt
    # The "bot"/"unknown" placeholder ids are excluded (cannot <@>-tag a bot_id).
    txt2 = build_roster_text({"bot": "Bot", "U123": "Peter"}, bot_user_id="UBOT")
    assert "<@bot>" not in txt2
    assert "<@U123>" in txt2
