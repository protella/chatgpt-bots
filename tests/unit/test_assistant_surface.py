"""Agent split-view (Assistant surface) adapter tests.

Covers the assistant_thread_started greeting + suggested prompts, context_changed logging,
best-effort thread titling, flag gating, and a regression check that classic DM messages
still route through _handle_slack_message (the adapter must not touch the message pipeline).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from config import config
from slack_client.event_handlers.assistant_events import SlackAssistantEventsMixin
from slack_client.event_handlers.registration import SlackRegistrationMixin
from slack_client.utilities import SlackUtilitiesMixin


class _Bot(SlackAssistantEventsMixin, SlackUtilitiesMixin):
    def log_debug(self, *a, **k): pass
    def log_info(self, *a, **k): pass
    def log_warning(self, *a, **k): pass
    def log_error(self, *a, **k): pass


def _make_bot():
    bot = _Bot.__new__(_Bot)
    bot.bot_user_id = "UBOT"
    bot.bot_id = "BBOT"
    bot.app_id = None
    bot.app = MagicMock()
    bot.app.client = MagicMock()
    bot.app.client.chat_postMessage = AsyncMock(return_value={"ok": True})
    bot.app.client.assistant_threads_setSuggestedPrompts = AsyncMock(return_value={"ok": True})
    bot.app.client.assistant_threads_setTitle = AsyncMock(return_value={"ok": True})
    return bot


def _started_event(**ctx):
    return {
        "type": "assistant_thread_started",
        "assistant_thread": {"channel_id": "D123", "thread_ts": "111.222", "context": ctx},
    }


# --- assistant_thread_started ---

@pytest.mark.asyncio
async def test_thread_started_posts_greeting_and_prompts():
    bot = _make_bot()
    await bot._handle_assistant_thread_started(_started_event(), None)

    kwargs = bot.app.client.chat_postMessage.call_args.kwargs
    assert kwargs["channel"] == "D123"
    assert kwargs["thread_ts"] == "111.222"
    assert kwargs["text"] == config.assistant_greeting

    pk = bot.app.client.assistant_threads_setSuggestedPrompts.call_args.kwargs
    assert pk["channel_id"] == "D123" and pk["thread_ts"] == "111.222"
    prompts = pk["prompts"]
    assert 1 <= len(prompts) <= 4
    assert all(set(p) == {"title", "message"} for p in prompts)
    assert all(len(p["title"]) <= 38 for p in prompts)


@pytest.mark.asyncio
async def test_thread_started_flag_off_is_noop(monkeypatch):
    bot = _make_bot()
    monkeypatch.setattr(config, "enable_assistant_surface", False)
    await bot._handle_assistant_thread_started(_started_event(), None)
    bot.app.client.chat_postMessage.assert_not_called()
    bot.app.client.assistant_threads_setSuggestedPrompts.assert_not_called()


@pytest.mark.asyncio
async def test_thread_started_malformed_payload_is_safe():
    bot = _make_bot()
    for evt in ({}, {"assistant_thread": None}, {"assistant_thread": {"channel_id": "D1"}}):
        await bot._handle_assistant_thread_started(evt, None)  # must not raise
    bot.app.client.chat_postMessage.assert_not_called()


@pytest.mark.asyncio
async def test_thread_started_survives_api_failures():
    bot = _make_bot()
    bot.app.client.chat_postMessage = AsyncMock(side_effect=RuntimeError("down"))
    bot.app.client.assistant_threads_setSuggestedPrompts = AsyncMock(side_effect=RuntimeError("down"))
    await bot._handle_assistant_thread_started(_started_event(), None)  # must not raise


# --- assistant_thread_context_changed ---

@pytest.mark.asyncio
async def test_context_changed_logs_without_error():
    bot = _make_bot()
    await bot._handle_assistant_thread_context_changed(_started_event(channel_id="C42"))
    await bot._handle_assistant_thread_context_changed({})  # malformed also fine


# --- thread titles ---

@pytest.mark.asyncio
async def test_title_set_once_and_truncated():
    bot = _make_bot()
    long_text = "x" * 100
    await bot._maybe_set_assistant_thread_title("D123", "111.222", long_text)
    kwargs = bot.app.client.assistant_threads_setTitle.call_args.kwargs
    assert kwargs["channel_id"] == "D123" and kwargs["thread_ts"] == "111.222"
    assert len(kwargs["title"]) == 60 and kwargs["title"].endswith("…")

    # Second message in the same thread must NOT retitle
    await bot._maybe_set_assistant_thread_title("D123", "111.222", "follow-up")
    assert bot.app.client.assistant_threads_setTitle.call_count == 1


@pytest.mark.asyncio
async def test_title_skips_non_dm_and_tolerates_failure(monkeypatch):
    bot = _make_bot()
    await bot._maybe_set_assistant_thread_title("C123", "1.2", "channel thread")  # not a DM
    bot.app.client.assistant_threads_setTitle.assert_not_called()

    bot.app.client.assistant_threads_setTitle = AsyncMock(side_effect=RuntimeError("not an assistant thread"))
    await bot._maybe_set_assistant_thread_title("D9", "2.3", "hello")  # must not raise

    monkeypatch.setattr(config, "enable_assistant_surface", False)
    bot2 = _make_bot()
    await bot2._maybe_set_assistant_thread_title("D1", "3.4", "hi")
    bot2.app.client.assistant_threads_setTitle.assert_not_called()


# --- regression: classic DM routing is untouched ---

class _RegBot(SlackRegistrationMixin, SlackAssistantEventsMixin, SlackUtilitiesMixin):
    def log_debug(self, *a, **k): pass
    def log_info(self, *a, **k): pass

    def __init__(self):
        self.bot_user_id = "UBOT"
        self.bot_id = "BBOT"
        self.app_id = None
        self._handle_slack_message = AsyncMock()
        self._handle_channel_message = AsyncMock()
        self._register_settings_handlers = MagicMock()
        self._handlers = {}
        self.app = MagicMock()

        def _event(name):
            def _decorator(fn):
                self._handlers[name] = fn
                return fn
            return _decorator

        self.app.event = _event
        self._register_handlers()


@pytest.mark.asyncio
async def test_dm_message_still_routes_to_handle_slack_message():
    bot = _RegBot()
    assert "assistant_thread_started" in bot._handlers  # adapter registered
    assert "assistant_thread_context_changed" in bot._handlers

    dm_event = {"channel_type": "im", "channel": "D123", "ts": "5.6", "user": "UHUMAN", "text": "hi"}
    await bot._handlers["message"](event=dm_event, say=None, client=None)
    bot._handle_slack_message.assert_awaited_once()
