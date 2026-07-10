"""Phase 3 — status & streaming refactor.

Covers: NativeStreamSession (start/update-delta/finish + graceful failure), assistant
setStatus (success / graceful no-op / disabled), supports_native_streaming gating, branded
loading-message config, and the outbound self-prefix strip.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from slack_sdk.errors import SlackApiError

from config import config
from slack_client.messaging import NativeStreamSession, SlackMessagingMixin
from slack_client.formatting.text import SlackFormattingMixin, strip_leading_self_prefix


class _MsgClient(SlackMessagingMixin):
    """Minimal host for the messaging mixin (just app.client + no-op logging)."""
    def __init__(self, client):
        self.app = SimpleNamespace(client=client)

    def log_debug(self, *a, **k):
        pass

    log_info = log_warning = log_error = log_debug


def _api_error(code):
    return SlackApiError(code, {"error": code})


# ---------------- NativeStreamSession ----------------

@pytest.mark.asyncio
async def test_native_stream_start_update_finish_sends_deltas():
    client = SimpleNamespace(
        chat_startStream=AsyncMock(return_value={"ts": "100.1"}),
        chat_appendStream=AsyncMock(),
        chat_stopStream=AsyncMock(),
    )
    sess = NativeStreamSession(client, "C1", "T1")

    assert await sess.start("") is True
    assert sess.active and sess.ts == "100.1"

    await sess.update("Hello")
    client.chat_appendStream.assert_awaited_with(channel="C1", ts="100.1", markdown_text="Hello")

    await sess.update("Hello world")  # only the new tail is appended
    client.chat_appendStream.assert_awaited_with(channel="C1", ts="100.1", markdown_text=" world")

    await sess.finish("Hello world")  # nothing new -> stop without extra text
    client.chat_stopStream.assert_awaited_once()


@pytest.mark.asyncio
async def test_native_stream_top_level_skips_start():
    # chat.startStream requires thread_ts — a top-level (threadless) reply must not
    # even attempt the call (it 400s with "missing required field: thread_ts").
    client = SimpleNamespace(chat_startStream=AsyncMock())
    sess = NativeStreamSession(client, "C1", None)
    assert await sess.start("hi") is False
    assert sess.active is False
    client.chat_startStream.assert_not_called()


@pytest.mark.asyncio
async def test_native_stream_start_failure_is_inert():
    client = SimpleNamespace(chat_startStream=AsyncMock(side_effect=RuntimeError("boom")))
    sess = NativeStreamSession(client, "C1", "T1")
    assert await sess.start("x") is False
    assert sess.active is False
    # update on an inert session is a no-op returning False (caller falls back)
    assert await sess.update("anything") is False


@pytest.mark.asyncio
async def test_native_stream_append_failure_flips_inactive():
    client = SimpleNamespace(
        chat_startStream=AsyncMock(return_value={"ts": "1"}),
        chat_appendStream=AsyncMock(side_effect=_api_error("rate_limited")),
    )
    sess = NativeStreamSession(client, "C1", "T1")
    await sess.start("")
    assert await sess.update("text") is False
    assert sess.active is False


# ---------------- assistant setStatus ----------------

@pytest.mark.asyncio
async def test_set_assistant_status_success(monkeypatch):
    monkeypatch.setattr(config, "enable_assistant_status", True)
    client = SimpleNamespace(assistant_threads_setStatus=AsyncMock())
    host = _MsgClient(client)
    assert await host.set_assistant_status("C1", "T1") is True
    _, kwargs = client.assistant_threads_setStatus.call_args
    assert kwargs["channel_id"] == "C1" and kwargs["thread_ts"] == "T1"
    # One random pool message rides in BOTH fields (a non-empty status is what
    # renders; "" is the clear signal), sanitized for the plain-text surface.
    from slack_client.messaging import _status_plain_text
    pool = {_status_plain_text(m) for m in config.get_loading_messages()}
    assert kwargs["status"] in pool
    assert kwargs["loading_messages"] == [kwargs["status"]]


@pytest.mark.asyncio
async def test_set_assistant_status_graceful_in_plain_channel(monkeypatch):
    monkeypatch.setattr(config, "enable_assistant_status", True)
    client = SimpleNamespace(assistant_threads_setStatus=AsyncMock(side_effect=_api_error("not_in_assistant_thread")))
    host = _MsgClient(client)
    assert await host.set_assistant_status("C1", "T1") is False  # no raise


@pytest.mark.asyncio
async def test_set_assistant_status_disabled(monkeypatch):
    monkeypatch.setattr(config, "enable_assistant_status", False)
    client = SimpleNamespace(assistant_threads_setStatus=AsyncMock())
    host = _MsgClient(client)
    assert await host.set_assistant_status("C1", "T1") is False
    client.assistant_threads_setStatus.assert_not_called()


# ---------------- supports_native_streaming gating ----------------

@pytest.mark.parametrize("native,stream,expected", [
    (True, True, True),
    (False, True, False),
    (True, False, False),
])
def test_supports_native_streaming_gating(monkeypatch, native, stream, expected):
    monkeypatch.setattr(config, "slack_native_streaming", native)
    monkeypatch.setattr(config, "enable_streaming", stream)
    monkeypatch.setattr(config, "slack_streaming", stream)
    client = SimpleNamespace(chat_startStream=AsyncMock())
    host = _MsgClient(client)
    assert host.supports_native_streaming() is expected


# ---------------- branded loading messages config ----------------

def test_loading_messages_config_present():
    assert isinstance(config.status_loading_messages, list) and config.status_loading_messages
    assert isinstance(config.status_loading_fallback, str) and config.status_loading_fallback


def test_random_loading_message_draws_from_pool():
    pool = set(config.get_loading_messages())
    assert config.random_loading_message() in pool


@pytest.mark.asyncio
async def test_thinking_indicator_fallback_uses_loading_pool(monkeypatch):
    # The legacy placeholder (posted only where setStatus fails) draws from the
    # same variance pool as the native status — no baked-in "Thinking..." text.
    monkeypatch.setattr(config, "enable_assistant_status", False)
    client = SimpleNamespace(chat_postMessage=AsyncMock(return_value={"ts": "1.0"}))
    host = _MsgClient(client)
    assert await host.send_thinking_indicator("C1", "T1") == "1.0"
    text = client.chat_postMessage.call_args.kwargs["text"]
    assert text.startswith(config.circle_loader_emoji)
    assert text.split(" ", 1)[1] in set(config.get_loading_messages())


# ---------------- outbound self-prefix strip ----------------

@pytest.mark.parametrize("text,expected", [
    ("ChatGPT: hello", "hello"),
    ("ChatGPT-Dev: yo there", "yo there"),
    ("chatgpt: lower", "lower"),  # case-insensitive
    ("  Assistant:   padded", "padded"),  # leading ws + multi-space after colon; trailing preserved
])
def test_strip_self_prefix_strips_known_names(text, expected):
    names = ["ChatGPT", "ChatGPT-Dev", "Assistant", "Bot"]
    assert strip_leading_self_prefix(text, names) == expected


@pytest.mark.parametrize("text", ["Note: keep this", "Step 1: do it", "Peter: hi", "no colon here", ""])
def test_strip_self_prefix_leaves_others(text):
    names = ["ChatGPT", "ChatGPT-Dev", "Assistant", "Bot"]
    assert strip_leading_self_prefix(text, names) == text


def test_strip_self_prefix_no_names_is_noop():
    assert strip_leading_self_prefix("ChatGPT: hi", None) == "ChatGPT: hi"


class _Fmt(SlackFormattingMixin):
    def __init__(self):
        self.user_cache = {}
        self.bot_user_id = None
        self.markdown_converter = SimpleNamespace(convert=lambda t: t)


def test_format_text_strips_self_prefix():
    assert _Fmt().format_text("ChatGPT: hello world") == "hello world"
