"""Status-only DM indicator (no 'Working on it...' double indicator).

Contract: in DMs where assistant.threads.setStatus succeeds, the composer
status is the SOLE progress indicator — send_thinking_indicator posts no
placeholder message and returns None. Every downstream consumer must be
correct with a None ts: phase updates route to setStatus, streaming seeds
its own message lazily, deletes/edits no-op, errors post fresh messages.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import config
from slack_client.messaging import SlackMessagingMixin


class _MsgClient(SlackMessagingMixin):
    """Minimal host for the messaging mixin (just app.client + no-op logging)."""
    def __init__(self, client):
        self.app = SimpleNamespace(client=client)

    def log_debug(self, *a, **k):
        pass

    log_info = log_warning = log_error = log_debug


# ---------------- send_thinking_indicator: two-surface contract ----------------

@pytest.mark.asyncio
async def test_dm_with_status_posts_no_placeholder(monkeypatch):
    monkeypatch.setattr(config, "enable_assistant_status", True)
    client = SimpleNamespace(
        assistant_threads_setStatus=AsyncMock(),
        chat_postMessage=AsyncMock(return_value={"ts": "1.1"}),
    )
    host = _MsgClient(client)
    ts = await host.send_thinking_indicator("D123", "T1")
    assert ts is None
    client.chat_postMessage.assert_not_awaited()
    client.assistant_threads_setStatus.assert_awaited_once()


@pytest.mark.asyncio
async def test_dm_with_status_failure_falls_back_to_placeholder(monkeypatch):
    from slack_sdk.errors import SlackApiError
    monkeypatch.setattr(config, "enable_assistant_status", True)
    client = SimpleNamespace(
        assistant_threads_setStatus=AsyncMock(
            side_effect=SlackApiError("nope", {"error": "not_allowed"})),
        chat_postMessage=AsyncMock(return_value={"ts": "1.1"}),
    )
    host = _MsgClient(client)
    ts = await host.send_thinking_indicator("D123", "T1")
    assert ts == "1.1"
    client.chat_postMessage.assert_awaited_once()


@pytest.mark.asyncio
async def test_channel_always_posts_placeholder(monkeypatch):
    """setStatus no-ops in plain channels — the message stays the indicator."""
    from slack_sdk.errors import SlackApiError
    monkeypatch.setattr(config, "enable_assistant_status", True)
    client = SimpleNamespace(
        assistant_threads_setStatus=AsyncMock(
            side_effect=SlackApiError("nope", {"error": "not_in_assistant_thread"})),
        chat_postMessage=AsyncMock(return_value={"ts": "2.2"}),
    )
    host = _MsgClient(client)
    ts = await host.send_thinking_indicator("C123", "T1")
    assert ts == "2.2"
    client.chat_postMessage.assert_awaited_once()


@pytest.mark.asyncio
async def test_channel_with_status_success_is_status_only(monkeypatch):
    """The June-2026 agent surface renders the composer status in CHANNEL threads
    too (verified live 2026-07-09) — wherever setStatus succeeds, it is the sole
    indicator; no placeholder message."""
    monkeypatch.setattr(config, "enable_assistant_status", True)
    client = SimpleNamespace(
        assistant_threads_setStatus=AsyncMock(),
        chat_postMessage=AsyncMock(return_value={"ts": "3.3"}),
    )
    host = _MsgClient(client)
    ts = await host.send_thinking_indicator("C123", "T1")
    assert ts is None
    client.chat_postMessage.assert_not_awaited()


@pytest.mark.asyncio
async def test_channel_with_status_failure_posts_placeholder(monkeypatch):
    """Non-agent contexts where setStatus fails keep the visible message indicator."""
    monkeypatch.setattr(config, "enable_assistant_status", True)
    from slack_sdk.errors import SlackApiError
    err = SlackApiError("nope", response={"error": "not_allowed"})
    client = SimpleNamespace(
        assistant_threads_setStatus=AsyncMock(side_effect=err),
        chat_postMessage=AsyncMock(return_value={"ts": "3.3"}),
    )
    host = _MsgClient(client)
    ts = await host.send_thinking_indicator("C123", "T1")
    assert ts == "3.3"


@pytest.mark.asyncio
async def test_dm_with_assistant_status_disabled_posts_placeholder(monkeypatch):
    monkeypatch.setattr(config, "enable_assistant_status", False)
    client = SimpleNamespace(
        assistant_threads_setStatus=AsyncMock(),
        chat_postMessage=AsyncMock(return_value={"ts": "4.4"}),
    )
    host = _MsgClient(client)
    ts = await host.send_thinking_indicator("D123", "T1")
    assert ts == "4.4"
    client.assistant_threads_setStatus.assert_not_awaited()


# ---------------- _update_status: phase text routes to setStatus ----------------

class _ProcHost:
    """Minimal host exposing _update_status + a captured scheduler."""
    def __init__(self):
        from message_processor.utilities import MessageUtilitiesMixin
        self._update_status = MessageUtilitiesMixin._update_status.__get__(self)
        self.scheduled = []

    def _schedule_async_call(self, coro):
        self.scheduled.append(coro)
        coro.close()  # never awaited in tests — close to avoid warnings

    def log_debug(self, *a, **k):
        pass

    log_info = log_warning = log_error = log_debug


def test_update_status_none_ts_dm_routes_to_set_assistant_status():
    host = _ProcHost()
    client = MagicMock()
    client.set_assistant_status = MagicMock(side_effect=lambda *a, **k: _dummy_coro())
    host._update_status(client, "D1", None, "Understanding your request...",
                        thread_id="T1")
    client.set_assistant_status.assert_called_once_with(
        "D1", "T1", status="Understanding your request...")
    assert len(host.scheduled) == 1


def test_update_status_none_ts_channel_routes_to_status():
    # A None ts means the turn is status-only — which can now be a channel
    # thread on the agent surface, not just a DM. Phase updates follow setStatus.
    host = _ProcHost()
    client = MagicMock()
    client.set_assistant_status = MagicMock(side_effect=lambda *a, **k: _dummy_coro())
    host._update_status(client, "C1", None, "Understanding your request...",
                        thread_id="T1")
    client.set_assistant_status.assert_called_once()
    assert len(host.scheduled) == 1


def test_update_status_none_ts_without_thread_id_is_noop():
    host = _ProcHost()
    client = MagicMock()
    client.set_assistant_status = MagicMock(side_effect=lambda *a, **k: _dummy_coro())
    host._update_status(client, "D1", None, "phase text")
    client.set_assistant_status.assert_not_called()


def test_update_status_with_ts_edits_message():
    host = _ProcHost()
    client = MagicMock()
    client.update_message = MagicMock(side_effect=lambda *a, **k: _dummy_coro())
    host._update_status(client, "D1", "123.45", "phase text", thread_id="T1")
    client.update_message.assert_called_once()
    assert len(host.scheduled) == 1


def _dummy_coro():
    async def _noop():
        pass
    return _noop()


# ---------------- streaming gate: None ts allowed only with native ----------------

def _text_handler_host():
    """MessageProcessor-ish host exposing the real _handle_text_response."""
    from message_processor.handlers.text import TextHandlerMixin
    host = MagicMock()
    host._handle_text_response = TextHandlerMixin._handle_text_response.__get__(host)
    host._handle_streaming_text_response = AsyncMock(return_value="STREAMED")
    return host


@pytest.mark.asyncio
async def test_gate_none_ts_native_capable_streams(monkeypatch):
    from base_client import Message
    host = _text_handler_host()
    client = MagicMock()
    client.supports_streaming = MagicMock(return_value=True)
    client.supports_native_streaming = MagicMock(return_value=True)
    msg = Message(text="hi", user_id="U1", channel_id="D1", thread_id="T1")
    thread_state = MagicMock()

    async def fake_config(**kw):
        return {"enable_streaming": True}
    with patch.object(config, "get_thread_config_async", side_effect=fake_config):
        result = await host._handle_text_response(
            "hello", thread_state, client, msg, thinking_id=None)
    assert result == "STREAMED"
    host._handle_streaming_text_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_gate_none_ts_without_native_does_not_stream(monkeypatch):
    """No placeholder + no native path → non-streaming (never a dead edit loop)."""
    from base_client import Message
    from message_processor.handlers.text import TextHandlerMixin
    host = MagicMock()
    host._handle_streaming_text_response = AsyncMock()
    real = TextHandlerMixin._handle_text_response

    client = MagicMock(spec=[])  # no supports_native_streaming attribute at all
    client.supports_streaming = MagicMock(return_value=True)
    msg = Message(text="hi", user_id="U1", channel_id="D1", thread_id="T1")

    async def fake_config(**kw):
        return {"enable_streaming": True}

    called = {}

    async def probe(self, *a, **kw):
        # Reaching the non-streaming section means the gate rejected streaming;
        # abort here — the rest of the method needs a full harness.
        called["non_streaming"] = True
        raise RuntimeError("stop-probe")

    with patch.object(config, "get_thread_config_async", side_effect=fake_config):
        src_gate_streams = False
        try:
            # Run the real method just far enough to see which branch it takes.
            import types
            bound = types.MethodType(real, host)
            # Patch the streaming handler to detect wrong-branch routing
            host._handle_streaming_text_response = AsyncMock(
                side_effect=AssertionError("should not stream without native"))
            thread_state = MagicMock()
            thread_state.messages = []
            await bound("hello", thread_state, client, msg, thinking_id=None)
        except AssertionError:
            src_gate_streams = True
        except Exception:
            pass  # non-streaming path hit real logic and failed on mocks — fine
    assert not src_gate_streams


# ---------------- vision gate parity ----------------

def test_vision_gates_stay_in_sync():
    """The vision streaming gate and its 'streamed' metadata gate must use the
    same condition — source-level check to prevent double-posting drift."""
    import inspect
    from message_processor.handlers import vision
    src = inspect.getsource(vision)
    assert src.count("thinking_id is not None or native_capable") >= 2, (
        "vision.py's streaming gate and streamed-metadata gate must both use "
        "(thinking_id is not None or native_capable)"
    )


# ---------------- image handlers: never store a None status id ----------------

def test_image_handlers_guard_status_message_id():
    import inspect
    from message_processor.handlers import image_gen, image_edit
    for mod in (image_gen, image_edit):
        src = inspect.getsource(mod)
        for i, line in enumerate(src.splitlines()):
            if 'response_metadata["status_message_id"]' in line and "=" in line.split("status_message_id")[1]:
                window = "\n".join(src.splitlines()[max(0, i - 2):i])
                assert "if generating_id" in window or "if editing_id" in window, (
                    f"{mod.__name__}: status_message_id stored without a None guard"
                )


# --------------------------------------------------------------------------- status text sanitization
class TestStatusPlainText:
    def test_known_shortcodes_become_unicode(self):
        from slack_client.messaging import _status_plain_text
        assert _status_plain_text(":hourglass_flowing_sand: working on it…") == "⏳ working on it…"
        assert _status_plain_text(":mag: crunching the data…") == "🔍 crunching the data…"

    def test_unknown_and_custom_shortcodes_are_stripped(self):
        from slack_client.messaging import _status_plain_text
        # Workspace custom emoji have no Unicode form — strip, don't show ":datassential:" literally.
        assert _status_plain_text(":datassential: digging in…") == "digging in…"

    def test_all_shortcode_string_falls_back(self):
        from slack_client.messaging import _status_plain_text
        assert _status_plain_text(":datassential:") == "working on it…"
        assert _status_plain_text("") == "working on it…"

    @pytest.mark.asyncio
    async def test_explicit_phase_status_sends_same_text_both_fields(self, monkeypatch):
        # Slack renders nothing without a non-empty status ("" is the clear signal,
        # verified live 2026-07-10), so a phase update sends ONE sanitized text in
        # BOTH fields — identical surfaces, no mismatched dual indicator.
        from config import config
        monkeypatch.setattr(config, "enable_assistant_status", True)
        host = _MsgClient(SimpleNamespace(assistant_threads_setStatus=AsyncMock()))
        await host.set_assistant_status("D1", "1.0", status=":hourglass_flowing_sand: pulling it together…")
        kwargs = host.app.client.assistant_threads_setStatus.await_args.kwargs
        assert kwargs["status"] == "⏳ pulling it together…"
        assert kwargs["loading_messages"] == ["⏳ pulling it together…"]

    @pytest.mark.asyncio
    async def test_initial_indicator_picks_one_pool_message_both_fields(self, monkeypatch):
        from config import config
        monkeypatch.setattr(config, "enable_assistant_status", True)
        monkeypatch.setattr(config, "status_loading_messages_inline", True)
        monkeypatch.setattr(config, "status_loading_messages", [":mag: one…", ":datassential: two…"])
        host = _MsgClient(SimpleNamespace(assistant_threads_setStatus=AsyncMock()))
        await host.set_assistant_status("D1", "1.0")
        kwargs = host.app.client.assistant_threads_setStatus.await_args.kwargs
        assert kwargs["status"] in ("🔍 one…", "two…")
        assert kwargs["loading_messages"] == [kwargs["status"]]

    @pytest.mark.asyncio
    async def test_clear_sends_bare_empty_status(self, monkeypatch):
        # The API rejects loading_messages=[] and treats status="" as the clear —
        # a clear must go out bare (needed explicitly after native-streamed replies,
        # which never trip Slack's auto-clear).
        from config import config
        monkeypatch.setattr(config, "enable_assistant_status", True)
        host = _MsgClient(SimpleNamespace(assistant_threads_setStatus=AsyncMock()))
        assert await host.clear_assistant_status("D1", "1.0") is True
        kwargs = host.app.client.assistant_threads_setStatus.await_args.kwargs
        assert kwargs["status"] == ""
        assert "loading_messages" not in kwargs
