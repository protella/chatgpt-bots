"""Phase G — agent_view migration + native streaming wiring.

Covers: NativeStreamCoordinator (split floor, marker shapes, part rolling, fence
continuity, inert fallback, finalize suffix), the app_home_opened tab filter +
greeting dedup (incl. cross-dedup with the legacy assistant_thread_started),
app_context_changed logging, the setStatus participation guard, and the marker
round-trip guarantees for the markdown-flavored shapes the native sink writes.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from message_markers import (
    CONTINUATION_TRAILER,
    continuation_trailer_markdown,
    ends_with_continuation,
    part_prefix,
    part_prefix_markdown,
    starts_as_continuation,
    strip_continuation_markers,
)
from slack_client.event_handlers.assistant_events import SlackAssistantEventsMixin
from slack_client.messaging import NativeStreamSession
from streaming.native_sink import NativeStreamCoordinator, find_stream_split


# ---------------- test doubles ----------------

class _FakeSlackSDK:
    """chat.startStream/appendStream/stopStream double that records every call."""

    def __init__(self, fail_start=False, fail_append_after=None, fail_stop=False):
        self._n = 0
        self._fail_start = fail_start
        self._fail_append_after = fail_append_after
        self._fail_stop = fail_stop
        self._appends = 0
        self.messages = {}  # ts -> accumulated markdown text
        self.stopped = set()

    async def chat_startStream(self, channel, thread_ts=None, markdown_text=None,
                               recipient_team_id=None):
        if self._fail_start:
            raise RuntimeError("startStream down")
        self._n += 1
        ts = f"{self._n}00.{self._n}"
        self.messages[ts] = markdown_text or ""
        return {"ts": ts}

    async def chat_appendStream(self, channel, ts, markdown_text):
        self._appends += 1
        if self._fail_append_after is not None and self._appends > self._fail_append_after:
            raise RuntimeError("appendStream down")
        self.messages[ts] += markdown_text

    async def chat_stopStream(self, channel, ts, markdown_text=None, blocks=None):
        if self._fail_stop:
            raise RuntimeError("stopStream down")
        if markdown_text:
            self.messages[ts] += markdown_text
        self.stopped.add(ts)


class _FakeClient:
    """Platform-client double exposing only what the coordinator uses."""

    def __init__(self, sdk):
        self.sdk = sdk

    def begin_native_stream(self, channel_id, thread_id):
        return NativeStreamSession(self.sdk, channel_id, thread_id, team_id="TEAM1")


def _slack_stores(markdown: str) -> str:
    """Simulate Slack's markdown→mrkdwn conversion for the marker shapes we write."""
    return markdown.replace("**", "*")


# ---------------- find_stream_split ----------------

def test_split_prefers_paragraph_boundary():
    text = "a" * 100 + "\n\n" + "b" * 100
    assert find_stream_split(text, 150) == 102  # right after the \n\n


def test_split_respects_floor_of_already_sent_text():
    text = "a" * 100 + "\n\n" + "b" * 100
    # everything before the paragraph break is already sent -> can't split there
    split = find_stream_split(text, 150, floor=120)
    assert 120 <= split <= 150


def test_split_floor_at_or_past_limit_returns_floor():
    assert find_stream_split("x" * 300, 100, floor=100) == 100
    assert find_stream_split("x" * 300, 100, floor=150) == 150


# ---------------- marker shapes (the R2 rule) ----------------

def test_markdown_markers_store_as_canonical_mrkdwn_shapes():
    stored_trailer = _slack_stores(continuation_trailer_markdown())
    assert stored_trailer.strip() == CONTINUATION_TRAILER
    assert ends_with_continuation("body" + stored_trailer)

    stored_prefix = _slack_stores(part_prefix_markdown(2))
    assert stored_prefix == part_prefix(2)
    assert starts_as_continuation(stored_prefix + "body")


def test_strip_handles_markdown_and_italic_variants():
    for trailer in ("*Continued in next message...*",
                    "**Continued in next message...**",
                    "_Continued in next message..._"):
        assert strip_continuation_markers(f"hello\n\n{trailer}") == "hello"
    for prefix in ("*Part 3 (continued)*\n\n", "**Part 3 (continued)**\n\n",
                   "_Part 3 (continued)_\n\n"):
        assert strip_continuation_markers(f"{prefix}world") == "world"


# ---------------- NativeStreamCoordinator ----------------

@pytest.mark.asyncio
async def test_coordinator_streams_single_part():
    sdk = _FakeSlackSDK()
    coord = NativeStreamCoordinator(_FakeClient(sdk), "C1", "T1", char_limit=500)
    assert await coord.start() is True
    ok, overflow = await coord.update("Hello")
    assert ok and overflow is None
    ok, overflow = await coord.update("Hello world")
    assert ok and overflow is None
    assert await coord.finalize("Hello world", suffix="\n\n_Used Tools: web_search_")
    ts = coord.current_ts
    assert sdk.messages[ts] == "Hello world\n\n_Used Tools: web_search_"
    assert ts in sdk.stopped


@pytest.mark.asyncio
async def test_coordinator_rolls_parts_with_shared_markers():
    sdk = _FakeSlackSDK()
    coord = NativeStreamCoordinator(_FakeClient(sdk), "C1", "T1", char_limit=200)
    assert await coord.start()
    text = ("alpha " * 30 + "\n\n" + "beta " * 30).strip()  # > 200 chars
    ok, overflow = await coord.update(text)
    assert ok and overflow is not None
    assert coord.part == 2

    first_ts, second_ts = coord.part_ts
    stored_first = _slack_stores(sdk.messages[first_ts])
    assert ends_with_continuation(stored_first)
    assert first_ts in sdk.stopped

    stored_second = _slack_stores(sdk.messages[second_ts])
    assert starts_as_continuation(stored_second)
    # the merger must reassemble the original text exactly
    merged = (strip_continuation_markers(stored_first).rstrip() + "\n\n"
              + strip_continuation_markers(stored_second))
    assert "".join(merged.split()) == "".join(text.split())


@pytest.mark.asyncio
async def test_coordinator_reopens_code_fence_across_roll():
    sdk = _FakeSlackSDK()
    coord = NativeStreamCoordinator(_FakeClient(sdk), "C1", "T1", char_limit=200)
    assert await coord.start()
    code = "```python\n" + ("x = 1\n" * 60)  # long, still-open fence
    ok, overflow = await coord.update(code)
    assert ok and overflow is not None
    first_ts, second_ts = coord.part_ts
    first = sdk.messages[first_ts]
    # part 1 closed the fence before the trailer; part 2 reopened it with the hint
    assert first.count("```") % 2 == 0
    assert "```python\n" in sdk.messages[second_ts]


@pytest.mark.asyncio
async def test_coordinator_start_failure_marks_failed():
    coord = NativeStreamCoordinator(_FakeClient(_FakeSlackSDK(fail_start=True)),
                                    "C1", "T1", char_limit=500)
    assert await coord.start() is False
    assert coord.failed
    ok, overflow = await coord.update("hi")
    assert not ok and overflow is None


@pytest.mark.asyncio
async def test_coordinator_mid_stream_failure_exposes_ts_for_fallback():
    sdk = _FakeSlackSDK(fail_append_after=1)
    coord = NativeStreamCoordinator(_FakeClient(sdk), "C1", "T1", char_limit=500)
    assert await coord.start()
    ok, _ = await coord.update("first")     # append #1 ok
    assert ok
    ok, _ = await coord.update("first two")  # append #2 fails -> inert
    assert not ok and coord.failed
    assert coord.current_ts is not None      # legacy edits continue on this message


@pytest.mark.asyncio
async def test_coordinator_finalize_failure_reports_false():
    sdk = _FakeSlackSDK(fail_stop=True)
    coord = NativeStreamCoordinator(_FakeClient(sdk), "C1", "T1", char_limit=500)
    assert await coord.start()
    assert await coord.finalize("some text") is False
    assert coord.failed  # caller falls back to a legacy edit on current_ts


# ---------------- agent_view event handlers ----------------

class _AssistantHost(SlackAssistantEventsMixin):
    def __init__(self, history_messages=None):
        self.app = SimpleNamespace(client=SimpleNamespace(
            chat_postMessage=AsyncMock(),
            assistant_threads_setSuggestedPrompts=AsyncMock(),
            conversations_history=AsyncMock(return_value={"messages": history_messages or []}),
        ))
        self.debug_lines = []

    def log_debug(self, msg, *a, **k):
        self.debug_lines.append(str(msg))

    log_info = log_warning = log_error = log_debug


@pytest.mark.asyncio
async def test_app_home_opened_messages_tab_greets_once(monkeypatch):
    from config import config
    monkeypatch.setattr(config, "enable_assistant_surface", True)
    monkeypatch.setattr(config, "assistant_greeting", "Hi there!")
    host = _AssistantHost()

    await host._handle_app_home_opened({"tab": "messages", "channel": "D1", "user": "U1"}, None)
    host.app.client.chat_postMessage.assert_awaited_once_with(channel="D1", text="Hi there!")

    # second visit: deduped
    await host._handle_app_home_opened({"tab": "messages", "channel": "D1", "user": "U1"}, None)
    assert host.app.client.chat_postMessage.await_count == 1


@pytest.mark.asyncio
async def test_greeting_skipped_when_conversation_has_history(monkeypatch):
    # A returning user's DM has history — never re-greet (restarts forget the
    # in-memory dedup; the conversation itself is the source of truth).
    from config import config
    monkeypatch.setattr(config, "enable_assistant_surface", True)
    monkeypatch.setattr(config, "assistant_greeting", "Hi there!")
    host = _AssistantHost(history_messages=[{"ts": "1.0", "text": "old msg"}])
    await host._handle_app_home_opened({"tab": "messages", "channel": "D1", "user": "U1"}, None)
    host.app.client.chat_postMessage.assert_not_awaited()


@pytest.mark.asyncio
async def test_greeting_skipped_when_history_check_fails(monkeypatch):
    # Fail-open: if the history probe errors, skip the greeting (a spurious
    # greeting is worse than a missing one).
    from config import config
    monkeypatch.setattr(config, "enable_assistant_surface", True)
    monkeypatch.setattr(config, "assistant_greeting", "Hi there!")
    host = _AssistantHost()
    host.app.client.conversations_history = AsyncMock(side_effect=Exception("boom"))
    await host._handle_app_home_opened({"tab": "messages", "channel": "D1", "user": "U1"}, None)
    host.app.client.chat_postMessage.assert_not_awaited()


@pytest.mark.asyncio
async def test_app_home_opened_home_tab_is_filtered(monkeypatch):
    from config import config
    monkeypatch.setattr(config, "enable_assistant_surface", True)
    monkeypatch.setattr(config, "assistant_greeting", "Hi there!")
    host = _AssistantHost()
    await host._handle_app_home_opened({"tab": "home", "channel": "D1"}, None)
    host.app.client.chat_postMessage.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_and_agent_view_greetings_cross_dedupe(monkeypatch):
    from config import config
    monkeypatch.setattr(config, "enable_assistant_surface", True)
    monkeypatch.setattr(config, "assistant_greeting", "Hi there!")
    monkeypatch.setattr(config, "assistant_suggested_prompts", [])
    host = _AssistantHost()

    await host._handle_assistant_thread_started(
        {"assistant_thread": {"channel_id": "D1", "thread_ts": "1.0"}}, None)
    assert host.app.client.chat_postMessage.await_count == 1
    # agent_view event for the same DM channel: no second greeting
    await host._handle_app_home_opened({"tab": "messages", "channel": "D1"}, None)
    assert host.app.client.chat_postMessage.await_count == 1


@pytest.mark.asyncio
async def test_app_context_changed_logs_and_never_raises():
    host = _AssistantHost()
    await host._handle_app_context_changed({"context": {"channel_id": "C9"}})
    assert any("app_context_changed" in line for line in host.debug_lines)


# ---------------- setStatus participation guard ----------------

@pytest.mark.asyncio
async def test_participation_ignore_never_touches_status_or_indicator():
    """The engine's non-respond verdict must return BEFORE any indicator/setStatus
    call — on the agent_view surface setStatus auto-opens the thread (Phase G guard)."""
    from main import ChatBotV2
    from base_client import Message

    handler = ChatBotV2.__new__(ChatBotV2)  # skip heavy __init__
    handler.processor = SimpleNamespace(thread_manager=None)
    handler._run_participation_gate = AsyncMock(return_value=None)  # ignore/react/backoff

    client = MagicMock()
    client.send_thinking_indicator = AsyncMock()
    client.set_assistant_status = AsyncMock()

    msg = Message(channel_id="C1", thread_id="T1", user_id="U1", text="hi",
                  metadata={"participation_check": True, "ts": "T1"})
    await handler.handle_message(msg, client)

    client.send_thinking_indicator.assert_not_awaited()
    client.set_assistant_status.assert_not_awaited()


# ---------------- sink selection in the text handler ----------------

def test_streaming_handlers_wire_the_shared_native_sink():
    """Source guard: both streaming handlers must build NativeStreamCoordinator
    (never a private reimplementation) and never inline marker literals — the
    rebuild merger only recognizes the message_markers shapes (R2 rule)."""
    import pathlib
    for rel in ("message_processor/handlers/text.py", "message_processor/handlers/vision.py"):
        src = pathlib.Path(rel).read_text()
        assert "NativeStreamCoordinator(" in src, f"{rel} lost the native sink wiring"
        assert "Continued in next message" not in src, f"{rel} inlines a marker literal"
        assert "*Part {" not in src and '"*Part ' not in src, f"{rel} inlines a part prefix"


def test_supports_native_streaming_still_gates_on_flag(monkeypatch):
    """The coordinator is only built when the client advertises native support,
    which itself keys off SLACK_NATIVE_STREAMING (default off)."""
    from config import config
    from slack_client.messaging import SlackMessagingMixin

    host = SlackMessagingMixin.__new__(SlackMessagingMixin)
    host.app = SimpleNamespace(client=SimpleNamespace(chat_startStream=AsyncMock()))
    monkeypatch.setattr(config, "slack_native_streaming", False)
    monkeypatch.setattr(config, "enable_streaming", True)
    monkeypatch.setattr(config, "slack_streaming", True)
    assert host.supports_native_streaming() is False
    monkeypatch.setattr(config, "slack_native_streaming", True)
    assert host.supports_native_streaming() is True
