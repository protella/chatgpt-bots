"""Slack's native Stop button — the `agent_session_stopped` event side (spec §A.2)."""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import config
from message_processor import user_stop
from message_processor.stale_send_guard import ConversationWatermarks
from slack_client.event_handlers import agent_session
from slack_client.event_handlers.registration import SlackRegistrationMixin

PRESS = round(time.time() - 60, 6)
EVENT: Dict[str, Any] = {"type": "agent_session_stopped", "channel": "D100",
                         "thread_ts": "100.000100", "user": "UHUMAN1",
                         "event_ts": f"{PRESS:.6f}", "streaming_message_ts": ["101.0", "102.0"]}


@pytest.fixture(autouse=True)
def _clean_registry():
    user_stop.clear_for_tests()
    yield
    user_stop.clear_for_tests()


def _host(entries: List[Dict[str, Any]], *, running: bool = False) -> SimpleNamespace:
    """Each entry may carry `started` (epoch); default is before the press."""
    tm = MagicMock()
    tm.cancellable_in_flight.return_value = [
        {k: v for k, v in e.items() if k != "started"} for e in entries]
    tm.background_started_epochs.return_value = {
        e["job_id"]: e.get("started", PRESS - 30) for e in entries}
    tm.request_background_cancel.return_value = {"ok": True}
    return SimpleNamespace(processor=SimpleNamespace(thread_manager=tm),
                           turn_running_in_thread=MagicMock(return_value=running),
                           clear_assistant_status=AsyncMock(return_value=True),
                           app=SimpleNamespace(client=SimpleNamespace(
                               agents_sessions_setStatus=AsyncMock())))


def _sessions(host: Any) -> List[Any]:
    return [(c.kwargs["channel_id"], c.kwargs["thread_ts"], c.kwargs["status"])
            for c in host.app.client.agents_sessions_setStatus.await_args_list]


def test_handler_is_always_registered(monkeypatch):
    monkeypatch.setattr(config, "slack_stop_button", False)
    handlers: Dict[str, Any] = {}

    class _Bot(SlackRegistrationMixin):
        def __init__(self) -> None:
            self.app = MagicMock()
            self.app.event = lambda name: (lambda fn: handlers.setdefault(name, fn))
            self.app.action = lambda name: (lambda fn: fn)
            self._register_settings_handlers = MagicMock()

    _Bot()._register_handlers()
    assert "agent_session_stopped" in handlers


@pytest.mark.asyncio
async def test_flag_off_is_a_no_op(monkeypatch):
    monkeypatch.setattr(config, "slack_stop_button", False)
    host = _host([{"job_id": "j1"}])
    await agent_session.handle_agent_session_stopped(host, dict(EVENT))
    assert user_stop.stop_requested_since("D100", "100.000100", 0.0) is None
    host.processor.thread_manager.cancellable_in_flight.assert_not_called()
    host.clear_assistant_status.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["channel", "thread_ts", "user"])
async def test_malformed_payload_is_rejected(monkeypatch, missing):
    monkeypatch.setattr(config, "slack_stop_button", True)
    event = {k: v for k, v in EVENT.items() if k != missing}
    host = _host([{"job_id": "j1"}])
    await agent_session.handle_agent_session_stopped(host, event)
    assert user_stop.stop_requested_since("D100", "100.000100", 0.0) is None
    host.processor.thread_manager.request_background_cancel.assert_not_called()
    host.clear_assistant_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_request_recorded_with_event_ts_and_stream_ts(monkeypatch):
    monkeypatch.setattr(config, "slack_stop_button", True)
    pressed = round(time.time() - 5, 6)
    await agent_session.handle_agent_session_stopped(
        _host([]), dict(EVENT, event_ts=f"{pressed:.6f}"))
    req = user_stop.stop_requested_since("D100", "100.000100", pressed - 1)
    assert req is not None and req.at == pressed and req.user_id == "UHUMAN1"
    assert user_stop.stop_requested_since("D100", "100.000100", pressed + 1) is None
    assert user_stop.stream_was_stopped("D100", "100.000100", "101.0")
    assert user_stop.stream_was_stopped("D100", "100.000100", "102.0")


@pytest.mark.asyncio
async def test_background_cancel_called_per_entry(monkeypatch):
    monkeypatch.setattr(config, "slack_stop_button", True)
    host = _host([{"job_id": "j1", "kind": "job"},
                  {"job_id": "g1", "kind": "image_generation"}])
    await agent_session.handle_agent_session_stopped(host, dict(EVENT))
    tm = host.processor.thread_manager
    tm.cancellable_in_flight.assert_called_once_with("D100:100.000100")
    assert [c.args for c in tm.request_background_cancel.call_args_list] == [
        ("D100:100.000100", "j1"), ("D100:100.000100", "g1")]
    assert all(c.kwargs["reason"] == agent_session.STOP_CANCEL_REASON
               for c in tm.request_background_cancel.call_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("running", [False, True])
async def test_status_cleared_only_when_no_turn_is_running(monkeypatch, running):
    monkeypatch.setattr(config, "slack_stop_button", True)
    host = _host([], running=running)
    await agent_session.handle_agent_session_stopped(host, dict(EVENT))
    host.turn_running_in_thread.assert_called_once_with("D100", "100.000100")
    await agent_session.session_status(host).settle()
    if running:
        host.clear_assistant_status.assert_not_awaited()
        assert _sessions(host) == []
    else:
        host.clear_assistant_status.assert_awaited_once_with("D100", "100.000100")
        assert _sessions(host) == [("D100", "100.000100", "active")]


def test_turn_lookup_reads_the_thread_lease():
    from main import ChatBotV2

    bot = ChatBotV2.__new__(ChatBotV2)
    watermarks = ConversationWatermarks()
    bot.__dict__["_watermarks"] = watermarks
    assert bot._turn_running_in_thread("D100", "100.000100") is False
    message = SimpleNamespace(channel_id="D100", thread_id="100.000100",
                              metadata={"ts": "103.0", "sender_id": "UHUMAN1"})
    lease = watermarks.begin_turn(message)
    assert bot._turn_running_in_thread("D100", "100.000100") is True
    assert bot._turn_running_in_thread("D100", "999.0") is False
    lease.close()
    assert bot._turn_running_in_thread("D100", "100.000100") is False


@pytest.mark.asyncio
async def test_work_started_after_the_press_is_neither_cancelled_nor_cleared(monkeypatch):
    monkeypatch.setattr(config, "slack_stop_button", True)
    host = _host([{"job_id": "old"}, {"job_id": "new", "started": PRESS + 5}])
    await agent_session.handle_agent_session_stopped(host, dict(EVENT))
    tm = host.processor.thread_manager
    assert [c.args[1] for c in tm.request_background_cancel.call_args_list] == ["old"]
    host.clear_assistant_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_older_event_records_its_streams_and_cancels_nothing(monkeypatch):
    monkeypatch.setattr(config, "slack_stop_button", True)
    await agent_session.handle_agent_session_stopped(_host([]), dict(EVENT))
    late = _host([{"job_id": "j1", "started": PRESS - 50}])
    await agent_session.handle_agent_session_stopped(
        late, dict(EVENT, event_ts=f"{PRESS - 20:.6f}", streaming_message_ts=["90.0"]))
    late.processor.thread_manager.request_background_cancel.assert_not_called()
    late.clear_assistant_status.assert_not_awaited()
    assert user_stop.stream_was_stopped("D100", "100.000100", "90.0")
    req = user_stop.stop_requested_since("D100", "100.000100", PRESS - 1)
    assert req is not None and req.at == PRESS


def test_stream_ts_are_the_union_across_presses():
    user_stop.request_stop("D100", "100.000100", "UHUMAN1", event_ts=f"{PRESS:.6f}",
                           stream_ts=["101.0"])
    user_stop.request_stop("D100", "100.000100", "UHUMAN1", event_ts=f"{PRESS + 1:.6f}",
                           stream_ts=[])
    assert user_stop.stream_was_stopped("D100", "100.000100", "101.0")


@pytest.mark.asyncio
async def test_thread_manager_reports_wall_clock_starts():
    from message_processor.thread_manager import AsyncThreadStateManager

    tm = AsyncThreadStateManager(db=None)
    before = time.time()
    tm.register_research("D100:100.000100", "j1", "a deck")
    tm.register_generation("D100:100.000100", "g1", "a picture")
    started = tm.background_started_epochs("D100:100.000100")
    assert set(started) == {"j1", "g1"}
    assert all(v is not None and before <= v <= time.time() for v in started.values())


@pytest.mark.asyncio
async def test_admission_stamps_the_stop_since_on_the_message():
    from main import ChatBotV2

    bot = ChatBotV2(platform="slack")
    seen: Dict[str, Any] = {}

    async def _process(message: Any, *a: Any, **k: Any) -> Any:
        seen["since"] = message.metadata.get(user_stop.SINCE_KEY)
        return MagicMock(type="text", content="ok", metadata={"streamed": False})

    bot.processor = MagicMock()
    bot.processor.process_message = _process
    client = MagicMock()
    client.send_thinking_indicator = AsyncMock(return_value="thinking_1")
    client.delete_message = AsyncMock()
    client.send_message = AsyncMock()
    message = MagicMock(channel_id="D100", thread_id="100.000100",
                        metadata={"ts": "103.0", "sender_id": "UHUMAN1"})
    before = time.time()
    await bot.handle_message(message, client)
    assert seen["since"] is not None and before <= seen["since"] <= time.time()


# --- §A.4: the agent session's processing/active state ------------------------------------

def _session_host(background: Optional[Dict[str, float]] = None,
                  running: bool = False) -> SimpleNamespace:
    host = _host([], running=running)
    host.processor.thread_manager.background_started_epochs.return_value = background or {}
    return host


@pytest.mark.asyncio
async def test_first_visible_status_sends_processing_once(monkeypatch):
    from slack_client.messaging import SlackMessagingMixin

    monkeypatch.setattr(config, "slack_stop_button", True)
    monkeypatch.setattr(config, "enable_assistant_status", True)
    host = _session_host(running=True)
    host.app.client.assistant_threads_setStatus = AsyncMock()
    host.log_debug = MagicMock()
    await SlackMessagingMixin.set_assistant_status(host, "D100", "100.0", status="Thinking")
    await SlackMessagingMixin.set_assistant_status(host, "D100", "100.0", status="Searching")
    await agent_session.session_status(host).settle()
    assert _sessions(host) == [("D100", "100.0", "processing")]


@pytest.mark.asyncio
async def test_flag_off_makes_no_session_calls(monkeypatch):
    from slack_client.messaging import SlackMessagingMixin

    monkeypatch.setattr(config, "slack_stop_button", False)
    monkeypatch.setattr(config, "enable_assistant_status", True)
    host = _session_host()
    host.app.client.assistant_threads_setStatus = AsyncMock()
    host.log_debug = MagicMock()
    await SlackMessagingMixin.set_assistant_status(host, "D100", "100.0", status="Thinking")
    agent_session.turn_ended(host, "D100")
    agent_session.background_changed(host, "D100:100.0")
    await agent_session.session_status(host).settle()
    assert _sessions(host) == []


@pytest.mark.asyncio
async def test_turn_end_sends_active(monkeypatch):
    monkeypatch.setattr(config, "slack_stop_button", True)
    host = _session_host()
    tracker = agent_session.session_status(host)
    await tracker.mark_processing("D100", "100.0")
    agent_session.turn_ended(host, "D100")
    await tracker.settle()
    assert _sessions(host) == [("D100", "100.0", "processing"), ("D100", "100.0", "active")]


@pytest.mark.asyncio
async def test_active_is_held_back_while_a_job_runs(monkeypatch):
    monkeypatch.setattr(config, "slack_stop_button", True)
    host = _session_host(background={"j1": PRESS})
    tracker = agent_session.session_status(host)
    await tracker.mark_processing("D100", "100.0")
    agent_session.turn_ended(host, "D100")
    await tracker.settle()
    assert _sessions(host) == [("D100", "100.0", "processing")]
    host.processor.thread_manager.background_started_epochs.return_value = {}
    agent_session.background_changed(host, "D100:100.0")
    await tracker.settle()
    assert _sessions(host)[-1] == ("D100", "100.0", "active")


@pytest.mark.asyncio
async def test_job_start_and_end_pair(monkeypatch):
    from message_processor.thread_manager import AsyncThreadStateManager

    monkeypatch.setattr(config, "slack_stop_button", True)
    host = _session_host()
    tm = AsyncThreadStateManager(db=None)
    host.processor = SimpleNamespace(thread_manager=tm)
    tm.background_listener = lambda key: agent_session.background_changed(host, key)
    tracker = agent_session.session_status(host)
    await tracker.mark_processing("D100", "100.0")
    agent_session.turn_ended(host, "D100")
    await tracker.settle()
    tm.register_research("D100:100.0", "j1", "a deck")
    await tracker.settle()
    tm.finish_research("D100:100.0", "j1")
    await tracker.settle()
    tm.register_generation("D100:100.0", "g1", "a picture")
    await tracker.settle()
    tm.finish_generation("D100:100.0", "g1")
    await tracker.settle()
    assert [s for _, _, s in _sessions(host)] == [
        "processing", "active", "processing", "active", "processing", "active"]


@pytest.mark.asyncio
async def test_processing_is_refreshed_while_a_job_runs(monkeypatch):
    monkeypatch.setattr(config, "slack_stop_button", True)
    monkeypatch.setattr(agent_session, "SESSION_REFRESH_S", 0.01)
    host = _session_host(background={"j1": PRESS})
    tracker = agent_session.session_status(host)
    await tracker.mark_processing("D100", "100.0")
    agent_session.background_changed(host, "D100:100.0")
    await tracker.settle()
    await asyncio.sleep(0.05)
    assert len(_sessions(host)) >= 3          # the start plus at least two refreshes
    host.processor.thread_manager.background_started_epochs.return_value = {}
    await asyncio.sleep(0.03)
    assert not tracker._refresh               # the loop ends with the work


@pytest.mark.asyncio
@pytest.mark.parametrize("error", ["not_allowed", "missing_scope"])
async def test_a_refusal_latches_the_sessions_api_off(monkeypatch, error):
    from slack_sdk.errors import SlackApiError

    monkeypatch.setattr(config, "slack_stop_button", True)
    host = _session_host()
    host.app.client.agents_sessions_setStatus = AsyncMock(
        side_effect=SlackApiError("refused", {"ok": False, "error": error}))
    tracker = agent_session.session_status(host)
    await tracker.mark_processing("D100", "100.0")
    await tracker.mark_processing("D100", "200.0")
    agent_session.turn_ended(host, "D100")
    await tracker.settle()
    assert tracker.disabled
    assert host.app.client.agents_sessions_setStatus.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True])
async def test_every_turn_ending_reaches_the_session_choke(monkeypatch, fails):
    from main import ChatBotV2

    monkeypatch.setattr(config, "slack_stop_button", True)
    ended = MagicMock()
    monkeypatch.setattr(agent_session, "turn_ended", ended)
    bot = ChatBotV2(platform="slack")
    bot.processor = MagicMock()
    bot.processor.process_message = AsyncMock(
        side_effect=RuntimeError("boom") if fails else None,
        return_value=MagicMock(type="text", content="ok", metadata={"streamed": False}))
    client = MagicMock()
    client.send_thinking_indicator = AsyncMock(return_value="thinking_1")
    client.delete_message = AsyncMock()
    client.send_message = AsyncMock()
    client.update_message = AsyncMock()
    message = MagicMock(channel_id="D100", thread_id="100.000100",
                        metadata={"ts": "103.0", "sender_id": "UHUMAN1"})
    try:
        await bot.handle_message(message, client)
    except RuntimeError:
        pass
    ended.assert_called_once_with(client, "D100")
