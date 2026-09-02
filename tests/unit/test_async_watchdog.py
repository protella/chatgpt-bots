"""Async-lock watchdog: it detects, it reports once, and it never touches thread state.

The bug it fixes: a legitimate 473s turn (three sequential image edits) tripped a hardcoded
310s threshold, was logged as stuck every 10 seconds, and had `had_timeout` set on it — so the
NEXT turn opened with a false "my last answer never finished" notice.
"""
import pytest

from message_processor.thread_manager import AsyncThreadStateManager


@pytest.fixture
def manager():
    return AsyncThreadStateManager()


def test_threshold_is_derived_from_config(manager, monkeypatch):
    from message_processor import thread_manager as tm

    monkeypatch.setattr(tm.config, "max_tool_rounds", 10)
    monkeypatch.setattr(tm.config, "api_timeout_read", 180.0)
    monkeypatch.setattr(tm.config, "api_timeout_image", 300.0)

    # rounds * the longest per-round timeout + buffer — the 473s turn sits well inside it.
    assert manager._watchdog_max_lock_duration() == 10 * 300 + 10


@pytest.mark.asyncio
async def test_stuck_thread_is_logged_once_across_ticks(manager, monkeypatch):
    thread_key = "C0TEST:1700000000.000100"
    monkeypatch.setattr(
        manager._lock_manager, "get_stuck_threads",
        lambda max_duration: _stuck([thread_key]),
    )
    errors = []
    monkeypatch.setattr(manager, "log_error", lambda msg, **kw: errors.append(msg))

    reported: set = set()
    await manager._check_stuck_locks(3010, reported)
    await manager._check_stuck_locks(3010, reported)

    assert len(errors) == 1
    assert thread_key in errors[0]


@pytest.mark.asyncio
async def test_watchdog_does_not_mutate_thread_state(manager, monkeypatch):
    thread = await manager.get_or_create_thread_async("1700000000.000100", "C0TEST")
    thread.is_processing = True
    thread_key = "C0TEST:1700000000.000100"
    monkeypatch.setattr(
        manager._lock_manager, "get_stuck_threads",
        lambda max_duration: _stuck([thread_key]),
    )

    await manager._check_stuck_locks(3010, set())

    assert thread.had_timeout is False
    assert thread.is_processing is True


async def _stuck(keys):
    return keys
