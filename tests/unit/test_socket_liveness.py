"""F9 — socket-liveness monitor (detection-only).

Covers: envelope receipt refreshes the monotonic timestamp; ERROR when events AND
ping-pong are both frozen; a single WARNING per drought episode when only events are
stale; warning→error escalation still logs the error; recovery logged at INFO; disabled
at timeout 0; and the monitor NEVER calls any socket/reconnect primitive.
"""
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from slack_client.socket_liveness import SocketLivenessMonitor


def _logs():
    return SimpleNamespace(info=MagicMock(), warning=MagicMock(), error=MagicMock())


def _monitor(client, timeout=600, logs=None):
    logs = logs or _logs()
    m = SocketLivenessMonitor(
        client, timeout=timeout,
        log_info=logs.info, log_warning=logs.warning, log_error=logs.error)
    return m, logs


def _stale(monitor, seconds):
    """Force the last-event clock `seconds` into the past."""
    monitor.last_event_monotonic = time.monotonic() - seconds


# --------------------------------------------------------------- envelope seam

def test_attach_appends_listener_and_records_event():
    client = SimpleNamespace(message_listeners=[], last_ping_pong_time=time.time())
    m, _ = _monitor(client)
    assert m.attach() is True
    assert len(client.message_listeners) == 1
    _stale(m, 500)
    before = m.last_event_monotonic
    m.record_event()
    assert m.last_event_monotonic > before


def test_attach_gracefully_handles_missing_seam():
    m, _ = _monitor(SimpleNamespace())  # no message_listeners
    assert m.attach() is False


def test_attach_skips_when_disabled():
    # A disabled monitor (timeout 0) must NOT install a listener at all.
    client = SimpleNamespace(message_listeners=[], last_ping_pong_time=time.time())
    m, _ = _monitor(client, timeout=0)
    assert m.attach() is False
    assert client.message_listeners == []


def test_stop_detaches_listener():
    import asyncio
    client = SimpleNamespace(message_listeners=[], last_ping_pong_time=time.time())
    m, _ = _monitor(client)
    assert m.attach() is True
    assert len(client.message_listeners) == 1
    asyncio.run(m.stop())
    assert client.message_listeners == []       # listener removed
    assert m._listeners is None


def test_record_event_recovery_log_never_raises():
    # Logging from the envelope path must never raise into the socket client.
    client = SimpleNamespace(message_listeners=[], last_ping_pong_time=time.time())
    boom = MagicMock(side_effect=RuntimeError("log sink down"))
    m = SocketLivenessMonitor(client, timeout=600, log_info=boom,
                              log_warning=MagicMock(), log_error=MagicMock())
    m._episode = "warning"  # pretend a drought is active so recovery tries to log
    m.record_event()        # must not raise despite the failing logger
    assert m._episode is None


# --------------------------------------------------------------- check logic

def test_healthy_when_events_fresh():
    client = SimpleNamespace(last_ping_pong_time=time.time())
    m, logs = _monitor(client, timeout=600)
    m._check()  # last_event is ~now
    logs.error.assert_not_called()
    logs.warning.assert_not_called()


def test_error_when_events_and_pings_both_frozen():
    client = SimpleNamespace(last_ping_pong_time=time.time() - 700)
    m, logs = _monitor(client, timeout=600)
    _stale(m, 700)
    m._check()
    logs.error.assert_called_once()
    assert "presumed dead" in logs.error.call_args.args[0]
    logs.warning.assert_not_called()


def test_error_when_pings_never_observed():
    client = SimpleNamespace(last_ping_pong_time=None)
    m, logs = _monitor(client, timeout=600)
    _stale(m, 700)
    m._check()
    logs.error.assert_called_once()


def test_warning_once_per_episode_when_only_events_stale():
    client = SimpleNamespace(last_ping_pong_time=time.time())  # pings fresh
    m, logs = _monitor(client, timeout=600)
    _stale(m, 700)
    m._check()
    m._check()  # still stale, still fresh pings — must NOT warn again
    logs.warning.assert_called_once()
    logs.error.assert_not_called()


def test_warning_then_escalates_to_error():
    client = SimpleNamespace(last_ping_pong_time=time.time())
    m, logs = _monitor(client, timeout=600)
    _stale(m, 700)
    m._check()                       # WARNING (pings fresh)
    client.last_ping_pong_time = time.time() - 700  # pings now freeze too
    _stale(m, 700)
    m._check()                       # escalates → ERROR
    logs.warning.assert_called_once()
    logs.error.assert_called_once()


def test_recovery_logged_and_episode_reset():
    client = SimpleNamespace(last_ping_pong_time=time.time())
    m, logs = _monitor(client, timeout=600)
    _stale(m, 700)
    m._check()                       # enters warning episode
    m.record_event()                 # an envelope arrives
    logs.info.assert_called_once()
    assert "recovered" in logs.info.call_args.args[0]
    # A fresh drought after recovery warns again (new episode).
    _stale(m, 700)
    m._check()
    assert logs.warning.call_count == 2


# --------------------------------------------------------------- disabled + no reconnect

def test_timeout_zero_disables_monitor_task():
    client = SimpleNamespace(last_ping_pong_time=time.time())
    m, logs = _monitor(client, timeout=0)
    m.start()
    assert m._task is None
    assert any("disabled" in c.args[0] for c in logs.info.call_args_list)


def test_monitor_never_calls_socket_or_reconnect_methods():
    # A MagicMock client would auto-create any attribute; assert the reconnect/teardown
    # primitives are never touched by the detection-only monitor.
    client = MagicMock()
    client.last_ping_pong_time = time.time() - 700
    client.message_listeners = []
    m, _ = _monitor(client, timeout=600)
    m.attach()
    _stale(m, 700)
    m._check()
    m.record_event()
    for forbidden in ("connect", "connect_to_new_endpoint", "close", "disconnect",
                      "close_async", "start_async"):
        getattr(client, forbidden).assert_not_called()
