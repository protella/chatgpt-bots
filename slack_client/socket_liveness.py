"""Socket-liveness monitor (F9) — DETECTION ONLY.

Watches the async Socket Mode connection for the silent half-open death that slack_sdk's
own ping monitoring can miss (a healthy process receiving zero events). It records a
monotonic timestamp on every inbound envelope and, on a 60s cadence, logs when events have
been absent for longer than the configured window:

  * ping-pong ALSO frozen for the window  → ERROR (unambiguous death; restart likely
    needed). Never `max(last_event, last_ping_pong)`: in the half-open case pings stay
    fresh and an event-only trigger would never fire (F9-2).
  * pings still fresh                      → one WARNING per drought episode (idle vs.
    half-open are passively indistinguishable; WS ping/pong are control frames, not
    envelopes, so an idle workspace produces zero envelopes — F9-3).

It NEVER calls any reconnect/socket primitive (auto-reconnect was descoped 2026-07-10).
The monitor task never crashes the app: every cycle is guarded and it self-continues.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Optional


class SocketLivenessMonitor:
    def __init__(
        self,
        socket_client: Any,
        *,
        timeout: float,
        log_info: Callable[[str], None],
        log_warning: Callable[[str], None],
        log_error: Callable[[str], None],
        check_interval: float = 60.0,
    ) -> None:
        self._client = socket_client
        self._timeout = timeout
        self._check_interval = check_interval
        self._log_info = log_info
        self._log_warning = log_warning
        self._log_error = log_error
        # Monotonic clock for events (immune to wall-clock jumps); slack_sdk stamps
        # last_ping_pong_time with wall-clock time.time(), so pings are compared in that
        # frame separately.
        self.last_event_monotonic: float = time.monotonic()
        # Episode state: None (healthy) | "warning" (pings fresh) | "error" (both frozen).
        # One log per episode; a warning→error escalation logs once more; recovery resets.
        self._episode: Optional[str] = None
        self._task: Optional[asyncio.Task] = None
        # The listeners list we appended our callback to, kept so stop() can detach it
        # (an un-removed listener would leak and keep firing after the monitor is gone).
        self._listeners: Optional[list] = None

    @staticmethod
    def _safe_log(fn: Callable[[str], None], msg: str) -> None:
        """Call a logger, swallowing any error — logging must never raise into the socket
        client's envelope path (record_event fires from _on_message)."""
        try:
            fn(msg)
        except Exception:
            pass

    # --- envelope seam ---
    def attach(self) -> bool:
        """Append the envelope listener to the socket client's message_listeners.

        No-op (returns False) when the monitor is DISABLED (timeout <= 0) — a disabled
        monitor must not install a listener at all — or when the client exposes no such
        seam (older/mocked clients), in which case the monitor stays quiet unless started."""
        if self._timeout is None or self._timeout <= 0:
            return False
        client = self._client
        listeners = getattr(client, "message_listeners", None)
        if listeners is None or not hasattr(listeners, "append"):
            return False
        listeners.append(self._on_message)
        self._listeners = listeners
        return True

    async def _on_message(self, *args: Any, **kwargs: Any) -> None:
        """Fires on EVERY inbound Socket Mode envelope. Records freshness and logs recovery
        when an episode ends. Never raises into the socket client."""
        self.record_event()

    def record_event(self) -> None:
        self.last_event_monotonic = time.monotonic()
        if self._episode is not None:
            self._safe_log(self._log_info, "socket liveness recovered — envelopes resumed")
            self._episode = None

    # --- monitor task ---
    def start(self) -> None:
        """Start the periodic monitor task. No-op when disabled (timeout <= 0)."""
        if self._timeout is None or self._timeout <= 0:
            self._safe_log(self._log_info, "Socket-liveness monitor disabled (SOCKET_LIVENESS_TIMEOUT=0)")
            return
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run())
        self._safe_log(self._log_info,
            f"Socket-liveness monitor started (window {self._timeout:.0f}s, "
            f"check every {self._check_interval:.0f}s, detection-only)")

    async def _run(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._check_interval)
                self._check()
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001 — the monitor must never crash
                self._safe_log(self._log_warning, f"socket-liveness monitor cycle error (continuing): {e}")

    def _check(self) -> None:
        now = time.monotonic()
        events_idle = now - self.last_event_monotonic
        if events_idle <= self._timeout:
            return  # healthy — events flowing

        ppt = getattr(self._client, "last_ping_pong_time", None)
        ping_idle = (time.time() - ppt) if ppt else None
        both_frozen = ping_idle is None or ping_idle > self._timeout

        if both_frozen:
            # Unambiguous-death signature. Log once on entry / on warning→error escalation.
            if self._episode != "error":
                self._episode = "error"
                ping_desc = (f"frozen {ping_idle:.0f}s" if ping_idle is not None
                             else "never observed")
                self._safe_log(self._log_error,
                    f"socket presumed dead (no events {events_idle:.0f}s, "
                    f"ping-pong {ping_desc}) — restart likely required")
        else:
            # Idle or half-open — passively indistinguishable. One WARNING per episode.
            if self._episode is None:
                self._episode = "warning"
                self._safe_log(self._log_warning,
                    f"socket event drought: no envelopes for {events_idle:.0f}s, but "
                    f"ping-pong fresh ({ping_idle:.0f}s ago) — idle or half-open "
                    f"(passively indistinguishable)")

    async def stop(self) -> None:
        """Cancel the monitor task and detach the envelope listener (best-effort)."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        # Detach our listener so it stops firing and doesn't leak after teardown.
        if self._listeners is not None:
            try:
                if self._on_message in self._listeners:
                    self._listeners.remove(self._on_message)
            except (ValueError, TypeError):
                pass
            self._listeners = None
