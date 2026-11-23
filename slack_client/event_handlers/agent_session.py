"""Slack's native Stop button: the `agent_session_stopped` event (spec §A.2).

Stop is COOPERATIVE. This handler only records the request and asks the thread's background
work to stop; a running foreground turn notices the request at its own checkpoints
(`user_stop.stop_requested_since`) and ends through its normal path, which also clears the
status. Nothing here cancels the turn's task.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set, Tuple, cast

from slack_sdk.errors import SlackApiError

from config import config
from logger import setup_logger
from message_processor import user_stop

logger = setup_logger(__name__)

# Rendered on a cancelled job's status card as "cancelled — <reason>". Plain user copy, no tag.
STOP_CANCEL_REASON = "stopped from Slack"

# Slack drops a session's `processing` after an hour; re-send it this often while work runs.
SESSION_REFRESH_S = 30 * 60
# A sessions-API refusal that will not change for this process (app config, not the thread).
_LATCH_ERRORS = frozenset({"not_allowed", "missing_scope"})

_Key = Tuple[str, str]


class AgentSessionStatus:
    """The agent session's `processing` / `active` state, which is what shows Slack's Stop
    button (spec §A.4). Best-effort throughout: a failure is logged at DEBUG and never reaches
    a turn or a job.

    Every change is computed from the truth at send time (`reconcile`) and runs under one lock
    per thread, in the order it was scheduled, so a late "turn ended" can never overwrite the
    next turn's `processing`. The API is called only for threads where the assistant status
    landed (or Slack sent us a Stop), so plain channel threads never reach it."""

    def __init__(self, host: Any) -> None:
        self._host = host
        self._capable: Set[_Key] = set()
        self._state: Dict[_Key, str] = {}
        self._locks: Dict[_Key, asyncio.Lock] = {}
        self._refresh: Dict[_Key, "asyncio.Task[None]"] = {}
        self._tasks: Set["asyncio.Task[Any]"] = set()
        self.disabled = False

    # --- plumbing ---------------------------------------------------------------------------

    def enabled(self) -> bool:
        if not getattr(config, "slack_stop_button", False) or self.disabled:
            return False
        client = getattr(getattr(self._host, "app", None), "client", None)
        return hasattr(client, "agents_sessions_setStatus")

    def schedule(self, coro: Coroutine[Any, Any, Any]) -> None:
        """Run without blocking the caller; never raises."""
        try:
            task = asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            coro.close()
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def settle(self) -> None:
        """Wait for every scheduled change (tests and shutdown)."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)

    def _lock(self, key: _Key) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = self._locks[key] = asyncio.Lock()
        return lock

    async def _send(self, key: _Key, status: str) -> None:
        if not self.enabled():
            return
        try:
            await self._host.app.client.agents_sessions_setStatus(
                channel_id=key[0], thread_ts=key[1], status=status)
            self._state[key] = status
        except SlackApiError as e:
            err = e.response.get("error") if getattr(e, "response", None) else str(e)
            logger.debug(f"agent session setStatus({status}) failed for {key}: {err}")
            if err in _LATCH_ERRORS:
                self.disabled = True
                logger.info(f"agent session status disabled for this process ({err})")
        except Exception as e:  # noqa: BLE001 — never reaches the turn or the job
            logger.debug(f"agent session setStatus({status}) error for {key}: {e}")

    def _background_running(self, key: _Key) -> bool:
        processor = getattr(self._host, "processor", None)
        tm = getattr(processor, "thread_manager", None) if processor is not None else None
        if tm is None or not hasattr(tm, "background_started_epochs"):
            return False
        try:
            return bool(tm.background_started_epochs(f"{key[0]}:{key[1]}"))
        except Exception:  # noqa: BLE001
            return False

    def _foreground_running(self, key: _Key) -> bool:
        running = cast(Optional[Callable[[str, str], bool]],
                       getattr(self._host, "turn_running_in_thread", None))
        try:
            return bool(running(key[0], key[1])) if running is not None else False
        except Exception:  # noqa: BLE001
            return False

    # --- the three transitions --------------------------------------------------------------

    async def mark_processing(self, channel_id: str, thread_ts: str) -> None:
        """A turn's visible status landed: the session is working."""
        if not self.enabled():
            return
        key = (str(channel_id), str(thread_ts))
        self._capable.add(key)
        async with self._lock(key):
            if self._state.get(key) != "processing":
                await self._send(key, "processing")

    async def reconcile(self, channel_id: str, thread_ts: str, *, force: bool = False) -> None:
        """Set the state the thread is really in: `processing` while background work runs,
        untouched while a foreground turn runs (it reconciles on its way out), else `active`.
        `force` marks the thread as a session (Slack sent a Stop for it)."""
        if not self.enabled():
            return
        key = (str(channel_id), str(thread_ts))
        if force:
            self._capable.add(key)
        if key not in self._capable:
            return
        async with self._lock(key):
            if self._background_running(key):
                if self._state.get(key) != "processing":
                    await self._send(key, "processing")
                self._ensure_refresh(key)
                return
            if self._foreground_running(key):
                return
            if self._state.get(key) != "active":
                await self._send(key, "active")

    async def reconcile_channel(self, channel_id: str) -> None:
        """A turn ended in this channel: settle every thread we hold as `processing` there."""
        for key in [k for k, v in self._state.items()
                    if k[0] == str(channel_id) and v == "processing"]:
            await self.reconcile(*key)

    def _ensure_refresh(self, key: _Key) -> None:
        task = self._refresh.get(key)
        if task is not None and not task.done():
            return
        try:
            self._refresh[key] = asyncio.get_running_loop().create_task(self._refresh_loop(key))
        except RuntimeError:
            pass

    async def _refresh_loop(self, key: _Key) -> None:
        try:
            while True:
                await asyncio.sleep(SESSION_REFRESH_S)
                if not self.enabled():
                    return
                async with self._lock(key):
                    if not self._background_running(key):
                        return
                    await self._send(key, "processing")
        finally:
            if self._refresh.get(key) is asyncio.current_task():
                self._refresh.pop(key, None)


def session_status(host: Any) -> AgentSessionStatus:
    """The host's one AgentSessionStatus, made on first use."""
    existing = getattr(host, "_agent_session_status", None)
    if isinstance(existing, AgentSessionStatus):
        return existing
    made = AgentSessionStatus(host)
    try:
        setattr(host, "_agent_session_status", made)
    except Exception:  # noqa: BLE001
        pass
    return made


def note_status_set(host: Any, channel_id: str, thread_ts: str) -> None:
    """messaging.set_assistant_status: a visible status landed."""
    if getattr(config, "slack_stop_button", False):
        tracker = session_status(host)
        tracker.schedule(tracker.mark_processing(channel_id, thread_ts))


def note_status_cleared(host: Any, channel_id: str, thread_ts: str) -> None:
    """messaging.clear_assistant_status ran."""
    if getattr(config, "slack_stop_button", False):
        tracker = session_status(host)
        tracker.schedule(tracker.reconcile(channel_id, thread_ts))


def turn_ended(host: Any, channel_id: Optional[str]) -> None:
    """main.py's turn finally, after the lease closed: success, error, silence, Stop and
    suppression all end here."""
    if getattr(config, "slack_stop_button", False) and channel_id:
        tracker = session_status(host)
        tracker.schedule(tracker.reconcile_channel(channel_id))


def background_changed(host: Any, thread_key: str) -> None:
    """thread_manager's hook: a job or generation registered or finished in this thread."""
    if getattr(config, "slack_stop_button", False):
        channel_id, _, thread_ts = str(thread_key).partition(":")
        if channel_id and thread_ts:
            tracker = session_status(host)
            tracker.schedule(tracker.reconcile(channel_id, thread_ts))


def _stream_ts_list(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v]
    return [str(value)] if value else []


async def handle_agent_session_stopped(host: Any, event: Dict[str, Any]) -> None:
    """Record the Stop, cancel the thread's background work, and clear the status when no
    foreground turn is left to clear it. Never raises."""
    if not getattr(config, "slack_stop_button", False):
        return
    event = event or {}
    channel_id = event.get("channel") or event.get("channel_id")
    thread_ts = event.get("thread_ts")
    user_id = event.get("user") or event.get("user_id")
    if not channel_id or not thread_ts or not user_id:
        logger.warning(f"user_stop malformed agent_session_stopped payload: keys={sorted(event)}")
        return
    channel_id, thread_ts, user_id = str(channel_id), str(thread_ts), str(user_id)
    stream_ts = _stream_ts_list(event.get("streaming_message_ts"))
    pressed_at = user_stop.press_time(event.get("event_ts"))
    try:
        stored = user_stop.request_stop(channel_id, thread_ts, user_id,
                                        event_ts=event.get("event_ts"), stream_ts=stream_ts)
    except Exception as e:  # noqa: BLE001 — a lost request must not take the ack path down
        logger.warning(f"user_stop request not recorded for {channel_id}:{thread_ts}: {e}")
        return
    logger.info(f"user_stop requested channel={channel_id} thread={thread_ts} "
                f"streams={len(stream_ts)}")
    if stored.at > pressed_at:
        # A late event behind a newer press: its streams are recorded, everything else is
        # already the newer press's business.
        return

    thread_key = f"{channel_id}:{thread_ts}"
    processor = getattr(host, "processor", None)
    tm = getattr(processor, "thread_manager", None) if processor is not None else None
    started_after = False
    if tm is not None and hasattr(tm, "cancellable_in_flight"):
        try:
            entries = cast(List[Dict[str, Any]], tm.cancellable_in_flight(thread_key))
            started = cast(Dict[str, Optional[float]],
                           tm.background_started_epochs(thread_key))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"user_stop could not list background work for {thread_key}: {e}")
            entries, started = [], {}
        for entry in entries:
            began = started.get(str(entry.get("job_id")))
            if began is None or began >= pressed_at:
                started_after = True   # work the press never saw: not ours to stop
                continue
            try:
                result = tm.request_background_cancel(thread_key, entry.get("job_id"),
                                                      reason=STOP_CANCEL_REASON)
                logger.info(f"user_stop background_cancel {thread_key} "
                            f"job={entry.get('job_id')} result={result}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"user_stop background cancel failed for {thread_key}: {e}")
    if started_after:
        return  # the newer work owns the status

    running = cast(Optional[Callable[[str, str], bool]],
                   getattr(host, "turn_running_in_thread", None))
    try:
        turn_running = bool(running(channel_id, thread_ts)) if running is not None else False
    except Exception as e:  # noqa: BLE001
        logger.debug(f"user_stop turn lookup failed for {thread_key}: {e}")
        turn_running = False
    if turn_running:
        return  # the turn clears the status itself when it ends
    clear = getattr(host, "clear_assistant_status", None)
    if clear is not None:
        try:
            await clear(channel_id, thread_ts)
        except Exception as e:  # noqa: BLE001
            logger.debug(f"user_stop status clear failed for {thread_key}: {e}")
    tracker = session_status(host)
    tracker.schedule(tracker.reconcile(channel_id, thread_ts, force=True))
