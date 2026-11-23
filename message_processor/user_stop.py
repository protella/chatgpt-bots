"""User Stop requests from Slack's native Stop button (`agent_session_stopped`).

Docs/specs/slack_agent_surfaces_round.md §A. Stop is COOPERATIVE: nothing here cancels a
task. The event handler records a request; a running turn polls `stop_requested_since` at its
own safe checkpoints and ends through its normal path with whatever it already delivered.

A request only affects turns ADMITTED BEFORE the click (`since` < the request's `at`), so a
delayed or retried event can never stop the user's next message. Entries expire after
`_TTL_S`, and the registry is process-local (lost on restart, like all thread state).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from typing import Dict, FrozenSet, Optional, Tuple

_TTL_S = 15 * 60

# Message.metadata key main.py stamps with the turn's admission time (epoch) at lease open.
SINCE_KEY = "user_stop_since"


@dataclass(frozen=True)
class StopRequest:
    channel_id: str
    thread_ts: str
    user_id: str
    at: float  # epoch seconds the Stop was pressed (event_ts), else receipt time
    stream_ts: FrozenSet[str] = field(default_factory=frozenset)


_lock = threading.Lock()
_requests: Dict[Tuple[str, str], StopRequest] = {}


def _prune(now: float) -> None:
    for key in [k for k, r in _requests.items() if now - r.at > _TTL_S]:
        _requests.pop(key, None)


def press_time(event_ts: Optional[str]) -> float:
    """Epoch seconds of a press: the event's ts, else receipt time."""
    try:
        return float(event_ts) if event_ts else time.time()
    except (TypeError, ValueError):
        return time.time()


def request_stop(channel_id: str, thread_ts: str, user_id: str,
                 event_ts: Optional[str] = None,
                 stream_ts: Optional[list] = None) -> StopRequest:
    """Record a Stop for one thread and return the thread's stored request.

    The newest press sets `at` and `user_id` (it covers strictly more turns). Stream ts are the
    UNION across presses, so a later or out-of-order press never forgets a stream Slack already
    stopped."""
    at = press_time(event_ts)
    streams = frozenset(str(t) for t in (stream_ts or []) if t)
    with _lock:
        _prune(time.time())
        prev = _requests.get((channel_id, thread_ts))
        if prev is None:
            req = StopRequest(channel_id=channel_id, thread_ts=thread_ts, user_id=user_id,
                              at=at, stream_ts=streams)
        elif at >= prev.at:
            req = StopRequest(channel_id=channel_id, thread_ts=thread_ts, user_id=user_id,
                              at=at, stream_ts=prev.stream_ts | streams)
        else:
            req = replace(prev, stream_ts=prev.stream_ts | streams)
        _requests[(channel_id, thread_ts)] = req
        return req


def stop_requested_since(channel_id: Optional[str], thread_ts: Optional[str],
                         since: float) -> Optional[StopRequest]:
    """The Stop that applies to a turn admitted at `since` (epoch seconds), or None."""
    if not channel_id or not thread_ts:
        return None
    with _lock:
        req = _requests.get((channel_id, thread_ts))
    if req is None or req.at <= since or time.time() - req.at > _TTL_S:
        return None
    return req


def stream_was_stopped(channel_id: Optional[str], thread_ts: Optional[str],
                       stream_ts: Optional[str]) -> bool:
    """True when Slack itself stopped this stream message (it was listed in the event)."""
    if not channel_id or not thread_ts or not stream_ts:
        return False
    with _lock:
        req = _requests.get((channel_id, thread_ts))
    return bool(req and str(stream_ts) in req.stream_ts)


def clear_for_tests() -> None:
    with _lock:
        _requests.clear()
