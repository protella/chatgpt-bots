"""Per-message timestamps for model-visible thread context (F10).

Claude Tag stamps every message in its context with a local timestamp, letting it
reason across time gaps ("last night" vs "this morning", "you asked an hour ago"). Our
rebuilt history was bare `username: text` — the message `ts` lived only in metadata, so
the model could not perceive elapsed time. These helpers prefix each turn with a
deterministic stamp rendered from the message's immutable Slack `ts` in the sender's
profile timezone.

`render_message_timestamp` is a PURE function of (immutable ts, sender IANA tz) — no
`datetime.now()` — so every rebuild renders an identical stamp for the same message
(the F7-5 determinism standard). The config gate lives at the call sites (guard on
`config.enable_message_timestamps`); these helpers never read config.
"""
from __future__ import annotations

import datetime
from typing import Any, Dict, Optional

import pytz


def render_message_timestamp(ts: Any, tz: Optional[str] = "UTC") -> str:
    """Render one message's local timestamp, e.g. ``[Fri 2026-07-10 9:17 PM EDT]``.

    ``ts`` is the message's immutable Slack ts (epoch seconds, optionally with a
    ``.MMMMMM`` fraction); ``tz`` is the sender's IANA zone name, falling back to UTC
    when missing/invalid (e.g. other bots / unknown users). Minute precision, 12-hour
    clock with a NON zero-padded hour ("9:17 PM", not "09:17 PM") — the hour is
    formatted manually rather than via the platform-specific ``%-I`` so the output is
    identical across OSes. The tz label is ``strftime('%Z')`` falling back to the zone
    name. Returns "" when ``ts`` can't be parsed, so callers leave content untouched."""
    if ts is None:
        return ""
    try:
        epoch = float(ts)
    except (TypeError, ValueError):
        return ""
    try:
        zone = pytz.timezone(tz) if tz else pytz.UTC
    except Exception:
        zone = pytz.UTC
    when = datetime.datetime.fromtimestamp(epoch, tz=pytz.UTC).astimezone(zone)
    hour12 = when.hour % 12 or 12
    meridiem = "AM" if when.hour < 12 else "PM"
    weekday = when.strftime("%a")
    date_iso = when.strftime("%Y-%m-%d")
    tz_label = when.strftime("%Z") or getattr(zone, "zone", "UTC")
    return f"[{weekday} {date_iso} {hour12}:{when.minute:02d} {meridiem} {tz_label}]"


def stamp_content(content: Any, ts: Any, tz: Optional[str] = "UTC") -> Any:
    """Prefix ``content`` with the message timestamp (``[stamp] content``).

    Returns ``content`` unchanged when it is not a string or when the stamp can't be
    rendered (unparseable ts). Empty content yields the bare stamp. The stamp is always
    a PREFIX — it never interacts with any end-anchored suffix annotations the caller
    has already appended (``[used tools: …]`` / ``[reactions: …]``)."""
    if not isinstance(content, str):
        return content
    stamp = render_message_timestamp(ts, tz)
    if not stamp:
        return content
    return f"{stamp} {content}" if content else stamp


def sender_timezone(metadata: Optional[Dict[str, Any]], user_id: Optional[str],
                    user_cache: Optional[Dict[str, Any]]) -> str:
    """Resolve a message sender's IANA timezone from already-cached data ONLY.

    Prefers an explicit ``user_timezone`` on the message metadata (warm path carries it),
    then the client's in-memory ``user_cache`` (populated when the username was resolved),
    falling back to UTC. Deliberately cache-only and synchronous — no DB/API fetch — so
    the rebuild loop never adds a per-message uncached round-trip."""
    md = metadata or {}
    tz = md.get("user_timezone")
    if tz:
        return tz
    if user_id and isinstance(user_cache, dict):
        info = user_cache.get(user_id)
        if isinstance(info, dict) and info.get("timezone"):
            return info["timezone"]
    return "UTC"
