"""ParticipationEngine — Phase F decision engine for channel participation.

Replaces the one-word wake classifier with a judgment layer that decides, per
unprompted channel message, whether the bot should respond, react, stay silent,
or back off — using real channel context (ChannelPulse envelope), channel memory,
operator directives, and the bot's own recent participation rate.

Authority order (cheap → expensive), enforced in code not prompt:
  prefilters (message_events: own message / subtype / level=off / snoozed /
  addressed-short-circuit / mentions_only) → hard hourly throttle → debounce →
  ONE utility-model call → verdict.

@mentions, name-wakes, 1:1 threads, and DMs NEVER reach this engine — they are
answered directly (told to be quiet ≠ deaf).

Legacy compatibility — participation levels vs. response_mode:
  response_mode "off"          ≡ level "off"
  response_mode "tag_only"     ≡ level "mentions_only"
  response_mode "auto_respond" ≡ level "judicious" (default engine strictness)
  level "active" has no legacy equivalent (maps back to "auto_respond")
A row's participation_level, when set, WINS over its response_mode. The channel
modal writes both columns in lockstep so legacy readers stay consistent.
"""
from __future__ import annotations

import asyncio
import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import config

VALID_ACTIONS = ("respond", "react", "ignore", "backoff")
VALID_PLACEMENTS = ("thread", "channel")
VALID_LEVELS = ("off", "mentions_only", "judicious", "active")

MODE_TO_LEVEL = {"off": "off", "tag_only": "mentions_only", "auto_respond": "judicious"}
LEVEL_TO_MODE = {"off": "off", "mentions_only": "tag_only",
                 "judicious": "auto_respond", "active": "auto_respond"}


def resolve_participation_level(channel_settings: Optional[Dict[str, Any]]) -> str:
    """Effective participation level for a channel.

    participation_level (if set) wins; else derive from the row's response_mode;
    else from the global default mode. Unknown values degrade to mentions_only
    (the safe pre-F behavior)."""
    cs = channel_settings or {}
    level = (cs.get("participation_level") or "").strip().lower()
    if level in VALID_LEVELS:
        return level
    mode = (cs.get("response_mode")
            or getattr(config, "channel_response_mode", "tag_only")
            or "tag_only").strip().lower()
    return MODE_TO_LEVEL.get(mode, "mentions_only")


def is_snoozed(channel_settings: Optional[Dict[str, Any]],
               now: Optional[datetime.datetime] = None) -> bool:
    """True while the channel's snoozed_until timestamp is in the future.

    Snooze silences UNPROMPTED participation only — the addressed/mention path
    never consults this."""
    raw = (channel_settings or {}).get("snoozed_until")
    if not raw:
        return False
    try:
        expiry = datetime.datetime.fromisoformat(str(raw))
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=datetime.timezone.utc)
    except (ValueError, TypeError):
        return False  # malformed → treat as not snoozed (fail open, mentions unaffected)
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return expiry > now


def snooze_expiry_iso(hours: Optional[float] = None,
                      now: Optional[datetime.datetime] = None) -> str:
    """ISO-8601 UTC expiry for a new snooze (deterministic given `now`)."""
    hours = hours if hours is not None else getattr(config, "participation_snooze_hours", 4)
    now = now or datetime.datetime.now(datetime.timezone.utc)
    return (now + datetime.timedelta(hours=float(hours))).isoformat(timespec="seconds")


@dataclass
class ParticipationVerdict:
    action: str = "ignore"
    emoji: Optional[str] = None
    placement: str = "thread"
    reason: str = ""


class ParticipationEngine:
    """Debounced wrapper around one utility-model judgment call."""

    def __init__(self, openai_client):
        self.openai_client = openai_client
        # channel -> newest pending message ts (debounce supersession marker)
        self._latest: Dict[str, str] = {}

    # ---------------------------------------------------------------- rails

    def hourly_cap(self, level: str) -> int:
        base = int(getattr(config, "max_unprompted_replies_per_hour", 6))
        return base * 2 if level == "active" else base

    def over_throttle(self, pulse, channel_id: str, level: str) -> bool:
        """Hard self-throttle: at/over the unprompted cap the engine is not even
        called (addressed messages never pass through here, so mentions are
        never throttled)."""
        if pulse is None:
            return False
        try:
            return pulse.unprompted_count_last_hour(channel_id) >= self.hourly_cap(level)
        except Exception:
            return False

    # ------------------------------------------------------------- evaluate

    async def evaluate(self, *, channel_id: str, ts: str, text: str,
                       sender_name: Optional[str] = None,
                       is_thread_reply: bool = False,
                       level: str = "judicious",
                       directives: Optional[str] = None,
                       memory_facts: Optional[List[Dict[str, Any]]] = None,
                       channel_activity: Optional[str] = None,
                       unprompted_last_hour: int = 0,
                       name_hit: bool = False,
                       snoozed: bool = False,
                       sender_is_bot: bool = False) -> Optional[ParticipationVerdict]:
        """Debounced judgment. Returns None when superseded — a newer message in
        the same channel arrived during the debounce window, and ITS evaluation
        (whose envelope includes this message) covers the batch."""
        self._latest[channel_id] = ts
        wait = max(0.0, float(getattr(config, "participation_debounce_seconds", 3.0)))
        if wait:
            await asyncio.sleep(wait)
        if self._latest.get(channel_id) != ts:
            return None

        signals = {
            "sender_name": sender_name,
            "is_thread_reply": is_thread_reply,
            "strictness": level,
            "directives": directives,
            "memory_facts": memory_facts or [],
            "channel_activity": channel_activity,
            "unprompted_last_hour": int(unprompted_last_hour),
            "hourly_cap": self.hourly_cap(level),
            "name_hit": bool(name_hit),
            "snoozed": bool(snoozed),
            "sender_is_bot": bool(sender_is_bot),
        }
        try:
            raw = await self.openai_client.classify_participation(text=text, signals=signals)
        except Exception:
            raw = None  # fail-safe: silence, never spam
        return self.validate_verdict(raw)

    # ------------------------------------------------------------- validate

    @staticmethod
    def validate_verdict(raw: Any) -> ParticipationVerdict:
        """Coerce a raw model dict into a safe verdict. Anything malformed →
        ignore; a react emoji outside the allowlist falls back to the first
        allowlisted emoji (same choice the old wake gate made)."""
        if not isinstance(raw, dict):
            return ParticipationVerdict(action="ignore", reason="malformed-verdict")
        action = str(raw.get("action") or "").strip().lower()
        if action not in VALID_ACTIONS:
            return ParticipationVerdict(action="ignore", reason="invalid-action")
        emoji = None
        if action == "react":
            allow = list(getattr(config, "reaction_emojis", None) or ["eyes"])
            emoji = str(raw.get("emoji") or "").strip().strip(":")
            if emoji not in allow:
                emoji = allow[0]
        placement = str(raw.get("placement") or "thread").strip().lower()
        if placement not in VALID_PLACEMENTS:
            placement = "thread"
        return ParticipationVerdict(
            action=action, emoji=emoji, placement=placement,
            reason=str(raw.get("reason") or "")[:300],
        )
