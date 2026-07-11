"""ParticipationEngine — Phase F decision engine for channel participation.

Replaces the one-word wake classifier with a judgment layer that decides, per
unprompted channel message, whether the bot should respond, react, stay silent,
or back off — using real channel context (ChannelPulse envelope), channel memory,
operator directives, and the bot's own recent participation rate.

Authority order (cheap → expensive), enforced in code not prompt:
  prefilters (message_events: own message / subtype / level=off / muted-thread /
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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config import config, valid_emoji_name

def _ts_key(ts: Any) -> tuple:
    """Numeric (seconds, microseconds) sort key for a Slack ts — never lexical, so
    '9.0' sorts before '10.0' (F5 fix f)."""
    try:
        s, _, frac = str(ts).partition(".")
        return (int(s or 0), int((frac + "000000")[:6]))
    except (ValueError, TypeError):
        return (0, 0)


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


def render_capabilities_line(mcp_manager: Any = None) -> Optional[str]:
    """Semicolon-joined inventory of the assistant's own tools/data sources, for
    the participation classifier (F11). Pure function of already-loaded config +
    mcp_manager.servers — zero I/O, deterministic per process.

    - "web search" when config.enable_web_search;
    - "image generation and editing" (always true for this bot);
    - "analyzing images and documents shared in chat" (F14b — vision/document flows
      are core, so the classifier weighs "what do we think?" about an attached artifact);
    - one entry per MCP server when config.mcp_enabled_default AND mcp_manager is
      present AND has servers: each server's `server_description` (from
      mcp_config.json) falling back to its label. Servers iterate in insertion
      order (stable per process → cache-friendly).

    Nothing is hardcoded for any specific server. Returns None when the list would
    be empty (never happens in practice — image gen is unconditional — but guard)."""
    caps: List[str] = []
    if getattr(config, "enable_web_search", False):
        caps.append("web search")
    caps.append("image generation and editing")
    caps.append("analyzing images and documents shared in chat")
    if (getattr(config, "mcp_enabled_default", False)
            and mcp_manager is not None):
        try:
            has_servers = mcp_manager.has_mcp_servers()
        except Exception:
            has_servers = False
        if has_servers:
            for label, server_config in mcp_manager.servers.items():
                desc = (server_config or {}).get("server_description") or label
                caps.append(str(desc))
    if not caps:
        return None
    return "; ".join(caps)


@dataclass
class ParticipationVerdict:
    action: str = "ignore"
    emoji: Optional[str] = None
    placement: str = "thread"
    reason: str = ""
    # F19: "I'm looking at it" acknowledgment. Meaningful only with action="respond" —
    # set when the reply is worth giving AND implies real work (attachments, data/MCP
    # lookups, multi-step tools, long-form output). The gate drops ACK_REACTION_EMOJI on
    # the triggering message before dispatching. Coerced safely (absent/malformed → False).
    ack: bool = False


class ParticipationEngine:
    """Debounced wrapper around one utility-model judgment call."""

    def __init__(self, openai_client):
        self.openai_client = openai_client
        # conversation key -> newest pending message ts (debounce supersession marker).
        # F21: keyed per CONVERSATION, not per channel — a question in one thread must
        # never be silently dropped because an unrelated conversation posted something
        # newer elsewhere in the channel.
        self._latest: Dict[str, str] = {}

    @staticmethod
    def _conv_key(channel_id: str, ts: str, thread_root: Optional[str]) -> str:
        """F21 supersession scope: thread replies key by their root; all top-level
        messages share one "top" stream (a rapid top-level burst still collapses to
        its newest, whose envelope covers the rest)."""
        if thread_root and thread_root != ts:
            return f"{channel_id}|{thread_root}"
        return f"{channel_id}|top"

    def note_arrival(self, channel_id: str, ts: Optional[str],
                     thread_root: Optional[str] = None) -> None:
        """Register a message's ts as its conversation's newest — MONOTONICALLY (F5 fix b).

        Called at gate entry, BEFORE any await, so an older event delayed by memory/topic
        I/O can never overwrite a newer event's marker and win the debounce. Only a
        genuinely newer Slack ts advances the marker."""
        if not channel_id or not ts:
            return
        key = self._conv_key(channel_id, ts, thread_root)
        current = self._latest.get(key)
        if current is None or _ts_key(ts) > _ts_key(current):
            self._latest[key] = ts

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
                       sender_is_bot: bool = False,
                       channel_topic: Optional[str] = None,
                       capabilities: Optional[str] = None,
                       attachments: Optional[str] = None,
                       pulse: Any = None,
                       thread_root_ts: Optional[str] = None) -> Optional[ParticipationVerdict]:
        """Debounced judgment. Returns None when superseded — a newer message in
        the SAME conversation (this thread, or the top-level stream — F21) arrived
        during the debounce window, and ITS evaluation (whose tail/envelope includes
        this message) covers the batch. Activity in other conversations never
        supersedes."""
        self.note_arrival(channel_id, ts, thread_root_ts)  # monotonic; a stale caller can't clobber a newer marker
        wait = max(0.0, float(getattr(config, "participation_debounce_seconds", 3.0)))
        if wait:
            await asyncio.sleep(wait)
        if self._latest.get(self._conv_key(channel_id, ts, thread_root_ts)) != ts:
            return None

        # F5: render the thread tail HERE (after the debounce + supersession check) —
        # pure in-memory, zero latency, reflecting thread state at classification time.
        thread_tail = None
        if pulse is not None and thread_root_ts:
            try:
                thread_tail = pulse.render_thread_tail(
                    channel_id, thread_root_ts, before_ts=ts) or None
            except Exception:
                thread_tail = None

        signals = {
            "sender_name": sender_name,
            "is_thread_reply": is_thread_reply,
            "strictness": level,
            "directives": directives,
            "memory_facts": memory_facts or [],
            "channel_activity": channel_activity,
            "thread_tail": thread_tail,
            "unprompted_last_hour": int(unprompted_last_hour),
            "name_hit": bool(name_hit),
            "sender_is_bot": bool(sender_is_bot),
            "channel_topic": channel_topic,
            "capabilities": capabilities,
            "attachments": attachments,
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
        ignore. F20: by default any syntactically valid standard emoji name is
        accepted (a garbage name downgrades the react to ignore); when a
        REACTION_EMOJIS allowlist is set, an off-list emoji falls back to the
        first allowlisted emoji (the choice the old wake gate made)."""
        if not isinstance(raw, dict):
            return ParticipationVerdict(action="ignore", reason="malformed-verdict")
        action = str(raw.get("action") or "").strip().lower()
        if action not in VALID_ACTIONS:
            return ParticipationVerdict(action="ignore", reason="invalid-action")
        emoji = None
        if action == "react":
            allow = [e.strip().strip(":") for e in (getattr(config, "reaction_emojis", None) or []) if e and e.strip().strip(":")]
            emoji = str(raw.get("emoji") or "").strip().strip(":")
            if allow:
                if emoji not in allow:
                    emoji = allow[0]
            elif not valid_emoji_name(emoji):
                return ParticipationVerdict(action="ignore", reason="react-no-valid-emoji")
        placement = str(raw.get("placement") or "thread").strip().lower()
        if placement not in VALID_PLACEMENTS:
            placement = "thread"
        # F19: coerce ack to a plain bool (absent/malformed → False); only respond turns
        # carry it — a react/ignore/backoff verdict never acks.
        ack = action == "respond" and raw.get("ack") is True
        return ParticipationVerdict(
            action=action, emoji=emoji, placement=placement,
            reason=str(raw.get("reason") or "")[:300], ack=ack,
        )
