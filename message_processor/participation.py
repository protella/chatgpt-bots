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
import logging
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config import config, valid_emoji_name

logger = logging.getLogger(__name__)

# F27: cap the number of distinct conversation streams the burst-carry map tracks, so a
# long-lived process can't accumulate one pending bucket per (channel, author) forever.
_MAX_PENDING_KEYS = 512
# F27: how many earlier same-author messages a survivor may carry into one combined reply.
_MAX_BURST_CARRY = 3


def _ts_key(ts: Any) -> tuple:
    """Numeric (seconds, microseconds) sort key for a Slack ts — never lexical, so
    '9.0' sorts before '10.0' (F5 fix f)."""
    try:
        s, _, frac = str(ts).partition(".")
        return (int(s or 0), int((frac + "000000")[:6]))
    except (ValueError, TypeError):
        return (0, 0)


def _ts_seconds(ts: Any) -> float:
    """Float seconds for a Slack ts, for freshness-window arithmetic (F27)."""
    secs, micros = _ts_key(ts)
    return secs + micros / 1_000_000.0


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
    # F27: earlier messages from the SAME sender that this survivor carries forward, so its
    # single reply covers the whole same-author burst. Oldest-first, newest 3 at most.
    # Attached by evaluate() after validate_verdict returns (validate_verdict stays pure).
    burst_earlier: Optional[List[str]] = None


class ParticipationEngine:
    """Debounced wrapper around one utility-model judgment call."""

    def __init__(self, openai_client):
        self.openai_client = openai_client
        # conversation key -> newest pending message ts (debounce supersession marker).
        # F21: keyed per CONVERSATION, not per channel — a question in one thread must
        # never be silently dropped because an unrelated conversation posted something
        # newer elsewhere in the channel. F27: top-level streams are now keyed per SENDER
        # too, so two different people's unrelated top-level questions never collide.
        self._latest: Dict[str, str] = {}
        # F27: burst carry-forward. conversation key -> OrderedDict[ts -> text] of messages
        # seen in that stream but not yet consumed. A superseded evaluation leaves its entry
        # for the burst's survivor (the newest message) to collect, so ONE reply can cover
        # a same-author fast-follow rather than answering only the latest fragment. Bounded
        # by _MAX_PENDING_KEYS; each bucket self-drains when its survivor runs.
        self._pending: "OrderedDict[str, OrderedDict[str, str]]" = OrderedDict()

    @staticmethod
    def _conv_key(channel_id: str, ts: str, thread_root: Optional[str],
                  sender_id: Optional[str] = None) -> str:
        """Supersession scope. F21: thread replies key by their root (cross-author collapse
        in a thread is safe — the reply lands in-thread with full history). F27: top-level
        messages key per SENDER, so a same-author fast-follow supersedes (and gets carried
        into one combined reply) while two DIFFERENT people's unrelated top-level questions
        stay independent and are both answered."""
        if thread_root and thread_root != ts:
            return f"{channel_id}|{thread_root}"
        return f"{channel_id}|top|{sender_id or 'unknown'}"

    def note_arrival(self, channel_id: str, ts: Optional[str],
                     thread_root: Optional[str] = None,
                     sender_id: Optional[str] = None) -> None:
        """Register a message's ts as its conversation's newest — MONOTONICALLY (F5 fix b).

        Called at gate entry, BEFORE any await, so an older event delayed by memory/topic
        I/O can never overwrite a newer event's marker and win the debounce. Only a
        genuinely newer Slack ts advances the marker. F27: sender_id scopes the top-level
        stream key so a monotonic advance is per-author."""
        if not channel_id or not ts:
            return
        key = self._conv_key(channel_id, ts, thread_root, sender_id)
        current = self._latest.get(key)
        if current is None or _ts_key(ts) > _ts_key(current):
            self._latest[key] = ts

    def _register_pending(self, key: str, ts: str, text: Optional[str]) -> None:
        """F27: record (ts, text) in its conversation's pending bucket so a later survivor
        can carry it. Evicts the oldest bucket once the map exceeds _MAX_PENDING_KEYS."""
        bucket = self._pending.get(key)
        if bucket is None:
            bucket = OrderedDict()
            self._pending[key] = bucket
        bucket[ts] = text or ""
        self._pending.move_to_end(key)
        while len(self._pending) > _MAX_PENDING_KEYS:
            self._pending.popitem(last=False)

    def _collect_burst(self, key: str, own_ts: str, debounce_seconds: float) -> List[str]:
        """F27: called by the survivor of a debounce window to DRAIN its pending bucket.
        Collect-and-remove every pending entry in `key` strictly older than own_ts
        (oldest-first) plus own entry; drop entries older than own_ts − max(15, 5×debounce)
        seconds as stale leftovers (a survivor that never ran must not leak minutes-old
        texts into a fresh burst); cap the returned texts at the newest _MAX_BURST_CARRY,
        logging any further drop. The draining is unconditional (memory hygiene for every
        stream, thread or top-level); evaluate() decides whether to CARRY the result — only
        top-level survivors do, where the per-sender key guarantees the texts are same-author.
        Thread survivors call this to empty their bucket but discard the returned list."""
        bucket = self._pending.get(key)
        if not bucket:
            return []
        own_key = _ts_key(own_ts)
        window = max(15.0, 5.0 * float(debounce_seconds or 0.0))
        cutoff = _ts_seconds(own_ts) - window
        carried: List[tuple] = []  # (ts_key, text), strictly-older + fresh
        stale_dropped = 0
        for pts in list(bucket.keys()):
            if pts == own_ts:
                del bucket[pts]  # own entry — remove, never carry
                continue
            pkey = _ts_key(pts)
            if pkey >= own_key:
                continue  # newer/equal — leave for its own survivor
            text = bucket.pop(pts)  # collect-and-remove (removal prevents later leak)
            if _ts_seconds(pts) < cutoff:
                stale_dropped += 1
                continue
            carried.append((pkey, text))
        if not bucket:
            self._pending.pop(key, None)
        if stale_dropped:
            logger.debug(
                "F27: dropped %d stale pending entr%s from burst %s (older than freshness window)",
                stale_dropped, "y" if stale_dropped == 1 else "ies", key)
        carried.sort(key=lambda kt: kt[0])  # oldest-first
        texts = [t for _, t in carried]
        if len(texts) > _MAX_BURST_CARRY:
            dropped = len(texts) - _MAX_BURST_CARRY
            logger.debug(
                "F27: burst %s carried %d messages; keeping newest %d, dropping %d oldest",
                key, len(texts), _MAX_BURST_CARRY, dropped)
            texts = texts[-_MAX_BURST_CARRY:]
        return texts

    # ------------------------------------------------------------- evaluate

    async def evaluate(self, *, channel_id: str, ts: str, text: str,
                       sender_id: Optional[str] = None,
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
        """Debounced judgment. Returns None when superseded — a newer message in the SAME
        conversation (this thread, or this sender's top-level stream — F21/F27) arrived
        during the debounce window. F27: the survivor of a same-author TOP-LEVEL burst
        collects the superseded siblings' texts into burst_earlier so its ONE reply covers
        the whole burst; a superseded evaluation returns None but LEAVES its pending entry
        for that survivor. Activity in other conversations (or from other senders at top
        level) never supersedes.

        Burst CARRY is top-level-only: a top-level stream key (channel|top|<sender>) is
        per-author, so its collected siblings are guaranteed same-sender. A thread key
        (channel|root) still collapses cross-author (F21) and its survivor may be a
        DIFFERENT author than the superseded messages — carrying those would misattribute
        them, and the render sites label the carried text "the same sender". So thread
        survivors still DRAIN their bucket (load-bearing memory hygiene — a busy thread's
        bucket must not grow unbounded) but DISCARD the texts; in-thread coverage already
        works pre-F27 because the reply lands in-thread with full history."""
        key = self._conv_key(channel_id, ts, thread_root_ts, sender_id)
        is_top_level = not (thread_root_ts and thread_root_ts != ts)
        self.note_arrival(channel_id, ts, thread_root_ts, sender_id)  # monotonic; a stale caller can't clobber a newer marker
        self._register_pending(key, ts, text)  # F27: enroll before the await so the survivor can find us
        wait = max(0.0, float(getattr(config, "participation_debounce_seconds", 3.0)))
        if wait:
            await asyncio.sleep(wait)
        if self._latest.get(key) != ts:
            return None  # superseded — our pending entry stays for the burst's survivor

        # F27: we survived the debounce — always drain this stream's pending siblings (bucket
        # hygiene), but only CARRY them as a burst at top level, where the per-sender key
        # guarantees they are the same author. Thread survivors drain-and-discard.
        collected = self._collect_burst(key, ts, wait)
        burst_earlier = collected if is_top_level else []

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
            "burst_earlier": burst_earlier,
        }
        try:
            raw = await self.openai_client.classify_participation(text=text, signals=signals)
        except Exception:
            raw = None  # fail-safe: silence, never spam
        verdict = self.validate_verdict(raw)
        # F27: attach AFTER validate_verdict (which stays pure) so the survivor's reply can
        # be told about the earlier same-author messages it must also address.
        if burst_earlier:
            verdict.burst_earlier = burst_earlier
        return verdict

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
