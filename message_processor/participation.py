"""ParticipationEngine — Phase F decision engine for channel participation.

Replaces the one-word wake classifier with a judgment layer that decides, per
unprompted channel message, whether the bot should respond, react, stay silent,
or back off — using real channel context (ChannelPulse envelope), channel memory,
operator directives, and the bot's own recent participation rate.

Authority order (cheap → expensive), enforced in code not prompt:
  prefilters (message_events: own message / subtype / level=off / muted-thread /
  addressed-short-circuit / mentions_only) → debounce → ONE utility-model call → verdict.
  (The old hourly hard cap was retired in F17 — pacing is the model's judgment, not a ceiling.)

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
from message_processor import gate_vision

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

# Participation-backoff redesign (Layer 2): the taxonomy a `backoff` verdict carries so the
# engine can tell a passing "not now" from a durable "stop doing X here" — and never confuse
# a soft preference with an explicit settings change.
VALID_DIMENSIONS = ("reactions", "replies", "verbosity", "thread_participation")
VALID_DURABILITIES = ("momentary", "standing")
VALID_SCOPES = ("thread", "channel")
VALID_STRUCTURAL = ("none", "participation", "placement", "both")

MODE_TO_LEVEL = {"off": "off", "tag_only": "mentions_only", "auto_respond": "judicious"}
LEVEL_TO_MODE = {"off": "off", "mentions_only": "tag_only",
                 "judicious": "auto_respond", "active": "auto_respond"}


def _coerce_enum(value: Any, allowed: tuple) -> Optional[str]:
    """Lowercased value if it is one of `allowed`, else None (a malformed field never leaks)."""
    v = str(value or "").strip().lower()
    return v if v in allowed else None


def _coerce_memory_op(value: Any) -> str:
    """Normalize the `memory_op` field to none | add | delete | update:<id> | delete:<id>.

    `update:<id>`/`delete:<id>` target a specific channel-memory row. Bare `add`/`delete`
    (which carry no row id) are still accepted for backward-compatible parsing, but no longer
    drive a thread mute — thread-scoped feedback is guidance-only and persists nothing.
    Anything malformed degrades to "none" (no durable action)."""
    s = str(value or "none").strip().lower()
    if s in ("none", "add", "delete"):
        return s
    for op in ("update", "delete"):
        prefix = op + ":"
        if s.startswith(prefix):
            ident = s[len(prefix):].strip()
            if ident.isdigit():
                return f"{op}:{ident}"
    return "none"


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
    # Participation-backoff redesign (Layer 2): the taxonomy a `backoff` verdict carries.
    # All defaulted so a pre-redesign verdict ({action, emoji, placement, reason}) still parses
    # unchanged. `dimension` is which behavior the feedback is about; `durability` momentary vs
    # standing; `scope` "channel" (a channel-wide preference) vs "thread" (guidance for the
    # current message only — a thread-scoped aside persists nothing now that the per-thread mute
    # mechanism is gone); `guidance` the normalized preference text; `memory_op` the durable
    # record to make on CHANNEL memory (add / update:<id> / delete:<id> / none) — it is no
    # longer a thread-mute add/unmute verb; `structural_request` an explicit channel-settings
    # change that the main model, not the engine, applies via the gated
    # set_channel_participation tool.
    dimension: Optional[str] = None
    durability: Optional[str] = None
    scope: Optional[str] = None
    guidance: str = ""
    memory_op: str = "none"
    structural_request: str = "none"
    # F38: the `ack` bit is GONE. The classifier used to predict "this reply implies real
    # work" and the gate dropped a 👀 on the strength of that guess — before the model had
    # done anything, and often wrongly (it acked a passing comment). The 👀 is now staked by
    # the work itself, when a slow tool actually starts, and retracted if the work produces
    # nothing. A classifier cannot know that in advance, so it no longer tries.
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
                       channel_canvases: Optional[List[str]] = None,
                       channel_people: Optional[str] = None,
                       capabilities: Optional[str] = None,
                       attachments: Optional[str] = None,
                       images: Optional[List[Dict]] = None,
                       client: Any = None,
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

        # F40: the pixels, not the filename. Loaded HERE — after the supersession check — so a
        # superseded burst never downloads images for a verdict that is about to be discarded.
        # `image_status` tells the prompt the truth: seen, or attached-but-unavailable. Any
        # failure degrades to a text-only judgment; it must never turn into silence.
        image_parts, image_status = [], gate_vision.NONE
        if images and client is not None:
            try:
                image_parts, image_status = await gate_vision.load_for_gate(client, images)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Gate vision failed, judging on text alone: {e}")
                image_parts, image_status = [], gate_vision.UNAVAILABLE
        elif images:
            image_status = gate_vision.UNAVAILABLE

        signals = {
            "image_status": image_status,
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
            # F36: the gate never sees tool schemas, so without this a passive
            # "we should update the devops agenda" reads as idle chatter and it
            # stays silent — the main model never gets a turn to notice the canvas.
            "channel_canvases": channel_canvases,
            "channel_people": channel_people,
            "capabilities": capabilities,
            "attachments": attachments,
            "burst_earlier": burst_earlier,
        }
        try:
            # `images` only rides when there ARE images: the text-only call keeps its exact old
            # shape, so nothing that never sees a picture changes behaviour by one token.
            call_kwargs = {"text": text, "signals": signals}
            if image_parts:
                call_kwargs["images"] = image_parts
            raw = await self.openai_client.classify_participation(**call_kwargs)
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
    def _coerce_emoji(raw: dict, force_allowlist: bool) -> Optional[str]:
        """Resolve the verdict's emoji. F20: by default any syntactically valid standard emoji
        name is accepted; a REACTION_EMOJIS allowlist, when set, constrains the choice. For a
        REACT verdict (force_allowlist=True) an off-list/garbage name falls back to the first
        allowlisted emoji (the old wake gate's choice), and None means "downgrade to ignore".
        For a backoff ACK (force_allowlist=False) an off-list/garbage/empty name means simply
        no ack — a reaction is never forced onto the sender who just asked for restraint."""
        allow = [e.strip().strip(":") for e in (getattr(config, "reaction_emojis", None) or [])
                 if e and e.strip().strip(":")]
        emoji = str(raw.get("emoji") or "").strip().strip(":")
        if allow:
            if emoji in allow:
                return emoji
            return allow[0] if force_allowlist else None
        return emoji if valid_emoji_name(emoji) else None

    @staticmethod
    def validate_verdict(raw: Any) -> ParticipationVerdict:
        """Coerce a raw model dict into a safe verdict. Anything malformed →
        ignore. F20: by default any syntactically valid standard emoji name is
        accepted (a garbage name downgrades the react to ignore); when a
        REACTION_EMOJIS allowlist is set, an off-list emoji falls back to the
        first allowlisted emoji (the choice the old wake gate made).

        A `backoff` verdict additionally carries the participation-feedback taxonomy
        (dimension/durability/scope/guidance/memory_op/structural_request); each field is
        parsed defensively and defaults so the caller never has to guard for absence."""
        if not isinstance(raw, dict):
            return ParticipationVerdict(action="ignore", reason="malformed-verdict")
        action = str(raw.get("action") or "").strip().lower()
        if action not in VALID_ACTIONS:
            return ParticipationVerdict(action="ignore", reason="invalid-action")
        emoji = None
        if action == "react":
            emoji = ParticipationEngine._coerce_emoji(raw, force_allowlist=True)
            if emoji is None:
                return ParticipationVerdict(action="ignore", reason="react-no-valid-emoji")
        elif action == "backoff":
            # An OPTIONAL ack emoji; absent/garbage simply means no ack.
            emoji = ParticipationEngine._coerce_emoji(raw, force_allowlist=False)
        placement = str(raw.get("placement") or "thread").strip().lower()
        if placement not in VALID_PLACEMENTS:
            placement = "thread"
        # F38: an `ack` key from a stale prompt (or a model that remembers the old contract)
        # is simply ignored — the field is gone from the verdict.
        verdict = ParticipationVerdict(
            action=action, emoji=emoji, placement=placement,
            reason=str(raw.get("reason") or "")[:300],
        )
        if action == "backoff":
            verdict.dimension = _coerce_enum(raw.get("dimension"), VALID_DIMENSIONS)
            verdict.durability = _coerce_enum(raw.get("durability"), VALID_DURABILITIES)
            verdict.scope = _coerce_enum(raw.get("scope"), VALID_SCOPES)
            verdict.guidance = str(raw.get("guidance") or "").strip()[:300]
            verdict.memory_op = _coerce_memory_op(raw.get("memory_op"))
            structural = str(raw.get("structural_request") or "none").strip().lower()
            verdict.structural_request = structural if structural in VALID_STRUCTURAL else "none"
        return verdict
