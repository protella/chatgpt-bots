"""ParticipationEngine — the binary wake gate.

ONE QUESTION, ONE BIT. This engine decides whether the full responder runs on an unprompted
channel message. That is all it decides. It does not choose words, reactions, placement, or
settings; it does not rank how much value a reply would add; it does not report why. Those were
the rich gate's job, and the rich gate was wrong about them in a way that could not be fixed by
better prompting: it made visible-action decisions from a thin slice of context, before the model
that actually had the context ever got a turn.

WHY A BIT IS BETTER THAN A VERDICT. Every extra field the old verdict carried became a control
bus. `action` branched the caller four ways; `emoji` placed a reaction the responder then had to
be told about; `reason` was forwarded into the responder's prompt and pre-argued the turn;
`memory_op` and the backoff taxonomy wrote to the database from a classifier that had seen one
message. Each field was individually defensible and collectively a second, dumber assistant
sitting in front of the real one. Deleting them is the point of this commit, not a side effect.

THE TRADE IS DELIBERATE. A one-bit gate is worse at deciding "is this worth answering", because
it is judging on less. It compensates by being generous: if a full turn could plausibly be
useful, it wakes the responder, which has the whole thread, the tools, and the option of saying
nothing at all (declared silence). A false wake costs one utility call and ends in silence; a
false sleep loses the answer entirely. So the gate leans toward waking, and the responder — which
can actually tell — owns the decision to speak.

PROMPT INPUTS, and only these: the canonical channel-steering snapshot (commit 5, byte-for-byte
identical to the responder's copy), and the ordered source messages of this debounce cohort. No
pulse envelope, no thread tail, no people line, no topic, no canvases, no summary, no
capabilities, no name-hit, no level, no emoji palette, and no pixels. Those inputs existed to
support judgments the gate no longer makes.

Authority order (cheap → expensive), enforced in code and not by prompt:
  prefilters (message_events: own message / subtype / level=off / addressed short-circuit /
  mentions_only) → debounce cohort → ONE utility-model call → one bit.

@mentions, 1:1 thread continuations, and DMs NEVER reach this engine — they are answered
directly. Told to be quiet is not the same as deaf.

Legacy compatibility — participation levels vs. response_mode:
  response_mode "off"          ≡ level "off"
  response_mode "tag_only"     ≡ level "mentions_only"
  response_mode "auto_respond" ≡ level "on"
A row's participation_level, when set, WINS over its response_mode. The channel modal writes
both columns in lockstep so a rollback's legacy reader stays consistent.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config import config
from message_processor import participation_telemetry

# ONE comparator, shared with the stale-send guard. It lived here in a second copy, and two
# definitions of "which message is newer" is one too many for a codebase where both the cohort
# collapse and the send guard turn on that question.
from message_processor.stale_send_guard import primary_scope_key as _primary_scope_key
from message_processor.stale_send_guard import ts_key as _ts_key

logger = logging.getLogger(__name__)


# The three levels a channel can actually be in under a binary gate.
#
# `judicious` and `active` are gone. They were two dials on a rich gate that weighed "is this
# worth saying" — a question the binary gate does not ask and cannot answer, because it decides
# only whether the responder RUNS. Keeping both names would have promised a distinction the code
# no longer makes, so they migrate to one honest value.
#
#   off            — no channel response at all, INCLUDING an explicit @mention.
#   mentions_only  — a real @mention goes straight to the responder; a bare-name message is
#                    judged by the gate; a deterministic 1:1 continuation stays direct.
#   on             — all otherwise-eligible ambient and name traffic is judged by the gate.
VALID_LEVELS = ("off", "mentions_only", "on")

# The legacy response_mode column is still dual-written so a rollback can read it.
MODE_TO_LEVEL = {"off": "off", "tag_only": "mentions_only", "auto_respond": "on"}
LEVEL_TO_MODE = {"off": "off", "mentions_only": "tag_only", "on": "auto_respond"}


# The two levels that were merged into `on`. A pre-deploy modal still offers them, and a
# rollback-and-forward could reintroduce them, so the mapping outlives the migration that ran once.
_MERGED_INTO_ON = ("judicious", "active")


def normalize_legacy_level(value: Any) -> Any:
    """Map a retired level onto the one that replaced it, leaving everything else alone.

    `judicious` and `active` both meant "the gate judges every message", which is what `on` means
    now — so a submission carrying one of them is honoured rather than refused or written through
    verbatim. Anything else (including 'inherit' and None) passes through untouched."""
    if isinstance(value, str) and value.strip().lower() in _MERGED_INTO_ON:
        return "on"
    return value


def resolve_participation_level(channel_settings: Optional[Dict[str, Any]]) -> str:
    """Effective participation level for a channel.

    participation_level (if set) wins; else derive from the row's response_mode; else from the
    global default mode. Unknown values degrade to mentions_only — the quiet direction.

    ABSENT and PRESENT-BUT-INVALID are not the same thing, and conflating them escalates. An
    absent level means "this channel never chose one", so the legacy response_mode (or the global
    default) is the honest answer. A level that is present but not one we recognise — a
    `judicious` row the migration failed to reach, a hand-edited value, a future level after a
    rollback — is a channel that DID choose something we cannot honour, and falling through to
    response_mode would read `auto_respond` and resolve it to `on`. That turns an unreadable
    setting into the most talkative one available. Unrecognised means quiet."""
    cs = channel_settings or {}
    raw_level = cs.get("participation_level")
    level = (raw_level or "").strip().lower()
    if level in VALID_LEVELS:
        return level
    if level:
        logger.warning(
            "Unrecognised participation_level %r — treating this channel as mentions_only", level)
        return "mentions_only"
    mode = (cs.get("response_mode")
            or getattr(config, "channel_response_mode", "tag_only")
            or "tag_only").strip().lower()
    return MODE_TO_LEVEL.get(mode, "mentions_only")


def render_capabilities_line(mcp_manager: Any = None) -> Optional[str]:
    """Semicolon-joined inventory of the assistant's own tools/data sources.

    NO RUNTIME CALLERS. It existed so the rich gate could weigh whether the assistant was
    well-suited to answer an open question — an answerability judgment the binary gate does not
    make, because it decides only whether the responder RUNS and the responder knows its own tools.
    Kept for one release (it dies in the cleanup commit) so nothing else that might want an honest
    capability inventory has to reinvent it.

    Pure function of already-loaded config + mcp_manager.servers — zero I/O, deterministic per
    process.

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


# How many distinct edit-supersession MARKS to remember. Marks are bookkeeping about messages
# that have already been handled elsewhere, not enrolled messages, so bounding them can never
# discard something waiting for a turn. (The cohort map below is deliberately NOT bounded — see
# `_enroll_source`.)
_MAX_SUPERSESSION_KEYS = 512


# How an attachment is described to the gate: name plus KIND, and nothing else. Defined here,
# beside the record that carries it, because two places need to agree — the Slack facade builds
# these strings and the gate classifies on them, and a format invented at each end is how
# "captionless image" quietly starts matching a spreadsheet.
IMAGE_KIND = "image"
FILE_KIND = "file"


def describe_attachment(name: Optional[str], mimetype: Optional[str]) -> str:
    """One attachment as `name (kind)`. Names and types only — never content."""
    kind = IMAGE_KIND if str(mimetype or "").startswith("image/") else FILE_KIND
    return f"{name or 'file'} ({kind})"


def is_image_descriptor(descriptor: Any) -> bool:
    """Whether a descriptor built by ``describe_attachment`` names an image."""
    return str(descriptor or "").endswith(f"({IMAGE_KIND})")


@dataclass(frozen=True)
class SourceMessage:
    """ONE message the gate is judging, as typed data rather than prose.

    The old gate flattened a burst into a list of quoted strings and pasted them into the prompt
    with a sentence explaining what they were. That lost the sender, the time, and the topology —
    exactly the facts that decide whether a burst is one thought or two people talking — and the
    same lost detail then had to be re-invented as metadata prose for the responder. A record
    keeps them, so the classifier prompt and the responder's input can both be built from it.

    `attachments` carries names and types ONLY. No pixels, and no generated descriptions: the
    binary gate does not look at images (see the module docstring), and a description written by
    another model is a claim about content this gate cannot check.
    """

    ts: str
    text: str = ""
    sender_id: Optional[str] = None
    sender_name: Optional[str] = None
    sender_type: Optional[str] = None
    thread_root_ts: Optional[str] = None
    attachments: Tuple[str, ...] = ()
    # Present only for an edit: what it said before, what it says now, and whether the assistant
    # had already replied. Intrinsic to THIS message — not general channel history.
    edit: Optional[Dict[str, Any]] = None

    @property
    def is_thread_reply(self) -> bool:
        return bool(self.thread_root_ts and self.thread_root_ts != self.ts)


def source_from_message(message: Any) -> SourceMessage:
    """A SourceMessage from a dispatched Message.

    Lives here, beside the record it builds, so the queue drain and the gate cannot disagree about
    what a source IS — the drain needs this to hand a coalesced batch to the gate, and a second
    mapping written at the call site is how the two would drift."""
    meta = getattr(message, "metadata", None) or {}
    return SourceMessage(
        ts=str(meta.get("ts") or getattr(message, "thread_id", "") or ""),
        text=getattr(message, "text", "") or "",
        sender_id=getattr(message, "user_id", None),
        sender_name=meta.get("user_real_name") or meta.get("username"),
        sender_type=meta.get("sender_type"),
        thread_root_ts=getattr(message, "thread_id", None),
        attachments=tuple(meta.get("participation_attachments") or ()),
    )


@dataclass(frozen=True)
class WakeDecision:
    """The whole output of the gate: one bit.

    Nothing else, on purpose. A confidence would need a threshold nobody has measured. A reason
    would become the next control bus — the rich gate's `reason` was forwarded into the
    responder's prompt, where it pre-argued the turn and neutered the responder's own option to
    stay silent — and it would carry a summary of someone's message into the telemetry besides.
    """

    wake: bool


@dataclass
class GateEvaluation:
    """What ONE evaluation produced: the decision, or — when there is nothing to act on — why.

    `evaluate()` used to return a verdict-or-None, which left the caller unable to tell a cohort
    collapse from an edit cancellation from a provider outage. It could only find out by reading
    the engine's log lines, which is a side channel between two layers of one call: the caller
    owns the turn's single terminal event and has to be able to say what ended it.

    `decision` is None whenever there is no bit to act on. `decline_cause` says which kind of
    nothing it was — superseded | edit_superseded | classifier_error. Note the difference from
    the rich gate: a classifier failure no longer manufactures a decision. It produces None, and
    the caller ends the turn as `none` rather than scoring a fail-safe silence as judgment.

    `sources` is the cohort this evaluation actually judged, oldest first. It is returned (not
    just consumed) because the caller stamps it on the surviving Message so the responder answers
    the whole burst rather than only its newest fragment.
    """

    decision: Optional[WakeDecision] = None
    decline_cause: Optional[str] = None
    # The model call alone, in ms. NOT the gate's wall time, which is mostly debounce.
    classifier_ms: Optional[int] = None
    sources: Tuple[SourceMessage, ...] = ()


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
        # THE COHORT MAP: conversation key -> {ts: SourceMessage} for messages enrolled in this
        # stream and not yet judged. A superseded evaluation LEAVES its record for the cohort's
        # survivor (the newest message) to collect, so one turn answers the whole burst instead of
        # only its newest fragment.
        #
        # Deliberately UNBOUNDED, and that is a correctness requirement rather than an oversight.
        # It used to evict the oldest bucket past a cap and drop entries older than a freshness
        # window — both of which could silently discard a message somebody had actually sent, in
        # the one data structure whose whole job is not to lose it. Buckets are removed when their
        # survivor drains them (`_drain_cohort`) or when a stream is cancelled
        # (`discard_source`); a stream with an enrolled message always has a survivor, because the
        # newest ts of any cohort survives its own debounce unless a newer one takes over as
        # survivor.
        self._cohorts: "OrderedDict[str, OrderedDict[str, SourceMessage]]" = OrderedDict()
        # F52: messages whose in-flight evaluation an EDIT has explicitly cancelled. An edit
        # keeps the SAME ts, so a newer arrival can never supersede the original the ordinary
        # way (note_arrival is monotonic on ts); supersede() marks it here and evaluate() drops
        # the stale original — the edit's OWN re-evaluation, which carries edit context, is
        # exempt. conv key -> set of superseded ts. Bounded by _MAX_SUPERSESSION_KEYS.
        self._edit_superseded: "OrderedDict[str, set]" = OrderedDict()

    @staticmethod
    def _conv_key(channel_id: str, ts: str, thread_root: Optional[str],
                  sender_id: Optional[str] = None) -> str:
        """Supersession scope, from the shared scope helper (stale_send_guard).

        F21: thread replies key by their root — cross-author collapse in a thread is safe,
        since the reply lands in-thread with full history. F27: top-level messages key per
        SENDER, so a same-author fast-follow supersedes (and is carried into one combined
        reply) while two DIFFERENT people's unrelated top-level questions stay independent and
        are both answered. The send guard asks the same question about the same population, so
        it reads from the same table rather than a copy that can drift."""
        return _primary_scope_key(channel_id, ts, thread_root, sender_id)

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

    def supersede(self, channel_id: str, ts: Optional[str],
                  thread_root: Optional[str] = None,
                  sender_id: Optional[str] = None) -> None:
        """F52: cancel a message's in-flight participation evaluation because it was EDITED and
        the edit will be handled elsewhere — Slack's app_mention for a mention-added edit, or a
        fresh edit-context evaluation for a meaning edit. The original evaluation keyed on the
        SAME ts and so can never be superseded by a newer arrival; mark it here and evaluate()
        drops it, exactly as a newer burst message would. The edit's OWN re-evaluation carries
        edit context and is exempt. Idempotent; bounded by _MAX_SUPERSESSION_KEYS."""
        if not channel_id or not ts:
            return
        key = self._conv_key(channel_id, ts, thread_root, sender_id)
        bucket = self._edit_superseded.get(key)
        if bucket is None:
            bucket = set()
            self._edit_superseded[key] = bucket
        bucket.add(str(ts))
        self._edit_superseded.move_to_end(key)
        while len(self._edit_superseded) > _MAX_SUPERSESSION_KEYS:
            self._edit_superseded.popitem(last=False)

    def _consume_edit_supersession(self, key: str, ts: str) -> bool:
        """True (and clears the mark) iff `ts` was explicitly superseded by an edit for `key`.
        Consumed so a leaked mark can never affect a future message (ts is unique per message)."""
        bucket = self._edit_superseded.get(key)
        if bucket and str(ts) in bucket:
            bucket.discard(str(ts))
            if not bucket:
                self._edit_superseded.pop(key, None)
            return True
        return False

    def _enroll_source(self, key: str, source: SourceMessage) -> None:
        """Record a source in its conversation's cohort so a later survivor carries it.

        No cap and no eviction: see `self._cohorts`. Re-enrolling the same ts (an edit keeps its
        original timestamp) REPLACES the record, so the cohort holds the current text rather than
        two versions of one message."""
        bucket = self._cohorts.get(key)
        if bucket is None:
            bucket = OrderedDict()
            self._cohorts[key] = bucket
        bucket[source.ts] = source
        self._cohorts.move_to_end(key)

    def _drain_cohort(self, key: str, own_ts: str) -> Tuple[SourceMessage, ...]:
        """Called by the survivor of a debounce window: take every source in this stream up to and
        including its own, oldest first, and remove them from the map.

        Strictly-newer entries are LEFT behind — they belong to a later survivor, and taking them
        would judge a message whose own debounce has not finished. Everything else goes into this
        cohort whatever its age: the freshness window that used to drop old entries was a silent
        message-loss path, and a stale enrollment can only exist if its own evaluation never ran,
        which is a bug to see rather than to paper over.
        """
        bucket = self._cohorts.get(key)
        if not bucket:
            return ()
        own_key = _ts_key(own_ts)
        taken: List[Tuple[Tuple[int, int], SourceMessage]] = []
        for pts in list(bucket.keys()):
            if _ts_key(pts) > own_key:
                continue                      # newer — leave it for its own survivor
            taken.append((_ts_key(pts), bucket.pop(pts)))
        if not bucket:
            self._cohorts.pop(key, None)
        taken.sort(key=lambda pair: pair[0])   # oldest first: the order they were said in
        return tuple(source for _, source in taken)

    def discard_source(self, channel_id: str, ts: Optional[str],
                       thread_root: Optional[str] = None,
                       sender_id: Optional[str] = None) -> None:
        """Withdraw ONE cancelled source from its conversation, leaving everything else standing.

        The cohort map is unbounded, so a cancelled evaluation has to clean up after itself rather
        than wait to be evicted. What it must NOT do is clear the conversation: a cohort is shared
        by every message in that stream, and its other members are live evaluations sleeping
        through their own debounce.

        Both directions of that mistake lose messages, and neither is visible afterwards:

          * cancel the NEWEST and drop the bucket — the older sleepers' sources are gone, and
            `_latest` still names the cancelled ts, so when a sleeper wakes it finds itself
            superseded by a message that no longer exists and returns nothing. The whole stream
            evaporates.
          * cancel an OLDER one and drop the bucket — the live newer survivor loses the very
            sources it was going to answer for, and never learns they existed.

        So: remove our own record, and if we were the debounce marker, hand that role to the
        newest source still enrolled (or clear it when we were the last one out). Moving `_latest`
        backwards is safe here and only here — `note_arrival` is monotonic precisely so a stale
        ARRIVAL cannot do this, while a withdrawal genuinely means the newer message is gone."""
        if not channel_id or not ts:
            return
        key = self._conv_key(channel_id, ts, thread_root, sender_id)
        bucket = self._cohorts.get(key)
        if bucket is not None:
            bucket.pop(str(ts), None)
            if not bucket:
                self._cohorts.pop(key, None)
        if self._latest.get(key) != ts:
            return                      # somebody newer is already the survivor; nothing to hand on
        remaining = self._cohorts.get(key)
        if remaining:
            self._latest[key] = max(remaining, key=_ts_key)
        else:
            self._latest.pop(key, None)

    # ------------------------------------------------------------- evaluate

    async def evaluate(self, *, channel_id: str, ts: str, text: str,
                       sender_id: Optional[str] = None,
                       sender_name: Optional[str] = None,
                       sender_type: Optional[str] = None,
                       channel_steering_text: Optional[str] = None,
                       attachments: Optional[List[str]] = None,
                       client: Any = None,
                       thread_root_ts: Optional[str] = None,
                       edit_marker: Optional[str] = None,
                       carried_sources: Optional[List[SourceMessage]] = None,
                       attempt_id: Optional[str] = None) -> GateEvaluation:
        """Debounce, coalesce, ask once, return one bit.

        `decision` is None when there is nothing to act on, and `decline_cause` says which kind of
        nothing — the CALLER owns the turn's single terminal event and cannot read that off a log
        line. `attempt_id` rides the diagnostics and changes no behaviour.

        The cohort: every source enrolled in this conversation up to and including this one goes to
        the classifier, oldest first, and comes back on the GateEvaluation so the responder answers
        the whole burst. Conversation identity is unchanged — a thread keys on its root and
        collapses cross-author (the reply lands in-thread with full history); a top-level stream
        keys per sender, so two people's unrelated questions are never merged into one.

        Nothing here waits on anything else. The gate does not look at images, does not hold
        ambient work, and has no callback anyone can block on: the ambient worker analyses images
        on its own schedule, immediately, whatever this decides.
        """
        key = self._conv_key(channel_id, ts, thread_root_ts, sender_id)
        # Monotonic; a stale caller can't clobber a newer marker.
        self.note_arrival(channel_id, ts, thread_root_ts, sender_id)

        # The edit context belongs to THIS attempt or to nobody: it is keyed by the edit's own
        # marker, so the original attempt (which carries no marker) cannot pop the edit's context
        # and mistake itself for the edit. Taken BEFORE the debounce so the enrolled record holds
        # it — the cohort is what reaches the model, and an edit's before/after text is intrinsic
        # to the message rather than context about the channel.
        edit_context = self._take_edit_context(client, channel_id, ts, edit_marker)

        source = SourceMessage(
            ts=str(ts), text=text or "", sender_id=sender_id, sender_name=sender_name,
            sender_type=sender_type, thread_root_ts=thread_root_ts,
            attachments=tuple(attachments or ()), edit=edit_context)
        # Enrolled BEFORE the await, so a survivor that arrives during our debounce finds us.
        self._enroll_source(key, source)

        # Messages a Phase-Q queue drain folded into this turn. They never got a debounce window of
        # their own — they arrived while an earlier turn held the lock — so the gate would
        # otherwise judge this batch on its newest message alone and discard the rest. Enrolled
        # with the same identity rules as any other source; older ones are drained into this
        # cohort below, and one that is somehow newer is left for its own survivor.
        for carried in (carried_sources or ()):
            if carried.ts and carried.ts != source.ts:
                self._enroll_source(key, carried)

        wait = max(0.0, float(getattr(config, "participation_debounce_seconds", 3.0)))
        if wait:
            try:
                await asyncio.sleep(wait)
            except asyncio.CancelledError:
                # The one place this turn can be cancelled after enrolling and before draining.
                # The cohort map is deliberately unbounded — eviction there is a silent
                # message-loss path — so a cancelled stream has to clear itself explicitly or its
                # entry stays forever, and worse, gets swept into some later survivor's cohort as
                # a message from the distant past. Nothing else is owed here: the sources were
                # never judged, and re-raising leaves the caller's cancellation intact.
                self.discard_source(channel_id, ts, thread_root_ts, sender_id)
                raise

        if self._latest.get(key) != ts:
            # Superseded. Our record STAYS enrolled for the survivor, so nothing this person said
            # is lost — it arrives at both models as part of the survivor's cohort.
            participation_telemetry.gate_declined(
                channel_id, ts, cause="superseded", attempt_id=attempt_id,
                survivor_ts=self._latest.get(key))
            return GateEvaluation(decline_cause="superseded")

        # An EDIT explicitly cancelled THIS message's original evaluation. The edit is handled
        # elsewhere — Slack's app_mention for a mention-added edit, or the edit's own
        # marker-carrying re-evaluation — so the stale original must stay silent. Only the
        # context-free original consumes the mark; the edit's own attempt carries a marker and is
        # exempt, which is now structural rather than a coincidence of pop ordering.
        if edit_context is None and self._consume_edit_supersession(key, ts):
            participation_telemetry.gate_declined(channel_id, ts, cause="edit_superseded",
                                                  attempt_id=attempt_id)
            return GateEvaluation(decline_cause="edit_superseded")

        sources = self._drain_cohort(key, ts)
        if not sources:
            # Defensive: our own enrollment is always in there. An empty drain would mean the
            # cohort was cleared underneath us, and judging nothing is not a judgment.
            sources = (source,)

        # A cohort of nothing but captionless IMAGES. Someone dropped pictures into the channel and
        # said nothing — there is no question, no addressee and no text to judge, so there is
        # nothing for a wake decision to be about. Structural, and named as such in the ledger
        # rather than dressed up as the model choosing silence: no classifier runs and no responder
        # wakes. What DOES still happen matters — the ambient worker analyses the pictures on its
        # own schedule, and the caller catalogues the files — because declining to answer a wordless
        # upload is not the same as forgetting it happened. (A captionless message cannot be
        # name-tagged, and a real @mention never reaches this gate, so "captionless" is already
        # "untagged".)
        #
        # IMAGES SPECIFICALLY, and the distinction is not pedantry. A wordless PDF or spreadsheet
        # dropped into a channel is a document somebody may well want read — the responder can open
        # it, and often should — so treating "no caption" as "nothing to do" for those would skip
        # both models on exactly the material this bot is best at. A picture with no caption is the
        # narrow case where there is genuinely nothing being asked.
        attached = [d for s in sources for d in s.attachments]
        if (attached and all(is_image_descriptor(d) for d in attached)
                and not any((s.text or "").strip() for s in sources)):
            participation_telemetry.gate_declined(
                channel_id, ts, cause="image_only", attempt_id=attempt_id,
                source_count=len(sources))
            return GateEvaluation(decline_cause="image_only", sources=sources)

        # Timed around the model call and NOTHING else. The gate's own wall time is dominated by
        # the debounce sleep, so reading it as classifier latency blames the provider for a delay
        # we chose. Measured on failure too — a timeout's duration is the story.
        classifier_started = time.monotonic()
        detail: Optional[str] = None
        raw: Optional[bool] = None
        try:
            raw = await self.openai_client.classify_wake(
                sources=sources, channel_steering_text=channel_steering_text)
        except Exception as e:  # noqa: BLE001 — fail-safe is silence, never spam
            detail = type(e).__name__
        classifier_ms = int((time.monotonic() - classifier_started) * 1000)

        if raw is None:
            # No bit. NOT a decision, and deliberately not dressed as one: the rich gate
            # manufactured a fail-safe `ignore` here, which is silence either way but scored a
            # provider outage as the model choosing restraint. The caller ends the turn as `none`.
            participation_telemetry.gate_declined(
                channel_id, ts, cause="classifier_error", attempt_id=attempt_id,
                detail=detail, classifier_ms=classifier_ms,
                # WHICH model failed. gate_decision carries this, so without it here a utility
                # model swap can be judged on its decisions but not on its failure rate — and the
                # failure rate is the half that decides whether the swap was worth it.
                model=getattr(config, "utility_model", None))
            return GateEvaluation(decline_cause="classifier_error",
                                  classifier_ms=classifier_ms, sources=sources)

        return GateEvaluation(decision=WakeDecision(wake=bool(raw)),
                              classifier_ms=classifier_ms, sources=sources)

    @staticmethod
    def _take_edit_context(client: Any, channel_id: str, ts: str,
                           marker: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Pop this attempt's edit context from the Slack facade, where the message-events edit
        path stashed it — keyed by (channel, ts, MARKER).

        The marker is why this is safe. An edit keeps its original Slack timestamp, so
        (channel, ts) alone does not distinguish the edit's own re-evaluation from the stale
        original attempt it superseded, and whichever ran first popped the context. That made
        ownership a race: the original could arrive holding the edit's before/after text and
        conclude it WAS the edit, which also suppressed the supersession check that was supposed
        to silence it. Only the attempt carrying the edit's marker may consume it; an attempt with
        no marker (every ordinary message, and the superseded original) gets None.

        Popping rather than peeking means a second evaluation of the same edit falls back to a
        plain judgment instead of replaying stale before-text.

        The store must be an ACTUAL dict, not merely truthy. A MagicMock client answers every
        attribute with another truthy mock whose .pop() returns a mock, so `not store` let a fake
        edit context through — and a whole class of test then passed against a prompt production
        never sends."""
        if client is None or not channel_id or not ts or not marker:
            return None
        store = getattr(client, "_edit_reply_ctx_map", None)
        if not isinstance(store, dict) or not store:
            return None
        try:
            return store.pop(f"{channel_id}|{ts}|{marker}", None)
        except Exception:  # noqa: BLE001 — must never break the decision
            return None
