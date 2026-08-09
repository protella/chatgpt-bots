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

@mentions, thread continuations, and DMs NEVER reach this engine — they are answered directly.
A thread continuation is either of two rules: a strict 1:1 thread (level-independent), or, in an
`on` channel, a thread we have already posted in. Participation in a thread is itself the wake
signal — a thread we have posted in is one we are already part of, and the responder, which can
see the thread where this gate sees only the trigger text, decides what the turn owes, including
nothing. Told to be quiet is not the same as deaf.

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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
#                    judged by the gate; a strict 1:1 continuation stays direct, and NOTHING
#                    else skips the gate here (ruling 1A: the membership widening is `on`-only,
#                    because this level promises the user that nothing else wakes us).
#   on             — all otherwise-eligible ambient and name traffic is judged by the gate, and
#                    an untagged human reply in a thread we have posted in skips it entirely.
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


# How many distinct edit-supersession MARKS to remember. Marks are bookkeeping about messages
# that have already been handled elsewhere, not enrolled messages, so bounding them can never
# discard something waiting for a turn. (The cohort map below is deliberately NOT bounded — see
# `_enroll_source`.)
_MAX_SUPERSESSION_KEYS = 512


def _freeze(value: Any) -> Any:
    """A small payload as a comparable, immutable value — dicts and lists all the way down.

    Used on `SourceMessage.edit`, the one field of a source that is a live mutable object rather
    than a string. Sorted by the repr of the key so a mixed-key dict cannot raise here; the payload
    is always small (an edit's before/after text and one flag)."""
    if isinstance(value, dict):
        return tuple(sorted(((k, _freeze(v)) for k, v in value.items()), key=lambda kv: repr(kv[0])))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((repr(v) for v in value)))
    return value


def _activity_lru_max() -> int:
    """How many conversation streams keep an arrival marker (W5c). Read per call, not cached, so a
    test or an operator can change the bound the same way every other config value changes."""
    try:
        return max(1, int(getattr(config, "participation_activity_lru_max", 1024)))
    except (TypeError, ValueError):
        return 1024


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
class ArrivalMarker:
    """When a conversation stream was last seen speaking — W5c's whole state.

    `ts` is the Slack timestamp that was noted; `at` is the monotonic clock reading when it was
    FIRST noted, and the clock is what the coldness test reads. A Slack ts would measure the gap
    between two people's messages; `at` measures the gap between two ARRIVALS at this process,
    which is what "was this stream busy a moment ago" actually asks.
    """

    ts: str
    at: float


@dataclass(frozen=True)
class Arrival:
    """`note_arrival`'s whole answer about ONE message: its own record, and what came before it.

    Returned rather than looked up again later, and that is the point. The gate notes an arrival
    twice — at gate entry in main.py, then again inside `evaluate` — with real work in between, so
    by the second call the map may have moved on: another message in the same stream may have
    arrived, or the LRU may have evicted the stream entirely. Either would make the second call
    answer a different question from the first. The caller carries THIS record from the first call
    to the second instead, so a message's coldness is decided by the state at the moment it
    arrived, which is the only moment that describes it.
    """

    marker: ArrivalMarker
    prior: Optional[ArrivalMarker]


def _stream_was_cold(seen: Optional[Arrival], window: float) -> bool:
    """Whether nothing arrived in this conversation within the last debounce window.

    Judged on the arrival BEFORE this message. Nothing before it at all is the coldest case there
    is. Otherwise the gap is measured between the two ARRIVALS — both times are in the record the
    caller carries — against the window, the same `participation_debounce_seconds` that would be
    waited, because a burst is defined as messages inside one such window and nothing else needs a
    number.

    Not "how long ago was the previous message", which is a different question with a different
    answer: this is read after the caller's steering load, and clocking from now would let a slow
    turn age a genuine burst into a cold stream — the exact thing carrying the record exists to
    prevent.

    Anything that is not an `Arrival` means somebody replaced `note_arrival` with a stub, and the
    answer there is "not cold" — keep the wait, change nothing. A MagicMock answers every attribute
    with another truthy mock, so a stub must not be able to talk this into a behaviour the caller
    never asked for (the same lesson `_take_edit_context` learned the hard way).
    """
    if seen is None:
        return True
    if not isinstance(seen, Arrival) or not isinstance(seen.prior, (ArrivalMarker, type(None))):
        return False
    if seen.prior is None:
        return True
    return (seen.marker.at - seen.prior.at) > max(0.0, window)


@dataclass
class _Speculation:
    """One in-flight speculative classifier call, and the two facts its owner cannot read off the
    task itself.

    `superseded` is set by a NEWER arrival's enrollment, which cancels this call because the burst
    it judged has already been overtaken. The owner reads the flag rather than the task's state:
    "was I cancelled by someone else" and "am I cancelled yet" are different questions, and a task
    cancelled a microsecond ago still answers False to `cancelled()`. Awaiting it on that guess
    would raise CancelledError out of a turn that is entitled to a decision.

    `committed` is set by the owner just before it awaits the verdict, and closes the other side of
    the same race: past its supersession check the owner is going to USE this call, so a message
    arriving during the await must not cancel it out from under a decided turn.
    """

    task: "asyncio.Task[Tuple[Optional[bool], Optional[str], int]]"
    superseded: bool = False
    committed: bool = False


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

    `source_files` rides alongside it: (ts, attachment payloads) for the cohort members that
    arrived carrying files. The gate never looks at them — they are here because the members' own
    dispatches ended at the gate, so this is the last place their live payloads exist, and the turn
    that survives needs them to authorize a file whose message Slack has not propagated yet.
    """

    decision: Optional[WakeDecision] = None
    decline_cause: Optional[str] = None
    # The model call alone, in ms. NOT the gate's wall time, which is mostly debounce.
    classifier_ms: Optional[int] = None
    sources: Tuple[SourceMessage, ...] = ()
    source_files: Tuple[Tuple[str, Tuple[Dict[str, Any], ...]], ...] = ()


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
        # The enrolled messages' live FILE payloads, conversation key -> {ts: attachment dicts}.
        # Kept beside the cohort rather than on the record: a SourceMessage carries names and types
        # only (see the class), and nothing in this module reads these. They exist for the survivor,
        # whose turn needs a file id to be authorizable even when Slack's fetch has not caught up to
        # the message that carried it — and the carrying message's own dispatch ends here, so if the
        # gate drops the payload nothing downstream can recover it. Lifecycle follows `_cohorts`
        # exactly: written on enrollment, taken on drain, withdrawn on cancellation. Only messages
        # that actually carried files get an entry.
        self._cohort_files: Dict[str, Dict[str, Tuple[Dict[str, Any], ...]]] = {}
        # F52: messages whose in-flight evaluation an EDIT has explicitly cancelled. An edit
        # keeps the SAME ts, so a newer arrival can never supersede the original the ordinary
        # way (note_arrival is monotonic on ts); supersede() marks it here and evaluate() drops
        # the stale original — the edit's OWN re-evaluation, which carries edit context, is
        # exempt. conv key -> set of superseded ts. Bounded by _MAX_SUPERSESSION_KEYS.
        self._edit_superseded: "OrderedDict[str, set]" = OrderedDict()
        # W5c: the ARRIVAL map — conversation key -> the newest `Arrival` seen in that stream. Its
        # own structure, deliberately not folded into `_latest` or `_cohorts`: those two are
        # cleared the moment a cohort drains or a source is withdrawn, and "this conversation was
        # active seconds ago" has to outlive both — a drain is precisely when a stream is at its
        # busiest.
        #
        # Bounded by an LRU (`participation_activity_lru_max`), which unlike the cohort map is safe
        # here because this holds timestamps and never a message: an eviction can lose nothing
        # anybody said. What it CAN cost, stated honestly: a stream evicted while it is still
        # warm reads as cold to its next message, which is then judged immediately instead of
        # waiting — that burst is answered as two turns rather than one, exactly as a genuinely
        # cold start already is. It takes an eviction to reach that, and eviction only touches the
        # least recently spoken-in stream once a thousand others have spoken since. A message
        # already in flight is immune either way: its coldness was decided from the `Arrival` it
        # carries (see `Arrival`), not from this map.
        self._activity: "OrderedDict[str, Arrival]" = OrderedDict()
        # W5b's cap: the ONE live speculative call a conversation is allowed, keyed the same way as
        # everything else here. An entry lives from the moment its attempt starts speculating until
        # that attempt uses or discards it; a newer arrival in the same stream cancels what it
        # finds here (see `_supersede_stream_speculation`) before registering its own. Bounded by
        # the fact that every entry has a live evaluation behind it that will remove its own.
        self._speculations: Dict[str, _Speculation] = {}

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
                     sender_id: Optional[str] = None) -> Optional[Arrival]:
        """Register a message's ts as its conversation's newest — MONOTONICALLY (F5 fix b).

        Called at gate entry, BEFORE any await, so an older event delayed by memory/topic
        I/O can never overwrite a newer event's marker and win the debounce. Only a
        genuinely newer Slack ts advances the marker. F27: sender_id scopes the top-level
        stream key so a monotonic advance is per-author.

        RETURNS this arrival and the one before it (see `Arrival`), which is what W5c's adaptive
        debounce judges coldness against. It has to be the PRIOR value: a reading taken after the
        record was written would report the message's own arrival and call every stream warm.

        THIS METHOD IS CALLED TWICE FOR EVERY MESSAGE — gate entry, then `evaluate` — and other
        messages arrive in between. So the activity map is written in exactly ONE case: a ts
        strictly NEWER than whatever that stream currently holds. Every other call is a pure read.
        Not a move_to_end, not a re-time, not a reinsert:

          * a repeat of the current ts returns the stored record untouched — re-timing it would
            date the stream from the second look rather than the arrival, and even nudging LRU
            recency would let a doubled call outrank a stream that actually spoke more recently;
          * a ts that is not newer (our own second call after somebody overtook us, or an
            out-of-order older event) must not overwrite a record that describes a LATER message;
          * and if the stream was EVICTED between the two calls, a stale second call must not
            reinsert itself — `_latest` still names the newer message, and inserting here would
            both re-time a stream from an old message and evict some other stream to do it.

        A call that cannot reconstruct what the stream looked like when its ts first arrived
        answers conservatively — WARM, which keeps today's trailing wait. The direction matters: a
        wrong "warm" costs one debounce, a wrong "cold" splits a burst. Callers that need the exact
        answer carry the `Arrival` from their first call, which is what main.py does and what
        `evaluate`'s `arrival` is for."""
        if not channel_id or not ts:
            return None
        key = self._conv_key(channel_id, ts, thread_root, sender_id)
        current = self._latest.get(key)
        if current is None or _ts_key(ts) > _ts_key(current):
            self._latest[key] = ts

        now = time.monotonic()
        recorded = self._activity.get(key)
        if recorded is not None:
            if recorded.marker.ts == str(ts):
                return recorded                     # same message, second call — same answer
            if _ts_key(str(ts)) <= _ts_key(recorded.marker.ts):
                return Arrival(marker=ArrivalMarker(ts=str(ts), at=now),
                               prior=recorded.marker)
        elif current is not None and _ts_key(str(ts)) < _ts_key(current):
            # Evicted, and this call is a stale second look: the stream has a newer message than
            # us. Answer warm — something newer than us exists in this stream right now, and the
            # zero gap says so — but write nothing, so an old message cannot displace a live one.
            return Arrival(marker=ArrivalMarker(ts=str(ts), at=now),
                           prior=ArrivalMarker(ts=str(current), at=now))
        arrival = Arrival(marker=ArrivalMarker(ts=str(ts), at=now),
                          prior=recorded.marker if recorded is not None else None)
        self._activity[key] = arrival
        self._activity.move_to_end(key)
        while len(self._activity) > _activity_lru_max():
            self._activity.popitem(last=False)
        return arrival

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

    def _is_edit_superseded(self, key: str, ts: str) -> bool:
        """Whether an edit has marked `ts` — WITHOUT consuming the mark.

        The speculative path (W5b) needs to know that this attempt is already doomed so it does not
        spend a call on it, and consuming here would eat the mark the real check at window close is
        waiting for, letting the stale original speak after all."""
        bucket = self._edit_superseded.get(key)
        return bool(bucket and str(ts) in bucket)

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

    def _enroll_source(self, key: str, source: SourceMessage,
                       file_payloads: Sequence[Dict[str, Any]] = ()) -> None:
        """Record a source in its conversation's cohort so a later survivor carries it.

        No cap and no eviction: see `self._cohorts`. Re-enrolling the same ts (an edit keeps its
        original timestamp) REPLACES the record, so the cohort holds the current text rather than
        two versions of one message — and replaces its file payloads with it, for the same reason."""
        bucket = self._cohorts.get(key)
        if bucket is None:
            bucket = OrderedDict()
            self._cohorts[key] = bucket
        bucket[source.ts] = source
        self._cohorts.move_to_end(key)
        if file_payloads:
            self._cohort_files.setdefault(key, {})[source.ts] = tuple(file_payloads)
        elif key in self._cohort_files:
            remaining = self._cohort_files[key]
            remaining.pop(source.ts, None)
            if not remaining:
                del self._cohort_files[key]

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

    def _peek_cohort(self, key: str, own_ts: str) -> Tuple[SourceMessage, ...]:
        """Exactly what `_drain_cohort` WOULD return right now, taking nothing.

        W5b speculates on this tuple and then compares it, by value, against the real drain at
        window close. Reading through the same rule (everything up to and including our own ts,
        oldest first) is what makes that comparison meaningful: if the two differ, the cohort
        genuinely changed under us, and the speculation is about a burst that no longer exists."""
        bucket = self._cohorts.get(key)
        if not bucket:
            return ()
        own_key = _ts_key(own_ts)
        seen = [(_ts_key(pts), src) for pts, src in bucket.items() if _ts_key(pts) <= own_key]
        seen.sort(key=lambda pair: pair[0])
        return tuple(source for _, source in seen)

    def _take_cohort_files(self, key: str, sources: Sequence[SourceMessage]
                           ) -> Tuple[Tuple[str, Tuple[Dict[str, Any], ...]], ...]:
        """The live file payloads of the sources just drained, and only those.

        Called with what `_drain_cohort` returned, so the two structures come apart in step. It also
        drops any payload whose ts is no longer enrolled: this map hangs off an unbounded cohort, so
        it must never be able to keep alive a payload nobody is coming back for.
        """
        held = self._cohort_files.get(key)
        if not held:
            self._cohort_files.pop(key, None)
            return ()
        taken = tuple((source.ts, held.pop(source.ts)) for source in sources if source.ts in held)
        enrolled: Dict[str, SourceMessage] = self._cohorts.get(key) or OrderedDict()
        for orphan in [ts for ts in held if ts not in enrolled]:
            held.pop(orphan, None)
        if not held:
            self._cohort_files.pop(key, None)
        return taken

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
        held = self._cohort_files.get(key)
        if held is not None:
            held.pop(str(ts), None)
            if not held:
                self._cohort_files.pop(key, None)
        if self._latest.get(key) != ts:
            return                      # somebody newer is already the survivor; nothing to hand on
        remaining = self._cohorts.get(key)
        if remaining:
            self._latest[key] = max(remaining, key=_ts_key)
        else:
            self._latest.pop(key, None)

    # -------------------------------------------------------- classification helpers

    @staticmethod
    def _cohort_fingerprint(sources: Sequence[SourceMessage]) -> Tuple[Any, ...]:
        """The cohort as an immutable VALUE, for the one comparison W5b turns on.

        Comparing the records themselves compares object identity's poorer cousin: a SourceMessage
        is frozen, but its `edit` payload is an ordinary dict that the speculative prompt has
        already been rendered from. Mutate that dict mid-window and the two tuples still compare
        equal, so a verdict about the old before/after text would be committed as a judgment of the
        new one. Freezing the payload at speculation start makes the comparison a statement about
        what the model was actually shown, which is what it has to be."""
        return tuple((s.ts, s.text, s.sender_id, s.sender_name, s.sender_type, s.thread_root_ts,
                      tuple(s.attachments), _freeze(s.edit)) for s in sources)

    @staticmethod
    def _is_captionless_image_cohort(sources: Sequence[SourceMessage]) -> bool:
        """A cohort of nothing but wordless IMAGES — the one shape with nothing to judge.

        ONE definition, read by the structural decline at window close and by W5b's eligibility
        test at enrollment. Two copies would be two answers to "does this cohort get a classifier
        call", and the speculative half would start spending calls on the path the real half
        skips."""
        attached = [d for s in sources for d in s.attachments]
        return bool(attached and all(is_image_descriptor(d) for d in attached)
                    and not any((s.text or "").strip() for s in sources))

    async def _classify_cohort(self, sources: Tuple[SourceMessage, ...],
                               channel_steering_text: Optional[str]
                               ) -> Tuple[Optional[bool], Optional[str], int]:
        """The model call, timed, with its failure captured rather than raised: (bit, failure
        type name, ms). Shared by the speculative task and the ordinary post-debounce call so the
        two cannot drift into asking the question differently or timing it differently.

        Timed around the model call and NOTHING else. The gate's own wall time is dominated by the
        debounce sleep, so reading it as classifier latency blames the provider for a delay we
        chose. Measured on failure too — a timeout's duration is the story.

        A cancellation is a BaseException and so passes straight through the fail-safe below: a
        speculation somebody threw away is not a classifier failure and must not be recorded as
        one."""
        started = time.monotonic()
        raw: Optional[bool] = None
        detail: Optional[str] = None
        try:
            raw = await self.openai_client.classify_wake(
                sources=sources, channel_steering_text=channel_steering_text)
        except Exception as e:  # noqa: BLE001 — fail-safe is silence, never spam
            detail = type(e).__name__
        return raw, detail, int((time.monotonic() - started) * 1000)

    def _supersede_stream_speculation(self, key: str, *, channel_id: str, ts: str) -> None:
        """Cancel the speculation of whoever held this stream before us — at ENROLLMENT.

        THE CAP: at most one live speculative call per conversation. The moment a newer message
        enrolls, the older attempt is superseded by definition, and its speculation is a verdict
        about a burst that has already been overtaken. Left alone it runs to completion and is
        thrown away at the older attempt's own wake-up, so a five-message burst spent five calls to
        use one. Cancelled here it is aborted within a message of starting.

        Signal only: this cancels the API task and does not join it. The owner joins its own task
        on the superseded path, which is where the accounting for it belongs — and cancels NOTHING
        else. The older attempt's source stays enrolled for the survivor, its cohort is intact, and
        `discard_source` remains the only path that withdraws a message.

        A speculation whose owner has already committed to using it is left alone. That owner has
        passed its own supersession check and is awaiting the verdict; cancelling underneath it
        would turn a decided turn into a raised CancelledError.
        """
        live = self._speculations.get(key)
        if live is None or live.committed:
            return
        live.superseded = True
        live.task.cancel()
        logger.debug("Wake gate: speculative classification superseded early by %s/%s",
                     channel_id, ts)

    def _forget_speculation(self, key: str, spec: "Optional[_Speculation]") -> None:
        """Deregister OUR speculation, and only ours: a newer attempt may already have replaced the
        entry, and popping that one would leave its task with nobody to cancel it."""
        if spec is not None and self._speculations.get(key) is spec:
            self._speculations.pop(key, None)

    async def _abandon_speculation(self, key: str, spec: "Optional[_Speculation]", *, reason: str,
                                   channel_id: str, ts: str, join: bool = True) -> None:
        """Throw away a speculative verdict: cancel the API task and, normally, join it.

        This cancels ONE task and touches no other state. The source stays enrolled, the cohort
        stays intact, `_latest` is untouched — a superseded window still owes its words to the
        survivor, and `discard_source` remains the only path that withdraws a message.

        The result is dropped in silence beyond a debug line, and that is the point: a decision
        row is minted from what `evaluate` returns, so a speculative call that nobody uses must
        leave no verdict behind. One attempt, one decision.

        `join=False` is for the one caller that cannot await — the debounce's own cancellation
        handler, which is already unwinding and must re-raise promptly."""
        if spec is None:
            return
        self._forget_speculation(key, spec)
        spec.task.cancel()
        logger.debug("Wake gate: speculative classification discarded (%s) for %s/%s",
                     reason, channel_id, ts)
        if join:
            await asyncio.gather(spec.task, return_exceptions=True)

    # ------------------------------------------------------------- evaluate

    async def evaluate(self, *, channel_id: str, ts: str, text: str,
                       sender_id: Optional[str] = None,
                       sender_name: Optional[str] = None,
                       sender_type: Optional[str] = None,
                       channel_steering_text: Optional[str] = None,
                       attachments: Optional[List[str]] = None,
                       file_payloads: Optional[Sequence[Dict[str, Any]]] = None,
                       client: Any = None,
                       thread_root_ts: Optional[str] = None,
                       edit_marker: Optional[str] = None,
                       carried_sources: Optional[List[SourceMessage]] = None,
                       queue_drained: bool = False,
                       arrival: Optional[Arrival] = None,
                       attempt_id: Optional[str] = None) -> GateEvaluation:
        """Debounce, coalesce, ask once, return one bit.

        `decision` is None when there is nothing to act on, and `decline_cause` says which kind of
        nothing — the CALLER owns the turn's single terminal event and cannot read that off a log
        line. `attempt_id` rides the diagnostics and changes no behaviour.

        `arrival` is what `note_arrival` answered when the CALLER noted this message at gate entry,
        and it is how the adaptive debounce stays a statement about this message: the caller does
        real I/O between that call and this one, so re-deriving coldness here would read a stream
        that has since moved on. Omit it and this call derives its own, which is correct for any
        caller that has not already noted the arrival.

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
        #
        # `arrival` is this message's own record from the caller's gate-entry call, and when it is
        # present it — not this second call — decides coldness. Between the two calls the caller
        # does real I/O, during which another message in the same stream can arrive or the LRU can
        # evict the stream, and either would have this call answer a question about a different
        # moment than the one this message arrived in. The note itself still happens, because
        # `_latest` and the activity record are what a LATER message will read.
        noted = self.note_arrival(channel_id, ts, thread_root_ts, sender_id)
        seen = arrival if arrival is not None else noted

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
        # Enrolled BEFORE the await, so a survivor that arrives during our debounce finds us. The
        # raw payload rides along unread: this dispatch is the last holder of it, and if we lose our
        # own debounce the survivor's turn is the only place those files can still be authorized.
        self._enroll_source(key, source, file_payloads or ())

        # Messages a Phase-Q queue drain folded into this turn. They never got a debounce window of
        # their own — they arrived while an earlier turn held the lock — so the gate would
        # otherwise judge this batch on its newest message alone and discard the rest. Enrolled
        # with the same identity rules as any other source; older ones are drained into this
        # cohort below, and one that is somehow newer is left for its own survivor.
        for carried in (carried_sources or ()):
            if carried.ts and carried.ts != source.ts:
                self._enroll_source(key, carried)

        # Enrollment is the moment the stream changes hands. If we are its newest message, whoever
        # was speculating here is superseded — by definition, and now rather than at their own
        # wake-up — so their call is aborted while it is still cheap. An out-of-order older arrival
        # is not the survivor and cancels nothing.
        if self._latest.get(key) == ts:
            self._supersede_stream_speculation(key, channel_id=channel_id, ts=ts)

        debounce = max(0.0, float(getattr(config, "participation_debounce_seconds", 3.0)))
        # A Phase-Q drain redispatch has already been debounced, by the drain: it lingered with the
        # lock held so stragglers could join, then took the whole queue at once. Waiting the
        # debounce again would coalesce nothing — the batch is closed — and every message in it has
        # been waiting since before the previous turn ended. Only the sleep is skipped; supersession,
        # the cohort drain, classification and telemetry all run exactly as they do for any turn.
        #
        # W5c, the other way to earn a zero wait: the stream was COLD. The debounce exists to
        # collect a burst, and a burst is — by the debounce's own definition — messages within one
        # window of each other. If nothing arrived in this conversation within the last window,
        # there is no burst to collect and the wait buys nothing but three seconds of the person
        # watching a channel where nothing happens. No new constant governs this: the window IS
        # `participation_debounce_seconds`, so one tuned value still describes both halves of the
        # same behaviour. A stream that IS warm keeps today's trailing wait exactly.
        cold = _stream_was_cold(seen, debounce)
        wait = 0.0 if (queue_drained or cold) else debounce

        # W5b: the leading-edge speculative call. The debounce is dead time we already spend, so
        # the classifier can run INSIDE it and have its verdict in hand when the window closes.
        #
        # Two rules keep it honest. It starts only when the cohort AS IT STANDS RIGHT NOW would
        # earn a classifier call anyway — no path that skips the model today may gain an API call
        # because we guessed early — and its verdict is used only if the cohort at window close is
        # the same one it judged. Everything else discards it. And it exists only when there is a
        # window to hide in: `wait` of zero (a queue drain, or a cold stream above) starts nothing,
        # which is what makes "one attempt, one decision" structural rather than a promise — the
        # immediate path and the speculative path can never both be live for one evaluation.
        speculative: Optional[_Speculation] = None
        speculated_on: Tuple[Any, ...] = ()
        if wait:
            cohort_now = self._peek_cohort(key, ts)
            eligible = (
                bool(cohort_now)
                # Already superseded before we even slept — today's code asks nobody.
                and self._latest.get(key) == ts
                # A wordless pile of pictures is a structural decline, not a judgment.
                and not self._is_captionless_image_cohort(cohort_now)
                # An edit has already cancelled this attempt; PEEKED, never consumed, so the real
                # check at window close still finds its mark.
                and not (edit_context is None and self._is_edit_superseded(key, ts))
            )
            if eligible:
                # The fingerprint is taken HERE, before the task can run, so it records the cohort
                # the model is about to be shown rather than whatever that cohort has become by
                # the time the window closes.
                speculated_on = self._cohort_fingerprint(cohort_now)
                speculative = _Speculation(task=asyncio.create_task(
                    self._classify_cohort(cohort_now, channel_steering_text)))
                # Registered so the NEXT message in this stream can find it and cancel it. Ours
                # replaces whatever was here; we cancelled that one at enrollment.
                self._speculations[key] = speculative

            # The window itself. Started AFTER the speculation so the call is already in flight for
            # the whole of it — that is the entire latency win, and reversing these two lines would
            # give it back.
            try:
                await asyncio.sleep(wait)
            except asyncio.CancelledError:
                # The one place this turn can be cancelled after enrolling and before draining.
                # The cohort map is deliberately unbounded — eviction there is a silent
                # message-loss path — so a cancelled stream has to clear itself explicitly or its
                # entry stays forever, and worse, gets swept into some later survivor's cohort as
                # a message from the distant past. Nothing else is owed here: the sources were
                # never judged, and re-raising leaves the caller's cancellation intact.
                #
                # The speculative task is this turn's own child, so it goes with the turn — but
                # WITHOUT a join: we are already unwinding and the caller's cancellation must not
                # wait on an HTTP round trip. Cancelling it withdraws nothing from the cohort.
                await self._abandon_speculation(key, speculative, reason="evaluation cancelled",
                                                channel_id=channel_id, ts=ts, join=False)
                self.discard_source(channel_id, ts, thread_root_ts, sender_id)
                raise

        if self._latest.get(key) != ts:
            # Superseded. Our record STAYS enrolled for the survivor, so nothing this person said
            # is lost — it arrives at both models as part of the survivor's cohort. The speculative
            # call is the ONLY thing cancelled here: the survivor answers for this message, and it
            # needs the enrollment we are leaving behind.
            await self._abandon_speculation(key, speculative, reason="superseded",
                                            channel_id=channel_id, ts=ts)
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
            # The accepted cost of W5b, named where it happens: this attempt was eligible when the
            # window opened and the edit arrived during it, so a speculative call has already been
            # spent on a cohort nobody will hear about. Rare, one utility call, and it buys the
            # common case a whole debounce window.
            await self._abandon_speculation(key, speculative, reason="edit superseded",
                                            channel_id=channel_id, ts=ts)
            participation_telemetry.gate_declined(channel_id, ts, cause="edit_superseded",
                                                  attempt_id=attempt_id)
            return GateEvaluation(decline_cause="edit_superseded")

        sources = self._drain_cohort(key, ts)
        if not sources:
            # Defensive: our own enrollment is always in there. An empty drain would mean the
            # cohort was cleared underneath us, and judging nothing is not a judgment.
            sources = (source,)
        source_files = self._take_cohort_files(key, sources)

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
        if self._is_captionless_image_cohort(sources):
            await self._abandon_speculation(key, speculative, reason="image-only cohort",
                                            channel_id=channel_id, ts=ts)
            participation_telemetry.gate_declined(
                channel_id, ts, cause="image_only", attempt_id=attempt_id,
                source_count=len(sources))
            return GateEvaluation(decline_cause="image_only", sources=sources,
                                  source_files=source_files)

        # THE decision call — exactly one per evaluation, from one of two places.
        #
        # The speculative verdict counts only if the cohort that closed the window fingerprints
        # identically to the one it judged — a deep value taken at speculation start, so a mutable
        # edit payload cannot be changed underneath a verdict that was rendered from the old one.
        # Sources joining, an edit replacing a source's text in place, or a supersession consumed
        # mid-window all make it a verdict about a different burst. When it does not match it is
        # cancelled and thrown away — no row, no reuse — and the ordinary call runs exactly as it
        # does today, on the cohort that actually exists.
        if (speculative is not None and not speculative.superseded
                and speculated_on == self._cohort_fingerprint(sources)):
            # Claimed before the await: past the supersession check this verdict is the turn's, so
            # a message landing during the round trip must leave it alone rather than cancel a
            # decision out from under us. `superseded` is the other half — a newer arrival already
            # cancelled this call, so there is nothing to wait for and the fallback below runs a
            # normal one. Neither path can produce two decisions.
            speculative.committed = True
            self._forget_speculation(key, speculative)
            raw, detail, classifier_ms = await speculative.task
        else:
            await self._abandon_speculation(key, speculative, reason="cohort changed under speculation",
                                            channel_id=channel_id, ts=ts)
            raw, detail, classifier_ms = await self._classify_cohort(
                sources, channel_steering_text)

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
                                  classifier_ms=classifier_ms, sources=sources,
                                  source_files=source_files)

        return GateEvaluation(decision=WakeDecision(wake=bool(raw)),
                              classifier_ms=classifier_ms, sources=sources,
                              source_files=source_files)

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
