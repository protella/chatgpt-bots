"""Don't answer a message the conversation has already moved past.

THE FAILURE. Someone asks a question, thinks better of it, and immediately sends a correction —
or asks again in the same breath. Both messages are real, both wake a turn, and the first turn
is already several seconds into a model call when the second arrives. The second turn is the one
with the whole picture. The first one finishes anyway and posts an answer to a question that was
superseded before it was written, and the room reads two replies where a person would have sent
one.

The participation engine already collapses that burst BEFORE the model runs (its debounce and
supersession). This is the other half: the window AFTER a turn has committed to answering, where
nothing was watching. It is a small window and it is the expensive one, because by then we are
about to speak.

THE MECHANISM, and its limits. Every turn takes a LEASE at the moment it enters handle_message,
carrying the newest inbound ts it has accounted for. Every later turn in the same conversation
raises a WATERMARK. Immediately before the first Slack call that would create a visible answer,
the lease compares the two: if a newer inbound message is on the record, the surface is never
created and `StaleSendSuppressed` is raised — before Slack, not after.

The honest guarantee is exactly this and nothing more:

    if a newer admitted inbound ts is present before the local authorization check,
    the first answer surface is not created.

It is NOT a promise that no stale answer can ever post. Slack has no conditional post, so a
message admitted a microsecond after the check still races; event delivery has latency of its
own; a turn can await between admission and its check; and once a stream has STARTED, the
remainder goes out (a suppressed tail is a broken half-answer, which is worse than a late one).

NO TIMERS, NO POLICY. There is no freshness window, no grace period, no retention cap and no
tunable anywhere in this module. A watermark is a FACT — "this conversation received a newer
message" — and the moment it becomes a duration it becomes a guess that is wrong in both
directions. Entries live exactly as long as some lease in their scope is open, and are deleted
when the last one closes: bounded by concurrency, not by a clock.

RESTARTS. Process-local, deliberately. Watermarks and leases live in memory and die with the
process — which costs nothing, because the in-flight responders they would have protected die
with it too. There is no cross-restart or downtime guarantee, and none is claimed.

SCOPES. Two, and a message can be watched by both:

  ("thread", channel, root_ts)  every message, keyed by the thread it belongs to. A top-level
                                message is its own root. A reply arriving under a root stales a
                                turn that was answering that root — cross-author on purpose,
                                since the reply lands in the same thread with the full history.
  ("top", channel, sender_id)   top-level messages only, keyed per SENDER. One person's rapid
                                second question supersedes their first; two different people's
                                unrelated top-level questions never collide, and are both
                                answered.

`sender_id` is an immutable Slack identity (user / bot_id / app_id), never a display name.
When a message carries none, the top scope is OMITTED rather than bucketed under "unknown" —
collapsing unrelated senders into one scope would let a stranger's message silence an answer.
"""
import asyncio
import functools
import weakref
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, ParamSpec, Tuple, TypeVar

from logger import setup_logger

logger = setup_logger(name="slack_bot.StaleSendGuard")

Scope = Tuple[str, str, str]

# Lease states. `pending` — nothing visible yet, still cancellable. `committed` — a first
# surface exists in Slack, so the rest of the answer must follow it. `suppressed` — the guard
# refused, and every later visible attempt on this turn is refused too.
PENDING = "pending"
COMMITTED = "committed"
SUPPRESSED = "suppressed"


def ts_key(ts: Any) -> tuple:
    """Numeric (seconds, microseconds) sort key for a Slack ts — never lexical, so '9.0'
    sorts before '10.0'. THE comparator: participation.py imports this one rather than
    keeping its own, because two definitions of "newer" is one too many."""
    try:
        s, _, frac = str(ts).partition(".")
        return (int(s or 0), int((frac + "000000")[:6]))
    except (ValueError, TypeError):
        return (0, 0)


def is_newer(candidate: Any, baseline: Any) -> bool:
    """Strictly newer. An edit re-dispatch keeps its ORIGINAL ts, so it is never newer than
    itself — edit supersession stays the edit path's job, not this module's."""
    if candidate is None:
        return False
    if baseline is None:
        return True
    return ts_key(candidate) > ts_key(baseline)


def thread_scope(channel_id: Any, ts: Any, thread_root: Any) -> Optional[Scope]:
    """The scope every message has. A top-level message is the root of its own thread."""
    if not channel_id:
        return None
    root = thread_root or ts
    if not root:
        return None
    return ("thread", str(channel_id), str(root))


def top_scope(channel_id: Any, ts: Any, thread_root: Any,
              sender_id: Any) -> Optional[Scope]:
    """The per-sender scope, for TOP-LEVEL messages only. None for a thread reply (it is
    already covered by its thread) and None without an identity (see the module docstring:
    unrelated senders must never share a bucket)."""
    if not channel_id or not sender_id:
        return None
    if thread_root and ts and str(thread_root) != str(ts):
        return None
    return ("top", str(channel_id), str(sender_id))


def scopes_for(channel_id: Any, ts: Any, thread_root: Any,
               sender_id: Any = None) -> Tuple[Scope, ...]:
    """Every scope a message belongs to — one for a thread reply, two for a top-level post."""
    found = [s for s in (thread_scope(channel_id, ts, thread_root),
                         top_scope(channel_id, ts, thread_root, sender_id)) if s]
    return tuple(found)


def scopes_for_message(message: Any) -> Tuple[Scope, ...]:
    """Every scope a live Message belongs to, read exactly as `begin_turn` reads it — so a queued
    message and the lease it would open can never disagree about where they live."""
    meta = getattr(message, "metadata", None) or {}
    return scopes_for(getattr(message, "channel_id", None), meta.get("ts"),
                      getattr(message, "thread_id", None), meta.get("sender_id"))


def primary_scope_key(channel_id: Any, ts: Any, thread_root: Any,
                      sender_id: Any = None) -> str:
    """The single scope a message is SUPERSEDED within, as a string key.

    The participation engine's burst collapse asks the same thread-vs-top question, and used to
    answer it with its own copy of the rule. This is that rule, in one place: a thread reply
    collapses within its thread, a top-level message within its author's own top-level stream.

    ONE DELIBERATE DIFFERENCE from `scopes_for` above. Without a sender identity this still
    buckets under "unknown", while the send guard OMITS the top scope entirely. The asymmetry is
    the consequence, not the question: collapsing here means two unattributed messages share one
    answer, which is at worst a merged reply; omitting there means an unattributed message
    cannot silence somebody else's answer, which is the failure worth being strict about."""
    if thread_root and ts and str(thread_root) != str(ts):
        return f"{channel_id}|{thread_root}"
    return f"{channel_id}|top|{sender_id or 'unknown'}"


class StaleSendSuppressed(Exception):
    """The conversation moved on before this turn spoke. CONTROL FLOW, not an error.

    Raised INSTEAD OF calling Slack, so nothing failed and nothing needs retrying or
    apologizing for. Every broad `except` on a delivery path has to let it through: swallowed
    into a generic handler it becomes a "something went wrong" notice for a turn where nothing
    went wrong, which is a worse outcome than the duplicate answer this exists to prevent."""

    def __init__(self, *, scope: Optional[Scope] = None, last_seen_ts: Any = None,
                 observed_latest_ts: Any = None, surface: Optional[str] = None,
                 lease_token: Any = None):
        self.scope = scope
        self.last_seen_ts = last_seen_ts
        self.observed_latest_ts = observed_latest_ts
        self.surface = surface
        # Identity binding for reconsideration (§4a): the private token of the lease whose
        # authorize() raised this. Unforgeable — rearm/force require `is self._token`, so an
        # exception with identical scope/timestamp evidence from another lease can never pass.
        self.lease_token = lease_token
        # Single-owner telemetry rule (§5): the reconsideration runner marks a suppression it
        # has emitted a `stale_send` row for, so the terminal catch never double-counts it.
        self.telemetry_recorded = False
        super().__init__(
            f"newer message {observed_latest_ts} in {scope} supersedes this turn "
            f"(last seen {last_seen_ts}); {surface or 'surface'} not created")


@dataclass
class TurnSendLease:
    """One turn's claim on the right to speak, held from handle_message entry to its finally.

    Passed EXPLICITLY down every path that can post — never a global, never a contextvar. A
    turn that cannot see its own lease is a turn that will speak without checking, and an
    ambient one would attach itself to whatever coroutine happened to be running."""

    scopes: Tuple[Scope, ...] = ()
    last_seen_ts: Optional[str] = None
    # The newest source this turn OWNS (its trigger, which for a drained batch is the newest
    # message in that batch). Immutable. Context assembly must not pull in anything above it:
    # an older turn that absorbed a newer message would answer it, and so would the successor
    # that message already woke.
    ceiling_ts: Optional[str] = None
    state: str = PENDING
    # True while this turn is actually producing a reply (it holds the conversation lock and is
    # past the queued branch). A reconsideration pass reads it on NEWER turns: a newer message
    # whose own responder is running is that responder's to answer, not the stale draft's.
    responding: bool = False
    # EARLY STAND-DOWN. Who this turn was opened for (the immutable Slack identity begin_turn
    # scopes by), the asyncio task running it, and its TurnRuntime once main.py has one — what
    # a newer same-sender turn reads to stop this one before it spends a model call and a
    # rebuild on an answer the room no longer wants. `preempted_by` is the ceiling of the turn
    # that stopped it; set ONLY by `preempt_older_same_sender`, it is how main.py tells this
    # cancellation from a shutdown.
    sender_id: Optional[str] = None
    task: Any = field(default=None, repr=False)
    turn: Any = field(default=None, repr=False)
    preempted_by: Optional[str] = None
    # Set when the turn reaches its own cleanup. A turn already ending is never preempted: a
    # cancellation landing in its finally would skip the very cleanup a stand-down relies on.
    ending: bool = False
    # Set synchronously when the turn admits its first tool call (a local flight, a hosted tool,
    # a work claim). Such a turn may have spawned work that outlives a cancel of its own task,
    # so it is never preempted — eligibility, not armor.
    tools_started: bool = False
    # The stand-down's single chrome-cleanup task, once a preemption has been consumed. A later
    # cancellation (shutdown) waits for THIS task and re-raises; it never starts another.
    stand_down: Any = field(default=None, repr=False)
    # A stand-down asked for while the turn was mid-mutation of Slack: the newer turn's ceiling,
    # acted on when that mutation exits (and dropped there if the turn is no longer eligible).
    preempt_requested_by: Optional[str] = None
    _watermarks: Any = field(default=None, repr=False)
    _closed: bool = field(default=False, repr=False)
    # WHY this turn was suppressed, kept from the first refusal. A turn is refused once and then
    # refused again at every later surface it tries, and those later refusals used to carry no
    # evidence at all — a dozen log lines reading "newer message None in None supersedes this
    # turn", which names neither the message that superseded it nor the scope it happened in. The
    # fact does not change after the first refusal, so it is remembered rather than re-derived
    # (the watermark entry may be gone by then; the turn that raised it has closed its lease).
    _suppressed_scope: Optional[Scope] = field(default=None, repr=False)
    _suppressed_latest_ts: Optional[str] = field(default=None, repr=False)
    # The suppressing scope's EFFECTIVE baseline at suppression time (§4a). After a rearm the
    # scalar `last_seen_ts` is no longer what the check measured against, so a rethrow that
    # reconstructed the exception from the scalar would report the wrong value; the baseline
    # is remembered beside the other evidence and reported on every rethrow.
    _suppressed_baseline: Optional[str] = field(default=None, repr=False)
    # Per-scope reviewed baselines (§4a). Populated ONLY by rearm_after_reconsideration: a
    # reconsideration pass that examined the conversation through these timestamps. Empty until
    # the first rearm, and with the map empty authorize() is byte-equivalent to the pre-rearm
    # guard.
    _reviewed_through: Dict[Scope, str] = field(default_factory=dict, repr=False)
    # Set only by force_after_reconsideration: the model's recorded judgment that this reply
    # goes out without another staleness check. Skips ONLY the newer-message comparison.
    _force_waiver: bool = field(default=False, repr=False)
    # Unforgeable identity: every StaleSendSuppressed this lease raises carries it, and
    # rearm/force accept only an exception whose token IS this object.
    _token: object = field(default_factory=object, repr=False)
    # Ever refused by authorize(). A lease that was suppressed belongs to its reconsideration
    # runner from then on — even after a rearm puts it back to PENDING — and is never preempted.
    _ever_suppressed: bool = field(default=False, repr=False)

    # --- what this turn has accounted for -------------------------------------------------

    def advance_last_seen(self, ts: Any) -> None:
        """Account for an inbound message this turn is actually answering. Monotonic: a retry
        or a fallback may recompute the request, and the mark must never slide backwards."""
        if is_newer(ts, self.last_seen_ts):
            self.last_seen_ts = str(ts)

    def owns(self, ts: Any) -> bool:
        """Is this source at or below the turn's ceiling — i.e. assigned to THIS turn?"""
        if self.ceiling_ts is None or ts is None:
            return True
        return not is_newer(ts, self.ceiling_ts)

    # --- the authorization check ----------------------------------------------------------

    def authorize(self, surface: str) -> None:
        """Permit or refuse the first visible surface of this turn. Called SYNCHRONOUSLY,
        immediately before the Slack mutation — every await between the check and the call is
        window this cannot cover.

        `committed` allows everything that follows: once a message is up, the rest of the
        answer belongs with it. `suppressed` refuses forever — a turn does not get a second
        opinion because it tried a different surface."""
        if self.state == COMMITTED:
            return
        if self.state == SUPPRESSED:
            # The ORIGINAL evidence, rethrown: the surface differs, the reason does not. The
            # baseline is the suppressing scope's EFFECTIVE one, remembered at suppression
            # time — after a rearm the scalar `last_seen_ts` would be the wrong value.
            raise StaleSendSuppressed(
                scope=self._suppressed_scope, last_seen_ts=self._suppressed_baseline,
                observed_latest_ts=self._suppressed_latest_ts, surface=surface,
                lease_token=self._token)
        if self._force_waiver:
            # force_after_reconsideration: the newer-message comparison — and ONLY that — is
            # skipped for every mutation of this one logical delivery.
            return
        # Each scope's watermark is measured against that scope's EFFECTIVE baseline —
        # max(last_seen_ts, reviewed-through) — and among the scopes whose candidate exceeds
        # it, the NEWEST candidate is the suppression evidence. With no rearm on record every
        # baseline is last_seen_ts and this is exactly the original newest-selection check.
        latest: Optional[str] = None
        scope: Optional[Scope] = None
        baseline: Optional[str] = None
        if self._watermarks is not None:
            for candidate_scope in self.scopes:
                candidate = self._watermarks.latest_for(candidate_scope)
                effective = self._effective_baseline(candidate_scope)
                if is_newer(candidate, effective) and is_newer(candidate, latest):
                    latest, scope, baseline = candidate, candidate_scope, effective
        if latest is not None:
            self.state = SUPPRESSED
            self._ever_suppressed = True
            self._suppressed_scope = scope
            self._suppressed_latest_ts = latest
            self._suppressed_baseline = baseline
            logger.info(
                f"Stale send suppressed ({surface}): {scope} advanced to {latest}, "
                f"this turn had seen {baseline}")
            raise StaleSendSuppressed(
                scope=scope, last_seen_ts=baseline,
                observed_latest_ts=latest, surface=surface,
                lease_token=self._token)

    def _effective_baseline(self, scope: Scope) -> Optional[str]:
        """What this turn has accounted for IN THIS SCOPE: the newer of the admission-time mark
        and the scope's reviewed-through baseline (populated only by rearm)."""
        reviewed = self._reviewed_through.get(scope)
        if reviewed is None:
            return self.last_seen_ts
        if self.last_seen_ts is None or ts_key(reviewed) >= ts_key(self.last_seen_ts):
            return reviewed
        return self.last_seen_ts

    def observed_latest(self) -> Tuple[Optional[str], Optional[Scope]]:
        """The newest ts any of this lease's scopes has seen, and which scope saw it."""
        if self._watermarks is None:
            return None, None
        newest: Optional[str] = None
        found: Optional[Scope] = None
        for scope in self.scopes:
            candidate = self._watermarks.latest_for(scope)
            if is_newer(candidate, newest):
                newest, found = candidate, scope
        return newest, found

    def newer_responding_ts(self) -> List[str]:
        """The ceilings of OTHER open, responding leases in this lease's scopes that are newer
        than this lease's own ceiling — the newer messages that already have their own reply
        attempt running. Compared on the immutable `ceiling_ts`, never on last_seen or a
        reviewed-through baseline: the question is which message each turn OWNS. A suppressed
        lease still counts — it may yet post through its own reconsideration."""
        if self._watermarks is None:
            return []
        found: Dict[str, None] = {}
        for other in self._watermarks.leases_for(self.scopes):
            if (other is self or other._closed or not other.responding
                    or other.ceiling_ts is None):
                continue
            if is_newer(other.ceiling_ts, self.ceiling_ts):
                found[other.ceiling_ts] = None
        return sorted(found, key=ts_key)

    def newer_deciding_ts(self) -> List[str]:
        """The ceilings of OTHER open leases in this lease's scopes, newer than this one, that are
        NOT responding — a newer message whose turn is still deciding (the gate, the lock) and so
        may yet own an answer. Same comparison as `newer_responding_ts`, on `ceiling_ts`."""
        if self._watermarks is None:
            return []
        found: Dict[str, None] = {}
        for other in self._watermarks.leases_for(self.scopes):
            if (other is self or other._closed or other.responding
                    or other.ceiling_ts is None):
                continue
            if is_newer(other.ceiling_ts, self.ceiling_ts):
                found[other.ceiling_ts] = None
        return sorted(found, key=ts_key)

    def preempt_older_same_sender(self) -> List[str]:
        """EARLY STAND-DOWN: this turn is now actually answering, so stop every OLDER turn of the
        SAME sender in a shared scope that has put nothing in the room yet. Returns the ceilings
        it preempted.

        Only a PENDING lease that was never suppressed (a suppressed one is its reconsideration
        runner's — redo and the tooled pass included), whose turn has no visible action — no
        detached producer or background job, no reaction, no edit — and whose task is not this
        one. A different sender is never preempted: cross-author thread replies keep today's
        suppress-and-reconsider path. Synchronous; the cancelled turn ends through its own
        cleanup, which reads `preempted_by` to know why."""
        if self._watermarks is None or not self.sender_id or self.ceiling_ts is None:
            return []
        current = _current_task()
        preempted: List[str] = []
        for other in self._watermarks.leases_for(self.scopes):
            if (other is self or other.sender_id != self.sender_id
                    or other.ceiling_ts is None
                    or not is_newer(self.ceiling_ts, other.ceiling_ts)
                    or not other._preemptable()):
                continue
            task = other.task
            if task is None or task is current or task.done():
                continue
            if mutation_in_flight(task):
                # Mid-mutation of Slack: deferred, not dropped. The mutation's own exit stands
                # this turn down if it is still eligible then (`_run_deferred_preemption`).
                if other.preempt_requested_by is None:
                    other.preempt_requested_by = self.ceiling_ts
                continue
            other._stand_down(self.ceiling_ts, task)
            preempted.append(other.ceiling_ts)
        return preempted

    def _preemptable(self) -> bool:
        """Could this turn stand down right now? Open, not already preempted or ending, no local
        tool started, PENDING and never suppressed, and nothing shown — a 👀 claim aside, which
        its stand-down takes back."""
        return (not self._closed and self.preempted_by is None and not self.ending
                and not self.tools_started and self.state == PENDING
                and not self._ever_suppressed and not _has_visible_effects(self.turn))

    def _stand_down(self, by: str, task: Any) -> None:
        self.preempted_by = by
        self.preempt_requested_by = None
        task.cancel()
        logger.info(f"Early stand-down: turn for {self.ceiling_ts} preempted by the same "
                    f"sender's newer turn {by}")

    # --- reconsideration (§4a) ------------------------------------------------------------

    def validate_suppression(self, expected: StaleSendSuppressed) -> None:
        """Assert the live suppression is still exactly `expected` — the interim-post check
        (burst follow-ups R2-1). Same identity preconditions as rearm/force, and like them it
        raises ValueError on any failure; unlike them it changes NOTHING: the lease stays
        suppressed, so the answer itself still has to pass the rearm → deliver path."""
        if self.state != SUPPRESSED:
            raise ValueError(f"interim requires a suppressed lease, not {self.state}")
        if self._closed:
            raise ValueError("interim refused: the lease is closed")
        if expected.lease_token is not self._token:
            raise ValueError("interim refused: exception is not from this lease")
        if (expected.scope != self._suppressed_scope
                or expected.observed_latest_ts != self._suppressed_latest_ts):
            raise ValueError(
                "interim refused: exception evidence does not match this lease's suppression")

    def rearm_after_reconsideration(self, reviewed_through: Mapping[Scope, str],
                                    expected: StaleSendSuppressed) -> None:
        """A reconsideration pass reviewed the conversation through these per-scope
        timestamps; re-open this suppressed lease so the next authorize() measures against
        them.

        The reviewed-through values are computed by trusted runtime snapshot code, never
        supplied by the model. Every precondition failure leaves the lease UNCHANGED and
        raises ValueError — fail closed, the runner drops."""
        from slack_client.normalizer import parse_ts  # local: messaging.py imports this module

        # 1. Only a suppressed, still-open lease can be rearmed.
        if self.state != SUPPRESSED:
            raise ValueError(f"rearm requires a suppressed lease, not {self.state}")
        if self._closed:
            raise ValueError("rearm refused: the lease is closed")
        # 2. Identity binding: the exception must be the one THIS lease last raised.
        if expected.lease_token is not self._token:
            raise ValueError("rearm refused: exception is not from this lease")
        if (expected.scope != self._suppressed_scope
                or expected.observed_latest_ts != self._suppressed_latest_ts):
            raise ValueError(
                "rearm refused: exception evidence does not match this lease's suppression")
        # 3. Exactly this lease's scope set — a missing scope fails closed, an extra one is
        # rejected.
        if set(reviewed_through.keys()) != set(self.scopes):
            raise ValueError(
                f"rearm refused: reviewed_through keys {sorted(reviewed_through)} do not "
                f"match lease scopes {sorted(self.scopes)}")
        # 4. Every value parses as a ts and is monotonic against its scope's effective
        # baseline.
        for scope, value in reviewed_through.items():
            try:
                parse_ts(value)
            except Exception as exc:
                raise ValueError(
                    f"rearm refused: unparseable reviewed_through for {scope}: "
                    f"{value!r}") from exc
            effective = self._effective_baseline(scope)
            if effective is not None and ts_key(value) < ts_key(effective):
                raise ValueError(
                    f"rearm refused: reviewed_through for {scope} ({value!r}) is behind its "
                    f"effective baseline ({effective!r})")
        # 5. The suppressing message itself is covered by the review. Precondition 2 already
        # matched `expected.scope` against this lease's live suppression, so a None here is a
        # corrupted exception — refused like every other precondition failure.
        suppressing_scope = expected.scope
        if suppressing_scope is None:
            raise ValueError("rearm refused: exception carries no suppressing scope")
        covering = reviewed_through[suppressing_scope]
        if ts_key(covering) < ts_key(expected.observed_latest_ts):
            raise ValueError(
                f"rearm refused: review through {covering!r} does not cover the suppressing "
                f"message {expected.observed_latest_ts!r}")

        for scope, value in reviewed_through.items():
            self._reviewed_through[scope] = str(value)
        self._suppressed_scope = None
        self._suppressed_latest_ts = None
        self._suppressed_baseline = None
        self.state = PENDING

    def force_after_reconsideration(self, expected: StaleSendSuppressed) -> None:
        """The model's recorded judgment that this reply goes out without another staleness
        check. Same identity preconditions as rearm; any failure leaves the lease unchanged
        and raises ValueError."""
        if self.state != SUPPRESSED:
            raise ValueError(f"force requires a suppressed lease, not {self.state}")
        if self._closed:
            raise ValueError("force refused: the lease is closed")
        if expected.lease_token is not self._token:
            raise ValueError("force refused: exception is not from this lease")
        if (expected.scope != self._suppressed_scope
                or expected.observed_latest_ts != self._suppressed_latest_ts):
            raise ValueError(
                "force refused: exception evidence does not match this lease's suppression")
        self._suppressed_scope = None
        self._suppressed_latest_ts = None
        self._suppressed_baseline = None
        self.state = PENDING
        self._force_waiver = True

    def cancel_force_waiver(self) -> None:
        """Revoke an outstanding force waiver — the runner's `delivery_exception` path (§4f).
        Idempotent, and touches nothing else: the waiver never survives its delivery."""
        self._force_waiver = False

    def commit(self) -> None:
        """A surface LANDED. Called only after Slack confirms — a definitive failure leaves the
        lease pending so a retry is checked again, rather than waved through."""
        if self.state != SUPPRESSED:
            self.state = COMMITTED
            # A force waiver covers exactly one logical delivery; the first confirmed surface
            # ends it and normal semantics resume.
            self._force_waiver = False

    @property
    def suppressed(self) -> bool:
        return self.state == SUPPRESSED

    @property
    def committed(self) -> bool:
        return self.state == COMMITTED

    def close(self) -> None:
        """Release this turn's hold on its scope entries. Idempotent. Also revokes any
        outstanding force waiver — the waiver never survives its delivery (§4a)."""
        if self._closed:
            return
        self._closed = True
        self._force_waiver = False
        # The task→lease map has weak keys but this lease holds its task strongly, so a closed
        # lease left in it would keep a finished turn's task alive. Removed only if it is still
        # THIS lease (a task may run more than one turn).
        task = self.task
        if task is not None:
            try:
                if _TASK_LEASES.get(task) is self:
                    del _TASK_LEASES[task]
            except TypeError:      # not weak-referenceable (a stand-in): never registered
                pass
        if self._watermarks is not None:
            self._watermarks.release(self)


def _current_task() -> Optional["asyncio.Task[Any]"]:
    try:
        return asyncio.current_task()
    except RuntimeError:          # no running loop (synchronous callers, tests)
        return None


# EARLY STAND-DOWN's acceptance window. Slack can accept a post, a stream start, a thinking
# indicator, a status card or a reaction while the lease is still PENDING and before the turn has
# written down what landed. A cancel in that window would orphan visible content with no receipt,
# no destination and no cleanup — so every transport method that mutates Slack marks its task as
# MID-MUTATION for its whole body (acceptance AND accounting), and preemption skips such a task.
_P = ParamSpec("_P")
_R = TypeVar("_R")
_VISIBLE_INFLIGHT: "weakref.WeakKeyDictionary[asyncio.Task[Any], int]" = (
    weakref.WeakKeyDictionary())


def visible_mutation(method: Callable[_P, Awaitable[_R]]) -> Callable[_P, Awaitable[_R]]:
    """Mark the running task as mid-mutation of Slack for the length of `method`."""
    @functools.wraps(method)
    async def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        task = _current_task()
        if task is None:
            return await method(*args, **kwargs)
        _VISIBLE_INFLIGHT[task] = _VISIBLE_INFLIGHT.get(task, 0) + 1
        try:
            return await method(*args, **kwargs)
        finally:
            remaining = _VISIBLE_INFLIGHT.get(task, 1) - 1
            if remaining > 0:
                _VISIBLE_INFLIGHT[task] = remaining
            else:
                _VISIBLE_INFLIGHT.pop(task, None)
                _run_deferred_preemption(task)
    return wrapper


# The lease each turn task opened, so a mutation's exit can find a stand-down it deferred.
_TASK_LEASES: "weakref.WeakKeyDictionary[asyncio.Task[Any], TurnSendLease]" = (
    weakref.WeakKeyDictionary())


def request_stand_down(task: Any, by: str) -> bool:
    """Ask the turn running in `task` to stand down because `by` covered it — the normal early
    stand-down, with the same eligibility and cleanup, deferred if it is mid-mutation. Returns
    whether a lease was found (the turn was admitted). An ineligible turn is left to the stale
    guard and reconsideration."""
    try:
        lease = _TASK_LEASES.get(task)
    except TypeError:
        return False
    if lease is None:
        return False
    if task.done() or not lease._preemptable():
        return True
    if mutation_in_flight(task):
        if lease.preempt_requested_by is None:
            lease.preempt_requested_by = by
        return True
    lease._stand_down(by, task)
    return True


def _run_deferred_preemption(task: "asyncio.Task[Any]") -> None:
    """The task's last Slack mutation just ended — accepted or refused, and accounted for. A
    stand-down deferred while it ran happens now if the turn is still eligible; the cancel lands
    at the task's next await. Never raises."""
    try:
        lease = _TASK_LEASES.get(task)
        if lease is None or lease.preempt_requested_by is None:
            return
        requested = lease.preempt_requested_by
        lease.preempt_requested_by = None
        if lease.task is task and not task.done() and lease._preemptable():
            lease._stand_down(requested, task)
    except Exception as e:  # noqa: BLE001 — a missed stand-down leaves today's path intact
        logger.debug(f"deferred stand-down skipped: {e}")


def mutation_in_flight(task: Any) -> bool:
    """Is this task inside a Slack mutation that has not been accepted/refused and accounted?"""
    try:
        return bool(task is not None and _VISIBLE_INFLIGHT.get(task))
    except TypeError:             # not weak-referenceable (a stand-in): never mid-mutation
        return False


def _has_visible_effects(turn: Any) -> bool:
    """Has this turn already put something in the room that its ending must account for?"""
    if turn is None:
        return False
    return bool(getattr(turn, "visible_action_committed", False)
                or getattr(turn, "reaction_committed", False)
                or getattr(turn, "destinations", None)
                or getattr(turn, "edits", None))


@dataclass
class _Entry:
    latest_ts: Optional[str] = None
    # The open leases holding this entry, keyed by each lease's private token. The entry lives
    # exactly as long as this map is non-empty.
    leases: Dict[object, TurnSendLease] = field(default_factory=dict)


class ConversationWatermarks:
    """Process-wide record of the newest inbound message per conversation scope.

    ONE instance, on ChatBotV2. Deliberately not in ThreadStateManager: a top-level burst is a
    stream of separate thread keys, so its per-thread locks cannot see the collision.

    The map is bounded by CONCURRENCY, not by a policy: a scope exists only while some turn in
    it holds a lease, and disappears when the last one closes. No timers, no caps, no sweeps.
    """

    def __init__(self) -> None:
        self._entries: Dict[Scope, _Entry] = {}

    # --- lifecycle ------------------------------------------------------------------------

    def begin_turn(self, message: Any) -> TurnSendLease:
        """Open a lease AND record this message as its conversation's newest.

        Called at the first executable line of handle_message, before the gate and before any
        await. That placement is the definition of "admitted": the watermark advances for
        exactly the messages that reached a turn, so a message dropped before dispatch — our own
        post, a lifecycle subtype, a participation-off channel, an app_mention duplicate — can
        never silence an answer, because it never gets here."""
        meta = getattr(message, "metadata", None) or {}
        ts = meta.get("ts")
        scopes = scopes_for_message(message)
        sender = meta.get("sender_id")
        task = _current_task()
        lease = TurnSendLease(scopes=scopes, last_seen_ts=str(ts) if ts else None,
                              ceiling_ts=str(ts) if ts else None,
                              sender_id=str(sender) if sender else None,
                              task=task, _watermarks=self)
        if task is not None:
            _TASK_LEASES[task] = lease
        for scope in scopes:
            entry = self._entries.get(scope)
            if entry is None:
                entry = self._entries[scope] = _Entry()
            entry.leases[lease._token] = lease
            if is_newer(ts, entry.latest_ts):
                entry.latest_ts = str(ts)
        return lease

    def release(self, lease: TurnSendLease) -> None:
        """Drop a lease's hold. An entry survives while ANY lease in its scope is open — a
        newer turn that finishes first must not erase the watermark the older turn is about to
        read, which is the whole point of holding rather than timing."""
        for scope in lease.scopes:
            entry = self._entries.get(scope)
            if entry is None:
                continue
            entry.leases.pop(lease._token, None)
            if not entry.leases:
                self._entries.pop(scope, None)

    # --- reads ----------------------------------------------------------------------------

    def latest_for(self, scope: Scope) -> Optional[str]:
        entry = self._entries.get(scope)
        return entry.latest_ts if entry is not None else None

    def leases_for(self, scopes: Tuple[Scope, ...]) -> List[TurnSendLease]:
        """Every open lease holding any of these scopes, each once."""
        seen: Dict[object, TurnSendLease] = {}
        for scope in scopes:
            entry = self._entries.get(scope)
            if entry is not None:
                seen.update(entry.leases)
        return list(seen.values())

    @property
    def tracked_scopes(self) -> int:
        """Live scope count — for tests and diagnostics; there is no policy attached to it."""
        return len(self._entries)
