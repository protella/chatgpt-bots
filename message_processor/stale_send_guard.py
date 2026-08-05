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
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

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

    # --- reconsideration (§4a) ------------------------------------------------------------

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
        if self._watermarks is not None:
            self._watermarks.release(self)


@dataclass
class _Entry:
    latest_ts: Optional[str] = None
    holders: int = 0


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
        scopes = scopes_for(getattr(message, "channel_id", None), ts,
                            getattr(message, "thread_id", None), meta.get("sender_id"))
        lease = TurnSendLease(scopes=scopes, last_seen_ts=str(ts) if ts else None,
                              ceiling_ts=str(ts) if ts else None, _watermarks=self)
        for scope in scopes:
            entry = self._entries.get(scope)
            if entry is None:
                entry = self._entries[scope] = _Entry()
            entry.holders += 1
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
            entry.holders -= 1
            if entry.holders <= 0:
                self._entries.pop(scope, None)

    # --- reads ----------------------------------------------------------------------------

    def latest_for(self, scope: Scope) -> Optional[str]:
        entry = self._entries.get(scope)
        return entry.latest_ts if entry is not None else None

    @property
    def tracked_scopes(self) -> int:
        """Live scope count — for tests and diagnostics; there is no policy attached to it."""
        return len(self._entries)
