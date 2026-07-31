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
from typing import Any, Dict, Optional, Tuple

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
                 observed_latest_ts: Any = None, surface: Optional[str] = None):
        self.scope = scope
        self.last_seen_ts = last_seen_ts
        self.observed_latest_ts = observed_latest_ts
        self.surface = surface
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
            # The ORIGINAL evidence, rethrown: the surface differs, the reason does not.
            raise StaleSendSuppressed(
                scope=self._suppressed_scope, last_seen_ts=self.last_seen_ts,
                observed_latest_ts=self._suppressed_latest_ts, surface=surface)
        latest, scope = self.observed_latest()
        if latest is not None and is_newer(latest, self.last_seen_ts):
            self.state = SUPPRESSED
            self._suppressed_scope = scope
            self._suppressed_latest_ts = latest
            logger.info(
                f"Stale send suppressed ({surface}): {scope} advanced to {latest}, "
                f"this turn had seen {self.last_seen_ts}")
            raise StaleSendSuppressed(
                scope=scope, last_seen_ts=self.last_seen_ts,
                observed_latest_ts=latest, surface=surface)

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

    def commit(self) -> None:
        """A surface LANDED. Called only after Slack confirms — a definitive failure leaves the
        lease pending so a retry is checked again, rather than waved through."""
        if self.state != SUPPRESSED:
            self.state = COMMITTED

    @property
    def suppressed(self) -> bool:
        return self.state == SUPPRESSED

    @property
    def committed(self) -> bool:
        return self.state == COMMITTED

    def close(self) -> None:
        """Release this turn's hold on its scope entries. Idempotent."""
        if self._closed:
            return
        self._closed = True
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
