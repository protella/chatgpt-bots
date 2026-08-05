"""F38 — what a turn is allowed to SHOW, and what it has CLAIMED.

Two questions used to be answered by the same overloaded value, `thinking_id`:

    thinking_id is not None  -> we have a placeholder message to edit
    thinking_id is None      -> ...one of three completely different things

`None` meant "setStatus worked, the composer status is the indicator" (DMs and channel
threads on the agent surface), and it ALSO meant "the indicator failed outright", and under
the deferral below it would have meant "we deliberately showed nothing". Downstream code
read `None` as the first of those and cheerfully pushed phase updates to setStatus — which
renders a thinking status AND auto-opens the thread. Deferring the placeholder without
disentangling this would have moved the flash, not removed it.

So the turn carries its own state:

* ``progress_enabled`` — may this turn show speculative "working on it" chrome at all?
  False on a turn that may decide to say nothing. Nothing renders until the turn commits.
* ``silence_capable`` — the same predicate that decides whether ``no_response_needed`` is
  exposed to the model. One value drives both, so the tool and the UI policy can never drift
  apart: if the model can stay quiet, we don't pre-announce that it won't.
* ``reply_destination`` + ``destination_source`` + ``destination_selected`` +
  ``destination_locked`` — WHERE the reply goes, WHO decided, whether the decision has been
  made yet, and whether it can still change. Destination used to be a boolean
  (``place_in_channel``) recomputed in four places from a channel setting, a gate verdict and
  a "did the turn do real work" heuristic — so the streaming coordinator, the footer, the
  wake envelope and the ledger could each believe something different about one reply. Now
  one turn states it once, and the MODEL makes the call on the only route where there is a
  call to make (`set_reply_destination`).
* ``ack_lease`` — the receipt for a 👀 this turn placed, and the only thing that lets it be
  taken back.

The 👀 rule (the user's, verbatim): "Other human teammates don't drop eyes and then do
nothing, that's misleading. If it adds it, it needs to do something, or go back and remove
it." So 👀 is not "seen" and not "thinking about it" — it is a CLAIM ON WORK. It goes on when
real work starts and comes off if that work evaporates.

P3 added the other half of that sentence: what the turn has CAUSED.

* ``_tool_flights`` — one dispatch per (turn, tool_call_id). A duplicate dispatch of a call id
  already seen receives the first call's outcome instead of doing the work again, because for the
  image tools "doing the work again" is a second picture, a second bill and a second post.
* ``_effect_leases`` — an irreversible effect and the bookkeeping that accounts for it, held as one
  critical section. Settlement waits for a held lease, so "an accepted post is always registered
  before this turn's receipts settle" is a property of the code rather than of the timing.
* ``effects_revoked`` — a straggler that outlived its own cancellation may cause nothing more. It
  stops the NEXT effect and never interrupts one in flight; interrupting is the failure the lease
  exists to prevent.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from config import config
from logger import setup_logger
from message_processor import outbound_receipts, participation_telemetry
from slack_client.normalizer import TimestampError, parse_ts

logger = setup_logger(name="slack_bot.TurnRuntime")

# §2g. The ONLY tools whose results may widen this turn's post_to_thread targets. Every one of
# them returns content Slack authorized under the same both-members rule the read itself ran
# through, so the root it names is one the room genuinely holds. A name outside this set enrolls
# nothing — not because those tools are untrusted, but because a root they mention was never the
# thing they were asked for, and "the model saw a ts somewhere" is precisely what may not
# authorize a post.
DISCOVERY_SOURCES = frozenset({"fetch_thread_messages", "fetch_channel_history", "search_slack"})

# How long a straggling tool flight gets to unwind AFTER it has been cancelled. Cancellation is
# established, not requested: the turn waits for the CancelledError to actually land, because the
# alternative is settling this turn's receipts while a task that can still post is running.
TOOL_FLIGHT_CANCEL_GRACE_SECONDS = 5.0

# NO fixed bound on waiting for a held lease. There was one (30s), and it was shorter than a
# single Slack multipart send can legitimately take — so settlement could finalize a turn's
# receipts while its own post was still being accepted, which is the exact race the lease exists
# to close. Each lease body carries the deadline of its OWN transport (the Slack/HTTP client
# timeout every effect runs under), so waiting for lease COMPLETION terminates on its own.


class EffectRevoked(Exception):
    """This turn will not authorize NEW irreversible effects.

    Raised by ``run_leased_effect`` when the lease is refused, so the executor answers the model
    honestly instead of causing an effect the turn can no longer account for. It is NOT raised
    into a lease already held: revocation stops the next one, it never interrupts one in flight.
    """


async def run_effect(turn: Any, site: str,
                     factory: Callable[[], Awaitable[Any]]) -> Any:
    """Run one irreversible effect under ``turn``'s effect lease — the ONE definition of that.

    Used by every executor that causes something it cannot take back (a post, an upload, a
    container write), so "leased" means the same thing at each of them. Raises ``EffectRevoked``
    when the turn has withdrawn permission; with no turn able to hold a lease — a background
    agent's own registry, a stand-in object in a test — the effect runs exactly as it did before,
    once.
    """
    runner = getattr(turn, "run_leased_effect", None) if turn is not None else None
    if runner is None:
        return await factory()
    leased = runner(site, factory)
    if not hasattr(leased, "__await__"):
        return await factory()
    return await leased


def revoke_turn_effects(turn: Any, reason: str) -> bool:
    """Withdraw this turn's permission to cause NEW effects — the ONE definition of that.

    Called from the two lifecycle fences (the handler's pre-extraction drain, the outer
    finalizer) when the bookkeeping that proves quiescence has itself failed. At that point what
    is still running is unknown, and settling receipts around unknown state is how a post ends up
    outside every account of the turn. Revocation is what makes the unknown harmless: whatever is
    alive can no longer take a lease.

    Returns whether this turn is now unable to cause a new effect — True for a stand-in object
    with no revocation to perform, because a turn that holds no effects has nothing to withdraw.
    False is the fence's own last resort having failed, and a caller must not step over it: with
    unknown state that is also UNREVOKED, there is nothing left that makes settling honest.

    Never raises: the fence that calls this has no fallback left to raise into."""
    revoke = getattr(turn, "revoke_effects", None) if turn is not None else None
    if revoke is None:
        return True
    try:
        revoke(reason)
    except Exception as e:  # noqa: BLE001 — the fence's own last resort has no fallback left
        logger.error(f"Effects could not be revoked ({reason}): {e!r}")
        return False
    return True


async def await_turn_effects(turn: Any, reason: str) -> bool:
    """Wait out the leases this turn is ALREADY holding — the ONE definition of that.

    Revocation blocks the NEXT effect; it never interrupts one in flight. So a fence that has
    just revoked has stopped nothing that was already running, and what it still holds is an
    accepted post or upload whose bookkeeping is mid-write. `wait_for_effects` is
    completion-bound — every lease body carries its own transport's deadline — so this
    terminates without a second clock of our own.

    Never fails a turn: a stand-in object with no leases has nothing to wait for."""
    waiter = getattr(turn, "wait_for_effects", None) if turn is not None else None
    if waiter is None:
        return True
    try:
        held = waiter()
        if hasattr(held, "__await__"):
            await held
    except Exception as e:  # noqa: BLE001 — same fence, same absence of a fallback
        logger.error(f"Held effects could not be waited out ({reason}): {e!r}")
        return False
    return True


class TurnEffectsUnsettled(Exception):
    """Both fence moves failed: effects are in unknown state AND could not be revoked/awaited.

    Raised by the handler fence so result extraction never runs over unrevoked unknown state;
    the turn ends through the ordinary error path instead of describing work it cannot vouch
    for."""


class LaunchNotRecorded(Exception):
    """The launch transition could not be recorded, so the irreversible step must not happen.

    Raised by ``mark_tool_launched`` and turned into an honest tool error by each executor. The
    alternative — issuing the request anyway — leaves the one mechanism that stops a second
    dispatch broken at exactly the moment it was carrying weight.
    """


def mark_tool_launched(ctx: Any) -> None:
    """This call's side-effect request is being issued NOW — the ONE definition of that moment.

    Called ATOMICALLY immediately before the irreversible step (the Slack post, the image
    request, the detached job), by every executor that has one, so "launched" means the same
    thing at each of them: a duplicate dispatch of this call id must never issue a second one,
    and a cancellation from here on must never release the key.

    Raises whatever the flight raises. A tool that could not record its own launch is plumbing
    that has failed exactly where it was load-bearing, and running the irreversible step anyway
    is how one turn pays twice. With no flight on the context — no call id, a legacy dispatch,
    a background agent — there is nothing to record and nothing to fail.
    """
    flight = getattr(ctx, "tool_flight", None)
    if flight is None:
        return
    try:
        flight.mark_launched()
    except Exception as e:  # noqa: BLE001 — reported as its own refusal, never stepped over
        raise LaunchNotRecorded(str(e)) from e


@dataclass
class ToolFlight:
    """One (turn, tool_call_id) dispatch — the in-turn retry key (spec §6 d2).

    The point is not concurrency, it is IDENTITY: a call id that has already been dispatched must
    never be dispatched a second time, because for the image tools the second dispatch is a second
    picture, a second API bill and a second post. Nothing in production replays a round today
    (tool_loop dispatches each call once; container recovery retries the API request, not the
    tools; the timeout fallback drops the registry) — this is what stops the first mechanism that
    does from double-firing a generation.

    `deadline` is stamped ONCE, by the registry, from the tool's own resolved timeout — the
    handler and the outer finally read that stamp rather than re-deriving a bound from config,
    and a duplicate dispatch inherits it and can neither extend nor overwrite it.
    """

    tool_name: str
    fingerprint: str
    timeout: float
    deadline: float
    task: Optional[asyncio.Task] = field(default=None, repr=False)
    # True once the executor has issued the side-effect request it exists to protect (the detached
    # job scheduled, the image API request sent). Before it, a failure or a cancellation means
    # nothing happened and the key is cleared so a fresh call may try again; after it, the key
    # stays owned forever — a duplicate must never relaunch an effect already issued.
    launched: bool = False
    # §2g. Thread roots THIS EXECUTION's result claims, staged by the executor and committed by
    # whichever waiter selects the flight's real result for the model.
    #
    # ON THE FLIGHT RATHER THAN THE PER-CALL CONTEXT, and that is the whole point: one flight can
    # have SEVERAL waiters (a duplicate dispatch of the same call id joins it instead of running
    # the work again), and only the waiter that CREATED it has a per-call context. Staging held
    # per-context meant that if the original waiter was cancelled and a duplicate received the
    # completed result, the model was shown a result that granted nothing. The work is one
    # execution, so its claims are one list. Sibling isolation is unaffected — different calls
    # are different flights.
    staged_roots: List[Any] = field(default_factory=list, repr=False)

    def mark_launched(self) -> None:
        """Called by the executor ATOMICALLY immediately before the side-effect request."""
        self.launched = True


    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    @property
    def settled(self) -> bool:
        return self.task is not None and self.task.done()

    @property
    def pending(self) -> bool:
        return self.task is not None and not self.task.done()

# What a destination record can be. `observed` — Slack accepted a surface, so the room saw
# something land there. `committed` — that surface reached its final text. The two are separate
# states rather than one flag because an interrupted stream is genuinely both: it was seen, and
# it never finished, and a turn that reported only one of those would be lying either way.
DESTINATION_OBSERVED = "observed"
DESTINATION_COMMITTED = "committed"

# How the words got there. Enumerated so the ledger's `kind` column cannot drift into free text.
DEST_KIND_REPLY = "reply"
DEST_KIND_STREAM = "stream"
DEST_KIND_SPLIT = "split"
DEST_KIND_POST_TO_THREAD = "post_to_thread"
DEST_KIND_RECONCILED = "reconciled"

# The registry name of the tool that produces that record. Stated here, next to the kind, because
# the responder needs the NAME (to decide whether the cross-thread conduct paragraph rides this
# turn) and the ledger needs the KIND, and they are one string — a second literal in a second
# module is how the paragraph would end up describing a tool that is not on the table.
POST_TO_THREAD_TOOL = DEST_KIND_POST_TO_THREAD

# Where a reply can go. `dm` is its own value rather than a flavour of `thread`: a DM has no
# other option, and collapsing it into `thread` loses the distinction between "there was no
# choice" and "the choice was made and came out this way".
DESTINATION_DM = "dm"
DESTINATION_THREAD = "thread"
DESTINATION_CHANNEL = "channel"
DESTINATIONS = frozenset({DESTINATION_DM, DESTINATION_THREAD, DESTINATION_CHANNEL})
# The values the MODEL may choose between. A DM is not on the menu — nothing to decide.
SELECTABLE_DESTINATIONS = (DESTINATION_THREAD, DESTINATION_CHANNEL)

SOURCE_STRUCTURAL = "structural"
SOURCE_MODEL = "model"
SOURCE_DEFAULT = "default"


@dataclass
class DestinationRecord:
    """One place this turn's own words landed.

    `text` is held for the memory extractor, which needs the exchange and not a length, and is
    deliberately absent from `as_payload` — the ledger records that a reply of N characters went
    somewhere, never the reply.

    `complete` is the transport's own verdict on whether Slack took ALL of it. A multipart send
    that aborted partway commits the prefix that landed, so `text`/`chars` describe the room
    rather than the intention — and this flag is how a reader tells that apart from a whole one.
    """

    channel_id: Optional[str]
    thread_root_ts: Optional[str]
    first_ts: Optional[str]
    kind: str
    state: str = DESTINATION_OBSERVED
    chars: Optional[int] = None
    complete: bool = True
    text: Optional[str] = field(default=None, repr=False)

    @property
    def committed(self) -> bool:
        return self.state == DESTINATION_COMMITTED

    def as_payload(self) -> Dict[str, Any]:
        payload = {"channel_id": self.channel_id, "thread_root_ts": self.thread_root_ts,
                   "first_ts": self.first_ts, "state": self.state, "chars": self.chars,
                   "kind": self.kind}
        # Written only when it is news. A truncated delivery is the exceptional case, and the
        # ordinary row keeps the shape every existing reader was pinned against.
        if not self.complete:
            payload["complete"] = False
        return payload


@dataclass
class ModelAttempt:
    """One Responses API attempt on a channel turn, for the `model_response` event W3a emits.

    Sequenced per turn rather than globally: the question the ledger answers is "how many calls
    did this one turn take, and did any of them fork", which a process-wide counter cannot.
    """

    attempt_seq: int
    model: Optional[str] = None
    fork_reason: Optional[str] = None
    status: Optional[str] = None
    cached_input_tokens: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


@dataclass
class ReconsiderFacts:
    """How this turn's stale-draft reconsideration ended, for the `turn_outcome` event.

    Stamped on `TurnRuntime.reconsider` by the reconsideration runner before every return,
    rethrow, and cancellation propagation — so `turn.reconsider is not None` doubles as the
    once-per-turn gate. `passes` is the started-pass count, the same number the
    `reconsider_outcome` event records.
    """

    outcome: str
    passes: int
    forced: Optional[bool] = None
    error: Optional[str] = None

    def as_payload(self) -> Dict[str, Any]:
        """The nested dict `turn_outcome` carries. Inapplicable keys are OMITTED — `forced`
        only on posted outcomes, `error` only on `error_dropped` — because record() strips
        top-level Nones but nested values survive verbatim, and a nested null would give a
        group-by two buckets meaning the same thing."""
        payload: Dict[str, Any] = {"outcome": self.outcome, "passes": self.passes}
        if self.outcome in ("posted_asis", "posted_revised") and self.forced is not None:
            payload["forced"] = self.forced
        if self.outcome == "error_dropped" and self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass
class TurnRuntime:
    """Per-turn presentation + work-claim state. Created in main.py, threaded to the handlers."""

    silence_capable: bool = False
    progress_enabled: bool = True
    reply_thread_id: Optional[str] = None
    # WHO owns the messages this turn posts, for outbound receipts (spec §5). Minted for every
    # turn, not only receipt-carrying ones: the session half is what dead-session reconciliation
    # matches on, and a turn without one could never be told from a dead session's leftovers.
    turn_id: str = field(default_factory=outbound_receipts.next_turn_id)
    # Bound right after for_message, once the team id is known. None in a DM (nothing to record).
    receipt_ledger: Any = field(default=None, repr=False)
    # THIS TURN's capability profile — the composed thread config, resolved once at admission and
    # never re-read. It picks the model the turn trims against, whether an attachment is mounted
    # for the sandbox, and which tools are on the table; every retry path (context compaction,
    # MCP fallback, the streaming/non-streaming fork, the timeout retry) re-enters the handlers,
    # and a per-attempt read let a settings change land mid-turn and give one request old-model
    # trimming with new-model tools.
    capability_profile: Optional[dict] = field(default=None, repr=False)
    # --- where this turn's reply goes, stated rather than inferred ---
    # WHERE: "dm" | "thread" | "channel". Always populated, from the first moment of the turn.
    reply_destination: str = DESTINATION_THREAD
    # WHO decided: "structural" (the route left no choice), "model" (it called the tool), or
    # "default" (it was offered the choice and did not take it).
    destination_source: str = SOURCE_STRUCTURAL
    # False ONLY while an eligible top-level turn is still waiting for the model to choose. It
    # is the flag that keeps a surface from being minted in a place the answer may not go.
    destination_selected: bool = True
    # True once a reply surface exists. After that the destination is a fact about Slack, not a
    # preference — a late change would leave a message stranded in the other place.
    destination_locked: bool = False
    # The model was offered the choice, produced words, and never called the tool. Recorded (not
    # corrected): the answer is delivered in the default thread, and the miss is a prompt problem
    # worth counting rather than a delivery problem worth guessing about.
    destination_contract_miss: bool = False
    # Did THIS turn put one of our emoji on a message? Set by the react tool at the moment the
    # reaction lands, because that is the only place every reacting path passes through. It used
    # to be derived per-Response in the handlers, and only the no-reply branch actually built
    # the field — so a reaction-only turn and a reply that also reacted both told the ledger
    # "no emoji" while the reaction event sat two lines above it in the same file.
    reaction_committed: bool = False
    # The turn's stale-send lease (message_processor.stale_send_guard), attached in
    # handle_message. Carried HERE because the turn is already threaded explicitly to every
    # path that can post — including ToolContext.turn, which is how post_to_thread gets it.
    # A global or a contextvar would attach itself to whatever coroutine happened to be
    # running, which for a bot answering several conversations at once is the wrong turn.
    send_lease: Any = field(default=None, repr=False)
    # How the stale guard protected this turn, recorded by the streaming handler when it binds
    # rather than re-derived at telemetry time. "buffered" — every word was held locally and one
    # guarded send made at the end, so the check spans the whole model call. "start_only" — the
    # answer streamed live, so only its first surface could be refused. Re-deriving it from
    # `silence_capable` got it wrong for an addressed CHANNEL reply, which also buffers (it has
    # no native path) and was being reported as start_only.
    guard_mode: Optional[str] = None
    # How this turn's stale-draft reconsideration ended, or None when none ran. Populated by the
    # reconsideration runner before every return, rethrow, and cancellation propagation — its
    # non-None-ness IS the once-per-turn gate the interception wrappers consult, and
    # emit_turn_outcome attaches its as_payload() to the turn_outcome event.
    reconsider: Optional[ReconsiderFacts] = None
    ack_lease: Optional[dict] = field(default=None, repr=False)
    ack_target_ts: Optional[str] = None
    # Where that claim was staked. Kept beside the ts purely so settle_ack can report a RETRACTED
    # 👀 against the right conversation — by then the Message that carried it is out of scope.
    ack_channel_id: Optional[str] = None
    # ...and which gate attempt it belonged to, read off the message in claim_work for the same
    # reason. None when the turn was not a gate attempt at all, which is also the signal that
    # neither the claim nor its retraction belongs in the participation ledger.
    ack_attempt_id: Optional[str] = None
    visible_action_committed: bool = False
    _claiming: bool = field(default=False, repr=False)
    # --- the channel stream this turn is answering from (spec §1-§4) ---
    # H, pinned at admission and NEVER refreshed. Every fetch, every window predicate and every
    # retry reads this one value, so two attempts of one turn cannot answer different questions.
    H: Optional[str] = None
    # The pinned ChannelStream. Held so a retry reuses the build rather than re-reading the world
    # — a rebuild would produce a different stream and answer a question nobody asked.
    channel_stream: Any = field(default=None, repr=False)
    # Everything else the assembler needs and must not re-derive per attempt (the pinned
    # steering snapshot, the origin slice, this turn's raw attachment parts, the cohort
    # fallbacks). Set once by base.py; read by both text.py paths.
    channel_turn_context: Any = field(default=None, repr=False)
    # This turn's resolved channel tool exposure and catalogs, so the request the admission
    # estimate measured and the request that gets sent are built from the same evidence — and so
    # the canvas API call behind the catalog happens once per turn rather than once per attempt.
    channel_prepared: Any = field(default=None, repr=False)
    # Who has spoken in the ORIGIN thread, `{user_id: name}`, read off the pinned stream. The
    # roster's tail: people who may not have spoken inside the window but are plainly part of the
    # conversation this turn is in.
    channel_origin_participants: Optional[Dict[str, str]] = field(default=None, repr=False)
    # §2g. Roots this turn's TOOL RESULTS proved exist, in THIS channel. Turn-local, never
    # persisted, monotone within the turn. Separate from the stream's own labels so the two
    # sources stay distinguishable in telemetry and in the failure message.
    _discovered_roots: Set[str] = field(default_factory=set, repr=False)
    # Did a stream build actually happen? Distinguishes "channel turn that rendered the room"
    # from one that failed closed before the fetch — a distinction turn_outcome reports.
    stream_build_present: bool = False
    # Which fail-closed condition ended this turn, as one of the three declared codes
    # (stream_data_invalid, stream_over_budget, history_fetch_failed). None on a turn that did
    # not fail that way.
    turn_error: Optional[str] = None
    # WHERE this turn's own words landed. Appended at the first Slack-accepted surface, marked
    # committed when that surface reaches its final text.
    destinations: List[DestinationRecord] = field(default_factory=list)
    # §5.4a exit-path amendment. THE PROVENANCE INPUTS THE TURN OWNS, accrued as tools run
    # rather than handed over when the loop returns.
    #
    # The loop's own `local_tool_calls` accumulator is a LOCAL of a function that must return
    # for anyone to see it — so a later Responses round that fails, a cancellation, or a settle
    # failure takes the whole record with it, including the calls that already had effects. A
    # cross-thread post commits mid-loop and is not retractable; the record of what produced it
    # cannot be, either. Same shape as the loop's entries ({name, ok, gist}), recorded at the
    # same moment, so the two lists union cleanly.
    provenance_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    # The SAME rule for the other half of the inputs (codex final-round-2 #1). External usage —
    # web_search, the sandbox, MCP servers — is counted in a loop accumulator and in the
    # streaming handler's own callback closure, and both are as lost to a failed round as the
    # local records were. Names only; these carry no arguments and so no gist.
    provenance_external_tools: List[str] = field(default_factory=list)
    # Per-attempt API records for W3a's `model_response`. The carrier is the turn itself: the
    # request wrappers reach it through the same explicit threading everything else uses.
    model_attempts: List[ModelAttempt] = field(default_factory=list)
    # --- what this turn has DISPATCHED and what it is still CAUSING (spec §6 d2) ------------
    # (turn_id, tool_call_id) -> ToolFlight. Lives here rather than on the AssetLedger because
    # the question it answers is "has THIS TURN already dispatched that call", and a per-thread
    # ledger outlives the turn. Keyed with turn_id in it so the key is self-describing in a log.
    _tool_flights: Dict[Tuple[str, Any], ToolFlight] = field(default_factory=dict, repr=False)
    # How many id-less calls this turn has taken ownership of, purely to mint keys that cannot
    # collide with each other or with any model-supplied id.
    _anonymous_flights: int = field(default=0, repr=False)
    # Effect leases currently HELD: id(task) -> (site, task). A lease is taken immediately before
    # an irreversible effect and held through Slack acceptance AND that effect's receipt
    # mechanism, as one critical section — so "an accepted post is always registered before
    # settlement" is a property of the code rather than a hope about timing.
    _effect_leases: Dict[int, Tuple[str, asyncio.Future]] = field(default_factory=dict, repr=False)
    # No NEW effect may be started. Set when a straggler resisted cancellation, so whatever it is
    # still running cannot post anything this turn will never account for. Never interrupts a
    # lease already held.
    effects_revoked: bool = False

    # --- discovered thread roots (§2g) -------------------------------------------------------

    def _own_channel_id(self) -> Optional[str]:
        """THIS turn's channel, or None when it has no channel identity at all (a DM, a
        background agent, a bare TurnRuntime in a test). Read off the pins rather than stored
        separately, so it cannot disagree with the stream the turn is answering from."""
        for holder, attr in ((getattr(self, "channel_turn_context", None), "channel_id"),
                             (getattr(getattr(self, "channel_stream", None), "pinned", None),
                              "channel_id")):
            value = getattr(holder, attr, None) if holder is not None else None
            if isinstance(value, str) and value:
                return value
        return None

    def enroll_discovered_root(self, *, channel_id: str, root_ts: str, source: str) -> bool:
        """Enroll ONE root, from ONE tool result. Returns True when it was newly enrolled.

        Refuses — returning False, logging at DEBUG, never raising — when the channel is not
        this turn's channel, when the ts does not parse, or when `source` is not one of the
        three enrolling tools. Refusing is always safe: the model gets `unknown_thread` and an
        honest message, which is a tool call, not a wrong post.

        A non-`str` ts is refused before it is parsed. `parse_ts` stringifies whatever it is
        given, so a float would slip through as a plausible ts read out of a JSON number, and a
        root is a Slack ts — a string — or it is not a root.
        """
        if source not in DISCOVERY_SOURCES:
            logger.debug(f"discovered root refused: {source!r} is not an enrolling tool")
            return False
        own = self._own_channel_id()
        if not own or not isinstance(channel_id, str) or channel_id != own:
            logger.debug(f"discovered root refused: {channel_id!r} is not this turn's channel")
            return False
        if not isinstance(root_ts, str):
            logger.debug(f"discovered root refused: {root_ts!r} is not a timestamp")
            return False
        try:
            parse_ts(root_ts)
        except TimestampError:
            logger.debug(f"discovered root refused: {root_ts!r} does not parse")
            return False
        if root_ts in self._discovered_roots:
            return False
        self._discovered_roots.add(root_ts)
        return True

    @property
    def discovered_thread_roots(self) -> frozenset:
        return frozenset(self._discovered_roots)

    # --- destination records ---------------------------------------------------------------

    def note_destination_observed(self, *, channel_id: Optional[str], first_ts: Optional[str],
                                  kind: str, thread_root_ts: Optional[str] = None
                                  ) -> DestinationRecord:
        """Slack accepted a surface.

        A record is identified by its PLACE — (channel, first ts) — not by how the words got
        there. A stream that flushes fifty times has landed in one place, and fifty records would
        make one reply look like fifty; and the same surface can be re-described as it goes (a
        stream that turns out to be multipart), which is a refinement of one record rather than a
        second delivery.
        """
        for record in self.destinations:
            if record.first_ts == first_ts and record.channel_id == channel_id:
                return record
        record = DestinationRecord(channel_id=channel_id, thread_root_ts=thread_root_ts,
                                   first_ts=first_ts, kind=kind)
        self.destinations.append(record)
        return record

    def mark_destination_committed(self, *, first_ts: Optional[str], kind: str,
                                   text: Optional[str] = None,
                                   complete: bool = True,
                                   channel_id: Optional[str] = None,
                                   thread_root_ts: Optional[str] = None) -> None:
        """That surface reached its final text. A commit for a surface nobody observed still
        records it — a delivery we somehow missed observing is a bookkeeping bug, and dropping
        the commit would hide the delivery too.

        `text` must be what SLACK ACCEPTED, never what the model wrote: this is the only text
        the memory extractor reads, and committing an intended answer that half-failed would
        have the bot remember a conversation the room never had."""
        record = self.note_destination_observed(
            channel_id=channel_id, first_ts=first_ts, kind=kind, thread_root_ts=thread_root_ts)
        record.kind = kind
        record.state = DESTINATION_COMMITTED
        record.complete = complete
        if text is not None:
            record.text = text
            record.chars = len(text)

    def note_tool_call(self, record: Dict[str, Any]) -> None:
        """Record one dispatched local tool call on the TURN, as it happens (§5.4a amendment).

        Called by the tool loop at the same moment it appends to its own accumulator, so the two
        records are the same fact written down twice — once where the handler can read it on a
        clean return, and once where it survives a round that never returns at all.

        Never raises and never rejects: this is a ledger, and a turn that could not write down
        what it just did is worse off than one that wrote down something odd.
        """
        if isinstance(record, dict) and record.get("name"):
            self.provenance_tool_calls.append(dict(record))

    def note_external_tools(self, names: Any) -> None:
        """Record external/server-side tool usage on the TURN, as it happens.

        Deduped and order-preserving, so the list reads the way the attribution line does. A
        name the loop or the stream callback has already reported is not repeated.
        """
        if isinstance(names, str):
            names = [names]
        for name in (names or ()):
            if name and name not in self.provenance_external_tools:
                self.provenance_external_tools.append(str(name))

    @property
    def committed_destinations(self) -> List[DestinationRecord]:
        return [r for r in self.destinations if r.committed]

    def next_model_attempt(self, *, model: Optional[str] = None,
                           fork_reason: Optional[str] = None) -> ModelAttempt:
        attempt = ModelAttempt(attempt_seq=len(self.model_attempts) + 1, model=model,
                               fork_reason=fork_reason)
        self.model_attempts.append(attempt)
        return attempt

    # --- tool flights: one dispatch per (turn, call id) --------------------------------------

    def open_tool_flight(self, *, call_id: Any, tool_name: str, fingerprint: str,
                         timeout: float) -> Tuple[Optional[ToolFlight], bool]:
        """Claim this call id for this turn. Returns (flight, created).

        `(None, False)` means the SAME id arrived describing a DIFFERENT call — a different tool,
        or different arguments. That is never served from the first call's result: the two calls
        want different things, and handing the second the first's answer would be a wrong answer
        wearing a successful shape.

        Synchronous on purpose, with no await between the lookup and the insert: a round's calls
        run concurrently (dispatch_all gathers them), so anything else would let two siblings
        both decide they were first.
        """
        key = (self.turn_id, str(call_id))
        existing = self._tool_flights.get(key)
        if existing is not None:
            if existing.fingerprint != fingerprint:
                return None, False
            return existing, False
        flight = ToolFlight(tool_name=str(tool_name), fingerprint=fingerprint,
                            timeout=float(timeout),
                            deadline=time.monotonic() + float(timeout))
        self._tool_flights[key] = flight
        return flight, True

    def open_anonymous_tool_flight(self, *, tool_name: str, timeout: float) -> ToolFlight:
        """Take LIFECYCLE ownership of a call that carries no id. Never deduped.

        A call with no id cannot be a retry key — two id-less calls are two different calls, and
        collapsing them would drop real work. But that was the whole reason they were left
        untracked, and untracked is the one state the turn cannot survive: when a sibling's
        suppression propagates out of the round's `gather`, the id-less call keeps running with
        nothing draining it, nothing cancelling it and nothing revoking it, and it can still post
        after the ledger has settled — a message the room sees and no receipt claims.

        So it gets a flight under a SYNTHETIC key: a counter, in its own namespace, so it can
        never be found by a lookup and never collide with a model-supplied id. Everything else —
        the stamped deadline, the drain, the cancellation, the revocation — is identical.
        """
        self._anonymous_flights += 1
        seq = self._anonymous_flights
        flight = ToolFlight(tool_name=str(tool_name), fingerprint=f"anonymous:{seq}",
                            timeout=float(timeout),
                            deadline=time.monotonic() + float(timeout))
        self._tool_flights[(self.turn_id, ("anonymous", seq))] = flight
        return flight

    def launch_tool_flight(self, flight: ToolFlight,
                           coro: Awaitable[Any]) -> asyncio.Task:
        """Own the work as a TASK, so a caller that stops waiting does not stop the effect."""
        task = asyncio.ensure_future(self._fly(flight, coro))
        flight.task = task
        return task

    async def _fly(self, flight: ToolFlight, coro: Awaitable[Any]) -> Any:
        try:
            return await coro
        except BaseException:
            # A RAISED failure or a cancellation BEFORE the launch means nothing happened, so the
            # key must not survive to serve that failure to a call that could still succeed. After
            # the launch the key stays owned: an effect was issued, and a duplicate that relaunched
            # it would double it. A plain {"ok": false} result is a COMPLETED call, not this — it
            # is cached like any other answer, because the model asked and got told.
            if not flight.launched:
                self._drop_tool_flight(flight)
            raise

    def abandon_tool_flight(self, flight: ToolFlight) -> None:
        """Release a key whose call never started (the registry's plumbing failed before launch).

        Refuses once launched: after the side-effect request the key is what stops a second one,
        and handing it back would be worse than the failure that asked for it."""
        if flight is not None and not flight.launched:
            self._drop_tool_flight(flight)

    def _drop_tool_flight(self, flight: ToolFlight) -> None:
        for key, held in list(self._tool_flights.items()):
            if held is flight:
                self._tool_flights.pop(key, None)

    @property
    def pending_tool_flights(self) -> List[ToolFlight]:
        return [f for f in self._tool_flights.values() if f.pending]

    async def await_tool_flights(self) -> None:
        """Wait for every unsettled flight, each to its OWN stamped deadline.

        Called by the handler immediately BEFORE it reads what the round produced — before the
        sandbox assets are copied, before the response is classified — because a flight the
        dispatch stopped waiting for (its bound expired) may still be mounting an image or
        posting one, and extracting results around it reports a turn that did less than it did.
        """
        for flight in sorted(self.pending_tool_flights, key=lambda f: f.deadline):
            task = flight.task
            if task is None or task.done():
                continue
            remaining = flight.remaining()
            if remaining <= 0:
                continue
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=remaining)
            except asyncio.TimeoutError:
                continue
            except Exception:  # noqa: BLE001 — the dispatch owns this call's failure, not us
                continue

    async def finish_tool_flights(
            self, *, grace: float = TOOL_FLIGHT_CANCEL_GRACE_SECONDS) -> Tuple[str, ...]:
        """Settle every flight before this turn's receipts do. Returns the survivors' names.

        Drains what is left to the stamped deadlines, then CANCELS the stragglers and waits for
        the cancellation to actually land — a requested cancellation is not an established one,
        and settling receipts while a task that can still post is unwinding is exactly the race
        the invariant forbids. Anything still alive after the grace is CRITICAL-logged by name and
        has its effect permission revoked, so it cannot post something nothing will account for.
        """
        await self.await_tool_flights()
        stragglers = [f for f in self.pending_tool_flights if f.task is not None]
        if not stragglers:
            return ()
        for flight in stragglers:
            flight.task.cancel()
        _done, still = await asyncio.wait([f.task for f in stragglers], timeout=max(0.0, grace))
        for task in _done:
            if not task.cancelled():
                task.exception()  # consumed: a straggler's own error is not an exit warning
        survivors = tuple(f.tool_name for f in stragglers
                          if f.task is not None and f.task in still)
        if survivors:
            logger.critical(
                "Turn %s: tool flight(s) %s suppressed cancellation past the %.1fs grace — "
                "revoking their permission to cause further effects; anything they post from "
                "here is outside this turn's accounting",
                self.turn_id, ", ".join(survivors), grace)
            self.revoke_effects("tool flight resisted cancellation")
        return survivors

    # --- effect leases: an accepted effect is always accounted for ---------------------------

    def revoke_effects(self, reason: str) -> None:
        """No NEW irreversible effect may start. Idempotent; never interrupts a held lease."""
        if not self.effects_revoked:
            self.effects_revoked = True
            logger.warning("Turn %s: effects revoked (%s)", self.turn_id, reason)

    async def run_leased_effect(self, site: str,
                                factory: Callable[[], Awaitable[Any]]) -> Any:
        """Run one irreversible effect, and its bookkeeping, as a leased critical section.

        Two things this buys that a bare `await` does not:

        * REVOCATION IS CHECKED HERE, immediately before the effect — so a turn that has already
          given up on this task cannot have it post anyway (`EffectRevoked`, which the executor
          turns into an honest tool error).
        * A CALLER THAT STOPS WAITING DOES NOT STOP THE EFFECT. The body runs as its own task and
          the lease is released when THAT task settles, not when the awaiter walks away — so a
          cancelled or timed-out executor still finishes accepting the post and recording its
          receipt, and settlement (which waits for held leases) waits for it.

        Compensating cleanup — deleting a checklist, tearing down a surface — is deliberately NOT
        gated on this: taking something back is never the effect this protects against.
        """
        if self.effects_revoked:
            raise EffectRevoked(site)
        task = asyncio.ensure_future(factory())
        token = id(task)
        self._effect_leases[token] = (site, task)
        try:
            return await asyncio.shield(task)
        finally:
            if task.done():
                self._effect_leases.pop(token, None)
            else:
                task.add_done_callback(lambda t, k=token: self._release_lease(k, t))

    def _release_lease(self, token: int, task: asyncio.Future) -> None:
        """The lease is over. Retrieve the body's exception even though nobody is waiting for it:
        an awaiter that walked away (a cancelled executor) leaves this task's failure unconsumed,
        and an abandoned effect must not announce itself as a stray asyncio warning at exit."""
        self._effect_leases.pop(token, None)
        if not task.cancelled():
            task.exception()

    @property
    def held_effect_leases(self) -> Tuple[str, ...]:
        return tuple(site for site, task in self._effect_leases.values() if not task.done())

    async def wait_for_effects(self) -> Tuple[str, ...]:
        """Wait for every held effect lease to COMPLETE. Returns () — nothing is still held.

        Deliberately unbounded. A lease is only ever held across an effect whose body is already
        bounded by its own transport (the Slack client's timeout, the container upload's), so this
        terminates without a second, shorter clock of our own invention on top — and a shorter one
        is worse than none: it hands back "still held" to a caller whose only remaining option is
        to settle anyway, which is precisely the receiptless post the lease prevents.

        A lease body that FAILS is not a reason to keep waiting: the task is done either way, and
        its exception belongs to the executor that ran it (consumed here so an abandoned effect
        cannot surface as an "exception was never retrieved" warning at interpreter exit).
        """
        while True:
            live = [task for _site, task in self._effect_leases.values() if not task.done()]
            if not live:
                return ()
            done, _still = await asyncio.wait(live)
            for task in done:
                if not task.cancelled():
                    task.exception()  # consumed: the executor owns this effect's failure

    @property
    def final_post_only(self) -> bool:
        """F39 — the "(edited)" rule. Slack can only STREAM into a thread: chat.startStream
        REQUIRES thread_ts. So a reply headed for the top level of a channel has no native path,
        and the legacy fallback fakes streaming by posting a stub and chat.update-ing it — which
        stamps the message "(edited)" forever. A human teammate doesn't post a stub and revise it
        in public; they post once, finished.

        So a channel-destined turn writes NOTHING until the answer is whole. DMs and threads are
        unaffected: they stream exactly as before. This is now derived rather than stored,
        because it was only ever a restatement of where the reply is going, and two fields for
        one fact is how they came to disagree."""
        return self.reply_destination == DESTINATION_CHANNEL

    @classmethod
    def for_message(cls, message: Any, *, channel_post_allowed: bool = False) -> "TurnRuntime":
        """Open a turn with its destination already stated.

        Three of the four routes have no decision to make, and say so with
        `destination_source="structural"`: a DM has nowhere else to go, a reply inside an
        existing thread belongs to that thread, and a channel that forbids top-level replies has
        settled the question in its settings. Only a top-level trigger in a channel that allows
        both is genuinely open — and that turn starts UNSELECTED, defaulting to the thread, and
        shows nothing anywhere until the model chooses.

        Silence-capable == the routing fact of the same name, and nothing else (routing_facts.py).
        The config switch stays here: the route says whether silence is allowed, the flag says
        whether the tool that performs it exists. Mirrors text.py::_materialize_request_tools."""
        meta = getattr(message, "metadata", None) or {}
        silence_capable = (meta.get("silence_capable") is True
                           and bool(getattr(config, "enable_no_reply_tool", True)))
        channel_id = str(getattr(message, "channel_id", "") or "")
        thread_id = getattr(message, "thread_id", None)

        # The shared discriminator, not a prefix test: an outbound DM is addressed with
        # channel=<user_id>, so a "U"/"W" id is the same surface with a different name.
        from slack_client.utilities import is_dm_conversation

        if channel_id and is_dm_conversation(channel_id, meta.get("channel_type")):
            destination, selected = DESTINATION_DM, True
        elif meta.get("ts") != thread_id or not channel_post_allowed:
            # An existing thread, or a channel whose settings forbid a top-level reply.
            destination, selected = DESTINATION_THREAD, True
        else:
            # Both destinations legal: the model decides, and until it does the answer is
            # provisionally headed for the thread.
            destination, selected = DESTINATION_THREAD, False

        return cls(
            silence_capable=silence_capable,
            # A turn that may say nothing shows no chrome; neither does one that cannot show
            # chrome without editing it into the answer afterwards; neither does one that does
            # not yet know WHERE its chrome would go.
            progress_enabled=not (silence_capable
                                  or destination == DESTINATION_CHANNEL
                                  or not selected),
            reply_thread_id=None if destination == DESTINATION_CHANNEL else thread_id,
            reply_destination=destination,
            destination_source=SOURCE_STRUCTURAL if selected else SOURCE_DEFAULT,
            destination_selected=selected,
        )

    def bind_receipts(self, client: Any, message: Any) -> Any:
        """Open this turn's receipt ledger. No-op in DMs and on unknown surfaces.

        A CHANNEL ledger that comes back inactive is a defect, not a mode: this turn's words
        will not be recorded as ours and the rebuilt stream will not have them. main.py refuses
        such turns up front; this says so where the ledger is actually built, because a ledger
        object that quietly writes nothing is indistinguishable from a working one.
        """
        try:
            self.receipt_ledger = outbound_receipts.ledger_for(
                self.turn_id, getattr(client, "self_team_id", None),
                getattr(message, "channel_id", None))
            if self.receipt_ledger is not None and not self.receipt_ledger.active:
                logger.error(
                    "Channel turn %s in %s has an INACTIVE receipt ledger (team=%r) — anything "
                    "it posts will be missing from the channel stream",
                    self.turn_id, getattr(message, "channel_id", None),
                    getattr(client, "self_team_id", None))
        except Exception as e:  # noqa: BLE001 — bookkeeping never blocks a turn from starting
            logger.debug(f"Receipt ledger not bound: {e}")
            self.receipt_ledger = None
        return self.receipt_ledger

    def select_destination(self, destination: Any, message: Any = None) -> dict:
        """The model's choice, from `set_reply_destination`. Returns the tool result.

        Refuses rather than guesses, in every direction: an unrecognized value, a second call
        that contradicts the first, and any call once a surface exists. The last one matters
        most — after a message is up, moving the destination would leave that message stranded
        in a place the rest of the answer is not going. An identical repeat is fine and changes
        nothing (models re-state decisions; that is not a conflict)."""
        if destination not in SELECTABLE_DESTINATIONS:
            return {"ok": False, "error": "invalid_destination",
                    "message": ("`destination` must be exactly one of "
                                f"{', '.join(SELECTABLE_DESTINATIONS)}.")}
        if self.destination_locked:
            return {"ok": False, "error": "destination_locked",
                    "message": ("The reply has already started going out — its destination "
                                "cannot change now.")}
        if self.destination_source == SOURCE_STRUCTURAL:
            # A DM, a thread, a channel that forbids top-level replies, or a turn that already
            # posted a notice. The route decided, so there is nothing to choose — and the tool
            # is not offered on these turns at all. This is the backstop for the gap between
            # those two facts: the registry checks `enabled` when it builds the SCHEMA set, not
            # again at dispatch, so a call that arrives anyway must be refused by the state
            # rather than silently overwrite it.
            return {"ok": False, "error": "destination_not_open",
                    "message": ("This reply's destination is already settled by where the "
                                "conversation is — there is nothing to choose here.")}
        if self.destination_selected and self.destination_source == SOURCE_MODEL:
            if destination == self.reply_destination:
                return {"ok": True, "destination": destination, "idempotent": True}
            return {"ok": False, "error": "destination_conflict",
                    "message": (f"You already chose `{self.reply_destination}` for this reply. "
                                "One destination per turn.")}
        self.reply_destination = destination
        self.destination_source = SOURCE_MODEL
        self.destination_selected = True
        self.reply_thread_id = (None if destination == DESTINATION_CHANNEL
                                else getattr(message, "thread_id", self.reply_thread_id))
        return {"ok": True, "destination": destination}

    def settle_structural_thread(self) -> None:
        """A visible surface already exists in the thread — a prior-timeout notice, a
        failed-files notice — so the question is settled by fact rather than by preference.

        Both notices post BEFORE the model runs. Leaving the turn open after one would let the
        model send the answer to the channel top level, splitting a single turn across two
        surfaces: a warning in the thread and the answer somewhere else. So the destination
        becomes the thread, STRUCTURALLY (the route left no choice — this is not a model that
        declined to choose, so it is not a contract miss), and it locks. The tool is not exposed
        on a settled turn, so the model is never offered a choice that no longer exists.

        On a CHANNEL turn this is called before the request is assembled, not after the notice
        lands [r3-3]: admission measures and pins the request, so a destination settled afterwards
        would leave the admitted request advertising a tool the sent request then refuses."""
        if self.destination_locked:
            return
        self.reply_destination = DESTINATION_THREAD
        self.destination_source = SOURCE_STRUCTURAL
        self.destination_selected = True
        self.destination_locked = True

    def settle_default_destination(self) -> None:
        """Words are arriving and the model never chose. The answer is NOT dropped and NOT
        guessed at from its text: it goes to the default (the thread), and the miss is recorded
        so a prompt that keeps producing it is visible in the ledger rather than only in the
        room."""
        if self.destination_selected:
            return
        self.destination_selected = True
        self.destination_source = SOURCE_DEFAULT
        self.destination_contract_miss = True

    def lock_destination(self) -> None:
        """A reply surface now exists. Idempotent; called from every path that mints one."""
        self.destination_locked = True

    def resolve_reply_target(self, message: Any) -> Optional[str]:
        """The thread_ts a reply should be posted with — None means top-level in the channel.

        A pure mapping from the stated destination, nothing more. It used to also carry a
        substantive-work override that re-threaded a channel reply after the fact, which is a
        heuristic second-guessing a decision the model is now asked to make outright."""
        if self.reply_destination == DESTINATION_CHANNEL:
            return None
        return getattr(message, "thread_id", None) if message is not None else self.reply_thread_id

    async def claim_work(self, client: Any, message: Any) -> None:
        """Real work is starting: stake the 👀 claim on the triggering message.

        Idempotent — many tools may call this in one turn, and exactly one reaction lands.
        Call it AFTER a tool's arguments and capacity checks pass and immediately BEFORE the
        slow part begins, never from a generic 'a tool was mentioned' hook: a rejected call
        (an invalid argument, a duplicate background job) must not flash an eye it is about
        to retract.

        Purely additive and fails silent — an emoji is never worth failing a turn over."""
        if self.ack_lease is not None or self._claiming:
            return
        if not getattr(config, "enable_ack_reaction", True):
            return
        meta = getattr(message, "metadata", None) or {}
        react_ts = meta.get("ts") or getattr(message, "thread_id", None)
        channel_id = getattr(message, "channel_id", None)
        if not react_ts or not channel_id:
            return
        if not hasattr(client, "_reserve_and_react_owned"):
            return
        # Read the attempt id HERE, while the Message is still in scope — settle_ack runs at the
        # end of the turn with nothing but this object, the same reason ack_channel_id is stashed.
        attempt_id = participation_telemetry.attempt_id_for(message)
        self.ack_attempt_id = attempt_id
        self._claiming = True  # before the await: concurrent tool calls must not double-add
        try:
            # BOUNDED. This runs inside the tool callback, so for a hosted tool the Responses
            # event loop is waiting on us — a wedged Slack call would stall the web search or
            # the code run it is announcing. The emoji must never hold up the work.
            _result, lease = await asyncio.wait_for(
                client._reserve_and_react_owned(
                    channel_id, react_ts, config.ack_reaction_emoji),
                timeout=config.tool_call_timeout)
            self.ack_lease = lease
            self.ack_target_ts = react_ts
            self.ack_channel_id = channel_id
            # The work-claim 👀 never goes through execute_react_tool, so without this it is the
            # one reaction the room can see that leaves no record. It also has to be TELLABLE
            # from a chosen emoji: it is a fixed operational marker, and counting it as taste
            # would flatten the diversity number every analysis is trying to read.
            #
            # A LEASE is the only proof we placed it. Without one the emoji was already up
            # there — a previous turn's, or the model's own react tool — and `ok` says only
            # that it is present, not that we put it there. Recording that as `added` would
            # inflate the claim rate with reactions the bot never made.
            if attempt_id:
                if lease is not None:
                    outcome, detail = "added", None
                elif isinstance(_result, dict) and _result.get("ok") is True:
                    outcome, detail = "already_present", None
                else:
                    outcome = "failed"
                    detail = _result.get("error") if isinstance(_result, dict) else None
                participation_telemetry.reaction(
                    channel_id, react_ts, operation="add", result=outcome,
                    origin="work_claim", emoji=config.ack_reaction_emoji,
                    target_ts=react_ts, attempt_id=attempt_id, detail=detail)
        except asyncio.TimeoutError:
            logger.debug("Work-claim reaction timed out")
            if attempt_id:
                participation_telemetry.reaction(
                    channel_id, react_ts, operation="add", result="failed",
                    origin="work_claim", emoji=config.ack_reaction_emoji,
                    target_ts=react_ts, attempt_id=attempt_id, detail="timeout")
        except Exception as e:  # noqa: BLE001
            logger.debug(f"Work-claim reaction failed: {e}")
            if attempt_id:
                participation_telemetry.reaction(
                    channel_id, react_ts, operation="add", result="failed",
                    origin="work_claim", emoji=config.ack_reaction_emoji,
                    target_ts=react_ts, attempt_id=attempt_id, detail=type(e).__name__)
        finally:
            self._claiming = False

    async def settle_ack(self, client: Any, produced_output: bool) -> None:
        """End of turn. Did we actually do the thing we claimed?

        `produced_output` False — the model chose silence, the turn errored, it got queued,
        or it started work and then backed off (the other bot answered first) — so the claim
        was not honored and the 👀 comes back off. True: it stays."""
        lease = self.ack_lease
        if lease is None:
            return
        self.ack_lease = None
        try:
            if produced_output:
                if hasattr(client, "settle_reaction_lease"):
                    client.settle_reaction_lease(lease)
            elif hasattr(client, "remove_owned_reaction"):
                removed = await client.remove_owned_reaction(lease)
                # A retracted claim was still visible in the room for the length of the turn.
                # Recorded so "we promised work and delivered nothing" is countable — it is the
                # single most annoying failure this system can produce, and it is invisible
                # afterwards because the evidence deletes itself.
                #
                # The REAL return value, not an assumption: remove_owned_reaction refuses a
                # stale lease and returns False, leaving the 👀 up. A row saying we took it
                # back when it is still sitting there would describe the opposite of the room.
                if self.ack_attempt_id:
                    participation_telemetry.reaction(
                        self.ack_channel_id, self.ack_target_ts, operation="remove",
                        result="removed" if removed else "remove_failed",
                        origin="work_claim", emoji=config.ack_reaction_emoji,
                        target_ts=self.ack_target_ts, attempt_id=self.ack_attempt_id,
                        detail="retracted")
        except Exception as e:  # noqa: BLE001
            # The 👀 is still up there and we no longer hold the lease, so the claim is stranded.
            # Recorded for the same reason the honest False above is: a lifecycle that ends with
            # an add and no removal outcome reads as a claim that was HONORED, which is the exact
            # opposite of a turn that promised work, produced none, and then failed to clean up.
            # Only on the RETRACTION path: honoring a claim removes nothing, so a failure there
            # is not a failed removal and must not be written as one.
            if self.ack_attempt_id and not produced_output:
                participation_telemetry.reaction(
                    self.ack_channel_id, self.ack_target_ts, operation="remove",
                    result="remove_failed", origin="work_claim",
                    emoji=config.ack_reaction_emoji, target_ts=self.ack_target_ts,
                    attempt_id=self.ack_attempt_id, detail=type(e).__name__)
            logger.debug(f"Ack settle failed: {e}")
