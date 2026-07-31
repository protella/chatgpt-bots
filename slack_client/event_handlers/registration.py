from __future__ import annotations

import asyncio
import functools
from typing import Any, Awaitable, Callable, Dict, List, NamedTuple, Optional, Tuple

from config import config
from logger import setup_logger
from slack_client import admission_watermark
from slack_client.event_handlers import feedback as feedback_handlers
from slack_client.normalizer import MUTATION_SUBTYPES, mutation_activity_ts

# Must go through setup_logger: handlers are attached to `slack_bot.*` loggers with
# propagate=False, so a bare getLogger(__name__) writes to NOWHERE — and the one thing this
# module logs is the failed-index-import ERROR below, which has to be seen to be acted on.
logger = setup_logger(name="slack_bot.Registration")

try:
    from slack_client.event_handlers.activity_index import (
        feed_thread_activity_index as _feed_thread_activity_index)
except Exception as e:  # noqa: BLE001
    # The index is the only way pre-boundary roots are ever discovered, so losing it to a
    # packaging or import-cycle defect must be loud rather than a quietly degraded stream.
    logger.error(
        f"thread-activity index import FAILED ({type(e).__name__}: {e}) — channel stream "
        "root discovery is disabled until this is fixed", exc_info=True)
    _feed_thread_activity_index = None  # type: ignore[assignment]

feed_thread_activity_index = _feed_thread_activity_index


class IngressDrain(NamedTuple):
    """What the barrier established — which a bare count could not say.

    `gave_up` is the half that matters to the caller: a drain that hit its deadline cancelled its
    way to silence rather than being granted it, and a callback cancelled mid-write may not have
    persisted what it observed. `survived` names anything that outlived cancellation too.
    """

    gave_up: bool = False
    survived: Tuple[str, ...] = ()


_QUIET = IngressDrain()


class _IngressTracker:
    """The barrier that decides whether Slack ingress is actually quiet.

    Shutdown's contract is that no observation can arrive after the index retry worker stops, and
    `await client.stop()` does not establish it: production stop force-marks sessions closed and
    abandons the handler's own close, so a callback Bolt had already dispatched keeps running past
    that line. Two things therefore have to be false before ingress can be called quiet — no
    callback is running, and nothing is left that could still START one.

    Three properties make this a barrier rather than a hint, and each one replaces a way the
    first version could return "quiet" while it wasn't:

    * **The zero is never latched.** The waiter loops: it re-reads the count every time it wakes,
      so callback B entering after callback A set the idle edge cannot ride out on A's signal.
    * **A zero is re-proved across a grace window.** Bolt creates each callback as its own task,
      so at the instant the count hits zero there may be a dispatch that exists but has not
      reached the wrapper. A few event-loop passes plus a short sleep give it time to appear, and
      the zero only counts if it survives them.
    * **Dispatchers are waited out first.** The socket-mode close is itself able to hand Bolt
      events (see `track_dispatcher`), so the count is not even consulted until every registered
      dispatcher has finished.

    The wait is bounded, so the deadline is a fourth case rather than a hole: everything still able
    to run — callbacks AND dispatchers — is cancelled there and awaited under a second short bound.
    Cancellation then has to re-prove the same zero the granted path does, because a dispatcher
    unwinding from `CancelledError` can hand Bolt one last event: the cancel phase LOOPS
    (cancel, recheck, grace) so a callback that entered after any snapshot is absorbed rather than
    ignored. The window in which an event could be admitted with no worker left to persist it is
    therefore gone, unless a callback resists cancellation itself; that residual is CRITICAL-logged
    by name and reported to the caller, because it is a programming error and nothing here can
    outwait it.

    An `asyncio.Event` is the wakeup edge rather than a `Condition` for one concrete reason:
    `leave()` runs in a synchronous `finally` and cannot acquire a condition's lock. The recheck
    loop is what supplies the guarantee a condition would have given.

    Single-threaded by construction: Bolt dispatches on this loop, so nothing here needs a lock.
    """

    # A zero count is believed only after this many event-loop passes and this sleep leave it
    # standing. Passes catch a task that exists but has not run; the sleep covers a dispatch still
    # inside a socket read.
    _GRACE_PASSES = 3
    _GRACE_SLEEP = 0.05

    # How long the WHOLE cancel phase gets once the deadline has passed — every cancel-recheck
    # round together, not one apiece, so absorbing a late callback cannot multiply shutdown time.
    # A callback that ignores CancelledError for this long is a defect, and shutdown must not wait
    # on one forever.
    _CANCEL_GRACE = 2.0

    def __init__(self) -> None:
        self._count = 0
        self._seq = 0
        # entry id -> (task, listener name). Keyed per ENTRY, not per task: the timeout path has
        # to be able to cancel a straggler and name it, and one task could in principle hold two
        # entries.
        self._running: Dict[int, Tuple["asyncio.Task", str]] = {}
        self._dispatchers: Dict["asyncio.Task", str] = {}
        self._wake: Optional[asyncio.Event] = None

    @property
    def in_flight(self) -> int:
        return self._count

    def enter(self, name: str = "") -> int:
        self._seq += 1
        self._count += 1
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if task is not None:
            self._running[self._seq] = (task, name or "listener")
        return self._seq

    def leave(self, entry: int = 0) -> None:
        self._running.pop(entry, None)
        self._count = max(0, self._count - 1)
        # EVERY departure wakes the waiter, not only the one that reaches zero. The waiter's own
        # callback can be part of the count (see `_foreign_in_flight`), in which case zero never
        # arrives and a wait keyed on it would burn the whole deadline. A spurious wakeup costs one
        # recheck, and the loop rechecks regardless.
        if self._wake is not None:
            self._wake.set()

    def track_dispatcher(self, task: Optional["asyncio.Task"], *, name: str) -> None:
        """Register something that can still hand Bolt an event.

        Quiescence is not a claim about the callbacks that have already started; it is a claim
        that no more can start. A socket-mode close still in progress breaks that claim, so it is
        registered here and the barrier waits it out before it looks at the callback count at all.
        """
        if task is None or task.done():
            return
        self._dispatchers[task] = name
        task.add_done_callback(self._dispatcher_done)

    def _dispatcher_done(self, task: "asyncio.Task") -> None:
        self._dispatchers.pop(task, None)
        if self._wake is not None:
            self._wake.set()

    def _foreign_in_flight(self) -> int:
        """In-flight callbacks OTHER than the one this coroutine is running inside.

        Shutdown can be reached from a Bolt callback (a settings action, a slash command), and a
        callback that waited for itself to return would hold the barrier open until the deadline
        every single time. Its own entry is excluded; everything else still counts.
        """
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        if current is None:
            return self._count
        mine = sum(1 for task, _name in self._running.values() if task is current)
        return max(0, self._count - mine)

    async def wait_quiescent(self, timeout: float) -> IngressDrain:
        """Wait until no callback is running and nothing can start one.

        Bounded, and the bound is honest in both directions: a wedged callback must not hold
        shutdown open forever, and a wait that did not get what it asked for must not report
        success. The result says which of the two happened — see `_give_up`.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout)
        if self._wake is None:
            self._wake = asyncio.Event()

        await self._await_dispatchers(deadline)

        while True:
            while self._foreign_in_flight() > 0:
                # Cleared with no await between the read and the clear, so the edge cannot be
                # lost: the count is non-zero at this instant, so any set() worth seeing is still
                # ahead of us.
                self._wake.clear()
                remaining = deadline - loop.time()
                if remaining <= 0:
                    return await self._give_up()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    return await self._give_up()
            if await self._grace_holds(deadline):
                return _QUIET
            if deadline - loop.time() <= 0:
                return await self._give_up()

    async def _await_dispatchers(self, deadline: float) -> None:
        loop = asyncio.get_running_loop()
        while self._dispatchers:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return
            await asyncio.wait(list(self._dispatchers), timeout=remaining)

    async def _grace_holds(self, deadline: float) -> bool:
        """Re-prove a zero count. False means something appeared and the wait is not over."""
        loop = asyncio.get_running_loop()
        for _ in range(self._GRACE_PASSES):
            await asyncio.sleep(0)
            if self._foreign_in_flight() or self._dispatchers:
                return False
        await asyncio.sleep(min(self._GRACE_SLEEP, max(0.0, deadline - loop.time())))
        return not self._foreign_in_flight() and not self._dispatchers

    def _live_names(self) -> Tuple[str, ...]:
        """Everything that can still run, by name — dispatchers plus foreign listener entries.

        Read from CURRENT state, never from a cancellation snapshot: the reason the cancel phase
        loops at all is that a callback can enter after any snapshot was taken.

        INVARIANT the return path depends on: this is never empty while a foreign callback is in
        flight. `enter()` can only capture a task when it is called from one, so an entry made
        outside a task has no name to give — it is reported anonymously rather than dropped,
        because "nothing survived" alongside a non-zero count is the one answer that would send
        the caller on to close the database believing it was safe.
        """
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        names: List[str] = [name for task, name in self._dispatchers.items() if not task.done()]
        callbacks = 0
        for task, name in self._running.values():
            if task is current or task.done():
                continue
            names.append(name)
            callbacks += 1
        names.extend(["an unidentified Slack listener"]
                     * max(0, self._foreign_in_flight() - callbacks))
        return tuple(names)

    def _cancel_live(self, announced: set) -> List[Tuple["asyncio.Task", str]]:
        """Cancel everything that can still act; return the tasks cancelled.

        A DISPATCHER is cancelled for the same reason a callback is, and this is the whole
        difference between a barrier and a warning: a socket close still able to hand Bolt an event
        is exactly the interval in which an observation arrives with no worker left to persist it,
        and saying so in the log left that interval open. Taking it away is the only thing that
        closes it.

        `announced` carries the tasks already named in the log, so a straggler cancelled again on a
        later round does not repeat its CRITICAL line.
        """
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        victims: List[Tuple["asyncio.Task", str]] = []
        for task, name in list(self._dispatchers.items()):
            if task.done():
                continue
            if task not in announced:
                logger.critical(
                    f"{name} could still dispatch Slack events when ingress teardown gave up; "
                    "cancelling it — an event delivered from here on has no worker to persist it")
            victims.append((task, name))
        for task, name in list(self._running.values()):
            if task is current or task.done():
                continue
            if task not in announced:
                logger.critical(
                    f"Slack listener {name!r} was still running when ingress teardown gave up; "
                    "cancelling it so its ticket resolves instead of being abandoned pending")
            victims.append((task, name))
        for task, _name in victims:
            announced.add(task)
            task.cancel()
        return victims

    async def _give_up(self) -> IngressDrain:
        """The deadline passed. Cancel everything that can still act, re-prove the zero that
        cancellation was supposed to produce, and name whatever refused to stop.

        Cancelling once and reporting the survivors of THAT snapshot was still a lie: a dispatcher
        unwinding from `CancelledError` can schedule one final callback, which entered after the
        snapshot was taken and so appeared in neither the victim list nor the verdict — the drain
        said nothing survived while a callback was mid-write. So the phase loops: cancel, then run
        the same recheck + grace window the granted path runs, and cancel again whatever the window
        turned up. It ends when a grace window holds with nothing in flight, or when the phase's own
        bound expires — and then whatever is still live is reported, current state rather than
        snapshot.

        The bound is the whole phase's, not each round's. `_CANCEL_GRACE` is generous for a
        coroutine that merely has to unwind, so anything still running after it is not slow but
        cancellation-proof — reported rather than waited on, since the caller has a database to
        close. One grace cycle is always allowed to finish, so the phase never returns without
        having actually looked.
        """
        loop = asyncio.get_running_loop()
        cancel_deadline = loop.time() + self._CANCEL_GRACE
        announced: set = set()
        while True:
            victims = self._cancel_live(announced)
            remaining = cancel_deadline - loop.time()
            if victims and remaining > 0:
                done, _pending = await asyncio.wait([task for task, _ in victims],
                                                    timeout=remaining)
                for task in done:
                    # Consumed so a straggler that died of its own exception is not ALSO an
                    # "exception was never retrieved" line at interpreter exit.
                    if not task.cancelled():
                        task.exception()
            # Cancellation is not quiescence until it is re-proved. The grace deadline is its own,
            # not the caller's: the caller's has already passed, and clamping the sleep to it would
            # skip the very window in which a dispatcher's parting callback appears.
            if await self._grace_holds(loop.time() + self._GRACE_SLEEP):
                return IngressDrain(gave_up=True)
            if loop.time() >= cancel_deadline:
                break
        survived = self._live_names()
        for name in survived:
            logger.critical(
                f"{name} survived cancellation after {self._CANCEL_GRACE}s of ingress teardown — a "
                "Slack callback that resists CancelledError is a programming error; shutdown "
                "continues without it and anything it observes from here on is lost")
        return IngressDrain(gave_up=True, survived=survived)

    def reset(self) -> None:
        """Test seam only."""
        self._count = 0
        self._running.clear()
        for task in list(self._dispatchers):
            task.remove_done_callback(self._dispatcher_done)
        self._dispatchers.clear()
        self._wake = None


ingress = _IngressTracker()


def track_ingress(handler: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Count one Bolt callback for the whole time it runs.

    `functools.wraps` is load-bearing, not tidiness: Bolt builds a listener's kwargs from
    `inspect.getfullargspec(inspect.unwrap(func)).args`, so the wrapper is only injectable because
    `__wrapped__` leads back to the real signature.
    """

    @functools.wraps(handler)
    async def _tracked(*args: Any, **kwargs: Any) -> Any:
        entry = ingress.enter(getattr(handler, "__name__", "") or "listener")
        try:
            return await handler(*args, **kwargs)
        finally:
            ingress.leave(entry)

    # Declared so a test can walk the whole Bolt registration surface and prove nothing was left
    # out. A listener nobody counts is a listener shutdown does not wait for.
    _tracked.__ingress_tracked__ = True  # type: ignore[attr-defined]
    return _tracked


async def drain_ingress_callbacks(timeout: float = 5.0) -> IngressDrain:
    """Await Slack ingress quiescence, and say whether it was granted or taken by cancellation."""
    return await ingress.wait_quiescent(timeout)


def _admit(client_self, event) -> object:
    """Advance the channel watermark and take an index-readiness ticket. SYNCHRONOUS.

    Called from the raw listeners after the own-message check and BEFORE the first await. That
    ordering is the whole contract: H is the newest ts this process has ADMITTED, so an event
    Slack has already handed us must be visible to any turn that starts after it. One await in
    front of this and a turn could pin an H that excludes a message it is about to be asked
    about.

    An edit or a deletion advances the watermark with its ACTIVITY ts, not the subject's ts: the
    activity happened now, and the ts being edited may be hours old. That value is defined once,
    in the normalizer, and read from there — computing it a second time here is what made an edit
    with no outer `event_ts` a permanent admission failure while the index placed it perfectly well
    from the nested `edited.ts`.
    """
    if not isinstance(event, dict):
        return None
    channel_id = event.get("channel") or (event.get("item") or {}).get("channel")
    if not channel_id:
        return None
    subtype = event.get("subtype")
    if subtype == "message_changed":
        subject = event.get("message")
    elif subtype == "message_deleted":
        subject = event.get("previous_message")
    else:
        subject = event
    try:
        if client_self.is_own_message(subject if isinstance(subject, dict) else event):
            # Our own posts never advance H: a turn must not be able to wait on its own reply.
            return None
    except Exception:  # noqa: BLE001
        pass
    if subtype in MUTATION_SUBTYPES:
        admitted = mutation_activity_ts(event)
    else:
        admitted = event.get("ts") or event.get("event_ts")
    admission_watermark.observe(channel_id, admitted)
    ticket = admission_watermark.issue(channel_id)
    # A ts H cannot read did not advance the watermark, so a later turn could answer without this
    # message in its window. That is not something to log at WARNING and walk past: the
    # observation fails, and the channel refuses to answer until somebody looks at it.
    if not admission_watermark.observable(channel_id, admitted):
        admission_watermark.fail_observation(
            ticket, channel_id=channel_id, ts=admitted,
            reason="Slack delivered a timestamp the shared comparator cannot read")
    return ticket


class SlackRegistrationMixin:
    def _register_handlers(self):
        """Register Slack-specific event handlers."""

        @self.app.event("app_mention")
        @track_ingress
        async def handle_app_mention(event, say, client):
            # FIRST, before any await: admit the event, take its readiness ticket, and record who
            # spoke. Slack gives no contract that a twin `message` event always arrives for a
            # mention, so both listeners feed the tail; it dedupes on (channel, ts).
            index_ticket = _admit(self, event)
            if hasattr(self, "_feed_actor_tail"):
                self._feed_actor_tail(event)
            self.log_debug(f"App mention event: channel={event.get('channel')}, ts={event.get('ts')}")
            # F52: record this genuine Slack app_mention so the edit-reply path can tell that a
            # mention-added edit is already covered by Slack's own event (editing to add a mention
            # makes Slack deliver app_mention for the same ts) and skip a duplicate synthetic turn.
            if hasattr(self, "_note_app_mention_seen"):
                self._note_app_mention_seen(event.get("channel"), event.get("ts"))
            # F51: an @mention carries ambient content (images/links/files) too. Capture BEFORE
            # dispatch so it is kept even if the reply path drops it. Best-effort infra: guarded so
            # a registration host without the message-events mixin (test harnesses) still works.
            if hasattr(self, "_ambient_ingest"):
                await self._ambient_ingest(event, client)
            # Spec §4: Slack gives no contract that a twin `message` event always arrives for a
            # mention, so the index is fed from both listeners; the upsert is idempotent.
            if feed_thread_activity_index is not None:
                await feed_thread_activity_index(self, event, ticket=index_ticket)
            else:
                admission_watermark.complete_ok(index_ticket)
            # origin_verified: this event came straight from Slack, so it attests that the
            # sender is in this conversation (see attest_message_origin).
            await self._handle_slack_message(event, client, wake_source="app_mention",
                                             origin_verified=True)

        @self.app.event("message")
        @track_ingress
        async def handle_message(event, say, client):
            # FIRST, before any await: admit the event, take its readiness ticket, and record who
            # spoke (edits/deletions/tombstones maintain the tail too; DMs are skipped).
            index_ticket = _admit(self, event)
            if hasattr(self, "_feed_actor_tail"):
                self._feed_actor_tail(event)
            # F51: ambient capture + lifecycle (edits/deletions) runs FIRST, independent of
            # channel_type and ENABLE_CHANNEL_LISTENING — memory is a distinct setting from
            # whether the bot replies. Never blocks the wake path (offer_event only enqueues).
            # Guarded for registration hosts without the message-events mixin (test harnesses).
            if hasattr(self, "_ambient_ingest"):
                await self._ambient_ingest(event, client)
            # Spec §4: the only point that sees message_changed/message_deleted/tombstone raw
            # and still precedes every listening/participation/subtype filter.
            if feed_thread_activity_index is not None:
                await feed_thread_activity_index(self, event, ticket=index_ticket)
            else:
                admission_watermark.complete_ok(index_ticket)
            channel_type = event.get("channel_type")
            if channel_type == "im":
                # DMs from anyone except ourselves (other bots allowed so bot<->bot works).
                if not self.is_own_message(event):
                    self.log_debug(f"DM message event: channel={event.get('channel')}, ts={event.get('ts')}")
                    await self._handle_slack_message(event, client, wake_source="dm",
                                                     origin_verified=True)
            elif channel_type in ("channel", "group", "mpim"):
                # Phase 5 channel listening — gated by the master switch (DEFAULT OFF). When off,
                # non-mention channel messages are ignored entirely (mentions still arrive via
                # the app_mention event above).
                if config.enable_channel_listening:
                    await self._handle_channel_message(event, client)

        @self.app.event("file_deleted")
        @track_ingress
        async def handle_file_deleted(event):
            # F51: a Slack file removed → purge summaries derived from it (best-effort).
            await self._ambient_file_deleted(event)

        # --- agent_view lifecycle (June 2026 surface, Phase G) ---
        @self.app.event("app_home_opened")
        @track_ingress
        async def handle_app_home_opened(event, client):
            # Messages tab opened: greet once per channel (tab filter + dedup inside).
            await self._handle_app_home_opened(event, client)

        @self.app.event("app_context_changed")
        @track_ingress
        async def handle_app_context_changed(event):
            await self._handle_app_context_changed(event)

        # --- Track 4: channel join behavior ---
        @self.app.event("member_joined_channel")
        @track_ingress
        async def handle_member_joined_channel(event, client):
            # Bot added to a channel → post ONE public intro (bot-only trigger + DM/MPIM exclusion
            # + idempotency + detach all live inside the handler). Best-effort, never raises.
            await self._handle_member_joined_channel(event, client)

        # --- LEGACY agent surface (deprecated by agent_view) ---
        # Keep during the transition (whichever fires, the greeting dedup makes it
        # fire once); remove one release after the manifest fully flips to agent_view.
        @self.app.event("assistant_thread_started")
        @track_ingress
        async def handle_assistant_thread_started(event, client):
            # Agent split-view opened: greet + set suggested prompts (best-effort, flag-gated).
            await self._handle_assistant_thread_started(event, client)

        @self.app.event("assistant_thread_context_changed")
        @track_ingress
        async def handle_assistant_thread_context_changed(event):
            await self._handle_assistant_thread_context_changed(event)

        @self.app.event("reaction_added")
        @track_ingress
        async def handle_reaction_added(event):
            # Phase H: passive feedback ingestion — thumbs reactions on OUR OWN
            # messages land in the response_feedback sink. Strictly recording:
            # no LLM call, no reply, never raises. Everything else is ignored
            # (acked so Bolt doesn't log every reaction as unhandled).
            await feedback_handlers.ingest_reaction(self, event)

        @self.app.event("reaction_removed")
        @track_ingress
        async def handle_reaction_removed(event):
            # Nothing to do: reactions are rendered from the channel stream's own fetch, not from
            # in-memory counts that could drift. Registered purely so Bolt acks it instead of
            # logging every removal as an unhandled request.
            return

        # Phase H: native feedback buttons (context_actions block on DM/assistant
        # responses) arrive as ordinary block_actions.
        @self.app.action(feedback_handlers.FEEDBACK_ACTION_ID)
        @track_ingress
        async def handle_response_feedback(ack, body):
            await feedback_handlers.handle_feedback_action(self, ack, body)

        # Register settings-related handlers. Every callback in there is `@track_ingress` too —
        # they write to the same database this shutdown is about to close, so leaving them out
        # would have made "Slack ingress is quiet" a claim about only some of Slack's ingress.
        self._register_settings_handlers()
