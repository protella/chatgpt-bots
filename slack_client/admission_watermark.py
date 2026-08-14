"""Admission watermark + index-readiness tickets (spec §3, §4).

Two jobs, one module, because they have to be captured in the same synchronous step.

**H** is the newest ts this process has ADMITTED for a channel. It is pinned once per turn and
never refreshed, so the stream a turn renders is a fixed window rather than a moving one: two
independent builds of the same turn must produce the same bytes, and a watermark that advanced
mid-build would break that before anything else did.

**Tickets** answer a different question: has the thread-activity index caught up to the events
inside that window? A reply that landed just before H is discovered through the index, so a turn
that reads the index before the write lands would silently miss a whole thread. Every event that
will feed the index takes a ticket synchronously; the turn waits for the tickets it can be
affected by and no others. The ticket FRONTIER is captured atomically with H — post-H events can
neither delay nor fail a turn whose stream excludes them.

Degradation is frontier-scoped for the same reason, and this is worth being explicit about
because the looser reading is tempting: a failed write ABOVE the frontier concerns an event whose
activity_ts is greater than H, so it is outside this turn's window by construction. Discovery for
everything at or below H was covered by writes that already succeeded, and a post-H mutation is
next turn's business. So `drain` fails a turn closed if and only if a failed-unrepaired ticket
sits at or below its frontier. Every turn whose frontier reaches the failure still refuses to
answer, which is the fail-closed guarantee that matters; a turn that could not have seen the
event does not pay for it.

A ts the shared comparator cannot read is not skipped either (`observable` / `fail_observation`).
It cannot be replayed into anything, so the honest ends are the only ones taken: the ticket fails,
every turn whose frontier reaches it refuses to answer, and the loss is logged CRITICAL. The
channel then stays out of service for the life of the process, deliberately — Slack timestamps are
canonical, so an unreadable one is a protocol break, and answering from a window we cannot place
the event in is the worse failure.

Restart story — ACCEPTED RESIDUAL: retained failure events live in memory. A process death while
a channel is degraded loses the pending observation (at most a pre-floor root activity hint). The
window requires a DB that is failing while Slack events keep flowing, and the degraded flag makes
that condition loud rather than quiet. A durable journal would depend on the same failing DB.
Within a live process, observations are never silently dropped: a failure retains its event, the
retry worker replays it, and anything still pending at shutdown is logged CRITICAL.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from message_processor.client_contract import HistoryFetchError
from config import config
from logger import setup_logger
from slack_client.normalizer import parse_ts, ts_max
from slack_client.utilities import is_dm_conversation

logger = setup_logger(name="slack_bot.AdmissionWatermark")

PENDING = "pending"
OK = "ok"
FAILED = "failed"
REPAIRED = "repaired"

_DEFAULT_DRAIN_TIMEOUT = 10.0
_RETRY_BASE_SECONDS = 2.0
_RETRY_MAX_SECONDS = 60.0


def drain_timeout_seconds() -> float:
    return float(getattr(config, "index_drain_timeout_seconds", _DEFAULT_DRAIN_TIMEOUT))


async def _await_cancelled(task: "asyncio.Task") -> None:
    """Swallow a cancelled task's outcome so the cancellation is genuinely awaited."""
    try:
        await task
    except asyncio.CancelledError:
        pass
    except Exception as e:  # noqa: BLE001
        logger.debug(f"index repair worker ended with {e}")


@dataclass
class Ticket:
    """One index write a turn may have to wait for."""
    channel_id: str
    seq: int
    state: str = PENDING
    event: Any = None
    retry: Optional[Callable[[], Awaitable[bool]]] = None

    @property
    def resolved(self) -> bool:
        return self.state != PENDING


@dataclass(frozen=True)
class HPin:
    """H and the ticket frontier, captured in one synchronous step."""
    h: str
    frontier: int


@dataclass
class _Channel:
    watermark: Optional[str] = None
    seq: int = 0
    tickets: Dict[int, Ticket] = field(default_factory=dict)
    degraded: bool = False
    pulse: Optional[asyncio.Event] = None

    def signal(self) -> None:
        if self.pulse is not None:
            self.pulse.set()


class AdmissionWatermark:
    """Per-channel H and ticket state. One instance per process (see the module singleton).

    Per-channel state is tiny and bounded by workspace channel count, so nothing is ever
    evicted: a frontier that disappeared would make a later drain silently vacuous.
    """

    def __init__(self) -> None:
        self._channels: Dict[str, _Channel] = {}
        self._issuance_open = True
        self._worker: Optional[asyncio.Task] = None
        self._worker_wake: Optional[asyncio.Event] = None
        # Strong refs for cancelled workers being awaited out by reset().
        self._reapers: Set[asyncio.Task] = set()

    # -- synchronous listener surface -------------------------------------------------------

    def _channel(self, channel_id: str) -> _Channel:
        state = self._channels.get(channel_id)
        if state is None:
            state = self._channels[channel_id] = _Channel()
        return state

    def observe(self, channel_id: Optional[str], ts: Optional[str]) -> Optional[str]:
        """Advance the channel's watermark to `ts` if it is newer. Returns the new value.

        Synchronous and total: called from the raw listeners after the own-message check and
        BEFORE any await, so no event can be admitted between Slack handing it to us and H
        being able to see it. DMs have no stream and are skipped. It never raises — a listener
        is not a turn — but a ts it could not use is NOT quietly forgotten either: the caller
        asks `observable()` and fails the observation (see `fail_observation`).
        """
        if not channel_id or not ts or is_dm_conversation(channel_id):
            return None
        try:
            parse_ts(ts)
        except ValueError:
            return None
        state = self._channel(channel_id)
        state.watermark = ts_max(state.watermark, str(ts))
        return state.watermark

    def observable(self, channel_id: Optional[str], ts: Optional[str]) -> bool:
        """Could H have seen this event? False only for a CHANNEL event carrying a ts the shared
        comparator cannot read (or none at all) — the one case where the watermark silently did
        not advance and a later turn could answer without the message in front of it.

        DMs, and anything outside the stream, are True: there is nothing to observe there, which
        is not the same as a failure.
        """
        if not channel_id or is_dm_conversation(channel_id):
            return True
        try:
            parse_ts(ts)
        except ValueError:
            return False
        return True

    def fail_observation(self, ticket: Optional[Ticket], *, channel_id: Optional[str],
                         ts: Optional[str], reason: str) -> None:
        """An observation this process can never apply.

        A malformed timestamp cannot be replayed into anything — re-normalizing it fails the same
        way — so there is no repair, and the two honest ends are the only ones taken: the ticket
        fails (so every turn whose frontier reaches it refuses to answer rather than rendering a
        window it cannot prove), and the loss is logged CRITICAL. Ticketless leaves only the
        CRITICAL log. Shutdown no longer produces that case — issuance closes after ingress is
        proven quiet — so it means a callback that outran the teardown deadline, which the barrier
        has already named.

        Consequence, deliberately: a channel that sees an unreadable ts stays out of service for
        the life of the process. Slack timestamps are canonical, so this is a protocol break or a
        forged event, and answering from a window we cannot place it in is the worse failure.
        """
        logger.critical(
            f"thread-activity observation permanently lost: channel={channel_id} ts={ts!r} "
            f"reason={reason}"
            + ("" if ticket is not None
               else " (no readiness ticket — the event arrived after issuance closed)"))
        if ticket is None:
            return
        self.complete_failed(ticket, event=None, retry=None)

    def current(self, channel_id: Optional[str]) -> Optional[str]:
        state = self._channels.get(channel_id or "")
        return state.watermark if state else None

    def issue(self, channel_id: Optional[str]) -> Optional[Ticket]:
        """Take a readiness ticket for an event about to feed the index.

        Returns None once issuance has closed (shutdown), so a Bolt callback that outlives the
        retry worker cannot enqueue work nothing will ever complete.
        """
        if not channel_id or is_dm_conversation(channel_id) or not self._issuance_open:
            return None
        state = self._channel(channel_id)
        state.seq += 1
        ticket = Ticket(channel_id=channel_id, seq=state.seq)
        state.tickets[ticket.seq] = ticket
        return ticket

    def pin(self, channel_id: str, trigger_admission_ts: Optional[str]) -> HPin:
        """Capture (H, frontier) atomically. H = max(watermark, trigger ts).

        Synchronous by construction: an await between reading the watermark and reading the
        frontier would let an event land in one and not the other, and the turn would then wait
        for a ticket whose message it never fetched (or worse, not wait for one it did).
        """
        state = self._channel(channel_id)
        high = ts_max(state.watermark, trigger_admission_ts)
        if not high:
            raise HistoryFetchError(
                f"no admitted timestamp for {channel_id}; H cannot be pinned")
        parse_ts(high)
        return HPin(h=str(high), frontier=state.seq)

    def complete_ok(self, ticket: Optional[Ticket]) -> None:
        """Mark an index write successful. A ticket that has already RESOLVED is left alone.

        The two listener stages hold the same ticket: `_admit` can fail it (a ts H could not read)
        and the index feed then reports on it a moment later. Overwriting a failure with success
        there deleted the fail-closed guarantee outright — the ticket left the frontier, `drain`
        found nothing failed at or below it, and the turn answered from a window that was missing
        the very event the admission step refused to vouch for.
        """
        if ticket is None or ticket.resolved:
            return
        state = self._channels.get(ticket.channel_id)
        if state is None:
            return
        ticket.state = OK
        ticket.event = None
        ticket.retry = None
        state.tickets.pop(ticket.seq, None)
        state.signal()

    def complete_failed(self, ticket: Optional[Ticket], *, event: Any = None,
                        retry: Optional[Callable[[], Awaitable[bool]]] = None) -> None:
        """Mark an index write failed, retain its event, and put the channel out of service.

        The event is retained so the retry worker can replay it; the degraded flag is what makes
        a failing DB loud instead of a stream that is quietly missing threads.

        With NO ticket there is nothing to fail and nothing that will retry: the event was admitted
        after issuance closed, which now takes a callback outliving the ingress barrier's deadline.
        That observation is permanently lost, so it is logged CRITICAL — retaining nothing and
        saying nothing is the silent drop this whole mechanism exists to prevent.
        """
        if ticket is None:
            channel_id = getattr(event, "channel_id", None)
            logger.critical(
                "thread-activity index write failed for an event with no readiness ticket "
                f"(issuance already closed): channel={channel_id} "
                f"ts={getattr(event, 'subject_ts', None)!r} — observation permanently lost")
            return
        state = self._channels.get(ticket.channel_id)
        if state is None:
            return
        ticket.state = FAILED
        ticket.event = event
        ticket.retry = retry
        state.tickets[ticket.seq] = ticket
        state.degraded = True
        state.signal()
        logger.error(
            f"thread-activity index write failed for {ticket.channel_id} "
            f"(ticket {ticket.seq}); channel marked degraded until it is repaired")
        self._ensure_worker()

    def is_degraded(self, channel_id: Optional[str]) -> bool:
        """Channel-wide health, for logging and telemetry. NOT a turn gate — `drain` decides
        that, per frontier, because a failure above a turn's frontier is not its failure."""
        state = self._channels.get(channel_id or "")
        return bool(state and state.degraded)

    def pending_failures(self, channel_id: Optional[str]) -> List[Ticket]:
        state = self._channels.get(channel_id or "")
        if not state:
            return []
        return [t for t in state.tickets.values() if t.state == FAILED]

    def ticket_state(self, ticket: Optional[Ticket]) -> Optional[str]:
        return None if ticket is None else ticket.state

    # -- turn surface ----------------------------------------------------------------------

    async def drain(self, channel_id: str, frontier: int,
                    timeout: Optional[float] = None) -> None:
        """Wait until every ticket at or below `frontier` has completed successfully.

        Raises HistoryFetchError when one of them failed and has not been repaired, or on
        timeout. Tickets above the frontier are ignored ENTIRELY — pending or failed — because
        their events are outside this turn's window by construction (see the module docstring).
        """
        limit = drain_timeout_seconds() if timeout is None else float(timeout)
        state = self._channels.get(channel_id)
        if state is None:
            return
        if state.pulse is None:
            state.pulse = asyncio.Event()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, limit)
        while True:
            # Cleared BEFORE the checks: a completion that lands between them and the wait sets
            # the event again, so the wait returns immediately instead of sitting out the
            # timeout on a signal it already missed.
            state.pulse.clear()
            failed = [t.seq for t in state.tickets.values()
                      if t.seq <= frontier and t.state == FAILED]
            if failed:
                raise HistoryFetchError(
                    f"thread-activity index writes failed for {channel_id} "
                    f"(tickets {sorted(failed)}); the stream would be missing threads")
            pending = [t for t in state.tickets.values()
                       if t.seq <= frontier and t.state == PENDING]
            if not pending:
                return
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise HistoryFetchError(
                    f"thread-activity index did not settle for {channel_id} within "
                    f"{limit:g}s ({len(pending)} write(s) outstanding)")
            try:
                await asyncio.wait_for(state.pulse.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                continue

    # -- retry worker ----------------------------------------------------------------------

    def _ensure_worker(self) -> None:
        if self._worker is not None and not self._worker.done():
            if self._worker_wake is not None:
                self._worker_wake.set()
            return
        if not self._issuance_open:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._worker_wake = asyncio.Event()
        self._worker_wake.set()
        # Strong reference: an untracked task can be garbage-collected mid-repair, which would
        # leave a channel degraded forever with nothing trying to fix it.
        self._worker = loop.create_task(self._repair_loop(), name="index-repair-worker")
        self._worker.add_done_callback(self._log_worker_exit)

    @staticmethod
    def _log_worker_exit(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.error(f"index repair worker exited: {exc}")

    async def _repair_loop(self) -> None:
        delay = _RETRY_BASE_SECONDS
        while True:
            if self._worker_wake is not None:
                self._worker_wake.clear()
            repaired_any, remaining = await self._repair_pass()
            if not remaining:
                if self._worker_wake is None:
                    return
                await self._worker_wake.wait()
                delay = _RETRY_BASE_SECONDS
                continue
            delay = _RETRY_BASE_SECONDS if repaired_any else min(delay * 2, _RETRY_MAX_SECONDS)
            await asyncio.sleep(delay)

    async def _repair_pass(self) -> Tuple[bool, int]:
        repaired_any = False
        remaining = 0
        for channel_id, state in list(self._channels.items()):
            for ticket in [t for t in state.tickets.values() if t.state == FAILED]:
                if ticket.retry is None:
                    # Nothing to replay: the failure is unrepairable in this process, so it
                    # stays visible rather than clearing the flag on a lie.
                    remaining += 1
                    continue
                try:
                    ok = bool(await ticket.retry())
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        f"index repair for {channel_id} ticket {ticket.seq} failed again: {e}")
                    ok = False
                if ok:
                    ticket.state = REPAIRED
                    ticket.event = None
                    ticket.retry = None
                    repaired_any = True
                else:
                    remaining += 1
            self._settle_channel(state)
        return repaired_any, remaining

    def _settle_channel(self, state: _Channel) -> None:
        """Clear the degraded flag only when every retained failure is repaired, then prune the
        repaired tickets — once nothing is failing they are indistinguishable from successes."""
        if any(t.state == FAILED for t in state.tickets.values()):
            return
        if state.degraded:
            state.degraded = False
            logger.info("thread-activity index recovered; channel is back in service")
        for seq in [s for s, t in state.tickets.items() if t.state == REPAIRED]:
            state.tickets.pop(seq, None)
        state.signal()

    # -- lifecycle -------------------------------------------------------------------------

    def close_issuance(self) -> None:
        """Stop handing out tickets.

        Called at ONE point in shutdown: after Slack ingress is quiet and before the retry worker
        is drained. Both halves are load-bearing. Closing it any earlier opens an interval in which
        an admitted event gets no ticket, and a ticketless failure can only be logged, never
        repaired (`complete_failed`) — with the barrier in front of it, the only way in is a
        callback that survived cancellation at the barrier's deadline, which is CRITICAL-logged by
        name. Closing it any later would let a callback enqueue a repair behind the drain.
        """
        self._issuance_open = False

    async def shutdown(self, timeout: Optional[float] = None) -> None:
        """Final drain, then stop the worker. The DB must still be open when this runs.

        Ordering is the contract: issuance closes first, this runs second, DB teardown third.
        """
        self.close_issuance()
        limit = drain_timeout_seconds() if timeout is None else float(timeout)
        # ONE deadline across every channel. Per-channel timeouts multiply: a workspace with two
        # hundred channels and a hung DB would hold shutdown for 200 × the limit.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, limit)
        for channel_id, state in list(self._channels.items()):
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                await self.drain(channel_id, state.seq, timeout=remaining)
            except HistoryFetchError:
                pass
        worker, self._worker = self._worker, None
        if worker is not None:
            if not worker.done():
                worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
        residual = 0
        for channel_id, state in self._channels.items():
            for ticket in state.tickets.values():
                if ticket.state in (PENDING, FAILED):
                    residual += 1
                    subject = getattr(ticket.event, "subject_ts", None)
                    logger.critical(
                        f"thread-activity index observation lost at shutdown: "
                        f"channel={channel_id} ts={subject} ticket={ticket.seq} "
                        f"state={ticket.state}")
        if residual:
            logger.critical(
                f"{residual} thread-activity index observation(s) were never persisted; "
                "affected channels will rediscover them only on the next coverage sweep")

    def reset(self) -> None:
        """Test seam: forget every channel's state and reopen issuance.

        Cancels the retry worker and hands it to a reaper task so the cancellation is actually
        awaited — a bare `.cancel()` leaves the task pending past the end of the test that
        cancelled it. `await shutdown()` is still the right call when the caller can await.
        """
        self._channels.clear()
        self._issuance_open = True
        worker, self._worker = self._worker, None
        self._worker_wake = None
        if worker is None or worker.done():
            return
        worker.cancel()
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        reaper: asyncio.Task = loop.create_task(_await_cancelled(worker))
        self._reapers.add(reaper)
        reaper.add_done_callback(self._reapers.discard)


# The module-level singleton every listener and every turn reads.
watermark = AdmissionWatermark()


def observe(channel_id: Optional[str], ts: Optional[str]) -> Optional[str]:
    return watermark.observe(channel_id, ts)


def observable(channel_id: Optional[str], ts: Optional[str]) -> bool:
    return watermark.observable(channel_id, ts)


def fail_observation(ticket: Optional[Ticket], *, channel_id: Optional[str],
                     ts: Optional[str], reason: str) -> None:
    watermark.fail_observation(ticket, channel_id=channel_id, ts=ts, reason=reason)


def issue(channel_id: Optional[str]) -> Optional[Ticket]:
    return watermark.issue(channel_id)


def complete_ok(ticket: Optional[Ticket]) -> None:
    watermark.complete_ok(ticket)


def complete_failed(ticket: Optional[Ticket], *, event: Any = None,
                    retry: Optional[Callable[[], Awaitable[bool]]] = None) -> None:
    watermark.complete_failed(ticket, event=event, retry=retry)


def current(channel_id: Optional[str]) -> Optional[str]:
    return watermark.current(channel_id)


def pin(channel_id: str, trigger_admission_ts: Optional[str]) -> HPin:
    return watermark.pin(channel_id, trigger_admission_ts)


async def drain(channel_id: str, frontier: int, timeout: Optional[float] = None) -> None:
    await watermark.drain(channel_id, frontier, timeout=timeout)


def is_degraded(channel_id: Optional[str]) -> bool:
    return watermark.is_degraded(channel_id)


def close_issuance() -> None:
    watermark.close_issuance()


async def shutdown(timeout: Optional[float] = None) -> None:
    await watermark.shutdown(timeout=timeout)
