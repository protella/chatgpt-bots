"""Admission watermark H + index-readiness tickets (spec §3, §4).

H is pinned once per turn with the ticket frontier captured in the SAME synchronous step, so a
turn's window and the set of index writes it waits for can never disagree. The tests below hold
that line from both ends: nothing post-frontier may delay or fail a turn, and nothing at or below
it may be skipped.
"""
import asyncio
import logging

import pytest

from base_client import HistoryFetchError
from slack_client import admission_watermark
from slack_client.admission_watermark import (FAILED, OK, PENDING, REPAIRED,
                                              AdmissionWatermark)
from slack_client.normalizer import normalize_slack_event

TEAM = "T1"
CH = "C1"
CH2 = "C2"
TS = "100.000100"


@pytest.fixture
async def wm():
    instance = AdmissionWatermark()
    yield instance
    await instance.shutdown(timeout=0.01)
    instance.reset()


class _Client:
    self_team_id = TEAM
    bot_user_id = "UBOT"
    bot_id = "BBOT"
    app_id = "A123"

    def is_own_message(self, message):
        return False


def _retained_event(ts=TS, channel=CH):
    """A real NormalizedEvent, so the shutdown log's `subject_ts` is the field it claims."""
    event = normalize_slack_event(_Client(), {
        "type": "message", "channel": channel, "channel_type": "channel", "user": "U1",
        "ts": ts, "thread_ts": "99.000100", "text": "a reply", "event_ts": ts})
    assert event is not None and event.subject_ts == ts
    return event


async def _yield_until(predicate, tries=500):
    for _ in range(tries):
        if predicate():
            return True
        await asyncio.sleep(0)
    return False


async def _pending(coro):
    task = asyncio.ensure_future(coro)
    for _ in range(5):
        await asyncio.sleep(0)
    return task


# ------------------------------------------------------------------ observe / current

async def test_observe_keeps_the_numeric_max_per_channel(wm):
    assert wm.observe(CH, "100.000100") == "100.000100"
    assert wm.observe(CH, "200.000100") == "200.000100"
    assert wm.observe(CH2, "50.000100") == "50.000100"
    assert wm.current(CH) == "200.000100"
    assert wm.current(CH2) == "50.000100"


async def test_out_of_order_observation_never_walks_the_watermark_backward(wm):
    wm.observe(CH, "1000.000200")
    assert wm.observe(CH, "999.999900") == "1000.000200"
    assert wm.current(CH) == "1000.000200"


async def test_unequal_width_timestamps_compare_numerically(wm):
    """String comparison would keep "100.10": 0.5 is the later message, not the longer one."""
    wm.observe(CH, "100.10")
    assert wm.observe(CH, "100.5") == "100.5"
    assert wm.current(CH) == "100.5"


async def test_a_wider_older_fraction_still_loses(wm):
    wm.observe(CH, "100.5")
    assert wm.observe(CH, "100.10") == "100.5"


@pytest.mark.parametrize("channel", ["D123", "U123", "W123"])
async def test_dm_surfaces_are_never_watermarked(wm, channel):
    assert wm.observe(channel, TS) is None
    assert wm.current(channel) is None


async def test_an_unparseable_ts_is_ignored_and_never_raises(wm):
    wm.observe(CH, TS)
    assert wm.observe(CH, "not-a-ts") is None
    assert wm.current(CH) == TS
    assert wm.observe(CH, "") is None
    assert wm.observe(None, TS) is None


async def test_current_for_an_unknown_channel_is_none(wm):
    assert wm.current("C-never-seen") is None
    assert wm.current(None) is None


# ------------------------------------------------------------------ pin

async def test_pin_takes_the_trigger_ts_when_it_is_newer(wm):
    wm.observe(CH, "100.000100")
    assert wm.pin(CH, "300.000100").h == "300.000100"


async def test_pin_takes_the_watermark_when_it_is_newer(wm):
    wm.observe(CH, "300.000100")
    assert wm.pin(CH, "100.000100").h == "300.000100"


async def test_pin_frontier_is_the_last_issued_ticket(wm):
    wm.observe(CH, TS)
    wm.issue(CH)
    wm.issue(CH)
    wm.issue(CH2)
    assert wm.pin(CH, None).frontier == 2


async def test_pin_with_nothing_admitted_raises(wm):
    with pytest.raises(HistoryFetchError):
        wm.pin(CH, None)


async def test_pin_is_atomic_against_later_issuance(wm):
    wm.observe(CH, TS)
    wm.issue(CH)
    pinned = wm.pin(CH, None)
    later = wm.issue(CH)
    assert pinned.frontier == 1
    assert later.seq > pinned.frontier


# ------------------------------------------------------------------ issue

async def test_tickets_are_sequential_and_per_channel(wm):
    assert [wm.issue(CH).seq for _ in range(3)] == [1, 2, 3]
    assert wm.issue(CH2).seq == 1
    assert wm.issue(CH).seq == 4


async def test_a_dm_never_gets_a_ticket(wm):
    assert wm.issue("D123") is None
    assert wm.issue(None) is None


# ------------------------------------------------------------------ drain

async def test_drain_returns_immediately_when_every_ticket_completed(wm):
    tickets = [wm.issue(CH) for _ in range(3)]
    for ticket in tickets:
        wm.complete_ok(ticket)
    assert all(wm.ticket_state(t) == OK for t in tickets)
    await asyncio.wait_for(wm.drain(CH, 3, timeout=0.5), timeout=1.0)


async def test_drain_waits_for_a_pending_ticket_then_returns(wm):
    ticket = wm.issue(CH)
    task = await _pending(wm.drain(CH, 1, timeout=5.0))
    assert not task.done()
    wm.complete_ok(ticket)
    await asyncio.wait_for(task, timeout=1.0)


async def test_drain_ignores_tickets_above_the_frontier(wm):
    first = wm.issue(CH)
    wm.complete_ok(first)
    second = wm.issue(CH)
    await asyncio.wait_for(wm.drain(CH, 1, timeout=0.5), timeout=1.0)

    task = await _pending(wm.drain(CH, second.seq, timeout=5.0))
    assert not task.done()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_drain_on_an_unknown_channel_is_vacuous(wm):
    await asyncio.wait_for(wm.drain("C-never-seen", 5, timeout=0.01), timeout=1.0)


async def test_a_failed_ticket_fails_the_drain_and_degrades_the_channel(wm):
    ticket = wm.issue(CH)
    wm.complete_failed(ticket, event=_retained_event())
    with pytest.raises(HistoryFetchError):
        await wm.drain(CH, ticket.seq, timeout=0.5)
    assert wm.ticket_state(ticket) == FAILED
    assert wm.is_degraded(CH) is True
    assert ticket in wm.pending_failures(CH)


async def test_degradation_is_frontier_scoped(wm):
    """A write that failed ABOVE a turn's frontier concerns an event with activity_ts > H, so it
    is outside that turn's window by construction. Every turn whose frontier reaches the failure
    still refuses; a turn that could not have seen the event does not pay for it."""
    covered = wm.issue(CH)
    wm.complete_ok(covered)
    above = wm.issue(CH)
    wm.complete_failed(above, event=_retained_event())

    await asyncio.wait_for(wm.drain(CH, covered.seq, timeout=0.5), timeout=1.0)
    assert wm.is_degraded(CH) is True
    with pytest.raises(HistoryFetchError):
        await wm.drain(CH, above.seq, timeout=0.5)


async def test_a_pending_ticket_above_the_frontier_never_delays_a_drain(wm):
    covered = wm.issue(CH)
    wm.complete_ok(covered)
    wm.issue(CH)
    await asyncio.wait_for(wm.drain(CH, covered.seq, timeout=0.5), timeout=1.0)


async def test_an_unsettled_ticket_times_the_drain_out(wm):
    ticket = wm.issue(CH)
    with pytest.raises(HistoryFetchError, match="did not settle"):
        await wm.drain(CH, ticket.seq, timeout=0.01)


# ------------------------------------------------------------------ repair worker

async def test_the_worker_repairs_a_retained_failure_and_restores_service(wm, monkeypatch):
    monkeypatch.setattr(admission_watermark, "_RETRY_BASE_SECONDS", 0)
    monkeypatch.setattr(admission_watermark, "_RETRY_MAX_SECONDS", 0)
    attempts = []

    async def _retry():
        attempts.append(1)
        return len(attempts) > 1

    ticket = wm.issue(CH)
    wm.complete_failed(ticket, event=_retained_event(), retry=_retry)
    assert wm.is_degraded(CH) is True

    assert await _yield_until(lambda: not wm.is_degraded(CH))
    assert len(attempts) == 2
    assert wm.ticket_state(ticket) == REPAIRED
    assert ticket.event is None and ticket.retry is None
    assert wm.pending_failures(CH) == []
    assert ticket.seq not in wm._channels[CH].tickets
    await asyncio.wait_for(wm.drain(CH, ticket.seq, timeout=0.5), timeout=1.0)


async def test_a_failure_with_no_retry_keeps_the_channel_degraded(wm, monkeypatch):
    monkeypatch.setattr(admission_watermark, "_RETRY_BASE_SECONDS", 0)
    monkeypatch.setattr(admission_watermark, "_RETRY_MAX_SECONDS", 0)
    ticket = wm.issue(CH)
    wm.complete_failed(ticket, event=_retained_event(), retry=None)

    assert not await _yield_until(lambda: not wm.is_degraded(CH), tries=50)
    assert wm.pending_failures(CH) == [ticket]
    with pytest.raises(HistoryFetchError):
        await wm.drain(CH, ticket.seq, timeout=0.01)


async def test_a_repaired_ticket_is_history_and_never_fails_a_later_drain(wm, monkeypatch):
    """drain consults CURRENT state: a ticket that failed and was repaired is a success, and a
    later window that includes it must not be refused for something already fixed."""
    monkeypatch.setattr(admission_watermark, "_RETRY_BASE_SECONDS", 0)
    monkeypatch.setattr(admission_watermark, "_RETRY_MAX_SECONDS", 0)

    async def _retry():
        return True

    first = wm.issue(CH)
    wm.complete_failed(first, event=_retained_event(), retry=_retry)
    assert await _yield_until(lambda: not wm.is_degraded(CH))

    second = wm.issue(CH)
    wm.complete_ok(second)
    await asyncio.wait_for(wm.drain(CH, second.seq, timeout=0.5), timeout=1.0)


# ------------------------------------------------------------------ no eviction

async def test_no_channel_state_is_ever_evicted(wm):
    channels = [f"C{i:04d}" for i in range(200)]
    for index, channel in enumerate(channels):
        wm.observe(channel, f"{1000 + index}.000100")
        wm.issue(channel)
    for index, channel in enumerate(channels):
        assert wm.current(channel) == f"{1000 + index}.000100"
        assert wm.pin(channel, None).frontier == 1


# ------------------------------------------------------------------ shutdown race

async def test_issuance_closes_and_null_tickets_are_no_ops(wm):
    wm.close_issuance()
    assert wm.issue(CH) is None
    wm.complete_ok(None)
    wm.complete_failed(None)
    wm.complete_failed(None, event=_retained_event(), retry=None)
    assert wm.current(CH) is None


async def test_a_ticket_held_when_issuance_closed_is_still_drained_by_shutdown(
        wm, monkeypatch, caplog):
    monkeypatch.setattr(admission_watermark.logger, "propagate", True)
    ticket = wm.issue(CH)
    wm.close_issuance()
    assert wm.issue(CH) is None
    wm.complete_ok(ticket)

    with caplog.at_level(logging.CRITICAL, logger=admission_watermark.logger.name):
        await wm.shutdown(timeout=0.01)
    assert caplog.records == []
    assert wm._channels[CH].tickets == {}


async def test_shutdown_logs_critical_for_a_residual_failure(wm, monkeypatch, caplog):
    monkeypatch.setattr(admission_watermark.logger, "propagate", True)
    monkeypatch.setattr(admission_watermark, "_RETRY_BASE_SECONDS", 0)
    monkeypatch.setattr(admission_watermark, "_RETRY_MAX_SECONDS", 0)
    retained = _retained_event(ts="777.000100")
    ticket = wm.issue(CH)
    wm.complete_failed(ticket, event=retained, retry=None)
    worker = wm._worker
    assert worker is not None

    with caplog.at_level(logging.CRITICAL, logger=admission_watermark.logger.name):
        await wm.shutdown(timeout=0.01)

    text = "\n".join(r.getMessage() for r in caplog.records
                     if r.levelno >= logging.CRITICAL)
    assert CH in text
    assert retained.subject_ts in text
    assert "never persisted" in text
    assert worker.done()
    assert wm._worker is None


async def test_shutdown_reports_a_residual_pending_ticket(wm, monkeypatch, caplog):
    monkeypatch.setattr(admission_watermark.logger, "propagate", True)
    wm.issue(CH)
    with caplog.at_level(logging.CRITICAL, logger=admission_watermark.logger.name):
        await wm.shutdown(timeout=0.01)
    assert any(PENDING in r.getMessage() for r in caplog.records
               if r.levelno >= logging.CRITICAL)


# ------------------------------------------- unobservable timestamps (codex P2 review, F7e)

def test_a_channel_ts_the_comparator_cannot_read_is_not_observable(wm):
    assert wm.observable(CH, TS) is True
    assert wm.observable(CH, "not-a-timestamp") is False
    assert wm.observable(CH, None) is False
    # A DM has no stream to be missing from, so there is nothing to fail.
    assert wm.observable("D123", "not-a-timestamp") is True
    assert wm.observable(None, "not-a-timestamp") is True


def test_an_unparseable_ts_leaves_the_watermark_alone(wm):
    wm.observe(CH, TS)
    assert wm.observe(CH, "not-a-timestamp") is None
    assert wm.current(CH) == TS


async def test_a_failed_observation_fails_its_ticket_and_logs_critical(wm, monkeypatch, caplog):
    """The F7e hole: the watermark ignored the ts with a WARNING while the ticket completed OK, so
    every turn in that window was told the index was caught up on an event we could not place."""
    monkeypatch.setattr(admission_watermark.logger, "propagate", True)
    ticket = wm.issue(CH)
    with caplog.at_level(logging.CRITICAL, logger=admission_watermark.logger.name):
        wm.fail_observation(ticket, channel_id=CH, ts="nonsense",
                            reason="unparseable timestamp on an indexable event")

    assert ticket.state == FAILED
    assert wm.is_degraded(CH) is True
    with pytest.raises(HistoryFetchError):
        await wm.drain(CH, ticket.seq, timeout=0.01)
    critical = "\n".join(r.getMessage() for r in caplog.records
                         if r.levelno >= logging.CRITICAL)
    assert "nonsense" in critical and CH in critical
    # Nothing retained and nothing to replay: the failure is permanent by construction, so the
    # repair worker must not clear it on a lie.
    assert ticket.retry is None and ticket.event is None


def test_a_failed_observation_with_no_ticket_still_logs_critical(wm, monkeypatch, caplog):
    """The shutdown residual: issuance has closed, so there is no ticket to fail. The CRITICAL
    line is then the entire record, which is the accepted-residual contract — never silence."""
    monkeypatch.setattr(admission_watermark.logger, "propagate", True)
    wm.close_issuance()
    assert wm.issue(CH) is None
    with caplog.at_level(logging.CRITICAL, logger=admission_watermark.logger.name):
        wm.fail_observation(None, channel_id=CH, ts=TS, reason="closed issuance")
    critical = [r.getMessage() for r in caplog.records if r.levelno >= logging.CRITICAL]
    assert critical and "no readiness ticket" in critical[0]


async def test_a_later_success_cannot_undo_a_failed_observation(wm):
    """Both listener stages hold the same ticket: `_admit` can fail it and the index feed reports
    on it a moment later. Overwriting the failure with success deleted the guarantee outright — the
    ticket left the frontier and the turn answered from a window nothing had vouched for."""
    ticket = wm.issue(CH)
    wm.fail_observation(ticket, channel_id=CH, ts="nonsense", reason="unreadable ts")

    wm.complete_ok(ticket)

    assert ticket.state == FAILED
    assert wm.is_degraded(CH) is True
    with pytest.raises(HistoryFetchError):
        await wm.drain(CH, ticket.seq, timeout=0.01)


async def test_complete_ok_is_idempotent_for_a_pending_ticket(wm):
    ticket = wm.issue(CH)
    wm.complete_ok(ticket)
    wm.complete_ok(ticket)
    assert ticket.state == OK
    await wm.drain(CH, ticket.seq, timeout=0.01)


def test_a_ticketless_index_failure_is_logged_critical_not_dropped(wm, monkeypatch, caplog):
    """`complete_failed(None)` retained nothing and said nothing — the silent drop codex found in
    the interval between issuance closure and Slack ingress stopping."""
    monkeypatch.setattr(admission_watermark.logger, "propagate", True)
    retained = _retained_event(ts="888.000100")
    with caplog.at_level(logging.CRITICAL, logger=admission_watermark.logger.name):
        wm.complete_failed(None, event=retained, retry=None)
    critical = "\n".join(r.getMessage() for r in caplog.records
                         if r.levelno >= logging.CRITICAL)
    assert "888.000100" in critical and CH in critical
    assert "permanently lost" in critical


# ------------------------------------------------------------------ module facade

async def test_the_module_facade_delegates_to_the_singleton():
    # The one test that uses the process singleton rather than a fresh instance, so it starts by
    # forgetting whatever an earlier file left on it — a suite that ran main.py's shutdown first
    # would otherwise hand this a watermark with issuance already closed.
    admission_watermark.watermark.reset()
    try:
        assert admission_watermark.observe(CH, TS) == TS
        assert admission_watermark.current(CH) == TS
        assert admission_watermark.observe("D123", TS) is None

        first = admission_watermark.issue(CH)
        assert first.seq == 1
        assert first is admission_watermark.watermark._channels[CH].tickets[1]

        pinned = admission_watermark.pin(CH, "500.000100")
        assert (pinned.h, pinned.frontier) == ("500.000100", 1)

        admission_watermark.complete_ok(first)
        assert first.state == OK
        await admission_watermark.drain(CH, 1, timeout=0.01)

        second = admission_watermark.issue(CH)
        admission_watermark.complete_failed(second, event=_retained_event(), retry=None)
        assert admission_watermark.is_degraded(CH) is True
        with pytest.raises(HistoryFetchError):
            await admission_watermark.drain(CH, second.seq, timeout=0.01)

        admission_watermark.close_issuance()
        assert admission_watermark.issue(CH) is None
        await admission_watermark.shutdown(timeout=0.01)
        assert admission_watermark.watermark._worker is None
    finally:
        admission_watermark.watermark.reset()


# ---------------------------------- r3-5: the admission step's own timestamp resolution
#
# `_admit` runs before the first await in every raw listener, and it is the only thing that can
# refuse an observation for a ts H could not read. That refusal is permanent and unrepairable, so
# it must not fire for an event the index goes on to place perfectly well.


@pytest.fixture
def singleton():
    """`_admit` reads the module singleton, which is what the live listeners read."""
    admission_watermark.watermark.reset()
    yield admission_watermark.watermark
    admission_watermark.watermark.reset()


def _changed_event(message, **extra):
    event = {"type": "message", "subtype": "message_changed", "channel": CH,
             "channel_type": "channel", "message": message}
    event.update(extra)
    return event


def test_admit_falls_back_to_the_nested_edited_ts(singleton):
    """The blocker: an edit with no outer `event_ts`. Admission used to read that field and nothing
    else, so H never advanced, `observable` said no, and the channel went out of service for the
    life of the process — over an activity time that was sitting in `edited.ts`, where the index
    read it from without any trouble."""
    from slack_client.event_handlers.registration import _admit

    ticket = _admit(_Client(), _changed_event(
        {"type": "message", "user": "U1", "ts": "100.000100", "text": "new",
         "edited": {"user": "U1", "ts": "888.000100"}}))

    assert singleton.current(CH) == "888.000100"
    assert ticket is not None and ticket.state == PENDING
    assert singleton.is_degraded(CH) is False


def test_admit_prefers_the_outer_event_ts_when_slack_sends_one(singleton):
    """The fallback is a fallback: the envelope's own time is when the mutation happened, and
    `edited.ts` can be older than it."""
    from slack_client.event_handlers.registration import _admit

    _admit(_Client(), _changed_event(
        {"type": "message", "user": "U1", "ts": "100.000100", "text": "new",
         "edited": {"user": "U1", "ts": "888.000100"}}, event_ts="900.000900"))

    assert singleton.current(CH) == "900.000900"


def test_admit_still_refuses_an_edit_that_names_no_activity_time_at_all(singleton, monkeypatch,
                                                                       caplog):
    """The honest end is unchanged for the case that really has nothing to read: the observation
    fails, the channel refuses to answer, and the loss is CRITICAL rather than silent."""
    from slack_client.event_handlers.registration import _admit

    monkeypatch.setattr(admission_watermark.logger, "propagate", True)
    with caplog.at_level(logging.CRITICAL, logger=admission_watermark.logger.name):
        ticket = _admit(_Client(), _changed_event(
            {"type": "message", "user": "U1", "ts": "100.000100", "text": "new"}))

    assert ticket is not None and ticket.state == FAILED
    assert singleton.is_degraded(CH) is True
    critical = "\n".join(r.getMessage() for r in caplog.records
                         if r.levelno >= logging.CRITICAL)
    assert "permanently lost" in critical
