"""Wiring layer for outbound receipts (single-stream P1 §B1, spec §5).

The DB state machine is covered by test_receipts_db.py. What is tested here is the layer that
DECIDES what to write: which owner claims a message, when a chrome surface becomes conversation,
what happens to the row when the message is deleted, and — the failure the whole feature turns
on — that a turn which crashes, is cancelled, or loses its database still leaves the room's
words accounted for rather than silently excluded from the stream.
"""
import asyncio
import io
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import config
from database import DatabaseManager, OUTBOUND_RECEIPTS_EPOCH_KEY
from message_processor import outbound_receipts as orx
from message_processor.outbound_receipts import (ReceiptLedger, ReceiptService, _Op,
                                                 record_transport_post)
from message_processor.stale_send_guard import StaleSendSuppressed
from message_processor.turn_runtime import TurnRuntime

TEAM = "T1"
CH = "C0BKX77NU66"
DM = "D0000001"


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    db = DatabaseManager(platform="slack")
    yield db
    db.conn.close()


@pytest.fixture
def service(temp_db):
    orx.reset_service()
    svc = orx.install_service(temp_db)
    yield svc
    orx.reset_service()


@pytest.fixture
def events(monkeypatch):
    """Every `outbound_receipt` line the code under test produced, in order.

    Captured at the telemetry entry point rather than by reading the file: the sink writes on a
    listener thread, and what these tests are about is which transitions get a line and what it
    says, not the sink's own plumbing.
    """
    from message_processor import participation_telemetry

    captured = []
    monkeypatch.setattr(participation_telemetry, "outbound_receipt",
                        lambda **kwargs: captured.append(kwargs))
    return captured


@pytest.fixture
def barriers(monkeypatch):
    """Every `post_partial_post` seam entry, with its context."""
    from message_processor import dev_barriers

    fired = []

    async def _record(**context):
        fired.append(context)
        return True

    monkeypatch.setattr(dev_barriers, "post_partial_post", _record)
    return fired


async def _state(db, ts, channel=CH):
    row = await db.get_receipt_async(TEAM, channel, ts)
    return row["state"] if row else None


def _ledger(owner="s:1", channel=CH, barrier_eligible=False):
    return ReceiptLedger(owner, TEAM, channel, barrier_eligible=barrier_eligible)


# --------------------------------------------------------------- identity


def test_the_transition_log_reaches_the_configured_hierarchy():
    """Live battery F4: these lines went to `getLogger(__name__)`, which is outside `slack_bot.*`
    and therefore on the unconfigured root logger — so in a normal run the spec's per-transition
    record appeared NOWHERE, at any level, and the only evidence a receipt had moved was the
    DatabaseManager mirror one layer down."""
    assert orx.logger.name == "slack_bot.OutboundReceipts"
    assert orx.logger.handlers, "no handler attached — the transition lines go nowhere"


def test_session_id_is_shared_with_the_participation_ledger():
    import runtime_identity
    from message_processor import participation_telemetry

    assert participation_telemetry.SESSION_ID == runtime_identity.SESSION_ID


def test_every_turn_gets_a_distinct_session_scoped_turn_id():
    import runtime_identity

    first, second = TurnRuntime(), TurnRuntime()
    assert first.turn_id != second.turn_id
    assert first.turn_id.startswith(f"{runtime_identity.SESSION_ID}:")


def test_bind_receipts_opens_a_ledger_for_a_channel_and_none_for_a_dm():
    class _Client:
        self_team_id = TEAM

    class _Msg:
        def __init__(self, cid):
            self.channel_id = cid

    turn = TurnRuntime()
    assert turn.bind_receipts(_Client(), _Msg(CH)) is not None
    assert turn.receipt_ledger.owner_id == turn.turn_id
    assert TurnRuntime().bind_receipts(_Client(), _Msg(DM)) is None


# --------------------------------------------------------------- surface ruling


async def test_dm_and_user_targets_write_nothing(service, temp_db):
    for target in (DM, "U12345", "W12345"):
        ledger = _ledger(channel=target)
        assert not ledger.active
        await ledger.note_post("100.0", receipt_class="assistant_reply")
        await ledger.settle()
        assert await temp_db.get_receipt_async(TEAM, target, "100.0") is None


async def test_an_unknown_team_is_not_guessed_at(service, temp_db):
    ledger = ReceiptLedger("s:1", None, CH)
    assert not ledger.active
    await ledger.note_post("100.0", receipt_class="assistant_reply")
    assert await temp_db.get_receipt_async(TEAM, CH, "100.0") is None


# --------------------------------------------------------------- the turn lifecycle


async def test_post_then_settle_finalizes(service, temp_db):
    ledger = _ledger()
    await ledger.note_post("100.000100", thread_root_ts="99.0", receipt_class="assistant_reply")
    assert await _state(temp_db, "100.000100") == "in_flight"
    await ledger.settle()
    row = await temp_db.get_receipt_async(TEAM, CH, "100.000100")
    assert row["state"] == "finalized"
    assert row["thread_root_ts"] == "99.0"
    assert row["receipt_class"] == "assistant_reply"


async def test_settle_finalizes_every_part_as_one_unit(service, temp_db):
    ledger = _ledger()
    for ts in ("100.0", "101.0", "102.0"):
        await ledger.note_post(ts, thread_root_ts="99.0", receipt_class="assistant_reply")
    assert await ledger.settle() == 3
    for ts in ("100.0", "101.0", "102.0"):
        assert await _state(temp_db, ts) == "finalized"


async def test_settle_is_idempotent(service, temp_db):
    ledger = _ledger()
    await ledger.note_post("100.0", receipt_class="assistant_reply")
    assert await ledger.settle() == 1
    assert await ledger.settle() == 0


async def test_chrome_never_settles_into_the_stream(service, temp_db):
    ledger = _ledger()
    await ledger.note_chrome("100.0", receipt_class="chrome")
    await ledger.settle()
    assert await _state(temp_db, "100.0") == "chrome"


async def test_promotion_turns_a_placeholder_into_conversation(service, temp_db):
    ledger = _ledger()
    await ledger.note_chrome("100.0", receipt_class="chrome")
    await ledger.promote("100.0")
    assert await _state(temp_db, "100.0") == "in_flight"
    await ledger.settle()
    assert await _state(temp_db, "100.0") == "finalized"


async def test_repeated_promotion_costs_one_write(service, temp_db, monkeypatch):
    ledger = _ledger()
    await ledger.note_chrome("100.0", receipt_class="chrome")
    calls = []
    original = temp_db.register_receipt_async

    async def counting(*args, **kwargs):
        calls.append(args)
        return await original(*args, **kwargs)

    monkeypatch.setattr(temp_db, "register_receipt_async", counting)
    for _ in range(5):
        await ledger.promote("100.0")
    assert len(calls) == 1


async def test_demotion_takes_an_overwritten_partial_back_out(service, temp_db):
    ledger = _ledger()
    await ledger.note_post("100.0", receipt_class="assistant_reply")
    await ledger.demote("100.0")
    assert await _state(temp_db, "100.0") == "chrome"
    await ledger.settle()
    assert await _state(temp_db, "100.0") == "chrome"


async def test_transfer_moves_a_chrome_surface_between_owners(service, temp_db):
    await _ledger("s:1").note_chrome("100.0", receipt_class="chrome")
    assert (await temp_db.transfer_receipt_async(TEAM, CH, "100.0", "s:1", "s:2")).applied
    assert (await temp_db.get_receipt_async(TEAM, CH, "100.0"))["turn_id"] == "s:2"


async def test_transfer_refused_once_the_surface_carries_words(service, temp_db):
    await _ledger("s:1").note_post("100.0", receipt_class="assistant_reply")
    refused = await temp_db.transfer_receipt_async(TEAM, CH, "100.0", "s:1", "s:2")
    assert (refused.applied, refused.reason) == (False, "not_chrome_or_foreign")


async def test_abort_drops_the_row(service, temp_db):
    ledger = _ledger()
    await ledger.note_post("100.0", receipt_class="assistant_reply")
    await ledger.abort("100.0")
    assert await temp_db.get_receipt_async(TEAM, CH, "100.0") is None
    await ledger.settle()
    assert await temp_db.get_receipt_async(TEAM, CH, "100.0") is None


# --------------------------------------------------------------- central finalization


async def test_an_exception_inside_the_turn_still_settles(service, temp_db):
    ledger = _ledger()

    async def turn_body():
        await ledger.note_post("100.0", receipt_class="assistant_reply")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        try:
            await turn_body()
        finally:
            await orx.settle_ledger(ledger)
    assert await _state(temp_db, "100.0") == "finalized"


async def test_a_cancelled_turn_still_settles(service, temp_db):
    ledger = _ledger()
    started = asyncio.Event()

    async def turn_body():
        await ledger.note_post("100.0", receipt_class="assistant_reply")
        started.set()
        try:
            await asyncio.sleep(30)
        finally:
            await orx.settle_ledger(ledger)

    task = asyncio.ensure_future(turn_body())
    # Bounded: if turn_body dies before started.set() (e.g. a TypeError out of note_post), the
    # bare wait() would hang the whole suite rather than fail this test.
    await asyncio.wait_for(started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await _state(temp_db, "100.0") == "finalized"


async def test_settle_ledger_tolerates_no_ledger():
    await orx.settle_ledger(None)


# --------------------------------------------------------------- the retry lattice


class _BrokenDB:
    """A database that refuses every receipt write."""

    def __init__(self):
        self.attempts = 0

    async def _fail(self, *args, **kwargs):
        self.attempts += 1
        raise RuntimeError("db down")

    register_receipt_async = _fail
    register_chrome_async = _fail
    demote_receipt_chrome_async = _fail
    finalize_receipts_async = _fail
    delete_receipt_async = _fail


async def test_a_failed_write_is_queued_and_drains_later(temp_db):
    orx.reset_service()
    broken = _BrokenDB()
    svc = orx.install_service(broken)
    try:
        ledger = _ledger()
        await ledger.note_post("100.0", receipt_class="assistant_reply")
        assert svc.queue_depth == 1
        svc.db = temp_db
        assert await svc.drain_once() == 1
        assert svc.queue_depth == 0
        assert await _state(temp_db, "100.0") == "in_flight"
    finally:
        orx.reset_service()


async def test_a_deletion_tombstone_is_never_resurrected(temp_db):
    orx.reset_service()
    broken = _BrokenDB()
    svc = orx.install_service(broken)
    try:
        ledger = _ledger()
        await ledger.note_post("100.0", receipt_class="assistant_reply")      # queued: register
        await ledger.abort("100.0")          # queued: delete, absorbing
        # …and a late retry must NOT bring the row back.
        await ledger.note_post("100.0", receipt_class="assistant_reply")
        assert svc.queue_depth == 1
        svc.db = temp_db
        await svc.drain_once()
        assert await temp_db.get_receipt_async(TEAM, CH, "100.0") is None
    finally:
        orx.reset_service()


async def test_finalize_absorbs_a_queued_registration(temp_db):
    orx.reset_service()
    broken = _BrokenDB()
    svc = orx.install_service(broken)
    try:
        ledger = _ledger()
        await ledger.note_post("100.0", thread_root_ts="99.0", receipt_class="assistant_reply")
        await ledger.settle()
        assert svc.queue_depth == 1
        svc.db = temp_db
        await svc.drain_once()
        row = await temp_db.get_receipt_async(TEAM, CH, "100.0")
        assert row["state"] == "finalized"
        assert row["thread_root_ts"] == "99.0"
        assert row["receipt_class"] == "assistant_reply"
    finally:
        orx.reset_service()


async def test_a_still_failing_op_stays_queued(caplog):
    orx.reset_service()
    svc = orx.install_service(_BrokenDB())
    try:
        ledger = _ledger()
        await ledger.note_post("100.0", receipt_class="assistant_reply")
        with caplog.at_level(logging.CRITICAL):
            assert await svc.drain_once() == 0
        assert svc.queue_depth == 1
    finally:
        orx.reset_service()


def test_the_lattice_ranks_delete_above_everything():
    svc = ReceiptService(db=None)
    svc._enqueue(_Op("register", TEAM, CH, "1.0", "s:1"))
    svc._enqueue(_Op("delete", TEAM, CH, "1.0", "s:1"))
    svc._enqueue(_Op("finalize", TEAM, CH, "1.0", "s:1"))
    assert svc._queue[(TEAM, CH, "1.0")].kind == "delete"


def test_a_queued_root_survives_a_weaker_later_op():
    svc = ReceiptService(db=None)
    svc._enqueue(_Op("finalize", TEAM, CH, "1.0", "s:1", "99.0"))
    svc._enqueue(_Op("register", TEAM, CH, "1.0", "s:1"))
    op = svc._queue[(TEAM, CH, "1.0")]
    assert (op.kind, op.thread_root_ts) == ("finalize", "99.0")


# --------------------------------------------------------------- the ledger record


async def test_a_registration_is_written_down_as_it_happened(service, temp_db, events):
    await _ledger().note_post("100.0", thread_root_ts="99.0", receipt_class="assistant_reply")
    assert events == [{"channel_id": CH, "message_ts": "100.0", "owner_turn_id": "s:1",
                       "op": "register", "prior_state": "absent", "new_state": "in_flight",
                       "applied": True, "reason": "inserted"}]


async def test_a_chrome_registration_is_a_register_op_with_a_chrome_end_state(service, temp_db,
                                                                             events):
    """One op, because both are registrations; the row's own `new_state` is what says which
    surface it made."""
    await _ledger().note_chrome("100.0", receipt_class="chrome")
    assert [(e["op"], e["new_state"], e["reason"]) for e in events] == \
        [("register", "chrome", "inserted")]


async def test_a_promotion_records_the_state_it_moved(service, temp_db, events):
    ledger = _ledger()
    await ledger.note_chrome("100.0", receipt_class="chrome")
    await ledger.promote("100.0")
    assert [(e["op"], e["prior_state"], e["new_state"], e["reason"]) for e in events] == \
        [("register", "absent", "chrome", "inserted"),
         ("promote", "chrome", "in_flight", "transitioned")]


async def test_a_demotion_records_its_own_op(service, temp_db, events):
    ledger = _ledger()
    await ledger.note_post("100.0", receipt_class="assistant_reply")
    await ledger.demote("100.0")
    assert [(e["op"], e["prior_state"], e["new_state"], e["reason"]) for e in events][-1] == \
        ("demote", "in_flight", "chrome", "demoted")


async def test_a_refused_demotion_is_recorded_as_a_refusal_not_a_failure(service, temp_db,
                                                                        events):
    """The row belongs to somebody else. Nothing moved, and nothing broke either — the line that
    says so is the only evidence this ever happened."""
    await _ledger("s:OTHER").note_post("100.0", receipt_class="assistant_reply")
    await _ledger("s:1").demote("100.0")
    demotes = [e for e in events if e["op"] == "demote"]
    assert [(e["applied"], e["reason"]) for e in demotes] == \
        [(False, "not_in_flight_or_foreign")]


async def test_a_single_finalize_op_is_recorded(service, temp_db, events):
    await _ledger().note_post("100.0", orx.STATE_FINALIZED, "99.0",
                  receipt_class="assistant_reply")
    assert [(e["op"], e["prior_state"], e["new_state"], e["reason"]) for e in events] == \
        [("finalize", "absent", "finalized", "inserted")]


async def test_a_unit_settle_writes_one_line_per_message(service, temp_db, events):
    ledger = _ledger()
    for ts in ("100.0", "101.0", "102.0"):
        await ledger.note_post(ts, receipt_class="assistant_reply")
    events.clear()
    await ledger.settle()
    assert [(e["op"], e["message_ts"], e["prior_state"], e["new_state"], e["reason"])
            for e in events] == [
        ("finalize", "100.0", "in_flight", "finalized", "finalized"),
        ("finalize", "101.0", "in_flight", "finalized", "finalized"),
        ("finalize", "102.0", "in_flight", "finalized", "finalized")]


async def test_a_unit_settle_a_foreign_turn_owns_records_the_refusal_per_message(service, temp_db,
                                                                                events):
    """The zip is the point: one line each, attributed to the right message, with only the
    contested one refused."""
    ledger = _ledger("s:1")
    await ledger.note_post("100.0", receipt_class="assistant_reply")
    await ledger.note_post("101.0", receipt_class="assistant_reply")
    await temp_db.delete_receipt_async(TEAM, CH, "101.0")
    await temp_db.register_receipt_async(TEAM, CH, "101.0", "s:OTHER", "in_flight",
                                         receipt_class="assistant_reply")
    events.clear()
    await ledger.settle()
    assert [(e["message_ts"], e["applied"], e["reason"]) for e in events] == \
        [("100.0", True, "finalized"), ("101.0", False, "foreign_owner")]


async def test_a_deletion_records_the_state_it_removed(service, temp_db, events):
    ledger = _ledger()
    await ledger.note_post("100.0", receipt_class="assistant_reply")
    events.clear()
    await ledger.abort("100.0")
    assert events == [{"channel_id": CH, "message_ts": "100.0", "owner_turn_id": "s:1",
                       "op": "delete", "prior_state": "in_flight", "new_state": "absent",
                       "applied": True, "reason": "deleted"}]


async def test_a_resolved_share_is_recorded_as_a_pending_resolve(service, temp_db, events):
    await orx.record_pending_share(temp_db, team_id=TEAM, channel_id=CH, file_id="F1",
                                   owner_turn_id="s:1", thread_root_ts="99.0",
                                   receipt_class="artifact")
    assert events == [], "a pending-share row is not an outbound receipt"
    await orx.resolve_pending_share(temp_db, team_id=TEAM, channel_id=CH, file_id="F1",
                                   message_ts="150.0")
    assert [(e["op"], e["message_ts"], e["new_state"], e["applied"], e["reason"])
            for e in events] == \
        [("pending_resolve", "150.0", "finalized", True, "resolved")]


async def test_a_share_resolved_out_of_the_queue_records_both_halves(service, temp_db, events):
    """The queued write never reached the database, so the finalize goes straight in — and both
    the transition and the resolution it served are on the record."""
    class _Refusing:
        async def record_pending_share_async(self, *a, **k):
            raise RuntimeError("db busy")

    await orx.record_pending_share(_Refusing(), team_id=TEAM, channel_id=CH, file_id="F1",
                                   owner_turn_id="s:1", thread_root_ts="99.0",
                                   receipt_class="artifact")
    events.clear()
    assert await orx.resolve_pending_share(temp_db, team_id=TEAM, channel_id=CH, file_id="F1",
                                           message_ts="150.0")
    assert [(e["op"], e["reason"], e["applied"]) for e in events] == \
        [("finalize", "inserted", True), ("pending_resolve", "queued_finalize", True)]


async def test_a_failed_resolution_writes_nothing(service, temp_db, events):
    class _Exploding:
        async def resolve_pending_share_async(self, *a, **k):
            raise RuntimeError("db down")

    assert not await orx.resolve_pending_share(
        _Exploding(), team_id=TEAM, channel_id=CH, file_id="F1", message_ts="150.0")
    assert events == []


async def test_a_raising_write_writes_no_line_and_is_retried(service, temp_db, events):
    """A refusal is a completed write with a reason; a raise is not a transition at all. Only the
    second is retained by the lattice, and only the first earns a line."""
    orx.reset_service()
    svc = orx.install_service(_BrokenDB())
    try:
        await _ledger().note_post("100.0", receipt_class="assistant_reply")
        assert events == []
        assert svc.queue_depth == 1
    finally:
        orx.reset_service()


async def test_a_refused_transition_is_not_retried(service, temp_db, events):
    """`_write` returns True for an applied=False result: the database answered, and asking it the
    same question forever would never change the answer."""
    await _ledger("s:OTHER").note_post("100.0", receipt_class="assistant_reply")
    events.clear()
    await _ledger("s:1").note_post("100.0", receipt_class="assistant_reply")
    assert service.queue_depth == 0
    assert [(e["applied"], e["reason"]) for e in events] == [(False, "foreign_owner")]


async def test_the_ledger_speaks_the_declared_vocabulary(service, temp_db, events):
    """A typo in an op or a state does not fail anything at runtime — it silently invents a bucket
    and deflates the real one, so it is caught here instead."""
    from message_processor import participation_telemetry as pt

    ledger = _ledger()
    await ledger.note_chrome("100.0", receipt_class="chrome")
    await ledger.promote("100.0")
    await ledger.demote("100.0")
    await ledger.note_post("101.0", receipt_class="assistant_reply")
    await ledger.settle()
    await ledger.abort("101.0")
    await orx.record_pending_share(temp_db, team_id=TEAM, channel_id=CH, file_id="F1",
                                   owner_turn_id="s:1", receipt_class="artifact")
    await orx.resolve_pending_share(temp_db, team_id=TEAM, channel_id=CH, file_id="F1",
                                   message_ts="150.0")
    assert events
    for event in events:
        assert event["op"] in pt.RECEIPT_OPS, event
        for key in ("prior_state", "new_state"):
            assert event[key] is None or event[key] in pt.RECEIPT_STATES, event


# --------------------------------------------------------------- the partial-post seam


async def test_the_first_conversational_surface_pauses_a_turn_ledger(service, temp_db, barriers):
    ledger = orx.ledger_for("s:1", TEAM, CH)
    await ledger.note_post("100.0", receipt_class="assistant_reply")
    await ledger.note_post("101.0", receipt_class="assistant_reply")
    assert barriers == [{"channel_id": CH, "message_ts": "100.0", "owner": "s:1"}], \
        "the seam is the FIRST surface, not every one"


async def test_a_detached_job_ledger_never_pauses_a_battery(service, temp_db, barriers):
    """ROUND-4/5: a battery holding one turn at the seam must not freeze an unrelated detached
    post. A job's status card and its findings are on their own schedule, and the turn the harness
    is waiting to release is not coming."""
    job = orx.ledger_for_job("g1", TEAM, CH)
    assert not job._barrier_eligible
    await job.note_post("100.0", receipt_class="background_job")
    await job.promote("100.0")
    assert barriers == []


async def test_a_sys_owner_post_never_pauses_a_battery(service, temp_db, barriers):
    await record_transport_post(team_id=TEAM, channel_id=CH, message_ts="100.0", receipts=None,
                               receipt_kind="finalized", site="channel_intro",
                               receipt_class="system_notice")
    assert barriers == []


async def test_the_promote_path_reaches_the_seam_exactly_once(service, temp_db, barriers):
    """ROUND-4/5: on the legacy-placeholder path the first conversational surface is the PROMOTE,
    not the post — the message already exists as chrome and this is the edit that makes it an
    answer. Every later edit grows a surface the turn already owns."""
    ledger = orx.ledger_for("s:1", TEAM, CH)
    await ledger.note_chrome("100.0", receipt_class="chrome")
    assert barriers == [], "chrome is not conversation"
    await ledger.promote("100.0")
    await ledger.promote("100.0")
    await ledger.note_post("101.0", receipt_class="assistant_reply")
    assert barriers == [{"channel_id": CH, "message_ts": "100.0", "owner": "s:1"}]


async def test_the_seam_is_a_hard_no_op_without_the_env_flag(service, temp_db, monkeypatch):
    """No DEV_TURN_BARRIERS, nothing touched — this sits on the production turn path."""
    monkeypatch.delenv("DEV_TURN_BARRIERS", raising=False)
    ledger = orx.ledger_for("s:1", TEAM, CH)
    await ledger.note_post("100.0", receipt_class="assistant_reply")
    assert await _state(temp_db, "100.0") == "in_flight"


async def test_a_seam_that_raises_never_costs_the_receipt(service, temp_db, monkeypatch):
    from message_processor import dev_barriers

    async def _explode(**context):
        raise RuntimeError("no scratch dir")

    monkeypatch.setattr(dev_barriers, "post_partial_post", _explode)
    ledger = orx.ledger_for("s:1", TEAM, CH)
    await ledger.note_post("100.0", receipt_class="assistant_reply")
    assert await _state(temp_db, "100.0") == "in_flight"


# --------------------------------------------------------------- intent contract


async def test_a_durable_post_with_no_intent_is_loud_and_still_recorded(service, temp_db, caplog):
    with caplog.at_level(logging.ERROR):
        await record_transport_post(team_id=TEAM, channel_id=CH, message_ts="100.0",
                                    receipts=None, site="mystery_site",
                                    receipt_class="system_notice")
    assert "mystery_site" in caplog.text
    row = await temp_db.get_receipt_async(TEAM, CH, "100.0")
    assert row["state"] == "finalized"
    assert row["turn_id"] == orx.sys_owner()


async def test_a_declared_chrome_post_needs_no_ledger(service, temp_db, caplog):
    with caplog.at_level(logging.ERROR):
        await record_transport_post(team_id=TEAM, channel_id=CH, message_ts="100.0",
                                    receipts=None, receipt_kind="chrome", site="footer",
                                    receipt_class="chrome")
    assert caplog.text == ""
    assert await _state(temp_db, "100.0") == "chrome"


async def test_a_declared_finalized_post_needs_no_ledger(service, temp_db):
    await record_transport_post(team_id=TEAM, channel_id=CH, message_ts="100.0",
                                receipts=None, receipt_kind="finalized",
                                thread_root_ts="99.0", site="channel_intro",
                                receipt_class="system_notice")
    row = await temp_db.get_receipt_async(TEAM, CH, "100.0")
    assert (row["state"], row["thread_root_ts"]) == ("finalized", "99.0")


async def test_an_ephemeral_with_no_ts_records_nothing(service, temp_db, caplog):
    with caplog.at_level(logging.ERROR):
        await record_transport_post(team_id=TEAM, channel_id=CH, message_ts=None,
                                    receipts=None, site="ephemeral", receipt_class="system_notice")
    assert caplog.text == ""


async def test_a_dm_post_with_no_intent_is_not_an_error(service, caplog):
    with caplog.at_level(logging.ERROR):
        await record_transport_post(team_id=TEAM, channel_id=DM, message_ts="100.0",
                                    receipts=None, site="dm_reply",
                                    receipt_class="assistant_reply")
    assert caplog.text == ""


# --------------------------------------------------------------- transport wiring


class _FakeSlack:
    def __init__(self, share_ts="150.0"):
        self.posts = []
        self.uploads = []
        self.share_ts = share_ts
        self.info_calls = 0
        self.next_ts = iter([f"20{i}.0" for i in range(1, 40)])

    async def chat_postMessage(self, **kwargs):
        self.posts.append(kwargs)
        return {"ts": next(self.next_ts)}

    async def chat_update(self, **kwargs):
        return {"ok": True}

    async def files_upload_v2(self, **kwargs):
        self.uploads.append(kwargs)
        return {"files": [{"id": "F1", "url_private": "https://slack/f",
                           "permalink": "https://slack/p"}]}

    async def files_info(self, file):
        self.info_calls += 1
        shares = {"public": {CH: [{"ts": self.share_ts}]}} if self.share_ts else {}
        return {"file": {"shares": shares}}


class _FakeApp:
    def __init__(self, slack):
        self.client = slack


def _messaging(db, slack):
    """A SlackBot messaging mixin with just enough around it to post."""
    from slack_client.messaging import SlackMessagingMixin

    class _Bot(SlackMessagingMixin):
        MAX_MESSAGE_LENGTH = 3000

        def __init__(self):
            self.app = _FakeApp(slack)
            self.db = db
            self.self_team_id = TEAM

        def format_text(self, text):
            return text

        def _record_own_reply_pulse(self, *a, **k):
            return None

        def log_info(self, *a, **k):
            pass

        log_debug = log_warning = log_error = log_info

    return _Bot()


async def test_send_message_registers_its_post_under_the_turn(service, temp_db):
    bot = _messaging(temp_db, _FakeSlack())
    ledger = _ledger()
    ts = await bot.send_message(CH, "99.0", "hello", receipts=ledger,
                                receipt_class="assistant_reply")
    row = await temp_db.get_receipt_async(TEAM, CH, ts)
    assert (row["state"], row["turn_id"], row["thread_root_ts"]) == ("in_flight", "s:1", "99.0")
    assert row["receipt_class"] == "assistant_reply"


async def test_every_split_chunk_earns_a_receipt_and_they_finalize_together(service, temp_db):
    bot = _messaging(temp_db, _FakeSlack())
    bot.MAX_MESSAGE_LENGTH = 400
    ledger = _ledger()
    await bot.send_message(CH, "99.0", "x " * 900, receipts=ledger,
                           receipt_class="assistant_reply")
    assert len(ledger.pending_ts) > 1
    await ledger.settle()
    for ts in await temp_db.get_channel_receipts_async(TEAM, CH):
        assert ts["state"] == "finalized"
        assert ts["receipt_class"] == "assistant_reply"


async def test_a_thinking_placeholder_registers_as_chrome(service, temp_db):
    bot = _messaging(temp_db, _FakeSlack())

    async def no_status(*a, **k):
        return False

    bot.set_assistant_status = no_status
    ledger = _ledger()
    ts = await bot.send_thinking_indicator(CH, "99.0", receipts=ledger,
                                           receipt_class="chrome")
    assert await _state(temp_db, ts) == "chrome"


async def test_a_stale_send_suppression_passes_straight_through(service, temp_db):
    bot = _messaging(temp_db, _FakeSlack())

    class _Lease:
        def authorize(self, surface):
            raise StaleSendSuppressed(surface=surface)

        def commit(self):
            raise AssertionError("never reached")

    with pytest.raises(StaleSendSuppressed):
        await bot.send_message(CH, "99.0", "hi", lease=_Lease(), receipts=_ledger(),
                               receipt_class="assistant_reply")
    assert await temp_db.get_channel_receipts_async(TEAM, CH) == []


async def test_post_to_thread_records_the_target_root(service, temp_db):
    bot = _messaging(temp_db, _FakeSlack())
    ledger = _ledger()

    class _Turn:
        send_lease = None
        receipt_ledger = ledger
        visible_action_committed = False

    class _Ctx:
        channel_id = CH
        thread_ts = "50.0"
        trigger_ts = "50.1"
        turn = _Turn()

    result = await bot.execute_post_to_thread(_Ctx(), {"thread_ts": "77.0", "text": "over here"})
    assert result["ok"]
    row = await temp_db.get_receipt_async(TEAM, CH, result["posted_ts"])
    assert row["thread_root_ts"] == "77.0"
    assert row["receipt_class"] == "assistant_reply"


async def test_a_native_stream_claims_every_part(service, temp_db):
    from streaming.native_sink import NativeStreamCoordinator

    class _Session:
        def __init__(self, ts):
            self.ts = ts
            self.active = True
            self._sent = ""

        async def start(self, initial_text="", lease=None):
            self._sent = initial_text or ""
            return True

    class _Client:
        def __init__(self):
            self.made = 0

        def begin_native_stream(self, *a, **k):
            self.made += 1
            return _Session(f"30{self.made}.0")

    ledger = _ledger()
    coord = NativeStreamCoordinator(_Client(), CH, "99.0", char_limit=500, receipts=ledger)
    assert await coord.start()
    assert await _state(temp_db, "301.0") == "in_flight"
    assert ledger.pending_ts == ["301.0"]
    await ledger.settle()
    assert await _state(temp_db, "301.0") == "finalized"


# --------------------------------------------------------------- uploads


async def test_an_upload_records_a_pending_share_then_resolves_it(service, temp_db):
    ledger = _ledger()
    assert await orx.record_pending_share(
        temp_db, team_id=TEAM, channel_id=CH, file_id="F1",
        owner_turn_id=ledger.owner_id, thread_root_ts="99.0", receipt_class="artifact")
    assert len(await temp_db.get_pending_shares_async()) == 1

    assert await orx.resolve_pending_share(
        temp_db, team_id=TEAM, channel_id=CH, file_id="F1", message_ts="150.0")
    assert await temp_db.get_pending_shares_async() == []
    row = await temp_db.get_receipt_async(TEAM, CH, "150.0")
    assert (row["state"], row["turn_id"], row["thread_root_ts"]) == ("finalized", "s:1", "99.0")
    assert row["receipt_class"] == "artifact"


async def _finish_resolvers(service):
    """Await the detached share-ts polls the transport started."""
    tasks = list(service._resolvers)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def test_an_artifact_upload_resolves_its_own_share_receipt(service, temp_db):
    # The whole point of finding 1: nobody else is watching a plain file upload, so if the
    # transport does not start the poll the artifact never earns a receipt at all.
    bot = _messaging(temp_db, _FakeSlack())
    identity = await bot.send_file(CH, "99.0", io.BytesIO(b"x,y\n1,2\n"), "report.csv",
                                   receipt_class="artifact",
                                   receipts=_ledger())
    assert identity["file_id"] == "F1"
    assert len(await temp_db.get_pending_shares_async()) == 1

    await _finish_resolvers(service)
    assert await temp_db.get_pending_shares_async() == []
    row = await temp_db.get_receipt_async(TEAM, CH, "150.0")
    assert (row["state"], row["turn_id"], row["thread_root_ts"]) == ("finalized", "s:1", "99.0")
    assert row["receipt_class"] == "artifact"


async def test_an_unresolvable_artifact_keeps_its_row_for_boot_recovery(service, temp_db,
                                                                        caplog):
    bot = _messaging(temp_db, _FakeSlack(share_ts=None))
    with patch.object(config, "image_share_ts_timeout_seconds", 0.0):
        await bot.send_file(CH, "99.0", io.BytesIO(b"x"), "report.csv",
                            receipts=_ledger(), receipt_class="artifact")
        with caplog.at_level(logging.WARNING):
            await _finish_resolvers(service)
    assert len(await temp_db.get_pending_shares_async()) == 1
    assert await temp_db.get_channel_receipts_async(TEAM, CH) == []


async def test_an_image_upload_leaves_the_poll_to_image_delivery(service, temp_db):
    # One poll there feeds the indicator, provenance and the receipt; a second one here would
    # double the API calls to learn the same fact.
    bot = _messaging(temp_db, _FakeSlack())
    await bot.send_image(CH, "99.0", b"bytes", "pic.png", receipts=_ledger(),
                         receipt_class="artifact")
    assert not service._resolvers
    assert len(await temp_db.get_pending_shares_async()) == 1


async def test_a_dm_artifact_upload_starts_no_poll(service, temp_db):
    bot = _messaging(temp_db, _FakeSlack())
    await bot.send_file(DM, "99.0", io.BytesIO(b"x"), "report.csv",
                        receipts=_ledger(channel=DM), receipt_class="artifact")
    assert not service._resolvers
    assert await temp_db.get_pending_shares_async() == []


async def test_a_pending_row_that_never_landed_is_finalized_from_the_queue(service, temp_db):
    class _Refusing:
        async def record_pending_share_async(self, *a, **k):
            raise RuntimeError("db busy")

    assert not await orx.record_pending_share(
        _Refusing(), team_id=TEAM, channel_id=CH, file_id="F1", owner_turn_id="s:1",
        thread_root_ts="99.0", receipt_class="artifact")
    assert service.queue_depth == 1

    # The share ts arrived while the write was still queued: finalize the message directly
    # rather than writing a row only to delete it (and the row-less resolve would write
    # nothing at all).
    assert await orx.resolve_pending_share(temp_db, team_id=TEAM, channel_id=CH,
                                           file_id="F1", message_ts="150.0")
    assert service.queue_depth == 0
    row = await temp_db.get_receipt_async(TEAM, CH, "150.0")
    assert (row["state"], row["turn_id"], row["thread_root_ts"]) == ("finalized", "s:1", "99.0")
    assert row["receipt_class"] == "artifact"


async def test_a_queued_pending_row_drains_when_the_database_comes_back(service, temp_db):
    class _Refusing:
        async def record_pending_share_async(self, *a, **k):
            raise RuntimeError("db busy")

    await orx.record_pending_share(_Refusing(), team_id=TEAM, channel_id=CH, file_id="F1",
                                   owner_turn_id="s:1", thread_root_ts="99.0",
                                   receipt_class="artifact")
    assert await service.drain_once() == 1
    rows = await temp_db.get_pending_shares_async()
    assert (rows[0]["file_id"], rows[0]["owner_turn_id"]) == ("F1", "s:1")


async def test_a_deleted_file_drops_its_pending_row(service, temp_db):
    await orx.record_pending_share(temp_db, team_id=TEAM, channel_id=CH, file_id="F1",
                                   owner_turn_id="s:1", receipt_class="artifact")
    assert await orx.delete_pending_shares_for_file(temp_db, "F1") == 1
    assert await temp_db.get_pending_shares_async() == []


async def test_a_late_registration_cannot_resurrect_a_deleted_file(service, temp_db):
    """Codex's interleaving. `files_upload_v2` returning and Slack's `file_deleted` are separate,
    UNORDERED listeners, so the cleanup can run while there is nothing queued to absorb it — and
    the registration that arrives afterwards writes a pending row for a file that no longer
    exists. That row can never resolve: it is retried and logged critically on every boot from
    then on. A per-key queue tombstone cannot cover it, because by then the key is empty."""
    await orx.record_pending_share(temp_db, team_id=TEAM, channel_id=CH, file_id="F1",
                                   owner_turn_id="s:1", receipt_class="artifact")
    assert await orx.delete_pending_shares_for_file(temp_db, "F1") == 1
    # No row, and nothing queued to absorb what comes next.
    assert await temp_db.get_pending_shares_async() == []
    assert service.queue_depth == 0

    # …and NOW the upload's own registration lands.
    assert not await orx.record_pending_share(temp_db, team_id=TEAM, channel_id=CH, file_id="F1",
                                              owner_turn_id="s:1", receipt_class="artifact")
    assert await temp_db.get_pending_shares_async() == []
    assert service.queue_depth == 0


async def test_a_deletion_that_overlaps_the_registration_still_wins(service, temp_db):
    """Codex's OVERLAPPING interleaving, the one a single check cannot cover.

    The registration passes its tombstone check and starts writing. The deletion then stamps,
    scans for rows, sees nothing (the write has not committed), and finishes — and the row lands
    behind it, unresolvable, retried and logged critically on every boot forever after.

    The two paths are ordered by the deletion stamping BEFORE it scans, which makes a second,
    post-commit check on this side sufficient: whichever way they interleave, one of them sees
    the other."""
    real_record = temp_db.record_pending_share_async
    deletion_done = []

    async def _record_then_let_the_deletion_run(*a, **k):
        # The deletion lands INSIDE the write, before it commits.
        await orx.delete_pending_shares_for_file(temp_db, "F1")
        deletion_done.append(True)
        return await real_record(*a, **k)

    with patch.object(temp_db, "record_pending_share_async",
                      _record_then_let_the_deletion_run):
        wrote = await orx.record_pending_share(temp_db, team_id=TEAM, channel_id=CH,
                                               file_id="F1", owner_turn_id="s:1",
                                               receipt_class="artifact")

    assert deletion_done, "the interleaving never happened"
    assert not wrote
    assert await temp_db.get_pending_shares_async() == [], \
        "a pending row survived for a file Slack destroyed"


async def test_an_overlapping_deletion_is_retried_when_the_compensating_delete_fails(service,
                                                                                     temp_db):
    """Same overlap, but the compensating delete hits a busy database. It must be retained —
    the row it removes can never resolve, so losing the compensation is losing the row."""
    real_record = temp_db.record_pending_share_async

    async def _record_then_delete(*a, **k):
        await orx.delete_pending_shares_for_file(temp_db, "F1")
        return await real_record(*a, **k)

    async def _refuse_delete(*a, **k):
        raise RuntimeError("db busy")

    with patch.object(temp_db, "record_pending_share_async", _record_then_delete), \
         patch.object(temp_db, "delete_pending_share_async", _refuse_delete):
        assert not await orx.record_pending_share(temp_db, team_id=TEAM, channel_id=CH,
                                                  file_id="F1", owner_turn_id="s:1",
                                                  receipt_class="artifact")
    assert service.queue_depth == 1
    assert len(await temp_db.get_pending_shares_async()) == 1

    # The database comes back and the retained op finishes the job.
    assert await service.drain_once() == 1
    assert await temp_db.get_pending_shares_async() == []


async def test_a_queued_registration_that_overlaps_a_deletion_is_compensated(service, temp_db):
    """The same overlap on the LATTICE path: the drain's own pending-share write is in flight
    when the deletion stamps and scans."""
    class _Refusing:
        async def record_pending_share_async(self, *a, **k):
            raise RuntimeError("db busy")

    await orx.record_pending_share(_Refusing(), team_id=TEAM, channel_id=CH, file_id="F1",
                                   owner_turn_id="s:1", receipt_class="artifact")
    assert service.queue_depth == 1

    real_record = temp_db.record_pending_share_async

    async def _record_then_let_the_deletion_run(*a, **k):
        await orx.delete_pending_shares_for_file(temp_db, "F1")
        return await real_record(*a, **k)

    with patch.object(temp_db, "record_pending_share_async",
                      _record_then_let_the_deletion_run):
        await service.drain_once()

    assert await temp_db.get_pending_shares_async() == []


async def test_lattice_compensation_survives_a_closed_queue(service, temp_db):
    """The final drain writes a queued pending share, an overlapping deletion means it must come
    straight back out, and the delete itself fails. That retry is the service compensating its
    OWN write — a normal enqueue would be refused now that the queue is closed, and the stale
    row would outlive the process."""
    class _Refusing:
        async def record_pending_share_async(self, *a, **k):
            raise RuntimeError("db busy")

    await orx.record_pending_share(_Refusing(), team_id=TEAM, channel_id=CH, file_id="F1",
                                   owner_turn_id="s:1", receipt_class="artifact")
    assert service.queue_depth == 1

    real_record = temp_db.record_pending_share_async
    deletes = []

    async def _record_then_let_the_deletion_run(*a, **k):
        await orx.delete_pending_shares_for_file(temp_db, "F1")
        return await real_record(*a, **k)

    async def _refuse_first_delete(*a, **k):
        deletes.append(a)
        if len(deletes) == 1:
            raise RuntimeError("db busy")
        return True

    with patch.object(temp_db, "record_pending_share_async",
                      _record_then_let_the_deletion_run), \
         patch.object(temp_db, "delete_pending_share_async", _refuse_first_delete):
        await service.shutdown()

    # The compensation was retained despite the closed queue, and the retry ran.
    assert len(deletes) > 1, "the compensating delete was refused and never retried"
    assert service.queue_depth == 0


async def test_a_deleted_file_is_refused_even_when_the_row_never_landed(service, temp_db):
    """Same race, but the late registration hits a busy database. The lattice must not retain
    provenance for a file Slack has already confirmed gone."""
    class _Refusing:
        async def record_pending_share_async(self, *a, **k):
            raise RuntimeError("db busy")

    await orx.delete_pending_shares_for_file(temp_db, "F1")
    assert not await orx.record_pending_share(_Refusing(), team_id=TEAM, channel_id=CH,
                                              file_id="F1", owner_turn_id="s:1",
                                              receipt_class="artifact")
    assert service.queue_depth == 0


async def test_a_queued_pending_write_is_dropped_when_the_deletion_beats_its_drain(service,
                                                                                   temp_db):
    """The registration was queued BEFORE the deletion, and the drain runs after it. Committing
    it then would resurrect the row the cleanup just removed."""
    class _Refusing:
        async def record_pending_share_async(self, *a, **k):
            raise RuntimeError("db busy")

    await orx.record_pending_share(_Refusing(), team_id=TEAM, channel_id=CH, file_id="F1",
                                   owner_turn_id="s:1", receipt_class="artifact")
    assert service.queue_depth == 1
    service.note_file_deleted("F1")
    # Drained as a success — there is nothing left to write — and no row appears.
    assert await service.drain_once() == 1
    assert await temp_db.get_pending_shares_async() == []
    assert service.queue_depth == 0


async def test_the_deletion_tombstone_is_bounded(service):
    for i in range(orx._DELETED_FILE_MEMORY + 50):
        service.note_file_deleted(f"F{i}")
    assert len(service._deleted_files) == orx._DELETED_FILE_MEMORY
    assert service.file_is_deleted(f"F{orx._DELETED_FILE_MEMORY + 49}")
    assert not service.file_is_deleted("F0")  # the oldest fall off first


async def test_a_deleted_file_tombstones_a_queued_pending_write(service, temp_db):
    class _Refusing:
        async def record_pending_share_async(self, *a, **k):
            raise RuntimeError("db busy")

    await orx.record_pending_share(_Refusing(), team_id=TEAM, channel_id=CH, file_id="F1",
                                   owner_turn_id="s:1", receipt_class="artifact")
    await orx.delete_pending_shares_for_file(temp_db, "F1")
    await service.drain_once()
    assert await temp_db.get_pending_shares_async() == []


async def test_the_file_deleted_event_reaches_the_pending_cleanup(service, temp_db):
    from slack_client.event_handlers.message_events import SlackMessageEventsMixin

    class _Events(SlackMessageEventsMixin):
        def __init__(self, db):
            self.db = db

        def log_debug(self, *a, **k):
            pass

    await orx.record_pending_share(temp_db, team_id=TEAM, channel_id=CH, file_id="F1",
                                   owner_turn_id="s:1", receipt_class="artifact")
    await _Events(temp_db)._ambient_file_deleted({"file_id": "F1"})
    assert await temp_db.get_pending_shares_async() == []


async def test_a_dm_upload_records_no_pending_row(service, temp_db):
    assert not await orx.record_pending_share(
        temp_db, team_id=TEAM, channel_id=DM, file_id="F1", owner_turn_id="s:1",
        receipt_class="artifact")
    assert await temp_db.get_pending_shares_async() == []


async def test_a_failed_resolution_keeps_the_row_for_boot_recovery(service, temp_db):
    await orx.record_pending_share(temp_db, team_id=TEAM, channel_id=CH, file_id="F1",
                                   owner_turn_id="s:1", receipt_class="artifact")

    class _Exploding:
        async def resolve_pending_share_async(self, *a, **k):
            raise RuntimeError("db down")

    assert not await orx.resolve_pending_share(
        _Exploding(), team_id=TEAM, channel_id=CH, file_id="F1", message_ts="150.0")
    assert len(await temp_db.get_pending_shares_async()) == 1


async def test_boot_recovery_resolves_leftovers(service, temp_db):
    await orx.record_pending_share(temp_db, team_id=TEAM, channel_id=CH, file_id="F1",
                                   owner_turn_id="s_old:3", thread_root_ts="99.0",
                                   receipt_class="artifact")

    class _Client:
        async def resolve_file_share_ts(self, channel_id, file_id):
            return "150.0"

    assert await orx.recover_pending_shares(temp_db, _Client()) == 1
    assert await temp_db.get_pending_shares_async() == []
    assert await _state(temp_db, "150.0") == "finalized"


async def test_boot_recovery_retains_a_share_slack_still_cannot_place(service, temp_db, caplog):
    await orx.record_pending_share(temp_db, team_id=TEAM, channel_id=CH, file_id="F1",
                                   owner_turn_id="s_old:3", receipt_class="artifact")

    class _Client:
        async def resolve_file_share_ts(self, channel_id, file_id):
            return None

    with caplog.at_level(logging.CRITICAL):
        assert await orx.recover_pending_shares(temp_db, _Client()) == 0
    assert len(await temp_db.get_pending_shares_async()) == 1


# --------------------------------------------------------------- boot


async def test_dead_session_reconcile_spares_the_live_session(service, temp_db):
    live = orx.next_turn_id()
    await temp_db.register_receipt_async(TEAM, CH, "100.0", live, "in_flight",
                                         receipt_class="assistant_reply")
    await temp_db.register_receipt_async(TEAM, CH, "101.0", "deadsession:4", "in_flight",
                                         receipt_class="assistant_reply")

    import runtime_identity
    moved = await temp_db.finalize_dead_session_receipts_async(runtime_identity.SESSION_ID)
    assert [r["turn_id"] for r in moved] == ["deadsession:4"]
    assert await _state(temp_db, "100.0") == "in_flight"
    assert await _state(temp_db, "101.0") == "finalized"


async def test_reconcile_emits_one_row_per_recovered_message(service, temp_db, events):
    await temp_db.register_receipt_async(TEAM, CH, "100.0", orx.next_turn_id(), "in_flight",
                                         receipt_class="assistant_reply")
    await temp_db.register_receipt_async(TEAM, CH, "101.0", "deadsession:4", "in_flight",
                                         receipt_class="assistant_reply")
    await temp_db.register_receipt_async(TEAM, "C2", "102.0", "deadsession:5", "in_flight",
                                         receipt_class="assistant_reply")

    assert await orx.reconcile_dead_sessions(temp_db) == 2
    assert [(e["op"], e["message_ts"], e["owner_turn_id"], e["prior_state"], e["new_state"],
             e["applied"]) for e in events] == [
        ("reconcile_finalize", "101.0", "deadsession:4", "in_flight", "finalized", True),
        ("reconcile_finalize", "102.0", "deadsession:5", "in_flight", "finalized", True)]
    assert {e["channel_id"] for e in events} == {CH, "C2"}


async def test_reconcile_tolerates_a_database_that_only_returns_a_count(events):
    """Every startup test in the suite mocks this accessor with `return_value=0`. A count carries
    no per-row detail, so there is nothing to write — and inventing a row would be worse."""
    class _Counting:
        async def finalize_dead_session_receipts_async(self, live_session_id):
            return 3

    assert await orx.reconcile_dead_sessions(_Counting()) == 3
    assert events == []


async def test_reconcile_honours_an_explicit_live_session(temp_db):
    asked = []

    class _Recording:
        async def finalize_dead_session_receipts_async(self, live_session_id):
            asked.append(live_session_id)
            return []

    assert await orx.reconcile_dead_sessions(_Recording(), "other-session") == 0
    assert asked == ["other-session"]


async def test_reconcile_lets_a_failing_database_raise(temp_db):
    """Boot treats a failed reconcile as fatal — the previous session's replies would stay out of
    the stream for this whole process — so this must not be swallowed into a zero."""
    class _Broken:
        async def finalize_dead_session_receipts_async(self, live_session_id):
            raise RuntimeError("db down")

    with pytest.raises(RuntimeError):
        await orx.reconcile_dead_sessions(_Broken())


async def test_the_epoch_is_written_once_and_read_back(temp_db):
    first = await orx.establish_epoch(temp_db)
    assert first == await orx.establish_epoch(temp_db)
    assert await temp_db.get_meta_async(OUTBOUND_RECEIPTS_EPOCH_KEY) == first


async def test_an_unestablishable_epoch_raises_so_boot_can_refuse():
    class _Amnesiac:
        async def set_meta_if_absent_async(self, key, value):
            return True

        async def get_meta_async(self, key):
            return None

    with pytest.raises(RuntimeError):
        await orx.establish_epoch(_Amnesiac())


async def test_service_shutdown_drains_then_reports_what_it_could_not_write(temp_db, caplog):
    orx.reset_service()
    svc = orx.install_service(_BrokenDB())
    try:
        await _ledger().note_post("100.0", receipt_class="assistant_reply")
        with caplog.at_level(logging.CRITICAL):
            await svc.shutdown()
        assert "permanently omitted" in caplog.text
    finally:
        orx.reset_service()


async def test_a_settle_already_in_flight_still_writes_during_shutdown(service, temp_db):
    # `_accepting` closes the door on NEW producers, not on the settles shutdown is waiting for.
    ledger = _ledger()
    await ledger.note_post("100.0", receipt_class="assistant_reply")
    service.track(asyncio.ensure_future(ledger.settle()))
    await service.shutdown()
    assert await _state(temp_db, "100.0") == "finalized"


# --------------------------------------------------------------- callback admission


@pytest.fixture
def post_gate():
    orx.reset_channel_post_gate()
    yield orx.get_channel_post_gate()
    orx.reset_channel_post_gate()


async def test_a_callback_post_is_admitted_while_the_queue_is_open(post_gate):
    async with orx.channel_post_admission("settings_saved") as admitted:
        assert admitted
        assert post_gate.active_count == 1
    assert post_gate.active_count == 0


async def test_shutdown_waits_for_a_callback_that_is_mid_post(post_gate):
    """Socket Mode stays connected until the very end of shutdown, so a Bolt callback can be
    holding a half-finished post while the receipt queue closes underneath it."""
    posted = []
    inside = asyncio.Event()
    release = asyncio.Event()

    async def _callback():
        async with orx.channel_post_admission("settings_saved") as admitted:
            assert admitted
            inside.set()
            await release.wait()
            posted.append("notice")

    task = asyncio.ensure_future(_callback())
    await inside.wait()

    drain = asyncio.ensure_future(post_gate.drain(timeout=5))
    await asyncio.sleep(0)
    assert not drain.done(), "shutdown closed up under a live callback"
    release.set()
    await asyncio.wait_for(drain, timeout=5)

    assert posted == ["notice"]
    assert task.done()


async def test_a_callback_arriving_after_closure_never_posts(post_gate, caplog):
    """Better unsent than unreceipted: the message is refused BEFORE it goes out, because a post
    nobody can account for corrupts the channel's record for as long as it exists."""
    await post_gate.drain()
    posted = []

    with caplog.at_level(logging.WARNING):
        async with orx.channel_post_admission("settings_saved") as admitted:
            if admitted:
                posted.append("notice")

    assert posted == []
    assert "never sent" in caplog.text


async def test_a_callback_that_overruns_the_drain_is_cancelled(post_gate):
    inside = asyncio.Event()
    outcome = []

    async def _wedged():
        async with orx.channel_post_admission("settings_saved") as admitted:
            assert admitted
            inside.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                outcome.append("cancelled")
                raise

    task = asyncio.ensure_future(_wedged())
    await inside.wait()

    await asyncio.wait_for(post_gate.drain(timeout=0.05), timeout=5)

    assert outcome == ["cancelled"]
    assert task.done()


async def test_a_cancel_after_the_post_still_writes_the_receipt(post_gate, service, temp_db):
    """The other direction of the same failure. Admission stops a notice that would land after
    the queue closes — but a callback CANCELLED by the drain can be cut between Slack accepting
    the message and its row being written, which leaves exactly the unaccounted post the gate
    exists to prevent. The post-through-registration pair is not cancellable."""
    from slack_client.event_handlers.message_events import _post_onboarding_notice

    posted = asyncio.Event()
    finish_registration = asyncio.Event()

    client = MagicMock()

    async def _post(**kwargs):
        posted.set()
        return {"ok": True, "ts": "100.0"}

    client.chat_postMessage = _post

    real_record = orx.record_transport_post

    async def _stalled_record(**kwargs):
        # Slack has taken the message; the row is still being written when the drain fires.
        await finish_registration.wait()
        return await real_record(**kwargs)

    async def _callback():
        with patch.object(orx, "record_transport_post", _stalled_record):
            await _post_onboarding_notice(
                SimpleNamespace(self_team_id=TEAM), client, site="settings_reminder",
                receipt_channel=CH, thread_root_ts="99.0",
                channel=CH, thread_ts="99.0", text="configure me")

    task = asyncio.ensure_future(_callback())
    await posted.wait()

    drain = asyncio.ensure_future(post_gate.drain(timeout=0.01))
    await asyncio.sleep(0.05)
    # The callback has been cancelled by now; the protected pair has NOT.
    assert task.cancelled() or task.done()
    finish_registration.set()
    await asyncio.wait_for(drain, timeout=5)

    assert await _state(temp_db, "100.0") == "chrome", \
        "Slack accepted the post and nothing recorded it"


async def test_shutdown_does_not_move_on_until_a_slow_post_has_registered(post_gate, service,
                                                                          temp_db, caplog,
                                                                          monkeypatch):
    """No deadline on a post Slack may already have taken. Abandoning one is the same thing as
    never having shielded it — the message is in the room and nothing accounts for it, and
    everything after this point (the queue closing, the database going away, main's blanket
    task-cancel) assumes these are done."""
    monkeypatch.setattr(orx, "PROTECTED_POST_WATCHDOG_SECONDS", 0.05)
    release = asyncio.Event()

    async def _slow_pair():
        await release.wait()
        await _ledger().note_chrome("100.0", receipt_class="chrome")

    task = asyncio.ensure_future(_slow_pair())
    post_gate.protect(task)

    with caplog.at_level(logging.WARNING):
        drain = asyncio.ensure_future(post_gate.drain(timeout=0.01))
        await asyncio.sleep(0.2)
        # Still waiting, and saying so — not gone.
        assert not drain.done()
        assert not task.done()
        assert "Still waiting on 1 channel post" in caplog.text

        release.set()
        await asyncio.wait_for(drain, timeout=5)

    assert task.done()
    assert await _state(temp_db, "100.0") == "chrome"


async def test_a_registration_that_fails_is_carried_by_the_final_drain(post_gate, service,
                                                                       temp_db):
    """The third leg of the boundedness argument: a registration that cannot reach the database
    does not retry inline — it is retained as a lattice op. The gate drain runs BEFORE the queue
    closes, so that op lands in an open queue and the final drain writes it."""
    calls = []
    real_chrome = temp_db.register_chrome_async

    async def _fail_once(*a, **k):
        calls.append(a)
        if len(calls) == 1:
            raise RuntimeError("db busy")
        return await real_chrome(*a, **k)

    async def _pair():
        with patch.object(temp_db, "register_chrome_async", _fail_once):
            await _ledger().note_chrome("100.0", receipt_class="chrome")

    task = asyncio.ensure_future(_pair())
    post_gate.protect(task)

    await asyncio.wait_for(post_gate.drain(timeout=0.01), timeout=5)
    # The write failed and was retained rather than lost.
    assert service.queue_depth == 1
    assert await _state(temp_db, "100.0") is None

    # …and the receipt shutdown that follows carries it.
    await service.shutdown()
    assert await _state(temp_db, "100.0") == "chrome"
    assert service.queue_depth == 0


async def test_nested_admission_leaves_only_with_its_outermost_frame(post_gate):
    async with orx.channel_post_admission("outer") as outer:
        assert outer
        async with orx.channel_post_admission("inner") as inner:
            assert inner
        assert post_gate.active_count == 1, "the inner frame released the whole callback"
    assert post_gate.active_count == 0


async def test_draining_with_no_callbacks_in_flight_is_a_noop(post_gate):
    await post_gate.drain()
    assert post_gate.admitting is False


async def test_an_onboarding_notice_posts_and_registers_while_admitted(post_gate, service,
                                                                       temp_db):
    from slack_client.event_handlers.message_events import _post_onboarding_notice

    client = MagicMock()
    client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "100.0"})
    host = SimpleNamespace(self_team_id=TEAM)

    resp = await _post_onboarding_notice(
        host, client, site="settings_reminder", receipt_channel=CH, thread_root_ts="99.0",
        channel=CH, thread_ts="99.0", text="configure me")

    assert resp["ts"] == "100.0"
    client.chat_postMessage.assert_awaited_once()
    assert await _state(temp_db, "100.0") == "chrome"


async def test_an_onboarding_notice_is_not_sent_at_all_once_admission_closes(post_gate, service,
                                                                            temp_db):
    from slack_client.event_handlers.message_events import _post_onboarding_notice

    await post_gate.drain()
    client = MagicMock()
    client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "100.0"})

    resp = await _post_onboarding_notice(
        SimpleNamespace(self_team_id=TEAM), client, site="settings_reminder",
        receipt_channel=CH, thread_root_ts="99.0",
        channel=CH, thread_ts="99.0", text="configure me")

    assert resp is None
    client.chat_postMessage.assert_not_awaited()
    assert await _state(temp_db, "100.0") is None


def test_every_receipt_registering_callback_post_sits_under_the_gate():
    """The sweep. Socket Mode outlives the receipt queue, so ANY Bolt callback that posts
    durable channel content has to enter admission first — a new site added without one is
    exactly the hole this round closed, and it would be invisible at runtime."""
    import ast
    import pathlib

    registrars = {"_register_settings_receipt", "_register_raw_receipt"}

    def _called_name(call):
        return (call.func.attr if isinstance(call.func, ast.Attribute)
                else getattr(call.func, "id", None))

    root = pathlib.Path(__file__).resolve().parents[2] / "slack_client" / "event_handlers"
    unguarded = []
    for path in (root / "settings.py", root / "message_events.py"):
        tree = ast.parse(path.read_text())
        # Per CALL SITE, not per function: one guarded site in a handler says nothing about the
        # next one, and the whole point is that a newly added post cannot ride in unnoticed.
        guarded = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.AsyncWith):
                continue
            if not any(isinstance(item.context_expr, ast.Call)
                       and _called_name(item.context_expr) == "channel_post_admission"
                       for item in node.items):
                continue
            for inner in node.body:
                for call in ast.walk(inner):
                    if isinstance(call, ast.Call) and _called_name(call) in registrars:
                        guarded.add((call.lineno, call.col_offset))
        for call in ast.walk(tree):
            if not (isinstance(call, ast.Call) and _called_name(call) in registrars):
                continue
            if (call.lineno, call.col_offset) not in guarded:
                unguarded.append(f"{path.name}:{call.lineno}")
    assert not unguarded, f"receipt-registering posts outside the admission gate: {unguarded}"


async def test_a_producer_arriving_after_shutdown_is_refused_not_queued(temp_db, caplog):
    orx.reset_service()
    svc = orx.install_service(temp_db)
    try:
        await svc.shutdown()
        with caplog.at_level(logging.ERROR):
            await _ledger().note_post("100.0", receipt_class="assistant_reply")
        assert "refused" in caplog.text
        # No row, no queue entry, and no resurrected drain worker behind the final drain.
        assert await temp_db.get_receipt_async(TEAM, CH, "100.0") is None
        assert svc.queue_depth == 0
        assert svc._drain_task is None
    finally:
        orx.reset_service()


async def test_shutdown_cancels_a_settle_that_outran_its_shield(service, temp_db, monkeypatch):
    """A settle only gets SETTLE_TIMEOUT_SECONDS of the shutdown's attention. Left running past
    that, it would come back to a closed queue and a worker that will never run again — its rows
    refused, its messages permanently outside the stream. It is cancelled while the queue is
    still open instead."""
    monkeypatch.setattr(orx, "SETTLE_TIMEOUT_SECONDS", 0.05)
    entered, finished = asyncio.Event(), []

    async def _wedged():
        entered.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            finished.append("cancelled")
            raise

    task = asyncio.ensure_future(_wedged())
    service.track(task)
    await entered.wait()
    await asyncio.wait_for(service.shutdown(), timeout=5)
    assert finished == ["cancelled"]
    assert task.done()
    # …and only after the straggler is off the field does the door close.
    assert not service.accepting


async def test_shutdown_stops_the_drain_worker_before_its_own_final_pass(service, temp_db):
    """Two passes writing one queue at once is a race, and the loser's compensation arrives to
    find the door shut. The worker goes down while the queue is still open; the final drain then
    has the field to itself."""
    order = []
    real_drain = service.drain_once

    async def _tracked():
        order.append(("drain", service._drain_task is not None))
        return await real_drain()

    with patch.object(service, "drain_once", _tracked):
        # A live worker exists (any enqueue starts one).
        await _ledger().note_chrome("100.0", receipt_class="chrome")
        service._enqueue(_Op("finalize", TEAM, CH, "100.0", "s:1"))
        assert service._drain_task is not None
        await service.shutdown()

    # Every drain the shutdown ran saw no worker task alongside it.
    assert order and all(not had_worker for _label, had_worker in order)
    assert service._drain_task is None


async def test_a_compensating_delete_lands_even_during_the_final_drain(service, temp_db):
    """The final drain can revoke itself: a share resolves out from under the pending-share write
    it is committing. That compensation is the service undoing its OWN write, so a closed queue
    must still take it — refusing it would leave the stale row in the database for good."""
    class _Refusing:
        async def record_pending_share_async(self, *a, **k):
            raise RuntimeError("db busy")

    await orx.record_pending_share(_Refusing(), team_id=TEAM, channel_id=CH, file_id="F1",
                                   owner_turn_id="s:1", thread_root_ts="99.0",
                                   receipt_class="artifact")

    real_record = temp_db.record_pending_share_async
    resolved = []

    async def _record_then_resolve(*a, **k):
        out = await real_record(*a, **k)
        if not resolved:
            resolved.append(True)
            # The poll comes back mid-write, during shutdown's own final pass.
            await orx.resolve_pending_share(temp_db, team_id=TEAM, channel_id=CH,
                                            file_id="F1", message_ts="150.0")
        return out

    with patch.object(temp_db, "record_pending_share_async", _record_then_resolve):
        await service.shutdown()

    assert resolved
    # The row the final drain committed is gone again: the compensating delete_share was taken
    # by the closed queue and drained. (The resolver's own finalize is refused here, which is
    # only reachable because this test drives a resolve by hand — shutdown cancels every live
    # resolver before it gets this far.)
    assert await temp_db.get_pending_shares_async() == [], "a stale pending row survived shutdown"
    assert service.queue_depth == 0


async def test_a_resolve_that_lands_mid_drain_undoes_the_stale_pending_row(service, temp_db):
    """The drain is awaiting the pending-share write when the share ts arrives. The resolver
    finalizes the message — the later, authoritative fact — and the row the drain is about to
    commit describes work that is already done. It gets compensated, not left behind for boot
    recovery to retry forever."""
    real_record = temp_db.record_pending_share_async
    in_write = asyncio.Event()
    release = asyncio.Event()

    async def _slow_record(*a, **k):
        in_write.set()
        await release.wait()
        return await real_record(*a, **k)

    class _Refusing:
        async def record_pending_share_async(self, *a, **k):
            raise RuntimeError("db busy")

    await orx.record_pending_share(_Refusing(), team_id=TEAM, channel_id=CH, file_id="F1",
                                   owner_turn_id="s:1", thread_root_ts="99.0",
                                   receipt_class="artifact")
    assert service.queue_depth == 1

    with patch.object(temp_db, "record_pending_share_async", _slow_record):
        drain = asyncio.ensure_future(service.drain_once())
        await in_write.wait()
        # The poll comes back while that write is in flight.
        resolved = await orx.resolve_pending_share(temp_db, team_id=TEAM, channel_id=CH,
                                                   file_id="F1", message_ts="150.0")
        release.set()
        await drain

    assert resolved
    row = await temp_db.get_receipt_async(TEAM, CH, "150.0")
    assert (row["state"], row["turn_id"], row["thread_root_ts"]) == ("finalized", "s:1", "99.0")
    # The compensating deletion is durable: it drains like any other op.
    await service.drain_once()
    assert await temp_db.get_pending_shares_async() == []
    assert service.queue_depth == 0


async def test_a_file_deleted_cleanup_survives_a_transient_read_failure(service, temp_db):
    """Slack sends `file_deleted` exactly once. A read that fails at that moment used to lose
    the cleanup for the life of the database — the row survived, unresolvable, retried and
    logged critically on every boot."""
    await orx.record_pending_share(temp_db, team_id=TEAM, channel_id=CH, file_id="F1",
                                   owner_turn_id="s:1", receipt_class="artifact")

    async def _exploding(*a, **k):
        raise RuntimeError("db busy")

    with patch.object(temp_db, "get_pending_shares_async", _exploding):
        assert await orx.delete_pending_shares_for_file(temp_db, "F1") == 0
    assert service.queue_depth == 1
    assert len(await temp_db.get_pending_shares_async()) == 1

    # The database comes back and the retained op finishes the job.
    assert await service.drain_once() == 1
    assert await temp_db.get_pending_shares_async() == []


async def test_a_still_failing_file_wide_cleanup_stays_queued(service, temp_db):
    await orx.record_pending_share(temp_db, team_id=TEAM, channel_id=CH, file_id="F1",
                                   owner_turn_id="s:1", receipt_class="artifact")

    async def _exploding(*a, **k):
        raise RuntimeError("db busy")

    with patch.object(temp_db, "get_pending_shares_async", _exploding):
        await orx.delete_pending_shares_for_file(temp_db, "F1")
        assert await service.drain_once() == 0
        assert service.queue_depth == 1


async def test_shutdown_cancels_a_share_poll_rather_than_waiting_out_its_budget(service,
                                                                               temp_db):
    started = asyncio.Event()

    class _SlowClient:
        db = temp_db

        async def resolve_file_share_ts(self, channel_id, file_id):
            started.set()
            await asyncio.sleep(60)
            return "150.0"

    await orx.record_pending_share(temp_db, team_id=TEAM, channel_id=CH, file_id="F1",
                                   owner_turn_id="s:1", receipt_class="artifact")
    task = orx.schedule_share_resolution(_SlowClient(), temp_db, team_id=TEAM, channel_id=CH,
                                         file_id="F1")
    await started.wait()
    await asyncio.wait_for(service.shutdown(), timeout=5)
    assert task.cancelled()
    # Cancelled, so the row is exactly where boot recovery expects to find it.
    assert len(await temp_db.get_pending_shares_async()) == 1


# --------------------------------------------------------------- confirmed deletions


async def test_a_confirmed_raw_deletion_drops_its_receipt(service, temp_db):
    await temp_db.register_chrome_async(TEAM, CH, "100.0", orx.sys_owner(), receipt_class="chrome")
    await orx.delete_receipt_for(team_id=TEAM, channel_id=CH, message_ts="100.0",
                                 site="settings_reminder_cleanup")
    assert await temp_db.get_receipt_async(TEAM, CH, "100.0") is None


async def test_a_settings_reminder_takes_its_row_down_with_it(service, temp_db):
    from slack_client.event_handlers.settings import SlackSettingsHandlersMixin

    class _Settings(SlackSettingsHandlersMixin):
        self_team_id = TEAM

        def log_debug(self, *a, **k):
            pass

    await temp_db.register_chrome_async(TEAM, CH, "100.0", orx.sys_owner(), receipt_class="chrome")
    await _Settings()._drop_settings_receipt(CH, "100.0")
    assert await temp_db.get_receipt_async(TEAM, CH, "100.0") is None


async def test_an_aborted_checklist_message_drops_its_row(service, temp_db):
    from message_processor.handlers.image_gen import ImageJobMixin

    class _Job(ImageJobMixin):
        def log_warning(self, *a, **k):
            pass

    class _Client:
        async def delete_message(self, channel_id, ts):
            return True

    ledger = _ledger(owner=orx.job_owner("g1"))
    await ledger.note_chrome("100.0", receipt_class="background_job")
    checklist = SimpleNamespace(surface="message", mirrors_status=False, message_id="100.0")
    await _Job()._abort_checklist(checklist, _Client(), CH, "99.0", receipts=ledger)
    assert await temp_db.get_receipt_async(TEAM, CH, "100.0") is None


async def test_a_failed_delete_keeps_the_row(service, temp_db):
    from message_processor.handlers.image_gen import ImageJobMixin

    class _Job(ImageJobMixin):
        def log_warning(self, *a, **k):
            pass

    class _Client:
        async def delete_message(self, channel_id, ts):
            return False

    ledger = _ledger(owner=orx.job_owner("g1"))
    await ledger.note_chrome("100.0", receipt_class="background_job")
    checklist = SimpleNamespace(surface="message", mirrors_status=False, message_id="100.0")
    await _Job()._abort_checklist(checklist, _Client(), CH, "99.0", receipts=ledger)
    # The message is still in the channel; a row removed now would readmit it as an
    # own-message nobody claims.
    assert await _state(temp_db, "100.0") == "chrome"


async def test_a_failed_image_job_deletes_its_generating_surface_row(service, temp_db):
    from message_processor.handlers.image_gen import ImageJobMixin

    deleted = []

    class _Client:
        self_team_id = TEAM

        async def delete_message(self, channel_id, ts):
            deleted.append(ts)
            return True

        async def handle_error(self, *a, **k):
            return None

    class _TM:
        def finish_generation(self, *a, **k):
            pass

        def mark_upload_finished(self, *a, **k):
            pass

        def mark_needs_refresh(self, *a, **k):
            pass

    class _Job(ImageJobMixin):
        def __init__(self):
            self.thread_manager = _TM()
            self.openai_client = SimpleNamespace(generate_image=AsyncMock(
                side_effect=RuntimeError("no image today")))

        def log_error(self, *a, **k):
            pass

        log_debug = log_info = log_warning = log_error

    await temp_db.register_chrome_async(TEAM, CH, "300.0", orx.job_owner("g1"),
                                        receipt_class="chrome")
    await _Job()._finish_image_generation_background(
        client=_Client(), channel_id=CH, thread_id="99.0", thread_key=f"{CH}:99.0",
        prompt="a cat", enhance=False, conversation_history=[], thread_config={},
        checklist=None, generating_id="300.0", generation_id="g1")
    assert deleted == ["300.0"]
    assert await temp_db.get_receipt_async(TEAM, CH, "300.0") is None


# --------------------------------------------------------------- identity


class _NoIdentityClient:
    self_team_id = None
    bot_user_id = None

    def __init__(self, recovers=False):
        self.recovers = recovers
        self.asked = 0

    async def ensure_receipt_identity(self):
        self.asked += 1
        if self.recovers:
            self.self_team_id, self.bot_user_id = TEAM, "U_BOT"
        return bool(self.recovers)


async def _ready(client, message):
    from main import channel_identity_ready

    return await channel_identity_ready(client, message)


async def test_a_channel_turn_is_refused_when_identity_cannot_be_established(caplog):
    client = _NoIdentityClient()
    with caplog.at_level(logging.ERROR):
        ready = await _ready(client, SimpleNamespace(channel_id=CH))
    assert not ready
    assert client.asked == 1
    assert "Refusing a channel turn" in caplog.text


async def test_a_channel_turn_proceeds_once_identity_is_recovered(caplog):
    client = _NoIdentityClient(recovers=True)
    with caplog.at_level(logging.ERROR):
        assert await _ready(client, SimpleNamespace(channel_id=CH))
    assert caplog.text == ""


async def test_a_dm_never_waits_on_identity():
    client = _NoIdentityClient()
    assert await _ready(client, SimpleNamespace(channel_id=DM))
    assert client.asked == 0


async def test_a_client_that_is_not_the_receipts_transport_is_not_held_to_the_contract():
    assert await _ready(SimpleNamespace(), SimpleNamespace(channel_id=CH))


async def test_the_transport_re_resolves_a_missing_team_id(temp_db):
    class _Auth:
        async def auth_test(self):
            return {"ok": True, "user_id": "U_BOT", "bot_id": "B1", "user": "chatgpt-dev",
                    "team_id": TEAM}

    bot = _messaging(temp_db, _Auth())
    bot.self_team_id = None
    bot.bot_user_id = None
    assert await bot.ensure_receipt_identity()
    assert (bot.self_team_id, bot.bot_user_id) == (TEAM, "U_BOT")


async def test_an_inactive_channel_ledger_is_never_silent(service, caplog):
    class _Client:
        self_team_id = None

    turn = TurnRuntime()
    with caplog.at_level(logging.ERROR):
        turn.bind_receipts(_Client(), SimpleNamespace(channel_id=CH))
    assert "INACTIVE receipt ledger" in caplog.text


# --------------------------------------------------------------- the outer pipeline


def _processor(db, recorder):
    """MessageProcessor with everything past the config read stubbed out, so the test sees
    exactly what base.py asked config for."""
    from message_processor.base import MessageProcessor

    with patch("message_processor.base.AsyncThreadStateManager"), \
         patch("message_processor.base.OpenAIClient"):
        p = MessageProcessor()
    p.db = db
    p.thread_manager.acquire_thread_lock = AsyncMock(return_value=True)
    p.thread_manager.release_thread_lock = AsyncMock()
    p.thread_manager._token_counter.count_thread_tokens = MagicMock(return_value=0)
    p.thread_manager._token_counter.count_message_tokens = MagicMock(return_value=0)

    state = SimpleNamespace(had_timeout=False, messages=[], thread_ts="10.0", channel_id=CH,
                            root_author=("U1", "human"), config_overrides={}, participants={},
                            current_model=None, has_trimmed_messages=False)

    async def _state_for(*a, **k):
        return state

    async def _config(*args, **kwargs):
        recorder.append(kwargs.get("channel_turn"))
        return {"model": "gpt-5.6-sol", "enable_code_interpreter": False}

    p._get_or_rebuild_thread_state = _state_for
    # P2: a channel turn CREATES its state rather than rebuilding one, and renders a pinned
    # stream. Neither is what this file is about — the capability read either side of them is.
    p.get_or_create_channel_thread_state = _state_for
    p._build_channel_turn_stream = AsyncMock(return_value=None)
    p._admit_channel_request = AsyncMock()
    p._process_attachments = AsyncMock(return_value=([], [], []))
    p._handle_text_response = AsyncMock(side_effect=_StopTurn())
    p._config_patch = patch.object(config, "get_thread_config_async", _config)
    return p


class _StopTurn(Exception):
    """Ends the turn at a known point rather than letting a test catch bare Exception."""


async def _run_pipeline(p, channel_id):
    from base_client import Message

    msg = Message(text="hi", user_id="U1", channel_id=channel_id, thread_id="10.0",
                  metadata={"ts": "10.0"})
    client = MagicMock()
    client.send_message = AsyncMock()
    client.self_team_id = TEAM
    with p._config_patch:
        await p.process_message(msg, client, None)


async def test_the_outer_pipeline_resolves_a_channel_turn_channel_first(temp_db):
    seen = []
    await _run_pipeline(_processor(temp_db, seen), CH)
    # Not advisory: this config picks the model the turn trims against and decides whether an
    # attachment is mounted, so a requester-first read here changes the room's machine.
    assert seen and all(seen)


async def test_the_outer_pipeline_leaves_a_dm_on_the_legacy_read(temp_db):
    seen = []
    await _run_pipeline(_processor(temp_db, seen), DM)
    assert seen and not any(seen)


# ============================================================ retention: rows outliving the message
#
# Slack deletes messages in bulk by age and announces none of it — no event, and no
# retention-policy API off Grid. So the sweep infers the boundary from what still answers: one
# `conversations.history` probe per channel per night, and rows are pruned only when that probe
# comes back empty. A one-off manual delete is NOT this sweep's job (the delete event path handles
# it live), and nothing here tries to detect one.


def _retention_web(*, surviving=None, error=None):
    """A Slack client whose history probe answers what retention has left behind."""
    if error is not None:
        return SimpleNamespace(conversations_history=AsyncMock(side_effect=error))
    return SimpleNamespace(conversations_history=AsyncMock(
        return_value={"ok": True, "messages": list(surviving or [])}))


async def _seed_receipts(db, *timestamps):
    for ts in timestamps:
        await db.register_receipt_async(TEAM, CH, ts, "S:1", "finalized", None,
                                        receipt_class=orx.CLASS_ASSISTANT_REPLY)


async def test_receipts_are_pruned_when_slack_no_longer_has_the_messages(temp_db):
    """The probe comes back empty: everything at or below that ts has aged out of Slack, so the
    rows describing those messages are describing nothing."""
    await _seed_receipts(temp_db, "100.000100", "200.000100", "900.000100")
    web = _retention_web(surviving=[])

    pruned = await orx.sweep_receipts_past_retention(temp_db, web)

    assert pruned == 3
    assert await temp_db.get_channel_receipts_async(TEAM, CH) == []


async def test_the_sweep_stops_at_the_first_surviving_message(temp_db):
    """The normal night: retention has not reached our oldest receipt, so ONE call is the whole
    cost and nothing is deleted."""
    await _seed_receipts(temp_db, "100.000100", "200.000100")
    web = _retention_web(surviving=[{"ts": "100.000100", "text": "still here"}])

    pruned = await orx.sweep_receipts_past_retention(temp_db, web)

    assert pruned == 0
    assert web.conversations_history.await_count == 1
    assert len(await temp_db.get_channel_receipts_async(TEAM, CH)) == 2
    kwargs = web.conversations_history.await_args.kwargs
    assert kwargs["channel"] == CH and kwargs["oldest"] == "0" and kwargs["limit"] == 1
    # A hair past the receipt, so the message it describes is inside the probed window.
    assert float(kwargs["latest"]) > 100.000100


async def test_only_the_aged_rows_go_and_the_newer_ones_stay(temp_db):
    """The walk repeats from the new oldest, so a boundary that moved months is caught in one
    night — and stops the moment something survives."""
    await _seed_receipts(temp_db, "100.000100", "200.000100", "900.000100")
    answers = [{"ok": True, "messages": []},
               {"ok": True, "messages": []},
               {"ok": True, "messages": [{"ts": "900.000100", "text": "still here"}]}]
    web = SimpleNamespace(conversations_history=AsyncMock(side_effect=answers))

    pruned = await orx.sweep_receipts_past_retention(temp_db, web)

    assert pruned == 2
    assert [row["message_ts"] for row in await temp_db.get_channel_receipts_async(TEAM, CH)] == [
        "900.000100"]


async def test_a_failing_probe_skips_that_channel_rather_than_the_cleanup(temp_db):
    """This runs inside the nightly cleanup task, next to the database backup. A rate-limited
    channel is worth losing until tomorrow; the rest of the cleanup is not."""
    await _seed_receipts(temp_db, "100.000100")
    web = _retention_web(error=RuntimeError("ratelimited"))

    pruned = await orx.sweep_receipts_past_retention(temp_db, web)

    assert pruned == 0
    assert len(await temp_db.get_channel_receipts_async(TEAM, CH)) == 1, "nothing was deleted"


async def test_hidden_history_is_not_deleted_history(temp_db):
    """A free-plan workspace answers an out-of-window probe with NO messages and `is_limited`. The
    messages are still there — they come back the day the plan changes — so pruning on that would
    throw away receipts for messages that still exist."""
    await _seed_receipts(temp_db, "100.000100", "200.000100")
    web = SimpleNamespace(conversations_history=AsyncMock(
        return_value={"ok": True, "messages": [], "is_limited": True}))

    pruned = await orx.sweep_receipts_past_retention(temp_db, web)

    assert pruned == 0
    assert web.conversations_history.await_count == 1, "and it stops asking about that channel"
    assert len(await temp_db.get_channel_receipts_async(TEAM, CH)) == 2
