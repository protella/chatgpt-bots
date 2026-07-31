"""Outbound receipts + pending shares + bot_meta (single-stream P1, spec §5).

The state machine is the whole point: an own-message enters the channel stream ONLY with a
`finalized` receipt, so every guard here is protecting against either a lost message (a
finalize that didn't land) or a phantom one (a chrome surface that got promoted by accident).

Every write returns a `TransitionResult`, and the `reason` is asserted alongside `applied`
wherever the two refusals being told apart is the point: a bool made an absorbed finalize, a
foreign owner and a refused demotion look identical, and the ledger has to distinguish them.
"""
import asyncio

import pytest

from database import DatabaseManager, OUTBOUND_RECEIPTS_EPOCH_KEY

TEAM = "T1"
CH = "C1"


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    db = DatabaseManager(platform="slack")
    yield db
    db.conn.close()


async def _state(db, ts):
    row = await db.get_receipt_async(TEAM, CH, ts)
    return row["state"] if row else None


# --------------------------------------------------------------------- bot_meta

async def test_set_meta_if_absent_writes_once(temp_db):
    assert await temp_db.set_meta_if_absent_async(OUTBOUND_RECEIPTS_EPOCH_KEY, "1000.500000")
    assert not await temp_db.set_meta_if_absent_async(OUTBOUND_RECEIPTS_EPOCH_KEY, "2000.0")
    assert await temp_db.get_meta_async(OUTBOUND_RECEIPTS_EPOCH_KEY) == "1000.500000"
    assert temp_db.get_meta(OUTBOUND_RECEIPTS_EPOCH_KEY) == "1000.500000"


async def test_set_meta_overwrites_and_missing_key_is_none(temp_db):
    assert await temp_db.get_meta_async("nope") is None
    await temp_db.set_meta_async("k", "v1")
    await temp_db.set_meta_async("k", "v2")
    assert await temp_db.get_meta_async("k") == "v2"


# --------------------------------------------------------------------- registration

async def test_register_creates_in_flight_row(temp_db):
    result = await temp_db.register_receipt_async(TEAM, CH, "100.000100", "s1:1", "in_flight")
    assert (result.applied, result.prior_state, result.new_state, result.reason) == \
        (True, "absent", "in_flight", "inserted")
    row = await temp_db.get_receipt_async(TEAM, CH, "100.000100")
    assert row["state"] == "in_flight"
    assert row["turn_id"] == "s1:1"
    assert row["finalized_ts"] is None


async def test_a_repeat_registration_is_applied_but_unchanged(temp_db):
    """Today's return value is True and stays True — nothing moved, and nothing was refused
    either, so a reader counting refusals must not see this row."""
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight")
    result = await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight")
    assert (result.applied, result.prior_state, result.new_state, result.reason) == \
        (True, "in_flight", "in_flight", "unchanged")


async def test_register_finalized_stamps_finalized_ts(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:sys", "finalized")
    row = await temp_db.get_receipt_async(TEAM, CH, "100.0")
    assert row["finalized_ts"] is not None


async def test_same_owner_chrome_promotes_to_in_flight(temp_db):
    await temp_db.register_chrome_async(TEAM, CH, "100.0", "s1:1")
    result = await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight")
    assert (result.applied, result.prior_state, result.new_state, result.reason) == \
        (True, "chrome", "in_flight", "transitioned")
    assert await _state(temp_db, "100.0") == "in_flight"


async def test_cross_owner_promotion_refused(temp_db):
    await temp_db.register_chrome_async(TEAM, CH, "100.0", "s1:1")
    result = await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:2", "in_flight")
    assert (result.applied, result.prior_state, result.new_state, result.reason) == \
        (False, "chrome", "chrome", "foreign_owner")
    row = await temp_db.get_receipt_async(TEAM, CH, "100.0")
    assert (row["state"], row["turn_id"]) == ("chrome", "s1:1")


async def test_cross_owner_registration_never_steals_in_flight(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight")
    result = await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:9", "in_flight")
    assert (result.applied, result.prior_state, result.reason) == \
        (False, "in_flight", "foreign_owner")
    row = await temp_db.get_receipt_async(TEAM, CH, "100.0")
    assert row["turn_id"] == "s1:1"


async def test_register_chrome_never_demotes_in_flight(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight")
    result = await temp_db.register_chrome_async(TEAM, CH, "100.0", "s1:1")
    assert (result.applied, result.prior_state, result.new_state, result.reason) == \
        (False, "in_flight", "in_flight", "chrome_over_in_flight")
    assert await _state(temp_db, "100.0") == "in_flight"


async def test_late_registration_absorbed_by_finalized(temp_db):
    await temp_db.finalize_receipts_async(TEAM, CH, [("100.0", None)], "s1:1")
    late = await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight")
    chrome = await temp_db.register_chrome_async(TEAM, CH, "100.0", "s1:1")
    for result in (late, chrome):
        assert (result.applied, result.prior_state, result.new_state, result.reason) == \
            (False, "finalized", "finalized", "absorbed_finalized")
    assert await _state(temp_db, "100.0") == "finalized"


async def test_register_fills_null_root_but_never_clears_one(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight")
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight",
                                         thread_root_ts="90.0")
    assert (await temp_db.get_receipt_async(TEAM, CH, "100.0"))["thread_root_ts"] == "90.0"
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight")
    assert (await temp_db.get_receipt_async(TEAM, CH, "100.0"))["thread_root_ts"] == "90.0"


async def test_concurrent_registrations_leave_exactly_one_owner(temp_db):
    results = await asyncio.gather(
        temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight"),
        temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:2", "in_flight"))
    assert sorted(r.applied for r in results) == [False, True]
    winner, loser = ((results[0], results[1]) if results[0].applied
                     else (results[1], results[0]))
    assert winner.reason == "inserted"
    assert (loser.reason, loser.prior_state) == ("foreign_owner", "in_flight")
    row = await temp_db.get_receipt_async(TEAM, CH, "100.0")
    assert row["turn_id"] in ("s1:1", "s1:2")
    assert row["state"] == "in_flight"


async def test_invalid_state_rejected(temp_db):
    with pytest.raises(ValueError):
        await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "pending")


# --------------------------------------------------------------------- transfer / demote

async def test_transfer_only_from_chrome_and_only_from_expected_owner(temp_db):
    await temp_db.register_chrome_async(TEAM, CH, "100.0", "s1:1")
    refused = await temp_db.transfer_receipt_async(TEAM, CH, "100.0", "s1:OTHER", "s1:2")
    assert (refused.applied, refused.reason) == (False, "not_chrome_or_foreign")
    moved = await temp_db.transfer_receipt_async(TEAM, CH, "100.0", "s1:1", "s1:2")
    # The state never moves in a transfer; only the owner does.
    assert (moved.applied, moved.prior_state, moved.new_state, moved.reason) == \
        (True, "chrome", "chrome", "transferred")
    assert (await temp_db.get_receipt_async(TEAM, CH, "100.0"))["turn_id"] == "s1:2"


async def test_transfer_refused_on_in_flight_and_finalized(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight")
    on_in_flight = await temp_db.transfer_receipt_async(TEAM, CH, "100.0", "s1:1", "s1:2")
    await temp_db.finalize_receipts_async(TEAM, CH, [("200.0", None)], "s1:1")
    on_finalized = await temp_db.transfer_receipt_async(TEAM, CH, "200.0", "s1:1", "s1:2")
    for result in (on_in_flight, on_finalized):
        assert (result.applied, result.reason) == (False, "not_chrome_or_foreign")


async def test_demote_is_same_owner_in_flight_only(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight")
    foreign = await temp_db.demote_receipt_chrome_async(TEAM, CH, "100.0", "s1:2")
    assert (foreign.applied, foreign.reason) == (False, "not_in_flight_or_foreign")
    # A refusal claims no prior state: the guarded UPDATE never read one.
    assert foreign.prior_state is None
    demoted = await temp_db.demote_receipt_chrome_async(TEAM, CH, "100.0", "s1:1")
    assert (demoted.applied, demoted.prior_state, demoted.new_state, demoted.reason) == \
        (True, "in_flight", "chrome", "demoted")
    assert await _state(temp_db, "100.0") == "chrome"
    # Already chrome: nothing left to demote.
    again = await temp_db.demote_receipt_chrome_async(TEAM, CH, "100.0", "s1:1")
    assert (again.applied, again.reason) == (False, "not_in_flight_or_foreign")


async def test_demote_refused_on_finalized(temp_db):
    await temp_db.finalize_receipts_async(TEAM, CH, [("100.0", None)], "s1:1")
    result = await temp_db.demote_receipt_chrome_async(TEAM, CH, "100.0", "s1:1")
    assert (result.applied, result.reason) == (False, "not_in_flight_or_foreign")
    assert await _state(temp_db, "100.0") == "finalized"


# --------------------------------------------------------------------- finalize

async def test_finalize_unit_covers_every_part(temp_db):
    for ts in ("100.000100", "100.000200", "100.000300"):
        await temp_db.register_receipt_async(TEAM, CH, ts, "s1:1", "in_flight")
    results = await temp_db.finalize_receipts_async(
        TEAM, CH, [("100.000100", None), ("100.000200", None), ("100.000300", None)], "s1:1")
    assert [(r.applied, r.prior_state, r.new_state, r.reason) for r in results] == \
        [(True, "in_flight", "finalized", "finalized")] * 3
    rows = await temp_db.get_channel_receipts_async(TEAM, CH)
    assert [r["state"] for r in rows] == ["finalized"] * 3


async def test_finalize_results_are_one_per_record_in_input_order(temp_db):
    """Callers zip these two lists, so a dropped or reordered result silently reattributes every
    event after it to the wrong message."""
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight")
    await temp_db.register_receipt_async(TEAM, CH, "300.0", "s1:OTHER", "in_flight")
    records = [("100.0", None), ("200.0", None), ("300.0", None), ("", None)]
    results = await temp_db.finalize_receipts_async(TEAM, CH, records, "s1:1")
    assert len(results) == len(records)
    assert [r.reason for r in results] == \
        ["finalized", "inserted", "foreign_owner", "no_message_ts"]
    assert [r.prior_state for r in results] == ["in_flight", "absent", "in_flight", None]


async def test_finalize_inserts_missing_rows_with_their_roots(temp_db):
    results = await temp_db.finalize_receipts_async(TEAM, CH, [("100.0", "90.0")], "s1:1")
    assert [(r.applied, r.prior_state, r.new_state, r.reason) for r in results] == \
        [(True, "absent", "finalized", "inserted")]
    row = await temp_db.get_receipt_async(TEAM, CH, "100.0")
    assert row["state"] == "finalized"
    assert row["thread_root_ts"] == "90.0"
    assert row["turn_id"] == "s1:1"


async def test_finalize_never_overwrites_a_known_root_with_null(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight",
                                         thread_root_ts="90.0")
    await temp_db.finalize_receipts_async(TEAM, CH, [("100.0", None)], "s1:1")
    assert (await temp_db.get_receipt_async(TEAM, CH, "100.0"))["thread_root_ts"] == "90.0"


async def test_finalize_leaves_another_turns_row_alone(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:OTHER", "in_flight")
    results = await temp_db.finalize_receipts_async(TEAM, CH, [("100.0", None)], "s1:1")
    assert [(r.applied, r.prior_state, r.new_state, r.reason) for r in results] == \
        [(False, "in_flight", "in_flight", "foreign_owner")]
    row = await temp_db.get_receipt_async(TEAM, CH, "100.0")
    assert (row["state"], row["turn_id"]) == ("in_flight", "s1:OTHER")


async def test_finalize_is_idempotent_and_keeps_first_finalized_ts(temp_db):
    await temp_db.finalize_receipts_async(TEAM, CH, [("100.0", None)], "s1:1")
    first = (await temp_db.get_receipt_async(TEAM, CH, "100.0"))["finalized_ts"]
    again = await temp_db.finalize_receipts_async(TEAM, CH, [("100.0", None)], "s1:1")
    # Still applied — the row IS finalized under this turn — but nothing moved, and the reason is
    # what keeps a replayed finalize out of a count of first-time finalizations.
    assert [(r.applied, r.prior_state, r.reason) for r in again] == \
        [(True, "finalized", "already_finalized")]
    assert (await temp_db.get_receipt_async(TEAM, CH, "100.0"))["finalized_ts"] == first


async def test_finalize_empty_records_is_a_no_op(temp_db):
    assert await temp_db.finalize_receipts_async(TEAM, CH, [], "s1:1") == []


# --------------------------------------------------------------------- reads / scope

async def test_channel_receipts_are_scoped_and_ts_ordered(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "1000.000100", "s1:1", "in_flight")
    await temp_db.register_receipt_async(TEAM, CH, "999.999900", "s1:1", "in_flight")
    await temp_db.register_receipt_async(TEAM, "C2", "500.0", "s1:1", "in_flight")
    await temp_db.register_receipt_async("T2", CH, "500.0", "s1:1", "in_flight")

    rows = await temp_db.get_channel_receipts_async(TEAM, CH)
    assert [r["message_ts"] for r in rows] == ["999.999900", "1000.000100"]


async def test_delete_receipt(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight")
    deleted = await temp_db.delete_receipt_async(TEAM, CH, "100.0")
    # The state it removed, read in the delete's own transaction — an abandoned in_flight surface
    # and a deleted chrome placeholder are different facts about the room.
    assert (deleted.applied, deleted.prior_state, deleted.new_state, deleted.reason) == \
        (True, "in_flight", "absent", "deleted")
    assert await temp_db.get_receipt_async(TEAM, CH, "100.0") is None
    missing = await temp_db.delete_receipt_async(TEAM, CH, "100.0")
    assert (missing.applied, missing.prior_state, missing.new_state, missing.reason) == \
        (False, "absent", "absent", "no_row")


async def test_delete_reports_the_chrome_it_removed(temp_db):
    await temp_db.register_chrome_async(TEAM, CH, "100.0", "s1:1")
    deleted = await temp_db.delete_receipt_async(TEAM, CH, "100.0")
    assert (deleted.applied, deleted.prior_state, deleted.reason) == (True, "chrome", "deleted")


# --------------------------------------------------------------------- dead-session reconcile

async def test_dead_session_reconcile_spares_the_live_session(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "dead:1", "in_flight")
    await temp_db.register_receipt_async(TEAM, CH, "200.0", "live:1", "in_flight")
    await temp_db.register_chrome_async(TEAM, CH, "300.0", "dead:2")

    moved = await temp_db.finalize_dead_session_receipts_async("live")
    # The rows themselves, not a count: one recovered message is the unit the ledger records.
    assert [(r["message_ts"], r["turn_id"]) for r in moved] == [("100.0", "dead:1")]
    assert moved[0]["team_id"] == TEAM and moved[0]["channel_id"] == CH
    assert await _state(temp_db, "100.0") == "finalized"
    assert await _state(temp_db, "200.0") == "in_flight"
    # chrome is permanent exclusion — reconciliation must not promote it.
    assert await _state(temp_db, "300.0") == "chrome"


async def test_dead_session_reconcile_returns_nothing_when_there_is_nothing_to_move(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "200.0", "live:1", "in_flight")
    assert await temp_db.finalize_dead_session_receipts_async("live") == []


async def test_dead_session_matching_is_prefix_exact(temp_db):
    """A session whose id merely STARTS with the live one is still dead."""
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "liveX:1", "in_flight")
    await temp_db.register_receipt_async(TEAM, CH, "200.0", "live:1", "in_flight")
    moved = await temp_db.finalize_dead_session_receipts_async("live")
    assert [r["turn_id"] for r in moved] == ["liveX:1"]
    assert await _state(temp_db, "100.0") == "finalized"
    assert await _state(temp_db, "200.0") == "in_flight"


# --------------------------------------------------------------------- pending shares

async def test_record_pending_share_first_writer_wins(temp_db):
    assert await temp_db.record_pending_share_async(TEAM, CH, "F1", "s1:1", "90.0")
    assert not await temp_db.record_pending_share_async(TEAM, CH, "F1", "s1:2", "80.0")
    rows = await temp_db.get_pending_shares_async()
    assert len(rows) == 1
    assert rows[0]["owner_turn_id"] == "s1:1"
    assert rows[0]["thread_root_ts"] == "90.0"


async def test_resolve_finalizes_and_clears_atomically(temp_db):
    await temp_db.record_pending_share_async(TEAM, CH, "F1", "s1:1", "90.0")
    assert await temp_db.resolve_pending_share_async(TEAM, CH, "F1", "100.5")

    row = await temp_db.get_receipt_async(TEAM, CH, "100.5")
    assert row["state"] == "finalized"
    assert row["turn_id"] == "s1:1"
    assert row["thread_root_ts"] == "90.0"
    assert await temp_db.get_pending_shares_async() == []


async def test_resolve_is_idempotent_when_the_row_is_already_gone(temp_db):
    assert await temp_db.resolve_pending_share_async(TEAM, CH, "F_NONE", "100.5")
    assert await temp_db.get_receipt_async(TEAM, CH, "100.5") is None


async def test_resolve_finalizes_an_existing_in_flight_row(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "100.5", "s1:1", "in_flight")
    await temp_db.record_pending_share_async(TEAM, CH, "F1", "s1:1", "90.0")
    await temp_db.resolve_pending_share_async(TEAM, CH, "F1", "100.5")
    row = await temp_db.get_receipt_async(TEAM, CH, "100.5")
    assert row["state"] == "finalized"
    assert row["thread_root_ts"] == "90.0"


async def test_pending_row_survives_a_failed_resolution(temp_db):
    """No resolution call = nothing removed; boot recovery must still see it."""
    await temp_db.record_pending_share_async(TEAM, CH, "F1", "s1:1", None)
    assert len(await temp_db.get_pending_shares_async()) == 1
    # Only a Slack-confirmed deletion may drop it.
    assert await temp_db.delete_pending_share_async(TEAM, CH, "F1")
    assert await temp_db.get_pending_shares_async() == []
    assert not await temp_db.delete_pending_share_async(TEAM, CH, "F1")


async def test_pending_shares_scope_filter(temp_db):
    await temp_db.record_pending_share_async(TEAM, CH, "F1", "s1:1", None)
    await temp_db.record_pending_share_async("T2", CH, "F2", "s1:1", None)
    assert len(await temp_db.get_pending_shares_async()) == 2
    assert [r["file_id"] for r in await temp_db.get_pending_shares_async(TEAM)] == ["F1"]
