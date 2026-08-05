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
    result = await temp_db.register_receipt_async(TEAM, CH, "100.000100", "s1:1", "in_flight",
                                                  receipt_class="assistant_reply")
    assert (result.applied, result.prior_state, result.new_state, result.reason) == \
        (True, "absent", "in_flight", "inserted")
    row = await temp_db.get_receipt_async(TEAM, CH, "100.000100")
    assert row["state"] == "in_flight"
    assert row["turn_id"] == "s1:1"
    assert row["receipt_class"] == "assistant_reply"
    assert row["finalized_ts"] is None


async def test_a_repeat_registration_is_applied_but_unchanged(temp_db):
    """Today's return value is True and stays True — nothing moved, and nothing was refused
    either, so a reader counting refusals must not see this row."""
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight",
                                         receipt_class="assistant_reply")
    result = await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight",
                                                  receipt_class="assistant_reply")
    assert (result.applied, result.prior_state, result.new_state, result.reason) == \
        (True, "in_flight", "in_flight", "unchanged")


async def test_register_finalized_stamps_finalized_ts(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:sys", "finalized",
                                         receipt_class="system_notice")
    row = await temp_db.get_receipt_async(TEAM, CH, "100.0")
    assert row["finalized_ts"] is not None


async def test_same_owner_chrome_promotes_to_in_flight(temp_db):
    await temp_db.register_chrome_async(TEAM, CH, "100.0", "s1:1", receipt_class="chrome")
    result = await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight",
                                                  receipt_class="assistant_reply")
    assert (result.applied, result.prior_state, result.new_state, result.reason) == \
        (True, "chrome", "in_flight", "transitioned")
    assert await _state(temp_db, "100.0") == "in_flight"


async def test_cross_owner_promotion_refused(temp_db):
    await temp_db.register_chrome_async(TEAM, CH, "100.0", "s1:1", receipt_class="chrome")
    result = await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:2", "in_flight",
                                                  receipt_class="assistant_reply")
    assert (result.applied, result.prior_state, result.new_state, result.reason) == \
        (False, "chrome", "chrome", "foreign_owner")
    row = await temp_db.get_receipt_async(TEAM, CH, "100.0")
    assert (row["state"], row["turn_id"]) == ("chrome", "s1:1")


async def test_cross_owner_registration_never_steals_in_flight(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight",
                                         receipt_class="assistant_reply")
    result = await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:9", "in_flight",
                                                  receipt_class="assistant_reply")
    assert (result.applied, result.prior_state, result.reason) == \
        (False, "in_flight", "foreign_owner")
    row = await temp_db.get_receipt_async(TEAM, CH, "100.0")
    assert row["turn_id"] == "s1:1"


async def test_register_chrome_never_demotes_in_flight(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight",
                                         receipt_class="assistant_reply")
    # Class None: class arbitration sits UPSTREAM of the state refusal, and a claimed "chrome"
    # over the assistant_reply row would come back class_conflict — the state reason is the
    # subject here.
    result = await temp_db.register_chrome_async(TEAM, CH, "100.0", "s1:1", receipt_class=None)
    assert (result.applied, result.prior_state, result.new_state, result.reason) == \
        (False, "in_flight", "in_flight", "chrome_over_in_flight")
    assert await _state(temp_db, "100.0") == "in_flight"


async def test_late_registration_absorbed_by_finalized(temp_db):
    await temp_db.finalize_receipts_async(TEAM, CH, [("100.0", None, "assistant_reply")], "s1:1")
    late = await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight",
                                                receipt_class="assistant_reply")
    # Class None for the same reason as the chrome-over-in_flight probe: absorption is the
    # subject, and a conflicting class claim would be refused before the finalized check.
    chrome = await temp_db.register_chrome_async(TEAM, CH, "100.0", "s1:1", receipt_class=None)
    for result in (late, chrome):
        assert (result.applied, result.prior_state, result.new_state, result.reason) == \
            (False, "finalized", "finalized", "absorbed_finalized")
    assert await _state(temp_db, "100.0") == "finalized"


async def test_register_fills_null_root_but_never_clears_one(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight",
                                         receipt_class="assistant_reply")
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight",
                                         thread_root_ts="90.0", receipt_class="assistant_reply")
    assert (await temp_db.get_receipt_async(TEAM, CH, "100.0"))["thread_root_ts"] == "90.0"
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight",
                                         receipt_class="assistant_reply")
    assert (await temp_db.get_receipt_async(TEAM, CH, "100.0"))["thread_root_ts"] == "90.0"


async def test_concurrent_registrations_leave_exactly_one_owner(temp_db):
    results = await asyncio.gather(
        temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight",
                                       receipt_class="assistant_reply"),
        temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:2", "in_flight",
                                       receipt_class="assistant_reply"))
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
        await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "pending",
                                             receipt_class="assistant_reply")


# --------------------------------------------------------------------- transfer / demote

async def test_transfer_only_from_chrome_and_only_from_expected_owner(temp_db):
    await temp_db.register_chrome_async(TEAM, CH, "100.0", "s1:1", receipt_class="chrome")
    refused = await temp_db.transfer_receipt_async(TEAM, CH, "100.0", "s1:OTHER", "s1:2")
    assert (refused.applied, refused.reason) == (False, "not_chrome_or_foreign")
    moved = await temp_db.transfer_receipt_async(TEAM, CH, "100.0", "s1:1", "s1:2")
    # The state never moves in a transfer; only the owner does.
    assert (moved.applied, moved.prior_state, moved.new_state, moved.reason) == \
        (True, "chrome", "chrome", "transferred")
    assert (await temp_db.get_receipt_async(TEAM, CH, "100.0"))["turn_id"] == "s1:2"


async def test_transfer_refused_on_in_flight_and_finalized(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight",
                                         receipt_class="assistant_reply")
    on_in_flight = await temp_db.transfer_receipt_async(TEAM, CH, "100.0", "s1:1", "s1:2")
    await temp_db.finalize_receipts_async(TEAM, CH, [("200.0", None, "assistant_reply")], "s1:1")
    on_finalized = await temp_db.transfer_receipt_async(TEAM, CH, "200.0", "s1:1", "s1:2")
    for result in (on_in_flight, on_finalized):
        assert (result.applied, result.reason) == (False, "not_chrome_or_foreign")


async def test_demote_is_same_owner_in_flight_only(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight",
                                         receipt_class="assistant_reply")
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
    await temp_db.finalize_receipts_async(TEAM, CH, [("100.0", None, "assistant_reply")], "s1:1")
    result = await temp_db.demote_receipt_chrome_async(TEAM, CH, "100.0", "s1:1")
    assert (result.applied, result.reason) == (False, "not_in_flight_or_foreign")
    assert await _state(temp_db, "100.0") == "finalized"


# --------------------------------------------------------------------- finalize

async def test_finalize_unit_covers_every_part(temp_db):
    for ts in ("100.000100", "100.000200", "100.000300"):
        await temp_db.register_receipt_async(TEAM, CH, ts, "s1:1", "in_flight",
                                             receipt_class="assistant_reply")
    results = await temp_db.finalize_receipts_async(
        TEAM, CH, [("100.000100", None, "assistant_reply"),
                   ("100.000200", None, "assistant_reply"),
                   ("100.000300", None, "assistant_reply")], "s1:1")
    assert [(r.applied, r.prior_state, r.new_state, r.reason) for r in results] == \
        [(True, "in_flight", "finalized", "finalized")] * 3
    rows = await temp_db.get_channel_receipts_async(TEAM, CH)
    assert [r["state"] for r in rows] == ["finalized"] * 3


async def test_finalize_results_are_one_per_record_in_input_order(temp_db):
    """Callers zip these two lists, so a dropped or reordered result silently reattributes every
    event after it to the wrong message."""
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight",
                                         receipt_class="assistant_reply")
    await temp_db.register_receipt_async(TEAM, CH, "300.0", "s1:OTHER", "in_flight",
                                         receipt_class="assistant_reply")
    records = [("100.0", None, "assistant_reply"), ("200.0", None, "assistant_reply"),
               ("300.0", None, "assistant_reply"), ("", None, "assistant_reply")]
    results = await temp_db.finalize_receipts_async(TEAM, CH, records, "s1:1")
    assert len(results) == len(records)
    assert [r.reason for r in results] == \
        ["finalized", "inserted", "foreign_owner", "no_message_ts"]
    assert [r.prior_state for r in results] == ["in_flight", "absent", "in_flight", None]


async def test_finalize_inserts_missing_rows_with_their_roots(temp_db):
    results = await temp_db.finalize_receipts_async(
        TEAM, CH, [("100.0", "90.0", "assistant_reply")], "s1:1")
    assert [(r.applied, r.prior_state, r.new_state, r.reason) for r in results] == \
        [(True, "absent", "finalized", "inserted")]
    row = await temp_db.get_receipt_async(TEAM, CH, "100.0")
    assert row["state"] == "finalized"
    assert row["thread_root_ts"] == "90.0"
    assert row["turn_id"] == "s1:1"
    assert row["receipt_class"] == "assistant_reply"


async def test_finalize_never_overwrites_a_known_root_with_null(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight",
                                         thread_root_ts="90.0", receipt_class="assistant_reply")
    await temp_db.finalize_receipts_async(TEAM, CH, [("100.0", None, "assistant_reply")], "s1:1")
    assert (await temp_db.get_receipt_async(TEAM, CH, "100.0"))["thread_root_ts"] == "90.0"


async def test_finalize_leaves_another_turns_row_alone(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:OTHER", "in_flight",
                                         receipt_class="assistant_reply")
    results = await temp_db.finalize_receipts_async(
        TEAM, CH, [("100.0", None, "assistant_reply")], "s1:1")
    assert [(r.applied, r.prior_state, r.new_state, r.reason) for r in results] == \
        [(False, "in_flight", "in_flight", "foreign_owner")]
    row = await temp_db.get_receipt_async(TEAM, CH, "100.0")
    assert (row["state"], row["turn_id"]) == ("in_flight", "s1:OTHER")


async def test_finalize_is_idempotent_and_keeps_first_finalized_ts(temp_db):
    await temp_db.finalize_receipts_async(TEAM, CH, [("100.0", None, "assistant_reply")], "s1:1")
    first = (await temp_db.get_receipt_async(TEAM, CH, "100.0"))["finalized_ts"]
    again = await temp_db.finalize_receipts_async(
        TEAM, CH, [("100.0", None, "assistant_reply")], "s1:1")
    # Still applied — the row IS finalized under this turn — but nothing moved, and the reason is
    # what keeps a replayed finalize out of a count of first-time finalizations.
    assert [(r.applied, r.prior_state, r.reason) for r in again] == \
        [(True, "finalized", "already_finalized")]
    assert (await temp_db.get_receipt_async(TEAM, CH, "100.0"))["finalized_ts"] == first


async def test_finalize_empty_records_is_a_no_op(temp_db):
    assert await temp_db.finalize_receipts_async(TEAM, CH, [], "s1:1") == []


# --------------------------------------------------------------------- reads / scope

async def test_channel_receipts_are_scoped_and_ts_ordered(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "1000.000100", "s1:1", "in_flight",
                                         receipt_class="assistant_reply")
    await temp_db.register_receipt_async(TEAM, CH, "999.999900", "s1:1", "in_flight",
                                         receipt_class="assistant_reply")
    await temp_db.register_receipt_async(TEAM, "C2", "500.0", "s1:1", "in_flight",
                                         receipt_class="assistant_reply")
    await temp_db.register_receipt_async("T2", CH, "500.0", "s1:1", "in_flight",
                                         receipt_class="assistant_reply")

    rows = await temp_db.get_channel_receipts_async(TEAM, CH)
    assert [r["message_ts"] for r in rows] == ["999.999900", "1000.000100"]


async def test_delete_receipt(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s1:1", "in_flight",
                                         receipt_class="assistant_reply")
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
    await temp_db.register_chrome_async(TEAM, CH, "100.0", "s1:1", receipt_class="chrome")
    deleted = await temp_db.delete_receipt_async(TEAM, CH, "100.0")
    assert (deleted.applied, deleted.prior_state, deleted.reason) == (True, "chrome", "deleted")


# --------------------------------------------------------------------- dead-session reconcile

async def test_dead_session_reconcile_spares_the_live_session(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "dead:1", "in_flight",
                                         receipt_class="assistant_reply")
    await temp_db.register_receipt_async(TEAM, CH, "200.0", "live:1", "in_flight",
                                         receipt_class="assistant_reply")
    await temp_db.register_chrome_async(TEAM, CH, "300.0", "dead:2", receipt_class="chrome")

    moved = await temp_db.finalize_dead_session_receipts_async("live")
    # The rows themselves, not a count: one recovered message is the unit the ledger records.
    assert [(r["message_ts"], r["turn_id"]) for r in moved] == [("100.0", "dead:1")]
    assert moved[0]["team_id"] == TEAM and moved[0]["channel_id"] == CH
    assert await _state(temp_db, "100.0") == "finalized"
    assert await _state(temp_db, "200.0") == "in_flight"
    # chrome is permanent exclusion — reconciliation must not promote it.
    assert await _state(temp_db, "300.0") == "chrome"


async def test_dead_session_reconcile_returns_nothing_when_there_is_nothing_to_move(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "200.0", "live:1", "in_flight",
                                         receipt_class="assistant_reply")
    assert await temp_db.finalize_dead_session_receipts_async("live") == []


async def test_dead_session_matching_is_prefix_exact(temp_db):
    """A session whose id merely STARTS with the live one is still dead."""
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "liveX:1", "in_flight",
                                         receipt_class="assistant_reply")
    await temp_db.register_receipt_async(TEAM, CH, "200.0", "live:1", "in_flight",
                                         receipt_class="assistant_reply")
    moved = await temp_db.finalize_dead_session_receipts_async("live")
    assert [r["turn_id"] for r in moved] == ["liveX:1"]
    assert await _state(temp_db, "100.0") == "finalized"
    assert await _state(temp_db, "200.0") == "in_flight"


# --------------------------------------------------------------------- pending shares

async def test_record_pending_share_first_writer_wins(temp_db):
    assert await temp_db.record_pending_share_async(TEAM, CH, "F1", "s1:1", "90.0",
                                                    receipt_class="artifact")
    assert not await temp_db.record_pending_share_async(TEAM, CH, "F1", "s1:2", "80.0",
                                                        receipt_class="artifact")
    rows = await temp_db.get_pending_shares_async()
    assert len(rows) == 1
    assert rows[0]["owner_turn_id"] == "s1:1"
    assert rows[0]["thread_root_ts"] == "90.0"


async def test_resolve_finalizes_and_clears_atomically(temp_db):
    await temp_db.record_pending_share_async(TEAM, CH, "F1", "s1:1", "90.0",
                                             receipt_class="artifact")
    assert await temp_db.resolve_pending_share_async(TEAM, CH, "F1", "100.5")

    row = await temp_db.get_receipt_async(TEAM, CH, "100.5")
    assert row["state"] == "finalized"
    assert row["turn_id"] == "s1:1"
    assert row["thread_root_ts"] == "90.0"
    # The class the pending row carried resolves WITH the share (spec §4).
    assert row["receipt_class"] == "artifact"
    assert await temp_db.get_pending_shares_async() == []


async def test_resolve_is_idempotent_when_the_row_is_already_gone(temp_db):
    assert await temp_db.resolve_pending_share_async(TEAM, CH, "F_NONE", "100.5")
    assert await temp_db.get_receipt_async(TEAM, CH, "100.5") is None


async def test_resolve_finalizes_an_existing_in_flight_row(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "100.5", "s1:1", "in_flight",
                                         receipt_class="artifact")
    await temp_db.record_pending_share_async(TEAM, CH, "F1", "s1:1", "90.0",
                                             receipt_class="artifact")
    await temp_db.resolve_pending_share_async(TEAM, CH, "F1", "100.5")
    row = await temp_db.get_receipt_async(TEAM, CH, "100.5")
    assert row["state"] == "finalized"
    assert row["thread_root_ts"] == "90.0"


async def test_pending_row_survives_a_failed_resolution(temp_db):
    """No resolution call = nothing removed; boot recovery must still see it."""
    await temp_db.record_pending_share_async(TEAM, CH, "F1", "s1:1", None,
                                             receipt_class="artifact")
    assert len(await temp_db.get_pending_shares_async()) == 1
    # Only a Slack-confirmed deletion may drop it.
    assert await temp_db.delete_pending_share_async(TEAM, CH, "F1")
    assert await temp_db.get_pending_shares_async() == []
    assert not await temp_db.delete_pending_share_async(TEAM, CH, "F1")


async def test_pending_shares_scope_filter(temp_db):
    await temp_db.record_pending_share_async(TEAM, CH, "F1", "s1:1", None,
                                             receipt_class="artifact")
    await temp_db.record_pending_share_async("T2", CH, "F2", "s1:1", None,
                                             receipt_class="artifact")
    assert len(await temp_db.get_pending_shares_async()) == 2
    assert [r["file_id"] for r in await temp_db.get_pending_shares_async(TEAM)] == ["F1"]


# =============================== THE RENDER PIN: its subject is CANDIDATES (T47-T49)

async def test_the_render_pin_covers_candidates_not_selections(temp_db):
    """T47. RULING-4's circularity, dissolved. Eligibility needs receipt state and the receipt
    epoch, which live in the pin — so a pin restricted to already-SELECTED ids could never be
    built at all: selection could not precede the pin that selection was said to determine.

    The pin is therefore read over every CANDIDATE, and the selected set is a strict subset."""
    await temp_db.set_meta_if_absent_async(OUTBOUND_RECEIPTS_EPOCH_KEY, "1000.000000")
    candidates = [f"{2000 + i}.000000" for i in range(6)]
    # Half are our own posts with NO finalized receipt — ineligible, and only the pin can say so.
    await temp_db.register_receipt_async(TEAM, CH, candidates[0], "t1", "finalized",
                                         receipt_class="assistant_reply")
    await temp_db.register_receipt_async(TEAM, CH, candidates[1], "t2", "in_flight",
                                         receipt_class="assistant_reply")

    pin = await temp_db.read_channel_sidecars_for_async(TEAM, CH, candidates)

    assert pin["ids"] == sorted(candidates), "the pin's subject is every candidate"
    states = {r["message_ts"]: r["state"] for r in pin["receipts"]}
    assert states == {candidates[0]: "finalized", candidates[1]: "in_flight"}
    # The epoch is what grandfathers a pre-feature message; without it in the pin, eligibility
    # could not be computed for ANY of our own messages.
    assert pin["receipt_feature_epoch_ts"] == "1000.000000"


async def test_the_pin_reads_only_the_candidate_identities(temp_db):
    """T48. Artifact rows outside the candidate set are not returned, the `IN` chunking is
    exercised past SQLite's variable limit, and MULTI-ROW GROUPS survive.

    One message can carry three images — the accessor selects the row `id` and orders by
    `(ts, id)` precisely because of it. A merge that keyed those lists by timestamp would
    collapse each group to one row and silently drop the rest."""
    inside, outside = "5000.000000", "9000.000000"
    for i in range(3):
        temp_db.conn.execute(
            "INSERT INTO images (thread_id, message_ts, url, image_type, analysis) "
            "VALUES (?,?,?,?,?)", (f"{CH}:1.0", inside, f"u{i}", "generated", f"a{i}"))
    for i in range(2):
        temp_db.conn.execute(
            "INSERT INTO documents (thread_id, message_ts, filename, mime_type, summary) "
            "VALUES (?,?,?,?,?)", (f"{CH}:1.0", inside, f"d{i}.pdf", "application/pdf", "s"))
    for i in range(2):
        temp_db.conn.execute(
            "INSERT INTO ambient_artifacts (channel_id, source_ts, conversation_ts, kind, ref, "
            "status) VALUES (?,?,?,?,?,?)", (CH, inside, "1.0", "link", f"r{i}", "ready"))
    temp_db.conn.execute(
        "INSERT INTO images (thread_id, message_ts, url, image_type, analysis) "
        "VALUES (?,?,?,?,?)", (f"{CH}:1.0", outside, "OUTSIDE", "generated", "x"))
    temp_db.conn.commit()

    pin = await temp_db.read_channel_sidecars_for_async(TEAM, CH, [inside])
    assert len(pin["image_analyses"]) == 3
    assert len(pin["document_extractions"]) == 2
    assert len(pin["ambient_artifacts"]) == 2
    assert all(r["message_ts"] == inside for r in pin["image_analyses"])

    # CHUNKED past the variable limit: same rows, same `(ts, id)` order as a single-chunk read.
    padded = [f"8{i}.5" for i in range(600)] + [inside]
    chunked = await temp_db.read_channel_sidecars_for_async(TEAM, CH, padded)
    assert [(r["message_ts"], r["id"]) for r in chunked["image_analyses"]] == [
        (r["message_ts"], r["id"]) for r in pin["image_analyses"]]


async def test_the_merge_recomputes_its_hash_over_the_merged_material(temp_db):
    """T48's merge half. READ 2a and READ 2b over DISJOINT id sets merge into one pin whose
    hash equals what a SINGLE read over the union would have produced — never a hash of the two
    hashes, which would move whenever the periphery/origin split moved even though the rendered
    rows were identical, and would break the cache for nothing."""
    from message_processor.channel_stream import _freeze_sidecars, merge_sidecar_pins

    a, b = "5000.000000", "6000.000000"
    for ts, url in ((a, "peri"), (b, "orig")):
        temp_db.conn.execute(
            "INSERT INTO images (thread_id, message_ts, url, image_type, analysis) "
            "VALUES (?,?,?,?,?)", (f"{CH}:1.0", ts, url, "generated", "x"))
    temp_db.conn.commit()

    shared = _freeze_sidecars(await temp_db.read_channel_sidecars_for_async(TEAM, CH, [a]))
    origin = _freeze_sidecars(await temp_db.read_channel_sidecars_for_async(TEAM, CH, [b]))
    union = await temp_db.read_channel_sidecars_for_async(TEAM, CH, [a, b])

    merged = merge_sidecar_pins(shared, origin, ids=[a, b])
    assert len(merged.image_analyses) == 2
    assert merged.versions_hash == union["versions_hash"]


async def test_a_divergent_receipt_epoch_fails_the_turn_closed(temp_db):
    """T48's mismatch half. `receipt_feature_epoch_ts` is a property of the CHANNEL, not of an id
    list, so two reads in one turn against one database must agree. A difference means something
    is racing the epoch write and the two halves describe two different worlds."""
    from message_processor.channel_stream import (SidecarPinMismatch, _freeze_sidecars,
                                                  merge_sidecar_pins)

    base = await temp_db.read_channel_sidecars_for_async(TEAM, CH, [])
    shared = _freeze_sidecars({**base, "receipt_feature_epoch_ts": "1000.000000"})
    origin = _freeze_sidecars({**base, "receipt_feature_epoch_ts": "2000.000000"})

    with pytest.raises(SidecarPinMismatch):
        merge_sidecar_pins(shared, origin, ids=[])


async def test_the_pinned_anchor_and_inventory_are_what_render(temp_db):
    """T49. Mutate BOTH rows between READ 1 and serialization: the rendered floor and index
    clause are the PINNED values. They are frozen at the moment they are read and never re-read,
    which is what makes two builds of one turn comparable."""
    await temp_db.seed_channel_coverage_async(TEAM, CH, "100.000000")
    await temp_db.advance_channel_window_anchor_async(TEAM, CH, "500.000000", 1)

    pinned = await temp_db.read_channel_window_anchor_async(TEAM, CH)
    assert pinned["anchor"]["floor_ts"] == "500.000000"
    assert pinned["inventory"]["bootstrap_status"] == "pending"

    # The world moves underneath the turn…
    await temp_db.advance_channel_window_anchor_async(TEAM, CH, "900.000000", 1)
    await temp_db.acquire_coverage_sweep_async(TEAM, CH, "tok")
    await temp_db.advance_channel_coverage_async(TEAM, CH, "tok", None, "complete", "genesis")

    # …and the PINNED payload is unchanged, because it was read once.
    assert pinned["anchor"]["floor_ts"] == "500.000000"
    assert pinned["inventory"]["bootstrap_status"] == "pending"
    # A fresh read sees the new world — proving the mutation really landed.
    assert (await temp_db.read_channel_window_anchor_async(
        TEAM, CH))["anchor"]["floor_ts"] == "900.000000"


async def test_the_versions_hash_material_is_the_pinned_grammar(temp_db):
    """#8. The hash covers the epoch, the id list and the rows that RENDER — and nothing else.

    `coverage`, `window` and `activity` left with the window-form accessor: all three were
    discovery, and discovery moved to READ 1. They were inert in this payload, so removing them
    changed no value — but a hash whose material still named fields its contract had removed
    would be one edit away from meaning something nobody intended.

    Asserted against an INDEPENDENT recomputation rather than against the function itself, which
    is the gap codex named: comparing merged output to the same implementation is self-consistent
    without proving the specified material.
    """
    import hashlib
    import json

    payload = await temp_db.read_channel_sidecars_for_async(TEAM, CH, ["1.0", "2.0"])

    expected_material = {
        "epoch": payload.get("receipt_feature_epoch_ts"),
        "ids": list(payload["ids"]),
        "receipts": [[r.get("message_ts"), r.get("state"), r.get("turn_id"),
                      r.get("thread_root_ts"), r.get("receipt_class")]
                     for r in payload["receipts"]],
        "images": [[r.get("message_ts"), r.get("url"), r.get("analysis"),
                    (r.get("metadata") or {}).get("filename")
                    if isinstance(r.get("metadata"), dict) else None]
                   for r in payload["image_analyses"]],
        "documents": [[r.get("message_ts"), r.get("filename"), r.get("file_id"),
                       r.get("summary")] for r in payload["document_extractions"]],
        "ambient": [[r.get("source_ts"), r.get("kind"), r.get("ref"), r.get("status"),
                     r.get("derivation_source"), r.get("title"), r.get("summary")]
                    for r in payload["ambient_artifacts"]],
        "tools": sorted((ts, json.dumps(tools, sort_keys=True))
                        for ts, tools in payload["tool_usage"].items()),
    }
    blob = json.dumps(expected_material, sort_keys=True, separators=(",", ":"), default=str)
    assert payload["versions_hash"] == hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # And the retired members are not merely absent from the payload — feeding them in changes
    # NOTHING, which is what "no longer part of the grammar" has to mean.
    polluted = dict(payload)
    polluted.update({"coverage": {"inventory_start_ts": "5.0"}, "window": ("5.0", True),
                     "activity": [{"root_ts": "9.9", "dirty": 1}]})
    assert temp_db._sidecar_versions_hash(polluted) == payload["versions_hash"]


# --------------------------------------------------- the capability migration (respec §6.1)


def test_the_capability_migration_is_required_critical(tmp_path, monkeypatch):
    """T109. A database where the `ALTER TABLE` fails FAILS STARTUP.

    `_migration_step` logs and carries on, which is right for a migration whose absence announces
    itself. This one's does not: without the three columns every channel silently runs on the
    workspace defaults while the settings modal shows, and accepts, per-channel values that go
    nowhere. So the migration sits OUTSIDE `_migration_step` and its failure propagates.
    """
    import sqlite3

    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    db = DatabaseManager(platform="slack")

    class _RefusingConn:
        """The real connection, with `ADD COLUMN` broken — a disk that will not take the write."""

        def __init__(self, real):
            self._real = real

        def execute(self, sql, *args, **kwargs):
            if "ADD COLUMN" in sql:
                raise sqlite3.OperationalError("disk I/O error")
            return self._real.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._real, name)

    try:
        # Reproduce a pre-W4 database: the table is there, the capability columns are not.
        for column, _ddl in DatabaseManager._CHANNEL_CAPABILITY_DDL:
            db.conn.execute(f"ALTER TABLE channel_settings DROP COLUMN {column}")
        db.conn = _RefusingConn(db.conn)

        # Driven through `init_schema`, which is what startup runs — calling the migration
        # directly would pass even if it were wrapped back up in `_migration_step`, since that
        # wrapper lives at the call site. Which exception reaches the caller is not the claim;
        # that one does is. A swallowed failure here is a bot serving a partial profile.
        with pytest.raises((sqlite3.Error, RuntimeError)):
            db.init_schema()
    finally:
        db.conn.close()


def _unconstrained_channel_settings(db, web_search_check=False, omit=()):
    """A `channel_settings` as an intermediate revision would have left it: the capability
    columns present, without their CHECK constraints.

    `web_search_check` builds the PARTIALLY constrained variant — enable_web_search constrained,
    enable_mcp not. That is the state a detector reading the DDL as one blob calls healthy, since
    the string "enable_mcp" turns up downstream of the first CHECK in its own declaration.

    `omit` leaves capability columns off entirely, which is what makes the migration run its
    ADD COLUMN step — the MIXED shape, where a boot has both columns to add and an existing
    unconstrained one to rebuild for.
    """
    web_search = ("INTEGER CHECK (enable_web_search IS NULL OR enable_web_search IN (0, 1))"
                  if web_search_check else "INTEGER")
    capability = [("enable_web_search", web_search), ("enable_mcp", "INTEGER"),
                  ("image_model", "TEXT")]
    declarations = "".join(f"{name} {spec}, " for name, spec in capability if name not in omit)
    db.conn.execute("DROP TABLE channel_settings")
    db.conn.execute(f"""
        CREATE TABLE channel_settings (
            channel_id TEXT PRIMARY KEY, response_mode TEXT DEFAULT 'tag_only', directives TEXT,
            reply_in_channel BOOLEAN DEFAULT 0, participation_level TEXT, snoozed_until TEXT,
            muted_threads TEXT, model TEXT, reasoning_effort TEXT, verbosity TEXT,
            ambient_memory INTEGER, {declarations}
            updated_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_by TEXT)
    """)


def test_an_unconstrained_legacy_table_is_rebuilt_with_its_checks(tmp_path, monkeypatch):
    """SQLite cannot attach a constraint to an existing column, so a database whose capability
    columns were added without one is REBUILT — otherwise §6.1's claim that the storage layer
    makes `2` unreachable is simply false for those databases, and the ADD COLUMN step skips
    them forever because the names are already there.

    Rows survive the rebuild; a stored value outside {0, 1} normalizes to NULL, which is what the
    resolver already reads it as and the only way the new table's CHECK can accept the copy.
    """
    import sqlite3

    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    db = DatabaseManager(platform="slack")
    try:
        _unconstrained_channel_settings(db)
        db.conn.execute("INSERT INTO channel_settings (channel_id, enable_web_search, enable_mcp, "
                        "model, participation_level) VALUES ('C1', 2, 0, 'gpt-5.5', 'on')")
        db.conn.execute("INSERT INTO channel_settings (channel_id, enable_web_search) "
                        "VALUES ('C2', 1)")
        assert db._channel_settings_missing_checks() == ["enable_web_search", "enable_mcp"]

        db._migrate_channel_capability_columns()

        assert db._channel_settings_missing_checks() == []
        row = db.get_channel_settings("C1")
        assert row["enable_web_search"] is None      # the unusable 2, written down as inherit
        assert (row["enable_mcp"], row["model"]) == (0, "gpt-5.5")
        assert row["participation_level"] == "on"    # untouched columns rode across
        assert db.get_channel_settings("C2")["enable_web_search"] == 1
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute("INSERT INTO channel_settings (channel_id, enable_web_search) "
                            "VALUES ('C9', 2)")
        db._migrate_channel_capability_columns()     # idempotent: nothing left to rebuild
    finally:
        db.conn.close()


def test_a_partially_constrained_table_is_still_rebuilt(tmp_path, monkeypatch):
    """One column constrained and the other not is the state a whole-DDL search misses, because
    the unconstrained column's own declaration sits after the constrained one's CHECK and reads
    as if it were inside it. Detection is per clause, so the rebuild still fires — and BOTH
    columns come out constrained, which is the half the all-or-nothing case never probes.
    """
    import sqlite3

    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    db = DatabaseManager(platform="slack")
    try:
        _unconstrained_channel_settings(db, web_search_check=True)
        db.conn.execute("INSERT INTO channel_settings (channel_id, enable_mcp, verbosity) "
                        "VALUES ('C1', 1, 'high')")
        assert db._channel_settings_missing_checks() == ["enable_mcp"]

        db._migrate_channel_capability_columns()

        assert db._channel_settings_missing_checks() == []
        row = db.get_channel_settings("C1")
        assert (row["enable_mcp"], row["verbosity"]) == (1, "high")
        for column in ("enable_web_search", "enable_mcp"):
            with pytest.raises(sqlite3.IntegrityError):
                db.conn.execute(
                    f"INSERT INTO channel_settings (channel_id, {column}) VALUES (?, 2)",
                    (f"C_{column}",))
    finally:
        db.conn.close()


def test_an_unrecognised_column_fails_startup_with_the_data_intact(tmp_path, monkeypatch):
    """The rebuild copies BY NAME from a canonical column list, so a column that list does not
    know has no safe outcome: copying drops it and its data, continuing serves traffic on an
    unconstrained table. Both are refused — the boot FAILS, loudly, and the table is left exactly
    as it was for an operator to look at.
    """
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    db = DatabaseManager(platform="slack")
    try:
        _unconstrained_channel_settings(db)
        db.conn.execute("ALTER TABLE channel_settings ADD COLUMN someone_elses_column TEXT")
        db.conn.execute("INSERT INTO channel_settings (channel_id, someone_elses_column, model) "
                        "VALUES ('C1', 'precious', 'gpt-5.5')")
        before = db.conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'channel_settings'").fetchone()[0]
        errors = []
        db.log_error = lambda msg, *a, **k: errors.append(msg)

        with pytest.raises(RuntimeError, match="someone_elses_column"):
            db._migrate_channel_capability_columns()

        assert any("someone_elses_column" in msg for msg in errors)
        # Nothing was dropped and nothing was half-migrated: same DDL, same row.
        assert db.conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'channel_settings'").fetchone()[0] == before
        survivor = db.conn.execute(
            "SELECT someone_elses_column, model FROM channel_settings "
            "WHERE channel_id = 'C1'").fetchone()
        assert (survivor["someone_elses_column"], survivor["model"]) == ("precious", "gpt-5.5")
    finally:
        db.conn.close()


def test_an_unknown_column_is_refused_before_any_alter_runs(tmp_path, monkeypatch):
    """The MIXED shape: an unrecognised column, one capability column already present and
    unconstrained, and the other two absent entirely.

    That combination is the one where the refusal can arrive too late. ADD COLUMN is autocommitted
    on this connection, so adding the two missing columns first and only then discovering the
    rebuild cannot proceed would leave `sqlite_master` rewritten — a half-migrated table under a
    failed boot, which is not what "the table is untouched" promises. The unknown-column condition
    is therefore settled BEFORE the first ALTER whenever an existing column will force a rebuild.
    """
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    db = DatabaseManager(platform="slack")
    try:
        _unconstrained_channel_settings(db, omit=("enable_mcp", "image_model"))
        db.conn.execute("ALTER TABLE channel_settings ADD COLUMN someone_elses_column TEXT")
        db.conn.execute("INSERT INTO channel_settings (channel_id, someone_elses_column, "
                        "enable_web_search) VALUES ('C1', 'precious', 1)")
        before = db.conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'channel_settings'").fetchone()[0]
        assert "enable_mcp" not in before          # the ALTER really would have had work to do

        with pytest.raises(RuntimeError, match="someone_elses_column"):
            db._migrate_channel_capability_columns()

        assert db.conn.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'channel_settings'").fetchone()[0] == before
        survivor = db.conn.execute(
            "SELECT someone_elses_column, enable_web_search FROM channel_settings "
            "WHERE channel_id = 'C1'").fetchone()
        assert (survivor["someone_elses_column"], survivor["enable_web_search"]) == ("precious", 1)
    finally:
        db.conn.close()
