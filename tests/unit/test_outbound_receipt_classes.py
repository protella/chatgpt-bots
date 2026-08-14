"""Receipt classes (EDIT_OWN_MESSAGE spec §4).

What is pinned here, and nowhere else:
  * every producer stamps a class EXPLICITLY — the ledger API takes it as a required keyword
    with no default, and an AST walk over the producer modules proves no posting site slips a
    `receipts=` through without saying what it posted;
  * placeholder promotion atomically maps chrome → assistant_reply and retry demotion maps
    back; any OTHER same-message class conflict fails closed and logs;
  * the class round-trips the database — registration, settle, the batched sidecar read, and
    the pending-share path end-to-end (record → resolve, the queued-op short-circuit, and boot
    recovery);
  * legacy rows carry receipt_class IS NULL and are ineligible for anything class-gated —
    `receipt_has_class` never infers from state, owner or anything else.

No mock streams are used anywhere in this file (CLAUDE.md pitfall 6 — nothing here iterates a
stream at all; every async helper terminates by construction).
"""
import ast
import sqlite3
from pathlib import Path

import pytest

from database import DatabaseManager, _RECEIPT_CLASSES
from message_processor import outbound_receipts as orx
from message_processor.outbound_receipts import (RECEIPT_CLASSES, ReceiptLedger, _Op,
                                                 receipt_has_class, record_transport_post)

TEAM = "T1"
CH = "C0BKX77NU66"
REPO = Path(__file__).resolve().parents[2]


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


def _ledger(owner="s:1", channel=CH):
    return ReceiptLedger(owner, TEAM, channel, barrier_eligible=False)


async def _row(db, ts, channel=CH):
    return await db.get_receipt_async(TEAM, channel, ts)


# ------------------------------------------------------------------ the closed enum


def test_the_enum_is_closed_and_shared_with_the_db_layer():
    assert RECEIPT_CLASSES == ("assistant_reply", "correction_announcement", "system_notice",
                               "background_job", "artifact", "chrome")
    assert tuple(_RECEIPT_CLASSES) == RECEIPT_CLASSES


async def test_db_layer_rejects_an_unknown_class(temp_db):
    with pytest.raises(ValueError):
        await temp_db.register_receipt_async(TEAM, CH, "1.0", "s:1", "in_flight",
                                             receipt_class="reply")
    with pytest.raises(ValueError):
        await temp_db.finalize_receipts_async(TEAM, CH, [("1.0", None, "bogus")], "s:1")
    with pytest.raises(ValueError):
        await temp_db.record_pending_share_async(TEAM, CH, "F1", "s:1",
                                                 receipt_class="bogus")


# ------------------------------------------------------------------ required keyword, no default


async def test_note_post_requires_the_class_keyword(service):
    with pytest.raises(TypeError):
        await _ledger().note_post("100.0")  # noqa — the missing keyword IS the test


async def test_note_chrome_requires_the_class_keyword(service):
    with pytest.raises(TypeError):
        await _ledger().note_chrome("100.0")  # noqa


async def test_record_transport_post_requires_the_class_keyword(service):
    with pytest.raises(TypeError):
        await record_transport_post(team_id=TEAM, channel_id=CH, message_ts="100.0",
                                    receipts=None, receipt_kind="finalized")


async def test_db_registration_requires_the_class_keyword(temp_db):
    with pytest.raises(TypeError):
        await temp_db.register_receipt_async(TEAM, CH, "1.0", "s:1", "in_flight")


# ------------------------------------------------------------------ stamping round-trips


async def test_note_post_stamps_the_class_through_register_and_settle(service, temp_db):
    ledger = _ledger()
    await ledger.note_post("100.0", thread_root_ts="99.0",
                           receipt_class="assistant_reply")
    row = await _row(temp_db, "100.0")
    assert row["state"] == "in_flight"
    assert row["receipt_class"] == "assistant_reply"
    await ledger.settle()
    row = await _row(temp_db, "100.0")
    assert row["state"] == "finalized"
    assert row["receipt_class"] == "assistant_reply"


async def test_finalized_kind_inserts_with_its_class(service, temp_db):
    await _ledger().note_post("100.0", orx.STATE_FINALIZED, "99.0",
                              receipt_class="system_notice")
    row = await _row(temp_db, "100.0")
    assert row["state"] == "finalized"
    assert row["receipt_class"] == "system_notice"


async def test_chrome_state_and_chrome_class_are_orthogonal(service, temp_db):
    """A research status card is chrome-STATE with class background_job."""
    await _ledger().note_chrome("100.0", receipt_class="background_job")
    row = await _row(temp_db, "100.0")
    assert row["state"] == "chrome"
    assert row["receipt_class"] == "background_job"


async def test_an_invalid_class_is_recorded_null_and_logged(service, temp_db, caplog):
    with caplog.at_level("ERROR", logger="slack_bot.OutboundReceipts"):
        await _ledger().note_post("100.0", receipt_class="tyop")
    row = await _row(temp_db, "100.0")
    assert row is not None, "the delivery record must never be lost to a bad class"
    assert row["receipt_class"] is None
    assert any("not a valid class" in r.message for r in caplog.records)
    assert not receipt_has_class(row, "assistant_reply")


async def test_a_queued_registration_retains_its_class_through_the_lattice(
        service, temp_db, monkeypatch):
    """A DB blip queues the write; the drain must land the row WITH the producer's class."""
    original = temp_db.register_receipt_async
    calls = {"n": 0}

    async def _flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return await original(*args, **kwargs)

    monkeypatch.setattr(temp_db, "register_receipt_async", _flaky)
    await _ledger().note_post("100.0", receipt_class="assistant_reply")
    assert await _row(temp_db, "100.0") is None
    assert await service.drain_once() == 1
    row = await _row(temp_db, "100.0")
    assert row["receipt_class"] == "assistant_reply"


async def test_settle_finalizes_a_lost_registration_with_its_class(service, temp_db,
                                                                   monkeypatch):
    """The registration never reached the DB at all; the settle's insert carries the class."""

    async def _down(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    ledger = _ledger()
    monkeypatch.setattr(temp_db, "register_receipt_async", _down)
    await ledger.note_post("100.0", receipt_class="assistant_reply")
    # The register is queued; settle defers the finalize behind it, and the drain carries
    # both in order — register first (rank), then the queued finalize absorbs it.
    monkeypatch.undo()
    await ledger.settle()
    await service.drain_once()
    row = await _row(temp_db, "100.0")
    assert row["state"] == "finalized"
    assert row["receipt_class"] == "assistant_reply"


async def test_transport_lifecycle_posts_carry_their_class(service, temp_db):
    """record_transport_post with no ledger: sys-owner rows still say what they are."""
    await record_transport_post(team_id=TEAM, channel_id=CH, message_ts="100.0",
                                receipts=None, receipt_kind="finalized",
                                receipt_class="system_notice", site="settings_confirmation")
    await record_transport_post(team_id=TEAM, channel_id=CH, message_ts="101.0",
                                receipts=None, receipt_kind="chrome",
                                receipt_class="chrome", site="onboarding")
    assert (await _row(temp_db, "100.0"))["receipt_class"] == "system_notice"
    chrome = await _row(temp_db, "101.0")
    assert chrome["state"] == "chrome"
    assert chrome["receipt_class"] == "chrome"


# ------------------------------------------------------------------ promotion / demotion


async def test_placeholder_promotion_maps_chrome_to_assistant_reply(service, temp_db):
    ledger = _ledger()
    await ledger.note_chrome("100.0", receipt_class="chrome")
    await ledger.promote("100.0", thread_root_ts="99.0")
    row = await _row(temp_db, "100.0")
    assert row["state"] == "in_flight"
    assert row["receipt_class"] == "assistant_reply"
    await ledger.settle()
    row = await _row(temp_db, "100.0")
    assert row["state"] == "finalized"
    assert row["receipt_class"] == "assistant_reply"


async def test_retry_demotion_maps_the_class_back_to_chrome(service, temp_db):
    ledger = _ledger()
    await ledger.note_chrome("100.0", receipt_class="chrome")
    await ledger.promote("100.0")
    await ledger.demote("100.0")
    row = await _row(temp_db, "100.0")
    assert row["state"] == "chrome"
    assert row["receipt_class"] == "chrome"
    # And the demoted surface no longer settles as conversation.
    assert await ledger.settle() == 0


async def test_demoting_a_legacy_null_row_never_stamps_a_class(temp_db):
    """Demotion maps BACK; it is not a first stamping. A legacy row stays NULL."""
    temp_db.conn.execute(
        "INSERT INTO outbound_receipts (team_id, channel_id, message_ts, turn_id, state) "
        "VALUES (?, ?, ?, ?, 'in_flight')", (TEAM, CH, "100.0", "s:1"))
    result = await temp_db.demote_receipt_chrome_async(TEAM, CH, "100.0", "s:1")
    assert result.applied
    row = await _row(temp_db, "100.0")
    assert row["state"] == "chrome"
    assert row["receipt_class"] is None


async def test_promotion_of_a_legacy_null_placeholder_keeps_null(temp_db):
    """§11.2: NULL is IMMUTABLE — promotion stamps assistant_reply ONLY over a stored
    `chrome` class. A legacy NULL placeholder still promotes its STATE, but the class stays
    NULL: a legacy row can never become editable."""
    temp_db.conn.execute(
        "INSERT INTO outbound_receipts (team_id, channel_id, message_ts, turn_id, state) "
        "VALUES (?, ?, ?, ?, 'chrome')", (TEAM, CH, "100.0", "s:1"))
    result = await temp_db.register_receipt_async(TEAM, CH, "100.0", "s:1", "in_flight",
                                                  receipt_class="assistant_reply")
    assert result.applied
    row = await _row(temp_db, "100.0")
    assert row["state"] == "in_flight"
    assert row["receipt_class"] is None


# ------------------------------------------------------------------ conflicts fail closed


async def test_a_register_class_conflict_fails_closed_to_null_and_logs(temp_db, caplog):
    """§11.1: ANY class conflict leaves the row's class NULL — the ineligible terminal
    state — with an ERROR naming both classes; the registration itself stays refused."""
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s:1", "in_flight",
                                         receipt_class="system_notice")
    with caplog.at_level("ERROR"):
        result = await temp_db.register_receipt_async(
            TEAM, CH, "100.0", "s:1", "in_flight", "99.0",
            receipt_class="assistant_reply")
    assert not result.applied
    assert result.reason == "class_conflict"
    row = await _row(temp_db, "100.0")
    assert row["receipt_class"] is None, "nothing editable can come out of a conflict"
    # The refused registration writes ONLY the terminal class — not even the root fill.
    assert row["thread_root_ts"] is None
    assert any("class conflict" in r.getMessage() and "'system_notice'" in r.getMessage()
               and "'assistant_reply'" in r.getMessage() for r in caplog.records)


async def test_a_chrome_to_chrome_class_conflict_also_nulls(temp_db):
    """The promotion mapping is the ONLY sanctioned class change; chrome→background_job on
    the same message is a producer bug — refused, and the class fails closed to NULL."""
    await temp_db.register_chrome_async(TEAM, CH, "100.0", "s:1",
                                        receipt_class="chrome")
    result = await temp_db.register_chrome_async(TEAM, CH, "100.0", "s:1",
                                                 receipt_class="background_job")
    assert not result.applied
    assert result.reason == "class_conflict"
    assert (await _row(temp_db, "100.0"))["receipt_class"] is None


async def test_a_finalize_class_conflict_fails_closed_to_null_for_that_record(temp_db,
                                                                              caplog):
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s:1", "in_flight",
                                         receipt_class="system_notice")
    with caplog.at_level("ERROR"):
        results = await temp_db.finalize_receipts_async(
            TEAM, CH, [("100.0", None, "assistant_reply")], "s:1")
    assert len(results) == 1
    assert not results[0].applied
    assert results[0].reason == "class_conflict"
    row = await _row(temp_db, "100.0")
    assert row["state"] == "in_flight"          # no state change — fail closed
    assert row["receipt_class"] is None         # §11.1: the class lands terminal NULL
    assert any("class conflict" in r.getMessage() for r in caplog.records)


async def test_a_foreign_owner_finalize_conflict_still_nulls_the_class(temp_db, caplog):
    """§11.12/§11.18 regression: class arbitration is ORTHOGONAL to the ownership refusal.
    A finalize by ANOTHER turn claiming a different class is still refused as
    `foreign_owner` (no state change), but the row's class fails closed to NULL all the
    same — ownership must not shield an editable class from a proven disagreement."""
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s:1", "in_flight",
                                         receipt_class="assistant_reply")
    with caplog.at_level("ERROR"):
        results = await temp_db.finalize_receipts_async(
            TEAM, CH, [("100.0", None, "system_notice")], "s:2")
    assert len(results) == 1
    assert not results[0].applied
    assert results[0].reason == "foreign_owner"
    row = await _row(temp_db, "100.0")
    assert row["state"] == "in_flight"          # the ownership refusal stands
    assert row["turn_id"] == "s:1"
    assert row["receipt_class"] is None         # …and the conflicted class is terminal NULL
    assert any("class conflict" in r.getMessage() and "'assistant_reply'" in r.getMessage()
               and "'system_notice'" in r.getMessage() for r in caplog.records)


async def test_pending_resolution_never_stamps_an_existing_null_row(temp_db):
    """§11.12/§11.18 regression (legacy-NULL pending resolution): resolving a share onto an
    EXISTING row whose class is NULL — legacy or conflict-poisoned alike — must NOT refill
    it with the pending row's class. The state resolution lands; NULL stays NULL."""
    temp_db.conn.execute(
        "INSERT INTO outbound_receipts (team_id, channel_id, message_ts, turn_id, state) "
        "VALUES (?, ?, ?, ?, 'in_flight')", (TEAM, CH, "100.0", "s:1"))
    await temp_db.record_pending_share_async(TEAM, CH, "F1", "s:1",
                                             receipt_class="artifact")
    assert await temp_db.resolve_pending_share_async(TEAM, CH, "F1", "100.0")
    row = await _row(temp_db, "100.0")
    assert row["state"] == "finalized"
    assert row["receipt_class"] is None, "NULL is immutable — the resolve never stamps it"
    assert await temp_db.get_pending_shares_async() == []


async def test_a_null_class_is_never_filled_at_finalize(temp_db):
    """§11.2: NULL is immutable — no fill by a same-owner claim. The finalize itself still
    lands; the legacy row just stays ineligible forever."""
    temp_db.conn.execute(
        "INSERT INTO outbound_receipts (team_id, channel_id, message_ts, turn_id, state) "
        "VALUES (?, ?, ?, ?, 'in_flight')", (TEAM, CH, "100.0", "s:1"))
    result = await temp_db.finalize_receipts_async(
        TEAM, CH, [("100.0", None, "assistant_reply")], "s:1")
    assert result[0].applied
    row = await _row(temp_db, "100.0")
    assert row["state"] == "finalized"
    assert row["receipt_class"] is None


async def test_a_null_class_is_never_filled_at_register(temp_db):
    """§11.2's register half: a later same-owner claim over a NULL row changes nothing."""
    temp_db.conn.execute(
        "INSERT INTO outbound_receipts (team_id, channel_id, message_ts, turn_id, state) "
        "VALUES (?, ?, ?, ?, 'in_flight')", (TEAM, CH, "100.0", "s:1"))
    result = await temp_db.register_receipt_async(TEAM, CH, "100.0", "s:1", "in_flight",
                                                  receipt_class="assistant_reply")
    assert result.applied          # the (no-op) state registration is not refused…
    assert (await _row(temp_db, "100.0"))["receipt_class"] is None   # …but NULL stays NULL


async def test_a_duplicate_note_post_class_conflict_nulls_the_row(service, temp_db, caplog):
    """§11.1: a second in-flight claim naming a DIFFERENT class is a producer bug — the class
    fails closed to NULL in the ledger AND in the row, and the settle cannot re-stamp it."""
    ledger = _ledger()
    await ledger.note_post("100.0", receipt_class="assistant_reply")
    with caplog.at_level("ERROR", logger="slack_bot.OutboundReceipts"):
        await ledger.note_post("100.0", receipt_class="system_notice")
    assert any("class conflict" in r.message for r in caplog.records)
    assert (await _row(temp_db, "100.0"))["receipt_class"] is None
    await ledger.settle()
    row = await _row(temp_db, "100.0")
    assert row["state"] == "finalized"
    assert row["receipt_class"] is None


async def test_a_second_conflicting_pending_claim_logs_and_the_first_stands(temp_db, caplog):
    """§11.1's pending-share half: first writer owns the file; a second claim naming a
    DIFFERENT class is detected and ERROR-logged, and the first claim stands untouched."""
    assert await temp_db.record_pending_share_async(TEAM, CH, "F1", "s:1",
                                                    receipt_class="artifact")
    with caplog.at_level("ERROR"):
        assert not await temp_db.record_pending_share_async(TEAM, CH, "F1", "s:2",
                                                            receipt_class="background_job")
    rows = await temp_db.get_pending_shares_async()
    assert len(rows) == 1
    assert rows[0]["owner_turn_id"] == "s:1"
    assert rows[0]["receipt_class"] == "artifact"
    assert any("class conflict" in r.getMessage() and "'artifact'" in r.getMessage()
               and "'background_job'" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------ DB round-trip: reads


async def test_the_batched_sidecar_read_carries_the_class(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s:1", "finalized",
                                         receipt_class="assistant_reply")
    await temp_db.register_receipt_async(TEAM, CH, "101.0", "s:1", "finalized",
                                         receipt_class="system_notice")
    payload = await temp_db.read_channel_sidecars_for_async(TEAM, CH, ["100.0", "101.0"])
    by_ts = {r["message_ts"]: r for r in payload["receipts"]}
    assert by_ts["100.0"]["receipt_class"] == "assistant_reply"
    assert by_ts["101.0"]["receipt_class"] == "system_notice"


async def test_channel_and_single_reads_carry_the_class(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s:1", "finalized",
                                         receipt_class="artifact")
    assert (await temp_db.get_receipt_async(TEAM, CH, "100.0"))["receipt_class"] == "artifact"
    rows = await temp_db.get_channel_receipts_async(TEAM, CH)
    assert rows[0]["receipt_class"] == "artifact"


# ------------------------------------------------------------------ pending shares end-to-end


async def test_a_pending_share_resolves_into_a_receipt_with_its_class(service, temp_db):
    assert await orx.record_pending_share(
        temp_db, team_id=TEAM, channel_id=CH, file_id="F1", owner_turn_id="s:1",
        thread_root_ts="99.0", receipt_class="artifact")
    pending = await temp_db.get_pending_shares_async()
    assert pending[0]["receipt_class"] == "artifact"
    assert await orx.resolve_pending_share(temp_db, team_id=TEAM, channel_id=CH,
                                           file_id="F1", message_ts="100.0")
    row = await _row(temp_db, "100.0")
    assert row["state"] == "finalized"
    assert row["receipt_class"] == "artifact"
    assert await temp_db.get_pending_shares_async() == []


async def test_a_queued_pending_share_still_resolves_with_its_class(service, temp_db,
                                                                    monkeypatch):
    """The pending row never reached the DB; the queued op's class rides the finalize."""

    async def _down(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(temp_db, "record_pending_share_async", _down)
    await orx.record_pending_share(temp_db, team_id=TEAM, channel_id=CH, file_id="F1",
                                   owner_turn_id="s:1", receipt_class="artifact")
    monkeypatch.undo()
    assert await orx.resolve_pending_share(temp_db, team_id=TEAM, channel_id=CH,
                                           file_id="F1", message_ts="100.0")
    row = await _row(temp_db, "100.0")
    assert row["state"] == "finalized"
    assert row["receipt_class"] == "artifact"


async def test_boot_recovery_resolves_pending_shares_with_their_class(service, temp_db):
    await temp_db.record_pending_share_async(TEAM, CH, "F1", "s:dead:1", "99.0",
                                             receipt_class="artifact")

    class _Client:
        async def resolve_file_share_ts(self, channel_id, file_id):
            return "100.0"

    assert await orx.recover_pending_shares(temp_db, _Client()) == 1
    row = await _row(temp_db, "100.0")
    assert row["state"] == "finalized"
    assert row["receipt_class"] == "artifact"
    assert await temp_db.get_pending_shares_async() == []


async def test_resolution_onto_an_established_class_nulls_it_and_logs(temp_db, caplog):
    """§11.1: a share resolving onto a ts that already carries a DIFFERENT class is a
    conflict — the class fails closed to NULL (the ineligible terminal state) with an ERROR
    naming both classes. The STATE resolution still lands and the pending row is deleted
    (terminal — no eternal boot retry)."""
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "s:1", "finalized",
                                         receipt_class="assistant_reply")
    await temp_db.record_pending_share_async(TEAM, CH, "F1", "s:1",
                                             receipt_class="artifact")
    with caplog.at_level("ERROR"):
        assert await temp_db.resolve_pending_share_async(TEAM, CH, "F1", "100.0")
    row = await _row(temp_db, "100.0")
    assert row["state"] == "finalized"
    assert row["receipt_class"] is None
    assert await temp_db.get_pending_shares_async() == []
    assert any("class conflict" in r.getMessage() and "'assistant_reply'" in r.getMessage()
               and "'artifact'" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------ legacy rows / migration


def test_the_migration_adds_the_columns_and_leaves_legacy_rows_null(tmp_path, monkeypatch):
    """A pre-§4 database gains both nullable columns; its rows stay NULL forever."""
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    legacy = sqlite3.connect(f"{tmp_path}/slack.db")
    legacy.execute("""
        CREATE TABLE outbound_receipts (
            team_id TEXT NOT NULL, channel_id TEXT NOT NULL, message_ts TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            state TEXT NOT NULL CHECK (state IN ('in_flight', 'finalized', 'chrome')),
            thread_root_ts TEXT, created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            finalized_ts TIMESTAMP, PRIMARY KEY (team_id, channel_id, message_ts))""")
    legacy.execute("""
        CREATE TABLE pending_share_receipts (
            team_id TEXT NOT NULL, channel_id TEXT NOT NULL, file_id TEXT NOT NULL,
            owner_turn_id TEXT NOT NULL, thread_root_ts TEXT,
            created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (team_id, channel_id, file_id))""")
    legacy.execute(
        "INSERT INTO outbound_receipts (team_id, channel_id, message_ts, turn_id, state) "
        "VALUES (?, ?, '100.0', 's:old', 'finalized')", (TEAM, CH))
    legacy.execute(
        "INSERT INTO pending_share_receipts (team_id, channel_id, file_id, owner_turn_id) "
        "VALUES (?, ?, 'F1', 's:old')", (TEAM, CH))
    legacy.commit()
    legacy.close()

    db = DatabaseManager(platform="slack")
    try:
        row = db.conn.execute(
            "SELECT receipt_class FROM outbound_receipts WHERE message_ts = '100.0'"
        ).fetchone()
        assert row["receipt_class"] is None
        share = db.conn.execute(
            "SELECT receipt_class FROM pending_share_receipts WHERE file_id = 'F1'"
        ).fetchone()
        assert share["receipt_class"] is None
    finally:
        db.conn.close()


def test_receipt_has_class_is_exact_and_never_infers():
    assert receipt_has_class({"receipt_class": "assistant_reply"}, "assistant_reply")
    assert not receipt_has_class({"receipt_class": "system_notice"}, "assistant_reply")
    # Legacy NULL is ineligible even on a finalized own reply — no inference from state.
    legacy = {"receipt_class": None, "state": "finalized", "turn_id": "s:old:1"}
    assert not receipt_has_class(legacy, "assistant_reply")
    # Rows that never heard of the column are legacy too.
    assert not receipt_has_class({"state": "finalized"}, "assistant_reply")
    assert not receipt_has_class(None, "assistant_reply")

    class _Rec:
        receipt_class = "artifact"

    assert receipt_has_class(_Rec(), "artifact")
    with pytest.raises(ValueError):
        receipt_has_class({"receipt_class": "assistant_reply"}, "reply")


# ------------------------------------------------------------------ the queue carries the class


def test_the_op_lattice_merges_the_class_like_the_root(service):
    service._enqueue(_Op("register", TEAM, CH, "1.0", "s:1", None, "assistant_reply"))
    service._enqueue(_Op("finalize", TEAM, CH, "1.0", "s:1"))
    op = service._queue[(TEAM, CH, "1.0")]
    assert op.kind == "finalize"
    assert op.receipt_class == "assistant_reply"  # the stronger op inherits the claim
    # And the other direction: a weaker late op donates its class to the queued winner.
    service._enqueue(_Op("register", TEAM, CH, "2.0", "s:1"))
    service._enqueue(_Op("finalize", TEAM, CH, "2.0", "s:1", None, "system_notice"))
    service._enqueue(_Op("register", TEAM, CH, "2.0", "s:1", None, "system_notice"))
    assert service._queue[(TEAM, CH, "2.0")].receipt_class == "system_notice"


def test_a_lattice_class_conflict_poisons_the_surviving_op(service, caplog):
    """§11.1/§11.12's queue half: two queued claims disagreeing about one message's class
    fail closed to the DISTINCT poisoned sentinel on the surviving op — in BOTH merge
    directions — with an ERROR naming both classes, and the poison writes to the DB as
    NULL (`db_class`)."""
    # A stronger op arriving over a queued weaker claim with a different class.
    service._enqueue(_Op("register", TEAM, CH, "1.0", "s:1", None, "assistant_reply"))
    with caplog.at_level("ERROR", logger="slack_bot.OutboundReceipts"):
        service._enqueue(_Op("finalize", TEAM, CH, "1.0", "s:1", None, "system_notice"))
    op = service._queue[(TEAM, CH, "1.0")]
    assert op.kind == "finalize"
    assert op.receipt_class is orx.POISONED_CLASS
    assert op.db_class is None, "the sentinel reaches the database as NULL, nothing else"
    # A weaker late op conflicting with the queued winner.
    service._enqueue(_Op("finalize", TEAM, CH, "2.0", "s:1", None, "system_notice"))
    with caplog.at_level("ERROR", logger="slack_bot.OutboundReceipts"):
        service._enqueue(_Op("register", TEAM, CH, "2.0", "s:1", None, "assistant_reply"))
    assert service._queue[(TEAM, CH, "2.0")].receipt_class is orx.POISONED_CLASS
    conflicts = [r for r in caplog.records if "class conflict" in r.message]
    assert len(conflicts) == 2
    assert all("assistant_reply" in r.getMessage() and "system_notice" in r.getMessage()
               for r in conflicts)


def test_a_third_claim_cannot_refill_a_poisoned_class(service, caplog):
    """§11.12/§11.18 regression: after a conflict poisons the queued op, a THIRD claim —
    from either merge direction — never refills the class. Reproduces the shipped bug:
    assistant_reply → conflict (was NULL, refillable) → artifact quietly refilled."""
    service._enqueue(_Op("register", TEAM, CH, "1.0", "s:1", None, "assistant_reply"))
    with caplog.at_level("ERROR", logger="slack_bot.OutboundReceipts"):
        service._enqueue(_Op("finalize", TEAM, CH, "1.0", "s:1", None, "system_notice"))
    assert service._queue[(TEAM, CH, "1.0")].receipt_class is orx.POISONED_CLASS
    # Third claim, weaker rank: donates nothing to the poisoned winner.
    service._enqueue(_Op("register", TEAM, CH, "1.0", "s:1", None, "artifact"))
    op = service._queue[(TEAM, CH, "1.0")]
    assert op.receipt_class is orx.POISONED_CLASS and op.db_class is None
    # Third claim, equal/stronger rank: the replacement inherits the poison, not a class.
    service._enqueue(_Op("finalize", TEAM, CH, "1.0", "s:1", None, "artifact"))
    op = service._queue[(TEAM, CH, "1.0")]
    assert op.kind == "finalize"
    assert op.receipt_class is orx.POISONED_CLASS and op.db_class is None


def test_queued_conflicting_pending_claims_keep_the_first(service, caplog):
    """§11.12/§11.18 regression: two queued pending-share claims for ONE file disagreeing
    about its class — the FIRST claimant's op stands (owner AND class), the second is
    dropped with the ERROR log, never an equal-rank replacement."""
    first = _Op("pending_share", TEAM, CH, "", "s:1", "9.0", "artifact", file_id="F1")
    service._enqueue(first)
    with caplog.at_level("ERROR", logger="slack_bot.OutboundReceipts"):
        service._enqueue(_Op("pending_share", TEAM, CH, "", "s:2", None,
                             "background_job", file_id="F1"))
    op = service._queue[(TEAM, CH, "file:F1")]
    assert op is first, "the first claimant stands"
    assert op.owner == "s:1" and op.receipt_class == "artifact"
    assert any("first claim stands" in r.getMessage() and "artifact" in r.getMessage()
               and "background_job" in r.getMessage() for r in caplog.records)


def test_queued_pending_claims_are_unconditionally_first_writer(service, caplog):
    """§11.20: at equal rank ANY differing second claim is dropped — first-writer without a
    class-conflict precondition, mirroring the database's rule. The two reproduced cases the
    conflict-only guard used to let through as replacements: a SAME-class second claimant
    (the file's owner and root must not change hands) and a valid class arriving over a
    NULL-class first claim (stays NULL-first — no refill)."""
    # Same class, different owner: the first claimant keeps the file.
    first = _Op("pending_share", TEAM, CH, "", "s:1", "9.0", "artifact", file_id="F2")
    service._enqueue(first)
    with caplog.at_level("ERROR", logger="slack_bot.OutboundReceipts"):
        service._enqueue(_Op("pending_share", TEAM, CH, "", "s:2", None, "artifact",
                             file_id="F2"))
    op = service._queue[(TEAM, CH, "file:F2")]
    assert op is first, "the first claimant stands"
    assert op.owner == "s:1" and op.thread_root_ts == "9.0"
    assert op.receipt_class == "artifact"
    assert any("first claim stands" in r.getMessage() for r in caplog.records)
    # NULL-first-then-valid: first-writer means no class refill either.
    caplog.clear()
    null_first = _Op("pending_share", TEAM, CH, "", "s:1", None, None, file_id="F3")
    service._enqueue(null_first)
    with caplog.at_level("ERROR", logger="slack_bot.OutboundReceipts"):
        service._enqueue(_Op("pending_share", TEAM, CH, "", "s:1", None, "artifact",
                             file_id="F3"))
    op = service._queue[(TEAM, CH, "file:F3")]
    assert op is null_first, "stays NULL-first"
    assert op.receipt_class is None, "the later claim never refills the class"
    assert any("first claim stands" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------ every producer stamps


# The send-path transports whose `receipts=` argument opens a receipt: any call passing one
# must say its class in the same breath. `update_message_streaming` is exempt by design (its
# receipt effect is the PROMOTION, whose class is the fixed §4 mapping), as are teardown
# helpers (abort/demote carry no claim) and `handle_error` (it stamps system_notice itself).
_STAMPED_SENDERS = {"send_message", "send_message_async", "send_message_get_ts",
                    "update_message", "update_message_async"}

# §11.13: the FIXED-surface transports take `receipt_class` as a REQUIRED parameter from
# their producers (no hardcoding inside the transport) — so EVERY call, receipted or not,
# must say the class. This is the honest unreceipted-path half of the §11.9 walk.
_FIXED_SURFACE_SENDERS = {"post_status_card", "send_image", "send_image_async",
                          "send_file", "send_thinking_indicator",
                          "send_thinking_indicator_async"}

# §11.24: the producer sweep walks EVERY module under the production roots — no manual
# whitelist a new posting site could dodge by living in a file nobody listed.
_PRODUCER_ROOTS = ("message_processor", "slack_client", "streaming")
_PRODUCER_TOP_LEVEL = ("main.py",)


def _producer_files() -> list:
    files = [REPO / name for name in _PRODUCER_TOP_LEVEL]
    for root in _PRODUCER_ROOTS:
        files.extend(sorted((REPO / root).rglob("*.py")))
    return files


def _keywords(call: ast.Call) -> dict:
    return {kw.arg: kw.value for kw in call.keywords if kw.arg}


def _is_none(node) -> bool:
    return isinstance(node, ast.Constant) and node.value is None


def _offending_calls(tree: ast.AST, path: str) -> list:
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else (
            func.id if isinstance(func, ast.Name) else None)
        if name in ("record_transport_post", "note_post", "note_chrome"):
            if "receipt_class" not in _keywords(node):
                offenders.append(f"{path}:{node.lineno} {name} without receipt_class")
            continue
        if name in _STAMPED_SENDERS:
            kws = _keywords(node)
            if "receipts" in kws and not _is_none(kws["receipts"]):
                if "receipt_class" not in kws:
                    offenders.append(f"{path}:{node.lineno} {name}(receipts=...) "
                                     "without receipt_class")
            # §11.9's unreceipted half: a bare lifecycle claim needs its class in the same
            # breath (the no-ledger transport path books it under the sys owner AS that
            # class), and the legacy-seed transport must always declare SOME receipt intent
            # — the dormant sync wrapper posting with neither was exactly this bug.
            if "receipt_kind" in kws and "receipt_class" not in kws:
                offenders.append(f"{path}:{node.lineno} {name}(receipt_kind=...) "
                                 "without receipt_class")
            if (name == "send_message_get_ts" and "receipts" not in kws
                    and "receipt_kind" not in kws):
                offenders.append(f"{path}:{node.lineno} send_message_get_ts with no "
                                 "receipt intent at all")
            continue
        if name in _FIXED_SURFACE_SENDERS:
            # §11.13: the class is a required parameter — receipted AND unreceipted calls
            # alike must carry it (the transport has no default and no hardcoded stamp).
            if "receipt_class" not in _keywords(node):
                offenders.append(f"{path}:{node.lineno} {name} without receipt_class "
                                 "(§11.13 required parameter)")
    return offenders


def test_every_producer_call_site_stamps_a_class_explicitly():
    """The §4 inventory walk: no production posting site passes a ledger without a class.

    Static on purpose — the behavioral tests above prove the stamps LAND; this one proves no
    NEW producer can appear without saying what it posts. §11.24: the walk covers EVERY
    module under message_processor/, slack_client/ and streaming/ plus main.py — no manual
    whitelist (the client contract rides the message_processor/ root as client_contract.py).
    At runtime the ledger keywords and the §11.13 fixed-surface transports make the omission a
    TypeError (required, no default); the variable-class transports have defaulted signatures,
    so their contract is the §11.9 ValueError on receipts-without-class — proved behaviorally
    below; this walk makes any of it a test failure at review time.
    """
    files = _producer_files()
    assert len(files) > len(_PRODUCER_TOP_LEVEL) + len(_PRODUCER_ROOTS), \
        "the recursive walk found the production trees"
    offenders = []
    for path in files:
        rel = path.relative_to(REPO).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders.extend(_offending_calls(tree, rel))
    assert not offenders, "producers posting without a receipt class:\n" + "\n".join(offenders)


def test_receipt_bearing_transport_signatures_carry_the_class_keyword():
    """§11.9/§11.13: EVERY receipt-bearing transport signature takes `receipt_class` beside
    `receipts` — the class can always ride the same call — and the fixed-surface transports
    take it REQUIRED (keyword-only, no default: the producer must say the class, the
    transport hardcodes nothing)."""
    import inspect

    import message_processor.client_contract as base_client
    from slack_client.messaging import SlackMessagingMixin

    for func in (SlackMessagingMixin.send_message,
                 SlackMessagingMixin.send_message_get_ts,
                 SlackMessagingMixin.update_message,
                 SlackMessagingMixin.post_status_card,
                 SlackMessagingMixin.send_image,
                 SlackMessagingMixin.send_file,
                 SlackMessagingMixin.send_thinking_indicator,
                 base_client.BaseClient.send_message,
                 base_client.BaseClient.send_message_async,
                 base_client.BaseClient.update_message,
                 base_client.BaseClient.update_message_async,
                 base_client.BaseClient.send_image,
                 base_client.BaseClient.send_image_async,
                 base_client.BaseClient.send_file,
                 base_client.BaseClient.send_thinking_indicator,
                 base_client.BaseClient.send_thinking_indicator_async):
        params = inspect.signature(func).parameters
        assert "receipts" in params and "receipt_class" in params, func.__qualname__
    for func in (SlackMessagingMixin.post_status_card,
                 SlackMessagingMixin.send_image,
                 SlackMessagingMixin.send_file,
                 SlackMessagingMixin.send_thinking_indicator,
                 base_client.BaseClient.send_image,
                 base_client.BaseClient.send_image_async,
                 base_client.BaseClient.send_file,
                 base_client.BaseClient.send_thinking_indicator,
                 base_client.BaseClient.send_thinking_indicator_async):
        param = inspect.signature(func).parameters["receipt_class"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY, func.__qualname__
        assert param.default is inspect.Parameter.empty, (
            f"{func.__qualname__}: §11.13 requires the class with NO default")


async def test_transports_refuse_receipts_without_a_class():
    """§11.9/§11.13: receipts-without-class is a programming error — ValueError BEFORE any
    Slack call, never a message delivered into a class-less (NULL) row. Every transport in
    the §11.13 list refuses it, base seams included."""
    import io
    from unittest.mock import MagicMock

    import message_processor.client_contract as base_client
    from slack_client.messaging import SlackMessagingMixin

    host = MagicMock()
    with pytest.raises(ValueError):
        await SlackMessagingMixin.send_message.__get__(host)(
            "C1", "1.0", "words", receipts=object())
    with pytest.raises(ValueError):
        await SlackMessagingMixin.send_message_get_ts.__get__(host)(
            "C1", "1.0", "words", receipts=object())
    with pytest.raises(ValueError):
        await SlackMessagingMixin.update_message.__get__(host)(
            "C1", "1.0", "words", receipts=object())
    with pytest.raises(ValueError):
        await SlackMessagingMixin.post_status_card.__get__(host)(
            "C1", "1.0", "fallback", [], receipts=object(), receipt_class=None)
    with pytest.raises(ValueError):
        await SlackMessagingMixin.send_image.__get__(host)(
            "C1", "1.0", b"x", "a.png", receipts=object(), receipt_class=None)
    with pytest.raises(ValueError):
        await SlackMessagingMixin.send_file.__get__(host)(
            "C1", "1.0", io.BytesIO(b"x"), "a.csv", receipts=object(), receipt_class=None)
    with pytest.raises(ValueError):
        await SlackMessagingMixin.send_thinking_indicator.__get__(host)(
            "C1", "1.0", receipts=object(), receipt_class=None)
    host.app.client.chat_postMessage.assert_not_called()
    host.app.client.chat_update.assert_not_called()
    host.app.client.files_upload_v2.assert_not_called()
    # The base seams honor the same contract in their default implementations — §11.23:
    # EVERY receipt-bearing base method body calls the shared guard, so the contract lives
    # in the base itself, not only in the concrete overrides.
    base = MagicMock()
    with pytest.raises(ValueError):
        await base_client.BaseClient.update_message(base, "C1", "1.0", "t", receipts=object())
    with pytest.raises(ValueError):
        await base_client.BaseClient.update_message_async(base, "C1", "1.0", "t",
                                                          receipts=object())
    with pytest.raises(ValueError):
        await base_client.BaseClient.send_file(base, "C1", "1.0", io.BytesIO(b"x"),
                                               "a.csv", receipts=object(),
                                               receipt_class=None)
    with pytest.raises(ValueError):
        await base_client.BaseClient.send_message(base, "C1", "1.0", "t", receipts=object())
    with pytest.raises(ValueError):
        await base_client.BaseClient.send_message_async(base, "C1", "1.0", "t",
                                                        receipts=object())
    with pytest.raises(ValueError):
        await base_client.BaseClient.send_image(base, "C1", "1.0", b"x", "a.png",
                                                receipts=object(), receipt_class=None)
    with pytest.raises(ValueError):
        await base_client.BaseClient.send_image_async(base, "C1", "1.0", b"x", "a.png",
                                                      receipts=object(),
                                                      receipt_class=None)
    with pytest.raises(ValueError):
        await base_client.BaseClient.send_thinking_indicator(base, "C1", "1.0",
                                                             receipts=object(),
                                                             receipt_class=None)
    with pytest.raises(ValueError):
        await base_client.BaseClient.send_thinking_indicator_async(
            base, "C1", "1.0", receipts=object(), receipt_class=None)


def test_the_fixed_surface_stamps_match_the_inventory():
    """The §4 inventory stamps, in source. §11.13 moved the fixed-surface classes OUT of
    the transports and into their producers: placeholder → chrome (main + progress), status
    card → background_job (research), shares → artifact (image/file producers). The
    transports keep only the surfaces they own outright (the split-abort truncation notice,
    the footers) and post_to_thread's assistant_reply."""

    def stamp_near(rel: str, anchor: str, cls: str, window: int = 900) -> bool:
        src = (REPO / rel).read_text(encoding="utf-8")
        i = src.find(anchor)
        assert i != -1, f"anchor {anchor!r} not found in {rel}"
        return f'receipt_class="{cls}"' in src[max(0, i - window):i + window]

    msg = "slack_client/messaging.py"
    assert stamp_near(msg, 'site="send_message_truncation"', "system_notice", 300)
    assert stamp_near(msg, 'site="feedback_footer"', "chrome", 300)
    assert stamp_near(msg, 'site="response_footer"', "chrome", 300)
    assert stamp_near(msg, 'surface="post_to_thread"', "assistant_reply", 400)
    # §11.13: the producers stamp the fixed-surface classes — never the transports.
    assert stamp_near("main.py", "send_thinking_indicator(", "chrome", 300)
    assert stamp_near("message_processor/progress.py",
                      "send_thinking_indicator(", "chrome", 300)
    assert stamp_near("message_processor/research_tools.py",
                      "post_status_card(", "background_job", 400)
    assert stamp_near("message_processor/image_delivery.py",
                      "send_image(", "artifact", 400)
    assert stamp_near("message_processor/artifacts.py", "send_file(", "artifact", 300)
    # And the transports themselves carry NO hardcoded fixed-surface stamp any more.
    src = (REPO / msg).read_text(encoding="utf-8")
    for site in ('site="send_thinking_indicator"', 'site="post_status_card"'):
        i = src.find(site)
        assert i != -1
        assert "receipt_class=receipt_class" in src[max(0, i - 500):i + 500], (
            f"{site}: the transport must forward the producer's class, not its own")
