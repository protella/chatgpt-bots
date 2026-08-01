"""Channel-turn sidecar read + the channel documents uniqueness rule (single-stream P1, spec §3).

THE READ IS THE ID FORM — `read_channel_sidecars_for_async`, the render pin over exact message
identities (SHALLOW_STREAM_RESPEC §4.6). The `(floor, high]` WINDOW form it replaced is gone, and
with it every floor/ceiling predicate this file used to pin; the activity rows moved to
`read_channel_discovery_roots_async`, whose own predicate is covered in
`test_channel_window_selection.py`.

Every timestamp here is deliberately unequal-width: "999.999900" sorts ABOVE "1000.000100" as a
string and BELOW it as a number, so a lexicographic comparison anywhere in the ordering flips
exactly the rows these tests pin.
"""
import itertools
import json
from pathlib import Path

import pytest

from database import (OUTBOUND_RECEIPTS_EPOCH_KEY, UNATTENDED_SUMMARY_TEMPLATE,
                      _CHANNEL_DOCS_INDEX, _CHANNEL_DOCS_PREDICATE, DatabaseManager,
                      is_unattended_summary)

TEAM = "T1"
CH = "C1"
THREAD = f"{CH}:1000.000100"
OTHER_THREAD = "C2:1000.000100"
DM_THREAD = "D123:1000.000100"

INSIDE = "1500.0"

_seq = itertools.count(1)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    db = DatabaseManager(platform="slack")
    yield db
    db.conn.close()


# ------------------------------------------------------------------ seeding helpers

def _img(db, thread_id, message_ts, *, analysis="A", image_type="uploaded", metadata=None):
    db.conn.execute(
        "INSERT INTO images (thread_id, url, message_ts, image_type, analysis, metadata_json) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (thread_id, f"https://files/{next(_seq)}", message_ts, image_type, analysis,
         json.dumps(metadata) if metadata else None))


def _doc(db, thread_id, message_ts, *, filename="f.csv", mime="text/csv", file_id=None,
         summary="S", created_at=None):
    file_id = file_id if file_id is not None else f"F{next(_seq)}"
    if created_at is None:
        db.conn.execute(
            "INSERT INTO documents (thread_id, filename, mime_type, summary, file_id, message_ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (thread_id, filename, mime, summary, file_id, message_ts))
    else:
        db.conn.execute(
            "INSERT INTO documents (thread_id, filename, mime_type, summary, file_id, "
            " message_ts, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (thread_id, filename, mime, summary, file_id, message_ts, created_at))


def _art(db, channel_id, source_ts, *, kind="link", ref=None, status="pending", title="T",
         summary=None, derivation_source="fetch"):
    db.conn.execute(
        "INSERT INTO ambient_artifacts (channel_id, source_ts, conversation_ts, kind, ref, "
        " title, summary, status, derivation_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (channel_id, source_ts, source_ts, kind, ref or f"ref-{next(_seq)}", title, summary,
         status, derivation_source))


def _tools(db, channel_id, message_ts, tools_json='[{"tool_name": "search", "gist": "g"}]'):
    db.conn.execute(
        "INSERT INTO message_tool_usage (channel_id, message_ts, thread_key, tools_json) "
        "VALUES (?, ?, ?, ?)", (channel_id, message_ts, f"{channel_id}:{message_ts}", tools_json))


async def _read(db, *, ids, channel_id=CH, team_id=TEAM):
    return await db.read_channel_sidecars_for_async(team_id, channel_id, ids)


def _ts(rows, key="message_ts"):
    return [r[key] for r in rows]


def _assert_plain(value):
    if isinstance(value, dict):
        for key, inner in value.items():
            assert isinstance(key, str), repr(key)
            _assert_plain(inner)
    elif isinstance(value, (list, tuple)):
        for inner in value:
            _assert_plain(inner)
    else:
        assert value is None or isinstance(value, (str, int, float, bool)), repr(value)


# ------------------------------------------------------------------ payload shape

async def test_payload_has_the_documented_keys(temp_db):
    payload = await _read(temp_db, ids=[INSIDE])
    assert set(payload) == {
        "ids", "receipt_feature_epoch_ts", "receipts", "image_analyses",
        "document_extractions", "ambient_artifacts", "tool_usage", "versions_hash"}


async def test_the_id_list_comes_back_normalized_and_sorted(temp_db):
    """The pin carries its own subject: the hash means "these rows, for these messages", so the
    list it hashed has to be the list the caller can see."""
    payload = await _read(temp_db, ids=["1500.000200", INSIDE, INSIDE, "", "1499.999900"])
    assert payload["ids"] == ["1499.999900", "1500.0", "1500.000200"]


# ------------------------------------------------------------------ receipts + the feature epoch

async def test_receipts_are_read_whole_for_the_ids_asked_for(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, INSIDE, "turn-1", "finalized",
                                         thread_root_ts="1400.0")
    rows = {r["message_ts"]: r for r in (await _read(temp_db, ids=[INSIDE]))["receipts"]}
    assert set(rows) == {INSIDE}
    assert rows[INSIDE]["state"] == "finalized"
    assert rows[INSIDE]["turn_id"] == "turn-1"
    assert rows[INSIDE]["thread_root_ts"] == "1400.0"


async def test_receipts_are_ts_ordered_numerically(temp_db):
    for ts in ("1000.000100", "999.999900"):
        await temp_db.register_receipt_async(TEAM, CH, ts, "turn", "finalized")
    payload = await _read(temp_db, ids=["1000.000100", "999.999900"])
    assert _ts(payload["receipts"]) == ["999.999900", "1000.000100"]


async def test_receipt_feature_epoch_comes_from_bot_meta(temp_db):
    """`_self_role` needs it to decide whether one of our own messages predates the receipts era,
    and a None here excludes every one of them — so it rides every read, id list or not."""
    await temp_db.set_meta_if_absent_async(OUTBOUND_RECEIPTS_EPOCH_KEY, "1699999999.000000")
    assert (await _read(temp_db, ids=[INSIDE]))["receipt_feature_epoch_ts"] == "1699999999.000000"


async def test_receipt_feature_epoch_is_none_when_unset(temp_db):
    assert (await _read(temp_db, ids=[INSIDE]))["receipt_feature_epoch_ts"] is None


# ------------------------------------------------------------------ scoping

async def test_images_and_documents_are_scoped_to_this_channels_thread_prefix(temp_db):
    for thread in (THREAD, OTHER_THREAD, DM_THREAD):
        _img(temp_db, thread, INSIDE)
        _doc(temp_db, thread, INSIDE)

    payload = await _read(temp_db, ids=[INSIDE])
    assert [r["thread_id"] for r in payload["image_analyses"]] == [THREAD]
    assert [r["thread_id"] for r in payload["document_extractions"]] == [THREAD]


async def test_ambient_and_tool_usage_are_scoped_to_this_channel(temp_db):
    _art(temp_db, CH, INSIDE, ref="mine")
    _art(temp_db, "C2", INSIDE, ref="theirs")
    _art(temp_db, "D123", INSIDE, ref="dm")
    _tools(temp_db, CH, INSIDE)
    _tools(temp_db, "C2", INSIDE)
    _tools(temp_db, "D123", INSIDE)

    payload = await _read(temp_db, ids=[INSIDE])
    assert [r["ref"] for r in payload["ambient_artifacts"]] == ["mine"]
    assert list(payload["tool_usage"]) == [INSIDE]
    assert len(payload["tool_usage"]) == 1


async def test_receipts_are_scoped_to_team_and_channel(temp_db):
    for team, channel in ((TEAM, CH), (TEAM, "C2"), ("T2", CH)):
        await temp_db.register_receipt_async(team, channel, INSIDE, "turn", "finalized")

    assert len((await _read(temp_db, ids=[INSIDE]))["receipts"]) == 1


# ------------------------------------------------------------------ NULL ts + ordering

async def test_rows_with_a_null_message_ts_are_excluded(temp_db):
    _img(temp_db, THREAD, None)
    _doc(temp_db, THREAD, None)
    _img(temp_db, THREAD, INSIDE)
    _doc(temp_db, THREAD, INSIDE)

    payload = await _read(temp_db, ids=[INSIDE])
    assert _ts(payload["image_analyses"]) == [INSIDE]
    assert _ts(payload["document_extractions"]) == [INSIDE]


async def test_lists_are_ordered_by_numeric_ts_then_id(temp_db):
    out_of_order = ("1500.000200", "1499.999900", "1500.000100")
    for ts in out_of_order:
        _img(temp_db, THREAD, ts, analysis=f"a-{ts}")
        _doc(temp_db, THREAD, ts, summary=f"s-{ts}")
        _art(temp_db, CH, ts, ref=f"r-{ts}")
    # A same-ts pair to pin the id tiebreak.
    _img(temp_db, THREAD, "1500.000100", analysis="second-at-same-ts")
    _doc(temp_db, THREAD, "1500.000100", summary="second-at-same-ts")
    _art(temp_db, CH, "1500.000100", ref="second-at-same-ts")

    payload = await _read(temp_db, ids=list(out_of_order))
    expect = ["1499.999900", "1500.000100", "1500.000100", "1500.000200"]
    for key, field in (("image_analyses", "message_ts"), ("document_extractions", "message_ts"),
                       ("ambient_artifacts", "source_ts")):
        rows = payload[key]
        assert _ts(rows, field) == expect, key
        # id breaks the tie at 1500.000100, ascending.
        assert rows[1]["id"] < rows[2]["id"], key
    assert payload["image_analyses"][2]["analysis"] == "second-at-same-ts"
    assert payload["document_extractions"][2]["summary"] == "second-at-same-ts"
    assert payload["ambient_artifacts"][2]["ref"] == "second-at-same-ts"


# ------------------------------------------------------------------ tool usage

async def test_tool_usage_maps_ts_to_the_parsed_tools_list(temp_db):
    _tools(temp_db, CH, INSIDE, tools_json='[{"tool_name": "search_slack", "gist": "q"}]')
    payload = await _read(temp_db, ids=[INSIDE])
    assert payload["tool_usage"] == {INSIDE: [{"tool_name": "search_slack", "gist": "q"}]}


@pytest.mark.parametrize("bad", ["{not json", "", '{"tool_name": "x"}', "null", "17"])
async def test_unusable_tool_usage_json_is_skipped_not_raised(temp_db, bad):
    _tools(temp_db, CH, INSIDE, tools_json=bad)
    _tools(temp_db, CH, "1600.0")
    payload = await _read(temp_db, ids=[INSIDE, "1600.0"])
    assert list(payload["tool_usage"]) == ["1600.0"]


# ------------------------------------------------------------------ versions hash

async def test_versions_hash_is_stable_across_identical_reads(temp_db):
    _img(temp_db, THREAD, INSIDE, metadata={"filename": "chart.png"})
    _doc(temp_db, THREAD, INSIDE)
    _art(temp_db, CH, INSIDE)
    _tools(temp_db, CH, INSIDE)
    await temp_db.register_receipt_async(TEAM, CH, INSIDE, "turn", "finalized")

    first = (await _read(temp_db, ids=[INSIDE]))["versions_hash"]
    second = (await _read(temp_db, ids=[INSIDE]))["versions_hash"]
    assert first == second
    assert len(first) == 64


async def test_versions_hash_changes_when_a_document_summary_is_edited(temp_db):
    _doc(temp_db, THREAD, INSIDE, summary="first")
    before = (await _read(temp_db, ids=[INSIDE]))["versions_hash"]
    temp_db.conn.execute("UPDATE documents SET summary = 'rewritten'")
    assert (await _read(temp_db, ids=[INSIDE]))["versions_hash"] != before


async def test_versions_hash_changes_when_an_ambient_status_flips_to_ready(temp_db):
    _art(temp_db, CH, INSIDE, status="pending")
    before = (await _read(temp_db, ids=[INSIDE]))["versions_hash"]
    temp_db.conn.execute("UPDATE ambient_artifacts SET status = 'ready'")
    assert (await _read(temp_db, ids=[INSIDE]))["versions_hash"] != before


async def test_versions_hash_changes_when_a_receipt_is_promoted(temp_db):
    await temp_db.register_receipt_async(TEAM, CH, INSIDE, "turn", "in_flight")
    before = (await _read(temp_db, ids=[INSIDE]))["versions_hash"]
    await temp_db.register_receipt_async(TEAM, CH, INSIDE, "turn", "finalized")
    assert (await _read(temp_db, ids=[INSIDE]))["versions_hash"] != before


async def test_versions_hash_ignores_a_field_the_serializer_never_renders(temp_db):
    """image_type and mime_type are read but not hashed — touching them must not miss the cache."""
    _img(temp_db, THREAD, INSIDE, image_type="uploaded")
    _doc(temp_db, THREAD, INSIDE, mime="text/csv")
    before = (await _read(temp_db, ids=[INSIDE]))["versions_hash"]

    temp_db.conn.execute("UPDATE images SET image_type = 'generated'")
    temp_db.conn.execute("UPDATE documents SET mime_type = 'application/pdf'")

    payload = await _read(temp_db, ids=[INSIDE])
    assert payload["image_analyses"][0]["image_type"] == "generated"
    assert payload["document_extractions"][0]["mime_type"] == "application/pdf"
    assert payload["versions_hash"] == before


# ------------------------------------------------------------------ transaction discipline

async def test_payload_is_plain_committed_data(temp_db):
    _img(temp_db, THREAD, INSIDE, metadata={"filename": "x.png"})
    _doc(temp_db, THREAD, INSIDE)
    _art(temp_db, CH, INSIDE)
    _tools(temp_db, CH, INSIDE)
    await temp_db.register_receipt_async(TEAM, CH, INSIDE, "turn", "finalized")

    payload = await _read(temp_db, ids=[INSIDE])
    # Exact types on purpose: a sqlite3.Row / lazy mapping must not survive the accessor.
    assert type(payload) is dict  # noqa: E721
    for key in ("receipts", "image_analyses", "document_extractions", "ambient_artifacts"):
        for row in payload[key]:
            assert type(row) is dict, key  # noqa: E721
    _assert_plain(payload)


async def test_a_read_after_a_write_sees_the_write(temp_db):
    """The read transaction really commits and the connection closes — no held snapshot."""
    assert (await _read(temp_db, ids=[INSIDE]))["document_extractions"] == []
    assert await temp_db.save_document_if_absent_async(
        THREAD, "late.csv", "text/csv", file_id="FLATE", message_ts=INSIDE)
    rows = (await _read(temp_db, ids=[INSIDE]))["document_extractions"]
    assert [r["filename"] for r in rows] == ["late.csv"]



# ------------------------------------------------------------------ uniqueness migration

def _index_sql(db):
    row = db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
        (_CHANNEL_DOCS_INDEX,)).fetchone()
    return row["sql"] if row else None


def _doc_rows(db, thread_id):
    return db.conn.execute(
        "SELECT id, summary FROM documents WHERE thread_id = ? ORDER BY id",
        (thread_id,)).fetchall()


def test_the_unique_partial_index_exists_after_init(temp_db):
    sql = _index_sql(temp_db)
    assert sql is not None
    assert "UNIQUE" in sql.upper()
    assert _CHANNEL_DOCS_PREDICATE in sql
    assert "(thread_id, message_ts, file_id)" in sql


@pytest.mark.parametrize("placeholder_first", [True, False])
def test_migration_keeps_the_real_summary_over_the_placeholder(temp_db, placeholder_first):
    placeholder = UNATTENDED_SUMMARY_TEMPLATE.format(name="sales.csv")
    summaries = ([placeholder, "A real extraction."] if placeholder_first
                 else ["A real extraction.", placeholder])
    temp_db.conn.execute(f"DROP INDEX IF EXISTS {_CHANNEL_DOCS_INDEX}")
    for summary in summaries:
        _doc(temp_db, THREAD, INSIDE, filename="sales.csv", file_id="FDUP", summary=summary,
             created_at="2026-07-27 00:00:00")

    temp_db._migrate_channel_document_uniqueness()

    rows = _doc_rows(temp_db, THREAD)
    assert len(rows) == 1
    assert rows[0]["summary"] == "A real extraction."
    assert _index_sql(temp_db) is not None


def test_migration_tiebreak_keeps_the_newest_real_row(temp_db):
    temp_db.conn.execute(f"DROP INDEX IF EXISTS {_CHANNEL_DOCS_INDEX}")
    for summary in ("older real", "newer real"):
        _doc(temp_db, THREAD, INSIDE, filename="sales.csv", file_id="FDUP", summary=summary,
             created_at="2026-07-27 00:00:00")
    newest = _doc_rows(temp_db, THREAD)[-1]["id"]

    temp_db._migrate_channel_document_uniqueness()

    rows = _doc_rows(temp_db, THREAD)
    assert [(r["id"], r["summary"]) for r in rows] == [(newest, "newer real")]


def test_reopening_the_manager_runs_the_dedup(temp_db):
    temp_db.conn.execute(f"DROP INDEX IF EXISTS {_CHANNEL_DOCS_INDEX}")
    _doc(temp_db, THREAD, INSIDE, filename="a.csv", file_id="FDUP",
         summary=UNATTENDED_SUMMARY_TEMPLATE.format(name="a.csv"),
         created_at="2026-07-27 00:00:00")
    _doc(temp_db, THREAD, INSIDE, filename="a.csv", file_id="FDUP", summary="Real.",
         created_at="2026-07-27 00:00:00")

    reopened = DatabaseManager(platform="slack")
    try:
        rows = _doc_rows(reopened, THREAD)
        assert [r["summary"] for r in rows] == ["Real."]
        assert _index_sql(reopened) is not None
    finally:
        reopened.conn.close()


def test_dm_duplicates_are_neither_deduped_nor_constrained(temp_db):
    _doc(temp_db, DM_THREAD, INSIDE, filename="a.csv", file_id="FDM",
         summary=UNATTENDED_SUMMARY_TEMPLATE.format(name="a.csv"))
    _doc(temp_db, DM_THREAD, INSIDE, filename="a.csv", file_id="FDM", summary="Real.")

    temp_db._migrate_channel_document_uniqueness()
    assert len(_doc_rows(temp_db, DM_THREAD)) == 2


def test_save_document_still_duplicates_in_a_dm(temp_db):
    for _ in range(2):
        temp_db.save_document(thread_id=DM_THREAD, filename="a.csv", mime_type="text/csv",
                              summary="Real.", file_id="FDM", message_ts=INSIDE)
    assert len(_doc_rows(temp_db, DM_THREAD)) == 2


# ------------------------------------------------------------------ placeholder → richer write
#
# The two phases of ONE channel turn, both through their real writers against a real database.
# The stream's origin ingester writes the unattended placeholder before admission
# (save_document_if_absent_async); document finalization then writes the extraction it paid a
# utility model for through the legacy sync writer (save_document, reached via
# DocumentLedger.add_document). Mocking the phases separately hides the collision between them.

def _doc_row(db, thread_id, file_id):
    return db.conn.execute(
        "SELECT summary, total_pages, size_bytes, mime_type, url_private FROM documents "
        "WHERE thread_id = ? AND file_id = ?", (thread_id, file_id)).fetchone()


async def test_the_richer_write_upgrades_the_channel_placeholder_in_place(temp_db):
    assert await temp_db.save_document_if_absent_async(
        THREAD, "sales.csv", "text/csv",
        summary=UNATTENDED_SUMMARY_TEMPLATE.format(name="sales.csv"),
        file_id="FX", url_private="https://slack/FX", size_bytes=11, message_ts=INSIDE)

    temp_db.save_document(thread_id=THREAD, filename="sales.csv", mime_type="text/csv",
                          summary="A real extraction.", file_id="FX",
                          url_private="https://slack/FX", size_bytes=2048, total_pages=3,
                          message_ts=INSIDE)

    assert len(_doc_rows(temp_db, THREAD)) == 1
    row = _doc_row(temp_db, THREAD, "FX")
    # WHOLE row, not a field merge: every column comes from the write that actually read the file.
    assert (row["summary"], row["total_pages"], row["size_bytes"]) == ("A real extraction.", 3,
                                                                       2048)


async def test_the_ledger_writer_reaches_the_same_upgrade(temp_db):
    """The production phase-2 caller: utilities hands DocumentLedger.add_document the db."""
    from thread_manager import DocumentLedger

    assert await temp_db.save_document_if_absent_async(
        THREAD, "q3.pdf", "application/pdf",
        summary=UNATTENDED_SUMMARY_TEMPLATE.format(name="q3.pdf"),
        file_id="FY", url_private="https://slack/FY", message_ts=INSIDE)

    DocumentLedger(thread_ts=INSIDE).add_document(
        content="the extracted text", filename="q3.pdf", mime_type="application/pdf",
        summary="A three-page report.", total_pages=3, db=temp_db, thread_id=THREAD,
        message_ts=INSIDE, file_id="FY", url_private="https://slack/FY", size_bytes=999)

    rows = _doc_rows(temp_db, THREAD)
    assert [r["summary"] for r in rows] == ["A three-page report."]


async def test_the_placeholder_never_overwrites_a_real_summary(temp_db):
    """The reverse order. `catalog_unattended` also writes through save_document, so the upgrade
    must refuse to run backwards — the migration's rule holds at write time too."""
    temp_db.save_document(thread_id=THREAD, filename="sales.csv", mime_type="text/csv",
                          summary="A real extraction.", file_id="FZ", total_pages=3,
                          message_ts=INSIDE)
    temp_db.save_document(thread_id=THREAD, filename="sales.csv", mime_type="text/csv",
                          summary=UNATTENDED_SUMMARY_TEMPLATE.format(name="sales.csv"),
                          file_id="FZ", message_ts=INSIDE)

    assert len(_doc_rows(temp_db, THREAD)) == 1
    row = _doc_row(temp_db, THREAD, "FZ")
    assert (row["summary"], row["total_pages"]) == ("A real extraction.", 3)


async def test_a_second_richer_write_still_replaces_the_row(temp_db):
    for summary in ("First read.", "Read again after an edit."):
        temp_db.save_document(thread_id=THREAD, filename="sales.csv", mime_type="text/csv",
                              summary=summary, file_id="FW", message_ts=INSIDE)
    rows = _doc_rows(temp_db, THREAD)
    assert [r["summary"] for r in rows] == ["Read again after an edit."]


def test_the_dm_write_is_untouched_by_the_channel_upgrade(temp_db):
    """DM isolation, stated as behaviour rather than as a predicate: the same two writes that
    collapse to one upgraded row in a channel stay two independent observations in a DM."""
    temp_db.save_document(thread_id=DM_THREAD, filename="sales.csv", mime_type="text/csv",
                          summary=UNATTENDED_SUMMARY_TEMPLATE.format(name="sales.csv"),
                          file_id="FD", message_ts=INSIDE)
    temp_db.save_document(thread_id=DM_THREAD, filename="sales.csv", mime_type="text/csv",
                          summary="A real extraction.", file_id="FD", total_pages=3,
                          message_ts=INSIDE)

    rows = _doc_rows(temp_db, DM_THREAD)
    assert [r["summary"] for r in rows] == [
        UNATTENDED_SUMMARY_TEMPLATE.format(name="sales.csv"), "A real extraction."]


def test_a_channel_row_with_no_file_id_is_outside_the_upgrade_scope(temp_db):
    """The partial index cannot address a NULL file id, so those writes keep the legacy INSERT —
    `save_document_if_absent_async` is the path that dedupes them, under BEGIN IMMEDIATE."""
    for _ in range(2):
        temp_db.save_document(thread_id=THREAD, filename="pasted.txt", mime_type="text/plain",
                              summary="Real.", file_id=None, message_ts=INSIDE)
    assert len(_doc_rows(temp_db, THREAD)) == 2


# ------------------------------------------------------------------ save_document_if_absent_async

async def test_if_absent_inserts_once_for_the_same_share(temp_db):
    """A passing assertion here proves the ON CONFLICT target matches the partial index —
    a mismatched target is a SQLite error, not a silent no-op."""
    args = (THREAD, "a.csv", "text/csv")
    assert await temp_db.save_document_if_absent_async(
        *args, summary="Real.", file_id="FA", message_ts=INSIDE)
    assert not await temp_db.save_document_if_absent_async(
        *args, summary="Real.", file_id="FA", message_ts=INSIDE)
    assert len(_doc_rows(temp_db, THREAD)) == 1


async def test_if_absent_with_a_null_file_id_uses_lookup_before_insert(temp_db):
    args = (THREAD, "a.csv", "text/csv")
    assert await temp_db.save_document_if_absent_async(*args, message_ts=INSIDE)
    assert not await temp_db.save_document_if_absent_async(*args, message_ts=INSIDE)
    assert len(_doc_rows(temp_db, THREAD)) == 1


async def test_if_absent_with_a_null_file_id_yields_to_a_row_that_has_one(temp_db):
    """Intended idempotency, not a near-miss: the NULL-key lookup matches on
    (thread_id, filename, message_ts) and ignores file_id, so a second write with no file id
    refuses rather than adding a row that describes the same file with less about it."""
    assert await temp_db.save_document_if_absent_async(
        THREAD, "a.csv", "text/csv", summary="Real.", file_id="FA", message_ts=INSIDE)
    assert not await temp_db.save_document_if_absent_async(
        THREAD, "a.csv", "text/csv", summary="Real.", file_id=None, message_ts=INSIDE)
    rows = _doc_rows(temp_db, THREAD)
    assert len(rows) == 1
    assert temp_db.conn.execute(
        "SELECT file_id FROM documents WHERE id = ?", (rows[0]["id"],)).fetchone()[0] == "FA"


async def test_if_absent_inserts_again_for_a_different_message_ts(temp_db):
    for ts in (INSIDE, "1600.000100"):
        assert await temp_db.save_document_if_absent_async(
            THREAD, "a.csv", "text/csv", file_id="FA", message_ts=ts)
    assert len(_doc_rows(temp_db, THREAD)) == 2


async def test_if_absent_inserts_again_for_a_different_thread(temp_db):
    assert await temp_db.save_document_if_absent_async(
        THREAD, "a.csv", "text/csv", file_id="FA", message_ts=INSIDE)
    assert await temp_db.save_document_if_absent_async(
        f"{CH}:1700.0", "a.csv", "text/csv", file_id="FA", message_ts=INSIDE)
    assert len(_doc_rows(temp_db, THREAD)) == 1
    assert len(_doc_rows(temp_db, f"{CH}:1700.0")) == 1


# ------------------------------------------------------------------ the unattended sentinel

@pytest.mark.parametrize("summary,expected", [
    (UNATTENDED_SUMMARY_TEMPLATE.format(name="q3.xlsx"), True),
    ("Shared in this conversation (x.csv). Not yet read.", True),
    ("  Shared in this conversation (x.csv). Not yet read.  ", True),
    ("", False),
    (None, False),
    ("A three-page contract between two parties.", False),
    ("Shared in this conversation (", False),
    ("Shared in this conversation (). Not yet read.", False),
    ("Shared in this conversation with the team.", False),
])
def test_is_unattended_summary(summary, expected):
    assert is_unattended_summary(summary) is expected


def test_catalog_unattended_writes_the_shared_sentinel():
    """The writer uses the CONSTANT, so there is no second literal to drift.

    It held its own copy of the string until 2026-07-30, which is what this test was originally
    for. Now it pins the stronger property — nobody reintroduces a literal — and still checks
    the sentinel recognises what the writer actually produces.
    """
    source = Path(__file__).resolve().parents[2] / "message_processor" / "thread_files.py"
    text = source.read_text(encoding="utf-8")
    assert "from database import UNATTENDED_SUMMARY_TEMPLATE" in text
    assert "UNATTENDED_SUMMARY_TEMPLATE.format(name=name)" in text
    assert "Shared in this conversation" not in text
    assert is_unattended_summary(UNATTENDED_SUMMARY_TEMPLATE.format(name="x.csv"))
    assert is_unattended_summary("Shared in this conversation (x.csv). Not yet read.")
