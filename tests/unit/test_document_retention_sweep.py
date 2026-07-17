"""Retention sweep for stored document-extraction rows (delete_old_documents).

The sweep SLIMS rather than deletes (Finding 7): a document behind a compaction boundary is
never recreated by a rebuild, so DELETING its row made a still-in-Slack file unresolvable
(`document_not_found`) even though the summary head still referenced it. Slimming nulls the
bulky derived fields (summary/page_structure/metadata) while PRESERVING the reference row
(filename, thread/channel key, Slack file_id/url_private, mime_type, size, timestamps) so
read_document and thread rebuilds can always re-resolve and re-extract from Slack on demand.

The sweep is wired into main.py's scheduled cleanup_worker alongside the tool-usage and
ambient-artifact sweeps. These tests cover the DB method directly (the worker's nested async
closure isn't unit-addressable): age boundary, config-driven window, the UTC-vs-local-time trap
the docstring calls out (created_at is CURRENT_TIMESTAMP/UTC, so the cutoff must be computed in
SQL, not with datetime.now()), and that a slimmed row is still readable via read_document.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import config


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    from database import DatabaseManager
    db = DatabaseManager(platform="slack")
    yield db
    db.conn.close()


def _save(db, filename="doc.pdf"):
    db.save_document(
        thread_id="C1:100.0", filename=filename, mime_type="application/pdf",
        summary="a long derived summary of the document body",
        file_id="F1", url_private="https://slack/f1", size_bytes=4096,
        page_structure={"pages": [{"n": 1, "text": "page one body"}]},
        total_pages=1, metadata={"pages": 1}, message_ts="100.0")


def _row(db):
    docs = db.get_thread_documents("C1:100.0")
    return docs[0] if docs else None


def _count(db):
    return len(db.get_thread_documents("C1:100.0"))


def test_slims_rows_older_than_window_but_keeps_refs(temp_db):
    _save(temp_db)
    temp_db.conn.execute("UPDATE documents SET created_at = datetime('now', '-200 days')")
    temp_db.delete_old_documents(days=90)

    # Row SURVIVES (not deleted) with its Slack refs intact...
    assert _count(temp_db) == 1
    row = _row(temp_db)
    assert row["filename"] == "doc.pdf"
    assert row["file_id"] == "F1"
    assert row["url_private"] == "https://slack/f1"
    assert row["mime_type"] == "application/pdf"
    assert row["size_bytes"] == 4096
    # ...and the bulky DERIVED fields are cleared.
    assert row["summary"] is None
    assert row["page_structure"] is None
    assert not row.get("metadata")  # metadata_json nulled → no parsed metadata


def test_keeps_derived_bulk_for_rows_within_window(temp_db):
    _save(temp_db)
    temp_db.conn.execute("UPDATE documents SET created_at = datetime('now', '-50 days')")
    temp_db.delete_old_documents(days=90)
    row = _row(temp_db)
    assert _count(temp_db) == 1
    assert row["summary"] == "a long derived summary of the document body"
    assert row["page_structure"] is not None  # untouched inside the window


def test_honors_config_retention_days(temp_db, monkeypatch):
    # The window comes from config.document_retention_days at the cleanup call site. A row
    # aged 50 days keeps its bulk under the default 90 but is slimmed under a tightened 30.
    _save(temp_db)
    temp_db.conn.execute("UPDATE documents SET created_at = datetime('now', '-50 days')")
    monkeypatch.setattr(config, "document_retention_days", 90, raising=False)
    temp_db.delete_old_documents(days=config.document_retention_days)
    assert _row(temp_db)["summary"] is not None  # within 90d window
    monkeypatch.setattr(config, "document_retention_days", 30, raising=False)
    temp_db.delete_old_documents(days=config.document_retention_days)
    assert _row(temp_db)["summary"] is None      # past 30d window → slimmed
    assert _count(temp_db) == 1                  # but never deleted


def test_config_default_is_90(temp_db):
    assert config.document_retention_days == 90


def test_boundary_is_utc_correct(temp_db):
    """Guards the trap in the docstring: on a non-UTC host a Python datetime.now() cutoff
    (LOCAL) compared against a UTC created_at skews the window by the host offset. A just-
    written row (created_at = CURRENT_TIMESTAMP, UTC) must keep its summary under any window."""
    _save(temp_db)
    temp_db.delete_old_documents(days=1)
    assert _row(temp_db)["summary"] is not None

    # A row one hour inside the 90-day edge keeps its bulk; one hour past it is slimmed — both
    # measured in the same UTC clock the SQL cutoff uses. The row itself always survives.
    temp_db.conn.execute("UPDATE documents SET created_at = datetime('now', '-90 days', '+1 hour')")
    temp_db.delete_old_documents(days=90)
    assert _row(temp_db)["summary"] is not None

    temp_db.conn.execute("UPDATE documents SET created_at = datetime('now', '-90 days', '-1 hour')")
    temp_db.delete_old_documents(days=90)
    assert _count(temp_db) == 1
    assert _row(temp_db)["summary"] is None


def test_rehydrate_slimmed_row_in_place_no_duplicate(temp_db):
    """Item 6: a rebuild that re-derives a SLIMMED row must UPDATE it, not INSERT a second
    reference row. The documents table has no UNIQUE(thread_id, filename) constraint, so the old
    path (add_document → save_document each cycle) accumulated duplicate rows. restore_document_
    derived — the method the rebuild now uses whenever a row already EXISTS — re-hydrates in place.
    Running it twice (two retention/rebuild cycles) still leaves exactly one row, summary restored."""
    _save(temp_db)
    temp_db.conn.execute("UPDATE documents SET created_at = datetime('now', '-200 days')")
    temp_db.delete_old_documents(days=90)
    assert _count(temp_db) == 1 and _row(temp_db)["summary"] is None  # slimmed

    for _ in range(2):
        updated = temp_db.restore_document_derived(
            "C1:100.0", "doc.pdf",
            summary="re-derived summary",
            page_structure={"pages": [{"n": 1, "text": "page one body"}]},
            total_pages=1, size_bytes=4096, message_ts="100.0")
        assert updated == 1  # matched and updated the preserved row (never inserted)

    assert _count(temp_db) == 1                    # exactly one row — no duplicate accumulation
    row = _row(temp_db)
    assert row["summary"] == "re-derived summary"   # derived summary restored
    assert row["page_structure"] is not None
    assert row["file_id"] == "F1"                   # Slack ref preserved throughout


def test_restore_returns_zero_when_no_row_exists(temp_db):
    """A genuinely legacy document (no row at all) matches nothing, so the rebuild falls back to
    inserting a fresh row via add_document. restore_document_derived signals that with a 0 return
    and touches nothing."""
    updated = temp_db.restore_document_derived("C1:100.0", "never-stored.pdf", summary="x")
    assert updated == 0
    assert _count(temp_db) == 0


@pytest.mark.asyncio
async def test_slimmed_row_still_readable_via_read_document(temp_db, monkeypatch):
    """The whole point of slimming: a slimmed row (summary gone) is still fully readable because
    read_document re-resolves the Slack ref and re-extracts on demand — the summary is never on
    the read path."""
    from message_processor import document_tools
    from tool_registry import ToolContext

    _save(temp_db)
    temp_db.conn.execute("UPDATE documents SET created_at = datetime('now', '-200 days')")
    temp_db.delete_old_documents(days=90)
    assert _row(temp_db)["summary"] is None  # confirm it is slimmed

    # Slack still holds the file: download + re-extract succeed off the preserved ref.
    client = MagicMock()
    client.download_file = AsyncMock(return_value=b"%PDF-1.5 fake pdf bytes")
    monkeypatch.setattr(
        document_tools._document_handler, "safe_extract_content_async",
        AsyncMock(return_value={"content": "RE-EXTRACTED FROM SLACK ON DEMAND"}))

    ctx = ToolContext(channel_id="C1", thread_ts="100.0", client=client, db=temp_db)
    result = await document_tools.execute_read_document(ctx, {"filename": "doc.pdf"})

    assert result["ok"] is True
    assert result["content"] == "RE-EXTRACTED FROM SLACK ON DEMAND"
    client.download_file.assert_awaited()  # re-resolved via the preserved url_private/file_id
