"""Phase D2 — document architecture: summary+ref rows, read_document tool,
native PDF input, no content at rest.

Covers: the content-drop migration, attach-time summary flow, gap-honest
prompt, labeled injection, read_document (query/offset/LRU/file_deleted),
native input_file gating, spreadsheet schema-first, and guidance presence.
"""
import asyncio
import base64
import sqlite3
import inspect

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from config import config


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def _make_legacy_db(tmp_path, monkeypatch):
    """Simulate a pre-D2 database: fresh schema + re-added content column + a legacy row."""
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    from database import DatabaseManager
    db = DatabaseManager(platform="slack")
    db.conn.execute("ALTER TABLE documents ADD COLUMN content TEXT")
    db.get_or_create_thread("C1:111.0", "C1")
    db.conn.execute(
        "INSERT INTO documents (thread_id, filename, mime_type, content) VALUES (?,?,?,?)",
        ("C1:111.0", "report.pdf", "application/pdf", "FULL TEXT " * 500),
    )
    db.conn.close()


class TestContentDropMigration:
    def _open(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
        from database import DatabaseManager
        return DatabaseManager(platform="slack")

    def test_migration_drops_content_and_synthesizes_summary(self, tmp_path, monkeypatch):
        _make_legacy_db(tmp_path, monkeypatch)
        db = self._open(tmp_path, monkeypatch)
        cols = [c[1] for c in db.conn.execute("PRAGMA table_info(documents)").fetchall()]
        assert "content" not in cols
        assert "summary" in cols and "file_id" in cols and "url_private" in cols and "size_bytes" in cols
        row = db.conn.execute("SELECT summary FROM documents").fetchone()
        assert row[0].startswith("[excerpt of original")
        assert "FULL TEXT" in row[0]
        db.close()

    def test_migration_creates_tagged_backup(self, tmp_path, monkeypatch):
        _make_legacy_db(tmp_path, monkeypatch)
        db = self._open(tmp_path, monkeypatch)
        backups = list((tmp_path / "backups").glob("*pre-v3-doc-content-drop*"))
        assert backups, "tagged backup missing"
        db.close()

    def test_migration_idempotent(self, tmp_path, monkeypatch):
        _make_legacy_db(tmp_path, monkeypatch)
        db = self._open(tmp_path, monkeypatch)
        db.close()
        # Second open: content column gone -> migration must not run/fail
        db2 = self._open(tmp_path, monkeypatch)
        count = db2.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        assert count == 1
        db2.close()

    def test_fresh_schema_has_no_content_column(self, tmp_path, monkeypatch):
        db = self._open(tmp_path, monkeypatch)
        cols = [c[1] for c in db.conn.execute("PRAGMA table_info(documents)").fetchall()]
        assert "content" not in cols
        assert {"summary", "file_id", "url_private", "size_bytes"} <= set(cols)
        db.close()

    def test_save_document_persists_no_content(self, tmp_path, monkeypatch):
        db = self._open(tmp_path, monkeypatch)
        db.get_or_create_thread("C1:222.0", "C1")
        db.save_document(
            thread_id="C1:222.0", filename="a.pdf", mime_type="application/pdf",
            summary="sum", file_id="F123", url_private="https://files.slack.com/a.pdf",
            size_bytes=1234, total_pages=3, message_ts="222.1",
        )
        docs = db.get_thread_documents("C1:222.0")
        assert len(docs) == 1
        assert docs[0]["summary"] == "sum"
        assert docs[0]["file_id"] == "F123"
        assert "content" not in docs[0]
        db.close()


# ---------------------------------------------------------------------------
# Prompts / guidance
# ---------------------------------------------------------------------------

class TestPromptsAndGuidance:
    def test_doc_summarization_prompt_is_gap_honest(self):
        from prompts import DOCUMENT_SUMMARIZATION_PROMPT
        assert "GAP-HONEST" in DOCUMENT_SUMMARIZATION_PROMPT
        assert "not reproduce" in DOCUMENT_SUMMARIZATION_PROMPT

    def test_tools_guidance_anti_confabulation(self):
        from prompts import LOCAL_TOOLS_GUIDANCE
        assert "read_document" in LOCAL_TOOLS_GUIDANCE
        assert "Never estimate" in LOCAL_TOOLS_GUIDANCE


# ---------------------------------------------------------------------------
# Injection rendering
# ---------------------------------------------------------------------------

class _UtilHarness:
    from message_processor.utilities import MessageUtilitiesMixin as _M
    _build_message_with_documents = _M._build_message_with_documents
    _build_user_content = _M._build_user_content
    _native_file_eligible = _M._native_file_eligible
    _build_spreadsheet_schema_block = _M._build_spreadsheet_schema_block

    def log_warning(self, *a, **k): pass
    def log_info(self, *a, **k): pass
    def log_debug(self, *a, **k): pass


class TestInjection:
    def test_injects_summary_never_content(self):
        h = _UtilHarness()
        out = h._build_message_with_documents("Peter: what's in this?", [{
            "filename": "q3.pdf", "mimetype": "application/pdf",
            "summary": "Revenue up 4%. Detailed tables in sections 3-5 not reproduced here.",
            "content": "SECRET FULL CONTENT " * 100,
            "total_pages": 12, "size_bytes": 45000, "file_id": "F42",
        }])
        assert "=== DOCUMENT SUMMARY: q3.pdf ===" in out
        assert "full content available via read_document" in out
        assert "file_id: F42" in out
        assert "SECRET FULL CONTENT" not in out
        assert out.endswith("=== END DOCUMENT SUMMARY: q3.pdf ===")

    def test_missing_summary_falls_back_to_labeled_excerpt(self):
        h = _UtilHarness()
        out = h._build_message_with_documents("t", [{
            "filename": "x.txt", "mimetype": "text/plain",
            "content": "hello world " * 500,
        }])
        assert "[excerpt of original" in out
        # excerpt is bounded
        assert len(out) < 2500

    def test_rendering_is_deterministic(self):
        h = _UtilHarness()
        doc = {"filename": "a.pdf", "mimetype": "application/pdf",
               "summary": "stable", "total_pages": 2, "file_id": "F1"}
        assert (h._build_message_with_documents("t", [dict(doc)])
                == h._build_message_with_documents("t", [dict(doc)]))

    def test_summary_blocks_are_preserved_from_compaction(self):
        from message_processor.thread_management import ThreadManagementMixin
        src = inspect.getsource(ThreadManagementMixin)
        assert "=== DOCUMENT SUMMARY:" in src


# ---------------------------------------------------------------------------
# Native input_file
# ---------------------------------------------------------------------------

class TestNativeFileInput:
    def test_eligibility_gates(self):
        h = _UtilHarness()
        with patch.object(config, "enable_native_file_input", True), \
             patch.object(config, "native_file_max_mb", 32), \
             patch.object(config, "native_file_max_pages", 100):
            assert h._native_file_eligible("application/pdf", 1024, 10) is True
            assert h._native_file_eligible("application/pdf", 33 * 1024 * 1024, 10) is False
            assert h._native_file_eligible("application/pdf", 1024, 101) is False
            assert h._native_file_eligible("application/pdf", 1024, None) is True
            assert h._native_file_eligible("text/plain", 1024, 1) is False
        with patch.object(config, "enable_native_file_input", False):
            assert h._native_file_eligible("application/pdf", 1024, 10) is False

    def test_build_user_content_includes_file_parts(self):
        h = _UtilHarness()
        parts = h._build_user_content("text", [], [{
            "type": "input_file", "filename": "a.pdf",
            "file_data": "data:application/pdf;base64,QUJD",
        }])
        assert isinstance(parts, list)
        assert parts[0] == {"type": "input_text", "text": "text"}
        assert parts[1]["type"] == "input_file"
        assert parts[1]["file_data"].startswith("data:application/pdf;base64,")

    def test_build_user_content_plain_when_no_media(self):
        h = _UtilHarness()
        assert h._build_user_content("just text", [], None) == "just text"

    def test_input_file_part_shape_matches_sdk(self):
        # Verified against openai 2.45: ResponseInputFileParam accepts
        # type/filename/file_data (base64 data URI) — no Files API upload.
        from openai.types.responses import ResponseInputFileParam
        keys = ResponseInputFileParam.__annotations__.keys()
        assert {"type", "filename", "file_data"} <= set(keys)

    def test_scanned_pdf_skips_ocr_conversion_when_native(self):
        from document_handler import DocumentHandler
        h = DocumentHandler()
        fake_result = {"content": "", "total_pages": 2}
        with patch.object(h, "_parse_pdf_with_pdfplumber", return_value=fake_result), \
             patch.object(h, "_is_image_based_pdf", return_value=True), \
             patch.object(h, "convert_pdf_to_images") as convert:
            out = h.parse_pdf_structured(b"%PDF-fake", "scan.pdf", ocr_images=False)
            convert.assert_not_called()
            assert out["is_image_based"] is True
            assert "rendered pages" in out["content"]
        with patch.object(h, "_parse_pdf_with_pdfplumber", return_value=dict(fake_result)), \
             patch.object(h, "_is_image_based_pdf", return_value=True), \
             patch.object(h, "convert_pdf_to_images", return_value=[]) as convert:
            h.parse_pdf_structured(b"%PDF-fake", "scan.pdf", ocr_images=True)
            convert.assert_called_once()


# ---------------------------------------------------------------------------
# Spreadsheet schema-first
# ---------------------------------------------------------------------------

class TestSpreadsheetSchema:
    def test_schema_block_shape(self):
        h = _UtilHarness()
        block = h._build_spreadsheet_schema_block({
            "content": "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |",
            "page_structure": {"sheets": {
                "Q3": {"rows": 900, "columns": ["region", "revenue", "units"]},
                "Q4": {"rows": 40},
            }},
        }, "fin.xlsx")
        assert "Sheets (2): Q3, Q4" in block
        assert "900 rows" in block
        assert "columns: region, revenue, units" in block
        assert "Sample (first rows):" in block
        assert "read_document" in block

    @pytest.mark.asyncio
    async def test_summarize_for_attach_uses_schema_not_model_for_sheets(self):
        from message_processor.utilities import MessageUtilitiesMixin

        class H(_UtilHarness):
            _summarize_document_for_attach = MessageUtilitiesMixin._summarize_document_for_attach
            openai_client = MagicMock()

        h = H()
        h.openai_client.create_text_response = AsyncMock(return_value="MODEL SUMMARY")
        out = await h._summarize_document_for_attach(
            {"content": "| a |", "page_structure": {"sheets": {"S1": {}}}},
            "x.xlsx", "application/vnd.ms-excel")
        assert "read_document" in out
        h.openai_client.create_text_response.assert_not_called()

    @pytest.mark.asyncio
    async def test_summarize_for_attach_model_path_and_fallback(self):
        from message_processor.utilities import MessageUtilitiesMixin

        class H(_UtilHarness):
            _summarize_document_for_attach = MessageUtilitiesMixin._summarize_document_for_attach
            openai_client = MagicMock()

        h = H()
        h.openai_client.create_text_response = AsyncMock(return_value="MODEL SUMMARY")
        out = await h._summarize_document_for_attach({"content": "prose " * 50}, "a.pdf", "application/pdf")
        assert out == "MODEL SUMMARY"
        # Failure -> labeled excerpt
        h.openai_client.create_text_response = AsyncMock(side_effect=RuntimeError("api down"))
        out = await h._summarize_document_for_attach({"content": "prose " * 50}, "a.pdf", "application/pdf")
        assert out.startswith("[excerpt of original")


# ---------------------------------------------------------------------------
# read_document tool
# ---------------------------------------------------------------------------

def _ctx(docs, download=b"%PDF", channel="C1", thread="111.0"):
    from tool_registry import ToolContext
    db = MagicMock()
    db.get_thread_documents_async = AsyncMock(return_value=docs)
    client = MagicMock()
    client.download_file = AsyncMock(return_value=download)
    return ToolContext(channel_id=channel, thread_ts=thread, trigger_ts="1",
                       action_token=None, client=client, db=db, user_id="U1")


DOC_ROW = {"filename": "q3.pdf", "mime_type": "application/pdf",
           "file_id": "F42", "url_private": "https://files.slack.com/q3.pdf",
           "summary": "sum"}


class TestReadDocumentTool:
    def _fresh_cache(self, size=5):
        import message_processor.document_tools as dt
        dt._extraction_cache = dt.ExtractionCache(size)
        return dt

    @pytest.mark.asyncio
    async def test_happy_path_offset_slice(self):
        dt = self._fresh_cache()
        text = "The Q3 revenue was 4.2M. " * 400  # > SLICE_CHARS
        with patch.object(dt._document_handler, "safe_extract_content_async",
                          AsyncMock(return_value={"content": text})):
            out = await dt.execute_read_document(_ctx([dict(DOC_ROW)]), {"file_id": "F42"})
        assert out["ok"] is True
        assert out["total_chars"] == len(text)
        assert out["has_more"] is True and out["next_offset"] == dt.SLICE_CHARS
        assert out["content"] == text[:dt.SLICE_CHARS]

    @pytest.mark.asyncio
    async def test_query_returns_context_windows(self):
        dt = self._fresh_cache()
        text = ("filler " * 300) + "the churn metric is 2.7 percent" + (" filler" * 300)
        with patch.object(dt._document_handler, "safe_extract_content_async",
                          AsyncMock(return_value={"content": text})):
            out = await dt.execute_read_document(_ctx([dict(DOC_ROW)]),
                                                 {"filename": "q3.pdf", "query": "churn metric"})
        assert out["ok"] is True
        assert len(out["matches"]) == 1
        assert "2.7 percent" in out["matches"][0]["context"]

    @pytest.mark.asyncio
    async def test_query_no_match_gives_navigation_hint(self):
        dt = self._fresh_cache()
        with patch.object(dt._document_handler, "safe_extract_content_async",
                          AsyncMock(return_value={"content": "nothing to see"})):
            out = await dt.execute_read_document(_ctx([dict(DOC_ROW)]),
                                                 {"file_id": "F42", "query": "unicorns"})
        assert out["ok"] is True and out["matches"] == []
        assert "offset" in out["note"]

    @pytest.mark.asyncio
    async def test_deleted_file(self):
        dt = self._fresh_cache()
        out = await dt.execute_read_document(_ctx([dict(DOC_ROW)], download=None),
                                             {"file_id": "F42"})
        assert out == {"ok": False, "error": "file_deleted",
                       "hint": out["hint"]}
        assert "no longer available" in out["hint"]

    @pytest.mark.asyncio
    async def test_document_not_found_lists_known(self):
        dt = self._fresh_cache()
        out = await dt.execute_read_document(_ctx([dict(DOC_ROW)]), {"filename": "nope.docx"})
        assert out["ok"] is False and out["error"] == "document_not_found"
        assert out["known_documents"] == ["q3.pdf"]

    @pytest.mark.asyncio
    async def test_no_selector_defaults_to_newest(self):
        dt = self._fresh_cache()
        docs = [dict(DOC_ROW), {**DOC_ROW, "filename": "newer.pdf", "file_id": "F43"}]
        with patch.object(dt._document_handler, "safe_extract_content_async",
                          AsyncMock(return_value={"content": "x"})):
            out = await dt.execute_read_document(_ctx(docs), {})
        assert out["filename"] == "newer.pdf"

    @pytest.mark.asyncio
    async def test_lru_hit_skips_download_and_eviction_bounds(self):
        dt = self._fresh_cache(size=2)
        ctx = _ctx([dict(DOC_ROW)])
        with patch.object(dt._document_handler, "safe_extract_content_async",
                          AsyncMock(return_value={"content": "cached text"})) as ext:
            await dt.execute_read_document(ctx, {"file_id": "F42"})
            await dt.execute_read_document(ctx, {"file_id": "F42"})
            assert ext.await_count == 1
        assert ctx.client.download_file.await_count == 1
        # Eviction bound
        dt._extraction_cache.put("A", "1")
        dt._extraction_cache.put("B", "2")
        dt._extraction_cache.put("C", "3")
        assert len(dt._extraction_cache) == 2

    @pytest.mark.asyncio
    async def test_extraction_failure_wrapped(self):
        dt = self._fresh_cache()
        with patch.object(dt._document_handler, "safe_extract_content_async",
                          AsyncMock(return_value={"content": "", "error": "corrupt"})):
            out = await dt.execute_read_document(_ctx([dict(DOC_ROW)]), {"file_id": "F42"})
        assert out["ok"] is False and out["error"] == "extraction_failed"

    def test_bytesio_source_gate(self):
        # The tool must never touch disk: no tempfile/open-for-write usage.
        import message_processor.document_tools as dt
        src = inspect.getsource(dt)
        assert "tempfile" not in src
        assert "open(" not in src.replace("pdfplumber.open(", "")

    def test_registered_when_enabled(self):
        from tool_registry import ToolRegistry
        from message_processor.document_tools import register_document_tools
        reg = ToolRegistry()
        register_document_tools(reg)
        names = [s["name"] for s in reg.schemas()]
        assert "read_document" in names
