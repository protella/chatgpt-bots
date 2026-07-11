"""Phase D2 — document architecture: summary+ref rows, read_document tool,
native PDF input, no content at rest.

Covers: the content-drop migration, attach-time summary flow, gap-honest
prompt, labeled injection, read_document (query/offset/LRU/file_deleted),
native input_file gating, spreadsheet schema-first, and guidance presence.
"""
import inspect
import shutil

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
# F26: OCR text for image-only / scanned PDFs (later-turn read path)
# ---------------------------------------------------------------------------

def _poppler_missing():
    return shutil.which("pdftoppm") is None and shutil.which("pdfinfo") is None


class TestScannedPdfOcr:
    """OCR text is orthogonal to the vision page-image path: read_document wants text,
    the local attach path wants both. All degradations fall back, never raise."""

    def _image_based_handler(self, total_pages=2):
        from document_handler import DocumentHandler
        h = DocumentHandler()
        base = {"content": "", "total_pages": total_pages, "pages": [], "format": "pdf"}
        return h, base

    def test_ocr_text_builds_page_structured_content(self, monkeypatch):
        h, base = self._image_based_handler(total_pages=2)
        monkeypatch.setattr(config, "enable_pdf_ocr", True)
        monkeypatch.setattr(config, "ocr_max_pages", 20)
        with patch.object(h, "_parse_pdf_with_pdfplumber", return_value=dict(base)), \
             patch.object(h, "_is_image_based_pdf", return_value=True), \
             patch("document_handler.convert_from_bytes",
                   return_value=[MagicMock(), MagicMock()]), \
             patch("pytesseract.image_to_string", side_effect=["HELLO WORLD", "SECOND PAGE"]):
            out = h.parse_pdf_structured(b"%PDF-fake", "scan.pdf",
                                         ocr_images=False, ocr_text=True)
        assert out["ocr_text_used"] is True
        assert "[Page 1]" in out["content"] and "HELLO WORLD" in out["content"]
        assert "[Page 2]" in out["content"] and "SECOND PAGE" in out["content"]
        assert [(p["page"], p.get("ocr")) for p in out["pages"]] == [(1, True), (2, True)]

    def test_truncation_note_is_loud_when_pages_exceed_cap(self, monkeypatch):
        # Document has 5 pages but only 2 get OCR'd (cap) — the note must say so.
        h, base = self._image_based_handler(total_pages=5)
        monkeypatch.setattr(config, "enable_pdf_ocr", True)
        monkeypatch.setattr(config, "ocr_max_pages", 2)
        with patch.object(h, "_parse_pdf_with_pdfplumber", return_value=dict(base)), \
             patch.object(h, "_is_image_based_pdf", return_value=True), \
             patch("document_handler.convert_from_bytes",
                   return_value=[MagicMock(), MagicMock()]), \
             patch("pytesseract.image_to_string", side_effect=["A", "B"]):
            out = h.parse_pdf_structured(b"%PDF-fake", "big-scan.pdf",
                                         ocr_images=False, ocr_text=True)
        assert out["ocr_text_used"] is True
        assert "first 2 of 5" in out["content"]

    def test_tesseract_not_found_falls_back_no_raise(self, monkeypatch):
        import pytesseract
        h, base = self._image_based_handler(total_pages=1)
        monkeypatch.setattr(config, "enable_pdf_ocr", True)
        with patch.object(h, "_parse_pdf_with_pdfplumber", return_value=dict(base)), \
             patch.object(h, "_is_image_based_pdf", return_value=True), \
             patch("document_handler.convert_from_bytes", return_value=[MagicMock()]), \
             patch("pytesseract.image_to_string",
                   side_effect=pytesseract.TesseractNotFoundError()):
            out = h.parse_pdf_structured(b"%PDF-fake", "scan.pdf",
                                         ocr_images=False, ocr_text=True)
        assert "ocr_text_used" not in out
        assert "scanned document" in out["content"]

    def test_enable_pdf_ocr_false_skips_ocr_entirely(self, monkeypatch):
        h, base = self._image_based_handler(total_pages=2)
        monkeypatch.setattr(config, "enable_pdf_ocr", False)
        with patch.object(h, "_parse_pdf_with_pdfplumber", return_value=dict(base)), \
             patch.object(h, "_is_image_based_pdf", return_value=True), \
             patch.object(h, "ocr_pdf_pages") as ocr_spy:
            out = h.parse_pdf_structured(b"%PDF-fake", "scan.pdf",
                                         ocr_images=False, ocr_text=True)
        ocr_spy.assert_not_called()
        assert "ocr_text_used" not in out

    def test_local_attach_path_emits_both_page_images_and_ocr_text(self, monkeypatch):
        # ocr_images=True AND ocr_text=True (big-file local route): both must be present.
        h, base = self._image_based_handler(total_pages=1)
        monkeypatch.setattr(config, "enable_pdf_ocr", True)
        with patch.object(h, "_parse_pdf_with_pdfplumber", return_value=dict(base)), \
             patch.object(h, "_is_image_based_pdf", return_value=True), \
             patch.object(h, "convert_pdf_to_images",
                          return_value=[{"page": 1, "base64_data": "x", "mimetype": "image/png"}]), \
             patch("document_handler.convert_from_bytes", return_value=[MagicMock()]), \
             patch("pytesseract.image_to_string", return_value="INVOICE TOTAL 500"):
            out = h.parse_pdf_structured(b"%PDF-fake", "scan.pdf",
                                         ocr_images=True, ocr_text=True)
        assert out.get("page_images")  # vision path preserved
        assert out["ocr_text_used"] is True
        assert "INVOICE TOTAL 500" in out["content"]

    @pytest.mark.asyncio
    async def test_read_document_returns_ocr_text_for_scan(self):
        # read_document path: a scanned doc that yields nothing locally now comes back ok.
        import message_processor.document_tools as dt
        dt._extraction_cache = dt.ExtractionCache(5)
        captured = {}

        async def fake_extract(data, mime, name, ocr_images=True, ocr_text=False):
            captured["ocr_images"] = ocr_images
            captured["ocr_text"] = ocr_text
            return {"content": "[Page 1]\nSCANNED CONTRACT VALUE 4.2M"}

        with patch.object(dt._document_handler, "safe_extract_content_async",
                          side_effect=fake_extract):
            out = await dt.execute_read_document(_ctx([dict(DOC_ROW)]), {"file_id": "F42"})
        assert out["ok"] is True
        assert "SCANNED CONTRACT VALUE 4.2M" in out["content"]
        # Tool wants text, not page images.
        assert captured == {"ocr_images": False, "ocr_text": True}

    @pytest.mark.skipif(_poppler_missing() or shutil.which("tesseract") is None,
                        reason="requires poppler-utils and tesseract-ocr binaries")
    def test_real_ocr_end_to_end(self):
        # Build a genuine image-only PDF (no text layer) with PIL and OCR it for real.
        from io import BytesIO
        from PIL import Image, ImageDraw, ImageFont
        from document_handler import DocumentHandler

        img = Image.new("RGB", (1240, 1754), "white")
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 90)
        except OSError:
            font = ImageFont.load_default()
        draw.text((150, 700), "PINEAPPLE", fill="black", font=font)
        buf = BytesIO()
        img.save(buf, format="PDF")  # single page, no text layer

        out = DocumentHandler().parse_pdf_structured(
            buf.getvalue(), "scan.pdf", ocr_images=False, ocr_text=True)
        assert out.get("ocr_text_used") is True
        assert "PINEAPPLE" in out["content"]


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

def _ctx(docs, download=b"%PDF", channel="C1", thread="111.0", channel_docs=None):
    from tool_registry import ToolContext
    db = MagicMock()
    db.get_thread_documents_async = AsyncMock(return_value=docs)
    # F22: channel-wide fallback lookup. Defaults to the thread docs so tests that never
    # trigger the fallback (thread resolve hits) behave unchanged; pass channel_docs to
    # exercise the cross-thread path.
    db.get_channel_documents_async = AsyncMock(
        return_value=docs if channel_docs is None else channel_docs)
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
    async def test_query_no_match_returns_full_small_doc(self):
        # F25: a literal-substring miss must not dead-end — a small doc comes back whole
        # so one call answers the question even when the query phrasing doesn't match.
        dt = self._fresh_cache()
        text = "Backup vendor: Meridian Foods (code MF-2210)"
        with patch.object(dt._document_handler, "safe_extract_content_async",
                          AsyncMock(return_value={"content": text})):
            out = await dt.execute_read_document(_ctx([dict(DOC_ROW)]),
                                                 {"file_id": "F42", "query": "backup vendor code"})
        assert out["ok"] is True and out["matches"] == []
        assert out["content"] == text
        assert out["has_more"] is False
        assert "FULL document" in out["note"]

    @pytest.mark.asyncio
    async def test_query_no_match_returns_doc_start_with_navigation(self):
        # F25: on a large doc, the miss includes the START slice + next_offset.
        dt = self._fresh_cache()
        text = "nothing to see " * 500  # > SLICE_CHARS
        with patch.object(dt._document_handler, "safe_extract_content_async",
                          AsyncMock(return_value={"content": text})):
            out = await dt.execute_read_document(_ctx([dict(DOC_ROW)]),
                                                 {"file_id": "F42", "query": "unicorns"})
        assert out["ok"] is True and out["matches"] == []
        assert out["content"] == text[:dt.SLICE_CHARS]
        assert out["has_more"] is True and out["next_offset"] == dt.SLICE_CHARS
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

    # ---- F22: channel-wide document access ----

    @pytest.mark.asyncio
    async def test_channel_wide_hit_returns_content_and_origin(self):
        dt = self._fresh_cache()
        # Current thread has NO documents; the file lives in another thread of the channel.
        ctx = _ctx([], channel_docs=[dict(DOC_ROW)])
        with patch.object(dt._document_handler, "safe_extract_content_async",
                          AsyncMock(return_value={"content": "Q3 revenue was 4.2M"})):
            out = await dt.execute_read_document(ctx, {"file_id": "F42"})
        assert out["ok"] is True
        assert out["content"] == "Q3 revenue was 4.2M"
        assert out["origin"] == "shared in another conversation in this channel"

    @pytest.mark.asyncio
    async def test_in_thread_wins_over_channel_same_name(self):
        dt = self._fresh_cache()
        # Same filename in both threads: the current thread's copy must win, with no
        # channel-wide fallback and no origin note.
        thread_doc = {**DOC_ROW, "file_id": "F_THREAD"}
        channel_doc = {**DOC_ROW, "file_id": "F_CHANNEL"}
        ctx = _ctx([thread_doc], channel_docs=[channel_doc, thread_doc])
        with patch.object(dt._document_handler, "safe_extract_content_async",
                          AsyncMock(return_value={"content": "x"})):
            out = await dt.execute_read_document(ctx, {"filename": "q3.pdf"})
        assert out["ok"] is True
        assert "origin" not in out
        ctx.db.get_channel_documents_async.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_miss_lists_channel_wide_known(self):
        dt = self._fresh_cache()
        channel_docs = [dict(DOC_ROW), {**DOC_ROW, "filename": "plan.docx", "file_id": "F99"}]
        ctx = _ctx([], channel_docs=channel_docs)
        out = await dt.execute_read_document(ctx, {"filename": "nope.docx"})
        assert out["ok"] is False and out["error"] == "document_not_found"
        assert out["known_documents"] == ["q3.pdf", "plan.docx"]

    @pytest.mark.asyncio
    async def test_no_cross_channel_leak(self):
        dt = self._fresh_cache()
        # From channel C2, the channel-wide lookup returns nothing (the C1 doc is scoped
        # out by the DB prefix match) — the tool must not surface it.
        ctx = _ctx([], channel="C2", channel_docs=[])
        out = await dt.execute_read_document(ctx, {"file_id": "F42"})
        assert out["ok"] is False and out["error"] == "document_not_found"
        assert out["known_documents"] == []

    @pytest.mark.asyncio
    async def test_thread_scoped_path_byte_identical(self):
        dt = self._fresh_cache()
        text = "The Q3 revenue was 4.2M. " * 400
        # In-thread doc: result must be exactly the pre-F22 shape (no origin key).
        ctx = _ctx([dict(DOC_ROW)], channel_docs=[dict(DOC_ROW)])
        with patch.object(dt._document_handler, "safe_extract_content_async",
                          AsyncMock(return_value={"content": text})):
            out = await dt.execute_read_document(ctx, {"file_id": "F42"})
        assert "origin" not in out
        assert out["content"] == text[:dt.SLICE_CHARS]
        ctx.db.get_channel_documents_async.assert_not_awaited()

    def test_schema_description_mentions_channel_scope(self):
        from message_processor.document_tools import get_read_document_schema
        desc = get_read_document_schema()["description"]
        assert "ANYWHERE in this channel" in desc
        assert "current conversation is checked first" in desc
        # F25: the model must know no in-context summary is required and that attachment
        # notes / history are valid filename sources — its absence made it skip the tool
        # and wrongly declare a cross-thread file unreachable (live 2026-07-11).
        assert "No summary needs to be in context" in desc
        assert "attachment note" in desc

    def test_not_found_hint_teaches_channel_wide_reach(self):
        # F25: the miss hint must not imply a summary is required.
        import inspect
        from message_processor import document_tools as dt
        src = inspect.getsource(dt.execute_read_document)
        assert "any file shared in this channel is reachable" in src

    def test_guidance_teaches_cross_thread_reads(self):
        # F25: LOCAL_TOOLS_GUIDANCE explicitly licenses cross-thread reads by filename.
        from prompts import LOCAL_TOOLS_GUIDANCE
        assert "ANOTHER thread of this channel is readable too" in LOCAL_TOOLS_GUIDANCE
        assert "never declare a channel file unreachable without trying it" in LOCAL_TOOLS_GUIDANCE

    @pytest.mark.asyncio
    async def test_channel_wide_prefix_isolation_at_db(self, tmp_path):
        # DB-level proof of the privacy boundary: a doc in C1 is invisible to C2.
        import sqlite3
        from database import DatabaseManager
        db = DatabaseManager("test")
        db.db_path = f"{tmp_path}/t.db"
        db.conn = sqlite3.connect(db.db_path, check_same_thread=False, isolation_level=None)
        db.conn.row_factory = sqlite3.Row
        db.init_schema()
        db.save_document("C1:111.0", "a.pdf", "application/pdf", file_id="FA")
        db.save_document("C1:222.0", "b.pdf", "application/pdf", file_id="FB")
        db.save_document("C2:333.0", "c.pdf", "application/pdf", file_id="FC")
        db.conn.close()
        c1 = await db.get_channel_documents_async("C1")
        c2 = await db.get_channel_documents_async("C2")
        assert sorted(d["filename"] for d in c1) == ["a.pdf", "b.pdf"]
        assert [d["filename"] for d in c2] == ["c.pdf"]
