"""Canvases read as documents (spec: cancel_and_canvas_round, fix 3).

A canvas is a file whose bytes are HTML, and until now every document path downloaded with
``allow_html=False`` — so the downloader rejected the canvas body as a sign-in page and the model
was told the file had been deleted. These tests pin the three halves of the fix: the parser that
turns canvas HTML into markdown (and REFUSES a real sign-in page), the ``allow_html`` flag
reaching the downloader from every path that reads a document, and the mount catalog leaving
canvases out.

The refusal shape is the load-bearing part. ``parse_canvas`` must never raise: a raise lands in
``force_text_extraction``, which returns the sign-in page as truthy "partial" content, and every
caller reads truthy content as success — so the bot would quote Slack's login screen at the user.
"""
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from message_processor.canvas_content import CANVAS_MARKER, CANVAS_MIMETYPE, html_to_markdown
from message_processor.ingestion.document_handler import (
    MIME_TYPE_HANDLERS,
    SUPPORTED_DOCUMENT_MIMETYPES,
    DocumentHandler,
)

pytestmark = pytest.mark.unit


CANVAS_HTML = (
    '<div class="quip-canvas-content">'
    "<h2>Launch checklist</h2>"
    "<p>Ship on <strong>Friday</strong>.</p>"
    "<div data-section-style='7'><ul>"
    "<li>write the runbook</li>"
    "<li class='checked'>book the window</li>"
    "</ul></div>"
    "</div>"
).encode()

# What Slack actually serves when the token is wrong: a full login page, no canvas marker.
LOGIN_PAGE = (
    "<!DOCTYPE html><html><head><title>Sign in to Slack</title></head>"
    "<body><form action='/signin'><input name='email'></form></body></html>"
).encode()


# --------------------------------------------------------------------------- the parser

class TestParseCanvas:
    def test_canvas_html_becomes_markdown(self):
        result = DocumentHandler().parse_canvas(CANVAS_HTML, "Launch checklist")
        assert "error" not in result
        md = result["content"]
        assert "## Launch checklist" in md
        assert "**Friday**" in md
        assert "- [ ] write the runbook" in md
        assert "- [x] book the window" in md

    def test_a_sign_in_page_is_an_empty_content_error(self):
        """Pinned as an exact literal, not a prefix: an added key or reworded message changes
        what the model is told, and empty content is the only thing every caller reads as
        failure."""
        result = DocumentHandler().parse_canvas(LOGIN_PAGE, "Launch checklist")
        assert result == {
            "content": "",
            "error": "canvas_unavailable: Slack returned a sign-in page instead of the canvas",
            "format": "error",
        }

    def test_a_converter_crash_is_an_empty_content_error(self):
        """The converter is third-party HTML parsing — if it ever throws, the failure must stay
        a failure rather than escaping into the raw-text fallback."""
        handler = DocumentHandler()
        with patch("message_processor.ingestion.document_handler.html_to_markdown", side_effect=RuntimeError("bs4 blew up")):
            result = handler.parse_canvas(CANVAS_HTML, "Launch checklist")
        assert result == {
            "content": "",
            "error": "canvas_conversion_failed: bs4 blew up",
            "format": "error",
        }

    def test_a_sign_in_page_never_reaches_force_text_extraction(self):
        """The whole point of returning (rather than raising): force_text_extraction would hand
        back the login page as truthy content, which every caller reads as success."""
        handler = DocumentHandler()
        with patch.object(handler, "force_text_extraction",
                          side_effect=AssertionError("fallback must not run")) as fallback:
            result = handler.safe_extract_content(LOGIN_PAGE, CANVAS_MIMETYPE, "Spec")
        fallback.assert_not_called()
        assert result == {
            "content": "",
            "error": "canvas_unavailable: Slack returned a sign-in page instead of the canvas",
            "format": "error",
            "filename": "Spec",
            "mime_type": CANVAS_MIMETYPE,
            "size_bytes": len(LOGIN_PAGE),
        }

    def test_a_converter_crash_never_reaches_force_text_extraction(self):
        handler = DocumentHandler()
        with patch("message_processor.ingestion.document_handler.html_to_markdown", side_effect=RuntimeError("bs4 blew up")), \
                patch.object(handler, "force_text_extraction",
                             side_effect=AssertionError("fallback must not run")) as fallback:
            result = handler.safe_extract_content(CANVAS_HTML, CANVAS_MIMETYPE, "Spec")
        fallback.assert_not_called()
        assert result == {
            "content": "",
            "error": "canvas_conversion_failed: bs4 blew up",
            "format": "error",
            "filename": "Spec",
            "mime_type": CANVAS_MIMETYPE,
            "size_bytes": len(CANVAS_HTML),
        }


class TestCanvasRouting:
    def test_canvas_mimetype_is_supported_and_has_a_handler(self):
        assert CANVAS_MIMETYPE in SUPPORTED_DOCUMENT_MIMETYPES
        assert MIME_TYPE_HANDLERS[CANVAS_MIMETYPE] == "parse_canvas"
        assert callable(getattr(DocumentHandler(), "parse_canvas"))

    def test_canvas_is_admitted_as_a_document(self):
        assert DocumentHandler().is_document_file("Launch checklist", CANVAS_MIMETYPE)

    def test_mimetype_beats_a_misleading_extension(self):
        """A canvas keeps whatever name its author typed. Extension-first routing would send
        `notes.md` to parse_text and dump raw HTML at the model."""
        result = DocumentHandler().safe_extract_content(CANVAS_HTML, CANVAS_MIMETYPE, "notes.md")
        assert "## Launch checklist" in result["content"]
        assert CANVAS_MARKER not in result["content"]

    def test_a_real_markdown_file_still_routes_to_parse_text(self):
        result = DocumentHandler().safe_extract_content(b"# plain\n", "text/markdown", "notes.md")
        assert result["content"] == "# plain\n"


# --------------------------------------------------------------------------- allow_html plumbing

class TestDownloadSignatures:
    def test_both_base_client_methods_declare_allow_html(self):
        from message_processor.client_contract import BaseClient

        for name in ("download_file", "download_file_async"):
            params = inspect.signature(getattr(BaseClient, name)).parameters
            assert "allow_html" in params, f"{name} must accept allow_html"
            assert params["allow_html"].default is False

    @pytest.mark.asyncio
    async def test_slack_async_wrapper_forwards_the_flag(self):
        from slack_client.base import SlackBot

        download_file = AsyncMock(return_value=b"<div>")
        stub = SimpleNamespace(download_file=download_file)
        out = await SlackBot.download_file_async(stub, "https://files.slack.com/x", "F1",
                                                 allow_html=True)
        assert out == b"<div>"
        download_file.assert_awaited_once_with("https://files.slack.com/x", "F1",
                                               allow_html=True, max_bytes=None)

    @pytest.mark.asyncio
    async def test_slack_async_wrapper_defaults_the_flag_off(self):
        from slack_client.base import SlackBot

        download_file = AsyncMock(return_value=b"bytes")
        stub = SimpleNamespace(download_file=download_file)
        await SlackBot.download_file_async(stub, "https://files.slack.com/x")
        download_file.assert_awaited_once_with("https://files.slack.com/x", None,
                                               allow_html=False, max_bytes=None)


CANVAS_ROW = {"filename": "Launch checklist", "mime_type": CANVAS_MIMETYPE,
              "file_id": "F90", "url_private": "https://files.slack.com/canvas",
              "summary": "the checklist"}
PDF_ROW = {"filename": "q3.pdf", "mime_type": "application/pdf",
           "file_id": "F42", "url_private": "https://files.slack.com/q3.pdf",
           "summary": "sum"}


def _read_ctx(row, download):
    from message_processor.tool_registry import ToolContext

    db = MagicMock()
    db.get_thread_documents_async = AsyncMock(return_value=[dict(row)])
    db.get_channel_documents_async = AsyncMock(return_value=[dict(row)])
    client = MagicMock()
    client.download_file = AsyncMock(return_value=download)
    return ToolContext(channel_id="C1", thread_ts="111.0", trigger_ts="1",
                       action_token=None, client=client, db=db, user_id="U1")


class TestReadDocumentPath:
    def _fresh_cache(self):
        import message_processor.document_tools as dt
        dt._extraction_cache = dt.ExtractionCache(5)
        return dt

    @pytest.mark.asyncio
    async def test_read_document_asks_for_html_on_a_canvas(self):
        dt = self._fresh_cache()
        ctx = _read_ctx(CANVAS_ROW, CANVAS_HTML)
        out = await dt.execute_read_document(ctx, {"file_id": "F90"})
        assert ctx.client.download_file.await_args.kwargs["allow_html"] is True
        assert out["ok"] is True
        assert "- [x] book the window" in out["content"]

    @pytest.mark.asyncio
    async def test_read_document_leaves_the_guard_on_for_everything_else(self):
        dt = self._fresh_cache()
        ctx = _read_ctx(PDF_ROW, b"%PDF")
        with patch.object(dt._document_handler, "safe_extract_content_async",
                          AsyncMock(return_value={"content": "prose"})):
            await dt.execute_read_document(ctx, {"file_id": "F42"})
        assert ctx.client.download_file.await_args.kwargs["allow_html"] is False

    @pytest.mark.asyncio
    async def test_read_document_reports_a_sign_in_page_as_a_failure(self):
        """Not "here is a login form" — the empty-content error result has to surface as
        extraction_failed, carrying the canvas_unavailable detail."""
        dt = self._fresh_cache()
        ctx = _read_ctx(CANVAS_ROW, LOGIN_PAGE)
        out = await dt.execute_read_document(ctx, {"file_id": "F90"})
        assert out["ok"] is False
        assert out["error"] == "extraction_failed"
        assert "canvas_unavailable" in out["detail"]


class TestArrivalPath:
    """The arrival pipeline (`_process_attachments`) — the path a freshly shared canvas takes."""

    def _proc(self):
        from message_processor.utilities import MessageUtilitiesMixin

        class _Proc(MessageUtilitiesMixin):
            def __init__(self):
                self.db = None
                self.document_handler = DocumentHandler()
                self.image_url_handler = MagicMock()
                self.image_url_handler.max_image_size = 20 * 1024 * 1024
                self.image_url_handler.process_urls_from_text = AsyncMock(return_value=([], []))
                self.thread_manager = MagicMock()
                for name in ("log_info", "log_debug", "log_warning", "log_error"):
                    setattr(self, name, MagicMock())

        return _Proc()

    def _msg(self, attachment):
        return SimpleNamespace(attachments=[attachment], text="",
                               channel_id="C1", thread_id="123.456",
                               metadata={"ts": "123.456"})

    @pytest.mark.asyncio
    async def test_arrival_asks_for_html_on_a_canvas(self):
        proc = self._proc()
        client = MagicMock()
        client.download_file = AsyncMock(return_value=CANVAS_HTML)
        att = {"type": "file", "name": "Launch checklist", "mimetype": CANVAS_MIMETYPE,
               "url": "https://files.slack.com/canvas", "id": "F90", "size": len(CANVAS_HTML)}

        with patch.object(proc, "_summarize_document_for_attach",
                          AsyncMock(return_value="a checklist")):
            _images, docs, unsupported = await proc._process_attachments(self._msg(att), client)

        assert client.download_file.await_args.kwargs["allow_html"] is True
        assert unsupported == []
        assert len(docs) == 1
        assert "- [x] book the window" in docs[0]["content"]

    @pytest.mark.asyncio
    async def test_arrival_reports_a_sign_in_page_as_a_failed_attachment(self):
        """The failure has to reach the model as a failed attachment, not vanish. Empty content
        is what carries it: were the login page returned as truthy "partial" text, this canvas
        would join document_inputs and the bot would answer from Slack's sign-in screen."""
        proc = self._proc()
        client = MagicMock()
        client.download_file = AsyncMock(return_value=LOGIN_PAGE)
        att = {"type": "file", "name": "Launch checklist", "mimetype": CANVAS_MIMETYPE,
               "url": "https://files.slack.com/canvas", "id": "F90", "size": len(LOGIN_PAGE)}

        _images, docs, unsupported = await proc._process_attachments(self._msg(att), client)

        assert client.download_file.await_args.kwargs["allow_html"] is True
        assert docs == []
        assert unsupported == [{"name": "Launch checklist", "type": "file",
                                "mimetype": CANVAS_MIMETYPE}]

    @pytest.mark.asyncio
    async def test_arrival_leaves_the_guard_on_for_everything_else(self):
        proc = self._proc()
        client = MagicMock()
        client.download_file = AsyncMock(return_value=b"plain text")
        att = {"type": "file", "name": "notes.txt", "mimetype": "text/plain",
               "url": "https://files.slack.com/notes.txt", "id": "F91", "size": 10}

        with patch.object(proc, "_summarize_document_for_attach",
                          AsyncMock(return_value="notes")):
            await proc._process_attachments(self._msg(att), client)

        assert client.download_file.await_args.kwargs["allow_html"] is False


class TestColdRebuildPath:
    @pytest.mark.asyncio
    async def test_rebuild_asks_for_html_on_a_canvas(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
        from message_processor.client_contract import Message
        from database import DatabaseManager
        from message_processor.thread_management import ThreadManagementMixin
        from message_processor.utilities import MessageUtilitiesMixin
        from message_processor.thread_manager import AsyncThreadStateManager

        class _Proc(ThreadManagementMixin, MessageUtilitiesMixin):
            def __init__(self, db):
                self.db = db
                self.thread_manager = AsyncThreadStateManager(db=db)
                self.openai_client = None
                self.document_handler = MagicMock()
                self.document_handler.is_document_file = MagicMock(return_value=True)
                self.document_handler.safe_extract_content_async = AsyncMock(return_value={})

            def log_info(self, *a, **k): pass
            log_debug = log_warning = log_error = log_info

            def _update_status(self, *a, **k): pass

        db = DatabaseManager(platform="slack")
        try:
            proc = _Proc(db)
            att = {"type": "file", "name": "Launch checklist", "mimetype": CANVAS_MIMETYPE,
                   "url": "https://files.slack.com/canvas", "id": "F90"}
            history = [Message(text="here it is", user_id="U1", channel_id="C1",
                               thread_id="100.0", attachments=[att],
                               metadata={"ts": "100.0", "is_bot": False,
                                         "sender_type": "human", "username": "Dana"})]
            client = MagicMock()
            client.get_thread_history = AsyncMock(return_value=history)
            client.name = "slack"
            client.user_cache = {}
            client.bot_user_id = "UBOT"
            client.download_file = AsyncMock(return_value=CANVAS_HTML)
            incoming = Message(text="what does it say?", user_id="U1", channel_id="C1",
                               thread_id="100.0", attachments=[], metadata={"ts": "200.0"})

            await proc._get_or_rebuild_thread_state(incoming, client)
        finally:
            db.conn.close()

        assert client.download_file.await_args.kwargs["allow_html"] is True


class TestAmbientPath:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("mimetype,expected", [
        (CANVAS_MIMETYPE, True),
        ("application/pdf", False),
        ("image/png", False),
    ])
    async def test_ambient_download_flags_only_canvases(self, mimetype, expected):
        from message_processor.ambient_memory import AmbientArtifactService, _Job

        client = MagicMock()
        client.download_file = AsyncMock(return_value=b"bytes")
        stub = SimpleNamespace(_client=client,
                               config=SimpleNamespace(ambient_file_max_bytes=8 * 1024 * 1024))
        job = _Job(kind="file", channel_id="C1", source_ts="1", conversation_ts="1",
                   ref="F90", url="https://files.slack.com/canvas", filename="Launch checklist",
                   mimetype=mimetype)

        await AmbientArtifactService._download(stub, job)

        assert client.download_file.await_args.kwargs["allow_html"] is expected


# --------------------------------------------------------------------------- mount catalog

class TestMountCatalog:
    @pytest.mark.asyncio
    async def test_a_canvas_is_not_mountable(self):
        """Raw canvas HTML is not a useful build input, and mounting would need its own flag
        plumbing. The job model reads canvases through history / read_document instead."""
        from message_processor import thread_files

        db = MagicMock()
        db.find_thread_images_async = AsyncMock(return_value=[])
        db.get_thread_documents_async = AsyncMock(return_value=[
            {"id": 1, "filename": "Launch checklist", "mime_type": CANVAS_MIMETYPE,
             "file_id": "F90", "url_private": "https://files.slack.com/canvas",
             "summary": "the checklist"},
            {"id": 2, "filename": "q3.pdf", "mime_type": "application/pdf",
             "file_id": "F42", "url_private": "https://files.slack.com/q3.pdf",
             "summary": "sum"},
        ])

        entries = await thread_files.build_catalog(db, "C1:111.0")

        assert [e["filename"] for e in entries] == ["q3.pdf"]


# --------------------------------------------------------------------------- module move

def test_canvas_tools_still_exposes_the_converter_under_its_old_names():
    """`tests/unit/test_canvas_tools.py` imports `canvas_tools._html_to_markdown` and must keep
    passing unchanged — the converter moved, the names it was reached by did not."""
    from message_processor import canvas_tools

    assert canvas_tools._html_to_markdown is html_to_markdown
    assert canvas_tools._CANVAS_MARKER == CANVAS_MARKER
