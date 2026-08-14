"""A document our parsers cannot read must fail as a failure — never as content.

Reconstruction of a prod incident: two files named `.xlsx` whose bytes were not a ZIP failed
openpyxl and the CSV fallback, and the failure placeholder came back TRUTHY. Every caller reads
truthy content as success, so the raw bytes were base64'd to the API as an `input_file` part,
the API answered 400 `invalid_file`, and the whole turn — including an innocent PDF attached to
the same message — died with a generic "Something Went Wrong".

The contract this file pins:
  1. bytes that are not a spreadsheet in any readable format end at `format: 'error'`
  2. `format: 'error'` routes to unsupported_files and NEVER produces an input_file part
  3. formats we CAN now read (legacy .xls, an HTML table under a spreadsheet name) parse
  4. a partial extraction — real recovered text plus an error note — still passes through
  5. a failed document keeps a metadata-only row, so it still has a mount id
  6. both failure renderers say what happened instead of "unsupported file type"
"""
import os

import pytest
from unittest.mock import AsyncMock, MagicMock

from message_processor.client_contract import Message
from message_processor.ingestion.document_handler import DocumentHandler, SpreadsheetFormatMismatch
from message_processor.base import MessageProcessor
from message_processor.thread_files import build_catalog
from message_processor.utilities import MessageUtilitiesMixin
from message_processor.thread_management import ThreadManagementMixin
from message_processor.thread_manager import AsyncThreadStateManager

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


# --------------------------------------------------------------------------- harness

class _Proc(ThreadManagementMixin, MessageUtilitiesMixin):
    """Minimal processor binding the real mixins."""

    def __init__(self, db=None):
        self.db = db
        self.thread_manager = AsyncThreadStateManager(db=db)
        self.openai_client = MagicMock()
        self.document_handler = DocumentHandler()

    def log_info(self, *a, **k): pass
    def log_debug(self, *a, **k): pass
    def log_warning(self, *a, **k): pass
    def log_error(self, *a, **k): pass
    def _update_status(self, *a, **k): pass


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    from database import DatabaseManager
    db = DatabaseManager(platform="slack")
    yield db
    db.conn.close()


@pytest.fixture
def handler():
    return DocumentHandler()


def _message(name="quarterly-figures.xlsx", mimetype=XLSX_MIME):
    return Message(
        text="what does this say?", user_id="U1", channel_id="D1", thread_id="100.0",
        attachments=[{"type": "file", "name": name, "id": "F1", "mimetype": mimetype,
                      "url": "https://files.example.com/quarterly-figures.xlsx", "size": 2048}],
        metadata={"ts": "100.0", "username": "Dana Whitfield"},
    )


def _client(payload):
    client = MagicMock()
    client.download_file = AsyncMock(return_value=payload)
    return client


async def _run(proc, payload, message=None):
    """Run the attachment pipeline over one downloaded document."""
    proc.image_url_handler = MagicMock(max_image_size=20 * 1024 * 1024)
    proc.image_url_handler.process_urls_from_text = AsyncMock(return_value=([], []))
    proc._summarize_document_for_attach = AsyncMock(return_value="a summary")
    return await proc._process_attachments(message or _message(), _client(payload),
                                           code_interpreter_enabled=True)


# ------------------------------------------------- 1. unreadable bytes are a failure

@pytest.mark.parametrize("label,payload", [
    ("high-entropy binary", os.urandom(2048)),
    ("a compressed blob", b"\x1f\x8b\x08\x00" + os.urandom(512)),
    ("an image renamed .xlsx", b"\xff\xd8\xff\xe0\x00\x10JFIF" + os.urandom(512)),
    ("a ZIP that is not a workbook", b"PK\x03\x04" + b"\x00" * 256),
    ("an OLE2 file that is not a workbook", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 256),
])
def test_bytes_that_are_not_a_spreadsheet_extract_as_error(handler, label, payload):
    result = handler.safe_extract_content(payload, XLSX_MIME, "quarterly-figures.xlsx")
    assert result["format"] == "error", label
    assert result["error"]


def test_a_real_csv_under_a_spreadsheet_name_still_parses(handler):
    """The recovery path must not become a wall: BI tools rename CSVs to .xlsx constantly, and
    that file has always parsed. Latin-1 too — the encoding fallback is why it works."""
    plain = handler.safe_extract_content(b"region,units\nNorth,12\nSouth,8\n",
                                         XLSX_MIME, "figures.xlsx")
    assert plain["format"] == "csv" and "North" in plain["content"]

    latin1 = handler.safe_extract_content("region,units\nMünchen,12\n".encode("latin-1"),
                                          XLSX_MIME, "figures.xlsx")
    assert latin1["format"] == "csv" and "München" in latin1["content"]


def test_the_no_content_placeholder_is_not_mistaken_for_recovered_text(handler):
    """force_text_extraction has no failure channel — it returns a bracketed placeholder. The
    placeholder is truthy, which is the whole bug, so the prefix is what marks it as failure."""
    result = handler.safe_extract_content(b"\x00" * 64, XLSX_MIME, "figures.xlsx")
    assert result["format"] == "error"
    assert result["content"].startswith("[Unable to extract")


# --------------------------------------------------------- 2. failure never reaches the API

@pytest.mark.asyncio
async def test_a_failed_document_is_routed_to_unsupported_and_never_base64d(temp_db):
    """The incident in one assertion: no document input, no base64 payload, and a named
    failure the turn can talk about instead of a 400."""
    proc = _Proc(db=temp_db)
    _images, documents, unsupported = await _run(proc, os.urandom(2048))

    assert documents == []
    assert [f["name"] for f in unsupported] == ["quarterly-figures.xlsx"]
    assert unsupported[0]["error"] == "extraction_failed"
    assert unsupported[0]["detail"]


@pytest.mark.asyncio
async def test_an_innocent_co_attachment_survives_a_failed_one(temp_db):
    """The 400 killed a good PDF attached to the same message. Failure is now per file."""
    proc = _Proc(db=temp_db)
    proc.document_handler = MagicMock()
    proc.document_handler.max_document_size = 50 * 1024 * 1024
    proc.document_handler.is_document_file = MagicMock(return_value=True)

    async def _extract(data, mimetype, name, **kw):
        if name.endswith(".pdf"):
            return {"content": "the quarterly numbers", "total_pages": 2, "format": "pdf"}
        return {"content": "[Unable to extract text from corrupted x - binary format]",
                "format": "error", "error": "Document could not be parsed"}

    proc.document_handler.safe_extract_content_async = _extract

    message = _message()
    message.attachments.append({"type": "file", "name": "summary.pdf", "id": "F2",
                                "mimetype": "application/pdf",
                                "url": "https://files.example.com/summary.pdf", "size": 64})

    _images, documents, unsupported = await _run(proc, b"%PDF-1.4 data", message)

    assert [d["filename"] for d in documents] == ["summary.pdf"]
    assert [f["name"] for f in unsupported] == ["quarterly-figures.xlsx"]


@pytest.mark.asyncio
async def test_a_partial_extraction_is_still_a_usable_document(temp_db):
    """A partial extraction returns REAL recovered text plus an error note. It is not a
    failure and must keep behaving exactly as it does today — including its native payload."""
    proc = _Proc(db=temp_db)
    proc.document_handler = MagicMock()
    proc.document_handler.max_document_size = 50 * 1024 * 1024
    proc.document_handler.is_document_file = MagicMock(return_value=True)
    proc.document_handler.safe_extract_content_async = AsyncMock(return_value={
        "content": "Q3 revenue by region\nNorth 12\nSouth 8",
        "format": "text",
        "error": "Partial extraction - document may be malformed: bad header",
    })

    # A real workbook (ZIP magic) that merely parsed badly — the container claim holds, so the
    # native route is still open to it.
    _images, documents, unsupported = await _run(proc, b"PK\x03\x04" + b"\x00" * 256)

    assert unsupported == []
    assert documents[0]["content"].startswith("Q3 revenue")
    assert documents[0]["native"] is True and documents[0]["file_data_b64"]


@pytest.mark.asyncio
async def test_a_workbook_mimetype_that_is_not_a_workbook_is_never_uploaded(temp_db):
    """The text gate reads a bounded head, so a file can lead with CSV-shaped text for pages and
    turn to binary after it — extraction succeeds, content is real, and the mimetype would then
    make it native. The API opens bytes by their DECLARED type, so those bytes still 400.

    The upload decision is therefore made on the container claim, not on how extraction went:
    an .xlsx mimetype must be a ZIP. The recovered text still rides along as extracted text.
    """
    proc = _Proc(db=temp_db)
    prefix = b"region,units\n" + b"".join(b"North,%d\n" % i for i in range(1200))
    assert len(prefix) > 8192  # past any bounded head-scan
    payload = prefix + b"\x00\x01\x02" + os.urandom(2048)

    extracted = proc.document_handler.safe_extract_content(payload, XLSX_MIME, "quarterly.xlsx")
    assert extracted["format"] == "csv" and extracted["content"]  # extraction "succeeds"

    _images, documents, unsupported = await _run(proc, payload)

    assert unsupported == []
    assert len(documents) == 1
    assert documents[0]["native"] is False
    assert documents[0]["file_data_b64"] is None
    assert "North" in documents[0]["content"]


def test_the_container_claim_is_checked_for_every_office_mimetype():
    """A real workbook still uploads; the OLE2 family is held to its own magic."""
    from message_processor.ingestion.document_handler import container_magic_mismatch

    docx = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert container_magic_mismatch(XLSX_MIME, b"region,units\n1,2\n") is True
    assert container_magic_mismatch(XLSX_MIME, b"PK\x03\x04rest") is False
    assert container_magic_mismatch(docx, b"\xd0\xcf\x11\xe0") is True
    assert container_magic_mismatch("application/vnd.ms-excel", b"\xd0\xcf\x11\xe0x") is False
    # No container claim, nothing to check — a CSV is whatever its text says it is.
    assert container_magic_mismatch("text/csv", b"region,units\n1,2\n") is False


# ------------------------------------------------------ 3. formats we can now actually read

def test_an_html_table_saved_as_xlsx_is_recovered(handler):
    """The common BI export: an HTML table under a spreadsheet name. read_html is genuine
    recovery, not a guess — the leading '<' is what routes it there."""
    html = (b"<html><body><table>"
            b"<tr><th>Region</th><th>Units</th></tr>"
            b"<tr><td>North</td><td>12</td></tr>"
            b"<tr><td>South</td><td>8</td></tr>"
            b"</table></body></html>")
    result = handler.safe_extract_content(html, XLSX_MIME, "regional-figures.xlsx")

    assert result["format"] == "excel"
    assert "Region" in result["content"] and "North" in result["content"]
    assert result["total_sheets"] == 1


def test_a_legacy_xls_is_read_with_xlrd(handler):
    """OLE2 magic picks the xlrd engine — openpyxl cannot open a legacy .xls at all. The
    engine choice is the contract; the reader itself is xlrd's business."""
    import importlib.util
    assert importlib.util.find_spec("xlrd"), "xlrd must be installed for the OLE2 branch to read"

    import pandas as real_pd
    frame = real_pd.DataFrame({"Region": ["North"], "Units": [12]})
    fake_pd = MagicMock()
    fake_pd.read_excel = MagicMock(return_value={"Sheet1": frame})

    result = handler._parse_excel_with_pandas(
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 64, "legacy.xls", fake_pd)

    assert fake_pd.read_excel.call_args.kwargs["engine"] == "xlrd"
    assert result["format"] == "excel" and "North" in result["content"]


def test_a_zip_still_goes_to_openpyxl(handler):
    """The sniff must not change the path a real .xlsx takes."""
    fake_pd = MagicMock()
    fake_pd.read_excel = MagicMock(side_effect=ValueError("boom"))

    with pytest.raises(SpreadsheetFormatMismatch):
        handler._parse_excel_with_pandas(b"PK\x03\x04" + b"\x00" * 64, "book.xlsx", fake_pd)

    assert fake_pd.read_excel.call_args_list[0].kwargs["engine"] == "openpyxl"


# --------------------------------------------------------- 4. the file is still mountable

@pytest.mark.asyncio
async def test_a_failed_document_keeps_a_metadata_only_row_the_catalog_can_offer(temp_db):
    """"Mount it and convert it in the sandbox" is advice the model needs an id to act on.
    The row carries the failure reason and the Slack ref — never bytes, never content."""
    proc = _Proc(db=temp_db)
    await _run(proc, os.urandom(2048))

    rows = await temp_db.get_thread_documents_async("D1:100.0")
    assert [r["filename"] for r in rows] == ["quarterly-figures.xlsx"]
    assert rows[0]["url_private"] and rows[0]["file_id"] == "F1"
    assert "Could not be read" in rows[0]["summary"]

    catalog = await build_catalog(temp_db, "D1:100.0")
    assert [e["filename"] for e in catalog] == ["quarterly-figures.xlsx"]
    assert "Could not be read" in catalog[0]["description"]


# ------------------------------------------------------------------- 5. both renderers

def test_the_model_is_told_why_and_what_to_do_about_it():
    reasons = MessageProcessor._failed_file_reasons([{
        "name": "quarterly-figures.xlsx",
        "type": "file",
        "mimetype": XLSX_MIME,
        "error": "extraction_failed",
        "detail": "Document could not be parsed",
    }])

    assert len(reasons) == 1
    name, reason = reasons[0]
    assert name == "quarterly-figures.xlsx"
    assert "could not be read" in reason
    assert "mount_file" in reason and "sandbox" in reason
    assert "unsupported file type" not in reason


def test_the_card_does_not_call_a_failed_xlsx_an_unsupported_type():
    """The supported-formats explainer would list XLSX as supported directly underneath a
    failed .xlsx, which is the least useful thing the card could say."""
    notice = MessageProcessor._build_failed_files_notice([{
        "name": "quarterly-figures.xlsx",
        "type": "file",
        "mimetype": XLSX_MIME,
        "error": "extraction_failed",
        "detail": "Document could not be parsed",
    }])

    assert "Couldn't Read File" in notice
    assert "quarterly-figures.xlsx" in notice
    assert "Unsupported File Type" not in notice


def test_the_other_failure_categories_are_untouched():
    """Same list, four kinds of failure — the new one must not swallow the others."""
    files = [
        {"name": "huge.pdf", "type": "file", "mimetype": "application/pdf",
         "too_large": True, "size_bytes": 60 * 1024 * 1024, "limit_bytes": 50 * 1024 * 1024},
        {"name": "gone.docx", "type": "file", "mimetype": "application/msword",
         "error": "download_failed"},
        {"name": "notes.numbers", "type": "file", "mimetype": "application/x-iwork"},
        {"name": "broken.xlsx", "type": "file", "mimetype": XLSX_MIME,
         "error": "extraction_failed", "detail": "Document could not be parsed"},
    ]
    notice = MessageProcessor._build_failed_files_notice(files)
    assert "File Too Large" in notice
    assert "Couldn't Download File" in notice
    assert "Unsupported File Type" in notice and "notes.numbers" in notice
    assert "Couldn't Read File" in notice and "broken.xlsx" in notice

    reasons = dict(MessageProcessor._failed_file_reasons(files))
    assert "too large" in reasons["huge.pdf"]
    assert "downloaded" in reasons["gone.docx"]
    assert "unsupported file type" in reasons["notes.numbers"]
    assert "mount_file" in reasons["broken.xlsx"]
