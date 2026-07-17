"""F49 — widened file-type gate + central extension→handler routing.

Covers the gate ordering (denylist beats a lying mimetype), the extension/known-
filename admission, the dispatch parity between admission and routing (the biggest
hole the audit found), and the three new handlers (rtf/eml/ipynb) plus the .tsv
end-to-end path.
"""
import json

import pytest

from document_handler import (
    DENIED_EXTENSIONS,
    DOCUMENT_EXTENSIONS,
    EXTENSION_HANDLERS,
    KNOWN_FILENAMES,
    MIME_TYPE_HANDLERS,
    SUPPORTED_DOCUMENT_MIMETYPES,
    DocumentHandler,
)


@pytest.fixture
def handler():
    return DocumentHandler()


# --- gate: admission ---

def test_tsv_admitted_both_labels(handler):
    # The incident: Slack sends text/tab-separated-values, which used to be rejected.
    assert handler.is_document_file("data.tsv", "text/tab-separated-values")
    assert handler.is_document_file("data.tsv", "text/tsv")
    # And when Slack forwards it as an opaque blob, the extension still admits it.
    assert handler.is_document_file("data.tsv", "application/octet-stream")


def test_text_family_catch_all(handler):
    # Any correctly-labeled text/* type is admitted even without a known extension.
    assert handler.is_document_file("weird.xyzzy", "text/x-lisp")
    assert handler.is_document_file("code.rs", "text/rust")


def test_typescript_survives_video_mp2t(handler):
    # Slack sends video/mp2t for a .ts file — it fails the mimetype check and is
    # admitted ONLY via the extension. This is why mimetype must not short-circuit.
    assert handler.is_document_file("app.ts", "video/mp2t")
    assert handler._handler_for_filename("app.ts") == "parse_text"


@pytest.mark.parametrize("name", ["prod.env", "id_rsa.pem", "server.key",
                                  "archive.zip", "backup.tar.gz", "movie.mp4",
                                  "song.mp3", "app.exe", "font.woff2",
                                  # legacy binary Office
                                  "old.doc", "deck.ppt", "book.xlsb",
                                  # deliberately-dropped containers/ebook/message/columnar
                                  "notes.odt", "sheet.ods", "slides.odp",
                                  "book.epub", "mail.msg", "data.parquet",
                                  # raster/vector images belong to the vision path
                                  "photo.jpg", "photo.jpeg", "logo.png", "anim.gif",
                                  "pic.webp", "scan.tiff", "iphone.heic", "vector.svg"])
def test_denylist_beats_lying_text_mimetype(handler, name):
    # A secret or binary mislabeled text/plain must STILL be refused: latin-1 would
    # decode any bytes into confident mojibake, which is a regression, not coverage.
    # These use an HONEST text/plain LIE (Slack labels a file from its name) — the denylist
    # runs before the text/* catch-all, so the lie can't smuggle a binary past the gate.
    assert not handler.is_document_file(name, "text/plain")


@pytest.mark.parametrize("name,mimetype", [
    ("old.doc", "application/msword"),
    ("deck.ppt", "application/vnd.ms-powerpoint"),
    ("notes.odt", "application/vnd.oasis.opendocument.text"),
    ("book.epub", "application/epub+zip"),
    ("mail.msg", "application/vnd.ms-outlook"),
    ("data.parquet", "application/octet-stream"),
    ("photo.jpg", "image/jpeg"),
    ("iphone.heic", "image/heic"),
])
def test_dropped_and_binary_types_rejected_with_honest_mime(handler, name, mimetype):
    # Even with their REAL mimetype these deliberately-unsupported types stay out — the
    # denylist is the gate, not a mimetype quirk.
    assert not handler.is_document_file(name, mimetype)


def test_filename_none_does_not_crash(handler):
    # Slack documents `name` as nullable; the old code called .lower() unconditionally.
    assert handler.is_document_file(None, "text/plain") is True
    assert handler.is_document_file(None, None) is False
    assert handler.is_document_file(None, "application/zip") is False


def test_no_extension_rejected_but_known_filenames_admitted(handler):
    # A blanket "no extension -> text" rule would admit binary blobs; only the
    # curated extensionless basenames are admitted.
    assert not handler.is_document_file("file_without_extension")
    assert not handler.is_document_file("")
    for name in ("Dockerfile", "makefile", "README", "LICENSE", ".gitignore"):
        assert handler.is_document_file(name), name


def test_images_and_legacy_office_stay_out(handler):
    assert not handler.is_document_file("image.jpg", "image/jpeg")
    assert not handler.is_document_file("video.mp4")
    assert not handler.is_document_file("old.ppt", "application/vnd.ms-powerpoint")
    assert not handler.is_document_file("old.doc", "application/msword")


def test_denied_and_document_extension_sets_are_disjoint():
    assert not (set(DENIED_EXTENSIONS) & DOCUMENT_EXTENSIONS)


# --- dispatch parity (admission <-> routing) ---

def test_every_extension_routes_to_its_central_handler(handler):
    """The central map is the single source of truth: resolving a filename must
    return exactly the handler the map declares for its extension."""
    for ext, expected in EXTENSION_HANDLERS.items():
        assert handler._handler_for_filename(f"file{ext}") == expected, ext


def test_known_filenames_route_to_text(handler):
    for name in KNOWN_FILENAMES:
        assert handler._handler_for_filename(name) == "parse_text", name


def test_longest_extension_wins(handler):
    # A compound name routes by its real trailing extension, not an earlier one.
    assert handler._handler_for_filename("report.pdf.txt") == "parse_text"
    assert handler._handler_for_filename("data.csv.xlsx") == "parse_excel_adaptive"


def test_admitted_extension_dispatches_to_resolved_handler(handler, monkeypatch):
    """End-to-end: safe_extract_content routes an octet-stream upload by extension.
    This is the exact hole the audit flagged — .odt/.ipynb/.eml arriving as
    application/octet-stream used to be latin-1'd into mojibake."""
    calls = {}

    def make_spy(name):
        def _spy(file_data, filename, *a, **k):
            calls["handler"] = name
            return {"content": "ok", "format": name}
        return _spy

    for method_name in set(EXTENSION_HANDLERS.values()):
        monkeypatch.setattr(handler, method_name, make_spy(method_name))

    for ext, expected in EXTENSION_HANDLERS.items():
        calls.clear()
        # PK-prefixed office formats hit the zip-bomb probe first; use plain bytes.
        handler.safe_extract_content(b"payload", "application/octet-stream", f"f{ext}")
        assert calls.get("handler") == expected, ext


def test_ipynb_mislabeled_json_is_not_context_bombed(handler):
    # If Slack labels a notebook application/json, mimetype-first dispatch would dump
    # raw JSON (megabytes of base64 outputs). Extension-first routes to parse_notebook.
    nb = {"cells": [{"cell_type": "code", "source": "print(1)",
                     "outputs": [{"data": {"image/png": "B64" * 5000}}]}]}
    result = handler.safe_extract_content(
        json.dumps(nb).encode(), "application/json", "nb.ipynb")
    assert result["format"] == "ipynb"
    assert "B64B64" not in result["content"]


def test_supported_mimetypes_subset_of_handlers():
    missing = SUPPORTED_DOCUMENT_MIMETYPES - set(MIME_TYPE_HANDLERS)
    assert not missing, missing
    h = DocumentHandler()
    for mimetype, name in MIME_TYPE_HANDLERS.items():
        assert callable(getattr(h, name, None)), (mimetype, name)


# --- new handlers with real bytes ---

def test_parse_rtf_strips_control_codes(handler):
    rtf = rb"{\rtf1\ansi\deff0 {\fonttbl {\f0 Times;}}\f0\fs24 Hello \b bold\b0  world.\par}"
    result = handler.safe_extract_content(rtf, "application/rtf", "note.rtf")
    assert result["format"] == "rtf"
    assert result.get("error") is None
    assert "Hello" in result["content"] and "bold" in result["content"]
    assert "\\rtf1" not in result["content"] and "fonttbl" not in result["content"]


def test_parse_email_headers_and_body(handler):
    eml = (b"From: alice@example.com\r\nTo: bob@example.com\r\n"
           b"Subject: Lunch plans\r\nDate: Mon, 1 Jan 2024 12:00:00 +0000\r\n"
           b"Content-Type: text/plain\r\n\r\nAre we still on for lunch tomorrow?\r\n")
    result = handler.safe_extract_content(eml, "message/rfc822", "invite.eml")
    assert result["format"] == "eml"
    assert "From: alice@example.com" in result["content"]
    assert "Subject: Lunch plans" in result["content"]
    assert "lunch tomorrow" in result["content"].lower()


def test_parse_email_multipart_skips_attachment(handler):
    eml = (
        b"From: a@x.com\r\nTo: b@y.com\r\nSubject: With file\r\n"
        b'Content-Type: multipart/mixed; boundary="B"\r\n\r\n'
        b"--B\r\nContent-Type: text/plain\r\n\r\nThe body text.\r\n"
        b"--B\r\nContent-Type: application/octet-stream\r\n"
        b'Content-Disposition: attachment; filename="secret.bin"\r\n\r\n'
        b"BINARYPAYLOAD\r\n--B--\r\n"
    )
    result = handler.safe_extract_content(eml, "message/rfc822", "m.eml")
    assert "The body text." in result["content"]
    assert "BINARYPAYLOAD" not in result["content"]


def test_parse_notebook_strips_outputs(handler):
    nb = {"cells": [
        {"cell_type": "markdown", "source": ["# Analysis\n", "Notes here"]},
        {"cell_type": "code", "execution_count": 7, "source": "x = compute()",
         "outputs": [{"output_type": "stream", "text": "SHOULD_NOT_APPEAR"},
                     {"data": {"image/png": "PNGB64" * 4000}}]},
        {"cell_type": "code", "source": ""},  # empty cell dropped
    ], "nbformat": 4}
    result = handler.safe_extract_content(
        json.dumps(nb).encode(), "application/x-ipynb+json", "a.ipynb")
    assert result["format"] == "ipynb"
    assert result["total_cells"] == 3
    assert "# Analysis" in result["content"]
    assert "x = compute()" in result["content"]
    assert "SHOULD_NOT_APPEAR" not in result["content"]
    assert "PNGB64PNGB64" not in result["content"]


# --- .tsv end-to-end + delimiter routing ---

def test_tsv_end_to_end(handler):
    tsv = b"Name\tAge\tCity\nAlice\t30\tNYC\nBob\t25\tLA\n"
    result = handler.safe_extract_content(tsv, "text/tab-separated-values", "data.tsv")
    assert result.get("format") != "error"
    assert result.get("cols") == 3
    assert "Alice" in result["content"] and "NYC" in result["content"]


@pytest.mark.parametrize("name", ["data.tsv", "data.tab", "data.psv"])
def test_delimited_text_routes_through_csv_parser(handler, monkeypatch, name):
    seen = {}

    def spy(file_data, filename, pd):
        seen["called"] = filename
        return {"content": "csv", "format": "csv"}

    monkeypatch.setattr(handler, "_parse_csv_with_pandas", spy)
    handler.parse_excel_adaptive(b"a\tb\n1\t2\n", name)
    assert seen.get("called") == name


# --- support card ---

def test_support_card_common_plus_more():
    from message_processor.base import MessageProcessor
    files = [{"name": "mystery.qux", "mimetype": "application/x-qux"}]
    text = MessageProcessor._build_failed_files_notice(files)
    assert "Unsupported File Type" in text
    assert "PDF" in text and "DOCX" in text and "TSV" in text
    assert "and " in text and "more" in text  # "and N more"
    # The rendered common list must not resurrect legacy binary types.
    assert "DOC," not in text and "PPT," not in text
