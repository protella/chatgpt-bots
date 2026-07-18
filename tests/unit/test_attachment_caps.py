"""Regression tests for the attachment/document pre-download caps and persistence parity
(FIX workstream A1: findings F5, F17, F18, F19, F26).

All exercise the real ``MessageUtilitiesMixin._process_attachments`` — the method that
actually ships the behavior — through a minimal harness.
"""

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from image_validation import TOO_LARGE_AFTER_CONVERSION
from message_processor.utilities import MessageUtilitiesMixin

pytestmark = pytest.mark.asyncio


def _png(size=(4, 4)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _bmp(size=(8, 8)) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, (200, 100, 50)).save(buf, format="BMP")
    return buf.getvalue()


class _Proc(MessageUtilitiesMixin):
    """Minimal harness — mirrors the real collaborators the mixin touches."""

    def __init__(self, db=None, document_handler=None, image_cap=20 * 1024 * 1024):
        self.db = db
        self.document_handler = document_handler
        self.image_url_handler = MagicMock()
        # Real int so the F5 pre-gate / F19 post-transcode ceiling can compare.
        self.image_url_handler.max_image_size = image_cap
        self.image_url_handler.process_urls_from_text = AsyncMock(return_value=([], []))
        self.thread_manager = MagicMock()
        for name in ("log_info", "log_debug", "log_warning", "log_error"):
            setattr(self, name, MagicMock())


def _msg(attachments=None, text=""):
    return SimpleNamespace(attachments=attachments or [], text=text,
                           channel_id="C1", thread_id="123.456",
                           metadata={"ts": "123.456"})


class SlackBot:  # name is what the URL branch checks (client.__class__.__name__)
    pass


# --------------------------------------------------------------- F5: pre-download caps

async def test_oversized_image_declared_size_skips_download():
    proc = _Proc(image_cap=1000)
    client = MagicMock()
    client.download_file = AsyncMock(return_value=_png())
    att = {"type": "image", "name": "big.png", "mimetype": "image/png",
           "url": "u", "id": "F1", "size": 5000}
    images, _docs, unsupported = await proc._process_attachments(_msg([att]), client)

    assert images == []
    client.download_file.assert_not_called()  # never buffered into memory
    assert len(unsupported) == 1
    assert unsupported[0]["name"] == "big.png"
    assert unsupported[0]["reason"] == TOO_LARGE_AFTER_CONVERSION


async def test_supported_image_download_passes_max_bytes_cap():
    proc = _Proc(image_cap=12345)
    client = MagicMock()
    client.download_file = AsyncMock(return_value=_png())
    att = {"type": "image", "name": "ok.png", "mimetype": "image/png",
           "url": "u", "id": "F1", "size": 10}  # under the cap
    images, _docs, unsupported = await proc._process_attachments(_msg([att]), client)

    assert len(images) == 1 and unsupported == []
    assert client.download_file.call_args.kwargs.get("max_bytes") == 12345


async def test_oversized_document_declared_size_skips_download():
    dh = MagicMock()
    dh.is_document_file = MagicMock(return_value=True)
    dh.max_document_size = 1000
    proc = _Proc(document_handler=dh)
    client = MagicMock()
    client.download_file = AsyncMock(return_value=b"x")
    att = {"type": "file", "name": "huge.pdf", "mimetype": "application/pdf",
           "url": "u", "id": "F2", "size": 99999}
    _images, docs, unsupported = await proc._process_attachments(_msg([att]), client)

    assert docs == []
    client.download_file.assert_not_called()
    entry = unsupported[0]
    assert entry["name"] == "huge.pdf"
    assert entry.get("too_large") is True
    assert entry["size_bytes"] == 99999 and entry["limit_bytes"] == 1000


async def test_supported_document_download_passes_max_bytes_cap():
    dh = MagicMock()
    dh.is_document_file = MagicMock(return_value=True)
    dh.max_document_size = 777
    # extraction fails (returns falsy) — we only assert the download cap here
    dh.safe_extract_content_async = AsyncMock(return_value=None)
    proc = _Proc(document_handler=dh)
    client = MagicMock()
    client.download_file = AsyncMock(return_value=b"tiny")
    att = {"type": "file", "name": "ok.pdf", "mimetype": "application/pdf",
           "url": "u", "id": "F2", "size": 10}
    await proc._process_attachments(_msg([att]), client)

    assert client.download_file.call_args.kwargs.get("max_bytes") == 777


# ----------------------------------------------------------- F19: post-transcode ceiling

async def test_transcoded_image_over_cap_is_rejected():
    # A BMP (no declared size, so the pre-gate is skipped) decodes and re-encodes to a PNG
    # larger than a tiny ceiling; the result must be caught before it's base64'd in.
    proc = _Proc(image_cap=10)
    client = MagicMock()
    client.download_file = AsyncMock(return_value=_bmp())
    att = {"type": "image", "name": "pic.bmp", "mimetype": "image/bmp",
           "url": "u", "id": "F1"}
    images, _docs, unsupported = await proc._process_attachments(_msg([att]), client)

    assert images == []
    assert unsupported[0]["reason"] == TOO_LARGE_AFTER_CONVERSION


# ----------------------------------------------------- F17: page-gate re-extraction honesty

async def test_scanned_pdf_over_page_cap_reextracts_with_rendering():
    dh = MagicMock()
    dh.is_document_file = MagicMock(return_value=True)
    dh.max_document_size = 50 * 1024 * 1024

    first = {  # ocr_images=False bet on native; note promises rendered pages, no content
        "content": "[Note: ... The document is being provided to the model as rendered pages.]",
        "is_image_based": True, "total_pages": 200, "metadata": {},
    }
    second = {  # ocr_images=True: real page images now exist
        "content": "[scanned document ... converted 1 page(s) to images.]",
        "is_image_based": True, "total_pages": 200, "metadata": {},
        "page_images": [{"mimetype": "image/png", "base64_data": "QUJD", "page": 1}],
    }
    seen_ocr_images = []

    async def fake_extract(data, mime, name, ocr_images=True, ocr_text=False):
        seen_ocr_images.append(ocr_images)
        return first if ocr_images is False else second

    dh.safe_extract_content_async = fake_extract
    proc = _Proc(document_handler=dh)
    proc._summarize_document_for_attach = AsyncMock(return_value="SUMMARY")
    proc.thread_manager.get_or_create_document_ledger = MagicMock(return_value=MagicMock())

    client = MagicMock()
    client.download_file = AsyncMock(return_value=b"%PDF-1.4 tiny")
    att = {"type": "file", "name": "scan.pdf", "mimetype": "application/pdf",
           "url": "u", "id": "F3"}
    images, docs, _unsupported = await proc._process_attachments(_msg([att]), client)

    # Bet-then-correct: first pass skipped rendering, the page gate then forced a re-extract.
    assert seen_ocr_images == [False, True]
    assert docs and docs[0]["native"] is False
    # The re-extracted page image actually rode the turn (not a contentless promise).
    assert any(i.get("source") == "pdf_page" for i in images)


# ------------------------------------------------------- F18: URL-borne image persistence

async def test_external_url_image_is_persisted():
    db = MagicMock()
    db.save_image_metadata_async = AsyncMock()
    proc = _Proc(db=db)
    proc.image_url_handler.process_urls_from_text = AsyncMock(return_value=(
        [{"url": "http://x/img.png", "mimetype": "image/png", "base64_data": "QUJD"}], []))
    client = MagicMock()
    images, _docs, _unsupported = await proc._process_attachments(
        _msg(text="see http://x/img.png"), client)

    assert len(images) == 1
    db.save_image_metadata_async.assert_awaited_once()
    kwargs = db.save_image_metadata_async.call_args.kwargs
    assert kwargs["url"] == "http://x/img.png"
    assert kwargs["image_type"] == "uploaded"
    # Metadata/URL only — never base64 into the DB.
    assert "QUJD" not in str(kwargs.get("metadata"))


async def test_slack_url_image_is_persisted():
    db = MagicMock()
    db.save_image_metadata_async = AsyncMock()
    proc = _Proc(db=db)
    url = "https://files.slack.com/files-pri/T1-F9/pic.png"
    client = SlackBot()
    client.download_file = AsyncMock(return_value=_png())
    client.extract_file_id_from_url = MagicMock(return_value="F9")
    proc._extract_slack_file_urls = MagicMock(return_value=[url])
    images, _docs, _unsupported = await proc._process_attachments(_msg(text=f"see {url}"), client)

    assert len(images) == 1
    db.save_image_metadata_async.assert_awaited()
    kwargs = db.save_image_metadata_async.call_args.kwargs
    assert kwargs["url"] == url
    assert kwargs["metadata"]["source"] == "slack_url"


# --------------------------------------- F26: Slack-URL document parity (summary/ledger/cache)

async def test_slack_url_document_warms_ledger_and_extraction_cache():
    from message_processor.document_tools import _extraction_cache

    dh = MagicMock()
    dh.max_document_size = 50 * 1024 * 1024
    extracted = {"content": "col1,col2\n1,2",
                 "page_structure": {"sheets": {"Sheet1": {"rows": 1}}},
                 "total_pages": 1, "metadata": {}}

    async def fake_extract(*a, **k):
        return extracted

    dh.safe_extract_content_async = fake_extract
    proc = _Proc(document_handler=dh)
    ledger = MagicMock()
    proc.thread_manager.get_or_create_document_ledger = MagicMock(return_value=ledger)

    url = "https://files.slack.com/files-pri/T1-F7/data.csv"
    client = SlackBot()
    client.download_file = AsyncMock(return_value=b"col1,col2\n1,2")
    client.extract_file_id_from_url = MagicMock(return_value="F7")
    proc._extract_slack_file_urls = MagicMock(return_value=[url])
    _images, docs, _unsupported = await proc._process_attachments(_msg(text=f"see {url}"), client)

    assert len(docs) == 1
    assert docs[0]["summary"]  # deterministic schema block, not empty
    ledger.add_document.assert_called_once()
    assert _extraction_cache.get("F7") == "col1,col2\n1,2"


# ------------------------------- F5 (T1-5): Slack-URL-in-text downloads are also capped

async def test_slack_url_image_download_passes_image_cap():
    proc = _Proc(image_cap=20 * 1024 * 1024)
    url = "https://files.slack.com/files-pri/T1-F9/pic.png"
    client = SlackBot()
    client.download_file = AsyncMock(return_value=_png())
    client.extract_file_id_from_url = MagicMock(return_value="F9")
    proc._extract_slack_file_urls = MagicMock(return_value=[url])
    images, _docs, _unsupported = await proc._process_attachments(_msg(text=f"see {url}"), client)

    assert len(images) == 1
    # Size is unknown for a pasted permalink; the streamed cap is the only guard.
    assert client.download_file.call_args.kwargs.get("max_bytes") == 20 * 1024 * 1024


async def test_slack_url_document_download_passes_document_cap():
    dh = MagicMock()
    dh.max_document_size = 50 * 1024 * 1024

    async def fake_extract(*a, **k):
        return {"content": "c1,c2\n1,2",
                "page_structure": {"sheets": {"S": {"rows": 1}}},
                "total_pages": 1, "metadata": {}}

    dh.safe_extract_content_async = fake_extract
    proc = _Proc(document_handler=dh)
    proc.thread_manager.get_or_create_document_ledger = MagicMock(return_value=MagicMock())
    url = "https://files.slack.com/files-pri/T1-F7/data.csv"
    client = SlackBot()
    client.download_file = AsyncMock(return_value=b"c1,c2\n1,2")
    client.extract_file_id_from_url = MagicMock(return_value="F7")
    proc._extract_slack_file_urls = MagicMock(return_value=[url])
    _images, docs, _unsupported = await proc._process_attachments(_msg(text=f"see {url}"), client)

    assert len(docs) == 1
    assert client.download_file.call_args.kwargs.get("max_bytes") == 50 * 1024 * 1024


async def test_slack_url_download_abort_surfaces_download_failure():
    # A stream aborted at the cap returns None (the hard guarantee for unknown-size URLs);
    # that must degrade to the honest download-failed notice, not a silent drop.
    dh = MagicMock()
    dh.max_document_size = 50 * 1024 * 1024
    proc = _Proc(document_handler=dh)
    url = "https://files.slack.com/files-pri/T1-F7/big.pdf"
    client = SlackBot()
    client.download_file = AsyncMock(return_value=None)
    client.extract_file_id_from_url = MagicMock(return_value="F7")
    proc._extract_slack_file_urls = MagicMock(return_value=[url])
    _images, docs, unsupported = await proc._process_attachments(_msg(text=f"see {url}"), client)

    assert docs == []
    assert client.download_file.call_args.kwargs.get("max_bytes") == 50 * 1024 * 1024
    assert any(f.get("error") == "download_failed" for f in unsupported)
