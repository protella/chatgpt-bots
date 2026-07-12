"""F32: code-interpreter artifacts — listing, validation, dedupe, delivery.

Design facts these tests encode (all verified live against the real API):

* The container LISTING is the only publication source. `container_file_citation` annotations
  appear only when the model writes a `sandbox:` link — which our prompt forbids — so with our
  prompt the model cites NOTHING. The first version of this feature shipped citation-driven:
  every unit test passed and it published zero files in production. Hence the listing.
* A user's own attachment mounts into the SAME container (`source == "user"`). The
  `source == "assistant"` filter is the only thing stopping us from uploading someone's
  spreadsheet back into their channel.
* The model names these files, so every filename is untrusted input.
* Nothing here may raise into the turn: the answer has already posted.
"""
import io
import zipfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from message_processor import artifacts as artifacts_mod
from message_processor.artifacts import (

    _magic_ok,
    collect_container_ids,
    publish_artifacts,
    resolve_container_artifacts,
    sanitize_filename,
    strip_sandbox_links,
)
from message_processor.handlers.text import TextHandlerMixin
from message_processor.utilities import MessageUtilitiesMixin

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
PDF = b"%PDF-1.7\n" + b"x" * 32
CSV = b"region,units\nNorth,65316\n"


_MAIN_TYPES = {
    "xlsx": ("application/vnd.openxmlformats-officedocument."
             "spreadsheetml.sheet.main+xml"),
    "docx": ("application/vnd.openxmlformats-officedocument."
             "wordprocessingml.document.main+xml"),
    "pptx": ("application/vnd.openxmlformats-officedocument."
             "presentationml.presentation.main+xml"),
}


def _ooxml(extra_names=(), kind="xlsx", content_types=None):
    """A structurally valid OOXML package.

    `content_types` overrides the declared body — that is the whole point of the validation:
    the bytes of `[Content_Types].xml` are what make this a workbook rather than a zip with a
    suggestively-named entry.
    """
    if content_types is None:
        content_types = (
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/'
            'package/2006/content-types">'
            f'<Override PartName="/xl/workbook.xml" ContentType="{_MAIN_TYPES[kind]}"/>'
            '</Types>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("xl/workbook.xml", "<workbook/>")
        for n in extra_names:
            zf.writestr(n, b"\x00\x01")
    return buf.getvalue()


XLSX = _ooxml()
DOCX = _ooxml(kind="docx")
XLSM_MACRO = _ooxml(["xl/vbaProject.bin"])
# The exact shape the old check waved through: a zip carrying an EMPTY `[Content_Types].xml`.
# Its presence proved nothing; only its contents do.
DUMMY_CONTENT_TYPES = _ooxml(content_types="<Types/>")
BARE_ZIP = b"PK\x03\x04" + b"\x00" * 40  # a zip that is NOT an OOXML package


@pytest.fixture(autouse=True)
def _reset_published_memory():
    """publish_artifacts remembers ids process-wide (the reused-container guard). Tests share
    ids like "f1", so without a reset a later test's file is skipped as already-published."""
    artifacts_mod._published_file_ids.clear()
    yield
    artifacts_mod._published_file_ids.clear()


# --------------------------------------------------------------------------------- helpers

def _cfile(id_, path, source="assistant", size=None):
    f = MagicMock()
    f.id = id_
    f.path = path
    f.source = source
    f.bytes = size
    return f


class _Pager:
    """containers.files.list() returns an async-iterable paginator (not a coroutine)."""
    def __init__(self, files):
        self._files = files

    def __aiter__(self):
        async def gen():
            for f in self._files:
                yield f
        return gen()


class _StreamedBody:
    """`content.with_streaming_response.retrieve(...)` is an async CONTEXT MANAGER whose body is
    pulled in chunks — the shape the size cap depends on (a buffered read is not a cap at all)."""

    def __init__(self, payload, headers=None, chunk=64 * 1024):
        self._payload = payload
        self.headers = headers or {}
        self._chunk = chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def iter_bytes(self):
        for i in range(0, len(self._payload), self._chunk):
            yield self._payload[i:i + self._chunk]


def _streaming_retriever(payload_for):
    """Build the `with_streaming_response.retrieve` stand-in (a plain callable, NOT a coroutine —
    it returns the async context manager directly)."""
    def _retrieve(file_id, container_id=None):
        return payload_for(file_id)
    return MagicMock(side_effect=_retrieve)


def _openai(files=(), payload=PNG, headers=None):
    oc = MagicMock()
    oc.client.containers.files.list = MagicMock(return_value=_Pager(list(files)))
    oc.client.containers.files.content.with_streaming_response.retrieve = _streaming_retriever(
        lambda fid: _StreamedBody(payload, headers))
    return oc


def _openai_payloads(files, payload_by_id):
    oc = MagicMock()
    oc.client.containers.files.list = MagicMock(return_value=_Pager(list(files)))
    oc.client.containers.files.content.with_streaming_response.retrieve = _streaming_retriever(
        lambda fid: _StreamedBody(payload_by_id[fid]))
    return oc


_UPLOAD = {"file_id": "F123", "url_private": "https://files.slack.com/x",
           "permalink": "https://p"}


def _client(upload=_UPLOAD):
    """upload=None models a failed Slack upload (send_file's documented failure contract)."""
    c = MagicMock()
    c.send_file = AsyncMock(return_value=upload)
    return c


# ----------------------------------------------------------------------------- containers

class TestCollectContainerIds:
    def test_unique_in_order(self):
        sink = [{"container_id": "c1"}, {"container_id": "c1"}, {"container_id": "c2"}]
        assert collect_container_ids(sink) == ["c1", "c2"]

    def test_each_tool_loop_round_has_its_own_container(self):
        """Every `auto` API call gets a FRESH container, and the loop makes one call per
        round. List only the last and the earlier round's chart is silently dropped."""
        sink = [{"container_id": "round1"}, {"container_id": "round2"}]
        assert collect_container_ids(sink) == ["round1", "round2"]

    def test_empty(self):
        assert collect_container_ids(None) == []
        assert collect_container_ids([]) == []


class TestNoteContainerSink:
    def test_sinks_the_container_id(self):
        from openai_client.api.responses import _note_container
        sink = []
        _note_container(sink, MagicMock(container_id="cntr_x"))
        assert sink == [{"container_id": "cntr_x"}]

    def test_none_sink_is_a_noop(self):
        from openai_client.api.responses import _note_container
        _note_container(None, MagicMock(container_id="c"))  # must not raise

    def test_never_raises(self):
        from openai_client.api.responses import _note_container
        broken = MagicMock()
        type(broken).container_id = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        sink = []
        _note_container(sink, broken)
        assert sink == []


@pytest.mark.asyncio
class TestResolveContainerArtifacts:
    async def test_lists_assistant_files(self):
        refs = await resolve_container_artifacts(
            _openai([_cfile("f1", "/mnt/data/chart.png")]), ["c1"])
        assert [(r.file_id, r.filename) for r in refs] == [("f1", "chart.png")]

    async def test_user_attachment_is_never_republished(self):
        """THE confidentiality boundary. The user's own upload mounts into this container as
        source='user'; without the filter we hand their spreadsheet back to the channel."""
        refs = await resolve_container_artifacts(_openai([
            _cfile("u1", "/mnt/data/abc-their_salaries.xlsx", source="user"),
            _cfile("f1", "/mnt/data/chart.png", source="assistant"),
        ]), ["c1"])
        assert [r.file_id for r in refs] == ["f1"]

    async def test_unknown_source_fails_closed(self):
        """If the API renames or drops `source`, skip the file rather than leak one."""
        refs = await resolve_container_artifacts(
            _openai([_cfile("x1", "/mnt/data/thing.png", source=None)]), ["c1"])
        assert refs == []

    async def test_walks_every_container(self):
        oc = MagicMock()
        pages = {"c1": _Pager([_cfile("f1", "/mnt/data/a.png")]),
                 "c2": _Pager([_cfile("f2", "/mnt/data/b.png")])}
        oc.client.containers.files.list = MagicMock(side_effect=lambda container_id: pages[container_id])
        refs = await resolve_container_artifacts(oc, ["c1", "c2"])
        assert [r.file_id for r in refs] == ["f1", "f2"]

    async def test_expired_container_yields_nothing(self):
        oc = MagicMock()
        oc.client.containers.files.list = MagicMock(side_effect=RuntimeError("expired"))
        assert await resolve_container_artifacts(oc, ["dead"]) == []

    async def test_no_containers(self):
        assert await resolve_container_artifacts(MagicMock(), []) == []


# ------------------------------------------------------------------------------- delivery

@pytest.mark.asyncio
class TestPublishArtifacts:
    async def test_publishes_a_file_the_model_never_cited(self):
        """THE regression this feature first shipped with: our prompt forbids sandbox links,
        so the API emits ZERO citations. Artifacts must publish anyway."""
        client = _client()
        out = await publish_artifacts(
            openai_client=_openai([_cfile("f1", "/mnt/data/revenue.png")]), client=client,
            channel_id="C1", thread_id="1.0", thread_key="C1:1.0",
            container_ids=["cntr_1"], db=None)
        assert [p["filename"] for p in out] == ["revenue.png"]
        # bytes go up as BytesIO — never a path, never disk
        assert isinstance(client.send_file.call_args.kwargs["file_data"], io.BytesIO)

    async def test_no_containers_is_a_noop(self):
        client = _client()
        assert await publish_artifacts(
            openai_client=_openai(), client=client, channel_id="C1", thread_id="1.0",
            thread_key="C1:1.0", container_ids=[], db=None) == []
        client.send_file.assert_not_called()

    async def test_code_ran_but_saved_nothing(self):
        client = _client()
        out = await publish_artifacts(
            openai_client=_openai([]), client=client, channel_id="C1", thread_id="1.0",
            thread_key="C1:1.0", container_ids=["c1"], db=None)
        assert out == []
        client.send_file.assert_not_called()

    async def test_identical_bytes_post_once(self):
        """The model saved a chart AND displayed it: same bytes, one upload."""
        client = _client()
        out = await publish_artifacts(
            openai_client=_openai([_cfile("f1", "/mnt/data/a.png"),
                                   _cfile("f2", "/mnt/data/b.png")]),
            client=client, channel_id="C1", thread_id="1.0", thread_key="C1:1.0",
            container_ids=["c1"], db=None)
        assert len(out) == 1
        assert client.send_file.call_count == 1

    async def test_distinct_charts_both_post(self):
        """Two genuinely different charts must BOTH survive — content, not filename, decides."""
        files = [_cfile("f1", "/mnt/data/revenue.png"),
                 _cfile("f2", "/mnt/data/cfile_deadbeef99.png")]
        oc = _openai_payloads(files, {"f1": PNG, "f2": PNG + b"different"})
        client = _client()
        out = await publish_artifacts(
            openai_client=oc, client=client, channel_id="C1", thread_id="1.0",
            thread_key="C1:1.0", container_ids=["c1"], db=None)
        assert len(out) == 2
        # the display-only render gets a readable name, not an internal id
        assert "revenue.png" in [p["filename"] for p in out]
        assert any(p["filename"].startswith("output_") for p in out)

    async def test_reused_container_does_not_repost(self):
        oc = _openai([_cfile("f1", "/mnt/data/chart.png")])
        first = await publish_artifacts(
            openai_client=oc, client=_client(), channel_id="C1", thread_id="1.0",
            thread_key="C1:1.0", container_ids=["c1"], db=None)
        assert len(first) == 1

        client2 = _client()
        second = await publish_artifacts(
            openai_client=oc, client=client2, channel_id="C1", thread_id="1.0",
            thread_key="C1:1.0", container_ids=["c1"], db=None)
        assert second == []
        client2.send_file.assert_not_called()

    async def test_cap_counts_accepted_uploads_not_candidates(self):
        """An intermediate the model happened to write must not eat the budget the real
        deliverable needed: the first two files here are unpublishable."""
        files = [_cfile("f1", "/mnt/data/junk.exe"),        # rejected: extension
                 _cfile("f2", "/mnt/data/notes.bin"),       # rejected: extension
                 _cfile("f3", "/mnt/data/report.png"),      # valid
                 _cfile("f4", "/mnt/data/data.csv")]        # valid
        oc = _openai_payloads(files, {"f3": PNG, "f4": CSV})
        with patch("message_processor.artifacts.config") as cfg:
            cfg.artifact_max_files = 2
            cfg.artifact_max_mb = 25
            cfg.artifact_allowed_extensions = ["png", "csv"]
            out = await publish_artifacts(
                openai_client=oc, client=_client(), channel_id="C1", thread_id="1.0",
                thread_key="C1:1.0", container_ids=["c1"], db=None)
        assert sorted(p["filename"] for p in out) == ["data.csv", "report.png"]

    async def test_oversize_rejected_by_declared_length_before_download(self):
        """Preflight on content-length so a huge file is refused before we copy it in."""
        oc = MagicMock()
        oc.client.containers.files.list = MagicMock(
            return_value=_Pager([_cfile("f1", "/mnt/data/huge.png")]))
        oc.client.containers.files.content.retrieve = AsyncMock(
            return_value=MagicMock(content=PNG, headers={"content-length": str(999 * 1024 * 1024)}))
        client = _client()
        with patch("message_processor.artifacts.config") as cfg:
            cfg.artifact_max_files = 4
            cfg.artifact_max_mb = 25
            cfg.artifact_allowed_extensions = ["png"]
            out = await publish_artifacts(
                openai_client=oc, client=client, channel_id="C1", thread_id="1.0",
                thread_key="C1:1.0", container_ids=["c1"], db=None)
        assert out == []
        client.send_file.assert_not_called()

    async def test_oversize_rejected_after_download_when_no_header(self):
        client = _client()
        with patch("message_processor.artifacts.config") as cfg:
            cfg.artifact_max_files = 4
            cfg.artifact_max_mb = 1
            cfg.artifact_allowed_extensions = ["png"]
            big = PNG + b"\x00" * (2 * 1024 * 1024)
            out = await publish_artifacts(
                openai_client=_openai([_cfile("f1", "/mnt/data/big.png")], payload=big),
                client=client, channel_id="C1", thread_id="1.0", thread_key="C1:1.0",
                container_ids=["c1"], db=None)
        assert out == []
        client.send_file.assert_not_called()

    async def test_content_type_mismatch_never_uploads(self):
        client = _client()
        out = await publish_artifacts(
            openai_client=_openai([_cfile("f1", "/mnt/data/chart.png")], payload=PDF),
            client=client, channel_id="C1", thread_id="1.0", thread_key="C1:1.0",
            container_ids=["c1"], db=None)
        assert out == []
        client.send_file.assert_not_called()

    async def test_disallowed_extension_never_uploads(self):
        client = _client()
        out = await publish_artifacts(
            openai_client=_openai([_cfile("f1", "/mnt/data/evil.exe")]), client=client,
            channel_id="C1", thread_id="1.0", thread_key="C1:1.0",
            container_ids=["c1"], db=None)
        assert out == []
        client.send_file.assert_not_called()

    async def test_upload_without_file_identity_is_not_success(self):
        """A Slack response we can't use is a failure: we could never find the file again."""
        client = _client(upload={"url_private": "https://x"})  # no file_id
        out = await publish_artifacts(
            openai_client=_openai([_cfile("f1", "/mnt/data/chart.png")]), client=client,
            channel_id="C1", thread_id="1.0", thread_key="C1:1.0",
            container_ids=["c1"], db=None)
        assert out == []

    async def test_upload_failure_is_not_fatal(self):
        out = await publish_artifacts(
            openai_client=_openai([_cfile("f1", "/mnt/data/chart.png")]),
            client=_client(upload=None), channel_id="C1", thread_id="1.0",
            thread_key="C1:1.0", container_ids=["c1"], db=None)
        assert out == []

    async def test_upload_raising_does_not_escape(self):
        """publish_artifacts runs AFTER the answer posted — it must never raise into the turn."""
        client = MagicMock()
        client.send_file = AsyncMock(side_effect=RuntimeError("slack exploded"))
        out = await publish_artifacts(
            openai_client=_openai([_cfile("f1", "/mnt/data/chart.png")]), client=client,
            channel_id="C1", thread_id="1.0", thread_key="C1:1.0",
            container_ids=["c1"], db=None)
        assert out == []

    async def test_one_bad_artifact_does_not_block_the_others(self):
        files = [_cfile("f1", "/mnt/data/evil.exe"), _cfile("f2", "/mnt/data/good.png")]
        oc = _openai_payloads(files, {"f2": PNG})
        out = await publish_artifacts(
            openai_client=oc, client=_client(), channel_id="C1", thread_id="1.0",
            thread_key="C1:1.0", container_ids=["c1"], db=None)
        assert [p["filename"] for p in out] == ["good.png"]


@pytest.mark.asyncio
class TestPersistence:
    async def test_document_artifact_goes_to_the_documents_table(self):
        db = MagicMock()
        await publish_artifacts(
            openai_client=_openai([_cfile("f1", "/mnt/data/totals.csv")], payload=CSV),
            client=_client(), channel_id="C1", thread_id="1.0", thread_key="C1:1.0",
            container_ids=["c1"], db=db, message_ts="1.0")
        doc = db.save_document.call_args.kwargs
        assert doc["file_id"] == "F123"          # the Slack ref, so read_document can re-derive
        assert doc["mime_type"] == "text/csv"
        assert doc["metadata"]["source"] == "generated"

    async def test_image_artifact_does_not_go_to_the_documents_table(self):
        """A PNG filed as a "document" LOOKS re-readable but isn't: read_document runs its
        input through DocumentHandler, which has no image parser. Route images to images."""
        db = MagicMock()
        db.save_image_metadata_async = AsyncMock()
        await publish_artifacts(
            openai_client=_openai([_cfile("f1", "/mnt/data/chart.png")]), client=_client(),
            channel_id="C1", thread_id="1.0", thread_key="C1:1.0",
            container_ids=["c1"], db=db, message_ts="1.0")
        db.save_document.assert_not_called()
        db.save_image_metadata_async.assert_awaited_once()
        assert db.save_image_metadata_async.call_args.kwargs["image_type"] == "generated"

    async def test_db_failure_never_unposts_the_file(self):
        db = MagicMock()
        db.save_document.side_effect = RuntimeError("db down")
        out = await publish_artifacts(
            openai_client=_openai([_cfile("f1", "/mnt/data/totals.csv")], payload=CSV),
            client=_client(), channel_id="C1", thread_id="1.0", thread_key="C1:1.0",
            container_ids=["c1"], db=db)
        assert len(out) == 1  # already in Slack; a persistence error must not retract it


# ------------------------------------------------------------------------- sandbox links

class TestStripSandboxLinks:
    def test_markdown_link_keeps_its_label(self):
        out = strip_sandbox_links("Here.\n\n[Download chart.png](sandbox:/mnt/data/chart.png)")
        assert "sandbox:" not in out and "chart.png" in out

    def test_bare_path_removed(self):
        assert "sandbox:" not in strip_sandbox_links("saved to sandbox:/mnt/data/x.csv done")

    def test_plain_text_untouched(self):
        text = "North leads at 65,316 units."
        assert strip_sandbox_links(text) == text

    def test_empty(self):
        assert strip_sandbox_links("") == ""

    def test_link_only_reply_does_not_become_whitespace(self):
        assert strip_sandbox_links("[chart.png](sandbox:/mnt/data/chart.png)").strip() == "chart.png"


# ---------------------------------------------------------------------- filename safety

class TestSanitizeFilename:
    @pytest.mark.parametrize("name", [
        "../../etc/passwd.png", "/abs/path/chart.png", "dir\\sub\\chart.png",
    ])
    def test_paths_collapse_to_a_basename(self, name):
        out = sanitize_filename(name)
        assert out and "/" not in out and "\\" not in out and ".." not in out

    @pytest.mark.parametrize("name", [
        "evil.exe", "run.sh", "lib.so", "macro.xlsm", "doc.docm",
        "page.html", "vector.svg", "archive.zip",
    ])
    def test_disallowed_extensions_refused(self, name):
        assert sanitize_filename(name) is None

    def test_extensionless_and_dotfiles_refused(self):
        assert sanitize_filename("README") is None
        assert sanitize_filename(".bashrc") is None
        assert sanitize_filename("") is None
        assert sanitize_filename(None) is None

    def test_display_only_image_gets_a_readable_name(self):
        assert sanitize_filename("cfile_6a5302c93590819.png", fallback_index=2) == "output_2.png"

    def test_normal_name_preserved(self):
        assert sanitize_filename("revenue_by_region.png") == "revenue_by_region.png"

    def test_control_characters_stripped(self):
        out = sanitize_filename("ch\x00art\n.png")
        assert out and "\x00" not in out and "\n" not in out

    def test_long_name_truncated_keeping_extension(self):
        out = sanitize_filename("a" * 300 + ".csv")
        assert out.endswith(".csv") and len(out) <= 90


# ------------------------------------------------------------------- content validation

class TestMagicBytes:
    def test_png_pdf_accepted(self):
        assert _magic_ok(PNG, "png")
        assert _magic_ok(PDF, "pdf")

    def test_webp_offset_signature(self):
        assert _magic_ok(b"RIFF\x00\x00\x00\x00WEBPxxxx", "webp")
        assert not _magic_ok(b"RIFF\x00\x00\x00\x00XXXXxxxx", "webp")

    def test_extension_content_mismatch_rejected(self):
        assert not _magic_ok(PDF, "png")
        assert not _magic_ok(PNG, "pdf")

    def test_real_ooxml_accepted(self):
        assert _magic_ok(XLSX, "xlsx")

    def test_bare_zip_renamed_xlsx_rejected(self):
        """ZIP magic alone is worthless — any archive would pass. Require the OOXML part."""
        assert not _magic_ok(BARE_ZIP, "xlsx")

    def test_macro_bearing_workbook_rejected(self):
        """We do not hand a colleague a file with a VBA project in it."""
        assert not _magic_ok(XLSM_MACRO, "xlsx")

    def test_dummy_content_types_rejected(self):
        """The presence of `[Content_Types].xml` proves NOTHING — any zip can carry a file by
        that name. Only its declared main-part type makes the package a workbook."""
        assert not _magic_ok(DUMMY_CONTENT_TYPES, "xlsx")

    def test_ooxml_must_match_the_claimed_extension(self):
        """A Word document wearing an .xlsx name: Slack would offer a spreadsheet preview for a
        file Excel cannot open."""
        assert _magic_ok(DOCX, "docx")
        assert not _magic_ok(DOCX, "xlsx")
        assert not _magic_ok(XLSX, "pptx")

    def test_macro_enabled_content_type_rejected(self):
        """Named .xlsx, carrying no vbaProject.bin, but DECLARING itself macro-enabled."""
        sneaky = _ooxml(content_types=(
            '<Types><Override PartName="/xl/workbook.xml" ContentType='
            '"application/vnd.ms-excel.sheet.macroEnabled.main+xml"/></Types>'))
        assert not _magic_ok(sneaky, "xlsx")

    def test_text_must_decode_as_utf8(self):
        assert _magic_ok(CSV, "csv")
        assert not _magic_ok(b"\xff\xfe\x00binary", "csv")

    def test_binary_disguised_as_text_rejected(self):
        assert not _magic_ok(b"col,val\n\x00\x01\x02", "csv")

    def test_empty_rejected(self):
        assert not _magic_ok(b"", "png")


# ----------------------------------------------------------------------------- tools array

class TestToolsArray:
    def _mixin(self):
        h = MagicMock()
        h.log_debug = MagicMock()
        h.mcp_manager.has_mcp_servers.return_value = False
        return h

    def test_added_when_enabled(self):
        with patch("message_processor.handlers.text.config") as cfg:
            cfg.enable_web_search = False
            cfg.enable_code_interpreter = True
            cfg.mcp_enabled_default = False
            tools = TextHandlerMixin._build_tools_array(
                self._mixin(), {"enable_web_search": False, "enable_mcp": False}, "gpt-5.6-sol")
        assert tools == [{"type": "code_interpreter", "container": {"type": "auto"}}]

    def test_absent_when_globally_disabled(self):
        with patch("message_processor.handlers.text.config") as cfg:
            cfg.enable_web_search = True
            cfg.enable_code_interpreter = False
            cfg.mcp_enabled_default = False
            tools = TextHandlerMixin._build_tools_array(
                self._mixin(), {"enable_web_search": True, "enable_mcp": False}, "gpt-5.6-sol")
        assert tools == [{"type": "web_search"}]

    def test_thread_override_beats_global(self):
        with patch("message_processor.handlers.text.config") as cfg:
            cfg.enable_web_search = False
            cfg.enable_code_interpreter = True
            cfg.mcp_enabled_default = False
            tools = TextHandlerMixin._build_tools_array(
                self._mixin(), {"enable_web_search": False, "enable_mcp": False,
                                "enable_code_interpreter": False}, "gpt-5.6-sol")
        assert tools is None

    def test_composes_with_web_search(self):
        with patch("message_processor.handlers.text.config") as cfg:
            cfg.enable_web_search = True
            cfg.enable_code_interpreter = True
            cfg.mcp_enabled_default = False
            tools = TextHandlerMixin._build_tools_array(
                self._mixin(), {"enable_web_search": True, "enable_mcp": False}, "gpt-5.6-sol")
        assert [t["type"] for t in tools] == ["web_search", "code_interpreter"]

    def test_deep_research_does_not_inherit_code_interpreter(self):
        """The research job has no artifact sink and its own delivery path — inheriting CI
        would bill us for a container whose files are then dropped on the floor."""
        import inspect
        from message_processor import research_tools
        src = inspect.getsource(research_tools)
        assert 't.get("type") != "code_interpreter"' in src


# -------------------------------------------------------------- raw-file mounting (the point)

class TestNativeFileMounting:
    """Without this, "analyze the CSV you uploaded" is a lie: the model would only ever see a
    truncated text extraction and would do its arithmetic in its head."""

    def _mixin(self):
        from message_processor.utilities import MessageUtilitiesMixin
        return MessageUtilitiesMixin

    @pytest.mark.parametrize("mime", [
        "text/csv",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
    ])
    def test_spreadsheets_mount_when_code_interpreter_is_on(self, mime):
        m = self._mixin()
        with patch("message_processor.utilities.config") as cfg:
            cfg.enable_native_file_input = True
            cfg.enable_code_interpreter = True
            cfg.native_file_max_mb = 32
            cfg.native_file_max_pages = 100
            assert m._native_file_eligible(m, mime, 1024, None) is True

    def test_spreadsheets_do_not_mount_when_code_interpreter_is_off(self):
        """No sandbox to open it in — mounting would just burn tokens."""
        m = self._mixin()
        with patch("message_processor.utilities.config") as cfg:
            cfg.enable_native_file_input = True
            cfg.enable_code_interpreter = False
            cfg.native_file_max_mb = 32
            cfg.native_file_max_pages = 100
            assert m._native_file_eligible(m, "text/csv", 1024, None) is False

    def test_pdf_still_mounts_regardless(self):
        m = self._mixin()
        with patch("message_processor.utilities.config") as cfg:
            cfg.enable_native_file_input = True
            cfg.enable_code_interpreter = False
            cfg.native_file_max_mb = 32
            cfg.native_file_max_pages = 100
            assert m._native_file_eligible(m, "application/pdf", 1024, 10) is True

    def test_oversized_spreadsheet_falls_back_to_extraction(self):
        m = self._mixin()
        with patch("message_processor.utilities.config") as cfg:
            cfg.enable_native_file_input = True
            cfg.enable_code_interpreter = True
            cfg.native_file_max_mb = 1
            cfg.native_file_max_pages = 100
            assert m._native_file_eligible(m, "text/csv", 5 * 1024 * 1024, None) is False

    def test_unrelated_type_still_refused(self):
        m = self._mixin()
        with patch("message_processor.utilities.config") as cfg:
            cfg.enable_native_file_input = True
            cfg.enable_code_interpreter = True
            cfg.native_file_max_mb = 32
            cfg.native_file_max_pages = 100
            assert m._native_file_eligible(m, "text/x-python", 1024, None) is False

    def test_data_url_carries_the_real_mimetype(self):
        """Hard-coding application/pdf would hand the API a CSV wearing a PDF content type."""
        import inspect
        from message_processor import base as mp_base
        src = inspect.getsource(mp_base)
        assert 'f"data:{mimetype};base64,{doc[\'file_data_b64\']}"' in src


# ------------------------------------------------------------------------ plumbing contracts

class TestSinkPlumbing:
    """The F30.1 bug was a wrapper silently dropping a sink. Guard every seam."""

    @pytest.mark.parametrize("method", [
        "create_text_response_with_tools",
        "create_streaming_response_with_tools",
        "_create_text_response_with_tools_with_timeout",
    ])
    def test_wrappers_accept_and_forward_artifacts_sink(self, method):
        import inspect
        from openai_client.base import OpenAIClient
        fn = getattr(OpenAIClient, method)
        assert "artifacts_sink" in inspect.signature(fn).parameters
        # accepting it but not passing it on is exactly the F30.1 bug — check the body
        assert "artifacts_sink=artifacts_sink" in inspect.getsource(fn)

    @pytest.mark.parametrize("method", [
        "create_text_response_with_tools",
        "create_streaming_response_with_tools",
    ])
    def test_wrappers_accept_mcp_results_sink(self, method):
        """Regression: handlers/text.py always passed this; the wrappers never took it, so the
        no-tool-loop branch raised TypeError."""
        import inspect
        from openai_client.base import OpenAIClient
        params = inspect.signature(getattr(OpenAIClient, method)).parameters
        assert "mcp_results_sink" in params

    def test_base_client_send_file_declines_instead_of_raising(self):
        import inspect
        from base_client import BaseClient
        assert "send_file" not in getattr(BaseClient, "__abstractmethods__", set())
        assert inspect.iscoroutinefunction(BaseClient.send_file)


class TestPersistentContainerWiring:
    """F32.1: the sandbox is scoped to the THREAD, so its id must reach the tools array."""

    def _mixin(self):
        h = MagicMock()
        h.log_debug = MagicMock()
        h.mcp_manager.has_mcp_servers.return_value = False
        return h

    def _ci_tool(self, tools):
        return next(t for t in tools if t["type"] == "code_interpreter")

    def test_thread_container_id_is_sent_verbatim(self):
        """The whole feature: reuse the thread's sandbox rather than a fresh throwaway."""
        with patch("message_processor.handlers.text.config") as cfg:
            cfg.enable_web_search = False
            cfg.enable_code_interpreter = True
            cfg.mcp_enabled_default = False
            tools = TextHandlerMixin._build_tools_array(
                self._mixin(), {"enable_web_search": False, "enable_mcp": False},
                "gpt-5.6-sol", ci_container="cntr_abc123")
        assert self._ci_tool(tools)["container"] == "cntr_abc123"

    def test_falls_back_to_auto_when_unresolved(self):
        """Container trouble must cost continuity, never the tool itself."""
        with patch("message_processor.handlers.text.config") as cfg:
            cfg.enable_web_search = False
            cfg.enable_code_interpreter = True
            cfg.mcp_enabled_default = False
            tools = TextHandlerMixin._build_tools_array(
                self._mixin(), {"enable_web_search": False, "enable_mcp": False},
                "gpt-5.6-sol", ci_container=None)
        assert self._ci_tool(tools)["container"] == {"type": "auto"}

    @pytest.mark.asyncio
    async def test_resolver_returns_none_when_ci_disabled(self):
        """Don't mint a container for a turn that will never use one."""
        h = MagicMock()
        h.container_manager = MagicMock(get_or_create=AsyncMock(return_value="cntr_x"))
        with patch("message_processor.handlers.text.config") as cfg:
            cfg.enable_code_interpreter = False
            got = await TextHandlerMixin._resolve_ci_container(
                h, {"enable_code_interpreter": False}, "C1:1.1")
        assert got is None
        h.container_manager.get_or_create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_resolver_scopes_by_thread_key(self):
        h = MagicMock()
        h.container_manager = MagicMock(get_or_create=AsyncMock(return_value="cntr_x"))
        with patch("message_processor.handlers.text.config") as cfg:
            cfg.enable_code_interpreter = True
            got = await TextHandlerMixin._resolve_ci_container(h, {}, "C1:99.9")
        assert got == "cntr_x"
        h.container_manager.get_or_create.assert_awaited_once_with("C1:99.9")

    @pytest.mark.asyncio
    async def test_resolver_degrades_to_auto_on_failure(self):
        h = MagicMock()
        h.log_warning = MagicMock()
        h.container_manager = MagicMock(
            get_or_create=AsyncMock(side_effect=RuntimeError("openai down")))
        with patch("message_processor.handlers.text.config") as cfg:
            cfg.enable_code_interpreter = True
            got = await TextHandlerMixin._resolve_ci_container(h, {}, "C1:1.1")
        assert got == {"type": "auto"}


@pytest.mark.asyncio
class TestDurableDedupe:
    """A reused container's listing is CUMULATIVE — it holds every file the thread ever wrote.

    The in-memory guard dies with the process, so without a durable record a restart
    mid-conversation re-uploads turn 1's chart alongside turn 2's.
    """

    def _client(self):
        c = MagicMock()
        c.send_file = AsyncMock(return_value={"file_id": "F1", "url_private": "u",
                                              "permalink": "p"})
        return c

    def _openai_listing(self, *file_ids):
        files = [MagicMock(id=fid, source="assistant", path=f"/mnt/data/{fid}.png")
                 for fid in file_ids]

        async def _aiter(*a, **k):
            for f in files:
                yield f

        raw = MagicMock()
        raw.containers.files.list = MagicMock(side_effect=lambda **k: _aiter())
        raw.containers.files.content.with_streaming_response.retrieve = _streaming_retriever(
            lambda fid: _StreamedBody(b"\x89PNG\r\n\x1a\n" + b"payload"))
        oc = MagicMock()
        oc.client = raw
        return oc

    async def test_previously_published_file_is_not_reposted(self):
        artifacts_mod._published_file_ids.clear()
        cm = MagicMock(
            get_published_files=AsyncMock(return_value=["cfile_turn1"]),
            remember_published=AsyncMock(),
        )
        client = self._client()

        published = await publish_artifacts(
            openai_client=self._openai_listing("cfile_turn1"), client=client,
            channel_id="C1", thread_id="1.1", thread_key="C1:1.1",
            container_ids=["cntr_a"], container_manager=cm)

        assert published == []
        client.send_file.assert_not_awaited()

    async def test_new_file_in_a_reused_container_still_publishes(self):
        artifacts_mod._published_file_ids.clear()
        cm = MagicMock(
            get_published_files=AsyncMock(return_value=["cfile_turn1"]),
            remember_published=AsyncMock(),
        )

        published = await publish_artifacts(
            openai_client=self._openai_listing("cfile_turn1", "cfile_turn2"),
            client=self._client(), channel_id="C1", thread_id="1.1", thread_key="C1:1.1",
            container_ids=["cntr_a"], container_manager=cm)

        assert len(published) == 1
        cm.remember_published.assert_awaited_once_with("C1:1.1", "cntr_a", ["cfile_turn2"])

    async def test_published_ids_are_recorded_durably(self):
        artifacts_mod._published_file_ids.clear()
        cm = MagicMock(get_published_files=AsyncMock(return_value=[]),
                       remember_published=AsyncMock())

        await publish_artifacts(
            openai_client=self._openai_listing("cfile_new"), client=self._client(),
            channel_id="C1", thread_id="1.1", thread_key="C1:1.1",
            container_ids=["cntr_a"], container_manager=cm)

        cm.remember_published.assert_awaited_once_with("C1:1.1", "cntr_a", ["cfile_new"])

    async def test_record_lookup_failure_does_not_break_publishing(self):
        """Worst case we re-post a file. Never fail the turn over bookkeeping."""
        artifacts_mod._published_file_ids.clear()
        cm = MagicMock(
            get_published_files=AsyncMock(side_effect=RuntimeError("db gone")),
            remember_published=AsyncMock(),
        )

        published = await publish_artifacts(
            openai_client=self._openai_listing("cfile_new"), client=self._client(),
            channel_id="C1", thread_id="1.1", thread_key="C1:1.1",
            container_ids=["cntr_a"], container_manager=cm)

        assert len(published) == 1

    async def test_works_without_a_container_manager(self):
        """Back-compat: the `auto` path passes no manager."""
        artifacts_mod._published_file_ids.clear()

        published = await publish_artifacts(
            openai_client=self._openai_listing("cfile_new"), client=self._client(),
            channel_id="C1", thread_id="1.1", thread_key="C1:1.1",
            container_ids=["cntr_a"])

        assert len(published) == 1


@pytest.mark.asyncio
class TestDownloadIsBounded:
    """`content.retrieve()` buffers the WHOLE body before returning, so a size check made after
    it is not a cap at all — the model can write a multi-gigabyte file and we are already
    holding it. The download must stream and abort."""

    async def test_oversized_body_is_abandoned_mid_stream(self):
        from message_processor.artifacts import ArtifactRef, _download

        max_bytes = 1024
        chunks_pulled = []

        class _Body:
            headers = {}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *e):
                return False

            async def iter_bytes(self):
                # 100 chunks of 1KB. A bounded reader must stop after ~2, not drain all 100.
                for i in range(100):
                    chunks_pulled.append(i)
                    yield b"x" * 1024

        oc = MagicMock()
        oc.client.containers.files.content.with_streaming_response.retrieve = MagicMock(
            return_value=_Body())

        data = await _download(oc, ArtifactRef("c1", "f1", "huge.png"), max_bytes)

        assert data is None
        assert len(chunks_pulled) < 5, "must abort early, not buffer the whole body"

    async def test_content_length_header_refuses_before_any_read(self):
        from message_processor.artifacts import ArtifactRef, _download

        pulled = []

        class _Body:
            headers = {"content-length": str(500 * 1024 * 1024)}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *e):
                return False

            async def iter_bytes(self):
                pulled.append(1)
                yield b"x"

        oc = MagicMock()
        oc.client.containers.files.content.with_streaming_response.retrieve = MagicMock(
            return_value=_Body())

        assert await _download(oc, ArtifactRef("c1", "f1", "huge.png"), 1024) is None
        assert pulled == [], "declared length already exceeded the cap — read nothing"

    async def test_listing_size_refuses_without_a_request_at_all(self):
        """The listing reports `bytes` for some files. When it does, it is a free rejection."""
        from message_processor.artifacts import ArtifactRef, _download

        oc = MagicMock()
        ref = ArtifactRef("c1", "f1", "huge.png", size_bytes=99 * 1024 * 1024)

        assert await _download(oc, ref, 1024) is None
        oc.client.containers.files.content.with_streaming_response.retrieve.assert_not_called()

    async def test_normal_file_still_downloads(self):
        from message_processor.artifacts import ArtifactRef, _download

        oc = _openai()
        data = await _download(oc, ArtifactRef("c1", "f1", "chart.png"), 25 * 1024 * 1024)
        assert data == PNG

    async def test_listing_carries_the_size_when_the_api_reports_it(self):
        refs = await resolve_container_artifacts(
            _openai([_cfile("f1", "/mnt/data/c.png", size=4242)]), ["c1"])
        assert refs[0].size_bytes == 4242

    async def test_absent_listing_size_is_not_a_rejection(self):
        """`bytes` comes back null for assistant files — that must not block publication."""
        refs = await resolve_container_artifacts(
            _openai([_cfile("f1", "/mnt/data/c.png", size=None)]), ["c1"])
        assert refs[0].size_bytes is None


class TestContainerGoneRecovery:
    """A container verified at turn start can idle-expire before tool-loop round 3. That 404
    would fail the WHOLE turn — the user gets an error instead of an answer."""

    def test_demote_swaps_the_dead_id_for_auto(self):
        from openai_client.container_errors import demote_container_tools

        tools, changed = demote_container_tools([
            {"type": "web_search"},
            {"type": "code_interpreter", "container": "cntr_dead"},
        ])
        assert changed is True
        assert tools[0] == {"type": "web_search"}          # untouched
        assert tools[1]["container"] == {"type": "auto"}

    def test_demote_reports_nothing_to_do_for_an_auto_container(self):
        """Then the 404 was not about a container we chose, and a retry would fail identically."""
        from openai_client.container_errors import demote_container_tools

        tools, changed = demote_container_tools(
            [{"type": "code_interpreter", "container": {"type": "auto"}}])
        assert changed is False

    def test_persistent_ids_extracted_for_invalidation(self):
        from openai_client.container_errors import persistent_container_ids

        assert persistent_container_ids([
            {"type": "code_interpreter", "container": "cntr_a"},
            {"type": "code_interpreter", "container": {"type": "auto"}},
            {"type": "web_search"},
        ]) == ["cntr_a"]

    @pytest.mark.asyncio
    async def test_dead_container_retries_once_with_auto(self):
        from openai_client.api.responses import _create_with_container_recovery

        gone = Exception("Container with id 'cntr_dead' not found.")
        gone.status_code = 404
        calls = []

        async def _safe(_create, operation_type=None, **params):
            calls.append(params["tools"])
            if len(calls) == 1:
                raise gone
            return MagicMock(output=[], usage=None)

        self_ = MagicMock()
        self_._safe_api_call = AsyncMock(side_effect=_safe)
        sink = []

        await _create_with_container_recovery(
            self_,
            {"tools": [{"type": "code_interpreter", "container": "cntr_dead"}], "model": "m"},
            "text_normal", container_gone_sink=sink)

        assert len(calls) == 2, "must retry once rather than fail the turn"
        assert calls[1][0]["container"] == {"type": "auto"}
        assert sink == ["cntr_dead"], "the caller needs the dead id to drop its DB binding"

    @pytest.mark.asyncio
    async def test_an_unrelated_error_is_not_retried(self):
        from openai_client.api.responses import _create_with_container_recovery

        boom = Exception("rate limited")
        boom.status_code = 429
        self_ = MagicMock()
        self_._safe_api_call = AsyncMock(side_effect=boom)

        with pytest.raises(Exception, match="rate limited"):
            await _create_with_container_recovery(
                self_, {"tools": [{"type": "code_interpreter", "container": "c"}]},
                "text_normal")
        assert self_._safe_api_call.await_count == 1


class TestArtifactsSurviveRetries:
    """An attempt that ran code interpreter and THEN failed (an MCP error, a timeout) still left
    real files in the sandbox. A fresh per-attempt sink threw that container id away and the
    file was never published."""

    def test_handlers_accept_a_shared_accumulator(self):
        import inspect
        for fn in (TextHandlerMixin._handle_text_response,
                   TextHandlerMixin._handle_streaming_text_response):
            assert "artifacts_acc" in inspect.signature(fn).parameters

    def test_every_retry_path_forwards_the_accumulator(self):
        """A retry that silently starts a new sink is exactly the bug."""
        import inspect
        src = inspect.getsource(TextHandlerMixin)
        # each re-entry into a handler must carry the accumulator forward
        reentries = src.count("self._handle_text_response(") + \
            src.count("self._handle_streaming_text_response(")
        forwards = src.count("artifacts_acc=")
        # -2 for the two def sites picked up by the count above
        assert forwards >= reentries - 2, "a retry path drops the artifact accumulator"

    def test_turn_level_accumulator_exists_in_process_message(self):
        """base.py's timeout retry re-enters the handler too — it must share the same sink."""
        import inspect
        from message_processor.base import MessageProcessor
        src = inspect.getsource(MessageProcessor.process_message)
        assert "turn_artifacts" in src
        assert src.count("artifacts_acc=turn_artifacts") >= 4


class TestMountingHonoursTheThreadSetting:
    """Attachment mounting and the tools array must resolve the CI setting the SAME way. Reading
    the global in one place and the per-thread override in the other desynchronizes them."""

    class _Util(MessageUtilitiesMixin):
        pass

    def test_thread_off_does_not_mount_a_spreadsheet(self):
        """Global on, thread off: shipping spreadsheet bytes to a model with no sandbox to open
        them is pure wasted tokens."""
        with patch("message_processor.utilities.config") as cfg:
            cfg.enable_native_file_input = True
            cfg.enable_code_interpreter = True          # global ON
            cfg.native_file_max_mb = 25
            cfg.native_file_max_pages = 100
            assert not self._Util()._native_file_eligible(
                "text/csv", 1000, None, code_interpreter_enabled=False)

    def test_thread_on_mounts_even_when_the_global_default_is_off(self):
        """Global off, thread on: the tool is there, so the file it was turned on for must be."""
        with patch("message_processor.utilities.config") as cfg:
            cfg.enable_native_file_input = True
            cfg.enable_code_interpreter = False         # global OFF
            cfg.native_file_max_mb = 25
            cfg.native_file_max_pages = 100
            assert self._Util()._native_file_eligible(
                "text/csv", 1000, None, code_interpreter_enabled=True)

    def test_pdf_is_unaffected_by_the_ci_setting(self):
        """PDFs mount because the API renders their pages, not because of the sandbox."""
        with patch("message_processor.utilities.config") as cfg:
            cfg.enable_native_file_input = True
            cfg.enable_code_interpreter = False
            cfg.native_file_max_mb = 25
            cfg.native_file_max_pages = 100
            assert self._Util()._native_file_eligible(
                "application/pdf", 1000, 3, code_interpreter_enabled=False)

    def test_processor_passes_the_resolved_thread_setting(self):
        import inspect
        from message_processor.base import MessageProcessor
        src = inspect.getsource(MessageProcessor.process_message)
        assert "code_interpreter_enabled=thread_config.get(" in src


@pytest.mark.asyncio
class TestDeadContainerBindingIsDropped:
    """After the API layer rescues a mid-turn container death, the stale DB binding must go —
    otherwise the next turn offers the same corpse and pays a pointless retrieve() to learn it
    is dead."""

    async def test_dead_id_is_invalidated_scoped_to_its_container(self):
        h = MagicMock()
        h.container_manager = MagicMock(invalidate=AsyncMock())
        await TextHandlerMixin._drop_dead_containers(h, ["cntr_dead"], "C1:1.1")
        h.container_manager.invalidate.assert_awaited_once_with("C1:1.1", "cntr_dead")

    async def test_no_deaths_is_a_noop(self):
        h = MagicMock()
        h.container_manager = MagicMock(invalidate=AsyncMock())
        await TextHandlerMixin._drop_dead_containers(h, [], "C1:1.1")
        h.container_manager.invalidate.assert_not_awaited()

    async def test_invalidation_failure_never_breaks_the_turn(self):
        h = MagicMock()
        h.log_warning = MagicMock()
        h.container_manager = MagicMock(invalidate=AsyncMock(side_effect=RuntimeError("db")))
        await TextHandlerMixin._drop_dead_containers(h, ["cntr_dead"], "C1:1.1")  # must not raise

    def test_both_handler_paths_consume_the_sink(self):
        import inspect
        src = inspect.getsource(TextHandlerMixin)
        assert src.count("container_gone_sink=containers_gone") >= 5
        assert src.count("await self._drop_dead_containers(containers_gone") == 2


class TestAttributionHidesInternalProcessing:
    """Attribution answers "where did this come from" — so it lists SOURCES, not plumbing.

    code_interpreter is the model doing its own arithmetic. Showing it told the user nothing
    and put "_Tools Used: code_interpreter_" under every computed answer.
    """

    def test_code_interpreter_is_not_shown(self):
        from message_processor.tool_provenance import visible_attribution_tools
        assert visible_attribution_tools(["code_interpreter"]) == []

    def test_external_sources_are_still_shown(self):
        from message_processor.tool_provenance import visible_attribution_tools
        assert visible_attribution_tools(["web_search", "code_interpreter", "datassential"]) == [
            "web_search", "datassential"]

    def test_empty_is_safe(self):
        from message_processor.tool_provenance import visible_attribution_tools
        assert visible_attribution_tools(None) == []

    def test_provenance_record_still_keeps_code_interpreter(self):
        """The model must still be able to answer "how did you get that?" — the F7 record is a
        separate list and is NOT filtered."""
        import inspect
        from message_processor.handlers.text import TextHandlerMixin
        src = inspect.getsource(TextHandlerMixin)
        # the attribution note is built from the filtered list...
        assert "', '.join(attribution_tools)" in src
        # ...and the filter is applied to a SEPARATE variable, not in place
        assert "attribution_tools = visible_attribution_tools(" in src


class TestProvenanceEchoIsStripped:
    """The model IMITATES the `[used tools: …]` annotations it sees on its own past replies.

    Observed live: a reply that read "56,088\\n[used tools: code_interpreter]". These lines are
    ours to write, never the model's, so they never survive into a posted message.
    """

    def test_echoed_used_tools_line_is_removed(self):
        from message_processor.tool_provenance import strip_provenance_echo
        assert strip_provenance_echo("56,088\n[used tools: code_interpreter]") == "56,088"

    def test_echoed_tool_results_line_is_removed(self):
        from message_processor.tool_provenance import strip_provenance_echo
        assert strip_provenance_echo("Answer.\n[tool results: srv → x]") == "Answer."

    def test_ordinary_text_untouched(self):
        from message_processor.tool_provenance import strip_provenance_echo
        text = "Here are the totals:\n• East: 45,868\n[see the attached chart]"
        assert strip_provenance_echo(text) == text

    def test_no_annotation_is_a_cheap_noop(self):
        from message_processor.tool_provenance import strip_provenance_echo
        assert strip_provenance_echo("plain answer") == "plain answer"

    def test_applied_on_every_path_that_posts_text(self):
        import inspect
        from message_processor.handlers.text import TextHandlerMixin
        src = inspect.getsource(TextHandlerMixin)
        # both the non-streaming and streaming commits, and the native finalize
        assert src.count("strip_provenance_echo(strip_sandbox_links(") == 3


class TestAbortedStreamDoesNotDuplicateTheAnswer:
    """The "42 / 42" bug, seen live.

    A native stream had committed "42" when its container died. Cleanup called chat.update on a
    message still in Slack's STREAMING state, which Slack refuses (streaming_state_conflict), so
    the partial was never removed — and the non-streaming fallback then posted the answer again.
    Two real messages, same answer.

    The stream must be STOPPED (abandon → stopStream) before its message is touchable, and the
    dead partial deleted before the retry re-answers.
    """

    def test_native_stream_is_abandoned_before_the_fallback(self):
        import inspect
        from message_processor.handlers.text import TextHandlerMixin
        src = inspect.getsource(TextHandlerMixin)
        assert "Native stream abandon failed before non-streaming fallback" in src

    def test_abandoned_partial_is_deleted_so_the_retry_does_not_duplicate_it(self):
        import inspect
        from message_processor.handlers.text import TextHandlerMixin
        src = inspect.getsource(TextHandlerMixin)
        assert "Deleted abandoned partial stream before non-streaming fallback" in src

    def test_mcp_streaming_retry_keeps_its_partial(self):
        """An MCP failure retries WITH streaming and continues the same message — deleting the
        partial there would throw away text we are about to keep building on."""
        import inspect
        from message_processor.handlers.text import TextHandlerMixin
        src = inspect.getsource(TextHandlerMixin)
        assert "and native_coord.current_ts and not failed_mcp_server" in src
