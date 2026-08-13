"""F35 — mount_file: putting a thread file's REAL bytes into the code sandbox."""
import gzip
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


from message_processor import file_mount
from message_processor.image_tools import CI_CONTAINER_KEY
from tool_registry import ToolContext


class _BadRequest(Exception):
    """The upload endpoint's CONTENT refusal, in the shape the SDK raises it: the class name
    and `status_code` 400 are both what file_mount's classifier reads."""

    def __init__(self, message: str):
        super().__init__(message)
        self.status_code = 400


@pytest.fixture(autouse=True)
def _clear_mount_cache():
    # The cache is process-lifetime by design; tests must not inherit each other's mounts.
    file_mount._MOUNTS.clear()
    yield
    file_mount._MOUNTS.clear()


def _entry(file_id="file_doc_1", filename="sales.csv", **kw):
    base = {"file_id": file_id, "kind": "document", "origin": "uploaded",
            "filename": filename, "mime_type": "text/csv", "size_bytes": 12,
            "url": "https://files.slack.com/sales.csv", "slack_file_id": "F1",
            "description": "sales", "created_at": "2026-07-12T10:00:00"}
    base.update(kw)
    return base


def _ctx(entries=None, container="cntr_abc", data=b"region,rev\nEast,60000\n"):
    client = MagicMock()
    client.download_file = AsyncMock(return_value=data)

    created = SimpleNamespace(id="cfile_1", path="/mnt/data/sales.csv")
    raw = MagicMock()
    raw.containers.files.create = AsyncMock(return_value=created)
    processor = MagicMock()
    processor.openai_client.client = raw

    return ToolContext(
        channel_id="C1", thread_ts="123.45", client=client, processor=processor,
        container_id=container,
        thread_files=entries if entries is not None else [_entry()],
        mounted_files=[],
    ), raw


@pytest.mark.unit
class TestContainerRecycled:
    """F15: ToolContext.container_recycled() — the boolean the byte-pushing tools gate on."""

    def test_false_when_sink_empty_or_none(self):
        assert ToolContext(container_id="c1").container_recycled() is False
        assert ToolContext(container_id="c1", container_gone_sink=[]).container_recycled() is False

    def test_true_when_own_container_is_in_the_sink(self):
        assert ToolContext(container_id="c1",
                           container_gone_sink=["c1"]).container_recycled() is True

    def test_false_for_a_different_dead_container(self):
        assert ToolContext(container_id="c1",
                           container_gone_sink=["c2"]).container_recycled() is False

    def test_false_when_there_is_no_container(self):
        assert ToolContext(container_id=None,
                           container_gone_sink=["c1"]).container_recycled() is False


@pytest.mark.unit
class TestSchemaGating:
    def test_offered_without_an_addressable_container(self):
        # W3: every turn now starts on {"type":"auto"}, so hiding the tool here would hide it on
        # the first turn of every conversation — exactly the turn someone drops a spreadsheet in.
        # The executor mints an addressable container on demand instead.
        cfg = {CI_CONTAINER_KEY: None, file_mount.FILES_KEY: [_entry()]}
        assert file_mount.get_mount_file_schema(cfg)["name"] == "mount_file"

    def test_hidden_with_no_files(self):
        cfg = {CI_CONTAINER_KEY: "cntr_abc", file_mount.FILES_KEY: []}
        assert file_mount.get_mount_file_schema(cfg) is None

    def test_ids_are_a_literal_enum(self):
        cfg = {CI_CONTAINER_KEY: "cntr_abc",
               file_mount.FILES_KEY: [_entry("file_doc_1"), _entry("file_img_9")]}
        schema = file_mount.get_mount_file_schema(cfg)
        assert schema["name"] == "mount_file"
        enum = schema["parameters"]["properties"]["file_id"]["enum"]
        assert enum == ["file_doc_1", "file_img_9"]
        # The model must be able to tell them apart without guessing.
        assert "sales.csv" in schema["description"]

    def test_no_thread_config_hides_it(self):
        assert file_mount.get_mount_file_schema(None) is None


@pytest.mark.unit
class TestSandboxSwitchGating:
    """W3 made the sandbox switch load-bearing for this tool. It used to need an addressable
    container id, which a turn with code interpreter OFF could never have; now the executor
    MINTS one, so a turn whose request carries no code_interpreter declaration would bind a
    container nothing can open. The switch is resolved per-thread, exactly as the tools array
    resolves it."""

    def _names(self, cfg, surface=None):
        from tool_registry import ToolRegistry
        registry = ToolRegistry()
        file_mount.register_file_mount_tools(registry)
        kwargs = {"surface": surface} if surface else {}
        return {s["name"] for s in registry.schemas(cfg, **kwargs)}

    def test_hidden_on_both_surfaces_when_the_thread_turned_the_sandbox_off(self):
        cfg = {"enable_code_interpreter": False, file_mount.FILES_KEY: [_entry()]}
        assert "mount_file" not in self._names(cfg)
        assert "mount_file" not in self._names(cfg, surface="channel")

    def test_hidden_on_both_surfaces_when_the_sandbox_is_off_globally(self, monkeypatch):
        from config import config as cfg_obj
        monkeypatch.setattr(cfg_obj, "enable_code_interpreter", False)
        cfg = {file_mount.FILES_KEY: [_entry()]}
        assert "mount_file" not in self._names(cfg)
        assert "mount_file" not in self._names(cfg, surface="channel")

    def test_the_thread_override_beats_a_global_off(self, monkeypatch):
        from config import config as cfg_obj
        monkeypatch.setattr(cfg_obj, "enable_code_interpreter", False)
        cfg = {"enable_code_interpreter": True, file_mount.FILES_KEY: [_entry()]}
        assert "mount_file" in self._names(cfg)
        assert "mount_file" in self._names(cfg, surface="channel")

    def test_an_unaddressable_container_is_still_offered(self, monkeypatch):
        """The gate asks whether the sandbox is ON, never whether it has an id yet — `auto` is
        the normal state of a first turn and the tool belongs there."""
        from config import config as cfg_obj
        monkeypatch.setattr(cfg_obj, "enable_code_interpreter", True)
        cfg = {CI_CONTAINER_KEY: None, file_mount.FILES_KEY: [_entry()]}
        assert "mount_file" in self._names(cfg)
        assert "mount_file" in self._names(cfg, surface="channel")


@pytest.mark.unit
class TestExecute:
    async def test_mounts_and_returns_the_api_assigned_path(self):
        ctx, raw = _ctx()
        result = await file_mount.execute_mount_file(ctx, {"file_id": "file_doc_1"})

        assert result["ok"] is True
        assert result["path"] == "/mnt/data/sales.csv"
        raw.containers.files.create.assert_awaited_once()
        assert raw.containers.files.create.call_args.kwargs["container_id"] == "cntr_abc"

    async def test_recycled_container_is_refused_without_uploading(self):
        # F15: the sandbox idle-expired mid-turn and its id landed in container_gone_sink (the
        # SAME list the API records dead containers into). Mounting into the corpse would be
        # invisible to the model, so fail fast — no download, no container upload.
        ctx, raw = _ctx()
        ctx.container_gone_sink = ["cntr_abc"]  # == ctx.container_id
        result = await file_mount.execute_mount_file(ctx, {"file_id": "file_doc_1"})

        assert result["ok"] is False
        assert result["error"] == "container_recycled"
        ctx.client.download_file.assert_not_awaited()
        raw.containers.files.create.assert_not_awaited()

    async def test_unrelated_dead_container_does_not_block_mount(self):
        # A different thread's dead container in the sink must not trip our live one.
        ctx, raw = _ctx()
        ctx.container_gone_sink = ["cntr_someone_else"]
        result = await file_mount.execute_mount_file(ctx, {"file_id": "file_doc_1"})

        assert result["ok"] is True
        raw.containers.files.create.assert_awaited_once()

    async def test_an_auto_turn_mints_a_container_and_mounts_into_it(self):
        """W3's bridge. The turn started on {"type":"auto"} — no addressable id — and mounting
        is precisely what happens BEFORE the model runs any code, so waiting for adoption would
        never resolve. The executor asks for a container, and the tool loop names it in the next
        round's declaration so the model can open the file."""
        from tool_registry import SandboxHolder

        async def _fill(thread_key, h):
            h.container_id = "cntr_made"      # what ContainerManager.bridge_container does
            return "cntr_made"

        ctx, raw = _ctx(container=None)
        manager = MagicMock(bridge_container=AsyncMock(side_effect=_fill))
        ctx.sandbox = SandboxHolder(manager=manager, thread_key="C1:123.45")

        result = await file_mount.execute_mount_file(ctx, {"file_id": "file_doc_1"})

        assert result["ok"] is True
        assert raw.containers.files.create.call_args.kwargs["container_id"] == "cntr_made"
        # SHARED, not copied: the loop and the sibling calls read the same answer.
        assert ctx.sandbox.container_id == "cntr_made"

    async def test_no_sandbox_available_is_refused_without_downloading(self):
        ctx, raw = _ctx(container=None)   # no holder either — nothing can mint one

        result = await file_mount.execute_mount_file(ctx, {"file_id": "file_doc_1"})

        assert result["ok"] is False and result["error"] == "sandbox_unavailable"
        ctx.client.download_file.assert_not_awaited()
        raw.containers.files.create.assert_not_awaited()

    async def test_an_unadvertised_id_is_refused(self):
        ctx, raw = _ctx()
        result = await file_mount.execute_mount_file(ctx, {"file_id": "file_doc_999"})

        assert result["ok"] is False
        assert result["error"] == "unknown_file_id"
        # Never guess at what was meant — the wrong file silently corrupts what's built from it.
        raw.containers.files.create.assert_not_awaited()

    async def test_mounting_is_idempotent(self):
        ctx, raw = _ctx()
        first = await file_mount.execute_mount_file(ctx, {"file_id": "file_doc_1"})
        second = await file_mount.execute_mount_file(ctx, {"file_id": "file_doc_1"})

        assert second["path"] == first["path"]
        assert second["already_mounted"] is True
        assert raw.containers.files.create.await_count == 1

    async def test_a_new_container_re_mounts(self):
        # The lifeline for "come back after lunch": the old container expired, so the same file
        # must go into the new one rather than resolving to a stale path.
        ctx, raw = _ctx()
        await file_mount.execute_mount_file(ctx, {"file_id": "file_doc_1"})
        ctx.container_id = "cntr_fresh"
        result = await file_mount.execute_mount_file(ctx, {"file_id": "file_doc_1"})

        assert result["ok"] is True
        assert raw.containers.files.create.await_count == 2
        assert raw.containers.files.create.call_args.kwargs["container_id"] == "cntr_fresh"

    async def test_a_later_turn_reuses_a_live_containers_mount(self):
        # Round 2 of an iteration ("make the logo bigger") gets a FRESH ToolContext but the
        # same live container. Re-uploading the same asset every round is pure waste.
        first_ctx, raw = _ctx()
        await file_mount.execute_mount_file(first_ctx, {"file_id": "file_doc_1"})

        next_turn_ctx, _ = _ctx()
        next_turn_ctx.processor = first_ctx.processor  # same process, same container
        result = await file_mount.execute_mount_file(next_turn_ctx, {"file_id": "file_doc_1"})

        assert result["already_mounted"] is True
        assert result["path"] == "/mnt/data/sales.csv"
        assert raw.containers.files.create.await_count == 1
        # The digest still has to reach this turn's context, or the publisher could post the
        # user's own file back at them.
        assert file_mount.mounted_digests(next_turn_ctx)

    async def test_deleted_slack_file_is_honest(self):
        ctx, _ = _ctx(data=None)
        result = await file_mount.execute_mount_file(ctx, {"file_id": "file_doc_1"})
        assert result["ok"] is False
        assert result["error"] == "file_unavailable"

    async def test_oversize_file_refused(self, monkeypatch):
        monkeypatch.setattr(file_mount.config, "artifact_max_mb", 1)
        ctx, raw = _ctx(data=b"x" * (2 * 1024 * 1024))
        result = await file_mount.execute_mount_file(ctx, {"file_id": "file_doc_1"})

        assert result["ok"] is False
        assert result["error"] == "file_too_large"
        raw.containers.files.create.assert_not_awaited()

    async def test_no_container_is_refused(self):
        ctx, _ = _ctx(container=None)
        result = await file_mount.execute_mount_file(ctx, {"file_id": "file_doc_1"})
        assert result["error"] == "sandbox_unavailable"

    async def test_upload_without_a_path_is_a_failure(self):
        # The API assigns the path; without one the model has no way to open the file.
        ctx, raw = _ctx()
        raw.containers.files.create = AsyncMock(
            return_value=SimpleNamespace(id="cfile_1", path=None))
        result = await file_mount.execute_mount_file(ctx, {"file_id": "file_doc_1"})
        assert result["error"] == "mount_failed"

    async def test_records_the_digest_so_the_input_cannot_be_published_back(self):
        data = b"region,rev\nEast,60000\n"
        ctx, _ = _ctx(data=data)
        await file_mount.execute_mount_file(ctx, {"file_id": "file_doc_1"})

        assert file_mount.mounted_digests(ctx) == [hashlib.sha256(data).hexdigest()]


@pytest.mark.unit
class TestRefusedBytesRideGzip:
    """The upload endpoint judges CONTENT, not filenames (probed live 2026-08-12: SVG refused
    under .svg/.xml/.dat/.bin/.svg.txt; the same bytes gzipped accepted). A file a USER shared
    hits that wall exactly like a fetched one, so mount_file goes through the same transport."""

    async def test_a_refused_file_is_mounted_gzip_wrapped_and_says_so(self):
        svg = b'<svg xmlns="http://www.w3.org/2000/svg"><rect width="4" height="4"/></svg>'
        ctx, raw = _ctx(entries=[_entry(filename="logo.svg", mime_type="image/svg+xml")],
                        data=svg)
        seen = []

        async def _create(container_id, file):
            seen.append((file.name, file.getvalue()))
            if len(seen) == 1:
                raise _BadRequest("You uploaded an invalid file")
            return SimpleNamespace(id="cfile_1", path=f"/mnt/data/{file.name}")

        raw.containers.files.create = AsyncMock(side_effect=_create)

        result = await file_mount.execute_mount_file(ctx, {"file_id": "file_doc_1"})

        assert result["ok"] is True and result["gzipped"] is True
        assert seen[0] == ("logo.svg", svg)          # the real bytes were offered first
        assert seen[1][0] == "logo.svg.gz"
        assert gzip.decompress(seen[1][1]) == svg
        assert "GZIP-COMPRESSED" in result["message"]

    async def test_the_cache_hit_repeats_the_gzip_note(self):
        """Round two returns from the cache. Saying it plainly only on the round that uploaded
        would leave the next round handing a gzip blob to a parser that fails on it."""
        svg = b"<svg/>"
        ctx, raw = _ctx(entries=[_entry(filename="logo.svg")], data=svg)
        seen = []

        async def _create(container_id, file):
            seen.append(file.name)
            if len(seen) == 1:
                raise _BadRequest("You uploaded an invalid file")
            return SimpleNamespace(id="cfile_1", path=f"/mnt/data/{file.name}")

        raw.containers.files.create = AsyncMock(side_effect=_create)
        await file_mount.execute_mount_file(ctx, {"file_id": "file_doc_1"})

        again = await file_mount.execute_mount_file(ctx, {"file_id": "file_doc_1"})

        assert again["already_mounted"] is True and again["gzipped"] is True
        assert "GZIP-COMPRESSED" in again["message"]
        assert len(seen) == 2  # nothing re-uploaded

    async def test_an_ordinary_mount_says_nothing_about_gzip(self):
        ctx, _ = _ctx()
        result = await file_mount.execute_mount_file(ctx, {"file_id": "file_doc_1"})

        assert "gzipped" not in result and "GZIP" not in result["message"]

    async def test_a_refusal_that_survives_gzipping_fails_honestly(self):
        ctx, raw = _ctx()
        raw.containers.files.create = AsyncMock(
            side_effect=_BadRequest("You uploaded an invalid file"))

        result = await file_mount.execute_mount_file(ctx, {"file_id": "file_doc_1"})

        assert result["ok"] is False and result["error"] == "mount_failed"
        assert raw.containers.files.create.await_count == 2  # one retry, not a loop


@pytest.mark.unit
class TestSafeName:
    @pytest.mark.parametrize("raw,expected", [
        ("../../etc/passwd", "etcpasswd"),   # separators dropped, then the leading dots
        ("a/b.csv", "ab.csv"),
        (".hidden", "hidden"),
        ("", "file"),
    ])
    def test_cannot_escape_mnt_data(self, raw, expected):
        assert file_mount._safe_name(raw) == expected


@pytest.mark.unit
class TestStaticChannelSchema:
    """On the channel surface the tools array must not move when a thread gains a file or a
    sandbox — both are per-thread facts, and the array sits in the cached prefix. So the schema
    stops gating on them and the executor answers instead."""

    def _cfgs(self):
        return [
            None, {},
            {CI_CONTAINER_KEY: "cntr_abc", file_mount.FILES_KEY: [_entry()]},
            {CI_CONTAINER_KEY: None, file_mount.FILES_KEY: []},
            {CI_CONTAINER_KEY: "cntr_other",
             file_mount.FILES_KEY: [_entry(file_id="file_doc_9", filename="q3.xlsx")],
             "user_id": "U_B"},
        ]

    def test_byte_stable_across_containers_files_and_requesters(self):
        import json
        rendered = {json.dumps(file_mount.get_mount_file_schema_static(cfg), sort_keys=True)
                    for cfg in self._cfgs()}
        assert len(rendered) == 1

    def test_no_file_ids_or_enum_leak_in(self):
        import json
        schema = file_mount.get_mount_file_schema_static(
            {CI_CONTAINER_KEY: "cntr_abc", file_mount.FILES_KEY: [_entry()]})
        assert "enum" not in schema["parameters"]["properties"]["file_id"]
        blob = json.dumps(schema)
        assert "file_doc_1" not in blob and "sales.csv" not in blob
        assert "evidence" in schema["description"]
        # The steering that made the tool used correctly has to survive the trim.
        assert "INGREDIENT" in schema["description"]
        assert "idempotent" in schema["description"]

    def test_never_hidden(self):
        for cfg in self._cfgs():
            assert file_mount.get_mount_file_schema_static(cfg)["name"] == "mount_file"

    def test_the_dynamic_factory_is_untouched(self):
        cfg = {CI_CONTAINER_KEY: "cntr_abc", file_mount.FILES_KEY: [_entry()]}
        assert file_mount.get_mount_file_schema(cfg)["parameters"]["properties"][
            "file_id"]["enum"] == ["file_doc_1"]
        assert file_mount.get_mount_file_schema({CI_CONTAINER_KEY: "cntr_abc",
                                                 file_mount.FILES_KEY: []}) is None


@pytest.mark.unit
class TestHonestEmptyAnswers:
    """Everything the static schema stopped gating on becomes an executor answer the model can
    act on — never a silent success and never a guess."""

    async def test_no_files_in_the_thread_says_so(self):
        ctx, raw = _ctx(entries=[])
        result = await file_mount.execute_mount_file(ctx, {"file_id": "file_doc_1"})

        assert result["ok"] is False and result["error"] == "unknown_file_id"
        assert result["valid_file_ids"] == []
        assert "no files in this thread" in result["message"]
        raw.containers.files.create.assert_not_awaited()

    async def test_an_unknown_id_lists_the_ids_that_would_have_worked(self):
        ctx, _ = _ctx()
        result = await file_mount.execute_mount_file(ctx, {"file_id": "file_doc_999"})

        assert result["valid_file_ids"] == ["file_doc_1"]
        assert "not a file in this thread" in result["message"]

    async def test_no_sandbox_says_sandbox_unavailable(self):
        ctx, raw = _ctx(container=None)
        result = await file_mount.execute_mount_file(ctx, {"file_id": "file_doc_1"})

        assert result["ok"] is False and result["error"] == "sandbox_unavailable"
        raw.containers.files.create.assert_not_awaited()


@pytest.mark.unit
class TestFileEvidenceLines:
    def test_lines_carry_the_ids_names_and_types(self):
        from message_processor import thread_files

        lines = thread_files.catalog_evidence_lines(
            [_entry(), _entry(file_id="file_img_2", filename="shot.png",
                              mime_type="image/png", description="a screenshot")])

        assert lines[0] == thread_files.EVIDENCE_HEADER
        assert len(lines) == 3
        assert "file_doc_1" in lines[1] and "sales.csv" in lines[1]
        assert "file_img_2" in lines[2] and "a screenshot" in lines[2]
        assert all("\n" not in line for line in lines)

    def test_an_empty_catalog_is_stated_not_omitted(self):
        from message_processor import thread_files

        assert thread_files.catalog_evidence_lines([]) == \
               thread_files.catalog_evidence_lines(None)
        assert "none" in thread_files.catalog_evidence_lines([])[1]
