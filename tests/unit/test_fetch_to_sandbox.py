"""fetch_url_to_sandbox — web bytes into the sandbox, and the raw fetch mode behind it.

Two things are being defended. The fetcher's raw mode must return the BODY of anything without
loosening a single guard above it (the SSRF validation, the byte cap and the timeouts are the
same code path). And the executor must behave like every other byte-pushing tool: the dead
container is refused, the digest is recorded so the raw file can never be posted, and no failure
escapes as an exception.
"""
import gzip
import hashlib
import socket
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

import pytest

import message_processor.ingestion.ambient_fetch as ambient_fetch
from message_processor import fetch_to_sandbox
from message_processor.tool_registry import ToolContext, ToolRegistry

SVG = b'<svg xmlns="http://www.w3.org/2000/svg"><rect width="4" height="4"/></svg>'
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32

FETCH_KW = dict(max_bytes=1_000_000, connect_timeout=1, read_timeout=1, total_timeout=2,
                max_redirects=3, max_chars=0)


@pytest.fixture(autouse=True)
def _reset_published_memory():
    """The staging tests below use file id "f1", which is the id half the artifact suites reach
    for, and publish_artifacts remembers published ids process-wide. Any earlier test that
    published an "f1" makes this file's candidate look already-published, and it is dropped
    before staging can hold it back — which reads as the suppression guard failing. Clearing at
    both ends protects this file whichever way the suite is ordered. Same fixture as
    test_artifacts.py."""
    from message_processor import artifacts as artifacts_mod

    artifacts_mod._published_file_ids.clear()
    yield
    artifacts_mod._published_file_ids.clear()


@pytest.fixture
def _fetch_seams():
    yield
    ambient_fetch.set_resolver(
        lambda host, port: socket.getaddrinfo(host, port or 0, proto=socket.IPPROTO_TCP))
    ambient_fetch.set_opener(None)


class BadRequestError(Exception):
    """The upload endpoint's content refusal, in the shape the SDK raises it — the class name
    and `status_code` 400 are both what the classifier reads. Constructing the real
    `openai.BadRequestError` needs an httpx response; this carries the two facts that matter."""

    def __init__(self, message: str):
        super().__init__(message)
        self.status_code = 400


def _resolver_ok(host, port):
    return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
             ("93.184.216.34", port or 0))]


def _opener_returning(status=200, headers=None, chunks=()):
    async def _opener(url, validated_ips, **kw):
        async def _iter(chunk_size):
            for c in chunks:
                yield c

        async def _release():
            return None

        return ambient_fetch._RawResponse(status=status, headers=headers or {}, url=url,
                                          iter_chunks=_iter, release=_release)
    return _opener


def _ctx(container="cntr_x", gone=None):
    created: List[Any] = []

    async def _create(container_id, file):
        created.append((container_id, file.name, file.getvalue()))
        return SimpleNamespace(id="cfile_1", path=f"/mnt/data/{file.name}")

    raw = MagicMock()
    raw.containers.files.create = AsyncMock(side_effect=_create)
    processor = MagicMock()
    processor.openai_client.client = raw
    ctx = ToolContext(channel_id="C0FETCH01", thread_ts="1.0", client=MagicMock(),
                      processor=processor, container_id=container,
                      container_gone_sink=list(gone or []), mounted_files=[])
    return ctx, created


class _Listing:
    """containers.files.list() is an async-iterable paginator, and content retrieval is an async
    context manager pulled in chunks — the two shapes stage_artifacts actually consumes."""

    def __init__(self, files: List[Any], payload: bytes):
        self._files = files
        self._payload = payload

    def pager(self, **kwargs: Any):
        files = self._files

        class _Pager:
            def __aiter__(self):
                async def _gen():
                    for f in files:
                        yield f
                return _gen()

        return _Pager()

    def body(self, file_id: str, container_id: Any = None):
        payload = self._payload

        class _Body:
            headers: dict = {}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def iter_bytes(self):
                yield payload

        return _Body()


def _container_listing(files: List[Any], payload: bytes):
    oc = MagicMock()
    listing = _Listing(files, payload)
    oc.client.containers.files.list = MagicMock(side_effect=listing.pager)
    oc.client.containers.files.content.with_streaming_response.retrieve = MagicMock(
        side_effect=listing.body)
    return oc


def _staged(url: str, **kwargs: Any):
    assert kwargs.get("raw_mode") is True, "the sandbox fetch must run in raw mode"
    return ambient_fetch.FetchResult(kind="bytes", final_url="https://logo.example/mark.svg",
                                     content_type="image/svg+xml", raw_bytes=SVG)


@pytest.mark.unit
@pytest.mark.asyncio
class TestRawFetchMode:
    async def test_it_returns_bytes_for_a_type_the_text_branch_refuses(self, _fetch_seams):
        ambient_fetch.set_resolver(_resolver_ok)
        ambient_fetch.set_opener(_opener_returning(
            headers={"Content-Type": "image/svg+xml"}, chunks=[SVG]))

        refused = await ambient_fetch.fetch_url("https://x.example/m.svg", **FETCH_KW)
        assert refused.kind == "error" and refused.error_code == ambient_fetch.ERR_UNSUPPORTED_TYPE

        raw = await ambient_fetch.fetch_url("https://x.example/m.svg", raw_mode=True, **FETCH_KW)
        assert raw.kind == "bytes" and raw.raw_bytes == SVG
        assert raw.content_type == "image/svg+xml" and raw.ok is True

    async def test_raw_mode_never_runs_extraction(self, monkeypatch, _fetch_seams):
        def _boom(*a, **k):
            raise AssertionError("extraction ran for a raw fetch")

        monkeypatch.setattr(ambient_fetch, "_extract", _boom)
        ambient_fetch.set_resolver(_resolver_ok)
        ambient_fetch.set_opener(_opener_returning(
            headers={"Content-Type": "text/html"}, chunks=[b"<p>words</p>"]))

        res = await ambient_fetch.fetch_url("https://x.example/p", raw_mode=True, **FETCH_KW)
        assert res.kind == "bytes" and res.raw_bytes == b"<p>words</p>"

    async def test_an_undeclared_type_falls_back_to_the_magic_sniff(self, _fetch_seams):
        ambient_fetch.set_resolver(_resolver_ok)
        ambient_fetch.set_opener(_opener_returning(chunks=[PNG]))

        res = await ambient_fetch.fetch_url("https://x.example/i", raw_mode=True, **FETCH_KW)
        assert res.kind == "bytes" and res.content_type == "image/png"

    async def test_the_byte_cap_still_bounds_a_raw_fetch(self, _fetch_seams):
        ambient_fetch.set_resolver(_resolver_ok)
        ambient_fetch.set_opener(_opener_returning(
            headers={"Content-Type": "application/zip"}, chunks=[b"x" * 5000]))

        kw = dict(FETCH_KW, max_bytes=100)
        res = await ambient_fetch.fetch_url("https://x.example/big.zip", raw_mode=True, **kw)
        assert res.kind == "error" and res.error_code == ambient_fetch.ERR_TOO_LARGE

    async def test_the_ssrf_guard_still_refuses_a_private_address(self, _fetch_seams):
        def _internal(host, port):
            return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "",
                     ("169.254.169.254", port or 0))]

        ambient_fetch.set_resolver(_internal)
        ambient_fetch.set_opener(_opener_returning(chunks=[SVG]))

        res = await ambient_fetch.fetch_url("https://metadata.example/x", raw_mode=True,
                                            **FETCH_KW)
        assert res.kind == "error" and res.error_code == ambient_fetch.ERR_BLOCKED_SSRF

    async def test_a_blocked_slack_host_is_still_refused(self, _fetch_seams):
        ambient_fetch.set_resolver(_resolver_ok)
        ambient_fetch.set_opener(_opener_returning(chunks=[SVG]))
        res = await ambient_fetch.fetch_url("https://files.slack.com/x.svg", raw_mode=True,
                                            **FETCH_KW)
        assert res.kind == "error" and res.error_code == ambient_fetch.ERR_BLOCKED_SSRF


@pytest.mark.unit
@pytest.mark.asyncio
class TestExecutor:
    async def test_it_stages_the_bytes_and_hands_back_the_path(self, monkeypatch):
        monkeypatch.setattr(ambient_fetch, "fetch_url", AsyncMock(side_effect=_staged))
        ctx, created = _ctx()

        result = await fetch_to_sandbox.execute_fetch_url_to_sandbox(
            ctx, {"url": "https://logo.example/mark.svg"})

        assert result["ok"] is True and result["path"] == "/mnt/data/mark.svg"
        assert result["content_type"] == "image/svg+xml"
        assert result["size_bytes"] == len(SVG)
        assert created[0][2] == SVG

    async def test_the_url_names_the_file_and_an_override_wins(self, monkeypatch):
        monkeypatch.setattr(ambient_fetch, "fetch_url", AsyncMock(side_effect=_staged))
        ctx, created = _ctx()

        await fetch_to_sandbox.execute_fetch_url_to_sandbox(
            ctx, {"url": "https://logo.example/mark.svg", "filename": "brand.svg"})

        assert created[0][1] == "brand.svg"

    async def test_a_pathless_url_is_named_from_its_content_type(self):
        assert fetch_to_sandbox._derive_filename(
            "https://logo.example/", "image/svg+xml; charset=utf-8", None) == "download.svg"

    async def test_the_digest_is_recorded_so_the_raw_file_is_never_posted(self, monkeypatch):
        monkeypatch.setattr(ambient_fetch, "fetch_url", AsyncMock(side_effect=_staged))
        ctx, _ = _ctx()

        await fetch_to_sandbox.execute_fetch_url_to_sandbox(
            ctx, {"url": "https://logo.example/mark.svg"})

        from message_processor.file_mount import mounted_digests
        assert len(mounted_digests(ctx)) == 1

    async def test_a_recycled_container_refuses_rather_than_writing_into_a_corpse(self,
                                                                                 monkeypatch):
        monkeypatch.setattr(ambient_fetch, "fetch_url", AsyncMock(side_effect=_staged))
        ctx, created = _ctx(container="cntr_dead", gone=["cntr_dead"])

        result = await fetch_to_sandbox.execute_fetch_url_to_sandbox(
            ctx, {"url": "https://logo.example/mark.svg"})

        assert result["ok"] is False and result["error"] == "container_recycled"
        assert created == []

    async def test_no_sandbox_is_an_honest_refusal(self, monkeypatch):
        monkeypatch.setattr(ambient_fetch, "fetch_url", AsyncMock(side_effect=_staged))
        ctx, created = _ctx(container=None)

        result = await fetch_to_sandbox.execute_fetch_url_to_sandbox(
            ctx, {"url": "https://logo.example/mark.svg"})

        assert result["ok"] is False and result["error"] == "sandbox_unavailable"
        assert created == []

    async def test_a_failed_fetch_carries_the_code_and_a_redacted_url(self, monkeypatch):
        failure = ambient_fetch.FetchResult(
            kind="error", error_code=ambient_fetch.ERR_BLOCKED_SSRF, error_detail="blocked",
            final_url="https://logo.example/mark.svg?sig=SECRET")
        monkeypatch.setattr(ambient_fetch, "fetch_url",
                            AsyncMock(return_value=failure))
        ctx, created = _ctx()

        result = await fetch_to_sandbox.execute_fetch_url_to_sandbox(
            ctx, {"url": "https://logo.example/mark.svg?sig=SECRET"})

        assert result["ok"] is False and result["error"] == ambient_fetch.ERR_BLOCKED_SSRF
        assert "SECRET" not in result["source_url"]
        assert created == []

    async def test_a_missing_url_is_refused_before_any_egress(self, monkeypatch):
        fetcher = AsyncMock(side_effect=_staged)
        monkeypatch.setattr(ambient_fetch, "fetch_url", fetcher)
        ctx, _ = _ctx()

        result = await fetch_to_sandbox.execute_fetch_url_to_sandbox(ctx, {"url": {"nope": 1}})

        assert result["ok"] is False and result["error"] == "missing_url"
        fetcher.assert_not_awaited()

    async def test_refused_bytes_are_retried_gzip_wrapped_with_a_decompress_note(self,
                                                                                monkeypatch):
        """Measured live 2026-08-12: the upload endpoint content-sniffs and 400s on SVG under
        every filename tried (.svg/.xml/.dat/.bin/.svg.txt), while the same bytes gzipped go
        straight in. Without the retry the logo scenario this tool exists for still fails."""
        monkeypatch.setattr(ambient_fetch, "fetch_url", AsyncMock(side_effect=_staged))
        ctx, created = _ctx()
        calls: List[Any] = []

        async def _create(container_id, file):
            calls.append((file.name, file.getvalue()))
            if len(calls) == 1:
                raise BadRequestError("You uploaded an invalid file")
            return SimpleNamespace(id="cfile_1", path=f"/mnt/data/{file.name}")

        ctx.processor.openai_client.client.containers.files.create = AsyncMock(
            side_effect=_create)

        result = await fetch_to_sandbox.execute_fetch_url_to_sandbox(
            ctx, {"url": "https://logo.example/mark.svg"})

        assert result["ok"] is True and result["gzipped"] is True
        # The real bytes were offered first; only the refusal produced the wrapper.
        assert calls[0] == ("mark.svg", SVG)
        assert calls[1][0] == "mark.svg.gz"
        assert gzip.decompress(calls[1][1]) == SVG
        assert result["path"] == "/mnt/data/mark.svg.gz"
        # The note has to name the wrapper, or the model hands a gzip blob to cairosvg and
        # reports the ASSET as broken.
        assert "GZIP-COMPRESSED" in result["message"] and "gzip" in result["message"]
        # Both the compressed bytes and what they decompress to are suppressed from publishing.
        from message_processor.file_mount import mounted_digests
        assert len(mounted_digests(ctx)) == 2

    async def test_a_refusal_that_survives_gzipping_is_an_honest_failure(self, monkeypatch):
        monkeypatch.setattr(ambient_fetch, "fetch_url", AsyncMock(side_effect=_staged))
        ctx, _ = _ctx()
        calls: List[Any] = []

        async def _always_refuse(container_id, file):
            calls.append(file.name)
            raise BadRequestError("You uploaded an invalid file")

        ctx.processor.openai_client.client.containers.files.create = AsyncMock(
            side_effect=_always_refuse)

        result = await fetch_to_sandbox.execute_fetch_url_to_sandbox(
            ctx, {"url": "https://logo.example/mark.svg"})

        assert result["ok"] is False and result["error"] == "stage_failed"
        # Exactly ONE retry — a refusal loop is not a recovery.
        assert calls == ["mark.svg", "mark.svg.gz"]

    async def test_a_failure_that_is_not_a_400_is_not_retried(self, monkeypatch):
        monkeypatch.setattr(ambient_fetch, "fetch_url", AsyncMock(side_effect=_staged))
        ctx, _ = _ctx()
        create = AsyncMock(side_effect=RuntimeError("connection reset"))
        ctx.processor.openai_client.client.containers.files.create = create

        result = await fetch_to_sandbox.execute_fetch_url_to_sandbox(
            ctx, {"url": "https://logo.example/mark.svg"})

        assert result["ok"] is False and result["error"] == "stage_failed"
        assert create.await_count == 1

    async def test_an_upload_failure_is_a_result_not_an_exception(self, monkeypatch):
        monkeypatch.setattr(ambient_fetch, "fetch_url", AsyncMock(side_effect=_staged))
        ctx, _ = _ctx()
        ctx.processor.openai_client.client.containers.files.create = AsyncMock(
            side_effect=RuntimeError("container gone"))

        result = await fetch_to_sandbox.execute_fetch_url_to_sandbox(
            ctx, {"url": "https://logo.example/mark.svg"})

        assert result["ok"] is False and result["error"] == "stage_failed"


@pytest.mark.unit
@pytest.mark.asyncio
class TestSuppressionIsExplained:
    """Live 2026-08-12: a build fetched a PNG, verified it, then copied it to a new name.
    Byte-identical, so the publisher held it back as a mounted input — right. But the delivery
    model saw an empty manifest with no reason attached and told the user the file "didn't make
    it back from the build": a false failure report, with two working delivery routes open. The
    guard stays exactly as strict; what it now does is SAY SO."""

    async def test_staging_reports_which_names_it_held_back_as_inputs(self):
        from message_processor import artifacts as artifacts_mod

        fetched = b"\x89PNG\r\n\x1a\n" + b"logo-bytes"
        digest = hashlib.sha256(fetched).hexdigest()
        listed = MagicMock(id="f1", source="assistant", path="/mnt/data/logo_copy.png")
        openai_client = _container_listing([listed], fetched)
        suppressed: List[str] = []

        staged = await artifacts_mod.stage_artifacts(
            openai_client=openai_client, container_ids=["cntr_1"],
            ledger_key="C1:100.0#job:j1", suppress_digests={digest},
            suppressed_inputs_out=suppressed)

        # The guard itself is untouched — the copy is still not published.
        assert staged == []
        assert suppressed == ["logo_copy.png"]

    async def test_the_delivery_model_is_told_why_the_manifest_is_short(self, monkeypatch):
        from message_processor import research_tools as rt

        seen = {}

        async def _capture(**kwargs):
            seen["messages"] = kwargs.get("messages")
            return {"text": "", "tools_used": [], "local_tool_calls": []}

        processor = MagicMock()
        processor.openai_client.create_streaming_response_with_tool_loop = _capture

        await rt._plan_delivery(
            processor, job_id="j1", task="post the logo", report="", staged=[],
            snapshot=[], system_prompt="DEV", model="gpt-5.6-sol",
            channel_id="C1", thread_root="100.0",
            suppressed_inputs=["logo_copy.png"])

        instruction = seen["messages"][-1]["content"]
        assert "logo_copy.png" in instruction
        # The two claims the live failure needed: nothing broke, and there IS a route.
        assert "NOTHING FAILED" in instruction
        assert "import_web_image" in instruction

    async def test_nothing_is_said_when_nothing_was_held_back(self, monkeypatch):
        from message_processor import research_tools as rt

        seen = {}

        async def _capture(**kwargs):
            seen["messages"] = kwargs.get("messages")
            return {"text": "", "tools_used": [], "local_tool_calls": []}

        processor = MagicMock()
        processor.openai_client.create_streaming_response_with_tool_loop = _capture

        await rt._plan_delivery(
            processor, job_id="j1", task="write a report", report="findings", staged=[],
            snapshot=[], system_prompt="DEV", model="gpt-5.6-sol",
            channel_id="C1", thread_root="100.0")

        assert "HELD BACK AS INPUTS" not in seen["messages"][-1]["content"]


@pytest.mark.unit
class TestRegistration:
    def test_it_is_hidden_when_the_sandbox_is_off(self):
        registry = ToolRegistry()
        fetch_to_sandbox.register_fetch_to_sandbox_tool(registry)
        assert "fetch_url_to_sandbox" in {s["name"] for s in registry.schemas({})}
        assert "fetch_url_to_sandbox" in {
            s["name"] for s in registry.schemas({}, surface="channel")}

        off = {"enable_code_interpreter": False}
        assert "fetch_url_to_sandbox" not in {s["name"] for s in registry.schemas(off)}
        assert "fetch_url_to_sandbox" not in {
            s["name"] for s in registry.schemas(off, surface="channel")}
