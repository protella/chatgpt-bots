"""Fix Workstream C — regression tests for the artifact / background-job / config findings.

Each test names the finding it pins:

* F6  — a DECLARED deliverable is never dropped by the chat-turn heuristics, and the completion
        card goes amber when a declared file did not reach the thread.
* F14 — a declared deliverable is downloaded even when it was written after a budget's worth of
        intermediate files (selection ORDER, not budget size, was the bug).
* F16 — staging is time-bounded from the INSIDE: a slow container yields what it already staged
        instead of losing everything to an outer cancel.
* F25 — `.zip` is a publishable artifact, so the advertised "archive" deliverable can exist.
* F30 — DEFAULT_MAX_TOKENS defaults to 32768.
* F8  — `enable_image_tools` exists on BotConfig, defaulting True.
"""
import hashlib
import io
import os
import zipfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import BotConfig
from message_processor import artifacts as artifacts_mod
from message_processor.artifacts import (
    ArtifactRef,
    _gather_candidates,
    _magic_ok,
    _prioritize_declared,
    _select_candidates,
    publish_artifacts,
    sanitize_filename,
    stage_artifacts,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
PDF = b"%PDF-1.7\n" + b"x" * 32
ZIP = b"PK\x03\x04" + b"\x00" * 40
EMPTY_ZIP = b"PK\x05\x06" + b"\x00" * 18


# --------------------------------------------------------------------------------- helpers
# (kept local so this file does not depend on another test module's internals)

def _cfile(id_, path, source="assistant", size=None):
    f = MagicMock()
    f.id = id_
    f.path = path
    f.source = source
    f.bytes = size
    return f


class _Pager:
    def __init__(self, files):
        self._files = files

    def __aiter__(self):
        async def gen():
            for f in self._files:
                yield f
        return gen()


class _StreamedBody:
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


def _openai_payloads(files, payload_by_id):
    def _retrieve(file_id, container_id=None):
        return _StreamedBody(payload_by_id[file_id])
    oc = MagicMock()
    oc.client.containers.files.list = MagicMock(return_value=_Pager(list(files)))
    oc.client.containers.files.content.with_streaming_response.retrieve = MagicMock(
        side_effect=_retrieve)
    return oc


_UPLOAD = {"file_id": "F123", "url_private": "https://files.slack.com/x", "permalink": "https://p"}


def _client(upload=_UPLOAD):
    c = MagicMock()
    c.send_file = AsyncMock(return_value=upload)
    return c


def _cand(filename, data, ext=None):
    ext = ext or filename.rpartition(".")[2].lower()
    return {"ref": ArtifactRef(container_id="c1", file_id=filename, filename=filename),
            "filename": filename, "ext": ext, "data": data,
            "digest": hashlib.sha256(data).hexdigest()}


def _pptx(tag=b""):
    """A structurally valid .pptx whose bytes are unique per `tag` (distinct digests)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/'
            'package/2006/content-types"><Override PartName="/ppt/presentation.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.'
            'presentationml.presentation.main+xml"/></Types>')
        zf.writestr("ppt/presentation.xml", "<presentation/>")
        if tag:
            zf.writestr("ppt/_unique.bin", tag)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _reset_published_memory():
    artifacts_mod._published_file_ids.clear()
    yield
    artifacts_mod._published_file_ids.clear()


# ------------------------------------------------------------------------------------- F6

class TestDeclaredDeliverablesSurviveHeuristics:
    """A declared report.pdf + social-card.png must BOTH ship. Before the fix the PNG was
    suppressed as 'an ingredient of the document being published' the moment the PDF made it a
    document turn — even though the caller explicitly asked for the image."""

    async def test_declared_image_survives_beside_a_declared_document(self):
        files = [_cfile("f1", "/mnt/data/report.pdf"),
                 _cfile("f2", "/mnt/data/social-card.png")]
        oc = _openai_payloads(files, {"f1": PDF, "f2": PNG})
        out = await publish_artifacts(
            openai_client=oc, client=_client(), channel_id="C1", thread_id="1.0",
            thread_key="C1:1.0", container_ids=["c1"], db=None,
            expect_filenames=["report.pdf", "social-card.png"])
        assert sorted(p["filename"] for p in out) == ["report.pdf", "social-card.png"]

    def test_selection_keeps_a_declared_image_next_to_a_declared_pdf(self):
        cands = [_cand("report.pdf", PDF), _cand("social-card.png", PNG)]
        accepted = _select_candidates(
            cands, suppress_digests=set(),
            expect_filenames=["report.pdf", "social-card.png"])
        assert sorted(c["filename"] for c in accepted) == ["report.pdf", "social-card.png"]

    def test_undeclared_image_beside_a_declared_pdf_is_still_suppressed(self):
        # The exemption is scoped to DECLARED files. A loose chart that was NOT asked for is
        # still an ingredient of the PDF and must not ship.
        cands = [_cand("report.pdf", PDF), _cand("scratch.png", PNG)]
        accepted = _select_candidates(
            cands, suppress_digests=set(), expect_filenames=["report.pdf"])
        assert [c["filename"] for c in accepted] == ["report.pdf"]

    def test_declared_drafts_of_one_type_are_not_pruned_as_superseded(self):
        # Two declared PDFs are two deliverables, not a draft and its replacement.
        cands = [_cand("summary.pdf", PDF), _cand("appendix.pdf", PDF + b"2")]
        accepted = _select_candidates(
            cands, suppress_digests=set(),
            expect_filenames=["summary.pdf", "appendix.pdf"])
        assert sorted(c["filename"] for c in accepted) == ["appendix.pdf", "summary.pdf"]


class TestSameExtensionDraftsDoNotAllCountAsDeclared:
    """T1-6: a shared extension is a WEAK signal. `deck.pptx` declared must not make every
    `.pptx` draft a declared deliverable — that exempted them all from the superseded-draft
    filter and published the whole pile. Extension fallback selects at most ONE per entry."""

    def test_twenty_pptx_drafts_yield_exactly_one_for_a_declared_deck(self):
        cands = [_cand(f"draft_{i:02d}.pptx", b"PK\x03\x04" + bytes([i])) for i in range(20)]
        accepted = _select_candidates(
            cands, suppress_digests=set(), expect_filenames=["deck.pptx"])
        assert len(accepted) == 1
        # Newest same-extension file (last in creation order) is the finished one.
        assert accepted[0]["filename"] == "draft_19.pptx"

    def test_exact_named_deck_wins_over_same_ext_drafts(self):
        cands = [_cand(f"draft_{i:02d}.pptx", b"PK\x03\x04" + bytes([i])) for i in range(5)]
        cands.append(_cand("deck.pptx", b"PK\x03\x04-the-real-one"))
        accepted = _select_candidates(
            cands, suppress_digests=set(), expect_filenames=["deck.pptx"])
        assert [c["filename"] for c in accepted] == ["deck.pptx"]

    def test_two_declared_same_ext_entries_select_two_files(self):
        cands = [_cand("x.pdf", PDF + b"1"), _cand("y.pdf", PDF + b"2")]
        accepted = _select_candidates(
            cands, suppress_digests=set(), expect_filenames=["a.pdf", "b.pdf"])
        assert sorted(c["filename"] for c in accepted) == ["x.pdf", "y.pdf"]

    async def test_twenty_pptx_drafts_publish_exactly_one_end_to_end(self, monkeypatch):
        # Budget = max(cap*4, cap); cap=5 makes it 20 so every draft downloads and the pick is
        # deterministic — the whole pipeline (gather → prioritize → select → upload) ships one.
        monkeypatch.setattr(artifacts_mod.config, "artifact_max_files", 5)
        files = [_cfile(f"f{i}", f"/mnt/data/draft_{i:02d}.pptx") for i in range(20)]
        payloads = {f"f{i}": _pptx(bytes([i])) for i in range(20)}
        oc = _openai_payloads(files, payloads)
        out = await publish_artifacts(
            openai_client=oc, client=_client(), channel_id="C1", thread_id="1.0",
            thread_key="C1:1.0", container_ids=["c1"], db=None,
            expect_filenames=["deck.pptx"])
        assert len(out) == 1
        assert out[0]["filename"] == "draft_19.pptx"

    async def test_true_newest_survives_budget_truncation(self, monkeypatch):
        # 21 same-ext drafts, budget 20: the newest (draft_20, last in the LISTING) would be cut
        # if "newest" were resolved among downloaded candidates. Prioritization promotes it into
        # the budget from listing metadata, so it is the one that ships.
        monkeypatch.setattr(artifacts_mod.config, "artifact_max_files", 5)  # budget = 20
        files = [_cfile(f"f{i}", f"/mnt/data/draft_{i:02d}.pptx") for i in range(21)]
        payloads = {f"f{i}": _pptx(bytes([i]) + b"pad") for i in range(21)}
        oc = _openai_payloads(files, payloads)
        out = await publish_artifacts(
            openai_client=oc, client=_client(), channel_id="C1", thread_id="1.0",
            thread_key="C1:1.0", container_ids=["c1"], db=None,
            expect_filenames=["deck.pptx"])
        assert len(out) == 1
        assert out[0]["filename"] == "draft_20.pptx"


class TestUndeliveredDeliverables:
    """The completion-card honesty half of F6: name what the user asked for and did NOT get."""

    def test_no_false_green_when_one_pptx_covers_two_declared_pptx(self):
        # T1-6: a single shipped .pptx must not satisfy TWO declared .pptx entries.
        from message_processor.research_tools import _undelivered_deliverables
        deliverables = [{"type": "powerpoint", "filename": "deck.pptx", "description": "x"},
                        {"type": "powerpoint", "filename": "appendix.pptx", "description": "y"}]
        missing = _undelivered_deliverables(deliverables, [{"filename": "deck.pptx"}])
        assert [d["filename"] for d in missing] == ["appendix.pptx"]

    def test_missing_declared_file_is_reported(self):
        from message_processor.research_tools import _undelivered_deliverables
        deliverables = [{"type": "pdf", "filename": "report.pdf", "description": "x"},
                        {"type": "image", "filename": "social-card.png", "description": "y"}]
        missing = _undelivered_deliverables(deliverables, [{"filename": "report.pdf"}])
        assert [d["filename"] for d in missing] == ["social-card.png"]

    def test_all_delivered_reports_nothing_missing(self):
        from message_processor.research_tools import _undelivered_deliverables
        deliverables = [{"type": "pdf", "filename": "report.pdf", "description": "x"},
                        {"type": "image", "filename": "social-card.png", "description": "y"}]
        published = [{"filename": "report.pdf"}, {"filename": "social-card.png"}]
        assert _undelivered_deliverables(deliverables, published) == []

    def test_extension_fallback_covers_a_renamed_deliverable(self):
        from message_processor.research_tools import _undelivered_deliverables
        deliverables = [{"type": "pdf", "filename": "report.pdf", "description": "x"}]
        # The model saved it under a different name but the right extension.
        assert _undelivered_deliverables(deliverables, [{"filename": "final-report.pdf"}]) == []

    def test_extension_is_consumed_once_so_two_declared_pdfs_need_two_files(self):
        from message_processor.research_tools import _undelivered_deliverables
        deliverables = [{"type": "pdf", "filename": "a.pdf", "description": "x"},
                        {"type": "pdf", "filename": "b.pdf", "description": "y"}]
        # Only one PDF shipped — one of the two declared PDFs is genuinely missing.
        missing = _undelivered_deliverables(deliverables, [{"filename": "a.pdf"}])
        assert len(missing) == 1


# ------------------------------------------------------------------------------------ F14

class TestDeclaredFilesArePrioritizedForDownload:
    """The download budget is spent in listing order. A deliverable written after a budget's
    worth of intermediates was never even downloaded — so ORDER, not budget size, is the fix."""

    def test_declared_file_sorts_to_the_front_by_name(self):
        refs = [ArtifactRef("c1", f"f{i}", f"chart_{i}.png") for i in range(20)]
        refs.append(ArtifactRef("c1", "deck", "deck.pptx"))
        ordered = _prioritize_declared(refs, ["deck.pptx"])
        assert ordered[0].filename == "deck.pptx"

    def test_declared_file_sorts_to_the_front_by_extension(self):
        # The model does not always honour the declared filename — extension still wins.
        refs = [ArtifactRef("c1", f"f{i}", f"chart_{i}.png") for i in range(20)]
        refs.append(ArtifactRef("c1", "rep", "final-report.pdf"))
        ordered = _prioritize_declared(refs, ["report.pdf"])
        assert ordered[0].filename == "final-report.pdf"

    def test_no_manifest_leaves_order_untouched(self):
        refs = [ArtifactRef("c1", "a", "a.png"), ArtifactRef("c1", "b", "b.png")]
        assert _prioritize_declared(refs, []) == refs

    def test_exact_name_sorts_ahead_of_same_extension_drafts(self):
        # T2-14: the exactly-named deliverable must not be crowded out of the download budget by
        # a swarm of same-extension drafts that landed earlier in the listing.
        refs = [ArtifactRef("c1", f"d{i}", f"draft_{i:02d}.pptx") for i in range(20)]
        refs.insert(19, ArtifactRef("c1", "deck", "deck.pptx"))
        ordered = _prioritize_declared(refs, ["deck.pptx"])
        assert ordered[0].filename == "deck.pptx"

    async def test_declared_file_past_the_budget_is_still_downloaded(self, monkeypatch):
        # Force a small budget: max(cap*4, cap) with cap=1 is 4. Put the deck 6th in listing
        # order — without prioritization it sits past the budget and is never fetched.
        monkeypatch.setattr(artifacts_mod.config, "artifact_max_files", 1)
        files = [_cfile(f"f{i}", f"/mnt/data/chart_{i}.png") for i in range(5)]
        files.append(_cfile("deck", "/mnt/data/deck.pdf"))
        payloads = {f"f{i}": PNG + bytes([i]) for i in range(5)}
        payloads["deck"] = PDF
        oc = _openai_payloads(files, payloads)
        candidates, _skipped = await _gather_candidates(
            openai_client=oc, container_ids=["c1"], container_manager=None,
            ledger_key="C1:1.0", expect_filenames=["deck.pdf"])
        assert "deck.pdf" in [c["filename"] for c in candidates]


# ------------------------------------------------------------------------------------ F16

class TestStagingIsTimeBoundedFromInside:
    """A single outer wait_for used to cancel the whole coroutine on timeout and discard every
    file already staged. The bound now lives in the download loop: partial results survive."""

    async def test_deadline_returns_what_was_already_gathered(self, monkeypatch):
        files = [_cfile("f1", "/mnt/data/a.png"),
                 _cfile("f2", "/mnt/data/b.png"),
                 _cfile("f3", "/mnt/data/c.png")]
        oc = _openai_payloads(files, {"f1": PNG, "f2": PNG + b"2", "f3": PNG + b"3"})
        # Drive a virtual clock that ONLY each download advances (patching the global
        # time.monotonic directly would also skew asyncio's own timers). Two downloads land
        # under the 100s deadline; the third trips it, and the two already gathered survive.
        state = {"t": 0.0}
        real_download = artifacts_mod._download

        async def _clocked_download(*a, **k):
            state["t"] += 60
            return await real_download(*a, **k)

        monkeypatch.setattr(artifacts_mod.time, "monotonic", lambda: state["t"])
        monkeypatch.setattr(artifacts_mod, "_download", _clocked_download)
        candidates, skipped = await _gather_candidates(
            openai_client=oc, container_ids=["c1"], container_manager=None,
            ledger_key="C1:1.0", deadline=100)
        assert len(candidates) == 2
        assert skipped == 1

    async def test_last_download_is_bounded_by_remaining_budget(self, monkeypatch):
        # T2-16: an in-flight download must not run a full _API_TIMEOUT past the deadline — its
        # timeout shrinks to whatever is left of the budget.
        files = [_cfile("f1", "/mnt/data/a.png"), _cfile("f2", "/mnt/data/b.png")]
        oc = _openai_payloads(files, {"f1": PNG, "f2": PNG + b"2"})
        state = {"t": 0.0}
        seen = []

        async def _rec_download(openai_client, ref, max_bytes,
                                timeout=artifacts_mod._API_TIMEOUT):
            seen.append(timeout)
            state["t"] += 95  # each download eats most of the 100s budget
            return PNG

        monkeypatch.setattr(artifacts_mod.time, "monotonic", lambda: state["t"])
        monkeypatch.setattr(artifacts_mod, "_download", _rec_download)
        await _gather_candidates(
            openai_client=oc, container_ids=["c1"], container_manager=None,
            ledger_key="C1:1.0", deadline=100)
        assert seen[0] == pytest.approx(artifacts_mod._API_TIMEOUT)   # min(30, 100) = 30
        assert seen[1] == pytest.approx(5.0)                          # min(30, 100-95) = 5
        assert seen[1] < artifacts_mod._API_TIMEOUT

    async def test_time_budget_does_not_discard_a_completed_stage(self):
        # A budget that never expires must stage everything — no premature loss.
        files = [_cfile("f1", "/mnt/data/chart.png")]
        oc = _openai_payloads(files, {"f1": PNG})
        staged = await stage_artifacts(
            openai_client=oc, container_ids=["c1"], container_manager=None,
            ledger_key="C1:1.0", time_budget=1000)
        assert [s.filename for s in staged] == ["chart.png"]


# ------------------------------------------------------------------------------------ F25

class TestZipIsPublishable:
    """An 'archive' deliverable is advertised (research_tools) and now actually deliverable."""

    def test_magic_accepts_a_real_zip(self):
        assert _magic_ok(ZIP, "zip") is True
        assert _magic_ok(EMPTY_ZIP, "zip") is True

    def test_magic_rejects_non_zip_bytes_named_zip(self):
        assert _magic_ok(b"not a zip at all", "zip") is False

    def test_sanitize_allows_zip(self):
        assert sanitize_filename("bundle.zip") == "bundle.zip"

    def test_zip_is_in_the_default_allowlist(self, mock_env):
        assert "zip" in BotConfig().artifact_allowed_extensions

    async def test_a_zip_artifact_publishes_end_to_end(self):
        files = [_cfile("f1", "/mnt/data/bundle.zip")]
        oc = _openai_payloads(files, {"f1": ZIP})
        out = await publish_artifacts(
            openai_client=oc, client=_client(), channel_id="C1", thread_id="1.0",
            thread_key="C1:1.0", container_ids=["c1"], db=None)
        assert [p["filename"] for p in out] == ["bundle.zip"]


# ------------------------------------------------------------------------------- F30 / F8

class TestConfigDefaults:
    def test_default_max_tokens_defaults_to_32768(self, monkeypatch):
        monkeypatch.delenv("DEFAULT_MAX_TOKENS", raising=False)
        assert BotConfig().default_max_tokens == 32768

    def test_enable_image_tools_defaults_true(self, monkeypatch):
        monkeypatch.delenv("ENABLE_IMAGE_TOOLS", raising=False)
        assert BotConfig().enable_image_tools is True

    @patch.dict(os.environ, {"ENABLE_IMAGE_TOOLS": "false"})
    def test_enable_image_tools_honours_env_false(self):
        assert BotConfig().enable_image_tools is False
