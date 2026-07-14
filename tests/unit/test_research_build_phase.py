"""F35 — the deep-research BUILD phase: research → file, in a code sandbox.

The job used to be read-only; it stripped code_interpreter outright because it "has no artifact
sink and its own delivery path". This is that sink. The properties worth defending:

* a plain research job is COMPLETELY unchanged (no container, no cost, no build loop);
* the build phase gets its OWN container — sharing the thread's would let a concurrent chat
  turn's baseline snapshot silently mark the half-built deck as "already published", and the
  deck would never be posted;
* files publish AFTER the report, so the thread reads card → report → deck;
* the card's terminal state reflects what SHIPPED, never what the model claimed.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from message_processor import research_tools as rt


@pytest.mark.unit
class TestCleanDeliverables:
    def test_a_well_formed_deliverable_survives(self):
        out = rt._clean_deliverables([
            {"type": "pdf", "description": "AI timeline with charts", "filename": "ai.pdf"}])
        assert out == [{"type": "pdf", "description": "AI timeline with charts",
                        "filename": "ai.pdf"}]

    def test_junk_is_dropped_not_guessed_at(self):
        # A malformed entry must produce NO build phase, never a broken one.
        assert rt._clean_deliverables([{"type": "hologram", "description": "x"}]) == []
        assert rt._clean_deliverables([{"type": "pdf"}]) == []          # no description
        assert rt._clean_deliverables(["a string"]) == []
        assert rt._clean_deliverables(None) == []

    def test_missing_filename_is_synthesised_with_the_right_extension(self):
        out = rt._clean_deliverables([{"type": "powerpoint", "description": "deck"}])
        assert out[0]["filename"].endswith(".pptx")

    def test_capped(self):
        many = [{"type": "pdf", "description": f"d{i}"} for i in range(10)]
        assert len(rt._clean_deliverables(many)) == rt.MAX_DELIVERABLES


def _processor(container="cntr_job1"):
    processor = MagicMock()
    processor.log_info = MagicMock()
    processor.log_error = MagicMock()
    processor.log_warning = MagicMock()
    processor.log_debug = MagicMock()
    processor.db = MagicMock()
    processor.db.find_thread_images_async = AsyncMock(return_value=[])
    processor.db.get_thread_documents_async = AsyncMock(return_value=[])
    processor.container_manager = MagicMock()
    processor.container_manager.get_or_create = AsyncMock(return_value=container)
    processor.container_manager.invalidate = AsyncMock()
    processor.openai_client = MagicMock()
    return processor


def _card(plan=("Research the thing", "Build the deck")):
    card = MagicMock()
    card.set_todos = AsyncMock()
    # F37: "Building the deck…" is a PHASE (the replaceable status line), not a todo — it must
    # not permanently spend one of the card's four lines.
    card.set_phase = AsyncMock()
    # The build phase is a FRESH model loop: it must be handed the live list, or it restarts the
    # plan from scratch instead of revising it. Real _TodoState, so as_prompt_block() is real.
    card.todos = rt._TodoState(list(plan))
    return card


@pytest.mark.unit
class TestBuildPhase:
    async def test_uses_its_own_container_not_the_threads(self, monkeypatch):
        # The whole point. A chat turn calling get_or_create on the SHARED container would
        # baseline the job's half-written deck as already-published, and the publisher would
        # then skip it — the deck would vanish, silently, the more the user chatted.
        processor = _processor()
        seen = {}

        async def fake_stream(_proc, **kw):
            seen["tools"] = kw["tools"]
            seen["registry"] = kw["registry"]
            seen["ctx"] = kw["tool_context"]
            return {"text": "built", "tools_used": []}

        monkeypatch.setattr(rt, "_consume_research_stream", fake_stream)

        build = await rt._run_build_phase(
            processor=processor, client=MagicMock(), channel_id="C1", thread_root="1.0",
            thread_key="C1:1.0", job_id="abc123", task="t", findings="f",
            deliverables=[{"type": "pdf", "description": "d", "filename": "d.pdf"}],
            snapshot=[], thread_config={}, system_prompt=None, model="gpt-5.6-sol",
            card=_card())

        # Job-scoped ledger, keyed off the job id — never the bare thread key.
        assert build["ledger_key"] == "C1:1.0#job:abc123"
        processor.container_manager.get_or_create.assert_awaited_once_with("C1:1.0#job:abc123")

    async def test_build_tools_exclude_the_slack_posting_image_tools(self, monkeypatch):
        # generate_image is DETACHED and posts straight to Slack: inside a build it would land
        # a loose image in the thread instead of in the deck, and could arrive after the job
        # ended. edit_image posts too. A build phase may only make INGREDIENTS.
        processor = _processor()
        captured = {}

        async def fake_stream(_proc, **kw):
            captured["tools"] = kw["tools"]
            return {"text": "", "tools_used": []}

        monkeypatch.setattr(rt, "_consume_research_stream", fake_stream)

        await rt._run_build_phase(
            processor=processor, client=MagicMock(), channel_id="C1", thread_root="1.0",
            thread_key="C1:1.0", job_id="j", task="t", findings="f",
            deliverables=[{"type": "pdf", "description": "d", "filename": "d.pdf"}],
            snapshot=[], thread_config={}, system_prompt=None, model="gpt-5.6-sol",
            card=_card())

        names = {t.get("name") for t in captured["tools"] if t.get("type") == "function"}
        assert "generate_image" not in names
        assert "edit_image" not in names
        assert "create_image_asset" in names      # the sandbox one IS offered
        assert "update_todos" in names
        # and the sandbox itself, bound to the job's container
        ci = [t for t in captured["tools"] if t.get("type") == "code_interpreter"]
        assert ci and ci[0]["container"] == "cntr_job1"

    async def test_no_addressable_container_fails_honestly(self, monkeypatch):
        # An `auto` container has no id: nothing can be mounted into it and its listing cannot
        # be read back. A build phase without those isn't degraded, it's a lie.
        processor = _processor(container={"type": "auto"})
        monkeypatch.setattr(rt, "_consume_research_stream", AsyncMock())

        build = await rt._run_build_phase(
            processor=processor, client=MagicMock(), channel_id="C1", thread_root="1.0",
            thread_key="C1:1.0", job_id="j", task="t", findings="f",
            deliverables=[{"type": "pdf", "description": "d", "filename": "d.pdf"}],
            snapshot=[], thread_config={}, system_prompt=None, model="gpt-5.6-sol",
            card=_card())

        assert build is None

    async def test_a_timeout_still_publishes_what_was_written(self, monkeypatch):
        # A deck finished at second 599 is still a deck.
        import asyncio
        processor = _processor()

        async def timing_out(_proc, **kw):
            raise asyncio.TimeoutError()

        monkeypatch.setattr(rt, "_consume_research_stream", timing_out)

        build = await rt._run_build_phase(
            processor=processor, client=MagicMock(), channel_id="C1", thread_root="1.0",
            thread_key="C1:1.0", job_id="j", task="t", findings="f",
            deliverables=[{"type": "pdf", "description": "d", "filename": "d.pdf"}],
            snapshot=[], thread_config={}, system_prompt=None, model="gpt-5.6-sol",
            card=_card())

        assert build is not None
        assert build["container_ids"] == ["cntr_job1"]

    async def test_the_users_image_settings_reach_the_build(self, monkeypatch):
        # The image MODEL is a hard constraint from the user's prefs; a build phase that
        # silently fell back to defaults would ignore what they chose.
        processor = _processor()
        captured = {}

        async def fake_stream(_proc, **kw):
            captured["ctx"] = kw["tool_context"]
            return {"text": "", "tools_used": []}

        monkeypatch.setattr(rt, "_consume_research_stream", fake_stream)

        await rt._run_build_phase(
            processor=processor, client=MagicMock(), channel_id="C1", thread_root="1.0",
            thread_key="C1:1.0", job_id="j", task="t", findings="f",
            deliverables=[{"type": "pdf", "description": "d", "filename": "d.pdf"}],
            snapshot=[], thread_config={"image_model": "gpt-image-1"},
            system_prompt=None, model="gpt-5.6-sol", card=_card())

        assert captured["ctx"].thread_config["image_model"] == "gpt-image-1"
        assert captured["ctx"].container_id == "cntr_job1"


@pytest.mark.unit
class TestResearchInstruction:
    def test_a_build_job_is_told_to_write_the_numbers_down(self):
        # The build phase sees ONLY the report. If the research writes prose about figures
        # instead of the figures, the charts have nothing real to plot — and an image model
        # asked to draw a chart invents the data.
        addendum = rt._RESEARCH_FOR_BUILD_ADDENDUM.format(deliverables="- x.pdf (pdf): d")
        assert "table" in addendum.lower()
        assert "invent" in addendum.lower()

    def test_the_build_instruction_forbids_claiming_delivery(self):
        text = rt._BUILD_JOB_INSTRUCTION
        assert "sandbox:" in text          # never write a dead link
        assert "attached" in text.lower()  # never claim a file was attached
        # Describing the deck instead of building it is THE failure mode.
        assert "describ" in text.lower()
