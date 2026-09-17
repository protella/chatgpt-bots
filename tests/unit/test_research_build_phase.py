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
import hashlib
import io
import itertools
import zipfile
from unittest.mock import AsyncMock, MagicMock

import openai
import pytest

from message_processor import artifacts as artifacts_mod
from message_processor import containers as containers_mod
from message_processor import research_tools as rt
from message_processor.artifacts import StagedArtifact
from slack_client.messaging import CardWriteResult


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
    # W3: a build resolves through create_explicit — `auto` gives it neither a mount target nor
    # a readable listing. get_or_create is left as a bare MagicMock deliberately: awaiting it
    # would raise, so a regression back to it fails loudly here.
    processor.container_manager.create_explicit = AsyncMock(return_value=container)
    processor.container_manager.invalidate = AsyncMock()
    processor.openai_client = MagicMock()
    return processor


def _card(plan=("Research the thing", "Build the deck")):
    card = MagicMock()
    card.set_todos = AsyncMock()
    # F37: "Building the deck…" is a PHASE (the replaceable status line), not a todo — it must
    # not permanently spend one of the card's four lines.
    card.set_phase = AsyncMock()
    card.set_alert = AsyncMock()
    # The build phase is a FRESH model loop: it must be handed the live list, or it restarts the
    # plan from scratch instead of revising it. Real _TodoState, so as_prompt_block() is real.
    card.todos = rt._TodoState(list(plan))
    return card


@pytest.mark.unit
class TestBuildPhase:
    async def test_uses_its_own_container_not_the_threads(self, monkeypatch):
        # The whole point. A build resolving the SHARED thread container would
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
        processor.container_manager.create_explicit.assert_awaited_once_with(
            "C1:1.0#job:abc123")

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

    async def test_the_build_phase_really_has_no_web_access(self, monkeypatch):
        # Its instruction tells the model it cannot search and must not write a claim its source
        # material doesn't establish. If a web tool ever gets wired in here, that becomes a lie
        # the model has no way to detect — and honest sourcing turns into a guess.
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

        assert not any("search" in str(t.get("type", "")) or "search" in str(t.get("name", ""))
                       for t in captured["tools"]), captured["tools"]

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

    async def test_the_build_carries_no_ration_and_no_clock(self, monkeypatch):
        # A build is done when the files are built and verified — running out of rounds mid-build
        # is the difference between a deck and an apology, and an elapsed-time wall cut a working
        # build just as readily. Neither exists: nothing here hands the stream a deadline.
        processor = _processor()
        seen = {}

        async def fake_stream(_proc, **kw):
            seen.update(kw)
            return {"text": "built", "tools_used": []}

        monkeypatch.setattr(rt, "_consume_research_stream", fake_stream)

        await rt._run_build_phase(
            processor=processor, client=MagicMock(), channel_id="C1", thread_root="1.0",
            thread_key="C1:1.0", job_id="j", task="t", findings="f",
            deliverables=[{"type": "pdf", "description": "d", "filename": "d.pdf"}],
            snapshot=[], thread_config={}, system_prompt=None, model="gpt-5.6-sol",
            card=_card())

        assert "max_rounds" not in seen, "the build phase must not carry a round ration"
        assert not {"wrap_up_at", "wind_down_at"} & set(seen), "the build must carry no deadline"

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

    async def test_the_workers_own_account_of_the_build_comes_back(self, monkeypatch):
        # In `build` mode there is no research report, so these notes are the ONLY narrative the
        # delivering model gets. Dropped, it can name the files it is posting and nothing more —
        # seen live as a job that ran a test and then declined to state the result.
        processor = _processor()

        async def fake_stream(_proc, **kw):
            return {"text": "Mounted the marker; the token matched round 1.", "tools_used": []}

        monkeypatch.setattr(rt, "_consume_research_stream", fake_stream)

        build = await rt._run_build_phase(
            processor=processor, client=MagicMock(), channel_id="C1", thread_root="1.0",
            thread_key="C1:1.0", job_id="j", task="t", findings="f",
            deliverables=[{"type": "txt", "description": "d", "filename": "d.txt"}],
            snapshot=[], thread_config={}, system_prompt=None, model="gpt-5.6-sol",
            card=_card())

        assert build["notes"] == "Mounted the marker; the token matched round 1."

    async def test_a_timed_out_build_still_reports_empty_notes_not_a_missing_key(self, monkeypatch):
        # The timeout path never assigns the stream's return value. A missing key here would
        # KeyError the delivery hand-off on exactly the job that already went wrong.
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

        assert build["notes"] == ""

    async def test_the_build_can_edit_an_image_but_not_post_one(self, monkeypatch):
        # A build with no way to MODIFY an image does the only thing left to it: live, a job
        # asked to edit a thread image drew tinted polygons over the source in the sandbox and
        # shipped what was, measurably, a copy. edit_image_asset is the route — same sources as
        # edit_image, result into the container instead of into Slack.
        processor = _processor()
        processor.db.find_thread_images_async = AsyncMock(return_value=[
            {"id": 7, "url": "https://files.slack.com/pizza.png", "image_type": "generated",
             "prompt": "an enhanced prompt", "analysis": "A pizza on a wooden board",
             "created_at": "2026-09-10T09:20:00"}])
        captured = {}

        async def fake_stream(_proc, **kw):
            captured["tools"] = kw["tools"]
            captured["ctx"] = kw["tool_context"]
            return {"text": "", "tools_used": []}

        monkeypatch.setattr(rt, "_consume_research_stream", fake_stream)

        await rt._run_build_phase(
            processor=processor, client=MagicMock(), channel_id="C1", thread_root="1.0",
            thread_key="C1:1.0", job_id="j", task="t", findings="f",
            deliverables=[{"type": "pdf", "description": "d", "filename": "d.pdf"}],
            snapshot=[], thread_config={}, system_prompt=None, model="gpt-5.6-sol",
            card=_card())

        names = {t.get("name") for t in captured["tools"] if t.get("type") == "function"}
        assert "edit_image_asset" in names
        assert "create_image_asset" in names
        assert "generate_image" not in names     # detached, posts straight to Slack
        assert "edit_image" not in names         # posts straight to Slack

        # The EXECUTOR resolves ids against the context, not against thread_config. Advertise
        # an id in the schema and leave the context empty and every call comes back
        # unknown_image_id — a tool that is offered and cannot work. (The schema LISTS the ids in
        # its description rather than enumerating them since ruling 15, so the listing is what is
        # checked here; the context is what has to agree with it.)
        advertised = next(t["description"] for t in captured["tools"]
                          if t.get("name") == "edit_image_asset")
        assert "img_7" in advertised
        assert [e["image_id"] for e in captured["ctx"].image_catalog] == ["img_7"]


def _transient_stream_error(message="An error occurred while processing your request."):
    """The live failure (job 5e58a49b615f): a Responses SSE stream that died 281s into a build
    and surfaced as a BARE openai.APIError — no status code, no response object at all."""
    return openai.APIError(message, request=None, body=None)


def _status_error(cls, status):
    response = MagicMock()
    response.status_code = status
    response.headers = {}
    return cls("the provider said no", response=response, body=None)


async def _build(processor, card, **kw):
    return await rt._run_build_phase(
        processor=processor, client=MagicMock(), channel_id="C1", thread_root="1.0",
        thread_key="C1:1.0", job_id="j", task="t", findings="f",
        deliverables=[{"type": "pdf", "description": "d", "filename": "d.pdf"}],
        snapshot=[], thread_config={}, system_prompt=None, model="gpt-5.6-sol",
        card=card, **kw)


@pytest.mark.unit
class TestBuildPhaseRetry:
    """A transient provider error kills the STREAM, not the container — and the container is
    where the build actually lives. Starting over would throw away minutes of real work; the
    retry re-enters the same container and tells the model to look before it acts."""

    async def test_a_cut_off_stream_resumes_against_the_same_container(self, monkeypatch):
        processor = _processor()
        card = _card()
        monkeypatch.setattr(rt.config, "deep_research_build_retries", 2)
        calls = []

        async def flaky(_proc, **kw):
            calls.append(kw["messages"])
            if len(calls) == 1:
                raise _transient_stream_error()
            return {"text": "deck built", "tools_used": []}

        monkeypatch.setattr(rt, "_consume_research_stream", flaky)

        build = await _build(processor, card)

        assert build["notes"] == "deck built"
        assert len(calls) == 2
        # The resume note is APPENDED — the original brief is not replaced or rebuilt.
        assert calls[1][:len(calls[0])] == calls[0]
        resume = [i for i in calls[1] if rt._BUILD_RESUME_NOTE in str(i.get("content"))]
        assert len(resume) == 1 and resume[0]["role"] == "user"
        # The user hears about it on the ONE block that renders in every card state.
        assert [c.args[0] for c in card.set_alert.await_args_list] == [
            "Provider hiccup — retrying (2/3)…"]

    async def test_one_resume_note_total_however_many_retries(self, monkeypatch):
        # Re-appending it is not a stronger instruction, just a duplicate one.
        processor = _processor()
        monkeypatch.setattr(rt.config, "deep_research_build_retries", 2)
        calls = []

        async def flaky(_proc, **kw):
            calls.append(kw["messages"])
            if len(calls) < 3:
                raise _transient_stream_error()
            return {"text": "deck built", "tools_used": []}

        monkeypatch.setattr(rt, "_consume_research_stream", flaky)

        await _build(processor, _card())

        assert len(calls) == 3
        assert sum(1 for i in calls[2] if rt._BUILD_RESUME_NOTE in str(i.get("content"))) == 1

    async def test_the_resume_note_replays_steering_the_dead_attempt_had_drained(self,
                                                                                 monkeypatch):
        # The drain is DESTRUCTIVE and those rounds died with the stream. Replaying an
        # instruction the model already honoured is harmless; losing one silently reinstates
        # whatever the user asked to have dropped.
        processor = _processor()
        monkeypatch.setattr(rt.config, "deep_research_build_retries", 2)
        calls = []

        async def flaky(_proc, **kw):
            calls.append(kw["messages"])
            if len(calls) == 1:
                raise _transient_stream_error()
            return {"text": "ok", "tools_used": []}

        monkeypatch.setattr(rt, "_consume_research_stream", flaky)

        await _build(processor, _card(), applied_notes=["drop the competitor section"])

        resume = [i for i in calls[1] if rt._BUILD_RESUME_NOTE in str(i.get("content"))][0]
        assert "drop the competitor section" in resume["content"]
        assert "must still honor" in resume["content"]

    async def test_a_4xx_is_terminal(self, monkeypatch):
        # The SDK runs with max_retries=0 here by design, and this path adds no backoff: a 4xx
        # means the request itself is wrong, and re-sending it byte-for-byte cannot fix it.
        processor = _processor()
        card = _card()
        monkeypatch.setattr(rt.config, "deep_research_build_retries", 2)
        calls = []

        async def bad_request(_proc, **kw):
            calls.append(kw["messages"])
            raise _status_error(openai.BadRequestError, 400)

        monkeypatch.setattr(rt, "_consume_research_stream", bad_request)

        build = await _build(processor, card)

        assert len(calls) == 1
        card.set_alert.assert_not_awaited()
        assert build["notes"] == ""
        assert processor.log_error.called

    async def test_a_dead_container_is_not_retried(self, monkeypatch):
        # R4: the retry would name the corpse in its own tools array. That case already has its
        # own honest messaging.
        processor = _processor()
        card = _card()
        monkeypatch.setattr(rt.config, "deep_research_build_retries", 2)
        calls = []

        async def gone(_proc, **kw):
            calls.append(kw["messages"])
            raise _transient_stream_error("Container with id 'cntr_job1' not found.")

        monkeypatch.setattr(rt, "_consume_research_stream", gone)

        await _build(processor, card)

        assert len(calls) == 1
        card.set_alert.assert_not_awaited()

    async def test_exhausted_retries_still_hand_the_publisher_the_container(self, monkeypatch):
        # Whatever the dead attempts DID write is still in there, and the container listing is
        # what the publisher ships from — so the phase must still return, not vanish.
        processor = _processor()
        card = _card()
        monkeypatch.setattr(rt.config, "deep_research_build_retries", 2)
        calls = []

        async def always_flaky(_proc, **kw):
            calls.append(kw["messages"])
            raise _transient_stream_error()

        monkeypatch.setattr(rt, "_consume_research_stream", always_flaky)

        build = await _build(processor, card)

        assert len(calls) == 3
        assert build is not None and build["notes"] == ""
        assert build["container_ids"] == ["cntr_job1"]
        # One announcement per retry, and none after the last attempt.
        assert len(card.set_alert.await_args_list) == 2
        # The existing final-failure line survives, so log greps keep working.
        assert any("failed" in str(c.args[0]) for c in processor.log_error.call_args_list)

    async def test_the_resume_note_is_rewritten_before_every_retry(self, monkeypatch):
        # `applied_notes` keeps growing while the job runs. Freezing the note at the first retry
        # would lose whatever the SECOND attempt drained and then died holding.
        processor = _processor()
        monkeypatch.setattr(rt.config, "deep_research_build_retries", 2)
        applied = []
        calls = []

        async def flaky(_proc, **kw):
            calls.append(kw["messages"])
            if len(calls) < 3:
                applied.append(f"note {len(calls)}")
                raise _transient_stream_error()
            return {"text": "ok", "tools_used": []}

        monkeypatch.setattr(rt, "_consume_research_stream", flaky)

        await _build(processor, _card(), applied_notes=applied)

        assert len(calls) == 3
        resume = [i for i in calls[2] if rt._BUILD_RESUME_NOTE in str(i.get("content"))]
        assert len(resume) == 1
        assert "note 1" in resume[0]["content"] and "note 2" in resume[0]["content"]

    async def test_a_request_timeout_came_from_below_and_is_retried(self, monkeypatch):
        # `_safe_api_call` re-raises openai.APITimeoutError as a BUILTIN TimeoutError. Nothing
        # up here bounds the build by elapsed time any more, so every TimeoutError reaching this
        # loop is the transport watchdog on ONE request — as transient as a dropped stream, and
        # the container it was building in is still there.
        processor = _processor()
        card = _card()
        monkeypatch.setattr(rt.config, "deep_research_build_retries", 2)
        calls = []

        async def flaky(_proc, **kw):
            calls.append(kw["messages"])
            if len(calls) == 1:
                raise TimeoutError("OpenAI API call timed out after 300 seconds")
            return {"text": "deck built", "tools_used": []}

        monkeypatch.setattr(rt, "_consume_research_stream", flaky)

        build = await _build(processor, card)

        assert len(calls) == 2 and build["notes"] == "deck built"
        assert card.set_alert.await_count == 1

    async def test_a_timeout_on_the_last_attempt_ships_what_exists(self, monkeypatch):
        # Retries are the only bound left. When they run out, ship whatever the container holds —
        # and log it, because the "timed out" line is what log greps look for.
        processor = _processor()
        card = _card()
        monkeypatch.setattr(rt.config, "deep_research_build_retries", 1)
        calls = []

        async def timing_out(_proc, **kw):
            calls.append(kw["messages"])
            raise TimeoutError()

        monkeypatch.setattr(rt, "_consume_research_stream", timing_out)

        build = await _build(processor, card)

        assert len(calls) == 2 and build["notes"] == ""
        assert any("timed out" in str(c.args[0])
                   for c in processor.log_warning.call_args_list)

    async def test_a_demoted_container_makes_the_failure_terminal(self, monkeypatch):
        # R4 leak: if the recovery layer underneath swapped this call onto a throwaway `auto`
        # container, the work is not where the resume note swears it is.
        processor = _processor()
        card = _card()
        monkeypatch.setattr(rt.config, "deep_research_build_retries", 2)
        calls = []

        async def demoted(_proc, **kw):
            calls.append(kw["messages"])
            kw["container_gone_sink"].append({"container_id": "cntr_job1"})
            raise _transient_stream_error()

        monkeypatch.setattr(rt, "_consume_research_stream", demoted)

        await _build(processor, card)

        assert len(calls) == 1
        card.set_alert.assert_not_awaited()


class _FakeCardClient:
    """Records what the card actually rendered. No sleeping, no network."""

    def __init__(self):
        self.blocks = []

    async def post_status_card(self, *_a, **_kw):
        return "111.0"

    async def update_status_card(self, _channel, _ts, _text, blocks):
        self.blocks.append(blocks)
        # Same return type as the transport (F38).
        return CardWriteResult(ok=True)


def _context_text(blocks):
    """The context block as one string: F38 renders the live line as its own `plain_text`
    element beside the static mrkdwn half, and Slack shows them as one line."""
    return " ".join(e["text"] for e in blocks[1]["elements"])


@pytest.mark.unit
class TestCardAlert:
    async def test_the_alert_is_visible_where_a_phase_is_not_and_clears_when_work_resumes(self):
        # set_phase is deliberately NOT rendered while a todo is in_progress (the spinning item
        # IS "what I'm doing now") — which is exactly the state a mid-build retry happens in.
        # The alert rides the context block instead, which renders unconditionally.
        client = _FakeCardClient()
        clock = itertools.count(0, 20).__next__       # every tick past the pacing window
        card = rt._ResearchCard(processor=MagicMock(), client=client, channel_id="C1",
                                thread_root="1.0", task="t", label=None,
                                todos=rt._TodoState(["Build the deck"]), clock=clock)
        await card.start()
        await card.set_todos([{"text": "Build the deck", "status": "in_progress"}])
        await card.set_phase("Building the deck…")
        assert not any("Building the deck" in line for line in card._visible_lines())

        await card.set_alert("Provider hiccup — retrying (2/3)…")
        await card._writer_tick()
        assert "Provider hiccup — retrying (2/3)…" in _context_text(client.blocks[-1])

        # The model driving the card again IS the "moving again" signal — nothing has to
        # remember to take the alert down.
        await card.set_todos([{"text": "Build the deck", "status": "done"}])
        await card._writer_tick()
        assert "retrying" not in _context_text(client.blocks[-1])

    async def test_a_writer_tick_never_takes_the_alert_down_by_itself(self):
        # The card talking to itself is not the job moving again. A tick that rendered the
        # alert, or a write that succeeded, must leave it standing — only the model producing
        # something takes it down, because only that is evidence the hiccup is over.
        client = _FakeCardClient()
        clock = itertools.count(0, 20).__next__
        card = rt._ResearchCard(processor=MagicMock(), client=client, channel_id="C1",
                                thread_root="1.0", task="t", label=None,
                                todos=rt._TodoState(["Build the deck"]), clock=clock)
        await card.start()
        await card.set_alert("Provider hiccup — retrying (2/3)…")
        for _ in range(3):
            await card._writer_tick()
        assert "retrying" in _context_text(client.blocks[-1])

    async def test_an_alert_does_not_wait_the_full_pacing_interval(self):
        # Ordinary state is paced at 15s; an alert is news, and news 15 seconds late is not
        # news. It buys the Slack floor instead — prompt, never unpaced.
        client = _FakeCardClient()
        clock = [0.0]
        card = rt._ResearchCard(processor=MagicMock(), client=client, channel_id="C1",
                                thread_root="1.0", task="t", label=None,
                                todos=rt._TodoState(["Build the deck"]),
                                clock=lambda: clock[0])
        await card.start()
        await card.set_todos([{"text": "Build the deck", "status": "in_progress"}])
        clock[0] = rt._card_throttle_s() + 0.1
        await card._writer_tick()
        assert client.blocks == []          # ordinary state waits for the pacing window

        await card.set_alert("Provider hiccup — retrying (2/3)…")
        await card._writer_tick()
        assert "retrying" in _context_text(client.blocks[-1])

    async def test_a_steering_bump_clears_it_once_it_has_been_seen(self):
        # The LATCH. An alert raised and answered inside one pacing window would otherwise be
        # coalesced out of existence, and the user would be left with an unexplained pause. So
        # the clear waits until a write has actually carried the alert.
        client = _FakeCardClient()
        clock = itertools.count(0, 20).__next__
        card = rt._ResearchCard(processor=MagicMock(), client=client, channel_id="C1",
                                thread_root="1.0", task="t", label=None,
                                todos=rt._TodoState(["Build the deck"]), clock=clock)
        await card.start()
        await card.set_alert("Provider hiccup — retrying (2/3)…")
        await card.note_steering(1)                  # the answer arrives before the render
        await card._writer_tick()
        assert "retrying" in _context_text(client.blocks[-1])     # ...and is still shown

        await card.note_steering(1)                  # now that it has been seen, it comes down
        await card._writer_tick()
        assert "retrying" not in _context_text(client.blocks[-1])
        assert "2 updates passed along" in _context_text(client.blocks[-1])

    async def test_an_activity_counter_bump_clears_it_once_it_has_been_seen(self):
        client = _FakeCardClient()
        clock = itertools.count(0, 20).__next__
        card = rt._ResearchCard(processor=MagicMock(), client=client, channel_id="C1",
                                thread_root="1.0", task="t", label=None,
                                todos=rt._TodoState(["Dig"]), clock=clock)
        await card.start()
        await card.set_alert("Provider hiccup — retrying (2/3)…")
        await card._writer_tick()                    # the alert is on the card
        await card.note_web_search()
        await card._writer_tick()
        assert "retrying" not in _context_text(client.blocks[-1])
        assert "1 web search" in _context_text(client.blocks[-1])


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


# --------------------------------------------------------- native-file admission pricing
#
# The live failure (job 25cc4feb4bfc): a 2,287-token thread holding a 24-page PDF converted
# from a 76MB pptx. The PDF's base64 blob was charged one token per CHARACTER, the build was
# estimated at 2.27M tokens against a 919,800 limit, and the 2,867-character revision master
# the job had been asked to edit was dropped for want of room nothing was occupying. The
# request then succeeded — which is the proof the number was fiction.

_MASTER_NAME = "report.docx"


def _doc_row(filename, *, size_bytes=None, total_pages=None):
    return {"id": 1, "created_at": "2026-09-10T20:50:00", "filename": filename,
            "mime_type": "application/pdf" if filename.endswith(".pdf") else
                         "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "file_id": f"F_{filename}", "url_private": f"https://files.slack.com/{filename}",
            "size_bytes": size_bytes, "total_pages": total_pages}


def _native_pdf_snapshot(filename="deck.pdf", b64_chars=2_200_000):
    """A turn carrying a native PDF part, as the dispatch path assembles one."""
    return [{"role": "user", "content": [
        {"type": "input_text", "text": "revise the report"},
        {"type": "input_file", "filename": filename, "file_data": "B" * b64_chars}]}]


async def _run_revision_build(monkeypatch, *, doc_rows, snapshot, master_text="THE OLD BODY"):
    """Drive the real build phase on a REVISION, capturing what the model would receive."""
    processor = _processor()
    processor.db.get_thread_documents_async = AsyncMock(return_value=doc_rows)
    captured = {}

    async def fake_load(_client, _row, **_kw):
        return {"content": master_text, "cached": False}

    async def fake_stream(_proc, **kw):
        captured["messages"] = kw["messages"]
        return {"text": "built", "tools_used": []}

    monkeypatch.setattr(rt.document_tools, "load_document_text", fake_load)
    monkeypatch.setattr(rt, "_consume_research_stream", fake_stream)

    await rt._run_build_phase(
        processor=processor, client=MagicMock(), channel_id="C1", thread_root="1.0",
        thread_key="C1:1.0", job_id="j", task="revise the totals", findings="f",
        deliverables=[{"type": "document", "description": "the doc",
                       "filename": _MASTER_NAME}],
        snapshot=snapshot, thread_config={}, system_prompt="SYS", model="gpt-5.6-sol",
        card=_card(), revises=[_MASTER_NAME])
    captured["warnings"] = [c.args[0] for c in processor.log_warning.call_args_list]
    return captured


@pytest.mark.unit
class TestNativeFileAdmission:
    async def test_a_paged_native_file_is_priced_by_its_pages_not_its_base64_length(
            self, monkeypatch):
        captured = await _run_revision_build(
            monkeypatch,
            doc_rows=[_doc_row(_MASTER_NAME),
                      _doc_row("deck.pdf", size_bytes=76_000_000, total_pages=24)],
            snapshot=_native_pdf_snapshot())

        joined = "\n".join(str(m.get("content")) for m in captured["messages"])
        assert "THE OLD BODY" in joined          # the master the job was asked to edit
        assert not any("does not fit" in w for w in captured["warnings"])

    def test_an_unmatched_part_still_prices_at_its_base64_length(self):
        # No plumbing carries metadata for a part the documents table has never heard of, so
        # the blob's length stays the bound there — today's behaviour, no new failure mode.
        items = _native_pdf_snapshot(b64_chars=4096)
        assert rt._native_file_bounds(items, {}) == [4096]
        assert rt._native_file_bounds(items, {"other.pdf": (1, 1)}) == [4096]
        # Matched but with no page count (a CSV/XLSX) is the byte count, which is that same
        # behaviour and the right bound for a file the API reads as text.
        assert rt._native_file_bounds(items, {"deck.pdf": (900, None)}) == [900]

    async def test_a_genuinely_oversized_master_still_drops_out(self, monkeypatch):
        # The gate is CORRECTED, not disabled. Priced honestly, a 320-page PDF plus a 140k-char
        # master really does not fit — and the master is the part that gives way.
        captured = await _run_revision_build(
            monkeypatch,
            doc_rows=[_doc_row(_MASTER_NAME),
                      _doc_row("deck.pdf", size_bytes=76_000_000, total_pages=320)],
            snapshot=_native_pdf_snapshot(b64_chars=1000),
            master_text="X" * 140_000)

        joined = "\n".join(str(m.get("content")) for m in captured["messages"])
        assert "X" * 140_000 not in joined
        assert "could not be loaded for this revision (too large to inline)" in joined
        assert any("does not fit" in w for w in captured["warnings"])


# ------------------------------------------------- a sandbox that can no longer run code
#
# Live 2026-09-14 (jobs d8b2126e527f and bc9fce24794d): an OOM killed the sandbox kernel, the
# container went on reporting `running`, and the retry loop burned the remaining attempts against
# the corpse in seconds — then one of the jobs published a stale intermediate. Files survive a
# dead kernel, so they are banked out before the sandbox is abandoned.

_WEDGED_MESSAGE = "There was an issue with your request. Please check your inputs and try again"


def _wedged_stream_error():
    """The measured shape: a status-less APIError carrying the API's generic body."""
    return openai.APIError(_WEDGED_MESSAGE, request=None,
                           body={"type": "invalid_request_error", "code": None,
                                 "message": _WEDGED_MESSAGE, "param": None})


def _staged(filename, *, idx=1, tag="new"):
    """A StagedArtifact as stage_artifacts issues them — every pass numbers from art_1."""
    return StagedArtifact(artifact_id=f"art_{idx}", filename=filename,
                          ext=filename.rsplit(".", 1)[-1], size_bytes=10,
                          candidate={"filename": filename, "generation": tag})


class _Staging:
    """Answers stage_artifacts calls in order and records what each one was asked for."""

    def __init__(self, *responses, order=None):
        self.calls = []
        self._responses = list(responses)
        self._order = order

    async def __call__(self, **kw):
        self.calls.append(kw)
        if self._order is not None:
            self._order.append("stage")
        reply = self._responses.pop(0) if self._responses else {}
        for name in reply.get("suppressed", ()):
            kw["suppressed_inputs_out"].append(name)
        for pair in reply.get("embedded", ()):
            kw["embedded_out"].append(pair)
        # Everything the generation PRODUCED, as the real one fills it: pre-selection, so the
        # accepted files plus everything a selection rule held back.
        produced = list(reply.get("produced", ())) or [
            *reply.get("files", ()), *reply.get("suppressed", ()),
            *(n for n, _o in reply.get("embedded", ()))]
        for name in produced:
            if name not in kw["produced_out"]:
                kw["produced_out"].append(name)
        return [_staged(n, idx=i, tag=reply.get("tag", "new"))
                for i, n in enumerate(reply.get("files", ()), start=1)]


def _wedge_harness(monkeypatch, *, retries=2, replacement="cntr_new", responses=(), order=None):
    """The two collaborators a replacement needs: staging, and the container swap itself."""
    monkeypatch.setattr(rt.config, "deep_research_build_retries", retries)
    staging = _Staging(*responses, order=order)
    monkeypatch.setattr(artifacts_mod, "stage_artifacts", staging)

    async def _replace(_manager, _ledger_key, _old_id):
        if order is not None:
            order.append("replace")
        return replacement

    replace = AsyncMock(side_effect=_replace)
    monkeypatch.setattr(containers_mod, "replace_wedged_container", replace)
    return staging, replace


def _ci_container(tools):
    return next(t["container"] for t in tools if t.get("type") == "code_interpreter")


def _resume_content(messages):
    """The ONE resume item an attempt carried, or None. Both variants open the same way."""
    found = [str(m.get("content")) for m in messages
             if str(m.get("content")).startswith("[Your previous stream")]
    assert len(found) <= 1
    return found[0] if found else None


def _wedge_then_ok(seen):
    """A stream that dies wedged on its first attempt and succeeds on the next."""
    async def stream(_proc, **kw):
        ctx = kw["tool_context"]
        seen.append({"tools": [dict(t) for t in kw["tools"]], "messages": kw["messages"],
                     "assets": list(ctx.sandbox_image_assets or []), "ctx": ctx})
        kw["artifacts_sink"].append({"container_id": ctx.container_id})
        if len(seen) == 1:
            # What a dead attempt leaves behind: ingredients that only existed in that sandbox.
            ctx.sandbox_image_assets.append({"filename": "spent.png", "asset_id": "a1"})
            raise _wedged_stream_error()
        return {"text": "deck built", "tools_used": []}
    return stream


@pytest.mark.unit
class TestWedgedSandboxReplacement:
    async def test_the_abandoned_sandbox_is_banked_then_replaced_everywhere(self, monkeypatch):
        processor = _processor()
        card = _card()
        order = []
        staging, replace = _wedge_harness(
            monkeypatch, order=order,
            responses=[{"files": ["chart.png"], "tag": "old"}])
        seen = []
        monkeypatch.setattr(rt, "_consume_research_stream", _wedge_then_ok(seen))

        build = await _build(processor, card)

        # Banked BEFORE the swap: the old container's files die with it, and a build has no
        # clock that guarantees final staging reaches it before its idle cap.
        assert order == ["stage", "replace"]
        assert staging.calls[0]["container_ids"] == ["cntr_job1"]
        assert staging.calls[0]["ledger_key"] == "C1:1.0#job:j"
        assert staging.calls[0]["expect_filenames"] == ["d.pdf"]
        replace.assert_awaited_once_with(processor.container_manager, "C1:1.0#job:j", "cntr_job1")
        # Every name for the old id now says the replacement.
        assert len(seen) == 2 and build["notes"] == "deck built"
        assert _ci_container(seen[0]["tools"]) == "cntr_job1"
        assert _ci_container(seen[1]["tools"]) == "cntr_new"
        assert seen[1]["ctx"].container_id == "cntr_new"
        # Spent ingredients cleared — left behind, the image tools refuse to remake them.
        assert seen[1]["assets"] == [] and seen[1]["ctx"].sandbox_image_assets == []
        assert build["banked_container_ids"] == ["cntr_job1"]
        # Resumed honestly: the note says the sandbox is empty, and names the banked file so the
        # model neither rebuilds it nor assumes the rest survived.
        note = _resume_content(seen[1]["messages"])
        assert "REPLACED" in note and rt._BUILD_RESUME_NOTE not in note
        assert "chart.png" in note

    async def test_a_second_wedge_is_terminal(self, monkeypatch):
        # Not a second dead sandbox — a request the API keeps refusing.
        processor = _processor()
        _, replace = _wedge_harness(monkeypatch)
        calls = []

        async def always_wedged(_proc, **kw):
            calls.append(kw["messages"])
            kw["artifacts_sink"].append({"container_id": kw["tool_context"].container_id})
            raise _wedged_stream_error()

        monkeypatch.setattr(rt, "_consume_research_stream", always_wedged)

        build = await _build(processor, _card())

        assert len(calls) == 2
        replace.assert_awaited_once()
        assert build is not None and build["notes"] == ""

    async def test_a_failed_replacement_is_terminal_and_never_retries_the_old_sandbox(
            self, monkeypatch):
        processor = _processor()
        staging, replace = _wedge_harness(
            monkeypatch, replacement=None,
            responses=[{"files": ["chart.png"], "tag": "old"}])
        seen = []
        monkeypatch.setattr(rt, "_consume_research_stream", _wedge_then_ok(seen))

        build = await _build(processor, _card())

        assert len(seen) == 1
        replace.assert_awaited_once()
        # Banked anyway: those files are in hand, so the abandoned container must not be read
        # again — and what it held still reaches the publisher.
        assert build["banked_container_ids"] == ["cntr_job1"]
        assert [s.filename for s in build["banked_staged"]] == ["chart.png"]

# ---------------------------------------------- the merge against REAL selection
#
# Substituting staging proves the control flow and nothing about the merge: a fake hands back
# names that are already selected, while the real pass drops a document's ingredients, superseded
# drafts and duplicate content and records those names NOWHERE. So these run the real
# gather + select, with only the container listing and the downloads faked.

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
_PDF = b"%PDF-1.7\n" + b"x" * 32
_PPTX_MAIN = ("application/vnd.openxmlformats-officedocument."
              "presentationml.presentation.main+xml")


def _pptx(members=()):
    """A structurally valid .pptx — the zip entries are what make an embedded member findable."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml",
                    '<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/'
                    'package/2006/content-types"><Override PartName="/ppt/presentation.xml" '
                    f'ContentType="{_PPTX_MAIN}"/></Types>')
        zf.writestr("ppt/presentation.xml", "<presentation/>")
        for name, data in members:
            zf.writestr(name, data)
    return buf.getvalue()


class _Pager:
    """containers.files.list() returns an async-iterable paginator, not a coroutine."""

    def __init__(self, files):
        self._files = files

    def __aiter__(self):
        async def gen():
            for f in self._files:
                yield f
        return gen()


class _Body:
    """content.with_streaming_response.retrieve(...) is an async context manager read in chunks."""

    def __init__(self, payload):
        self._payload = payload
        self.headers = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def iter_bytes(self):
        yield self._payload


def _openai_container(files):
    """files: {file_id: (filename, bytes)} — one container's worth of assistant-written files."""
    oc = MagicMock()
    listed = [MagicMock(id=fid, source="assistant", path=f"/mnt/data/{name}", bytes=len(data))
              for fid, (name, data) in files.items()]
    oc.client.containers.files.list = MagicMock(return_value=_Pager(listed))
    oc.client.containers.files.content.with_streaming_response.retrieve = MagicMock(
        side_effect=lambda fid, container_id=None: _Body(files[fid][1]))
    return oc


@pytest.mark.unit
class TestMergeAgainstRealSelection:
    async def test_a_fresh_ingredient_keeps_the_stale_copy_out(self):
        # The replacement built deck.pdf AND chart.png; real selection suppresses the chart as an
        # ingredient of the document going out, recording the name nowhere. Reading only the
        # accepted list, the merge published the BANKED chart.png beside the fresh deck.
        processor = _processor()
        processor.openai_client = _openai_container(
            {"f1": ("chart.png", _PNG), "f2": ("deck.pdf", _PDF)})
        build = {"ledger_key": "C1:1.0#job:j", "container_ids": ["cntr_new"],
                 "suppress_digests": [], "expect_filenames": ["deck.pdf"],
                 "banked_container_ids": ["cntr_job1"],
                 "banked_staged": [_staged("chart.png", tag="old")],
                 "banked_suppressed_inputs": [], "banked_embedded": [], "banked_produced": []}

        staged = await rt._stage_build(processor, job_id="j", build=build)

        assert [s.filename for s in staged] == ["deck.pdf"]

    async def test_an_unchanged_document_does_not_discard_the_banked_member(self):
        # The replacement only re-made the deck the user had already given us: real selection
        # holds it back as an unchanged input AND records photo.png as embedded in it. Crediting
        # that pair before checking whether the deck ships left an empty manifest and an empty
        # embedded record — the banked photo dropped for a document that never went out.
        processor = _processor()
        deck = _pptx([("ppt/media/image1.png", _PNG)])
        processor.openai_client = _openai_container(
            {"f1": ("deck.pptx", deck), "f2": ("photo.png", _PNG)})
        build = {"ledger_key": "C1:1.0#job:j", "container_ids": ["cntr_new"],
                 # The deck is byte-identical to what we mounted in.
                 "suppress_digests": [hashlib.sha256(deck).hexdigest()],
                 "expect_filenames": ["deck.pptx"],
                 "banked_container_ids": ["cntr_job1"],
                 "banked_staged": [_staged("photo.png", tag="old")],
                 "banked_suppressed_inputs": [], "banked_embedded": [], "banked_produced": []}

        staged = await rt._stage_build(processor, job_id="j", build=build)

        assert [(s.filename, s.candidate["generation"]) for s in staged] == [("photo.png", "old")]
        # And no contradictory metadata: nothing claims the photo shipped inside a deck that did
        # not ship, and nothing calls a delivered filename an unchanged input.
        assert not build.get("embedded_ingredients")
        assert build.get("suppressed_inputs") == ["deck.pptx"]

    async def test_a_delivered_name_is_never_also_reported_as_an_unchanged_input(self):
        # The abandoned pass held photo.png back as unchanged; the replacement transformed it and
        # folded it into the deck that ships. Keeping both records had the delivery prompt call one
        # filename byte-identical AND freshly built.
        processor = _processor()
        deck = _pptx([("ppt/media/image1.png", _PNG)])
        processor.openai_client = _openai_container(
            {"f1": ("deck.pptx", deck), "f2": ("photo.png", _PNG)})
        build = {"ledger_key": "C1:1.0#job:j", "container_ids": ["cntr_new"],
                 "suppress_digests": [], "expect_filenames": ["deck.pptx"],
                 "banked_container_ids": ["cntr_job1"], "banked_staged": [],
                 "banked_suppressed_inputs": ["photo.png"], "banked_embedded": [],
                 "banked_produced": []}

        staged = await rt._stage_build(processor, job_id="j", build=build)

        assert [s.filename for s in staged] == ["deck.pptx"]
        assert build["embedded_ingredients"] == [("photo.png", "deck.pptx")]
        assert not build.get("suppressed_inputs")


# ------------------------------------------------------- the wedge that raises nothing
#
# Measured 2026-09-15 against a deliberately wedged container: the `code_interpreter_call` finishes
# with a terminal status and `outputs=[]`, and the response then completes with `error=null`. No
# exception reaches the build at all, so a recovery hanging off the exception never fires and the
# job keeps taking rounds in a sandbox that cannot run a line. The detector records the container
# in the artifacts sink; the build reads it there.


def _mark_wedged(sink, *ids):
    """What the detector writes — mark_container_wedged's entry shape."""
    from openai_client.container_errors import mark_container_wedged
    mark_container_wedged(sink, list(ids))


@pytest.mark.unit
class TestSilentWedge:
    async def test_a_marked_container_is_banked_and_replaced_with_no_exception(self,
                                                                              monkeypatch):
        processor = _processor()
        card = _card()
        order = []
        staging, replace = _wedge_harness(
            monkeypatch, order=order, responses=[{"files": ["chart.png"], "tag": "old"}])
        seen = []

        async def quietly_wedged(_proc, **kw):
            ctx = kw["tool_context"]
            seen.append({"tools": [dict(t) for t in kw["tools"]], "messages": kw["messages"],
                         "ctx": ctx})
            if len(seen) == 1:
                ctx.sandbox_image_assets.append({"filename": "spent.png", "asset_id": "a1"})
                # The whole point: it RETURNS. Cleanly, with the model's own account of a build
                # that never happened.
                _mark_wedged(kw["artifacts_sink"], ctx.container_id)
                kw["container_gone_sink"].append(ctx.container_id)
                return {"text": "the sandbox errored", "tools_used": []}
            return {"text": "deck built", "tools_used": []}

        monkeypatch.setattr(rt, "_consume_research_stream", quietly_wedged)

        build = await _build(processor, card)

        assert order == ["stage", "replace"]
        assert len(seen) == 2
        assert _ci_container(seen[1]["tools"]) == "cntr_new"
        assert seen[1]["ctx"].container_id == "cntr_new"
        assert seen[1]["ctx"].sandbox_image_assets == []
        assert build["banked_container_ids"] == ["cntr_job1"]
        # The narration of a build that never ran does not survive into delivery.
        assert build["notes"] == "deck built"
        note = _resume_content(seen[1]["messages"])
        assert "REPLACED" in note and "chart.png" in note
        # The gone-record for the abandoned sandbox is cleared, or the replacement would inherit
        # the bridge tools' refusal and its own transport hiccups would go terminal.
        assert seen[1]["ctx"].container_gone_sink == []

    async def test_a_silent_wedge_then_an_exception_wedge_is_still_one_replacement(self,
                                                                                  monkeypatch):
        processor = _processor()
        _, replace = _wedge_harness(monkeypatch)
        seen = []

        async def wedged_twice(_proc, **kw):
            ctx = kw["tool_context"]
            seen.append(kw["messages"])
            if len(seen) == 1:
                _mark_wedged(kw["artifacts_sink"], ctx.container_id)
                return {"text": "the sandbox errored", "tools_used": []}
            raise _wedged_stream_error()

        monkeypatch.setattr(rt, "_consume_research_stream", wedged_twice)

        build = await _build(processor, _card())

        assert len(seen) == 2
        replace.assert_awaited_once()
        assert build is not None and build["notes"] == ""
