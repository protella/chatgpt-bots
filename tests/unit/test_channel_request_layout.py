"""The channel request: one canonical layout, admitted before it is sent (spec §3).

What this file protects, in order of how expensive it is to get wrong:

1. THE BREAKPOINT LINE. Everything above it is a function of (channel, window, H) and is shared
   by every person who speaks in the channel; everything below it varies with who asked. Move one
   item across that line and the single stream stops paying for itself — silently, because the
   request still succeeds and just costs full price.
2. ROLE AUTHORITY. Every piece of post-breakpoint evidence is content people wrote. The moment one
   of them arrives as `developer`, a channel topic becomes an instruction.
3. THE ADMISSION ESTIMATE. It is the last thing before the first API call of the turn, it charges
   raw media at its ceiling, and the summaries that follow it are capped to what it reserved.
4. THREADSTATE IS NEVER TOUCHED. The tripwire for the whole redesign.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from base_client import Message
from config import config
from message_processor import channel_request
from message_processor.channel_request import (BATCHED_IMAGE_CAP, IMAGE_TOKEN_BOUND,
                                               PDF_PAGE_TOKEN_BOUND, cap_summary_to_reserve,
                                               estimate_admission, native_file_token_bound,
                                               prompt_cache_key, to_input_items)
from message_processor.channel_stream import END_MARKER_TEXT, StreamOverBudgetError
from message_processor.turn_runtime import TurnRuntime
from tests.unit.channel_turn_harness import (build_stream, file_ref, item_texts,
                                             no_tools_prepared, normalized,
                                             pin_channel_turn, steering, thread_config)


# --------------------------------------------------------------------------- harness

def _host():
    """A processor stand-in that binds the real assembler and the real mixin builders it calls."""
    from message_processor.handlers.text import TextHandlerMixin
    from message_processor.utilities import MessageUtilitiesMixin

    host = MagicMock()
    host._assemble_channel_attempt = TextHandlerMixin._assemble_channel_attempt.__get__(host)
    host._channel_prepared_tools = TextHandlerMixin._channel_prepared_tools.__get__(host)
    host._build_time_suffix_context = (
        MessageUtilitiesMixin._build_time_suffix_context.__get__(host))
    host._build_message_with_documents = (
        MessageUtilitiesMixin._build_message_with_documents.__get__(host))
    host._build_generation_inflight_note = MagicMock(return_value=None)
    host._build_research_inflight_note = MagicMock(return_value=None)
    host._get_system_prompt = MagicMock(return_value="SYSTEM")
    host._build_tools_array = MagicMock(return_value=None)
    return host


def _message(channel="C1", thread="10.0", ts="10.0", text="hi"):
    return Message(text=text, user_id="U1", channel_id=channel, thread_id=thread,
                   metadata={"ts": ts, "username": "Alice"})


async def _assemble(host, turn, *, model="gpt-5.6-sol", tools_disabled=False,
                    with_estimate=False, message=None):
    request, *_ = await host._assemble_channel_attempt(
        MagicMock(), message or _message(), SimpleNamespace(), turn, thread_config(), model,
        thread_key="C1:10.0", tools_disabled=tools_disabled, with_estimate=with_estimate)
    return request


def _breakpoint_index(items):
    for index, item in enumerate(items):
        content = item.get("content")
        if isinstance(content, list) and any(
                isinstance(p, dict) and p.get("prompt_cache_breakpoint") for p in content):
            return index
    return -1


# --------------------------------------------------------------------------- the line

@pytest.mark.asyncio
async def test_the_stream_is_the_prefix_and_the_end_marker_carries_the_breakpoint():
    turn = TurnRuntime()
    pin_channel_turn(turn, prepared=no_tools_prepared(),
                     messages=[normalized("1.0", "morning"), normalized("2.0", "any news?",
                                                                        sender_id="U2")])
    items = to_input_items(await _assemble(_host(), turn))

    marker = _breakpoint_index(items)
    assert marker == 3        # horizon + two messages, then the marker
    assert items[0]["content"].startswith("[STREAM HORIZON: no summary; coverage begins at")
    assert items[marker]["content"][0]["text"] == END_MARKER_TEXT
    assert "morning" in items[1]["content"] and "any news?" in items[2]["content"]


@pytest.mark.asyncio
async def test_nothing_per_requester_rides_above_the_breakpoint():
    """The load-bearing invariant. Two people asking the same room the same question must share
    every byte up to the marker; if anything about the asker leaks above it, they share none."""
    host = _host()
    stream = build_stream([normalized("1.0", "shared history")])

    def _turn_for(user_id, name, email):
        turn = TurnRuntime()
        pin_channel_turn(
            turn, stream=stream, prepared=no_tools_prepared(),
            requester=channel_request.RequesterFacts(
                user_id=user_id, real_name=name, email=email, tz_label="EST"),
            config=thread_config(custom_instructions=f"{name} likes bullet points"))
        return turn

    first = to_input_items(await _assemble(host, _turn_for("U1", "Alice", "a@x.com")))
    second = to_input_items(await _assemble(host, _turn_for("U2", "Bob", "b@x.com")))

    prefix_a = first[:_breakpoint_index(first) + 1]
    prefix_b = second[:_breakpoint_index(second) + 1]
    assert prefix_a == prefix_b
    # …and the difference is genuinely there, below the line, or this test proves nothing.
    assert "Alice" in "\n".join(item_texts(first)) and "Alice" not in "\n".join(
        item_texts(prefix_a))
    assert "Bob" in "\n".join(item_texts(second))


@pytest.mark.asyncio
async def test_the_cache_key_is_the_channels_not_the_threads():
    turn = TurnRuntime()
    pin_channel_turn(turn, prepared=no_tools_prepared())
    request = await _assemble(_host(), turn)
    assert request.prompt_cache_key == "chan:T1:C1"
    assert prompt_cache_key(None, None) == "chan:unknown:unknown"


@pytest.mark.asyncio
async def test_the_developer_suffix_is_last_and_alone():
    turn = TurnRuntime()
    pin_channel_turn(turn, prepared=no_tools_prepared(),
                     steering_snapshot=steering(policy="Only speak up about deploys.",
                                                facts="The repo is public."),
                     channel_info={"name": "ops", "participation_level": "on",
                                   "reply_in_channel": False},
                     num_members=12)
    items = to_input_items(await _assemble(_host(), turn))

    assert [i["role"] for i in items].count("developer") == 1
    suffix = items[-1]
    assert suffix["role"] == "developer"
    # The policy is an instruction and rides here; the remembered FACT is not and does not.
    assert "Only speak up about deploys." in suffix["content"]
    assert "The repo is public." not in suffix["content"]
    assert "Your replies stay inside a thread." in suffix["content"]
    assert "12 people" in suffix["content"] or "12" in suffix["content"]


@pytest.mark.asyncio
async def test_remembered_facts_arrive_as_user_evidence_never_developer():
    turn = TurnRuntime()
    pin_channel_turn(turn, prepared=no_tools_prepared(),
                     steering_snapshot=steering(facts="Deploys happen on Thursdays."))
    items = to_input_items(await _assemble(_host(), turn))
    carrying = [i for i in items if "Deploys happen on Thursdays." in str(i.get("content"))]
    assert len(carrying) == 1
    assert carrying[0]["role"] == "user"


@pytest.mark.asyncio
async def test_the_topic_and_the_custom_instructions_are_user_evidence_too():
    turn = TurnRuntime()
    pin_channel_turn(turn, prepared=no_tools_prepared(),
                     channel_info={"name": "ops", "topic": "runbooks live in the canvas"},
                     config=thread_config(custom_instructions="Answer in British English."))
    items = to_input_items(await _assemble(_host(), turn))
    for needle in ("runbooks live in the canvas", "Answer in British English."):
        carrying = [i for i in items if needle in str(i.get("content"))]
        assert len(carrying) == 1, needle
        assert carrying[0]["role"] == "user", needle


@pytest.mark.asyncio
async def test_the_instructions_carry_no_requester_and_no_steering():
    """The head of the prefix. The DM prompt puts the speaker's name, their custom instructions,
    the thread roster and the steering block in here; on a channel turn every one of those is a
    per-requester fork of a prefix the whole channel shares."""
    host = _host()
    turn = TurnRuntime()
    pin_channel_turn(turn, prepared=no_tools_prepared(),
                     config=thread_config(custom_instructions="be terse"))
    await _assemble(host, turn)

    args, kwargs = host._get_system_prompt.call_args
    assert args[1] == "UTC"                        # not the requester's zone
    assert args[3] is None and args[4] is None     # no real name, no email
    assert args[5] is None                         # no model line (the suffix states it)
    assert args[8] is None                         # no custom instructions
    assert kwargs["participant_roster"] is None
    assert kwargs["channel_steering"] is None
    assert kwargs["channel_info"] == {}            # a channel, with its furniture in evidence
    assert kwargs["include_date"] is False         # …and no clock: see the two tests below


@pytest.mark.asyncio
async def test_the_invariant_prefix_carries_no_date_and_the_suffix_carries_it_instead():
    """[f18] "Invariant per bot version and channel" and "carries today's date" cannot both be
    true. A dated prefix is a guaranteed cache miss for every channel at every midnight — the one
    thing the single stream exists to keep identical, changing on its own, once a day, forever.
    """
    from message_processor.utilities import MessageUtilitiesMixin

    host = _host()
    host._get_system_prompt = MessageUtilitiesMixin._get_system_prompt.__get__(host)
    client = MagicMock()
    client.name = "Slack"
    client.tool_registry = None

    turn = TurnRuntime()
    pin_channel_turn(turn, prepared=no_tools_prepared())
    request, *_ = await host._assemble_channel_attempt(
        client, _message(), SimpleNamespace(), turn, thread_config(), "gpt-5.6-sol",
        thread_key="C1:10.0")

    assert "Today's date" not in request.instructions
    assert "precise current time" not in request.instructions
    # The DM prefix still carries it — this is a channel deviation, not a global removal.
    assert "Today's date" in host._get_system_prompt(client, "UTC")
    # Nothing is lost: the suffix states the date and the minute together, below the breakpoint.
    suffix = to_input_items(request)[-1]
    assert suffix["role"] == "developer"
    assert "Current date and time:" in suffix["content"]


@pytest.mark.asyncio
async def test_the_rendered_time_is_pinned_so_a_retry_states_the_same_moment():
    """[f19] The suffix's evidence is the turn's, and a retry re-assembles from the pins. Reading
    the clock again meant a retry that crossed a minute boundary — or a midnight — told the model a
    different "now" than the attempt it was retrying."""
    host = _host()
    host._build_time_suffix_context = MagicMock(side_effect=["[at 10:59 AM]", "[at 11:00 AM]"])
    turn = TurnRuntime()
    pin_channel_turn(turn, prepared=no_tools_prepared())

    first = "\n".join(item_texts(to_input_items(await _assemble(host, turn))))
    again = "\n".join(item_texts(to_input_items(await _assemble(host, turn))))

    assert "[at 10:59 AM]" in first and "[at 10:59 AM]" in again
    assert "[at 11:00 AM]" not in again
    host._build_time_suffix_context.assert_called_once()


@pytest.mark.asyncio
async def test_the_job_state_evidence_is_pinned_at_admission_too():
    """[r3-11] Both in-flight notes read the live ThreadManager, so a job that started between the
    admission estimate and the request it admitted added bytes nothing had charged, a job that
    finished changed the evidence under the responder, and a retry could see a third state again."""
    host = _host()
    host._build_generation_inflight_note = MagicMock(
        side_effect=["[an image is being generated]", None, None])
    host._build_research_inflight_note = MagicMock(
        side_effect=[None, "[a deck is being built]", "[a deck is being built]"])
    turn = TurnRuntime()
    pin_channel_turn(turn, prepared=no_tools_prepared())

    admitted = "\n".join(item_texts(to_input_items(await _assemble(host, turn,
                                                                   with_estimate=True))))
    retried = "\n".join(item_texts(to_input_items(await _assemble(host, turn))))

    assert "[an image is being generated]" in admitted
    assert admitted.count("[a deck is being built]") == 0
    # The retry states what was true at admission, not what is true now.
    assert "[an image is being generated]" in retried
    assert "[a deck is being built]" not in retried
    host._build_generation_inflight_note.assert_called_once()
    host._build_research_inflight_note.assert_called_once()


# --------------------------------------------------------------------------- the trigger

@pytest.mark.asyncio
async def test_the_trigger_supplement_carries_the_raw_parts_after_the_breakpoint():
    turn = TurnRuntime()
    pin_channel_turn(
        turn, prepared=no_tools_prepared(),
        image_parts=[{"type": "input_image", "image_url": "data:image/png;base64,AAA",
                      "source": "attachment", "filename": "shot.png", "url": "u",
                      "file_id": "F1"}],
        document_inputs=[{"filename": "q3.pdf", "mimetype": "application/pdf",
                          "summary": "Revenue up 4%.", "content": "raw text",
                          "native": True, "file_data_b64": "QUJD", "size_bytes": 100,
                          "total_pages": 3}],
        file_parts=[{"type": "input_file", "filename": "q3.pdf",
                     "file_data": "data:application/pdf;base64,QUJD"}])
    items = to_input_items(await _assemble(_host(), turn))

    marker = _breakpoint_index(items)
    supplement = items[marker + 1]
    assert supplement["role"] == "user"
    kinds = [p["type"] for p in supplement["content"]]
    assert kinds == ["input_text", "input_image", "input_file"]
    text = supplement["content"][0]["text"]
    assert "ts=10.0" in text and "Revenue up 4%." in text
    # api_part ran: our bookkeeping keys never reach the API.
    assert "source" not in supplement["content"][1] and "file_id" not in supplement["content"][1]


@pytest.mark.asyncio
async def test_a_trigger_the_stream_already_holds_needs_no_verbatim_fallback():
    turn = TurnRuntime()
    pin_channel_turn(turn, prepared=no_tools_prepared(),
                     messages=[normalized("10.0", "what's the deploy window?")],
                     trigger_ts="10.0", trigger_text="what's the deploy window?")
    blob = "\n".join(item_texts(to_input_items(await _assemble(_host(), turn))))
    assert "has not reached the channel stream yet" not in blob


@pytest.mark.asyncio
async def test_a_trigger_slack_had_not_propagated_is_quoted_verbatim():
    """Slack will answer a message before `conversations.history` returns it. Without this the
    model is asked to reply to something it cannot see, which reads as being ignored."""
    turn = TurnRuntime()
    pin_channel_turn(turn, prepared=no_tools_prepared(),
                     messages=[normalized("1.0", "older chatter")],
                     trigger_ts="10.0", trigger_text="what's the deploy window?")
    items = to_input_items(await _assemble(_host(), turn))
    quoted = [i for i in items if "has not reached the channel stream yet" in str(i.get("content"))]
    assert len(quoted) == 1
    assert quoted[0]["role"] == "user"
    assert "what's the deploy window?" in quoted[0]["content"]


@pytest.mark.asyncio
async def test_carried_catch_up_images_ride_their_own_block_under_the_cap():
    carried = [{"type": "input_image", "image_url": f"data:image/png;base64,IMG{i}"}
               for i in range(BATCHED_IMAGE_CAP + 3)]
    turn = TurnRuntime()
    pin_channel_turn(turn, prepared=no_tools_prepared(), batched_image_parts=carried)
    items = to_input_items(await _assemble(_host(), turn))

    block = next(i for i in items
                 if isinstance(i.get("content"), list)
                 and "earlier messages in this catch-up" in i["content"][0].get("text", ""))
    assert block["role"] == "user"
    images = [p for p in block["content"] if p["type"] == "input_image"]
    assert len(images) == BATCHED_IMAGE_CAP
    # The overflow is STATED. A model that silently saw ten of thirteen would answer as if it
    # had seen all of them.
    assert "3 image(s)" in block["content"][0]["text"]


@pytest.mark.asyncio
async def test_the_triggers_own_images_win_the_slots_and_the_loss_is_still_stated():
    """[r2-9] The trigger's own images take the slots — but a catch-up whose pictures were ALL
    dropped used to be reported as nothing at all, and the model answered as if they had never
    existed. There is no image to carry, so the marker travels on its own."""
    own = [{"type": "input_image", "image_url": f"data:image/png;base64,OWN{i}"}
           for i in range(BATCHED_IMAGE_CAP)]
    turn = TurnRuntime()
    pin_channel_turn(turn, prepared=no_tools_prepared(), image_parts=own,
                     batched_image_parts=[{"type": "input_image", "image_url": "x"},
                                          {"type": "input_image", "image_url": "y"}])
    items = to_input_items(await _assemble(_host(), turn))
    blob = "\n".join(item_texts(items))
    assert "2 image(s) from earlier messages in this catch-up could not be attached at all" in blob
    # Text only: there were no slots, so it must not smuggle an image part in behind the notice.
    notice = next(i for i in items if "could not be attached at all" in str(i.get("content")))
    assert notice["role"] == "user" and isinstance(notice["content"], str)


@pytest.mark.asyncio
async def test_images_trimmed_before_the_turn_are_reported_even_with_nothing_to_carry():
    """[r2-9] Same silence from the other direction: the batch was already trimmed upstream, so
    there are no parts left to carry and a count of what was lost."""
    from dataclasses import replace

    turn = TurnRuntime()
    ctx = pin_channel_turn(turn, prepared=no_tools_prepared(), batched_image_parts=[])
    turn.channel_turn_context = replace(ctx, batched_images_omitted=4)
    blob = "\n".join(item_texts(to_input_items(await _assemble(_host(), turn))))
    assert "4 image(s)" in blob


@pytest.mark.asyncio
async def test_a_catch_up_that_lost_no_images_says_nothing():
    turn = TurnRuntime()
    pin_channel_turn(turn, prepared=no_tools_prepared(), batched_image_parts=[])
    blob = "\n".join(item_texts(to_input_items(await _assemble(_host(), turn))))
    assert "catch-up" not in blob


# --------------------------------------------------------------------------- file authorization

def test_the_pinned_window_authorizes_every_file_it_rendered():
    stream = build_stream([
        normalized("1.0", "here", files=[file_ref("F1", "runbook.pdf")]),
        normalized("2.0", "and this", thread_root_ts="1.0",
                   files=[file_ref("F2", "chart.png", "image/png", kind="image")])])
    catalog = channel_request.canonical_files_from_stream(stream)
    assert set(catalog) == {"F1", "F2"}
    assert catalog["F1"]["message_ts"] == "1.0"
    assert catalog["F2"]["thread_root_ts"] == "1.0"      # a reply names its thread


def test_a_file_with_no_id_is_not_authorized():
    """A filename is a hint; an id is permission. A ref Slack gave us without one cannot be
    resolved, and inventing a key for it would put an unauthorized entry in the catalog."""
    stream = build_stream([normalized("1.0", "x", files=[file_ref(file_id=None)])])
    assert channel_request.canonical_files_from_stream(stream) == {}


def test_an_absent_sources_files_are_merged_from_the_live_payload():
    stream = build_stream([normalized("2.0", "what do the numbers say?")])
    source = channel_request.CohortSource(
        ts="1.0", text="numbers",
        files=channel_request.file_refs_from_attachments([
            {"id": "F9", "name": "data.csv", "mimetype": "text/csv", "size": 40,
             "url": "https://files.slack.com/files-pri/T1-F9/data.csv"}]))
    merged = channel_request.merge_absent_source_files(
        channel_request.canonical_files_from_stream(stream), [source], "C1")
    assert merged["F9"]["mime_type"] == "text/csv"
    assert merged["F9"]["kind"] == "file"


@pytest.mark.asyncio
async def test_read_document_resolves_a_file_only_the_stream_knew_about():
    """The point of the catalog: a document dropped in ANOTHER thread of this channel is readable
    on the turn it is asked about, not one turn later once a `documents` row exists."""
    from message_processor.document_tools import execute_read_document
    from tool_registry import ToolContext

    db = MagicMock()
    db.get_thread_documents_async = AsyncMock(return_value=[])
    db.get_channel_documents_async = AsyncMock(return_value=[])
    client = MagicMock()
    client.download_file = AsyncMock(return_value=b"%PDF-1.4 fake")
    ctx = ToolContext(channel_id="C1", thread_ts="10.0", db=db, client=client,
                      canonical_files={"F1": {
                          "file_id": "F1", "filename": "runbook.pdf",
                          "mime_type": "application/pdf",
                          "url_private": "https://files.slack.com/x", "kind": "file"}})
    with patch("message_processor.document_tools._document_handler") as handler:
        handler.safe_extract_content_async = AsyncMock(
            return_value={"content": "step one: do not panic"})
        out = await execute_read_document(ctx, {"filename": "runbook.pdf"})
    assert out["ok"] is True
    assert "do not panic" in out["content"]
    assert out["origin"] == "shared elsewhere in this channel"


@pytest.mark.asyncio
async def test_read_document_refuses_an_image_from_the_catalog():
    """`read_document` extracts text; an image has none. `view_image` is the tool for those, and
    silently returning an extraction failure for a PNG would look like a broken file."""
    from message_processor.document_tools import execute_read_document
    from tool_registry import ToolContext

    db = MagicMock()
    db.get_thread_documents_async = AsyncMock(return_value=[])
    db.get_channel_documents_async = AsyncMock(return_value=[])
    ctx = ToolContext(channel_id="C1", thread_ts="10.0", db=db, client=MagicMock(),
                      canonical_files={"F2": {"file_id": "F2", "filename": "chart.png",
                                              "mime_type": "image/png",
                                              "url_private": "u", "kind": "image"}})
    out = await execute_read_document(ctx, {"filename": "chart.png"})
    assert out["ok"] is False and out["error"] == "document_not_found"


# --------------------------------------------------------- r3-4: failures are evidence, not silence

class TestAFailedAttachmentIsVisibleToTheResponder:
    """[r3-4] On a channel turn the failure notice is a real Slack message, so it is deliberately
    kept out of ThreadState — and then nothing put it anywhere the responder could see. The stream
    records that a file was attached, so silence about the failure is the one outcome that reads as
    "I have that file".
    """

    @pytest.mark.asyncio
    async def test_the_names_are_rendered_as_explicit_failure_evidence(self):
        turn = TurnRuntime()
        pin_channel_turn(turn, prepared=no_tools_prepared(),
                         failed_attachment_names=("budget.numbers", "scan.tif"))
        request = await _assemble(_host(), turn)
        rendered = "\n".join(item_texts(request.input_items))
        assert "FAILED to load" in rendered
        assert "budget.numbers" in rendered and "scan.tif" in rendered
        # And it says what to do about it, because a model told only "not present" invents contents.
        assert "do not guess" in rendered

    @pytest.mark.asyncio
    async def test_a_turn_whose_every_attachment_failed_still_says_so(self):
        """The supplement used to exist only to carry successful parts, so a turn with text and one
        broken file rendered nothing at all."""
        turn = TurnRuntime()
        pin_channel_turn(turn, prepared=no_tools_prepared(),
                         failed_attachment_names=("scan.tif",))
        rendered = "\n".join(item_texts((await _assemble(_host(), turn)).input_items))
        assert "scan.tif" in rendered
        # No "their contents are here" header, which would be flatly untrue here.
        assert "their contents are here" not in rendered

    @pytest.mark.asyncio
    async def test_a_turn_with_no_failures_renders_nothing_extra(self):
        turn = TurnRuntime()
        pin_channel_turn(turn, prepared=no_tools_prepared())
        rendered = "\n".join(item_texts((await _assemble(_host(), turn)).input_items))
        assert "FAILED to load" not in rendered

    @pytest.mark.asyncio
    async def test_the_evidence_claims_the_notice_only_when_slack_confirmed_it(self):
        """[r4-4] "The user has been told they failed" was asserted unconditionally, and
        `send_message` returns None on a SlackApiError it swallowed. A responder that believes the
        failure is already acknowledged does not acknowledge it either, so a dropped notice meant
        nobody mentioned the file at all. The wording follows the delivery."""
        host = _host()

        posted = TurnRuntime()
        ctx = pin_channel_turn(posted, prepared=no_tools_prepared(),
                               failed_attachment_names=("scan.tif",))
        ctx.notice_delivery["failed_attachments"] = True
        rendered = "\n".join(item_texts((await _assemble(host, posted)).input_items))
        assert "The user has been told they failed" in rendered

        dropped = TurnRuntime()
        ctx = pin_channel_turn(dropped, prepared=no_tools_prepared(),
                               failed_attachment_names=("scan.tif",))
        ctx.notice_delivery["failed_attachments"] = False
        rendered = "\n".join(item_texts((await _assemble(host, dropped)).input_items))
        assert "has been told" not in rendered
        assert "may NOT have been notified" in rendered
        assert "acknowledge the failure in your reply" in rendered

    @pytest.mark.asyncio
    async def test_an_unknown_delivery_renders_the_wording_admission_charged(self):
        """The notice posts AFTER the request is measured, so at admission neither outcome is known.
        The longer wording is what gets charged, and both real outcomes fit inside it."""
        host = _host()
        unknown = TurnRuntime()
        pin_channel_turn(unknown, prepared=no_tools_prepared(),
                         failed_attachment_names=("scan.tif",))
        charged = await _assemble(host, unknown, with_estimate=True)
        assert "may NOT have been notified" in "\n".join(item_texts(charged.input_items))

        for landed in (True, False):
            turn = TurnRuntime()
            ctx = pin_channel_turn(turn, prepared=no_tools_prepared(),
                                   failed_attachment_names=("scan.tif",))
            ctx.notice_delivery["failed_attachments"] = landed
            sent = await _assemble(host, turn, with_estimate=True)
            assert sent.estimate.total_tokens <= charged.estimate.total_tokens, landed

    @pytest.mark.asyncio
    async def test_the_failure_evidence_is_charged_by_admission(self):
        """It is bytes in the request like any other evidence, so it has to be paid for."""
        plain = TurnRuntime()
        pin_channel_turn(plain, prepared=no_tools_prepared())
        failed = TurnRuntime()
        pin_channel_turn(failed, prepared=no_tools_prepared(),
                         failed_attachment_names=("budget.numbers",))
        host = _host()
        bare = await _assemble(host, plain, with_estimate=True)
        with_note = await _assemble(host, failed, with_estimate=True)
        assert with_note.estimate.total_tokens > bare.estimate.total_tokens


# --------------------------------------------------------------------------- admission

@pytest.fixture
def counted_admission():
    """The exact-count path, for the tests that assert measured token NUMBERS.

    Those numbers only hold when the o200k vocabulary is actually loaded; with a cold tiktoken cache
    and no egress the estimator answers from its byte-ratio fallback, which is correct behaviour and
    a different set of figures.
    """
    import token_counter
    if token_counter.wait_for_admission_encoder() is None:
        pytest.skip("o200k vocabulary unavailable — the fallback path is covered separately")


def test_an_image_is_charged_its_ceiling_not_its_base64():
    """Counting the data URI as text reports tens of millions of tokens for one screenshot and
    refuses every turn that had a picture in it."""
    huge = "A" * 4_000_000
    items = [{"role": "user", "content": [
        {"type": "input_text", "text": "look"},
        {"type": "input_image", "image_url": f"data:image/png;base64,{huge}"}]}]
    estimate = estimate_admission(instructions="", input_items=items, tools=None,
                                 raw_document_texts=(), native_file_bounds=(),
                                 model="gpt-5.6-sol")
    assert estimate.breakdown["images"] == IMAGE_TOKEN_BOUND
    assert estimate.breakdown["items"] < 10
    assert estimate.fits


def test_a_native_file_is_charged_the_worse_of_bytes_and_pages():
    assert native_file_token_bound(5000, None) == 5000
    assert native_file_token_bound(5000, 4) == 4 * PDF_PAGE_TOKEN_BOUND
    assert native_file_token_bound(None, None) == 0
    # Uncapped on purpose: a file whose worst case cannot fit genuinely cannot be guaranteed to.
    assert native_file_token_bound(50_000_000, None) == 50_000_000


def test_raw_document_text_is_charged_whole_and_reserved_at_its_charge():
    """[r3-1] A document is charged its whole raw text even though only a summary will be sent, and
    the reserve IS that charge: the turn already paid for those bytes, so handing the summary any
    less would throw away room it had bought. A tighter, counted reserve looks harmless and is not —
    a short document's counted reserve will not even hold the truncation marker, so its summary
    would be dropped in full.

    Needs no tokenizer, which is why it is not gated on one.
    """
    estimate = estimate_admission(
        instructions="", input_items=[], tools=None,
        raw_document_texts=(("a.pdf", "x" * 400), ("b.csv", "y" * 800)),
        native_file_bounds=(), model="gpt-5.6-sol")
    assert estimate.document_reserves == (("a.pdf", 400), ("b.csv", 800))
    assert estimate.breakdown["document_text"] == 1200


def test_two_documents_sharing_a_reserve_key_are_charged_and_granted_twice():
    """[r4-3] `raw_document_texts` keys by file_id/url/filename, and Slack will happily deliver the
    same file twice. Collapsed into a mapping, the pair was charged once and then granted that one
    reserve EACH at finalization — two summaries spending room bought once, inside a request already
    admitted. One entry per document, in charge order."""
    estimate = estimate_admission(
        instructions="", input_items=[], tools=None,
        raw_document_texts=(("F1", "x" * 300), ("F1", "y" * 500)),
        native_file_bounds=(), model="gpt-5.6-sol")
    assert estimate.document_reserves == (("F1", 300), ("F1", 500))
    assert estimate.breakdown["document_text"] == 800


def test_every_item_pays_for_framing_its_text_cannot_show():
    """[r2-4] Role wrappers and message delimiters are tokens the API adds around content we never
    see, so a request assembled from forty items is not the sum of forty strings. Forty EMPTY items
    used to cost zero, which is a bound that does not bound anything."""
    from token_counter import ITEM_STRUCTURAL_OVERHEAD

    empty = [{"role": "user", "content": ""} for _ in range(40)]
    estimate = estimate_admission(instructions="", input_items=empty, tools=None,
                                  raw_document_texts=(), native_file_bounds=(),
                                  model="gpt-5.6-sol")
    # Forty items plus the developer instructions, which is framed the same way.
    assert estimate.breakdown["structure"] == 41 * ITEM_STRUCTURAL_OVERHEAD
    assert estimate.total_tokens >= 40 * ITEM_STRUCTURAL_OVERHEAD


# Ground truth: the o200k_base token count for each of these exact literals, recorded as a NUMBER so
# the assertions below cannot be an estimator's own formula compared against itself. The hex digest
# is the case that matters — 1600 real tokens for 1600 bytes, which is the worst ratio any byte-level
# BPE tokenizer can reach, and which the shipped bytes/3 estimate charged 534 for.
_MEASURED_O200K_TOKENS = (
    ("3f9a1c7e" * 200, 1600),                                        # hex digest, 1.00 bytes/token
    ("QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=" * 60, 1260),             # base64
    ("1,2.5,foo,2026-07-30,,7\n" * 100, 1700),                       # dense CSV
    ('{"a":1,"bb":22,"ccc":"ddd","e":[1,2,3]}' * 60, 1260),          # minified JSON
    ("日本語のテキスト" * 100, 600),                                   # CJK
    ("\U0001f600\U0001f680\U0001f9ea\U0001f4a1" * 100, 800),         # emoji
)


def test_the_admission_charge_is_never_below_the_real_token_count():
    """[r3-1] THE admission contract, and it is a bound rather than an estimate: a byte-level BPE
    token consumes at least one input byte, so no such tokenizer can emit more tokens than the text
    has bytes. Every literal below satisfies it by construction — which is the point. A charge that
    needs no vocabulary cannot be wrong about a vocabulary it has never seen, which is exactly the
    hole a counted estimate plus a 1.15x multiplier left open for gpt-5.6's unpublished table.

    Not gated on the tokenizer loading: the charge does not use one.
    """
    from token_counter import admission_charge, estimate_tokens

    for text, measured in _MEASURED_O200K_TOKENS:
        assert admission_charge(text) >= measured, text[:20]
        # And every one of these is content chars/4 would have waved through.
        assert estimate_tokens(text) < measured, text[:20]

    # The bound is TIGHT, not merely large: a hex digest really does cost one token per byte, so
    # nothing smaller than the byte count would have held for it.
    hex_log, hex_tokens = _MEASURED_O200K_TOKENS[0]
    assert admission_charge(hex_log) == hex_tokens


def test_the_admission_charge_needs_no_tokenizer_at_all(monkeypatch):
    """[r3-1] A cold tiktoken cache and no egress used to mean admission answered from utf-8 bytes/2
    — a number the module's own comment admitted was below the worst case. The charge has no
    tokenizer to lose now, so the fallback question does not arise for admission.
    """
    import token_counter

    monkeypatch.setattr(token_counter, "_admission_encoder", lambda: None)
    for text, measured in _MEASURED_O200K_TOKENS:
        size = len(text.encode("utf-8"))
        assert token_counter.admission_charge(text) == size
        assert token_counter.admission_charge(text) >= measured, text[:20]
        # And the RESERVE estimator, with no vocabulary, falls back to the same byte bound rather
        # than to a ratio that would license more bytes than were charged.
        assert token_counter.estimate_tokens_conservative(text) == size


def test_the_reserve_estimator_covers_the_measured_token_count(counted_admission):
    """The counted estimator no longer decides whether a turn may be sent, but it does decide how
    much of a summary survives, so it still has to be at least the real count of what it measures.
    """
    from token_counter import estimate_tokens_conservative

    for text, measured in _MEASURED_O200K_TOKENS:
        assert estimate_tokens_conservative(text) >= measured, text[:20]


class TestTheReserveTokenizerLoadsWithoutWedgingATurn:
    """[r3-2] The loader runs in a daemon thread because `get_encoding` fetches a vocabulary over
    the network on a cold cache, and nothing may block a channel turn on that. Two seams broke:
    a caller that arrived while BOOT's load was still running skipped the grace entirely, and a
    failed load was remembered forever.
    """

    @pytest.fixture(autouse=True)
    def _fresh_loader(self, monkeypatch):
        import token_counter
        monkeypatch.setattr(token_counter, "_ENCODER_SLOT", {})
        monkeypatch.setattr(token_counter, "_ENCODER_THREAD", None)
        monkeypatch.setattr(token_counter, "_ENCODER_ATTEMPTS", 0)

    def test_a_caller_waits_on_the_load_boot_started(self, monkeypatch):
        """Boot kicks the load off without waiting (`timeout=0`), so the first turn finds a thread
        it did not start. It used to get None back immediately and answer from the fallback with a
        perfectly good vocabulary arriving milliseconds later."""
        import threading

        import token_counter

        release = threading.Event()

        def _slow_load():
            release.wait(5.0)
            with token_counter._ENCODER_LOCK:
                token_counter._ENCODER_SLOT["encoder"] = "VOCAB"
                token_counter._ENCODER_THREAD = None

        monkeypatch.setattr(token_counter, "_load_admission_encoder", _slow_load)
        # Boot: starts the thread, waits for nothing.
        assert token_counter.wait_for_admission_encoder(timeout=0) is None
        boot_thread = token_counter._ENCODER_THREAD
        assert boot_thread is not None and boot_thread.is_alive()

        # The first turn arrives mid-load. It must join BOOT's thread, not decline to wait.
        joined = []
        monkeypatch.setattr(boot_thread, "join",
                            lambda t=None: (joined.append(t), release.set(),
                                            threading.Thread.join(boot_thread, 5.0))[0])
        assert token_counter._admission_encoder() == "VOCAB"
        assert joined == [token_counter._ENCODER_GRACE_SECONDS]

    def test_a_failed_load_is_retried_by_a_later_caller(self, monkeypatch):
        """A cold cache with momentarily no network is not a permanent verdict. Storing None as the
        answer meant one unlucky boot cost the whole process its exact counts."""
        import token_counter

        attempts = []

        def _flaky_load():
            attempts.append(1)
            encoder = "VOCAB" if len(attempts) > 1 else None
            with token_counter._ENCODER_LOCK:
                if encoder is not None:
                    token_counter._ENCODER_SLOT["encoder"] = encoder
                token_counter._ENCODER_THREAD = None

        monkeypatch.setattr(token_counter, "_load_admission_encoder", _flaky_load)
        assert token_counter.wait_for_admission_encoder(timeout=5.0) is None
        assert len(attempts) == 1
        # The next caller tries again rather than trusting a remembered failure.
        assert token_counter.wait_for_admission_encoder(timeout=5.0) == "VOCAB"
        assert len(attempts) == 2

    def test_retries_are_bounded_so_a_dead_box_stops_paying_the_grace(self, monkeypatch):
        """A permanently offline box must not spend the grace on every request forever."""
        import token_counter

        attempts = []

        def _always_fails():
            attempts.append(1)
            with token_counter._ENCODER_LOCK:
                token_counter._ENCODER_THREAD = None

        monkeypatch.setattr(token_counter, "_load_admission_encoder", _always_fails)
        for _ in range(token_counter._ENCODER_MAX_ATTEMPTS + 4):
            assert token_counter._admission_encoder() is None
        assert len(attempts) == token_counter._ENCODER_MAX_ATTEMPTS
        # And the reserve estimator keeps answering throughout, from the byte bound.
        assert token_counter.estimate_tokens_conservative("hello") == 5


def test_a_special_token_literal_in_a_message_does_not_break_admission():
    """tiktoken's `encode` REFUSES text containing a special-token literal, and a user is entirely
    capable of typing one into a channel. `encode_ordinary` treats it as the text it is."""
    from token_counter import admission_charge, estimate_tokens_conservative

    typed = "what happens if I say <|endoftext|> out loud?"
    assert estimate_tokens_conservative(typed) > 0
    assert admission_charge(typed) == len(typed.encode("utf-8"))


def test_a_summary_can_never_exceed_what_the_estimate_reserved_for_it():
    long_summary = "word " * 500
    capped = cap_summary_to_reserve(long_summary, 50)
    # Measured in the ADMISSION currency [r3-1]: the reserve licenses bytes, because bytes are what
    # the document it replaces was charged. A token estimate here would let a prose summary of a
    # dense document run several times the bytes it was admitted at.
    assert len(capped.encode("utf-8")) <= 50
    assert capped.endswith("[summary truncated to its admitted size]")
    # Under the reserve, it is returned untouched — this fires almost never and must not mangle.
    assert cap_summary_to_reserve("short", 1000) == "short"


class TestTheCapHoldsAtEveryReserve:
    """[f10] The cap is the second half of the admission guarantee: the request was admitted at a
    size that charged this document's RAW text, so the summary standing in for it may not exceed
    that. Every one of these returned MORE than the reserve before the fix — the zero case
    returned the entire summary.
    """

    @staticmethod
    def _fits(text, reserve):
        # utf-8 bytes: the currency the document's own charge was in [r3-1].
        return len(text.encode("utf-8")) <= reserve

    def test_zero_reserve_admits_nothing(self):
        # A document the estimate charged nothing for is a document with no text. Handing back the
        # whole summary here hands the API bytes the budget check never saw.
        assert cap_summary_to_reserve("a real summary of a real document", 0) == ""
        assert cap_summary_to_reserve("a real summary", -25) == ""

    def test_a_reserve_too_small_for_the_marker_admits_nothing(self):
        # The honest note costs what it costs. Below that there is no truthful output, so the answer
        # is nothing — never the marker's own overflow.
        note = len(channel_request.TRUNCATION_NOTE.encode("utf-8"))
        assert cap_summary_to_reserve("x" * 5000, 1) == ""
        assert cap_summary_to_reserve("x" * 5000, note - 1) == ""
        assert cap_summary_to_reserve("x" * 5000, note).endswith(channel_request.TRUNCATION_NOTE)
        assert self._fits(cap_summary_to_reserve("x" * 5000, note), note)

    def test_a_reserve_below_the_marker_but_above_nothing_still_admits_nothing(self):
        """Every reserve from zero up to the marker's own size, exhaustively — the boundary is the
        one place a cap gets to return something that does not fit."""
        note = len(channel_request.TRUNCATION_NOTE.encode("utf-8"))
        for reserve in range(0, note):
            assert cap_summary_to_reserve("x" * 5000, reserve) == "", reserve

    def test_a_tiny_reserve_keeps_the_marker_and_fits(self):
        out = cap_summary_to_reserve("x" * 5000, 60)
        assert out.endswith("[summary truncated to its admitted size]")
        assert self._fits(out, 60)

    def test_a_multibyte_summary_is_cut_on_a_character_boundary_and_fits(self):
        """Cutting a byte prefix can land inside a multi-byte character; the result must still be
        valid text AND still fit, at every reserve."""
        for reserve in (15, 20, 50, 200, 999):
            out = cap_summary_to_reserve("日本語の要約です。" * 400, reserve)
            assert self._fits(out, reserve), (reserve, out[:40])
            out.encode("utf-8").decode("utf-8")  # no lone surrogate / partial character survived

    def test_the_capped_result_is_always_a_prefix_of_the_summary(self):
        summary = "The document lists every regional total for Q3 and flags two outliers."
        out = cap_summary_to_reserve(summary, 60)
        head = out.split("\n[summary truncated")[0]
        assert summary.startswith(head) and head


@pytest.mark.asyncio
async def test_an_over_budget_request_is_refused_before_any_api_call():
    turn = TurnRuntime()
    pin_channel_turn(
        turn, prepared=no_tools_prepared(),
        document_inputs=[{"filename": "huge.pdf", "mimetype": "application/pdf",
                          "content": "z" * 8_000_000, "summary": None, "native": False,
                          "size_bytes": 8_000_000}])
    request = await _assemble(_host(), turn, with_estimate=True)
    assert not request.estimate.fits
    with pytest.raises(StreamOverBudgetError) as raised:
        channel_request.raise_if_over_budget(request.estimate, channel_id="C1")
    assert "over the" in str(raised.value)


@pytest.mark.asyncio
async def test_a_refusal_reports_the_real_count_beside_the_charged_bound(caplog):
    """The charge is a bound, so it over-charges prose about 4.5x. When a window is refused, the one
    question left is whether it was genuinely too big or whether the bound refused something that
    would have fit — so the refusal logs both figures. Counted only here, never on the path that
    fits."""
    turn = TurnRuntime()
    pin_channel_turn(
        turn, prepared=no_tools_prepared(),
        document_inputs=[{"filename": "huge.pdf", "mimetype": "application/pdf",
                          "content": "prose about quarterly totals " * 100_000, "summary": None,
                          "native": False, "size_bytes": 2_800_000}])
    request = await _assemble(_host(), turn, with_estimate=True)
    assert not request.estimate.fits
    with caplog.at_level("WARNING"), pytest.raises(StreamOverBudgetError):
        channel_request.raise_if_over_budget(request.estimate, channel_id="C1",
                                             counted_text=request.countable_text)
    assert any("real o200k tokens" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_a_request_that_fits_is_never_counted():
    """`countable_text` joins every string in the request; asking for it on the happy path would
    add a full pass over the whole window to every turn."""
    turn = TurnRuntime()
    pin_channel_turn(turn, prepared=no_tools_prepared())
    request = await _assemble(_host(), turn, with_estimate=True)
    assert request.estimate.fits
    with patch.object(channel_request, "estimate_tokens_conservative") as counted:
        channel_request.raise_if_over_budget(request.estimate, channel_id="C1",
                                             counted_text="")
    counted.assert_not_called()


@pytest.mark.asyncio
async def test_the_estimate_runs_before_summarization_and_caps_what_follows():
    """The ordering [r3-4]: the summarizer IS a Responses API call, so a turn that could never
    have been sent must not spend one per attached document to find that out — and once the
    request has been admitted at a size, the summary it then renders cannot exceed it."""
    from message_processor.base import MessageProcessor

    with patch("message_processor.base.AsyncThreadStateManager"), \
         patch("message_processor.base.OpenAIClient"):
        processor = MessageProcessor()
    processor.db = None
    order = []
    processor._summarize_document_for_attach = AsyncMock(
        side_effect=lambda *a, **k: order.append("summarize") or ("SUM " * 400))
    processor._update_status = MagicMock()
    processor.thread_manager.get_or_create_document_ledger = MagicMock()
    processor._build_channel_info = AsyncMock(return_value=None)
    processor._build_tools_array = MagicMock(return_value=None)
    processor._get_system_prompt = MagicMock(return_value="SYSTEM")
    processor._prepare_channel_turn_tools = AsyncMock(
        return_value=no_tools_prepared())

    doc = {"filename": "q3.pdf", "mimetype": "application/pdf", "content": "x" * 4000,
           "summary": None, "native": False, "size_bytes": 4000}
    processor._stage_document_summary(
        doc, {"content": doc["content"]}, _message(), url_private="u", size_bytes=4000)

    turn = TurnRuntime()
    pin_channel_turn(turn, prepared=no_tools_prepared())
    real_estimate = channel_request.estimate_admission

    def _tracking_estimate(**kw):
        order.append("estimate")
        return real_estimate(**kw)

    with patch.object(channel_request, "estimate_admission", _tracking_estimate):
        await processor._admit_channel_request(
            _message(), MagicMock(), turn, SimpleNamespace(channel_id="C1", thread_ts="10.0"),
            thread_config(), None, stream=turn.channel_stream, steering=steering(),
            image_inputs=[], file_inputs=[], document_inputs=[doc],
            batched_image_inputs=[], batched_images_omitted=0)

    assert order == ["estimate", "summarize"]
    # Capped to the reserve, in the currency the reserve licenses: the summary that was actually
    # rendered occupies no more bytes than the document's raw text was charged.
    assert len(doc["summary"].encode("utf-8")) <= len(("x" * 4000).encode("utf-8"))


@pytest.mark.asyncio
async def test_a_queued_documents_summary_waits_for_the_catch_up_turns_admission():
    """[r5-2] The same ordering, for a document an EARLIER queued message brought. The drain that
    folded that message into this catch-up could not summarize it — the turn did not exist yet, so
    there was nothing to admit — and doing it anyway spent a Responses call on a turn that might
    still be refused. It is staged there and finalized here, after the estimate.

    With NO reserve: the estimate never charged it, because it is not in this request. What it is
    here for is the ledger row that read_document/mount_file reach it by, so capping its summary
    against room it never occupies would only destroy the row's content."""
    from message_processor.base import MessageProcessor

    with patch("message_processor.base.AsyncThreadStateManager"), \
         patch("message_processor.base.OpenAIClient"):
        processor = MessageProcessor()
    processor.db = None
    order = []
    processor._summarize_document_for_attach = AsyncMock(
        side_effect=lambda *a, **k: order.append("summarize") or ("SUM " * 400))
    processor._update_status = MagicMock()
    processor.thread_manager.get_or_create_document_ledger = MagicMock()
    processor._build_channel_info = AsyncMock(return_value=None)
    processor._build_tools_array = MagicMock(return_value=None)
    processor._get_system_prompt = MagicMock(return_value="SYSTEM")
    processor._prepare_channel_turn_tools = AsyncMock(return_value=no_tools_prepared())

    carried = {"filename": "earlier.pdf", "mimetype": "application/pdf", "content": "y" * 300,
               "summary": None, "native": False, "size_bytes": 300}
    processor._stage_document_summary(
        carried, {"content": carried["content"]}, _message(), url_private="u", size_bytes=300)

    message = _message()
    message.metadata["batched_deferred_documents"] = [carried]
    turn = TurnRuntime()
    pin_channel_turn(turn, prepared=no_tools_prepared())
    real_estimate = channel_request.estimate_admission

    def _tracking_estimate(**kw):
        order.append("estimate")
        return real_estimate(**kw)

    with patch.object(channel_request, "estimate_admission", _tracking_estimate):
        await processor._admit_channel_request(
            message, MagicMock(), turn, SimpleNamespace(channel_id="C1", thread_ts="10.0"),
            thread_config(), None, stream=turn.channel_stream, steering=steering(),
            image_inputs=[], file_inputs=[], document_inputs=[],
            batched_image_inputs=[], batched_images_omitted=0)

    assert order == ["estimate", "summarize"]
    assert carried["summary"] == "SUM " * 400, "an uncharged summary must not be capped"
    assert "_persist" not in carried, "the staged step must be consumed, not left for a retry"


@pytest.mark.asyncio
async def test_the_admitted_size_bounds_the_request_that_is_actually_sent():
    """[f10] THE invariant, end to end: a request that passed admission cannot then grow past the
    budget it was admitted against. Everything else in this section protects one term of it; this
    measures the assembled request AFTER summarization against the estimate that admitted it, with
    a summarizer that returns far more prose than the document it describes.
    """
    from message_processor.base import MessageProcessor

    with patch("message_processor.base.AsyncThreadStateManager"), \
         patch("message_processor.base.OpenAIClient"):
        processor = MessageProcessor()
    processor.db = None
    processor._summarize_document_for_attach = AsyncMock(return_value="verbose prose " * 2000)
    processor._update_status = MagicMock()
    processor.thread_manager.get_or_create_document_ledger = MagicMock()
    processor._build_channel_info = AsyncMock(return_value=None)
    processor._build_tools_array = MagicMock(return_value=None)
    processor._get_system_prompt = MagicMock(return_value="SYSTEM")
    processor._prepare_channel_turn_tools = AsyncMock(return_value=no_tools_prepared())

    # A short document: the reserve is small, so an unbounded summary would blow straight through it.
    doc = {"filename": "note.txt", "mimetype": "text/plain", "content": "x" * 150,
           "summary": None, "native": False, "size_bytes": 150}
    processor._stage_document_summary(
        doc, {"content": doc["content"]}, _message(), url_private="u", size_bytes=150)

    turn = TurnRuntime()
    pin_channel_turn(turn, prepared=no_tools_prepared(), document_inputs=[doc])
    admitted = []
    real_estimate = channel_request.estimate_admission

    def _capture(**kw):
        estimate = real_estimate(**kw)
        admitted.append(estimate)
        return estimate

    with patch.object(channel_request, "estimate_admission", _capture):
        await processor._admit_channel_request(
            _message(), MagicMock(), turn, SimpleNamespace(channel_id="C1", thread_ts="10.0"),
            thread_config(), None, stream=turn.channel_stream, steering=steering(),
            image_inputs=[], file_inputs=[], document_inputs=[doc],
            batched_image_inputs=[], batched_images_omitted=0)

    assert doc["summary"].endswith("[summary truncated to its admitted size]")
    sent = await _assemble(_host(), turn, with_estimate=True)
    assert sent.estimate.total_tokens <= admitted[0].total_tokens
    assert sent.estimate.fits


# --------------------------------------------------------------------------- the pinned hashes

class TestThePinnedHashesDescribeWhatTheTurnActuallyRan:
    """[f11] These two hashes are pure attribution: they change nothing about the request, and they
    are the only way to tell a stream built against a different bot from one built against a
    different channel. A hash that names the wrong things is worse than no hash — it reports forks
    that are not forks (a requester's temperature) and misses the ones that are (the sandbox).
    """

    def test_the_profile_is_the_channels_own_capability_keys(self):
        from config import CHANNEL_CAPABILITY_KEYS
        keys = channel_request._CAPABILITY_PROFILE_KEYS
        # Spec §3b's list, in full — including the two that were missing.
        assert set(CHANNEL_CAPABILITY_KEYS) <= set(keys)
        assert "enable_mcp" in keys and "image_model" in keys

    def test_the_requesters_own_settings_are_not_a_capability_fork(self):
        """On a channel turn these still come from whoever spoke, so hashing them made every
        speaker a different capability profile — the exact fork the pin exists to detect."""
        for key in ("temperature", "top_p", "enable_streaming"):
            assert key not in channel_request._CAPABILITY_PROFILE_KEYS
        assert (channel_request.capability_profile_hash(
                    thread_config(temperature=0.2, top_p=0.5, enable_streaming=True))
                == channel_request.capability_profile_hash(
                    thread_config(temperature=1.0, top_p=1.0, enable_streaming=False)))

    @pytest.mark.parametrize("key,other", [
        ("model", "gpt-5.5"), ("reasoning_effort", "xhigh"), ("verbosity", "low"),
        ("enable_web_search", True), ("enable_mcp", True), ("image_model", "gpt-image-1"),
        ("enable_code_interpreter", True),
    ])
    def test_every_channel_capability_moves_the_hash(self, key, other):
        base = thread_config(**{k: v for k, v in (
            ("enable_mcp", False), ("image_model", "gpt-image-2"))})
        assert (channel_request.capability_profile_hash(base)
                != channel_request.capability_profile_hash(thread_config(
                    **{**{"enable_mcp": False, "image_model": "gpt-image-2"}, key: other})))

    def test_the_tool_digest_covers_the_hosted_tools_and_not_only_the_registry(self):
        """`_build_tools_array` grows web search, the sandbox and the MCP servers onto the array
        after the registry has had its say. A digest of registry schemas alone claimed to identify a
        tool set it had only seen part of."""
        off = dict(enable_web_search=False, enable_code_interpreter=False, enable_mcp=False)
        base = channel_request.tool_schema_version(None, thread_config(**off))
        for key in ("enable_web_search", "enable_code_interpreter", "enable_mcp"):
            assert channel_request.tool_schema_version(
                None, thread_config(**{**off, key: True})) != base, key

    def test_the_digest_names_each_hosted_tools_documented_fork(self):
        """Two of the four documented cache-fork exceptions live on these tools (spec §3a). The
        digest says so rather than pretending the schemas are the whole story — while the container
        id and the excluded label themselves stay out, being per-attempt."""
        entries = channel_request.hosted_tool_digest_entries(
            thread_config(enable_web_search=True, enable_code_interpreter=True, enable_mcp=True),
            mcp_manager=SimpleNamespace(get_server_labels=lambda: ["reportpro", "datassential"]))
        assert entries == ["hosted:web_search",
                           "hosted:code_interpreter fork=container_id",
                           "hosted:mcp fork=mcp_exclusion servers=datassential,reportpro"]

    def test_the_server_list_moves_the_digest_and_a_broken_manager_never_fails_the_turn(self):
        cfg = thread_config(enable_mcp=True, enable_web_search=False,
                            enable_code_interpreter=False)
        one = channel_request.tool_schema_version(
            None, cfg, mcp_manager=SimpleNamespace(get_server_labels=lambda: ["reportpro"]))
        two = channel_request.tool_schema_version(
            None, cfg, mcp_manager=SimpleNamespace(get_server_labels=lambda: ["reportpro", "x"]))
        assert one != two

        def _explode():
            raise RuntimeError("mcp config never loaded")

        assert channel_request.tool_schema_version(
            None, cfg, mcp_manager=SimpleNamespace(get_server_labels=_explode))

    def test_a_local_schema_change_still_moves_the_digest(self):
        cfg = thread_config(enable_web_search=False, enable_code_interpreter=False,
                            enable_mcp=False)
        one = SimpleNamespace(schemas=lambda c, surface=None: [{"name": "react_to_message"}])
        two = SimpleNamespace(schemas=lambda c, surface=None: [{"name": "react_to_message",
                                                               "strict": True}])
        assert (channel_request.tool_schema_version(one, cfg)
                != channel_request.tool_schema_version(two, cfg))


def test_actor_recency_reads_the_shared_numeric_comparator():
    """[nit23] One parser, one comparator. A float parse scored an unreadable ts as 0.0 and
    attributed someone's newest message to their oldest; a lexical compare calls 9.0 newer than
    10.0. Both are silent, and the roster is what tells the model who is in the room."""
    from message_processor.channel_stream import StreamTimestampError

    stream = build_stream([normalized("9.0", "older"), normalized("10.0", "newer")])
    actors = {a.user_id: a for a in channel_request._stream_actors(stream)}
    assert actors["U1"].last_ts == "10.0"

    # And a record whose ts cannot be parsed fails the turn CLOSED, in the class base.py turns into
    # an honest notice — rather than sorting to the bottom of the roster and saying nothing.
    smuggled = SimpleNamespace(pinned=SimpleNamespace(
        actor_names={"U1": "Alice"},
        fetch_snapshot=(normalized("1.0", "fine"), normalized("not-a-ts", "broken"))))
    with pytest.raises(StreamTimestampError):
        channel_request._stream_actors(smuggled)


class TestTheRequestNamesTheModelItIsSentTo:
    """[r2-7] With WEB_SEARCH_MODEL configured, a turn that has web search on is sent to THAT model
    — both text handlers resolve it that way before assembly. Everything that names the model has to
    agree: otherwise the prompt tells the model the wrong name, cutoff and context window, and
    telemetry files the stream under a capability profile it was never run at."""

    def test_the_helper_resolves_the_same_way_the_handlers_do(self, monkeypatch):
        from message_processor.utilities import effective_request_model

        monkeypatch.setattr(config, "web_search_model", "gpt-5.6-terra")
        assert effective_request_model(
            thread_config(model="gpt-5.6-sol", enable_web_search=True)) == "gpt-5.6-terra"
        assert effective_request_model(
            thread_config(model="gpt-5.6-sol", enable_web_search=False)) == "gpt-5.6-sol"
        monkeypatch.setattr(config, "web_search_model", "")
        assert effective_request_model(
            thread_config(model="gpt-5.6-sol", enable_web_search=True)) == "gpt-5.6-sol"

    @pytest.mark.asyncio
    async def test_the_capability_suffix_names_the_effective_model(self, monkeypatch):
        monkeypatch.setattr(config, "web_search_model", "gpt-5.5")
        turn = TurnRuntime()
        pin_channel_turn(turn, prepared=no_tools_prepared(),
                         config=thread_config(model="gpt-5.6-sol", enable_web_search=True))
        blob = "\n".join(item_texts(to_input_items(await _assemble(_host(), turn))))
        assert "model: gpt-5.5" in blob
        assert "model: gpt-5.6-sol" not in blob
        # And the window quoted alongside it is that model's, from the accounting's own resolver.
        assert f"{config.get_model_token_limit('gpt-5.5'):,}" in blob

    def test_the_profile_hash_follows_the_effective_model(self, monkeypatch):
        searching = thread_config(model="gpt-5.6-sol", enable_web_search=True)
        monkeypatch.setattr(config, "web_search_model", "")
        stored_only = channel_request.capability_profile_hash(searching)
        monkeypatch.setattr(config, "web_search_model", "gpt-5.5")
        assert channel_request.capability_profile_hash(searching) != stored_only
        # It is the same capability profile as a thread already set to that model — which is the
        # point: the two runs really are the same model, window and cutoff.
        assert (channel_request.capability_profile_hash(searching)
                == channel_request.capability_profile_hash(
                    thread_config(model="gpt-5.5", enable_web_search=True)))

    def test_a_turn_with_search_off_is_unaffected_by_the_search_model(self, monkeypatch):
        off = thread_config(model="gpt-5.6-sol", enable_web_search=False)
        monkeypatch.setattr(config, "web_search_model", "")
        plain = channel_request.capability_profile_hash(off)
        monkeypatch.setattr(config, "web_search_model", "gpt-5.5")
        assert channel_request.capability_profile_hash(off) == plain

    def test_every_request_site_calls_the_resolver_instead_of_copying_it(self):
        """[r3-12] The three sites that pick the model to SEND to used to each restate the
        expression. They agreed, which is exactly why the drift would have been silent."""
        import inspect

        import message_processor.base as base
        import message_processor.handlers.text as th

        for module in (base, th):
            src = inspect.getsource(module)
            assert "config.web_search_model or thread_config" not in src, module.__name__
            assert "effective_request_model(thread_config)" in src, module.__name__


# --------------------------------------------------------------------------- retries

@pytest.mark.asyncio
async def test_a_retry_reassembles_from_the_pins_and_only_the_model_changes():
    """Retries reuse ALL pinned state. A rebuild would produce a different window, and the retry
    would then answer a question nobody asked."""
    host = _host()
    turn = TurnRuntime()
    pin_channel_turn(turn, prepared=no_tools_prepared(),
                     messages=[normalized("1.0", "the only history there is")])
    first = await _assemble(host, turn)
    again = await _assemble(host, turn)
    assert first.evidence_hash == again.evidence_hash
    assert to_input_items(first) == to_input_items(again)

    # A 5.5 fork loses the explicit breakpoint (unsupported there) and nothing else.
    on_55 = await _assemble(host, turn, model="gpt-5.5")
    assert _breakpoint_index(to_input_items(on_55)) == -1
    assert on_55.evidence_hash == first.evidence_hash


@pytest.mark.asyncio
async def test_the_turns_tool_exposure_is_resolved_once_and_reused():
    host = _host()
    host._prepare_channel_turn_tools = AsyncMock(return_value=no_tools_prepared())
    turn = TurnRuntime()
    pin_channel_turn(turn)
    await _assemble(host, turn)
    await _assemble(host, turn)
    host._prepare_channel_turn_tools.assert_awaited_once()


# --------------------------------------------------------------------------- the tripwire

@pytest.mark.asyncio
async def test_a_channel_turn_never_mutates_thread_state_messages():
    """THE tripwire. A channel turn's transcript is Slack. Anything appended here is invisible to
    the request and would be replayed as history the next time a DM-shaped code path read it."""
    from message_processor.base import MessageProcessor
    from base_client import Response

    with patch("message_processor.base.AsyncThreadStateManager"), \
         patch("message_processor.base.OpenAIClient"):
        processor = MessageProcessor()

    class _State(SimpleNamespace):
        def add_message(self, role, content, metadata=None, **kw):
            self.messages.append({"role": role, "content": content})

    state = _State(messages=[], thread_ts="10.0", channel_id="C1", had_timeout=False,
                   root_author=("U1", "human"), config_overrides={}, participants={},
                   current_model=None, has_trimmed_messages=False)
    processor.db = MagicMock()
    processor.db.get_channel_memory_async = AsyncMock(return_value=[])
    processor.db.get_channel_policy_async = AsyncMock(return_value=None)
    processor.thread_manager.acquire_thread_lock = AsyncMock(return_value=True)
    processor.thread_manager.release_thread_lock = AsyncMock()
    processor.get_or_create_channel_thread_state = AsyncMock(return_value=state)
    processor._build_channel_turn_stream = AsyncMock(
        return_value=build_stream([normalized("10.0", "hi")]))
    processor._process_attachments = AsyncMock(
        return_value=([], [], [{"name": "broken.pdf", "error": "download_failed"}]))
    processor._admit_channel_request = AsyncMock()
    processor._handle_text_response = AsyncMock(
        return_value=Response(type="text", content="answered"))
    client = MagicMock()
    client.self_team_id = "T1"
    client.send_message = AsyncMock(return_value="11.0")

    turn = TurnRuntime()
    await processor.process_message(_message(), client, None, turn=turn)

    # A failed attachment, a delivered reply, a whole turn — and the list is still empty.
    assert state.messages == []


@pytest.mark.asyncio
async def test_the_assembler_refuses_a_channel_turn_with_no_pins():
    """Fail loudly rather than invent a window. A channel turn that reached here without its
    stream is a bug in the ordering, and answering from nothing would hide it."""
    with pytest.raises(RuntimeError, match="no pinned context"):
        await _assemble(_host(), TurnRuntime())


def test_the_retired_channel_helpers_are_gone():
    from message_processor.utilities import MessageUtilitiesMixin
    for name in ("_build_pulse_envelope", "_build_channel_summary_block",
                 "_build_channel_people_line", "_build_taggable_speakers_block",
                 "_merge_gate_cohort"):
        assert not hasattr(MessageUtilitiesMixin, name), name


# --------------------------------------------------------------------------- H + fail-closed

@pytest.mark.asyncio
async def test_h_is_pinned_at_admission_and_carried_on_the_turn():
    from message_processor.base import MessageProcessor
    from slack_client import admission_watermark

    with patch("message_processor.base.AsyncThreadStateManager"), \
         patch("message_processor.base.OpenAIClient"):
        processor = MessageProcessor()
    processor.db = None
    processor.thread_manager.acquire_thread_lock = AsyncMock(return_value=True)
    processor.thread_manager.release_thread_lock = AsyncMock()
    turn = TurnRuntime()

    class _Stop(Exception):
        pass

    processor.get_or_create_channel_thread_state = AsyncMock(side_effect=_Stop())
    admission_watermark.observe("C1", "50.0")
    with patch.object(config, "enable_channel_memory", False):
        await processor.process_message(_message(ts="10.0"), MagicMock(), None, turn=turn)
    # H is the MAX of the channel's watermark and this trigger — never just the trigger.
    assert turn.H == "50.0"


@pytest.mark.asyncio
async def test_a_malformed_trigger_timestamp_fails_closed_and_releases_the_lock():
    """[r3-24] A bad ts anywhere on the turn path fails the turn. It must not also wedge the
    conversation: the pin sits inside the try whose finally releases the lock."""
    from message_processor.base import MessageProcessor

    with patch("message_processor.base.AsyncThreadStateManager"), \
         patch("message_processor.base.OpenAIClient"):
        processor = MessageProcessor()
    processor.db = None
    processor.thread_manager.acquire_thread_lock = AsyncMock(return_value=True)
    processor.thread_manager.release_thread_lock = AsyncMock()
    message = Message(text="hi", user_id="U1", channel_id="CBAD", thread_id="not-a-ts",
                      metadata={"ts": "not-a-ts"})
    response = await processor.process_message(message, MagicMock(), None, turn=TurnRuntime())
    assert response.type == "error"
    processor.thread_manager.release_thread_lock.assert_awaited()


@pytest.mark.parametrize("error_name,code,needle", [
    ("CoverageNotReady", "coverage_not_ready", "Still Catching Up"),
    ("SnapshotUnsupportedError", "snapshot_unsupported", "Can't Read This Channel"),
    ("StreamOverBudgetError", "stream_over_budget", "Too Much For One Request"),
])
def test_each_fail_closed_condition_gets_its_own_honest_notice(error_name, code, needle):
    """Three states the runtime can be in, three different things the user can do about it. A
    shared 'something went wrong' would withhold the only actionable part."""
    from message_processor import channel_stream
    from message_processor.base import MessageProcessor

    resolved, notice = MessageProcessor._channel_stream_failure(
        getattr(channel_stream, error_name)("boom"))
    assert resolved == code
    assert needle in notice["message"]
    assert notice["status"] and not notice["status"].startswith("Something")


@pytest.mark.asyncio
async def test_a_mounted_spreadsheet_is_charged_by_its_bytes():
    """A CSV/XLSX rides the turn as a native input_file so it auto-mounts in the sandbox (F32).
    It has no page count, so the bound is one token per byte — the true worst case for
    high-entropy text — and a 50k-row CSV is therefore charged what it could actually cost rather
    than what its extracted preview does."""
    turn = TurnRuntime()
    pin_channel_turn(
        turn, prepared=no_tools_prepared(),
        document_inputs=[{"filename": "rows.csv", "mimetype": "text/csv",
                          "content": "a,b\n1,2\n", "summary": "two columns",
                          "native": True, "file_data_b64": "QUJD", "size_bytes": 120_000,
                          "total_pages": None}])
    request = await _assemble(_host(), turn, with_estimate=True)
    assert request.estimate.breakdown["native_files"] == 120_000
    # …and the same file WITHOUT the sandbox is not a native part at all, so nothing is charged.
    plain = TurnRuntime()
    pin_channel_turn(
        plain, prepared=no_tools_prepared(),
        document_inputs=[{"filename": "rows.csv", "mimetype": "text/csv",
                          "content": "a,b\n1,2\n", "summary": "two columns",
                          "native": False, "file_data_b64": None, "size_bytes": 120_000}])
    assert (await _assemble(_host(), plain, with_estimate=True)
            ).estimate.breakdown["native_files"] == 0


@pytest.mark.asyncio
async def test_the_metadata_the_lease_reads_never_reaches_the_api():
    """The stale guard reads `metadata.ts` off role:user items, so the assembler has to carry it —
    and the channel layout builder has to strip it, or every turn is a 400."""
    from openai_client.base import _build_request_params

    turn = TurnRuntime()
    pin_channel_turn(turn, prepared=no_tools_prepared(),
                     messages=[normalized("1.0", "a"), normalized("2.0", "b", sender_id="U2")])
    items = to_input_items(await _assemble(_host(), turn))
    assert any(i.get("metadata", {}).get("ts") for i in items)

    params = _build_request_params(model="gpt-5.6-sol", input_items=items,
                                  system_prompt="SYSTEM", layout="channel")
    assert all("metadata" not in sent for sent in params["input"])
    assert params["instructions"] == "SYSTEM"


# --------------------------------------------------------------------------- origin side state

@pytest.mark.asyncio
async def test_the_origin_slice_registers_its_files_and_people_idempotently():
    """Step 10. A channel turn has no rebuild, so nothing walks the thread writing the rows the
    tools address as side effects. An image with no `images` row is not editable and a document
    with no `documents` row is not readable — and because this runs on EVERY turn in the thread, a
    plain insert would add a row per turn per file forever."""
    from message_processor.base import MessageProcessor

    with patch("message_processor.base.AsyncThreadStateManager"), \
         patch("message_processor.base.OpenAIClient"):
        processor = MessageProcessor()
    db = MagicMock()
    db.find_thread_images_async = AsyncMock(return_value=[])
    db.save_image_metadata_async = AsyncMock()
    db.save_document_if_absent_async = AsyncMock(return_value=True)
    processor.db = db
    processor.document_handler = MagicMock(is_document_file=MagicMock(return_value=True))

    slice_messages = [
        normalized("1.0", "the deploy failed", sender_id="U2"),
        normalized("2.0", "log attached", sender_id="U3", thread_root_ts="1.0",
                   files=[file_ref("F7", "deploy.log", "text/plain"),
                          file_ref("F8", "graph.png", "image/png", kind="image")]),
    ]
    state = SimpleNamespace(channel_id="C1", thread_ts="1.0", participants={}, root_author=None)
    await processor.ingest_channel_origin_slice(
        state, slice_messages, db=db, actor_names={"U2": "Bob", "U3": "Cleo"})

    # The thread's ROOT author, for the coordinates block's "is the asker the one who started it".
    assert state.root_author == ("U2", "human")
    assert state.participants == {"U2": "Bob", "U3": "Cleo"}
    # Idempotent by construction: the document write is the ON CONFLICT DO NOTHING one, and the
    # image write is skipped when a row for that url already exists.
    db.save_document_if_absent_async.assert_awaited_once()
    assert db.save_document_if_absent_async.await_args.args[1] == "deploy.log"
    db.save_image_metadata_async.assert_awaited_once()

    db.find_thread_images_async = AsyncMock(
        return_value=[{"url": "https://files.slack.com/files-pri/T1-F1/report.pdf"}])
    db.save_image_metadata_async.reset_mock()
    await processor.ingest_channel_origin_slice(
        state, [normalized("3.0", "again", files=[
            file_ref("F8", "graph.png", "image/png",
                     url="https://files.slack.com/files-pri/T1-F1/report.pdf", kind="image")])],
        db=db)
    db.save_image_metadata_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_file_the_ingester_cannot_fetch_is_skipped_not_fatal():
    """A side-state write that fails costs one tool its target. It must never cost the turn."""
    from message_processor.base import MessageProcessor

    with patch("message_processor.base.AsyncThreadStateManager"), \
         patch("message_processor.base.OpenAIClient"):
        processor = MessageProcessor()
    db = MagicMock()
    db.find_thread_images_async = AsyncMock(side_effect=RuntimeError("db is gone"))
    db.save_document_if_absent_async = AsyncMock(side_effect=RuntimeError("db is gone"))
    processor.db = db
    processor.document_handler = MagicMock(is_document_file=MagicMock(return_value=True))
    state = SimpleNamespace(channel_id="C1", thread_ts="1.0", participants={}, root_author=None)
    await processor.ingest_channel_origin_slice(
        state, [normalized("1.0", "x", files=[file_ref(), file_ref("F2", "a.png", "image/png",
                                                                  kind="image")])], db=db)
    assert state.root_author == ("U1", "human")     # the turn still learned what it could


@pytest.mark.asyncio
async def test_the_tool_context_actually_carries_the_canonical_file_catalog():
    """A catalog nobody is handed is a catalog that does nothing. The wire from the pinned window
    to the executors is the whole feature, and it is invisible when it breaks: `read_document`
    simply reports the file as not found, exactly as it did before the catalog existed."""
    from message_processor.handlers.text import TextHandlerMixin

    host = MagicMock()
    host._build_tool_context = TextHandlerMixin._build_tool_context.__get__(host)
    host.db = None
    turn = TurnRuntime()
    pin_channel_turn(turn, prepared=no_tools_prepared(),
                     messages=[normalized("1.0", "here", files=[file_ref("F1", "runbook.pdf")])])
    ctx = host._build_tool_context(_message(), MagicMock(), {}, turn=turn)
    assert list(ctx.canonical_files) == ["F1"]

    # A DM turn has no window and must carry nothing — an empty catalog authorizes nothing.
    assert host._build_tool_context(_message(), MagicMock(), {},
                                    turn=TurnRuntime()).canonical_files == {}


@pytest.mark.asyncio
async def test_a_committed_reply_is_what_channel_memory_reads():
    """Memory is written from what the room SAW. A turn that produced words and then failed to
    deliver them has no exchange to remember, and recording one would have the bot remember a
    conversation that never happened."""
    import main as main_module

    app = SimpleNamespace(
        processor=MagicMock(), snapshot_coordinator=None,
        _schedule_channel_memory=None)
    app._schedule_channel_memory = main_module.ChatBotV2._schedule_channel_memory.__get__(app)
    scheduled = []
    app.processor._schedule_async_call = MagicMock(
        side_effect=lambda coro: scheduled.append(coro) or coro.close())
    app.processor.extract_channel_memory_from_exchange = MagicMock(return_value=MagicMock(
        close=MagicMock()))

    turn = TurnRuntime()
    turn.stream_build_present = True
    with patch.object(config, "enable_memory_extraction_fallback", True):
        # Observed but never committed — an interrupted stream. Nothing to remember.
        turn.note_destination_observed(channel_id="C1", first_ts="11.0", kind="stream")
        app._schedule_channel_memory(_message(), turn)
        assert not scheduled

        turn.mark_destination_committed(first_ts="11.0", kind="stream", text="the answer",
                                       channel_id="C1")
        app._schedule_channel_memory(_message(), turn)
        assert len(scheduled) == 1
    args = app.processor.extract_channel_memory_from_exchange.call_args.args
    assert args[0] == "C1" and args[1] == "hi" and args[2] == "the answer"


def test_a_destination_record_reports_chars_and_never_the_reply_itself():
    turn = TurnRuntime()
    turn.mark_destination_committed(first_ts="11.0", kind="reply", text="hello there",
                                    channel_id="C1", thread_root_ts="10.0")
    payload = turn.destinations[0].as_payload()
    assert payload == {"channel_id": "C1", "thread_root_ts": "10.0", "first_ts": "11.0",
                       "state": "committed", "chars": 11, "kind": "reply"}
    assert "text" not in payload


def test_one_surface_is_one_record_however_many_times_it_flushes():
    turn = TurnRuntime()
    for _ in range(50):
        turn.note_destination_observed(channel_id="C1", first_ts="11.0", kind="stream")
    assert len(turn.destinations) == 1
    # …and a stream that turns out to be multipart REFINES that record rather than adding one.
    turn.mark_destination_committed(first_ts="11.0", kind="split", text="x", channel_id="C1")
    assert len(turn.destinations) == 1 and turn.destinations[0].kind == "split"
