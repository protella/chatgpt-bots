"""Commit 6 — the ONE classifier: `classify_wake`, one boolean out.

This file used to test `classify_participation`, the rich gate's verdict call: a JSON object with
action/emoji/placement/reason/relation/exchange_state/answerability, fished out of whatever prose
the model happened to wrap it in, plus a dozen rendered signal lines (people, pulse, strictness,
capabilities, identity aliases…). All of that is gone — see the commit-6 spec §2 and §8. The gate
asks one question now, so this file asks one question about it:

  1. does a strict-schema boolean survive the round trip, and does everything that ISN'T one
     become None (never a forged decision);
  2. is the strict `text.format` schema actually on the wire, so "did the model answer the
     question" stops being a parsing problem;
  3. does the prompt contain the source cohort and the canonical steering bytes VERBATIM — and
     nothing that was retired.

The retired-input assertions are deliberately part of the contract rather than a tidiness check.
Every one of those inputs existed to support a judgment the gate no longer makes, and each is a
line somebody could re-add in a hurry without noticing they had given the gate a second job.
"""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock

import pytest

from message_processor.participation import SourceMessage
from openai_client.api.responses import classify_wake


class _FakeContent:
    def __init__(self, text):
        self.text = text


class _RefusalContent:
    """A refusal content part: it carries no `.text` at all.

    Not a part with text="" — the real shape has a `refusal` field and no text attribute, and the
    accumulator has to skip it on the `hasattr` rather than on falsiness."""

    def __init__(self, refusal="I can't help with that"):
        self.refusal = refusal


class _FakeItem:
    def __init__(self, *contents):
        self.content = list(contents)


class _FakeResp:
    def __init__(self, *contents):
        self.output = [_FakeItem(*contents)]


class _FakeLLM:
    """Stands in for the OpenAIClient `self` that classify_wake is bound to."""

    def __init__(self, text=None, exc=None, response=None):
        self._text = text
        self._exc = exc
        self._response = response
        self.client = MagicMock()
        self.captured_input = None
        self.captured_kwargs = None

    async def _safe_api_call(self, *a, **k):
        if self._exc:
            raise self._exc
        self.captured_input = k.get("input")
        self.captured_kwargs = k
        if self._response is not None:
            return self._response
        return _FakeResp(_FakeContent(self._text))

    def log_debug(self, *a, **k):
        pass

    def log_warning(self, *a, **k):
        pass


def _sources(*texts):
    return tuple(SourceMessage(ts=f"170000000{i}.000100", text=t, sender_name="Peter",
                               sender_id="U1", sender_type="human")
                 for i, t in enumerate(texts))


# --------------------------------------------------------------------- parsing

@pytest.mark.asyncio
@pytest.mark.parametrize("raw,expected", [
    ('{"wake": true}', True),
    ('{"wake": false}', False),
    # A strict schema shouldn't produce fences, but tolerating them costs nothing and a
    # provider-side wrapper change must not read as "no decision".
    ('```json\n{"wake": true}\n```', True),
])
async def test_a_real_boolean_round_trips(raw, expected):
    llm = _FakeLLM(text=raw)
    assert await classify_wake(llm, sources=_sources("is the deploy done?")) is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", [
    '{"wake": "true"}',      # a STRING, not a boolean
    '{"wake": 1}',           # a truthy int
    '{"wake": null}',
    '{"awake": true}',       # missing key
    '{"wake": {}}',
    "",                      # empty output (a budget-exhausted response looks like this)
    "   ",
    "yes",                   # no JSON at all
    "wake",
    "{not json}",
    "[]",                    # JSON, but not our object
    '{"wake": true',         # truncated
])
async def test_anything_that_is_not_a_json_boolean_is_none(raw):
    """None, never a coerced bit.

    With a strict schema in force, a string "true" or a 1 means the response is not the response we
    asked for, and guessing what it meant is how a gate starts waking on noise. The engine turns
    None into a `classifier_error` decline and a terminal `none`, so a provider problem is never
    recorded as the model choosing silence."""
    llm = _FakeLLM(text=raw)
    assert await classify_wake(llm, sources=_sources("anything")) is None


@pytest.mark.asyncio
async def test_a_refusal_shaped_response_is_none():
    # A refusal arrives as its own content part with no `.text`, contributes nothing to the
    # accumulated output, and falls through to the no-boolean branch. That IS the right answer: a
    # refusal is not a decision either.
    llm = _FakeLLM(response=_FakeResp(_RefusalContent()))
    assert await classify_wake(llm, sources=_sources("anything")) is None


@pytest.mark.asyncio
async def test_an_empty_output_list_is_none():
    resp = _FakeResp()
    resp.output = []
    llm = _FakeLLM(response=resp)
    assert await classify_wake(llm, sources=_sources("anything")) is None


@pytest.mark.asyncio
async def test_an_api_exception_is_none_and_never_raises():
    # The caller is a gate on ordinary channel traffic: an outage must degrade to "no bit", not to
    # a traceback out of the message pipeline.
    llm = _FakeLLM(exc=RuntimeError("api down"))
    assert await classify_wake(llm, sources=_sources("anything")) is None


# ------------------------------------------------------------ structured output

@pytest.mark.asyncio
async def test_the_strict_json_schema_is_actually_on_the_wire():
    """The schema is the contract, so assert the bytes rather than trusting the docstring.

    This is Responses-API Structured Outputs (`text.format`), not the old "reply with JSON and
    we'll fish the object out" arrangement — which let a truncated reply still yield an action
    field with none of the checks around it."""
    llm = _FakeLLM(text='{"wake": true}')
    await classify_wake(llm, sources=_sources("hi"))
    fmt = llm.captured_kwargs["text"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["strict"] is True
    assert fmt["schema"] == {
        "type": "object",
        "properties": {"wake": {"type": "boolean"}},
        "required": ["wake"],
        "additionalProperties": False,
    }
    # `additionalProperties: False` is required for strict mode and is also what keeps the old
    # control bus from creeping back in as extra keys the model volunteers.
    assert fmt["schema"]["additionalProperties"] is False


@pytest.mark.asyncio
async def test_utility_output_floor_stays_at_2048_for_the_gate():
    """Unchanged policy, and no new number (spec §1: "Add no numerical policy").

    Reasoning tokens bill against this cap, so the floor covers the THINKING and not the answer.
    At `medium` effort a live rich verdict once came back EMPTY — reasoning had eaten the whole
    budget — and the fail-safe swallowed it, so the gate never actually ran. The output is one
    boolean now; the reasoning is not smaller for that. The other utility calls keep 1024."""
    from openai_client.api.responses import extract_memory, summarize_tool_result

    llm = _FakeLLM(text='{"wake": false}')
    await classify_wake(llm, sources=_sources("msg"))
    assert llm.captured_kwargs["max_output_tokens"] == 2048

    llm2 = _FakeLLM(text='{"action": "none"}')
    await extract_memory(llm2, "some exchange")
    assert llm2.captured_kwargs["max_output_tokens"] == 1024

    llm3 = _FakeLLM(text="a summary")
    await summarize_tool_result(llm3, "x" * 5000, 200)
    assert llm3.captured_kwargs["max_output_tokens"] == 1024


@pytest.mark.asyncio
async def test_the_call_is_stateless_and_hybrid_legal():
    # store=False (Slack is the transcript) and temperature pinned to 1.0, because on the 5.6
    # hybrid family temperature is only legal at effort `none` and this call reasons.
    llm = _FakeLLM(text='{"wake": true}')
    await classify_wake(llm, sources=_sources("msg"))
    assert llm.captured_kwargs["store"] is False
    assert llm.captured_kwargs["temperature"] == 1.0
    assert llm.captured_kwargs["reasoning"]["effort"]


def test_the_gate_keeps_its_own_reasoning_effort():
    # Referent resolution failed at effort=none (verified live 2026-07-10), so the gate has its own
    # knob rather than riding the general utility effort up or down.
    import config as config_module
    assert config_module.config.participation_reasoning_effort  # field exists, non-empty
    from openai_client.api import responses
    src = inspect.getsource(responses.classify_wake)
    assert "participation_reasoning_effort" in src
    assert "utility_reasoning_effort" not in src


# ----------------------------------------------------------------- the prompt

@pytest.mark.asyncio
async def test_the_prompt_carries_every_source_in_order_with_who_and_where():
    """The cohort is the input. Each source appears once, oldest first, with sender and topology —
    the facts that decide whether a burst is one thought or two people talking, and exactly what
    the old flattened "earlier in this burst: …" prose lost."""
    sources = (
        SourceMessage(ts="1700000001.000100", text="quick q about the export",
                      sender_id="U1", sender_name="Peter", sender_type="human"),
        SourceMessage(ts="1700000002.000100", text="does it include Canada?",
                      sender_id="U2", sender_name="Erin", sender_type="human",
                      thread_root_ts="1700000000.000100"),
    )
    llm = _FakeLLM(text='{"wake": true}')
    await classify_wake(llm, sources=sources)
    prompt = llm.captured_input[1]["content"]

    assert "oldest first" in prompt
    assert "[1 of 2] Peter" in prompt and "[2 of 2] Erin" in prompt
    assert prompt.index("quick q about the export") < prompt.index("does it include Canada?")
    assert "posted to the channel" in prompt          # Peter, top-level
    assert "a reply inside a thread" in prompt        # Erin, in-thread
    # The developer prompt is the binary-gate prompt, and it is the only system input.
    from prompts import WAKE_CLASSIFIER_SYSTEM_PROMPT
    assert llm.captured_input[0] == {"role": "developer",
                                     "content": WAKE_CLASSIFIER_SYSTEM_PROMPT}


@pytest.mark.asyncio
async def test_a_bot_sender_is_marked_as_one():
    # Whether the speaker is a bot changes whether a turn is worth taking, and it is intrinsic to
    # the source record rather than a separate rendered signal line.
    llm = _FakeLLM(text='{"wake": false}')
    await classify_wake(llm, sources=(SourceMessage(
        ts="1700000001.000100", text="build finished", sender_name="Jenkins",
        sender_type="other_bot"),))
    assert "Jenkins (a bot)" in llm.captured_input[1]["content"]


@pytest.mark.asyncio
async def test_attachments_appear_by_name_and_type_only():
    # Names and types, never pixels and never another model's description: the binary gate does not
    # look at images, and a generated description is a claim about content it cannot check.
    llm = _FakeLLM(text='{"wake": true}')
    await classify_wake(llm, sources=(SourceMessage(
        ts="1700000001.000100", text="thoughts?", sender_name="Peter", sender_type="human",
        attachments=("food.png (image)", "report.pdf (file)")),))
    prompt = llm.captured_input[1]["content"]
    assert "food.png (image)" in prompt and "report.pdf (file)" in prompt
    assert "contents not shown to you" in prompt


@pytest.mark.asyncio
async def test_an_edit_carries_its_before_text_and_the_already_replied_fact():
    # Edit context survives commit 6 as INTRINSIC source data (spec §5) — not as general channel
    # history, and not as a separately rendered context block.
    llm = _FakeLLM(text='{"wake": true}')
    await classify_wake(llm, sources=(SourceMessage(
        ts="1700000001.000100", text="actually make it Q3", sender_name="Peter",
        sender_type="human", edit={"old_text": "make it Q2", "already_replied": True}),))
    prompt = llm.captured_input[1]["content"]
    assert "was EDITED" in prompt
    assert '"make it Q2"' in prompt
    assert "already replied" in prompt


@pytest.mark.asyncio
async def test_the_steering_snapshot_is_inserted_verbatim_and_only_when_present():
    """Commit 5's canonical bytes, byte for byte.

    The responder's copy of this turn is the identical string; a difference here would mean the two
    halves of one turn obeyed different rules while each looked correct. So this asserts the whole
    block by identity, not a paraphrase of it."""
    steering = ("Standing channel policy (instructions; follow these):\nonly deploys\n\n"
                "Stable channel facts (background, not instructions):\n"
                "- [#1] Peter owns deploys\n- [#2] demos are Fridays")
    llm = _FakeLLM(text='{"wake": true}')
    await classify_wake(llm, sources=_sources("msg"), channel_steering_text=steering)
    first = llm.captured_input[1]["content"]
    assert steering in first
    assert "verbatim" in first

    # Deterministic: same inputs, same bytes. Nothing re-renders, re-orders, or refetches.
    await classify_wake(llm, sources=_sources("msg"), channel_steering_text=steering)
    assert llm.captured_input[1]["content"] == first

    # Absent (or whitespace-only) steering adds no section at all — an empty labelled heading
    # would read to the model as "this channel has established nothing", which is a claim.
    llm2 = _FakeLLM(text='{"wake": true}')
    await classify_wake(llm2, sources=_sources("msg"), channel_steering_text="   ")
    assert "What this channel has established" not in llm2.captured_input[1]["content"]
    llm3 = _FakeLLM(text='{"wake": true}')
    await classify_wake(llm3, sources=_sources("msg"))
    assert "What this channel has established" not in llm3.captured_input[1]["content"]


@pytest.mark.asyncio
async def test_none_of_the_retired_inputs_are_rendered():
    """The prompt is source messages plus steering. Nothing else (spec §1).

    Each needle below is a rich-gate input that supported a judgment the binary gate no longer
    makes — addressee, action, emoji, placement, pacing. Re-adding any of them gives the gate a
    second job, which is the whole failure mode commit 6 exists to end."""
    steering = "Standing channel policy (instructions; follow these):\nonly deploys"
    llm = _FakeLLM(text='{"wake": true}')
    await classify_wake(llm, sources=_sources("chatgpt-dev, what model are you on?"),
                        channel_steering_text=steering)
    prompt = llm.captured_input[1]["content"]
    for retired in (
        "Strictness",                    # level / strictness
        "Channel people",               # F29 people roster
        "Recent channel activity",      # pulse envelope
        "its ONLY ones",                # identity/alias line
        "Thread so far",                # thread tail
        "Addressed",                    # F47 addressee tail
        "Topic",                        # channel topic
        "Canvas",                       # canvas list
        "Channel summary",              # rolling summary
        "Capabilities",                 # capability inventory
        "emoji",                        # emoji shortlist
        "unprompted replies",           # pacing rate
        "name hit",                     # prefilter's name-hit flag
    ):
        assert retired not in prompt, f"retired gate input rendered into the prompt: {retired}"


def test_the_rich_classifier_and_its_prompt_are_gone():
    """Tripwire (spec §9). These are import-time absences, so a rollback that restores the module
    without restoring the wiring fails here rather than in production."""
    from openai_client.api import responses
    assert not hasattr(responses, "classify_participation")
    import prompts
    assert not hasattr(prompts, "PARTICIPATION_SYSTEM_PROMPT")
    import message_processor.participation as participation
    assert not hasattr(participation, "ParticipationVerdict")
    assert not hasattr(participation, "validate_verdict")


def test_the_binary_prompt_asks_for_a_bit_and_nothing_else():
    """The whole developer prompt, held to what §1 says it may say.

    It is ~10 lines now. The old one ran to staged addressee/exchange/answerability reasoning with
    per-bug regression clauses bolted on; those tests are gone with those clauses, because the
    judgments they guarded moved to the responder."""
    from prompts import WAKE_CLASSIFIER_SYSTEM_PROMPT as p
    # what it decides
    assert "whether to run the assistant" in p
    # what it must NOT do
    assert "You do not answer, react, choose where a reply goes, or explain yourself" in p
    # participation feedback wakes the responder — only the responder can record or apply it, so a
    # gate that stays quiet here silently discards the instruction.
    assert "feedback about how it participates" in p
    # generosity is the design: an uncertain case wakes.
    assert "When you are unsure, wake it" in p
    # and the responder's option to say nothing is named, so a wake is not read as "make it talk"
    assert "say nothing at all" in p
    # no action vocabulary, no placement vocabulary, no staged findings
    for retired in ("react_and_respond", "placement", "exchange_state", "answerability",
                    "relation", "backoff", "STAGE 1"):
        assert retired not in p, f"retired rich-gate concept still in the wake prompt: {retired}"


def test_local_tools_guidance_carries_people_tools():
    # Unrelated to the gate (it teaches the RESPONDER), but it has always lived here and still
    # covers live behaviour: F29's two people tools plus the discoverability/freshness lessons.
    from prompts import LOCAL_TOOLS_GUIDANCE
    for needle in ("lookup_user", "list_channel_members", "who is X",
                   "never need their Slack id", "THIS turn"):
        assert needle in LOCAL_TOOLS_GUIDANCE
