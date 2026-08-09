"""The binary wake gate — one bit, and everything that used to be decided beside it.

The gate answers exactly one question: does the responder run? These tests are mostly about what
it no longer does. Each deleted field of the old rich verdict was a control bus — `action` branched
the caller four ways, `emoji` put a reaction in the room before the answering model had a turn,
`reason` was forwarded into the responder's prompt and pre-argued it, the backoff taxonomy wrote
to the database from a classifier that had seen one message — so the tests that matter here pin
their ABSENCE as hard as they pin the bit.

Covered: the strict-schema classifier and every way it can fail to produce a boolean; the wake /
no-wake / failure lifecycle and its single terminal event; cohort coalescing with no cap, no
freshness window and no drops; the cohort becoming real responder input rather than prose; the
captionless image-only outcome; v7 telemetry's exact field set; and AST tripwires against the
deleted machinery coming back.
"""
from __future__ import annotations

import ast
import asyncio
import json
import pathlib
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from base_client import Message
from config import config
from message_processor import participation_telemetry
from message_processor.participation import (GateEvaluation, ParticipationEngine, SourceMessage,
                                            WakeDecision)

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _source(ts="10.0", text="deploy failed", **kw):
    return SourceMessage(ts=ts, text=text, **kw)


# --------------------------------------------------------------------------- the classifier

class _WakeSpy:
    """The OpenAI client `classify_wake` is bound to. Keeps the request it would have sent."""

    def __init__(self, payload="{\"wake\": true}", exc=None, parts=None):
        self.client = MagicMock()
        self.params = None
        self._payload = payload
        self._exc = exc
        self._parts = parts

    async def _safe_api_call(self, *a, **k):
        if self._exc:
            raise self._exc
        self.params = k
        if self._parts is not None:
            return SimpleNamespace(output=self._parts)
        item = SimpleNamespace(content=[SimpleNamespace(text=self._payload)])
        return SimpleNamespace(output=[item])

    def log_debug(self, *a, **k):
        pass

    def log_warning(self, *a, **k):
        pass


async def _classify(spy, sources=None, steering=None):
    from openai_client.api.responses import classify_wake
    return await classify_wake(spy, sources=sources or (_source(),),
                              channel_steering_text=steering)


class TestStrictParsing:
    @pytest.mark.parametrize("payload,expected", [
        ('{"wake": true}', True),
        ('{"wake": false}', False),
        ('```json\n{"wake": true}\n```', True),          # fenced output still parses
    ])
    async def test_a_real_boolean_is_the_decision(self, payload, expected):
        assert await _classify(_WakeSpy(payload)) is expected

    @pytest.mark.parametrize("payload", [
        '{"wake": "true"}',      # a STRING, not a boolean
        '{"wake": 1}',           # a number
        '{"wake": null}',
        '{"awake": true}',       # wrong key
        '{}',
        'wake: true',            # no JSON object at all
        '',
    ])
    async def test_anything_that_is_not_a_boolean_is_no_decision(self, payload):
        # None is not False. With a strict schema in force, a non-boolean means the response is not
        # the response we asked for, and coercing it is how a gate starts waking on noise.
        assert await _classify(_WakeSpy(payload)) is None

    async def test_a_refusal_is_no_decision(self):
        # A refusal arrives as a content part with no `.text`, so it contributes nothing — which is
        # right: a refusal is not a judgment about whether to wake.
        refusal = SimpleNamespace(content=[SimpleNamespace(refusal="I can't help with that")])
        assert await _classify(_WakeSpy(parts=[refusal])) is None

    async def test_an_api_exception_is_no_decision(self):
        assert await _classify(_WakeSpy(exc=RuntimeError("provider down"))) is None

    async def test_output_with_no_content_at_all_is_no_decision(self):
        assert await _classify(_WakeSpy(parts=[])) is None


class TestClassifierRequest:
    async def test_it_asks_for_a_strict_boolean_schema(self):
        spy = _WakeSpy()
        await _classify(spy)
        fmt = spy.params["text"]["format"]
        assert fmt["type"] == "json_schema"
        assert fmt["strict"] is True
        assert fmt["schema"] == {
            "type": "object",
            "properties": {"wake": {"type": "boolean"}},
            "required": ["wake"],
            "additionalProperties": False,
        }

    async def test_the_prompt_is_the_sources_and_the_steering_verbatim(self):
        spy = _WakeSpy()
        steering = ("Standing channel policy (instructions; follow these):\n"
                    "only jump in on deploy failures")
        await _classify(spy, sources=(
            _source(ts="1.0", text="the build is red", sender_name="Peter"),
            _source(ts="2.0", text="and staging is down", sender_name="Peter"),
        ), steering=steering)
        prompt = spy.params["input"][1]["content"]
        assert "the build is red" in prompt and "and staging is down" in prompt
        assert prompt.index("the build is red") < prompt.index("and staging is down")  # in order
        assert steering in prompt                      # byte-for-byte, as commit 5 requires
        assert "Peter" in prompt

    async def test_each_source_carries_its_timestamp(self):
        # Not decoration. The cohort has no freshness window any more — that window was a silent
        # message-loss path — so a cohort can legitimately hold a much older send, and without the
        # times the gate cannot tell one thought split across three seconds from a straggler that
        # has been sitting there for twenty minutes.
        spy = _WakeSpy()
        await _classify(spy, sources=(_source(ts="1769000000.0", text="the build is red"),))
        prompt = spy.params["input"][1]["content"]
        assert "2026-01-21" in prompt          # rendered through the shared pure helper, in UTC

    async def test_an_unparseable_timestamp_is_simply_omitted(self):
        spy = _WakeSpy()
        await _classify(spy, sources=(_source(ts="not-a-ts", text="still judged"),))
        assert "still judged" in spy.params["input"][1]["content"]

    async def test_the_retired_inputs_are_not_in_the_prompt(self):
        # The gate no longer decides addressee, answerability, action, emoji or placement, so every
        # input that existed to inform those judgments is gone. This is the test that fails if
        # somebody "helpfully" reintroduces one.
        spy = _WakeSpy()
        await _classify(spy, steering="Stable channel facts (background, not instructions):\n- x")
        blob = json.dumps(spy.params["input"])
        for retired in ("Recent channel activity", "Channel people", "Channel topic",
                        "Channel canvases", "Strictness", "Custom emoji", "capabilities",
                        "image_observations", "placement", "relation", "exchange_state"):
            assert retired not in blob, f"the gate prompt carries a retired input: {retired}"

    async def test_attachments_are_named_never_described(self):
        spy = _WakeSpy()
        await _classify(spy, sources=(_source(text="", attachments=("chart.png (image)",)),))
        prompt = spy.params["input"][1]["content"]
        assert "chart.png (image)" in prompt
        assert "contents not shown to you" in prompt

    async def test_an_edited_source_carries_its_before_text(self):
        spy = _WakeSpy()
        await _classify(spy, sources=(_source(
            text="can you check staging",
            edit={"old_text": "can you check stagin", "already_replied": True}),))
        prompt = spy.params["input"][1]["content"]
        assert "can you check stagin" in prompt
        assert "already replied" in prompt

    async def test_no_image_ever_rides_the_request(self):
        spy = _WakeSpy()
        await _classify(spy, sources=(_source(attachments=("photo.png (image)",)),))
        # A single string content, not a multipart list: there is nowhere for pixels to be.
        assert isinstance(spy.params["input"][1]["content"], str)


# --------------------------------------------------------------------------- engine lifecycle

def _engine(wake=True, exc=None):
    llm = MagicMock()
    if exc is not None:
        llm.classify_wake = AsyncMock(side_effect=exc)
    else:
        llm.classify_wake = AsyncMock(return_value=wake)
    return ParticipationEngine(llm), llm


async def _evaluate(engine, *, ts="10.0", text="deploy failed", channel="C1", **kw):
    return await engine.evaluate(channel_id=channel, ts=ts, text=text,
                                 sender_id=kw.pop("sender_id", "U1"), **kw)


def _slow_engine(wake=True, delay=0.05):
    """An engine whose classifier takes long enough to be CANCELLED while it is running.

    The ordinary `_engine` fake returns without ever awaiting, so a call is over before anything
    could interrupt it and every cancellation reads as a no-op. Anything asserting that a call was
    aborted in flight needs a classifier that can actually be caught mid-call, and needs to count
    calls STARTED and calls COMPLETED separately.

    Returns the engine, its client, and the two lists — started cohorts and completed cohorts.
    """
    started, completed = [], []

    async def _classify(*, sources, channel_steering_text=None):
        started.append(tuple(sources))
        await asyncio.sleep(delay)
        completed.append(tuple(sources))
        return wake

    llm = MagicMock()
    llm.classify_wake = AsyncMock(side_effect=_classify)
    return ParticipationEngine(llm), llm, started, completed


def _warm(engine, *, channel="C1", ts="0.5", thread_root=None, sender_id="U1"):
    """Make a stream WARM, so the message under test still gets a debounce window to sit in.

    W5c: a conversation with no arrival inside the last debounce window is judged immediately —
    there is no burst to collect and nothing to wait for. Every test about what happens INSIDE the
    window therefore needs the stream to be active first, which is exactly one prior arrival."""
    engine.note_arrival(channel, ts, thread_root, sender_id)


@pytest.fixture(autouse=True)
def _no_debounce(monkeypatch):
    """The debounce is a sleep, and every test here is about what happens either side of it."""
    monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)


class TestEngineDecisions:
    async def test_wake_true(self):
        engine, _ = _engine(True)
        ev = await _evaluate(engine)
        assert ev.decision == WakeDecision(wake=True)
        assert ev.decline_cause is None
        assert isinstance(ev.classifier_ms, int)

    async def test_wake_false_is_a_decision_not_a_decline(self):
        engine, _ = _engine(False)
        ev = await _evaluate(engine)
        assert ev.decision == WakeDecision(wake=False)
        assert ev.decline_cause is None

    async def test_the_decision_carries_nothing_but_the_bit(self):
        # No confidence, no reason, no action, no metadata. A reason would become the next control
        # bus and would carry a summary of someone's message into the logs besides.
        assert set(WakeDecision.__dataclass_fields__) == {"wake"}

    async def test_a_classifier_failure_is_a_decline_with_no_decision(self):
        engine, _ = _engine(exc=RuntimeError("provider down"))
        ev = await _evaluate(engine)
        assert ev.decision is None
        assert ev.decline_cause == "classifier_error"

    async def test_no_boolean_is_also_a_decline(self):
        engine, _ = _engine(None)
        ev = await _evaluate(engine)
        assert ev.decision is None and ev.decline_cause == "classifier_error"

    async def test_the_steering_text_reaches_the_classifier_untouched(self):
        engine, llm = _engine(True)
        steering = "Standing channel policy (instructions; follow these):\nonly deploys"
        await _evaluate(engine, channel_steering_text=steering)
        assert llm.classify_wake.await_args.kwargs["channel_steering_text"] == steering


class TestCohort:
    async def test_a_top_level_burst_reaches_the_model_once_in_order(self):
        engine, llm = _engine(True)
        # Two earlier sends are superseded by the third; the survivor judges all three.
        for ts in ("1.0", "2.0"):
            engine.note_arrival("C1", "3.0", None, "U1")       # a newer arrival exists
            ev = await _evaluate(engine, ts=ts, text=f"part {ts}")
            assert ev.decline_cause == "superseded"
        ev = await _evaluate(engine, ts="3.0", text="part 3.0")
        assert ev.decision.wake is True
        sent = llm.classify_wake.await_args.kwargs["sources"]
        assert [s.ts for s in sent] == ["1.0", "2.0", "3.0"]    # oldest first
        assert [s.ts for s in ev.sources] == ["1.0", "2.0", "3.0"]
        assert llm.classify_wake.await_count == 1               # ONE call for the whole burst

    async def test_a_thread_cohort_collapses_across_authors(self):
        engine, llm = _engine(True)
        engine.note_arrival("C1", "2.0", "root", "U2")
        await _evaluate(engine, ts="1.0", sender_id="U1", thread_root_ts="root", text="from U1")
        ev = await _evaluate(engine, ts="2.0", sender_id="U2", thread_root_ts="root", text="from U2")
        assert [s.ts for s in ev.sources] == ["1.0", "2.0"]
        assert {s.sender_id for s in ev.sources} == {"U1", "U2"}

    async def test_two_senders_at_top_level_are_not_merged(self):
        # A top-level stream keys per sender, so two people's unrelated questions are independent
        # and BOTH get judged.
        engine, llm = _engine(True)
        a = await _evaluate(engine, ts="1.0", sender_id="U1", text="mine")
        b = await _evaluate(engine, ts="2.0", sender_id="U2", text="theirs")
        assert a.decision.wake and b.decision.wake
        assert [s.ts for s in a.sources] == ["1.0"]
        assert [s.ts for s in b.sources] == ["2.0"]

    async def test_nothing_is_capped_or_dropped(self):
        # There is no _MAX_BURST_CARRY any more and no freshness window. Five sends, one of them
        # deliberately old, all arrive: a structure whose whole job is not to lose a message must
        # not silently evict one.
        engine, llm = _engine(True)
        engine.note_arrival("C1", "5000.0", None, "U1")
        for ts in ("1.0", "1000.0", "2000.0", "3000.0"):
            await _evaluate(engine, ts=ts, text=f"part {ts}")
        ev = await _evaluate(engine, ts="5000.0", text="last")
        assert [s.ts for s in ev.sources] == ["1.0", "1000.0", "2000.0", "3000.0", "5000.0"]

    async def test_a_newer_arrival_is_left_for_its_own_survivor(self):
        engine, _ = _engine(True)
        engine._enroll_source("k", _source(ts="9.0", text="not yet debounced"))
        engine._enroll_source("k", _source(ts="1.0", text="mine"))
        taken = engine._drain_cohort("k", "1.0")
        assert [s.ts for s in taken] == ["1.0"]
        assert "9.0" in engine._cohorts["k"]

    async def test_a_survivor_drain_empties_the_bucket(self):
        engine, _ = _engine(True)
        await _evaluate(engine, ts="1.0")
        assert engine._cohorts == {}          # no leak: the map is unbounded, so it must self-clear

    async def test_discard_source_forgets_a_cancelled_source(self):
        engine, _ = _engine(True)
        engine._enroll_source(engine._conv_key("C1", "1.0", None, "U1"), _source(ts="1.0"))
        engine.discard_source("C1", "1.0", None, "U1")
        assert engine._cohorts == {}

    async def test_a_superseded_attempt_leaves_its_words_for_the_survivor(self):
        engine, llm = _engine(True)
        engine.note_arrival("C1", "2.0", None, "U1")
        ev = await _evaluate(engine, ts="1.0", text="the part that must not be lost")
        assert ev.decline_cause == "superseded" and ev.sources == ()
        survivor = await _evaluate(engine, ts="2.0", text="the rest")
        assert "the part that must not be lost" in [s.text for s in survivor.sources]


class TestImageOnlyCohort:
    async def test_a_captionless_upload_declines_structurally(self):
        # No text means no question, no addressee and nothing for a wake decision to be about. It
        # is named in the ledger rather than dressed up as the model choosing silence.
        engine, llm = _engine(True)
        ev = await _evaluate(engine, text="", attachments=["photo.png (image)"])
        assert ev.decline_cause == "image_only"
        assert ev.decision is None
        llm.classify_wake.assert_not_awaited()          # no classifier runs at all

    async def test_a_caption_makes_it_an_ordinary_judgment(self):
        engine, llm = _engine(True)
        ev = await _evaluate(engine, text="what do you make of this?",
                             attachments=["photo.png (image)"])
        assert ev.decision.wake is True
        llm.classify_wake.assert_awaited_once()

    async def test_one_captioned_sibling_rescues_the_cohort(self):
        engine, llm = _engine(True)
        engine.note_arrival("C1", "2.0", None, "U1")
        await _evaluate(engine, ts="1.0", text="", attachments=["photo.png (image)"])
        ev = await _evaluate(engine, ts="2.0", text="thoughts?")
        assert ev.decision.wake is True

    async def test_a_textless_message_with_no_files_is_not_image_only(self):
        engine, llm = _engine(False)
        ev = await _evaluate(engine, text="")
        assert ev.decline_cause is None and ev.decision.wake is False


class TestEditOwnership:
    def _facade(self, marker="m1"):
        client = MagicMock()
        client._edit_reply_ctx_map = {
            f"C1|10.0|{marker}": {"old_text": "befor", "new_text": "before",
                                  "already_replied": False}}
        return client

    async def test_only_the_marked_attempt_consumes_the_context(self):
        engine, llm = _engine(True)
        client = self._facade()
        ev = await _evaluate(engine, client=client, edit_marker="m1")
        assert ev.sources[0].edit is not None
        assert client._edit_reply_ctx_map == {}          # consumed

    async def test_the_original_attempt_cannot_claim_it(self):
        # The bug this fixes: an edit keeps its original ts, so (channel, ts) alone could not tell
        # the edit's attempt from the stale original it superseded — and whichever ran first popped
        # the context. The original could then conclude it WAS the edit, which also suppressed the
        # supersession check meant to silence it.
        engine, llm = _engine(True)
        client = self._facade()
        ev = await _evaluate(engine, client=client, edit_marker=None)
        assert ev.sources[0].edit is None
        assert client._edit_reply_ctx_map != {}          # still there for its rightful owner

    async def test_a_different_marker_cannot_claim_it(self):
        engine, llm = _engine(True)
        client = self._facade(marker="m1")
        ev = await _evaluate(engine, client=client, edit_marker="m2")
        assert ev.sources[0].edit is None

    async def test_the_marker_less_original_still_consumes_its_supersession(self):
        engine, llm = _engine(True)
        engine.supersede("C1", "10.0", None, "U1")
        ev = await _evaluate(engine, ts="10.0")
        assert ev.decline_cause == "edit_superseded"

    async def test_the_marked_edit_attempt_is_exempt_from_that_mark(self):
        engine, llm = _engine(True)
        client = self._facade()
        engine.supersede("C1", "10.0", None, "U1")
        ev = await _evaluate(engine, ts="10.0", client=client, edit_marker="m1")
        assert ev.decision.wake is True


# --------------------------------------------------------------------------- the caller

def _gate_app(engine, evaluation):
    """main.ChatBotV2 wired down to the gate path."""
    from main import ChatBotV2

    app = ChatBotV2.__new__(ChatBotV2)
    app.processor = MagicMock()
    app.processor.db = MagicMock()
    app.processor.db.get_channel_memory_async = AsyncMock(return_value=[])
    app.processor.db.get_channel_policy_async = AsyncMock(return_value=None)
    app.participation_engine = MagicMock()
    app.participation_engine.evaluate = AsyncMock(return_value=evaluation)
    app.participation_engine.note_arrival = MagicMock()
    client = MagicMock()
    client.bot_handle = "ChatGPT"
    return app, client


def _gate_msg(**meta):
    md = {"ts": "10.0", "gate_required": True, "participation_level": "on"}
    md.update(meta)
    return Message(text="deploy failed", user_id="U1", channel_id="C1", thread_id="10.0",
                   metadata=md)


class TestGateCaller:
    async def _run(self, evaluation, monkeypatch):
        terminals, decisions, declines = [], [], []
        monkeypatch.setattr(participation_telemetry, "finish_attempt",
                            lambda msg, kind, **kw: terminals.append((kind, kw)))
        monkeypatch.setattr(participation_telemetry, "gate_decision",
                            lambda *a, **kw: decisions.append(kw))
        monkeypatch.setattr(participation_telemetry, "gate_declined",
                            lambda *a, **kw: declines.append(kw))
        app, client = _gate_app(None, evaluation)
        msg = _gate_msg()
        out = await app._gate_verdict(msg, client)
        return out, terminals, decisions, msg

    async def test_a_wake_hands_the_turn_on_with_exactly_one_ledger_decision(self, monkeypatch):
        ev = GateEvaluation(decision=WakeDecision(wake=True), classifier_ms=12,
                            sources=(_source(ts="9.0"), _source(ts="10.0")))
        out, terminals, decisions, msg = await self._run(ev, monkeypatch)
        assert out == WakeDecision(wake=True)
        assert terminals == []                  # the responder owns the terminal on a wake
        assert len(decisions) == 1
        assert decisions[0]["wake"] is True
        assert decisions[0]["source_count"] == 2
        assert decisions[0]["newest_source_ts"] == "10.0"
        # The cohort rides on for the responder to answer.
        assert msg.metadata["gate_sources"] == ev.sources

    async def test_no_wake_is_a_silence_with_no_reason_of_any_kind(self, monkeypatch):
        ev = GateEvaluation(decision=WakeDecision(wake=False), classifier_ms=9,
                            sources=(_source(),))
        out, terminals, decisions, msg = await self._run(ev, monkeypatch)
        assert out is None
        assert len(terminals) == 1
        kind, kwargs = terminals[0]
        assert kind == "silence"
        assert kwargs.get("ended_by") == "gate"
        # NO silence_reason: that eight-value enum belongs to the responder, which can say why it
        # chose quiet after seeing everything. The gate knows only that it did not open.
        assert "silence_reason" not in kwargs
        assert "reason" not in kwargs and "action" not in kwargs
        assert len(decisions) == 1 and decisions[0]["wake"] is False

    async def test_a_classifier_failure_is_terminal_none_and_no_decision(self, monkeypatch):
        ev = GateEvaluation(decline_cause="classifier_error", classifier_ms=30)
        out, terminals, decisions, _ = await self._run(ev, monkeypatch)
        assert out is None
        assert [k for k, _ in terminals] == ["none"]
        assert terminals[0][1].get("cause") == "classifier_error"
        assert decisions == []                  # nothing was decided, so nothing is recorded

    @pytest.mark.parametrize("cause", ["superseded", "edit_superseded", "image_only"])
    async def test_every_decline_is_terminal_none(self, cause, monkeypatch):
        out, terminals, decisions, _ = await self._run(
            GateEvaluation(decline_cause=cause), monkeypatch)
        assert out is None
        assert [k for k, _ in terminals] == ["none"]
        assert decisions == []

    async def test_the_gate_places_no_reaction_on_any_outcome(self, monkeypatch):
        # The rich gate could react instead of replying, which put an emoji in the room before the
        # answering model had a turn — and then had to tell it so it wouldn't add a second one.
        from main import ChatBotV2
        for name in ("_place_gate_reaction", "_apply_backoff", "_backoff_ack",
                     "_apply_pref_memory", "_own_pref_row", "_pref_memory_content"):
            assert not hasattr(ChatBotV2, name), f"{name} should be gone with the rich gate"

    async def test_a_no_wake_catalogues_unattended_attachments(self, monkeypatch):
        # Deciding not to REPLY is not deciding to FORGET. A dropped file once ceased to exist as
        # far as the bot was concerned while the CSV sat in the channel.
        scheduled = []
        monkeypatch.setattr(participation_telemetry, "finish_attempt", lambda *a, **k: None)
        monkeypatch.setattr(participation_telemetry, "gate_decision", lambda *a, **k: None)
        monkeypatch.setattr(participation_telemetry, "mark_gate_woke", lambda *a, **k: None)
        app, client = _gate_app(None, GateEvaluation(decision=WakeDecision(wake=False)))
        app.processor._schedule_async_call = MagicMock(side_effect=lambda c: scheduled.append(c))
        msg = _gate_msg()
        msg.attachments = [{"type": "image", "url": "http://x/y.png"}]
        with patch("message_processor.thread_files.catalog_unattended",
                   return_value=MagicMock()) as catalog:
            await app._run_participation_gate(msg, client)
        assert catalog.called and scheduled


# --------------------------------------------------------------------------- responder input

class TestCohortBecomesInput:
    """The cohort still reaches the model — as evidence, not as forged history.

    It used to be written into `ThreadState.messages` as synthetic user turns. A channel turn does
    not send that list any more: every burst member Slack propagated is in the stream with its own
    real header, and only the ones that arrived too late to be fetched are quoted back."""

    def _ctx(self, msg, *, stream_has=()):
        from message_processor import channel_request
        from tests.unit.channel_turn_harness import normalized, pin_channel_turn
        from message_processor.turn_runtime import TurnRuntime

        turn = TurnRuntime()
        messages = [normalized(ts, f"stream copy of {ts}") for ts in stream_has]
        return pin_channel_turn(
            turn, messages=messages or [normalized("3.0", "anyone looking?")],
            trigger_ts="3.0",
            cohort_sources=channel_request.cohort_sources_from_message(msg))

    def test_earlier_sources_are_quoted_as_awaiting_stream_evidence(self):
        from message_processor.channel_request import build_cohort_fallback

        msg = SimpleNamespace(metadata={"ts": "3.0", "gate_sources": (
            _source(ts="1.0", text="the build is red", sender_name="Peter"),
            _source(ts="3.0", text="anyone looking?", sender_name="Peter"))})
        block = build_cohort_fallback(self._ctx(msg))
        assert block["role"] == "user"          # somebody's words, never developer voice
        assert "Peter (ts=1.0): the build is red" in block["content"]
        # The trigger itself is not "also" anything — it is what this turn is answering.
        assert "anyone looking?" not in block["content"]

    def test_a_source_the_stream_already_contains_is_not_quoted_twice(self):
        from message_processor.channel_request import build_cohort_fallback

        msg = SimpleNamespace(metadata={"ts": "3.0", "gate_sources": (
            _source(ts="1.0", text="already fetched"),)})
        assert build_cohort_fallback(self._ctx(msg, stream_has=("1.0", "3.0"))) is None

    def test_attachments_are_named_in_the_quoted_source(self):
        from message_processor.channel_request import build_cohort_fallback

        msg = SimpleNamespace(metadata={"ts": "3.0", "gate_sources": (
            _source(ts="1.0", text="", sender_name="Peter",
                    attachments=("chart.png (image)",)),)})
        assert "[attached: chart.png (image)]" in build_cohort_fallback(self._ctx(msg))["content"]

    def test_no_cohort_is_a_noop(self):
        from message_processor.channel_request import build_cohort_fallback

        assert build_cohort_fallback(self._ctx(SimpleNamespace(metadata={}))) is None

    def test_the_reader_turns_a_staged_payload_into_an_authorized_file(self):
        """[r4-4] The READER half, with the metadata hand-written: given a staged payload, the
        FileRef is built and the id lands in the turn's canonical catalog. Who stages it is a
        separate question, answered on the production path — the gate stamps its own cohort's
        payloads (`test_a_superseded_arrivals_file_is_staged_by_the_gate_itself` below) and the
        queue drain stages the batch's (test_conversational_queue.py)."""
        from message_processor.channel_request import (canonical_files_from_stream,
                                                       cohort_sources_from_message,
                                                       merge_absent_source_files)
        from tests.unit.channel_turn_harness import normalized, build_stream

        msg = SimpleNamespace(metadata={
            "ts": "3.0",
            "gate_sources": (_source(ts="1.0", text="numbers",
                                     attachments=("data.csv (file)",)),),
            "batched_file_refs": [{"ts": "1.0", "attachments": [
                {"id": "F9", "name": "data.csv", "mimetype": "text/csv",
                 "url": "https://files.slack.com/files-pri/T1-F9/data.csv", "size": 12}]}]})
        stream = build_stream([normalized("3.0", "what do the numbers say?")])
        merged = merge_absent_source_files(
            canonical_files_from_stream(stream), cohort_sources_from_message(msg), "C1")
        assert merged["F9"]["filename"] == "data.csv"
        assert merged["F9"]["message_ts"] == "1.0"

    async def test_a_superseded_arrivals_file_is_staged_by_the_gate_itself(self, monkeypatch):
        """[r6-3] The WRITER, on the production path: the live failure that started this.

        Someone drops a CSV and immediately asks about it. The CSV's own dispatch loses its debounce
        to the question and ends at the gate — so that dispatch is the last holder of the file's
        payload. The gate keeps it with the enrolled record and stamps it on the survivor, and the
        survivor's turn can authorize the id even though Slack's window never returned the message.

        The test never writes `batched_file_refs`; it asserts that the gate did.
        """
        from message_processor.channel_request import (canonical_files_from_stream,
                                                       cohort_sources_from_message,
                                                       merge_absent_source_files)
        from tests.unit.channel_turn_harness import build_stream, normalized

        monkeypatch.setattr(config, "participation_debounce_seconds", 0.2, raising=False)
        for name in ("finish_attempt", "gate_decision", "gate_declined", "begin_attempt",
                     "attempt_id_for"):
            monkeypatch.setattr(participation_telemetry, name, lambda *a, **k: None)
        engine, _ = _engine(True)
        # W5c: the CSV's dispatch only loses a debounce it actually waits, so the thread has to be
        # warm — which it is in the live case this reconstructs, where the file lands mid-thread.
        _warm(engine, ts="14.0", thread_root="10.0")
        app, client = _gate_app(None, None)
        app.participation_engine = engine

        csv = {"type": "file", "name": "data.csv", "id": "F9", "mimetype": "text/csv",
               "url": "https://files.slack.com/files-pri/T1-F9/data.csv", "size": 12}
        with_file = _gate_msg(ts="15.0", participation_attachments=("data.csv (file)",))
        with_file.text = "here are the numbers"
        with_file.attachments = [csv]
        question = _gate_msg(ts="20.0")
        question.text = "what do the numbers say?"
        question.attachments = []

        superseded = asyncio.ensure_future(app._gate_verdict(with_file, client))
        for _ in range(400):                      # enrolled and asleep in its own debounce
            if engine._cohorts:
                break
            await asyncio.sleep(0.005)
        assert engine._cohorts, "the file's dispatch never enrolled"

        decision = await asyncio.wait_for(app._gate_verdict(question, client), timeout=5)
        assert decision.wake is True
        assert await asyncio.wait_for(superseded, timeout=5) is None

        assert question.metadata["batched_file_refs"] == [{"ts": "15.0", "attachments": [csv]}]
        # And the far end: the window Slack returned has no trace of ts 15.0, and the id is
        # authorized anyway.
        stream = build_stream([normalized("20.0", "what do the numbers say?")])
        merged = merge_absent_source_files(
            canonical_files_from_stream(stream), cohort_sources_from_message(question), "C1")
        assert merged["F9"]["message_ts"] == "15.0"
        assert engine._cohort_files == {}, "the payload map must not outlive the cohort"

    def test_the_cohort_never_touches_thread_state(self):
        """The tripwire for the whole change: the helper that wrote synthetic history is gone."""
        from message_processor.utilities import MessageUtilitiesMixin
        assert not hasattr(MessageUtilitiesMixin, "_merge_gate_cohort")

    def test_the_burst_prose_is_gone(self):
        # It used to arrive as a metadata line quoting the earlier texts inside a block the prompt
        # also tells the model not to trust as content — with no sender, time or attachments.
        from message_processor.utilities import MessageUtilitiesMixin
        assert not hasattr(MessageUtilitiesMixin, "_wake_burst_line")
        assert not hasattr(MessageUtilitiesMixin, "_reacted_already_note")


# --------------------------------------------------------------------------- telemetry v8

class TestTelemetryV8:
    def test_contract_and_gate_name(self):
        """CV8 added the single-stream events and a second population keyed by turn_id, so the
        event set an attempt can produce changed. GATE_CONTRACT did NOT move: the gate is the
        same one bit it was under v7, and pooling its rows across the version boundary is
        legitimate in a way that pooling rich-gate rows never was."""
        assert participation_telemetry.CONTRACT_VERSION == 10   # v10: edit_own_message
        assert participation_telemetry.GATE_CONTRACT == "binary-v1"

    def test_gate_decision_writes_exactly_the_v7_field_set(self, monkeypatch):
        written = {}
        monkeypatch.setattr(participation_telemetry, "record",
                            lambda event, **fields: written.update({"event": event, **fields}))
        participation_telemetry.gate_decision(
            "C1", "10.0", wake=True, attempt_id="a1", gate_ms=3100, classifier_ms=740,
            source_count=3, newest_source_ts="12.0")
        assert written["event"] == "gate_decision"
        assert set(written) == {
            "event", "channel_id", "trigger_ts", "attempt_id",
            "wake", "model", "gate_ms", "classifier_ms", "source_count", "newest_source_ts"}
        for retired in ("action", "emoji", "placement", "reason", "relation", "exchange_state",
                        "answerability", "overruled_by", "dimension", "durability", "scope",
                        "memory_op", "structural_request"):
            assert retired not in written

    def test_image_only_is_a_known_decline_cause(self):
        assert "image_only" in participation_telemetry.DECLINE_CAUSES


# --------------------------------------------------------------------------- tripwires

def _our_sources():
    skip = {"tests", "venv", ".venv", "node_modules", "build", "dist", "__pycache__",
            "site-packages", "Docs"}
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT)
        if set(rel.parts) & skip or any(p.startswith(".") for p in rel.parts):
            continue
        yield rel, path.read_text(encoding="utf-8")


@pytest.mark.parametrize("symbol", [
    "ParticipationVerdict", "validate_verdict", "_apply_invariants", "classify_participation",
    "PARTICIPATION_SYSTEM_PROMPT", "_place_gate_reaction", "_apply_backoff", "_backoff_ack",
    "participation_reaction_emoji", "participation_burst_earlier", "participation_reason",
    "participation_images", "_MAX_BURST_CARRY", "SPEAKING_ACTIONS",
])
def test_the_rich_gate_is_gone_from_every_source_file(symbol):
    """Leaving any of these executable changes behaviour, which is why they die in this commit
    rather than in the cleanup one.

    Word-boundary matched, because a substring search reads `participation_reasoning_effort` — a
    live config field — as the deleted `participation_reason` metadata key."""
    import re
    pattern = re.compile(rf"\b{re.escape(symbol)}\b")
    offenders = [f"{rel}" for rel, src in _our_sources() if pattern.search(src)]
    assert not offenders, f"{symbol} still appears in: {', '.join(offenders)}"


def test_nothing_imports_gate_vision_any_more():
    """The module may wait for the cleanup commit, but it must have zero runtime callers: the gate
    does not look at images, and an image hold with no resolver would delay every ambient
    analysis."""
    # AST, not text: the module is named in a couple of explanatory comments and docstrings (it
    # is where the mimetype allowlist came from), and a prose mention is not a caller.
    offenders = []
    for rel, src in _our_sources():
        if rel.name == "gate_vision.py":
            continue
        tree = ast.parse(src, filename=str(rel))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name.split(".")[-1] == "gate_vision" for a in node.names):
                    offenders.append(f"{rel}:{node.lineno} (import)")
            elif isinstance(node, ast.ImportFrom):
                if any(a.name == "gate_vision" for a in node.names) \
                        or (node.module or "").endswith("gate_vision"):
                    offenders.append(f"{rel}:{node.lineno} (import)")
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                    and node.value.id == "gate_vision":
                offenders.append(f"{rel}:{node.lineno} (use)")
    assert not offenders, "gate_vision still has callers at: " + ", ".join(offenders)


def test_nothing_defers_ambient_images():
    """Nothing may wait for a resolver the gate no longer calls."""
    offenders = []
    for rel, src in _our_sources():
        if rel.name == "ambient_memory.py":
            continue                      # it DEFINES the parameter; the point is nobody sets it
        for lineno, line in enumerate(src.splitlines(), 1):
            if "defer_images=" in line and not line.strip().startswith("#"):
                offenders.append(f"{rel}:{lineno}")
    assert not offenders, "ambient images are still deferred at: " + ", ".join(offenders)


def test_the_gate_has_exactly_one_classifier():
    src = (ROOT / "openai_client/api/responses.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {n.name for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
             and n.name.startswith("classify")}
    assert names == {"classify_wake"}


def test_judicious_and_active_are_no_longer_legal_levels():
    from message_processor.participation import LEVEL_TO_MODE, MODE_TO_LEVEL, VALID_LEVELS
    assert VALID_LEVELS == ("off", "mentions_only", "on")
    assert "judicious" not in VALID_LEVELS and "active" not in VALID_LEVELS
    assert MODE_TO_LEVEL["auto_respond"] == "on"
    assert LEVEL_TO_MODE["on"] == "auto_respond"


# --------------------------------------------------------------------------- feedback + queueing

def test_the_prompt_tells_the_gate_to_wake_on_participation_feedback():
    """Feedback about how the assistant participates asks for nothing, so a gate optimising for
    "is a reply wanted" would sleep through it — and only the responder can record the preference
    or change the setting, so sleeping DISCARDS the instruction. The prompt says so explicitly,
    because it is the one case where waking is right despite there being no question."""
    from prompts import WAKE_CLASSIFIER_SYSTEM_PROMPT as p

    assert "feedback about how it participates" in p
    assert "silently discards the instruction" in p
    # And the gate is told what a turn can do, so "nothing to answer" is not read as "nothing to
    # do": the responder can react, change a setting, remember something, or stay silent.
    for capability in ("emoji reaction", "change this channel's settings", "remember something",
                       "say nothing at all"):
        assert capability in p


def test_the_prompt_forbids_the_gate_from_doing_the_responders_job():
    from prompts import WAKE_CLASSIFIER_SYSTEM_PROMPT as p

    assert "You are not the assistant" in p
    assert "You only decide whether the assistant gets a turn" in p
    # And it is told to prefer waking when the doubt is whether a task needs it — a false wake
    # costs a call and can end in silence; a false sleep loses the answer.
    assert "If you cannot tell whether a genuine task or question needs it, wake it" in p


class TestQueueBatchAndCohortInterop:
    """Two different coalescing mechanisms, and they must not double-count each other.

    The debounce cohort holds arrivals that landed BEFORE the responder started; the Phase-Q queue
    drain holds arrivals that landed while it was running. A message can legitimately be in both
    paths' sights, and the model must see it once. Under one stream the dedup key is no longer
    "already in ThreadState" — it is "already in the pinned window", which is the stronger fact.
    """

    def _absent(self, msg, stream_has):
        from message_processor import channel_request
        from tests.unit.channel_turn_harness import normalized, pin_channel_turn
        from message_processor.turn_runtime import TurnRuntime

        ctx = pin_channel_turn(
            TurnRuntime(), messages=[normalized(ts, "fetched") for ts in stream_has],
            trigger_ts="4.0",
            cohort_sources=channel_request.cohort_sources_from_message(msg))
        block = channel_request.build_cohort_fallback(ctx)
        return block["content"] if block else ""

    def test_a_source_the_window_already_fetched_is_not_quoted(self):
        msg = SimpleNamespace(metadata={"ts": "4.0", "gate_sources": (
            _source(ts="1.0", text="a"), _source(ts="3.0", text="c"),
            _source(ts="4.0", text="trigger"))})
        quoted = self._absent(msg, stream_has=("1.0", "2.0", "4.0"))
        assert "ts=3.0" in quoted                 # only the one nobody had
        assert "ts=1.0" not in quoted and "ts=4.0" not in quoted

    def test_the_gate_cohort_and_the_carried_batch_are_deduplicated(self):
        """A queue drain hands its absorbed batch on as `carried_gate_sources`, and the gate may
        have judged the same message. One quote, not two."""
        from message_processor.channel_request import cohort_sources_from_message

        msg = SimpleNamespace(metadata={
            "ts": "4.0",
            "gate_sources": (_source(ts="1.0", text="a"),),
            "carried_gate_sources": (_source(ts="1.0", text="a"),
                                     _source(ts="2.0", text="b"))})
        sources = cohort_sources_from_message(msg)
        assert [s.ts for s in sources] == ["1.0", "2.0"]

    def test_assembling_twice_is_idempotent(self):
        """A retry re-assembles from the same pins, so the quoted set cannot grow."""
        from message_processor import channel_request
        from tests.unit.channel_turn_harness import normalized, pin_channel_turn
        from message_processor.turn_runtime import TurnRuntime

        msg = SimpleNamespace(metadata={"ts": "2.0", "gate_sources": (_source(ts="1.0"),)})
        ctx = pin_channel_turn(
            TurnRuntime(), messages=[normalized("2.0", "trigger")], trigger_ts="2.0",
            cohort_sources=channel_request.cohort_sources_from_message(msg))
        first = channel_request.build_cohort_fallback(ctx)["content"]
        assert channel_request.build_cohort_fallback(ctx)["content"] == first


async def test_ambient_images_are_admitted_immediately():
    """The hold existed so ONE vision look served both the gate's verdict and the stored
    observation. The gate does not look at images now, so a hold would only delay analysis — and
    worse, its resolver was the gate's own post-verdict callback, which no longer fires.

    The always-False predicate that stood in for the hold is gone with it, so this now drives the
    real ingest seam and asserts what it hands the service: the event, the facade, and no third
    thing that could ask for a hold."""
    from slack_client.event_handlers.message_events import SlackMessageEventsMixin

    class _Svc:
        def __init__(self):
            self.offers = []

        def offer_event(self, event, client, **kw):
            self.offers.append((event, client, kw))

    svc = _Svc()
    mixin = SlackMessageEventsMixin.__new__(SlackMessageEventsMixin)
    mixin.processor = SimpleNamespace(ambient_service=svc)
    mixin.is_own_message = lambda _m: False
    mixin.log_debug = lambda *_a, **_k: None
    assert not hasattr(SlackMessageEventsMixin, "_gate_will_see_images")

    event = {"channel": "C1", "ts": "1.1", "subtype": "file_share",
             "files": [{"id": "F1", "name": "x.png", "mimetype": "image/png"}]}
    await mixin._ambient_ingest(event, MagicMock())

    assert len(svc.offers) == 1
    offered, client, kw = svc.offers[0]
    assert offered is event and client is mixin      # the facade, never the raw Bolt client
    assert kw == {}                                  # nothing asks for a hold


def test_attachment_descriptors_are_names_and_types_only():
    from slack_client.event_handlers.message_events import _attachment_descriptors

    assert _attachment_descriptors(None) == ()
    assert _attachment_descriptors([
        {"name": "food.png", "mimetype": "image/png"},
        {"name": "report.pdf", "mimetype": "application/pdf"},
    ]) == ("food.png (image)", "report.pdf (file)")


async def test_process_message_puts_the_cohort_in_front_of_the_handler():
    """The other half of the contract, and the half that has been silently broken before: a helper
    that builds context correctly is worthless if no call site invokes it. (The channel-memory
    block was assembled into an attribute nothing read, for months.) So this drives the real
    process_message and inspects the request the handler was actually handed — and asserts the
    thing that used to happen instead: nothing was written into ThreadState."""
    from message_processor.base import MessageProcessor
    from message_processor.channel_request import to_input_items
    from tests.unit.channel_turn_harness import item_texts, normalized, build_stream

    with patch("message_processor.base.AsyncThreadStateManager"), \
         patch("message_processor.base.OpenAIClient"):
        p = MessageProcessor()

    state = SimpleNamespace(had_timeout=False, messages=[], thread_ts="10.0", channel_id="C1",
                            root_author=("U1", "human"), config_overrides={}, participants={},
                            current_model=None, has_trimmed_messages=False)
    p.db = MagicMock()
    p.db.get_channel_memory_async = AsyncMock(return_value=[])
    p.db.get_channel_policy_async = AsyncMock(return_value=None)
    p.thread_manager.acquire_thread_lock = AsyncMock(return_value=True)
    p.thread_manager.release_thread_lock = AsyncMock()

    async def _state_for(*a, **k):
        return state

    p.get_or_create_channel_thread_state = _state_for
    p._process_attachments = AsyncMock(return_value=([], [], []))
    p._build_channel_info = AsyncMock(return_value=None)
    p._build_tools_array = MagicMock(return_value=None)
    p._get_system_prompt = MagicMock(return_value="sys")
    p._prepare_channel_turn_tools = AsyncMock(return_value=(None, {}, False, None, None))
    p.ingest_channel_origin_slice = AsyncMock()

    # The stream fetched the trigger but not the earlier send — the propagation lag this exists for.
    stream = build_stream([normalized("3.0", "anyone looking?", sender_id="U1")], h="3.0")
    p._build_channel_turn_stream = AsyncMock(return_value=stream)

    captured = {}

    async def _handler(user_content, thread_state, client, message, thinking_id, **kw):
        request, *_ = await p._assemble_channel_attempt(
            client, message, thread_state, kw["turn"], {"model": "gpt-5.6-sol"}, "gpt-5.6-sol",
            thread_key="C1:10.0")
        captured["texts"] = item_texts(to_input_items(request))
        raise _CohortStop()

    class _CohortStop(Exception):
        pass

    p._handle_text_response = AsyncMock(side_effect=_handler)

    msg = Message(text="anyone looking?", user_id="U1", channel_id="C1", thread_id="10.0",
                  metadata={"ts": "3.0", "gate_required": True, "gate_sources": (
                      _source(ts="1.0", text="the build is red", sender_name="Peter"),
                      _source(ts="3.0", text="anyone looking?", sender_name="Peter"))})
    client = MagicMock()
    client.send_message = AsyncMock()
    client.self_team_id = "T1"
    with patch.object(config, "enable_channel_memory", True):
        await p.process_message(msg, client, None)

    p._handle_text_response.assert_awaited()
    blob = "\n".join(captured["texts"])
    assert "the build is red" in blob, "the earlier send of the burst never reached the model"
    # …and NOTHING was written into the thread state to get it there.
    assert state.messages == []


# --------------------------------------------------------------------------- review round fixes

class TestQueuedBatchIsNotDecidedForInAbsentia:
    """The blocker. The Phase-Q drain folds queued messages into one catch-up turn whose TRIGGER
    is the newest of them. If that trigger is gate-routed and the gate says no, everything that
    queued behind it is discarded — including messages that had ALREADY earned an answer — purely
    because something ambient landed last. They are still in Slack and in the thread state; what is
    lost is the reply somebody was waiting for, with no trace anywhere."""

    def _msg(self, ts, **meta):
        md = {"ts": ts, "gate_required": True, "gate_woke": False}
        md.update(meta)
        return Message(text=f"msg {ts}", user_id="U1", channel_id="C1", thread_id="10.0",
                       metadata=md)

    async def test_the_whole_batch_reaches_the_classifier(self):
        engine, llm = _engine(True)
        from message_processor.participation import source_from_message

        carried = [source_from_message(self._msg("1.0")), source_from_message(self._msg("2.0"))]
        ev = await _evaluate(engine, ts="3.0", text="the trigger", carried_sources=carried)
        assert [s.ts for s in ev.sources] == ["1.0", "2.0", "3.0"]
        assert [s.ts for s in llm.classify_wake.await_args.kwargs["sources"]] == \
            ["1.0", "2.0", "3.0"]

    async def test_the_drain_stamps_the_batch_on_the_trigger(self):
        from message_processor import routing_facts

        batch = [self._msg("1.0"), self._msg("2.0"), self._msg("3.0")]
        trigger = batch[-1]
        # What _dispatch_pending_batch does with the members it absorbs.
        from message_processor.participation import source_from_message
        routing_facts.absorb_owed_answer(trigger, batch[:-1])
        trigger.metadata["carried_gate_sources"] = tuple(
            source_from_message(m) for m in batch[:-1])
        assert [s.ts for s in trigger.metadata["carried_gate_sources"]] == ["1.0", "2.0"]
        assert trigger.metadata["gate_required"] is True     # nothing here was owed yet

    @pytest.mark.parametrize("owed_meta", [
        {"gate_required": False},                    # addressed: an @mention that queued
        {"gate_required": True, "gate_woke": True},  # a gate already said wake
    ])
    async def test_an_answer_already_owed_survives_absorption(self, owed_meta):
        from message_processor import routing_facts

        owed = self._msg("1.0", **owed_meta)
        trigger = self._msg("2.0")
        assert routing_facts.absorb_owed_answer(trigger, [owed]) is True
        assert trigger.metadata["gate_required"] is False
        assert trigger.metadata["gate_woke"] is False

    async def test_an_unjudged_batch_still_goes_to_the_gate(self):
        from message_processor import routing_facts

        trigger = self._msg("2.0")
        assert routing_facts.absorb_owed_answer(trigger, [self._msg("1.0")]) is False
        assert trigger.metadata["gate_required"] is True

    def test_owes_answer_reads_both_ways_of_earning_a_turn(self):
        from message_processor import routing_facts

        assert routing_facts.owes_answer(self._msg("1.0", gate_required=False)) is True
        assert routing_facts.owes_answer(self._msg("1.0", gate_woke=True)) is True
        assert routing_facts.owes_answer(self._msg("1.0")) is False

    async def test_a_carried_source_is_not_enrolled_twice(self):
        engine, llm = _engine(True)
        ev = await _evaluate(engine, ts="1.0", text="mine",
                             carried_sources=[_source(ts="1.0", text="duplicate of me")])
        assert [s.ts for s in ev.sources] == ["1.0"]
        assert ev.sources[0].text == "mine"


class TestQueueDrainsSkipTheDebounce:
    """W5a. The debounce exists to let a burst finish arriving before the gate judges it. A
    Phase-Q drain has already done exactly that — it lingered with the lock held so stragglers
    could join, then popped the whole queue — so the redispatch arrives with its batch closed.
    Sleeping the debounce again coalesces nothing and delays messages that have been waiting since
    before the previous turn ended.

    The signal is an explicit flag, NOT `carried_gate_sources`: that is stamped only for a batch of
    more than one, so inferring from it would leave the single-message drain paying in full.
    """

    def _drain_proc(self, manager):
        from message_processor.base import MessageProcessor

        proc = SimpleNamespace()
        proc.thread_manager = manager
        proc.db = None
        proc._format_user_content_with_username = lambda content, m: content
        proc._add_message_with_token_management = MagicMock()
        proc._schedule_async_call = MagicMock()
        for name in ("log_info", "log_debug", "log_warning", "log_error"):
            setattr(proc, name, lambda *a, **k: None)
        proc._dispatch_pending_batch = MessageProcessor._dispatch_pending_batch.__get__(proc)
        return proc

    def _queued(self, ts):
        return Message(text=f"msg {ts}", user_id="U1", channel_id="C1", thread_id="10.0",
                       metadata={"ts": ts})

    # -- the stamp

    @pytest.mark.parametrize("size", [1, 3])
    async def test_every_redispatch_is_stamped_whatever_the_batch_size(self, size):
        from thread_manager import AsyncThreadStateManager

        manager = AsyncThreadStateManager(db=None)
        manager.get_thread_async = AsyncMock(return_value=MagicMock())
        key = "C1:10.0"
        for i in range(size):
            manager.enqueue_pending(key, self._queued(f"{i + 1}.0"))
        proc = self._drain_proc(manager)
        client = MagicMock()
        client.edit_dispatch_marker = None          # not the edit-stale path

        with patch.object(config, "queue_drain_linger_seconds", 0.0):
            await proc._dispatch_pending_batch(self._queued("9.0"), client, key)

        trigger = client.message_handler.call_args.args[0]
        assert trigger.metadata["queue_drained"] is True
        # A batch of one carries the flag and no carried sources — the case the old signal missed.
        if size == 1:
            assert "carried_gate_sources" not in trigger.metadata

    def test_an_ordinary_arrival_carries_no_flag(self):
        assert "queue_drained" not in (self._queued("1.0").metadata or {})

    async def test_a_trigger_whose_metadata_is_none_is_still_stamped(self):
        """The stamp sits AFTER the `if trigger.metadata is None: trigger.metadata = {}` guard, and
        that ordering is the whole reason it is where it is. Stamped any earlier — beside
        `carried_gate_sources`, the obvious home — a trigger arriving without metadata would need
        an isinstance check that silently skips it, and the drain would go back to paying the full
        debounce with nothing failing to say so."""
        from thread_manager import AsyncThreadStateManager

        manager = AsyncThreadStateManager(db=None)
        manager.get_thread_async = AsyncMock(return_value=MagicMock())
        key = "C1:10.0"
        queued = self._queued("1.0")
        queued.metadata = None                  # __post_init__ fills it; something later cleared it
        manager.enqueue_pending(key, queued)
        proc = self._drain_proc(manager)
        client = MagicMock()
        client.edit_dispatch_marker = None

        with patch.object(config, "queue_drain_linger_seconds", 0.0):
            await proc._dispatch_pending_batch(self._queued("9.0"), client, key)

        trigger = client.message_handler.call_args.args[0]
        assert trigger.metadata["queue_drained"] is True

    # -- the read, at the real call site

    async def _gate_verdict(self, monkeypatch, **meta):
        """main.py's own gate call, driven for real: the flag is stamped in one module and read in
        another, and a rename on either side would leave every test above green."""
        for name in ("finish_attempt", "gate_decision", "gate_declined"):
            monkeypatch.setattr(participation_telemetry, name, lambda *a, **kw: None)
        app, client = _gate_app(None, GateEvaluation(decision=WakeDecision(wake=True),
                                                     sources=(_source(),)))
        await app._gate_verdict(_gate_msg(**meta), client)
        return app.participation_engine.evaluate.await_args.kwargs

    async def test_the_call_site_passes_the_flag_through_on_a_drained_turn(self, monkeypatch):
        kwargs = await self._gate_verdict(monkeypatch, queue_drained=True)
        assert kwargs["queue_drained"] is True

    async def test_an_ordinary_turn_reaches_the_gate_with_the_flag_false(self, monkeypatch):
        """Nothing leaks. A turn that never went near the queue is judged exactly as before, and
        `False` specifically — not a missing key the engine has to default for."""
        kwargs = await self._gate_verdict(monkeypatch)
        assert kwargs["queue_drained"] is False

    # -- the wait

    async def test_a_drained_redispatch_does_not_sleep(self, monkeypatch):
        monkeypatch.setattr(config, "participation_debounce_seconds", 30, raising=False)
        engine, _ = _engine(True)
        # Warm, so the zero wait is the DRAIN's doing and not W5c's — a drain lands in a stream
        # that has just been busy, which is the only reason the queue had anything in it.
        _warm(engine)
        with patch("message_processor.participation.asyncio.sleep", new=AsyncMock()) as slept:
            ev = await _evaluate(engine, queue_drained=True)
        slept.assert_not_awaited()
        assert ev.decision == WakeDecision(wake=True)

    async def test_an_ordinary_arrival_still_waits_the_configured_debounce(self, monkeypatch):
        monkeypatch.setattr(config, "participation_debounce_seconds", 30, raising=False)
        engine, _ = _engine(True)
        _warm(engine)                                    # W5c: a WARM stream still waits
        with patch("message_processor.participation.asyncio.sleep", new=AsyncMock()) as slept:
            await _evaluate(engine)
        slept.assert_awaited_once_with(30)

    def test_the_flag_defaults_false(self):
        import inspect
        from message_processor.participation import ParticipationEngine as _Engine

        assert inspect.signature(_Engine.evaluate).parameters["queue_drained"].default is False

    # -- and nothing else changes

    async def test_the_drained_cohort_is_judged_exactly_as_any_other(self, monkeypatch):
        """Only the sleep is skipped. The batch the drain carried still reaches the classifier
        whole, in order, and the verdict still comes back as a decision."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 30, raising=False)
        engine, llm = _engine(True)
        carried = [_source(ts="1.0", text="a"), _source(ts="2.0", text="b")]
        with patch("message_processor.participation.asyncio.sleep", new=AsyncMock()):
            ev = await _evaluate(engine, ts="3.0", text="the trigger",
                                 carried_sources=carried, queue_drained=True)
        assert [s.ts for s in ev.sources] == ["1.0", "2.0", "3.0"]
        assert [s.ts for s in llm.classify_wake.await_args.kwargs["sources"]] == \
            ["1.0", "2.0", "3.0"]

    async def test_a_drained_turn_can_still_be_superseded(self, monkeypatch):
        """Skipping the wait does not skip the supersession check — a newer arrival in the same
        stream still owns the turn, and the drained one still declines rather than answering."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 30, raising=False)
        engine, llm = _engine(True)
        engine.note_arrival("C1", "9.0", None, "U1")     # someone newer is already waiting
        with patch("message_processor.participation.asyncio.sleep", new=AsyncMock()):
            ev = await _evaluate(engine, ts="3.0", queue_drained=True)
        assert ev.decision is None and ev.decline_cause == "superseded"
        llm.classify_wake.assert_not_awaited()


class _Clock:
    """Stands in for `time` inside participation.py, where only `monotonic` is ever read.

    Patched onto the MODULE's name rather than onto the real `time` module, so the event loop's
    own clock is left alone — a frozen loop clock hangs the test instead of failing it.
    """

    def __init__(self):
        self.now = time.monotonic()

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def _clock(monkeypatch):
    from message_processor import participation as participation_module

    clock = _Clock()
    monkeypatch.setattr(participation_module, "time", clock)
    return clock


async def _until(predicate, timeout=2.0):
    """Poll until `predicate()` holds. Returns whether it ever did, so a caller can assert on it
    rather than hang: these tests race a background task against a real debounce."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.001)
    return predicate()


class TestSpeculativeClassification:
    """W5b. The debounce is dead time we already spend, so the classifier runs inside it and the
    verdict is waiting when the window closes — about a second of the gate's wall time, hidden.

    Two invariants make that free rather than clever. The call starts only when the cohort AS IT
    STANDS would earn a classifier call anyway, so nothing that skips the model today starts
    spending on it; and the verdict counts only if the cohort that closes the window is the one it
    judged. Everything else is cancelled and thrown away — one attempt, one decision.
    """

    async def test_the_verdict_is_already_in_hand_when_the_window_closes(self, monkeypatch):
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.4, raising=False)
        engine, llm = _engine(True)
        _warm(engine)
        task = asyncio.ensure_future(_evaluate(engine, ts="1.0"))
        assert await _until(lambda: llm.classify_wake.await_count == 1)
        assert not task.done()                       # asked DURING the window, not after it
        ev = await asyncio.wait_for(task, timeout=5)
        assert ev.decision == WakeDecision(wake=True)
        assert llm.classify_wake.await_count == 1    # and the close spends nothing more

    async def test_a_cold_stream_speculates_on_nothing(self, monkeypatch):
        """The W5b/W5c seam, and the whole of the one-attempt guarantee: with no window to hide
        in there is no speculation to reconcile, so the immediate path is the only path."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 30, raising=False)
        engine, llm = _engine(True)
        with patch("message_processor.participation.asyncio.sleep", new=AsyncMock()) as slept:
            ev = await _evaluate(engine, ts="1.0")
        slept.assert_not_awaited()
        assert ev.decision == WakeDecision(wake=True)
        assert llm.classify_wake.await_count == 1

    async def test_a_drained_redispatch_speculates_on_nothing(self, monkeypatch):
        monkeypatch.setattr(config, "participation_debounce_seconds", 30, raising=False)
        engine, llm = _engine(True)
        _warm(engine)
        with patch("message_processor.participation.asyncio.sleep", new=AsyncMock()):
            ev = await _evaluate(engine, ts="1.0", queue_drained=True)
        assert ev.decision == WakeDecision(wake=True)
        assert llm.classify_wake.await_count == 1

    # -- ineligible at START never gains a call

    async def test_a_captionless_image_cohort_is_never_speculated_on(self, monkeypatch):
        """The rule that keeps this free: a shape that skips the model today must not start
        paying for one because we guessed early. The window is waited in full and nobody is
        asked."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
        engine, llm = _engine(True)
        _warm(engine)
        ev = await _evaluate(engine, ts="1.0", text="", attachments=["photo.png (image)"])
        assert ev.decline_cause == "image_only"
        llm.classify_wake.assert_not_awaited()

    async def test_an_already_edit_superseded_attempt_is_never_speculated_on(self, monkeypatch):
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
        engine, llm = _engine(True)
        _warm(engine)
        engine.supersede("C1", "1.0", None, "U1")     # the edit landed before this attempt slept
        ev = await _evaluate(engine, ts="1.0")
        assert ev.decline_cause == "edit_superseded"
        llm.classify_wake.assert_not_awaited()

    async def test_the_peek_at_the_edit_mark_does_not_consume_it(self, monkeypatch):
        """Eligibility PEEKS. Consuming the mark would let the stale original speak after all —
        the real check at window close would find nothing and judge it an ordinary message."""
        engine, _ = _engine(True)
        key = engine._conv_key("C1", "1.0", None, "U1")
        engine.supersede("C1", "1.0", None, "U1")
        assert engine._is_edit_superseded(key, "1.0") is True
        assert engine._is_edit_superseded(key, "1.0") is True     # still there to be consumed
        assert engine._consume_edit_supersession(key, "1.0") is True
        assert engine._is_edit_superseded(key, "1.0") is False

    async def test_an_attempt_already_superseded_at_the_start_is_never_speculated_on(
            self, monkeypatch):
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
        engine, llm = _engine(True)
        engine.note_arrival("C1", "9.0", None, "U1")   # newer already waiting; also warms the stream
        ev = await _evaluate(engine, ts="3.0")
        assert ev.decline_cause == "superseded"
        llm.classify_wake.assert_not_awaited()

    # -- discarded at COMMIT

    async def test_a_cohort_that_changed_under_the_speculation_is_re_judged(self, monkeypatch):
        """A source enrolling late — its own dispatch delayed by I/O — is the case F5 exists for.
        The speculative verdict is about a burst that no longer exists, so it is dropped and the
        ordinary post-debounce call runs on the cohort that actually closed the window."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.3, raising=False)
        engine, llm = _engine(True)
        _warm(engine)
        task = asyncio.ensure_future(_evaluate(engine, ts="2.0", text="the survivor"))
        assert await _until(lambda: llm.classify_wake.await_count == 1)
        key = engine._conv_key("C1", "2.0", None, "U1")
        engine._enroll_source(key, _source(ts="1.0", text="joined late"))
        ev = await asyncio.wait_for(task, timeout=5)
        assert llm.classify_wake.await_count == 2      # the speculation was thrown away, not used
        judged = llm.classify_wake.await_args.kwargs["sources"]
        assert [s.text for s in judged] == ["joined late", "the survivor"]
        assert [s.ts for s in ev.sources] == ["1.0", "2.0"]
        assert ev.decision == WakeDecision(wake=True)   # ONE decision, from the second call

    async def test_an_edit_payload_mutated_mid_window_discards_the_speculation(self, monkeypatch):
        """The one part of a source that is not frozen. `SourceMessage.edit` is an ordinary dict,
        so comparing the records alone would call two different before/after texts equal and commit
        a verdict the model was never shown. The comparison is a deep value taken at speculation
        start, so the mutation is visible and the cohort is re-judged."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.3, raising=False)
        engine, llm = _engine(True)
        _warm(engine)
        payload = {"old_text": "review", "new_text": "review the Q3 numbers",
                   "already_replied": False}
        key = engine._conv_key("C1", "1.0", None, "U1")
        # An edited sibling already in the cohort — the survivor's own record is rebuilt by
        # `evaluate`, so the mutable payload has to belong to a message that is only carried.
        engine._enroll_source(key, _source(ts="0.9", text="review the Q3 numbers", edit=payload))
        task = asyncio.ensure_future(_evaluate(engine, ts="1.0", text="and the margins?"))
        assert await _until(lambda: llm.classify_wake.await_count == 1)
        payload["new_text"] = "actually, cancel the Q3 review"      # the same dict, new meaning
        ev = await asyncio.wait_for(task, timeout=5)
        assert llm.classify_wake.await_count == 2      # the stale verdict was not committed
        judged = llm.classify_wake.await_args.kwargs["sources"]
        assert judged[0].edit["new_text"] == "actually, cancel the Q3 review"
        assert [s.ts for s in ev.sources] == ["0.9", "1.0"]
        assert ev.decision == WakeDecision(wake=True)

    async def test_an_untouched_edit_payload_still_reuses_the_speculation(self, monkeypatch):
        """The control for the test above: freezing the payload must not make every edit-carrying
        cohort fail its own comparison."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.3, raising=False)
        engine, llm = _engine(True)
        _warm(engine)
        key = engine._conv_key("C1", "1.0", None, "U1")
        engine._enroll_source(key, _source(ts="0.9", text="review the Q3 numbers",
                                           edit={"old_text": "review", "already_replied": False}))
        task = asyncio.ensure_future(_evaluate(engine, ts="1.0", text="and the margins?"))
        assert await _until(lambda: llm.classify_wake.await_count == 1)
        ev = await asyncio.wait_for(task, timeout=5)
        assert llm.classify_wake.await_count == 1
        assert [s.ts for s in ev.sources] == ["0.9", "1.0"]
        assert ev.decision == WakeDecision(wake=True)

    async def test_supersession_cancels_the_speculation_and_nothing_else(self, monkeypatch):
        """`discard_source` stays the only path that withdraws a message. Being superseded cancels
        one API task; the source it belongs to is still enrolled, because the survivor is going to
        answer for it."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.3, raising=False)
        engine, llm = _engine(True)
        _warm(engine)
        task = asyncio.ensure_future(_evaluate(engine, ts="1.0", text="mine"))
        assert await _until(lambda: llm.classify_wake.await_count == 1)
        engine.note_arrival("C1", "2.0", None, "U1")
        ev = await asyncio.wait_for(task, timeout=5)
        assert ev.decline_cause == "superseded" and ev.decision is None
        key = engine._conv_key("C1", "1.0", None, "U1")
        assert list(engine._cohorts[key]) == ["1.0"]   # still enrolled for the survivor
        assert engine._latest[key] == "2.0"
        assert [s.text for s in engine._drain_cohort(key, "2.0")] == ["mine"]

    async def test_a_discarded_speculation_mints_no_decision_row(self, monkeypatch):
        """One attempt, one decision. A speculative call whose verdict nobody uses must leave no
        trace in the ledger — the engine reports the decline and nothing else."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.3, raising=False)
        decisions, declines = [], []
        monkeypatch.setattr(participation_telemetry, "gate_decision",
                            lambda *a, **kw: decisions.append(kw))
        monkeypatch.setattr(participation_telemetry, "gate_declined",
                            lambda *a, **kw: declines.append(kw.get("cause")))
        engine, llm = _engine(True)
        _warm(engine)
        task = asyncio.ensure_future(_evaluate(engine, ts="1.0"))
        assert await _until(lambda: llm.classify_wake.await_count == 1)
        engine.note_arrival("C1", "2.0", None, "U1")
        ev = await asyncio.wait_for(task, timeout=5)
        assert ev.decision is None
        assert decisions == []                 # the verdict that came back was never scored
        assert declines == ["superseded"]      # one attempt, one terminal event

    async def test_a_cancelled_evaluation_takes_its_speculation_with_it(self, monkeypatch):
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.3, raising=False)
        engine, llm = _engine(True)
        _warm(engine)
        task = asyncio.ensure_future(_evaluate(engine, ts="1.0"))
        assert await _until(lambda: llm.classify_wake.await_count == 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert engine._cohorts == {}           # the ordinary cancellation contract, unchanged
        await asyncio.sleep(0)
        assert llm.classify_wake.await_count == 1


class TestOneLiveSpeculationPerStream:
    """The cap (owner, 2026-08-08). A speculative call belongs to the message that started it, and
    the moment a newer message enrolls, that message is superseded — by definition, since
    enrollment is what advances the stream. Leaving the call running spends a whole judgment to
    throw it away at the older attempt's own wake-up, which is how a five-message burst cost five
    calls to use one. Cancelling it at enrollment aborts it within a message of starting.

    What it may touch is exactly one API task. Not the cohort, not `_latest`, not the enrollment
    the survivor is going to answer for.
    """

    async def test_a_newer_arrival_cancels_the_older_attempts_call_in_flight(self, monkeypatch):
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.3, raising=False)
        engine, _, started, completed = _slow_engine()
        _warm(engine)
        older = asyncio.ensure_future(_evaluate(engine, ts="1.0", text="first"))
        assert await _until(lambda: len(started) == 1)
        newer = asyncio.ensure_future(_evaluate(engine, ts="2.0", text="second"))
        assert await _until(lambda: len(started) == 2)

        ra, rb = await asyncio.gather(older, newer)
        assert ra.decline_cause == "superseded" and ra.decision is None
        assert rb.decision == WakeDecision(wake=True)
        assert len(started) == 2 and len(completed) == 1     # the first was aborted, not finished
        # And the cancellation took nothing else with it: the older message is in the cohort the
        # survivor was judged on, which is the whole reason it stays enrolled.
        assert [s.text for s in rb.sources] == ["first", "second"]
        assert [s.text for s in completed[0]] == ["first", "second"]

    async def test_a_five_message_burst_completes_one_call(self, monkeypatch):
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.3, raising=False)
        engine, _, started, completed = _slow_engine()
        _warm(engine)
        tasks = []
        for i in range(1, 6):
            tasks.append(asyncio.ensure_future(_evaluate(engine, ts=f"{i}.0", text=f"m{i}")))
            assert await _until(lambda n=i: len(started) == n)
        results = await asyncio.gather(*tasks)

        assert len(started) == 5 and len(completed) == 1
        decisions = [r for r in results if r.decision is not None]
        assert len(decisions) == 1
        assert [s.text for s in decisions[0].sources] == ["m1", "m2", "m3", "m4", "m5"]

    async def test_the_stream_holds_at_most_one_speculation(self, monkeypatch):
        """Three messages landing back to back, before the loop can dispatch any of the calls: the
        first two speculations are cancelled before they ever run, so they cost NOTHING at all —
        not an aborted request, not a token. Only the survivor's reaches the classifier."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.3, raising=False)
        engine, _, started, completed = _slow_engine()
        _warm(engine)
        key = engine._conv_key("C1", "1.0", None, "U1")
        tasks = [asyncio.ensure_future(_evaluate(engine, ts=f"{i}.0")) for i in range(1, 4)]
        assert await _until(lambda: len(started) == 1)

        assert len(engine._speculations) == 1      # never two live in one conversation
        assert engine._speculations[key].task is not None
        results = await asyncio.gather(*tasks)
        assert len(started) == 1 and len(completed) == 1
        assert len([r for r in results if r.decision is not None]) == 1
        assert engine._speculations == {}          # every attempt removed its own on the way out

    async def test_an_older_arrival_cancels_nothing(self, monkeypatch):
        """Only the stream's newest may cancel. An out-of-order older event is not the survivor,
        and taking the live attempt's call would leave the turn that is going to answer with
        nothing in hand."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.3, raising=False)
        engine, _, started, completed = _slow_engine()
        _warm(engine)
        survivor = asyncio.ensure_future(_evaluate(engine, ts="9.0", text="the newest"))
        assert await _until(lambda: len(started) == 1)
        late = asyncio.ensure_future(_evaluate(engine, ts="2.0", text="delayed by I/O"))
        ev_late, ev = await asyncio.gather(late, survivor)
        assert ev.decision == WakeDecision(wake=True)
        assert ev_late.decline_cause == "superseded"
        # NOTHING was aborted: every call that started also finished. The survivor's speculation
        # ran to completion and was then discarded on its own merits — the late message joined its
        # cohort, so the verdict was about a burst that had changed — and the re-judge is the
        # second completed call. Had the late arrival been allowed to cancel, the first would have
        # finished 0 times instead of 1.
        assert len(started) == 2 and len(completed) == 2
        assert [s.text for s in ev.sources] == ["delayed by I/O", "the newest"]

    async def test_an_attempt_that_inherits_the_stream_back_classifies_normally(self, monkeypatch):
        """The fallback, and the reason a cancelled speculation is a flag rather than a raised
        exception. The newer attempt is withdrawn — cancelled mid-debounce, so `discard_source`
        hands `_latest` back — and the older attempt wakes up as the survivor after all, holding a
        speculation somebody else cancelled. It runs an ordinary post-debounce call and decides
        once: never a missing verdict, never a second decision."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.3, raising=False)
        engine, _, started, completed = _slow_engine()
        _warm(engine)
        key = engine._conv_key("C1", "1.0", None, "U1")
        older = asyncio.ensure_future(_evaluate(engine, ts="1.0", text="mine"))
        assert await _until(lambda: len(started) == 1)
        newer = asyncio.ensure_future(_evaluate(engine, ts="2.0", text="theirs"))
        assert await _until(lambda: len(started) == 2)

        newer.cancel()                              # withdrawn: discard_source hands the stream back
        with pytest.raises(asyncio.CancelledError):
            await newer
        assert engine._latest[key] == "1.0"

        ev = await asyncio.wait_for(older, timeout=5)
        assert ev.decline_cause is None
        assert ev.decision == WakeDecision(wake=True)          # one decision, from a fresh call
        assert [s.text for s in ev.sources] == ["mine"]
        assert [s.text for s in completed[-1]] == ["mine"]
        assert engine._speculations == {}


class TestAdaptiveDebounce:
    """W5c. The debounce collects a burst, and a burst is — by the debounce's own definition —
    messages inside one window of each other. A stream nobody has spoken in for that long has no
    burst to collect, so the wait buys nothing but three seconds of someone watching a quiet
    channel. No new constant: the activity window IS `participation_debounce_seconds`.
    """

    async def test_a_cold_stream_is_judged_immediately(self, monkeypatch):
        monkeypatch.setattr(config, "participation_debounce_seconds", 30, raising=False)
        engine, _ = _engine(True)
        with patch("message_processor.participation.asyncio.sleep", new=AsyncMock()) as slept:
            ev = await _evaluate(engine)
        slept.assert_not_awaited()
        assert ev.decision == WakeDecision(wake=True)

    async def test_a_warm_stream_keeps_the_trailing_wait(self, monkeypatch):
        monkeypatch.setattr(config, "participation_debounce_seconds", 30, raising=False)
        engine, _ = _engine(True)
        _warm(engine)
        with patch("message_processor.participation.asyncio.sleep", new=AsyncMock()) as slept:
            await _evaluate(engine)
        slept.assert_awaited_once_with(30)

    async def test_warmth_expires_with_the_window(self, monkeypatch):
        """The window is the debounce, both ways: an arrival older than one window is not a burst
        this message belongs to, and the stream is cold again."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
        engine, _ = _engine(True)
        _warm(engine)
        await asyncio.sleep(0.12)              # longer ago than one window: warm, then not
        with patch("message_processor.participation.asyncio.sleep", new=AsyncMock()) as slept:
            await _evaluate(engine)
        slept.assert_not_awaited()

    async def test_a_second_sender_at_top_level_is_its_own_stream(self, monkeypatch):
        """Coldness is per `_conv_key`, like everything else here: one person talking does not
        make another person's unrelated top-level question part of a burst."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 30, raising=False)
        engine, _ = _engine(True)
        _warm(engine, sender_id="U1")
        with patch("message_processor.participation.asyncio.sleep", new=AsyncMock()) as slept:
            await _evaluate(engine, ts="1.0", sender_id="U2")
        slept.assert_not_awaited()

    # -- note_arrival's contract

    def test_note_arrival_returns_the_arrival_before_it(self):
        engine, _ = _engine(True)
        first = engine.note_arrival("C1", "1.0", None, "U1")
        assert first is not None and first.prior is None       # first one this stream has seen
        second = engine.note_arrival("C1", "2.0", None, "U1")
        assert second is not None and second.prior is not None
        assert second.prior.ts == "1.0"

    def test_note_arrival_is_idempotent_for_one_message(self):
        """It is called twice for every message — at gate entry and inside `evaluate` — and both
        callers have to be told the same thing about the same message. A reading taken after the
        record was written would report the message's own arrival and call every stream warm."""
        engine, _ = _engine(True)
        engine.note_arrival("C1", "1.0", None, "U1")
        first = engine.note_arrival("C1", "2.0", None, "U1")
        second = engine.note_arrival("C1", "2.0", None, "U1")
        assert first is not None and second is not None
        assert first.prior is not None and second.prior is not None
        assert first.prior.ts == second.prior.ts == "1.0"
        assert first.prior.at == second.prior.at        # not re-timed, either
        assert first.marker.at == second.marker.at

    def test_a_message_arriving_between_the_two_calls_changes_neither_of_them(self):
        """The double call is not two calls in a row — real messages land between them.

        A is noted at gate entry, B arrives while A's turn is loading channel steering, and then A
        notes itself again inside `evaluate`. That second call must not rewind the stream to A: B
        is the newer message, its arrival time is what the NEXT message will be judged against, and
        overwriting it with A's stale second look is how a live burst reads as cold.
        """
        engine, _ = _engine(True)
        key = engine._conv_key("C1", "1.0", None, "U1")
        a1 = engine.note_arrival("C1", "1.0", None, "U1")      # gate entry for A
        b1 = engine.note_arrival("C1", "2.0", None, "U1")      # B arrives mid-turn
        recorded = engine._activity[key]
        a2 = engine.note_arrival("C1", "1.0", None, "U1")      # A again, inside evaluate

        assert engine._activity[key] is recorded               # B is still the stream's marker…
        assert engine._activity[key].marker.ts == "2.0"
        assert engine._activity[key].marker.at == recorded.marker.at    # …and it was not re-timed
        # A's second look cannot reconstruct A's own moment, so it answers the safe way: warm,
        # which keeps the wait. A wrong "warm" costs one debounce; a wrong "cold" splits a burst.
        assert a2 is not None and a2.prior is not None and a2.prior.ts == "2.0"
        # …and the exact answer is still available, because the caller kept it from the first call.
        assert a1 is not None and a1.prior is None
        assert b1 is not None and b1.prior is not None and b1.prior.ts == "1.0"

    async def test_the_carried_arrival_decides_coldness_not_the_map(self, monkeypatch):
        """The production shape of the same interleaving: A's turn skips the wait because A was
        cold when A arrived, no matter what has happened to the stream — or the clock — since."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 30, raising=False)
        clock = _clock(monkeypatch)
        engine, _ = _engine(True)
        a1 = engine.note_arrival("C1", "1.0", None, "U1")      # cold: nothing before it
        engine.note_arrival("C1", "2.0", None, "U1")           # B lands during A's steering load
        clock.advance(120)                                     # ...and that load was slow
        with patch("message_processor.participation.asyncio.sleep", new=AsyncMock()) as slept:
            await _evaluate(engine, ts="1.0", arrival=a1)
        slept.assert_not_awaited()

    async def test_the_carried_arrival_also_keeps_a_wait_the_map_would_have_dropped(
            self, monkeypatch):
        """The other direction, and the two ways it can be lost: A arrived a moment after B, so A
        waits — even though the LRU threw the whole stream away while A's turn was loading, and
        even though that load took longer than the window itself.

        The second half is why coldness is the gap between the two ARRIVALS rather than the gap
        between the predecessor and now. Clocking from now, a slow steering read would age a
        genuine burst into a cold stream and split it, which is exactly what carrying the record
        is supposed to prevent."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 30, raising=False)
        clock = _clock(monkeypatch)
        engine, _ = _engine(True)
        _warm(engine)
        a1 = engine.note_arrival("C1", "1.0", None, "U1")
        monkeypatch.setattr(config, "participation_activity_lru_max", 1, raising=False)
        engine.note_arrival("C9", "5.0", None, "U9")           # evicts C1's stream entirely
        assert engine._conv_key("C1", "1.0", None, "U1") not in engine._activity
        clock.advance(120)                                     # four windows of steering I/O
        with patch("message_processor.participation.asyncio.sleep", new=AsyncMock()) as slept:
            await _evaluate(engine, ts="1.0", arrival=a1)
        slept.assert_awaited_once_with(30)

    def test_a_repeated_call_mutates_the_map_in_no_way(self):
        """Not the record, not its timing, not even LRU recency. Nudging recency on a doubled
        call would let a stream that spoke once outrank one that is speaking now, and eviction
        order is the only thing standing between the map and unbounded growth."""
        engine, _ = _engine(True)
        k1 = engine._conv_key("C1", "1.0", None, "U1")
        engine.note_arrival("C1", "1.0", None, "U1")
        engine.note_arrival("C2", "1.0", None, "U2")
        recorded = engine._activity[k1]
        order = list(engine._activity)
        again = engine.note_arrival("C1", "1.0", None, "U1")
        assert again is recorded                    # the stored answer, not a rebuilt one
        assert engine._activity[k1] is recorded
        assert list(engine._activity) == order      # k1 is still the next one out

    def test_a_stale_second_call_cannot_reinsert_an_evicted_stream(self, monkeypatch):
        """The nastier half of the same rule. A's stream is evicted while A's turn loads, and A
        then notes itself again inside `evaluate`. Reinserting there would re-time a stream from a
        message that is no longer its newest AND evict a live stream to make room."""
        engine, _ = _engine(True)
        k9 = engine._conv_key("C9", "5.0", None, "U9")
        engine.note_arrival("C1", "1.0", None, "U1")           # A's gate entry
        engine.note_arrival("C1", "2.0", None, "U1")           # B, newer, same stream
        monkeypatch.setattr(config, "participation_activity_lru_max", 1, raising=False)
        engine.note_arrival("C9", "5.0", None, "U9")           # evicts A's stream
        assert list(engine._activity) == [k9]

        stale = engine.note_arrival("C1", "1.0", None, "U1")   # A again, inside evaluate
        assert list(engine._activity) == [k9]                  # nothing displaced, nothing revived
        assert stale is not None and stale.prior is not None
        assert stale.prior.ts == "2.0"                         # warm: B is out there

    async def test_the_gate_hands_evaluate_the_arrival_it_noted(self, monkeypatch):
        """The wiring, at the real call site: noted in one place and read in another, so a rename
        on either side would leave every test above green."""
        for name in ("finish_attempt", "gate_decision", "gate_declined"):
            monkeypatch.setattr(participation_telemetry, name, lambda *a, **kw: None)
        app, client = _gate_app(None, GateEvaluation(decision=WakeDecision(wake=True),
                                                     sources=(_source(),)))
        await app._gate_verdict(_gate_msg(), client)
        kwargs = app.participation_engine.evaluate.await_args.kwargs
        assert kwargs["arrival"] is app.participation_engine.note_arrival.return_value

    def test_a_missing_channel_or_ts_notes_nothing(self):
        engine, _ = _engine(True)
        assert engine.note_arrival("", "1.0") is None
        assert engine.note_arrival("C1", None) is None
        assert engine._activity == {}

    # -- the activity map is its own structure

    def test_a_drain_never_erases_the_activity_record(self):
        """The reason this is not folded into `_cohorts`: a drain is precisely when a stream is at
        its busiest, and forgetting it there would make the next message of a live burst look
        cold."""
        engine, _ = _engine(True)
        key = engine._conv_key("C1", "1.0", None, "U1")
        engine.note_arrival("C1", "1.0", None, "U1")
        engine._enroll_source(key, _source(ts="1.0"))
        engine._drain_cohort(key, "1.0")
        assert engine._cohorts == {}
        assert engine._activity[key].marker.ts == "1.0"

    def test_a_discarded_source_never_erases_the_activity_record(self):
        engine, _ = _engine(True)
        key = engine._conv_key("C1", "1.0", None, "U1")
        engine.note_arrival("C1", "1.0", None, "U1")
        engine._enroll_source(key, _source(ts="1.0"))
        engine.discard_source("C1", "1.0", None, "U1")
        assert engine._cohorts == {} and key not in engine._latest
        assert engine._activity[key].marker.ts == "1.0"

    def test_the_activity_map_is_bounded(self, monkeypatch):
        """Unlike the cohort map, this one holds timestamps and never a message, so evicting the
        oldest key can lose nothing anybody said — the worst it costs is one extra debounce on a
        stream nothing has touched in a thousand conversations."""
        monkeypatch.setattr(config, "participation_activity_lru_max", 4, raising=False)
        engine, _ = _engine(True)
        for i in range(20):
            engine.note_arrival("C1", "1.0", None, f"U{i}")
        assert len(engine._activity) == 4
        assert list(engine._activity) == [engine._conv_key("C1", "1.0", None, f"U{i}")
                                          for i in range(16, 20)]

    def test_the_default_bound_is_the_configured_one(self):
        from message_processor.participation import _activity_lru_max

        assert config.participation_activity_lru_max == 1024
        assert _activity_lru_max() == 1024


class TestImageOnlyIsImagesOnly:
    """A wordless PDF or spreadsheet is a document somebody may well want read — the responder can
    open it, and often should. Treating "no caption" as "nothing to do" for those skipped BOTH
    models on exactly the material this bot is best at."""

    async def test_a_captionless_document_still_reaches_the_classifier(self):
        engine, llm = _engine(True)
        ev = await _evaluate(engine, text="", attachments=["q3-forecast.csv (file)"])
        assert ev.decline_cause is None and ev.decision.wake is True
        llm.classify_wake.assert_awaited_once()

    async def test_a_mixed_captionless_upload_reaches_the_classifier(self):
        engine, llm = _engine(True)
        ev = await _evaluate(engine, text="",
                             attachments=["chart.png (image)", "notes.pdf (file)"])
        assert ev.decline_cause is None
        llm.classify_wake.assert_awaited_once()

    async def test_captionless_images_alone_still_decline(self):
        engine, llm = _engine(True)
        ev = await _evaluate(engine, text="",
                             attachments=["a.png (image)", "b.jpg (image)"])
        assert ev.decline_cause == "image_only"
        llm.classify_wake.assert_not_awaited()

    def test_one_definition_of_the_descriptor_format(self):
        # The Slack facade builds these strings and the gate classifies on them. Two formats
        # invented at each end is how "captionless image" quietly starts matching a spreadsheet.
        from message_processor.participation import describe_attachment, is_image_descriptor
        from slack_client.event_handlers.message_events import _attachment_descriptors

        assert describe_attachment("a.png", "image/png") == "a.png (image)"
        assert describe_attachment("b.pdf", "application/pdf") == "b.pdf (file)"
        assert is_image_descriptor("a.png (image)") and not is_image_descriptor("b.pdf (file)")
        assert _attachment_descriptors([{"name": "a.png", "mimetype": "image/png"}]) == \
            (describe_attachment("a.png", "image/png"),)


class TestLevelResolutionIsQuietWhenConfused:
    def test_a_present_but_unrecognised_level_does_not_escalate(self):
        from message_processor.participation import resolve_participation_level

        # `judicious` with a legacy auto_respond beside it used to fall through to the mode and
        # resolve to `on` — turning a setting we cannot read into the most talkative one available.
        assert resolve_participation_level(
            {"participation_level": "judicious", "response_mode": "auto_respond"}) == "mentions_only"
        assert resolve_participation_level({"participation_level": "banana"}) == "mentions_only"

    def test_an_absent_level_still_falls_back_to_the_legacy_mode(self):
        from message_processor.participation import resolve_participation_level

        assert resolve_participation_level({"response_mode": "auto_respond"}) == "on"
        assert resolve_participation_level({"participation_level": "", "response_mode": "off"}) == "off"

    def test_the_legal_levels_resolve_to_themselves(self):
        from message_processor.participation import resolve_participation_level

        for level in ("off", "mentions_only", "on"):
            assert resolve_participation_level({"participation_level": level}) == level

    def test_a_retired_level_from_an_old_modal_is_normalized_not_stored(self):
        from message_processor.participation import normalize_legacy_level

        assert normalize_legacy_level("judicious") == "on"
        assert normalize_legacy_level("active") == "on"
        assert normalize_legacy_level("inherit") == "inherit"
        assert normalize_legacy_level("off") == "off"
        assert normalize_legacy_level(None) is None


class TestIncompleteResponseIsNoDecision:
    async def test_an_incomplete_response_is_declined_even_when_it_parses(self):
        # It got partway through and the budget ran out. Whatever text is there is whatever had
        # been emitted when the lights went out — not an answer.
        spy = _WakeSpy('{"wake": true}')
        item = SimpleNamespace(content=[SimpleNamespace(text='{"wake": true}')])

        async def _incomplete(*a, **k):
            spy.params = k
            return SimpleNamespace(
                output=[item], status="incomplete",
                incomplete_details=SimpleNamespace(reason="max_output_tokens"))

        spy._safe_api_call = _incomplete
        assert await _classify(spy) is None

    async def test_a_completed_response_is_honoured(self):
        spy = _WakeSpy('{"wake": true}')
        real = spy._safe_api_call

        async def _completed(*a, **k):
            resp = await real(*a, **k)
            resp.status = "completed"
            return resp

        spy._safe_api_call = _completed
        assert await _classify(spy) is True


class TestCancellationClearsTheCohort:
    async def test_a_cancelled_debounce_leaves_nothing_enrolled(self, monkeypatch):
        # The cohort map is deliberately unbounded — eviction there is a silent message-loss path —
        # so a cancelled stream has to clear itself, or its entry stays forever AND gets swept into
        # some later survivor's cohort as a message from the distant past.
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.2, raising=False)
        engine, _ = _engine(True)
        _warm(engine)                   # W5c: only a warm stream has a debounce to cancel
        task = asyncio.ensure_future(_evaluate(engine, ts="1.0"))
        await asyncio.sleep(0)          # let it enrol and reach the sleep
        assert engine._cohorts != {}
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert engine._cohorts == {}


class TestCancellationWithdrawsOnlyItself:
    """A cohort is shared by every message in a stream, and its other members are live evaluations
    asleep in their own debounce. Clearing the conversation on cancellation loses messages in both
    directions, and neither loss is visible afterwards."""

    async def test_cancelling_the_newest_leaves_the_older_sleeper_able_to_survive(self, monkeypatch):
        # The nastier direction. Dropping the bucket here would delete the sleeper's source AND
        # leave `_latest` naming a message that no longer exists, so the sleeper would wake, find
        # itself superseded by a ghost, and return nothing. The whole stream evaporates.
        # A debounce short enough that the survivor finishes inside the test, long enough that
        # both tasks are demonstrably asleep in it when the cancellation lands.
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.2, raising=False)
        engine, llm = _engine(True)
        _warm(engine)
        older = asyncio.ensure_future(_evaluate(engine, ts="1.0", text="the first thing I said"))
        await asyncio.sleep(0)
        newer = asyncio.ensure_future(_evaluate(engine, ts="2.0", text="and the second"))
        await asyncio.sleep(0)

        newer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await newer

        # The older one is now the survivor: still enrolled, and the debounce marker was handed on.
        key = engine._conv_key("C1", "1.0", None, "U1")
        assert list(engine._cohorts[key]) == ["1.0"]
        assert engine._latest[key] == "1.0"

        ev = await asyncio.wait_for(older, timeout=5)
        assert ev.decline_cause is None
        assert [s.text for s in ev.sources] == ["the first thing I said"]

    async def test_cancelling_an_older_one_leaves_the_live_survivor_whole(self, monkeypatch):
        # The other direction: the newer evaluation is the survivor and is going to answer for the
        # stream. Dropping the bucket would take its sources with it.
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.2, raising=False)
        engine, llm = _engine(True)
        _warm(engine)
        older = asyncio.ensure_future(_evaluate(engine, ts="1.0", text="mine"))
        await asyncio.sleep(0)
        newer = asyncio.ensure_future(_evaluate(engine, ts="2.0", text="theirs"))
        await asyncio.sleep(0)

        older.cancel()
        with pytest.raises(asyncio.CancelledError):
            await older

        key = engine._conv_key("C1", "2.0", None, "U1")
        assert list(engine._cohorts[key]) == ["2.0"]
        assert engine._latest[key] == "2.0"        # untouched: we were never the marker

        ev = await asyncio.wait_for(newer, timeout=5)
        assert ev.decision.wake is True
        assert [s.text for s in ev.sources] == ["theirs"]

    async def test_the_last_one_out_clears_the_marker_too(self, monkeypatch):
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.2, raising=False)
        engine, _ = _engine(True)
        _warm(engine)
        task = asyncio.ensure_future(_evaluate(engine, ts="1.0"))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        key = engine._conv_key("C1", "1.0", None, "U1")
        assert key not in engine._cohorts and key not in engine._latest


class TestAbsorbedObligationsTravelWithTheAnswer:
    """Clearing the gate requirement is only half of what an owed answer carries."""

    def _msg(self, ts, **meta):
        md = {"ts": ts, "gate_required": True, "gate_woke": False, "silence_capable": True}
        md.update(meta)
        return Message(text=f"msg {ts}", user_id="U1", channel_id="C1", thread_id="10.0",
                       metadata=md)

    def test_an_absorbed_mention_makes_the_turn_owe_words(self):
        from message_processor import routing_facts

        # The trigger is ambient, so it arrived silence-capable. Leaving it that way lets the
        # responder end with no_response_needed on a batch containing a message somebody addressed
        # to us directly — the original bug wearing a different hat.
        mention = self._msg("1.0", gate_required=False, silence_capable=False)
        trigger = self._msg("2.0")
        assert routing_facts.absorb_owed_answer(trigger, [mention]) is True
        assert trigger.metadata["gate_required"] is False
        assert trigger.metadata["silence_capable"] is False

    def test_an_absorbed_woken_ambient_message_may_still_end_in_silence(self):
        from message_processor import routing_facts

        # Never addressed to us. The responder sees the whole conversation where the gate saw one
        # moment, and is entitled to conclude there is nothing worth adding.
        woken = self._msg("1.0", gate_woke=True)
        trigger = self._msg("2.0")
        assert routing_facts.absorb_owed_answer(trigger, [woken]) is True
        assert trigger.metadata["gate_required"] is False
        assert trigger.metadata["silence_capable"] is True

    def test_a_mixed_batch_takes_the_stronger_obligation(self):
        from message_processor import routing_facts

        trigger = self._msg("3.0")
        assert routing_facts.absorb_owed_answer(
            trigger, [self._msg("1.0", gate_woke=True),
                      self._msg("2.0", gate_required=False, silence_capable=False)]) is True
        assert trigger.metadata["silence_capable"] is False

    def test_an_absorbed_thread_continuation_keeps_the_right_to_stay_silent(self):
        from message_processor import routing_facts

        # A thread continuation (strict 1:1, or membership in an `on` channel) is UNGATED but
        # silence-capable on purpose: no gate ran,
        # so the responder is the only decider and may decide there is nothing to say. Reading
        # "ungated" as "addressed" would take that decision away — the same class of mistake as
        # the blocker it sits next to, in the opposite direction: manufacturing words rather than
        # losing them.
        continuation = self._msg("1.0", gate_required=False, silence_capable=True)
        assert routing_facts.owes_answer(continuation) is True     # it is owed a TURN
        assert routing_facts.owes_words(continuation) is False     # but not words

        trigger = self._msg("2.0")
        assert routing_facts.absorb_owed_answer(trigger, [continuation]) is True
        assert trigger.metadata["gate_required"] is False          # the turn still runs
        assert trigger.metadata["silence_capable"] is True         # and may still end quietly

    def test_an_unstamped_message_never_claims_to_owe_words(self):
        from message_processor import routing_facts

        bare = Message(text="x", user_id="U1", channel_id="C1", thread_id="1.0", metadata={})
        assert routing_facts.owes_words(bare) is False

    def test_owes_words_is_narrower_than_owes_answer(self):
        from message_processor import routing_facts

        woken = self._msg("1.0", gate_woke=True)
        assert routing_facts.owes_answer(woken) is True
        assert routing_facts.owes_words(woken) is False

    def test_the_closed_attempt_is_detached_so_the_turn_can_close_its_own(self):
        # Left in place, the closed attempt attributes this turn's reactions to it AND makes
        # finish_attempt a no-op, so the turn the room actually saw is missing from the ledger.
        trigger = self._msg("2.0")
        participation_telemetry.begin_attempt(trigger)
        assert participation_telemetry.finish_attempt(trigger, "queued") is True
        assert participation_telemetry.attempt_id_for(trigger) is not None

        detached = participation_telemetry.detach_attempt(trigger)
        assert detached
        assert participation_telemetry.attempt_id_for(trigger) is None
        assert participation_telemetry.CLOSED_KEY not in trigger.metadata
        assert participation_telemetry.PARENT_KEY not in trigger.metadata
        # An ungated route writes no terminal, exactly like a DM or an @mention.
        assert participation_telemetry.finish_attempt(trigger, "text") is False

    def test_detaching_nothing_is_a_noop(self):
        assert participation_telemetry.detach_attempt(self._msg("1.0")) is None
        assert participation_telemetry.detach_attempt(None) is None

    async def test_the_drain_detaches_and_stages_the_closed_attempt(self):
        from message_processor import routing_facts

        mention = self._msg("1.0", gate_required=False)
        trigger = self._msg("2.0")
        closed = participation_telemetry.begin_attempt(trigger)
        participation_telemetry.finish_attempt(trigger, "queued")

        # What _dispatch_pending_batch does once absorption applies.
        assert routing_facts.absorb_owed_answer(trigger, [mention]) is True
        detached = participation_telemetry.detach_attempt(trigger)
        participation_telemetry.stage_queue_links(trigger, [detached])

        assert detached == closed
        assert closed in trigger.metadata[participation_telemetry.BATCHED_SOURCES_KEY]
