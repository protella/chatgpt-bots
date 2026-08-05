"""Stale reconsideration — the runner and its delivery sites (STALE_RECONSIDERATION §4b–§4f, §8).

Three layers, mirroring the mechanism:

* THE RUNNER, driven directly over a real lease, a real pinned `ChannelTurnContext` and a real
  serialized fresh stream, with the snapshot seam and the decision wrapper substituted. This is
  where the §4e loop, the §4f matrix, the reviewed-through extraction and the telemetry
  single-owner rules are pinned.
* THE STREAMING SITES, driven through the REAL `_handle_streaming_text_response` against a fake
  Slack that mirrors the transport's lease semantics — the buffered/direct final post and the
  final-correction branch, each × {post-asis, post-revised, force, skip, delivery_failed}.
* MAIN.PY, driven through the real `handle_message` — the non-streaming site, the terminal
  catch's single-owner marker, and the artifact/sandbox rescue gates.

Every OpenAI/Slack boundary is mocked; every mock stream terminates and yields real strings
(CLAUDE.md pitfall 6).
"""
from __future__ import annotations

import ast
import asyncio
import inspect
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import message_processor.reconsideration as reconsideration
from base_client import Message, Response
from config import config
from message_processor import participation_telemetry
from message_processor.channel_request import to_input_items
from message_processor.reconsideration import (RECONSIDER_FUSE_PASSES,
                                               build_reconsideration_request, draft_fence,
                                               intercept_stale_send, reconsider_stale_draft,
                                               reconsideration_item, reviewed_through_map,
                                               select_reconsideration_model,
                                               suppressing_ts_present)
from message_processor.stale_send_guard import (COMMITTED, PENDING, ConversationWatermarks,
                                                StaleSendSuppressed)
from message_processor.turn_runtime import TurnRuntime
from openai_client.api.responses import (ReconsiderationDecision,
                                         ReconsiderationDecisionError)
from prompts import RECONSIDERATION_INSTRUCTION
from tests.unit.channel_turn_harness import build_stream, normalized, pin_channel_turn

CH = "C1"
TRIGGER_TS = "10.0"
NEWER_TS = "11.0"


# =============================================================================== fakes


def _msg(ts: str = TRIGGER_TS, channel: str = CH, thread: Optional[str] = None,
         sender: Optional[str] = "U1") -> SimpleNamespace:
    meta: Dict[str, Any] = {"ts": ts}
    if sender:
        meta["sender_id"] = sender
    return SimpleNamespace(channel_id=channel, thread_id=thread or ts, user_id=sender,
                           text="hi", attachments=None, metadata=meta)


class FakeDecider:
    """Stands in for `create_reconsideration_decision`, honoring its attempt-open contract:
    it opens the ModelAttempt itself and invokes `on_attempt_open(seq)` BEFORE producing the
    scripted decision (or raising the scripted exception)."""

    def __init__(self, script: List[Any]):
        self.script = list(script)
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, *, input_items, instructions=None, model=None,
                       reasoning_effort=None, verbosity=None, max_output_tokens=None,
                       temperature=None, prompt_cache_key=None, attempt_sink=None,
                       on_attempt_open=None) -> ReconsiderationDecision:
        self.calls.append(dict(
            input_items=input_items, instructions=instructions, model=model,
            reasoning_effort=reasoning_effort, verbosity=verbosity,
            max_output_tokens=max_output_tokens, temperature=temperature,
            prompt_cache_key=prompt_cache_key))
        attempt = attempt_sink.open(model) if attempt_sink is not None else None
        seq = getattr(attempt, "attempt_seq", None)
        if on_attempt_open is not None:
            on_attempt_open(seq)
        step = self.script.pop(0)
        if isinstance(step, BaseException):
            if attempt_sink is not None:
                attempt_sink.close(attempt, status="error",
                                   detail=getattr(step, "detail", type(step).__name__))
            raise step
        if attempt_sink is not None:
            attempt_sink.close(attempt, status="ok")
        return step


def _decision(decision: str, text: Optional[str] = None) -> ReconsiderationDecision:
    return ReconsiderationDecision(decision=decision, text=text)


def _processor(decider: FakeDecider) -> MagicMock:
    processor = MagicMock()
    processor._get_system_prompt.return_value = "SYSTEM-PROMPT"
    processor._build_time_suffix_context.return_value = "[time: pinned]"
    processor._build_generation_inflight_note.return_value = None
    processor._build_research_inflight_note.return_value = None
    processor.db = None
    processor.openai_client = SimpleNamespace(create_reconsideration_decision=decider)
    return processor


def _fresh_stream(*extra_messages, include_newer: bool = True):
    """The rebuilt window: the trigger plus (by default) the suppressing newer reply."""
    messages = [normalized(TRIGGER_TS, "the question", sender_id="U1")]
    if include_newer:
        messages.append(normalized(NEWER_TS, "the racer", sender_id="U2",
                                   thread_root_ts=TRIGGER_TS))
    messages.extend(extra_messages)
    return build_stream(messages, channel_id=CH, team_id="T1")


class _Rig:
    """One suppressed channel turn, ready for the runner."""

    def __init__(self, monkeypatch, *, script: List[Any], draft: str = "the draft answer",
                 fresh_stream=None, sender: Optional[str] = "U1"):
        self.marks = ConversationWatermarks()
        self.message = _msg(sender=sender)
        self.lease = self.marks.begin_turn(self.message)
        self.turn = TurnRuntime.for_message(self.message, channel_post_allowed=False)
        self.turn.send_lease = self.lease
        self.turn.guard_mode = "buffered"
        self.ctx = pin_channel_turn(self.turn, trigger_ts=TRIGGER_TS,
                                    origin_thread_ts=None, channel_id=CH)
        self.decider = FakeDecider(script)
        self.processor = _processor(self.decider)
        self.draft = draft
        self.delivered: List[str] = []
        self.fixed_stream = fresh_stream
        self.raced: List[str] = []
        self.snapshot_calls: List[Dict[str, Any]] = []

        async def _snapshot(**kwargs):
            """The pure seam, substituted: the rebuilt window reflects everything that has
            raced so far (a real snapshot pins a fresh H), unless a test fixed the stream."""
            self.snapshot_calls.append(kwargs)
            if self.fixed_stream is not None:
                return SimpleNamespace(stream=self.fixed_stream)
            messages = [normalized(TRIGGER_TS, "the question", sender_id="U1")]
            messages += [normalized(ts, f"racer at {ts}", sender_id="U2",
                                    thread_root_ts=TRIGGER_TS) for ts in self.raced]
            return SimpleNamespace(stream=build_stream(messages, channel_id=CH, team_id="T1"))

        monkeypatch.setattr(reconsideration, "build_reconsideration_snapshot", _snapshot)

    def race(self, ts: str = NEWER_TS, thread: str = TRIGGER_TS) -> None:
        """A newer message lands in the trigger's thread scope."""
        self.raced.append(ts)
        self.marks.begin_turn(_msg(ts=ts, thread=thread, sender="U2"))

    def accepting_deliver(self, ts: str = "99.9"):
        async def _deliver(text: str) -> Optional[str]:
            self.lease.authorize("final_post")
            self.delivered.append(text)
            self.lease.commit()
            return ts
        return _deliver

    async def run(self, deliver, suppressed=None):
        exc = suppressed if suppressed is not None else self.suppress()
        return await reconsider_stale_draft(
            processor=self.processor, client=SimpleNamespace(bot_user_id="UBOT"),
            message=self.message, turn=self.turn, lease=self.lease, suppressed=exc,
            draft=self.draft, deliver=deliver)

    def suppress(self) -> StaleSendSuppressed:
        """The initiating refusal, raised by THIS lease exactly as authorize() raises it."""
        try:
            self.lease.authorize("final_post")
        except StaleSendSuppressed as exc:
            return exc
        raise AssertionError("the lease was not suppressed — race() first")


@pytest.fixture
def events(monkeypatch):
    captured: List[Dict[str, Any]] = []

    def _record(event, *, channel_id=None, trigger_ts=None, **fields):
        captured.append({"event": event, "channel_id": channel_id,
                         "trigger_ts": trigger_ts, **fields})

    monkeypatch.setattr(participation_telemetry, "record", _record)
    return captured


def _named(events, name):
    return [e for e in events if e["event"] == name]


@pytest.fixture
def receipt_service(tmp_path, monkeypatch):
    """A real `ReceiptService` over a temp DB, so tests that care about receipts can drive a
    REAL `ReceiptLedger`: a ledger with no service is inert (`active` is False), and a fake that
    "commits" would prove nothing about registration or promotion. Yields the DB so a test can
    read the row state back."""
    from database import DatabaseManager
    from message_processor import outbound_receipts as orx

    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    db = DatabaseManager(platform="slack")
    orx.reset_service()
    orx.install_service(db)
    yield db
    orx.reset_service()
    db.conn.close()


def _real_ledger(owner: str, channel: str = CH):
    from message_processor.outbound_receipts import ReceiptLedger

    return ReceiptLedger(owner, "T1", channel)


# =============================================================================== the runner


@pytest.mark.asyncio
async def test_post_asis_rearms_delivers_and_records(events, monkeypatch):
    rig = _Rig(monkeypatch, script=[_decision("post", None)])
    rig.race()
    exc = rig.suppress()

    ts = await rig.run(rig.accepting_deliver(), suppressed=exc)

    assert ts == "99.9"
    assert rig.delivered == ["the draft answer"]
    assert rig.lease.state == COMMITTED
    facts = rig.turn.reconsider
    assert (facts.outcome, facts.passes, facts.forced) == ("posted_asis", 1, False)
    # Single-owner: the runner emitted the suppression row and marked the exception.
    assert exc.telemetry_recorded is True
    rows = _named(events, "stale_send")
    assert len(rows) == 1
    assert rows[0]["turn_id"] == rig.turn.turn_id
    assert rows[0]["observed_latest_ts"] == NEWER_TS
    assert rows[0]["scope"] == "thread"
    starts = _named(events, "reconsider_start")
    assert [s["pass"] for s in starts] == [1]
    assert starts[0]["scope"] == ["thread", CH, TRIGGER_TS]
    assert starts[0]["observed_latest_ts"] == NEWER_TS
    assert starts[0]["turn_id"] == rig.turn.turn_id
    assert starts[0]["attempt_id"] is None          # ungated channel turn has none; dropped
                                                    # from the wire by the emitter's null rule
    outcomes = _named(events, "reconsider_outcome")
    assert len(outcomes) == 1
    assert (outcomes[0]["outcome"], outcomes[0]["passes"], outcomes[0]["forced"]) == (
        "posted_asis", 1, False)
    # The reconsideration attempt is a NEW ModelAttempt of the SAME turn, fork-reasoned.
    assert [a.fork_reason for a in rig.turn.model_attempts] == ["stale_reconsideration"]


@pytest.mark.asyncio
async def test_post_with_stripped_equal_text_delivers_the_draft_unchanged(events, monkeypatch):
    rig = _Rig(monkeypatch, script=[_decision("post", "  the draft answer \n")])
    rig.race()
    await rig.run(rig.accepting_deliver())
    assert rig.delivered == ["the draft answer"]     # NOT the whitespace variant
    assert rig.turn.reconsider.outcome == "posted_asis"


@pytest.mark.asyncio
async def test_post_revised_delivers_the_revision_and_classifies_against_the_initial(
        events, monkeypatch):
    rig = _Rig(monkeypatch, script=[_decision("post", "a better answer")])
    rig.race()
    ts = await rig.run(rig.accepting_deliver())
    assert ts == "99.9"
    assert rig.delivered == ["a better answer"]
    assert rig.turn.reconsider.outcome == "posted_revised"
    assert _named(events, "reconsider_outcome")[0]["outcome"] == "posted_revised"


@pytest.mark.asyncio
async def test_force_post_waives_the_check_and_records_forced(events, monkeypatch):
    """force_post delivers WITHOUT a rearm: the watermark still holds the unreviewed newer ts,
    so only the waiver lets the send through — and the first commit clears it."""
    rig = _Rig(monkeypatch, script=[_decision("force_post", None)])
    rig.race()
    ts = await rig.run(rig.accepting_deliver())
    assert ts == "99.9"
    assert rig.delivered == ["the draft answer"]
    assert rig.lease.state == COMMITTED
    assert rig.lease._force_waiver is False          # cleared by commit()
    assert (rig.turn.reconsider.outcome, rig.turn.reconsider.forced) == ("posted_asis", True)
    assert _named(events, "reconsider_outcome")[0]["forced"] is True


@pytest.mark.asyncio
async def test_force_revised(events, monkeypatch):
    rig = _Rig(monkeypatch, script=[_decision("force_post", "rewritten")])
    rig.race()
    await rig.run(rig.accepting_deliver())
    assert rig.delivered == ["rewritten"]
    assert (rig.turn.reconsider.outcome, rig.turn.reconsider.forced) == (
        "posted_revised", True)


@pytest.mark.asyncio
async def test_skip_rethrows_the_marked_suppression_and_never_delivers(events, monkeypatch):
    rig = _Rig(monkeypatch, script=[_decision("skip", None)])
    rig.race()
    exc = rig.suppress()
    deliver = AsyncMock()

    with pytest.raises(StaleSendSuppressed) as raised:
        await rig.run(deliver, suppressed=exc)

    assert raised.value is exc                        # the ORIGINAL exception identity
    assert exc.telemetry_recorded is True
    deliver.assert_not_awaited()                      # skip proves no subsequent send
    assert rig.lease.state != COMMITTED
    assert rig.turn.reconsider.outcome == "skipped"
    assert _named(events, "reconsider_outcome")[0]["outcome"] == "skipped"
    assert "forced" not in rig.turn.reconsider.as_payload()


# ------------------------------------------------------------------ the §4f matrix


@pytest.mark.asyncio
async def test_suppressing_ts_absent_from_the_rebuilt_input_fails_closed(events, monkeypatch):
    """§4a: missing, deleted, malformed or filtered — the review cannot claim coverage."""
    rig = _Rig(monkeypatch, script=[_decision("post", None)],
               fresh_stream=_fresh_stream(include_newer=False))
    rig.race()
    exc = rig.suppress()
    with pytest.raises(StaleSendSuppressed) as raised:
        await rig.run(rig.accepting_deliver(), suppressed=exc)
    assert raised.value is exc
    assert (rig.turn.reconsider.outcome, rig.turn.reconsider.error) == (
        "error_dropped", "context_rebuild")
    assert rig.decider.calls == []                    # no model call was spent
    assert _named(events, "reconsider_start") == []
    assert _named(events, "reconsider_outcome")[0]["error"] == "context_rebuild"
    assert _named(events, "reconsider_outcome")[0]["passes"] == 0


@pytest.mark.asyncio
async def test_snapshot_failure_is_context_rebuild(events, monkeypatch):
    rig = _Rig(monkeypatch, script=[_decision("post", None)])
    rig.race()

    async def _boom(**kwargs):
        raise RuntimeError("slack fell over")

    monkeypatch.setattr(reconsideration, "build_reconsideration_snapshot", _boom)
    with pytest.raises(StaleSendSuppressed):
        await rig.run(rig.accepting_deliver())
    assert (rig.turn.reconsider.outcome, rig.turn.reconsider.error) == (
        "error_dropped", "context_rebuild")


@pytest.mark.asyncio
async def test_admission_overflow_refuses_readonly(events, monkeypatch):
    rig = _Rig(monkeypatch, script=[_decision("post", None)])
    rig.race()
    monkeypatch.setattr(
        reconsideration, "estimate_admission",
        lambda **kw: SimpleNamespace(fits=False, total_tokens=2, limit_tokens=1))
    with pytest.raises(StaleSendSuppressed):
        await rig.run(rig.accepting_deliver())
    assert (rig.turn.reconsider.outcome, rig.turn.reconsider.error) == (
        "error_dropped", "admission_overflow")
    assert rig.decider.calls == []


@pytest.mark.asyncio
async def test_an_estimate_whose_result_raises_is_request_build_and_stamps_the_gate(
        events, monkeypatch):
    """§4f: `request_build` covers model selection through assembly AND estimate consumption —
    an estimate OBJECT whose result property raises is classified, never an escape that would
    leave the once-per-turn gate unstamped."""
    class _PoisonedEstimate:
        total_tokens = 1
        limit_tokens = 2

        @property
        def fits(self):
            raise RuntimeError("poisoned estimate result")

    rig = _Rig(monkeypatch, script=[_decision("post", None)])
    rig.race()
    monkeypatch.setattr(reconsideration, "estimate_admission",
                        lambda **kw: _PoisonedEstimate())
    exc = rig.suppress()
    with pytest.raises(StaleSendSuppressed) as raised:
        await rig.run(rig.accepting_deliver(), suppressed=exc)
    assert raised.value is exc
    assert (rig.turn.reconsider.outcome, rig.turn.reconsider.error) == (
        "error_dropped", "request_build")                # gate stamped, subtype named
    assert rig.decider.calls == []                       # no model call was spent
    assert _named(events, "reconsider_outcome")[0]["error"] == "request_build"


@pytest.mark.asyncio
async def test_model_failure_gives_up_with_the_subtype(events, monkeypatch):
    rig = _Rig(monkeypatch, script=[ReconsiderationDecisionError("schema_invalid")])
    rig.race()
    exc = rig.suppress()
    with pytest.raises(StaleSendSuppressed) as raised:
        await rig.run(rig.accepting_deliver(), suppressed=exc)
    assert raised.value is exc
    assert (rig.turn.reconsider.outcome, rig.turn.reconsider.error) == (
        "error_dropped", "model_failure")
    # The pass HAD started: its reconsider_start row exists and `passes` counts it.
    assert [s["pass"] for s in _named(events, "reconsider_start")] == [1]
    assert _named(events, "reconsider_outcome")[0]["passes"] == 1


@pytest.mark.asyncio
async def test_guard_rearm_failure_gives_up(events, monkeypatch):
    """An exception with the right evidence but a FOREIGN lease token can never rearm."""
    rig = _Rig(monkeypatch, script=[_decision("post", None)])
    rig.race()
    real = rig.suppress()
    forged = StaleSendSuppressed(scope=real.scope, last_seen_ts=real.last_seen_ts,
                                 observed_latest_ts=real.observed_latest_ts,
                                 surface=real.surface, lease_token=object())
    with pytest.raises(StaleSendSuppressed) as raised:
        await rig.run(rig.accepting_deliver(), suppressed=forged)
    assert raised.value is forged
    assert (rig.turn.reconsider.outcome, rig.turn.reconsider.error) == (
        "error_dropped", "guard_rearm_failed")
    assert rig.delivered == []


@pytest.mark.asyncio
async def test_delivery_failed_returns_none_with_no_stale_rethrow(events, monkeypatch):
    rig = _Rig(monkeypatch, script=[_decision("post", None)])
    rig.race()
    exc = rig.suppress()

    async def _refusing_deliver(text: str) -> Optional[str]:
        rig.lease.authorize("final_post")
        return None                                  # Slack swallowed the failure

    result = await rig.run(_refusing_deliver, suppressed=exc)

    assert result is None                            # returned to the site, NOT raised
    assert (rig.turn.reconsider.outcome, rig.turn.reconsider.error) == (
        "error_dropped", "delivery_failed")
    assert exc.telemetry_recorded is True
    assert len(_named(events, "stale_send")) == 1    # exactly the one suppression row
    outcome = _named(events, "reconsider_outcome")[0]
    assert (outcome["outcome"], outcome["error"]) == ("error_dropped", "delivery_failed")
    assert "forced" not in rig.turn.reconsider.as_payload()


@pytest.mark.asyncio
async def test_delivery_exception_returns_none_and_cancels_the_force_waiver(
        events, monkeypatch):
    """§4f review r8: a NON-ENUMERATED exception out of `deliver` ends exactly like
    `delivery_failed` — outcome emitted, None returned to the site, no stale rethrow — because
    physical acceptance is UNKNOWN there. And the waiver never survives its delivery: the runner
    cancels it explicitly rather than leaving a live blank cheque on an uncommitted lease."""
    rig = _Rig(monkeypatch, script=[_decision("force_post", None)])
    rig.race()
    exc = rig.suppress()

    async def _raising_deliver(text: str) -> Optional[str]:
        rig.lease.authorize("final_post")             # the waiver let this through
        raise RuntimeError("the transport exploded")

    result = await rig.run(_raising_deliver, suppressed=exc)

    assert result is None                             # returned to the site, NOT raised
    assert (rig.turn.reconsider.outcome, rig.turn.reconsider.error) == (
        "error_dropped", "delivery_exception")
    assert rig.lease._force_waiver is False           # cancel_force_waiver() ran
    assert rig.lease.state != COMMITTED
    assert "forced" not in rig.turn.reconsider.as_payload()
    rows = _named(events, "stale_send")
    assert len(rows) == 1 and exc.telemetry_recorded is True
    outcome = _named(events, "reconsider_outcome")[0]
    assert (outcome["outcome"], outcome["error"]) == ("error_dropped", "delivery_exception")


@pytest.mark.asyncio
async def test_request_assembly_failure_is_request_build_not_context_rebuild(
        events, monkeypatch):
    """§4f review r8: model-selection / profile / assembly / estimation failures are
    `request_build`. A programming failure must not masquerade as a Slack-history failure —
    the snapshot right above it SUCCEEDED."""
    rig = _Rig(monkeypatch, script=[_decision("post", None)])
    rig.race()
    exc = rig.suppress()

    def _boom(**kwargs):
        raise TypeError("assemble_channel_request got an unexpected keyword")

    monkeypatch.setattr(reconsideration, "assemble_channel_request", _boom)
    with pytest.raises(StaleSendSuppressed) as raised:
        await rig.run(rig.accepting_deliver(), suppressed=exc)

    assert raised.value is exc
    assert (rig.turn.reconsider.outcome, rig.turn.reconsider.error) == (
        "error_dropped", "request_build")
    assert len(rig.snapshot_calls) == 1               # the history rebuild was fine
    assert rig.decider.calls == []                    # no model call was spent
    assert _named(events, "reconsider_outcome")[0]["error"] == "request_build"


@pytest.mark.asyncio
async def test_epoch_refusal_after_rearm_is_epoch_invalidated(events, monkeypatch):
    from message_processor.epoch_fence import EpochEffectRefused

    rig = _Rig(monkeypatch, script=[_decision("post", None)])
    rig.race()

    async def _fenced_deliver(text: str) -> Optional[str]:
        rig.lease.authorize("final_post")            # rearm let this through
        raise EpochEffectRefused("stale epoch")

    with pytest.raises(StaleSendSuppressed):
        await rig.run(_fenced_deliver)
    assert (rig.turn.reconsider.outcome, rig.turn.reconsider.error) == (
        "error_dropped", "epoch_invalidated")


@pytest.mark.asyncio
async def test_epoch_refusal_after_force(events, monkeypatch):
    from message_processor.epoch_fence import EpochEffectRefused

    rig = _Rig(monkeypatch, script=[_decision("force_post", None)])
    rig.race()

    async def _fenced_deliver(text: str) -> Optional[str]:
        rig.lease.authorize("final_post")            # waiver let this through
        raise EpochEffectRefused("stale epoch")

    with pytest.raises(StaleSendSuppressed):
        await rig.run(_fenced_deliver)
    assert rig.turn.reconsider.error == "epoch_invalidated"


# ------------------------------------------------------------------ the loop and the fuse


@pytest.mark.asyncio
async def test_a_rerace_becomes_the_next_pass_and_rearms_on_the_newest_exception(
        events, monkeypatch):
    """Pass 1 delivers into a fresh race; pass 2 reviews through the NEWEST ts and lands."""
    rig = _Rig(monkeypatch, script=[_decision("post", None), _decision("post", None)])
    rig.race()

    raced = {"done": False}

    async def _deliver(text: str) -> Optional[str]:
        if not raced["done"]:
            raced["done"] = True
            rig.race(ts="12.0")                      # lands between rearm and the send
        rig.lease.authorize("final_post")
        rig.delivered.append(text)
        rig.lease.commit()
        return "99.9"

    ts = await rig.run(_deliver)

    assert ts == "99.9"
    assert rig.delivered == ["the draft answer"]
    assert len(rig.snapshot_calls) == 2              # one pure snapshot per pass
    assert [s["pass"] for s in _named(events, "reconsider_start")] == [1, 2]
    assert len(_named(events, "stale_send")) == 2    # initiating refusal + the re-race
    outcome = _named(events, "reconsider_outcome")
    assert len(outcome) == 1 and outcome[0]["passes"] == 2
    # The second start row carries the SECOND suppression's evidence.
    assert _named(events, "reconsider_start")[1]["observed_latest_ts"] == "12.0"


@pytest.mark.asyncio
async def test_pass1_revision_then_pass2_rerace_evaluates_the_revised_draft(
        events, monkeypatch):
    rig = _Rig(monkeypatch, script=[_decision("post", "the revised answer"),
                                    _decision("post", None)])
    rig.race()
    raced = {"done": False}

    async def _deliver(text: str) -> Optional[str]:
        if not raced["done"]:
            raced["done"] = True
            rig.race(ts="12.0")
        rig.lease.authorize("final_post")
        rig.delivered.append(text)
        rig.lease.commit()
        return "99.9"

    await rig.run(_deliver)

    # Pass 2 QUOTED the revised draft, delivered it, and classified against the initial.
    pass2_quoted = rig.decider.calls[1]["input_items"][-1]["content"]
    assert "the revised answer" in pass2_quoted
    assert "the draft answer" not in pass2_quoted
    assert rig.delivered == ["the revised answer"]
    assert rig.turn.reconsider.outcome == "posted_revised"


@pytest.mark.asyncio
async def test_the_fuse_fires_only_when_a_sixth_pass_would_begin(events, monkeypatch):
    """Five completed passes are legal; the SIXTH is the malfunction backstop."""
    rig = _Rig(monkeypatch, script=[_decision("post", None)] * RECONSIDER_FUSE_PASSES)
    rig.race()
    race_ts = {"n": 11}

    async def _always_raced(text: str) -> Optional[str]:
        race_ts["n"] += 1
        rig.race(ts=f"{race_ts['n']}.0")
        rig.lease.authorize("final_post")            # always one message behind
        raise AssertionError("unreachable — authorize raised")

    with pytest.raises(StaleSendSuppressed) as raised:
        await rig.run(_always_raced)

    assert rig.turn.reconsider.outcome == "fuse_dropped"
    assert rig.turn.reconsider.passes == RECONSIDER_FUSE_PASSES
    assert [s["pass"] for s in _named(events, "reconsider_start")] == [1, 2, 3, 4, 5]
    # 1 initiating + 5 re-races: every handled suppression got its row.
    assert len(_named(events, "stale_send")) == 6
    # The rethrown suppression is the NEWEST one, marked.
    assert raised.value.observed_latest_ts == "16.0"
    assert raised.value.telemetry_recorded is True
    assert _named(events, "reconsider_outcome")[0]["outcome"] == "fuse_dropped"
    assert rig.decider.calls and len(rig.decider.calls) == RECONSIDER_FUSE_PASSES


@pytest.mark.asyncio
@pytest.mark.parametrize("final_decision,outcome", [("force_post", "posted_asis"),
                                                    ("skip", "skipped")])
async def test_force_and_skip_are_legal_on_pass_five(final_decision, outcome,
                                                     events, monkeypatch):
    rig = _Rig(monkeypatch, script=[_decision("post", None)] * 4
               + [_decision(final_decision, None)])
    rig.race()
    race_ts = {"n": 11}

    async def _deliver(text: str) -> Optional[str]:
        rig.lease.authorize("final_post")
        rig.delivered.append(text)
        rig.lease.commit()
        return "99.9"

    async def _raced_four_times(text: str) -> Optional[str]:
        if race_ts["n"] < 15:
            race_ts["n"] += 1
            rig.race(ts=f"{race_ts['n']}.0")
        return await _deliver(text)

    if final_decision == "skip":
        with pytest.raises(StaleSendSuppressed):
            await rig.run(_raced_four_times)
    else:
        assert await rig.run(_raced_four_times) == "99.9"
    assert rig.turn.reconsider.outcome == outcome
    assert rig.turn.reconsider.passes == RECONSIDER_FUSE_PASSES


# ------------------------------------------------------------------ cancellation


@pytest.mark.asyncio
async def test_cancellation_before_acceptance_leaves_no_surface_and_one_cancelled_row(
        events, monkeypatch):
    rig = _Rig(monkeypatch, script=[_decision("post", None)])
    rig.race()

    async def _cancelled_deliver(text: str) -> Optional[str]:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await rig.run(_cancelled_deliver)

    assert rig.delivered == []
    assert rig.turn.reconsider.outcome == "cancelled"
    assert rig.turn.reconsider.passes == 1
    outcomes = _named(events, "reconsider_outcome")
    assert len(outcomes) == 1 and outcomes[0]["outcome"] == "cancelled"


@pytest.mark.asyncio
async def test_cancellation_during_snapshot_cannot_orphan_the_suppression_row(
        events, monkeypatch):
    """The suppression row is emitted BEFORE any await of its pass (§4b), so a mid-pass
    cancellation leaves a stale_send row with no start/outcome rows orphaned above it."""
    rig = _Rig(monkeypatch, script=[])
    rig.race()

    async def _cancelled_snapshot(**kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(reconsideration, "build_reconsideration_snapshot",
                        _cancelled_snapshot)
    with pytest.raises(asyncio.CancelledError):
        await rig.run(rig.accepting_deliver())

    assert len(_named(events, "stale_send")) == 1
    assert _named(events, "reconsider_start") == []
    assert rig.turn.reconsider.outcome == "cancelled"
    assert rig.turn.reconsider.passes == 0


# The two AFTER-acceptance residue cases live with the sites below: they are about what
# the real closure and transport leave behind, which a hand-rolled deliver cannot show.


# ------------------------------------------------------------------ the once-per-turn gate


@pytest.mark.asyncio
async def test_a_suppression_after_the_runner_ended_is_rethrown_unmarked(events, monkeypatch):
    rig = _Rig(monkeypatch, script=[])
    rig.race()
    rig.turn.reconsider = SimpleNamespace(outcome="skipped", passes=1)   # a runner already ran
    exc = rig.suppress()
    with pytest.raises(StaleSendSuppressed) as raised:
        await intercept_stale_send(
            processor=rig.processor, client=None, message=rig.message, turn=rig.turn,
            lease=rig.lease, suppressed=exc, draft="d", deliver=AsyncMock())
    assert raised.value is exc
    assert exc.telemetry_recorded is False           # UNMARKED — the terminal catch owns it
    assert events == []                              # no second runner, no rows


@pytest.mark.asyncio
async def test_dm_and_leaseless_suppressions_rethrow_untouched(events, monkeypatch):
    rig = _Rig(monkeypatch, script=[])
    rig.race()
    exc = rig.suppress()
    for kwargs in ({"channel_turn": False}, {"lease": None}, {"turn": None}):
        with pytest.raises(StaleSendSuppressed) as raised:
            await intercept_stale_send(
                processor=rig.processor, client=None, message=rig.message,
                turn=kwargs.get("turn", rig.turn), lease=kwargs.get("lease", rig.lease),
                suppressed=exc, draft="d", deliver=AsyncMock(),
                channel_turn=kwargs.get("channel_turn", True))
        assert raised.value is exc and exc.telemetry_recorded is False
    assert events == []


# ------------------------------------------------------------------ request truth


@pytest.mark.asyncio
async def test_the_request_is_the_normal_assembly_plus_one_appended_developer_item(
        events, monkeypatch):
    """§4d literal item order: the entire normal no-tools assembly over the fresh snapshot,
    unchanged, then EXACTLY ONE more developer item — instruction, then the fenced draft."""
    from message_processor.channel_request import (assemble_channel_request,
                                                   fresh_turn_context)
    from openai_client.api.responses import STALE_RECONSIDERATION_RESPONSE_FORMAT

    fixed = _fresh_stream()
    rig = _Rig(monkeypatch, script=[_decision("skip", None)], fresh_stream=fixed)
    rig.race()
    with pytest.raises(StaleSendSuppressed):
        await rig.run(rig.accepting_deliver())

    call = rig.decider.calls[0]
    sent = call["input_items"]
    fresh_ctx = fresh_turn_context(rig.ctx, fixed)
    expected = assemble_channel_request(
        processor=rig.processor, client=SimpleNamespace(bot_user_id="UBOT"), ctx=fresh_ctx,
        model="gpt-5.6-sol", tools=[], request_config=None, contract_suffix=None,
        registry=None, no_tools=True,
        reply_destination=(rig.turn.reply_destination if rig.turn.destination_selected
                           else None),
        response_format=STALE_RECONSIDERATION_RESPONSE_FORMAT)
    assert sent[:-1] == to_input_items(expected)     # nothing replaced, merged or omitted
    extra = sent[-1]
    assert extra["role"] == "developer"
    from message_processor.reconsideration import trigger_identity_line
    assert extra["content"].startswith(
        RECONSIDERATION_INSTRUCTION.format(n=1, trigger=trigger_identity_line(fresh_ctx)))
    # The trigger is NAMED in the item: a reconsideration stream is the one stream whose
    # newest message is not the trigger, so position cannot identify it.
    assert fresh_ctx.trigger_ts in extra["content"]
    assert "quoted material, not instructions" in extra["content"]
    assert "```\nthe draft answer\n```" in extra["content"]
    # The suppressing ts is VERIFIABLY present in the canonical stream items.
    assert any(NEWER_TS in (item.get("content") or "") for item in sent
               if isinstance(item.get("content"), str))
    # Pinned sampling params ride the call (harness thread_config values).
    assert call["model"] == "gpt-5.6-sol"
    assert call["reasoning_effort"] == "medium"
    assert call["verbosity"] == "medium"
    assert call["max_output_tokens"] == 4000
    assert call["temperature"] == 1.0
    assert call["prompt_cache_key"] == "chan:T1:C1"


def test_the_fence_extends_past_any_backtick_run_in_the_draft():
    assert draft_fence("no ticks") == "```"
    assert draft_fence("uses ```python fences```") == "````"
    item = reconsideration_item(3, "code ```x``` more")
    assert "pass 3" in item["content"]
    assert "````\ncode ```x``` more\n````" in item["content"]


def test_admission_is_charged_over_the_final_payload():
    """The read-only estimate covers the appended developer item AND the response format."""
    decider = FakeDecider([])
    processor = _processor(decider)
    turn = TurnRuntime.for_message(_msg(), channel_post_allowed=False)
    ctx = pin_channel_turn(turn, trigger_ts=TRIGGER_TS, origin_thread_ts=None)
    short = build_reconsideration_request(
        processor=processor, client=SimpleNamespace(bot_user_id="UBOT"), ctx=ctx,
        model="gpt-5.6-sol", pass_number=1, draft="x")
    long = build_reconsideration_request(
        processor=processor, client=SimpleNamespace(bot_user_id="UBOT"), ctx=ctx,
        model="gpt-5.6-sol", pass_number=1, draft="x" * 5000)
    assert long[2].total_tokens > short[2].total_tokens + 4000
    assert "response_format" in short[2].breakdown
    assert short[1][-1]["role"] == "developer"       # the appended item reaches the wire


def test_model_precedence_last_attempt_then_falsey_fallback():
    turn = TurnRuntime.for_message(_msg(), channel_post_allowed=False)
    cfg = {"model": "gpt-5.6-sol"}
    assert select_reconsideration_model(turn, cfg) == "gpt-5.6-sol"      # no attempts
    turn.next_model_attempt(model="gpt-5.5", fork_reason="initial")
    assert select_reconsideration_model(turn, cfg) == "gpt-5.5"          # last attempt's model
    turn.next_model_attempt(model=None, fork_reason="retry")
    assert select_reconsideration_model(turn, cfg) == "gpt-5.6-sol"      # falsey ⇒ fallback


@pytest.mark.asyncio
async def test_attempt_open_telemetry_failure_omits_the_seq_and_proceeds(events, monkeypatch):
    rig = _Rig(monkeypatch, script=[_decision("post", None)])
    rig.race()
    monkeypatch.setattr(rig.turn, "next_model_attempt",
                        MagicMock(side_effect=RuntimeError("ledger closed")))
    ts = await rig.run(rig.accepting_deliver())
    assert ts == "99.9"                              # telemetry never blocks the call
    starts = _named(events, "reconsider_start")
    # The runner passes `model_attempt_seq=None` when `open()` gave it no attempt. This fixture
    # captures record()'s kwargs BEFORE the emitter's drop-None rule, so None is exactly what it
    # sees; the wire-level OMISSION is covered in tests/unit/test_participation_telemetry.py.
    assert len(starts) == 1
    assert "model_attempt_seq" in starts[0] and starts[0]["model_attempt_seq"] is None


# ------------------------------------------------------------------ reviewed-through


def test_reviewed_through_extraction_is_per_scope_and_floored_at_the_baseline():
    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg(sender="U1"))      # scopes: thread(10.0) + top(C1,U1)
    stream = build_stream([
        normalized(TRIGGER_TS, "the question", sender_id="U1"),
        normalized(NEWER_TS, "reply racer", sender_id="U2", thread_root_ts=TRIGGER_TS),
        normalized("12.0", "unrelated top-level from someone else", sender_id="U3"),
    ])
    reviewed = reviewed_through_map(lease, stream)
    assert reviewed[("thread", CH, TRIGGER_TS)] == NEWER_TS
    # U3's top-level message is NOT in U1's top scope; the scope floors at the baseline.
    assert reviewed[("top", CH, "U1")] == TRIGGER_TS
    assert set(reviewed) == set(lease.scopes)


def test_a_self_authored_reply_never_advances_a_baseline_and_rearm_fails_closed():
    """§4a: reviewed-through is INBOUND evidence only, matching the sites' inbound accounting
    (handlers/text.py skips `sender_type == "self"`). Our own newer serialized post must not
    stand in for a missing human suppressor: the scope floors at its baseline, and the rearm
    preconditions then refuse because the review does not cover the suppressing message."""
    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg(sender="U1"))          # scopes: thread(10.0) + top(C1,U1)
    marks.begin_turn(_msg(ts=NEWER_TS, thread=TRIGGER_TS, sender="U2"))
    suppressed = None
    try:
        lease.authorize("final_post")
    except StaleSendSuppressed as exc:
        suppressed = exc
    assert suppressed is not None, "the lease was not suppressed"
    # The rebuilt window holds our own reply NEWER than the suppressor — and NOT the human
    # suppressor itself.
    stream = build_stream([
        normalized(TRIGGER_TS, "the question", sender_id="U1"),
        normalized("12.0", "our own interim post", sender_id="UBOT", sender_type="self",
                   thread_root_ts=TRIGGER_TS),
    ])
    reviewed = reviewed_through_map(lease, stream)
    assert reviewed[("thread", CH, TRIGGER_TS)] == TRIGGER_TS    # floored, not "12.0"
    with pytest.raises(ValueError):
        lease.rearm_after_reconsideration(reviewed, suppressed)


def test_framing_items_never_advance_a_scope_and_presence_is_metadata_only():
    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg(sender=None))      # thread scope only
    stream = build_stream([normalized(TRIGGER_TS, f"mentions ts {NEWER_TS} in prose",
                                      sender_id="U1")])
    # The horizon/end markers carry no ts; prose mentioning a ts is not presence.
    assert suppressing_ts_present(stream, NEWER_TS) is False
    assert reviewed_through_map(lease, stream) == {
        ("thread", CH, TRIGGER_TS): TRIGGER_TS}


# =============================================================================== the sites


class SiteSlack:
    """A fake transport mirroring the REAL lease and receipt semantics of
    `slack_client/messaging.py`, because a fake that just "authorizes once and commits" cannot
    show what the forced-delivery mandate is about.

    ACCEPTED-RISK REPLICA (review r2, finding 7): this class restates SlackMessagingMixin's
    split / receipt / promotion / cancellation ORDERING by hand — the split loop and truncation
    notice (messaging.py:941/:997), the streaming-update promote-then-commit
    (messaging.py:2790-2798) — and nothing keeps the two in step automatically. A change to
    that ordering in messaging.py MUST be mirrored here, or these tests keep proving the old
    transport. The line anchors below are the checklist:

    * `send_message` authorizes ONCE at entry (`messaging.py:832`; `:835` is the epoch fence). A message that fits posts,
      reports its Delivery, writes its receipt, sets footer attachment, then commits — in that
      order (`:895`–`:911`).
    * A SPLIT reauthorizes before EVERY chunk attempt (`:947`), commits on the FIRST accepted
      chunk before anything else (`:955`), and writes a per-part receipt (`:965`). A chunk that
      fails twice aborts the remainder and reauthorizes AGAIN before the truncation notice
      (`:997`), which is a finalized turn-owned receipt (`:1005`); zero chunks ⇒ return None.
    * `update_message_streaming` authorizes at entry, promotes the receipt (`:2793`), then
      commits (`:2797`).

    Every authorization is recorded with the lease state and waiver as they were AT THAT MOMENT,
    which is the only way to see that a forced first chunk went through the waiver and its
    successors went through a committed lease. Terminating, real strings only.
    """

    MAX_MESSAGE_LENGTH = 3900
    _FOOTER_INLINE_MAX = 180                         # messaging.py:744

    def __init__(self, refuse_post: bool = False, refuse_update: bool = False,
                 truncation_notice_on_refusal: bool = False,
                 fail_chunk: Optional[int] = None,
                 footer_blocks: Optional[List[Dict[str, Any]]] = None,
                 cancel_after_accept: bool = False, cancel_after_edit: bool = False):
        self.refuse_post = refuse_post
        self.refuse_update = refuse_update
        self.truncation_notice_on_refusal = truncation_notice_on_refusal
        self.fail_chunk = fail_chunk                 # 1-based chunk that fails both attempts
        self.footer_blocks = footer_blocks
        self.cancel_after_accept = cancel_after_accept
        self.cancel_after_edit = cancel_after_edit
        self.posts: List[Dict[str, Any]] = []        # landed messages
        self.send_calls: List[Dict[str, Any]] = []   # what the SITE handed the transport
        self.updates: List[Dict[str, Any]] = []      # landed edits
        self.notices: List[str] = []                 # turn-owned truncation notices
        self.authorizations: List[Dict[str, Any]] = []
        self._n = 0

    # -- capabilities
    def supports_streaming(self):
        return True

    def get_streaming_config(self):
        return {"update_interval": 0.0, "buffer_size": 1, "min_interval": 0.0}

    def supports_native_streaming(self):
        return False

    def format_text(self, t):
        return t

    def attachable_footer_blocks(self, channel_id, model=None):
        return self.footer_blocks

    async def set_assistant_status(self, channel, thread, status=""):
        return None

    async def delete_message(self, channel, ts):
        return True

    async def update_message(self, channel, ts, text, receipts=None, receipt_kind=None,
                             receipt_class=None):
        return True                                  # chrome writes; never answer text

    # -- lease and receipt plumbing
    def _authorize(self, lease, surface: str, what: str) -> None:
        """One authorization, with the lease as it stood when the transport asked."""
        record = {"what": what, "surface": surface, "state": lease.state,
                  "waiver": getattr(lease, "_force_waiver", None), "ok": False}
        self.authorizations.append(record)
        lease.authorize(surface)
        record["ok"] = True

    async def _receipt(self, ts, receipts, kind, thread, receipt_class) -> None:
        """messaging.py's `_record_receipt`, verbatim seam and all."""
        from message_processor.outbound_receipts import record_transport_post

        await record_transport_post(team_id="T1", channel_id=CH, message_ts=ts,
                                    receipts=receipts, receipt_kind=kind,
                                    receipt_class=receipt_class,
                                    thread_root_ts=thread, site="site_slack")

    def _footer_rides(self, text: str, blocks) -> bool:
        """messaging.py:760 — the footer rides the message only when the reply fits one
        section block."""
        return bool(blocks) and len(text) <= self._FOOTER_INLINE_MAX and text.count("\n") <= 2

    # -- writes
    async def send_message_get_ts(self, channel, thread, text, lease=None, surface=None,
                                  receipts=None, receipt_kind=None, receipt_class=None):
        if lease is not None:
            lease.authorize(surface or "legacy_seed")
        return {"success": False}                    # no lazy seed in these scenarios

    async def send_message(self, channel, thread, text, blocks=None, meta_out=None,
                           username=None, lease=None, surface=None, receipts=None,
                           receipt_kind=None, receipt_class=None, on_first_accept=None):
        from slack_client.messaging import Delivery

        surface = surface or "final_post"
        self.send_calls.append({"thread": thread, "text": text, "blocks": blocks,
                                "surface": surface, "receipt_class": receipt_class})
        if lease is not None:
            self._authorize(lease, surface, "entry")
        if self.refuse_post:
            if self.truncation_notice_on_refusal:
                self.notices.append("Response was cut off due to length limits")
            return None

        if len(text) <= self.MAX_MESSAGE_LENGTH:
            self._n += 1
            ts = f"post-{self._n}"
            attached = self._footer_rides(text, blocks)
            self.posts.append({"ts": ts, "thread": thread, "text": text,
                               "blocks": blocks if attached else None})
            if on_first_accept is not None:
                on_first_accept(ts)
            if self.cancel_after_accept:
                # §4b r4-4: Slack ACCEPTED the post; the turn dies before the receipt write and
                # before the commit.
                raise asyncio.CancelledError()
            if meta_out is not None:
                meta_out["delivery"] = Delivery(first_ts=ts, text=text, complete=True,
                                                parts_delivered=1, parts_total=1)
            await self._receipt(ts, receipts, receipt_kind, thread, receipt_class)
            if meta_out is not None:
                meta_out["footer_attached"] = attached
            if lease is not None:
                lease.commit()
            return ts

        if meta_out is not None:
            meta_out["footer_attached"] = False      # split replies never attach the footer
        chunks = [text[i:i + self.MAX_MESSAGE_LENGTH]
                  for i in range(0, len(text), self.MAX_MESSAGE_LENGTH)]
        first_ts = None
        delivered_parts = 0
        truncated_at = None
        for i, chunk in enumerate(chunks):
            posted = False
            for attempt in (1, 2):
                if lease is not None:
                    self._authorize(lease, surface, f"chunk{i + 1}/attempt{attempt}")
                if self.fail_chunk == i + 1:
                    continue                         # scripted: this chunk fails both attempts
                self._n += 1
                ts = f"post-{self._n}"
                self.posts.append({"ts": ts, "thread": thread, "text": chunk, "blocks": None})
                if first_ts is None:
                    first_ts = ts
                    if lease is not None:
                        lease.commit()
                    if on_first_accept is not None:
                        on_first_accept(first_ts)
                delivered_parts = i + 1
                await self._receipt(ts, receipts, receipt_kind, thread, receipt_class)
                posted = True
                break
            if not posted:
                truncated_at = i + 1
                if meta_out is not None:
                    meta_out["split_truncated"] = True
                if lease is not None:
                    self._authorize(lease, surface, "truncation_notice")
                self._n += 1
                self.notices.append(f"⚠️ This message was cut off — the remaining "
                                    f"{len(chunks) - i} part(s) failed to post to Slack.")
                # The split-abort truncation notice stamps its own class (messaging.py).
                await self._receipt(f"post-{self._n}", receipts, "finalized", thread,
                                    "system_notice")
                break
        if first_ts and meta_out is not None:
            meta_out["delivery"] = Delivery(
                first_ts=first_ts, text="\n\n".join(chunks[:delivered_parts]),
                complete=truncated_at is None, parts_delivered=delivered_parts,
                parts_total=len(chunks), split=True, truncated_at=truncated_at)
        return first_ts

    async def update_message_streaming(self, channel, ts, text, lease=None, surface=None,
                                       receipts=None):
        if lease is not None:
            self._authorize(lease, surface or "legacy_update", "update")
        if self.refuse_update:
            return {"success": False, "error": "message_not_found"}
        self.updates.append({"ts": ts, "text": text})
        if self.cancel_after_edit:
            # §4b r4-4: chat_update landed; the turn dies BEFORE ReceiptLedger.promote().
            raise asyncio.CancelledError()
        if receipts is not None:
            await receipts.promote(ts)               # messaging.py:2793
        if lease is not None:
            lease.commit()                           # messaging.py:2797
        return {"success": True}


def _zero_chunk_slack() -> SiteSlack:
    """A transport whose POST-REARM send lands zero chunks and posts the truncation notice in
    their place — `messaging.py:997`/`:1012`: the notice is guarded (it would be the turn's only
    visible words), the lease never commits, and the send returns None."""
    slack = SiteSlack()
    original = slack.send_message
    attempts = {"n": 0}

    async def _refuse_after_rearm(*args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return await original(*args, **kwargs)   # first attempt: suppressed by the lease
        lease = kwargs.get("lease")
        if lease is not None:
            slack._authorize(lease, kwargs.get("surface") or "final_post",
                             "truncation_notice")
        slack.notices.append("Response was cut off due to length limits")
        return None                                  # zero chunks landed; notice posted

    slack.send_message = _refuse_after_rearm
    return slack


class SiteOpenAI:
    """Streams real string chunks, TERMINATES, and returns `final_text` as the API result —
    so the completion's final text can differ from what streamed (zero chunks ⇒ the correction
    branch converts the chrome surface)."""

    def __init__(self, final_text: str, chunks: Optional[List[str]] = None,
                 decisions: Optional[List[Any]] = None):
        self.final_text = final_text
        self.chunks = list(chunks or [])
        self.create_reconsideration_decision = FakeDecider(list(decisions or []))

    async def create_streaming_response(self, messages=None, stream_callback=None,
                                        tool_callback=None, **kw):
        for chunk in self.chunks:
            await stream_callback(chunk)
        await stream_callback(None)
        return self.final_text


def _site_processor(openai):
    from message_processor.base import MessageProcessor

    with patch("message_processor.base.AsyncThreadStateManager"), \
         patch("message_processor.base.OpenAIClient"):
        p = MessageProcessor()
    p.openai_client = openai
    p.db = None

    async def _passthru(m, *a, **k):
        return m

    async def _none(*a, **k):
        return None

    p._add_message_with_token_management = MagicMock()
    p._inject_image_analyses = _passthru
    p._pre_trim_messages_for_api = _passthru
    p._get_system_prompt = MagicMock(return_value="SYSTEM-PROMPT")
    p._build_time_suffix_context = MagicMock(return_value="[time: pinned]")
    p._build_generation_inflight_note = MagicMock(return_value=None)
    p._build_research_inflight_note = MagicMock(return_value=None)
    p._build_tools_array = MagicMock(return_value=[])
    p._materialize_request_tools = MagicMock(return_value=(None, {}, False, ""))
    p._persist_tool_provenance = MagicMock()
    p._schedule_async_call = MagicMock()
    p._build_channel_info = _none
    p._async_post_response_cleanup = _none
    p._drop_dead_containers = _none
    p._resolve_ci_container = _none
    return p


def _site_thread_state():
    return SimpleNamespace(
        messages=[{"role": "user", "content": "hi"}], channel_id=CH, thread_ts=TRIGGER_TS,
        current_model="gpt-5.6-sol", config_overrides={}, has_summary_head=False,
        channel_directives=None, record_usage=MagicMock(), last_usage=None)


class SiteRig:
    """A suppressed CHANNEL streaming turn driven through the real handler."""

    def __init__(self, monkeypatch, *, decisions: List[Any], final_text: str,
                 slack: Optional[SiteSlack] = None, chunks: Optional[List[str]] = None,
                 receipts: bool = False):
        monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
        self.slack = slack if slack is not None else SiteSlack()
        self.openai = SiteOpenAI(final_text, chunks=chunks, decisions=decisions)
        self.processor = _site_processor(self.openai)
        self.message = _msg()
        self.marks = ConversationWatermarks()
        self.lease = self.marks.begin_turn(self.message)
        self.turn = TurnRuntime.for_message(self.message, channel_post_allowed=False)
        self.turn.send_lease = self.lease
        # `receipts=True` binds a REAL ReceiptLedger (needs the `receipt_service` fixture).
        self.ledger = _real_ledger(self.turn.turn_id) if receipts else None
        self.turn.receipt_ledger = self.ledger
        # Pin what base.py pins: the capability profile and the turn context + prepared tools.
        from tests.unit.channel_turn_harness import thread_config
        self.cfg = thread_config()
        self.turn.capability_profile = self.cfg
        prepared = (None, {}, False, "", None)
        self.ctx = pin_channel_turn(self.turn, trigger_ts=TRIGGER_TS,
                                    origin_thread_ts=TRIGGER_TS, config=self.cfg,
                                    prepared=prepared)
        # Race: a reply lands under the trigger's thread while the model is writing.
        self.marks.begin_turn(_msg(ts=NEWER_TS, thread=TRIGGER_TS, sender="U2"))

        async def _snapshot(**kwargs):
            return SimpleNamespace(stream=_fresh_stream())

        monkeypatch.setattr(reconsideration, "build_reconsideration_snapshot", _snapshot)

    async def run(self, thinking_id: Optional[str] = None) -> Response:
        return await self.processor._handle_streaming_text_response(
            "hi", _site_thread_state(), self.slack, self.message, thinking_id, None,
            turn=self.turn)


# ------------------------------------------------------------------ the buffered final post


@pytest.mark.asyncio
async def test_direct_final_post_posts_asis_and_bookkeeps_the_chosen_text(events, monkeypatch):
    rig = SiteRig(monkeypatch, decisions=[_decision("post", None)], final_text="the answer")
    resp = await rig.run()
    assert [p["text"] for p in rig.slack.posts] == ["the answer"]
    assert resp.metadata["streamed"] is True and resp.metadata["posted"] is True
    assert resp.content == "the answer"
    assert rig.lease.state == COMMITTED
    assert rig.turn.reconsider.outcome == "posted_asis"
    committed = rig.turn.committed_destinations
    assert len(committed) == 1 and committed[0].text == "the answer"
    # F7 keyed on the NEW Slack ts.
    rig.processor._persist_tool_provenance.assert_called_once()
    assert rig.processor._persist_tool_provenance.call_args.args[1] == rig.slack.posts[0]["ts"]


@pytest.mark.asyncio
async def test_direct_final_post_revised_replaces_the_canonical_text_before_send(
        events, monkeypatch):
    rig = SiteRig(monkeypatch, decisions=[_decision("post", "a shorter answer")],
                  final_text="the long draft")
    resp = await rig.run()
    assert [p["text"] for p in rig.slack.posts] == ["a shorter answer"]
    # Every bookkeeping path read the CHOSEN text — content, destination, provenance key.
    assert resp.content == "a shorter answer"
    assert rig.turn.committed_destinations[0].text == "a shorter answer"
    assert rig.turn.reconsider.outcome == "posted_revised"


@pytest.mark.asyncio
async def test_direct_final_post_forced_asis(events, monkeypatch):
    rig = SiteRig(monkeypatch, decisions=[_decision("force_post", None)],
                  final_text="the answer")
    resp = await rig.run()
    assert [p["text"] for p in rig.slack.posts] == ["the answer"]
    assert (rig.turn.reconsider.outcome, rig.turn.reconsider.forced) == ("posted_asis", True)
    assert rig.lease.state == COMMITTED and rig.lease._force_waiver is False
    assert resp.metadata["posted"] is True
    assert rig.turn.committed_destinations[0].text == "the answer"


@pytest.mark.asyncio
async def test_direct_final_post_forced_split_delivery(events, monkeypatch, receipt_service):
    """The §8 forced-delivery mandate, through the transport's real shape. A forced REVISED
    reply long enough to split: the waiver carries the FIRST chunk while the lease is still
    PENDING, that chunk's acceptance commits and clears the waiver, and the later chunks
    reauthorize against a committed lease — every one of them a real `authorize` call, as
    messaging.py makes them. Each part earns its own receipt in a REAL ledger, the footer
    stands down (a split never carries it), and F7 plus the destination record read the CHOSEN
    text."""
    long_text = "y" * 8000
    slack = SiteSlack(footer_blocks=[{"type": "actions", "elements": []}])
    rig = SiteRig(monkeypatch, decisions=[_decision("force_post", long_text)],
                  final_text="the draft", slack=slack, receipts=True)
    resp = await rig.run()

    assert len(slack.posts) == 3                     # 8000 chars over a 3900 limit
    chunk_auths = [a for a in slack.authorizations if a["what"].startswith("chunk")]
    assert [a["what"] for a in chunk_auths] == ["chunk1/attempt1", "chunk2/attempt1",
                                                "chunk3/attempt1"]
    assert all(a["ok"] for a in chunk_auths)
    assert (chunk_auths[0]["state"], chunk_auths[0]["waiver"]) == (PENDING, True)
    assert all((a["state"], a["waiver"]) == (COMMITTED, False) for a in chunk_auths[1:])
    assert rig.lease.state == COMMITTED and rig.lease._force_waiver is False

    # Receipts: one per part, registered through the real ledger and settling with the turn.
    assert rig.ledger.pending_ts == [p["ts"] for p in slack.posts]
    for post in slack.posts:
        row = await receipt_service.get_receipt_async("T1", CH, post["ts"])
        assert row["state"] == "in_flight" and row["turn_id"] == rig.turn.turn_id

    # The CHOSEN text drives every bookkeeping path the site owns — including the footer, whose
    # blocks the site built and the closure carried into the re-run send.
    assert resp.content == long_text
    # The refused attempt carried the draft; the re-run carried the revision — the canonical
    # text was replaced BEFORE the send, not after it.
    assert [c["text"] for c in slack.send_calls] == ["the draft", long_text]
    assert all(c["blocks"] == slack.footer_blocks for c in slack.send_calls)
    committed = rig.turn.committed_destinations
    assert len(committed) == 1 and committed[0].kind == "split"
    assert committed[0].text == long_text and committed[0].complete is True
    assert resp.metadata["footer_attached"] is False   # a split never carries the chrome
    assert rig.processor._persist_tool_provenance.call_args.args[1] == slack.posts[0]["ts"]
    assert (rig.turn.reconsider.outcome, rig.turn.reconsider.forced) == (
        "posted_revised", True)
    assert _named(events, "reconsider_outcome")[0]["forced"] is True


@pytest.mark.asyncio
async def test_forced_split_reauthorizes_before_the_truncation_notice(events, monkeypatch):
    """§4a: the truncation notice is a mutation like any other and reauthorizes
    (`messaging.py:997`). Chunk 2 fails twice here, so chunk 1 has already committed the lease
    and the notice goes through that — and the destination is committed from what Slack
    ACTUALLY took, not from the text we handed over."""
    slack = SiteSlack(fail_chunk=2)
    rig = SiteRig(monkeypatch, decisions=[_decision("force_post", "y" * 8000)],
                  final_text="the draft", slack=slack)
    resp = await rig.run()

    guarded = [a for a in slack.authorizations
               if a["what"].startswith("chunk") or a["what"] == "truncation_notice"]
    assert [a["what"] for a in guarded] == ["chunk1/attempt1", "chunk2/attempt1",
                                            "chunk2/attempt2", "truncation_notice"]
    notice = guarded[-1]
    assert notice["ok"] is True and notice["state"] == COMMITTED
    assert len(slack.posts) == 1 and len(slack.notices) == 1
    committed = rig.turn.committed_destinations
    assert len(committed) == 1 and committed[0].complete is False
    assert committed[0].text == slack.posts[0]["text"]
    assert resp.metadata["posted"] is True           # the room can read the first chunk
    assert rig.turn.reconsider.outcome == "posted_revised"


@pytest.mark.asyncio
async def test_direct_final_post_skip_raises_and_posts_nothing(events, monkeypatch):
    rig = SiteRig(monkeypatch, decisions=[_decision("skip", None)], final_text="the answer")
    with pytest.raises(StaleSendSuppressed) as raised:
        await rig.run()
    assert rig.slack.posts == []
    assert raised.value.telemetry_recorded is True   # main.py's catch will not double-count
    assert rig.turn.reconsider.outcome == "skipped"
    rig.processor._persist_tool_provenance.assert_not_called()


@pytest.mark.asyncio
async def test_direct_final_post_delivery_failed_never_activates_the_rescue(
        events, monkeypatch):
    """r5-1 + the zero-chunk notice: deliver returns None (after posting the unleased
    truncation notice), and the site returns streamed=True/posted=False — the
    `final_post_failed` hand-back that would make main.py re-send the refused draft stays
    OFF, and the notice never reaches the turn's destinations (the ruled contract
    exception)."""
    slack = _zero_chunk_slack()
    rig = SiteRig(monkeypatch, decisions=[_decision("post", None)], final_text="the answer",
                  slack=slack)
    resp = await rig.run()

    assert resp.metadata["streamed"] is True         # NOT flipped to the hand-back shape
    assert resp.metadata.get("posted") is False      # honest: nothing delivered
    assert "tool_provenance" not in resp.metadata    # final_post_failed never armed
    assert slack.posts == [] and len(slack.notices) == 1
    assert rig.turn.destinations == []               # the notice is absent from destinations
    assert (rig.turn.reconsider.outcome, rig.turn.reconsider.error) == (
        "error_dropped", "delivery_failed")
    assert len(_named(events, "stale_send")) == 1    # single suppression row, runner-owned
    assert len(_named(events, "reconsider_outcome")) == 1


@pytest.mark.asyncio
async def test_zero_chunk_reconsidered_send_classifies_as_delivery_failed(events, monkeypatch):
    """§8 end-to-end: the response the zero-chunk site hands back, run through main.py's OWN
    terminal classifier. `stale_suppressed` claims the room saw nothing, and a truncation notice
    is something — so the label has to be `delivery_failed`, which is what today's
    delivery-failure accounting already means."""
    from main import ChatBotV2

    slack = _zero_chunk_slack()
    rig = SiteRig(monkeypatch, decisions=[_decision("post", None)], final_text="the answer",
                  slack=slack)
    resp = await rig.run()

    assert ChatBotV2._classify_visible_action(resp, rig.turn) == "delivery_failed"


@pytest.mark.asyncio
async def test_direct_final_post_delivery_exception_hands_nothing_back(events, monkeypatch):
    """FIX-1 at the buffered site: the closure's send RAISES. The runner ends
    `error_dropped(delivery_exception)` and returns None, and the site takes the same
    no-rescue path as `delivery_failed` — `streamed=True`/`posted=False` with the
    `final_post_failed` hand-back OFF, so main.py never re-sends the refused draft."""
    slack = SiteSlack()
    rig = SiteRig(monkeypatch, decisions=[_decision("post", None)], final_text="the answer",
                  slack=slack)
    original = slack.send_message
    attempts = {"n": 0}

    async def _raise_after_rearm(*args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return await original(*args, **kwargs)   # first attempt: suppressed by the lease
        raise RuntimeError("the transport exploded")

    slack.send_message = _raise_after_rearm
    resp = await rig.run()

    assert resp.metadata["streamed"] is True
    assert resp.metadata.get("posted") is False
    assert "tool_provenance" not in resp.metadata    # final_post_failed never armed
    assert slack.posts == [] and rig.turn.destinations == []
    assert (rig.turn.reconsider.outcome, rig.turn.reconsider.error) == (
        "error_dropped", "delivery_exception")
    assert len(_named(events, "reconsider_outcome")) == 1
    assert len(_named(events, "stale_send")) == 1


# ------------------------------------------------------------------ the correction site


@pytest.mark.asyncio
async def test_correction_site_posts_asis_into_the_existing_surface(events, monkeypatch):
    rig = SiteRig(monkeypatch, decisions=[_decision("post", None)], final_text="the answer")
    resp = await rig.run(thinking_id="chrome-1")
    assert rig.slack.updates == [{"ts": "chrome-1", "text": "the answer"}]
    assert rig.slack.posts == []
    assert resp.metadata["posted"] is True
    assert rig.turn.reconsider.outcome == "posted_asis"
    # F7 keyed on the converted chrome surface.
    assert rig.processor._persist_tool_provenance.call_args.args[1] == "chrome-1"


@pytest.mark.asyncio
async def test_correction_site_revised_over_3900_reruns_the_cut_and_overflow(
        events, monkeypatch):
    from message_markers import CONTINUATION_HEAD

    revised = "z" * 4500
    rig = SiteRig(monkeypatch, decisions=[_decision("post", revised)],
                  final_text="short draft")
    resp = await rig.run(thinking_id="chrome-1")
    # The branch's own computation ran on the REVISED text: truncated head + continuation tail.
    assert len(rig.slack.updates) == 1
    head = rig.slack.updates[0]["text"]
    assert head.startswith("z") and len(head) < 4000
    assert len(rig.slack.posts) == 1
    assert rig.slack.posts[0]["text"].startswith(CONTINUATION_HEAD)
    assert rig.turn.reconsider.outcome == "posted_revised"
    assert resp.metadata["posted"] is True


@pytest.mark.asyncio
async def test_correction_site_head_failure_withholds_the_tail(events, monkeypatch):
    """r4-2: in a reconsidered delivery a failed head means NO tail — a continuation must
    never stand over an unconverted chrome head — and the ending is delivery_failed."""
    slack = SiteSlack(refuse_update=True)
    revised = "z" * 4500
    rig = SiteRig(monkeypatch, decisions=[_decision("post", revised)],
                  final_text="short draft", slack=slack)
    resp = await rig.run(thinking_id="chrome-1")
    assert slack.posts == []                         # the tail was withheld
    assert slack.updates == []                       # nothing landed at all
    assert (rig.turn.reconsider.outcome, rig.turn.reconsider.error) == (
        "error_dropped", "delivery_failed")
    assert resp.metadata.get("posted") is False
    assert resp.metadata["streamed"] is True


@pytest.mark.asyncio
async def test_correction_site_force_promotes_the_chrome_receipt(events, monkeypatch,
                                                                 receipt_service):
    """Forced delivery into an existing chrome surface, through a REAL ReceiptLedger. The edit
    that carries the answer is exactly where that surface stops being chrome
    (`messaging.py:2793`), so the mandate's "receipt promotion" is only shown by driving
    `promote()` itself — a fake that commits the lease and calls it done proves nothing."""
    rig = SiteRig(monkeypatch, decisions=[_decision("force_post", None)],
                  final_text="the answer", receipts=True)
    await rig.ledger.note_chrome("chrome-1", TRIGGER_TS, receipt_class="chrome")
    assert (await receipt_service.get_receipt_async("T1", CH, "chrome-1"))["state"] == "chrome"

    resp = await rig.run(thinking_id="chrome-1")

    assert rig.slack.updates == [{"ts": "chrome-1", "text": "the answer"}]
    assert resp.metadata["posted"] is True
    assert (rig.turn.reconsider.outcome, rig.turn.reconsider.forced) == ("posted_asis", True)
    assert rig.lease.state == COMMITTED and rig.lease._force_waiver is False
    # PROMOTED: the surface now belongs to the turn and settles with it.
    assert rig.ledger.pending_ts == ["chrome-1"]
    row = await receipt_service.get_receipt_async("T1", CH, "chrome-1")
    assert row["state"] == "in_flight" and row["turn_id"] == rig.turn.turn_id


@pytest.mark.asyncio
async def test_correction_site_forced_revised(events, monkeypatch):
    rig = SiteRig(monkeypatch, decisions=[_decision("force_post", "a better answer")],
                  final_text="the answer")
    resp = await rig.run(thinking_id="chrome-1")
    assert rig.slack.updates == [{"ts": "chrome-1", "text": "a better answer"}]
    assert resp.content == "a better answer"
    assert (rig.turn.reconsider.outcome, rig.turn.reconsider.forced) == (
        "posted_revised", True)


@pytest.mark.asyncio
async def test_correction_site_skip_writes_nothing(events, monkeypatch):
    rig = SiteRig(monkeypatch, decisions=[_decision("skip", None)], final_text="the answer")
    with pytest.raises(StaleSendSuppressed):
        await rig.run(thinking_id="chrome-2")
    assert rig.slack.updates == [] and rig.slack.posts == []
    assert rig.turn.reconsider.outcome == "skipped"


# --------------------------------------------- cancellation residue at the sites


@pytest.mark.asyncio
async def test_cancellation_after_acceptance_leaves_the_enumerated_post_residue(
        events, monkeypatch, receipt_service):
    """§4b r4-4, direct post, driven through the REAL closure and transport: the send is
    cancelled between Slack's acceptance and the receipt write. The enumerated residue — post
    VISIBLE, destination already OBSERVED (`messaging.py:893`), receipt unregistered, lease
    UNCOMMITTED — is today's cancellation window, and these are the states to assert rather
    than "no invariant violation"."""
    slack = SiteSlack(cancel_after_accept=True)
    rig = SiteRig(monkeypatch, decisions=[_decision("post", None)], final_text="the answer",
                  slack=slack, receipts=True)

    with pytest.raises(asyncio.CancelledError):
        await rig.run()

    assert [p["text"] for p in slack.posts] == ["the answer"]     # visible in the room
    observed = [r for r in rig.turn.destinations if r.first_ts == slack.posts[0]["ts"]]
    assert observed and not observed[0].committed    # observed, never committed
    assert rig.lease.state == PENDING                # rearm re-opened it; commit never ran
    assert rig.ledger.pending_ts == []               # the receipt write never ran
    assert await receipt_service.get_receipt_async("T1", CH, slack.posts[0]["ts"]) is None
    assert rig.turn.reconsider.outcome == "cancelled"


@pytest.mark.asyncio
async def test_correction_edit_cancelled_before_promotion_stays_chrome_and_excluded(
        events, monkeypatch, receipt_service):
    """§4b r4-4, correction edit, the BEFORE-promotion seam, through the real closure: the
    answer is visibly edited, the receipt is still `chrome` in a real ledger, no destination was
    observed, no lease commit — and a chrome receipt renders no assistant item, so the edited
    answer is excluded from future rebuilt streams. (Cancellation DURING promotion is the
    alternative state: promote() stages the ts before its first await and settlement finalizes
    staged rows, so the receipt may end FINALIZED and the answer included — still no commit,
    still no destination.)"""
    from message_processor.channel_stream import ReceiptRec
    from tests.unit.channel_turn_harness import sidecars

    chrome_ts = "10.5"                               # a real Slack ts: the stream parses it
    slack = SiteSlack(cancel_after_edit=True)
    rig = SiteRig(monkeypatch, decisions=[_decision("post", None)], final_text="the answer",
                  slack=slack, receipts=True)
    await rig.ledger.note_chrome(chrome_ts, TRIGGER_TS, receipt_class="chrome")

    with pytest.raises(asyncio.CancelledError):
        await rig.run(thinking_id=chrome_ts)

    assert slack.updates == [{"ts": chrome_ts, "text": "the answer"}]   # visibly edited
    assert rig.turn.destinations == []               # correction edits observe no destination
    assert rig.lease.state == PENDING
    assert rig.ledger.pending_ts == []               # promote() never entered
    assert (await receipt_service.get_receipt_async(
        "T1", CH, chrome_ts))["state"] == "chrome"
    assert rig.turn.reconsider.outcome == "cancelled"
    # Future-stream exclusion: a self message whose receipt is still `chrome` renders NOTHING.
    receipts = (ReceiptRec(ts=chrome_ts, state="chrome", turn_id="t",
                           thread_root_ts=TRIGGER_TS),)
    stream = build_stream(
        [normalized(chrome_ts, "the answer", sender_id="UBOT", sender_type="self")],
        pinned_sidecars=sidecars(receipts=receipts))
    assert all(item.metadata.get("ts") != chrome_ts for item in stream.message_items)


# ------------------------------------------------------------------ exclusion regressions


def _text_source() -> str:
    import message_processor.handlers.text as text_module
    return inspect.getsource(text_module)


def test_current_part_gt_1_branches_stay_excluded():
    """§3: `current_part > 1` implies a committed lease (the only routes there commit on their
    first visible success), so those branches keep the bare re-raise and never invoke the
    runner. Pinned structurally: the branch's StaleSendSuppressed handler is a bare `raise`."""
    tree = ast.parse(_text_source())
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test_src = ast.unparse(node.test)
        if test_src == "current_part > 1":
            # The branch BODY only — `orelse` is the sibling correction branch, which IS
            # covered and does call the runner.
            for handler in [h for stmt in node.body for n in ast.walk(stmt)
                            if isinstance(n, ast.Try) for h in n.handlers]:
                if handler.type is not None and "StaleSendSuppressed" in ast.unparse(handler.type):
                    found.append(handler)
    assert found, "the current_part > 1 branch (and its suppression handler) must exist"
    for handler in found:
        body_src = ast.unparse(handler)
        assert "intercept_stale_send" not in body_src
        assert any(isinstance(n, ast.Raise) and n.exc is None for n in handler.body), (
            "the excluded overflow branch must re-raise the suppression untouched")


def _parent_map(tree):
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _innermost_guard(node, parents):
    """The nearest `if` test that DOMINATES this node — None when nothing guards it."""
    child, parent = node, parents.get(node)
    while parent is not None:
        if isinstance(parent, ast.If) and child in parent.body:
            return ast.unparse(parent.test)
        child, parent = parent, parents.get(parent)
    return None


def _enclosing_function(node, parents):
    parent = parents.get(node)
    while parent is not None and not isinstance(parent, (ast.FunctionDef,
                                                         ast.AsyncFunctionDef)):
        parent = parents.get(parent)
    return parent


def _awaited_producer(func, names, dotted):
    """The `x = await <dotted>(...)` inside `func` that binds `names`, or None.

    ACCEPTED RISK (review r2, finding 8): the domination check built on this is a TRIPWIRE,
    not a dataflow proof — it finds the guard expression syntactically and trusts the human
    argument (in the test docstring) that the guarding name really carries the awaited call's
    success. A refactor that keeps the names but reroutes the values would still pass it."""
    for node in ast.walk(func):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Await):
            continue
        call = node.value.value
        if not isinstance(call, ast.Call) or ast.unparse(call.func) != dotted:
            continue
        bound = {n.id for target in node.targets for n in ast.walk(target)
                 if isinstance(n, ast.Name)}
        if names <= bound:
            return call
    return None


def test_current_part_assignments_are_dominated_by_a_committed_lease():
    """The reachability half of the §3 proof, pinned at the ASSIGNMENT sites rather than only
    at the excluded branch. `current_part` leaves 1 in exactly three places, and each one sits
    inside a branch guarded by a VISIBLE mutation that already succeeded:

    * the native part ROLL — guarded by `overflow is not None` from
      `await native_coord.update(...)`, which reaches `_roll` only on a live session;
    * the legacy OVERFLOW — guarded by `result['success']` from an answer-bearing
      `client.update_message_streaming(..., lease=...)`;
    * the native FINALIZE — guarded by `native_finalized` from `await native_coord.finalize(...)`,
      the same live-session proof as the roll.

    And each of those mutations commits the lease: `update_message_streaming` on success
    (`messaging.py:2797`), and a native session at `start` (`messaging.py:329`) — which
    `update`/`finalize` require, since both bail unless the session is active. A committed lease
    never raises, so `current_part > 1` can never see a suppression."""
    tree = ast.parse(_text_source())
    parents = _parent_map(tree)

    writers = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AugAssign):
            target = node.target
        else:
            continue
        if isinstance(target, ast.Name) and target.id == "current_part":
            writers[node.lineno] = (ast.unparse(node), node)

    raising = {line: (src, node) for line, (src, node) in writers.items()
               if src in ("current_part = native_coord.part", "current_part += 1")}
    assert len(raising) == 3, (
        f"current_part writers changed — reprove §3 before editing this test: "
        f"{ {line: src for line, (src, _) in writers.items()} }")

    guards = {(src, _innermost_guard(node, parents)) for src, node in raising.values()}
    assert guards == {
        ("current_part = native_coord.part", "overflow is not None"),
        ("current_part += 1", "result['success']"),
        ("current_part = native_coord.part", "native_finalized"),
    }, f"a current_part write moved out from behind its success guard: {guards}"

    for src, node in raising.values():
        func = _enclosing_function(node, parents)
        guard = _innermost_guard(node, parents)
        if guard == "overflow is not None":
            producer = _awaited_producer(func, {"ok", "overflow"}, "native_coord.update")
        elif guard == "native_finalized":
            producer = _awaited_producer(func, {"native_finalized"}, "native_coord.finalize")
        else:
            producer = _awaited_producer(func, {"result"},
                                         "client.update_message_streaming")
            assert producer is not None and any(
                kw.arg == "lease" for kw in producer.keywords), (
                "the legacy overflow's guarding update must be LEASED — its success is what "
                "commits the lease")
        assert producer is not None, f"no awaited producer for the guard {guard!r}"

    from slack_client.messaging import NativeStreamSession, SlackMessagingMixin
    from streaming.native_sink import NativeStreamCoordinator

    assert "lease.commit()" in inspect.getsource(SlackMessagingMixin.update_message_streaming)
    assert "lease.commit()" in inspect.getsource(NativeStreamSession.start)
    # `update` (and `_roll` through it) and `finalize` do nothing on a session that never
    # started — which is where the lease was committed.
    assert "if not self.active" in inspect.getsource(NativeStreamCoordinator.update)
    assert "if not self.active" in inspect.getsource(NativeStreamCoordinator.finalize)


def test_a_committed_lease_never_reaches_reconsideration():
    """The reachability half of the §3 proof: once a lease is COMMITTED, authorize never
    raises, so a branch only reachable after a committed first surface cannot see a
    suppression at all."""
    marks = ConversationWatermarks()
    lease = marks.begin_turn(_msg())
    lease.commit()
    marks.begin_turn(_msg(ts=NEWER_TS, thread=TRIGGER_TS, sender="U2"))
    lease.authorize("legacy_update")                 # no raise despite the newer message


def test_correction_edit_closures_keep_their_epoch_exemption():
    """§4a r3-6: `update_message_streaming` performs no epoch check — epoch_invalidated is
    reachable only at NEW-POST closures, where the fence exists in send_message. The exemption
    is intentional and documented; this pins it."""
    from slack_client.messaging import SlackMessagingMixin

    assert "_epoch_authorize" not in inspect.getsource(
        SlackMessagingMixin.update_message_streaming)
    assert "_epoch_authorize" in inspect.getsource(SlackMessagingMixin.send_message)


# =============================================================================== main.py


def _main_bot():
    from main import ChatBotV2

    bot = ChatBotV2(platform="slack")
    bot.processor = MagicMock()
    bot.processor.db = None
    bot.processor._get_system_prompt.return_value = "SYSTEM-PROMPT"
    bot.processor._build_time_suffix_context.return_value = "[time: pinned]"
    bot.processor._build_generation_inflight_note.return_value = None
    bot.processor._build_research_inflight_note.return_value = None
    return bot


def _main_client(send_message):
    client = MagicMock()
    client.self_team_id = "T1"
    client.bot_user_id = "UBOT"
    client.send_thinking_indicator = AsyncMock(return_value=None)
    client.delete_message = AsyncMock(return_value=True)
    client.send_message = send_message
    client.handle_error = AsyncMock()
    client.maybe_post_response_footer = AsyncMock()
    client.attachable_footer_blocks = MagicMock(return_value=None)
    client.clear_assistant_status = AsyncMock()
    return client


def _main_message(channel: str = CH) -> Message:
    return Message(text="hi", user_id="U1", channel_id=channel, thread_id=TRIGGER_TS,
                   metadata={"ts": TRIGGER_TS, "sender_id": "U1"})


class MainRig:
    """The real handle_message over a mocked processor: the model 'returns' a non-streamed
    draft, a newer message lands mid-turn, and the :1111 send is refused by the real lease."""

    def __init__(self, monkeypatch, *, decisions: List[Any], draft: str = "the answer",
                 metadata: Optional[Dict[str, Any]] = None, channel: str = CH,
                 deliver_result: str = "accept",
                 destination: Optional[str] = None,
                 footer_blocks: Optional[List[Dict[str, Any]]] = None):
        self.bot = _main_bot()
        self.message = _main_message(channel)
        self.decider = FakeDecider(decisions)
        self.bot.processor.openai_client = SimpleNamespace(
            create_reconsideration_decision=self.decider)
        self.bot.processor._persist_tool_provenance = MagicMock()
        self.deliver_result = deliver_result
        self.sent: List[Dict[str, Any]] = []
        self.attempts = 0
        self.turn: Optional[TurnRuntime] = None
        self.response = Response(type="text", content=draft,
                                 metadata={"streamed": False, "model": "gpt-5.6-sol",
                                           **(metadata or {})})
        rig = self

        async def _process(message, client, thinking_id=None, turn=None):
            rig.turn = turn
            pin_channel_turn(turn, trigger_ts=TRIGGER_TS, origin_thread_ts=TRIGGER_TS,
                             channel_id=channel)
            if destination is not None:
                # F39: a reply headed for the channel top level, whose reply target is None.
                turn.reply_destination = destination
            # The race: a newer reply lands while the model is writing.
            rig.bot.watermarks.begin_turn(_msg(ts=NEWER_TS, thread=TRIGGER_TS, sender="U2",
                                               channel=channel))
            return rig.response

        self.bot.processor.process_message = AsyncMock(side_effect=_process)

        async def _send(channel_id, thread_id, text, blocks=None, meta_out=None, lease=None,
                        receipts=None, on_first_accept=None, **kw):
            rig.attempts += 1
            if lease is not None:
                lease.authorize("final_post")
            # "raise" scripts a NON-ENUMERATED failure out of the reconsidered send — the
            # first attempt is the one the lease refuses, so this only ever fires post-rearm.
            if rig.deliver_result == "raise":
                raise RuntimeError("the transport exploded")
            if rig.deliver_result == "none":
                return None
            self.sent.append({"thread": thread_id, "text": text, "blocks": blocks})
            if lease is not None:
                lease.commit()
            if on_first_accept is not None:
                on_first_accept("200.1")
            if meta_out is not None:
                meta_out["footer_attached"] = bool(blocks)
            return "200.1"

        self.client = _main_client(_send)
        if footer_blocks is not None:
            self.client.attachable_footer_blocks = MagicMock(return_value=footer_blocks)

        async def _snapshot(**kwargs):
            return SimpleNamespace(stream=_fresh_stream())

        monkeypatch.setattr(reconsideration, "build_reconsideration_snapshot", _snapshot)

    async def run(self):
        await self.bot.handle_message(self.message, self.client)


@pytest.mark.asyncio
async def test_main_site_posts_the_reconsidered_reply_and_reports_it(events, monkeypatch):
    rig = MainRig(monkeypatch, decisions=[_decision("post", "the revised answer")])
    await rig.run()
    assert [s["text"] for s in rig.sent] == ["the revised answer"]
    assert rig.response.content == "the revised answer"   # canonical replacement
    assert rig.response.metadata["posted"] is True
    assert rig.turn.reconsider.outcome == "posted_revised"
    # F7 keyed on the NEW Slack ts.
    persist = rig.bot.processor._persist_tool_provenance
    assert persist.call_args.args[1] == "200.1"
    # One suppression row (runner-owned), the outcome, and the turn row carrying the facts.
    assert len(_named(events, "stale_send")) == 1
    assert _named(events, "stale_send")[0]["turn_id"] == rig.turn.turn_id
    assert len(_named(events, "reconsider_outcome")) == 1
    turn_rows = _named(events, "turn_outcome")
    assert turn_rows and turn_rows[0]["reconsider"] == {
        "outcome": "posted_revised", "passes": 1, "forced": False}
    # §8 correspondence: a posted reconsideration is an ORDINARY reply in the ledger.
    assert turn_rows[0]["kind"] == "reply"


@pytest.mark.asyncio
@pytest.mark.parametrize("decision,text,outcome,forced,delivered", [
    ("post", None, "posted_asis", False, "the answer"),
    ("force_post", None, "posted_asis", True, "the answer"),
    ("force_post", "rewritten for the room", "posted_revised", True, "rewritten for the room"),
])
async def test_main_site_decision_cells(decision, text, outcome, forced, delivered,
                                        events, monkeypatch):
    """The rest of the main site's cross-product (post-revised has its own test above)."""
    rig = MainRig(monkeypatch, decisions=[_decision(decision, text)])
    await rig.run()
    assert [s["text"] for s in rig.sent] == [delivered]
    assert rig.response.content == delivered          # canonical replacement, every cell
    assert rig.response.metadata["posted"] is True
    assert (rig.turn.reconsider.outcome, rig.turn.reconsider.forced) == (outcome, forced)
    assert rig.turn.send_lease.state == COMMITTED
    assert rig.turn.send_lease._force_waiver is False  # commit clears it, forced or not
    assert _named(events, "turn_outcome")[0]["kind"] == "reply"


@pytest.mark.asyncio
async def test_main_site_skip_sends_nothing_and_runs_the_terminal_path(events, monkeypatch):
    """Skip at the main site: the runner emits `skipped` and re-raises the MARKED suppression,
    so nothing is sent, main.py's terminal stale handling runs as it always has, and the row
    count stays one."""
    rig = MainRig(monkeypatch, decisions=[_decision("skip", None)])
    await rig.run()
    assert rig.sent == []
    assert rig.turn.reconsider.outcome == "skipped"
    assert len(_named(events, "stale_send")) == 1     # runner-owned; the catch added none
    assert _named(events, "reconsider_outcome")[0]["outcome"] == "skipped"
    assert _named(events, "turn_outcome")[0]["kind"] == "stale_suppressed"


@pytest.mark.asyncio
async def test_main_site_attaches_the_footer_to_the_chosen_text(events, monkeypatch):
    """§4b: the closure replaces `response.content` BEFORE the send, so the footer chrome the
    site built rides the reconsidered message and the separate-footer fallback reads the
    REVISED text — no second code path anywhere in the footer bookkeeping."""
    blocks = [{"type": "actions", "elements": []}]
    rig = MainRig(monkeypatch, decisions=[_decision("post", "the revised answer")],
                  footer_blocks=blocks)
    await rig.run()

    assert [s["text"] for s in rig.sent] == ["the revised answer"]
    assert rig.sent[0]["blocks"] == blocks            # the chrome rode the reconsidered send
    assert rig.response.metadata["footer_attached"] is True
    footer = rig.client.maybe_post_response_footer
    footer.assert_awaited_once()
    assert footer.await_args.args[1].content == "the revised answer"


@pytest.mark.asyncio
async def test_main_site_top_level_reply_keeps_its_none_target(events, monkeypatch):
    """F39 (§4b): the closure re-runs the site's OWN send, so a reply headed for the channel
    top level — whose reply target is None — is re-sent top-level rather than quietly
    threaded."""
    from message_processor.turn_runtime import DESTINATION_CHANNEL

    rig = MainRig(monkeypatch, decisions=[_decision("post", "the revised answer")],
                  destination=DESTINATION_CHANNEL)
    await rig.run()

    assert rig.turn.resolve_reply_target(rig.message) is None
    assert [s["thread"] for s in rig.sent] == [None]
    assert rig.turn.reconsider.outcome == "posted_revised"


@pytest.mark.asyncio
async def test_main_site_revision_keeps_the_original_tool_provenance(events, monkeypatch):
    """§8: a revised reply is still the SAME turn's work. The F7 record persists the original
    provenance unchanged, keyed on the NEW Slack ts, while content and destination accounting
    follow the revised text."""
    provenance = [{"tool_name": "web_search", "gist": ""},
                  {"tool_name": "fetch_channel_history", "gist": "limit=50"}]
    rig = MainRig(monkeypatch, decisions=[_decision("post", "the revised answer")],
                  metadata={"tool_provenance": provenance})
    await rig.run()

    persist = rig.bot.processor._persist_tool_provenance
    persist.assert_called_once()
    assert persist.call_args.args[1] == "200.1"       # the NEW ts
    assert persist.call_args.args[3] is provenance    # unchanged, not rebuilt from the revision
    assert rig.response.content == "the revised answer"
    assert [s["text"] for s in rig.sent] == ["the revised answer"]
    assert rig.turn.committed_destinations[0].text == "the revised answer"


@pytest.mark.asyncio
async def test_main_delivery_exception_suppresses_both_artifact_rescues(events, monkeypatch):
    """FIX-1 at the main site: the closure's send RAISES with BOTH collections populated. The
    runner returns None on `delivery_exception`, so main.py's r6-1 gates fire exactly as they
    do for `delivery_failed` — no artifact publish, no image rescue, no fallback send."""
    published = AsyncMock()
    monkeypatch.setattr("message_processor.artifacts.publish_artifacts", published)
    rig = MainRig(monkeypatch, decisions=[_decision("post", None)],
                  metadata={"artifact_containers": ["cntr_1"],
                            "sandbox_image_assets": [{"asset": "a1"}],
                            "mounted_digests": []},
                  deliver_result="raise")
    rig.bot._rescue_sandbox_images = AsyncMock()
    await rig.run()

    published.assert_not_awaited()
    rig.bot._rescue_sandbox_images.assert_not_awaited()
    assert rig.sent == []                             # no fallback send anywhere
    assert (rig.turn.reconsider.outcome, rig.turn.reconsider.error) == (
        "error_dropped", "delivery_exception")
    assert rig.response.metadata["posted"] is False
    assert len(_named(events, "stale_send")) == 1     # the runner's row; terminal added none
    assert len(_named(events, "reconsider_outcome")) == 1
    assert len(rig.decider.calls) == 1                # a single runner invocation


@pytest.mark.asyncio
async def test_main_delivery_failed_suppresses_both_artifact_rescues(events, monkeypatch):
    """§4b r6-1, the NORMAL-RETURN branch specifically: with BOTH collections populated, a
    delivery_failed reconsideration yields NO artifact publish and NO sandbox-image rescue,
    no fallback send, no second runner — and the emitted outcome stands."""
    published = AsyncMock()
    monkeypatch.setattr("message_processor.artifacts.publish_artifacts", published)
    rig = MainRig(monkeypatch, decisions=[_decision("post", None)],
                  metadata={"artifact_containers": ["cntr_1"],
                            "sandbox_image_assets": [{"asset": "a1"}],
                            "mounted_digests": []},
                  deliver_result="none")
    rig.bot._rescue_sandbox_images = AsyncMock()
    await rig.run()

    published.assert_not_awaited()
    rig.bot._rescue_sandbox_images.assert_not_awaited()
    assert rig.sent == []                            # no fallback send anywhere
    assert (rig.turn.reconsider.outcome, rig.turn.reconsider.error) == (
        "error_dropped", "delivery_failed")
    assert rig.response.metadata["posted"] is False
    assert len(_named(events, "stale_send")) == 1    # the runner's row; terminal added none
    assert len(_named(events, "reconsider_outcome")) == 1
    assert len(rig.decider.calls) == 1               # a single runner invocation
    # The terminal label is the delivery failure, never "suppressed, room saw nothing".
    assert _named(events, "turn_outcome")[0]["kind"] == "delivery_failed"


@pytest.mark.asyncio
async def test_main_terminal_catch_skips_emission_for_marked_suppressions_only(
        events, monkeypatch):
    """Single-owner both ways: a MARKED suppression reaching the terminal catch emits no second
    stale_send row (cleanup still runs); an UNMARKED one is the catch's to emit, with
    turn_id."""
    bot = _main_bot()
    message = _main_message()
    marked = StaleSendSuppressed(scope=("thread", CH, TRIGGER_TS), last_seen_ts=TRIGGER_TS,
                                 observed_latest_ts=NEWER_TS, surface="final_post")
    marked.telemetry_recorded = True
    bot.processor.process_message = AsyncMock(side_effect=marked)
    await bot.handle_message(message, _main_client(AsyncMock()))
    assert _named(events, "stale_send") == []

    events.clear()
    unmarked = StaleSendSuppressed(scope=("thread", CH, TRIGGER_TS), last_seen_ts=TRIGGER_TS,
                                   observed_latest_ts=NEWER_TS, surface="final_post")
    bot.processor.process_message = AsyncMock(side_effect=unmarked)
    await bot.handle_message(message, _main_client(AsyncMock()))
    rows = _named(events, "stale_send")
    assert len(rows) == 1 and rows[0]["turn_id"]


@pytest.mark.asyncio
async def test_main_once_gate_suppression_gets_exactly_one_terminal_row(events, monkeypatch):
    """A suppression observed AFTER the runner ended: rethrown unmarked by the wrapper,
    emitted once by the terminal catch, and no second reconsider outcome appears."""
    rig = MainRig(monkeypatch, decisions=[])

    async def _process(message, client, thinking_id=None, turn=None):
        rig.turn = turn
        pin_channel_turn(turn, trigger_ts=TRIGGER_TS, origin_thread_ts=TRIGGER_TS)
        turn.reconsider = SimpleNamespace(outcome="skipped", passes=1,
                                          as_payload=lambda: {"outcome": "skipped",
                                                              "passes": 1})
        rig.bot.watermarks.begin_turn(_msg(ts=NEWER_TS, thread=TRIGGER_TS, sender="U2"))
        return rig.response

    rig.bot.processor.process_message = AsyncMock(side_effect=_process)
    await rig.run()
    assert rig.sent == []
    rows = _named(events, "stale_send")
    assert len(rows) == 1 and rows[0]["turn_id"] == rig.turn.turn_id
    assert _named(events, "reconsider_outcome") == []
    assert rig.decider.calls == []                   # no second runner


@pytest.mark.asyncio
async def test_main_dm_suppression_drops_as_today_with_no_runner(events, monkeypatch):
    rig = MainRig(monkeypatch, decisions=[_decision("post", None)], channel="D123")
    await rig.run()
    assert rig.sent == []                            # suppressed, nothing posted
    assert rig.decider.calls == []                   # the runner never ran
    assert rig.turn.reconsider is None
    assert len(_named(events, "stale_send")) == 1    # the terminal catch's row


# =============================================================================== telemetry


@pytest.mark.asyncio
async def test_stale_send_rows_on_success_and_give_up_are_single_owner(events, monkeypatch):
    """One row per suppression EVENT, wherever the invocation ends — never zero, never two."""
    # Success:
    rig = _Rig(monkeypatch, script=[_decision("post", None)])
    rig.race()
    await rig.run(rig.accepting_deliver())
    assert len(_named(events, "stale_send")) == 1
    events.clear()
    # Give-up:
    rig2 = _Rig(monkeypatch, script=[ReconsiderationDecisionError("empty")])
    rig2.race()
    with pytest.raises(StaleSendSuppressed):
        await rig2.run(rig2.accepting_deliver())
    assert len(_named(events, "stale_send")) == 1
