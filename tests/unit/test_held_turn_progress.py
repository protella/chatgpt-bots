"""Agent surfaces §G (rulings GR1–GR13): held turns show progress once committed.

A held turn (silence-capable or destination-open) still buffers its whole answer; once it
commits to work it shows a TEMPORARY progress card — its own message, a chrome receipt — that
is deleted when the turn backs out, or right after the real answer lands. Every mock stream
here terminates and yields real strings (pitfall #6).
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

import message_processor.reconsideration as reconsideration
from config import config
from message_processor import participation_telemetry, user_stop
from message_processor.base import committed_queue_coverage
from message_processor.stale_send_guard import ConversationWatermarks, StaleSendSuppressed
from message_processor.turn_runtime import TurnRuntime
from openai_client.api.responses import ReconsiderationDecision
from tests.unit.channel_turn_harness import pin_channel_turn, thread_config
from tests.unit.test_reconsideration import (CH, NEWER_TS, TRIGGER_TS, SiteOpenAI, SiteSlack,
                                             _fresh_stream, _msg, _real_ledger, _site_processor,
                                             _site_thread_state, receipt_service)  # noqa: F401

_ = receipt_service      # the real receipt writer fixture, re-exported for these tests

PLAN = {"title": "Writing the essay", "steps": [
    {"id": "a", "title": "Draft it", "status": "in_progress", "result": None}]}


@pytest.fixture(autouse=True)
def _clean_stop_registry():
    user_stop.clear_for_tests()
    yield
    user_stop.clear_for_tests()


class HeldSlack(SiteSlack):
    """SiteSlack plus the card surface and reaction, recording one ordered event log."""

    def __init__(self, *, delete_ok: bool = True, card_raises: Optional[BaseException] = None,
                 on_delete=None, **kw):
        super().__init__(**kw)
        self.events: List[tuple] = []
        self.cards: List[Dict[str, Any]] = []
        self.delete_ok = delete_ok
        self.card_raises = card_raises
        self.on_delete = on_delete

    async def post_status_card(self, channel, thread, text, blocks, username=None,
                               receipts=None, *, receipt_class, error_out=None, metadata=None):
        if self.card_raises is not None:
            raise self.card_raises
        ts = f"card-{len(self.cards) + 1}"
        self.cards.append({"ts": ts, "thread": thread, "text": text, "blocks": blocks,
                           "receipt_class": receipt_class, "metadata": metadata})
        self.events.append(("card", ts))
        await self._receipt(ts, receipts, "chrome", thread, receipt_class)
        return ts

    async def update_status_card(self, channel, ts, text, blocks, receipts=None):
        self.events.append(("card_update", ts))
        return True

    async def delete_message(self, channel, ts):
        if self.on_delete is not None:
            await self.on_delete(ts)
        self.events.append(("delete", ts))
        ok = self.delete_ok(ts) if callable(self.delete_ok) else self.delete_ok
        return ok

    async def _reserve_and_react_owned(self, channel, ts, emoji):
        self.events.append(("eyes", ts))
        return {"ok": True}, "ack-lease"

    async def send_message(self, *args, **kwargs):
        stop_check = kwargs.pop("stop_check", None)     # SiteSlack's replica predates it
        if stop_check is not None and stop_check():
            return None
        posted = await super().send_message(*args, **kwargs)
        self.events.append(("post", posted))
        return posted


class PlanningOpenAI(SiteOpenAI):
    """A tool round that writes a plan (through the turn's sink, as update_plan does), then the
    answer text. `before_end` runs after the text, before the loop returns."""

    def __init__(self, turn_ref: Dict[str, Any], final_text: str = "The essay.",
                 decisions: Optional[List[Any]] = None, plan: Optional[dict] = None,
                 before_end=None, fail_with: Optional[BaseException] = None,
                 after_plan=None, fail_times: int = 1):
        super().__init__(final_text, chunks=[final_text], decisions=decisions)
        self.turn_ref = turn_ref
        self.plan = PLAN if plan is None else plan
        self.before_end = before_end
        self.after_plan = after_plan
        self.fail = fail_with
        self.fail_times = fail_times
        self.calls = 0

    async def create_streaming_response_with_tool_loop(
            self, messages=None, tools=None, registry=None, tool_context=None,
            stream_callback=None, tool_callback=None, hidden_suppression_sink=None, **kw):
        self.calls += 1
        sink = getattr(self.turn_ref["turn"], "plan_sink", None)
        if callable(sink) and self.plan:
            await sink(self.plan)
        if self.after_plan is not None:
            self.after_plan()
        if self.fail is not None and self.calls <= self.fail_times:
            raise self.fail
        for chunk in self.chunks:
            await stream_callback(chunk)
        if self.before_end is not None:
            self.before_end()
        await stream_callback(None)
        return {"text": self.final_text, "segments": [self.final_text], "tools_used": [],
                "local_tool_calls": []}


class HeldRig:
    def __init__(self, monkeypatch, *, slack: Optional[HeldSlack] = None, race: bool = False,
                 open_destination: bool = False, flag: bool = True, receipts: bool = False,
                 **openai_kw):
        monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
        monkeypatch.setattr(config, "enable_ack_reaction", True, raising=False)
        monkeypatch.setattr(config, "slack_stream_task_cards", True)
        monkeypatch.setattr(config, "slack_held_turn_progress", flag)
        self.slack = slack if slack is not None else HeldSlack()
        self.turn_ref: Dict[str, Any] = {}
        self.openai = PlanningOpenAI(self.turn_ref, **openai_kw)
        self.processor = _site_processor(self.openai)
        self.processor._build_tools_array = MagicMock(
            return_value=[{"type": "function", "name": "update_plan"}])
        self.processor._build_tool_context = MagicMock(
            return_value=SimpleNamespace(background_job_started=False, sandbox_image_assets=[],
                                         mounted_files=[], current_input=None,
                                         system_prompt=None, model=None))
        self.processor._persist_destination_provenance = MagicMock()
        self.message = _msg()
        self.message.metadata[user_stop.SINCE_KEY] = time.time() - 1
        self.marks = ConversationWatermarks()
        self.lease = self.marks.begin_turn(self.message)
        self.turn = TurnRuntime.for_message(self.message, channel_post_allowed=open_destination)
        self.turn.send_lease = self.lease
        self.turn.silence_capable = True
        self.turn_ref["turn"] = self.turn
        self.ledger = _real_ledger(self.turn.turn_id) if receipts else None
        self.turn.receipt_ledger = self.ledger
        self.processor.mcp_manager = SimpleNamespace(
            get_server_labels=lambda: ["srv", "other"],
            cache_discovered_tools_payload=lambda *a, **k: None)
        cfg = thread_config()
        self.turn.capability_profile = cfg
        pin_channel_turn(self.turn, trigger_ts=TRIGGER_TS, origin_thread_ts=TRIGGER_TS,
                         config=cfg, prepared=(SimpleNamespace(), {}, False, "", None),
                         channel_id=CH)
        if race:
            async def _snapshot(**kwargs):
                return SimpleNamespace(stream=_fresh_stream())
            monkeypatch.setattr(reconsideration, "build_reconsideration_snapshot", _snapshot)

    async def run(self):
        return await self.processor._handle_streaming_text_response(
            "hi", _site_thread_state(), self.slack, self.message, None, None, turn=self.turn)

    def kinds(self) -> List[str]:
        return [e[0] for e in self.slack.events]

    def race_now(self) -> None:
        """A newer message in the trigger's thread — the frontier advances."""
        self.marks.begin_turn(_msg(ts=NEWER_TS, thread=TRIGGER_TS, sender="U2"))


def _revised(text: str = "The revised essay.") -> ReconsiderationDecision:
    return ReconsiderationDecision(decision="post", text=text, answers_newer=True)


# ------------------------------------------------------------------ the acceptance sequence

@pytest.mark.asyncio
async def test_plan_card_then_suppression_then_a_fresh_revised_answer(monkeypatch,
                                                                      receipt_service):
    from message_processor.client_contract import Message
    from message_processor.thread_manager import AsyncThreadStateManager

    holder: Dict[str, Any] = {}
    rows_while_live: List[Any] = []

    async def _check_receipt(ts):
        rows_while_live.append(await receipt_service.get_receipt_async("T1", CH, ts))

    slack = HeldSlack(on_delete=_check_receipt)
    rig = HeldRig(monkeypatch, slack=slack, race=True, receipts=True, decisions=[_revised()],
                  after_plan=lambda: holder["rig"].race_now())   # frontier moves AFTER the card
    holder["rig"] = rig
    resp = await rig.run()
    card = rig.slack.cards[0]
    assert card["receipt_class"] == "chrome" and card["thread"] == TRIGGER_TS
    assert card["metadata"]["event_type"] == "held_progress"
    assert card["blocks"][0]["type"] == "plan"
    assert rows_while_live[0]["state"] == "chrome"                 # the real receipt writer
    assert await receipt_service.get_receipt_async("T1", CH, card["ts"]) is None  # aborted
    kinds = rig.kinds()
    assert kinds.index("card") < kinds.index("delete") < kinds.index("post")
    answer_ts = [e[1] for e in rig.slack.events if e[0] == "post" and e[1]][-1]
    assert answer_ts != card["ts"]
    assert [p["text"] for p in rig.slack.posts] == ["The revised essay."]
    assert rig.turn.reconsider.outcome == "posted_revised"
    coverage = committed_queue_coverage(rig.turn)
    assert coverage is not None and coverage.reply_ts == answer_ts
    # Queue retirement actually exercised: the newer follow-up is retired, nothing else.
    tm = AsyncThreadStateManager()
    key = f"{CH}:{TRIGGER_TS}"
    newer = Message(text="the racer", user_id="U2", channel_id=CH, thread_id=TRIGGER_TS,
                    metadata={"ts": NEWER_TS})
    other = Message(text="unrelated", user_id="U2", channel_id=CH, thread_id=TRIGGER_TS,
                    metadata={"ts": "12.0"})
    tm.enqueue_pending(key, newer)
    tm.enqueue_pending(key, other)
    assert tm.retire_covered_pending(key, coverage.covered_ts) == [newer]
    assert resp.metadata["posted"] is True
    assert all(d.first_ts != card["ts"] for d in rig.turn.destinations)
    assert rig.ledger.pending_ts == [answer_ts]                    # the card is no answer part


@pytest.mark.asyncio
async def test_a_skipped_reconsideration_deletes_the_card_and_posts_nothing(monkeypatch):
    holder: Dict[str, Any] = {}
    rig = HeldRig(monkeypatch, race=True,
                  decisions=[ReconsiderationDecision(decision="skip", text=None)],
                  after_plan=lambda: holder["rig"].race_now())
    holder["rig"] = rig
    with pytest.raises(StaleSendSuppressed):
        await rig.run()
    assert rig.slack.posts == [] and "delete" in rig.kinds()
    assert rig.turn.held_card_removed is not None


@pytest.mark.asyncio
async def test_a_partially_delivered_answer_still_retires_the_card(monkeypatch):
    long_text = ("word " * 400 + "\n\n") * 3
    rig = HeldRig(monkeypatch, slack=HeldSlack(fail_chunk=2), final_text=long_text)
    await rig.run()
    kinds = rig.kinds()
    assert "post" in kinds and "delete" in kinds
    assert kinds.index("post") < kinds.index("delete")          # the answer's part 1 first


class RealSendSlack(HeldSlack):
    """HeldSlack, but the answer goes through the REAL send_message over a fake Slack client."""

    def __init__(self, on_post=None, **kw):
        from slack_client.messaging import SlackMessagingMixin
        super().__init__(**kw)
        self.app = MagicMock()
        self.posted: List[Dict[str, Any]] = []
        self._on_post = on_post

        async def _post(**kwargs):
            self.posted.append(kwargs)
            if self._on_post is not None:
                self._on_post(kwargs)
            return {"ok": True, "ts": f"real-{len(self.posted)}"}
        self.app.client.chat_postMessage = _post
        mixin = SlackMessagingMixin
        for name in ("_split_message", "_compose_reply_with_footer", "_markdown_tail_for",
                     "_markdown_body", "_reconcile_uncertain_post"):
            setattr(self, name, getattr(mixin, name).__get__(self))
        self._real_send = mixin.send_message.__get__(self)

    _SECTION_TEXT_LIMIT = 2900
    _FOOTER_INLINE_MAX = 180

    def log_info(self, *a, **k): pass
    log_debug = log_warning = log_error = log_info

    async def _record_receipt(self, *a, **k):
        return None

    async def send_message(self, *args, **kwargs):
        posted = await self._real_send(*args, **kwargs)
        self.events.append(("post", posted))
        return posted


@pytest.mark.asyncio
async def test_stop_during_final_delivery_publishes_nothing_more(monkeypatch):
    monkeypatch.setattr(config, "slack_stop_button", True)

    def _press_after_first_part(kwargs):
        user_stop.request_stop(CH, TRIGGER_TS, "U1", event_ts=str(time.time() + 0.5))
    slack = RealSendSlack(on_post=_press_after_first_part)
    long_text = ("word " * 400 + "\n\n") * 3
    rig = HeldRig(monkeypatch, slack=slack, final_text=long_text)
    await rig.run()
    assert len(slack.posted) == 1                                   # part 1 only, then Stop
    assert not any("cut off" in (kw.get("text") or "") for kw in slack.posted)
    assert "delete" in rig.kinds()


@pytest.mark.asyncio
async def test_stop_before_final_delivery_posts_nothing_and_removes_the_card(monkeypatch):
    monkeypatch.setattr(config, "slack_stop_button", True)

    def _press():
        user_stop.request_stop(CH, TRIGGER_TS, "U1", event_ts=str(time.time() + 0.5))
    rig = HeldRig(monkeypatch, before_end=_press)
    resp = await rig.run()
    assert rig.slack.posts == [] and "delete" in rig.kinds()
    assert resp.metadata["user_stopped"] is True
    assert rig.turn.held_card_removed == "stop"


@pytest.mark.asyncio
async def test_stop_during_suppression_cleanup_skips_reconsideration(monkeypatch):
    monkeypatch.setattr(config, "slack_stop_button", True)

    async def _press_while_deleting(ts):
        user_stop.request_stop(CH, TRIGGER_TS, "U1", event_ts=str(time.time() + 0.5))
    holder: Dict[str, Any] = {}
    rig = HeldRig(monkeypatch, slack=HeldSlack(on_delete=_press_while_deleting), race=True,
                  decisions=[_revised()], after_plan=lambda: holder["rig"].race_now())
    holder["rig"] = rig
    resp = await rig.run()
    assert rig.slack.posts == [] and resp.metadata["user_stopped"] is True
    assert rig.openai.create_reconsideration_decision.calls == []  # never reconsidered


@pytest.mark.asyncio
async def test_a_real_mcp_retry_retires_the_card_before_the_successor(monkeypatch):
    rig = HeldRig(monkeypatch, fail_with=RuntimeError("Error: MCP server 'srv' failed"))
    await rig.run()
    assert rig.openai.calls == 2                                    # the recursive attempt ran
    events = rig.slack.events
    assert [e for e in events if e[0] in ("card", "delete")] == [
        ("card", "card-1"), ("delete", "card-1"), ("card", "card-2"), ("delete", "card-2")]
    assert events.index(("delete", "card-1")) < events.index(("card", "card-2"))
    assert len(rig.slack.posts) == 1 and rig.slack.posts[0]["text"].startswith("The essay.")
    assert rig.turn.held_card_removed == "answer"                   # the last reason wins
    assert getattr(rig.turn, "held_progress_hook", None) is None


@pytest.mark.asyncio
async def test_a_failed_delete_means_no_second_card_across_the_retry(monkeypatch):
    rig = HeldRig(monkeypatch, slack=HeldSlack(delete_ok=lambda ts: ts != "card-1"),
                  fail_with=RuntimeError("Error: MCP server 'srv' failed"))
    await rig.run()
    assert rig.openai.calls == 2 and len(rig.slack.cards) == 1      # review 8: turn-level
    assert rig.turn.held_card_remove_failed is True


@pytest.mark.asyncio
async def test_a_failed_card_deletion_keeps_the_chrome_and_says_so(monkeypatch):
    rig = HeldRig(monkeypatch, slack=HeldSlack(delete_ok=False))
    await rig.run()
    assert [p["text"] for p in rig.slack.posts] == ["The essay."]
    assert rig.turn.held_card_remove_failed is True and rig.turn.held_card_removed is None


@pytest.mark.asyncio
async def test_the_participation_probe_ignores_a_held_progress_card():
    from slack_client.event_handlers.message_events import SlackMessageEventsMixin

    def _host(messages):
        host = MagicMock()
        host.app.client.conversations_replies = AsyncMock(return_value={"messages": messages})
        host.is_own_message = lambda m: m.get("user") == "UBOT"
        host.classify_sender = lambda m: "human"
        return host

    card = {"ts": "10.5", "user": "UBOT", "text": ":hourglass_flowing_sand: Working on a reply…",
            "blocks": [{"type": "plan", "title": "x", "tasks": []}]}       # history: no metadata
    present, humans, bots = await SlackMessageEventsMixin._thread_participation(
        _host([{"ts": "10.0", "user": "U1", "text": "hi"}, card]), CH, "10.0")
    assert present is False and humans == 1 and bots == 0
    # The same text WITHOUT the plan block is a real reply of ours: presence, as ever.
    plain = {"ts": "10.6", "user": "UBOT", "text": "⏳ Working on a reply…"}
    present, _, _ = await SlackMessageEventsMixin._thread_participation(
        _host([{"ts": "10.0", "user": "U1", "text": "hi"}, plain]), CH, "10.0")
    assert present is True


@pytest.mark.asyncio
async def test_reconciliation_skips_held_cards_and_finds_a_carded_stream():
    from slack_client.messaging import SlackMessagingMixin

    now = time.time()
    bot = MagicMock()
    bot.is_own_message = lambda m: True
    bot._normalize_for_match = SlackMessagingMixin._normalize_for_match
    bot._RECONCILE_WINDOW_SECS = 120
    bot._RECONCILE_CLOCK_SKEW_SECS = 5
    bot._RECONCILE_PREFIX_MIN_LEN = 200
    bot._RECONCILE_MAX_PAGES = 1
    card = {"ts": f"{now:.6f}", "text": "The answer to your question.",
            "metadata": {"event_type": "held_progress"}}
    carded = {"ts": f"{now + 1:.6f}",
              "text": "Searching the web The answer to your question."}
    bot.app.client.conversations_replies = AsyncMock(return_value={"messages": [card]})
    reconcile = SlackMessagingMixin._reconcile_uncertain_post
    assert await reconcile(bot, CH, TRIGGER_TS, "The answer to your question.", now) is None
    bot.app.client.conversations_replies = AsyncMock(return_value={"messages": [carded]})
    assert await reconcile(bot, CH, TRIGGER_TS, "The answer to your question.", now) is None
    assert await reconcile(bot, CH, TRIGGER_TS, "The answer to your question.", now,
                           contained=True) == carded["ts"]


def test_plan_task_ids_are_unique_however_the_model_spells_them():
    from streaming.native_sink import StepCards
    steps = StepCards()
    steps.apply_plan("P", [{"id": "a:b", "title": "One", "status": "complete", "result": "ok"},
                           {"id": "a-b", "title": "Two", "status": "pending", "result": None}])
    snap = steps.progress_snapshot()
    assert snap is not None
    ids = [t["id"] for t in snap[1]]
    assert len(set(ids)) == 2 and snap[1][0]["result"] == "ok"


@pytest.mark.asyncio
async def test_an_open_destination_turn_shows_eyes_only(monkeypatch):
    rig = HeldRig(monkeypatch, open_destination=True)
    assert rig.turn.destination_selected is False
    await rig.run()
    assert rig.slack.cards == [] and ("eyes", TRIGGER_TS) in rig.slack.events
    assert rig.turn.held_first_signal == "plan"


@pytest.mark.asyncio
async def test_an_open_destination_turn_never_flashes_a_card_when_it_picks_the_thread(
        monkeypatch):
    # Owner 2026-11-21: plan first, then "[[reply:thread]]" binds the destination — 👀 only.
    rig = HeldRig(monkeypatch, open_destination=True,
                  final_text="[[reply:thread]]The answer.")
    await rig.run()
    assert rig.turn.destination_selected is True
    assert rig.slack.cards == [] and ("eyes", TRIGGER_TS) in rig.slack.events


@pytest.mark.asyncio
async def test_plan_then_delivered_deletes_the_card_after_the_answer_lands(monkeypatch):
    rig = HeldRig(monkeypatch)
    resp = await rig.run()
    kinds = rig.kinds()
    assert kinds.index("card") < kinds.index("post") < kinds.index("delete")
    assert resp.metadata["posted"] is True and rig.turn.held_card_opened is True
    assert rig.turn.held_card_removed == "answer"


@pytest.mark.asyncio
async def test_flag_off_changes_nothing(monkeypatch):
    rig = HeldRig(monkeypatch, flag=False)
    await rig.run()
    assert rig.slack.cards == [] and "eyes" not in rig.kinds() and "delete" not in rig.kinds()
    assert rig.turn.held_first_signal is None


def test_no_held_facts_with_task_cards_off(monkeypatch):
    monkeypatch.setattr(config, "slack_held_turn_progress", True)
    monkeypatch.setattr(config, "slack_stream_task_cards", False)
    turn = TurnRuntime.for_message(_msg(), channel_post_allowed=False)
    turn.note_first_visible("answer")
    turn.note_progress_signal("work")
    assert turn.held_first_visible is None and turn.held_first_signal is None


def test_outcome_rows_carry_the_held_facts_only_with_the_flag(monkeypatch):
    monkeypatch.setattr(config, "slack_stream_task_cards", True)
    rows: List[Dict[str, Any]] = []
    monkeypatch.setattr(participation_telemetry, "record",
                        lambda event, **fields: rows.append({"event": event, **fields}))
    turn = SimpleNamespace(turn_id="t1", destinations=[], edits=[], held_first_signal="plan",
                           held_first_signal_at=100.5, held_first_visible="card",
                           held_first_visible_at=101.0, held_card_opened=True,
                           held_card_removed="delivered", held_card_remove_failed=False)
    for flag in (False, True):
        monkeypatch.setattr(config, "slack_held_turn_progress", flag)
        rows.clear()
        setattr(turn, participation_telemetry.TURN_OUTCOME_KEY, False)
        participation_telemetry.emit_turn_outcome(turn, channel_id=CH, trigger_ts="10.0",
                                                  kind="reply", admitted_at=100.0)
        row = rows[0]
        if flag:
            assert row["first_signal"] == "plan" and row["first_signal_ms"] == 500
            assert row["first_visible"] == "card" and row["first_visible_ms"] == 1000
            assert row["held_card_opened"] is True and row["held_card_removed"] == "delivered"
        else:
            assert "first_signal" not in row or row["first_signal"] is None
