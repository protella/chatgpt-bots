"""What the room ACTUALLY saw, and when we knew it (codex P2 review, findings 13/14/17).

Three questions, one theme — a record that describes the intention rather than the delivery:

* F13 the transport reported the ts of the first chunk and nothing about the rest, so a split
  that aborted at chunk 2 of 4 was COMMITTED as the whole answer. Channel memory reads committed
  records, so the bot came away remembering an exchange the room never had.
* F14 destination records were appended after the whole send coroutine returned. A cancellation
  between the first accepted chunk and the last left visible text and receipts with nothing in
  the ledger claiming them. `kind` was also wrong: an ordinary multipart reply filed itself as
  `reply`, and a legacy overflow that genuinely split filed itself as `stream`.
* F17 the timeout and streaming-fallback retries popped the last user message off
  ThreadState.messages. A channel turn never appends its input there, so the pop took a genuine
  historical entry off a reused state instead of this turn's copy.
"""
import asyncio
import inspect
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest
from slack_sdk.errors import SlackApiError

from base_client import Message, Response
from markdown_converter import MarkdownConverter
from message_processor.turn_runtime import TurnRuntime
from slack_client.formatting.text import SlackFormattingMixin
from slack_client.messaging import Delivery, SlackMessagingMixin
from slack_client.utilities import SlackUtilitiesMixin
from tool_registry import ToolContext


class _Bot(SlackMessagingMixin, SlackFormattingMixin, SlackUtilitiesMixin):
    """The real messaging mixins over a mocked Slack web client."""

    MAX_MESSAGE_LENGTH = 3900

    def __init__(self):
        self.bot_id = "B07SELF"
        self.bot_user_id = "U07SELF"
        self.app_id = None
        self.app = MagicMock()
        self.markdown_converter = MarkdownConverter(platform="slack")

    def log_info(self, *a, **k): pass
    log_debug = log_warning = log_error = log_info

    # NO format_text override. The mixin's real one runs, which is the point of r2-10: an identity
    # double made "what the room saw" trivially equal to what the caller asked for, and hid that
    # Delivery was reporting the pre-format text.


def _slack_error(error="msg_too_long", status=400):
    resp = MagicMock()
    resp.get = lambda key, default=None: {"error": error}.get(key, default)
    resp.status_code = status
    resp.headers = {}
    return SlackApiError(message=error, response=resp)


def _long_text(paragraphs=8):
    """Long enough that fence_safe_chunks produces several chunks."""
    return ("para " * 300 + "\n\n") * paragraphs


# --------------------------------------------------------------- F13: the transport reports

@pytest.mark.asyncio
async def test_a_whole_send_reports_the_text_slack_accepted():
    b = _Bot()
    b.app.client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "111.0"})
    meta = {}
    ts = await b.send_message("C1", "T1", "short reply", meta_out=meta)
    delivery = meta["delivery"]
    assert (ts, delivery.first_ts) == ("111.0", "111.0")
    assert delivery.complete is True and delivery.split is False
    assert delivery.text == "short reply"
    assert (delivery.parts_delivered, delivery.parts_total) == (1, 1)


@pytest.mark.asyncio
async def test_a_split_that_aborts_reports_only_what_slack_took(monkeypatch):
    """The heart of F13. Chunk 2 fails both attempts, so parts 2..N never post — and what comes
    back must describe the prefix that landed, not the answer we meant to give."""
    b = _Bot()
    monkeypatch.setattr("slack_client.messaging.asyncio.sleep", AsyncMock())
    posted = []

    async def post(channel, thread_ts, text, **kw):
        posted.append(text)
        if len(posted) >= 2:
            raise _slack_error()
        return {"ok": True, "ts": f"ts{len(posted)}"}

    b.app.client.chat_postMessage = AsyncMock(side_effect=post)
    whole = _long_text()
    meta = {}
    first_ts = await b.send_message("C1", "T1", whole, meta_out=meta)

    assert first_ts == "ts1"
    delivery = meta["delivery"]
    assert delivery.complete is False, "a truncated delivery must not claim completeness"
    assert delivery.truncated_at == 2 and delivery.parts_delivered == 1
    assert delivery.parts_total > 1 and delivery.split is True
    # The delivered text is a PREFIX of the answer, never the whole of it.
    assert len(delivery.text) < len(whole)
    assert whole.startswith(delivery.text[:200])
    assert meta["split_truncated"] is True


@pytest.mark.asyncio
async def test_a_split_that_lands_whole_reports_exactly_what_slack_was_given():
    """r2-10: this used to report the caller's own markdown for a complete split — a delivery
    described in text the room never received."""
    from message_markers import CONTINUATION_HEAD

    b = _Bot()
    posted = []

    async def post(channel, thread_ts, text, **kw):
        posted.append(text)
        return {"ok": True, "ts": f"ts{len(posted)}"}

    b.app.client.chat_postMessage = AsyncMock(side_effect=post)
    meta = {}
    await b.send_message("C1", "T1", _long_text(), meta_out=meta)
    delivery = meta["delivery"]

    assert delivery.complete is True and delivery.split is True
    assert delivery.parts_delivered == delivery.parts_total == len(posted) > 1
    # The continuation heads are chrome the rebuild strips; everything else is verbatim.
    bodies = [posted[0]] + [p.split(f"{CONTINUATION_HEAD}\n\n", 1)[1] for p in posted[1:]]
    assert delivery.text == "\n\n".join(bodies)


@pytest.mark.asyncio
async def test_the_reported_text_is_the_formatted_text_slack_accepted():
    """Formatting is not cosmetic to this record: channel memory commits `delivery.text` as what
    the room saw, so reporting the source markdown made that claim false in every send
    `format_text` touched."""
    b = _Bot()
    posted = []

    async def post(channel, thread_ts, text, **kw):
        posted.append(text)
        return {"ok": True, "ts": "111.0"}

    b.app.client.chat_postMessage = AsyncMock(side_effect=post)
    meta = {}
    source = "**bold** and a [link](https://example.com)"
    await b.send_message("C1", "T1", source, meta_out=meta)

    assert posted == [meta["delivery"].text]
    assert meta["delivery"].text != source, "format_text rewrote it; the record must follow"


# ----------------------------------------------------- F14: observed at the first accepted part

@pytest.mark.asyncio
async def test_the_first_chunk_is_announced_before_the_next_one_is_attempted(monkeypatch):
    b = _Bot()
    monkeypatch.setattr("slack_client.messaging.asyncio.sleep", AsyncMock())
    events = []

    async def post(channel, thread_ts, text, **kw):
        events.append("post")
        return {"ok": True, "ts": f"ts{len([e for e in events if e == 'post'])}"}

    b.app.client.chat_postMessage = AsyncMock(side_effect=post)
    await b.send_message("C1", "T1", _long_text(),
                         on_first_accept=lambda ts: events.append(f"accept:{ts}"))
    assert events[0] == "post" and events[1] == "accept:ts1", events[:3]
    assert events.count("post") > 2, "expected a multi-chunk split"
    assert len([e for e in events if e.startswith("accept")]) == 1


@pytest.mark.asyncio
async def test_a_cancelled_split_still_leaves_an_observed_record():
    """The case the first-surface hook exists for: part 1 is in the room, the turn is cancelled
    before part 2, and the ledger must still say words landed there."""
    b = _Bot()
    turn = TurnRuntime()
    wedged = asyncio.Event()
    posted = []

    async def post(channel, thread_ts, text, **kw):
        posted.append(text)
        if len(posted) == 1:
            return {"ok": True, "ts": "ts1"}
        await wedged.wait()  # never set: the second chunk hangs
        return {"ok": True, "ts": "ts2"}

    b.app.client.chat_postMessage = AsyncMock(side_effect=post)

    def _observe(ts):
        turn.note_destination_observed(channel_id="C1", first_ts=ts, kind="reply",
                                       thread_root_ts="T1")

    task = asyncio.ensure_future(
        b.send_message("C1", "T1", _long_text(), on_first_accept=_observe))
    for _ in range(100):
        await asyncio.sleep(0)
        if len(posted) >= 2:
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(turn.destinations) == 1
    record = turn.destinations[0]
    assert record.first_ts == "ts1" and record.state == "observed"
    assert record.committed is False, "an interrupted send is seen, not finished"


# ------------------------------------------------------- F13/F14 through main.py's reply branch

def _make_bot():
    from main import ChatBotV2
    bot = ChatBotV2(platform="slack")
    bot.processor = MagicMock()
    bot.processor._persist_tool_provenance = MagicMock()
    return bot


def _client(delivery, first_ts="posted.1"):
    """A client whose send reports exactly the delivery this test is about."""
    c = MagicMock()
    c.send_thinking_indicator = AsyncMock(return_value=None)
    c.delete_message = AsyncMock()
    c.format_text = MagicMock(side_effect=lambda t: t)
    c.maybe_post_response_footer = AsyncMock()
    del c.attachable_footer_blocks  # keep the branch simple: no footer chrome

    async def _send(channel_id, thread_id, text, blocks=None, meta_out=None, lease=None,
                    surface=None, receipts=None, receipt_kind=None, receipt_class=None, on_first_accept=None):
        if on_first_accept is not None:
            on_first_accept(first_ts)
        if meta_out is not None and delivery is not None:
            meta_out["delivery"] = delivery
        return first_ts

    c.send_message = AsyncMock(side_effect=_send)
    return c


async def _run_reply(bot, client, content="the whole answer"):
    """Drive main.py's non-streamed reply branch and hand back the turn it opened.

    The turn is a local in handle_message; `_emit_turn_start` is the one seam it is handed to,
    which makes it the cheapest honest way to see the records the branch actually wrote.
    """
    captured = {}
    bot._emit_turn_start = lambda message, turn, **kw: captured.setdefault("turn", turn)
    resp = Response(type="text", content=content, metadata={"streamed": False, "model": "m"})
    bot.processor.process_message = AsyncMock(return_value=resp)
    message = Message(text="q", user_id="U1", channel_id="C1", thread_id="T1",
                      metadata={"ts": "200.0"})
    await bot.handle_message(message, client)
    return captured["turn"]


@pytest.mark.asyncio
async def test_main_commits_the_delivered_prefix_and_flags_it_incomplete():
    bot = _make_bot()
    delivery = Delivery(first_ts="posted.1", text="the delivered half", complete=False,
                        parts_delivered=1, parts_total=3, split=True, truncated_at=2)
    turn = await _run_reply(bot, _client(delivery),
                            content="the whole answer nobody fully received")

    assert len(turn.destinations) == 1
    record = turn.destinations[0]
    assert record.committed is True
    assert record.text == "the delivered half", "committed the intention, not the delivery"
    assert record.chars == len("the delivered half")
    assert record.complete is False
    assert record.kind == "split", "a multipart plain send is a split, not a plain reply"
    assert record.as_payload()["complete"] is False


@pytest.mark.asyncio
async def test_main_labels_a_whole_multipart_send_a_split():
    bot = _make_bot()
    delivery = Delivery(first_ts="posted.1", text="all of it", complete=True,
                        parts_delivered=2, parts_total=2, split=True)
    turn = await _run_reply(bot, _client(delivery), content="all of it")
    record = turn.destinations[0]
    assert record.kind == "split" and record.complete is True
    assert record.as_payload().get("complete") is None, (
        "a complete delivery keeps the payload shape every reader was pinned against")


@pytest.mark.asyncio
async def test_main_observes_the_reply_at_the_first_accepted_surface():
    """The observe hook fires from inside the send, so the record exists even for a delivery
    whose completion never comes back."""
    bot = _make_bot()
    captured = {}
    bot._emit_turn_start = lambda message, turn, **kw: captured.setdefault("turn", turn)
    mid_send = []
    client = _client(None)

    async def _send(channel_id, thread_id, text, blocks=None, meta_out=None, lease=None,
                    surface=None, receipts=None, receipt_kind=None, receipt_class=None, on_first_accept=None):
        on_first_accept("posted.1")
        # Read the ledger while the send is STILL RUNNING — this is the whole contract.
        mid_send.extend([(r.first_ts, r.state) for r in captured["turn"].destinations])
        return "posted.1"

    client.send_message = AsyncMock(side_effect=_send)
    resp = Response(type="text", content="hi", metadata={"streamed": False, "model": "m"})
    bot.processor.process_message = AsyncMock(return_value=resp)
    await bot.handle_message(
        Message(text="q", user_id="U1", channel_id="C1", thread_id="T1",
                metadata={"ts": "200.0"}), client)

    assert mid_send == [("posted.1", "observed")]


@pytest.mark.asyncio
async def test_channel_memory_reads_the_delivered_text_only():
    """main.py:_schedule_channel_memory is the consumer F13 was actually lying to."""
    from config import config

    bot = _make_bot()
    scheduled = []
    bot.processor._schedule_async_call = MagicMock(side_effect=scheduled.append)
    bot.processor.extract_channel_memory_from_exchange = MagicMock(return_value=None)
    turn = TurnRuntime()
    turn.stream_build_present = True
    turn.mark_destination_committed(first_ts="11.0", kind="split", text="only this landed",
                                    complete=False, channel_id="C1")
    message = Message(text="hi", user_id="U1", channel_id="C1", thread_id="10.0",
                      metadata={"ts": "10.0"})
    original = config.enable_memory_extraction_fallback
    config.enable_memory_extraction_fallback = True
    try:
        bot._schedule_channel_memory(message, turn)
    finally:
        config.enable_memory_extraction_fallback = original

    assert len(scheduled) == 1
    args = bot.processor.extract_channel_memory_from_exchange.call_args.args
    assert args[2] == "only this landed"


# -------------------------------------------------------------- F13/F14 through post_to_thread

def _post_host(post_side_effect):
    s = MagicMock()
    s.execute_post_to_thread = SlackMessagingMixin.execute_post_to_thread.__get__(s)
    s.send_message = SlackMessagingMixin.send_message.__get__(s)
    s.format_text = SlackFormattingMixin.format_text.__get__(s)
    s._encode_mentions = lambda t: t
    s.markdown_converter = MarkdownConverter(platform="slack")
    s._record_receipt = AsyncMock()
    s._compose_reply_with_footer = SlackMessagingMixin._compose_reply_with_footer.__get__(s)
    s._split_message = SlackMessagingMixin._split_message.__get__(s)
    s.MAX_MESSAGE_LENGTH = 3900
    s.app.client.chat_postMessage = AsyncMock(side_effect=post_side_effect)
    return s


def _ctx(turn):
    ctx = ToolContext(channel_id="C1", thread_ts="root.1", trigger_ts="msg.1",
                      client=MagicMock(), db=MagicMock())
    ctx.turn = turn
    return ctx


@pytest.mark.asyncio
async def test_post_to_thread_commits_only_the_parts_that_reached_the_thread(monkeypatch):
    monkeypatch.setattr("slack_client.messaging.asyncio.sleep", AsyncMock())
    posted = []

    async def post(channel, thread_ts, text, **kw):
        posted.append(text)
        if len(posted) >= 2:
            raise _slack_error()
        return {"ok": True, "ts": f"ts{len(posted)}"}

    host = _post_host(post)
    turn = TurnRuntime()
    whole = _long_text()
    out = await host.execute_post_to_thread(
        _ctx(turn), {"thread_ts": "OTHER.9", "text": whole})

    assert out["ok"] is True and out["truncated"] is True
    record = turn.destinations[0]
    assert record.thread_root_ts == "OTHER.9" and record.kind == "post_to_thread"
    assert record.committed is True and record.complete is False
    assert len(record.text) < len(whole)


@pytest.mark.asyncio
async def test_post_to_thread_observes_before_it_commits():
    states = []

    async def post(channel, thread_ts, text, **kw):
        return {"ok": True, "ts": "900.0"}

    host = _post_host(post)
    turn = TurnRuntime()
    real = turn.note_destination_observed

    def _spy(**kw):
        record = real(**kw)
        states.append(record.state)
        return record

    turn.note_destination_observed = _spy
    out = await host.execute_post_to_thread(
        _ctx(turn), {"thread_ts": "OTHER.9", "text": "hello over there"})

    assert out["ok"] is True and "truncated" not in out
    assert states[0] == "observed", "the first record was minted already committed"
    record = turn.destinations[0]
    assert record.committed is True and record.complete is True
    assert record.text == "hello over there"


# ------------------------------------------------------ F13: the legacy overflow continuation

def test_the_legacy_overflow_continuation_result_is_read():
    """Both overflow branches post the remainder as a second message. The result of that post is
    now READ: a continuation that never landed leaves a truncated answer in the room, and the
    committed record has to say the shorter thing."""
    import message_processor.handlers.text as th

    src = inspect.getsource(th)
    for marker in ("W1: the buffer can outgrow", "# Send the rest as new messages"):
        window = src[src.index(marker):src.index(marker) + 2400]
        assert "overflow_ts = await client.send_message(" in window
        assert "delivery_complete = False" in window
        assert "_delivered_without_tail(" in window
        assert "delivery_split = True" in window


def test_the_committed_stream_text_prefers_what_was_delivered():
    import message_processor.handlers.text as th

    src = inspect.getsource(th)
    window = src[src.index("if native_multipart or delivery_split:"):][:1200]
    assert "kind = DEST_KIND_SPLIT" in window
    # A turn that never streamed anywhere posted once: that is a reply, not a stream.
    assert "elif delivery_direct_post:\n                    kind = DEST_KIND_REPLY" in window
    assert "delivered_text_override if delivered_text_override is not None" in window
    assert "complete=delivery_complete" in window


def test_delivered_without_tail_drops_a_recognizable_undelivered_suffix():
    from message_processor.handlers.text import _delivered_without_tail

    assert _delivered_without_tail("first part\n\nsecond part", "second part") == "first part"
    # Conservative when the tail is not a suffix (attribution chrome landed in between): the
    # whole text comes back and only the completeness flag carries the bad news.
    assert _delivered_without_tail("first part", "something else") == "first part"
    assert _delivered_without_tail("first part", "") == "first part"


# ----------------------------------------------------------- F17: the retry pops are DM-only

def test_the_timeout_retry_pop_is_guarded_by_the_surface():
    import message_processor.base as base

    src = inspect.getsource(base)
    window = src[src.index("F7: the first attempt appended this turn's user message"):][:1200]
    assert "not channel_turn and thread_state.messages" in window


def test_the_streaming_fallback_pop_is_guarded_by_the_surface():
    import message_processor.handlers.text as th

    src = inspect.getsource(th)
    window = src[src.index("# Remove the message that was just added by streaming attempt"):][:900]
    assert "not channel_turn and thread_state.messages" in window


# ------------------------- r2-11: permanent prose never stands alone (codex P2 review round 2)

async def _channel_turn_with_a_prior_timeout(*, admission_fails: bool, had_timeout: bool = True,
                                            unsupported=(), open_top_level: bool = False,
                                            trigger_text: str = "what happened to the Q3 numbers",
                                            gate_sources=(), notice_lands: bool = True):
    """Drive the real process_message for a CHANNEL turn that owes prose, and report
    (the order of the steps that matter, the response, what admission saw)."""
    from unittest.mock import patch

    from message_processor.base import MessageProcessor
    from message_processor.channel_stream import StreamOverBudgetError

    with patch("message_processor.base.AsyncThreadStateManager"), \
         patch("message_processor.base.OpenAIClient"):
        p = MessageProcessor()

    order = []
    seen = {}
    state = SimpleNamespace(had_timeout=had_timeout, messages=[], thread_ts="10.0",
                            channel_id="C1", root_author=("U1", "human"), config_overrides={},
                            participants={}, current_model=None, has_trimmed_messages=False)

    async def _state(*a, **k):
        return state

    async def _stream(*a, **k):
        order.append("stream")
        return None

    async def _admit(*a, **k):
        order.append("admission")
        # What the request being measured would have been assembled from: whether the destination
        # was already settled and locked, and which failures it was told about.
        turn_arg = a[2] if len(a) > 2 else k.get("turn")
        seen["destination_selected"] = turn_arg.destination_selected
        seen["destination_locked"] = turn_arg.destination_locked
        seen["reply_destination"] = turn_arg.reply_destination
        seen["failed_attachment_names"] = k.get("failed_attachment_names")
        # base.py pins the turn context here, and the notice's delivery is recorded ON it [r4-4].
        turn_arg.channel_turn_context = SimpleNamespace(notice_delivery={})
        seen["ctx"] = turn_arg.channel_turn_context
        if admission_fails:
            raise StreamOverBudgetError("C1: more than fits in one request")

    async def _send(**kwargs):
        text = kwargs.get("text") or ""
        if "never finished" in text:
            order.append("notice")
        elif "I'll process your text/image/document request now" in text:
            order.append("failed_files")
            # A SlackApiError swallowed by send_message comes back as None, not an exception.
            return "posted.1" if notice_lands else None
        return "posted.1"

    p.thread_manager.acquire_thread_lock = AsyncMock(return_value=True)
    p.thread_manager.release_thread_lock = AsyncMock()
    # Patching methods onto a real processor is the point of this harness, hence the
    # [method-assign] ignores.
    p._get_or_rebuild_thread_state = _state                      # type: ignore[method-assign]
    p.get_or_create_channel_thread_state = _state                # type: ignore[method-assign]
    p._build_channel_turn_stream = _stream                       # type: ignore[method-assign]
    p._admit_channel_request = _admit                            # type: ignore[method-assign]
    p._handle_text_response = AsyncMock(return_value=Response(type="text", content="ok"))  # type: ignore[method-assign]
    p._build_channel_info = AsyncMock(return_value="")           # type: ignore[method-assign]
    p._process_attachments = AsyncMock(return_value=([], [], list(unsupported)))  # type: ignore[method-assign]

    client = MagicMock()
    client.send_message = AsyncMock(side_effect=_send)
    client.update_message = AsyncMock()
    ts = "10.0"
    meta: Dict[str, Any] = {"ts": ts}
    if gate_sources:
        meta["gate_sources"] = list(gate_sources)
    message = Message(text=trigger_text, user_id="U1", channel_id="C1",
                      thread_id=ts, metadata=meta)
    if open_top_level:
        # A top-level trigger in a channel that allows top-level replies: both destinations are
        # still legal and the model has not chosen. This is the shape r3-3 broke.
        turn = TurnRuntime(silence_capable=False, progress_enabled=False, reply_thread_id=ts,
                           destination_selected=False, destination_source="default")
    else:
        turn = TurnRuntime(silence_capable=False, progress_enabled=True, reply_thread_id=ts)

    response = await p.process_message(message, client, None, turn=turn)
    return order, response, seen


@pytest.mark.asyncio
async def test_the_recovery_notice_waits_for_the_stream_and_the_admission():
    """It used to post before either. A fail-closed condition then left "Picking up from here"
    standing in the thread as the turn's only visible words — a promise it had just broken."""
    order, _, _ = await _channel_turn_with_a_prior_timeout(admission_fails=False)
    assert order == ["stream", "admission", "notice"], order


@pytest.mark.asyncio
async def test_a_turn_that_cannot_see_the_room_never_promises_to_pick_up_from_here():
    order, response, _ = await _channel_turn_with_a_prior_timeout(admission_fails=True)
    assert "notice" not in order, "the notice stood alone on a turn that could not answer"
    assert response.type == "error"
    # The turn still says something honest — the over-budget notice, not the recovery promise.
    assert "larger than I can send in one go" in response.content


# ------------------- r3-3/r3-4: the notice settles the destination BEFORE the request is measured

@pytest.mark.asyncio
async def test_an_owed_notice_locks_the_destination_before_admission_measures_the_request():
    """[r3-3] The open top-level turn, which the thread-shaped tests could not reach.

    Deferring the notice past admission fixed one bug and opened another: admission pinned a tool
    tuple that still exposed `set_reply_destination` and a suffix that said nothing about where the
    reply was going, then the notice settled the destination, and the request actually SENT carried
    `reply_destination=thread` and refused that tool at runtime. Bytes admitted stopped matching
    bytes sent, and the prompt contradicted the lock. So the lock comes first.
    """
    order, _, seen = await _channel_turn_with_a_prior_timeout(
        admission_fails=False, open_top_level=True)
    assert order == ["stream", "admission", "notice"], order
    assert seen["destination_selected"] is True
    assert seen["destination_locked"] is True
    assert seen["reply_destination"] == "thread"


@pytest.mark.asyncio
async def test_an_unowed_turn_keeps_its_destination_choice_open_through_admission():
    """The counterpart: nothing owed, nothing settled. Pre-settling every top-level turn would
    quietly delete the model's destination choice."""
    order, _, seen = await _channel_turn_with_a_prior_timeout(
        admission_fails=False, had_timeout=False, open_top_level=True)
    assert order == ["stream", "admission"], order
    assert seen["destination_selected"] is False
    assert seen["destination_locked"] is False


@pytest.mark.asyncio
async def test_the_mixed_failed_file_notice_also_waits_for_admission():
    """[r3-4] "I'll process your text/image/document request now" is the same kind of promise as the
    recovery notice, and it was posted before the stream was even built."""
    order, _, seen = await _channel_turn_with_a_prior_timeout(
        admission_fails=False, had_timeout=False,
        unsupported=[{"name": "budget.numbers", "reason": "unsupported"}])
    assert order == ["stream", "admission", "failed_files"], order
    assert seen["destination_selected"] is True and seen["destination_locked"] is True


@pytest.mark.asyncio
async def test_a_failed_file_promise_never_stands_alone_on_an_over_budget_turn():
    order, response, _ = await _channel_turn_with_a_prior_timeout(
        admission_fails=True, had_timeout=False,
        unsupported=[{"name": "budget.numbers", "reason": "unsupported"}])
    assert "failed_files" not in order, "the promise outlived the turn that made it"
    assert response.type == "error"


@pytest.mark.asyncio
async def test_the_responder_is_told_which_attachments_failed():
    """[r3-4] The channel transcript is Slack, so the failure notice was deliberately kept out of
    ThreadState — and then nothing put it anywhere else. The stream says a file was attached, so
    silence about the failure reads to the model as "I have that file"."""
    _, _, seen = await _channel_turn_with_a_prior_timeout(
        admission_fails=False, had_timeout=False,
        unsupported=[{"name": "budget.numbers", "reason": "unsupported"},
                     {"name": "scan.tif", "reason": "unsupported"}])
    assert seen["failed_attachment_names"] == ("budget.numbers", "scan.tif")


# ------------------------------- r4-4: the evidence may not assert a delivery that did not happen

@pytest.mark.asyncio
async def test_a_landed_failed_file_notice_is_recorded_as_delivered():
    _, _, seen = await _channel_turn_with_a_prior_timeout(
        admission_fails=False, had_timeout=False,
        unsupported=[{"name": "budget.numbers", "reason": "unsupported"}])
    assert seen["ctx"].notice_delivery == {"failed_attachments": True}


@pytest.mark.asyncio
async def test_a_dropped_failed_file_notice_is_recorded_as_undelivered():
    """[r4-4] `send_message` swallows a SlackApiError and returns None. The evidence then told the
    responder "the user has been told they failed", so the failure went unmentioned by both of us —
    the bot answered around a file nobody had been told about."""
    _, _, seen = await _channel_turn_with_a_prior_timeout(
        admission_fails=False, had_timeout=False, notice_lands=False,
        unsupported=[{"name": "budget.numbers", "reason": "unsupported"}])
    assert seen["ctx"].notice_delivery == {"failed_attachments": False}


# ------------------- r4-5: the unsupported-only shortcut is about the TURN, not just the trigger

@pytest.mark.asyncio
async def test_a_failed_file_only_trigger_still_answers_the_cohort_behind_it():
    """[r4-5] A catch-up turn answers earlier messages too. The shortcut asked only whether the
    TRIGGER had anything left — so a trigger whose one attachment failed returned the failure notice
    and stopped, and every sender in the accepted cohort got silence."""
    order, response, _ = await _channel_turn_with_a_prior_timeout(
        admission_fails=False, had_timeout=False, trigger_text="",
        unsupported=[{"name": "budget.numbers", "reason": "unsupported"}],
        gate_sources=[SimpleNamespace(ts="9.0", text="what about Q3?", sender_name="Bob",
                                      sender_id="U2", attachments=())])
    assert order == ["stream", "admission", "failed_files"], order
    assert response.content == "ok", "the cohort was dropped with the trigger's failed file"


@pytest.mark.asyncio
async def test_a_failed_file_only_trigger_with_nothing_behind_it_still_shortcuts():
    """The counterpart: no cohort, nothing else to answer, so the notice IS the turn. Proceeding
    here would call the model with a request whose only content is a file it cannot read."""
    order, response, _ = await _channel_turn_with_a_prior_timeout(
        admission_fails=False, had_timeout=False, trigger_text="",
        unsupported=[{"name": "budget.numbers", "reason": "unsupported"}])
    assert "admission" not in order, order
    assert "budget.numbers" in response.content, response.content


# ------------------------------------------------- a FOREIGN post is not this exchange (§2c)

def _memory_join(bot, turn, message):
    """What `_schedule_channel_memory` would hand the extractor, or None when it declines."""
    from config import config

    scheduled = []
    bot.processor._schedule_async_call = MagicMock(side_effect=scheduled.append)
    bot.processor.extract_channel_memory_from_exchange = MagicMock(return_value=None)
    original = config.enable_memory_extraction_fallback
    config.enable_memory_extraction_fallback = True
    try:
        bot._schedule_channel_memory(message, turn)
    finally:
        config.enable_memory_extraction_fallback = original
    if not scheduled:
        return None
    return bot.processor.extract_channel_memory_from_exchange.call_args.args[2]


def _channel_message():
    return Message(text="what happened with the deploy?", user_id="U1", channel_id="C1",
                   thread_id="10.0", metadata={"ts": "10.0"})


def test_a_cross_thread_post_is_not_remembered_as_this_threads_exchange():
    """The pairing would be a fiction: the question was asked HERE and the answer went THERE, so
    "A asked X, we said Y" describes a conversation that happened in neither thread. The post is
    still observable in turn_outcome's destinations — this is about what gets persisted as memory."""
    from message_processor.turn_runtime import DEST_KIND_POST_TO_THREAD

    bot = _make_bot()
    turn = TurnRuntime()
    turn.stream_build_present = True
    turn.mark_destination_committed(first_ts="900.0", kind=DEST_KIND_POST_TO_THREAD,
                                    text="the answer, over in thread C", channel_id="C1",
                                    thread_root_ts="OTHER.9")

    assert _memory_join(bot, turn, _channel_message()) is None
    # …and the record itself is untouched: the ledger still reports the delivery.
    assert len(turn.committed_destinations) == 1


def test_a_reply_alongside_a_cross_thread_post_still_remembers_the_reply():
    """The filter is per RECORD, not per turn. A turn that answered here AND closed a loop
    elsewhere has a real exchange to remember — just not the foreign half of it."""
    from message_processor.turn_runtime import DEST_KIND_POST_TO_THREAD, DEST_KIND_REPLY

    bot = _make_bot()
    turn = TurnRuntime()
    turn.stream_build_present = True
    turn.mark_destination_committed(first_ts="11.0", kind=DEST_KIND_REPLY,
                                    text="answered here", channel_id="C1")
    turn.mark_destination_committed(first_ts="900.0", kind=DEST_KIND_POST_TO_THREAD,
                                    text="and closed the loop over there", channel_id="C1",
                                    thread_root_ts="OTHER.9")

    joined = _memory_join(bot, turn, _channel_message())
    assert joined == "answered here"
    assert "over there" not in (joined or "")
