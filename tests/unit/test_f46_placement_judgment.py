"""F46 — the threading JUDGMENT call: a top-level channel reply threads when the turn is
thread-worthy (real tool work, or a deliberately-requested long-form deliverable) and stays
top-level for a quick answer. Two independently-testable parts:

  PART A  did_substantive_work force-to-thread override (DEFAULT ON) — a turn that ran a hosted
          tool / MCP call / slow local deliverable forces its top-level channel reply into a
          thread at final-post time. Driven by TurnRuntime.mark_substantive_work() +
          resolve_reply_target(), the late flip in text.py, and main.py's post-return rebind.

  PART B  the model placement decision (DEFAULT OFF, behind enable_mention_placement_model) — a
          lean utility call decides thread vs channel for a MENTION (which runs no participation
          gate and so carries no placement verdict), catching no-tool long-form the override can't.

Real decision code, stubbed I/O — no network/DB. The end-to-end streaming flip reuses the F39
FakeSlack harness (test_reply_surface); the main.py-level tests reuse test_mention_placement's.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import config
from message_processor.turn_runtime import TurnRuntime
from openai_client.api.responses import classify_placement


# ============================================================ Part A: TurnRuntime state

def _msg(metadata=None, thread_id="10.0"):
    return SimpleNamespace(metadata=metadata if metadata is not None else {}, thread_id=thread_id)


def test_did_substantive_work_defaults_false():
    turn = TurnRuntime()
    assert turn.did_substantive_work is False


def test_mark_substantive_work_sets_flag_even_with_ack_disabled(monkeypatch):
    # The whole point of a SEPARATE flag: claim_work()/the 👀 early-returns when the ack reaction
    # is off, so the work signal must be recorded independently or the override never fires.
    monkeypatch.setattr(config, "enable_ack_reaction", False, raising=False)
    turn = TurnRuntime()
    turn.mark_substantive_work()
    assert turn.did_substantive_work is True


# ------------------------------------------------------------ resolve_reply_target

def test_resolve_flips_top_level_reply_that_did_work():
    # final_post_only top-level reply (reply_thread_id None) + work done → threads under the
    # trigger AND flips place_in_channel False so attribution/footer render as a threaded reply.
    turn = TurnRuntime(final_post_only=True, reply_thread_id=None, did_substantive_work=True)
    message = _msg({"place_in_channel": True})
    assert turn.resolve_reply_target(message) == "10.0"
    assert message.metadata["place_in_channel"] is False


def test_resolve_leaves_top_level_reply_that_did_no_work():
    # No substantive work → stays top-level: returns None, never touches place_in_channel.
    turn = TurnRuntime(final_post_only=True, reply_thread_id=None, did_substantive_work=False)
    message = _msg({"place_in_channel": True})
    assert turn.resolve_reply_target(message) is None
    assert message.metadata["place_in_channel"] is True


def test_resolve_never_moves_an_in_thread_reply_outward():
    # A reply already targeted at a thread is returned unchanged, work or not — the override only
    # ever moves channel → thread, never the reverse.
    turn = TurnRuntime(final_post_only=False, reply_thread_id="10.0", did_substantive_work=True)
    message = _msg({"place_in_channel": False})
    assert turn.resolve_reply_target(message) == "10.0"
    assert message.metadata["place_in_channel"] is False  # untouched


def test_resolve_is_idempotent():
    # Called at attribution-build AND again at the send — the second call must not thrash.
    turn = TurnRuntime(final_post_only=True, reply_thread_id=None, did_substantive_work=True)
    message = _msg({"place_in_channel": True})
    first = turn.resolve_reply_target(message)
    second = turn.resolve_reply_target(message)
    assert first == second == "10.0"
    assert message.metadata["place_in_channel"] is False


def test_resolve_fails_open_on_a_bad_message():
    # message.metadata is not a dict → the flip is skipped, but the target still resolves and
    # nothing raises (fail-open: a placement error must never break or drop a reply).
    turn = TurnRuntime(final_post_only=True, reply_thread_id=None, did_substantive_work=True)
    message = SimpleNamespace(metadata=None, thread_id="10.0")
    assert turn.resolve_reply_target(message) == "10.0"


# ------------------------------------------------- non-streaming hosted-tool work (FIX 2)

def test_hosted_or_mcp_used_classification():
    # FIX 2: hosted (web_search/file_search/image_generation), MCP, and code_interpreter count as
    # thread-worthy work; an empty list does not. (Local tools are stripped by the caller BEFORE
    # this runs, so an all-local turn arrives here as [] — see the seam tests below.)
    from message_processor.handlers.text import _hosted_or_mcp_used
    assert _hosted_or_mcp_used(["web_search"]) is True
    assert _hosted_or_mcp_used(["mcp:datassential"]) is True
    assert _hosted_or_mcp_used(["image_generation"]) is True
    assert _hosted_or_mcp_used(["code_interpreter"]) is True     # hidden from attribution, still work
    assert _hosted_or_mcp_used([]) is False
    assert _hosted_or_mcp_used(None) is False


def _nonstreaming_seam(tools_used, local_tool_calls):
    """Replay the non-streaming final-post seam (text.py): strip local tools, mark substantive
    work for any hosted/MCP/code tool, then resolve placement for a top-level channel reply."""
    from message_processor.handlers.text import _hosted_or_mcp_used
    local_names = {c.get("name") for c in local_tool_calls if c.get("name")}
    external = [t for t in tools_used if t not in local_names]
    turn = TurnRuntime(final_post_only=True, reply_thread_id=None)
    message = _msg({"place_in_channel": True})
    if _hosted_or_mcp_used(external):
        turn.mark_substantive_work()
    target = turn.resolve_reply_target(message)
    return turn, target, message


def test_nonstreaming_hosted_tool_threads_top_level_reply():
    # A non-streaming turn that ran web_search → substantive work → its top-level reply threads.
    turn, target, message = _nonstreaming_seam(["web_search"], local_tool_calls=[])
    assert turn.did_substantive_work is True
    assert target == "10.0"
    assert message.metadata["place_in_channel"] is False


def test_nonstreaming_local_only_stays_top_level():
    # A non-streaming turn that used only a fast LOCAL tool (fetch_channel_history) → the local
    # name is stripped before the check, nothing marks work, and the reply stays top-level.
    turn, target, message = _nonstreaming_seam(
        ["fetch_channel_history"],
        local_tool_calls=[{"name": "fetch_channel_history", "ok": True}])
    assert turn.did_substantive_work is False
    assert target is None
    assert message.metadata["place_in_channel"] is True


# ============================================================ Part B: classify_placement

class _FakeContent:
    def __init__(self, text):
        self.text = text


class _FakeItem:
    def __init__(self, text):
        self.content = [_FakeContent(text)]


class _FakeResp:
    def __init__(self, text):
        self.output = [_FakeItem(text)]


class _FakeLLM:
    """Stands in for the OpenAIClient `self` classify_placement is bound to."""

    def __init__(self, text=None, exc=None):
        self._text = text
        self._exc = exc
        self.client = MagicMock()
        self.captured_input = None

    async def _safe_api_call(self, *a, **k):
        if self._exc:
            raise self._exc
        self.captured_input = k.get("input")
        return _FakeResp(self._text)

    def log_debug(self, *a, **k):
        pass

    def log_warning(self, *a, **k):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,expected", [
    ('{"placement": "thread", "reason": "long-form deliverable"}', "thread"),
    ('{"placement": "channel", "reason": "quick answer"}', "channel"),
    # code fences / surrounding prose tolerated
    ('```json\n{"placement": "thread"}\n```', "thread"),
    ('Sure: {"placement": "channel"} hope that helps', "channel"),
])
async def test_classify_placement_parses_verdict(raw, expected):
    llm = _FakeLLM(text=raw)
    assert await classify_placement(llm, "write me a three-paragraph story") == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["", "banana", "thread", "{not json}", "[]",
                                 '{"placement": "sideways"}'])
async def test_classify_placement_garbage_fails_open_to_channel(raw):
    # Fail OPEN to today's top-level behavior: anything unparseable / unexpected → "channel".
    llm = _FakeLLM(text=raw)
    assert await classify_placement(llm, "anything") == "channel"


@pytest.mark.asyncio
async def test_classify_placement_api_error_fails_open_to_channel():
    llm = _FakeLLM(exc=RuntimeError("utility model down"))
    assert await classify_placement(llm, "anything") == "channel"


@pytest.mark.asyncio
async def test_classify_placement_renders_optional_activity_but_stays_minimal():
    llm = _FakeLLM(text='{"placement": "channel"}')
    await classify_placement(llm, "hi", signals={"channel_activity": "[Recent]\n- Peter: hi"})
    prompt = llm.captured_input[1]["content"]
    assert "Latest message:\nhi" in prompt
    assert "[Recent]" in prompt
    # Absent the signal, no activity block is fabricated.
    llm2 = _FakeLLM(text='{"placement": "channel"}')
    await classify_placement(llm2, "hi")
    assert "Recent channel activity" not in llm2.captured_input[1]["content"]


# ============================================================ main.py: Part B seam + rebind

# Reuse the exact handle_message harness the mention-placement suite already validates.
from tests.unit.test_mention_placement import _place_app, _mention_msg  # noqa: E402


@pytest.mark.asyncio
async def test_mention_placement_flag_off_never_calls_and_stays_top_level(monkeypatch):
    monkeypatch.setattr(config, "enable_mention_placement_model", False, raising=False)
    app, client, _ = _place_app()
    app.processor.openai_client.classify_placement = AsyncMock(return_value="thread")
    await app.handle_message(_mention_msg({"reply_in_channel": True}), client)
    app.processor.openai_client.classify_placement.assert_not_awaited()
    assert client.send_message.await_args.args[1] is None  # top-level (post_thread_id None)


@pytest.mark.asyncio
async def test_mention_placement_flag_on_thread_verdict_threads(monkeypatch):
    monkeypatch.setattr(config, "enable_mention_placement_model", True, raising=False)
    app, client, _ = _place_app()
    app.processor.openai_client.classify_placement = AsyncMock(return_value="thread")
    await app.handle_message(_mention_msg({"reply_in_channel": True}), client)
    app.processor.openai_client.classify_placement.assert_awaited_once()
    assert client.send_message.await_args.args[1] == "10.0"  # threaded (verdict wins)


@pytest.mark.asyncio
async def test_mention_placement_flag_on_channel_verdict_stays_top_level(monkeypatch):
    monkeypatch.setattr(config, "enable_mention_placement_model", True, raising=False)
    app, client, _ = _place_app()
    app.processor.openai_client.classify_placement = AsyncMock(return_value="channel")
    await app.handle_message(_mention_msg({"reply_in_channel": True}), client)
    assert client.send_message.await_args.args[1] is None  # top-level


@pytest.mark.asyncio
async def test_mention_placement_flag_on_skips_in_thread_trigger(monkeypatch):
    # An in-thread trigger (ts != thread_id) is never a top-level trigger — the model call is
    # skipped entirely and the reply threads as before.
    monkeypatch.setattr(config, "enable_mention_placement_model", True, raising=False)
    app, client, _ = _place_app()
    app.processor.openai_client.classify_placement = AsyncMock(return_value="thread")
    await app.handle_message(
        _mention_msg({"ts": "11.0", "reply_in_channel": True}, thread_id="10.0"), client)
    app.processor.openai_client.classify_placement.assert_not_awaited()
    assert client.send_message.await_args.args[1] == "10.0"


@pytest.mark.asyncio
async def test_mention_placement_flag_on_skips_dm(monkeypatch):
    monkeypatch.setattr(config, "enable_mention_placement_model", True, raising=False)
    app, client, _ = _place_app()
    app.processor.openai_client.classify_placement = AsyncMock(return_value="thread")
    await app.handle_message(
        _mention_msg({"reply_in_channel": True}, channel_id="D123"), client)
    app.processor.openai_client.classify_placement.assert_not_awaited()
    assert client.send_message.await_args.args[1] == "10.0"  # DMs never move top-level


@pytest.mark.asyncio
async def test_main_rebinds_post_thread_id_after_a_late_flip(monkeypatch):
    # Part A wiring at the main.py seam: the handler flipped a top-level channel reply into a
    # thread (resolve_reply_target mutates message.metadata, NOT main.py's locals). The
    # post-return rebind must pick that up so the fallback send targets the thread.
    monkeypatch.setattr(config, "enable_mention_placement_model", False, raising=False)
    app, client, resp = _place_app()

    async def _flip(message, *a, **k):
        # Simulate text.py's late resolve on a substantive-work top-level turn.
        if isinstance(message.metadata, dict):
            message.metadata["place_in_channel"] = False
        return resp

    app.processor.process_message = _flip
    await app.handle_message(_mention_msg({"reply_in_channel": True}), client)
    assert client.send_message.await_args.args[1] == "10.0"  # rebound to the thread


@pytest.mark.asyncio
async def test_main_leaves_top_level_when_no_flip_happens(monkeypatch):
    # The rebind is fail-open: an unflipped top-level reply (place_in_channel stays True as
    # main.py stamped it) still posts top-level.
    monkeypatch.setattr(config, "enable_mention_placement_model", False, raising=False)
    app, client, _ = _place_app()
    await app.handle_message(_mention_msg({"reply_in_channel": True}), client)
    assert client.send_message.await_args.args[1] is None


# ============================================================ end-to-end: streaming flip

# The F39 harness drives the REAL streaming handler against a fake Slack.
from tests.unit.test_reply_surface import (  # noqa: E402
    FakeSlack, FakeOpenAI, _message, _thread_state, _processor, _run)


@pytest.mark.asyncio
async def test_streaming_top_level_reply_threads_after_substantive_work(monkeypatch):
    """A top-level channel reply (final_post_only) whose turn did substantive work is posted into
    the thread under the trigger, not top-level — and place_in_channel is flipped so its
    attribution/footer render threaded. Proves A3's resolve-before-attribution + effective_target
    send end to end through the real handler."""
    monkeypatch.setattr(config, "enable_no_reply_tool", True, raising=False)
    slack = FakeSlack(native=True)
    targets = []
    _orig_send = slack.send_message

    async def _capture(channel, thread, text, **k):
        targets.append(thread)
        return await _orig_send(channel, thread, text, **k)

    slack.send_message = _capture
    processor = _processor(FakeOpenAI(["Here is the answer."]))
    msg, ts = _message(), _thread_state()          # channel C1, trigger + thread 10.0
    msg.metadata["place_in_channel"] = True         # main.py stamped a top-level reply
    turn = TurnRuntime.for_message(msg, None)       # top-level → final_post_only
    assert turn.final_post_only
    turn.mark_substantive_work()                    # a hosted/local deliverable ran this turn

    resp = await _run(processor, slack, msg, ts, turn)

    assert resp.metadata["posted"] is True
    assert targets == ["10.0"], f"the reply should have threaded under the trigger, not {targets}"
    assert msg.metadata["place_in_channel"] is False, "the flip must render attribution threaded"
    assert slack.edits == [], "still posted once, never edited into existence"


@pytest.mark.asyncio
async def test_streaming_top_level_reply_stays_top_level_without_work(monkeypatch):
    """Contrast: no substantive work → the same top-level turn stays top-level (target None) and
    place_in_channel is untouched. The override only ever moves channel → thread."""
    monkeypatch.setattr(config, "enable_no_reply_tool", True, raising=False)
    slack = FakeSlack(native=True)
    targets = []
    _orig_send = slack.send_message

    async def _capture(channel, thread, text, **k):
        targets.append(thread)
        return await _orig_send(channel, thread, text, **k)

    slack.send_message = _capture
    processor = _processor(FakeOpenAI(["Quick answer."]))
    msg, ts = _message(), _thread_state()
    msg.metadata["place_in_channel"] = True
    turn = TurnRuntime.for_message(msg, None)       # top-level, but no work marked

    resp = await _run(processor, slack, msg, ts, turn)

    assert resp.metadata["posted"] is True
    assert targets == [None], f"a quick answer should stay top-level, not {targets}"
    assert msg.metadata["place_in_channel"] is True
