"""Agent surfaces round, streaming side (Docs/specs/slack_agent_surfaces_round.md §A.3, §C, §E).

§A.3 the native Stop is cooperative: checkpoints, never a cancellation. §C step cards ride a
chunks-mode native stream. §E a buffered threaded reply posts as a one-shot native stream so the
settings button rides it. Every mock stream terminates and yields real strings (pitfall #6).
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from slack_sdk.errors import SlackApiError

from config import config
from message_processor import user_stop
from message_processor.stale_send_guard import ConversationWatermarks
from message_processor.tool_registry import ToolContext
from message_processor.turn_runtime import TurnRuntime
from openai_client.api import responses as R
from openai_client.api import tool_loop
from slack_client.messaging import NativeStreamSession
from streaming.native_sink import NativeStreamCoordinator, StepCards
from tests.unit.channel_turn_harness import pin_channel_turn, thread_config
from tests.unit.test_reconsideration import (CH, TRIGGER_TS, SiteOpenAI, SiteSlack,
                                             _msg, _site_processor, _site_thread_state)

FOOTER = [{"type": "actions", "elements": [{"type": "button", "action_id": "open_settings"}]}]


def _tasks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [c for c in chunks if c["type"] == "task_update"]


def _titles(chunks: List[Dict[str, Any]]) -> List[str]:
    return [c["title"] for c in chunks if c["type"] == "plan_update"]


@pytest.fixture(autouse=True)
def _clean_stop_registry():
    user_stop.clear_for_tests()
    yield
    user_stop.clear_for_tests()


def _press_stop(stream_ts: Optional[List[str]] = None) -> None:
    """The Stop event, as the event handler records it — pressed AFTER the turn was admitted."""
    user_stop.request_stop(CH, TRIGGER_TS, "U1", event_ts=str(time.time() + 0.5),
                           stream_ts=stream_ts)


def _api_error(code: str) -> SlackApiError:
    return SlackApiError(code, {"ok": False, "error": code})


# ------------------------------------------------------------------ Slack SDK double

class _SDK:
    """chat.startStream/appendStream/stopStream double recording every call's kwargs."""

    def __init__(self, start_exc: Optional[BaseException] = None,
                 append_exc: Optional[BaseException] = None,
                 stop_exc: Optional[BaseException] = None):
        self.calls: List[tuple] = []
        self.start_exc, self.append_exc, self.stop_exc = start_exc, append_exc, stop_exc
        self._n = 0

    async def chat_startStream(self, **kw):
        self.calls.append(("start", kw))
        if self.start_exc is not None:
            raise self.start_exc
        self._n += 1
        return {"ts": f"{self._n}00.0"}

    async def chat_appendStream(self, **kw):
        self.calls.append(("append", kw))
        if self.append_exc is not None:
            raise self.append_exc

    async def chat_stopStream(self, **kw):
        self.calls.append(("stop", kw))
        if self.stop_exc is not None:
            raise self.stop_exc


class _Client:
    def __init__(self, sdk: _SDK):
        self.sdk = sdk

    def begin_native_stream(self, channel_id, thread_id, user_id=None, chunks_mode=False):
        return NativeStreamSession(self.sdk, channel_id, thread_id, team_id="T1",
                                   user_id=user_id or "U1", chunks_mode=chunks_mode)


# ------------------------------------------------------------------ §C wire shapes

@pytest.mark.asyncio
async def test_chunks_mode_never_sends_markdown_text_or_an_empty_chunk():
    sdk = _SDK()
    steps = StepCards()
    steps.record("hosted:ws_1", "Searching the web", "in_progress")  # before the stream exists
    coord = NativeStreamCoordinator(_Client(sdk), CH, TRIGGER_TS, char_limit=3000,
                                    chunks_mode=True, steps=steps)
    assert await coord.start()
    steps.record("hosted:ws_1", "Searching the web", "complete",
                 sources=["https://www.example.com/a", "https://www.example.com/a"],
                 kind="web_search")
    assert (await coord.update("Here is the answer."))[0]
    assert await coord.finalize("Here is the answer.", suffix="", blocks=FOOTER)

    assert [k for k, _ in sdk.calls] == ["start", "append", "append", "stop"]
    for _, kw in sdk.calls:
        assert "markdown_text" not in kw
        for chunk in kw.get("chunks", []):
            assert chunk["type"] != "markdown_text" or chunk["text"]
    start = sdk.calls[0][1]
    assert start["task_display_mode"] == "plan"
    assert start["chunks"] == [
        {"type": "plan_update", "title": "Searching the web (0 of 1 done)"},
        {"type": "task_update", "id": "s1", "title": "Searching the web",
         "status": "in_progress"}]                                  # early card opens the stream
    append = sdk.calls[1][1]["chunks"]
    task = _tasks(append)[0]
    assert task["status"] == "complete"
    assert "sources" not in task                                    # never mid-search
    assert append[-1] == {"type": "markdown_text", "text": "Here is the answer."}
    final = _tasks(sdk.calls[2][1]["chunks"])[0]                    # once, at the turn's end
    assert final["sources"] == [{"type": "url", "text": "example.com › a",
                                 "url": "https://www.example.com/a"}]
    assert sdk.calls[3][1] == {"channel": CH, "ts": "100.0", "blocks": FOOTER}  # blocks-only stop


@pytest.mark.asyncio
async def test_step_cards_wait_while_a_mention_is_withheld():
    sdk = _SDK()
    steps = StepCards()
    coord = NativeStreamCoordinator(_Client(sdk), CH, TRIGGER_TS, char_limit=3000,
                                    chunks_mode=True, steps=steps)
    await coord.start()
    await coord.update("Asking <@B0")                               # "<@B0" is withheld
    steps.record("local:1", "Reading a thread", "in_progress")
    assert await coord.push_steps()                                 # not a safe boundary
    assert steps.has_pending
    await coord.update("Asking <@B07ABC> now")                      # releases the mention…
    assert [c["type"] for c in sdk.calls[-1][1]["chunks"]] == ["markdown_text"]
    assert await coord.push_steps()                                 # …then the card follows it
    assert [c["type"] for c in sdk.calls[-1][1]["chunks"]] == ["plan_update", "task_update"]


def test_step_cards_cap_roll_and_final_close():
    steps = StepCards()
    for i in range(41):
        steps.record(f"k{i}", f"step {i}", "in_progress")
    assert len(_tasks(steps.drain())) == 40                         # the 41st shows no card
    steps.record("k40", "step 40", "complete")
    assert not steps.has_pending                                    # ignored for good
    steps.record("k0", "step 0", "complete")
    reopened = steps.reopen_for_new_part()
    assert _titles(reopened) == ["step 39 (1 of 40 done)"]          # the title opens the part
    assert {c["id"] for c in _tasks(reopened)} == {f"s{i + 1}" for i in range(40)}
    closed = _tasks(steps.close_open(success=False))
    assert {c["id"] for c in closed if c["status"] == "error"} == {f"s{i + 1}" for i in range(1, 40)}


# ------------------------------------------------------------------ §A.3.3 stop errors

@pytest.mark.asyncio
async def test_not_in_streaming_state_after_a_stop_is_terminal_success(monkeypatch):
    monkeypatch.setattr(config, "slack_stop_button", True)
    sdk = _SDK(append_exc=_api_error("message_not_in_streaming_state"),
               stop_exc=_api_error("message_not_in_streaming_state"))
    coord = NativeStreamCoordinator(_Client(sdk), CH, TRIGGER_TS, char_limit=3000)
    await coord.start()
    _press_stop(stream_ts=["100.0"])
    ok, overflow = await coord.update("partial answer")
    assert ok and overflow is None and not coord.failed and coord.user_stopped
    assert await coord.finalize("partial answer and more")          # the prefix stands
    assert [k for k, _ in sdk.calls].count("stop") == 0             # nothing more said into it


@pytest.mark.asyncio
async def test_a_stop_event_arriving_after_slacks_refusal_still_counts(monkeypatch):
    import asyncio
    import slack_client.messaging as messaging

    monkeypatch.setattr(config, "slack_stop_button", True)
    monkeypatch.setattr(messaging, "_STOP_EVENT_GRACE_S", 2.0)
    sdk = _SDK(append_exc=_api_error("message_not_in_streaming_state"))
    coord = NativeStreamCoordinator(_Client(sdk), CH, TRIGGER_TS, char_limit=3000)
    await coord.start()
    asyncio.get_running_loop().call_later(0.2, _press_stop)        # the event lags the refusal
    ok, _ = await coord.update("partial answer")
    assert ok and coord.user_stopped and not coord.failed


@pytest.mark.asyncio
async def test_a_stop_while_closing_a_part_never_opens_the_next(monkeypatch):
    monkeypatch.setattr(config, "slack_stop_button", True)
    sdk = _SDK(stop_exc=_api_error("message_not_in_streaming_state"))
    coord = NativeStreamCoordinator(_Client(sdk), CH, TRIGGER_TS, char_limit=200)
    await coord.start()
    _press_stop(stream_ts=["100.0"])
    ok, overflow = await coord.update("word " * 60)                 # forces a roll
    assert ok and overflow is None and coord.user_stopped
    assert [k for k, _ in sdk.calls] == ["start", "append", "stop"]  # no second start


@pytest.mark.asyncio
async def test_not_in_streaming_state_without_a_stop_keeps_todays_failure(monkeypatch):
    import slack_client.messaging as messaging

    monkeypatch.setattr(config, "slack_stop_button", True)
    monkeypatch.setattr(messaging, "_STOP_EVENT_GRACE_S", 0.05)
    sdk = _SDK(append_exc=_api_error("message_not_in_streaming_state"))
    coord = NativeStreamCoordinator(_Client(sdk), CH, TRIGGER_TS, char_limit=3000)
    await coord.start()
    ok, _ = await coord.update("partial answer")
    assert not ok and coord.failed and not coord.user_stopped


# ------------------------------------------------------------------ §A.3.2 model side

class _Q:
    def log_info(self, *a, **k): pass
    log_debug = log_warning = log_error = log_info


def _openai_double(response: Any) -> MagicMock:
    fake = MagicMock()
    fake.log_info = fake.log_debug = fake.log_warning = fake.log_error = lambda *a, **k: None
    fake._safe_api_call = AsyncMock(return_value=response)
    return fake


@pytest.mark.asyncio
async def test_stop_check_closes_the_model_stream_and_skips_the_flush():
    response = SimpleNamespace(close=AsyncMock())
    fake = _openai_double(response)
    stopped = {"now": False}

    async def _events(resp, op):
        for e in (SimpleNamespace(type="response.output_text.delta", delta="first ", output_index=0),
                  SimpleNamespace(type="response.output_text.delta", delta="second", output_index=0),
                  SimpleNamespace(type="response.completed", response=SimpleNamespace(usage=None))):
            yield e
    fake._safe_stream_iteration = _events
    seen: List[Optional[str]] = []

    def _cb(chunk):
        seen.append(chunk)
        stopped["now"] = True                                       # Stop lands after 1 delta

    text = await R.create_streaming_response_with_tools(
        fake, messages=[{"role": "user", "content": "hi"}], tools=[{"type": "web_search"}],
        stream_callback=_cb, model="gpt-5.6-sol", stop_check=lambda: stopped["now"])
    assert text == "first "
    assert seen == ["first "]                                       # no second delta, no flush
    response.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_hosted_progress_names_the_call_and_its_own_sources():
    fake = _openai_double(SimpleNamespace())
    sources = [{"url": f"https://site{i}.test/p"} for i in range(12)] + [{"url": "https://site0.test/p"}]
    item = SimpleNamespace(type="web_search_call", id="ws_9", status="completed",
                           action=SimpleNamespace(type="search", query="menu trends",
                                                  queries=None, sources=sources))

    async def _events(resp, op):
        for e in (SimpleNamespace(type="response.output_item.added", output_index=2,
                                  item=SimpleNamespace(type="web_search_call", id="ws_9")),
                  SimpleNamespace(type="response.web_search_call.in_progress", output_index=2),
                  SimpleNamespace(type="response.output_item.done", item=item),
                  SimpleNamespace(type="response.completed", response=SimpleNamespace(usage=None))):
            yield e
    fake._safe_stream_iteration = _events
    progress: List[Dict[str, Any]] = []
    await R.create_streaming_response_with_tools(
        fake, messages=[], tools=[{"type": "web_search"}], stream_callback=lambda c: None,
        model="gpt-5.6-sol", progress_callback=progress.append)
    hosted = [p for p in progress if p["kind"] == "hosted"]
    assert [(p["state"], p["call_id"]) for p in hosted] == [("start", "ws_9"), ("complete", "ws_9")]
    assert hosted[1]["sources"] == [f"https://site{i}.test/p" for i in range(10)]


def _call(name="fetch_thread_messages", call_id="c1"):
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"}


@pytest.mark.asyncio
async def test_a_stop_during_dispatch_ends_the_loop_after_the_round(monkeypatch):
    from message_processor.tool_registry import ToolRegistry

    stopped = {"now": False}
    rounds = {"n": 0}

    async def _exec(ctx, args):
        stopped["now"] = True                                       # pressed while it runs
        return {"ok": True}

    registry = ToolRegistry()
    registry.register({"type": "function", "name": "fetch_thread_messages", "description": "t",
                       "parameters": {"type": "object"}}, _exec)

    async def fake_round(client, messages, tools, stream_callback, tool_callback=None,
                         function_call_sink=None, tool_choice=None, stop_check=None, **params):
        rounds["n"] += 1
        assert stop_check is not None                               # forwarded into the round
        function_call_sink.extend([_call()])
        return "Let me look. "

    monkeypatch.setattr(tool_loop.responses_api, "create_streaming_response_with_tools", fake_round)
    steps: List[Dict[str, Any]] = []
    out = await tool_loop.create_streaming_response_with_tool_loop(
        _Q(), messages=[], tools=[], registry=registry, tool_context=ToolContext(),
        stream_callback=lambda c: None, aggregate_segments=True,
        stop_check=lambda: stopped["now"], step_callback=steps.append)
    assert rounds["n"] == 1                                         # no second round
    assert out["user_stopped"] is True and out["text"] == "Let me look. "
    assert [(s["id"], s["state"]) for s in steps] == [("local:1", "start"), ("local:1", "complete")]


def test_a_chained_call_that_never_ran_shows_no_step():
    emitted: List[Dict[str, Any]] = []
    steps = tool_loop._StepEmitter(emitted.append)
    first = {"name": "edit_canvas", "arguments": '{"canvas_id": "F1"}'}
    second = {"name": "edit_canvas", "arguments": '{"canvas_id": "F1"}'}
    keys = steps.start_upfront([first, second])
    assert list(keys) == [id(first)]                                # only the chain head starts
    steps.settle(keys, first, {"ok": False, "error": "boom"}, False)
    steps.settle(keys, second, {"ok": False, "error": "not_run"}, False)
    assert [(e["id"], e["state"]) for e in emitted] == [("local:1", "start"), ("local:1", "failed")]


def test_a_run_of_the_same_step_is_one_card_updated_in_place():
    steps = StepCards()
    for i in range(9):
        steps.record(f"local:{i}", "Searching Slack", "in_progress", details=f"q{i}")
    first = _tasks(steps.drain())
    assert len(first) == 1 and first[0]["id"] == "s1"
    assert first[0]["title"] == "Searching Slack (9 searches)"
    assert first[0]["status"] == "in_progress"
    assert first[0]["details"].split("\n") == [f"q{i}" for i in range(9)]   # newest last
    for i in range(9):
        steps.record(f"local:{i}", "Searching Slack", "complete",
                     outcome={"count": 2, "unit": "matches"})
    done = steps.drain()
    assert [(c["id"], c["status"]) for c in _tasks(done)] == [("s1", "complete")]
    assert "details" not in _tasks(done)[0]                         # append-only: no resend
    final = _tasks(steps.close_open(success=True))[0]
    assert final["output"] == "18 matches"                          # merged runs sum


def test_a_different_step_breaks_the_run():
    steps = StepCards()
    steps.record("a", "Searching the web", "in_progress")
    steps.record("b", "Reading a thread", "in_progress")
    steps.record("c", "Searching the web", "in_progress")
    steps.record("d", "Using remember fact", "in_progress")
    steps.record("e", "Using remember fact", "in_progress")
    assert [(c["id"], c["title"]) for c in _tasks(steps.drain())] == [
        ("s1", "Searching the web"), ("s2", "Reading a thread"), ("s3", "Searching the web"),
        ("s4", "Using remember fact (2 times)")]


def test_a_merged_run_with_mixed_outcomes():
    steps = StepCards()
    steps.record("a", "Opening a page", "complete", details="https://a.test/x",
                 sources=["https://a.test/x"])
    steps.record("b", "Opening a page", "error", details="https://b.test/y",
                 sources=["https://a.test/x", "https://b.test/y"],
                 outcome={"text": "Couldn't access it"})
    card = _tasks(steps.drain())[0]
    assert card["title"] == "Opening a page (2 pages)" and card["status"] == "complete"
    assert card["details"] == "https://a.test/x\nhttps://b.test/y"
    assert "sources" not in card                                    # only at the end
    steps.record("a", "Opening a page", "error")
    assert _tasks(steps.drain())[0]["status"] == "error"            # only when ALL failed
    assert _tasks(steps.close_open(True))[0]["output"] == "Couldn't access it"
    steps2 = StepCards()
    for i in range(10):
        steps2.record(str(i), "Searching the web", "complete", details="x" * 300)
    block = _tasks(steps2.drain())[0]["details"]
    assert len(block) == 1000 and block.endswith("…")


@pytest.mark.asyncio
async def test_a_parallel_round_merges_and_survives_a_roll():
    sdk = _SDK()
    steps = StepCards()

    def _feed(ev: Dict[str, Any]) -> None:
        steps.record(ev["id"], "Searching Slack",
                     {"start": "in_progress", "complete": "complete", "failed": "error"}[ev["state"]])

    emitter = tool_loop._StepEmitter(_feed)
    calls = [{"name": "search_slack", "arguments": "{}"} for _ in range(3)]
    emitter.start_upfront(calls)                                     # one round, three calls
    coord = NativeStreamCoordinator(_Client(sdk), CH, TRIGGER_TS, char_limit=200,
                                    chunks_mode=True, steps=steps)
    await coord.start()
    start_chunks = sdk.calls[0][1]["chunks"]
    assert [c["title"] for c in _tasks(start_chunks)] == ["Searching Slack (3 searches)"]
    await coord.update("word " * 60)                                 # rolls to part 2
    starts = [kw for k, kw in sdk.calls if k == "start"]
    assert all(kw["task_display_mode"] == "plan" for kw in starts)  # every part is a plan
    reopened = starts[1]["chunks"]
    assert _titles(reopened) == ["Searching Slack (0 of 1 done)"]
    assert _tasks(reopened)[0] == {"type": "task_update", "id": "s1",
                                   "title": "Searching Slack (3 searches)",
                                   "status": "in_progress"}


def test_the_plan_title_follows_the_work_and_counts_it():
    steps = StepCards()
    steps.record("a", "Searching the web", "in_progress")
    assert _titles(steps.drain()) == ["Searching the web (0 of 1 done)"]
    steps.record("a", "Searching the web", "complete")
    steps.record("b", "Reading a thread", "in_progress")
    assert _titles(steps.drain()) == ["Reading a thread (1 of 2 done)"]
    steps.record("b", "Reading a thread", "complete")
    assert _titles(steps.close_open(True)) == ["Worked through 2 steps (2 of 2 done)"]
    one = StepCards()
    one.record("x", "Searching Slack", "in_progress")
    one.drain()
    one.record("x", "Searching Slack", "complete")
    assert _titles(one.close_open(True)) == ["Searching Slack (1 of 1 done)"]   # one label
    one.plan_title = "Finding the launch notes"                     # the model-written swap
    one.record("x", "Searching Slack", "complete")
    assert _titles(one.drain()) == ["Finding the launch notes (1 of 1 done)"]


def test_result_lines_per_tool():
    from message_processor.handlers.text import _hosted_step_outcome, _local_step_outcome

    def line(label, outcome, status="complete"):
        steps = StepCards()
        steps.record("k", label, status, outcome=outcome)
        return _tasks(steps.close_open(True))[0].get("output")

    web = _hosted_step_outcome("web_search", "complete", {
        "sources": ["https://www.fool.com/a", "https://cnbc.com/b", "https://stocktitan.net/c",
                    "https://fool.com/d"], "source_count": 7})
    assert line("Searching the web", web) == "7 results from fool.com, cnbc.com, stocktitan.net"
    assert _hosted_step_outcome("web_search", "complete", {"sources": []}) is None
    assert line("Running code", _hosted_step_outcome("code_interpreter", "complete", {})) == "Done"
    assert line("Running code", _hosted_step_outcome("code_interpreter", "failed", {}),
                "error") == "Failed"
    assert line("Searching Slack",
                _local_step_outcome("search_slack", {"ok": True, "count": 1}, True)) == "1 match"
    assert line("Reading a thread", _local_step_outcome(
        "fetch_thread_messages", {"ok": True, "count": 12}, True)) == "12 messages"
    assert line("Reading a thread", _local_step_outcome(
        "fetch_thread_messages", {"ok": False, "error": "not_in_channel"}, False),
        "error") == "Couldn't access it"
    assert line("Pinning a message",
                _local_step_outcome("pin_message", {"ok": True}, True)) is None


def test_search_sources_are_asked_for_only_with_step_cards():
    tools = [{"type": "web_search"}]
    assert R._request_includes([], None, tools) == ["reasoning.encrypted_content",
                                                    "web_search_call.results"]
    assert R._request_includes([], None, tools, search_sources=True) == [
        "reasoning.encrypted_content", "web_search_call.results",
        "web_search_call.action.sources"]
    assert R._request_includes(None, None, tools) is None           # byte-identical when off


def test_every_registered_local_tool_has_a_natural_step_label(monkeypatch):
    """Built from the REAL registry (every flag on), so a new tool without a label fails here
    instead of reaching Slack as "Using <tool name>"."""
    import inspect

    from message_processor.handlers.text import _LOCAL_STEP_LABELS
    from slack_client.base import SlackBot

    for flag in ("enable_tool_loop", "enable_reactions", "enable_react_tool",
                 "enable_post_to_thread_tool", "enable_search_tool", "enable_read_document_tool",
                 "enable_fetch_url_tool", "enable_link_fetch", "enable_image_import_tool",
                 "enable_people_tools", "enable_deep_research", "enable_image_tools",
                 "enable_history_tools"):
        monkeypatch.setattr(config, flag, True, raising=False)
    monkeypatch.setattr(config, "reaction_emojis", [], raising=False)
    host = MagicMock()
    for name in dir(SlackBot):
        if "schema" not in name and name != "get_history_tools_for_openai":
            continue
        raw = inspect.getattr_static(SlackBot, name)
        if isinstance(raw, (staticmethod, classmethod)):
            setattr(host, name, getattr(SlackBot, name))
        elif callable(raw):
            setattr(host, name, raw.__get__(host))
        else:
            setattr(host, name, raw)
    names = set(SlackBot._build_tool_registry(host)._tools) - {"set_reply_destination",
                                                                "update_plan"}
    assert len(names) > 40
    assert sorted(names - set(_LOCAL_STEP_LABELS)) == []


def test_pending_never_counts_as_done():
    steps = StepCards()
    steps.record("a", "Searching the web", "pending")
    steps.record("b", "Reading a thread", "in_progress")
    assert _titles(steps.drain()) == ["Reading a thread (0 of 2 done)"]
    model = StepCards()
    model.apply_plan("Checking the numbers", [
        {"id": "1", "title": "Find the report", "status": "in_progress", "result": None},
        {"id": "2", "title": "Compare quarters", "status": "pending", "result": None},
        {"id": "3", "title": "Write it up", "status": "pending", "result": None}])
    assert _titles(model.drain()) == ["Checking the numbers (0 of 3 done)"]


def _plan(title, *steps):
    return title, [{"id": i, "title": t, "status": st, "result": r} for i, t, st, r in steps]


def test_the_model_plan_takes_over_and_renders_its_own_steps():
    steps = StepCards()
    steps.record("hosted:ws_1", "Searching the web", "in_progress")
    steps.drain()                                                   # an automatic card SHOWN
    steps.record("local:1", "Searching Slack", "in_progress")       # recorded, never shown
    steps.apply_plan(*_plan("Finding the launch notes",
                            ("a", "Search the web", "in_progress", None),
                            ("b", "Write it up", "pending", None)))
    out = steps.drain()
    assert _titles(out) == ["Finding the launch notes (0 of 2 done)"]
    ids = [c["id"] for c in _tasks(out)]
    assert ids[-2:] == ["m:a", "m:b"] and "s2" not in ids          # unshown card dropped
    steps.record("hosted:ws_1", "Searching the web", "complete")    # shown card: terminal still
    steps.record("local:2", "Reading a thread", "in_progress")      # NEW automatic: suppressed
    assert [c["id"] for c in _tasks(steps.drain())] == ["s1"]
    # Output once, when terminal AND resulted; a later correction is ignored.
    steps.apply_plan(*_plan("Finding the launch notes",
                            ("a", "Search the web", "complete", None),
                            ("b", "Write it up", "pending", None)))
    first = _tasks(steps.drain())
    assert "output" not in first[0]
    steps.apply_plan(*_plan("Finding the launch notes",
                            ("a", "Search the web", "complete", "Found three posts"),
                            ("b", "Write it up", "in_progress", None)))
    second = steps.drain()
    assert _tasks(second)[0]["output"] == "Found three posts"
    assert _titles(second) == []                                    # count unchanged: no title
    steps.apply_plan(*_plan("Finding the launch notes",
                            ("a", "Search the web", "complete", "Found five posts")))
    assert "output" not in _tasks(steps.drain())[0]                 # locked once emitted
    # Rolls re-send the WHOLE model plan, outputs of finished steps included (R15).
    reopened = steps.reopen_for_new_part()
    by_id = {c["id"]: c for c in _tasks(reopened)}
    assert by_id["m:a"]["output"] == "Found three posts" and "m:b" in by_id
    assert _titles(reopened) == ["Finding the launch notes (1 of 2 done)"]


def test_a_title_only_change_is_emitted_on_its_own():
    steps = StepCards()
    steps.apply_plan(*_plan("First take", ("a", "Look", "in_progress", None)))
    steps.drain()
    steps.apply_plan(*_plan("Better title", ("a", "Look", "in_progress", None)))
    out = steps.drain()
    assert _titles(out) == ["Better title (0 of 1 done)"]


def test_same_round_updates_apply_in_order_and_the_last_wins():
    steps = StepCards()
    steps.apply_plan(*_plan("P", ("a", "Search", "complete", "two matches")))
    steps.apply_plan(*_plan("P", ("a", "Search", "error", "search failed")))
    task = _tasks(steps.drain())[0]
    assert task["status"] == "error" and task["output"] == "search failed"


@pytest.mark.parametrize("ending, expect", [
    ("delivered", [("complete", None), ("complete", None)]),
    ("handoff", [("complete", "Continuing in the background"),
                 ("complete", "Continuing in the background")]),
    ("stopped", [("error", None), ("pending", None)]),
    ("failed", [("error", None), ("error", None)]),
])
def test_model_plan_endings(ending, expect):
    steps = StepCards()
    steps.apply_plan(*_plan("P", ("a", "Run", "in_progress", None),
                            ("b", "Report", "pending", None)))
    steps.drain()
    final = {c["id"]: c for c in _tasks(steps.close(ending))}
    got = [(final[i]["status"] if i in final else "pending", final.get(i, {}).get("output"))
           for i in ("m:a", "m:b")]
    assert got == expect


@pytest.mark.asyncio
async def test_update_plan_validates_then_hands_the_attempt_its_snapshot():
    from message_processor.plan_tool import execute_update_plan, get_update_plan_schema

    schema = get_update_plan_schema()
    assert schema["strict"] is True
    item = schema["parameters"]["properties"]["steps"]["items"]
    assert item["required"] == ["id", "title", "status", "result"]
    assert item["properties"]["result"]["type"] == ["string", "null"]

    got: List[Dict[str, Any]] = []

    async def _sink(plan):
        got.append(plan)

    ctx = SimpleNamespace(turn=SimpleNamespace(plan_sink=_sink))
    bad = await execute_update_plan(ctx, {"title": "x", "steps": []})
    assert bad["ok"] is False and got == []                         # nothing visible
    ok = await execute_update_plan(ctx, {"title": "Checking", "steps": [
        {"id": "a", "title": "Look", "status": "in_progress", "result": None}]})
    assert ok == {"ok": True} and got[0]["steps"][0]["id"] == "a"
    no_surface = await execute_update_plan(SimpleNamespace(turn=None), {"title": "t", "steps": [
        {"id": "a", "title": "Look", "status": "pending", "result": None}]})
    assert no_surface == {"ok": True}                               # a display, never an error


def test_the_plan_contract_never_reaches_a_job():
    from message_processor.prompts import UPDATE_PLAN_CONTRACT_SUFFIX, strip_plan_contract

    items = [{"role": "user", "content": "hi"},
             {"role": "developer", "content": f"[time: now]\n\n{UPDATE_PLAN_CONTRACT_SUFFIX}"},
             {"role": "developer", "content": [{"type": "input_text",
                                                "text": UPDATE_PLAN_CONTRACT_SUFFIX}]}]
    out = strip_plan_contract(items)
    assert out[1]["content"] == "[time: now]"
    assert out[2]["content"][0]["text"] == ""
    assert out[0] is items[0]


def _full_registry(monkeypatch, cards_on: bool):
    import inspect

    from slack_client.base import SlackBot

    for flag in ("enable_tool_loop", "enable_reactions", "enable_react_tool",
                 "enable_post_to_thread_tool", "enable_search_tool", "enable_read_document_tool",
                 "enable_fetch_url_tool", "enable_link_fetch", "enable_image_import_tool",
                 "enable_people_tools", "enable_deep_research", "enable_image_tools",
                 "enable_history_tools"):
        monkeypatch.setattr(config, flag, True, raising=False)
    monkeypatch.setattr(config, "reaction_emojis", [], raising=False)
    monkeypatch.setattr(config, "slack_stream_task_cards", cards_on)
    host = MagicMock()
    for name in dir(SlackBot):
        if "schema" not in name and name != "get_history_tools_for_openai":
            continue
        raw = inspect.getattr_static(SlackBot, name)
        if isinstance(raw, (staticmethod, classmethod)):
            setattr(host, name, getattr(SlackBot, name))
        elif callable(raw):
            setattr(host, name, raw.__get__(host))
        else:
            setattr(host, name, raw)
    return SlackBot._build_tool_registry(host)


@pytest.mark.parametrize("surface", ["dm", "channel"])
def test_flag_off_tools_and_contract_are_byte_identical(monkeypatch, surface):
    """Fixed expectation, both surfaces: with the flag off the attempt's tool array is the
    flag-on array minus update_plan, in the same order, and its contract carries no plan text;
    with the flag on the tool is LAST and the paragraph rides the same join."""
    from message_processor.client_contract import Message
    from message_processor.handlers.text import TextHandlerMixin
    from message_processor.prompts import UPDATE_PLAN_CONTRACT_SUFFIX
    from message_processor.tool_registry import SURFACE_CHANNEL, SURFACE_DM

    surf = SURFACE_CHANNEL if surface == "channel" else SURFACE_DM
    channel = "C1" if surface == "channel" else "D1"
    msg = Message(text="hi", user_id="U1", channel_id=channel, thread_id="1.0",
                  metadata={"ts": "1.0", "sender_type": "human"})
    results = {}
    for on in (False, True):
        registry = _full_registry(monkeypatch, on)
        host = SimpleNamespace(_get_tool_registry=lambda c, cfg, surface=SURFACE_DM: registry)
        reg, cfg, _no_reply, suffix = TextHandlerMixin._materialize_request_tools(
            host, SimpleNamespace(), {}, msg, tools_disabled=False, turn=None, surface=surf)
        results[on] = ([sch["name"] for sch in reg.schemas(cfg, surface=surf)], suffix)
    off_names, off_suffix = results[False]
    on_names, on_suffix = results[True]
    assert "update_plan" not in off_names and on_names[-1] == "update_plan"
    assert off_names == on_names[:-1]
    from message_processor import prompts
    fixed_off = (prompts.CHANNEL_CROSS_THREAD_CONDUCT_SUFFIX if surface == "channel" else None)
    assert off_suffix == fixed_off                                  # the pre-§F bytes exactly
    assert on_suffix == "\n\n".join(p for p in (fixed_off, UPDATE_PLAN_CONTRACT_SUFFIX) if p)
    # A tool-disabled attempt never hears about a tool it lacks.
    registry = _full_registry(monkeypatch, True)
    host = SimpleNamespace(_get_tool_registry=lambda c, cfg, surface=SURFACE_DM: registry)
    assert TextHandlerMixin._materialize_request_tools(
        host, SimpleNamespace(), {}, msg, tools_disabled=True, turn=None, surface=surf)[3] is None


def test_bad_source_urls_are_dropped_when_recorded():
    steps = StepCards()
    steps.record("k", "Searching the web", "complete",
                 sources=["https://[bad", "ftp://example.com/x", "https://ok.test/a"],
                 details="x" * 300, kind="web_search")
    assert len(_tasks(steps.drain())[0]["details"]) == 200
    chunk = _tasks(steps.close("delivered"))[0]
    assert [s["url"] for s in chunk["sources"]] == ["https://ok.test/a"]


def test_cited_sources_carry_titles_dedupe_and_cap():
    steps = StepCards()
    hits = [f"https://www.abs.gov.au/statistics/p{i}" for i in range(8)]
    steps.record("ws1", "Searching the web", "complete", sources=hits[:4], kind="web_search")
    steps.record("rt", "Reading a thread", "complete")
    steps.record("ws2", "Searching the web", "complete", sources=hits[4:], kind="web_search")
    steps.drain()
    steps.add_citations([{"url": hits[0], "title": "Population clock"},
                         {"url": "https://abs.gov.au/statistics/p0/", "title": "Dup"},
                         {"url": hits[5], "title": "T" * 120}]
                        + [{"url": f"https://elsewhere.test/{i}", "title": f"E{i}"}
                           for i in range(7)])
    final = {c["id"]: c.get("sources") for c in _tasks(steps.close("delivered"))}
    assert final["s1"] == [{"type": "url", "text": "Population clock", "url": hits[0]}]
    assert final["s3"][0] == {"type": "url", "text": "T" * 79 + "…", "url": hits[5]}
    assert len(final["s3"]) == 5                                    # unmatched → last search, ≤5
    assert final.get("s2") is None                                  # no search, no sources


def test_uncited_fallback_prefers_distinct_domains_with_path_labels():
    steps = StepCards()
    urls = ["https://www.abs.gov.au/statistics/people/population",
            "https://www.abs.gov.au/statistics/people/population",
            "https://abs.gov.au/statistics/economy",
            "https://www.abs.gov.au/census", "https://ourworld.test/data",
            "https://abs.gov.au/a", "https://abs.gov.au/b", "https://news.test/x"]
    steps.record("ws", "Searching the web", "complete", sources=urls, kind="web_search")
    steps.drain()
    chunk = _tasks(steps.close("delivered"))[0]
    assert [s["text"] for s in chunk["sources"]] == [
        "abs.gov.au › statistics › people", "ourworld.test › data", "news.test › x",
        "abs.gov.au › statistics › economy", "abs.gov.au › census"]


def test_model_plan_steps_get_the_sources_of_their_searches():
    steps = StepCards()
    steps.apply_plan("Finding stats", [
        {"id": "a", "title": "Search", "status": "in_progress", "result": None},
        {"id": "b", "title": "Write", "status": "pending", "result": None}])
    steps.record("ws", "Searching the web", "complete", sources=["https://a.test/x"],
                 kind="web_search")                                 # ran during step a
    steps.drain()
    steps.add_citations([{"url": "https://a.test/x", "title": "A page"},
                         {"url": "https://nowhere.test/y", "title": "Unmatched"}])
    final = {c["id"]: c.get("sources") for c in _tasks(steps.close("delivered"))}
    assert final["m:a"] == [{"type": "url", "text": "A page", "url": "https://a.test/x"},
                            {"type": "url", "text": "Unmatched", "url": "https://nowhere.test/y"}]


@pytest.mark.asyncio
async def test_sources_ride_only_the_final_part(monkeypatch):
    sdk = _SDK()
    steps = StepCards()
    steps.record("ws", "Searching the web", "complete", sources=["https://a.test/x"],
                 kind="web_search")
    coord = NativeStreamCoordinator(_Client(sdk), CH, TRIGGER_TS, char_limit=200,
                                    chunks_mode=True, steps=steps)
    await coord.start()
    ok, overflow = await coord.update("word " * 60)                 # roll to part 2
    assert ok and overflow is not None
    assert await coord.finalize(overflow)
    with_sources = [(i, k) for i, (k, kw) in enumerate(sdk.calls)
                    for ch in kw.get("chunks", []) if ch.get("sources")]
    starts = [i for i, (k, _) in enumerate(sdk.calls) if k == "start"]
    assert len(with_sources) == 1 and with_sources[0][0] > starts[1]   # part 2's final append


# ------------------------------------------------------------------ handler rig

class _StoppingOpenAI(SiteOpenAI):
    """Streams its chunks and presses Stop just before chunk `stop_at`."""

    def __init__(self, final_text: str, chunks: List[str], stop_at: Optional[int]):
        super().__init__(final_text, chunks=chunks)
        self.stop_at = stop_at

    async def create_streaming_response(self, messages=None, stream_callback=None,
                                        tool_callback=None, hidden_suppression_sink=None, **kw):
        for i, chunk in enumerate(self.chunks):
            if i == self.stop_at:
                _press_stop()
            await stream_callback(chunk)
        await stream_callback(None)
        return self.final_text


class _NativeSlack(SiteSlack):
    def __init__(self, sdk: _SDK, **kw):
        super().__init__(**kw)
        self.sdk = sdk

    def supports_native_streaming(self):
        return True

    def begin_native_stream(self, channel, thread_ts, user_id=None, chunks_mode=False):
        return NativeStreamSession(self.sdk, channel, thread_ts, team_id="T1",
                                   user_id=user_id or "U1", chunks_mode=chunks_mode)

    async def _reconcile_uncertain_post(self, channel, thread, text, attempt_start):
        return self.reconciled

    reconciled: Optional[str] = None


class _Rig:
    def __init__(self, monkeypatch, openai, slack, *, buffered: bool):
        monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
        self.slack, self.openai = slack, openai
        self.processor = _site_processor(openai)
        self.message = _msg()
        self.lease = ConversationWatermarks().begin_turn(self.message)
        self.turn = TurnRuntime.for_message(self.message, channel_post_allowed=False)
        self.turn.send_lease = self.lease
        self.turn.silence_capable = buffered
        cfg = thread_config()
        self.turn.capability_profile = cfg
        pin_channel_turn(self.turn, trigger_ts=TRIGGER_TS, origin_thread_ts=TRIGGER_TS,
                         config=cfg, prepared=(None, {}, False, "", None))

    async def stream(self, thinking_id: Optional[str] = None):
        return await self.processor._handle_streaming_text_response(
            "hi", _site_thread_state(), self.slack, self.message, thinking_id, None,
            turn=self.turn)


@pytest.mark.asyncio
async def test_a_stop_before_the_turn_starts_ends_it_silent(monkeypatch):
    monkeypatch.setattr(config, "slack_stop_button", True)
    openai = SiteOpenAI("never", chunks=["never"])
    openai.create_streaming_response = AsyncMock()                  # type: ignore[method-assign]
    rig = _Rig(monkeypatch, openai, SiteSlack(), buffered=False)
    # Admitted at T-10 (main.py's stamp), Stop at T-5: BEFORE the handler ever ran.
    now = time.time()
    rig.message.metadata[user_stop.SINCE_KEY] = now - 10
    user_stop.request_stop(CH, TRIGGER_TS, "U1", event_ts=str(now - 5))
    resp = await rig.processor._handle_text_response(
        "hi", _site_thread_state(), rig.slack, rig.message, None, None, turn=rig.turn)
    assert resp.content == "" and resp.metadata["user_stopped"] is True
    assert resp.metadata["terminal_action"] == "no_reply" and resp.metadata["posted"] is False
    openai.create_streaming_response.assert_not_awaited()
    assert rig.slack.posts == []


@pytest.mark.asyncio
async def test_a_buffered_turn_stopped_mid_generation_posts_nothing(monkeypatch):
    monkeypatch.setattr(config, "slack_stop_button", True)
    rig = _Rig(monkeypatch, _StoppingOpenAI("Part one. Part two.", ["Part one. ", "Part two."], 1),
               SiteSlack(footer_blocks=FOOTER), buffered=True)
    resp = await rig.stream()
    assert rig.slack.posts == [] and rig.slack.updates == []
    assert resp.metadata["user_stopped"] is True and resp.metadata["posted"] is False


@pytest.mark.asyncio
async def test_a_stopped_native_turn_records_only_what_slack_accepted(monkeypatch):
    monkeypatch.setattr(config, "slack_stop_button", True)
    sdk = _SDK()
    rig = _Rig(monkeypatch, _StoppingOpenAI("Part one. Part two.", ["Part one. ", "Part two."], 1),
               _NativeSlack(sdk), buffered=False)
    resp = await rig.stream()
    sent = "".join(c.get("markdown_text") or "" for _, c in sdk.calls)
    assert "Part two." not in sent                                  # never revealed
    assert resp.content == "Part one. " and resp.metadata["user_stopped"] is True


class _CardFirstOpenAI(SiteOpenAI):
    """A hosted search runs BEFORE any prose: progress, then the tool event, then the text."""

    async def create_streaming_response_with_tools(self, messages=None, tools=None,
                                                   stream_callback=None, tool_callback=None,
                                                   progress_callback=None, **kw):
        if progress_callback is not None:                            # only with cards on
            progress_callback({"kind": "hosted", "state": "start", "tool": "web_search",
                               "call_id": "ws_1", "label": None, "item_id": "ws_1"})
        await tool_callback("web_search", "started")
        for chunk in self.chunks:
            await stream_callback(chunk)
        await stream_callback(None)
        return self.final_text


@pytest.mark.asyncio
async def test_the_first_step_opens_an_addressed_stream_card_only(monkeypatch):
    monkeypatch.setattr(config, "slack_stream_task_cards", True)
    sdk = _SDK()
    rig = _Rig(monkeypatch, _CardFirstOpenAI("Found it.", chunks=["Found it."]),
               _NativeSlack(sdk), buffered=False)
    rig.processor._build_tools_array = lambda *a, **k: [{"type": "web_search"}]
    await rig.stream()
    kind, start = sdk.calls[0]
    assert kind == "start" and start["task_display_mode"] == "plan"
    assert start["chunks"] == [
        {"type": "plan_update", "title": "Searching the web (0 of 1 done)"},
        {"type": "task_update", "id": "s1", "title": "Searching the web",
         "status": "in_progress"}]


@pytest.mark.asyncio
async def test_a_turn_with_cards_and_no_prose_stops_with_the_cards(monkeypatch):
    monkeypatch.setattr(config, "slack_stream_task_cards", True)
    sdk = _SDK()
    rig = _Rig(monkeypatch, _CardFirstOpenAI("", chunks=[]), _NativeSlack(sdk), buffered=False)
    rig.processor._build_tools_array = lambda *a, **k: [{"type": "web_search"}]
    await rig.stream()
    assert [k for k, _ in sdk.calls] == ["start", "append", "stop"]
    assert _tasks(sdk.calls[1][1]["chunks"])[0] == {"type": "task_update", "id": "s1",
                                                     "title": "Searching the web",
                                                     "status": "complete"}


@pytest.mark.asyncio
@pytest.mark.parametrize("cards_on", [True, False])
async def test_tools_used_line_drops_only_when_cards_show(monkeypatch, cards_on):
    monkeypatch.setattr(config, "slack_stream_task_cards", cards_on)
    monkeypatch.setattr(config, "enable_tool_provenance", True)
    sdk = _SDK()
    rig = _Rig(monkeypatch, _CardFirstOpenAI("Found it.", chunks=["Found it."]),
               _NativeSlack(sdk), buffered=False)
    rig.processor._build_tools_array = lambda *a, **k: [{"type": "web_search"}]
    resp = await rig.stream()
    sent = "".join(c.get("markdown_text") or "" for _, c in sdk.calls) + "".join(
        ch.get("text", "") for _, c in sdk.calls for ch in c.get("chunks", []))
    assert ("Tools Used" in resp.content) is (not cards_on)
    assert ("Tools Used" in sent) is (not cards_on)
    # The provenance other code reads still names the tool either way.
    persisted = rig.processor._persist_tool_provenance.call_args.args[3]
    assert any("web_search" in str(p) for p in persisted)


class _PlanningOpenAI(SiteOpenAI):
    """A web search with update_plan in the same response (the prompt's shape): the search
    runs, then the plan call is dispatched, then the answer streams."""

    def __init__(self, turn_ref: Dict[str, Any], answer: str = "Found it."):
        super().__init__(answer, chunks=[answer])
        self.turn_ref = turn_ref

    async def create_streaming_response_with_tool_loop(
            self, messages=None, tools=None, registry=None, tool_context=None,
            stream_callback=None, tool_callback=None, step_callback=None,
            progress_callback=None, **kw):
        if progress_callback is not None:
            progress_callback({"kind": "hosted", "state": "start", "tool": "web_search",
                               "call_id": "ws_1", "label": None, "item_id": "ws_1"})
        await tool_callback("web_search", "started")
        await tool_callback("local:update_plan", "started")
        sink = getattr(self.turn_ref["turn"], "plan_sink", None)
        if callable(sink):
            await sink({"title": "Finding the launch notes", "steps": [
                {"id": "a", "title": "Search the web", "status": "complete",
                 "result": "Found <@B07ABC> notes"},
                {"id": "b", "title": "Write it up", "status": "in_progress", "result": None}]})
        await tool_callback("local:update_plan", "completed")
        for chunk in self.chunks:
            await stream_callback(chunk)
        await stream_callback(None)
        return {"text": self.final_text, "segments": [self.final_text],
                "tools_used": ["update_plan"],
                "local_tool_calls": [{"name": "update_plan", "ok": True, "gist": "x"}]}


def _planning_rig(monkeypatch, sdk, *, buffered: bool):
    monkeypatch.setattr(config, "slack_stream_task_cards", True)
    monkeypatch.setattr(config, "enable_tool_provenance", True)
    turn_ref: Dict[str, Any] = {}
    rig = _Rig(monkeypatch, _PlanningOpenAI(turn_ref), _NativeSlack(sdk, footer_blocks=FOOTER),
               buffered=buffered)
    turn_ref["turn"] = rig.turn
    rig.processor._build_tools_array = lambda *a, **k: [{"type": "web_search"},
                                                         {"type": "function",
                                                          "name": "update_plan"}]
    rig.processor._build_tool_context = lambda *a, **k: SimpleNamespace(
        background_job_started=False, sandbox_image_assets=[], mounted_files=[],
        current_input=None, system_prompt=None, model=None)
    rig.processor._persist_destination_provenance = lambda *a, **k: None
    pin_channel_turn(rig.turn, trigger_ts=TRIGGER_TS, origin_thread_ts=TRIGGER_TS,
                     config=rig.turn.capability_profile,
                     prepared=(SimpleNamespace(), {}, False, "", None))
    return rig


@pytest.mark.asyncio
async def test_the_model_plan_shows_live_and_drops_the_tools_line(monkeypatch):
    sdk = _SDK()
    rig = _planning_rig(monkeypatch, sdk, buffered=False)
    resp = await rig.stream()
    chunks = [ch for _, kw in sdk.calls for ch in kw.get("chunks", [])]
    assert "Finding the launch notes (1 of 2 done)" in _titles(chunks)
    model_tasks = [c for c in _tasks(chunks) if c["id"].startswith("m:")]
    assert {c["id"] for c in model_tasks} == {"m:a", "m:b"}
    out = next(c["output"] for c in model_tasks if c.get("output"))
    assert "<@B" not in out                                         # sanitized like answer text
    assert "Finding the launch notes (2 of 2 done)" in _titles(chunks)   # delivered closes it
    assert "Tools Used" not in resp.content                         # owner: the plan replaces it
    persisted = rig.processor._persist_tool_provenance.call_args.args[3]
    assert not any("update_plan" in str(p) for p in persisted)      # never provenance


@pytest.mark.asyncio
async def test_a_buffered_reply_carries_the_final_model_plan(monkeypatch):
    monkeypatch.setattr(config, "slack_buffered_oneshot_stream", True)
    sdk = _SDK()
    rig = _planning_rig(monkeypatch, sdk, buffered=True)
    await rig.stream()
    assert [k for k, _ in sdk.calls] == ["start", "stop"]           # one message
    start = sdk.calls[0][1]
    assert start["task_display_mode"] == "plan" and "markdown_text" not in start
    assert _titles(start["chunks"]) == ["Finding the launch notes (2 of 2 done)"]
    assert start["chunks"][-1]["type"] == "markdown_text"
    assert sdk.calls[1][1]["blocks"] == FOOTER


class _JobDispatchOpenAI(SiteOpenAI):
    """One tool round that dispatches a background job — the job claiming its ack through the
    turn's seam exactly as `start_background_job` does — and the model's own ack text."""

    def __init__(self, turn_ref: Dict[str, Any]):
        super().__init__("Starting that now.", chunks=[])
        self.turn_ref = turn_ref
        self.job_posted_its_own_ack: Optional[bool] = None

    async def create_streaming_response_with_tool_loop(
            self, messages=None, tools=None, registry=None, tool_context=None,
            stream_callback=None, tool_callback=None, step_callback=None, **kw):
        call = {"kind": "local", "id": "local:1", "name": "start_background_job",
                "arguments": "{}"}
        # THE PRODUCTION ORDER (R18): the legacy "started" callback fires BEFORE the step is
        # recorded, with no preamble and no hosted event before it — so no stream is open yet
        # when the job claims its ack.
        await tool_callback("local:start_background_job", "started")
        step_callback({**call, "state": "start"})
        sink = getattr(self.turn_ref["turn"], "job_ack_sink", None)
        self.job_posted_its_own_ack = not (callable(sink) and sink())
        tool_context.background_job_started = True
        step_callback({**call, "state": "complete"})
        await tool_callback("local:start_background_job", "completed")
        return {"text": self.final_text, "segments": [self.final_text], "tools_used": [],
                "local_tool_calls": [{"name": "start_background_job", "ok": True}]}


@pytest.mark.asyncio
async def test_a_dispatched_job_acks_inside_the_one_card_stream(monkeypatch):
    from message_processor.research_tools import _JOB_ACK_TEXT

    monkeypatch.setattr(config, "slack_stream_task_cards", True)
    sdk = _SDK()
    slack = _NativeSlack(sdk, footer_blocks=FOOTER)
    turn_ref: Dict[str, Any] = {}
    openai = _JobDispatchOpenAI(turn_ref)
    rig = _Rig(monkeypatch, openai, slack, buffered=False)
    turn_ref["turn"] = rig.turn
    rig.processor._build_tools_array = lambda *a, **k: [{"type": "function",
                                                          "name": "start_background_job"}]
    rig.processor._build_tool_context = lambda *a, **k: SimpleNamespace(
        background_job_started=False, sandbox_image_assets=[], mounted_files=[],
        current_input=None, system_prompt=None, model=None)
    rig.processor._persist_destination_provenance = lambda *a, **k: None
    pin_channel_turn(rig.turn, trigger_ts=TRIGGER_TS, origin_thread_ts=TRIGGER_TS,
                     config=rig.turn.capability_profile,
                     prepared=(SimpleNamespace(), {}, False, "", None))
    resp = await rig.stream()

    assert openai.job_posted_its_own_ack is False                   # the turn owns the ack
    assert slack.send_calls == [] and slack.posts == []             # no second message
    kinds = [k for k, _ in sdk.calls]
    assert kinds.count("start") == 1 and kinds[-1] == "stop"        # ONE message
    assert _tasks(sdk.calls[0][1]["chunks"])[0]["title"] == "Starting a background job"
    appended = [ch for k, kw in sdk.calls if k == "append" for ch in kw["chunks"]]
    assert {"type": "markdown_text", "text": _JOB_ACK_TEXT} in appended
    assert sdk.calls[-1][1]["blocks"] == FOOTER                     # stopped there, with footer
    assert resp.metadata["background_job_started"] is True
    assert getattr(rig.turn, "job_ack_sink", None) is None          # the seam is per attempt


@pytest.mark.asyncio
async def test_a_stopped_legacy_partial_is_finalized_as_is(monkeypatch):
    monkeypatch.setattr(config, "slack_stop_button", True)
    rig = _Rig(monkeypatch, _StoppingOpenAI("Part one. Part two.", ["Part one. ", "Part two."], 1),
               SiteSlack(), buffered=False)
    resp = await rig.stream(thinking_id="ph-1")
    assert rig.slack.posts == []                                    # never a second answer
    assert rig.slack.updates[-1] == {"ts": "ph-1", "text": "Part one. "}   # no loader, no "(stopped)"
    assert resp.content == "Part one. " and resp.metadata["user_stopped"] is True


# ------------------------------------------------------------------ §E one-shot stream

@pytest.mark.asyncio
async def test_a_buffered_threaded_reply_rides_one_native_stream_with_the_footer(monkeypatch):
    monkeypatch.setattr(config, "slack_buffered_oneshot_stream", True)
    sdk = _SDK()
    slack = _NativeSlack(sdk, footer_blocks=FOOTER)
    rig = _Rig(monkeypatch, SiteOpenAI("The answer, **whole**.", chunks=["The answer, **whole**."]),
               slack, buffered=True)
    resp = await rig.stream()
    assert slack.send_calls == []
    assert [k for k, _ in sdk.calls] == ["start", "stop"]
    assert sdk.calls[0][1]["markdown_text"] == "The answer, **whole**."   # raw markdown
    assert sdk.calls[1][1]["blocks"] == FOOTER
    assert resp.metadata["footer_attached"] is True and resp.metadata["posted"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("start_exc, reconciled, expect_send", [
    (_api_error("invalid_arguments"), None, True),                 # rejected ⇒ send_message
    (TimeoutError("read timed out"), "777.0", False),              # ambiguous, found ⇒ delivered
    (TimeoutError("read timed out"), None, True),                  # ambiguous, not found
])
async def test_one_shot_start_outcomes(monkeypatch, start_exc, reconciled, expect_send):
    monkeypatch.setattr(config, "slack_buffered_oneshot_stream", True)
    slack = _NativeSlack(_SDK(start_exc=start_exc), footer_blocks=FOOTER)
    slack.reconciled = reconciled
    rig = _Rig(monkeypatch, SiteOpenAI("Short answer.", chunks=["Short answer."]), slack,
               buffered=True)
    resp = await rig.stream()
    assert bool(slack.send_calls) is expect_send
    assert resp.metadata["posted"] is True
    if not expect_send:
        # The recovered stream is closed WITH the footer, and it rode because Slack took it.
        assert [k for k, _ in slack.sdk.calls] == ["start", "stop"]
        assert slack.sdk.calls[1][1]["blocks"] == FOOTER
        assert resp.metadata["footer_attached"] is True


@pytest.mark.asyncio
async def test_the_one_shot_sends_the_whole_body_in_its_opening_call(monkeypatch):
    monkeypatch.setattr(config, "slack_buffered_oneshot_stream", True)
    sdk = _SDK()
    slack = _NativeSlack(sdk, footer_blocks=FOOTER)
    rig = _Rig(monkeypatch, SiteOpenAI("Ask @Example", chunks=["Ask @Example"]), slack,
               buffered=True)
    await rig.stream()
    assert sdk.calls[0][1]["markdown_text"].endswith("Example")    # nothing held for later
    assert "markdown_text" not in sdk.calls[1][1]


@pytest.mark.asyncio
async def test_a_rejected_footer_closes_the_stream_and_the_separate_footer_posts(monkeypatch):
    monkeypatch.setattr(config, "slack_buffered_oneshot_stream", True)

    class _BlocksRefused(_SDK):
        async def chat_stopStream(self, **kw):
            self.calls.append(("stop", kw))
            if "blocks" in kw:
                raise _api_error("invalid_blocks")

    sdk = _BlocksRefused()
    rig = _Rig(monkeypatch, SiteOpenAI("Short answer.", chunks=["Short answer."]),
               _NativeSlack(sdk, footer_blocks=FOOTER), buffered=True)
    resp = await rig.stream()
    assert [("blocks" in kw) for k, kw in sdk.calls if k == "stop"] == [True, False]
    assert resp.metadata["footer_attached"] is False


# ------------------------------------------------------------------ §C.6 history

def _carded_payload() -> Dict[str, Any]:
    """The probed final shape: task_card, rich_text, actions; top-level text concatenated."""
    return {
        "type": "message", "ts": "200.0", "user": "UBOT", "bot_id": "B0BOT",
        "text": "Searching the web The menu trend is up 4%.",
        "blocks": [
            {"type": "task_card", "task_id": "s1", "title": "Searching the web",
             "status": "complete"},
            {"type": "rich_text", "elements": [{"type": "rich_text_section", "elements": [
                {"type": "text", "text": "The menu trend is up 4%."}]}]},
            {"type": "actions", "elements": []},
        ],
    }


def test_a_carded_own_message_reads_as_its_answer_only():
    from message_processor.channel_stream import classify_chrome
    from slack_client.normalizer import normalize_slack_message

    client = SimpleNamespace(bot_user_id="UBOT", bot_id="B0BOT", self_team_id="T1",
                             is_own_message=lambda m: True)
    msg = normalize_slack_message(client, _carded_payload(), channel_id=CH)
    assert msg is not None and msg.sender_type == "self"
    assert msg.text == "The menu trend is up 4%."
    # edit_own_message only sees own messages the stream keeps as answers, never as chrome.
    assert classify_chrome([msg], chrome_markers=()) == frozenset()


def test_the_carded_reader_keeps_paragraphs_and_lists():
    from slack_client.normalizer import carded_answer_text

    payload = {"blocks": [
        {"type": "task_card", "task_id": "s1", "title": "Searching the web"},
        {"type": "rich_text", "elements": [
            {"type": "rich_text_section", "elements": [{"type": "text", "text": "first paragraph\n"}]},
            {"type": "rich_text_section", "elements": [{"type": "text", "text": "second paragraph"}]},
            {"type": "rich_text_list", "style": "bullet", "elements": [
                {"type": "rich_text_section", "elements": [{"type": "text", "text": "one"}]},
                {"type": "rich_text_section", "elements": [{"type": "text", "text": "two"}]}]},
        ]},
    ]}
    assert carded_answer_text(payload) == "first paragraph\nsecond paragraph\n• one\n• two"


# ------------------------------------------------------------------ footer rides the last part

class _FooterBot:
    """send_message's real code on a stub Slack client (format_text is identity)."""

    @staticmethod
    def make(post=None):
        from slack_client.formatting.text import SlackFormattingMixin
        from slack_client.messaging import SlackMessagingMixin
        from slack_client.utilities import SlackUtilitiesMixin

        class _Bot(SlackMessagingMixin, SlackFormattingMixin, SlackUtilitiesMixin):
            MAX_MESSAGE_LENGTH = 3900

            def __init__(self):
                self.app = MagicMock()
                self.markdown_converter = MagicMock()

            def log_info(self, *a, **k): pass
            log_debug = log_warning = log_error = log_info

            def format_text(self, text):
                return text

        bot = _Bot()
        calls: List[Dict[str, Any]] = []

        async def _post(**kw):
            calls.append(kw)
            if post is not None:
                post(kw)
            return {"ok": True, "ts": f"{len(calls)}.0"}
        bot.app.client.chat_postMessage = _post
        return bot, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("flag_on", [True, False])
async def test_a_multi_part_reply_carries_the_footer_on_its_last_part(monkeypatch, flag_on):
    monkeypatch.setattr(config, "slack_buffered_oneshot_stream", flag_on)
    bot, calls = _FooterBot.make()
    text = ("word " * 400 + "\n\n") * 4                               # ~8k chars → 3 parts
    meta: Dict[str, Any] = {}
    ts = await bot.send_message(CH, TRIGGER_TS, text, blocks=FOOTER, meta_out=meta)
    assert ts == "1.0" and len(calls) >= 2
    assert all("blocks" not in kw for kw in calls[:-1])             # earlier parts plain
    if flag_on:
        last = calls[-1]["blocks"]
        assert last[0]["type"] == "markdown" and last[1:] == FOOTER
        assert last[0]["text"].startswith("**") and "word" in last[0]["text"]
        assert meta["footer_attached"] is True                      # no separate footer
    else:
        assert "blocks" not in calls[-1] and meta["footer_attached"] is False


@pytest.mark.asyncio
async def test_a_long_single_reply_carries_the_footer_and_falls_back_when_refused(monkeypatch):
    monkeypatch.setattr(config, "slack_buffered_oneshot_stream", True)
    bot, calls = _FooterBot.make()
    meta: Dict[str, Any] = {}
    await bot.send_message(CH, TRIGGER_TS, "x" * 3000, blocks=FOOTER, meta_out=meta)
    assert calls[0]["blocks"][0] == {"type": "markdown", "text": "x" * 3000}
    assert meta["footer_attached"] is True

    def _refuse(kw):
        if "blocks" in kw:
            raise _api_error("invalid_blocks")
    bot2, calls2 = _FooterBot.make(post=_refuse)
    meta2: Dict[str, Any] = {}
    assert await bot2.send_message(CH, TRIGGER_TS, "x" * 3000, blocks=FOOTER, meta_out=meta2)
    assert "blocks" not in calls2[-1] and meta2["footer_attached"] is False
    # Top-level replies stay chrome-free: no thread, no markdown footer.
    bot3, calls3 = _FooterBot.make()
    await bot3.send_message(CH, None, "x" * 3000, blocks=FOOTER, meta_out={})
    assert "blocks" not in calls3[0]


@pytest.mark.asyncio
async def test_a_legacy_edited_reply_gets_the_footer_by_edit(monkeypatch):
    bot, _ = _FooterBot.make()
    bot.app.client.chat_update = AsyncMock(return_value={"ok": True})
    assert await bot.attach_footer_markdown(CH, "5.0", "**Final** answer", FOOTER)
    kw = bot.app.client.chat_update.await_args.kwargs
    assert kw["ts"] == "5.0" and kw["blocks"] == [{"type": "markdown",
                                                   "text": "**Final** answer"}] + FOOTER


class _Lease:
    def __init__(self):
        self.authorized = 0

    def authorize(self, surface):
        self.authorized += 1

    def commit(self):
        pass


def _http_error(code: str, status: int = 200) -> SlackApiError:
    return SlackApiError(code, SimpleNamespace(status_code=status, get=lambda k, d=None: (
        code if k == "error" else d), headers={}))


@pytest.mark.asyncio
async def test_parts_are_byte_identical_and_dm_keeps_its_old_strip(monkeypatch):
    text = ("word " * 400 + "\n\n") * 4
    texts = {}
    for flag in (False, True):
        monkeypatch.setattr(config, "slack_buffered_oneshot_stream", flag)
        bot, calls = _FooterBot.make()
        await bot.send_message(CH, TRIGGER_TS, text, blocks=FOOTER, meta_out={})
        texts[flag] = [kw["text"] for kw in calls]
    assert texts[True] == texts[False]                              # ruling 1
    monkeypatch.setattr(config, "slack_buffered_oneshot_stream", True)
    bot, calls = _FooterBot.make()
    meta: Dict[str, Any] = {}
    await bot.send_message("D1", TRIGGER_TS, text, blocks=FOOTER, meta_out=meta)
    assert all("blocks" not in kw for kw in calls) and meta["footer_attached"] is False  # 7


@pytest.mark.asyncio
async def test_a_refused_footer_retries_plain_under_a_fresh_guard_and_stop(monkeypatch):
    monkeypatch.setattr(config, "slack_buffered_oneshot_stream", True)

    def _refuse(kw):
        if "blocks" in kw:
            raise _api_error("invalid_blocks")
    bot, calls = _FooterBot.make(post=_refuse)
    lease = _Lease()
    await bot.send_message(CH, TRIGGER_TS, "x" * 3000, blocks=FOOTER, meta_out={}, lease=lease)
    assert lease.authorized == 2 and "blocks" not in calls[-1]      # ruling 2
    stopped = {"now": False}

    def _refuse_then_stop(kw):
        stopped["now"] = True                                       # Stop lands mid-send
        _refuse(kw)
    bot2, calls2 = _FooterBot.make(post=_refuse_then_stop)
    assert await bot2.send_message(CH, TRIGGER_TS, "x" * 3000, blocks=FOOTER, meta_out={},
                                   stop_check=lambda: stopped["now"]) is None
    assert len(calls2) == 1                                         # Stop: no second send


@pytest.mark.asyncio
async def test_refused_blocks_fall_back_plain_after_a_429_and_on_the_short_path(monkeypatch):
    monkeypatch.setattr(config, "slack_buffered_oneshot_stream", True)
    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    seen = {"n": 0}

    def _rate_then_refuse(kw):
        if "blocks" in kw:
            seen["n"] += 1
            raise (_http_error("ratelimited", 429) if seen["n"] == 1
                   else _api_error("invalid_blocks"))
    bot, calls = _FooterBot.make(post=_rate_then_refuse)
    meta: Dict[str, Any] = {}
    await bot.send_message(CH, TRIGGER_TS, ("word " * 400 + "\n\n") * 4, blocks=FOOTER,
                           meta_out=meta)
    assert "blocks" not in calls[-1] and not meta.get("split_truncated")   # ruling 6
    bot2, calls2 = _FooterBot.make(post=lambda kw: (_ for _ in ()).throw(
        _api_error("invalid_blocks")) if "blocks" in kw else None)
    assert await bot2.send_message(CH, TRIGGER_TS, "short", blocks=FOOTER, meta_out={})
    assert "blocks" not in calls2[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [_http_error("internal_error"),
                                   _http_error("service_unavailable", 503)])
async def test_an_ambiguous_footer_part_is_reconciled_never_resent(monkeypatch, error):
    monkeypatch.setattr(config, "slack_buffered_oneshot_stream", True)

    def _ambiguous(kw):
        if "blocks" in kw:
            raise error
    bot, calls = _FooterBot.make(post=_ambiguous)
    bot._reconcile_uncertain_post = AsyncMock(return_value="9.9")
    meta: Dict[str, Any] = {}
    await bot.send_message(CH, TRIGGER_TS, ("word " * 400 + "\n\n") * 4, blocks=FOOTER,
                           meta_out=meta)
    assert bot._reconcile_uncertain_post.await_count == 1           # rulings 4/5
    assert sum(1 for kw in calls if "blocks" in kw) == 1            # never resent
    assert meta["footer_attached"] is True                          # no standalone footer


@pytest.mark.asyncio
async def test_the_footer_edit_is_fenced_reconciled_and_honest(monkeypatch):
    import slack_client.messaging as messaging

    bot, _ = _FooterBot.make()
    bot.app.client.chat_update = AsyncMock(side_effect=TimeoutError("read timed out"))
    bot._reconcile_uncertain_update = AsyncMock(return_value=True)
    assert await bot.attach_footer_markdown(CH, "5.0", "Final", FOOTER) is True   # ruling 4
    bot.app.client.chat_update = AsyncMock(side_effect=_api_error("invalid_blocks"))
    assert await bot.attach_footer_markdown(CH, "5.0", "Final", FOOTER) is False
    monkeypatch.setattr(messaging, "_epoch_refused", lambda *a, **k: True)
    bot.app.client.chat_update = AsyncMock()
    assert await bot.attach_footer_markdown(CH, "5.0", "Final", FOOTER) is False  # ruling 9
    bot.app.client.chat_update.assert_not_awaited()
    # Ruling 3: an edit the backstop truncated reports it, so no footer edit shows more.
    bot.app.client.chat_update = AsyncMock(return_value={"ok": True})
    result = await bot.update_message_streaming(CH, "5.0", "x" * 5000)
    assert result["success"] and result["truncated"] is True


@pytest.mark.asyncio
async def test_a_refused_one_shot_footer_is_never_recorded_as_attached():
    class _Refusing(_SDK):
        async def chat_stopStream(self, **kw):
            self.calls.append(("stop", kw))
            raise (_api_error("invalid_blocks") if "blocks" in kw
                   else TimeoutError("read timed out"))

    session = NativeStreamSession(_Refusing(), CH, TRIGGER_TS, team_id="T1", user_id="U1")
    assert await session.start_whole("Short answer.")
    await session.finish(blocks=FOOTER, close_on_block_reject=True)
    assert session.stop_outcome == "blocks_refused"                 # ruling 8: not "ambiguous"
