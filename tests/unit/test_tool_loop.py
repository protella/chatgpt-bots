"""Redesign Phases A + D — local function-call loop, ToolRegistry, react tool, footer shape.

All stubbed I/O: no live Slack, no live OpenAI, no legacy suite.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import config
from tool_registry import ToolContext, ToolRegistry, serialize_tool_result
from openai_client.api import tool_loop
from openai_client.api import responses as responses_api
from slack_client.messaging import SlackMessagingMixin
from message_processor.handlers.text import TextHandlerMixin


# --------------------------------------------------------------------------- helpers

class _Client:
    """Minimal OpenAIClient stand-in for the loop functions (they only use log_*)."""
    def log_info(self, *a, **k): pass
    def log_debug(self, *a, **k): pass
    def log_warning(self, *a, **k): pass
    def log_error(self, *a, **k): pass


def _registry_with(name="echo", executor=None, schema_extra=None):
    reg = ToolRegistry()
    schema = {"type": "function", "name": name, "description": "t", "parameters": {"type": "object"}}
    if schema_extra:
        schema.update(schema_extra)
    reg.register(schema, executor or (lambda ctx, args: _ok(args)))
    return reg


async def _ok(args):
    return {"ok": True, "echo": args}


def _call(name="echo", call_id="c1", arguments='{"x": 1}'):
    return {"call_id": call_id, "name": name, "arguments": arguments}


class _FakeRounds:
    """Scripted create_text_response_with_tools replacement: one entry per round."""
    def __init__(self, rounds):
        self.rounds = rounds  # list of (text, [calls])
        self.invocations = []

    async def __call__(self, client, messages, tools, return_metadata=True,
                       function_call_sink=None, tool_choice=None, **params):
        self.invocations.append({
            "messages": list(messages),
            "tool_choice": tool_choice,
        })
        idx = min(len(self.invocations) - 1, len(self.rounds) - 1)
        text, calls = self.rounds[idx]
        if tool_choice != "none" and function_call_sink is not None:
            function_call_sink.extend(calls)
        return {"text": text, "tools_used": []}


# --------------------------------------------------------------------------- registry

class TestToolRegistry:
    @pytest.mark.asyncio
    async def test_dispatch_executes_registered_tool(self):
        async def exec_(ctx, args):
            return {"ok": True, "got": args["q"], "channel": ctx.channel_id}
        reg = _registry_with(executor=exec_)
        out = await reg.dispatch(ToolContext(channel_id="C1"), "echo", '{"q": "hi"}')
        assert out == {"ok": True, "got": "hi", "channel": "C1"}

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        reg = _registry_with()
        out = await reg.dispatch(ToolContext(), "nope", "{}")
        assert out["ok"] is False and out["error"] == "unknown_tool"

    @pytest.mark.asyncio
    async def test_malformed_arguments(self):
        reg = _registry_with()
        out = await reg.dispatch(ToolContext(), "echo", "{not json")
        assert out["ok"] is False and out["error"] == "bad_arguments"

    @pytest.mark.asyncio
    async def test_executor_exception_is_wrapped(self):
        async def boom(ctx, args):
            raise RuntimeError("kaput")
        reg = _registry_with(executor=boom)
        out = await reg.dispatch(ToolContext(), "echo", "{}")
        assert out["ok"] is False and out["error"] == "execution_error" and "kaput" in out["message"]

    @pytest.mark.asyncio
    async def test_executor_timeout(self, monkeypatch):
        monkeypatch.setattr(config, "tool_call_timeout", 0.05)
        async def slow(ctx, args):
            await asyncio.sleep(1)
        reg = _registry_with(executor=slow)
        out = await reg.dispatch(ToolContext(), "echo", "{}")
        assert out["ok"] is False and out["error"] == "timeout"

    def test_enabled_gate_hides_schema(self):
        reg = ToolRegistry()
        reg.register({"type": "function", "name": "gated", "parameters": {}},
                     _ok, enabled=lambda cfg: cfg.get("allow", False))
        assert reg.schemas({"allow": False}) == []
        assert [s["name"] for s in reg.schemas({"allow": True})] == ["gated"]
        assert reg.has_tools({"allow": True}) and not reg.has_tools({"allow": False})

    def test_serialize_truncates(self, monkeypatch):
        monkeypatch.setattr(config, "tool_result_max_chars", 20)
        s = serialize_tool_result({"data": "x" * 100})
        assert len(s) <= 20 + len(" …[truncated]") and s.endswith("…[truncated]")


# --------------------------------------------------------------------------- the loop

class TestToolLoop:
    @pytest.mark.asyncio
    async def test_no_calls_passthrough(self, monkeypatch):
        fake = _FakeRounds([("plain answer", [])])
        monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools", fake)
        out = await tool_loop.create_text_response_with_tool_loop(
            _Client(), messages=[{"role": "user", "content": "hi"}], tools=[],
            registry=_registry_with(), tool_context=ToolContext())
        assert out["text"] == "plain answer"
        assert out["local_tool_calls"] == []
        assert len(fake.invocations) == 1

    @pytest.mark.asyncio
    async def test_multi_round_appends_call_and_output_items(self, monkeypatch):
        fake = _FakeRounds([
            ("", [_call(call_id="c1")]),
            ("", [_call(call_id="c2", arguments='{"x": 2}')]),
            ("final answer", []),
        ])
        monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools", fake)
        out = await tool_loop.create_text_response_with_tool_loop(
            _Client(), messages=[{"role": "user", "content": "hi"}], tools=[],
            registry=_registry_with(), tool_context=ToolContext())
        assert out["text"] == "final answer"
        assert [c["ok"] for c in out["local_tool_calls"]] == [True, True]
        assert "echo" in out["tools_used"]
        # Round 3's input: original message + 2×(function_call + function_call_output)
        final_input = fake.invocations[2]["messages"]
        types = [m.get("type") for m in final_input[1:]]
        assert types == ["function_call", "function_call_output"] * 2
        assert final_input[2]["call_id"] == "c1" and '"ok": true' in final_input[2]["output"]

    @pytest.mark.asyncio
    async def test_round_cap_forces_final_answer(self, monkeypatch):
        monkeypatch.setattr(config, "max_tool_rounds", 2)
        monkeypatch.setattr(config, "max_tool_calls_per_turn", 99)
        fake = _FakeRounds([("stubborn", [_call()])])  # would call tools forever
        monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools", fake)
        out = await tool_loop.create_text_response_with_tool_loop(
            _Client(), messages=[], tools=[], registry=_registry_with(), tool_context=ToolContext())
        # 2 tool rounds + 1 forced tool_choice="none" round
        assert len(fake.invocations) == 3
        assert fake.invocations[-1]["tool_choice"] == "none"
        assert out["text"] == "stubborn"
        assert len(out["local_tool_calls"]) == 2

    @pytest.mark.asyncio
    async def test_call_cap_forces_final_answer(self, monkeypatch):
        monkeypatch.setattr(config, "max_tool_rounds", 99)
        monkeypatch.setattr(config, "max_tool_calls_per_turn", 3)
        fake = _FakeRounds([("done", [_call(call_id="a"), _call(call_id="b"), _call(call_id="c")])])
        monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools", fake)
        out = await tool_loop.create_text_response_with_tool_loop(
            _Client(), messages=[], tools=[], registry=_registry_with(), tool_context=ToolContext())
        assert fake.invocations[-1]["tool_choice"] == "none"
        assert len(out["local_tool_calls"]) == 3

    @pytest.mark.asyncio
    async def test_reasoning_items_replayed_before_their_call(self, monkeypatch):
        """gpt-5.5 runs with reasoning — its reasoning items must be replayed in order
        ahead of the paired function_call or the stateless API rejects round 2."""
        reasoning_entry = {"type": "reasoning", "item": {"type": "reasoning", "id": "rs_1"}}
        fake = _FakeRounds([
            ("", [reasoning_entry, _call(call_id="c1")]),
            ("done", []),
        ])
        monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools", fake)
        out = await tool_loop.create_text_response_with_tool_loop(
            _Client(), messages=[{"role": "user", "content": "hi"}], tools=[],
            registry=_registry_with(), tool_context=ToolContext())
        assert out["text"] == "done"
        replayed = fake.invocations[1]["messages"][1:]
        assert [m.get("type") for m in replayed] == ["reasoning", "function_call", "function_call_output"]
        assert replayed[0]["id"] == "rs_1"
        assert replayed[1]["call_id"] == "c1" and replayed[2]["call_id"] == "c1"
        # reasoning entries don't count as tool executions
        assert out["local_tool_calls"] == [{"name": "echo", "ok": True}]

    @pytest.mark.asyncio
    async def test_tool_error_fed_back_not_raised(self, monkeypatch):
        async def boom(ctx, args):
            raise RuntimeError("dead")
        fake = _FakeRounds([("", [_call()]), ("answered anyway", [])])
        monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools", fake)
        out = await tool_loop.create_text_response_with_tool_loop(
            _Client(), messages=[], tools=[], registry=_registry_with(executor=boom),
            tool_context=ToolContext())
        assert out["text"] == "answered anyway"
        assert out["local_tool_calls"] == [{"name": "echo", "ok": False}]
        # The error result still went back to the model as a function_call_output
        output_item = fake.invocations[1]["messages"][-1]
        assert output_item["type"] == "function_call_output" and "execution_error" in output_item["output"]

    @pytest.mark.asyncio
    async def test_streaming_loop_status_callbacks_and_result(self, monkeypatch):
        rounds = [("", [_call()]), ("streamed final", [])]
        state = {"n": 0}

        async def fake_streaming(client, messages, tools, stream_callback, tool_callback=None,
                                 function_call_sink=None, tool_choice=None, **params):
            text, calls = rounds[min(state["n"], len(rounds) - 1)]
            state["n"] += 1
            if tool_choice != "none" and function_call_sink is not None:
                function_call_sink.extend(calls)
            return text

        monkeypatch.setattr(tool_loop.responses_api, "create_streaming_response_with_tools", fake_streaming)
        events = []
        out = await tool_loop.create_streaming_response_with_tool_loop(
            _Client(), messages=[], tools=[], registry=_registry_with(),
            tool_context=ToolContext(), stream_callback=lambda c: None,
            tool_callback=lambda t, s: events.append((t, s)))
        assert out["text"] == "streamed final"
        assert ("local:echo", "started") in events and ("local:echo", "completed") in events


# --------------------------------------------------------------------------- streaming event handling

class _StreamSelf(_Client):
    """Fake OpenAIClient exposing _safe_api_call/_safe_stream_iteration over scripted events."""
    def __init__(self, events):
        self._events = events

    async def _safe_api_call(self, fn, operation_type=None, **params):
        self.request_params = params
        return self._events

    async def _safe_stream_iteration(self, stream, operation_type=None):
        for e in stream:
            yield e

    @property
    def client(self):
        return SimpleNamespace(responses=SimpleNamespace(create=None))


def _ev(type_, **kw):
    return SimpleNamespace(type=type_, **kw)


class TestStreamingFunctionCallEvents:
    @pytest.mark.asyncio
    async def test_tool_round_suppresses_text_and_flush(self):
        item = SimpleNamespace(type="function_call", call_id="c9", name="echo", arguments='{"a":1}')
        events = [
            _ev("response.created"),
            _ev("response.output_item.added", item=item),
            _ev("response.output_text.delta", delta="let me check…"),
            _ev("response.output_item.done", item=item),
            _ev("response.completed"),
        ]
        chunks = []
        sink = []
        text = await responses_api.create_streaming_response_with_tools(
            _StreamSelf(events), messages=[{"role": "user", "content": "hi"}], tools=[],
            stream_callback=lambda c: chunks.append(c), function_call_sink=sink)
        assert sink == [{"type": "function_call", "call_id": "c9", "name": "echo", "arguments": '{"a":1}'}]
        assert chunks == []          # no preamble streamed, no None flush
        assert text == ""            # tool-round text discarded

    @pytest.mark.asyncio
    async def test_final_round_streams_and_flushes(self):
        events = [
            _ev("response.created"),
            _ev("response.output_text.delta", delta="hello "),
            _ev("response.output_text.delta", delta="world"),
            _ev("response.completed"),
        ]
        chunks = []
        sink = []
        text = await responses_api.create_streaming_response_with_tools(
            _StreamSelf(events), messages=[{"role": "user", "content": "hi"}], tools=[],
            stream_callback=lambda c: chunks.append(c), function_call_sink=sink)
        assert text == "hello world"
        assert chunks == ["hello ", "world", None]  # None = completion flush
        assert sink == []

    @pytest.mark.asyncio
    async def test_raw_items_pass_through_input(self):
        s = _StreamSelf([_ev("response.completed")])
        await responses_api.create_streaming_response_with_tools(
            s, messages=[
                {"role": "user", "content": "hi"},
                {"type": "function_call", "call_id": "c1", "name": "echo", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "{}"},
            ], tools=[], stream_callback=lambda c: None, function_call_sink=[])
        sent = s.request_params["input"]
        assert sent[1]["type"] == "function_call" and sent[2]["type"] == "function_call_output"

    @pytest.mark.asyncio
    async def test_tool_choice_forwarded(self):
        s = _StreamSelf([_ev("response.completed")])
        await responses_api.create_streaming_response_with_tools(
            s, messages=[{"role": "user", "content": "hi"}], tools=[],
            stream_callback=lambda c: None, tool_choice="none")
        assert s.request_params["tool_choice"] == "none"

    @pytest.mark.asyncio
    async def test_reasoning_only_round_still_flushes(self):
        """Reasoning items land in the sink, but without a function_call the round is
        final — the completion flush (stream_callback(None)) must still fire."""
        class _R:
            type = "reasoning"
            def model_dump(self, **kw):
                return {"type": "reasoning", "id": "rs_1", "summary": []}
        events = [
            _ev("response.output_item.done", item=_R()),
            _ev("response.output_text.delta", delta="answer"),
            _ev("response.completed"),
        ]
        chunks = []
        sink = []
        text = await responses_api.create_streaming_response_with_tools(
            _StreamSelf(events), messages=[{"role": "user", "content": "hi"}], tools=[],
            stream_callback=lambda c: chunks.append(c), function_call_sink=sink)
        assert text == "answer"
        assert chunks[-1] is None  # flush fired
        assert sink[0]["type"] == "reasoning"

    @pytest.mark.asyncio
    async def test_include_encrypted_reasoning_when_loop_active(self):
        s = _StreamSelf([_ev("response.completed")])
        await responses_api.create_streaming_response_with_tools(
            s, messages=[{"role": "user", "content": "hi"}], tools=[],
            stream_callback=lambda c: None, function_call_sink=[])
        assert s.request_params["include"] == ["reasoning.encrypted_content"]


# --------------------------------------------------------------------------- react tool

def _react_self():
    s = MagicMock()
    s.react = AsyncMock(return_value=True)
    s.get_react_tool_schema = SlackMessagingMixin.get_react_tool_schema.__get__(s)
    s.execute_react_tool = SlackMessagingMixin.execute_react_tool.__get__(s)
    s._reserve_and_react = SlackMessagingMixin._reserve_and_react.__get__(s)
    # start with no reservation guard (None so the executor initializes it)
    s._reaction_guard = None
    return s


class TestReactTool:
    def setup_method(self):
        self.ctx = ToolContext(channel_id="C1", thread_ts="100.0", trigger_ts="123.4")

    @pytest.mark.asyncio
    async def test_happy_path_defaults_to_trigger_ts(self, monkeypatch):
        monkeypatch.setattr(config, "enable_reactions", True)
        monkeypatch.setattr(config, "enable_react_tool", True)
        monkeypatch.setattr(config, "reaction_emojis", ["thumbsup", "tada"])
        s = _react_self()
        out = await s.execute_react_tool(self.ctx, {"emoji": "tada"})
        assert out["ok"] is True and out["ts"] == "123.4"
        s.react.assert_awaited_once_with("C1", "123.4", "tada")

    @pytest.mark.asyncio
    async def test_explicit_ts_and_colon_stripping(self, monkeypatch):
        monkeypatch.setattr(config, "enable_reactions", True)
        monkeypatch.setattr(config, "enable_react_tool", True)
        monkeypatch.setattr(config, "reaction_emojis", ["thumbsup"])
        s = _react_self()
        out = await s.execute_react_tool(self.ctx, {"emoji": ":thumbsup:", "ts": "999.9"})
        assert out["ok"] is True
        s.react.assert_awaited_once_with("C1", "999.9", "thumbsup")

    @pytest.mark.asyncio
    async def test_allowlist_refusal(self, monkeypatch):
        monkeypatch.setattr(config, "enable_reactions", True)
        monkeypatch.setattr(config, "enable_react_tool", True)
        monkeypatch.setattr(config, "reaction_emojis", ["thumbsup"])
        s = _react_self()
        out = await s.execute_react_tool(self.ctx, {"emoji": "middle_finger"})
        assert out["ok"] is False and out["error"] == "emoji_not_allowed"
        s.react.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_two_distinct_emoji_both_land(self, monkeypatch):
        # F6: a user asking for several distinct emoji on one message gets several.
        monkeypatch.setattr(config, "enable_reactions", True)
        monkeypatch.setattr(config, "enable_react_tool", True)
        monkeypatch.setattr(config, "reaction_emojis", ["thumbsup", "tada"])
        monkeypatch.setattr(config, "reaction_max_per_message", 4)
        s = _react_self()
        assert (await s.execute_react_tool(self.ctx, {"emoji": "thumbsup"}))["ok"] is True
        assert (await s.execute_react_tool(self.ctx, {"emoji": "tada"}))["ok"] is True
        assert s.react.await_count == 2

    @pytest.mark.asyncio
    async def test_duplicate_emoji_idempotent_no_slot(self, monkeypatch):
        # Same emoji twice: idempotent success, no second Slack call, no slot consumed.
        monkeypatch.setattr(config, "enable_reactions", True)
        monkeypatch.setattr(config, "enable_react_tool", True)
        monkeypatch.setattr(config, "reaction_emojis", ["thumbsup", "tada"])
        monkeypatch.setattr(config, "reaction_max_per_message", 1)
        s = _react_self()
        assert (await s.execute_react_tool(self.ctx, {"emoji": "thumbsup"}))["ok"] is True
        out = await s.execute_react_tool(self.ctx, {"emoji": "thumbsup"})
        assert out["ok"] is True and out.get("idempotent") is True
        assert s.react.await_count == 1  # cap=1 not consumed by the duplicate

    @pytest.mark.asyncio
    async def test_cap_refusal_at_n_plus_1(self, monkeypatch):
        monkeypatch.setattr(config, "enable_reactions", True)
        monkeypatch.setattr(config, "enable_react_tool", True)
        monkeypatch.setattr(config, "reaction_emojis", ["thumbsup", "tada", "eyes"])
        monkeypatch.setattr(config, "reaction_max_per_message", 2)
        s = _react_self()
        assert (await s.execute_react_tool(self.ctx, {"emoji": "thumbsup"}))["ok"] is True
        assert (await s.execute_react_tool(self.ctx, {"emoji": "tada"}))["ok"] is True
        out = await s.execute_react_tool(self.ctx, {"emoji": "eyes"})
        assert out["ok"] is False and out["error"] == "reaction_cap"
        assert "2" in out["message"]
        assert s.react.await_count == 2

    @pytest.mark.asyncio
    async def test_failed_react_rolls_back_reservation(self, monkeypatch):
        # A failed Slack call must free its slot so a later distinct emoji can still land.
        monkeypatch.setattr(config, "enable_reactions", True)
        monkeypatch.setattr(config, "enable_react_tool", True)
        monkeypatch.setattr(config, "reaction_emojis", ["thumbsup", "tada"])
        monkeypatch.setattr(config, "reaction_max_per_message", 1)
        s = _react_self()
        s.react = AsyncMock(side_effect=[False, True])
        assert (await s.execute_react_tool(self.ctx, {"emoji": "thumbsup"}))["ok"] is False
        # reservation rolled back → cap slot free again
        assert (await s.execute_react_tool(self.ctx, {"emoji": "tada"}))["ok"] is True

    @pytest.mark.asyncio
    async def test_concurrent_distinct_emoji_respect_cap(self, monkeypatch):
        # Siblings dispatched concurrently (as dispatch_all does) can't both pass a cap of 1.
        monkeypatch.setattr(config, "enable_reactions", True)
        monkeypatch.setattr(config, "enable_react_tool", True)
        monkeypatch.setattr(config, "reaction_emojis", ["thumbsup", "tada"])
        monkeypatch.setattr(config, "reaction_max_per_message", 1)
        s = _react_self()

        async def _slow_react(*a, **k):
            await asyncio.sleep(0)
            return True
        s.react = AsyncMock(side_effect=_slow_react)
        results = await asyncio.gather(
            s.execute_react_tool(self.ctx, {"emoji": "thumbsup"}),
            s.execute_react_tool(self.ctx, {"emoji": "tada"}),
        )
        oks = [r["ok"] for r in results]
        assert oks.count(True) == 1 and oks.count(False) == 1
        assert next(r for r in results if not r["ok"])["error"] == "reaction_cap"

    @pytest.mark.asyncio
    async def test_concurrent_identical_emoji_first_fails(self, monkeypatch):
        # Two concurrent identical-emoji calls where the OWNER fails: the duplicate must
        # NOT report success (round-2 fix a).
        monkeypatch.setattr(config, "enable_reactions", True)
        monkeypatch.setattr(config, "enable_react_tool", True)
        monkeypatch.setattr(config, "reaction_emojis", ["thumbsup"])
        monkeypatch.setattr(config, "reaction_max_per_message", 4)
        s = _react_self()
        started = asyncio.Event()

        async def _fail_react(*a, **k):
            started.set()
            await asyncio.sleep(0.01)
            return False
        s.react = AsyncMock(side_effect=_fail_react)
        results = await asyncio.gather(
            s.execute_react_tool(self.ctx, {"emoji": "thumbsup"}),
            s.execute_react_tool(self.ctx, {"emoji": "thumbsup"}),
        )
        assert all(r["ok"] is False for r in results)

    @pytest.mark.asyncio
    async def test_gating(self, monkeypatch):
        monkeypatch.setattr(config, "enable_reactions", True)
        monkeypatch.setattr(config, "enable_react_tool", False)
        s = _react_self()
        out = await s.execute_react_tool(self.ctx, {"emoji": "thumbsup"})
        assert out["ok"] is False and out["error"] == "disabled"

    def test_schema_enum_is_allowlist(self, monkeypatch):
        monkeypatch.setattr(config, "reaction_emojis", ["thumbsup", ":eyes:"])
        s = _react_self()
        schema = s.get_react_tool_schema()
        assert schema["name"] == "react_to_message"
        assert schema["parameters"]["properties"]["emoji"]["enum"] == ["thumbsup", "eyes"]


# --------------------------------------------------------------------------- processor glue

class TestProcessorGlue:
    def test_reaction_only_detection(self):
        is_ro = TextHandlerMixin._is_reaction_only
        assert is_ro("", [{"name": "react_to_message", "ok": True}]) is True
        assert is_ro("  \n", [{"name": "react_to_message", "ok": True}]) is True
        assert is_ro("text", [{"name": "react_to_message", "ok": True}]) is False
        assert is_ro("", [{"name": "react_to_message", "ok": False}]) is False
        assert is_ro("", [{"name": "fetch_thread_messages", "ok": True}]) is False
        assert is_ro("", []) is False
        assert is_ro("", None) is False

    def test_build_tools_array_appends_registry_schemas(self):
        h = MagicMock()
        h.log_debug = MagicMock()
        h.mcp_manager.has_mcp_servers.return_value = False
        reg = _registry_with(name="fetch_channel_history")
        tools = TextHandlerMixin._build_tools_array(
            h, {"enable_web_search": True, "enable_mcp": False}, "gpt-5.5", registry=reg)
        types = [(t.get("type"), t.get("name")) for t in tools]
        assert ("function", "fetch_channel_history") in types
        assert ("web_search", None) in types

    def test_build_tools_array_without_registry_unchanged(self):
        h = MagicMock()
        h.log_debug = MagicMock()
        h.mcp_manager.has_mcp_servers.return_value = False
        tools = TextHandlerMixin._build_tools_array(
            h, {"enable_web_search": True, "enable_mcp": False}, "gpt-5.5")
        assert tools == [{"type": "web_search"}]

    def test_registry_builder_registers_history_react_and_search(self, monkeypatch):
        from slack_client.base import SlackBot
        monkeypatch.setattr(config, "enable_history_tools", True)
        monkeypatch.setattr(config, "enable_reactions", True)
        monkeypatch.setattr(config, "enable_react_tool", True)
        monkeypatch.setattr(config, "reaction_emojis", ["thumbsup"])
        monkeypatch.setattr(config, "enable_search_tool", True)
        monkeypatch.setattr(config, "enable_channel_memory", True)
        s = MagicMock()
        s.get_history_tools_for_openai.return_value = [
            {"type": "function", "name": "fetch_channel_history", "parameters": {}},
            {"type": "function", "name": "fetch_thread_messages", "parameters": {}},
        ]
        s.get_react_tool_schema.return_value = {
            "type": "function", "name": "react_to_message", "parameters": {}}
        s.get_search_tool_schema.return_value = {
            "type": "function", "name": "search_slack", "parameters": {}}
        monkeypatch.setattr(config, "enable_read_document_tool", True)
        registry = SlackBot._build_tool_registry(s)
        names = {t["name"] for t in registry.schemas()}
        assert names == {"fetch_channel_history", "fetch_thread_messages",
                         "react_to_message", "search_slack",
                         "remember_fact", "update_fact", "forget_fact",
                         "read_document"}

    def test_registry_builder_search_gated_off(self, monkeypatch):
        from slack_client.base import SlackBot
        monkeypatch.setattr(config, "enable_history_tools", False)
        monkeypatch.setattr(config, "enable_reactions", False)
        monkeypatch.setattr(config, "enable_search_tool", False)
        s = MagicMock()
        s.get_history_tools_for_openai.return_value = []
        registry = SlackBot._build_tool_registry(s)
        assert "search_slack" not in {t["name"] for t in registry.schemas()}

    @pytest.mark.asyncio
    async def test_registry_history_executor_routes_by_name(self, monkeypatch):
        from slack_client.base import SlackBot
        monkeypatch.setattr(config, "enable_reactions", False)
        s = MagicMock()
        s.get_history_tools_for_openai.return_value = [
            {"type": "function", "name": "fetch_channel_history", "parameters": {}},
            {"type": "function", "name": "fetch_thread_messages", "parameters": {}},
        ]
        s.dispatch_history_tool_call = AsyncMock(return_value={"ok": True})
        registry = SlackBot._build_tool_registry(s)
        ctx = ToolContext()
        await registry.dispatch(ctx, "fetch_thread_messages", '{"channel_id": "C1"}')
        # ctx rides along so omitted channel_id/thread_ts default to the current conversation
        s.dispatch_history_tool_call.assert_awaited_once_with(
            "fetch_thread_messages", {"channel_id": "C1"}, ctx)
