"""Redesign Phases A + D — local function-call loop, ToolRegistry, react tool, footer shape.

All stubbed I/O: no live Slack, no live OpenAI, no legacy suite.
"""
import asyncio
from collections import Counter
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


def _ok_exec():
    """A second executor for registries that need more than one tool (F37 free-tool tests)."""
    return lambda ctx, args: _ok(args)


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

    @pytest.mark.asyncio
    async def test_per_tool_timeout_overrides_config(self, monkeypatch):
        # A generous global cap must not save a tool with its own tiny per-tool timeout.
        monkeypatch.setattr(config, "tool_call_timeout", 100.0)
        reg = ToolRegistry()
        async def slow(ctx, args):
            await asyncio.sleep(1)
        reg.register({"type": "function", "name": "slow", "parameters": {}},
                     slow, timeout=0.05)
        out = await reg.dispatch(ToolContext(), "slow", "{}")
        assert out["ok"] is False and out["error"] == "timeout"
        assert "0s" in out["message"]  # honest: reports the per-tool number, not config's

    @pytest.mark.asyncio
    async def test_default_tools_still_use_config_timeout(self, monkeypatch):
        # timeout=None (the default) falls through to config.tool_call_timeout.
        monkeypatch.setattr(config, "tool_call_timeout", 0.05)
        async def slow(ctx, args):
            await asyncio.sleep(1)
        reg = _registry_with(executor=slow)  # registered without a per-tool timeout
        out = await reg.dispatch(ToolContext(), "echo", "{}")
        assert out["ok"] is False and out["error"] == "timeout"

    @pytest.mark.asyncio
    async def test_per_tool_timeout_allows_slow_run_config_would_abort(self, monkeypatch):
        # The whole point: a run longer than the global cap succeeds under a larger per-tool one.
        monkeypatch.setattr(config, "tool_call_timeout", 0.02)
        reg = ToolRegistry()
        async def okish(ctx, args):
            await asyncio.sleep(0.1)
            return {"ok": True}
        reg.register({"type": "function", "name": "okish", "parameters": {}},
                     okish, timeout=5.0)
        out = await reg.dispatch(ToolContext(), "okish", "{}")
        assert out == {"ok": True}

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
        # reasoning entries don't count as tool executions (F7 also captures an arg gist)
        assert out["local_tool_calls"] == [{"name": "echo", "ok": True, "gist": "x=<str>"}]

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
        assert out["local_tool_calls"] == [{"name": "echo", "ok": False, "gist": "x=<str>"}]
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

    @pytest.mark.asyncio
    async def test_streaming_loop_returns_the_seam_joined_aggregate_not_just_the_last_round(
            self, monkeypatch):
        """A pre-tool preamble and the post-tool text are SEPARATE rounds. The old return handed
        back only the last round ("Fixed."), so the thread state remembered a different string
        than Slack displayed (which shows every round). Now the loop returns the canonical join,
        seam-separated exactly like the buffer the handler streams."""
        rounds = [("Fixing the chopsticks under Super Heavy.", [_call()]), ("Fixed.", [])]
        state = {"n": 0}

        async def fake_streaming(client, messages, tools, stream_callback, tool_callback=None,
                                 function_call_sink=None, tool_choice=None, **params):
            text, calls = rounds[min(state["n"], len(rounds) - 1)]
            state["n"] += 1
            if tool_choice != "none" and function_call_sink is not None:
                function_call_sink.extend(calls)
            return text

        monkeypatch.setattr(tool_loop.responses_api, "create_streaming_response_with_tools",
                            fake_streaming)
        out = await tool_loop.create_streaming_response_with_tool_loop(
            _Client(), messages=[], tools=[], registry=_registry_with(),
            tool_context=ToolContext(), stream_callback=lambda c: None,
            tool_callback=lambda t, s: None, aggregate_segments=True)
        assert out["text"] == "Fixing the chopsticks under Super Heavy.\n\nFixed.", out["text"]

    @pytest.mark.asyncio
    async def test_aggregation_is_opt_in_default_returns_final_round_only(self, monkeypatch):
        """Deep research reads result["text"] as the report and never streams the intermediate
        "I'll search…" preambles to Slack — so the DEFAULT must stay final-round-only, or those
        preambles leak into the report. Only the chat handler passes aggregate_segments=True."""
        rounds = [("I'll look into that.", [_call()]), ("The final report.", [])]
        state = {"n": 0}

        async def fake_streaming(client, messages, tools, stream_callback, tool_callback=None,
                                 function_call_sink=None, tool_choice=None, **params):
            text, calls = rounds[min(state["n"], len(rounds) - 1)]
            state["n"] += 1
            if tool_choice != "none" and function_call_sink is not None:
                function_call_sink.extend(calls)
            return text

        monkeypatch.setattr(tool_loop.responses_api, "create_streaming_response_with_tools",
                            fake_streaming)
        out = await tool_loop.create_streaming_response_with_tool_loop(
            _Client(), messages=[], tools=[], registry=_registry_with(),
            tool_context=ToolContext(), stream_callback=lambda c: None,
            tool_callback=lambda t, s: None)                      # no aggregate_segments
        assert out["text"] == "The final report.", out["text"]

    @pytest.mark.asyncio
    async def test_aggregate_keeps_whitespace_only_rounds_to_match_the_buffer(self, monkeypatch):
        """A wholly whitespace round is real committed text the handler's buffer keeps, so the
        aggregate must keep it too — dropping it would desync persisted text from the display
        ("A\\nB" on screen vs "A\\n\\nB" persisted)."""
        rounds = [("A", [_call()]), ("\n", [_call()]), ("B", [])]
        state = {"n": 0}

        async def fake_streaming(client, messages, tools, stream_callback, tool_callback=None,
                                 function_call_sink=None, tool_choice=None, **params):
            text, calls = rounds[min(state["n"], len(rounds) - 1)]
            state["n"] += 1
            if tool_choice != "none" and function_call_sink is not None:
                function_call_sink.extend(calls)
            return text

        monkeypatch.setattr(tool_loop.responses_api, "create_streaming_response_with_tools",
                            fake_streaming)
        out = await tool_loop.create_streaming_response_with_tool_loop(
            _Client(), messages=[], tools=[], registry=_registry_with(),
            tool_context=ToolContext(), stream_callback=lambda c: None,
            tool_callback=lambda t, s: None, aggregate_segments=True)
        assert out["text"] == "A\nB", out["text"]      # the "\n" round is kept; no doubled seam

    @pytest.mark.asyncio
    async def test_a_seeded_tool_choice_seeds_only_the_first_round(self, monkeypatch):
        """F37 forces the delivery-plan call with tool_choice="required". Left set, it would
        force the SAME tool again on the next round: the model would be made to re-answer a
        question it had already answered, and _plan_delivery's executor would OVERWRITE the first
        plan with the second. Round 1 forced; never again."""
        seen = []
        state = {"n": 0}

        async def fake_streaming(client, messages, tools, stream_callback, tool_callback=None,
                                 function_call_sink=None, tool_choice=None, **params):
            seen.append(tool_choice)
            # Only the forced round produces a call; afterwards the model would answer in text.
            if tool_choice == "required" and function_call_sink is not None:
                function_call_sink.extend([_call()])
            state["n"] += 1
            return "" if tool_choice == "required" else "done"

        monkeypatch.setattr(tool_loop.responses_api, "create_streaming_response_with_tools",
                            fake_streaming)
        out = await tool_loop.create_streaming_response_with_tool_loop(
            _Client(), messages=[], tools=[], registry=_registry_with(),
            tool_context=ToolContext(), stream_callback=lambda c: None,
            max_tool_rounds=1, max_tool_calls=1, tool_choice="required")

        assert seen[0] == "required"          # round 1 is forced...
        assert "required" not in seen[1:]     # ...and no round after it is
        assert seen[1] == "none"              # the cap of 1 drives the wind-down round
        assert out["text"] == "done"

    @pytest.mark.asyncio
    async def test_bookkeeping_calls_do_not_spend_the_budget(self, monkeypatch):
        """F37 — THE ship-blocker this feature was rejected for.

        A live todo list fires on every transition. On the meter, those calls eat the round
        budget that the build phase needs for mount_file / create_image_asset: the status card
        would starve the deck it is reporting on, and the loop would force a final answer before
        the file was ever built. A card update is not what a runaway guard is guarding against.

        Here: a cap of 2 productive rounds, and the model interleaves bookkeeping throughout.
        Both real tools must still get to run."""
        registry = _registry_with(name="update_todos")
        registry.register({"type": "function", "name": "mount_file",
                           "parameters": {"type": "object", "properties": {}}}, _ok_exec())
        rounds = [
            [_call("update_todos", "a1")],                  # free
            [_call("update_todos", "a2")],                  # free
            [_call("mount_file", "b1")],                    # productive #1
            [_call("update_todos", "a3")],                  # free
            [_call("mount_file", "b2")],                    # productive #2 -> hits the cap
            [],
        ]
        state = {"n": 0}
        dispatched = []

        async def fake_streaming(client, messages, tools, stream_callback, tool_callback=None,
                                 function_call_sink=None, tool_choice=None, **params):
            i = state["n"]
            state["n"] += 1
            calls = rounds[i] if i < len(rounds) else []
            if tool_choice != "none" and function_call_sink is not None:
                function_call_sink.extend(calls)
                dispatched.extend(c["name"] for c in calls)
            return "final" if tool_choice == "none" or not calls else ""

        monkeypatch.setattr(tool_loop.responses_api, "create_streaming_response_with_tools",
                            fake_streaming)
        out = await tool_loop.create_streaming_response_with_tool_loop(
            _Client(), messages=[], tools=[], registry=registry,
            tool_context=ToolContext(), stream_callback=lambda c: None,
            max_tool_rounds=2, max_tool_calls=2, free_tools=("update_todos",))

        # Three bookkeeping calls rode free, and BOTH productive calls still got their turn.
        assert dispatched.count("update_todos") == 3
        assert dispatched.count("mount_file") == 2, (
            "a chatty todo list starved the real work — the exact bug this exists to prevent")
        assert out["text"] == "final"

    @pytest.mark.asyncio
    async def test_free_does_not_mean_unbounded(self, monkeypatch):
        """"Free" is a budget exemption, not a licence to loop forever. A model that only ever
        calls update_todos is still a runaway — just a cheap one — so free rounds have a ceiling
        of their own and the loop is forced to answer."""
        state = {"n": 0}

        async def fake_streaming(client, messages, tools, stream_callback, tool_callback=None,
                                 function_call_sink=None, tool_choice=None, **params):
            state["n"] += 1
            if tool_choice != "none" and function_call_sink is not None:
                function_call_sink.extend([_call("update_todos", f"c{state['n']}")])
            return "gave up" if tool_choice == "none" else ""

        monkeypatch.setattr(tool_loop.responses_api, "create_streaming_response_with_tools",
                            fake_streaming)
        out = await tool_loop.create_streaming_response_with_tool_loop(
            _Client(), messages=[], tools=[], registry=_registry_with(name="update_todos"),
            tool_context=ToolContext(), stream_callback=lambda c: None,
            max_tool_rounds=2, max_tool_calls=2, free_tools=("update_todos",))

        # It terminated instead of spinning: the free ceiling is a multiple of the real cap.
        assert out["text"] == "gave up"
        assert state["n"] <= 2 * tool_loop._FREE_ROUND_CEILING + 2

    @pytest.mark.asyncio
    async def test_a_burst_of_free_calls_is_refused_before_it_is_dispatched(self, monkeypatch):
        """Codex review, round 2. Counting free calls only stops the NEXT round — by then the
        storm has already happened: a round's tool calls dispatch in PARALLEL, so one "free"
        round firing twenty update_todos means twenty concurrent executors and twenty Slack
        updates. The excess must be refused BEFORE dispatch.

        And refused, not dropped: a function_call left without a matching function_call_output
        is a 400 on the very next request. Every suppressed call still gets an answer."""
        executed = []

        async def _spy(ctx, args):
            executed.append(state["n"])          # which round dispatched it
            return {"ok": True}

        state = {"n": 0, "last_input": []}

        async def fake_streaming(client, messages, tools, stream_callback, tool_callback=None,
                                 function_call_sink=None, tool_choice=None, **params):
            state["n"] += 1
            state["last_input"] = list(messages)     # what the NEXT request would send
            if tool_choice != "none" and function_call_sink is not None:
                function_call_sink.extend(
                    _call("update_todos", f"c{state['n']}_{i}") for i in range(20))
            return "stopped" if tool_choice == "none" else ""

        monkeypatch.setattr(tool_loop.responses_api, "create_streaming_response_with_tools",
                            fake_streaming)
        out = await tool_loop.create_streaming_response_with_tool_loop(
            _Client(), messages=[], tools=[], registry=_registry_with("update_todos", _spy),
            tool_context=ToolContext(), stream_callback=lambda c: None,
            max_tool_rounds=2, max_tool_calls=2, free_tools=("update_todos",))

        assert out["text"] == "stopped"
        # No ROUND dispatched more than the burst cap — 18 of each 20 were never run at all.
        per_round = Counter(executed)
        assert per_round and max(per_round.values()) <= tool_loop._FREE_CALLS_PER_ROUND, (
            f"a round dispatched {max(per_round.values())} bookkeeping calls at once")
        # ...and yet EVERY call was ANSWERED: a function_call with no matching
        # function_call_output is a 400 on the very next request. Suppressed ≠ dropped.
        replayed = state["last_input"]
        made = [m for m in replayed if m.get("type") == "function_call"]
        answered = [m for m in replayed if m.get("type") == "function_call_output"]
        assert made and {m["call_id"] for m in made} == {m["call_id"] for m in answered}
        refused = [m for m in answered if "too_many_calls_this_round" in str(m.get("output"))]
        assert len(refused) == len(made) - len(executed)

    @pytest.mark.asyncio
    async def test_a_mixed_round_is_charged_normally(self, monkeypatch):
        """Bookkeeping riding ALONGSIDE real work does not launder the round. Only the free
        calls are free; the round itself did real work and is billed for it."""
        registry = _registry_with(name="update_todos")
        registry.register({"type": "function", "name": "mount_file",
                           "parameters": {"type": "object", "properties": {}}}, _ok_exec())
        state = {"n": 0}

        async def fake_streaming(client, messages, tools, stream_callback, tool_callback=None,
                                 function_call_sink=None, tool_choice=None, **params):
            state["n"] += 1
            if tool_choice != "none" and function_call_sink is not None:
                function_call_sink.extend([_call("update_todos", f"a{state['n']}"),
                                           _call("mount_file", f"b{state['n']}")])
            return "final" if tool_choice == "none" else ""

        monkeypatch.setattr(tool_loop.responses_api, "create_streaming_response_with_tools",
                            fake_streaming)
        await tool_loop.create_streaming_response_with_tool_loop(
            _Client(), messages=[], tools=[], registry=registry,
            tool_context=ToolContext(), stream_callback=lambda c: None,
            max_tool_rounds=1, max_tool_calls=9, free_tools=("update_todos",))

        # One mixed round exhausted the 1-round cap: round 2 is the forced wind-down.
        assert state["n"] == 2


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

    # F38: the guard now calls _react_add (which reports whether THIS call actually added
    # the reaction, vs Slack's already_reacted). Delegate to the `react` stub these tests
    # drive, treating a successful add as genuinely ours — the already_reacted case gets
    # its own explicit coverage in test_ack_lifecycle.py.
    async def _react_add(channel_id, ts, emoji):
        ok = await s.react(channel_id, ts, emoji)
        return ok, ok
    s._react_add = _react_add
    s.get_react_tool_schema = SlackMessagingMixin.get_react_tool_schema.__get__(s)
    s.execute_react_tool = SlackMessagingMixin.execute_react_tool.__get__(s)
    s._reserve_and_react = SlackMessagingMixin._reserve_and_react.__get__(s)
    s._reserve_and_react_owned = SlackMessagingMixin._reserve_and_react_owned.__get__(s)
    s._reserve_once = SlackMessagingMixin._reserve_once.__get__(s)
    s.settle_reaction_lease = SlackMessagingMixin.settle_reaction_lease.__get__(s)
    s._is_committed = SlackMessagingMixin._is_committed
    s._REMOVING = SlackMessagingMixin._REMOVING
    s._trim_reaction_guard = SlackMessagingMixin._trim_reaction_guard.__get__(s)
    s._REACTION_GUARD_MAX = SlackMessagingMixin._REACTION_GUARD_MAX
    s._REACTION_GUARD_RECENCY_S = SlackMessagingMixin._REACTION_GUARD_RECENCY_S
    # start with no reservation guard (None so the executor initializes it)
    s._reaction_guard = None
    s._reaction_guard_ts = None
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
    async def test_recent_entries_pinned_stale_evicted(self, monkeypatch):
        # Round-3: eviction PINS entries touched within the recency window — both a fresh
        # PENDING reservation AND the active turn's COMMITTED slots (so a burst of 2000+
        # reactions on other messages can't resurrect a message's consumed slots). Stale
        # committed entries are the ones evicted.
        from collections import OrderedDict
        import time as _time
        monkeypatch.setattr(config, "enable_reactions", True)
        monkeypatch.setattr(config, "enable_react_tool", True)
        monkeypatch.setattr(config, "reaction_emojis", ["thumbsup"])
        monkeypatch.setattr(config, "reaction_max_per_message", 4)
        s = _react_self()
        s.react = AsyncMock(return_value=True)

        now = _time.monotonic()
        guard = s._reaction_guard = OrderedDict()
        ts_map = s._reaction_guard_ts = {}
        stale = now - s._REACTION_GUARD_RECENCY_S - 10  # outside the recency window

        fresh_pending = asyncio.get_event_loop().create_future()
        guard[("C1", "fresh_pending")] = {"eyes": fresh_pending}
        ts_map[("C1", "fresh_pending")] = now
        guard[("C1", "recent_committed")] = {"thumbsup": True}
        ts_map[("C1", "recent_committed")] = now
        for i in range(2000):
            guard[("C1", f"c{i}")] = {"thumbsup": True}
            ts_map[("C1", f"c{i}")] = stale

        await s._reserve_and_react("C1", "brandnew", "thumbsup")

        assert ("C1", "fresh_pending") in guard       # fresh pending pinned
        assert ("C1", "recent_committed") in guard     # active-turn committed pinned (no resurrection)
        assert len(guard) <= 2000                       # stale committed evicted instead
        fresh_pending.set_result(False)

    @pytest.mark.asyncio
    async def test_stale_pending_future_is_evictable(self, monkeypatch):
        # Bounded expiry: a pending future untouched for the whole recency window is
        # abandoned and becomes evictable, so a never-resolving Future can't pin forever.
        from collections import OrderedDict
        import time as _time
        monkeypatch.setattr(config, "enable_reactions", True)
        monkeypatch.setattr(config, "enable_react_tool", True)
        monkeypatch.setattr(config, "reaction_emojis", ["thumbsup"])
        monkeypatch.setattr(config, "reaction_max_per_message", 4)
        s = _react_self()
        s.react = AsyncMock(return_value=True)

        now = _time.monotonic()
        guard = s._reaction_guard = OrderedDict()
        ts_map = s._reaction_guard_ts = {}
        stale = now - s._REACTION_GUARD_RECENCY_S - 10

        abandoned = asyncio.get_event_loop().create_future()
        guard[("C1", "abandoned")] = {"eyes": abandoned}
        ts_map[("C1", "abandoned")] = stale
        for i in range(2000):
            guard[("C1", f"c{i}")] = {"thumbsup": True}
            ts_map[("C1", f"c{i}")] = now  # fresh committed → pinned

        await s._reserve_and_react("C1", "brandnew", "thumbsup")

        assert ("C1", "abandoned") not in guard  # abandoned stale pending evicted
        abandoned.set_result(False)

    @pytest.mark.asyncio
    async def test_rollback_is_identity_conditional(self, monkeypatch):
        # F3: a failed reservation's finally must NOT delete a REPLACEMENT entry that a
        # concurrent recreate installed under the same key (different dict identity).
        from collections import OrderedDict
        monkeypatch.setattr(config, "enable_reactions", True)
        monkeypatch.setattr(config, "enable_react_tool", True)
        monkeypatch.setattr(config, "reaction_emojis", ["thumbsup"])
        monkeypatch.setattr(config, "reaction_max_per_message", 4)
        s = _react_self()
        guard = s._reaction_guard = OrderedDict()

        async def _fail_and_swap(*a, **k):
            # Mid-flight, a concurrent recreate installs a DIFFERENT dict for the key.
            guard[("C1", "k")] = {"tada": True}
            return False
        s.react = AsyncMock(side_effect=_fail_and_swap)

        out = await s._reserve_and_react("C1", "k", "thumbsup")
        assert out["ok"] is False
        # The owner's finally left the replacement entry intact.
        assert guard.get(("C1", "k")) == {"tada": True}

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

    @pytest.mark.asyncio
    async def test_gate_react_consumes_a_guard_slot(self, monkeypatch):
        # F6 addendum: the participation gate reacts via _reserve_and_react, so a later
        # main-model turn on the same message honestly sees the slot consumed.
        monkeypatch.setattr(config, "enable_reactions", True)
        monkeypatch.setattr(config, "enable_react_tool", True)
        monkeypatch.setattr(config, "reaction_emojis", ["eyes", "tada"])
        monkeypatch.setattr(config, "reaction_max_per_message", 1)
        s = _react_self()
        gate = await s._reserve_and_react("C1", "123.4", "eyes")
        assert gate["ok"] is True
        out = await s.execute_react_tool(
            ToolContext(channel_id="C1", trigger_ts="123.4"), {"emoji": "tada"})
        assert out["ok"] is False and out["error"] == "reaction_cap"


def test_participation_prompt_routes_explicit_reaction_requests_to_respond():
    # F6 addendum: an explicit "add some reactions" request must NOT die at the gate's
    # single-emoji react verdict — route it to respond so the main model places them.
    from prompts import PARTICIPATION_SYSTEM_PROMPT
    assert "EXPLICITLY asks the assistant to add a reaction" in PARTICIPATION_SYSTEM_PROMPT
    assert 'choose "respond"' in PARTICIPATION_SYSTEM_PROMPT


# --------------------------------------------------------------------------- processor glue

def _slack_tool_mock(history=None):
    """A stand-in SlackBot for _build_tool_registry.

    Every schema-returning method must hand back a REAL dict. A bare MagicMock is callable,
    and register() reads a callable schema as a per-request factory (F34's image tools shape
    themselves to the user's image model), so a MagicMock schema is a registration error —
    which would fail these tests for a reason that has nothing to do with what they assert.
    """
    s = MagicMock()
    s.get_history_tools_for_openai.return_value = history or []
    s.get_react_tool_schema.return_value = {
        "type": "function", "name": "react_to_message", "parameters": {}}
    s.get_search_tool_schema.return_value = {
        "type": "function", "name": "search_slack", "parameters": {}}
    s.get_post_to_thread_tool_schema.return_value = {
        "type": "function", "name": "post_to_thread", "parameters": {}}
    s.get_no_reply_tool_schema.return_value = {
        "type": "function", "name": "no_response_needed", "parameters": {}}
    return s


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
        # F32: code_interpreter off — this test asserts the registry-less array shape,
        # not the sandbox tool (which has its own coverage in test_artifacts.py).
        tools = TextHandlerMixin._build_tools_array(
            h, {"enable_web_search": True, "enable_mcp": False,
                "enable_code_interpreter": False}, "gpt-5.5")
        assert tools == [{"type": "web_search"}]

    def test_registry_builder_registers_history_react_and_search(self, monkeypatch):
        from slack_client.base import SlackBot
        monkeypatch.setattr(config, "enable_history_tools", True)
        monkeypatch.setattr(config, "enable_reactions", True)
        monkeypatch.setattr(config, "enable_react_tool", True)
        monkeypatch.setattr(config, "reaction_emojis", ["thumbsup"])
        monkeypatch.setattr(config, "enable_search_tool", True)
        monkeypatch.setattr(config, "enable_channel_memory", True)
        monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
        s = _slack_tool_mock(history=[
            {"type": "function", "name": "fetch_channel_history", "parameters": {}},
            {"type": "function", "name": "fetch_thread_messages", "parameters": {}},
        ])
        monkeypatch.setattr(config, "enable_read_document_tool", True)
        monkeypatch.setattr(config, "enable_people_tools", True)
        monkeypatch.setattr(config, "enable_deep_research", True)
        registry = SlackBot._build_tool_registry(s)
        names = {t["name"] for t in registry.schemas()}
        # Every gated-on tool is exposed. (Not an exact-set assertion: an always-on tool the
        # builder gains later — F34's generate_image was the first — is not a bug in the
        # wiring this test is about. The gates below are what it actually verifies.)
        assert {"fetch_channel_history", "fetch_thread_messages",
                "react_to_message", "search_slack", "post_to_thread",
                "remember_fact", "update_fact", "forget_fact",
                "read_document", "lookup_user", "list_channel_members",
                "start_background_job"} <= names
        # …and the per-request gates still hide what this request doesn't qualify for:
        # no_response_needed needs an unprompted turn (F2), and the image tools that depend on
        # turn state — a sandbox container / a non-empty image catalog — have neither here (F34).
        assert names.isdisjoint({"no_response_needed", "create_image_asset", "edit_image"})

    def test_registry_builder_search_gated_off(self, monkeypatch):
        from slack_client.base import SlackBot
        monkeypatch.setattr(config, "enable_history_tools", False)
        monkeypatch.setattr(config, "enable_reactions", False)
        monkeypatch.setattr(config, "enable_search_tool", False)
        s = _slack_tool_mock()
        registry = SlackBot._build_tool_registry(s)
        assert "search_slack" not in {t["name"] for t in registry.schemas()}

    @pytest.mark.asyncio
    async def test_registry_history_executor_routes_by_name(self, monkeypatch):
        from slack_client.base import SlackBot
        monkeypatch.setattr(config, "enable_reactions", False)
        s = _slack_tool_mock(history=[
            {"type": "function", "name": "fetch_channel_history", "parameters": {}},
            {"type": "function", "name": "fetch_thread_messages", "parameters": {}},
        ])
        s.dispatch_history_tool_call = AsyncMock(return_value={"ok": True})
        registry = SlackBot._build_tool_registry(s)
        ctx = ToolContext()
        await registry.dispatch(ctx, "fetch_thread_messages", '{"channel_id": "C1"}')
        # ctx rides along so omitted channel_id/thread_ts default to the current conversation
        s.dispatch_history_tool_call.assert_awaited_once_with(
            "fetch_thread_messages", {"channel_id": "C1"}, ctx)


class TestSegmentJoin:
    """The seam rule shared by the tool loop (returned aggregate) and the handler (Slack buffer):
    a paragraph break between round-segments, but only where the model didn't already leave one."""

    def test_jams_get_a_paragraph_break(self):
        from message_markers import segment_separator, join_segments
        assert segment_separator("under Super Heavy.", "Fixed.") == "\n\n"
        assert join_segments(["under Super Heavy.", "Fixed."]) == "under Super Heavy.\n\nFixed."

    def test_existing_whitespace_is_not_doubled(self):
        from message_markers import segment_separator
        assert segment_separator("done.\n", "next") == ""          # prior segment ends in space
        assert segment_separator("done.", "\nnext") == ""          # next begins with space
        assert segment_separator("done. ", "next") == ""

    def test_empty_segments_contribute_nothing(self):
        from message_markers import segment_separator, join_segments
        assert segment_separator("", "x") == "" and segment_separator("x", "") == ""
        assert join_segments(["only"]) == "only"
        assert join_segments(["a", "", "b"]) == "a\n\nb"           # the empty middle round vanishes
        assert join_segments([]) == ""
