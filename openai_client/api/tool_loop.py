"""Local function-call loop for the Responses API (redesign Phase A — the keystone).

Wraps the existing ``create_text_response_with_tools`` / ``create_streaming_response_with_tools``
calls in a loop: collect ``function_call`` items → dispatch through the ToolRegistry
(parallel, timeout-guarded) → append ``function_call`` + ``function_call_output`` items to the
input → re-invoke. Local tools compose with server-side tools (web_search, MCP) in the same
``tools`` array.

Caps: ``MAX_TOOL_ROUNDS`` rounds / ``MAX_TOOL_CALLS_PER_TURN`` total calls. On cap, one final
round runs with ``tool_choice="none"`` so the model must answer with what it has.

Streaming: intermediate rounds stream through the same callback, but their text deltas are
suppressed inside ``create_streaming_response_with_tools`` once a function_call appears in the
round (pre-tool preamble); ``tool_callback(f"local:{name}", ...)`` drives the status line while
tools run, and only the final round's text reaches the user.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from config import config
from tool_registry import ToolContext, ToolRegistry, serialize_tool_result

from . import responses as responses_api


def _call_ok(result: Any) -> bool:
    """A tool result counts as successful unless it explicitly says ok=False."""
    return not (isinstance(result, dict) and result.get("ok") is False)


def _function_calls(sink: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The dispatchable function_call entries of a round's sink (reasoning items excluded)."""
    return [e for e in sink if e.get("type", "function_call") == "function_call"]


async def _run_tool_round(
    self,
    registry: ToolRegistry,
    tool_context: ToolContext,
    sink: List[Dict[str, Any]],
    input_items: List[Dict[str, Any]],
    local_tool_calls: List[Dict[str, Any]],
    tool_callback: Optional[Callable[[str, str], Any]] = None,
) -> None:
    """Dispatch one round's calls, then replay the round's items (reasoning items in
    place, each function_call followed by its function_call_output) onto the input."""

    async def _notify(tool_id: str, status: str) -> None:
        if not tool_callback:
            return
        try:
            result = tool_callback(tool_id, status)
            if result is not None and hasattr(result, "__await__"):
                await result
        except Exception as e:  # noqa: BLE001 — status UI must never break the loop
            self.log_warning(f"Tool callback error for {tool_id}: {e}")

    calls = _function_calls(sink)
    for call in calls:
        await _notify(f"local:{call.get('name')}", "started")

    results = await registry.dispatch_all(tool_context, calls)
    result_by_id = {}
    for call, result in zip(calls, results):
        ok = _call_ok(result)
        local_tool_calls.append({"name": call.get("name"), "ok": ok})
        self.log_info(f"Local tool '{call.get('name')}' -> {'ok' if ok else 'error'}")
        result_by_id[id(call)] = result
        await _notify(f"local:{call.get('name')}", "completed")

    # Replay in encounter order — reasoning models require their reasoning items to
    # precede the paired function_call when the conversation is replayed statelessly.
    for entry in sink:
        if entry.get("type") == "reasoning":
            if entry.get("item"):
                input_items.append(entry["item"])
            continue
        input_items.append({
            "type": "function_call",
            "call_id": entry.get("call_id"),
            "name": entry.get("name"),
            "arguments": entry.get("arguments") or "{}",
        })
        input_items.append({
            "type": "function_call_output",
            "call_id": entry.get("call_id"),
            "output": serialize_tool_result(result_by_id.get(id(entry))),
        })


def _merge_used(tools_used_all: List[str], round_used: List[str]) -> None:
    for name in round_used:
        if name not in tools_used_all:
            tools_used_all.append(name)


async def create_text_response_with_tool_loop(
    self,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    registry: ToolRegistry,
    tool_context: ToolContext,
    **params: Any,
) -> Dict[str, Any]:
    """Non-streaming response with local tool execution.

    Returns {"text", "tools_used", "local_tool_calls"} — ``local_tool_calls`` is the
    ordered [{"name", "ok"}] record of every local call (e.g. for reaction-only detection).
    """
    input_items: List[Dict[str, Any]] = list(messages)
    tools_used_all: List[str] = []
    local_tool_calls: List[Dict[str, Any]] = []
    rounds = 0
    total_calls = 0
    tool_choice: Optional[str] = None

    while True:
        sink: List[Dict[str, Any]] = []
        result = await responses_api.create_text_response_with_tools(
            self,
            messages=input_items,
            tools=tools,
            return_metadata=True,
            function_call_sink=sink,
            tool_choice=tool_choice,
            **params,
        )
        _merge_used(tools_used_all, result.get("tools_used") or [])

        calls = _function_calls(sink)
        if not calls or tool_choice == "none":
            return {
                "text": result.get("text", ""),
                "tools_used": tools_used_all,
                "local_tool_calls": local_tool_calls,
            }

        rounds += 1
        total_calls += len(calls)
        await _run_tool_round(self, registry, tool_context, sink, input_items, local_tool_calls)
        _merge_used(tools_used_all, [c.get("name") for c in calls if c.get("name")])

        if rounds >= config.max_tool_rounds or total_calls >= config.max_tool_calls_per_turn:
            self.log_warning(
                f"Tool loop cap hit ({rounds} rounds / {total_calls} calls) — forcing final answer"
            )
            tool_choice = "none"


async def create_streaming_response_with_tool_loop(
    self,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    registry: ToolRegistry,
    tool_context: ToolContext,
    stream_callback: Callable[[Optional[str]], Any],
    tool_callback: Optional[Callable[[str, str], Any]] = None,
    **params: Any,
) -> Dict[str, Any]:
    """Streaming response with local tool execution.

    Returns {"text", "tools_used", "local_tool_calls"}. Intermediate (tool) rounds don't
    stream text to the user; the final round streams normally and fires the completion flush.
    """
    input_items: List[Dict[str, Any]] = list(messages)
    tools_used_all: List[str] = []
    local_tool_calls: List[Dict[str, Any]] = []
    rounds = 0
    total_calls = 0
    tool_choice: Optional[str] = None

    while True:
        sink: List[Dict[str, Any]] = []
        text = await responses_api.create_streaming_response_with_tools(
            self,
            messages=input_items,
            tools=tools,
            stream_callback=stream_callback,
            tool_callback=tool_callback,
            function_call_sink=sink,
            tool_choice=tool_choice,
            **params,
        )

        calls = _function_calls(sink)
        if not calls or tool_choice == "none":
            return {
                "text": text,
                "tools_used": tools_used_all,
                "local_tool_calls": local_tool_calls,
            }

        rounds += 1
        total_calls += len(calls)
        await _run_tool_round(
            self, registry, tool_context, sink, input_items, local_tool_calls, tool_callback
        )
        _merge_used(tools_used_all, [c.get("name") for c in calls if c.get("name")])

        if rounds >= config.max_tool_rounds or total_calls >= config.max_tool_calls_per_turn:
            self.log_warning(
                f"Tool loop cap hit ({rounds} rounds / {total_calls} calls) — forcing final answer"
            )
            tool_choice = "none"
