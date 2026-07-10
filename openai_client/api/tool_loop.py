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


# --- F2: no_response_needed terminal action ---

_NO_REPLY_TOOL = "no_response_needed"
_REACT_TOOL = "react_to_message"


def _no_reply_call(calls: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return next((c for c in calls if c.get("name") == _NO_REPLY_TOOL), None)


def _sanitize_reason(call: Dict[str, Any]) -> str:
    """Extract + sanitize the no_response_needed reason (control-stripped, length-capped)."""
    import json
    args = call.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args or "{}")
        except json.JSONDecodeError:
            args = {}
    reason = (args or {}).get("reason", "") if isinstance(args, dict) else ""
    reason = "".join(ch if ch.isprintable() else " " for ch in str(reason)).strip()
    return reason[:300]


async def _handle_no_reply_terminal(
    self,
    registry: ToolRegistry,
    tool_context: ToolContext,
    calls: List[Dict[str, Any]],
    terminal_call: Dict[str, Any],
    tools_used_all: List[str],
    local_tool_calls: List[Dict[str, Any]],
    remaining_budget: Optional[int] = None,
) -> Dict[str, Any]:
    """Terminal round: no_response_needed ends the turn. Only sibling react_to_message
    calls execute (filtered BEFORE dispatch — dispatch_all runs a round concurrently);
    other side-effect calls are suppressed with a logged skip. Returns a no_reply outcome
    with the (sanitized) reason; nothing is posted.

    F6 fix (b): react siblings still count against MAX_TOOL_CALLS_PER_TURN — the terminal
    branch runs before the loop's own cap check, so apply the remaining global budget to
    the react calls here."""
    react_calls = [c for c in calls if c.get("name") == _REACT_TOOL]
    budget = len(react_calls) if remaining_budget is None else max(0, int(remaining_budget))
    allowed_react_ids = {id(c) for c in react_calls[:budget]}
    exec_calls = [c for c in calls
                  if c.get("name") == _NO_REPLY_TOOL or id(c) in allowed_react_ids]
    exec_ids = {id(c) for c in exec_calls}
    skipped = [c.get("name") for c in calls if id(c) not in exec_ids]
    if skipped:
        self.log_info(f"{_NO_REPLY_TOOL} terminal — suppressing sibling calls: {skipped}")
    results = await registry.dispatch_all(tool_context, exec_calls)
    for call, result in zip(exec_calls, results):
        if call.get("name") == _REACT_TOOL:
            local_tool_calls.append({"name": call.get("name"), "ok": _call_ok(result)})
    _merge_used(tools_used_all, [c.get("name") for c in exec_calls if c.get("name")])
    reason = _sanitize_reason(terminal_call)
    self.log_info(f"{_NO_REPLY_TOOL}: ending turn without posting — reason: {reason!r}")
    return {
        "text": "",
        "tools_used": tools_used_all,
        "local_tool_calls": local_tool_calls,
        "terminal_action": "no_reply",
        "reason": reason,
    }


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

        terminal_call = _no_reply_call(calls)
        if terminal_call is not None:
            return await _handle_no_reply_terminal(
                self, registry, tool_context, calls, terminal_call,
                tools_used_all, local_tool_calls,
                remaining_budget=config.max_tool_calls_per_turn - total_calls)

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

        terminal_call = _no_reply_call(calls)
        if terminal_call is not None:
            return await _handle_no_reply_terminal(
                self, registry, tool_context, calls, terminal_call,
                tools_used_all, local_tool_calls,
                remaining_budget=config.max_tool_calls_per_turn - total_calls)

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
