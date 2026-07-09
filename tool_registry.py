"""Platform-agnostic local-tool registry for the Responses API function-call loop (Phase A).

The registry maps function-tool schemas to async executors. The loop
(``openai_client/api/tool_loop.py``) collects ``function_call`` items from a response,
dispatches them here, and feeds ``function_call_output`` items back to the model.

Executors receive a ``ToolContext`` (per-request platform state) and the parsed
arguments dict, and return a JSON-serializable dict. They must never raise to the
loop: dispatch wraps every failure (unknown tool, bad args, timeout, exception) into
an ``{"ok": False, "error": ...}`` result so a tool problem degrades the answer, not
the response.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional

from config import config


@dataclass
class ToolContext:
    """Per-request state passed to every executor (built by the message processor)."""
    channel_id: Optional[str] = None
    thread_ts: Optional[str] = None
    trigger_ts: Optional[str] = None      # ts of the message we're answering
    action_token: Optional[str] = None    # from the triggering Slack event (search API)
    client: Any = None                    # platform client (e.g. SlackBot)
    db: Any = None
    is_dm: bool = False


Executor = Callable[[ToolContext, Dict[str, Any]], Awaitable[Dict[str, Any]]]


class ToolRegistry:
    """Name → (schema, executor, enabled-gate). Gates are evaluated per request."""

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        schema: Dict[str, Any],
        executor: Executor,
        enabled: Optional[Callable[[dict], bool]] = None,
    ) -> None:
        name = schema.get("name")
        if not name:
            raise ValueError("Tool schema must have a 'name'")
        self._tools[name] = {"schema": schema, "executor": executor, "enabled": enabled}

    def schemas(self, thread_config: Optional[dict] = None) -> List[Dict[str, Any]]:
        """Schemas of the tools enabled for this request (a failing gate hides the tool)."""
        out = []
        for tool in self._tools.values():
            gate = tool["enabled"]
            try:
                if gate is None or gate(thread_config or {}):
                    out.append(tool["schema"])
            except Exception:
                continue
        return out

    def has_tools(self, thread_config: Optional[dict] = None) -> bool:
        return bool(self.schemas(thread_config))

    async def dispatch(self, ctx: ToolContext, name: str, arguments: Any) -> Dict[str, Any]:
        """Run one tool call. Never raises."""
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": "unknown_tool", "message": f"No tool named '{name}'."}

        if isinstance(arguments, str):
            try:
                args = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                return {"ok": False, "error": "bad_arguments", "message": "Arguments were not valid JSON."}
        else:
            args = arguments or {}
        if not isinstance(args, dict):
            return {"ok": False, "error": "bad_arguments", "message": "Arguments must be a JSON object."}

        try:
            return await asyncio.wait_for(
                tool["executor"](ctx, args), timeout=config.tool_call_timeout
            )
        except asyncio.TimeoutError:
            return {"ok": False, "error": "timeout",
                    "message": f"Tool '{name}' timed out after {config.tool_call_timeout:.0f}s."}
        except Exception as e:  # noqa: BLE001 — a tool bug must not kill the response
            return {"ok": False, "error": "execution_error", "message": str(e)[:500]}

    async def dispatch_all(self, ctx: ToolContext, calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run a round's calls in parallel; result order matches ``calls``."""
        return list(await asyncio.gather(
            *(self.dispatch(ctx, c.get("name", ""), c.get("arguments")) for c in calls)
        ))


def serialize_tool_result(result: Any) -> str:
    """JSON-encode an executor result, truncated to TOOL_RESULT_MAX_CHARS for the model."""
    try:
        s = json.dumps(result, ensure_ascii=False, default=str)
    except Exception:
        s = str(result)
    cap = config.tool_result_max_chars
    if len(s) > cap:
        s = s[:cap] + " …[truncated]"
    return s
