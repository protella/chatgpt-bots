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
    user_id: Optional[str] = None         # triggering user (provenance for memory writes)
    client: Any = None                    # platform client (e.g. SlackBot)
    db: Any = None
    is_dm: bool = False
    # F30: exposed by the tool loop so a detached job (start_deep_research) can snapshot the
    # CURRENT turn's full conversation by deep-copying `current_input` at call time. The
    # developer prompt rides separately in `system_prompt`; `model` is the thread's model.
    processor: Any = None                 # MessageProcessor (openai_client, scheduling, thread_manager)
    current_input: Optional[List[Any]] = None
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    # F30.1: set True by execute_start_background_job on a successful start, so the turn's
    # finalizer (handlers/text.py) can DROP the model's ack reply — the research status card
    # the job posts is the acknowledgment. Read back from the same context the loop shares.
    background_job_started: bool = False
    # F34: image tools. `thread_config` carries the resolved per-user settings (the image
    # MODEL is read from here and is NOT model-selectable). `container_id` is the SAME
    # persistent code-interpreter container already placed in the tools array — the image
    # asset tool must never resolve its own, or it would mount bytes into a container the
    # model cannot see. `image_catalog` is this turn's allowlist of editable images, so an
    # invented image id is rejected rather than silently editing the wrong picture.
    thread_config: Optional[Dict[str, Any]] = None
    container_id: Optional[str] = None
    image_catalog: Optional[List[Dict[str, Any]]] = None
    # Set True by a detached image generation, so the finalizer can drop the model's ack
    # reply the same way deep research does — the posted image IS the acknowledgment.
    image_generation_started: bool = False
    # Paths mounted into the container by create_image_asset this turn. If the turn ends
    # having published nothing, these are rescued to the thread rather than vanishing with
    # the container (see handlers/text.py) — a silent no-output turn is the worst failure.
    sandbox_image_assets: Optional[List[Dict[str, Any]]] = None
    # F35: mount_file. `thread_files` is this turn's allowlist of mountable files (images AND
    # documents behind one opaque id space) — the same authorization rule as `image_catalog`:
    # only ids we advertised resolve. `mounted_files` records what actually went into the
    # sandbox, so (a) a second mount of the same file is a no-op, and (b) the artifact
    # publisher can refuse to post a user's own file back at them, even byte-copied.
    thread_files: Optional[List[Dict[str, Any]]] = None
    mounted_files: Optional[List[Dict[str, Any]]] = None
    # Participation redesign (BLOCKER #3): True only when a HUMAN directly addressed the bot for a
    # structural change — a real <@bot> mention, OR a current message the participation classifier
    # judged an explicit structural request (handlers.text `_structural_change_authorized`, from
    # message.metadata sender_type/mentioned_self/gate_authorized_structural — NOT the loose
    # name-hit regex). It gates the structural set_channel_participation tool; the canvas-delete
    # tool has its own parallel strict signal, `_canvas_delete_authorized`. Left False
    # (fail-closed), so an unaddressed channel turn, a quoted/ambient name-drop,
    # or a non-human sender — the injection / hallucination / "being talked about ≠ talked to"
    # vector — can never flip channel settings even if the model emits the call.
    structural_change_authorized: bool = False
    # F38: the turn's presentation + work-claim state (message_processor.turn_runtime).
    # A slow local tool calls `await ctx.turn.claim_work(ctx.client, ctx.message)` once its
    # arguments and capacity checks have PASSED and immediately before the slow part starts —
    # never on entry, or a rejected call would flash a 👀 it is about to retract. A tool that
    # posts its own surface (a background job's card, a detached image) sets
    # `ctx.turn.visible_action_committed = True` so the turn counts as having produced output
    # even though its Response carries no text.
    turn: Any = None
    message: Any = None


Executor = Callable[[ToolContext, Dict[str, Any]], Awaitable[Dict[str, Any]]]


class ToolRegistry:
    """Name → (schema, executor, enabled-gate). Gates are evaluated per request."""

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        schema: Any,
        executor: Executor,
        enabled: Optional[Callable[[dict], bool]] = None,
        timeout: Optional[float] = None,
        name: Optional[str] = None,
    ) -> None:
        """Register a tool.

        ``schema`` is either a static dict or a FACTORY ``(thread_config) -> dict``, for a
        tool whose shape depends on the request (F34: the image tools' legal option values
        differ by the user's selected image model, and their description names the user's
        saved defaults). A factory must be given an explicit ``name``, since there is no
        dict to read it from.
        """
        if callable(schema):
            if not name:
                raise ValueError("A schema factory must be registered with an explicit name")
        else:
            name = schema.get("name")
            if not name:
                raise ValueError("Tool schema must have a 'name'")
        # timeout=None → the shared config.tool_call_timeout. A tool with a heavier
        # worst case (e.g. read_document, which may download + render + OCR a scan)
        # sets its own longer bound so the generic 20s cap can't abort it.
        self._tools[name] = {"schema": schema, "executor": executor,
                             "enabled": enabled, "timeout": timeout}

    def schemas(self, thread_config: Optional[dict] = None) -> List[Dict[str, Any]]:
        """Schemas of the tools enabled for this request (a failing gate hides the tool).

        A schema factory that raises hides its tool rather than failing the turn — the same
        fail-closed rule the enable-gate already follows.
        """
        out = []
        cfg = thread_config or {}
        for tool in self._tools.values():
            gate = tool["enabled"]
            try:
                if gate is not None and not gate(cfg):
                    continue
                schema = tool["schema"]
                if callable(schema):
                    schema = schema(cfg)
                    if not schema:
                        continue
                out.append(schema)
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

        timeout = tool.get("timeout")
        if timeout is None:
            timeout = config.tool_call_timeout
        try:
            return await asyncio.wait_for(tool["executor"](ctx, args), timeout=timeout)
        except asyncio.TimeoutError:
            return {"ok": False, "error": "timeout",
                    "message": f"Tool '{name}' timed out after {timeout:.0f}s."}
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
