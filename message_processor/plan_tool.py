"""`update_plan` — the model writes the plan its streamed reply shows (spec §F).

Agent surfaces §F (the Claude Tag look): on a chat turn that takes tool work, the model opens a
short plan — a title, a few steps, a one-line result per finished step — and keeps it true as it
goes. The stream renders it as Slack's plan display in place of the automatic tool cards (§C).

The tool is FREE bookkeeping (it produces nothing a user reads as an answer) and the plan is a
best-effort DISPLAY: a valid call always answers `{"ok": true}`, whether or not this turn has a
surface that can show it (rulings R1/R2). The executor holds no state of its own — the turn's
current attempt owns the plan and hands a sink to the call context (`turn.plan_sink`, R5), so two
concurrent turns can never write into each other's plan.

Exposed only while SLACK_STREAM_TASK_CARDS is on; with the flag off the tool and its prompt
paragraph do not exist, and every request is byte-identical.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from logger import setup_logger
from message_processor.tool_registry import ToolContext, ToolRegistry

logger = setup_logger(name="slack_bot.PlanTool")

UPDATE_PLAN = "update_plan"
PLAN_STEP_STATUSES = ("pending", "in_progress", "complete", "error")
PLAN_MAX_STEPS = 6
PLAN_TITLE_MAX = 80
PLAN_STEP_TITLE_MAX = 100
PLAN_RESULT_MAX = 120
_PLAN_ID_MAX = 40


def get_update_plan_schema() -> dict:
    """Strict (R7): every field required, `result` a required nullable string."""
    return {
        "type": "function",
        "name": UPDATE_PLAN,
        "description": (
            "Show the person what you are doing while you work: a short plan, displayed live "
            "above your reply. Each call REPLACES the whole plan. It posts nothing itself and "
            "costs no turn of its own — send it alongside the tool call it describes."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "What you are doing, in plain words (under 80 characters).",
                },
                "steps": {
                    "type": "array",
                    "description": "One to six steps, in order.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string",
                                "description": "A short id you choose; keep it stable across "
                                               "updates.",
                            },
                            "title": {
                                "type": "string",
                                "description": "The step, in plain words (under 100 "
                                               "characters).",
                            },
                            "status": {"type": "string", "enum": list(PLAN_STEP_STATUSES)},
                            "result": {
                                "type": ["string", "null"],
                                "description": "One plain line on what the step found or did, "
                                               "once it is done; null until then.",
                            },
                        },
                        "required": ["id", "title", "status", "result"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["title", "steps"],
            "additionalProperties": False,
        },
    }


def validate_plan(args: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """F's own schema, checked BEFORE anything is applied (R5). Returns (plan, None) or
    (None, why). Lengths are clipped later, at display; only shape is an error here."""
    if not isinstance(args, dict):
        return None, "arguments must be an object"
    title = args.get("title")
    if not isinstance(title, str) or not title.strip():
        return None, "title must be a non-empty string"
    steps = args.get("steps")
    if not isinstance(steps, list) or not steps:
        return None, "steps must be a non-empty list"
    if len(steps) > PLAN_MAX_STEPS:
        return None, f"at most {PLAN_MAX_STEPS} steps"
    clean: List[Dict[str, Any]] = []
    seen: set = set()
    for step in steps:
        if not isinstance(step, dict):
            return None, "each step must be an object"
        sid, stitle, status = step.get("id"), step.get("title"), step.get("status")
        result = step.get("result")
        if not isinstance(sid, str) or not sid.strip() or len(sid.strip()) > _PLAN_ID_MAX:
            return None, "each step needs a short non-empty id"
        if sid.strip() in seen:
            return None, f"duplicate step id {sid.strip()!r}"
        if not isinstance(stitle, str) or not stitle.strip():
            return None, "each step needs a non-empty title"
        if status not in PLAN_STEP_STATUSES:
            return None, f"status must be one of {', '.join(PLAN_STEP_STATUSES)}"
        if result is not None and not isinstance(result, str):
            return None, "result must be a string or null"
        seen.add(sid.strip())
        clean.append({"id": sid.strip(), "title": stitle, "status": status,
                      "result": result if isinstance(result, str) and result.strip() else None})
    return {"title": title, "steps": clean}, None


async def execute_update_plan(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Validate, hand the snapshot to the attempt's sink, answer `ok`. Never raises."""
    plan, why = validate_plan(args)
    if plan is None:
        return {"ok": False, "error": why}
    sink = getattr(getattr(ctx, "turn", None), "plan_sink", None)
    if callable(sink):
        try:
            await sink(plan)
        except Exception as e:  # noqa: BLE001 — a display never fails the call
            logger.debug(f"update_plan display skipped: {e}")
    return {"ok": True}


def register_plan_tool(registry: ToolRegistry) -> None:
    """Both surfaces, one static schema, no per-turn predicate (R1/R2: no cache fork)."""
    registry.register(get_update_plan_schema(), execute_update_plan)
