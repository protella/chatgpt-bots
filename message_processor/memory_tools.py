"""Model-invoked channel-memory tools (Phase C of the channel-teammate redesign).

Three thin wrappers over the existing ``channel_memory`` CRUD, exposed to the
function-call loop so the responding model decides in-flight what a colleague
would durably remember — replacing the post-response extraction pass (which
remains available behind ENABLE_MEMORY_EXTRACTION_FALLBACK for one release).

Rules enforced here, not by prompt:
- Channel surface only: memory is per-channel; DM calls are refused.
- Writes are always channel-scope, attributed to the triggering user.
- Workspace-scope rows are visible in context but read-only from a channel.
- The per-channel row cap (MEMORY_MAX_ROWS) is enforced on insert; at cap the
  model is told the oldest entries so it can update/forget instead.

Executors never raise: every failure is an ``{"ok": False, "error": ...}``
result (the registry would wrap an exception anyway, but clean errors give the
model something actionable).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from config import config
from message_processor.channel_steering import is_ordinary_fact
from tool_registry import ToolContext, ToolRegistry

# Keep stored facts to a concise sentence-or-two; hard cap guards the prompt.
MAX_FACT_CHARS = 500


def get_remember_fact_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "remember_fact",
        "description": (
            "Save a durable, channel-relevant BACKGROUND fact to this channel's long-term "
            "memory (decisions, conventions, recurring events, who owns what). "
            "Bias strongly against saving; most exchanges contain nothing durable. "
            "Update an existing [#id] fact instead of adding a near-duplicate. "
            "NOT for rules about your own behavior here — 'stay quiet unless tagged', 'keep "
            "answers short in this channel' and the like are the channel's standing policy: "
            "write those with set_channel_participation(standing_policy=...)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The fact, as one concise sentence.",
                },
                "scope": {
                    "type": "string",
                    "enum": ["channel"],
                    "description": "Memory scope (only 'channel' is allowed).",
                },
            },
            "required": ["content"],
        },
    }


def get_update_fact_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "update_fact",
        "description": (
            "Revise an existing channel-memory fact (shown in context as [#id]) when it "
            "changed or needs refinement. Prefer this over remember_fact for near-duplicates. "
            "It edits background facts only; to change a rule about how you behave here, "
            "replace the standing policy with set_channel_participation(standing_policy=...)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "The [#id] of the fact to revise."},
                "content": {"type": "string", "description": "The revised fact, one concise sentence."},
            },
            "required": ["id", "content"],
        },
    }


def get_forget_fact_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "forget_fact",
        "description": (
            "Delete a channel-memory fact (shown in context as [#id]) — when someone asks you "
            "to forget it, or it is obsolete/wrong. Background facts only; to drop a rule about "
            "how you behave here, replace the standing policy with "
            "set_channel_participation(standing_policy=...)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "The [#id] of the fact to delete."},
            },
            "required": ["id"],
        },
    }


def _channel_only_guard(ctx: ToolContext) -> Optional[Dict[str, Any]]:
    """Common preconditions; returns an error result or None to proceed."""
    if ctx.is_dm:
        return {"ok": False, "error": "memory_is_channel_only",
                "message": "Channel memory is not available in DMs."}
    if not ctx.channel_id:
        return {"ok": False, "error": "no_channel",
                "message": "No channel in this context."}
    if ctx.db is None:
        return {"ok": False, "error": "memory_unavailable",
                "message": "Memory storage is not available."}
    return None


async def _visible_row(ctx: ToolContext, memory_id: Any) -> Dict[str, Any]:
    """Resolve an id against the rows this channel can see.

    Returns {"row": ...} on success or an {"ok": False, ...} error result.
    Workspace-scope rows are readable context but not writable from a channel.
    """
    try:
        memory_id = int(memory_id)
    except (TypeError, ValueError):
        return {"ok": False, "error": "bad_arguments", "message": "id must be an integer."}
    rows = await ctx.db.get_channel_memory_async(ctx.channel_id)
    row = next((r for r in rows if r.get("id") == memory_id), None)
    if row is None:
        return {"ok": False, "error": "not_found",
                "message": f"No memory [#{memory_id}] in this channel."}
    if row.get("scope") != "channel":
        return {"ok": False, "error": "workspace_scope_readonly",
                "message": f"Memory [#{memory_id}] is workspace-shared and can't be changed from here."}
    if not is_ordinary_fact(row):
        # A recorded participation preference. Its id is visible to the participation gate (whose
        # backoff contract targets it) and therefore visible here too, but it is steering, not a
        # fact — the generic tools may read it and nothing more. The reserved policy row never
        # reaches this code at all: get_channel_memory_async does not return it.
        return {"ok": False, "error": "steering_row_readonly",
                "message": (f"[#{memory_id}] is a recorded participation preference, not a fact. "
                            "Change how you behave here with set_channel_participation instead.")}
    return {"row": row}


async def execute_remember_fact(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    err = _channel_only_guard(ctx)
    if err:
        return err
    content = (args.get("content") or "").strip()
    if not content:
        return {"ok": False, "error": "bad_arguments", "message": "content is required."}
    content = content[:MAX_FACT_CHARS]

    rows = await ctx.db.get_channel_memory_async(ctx.channel_id)
    # Steering rows never consume fact capacity: the cap exists to keep remembered facts from
    # crowding the prompt, and a channel whose preferences filled it could store nothing at all.
    chan_rows = [r for r in rows if r.get("scope") == "channel" and is_ordinary_fact(r)]
    cap = max(1, config.memory_max_rows)
    if len(chan_rows) >= cap:
        # rows arrive ordered updated_ts ASC, so the head is the stalest.
        oldest = [{"id": r["id"], "content": r["content"]} for r in chan_rows[:3]]
        return {"ok": False, "error": "memory_full",
                "hint": "forget or update something",
                "oldest": oldest}

    new_id = await ctx.db.add_channel_memory_async(
        ctx.channel_id, content, scope="channel", author=ctx.user_id
    )
    return {"ok": True, "id": new_id, "content": content}


async def execute_update_fact(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    err = _channel_only_guard(ctx)
    if err:
        return err
    content = (args.get("content") or "").strip()
    if not content:
        return {"ok": False, "error": "bad_arguments", "message": "content is required."}
    resolved = await _visible_row(ctx, args.get("id"))
    if "row" not in resolved:
        return resolved
    row = resolved["row"]
    # The scope-constrained update, even though _visible_row already refused everything but an
    # ordinary channel fact: this is a model-driven write, and the storage should be the last word
    # on what it can reach rather than a check three frames up.
    await ctx.db.update_channel_fact_async(row["id"], content[:MAX_FACT_CHARS])
    return {"ok": True, "id": row["id"], "content": content[:MAX_FACT_CHARS]}


async def execute_forget_fact(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    err = _channel_only_guard(ctx)
    if err:
        return err
    resolved = await _visible_row(ctx, args.get("id"))
    if "row" not in resolved:
        return resolved
    row = resolved["row"]
    await ctx.db.delete_channel_memory_async(row["id"])
    return {"ok": True, "id": row["id"], "forgot": row.get("content")}


def register_memory_tools(registry: ToolRegistry) -> None:
    """Register the three memory tools (call only when ENABLE_CHANNEL_MEMORY is on)."""
    registry.register(get_remember_fact_schema(), execute_remember_fact)
    registry.register(get_update_fact_schema(), execute_update_fact)
    registry.register(get_forget_fact_schema(), execute_forget_fact)
