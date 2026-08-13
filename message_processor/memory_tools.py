"""Model-invoked memory tools (Phase C of the channel-teammate redesign; user store from T2).

Four thin wrappers over the durable memory CRUD, exposed to the function-call loop so the
responding model decides in-flight what a colleague would durably remember — replacing the
post-response extraction pass (which remains available behind
ENABLE_MEMORY_EXTRACTION_FALLBACK for one release).

TWO STORES, ONE SET OF TOOLS. The surface decides which store a call reaches: a channel turn
reads and writes ``channel_memory`` for that channel, a DM turn reads and writes ``user_memory``
for the person being DMed. The model is not asked to choose — there is nothing to choose, since
neither store is reachable from the other's surface. The two surfaces carry DIFFERENT SCHEMA TEXT
for the same tool names (the channel wording says "this channel", the DM wording says "this
person"), and different flags: ENABLE_CHANNEL_MEMORY governs the channel surface,
ENABLE_USER_MEMORY the DM one. Either off disables exactly its own surface.

WHY DM FACTS NEVER APPEAR IN A CHANNEL (owner ruling, not a tunable): a personal memory is
written in a private conversation, and surfacing it in a room would publish something the person
told the bot alone. The store split is what makes that structural — the channel path has no
`user_id` to read by, and `channel_steering.load_snapshot` refuses to fetch user rows off a DM.

Rules enforced here, not by prompt:
- Writes are attributed to the triggering user, and channel writes are always channel-scope.
- Workspace-scope rows are visible in context but read-only from a channel.
- The row cap (MEMORY_MAX_ROWS) is enforced on insert for BOTH stores; at cap the model is told
  the oldest entries so it can update/forget instead.

Executors never raise: every failure is an ``{"ok": False, "error": ...}`` result (the registry
would wrap an exception anyway, but clean errors give the model something actionable).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from config import config
from message_processor.channel_steering import is_ordinary_fact
from tool_registry import ToolContext, ToolRegistry

# Keep stored facts to a concise sentence-or-two; hard cap guards the prompt.
MAX_FACT_CHARS = 500


def _text_arg(value: Any) -> Optional[str]:
    """A stripped string argument, or None when the model sent something that is not a string.

    The registry guarantees the ARGUMENTS are a dict and nothing about the values inside it, so
    `content` can arrive as a number, a list or a dict. Calling `.strip()` on one of those raises
    out of the executor, and an executor that raises breaks the never-raise contract the tool loop
    depends on. Callers fold None into their existing bad_arguments result.
    """
    if not isinstance(value, str):
        return None
    return value.strip()


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


def get_list_facts_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "list_facts",
        "description": (
            "List what you have remembered about THIS CHANNEL, with the same [#id] numbers "
            "update_fact and forget_fact take. Your context already shows the current facts, so "
            "call this only when you need to look one up rather than act on what you were "
            "shown — someone asks what you remember, or you are about to revise or delete a "
            "fact whose id you have not actually seen this turn. Never guess an id."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Optional case-insensitive substring; only facts containing it are "
                        "returned. Omit to list everything."
                    ),
                },
            },
            "required": [],
        },
    }


# --- the DM surface's schemas: same tool names, personal-memory wording ---------------------
#
# Separate functions rather than a shared template with the noun swapped: these are the words a
# model reads to decide whether to write, and the two surfaces genuinely want different judgment
# — a channel fact is about a shared room, a personal fact is about one colleague.

def get_remember_fact_dm_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "remember_fact",
        "description": (
            "Save a durable BACKGROUND fact about the person you are talking with, to your "
            "long-term memory of THEM (how they work, what they own, standing preferences, "
            "context that will still matter weeks from now). It is personal and private to this "
            "DM: it is never shown in channels. "
            "Bias strongly against saving; most exchanges contain nothing durable, and a "
            "passing detail saved forever is worse than one forgotten. "
            "Update an existing [#id] fact instead of adding a near-duplicate."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The fact, as one concise sentence.",
                },
            },
            "required": ["content"],
        },
    }


def get_update_fact_dm_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "update_fact",
        "description": (
            "Revise something you remember about this person (shown in context as [#id]) when "
            "it changed or needs refinement. Prefer this over remember_fact for near-duplicates."
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


def get_forget_fact_dm_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "forget_fact",
        "description": (
            "Delete something you remember about this person (shown in context as [#id]) — when "
            "they ask you to forget it, or it is obsolete or wrong. Asking to be forgotten is "
            "always honored; never argue for keeping it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "integer", "description": "The [#id] of the fact to delete."},
            },
            "required": ["id"],
        },
    }


def get_list_facts_dm_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "list_facts",
        "description": (
            "List what you remember about this person, with the same [#id] numbers update_fact "
            "and forget_fact take. Your context already shows them, so call this only when you "
            "need to look one up rather than act on what you were shown — they ask what you "
            "remember about them, or you are about to revise or delete a fact whose id you have "
            "not actually seen this turn. Never guess an id."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Optional case-insensitive substring; only facts containing it are "
                        "returned. Omit to list everything."
                    ),
                },
            },
            "required": [],
        },
    }


def _store_guard(ctx: ToolContext) -> Optional[Dict[str, Any]]:
    """Common preconditions for either store; an error result, or None to proceed.

    The flag checks are defence in depth, not the gate: registration hides each surface's schemas
    behind its own flag, so a model on a disabled surface never sees these tools. A replayed or
    hand-built call could still name one, and it should be refused rather than served from a store
    the operator turned off.
    """
    if ctx.db is None:
        return {"ok": False, "error": "memory_unavailable",
                "message": "Memory storage is not available."}
    if ctx.is_dm:
        if not config.enable_user_memory:
            return {"ok": False, "error": "memory_disabled",
                    "message": "Personal memory is turned off."}
        if not ctx.user_id:
            return {"ok": False, "error": "no_user",
                    "message": "No user in this context."}
        return None
    if not config.enable_channel_memory:
        return {"ok": False, "error": "memory_disabled",
                "message": "Channel memory is turned off."}
    if not ctx.channel_id:
        return {"ok": False, "error": "no_channel",
                "message": "No channel in this context."}
    return None


async def _visible_row(ctx: ToolContext, memory_id: Any) -> Dict[str, Any]:
    """Resolve an id against the rows THIS SURFACE can see.

    Returns {"row": ...} on success or an {"ok": False, ...} error result. In a DM that is the
    requester's own user rows and nothing else — another person's row simply is not found, which
    is the same answer an invented id gets and the only answer a caller should be able to tell
    apart. In a channel, workspace-scope rows are readable context but not writable from here.
    """
    try:
        memory_id = int(memory_id)
    except (TypeError, ValueError):
        return {"ok": False, "error": "bad_arguments", "message": "id must be an integer."}
    if ctx.is_dm:
        rows = await ctx.db.get_user_memory_async(ctx.user_id)
        row = next((r for r in rows if r.get("id") == memory_id), None)
        if row is None:
            return {"ok": False, "error": "not_found",
                    "message": f"You have no memory [#{memory_id}] about this person."}
        return {"row": row}
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


async def _facts_for_surface(ctx: ToolContext) -> List[Dict[str, Any]]:
    """The rows this surface's `[#id]`s name, in the order the prompt renders them (id-sorted).

    Channel: exactly what `render_snapshot` puts in the prompt — ordinary facts, channel and
    workspace scope. Anything the renderer skips is skipped here too, so a listed id is always an
    id the model could also have read off its own context.
    """
    if ctx.is_dm:
        rows = await ctx.db.get_user_memory_async(ctx.user_id) or []
    else:
        rows = [r for r in (await ctx.db.get_channel_memory_async(ctx.channel_id) or [])
                if is_ordinary_fact(r) and (r.get("content") or "").strip()]
    return sorted(rows, key=lambda r: r.get("id") or 0)


async def execute_remember_fact(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    err = _store_guard(ctx)
    if err:
        return err
    content = _text_arg(args.get("content"))
    if not content:
        return {"ok": False, "error": "bad_arguments",
                "message": "content is required, as a string."}
    content = content[:MAX_FACT_CHARS]
    cap = max(1, config.memory_max_rows)

    if ctx.is_dm:
        rows = await ctx.db.get_user_memory_async(ctx.user_id) or []
        if len(rows) >= cap:
            # rows arrive ordered updated_ts ASC, so the head is the stalest.
            oldest = [{"id": r["id"], "content": r["content"]} for r in rows[:3]]
            return {"ok": False, "error": "memory_full",
                    "hint": "forget or update something",
                    "oldest": oldest}
        new_id = await ctx.db.add_user_memory_async(ctx.user_id, content, author=ctx.user_id)
        return {"ok": True, "id": new_id, "content": content}

    rows = await ctx.db.get_channel_memory_async(ctx.channel_id)
    # Steering rows never consume fact capacity: the cap exists to keep remembered facts from
    # crowding the prompt, and a channel whose preferences filled it could store nothing at all.
    chan_rows = [r for r in rows if r.get("scope") == "channel" and is_ordinary_fact(r)]
    if len(chan_rows) >= cap:
        oldest = [{"id": r["id"], "content": r["content"]} for r in chan_rows[:3]]
        return {"ok": False, "error": "memory_full",
                "hint": "forget or update something",
                "oldest": oldest}

    new_id = await ctx.db.add_channel_memory_async(
        ctx.channel_id, content, scope="channel", author=ctx.user_id
    )
    return {"ok": True, "id": new_id, "content": content}


async def execute_update_fact(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    err = _store_guard(ctx)
    if err:
        return err
    content = _text_arg(args.get("content"))
    if not content:
        return {"ok": False, "error": "bad_arguments",
                "message": "content is required, as a string."}
    resolved = await _visible_row(ctx, args.get("id"))
    if "row" not in resolved:
        return resolved
    row = resolved["row"]
    content = content[:MAX_FACT_CHARS]
    # The store-constrained update, even though _visible_row already resolved the row against
    # exactly what this surface may reach: this is a model-driven write, and the storage should be
    # the last word on what it can touch rather than a check three frames up.
    if ctx.is_dm:
        await ctx.db.update_user_fact_async(ctx.user_id, row["id"], content)
    else:
        await ctx.db.update_channel_fact_async(row["id"], content)
    return {"ok": True, "id": row["id"], "content": content}


async def execute_forget_fact(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    err = _store_guard(ctx)
    if err:
        return err
    resolved = await _visible_row(ctx, args.get("id"))
    if "row" not in resolved:
        return resolved
    row = resolved["row"]
    if ctx.is_dm:
        await ctx.db.delete_user_memory_async(ctx.user_id, row["id"])
    else:
        await ctx.db.delete_channel_memory_async(row["id"])
    return {"ok": True, "id": row["id"], "forgot": row.get("content")}


async def execute_list_facts(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Read back this surface's facts with the ids the prompt shows.

    The blind-edit fix: update_fact and forget_fact take an id, and before this tool existed the
    only ids the model had ever seen were the ones this turn's snapshot happened to render.
    """
    err = _store_guard(ctx)
    if err:
        return err
    # An absent query means "show me everything"; a query that is PRESENT but not a string is the
    # same model error the write tools refuse, and silently filtering nothing would answer it with
    # the whole store as if the filter had been honoured.
    raw_query = args.get("query")
    if raw_query is not None and _text_arg(raw_query) is None:
        return {"ok": False, "error": "bad_arguments", "message": "query must be a string."}
    query = (_text_arg(raw_query) or "").lower()
    rows = await _facts_for_surface(ctx)
    facts: List[Dict[str, Any]] = []
    for row in rows:
        content = (row.get("content") or "").strip()
        if query and query not in content.lower():
            continue
        fact: Dict[str, Any] = {
            "id": row.get("id"),
            "content": content,
            "author": row.get("author"),
            "updated": row.get("updated_ts"),
        }
        if not ctx.is_dm and row.get("scope") != "channel":
            # Rendered to the model as a workspace fact and refused by update/forget; say so here
            # too, so the model does not spend a call discovering it.
            fact["read_only"] = True
        facts.append(fact)
    return {"ok": True, "scope": "user" if ctx.is_dm else "channel",
            "count": len(facts), "facts": facts}


def register_memory_tools(registry: ToolRegistry) -> None:
    """Register the memory tools on BOTH surfaces, each behind its own flag and schema text.

    Registered unconditionally by the client: the flags no longer decide whether the tools exist,
    they decide which SURFACE exposes them. `enabled` is the DM surface's gate (ENABLE_USER_MEMORY)
    and `channel_enabled` the channel one (ENABLE_CHANNEL_MEMORY); the executors refuse a call
    that reaches a disabled store anyway.
    """
    def _dm_on(_cfg: dict) -> bool:
        return bool(config.enable_user_memory)

    def _channel_on(_cfg: dict) -> bool:
        return bool(config.enable_channel_memory)

    registry.register(get_remember_fact_dm_schema(), execute_remember_fact,
                      enabled=_dm_on,
                      channel_schema=get_remember_fact_schema(), channel_enabled=_channel_on)
    registry.register(get_update_fact_dm_schema(), execute_update_fact,
                      enabled=_dm_on,
                      channel_schema=get_update_fact_schema(), channel_enabled=_channel_on)
    registry.register(get_forget_fact_dm_schema(), execute_forget_fact,
                      enabled=_dm_on,
                      channel_schema=get_forget_fact_schema(), channel_enabled=_channel_on)
    registry.register(get_list_facts_dm_schema(), execute_list_facts,
                      enabled=_dm_on,
                      channel_schema=get_list_facts_schema(), channel_enabled=_channel_on)
