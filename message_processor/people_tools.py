"""Model-invoked people awareness (F29 of the channel-teammate work).

Two thin, read-only tools over Slack's own profile data — Slack is the source of
truth for who people are (no new people table; see the F29 design notes):

- ``lookup_user`` — resolve a workspace member from a Slack id, an @name, or a
  display/real name seen ANYWHERE in context (chat, the "Channel people" line, a
  roster, channel memory), then return FRESH profile facts from users.info.
- ``list_channel_members`` — the CURRENT channel's roster (names + total count),
  capped with a LOUD truncation note so a big channel never silently lies.

Both return only workspace-visible profile data — nothing beyond what any member
sees on a profile card. Executors never raise: every failure is an
``{"ok": False, "error": ...}`` result.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from tool_registry import ToolContext, ToolRegistry

# list_channel_members: resolve at most this many names; the rest are a LOUD note.
MEMBERS_NAME_CAP = 50
_MEMBERS_PAGE_LIMIT = 200
# Safety bound on pagination so a pathological (thousands-strong) channel can't spin the
# roster call forever; when hit, the count is reported as a floor, not exact.
_MEMBERS_MAX_PAGES = 30

# A bare Slack user id (U…/W…) or a pasted "<@U…>" / "<@U…|name>" mention.
_MENTION_RE = re.compile(r"^<@([UW][A-Z0-9]+)(?:\|[^>]*)?>$")
_BARE_ID_RE = re.compile(r"^[UW][A-Z0-9]{6,}$")


def format_people_summary(num_members: Optional[int],
                          speakers: Optional[List[str]]) -> Optional[str]:
    """Shared one-line people summary used by BOTH the participation signal and the
    response suffix, so the two surfaces read identically:
        "~12 members; recently active: Erin Evans, Claude"
    Defensive: returns None when neither piece is known, and omits whichever is missing."""
    parts: List[str] = []
    try:
        n = int(num_members) if num_members is not None else None
    except (TypeError, ValueError):
        n = None
    if n and n > 0:
        parts.append(f"~{n} member" + ("s" if n != 1 else ""))
    names = [s.strip() for s in (speakers or []) if s and s.strip()]
    if names:
        parts.append("recently active: " + ", ".join(names))
    return "; ".join(parts) if parts else None


def get_lookup_user_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "lookup_user",
        "description": (
            "Look up a workspace member's profile — real name, title, status, timezone, email, "
            "and whether they're a bot. Use it whenever someone asks who a person is, "
            "or about their role, title, timezone, or status. You do NOT need a Slack id: ANY "
            "name you've seen is enough — a name in chat, in the 'Channel people' line, in a "
            "participant roster, or in channel memory. Pass what you have as `user` (a Slack id "
            "like U012ABC, an @name, or a display/real name) and this resolves it to FRESH data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "user": {
                    "type": "string",
                    "description": "A Slack user id (U012ABC), an @name, or a display/real name "
                                   "seen anywhere in chat, a people line, a roster, or channel memory.",
                },
            },
            "required": ["user"],
        },
    }


def get_list_channel_members_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "list_channel_members",
        "description": (
            "List the members of the CURRENT channel — their names and the total count. Use it "
            "when asked who is in this channel, who's here, or how many people are in the "
            "channel. Names resolve to real/display names; a large channel is truncated with an "
            "explicit note (never a silent cap)."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    }


def _api_client(ctx: ToolContext) -> Any:
    """The Slack Web API client (``bot.app.client``), or None on a non-Slack platform."""
    return getattr(getattr(ctx.client, "app", None), "client", None)


def _parse_user_id(raw: str) -> Optional[str]:
    """A bare Slack id or a pasted <@U…> mention → the id; else None (it's a name)."""
    m = _MENTION_RE.match(raw)
    if m:
        return m.group(1)
    if _BARE_ID_RE.match(raw):
        return raw
    return None


async def _resolve_by_name(ctx: ToolContext, raw: str) -> List[Dict[str, Any]]:
    """Case-insensitive EXACT match on username/real_name across the in-memory user_cache
    and the persisted user_info rows. Returns one dict per distinct matched user_id."""
    want = raw.lstrip("@").strip().lower()
    if not want:
        return []
    found: Dict[str, Dict[str, Any]] = {}

    def _consider(uid: Optional[str], username: Any, real_name: Any) -> None:
        if not uid or uid in found:
            return
        for val in ((username or "").strip().lower(), (real_name or "").strip().lower()):
            if val and val == want:
                found[uid] = {"user_id": uid, "username": username, "real_name": real_name}
                return

    cache = getattr(ctx.client, "user_cache", None) or {}
    for uid, info in list(cache.items()):
        if isinstance(info, dict):
            _consider(uid, info.get("username"), info.get("real_name"))

    try:
        rows = await ctx.db.get_all_users_async() if ctx.db else []
    except Exception:
        rows = []
    for r in rows or []:
        _consider(r.get("user_id"), r.get("username"), r.get("real_name"))

    return list(found.values())


def _profile_result(user_id: str, u: Dict[str, Any]) -> Dict[str, Any]:
    """Shape a users.info payload into the tool result — workspace-visible fields only."""
    profile = u.get("profile") or {}
    return {
        "ok": True,
        "id": user_id,
        "username": profile.get("display_name") or u.get("name"),
        "real_name": profile.get("real_name") or u.get("real_name"),
        "title": profile.get("title") or None,
        "status_text": profile.get("status_text") or None,
        "status_emoji": profile.get("status_emoji") or None,
        "timezone": u.get("tz_label") or u.get("tz") or None,
        "is_bot": bool(u.get("is_bot")),
        "email": profile.get("email") or None,
    }


async def execute_lookup_user(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve id/@name/display name → users.info (fresh). Never raises."""
    raw = (args.get("user") or "").strip()
    if not raw:
        return {"ok": False, "error": "bad_arguments",
                "hint": "Provide a Slack id, an @name, or a display/real name."}
    api = _api_client(ctx)
    if api is None:
        return {"ok": False, "error": "unavailable",
                "hint": "User lookup is not available on this platform."}

    user_id = _parse_user_id(raw)
    if user_id is None:
        matches = await _resolve_by_name(ctx, raw)
        if len(matches) > 1:
            return {"ok": False, "error": "ambiguous",
                    "candidates": [{"id": m["user_id"],
                                    "name": m.get("real_name") or m.get("username")}
                                   for m in matches[:10]],
                    "hint": "Several members match that name — pick one by its id and call again."}
        if not matches:
            return {"ok": False, "error": "not_found",
                    "hint": "No workspace member I've seen matches that name; try the exact "
                            "display or real name, or a Slack id (U012ABC)."}
        user_id = matches[0]["user_id"]

    try:
        resp = await api.users_info(user=user_id)
    except Exception as e:
        return {"ok": False, "error": "lookup_failed", "detail": str(e)[:200]}
    if not resp or not resp.get("ok"):
        return {"ok": False, "error": (resp or {}).get("error") or "lookup_failed",
                "hint": "That id didn't resolve to a workspace member."}
    return _profile_result(user_id, resp.get("user") or {})


async def execute_list_channel_members(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    """Current channel's roster (names + count), name-capped with a LOUD truncation note.
    Current channel only — always uses ctx.channel_id, never a caller-supplied one."""
    if ctx.is_dm:
        return {"ok": False, "error": "not_a_channel",
                "hint": "This is a DM, not a channel with a member roster."}
    channel_id = ctx.channel_id
    if not channel_id:
        return {"ok": False, "error": "no_channel", "hint": "No channel in this context."}
    api = _api_client(ctx)
    if api is None:
        return {"ok": False, "error": "unavailable",
                "hint": "Member listing is not available on this platform."}

    ids: List[str] = []
    cursor = ""
    pages = 0
    count_is_partial = False
    try:
        while True:
            resp = await api.conversations_members(
                channel=channel_id, cursor=cursor or None, limit=_MEMBERS_PAGE_LIMIT)
            if not resp or not resp.get("ok"):
                return {"ok": False, "error": (resp or {}).get("error") or "members_failed",
                        "hint": "Could not read this channel's membership."}
            ids.extend(resp.get("members") or [])
            cursor = (resp.get("response_metadata") or {}).get("next_cursor") or ""
            pages += 1
            if not cursor:
                break
            if pages >= _MEMBERS_MAX_PAGES:
                count_is_partial = True  # bailed on the page cap; total is a floor
                break
    except Exception as e:
        return {"ok": False, "error": "members_failed", "detail": str(e)[:200]}

    total = len(ids)
    shown_ids = ids[:MEMBERS_NAME_CAP]
    names: List[str] = []
    for uid in shown_ids:
        try:
            names.append(await ctx.client.get_username(uid, api))
        except Exception:
            names.append(uid)

    result: Dict[str, Any] = {
        "ok": True,
        "channel": channel_id,
        "total_members": total,
        "shown": len(names),
        "members": names,
    }
    if count_is_partial:
        result["count_is_partial"] = True
        result["total_members_note"] = (
            f"At least {total} members — the channel is large enough that the full roster "
            "wasn't paginated; treat the count as a floor.")
    if total > len(names):
        result["truncated"] = True
        result["note"] = (
            f"Only the first {len(names)} of {total} members are named here — this is a "
            "PARTIAL roster, not the full list; say so if you relay it.")
    return result


def register_people_tools(registry: ToolRegistry) -> None:
    """Register lookup_user + list_channel_members (call only when ENABLE_PEOPLE_TOOLS is on)."""
    registry.register(get_lookup_user_schema(), execute_lookup_user)
    registry.register(get_list_channel_members_schema(), execute_list_channel_members)
