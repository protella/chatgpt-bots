"""T6 — channel bookmarks: the links pinned to the top of a conversation.

A bookmark is the shortest-lived-context surface Slack has: it survives every scroll, every
thread, and every new joiner, which is exactly why it belongs to the PEOPLE in the channel and
not to the bot's own judgment. So `add_bookmark` and `remove_bookmark` are request-only in the
same sense `pin_message` is — the rule lives in the descriptions, because a bookmark bar someone
did not ask for is noise on a surface nobody can scroll past.

Three API facts, all of which shape the code below:

* **`bookmarks.add` needs four fields, not two.** `channel_id`, `title`, `type="link"` and
  `link=<url>`. The `type` is not optional and there is no default — omit it and Slack refuses.
  (`type="file"` bookmarks exist, are not returned by `bookmarks.list`, and cannot be removed —
  see `canvas_tools.execute_create_channel_canvas` for where that was probed. We only ever
  create `link` bookmarks.)
* **Removal takes an opaque `bookmark_id`**, which a person never sees and the model can only
  have gotten from a listing. The executor therefore proves the id SERVER-SIDE: it calls
  `bookmarks.list` itself, immediately before removing, and refuses any id that listing does not
  contain. It cannot lean on "the model listed them earlier this turn" — same-round tool calls
  run on shallow copies of the context (`tool_registry._per_call_context`), so a sibling call's
  discoveries are not visible here, and a previous ROUND's listing is a snapshot that may name a
  bookmark someone has since replaced. A fresh listing is the only proof that survives both.
* **The bookmark bar is shared and small.** Nothing here caps it, validates a URL shape, or
  guesses which bookmark "the docs one" means — an ambiguous ask comes back as the listing so
  the model can ask, and Slack's own error text is what a rejected write reports.

Executors never raise: every failure is an ``{"ok": False, "error": ...}`` result.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from logger import setup_logger
from message_processor.tool_registry import ToolContext, ToolRegistry

logger = setup_logger(name="slack_bot.BookmarkTools")


def _err(code: str, message: str, **extra: Any) -> Dict[str, Any]:
    # Logged as well as returned: the tool loop records only "-> error", so an unlogged refusal
    # is indistinguishable from a Slack outage when someone reads the logs afterwards.
    logger.info(f"bookmark tool refused: {code} — {message}")
    return {"ok": False, "error": code, "message": message, **extra}


def _web(ctx: ToolContext) -> Any:
    """The Slack WebClient underneath the platform client."""
    client = getattr(ctx, "client", None)
    app = getattr(client, "app", None)
    return getattr(app, "client", None) if app is not None else None


async def _async(fn: Any, **kwargs: Any) -> Any:
    """slack_sdk's WebClient here is the AsyncWebClient under Bolt's async app."""
    result = fn(**kwargs)
    if hasattr(result, "__await__"):
        result = await result
    return result


def _slack_error(exc: Exception) -> str:
    """Slack's own error name for a failed call, falling back to the exception text.

    The API error name (`invalid_arguments`, `not_in_channel`, `channel_not_found`) is the whole
    classification and is what the model can act on; `str(e)` on a SlackApiError is a wall of
    response dump that says the same thing less usefully.
    """
    resp = getattr(exc, "response", None)
    getter = getattr(resp, "get", None) if resp is not None else None
    if callable(getter):
        name = getter("error")
        if isinstance(name, str) and name:
            return name
    return str(exc)[:200]


def _entry(raw: Any) -> Optional[Dict[str, Any]]:
    """One bookmark, reduced to what a caller can act on. Junk entries are skipped, not fatal."""
    if not isinstance(raw, dict) or not raw.get("id"):
        return None
    return {"bookmark_id": raw["id"],
            "title": raw.get("title") or "",
            "link": raw.get("link") or "",
            "type": raw.get("type") or ""}


async def _fetch_bookmarks(ctx: ToolContext) -> List[Dict[str, Any]]:
    """The channel's bookmarks, live. Raises on a Slack failure — callers classify it."""
    web = _web(ctx)
    res = await _async(web.bookmarks_list, channel_id=ctx.channel_id)
    out = []
    for raw in (res.get("bookmarks") or []):
        entry = _entry(raw)
        if entry is not None:
            out.append(entry)
    return out


# --- schemas ------------------------------------------------------------------------------

def get_list_bookmarks_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "list_bookmarks",
        "description": (
            "List the bookmarks pinned to the top of this conversation — the links everyone "
            "here sees above the message history. Each one comes back with a bookmark_id, its "
            "title and its URL.\n\n"
            "Use it when someone asks what is bookmarked, when they refer to a link that sounds "
            "like it lives up there ('the runbook we pinned'), or before removing one — a "
            "bookmark_id is opaque and a listing is the only place to get a real one."
        ),
        "parameters": {"type": "object", "properties": {}, "required": [],
                       "additionalProperties": False},
    }


def get_add_bookmark_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "add_bookmark",
        "description": (
            "Pin a link to the top of this conversation as a bookmark, where everyone in it will "
            "see it from now on — only when someone here asks you to.\n\n"
            "The bookmark bar is a shared surface with very little room, and it is the first "
            "thing anyone sees. Never add one on your own initiative, never add one because a "
            "link seems useful, and never add one to make a point. If a person asks you to "
            "bookmark something, do it once; if it is not obvious which link or what to call it, "
            "ask rather than guessing.\n\n"
            "Adding is not the same as posting: if they just want to read something now, put the "
            "link in your reply instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": ("The label shown in the bookmark bar. Short and concrete — "
                                    "'Deploy runbook', not 'Link'. Use the words the person "
                                    "used for it."),
                },
                "url": {
                    "type": "string",
                    "description": "The full URL the bookmark opens, exactly as given.",
                },
            },
            "required": ["title", "url"],
            "additionalProperties": False,
        },
    }


def get_remove_bookmark_schema() -> Dict[str, Any]:
    return {
        "type": "function",
        "name": "remove_bookmark",
        "description": (
            "Remove a bookmark from the top of this conversation — only when someone here asks "
            "you to remove that specific one.\n\n"
            "It disappears for everyone, and it may be a link somebody else put there, so never "
            "tidy the bookmark bar on your own initiative and never remove one merely because it "
            "looks stale. If there is any doubt about which bookmark they mean, list them and "
            "ask.\n\n"
            "The bookmark_id must come from list_bookmarks — it is checked against the "
            "conversation's live bookmarks before anything is removed, so a remembered or "
            "guessed id is refused."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "bookmark_id": {
                    "type": "string",
                    "description": "The id of the bookmark to remove, from list_bookmarks.",
                },
            },
            "required": ["bookmark_id"],
            "additionalProperties": False,
        },
    }


# --- executors ----------------------------------------------------------------------------

async def execute_list_bookmarks(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    if _web(ctx) is None or not ctx.channel_id:
        return _err("unavailable", "Bookmarks aren't available right now.")
    try:
        bookmarks = await _fetch_bookmarks(ctx)
    except Exception as e:  # noqa: BLE001 — the tool contract is a dict, never a raise
        return _err("list_failed", f"Slack could not list the bookmarks: {_slack_error(e)}")
    return {"ok": True, "bookmarks": bookmarks, "count": len(bookmarks)}


async def execute_add_bookmark(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    title = args.get("title")
    url = args.get("url")
    # Type-checked before any coercion, so a malformed argument never reaches the workspace.
    if not isinstance(title, str) or not title.strip():
        return _err("missing_title", "A bookmark needs a title.")
    if not isinstance(url, str) or not url.strip():
        return _err("missing_url", "A bookmark needs a URL.")
    title = title.strip()
    url = url.strip()
    if _web(ctx) is None or not ctx.channel_id:
        return _err("unavailable", "Bookmarks aren't available right now.")

    try:
        # `type="link"` is REQUIRED by bookmarks.add and has no default; `link` is where the URL
        # goes. All four fields or a 400 — see the module docstring.
        res = await _async(_web(ctx).bookmarks_add, channel_id=ctx.channel_id, title=title,
                           type="link", link=url)
    except Exception as e:  # noqa: BLE001
        err = _slack_error(e)
        logger.warning(f"bookmarks.add failed in {ctx.channel_id}: {err}")
        return _err("add_failed", f"Slack refused to add the bookmark: {err}")

    entry = _entry(res.get("bookmark") or {}) or {}
    logger.info(f"Added bookmark {entry.get('bookmark_id')} ({title!r}) to {ctx.channel_id}")
    return {"ok": True, "bookmark_id": entry.get("bookmark_id"), "title": title, "link": url,
            "message": ("The bookmark is in this conversation's bookmark bar. Confirm it "
                        "briefly — everyone here can already see it, so don't paste the link "
                        "back as well.")}


async def execute_remove_bookmark(ctx: ToolContext, args: Dict[str, Any]) -> Dict[str, Any]:
    bookmark_id = args.get("bookmark_id")
    if not isinstance(bookmark_id, str) or not bookmark_id.strip():
        return _err("missing_bookmark_id", "A bookmark_id is required.")
    bookmark_id = bookmark_id.strip()
    if _web(ctx) is None or not ctx.channel_id:
        return _err("unavailable", "Bookmarks aren't available right now.")

    # THE PROOF, and it is deliberately a LIVE one. See the module docstring: a same-round
    # listing cannot reach this call, and an older one is a snapshot. Fail CLOSED — if the
    # listing cannot be read, nothing is removed, because an unverifiable id is exactly the
    # case this check exists for.
    try:
        bookmarks = await _fetch_bookmarks(ctx)
    except Exception as e:  # noqa: BLE001
        err = _slack_error(e)
        return _err("list_failed",
                    f"I couldn't read this conversation's bookmarks, so I removed nothing: "
                    f"{err}")

    match = next((b for b in bookmarks if b["bookmark_id"] == bookmark_id), None)
    if match is None:
        return _err("unknown_bookmark",
                    f"{bookmark_id} is not a bookmark in this conversation. Pick one of the "
                    "bookmarks listed here.",
                    bookmarks=bookmarks)

    try:
        await _async(_web(ctx).bookmarks_remove, channel_id=ctx.channel_id,
                     bookmark_id=bookmark_id)
    except Exception as e:  # noqa: BLE001
        err = _slack_error(e)
        logger.warning(f"bookmarks.remove failed in {ctx.channel_id}: {err}")
        return _err("remove_failed", f"Slack refused to remove the bookmark: {err}")

    # WARNING, not info: a removal is visible to everyone and nobody gets a notification about
    # it, so this line is the only record that the bar changed and who asked for it.
    logger.warning(f"REMOVED bookmark {bookmark_id} ({match['title']!r}) from "
                   f"{ctx.channel_id} (requested by {ctx.user_id})")
    return {"ok": True, "bookmark_id": bookmark_id, "title": match["title"],
            "link": match["link"], "removed": True,
            "message": "The bookmark is gone from the bookmark bar. Confirm it briefly."}


def register_bookmark_tools(registry: ToolRegistry) -> None:
    """list_bookmarks / add_bookmark / remove_bookmark, both surfaces.

    Static schemas: the bookmark bar is not a per-turn fact and putting the current bookmarks
    into a description would fork the cached prefix on every change. `list_bookmarks` is how the
    model finds out what is there, and `remove_bookmark` re-reads it live anyway.

    Registered ungated, like `pin_message`: there is no feature flag for these, and the
    request-only policy for the two writes lives in their descriptions.
    """
    registry.register(get_list_bookmarks_schema(), execute_list_bookmarks)
    registry.register(get_add_bookmark_schema(), execute_add_bookmark)
    registry.register(get_remove_bookmark_schema(), execute_remove_bookmark)
