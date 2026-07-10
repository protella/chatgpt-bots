from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from slack_sdk.errors import SlackApiError

from config import config


class SlackHistoryToolMixin:
    """On-demand Slack history-fetch tool (Phase 8).

    Lets the model deliberately pull a bounded slice of a thread's/channel's recent
    messages instead of front-loading everything. Privacy is enforced HERE, at the tool
    layer (never via prompt): content is only returned for public channels or channels the
    bot is a member of; a private channel the bot is not in is refused with no content.

    NOTE: the executor + schemas are complete and tested, but not yet wired to the model —
    the Responses API only runs server-side tools (web_search/MCP) and there is no local
    function-call loop. `get_history_tools_for_openai()` / `dispatch_history_tool_call()`
    are ready for that loop when it's built (see plan Phase 8 follow-up).
    """

    def get_history_tools_for_openai(self) -> List[Dict[str, Any]]:
        """Function-tool schemas for the Responses API (empty when the feature is disabled)."""
        if not config.enable_history_tools:
            return []
        cap = config.history_tool_max_messages
        return [
            {
                "type": "function",
                "name": "fetch_channel_history",
                "description": (
                    "Fetch a bounded slice of recent messages from a Slack channel the bot can "
                    "access (public channels, or private channels the bot is a member of). Use when "
                    "you need more context than the current thread provides."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "channel_id": {"type": "string", "description": "Slack channel ID, e.g. C0123ABC."},
                        "limit": {"type": "integer", "description": f"Max messages to return (1-{cap})."},
                    },
                    "required": ["channel_id"],
                },
            },
            {
                "type": "function",
                "name": "fetch_thread_messages",
                "description": (
                    "Fetch messages from a specific Slack thread in a channel the bot can access "
                    "(public, or private the bot is a member of)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "channel_id": {"type": "string", "description": "Slack channel ID."},
                        "thread_ts": {"type": "string", "description": "Thread root timestamp (ts)."},
                        "limit": {"type": "integer", "description": f"Max messages to return (1-{cap})."},
                    },
                    "required": ["channel_id", "thread_ts"],
                },
            },
        ]

    async def _channel_is_accessible(self, channel_id: str) -> Tuple[bool, str]:
        """Privacy gate: allow public channels and bot-member channels; refuse everything else.

        Returns (allowed, reason). Any lookup failure → (False, ...) so we never leak on error.
        """
        if not channel_id:
            return False, "missing_channel"
        try:
            resp = await self.app.client.conversations_info(channel=channel_id)
            ch = (resp.get("channel") or {}) if resp else {}
            is_private = ch.get("is_private", False)
            is_member = ch.get("is_member", False)
            if not is_private:
                return True, "public"
            if is_member:
                return True, "member"
            return False, "private_non_member"
        except SlackApiError as e:
            err = e.response.get("error", "unknown") if getattr(e, "response", None) else str(e)
            self.log_warning(f"history_tool: conversations_info failed for {channel_id}: {err}")
            return False, f"error:{err}"
        except Exception as e:
            self.log_warning(f"history_tool: access check error for {channel_id}: {e}")
            return False, "error"

    def _clamp_limit(self, limit: Optional[int]) -> int:
        cap = config.history_tool_max_messages
        if not limit:
            return cap
        try:
            return max(1, min(int(limit), cap))
        except (TypeError, ValueError):
            return cap

    async def fetch_history_tool(
        self, channel_id: str, limit: Optional[int] = None, thread_ts: Optional[str] = None
    ) -> Dict[str, Any]:
        """Privacy-gated fetch. Returns a structured dict; on refusal/error contains NO message content."""
        n = self._clamp_limit(limit)
        allowed, reason = await self._channel_is_accessible(channel_id)
        if not allowed:
            return {
                "ok": False,
                "error": "not_accessible",
                "reason": reason,
                "message": (
                    f"Channel {channel_id} is not accessible — it's a private channel the bot is "
                    "not a member of, or it doesn't exist. No content can be returned."
                ),
            }
        try:
            if thread_ts:
                resp = await self.app.client.conversations_replies(channel=channel_id, ts=thread_ts, limit=n)
            else:
                resp = await self.app.client.conversations_history(channel=channel_id, limit=n)
            all_messages = resp.get("messages") or []
            raw = all_messages[:n]
            messages = [
                {
                    "user": m.get("user") or m.get("username") or ("bot" if m.get("bot_id") else "unknown"),
                    "ts": m.get("ts"),
                    "text": m.get("text", ""),
                }
                for m in raw
            ]
            # R5: tell the model whether it saw a window or everything — otherwise
            # "50 messages" is indistinguishable from "the newest 50 of 5,000".
            has_more = bool(
                (resp.get("response_metadata") or {}).get("next_cursor")
                or resp.get("has_more")
                or len(all_messages) > n
            )
            result: Dict[str, Any] = {
                "ok": True,
                "channel": channel_id,
                "thread_ts": thread_ts,
                "count": len(messages),
                "has_more": has_more,
                "messages": messages,
            }
            if has_more:
                result["note"] = "Only the newest window was returned; older history exists beyond this."
            return result
        except SlackApiError as e:
            err = e.response.get("error", "unknown") if getattr(e, "response", None) else str(e)
            self.log_warning(f"history_tool: fetch failed for {channel_id}: {err}")
            return {"ok": False, "error": err, "message": f"Could not fetch history: {err}"}
        except Exception as e:
            self.log_error(f"history_tool: unexpected error for {channel_id}: {e}", exc_info=True)
            return {"ok": False, "error": "exception", "message": "Could not fetch history."}

    async def dispatch_history_tool_call(self, name: str, arguments: Any) -> Dict[str, Any]:
        """Route a model function-call (name + args) to the executor. For the future function-call loop."""
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                return {"ok": False, "error": "bad_arguments", "message": "Arguments were not valid JSON."}
        else:
            args = arguments or {}

        if name == "fetch_channel_history":
            return await self.fetch_history_tool(args.get("channel_id"), args.get("limit"))
        if name == "fetch_thread_messages":
            return await self.fetch_history_tool(args.get("channel_id"), args.get("limit"), args.get("thread_ts"))
        return {"ok": False, "error": "unknown_tool", "message": f"Unknown history tool: {name}"}
