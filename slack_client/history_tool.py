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

    Wired to the model through the local function-call loop (registered in
    SlackBot._build_tool_registry). Beyond history slices, this mixin also hosts the
    other on-demand workspace-context tools: message permalinks, channel info, pinned
    messages, and user profiles — same privacy gate, same graceful-refusal contract.
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
                    "you need more context than the current thread provides. Each message includes "
                    "its current emoji reactions (who reacted with what)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "channel_id": {"type": "string", "description": "Slack channel ID. Omit to use the CURRENT channel. Only pass an ID you have actually seen in context or from another tool — never guess one."},
                        "limit": {"type": "integer", "description": f"Max messages to return (1-{cap})."},
                    },
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "fetch_thread_messages",
                "description": (
                    "Fetch messages from a specific Slack thread in a channel the bot can access "
                    "(public, or private the bot is a member of). Each message includes its current "
                    "emoji reactions (who reacted with what) — use this to check up-to-date "
                    "reactions, including on the current thread's own messages."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "channel_id": {"type": "string", "description": "Slack channel ID. Omit to use the CURRENT channel; never guess an ID."},
                        "thread_ts": {"type": "string", "description": "Thread root timestamp (ts). Omit to use the CURRENT thread."},
                        "limit": {"type": "integer", "description": f"Max messages to return (1-{cap})."},
                    },
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "get_message_permalink",
                "description": (
                    "Get a permanent Slack link to a specific message (by channel and message ts). "
                    "Use when the user asks WHERE something was said or wants a pointer to an "
                    "earlier message — find the message first (history/search tools give you its "
                    "ts), then include the returned URL in your reply; Slack renders it as a "
                    "clickable message preview."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "channel_id": {"type": "string", "description": "Slack channel ID the message lives in. Omit for the CURRENT channel; never guess an ID."},
                        "message_ts": {"type": "string", "description": "The message's timestamp (ts), e.g. 1720500000.123456."},
                    },
                    "required": ["message_ts"],
                },
            },
            {
                "type": "function",
                "name": "fetch_channel_info",
                "description": (
                    "Get a channel's name, topic, purpose, member count, and privacy flag. Use for "
                    "questions about what a channel is for or its basic facts."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "channel_id": {"type": "string", "description": "Slack channel ID. Omit to use the CURRENT channel; never guess an ID."},
                    },
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "fetch_pinned_messages",
                "description": (
                    "List a channel's pinned messages (text, author, ts, permalink). Pins usually "
                    "hold a channel's important references — check them when asked about a "
                    "channel's key links, rules, or standing info."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "channel_id": {"type": "string", "description": "Slack channel ID. Omit to use the CURRENT channel; never guess an ID."},
                    },
                    "required": [],
                },
            },
            {
                "type": "function",
                "name": "fetch_user_profile",
                "description": (
                    "Look up a Slack user's profile: real name, display name, title, timezone, and "
                    "whether they're a bot. Use to answer who someone is or their local time; user "
                    "IDs appear in messages as <@U…> mentions and in reaction data."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string", "description": "Slack user ID, e.g. U0123ABC."},
                    },
                    "required": ["user_id"],
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
            messages = []
            for m in raw:
                entry = {
                    "user": m.get("user") or m.get("username") or ("bot" if m.get("bot_id") else "unknown"),
                    "ts": m.get("ts"),
                    "text": m.get("text", ""),
                }
                # F25: surface attached-file names so the model can reach a document
                # seen in fetched history via read_document (names, never content).
                # Cap = 10, Slack's own per-message attachment max — every file on a
                # message stays discoverable by name.
                if m.get("files"):
                    names = [f.get("name") for f in m["files"] if isinstance(f, dict) and f.get("name")]
                    if names:
                        entry["files"] = names[:10]
                # Emoji reactions on the message (who reacted with what) — lets the
                # model answer reaction questions with current data, since in-memory
                # thread state only carries reactions present at capture time.
                if m.get("reactions"):
                    entry["reactions"] = [
                        {
                            "emoji": r.get("name"),
                            "count": r.get("count") or len(r.get("users") or []),
                            "users": r.get("users") or [],
                        }
                        for r in m["reactions"] if r.get("name")
                    ]
                messages.append(entry)
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

    async def get_message_permalink_tool(self, channel_id: str, message_ts: str) -> Dict[str, Any]:
        """Permanent link to one message. Same privacy gate as history: no link into
        a private channel the bot is not a member of."""
        allowed, reason = await self._channel_is_accessible(channel_id)
        if not allowed:
            return {"ok": False, "error": "not_accessible", "reason": reason,
                    "message": f"Channel {channel_id} is not accessible — no link can be returned."}
        try:
            resp = await self.app.client.chat_getPermalink(channel=channel_id, message_ts=message_ts)
            return {"ok": True, "channel": channel_id, "message_ts": message_ts,
                    "permalink": resp.get("permalink")}
        except SlackApiError as e:
            err = e.response.get("error", "unknown") if getattr(e, "response", None) else str(e)
            self.log_warning(f"history_tool: permalink failed for {channel_id}/{message_ts}: {err}")
            return {"ok": False, "error": err, "message": f"Could not get a permalink: {err}"}

    async def fetch_channel_info_tool(self, channel_id: str) -> Dict[str, Any]:
        """Channel facts (name/topic/purpose/member count). Privacy-gated like history."""
        allowed, reason = await self._channel_is_accessible(channel_id)
        if not allowed:
            return {"ok": False, "error": "not_accessible", "reason": reason,
                    "message": f"Channel {channel_id} is not accessible — no info can be returned."}
        try:
            resp = await self.app.client.conversations_info(channel=channel_id, include_num_members=True)
            ch = resp.get("channel") or {}
            return {
                "ok": True,
                "channel": channel_id,
                "name": ch.get("name"),
                "topic": (ch.get("topic") or {}).get("value"),
                "purpose": (ch.get("purpose") or {}).get("value"),
                "num_members": ch.get("num_members"),
                "is_private": bool(ch.get("is_private")),
            }
        except SlackApiError as e:
            err = e.response.get("error", "unknown") if getattr(e, "response", None) else str(e)
            self.log_warning(f"history_tool: channel info failed for {channel_id}: {err}")
            return {"ok": False, "error": err, "message": f"Could not fetch channel info: {err}"}

    async def fetch_pinned_messages_tool(self, channel_id: str) -> Dict[str, Any]:
        """Pinned items for a channel. Privacy-gated; degrades with a clear message if
        the app is missing the pins:read scope (added to the manifest 2026-07-10)."""
        allowed, reason = await self._channel_is_accessible(channel_id)
        if not allowed:
            return {"ok": False, "error": "not_accessible", "reason": reason,
                    "message": f"Channel {channel_id} is not accessible — no pins can be returned."}
        try:
            resp = await self.app.client.pins_list(channel=channel_id)
            pins = []
            for item in resp.get("items") or []:
                msg = item.get("message") or {}
                if not msg:
                    continue  # pinned files et al. — messages only
                pins.append({
                    "user": msg.get("user") or msg.get("username") or ("bot" if msg.get("bot_id") else "unknown"),
                    "ts": msg.get("ts"),
                    "text": msg.get("text", ""),
                    "permalink": msg.get("permalink"),
                })
            return {"ok": True, "channel": channel_id, "count": len(pins), "pins": pins}
        except SlackApiError as e:
            err = e.response.get("error", "unknown") if getattr(e, "response", None) else str(e)
            if err == "missing_scope":
                return {"ok": False, "error": err,
                        "message": "The Slack app lacks the pins:read scope — an admin must update "
                                   "the app manifest and reinstall before pins can be read."}
            self.log_warning(f"history_tool: pins failed for {channel_id}: {err}")
            return {"ok": False, "error": err, "message": f"Could not fetch pins: {err}"}

    async def fetch_user_profile_tool(self, user_id: str) -> Dict[str, Any]:
        """Workspace-visible profile facts for one user (no email — keep it to what any
        member sees on the profile card)."""
        try:
            resp = await self.app.client.users_info(user=user_id)
            u = resp.get("user") or {}
            profile = u.get("profile") or {}
            return {
                "ok": True,
                "user_id": user_id,
                "real_name": profile.get("real_name") or u.get("real_name"),
                "display_name": profile.get("display_name") or None,
                "title": profile.get("title") or None,
                "timezone": u.get("tz"),
                "timezone_label": u.get("tz_label"),
                "is_bot": bool(u.get("is_bot")),
            }
        except SlackApiError as e:
            err = e.response.get("error", "unknown") if getattr(e, "response", None) else str(e)
            self.log_warning(f"history_tool: user profile failed for {user_id}: {err}")
            return {"ok": False, "error": err, "message": f"Could not fetch that user's profile: {err}"}

    async def dispatch_history_tool_call(self, name: str, arguments: Any, ctx: Any = None) -> Dict[str, Any]:
        """Route a model function-call (name + args) to its executor.

        ``channel_id``/``thread_ts`` default to the CURRENT conversation (from the
        ToolContext) when omitted — the model doesn't know Slack IDs it hasn't seen,
        and requiring them made it fabricate plausible ones (channel_not_found)."""
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments or "{}")
            except json.JSONDecodeError:
                return {"ok": False, "error": "bad_arguments", "message": "Arguments were not valid JSON."}
        else:
            args = arguments or {}

        channel_id = args.get("channel_id") or getattr(ctx, "channel_id", None)

        if name == "fetch_channel_history":
            return await self.fetch_history_tool(channel_id, args.get("limit"))
        if name == "fetch_thread_messages":
            thread_ts = args.get("thread_ts") or getattr(ctx, "thread_ts", None)
            if not thread_ts:
                return {"ok": False, "error": "bad_arguments",
                        "message": "No thread here — pass thread_ts or use fetch_channel_history."}
            return await self.fetch_history_tool(channel_id, args.get("limit"), thread_ts)
        if name == "get_message_permalink":
            return await self.get_message_permalink_tool(channel_id, args.get("message_ts"))
        if name == "fetch_channel_info":
            return await self.fetch_channel_info_tool(channel_id)
        if name == "fetch_pinned_messages":
            return await self.fetch_pinned_messages_tool(channel_id)
        if name == "fetch_user_profile":
            return await self.fetch_user_profile_tool(args.get("user_id"))
        return {"ok": False, "error": "unknown_tool", "message": f"Unknown history tool: {name}"}
