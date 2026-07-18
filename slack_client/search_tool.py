from __future__ import annotations

from typing import Any, Dict, List

from slack_sdk.errors import SlackApiError

from config import config


# API errors that mean the action token can't authorize a search right now.
# The exact expired-token error string is not documented; treat anything
# token-shaped as "search unavailable" so the model falls back to history tools.
# TODO(live): verify real token-TTL error strings on the dev bot and tighten this.
_TOKEN_ERRORS = {
    "invalid_action_token",
    "action_token_expired",
    "expired_action_token",
    "missing_action_token",
    "invalid_auth",
}

# The only channel types the API accepts. The env gate is intersected with this.
_VALID_CHANNEL_TYPES = {"public_channel", "private_channel", "im", "mpim"}


class SlackSearchToolMixin:
    """`search_slack` local tool (Phase B) — assistant.search.context.

    Privacy model (enforced in code, not prompt):
    - The API itself requires an `action_token` minted by the triggering user
      message/app_mention event, so the bot physically cannot search except in
      response to a user interaction. Multi-round tool loops reuse the same
      token; its TTL is undocumented, so token errors degrade to
      "search_unavailable" and the model falls back to the history tools.
    - `SEARCH_CHANNEL_TYPES` (default: public_channel,private_channel) bounds
      what the executor will ever request, regardless of what the manifest
      scopes would allow. DMs/group DMs stay out of reach unless the operator
      adds im/mpim here (which also requires the search:read.im/mpim scopes —
      installed, but off by code default for privacy). Prompt-injected "search
      his DMs" cannot widen this.
    """

    def get_search_tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "name": "search_slack",
            "description": (
                "Search Slack messages the bot is allowed to see (workspace-wide or the "
                "current channel). Use for finding older discussions, decisions, or context "
                "outside the current thread; prefer fetch_thread_messages/fetch_channel_history "
                "for things in the current conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for."},
                    "scope": {
                        "type": "string",
                        "enum": ["channel", "workspace"],
                        "description": "Limit results to the current channel, or search the whole workspace (default).",
                    },
                    "limit": {"type": "integer", "description": "Max results (1-20, default 10)."},
                },
                "required": ["query"],
            },
        }

    def _search_channel_types(self) -> List[str]:
        configured = [t.strip() for t in (config.search_channel_types or []) if t and t.strip()]
        return [t for t in configured if t in _VALID_CHANNEL_TYPES]

    @staticmethod
    def _clamp_search_limit(limit: Any) -> int:
        try:
            return max(1, min(int(limit), 20))
        except (TypeError, ValueError):
            return 10

    async def execute_search_tool(self, ctx, args: Dict[str, Any]) -> Dict[str, Any]:
        """Run assistant.search.context. Never raises; no content on refusal/error."""
        query = (args.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": "bad_arguments", "message": "query is required."}

        if not getattr(ctx, "action_token", None):
            # Older/replayed events (or non-AI-app surfaces) carry no token.
            # Log the cause: the registry's generic "-> error" line alone left four
            # live failures undiagnosable (2026-07-18).
            self.log_info("search_tool: unavailable — event carried no action_token")
            return {
                "ok": False,
                "error": "search_unavailable",
                "hint": "Search needs a fresh user message to authorize it. Use fetch_channel_history or fetch_thread_messages instead.",
            }

        channel_types = self._search_channel_types()
        if not channel_types:
            self.log_info("search_tool: refused — no searchable channel types configured")
            return {"ok": False, "error": "search_disabled", "message": "No searchable channel types are configured."}

        limit = self._clamp_search_limit(args.get("limit"))
        scope = (args.get("scope") or "workspace").strip().lower()

        request: Dict[str, Any] = {
            "query": query,
            "action_token": ctx.action_token,
            "channel_types": ",".join(channel_types),
            "content_types": "messages",
            "limit": limit,
        }
        if ctx.channel_id:
            # Boosts relevance for the conversation the request came from.
            request["context_channel_id"] = ctx.channel_id

        try:
            # slack-sdk 3.43.0 has no assistant_search_context wrapper yet;
            # the generic api_call hits the Web API method directly.
            resp = await self.app.client.api_call("assistant.search.context", data=request)
        except SlackApiError as e:
            err = e.response.get("error", "unknown") if getattr(e, "response", None) else str(e)
            if err in _TOKEN_ERRORS:
                self.log_info(f"search_tool: action token rejected ({err}) — degraded to search_unavailable")
                return {
                    "ok": False,
                    "error": "search_unavailable",
                    "hint": "The search authorization expired. Use fetch_channel_history or fetch_thread_messages instead.",
                }
            self.log_warning(f"search_tool: assistant.search.context failed: {err}")
            return {"ok": False, "error": err, "message": f"Search failed: {err}"}
        except Exception as e:
            self.log_error(f"search_tool: unexpected error: {e}", exc_info=True)
            return {"ok": False, "error": "exception", "message": "Search failed."}

        raw = resp.get("results", resp) or {}
        messages = raw.get("messages") or []
        results = []
        for m in messages:
            channel = m.get("channel_id") or (m.get("channel") or {}).get("id") or m.get("channel")
            if scope == "channel" and ctx.channel_id and channel != ctx.channel_id:
                continue
            results.append(
                {
                    "channel": channel,
                    "ts": m.get("message_ts") or m.get("ts"),
                    "author": m.get("author_user_id") or m.get("user") or m.get("username"),
                    "text": m.get("content") or m.get("text", ""),
                    "permalink": m.get("permalink"),
                }
            )
            if len(results) >= limit:
                break

        return {"ok": True, "query": query, "scope": scope, "count": len(results), "results": results}
