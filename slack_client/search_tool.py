from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

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

    KNOWN, DELIBERATE EXCEPTION to the channel-read authorization policy (2026-07, owner
    decision — do not "fix" this without asking). Every other channel-read surface
    (history_tool's five tools, lookup_channel, list_channel_members) requires BOTH the bot
    and the REQUESTER to be members of the target conversation. search_slack does not: with
    `search:read.public`, assistant.search.context returns public-channel content the
    requester is not a member of. That is accepted here — public channels are readable by any
    member of the workspace by design — so search is a known way to reach public content the
    both-members rule would refuse. The bounds below (action_token + channel-type allowlist)
    are what keeps it from reaching anything PRIVATE.

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

    if TYPE_CHECKING:  # provided by the host (SlackBot) and the mixed-in SlackHistoryToolMixin
        app: Any
        log_info: Any
        log_warning: Any
        log_error: Any
        resolve_usernames: Any
        # The canonical delivery-audience decision lives in SlackHistoryToolMixin; the two mixins
        # share ONE SlackBot instance, so it is reachable as self.… here.
        _delivery_allowed: Any

    def get_search_tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "name": "search_slack",
            "description": (
                "Search Slack messages the bot is allowed to see (workspace-wide or the "
                "current channel). Use for finding older discussions, decisions, or context "
                "outside the current thread; prefer fetch_thread_messages/fetch_channel_history "
                "for things in the current conversation. Each result carries its source channel "
                "id; pass that to resolve_channel_name to show the channel's name."
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

    @staticmethod
    def _parse_search_source_id(m: Dict[str, Any]) -> Optional[str]:
        """The hit's channel id, parsed DEFENSIVELY. `channel` may be a dict, a bare string id, or
        absent — the old `(m.get("channel") or {}).get("id")` RAISED on the string form. A hit with
        no positively-parsed string id returns None and is dropped in a multi-user surface."""
        cid = m.get("channel_id")
        if isinstance(cid, str) and cid:
            return cid
        ch = m.get("channel")
        if isinstance(ch, dict):
            cid = ch.get("id")
            return cid if isinstance(cid, str) and cid else None
        if isinstance(ch, str) and ch:
            return ch
        return None

    @staticmethod
    def _parse_search_source_team_ids(m: Dict[str, Any]) -> List[str]:
        """The DISTINCT workspace/team ids a hit carries (top-level and on any channel object),
        threaded to the delivery gate so a cross-workspace hit is dropped before the
        current-channel exemption (codex r3 #4). Returns ALL distinct values, not just the first
        (codex r4): more than one means the hit contradicts itself about its workspace, and the
        caller drops it rather than trust whichever field it happened to read first."""
        found: List[str] = []
        objs: List[Any] = [m]
        ch = m.get("channel")
        if isinstance(ch, dict):
            objs.append(ch)
        for obj in objs:
            for key in ("team_id", "team", "context_team_id"):
                v = obj.get(key)
                if isinstance(v, str) and v and v not in found:
                    found.append(v)
        return found

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

        requested_limit = self._clamp_search_limit(args.get("limit"))
        scope = (args.get("scope") or "workspace").strip().lower()

        # DELIVERY-AUDIENCE GATE (Option B). In a DM the audience is the asker alone, so search
        # keeps its full reach. In any multi-user surface (public/private channel, MPIM/group DM)
        # a hit is deliverable only if its source is the CURRENT channel or a PUBLIC INTERNAL
        # channel — everything else is dropped SILENTLY below, so a filtered result reads exactly
        # like a genuine no-match (no note, no count that would betray which private conversations
        # exist). The both-members RETRIEVAL rule is deliberately NOT reused here: a public channel
        # the bot is a non-member of is a legitimate hit, which retrieval would wrongly deny.
        is_dm_surface = bool(getattr(ctx, "is_dm", False))
        filtering_active = not is_dm_surface
        # False-empty mitigation: when filtering can drop hits, the deliverable ones may rank below
        # dropped private hits, so ask the API for its max and trim to the requested limit AFTER
        # filtering. Channel scope already constrained the API to the current channel via in:<#…>,
        # so its top-N is already deliverable — no widening needed there.
        api_limit = 20 if (filtering_active and scope == "workspace") else requested_limit

        # Channel scope must constrain at the API, not merely post-filter the top-N: a
        # workspace-wide query whose highest-ranked hits all live in other channels
        # returns a false "no matches" for the current one. Slack honours the
        # `in:<#CHANNEL_ID>` search operator inside the query string, so scope the query
        # itself; the channel != ctx.channel_id post-filter below stays as belt-and-braces.
        api_query = query
        if scope == "channel" and ctx.channel_id:
            api_query = f"{query} in:<#{ctx.channel_id}>"

        request: Dict[str, Any] = {
            "query": api_query,
            "action_token": ctx.action_token,
            "channel_types": ",".join(channel_types),
            "content_types": "messages",
            "limit": api_limit,
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

        # FILTER FIRST — before username resolution and permalink copying. Both cost API budget,
        # and resolving a dropped hit's author would leak (via users.info side effects and the
        # resolver's bounded slots) that a withheld conversation exists. Each surviving hit is run
        # through the CANONICAL `_delivery_allowed` rule (mixed in from the history tool on the one
        # SlackBot instance) — the SAME decision history and lookup use — so search can't drift from
        # it (codex r4). That rule applies the current-channel exemption, the public-internal source
        # check, cross-workspace rejection AND the ext-shared-DESTINATION lockdown (an externally
        # shared current channel may deliver only its own content), and drops an unparseable (None)
        # source on its own — so no separate None-guard is needed here.
        kept: List[Tuple[Dict[str, Any], Optional[str]]] = []
        for m in messages:
            if not isinstance(m, dict):
                # A malformed hit (a string/None in `messages`) would raise on .get() below.
                continue
            source = self._parse_search_source_id(m)
            # Channel scope was constrained at the API with in:<#…>; keep the post-filter as
            # belt-and-braces so a stray cross-channel hit can't ride a channel-scoped query.
            if scope == "channel" and ctx.channel_id and source != ctx.channel_id:
                continue
            if filtering_active:
                team_ids = self._parse_search_source_team_ids(m)
                if len(team_ids) > 1:
                    # The hit contradicts itself about its workspace → can't classify → drop.
                    continue
                deliverable, _dreason = await self._delivery_allowed(
                    source, ctx, source_team_id=(team_ids[0] if team_ids else None))
                if not deliverable:
                    continue
            kept.append((m, source))
            if len(kept) >= requested_limit:
                break

        # BF2: render authors by display name, not a raw Slack id — resolved in ONE read-only,
        # budgeted batch over the KEPT (deliverable) set, so searching never creates user rows or
        # bumps last_seen. An unresolved id stays raw.
        api_client = getattr(getattr(self, "app", None), "client", None)
        resolver = getattr(self, "resolve_usernames", None)
        # Ordered dedup in result order (Blocker 2): a hash-ordered set would let the remote
        # budget resolve a different subset across cold starts.
        author_ids = list(dict.fromkeys(
            aid for (m, _s) in kept if (aid := (m.get("author_user_id") or m.get("user")))))
        name_map = {}
        if author_ids and resolver:
            try:
                name_map = await resolver(author_ids, api_client)
            except Exception:
                name_map = {}
        results = []
        for (m, source) in kept:
            author_id = m.get("author_user_id") or m.get("user")
            author = name_map.get(author_id, author_id or m.get("username"))
            results.append(
                {
                    "channel": source,
                    "ts": m.get("message_ts") or m.get("ts"),
                    "author": author,
                    "text": m.get("content") or m.get("text", ""),
                    "permalink": m.get("permalink"),
                }
            )

        return {"ok": True, "query": query, "scope": scope, "count": len(results), "results": results}
