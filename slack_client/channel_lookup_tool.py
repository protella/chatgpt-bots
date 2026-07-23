"""`lookup_channel` — resolve a channel NAME to its id, inside the requester's own view.

Without this the bot dead-ends on "what's happening in #menu-insights?" asked from a DM: the
channel tools need an id, the model correctly refuses to invent one, and the turn ends in an
apology. This closes that gap WITHOUT becoming a channel-directory oracle.

Two rules make it safe:

1. **Names come only from what the requester can already see.** The candidate set is
   ``users.conversations(user=<requester>)`` — the channels that person is in — and nothing
   else. We never call ``conversations.list(types="private_channel")``: private channel names
   alone leak client, incident and personnel information ("#project-acme-layoffs") to anyone
   who can ask the bot a question, with no message ever being read. We do not scan the PUBLIC
   directory either, though its names are not secret: under the both-members rule a public
   channel the requester isn't in gets refused by rule 2 anyway, so the scan could never add
   an allowed result — only latency, and a false "I couldn't finish looking".
2. **A candidate is only returned if the channel-read gate would allow it.** users.conversations
   deliberately omits ``is_member``, so an entry proves the REQUESTER's membership and nothing
   about the bot's; each match is therefore run through ``_channel_is_accessible`` (which reads
   conversations.info and requires bot membership) before its id is handed back. Returning an id
   the tools would then refuse is both a leak and a lie.

Ambiguity is reported, never guessed, and an incomplete scan says so — "I couldn't finish
looking" must never be rendered as "no such channel".

Mixed into SlackBot alongside SlackHistoryToolMixin, whose authorization helpers
(``_channel_is_accessible``, ``_requester_conversations``, ``ACCESS_DENIED_MESSAGE``) it shares
so both surfaces answer from one policy and one request-scoped memo.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from config import config
from slack_client.history_tool import ACCESS_DENIED_MESSAGE

# Most candidate names resolve to one channel; a handful is already pathological. Each
# candidate costs a conversations.info, so the bot-membership proof stays bounded too.
_MAX_CANDIDATES = 10

_INCOMPLETE_MESSAGE = (
    "I couldn't finish searching the channel list, so this is INCOMPLETE — I found no match, "
    "but I can't tell you that channel doesn't exist. Ask again with the channel id if you "
    "have it."
)


def _normalize(raw: str) -> str:
    """"#Menu-Insights " → "menu-insights". Slack stores names lowercased; the model (and the
    people it quotes) type them with a leading # and stray case."""
    return (raw or "").strip().lstrip("#").strip().lower()


def _name_matches(entry: Dict[str, Any], want: str) -> bool:
    """EXACT match on either name Slack carries — never a prefix or fuzzy match, which would
    let "#eng" silently resolve to "#engineering-payroll"."""
    if not want:
        return False
    for key in ("name", "name_normalized"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip().lower() == want:
            return True
    return False


class SlackChannelLookupToolMixin:
    """`lookup_channel`: name → id, scoped to conversations the requester and bot share."""

    if TYPE_CHECKING:  # provided by the host (SlackBot) and SlackHistoryToolMixin
        app: Any
        log_warning: Any
        _channel_is_accessible: Any
        _delivery_allowed: Any
        _source_is_public: Any
        _authorize_channel_read: Any
        _requester_conversation_entries: Any

    def get_lookup_channel_tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "name": "lookup_channel",
            "description": (
                "Resolve a Slack channel NAME (\"#menu-insights\" or \"menu-insights\") to the "
                "channel id the other channel tools need. Use it whenever someone names a "
                "channel you don't already have an id for — including from a DM — instead of "
                "guessing an id or giving up. Only channels BOTH the person who asked and the "
                "bot are in can be resolved; if the name is ambiguous, or the search couldn't be "
                "completed, that is reported instead of a guess."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The channel name, with or without the leading #.",
                    },
                },
                "required": ["name"],
            },
        }

    def get_resolve_channel_name_tool_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "name": "resolve_channel_name",
            "description": (
                "Turn a Slack channel ID (for example one that came back in a search result) into "
                "its human-readable name. Public channels resolve for anyone — their names are "
                "visible to the whole workspace — and a private conversation resolves only where "
                "its name may be shown: to a member, and never surfaced into a channel where "
                "others would see it. When it returns nothing (\"not_accessible\"), the name "
                "CANNOT be shared here: do NOT guess it, invent it, describe it, or hint at what "
                "it might be — tell the user you can't resolve that one here, and offer to do it "
                "in a DM if that would help. Only pass an ID you actually have from a tool result "
                "or the user; never make one up."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel_id": {
                        "type": "string",
                        "description": "The Slack channel ID to name, e.g. C0123456789.",
                    },
                },
                "required": ["channel_id"],
            },
        }

    async def execute_resolve_channel_name(self, ctx: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        """id → name, scoped so a name is only ever surfaced where it may legitimately be seen.

        A PUBLIC channel's name is workspace-visible (it sits in every member's channel browser),
        so it resolves without either side being a member — but STILL only where it is DELIVERABLE:
        the delivery gate keeps a public name out of an externally-shared / cross-workspace
        destination. Everything else — a private conversation, an ext-shared or cross-team source,
        or an id neither side can see — is put through the strict both-members read gate; only if
        that ALLOWS it (a private channel we share, deliverable into THIS audience) is the name
        returned. Any refusal is the SAME generic message the read tools use, carrying no name, id,
        privacy flag, or existence signal, and the schema tells the model not to guess on refusal.
        """
        cid = (args.get("channel_id") or "").strip()
        if not cid:
            return {"ok": False, "error": "bad_arguments", "message": "Give me the channel ID to resolve."}
        if getattr(ctx, "requester_is_human", False) is not True:
            # Same stance as lookup_channel: only a person's own view resolves names. (A public
            # name is not sensitive, but keeping the whole tool human-gated matches that surface
            # and avoids another app's bot driving resolution on our token.)
            return {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}

        async def _name() -> Any:
            try:
                resp = await self.app.client.conversations_info(channel=cid)
            except Exception as e:  # SlackApiError included — a private channel we can't see 404s here
                self.log_warning(f"resolve_channel_name: info failed for {cid}: {e}")
                return None
            return ((resp.get("channel") or {}) if resp else {}).get("name")

        # Public + deliverable: name it, no membership required.
        deliverable, _dr = await self._delivery_allowed(cid, ctx)
        if deliverable and await self._source_is_public(cid, ctx):
            name = await _name()
            if name:
                return {"ok": True, "id": cid, "name": name}

        # Otherwise the strict both-members read gate (covers the current channel and private
        # conversations we share, and still enforces delivery). ALLOW → the name may be shown;
        # DENY / REDIRECT / unresolvable → the generic refusal, leaking nothing.
        verdict, reason = await self._authorize_channel_read(cid, ctx)
        if verdict == "ALLOW":
            name = await _name()
            if name:
                return {"ok": True, "id": cid, "name": name}

        self.log_warning(f"resolve_channel_name: refused {cid} (verdict={verdict})")
        return {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}

    async def _requester_channel_matches(self, requester: str, want: str,
                                         ctx: Any) -> Tuple[List[Dict[str, Any]], bool]:
        """The requester's own conversations, name-matched: (entries, complete).

        Shares the request-scoped memo with the authorization gate, so the walk that proved
        membership for a fetch also serves this lookup — one pagination per request.
        """
        entries, complete = await self._requester_conversation_entries(requester, ctx)
        return [e for e in entries if _name_matches(e, want)], complete

    async def execute_lookup_channel(self, ctx: Any, args: Dict[str, Any]) -> Dict[str, Any]:
        """Name → id. Never raises; every failure is an {"ok": False, ...} result."""
        want = _normalize(args.get("name") or "")
        if not want:
            return {"ok": False, "error": "bad_arguments",
                    "message": "Give me the channel name, e.g. #menu-insights."}
        api = getattr(getattr(self, "app", None), "client", None)
        if api is None:
            return {"ok": False, "error": "unavailable",
                    "message": "Channel lookup is not available here."}
        requester = getattr(ctx, "user_id", None)
        if not isinstance(requester, str) or not requester:
            # Same stance as the read tools: with nobody to authorize for, nothing resolves.
            return {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}

        if getattr(ctx, "requester_is_human", False) is not True:
            # Same rule as the read gate: only a person's own view resolves names. (The gate
            # below would refuse every candidate anyway; checking here avoids paying for a
            # membership walk on another bot's behalf first.)
            return {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}

        # ONLY the requester's own conversations. There is deliberately no public-directory
        # pass: under the both-members rule every directory hit that isn't already in this
        # list gets refused by the gate below, so scanning conversations.list could not
        # produce a single extra ALLOWED result — it would only spend up to ten more API
        # calls and let a public-side page-cap turn a definitive "no such shared channel"
        # into a much vaguer "I couldn't finish looking".
        mine, complete = await self._requester_channel_matches(requester, want, ctx)
        candidates: Dict[str, Dict[str, Any]] = {e["id"]: e for e in mine}

        # Prove the BOT's side through the one authorization gate.
        allowed: List[Dict[str, Any]] = []
        examined = list(candidates.items())[:_MAX_CANDIDATES]
        if len(candidates) > _MAX_CANDIDATES:
            # Unexamined same-name candidates make BOTH answers unsafe: "not found" might be
            # wrong, and a single hit can't be called unique. Say the search was incomplete.
            self.log_warning(
                f"lookup_channel: {len(candidates)} candidates for {want!r} exceeds cap")
            complete = False
        for cid, entry in examined:
            ok, _reason = await self._channel_is_accessible(cid, ctx)
            if ok:
                allowed.append(entry)

        # DELIVERY-AUDIENCE GATE (Option B). An id we hand back is one the model will then read
        # from and speak into THIS reply's audience. In a multi-user surface only a deliverable
        # source (the current channel or a public-internal one) may be NAMED; a private / DM /
        # MPIM / foreign match — even one both of us are in — is withheld behind the SAME generic
        # refusal the read tools use, revealing neither its id, its name, its is_private flag, nor
        # that it exists. A DM surface is the asker's alone, so it is unchanged.
        if not bool(getattr(ctx, "is_dm", False)) and allowed:
            deliverable: List[Dict[str, Any]] = []
            withheld = False
            for entry in allowed:
                ok, _dreason = await self._delivery_allowed(entry["id"], ctx)
                if ok:
                    deliverable.append(entry)
                else:
                    withheld = True
            if not deliverable and withheld:
                # A real match exists but cannot be spoken into this channel's audience. Same
                # generic refusal as the read tools — no id, no name, no is_private — so the reply
                # never confirms the channel exists to everyone who can see it.
                self.log_warning(f"lookup_channel: match withheld by delivery gate for {want!r}")
                return {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}
            allowed = deliverable

        # A single hit is returned even from an incomplete walk: it is a channel we PROVED
        # the requester and bot share, and withholding it would deny every lookup to anyone
        # whose conversation list outruns the budget. Incompleteness only makes "no match"
        # unsafe to assert — that is handled below.
        if len(allowed) == 1:
            entry = allowed[0]
            return {"ok": True, "id": entry["id"], "name": entry.get("name"),
                    "is_private": bool(entry.get("is_private"))}
        if len(allowed) > 1:
            return {
                "ok": False,
                "error": "ambiguous",
                "candidates": [{"id": e["id"], "name": e.get("name")} for e in allowed],
                "message": ("More than one channel matches that name — ask which one is meant, "
                            "or pass the id."),
            }
        if not complete:
            # Never "no such channel": we ran out of pages/time, which is an availability
            # failure, not evidence about the workspace.
            return {"ok": False, "error": "incomplete", "message": _INCOMPLETE_MESSAGE}
        return {"ok": False, "error": "not_found",
                "message": ("No channel by that name is one you and the bot are both in. If it "
                            "exists, either you or the bot would need to be added.")}


def register_channel_lookup_tool(registry: Any, bot: Any) -> None:
    """Register lookup_channel (no-op unless the history tools are on — an id resolves to
    nothing without them)."""
    if not config.enable_history_tools:
        return
    # The explicit name is redundant for the real dict schema (register() reads schema["name"])
    # but keeps registration well-defined for any host whose schema getter isn't a plain dict.
    registry.register(bot.get_lookup_channel_tool_schema(), bot.execute_lookup_channel,
                      name="lookup_channel")
    # Inverse direction (id → name), for naming channel ids that arrive from search results.
    # Deliberately NOT in CHANNEL_READ_TOOLS: it resolves only a NAME (public names are
    # workspace-visible), self-authorizes with a public fast-path + the strict gate fallback,
    # and never returns content — so it stays off the strict content-dispatch entirely.
    registry.register(bot.get_resolve_channel_name_tool_schema(), bot.execute_resolve_channel_name,
                      name="resolve_channel_name")


__all__ = ["SlackChannelLookupToolMixin", "register_channel_lookup_tool"]
