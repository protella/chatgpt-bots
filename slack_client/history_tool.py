from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

from slack_sdk.errors import SlackApiError

from config import config
from slack_client.formatting.blocks import extract_supplementary_text


# Safety ceiling on how many conversations_replies pages a single thread fetch will
# pull (1000 messages/page). Threads deeper than this are effectively nonexistent, but
# the cap keeps a pathological thread from spinning the loop; when it's hit we say so.
_MAX_THREAD_PAGES = 10

# Every model-callable tool that returns content from a named conversation. dispatch
# refuses anything outside this set BEFORE routing, so a tool added to the schema list
# but not to this set cannot execute at all — the authorization gate can't be skipped by
# forgetting to wire it. Kept in lockstep with get_history_tools_for_openai() by
# tests/unit/test_channel_scope_guard.py.
CHANNEL_READ_TOOLS = frozenset({
    "fetch_channel_history",
    "fetch_thread_messages",
    "get_message_permalink",
    "fetch_channel_info",
    "fetch_pinned_messages",
})

# The ONE thing the model is told when content cannot be surfaced — for EITHER reason. A
# retrieval DENIAL (nonexistent channel, bot not a member, requester not a member, deactivated
# account, malformed payload, page-cap exhaustion, API error) AND a delivery REDIRECT (the
# requester MAY read it, but it can't be spoken into this reply's audience) return this exact
# string. It must be byte-identical across both, and free of counts/names/reasons: varying it
# (or leaking a `reason` field) turns the refusal into an existence oracle that answers "is
# there a private channel with this id?" one probe at a time, and a redirect that admitted "this
# is a conversation you and I share" would confirm a private conversation's existence to the
# whole channel audience. It stays honest for a redirect (a DM really does work) and safe for a
# denial (a DM re-refuses, nothing learned). Detailed reasons go to the LOG only.
ACCESS_DENIED_MESSAGE = (
    "I can't share that here. If it's a conversation you and I both have access to, ask me "
    "about it in a DM and I'll help there."
)

# ---- delivery-audience gate (Option B) source-classification signals ----
# A source is "public internal" — deliverable into a multi-user audience — ONLY when Slack's own
# booleans positively say so. These are the flags whose truthiness DISqualifies it: any
# external/cross-org share, or a limited-access / record-backed channel that isn't workspace-open.
_SHARED_SOURCE_FLAGS = ("is_shared", "is_ext_shared", "is_org_shared", "is_pending_ext_shared")
# Limited-access / record-backed channels are gated even when is_private is false, so "any
# workspace member can read it" does not hold. Append any equivalent signal here as one surfaces.
_LIMITED_ACCESS_FLAGS = ("is_limited_access",)
# Channel keys that can carry a workspace/team id; any that disagrees with the bot's own team
# means the channel isn't ours to treat as internal-public.
_SOURCE_TEAM_FIELDS = ("context_team_id", "team_id", "team")
# Recognizable conversation-type flags. A destination object that carries NONE of them is
# unclassifiable — we can't positively confirm it's a normal internal channel, so it is treated
# as current-source-only (fail closed) rather than trusting a merely-absent share flag.
_CONVERSATION_TYPE_FLAGS = ("is_channel", "is_group", "is_private", "is_mpim", "is_im")

# Membership oracle: the requester's own conversation list (users.conversations), which is
# authoritative for public, private, group-DM and DM membership in one paginated walk and
# does NOT require scanning a channel's full roster.
_USER_CONVOS_TYPES = "public_channel,private_channel,mpim,im"
_USER_CONVOS_PAGE_LIMIT = 200
# `limit` is a CEILING, not a page size: users.conversations filters server-side AFTER paging,
# so it returns far fewer rows per page than asked. Measured live 2026-07-22 against this
# workspace: 37 conversations arrived as 22/7/7/1 over four pages — ~9 per page at limit=200,
# ~92ms each. A 10-page cap would therefore cover only ~90 conversations and start denying any
# user in more channels than that, silently, as "unverified". The page cap is a runaway guard;
# the TIME budget is the real governor.
_USER_CONVOS_MAX_PAGES = 60
# Sub-budget inside the tool's own 20s timeout (config.tool_call_timeout). At the measured
# ~92ms/page this covers ~85 pages ≈ 800 conversations, and the walk is paid once per request
# (memoized), never per tool call.
_USER_CONVOS_TIME_BUDGET_S = 8.0


class SlackHistoryToolMixin:
    """On-demand Slack history-fetch tool (Phase 8).

    Lets the model deliberately pull a bounded slice of a thread's/channel's recent
    messages instead of front-loading everything. Privacy is enforced HERE, at the tool
    layer (never via prompt): content is returned ONLY when BOTH the bot AND the person who
    asked are in the target conversation. Bot membership alone used to be enough, which
    leaked two ways — a private channel the bot was in but the requester was not, and (worse)
    any DM of the bot's, because conversations.info omits `is_private` on an IM and the old
    `ch.get("is_private", False)` therefore classified someone else's DM as "public".

    Wired to the model through the local function-call loop (registered in
    SlackBot._build_tool_registry). Beyond history slices, this mixin also hosts the
    other on-demand workspace-context tools: message permalinks, channel info, and pinned
    messages — same privacy gate, same graceful-refusal contract. (User profiles moved
    to lookup_user in people_tools.py — F29.)
    """

    if TYPE_CHECKING:  # provided by the host (SlackBot) — declared so this mixin type-checks
        app: Any
        log_warning: Any
        log_error: Any
        log_info: Any
        log_debug: Any
        classify_sender: Any
        resolve_usernames: Any
        self_team_id: Any
        bot_user_id: Any

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
                    "Fetch a bounded slice of recent messages from a Slack conversation that BOTH "
                    "the person who asked AND the bot are in — any channel you are both members "
                    "of, or this DM. Anything else (a channel only one of you is in, someone "
                    "else's DM) is refused with no content, so don't retry it with a different "
                    "id. Use when you need more context than the current thread provides. "
                    "Each message includes "
                    "its current emoji reactions (who reacted with what). A message carrying "
                    "\"reply_count\" has a THREAD under it whose replies are NOT in this result; "
                    "when that discussion is relevant to what's being asked, pass that message's "
                    "ts as fetch_thread_messages' thread_ts to read it."
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
                    "Fetch messages from a specific Slack thread, in a conversation BOTH the "
                    "person who asked and the bot are in. Each message includes its current "
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
                    "Get a permanent Slack link to a specific message (by channel and message ts), "
                    "in a conversation BOTH the person who asked and the bot are in. "
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
                    "Get a channel's name, topic, purpose, member count, and privacy flag — for a "
                    "channel BOTH the person who asked and the bot are in. Use for questions "
                    "about what a channel is for or its basic facts."
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
                    "List the pinned messages (text, author, ts, permalink) of a channel BOTH the "
                    "person who asked and the bot are in. Pins usually "
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
            # fetch_user_profile retired (F29): lookup_user in people_tools.py subsumes it
            # (id OR name resolution, fresh users.info, disambiguation) and keeps the same
            # profile-card-only privacy stance (no email).
        ]

    # ------------------------------------------------------------------ authorization

    def _access_memo(self, ctx: Any) -> Optional[Dict[Any, Any]]:
        """The request-scoped memo dict on this ToolContext, created on first use.

        Returns None when there is nowhere safe to hang it (no context, or an attribute that
        isn't a dict — e.g. a MagicMock in a test): callers then simply re-verify, which is
        slower but never wrong. NOT a cross-request cache; see ToolContext for why."""
        if ctx is None:
            return None
        memo = getattr(ctx, "channel_access_memo", None)
        if isinstance(memo, dict):
            return memo
        if memo is not None:
            return None
        try:
            memo = {}
            ctx.channel_access_memo = memo
            return memo
        except Exception:
            return None

    def _access_scope(self) -> str:
        """Identity the memo keys are scoped by, so a decision can never be read back under a
        different workspace/token. `org_deploy_enabled` is false today, so team_id is the whole
        story; keyed anyway rather than assuming one workspace forever."""
        return str(getattr(self, "self_team_id", None) or getattr(self, "bot_user_id", None) or "-")

    async def _memoized_access(self, ctx: Any, key: Any,
                               factory: Callable[[], Any], default: Any) -> Any:
        """Single-flight: the FIRST caller runs `factory`, concurrent callers await its future.

        A round's tool calls run together under asyncio.gather, so a plain "compute then store"
        memo would still launch the same full pagination walk five times over and invite rate
        limiting. The in-flight future — not just the result — is what's shared. Any failure
        (including cancellation) resolves to `default`, so a waiter can never hang on a future
        nobody will complete, and the fail-closed default is what propagates."""
        memo = self._access_memo(ctx)
        if memo is None:
            try:
                return await factory()
            except Exception:
                return default
        pending = memo.get(key)
        if pending is not None:
            try:
                # SHIELDED: awaiting a bare future propagates the waiter's cancellation INTO
                # the shared future. One cancelled waiter (a tool round timing out) would
                # otherwise leave a cancelled future in the memo, and every later
                # authorization in this request would raise CancelledError — which is a
                # BaseException, so it escapes the handlers below and aborts the turn.
                return await asyncio.shield(pending)
            except asyncio.CancelledError:
                # This waiter is being cancelled; the shared future is untouched. Propagate
                # rather than swallowing a cancellation that belongs to the caller.
                raise
            except Exception:
                return default
        fut = asyncio.get_running_loop().create_future()
        memo[key] = fut
        try:
            result = await factory()
        except Exception:
            result = default
        except BaseException:
            if not fut.done():
                fut.set_result(default)
            raise
        if not fut.done():
            fut.set_result(result)
        return result

    async def _requester_conversation_entries(self, requester_user_id: str,
                                              ctx: Any = None) -> Tuple[List[Dict[str, Any]], bool]:
        """(conversation objects the requester is in, complete) — from users.conversations.

        `complete` is False when an API error, the page cap, or the time budget cut the walk
        short. A partial set can still PROVE membership (an id in it is in it), but it can
        never disprove it: "not found before the cap" is authorization-UNAVAILABLE, not
        evidence of non-membership, and both callers must keep those apart.

        Note users.conversations omits `is_member`, so an entry here proves only the
        REQUESTER's side — the bot's side is established separately from conversations.info.

        Returns whole conversation objects (not just ids) because lookup_channel matches on
        their names; both surfaces then share ONE walk per request through the memo.
        """

        async def _fetch() -> Tuple[List[Dict[str, Any]], bool]:
            found: List[Dict[str, Any]] = []
            cursor = ""
            deadline = time.monotonic() + _USER_CONVOS_TIME_BUDGET_S
            for _ in range(_USER_CONVOS_MAX_PAGES):
                kwargs: Dict[str, Any] = {
                    "user": requester_user_id,
                    "types": _USER_CONVOS_TYPES,
                    "limit": _USER_CONVOS_PAGE_LIMIT,
                    # Archived conversations still carry membership; excluding them would
                    # deny a legitimate read of an archived channel's history.
                    "exclude_archived": False,
                }
                if cursor:
                    kwargs["cursor"] = cursor
                try:
                    resp = await self.app.client.users_conversations(**kwargs)
                except SlackApiError as e:
                    err = e.response.get("error", "unknown") if getattr(e, "response", None) else str(e)
                    self.log_warning(
                        f"history_tool: users.conversations failed for {requester_user_id}: {err}")
                    return found, False
                except Exception as e:
                    self.log_warning(
                        f"history_tool: users.conversations error for {requester_user_id}: {e}")
                    return found, False
                if not resp or not resp.get("ok"):
                    err = (resp or {}).get("error") if resp else "empty_response"
                    self.log_warning(
                        f"history_tool: users.conversations not ok for {requester_user_id}: {err}")
                    return found, False
                page = resp.get("channels")
                if not isinstance(page, list):
                    self.log_warning(
                        f"history_tool: users.conversations malformed for {requester_user_id}")
                    return found, False
                for entry in page:
                    if isinstance(entry, dict) and isinstance(entry.get("id"), str):
                        found.append(entry)
                # A short page is never proof of the end — only an empty cursor is.
                cursor = (resp.get("response_metadata") or {}).get("next_cursor") or ""
                if not cursor:
                    return found, True
                if time.monotonic() >= deadline:
                    self.log_warning(
                        f"history_tool: users.conversations time budget hit for {requester_user_id}")
                    return found, False
            self.log_warning(
                f"history_tool: users.conversations page cap hit for {requester_user_id}")
            return found, False

        return await self._memoized_access(
            ctx, ("user_conversations", self._access_scope(), requester_user_id),
            _fetch, ([], False))

    async def _requester_conversations(self, requester_user_id: str,
                                       ctx: Any = None) -> Tuple[Set[str], bool]:
        """The id set of the above, for membership tests."""
        entries, complete = await self._requester_conversation_entries(requester_user_id, ctx)
        return {e["id"] for e in entries if isinstance(e.get("id"), str)}, complete

    async def _evaluate_channel_access(self, channel_id: Optional[str], requester_user_id: Optional[str],
                                       ctx: Any = None) -> Tuple[bool, str]:
        """The policy. Returns (allowed, reason); the reason is for LOGS, never the model.

        Type is decided by Slack's own booleans, never by an id prefix (private channels in
        this workspace carry C- prefixes), and the fields are read tolerantly — a real IM
        legitimately carries no `is_private` and no `is_member`.
        """
        if not channel_id:
            return False, "missing_channel"
        if not requester_user_id:
            # No requester (background job, replay, hand-built context) → public-only would
            # still be a user-scoped claim we cannot make. Nothing is readable.
            return False, "no_requester"

        # Slack delivered THIS turn's human message from THIS conversation, so the REQUESTER's
        # side is already proven by the event itself and needs no lookup. Attestation is
        # stamped only at genuine live entry points and never inherited by synthetic or
        # detached contexts, so a replayed channel id cannot claim it. It substitutes for the
        # membership WALK only — the bot's own side is still verified from conversations.info
        # below, because that check is one cheap call and the attestation chain is the single
        # most consequential shortcut in this design; it should not be the only thing standing
        # between a stamping bug and someone else's channel.
        attested = (getattr(ctx, "origin_membership_attested", False) is True
                    and getattr(ctx, "channel_id", None) == channel_id)

        if not attested:
            # Requester side FIRST, before conversations.info. Order matters for
            # confidentiality, not just cost: checking the bot's side first made a
            # nonexistent/bot-inaccessible id fail in one fast call while a real one the
            # requester lacked paid a multi-second walk, so response LATENCY answered "is the
            # bot in this channel?" even though the refusal payload was byte-identical. Walking
            # first makes every denial cost the same, and skips the info call on that path.
            # (No oracle is created for the attested channel: that is the conversation the
            # requester is demonstrably already in.)
            member_ids, complete = await self._requester_conversations(requester_user_id, ctx)
            if channel_id not in member_ids:
                # "Not found before the cap" is authorization-UNAVAILABLE, never proof of
                # non-membership — kept apart for the log, identical to the model either way.
                return False, ("requester_not_member" if complete
                               else "requester_membership_unverified")

        try:
            resp = await self.app.client.conversations_info(channel=channel_id)
        except SlackApiError as e:
            err = e.response.get("error", "unknown") if getattr(e, "response", None) else str(e)
            self.log_warning(f"history_tool: conversations_info failed for {channel_id}: {err}")
            return False, f"info_error:{err}"
        except Exception as e:
            self.log_warning(f"history_tool: access check error for {channel_id}: {e}")
            return False, "info_error"
        ch = (resp.get("channel") or {}) if resp else {}
        if not isinstance(ch, dict) or not ch:
            return False, "no_channel_info"

        is_im = ch.get("is_im") is True
        is_mpim = ch.get("is_mpim") is True
        if is_im and is_mpim:
            return False, "contradictory_type"

        if is_im:
            # A DM. Measured live 2026-07-22: users.conversations(user=X, types="im") on the
            # BOT token returns exactly ONE im — the bot's own DM with X — so an unattested
            # DM that reached here was already placed in the requester's list. We confirm the
            # participant directly anyway rather than inferring it: a DM is the
            # highest-consequence shape to misjudge (the old gate classified every IM as
            # "public", which made every DM the bot had readable by anyone who named its id).
            if ch.get("is_user_deleted") is True:
                # A deactivated account cannot start a fresh turn — only a stale or replayed
                # context can present one, so this is exactly where we want to fail closed.
                return False, "im_user_deleted"
            partner = ch.get("user")
            if isinstance(partner, str) and partner and partner == requester_user_id:
                return True, "im_participant"
            return False, "im_not_participant"

        # Channel, private channel, or group DM: BOTH sides must be members.
        is_member = ch.get("is_member")
        if is_member is not True:
            # Slack omits is_member on mpims; the bot token only surfaces a group DM it is
            # itself in, so for that ONE shape the requester-side check below carries both.
            if not (is_mpim and is_member is None):
                return False, "bot_not_member"
        elif not (is_mpim or ch.get("is_channel") is True or ch.get("is_group") is True
                  or ch.get("is_private") is not None):
            # Nothing identifies this as a conversation shape we understand.
            return False, "unrecognized_conversation"

        # Requester membership was established above — by the walk, or by Slack having
        # delivered this very turn from this conversation.
        return True, "origin_attested" if attested else "both_members"

    async def _channel_is_accessible(self, channel_id: Optional[str],
                                     ctx: Any = None) -> Tuple[bool, str]:
        """Authorization gate for every channel-read tool: (allowed, reason).

        Decided ONCE per (requester, channel) per request and memoized — authorization is
        fixed at request admission, so a mid-turn removal is an accepted TOCTOU window rather
        than a source of half-answered turns. Any failure anywhere → (False, ...), because the
        only safe answer to "I couldn't tell" is no."""
        requester = getattr(ctx, "user_id", None)
        if not isinstance(requester, str) or not requester:
            requester = None
        if requester is not None and getattr(ctx, "requester_is_human", False) is not True:
            # The policy is "conversations you and the bot are both in" — *you*, a person.
            # Another app's bot posting via its own bot token carries a real U… user id in
            # `event["user"]`, so without this its membership would satisfy the requester side:
            # bot A asks us, in channel Y, to read channel X that A and we share, and X's
            # content lands in front of whoever is in Y. A bot is not someone we owe a view of
            # a conversation to. Keyed on the CLASSIFIED sender (post `dev_treat_bot_ids_as_human`
            # allowlist), not on bot_id presence, so user-token dev testing still authorizes.
            return False, "requester_not_human"
        return await self._memoized_access(
            ctx, ("access", self._access_scope(), requester, channel_id),
            lambda: self._evaluate_channel_access(channel_id, requester, ctx),
            (False, "error"))

    def _access_denied(self, channel_id: Optional[str], reason: str, surface: str) -> Dict[str, Any]:
        """The single refusal payload. Identical for every reason and every tool — the model
        must not be able to tell a nonexistent channel from a forbidden one."""
        self.log_warning(
            f"history_tool: {surface} denied for channel={channel_id or '-'} reason={reason}")
        return {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}

    # ------------------------------------------------------------- delivery-audience gate
    #
    # A SECOND layer, ON TOP of retrieval (_channel_is_accessible). Retrieval answers "may the
    # REQUESTER read this conversation?"; delivery answers "may that content be spoken into the
    # CURRENT reply's audience?". They are different questions: a DM the requester and bot share
    # passes retrieval, but reading it aloud INTO a channel would expose it to everyone in that
    # channel. Option B (owner-chosen 2026-07-23): in a DM the audience is the asker alone, so
    # FULL power; in any multi-user surface only the CURRENT channel or a PUBLIC INTERNAL source
    # may be delivered — everything else is withheld behind the same generic refusal as a denial.
    #
    # Caching: the source-public and destination-classification verdicts ride the SAME
    # request-scoped single-flight memo (channel_access_memo) under their own namespaced keys, so
    # they die with the request and never become a TTL cache. The accepted within-request TOCTOU
    # is the same bounded window as the retrieval gate's: a public→private flip mid-turn can leak
    # through the cached verdict until the request ends. Keys carry the source's team id, so a
    # verdict is never reused across workspaces for the same channel id.

    def _bot_team_id(self) -> Optional[str]:
        """The bot's own workspace/team id (auth_test's team_id). None when unknown — and an
        unknown bot team makes every team comparison fail SAFE (a source we can't prove is ours
        is never treated as internal-public)."""
        tid = getattr(self, "self_team_id", None)
        return tid if isinstance(tid, str) and tid else None

    def _classify_source_public(self, ch: Any, ctx: Any,
                                source_team_id: Optional[str] = None) -> bool:
        """True ONLY for a channel we can POSITIVELY classify as public-and-internal.

        Failing safe is the whole point: a MISSING field is never read as "public". Slack sends a
        genuine public channel `is_private: false` EXPLICITLY (verified live), so requiring the
        key present-and-False costs nothing real while refusing to guess on a malformed or partial
        payload — including an IM, whose info carries no is_private at all. One classifier, used
        by every surface, so the rule can never drift between history, search and lookup."""
        if not isinstance(ch, dict) or not ch:
            return False
        if ch.get("is_channel") is not True:
            return False
        # Present AND False — NOT merely `is not True`. None/missing → not public (codex r3 #3).
        if ch.get("is_private") is not False:
            return False
        if ch.get("is_im") is True or ch.get("is_mpim") is True:
            return False
        # Any external / cross-org share disqualifies: those members are not "the workspace".
        if any(ch.get(flag) for flag in _SHARED_SOURCE_FLAGS):
            return False
        # Limited-access / record-backed channels are NOT workspace-public even when is_private is
        # false (codex r3 #1): membership there is gated, so "any workspace member can read it"
        # does not hold.
        if any(ch.get(flag) for flag in _LIMITED_ACCESS_FLAGS):
            return False
        # Same-workspace: without knowing our OWN team we cannot prove any channel is
        # workspace-internal, so an unknown bot team refuses to classify anything as public. That
        # honors _bot_team_id's fail-safe contract (codex r4): self_team_id is set from auth_test at
        # startup, so this only trips on a broken-auth bot — where public-nothing is correct.
        bot_team = self._bot_team_id()
        if not bot_team:
            return False
        # A hit's own team_id (threaded from a search result) OR any team id the channel object
        # carries must agree with ours; a single disagreement → not internal (codex r3 #4).
        present: List[str] = []
        if isinstance(source_team_id, str) and source_team_id:
            present.append(source_team_id)
        for key in _SOURCE_TEAM_FIELDS:
            val = ch.get(key)
            if isinstance(val, str) and val:
                present.append(val)
        shared_team_ids = ch.get("shared_team_ids")
        if isinstance(shared_team_ids, list):
            present.extend(t for t in shared_team_ids if isinstance(t, str) and t)
        if any(t != bot_team for t in present):
            return False
        return True

    async def _source_is_public(self, target: Optional[str], ctx: Any,
                                source_team_id: Optional[str] = None) -> bool:
        """Is `target` a public-internal channel? Memoized per (scope, team_id, channel).

        conversations.info SUCCEEDS on a public channel the bot isn't in (returns is_private:false
        explicitly) and returns channel_not_found on a private one it isn't in — so a non-member's
        public read classifies True while a private one fails closed. The memo is keyed by team_id
        too, so a same-channel-id verdict is never reused across workspaces (codex r3 #4). Routed
        through the SAME single-flight the retrieval memo uses; the fail-closed default (False)
        flows as the future's resolved value, never a plain bool the awaiting mechanism would
        mishandle (codex point 6)."""
        if not target:
            return False
        key = ("source_public", self._access_scope(), source_team_id or "-", target)
        return await self._memoized_access(
            ctx, key, lambda: self._eval_source_public(target, ctx, source_team_id), False)

    async def _eval_source_public(self, target: str, ctx: Any,
                                  source_team_id: Optional[str]) -> bool:
        try:
            resp = await self.app.client.conversations_info(channel=target)
        except SlackApiError as e:
            err = e.response.get("error", "unknown") if getattr(e, "response", None) else str(e)
            # A private channel the bot isn't in answers channel_not_found here — exactly the
            # fail-closed case. Logged, never surfaced.
            self.log_warning(f"history_tool: source_public info failed for {target}: {err}")
            return False
        except Exception as e:
            self.log_warning(f"history_tool: source_public error for {target}: {e}")
            return False
        ch = (resp.get("channel") or {}) if resp else {}
        return self._classify_source_public(ch, ctx, source_team_id)

    async def _destination_forces_current_only(self, ctx: Any) -> bool:
        """True when the CURRENT reply's channel is external/cross-org (or can't be classified),
        so ONLY current-channel content may be delivered — closing an internal→external-org leak
        (codex r3 #2). One memoized conversations.info on ctx.channel_id (NOT a roster scan);
        undetermined FAILS CLOSED. Costs no internal capability: a normal internal channel returns
        all-shared-flags-false → False, and delivery proceeds to the public-source check."""
        channel_id = getattr(ctx, "channel_id", None)
        if not channel_id:
            return True  # no destination we can classify → fail closed
        key = ("dest_current_only", self._access_scope(), channel_id)
        return await self._memoized_access(
            ctx, key, lambda: self._eval_dest_forces_current_only(channel_id, ctx), True)

    async def _eval_dest_forces_current_only(self, channel_id: str, ctx: Any) -> bool:
        try:
            resp = await self.app.client.conversations_info(channel=channel_id)
        except SlackApiError as e:
            err = e.response.get("error", "unknown") if getattr(e, "response", None) else str(e)
            self.log_warning(f"history_tool: dest info failed for {channel_id}: {err}")
            return True   # can't determine → fail closed
        except Exception as e:
            self.log_warning(f"history_tool: dest info error for {channel_id}: {e}")
            return True
        ch = (resp.get("channel") or {}) if resp else {}
        if not isinstance(ch, dict) or not ch:
            return True   # malformed → fail closed
        if any(ch.get(flag) for flag in _SHARED_SOURCE_FLAGS):
            return True   # externally / org shared destination
        # POSITIVE confirmation before trusting a "False": a nonempty-but-partial object that
        # carries none of the recognizable type flags is unclassifiable — an absent is_shared may
        # be an omission, not a guarantee — so it is current-source-only (codex r4).
        if all(ch.get(flag) is None for flag in _CONVERSATION_TYPE_FLAGS):
            return True
        # Without our OWN team id we can't prove the destination is same-workspace → current-only
        # (codex r4: the cross-team check must not fall through to unrestricted when it's unknown).
        bot_team = self._bot_team_id()
        if not bot_team:
            return True
        # Cross-team destination: any team id on the channel disagreeing with ours forces
        # current-source-only, even if the shared flags happen to be unset.
        team_ids: List[str] = []
        for key in _SOURCE_TEAM_FIELDS:
            val = ch.get(key)
            if isinstance(val, str) and val:
                team_ids.append(val)
        shared_team_ids = ch.get("shared_team_ids")
        if isinstance(shared_team_ids, list):
            team_ids.extend(t for t in shared_team_ids if isinstance(t, str) and t)
        if any(t != bot_team for t in team_ids):
            return True
        return False

    async def _delivery_allowed(self, target_channel_id: Optional[str], ctx: Any,
                                source_team_id: Optional[str] = None) -> Tuple[bool, str]:
        """(deliverable, reason). The reason is for LOGS only — never the model (see redirect)."""
        if getattr(ctx, "is_dm", False):
            # A true 1:1 IM: the audience is the asker alone, so retrieval already settled it.
            return True, "dm_surface"
        # Team identity is checked BEFORE the current-channel exemption (codex r3 #4): a search
        # hit claiming target == ctx.channel_id but carrying a foreign team_id must not ride it.
        if source_team_id and source_team_id != self._bot_team_id():
            return False, "cross_workspace"
        if target_channel_id and target_channel_id == getattr(ctx, "channel_id", None):
            return True, "current_channel"
        if await self._destination_forces_current_only(ctx):
            return False, "restricted_destination"
        if await self._source_is_public(target_channel_id, ctx, source_team_id):
            return True, "public_source"
        return False, "private_source_withheld"

    async def _authorize_channel_read(self, channel_id: Optional[str],
                                      ctx: Any = None) -> Tuple[str, str]:
        """Tri-state façade over retrieval + delivery: (verdict, reason), verdict ∈
        {ALLOW, DENY, REDIRECT}. Retrieval is unchanged and runs first — DENY when it refuses;
        otherwise REDIRECT when the content can't be spoken into this audience, else ALLOW."""
        allowed, reason = await self._channel_is_accessible(channel_id, ctx)
        if not allowed:
            return "DENY", reason
        deliverable, dreason = await self._delivery_allowed(channel_id, ctx)
        return ("ALLOW", dreason) if deliverable else ("REDIRECT", dreason)

    def _delivery_redirect(self, channel_id: Optional[str], reason: str,
                           surface: str) -> Dict[str, Any]:
        """Refusal for a REDIRECT — content the requester MAY read but that can't go to THIS
        audience. BYTE-IDENTICAL to _access_denied's payload: the destination audience (not the
        requester) is who must learn nothing, so a redirect has to be indistinguishable from a
        denial — no counts, no names, no 'this is a private conversation you share' confirmation.
        Only the LOGGED reason differs."""
        self.log_warning(
            f"history_tool: {surface} redirected for channel={channel_id or '-'} reason={reason}")
        return {"ok": False, "error": "not_accessible", "message": ACCESS_DENIED_MESSAGE}

    def _clamp_limit(self, limit: Optional[int]) -> int:
        cap = config.history_tool_max_messages
        if not limit:
            return cap
        try:
            return max(1, min(int(limit), cap))
        except (TypeError, ValueError):
            return cap

    def _text_with_supplementary(self, msg: Dict[str, Any]) -> str:
        """F48: a fetched message's text PLUS whatever Slack delivered outside it —
        table blocks, unfurls, quoted messages, webhook attachment fields. Without this,
        a message the model fetches by tool reads as empty when its content was never in
        `text`. Mentions are left as Slack sent them (this surface never cleaned them).
        Skipped for our own messages — our cards live in these fields (F47)."""
        text = msg.get("text", "") or ""
        try:
            if self.classify_sender(msg) == "self":
                return text
        except Exception:
            pass  # identity not wired -> fall through; extraction is still fail-open
        supplementary = extract_supplementary_text(msg, primary_text=text)
        if not supplementary:
            return text
        return f"{text}\n\n{supplementary}" if text.strip() else supplementary

    async def fetch_history_tool(
        self, channel_id: Optional[str], limit: Optional[int] = None,
        thread_ts: Optional[str] = None,
        ctx: Any = None,
    ) -> Dict[str, Any]:
        """Authorized fetch. Returns a structured dict; on refusal/error contains NO content.

        Authorizes itself rather than trusting the dispatcher, so a direct caller (a future
        code path, a test, a refactor) cannot reach content by skipping the gate. The decision
        is memoized on `ctx`, so the dispatcher's check and this one cost one lookup, not two.
        """
        n = self._clamp_limit(limit)
        verdict, reason = await self._authorize_channel_read(channel_id, ctx)
        if verdict == "DENY":
            return self._access_denied(channel_id, reason, "fetch_history")
        if verdict == "REDIRECT":
            return self._delivery_redirect(channel_id, reason, "fetch_history")
        thread_truncated = False
        try:
            if thread_ts:
                # conversations_replies returns ASCENDING from the thread root, so a
                # bare limit=n would keep the OLDEST n while the note below promises the
                # newest. Page to the END of the thread (bounded by _MAX_THREAD_PAGES,
                # following the response cursor) and keep the NEWEST n — the root for
                # context, then the most recent replies — so "recent messages" is true.
                # A single page tops out at 1000, so a thread over that would otherwise
                # return old messages under a "newest window" label.
                all_messages: List[Dict[str, Any]] = []
                cursor: Optional[str] = None
                for _ in range(_MAX_THREAD_PAGES):
                    kwargs: Dict[str, Any] = {"channel": channel_id, "ts": thread_ts, "limit": 1000}
                    if cursor:
                        kwargs["cursor"] = cursor
                    resp = await self.app.client.conversations_replies(**kwargs)
                    all_messages.extend(resp.get("messages") or [])
                    cursor = (resp.get("response_metadata") or {}).get("next_cursor")
                    if not cursor:
                        break
                else:
                    # Ran the page cap without draining the cursor: the true newest replies
                    # are beyond our reach, so the tail below is the newest of what we SAW.
                    thread_truncated = bool(cursor)
                if len(all_messages) > n:
                    # Root (all_messages[0]) + newest (n-1) replies; ascending order kept.
                    raw = all_messages[:1] + all_messages[-(n - 1):] if n > 1 else all_messages[:1]
                else:
                    raw = all_messages
            else:
                # conversations_history is DESCENDING (newest first), so the first n are
                # already the newest.
                resp = await self.app.client.conversations_history(channel=channel_id, limit=n)
                all_messages = resp.get("messages") or []
                raw = all_messages[:n]
            # BF2: render human authors by display name, not a raw Slack id (Slack is the only
            # transcript, so a fetched author must read the same as the live path). Resolve the
            # whole bounded result set in ONE read-only, budgeted batch — reading history must
            # not create user rows or bump last_seen. An unresolved id stays raw.
            api_client = getattr(getattr(self, "app", None), "client", None)
            resolver = getattr(self, "resolve_usernames", None)
            # Ordered dedup in result order (Blocker 2): a hash-ordered set would let the remote
            # budget resolve a different subset across cold starts.
            author_ids = list(dict.fromkeys(
                m.get("user") for m in raw if m.get("user") and not m.get("bot_id")))
            name_map = {}
            if author_ids and resolver:
                try:
                    name_map = await resolver(author_ids, api_client)
                except Exception:
                    name_map = {}
            messages = []
            for m in raw:
                author = m.get("user") or m.get("username") or ("bot" if m.get("bot_id") else "unknown")
                if m.get("user") and not m.get("bot_id"):
                    author = name_map.get(m.get("user"), author)
                entry = {
                    "user": author,
                    "ts": m.get("ts"),
                    "text": self._text_with_supplementary(m),
                }
                # A thread hangs off this message: without it, a parent with forty replies
                # reads exactly like a dead one-liner and the model can't tell there is
                # anything to fetch. Fresh from the API on every call (never a stored
                # snapshot), so the count is accurate at fetch time; pass this `ts` as
                # fetch_thread_messages' thread_ts to read the replies.
                if m.get("reply_count"):
                    entry["reply_count"] = m["reply_count"]
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
                thread_truncated
                or (resp.get("response_metadata") or {}).get("next_cursor")
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
                # Be honest about which kind of "more" this is: a normal trim (we have the
                # true newest) vs. a thread so long we couldn't page to its actual end.
                result["note"] = (
                    "This thread is longer than the tool can page through; the newest "
                    "window returned may not include the most recent messages."
                    if thread_truncated
                    else "Only the newest window was returned; older history exists beyond this."
                )
            return result
        except SlackApiError as e:
            err = e.response.get("error", "unknown") if getattr(e, "response", None) else str(e)
            self.log_warning(f"history_tool: fetch failed for {channel_id}: {err}")
            return {"ok": False, "error": err, "message": f"Could not fetch history: {err}"}
        except Exception as e:
            self.log_error(f"history_tool: unexpected error for {channel_id}: {e}", exc_info=True)
            return {"ok": False, "error": "exception", "message": "Could not fetch history."}

    async def get_message_permalink_tool(self, channel_id: Optional[str], message_ts: str,
                                         ctx: Any = None) -> Dict[str, Any]:
        """Permanent link to one message. Same gate as history — a permalink is a pointer
        into a conversation, so it is only minted for one both of us are in AND may deliver here."""
        verdict, reason = await self._authorize_channel_read(channel_id, ctx)
        if verdict == "DENY":
            return self._access_denied(channel_id, reason, "permalink")
        if verdict == "REDIRECT":
            return self._delivery_redirect(channel_id, reason, "permalink")
        try:
            resp = await self.app.client.chat_getPermalink(channel=channel_id, message_ts=message_ts)
            return {"ok": True, "channel": channel_id, "message_ts": message_ts,
                    "permalink": resp.get("permalink")}
        except SlackApiError as e:
            err = e.response.get("error", "unknown") if getattr(e, "response", None) else str(e)
            self.log_warning(f"history_tool: permalink failed for {channel_id}/{message_ts}: {err}")
            return {"ok": False, "error": err, "message": f"Could not get a permalink: {err}"}

    async def fetch_channel_info_tool(self, channel_id: Optional[str],
                                      ctx: Any = None) -> Dict[str, Any]:
        """Channel facts (name/topic/purpose/member count). Gated like history — a channel's
        name and purpose are content too."""
        verdict, reason = await self._authorize_channel_read(channel_id, ctx)
        if verdict == "DENY":
            return self._access_denied(channel_id, reason, "channel_info")
        if verdict == "REDIRECT":
            return self._delivery_redirect(channel_id, reason, "channel_info")
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

    async def fetch_pinned_messages_tool(self, channel_id: Optional[str],
                                         ctx: Any = None) -> Dict[str, Any]:
        """Pinned items for a channel. Authorization-gated; degrades with a clear message if
        the app is missing the pins:read scope (added to the manifest 2026-07-10)."""
        verdict, reason = await self._authorize_channel_read(channel_id, ctx)
        if verdict == "DENY":
            return self._access_denied(channel_id, reason, "pins")
        if verdict == "REDIRECT":
            return self._delivery_redirect(channel_id, reason, "pins")
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
                    "text": self._text_with_supplementary(msg),
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

        # Unknown names die here, BEFORE the gate — and, more importantly, a tool that is not
        # in CHANNEL_READ_TOOLS can never be routed below, so the authorization check cannot be
        # bypassed by adding a schema and forgetting to gate it.
        if name not in CHANNEL_READ_TOOLS:
            return {"ok": False, "error": "unknown_tool", "message": f"Unknown history tool: {name}"}

        channel_id = args.get("channel_id") or getattr(ctx, "channel_id", None)

        # THE gate: one decision per (requester, channel) per request, applied to every tool
        # uniformly — including the current channel, which is exempted from RETRIEVAL only by an
        # explicit attestation on the context, never by comparing ids. The tri-state façade adds
        # the DELIVERY layer on top: DENY (requester can't read) and REDIRECT (requester can read
        # but it can't be delivered into this audience) are byte-identical to the model.
        verdict, reason = await self._authorize_channel_read(channel_id, ctx)
        if verdict == "DENY":
            return self._access_denied(channel_id, reason, name)
        if verdict == "REDIRECT":
            return self._delivery_redirect(channel_id, reason, name)

        if name == "fetch_channel_history":
            return await self.fetch_history_tool(channel_id, args.get("limit"), ctx=ctx)
        if name == "fetch_thread_messages":
            thread_ts = args.get("thread_ts") or getattr(ctx, "thread_ts", None)
            if not thread_ts:
                return {"ok": False, "error": "bad_arguments",
                        "message": "No thread here — pass thread_ts or use fetch_channel_history."}
            return await self.fetch_history_tool(channel_id, args.get("limit"), thread_ts, ctx=ctx)
        if name == "get_message_permalink":
            return await self.get_message_permalink_tool(channel_id, args.get("message_ts"), ctx=ctx)
        if name == "fetch_channel_info":
            return await self.fetch_channel_info_tool(channel_id, ctx=ctx)
        if name == "fetch_pinned_messages":
            return await self.fetch_pinned_messages_tool(channel_id, ctx=ctx)
        # In CHANNEL_READ_TOOLS but unrouted: a half-wired tool, refused rather than guessed.
        return {"ok": False, "error": "unknown_tool", "message": f"Unknown history tool: {name}"}
