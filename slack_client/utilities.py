from __future__ import annotations

import aiohttp
import asyncio
import logging
import re
import time
from contextlib import contextmanager
from typing import (TYPE_CHECKING, AbstractSet, Any, Dict, Iterable, Iterator, List,
                    Optional, Set)

from slack_sdk.errors import SlackApiError

from config import config
from slack_client._host import _Host

logger = logging.getLogger(__name__)


# Author ids that users.info can never resolve, so looking them up only buys a 404 and a
# wasted slot in the resolver's remote budget:
#   "U00" — the placeholder assistant.search.context returns as the author of some results
#           (observed live; users.info answers user_not_found).
#   "B…"  — a BOT object id. A bot's USER identity is a separate U/W id; passing the B id is
#           simply the wrong argument for users.info.
# Deliberately NOT a length or character-class check: Slack has lengthened ids before and
# tells apps not to assume their length or composition, so a `^[UW][A-Z0-9]{8,11}$` guard
# would start silently dropping real users the day ids grow again.
_SENTINEL_USER_IDS = frozenset({"U00"})


def _is_resolvable_user_id(uid: str) -> bool:
    """False for ids users.info cannot resolve by construction (see above)."""
    if uid in _SENTINEL_USER_IDS:
        return False
    return not uid.startswith("B")


def is_user_shaped_id(uid: Optional[str]) -> bool:
    """True only for an id that names a Slack USER — `U…`, or `W…` on Enterprise Grid.

    THE one test for "can this id be mentioned, or looked up as a person": a bot OBJECT id (B…)
    and an app id (A…) can be neither. A prefix test only, never a length or character-class one,
    for the reason given above _SENTINEL_USER_IDS: Slack lengthens ids."""
    return bool(uid) and str(uid)[0].upper() in ("U", "W")


# The bot_id -> user_id cache's size bound. Mirrors PARTICIPATION_ACTIVITY_LRU_MAX's default
# (config.participation_activity_lru_max = 1024) rather than inventing a number: same kind of
# quantity — a per-process id map that is tiny in practice and must not be unbounded in theory.
# Not configurable, because unlike the activity LRU nothing about a workspace's bot count varies
# enough to tune.
BOT_USER_ID_CACHE_MAX = 1024


def is_dm_conversation(channel_id: Optional[str], channel_type: Optional[str] = None) -> bool:
    """True for a 1:1 DM surface, False for a channel-shaped one (spec §8 surface ruling).

    The permanent DM/channel discriminator: DMs keep the legacy layout and tool surface and get
    no receipts or index; "mpim" is channel-shaped and already routes that way today.

    A "U" prefix counts as DM because outbound DMs are posted with channel=<user_id>. An
    unrecognized id falls to the DM side and logs: the fail-safe for a surface we cannot
    classify is "no receipt row for it", never a bogus row attributed to a channel.
    """
    kind = (channel_type or "").strip().lower()
    if kind == "im":
        return True
    if kind in ("channel", "group", "mpim", "private_channel", "public_channel"):
        return False

    cid = (channel_id or "").strip()
    if not cid:
        logger.warning("is_dm_conversation: empty conversation id; treating as DM")
        return True
    prefix = cid[0].upper()
    if prefix in ("D", "U", "W"):
        return True
    if prefix in ("C", "G"):
        return False
    logger.warning(
        f"is_dm_conversation: unrecognized conversation id {cid!r}; treating as DM")
    return True


def strip_citations(text: str) -> str:
    """
    Strip OpenAI Responses API citation markers from text.

    OpenAI's Responses API automatically adds citation markers (cite:...:) when incorporating
    tool results (MCP, web_search, etc.) into responses. These render as clickable links in
    ChatGPT's web UI but appear as garbage text in other clients like Slack.

    Note: MCP servers return clean data - the cite markers are added by OpenAI, not the servers.

    Citation formats to REMOVE (MCP tool results):
    - cite:emoji:mcp_<server>.<tool>result<N>:emoji:
    - cite:emoji:turn<N>:emoji: (simple tool reference)
    - cite:emoji:turn<N>read_documentation:emoji:

    Citation formats to PRESERVE (web search - these may render as links):
    - cite:emoji:turn<N>search<N>:emoji:

    Args:
        text: Text potentially containing citation markers

    Returns:
        Text with MCP/tool citation markers removed, web search citations preserved
    """
    # Patterns for OpenAI-added citation markers to remove:
    # These are added by OpenAI's Responses API to attribute tool call results.
    # Format: cite:emoji:reference:emoji: where reference identifies the source.
    # We preserve web_search citations (turn<N>search<N>) as they may render as links.
    tool_citation_patterns = [
        # MCP server references: cite:ship:mcp_aws_knowledge.tool_name...:walking:
        r'\s*cite:[^:]+:mcp_[^:]+(?::[^:]+)*:[^:]+:\s*',
        # Tool action patterns: read_, get_, list_, fetch_, retrieve_
        r'\s*cite:[^:]+:[^:]*(?:read_|get_|list_|fetch_|retrieve_)[^:]+(?::[^:]+)*:[^:]+:\s*',
        r'\s*cite:[^:]+:[^:]*(?:_documentation|_library|_docs)[^:]*(?::[^:]+)*:[^:]+:\s*',
        # Nested citations (MCP + web search mixed in same marker)
        r'\s*cite:[^:]+:turn\d+search\d+:[^:]+:turn\d+[^:]*(?::[^:]+)*:[^:]+:\s*',
        # Simple tool citations: turn<N> without search (e.g., cite:ship:turn0:walking:)
        # Note: turn<N>search<N> = web search (preserved), turn<N> alone = tool result (removed)
        r'\s*cite:[^:]+:turn\d+:[^:]+:\s*',
    ]

    cleaned_text = text
    for pattern in tool_citation_patterns:
        cleaned_text = re.sub(pattern, ' ', cleaned_text)

    # Clean up any double spaces created by removing citations (preserve newlines)
    cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)

    return cleaned_text


# THE ONE DEFINITION OF THE REMOTE-LOOKUP BUDGET. It lives here, beside the resolver that
# spends it, because a second literal in a caller would be two definitions of one budget — and
# the day someone raised the cap, a caller computing "the remainder" would hand out lookups the
# resolver would not honour.
ACTOR_REMOTE_LOOKUP_DEFAULT = 25


class SlackUtilitiesMixin(_Host):
    if TYPE_CHECKING:
        # Owned by SlackBot.__init__ and filled in by _ensure_self_identity below; not part
        # of the host contract because only this mixin reads them.
        bot_user_id: Optional[str]
        bot_id: Optional[str]
        bot_handle: Optional[str]
        self_team_id: Optional[str]
        # bot OBJECT id -> that bot's USER id, or None when Slack definitively has none. See
        # _bot_user_ids() for why the negative entry is as important as the positive one, and
        # _bot_id_flights() for the per-key single flight that fills them.
        _bot_user_id_cache: Dict[str, Optional[str]]
        _bot_user_id_flights: Dict[str, Any]
        _bot_user_id_pins: Dict[str, int]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._session = None  # Reusable aiohttp session

    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a reusable aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _cleanup_session(self):
        """Clean up aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self.log_debug("SlackUtilities aiohttp session closed")

    async def _ensure_self_identity(self) -> None:
        """Resolve and cache the bot's own identity (user_id + bot_id) via auth_test.

        Called once on start. Idempotent and best-effort: failures are logged and leave the
        identity unset (callers degrade gracefully). app_id is left for opportunistic capture
        from inbound events (auth.test does not return it)."""
        if self.bot_user_id:
            return
        try:
            resp = await self.app.client.auth_test()
            if resp.get("ok"):
                self.bot_user_id = resp.get("user_id")
                self.bot_id = resp.get("bot_id")
                # The handle people actually type ("chatgpt-dev"). Free — auth.test already
                # returns it. The participation gate is shown only config's BOT_NAME_ALIASES,
                # and when the deployed handle carries an environment suffix the alias list
                # doesn't mention, the gate concluded a message addressed to it was addressed
                # to a DIFFERENT assistant and stayed silent. Config says who it answers to;
                # this says who it actually IS.
                self.bot_handle = resp.get("user")
                # team_id is required by chat.startStream (recipient_team_id) for
                # channel streaming; stash it here from the same auth.test call.
                self.self_team_id = resp.get("team_id")
                self.log_info(f"Resolved bot self-identity: user_id={self.bot_user_id}, "
                              f"bot_id={self.bot_id}, handle={self.bot_handle!r}")
            else:
                self.log_warning(f"auth_test returned not ok: {resp.get('error')}")
        except Exception as e:
            self.log_warning(f"Could not resolve bot self-identity via auth_test: {e}")

    def is_own_message(self, msg: dict) -> bool:
        """True if a Slack event/history-message dict was posted by this bot itself."""
        if not isinstance(msg, dict):
            return False
        if self.bot_id and msg.get("bot_id") == self.bot_id:
            return True
        if self.bot_user_id and msg.get("user") == self.bot_user_id:
            return True
        if self.app_id and (msg.get("app_id") == self.app_id or msg.get("api_app_id") == self.app_id):
            return True
        return False

    def classify_sender(self, msg: dict) -> str:
        """Classify a Slack event/history-message dict as 'self', 'other_bot', or 'human'.

        Keys on bot_id/app_id PRESENCE (not subtype == 'bot_message', which misses
        app-posted messages)."""
        if not isinstance(msg, dict):
            return "human"
        if self.is_own_message(msg):
            return "self"
        if msg.get("bot_id") or msg.get("app_id") or msg.get("api_app_id"):
            # Dev harness carve-out: user-token (xoxp) posts carry the app's bot_id even
            # though a human authored them; the allowlist (empty in prod) restores that.
            if str(msg.get("bot_id") or "") in (config.dev_treat_bot_ids_as_human or []):
                return "human"
            return "other_bot"
        return "human"

    # --- bot OBJECT id -> bot USER id ----------------------------------------------------------
    #
    # A peer app posting in "agent mode" sends `subtype: bot_message` with a `username` override,
    # `bot_id` + `app_id`, and NO `user` field at all — so every id we can name it by is a B id.
    # A B id is not a mention target: `<@B…>` renders as raw text in the channel. The same app
    # posting in its other mode carries its real bot USER id (a U id) and everything downstream
    # already works, so the fix is to learn the U id for a B id and use it everywhere the B id
    # would otherwise have stood in for an actor.
    #
    # bots.info answers that question, and the answer never changes for the life of a bot, so it
    # is cached for the process. A DEFINITIVE "this bot has no user id" is cached too — that bot
    # is re-encountered on every message it posts, and the negative entry is the difference
    # between one wasted call and one per message forever. A TRANSIENT failure (an exception: a
    # timeout, a rate limit, a socket) is NOT cached: caching it would blind the process to a
    # perfectly resolvable bot until restart. The per-batch budget below is what stops the retry
    # that implies from becoming a hammer.

    def _bot_user_ids(self) -> Dict[str, Optional[str]]:
        """The per-process bot_id -> user_id map, created on first use."""
        cache = getattr(self, "_bot_user_id_cache", None)
        if cache is None:
            cache = {}
            self._bot_user_id_cache = cache
        return cache

    def _bot_id_pins(self) -> Dict[str, int]:
        """Reference counts of bot ids currently pinned against eviction — one entry per batch
        that named the id, because two batches can pin the same id and the second must not be
        unpinned by the first one finishing."""
        pins = getattr(self, "_bot_user_id_pins", None)
        if pins is None:
            pins = {}
            self._bot_user_id_pins = pins
        return pins

    @contextmanager
    def _pinned_bot_ids(self, keys: AbstractSet[str]) -> Iterator[None]:
        """Pin a batch's ids for the WHOLE prime call, waiting included.

        The register is shared, not per-call, because a batch spends most of its time waiting on
        somebody else's flight: a `protect` set carried only on its own resolve calls is not
        consulted when the OTHER batch's resolution writes the cache, and the waiting batch's
        already-known entry can be evicted out from under it."""
        pins = self._bot_id_pins()
        for key in keys:
            pins[key] = pins.get(key, 0) + 1
        try:
            yield
        finally:
            for key in keys:
                remaining = pins.get(key, 1) - 1
                if remaining > 0:
                    pins[key] = remaining
                else:
                    pins.pop(key, None)

    def _cache_bot_user_id(self, key: str, user_id: Optional[str]) -> None:
        """Write one entry, size-bounded, and NEVER over a positive one.

        A resolved U id is permanent truth about that bot; a later miss (a rate-limited retry, a
        bots.info that answered without the field) is not news, and letting it win would send an
        already-correct actor back to being named by its B id mid-process."""
        cache = self._bot_user_ids()
        if cache.get(key):
            return
        if key not in cache:
            # Oldest insertion first — dicts preserve it. Pinned ids are skipped, so a batch
            # cannot evict an entry it (or a batch waiting beside it) already counted as known;
            # when everything evictable is gone the bound is briefly exceeded instead. It is a
            # PLATEAU, not a new ceiling: this drains back down to the cap on the next write that
            # finds anything evictable, rather than treating the overflow as the size from then on.
            pins = self._bot_id_pins()
            for victim in list(cache):
                if len(cache) < BOT_USER_ID_CACHE_MAX:
                    break
                if pins.get(victim):
                    continue
                cache.pop(victim, None)
        cache[key] = user_id

    def _bot_id_flights(self) -> Dict[str, Any]:
        """In-flight bots.info lookups, keyed by bot id — the single-flight register."""
        flights = getattr(self, "_bot_user_id_flights", None)
        if flights is None:
            flights = {}
            self._bot_user_id_flights = flights
        return flights

    def bot_user_id_for(self, bot_id: Optional[str]) -> Optional[str]:
        """SYNC peek at the cache — NEVER an API call.

        `normalize_slack_message` cannot await, and the sync paths (the normalizer, the outbound
        mention guard) must be able to ask this question without becoming async. An id nobody has
        primed yet simply answers None, and the caller falls back to what it did before."""
        if not bot_id:
            return None
        return self._bot_user_ids().get(str(bot_id))

    async def resolve_bot_user_id(self, bot_id: Optional[str]) -> Optional[str]:
        """The bot USER id behind a bot OBJECT id, via bots.info, single-flighted and cached.

        Best-effort by construction: any failure returns None. A lookup that cannot answer must
        never fail a fetch or a turn — the worst case is the pre-existing behavior, where the B id
        stands in for the actor.

        SINGLE FLIGHT IS PER CALL, NOT PER CACHE ENTRY. Two fetches of one channel run
        concurrently (the shared window and the origin thread; the search fan-out likewise) and
        must not each pay for the same answer — but a FAILURE is not cached, so waiters that
        re-checked only the cache would have gone on to make N serialized calls after one timeout.
        They share the outcome of the flight they queued behind instead, including its failure.
        A call that starts AFTER that flight finished finds no flight and no entry, and retries:
        the failure is scoped to the flight, not to the process.

        ONLY a U/W-shaped answer is kept as a positive. bots.info returning something else is not
        a mention target, and a B-shaped "user id" cached here would feed the outbound guard the
        exact id it exists to keep out of the channel."""
        if not bot_id:
            return None
        key = str(bot_id)
        cache = self._bot_user_ids()
        if key in cache:
            return cache[key]
        flights = self._bot_id_flights()
        queued = flights.get(key)
        if queued is not None:
            try:
                # Shielded: a waiter that is itself cancelled must not cancel the flight the
                # other waiters are still reading.
                return await asyncio.shield(queued)
            except Exception:  # noqa: BLE001 — the flight's failure is this waiter's None
                return None
        flight = asyncio.get_running_loop().create_future()
        flights[key] = flight
        user_id: Optional[str] = None
        try:
            resp = await self.app.client.bots_info(bot=key)
            raw = (resp.get("bot") or {}).get("user_id") if resp and resp.get("ok") else None
            user_id = str(raw) if raw and is_user_shaped_id(str(raw)) else None
            self._cache_bot_user_id(key, user_id)
        except Exception as e:  # noqa: BLE001 — transient: NOT cached, never raised
            self.log_debug(f"bots.info could not resolve {key}: {e}")
            user_id = None
        finally:
            # Cleared BEFORE the result is published, so nobody can queue on a finished flight;
            # a cancellation lands here too, and the waiters get None rather than hanging.
            flights.pop(key, None)
            if not flight.done():
                flight.set_result(user_id)
        return user_id

    async def prime_bot_user_ids(self, payloads: Iterable[Any]) -> None:
        """Warm the cache for a BATCH of raw Slack payloads before they are normalized.

        The async fetch/event seams call this so the sync normalizer downstream can answer from
        memory. Only payloads with a `bot_id` and NO `user` are interesting — anything carrying a
        `user` already names its actor — and only ids the cache has never seen cost a call.

        The budget is the SAME one the display-name resolver spends for a request
        (ACTOR_REMOTE_LOOKUP_DEFAULT): both are "remote id lookups one batch may pay for", and a
        transient failure leaves its id uncached, so without a ceiling a page full of an
        unreachable bot's messages could turn one fetch into a bots.info ladder."""
        cache = self._bot_user_ids()
        unknown: List[str] = []
        batch: Set[str] = set()
        for payload in payloads or ():
            if not isinstance(payload, dict) or payload.get("user"):
                continue
            bot_id = payload.get("bot_id")
            if not bot_id:
                continue
            key = str(bot_id)
            batch.add(key)
            if key in cache or key in unknown:
                continue
            unknown.append(key)
        # THE WHOLE BATCH is pinned for the whole call, not just the ids being resolved: an entry
        # this batch already counted as known must still be there when the caller normalizes, or a
        # bot resolved a moment ago comes back a B id because some batch evicted it meanwhile.
        with self._pinned_bot_ids(batch):
            for key in unknown[:ACTOR_REMOTE_LOOKUP_DEFAULT]:
                await self.resolve_bot_user_id(key)

    async def get_channel_context(self, channel_id: Optional[str]) -> Optional[dict]:
        """Cached channel metadata (name/topic/purpose/num_members) for prompt context.

        Returns {"name", "topic", "purpose", "num_members"} for real channels, None for
        DMs/MPIMs or on any failure (F29: num_members is defensive — absent from the API
        payload → None). TTL-cached per channel so this costs one conversations.info
        call per channel per window — topic edits arrive with a subtype and never
        reach the dispatch path, so a short TTL is the refresh mechanism."""
        if not channel_id:
            return None
        cache = getattr(self, "_channel_ctx_cache", None)
        if cache is None:
            cache = {}
            self._channel_ctx_cache = cache
        now = time.monotonic()
        hit = cache.get(channel_id)
        if hit and hit[0] > now:
            return hit[1]
        try:
            resp = await self.app.client.conversations_info(
                channel=channel_id, include_num_members=True)
            ch = (resp.get("channel") or {}) if resp else {}
            if ch.get("is_im") or ch.get("is_mpim"):
                data = None
            else:
                # Slack HTML-escapes &/</> in these fields; undo it for the prompt
                def _clean(s: str) -> str:
                    return s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").strip()
                data = {
                    "name": ch.get("name") or "",
                    "topic": _clean((ch.get("topic") or {}).get("value") or ""),
                    "purpose": _clean((ch.get("purpose") or {}).get("value") or ""),
                    "num_members": ch.get("num_members"),
                }
            cache[channel_id] = (now + 900, data)  # 15 min
            return data
        except Exception as e:
            self.log_debug(f"channel context fetch failed for {channel_id}: {e}")
            cache[channel_id] = (now + 60, None)  # brief negative cache; don't hammer on errors
            return None

    def get_cached_channel_context(self, channel_id: Optional[str]) -> Optional[dict]:
        """F29: sync peek into the get_channel_context TTL cache — NO API call.

        Returns the cached metadata dict if present and unexpired, else None. Lets the
        volatile response suffix read num_members without an await (the async system-prompt
        path warmed this cache earlier in the same request)."""
        if not channel_id:
            return None
        cache = getattr(self, "_channel_ctx_cache", None)
        if not cache:
            return None
        hit = cache.get(channel_id)
        if hit and hit[0] > time.monotonic():
            return hit[1]
        return None

    async def get_username(self, user_id: str, client) -> str:
        """Get username from user ID, with caching"""
        # Check memory cache first
        if user_id in self.user_cache and 'username' in self.user_cache[user_id]:
            return self.user_cache[user_id]['username']
        
        # Check database for user
        user_data = await self.db.get_or_create_user_async(user_id)
        if user_data.get('username'):
            # Load full user info from DB to memory cache
            user_info = await self.db.get_user_info_async(user_id)
            if user_info:
                self.user_cache[user_id] = {
                    'username': user_data['username'],
                    'real_name': user_info.get('real_name'),
                    'email': user_info.get('email'),
                    'timezone': user_info.get('timezone', 'UTC'),
                    'tz_label': user_info.get('tz_label', 'UTC'),
                    'tz_offset': user_info.get('tz_offset', 0)
                }
                self.log_debug(f"Loaded user info from DB for {user_id}: email={user_info.get('email')}, real_name={user_info.get('real_name')}, timezone={user_info.get('timezone')}")
                return user_data['username']
        
        try:
            # Fetch user info from Slack API
            result = await client.users_info(user=user_id)
            if result["ok"]:
                user_info = result["user"]
                # Get both display name and real name
                display_name = user_info.get("profile", {}).get("display_name")
                real_name = user_info.get("profile", {}).get("real_name")
                email = user_info.get("profile", {}).get("email")
                # Prefer display name, fall back to real name, then just the ID
                username = display_name or real_name or user_info.get("name") or user_id
                
                # Debug log for email
                self.log_debug(f"Fetched user info for {user_id}: email={email}, real_name={real_name}")
                
                # Cache both username and timezone info in memory
                self.user_cache[user_id] = {
                    'username': username,
                    'real_name': real_name,
                    'email': email,
                    'timezone': user_info.get('tz', 'UTC'),
                    'tz_label': user_info.get('tz_label', 'UTC'),
                    'tz_offset': user_info.get('tz_offset', 0)
                }
                
                # Save to database with all user info
                await self.db.get_or_create_user_async(user_id, username)
                await self.db.save_user_info_async(
                    user_id,
                    username=username,
                    real_name=real_name,
                    email=email,
                    timezone=user_info.get('tz', 'UTC'),
                    tz_label=user_info.get('tz_label', 'UTC'),
                    tz_offset=user_info.get('tz_offset', 0)
                )
                
                self.log_debug(f"Cached timezone info for {username}: tz={user_info.get('tz')}, tz_label={user_info.get('tz_label')}")
                return username
        except Exception as e:
            self.log_debug(f"Could not fetch username for {user_id}: {e}")

        return user_id  # Fallback to user ID if fetch fails

    async def resolve_usernames(
        self, user_ids: Iterable[str], api_client,
        max_remote_lookups: int = ACTOR_REMOTE_LOOKUP_DEFAULT,
        stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """BF2: batched, request-scoped, READ-ONLY display-name resolver.

        Returns ``{user_id: display_name}`` for the ids it could resolve; ids it could
        not resolve are OMITTED so the caller falls back to the raw id. Used by the
        rebuild + tool-result read paths (get_thread_history, history_tool, search_tool),
        which must never mutate user state just for reading old messages.

        Unlike ``get_username`` — which the LIVE event paths use and which get_or_creates
        the user row and bumps ``last_seen`` — this writes nothing: it serves the in-memory
        cache, then a read-only DB lookup (``get_user_info_async``), then at most
        ``max_remote_lookups`` Slack ``users.info`` calls for the WHOLE request. Ids are
        deduped first; failures are negative-cached for this call; ids past the budget stay
        raw. So a cold rebuild of an old thread can't fan out into hundreds of sequential
        lookups or delay the turn, and a successful fetch is kept in the memory cache only.

        `stats`, when given, is FILLED IN PLACE with what this call actually spent:
        ``budget`` (the cap it was given), ``remote_lookups`` (users.info calls made) and
        ``attempted_ids`` (every id it TRIED to resolve, resolved or not). A caller that must
        split one remote budget across two resolutions — the shallow window's periphery and
        origin phases — cannot compute the remainder any other way, and an id this call
        attempted and failed must never be retried by the second, or one block's rendering of an
        actor would depend on which of them ran.

        THE CALLER SEEDS IT (never this method): the keys are read back unconditionally, and a
        legal path here returns without reaching the remote pass at all.
        """
        if stats is not None:
            stats["budget"] = int(max_remote_lookups)
        cache = self.user_cache  # created by the client; get_username assumes it too

        resolved: Dict[str, str] = {}
        pending: list = []
        for uid in user_ids:
            if not uid or uid in resolved or uid in pending:
                continue
            if not _is_resolvable_user_id(uid):
                continue
            info = cache.get(uid)
            name = info.get("username") if isinstance(info, dict) else None
            if name:
                resolved[uid] = name
            else:
                pending.append(uid)

        # Read-only DB pass — ONE bulk read for all pending ids (never get_or_create; reading
        # must not create rows). `pending` preserves input order, so `still` stays ordered.
        still: list = []
        rows = {}
        if pending:
            try:
                rows = await self.db.get_user_infos_async(pending)
            except Exception as e:
                self.log_debug(f"resolve_usernames bulk DB read failed: {e}")
                rows = {}
        for uid in pending:
            info = rows.get(uid) or {}
            name = info.get("username")
            if name:
                cache[uid] = {
                    "username": name,
                    "real_name": info.get("real_name"),
                    "email": info.get("email"),
                    "timezone": info.get("timezone", "UTC"),
                    "tz_label": info.get("tz_label", "UTC"),
                    "tz_offset": info.get("tz_offset", 0),
                }
                resolved[uid] = name
            else:
                still.append(uid)

        # Remote pass — budget-bounded, negative-cached for THIS request, no DB write.
        if still and api_client is not None and max_remote_lookups > 0:
            failed: set = set()
            budget = max_remote_lookups
            for uid in still:
                if budget <= 0:
                    break
                if uid in failed:
                    continue
                budget -= 1
                if stats is not None:
                    stats["remote_lookups"] = int(stats.get("remote_lookups", 0)) + 1
                    stats.setdefault("attempted_ids", set()).add(uid)
                try:
                    result = await api_client.users_info(user=uid)
                except Exception as e:
                    self.log_debug(f"resolve_usernames users.info failed for {uid}: {e}")
                    failed.add(uid)
                    continue
                user = (result or {}).get("user") or {}
                if not (result and result.get("ok")) or not user:
                    failed.add(uid)
                    continue
                profile = user.get("profile", {}) or {}
                name = (profile.get("display_name") or profile.get("real_name")
                        or user.get("name"))
                if not name:
                    failed.add(uid)
                    continue
                cache[uid] = {
                    "username": name,
                    "real_name": profile.get("real_name"),
                    "email": profile.get("email"),
                    "timezone": user.get("tz", "UTC"),
                    "tz_label": user.get("tz_label", "UTC"),
                    "tz_offset": user.get("tz_offset", 0),
                }
                resolved[uid] = name
        return resolved

    async def get_user_timezone(self, user_id: str, client) -> str:
        """Get user's timezone, fetching if necessary"""
        # Check memory cache first
        if user_id in self.user_cache and 'timezone' in self.user_cache[user_id]:
            return self.user_cache[user_id]['timezone']
        
        # Check database
        tz_info = await self.db.get_user_timezone_async(user_id)
        if tz_info:
            # Load to memory cache
            if user_id not in self.user_cache:
                self.user_cache[user_id] = {}
            self.user_cache[user_id]['timezone'] = tz_info[0]
            self.user_cache[user_id]['tz_label'] = tz_info[1]
            self.user_cache[user_id]['tz_offset'] = tz_info[2] or 0
            return tz_info[0]
        
        # Fetch user info (which will also cache it)
        await self.get_username(user_id, client)
        
        # Return timezone from cache or default to UTC
        if user_id in self.user_cache and 'timezone' in self.user_cache[user_id]:
            return self.user_cache[user_id]['timezone']
        
        return 'UTC'  # Default fallback

    def extract_file_id_from_url(self, file_url: str) -> Optional[str]:
        """Extract file ID from a Slack file URL
        
        Args:
            file_url: The Slack file URL
            
        Returns:
            File ID if found, None otherwise
        """
        import re
        
        # Try to extract file ID from the URL
        patterns = [
            r'/files-pri/[^/]+-([^/]+)/',  # files-pri format
            r'/files/[^/]+/([^/]+)/',       # permalink format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, file_url)
            if match:
                file_id = match.group(1)
                self.log_debug(f"Extracted file ID from URL: {file_id}")
                return file_id
        
        return None

    async def _read_response_capped(self, response, max_bytes: Optional[int]) -> Optional[bytes]:
        """Read a response body. When `max_bytes` is set, STREAM and stop at max_bytes+1,
        returning None if the cap is exceeded — so an ambient download can't buffer an unbounded
        (missing/dishonest Content-Length) body into memory. `max_bytes=None` → the original
        unbounded read (addressed-path behavior, unchanged)."""
        if max_bytes is None:
            return await response.read()
        limit = int(max_bytes)
        buf = bytearray()
        async for chunk in response.content.iter_chunked(64 * 1024):
            if not chunk:
                continue
            buf.extend(chunk)
            if len(buf) > limit:
                self.log_warning(f"Download exceeded byte cap ({limit}); aborting stream")
                return None
        return bytes(buf)

    async def download_file(self, file_url: str, file_id: Optional[str] = None,
                            allow_html: bool = False,
                            max_bytes: Optional[int] = None) -> Optional[bytes]:
        """Download a file from Slack

        Args:
            file_url: The Slack file URL (can be url_private or permalink)
            file_id: Optional file ID (will be extracted from URL if not provided)
            allow_html: Accept a text/html body instead of rejecting it.
            max_bytes: When set, stream and abort at max_bytes+1 (returns None) instead of
                buffering the whole body. The ambient workers pass their smaller ceiling here;
                the addressed path leaves it None (unbounded read, unchanged).

        An HTML body normally means the download FAILED — Slack serves a login page rather than
        a 401 when auth is wrong, so HTML where an image should be is the signature of a bad
        token, and returning it would hand the caller a web page dressed as a PNG.

        A canvas is the exception: its content genuinely IS html (there is no canvases.read —
        you fetch url_private and get markup). So canvases opt in, and check for themselves
        that what came back is a canvas rather than a login screen.
        """
        try:
            # If file_id not provided, try to extract from URL
            if not file_id:
                # URL format: https://files.slack.com/files-pri/[TEAM]-[FILE_ID]/filename
                # or https://[team].slack.com/files/[USER]/[FILE_ID]/filename
                import re

                # Try to extract file ID from the URL
                patterns = [
                    r'/files-pri/[^/]+-([^/]+)/',  # files-pri format
                    r'/files/[^/]+/([^/]+)/',       # permalink format
                ]

                for pattern in patterns:
                    match = re.search(pattern, file_url)
                    if match:
                        file_id = match.group(1)
                        self.log_debug(f"Extracted file ID from URL: {file_id}")
                        break

                if not file_id:
                    # If we can't extract ID, try direct download with the URL
                    self.log_debug("Could not extract file ID, trying direct download")
                    headers = {"Authorization": f"Bearer {config.slack_bot_token}"}

                    session = self._get_session()
                    try:
                        async with session.get(file_url, headers=headers) as response:
                            if response.status == 200:
                                return await self._read_response_capped(response, max_bytes)
                            else:
                                self.log_error(f"Failed to download file directly: HTTP {response.status}")
                                return None
                    except aiohttp.ClientError as e:
                        self.log_error(f"Network error downloading file directly: {e}")
                        return None
            
            # Get file info to get the private URL
            self.log_debug(f"Getting file info for file ID: {file_id}")
            file_info = await self.app.client.files_info(file=file_id)
            
            # Check if file exists and is accessible
            if not file_info.get("ok"):
                self.log_error(f"Failed to get file info: {file_info.get('error', 'Unknown error')}")
                return None
            
            # Get the URL for downloading
            file_data = file_info.get("file", {})
            url_private = file_data.get("url_private") or file_data.get("url_private_download")
            
            if not url_private:
                self.log_error("No private URL found in file info")
                self.log_debug(f"File info keys: {file_data.keys()}")
                return None
            
            self.log_debug(f"Downloading from private URL: {url_private[:50]}...")
            
            # Download file using aiohttp with auth header
            headers = {"Authorization": f"Bearer {config.slack_bot_token}"}

            session = self._get_session()
            try:
                async with session.get(url_private, headers=headers) as response:
                    if response.status == 200:
                        # Check if we got actual image data
                        content_type = response.headers.get('content-type', '').lower()
                        if 'text/html' in content_type and not allow_html:
                            self.log_error("Got HTML instead of image data from private URL")
                            text_preview = await response.text()
                            self.log_debug(f"Response preview: {text_preview[:200]}")
                            return None
                        return await self._read_response_capped(response, max_bytes)
                    else:
                        self.log_error(f"Failed to download file: HTTP {response.status}")
                        return None
            except aiohttp.ClientError as e:
                self.log_error(f"Network error downloading file: {e}")
                return None
            
        except SlackApiError as e:
            self.log_error(f"Error getting file info: {e}")
            return None
        except Exception as e:
            self.log_error(f"Error downloading file: {e}")
            return None
