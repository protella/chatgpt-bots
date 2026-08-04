"""The live battery's harness — clients, pollers, correlation, state restores (RESPEC §7.1a).

WHY THIS FILE EXISTS AT ALL. The deadline and poll constants below were MEASURED in the shipped
P2 and P3 live batteries, but the functions carrying them lived in a session scratch directory,
so a contract was citing numbers nothing in the repository defined. W5 gives them a home. The
values are not re-derived and are not tuned here.

THE TWO CLIENTS ARE NOT INTERCHANGEABLE. `USER` posts, uploads and reacts; `BOT` reads history,
replies, reactions and receipts. **The bot token cannot trigger the bot** — a seed posted with it
is a message the gate never judges — which is why every seed goes out over the user token and why
the harness holds two clients rather than one.

**NOTHING HERE DELETES A MESSAGE** (owner ruling, 2026-08-02). The battery's messages stay in the
channel: the owner watches the run live and reads the room afterwards, so a harness that tidied up
behind itself would be erasing the evidence they are reading. Reports still list every ts the run
seeded and every ts it observed, as bookkeeping. The only thing a row puts back is DURABLE STATE —
a channel setting, a memory row, the window anchor — because that is bot configuration rather than
conversation.

**AND NOTHING IT POSTS IS TOKEN-SHAPED** (same ruling). Every seed reads as a message a coworker
would type. Run-uniqueness comes from naturally-worded facts — a plausible supplier name, a
specific quantity, a date — minted here from the run's nonce, which itself is never posted.

POLLERS ARE HONEST. "Never raise" would turn an expired token, a missing scope or a malformed
response into "the bot didn't answer", and the battery would report a behavioural failure that is
really an operator error. Every misconfiguration raises, named, with the Slack code attached; the
ONLY silent outcome is "Slack answered normally and there was nothing", which returns empty.

WHAT THIS MODULE NEVER DOES. It never writes `DEV_TREAT_BOT_IDS_AS_HUMAN`, and it cannot: the bot
reads that list once at import from the environment it booted with, so mutating this process's
environment would change nothing in the bot while "passing" a restore assertion. The allowlist is
PRECONFIGURED operator state that the preflight READS and asserts. See `tests/live/README.md`.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import time
import unicodedata
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Any, AsyncIterator, Callable, Dict, Iterable, Iterator, List, Optional,
                    Sequence, Tuple)

from slack_sdk.errors import SlackApiError
from slack_sdk.http_retry.builtin_async_handlers import AsyncRateLimitErrorRetryHandler
from slack_sdk.web.async_client import AsyncWebClient

from config import BotConfig
from message_processor import dev_barriers
from message_processor.dev_barriers import POST_ADMISSION, POST_PARTIAL_POST
from message_processor.participation_telemetry import LOG_NAME as LEDGER_LOG_NAME
from slack_client.normalizer import TimestampError, parse_ts

# --------------------------------------------------------------------------- measured constants

REPLY_DEADLINE_SECONDS = 180.0      # measured in the P2/P3 shipped batteries
REPLY_POLL_SECONDS = 5.0            # same
SLOW_TURN_DEADLINE_SECONDS = 200.0  # tool / build / image turns, same provenance
SEED_PACE_SECONDS = 1.0             # Slack's documented chat.postMessage guidance, ~1/s/channel

# A THIRD-PARTY APP ANSWERS ON ITS OWN SCHEDULE, and grading our restraint against it must not be
# bounded by what WE consider a slow turn. Measured in C0BKX77NU66 on 2026-08-02 across every
# surviving thread Claude Tag replied in: first replies at 31.0s, 82.8s, 100.5s and 477.6s — one in
# four already exceeds the 180s we allow our own bot, which is why the 2026-08-01 pass reported
# `error` for a row whose premise was merely slow. 600s covers the observed maximum with margin.
#
# IT BOUNDS WAITS ON ANOTHER APP AND NOTHING ELSE. `REPLY_DEADLINE_SECONDS` and
# `SLOW_TURN_DEADLINE_SECONDS` are untouched: a third party being slow is never a reason to relax
# what we grade OURSELVES on, and a row that waited 600s for our own bot would be hiding a hang.
THIRD_PARTY_REPLY_DEADLINE_SECONDS = 600.0


# A BARRIER IS NOT A SLACK WAIT and does not use the Slack constants (§9's wait table). It is an
# in-process file handshake whose bound is the bot's OWN timeout — wait longer than the bot will
# pause and the wait outlives the thing it is waiting for. The poll is `dev_barriers`' own, read
# from the module rather than restated so the two cannot drift apart.
BARRIER_DEADLINE_SECONDS = 120.0    # DEV_TURN_BARRIERS_TIMEOUT's default
BARRIER_POLL_SECONDS = dev_barriers._POLL_SECONDS   # noqa: SLF001 — §9 names this exact value


def barrier_deadline() -> float:
    """The bot's configured barrier timeout, so the harness never waits past its own release."""
    try:
        return max(0.0, float(os.environ.get("DEV_TURN_BARRIERS_TIMEOUT")
                              or BARRIER_DEADLINE_SECONDS))
    except ValueError:
        return BARRIER_DEADLINE_SECONDS

# How many rotations of the participation ledger a reader walks. A long battery can rotate
# mid-run, and a reader that only opened the live file would lose the rows it is waiting for.
LEDGER_ROTATIONS = 5

# The names that count as "the bot went and looked" for rows 1, 2 and 4. Enumerated as a
# predicate rather than a set because the fetch family is open: `fetch_channel_history`,
# `fetch_thread_messages`, `fetch_pinned_messages`, `fetch_channel_info` are today's members and
# a new one should not silently fall outside the rows that grade this.
SEARCH_TOOL_NAME = "search_slack"
FETCH_TOOL_PREFIX = "fetch_"

_ENV_USER_TOKEN = "SLACK_TEST_USER_TOKEN"
_ENV_BOT_TOKEN = "SLACK_BOT_TOKEN"
_ENV_BARRIER_DIR = "DEV_TURN_BARRIERS_DIR"

_AUTH_CODES = frozenset({"invalid_auth", "not_authed", "account_inactive"})
_SCOPE_CODES = frozenset({"missing_scope", "not_in_channel", "channel_not_found"})
_RATELIMITED = "ratelimited"

# `window_anchor` is a FOURTH kind beyond §7.1a's three: row 8 advances durable
# selection state, which none of the original three describes. Flagged for the spec.
RESTORE_KINDS = ("channel_setting", "channel_memory", "steering", "window_anchor")
ROW_STATUSES = ("pass", "unrestored", "fail", "skipped", "error")


# ------------------------------------------------------------------------------ the raise taxonomy

class HarnessError(Exception):
    """Base for every failure the harness itself reports, so a runner can tell one from a bug."""


class HarnessStartupError(HarnessError):
    """A prerequisite is missing BEFORE anything ran — a token, a directory.

    Named and raised at startup rather than surfacing as a row failure fifty seconds later,
    which is what a bare `KeyError` from `os.environ[...]` at import time would have produced.
    """


class _CodedHarnessError(HarnessError):
    """A Slack refusal, with the code Slack gave attached rather than folded into prose."""

    def __init__(self, message: str, code: Optional[str] = None) -> None:
        super().__init__(message)
        self.code = code


class HarnessAuthError(_CodedHarnessError):
    """`invalid_auth`, `not_authed`, `account_inactive` — the token is wrong, not the bot."""


class HarnessScopeError(_CodedHarnessError):
    """`missing_scope`, `not_in_channel`, `channel_not_found` — the app cannot see the room."""


class HarnessProtocolError(HarnessError):
    """Slack answered with something that is not the documented shape, or the evidence is broken.

    BROKEN EVIDENCE IS NOT SILENCE. A provenance row whose JSON will not parse is a different
    fact from no row at all, and degrading it to "zero tools" would manufacture exactly the false
    reading rows 1, 2 and 4 exist to avoid.
    """


class HarnessApiError(_CodedHarnessError):
    """Any other `ok: false` that persisted to the deadline, including a retried `ratelimited`."""


class HarnessPreflightError(HarnessError):
    """The battery's premise is not configured — the allowlist, or a party that will not resolve."""


class HarnessCorrelationError(HarnessError):
    """A Slack ts could not be tied to exactly one turn."""


# ---------------------------------------------------------------------------- injectable seams
#
# Time and the two stores are reached through module-level functions so the network-free tests
# can drive a fake clock and a fake row store by monkeypatching this module. Every internal call
# site looks them up as globals for that reason; do not bind them into locals or defaults.

def _now() -> float:
    return time.monotonic()


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def db_path() -> Path:
    """The bot's own database, from the bot's own config — never a literal path."""
    return Path(BotConfig().database_dir) / "slack.db"


def ledger_dir() -> Path:
    """The participation ledger's directory, from the bot's own config (`config.log_directory`)."""
    return Path(BotConfig().log_directory or "logs")


class _Deadline:
    """A wall-clock bound read through `_now`, so a fake clock governs every poll loop."""

    def __init__(self, seconds: float) -> None:
        self.expires_at = _now() + max(0.0, seconds)

    def expired(self) -> bool:
        return _now() >= self.expires_at

    def remaining(self) -> float:
        """What is left of the ONE declared bound, so a two-phase wait cannot double it."""
        return max(0.0, self.expires_at - _now())

    async def tick(self, poll: float) -> None:
        """Sleep one poll interval, or the rest of the bound — WHICHEVER IS SHORTER.

        A plain `sleep(poll)` after an expiry check overshoots by up to one interval, so a
        declared 180-second deadline actually runs to 185.

        WHAT THIS DOES AND DOES NOT PROMISE. It removes overshoot from the POLLING SLEEPS. It
        does not bound the individual Slack or database call a loop is about to make: an
        operation started just before expiry runs to its own transport timeout, so a bound is
        "no extra waiting", not "returns by this instant". Wrapping each call in `remaining()`
        would be the stronger guarantee and is not claimed here.
        """
        await _sleep(min(poll, self.remaining()))


# ------------------------------------------------------------------------------------ identity

@dataclass(frozen=True)
class PartyIdentity:
    """A Slack app has TWO ids and they are not interchangeable.

    `bot_id` (B…) is what `DEV_TREAT_BOT_IDS_AS_HUMAN` holds and what a bot-authored message
    carries in its `bot_id` field. `user_id` is the bot-USER id — what a MENTION must render as,
    `<@user_id>`. Matching a reply against one field alone silently misses replies, because Slack
    fills the two differently depending on how a message was posted.
    """

    bot_id: str
    user_id: str


@dataclass(frozen=True)
class Observed:
    """One message a poller saw, with Slack's RAW author fields kept, both of them.

    Never a pre-resolved "author" string: the routing rule compares against a `PartyIdentity` on
    either field, and cleanup has to tell our bot's artifacts from a third app's.
    """

    ts: str
    text: str
    thread_ts: Optional[str]
    channel: str
    user: Optional[str]
    bot_id: Optional[str]


@dataclass(frozen=True)
class ProvenanceRead:
    """What one provenance poll actually learned.

    A bare tuple could not express it: `()` had to mean BOTH "no row exists" and "a row exists
    and names nothing", and those are the two states row 1's asymmetric grading turns on.
    """

    row_present: bool
    names: Tuple[str, ...]


def identity_matches(user: Optional[str], bot_id: Optional[str], identity: PartyIdentity) -> bool:
    """EITHER field matches: raw `user` == `user_id`, or raw `bot_id` == `bot_id`."""
    if identity.user_id and user == identity.user_id:
        return True
    if identity.bot_id and bot_id == identity.bot_id:
        return True
    return False


def observed_matches(observed: Observed, identity: PartyIdentity) -> bool:
    return identity_matches(observed.user, observed.bot_id, identity)


# ------------------------------------------------------------------------------------- clients

@dataclass(frozen=True)
class Clients:
    user: AsyncWebClient   # POSTS, uploads, reacts — the human side of every seed
    bot: AsyncWebClient    # READS history, replies, reactions, receipts — it writes nothing


_clients: Optional[Clients] = None
_bot_auth_data: Optional[Dict[str, Any]] = None
_harness_user_identity: Optional[PartyIdentity] = None


def _require_env(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise HarnessStartupError(
            f"{name} is not set. The live battery needs both {_ENV_USER_TOKEN} (posts seeds as a "
            f"human — the bot token cannot trigger the bot) and {_ENV_BOT_TOKEN} (reads history "
            f"and the bot's own replies).")
    return value


def build_clients() -> Clients:
    """Construct both clients and install the SDK's OWN rate-limit handler on each.

    The ASYNC handler, not the sync one: both clients are `AsyncWebClient`, whose retry pipeline
    calls `can_retry_async`, which `RateLimitErrorRetryHandler` does not implement — installing it
    would silently never retry. The harness writes no retry logic of its own.
    """
    user = AsyncWebClient(token=_require_env(_ENV_USER_TOKEN))
    bot = AsyncWebClient(token=_require_env(_ENV_BOT_TOKEN))
    for client in (user, bot):
        client.retry_handlers.append(AsyncRateLimitErrorRetryHandler(max_retry_count=3))
    return Clients(user=user, bot=bot)


def clients() -> Clients:
    """The run's client pair, built once. Missing either token is a named STARTUP failure."""
    global _clients
    if _clients is None:
        _clients = build_clients()
    return _clients


def reset_caches() -> None:
    """Drop the per-run caches. For tests, and for a runner that re-reads its environment."""
    global _clients, _bot_auth_data, _harness_user_identity
    _clients = None
    _bot_auth_data = None
    _harness_user_identity = None


async def _bot_auth() -> Dict[str, Any]:
    """ONE `auth.test` on the bot token, cached for the run.

    Identity and workspace both come off it, so the pair plus the team id cost one call. Resolving
    any of them per poll would spend a call per 5-second tick; hardcoding them would break the
    moment the dev app is reinstalled.
    """
    global _bot_auth_data
    if _bot_auth_data is None:
        _bot_auth_data = await _call(clients().bot, "auth_test")
    return _bot_auth_data


async def bot_identity() -> PartyIdentity:
    """OUR bot's own identity PAIR. `auth.test` yields `user_id` and, on a bot token, `bot_id`."""
    data = await _bot_auth()
    identity = PartyIdentity(bot_id=str(data.get("bot_id") or ""),
                             user_id=str(data.get("user_id") or ""))
    if not identity.bot_id and not identity.user_id:
        raise HarnessProtocolError("auth.test returned neither bot_id nor user_id")
    return identity


async def bot_team_id() -> str:
    """The workspace the bot is installed in — `channel_window_anchor` is keyed by it."""
    team = (await _bot_auth()).get("team_id")
    if not team:
        raise HarnessProtocolError("auth.test returned no team_id")
    return str(team)


_harness_user_names: Optional[Tuple[str, ...]] = None


async def harness_user_display_names() -> Tuple[str, ...]:
    """Every name Slack knows the harness operator by — real name, display name, handle.

    The ack predicate treats an addressee's name as ADDRESS (like a mention), and live the person
    the bot thanks is the real operator, so a "Thanks, Peter." must be gradable. Cached per run.
    """
    global _harness_user_names
    if _harness_user_names is None:
        ident = await harness_user_identity()
        data = await _call(clients().bot, "users_info", user=ident.user_id)
        user = data.get("user") or {}
        profile = user.get("profile") or {}
        _harness_user_names = tuple({n for n in (user.get("real_name"), user.get("name"),
                                                 profile.get("display_name"),
                                                 profile.get("real_name")) if n})
    return _harness_user_names


async def harness_user_identity() -> PartyIdentity:
    """The HUMAN operator the seeds are posted as.

    The report's bookkeeping needs it to tell "a message we posted" from "a third app's".
    A user token carries no `bot_id` of its own, so that half is empty and only `user_id` matches.
    """
    global _harness_user_identity
    if _harness_user_identity is None:
        data = await _call(clients().user, "auth_test")
        identity = PartyIdentity(bot_id=str(data.get("bot_id") or ""),
                                 user_id=str(data.get("user_id") or ""))
        if not identity.user_id:
            # An empty user_id matches NOTHING, so every seed the harness posted would fall
            # through the operator branch of the router into the third-party bucket, and the
            # report would attribute the run's own messages to somebody else.
            raise HarnessPreflightError(
                f"{_ENV_USER_TOKEN}'s auth.test returned no user_id; cleanup could not tell the "
                f"harness's own seeds from a third app's messages.")
        team = data.get("team_id")
        if not team:
            raise HarnessProtocolError(f"{_ENV_USER_TOKEN}'s auth.test returned no team_id")
        bot_team = await bot_team_id()
        if str(team) != bot_team:
            # The premise of the whole battery is that our seeds carry OUR app's allowlisted
            # identity. Two tokens in two workspaces cannot satisfy it, and the failure would
            # surface as inexplicable gate behaviour rather than as a configuration error.
            raise HarnessPreflightError(
                f"{_ENV_USER_TOKEN} is in workspace {team!r} but {_ENV_BOT_TOKEN} is in "
                f"{bot_team!r}; the harness must post as a human in the bot's own workspace.")
        _harness_user_identity = identity
    return _harness_user_identity


# --------------------------------------------------------------------------------- the preflight

def allowlisted_bot_ids() -> List[str]:
    """The bot's OWN view of `DEV_TREAT_BOT_IDS_AS_HUMAN`, read the way the bot reads it."""
    return [str(entry).strip() for entry in (BotConfig().dev_treat_bot_ids_as_human or [])
            if str(entry).strip()]


async def _bot_record(bot_id: str) -> Dict[str, Any]:
    """One `bots.info` row, or a named preflight failure. Never a bare dict-or-None."""
    data = await _call(clients().bot, "bots_info", bot=bot_id)
    row = data.get("bot")
    if not isinstance(row, dict):
        raise HarnessPreflightError(
            f"bots.info({bot_id!r}) returned no bot record, so DEV_TREAT_BOT_IDS_AS_HUMAN names "
            f"an id this workspace does not know.")
    return row


async def claude_tag_identity() -> PartyIdentity:
    """PREFLIGHT part 1 — the second party's PAIR, derived from the allowlist. No history search.

    THE ALLOWLIST VALUE IS THE BOT_ID. `DEV_TREAT_BOT_IDS_AS_HUMAN` exists precisely to name the
    bot_ids that must classify as human, so the id is already configured before the bot boots and
    there is nothing to discover by scanning the channel — which could not work anyway, since
    `conversations.history` does not return ordinary thread replies.

    **ONE SLACK APP OWNS TWO BOT RECORDS, AND ONLY ONE OF THEM APPEARS ON OUR SEEDS.** This is the
    fact that cost the 2026-08-01 battery pass, so it is stated exactly:

    | record | `bots.info` shape | where it shows up |
    |---|---|---|
    | user-token posting record | our `app_id`, **no `user_id`** | **every seed `chat.postMessage` posts with `SLACK_TEST_USER_TOKEN`** |
    | bot-token record | our `app_id`, `user_id` = our bot user | messages our BOT posts; what `auth.test` returns |

    `auth.test` returns the SECOND one. An earlier version of this function required *that* id in
    the allowlist, on the assumption that it was what seeds carry. It is not. Configuring the
    allowlist to satisfy that check removed the id the carve-out actually needs, every seed
    classified as `other_bot`, and eleven rows graded the bot's behaviour toward a bot — silently,
    because the check "passed". The bot-token id is inert in the allowlist anyway:
    `classify_sender` calls `is_own_message` first, which matches it and returns `self` before the
    carve-out is ever consulted (`slack_client/utilities.py`).

    So the partition is BY APP, resolved from Slack rather than assumed:

    1. Learn OUR app_id from `bots.info` on our own bot_id.
    2. Resolve every allowlisted entry, deduped, preserving first-seen order. A repeated entry is a
       typo in a comma-separated env var, not a second party.
    3. REQUIRE THE SEED-BEARING RECORD: an entry with our app_id and no `user_id`. This is the one
       check that would have caught the 2026-08-01 misconfiguration, and it is the only one whose
       absence corrupts every row rather than failing one.
    4. DISCARD every entry belonging to our app, whichever record it is.
    5. REQUIRE EXACTLY ONE foreign entry to remain — that is the party the battery grades. Zero or
       more than one raises naming the entries, because a battery that guessed would measure
       restraint against the wrong app.

    THE COST IS `1 + len(allowlist)` LOOKUPS, and the ordering changed with it: the old contract
    checked cardinality before spending any call, which is no longer possible because an entry's
    app is not knowable without asking Slack. An empty allowlist still costs zero calls.
    """
    ours = await bot_identity()
    raw = allowlisted_bot_ids()
    if not raw:
        raise HarnessPreflightError(
            "DEV_TREAT_BOT_IDS_AS_HUMAN is empty. Every seed the harness posts carries our app's "
            "user-token bot_id, so without the carve-out the bot classifies its own test operator "
            "as a bot and every row's gate input is wrong.")
    if not ours.bot_id:
        raise HarnessPreflightError(
            "auth.test returned no bot_id, so the harness cannot tell our app's allowlist entries "
            "from the third party's.")

    our_app_id = str((await _bot_record(ours.bot_id)).get("app_id") or "")
    if not our_app_id:
        raise HarnessPreflightError(
            f"bots.info({ours.bot_id!r}) returned no app_id; without it our own records cannot be "
            f"told apart from the party the battery grades.")

    seen: List[str] = []
    for entry in raw:
        if entry not in seen:
            seen.append(entry)

    seed_record: Optional[str] = None
    foreign: List[Tuple[str, Dict[str, Any]]] = []
    for entry in seen:
        row = await _bot_record(entry)
        if str(row.get("app_id") or "") == our_app_id:
            # OUR app. The one WITHOUT a user_id is the record a user-token post carries.
            if not row.get("user_id"):
                seed_record = entry
            continue
        # Kept WITH its record: resolving the survivor a second time below would spend a call
        # re-deriving an answer already in hand, and the count is asserted.
        foreign.append((entry, row))

    if seed_record is None:
        raise HarnessPreflightError(
            f"DEV_TREAT_BOT_IDS_AS_HUMAN ({raw!r}) names no user-token posting record for app "
            f"{our_app_id!r} — an entry with our app_id and no user_id. That id, NOT the "
            f"{ours.bot_id!r} that auth.test returns, is what every seed the harness posts "
            f"carries; without it the bot classifies its own test operator as a bot and every "
            f"row's gate input is wrong.")
    if len(foreign) != 1:
        raise HarnessPreflightError(
            f"DEV_TREAT_BOT_IDS_AS_HUMAN must leave EXACTLY ONE entry after discarding app "
            f"{our_app_id!r}'s own records; it leaves {len(foreign)}: "
            f"{[entry for entry, _ in foreign]!r}. That entry is the party the battery grades.")

    tag_bot_id, bot_row = foreign[0]
    if not bot_row.get("user_id"):
        raise HarnessPreflightError(
            f"bots.info({tag_bot_id!r}) returned no user_id; a mention needs it, and the literal "
            f"text '@Claude' mentions nobody.")
    return PartyIdentity(bot_id=tag_bot_id, user_id=str(bot_row["user_id"]))


async def assert_claude_tag_allowlisted() -> PartyIdentity:
    """PREFLIGHT part 2. Returns the second party's PAIR, or ABORTS THE BATTERY.

    The allowlist is re-read AFTER the derive, and the derived `bot_id` is checked against that
    fresh read. Because the id is DERIVED from the allowlist, this cannot fail in production —
    which is the point: it guards the derive-then-verify SEAM, so an implementation that derives
    and then simply trusts is distinguishable from one that verifies. The allowlist is keyed by
    `bot_id`, never by `user_id`.

    IT WRITES NOTHING. There is no `Restore` entry for the allowlist, because nothing was changed.
    """
    identity = await claude_tag_identity()
    allowlist = allowlisted_bot_ids()
    if identity.bot_id not in allowlist:
        raise HarnessPreflightError(
            f"Claude Tag's bot_id {identity.bot_id!r} is not in DEV_TREAT_BOT_IDS_AS_HUMAN "
            f"({allowlist!r}). Set it in the bot's .env and restart the bot; the harness cannot "
            f"set it from here.")
    return identity


# ------------------------------------------------------------------------------- the Slack call

def _slack_code(error: SlackApiError) -> Optional[str]:
    """The Slack error code, read only off the shapes that genuinely carry one."""
    response = getattr(error, "response", None)
    data = getattr(response, "data", response)
    if isinstance(data, dict):
        code = data.get("error")
        if isinstance(code, str) and code:
            return code
    return None


def _classify(code: Optional[str], context: str) -> HarnessError:
    if code in _AUTH_CODES:
        return HarnessAuthError(f"{context}: {code}", code=code)
    if code in _SCOPE_CODES:
        return HarnessScopeError(f"{context}: {code}", code=code)
    return HarnessApiError(f"{context}: {code}", code=code)


async def _call(client: Any, method: str, **kwargs: Any) -> Dict[str, Any]:
    """One Slack call, with §7.1a's raise taxonomy applied to whatever comes back.

    `ratelimited` reaching here means the SDK's own retry handler already gave up, so it is
    reported rather than retried again — the harness writes no retry logic.
    """
    try:
        response = await getattr(client, method)(**kwargs)
    except SlackApiError as e:
        raise _classify(_slack_code(e), f"{method} refused") from e
    data = getattr(response, "data", response)
    if not isinstance(data, dict):
        raise HarnessProtocolError(f"{method} returned {type(data).__name__}, not an object")
    if data.get("ok") is False:
        raise _classify(str(data.get("error") or "unknown"), f"{method} refused")
    return data


def _require_list(data: Dict[str, Any], key: str, method: str) -> List[Dict[str, Any]]:
    value = data.get(key)
    if value is None:
        raise HarnessProtocolError(f"{method} returned no {key!r}")
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise HarnessProtocolError(f"{method}'s {key!r} is not a list of objects")
    return value


def _next_cursor(data: Dict[str, Any], method: str) -> Optional[str]:
    """The next page token, or None at the end.

    A `response_metadata` that is not an object, or a `next_cursor` that is not a string, is a
    MALFORMED response rather than the end of the walk. Treating it as the end is how a truncated
    thread becomes a silent, confident answer about a conversation nobody saw the rest of.
    """
    meta = data.get("response_metadata")
    if meta is None:
        return None
    if not isinstance(meta, dict):
        raise HarnessProtocolError(f"{method}'s response_metadata is not an object")
    cursor = meta.get("next_cursor")
    if cursor is None or cursor == "":
        return None
    if not isinstance(cursor, str):
        raise HarnessProtocolError(f"{method}'s next_cursor is {type(cursor).__name__}, not a str")
    return cursor.strip() or None


def _checked_ts(raw: Dict[str, Any], method: str) -> str:
    """Every message Slack returns has a `ts`. One without is broken evidence, not an old message.

    Coercing a missing ts to `""` would let it fall out of every `_newer` comparison silently —
    the message would vanish from a poller's view and the row would report that the bot said
    nothing.
    """
    ts = raw.get("ts")
    if not isinstance(ts, str) or not ts.strip():
        raise HarnessProtocolError(f"{method} returned a message with no usable ts")
    # And it must be a REAL timestamp. A non-numeric string passes an emptiness check and then
    # flows into an equality compare that never reaches the parser — so malformed evidence would
    # be accepted on the exact-message path while the polling paths rejected it.
    ts_key(ts, what=f"{method} message ts")
    return ts


def _as_observed(raw: Dict[str, Any], channel: str, method: str = "slack") -> Observed:
    return Observed(ts=_checked_ts(raw, method), text=str(raw.get("text") or ""),
                    thread_ts=raw.get("thread_ts"), channel=channel,
                    user=raw.get("user"), bot_id=raw.get("bot_id"))


def ts_key(raw: str, *, what: str = "timestamp") -> Tuple[int, int]:
    """A Slack ts as the canonical `(seconds, microseconds)` key.

    NEVER `float`, and never string ordering. `1752600000.000001` and `1752600000.000002` are
    distinct messages, floats decide the boundary by rounding, and lexical compare puts
    `"999.9"` after `"1000.0"`. The production contract rejects both, so the harness uses the
    same comparator the window predicates do — a harness that disagreed with the window about
    which message is newer would grade the wrong side of every boundary.
    """
    try:
        return parse_ts(raw)
    except TimestampError as e:
        raise HarnessProtocolError(f"unusable {what}: {raw!r} ({e})") from e


def ts_lt(a: str, b: str) -> bool:
    return ts_key(a) < ts_key(b)


def ts_ge(a: str, b: str) -> bool:
    return ts_key(a) >= ts_key(b)


def _newer(ts: str, after_ts: str) -> bool:
    return ts_key(ts) > ts_key(after_ts, what="lower bound")


# ------------------------------------------------------------------------------------- pollers

async def fetch_thread_complete(channel: str, root_ts: str, *,
                                oldest: Optional[str] = None,
                                latest: Optional[str] = None,
                                inclusive: bool = False) -> List[Dict[str, Any]]:
    """Page `conversations.replies` to COMPLETION, following `next_cursor`.

    Both bounds are forwarded to Slack unchanged. A thread that already holds 120 replies returns
    its newest ones on a LATER page, so a single-page read is a silent truncation — and a row that
    asserted against one would be asserting about a thread it never saw the end of.
    """
    messages: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    seen: set = set()
    while True:
        params: Dict[str, Any] = {"channel": channel, "ts": root_ts, "limit": 200}
        if oldest is not None:
            params["oldest"] = oldest
        if latest is not None:
            params["latest"] = latest
        if inclusive:
            # Slack's bounds EXCLUDE the boundary by default, so an exact-ts read
            # (`oldest == latest == ts`) returns nothing at all and a real message reads as
            # missing. Only the exact-message fetch asks for this; the polling walks want the
            # exclusive default so `after_ts` is not re-returned every tick.
            params["inclusive"] = True
        if cursor:
            params["cursor"] = cursor
        data = await _call(clients().bot, "conversations_replies", **params)
        messages.extend(_require_list(data, "messages", "conversations.replies"))
        cursor = _next_cursor(data, "conversations.replies")
        if not cursor:
            return messages
        if cursor in seen:
            # A cursor that repeats is not progress. Following it is an infinite loop that reads
            # from the outside exactly like a bot that never answered.
            raise HarnessProtocolError(
                f"conversations.replies repeated cursor {cursor!r}; pagination is not advancing")
        seen.add(cursor)


async def await_thread_visible(channel: str, root_ts: str, *, expected_ts: Sequence[str],
                               stable_reads: int = 2,
                               deadline: float = REPLY_DEADLINE_SECONDS,
                               poll: float = REPLY_POLL_SECONDS) -> List[Dict[str, Any]]:
    """Wait until a thread reads back EVERY named message, the way the consumer will read it.

    A MESSAGE IS NOT READABLE THE INSTANT `chat.postMessage` RETURNS ITS TS. Row 10 seeded two
    threads and launched the stream probe immediately; the probe died on `OriginFetchError: origin
    thread … came back empty for a reply-triggered turn`.

    **IT IS THE TS VALUES, NOT THE COUNT.** An earlier version waited for `len(messages) >=
    expected` — which any unrelated reply in the same thread satisfies while the message the
    caller actually seeded is still invisible. The caller names what it posted and every one of
    them must come back.

    **AND AN UNBOUNDED READ IS NOT THE SAME QUESTION.** The first version polled a plain
    `conversations.replies`, saw everything at once, and the probe launched half a second later
    still came back with NOTHING. Production reads an origin thread as `latest=<horizon>,
    inclusive=True` (`fetch_origin_thread`), so that is the read this waits on, with a fresh
    horizon each tick; and it wants the answer twice, because one complete-looking read is exactly
    what the failing run already had.
    """
    wanted = {str(ts) for ts in expected_ts if str(ts).strip()}
    if not wanted:
        raise HarnessError("await_thread_visible needs the ts values it is waiting for; a count "
                           "is satisfied by somebody else's reply")
    bound = _Deadline(deadline)
    stable = 0
    messages: List[Dict[str, Any]] = []
    while True:
        # The SHAPE production uses, horizon and all — an unbounded read answered a question the
        # probe was not asking.
        messages = await fetch_thread_complete(channel, root_ts, latest=f"{time.time():.6f}",
                                               inclusive=True)
        present = {_checked_ts(m, "conversations.replies") for m in messages}
        missing = sorted(wanted - present)
        stable = stable + 1 if not missing else 0
        if stable >= stable_reads:
            return messages
        if bound.expired():
            raise HarnessError(
                f"thread {root_ts} in {channel} still does not show {missing} after {deadline}s; "
                f"a reader started now would see a thread that is not there yet")
        await bound.tick(poll)


async def _fetch_history_complete(channel: str, oldest: str) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    seen: set = set()
    while True:
        params: Dict[str, Any] = {"channel": channel, "oldest": oldest, "inclusive": False,
                                  "limit": 200}
        if cursor:
            params["cursor"] = cursor
        data = await _call(clients().bot, "conversations_history", **params)
        messages.extend(_require_list(data, "messages", "conversations.history"))
        cursor = _next_cursor(data, "conversations.history")
        if not cursor:
            return messages
        if cursor in seen:
            raise HarnessProtocolError(
                f"conversations.history repeated cursor {cursor!r}; pagination is not advancing")
        seen.add(cursor)


async def _poll(work: Callable[[], Any], *, deadline: float, poll: float, empty: Any) -> Any:
    """Run `work` every `poll` seconds until it answers or the deadline expires.

    A transient `HarnessApiError` does not end the loop — §7.1a's table says "any other ok:false
    PERSISTING to the deadline" — but it is kept and raised if the deadline arrives with nothing.
    Auth, scope and protocol errors are configuration, not weather, and propagate immediately.
    """
    bound = _Deadline(deadline)
    last_api_error: Optional[HarnessApiError] = None
    while True:
        try:
            found = await work()
            if found:
                return found
            last_api_error = None
        except HarnessApiError as e:
            last_api_error = e
        if bound.expired():
            break
        await bound.tick(poll)
    if last_api_error is not None:
        raise last_api_error
    return empty


async def wait_bot_reply(channel: str, thread_ts: str, after_ts: str, *,
                         author: Optional[PartyIdentity] = None,
                         deadline: float = REPLY_DEADLINE_SECONDS,
                         poll: float = REPLY_POLL_SECONDS) -> List[Observed]:
    """Poll a THREAD for messages newer than `after_ts` authored by `author`.

    `author` DEFAULTS to OUR OWN pair, which is what every row but one wants. Row 9a passes
    Claude Tag's pair: its graded trigger is Claude Tag's reply, and a poller hardwired to our own
    ids could never observe it.

    EACH TICK CALLS `fetch_thread_complete(..., oldest=after_ts)` — never a bare single-page
    `conversations.replies`, or the poller would watch page 1 forever while the answer sat on
    page 2.
    """
    target = author or await bot_identity()

    async def _tick() -> List[Observed]:
        raw = await fetch_thread_complete(channel, thread_ts, oldest=after_ts)
        return [_as_observed(m, channel, "conversations.replies") for m in raw
                if _newer(_checked_ts(m, "conversations.replies"), after_ts)
                and identity_matches(m.get("user"), m.get("bot_id"), target)]

    return await _poll(_tick, deadline=deadline, poll=poll, empty=[])


async def wait_bot_reply_channel(channel: str, after_ts: str, *,
                                 author: Optional[PartyIdentity] = None,
                                 deadline: float = REPLY_DEADLINE_SECONDS,
                                 poll: float = REPLY_POLL_SECONDS) -> List[Observed]:
    """Poll TOP-LEVEL history for messages newer than `after_ts` authored by `author`.

    IT EXCLUDES `thread_broadcast`. A broadcast reply appears in the top-level history feed but is
    NOT a top-level message, so a row asserting "the bot answered in the channel" would pass on a
    broadcast of a THREAD reply — the opposite of what it is testing.
    """
    target = author or await bot_identity()

    async def _tick() -> List[Observed]:
        raw = await _fetch_history_complete(channel, after_ts)
        return [_as_observed(m, channel, "conversations.history") for m in raw
                if m.get("subtype") != "thread_broadcast"
                and _newer(_checked_ts(m, "conversations.history"), after_ts)
                and identity_matches(m.get("user"), m.get("bot_id"), target)]

    return await _poll(_tick, deadline=deadline, poll=poll, empty=[])


async def wait_bot_reaction(channel: str, message_ts: str, *,
                            author: Optional[PartyIdentity] = None,
                            deadline: float = REPLY_DEADLINE_SECONDS,
                            poll: float = REPLY_POLL_SECONDS) -> Tuple[str, ...]:
    """Poll `reactions.get` for reactions the BOT added to one message.

    Required by every reaction-only row: a reaction is not a message and no message poller sees
    one. A reaction by anyone else on the same message is not the bot's answer and is ignored.
    """
    target = author or await bot_identity()

    async def _tick() -> Tuple[str, ...]:
        data = await _call(clients().bot, "reactions_get", channel=channel,
                           timestamp=message_ts, full=True)
        message = data.get("message")
        if not isinstance(message, dict):
            raise HarnessProtocolError("reactions.get returned no message object")
        names: List[str] = []
        for entry in message.get("reactions") or []:
            if not isinstance(entry, dict):
                raise HarnessProtocolError("reactions.get returned a non-object reaction")
            users = entry.get("users") or []
            if target.user_id and target.user_id in users:
                names.append(str(entry.get("name") or ""))
        return tuple(names)

    return await _poll(_tick, deadline=deadline, poll=poll, empty=())


@dataclass(frozen=True)
class Receipt:
    """One surface a turn owns, from the bot's own `outbound_receipts` row.

    `state` is `in_flight` (the words are still arriving), `finalized` (the surface reached its
    final text) or `chrome` (a thinking indicator or other framing — a surface the bot created
    but not something it SAID).
    """

    message_ts: str
    state: str
    thread_root_ts: Optional[str]

    @property
    def is_prose(self) -> bool:
        return self.state != "chrome"


async def read_turn_receipts(team_id: str, channel: str, turn_id: str) -> Tuple[Receipt, ...]:
    """Every surface one turn owns, from the store the bot itself writes.

    THIS IS THE CORRELATION A TIME WINDOW CANNOT GIVE. "A bot message newer than my trigger" is
    satisfied by any other conversation's reply in a shared channel — which would let a row pass
    on someone else's output — grading our bot on another conversation's reply. A receipt names
    the turn, so the surfaces it returns are this trigger's and nothing else's.
    """
    def _read() -> Tuple[Receipt, ...]:
        conn = sqlite3.connect(str(db_path()))
        try:
            rows = conn.execute(
                "SELECT message_ts, state, thread_root_ts FROM outbound_receipts "
                "WHERE team_id = ? AND channel_id = ? AND turn_id = ?",
                (team_id, channel, turn_id)).fetchall()
        finally:
            conn.close()
        # READ UNSORTED, THEN ORDER BY THE CANONICAL KEY. SQL's `ORDER BY message_ts` is a
        # lexical sort, which puts "1000.000001" before "999.999999" — so a split reply's parts
        # would come back out of order and the row would grade the wrong surface as "the first".
        # Validating each ts here also means a corrupt receipt is broken evidence rather than a
        # silently mis-ordered one.
        receipts = [Receipt(message_ts=str(r[0]), state=str(r[1]), thread_root_ts=r[2])
                    for r in rows]
        for receipt in receipts:
            ts_key(receipt.message_ts, what="receipt message_ts")
        return tuple(sorted(receipts, key=lambda r: ts_key(r.message_ts)))

    return await asyncio.to_thread(_read)


async def read_receipt_state(team_id: str, channel: str, message_ts: str) -> Optional[str]:
    """One surface's receipt state, or None when no receipt exists for that ts."""
    def _read() -> Optional[str]:
        conn = sqlite3.connect(str(db_path()))
        try:
            row = conn.execute(
                "SELECT state FROM outbound_receipts "
                "WHERE team_id = ? AND channel_id = ? AND message_ts = ?",
                (team_id, channel, message_ts)).fetchone()
        finally:
            conn.close()
        return str(row[0]) if row is not None else None

    return await asyncio.to_thread(_read)


# Recorded on EVERY row that reads a turn through the settle path. Verbatim, so a reader can
# grep the reports for it — and never an assertion, because it describes a bound on what the
# harness can see rather than anything the bot did.
#
# WHAT IT BOUNDS is the row's own inventory of a turn: the receipt drain worker retries
# indefinitely, so a surface whose receipt lands after the stability window is not in this row's
# `observed_ts` and its text was never read. The late chrome receipt in T121's fixture is exactly
# such a surface.
SETTLE_LIMITATION = ("surface enumeration bounded by the stability window; a receipt that lands "
                     "after it is absent from this row's observed_ts and its text was never "
                     "graded")

# Outcome kinds that MUST name at least one destination. Deliberately narrow: `reply` is the one
# kind whose whole meaning is "words landed somewhere", so a `reply` with no destination is a
# contradiction in the evidence. Other delivering-ish kinds (`detached`, `delivery_failed`) are
# NOT listed, because inventing a rule for them would fail legitimate turns.
_KINDS_THAT_MUST_DELIVER = frozenset({"reply"})


def delivered_surfaces(outcome: Dict[str, Any]) -> set:
    """The ts values the turn's OWN `turn_outcome` says it posted. FAILS CLOSED.

    Not a guess and not a count: `destinations[].first_ts` is the turn's own record of where its
    words landed, so it is the one expectation the harness can hold the receipt store to without
    inventing a number.

    **A MALFORMED OR ABSENT DESTINATION LIST IS NOT AN EMPTY ONE.** Reading it as empty makes a
    `reply` outcome indistinguishable from a `silence`, so a turn that posted would be graded as
    one that stayed quiet and its surface would never be cleaned up. Damaged evidence raises.
    """
    kind = str(outcome.get("kind") or "")
    raw = outcome.get("destinations")
    if raw is None:
        if kind in _KINDS_THAT_MUST_DELIVER:
            raise HarnessProtocolError(
                f"turn_outcome kind={kind!r} carries no destinations; a reply that landed "
                f"nowhere is a contradiction, and reading it as silence would drop whatever it "
                f"did post out of the row's record of what the turn put in the room.")
        return set()
    if not isinstance(raw, list):
        raise HarnessProtocolError(
            f"turn_outcome.destinations is {type(raw).__name__}, not a list")
    found = set()
    for destination in raw:
        if not isinstance(destination, dict):
            raise HarnessProtocolError("turn_outcome.destinations holds a non-object entry")
        first_ts = destination.get("first_ts")
        if first_ts is None:
            continue
        if not isinstance(first_ts, str) or not first_ts.strip():
            raise HarnessProtocolError(
                f"turn_outcome destination carries an unusable first_ts {first_ts!r}")
        # And it must be a REAL timestamp. A non-numeric string would go on to be compared for
        # equality against receipt ts values, match nothing, and read as a destination whose
        # receipt never arrived — a confusing failure rather than the plain "this evidence is
        # damaged" that it is.
        ts_key(first_ts, what="turn_outcome destination first_ts")
        found.add(first_ts)
    if kind in _KINDS_THAT_MUST_DELIVER and not found:
        raise HarnessProtocolError(
            f"turn_outcome kind={kind!r} names no usable destination ts; reading that as silence "
            f"would drop whatever it posted out of the row's record.")
    return found


async def await_turn_settled(team_id: str, channel: str, turn_id: str, *,
                             deadline: float = REPLY_DEADLINE_SECONDS,
                             poll: float = REPLY_POLL_SECONDS) -> Tuple[Receipt, ...]:
    """Wait for the turn's completion fence, THEN for its receipts — under ONE declared bound.

    "A receipt exists and none currently says `in_flight`" IS NOT SETTLEMENT. A chrome-only
    snapshot satisfies it before a word has landed, and so does the first finalized part of a
    split reply before its later parts register.

    `turn_outcome` is the fence the bot already publishes, but **IT IS NOT AN UNCONDITIONAL
    PROMISE THAT RECEIPTS SETTLED**, and three production paths prove it:

      * `_finalize_turn_effects` returns WITHOUT settling when a flight drain fails and the
        revocation fails too — deliberately, because a turn that cannot stop its own effects must
        not finalize — and `main` still emits `turn_outcome` afterwards;
      * `settle_ledger` catches its own 10-second timeout, logs that the drain worker owns the
        settle, and returns — after which the outcome is emitted while finalization is still
        background-owned;
      * **a failed registration is QUEUED** (`ReceiptService.apply` enqueues when the write
        fails, and a queued op merges per key), so a later finalize can absorb it.

    So this requires: every ts the OUTCOME names has a receipt; nothing is `in_flight`; and the
    set is STABLE across two consecutive reads. Missing or in-flight at the bound RAISE.

    **THE STABILITY WINDOW IS NOT A COMPLETION FENCE, AND THIS IS THE HONEST BOUND ON THIS
    HARNESS.** The receipt drain worker retries every `_DRAIN_INTERVAL_SECONDS` = 2.0 seconds,
    INDEFINITELY (`outbound_receipts.py`). A queued registration can therefore fail through both
    reads of the window and succeed after it: the sequence "complete-looking set → identical set
    → late split or chrome receipt" passes here undetected, and the silent case has the same hole,
    since queued chrome can materialize after two empty reads. **No finite number of identical
    polls can prove that an unobservable queue has drained** — only a durable, production-owned
    completion fact could, and adding one would mean changing the shipped receipts subsystem,
    which is outside this harness's scope.

    What that costs, stated: a surface landing after the window is missing from the row's
    `observed_ts`, and its text was never read — so a row grading "the bot answered X" saw the
    surfaces it could see and not provably all of them. `SETTLE_LIMITATION` is recorded as an
    observation on every row that comes through here, so no green row implies a completeness
    guarantee this cannot give.

    **ONE BOUND TOTAL.** The receipt wait continues inside the deadline the outcome wait started.
    """
    bound = _Deadline(deadline)
    outcome = await wait_for_telemetry(turn_id, "turn_outcome", deadline=bound.remaining(),
                                       poll=poll)
    expected = delivered_surfaces(outcome)
    previous: Optional[Tuple[Tuple[str, str], ...]] = None
    while True:
        receipts = await read_turn_receipts(team_id, channel, turn_id)
        snapshot = tuple((r.message_ts, r.state) for r in receipts)
        in_flight = [r.message_ts for r in receipts if r.state == "in_flight"]
        missing = sorted(expected - {r.message_ts for r in receipts})
        if not in_flight and not missing and previous == snapshot:
            return receipts
        if bound.expired():
            if in_flight or missing:
                raise HarnessProtocolError(
                    f"turn {turn_id} reported its outcome, but after {deadline}s "
                    f"{len(in_flight)} receipt(s) are still in_flight ({in_flight}) and "
                    f"{len(missing)} destination(s) it says it posted have no receipt at all "
                    f"({missing}). Either the turn could not revoke its own effects, its settle "
                    f"timed out into the drain worker, or a failed registration is still queued "
                    f"— the surface set is incomplete, and grading half a reply is worse than "
                    f"saying so.")
            return receipts
        previous = snapshot
        await bound.tick(poll)


async def fetch_message(channel: str, ts: str, *,
                        thread_root_ts: Optional[str] = None) -> Optional[Observed]:
    """One message by ts, from the thread it lives in or from top-level history."""
    if thread_root_ts:
        raw = await fetch_thread_complete(channel, thread_root_ts, oldest=ts, latest=ts,
                                          inclusive=True)
    else:
        data = await _call(clients().bot, "conversations_history", channel=channel,
                           oldest=ts, latest=ts, inclusive=True, limit=1)
        raw = _require_list(data, "messages", "conversations.history")
    for message in raw:
        if _checked_ts(message, "slack") == ts:
            return _as_observed(message, channel, "slack")
    return None


async def observe_turn_output(team_id: str, channel: str, turn_id: str, *,
                              deadline: float = REPLY_DEADLINE_SECONDS,
                              poll: float = REPLY_POLL_SECONDS,
                              ctx: Optional["RowContext"] = None) -> List[Observed]:
    """Everything ONE turn posted, correlated by receipt and resolved to text.

    This is what a row grades an answer against. Recording into `ctx.observed_ts` happens for
    EVERY receipt including chrome — it is the report's index to what the turn put in the room —
    while the returned `Observed` list is the PROSE only, because "did the bot answer" is a
    question about words and a thinking indicator is not an answer.

    THE PROSE IS ALSO COPIED INTO `ctx.evidence`, and that is not decoration: it is the exact text
    an assertion graded, beside the assertion's verdict. The 2026-08-01 run recorded only that row
    2's reply "lacked the seeded decision", which left no way to tell a bot that had found the
    target and summarised the number away from one that never found it. Keyed by turn_id so a row
    that reads several turns keeps them apart.
    """
    if ctx is not None and not any(name == SETTLE_LIMITATION for name, _, _ in ctx.observations):
        # BEFORE the read, not after. `await_turn_settled` can raise — a timeout, a protocol
        # error — and a row whose settle FAILED is exactly the one whose report must not look
        # like it carried a completeness guarantee. Recorded once per row whatever the row's
        # shape: it is a property of the READING, not of any one turn, so a row that reads three
        # turns has one limitation, not three.
        ctx.observe(SETTLE_LIMITATION, {"turn_id": turn_id, "nonce": ctx.nonce}, False)
    receipts = await await_turn_settled(team_id, channel, turn_id, deadline=deadline, poll=poll)
    observed: List[Observed] = []
    for receipt in receipts:
        if ctx is not None and receipt.message_ts not in ctx.observed_ts:
            ctx.observed_ts.append(receipt.message_ts)
        if not receipt.is_prose:
            continue
        message = await fetch_message(channel, receipt.message_ts,
                                      thread_root_ts=receipt.thread_root_ts)
        if message is not None:
            observed.append(message)
    if ctx is not None:
        graded = ctx.evidence.setdefault("observed_text", {})
        if isinstance(graded, dict):
            graded[turn_id] = [{"ts": o.ts, "thread_ts": o.thread_ts, "text": o.text}
                               for o in observed]
    return observed


# --------------------------------------------------------------------------- tool provenance

def is_search_or_history_name(name: str) -> bool:
    """Rows 1, 2 and 4's subject: did the bot go and LOOK rather than answer from the window?"""
    return name == SEARCH_TOOL_NAME or name.startswith(FETCH_TOOL_PREFIX)


async def read_tool_provenance_row(channel: str, message_ts: str) -> Optional[str]:
    """The raw `tools_json` recorded for one posted message, or None when no row exists.

    The harness issues the query itself, exactly as the writer does — NO new production accessor
    is added for the battery, and this is the same file and access mode cleanup already uses.
    """
    def _read() -> Optional[str]:
        path = str(db_path())
        conn = sqlite3.connect(path)
        try:
            cursor = conn.execute(
                "SELECT tools_json FROM message_tool_usage "
                "WHERE channel_id = ? AND message_ts = ?", (channel, message_ts))
            row = cursor.fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return row[0] if row[0] is not None else ""

    return await asyncio.to_thread(_read)


def _parse_tool_names(raw: str) -> Tuple[str, ...]:
    """Names off one `tools_json` payload. MALFORMED EVIDENCE RAISES; it never degrades to ().

    A row that will not parse, or parses to something that is not a list of objects carrying
    `tool_name`, is BROKEN evidence. Returning `()` for it would manufacture the exact false
    "zero tools" reading rows 1, 2 and 4 turn on — absence of the ROW is a different fact from a
    row that cannot be read, and only the first is silence.
    """
    try:
        parsed = json.loads(raw or "[]")
    except (TypeError, ValueError) as e:
        raise HarnessProtocolError(f"message_tool_usage.tools_json will not parse: {e}") from e
    if not isinstance(parsed, list):
        raise HarnessProtocolError("message_tool_usage.tools_json is not a list")
    names: List[str] = []
    for entry in parsed:
        if not isinstance(entry, dict) or "tool_name" not in entry:
            raise HarnessProtocolError(
                "message_tool_usage.tools_json holds an entry with no tool_name")
        names.append(str(entry["tool_name"]))
    return tuple(names)


async def await_tools_used_for(channel: str, message_ts: str, *,
                               required_name: Optional[Callable[[str], bool]] = None,
                               deadline: float = REPLY_DEADLINE_SECONDS,
                               poll: float = REPLY_POLL_SECONDS) -> ProvenanceRead:
    """The tool NAMES recorded for one posted message. POLLS, and RAISES rather than guessing.

    `required_name` IS THE POLL PREDICATE, NOT A POST-FILTER. The writer UNION-MERGES later passes
    into the same row, so a row can EXIST while still missing the name the caller is waiting for —
    a first pass records `fetch_channel_info`, a second adds `search_slack`. Returning on "a row
    appeared" would hand rows 2 and 4 an early, incomplete answer and fail them for a tool that
    arrived a moment later. With no predicate it returns the first row it sees.

    IT POLLS UNDER THE ROW'S DEADLINE because the write is fire-and-forget and lands shortly AFTER
    the reply the harness just observed; a single read at reply time would race it.

    THE THREE TERMINAL STATES: absent -> (False, ()); present-but-empty -> (True, ()); present and
    non-matching at the deadline -> (True, the names it holds). `row_present` is never inferred
    from `names`.
    """
    bound = _Deadline(deadline)
    last = ProvenanceRead(row_present=False, names=())
    while True:
        raw = await read_tool_provenance_row(channel, message_ts)
        if raw is not None:
            last = ProvenanceRead(row_present=True, names=_parse_tool_names(raw))
            if required_name is None or any(required_name(name) for name in last.names):
                return last
        if bound.expired():
            return last
        await bound.tick(poll)


async def read_window_anchor(team_id: str, channel: str) -> Optional[Tuple[str, int]]:
    """The stored anchor as `(floor_ts, selection_version)`, or None when no row exists.

    THE VERSION IS PART OF THE ANCHOR, not decoration. Two builds can land the same `floor_ts`
    under different selection versions, so a compare that looked only at the floor would think it
    was still holding its own write when a different build had replaced it.

    Row 8 grades "the floor moved forward", which is a comparison against the anchor stored
    BEFORE it seeded — there is no other honest baseline, since a read afterwards is the very
    value under test. Keyed by `(team_id, channel_id)`, the table's own primary key: keying by
    channel alone would match another workspace's row.
    """
    def _read() -> Optional[Tuple[str, int]]:
        conn = sqlite3.connect(str(db_path()))
        try:
            row = conn.execute(
                "SELECT floor_ts, selection_version FROM channel_window_anchor "
                "WHERE team_id = ? AND channel_id = ?", (team_id, channel)).fetchone()
        finally:
            conn.close()
        if row is None or row[0] is None:
            return None
        return str(row[0]), int(row[1] or 0)

    return await asyncio.to_thread(_read)


# ------------------------------------------------------------------------------- the fence

# The lease states that mean a scope is SPOKEN FOR. `released` is free, and an EXPIRED lease in
# any of the first three is dead. `invalidated` refuses whatever its expiry says: it is the
# deny-only state a restart installs, and it is cleared only by a human.
_FENCE_BUSY_STATES = frozenset({"armed", "active", "closing"})
_FENCE_DENY_STATE = "invalidated"
_FENCE_RELEASED_STATE = "released"


async def read_epoch_fence_lease(team_id: str, channel: str) -> Optional[Dict[str, Any]]:
    """The DURABLE lease row for one channel, or None when there has never been a fence.

    IT MUST BE THE DURABLE ROW, NOT THE RUNTIME REGISTRY. `epoch_fence._FENCES` is a dict in the
    BOT's process; this harness is a different process and would read an empty one every time and
    conclude "no fence" — the same mistake as a harness that "sets" a boot-loaded env var and
    then asserts it worked. The lease table is the only thing both processes can see.

    A MISSING TABLE IS NOT AN ERROR: it is created only by the flag-gated fence watcher, so its
    absence proves no fence has ever run here.
    """
    def _read() -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(str(db_path()))
        try:
            row = conn.execute(
                "SELECT state, expiry_ts, lease_id, test_epoch_id FROM epoch_fence_lease "
                "WHERE team_id = ? AND channel_id = ?", (team_id, channel)).fetchone()
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                return None
            raise
        finally:
            conn.close()
        if row is None:
            return None
        return {"state": str(row[0]), "expiry_ts": str(row[1]), "lease_id": str(row[2]),
                "test_epoch_id": row[3]}

    return await asyncio.to_thread(_read)


def _lease_expiry(raw: str) -> Tuple[bool, bool]:
    """`(parsed_ok, expired)` for a lease's expiry, through the canonical comparator."""
    try:
        return True, parse_ts(f"{time.time():.6f}") >= parse_ts(raw)
    except (TimestampError, TypeError, ValueError):
        return False, False


async def assert_channel_unfenced(team_id: str, channel: str) -> None:
    """PREFLIGHT — refuse to run inside somebody else's fence, and FAIL CLOSED.

    Every row in this battery is written to run UNFENCED. Under an active fence, channel
    settings, memory, steering and the window anchor are served from an in-memory overlay instead
    of SQLite, and effect authorization changes — so the rows that read the anchor, read recorded
    tool provenance or grade receipt membership would be measuring the fence rather than the
    shipped path. Documenting "unfenced" without checking it is a premise, not a fact.

    **A MALFORMED LEASE IS NOT PROOF THE CHANNEL IS FREE.** An unparseable expiry read as
    "expired", or an unrecognised state waved through, turns damaged evidence into permission —
    and the failure mode is a whole battery silently grading an overlay. The only two ways past
    this check are a recognised RELEASED row, or a recognised BUSY row whose expiry parses and
    has genuinely passed. Everything else refuses.
    """
    lease = await read_epoch_fence_lease(team_id, channel)
    if lease is None:
        return
    state = lease["state"]
    if state == _FENCE_RELEASED_STATE:
        return
    if state == _FENCE_DENY_STATE:
        raise HarnessPreflightError(
            f"{channel} holds an INVALIDATED epoch fence lease ({lease['lease_id']}). That state "
            f"is cleared only by an explicit release, by a human who has looked at why the "
            f"previous battery did not survive its process. Refusing to run.")
    if state not in _FENCE_BUSY_STATES:
        raise HarnessPreflightError(
            f"{channel} holds an epoch fence lease in an UNRECOGNISED state {state!r} "
            f"({lease['lease_id']}). A lease this harness cannot interpret is not evidence that "
            f"the channel is free.")
    parsed, expired = _lease_expiry(lease["expiry_ts"])
    if not parsed:
        raise HarnessPreflightError(
            f"{channel}'s epoch fence lease ({lease['lease_id']}, {state}) carries an unparseable "
            f"expiry {lease['expiry_ts']!r}. Damaged evidence is not permission.")
    if not expired:
        raise HarnessPreflightError(
            f"{channel} is fenced: lease {lease['lease_id']} is {state!r} until "
            f"{lease['expiry_ts']}. Every row here runs unfenced by design, so a run now would "
            f"measure the fence's overlay instead of the shipped path.")


# ------------------------------------------------------------------------------- the ledger

def ledger_paths() -> List[Path]:
    """The live ledger and its rotations, newest first.

    A long battery can rotate mid-run, so a reader that only opened `participation.jsonl` would
    lose the rows it is waiting for.
    """
    base = ledger_dir()
    paths = [base / LEDGER_LOG_NAME]
    paths.extend(base / f"{LEDGER_LOG_NAME}.{i}" for i in range(1, LEDGER_ROTATIONS + 1))
    return paths


def read_ledger_events(event: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    """Every parseable ledger object, optionally filtered by event name.

    A line that will not parse is SKIPPED rather than raised on: the ledger is append-only from a
    live process and its last line can be half-written at the instant we read it.
    """
    for path in ledger_paths():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except ValueError:
                continue
            if isinstance(payload, dict) and (event is None or payload.get("event") == event):
                yield payload


@dataclass(frozen=True)
class TriggerVerdict:
    """What ONE inbound message did — across the TWO shapes the ledger can record.

    MEASURED, not assumed. A live probe showed the two are genuinely different rows:

      * a message that WOKE the bot emits `turn_start` → `stream_render` → `model_response` →
        **`turn_outcome(kind=…)`**;
      * a message the GATE DECLINED emits `gate_start` → `gate_decision` →
        **`visible_action(kind=silence)`** and **no `turn_start` at all**.

    Rows that grade restraint (9a, 9b, 9c, 9d) ask "what did this message make the bot do", and
    that question spans both shapes. Reading only `turn_outcome` makes a declined message look
    like a correlation failure — the row reports `error` for the very restraint it was measuring.
    """

    kind: str                      # 'silence' | 'reaction_only' | 'reply' | …
    woke: bool                     # did a turn open at all?
    turn_id: Optional[str]
    source: str                    # 'turn_outcome' | 'visible_action'


def classify_trigger(events: Sequence[Dict[str, Any]]) -> Optional[TriggerVerdict]:
    """The verdict for one trigger, from that trigger's ledger rows. PURE.

    Prefers the WOKEN shape when both are present: a turn that opened and then reported its
    outcome is the more complete account, and `visible_action` is the gate's attempt record which
    can also accompany a turn.
    """
    outcome = next((e for e in events if e.get("event") == "turn_outcome"), None)
    if outcome is not None:
        return TriggerVerdict(kind=str(outcome.get("kind") or ""), woke=True,
                              turn_id=outcome.get("turn_id"), source="turn_outcome")
    started = next((e for e in events if e.get("event") == "turn_start"), None)
    visible = next((e for e in events if e.get("event") == "visible_action"), None)
    if visible is not None:
        return TriggerVerdict(kind=str(visible.get("kind") or ""), woke=started is not None,
                              turn_id=(started or {}).get("turn_id"), source="visible_action")
    return None


async def await_trigger_verdict(channel: str, trigger_ts: str, *,
                                deadline: float = REPLY_DEADLINE_SECONDS,
                                poll: float = REPLY_POLL_SECONDS) -> TriggerVerdict:
    """Poll the ledger until this trigger has a verdict in EITHER shape."""
    bound = _Deadline(deadline)
    while True:
        events = [e for e in read_ledger_events()
                  if e.get("channel_id") == channel and e.get("trigger_ts") == trigger_ts]
        verdict = classify_trigger(events)
        if verdict is not None:
            return verdict
        if bound.expired():
            raise HarnessCorrelationError(
                f"trigger {trigger_ts} in {channel} produced neither a turn_outcome nor a "
                f"visible_action within {deadline}s — the bot never judged it at all")
        await bound.tick(poll)


async def find_turn_id(channel: str, trigger_ts: str, *,
                       deadline: float = REPLY_DEADLINE_SECONDS,
                       poll: float = REPLY_POLL_SECONDS) -> str:
    """Poll the ledger for the UNIQUE `turn_start` whose (channel_id, trigger_ts) match.

    RAISES at the deadline on ZERO matches — the turn never started, which is itself the row's
    answer and must be reported as an error rather than silently waited out — and IMMEDIATELY on
    MORE THAN ONE, because two turns claiming one trigger means the row's premise is broken and
    every later assertion would land on an arbitrary one of them.
    """
    bound = _Deadline(deadline)
    while True:
        found: List[str] = []
        for row in read_ledger_events("turn_start"):
            if row.get("channel_id") != channel or row.get("trigger_ts") != trigger_ts:
                continue
            turn_id = row.get("turn_id")
            if turn_id and turn_id not in found:
                found.append(str(turn_id))
        if len(found) > 1:
            raise HarnessCorrelationError(
                f"{len(found)} turns claim trigger {trigger_ts} in {channel}: {found!r}")
        if len(found) == 1:
            return found[0]
        if bound.expired():
            raise HarnessCorrelationError(
                f"no turn_start for trigger {trigger_ts} in {channel} within {deadline}s")
        await bound.tick(poll)


async def wait_for_telemetry(turn_id: str, event: str, *,
                             deadline: float = REPLY_DEADLINE_SECONDS,
                             poll: float = REPLY_POLL_SECONDS) -> Dict[str, Any]:
    """The LAST `event` row carrying `turn_id`. Raises at the deadline rather than returning {}.

    An empty dict would read as "the turn did that and reported nothing", which is a behavioural
    claim the harness has no evidence for.
    """
    bound = _Deadline(deadline)
    while True:
        matches = [row for row in read_ledger_events(event) if row.get("turn_id") == turn_id]
        if matches:
            return matches[-1]
        if bound.expired():
            raise HarnessCorrelationError(
                f"no {event} row for turn {turn_id} within {deadline}s")
        await bound.tick(poll)


# ------------------------------------------------------------------------------- the barriers

def _barrier_dir() -> Path:
    path = (os.environ.get(_ENV_BARRIER_DIR) or "").strip()
    if not path:
        raise HarnessStartupError(
            f"{_ENV_BARRIER_DIR} is not set. Rows that freeze a turn mid-flight need the same "
            f"directory the bot was started with, and DEV_TURN_BARRIERS must name the seam.")
    return Path(path)


# WHAT EACH SEAM KEYS ITS BARRIER ON — MEASURED FROM THE PRODUCTION CALL SITES, NOT ASSUMED.
#
# `dev_barriers.operation_id` tries `turn_id`, then `message_ts`, then `attempt_id`, and the two
# seams hand it DIFFERENT contexts:
#
#   * `post_admission`    — `base.py` passes `barrier_context={"turn_id": …}`, so it keys by TURN.
#   * `post_partial_post` — `outbound_receipts.py` passes `channel_id`/`message_ts`/`owner` and
#     **no turn_id**, so it falls through to the MESSAGE TS of the surface it froze on.
#
# A live run proved the second one: the file the bot created was
# `post_partial_post.<the bot's reply ts>.0.waiting`. A row that released on a turn id would never
# match it — the turn would sit at the seam until its own timeout, holding a message that NO token
# can delete (`cant_delete_message`) or update (`streaming_state_conflict`) until it is freed.
#
# THE LESSON THIS TABLE ENCODES: derive, never restate. `post_admission` works only because the
# path is built through `dev_barriers.barrier_key`, so its slugging (a turn id's `:` becomes `_`)
# matches automatically. Everything about a barrier's identity belongs to `dev_barriers`; the
# harness's job is to hand it the right operation, and THAT is what this table pins.
SEAM_KEYED_BY = {POST_ADMISSION: "turn_id", POST_PARTIAL_POST: "message_ts"}


def barrier_operation(seam: str, *, turn_id: Optional[str] = None,
                      message_ts: Optional[str] = None) -> str:
    """The operation value `seam` will key on. Raises rather than guessing."""
    source = SEAM_KEYED_BY.get(seam)
    if source is None:
        raise HarnessError(f"unknown barrier seam {seam!r}")
    value = turn_id if source == "turn_id" else message_ts
    if not value:
        raise HarnessError(
            f"the {seam} barrier keys on {source}, and none was supplied. Keying it on anything "
            f"else produces a path no waiter is watching.")
    return str(value)


def barrier_release_path(seam: str, operation: str) -> Path:
    """`<DEV_TURN_BARRIERS_DIR>/<seam>.<operation>.0.release` — the frozen three-part key.

    The key fragment is DERIVED from `dev_barriers.barrier_key` rather than restated, so the epoch
    component stays whatever §3.11 says it is and the slugging stays whatever `dev_barriers` does.
    `operation` is what THAT SEAM keys on — see `SEAM_KEYED_BY`, and use `barrier_operation` to
    pick it rather than assuming a turn id.
    """
    key = dev_barriers.barrier_key(seam, {"turn_id": operation})
    return _barrier_dir() / f"{seam}.{key}.release"


def barrier_waiting_path(seam: str, operation: str) -> Path:
    """The `.waiting` announcement the barrier writes when it actually froze."""
    key = dev_barriers.barrier_key(seam, {"turn_id": operation})
    return _barrier_dir() / f"{seam}.{key}.waiting"


async def wait_barrier_reached(seam: str, operation: str, *,
                               deadline: Optional[float] = None,
                               poll: float = BARRIER_POLL_SECONDS) -> None:
    """Block until the turn announces it is frozen at `seam`. Raises if it never arrives.

    THE BARRIER CONSTANTS, NOT THE SLACK ONES (§9's wait table). The default deadline is the
    bot's own `DEV_TURN_BARRIERS_TIMEOUT`, because waiting longer than the bot will pause means
    waiting for a file that has already been cleaned up; the poll is `dev_barriers`' own 0.1s,
    because a 5-second tick against an in-process file handshake spends most of a frozen turn's
    budget doing nothing.
    """
    bound = _Deadline(barrier_deadline() if deadline is None else deadline)
    waiting = barrier_waiting_path(seam, operation)
    while True:
        if waiting.exists():
            return
        if bound.expired():
            raise HarnessCorrelationError(
                f"operation {operation} never reached the {seam} barrier "
                f"(watched {waiting}); is DEV_TURN_BARRIERS naming the seam?")
        await bound.tick(poll)


@asynccontextmanager
async def freeing_other_turns(seam: str, hold_operation: str) -> AsyncIterator[List[str]]:
    """Release every turn that stops at `seam` EXCEPT the one this row is holding.

    **THE SEAM IS PROCESS-GLOBAL AND ROW 7 NEEDS TWO MORE TURNS TO RUN THROUGH IT.** Measured, twice:
    with `post_partial_post` armed, A froze as intended and the row's own B and C turns then froze
    at the same seam on their first streamed chunk. Nobody was watching those keys, so B's reply
    sat until the bot's own timeout and C never produced words at all — the row reported
    `error: turn C never replied` with two of its six assertions already green, and left a
    streaming message in the channel that no token could delete until the seam let go.

    So the row holds ONE key and this frees the rest, by name, for as long as the row is running.
    It never touches `hold_operation`: that is the whole experiment.
    """
    held = barrier_waiting_path(seam, hold_operation).name
    freed: List[str] = []

    async def _reap() -> None:
        while True:
            for waiting in sorted(_barrier_dir().glob(f"{seam}.*.waiting")):
                if waiting.name == held:
                    continue
                release = waiting.parent / (waiting.name[: -len(".waiting")] + ".release")
                if not release.exists():
                    release.write_text("go", encoding="utf-8")
                    freed.append(waiting.name)
            await _sleep(BARRIER_POLL_SECONDS)

    task = asyncio.ensure_future(_reap())
    try:
        yield freed
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001 — a reaper failure is not
            pass                                     # the row's result


def release_barrier(seam: str, operation: str) -> None:
    """Free ONE frozen turn, by its key. Never the unkeyed broadcast — a concurrent row's turn
    is sitting at the same seam and must not be released by ours."""
    path = barrier_release_path(seam, operation)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("go", encoding="utf-8")


async def await_in_flight_surface(team_id: str, channel: str, turn_id: str, *,
                                  deadline: float = SLOW_TURN_DEADLINE_SECONDS,
                                  poll: float = BARRIER_POLL_SECONDS) -> str:
    """The message_ts of the first `in_flight` surface a turn owns.

    Polled BEFORE the barrier wait, because the partial-post seam keys on this value and the
    receipt row is written *before* the seam fires (`ReceiptLedger` announces on its first
    in_flight surface, after the row is written and before anything finalizes). So this is
    available in time to build the barrier path from it.
    """
    bound = _Deadline(deadline)
    while True:
        in_flight = [r.message_ts for r in await read_turn_receipts(team_id, channel, turn_id)
                     if r.state == "in_flight"]
        if in_flight:
            return sorted(in_flight, key=ts_key)[0]
        if bound.expired():
            raise HarnessCorrelationError(
                f"turn {turn_id} never registered an in_flight surface within {deadline}s, so "
                f"there is nothing for the partial-post barrier to key on")
        await bound.tick(poll)


# ---------------------------------------------------------------------------------- seeding# ---------------------------------------------------------------------------------- seeding

def mint_nonce(row: str) -> str:
    """The run's identity for one row. IT IS NEVER POSTED — it seeds the facts that are.

    Every message this battery puts in the channel reads as ordinary human chatter (owner ruling,
    2026-08-02), so there is no marker to match on. What makes a run distinguishable from the one
    before it is the FACTS it seeds: a supplier nobody has mentioned, a quantity nobody has quoted.
    Those are derived from this value, so one nonce in the report reproduces every word the run
    said, while the room never sees a token.
    """
    return f"{row}-{uuid.uuid4().hex[:8]}"


def numeric_token(nonce: str, *, digits: int = 6) -> str:
    """A RUN-UNIQUE number derived from the nonce. The raw digits behind every seeded quantity.

    A FIXED literal cannot carry a "the bot echoed our fact" assertion: the same number sits in the
    channel's history from every previous run, so a stale answer — or a lucky guess at a four-digit
    number — satisfies it without the bot having read this run's fact at all.
    """
    span = 10 ** digits
    value = int(uuid.uuid5(uuid.NAMESPACE_OID, nonce).hex[:12], 16) % (span - span // 10)
    return str(value + span // 10)


# ------------------------------------------------------------------ naturally-worded seed facts
#
# THE RULE THESE EXIST TO KEEP (owner, 2026-08-02): nothing the harness posts may look like a test.
# No token-shaped nonces, no ALL-CAPS marker words, no meta text about batteries or probes. A
# reader scrolling the channel should see coworkers talking. So unguessability moves OUT of the
# text's shape and INTO its content: a supplier that does not exist, a quantity nobody has quoted
# before, a date. Each is derived from the row's nonce, so it is reproducible from the report and
# different every run.

# WIDENED DELIBERATELY (codex, 2026-08-03), and widening is all it is. A name out of this space
# is the SEARCHABLE half of a buried fact, never the evidence on its own: chatter mints from the
# same space, nothing is ever deleted, so every past run's names stay in the channel and one of
# them will eventually be this run's. 48 x 16 x 12 is 9,216 combinations against the old 3,072 —
# it lengthens the odds, `chatter_lines` keeps the running row's own bulk clear, and the FIGURE
# seeded beside the name is what makes the pair evidence.
_COMPANY_STEMS = ("Halver", "Kest", "Bram", "Ordway", "Penfield", "Marlo", "Quillan", "Rederick",
                  "Sable", "Thorn", "Vantre", "Wexford", "Ashgrove", "Brindle", "Caldwell",
                  "Dunmore", "Everly", "Fairholt", "Garrick", "Hollings", "Ivener", "Jarrow",
                  "Keswick", "Lanmore", "Merrow", "Norbury", "Oakhurst", "Pellman", "Ravensby",
                  "Stanwick", "Tilbrook", "Ulverton", "Ambleside", "Bracken", "Crowmar",
                  "Delford", "Eastmere", "Foxley", "Grantham", "Harlow", "Inglet", "Kirby",
                  "Ludgate", "Marchant", "Netherby", "Oxwell", "Pendle", "Rookwood")
_COMPANY_TAILS = ("son", "ley", "wood", "field", "brook", "stone", "gate", "mont", "worth",
                  "bury", "dale", "ridge", "combe", "haven", "moor", "thwaite")
_COMPANY_KINDS = ("Freight", "Logistics", "Packaging", "Produce", "Dairy", "Foods", "Supply Co",
                  "Distribution", "Cold Chain", "Haulage", "Wholesale", "Provisions")

_FIRST_NAMES = ("Dana", "Marcus", "Priya", "Ellis", "Nora", "Theo", "Ivy", "Roman", "Cass",
                "Beatriz", "Owen", "Simone", "Rafa", "Junie", "Malik", "Greta")
_PLACES = ("kitchen", "back stairwell", "loading dock", "east lobby", "print room", "break room",
           "north entrance", "annex")
_FLOORS = ("second floor", "third floor", "ground floor", "mezzanine")
_TEAMS = ("ops", "warehouse", "finance", "procurement", "front-of-house", "receiving")
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August",
           "September", "October", "November", "December")


def _draw(seed: str, salt: str, span: int) -> int:
    """A stable index into a word bank, derived from the run's nonce. Never `random`.

    Deterministic so the report's nonce reproduces the exact words the run posted — the only way
    to read a failed assertion after the fact, now that nothing is deleted and the channel itself
    is the record.
    """
    return int(uuid.uuid5(uuid.NAMESPACE_OID, f"{seed}|{salt}").hex[:12], 16) % max(1, span)


def vendor_name(seed: str, salt: str = "vendor") -> str:
    """A supplier that does not exist, in ~9,000 combinations. HALF of a buried fact.

    Plausible enough that nobody reading the channel sees a test marker, and specific enough to be
    worth searching for — but ~9,000 names in a channel that is never cleared is not proof of
    anything on its own, and a row that graded a name alone would eventually credit a coincidence.
    The rows that grade information flow pair it with a run-unique `money`/`quantity` figure in the
    same sentence and require both; that pair is ~10^8.
    """
    stem = _COMPANY_STEMS[_draw(seed, f"{salt}-stem", len(_COMPANY_STEMS))]
    tail = _COMPANY_TAILS[_draw(seed, f"{salt}-tail", len(_COMPANY_TAILS))]
    kind = _COMPANY_KINDS[_draw(seed, f"{salt}-kind", len(_COMPANY_KINDS))]
    return f"{stem}{tail} {kind}"


def person_name(seed: str, salt: str = "person") -> str:
    return _FIRST_NAMES[_draw(seed, salt, len(_FIRST_NAMES))]


def quantity(seed: str, salt: str = "qty", *, digits: int = 5) -> str:
    """A run-unique count, written the way a person writes one: `8,472`.

    The comma is the point. Graders compare DIGIT-NORMALIZED (`states_number`), because a bot that
    answers "847,800 crates" to a seeded `847800` has read the fact — and the 2026-08-02 run
    graded exactly that reply as a failure on the punctuation alone.
    """
    return f"{int(numeric_token(f'{seed}|{salt}', digits=digits)):,}"


def money(seed: str, salt: str = "money") -> str:
    """A quoted price, `$41,770` — a decision value a coworker would actually write down."""
    return f"${quantity(seed, salt)}"


def date_phrase(seed: str, salt: str = "date") -> str:
    month = _MONTHS[_draw(seed, f"{salt}-month", len(_MONTHS))]
    return f"{month} {_draw(seed, f'{salt}-day', 27) + 1}"


# The work a quote could be FOR. It exists so a row can ASK about a buried fact without naming
# either half of what it grades: the trigger says "the loading dock resurfacing", the answer has
# to supply the supplier and the figure. Ambiguity here costs a confusing question, never a false
# pass — the conjunction is what carries soundness.
_WORKS = ("resurfacing", "rewiring", "repaint", "shelving refit", "door replacement",
          "drainage work", "lighting upgrade", "flooring job")


def project_phrase(seed: str, salt: str = "project") -> str:
    place = _PLACES[_draw(seed, f"{salt}-place", len(_PLACES))]
    return f"the {place} {_WORKS[_draw(seed, f'{salt}-work', len(_WORKS))]}"


def weekday(seed: str, salt: str = "weekday") -> str:
    return _WEEKDAYS[_draw(seed, salt, len(_WEEKDAYS))]


# The chatter that pushes a fact below the window floor. DELIBERATELY LOW VALUE: statements, never
# questions, never a mention — so the gate declines them and ~101 of them cost the run nothing but
# their seeding time. They are also the reason nothing here says "filler": that word in a hundred
# messages is exactly the synthetic marker the ruling removed.
#
# EVERY TEMPLATE CARRIES AT LEAST TWO VARYING SLOTS, and that is not decoration. A line is
# re-minted with a new fill when it repeats, and after eight tries it moves to the next template —
# so a template with only four possible sentences (there are four floors) exhausts itself and
# bumps, which showed up live as "sent the wrong pallet labels again" twice in a row. Measured: 7
# adjacent repeats per 101 lines with the one-slot versions, and none once every template had
# a person or a supplier in it too.
_CHATTER = (
    "{person} is out {weekday}, back the day after",
    "{vendor} moved our delivery to {weekday}",
    "the coffee machine on the {floor} is out of beans again, {person} put in an order",
    "{person} left pastries in the {place}",
    "badge reader on the {floor} door is being flaky today, {person} had to buzz in",
    "{vendor} sent the wrong pallet labels again, {person} is sorting it out",
    "printer by the {place} is jammed, {person} is poking at it",
    "{count} boxes turned up in reception for {person}",
    "{person} cleaned out the {place} fridge this morning",
    "{person} moved the {team} standup to {clock}",
    "lunch truck is parked by the {place} until {clock}, {person} is heading down",
    "the {floor} thermostat is arctic again, {person} is hunting for the panel",
    "{person} says the {team} handover notes are done",
    "wifi in the {place} dropped for a minute there, {person} noticed it too",
    "{vendor} confirmed the {count} case order",
    "{person} is covering {team} phones {weekday}",
)


def _chatter_line(seed: str, index: int, salt: int) -> str:
    key = f"{index}-{salt}"
    # ROTATED BY INDEX ALONE, not drawn and not salted. Drawing each line's template independently
    # clusters — eight lines came back with three copies of the thermostat one. Adding the retry
    # salt to the rotation was the same bug one step later: a line that retried landed on the NEXT
    # line's template, so the live channel showed "sent the wrong pallet labels again" twice in a
    # row. The salt varies the FILL; the rotation stays strict.
    #
    # THE SALT MOVES THE TEMPLATE ONLY AFTER EIGHT FAILED FILLS. Some templates have few
    # combinations — there are four floors, so the thermostat line can be written four ways — and
    # a hundred lines use each template six or seven times. With the template pinned outright, the
    # generator ran out of distinct fills at index 67 and raised. Eight tries first, then move on.
    template = _CHATTER[(_draw(seed, "chatter-order", len(_CHATTER)) + index + salt // 8)
                        % len(_CHATTER)]
    return template.format(
        person=person_name(seed, f"p-{key}"),
        vendor=vendor_name(seed, f"v-{key}"),
        weekday=weekday(seed, f"w-{key}"),
        place=_PLACES[_draw(seed, f"pl-{key}", len(_PLACES))],
        floor=_FLOORS[_draw(seed, f"fl-{key}", len(_FLOORS))],
        team=_TEAMS[_draw(seed, f"tm-{key}", len(_TEAMS))],
        count=_draw(seed, f"c-{key}", 40) + 3,
        clock=f"{_draw(seed, f'h-{key}', 8) + 9}:{'00' if _draw(seed, f'm-{key}', 2) else '30'}")


def chatter_lines(seed: str, count: int, *, avoid: Sequence[str] = (),
                  avoid_names: Sequence[str] = ()) -> List[str]:
    """`count` DISTINCT lines of channel noise, none of them stating a fact the row grades on.

    `avoid` is the row's own seeded quantities and `avoid_names` its seeded suppliers. A chatter
    line that stated either would put the graded fact ABOVE the floor, in the rendered window —
    and the row would then pass without the bot ever searching, which is the one thing it exists
    to prove.

    **THE NAMES MATTER AS MUCH AS THE NUMBERS, and that cost a review round.** Chatter mints its
    suppliers from the SAME `vendor_name` space a row seeds from, so a bulk line naming row 4's
    graded supplier is not a remote possibility — codex generated a real collision
    (`Stanwickgate Foods` in both). The pair a buried-fact row grades stays sound either way,
    but a duplicated name inside the window hands the model half the answer for free.

    **THE GUARD IS NOISE CONTROL, NOT THE PROOF.** It covers one row's own chatter and nothing
    else — not another row's, not a previous run's, and nothing is ever deleted, so names from
    this ~9,000-name space accumulate in the channel permanently. That is why a name alone was
    never sufficient evidence that information flowed, and why every buried-fact row now grades a
    NAME AND A FIGURE seeded in one sentence (~10^8 pairs): two independent collisions would have
    to land together on the same run for a false pass. The guard still earns its keep by keeping
    either half out of the window this run.
    """
    wanted = [digits_of(value) for value in avoid if digits_of(value)]
    names = [name for name in avoid_names if str(name).strip()]
    lines: List[str] = []
    seen: set = set()
    for index in range(count):
        for salt in range(64):
            line = _chatter_line(seed, index, salt)
            if (line in seen or any(states_number(line, value) for value in wanted)
                    or any(states_phrase(line, name) for name in names)):
                continue
            seen.add(line)
            lines.append(line)
            break
        else:  # pragma: no cover — 64 draws colliding means the banks shrank to nothing
            raise HarnessError(
                f"could not mint a distinct chatter line for index {index}; the word banks are "
                f"too small for {count} lines")
    return lines


# ------------------------------------------------------------------------ reading what it said

# Separators a person puts INSIDE a number — `847,800`, `12 480`, `1.250`. Stripped only BETWEEN
# digits, so sentence punctuation and word spacing survive and `18` cannot swallow the `18` in
# `1800`.
_IN_NUMBER = re.compile("(?<=\\d)[,.\\u0020\\u00a0\\u202f](?=\\d)")

# A ZERO-CENTS TAIL IS THE SAME NUMBER WRITTEN OUT: `$41,770.00` is `$41,770`. Stripped BEFORE
# the separators, or `.00` merges into the digits and `4177000` matches nothing. NON-zero cents
# are left alone on purpose — `41,770.50` is a different figure and must not satisfy a seeded
# `41,770`. The lookahead keeps a thousands group safe: in `1,000` the `,0` is followed by more
# digits, so it is not a cents tail.
_ZERO_CENTS = re.compile(r"(?<=\d)[.,]0{1,2}(?!\d)")


def digits_of(value: str) -> str:
    return "".join(ch for ch in str(value) if ch.isdigit())


def _number_normalized(value: str) -> str:
    """Zero cents dropped, then in-number separators dropped. The order matters — see `_ZERO_CENTS`."""
    return _IN_NUMBER.sub("", _ZERO_CENTS.sub("", str(value)))


def states_number(text: str, value: str) -> bool:
    """Does `text` state `value` as a number, in any ordinary human formatting?

    DIGIT-NORMALIZED ON BOTH SIDES, and bounded so a longer number does not count as a match. The
    2026-08-02 run seeded `847800` crates, the bot answered "847,800 crates.", and a verbatim
    containment check graded a correct answer as a failure. Punctuation is the writer's choice;
    the number is the fact — and that includes a model that writes a price out as `$41,770.00`,
    which this docstring used to claim and the code used to reject.
    """
    wanted = digits_of(_number_normalized(value))
    if not wanted:
        return False
    normalized = _number_normalized(text)
    return re.search(rf"(?<!\d){re.escape(wanted)}(?!\d)", normalized) is not None


def states_phrase(text: str, phrase: str) -> bool:
    """Does `text` NAME `phrase` — case- and whitespace-insensitively, on word boundaries?

    BOUNDED, because plain containment is not naming. `Kestwood Freight` sits inside
    `NotKestwood Freightening`, and the name half of a buried-fact conjunction is only evidence
    when the reply actually NAMES the supplier — a claim a substring match does not carry.
    Surrounding punctuation is fine: `(Kestwood Freight),` names it.
    """
    needle = " ".join(str(phrase).split()).casefold()
    if not needle:
        return False
    haystack = " ".join(str(text).split()).casefold()
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


# The ack predicate is STRICT BY DESIGN (codex, ack rounds #1/#2). No deterministic check can
# decide "would these words read the same if the post never happened", so the two error directions
# are priced differently: a false FAILURE surfaces in a recording run and gets reviewed; a false
# PASS is a reporting regression the oracle exists to catch and nobody ever sees. So the words
# must FULL-MATCH a receipt grammar — every word accounted for by a receipt phrase, not merely one
# receipt word present somewhere ("Got it — I put it where it belongs." contains a receipt and is
# a report). "Thanks for answering!" and a plain-text "Thanks, Priya!" failing here is the strict
# side of that trade, accepted on purpose; a @-mention form of the same ack passes (mentions are
# address, not content). No length limit: the grammar bounds what the words can SAY, and
# `_receipt_shape` bounds how much of it there is — one phrase, or a pair of two different ones
# (codex ack #3/#4 — repetition of receipt phrases is not an acknowledgment either).
_ACK_PHRASES = tuple(p.split() for p in (
    "thank you so much", "thanks so much", "thanks a lot", "thank you", "thanks", "thx", "ty",
    "cheers", "much appreciated", "appreciate it", "appreciate that", "appreciated",
    "got it", "noted", "good to know", "great", "perfect", "nice", "nice one", "awesome",
    "sweet", "roger that", "roger", "finally", "glad to hear it", "glad to hear", "love it",
    "ok", "okay", "makes sense", "that helps",
    "thanks for the update", "thanks for the heads up", "thanks for letting me know",
))
# The stem list survives as the reason-giver and the belt: a stem names WHY the words failed
# ("it reports the post") where the grammar can only say they are not a receipt. It stays words a
# NEWS-directed ack has no need of — "thanks for the update" is grammar, so "update" is no stem.
_ACK_REPORTING = re.compile(
    r"\b(post\w*|thread\w*|repl(?:y|ies|ied)|answer\w*|done|sent|sorted|handled|forwarded|"
    r"shared|passed|updated|filed|took care|taken care|will do|on it)\b"
    r"|over there|in there|with them|archives/",
    re.IGNORECASE)
# Real Slack ids are uppercase alphanumeric; the scenario corpus mints readable synthetic ids
# ("U-tessa"), and the predicate has to recognize its callers' own mentions (codex ack #14).
_SLACK_MENTION = re.compile(r"<@([A-Z][A-Za-z0-9\-]*)>")
_VOCATIVE_AFTER = re.compile(r"\s*(?:[.,;:!?…\-—]|$)")
_VOCATIVE_BEFORE = re.compile(r"(?:^|[.,;:!?…\-—])\s*$")


def _tail_is_receipt_phrase(before: str) -> bool:
    """Do the words since the last punctuation form a complete plain receipt phrase? The seam
    allowance that keeps the measured "thanks <@U…> — got it" legal: a mention directly after a
    finished receipt phrase, with punctuation following, is address between units."""
    segment = re.split(r"[.,;:!?…\-—]", before)[-1]
    return tuple(w.casefold() for w in _WORD.findall(segment)) in _ACK_PHRASE_SET


def _strip_address_mentions(text: str,
                            addressee_ids: Sequence[str]) -> Tuple[str, Optional[str]]:
    """Remove mentions that are ADDRESS; name the violation when one is CONTENT (codex ack #13).

    The mention path must obey the same structural rules as plain names: removal only at a
    vocative boundary (start-or-punctuation before, punctuation-or-end after) or at a unit seam
    (directly after a complete receipt phrase, punctuation after) — so
    "Thanks for <@U…> closing the loop." and "Thanks for checking <@U…>." keep their mentions
    and fail, while "Thanks, <@U…>!" and "thanks <@U…> — got it" strip. When the caller vouches
    ids, a mention of anyone else — the bot itself included — is a violation outright.
    """
    while True:
        m = _SLACK_MENTION.search(text)
        if m is None:
            return text, None
        if addressee_ids and m.group(1) not in addressee_ids:
            return text, f"it mentions <@{m.group(1)}>, who is not the person being answered"
        before, after = text[:m.start()], text[m.end():]
        if _VOCATIVE_AFTER.match(after) is None or not (
                _VOCATIVE_BEFORE.search(before) or _tail_is_receipt_phrase(before)):
            return text, "a mention sits inside the words rather than addressing anyone"
        text = before + " " + after
# Slack transports emoji in message text as colon-codes (`:wave:`, `:+1:`). A code is only safe
# to treat as emoji when its ALIAS is one this predicate can vouch for as a receipt — a colon
# wrapper is just text, and `:done:` / `:posted:` / `:90742:` would otherwise launder the exact
# content every other check exists to catch (codex ack #4). Unknown aliases FAIL, per the
# strictness doctrine; workspace-custom receipt emoji can be added here when they earn it.
_SLACK_EMOJI_CODE = re.compile(r":([a-z0-9_+\-]+):")
# ONE mapping, alias → rendered base character, and both vouched sets derive from it (codex ack
# #6): the same acknowledgment must grade the same whether Slack hands the oracle `:hearts:` or
# `♥️`, or the two surfaces drift. The parity test walks every entry in both representations.
_ACK_EMOJI = {
    "+1": "👍", "thumbsup": "👍",
    "wave": "👋",
    "pray": "🙏",
    "raised_hands": "🙌",
    "tada": "🎉",
    "clap": "👏",
    "heart": "❤", "hearts": "♥",
    "heavy_check_mark": "✔", "white_check_mark": "✅",
    "ballot_box_with_check": "☑",
    "100": "💯",
    "muscle": "💪",
    "handshake": "🤝",
    "saluting_face": "🫡", "salute": "🫡",
    "bow": "🙇",
    "fire": "🔥",
    "rocket": "🚀",
    "ok_hand": "👌",
    "ok": "🆗",  # Slack's :ok: renders 🆗, not 👌 (codex ack #7's external golden)
}
_ACK_EMOJI_ALIASES = frozenset(_ACK_EMOJI)
# A receipt is at most a couple of emoji beside (or instead of) the words — a wall of :wave: is
# noise wearing an emoji's clothes.
ACK_MAX_EMOJI = 3
# The rendered-emoji twin, derived from the SAME mapping, base forms only (VS16 stripped before
# matching). Unicode category So is NOT an emoji test (codex ack #5): ℗, ⌘ and ♩ are So and
# acknowledge nothing, so any So character OUTSIDE this set fails — which also makes a
# multi-person ZWJ cluster (👨‍👩‍👧‍👦) a strict-side rejection rather than a miscounted "four
# emoji". Skin-tone modifiers are category Sk and ride along untouched.
_ACK_EMOJI_CHARS = frozenset(_ACK_EMOJI.values())
_VS16 = "️"
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

_ACK_PHRASE_SET = frozenset(tuple(p) for p in _ACK_PHRASES)

# Gratitude heads that can take a "for <reason>" tail — and the reason is a VOUCHED SET, not
# free prose (codex ack #10 closed the argument: the ruled boundary is EVENT-DEPENDENCE, and no
# grammatical screen — actor pronouns, stems, length — can prove a tail does not depend on the
# post having happened; "thanks for successful placement" needs no pronoun and no stem). The set
# holds the measured reasons (real trials: "closing the loop", "checking") plus their near
# neighbors, all acts OF THE INTERLOCUTOR that stand whether or not any post landed; it grows
# the way the phrase list does — when a real ack is measured, not speculatively.
_ACK_GRATITUDE_HEADS = (("thanks",), ("thank", "you"), ("thx",), ("ty",))
_ACK_REASONS = frozenset(tuple(p.split()) for p in (
    "checking", "confirming", "clarifying", "closing the loop", "flagging this", "flagging it",
    "flagging that", "following up", "the follow up", "the update", "the heads up",
    "the quick turnaround", "the context", "the clarification", "the info", "the information",
    "the detail", "the details", "tracking that down", "digging that up",
))


def _is_gratitude_with_reason(seg: Tuple[str, ...]) -> bool:
    for head in _ACK_GRATITUDE_HEADS:
        h = len(head)
        if seg[:h] == head and len(seg) > h + 1 and seg[h] == "for" and (
                seg[h + 1:] in _ACK_REASONS):
            return True
    return False


def _receipt_shape(tokens: List[str]) -> bool:
    """Is the WHOLE token list one receipt unit, or a receipt phrase followed by a different unit?

    A unit is a receipt phrase, or a gratitude head with a "for"-tail (see above). The pair is a
    literal structure, not a count (codex ack #4): the measured maximum is the complementary pair
    ("Got it — thanks.", "Got it — thanks for closing the loop."), and "thanks thanks" is
    repetition, not a receipt. Every word must earn its place — one receipt word amid free prose
    is exactly the false pass codex's round-2 counterexamples demonstrated."""
    t = tuple(tokens)
    if t in _ACK_PHRASE_SET or _is_gratitude_with_reason(t):
        return True
    for i in range(1, len(t)):
        head, tail = t[:i], t[i:]
        if head in _ACK_PHRASE_SET and head != tail and (
                tail in _ACK_PHRASE_SET or _is_gratitude_with_reason(tail)):
            return True
    return False


def origin_ack_violation(text: str, *, fragments: Sequence[str] = (),
                         addressees: Sequence[str] = (),
                         addressee_ids: Sequence[str] = ()) -> Optional[str]:
    """Why origin words are NOT the brief non-reporting acknowledgment the owner allowed — or None.

    OWNER RULING 2026-08-03: on a cross-thread turn the origin may get a short human
    acknowledgment to the person who handed over the piece ("Got it — thanks.", "👍") — words that
    would read exactly the same if the post had never happened. What stays prohibited is the post
    leaking back: reporting it, pointing at it, or restating any part of the answer here. The
    deterministic checks, strict by design (see the note above the grammars):

    - DIGITS: every figure the harnesses seed is numeric, and an ack has no business carrying ANY
      number — a digit in the origin is content that belongs only in the target thread.
    - ANSWER FRAGMENTS: the caller names the non-numeric halves of its seeded answer (a supplier
      the origin never saw), and naming one here is the answer leaking, not gratitude.
    - ADDRESSEES: the caller may name the people in the room, and their names are ADDRESS, the
      same rule as @-mentions (measured in: a real trial wrote "Thanks, Tessa." and the grammar
      called a person's name free prose). Stripped word-bounded, full names and their word parts,
      before the grammar and only for the grammar — a name is never a stem, a digit, or content.
    - REPORTING STEMS: post/thread/reply/answer*/done/sent/sorted/"with them"/… — the measured
      "Done." regression fails by name, and first-person past action fails with it. Checked
      before the grammar so the finding names the offence.
    - EMOJI, VOUCHED ONLY: a Slack `:code:` counts as emoji only when its alias is in
      `_ACK_EMOJI_ALIASES` — `:done:` and `:90742:` are text in a colon costume and fail
      (codex ack #4) — and a rendered symbol only when it is in `_ACK_EMOJI_CHARS`, because
      category So is not an emoji test (codex ack #5: ⌘ acknowledges nothing). At most
      `ACK_MAX_EMOJI` emoji, codes and rendered combined.
    - RECEIPT GRAMMAR, FULL-MATCH AND SHAPED: every word — in any script, via Unicode word
      extraction — must form ONE receipt phrase or a pair of two DIFFERENT ones ("Got it —
      thanks."); "thanks thanks" is repetition. A wordless line passes only when an actual emoji
      is there to do the acknowledging — never bare punctuation or a bare mention (codex ack #3:
      "..." acknowledges nothing).
    """
    words = (text or "").strip()
    if not words:
        return None
    # Mentions and vouched emoji codes come out before the digit check: an ADDRESS-position
    # <@U123ABCDE> is address, `:+1:` is an emoji, and neither one's digits are the answer's. An
    # UNKNOWN alias, an unvouched id, or a mention sitting inside a clause never gets that pass —
    # each fails here, before it can launder anything.
    stripped, mention_violation = _strip_address_mentions(words, addressee_ids)
    if mention_violation:
        return mention_violation
    aliases = _SLACK_EMOJI_CODE.findall(stripped)
    unknown = [a for a in aliases if a not in _ACK_EMOJI_ALIASES]
    if unknown:
        return f"it carries an emoji code I cannot vouch for (:{unknown[0]}:)"
    stripped = _SLACK_EMOJI_CODE.sub(" ", stripped)
    if re.search(r"\d", stripped):
        return "it carries a figure, and content belongs only in the target thread"
    for fragment in fragments:
        if states_phrase(stripped, fragment):
            return f"it carries part of the answer ({fragment!r})"
    hit = _ACK_REPORTING.search(stripped)
    if hit:
        return f"it reports the post ({hit.group(0)!r})"
    symbols = [ch for ch in words.replace(_VS16, "") if unicodedata.category(ch) == "So"]
    unvouched_symbols = [ch for ch in symbols if ch not in _ACK_EMOJI_CHARS]
    if unvouched_symbols:
        return f"it carries a symbol I cannot vouch for ({unvouched_symbols[0]!r})"
    emoji_count = len(aliases) + len(symbols)
    if emoji_count > ACK_MAX_EMOJI:
        return f"{emoji_count} emoji is a pile, not an acknowledgment"
    # VOCATIVE POSITIONS ONLY (codex ack #11/#12): a name is stripped only with a vocative
    # boundary on BOTH sides — start-or-punctuation before it, punctuation-or-end after it — so
    # "Thanks, Tessa." and "Tessa, thanks." strip while a name inside a clause never does:
    # "Thanks for ChatGPT closing the loop." and "Thanks for checking Tessa." both keep the name
    # and die in the grammar. The comma-less "Thanks Tessa." failing too is the strict side.
    for name in addressees:
        for part in (str(name), *str(name).split()):
            stripped = re.sub(
                rf"(^|[.,;:!?…\-—]\s*){re.escape(part)}(?=\s*(?:[.,;:!?…\-—]|$))", r"\1",
                stripped, flags=re.IGNORECASE)
    tokens = [w.casefold() for w in _WORD.findall(stripped)]
    if tokens:
        if not _receipt_shape(tokens):
            return ("its words do not full-match the receipt grammar — the license is for "
                    "acknowledgments only")
        return None
    if not emoji_count:
        return ("it has no words and no emoji — punctuation or a bare mention acknowledges "
                "nothing")
    return None


async def pace(seconds: float) -> None:
    """Wait one poll interval, through the harness's own clock seam.

    Rows call this rather than `asyncio.sleep` so a fake clock governs them too — and so a reader
    can tell a POLL INTERVAL (legitimate) from a fixed sleep standing in for a wait (forbidden by
    §9 as a race inducer).
    """
    await _sleep(seconds)


async def post_seed(channel: str, text: str, *, thread_ts: Optional[str] = None,
                    ctx: Optional["RowContext"] = None) -> str:
    """Post ONE seed as the human operator and record its ts in the row's bookkeeping.

    The USER token, always: the bot token cannot trigger the bot, so a seed posted with it is a
    message the gate never judges. The ts is kept because the report is the only index back to a
    message the run posted — nothing here deletes anything.
    """
    params: Dict[str, Any] = {"channel": channel, "text": text}
    if thread_ts:
        params["thread_ts"] = thread_ts
    data = await _call(clients().user, "chat_postMessage", **params)
    ts = data.get("ts")
    if not isinstance(ts, str) or not ts:
        raise HarnessProtocolError("chat.postMessage returned no ts")
    if ctx is not None:
        ctx.seeded_ts.append(ts)
    return ts


async def seed_messages(channel: str, texts: Sequence[str], *, thread_ts: Optional[str] = None,
                        ctx: Optional["RowContext"] = None) -> List[str]:
    """Bulk-seed, paced at `SEED_PACE_SECONDS` BETWEEN posts.

    Slack's documented guidance for `chat.postMessage` is roughly one message per second per
    channel. A 120-reply seed therefore takes ~120 seconds and a 101-root seed ~101 seconds before
    the row's trigger goes out. THAT IS EXPECTED AND IS NOT A HANG.
    """
    stamps: List[str] = []
    for index, text in enumerate(texts):
        if index:
            await _sleep(SEED_PACE_SECONDS)
        stamps.append(await post_seed(channel, text, thread_ts=thread_ts, ctx=ctx))
    return stamps


async def record_observed(ctx: "RowContext", observations: Iterable[Observed], *,
                          ours: PartyIdentity, operator: PartyIdentity) -> None:
    """Route what a wait SAW into the report's three bookkeeping buckets, by AUTHOR.

    Our bot's pair, our operator's, or neither — the third being a message some other app put
    in the room during our exchange, which the report lists so a reader can tell it apart from
    ours. Nothing here is deleted; the buckets are an index, not a work list.
    """
    for observed in observations:
        if observed_matches(observed, ours):
            if observed.ts not in ctx.observed_ts:
                ctx.observed_ts.append(observed.ts)
        elif observed_matches(observed, operator):
            if observed.ts not in ctx.seeded_ts:
                ctx.seeded_ts.append(observed.ts)
        elif observed.ts not in ctx.external_ts:
            ctx.external_ts.append(observed.ts)


# ------------------------------------------------------------------- row state and restores

@dataclass
class RowContext:
    row: str
    nonce: str
    channel: str
    # The workspace, so a row can key the receipt / anchor / lease stores without a second
    # `auth.test`, and the VERIFIED second party, so row 9a reuses the preflight's `bots.info`
    # answer instead of resolving Claude Tag a second time.
    team_id: str = ""
    claude: Optional[PartyIdentity] = None
    seeded_ts: List[str] = field(default_factory=list)      # WE posted these (user token)
    observed_ts: List[str] = field(default_factory=list)    # OUR BOT posted these (bot token)
    external_ts: List[str] = field(default_factory=list)    # a THIRD APP posted these
    restores: List["Restore"] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)
    assertions: List[Tuple[str, bool]] = field(default_factory=list)
    observations: List[Tuple[str, Any, bool]] = field(default_factory=list)
    notes: str = ""

    def assert_that(self, name: str, ok: bool) -> None:
        """Record one GRADED claim. Any false one makes the row `fail`."""
        self.assertions.append((name, bool(ok)))

    def observe(self, name: str, value: Any, observed: bool) -> None:
        """Record something the row SAW but cannot soundly grade. Never changes the status."""
        self.observations.append((name, value, bool(observed)))


@dataclass(frozen=True)
class Restore:
    """A durable value a row changed, and enough to put it back honestly.

    `existed` IS SEPARATE FROM `prior`, and conflating them corrupts settings. `channel_settings`
    columns are nullable and NULL MEANS "inherit the global default" — a real, intended value. So
    `prior=None` is ambiguous on its own: it could mean the column was NULL, or that the row did
    not exist. Restoring the first by DELETING the row would erase every OTHER setting on that
    channel.
    """

    kind: str                 # 'channel_setting' | 'channel_memory' | 'steering'
    key: str                  # the column, the memory row id, or ''
    prior: Any                # the value read BEFORE the change
    existed: bool             # did the ROW/VALUE exist at all before the change?


@dataclass(frozen=True)
class CleanupResult:
    """What a row PUT BACK. Messages are not in this picture: nothing is ever deleted.

    Only durable bot state appears here — a channel setting, a memory row, the window anchor —
    because those are configuration the next turn reads, not conversation the owner is reading.
    """

    restored: Tuple[str, ...]
    restore_failures: Tuple[str, ...]   # restores that could NOT be applied, by kind:key


def _restore_window_anchor(conn: sqlite3.Connection, restore: Restore, channel: str,
                           team_id: str) -> None:
    """COMPARE-AND-RESTORE over the COMPLETE anchor tuple.

    `key` is `"<floor_ts>|<selection_version>"` — the anchor WE left behind, and the compare
    value. `prior` is the `(floor_ts, selection_version)` that was there before us, or None.

    COMPARE ON BOTH COLUMNS. Two builds can arrive at the same `floor_ts` under different
    selection versions; a compare that looked only at the floor would mistake another build's
    anchor for its own and overwrite it.

    Losing the compare is not a bug and not our litter — it means a legitimate turn advanced the
    anchor after us — but it IS a durable change the battery made and could not undo, so it is
    reported as a restore failure and the row lands on `unrestored`. Silently claiming success
    here would hide the one durable change this row can make and fail to undo.
    """
    if not team_id:
        raise HarnessError("a window_anchor restore needs the workspace it was read from")
    expected_floor, _, expected_version = str(restore.key).partition("|")
    if not expected_version:
        raise HarnessError(f"a window_anchor restore key must be '<floor_ts>|<selection_version>', "
                           f"got {restore.key!r}")
    if not restore.existed:
        cursor = conn.execute(
            "DELETE FROM channel_window_anchor "
            "WHERE team_id = ? AND channel_id = ? AND floor_ts = ? AND selection_version = ?",
            (team_id, channel, expected_floor, int(expected_version)))
    else:
        prior_floor, prior_version = restore.prior
        cursor = conn.execute(
            "UPDATE channel_window_anchor SET floor_ts = ?, selection_version = ? "
            "WHERE team_id = ? AND channel_id = ? AND floor_ts = ? AND selection_version = ?",
            (prior_floor, int(prior_version), team_id, channel, expected_floor,
             int(expected_version)))
    if cursor.rowcount < 1:
        raise HarnessError(
            f"the window anchor for {channel} is no longer {restore.key!r}; a concurrent advance "
            f"won the compare and the battery's floor was left in place")


def apply_restore(restore: Restore, channel: str, team_id: str = "") -> None:
    """Put one durable value back. RAISES on anything it cannot do, so the caller can record it.

    | `existed` | `prior`                | Restore does                                    |
    |-----------|------------------------|-------------------------------------------------|
    | `True`    | any value incl. `None` | write `prior` back — `None` writes NULL, which  |
    |           |                        | is "inherit"                                    |
    | `False`   | ignored                | DELETE the row / memory entry — it genuinely    |
    |           |                        | was not there                                   |
    """
    conn = sqlite3.connect(str(db_path()), isolation_level=None)
    try:
        if restore.kind == "channel_setting":
            if not restore.existed:
                conn.execute("DELETE FROM channel_settings WHERE channel_id = ?", (channel,))
                return
            columns = tuple(str(row[1])
                            for row in conn.execute("PRAGMA table_info(channel_settings)"))
            if restore.key not in columns:
                raise HarnessError(
                    f"channel_settings has no column {restore.key!r}; refusing to guess")
            # The column name is interpolated only after it has been matched against the live
            # schema — a bare f-string here would take whatever a caller typed straight into SQL.
            cursor = conn.execute(
                f"UPDATE channel_settings SET {restore.key} = ? WHERE channel_id = ?",
                (restore.prior, channel))
            if cursor.rowcount < 1:
                raise HarnessError(
                    f"channel_settings row for {channel} is gone; cannot restore {restore.key}")
        elif restore.kind == "channel_memory":
            if not restore.existed:
                conn.execute("DELETE FROM channel_memory WHERE id = ?", (restore.key,))
                return
            cursor = conn.execute("UPDATE channel_memory SET content = ? WHERE id = ?",
                                  (restore.prior, restore.key))
            if cursor.rowcount < 1:
                raise HarnessError(f"channel_memory row {restore.key} is gone; cannot restore it")
        elif restore.kind == "steering":
            # Steering is the channel's single `scope='policy'` memory row (database.py's
            # set_channel_policy_async), so the key is '' and the channel identifies it.
            if not restore.existed:
                conn.execute(
                    "DELETE FROM channel_memory WHERE scope = 'policy' AND channel_id = ?",
                    (channel,))
                return
            cursor = conn.execute(
                "UPDATE channel_memory SET content = ? WHERE scope = 'policy' AND channel_id = ?",
                (restore.prior, channel))
            if cursor.rowcount < 1:
                raise HarnessError(f"no steering row for {channel}; cannot restore it")
        elif restore.kind == "window_anchor":
            _restore_window_anchor(conn, restore, channel, team_id)
        else:
            raise HarnessError(f"unknown Restore.kind {restore.kind!r}")
    finally:
        conn.close()


async def cleanup_row(ctx: RowContext) -> CleanupResult:
    """Put back the DURABLE STATE a row changed. It deletes nothing. NEVER RAISES.

    **NO MESSAGE IS EVER REMOVED** (owner ruling, 2026-08-02). Everything the battery posts stays
    in the channel: the owner watches the run happen and reads the room afterwards, so deleting
    the seeds would destroy the record they are reading, and the earlier deleting version also
    took a failed assertion's evidence with it. `seeded_ts`, `observed_ts` and `external_ts` are
    still collected — they ride in the report as bookkeeping, so a reader can find every message
    a row put in the room without matching on its content.

    What IS put back is bot configuration: a channel setting, a memory row, the window anchor.
    Those are state the next turn READS, not conversation, and a row that left the window anchored
    at a floor it invented would change how every later turn in this channel builds.

    A restore that cannot be applied is RECORDED rather than raised, and it downgrades the row
    (`status_for`): the battery changed durable state and could not undo it, which the report has
    to say out loud even when every assertion held.
    """
    restored: List[str] = []
    restore_failures: List[str] = []

    for restore in ctx.restores:
        label = f"{restore.kind}:{restore.key}"
        try:
            await asyncio.to_thread(apply_restore, restore, ctx.channel, ctx.team_id)
            restored.append(label)
        except Exception:  # noqa: BLE001 — a channel left on state the battery invented is
            restore_failures.append(label)   # exactly what the downgrade exists to surface

    return CleanupResult(restored=tuple(restored), restore_failures=tuple(restore_failures))


# ------------------------------------------------------------------------------- the report

def status_for(ctx: RowContext, cleanup: CleanupResult) -> str:
    """The row's verdict. OBSERVATIONS CANNOT CHANGE IT.

    They carry what the row SAW in the direction it cannot soundly grade — today exactly one
    thing, row 1's tool reading in the ABSENCE direction — and a runner that promoted one to an
    assertion would be manufacturing the false pass the split exists to prevent.

    MESSAGES LEFT IN THE CHANNEL ARE NOT A DEFECT and never appear here: the battery is not
    supposed to remove them. `unrestored` is the one non-behavioural downgrade — durable bot state
    the row changed and could not put back.
    """
    if any(not ok for _, ok in ctx.assertions):
        return "fail"
    if cleanup.restore_failures:
        return "unrestored"
    return "pass"


def build_report(ctx: RowContext, *, status: str, started_at: float, finished_at: float,
                 cleanup: Optional[CleanupResult] = None) -> Dict[str, Any]:
    """One row object, in §7.1a's pinned schema.

    THE THREE TS LISTS ARE THE ROW'S BOOKKEEPING, and now that nothing is deleted they are the
    only index into what the run put in the room: `seeded_ts` (we posted), `observed_ts` (our bot
    posted) and `external_ts` (a third app posted). A reader with the report can find every one of
    them by ts, without matching on text — which is what lets the messages read as ordinary
    chatter and still be traceable.
    """
    empty = CleanupResult(restored=(), restore_failures=())
    result = cleanup if cleanup is not None else empty
    return {
        "row": ctx.row,
        "status": status,
        "nonce": ctx.nonce,
        "started_at": started_at,
        "finished_at": finished_at,
        "seeded_ts": list(ctx.seeded_ts),
        "observed_ts": list(ctx.observed_ts),
        "external_ts": list(ctx.external_ts),
        "evidence": dict(ctx.evidence),
        "assertions": [{"name": name, "ok": ok} for name, ok in ctx.assertions],
        "observations": [{"name": name, "value": value, "observed": observed}
                         for name, value, observed in ctx.observations],
        "cleanup": {
            "restored": list(result.restored),
            "restore_failures": list(result.restore_failures),
        },
        "notes": ctx.notes,
    }


def write_report(rows: Sequence[Dict[str, Any]], out: Path) -> None:
    """The run writes ONE array of row objects. Its only output."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(list(rows), indent=2, default=str) + "\n", encoding="utf-8")
