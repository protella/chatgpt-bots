from __future__ import annotations

import asyncio
import heapq
import re
import time
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Pattern, Sequence, Set, Tuple
from urllib.parse import parse_qs, urlsplit

from slack_sdk.errors import SlackApiError

from base_client import HistoryFetchError
from config import config
from slack_client.history_fetch import (FetchBudget, HistoryPageError, iter_pages,
                                        slack_error_code)
from slack_client.messaging import is_self_chrome_message
from slack_client.normalizer import (ORIGIN_HISTORY, ORIGIN_REPLIES, TimestampError,
                                     normalize_slack_message, parse_ts)
from tool_registry import stage_discovered_root


class SearchBackend(Enum):
    """WHICH implementation answers `search_slack` on this surface. PRIVATE — it is never a
    schema field, never a config value, and the model is never told which one ran.

    A DM keeps Slack's own assistant index (`assistant.search.context`), which needs an
    action_token Slack mints only on mention/DM events and reaches the whole workspace.
    A channel or MPIM gets a bot-token keyword scan of THAT channel, which needs no token — the
    two live defects this split fixes are an ambient channel turn that cannot search at all
    ("event carried no action_token") and an index that returned zero exact-recall hits for
    content seeded twelve hours earlier.

    `SERVICE_INDEX` is a RESERVED seam for the later service-account backend (shape 2). It is
    declared so the selector's shape is honest about what is coming and is deliberately NOT
    selectable: nothing returns it, and no config value names it.
    """

    ASSISTANT_CONTEXT = "assistant_context"
    IN_CHANNEL_SCAN = "in_channel_scan"
    SERVICE_INDEX = "service_index"  # reserved — not selectable


def search_backend_for(ctx: Any) -> SearchBackend:
    """The backend for one request — a pure function of the SURFACE, and nothing else.

    A true 1:1 IM is the only DM: `ToolContext.is_dm` is stamped from `is_dm_conversation`, which
    classifies an MPIM as channel-shaped, so a group DM lands on the in-channel scan with every
    other multi-user surface.
    """
    return (SearchBackend.ASSISTANT_CONTEXT if getattr(ctx, "is_dm", False)
            else SearchBackend.IN_CHANNEL_SCAN)


# ---------------------------------------------------------------- matching (spec §S2, pure)
#
# EXACT PHRASE FIRST, THEN OR-WITH-SCORING, and never "every token must appear". Strict AND is
# too brittle for the way a person asks: "what figure did we settle on?" has to be able to find
# "the accepted quote was $41,770". Stopword exclusion is what keeps the OR from matching on
# "the", and it is a PINNED LITERAL LIST — no corpus-derived weighting (no IDF) in v1, because
# a rare-token weight computed from whatever the scan happened to fetch makes the same query
# rank differently on two runs over the same channel.

# Function words that carry no retrieval signal. Numbers are never here: a figure is usually the
# most distinctive thing in a question.
_SEARCH_STOPWORDS = frozenset({
    "a", "about", "after", "again", "all", "also", "am", "an", "and", "any", "anyone", "are",
    "as", "at", "back", "be", "because", "been", "before", "being", "but", "by", "can", "could",
    "did", "do", "does", "doing", "done", "down", "each", "even", "ever", "every", "for", "from",
    "get", "got", "had", "has", "have", "he", "her", "here", "hers", "him", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "just", "know", "like", "me", "might", "mine", "more",
    "most", "much", "must", "my", "no", "nor", "not", "now", "of", "off", "on", "one", "only",
    "or", "other", "our", "ours", "out", "over", "own", "really", "said", "same", "say", "says",
    "she", "should", "so", "some", "still", "such", "than", "that", "the", "their", "theirs",
    "them", "then", "there", "these", "they", "thing", "things", "this", "those", "to", "too",
    "up", "us", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "would", "you", "your", "yours",
})

# A ZERO-CENTS TAIL IS THE SAME NUMBER WRITTEN OUT (`$41,770.00` is `$41,770`), stripped BEFORE
# the separators or `.00` merges into the digits. NON-zero cents are left alone: `41,770.50` is a
# different figure. Same two expressions, same order, as the live battery's `states_number` — the
# grader and the retriever must agree about when two numbers are the same number.
_ZERO_CENTS_RE = re.compile(r"(?<=\d)[.,]0{1,2}(?!\d)")
_IN_NUMBER_RE = re.compile("(?<=\\d)[,.\u0020\u00a0\u202f](?=\\d)")
# Word-bounded alphanumeric runs, Unicode-aware: `[^\W_]` is "word character except underscore",
# so `résumé` and `41770` tokenize whole while `snake_case` splits.
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def normalize_search_text(raw: Any) -> str:
    """NFKC + casefold + numeric-separator + whitespace normalization — ONE function, used for
    both the query and every candidate, so the two can never be normalized differently."""
    text = unicodedata.normalize("NFKC", str(raw or "")).casefold()
    text = _IN_NUMBER_RE.sub("", _ZERO_CENTS_RE.sub("", text))
    return " ".join(text.split())


def search_tokens(normalized: str) -> List[str]:
    return _TOKEN_RE.findall(normalized)


@dataclass(frozen=True)
class SearchQuery:
    """One parsed query: the normalized phrase and its distinct non-stopword tokens."""

    raw: str
    phrase: str
    content_tokens: Tuple[str, ...]
    phrase_re: Optional[Pattern[str]] = field(default=None, compare=False, repr=False)

    def phrase_hit(self, normalized_text: str) -> bool:
        """WORD-BOUNDED, never bare containment: `cert` must not match inside `certificate`."""
        return bool(self.phrase_re and self.phrase_re.search(normalized_text))


def build_search_query(raw: str) -> SearchQuery:
    phrase = normalize_search_text(raw)
    tokens = tuple(dict.fromkeys(t for t in search_tokens(phrase)
                                 if t not in _SEARCH_STOPWORDS))
    pattern = (re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", re.UNICODE) if phrase else None)
    return SearchQuery(raw=str(raw), phrase=phrase, content_tokens=tokens, phrase_re=pattern)


def score_search_text(query: SearchQuery, normalized_text: str) -> Optional[Tuple[int, int, float]]:
    """`(phrase, distinct content tokens matched, proportion matched)`, or None when the text
    does not qualify at all.

    QUALIFYING is the OR: the exact normalized phrase, or at least one non-stopword whole-token
    match. A query that is nothing but stopwords therefore qualifies on the phrase alone — which
    is the honest answer for "what did we do about it": it finds that sentence and nothing else.
    """
    if not normalized_text:
        return None
    phrase = 1 if query.phrase_hit(normalized_text) else 0
    matched = 0
    if query.content_tokens:
        present = set(search_tokens(normalized_text))
        matched = sum(1 for t in query.content_tokens if t in present)
    if not phrase and not matched:
        return None
    proportion = (matched / len(query.content_tokens)) if query.content_tokens else float(phrase)
    return (phrase, matched, proportion)


# The one receipt state that puts one of our own messages back in front of the model. Same
# literal the channel stream uses (`channel_stream.RECEIPT_FINALIZED`), restated rather than
# imported: this module must not pull the stream builder in behind a Slack tool.
RECEIPT_FINALIZED = "finalized"

# Below this much remaining budget the receipt read is not attempted at all. A busy timeout of a
# couple of milliseconds would fail against any momentarily-locked database, and the outcome is
# identical either way — own messages excluded — so the honest move is to skip the call rather
# than spend the last of the clock manufacturing a lock error (codex verify P2: "floor it").
_RECEIPT_READ_MIN_SECONDS = 0.05

# A payload the scan could not READ — as against one the normalizer declines because it is not a
# message at all (a join notice, a reply-count notification). The difference is the whole of codex
# review #2: a declined subtype is a decision and leaves coverage complete; an unreadable payload
# is a hole, and a hole that reports `complete: true` certifies a horizon over records nobody saw.
_UNREADABLE: Any = object()

# The stop reasons, STRONGEST FIRST. A scan can trip more than one on its way out — a root's
# replies fetch can fail while the clock is already spent — and the block reports ONE, so the
# reason that ended the most work wins. `deadline` outranks everything because it stops every
# root at once; a per-thread ceiling is the weakest because the scan carries on to other roots.
_STOP_PRECEDENCE: Tuple[str, ...] = (
    "deadline", "history_page_ceiling", "reply_page_ceiling", "thread_page_ceiling",
)


def _stop_rank(reason: Optional[str]) -> int:
    """A named API failure sorts just under the budget stops: it ended one fetch, not the scan."""
    if reason is None:
        return 99
    try:
        return _STOP_PRECEDENCE.index(reason)
    except ValueError:
        return len(_STOP_PRECEDENCE)


def _is_own_chrome(text: str, raw: Any) -> bool:
    """Is one of OUR OWN messages transient UI chrome (a placeholder, a status card, a footer)?

    FAIL-CLOSED, unlike the stream's fail-open classifier, and the direction is deliberate: this
    answer only ever decides whether a GRANDFATHERED pre-epoch own message may be searched, so a
    classification we could not make excludes it rather than attributing chrome to ourselves as
    something we once said.
    """
    try:
        return bool(is_self_chrome_message(text, raw if isinstance(raw, dict) else {"text": text}))
    except Exception:  # noqa: BLE001
        return True


@dataclass
class _ScanCandidate:
    """One message that QUALIFIED: enough to rank it, and the raw payload the shaper needs.

    `team_ids` is filled by the delivery check, which runs BEFORE the candidate competes for a
    place in the pool, so the provenance the enrollment rule needs is already in hand and the
    shaper never has to re-derive it.
    """

    ts: str
    thread_root_ts: Optional[str]
    text: str
    raw: Dict[str, Any]
    score: Tuple[int, int, float]
    ts_key: Tuple[int, int]
    is_self: bool = False
    team_ids: List[str] = field(default_factory=list)

    @property
    def rank(self) -> Tuple[int, int, float, Tuple[int, int]]:
        """Exact phrase, then distinct content tokens, then proportion, then RECENCY — the tie
        break, because two equally-good matches are not equally useful and the newer one is the
        one a person means."""
        return (self.score[0], self.score[1], self.score[2], self.ts_key)


class _ChannelScan:
    """The in-channel scan's bookkeeping — PURE: no Slack, no database, no clock.

    It exists so the parts that decide what was seen, what qualified and what the coverage block
    says can be exercised without a network, and so the executor's own body stays about IO.

    TWO RETENTION POOLS, and the second is not an optimization. A message of OUR OWN is only
    searchable with receipt evidence (§S3), and that evidence is one batched database read that
    cannot run until the walk knows which of our own messages it saw. Holding unverified self
    candidates in the SAME pool would let an in-flight half-written reply — which, being the
    answer to this very question, tends to match it well — evict a real result and then be
    dropped, silently returning fewer matches than the channel holds.

    NOTHING ENTERS A POOL UNTIL IT IS DELIVERABLE (codex review #3). Qualification and retention
    are two calls — `admit` then `retain` — with the caller's `_delivery_allowed` check between
    them, because the pool holds only `limit` entries: a filter applied AFTER the competition
    lets a high-ranking undeliverable candidate evict a deliverable one and return nothing where
    the channel had a perfectly good answer.

    THE TRIGGER FENCE IS A CONSTRUCTOR ARGUMENT, ALREADY PARSED (codex review #6). It used to
    accept an unparseable ts and quietly fall back to "no bound", which is the one failure mode
    a fence must not have; the executor now refuses the call instead, and this class cannot be
    built without a bound at all.
    """

    def __init__(self, *, query: SearchQuery, trigger_key: Tuple[int, int], limit: int):
        self.query = query
        self.limit = max(1, int(limit))
        self.trigger_key: Tuple[int, int] = trigger_key
        self.seen: Set[str] = set()
        self.scanned = 0
        self.threads_scanned = 0
        self.roots: Dict[str, Tuple[int, int]] = {}
        self.self_ts: List[str] = []
        # ts -> (has content, is chrome). Two booleans instead of the payload: the receipt rule
        # needs exactly these about an own message, and holding the raw dicts for every one of
        # our own posts in a fifty-page span to answer them would keep the whole walk in memory.
        self.self_facts: Dict[str, Tuple[bool, bool]] = {}
        self.unfetched_roots = False
        self.stopped_reason: Optional[str] = None
        self._seq = 0
        self._pool: List[Tuple[Tuple[int, int, float, Tuple[int, int]], int, _ScanCandidate]] = []
        self._self_pool: List[
            Tuple[Tuple[int, int, float, Tuple[int, int]], int, _ScanCandidate]] = []

    # -- stops ---------------------------------------------------------------------------

    def stop(self, reason: str) -> None:
        if _stop_rank(reason) < _stop_rank(self.stopped_reason):
            self.stopped_reason = reason

    # -- the walk ------------------------------------------------------------------------

    def admit(self, normalized: Any, raw: Dict[str, Any]) -> Optional[_ScanCandidate]:
        """One fetched message: fence it, record what it says about threads, then score it.

        Returns the candidate that QUALIFIED and still needs the delivery check, or None. It does
        NOT retain — see the class note: a candidate the delivery rule will refuse must never
        have competed for a place.

        DEDUPED BY ts BEFORE ANYTHING IS COUNTED. A reply-bearing root arrives twice — once in
        the history page, once as `conversations.replies`' first entry — and a broadcast arrives
        twice as well, so a count that did not dedupe would report a coverage number larger than
        the number of messages that exist.
        """
        ts = normalized.ts
        if ts in self.seen:
            return None
        self.seen.add(ts)
        # STRICTLY OLDER THAN THE TRIGGER. Without this the message we are answering is itself
        # usually the newest and strongest match for its own question, and anything posted after
        # it was never part of the room state this turn is reasoning about.
        if parse_ts(ts) >= self.trigger_key:
            return None
        self._note_thread(normalized)
        candidate = self._candidate(normalized, raw)
        if normalized.sender_type == "self":
            # NOT counted as scanned yet: an own message without receipt evidence was never
            # searchable, so counting it here would inflate the coverage block with messages the
            # scan is about to refuse to read.
            self.self_ts.append(ts)
            self.self_facts[ts] = (bool(normalized.text.strip() or normalized.files),
                                   _is_own_chrome(normalized.text, raw))
            if candidate is not None:
                candidate.is_self = True
            return candidate
        self.scanned += 1
        return candidate

    def retain(self, candidate: _ScanCandidate) -> None:
        """Keep a candidate the delivery rule has already cleared."""
        self._push(self._self_pool if candidate.is_self else self._pool, candidate)

    def _note_thread(self, normalized: Any) -> None:
        """Roots worth fetching replies for, keyed by NEWEST KNOWN ACTIVITY.

        Two shapes, and the second is the one root-only matching misses: a reply-bearing root
        discovered in the span, and a root NAMED by a `thread_broadcast` — which is how a thread
        older than the whole history span becomes reachable at all.
        """
        root = normalized.thread_root_ts or normalized.ts
        activity: Optional[str] = None
        if normalized.thread_root_ts and normalized.thread_root_ts != normalized.ts:
            # A broadcast: its own ts is the newest activity we can prove for that thread.
            activity = normalized.ts
        elif normalized.reply_count or normalized.latest_reply:
            activity = normalized.latest_reply or normalized.ts
        if activity is None:
            return
        try:
            key = parse_ts(activity)
        except TimestampError:
            return
        current = self.roots.get(root)
        if current is None or key > current:
            self.roots[root] = key

    def _candidate(self, normalized: Any, raw: Dict[str, Any]) -> Optional[_ScanCandidate]:
        score = score_search_text(self.query, normalize_search_text(normalized.text))
        if score is None:
            return None
        try:
            ts_key = parse_ts(normalized.ts)
        except TimestampError:
            return None
        return _ScanCandidate(ts=normalized.ts, thread_root_ts=normalized.thread_root_ts,
                              text=normalized.text, raw=raw, score=score, ts_key=ts_key)

    def _push(self, pool: List[Any], candidate: _ScanCandidate) -> None:
        """Retain only the best `limit`, as a min-heap. Fifty history pages can be ten thousand
        messages; buffering the transcript to sort it at the end is the thing this refuses."""
        self._seq += 1
        entry = (candidate.rank, self._seq, candidate)
        if len(pool) < self.limit:
            heapq.heappush(pool, entry)
        else:
            heapq.heappushpop(pool, entry)

    # -- results -------------------------------------------------------------------------

    def searchable_self(self, *, receipts: Dict[str, str],
                        epoch: Optional[str]) -> Set[str]:
        """WHICH of our own messages this scan may read — the stream builder's rule verbatim.

        A FINALIZED receipt is the only thing that admits an own message after the receipts
        epoch. `in_flight` is a reply still being written; `chrome` was never a reply; and a
        post-epoch message with NO row is evidence that a registration was lost, not evidence
        that we said something. Only messages that PREDATE the epoch are grandfathered, because
        those have no row by construction rather than by failure — and a missing or unusable
        epoch cannot establish that, so it admits nothing.
        """
        out: Set[str] = set()
        for ts in self.self_ts:
            state = receipts.get(ts)
            if state is not None:
                if state == RECEIPT_FINALIZED:
                    out.add(ts)
                continue
            has_content, chrome = self.self_facts.get(ts, (False, True))
            if not has_content or not epoch or chrome:
                continue
            try:
                if parse_ts(ts) < parse_ts(epoch):
                    out.add(ts)
            except TimestampError:
                continue
        return out

    def admit_self(self, searchable_ts: Set[str]) -> None:
        """Fold in the own-messages the receipt read proved searchable, and count them.

        These already passed the delivery check when they were admitted; the receipt read is the
        second gate, not a re-run of the first.
        """
        self.scanned += len(searchable_ts & set(self.self_ts))
        for _rank, _seq, candidate in list(self._self_pool):
            if candidate.ts in searchable_ts:
                self._push(self._pool, candidate)
        self._self_pool.clear()

    def ranked(self) -> List[_ScanCandidate]:
        return [c for _rank, _seq, c in sorted(self._pool, key=lambda e: (e[0], e[1]),
                                               reverse=True)][:self.limit]

    @property
    def complete(self) -> bool:
        return self.stopped_reason is None and not self.unfetched_roots


def coverage_note(*, complete: bool, messages: int, threads: int,
                  stopped_reason: Optional[str]) -> str:
    """The human sentence in the coverage block. PARTIAL COVERAGE MUST NEVER READ AS "no
    matches" — a model that is told only `count: 0` has no way to tell an empty channel from a
    scan that ran out of budget two thousand messages short of the answer.

    AND A COMPLETE SCAN CLAIMS COMPLETENESS OVER WHAT IT MAY RETURN, not over the room (codex
    review #5, S6 addendum). Two kinds of message are deliberately withheld even from a scan
    that read every page: our own posts without receipt evidence, and messages stamped with
    another workspace — an external participant in a Slack Connect channel writes in plain view
    and is still invisible here. "Everything this channel holds" would be a false sentence for
    exactly the rooms where it matters most.
    """
    head = (f"Searched {messages:,} message{'' if messages == 1 else 's'} across "
            f"{threads:,} thread{'' if threads == 1 else 's'} in this channel, newest activity "
            f"first")
    if complete:
        return (f"{head}; nothing older or deeper was left unread. A term that is not in the "
                "results was not in the messages this search may return — which is not the same "
                "as never said in this room, since our own unfinalized posts and anyone writing "
                "from another workspace are never searchable here.")
    tails = {
        "deadline": "stopped at the time budget",
        "history_page_ceiling": "stopped at the history-page budget",
        "reply_page_ceiling": "stopped at the reply-page budget",
        "thread_page_ceiling": "stopped at a single thread's page limit",
        "history_data_invalid": "skipped messages it could not read",
        "reply_data_invalid": "skipped thread replies it could not read",
        # NOT a Slack failure, and the generic tail said it was. This is the local receipt
        # ledger — the evidence that says which of our OWN messages may be replayed — and
        # naming the wrong system in the one sentence a person reads is how an operator spends
        # an afternoon looking at the Slack API dashboard.
        "receipt_read_failed": ("could not read the receipt evidence for its own messages, so "
                                "none of them were searched"),
    }
    tail = tails.get(stopped_reason or "", f"stopped after a Slack API failure ({stopped_reason})")
    return (f"{head}; {tail}. Some of this channel was NOT read, so absence here is not evidence "
            "of absence in the channel.")


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
    """`search_slack` — ONE tool name over TWO backends, split by surface (`search_backend_for`).

    DM (true IM) → `assistant.search.context`, Phase B, unchanged: Slack's own index, the whole
    workspace, an action_token, and the privacy model documented below.

    CHANNEL / MPIM → `_execute_in_channel_scan`: a bot-token keyword scan of THAT channel's
    history plus the replies inside its threads. It exists because the index could not serve a
    channel turn — an unmentioned turn carries no action_token and so could not search at all,
    and the index returned zero exact-recall hits for content seeded twelve hours earlier
    (measured 2026-08-03). Being a direct read, it sees only what the channel holds, and it
    reports how much of that it managed to read (§S7) rather than letting a spent budget look
    like an empty channel.

    ITS ONE STRUCTURAL LIMITATION, stated wherever the tool is described (§S10): a recent reply
    beneath a thread root OLDER than the scanned history span is undiscoverable unless a
    `thread_broadcast` names that root, because Slack offers no current-channel index of
    replies. The reserved `SearchBackend.SERVICE_INDEX` seam may close that; v1 does not.

    The DM path's own bounds follow, and none of them apply to the channel scan — which is
    bounded instead by the canonical channel-read authorization, the current channel, and the
    receipt discipline that keeps our own in-flight messages out of a search of ourselves.

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
        log_debug: Any
        log_warning: Any
        log_error: Any
        resolve_usernames: Any
        # The canonical delivery-audience decision lives in SlackHistoryToolMixin; the two mixins
        # share ONE SlackBot instance, so it is reachable as self.… here.
        _delivery_allowed: Any
        _bot_team_id: Any
        # The channel-read authorization façade and its two refusal payloads, also from the
        # history mixin: the in-channel scan authorizes the current channel through the SAME
        # canonical path every other channel read uses.
        _authorize_channel_read: Any
        _access_denied: Any
        _delivery_redirect: Any

    # The DM surface's wording, frozen: DM tool bytes are a contract, and W3 is a channel
    # feature. The channel variant below adds the one sentence `thread_ts` earns.
    _SEARCH_DESCRIPTION = (
        "Search Slack messages the bot is allowed to see (workspace-wide or the "
        "current channel). Use for finding older discussions, decisions, or context "
        "outside the current thread; prefer fetch_thread_messages/fetch_channel_history "
        "for things in the current conversation. Each result carries its source channel "
        "id; pass that to resolve_channel_name to show the channel's name."
    )
    # The CHANNEL surface's wording, and it describes a DIFFERENT TOOL BEHIND THE SAME NAME:
    # a bot-token keyword scan of this channel, not the workspace index. It says so plainly,
    # because a description that promised workspace reach would have the model asking for
    # something the executor cannot do and reading an honest "not here" as "nowhere".
    _SEARCH_CHANNEL_DESCRIPTION = (
        "Keyword search of THIS channel — the messages it still holds, including the replies "
        "inside its threads. Use distinctive words you expect to appear in the message itself "
        "(a name, a figure, a project word), not a paraphrase of the question: this matches "
        "words, not meaning, and a message qualifies on any one of them. It reads this channel "
        "only, never the wider workspace, and never anything posted after the message you are "
        "answering. Each result carries its ts and, when it sits in a thread, its thread_ts — a "
        "result with no thread_ts is not a thread, so call fetch_thread_messages if you need to "
        "read or answer in one. The result also reports how much of the channel it managed to "
        "read: partial coverage is not the same as no matches, and a recent reply under a "
        "thread root older than that span is not reachable this way."
    )

    def _search_tool_schema(self, description: str) -> Dict[str, Any]:
        return {
            "type": "function",
            "name": "search_slack",
            "description": description,
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

    def get_search_tool_schema(self) -> Dict[str, Any]:
        """The DM surface's schema — byte-identical to pre-W3."""
        return self._search_tool_schema(self._SEARCH_DESCRIPTION)

    def get_search_tool_channel_schema(self, thread_config: Optional[dict] = None
                                       ) -> Dict[str, Any]:
        """The CHANNEL surface's schema. STATIC (``thread_config`` is accepted and ignored so the
        registry can call it like any channel schema) — it changes when the bot changes, never
        when a message does, which is what keeps it inside the cached prefix.

        NO `scope`. The channel backend reads the current channel and nothing else, so a
        workspace/channel choice would be a parameter with one legal value and one lie; the
        executor refuses an explicitly non-channel scope rather than silently narrowing it.
        The DM schema keeps its own bytes, because a DM keeps its own backend and result shape.
        """
        return {
            "type": "function",
            "name": "search_slack",
            "description": self._SEARCH_CHANNEL_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": ("The distinctive words to look for — names, figures, "
                                        "project words. Not a question."),
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

    @staticmethod
    def _root_ts(value: Any) -> Optional[str]:
        """A Slack ts, or None. A non-`str` is never one: `parse_ts` stringifies whatever it is
        handed, so a JSON number would arrive looking like a perfectly good timestamp."""
        if not isinstance(value, str) or not value:
            return None
        try:
            parse_ts(value)
        except TimestampError:
            return None
        return value

    def _hit_thread_root(self, m: Dict[str, Any]) -> Optional[str]:
        """A hit's THREAD ROOT, or None. Two sources and no third.

        Slack's search hit carries a reply's own ts, and a reply's ts is not a root — posting
        to it would target a message that has no thread. So the root comes either from the
        payload's own `thread_ts`, or from the `thread_ts` query parameter Slack stamps on a
        thread reply's permalink (`…/archives/<C>/p<ts>?thread_ts=<root>&cid=<C>`). Neither
        present means this hit enrolls NOTHING, which is the normal case for a top-level match.

        The permalink route additionally requires the LINK TO AGREE WITH THE HIT about which
        channel it is — both the `/archives/<C>` segment and the `cid` parameter — because
        parsing `thread_ts` alone would let a link pointing into another conversation authorize
        a root under this channel's id. Absent counts as disagreement, not as a pass.

        Parsed with `urllib.parse`, never a regex: a permalink is a URL, and a pattern that
        looks for `thread_ts=` in a string will find it in a fragment, in an encoded parameter,
        or in the middle of somebody else's value.
        """
        raw = self._root_ts(m.get("thread_ts"))
        if raw:
            return raw
        permalink = m.get("permalink")
        if not isinstance(permalink, str) or not permalink:
            return None
        source = self._parse_search_source_id(m)
        if not source:
            return None
        try:
            parts = urlsplit(permalink)
            params = parse_qs(parts.query)
        except ValueError:
            return None
        segments = [s for s in parts.path.split("/") if s]
        if "archives" not in segments:
            return None
        named = segments.index("archives") + 1
        if named >= len(segments) or segments[named] != source:
            return None
        # Exactly one value each: a repeated parameter is a link that contradicts itself, and
        # there is no reading of it that is safer than refusing.
        cid = params.get("cid") or []
        roots = params.get("thread_ts") or []
        if len(cid) != 1 or cid[0] != source or len(roots) != 1:
            return None
        return self._root_ts(roots[0])

    async def execute_search_tool(self, ctx, args: Dict[str, Any]) -> Dict[str, Any]:
        """`search_slack` — validate the query, pick the backend for this SURFACE, run it.

        The split is the whole design (§S1). A DM keeps `assistant.search.context` byte for
        byte: same token gate, same request, same result keys, same refusals. A channel or MPIM
        runs the bot-token in-channel scan, which needs no action_token and reaches this channel
        only. Never raises; no content on refusal or error.
        """
        query = (args.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": "bad_arguments", "message": "query is required."}
        if search_backend_for(ctx) is SearchBackend.IN_CHANNEL_SCAN:
            return await self._execute_in_channel_scan(ctx, args, query)
        return await self._execute_assistant_context(ctx, args, query)

    async def _execute_assistant_context(self, ctx, args: Dict[str, Any],
                                         query: str) -> Dict[str, Any]:
        """THE DM PATH, UNCHANGED. Every line below is pre-split behaviour — the action-token
        gate, the channel-type bound, the `in:<#…>` scope operator, the delivery filter, the
        result bytes. DM tool bytes are a contract; nothing here may drift with the channel
        backend."""
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
        # The team ids ride along with each kept hit (§2g): `results` entries are shaped down to
        # {channel, ts, author, thread_ts, text, permalink} and DISCARD the provenance, and the
        # enrollment rule below — which runs after the whole result is built — cannot check a
        # workspace it can no longer see.
        kept: List[Tuple[Dict[str, Any], Optional[str], List[str]]] = []
        for m in messages:
            if not isinstance(m, dict):
                # A malformed hit (a string/None in `messages`) would raise on .get() below.
                continue
            source = self._parse_search_source_id(m)
            team_ids = self._parse_search_source_team_ids(m)
            # Channel scope was constrained at the API with in:<#…>; keep the post-filter as
            # belt-and-braces so a stray cross-channel hit can't ride a channel-scoped query.
            if scope == "channel" and ctx.channel_id and source != ctx.channel_id:
                continue
            if filtering_active:
                if len(team_ids) > 1:
                    # The hit contradicts itself about its workspace → can't classify → drop.
                    continue
                deliverable, _dreason = await self._delivery_allowed(
                    source, ctx, source_team_id=(team_ids[0] if team_ids else None))
                if not deliverable:
                    continue
            kept.append((m, source, team_ids))
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
            aid for (m, _s, _t) in kept if (aid := (m.get("author_user_id") or m.get("user")))))
        name_map = {}
        if author_ids and resolver:
            try:
                name_map = await resolver(author_ids, api_client)
            except Exception:
                name_map = {}
        results = []
        for (m, source, _team_ids) in kept:
            author_id = m.get("author_user_id") or m.get("user")
            author = name_map.get(author_id, author_id or m.get("username"))
            entry = {
                "channel": source,
                "ts": m.get("message_ts") or m.get("ts"),
                "author": author,
                "text": m.get("content") or m.get("text", ""),
                "permalink": m.get("permalink"),
            }
            if not is_dm_surface:
                # CHANNEL SURFACE ONLY. `thread_ts` exists to make a root actionable, and
                # post_to_thread is a channel tool — a DM has no stream, no allowlist and nothing
                # to do with the field. DM result bytes are a contract, so the DM shape is the
                # pre-W3 shape exactly, and the DM schema (which never mentions the field) says
                # the same thing.
                entry["thread_ts"] = self._hit_thread_root(m)
            results.append(entry)

        payload = {"ok": True, "query": query, "scope": scope,
                   "count": len(results), "results": results}
        self._enroll_search_roots(ctx, payload, kept)
        return payload

    # ------------------------------------------------ the in-channel scan (§S3–S7, §S10)

    async def _execute_in_channel_scan(self, ctx, args: Dict[str, Any],
                                       query: str) -> Dict[str, Any]:
        """Keyword-scan the CURRENT channel on the BOT token.

        NO SLACK FAILURE ESCAPES: a refused page, a spent budget and a malformed page all become
        an honest coverage block on a successful result, never an exception and never a bare
        empty answer. A defect in this code still propagates to the registry, which turns it into
        `execution_error` — the same contract the DM path has, and the reason a broken tool
        degrades one answer instead of killing the turn.

        A BOUNDED CURRENT-CHANNEL RETRIEVER, not an index. It walks `conversations.history`
        newest-first and fetches `conversations.replies` for every reply-bearing root it finds —
        root-only matching would reproduce the defect this replaces, because the fact usually
        lives in a reply — under ONE absolute deadline and three page budgets, and it says in
        the result how much it managed to read.

        THE LIMITATION IS STRUCTURAL AND STATED EVERYWHERE (§S10): a recent reply beneath a root
        older than the history span is undiscoverable unless a `thread_broadcast` exposes it.
        Slack offers no current-channel reply index; the later service-account backend may close
        that gap, and v1 does not promise it.

        NOTHING FETCHED IS PERSISTED (CLAUDE.md rule 4): pages are scored and dropped, the
        database is READ for receipt evidence and never written.
        """
        channel_id = getattr(ctx, "channel_id", None)
        if not channel_id:
            return {"ok": False, "error": "bad_arguments",
                    "message": "There is no channel here to search."}
        # The channel schema has no `scope`, so an explicit one is a call built against the DM
        # schema (or an invented argument). Refused rather than silently narrowed: answering a
        # workspace request with one channel's contents, unremarked, is how "not here" gets read
        # as "nowhere".
        requested_scope = args.get("scope")
        if requested_scope is not None:
            wanted = str(requested_scope).strip().lower()
            if wanted and wanted != "channel":
                return {"ok": False, "error": "bad_arguments",
                        "message": ("search_slack reads the current channel only here; it has "
                                    "no workspace scope. Drop the scope argument.")}

        # THE FENCE IS A PRECONDITION, NOT A BEST EFFORT (codex review #6). Every result this
        # backend returns must predate the message being answered, and an absent or unparseable
        # trigger cannot establish that for a single message. The old code turned it into "no
        # bound" and scanned the present; refusing before any fetch is the only reading of §S3
        # that a malformed or replayed context cannot widen.
        try:
            trigger_key = parse_ts(getattr(ctx, "trigger_ts", None))
        except TimestampError:
            self.log_info("search_tool: refused — the context carries no usable trigger_ts, so "
                          "the scan has no point in time to search before")
            return {
                "ok": False,
                "error": "search_unavailable",
                "hint": ("Search needs the message it is answering. Use fetch_channel_history "
                         "or fetch_thread_messages instead."),
            }

        # The CANONICAL authorization path, once, before anything is fetched. Bot membership is
        # implied operationally — Slack delivered this turn from here — but a direct caller or a
        # hand-built context has proved nothing, and this is the gate every other channel read
        # goes through.
        verdict, reason = await self._authorize_channel_read(channel_id, ctx)
        if verdict == "DENY":
            return self._access_denied(channel_id, reason, "search_slack")
        if verdict == "REDIRECT":
            return self._delivery_redirect(channel_id, reason, "search_slack")

        limit = self._clamp_search_limit(args.get("limit"))
        scan = _ChannelScan(query=build_search_query(query), trigger_key=trigger_key, limit=limit)
        # ONE ABSOLUTE DEADLINE FOR EVERY AWAIT THIS CALL MAKES — history pages, reply pages, the
        # receipt read, username resolution and permalink enrichment (§S5, codex review #1).
        # Budgets built with their own `total_seconds` would be separate windows; unbounded
        # post-fetch awaits would be worse still, because the outer 20s tool timeout would fire
        # and the honest coverage block would never be returned at all.
        deadline_at = time.monotonic() + float(config.search_fetch_total_seconds)
        history_ceiling = max(1, int(config.search_history_page_ceiling))
        reply_ceiling = max(1, int(config.search_reply_page_ceiling))
        history_budget = FetchBudget(deadline_at=deadline_at, page_ceiling=history_ceiling)
        reply_budget = FetchBudget(deadline_at=deadline_at, page_ceiling=reply_ceiling)

        await self._scan_history(ctx, channel_id, scan, history_budget, history_ceiling)
        await self._scan_replies(ctx, channel_id, scan, reply_budget, reply_ceiling)
        await self._admit_self_messages(ctx, channel_id, scan, deadline_at)

        results, kept = await self._shape_scan_results(ctx, channel_id, scan, deadline_at)
        coverage = {
            "complete": scan.complete,
            "messages_scanned": scan.scanned,
            "threads_scanned": scan.threads_scanned,
            "history_pages": history_budget.pages_used,
            "reply_pages": reply_budget.pages_used,
            "stopped_reason": scan.stopped_reason,
            "note": coverage_note(complete=scan.complete, messages=scan.scanned,
                                  threads=scan.threads_scanned,
                                  stopped_reason=scan.stopped_reason),
        }
        self.log_info(
            f"search_tool: in-channel scan of {channel_id} read {scan.scanned} messages / "
            f"{scan.threads_scanned} threads ({history_budget.pages_used} history pages, "
            f"{reply_budget.pages_used} reply pages), complete={scan.complete}, "
            f"stopped={scan.stopped_reason or '-'}, kept {len(results)}")
        payload = {"ok": True, "query": query, "scope": "channel",
                   "count": len(results), "results": results, "coverage": coverage}
        self._enroll_search_roots(ctx, payload, kept)
        return payload

    def _normalize_scanned(self, raw: Any, channel_id: str, origin: str) -> Optional[Any]:
        """One fetched payload → the SAME normalized message the channel stream renders from.

        THREE OUTCOMES, and the third is the one codex review #2 is about:
          * a NormalizedMessage — a message;
          * `None` — DECLINED. A join notice or a reply-count notification is not a message, and
            skipping it leaves coverage honestly complete;
          * `_UNREADABLE` — a payload we could not read. Skipping it is still the right thing to
            do with one bad message, but it is a HOLE, and the caller marks the scan incomplete
            rather than certifying a channel it did not entirely see.

        A SLACK TIMESTAMP IS A STRING (codex review #4). `secondary_ts` stringifies whatever it
        is handed, so a JSON number in `thread_ts` would arrive looking like a perfectly good
        root — and that root flows straight into `stage_discovered_root`, widening where this
        turn may post. The shared normalizer's wider behaviour is not ours to change; refusing
        the malformed type at THIS boundary is, and a security-sensitive consumer does not have
        to accept an external type it never expects. Same rule for `ts`, which is the identity
        the model quotes back at us.

        This is the artifact-aware path §S6 names: on a channel surface `_text_with_supplementary`
        routes to exactly this normalizer, so a message read by search and the same message read
        by the history tool cannot disagree about what it said.
        """
        if not isinstance(raw, dict):
            return _UNREADABLE
        for key in ("ts", "thread_ts"):
            value = raw.get(key)
            if key in raw and value is not None and (not isinstance(value, str) or not value):
                return _UNREADABLE
        try:
            return normalize_slack_message(self, raw, channel_id=channel_id, origin=origin)
        except TimestampError:
            return _UNREADABLE
        except Exception:  # noqa: BLE001 — one unreadable payload never ends a scan
            return _UNREADABLE

    async def _score_page(self, ctx, channel_id: str, scan: "_ChannelScan",
                          page: Sequence[Any], origin: str, phase: str) -> None:
        """One page: normalize, qualify, CHECK DELIVERY, retain — in that order.

        The delivery check runs here, per qualifying candidate, and not in the shaper (codex
        review #3). The pool holds `limit` entries: filtering after the competition lets a
        high-ranking undeliverable candidate evict a deliverable one and hand back an empty
        result for a channel that had the answer. It is also what §S6 says on its face — every
        candidate goes through the canonical rule — and for a candidate from the current channel
        with our own team stamp the rule answers from memory, with no API call.
        """
        for raw in page:
            normalized = self._normalize_scanned(raw, channel_id, origin)
            if normalized is _UNREADABLE:
                # A hole, not a decision: coverage may not claim completeness over it.
                scan.stop(f"{phase}_data_invalid")
                self.log_warning(
                    f"search_tool: skipped an unreadable message in {channel_id} ({phase})")
                continue
            if normalized is None:
                continue          # declined subtype — not a message, and not a gap either
            candidate = scan.admit(normalized, raw)
            if candidate is None:
                continue
            candidate.team_ids = self._scan_team_ids(raw)
            if len(candidate.team_ids) > 1:
                # It contradicts itself about its workspace → it cannot be classified → dropped.
                continue
            deliverable, _reason = await self._delivery_allowed(
                channel_id, ctx,
                source_team_id=(candidate.team_ids[0] if candidate.team_ids else None))
            if deliverable:
                scan.retain(candidate)

    def _scan_stop_reason(self, error: BaseException, *, budget: FetchBudget,
                          ceiling: int, phase: str) -> str:
        """Which bound actually ended this fetch — asked of the BUDGET, not of the message text.

        `HistoryFetchError` says the same sentence for a spent clock and a spent page count, and
        a coverage block that guessed between them would be exactly the dishonest reporting the
        block exists to replace.
        """
        if budget.remaining_seconds() <= 0:
            return "deadline"
        if budget.pages_used >= ceiling:
            return f"{phase}_page_ceiling"
        code = (error.code if isinstance(error, HistoryPageError) and error.code
                else slack_error_code(error))
        if not code:
            # A retry ladder that ran out wraps the Slack error as its CAUSE ("failed after N
            # attempt(s)"), and `ratelimited` is the reason worth naming most.
            cause = getattr(error, "__cause__", None)
            code = slack_error_code(cause) if cause is not None else ""
        return f"{phase}_error:{code}" if code else f"{phase}_error"

    async def _scan_history(self, ctx, channel_id: str, scan: "_ChannelScan",
                            budget: FetchBudget, ceiling: int) -> None:
        """The newest-first history walk, bounded ABOVE by the trigger.

        `latest=trigger_ts, inclusive=False` asks Slack for strictly-older messages; the scan's
        own fence repeats it, because a bound asked of the API is a request and a bound applied
        locally is the rule. Both exist, and the executor has already refused any context that
        could not supply one.
        """
        client = getattr(getattr(self, "app", None), "client", None)
        method = getattr(client, "conversations_history", None)
        if method is None:
            scan.stop("history_error")
            return
        trigger_ts = getattr(ctx, "trigger_ts", None)
        try:
            async for page in iter_pages(
                    method, channel_id=channel_id,
                    latest=(str(trigger_ts) if trigger_ts else None), inclusive=False,
                    limit=config.search_history_page_size, budget=budget,
                    label="search history"):
                await self._score_page(ctx, channel_id, scan, page, ORIGIN_HISTORY, "history")
        except HistoryFetchError as e:
            scan.stop(self._scan_stop_reason(e, budget=budget, ceiling=ceiling, phase="history"))
            self.log_warning(f"search_tool: history walk of {channel_id} stopped: {e}")
        except SlackApiError as e:
            # A refusal the pager did not already wrap. Everything NOT caught here — a defect in
            # our own scoring, gating or shaping — propagates to the registry as `execution_error`
            # rather than being laundered into a coverage reason (codex review #2): a bug must not
            # be reported to the model as "Slack was slow".
            scan.stop(f"history_error:{slack_error_code(e) or 'unknown'}")
            self.log_warning(f"search_tool: history walk of {channel_id} failed: {e}")

    async def _scan_replies(self, ctx, channel_id: str, scan: "_ChannelScan",
                            budget: FetchBudget, ceiling: int) -> None:
        """Replies for EVERY reply-bearing root the span discovered, newest activity first.

        Scheduling is by newest KNOWN activity (`latest_reply`, then a broadcast's ts, then the
        root's own ts) because the reply budget is global: when it runs out, what it will have
        spent itself on is the threads most likely to hold what was just asked about.

        Slack pages `conversations.replies` CHRONOLOGICALLY. Only the scheduling and the final
        ranking are newest-first; nothing here claims the messages inside a thread arrive that
        way.
        """
        if not scan.roots:
            return
        client = getattr(getattr(self, "app", None), "client", None)
        method = getattr(client, "conversations_replies", None)
        if method is None:
            scan.unfetched_roots = True
            scan.stop("reply_error")
            return
        roots = [root for root, _key in sorted(scan.roots.items(), key=lambda kv: kv[1],
                                               reverse=True)]
        concurrency = max(1, int(config.search_reply_fetch_concurrency))
        index = 0
        while index < len(roots):
            # THE REASON IS RECORDED WHERE THE WORK STOPS. Breaking out on a spent budget without
            # naming it would report `complete: false` with `stopped_reason: null` — a coverage
            # block that admits it is partial and refuses to say why.
            if budget.remaining_seconds() <= 0:
                scan.stop("deadline")
                break
            if budget.pages_used >= ceiling:
                scan.stop("reply_page_ceiling")
                break
            chunk = roots[index:index + concurrency]
            await asyncio.gather(*(self._scan_one_thread(ctx, channel_id, root, scan, method,
                                                         budget=budget, ceiling=ceiling)
                                   for root in chunk))
            index += len(chunk)
            if scan.stopped_reason in ("deadline", "reply_page_ceiling"):
                break
        if index < len(roots):
            # Roots we never asked about at all: the coverage block must not call this complete.
            scan.unfetched_roots = True

    async def _scan_one_thread(self, ctx, channel_id: str, root_ts: str, scan: "_ChannelScan",
                               method: Any, *, budget: FetchBudget, ceiling: int) -> None:
        per_thread = max(1, int(config.search_reply_per_thread_page_ceiling))
        pages = 0
        try:
            async for page in iter_pages(
                    method, channel_id=channel_id, inclusive=True,
                    limit=config.search_history_page_size, budget=budget,
                    extra_params={"ts": str(root_ts)}, label="search replies"):
                pages += 1
                await self._score_page(ctx, channel_id, scan, page, ORIGIN_REPLIES, "reply")
                if pages >= per_thread:
                    # Reported even when the thread happened to END here: a coverage claim we
                    # cannot prove must fail toward "incomplete", never toward "that was all".
                    scan.stop("thread_page_ceiling")
                    break
        except HistoryFetchError as e:
            scan.stop(self._scan_stop_reason(e, budget=budget, ceiling=ceiling, phase="reply"))
            self.log_warning(f"search_tool: replies of {channel_id}/{root_ts} stopped: {e}")
        except SlackApiError as e:
            # Same rule as the history walk: Slack's refusals become coverage, our own defects
            # propagate.
            scan.stop(f"reply_error:{slack_error_code(e) or 'unknown'}")
            self.log_warning(f"search_tool: replies of {channel_id}/{root_ts} failed: {e}")
        if pages:
            scan.threads_scanned += 1

    async def _admit_self_messages(self, ctx, channel_id: str, scan: "_ChannelScan",
                                   deadline_at: float) -> None:
        """THE RECEIPT GATE, ONE BATCHED READ (§S3) — the stream builder's rule, applied here.

        A direct scan can see what the old index's lag hid: a partial post this very turn is
        still writing. So one of OUR OWN messages is searchable only with a `finalized` receipt,
        or by predating the receipts epoch and not being chrome. Evidence we cannot READ — no
        database on the context, no team identity, a failed query, OR A SPENT DEADLINE (§S5,
        codex review #1) — excludes every own message, which is the same direction the stream
        fails in. The read is bounded by what is left of the scan's own clock: this runs after
        the pagers, and an unbounded database await here is how the outer tool timeout fires
        before the coverage block is ever built.
        """
        if not scan.self_ts:
            return
        db = getattr(ctx, "db", None)
        team_id = self._bot_team_id()
        remaining = deadline_at - time.monotonic()
        if db is None or not team_id or remaining <= _RECEIPT_READ_MIN_SECONDS:
            spent = remaining <= _RECEIPT_READ_MIN_SECONDS
            self.log_warning(
                f"search_tool: no receipt evidence available "
                f"({'deadline spent' if spent else 'db/team missing'}) — excluding "
                f"{len(scan.self_ts)} own message(s) from the scan of {channel_id}")
            if spent:
                scan.stop("deadline")
            scan.admit_self(set())
            return
        try:
            # BOUNDED TWICE, AND THE INNER BOUND IS THE LOAD-BEARING ONE (codex verify P2).
            # `wait_for` cancels the coroutine at expiry and then waits for it to UNWIND, and
            # unwinding drains the SQLite work already queued — so a read blocked on a locked
            # database overshoots by the connection's whole busy timeout (5s by house default),
            # long after the scan promised an answer. Capping the busy timeout at what is left
            # of the budget puts the bound inside the connection, where a lock wait can actually
            # be cut short; `wait_for` stays as the outer guard for everything else.
            payload = await asyncio.wait_for(
                db.read_channel_sidecars_for_async(team_id, channel_id, scan.self_ts,
                                                   busy_timeout_ms=int(remaining * 1000)),
                timeout=remaining)
        except asyncio.TimeoutError:
            self.log_warning(
                f"search_tool: receipt read outlived the scan deadline for {channel_id} — "
                f"excluding {len(scan.self_ts)} own message(s)")
            scan.stop("deadline")
            scan.admit_self(set())
            return
        except Exception as e:  # noqa: BLE001
            # Evidence we could not read is a GAP, not a policy exclusion: the coverage block
            # must not go on to claim it read everything (the same rule as an unreadable page).
            self.log_warning(f"search_tool: receipt read failed for {channel_id}: {e}")
            scan.stop("receipt_read_failed")
            scan.admit_self(set())
            return
        rows = (payload or {}).get("receipts") or []
        states = {str(r.get("message_ts")): str(r.get("state"))
                  for r in rows if isinstance(r, dict) and r.get("message_ts")}
        scan.admit_self(scan.searchable_self(
            receipts=states, epoch=(payload or {}).get("receipt_feature_epoch_ts")))

    def _scan_team_ids(self, raw: Dict[str, Any]) -> List[str]:
        """The workspace ids one scanned message carries, or OURS when it carries none.

        The fallback is what the fetch itself proves: this payload came back from the current
        channel, on our own bot token, through an authorization gate — an unstamped local
        message (our own posts carry no `team`) is not evidence of a foreign workspace. A
        message that stamps a DIFFERENT team still says so, and a message that stamps two
        contradicts itself and is dropped by the caller.
        """
        ids = self._parse_search_source_team_ids(raw)
        if ids:
            return ids
        bot_team = self._bot_team_id()
        return [bot_team] if bot_team else []

    async def _shape_scan_results(self, ctx, channel_id: str, scan: "_ChannelScan",
                                  deadline_at: float
                                  ) -> Tuple[List[Dict[str, Any]],
                                             List[Tuple[Dict[str, Any], Optional[str],
                                                        List[str]]]]:
        """Ranked candidates → the result entries.

        THE DELIVERY FILTER IS NOT HERE ANY MORE (codex review #3). Every candidate passed
        `_delivery_allowed` at qualification time, before it could compete for a place in the
        bounded pool, so what arrives here is already deliverable and the shaper is about
        rendering. Names and permalinks are resolved only for these final results — both cost API
        budget, and neither should ever be spent on a candidate the model will not see.
        """
        kept: List[_ScanCandidate] = scan.ranked()

        api_client = getattr(getattr(self, "app", None), "client", None)
        resolver = getattr(self, "resolve_usernames", None)
        # Ordered dedup in result order, as the assistant path: a hash-ordered set would let the
        # resolver's remote budget cover a different subset across cold starts.
        author_ids = list(dict.fromkeys(
            c.raw.get("user") for c in kept
            if c.raw.get("user") and not c.raw.get("bot_id")))
        name_map: Dict[str, str] = {}
        remaining = deadline_at - time.monotonic()
        if author_ids and resolver and remaining > 0:
            try:
                # BOUNDED BY THE SAME DEADLINE (§S5, codex review #1). `resolve_usernames` may
                # make sequential users.info calls, and an unbounded one here would let the outer
                # tool timeout kill a scan that had already done all of its real work. Out of
                # time means RAW IDS — a less readable answer, never a lost one.
                name_map = await asyncio.wait_for(resolver(author_ids, api_client),
                                                  timeout=remaining)
            except asyncio.TimeoutError:
                self.log_warning(
                    f"search_tool: username resolution outlived the scan deadline for "
                    f"{channel_id} — returning raw author ids")
                name_map = {}
            except Exception:  # noqa: BLE001
                name_map = {}
        permalinks = await self._scan_permalinks(channel_id, kept, deadline_at)

        results: List[Dict[str, Any]] = []
        enrolled: List[Tuple[Dict[str, Any], Optional[str], List[str]]] = []
        for candidate in kept:
            raw = candidate.raw
            user_id = raw.get("user")
            author: Any = (user_id or raw.get("username")
                           or ("bot" if raw.get("bot_id") else "unknown"))
            if user_id and not raw.get("bot_id"):
                author = name_map.get(str(user_id), author)
            results.append({
                # ALWAYS the trusted current channel, never an argument and never a field off
                # the payload: this backend cannot return anything from anywhere else.
                "channel": channel_id,
                "ts": candidate.ts,
                "author": author,
                "text": candidate.text,
                # Present even when enrichment failed or the deadline ran out. A missing link
                # never invalidates a match — the ts and the channel are the finding.
                "permalink": permalinks.get(candidate.ts),
                # Straight from the normalized message, which is what makes a result actionable:
                # post_to_thread needs a ROOT, and a reply's own ts is not one.
                "thread_ts": candidate.thread_root_ts,
            })
            enrolled.append((raw, channel_id, candidate.team_ids))
        return results, enrolled

    async def _scan_permalinks(self, channel_id: str, candidates: Sequence["_ScanCandidate"],
                               deadline_at: float) -> Dict[str, str]:
        """`chat.getPermalink` for the FINAL results only, under the scan's own deadline.

        `conversations.history` supplies no permalinks and the old result shape carries one, so
        the link is minted here — bounded by the result limit, the same absolute deadline, and
        the same concurrency as the reply fan-out. Any failure is simply a missing link.
        """
        out: Dict[str, str] = {}
        if not candidates:
            return out
        client = getattr(getattr(self, "app", None), "client", None)
        method = getattr(client, "chat_getPermalink", None)
        if method is None:
            return out
        semaphore = asyncio.Semaphore(max(1, int(config.search_reply_fetch_concurrency)))

        async def _one(ts: str) -> None:
            async with semaphore:
                remaining = deadline_at - time.monotonic()
                if remaining <= 0:
                    return
                try:
                    resp = await asyncio.wait_for(
                        method(channel=channel_id, message_ts=ts), timeout=remaining)
                except Exception:  # noqa: BLE001 — a link is an enrichment, never a result
                    return
                link = resp.get("permalink") if resp is not None else None
                if isinstance(link, str) and link:
                    out[ts] = link

        await asyncio.gather(*(_one(c.ts) for c in candidates))
        return out

    def _enroll_search_roots(self, ctx, payload: Dict[str, Any],
                             kept: List[Tuple[Dict[str, Any], Optional[str], List[str]]]) -> None:
        """§2g. THE LAST STEP BEFORE THE RESULT IS RETURNED — every root this search CLAIMS,
        walked over the FINISHED payload beside the provenance retained for each hit.

        Last, and deliberately not inside the kept loop, because authorization follows what
        Slack actually returned: a shaping error, an aborted name resolution or a validation
        refusal above this line means the model never saw the hit, and a hit it never saw must
        not widen where it may post.

        IT STAGES; IT DOES NOT ENROL. Whether this result is the one the model receives is not
        knowable here — this executor is still running when that is decided, and reading a clock
        to guess would be answering a question the caller owns. `stage_discovered_root` records
        the claim and the FIELD it came from; `ToolRegistry` commits the subset that survives
        into the delivered, clipped payload, at the moment it selects this result.

        A ctx with no staging area stages nothing and is NOT an error — that is a DM, a
        background agent, or a hand-built context outside the registry.
        """
        try:
            bot_team = self._bot_team_id()
        except Exception:  # noqa: BLE001 — an unknowable own team claims nothing, below
            bot_team = None
        for entry, (_m, _source, team_ids) in zip(payload.get("results") or (), kept):
            # Absent, contradictory, or somebody else's workspace. Stricter than the delivery
            # gate on purpose: that gate lets an UNATTRIBUTED hit ride the current-channel
            # exemption, and "deliverable" is a lower bar than "somewhere this turn may post".
            if not bot_team or len(team_ids) != 1 or team_ids[0] != bot_team:
                continue
            if not entry.get("thread_ts"):
                continue
            stage_discovered_root(ctx, channel_id=entry.get("channel"),
                                  root_ts=entry.get("thread_ts"), source="search_slack",
                                  field="thread_ts")
