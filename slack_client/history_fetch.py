"""Shared Slack history/replies pager (spec §4).

One implementation of "walk a conversations.history or conversations.replies cursor", used by
both the turn-path stream builder and the coverage bootstrap. The two differ in exactly two
ways, and both are parameters rather than forks:

  * `inclusive` — the turn path fetches inclusive=true and applies its own window predicate,
    so a floor message is never lost to Slack's boundary semantics; the bootstrap resumes
    backward from a ts it has already processed and must EXCLUDE it or it re-walks a page
    forever.
  * anomaly handling — on the turn path a page that claims `has_more` and hands us no cursor
    means we cannot prove we saw the whole window, so the turn fails closed. The bootstrap
    leaves its coverage row `running` and retries later, so it reads the PageResult itself.

The per-turn budget is global, not per-fetch: one deadline and one page counter shared by every
concurrent replies fetch, so a channel with two hundred live threads cannot turn one turn into
two hundred sequential retry ladders.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import (Any, AsyncIterator, Awaitable, Callable, Dict, List, Mapping,
                    Optional)

from slack_sdk.errors import SlackApiError

from message_processor.client_contract import HistoryFetchError
from config import config
from logger import setup_logger
from slack_client.normalizer import TimestampError, parse_ts

logger = setup_logger(name="slack_bot.HistoryFetch")


class HistoryPageError(HistoryFetchError):
    """A page Slack REFUSED, carrying the error code so a caller that classifies refusals
    (the coverage bootstrap: channel gone vs. missing scope vs. transient) still can."""

    def __init__(self, message: str, *, code: str = "",
                 retry_after: Optional[float] = None):
        super().__init__(message)
        self.code = code
        self.retry_after = retry_after


class HistoryPageInvalid(HistoryFetchError):
    """A page Slack ANSWERED that is not the shape a page is.

    Distinct from a refusal: `ok` was true and there is no error code to classify, but what came
    back cannot be read as messages. This used to be absorbed — a list response became an empty
    terminal page and `["bad", <message>]` became a successful partial one — which is the worst
    possible handling of a protocol break: a turn renders an incomplete window and the coverage
    bootstrap certifies a horizon over records it never saw.
    """


@dataclass(frozen=True)
class PageResult:
    messages: List[Dict[str, Any]]
    next_cursor: str
    has_more: bool
    is_limited: bool
    raw: Dict[str, Any]

    @property
    def claims_more(self) -> bool:
        return bool(self.has_more or self.next_cursor)


# The "no ceiling given" sentinel. Module-level so callers can name it and tests can assert
# identity. A plain object() is deliberate: any typed default would be a value some caller could
# pass by accident, and `None` already means something else here (see the three-state table).
_USE_CONFIG_CEILING: Any = object()


class FetchBudget:
    """One per COMPONENT: a wall-clock deadline and an atomic total-page counter.

    The counter is charged under a lock because concurrent replies fetches share it; without
    that, N tasks each read the same count and the ceiling bounds one fetch instead of the turn.

    THE PAGE CEILING HAS THREE STATES, and the sentinel is what makes them expressible:

      * `page_ceiling` OMITTED  -> read `config.history_page_ceiling` (every landed caller);
      * `page_ceiling=None`     -> UNBOUNDED: pages are still counted and the deadline is still
                                   checked, so the component is bounded by wall clock alone;
      * `page_ceiling=<int>`    -> exactly that ceiling.

    `None` cannot mean "unbounded" and "read config" at once, which is why omission is a distinct
    sentinel rather than `None` (SHALLOW_STREAM_RESPEC §4.3).

    THE DEADLINE HAS TWO FORMS, and only one of them can be SHARED:

      * `total_seconds` — a window that starts when THIS budget is constructed. Three budgets
        built with `total_seconds=60` therefore have three different deadlines and a turn could
        spend 180 seconds;
      * `deadline_at` — an ABSOLUTE `time.monotonic()` instant, computed once by the caller and
        handed to every component. When it is set `total_seconds` is IGNORED.
    """

    def __init__(self, *, total_seconds: Optional[float] = None,
                 deadline_at: Optional[float] = None,
                 page_ceiling: Any = _USE_CONFIG_CEILING,
                 clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._started = clock()
        self._deadline_at = None if deadline_at is None else float(deadline_at)
        self._total_seconds = float(
            config.fetch_retry_total_seconds if total_seconds is None else total_seconds)
        if page_ceiling is _USE_CONFIG_CEILING:
            self._ceiling: Optional[int] = int(config.history_page_ceiling)
        elif page_ceiling is None:
            self._ceiling = None
        else:
            self._ceiling = int(page_ceiling)
        self._pages = 0
        self._lock = asyncio.Lock()

    @property
    def pages_used(self) -> int:
        return self._pages

    @property
    def deadline_at(self) -> Optional[float]:
        """The ABSOLUTE deadline this budget was built against, or None.

        Public because the one-deadline rule is only enforceable at the boundary that can read
        it: an origin fetch carries this forward on its result so the pin that consumes it can
        prove both components were built against the same clock.
        """
        return self._deadline_at

    def remaining_seconds(self) -> float:
        if self._deadline_at is not None:
            return self._deadline_at - self._clock()
        return self._total_seconds - (self._clock() - self._started)

    def check_deadline(self) -> None:
        if self.remaining_seconds() <= 0:
            budget = (f"{self._total_seconds:g}s" if self._deadline_at is None
                      else "shared-deadline")
            raise HistoryFetchError(
                f"channel history fetch exceeded its {budget} budget")

    async def charge_page(self) -> int:
        async with self._lock:
            # An unbounded budget still COUNTS — `pages_used` is reported per component — it
            # simply has no ceiling to refuse against.
            if self._ceiling is not None and self._pages >= self._ceiling:
                raise HistoryFetchError(
                    f"channel history fetch hit the {self._ceiling}-page ceiling")
            self._pages += 1
            return self._pages


def retry_after_seconds(error: BaseException) -> Optional[float]:
    """Slack's Retry-After, when the error carries one."""
    headers = getattr(getattr(error, "response", None), "headers", None)
    if not headers:
        return None
    try:
        for key, value in headers.items():
            if str(key).lower() == "retry-after":
                return max(0.0, float(value))
    except (AttributeError, TypeError, ValueError):
        return None
    return None


def slack_error_code(error: BaseException) -> str:
    response = getattr(error, "response", None)
    try:
        return str(response.get("error") or "") if response is not None else ""
    except (AttributeError, TypeError):
        return ""


def _page_data(resp: Any, label: str) -> Dict[str, Any]:
    """One page's payload as a plain dict.

    A slack_sdk response is NOT a mapping: it answers `get`/`[]` but carries its payload on
    `.data` and supports neither `keys()` nor iteration, so `dict(response)` raises TypeError on
    the real client — every production page went through that line. Read `.data` first and treat
    the mapping protocol as the fallback for plain dicts and test doubles.
    """
    payload = getattr(resp, "data", resp)
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, Mapping):
        return dict(payload)
    raise HistoryPageInvalid(f"{label} returned {type(payload).__name__}, not a page")


def _page_result(resp: Any, *, label: str = "history", require_ts: bool = False) -> PageResult:
    """The response → PageResult adapter, and the only place a page's SHAPE is judged.

    Every branch here raises rather than salvaging. Filtering the unreadable parts out and
    returning the rest is what made a malformed response indistinguishable from a real page:
    `messages` as a string became an empty terminal page (a certified horizon over an unknown
    number of records), and one bad element became a page reported as whole.

    `require_ts` is the coverage bootstrap's half: its contract is "everything at or after this ts
    is in the index", so an element it cannot place is a false horizon. The turn path leaves the
    timestamp to the normalizer instead, which raises StreamTimestampError — a distinct, honest
    notice that says retrying will not clear it, rather than borrowing Slack's "try again".
    """
    data = _page_data(resp, label)
    if "messages" not in data:
        raise HistoryPageInvalid(f"{label} page carries no messages field at all")
    raw_messages = data["messages"]
    if not isinstance(raw_messages, list):
        raise HistoryPageInvalid(
            f"{label} page carries {type(raw_messages).__name__} where its messages should be")
    for index, entry in enumerate(raw_messages):
        if not isinstance(entry, dict):
            raise HistoryPageInvalid(
                f"{label} page element {index} is a {type(entry).__name__}, not a message")
        if require_ts:
            try:
                parse_ts(entry.get("ts"))
            except TimestampError as e:
                raise HistoryPageInvalid(
                    f"{label} page element {index} carries an unusable timestamp: {e}") from e
    metadata = data.get("response_metadata")
    cursor = metadata.get("next_cursor") or "" if isinstance(metadata, dict) else ""
    return PageResult(messages=list(raw_messages), next_cursor=str(cursor),
                      has_more=bool(data.get("has_more")),
                      is_limited=bool(data.get("is_limited")), raw=dict(data))


async def fetch_page(method: Callable[..., Awaitable[Any]], params: Dict[str, Any], *,
                     budget: Optional[FetchBudget] = None,
                     attempts: Optional[int] = None,
                     sleeper: Optional[Callable[[float], Awaitable[None]]] = None,
                     require_ts: bool = False,
                     label: str = "history") -> PageResult:
    """One page, with retries. Raises HistoryFetchError when the page cannot be had.

    With a `budget`, its remaining wall clock is the per-attempt timeout on the Slack request
    itself and is re-checked after the response arrives — the deadline bounds the fetch, not
    merely the decision to start one.

    `sleeper` lets the bootstrap park with its sweep claim held and its heartbeat bumped
    instead of a bare asyncio.sleep.
    """
    tries = max(1, int(config.fetch_retry_attempts if attempts is None else attempts))
    sleep = sleeper or asyncio.sleep
    if budget is not None:
        budget.check_deadline()
        await budget.charge_page()
    last: Optional[BaseException] = None
    for attempt in range(tries):
        try:
            if budget is None:
                resp = await method(**params)
            else:
                # The remaining budget IS the per-attempt timeout. Checking the deadline and then
                # awaiting an unbounded request bounds only the moment before the call: a Slack
                # request that hangs for a minute used to finish well past the turn's deadline
                # and still be accepted into the pinned stream.
                remaining = budget.remaining_seconds()
                if remaining <= 0:
                    budget.check_deadline()
                resp = await asyncio.wait_for(method(**params), timeout=remaining)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError as e:
            # Out of budget, not a transient refusal: retrying cannot help, so fail closed here.
            raise HistoryFetchError(
                f"{label} outlived the fetch budget mid-request") from e
        except SlackApiError as e:
            last = e
            delay = retry_after_seconds(e)
            code = slack_error_code(e)
            if code and code != "ratelimited" and delay is None:
                raise HistoryPageError(f"{label} refused: {code}", code=code) from e
        except Exception as e:  # noqa: BLE001
            last = e
            delay = None
        else:
            # A page that came back after the deadline is not accepted either: wait_for's timer
            # and the budget's clock are two different measurements, and only this one is the
            # turn's promise.
            if budget is not None:
                budget.check_deadline()
            if resp is None or (hasattr(resp, "get") and resp.get("ok") is False):
                code = str((resp or {}).get("error") or "" ) if resp is not None else "empty_response"
                raise HistoryPageError(f"{label} not ok: {code}", code=code)
            return _page_result(resp, label=label, require_ts=require_ts)
        if attempt >= tries - 1:
            break
        wait = delay if delay is not None else float(2 ** attempt)
        if budget is not None:
            remaining = budget.remaining_seconds()
            if remaining <= 0 or wait > remaining:
                raise HistoryFetchError(
                    f"{label} retry would outlive the fetch budget") from last
        await sleep(wait)
    raise HistoryFetchError(f"{label} failed after {tries} attempt(s): {last}") from last


async def iter_pages(method: Callable[..., Awaitable[Any]], *,
                     channel_id: str,
                     oldest: Optional[str] = None,
                     latest: Optional[str] = None,
                     inclusive: bool = True,
                     limit: Optional[int] = None,
                     budget: Optional[FetchBudget] = None,
                     attempts: Optional[int] = None,
                     sleeper: Optional[Callable[[float], Awaitable[None]]] = None,
                     extra_params: Optional[Dict[str, Any]] = None,
                     require_ts: bool = False,
                     label: str = "history") -> AsyncIterator[List[Dict[str, Any]]]:
    """Yield each page's raw message dicts AS IT ARRIVES, so a caller can stop early.

    THIS IS `page_messages`' LOOP, EXPOSED RATHER THAN REWRITTEN. Every check is retained and the
    cursor is followed identically; the only difference is that a page is handed to the caller
    instead of accumulated. It exists because the shallow window's history walk must stop as soon
    as it has proved enough roots — a function that buffers the whole window has already fetched
    and paid for the pages it should have stopped before.

    EVERY ANOMALY STILL RAISES rather than being yielded:
      * `has_more`/cursor present with no `next_cursor` -> HistoryFetchError
      * an empty page that still claims more            -> HistoryFetchError
      * a page whose SHAPE is not a page                -> HistoryPageInvalid (via fetch_page)

    A consumer that breaks out early simply stops asking; it can never suppress a raise, because
    the raise happens while PRODUCING the page it is about to receive.
    """
    params: Dict[str, Any] = {
        "channel": channel_id,
        "limit": max(1, int(config.history_page_size if limit is None else limit)),
        "inclusive": bool(inclusive),
    }
    if oldest is not None:
        params["oldest"] = str(oldest)
    if latest is not None:
        params["latest"] = str(latest)
    if extra_params:
        params.update(extra_params)

    cursor = ""
    while True:
        page_params = dict(params)
        if cursor:
            page_params["cursor"] = cursor
        page = await fetch_page(method, page_params, budget=budget, attempts=attempts,
                                sleeper=sleeper, require_ts=require_ts, label=label)

        # VALIDATED BEFORE THE YIELD, and that ordering is the whole guarantee. A consumer that
        # breaks immediately after receiving a page never resumes this generator, so anything
        # checked AFTER the yield is skipped entirely by generator cleanup — the early-stopping
        # walk would swallow exactly the malformed-continuation failures this function promises
        # to raise. Validating first means a page is proved sound before anyone can act on it.
        if page.claims_more:
            if not page.next_cursor:
                raise HistoryFetchError(
                    f"{label} for {channel_id} claimed more messages with no cursor to follow")
            if not page.messages:
                raise HistoryFetchError(
                    f"{label} for {channel_id} returned an empty page that still claimed more")

        yield page.messages
        if not page.claims_more:
            return
        cursor = page.next_cursor


async def page_messages(method: Callable[..., Awaitable[Any]], *,
                        channel_id: str,
                        oldest: Optional[str] = None,
                        latest: Optional[str] = None,
                        inclusive: bool = True,
                        limit: Optional[int] = None,
                        budget: Optional[FetchBudget] = None,
                        attempts: Optional[int] = None,
                        sleeper: Optional[Callable[[float], Awaitable[None]]] = None,
                        extra_params: Optional[Dict[str, Any]] = None,
                        require_ts: bool = False,
                        label: str = "history") -> List[Dict[str, Any]]:
    """Walk every page of one window and return the raw message dicts.

    UNCHANGED for every existing caller — it is now literally the drain of `iter_pages`, with the
    same signature, the same return and the same raises. The coverage sweep and the reply and
    origin fetches keep using it and are byte-identical in behaviour.

    Fails closed on an anomalous page: `has_more` with no cursor, an empty page that still claims
    more, or a page whose shape is not a page at all (`_page_result`) — each one means we cannot
    honestly say we saw the whole window.
    """
    collected: List[Dict[str, Any]] = []
    async for messages in iter_pages(
            method, channel_id=channel_id, oldest=oldest, latest=latest, inclusive=inclusive,
            limit=limit, budget=budget, attempts=attempts, sleeper=sleeper,
            extra_params=extra_params, require_ts=require_ts, label=label):
        collected.extend(messages)
    return collected
