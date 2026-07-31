"""The turn path refuses rather than guessing (codex P2 review, findings 7e / 8 / 9).

Each of these was a place where something we could not read was quietly stepped over, leaving a
turn to answer from a window it had no right to call complete:

* F7e a live event whose timestamp the shared comparator cannot parse was logged at WARNING and
  skipped, while its readiness ticket completed SUCCESSFULLY — telling every turn in that window
  the index was caught up on an event nobody could place.
* F8  a history page with a malformed record was processed for the records it COULD read and then
  advanced coverage to the page's oldest ts, minting a horizon that claims the skipped record was
  indexed. Nothing revisits a horizon.
* F9  the per-turn fetch deadline was checked before the Slack call and never applied TO it, so a
  request could return long after the turn's budget was spent and still be accepted.
"""
import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from base_client import HistoryFetchError
from slack_client import admission_watermark
from slack_client.admission_watermark import FAILED, AdmissionWatermark
from slack_client.event_handlers import activity_index
from slack_client.event_handlers.activity_index import ChannelCoverageBootstrap
from slack_client.event_handlers.registration import _admit
from slack_client.history_fetch import FetchBudget, fetch_page

TEAM = "T1"
CH = "C1"
_NOW = time.time()


class _Client:
    self_team_id = TEAM
    bot_user_id = "UBOT"
    bot_id = "BBOT"
    app_id = "A123"

    def __init__(self, db=None):
        self.db = db

    def is_own_message(self, message):
        return False


# ------------------------------------------------------- F8: a malformed page has no horizon


def _cfg(**overrides):
    values = {"coverage_bootstrap_days": 90, "history_page_size": 200,
              "history_page_ceiling": 50, "fetch_retry_attempts": 3,
              "coverage_sweep_concurrency": 2}
    values.update(overrides)
    return SimpleNamespace(**values)


def _boot(db, cfg=None):
    boot = ChannelCoverageBootstrap(_Client(db), db=db, cfg=cfg or _cfg())
    boot._semaphore = asyncio.Semaphore(2)
    return boot


def _db():
    db = MagicMock()
    db.seed_channel_coverage_async = AsyncMock()
    db.record_thread_activity_async = AsyncMock()
    db.advance_channel_coverage_async = AsyncMock(return_value=True)
    db.get_channel_coverage_async = AsyncMock(return_value={
        "bootstrap_status": "running", "sweep_token": "tok",
        "coverage_start_ts": f"{_NOW:.6f}"})
    return db


# --------------------------------------------------- F7e: the live listener admission path


@pytest.fixture
def wm(monkeypatch):
    instance = AdmissionWatermark()
    monkeypatch.setattr(admission_watermark, "watermark", instance)
    yield instance
    instance.reset()


def test_admit_fails_the_ticket_when_slack_hands_us_an_unreadable_ts(wm):
    ticket = _admit(_Client(), {"type": "message", "channel": CH, "channel_type": "channel",
                                "user": "U1", "ts": "wat", "event_ts": "wat"})
    assert ticket is not None and ticket.state == FAILED
    assert wm.is_degraded(CH) is True


def test_admit_completes_normally_for_a_readable_ts(wm):
    ts = f"{_NOW:.6f}"
    ticket = _admit(_Client(), {"type": "message", "channel": CH, "channel_type": "channel",
                                "user": "U1", "ts": ts, "event_ts": ts})
    assert ticket is not None and ticket.state == "pending"
    assert wm.current(CH) == ts
    assert wm.is_degraded(CH) is False


def test_a_dm_with_a_broken_ts_takes_no_ticket_and_degrades_nothing(wm):
    ticket = _admit(_Client(), {"type": "message", "channel": "D123", "channel_type": "im",
                                "user": "U1", "ts": "wat", "event_ts": "wat"})
    assert ticket is None
    assert wm.is_degraded("D123") is False


@pytest.mark.asyncio
async def test_the_index_feed_fails_the_ticket_on_an_unparseable_ts(wm):
    """The feed used to complete_ok here. A ticket that reports success for an event it could not
    normalize is worse than a failure: the turn proceeds."""
    client = _Client(db=MagicMock())
    ticket = wm.issue(CH)
    await activity_index.feed_thread_activity_index(
        client, {"type": "message", "channel": CH, "channel_type": "channel", "user": "U1",
                 "ts": "1.2.3", "thread_ts": "9.0", "text": "hi"}, ticket=ticket)

    assert ticket.state == FAILED
    with pytest.raises(HistoryFetchError):
        await wm.drain(CH, ticket.seq, timeout=0.01)


@pytest.mark.asyncio
async def test_the_index_feed_still_completes_ok_for_an_unsupported_subtype(wm):
    """A subtype the normalizer legitimately declines is NOT a malformed record — it must not
    take a channel out of service."""
    client = _Client(db=_db())
    ticket = wm.issue(CH)
    ts = f"{_NOW:.6f}"
    await activity_index.feed_thread_activity_index(
        client, {"type": "message", "subtype": "channel_join", "channel": CH,
                 "channel_type": "channel", "user": "U1", "ts": ts}, ticket=ticket)

    assert ticket.state == "ok"
    assert wm.is_degraded(CH) is False
    await wm.drain(CH, ticket.seq, timeout=0.01)


@pytest.mark.asyncio
async def test_a_clean_page_reports_complete_and_its_oldest_ts():
    boot = _boot(_db())
    oldest, complete = await boot._process_page(TEAM, CH, [
        {"ts": f"{_NOW - 10:.6f}", "reply_count": 2, "latest_reply": f"{_NOW - 5:.6f}"},
        {"ts": f"{_NOW - 20:.6f}"},
    ])
    assert complete is True and oldest == f"{_NOW - 20:.6f}"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [
    {"reply_count": 3},                      # no ts at all
    {"ts": "not-a-ts", "reply_count": 3},    # a ts nothing can compare
    "not even a dict",
])
async def test_a_page_with_a_record_we_cannot_place_is_incomplete(bad):
    db = _db()
    boot = _boot(db)
    oldest, complete = await boot._process_page(TEAM, CH, [
        {"ts": f"{_NOW - 10:.6f}"}, bad,
    ])
    assert complete is False, "a skipped record must make the page incomplete"
    assert oldest == f"{_NOW - 10:.6f}", "the readable records still report their oldest ts"
    # An unreadable root is not recorded either — the index would hold a ts no window predicate
    # could ever compare.
    for call in db.record_thread_activity_async.await_args_list:
        assert call.args[2] != "not-a-ts"


@pytest.mark.asyncio
@pytest.mark.parametrize("secondary", [{"thread_ts": "whenever"},
                                       {"latest_reply": "whenever"}])
async def test_a_record_whose_SECONDARY_ts_is_unreadable_writes_nothing(secondary):
    """r2-3: only the parent ts used to be parsed, so a malformed thread_ts or latest_reply was
    persisted verbatim — a row discovered under a ts nothing can compare, sitting below a horizon
    that says its page was fully recorded."""
    db = _db()
    boot = _boot(db)
    oldest, complete = await boot._process_page(TEAM, CH, [
        {"ts": f"{_NOW - 10:.6f}", "reply_count": 2, **secondary},
    ])
    assert complete is False, "an unplaceable secondary ts must make the page incomplete"
    assert oldest == f"{_NOW - 10:.6f}"
    db.record_thread_activity_async.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("secondary", [{"thread_ts": ""}, {"thread_ts": 0},
                                       {"latest_reply": ""}, {"latest_reply": 0},
                                       {"latest_reply": False}])
async def test_a_secondary_ts_that_is_PRESENT_and_falsey_writes_nothing(secondary):
    """r3-7. Only truthy values were validated, so `""`, `0` and `false` — values Slack PUT in the
    field — read as "the field is absent". A parent with an empty thread_ts was recorded as its own
    root, and coverage then advanced over a thread it never indexed properly. Present and falsey is
    malformed, which is the same verdict as unreadable."""
    db = _db()
    boot = _boot(db)
    oldest, complete = await boot._process_page(TEAM, CH, [
        {"ts": f"{_NOW - 10:.6f}", "reply_count": 2, **secondary},
    ])
    assert complete is False, "a present-but-empty secondary ts must make the page incomplete"
    assert oldest == f"{_NOW - 10:.6f}"
    db.record_thread_activity_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_absent_secondary_ts_is_not_malformed():
    """Absent is absent: the field Slack simply did not send is the ordinary case, and a `null`
    means the same thing. Only a value that is THERE and unusable fails the page."""
    db = _db()
    boot = _boot(db)
    oldest, complete = await boot._process_page(TEAM, CH, [
        {"ts": f"{_NOW - 10:.6f}", "reply_count": 2, "thread_ts": None,
         "latest_reply": None},
    ])
    assert complete is True
    assert oldest == f"{_NOW - 10:.6f}"
    # No latest_reply to place the replies with, so the row is recorded dirty rather than skipped.
    assert db.record_thread_activity_async.await_args.kwargs["mark_dirty"] is True


@pytest.mark.asyncio
async def test_a_readable_secondary_ts_still_records_normally():
    """The guard above must not have frozen the ordinary parent hint."""
    db = _db()
    boot = _boot(db)
    await boot._process_page(TEAM, CH, [
        {"ts": f"{_NOW - 10:.6f}", "thread_ts": f"{_NOW - 30:.6f}",
         "latest_reply": f"{_NOW - 5:.6f}", "reply_count": 2},
    ])
    assert db.record_thread_activity_async.await_args.args[2] == f"{_NOW - 30:.6f}"


# --------------------------------------------- r2-1: the adapter, not the caller, judges a page


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    ["messages", "as", "a", "list"],                        # not a page at all
    {"ok": True, "messages": "not a list"},                 # became an empty terminal page
    {"ok": True},                                           # ditto, by omission
    {"ok": True, "messages": ["bad", {"ts": "1.0"}]},        # became a successful PARTIAL page
])
async def test_a_page_that_is_not_a_page_is_refused_by_the_pager(payload):
    """The lossy adapter was the real hole: it filtered what it could not read and returned the
    rest, so a protocol break arrived at the caller wearing the shape of a good page."""
    from slack_client.history_fetch import HistoryPageInvalid, page_messages

    method = AsyncMock(return_value=payload)
    with pytest.raises(HistoryPageInvalid):
        await page_messages(method, channel_id=CH, attempts=1)


@pytest.mark.asyncio
async def test_the_sweep_requires_a_placeable_ts_on_every_record():
    """The bootstrap's contract needs one: its horizon claims every record at or after a ts is
    indexed, so an element it cannot place would certify a false one."""
    from slack_client.history_fetch import HistoryPageInvalid, fetch_page

    method = AsyncMock(return_value={"ok": True, "messages": [{"ts": "nope"}]})
    with pytest.raises(HistoryPageInvalid):
        await fetch_page(method, {"channel": CH}, attempts=1, require_ts=True)
    # The turn path leaves it to the normalizer, which raises a StreamTimestampError instead — a
    # notice that says retrying will not help rather than borrowing Slack's "try again".
    assert (await fetch_page(method, {"channel": CH}, attempts=1)).messages == [{"ts": "nope"}]


@pytest.mark.asyncio
async def test_a_slack_response_object_is_read_through_its_data():
    """A slack_sdk response answers `get` but supports neither keys() nor iteration, so the old
    `dict(response)` raised TypeError on every real page."""
    from slack_sdk.web.async_slack_response import AsyncSlackResponse

    from slack_client.history_fetch import fetch_page

    resp = AsyncSlackResponse(
        client=None, http_verb="GET", api_url="https://slack.com/api/conversations.history",
        req_args={}, data={"ok": True, "messages": [{"ts": "1.0"}], "has_more": False},
        headers={}, status_code=200)
    page = await fetch_page(AsyncMock(return_value=resp), {"channel": CH}, attempts=1)
    assert [m["ts"] for m in page.messages] == ["1.0"]


@pytest.mark.asyncio
@pytest.mark.parametrize("age_days, expected_status", [(2, "limited"), (0, "running")])
async def test_the_depth_wall_is_judged_by_the_shared_comparator(monkeypatch, age_days,
                                                                 expected_status):
    """r2-12: page ordering and the depth wall went through a second, float comparator. Both use
    parse_ts now — which also means a tuple is being compared to a tuple, so a regression here
    raises rather than quietly rounding a boundary."""
    db = _db()
    boot = _boot(db, cfg=_cfg(coverage_bootstrap_days=1))
    oldest = f"{_NOW - age_days * 86400:.6f}"
    monkeypatch.setattr(
        boot, "_fetch_page",
        AsyncMock(return_value=(SimpleNamespace(
            messages=[{"ts": oldest}], next_cursor="c1", has_more=True,
            is_limited=False), None)))

    await boot._sweep_pass(TEAM, CH, "tok")

    args = db.advance_channel_coverage_async.await_args.args
    assert (args[3], args[4]) == (oldest, expected_status)


@pytest.mark.asyncio
async def test_an_unreadable_page_leaves_coverage_exactly_where_it_was(monkeypatch):
    """Same verdict as a malformed record found during processing: no horizon over records we
    never read."""
    db = _db()
    boot = _boot(db)
    monkeypatch.setattr(boot, "_web_method",
                        lambda name: AsyncMock(return_value={"ok": True, "messages": ["bad"]}))

    outcome = await boot._sweep_pass(TEAM, CH, "tok")

    assert outcome == "abandon"
    db.advance_channel_coverage_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_coverage_is_not_advanced_past_a_record_it_skipped(monkeypatch):
    db = _db()
    boot = _boot(db)
    monkeypatch.setattr(
        boot, "_fetch_page",
        AsyncMock(return_value=(SimpleNamespace(
            messages=[{"ts": f"{_NOW - 10:.6f}"}, {"ts": "garbage"}],
            next_cursor="", has_more=True, is_limited=False), None)))

    outcome = await boot._sweep_pass(TEAM, CH, "tok")

    assert outcome == "abandon"
    db.advance_channel_coverage_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_clean_page_still_advances_coverage(monkeypatch):
    """The guard above must not have frozen the ordinary path."""
    db = _db()
    boot = _boot(db)
    monkeypatch.setattr(
        boot, "_fetch_page",
        AsyncMock(return_value=(SimpleNamespace(
            messages=[{"ts": f"{_NOW - 10:.6f}"}],
            next_cursor="", has_more=False, is_limited=False), None)))

    outcome = await boot._sweep_pass(TEAM, CH, "tok")

    assert outcome == "terminal"
    db.advance_channel_coverage_async.assert_awaited()
    assert db.advance_channel_coverage_async.await_args.args[3] == f"{_NOW - 10:.6f}"


# ------------------------------------------------- F9: the deadline bounds the request itself


class _Clock:
    """A hand-cranked monotonic clock, so a budget test never actually waits."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


@pytest.mark.asyncio
async def test_a_request_that_outlives_the_budget_is_not_accepted():
    clock = _Clock()
    # A real (tiny) remaining budget: this is the one assertion that must go through wait_for's
    # own timer, because the point is that the AWAIT is bounded.
    budget = FetchBudget(total_seconds=0.05, page_ceiling=10, clock=clock)
    started = asyncio.Event()

    async def method(**params):
        started.set()
        await asyncio.sleep(3600)  # a wedged Slack call
        return {"ok": True, "messages": []}

    task = asyncio.ensure_future(fetch_page(method, {"channel": CH}, budget=budget,
                                            attempts=1, label="history"))
    await started.wait()
    with pytest.raises(HistoryFetchError, match="outlived the fetch budget"):
        await asyncio.wait_for(task, timeout=10)


@pytest.mark.asyncio
async def test_a_page_that_returns_after_the_deadline_is_refused():
    """wait_for's timer and the budget's clock are two measurements; only the budget is the
    turn's promise, so a page that comes back late is still refused."""
    clock = _Clock()
    budget = FetchBudget(total_seconds=5.0, page_ceiling=10, clock=clock)

    async def method(**params):
        clock.now += 99.0  # the request "took" far longer than the budget allowed
        return {"ok": True, "messages": [{"ts": "1.0"}]}

    with pytest.raises(HistoryFetchError, match="budget"):
        await fetch_page(method, {"channel": CH}, budget=budget, attempts=1)


@pytest.mark.asyncio
async def test_an_in_budget_page_is_returned_unchanged():
    clock = _Clock()
    budget = FetchBudget(total_seconds=30.0, page_ceiling=10, clock=clock)

    async def method(**params):
        clock.now += 0.25
        return {"ok": True, "messages": [{"ts": "1.0"}], "has_more": False}

    page = await fetch_page(method, {"channel": CH}, budget=budget, attempts=1)
    assert [m["ts"] for m in page.messages] == ["1.0"]
    assert budget.pages_used == 1


@pytest.mark.asyncio
async def test_an_unbudgeted_fetch_is_not_wrapped_in_a_timeout():
    """The coverage sweep passes no budget: it parks with its own claim held, and wrapping it in
    a per-attempt timeout would cancel a legitimate long page."""
    async def method(**params):
        await asyncio.sleep(0)
        return {"ok": True, "messages": [], "has_more": False}

    page = await fetch_page(method, {"channel": CH}, attempts=1)
    assert page.messages == []
