"""Turn-path fetch, root discovery and the fail-closed gates around them.

The serializer is tested separately as a pure function; this file is about everything that has
to happen before it can be called — the window, the four root sources, the shared budget, and
the ways a turn refuses to answer rather than answer from a partial view.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest

from slack_sdk.errors import SlackApiError

from base_client import HistoryFetchError
from config import config
from message_processor import channel_stream
from message_processor.channel_stream import (
    StreamTimestampError,
    _build_actor_map,
    _discovery_roots,
    OriginFetchError,
    ChannelStreamError,
    OriginFetch,
    fetch_origin_thread,
    build_channel_stream,
)
from slack_client import actor_tail as actor_tail_module
from slack_client import admission_watermark
from slack_client.history_fetch import (FetchBudget, HistoryPageError,
                                        fetch_page, page_messages)
from slack_client.normalizer import (ORIGIN_HISTORY, ORIGIN_REPLIES, TimestampError,
                                     normalize_slack_message, parse_ts)

TEAM = "T1"
CH = "C0BKX77NU66"
FLOOR = "1700000000.000000"
H = "1700009999.000000"


class _Client:
    def __init__(self):
        self.self_team_id = TEAM
        self.bot_user_id = "U_BOT"
        self.bot_id = "B_BOT"
        self.app_id = "A_BOT"
        self.bot_handle = "chatgpt-dev"
        self.app = MagicMock()
        self.app.client.conversations_history = AsyncMock(return_value={"ok": True,
                                                                       "messages": []})
        self.app.client.conversations_replies = AsyncMock(return_value={"ok": True,
                                                                       "messages": []})
        self.resolved = []

    def is_own_message(self, msg):
        return bool(msg) and (msg.get("bot_id") == self.bot_id
                              or msg.get("user") == self.bot_user_id)

    def classify_sender(self, msg):
        if self.is_own_message(msg):
            return "self"
        if msg.get("bot_id") or msg.get("app_id"):
            return "other_bot"
        return "human"

    async def resolve_usernames(self, ids, api_client, max_remote_lookups=25):
        self.resolved.append(list(ids))
        return {uid: f"name-{uid}" for uid in ids if uid != "U_UNKNOWN"}


async def _stream(**kwargs):
    """`build_channel_stream` returns the BUILD RESULT now — a carrier holding the stream plus
    the three facts that postdate it. Most tests here are about the stream, so they take it
    through this; the ones about `reselected`, `anchor_advanced` or the page counts read the
    carrier directly."""
    result = await build_channel_stream(**kwargs)
    return result.stream


_ANCHOR = {"floor_ts": FLOOR, "selection_version": 1}


def _db(*, coverage=("complete", "genesis"), anchor=_ANCHOR, activity_roots=None,
        receipt_roots=(), **sidecars):
    """The three accessors the split-phase build reads, and nothing else.

    READ 1 stage 1 (anchor + inventory), READ 1 stage 2 (discovery), READ 2 (the render pin over
    exact ids). The window-form sidecar read is not on the turn path any more.
    """
    db = MagicMock()
    db.read_channel_window_anchor_async = AsyncMock(return_value={
        "anchor": anchor,
        "inventory": ({"inventory_start_ts": FLOOR, "bootstrap_status": coverage[0],
                       "reason": coverage[1]} if coverage else None),
    })
    db.read_channel_discovery_roots_async = AsyncMock(return_value={
        "activity_roots": dict(activity_roots or {}),
        "receipt_roots": tuple(receipt_roots),
    })
    payload = {
        "ids": [], "receipt_feature_epoch_ts": None, "receipts": [],
        "image_analyses": [], "document_extractions": [], "ambient_artifacts": [],
        "tool_usage": {}, "versions_hash": "h",
    }
    payload.update(sidecars)
    db.read_channel_sidecars_for_async = AsyncMock(return_value=payload)
    db.advance_channel_window_anchor_async = AsyncMock(return_value=True)
    db.clear_thread_dirty_async = AsyncMock(return_value=True)
    return db


def raw(ts, *, text="hi", user="U1", root=None, **extra):
    payload = {"ts": ts, "text": text, "user": user}
    if root:
        payload["thread_ts"] = root
    payload.update(extra)
    return payload


@pytest.fixture(autouse=True)
def _clean_singletons():
    admission_watermark.watermark.reset()
    actor_tail_module.actor_tail.reset()
    yield
    admission_watermark.watermark.reset()
    actor_tail_module.actor_tail.reset()


@pytest.fixture
def client():
    return _Client()


# ------------------------------------------------------------------ the pager

async def test_pager_walks_cursors_and_keeps_every_page():
    pages = [
        {"ok": True, "messages": [raw("100.1")], "has_more": True,
         "response_metadata": {"next_cursor": "c1"}},
        {"ok": True, "messages": [raw("100.2")], "has_more": False},
    ]
    method = AsyncMock(side_effect=pages)
    got = await page_messages(method, channel_id=CH, oldest=FLOOR, latest=H)
    assert [m["ts"] for m in got] == ["100.1", "100.2"]
    assert method.call_args_list[0].kwargs["inclusive"] is True
    assert method.call_args_list[1].kwargs["cursor"] == "c1"


async def test_pager_fails_closed_when_more_is_claimed_with_no_cursor():
    method = AsyncMock(return_value={"ok": True, "messages": [raw("100.1")], "has_more": True})
    with pytest.raises(HistoryFetchError, match="no cursor"):
        await page_messages(method, channel_id=CH)


async def test_pager_fails_closed_on_an_empty_page_that_claims_more():
    method = AsyncMock(return_value={"ok": True, "messages": [], "has_more": True,
                                     "response_metadata": {"next_cursor": "c1"}})
    with pytest.raises(HistoryFetchError, match="empty page"):
        await page_messages(method, channel_id=CH)


async def test_pager_parameterizes_inclusive_for_the_bootstrap_resume():
    method = AsyncMock(return_value={"ok": True, "messages": []})
    await page_messages(method, channel_id=CH, inclusive=False)
    assert method.call_args.kwargs["inclusive"] is False


async def test_budget_page_counter_is_shared_and_atomic():
    budget = FetchBudget(page_ceiling=3, total_seconds=60)
    method = AsyncMock(return_value={"ok": True, "messages": []})

    async def _one():
        await page_messages(method, channel_id=CH, budget=budget)

    await asyncio.gather(_one(), _one(), _one())
    assert budget.pages_used == 3
    with pytest.raises(HistoryFetchError, match="page ceiling"):
        await _one()


async def test_budget_deadline_stops_a_fetch():
    now = [0.0]
    budget = FetchBudget(total_seconds=10, page_ceiling=50, clock=lambda: now[0])
    calls = []

    async def _slow_page(**kwargs):
        calls.append(kwargs)
        now[0] += 99.0
        return {"ok": True, "messages": [raw("100.1")], "has_more": True,
                "response_metadata": {"next_cursor": "c1"}}

    with pytest.raises(HistoryFetchError, match="budget"):
        await page_messages(AsyncMock(side_effect=_slow_page), channel_id=CH, budget=budget)
    assert len(calls) == 1


async def test_fetch_page_retries_then_raises_with_the_error_code():
    from slack_client.history_fetch import HistoryPageError
    method = AsyncMock(return_value={"ok": False, "error": "channel_not_found"})
    with pytest.raises(HistoryPageError) as exc:
        await fetch_page(method, {"channel": CH}, attempts=2)
    assert exc.value.code == "channel_not_found"


async def test_fetch_page_recovers_on_a_later_attempt():
    method = AsyncMock(side_effect=[RuntimeError("flaky"), {"ok": True, "messages": []}])
    slept = []
    page = await fetch_page(method, {"channel": CH}, attempts=3,
                            sleeper=lambda d: slept.append(d) or asyncio.sleep(0))
    assert page.messages == []
    assert slept


# ------------------------------------------------------------------ window predicate

async def test_local_window_predicate_keeps_the_floor_and_h_and_drops_outside(client):
    client.app.client.conversations_history.return_value = {
        "ok": True, "messages": [raw(FLOOR), raw("1699999999.999999"), raw(H),
                                 raw("1700010000.000000")]}
    stream = await _stream(client=client, db=_db(), team_id=TEAM, channel_id=CH,
                                        h=H)
    assert [m.ts for m in stream.pinned.fetch_snapshot] == [FLOOR, H]


async def test_history_is_fetched_inclusive_with_the_window_as_bounds(client):
    await _stream(client=client, db=_db(), team_id=TEAM, channel_id=CH, h=H)
    kwargs = client.app.client.conversations_history.call_args.kwargs
    assert kwargs["oldest"] == FLOOR and kwargs["latest"] == H
    assert kwargs["inclusive"] is True
    assert kwargs["limit"] == config.history_page_size


# ------------------------------------------------------------------ fail-closed gates

@pytest.mark.parametrize("status", ["pending", "running"])
async def test_a_cold_inventory_no_longer_fails_a_turn(client, status):
    """T10. §2f: the inventory never gates a turn. A channel mid-sweep answers from what it can
    reach and DECLARES the state in its horizon, where it used to refuse outright."""
    stream = await _stream(client=client, db=_db(coverage=(status, None)),
                                        team_id=TEAM, channel_id=CH, h=H)
    assert stream.pinned.coverage.state == "cold"
    assert "have not indexed this channel's older threads yet" in stream.horizon_item.content
    with pytest.raises(ImportError):
        from message_processor.channel_stream import CoverageNotReady  # noqa: F401


async def test_an_absent_inventory_row_fetches_unfloored_rather_than_refusing(client):
    """T10's other half. NO row at all means no recorded floor, so there is no floor PREDICATE:
    the fetch takes what `(oldest reachable, H]` gives and the page ceiling is the backstop.

    Refusing here would have been incoherent with the case above — an absent row is strictly
    LESS settled than a `running` one, so a design where `running` answers and `absent` refuses
    has the two backwards.
    """
    db = _db(coverage=None, anchor=None)
    stream = await _stream(client=client, db=db, team_id=TEAM, channel_id=CH, h=H)

    assert stream.pinned.coverage is None
    assert stream.pinned.inventory_state == "absent"
    kwargs = client.app.client.conversations_history.call_args.kwargs
    assert "oldest" not in kwargs, "with no anchor there is no floor to page from"
    assert kwargs["latest"] == H
    assert "have not indexed this channel's older threads yet" in stream.horizon_item.content


async def test_an_inventory_floor_newer_than_h_renders_an_empty_stream(client):
    """The second raise site, also gone. A floor above H is an empty window, and an empty window
    is a thing the serializer has always been able to render: horizon, no messages, end marker.

    The inversion is decided LOCALLY and skips ALL of discovery. Asking Slack for
    `oldest > latest` is undocumented, so a mock that cheerfully returns success would prove only
    that our serializer copes with a cooperative API — not what happens against a strict one.

    THE SIDECARS ARE SEEDED ON PURPOSE. `_discovery_roots` draws roots from the activity index and
    the receipt ledger as well as from history, so a dirty root reaches `conversations.replies`
    with the same inverted bounds even when the history call is skipped. With empty activity rows
    the "neither API called" assertion below is vacuous — it passes against a build that skips
    only the history call, which is exactly the half-fix this test exists to catch.
    """
    db = _db(
        anchor={"floor_ts": FLOOR, "selection_version": 1},
        activity_roots={"1698000000.000000": "1698000500.000000"},
        receipt_roots=("1698000700.000000",),
    )
    stream = await _stream(client=client, db=db, team_id=TEAM, channel_id=CH,
                           h="1699000000.000000")
    assert stream.message_count == 0
    assert stream.items == (stream.horizon_item, stream.end_marker_item)
    client.app.client.conversations_history.assert_not_called()
    client.app.client.conversations_replies.assert_not_called()


async def test_a_failed_index_write_inside_the_window_fails_the_turn_closed(client):
    ticket = admission_watermark.watermark.issue(CH)
    admission_watermark.watermark.complete_failed(ticket)
    with pytest.raises(HistoryFetchError, match="failed"):
        await _stream(client=client, db=_db(), team_id=TEAM, channel_id=CH, h=H,
                                   frontier=ticket.seq)
    client.app.client.conversations_history.assert_not_called()
    await admission_watermark.watermark.shutdown(timeout=0.01)


async def test_a_failed_index_write_above_the_frontier_does_not_fail_the_turn(client):
    """Its event's activity_ts is greater than H, so it is outside this window by construction —
    the turn could not have rendered it either way."""
    admission_watermark.watermark.observe(CH, "1700000100.000000")
    pin = admission_watermark.watermark.pin(CH, "1700000100.000000")
    later = admission_watermark.watermark.issue(CH)
    admission_watermark.watermark.complete_failed(later)
    stream = await _stream(client=client, db=_db(), team_id=TEAM, channel_id=CH,
                                        h=H, frontier=pin.frontier)
    assert stream.message_count == 0
    assert admission_watermark.is_degraded(CH) is True   # still loud, just not this turn's
    await admission_watermark.watermark.shutdown(timeout=0.01)


async def test_the_turn_waits_on_the_index_frontier(client):
    ticket = admission_watermark.watermark.issue(CH)
    task = asyncio.ensure_future(build_channel_stream(
        client=client, db=_db(), team_id=TEAM, channel_id=CH, h=H, frontier=ticket.seq,
        drain_timeout=5))
    await asyncio.sleep(0)
    assert not task.done()
    admission_watermark.watermark.complete_ok(ticket)
    await task
    assert task.result().stream.message_count == 0


async def test_a_malformed_h_fails_closed(client):
    with pytest.raises(StreamTimestampError):
        await _stream(client=client, db=_db(), team_id=TEAM, channel_id=CH,
                                   h="not-a-ts")


@pytest.mark.parametrize("payload", [
    {"text": "no ts at all", "user": "U1"},
    {"ts": "", "text": "empty", "user": "U1"},
    {"ts": "yesterday", "text": "words", "user": "U1"},
])
async def test_a_fetched_message_we_cannot_place_in_time_fails_the_turn(client, payload):
    """Not skipped. A message the window predicate cannot judge is a hole in a stream whose whole
    claim is that it shows the room, and nothing downstream could ever see the hole."""
    client.app.client.conversations_history.return_value = {
        "ok": True, "messages": [raw("1700000100.000000"), payload]}
    with pytest.raises(StreamTimestampError):
        await _stream(client=client, db=_db(), team_id=TEAM, channel_id=CH, h=H)


async def test_a_declined_subtype_is_still_skipped_not_raised(client):
    """The difference that matters: a join notice is a decision about what counts as a message,
    not a record we failed to read."""
    client.app.client.conversations_history.return_value = {
        "ok": True, "messages": [raw("1700000100.000000"),
                                 raw("1700000200.000000", subtype="channel_join")]}
    stream = await _stream(client=client, db=_db(), team_id=TEAM, channel_id=CH,
                                        h=H)
    assert [m.ts for m in stream.pinned.fetch_snapshot] == ["1700000100.000000"]


async def test_a_sidecar_row_with_an_unusable_ts_fails_the_turn(client):
    """The pin still fails closed on a row it cannot place in time — but it is READ AFTER the
    fetch now, because its subject is the candidate identities the fetch returned. A row we
    quietly dropped would decide the role of one of our own messages with nothing saying so."""
    db = _db(receipts=[{"message_ts": "soon", "state": "finalized", "turn_id": "t",
                        "thread_root_ts": None}])
    with pytest.raises(StreamTimestampError):
        await _stream(client=client, db=db, team_id=TEAM, channel_id=CH, h=H)


# ------------------------------------------------------------------ root discovery

def _norm(client, payload, origin=ORIGIN_HISTORY):
    return normalize_slack_message(client, payload, channel_id=CH, origin=origin, team_id=TEAM)


def test_discovery_roots_takes_all_four_sources(client):
    history = [
        _norm(client, raw("1700000100.000000", reply_count=2, latest_reply="1700000200.0")),
        _norm(client, raw("1700000300.000000", root="1700000050.000000")),
        _norm(client, raw("1700000400.000000")),
    ]
    roots = _discovery_roots(history, {"activity_roots": {"1699000000.000000": None},
                                   "receipt_roots": ("1698000000.000000",)}, H)
    assert set(roots) == {"1700000100.000000", "1700000050.000000",
                         "1699000000.000000", "1698000000.000000"}
    assert "1700000400.000000" not in roots      # no hints, no replies to fetch


def test_discovery_roots_drops_roots_newer_than_h(client):
    assert _discovery_roots([], {"activity_roots": {"1800000000.000000": None},
                                 "receipt_roots": ()}, H) == []


@pytest.mark.parametrize("receipt_ts", ["1699000000.000000", "1800000000.000000"])
def test_discovery_roots_ignores_a_receipt_outside_the_window(client, receipt_ts):
    """A receipt whose own message is outside the window can decide nothing about this stream, so
    its thread must not join this turn's fetch work — that is how a receipt committed after
    admission came to spend an older turn's page budget, and could fail it."""
    assert _discovery_roots([], {"activity_roots": {},
                                 "receipt_roots": (receipt_ts,)}, H) == (
        [] if float(receipt_ts) > float(H) else [receipt_ts])


async def test_a_post_h_receipt_adds_no_fetch_work_to_an_admitted_turn(client):
    """End to end: even handed the row, the build schedules nothing for it. A replies fetch here
    would spend the shared page budget of a turn whose stream cannot contain the message, and a
    failure fetching it would fail that turn."""
    db = _db(receipts=[{"message_ts": "1800000000.000000", "state": "finalized",
                        "turn_id": "later", "thread_root_ts": "1700000100.000000"}])
    stream = await _stream(client=client, db=db, team_id=TEAM, channel_id=CH, h=H)
    client.app.client.conversations_replies.assert_not_called()
    assert stream.message_count == 0


def test_a_thread_ts_that_cannot_be_placed_never_reaches_the_inventory(client):
    """Two guards, one rule: discarding an unplaceable root would drop a whole thread from the
    window with nothing saying so. The normalizer refuses it first (r2-2), and the inventory still
    refuses a record handed to it any other way."""
    from message_processor.channel_stream import StreamTimestampError
    with pytest.raises(TimestampError):
        _norm(client, raw("1700000100.000000", root="whenever"))

    placed = _norm(client, raw("1700000100.000000"))
    unplaceable = replace(placed, thread_root_ts="whenever")
    with pytest.raises(StreamTimestampError):
        _discovery_roots([unplaceable], {"activity_roots": {}, "receipt_roots": ()}, H)


async def test_replies_are_fetched_per_root_with_the_window_and_the_root_ts(client):
    client.app.client.conversations_history.return_value = {
        "ok": True, "messages": [raw("1700000100.000000", reply_count=1,
                                     latest_reply="1700000200.000000")]}
    client.app.client.conversations_replies.return_value = {
        "ok": True, "messages": [raw("1700000100.000000", reply_count=1),
                                 raw("1700000200.000000", root="1700000100.000000")]}
    stream = await _stream(client=client, db=_db(), team_id=TEAM, channel_id=CH,
                                        h=H)
    kwargs = client.app.client.conversations_replies.call_args.kwargs
    assert kwargs["ts"] == "1700000100.000000"
    assert (kwargs["oldest"], kwargs["latest"], kwargs["inclusive"]) == (FLOOR, H, True)
    assert [m.ts for m in stream.pinned.fetch_snapshot] == ["1700000100.000000",
                                                            "1700000200.000000"]


async def test_a_pre_floor_root_is_reached_only_through_the_index(client):
    """The case the index exists for: the parent keeps its original ts, so history(oldest=floor)
    can never surface it, and its in-window replies would be invisible."""
    # A recent top-level message, so the walk fetches SOMETHING: with zero events of any kind
    # the build short-circuits before discovery and there is no effective floor to window on.
    client.app.client.conversations_history.return_value = {
        "ok": True, "messages": [raw("1700000400.000000", text="recent")]}
    client.app.client.conversations_replies.return_value = {
        "ok": True, "messages": [raw("1699000000.000000", text="old root"),
                                 raw("1700000500.000000", text="new reply",
                                     root="1699000000.000000")]}
    stream = await _stream(
        client=client,
        db=_db(activity_roots={"1699000000.000000": "1700000500.000000"}),
        team_id=TEAM, channel_id=CH, h=H)
    # The root itself is outside the window and does NOT render; its reply does.
    # The reply IS discovered — only the activity index could have surfaced it, because
    # `conversations.history` returns top-level messages and the root predates the window.
    assert "1700000500.000000" in [m.ts for m in stream.pinned.fetch_snapshot]
    # The root itself is below the floor and does NOT render as its own item.
    assert "1699000000.000000" not in [i.metadata.get("ts") for i in stream.message_items]


async def test_dirty_is_cleared_compare_and_clear_after_a_complete_replies_fetch(client):
    db = _db(activity_roots={"1700000100.000000": "1700000500.000000"})
    client.app.client.conversations_history.return_value = {
        "ok": True, "messages": [raw("1700000400.000000", text="recent")]}
    client.app.client.conversations_replies.return_value = {"ok": True, "messages": []}
    await _stream(client=client, db=db, team_id=TEAM, channel_id=CH, h=H)
    db.clear_thread_dirty_async.assert_awaited_once_with(
        TEAM, CH, "1700000100.000000", if_event_ts_equals="1700000500.000000")


async def test_dirty_is_not_cleared_for_a_root_the_index_never_flagged(client):
    client.app.client.conversations_history.return_value = {
        "ok": True, "messages": [raw("1700000100.000000", reply_count=1,
                                     latest_reply="1700000200.000000")]}
    db = _db()
    await _stream(client=client, db=db, team_id=TEAM, channel_id=CH, h=H)
    db.clear_thread_dirty_async.assert_not_called()


async def test_a_failed_replies_fetch_cancels_and_awaits_its_siblings(client):
    started = []
    finished = []

    async def _replies(**kwargs):
        root = kwargs["ts"]
        started.append(root)
        if root == "1700000100.000000":
            raise RuntimeError("boom")
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            finished.append(root)
            raise
        return {"ok": True, "messages": []}

    client.app.client.conversations_replies = AsyncMock(side_effect=_replies)
    client.app.client.conversations_history.return_value = {
        "ok": True, "messages": [raw("1700000400.000000", text="recent")]}
    roots = {"1700000100.000000": None, "1700000900.000000": None}
    with pytest.raises(HistoryFetchError):
        await _stream(client=client, db=_db(activity_roots=roots), team_id=TEAM,
                      channel_id=CH, h=H,
                      # An injected budget must carry the SHARED ABSOLUTE deadline. The
                      # `total_seconds` form gives this component its own clock, which the
                      # composer now refuses — three such budgets could spend 3x the window.
                      history_budget=FetchBudget(page_ceiling=50,
                                                 deadline_at=time.monotonic() + 60))
    assert finished == ["1700000900.000000"]


async def test_replies_concurrency_is_bounded(client, monkeypatch):
    monkeypatch.setattr(config, "reply_fetch_concurrency", 2)
    live = 0
    peak = 0

    async def _replies(**kwargs):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0)
        live -= 1
        return {"ok": True, "messages": []}

    client.app.client.conversations_replies = AsyncMock(side_effect=_replies)
    activity = [{"root_ts": f"17000001{i:02d}.000000", "dirty": 1} for i in range(6)]
    await _stream(client=client, db=_db(activity=activity), team_id=TEAM,
                               channel_id=CH, h=H,
                               history_budget=FetchBudget(
                                   page_ceiling=50, deadline_at=time.monotonic() + 60))
    assert peak <= 2


# ------------------------------------------------------------------ actor map

async def test_actor_map_covers_senders_and_mentions_and_leaves_unknowns_raw(client):
    messages = [
        _norm(client, raw("100.1", user="U1", text="hey <@U2> and <@U_UNKNOWN>")),
        _norm(client, raw("100.2", user="B9", text="bot said", bot_id="B9",
                          username="Jira")),
        _norm(client, raw("100.3", user=client.bot_user_id, text="mine")),
    ]
    actors = dict(await _build_actor_map(client, messages))
    assert actors["U1"] == "name-U1" and actors["U2"] == "name-U2"
    assert actors["B9"] == "Jira"
    assert actors[client.bot_user_id] == "chatgpt-dev"
    assert "U_UNKNOWN" not in actors


async def test_actor_map_resolution_is_read_only(client):
    messages = [_norm(client, raw("100.1", user="U1"))]
    await _build_actor_map(client, messages)
    assert client.resolved == [["U1"]]     # one batched, budgeted, read-only call


async def test_actor_map_survives_a_resolver_failure(client):
    client.resolve_usernames = AsyncMock(side_effect=RuntimeError("slack down"))
    actors = dict(await _build_actor_map(client, [_norm(client, raw("100.1", user="U1"))]))
    assert actors == {}


# ------------------------------------------------------------------ actor tail hydration

async def test_actor_tail_is_hydrated_from_the_fetch(client):
    client.app.client.conversations_history.return_value = {
        "ok": True, "messages": [raw("1700000100.000000", user="B9", bot_id="B9")]}
    await _stream(client=client, db=_db(), team_id=TEAM, channel_id=CH, h=H)
    assert actor_tail_module.thread_has_other_bot(CH, "1700000100.000000") is True


async def test_our_own_messages_are_not_hydrated_into_the_tail(client):
    """The tail answers "has another bot spoken in this thread". A self record does not count as
    another bot but does take a slot in a bounded ring, so hydrating them can evict the other-bot
    record the continuation veto reads — and the veto silently stops holding."""
    thread = "1700000100.000000"
    client.app.client.conversations_history.return_value = {"ok": True, "messages": [
        raw(thread, user="B9", bot_id="B9", reply_count=1, latest_reply="1700000900.000000")]}
    client.app.client.conversations_replies.return_value = {"ok": True, "messages": [
        raw(thread, user="B9", bot_id="B9"),
        *[raw(f"17000002{i:02d}.000000", user=client.bot_user_id, root=thread)
          for i in range(9)]]}

    await _stream(client=client, db=_db(), team_id=TEAM, channel_id=CH, h=H)

    assert actor_tail_module.thread_has_other_bot(CH, thread) is True
    entries = actor_tail_module.actor_tail.entries(CH, thread)
    assert all(e.sender_type != "self" for e in entries)


async def test_a_live_mutation_mid_fetch_makes_the_reconcile_skip(client):
    async def _history(**kwargs):
        actor_tail_module.record(CH, ts="1700000900.000000", root_ts="1700000900.000000",
                                 is_bot=False, sender_type="human")
        return {"ok": True, "messages": [raw("1700000100.000000", user="B9", bot_id="B9")]}

    client.app.client.conversations_history = AsyncMock(side_effect=_history)
    await _stream(client=client, db=_db(), team_id=TEAM, channel_id=CH, h=H)
    # Live won: the fetched bot entry was NOT written, and the live entry survived.
    assert actor_tail_module.thread_has_other_bot(CH, "1700000100.000000") is False
    assert actor_tail_module.actor_tail.entries(CH, "1700000900.000000")


# ------------------------------------------------------------------ pinning + determinism

async def test_the_pinned_tuple_records_the_window_and_the_hashes(client):
    stream = await _stream(client=client, db=_db(), team_id=TEAM, channel_id=CH,
                                        h=H, capability_profile_hash="cap",
                                        tool_schema_version="v9")
    pin = stream.pinned
    assert pin.window == (FLOOR, True)
    assert pin.H == H
    assert pin.sidecar_versions_hash == "h"
    assert pin.capability_profile_hash == "cap" and pin.tool_schema_version == "v9"
    assert pin.serializer_config_hash and pin.actor_map_hash
    # `boundary` is retired: the field is now named for what it IS, the periphery floor.
    assert stream.stream_render_fields()["periphery_floor_ts"] == pin.periphery_floor_ts
    assert stream.stream_render_fields()["H"] == H


async def test_two_independent_builds_agree_byte_for_byte(client):
    """The single-stream claim: the canonical build never sees the origin thread or the
    requester, so two turns in the same channel share the whole cacheable prefix."""
    page = {"ok": True, "messages": [raw("1700000100.000000", user="U1", text="a"),
                                     raw("1700000200.000000", user="U2", text="b",
                                         root="1700000100.000000")]}
    replies = {"ok": True, "messages": [raw("1700000100.000000", user="U1", text="a",
                                            reply_count=1),
                                        raw("1700000200.000000", user="U2", text="b",
                                            root="1700000100.000000")]}

    async def _build():
        c = _Client()
        c.app.client.conversations_history = AsyncMock(return_value=dict(page))
        c.app.client.conversations_replies = AsyncMock(return_value=dict(replies))
        actor_tail_module.actor_tail.reset()
        return await _stream(client=c, db=_db(), team_id=TEAM, channel_id=CH, h=H)

    first, second = await _build(), await _build()
    assert [i.content for i in first.items] == [i.content for i in second.items]
    assert first.stream_sha256 == second.stream_sha256
    assert first.pinned.actor_map_hash == second.pinned.actor_map_hash
    assert first.receipts_membership_hash == second.receipts_membership_hash


async def test_replies_origin_wins_the_dedup_in_a_real_build(client):
    client.app.client.conversations_history.return_value = {
        "ok": True, "messages": [raw("1700000100.000000", reply_count=1,
                                     latest_reply="1700000200.000000"),
                                 raw("1700000200.000000", root="1700000100.000000",
                                     subtype="thread_broadcast")]}
    client.app.client.conversations_replies.return_value = {
        "ok": True, "messages": [raw("1700000200.000000", root="1700000100.000000",
                                     subtype="thread_broadcast")]}
    stream = await _stream(client=client, db=_db(), team_id=TEAM, channel_id=CH,
                                        h=H)
    snapshot = {m.ts: m for m in stream.pinned.fetch_snapshot}
    assert snapshot["1700000200.000000"].origin == ORIGIN_REPLIES
    assert snapshot["1700000200.000000"].is_broadcast is True
    assert stream.message_count == 2


async def test_the_dev_barrier_fires_after_read_one_and_before_any_fetch(client, monkeypatch):
    """The seam a live battery freezes to prove the stream is current as of admission.

    Its subject MOVED in W2: the barrier now sits after READ 1 (the anchor and inventory, both
    pinned) and before any Slack call. The RENDER pin comes after the fetch, because its subject
    is the candidate identities the fetch returned — so hooking the sidecar read here would wrap
    a call that happens later, or (as it did) a name that no longer exists at all.
    """
    order = []
    db = _db()
    original = db.read_channel_window_anchor_async

    async def _read1(*a, **k):
        order.append("read1")
        return await original(*a, **k)

    db.read_channel_window_anchor_async = _read1

    async def _barrier(**context):
        order.append("barrier")
        return True

    monkeypatch.setattr(channel_stream.dev_barriers, "post_admission", _barrier)

    async def _history(**kwargs):
        order.append("history")
        return {"ok": True, "messages": []}

    client.app.client.conversations_history = AsyncMock(side_effect=_history)
    await _stream(client=client, db=db, team_id=TEAM, channel_id=CH, h=H)
    assert order == ["read1", "barrier", "history"]


# ============================================ the ORIGIN thread (§2e item 3, T19-T23)

ORIGIN = "1699500000.000000"


def _pages(*batches):
    """A `conversations.replies` stand-in that walks cursors, recording its call kwargs."""
    calls = []

    async def method(**kwargs):
        calls.append(kwargs)
        index = int(kwargs["cursor"]) if kwargs.get("cursor") else 0
        batch = batches[index]
        page = {"ok": True, "messages": list(batch)}
        if index + 1 < len(batches):
            page["has_more"] = True
            page["response_metadata"] = {"next_cursor": str(index + 1)}
        return page

    method.calls = calls
    return method


async def test_the_origin_thread_is_paged_to_completion(client):
    """T19. Three pages, all present, and the request carries `ts=<root>`, `latest=H`,
    `inclusive=True` and NO `oldest` — the origin is never floored, because a thread is complete
    or it is not answered from at all."""
    method = _pages([raw(ORIGIN, text="root")],
                    [raw("1699500100.000000", text="mid", root=ORIGIN)],
                    [raw("1699500200.000000", text="last", root=ORIGIN)])
    client.app.client.conversations_replies = method

    fetch = await fetch_origin_thread(client, CH, ORIGIN, H,
                                      FetchBudget(page_ceiling=None), trigger_ts=None)

    assert [m.ts for m in fetch.messages] == [ORIGIN, "1699500100.000000",
                                              "1699500200.000000"]
    assert method.calls[0]["ts"] == ORIGIN
    assert method.calls[0]["latest"] == H and method.calls[0]["inclusive"] is True
    assert "oldest" not in method.calls[0], "an origin fetch is NEVER floored"


async def test_origin_failures_are_named_not_string_matched(client):
    """T20. The origin raises `OriginFetchError`, `_channel_stream_failure` maps it to
    `origin_fetch_failed` BY TYPE, and a periphery failure still maps to
    `history_fetch_failed`. Collapsing both onto `HistoryFetchError` loses the distinction."""
    from message_processor.base import MessageProcessor

    async def _broken(**kwargs):
        return {"ok": True, "messages": [{"ts": "1.0"}], "has_more": True}

    client.app.client.conversations_replies = _broken
    with pytest.raises(OriginFetchError):
        await fetch_origin_thread(client, CH, ORIGIN, H, FetchBudget(page_ceiling=None),
                                  trigger_ts=None)

    origin_code, _ = MessageProcessor._channel_stream_failure(OriginFetchError("x"))
    history_code, _ = MessageProcessor._channel_stream_failure(HistoryFetchError("y"))
    assert origin_code == "origin_fetch_failed"
    assert history_code == "history_fetch_failed"


@pytest.mark.parametrize("outcome,code,top_level,expected", [
    # §2e's table, row by row, EACH under both root/trigger relations.
    ("empty", None, True, "fallback"),          # `ok`, zero messages
    ("empty", None, False, "fail"),
    ("messages", None, True, "normal"),         # `ok`, messages
    ("messages", None, False, "normal"),
    ("error", "thread_not_found", True, "fallback"),
    ("error", "thread_not_found", False, "fail"),
    ("error", "channel_not_found", True, "fail"),   # any other error code
    ("error", "channel_not_found", False, "fail"),
    ("error", "not_in_channel", True, "fail"),
    ("error", "not_in_channel", False, "fail"),
    ("error", "invalid_auth", True, "fail"),
    ("error", "invalid_auth", False, "fail"),
    ("error", "access_denied", True, "fail"),
    ("error", "access_denied", False, "fail"),
    ("error", "ratelimited", True, "fail"),
    ("error", "ratelimited", False, "fail"),
    ("error", "some_code_nobody_has_seen", True, "fail"),   # UNRECOGNISED -> fails closed
    ("error", "some_code_nobody_has_seen", False, "fail"),
    ("malformed", None, True, "fail"),          # cursor contradiction
    ("malformed", None, False, "fail"),
    ("exhausted", None, True, "fail"),          # budget / wall clock
    ("exhausted", None, False, "fail"),
])
async def test_the_origin_error_taxonomy_table(client, monkeypatch, outcome, code, top_level,
                                               expected):
    """T21. §2e's table is an ALLOWLIST: only `ok`-with-zero-messages and `thread_not_found` on a
    TOP-LEVEL trigger reach the fallback. Everything else — including a code nobody has seen —
    fails closed, which is the allowlist's default and what makes the unverified
    `thread_not_found` safe to write down."""
    # `ratelimited` is the one code that goes round the retry ladder; one attempt IS "past
    # retries", which is the state the table's row names.
    monkeypatch.setattr(config, "fetch_retry_attempts", 1, raising=False)

    if outcome == "empty":
        async def _replies(**kwargs):
            return {"ok": True, "messages": []}
    elif outcome == "messages":
        async def _replies(**kwargs):
            return {"ok": True, "messages": [raw(ORIGIN, text="root")]}
    elif outcome == "malformed":
        async def _replies(**kwargs):
            return {"ok": True, "messages": [raw(ORIGIN)], "has_more": True}
    else:
        async def _replies(**kwargs):
            raise SlackApiError("boom", {"ok": False, "error": code})

    client.app.client.conversations_replies = _replies
    trigger = ORIGIN if top_level else "1699900000.000000"
    budget = (FetchBudget(deadline_at=0.0, page_ceiling=None) if outcome == "exhausted"
              else FetchBudget(page_ceiling=None))

    if expected == "fallback":
        fetch = await fetch_origin_thread(client, CH, ORIGIN, H, budget, trigger_ts=trigger)
        assert fetch.empty_fallback is True and fetch.messages == ()
    elif expected == "normal":
        fetch = await fetch_origin_thread(client, CH, ORIGIN, H, budget, trigger_ts=trigger)
        assert fetch.empty_fallback is False
        assert [m.ts for m in fetch.messages] == [ORIGIN]
    else:
        with pytest.raises(OriginFetchError):
            await fetch_origin_thread(client, CH, ORIGIN, H, budget, trigger_ts=trigger)


async def test_an_empty_established_thread_fails_closed(client):
    """T22. A REPLY-triggered turn whose thread comes back empty FAILS — silently replacing a
    real thread with one message is the corruption OWNER-2 forbids. The top-level propagation
    case in the same table passes."""
    async def _replies(**kwargs):
        return {"ok": True, "messages": []}

    client.app.client.conversations_replies = _replies
    with pytest.raises(OriginFetchError):
        await fetch_origin_thread(client, CH, ORIGIN, H, FetchBudget(page_ceiling=None),
                                  trigger_ts="1699900000.000000")

    ok = await fetch_origin_thread(client, CH, ORIGIN, H, FetchBudget(page_ceiling=None),
                                   trigger_ts=ORIGIN)
    assert ok.empty_fallback is True


def _outstanding_tasks():
    """Every task still alive besides the one running this test."""
    return [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]


async def test_a_failing_component_cancels_and_awaits_the_other(client):
    """T23. The two components run concurrently, so whichever fails first must CANCEL the other
    and AWAIT it before propagating — AND VICE VERSA. An orphaned fetch would keep spending wall
    clock after the turn had already failed, so no task may outlive the turn either way."""
    # --- the ORIGIN fails; the PERIPHERY is cancelled and awaited ---------------------------
    awaited = []

    async def _slow_history(**kwargs):
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            awaited.append("history")
            raise
        return {"ok": True, "messages": []}

    async def _failing_origin(**kwargs):
        await asyncio.sleep(0)
        raise SlackApiError("boom", {"ok": False, "error": "invalid_auth"})

    client.app.client.conversations_history = AsyncMock(side_effect=_slow_history)
    client.app.client.conversations_replies = AsyncMock(side_effect=_failing_origin)

    before = _outstanding_tasks()
    with pytest.raises(OriginFetchError):
        await _stream(client=client, db=_db(), team_id=TEAM, channel_id=CH, h=H,
                      origin_root_ts=ORIGIN, trigger_ts=ORIGIN)
    assert awaited == ["history"], "the surviving component must be cancelled AND awaited"
    assert _outstanding_tasks() == before, "no fetch task may outlive the turn"

    # --- AND VICE VERSA: the PERIPHERY fails; the ORIGIN is cancelled and awaited ------------
    other = _Client()
    awaited_origin = []

    async def _failing_history(**kwargs):
        await asyncio.sleep(0)
        raise SlackApiError("boom", {"ok": False, "error": "channel_not_found"})

    async def _slow_origin(**kwargs):
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            awaited_origin.append("origin")
            raise
        return {"ok": True, "messages": []}

    other.app.client.conversations_history = AsyncMock(side_effect=_failing_history)
    other.app.client.conversations_replies = AsyncMock(side_effect=_slow_origin)

    before = _outstanding_tasks()
    with pytest.raises(HistoryFetchError):
        await _stream(client=other, db=_db(), team_id=TEAM, channel_id=CH, h=H,
                      origin_root_ts=ORIGIN, trigger_ts=ORIGIN)
    assert awaited_origin == ["origin"], "the origin must be cancelled AND awaited too"
    assert _outstanding_tasks() == before, "no fetch task may outlive the turn"


# ================================ the walk, discovery and the three budgets (T35-T76)

def _walk(pages):
    """A `conversations.history` stand-in that walks cursors and records its call kwargs."""
    calls = []

    async def method(**kwargs):
        calls.append(kwargs)
        index = int(kwargs["cursor"]) if kwargs.get("cursor") else 0
        page = {"ok": True, "messages": list(pages[index])}
        if index + 1 < len(pages):
            page["has_more"] = True
            page["response_metadata"] = {"next_cursor": str(index + 1)}
        return page

    method.calls = calls
    return method


def _roots(n, *, start=1700000000, sender="U1"):
    return [raw(f"{start + i}.000000", text=f"r{i}", user=sender) for i in range(n)]


async def test_a_message_above_h_is_in_neither_block(client):
    """T35. H is pinned at admission, so an event arriving after it belongs to the NEXT turn —
    in the periphery and in the origin alike."""
    above = "1700009999.999999"
    client.app.client.conversations_history = _walk([[raw("1700000100.000000"), raw(above)]])
    client.app.client.conversations_replies.return_value = {
        "ok": True, "messages": [raw(ORIGIN, text="root"), raw(above, root=ORIGIN)]}

    stream = await _stream(client=client, db=_db(), team_id=TEAM, channel_id=CH,
                           h="1700005000.000000", origin_root_ts=ORIGIN, trigger_ts=ORIGIN)

    assert above not in [m.ts for m in stream.pinned.fetch_snapshot]
    assert above not in [m.ts for m in stream.pinned.origin_snapshot]


async def test_the_walk_stops_on_guaranteed_eligible_roots(client, monkeypatch):
    """T64. A window whose newest roots are mostly OUR OWN receiptless posts: the walk does NOT
    stop there — it continues until CEILING+1 GUARANTEED-ELIGIBLE roots are proved.

    Stopping on RAW roots would leave fewer than TARGET survivors while older eligible roots were
    never fetched, which is the under-fill the predicate exists to make impossible."""
    monkeypatch.setattr(config, "channel_window_target", 2, raising=False)
    monkeypatch.setattr(config, "channel_window_ceiling", 4, raising=False)

    ours = [raw(f"{1700009000 + i}.000000", user="UBOT", bot_id="B_BOT") for i in range(6)]
    theirs = _roots(6, start=1700000000)
    client.app.client.conversations_history = _walk([ours, theirs])

    await _stream(client=client, db=_db(anchor=None), team_id=TEAM, channel_id=CH, h=H)
    # It had to ask for the SECOND page: the first held six raw roots but zero it could count.
    assert len(client.app.client.conversations_history.calls) == 2


async def test_a_stale_floor_still_stops_early(client, monkeypatch):
    """T39. A floor far below thousands of newer roots: the walk stops on the COUNT rather than
    paging all the way down to it. Stopping early is CORRECT — the count rule re-anchors above
    that depth anyway, so the abandoned pages are exactly what the shallow design refuses to
    pay for."""
    monkeypatch.setattr(config, "channel_window_target", 2, raising=False)
    monkeypatch.setattr(config, "channel_window_ceiling", 3, raising=False)

    pages = [_roots(4, start=1700000000 + p * 100) for p in range(5)]
    client.app.client.conversations_history = _walk(pages)

    result = await build_channel_stream(
        client=client, db=_db(anchor={"floor_ts": "1.0", "selection_version": 1}),
        team_id=TEAM, channel_id=CH, h=H)
    assert len(client.app.client.conversations_history.calls) == 1, (
        "the count bound must fire before the floor is reached")
    # AND THE COUNT RULE RE-ANCHORS ABOVE IT: four guaranteed-eligible roots over a ceiling of
    # three, so `F'` is the oldest of the newest TARGET roots — nowhere near the stale floor.
    assert result.stream.periphery_floor_ts == "1700000002.000000"
    assert result.reselected is True


async def test_discovery_windows_on_the_effective_floor(client, monkeypatch):
    """T65. A stale stored floor months below the traffic — `F` IS SET and the walk does NOT
    reach it: discovery is windowed on the walk's EARLY-STOP floor, the oldest fetched EVENT, not
    the stored one. This is the WARM EARLY-STOP branch of step 7's precedence and it is the common
    case, not an edge.

    What it bounds is the DB ROOT FAN-OUT, not merely the history depth: the index is modelled
    as the accessor really behaves, so windowing on `F` hands back the months of roots below the
    window and the turn fans out to every one of them — C05, reached through the database
    instead of through Slack."""
    monkeypatch.setattr(config, "channel_window_target", 2, raising=False)
    monkeypatch.setattr(config, "channel_window_ceiling", 3, raising=False)

    ancient = ["1690000000.000000", "1690000100.000000", "1690000200.000000"]
    in_window = "1700000300.000000"

    async def _discovery(team, channel, *, floor_ts, high_ts):
        floor = parse_ts(floor_ts)
        return {"activity_roots": {r: None for r in (*ancient, in_window)
                                   if parse_ts(r) >= floor},
                "receipt_roots": ()}

    pages = [_roots(4, start=1700000000 + p * 100) for p in range(3)]
    client.app.client.conversations_history = _walk(pages)
    client.app.client.conversations_replies.return_value = {"ok": True, "messages": []}
    db = _db(anchor={"floor_ts": "1.0", "selection_version": 1})
    db.read_channel_discovery_roots_async = AsyncMock(side_effect=_discovery)

    await _stream(client=client, db=db, team_id=TEAM, channel_id=CH, h=H)

    floor_used = db.read_channel_discovery_roots_async.call_args.kwargs["floor_ts"]
    assert floor_used == "1700000000.000000", (
        "windowing on the STORED floor is the fan-out this prevents")
    fanned = sorted(c.kwargs["ts"]
                    for c in client.app.client.conversations_replies.call_args_list)
    assert fanned == [in_window], "the DB root fan-out is what the effective floor bounds"


async def test_a_cold_channel_defers_its_activity_read(client):
    """T38. With no anchor row: READ 1 returns anchor+inventory ONLY, the walk runs, and the
    activity read runs AFTER it — windowed against the floor the walk actually produced.

    The index here is modelled as the accessor really behaves, returning only roots at or above
    the floor it is handed. That is what makes the fan-out assertion load-bearing: an index
    queried BEFORE the walk has no floor to window on, hands back the ancient root as well, and
    the turn pays a `conversations.replies` for a thread the window cannot contain."""
    order = []
    ancient = "1690000000.000000"
    recent = "1700000400.000000"

    async def _anchor(*a, **k):
        order.append("read1")
        return {"anchor": None, "inventory": None}

    async def _discovery(team, channel, *, floor_ts, high_ts):
        order.append("discovery")
        floor = parse_ts(floor_ts)
        return {"activity_roots": {r: None for r in (ancient, recent)
                                   if parse_ts(r) >= floor},
                "receipt_roots": ()}

    async def _history(**kwargs):
        order.append("history")
        return {"ok": True, "messages": [raw(recent)]}

    db = _db(anchor=None)
    db.read_channel_window_anchor_async = AsyncMock(side_effect=_anchor)
    db.read_channel_discovery_roots_async = AsyncMock(side_effect=_discovery)
    client.app.client.conversations_history = AsyncMock(side_effect=_history)
    client.app.client.conversations_replies.return_value = {"ok": True, "messages": []}

    await _stream(client=client, db=db, team_id=TEAM, channel_id=CH, h=H)

    assert order == ["read1", "history", "discovery"]
    # THE PROVISIONAL FLOOR: the oldest event the walk actually fetched, not "0" and not a floor
    # nobody read.
    assert db.read_channel_discovery_roots_async.call_args.kwargs["floor_ts"] == recent
    fanned = [c.kwargs["ts"] for c in client.app.client.conversations_replies.call_args_list]
    assert fanned == [recent], "an unwindowed index read drags the ancient root into the fan-out"


async def test_the_three_budgets_are_independent(client):
    """T40. The history walk keeps the page ceiling; the reply fan-out and the origin leave the
    page budget entirely. One shared budget cannot serve this design — a hundred reply-bearing
    roots and a forty-page origin spend 140 pages between them, and the walk's own fifty-page
    ceiling would have refused the turn long before either finished."""
    origin_pages = 40
    roots = _roots(100, start=1700000000)

    async def _replies(**kwargs):
        ts = kwargs["ts"]
        if ts == ORIGIN:
            index = int(kwargs["cursor"]) if kwargs.get("cursor") else 0
            page = {"ok": True, "messages": [raw(f"{1699500000 + index}.000000",
                                                 root=ORIGIN, text=f"o{index}")]}
            if index + 1 < origin_pages:
                page["has_more"] = True
                page["response_metadata"] = {"next_cursor": str(index + 1)}
            return page
        return {"ok": True, "messages": [raw(ts, reply_count=1),
                                         raw(f"{ts.split('.')[0]}.000500", root=ts)]}

    client.app.client.conversations_history = _walk([
        [dict(r, reply_count=1, latest_reply=f"{r['ts'].split('.')[0]}.000500") for r in roots]])
    client.app.client.conversations_replies = AsyncMock(side_effect=_replies)

    history_budget = FetchBudget(deadline_at=1e12)          # omitted ceiling == config's
    reply_budget = FetchBudget(deadline_at=1e12, page_ceiling=None)
    origin_budget = FetchBudget(deadline_at=1e12, page_ceiling=None)

    result = await build_channel_stream(
        client=client, db=_db(), team_id=TEAM, channel_id=CH, h=H,
        origin_root_ts=ORIGIN, trigger_ts=ORIGIN,
        history_budget=history_budget, reply_budget=reply_budget,
        origin_budget=origin_budget)

    assert history_budget.pages_used <= config.history_page_ceiling
    assert reply_budget.pages_used == 100, "the fan-out leaves the page budget entirely"
    assert origin_budget.pages_used == origin_pages, "and so does the origin"
    assert result.pages == (history_budget.pages_used, 100, origin_pages)
    # ONE SHARED BUDGET COULD NOT HAVE SERVED THIS TURN: the three components spent well past the
    # walk's own ceiling, and a single counter would have refused mid-fan-out.
    assert (history_budget.pages_used + reply_budget.pages_used
            + origin_budget.pages_used) > config.history_page_ceiling


async def test_the_three_budgets_share_one_deadline(client, monkeypatch):
    """T68. THE BUILDER constructs all three budgets itself, from ONE absolute deadline taken at
    fetch start — so a walk that burns 55 of the 60 seconds leaves the reply and origin components
    ~5s, not 60s each.

    The clock is substituted on the budgets the BUILDER made, which is the only way to see the
    property: three independently started 60-second windows carry no absolute deadline at all and
    would each still report a full minute after the walk had spent one."""
    made = []
    clock_now = [0.0]
    real_budget = channel_stream.FetchBudget

    def _recording_budget(**kwargs):
        budget = real_budget(**kwargs)
        if not made:
            # Anchor the fake clock at fetch start: the builder's own deadline, minus the window
            # it took it from.
            clock_now[0] = float(budget.deadline_at) - float(config.fetch_retry_total_seconds)
        budget._clock = lambda: clock_now[0]
        made.append(budget)
        return budget

    monkeypatch.setattr(channel_stream, "FetchBudget", _recording_budget)

    async def _history(**kwargs):
        clock_now[0] += 55.0            # the walk burns 55 of the 60 seconds
        return {"ok": True, "messages": [raw("1700000100.000000")]}

    client.app.client.conversations_history = AsyncMock(side_effect=_history)
    client.app.client.conversations_replies.return_value = {
        "ok": True, "messages": [raw(ORIGIN, text="root")]}

    await _stream(client=client, db=_db(), team_id=TEAM, channel_id=CH, h=H,
                  origin_root_ts=ORIGIN, trigger_ts=ORIGIN)

    assert len(made) == 3, "the builder owns all three budgets; production injects none"
    deadlines = {b.deadline_at for b in made}
    assert len(deadlines) == 1 and None not in deadlines
    for budget in made:
        assert abs(budget.remaining_seconds() - 5.0) < 0.001

    # And the pin REFUSES a component built against a different clock — a wiring assertion, not
    # a timeout: nothing later would name the defect.
    from message_processor.channel_stream import build_origin_pin
    monkeypatch.setattr(channel_stream, "FetchBudget", real_budget)
    shared = await _shared_for(client)
    mismatched = OriginFetch(origin_root_ts=ORIGIN, messages=(), pages=0,
                             empty_fallback=False, deadline_at=shared.deadline_at + 1)
    with pytest.raises(ChannelStreamError):
        await build_origin_pin(shared, mismatched, db=_db())


async def _shared_for(client):
    from message_processor.channel_stream import build_channel_pin, prepare_channel_turn
    prepared = await prepare_channel_turn(client=client, db=_db(), team_id=TEAM,
                                          channel_id=CH, h=H)
    return await build_channel_pin(prepared, client=client, db=_db(), deadline_at=1e12)


async def test_an_incomplete_reply_fetch_for_a_selected_root_fails_closed(client):
    """T41. §2e item 4: a root the selection depends on whose replies could not be read
    completely fails the turn. A partial periphery is never presented as the newest N."""
    client.app.client.conversations_history = _walk([[raw("1700000100.000000", reply_count=2,
                                                          latest_reply="1700000200.000000")]])

    async def _broken(**kwargs):
        return {"ok": True, "messages": [raw("1700000200.000000")], "has_more": True}

    client.app.client.conversations_replies = AsyncMock(side_effect=_broken)
    with pytest.raises(HistoryFetchError):
        await _stream(client=client, db=_db(), team_id=TEAM, channel_id=CH, h=H)


async def test_a_floor_newer_than_h_makes_no_history_call(client):
    """T69. With `F > H` there is no window to fetch: zero history calls, an empty periphery, and
    the horizon renders the ORDINARY `from <F>` grammar with zero message items — NOT the
    zero-floor variant, which is reserved for a channel with no eligible events at all."""
    db = _db(anchor={"floor_ts": "1700009000.000000", "selection_version": 1})
    stream = await _stream(client=client, db=db, team_id=TEAM, channel_id=CH,
                           h="1700000000.000000")

    client.app.client.conversations_history.assert_not_called()
    assert stream.message_count == 0
    assert stream.periphery_floor_ts == "1700009000.000000"
    assert stream.horizon_item.content.startswith(
        "[STREAM HORIZON: the recent activity in this channel, from 1700009000.000000")


async def test_a_dirty_root_is_cleared_after_a_complete_fetch(client):
    """T73. Compare-and-clear against the PINNED event ts. Without it every dirty root stays
    dirty forever — a dirty row is exempt from the floor, so an uncleared root is re-fetched on
    every turn for the life of the channel."""
    client.app.client.conversations_history = _walk([[raw("1700000400.000000")]])
    client.app.client.conversations_replies.return_value = {"ok": True, "messages": []}

    db = _db(activity_roots={"1700000100.000000": "1700000500.000000"})
    await _stream(client=client, db=db, team_id=TEAM, channel_id=CH, h=H)
    db.clear_thread_dirty_async.assert_awaited_once_with(
        TEAM, CH, "1700000100.000000", if_event_ts_equals="1700000500.000000")

    # UNDER probe=True nothing is cleared at all.
    from message_processor.channel_stream import build_channel_pin, prepare_channel_turn
    probe_db = _db(activity_roots={"1700000100.000000": "1700000500.000000"})
    prepared = await prepare_channel_turn(client=client, db=probe_db, team_id=TEAM,
                                          channel_id=CH, h=H)
    await build_channel_pin(prepared, client=client, db=probe_db, deadline_at=1e12, probe=True)
    probe_db.clear_thread_dirty_async.assert_not_awaited()


async def test_the_actor_tail_is_reconciled_once_and_loses_to_a_live_event(client,
                                                                          monkeypatch):
    """T76. Reconcile runs EXACTLY ONCE per build, BEFORE the builder returns, over the FINAL
    periphery with self messages filtered — and it LOSES to a live event.

    The generation is captured BEFORE the fetch, so a live event landing mid-build moves it and
    the guard refuses to overwrite what the live feed already recorded. Capturing it afterwards
    would compare the value against itself and the guard could never fire."""
    calls = []
    real = actor_tail_module.reconcile_window
    monkeypatch.setattr(actor_tail_module, "reconcile_window",
                        lambda *a, **k: (calls.append((a, k)), real(*a, **k))[1])

    client.app.client.conversations_history = _walk([[raw("1700000100.000000")]])
    await _stream(client=client, db=_db(), team_id=TEAM, channel_id=CH, h=H)

    assert len(calls) == 1, "exactly one reconcile per build"
    window = calls[0][1]["window"]
    assert window[2] == H and window[1] is True
    assert all(r.sender_type != "self" for r in calls[0][0][1] if hasattr(r, "sender_type"))


# ============================ the AMENDED 7a: discovery always runs (T29, T37, T70)

async def test_a_reply_under_a_pre_floor_root_is_discovered(client):
    """T37. THE CASE THE INDEX EXISTS FOR, and the one the original 7a lost.

    `conversations.history` returns TOP-LEVEL messages only, so a reply posted today under a
    six-month-old root walks back ZERO history events. Under the amended rule discovery still
    runs — windowed on the inventory floor, because there is no fetched event to floor on — and
    the reply is fetched from the index. Seeding it from the history walk would not count:
    history structurally cannot surface it.
    """
    pre_floor_root = "1600000000.000000"
    client.app.client.conversations_history = _walk([[]])          # ZERO top-level events
    client.app.client.conversations_replies.return_value = {
        "ok": True, "messages": [raw(pre_floor_root, text="ancient root"),
                                 raw("1700000500.000000", text="today's reply",
                                     root=pre_floor_root)]}

    db = _db(activity_roots={pre_floor_root: "1700000500.000000"})
    stream = await _stream(client=client, db=db, team_id=TEAM, channel_id=CH, h=H)

    db.read_channel_discovery_roots_async.assert_awaited()
    assert "1700000500.000000" in [m.ts for m in stream.pinned.fetch_snapshot], (
        "a zero-event walk must still reach the index")


async def test_zero_fetched_events_skips_discovery_entirely(client):
    """T70. The shortcut needs BOTH conditions — zero history events AND zero discovery
    candidates. Keying on the empty walk alone skipped the read and lost T37's reply.

    Both stored-floor cases: with a floor at the current SELECTION_VERSION it is PRESERVED and no
    anchor row is written; with none, the floor is the sentinel."""
    client.app.client.conversations_history = _walk([[]])
    db = _db(anchor={"floor_ts": "1700000000.000000", "selection_version": 1})
    stream = await _stream(client=client, db=db, team_id=TEAM, channel_id=CH, h=H)

    # Discovery IS issued; it simply comes back empty, so the fan-out is not entered.
    db.read_channel_discovery_roots_async.assert_awaited()
    client.app.client.conversations_replies.assert_not_called()
    assert stream.periphery_floor_ts == "1700000000.000000", "a stored floor is PRESERVED"
    db.advance_channel_window_anchor_async.assert_not_awaited()

    # With NO stored floor the sentinel is what a genuinely empty channel selects.
    empty = _db(anchor=None, coverage=None)
    bare = await _stream(client=client, db=empty, team_id=TEAM, channel_id=CH, h=H)
    assert bare.periphery_floor_ts == ""


async def test_an_empty_channel_builds_and_answers(client):
    """T29. Zero eligible events end to end: the sentinel floor, NO anchor row written, the
    horizon's zero-roots variant, the end marker still present, and the turn answers from its
    origin alone. `parse_ts` is never called on the sentinel."""
    client.app.client.conversations_history = _walk([[]])
    client.app.client.conversations_replies.return_value = {
        "ok": True, "messages": [raw(ORIGIN, text="the only conversation")]}

    db = _db(anchor=None, coverage=None)
    stream = await _stream(client=client, db=db, team_id=TEAM, channel_id=CH, h=H,
                           origin_root_ts=ORIGIN, trigger_ts=ORIGIN)

    assert stream.periphery_floor_ts == ""
    assert stream.message_items == () and stream.message_count == 0
    db.advance_channel_window_anchor_async.assert_not_awaited()
    assert stream.horizon_item.content.startswith(
        "[STREAM HORIZON: no recent messages in this channel")
    assert stream.items[-1] is stream.end_marker_item
    # It still answers — from the origin thread alone.
    assert [i.metadata["ts"] for i in stream.origin_items] == [ORIGIN]


async def test_injected_budgets_must_share_one_absolute_deadline(client):
    """#7. The one-deadline rule is only a rule if the SEAM enforces it.

    Three budgets built with `total_seconds=60` each record their own start instant, so a turn
    could spend 180 seconds while every budget reported itself within bounds. That is the defect
    `deadline_at` exists to remove, and accepting an all-None set would readmit it through the
    test seam — the injection would look validated while the components ran on three clocks.
    """
    shared = time.monotonic() + 60

    # THE LEGACY FORM IS REFUSED, even though all three "agree" at None.
    with pytest.raises(ChannelStreamError, match="total_seconds form"):
        await _stream(client=client, db=_db(), team_id=TEAM, channel_id=CH, h=H,
                      history_budget=FetchBudget(total_seconds=60),
                      reply_budget=FetchBudget(total_seconds=60, page_ceiling=None),
                      origin_budget=FetchBudget(total_seconds=60, page_ceiling=None))

    # TWO DIFFERENT absolute deadlines are refused too.
    with pytest.raises(ChannelStreamError, match="SAME absolute deadline_at"):
        await _stream(client=client, db=_db(), team_id=TEAM, channel_id=CH, h=H,
                      history_budget=FetchBudget(deadline_at=shared),
                      reply_budget=FetchBudget(deadline_at=shared + 5, page_ceiling=None))

    # ONE shared absolute deadline is the accepted shape.
    stream = await _stream(client=client, db=_db(), team_id=TEAM, channel_id=CH, h=H,
                           history_budget=FetchBudget(deadline_at=shared),
                           reply_budget=FetchBudget(deadline_at=shared, page_ceiling=None),
                           origin_budget=FetchBudget(deadline_at=shared, page_ceiling=None))
    assert stream is not None

    # AND THE DIRECT PHASE SEAM IS VALIDATED TOO — `build_channel_pin` is callable on its own
    # (the probe does exactly that), so a budget disagreeing with the deadline it was handed is
    # a component running on a clock nobody else shares.
    prepared = await channel_stream.prepare_channel_turn(
        client=client, db=_db(), team_id=TEAM, channel_id=CH, h=H, frontier=0)
    with pytest.raises(ChannelStreamError, match="ONE absolute clock"):
        await channel_stream.build_channel_pin(
            prepared, client=client, db=_db(), deadline_at=shared,
            history_budget=FetchBudget(deadline_at=shared + 99))

    # A MISSING deadline is refused at both direct seams, not just a mismatched one. The
    # composer can no longer produce this shape, but the phases are callable on their own — the
    # probe calls them — and `deadline_at=None` would let each build its own `total_seconds`
    # window, which is the three-clocks defect arriving through the back door.
    with pytest.raises(ChannelStreamError, match="requires an absolute deadline_at"):
        await channel_stream.build_channel_pin(
            prepared, client=client, db=_db(), deadline_at=None)

    # And at the origin seam, where ABSENT ON BOTH SIDES would otherwise pass an equality test:
    # `None == None` is two independent clocks, not one shared one.
    unclocked = await channel_stream.build_channel_pin(
        prepared, client=client, db=_db(), deadline_at=shared)
    unclocked = replace(unclocked, deadline_at=None)
    with pytest.raises(ChannelStreamError, match="must carry an absolute deadline"):
        await channel_stream.build_origin_pin(
            unclocked,
            OriginFetch(origin_root_ts=None, messages=(), pages=0, empty_fallback=False,
                        deadline_at=None),
            db=_db())


async def test_an_empty_origin_fallback_reports_the_pages_it_spent(client):
    """#6. Both legal empty-origin fallbacks report `budget.pages_used`, never a flat zero.

    Slack was called and the attempt was paid for. Reporting 0 makes a fallback turn look free,
    and a channel taking this path on every turn would show a page bill of nothing at all —
    which is exactly the signal an operator would need to notice it.
    """
    # SHAPE ONE: `ok` with zero messages, on a top-level trigger.
    client.app.client.conversations_replies = AsyncMock(
        return_value={"ok": True, "messages": []})
    budget = FetchBudget(deadline_at=time.monotonic() + 60, page_ceiling=None)
    fetch = await fetch_origin_thread(client, CH, "1700000100.000000", H, budget,
                                      "1700000100.000000")
    assert fetch.empty_fallback is True
    assert fetch.pages == budget.pages_used == 1

    # SHAPE TWO: `thread_not_found`, the documented code, on a top-level trigger.
    client.app.client.conversations_replies = AsyncMock(
        side_effect=SlackApiError("thread_not_found",
                                  {"ok": False, "error": "thread_not_found"}))
    budget_two = FetchBudget(deadline_at=time.monotonic() + 60, page_ceiling=None)
    fetch_two = await fetch_origin_thread(client, CH, "1700000100.000000", H, budget_two,
                                          "1700000100.000000")
    assert fetch_two.empty_fallback is True
    assert fetch_two.pages == budget_two.pages_used


async def test_a_malformed_persisted_floor_fails_closed_by_name(client):
    """#5. A stored anchor floor that does not parse is a malformed record OF OURS.

    Unchecked it reached a raw `parse_ts` deep in selection and raised the normalizer's
    `TimestampError`, which is not a `ChannelStreamError` — so the turn fell through to the
    generic handler, told the user "something went wrong", and recorded no code at all. Checked
    at READ 1 it fails closed by NAME, which is what `stream_data_invalid` is for.
    """
    db = _db(anchor={"floor_ts": "not-a-timestamp", "selection_version": 1})
    with pytest.raises(StreamTimestampError, match="persisted window anchor floor"):
        await _stream(client=client, db=db, team_id=TEAM, channel_id=CH, h=H)

    # And the turn path maps that type to the honest code, not to the generic notice.
    from message_processor.base import MessageProcessor
    code, notice = MessageProcessor._channel_stream_failure(
        StreamTimestampError("persisted window anchor floor: bad"))
    assert code == "stream_data_invalid"
    assert "timestamp" in notice["status"].lower()


async def test_the_resolver_bounds_survive_an_internal_type_error(client):
    """#13. Resolver capability is read from the SIGNATURE, never inferred from a `TypeError`.

    Catching TypeError to mean "older signature" also catches one raised INSIDE a real resolver.
    The retry then ran the work a second time AND dropped `max_remote_lookups`, so the origin
    phase could spend the full default 25 instead of its remainder — an overspend with no symptom
    that would ever name its cause.
    """
    calls = []

    class _Exploding:
        bot_handle = "chatgpt-dev"
        app = None

        async def resolve_usernames(self, ids, api_client, max_remote_lookups=25, stats=None):
            calls.append(max_remote_lookups)
            raise TypeError("something inside the resolver, not a signature mismatch")

    names = await _build_actor_map(_Exploding(), [
        _norm(client, raw("100.1", user="U1", text="hi"))], max_remote_lookups=3)
    # ONE attempt, and the failure is swallowed into an empty map as before.
    assert calls == [3], "an internal TypeError must not trigger a second, unbounded attempt"
    assert names == ()

    # A LEGACY resolver — one that genuinely lacks the newer keywords — is still called, and is
    # called WITHOUT them rather than being handed arguments it cannot take.
    legacy_calls = []

    class _Legacy:
        bot_handle = "chatgpt-dev"
        app = None

        async def resolve_usernames(self, ids, api_client):
            legacy_calls.append(list(ids))
            return {uid: f"legacy-{uid}" for uid in ids}

    legacy_names = await _build_actor_map(_Legacy(), [
        _norm(client, raw("100.2", user="U7", text="hi"))], max_remote_lookups=3)
    assert legacy_calls == [["U7"]]
    assert dict(legacy_names)["U7"] == "legacy-U7"


async def test_a_dead_unseen_root_is_dropped_and_its_activity_row_cleaned_up(client):
    """T128 (W2-LIVE-2). A root the index remembers but Slack has DELETED must not be able to
    kill the channel forever.

    This is the state that took #dev-ops down: a dirty activity row for a root whose message is
    gone, `conversations.replies` answering `thread_not_found`, and failure condition 4 refusing
    the turn — on every turn, permanently, because the row is exempt from the floor and re-fires.
    A fail-closed turn is loud-and-harmless ONCE, not as a way of life.

    The narrow ruling is what this pins: DROP only when the root is UNSEEN — named by the DB
    alone, with nothing in the fetched pages vouching for it. A SEEN root answering the same code
    is live evidence contradicting the API, and dropping it would hide a broadcast the room can
    still see.
    """
    dead_root = "1700000100.000000"
    pinned_ts = "1700000150.000000"

    def _replies_thread_not_found(**kwargs):
        raise SlackApiError("thread_not_found", {"ok": False, "error": "thread_not_found"})

    class _CleanupDB:
        """The discovery reads, plus a recording compare-and-delete."""

        def __init__(self, stored_ts=pinned_ts):
            self.inner = _db(activity_roots={dead_root: pinned_ts})
            self.deletes = []
            self.stored_ts = stored_ts

        def __getattr__(self, name):
            return getattr(self.inner, name)

        async def delete_thread_activity_if_unchanged_async(self, team_id, channel_id, root_ts,
                                                            if_event_ts_equals):
            self.deletes.append((root_ts, if_event_ts_equals))
            # The real accessor's compare, reproduced: the row goes only if nothing moved.
            return if_event_ts_equals == self.stored_ts

    # The window itself holds an ordinary root, so the build has something to render without the
    # dead one — the point is that it SUCCEEDS, not merely that it stops raising.
    client.app.client.conversations_history.return_value = {
        "ok": True, "messages": [raw("1700000400.000000", text="a live message")]}
    client.app.client.conversations_replies = AsyncMock(side_effect=_replies_thread_not_found)

    # CASE 1 — UNSEEN root, pinned ts UNCHANGED: the build succeeds without it, and the row is
    # compare-and-deleted against the ts READ 1 pinned.
    db = _CleanupDB()
    stream = await _stream(client=client, db=db, team_id=TEAM, channel_id=CH, h=H)
    assert stream.message_count == 1, "the build must succeed on what remains"
    assert dead_root not in {m.ts for m in stream.pinned.fetch_snapshot}, (
        "a dropped root contributes no candidate")
    assert db.deletes == [(dead_root, pinned_ts)]

    # CASE 2 — the pinned ts MOVED since READ 1: the compare refuses and the row is retained for
    # the next turn to reassess. The build still succeeds.
    moved = _CleanupDB(stored_ts="1700000999.000000")
    stream_moved = await _stream(client=client, db=moved, team_id=TEAM, channel_id=CH, h=H)
    assert stream_moved.message_count == 1
    assert moved.deletes == [(dead_root, pinned_ts)]
    assert moved.deletes[0][1] != moved.stored_ts, "the row survives a moved pin"

    # CASE 3 — THE PROBE DELETES NOTHING. Zero durable writes still stands.
    probe_db = _CleanupDB()
    prepared = await channel_stream.prepare_channel_turn(
        client=client, db=probe_db, team_id=TEAM, channel_id=CH, h=H, frontier=0)
    await channel_stream.build_channel_pin(
        prepared, client=client, db=probe_db, probe=True,
        deadline_at=time.monotonic() + 60)
    assert probe_db.deletes == []

    # CASE 4 — A SEEN root answering the SAME code FAILS CLOSED. Here the history pages carry a
    # broadcast naming the root, so something visible vouches for it and a drop would hide it.
    client.app.client.conversations_history.return_value = {
        "ok": True, "messages": [raw("1700000500.000000", root=dead_root,
                                     subtype="thread_broadcast")]}
    with pytest.raises(HistoryFetchError):
        await _stream(client=client, db=_CleanupDB(), team_id=TEAM, channel_id=CH, h=H)

    # CASE 5 — ANY OTHER CODE fails closed even on an unseen root. The carve-out is one code.
    client.app.client.conversations_history.return_value = {
        "ok": True, "messages": [raw("1700000400.000000", text="a live message")]}
    client.app.client.conversations_replies = AsyncMock(
        side_effect=SlackApiError("channel_not_found",
                                  {"ok": False, "error": "channel_not_found"}))
    with pytest.raises(HistoryFetchError):
        await _stream(client=client, db=_CleanupDB(), team_id=TEAM, channel_id=CH, h=H)

    # CASE 6 — ONLY TWO SHAPES MAY SPEAK THE TAXONOMY. The drop turns on a Slack code, so the
    # code must come from something that genuinely carries one: a `HistoryPageError` the pager
    # already parsed, or a raw `SlackApiError`. Reading `.code` off ANY exception would let an
    # unrelated failure — a wrapper, a library error, one of our own future exceptions —
    # impersonate Slack and drop a live root.
    #
    # Asserted on the extraction itself, because the pager wraps a foreign exception into
    # `HistoryFetchError` before the fan-out ever sees it: driving it end to end would prove the
    # WRAPPER fails closed and say nothing about whose `.code` was trusted.
    class _ImpostorError(RuntimeError):
        code = "thread_not_found"

    assert channel_stream._origin_error_code(_ImpostorError("not Slack")) is None
    assert channel_stream._origin_error_code(
        HistoryPageError("refused", code="thread_not_found")) == "thread_not_found"
    assert channel_stream._origin_error_code(
        SlackApiError("thread_not_found",
                      {"ok": False, "error": "thread_not_found"})) == "thread_not_found"

    # The END-TO-END `HistoryPageError` path is already what cases 1-3 drive: `fetch_page` turns
    # a Slack refusal into `HistoryPageError(code=…)` itself (history_fetch.py:278), so injecting
    # `SlackApiError` at the API boundary is what produces the real production shape. Raising
    # `HistoryPageError` from the mocked method instead would inject BELOW that conversion, hit
    # the generic retry handler, and prove nothing about the taxonomy.
