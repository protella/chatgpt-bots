"""Turn-path fetch, root discovery and the fail-closed gates around them.

The serializer is tested separately as a pure function; this file is about everything that has
to happen before it can be called — the window, the four root sources, the shared budget, and
the four ways a turn refuses to answer rather than answer from a partial view.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock

import pytest

from base_client import HistoryFetchError
from config import config
from message_processor import channel_stream
from message_processor.channel_stream import (
    CoverageNotReady,
    SnapshotUnsupportedError,
    StreamTimestampError,
    _build_actor_map,
    _root_inventory,
    build_channel_stream,
)
from slack_client import actor_tail as actor_tail_module
from slack_client import admission_watermark
from slack_client.history_fetch import FetchBudget, fetch_page, page_messages
from slack_client.normalizer import (ORIGIN_HISTORY, ORIGIN_REPLIES, TimestampError,
                                     normalize_slack_message)

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


def _db(*, coverage=("complete", "genesis"), snapshot=None, **sidecars):
    db = MagicMock()
    payload = {
        "window": (FLOOR, True),
        "coverage": ({"coverage_start_ts": FLOOR, "bootstrap_status": coverage[0],
                      "reason": coverage[1]} if coverage else None),
        "receipt_feature_epoch_ts": None, "receipts": [], "activity": [],
        "image_analyses": [], "document_extractions": [], "ambient_artifacts": [],
        "tool_usage": {}, "versions_hash": "h",
    }
    payload.update(sidecars)
    db.read_channel_sidecars_async = AsyncMock(return_value=payload)
    db.get_active_snapshot_async = AsyncMock(return_value=snapshot)
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
    stream = await build_channel_stream(client=client, db=_db(), team_id=TEAM, channel_id=CH,
                                        h=H)
    assert [m.ts for m in stream.pinned.fetch_snapshot] == [FLOOR, H]


async def test_history_is_fetched_inclusive_with_the_window_as_bounds(client):
    await build_channel_stream(client=client, db=_db(), team_id=TEAM, channel_id=CH, h=H)
    kwargs = client.app.client.conversations_history.call_args.kwargs
    assert kwargs["oldest"] == FLOOR and kwargs["latest"] == H
    assert kwargs["inclusive"] is True
    assert kwargs["limit"] == config.history_page_size


# ------------------------------------------------------------------ fail-closed gates

async def test_any_snapshot_pointer_stops_the_turn(client):
    db = _db(snapshot={"snapshot_id": "s1", "generation": 2, "boundary_ts": FLOOR})
    with pytest.raises(SnapshotUnsupportedError):
        await build_channel_stream(client=client, db=db, team_id=TEAM, channel_id=CH, h=H)
    client.app.client.conversations_history.assert_not_called()


async def test_an_invalidated_pointer_also_stops_the_turn(client):
    db = _db(snapshot={"snapshot_id": "s1", "invalidated_at": "now", "boundary_ts": FLOOR})
    with pytest.raises(SnapshotUnsupportedError):
        await build_channel_stream(client=client, db=db, team_id=TEAM, channel_id=CH, h=H)


async def test_snapshot_read_is_observable(client, monkeypatch):
    from message_processor import participation_telemetry
    emitted = []
    monkeypatch.setattr(participation_telemetry, "compaction_snapshot",
                        lambda **kw: emitted.append(kw), raising=False)
    db = _db(snapshot={"snapshot_id": "s1", "generation": 4, "boundary_ts": FLOOR})
    with pytest.raises(SnapshotUnsupportedError):
        await build_channel_stream(client=client, db=db, team_id=TEAM, channel_id=CH, h=H)
    assert emitted and emitted[0]["op"] == "read" and emitted[0]["snapshot_id"] == "s1"


@pytest.mark.parametrize("status", ["pending", "running"])
async def test_non_terminal_coverage_fails_closed(client, status):
    with pytest.raises(CoverageNotReady):
        await build_channel_stream(client=client, db=_db(coverage=(status, None)),
                                   team_id=TEAM, channel_id=CH, h=H)


async def test_unseeded_coverage_fails_closed(client):
    db = _db(coverage=None)
    db.read_channel_sidecars_async.return_value["window"] = None
    with pytest.raises(CoverageNotReady):
        await build_channel_stream(client=client, db=db, team_id=TEAM, channel_id=CH, h=H)


async def test_coverage_start_after_h_fails_closed(client):
    with pytest.raises(CoverageNotReady, match="after H"):
        await build_channel_stream(client=client, db=_db(), team_id=TEAM, channel_id=CH,
                                   h="1699000000.000000")


async def test_a_failed_index_write_inside_the_window_fails_the_turn_closed(client):
    ticket = admission_watermark.watermark.issue(CH)
    admission_watermark.watermark.complete_failed(ticket)
    with pytest.raises(HistoryFetchError, match="failed"):
        await build_channel_stream(client=client, db=_db(), team_id=TEAM, channel_id=CH, h=H,
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
    stream = await build_channel_stream(client=client, db=_db(), team_id=TEAM, channel_id=CH,
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
    assert task.result().message_count == 0


async def test_a_malformed_h_fails_closed(client):
    with pytest.raises(StreamTimestampError):
        await build_channel_stream(client=client, db=_db(), team_id=TEAM, channel_id=CH,
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
        await build_channel_stream(client=client, db=_db(), team_id=TEAM, channel_id=CH, h=H)


async def test_a_declined_subtype_is_still_skipped_not_raised(client):
    """The difference that matters: a join notice is a decision about what counts as a message,
    not a record we failed to read."""
    client.app.client.conversations_history.return_value = {
        "ok": True, "messages": [raw("1700000100.000000"),
                                 raw("1700000200.000000", subtype="channel_join")]}
    stream = await build_channel_stream(client=client, db=_db(), team_id=TEAM, channel_id=CH,
                                        h=H)
    assert [m.ts for m in stream.pinned.fetch_snapshot] == ["1700000100.000000"]


async def test_a_sidecar_row_with_an_unusable_ts_fails_the_turn_before_any_fetch(client):
    db = _db(receipts=[{"message_ts": "soon", "state": "finalized", "turn_id": "t",
                        "thread_root_ts": None}])
    with pytest.raises(StreamTimestampError):
        await build_channel_stream(client=client, db=db, team_id=TEAM, channel_id=CH, h=H)
    client.app.client.conversations_history.assert_not_called()


# ------------------------------------------------------------------ root discovery

def _norm(client, payload, origin=ORIGIN_HISTORY):
    return normalize_slack_message(client, payload, channel_id=CH, origin=origin, team_id=TEAM)


def test_root_inventory_takes_all_four_sources(client):
    from message_processor.channel_stream import ReceiptRec, _freeze_sidecars
    history = [
        _norm(client, raw("1700000100.000000", reply_count=2, latest_reply="1700000200.0")),
        _norm(client, raw("1700000300.000000", root="1700000050.000000")),
        _norm(client, raw("1700000400.000000")),
    ]
    cards = _freeze_sidecars({
        "window": (FLOOR, True),
        "coverage": {"coverage_start_ts": FLOOR, "bootstrap_status": "complete",
                     "reason": "genesis"},
        "activity": [{"root_ts": "1699000000.000000", "last_index_event_ts": "1700000500.0",
                      "dirty": 1}],
        "receipts": [{"message_ts": "1700000600.000000", "state": "finalized",
                      "turn_id": "t", "thread_root_ts": "1698000000.000000"}],
    })
    roots = _root_inventory(history, cards, H)
    assert set(roots) == {"1700000100.000000", "1700000050.000000",
                         "1699000000.000000", "1698000000.000000"}
    assert "1700000400.000000" not in roots      # no hints, no replies to fetch
    assert isinstance(cards.receipts[0], ReceiptRec)


def test_root_inventory_drops_roots_newer_than_h(client):
    from message_processor.channel_stream import _freeze_sidecars
    cards = _freeze_sidecars({"window": (FLOOR, True),
                              "activity": [{"root_ts": "1800000000.000000", "dirty": 1}]})
    assert _root_inventory([], cards, H) == []


@pytest.mark.parametrize("receipt_ts", ["1699000000.000000", "1800000000.000000"])
def test_root_inventory_ignores_a_receipt_outside_the_window(client, receipt_ts):
    """A receipt whose own message is outside the window can decide nothing about this stream, so
    its thread must not join this turn's fetch work — that is how a receipt committed after
    admission came to spend an older turn's page budget, and could fail it."""
    from message_processor.channel_stream import _freeze_sidecars
    cards = _freeze_sidecars({
        "window": (FLOOR, True),
        "receipts": [{"message_ts": receipt_ts, "state": "finalized", "turn_id": "t",
                      "thread_root_ts": "1698000000.000000"}]})
    assert _root_inventory([], cards, H) == []


async def test_a_post_h_receipt_adds_no_fetch_work_to_an_admitted_turn(client):
    """End to end: even handed the row, the build schedules nothing for it. A replies fetch here
    would spend the shared page budget of a turn whose stream cannot contain the message, and a
    failure fetching it would fail that turn."""
    db = _db(receipts=[{"message_ts": "1800000000.000000", "state": "finalized",
                        "turn_id": "later", "thread_root_ts": "1700000100.000000"}])
    stream = await build_channel_stream(client=client, db=db, team_id=TEAM, channel_id=CH, h=H)
    client.app.client.conversations_replies.assert_not_called()
    assert stream.message_count == 0


def test_a_thread_ts_that_cannot_be_placed_never_reaches_the_inventory(client):
    """Two guards, one rule: discarding an unplaceable root would drop a whole thread from the
    window with nothing saying so. The normalizer refuses it first (r2-2), and the inventory still
    refuses a record handed to it any other way."""
    from message_processor.channel_stream import StreamTimestampError, _freeze_sidecars
    with pytest.raises(TimestampError):
        _norm(client, raw("1700000100.000000", root="whenever"))

    placed = _norm(client, raw("1700000100.000000"))
    unplaceable = replace(placed, thread_root_ts="whenever")
    with pytest.raises(StreamTimestampError):
        _root_inventory([unplaceable], _freeze_sidecars({"window": (FLOOR, True)}), H)


async def test_replies_are_fetched_per_root_with_the_window_and_the_root_ts(client):
    client.app.client.conversations_history.return_value = {
        "ok": True, "messages": [raw("1700000100.000000", reply_count=1,
                                     latest_reply="1700000200.000000")]}
    client.app.client.conversations_replies.return_value = {
        "ok": True, "messages": [raw("1700000100.000000", reply_count=1),
                                 raw("1700000200.000000", root="1700000100.000000")]}
    stream = await build_channel_stream(client=client, db=_db(), team_id=TEAM, channel_id=CH,
                                        h=H)
    kwargs = client.app.client.conversations_replies.call_args.kwargs
    assert kwargs["ts"] == "1700000100.000000"
    assert (kwargs["oldest"], kwargs["latest"], kwargs["inclusive"]) == (FLOOR, H, True)
    assert [m.ts for m in stream.pinned.fetch_snapshot] == ["1700000100.000000",
                                                            "1700000200.000000"]


async def test_a_pre_floor_root_is_reached_only_through_the_index(client):
    """The case the index exists for: the parent keeps its original ts, so history(oldest=floor)
    can never surface it, and its in-window replies would be invisible."""
    activity = [{"root_ts": "1699000000.000000", "last_observed_reply_ts": "1700000500.000000",
                 "last_index_event_ts": "1700000500.000000", "dirty": 0}]
    client.app.client.conversations_replies.return_value = {
        "ok": True, "messages": [raw("1699000000.000000", text="old root"),
                                 raw("1700000500.000000", text="new reply",
                                     root="1699000000.000000")]}
    stream = await build_channel_stream(client=client, db=_db(activity=activity), team_id=TEAM,
                                        channel_id=CH, h=H)
    # The root itself is outside the window and does NOT render; its reply does.
    assert [m.ts for m in stream.pinned.fetch_snapshot] == ["1700000500.000000"]


async def test_dirty_is_cleared_compare_and_clear_after_a_complete_replies_fetch(client):
    activity = [{"root_ts": "1700000100.000000", "last_index_event_ts": "1700000500.000000",
                 "dirty": 1}]
    db = _db(activity=activity)
    client.app.client.conversations_replies.return_value = {"ok": True, "messages": []}
    await build_channel_stream(client=client, db=db, team_id=TEAM, channel_id=CH, h=H)
    db.clear_thread_dirty_async.assert_awaited_once_with(
        TEAM, CH, "1700000100.000000", if_event_ts_equals="1700000500.000000")


async def test_dirty_is_not_cleared_for_a_root_the_index_never_flagged(client):
    client.app.client.conversations_history.return_value = {
        "ok": True, "messages": [raw("1700000100.000000", reply_count=1,
                                     latest_reply="1700000200.000000")]}
    db = _db()
    await build_channel_stream(client=client, db=db, team_id=TEAM, channel_id=CH, h=H)
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
    activity = [{"root_ts": "1700000100.000000", "dirty": 1},
                {"root_ts": "1700000900.000000", "dirty": 1}]
    with pytest.raises(HistoryFetchError):
        await build_channel_stream(client=client, db=_db(activity=activity), team_id=TEAM,
                                   channel_id=CH, h=H,
                                   budget=FetchBudget(page_ceiling=50, total_seconds=60))
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
    await build_channel_stream(client=client, db=_db(activity=activity), team_id=TEAM,
                               channel_id=CH, h=H,
                               budget=FetchBudget(page_ceiling=50, total_seconds=60))
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
    await build_channel_stream(client=client, db=_db(), team_id=TEAM, channel_id=CH, h=H)
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

    await build_channel_stream(client=client, db=_db(), team_id=TEAM, channel_id=CH, h=H)

    assert actor_tail_module.thread_has_other_bot(CH, thread) is True
    entries = actor_tail_module.actor_tail.entries(CH, thread)
    assert all(e.sender_type != "self" for e in entries)


async def test_a_live_mutation_mid_fetch_makes_the_reconcile_skip(client):
    async def _history(**kwargs):
        actor_tail_module.record(CH, ts="1700000900.000000", root_ts="1700000900.000000",
                                 is_bot=False, sender_type="human")
        return {"ok": True, "messages": [raw("1700000100.000000", user="B9", bot_id="B9")]}

    client.app.client.conversations_history = AsyncMock(side_effect=_history)
    await build_channel_stream(client=client, db=_db(), team_id=TEAM, channel_id=CH, h=H)
    # Live won: the fetched bot entry was NOT written, and the live entry survived.
    assert actor_tail_module.thread_has_other_bot(CH, "1700000100.000000") is False
    assert actor_tail_module.actor_tail.entries(CH, "1700000900.000000")


# ------------------------------------------------------------------ pinning + determinism

async def test_the_pinned_tuple_records_the_window_and_the_hashes(client):
    stream = await build_channel_stream(client=client, db=_db(), team_id=TEAM, channel_id=CH,
                                        h=H, capability_profile_hash="cap",
                                        tool_schema_version="v9")
    pin = stream.pinned
    assert pin.window == (FLOOR, True)
    assert pin.H == H and pin.snapshot is None
    assert pin.sidecar_versions_hash == "h"
    assert pin.capability_profile_hash == "cap" and pin.tool_schema_version == "v9"
    assert pin.serializer_config_hash and pin.actor_map_hash
    assert stream.stream_render_fields()["boundary"] == FLOOR
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
        return await build_channel_stream(client=c, db=_db(), team_id=TEAM, channel_id=CH, h=H)

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
    stream = await build_channel_stream(client=client, db=_db(), team_id=TEAM, channel_id=CH,
                                        h=H)
    snapshot = {m.ts: m for m in stream.pinned.fetch_snapshot}
    assert snapshot["1700000200.000000"].origin == ORIGIN_REPLIES
    assert snapshot["1700000200.000000"].is_broadcast is True
    assert stream.message_count == 2


async def test_the_dev_barrier_fires_after_the_sidecar_pin_and_before_any_fetch(
        client, monkeypatch):
    order = []
    db = _db()
    original = db.read_channel_sidecars_async

    async def _read(*a, **k):
        order.append("sidecars")
        return await original(*a, **k)

    db.read_channel_sidecars_async = _read

    async def _barrier(**context):
        order.append("barrier")
        return True

    monkeypatch.setattr(channel_stream.dev_barriers, "post_admission", _barrier)

    async def _history(**kwargs):
        order.append("history")
        return {"ok": True, "messages": []}

    client.app.client.conversations_history = AsyncMock(side_effect=_history)
    await build_channel_stream(client=client, db=db, team_id=TEAM, channel_id=CH, h=H)
    assert order == ["sidecars", "barrier", "history"]
