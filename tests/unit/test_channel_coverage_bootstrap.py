"""Channel coverage bootstrap sweep (single-stream P1, spec §4).

Coverage is a promise about how far back the stream can honestly reach, so the sweep only
ever moves it to the oldest ts of a page it FULLY processed, and only while holding the
channel's claim. Everything here is about that discipline: a crash mid-page loses nothing, a
page-ceiling park keeps the claim, and the concurrency semaphore never sits idle across a sleep.
"""
import asyncio
import time
from types import SimpleNamespace

import pytest
from slack_sdk.errors import SlackApiError

from database import DatabaseManager
from slack_client.event_handlers import activity_index
from slack_client.event_handlers.activity_index import ChannelCoverageBootstrap

TEAM = "T1"
CH = "C1"
# Real Slack ts values: the sweep's depth cap is measured against wall-clock, so a synthetic
# "3000.0" would read as 1970 and trip the 90-day floor on the first page.
_NOW = time.time()
SEED = f"{_NOW:.6f}"
TS_A = f"{_NOW - 100:.6f}"
TS_A_REPLY = f"{_NOW - 50:.6f}"
TS_B = f"{_NOW - 200:.6f}"
TS_C = f"{_NOW - 300:.6f}"
TS_C_REPLY = f"{_NOW - 250:.6f}"


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    db = DatabaseManager(platform="slack")
    yield db
    db.conn.close()


@pytest.fixture
def sleeps(monkeypatch):
    """Every sleep is recorded and skipped; the loop still gets a turn."""
    real_sleep = asyncio.sleep
    recorded = []

    async def _fake(delay, *args, **kwargs):
        recorded.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _fake)
    return recorded


class _Web:
    def __init__(self, pages=None, conversation_pages=None):
        self.pages = list(pages or [])
        self.conversation_pages = list(conversation_pages or [])
        self.history_calls = []
        self.conversation_calls = []
        self.on_history = None

    async def conversations_history(self, **kwargs):
        self.history_calls.append(kwargs)
        if self.on_history is not None:
            self.on_history()
        assert self.pages, "conversations.history called more times than programmed"
        page = self.pages.pop(0)
        if isinstance(page, Exception):
            raise page
        return page

    async def users_conversations(self, **kwargs):
        self.conversation_calls.append(kwargs)
        assert self.conversation_pages, "users.conversations called too many times"
        page = self.conversation_pages.pop(0)
        if isinstance(page, Exception):
            raise page
        return page


class _Client:
    def __init__(self, db, web, team=TEAM, bot_user_id="UBOT"):
        self.db = db
        self.app = SimpleNamespace(client=web)
        self.self_team_id = team
        self.bot_user_id = bot_user_id


def _cfg(**overrides):
    values = {
        "coverage_bootstrap_days": 90,
        "history_page_size": 200,
        "history_page_ceiling": 50,
        "fetch_retry_attempts": 3,
        "coverage_sweep_concurrency": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _boot(db, web, cfg=None, client=None):
    boot = ChannelCoverageBootstrap(client or _Client(db, web), db=db, cfg=cfg or _cfg())
    boot._semaphore = asyncio.Semaphore(max(1, int(boot.config.coverage_sweep_concurrency)))
    return boot


def _page(messages, has_more=False, cursor="", is_limited=False):
    resp = {"ok": True, "messages": messages, "has_more": has_more}
    if cursor:
        resp["response_metadata"] = {"next_cursor": cursor}
    if is_limited:
        resp["is_limited"] = True
    return resp


def _parent(ts, reply_count=0, latest_reply=None, user="U1"):
    msg = {"type": "message", "ts": ts, "user": user, "text": "root"}
    if reply_count:
        msg["reply_count"] = reply_count
        msg["thread_ts"] = ts
        if latest_reply:
            msg["latest_reply"] = latest_reply
    return msg


def _rate_limited(retry_after="7"):
    response = _FakeResponse({"ok": False, "error": "ratelimited"},
                             {"Retry-After": retry_after})
    return SlackApiError("ratelimited", response)


class _FakeResponse(dict):
    def __init__(self, data, headers=None):
        super().__init__(data)
        self.headers = headers or {}


async def _coverage(db, channel=CH):
    return await db.get_channel_coverage_async(TEAM, channel)


# ------------------------------------------------------------------ the happy walk

async def test_cursor_pagination_records_hints_and_completes(temp_db, sleeps):
    web = _Web(pages=[
        _page([_parent(TS_A, reply_count=2, latest_reply=TS_A_REPLY),
               _parent(TS_B)], has_more=True, cursor="c1"),
        _page([_parent(TS_C, reply_count=5)]),
    ])
    await temp_db.seed_channel_coverage_async(TEAM, CH, SEED)
    boot = _boot(temp_db, web)

    await boot._sweep_channel(TEAM, CH)

    assert web.history_calls[0]["latest"] == SEED
    assert web.history_calls[0]["inclusive"] is False
    assert web.history_calls[0]["limit"] == 200
    assert web.history_calls[1]["cursor"] == "c1"

    rows = await temp_db.get_thread_activity_async(TEAM, CH)
    by_root = {r["root_ts"]: r for r in rows}
    assert by_root[TS_A]["last_observed_reply_ts"] == TS_A_REPLY
    assert by_root[TS_A]["advisory_reply_count"] == 2
    assert by_root[TS_A]["dirty"] == 0
    # A count with no latest_reply says replies exist without saying where.
    assert by_root[TS_C]["dirty"] == 1
    assert TS_B not in by_root

    row = await _coverage(temp_db)
    assert row["bootstrap_status"] == "complete"
    assert row["inventory_reason"] == "genesis"
    assert row["inventory_start_ts"] == TS_C
    assert (TEAM, CH) in boot._settled


async def test_restart_resumes_from_the_persisted_coverage_start(temp_db, sleeps):
    web = _Web(pages=[_page([_parent(TS_C)])])
    await temp_db.seed_channel_coverage_async(TEAM, CH, TS_A)
    boot = _boot(temp_db, web)

    await boot._sweep_channel(TEAM, CH)

    assert web.history_calls[0]["latest"] == TS_A
    assert "cursor" not in web.history_calls[0]


async def test_a_page_that_fails_midway_never_advances_coverage(temp_db, sleeps):
    web = _Web(pages=[_page([_parent(TS_A, reply_count=1,
                                     latest_reply=TS_A_REPLY),
                             _parent(TS_C, reply_count=1,
                                     latest_reply=TS_C_REPLY)], has_more=True,
                            cursor="c1")])
    await temp_db.seed_channel_coverage_async(TEAM, CH, SEED)
    boot = _boot(temp_db, web)

    calls = {"n": 0}
    original = temp_db.record_thread_activity_async

    async def _flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("write failed")
        return await original(*args, **kwargs)

    temp_db.record_thread_activity_async = _flaky
    await boot._sweep_channel(TEAM, CH)

    row = await _coverage(temp_db)
    assert row["inventory_start_ts"] == SEED
    assert row["bootstrap_status"] == "running"


# ------------------------------------------------------------------ parking & claims

async def test_page_ceiling_parks_and_resumes_with_the_claim_held(temp_db, sleeps):
    web = _Web(pages=[
        _page([_parent(TS_A, reply_count=1, latest_reply=TS_A_REPLY)],
              has_more=True, cursor="c1"),
        _page([_parent(TS_C)]),
    ])
    await temp_db.seed_channel_coverage_async(TEAM, CH, SEED)
    boot = _boot(temp_db, web, cfg=_cfg(history_page_ceiling=1))

    tokens = []
    web.on_history = lambda: tokens.append(temp_db.conn.execute(
        "SELECT sweep_token FROM channel_coverage WHERE channel_id = ?",
        (CH,)).fetchone()[0])

    await boot._sweep_channel(TEAM, CH)

    assert len(web.history_calls) == 2
    assert tokens[0] == tokens[1] and tokens[0]
    # The park is a resume tick, not a retry backoff.
    assert any(45.0 <= delay <= 75.0 for delay in sleeps)
    # Second pass restarts from the persisted ts, with no cursor to inherit.
    assert web.history_calls[1]["latest"] == TS_A
    assert "cursor" not in web.history_calls[1]
    assert (await _coverage(temp_db))["bootstrap_status"] == "complete"


async def test_a_stale_heartbeat_lets_another_worker_take_the_channel(temp_db, sleeps):
    await temp_db.seed_channel_coverage_async(TEAM, CH, SEED)
    assert await temp_db.acquire_coverage_sweep_async(TEAM, CH, "dead-worker") is True
    temp_db.conn.execute(
        "UPDATE channel_coverage SET heartbeat_ts = datetime('now', '-11 minutes') "
        "WHERE team_id = ? AND channel_id = ?", (TEAM, CH))

    web = _Web(pages=[_page([_parent(TS_C)])])
    boot = _boot(temp_db, web)
    await boot._sweep_channel(TEAM, CH)

    row = await _coverage(temp_db)
    assert row["sweep_token"] != "dead-worker"
    assert row["bootstrap_status"] == "complete"


async def test_a_parked_worker_whose_claim_was_taken_stops(temp_db, sleeps):
    await temp_db.seed_channel_coverage_async(TEAM, CH, SEED)
    boot = _boot(temp_db, _Web())
    with pytest.raises(activity_index._SweepTokenLost):
        await boot._park(TEAM, CH, "not-my-token")
    assert sleeps == []


async def test_a_lost_claim_mid_pass_abandons_the_channel(temp_db, sleeps):
    web = _Web(pages=[_page([_parent(TS_C)], has_more=True, cursor="c1")])
    await temp_db.seed_channel_coverage_async(TEAM, CH, SEED)
    boot = _boot(temp_db, web)

    async def _refused(*args, **kwargs):
        return False

    temp_db.advance_channel_coverage_async = _refused
    await boot._sweep_channel(TEAM, CH)
    assert (TEAM, CH) not in boot._settled


async def test_a_terminal_channel_is_never_reclaimed(temp_db, sleeps):
    await temp_db.seed_channel_coverage_async(TEAM, CH, SEED)
    await temp_db.acquire_coverage_sweep_async(TEAM, CH, "tok")
    await temp_db.advance_channel_coverage_async(TEAM, CH, "tok", TS_C, "complete",
                                                 "genesis")
    web = _Web()
    boot = _boot(temp_db, web)

    await boot._sweep_channel(TEAM, CH)

    assert web.history_calls == []
    assert (TEAM, CH) in boot._settled


# ------------------------------------------------------------------ terminal states

async def test_slack_retention_wall_declares_limited(temp_db, sleeps):
    web = _Web(pages=[_page([_parent(TS_C)], has_more=True, cursor="c1",
                            is_limited=True)])
    await temp_db.seed_channel_coverage_async(TEAM, CH, SEED)
    boot = _boot(temp_db, web)

    await boot._sweep_channel(TEAM, CH)

    row = await _coverage(temp_db)
    assert (row["bootstrap_status"], row["inventory_reason"]) == ("limited", "retention")
    assert row["inventory_start_ts"] == TS_C


async def test_exhausted_history_declares_complete(temp_db, sleeps):
    web = _Web(pages=[_page([_parent(TS_C)])])
    await temp_db.seed_channel_coverage_async(TEAM, CH, SEED)
    boot = _boot(temp_db, web)

    await boot._sweep_channel(TEAM, CH)

    row = await _coverage(temp_db)
    assert (row["bootstrap_status"], row["inventory_reason"]) == ("complete", "genesis")


async def test_the_configured_depth_declares_limited(temp_db, sleeps):
    old = f"{time.time() - 10 * 86400:.6f}"
    web = _Web(pages=[_page([_parent(old)], has_more=True, cursor="c1")])
    await temp_db.seed_channel_coverage_async(TEAM, CH, f"{time.time():.6f}")
    boot = _boot(temp_db, web, cfg=_cfg(coverage_bootstrap_days=1))

    await boot._sweep_channel(TEAM, CH)

    row = await _coverage(temp_db)
    assert (row["bootstrap_status"], row["inventory_reason"]) == ("limited", "depth_config")
    assert row["inventory_start_ts"] == old


async def test_an_empty_channel_completes_without_rows(temp_db, sleeps):
    web = _Web(pages=[_page([])])
    await temp_db.seed_channel_coverage_async(TEAM, CH, SEED)
    boot = _boot(temp_db, web)

    await boot._sweep_channel(TEAM, CH)

    assert await temp_db.get_thread_activity_async(TEAM, CH) == []
    row = await _coverage(temp_db)
    assert row["bootstrap_status"] == "complete"
    assert row["inventory_start_ts"] == SEED


# ------------------------------------------------------------------ failure handling

async def test_retry_after_is_honored_then_the_page_lands(temp_db, sleeps):
    web = _Web(pages=[_rate_limited("7"),
                      _page([_parent(TS_C, reply_count=1,
                                     latest_reply=TS_C_REPLY)])])
    await temp_db.seed_channel_coverage_async(TEAM, CH, SEED)
    boot = _boot(temp_db, web)

    await boot._sweep_channel(TEAM, CH)

    assert 7.0 in sleeps
    rows = await temp_db.get_thread_activity_async(TEAM, CH)
    assert rows[0]["last_observed_reply_ts"] == TS_C_REPLY
    assert (await _coverage(temp_db))["bootstrap_status"] == "complete"


async def test_an_unreachable_channel_is_settled_in_the_database_too(temp_db, sleeps):
    """A worker that stops must leave the row agreeing with it: settling in process while the
    row still says `running` is a channel nobody ever retries and every reader thinks is live."""
    error = SlackApiError("nope", _FakeResponse({"ok": False, "error": "not_in_channel"}))
    web = _Web(pages=[error])
    await temp_db.seed_channel_coverage_async(TEAM, CH, SEED)
    boot = _boot(temp_db, web)

    await boot._sweep_channel(TEAM, CH)

    assert len(web.history_calls) == 1
    assert (TEAM, CH) in boot._settled
    row = await _coverage(temp_db)
    assert (row["bootstrap_status"], row["inventory_reason"]) == ("limited", "unavailable")


async def test_an_app_level_refusal_stays_retryable(temp_db, sleeps):
    """A missing scope is fixed by reinstalling the app, not by anything about this channel —
    persisting `limited` would outlive the fix, so the row stays running and unsettled."""
    error = SlackApiError("nope", _FakeResponse({"ok": False, "error": "missing_scope"}))
    web = _Web(pages=[error])
    await temp_db.seed_channel_coverage_async(TEAM, CH, SEED)
    boot = _boot(temp_db, web)

    await boot._sweep_channel(TEAM, CH)

    assert (TEAM, CH) not in boot._settled
    assert (await _coverage(temp_db))["bootstrap_status"] == "running"


async def test_an_empty_page_promising_more_leaves_the_channel_retryable(temp_db, sleeps):
    """Slack claims has_more but hands back neither a message nor a cursor. There is nothing
    honest to persist, so the worker stops WITHOUT settling and the stale window retries."""
    web = _Web(pages=[_page([], has_more=True)])
    await temp_db.seed_channel_coverage_async(TEAM, CH, SEED)
    boot = _boot(temp_db, web)

    await boot._sweep_channel(TEAM, CH)

    assert (TEAM, CH) not in boot._settled
    row = await _coverage(temp_db)
    assert row["bootstrap_status"] == "running"
    assert row["inventory_start_ts"] == SEED


async def test_a_client_without_a_history_method_never_settles(temp_db, sleeps):
    boot = _boot(temp_db, _Web(), client=_Client(temp_db, SimpleNamespace()))
    await temp_db.seed_channel_coverage_async(TEAM, CH, SEED)

    await boot._sweep_channel(TEAM, CH)

    assert (TEAM, CH) not in boot._settled
    assert (await _coverage(temp_db))["bootstrap_status"] == "running"


async def test_the_retry_budget_parks_instead_of_spinning(temp_db, sleeps):
    web = _Web(pages=[_rate_limited("3"), _rate_limited("3"),
                      _page([_parent(TS_C)])])
    await temp_db.seed_channel_coverage_async(TEAM, CH, SEED)
    boot = _boot(temp_db, web, cfg=_cfg(fetch_retry_attempts=2))

    await boot._sweep_channel(TEAM, CH)

    # Two attempts, one Retry-After sleep between them, then a park and a fresh pass.
    assert sleeps[0] == 3.0
    assert any(45.0 <= delay <= 75.0 for delay in sleeps)
    assert (await _coverage(temp_db))["bootstrap_status"] == "complete"


async def test_the_semaphore_is_released_across_every_sleep(temp_db, monkeypatch):
    real_sleep = asyncio.sleep
    observed = []
    web = _Web(pages=[_rate_limited("2"),
                      _page([_parent(TS_A)], has_more=True, cursor="c1"),
                      _page([_parent(TS_C)])])
    await temp_db.seed_channel_coverage_async(TEAM, CH, SEED)
    boot = _boot(temp_db, web, cfg=_cfg(coverage_sweep_concurrency=1,
                                        history_page_ceiling=1))

    async def _fake(delay, *args, **kwargs):
        observed.append(boot._semaphore.locked())
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", _fake)
    await boot._sweep_channel(TEAM, CH)

    assert observed and not any(observed)


# ------------------------------------------------------------------ discovery & lifecycle

async def test_users_conversations_is_fully_paginated_and_seeds(temp_db, sleeps):
    web = _Web(conversation_pages=[
        {"ok": True, "channels": [{"id": "C100"}, {"id": "C200", "is_archived": True},
                                  {"id": "D300"}],
         "response_metadata": {"next_cursor": "next"}},
        {"ok": True, "channels": [{"id": "G400"}]},
    ])
    boot = _boot(temp_db, web)

    await boot._discover_channels(TEAM)

    assert web.conversation_calls[0]["types"] == "public_channel,private_channel,mpim"
    assert web.conversation_calls[1]["cursor"] == "next"
    assert await _coverage(temp_db, "C100") is not None
    assert await _coverage(temp_db, "G400") is not None
    # Archived channels take no new turns; a DM id is not a channel surface at all.
    assert await _coverage(temp_db, "C200") is None
    assert await _coverage(temp_db, "D300") is None


async def test_a_failed_discovery_is_retried_on_the_next_tick(temp_db):
    """Discovery running once meant a workspace that was rate-limited at boot never got its
    channels seeded at all."""
    web = _Web(conversation_pages=[RuntimeError("slack unavailable"),
                                   {"ok": True, "channels": [{"id": "C100"}]}])
    boot = _boot(temp_db, web)

    assert await boot._discover_channels(TEAM) is False
    assert await _coverage(temp_db, "C100") is None

    assert await boot._discover_channels(TEAM) is True
    assert await _coverage(temp_db, "C100") is not None


async def test_discovery_stops_at_the_configured_page_cap(temp_db):
    web = _Web(conversation_pages=[{"ok": True, "channels": [{"id": "C100"}],
                                    "response_metadata": {"next_cursor": "next"}}])
    boot = _boot(temp_db, web, cfg=_cfg(history_page_ceiling=1))

    # The cap is a deliberate ceiling, not a failure — walking the same prefix forever would
    # be worse than covering the first N pages.
    assert await boot._discover_channels(TEAM) is True
    assert len(web.conversation_calls) == 1


async def test_the_sweep_waits_for_both_halves_of_the_identity(temp_db, monkeypatch):
    monkeypatch.setattr(activity_index, "_IDENTITY_POLL_SECONDS", 0.01)
    client = _Client(temp_db, _Web(), team=None, bot_user_id=None)
    boot = _boot(temp_db, _Web(), client=client)

    waiter = asyncio.create_task(boot._await_identity())
    await asyncio.sleep(0.03)
    assert not waiter.done()

    client.bot_user_id = "UBOT"
    await asyncio.sleep(0.03)
    assert not waiter.done()

    client.self_team_id = TEAM
    assert await asyncio.wait_for(waiter, timeout=1.0) == TEAM


async def test_start_and_stop_are_clean_without_an_identity(temp_db, monkeypatch):
    monkeypatch.setattr(activity_index, "_IDENTITY_POLL_SECONDS", 0.01)
    client = _Client(temp_db, _Web(), team=None, bot_user_id=None)
    boot = ChannelCoverageBootstrap(client, db=temp_db, cfg=_cfg())

    boot.start()
    await asyncio.sleep(0.03)
    await boot.stop()

    assert boot._task is None
    assert boot._workers == {}


async def test_the_supervisor_spawns_one_worker_per_seeded_channel(temp_db):
    """Discovery seeds C100; C200 arrived lazily from a live index feed. Both get a worker,
    and a DM id that somehow reached the set never does."""
    web = _Web(conversation_pages=[{"ok": True, "channels": [{"id": "C100"}]}])
    client = _Client(temp_db, web)
    boot = ChannelCoverageBootstrap(client, db=temp_db, cfg=_cfg())
    swept = []

    async def _record(team, channel):
        swept.append((team, channel))

    boot._sweep_channel = _record
    activity_index._seeded_channels(client).update({"C200", "D300"})

    boot.start()
    for _ in range(200):
        await asyncio.sleep(0.01)
        if len(swept) >= 2:
            break
    await boot.stop()

    assert sorted(swept) == [(TEAM, "C100"), (TEAM, "C200")]


async def test_a_failed_seed_leaves_the_discovery_walk_incomplete(temp_db):
    """A quiet channel is only ever reached by discovery — nothing else puts it in the seeded
    set — so a seed that failed must not be reported as a finished walk."""
    web = _Web(conversation_pages=[{"ok": True, "channels": [{"id": "C100"}, {"id": "C200"}]},
                                   {"ok": True, "channels": [{"id": "C100"}, {"id": "C200"}]}])
    boot = _boot(temp_db, web)
    real_seed = temp_db.seed_channel_coverage_async
    failed = []

    async def _seed(team, channel, start_ts):
        if channel == "C200" and not failed:
            failed.append(channel)
            raise RuntimeError("db busy")
        return await real_seed(team, channel, start_ts)

    temp_db.seed_channel_coverage_async = _seed

    assert await boot._discover_channels(TEAM) is False
    assert await _coverage(temp_db, "C100") is not None
    assert await _coverage(temp_db, "C200") is None

    assert await boot._discover_channels(TEAM) is True
    assert await _coverage(temp_db, "C200") is not None


async def test_an_unreachable_channel_that_cannot_be_persisted_stays_unsettled(temp_db, sleeps):
    """Settling in process off an unwritten verdict is the same bug from the other side: the row
    still says `running`, so only the stale-heartbeat window can bring the channel back."""
    error = SlackApiError("nope", _FakeResponse({"ok": False, "error": "not_in_channel"}))
    web = _Web(pages=[error])
    await temp_db.seed_channel_coverage_async(TEAM, CH, SEED)
    boot = _boot(temp_db, web)

    async def _refuse(*args, **kwargs):
        raise RuntimeError("db down")

    temp_db.advance_channel_coverage_async = _refuse

    await boot._sweep_channel(TEAM, CH)

    assert (TEAM, CH) not in boot._settled


async def test_a_rejoin_reactivates_an_unavailable_channel(temp_db, sleeps):
    """not_in_channel is terminal but reversible: a re-invite has to hand the channel back to
    the sweep, which never looks at terminal rows on its own."""
    error = SlackApiError("nope", _FakeResponse({"ok": False, "error": "not_in_channel"}))
    web = _Web(pages=[error, _page([_parent(TS_A, reply_count=2, latest_reply=TS_A_REPLY)])])
    await temp_db.seed_channel_coverage_async(TEAM, CH, SEED)
    client = _Client(temp_db, web)
    boot = _boot(temp_db, web, client=client)

    await boot._sweep_channel(TEAM, CH)
    assert (TEAM, CH) in boot._settled

    assert await activity_index.reset_channel_coverage(client, CH) is True

    row = await _coverage(temp_db)
    assert (row["bootstrap_status"], row["inventory_reason"], row["sweep_token"]) == \
        ("pending", None, None)
    # The resume point is untouched, so the reclaimed walk picks up where it stopped.
    assert row["inventory_start_ts"] == SEED
    assert (TEAM, CH) not in boot._settled
    assert CH in activity_index._seeded_channels(client)

    # And the sweep really can claim it again.
    await boot._sweep_channel(TEAM, CH)
    assert (await _coverage(temp_db))["bootstrap_status"] == "complete"


async def test_a_rejoin_never_reopens_a_genuinely_finished_channel(temp_db):
    """`complete` and a real retention wall are facts about history, not about reachability."""
    client = _Client(temp_db, _Web())
    await temp_db.seed_channel_coverage_async(TEAM, CH, SEED)
    token = "tok"
    assert await temp_db.acquire_coverage_sweep_async(TEAM, CH, token)
    await temp_db.advance_channel_coverage_async(TEAM, CH, token, None, "limited", "retention")

    assert await activity_index.reset_channel_coverage(client, CH) is False

    row = await _coverage(temp_db)
    assert (row["bootstrap_status"], row["inventory_reason"]) == ("limited", "retention")


async def test_a_rejoin_seeds_a_channel_the_bot_has_never_seen(temp_db):
    client = _Client(temp_db, _Web())

    assert await activity_index.reset_channel_coverage(client, "C999") is False

    assert await _coverage(temp_db, "C999") is not None


# --------------------------------------- latest_reply without a count (P2 [r3-13])
# reply_count is the field Slack omits on a trimmed or otherwise unusual parent, and the sweep
# used to require it. A root whose only advertisement was `latest_reply` was therefore recorded
# as nothing at all, and every stream after that missed the pre-boundary thread it named.

async def test_a_latest_reply_with_no_count_is_still_persisted(temp_db, sleeps):
    parent = {"type": "message", "ts": TS_A, "user": "U1", "text": "root",
              "thread_ts": TS_A, "latest_reply": TS_A_REPLY}
    await temp_db.seed_channel_coverage_async(TEAM, CH, SEED)
    await _boot(temp_db, _Web(pages=[_page([parent])]))._sweep_channel(TEAM, CH)

    row = {r["root_ts"]: r for r in await temp_db.get_thread_activity_async(TEAM, CH)}[TS_A]
    assert row["last_observed_reply_ts"] == TS_A_REPLY
    assert row["advisory_reply_count"] is None
    assert row["dirty"] == 0


async def test_a_zero_count_with_a_latest_reply_is_still_persisted(temp_db, sleeps):
    parent = {"type": "message", "ts": TS_A, "user": "U1", "text": "root",
              "thread_ts": TS_A, "reply_count": 0, "latest_reply": TS_A_REPLY}
    await temp_db.seed_channel_coverage_async(TEAM, CH, SEED)
    await _boot(temp_db, _Web(pages=[_page([parent])]))._sweep_channel(TEAM, CH)

    row = {r["root_ts"]: r for r in await temp_db.get_thread_activity_async(TEAM, CH)}[TS_A]
    assert row["last_observed_reply_ts"] == TS_A_REPLY


async def test_a_parent_with_neither_hint_still_writes_nothing(temp_db, sleeps):
    await temp_db.seed_channel_coverage_async(TEAM, CH, SEED)
    await _boot(temp_db, _Web(pages=[_page([_parent(TS_A)])]))._sweep_channel(TEAM, CH)
    assert await temp_db.get_thread_activity_async(TEAM, CH) == []
