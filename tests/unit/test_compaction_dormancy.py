"""Obligation dormancy, obsolescence and cancellation intent (plan §1m).

Appendix C tests 19, 85, 91, 92, 95 (a-h), 96 (a-e), 97, 101. The through-line is that a channel
that cannot be compacted must STOP COSTING ANYTHING without the obligation being lost — and that
the backoff cannot be walked around by the front door, which is what makes "every trigger path
arbitrates through the row BEFORE creating any task" the load-bearing sentence in the section.

The traps these tests are written against, named so nobody softens one later:

- suppressing only the OBLIGATION's own recompaction achieves nothing, because paths 1-3 would
  enqueue an ordinary compaction for the same channel and bypass the backoff anyway;
- DEADLINE EXPIRY ALONE runs nothing, and BOOT HYDRATION IS NOT A TRIGGER — otherwise a
  crash-looping process revives for free, which is exactly the spin the backoff exists to prevent;
- a SAME-PROFILE supersession is not obsolescence: routing it there retires a live, current-profile
  obligation because an older attempt finished late.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest

from database import PROD_NAMESPACE, DatabaseManager
from message_processor import channel_snapshots as cs
from message_processor.channel_snapshots import ChannelSnapshotCoordinator

TEAM = "T1"
CH = "C0BKX77NU66"
NS = PROD_NAMESPACE
V2 = 2
PROFILE = "gpt-5.6-luna:400000:320000:280000"
OTHER = "gpt-5.6-luna:400000:300000:260000"
KEY = (TEAM, CH, NS)

PAST = (datetime.now() - timedelta(hours=2)).isoformat()
FUTURE = (datetime.now() + timedelta(hours=2)).isoformat()


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    db = DatabaseManager(platform="slack")
    yield db
    db.conn.close()


@pytest.fixture
def coord(temp_db, monkeypatch):
    """A coordinator whose live profile is PINNED, so a test says what changed rather than
    depending on whatever model config happens to name."""
    coordinator = ChannelSnapshotCoordinator(temp_db, retry_delay=0.0)
    coordinator._profile = PROFILE

    async def _sizing(team_id, channel_id):
        return {"model": "gpt-5.6-luna", "window": 400000, "trigger_tokens": 320000,
                "target_tokens": 280000, "sizing_profile": coordinator._profile}

    monkeypatch.setattr(coordinator, "sizing_for", _sizing)
    coordinator._accepting = True
    return coordinator


async def enqueue(db, *, profile=PROFILE, headroom=90000, snapshot_id="s1", generation=1,
                  reason="fixed_headroom_revalidation"):
    await db.merge_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, profile_key=profile,
        required_headroom=headroom, obligated_snapshot_id=snapshot_id,
        obligated_generation=generation, reason=reason)


async def make_dormant(db, *, deadline, profile=PROFILE):
    assert await db.cas_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, expect_state="active", new_state="dormant",
        dormant_profile_key=profile, next_attempt_after=deadline)


def watch(coord) -> List[Dict[str, Any]]:
    """Record every compaction the coordinator would actually START."""
    started: List[Dict[str, Any]] = []

    async def _compact(key, *, path, evidence, sizing, obligation):
        started.append({"key": key, "path": path, "obligation": obligation})
        return {"outcome": "published", "snapshot_id": "x"}

    coord._compact = _compact
    return started


async def settle(coord):
    """Drain the task map AND anything the completion hook spawned off the back of it.

    The hook runs as its own task, deliberately — it is a handoff, not part of the finishing
    task — so settling has to keep going until the map is genuinely empty.
    """
    for _ in range(20):
        tasks = [t for t in list(coord._tasks.values()) if not t.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.sleep(0.01)
        if not [t for t in coord._tasks.values() if not t.done()]:
            await asyncio.sleep(0.01)
            if not [t for t in coord._tasks.values() if not t.done()]:
                return


def checkpoint_row(*, crawl_id, profile=PROFILE, parent=None) -> Dict[str, Any]:
    return {
        "team_id": TEAM, "channel_id": CH, "namespace": NS, "crawl_id": crawl_id,
        "crawl_mode": "raw", "phase": 1, "pinned_H": "9000.0", "mutation_frontier": 0,
        "source_floor_ts": "1.0", "input_floor_ts": "1.0", "input_floor_inclusive": 1,
        "parent_snapshot_id": parent, "serializer_version": V2,
        "serializer_config_hash": "cfg", "prompt_version": "v1", "sizing_profile": profile,
        "headroom_source": "fixed", "headroom_tokens": 80000, "profile_version": "fixed:80000",
        "root_inventory": {}, "history_span_density": [], "actor_snapshot": {},
        "actor_snapshot_hash": "", "chunk_index": 0, "chunk_hashes": [], "chunk_aggregates": [],
        "chunk_summaries": {}, "frozen_renders": {}, "frozen_receipts": {}, "attempt_seq": 0,
        "attempt_tokens_in": 0, "attempt_tokens_out": 0, "attempt_cached_input_tokens": 0,
        "attempt_call_count": 0, "event_count": 0, "consecutive_discards": 0,
        "updated_at": "1.0",
    }


def outbox_row(crawl_id, *, at=1000.0, status="discarded"):
    return {"crawl_id": crawl_id, "attempt_seq": 0, "event_seq": 0, "created_ts": at,
            "body": {"event": "compaction_snapshot", "crawl_id": crawl_id, "attempt_seq": 0,
                     "event_seq": 0, "team_id": TEAM, "channel_id": CH, "namespace": NS,
                     "at": at, "op": "build", "model": "m", "tokens_in": 3, "tokens_out": 4,
                     "cached_input_tokens": 0, "call_count": 2, "status": status,
                     "reason": "no_fit"}}


# ===================================================================== 95(a) — the front door

@pytest.mark.parametrize("path", [cs.PATH_NEAR_TRIGGER, cs.PATH_OVER_BUDGET,
                                  cs.PATH_PAGE_CEILING])
async def test_every_trigger_path_enqueues_nothing_at_all_before_the_deadline(temp_db, coord,
                                                                             path):
    """Appendix C test 95(a). ALL THREE production paths arbitrate through the row.

    A test that only suppresses the obligation's own recompaction misses the bypass entirely:
    paths 1-3 would enqueue an ORDINARY compaction for the same channel and the backoff would be
    walked past by the front door.
    """
    await enqueue(temp_db)
    await make_dormant(temp_db, deadline=FUTURE)
    started = watch(coord)
    coord.trigger(team_id=TEAM, channel_id=CH, namespace=NS, path=path, headroom_tokens=1000)
    await settle(coord)
    assert started == []
    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert row["state"] == "dormant" and row["next_attempt_after"] == FUTURE


# ===================================================================== 95(b) — one revival

async def test_a_trigger_at_or_after_the_deadline_revives_exactly_one_attempt(temp_db, coord):
    await enqueue(temp_db)
    await make_dormant(temp_db, deadline=PAST)
    started = watch(coord)
    coord.trigger(team_id=TEAM, channel_id=CH, namespace=NS, path=cs.PATH_NEAR_TRIGGER)
    await settle(coord)
    assert len(started) == 1
    assert (await temp_db.load_pending_recompaction_async(TEAM, CH, NS))["state"] == "active"


async def test_concurrent_triggers_race_on_one_cas_and_the_loser_enqueues_nothing(temp_db,
                                                                                 coord):
    """Appendix C test 95(b)'s second half, asserted at the ARBITRATION rather than the task map.

    Process-local single-flight would hide this: the CAS is what makes the guarantee hold across
    processes, so the race has to be run against the CAS itself.
    """
    await enqueue(temp_db)
    await make_dormant(temp_db, deadline=PAST)
    results = await asyncio.gather(*[coord.arbitrate(KEY, profile=PROFILE) for _ in range(5)])
    assert sum(1 for r in results if r["run"]) == 1
    assert {r["reason"] for r in results if not r["run"]} == {"lost_revival"}


# ===================================================================== 95(c)/(d)/(e)

async def test_an_undominated_attempt_writes_a_fresh_deadline(temp_db, coord):
    """Appendix C test 95(c). The backoff REPEATS, so a later attempt cannot spin."""
    await enqueue(temp_db)
    crawl_id = uuid.uuid4().hex
    await temp_db.upsert_crawl_checkpoint_async(checkpoint_row(crawl_id=crawl_id))
    await coord._terminal_publish_nothing(
        KEY, outcome={"outcome": "publish_nothing", "reason": "over_trigger",
                      "outbox_rows": [outbox_row(crawl_id)]},
        sizing={"sizing_profile": PROFILE},
        obligation=await temp_db.load_pending_recompaction_async(TEAM, CH, NS))
    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert row["state"] == "dormant"
    assert row["dormant_profile_key"] == PROFILE
    assert datetime.fromisoformat(row["next_attempt_after"]) > datetime.now()


async def test_deadline_expiry_alone_runs_nothing(temp_db, coord):
    """Appendix C test 95(d). A TRIGGER IS REQUIRED.

    A channel that has gone quiet has no new content to compact and nobody waiting on it; waking
    on a timer to recompact a silent channel would spend model calls to change nothing.
    """
    await enqueue(temp_db)
    await make_dormant(temp_db, deadline=PAST)
    started = watch(coord)
    await asyncio.sleep(0.05)          # the deadline is long past and nothing is watching it
    await settle(coord)
    assert started == []
    assert (await temp_db.load_pending_recompaction_async(TEAM, CH, NS))["state"] == "dormant"


async def test_a_restart_after_expiry_leaves_the_row_dormant(temp_db, coord):
    """Appendix C test 95(e). BOOT HYDRATION IS NOT A TRIGGER — otherwise a crash-looping process
    revives for free. The dormant state is durable in `pending_recompaction`, not the crawl
    checkpoint, which is gone by then."""
    await enqueue(temp_db)
    await make_dormant(temp_db, deadline=PAST)
    started = watch(coord)
    coord._accepting = False
    await coord.start()
    try:
        await settle(coord)
        assert started == []
        assert (await temp_db.load_pending_recompaction_async(TEAM, CH, NS))["state"] == "dormant"
    finally:
        await coord.stop()


async def test_hydration_does_start_an_active_obligation(temp_db, coord):
    """The other half of hydration: an ACTIVE row means "this should be running", which is also
    the repair for the CAS-to-registration seam."""
    await enqueue(temp_db)
    started = watch(coord)
    coord._accepting = False
    await coord.start()
    try:
        await settle(coord)
        assert len(started) == 1
    finally:
        await coord.stop()


# ===================================================================== 95(f) — publish-nothing

async def test_publish_nothing_exhaustion_terminates_the_crawl(temp_db, coord):
    """Appendix C test 95(f). ONE COMMIT: dormancy + the discarded telemetry outbox row +
    checkpoint and skeleton deletion + the parent-sweep protection release.

    ASSERTED ACROSS RESTART: after reboot no checkpoint survives. An implementation leaving it
    behind resumes against the STALE pinned H — a ceiling the channel has long since moved past —
    and publishes a boundary that was already wrong when the attempt gave up.
    """
    await enqueue(temp_db)
    crawl_id = uuid.uuid4().hex
    await temp_db.upsert_crawl_checkpoint_async(checkpoint_row(crawl_id=crawl_id))
    obligation = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    await coord._terminal_publish_nothing(
        KEY, outcome={"outcome": "publish_nothing", "reason": "over_trigger",
                      "outbox_rows": [outbox_row(crawl_id)]},
        sizing={"sizing_profile": PROFILE}, obligation=obligation)

    assert await temp_db.load_crawl_checkpoint_async(TEAM, CH, NS) is None
    assert (await temp_db.load_pending_recompaction_async(TEAM, CH, NS))["state"] == "dormant"
    queued = await temp_db.read_outbox_batch_async(10)
    assert [(r["crawl_id"], r["body"]["status"]) for r in queued] == [(crawl_id, "discarded")]
    # It COMMITS the row; it does not EMIT it. Emission is the drainer's job, in a different
    # store — a terminal that emitted directly would break the ordered-delivery guarantee.
    assert queued[0]["body"]["op"] == "build"


async def test_a_compaction_stalled_channel_logs_critical_once_per_boot(temp_db, coord, caplog):
    """Appendix C test 19. A stalled channel must not spin the coordinator, and one CRITICAL per
    trigger would bury the message it needs to surface."""
    import logging
    await enqueue(temp_db)
    crawl_id = uuid.uuid4().hex
    await temp_db.upsert_crawl_checkpoint_async(checkpoint_row(crawl_id=crawl_id))
    obligation = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    with caplog.at_level(logging.CRITICAL, logger=cs.logger.name):
        for _ in range(3):
            await coord._terminal_publish_nothing(
                KEY, outcome={"outcome": "publish_nothing", "reason": "over_trigger"},
                sizing={"sizing_profile": PROFILE}, obligation=obligation)
    criticals = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
    assert len(criticals) == 1
    assert "COMPACTION-STALLED" in criticals[0].getMessage()


async def test_a_profile_change_during_publish_nothing_takes_the_obsolescence_path(temp_db,
                                                                                   coord):
    """Appendix C test 95(f)'s last clause. Without the expected-state CAS, A's terminal
    transaction would mark a freshly enqueued B obligation dormant — putting a never-attempted
    obligation straight into an hour's backoff."""
    await enqueue(temp_db, profile=OTHER, snapshot_id="sB", generation=9)
    crawl_id = uuid.uuid4().hex
    await temp_db.upsert_crawl_checkpoint_async(checkpoint_row(crawl_id=crawl_id))
    stale = {"obligated_snapshot_id": "sA", "obligated_generation": 1}
    await coord._terminal_publish_nothing(
        KEY, outcome={"outcome": "publish_nothing", "reason": "over_trigger"},
        sizing={"sizing_profile": PROFILE}, obligation=stale)
    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    # B was NOT put to sleep by A's terminal.
    assert row is None or row["state"] == "active"
    assert await temp_db.load_crawl_checkpoint_async(TEAM, CH, NS) is None


# ===================================================================== 95(f2) — supersession

async def test_a_same_profile_supersession_discards_only_the_stale_attempt(temp_db, coord):
    """Appendix C test 95(f2). Routing this to the obsolescence path retires a live,
    current-profile obligation purely because an older attempt finished late."""
    await enqueue(temp_db, snapshot_id="sNEW", generation=12)
    crawl_id = uuid.uuid4().hex
    await temp_db.upsert_crawl_checkpoint_async(checkpoint_row(crawl_id=crawl_id))
    stale = {"obligated_snapshot_id": "sOLD", "obligated_generation": 3}
    await coord._terminal_publish_nothing(
        KEY, outcome={"outcome": "publish_nothing", "reason": "over_trigger"},
        sizing={"sizing_profile": PROFILE}, obligation=stale)
    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert row is not None and row["state"] == "active"          # STAYS SCHEDULED
    assert row["requirements"] == {PROFILE: 90000}
    assert await temp_db.load_crawl_checkpoint_async(TEAM, CH, NS) is None


async def test_the_successor_starts_when_the_stale_task_vacates_with_no_further_trigger(
        temp_db, coord):
    """Appendix C test 95(f2)'s completion hook. "Stays scheduled" is not enough on its own: the
    stale task holds the single-flight slot, so the successor cannot start while it runs, and
    waiting for the next trigger could leave a ready obligation idle indefinitely.

    A test that delivers another trigger to get it going is hiding the missing hook — so no
    second trigger is delivered here.
    """
    await enqueue(temp_db, snapshot_id="sOLD", generation=3)
    started: List[tuple] = []

    async def _compact(key, *, path, evidence, sizing, obligation):
        started.append((obligation or {}).get("obligated_snapshot_id"))
        if len(started) == 1:
            # The SUCCESSOR lands while the stale attempt is still holding the slot.
            await enqueue(temp_db, snapshot_id="sNEW", generation=12)
        return {"outcome": "publish_nothing"}

    coord._compact = _compact
    coord._terminal_publish_nothing = AsyncMock(return_value=None)
    coord.trigger(team_id=TEAM, channel_id=CH, namespace=NS, path=cs.PATH_NEAR_TRIGGER)
    await settle(coord)
    # ONE trigger was delivered; the hook produced the second run, for the NEWER pair.
    assert started == ["sOLD", "sNEW"]


async def test_the_hook_does_not_rerun_the_pair_that_just_finished(temp_db, coord):
    """The other side of the handoff, and the reason it is bounded.

    Restarting the SAME obligated pair would run an attempt that just failed to satisfy it again
    with nothing changed, forever. The backoff that bounds a genuinely unsatisfiable obligation
    lives in the publish-nothing terminal, not here.
    """
    await enqueue(temp_db, snapshot_id="sOLD", generation=3)
    started = watch(coord)
    coord._terminal_publish_nothing = AsyncMock(return_value=None)
    coord.trigger(team_id=TEAM, channel_id=CH, namespace=NS, path=cs.PATH_NEAR_TRIGGER)
    await settle(coord)
    assert len(started) == 1


async def test_the_hook_starts_nothing_when_the_row_is_dormant(temp_db, coord):
    """95(f2)'s DORMANT-STATE VACATE. Its deadline is preserved and honoured exactly as §1m
    requires; a completion hook that started it would be a backoff bypass with a different
    name."""
    await enqueue(temp_db)
    await make_dormant(temp_db, deadline=FUTURE)
    started = watch(coord)
    await coord._completion_hook(KEY)
    # THE HOOK ITSELF must spawn nothing. Arbitration would refuse a dormant row anyway, but a
    # hook that leaned on that would still create a task per vacated slot on a channel the
    # backoff exists to leave alone.
    assert coord._tasks == {}
    await settle(coord)
    assert started == []
    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert row["state"] == "dormant" and row["next_attempt_after"] == FUTURE


async def test_the_hook_spawns_nothing_while_the_coordinator_is_closing(temp_db, coord):
    """95(f2)'s SHUTDOWN VACATE. The accepting flag is checked FIRST, so a task vacating during
    the drain cannot start fresh work the drain has already passed."""
    await enqueue(temp_db)
    started = watch(coord)
    coord._accepting = False
    await coord._completion_hook(KEY)
    await settle(coord)
    assert started == [] and coord._tasks == {}


# ===================================================================== 95(g) — the seam

async def test_a_trigger_recreates_an_active_row_with_no_live_task(temp_db, coord):
    """Appendix C test 95(g). The CAS commits BEFORE the task is registered, so a cancellation or
    a task-creation failure in between leaves an ACTIVE row with no live task. The repair is
    deliberately simple: anything noticing it starts it, single-flight."""
    await enqueue(temp_db)
    assert coord._tasks == {}                    # the orphan state: active row, no task
    started = watch(coord)
    coord.trigger(team_id=TEAM, channel_id=CH, namespace=NS, path=cs.PATH_NEAR_TRIGGER)
    await settle(coord)
    assert len(started) == 1


async def test_single_flight_admits_one_task_per_channel(temp_db, coord):
    await enqueue(temp_db)
    release = asyncio.Event()
    started: List[int] = []

    async def _compact(key, *, path, evidence, sizing, obligation):
        started.append(1)
        await release.wait()
        return {"outcome": "published"}

    coord._compact = _compact
    for _ in range(4):
        coord.trigger(team_id=TEAM, channel_id=CH, namespace=NS, path=cs.PATH_NEAR_TRIGGER)
    await asyncio.sleep(0.05)
    assert len(started) == 1
    release.set()
    await settle(coord)


# ===================================================================== 95(h) — fail closed

async def test_a_dormant_row_with_no_deadline_cannot_even_be_written(temp_db, coord):
    """Half of 95(h) is enforced BELOW the accessor by a CHECK constraint, so that shape is
    unreachable rather than merely fail-closed. The accessor's own validation covers what SQLite
    cannot express — a `dormant_profile_key` that is not a key in `requirements`."""
    import sqlite3
    await enqueue(temp_db)
    await make_dormant(temp_db, deadline=FUTURE)
    with pytest.raises(sqlite3.IntegrityError):
        temp_db.conn.execute("UPDATE pending_recompaction SET next_attempt_after = NULL")


@pytest.mark.parametrize("broken", ["dormant_profile_key", "requirements"])
async def test_a_malformed_row_refuses_to_arbitrate(temp_db, coord, broken):
    """Appendix C test 95(h). A test expecting it to behave as ACTIVE is asserting the bypass —
    reading a malformed row as active is the one outcome this machinery exists to prevent."""
    await enqueue(temp_db)
    await make_dormant(temp_db, deadline=FUTURE)
    if broken == "dormant_profile_key":
        temp_db.conn.execute(
            "UPDATE pending_recompaction SET dormant_profile_key = 'not-in-the-map'")
    else:
        temp_db.conn.execute("UPDATE pending_recompaction SET requirements = 'not json'")
    temp_db.conn.commit()

    started = watch(coord)
    coord.trigger(team_id=TEAM, channel_id=CH, namespace=NS, path=cs.PATH_NEAR_TRIGGER)
    await settle(coord)
    assert started == []
    decision = await coord.arbitrate(KEY, profile=PROFILE)
    assert decision["run"] is False and decision["reason"] == "malformed"


# ===================================================================== 96 — obsolescence

async def test_an_obsolete_entry_is_retired_at_boot_hydration(temp_db, coord, caplog):
    import logging
    await enqueue(temp_db, profile=OTHER)
    with caplog.at_level(logging.INFO):
        coord._accepting = False
        await coord.start()
        try:
            await settle(coord)
        finally:
            await coord.stop()
    assert await temp_db.load_pending_recompaction_async(TEAM, CH, NS) is None
    assert any(OTHER in r.getMessage() for r in caplog.records)


async def test_an_obsolete_entry_is_retired_at_task_acquisition(temp_db, coord):
    """Reconciliation point 2: checked as the task is CLAIMED, so a task cannot start against a
    profile that changed while it sat in the map."""
    await enqueue(temp_db, profile=OTHER)
    decision = await coord.arbitrate(KEY, profile=PROFILE)
    assert await temp_db.load_pending_recompaction_async(TEAM, CH, NS) is None
    assert decision["run"] is True and decision["reason"] == "obsolete_cleared"


async def test_a_live_settings_change_retires_it_with_no_restart(temp_db, coord):
    """Appendix C test 96's third point — THE CASE A HYDRATION-ONLY IMPLEMENTATION FAILS.

    P4's own settings modal changes the channel model at runtime; waiting for the next boot would
    leave an obligation keyed to a model nobody uses any more.
    """
    await enqueue(temp_db, profile=OTHER)
    retired = await coord.reconcile_profile(TEAM, CH, namespace=NS, current_profile=PROFILE)
    assert retired == [OTHER]
    assert await temp_db.load_pending_recompaction_async(TEAM, CH, NS) is None


async def test_the_modal_race_leaves_b_active_rather_than_inheriting_a_dormancy(temp_db, coord):
    """Appendix C test 96(a). The invariant holds only AFTER reconciliation, and there is a window
    before it: settings commit under B, an enqueue under B lands, and only THEN is A pruned.

    `dormant_profile_key` is what makes that decidable — without it B would sit under A's dormancy
    and A's deadline, silently suppressed before it was ever attempted.
    """
    await enqueue(temp_db, profile=OTHER, snapshot_id="sA", generation=1)
    await make_dormant(temp_db, deadline=FUTURE, profile=OTHER)
    await enqueue(temp_db, profile=PROFILE, snapshot_id="sB", generation=2)   # B lands under A's
    coord._profile = PROFILE
    await coord.reconcile_profile(TEAM, CH, namespace=NS, current_profile=PROFILE)

    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert row["requirements"] == {PROFILE: 90000}
    assert row["state"] == "active"
    assert row["next_attempt_after"] is None and row["dormant_profile_key"] is None


async def test_a_running_old_profile_task_is_cancelled_and_leaves_no_orphan(temp_db, coord):
    """Appendix C test 96(b) and 96(d) together: the running task is signalled, the CANCELLATION
    INTENT is written BEFORE the requirement goes, and nothing is left behind."""
    await enqueue(temp_db, profile=OTHER)
    crawl_id = uuid.uuid4().hex
    await temp_db.upsert_crawl_checkpoint_async(checkpoint_row(crawl_id=crawl_id, profile=OTHER))

    running = asyncio.Event()
    cancelled = asyncio.Event()

    async def _compact(key, *, path, evidence, sizing, obligation):
        running.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return {"outcome": "published"}

    coord._compact = _compact
    coord._profile = OTHER
    coord.trigger(team_id=TEAM, channel_id=CH, namespace=NS, path=cs.PATH_NEAR_TRIGGER)
    await asyncio.wait_for(running.wait(), timeout=2.0)

    coord._profile = PROFILE
    retired = await coord.reconcile_profile(TEAM, CH, namespace=NS, current_profile=PROFILE)
    await settle(coord)
    assert retired == [OTHER]
    assert cancelled.is_set()
    intents = await temp_db.all_cancellation_intents_async()
    assert [i["crawl_id"] for i in intents] == [crawl_id]
    assert i_reason(intents) == "obsolete_profile"


def i_reason(intents) -> str:
    return str(intents[0]["reason"])


async def test_the_intent_and_the_requirement_removal_commit_together(temp_db, coord):
    """Appendix C test 101's first clause. The intent is written BEFORE the requirement is gone —
    never after — so boot can never find an orphan checkpoint with no obligation to explain it."""
    await enqueue(temp_db, profile=OTHER)
    crawl_id = uuid.uuid4().hex
    await temp_db.upsert_crawl_checkpoint_async(checkpoint_row(crawl_id=crawl_id, profile=OTHER))
    await coord.reconcile_profile(TEAM, CH, namespace=NS, current_profile=PROFILE)
    assert await temp_db.get_cancellation_intent_async(TEAM, CH, NS, crawl_id) is not None
    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert row is None or OTHER not in row["requirements"]


async def test_a_repeated_reconciliation_is_idempotent_and_first_write_wins(temp_db, coord):
    """Appendix C test 101's idempotency clause: a conflicting `reason` or
    `obligated_snapshot_id` cannot overwrite the first."""
    crawl_id = uuid.uuid4().hex
    for reason, sid in (("obsolete_profile", "sA"), ("something_else", "sB")):
        await temp_db.write_cancellation_intent_async(
            {"team_id": TEAM, "channel_id": CH, "namespace": NS, "crawl_id": crawl_id,
             "obligated_snapshot_id": sid, "reason": reason, "created_ts": "1.0"})
    intent = await temp_db.get_cancellation_intent_async(TEAM, CH, NS, crawl_id)
    assert (intent["reason"], intent["obligated_snapshot_id"]) == ("obsolete_profile", "sA")
    assert len(await temp_db.all_cancellation_intents_async()) == 1


# ===================================================================== 101 — boot recovery

async def test_boot_finishes_a_discard_from_intent_plus_orphan_checkpoint(temp_db, coord):
    """Appendix C tests 96(d2) and 101(a). An intent plus an orphan checkpoint is a DETERMINISTIC
    INSTRUCTION: finish the discard. An implementation removing the requirement without persisting
    intent first leaves unattributable state."""
    crawl_id = uuid.uuid4().hex
    await temp_db.upsert_crawl_checkpoint_async(checkpoint_row(crawl_id=crawl_id))
    await temp_db.write_cancellation_intent_async(
        {"team_id": TEAM, "channel_id": CH, "namespace": NS, "crawl_id": crawl_id,
         "obligated_snapshot_id": "sA", "reason": "obsolete_profile", "created_ts": "1.0"})

    fresh = ChannelSnapshotCoordinator(temp_db, retry_delay=0.0)
    assert await fresh.recover_cancellations() == 1
    assert await temp_db.load_crawl_checkpoint_async(TEAM, CH, NS) is None
    assert await temp_db.all_cancellation_intents_async() == []


async def test_recovery_replays_safely(temp_db, coord):
    """Appendix C test 101(b). Recovery is idempotent, so a crash during it replays safely."""
    crawl_id = uuid.uuid4().hex
    await temp_db.upsert_crawl_checkpoint_async(checkpoint_row(crawl_id=crawl_id))
    await temp_db.write_cancellation_intent_async(
        {"team_id": TEAM, "channel_id": CH, "namespace": NS, "crawl_id": crawl_id,
         "obligated_snapshot_id": "sA", "reason": "obsolete_profile", "created_ts": "1.0"})
    await coord.recover_cancellations()
    assert await coord.recover_cancellations() == 0          # nothing half-applied, nothing left


async def test_live_cleanup_and_boot_recovery_call_the_same_atomic_accessor(temp_db, coord,
                                                                           monkeypatch):
    """Appendix C test 101(c), as a DIVERGENCE TEST.

    One transaction deleting both the checkpoint and the intent CANNOT crash between them, so
    intent-without-checkpoint is impossible — but only while both paths go through that one
    accessor. Two implementations is exactly how the divergence gets introduced, so this test
    monkeypatches the SHARED accessor and asserts BOTH paths stop working.
    """
    calls: List[str] = []
    original = temp_db.finish_cancellation_discard_async

    async def _shared(**kw):
        calls.append(kw["crawl_id"])
        return await original(**kw)

    monkeypatch.setattr(temp_db, "finish_cancellation_discard_async", _shared)

    live_id, boot_id = uuid.uuid4().hex, uuid.uuid4().hex
    for crawl_id in (live_id, boot_id):
        await temp_db.write_cancellation_intent_async(
            {"team_id": TEAM, "channel_id": CH, "namespace": NS, "crawl_id": crawl_id,
             "obligated_snapshot_id": "sA", "reason": "obsolete_profile", "created_ts": "1.0"})
    # The LIVE chunk-boundary path...
    await coord.finish_discard(team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=live_id)
    # ...and the BOOT path.
    await coord.recover_cancellations()
    assert set(calls) == {live_id, boot_id}


# ===================================================================== 85 / 91 / 92 / 97

async def test_the_requirement_map_takes_the_max_per_profile_key(temp_db, coord):
    """Appendix C tests 85 and 92. Generation 11 measured at 40k must NOT erase generation 10's
    still-unsatisfied 90k requirement under the same key, which a scalar headroom field would."""
    await enqueue(temp_db, headroom=90000, snapshot_id="s10", generation=10)
    await enqueue(temp_db, headroom=40000, snapshot_id="s11", generation=11)
    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert row["requirements"] == {PROFILE: 90000}


async def test_a_lower_generation_merge_moves_neither_half_of_the_pair(temp_db, coord):
    """Appendix C test 97. A row naming a snapshot that is not its claimed generation is a
    failure, so the id and the generation move only TOGETHER and only upward."""
    await enqueue(temp_db, snapshot_id="s5", generation=5)
    await enqueue(temp_db, snapshot_id="s3", generation=3)
    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert (row["obligated_snapshot_id"], row["obligated_generation"]) == ("s5", 5)
    await enqueue(temp_db, snapshot_id="s9", generation=9)
    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert (row["obligated_snapshot_id"], row["obligated_generation"]) == ("s9", 9)


async def test_entries_under_untouched_keys_survive(temp_db, coord):
    await enqueue(temp_db, profile=OTHER, headroom=50000, snapshot_id="s1", generation=1)
    await enqueue(temp_db, profile=PROFILE, headroom=10000, snapshot_id="s2", generation=2)
    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert row["requirements"] == {OTHER: 50000, PROFILE: 10000}


async def test_concurrent_enqueues_never_lose_the_larger_requirement(temp_db, coord):
    """Appendix C test 92's concurrency clause. A plain read-merge-write would let two enqueues
    interleave and lose the larger per-key requirement, which is the exact failure the
    `BEGIN IMMEDIATE` read-modify-write exists to prevent."""
    await asyncio.gather(*[
        enqueue(temp_db, headroom=headroom, snapshot_id=f"s{i}", generation=i)
        for i, headroom in enumerate([10, 90000, 200, 40000], start=1)])
    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert row["requirements"] == {PROFILE: 90000}
    stored = temp_db.conn.execute(
        "SELECT requirements FROM pending_recompaction").fetchone()[0]
    assert stored == '{"%s":90000}' % PROFILE          # canonical JSON, sorted, no whitespace
