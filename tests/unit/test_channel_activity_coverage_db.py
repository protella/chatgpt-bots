"""Thread-activity index + coverage bootstrap state (single-stream P1, spec §4).

The index exists to surface a pre-boundary root whose replies are post-boundary —
conversations.history can never do that. So the bias throughout is toward an EXTRA fetch:
hints move forward only, counts never talk us out of a look, and a mutation stays dirty until
the reader clears it against the exact event it saw.
"""
import pytest

from database import DatabaseManager

TEAM = "T1"
CH = "C1"
ROOT = "100.000100"


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    db = DatabaseManager(platform="slack")
    yield db
    db.conn.close()


async def _row(db, root=ROOT):
    rows = await db.get_thread_activity_async(TEAM, CH)
    return next((r for r in rows if r["root_ts"] == root), None)


def _age_heartbeat(db, minutes, channel_id=CH):
    db.conn.execute(
        "UPDATE channel_coverage SET heartbeat_ts = datetime('now', ?) "
        "WHERE team_id = ? AND channel_id = ?",
        (f"-{minutes} minutes", TEAM, channel_id))


# --------------------------------------------------------------------- activity index

async def test_first_observation_creates_the_row(temp_db):
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, reply_ts="101.0",
                                               event_ts="101.0")
    row = await _row(temp_db)
    assert row["last_observed_reply_ts"] == "101.0"
    assert row["last_index_event_ts"] == "101.0"
    assert row["dirty"] == 0


async def test_reply_ts_is_monotonic_across_out_of_order_writes(temp_db):
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, reply_ts="1000.000100")
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, reply_ts="999.999900")
    assert (await _row(temp_db))["last_observed_reply_ts"] == "1000.000100"
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, reply_ts="1000.000200")
    assert (await _row(temp_db))["last_observed_reply_ts"] == "1000.000200"


async def test_blind_write_never_erases_a_known_ts(temp_db):
    """SQLite's scalar MAX(NULL, x) is NULL — the CASE ladder is what stops this."""
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, reply_ts="101.0",
                                               event_ts="101.0")
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, mark_dirty=True)
    row = await _row(temp_db)
    assert row["last_observed_reply_ts"] == "101.0"
    assert row["last_index_event_ts"] == "101.0"


async def test_event_ts_is_independently_monotonic(temp_db):
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, event_ts="1000.000100")
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, event_ts="999.999900")
    assert (await _row(temp_db))["last_index_event_ts"] == "1000.000100"


async def test_count_accepted_on_a_current_observation(temp_db):
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, reply_ts="101.0",
                                               reply_count=2, event_ts="101.0")
    assert (await _row(temp_db))["advisory_reply_count"] == 2
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, reply_ts="102.0",
                                               reply_count=3, event_ts="102.0")
    assert (await _row(temp_db))["advisory_reply_count"] == 3


async def test_count_from_a_stale_observation_is_ignored(temp_db):
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, reply_ts="1000.000100",
                                               reply_count=5, event_ts="1000.000100")
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, reply_ts="999.999900",
                                               reply_count=9, event_ts="999.999900")
    assert (await _row(temp_db))["advisory_reply_count"] == 5


async def test_count_never_regresses(temp_db):
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, reply_ts="101.0",
                                               reply_count=5, event_ts="101.0")
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, reply_ts="102.0",
                                               reply_count=1, event_ts="102.0")
    assert (await _row(temp_db))["advisory_reply_count"] == 5


async def test_count_at_the_same_event_ts_still_applies(temp_db):
    """The rule is `>=` — a second observation of the same event may refine the count."""
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, reply_ts="101.0",
                                               reply_count=2, event_ts="101.0")
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, reply_ts="101.0",
                                               reply_count=4, event_ts="101.0")
    assert (await _row(temp_db))["advisory_reply_count"] == 4


async def test_count_without_a_latest_reply_marks_dirty(temp_db):
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, reply_count=3)
    row = await _row(temp_db)
    assert row["advisory_reply_count"] == 3
    assert row["dirty"] == 1


async def test_zero_count_without_a_reply_does_not_mark_dirty(temp_db):
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, reply_count=0)
    assert (await _row(temp_db))["dirty"] == 0


async def test_dirty_is_sticky(temp_db):
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, reply_ts="101.0",
                                               event_ts="101.0", mark_dirty=True)
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, reply_ts="102.0",
                                               event_ts="102.0")
    assert (await _row(temp_db))["dirty"] == 1


async def test_compare_and_clear_only_clears_what_the_reader_saw(temp_db):
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, event_ts="101.0",
                                               mark_dirty=True)
    assert not await temp_db.clear_thread_dirty_async(TEAM, CH, ROOT, "100.0")
    assert (await _row(temp_db))["dirty"] == 1
    assert await temp_db.clear_thread_dirty_async(TEAM, CH, ROOT, "101.0")
    assert (await _row(temp_db))["dirty"] == 0


async def test_compare_and_clear_races_a_newer_event(temp_db):
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, event_ts="101.0",
                                               mark_dirty=True)
    # A newer mutation lands while the reader is fetching.
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, event_ts="102.0",
                                               mark_dirty=True)
    assert not await temp_db.clear_thread_dirty_async(TEAM, CH, ROOT, "101.0")
    assert (await _row(temp_db))["dirty"] == 1


async def test_compare_and_clear_against_a_null_event_ts(temp_db):
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, reply_count=2)
    assert (await _row(temp_db))["last_index_event_ts"] is None
    assert await temp_db.clear_thread_dirty_async(TEAM, CH, ROOT, None)
    assert (await _row(temp_db))["dirty"] == 0


# --------------------------------------------------------------------- since + dirty query

async def test_since_filters_on_reply_and_event_activity(temp_db):
    await temp_db.record_thread_activity_async(TEAM, CH, "100.0", reply_ts="1000.000100")
    await temp_db.record_thread_activity_async(TEAM, CH, "200.0", reply_ts="999.999900")
    await temp_db.record_thread_activity_async(TEAM, CH, "300.0", event_ts="1000.000200")

    roots = [r["root_ts"] for r in
             await temp_db.get_thread_activity_async(TEAM, CH, since_ts="999.999950")]
    assert roots == ["100.0", "300.0"]


async def test_dirty_rows_come_back_regardless_of_since(temp_db):
    await temp_db.record_thread_activity_async(TEAM, CH, "100.0", reply_ts="10.0",
                                               event_ts="10.0", mark_dirty=True)
    roots = [r["root_ts"] for r in
             await temp_db.get_thread_activity_async(TEAM, CH, since_ts="9999.0")]
    assert roots == ["100.0"]


async def test_since_none_returns_every_row_ts_ordered(temp_db):
    await temp_db.record_thread_activity_async(TEAM, CH, "1000.000100", reply_ts="1.0")
    await temp_db.record_thread_activity_async(TEAM, CH, "999.999900", reply_ts="1.0")
    roots = [r["root_ts"] for r in await temp_db.get_thread_activity_async(TEAM, CH)]
    assert roots == ["999.999900", "1000.000100"]


async def test_activity_is_scoped_to_team_and_channel(temp_db):
    await temp_db.record_thread_activity_async(TEAM, CH, ROOT, reply_ts="101.0")
    await temp_db.record_thread_activity_async(TEAM, "C2", ROOT, reply_ts="101.0")
    await temp_db.record_thread_activity_async("T2", CH, ROOT, reply_ts="101.0")
    assert len(await temp_db.get_thread_activity_async(TEAM, CH)) == 1


# --------------------------------------------------------------------- coverage

async def test_seed_is_write_once(temp_db):
    assert await temp_db.seed_channel_coverage_async(TEAM, CH, "1000.0")
    assert not await temp_db.seed_channel_coverage_async(TEAM, CH, "500.0")
    row = await temp_db.get_channel_coverage_async(TEAM, CH)
    assert row["coverage_start_ts"] == "1000.0"
    assert row["bootstrap_status"] == "pending"


async def test_acquire_claims_a_pending_channel_once(temp_db):
    await temp_db.seed_channel_coverage_async(TEAM, CH, "1000.0")
    assert await temp_db.acquire_coverage_sweep_async(TEAM, CH, "tok-a")
    assert not await temp_db.acquire_coverage_sweep_async(TEAM, CH, "tok-b")
    row = await temp_db.get_channel_coverage_async(TEAM, CH)
    assert (row["sweep_token"], row["bootstrap_status"]) == ("tok-a", "running")


async def test_stale_heartbeat_lets_another_worker_take_over(temp_db):
    await temp_db.seed_channel_coverage_async(TEAM, CH, "1000.0")
    await temp_db.acquire_coverage_sweep_async(TEAM, CH, "tok-a")
    _age_heartbeat(temp_db, 11)
    assert await temp_db.acquire_coverage_sweep_async(TEAM, CH, "tok-b")
    assert (await temp_db.get_channel_coverage_async(TEAM, CH))["sweep_token"] == "tok-b"


async def test_heartbeat_keeps_a_parked_worker_alive(temp_db):
    await temp_db.seed_channel_coverage_async(TEAM, CH, "1000.0")
    await temp_db.acquire_coverage_sweep_async(TEAM, CH, "tok-a")
    _age_heartbeat(temp_db, 11)
    assert await temp_db.heartbeat_coverage_sweep_async(TEAM, CH, "tok-a")
    assert not await temp_db.acquire_coverage_sweep_async(TEAM, CH, "tok-b")


async def test_heartbeat_is_token_guarded(temp_db):
    await temp_db.seed_channel_coverage_async(TEAM, CH, "1000.0")
    await temp_db.acquire_coverage_sweep_async(TEAM, CH, "tok-a")
    assert not await temp_db.heartbeat_coverage_sweep_async(TEAM, CH, "tok-b")


@pytest.mark.parametrize("terminal", ["complete", "limited"])
async def test_terminal_rows_are_never_reacquired(temp_db, terminal):
    await temp_db.seed_channel_coverage_async(TEAM, CH, "1000.0")
    await temp_db.acquire_coverage_sweep_async(TEAM, CH, "tok-a")
    await temp_db.advance_channel_coverage_async(TEAM, CH, "tok-a", "900.0", terminal, "reason")
    _age_heartbeat(temp_db, 60)
    assert not await temp_db.acquire_coverage_sweep_async(TEAM, CH, "tok-b")


async def test_advance_is_token_guarded(temp_db):
    await temp_db.seed_channel_coverage_async(TEAM, CH, "1000.0")
    await temp_db.acquire_coverage_sweep_async(TEAM, CH, "tok-a")
    assert not await temp_db.advance_channel_coverage_async(
        TEAM, CH, "tok-b", "900.0", "running")
    assert (await temp_db.get_channel_coverage_async(TEAM, CH))["coverage_start_ts"] == "1000.0"


async def test_coverage_start_only_moves_backward(temp_db):
    await temp_db.seed_channel_coverage_async(TEAM, CH, "1000.000100")
    await temp_db.acquire_coverage_sweep_async(TEAM, CH, "tok-a")
    assert await temp_db.advance_channel_coverage_async(
        TEAM, CH, "tok-a", "999.999900", "running")
    assert (await temp_db.get_channel_coverage_async(TEAM, CH))["coverage_start_ts"] == "999.999900"
    # A forward ts would shrink the declared horizon — refused silently, claim intact.
    assert await temp_db.advance_channel_coverage_async(
        TEAM, CH, "tok-a", "1000.000100", "running")
    assert (await temp_db.get_channel_coverage_async(TEAM, CH))["coverage_start_ts"] == "999.999900"


async def test_advance_without_a_new_start_just_sets_status(temp_db):
    await temp_db.seed_channel_coverage_async(TEAM, CH, "1000.0")
    await temp_db.acquire_coverage_sweep_async(TEAM, CH, "tok-a")
    await temp_db.advance_channel_coverage_async(TEAM, CH, "tok-a", None, "limited",
                                                 "retention")
    row = await temp_db.get_channel_coverage_async(TEAM, CH)
    assert (row["coverage_start_ts"], row["bootstrap_status"], row["coverage_reason"]) == (
        "1000.0", "limited", "retention")


@pytest.mark.parametrize("terminal,reason", [
    ("limited", "retention"),      # Slack said is_limited
    ("complete", None),            # history exhausted: genesis reached
    ("limited", "depth_config"),   # our own depth cap
])
async def test_terminal_states_stick_against_a_later_running(temp_db, terminal, reason):
    await temp_db.seed_channel_coverage_async(TEAM, CH, "1000.0")
    await temp_db.acquire_coverage_sweep_async(TEAM, CH, "tok-a")
    await temp_db.advance_channel_coverage_async(TEAM, CH, "tok-a", "900.0", terminal, reason)
    await temp_db.advance_channel_coverage_async(TEAM, CH, "tok-a", "800.0", "running", "tick")
    row = await temp_db.get_channel_coverage_async(TEAM, CH)
    assert (row["bootstrap_status"], row["coverage_reason"]) == (terminal, reason)
    # Coverage itself still extended — the horizon is honest even after the status settles.
    assert row["coverage_start_ts"] == "800.0"


async def test_ceiling_pause_keeps_the_row_running(temp_db):
    await temp_db.seed_channel_coverage_async(TEAM, CH, "1000.0")
    await temp_db.acquire_coverage_sweep_async(TEAM, CH, "tok-a")
    await temp_db.advance_channel_coverage_async(TEAM, CH, "tok-a", "900.0", "running")
    row = await temp_db.get_channel_coverage_async(TEAM, CH)
    assert (row["bootstrap_status"], row["sweep_token"]) == ("running", "tok-a")


async def test_invalid_status_rejected(temp_db):
    with pytest.raises(ValueError):
        await temp_db.advance_channel_coverage_async(TEAM, CH, "tok", "1.0", "done")


async def test_missing_coverage_row_reads_as_none(temp_db):
    assert await temp_db.get_channel_coverage_async(TEAM, "C_NONE") is None
    assert not await temp_db.acquire_coverage_sweep_async(TEAM, "C_NONE", "tok")


async def test_reset_reclaims_an_unavailable_channel(temp_db):
    """not_in_channel/is_archived are terminal but reversible, and no other accessor can demote
    a terminal row — acquire skips them and advance refuses to talk one down."""
    await temp_db.seed_channel_coverage_async(TEAM, CH, "1000.0")
    await temp_db.acquire_coverage_sweep_async(TEAM, CH, "tok-a")
    await temp_db.advance_channel_coverage_async(TEAM, CH, "tok-a", "900.0", "limited",
                                                 "unavailable")

    assert await temp_db.reset_channel_coverage_async(TEAM, CH)

    row = await temp_db.get_channel_coverage_async(TEAM, CH)
    assert (row["bootstrap_status"], row["coverage_reason"], row["sweep_token"],
            row["heartbeat_ts"]) == ("pending", None, None, None)
    # The horizon already walked is not thrown away — the reclaimed sweep resumes from it.
    assert row["coverage_start_ts"] == "900.0"
    # And a fresh worker can now claim it, which the terminal row refused.
    assert await temp_db.acquire_coverage_sweep_async(TEAM, CH, "tok-b")


@pytest.mark.parametrize("terminal,reason", [
    ("limited", "retention"),      # Slack's own retention wall
    ("complete", None),            # genesis reached
    ("limited", "depth_config"),   # our configured depth cap
])
async def test_reset_refuses_a_genuinely_finished_channel(temp_db, terminal, reason):
    """These are facts about how far history goes, not about whether we can reach the channel —
    a rejoin must never re-walk them."""
    await temp_db.seed_channel_coverage_async(TEAM, CH, "1000.0")
    await temp_db.acquire_coverage_sweep_async(TEAM, CH, "tok-a")
    await temp_db.advance_channel_coverage_async(TEAM, CH, "tok-a", "900.0", terminal, reason)

    assert not await temp_db.reset_channel_coverage_async(TEAM, CH)

    row = await temp_db.get_channel_coverage_async(TEAM, CH)
    assert (row["bootstrap_status"], row["coverage_reason"]) == (terminal, reason)
    assert row["sweep_token"] == "tok-a"


async def test_reset_leaves_a_live_sweep_alone(temp_db):
    await temp_db.seed_channel_coverage_async(TEAM, CH, "1000.0")
    await temp_db.acquire_coverage_sweep_async(TEAM, CH, "tok-a")

    assert not await temp_db.reset_channel_coverage_async(TEAM, CH)

    row = await temp_db.get_channel_coverage_async(TEAM, CH)
    assert (row["bootstrap_status"], row["sweep_token"]) == ("running", "tok-a")


async def test_reset_is_scoped_and_tolerates_a_missing_row(temp_db):
    await temp_db.seed_channel_coverage_async(TEAM, CH, "1000.0")
    await temp_db.acquire_coverage_sweep_async(TEAM, CH, "tok-a")
    await temp_db.advance_channel_coverage_async(TEAM, CH, "tok-a", None, "limited",
                                                 "unavailable")

    assert not await temp_db.reset_channel_coverage_async("T_OTHER", CH)
    assert not await temp_db.reset_channel_coverage_async(TEAM, "C_NONE")
    assert (await temp_db.get_channel_coverage_async(TEAM, CH))["bootstrap_status"] == "limited"
