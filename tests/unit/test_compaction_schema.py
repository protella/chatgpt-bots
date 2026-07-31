"""P4a compaction schema — mutation observations (§1c, interfaces §3.5).

The activity upsert and the mutation insert are ONE ticketed unit: the admission ticket
completes only when both have committed, and a replayed delivery must land exactly one
observation row. Everything here is protecting one of those two properties.
"""
import asyncio
import json
import sqlite3

import pytest

from database import DatabaseManager, PROD_NAMESPACE, canonical_body_bytes

TEAM = "T1"
CH = "C1"
NS = PROD_NAMESPACE
SV = 2
CRAWL = "crawl-1"
PROFILE = "gpt-5.6-luna:400000:320000:280000"
OTHER_PROFILE = "gpt-5.6-luna:400000:300000:260000"


def _snapshot(**kw):
    """A complete v2 candidate. Every sizing field is present — a v2 row missing one is
    rejected by the accessor, so the fixture must not model that."""
    body = kw.pop("payload_text", "summary").encode("utf-8")
    base = dict(team_id=TEAM, channel_id=CH, namespace=NS, serializer_version=SV,
                boundary_ts="1000.000100", source_floor_ts="1.000000",
                parent_snapshot_id=None, prompt_version="v1", model="gpt-5.6-luna",
                source_hash="sh", payload_bytes=body, anchor_payload_bytes=b"",
                mutation_frontier=0, headroom_source="measured", headroom_tokens=90000,
                effective_window=400000, sizing_profile=PROFILE, fit_result="under_target")
    base.update(kw)
    return base


async def _checkpoint(db, *, crawl_id=CRAWL, mutation_frontier=0, parent=None, mode="raw",
                      **kw):
    row = dict(team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=crawl_id, crawl_mode=mode,
               phase=1, pinned_H="9000.0", mutation_frontier=mutation_frontier,
               source_floor_ts="1.0", input_floor_ts="1.0", input_floor_inclusive=1,
               parent_snapshot_id=parent, serializer_version=SV,
               serializer_config_hash="sc", prompt_version="v1", sizing_profile=PROFILE,
               headroom_source="fixed", headroom_tokens=80000,
               profile_version="fixed:80000", actor_snapshot_hash="ah", chunk_index=0,
               attempt_seq=0, attempt_tokens_in=0, attempt_tokens_out=0,
               attempt_cached_input_tokens=0, attempt_call_count=0, event_count=0,
               consecutive_discards=0)
    row.update(kw)
    await db.upsert_crawl_checkpoint_async(row)
    return row


def _event(ts, *, root_ts="0", kind_rank=0, source_rank=1, actor="U1", fingerprint=None):
    return {"ts": ts, "root_ts": root_ts, "kind_rank": kind_rank, "source_rank": source_rank,
            "actor_id": actor, "projected_byte_len": 10, "base_canonical_bytes": 8,
            "projection_sha256": fingerprint or f"fp-{ts}-{source_rank}"}


def _body(op="build", *, crawl_id=CRAWL, attempt_seq=0, event_seq=None, at=None, **kw):
    """A canonical body that passes all six clauses, so a test that wants a failure has to
    break exactly one thing."""
    event_seq = (0 if op == "build" else 1) if event_seq is None else event_seq
    body = {"event": "compaction_snapshot", "crawl_id": crawl_id, "attempt_seq": attempt_seq,
            "event_seq": event_seq, "team_id": TEAM, "channel_id": CH, "namespace": NS,
            "at": 1700000000.5 if at is None else at, "op": op}
    if op == "build":
        body.update(model="gpt-5.6-luna", tokens_in=10, tokens_out=20,
                    cached_input_tokens=0, call_count=2, status="ok")
    else:
        body.update(snapshot_id="s1", generation=1, boundary_ts="1000.0",
                    fit_result="under_target", serializer_version=SV)
    body.update(kw)
    return body


def _row(body):
    return {"crawl_id": body["crawl_id"], "attempt_seq": body["attempt_seq"],
            "event_seq": body["event_seq"], "body": body}


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    db = DatabaseManager(platform="slack")
    yield db
    db.conn.close()


def _obs(root_ts="100.0", **kw):
    base = {"team_id": TEAM, "channel_id": CH, "root_ts": root_ts, "reply_ts": None,
            "event_ts": None, "mark_dirty": False}
    base.update(kw)
    return base


def _mut(subject_ts="101.0", kind="edit", identity="Ev123", observed_at="102.0"):
    return {"team_id": TEAM, "channel_id": CH, "subject_ts": subject_ts, "kind": kind,
            "observation_identity": identity, "observed_at": observed_at}


def _rows(db):
    return [dict(r) for r in db.conn.execute(
        "SELECT * FROM snapshot_mutation_observations ORDER BY id")]


async def test_both_halves_commit_together(temp_db):
    await temp_db.record_activity_and_mutation_async(
        observation=_obs(reply_ts="101.0", event_ts="101.0", mark_dirty=True),
        mutation=_mut())

    activity = await temp_db.get_thread_activity_async(TEAM, CH)
    assert len(activity) == 1
    assert activity[0]["dirty"] == 1
    assert activity[0]["last_observed_reply_ts"] == "101.0"

    rows = _rows(temp_db)
    assert len(rows) == 1
    assert (rows[0]["subject_ts"], rows[0]["kind"], rows[0]["observation_identity"]) == (
        "101.0", "edit", "Ev123")


async def test_replay_is_idempotent(temp_db):
    """Test C2: the same delivery twice leaves exactly one observation."""
    for _ in range(3):
        await temp_db.record_activity_and_mutation_async(
            observation=_obs(event_ts="101.0", mark_dirty=True), mutation=_mut())
    assert len(_rows(temp_db)) == 1


async def test_identity_separates_distinct_deliveries(temp_db):
    await temp_db.record_activity_and_mutation_async(
        observation=None, mutation=_mut(identity="Ev1"))
    await temp_db.record_activity_and_mutation_async(
        observation=None, mutation=_mut(identity="Ev2"))
    await temp_db.record_activity_and_mutation_async(
        observation=None, mutation=_mut(kind="delete", identity="Ev1"))
    assert len(_rows(temp_db)) == 3


async def test_either_half_may_be_absent(temp_db):
    await temp_db.record_activity_and_mutation_async(observation=_obs(), mutation=None)
    assert len(await temp_db.get_thread_activity_async(TEAM, CH)) == 1
    assert _rows(temp_db) == []

    await temp_db.record_activity_and_mutation_async(observation=None, mutation=_mut())
    assert len(_rows(temp_db)) == 1


async def test_both_none_is_a_legal_no_op(temp_db):
    await temp_db.record_activity_and_mutation_async(observation=None, mutation=None)
    assert _rows(temp_db) == []


async def test_activity_half_keeps_the_monotonic_merge(temp_db):
    """The shared upsert rule — a blind later write must not erase a known ts."""
    await temp_db.record_activity_and_mutation_async(
        observation=_obs(reply_ts="1000.000100", event_ts="1000.000100"), mutation=None)
    await temp_db.record_activity_and_mutation_async(
        observation=_obs(reply_ts="999.999900", event_ts="999.999900"), mutation=_mut())
    row = (await temp_db.get_thread_activity_async(TEAM, CH))[0]
    assert row["last_observed_reply_ts"] == "1000.000100"
    assert row["last_index_event_ts"] == "1000.000100"


@pytest.mark.parametrize("bad", [None, ""])
async def test_empty_identity_is_refused(temp_db, bad):
    """A NULL/empty identity would defeat the unique key — SQLite NULLs are distinct."""
    with pytest.raises(ValueError):
        await temp_db.record_activity_and_mutation_async(
            observation=None, mutation=_mut(identity=bad))
    assert _rows(temp_db) == []


async def test_unknown_kind_is_refused(temp_db):
    with pytest.raises(ValueError):
        await temp_db.record_activity_and_mutation_async(
            observation=None, mutation=_mut(kind="tombstone"))


async def test_a_refused_mutation_writes_no_activity_row(temp_db):
    """The unit fails whole: a rejected mutation must not leave its activity half behind."""
    with pytest.raises(ValueError):
        await temp_db.record_activity_and_mutation_async(
            observation=_obs(reply_ts="101.0"), mutation=_mut(kind="nope"))
    assert await temp_db.get_thread_activity_async(TEAM, CH) == []


async def test_frontier_is_zero_until_something_is_observed(temp_db):
    assert await temp_db.max_mutation_observation_id_async(TEAM, CH) == 0
    await temp_db.record_activity_and_mutation_async(observation=None, mutation=_mut())
    assert await temp_db.max_mutation_observation_id_async(TEAM, CH) == 1


async def test_frontier_is_tenant_scoped(temp_db):
    await temp_db.record_activity_and_mutation_async(observation=None, mutation=_mut())
    other = _mut()
    other["channel_id"] = "C2"
    await temp_db.record_activity_and_mutation_async(observation=None, mutation=other)

    assert await temp_db.max_mutation_observation_id_async(TEAM, CH) == 1
    assert await temp_db.max_mutation_observation_id_async(TEAM, "C2") == 2
    assert await temp_db.max_mutation_observation_id_async("T2", CH) == 0


async def test_ids_are_never_reused_after_deletion(temp_db):
    """AUTOINCREMENT, not rowid: retention deletes rows and a reused id would make a
    persisted frontier compare wrongly."""
    await temp_db.record_activity_and_mutation_async(observation=None, mutation=_mut())
    temp_db.conn.execute("DELETE FROM snapshot_mutation_observations")
    await temp_db.record_activity_and_mutation_async(
        observation=None, mutation=_mut(identity="Ev999"))
    assert _rows(temp_db)[0]["id"] == 2


# ===================================================================== §1c scope + retention

async def test_observations_after_unions_span_and_explicit_roots(temp_db):
    """Test 4 / §1d: the span predicate and the anchor-root set are OR'd, never AND'd.

    An anchor root may PREDATE `source_floor_ts`, so the span predicate cannot reach it by
    construction. ANDing the two would silently UNDER-INVALIDATE — the exact failure §1c
    exists to prevent — because a pre-floor anchor mutation would satisfy the ts set and fail
    the span, and vanish.
    """
    for ts, identity in (("50.0", "pre"), ("500.0", "in"), ("9000.0", "post")):
        await temp_db.record_activity_and_mutation_async(
            observation=None, mutation=_mut(subject_ts=ts, identity=identity))

    rows = await temp_db.mutation_observations_after_async(
        TEAM, CH, 0, floor_ts="100.0", high_ts="1000.0", subject_ts_in=["50.0"])
    assert [r["subject_ts"] for r in rows] == ["50.0", "500.0"]

    span_only = await temp_db.mutation_observations_after_async(
        TEAM, CH, 0, floor_ts="100.0", high_ts="1000.0")
    assert [r["subject_ts"] for r in span_only] == ["500.0"]

    roots_only = await temp_db.mutation_observations_after_async(
        TEAM, CH, 0, subject_ts_in=["50.0", "9000.0"])
    assert [r["subject_ts"] for r in roots_only] == ["50.0", "9000.0"]

    assert len(await temp_db.mutation_observations_after_async(TEAM, CH, 0)) == 3


async def test_observations_after_honours_the_frontier(temp_db):
    await temp_db.record_activity_and_mutation_async(observation=None, mutation=_mut())
    first = await temp_db.max_mutation_observation_id_async(TEAM, CH)
    await temp_db.record_activity_and_mutation_async(
        observation=None, mutation=_mut(subject_ts="200.0", identity="Ev2"))
    rows = await temp_db.mutation_observations_after_async(TEAM, CH, first)
    assert [r["subject_ts"] for r in rows] == ["200.0"]


async def test_retention_watermark_keeps_rows_a_live_checkpoint_still_needs(temp_db):
    """Test 3: rows below W are deleted, rows at or above are kept, with a LIVE CRAWL
    CHECKPOINT holding the watermark down."""
    for i in range(5):
        await temp_db.record_activity_and_mutation_async(
            observation=None, mutation=_mut(subject_ts=f"{100 + i}.0", identity=f"E{i}"))
    assert await temp_db.max_mutation_observation_id_async(TEAM, CH) == 5

    await _checkpoint(temp_db, mutation_frontier=3)
    assert await temp_db.sweep_mutation_observations_async(TEAM, CH) == 2
    assert [r["id"] for r in _rows(temp_db)] == [3, 4, 5]

    # With no generation, checkpoint or candidate, W is max id + 1 and everything goes.
    await temp_db.delete_crawl_state_async(TEAM, CH, NS, CRAWL)
    assert await temp_db.sweep_mutation_observations_async(TEAM, CH) == 3
    assert _rows(temp_db) == []


async def test_retention_watermark_takes_the_min_over_every_source(temp_db):
    for i in range(6):
        await temp_db.record_activity_and_mutation_async(
            observation=None, mutation=_mut(subject_ts=f"{100 + i}.0", identity=f"E{i}"))
    await _checkpoint(temp_db, mutation_frontier=5)
    # An in-flight candidate pinned an OLDER frontier; it must hold the watermark down.
    await temp_db.insert_compaction_candidate_async(snapshot=_snapshot(mutation_frontier=2))

    await temp_db.sweep_mutation_observations_async(TEAM, CH)
    assert [r["id"] for r in _rows(temp_db)] == [2, 3, 4, 5, 6]


# ===================================================================== schema shape (test 65)

SKELETON_COLUMNS = ["crawl_id", "seq", "ts", "root_ts", "kind_rank", "source_rank", "actor_id",
                    "projected_byte_len", "base_canonical_bytes", "projection_sha256"]


def test_the_event_skeleton_has_no_text_column(temp_db):
    """Test 65 — the load-bearing one, asserted on SCHEMA SHAPE rather than on string absence.

    Every field is a timestamp, an id, a rank, a byte count or a HASH, so the
    never-persist-conversation-history rule holds by shape: there is nowhere for message text
    to go. A SHA-256 fingerprint is not text and cannot be read back as text.
    """
    columns = [r["name"] for r in
               temp_db.conn.execute("PRAGMA table_info(compaction_event_skeleton)")]
    assert columns == SKELETON_COLUMNS


def test_the_skeleton_seq_index_is_partial(temp_db):
    """A partial UNIQUE (crawl_id, seq) WHERE seq IS NOT NULL — chosen over a seal-time check
    so the constraint holds CONTINUOUSLY after sealing, not only at the instant it was
    verified."""
    sql = temp_db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'idx_skeleton_seq'").fetchone()[0]
    assert "WHERE seq IS NOT NULL" in sql
    assert "UNIQUE" in sql


def test_no_compaction_table_holds_message_text(temp_db):
    """The checkpoint accessors accept only the named derived forms. `chunk_summaries` are
    permitted DERIVED summaries — a substring scan for verbatim overlap would fail a compliant
    implementation, so the assertion is on the column vocabulary instead."""
    from database import _CHECKPOINT_COLUMN_NAMES
    columns = {r["name"] for r in
               temp_db.conn.execute("PRAGMA table_info(compaction_crawl_checkpoints)")}
    assert columns == set(_CHECKPOINT_COLUMN_NAMES)
    forbidden = {"text", "body", "message_text", "raw", "page_payload", "content",
                 "projected_text", "transcript"}
    assert not (columns & forbidden)


def test_mutation_observations_use_autoincrement(temp_db):
    """AUTOINCREMENT is REQUIRED, not incidental: retention deletes rows and a reused rowid
    would make a persisted frontier compare wrongly."""
    sql = temp_db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE name = 'snapshot_mutation_observations'"
    ).fetchone()[0]
    assert "AUTOINCREMENT" in sql


# ===================================================================== crawl page atomicity

async def test_a_page_commits_its_rows_and_its_cursor_together(temp_db):
    """Test 81: neither a permanent gap (cursor ahead of rows) nor replay ambiguity (rows
    ahead of cursor) is reachable — a cursor never advances outside the transaction that
    commits its rows."""
    await _checkpoint(temp_db)
    await temp_db.commit_crawl_page_async(
        team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
        skeleton_rows=[_event("100.0"), _event("110.0")],
        checkpoint_patch={"inventory_cursor_ts": "100.0"})

    assert await temp_db.skeleton_count_async(CRAWL) == 2
    assert (await temp_db.load_crawl_checkpoint_async(
        TEAM, CH, NS))["inventory_cursor_ts"] == "100.0"


async def test_a_failed_page_advances_neither(temp_db):
    """A bad row in the page must leave the cursor where it was: an advanced cursor with
    uncommitted rows is a gap nothing later notices."""
    await _checkpoint(temp_db)
    await temp_db.commit_crawl_page_async(
        team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
        skeleton_rows=[_event("100.0")], checkpoint_patch={"inventory_cursor_ts": "100.0"})

    bad = _event("90.0")
    bad.pop("projection_sha256")
    with pytest.raises(KeyError):
        await temp_db.commit_crawl_page_async(
            team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
            skeleton_rows=[_event("95.0"), bad],
            checkpoint_patch={"inventory_cursor_ts": "90.0"})

    assert await temp_db.skeleton_count_async(CRAWL) == 1
    assert (await temp_db.load_crawl_checkpoint_async(
        TEAM, CH, NS))["inventory_cursor_ts"] == "100.0"


@pytest.mark.parametrize("history_first", [True, False])
async def test_replies_win_the_broadcast_in_either_arrival_order(temp_db, history_first):
    """Test 87: a broadcast arrives twice — the history copy with the sentinel root "0", the
    replies copy with the real root. `source_rank` is DURABLE precisely so precedence survives
    a restart, so arrival order is irrelevant."""
    await _checkpoint(temp_db)
    history = _event("200.0", root_ts="0", source_rank=1, actor="U1")
    replies = _event("200.0", root_ts="150.0", source_rank=2, actor="U2")
    order = [history, replies] if history_first else [replies, history]
    for row in order:
        await temp_db.commit_crawl_page_async(
            team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL, skeleton_rows=[row],
            checkpoint_patch={})

    await temp_db.seal_event_skeleton_async(CRAWL)
    rows = await temp_db.skeleton_slice_async(CRAWL, 0, 10)
    assert len(rows) == 1
    assert (rows[0]["root_ts"], rows[0]["source_rank"]) == ("150.0", 2)


async def test_source_rank_precedence_survives_a_restart(temp_db, tmp_path, monkeypatch):
    """Test 87, the restart case: a history walk REPLAYED after a crash must not overwrite a
    replies copy already committed. Rank is persisted for exactly this."""
    await _checkpoint(temp_db)
    await temp_db.commit_crawl_page_async(
        team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
        skeleton_rows=[_event("200.0", root_ts="150.0", source_rank=2, actor="U2")],
        checkpoint_patch={})
    temp_db.conn.close()

    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    rebooted = DatabaseManager(platform="slack")
    try:
        await rebooted.commit_crawl_page_async(
            team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
            skeleton_rows=[_event("200.0", root_ts="0", source_rank=1, actor="U1")],
            checkpoint_patch={})
        await rebooted.seal_event_skeleton_async(CRAWL)
        rows = await rebooted.skeleton_slice_async(CRAWL, 0, 10)
        assert len(rows) == 1
        assert (rows[0]["root_ts"], rows[0]["source_rank"]) == ("150.0", 2)
    finally:
        rebooted.conn.close()


# ===================================================================== sealing (test 82)

async def test_sealing_assigns_contiguous_seq_in_composite_order(temp_db):
    await _checkpoint(temp_db)
    await temp_db.commit_crawl_page_async(
        team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
        skeleton_rows=[_event("300.0"), _event("100.0"), _event("200.0"),
                       _event("200.0", kind_rank=2)],
        checkpoint_patch={})

    sealed = await temp_db.seal_event_skeleton_async(CRAWL)
    assert sealed["events"] == 4
    rows = await temp_db.skeleton_slice_async(CRAWL, 0, 10)
    assert [r["seq"] for r in rows] == [0, 1, 2, 3]
    assert [(r["ts"], r["kind_rank"]) for r in rows] == [
        ("100.0", 0), ("200.0", 0), ("200.0", 2), ("300.0", 0)]


async def test_sealing_orders_by_ts_numerically_not_lexically(temp_db):
    """"1000.5" > "1000.10" numerically and the other way round as strings."""
    await _checkpoint(temp_db)
    await temp_db.commit_crawl_page_async(
        team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
        skeleton_rows=[_event("1000.5"), _event("1000.10")], checkpoint_patch={})
    await temp_db.seal_event_skeleton_async(CRAWL)
    assert [r["ts"] for r in await temp_db.skeleton_slice_async(CRAWL, 0, 10)] == [
        "1000.10", "1000.5"]


async def test_sealing_recomputes_root_aggregates_from_sealed_rows(temp_db):
    """Test 82: walk-time aggregates saw PRE-PRECEDENCE duplicates, so only the sealed rows
    are authoritative — whatever the walks accumulated is discarded."""
    await _checkpoint(temp_db, root_inventory={
        "150.0": {"root_ts": "150.0", "done": False, "reply_count": 99,
                  "last_canonical_message_ts": "999999.0", "root_snippet_len": 12}})
    await temp_db.commit_crawl_page_async(
        team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
        skeleton_rows=[_event("150.0", root_ts="150.0", source_rank=2),
                       _event("160.0", root_ts="150.0", source_rank=2),
                       _event("170.0", root_ts="150.0", source_rank=2)],
        checkpoint_patch={})

    sealed = await temp_db.seal_event_skeleton_async(CRAWL)
    assert sealed["roots"]["150.0"]["reply_count"] == 2          # the root's own row is not one
    assert sealed["roots"]["150.0"]["last_canonical_message_ts"] == "170.0"

    inventory = (await temp_db.load_crawl_checkpoint_async(TEAM, CH, NS))["root_inventory"]
    assert inventory["150.0"]["reply_count"] == 2
    assert inventory["150.0"]["last_canonical_message_ts"] == "170.0"
    assert inventory["150.0"]["root_snippet_len"] == 12          # phase I's own capture stands


async def test_phase_two_refuses_to_start_before_sealing_commits(temp_db):
    """Test 82: chunks are index ranges over `seq`, and `seq` does not exist until sealing
    assigns it — a phase II that started early would read an arbitrary partial chunk."""
    await _checkpoint(temp_db)
    await temp_db.commit_crawl_page_async(
        team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
        skeleton_rows=[_event("100.0")], checkpoint_patch={})

    assert (await temp_db.load_crawl_checkpoint_async(TEAM, CH, NS))["phase"] == 1
    with pytest.raises(ValueError):
        await temp_db.skeleton_slice_async(CRAWL, 0, 500)

    await temp_db.seal_event_skeleton_async(CRAWL)
    assert (await temp_db.load_crawl_checkpoint_async(TEAM, CH, NS))["phase"] == 2
    assert len(await temp_db.skeleton_slice_async(CRAWL, 0, 500)) == 1


async def test_a_duplicate_seq_is_refused_by_the_partial_index(temp_db):
    await _checkpoint(temp_db)
    await temp_db.commit_crawl_page_async(
        team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
        skeleton_rows=[_event("100.0"), _event("200.0")], checkpoint_patch={})
    await temp_db.seal_event_skeleton_async(CRAWL)
    with pytest.raises(sqlite3.IntegrityError):
        temp_db.conn.execute(
            "UPDATE compaction_event_skeleton SET seq = 0 WHERE ts = '200.0'")


async def test_many_candidate_rows_may_share_a_null_seq(temp_db):
    """The index is PARTIAL, so unsealed rows coexist; NULLs would not be distinct under a
    plain unique index on (crawl_id, seq)."""
    await _checkpoint(temp_db)
    await temp_db.commit_crawl_page_async(
        team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
        skeleton_rows=[_event(f"{100 + i}.0") for i in range(5)], checkpoint_patch={})
    assert await temp_db.skeleton_count_async(CRAWL) == 5


async def test_the_skeleton_dies_with_its_crawl(temp_db):
    """Test 78: deleted at publication and swept with its checkpoint, so it never outlives
    the crawl that built it."""
    await _checkpoint(temp_db)
    await temp_db.commit_crawl_page_async(
        team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
        skeleton_rows=[_event("100.0")], checkpoint_patch={})

    sid = await temp_db.insert_compaction_candidate_async(snapshot=_snapshot())
    result = await temp_db.publish_compaction_candidate_async(
        team_id=TEAM, channel_id=CH, namespace=NS, serializer_version=SV, snapshot_id=sid,
        expected_previous_id=None, source_floor_ts="1.0", boundary_ts="1000.000100",
        mutation_frontier=0, current_profile=PROFILE, crawl_id=CRAWL)
    assert result["won"]
    assert await temp_db.skeleton_count_async(CRAWL) == 0
    assert await temp_db.load_crawl_checkpoint_async(TEAM, CH, NS) is None


async def test_the_sweep_removes_an_orphaned_skeleton(temp_db):
    """Test 78, the sweep half: a skeleton whose checkpoint is gone describes nothing."""
    await _checkpoint(temp_db)
    await temp_db.commit_crawl_page_async(
        team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
        skeleton_rows=[_event("100.0")], checkpoint_patch={})
    temp_db.conn.execute("DELETE FROM compaction_crawl_checkpoints")

    await temp_db.sweep_snapshots_async()
    assert await temp_db.skeleton_count_async(CRAWL) == 0


# ===================================================================== telemetry outbox (§1l)

def _outbox(db):
    return [dict(r) for r in db.conn.execute(
        "SELECT * FROM compaction_telemetry_outbox ORDER BY outbox_seq")]


async def test_a_byte_identical_reinsert_is_idempotent_success(temp_db):
    """Test 100(l): re-inserting a BYTE-IDENTICAL payload under an existing identity is
    idempotent success — this is a retry of work already durably recorded."""
    body = _body()
    await temp_db.insert_outbox_rows_async([_row(body)])
    await temp_db.insert_outbox_rows_async([_row(json.loads(json.dumps(body)))])
    rows = _outbox(temp_db)
    assert len(rows) == 1
    assert rows[0]["payload"].encode("utf-8") == canonical_body_bytes(body)


async def test_a_different_payload_under_one_identity_rolls_the_transaction_back(temp_db):
    """Test 100(l): the case `INSERT OR IGNORE` silently swallows. Two distinct events wearing
    one identity is a defect at its source, and the ENCLOSING state change must not commit on
    top of it — so the whole transaction rolls back, not just the row."""
    await temp_db.insert_outbox_rows_async([_row(_body())])
    conflicting = _body(tokens_in=999)

    with pytest.raises(ValueError):
        await temp_db.insert_outbox_rows_async([
            _row(_body(crawl_id="crawl-2")), _row(conflicting)])

    rows = _outbox(temp_db)
    assert len(rows) == 1                       # the innocent row rolled back with the bad one
    assert json.loads(rows[0]["payload"])["tokens_in"] == 10


async def test_a_publish_at_sequence_zero_never_reaches_the_table(temp_db):
    """Clause 5. A misplaced op acknowledged and deleted would permanently violate the
    ordering guarantee with nothing left to inspect."""
    with pytest.raises(ValueError):
        await temp_db.insert_outbox_rows_async([_row(_body("publish", event_seq=0))])
    assert _outbox(temp_db) == []


async def test_an_identity_mismatch_never_reaches_the_table(temp_db):
    """Clause 3: the checker would otherwise dedup by an identity the row does not have."""
    body = _body()
    entry = _row(body)
    entry["event_seq"] = 1
    with pytest.raises(ValueError):
        await temp_db.insert_outbox_rows_async([entry])
    assert _outbox(temp_db) == []


async def test_a_wrapper_key_in_the_body_never_reaches_the_table(temp_db):
    """Clause 6: `v`/`session`/`gate_contract` are EMISSION PROVENANCE, added by the drainer.
    A body carrying one is REFUSED, never merged."""
    for key in ("v", "session", "gate_contract"):
        with pytest.raises(ValueError):
            await temp_db.insert_outbox_rows_async([_row(_body(**{key: "x"}))])
    assert _outbox(temp_db) == []


async def test_created_ts_is_the_bodys_own_at(temp_db):
    """`at` IS the row's created_ts, never now(): reconstructing the body from persisted facts
    must yield identical bytes, and a fresh timestamp would break that."""
    await temp_db.insert_outbox_rows_async([_row(_body(at=1712345678.25))])
    row = _outbox(temp_db)[0]
    assert row["created_ts"] == 1712345678.25
    assert isinstance(row["created_ts"], float)


async def test_read_orders_by_outbox_seq_not_the_identity_triple(temp_db):
    """Test 100(i): crawl_id is a random uuid4, so identity order is NOT time order. A newcomer
    sorting BELOW a queued crawl_id must not jump ahead of rows queued before it."""
    await temp_db.insert_outbox_rows_async([_row(_body(crawl_id="zzz"))])
    await temp_db.insert_outbox_rows_async([_row(_body(crawl_id="aaa"))])
    batch = await temp_db.read_outbox_batch_async()
    assert [r["crawl_id"] for r in batch] == ["zzz", "aaa"]
    assert [r["outbox_seq"] for r in batch] == sorted(r["outbox_seq"] for r in batch)


async def test_an_attempts_rows_are_contiguous_and_ascending(temp_db):
    """Per-attempt order is preserved BY CONSTRUCTION: one transaction, inserted in event_seq
    order, so build reaches the ledger before publish without the triple ordering anything."""
    await temp_db.insert_outbox_rows_async([_row(_body("build")), _row(_body("publish"))])
    batch = await temp_db.read_outbox_batch_async()
    assert [r["body"]["op"] for r in batch] == ["build", "publish"]
    assert batch[1]["outbox_seq"] == batch[0]["outbox_seq"] + 1


async def test_a_row_corrupted_after_landing_is_flagged_on_read(temp_db):
    """Insert-time validation alone is not enough: a row can be corrupted AFTER it lands, and
    a corrupted-but-finite `at` passes every other clause."""
    await temp_db.insert_outbox_rows_async([_row(_body())])
    seq = _outbox(temp_db)[0]["outbox_seq"]
    corrupted = _body(at=1700000000.5)
    corrupted["at"] = 1700000999.5
    temp_db.conn.execute(
        "UPDATE compaction_telemetry_outbox SET payload = ? WHERE outbox_seq = ?",
        (canonical_body_bytes(corrupted).decode("utf-8"), seq))

    batch = await temp_db.read_outbox_batch_async()
    assert batch[0]["invalid"]


async def test_delete_only_removes_the_named_row(temp_db):
    await temp_db.insert_outbox_rows_async([_row(_body("build")), _row(_body("publish"))])
    batch = await temp_db.read_outbox_batch_async()
    assert await temp_db.delete_outbox_row_async(batch[0]["outbox_seq"])
    assert not await temp_db.delete_outbox_row_async(batch[0]["outbox_seq"])
    assert [r["outbox_seq"] for r in _outbox(temp_db)] == [batch[1]["outbox_seq"]]


async def test_outbox_seq_is_never_reused(temp_db):
    """AUTOINCREMENT: delivery order must stay monotonic across the deletes the drainer does
    after every acknowledgement."""
    await temp_db.insert_outbox_rows_async([_row(_body())])
    first = _outbox(temp_db)[0]["outbox_seq"]
    await temp_db.delete_outbox_row_async(first)
    await temp_db.insert_outbox_rows_async([_row(_body(crawl_id="crawl-2"))])
    assert _outbox(temp_db)[0]["outbox_seq"] > first


async def test_an_uncertain_commit_retry_reconstructs_identical_bytes(temp_db, tmp_path,
                                                                     monkeypatch):
    """Test 100(m): the retry must RECONSTRUCT the body from PERSISTED TERMINAL FACTS, not
    reuse the original Python object. Asserted across a simulated restart, so the reconstructing
    session differs from the original — holding the first payload in memory would pass even for
    an implementation stamping now() or a session id."""
    facts = {"crawl_id": CRAWL, "attempt_seq": 0, "at": 1700000000.5}
    await temp_db.insert_outbox_rows_async([_row(_body(**{"crawl_id": facts["crawl_id"],
                                                          "at": facts["at"]}))])
    original = _outbox(temp_db)[0]["payload"]
    temp_db.conn.close()

    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    rebooted = DatabaseManager(platform="slack")
    try:
        rebuilt = _body(crawl_id=facts["crawl_id"], attempt_seq=facts["attempt_seq"],
                        at=facts["at"])
        await rebooted.insert_outbox_rows_async([_row(rebuilt)])
        rows = _outbox(rebooted)
        assert len(rows) == 1
        assert rows[0]["payload"] == original
    finally:
        rebooted.conn.close()


# =========================================================== pending recompaction (§1m)

async def test_the_map_takes_the_max_per_profile_key(temp_db):
    """Test 92: generation 11 measured at 40k must NOT erase generation 10's still-unsatisfied
    90k requirement under the same key, which a scalar headroom field would do."""
    await temp_db.merge_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, profile_key=PROFILE,
        required_headroom=90000, obligated_snapshot_id="s10", obligated_generation=10,
        reason="fixed")
    await temp_db.merge_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, profile_key=PROFILE,
        required_headroom=40000, obligated_snapshot_id="s11", obligated_generation=11,
        reason="fixed")
    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert row["requirements"] == {PROFILE: 90000}


async def test_untouched_keys_survive_a_merge_and_created_ts_stays_earliest(temp_db):
    await temp_db.merge_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, profile_key=OTHER_PROFILE,
        required_headroom=50000, obligated_snapshot_id="s1", obligated_generation=1,
        reason="first")
    first_created = (await temp_db.load_pending_recompaction_async(
        TEAM, CH, NS))["created_ts"]
    await asyncio.sleep(0.01)
    await temp_db.merge_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, profile_key=PROFILE,
        required_headroom=70000, obligated_snapshot_id="s2", obligated_generation=2,
        reason="second")

    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert row["requirements"] == {OTHER_PROFILE: 50000, PROFILE: 70000}
    assert row["created_ts"] == first_created


async def test_the_obligated_pair_moves_together_or_not_at_all(temp_db):
    """Test 97: a losing merge changes NEITHER field; a winning one replaces BOTH. A row
    naming a snapshot that is not its claimed generation is a failure."""
    await temp_db.merge_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, profile_key=PROFILE,
        required_headroom=10, obligated_snapshot_id="s5", obligated_generation=5,
        reason="fixed")
    await temp_db.merge_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, profile_key=PROFILE,
        required_headroom=20, obligated_snapshot_id="s3", obligated_generation=3,
        reason="fixed")
    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert (row["obligated_snapshot_id"], row["obligated_generation"]) == ("s5", 5)

    await temp_db.merge_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, profile_key=PROFILE,
        required_headroom=20, obligated_snapshot_id="s9", obligated_generation=9,
        reason="fixed")
    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert (row["obligated_snapshot_id"], row["obligated_generation"]) == ("s9", 9)


async def test_concurrent_merges_never_lose_the_larger_requirement(temp_db):
    """Test 92: `BEGIN IMMEDIATE` read-modify-write. A plain read-merge-write would let two
    enqueues interleave and lose exactly the value the map exists to keep."""
    await asyncio.gather(*[
        temp_db.merge_pending_recompaction_async(
            team_id=TEAM, channel_id=CH, namespace=NS, profile_key=PROFILE,
            required_headroom=headroom, obligated_snapshot_id=f"s{i}",
            obligated_generation=i, reason="fixed")
        for i, headroom in enumerate([10000, 90000, 30000, 50000])])
    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert row["requirements"] == {PROFILE: 90000}


async def test_the_requirement_map_is_canonically_encoded(temp_db):
    """Sorted keys, no insignificant whitespace, so two encodings of the same map are
    byte-identical."""
    for key in (OTHER_PROFILE, PROFILE):
        await temp_db.merge_pending_recompaction_async(
            team_id=TEAM, channel_id=CH, namespace=NS, profile_key=key,
            required_headroom=1, obligated_snapshot_id="s1", obligated_generation=1,
            reason="fixed")
    raw = temp_db.conn.execute("SELECT requirements FROM pending_recompaction").fetchone()[0]
    assert raw == json.dumps(json.loads(raw), sort_keys=True, separators=(",", ":"))


async def test_a_dormant_row_missing_its_deadline_cannot_be_written(temp_db):
    """The two structural dormancy invariants are DB CHECK constraints, not comments."""
    await temp_db.merge_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, profile_key=PROFILE, required_headroom=1,
        obligated_snapshot_id="s1", obligated_generation=1, reason="fixed")
    with pytest.raises(sqlite3.IntegrityError):
        temp_db.conn.execute(
            "UPDATE pending_recompaction SET state = 'dormant', dormant_profile_key = 'p'")
    with pytest.raises(sqlite3.IntegrityError):
        temp_db.conn.execute(
            "UPDATE pending_recompaction SET next_attempt_after = '2030-01-01'")


async def test_a_dormancy_key_absent_from_the_map_fails_closed(temp_db, caplog):
    """The third invariant is JSON and cannot be a CHECK, so the ACCESSOR enforces it — and it
    fails CLOSED. Reading a malformed row as ACTIVE would let it bypass the backoff, which is
    the one outcome this machinery exists to prevent."""
    await temp_db.merge_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, profile_key=PROFILE, required_headroom=1,
        obligated_snapshot_id="s1", obligated_generation=1, reason="fixed")
    temp_db.conn.execute(
        "UPDATE pending_recompaction SET state = 'dormant', "
        "dormant_profile_key = 'a-profile-nobody-uses', next_attempt_after = '2030-01-01'")

    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert row["state"] == "dormant"
    assert row["malformed"] == "dormant_profile_key"


async def test_the_malformed_critical_is_bounded_once_per_row_per_boot(temp_db, caplog):
    await temp_db.merge_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, profile_key=PROFILE, required_headroom=1,
        obligated_snapshot_id="s1", obligated_generation=1, reason="fixed")
    temp_db.conn.execute("UPDATE pending_recompaction SET requirements = 'not json'")

    with caplog.at_level("ERROR"):
        for _ in range(5):
            await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert sum("CRITICAL: pending_recompaction" in r.message for r in caplog.records) == 1


async def test_the_revival_cas_checks_the_profile_the_pair_and_the_deadline(temp_db):
    await temp_db.merge_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, profile_key=PROFILE, required_headroom=1,
        obligated_snapshot_id="s1", obligated_generation=1, reason="fixed")
    assert await temp_db.cas_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, expect_state="active",
        new_state="dormant", dormant_profile_key=PROFILE,
        next_attempt_after="2030-01-01T00:00:00")

    # Wrong profile, wrong pair, and an unexpired deadline each lose.
    assert not await temp_db.cas_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, expect_state="dormant",
        new_state="active", expect_profile_key=OTHER_PROFILE, deadline_passed=True)
    assert not await temp_db.cas_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, expect_state="dormant",
        new_state="active", expect_pair=("s2", 2), deadline_passed=True)
    assert not await temp_db.cas_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, expect_state="dormant",
        new_state="active", expect_profile_key=PROFILE, deadline_passed=True)

    temp_db.conn.execute(
        "UPDATE pending_recompaction SET next_attempt_after = '2000-01-01T00:00:00'")
    assert await temp_db.cas_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, expect_state="dormant", new_state="active",
        expect_profile_key=PROFILE, expect_pair=("s1", 1), deadline_passed=True)
    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert (row["state"], row["dormant_profile_key"], row["next_attempt_after"]) == (
        "active", None, None)


async def test_only_one_of_two_racing_revivals_wins(temp_db):
    await temp_db.merge_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, profile_key=PROFILE, required_headroom=1,
        obligated_snapshot_id="s1", obligated_generation=1, reason="fixed")
    await temp_db.cas_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, expect_state="active", new_state="dormant",
        dormant_profile_key=PROFILE, next_attempt_after="2000-01-01T00:00:00")

    results = await asyncio.gather(*[
        temp_db.cas_pending_recompaction_async(
            team_id=TEAM, channel_id=CH, namespace=NS, expect_state="dormant",
            new_state="active", expect_profile_key=PROFILE, deadline_passed=True)
        for _ in range(2)])
    assert sorted(results) == [False, True]


async def test_obsolete_profiles_are_retired_and_stale_dormancy_is_cleared(temp_db):
    """After a model/window/threshold change no current-profile publication can dominate the
    old key, so the entry is DELETED. Dormancy that belonged to the retired key is CLEARED
    rather than inherited — otherwise a brand-new obligation sits under the old one's deadline,
    suppressed before it was ever attempted."""
    for key, headroom in ((OTHER_PROFILE, 50000), (PROFILE, 70000)):
        await temp_db.merge_pending_recompaction_async(
            team_id=TEAM, channel_id=CH, namespace=NS, profile_key=key,
            required_headroom=headroom, obligated_snapshot_id="s1", obligated_generation=1,
            reason="fixed")
    await temp_db.cas_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, expect_state="active", new_state="dormant",
        dormant_profile_key=OTHER_PROFILE, next_attempt_after="2030-01-01T00:00:00")

    retired = await temp_db.reconcile_pending_profiles_async(
        team_id=TEAM, channel_id=CH, namespace=NS, current_profile=PROFILE)
    assert retired == [OTHER_PROFILE]
    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert row["requirements"] == {PROFILE: 70000}
    assert (row["state"], row["next_attempt_after"]) == ("active", None)


async def test_reconciliation_deletes_the_row_when_the_map_empties(temp_db):
    await temp_db.merge_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, profile_key=OTHER_PROFILE,
        required_headroom=1, obligated_snapshot_id="s1", obligated_generation=1,
        reason="fixed")
    assert await temp_db.reconcile_pending_profiles_async(
        team_id=TEAM, channel_id=CH, namespace=NS, current_profile=PROFILE) == [OTHER_PROFILE]
    assert await temp_db.load_pending_recompaction_async(TEAM, CH, NS) is None


async def test_reconciliation_keeps_a_current_profile_dormancy(temp_db):
    for key in (OTHER_PROFILE, PROFILE):
        await temp_db.merge_pending_recompaction_async(
            team_id=TEAM, channel_id=CH, namespace=NS, profile_key=key, required_headroom=1,
            obligated_snapshot_id="s1", obligated_generation=1, reason="fixed")
    await temp_db.cas_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, expect_state="active", new_state="dormant",
        dormant_profile_key=PROFILE, next_attempt_after="2030-01-01T00:00:00")

    await temp_db.reconcile_pending_profiles_async(
        team_id=TEAM, channel_id=CH, namespace=NS, current_profile=PROFILE)
    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert (row["state"], row["next_attempt_after"]) == ("dormant", "2030-01-01T00:00:00")


async def test_boot_hydration_reads_every_pending_row(temp_db):
    """Test 80: a durable enqueue is a ROW, not a task-map entry — the coordinator's map is
    process-local and dies with the process."""
    for channel in ("C1", "C2"):
        await temp_db.merge_pending_recompaction_async(
            team_id=TEAM, channel_id=channel, namespace=NS, profile_key=PROFILE,
            required_headroom=1, obligated_snapshot_id="s1", obligated_generation=1,
            reason="fixed")
    assert {r["channel_id"] for r in await temp_db.all_pending_recompactions_async()} == {
        "C1", "C2"}


# =========================================================== cancellation intent (§1m)

def _intents(db):
    return [dict(r) for r in db.conn.execute(
        "SELECT * FROM compaction_cancellation_intent")]


async def test_the_intent_and_the_requirement_removal_commit_together(temp_db):
    """Test 101: the intent is inserted in the SAME transaction that removes the requirement,
    never after — otherwise boot finds an orphan checkpoint with no obligation to explain it."""
    for key in (OTHER_PROFILE, PROFILE):
        await temp_db.merge_pending_recompaction_async(
            team_id=TEAM, channel_id=CH, namespace=NS, profile_key=key, required_headroom=1,
            obligated_snapshot_id="s1", obligated_generation=1, reason="fixed")

    await temp_db.write_cancellation_intent_async(
        {"team_id": TEAM, "channel_id": CH, "namespace": NS, "crawl_id": CRAWL,
         "obligated_snapshot_id": "s1", "reason": "profile_changed",
         "created_ts": "2026-01-01"},
        retire_keys=[OTHER_PROFILE])

    assert len(_intents(temp_db)) == 1
    assert (await temp_db.load_pending_recompaction_async(
        TEAM, CH, NS))["requirements"] == {PROFILE: 1}


async def test_a_repeated_reconciliation_of_one_crawl_is_first_write_wins(temp_db):
    """Test 101: duplicate intents collide on the primary key, so a conflicting `reason` or
    `obligated_snapshot_id` cannot overwrite the first."""
    intent = {"team_id": TEAM, "channel_id": CH, "namespace": NS, "crawl_id": CRAWL,
              "obligated_snapshot_id": "s1", "reason": "profile_changed",
              "created_ts": "2026-01-01"}
    await temp_db.write_cancellation_intent_async(intent)
    await temp_db.write_cancellation_intent_async(
        {**intent, "obligated_snapshot_id": "s2", "reason": "something_else"})

    rows = _intents(temp_db)
    assert len(rows) == 1
    assert (rows[0]["obligated_snapshot_id"], rows[0]["reason"]) == ("s1", "profile_changed")


async def test_boot_recovery_finishes_the_discard_in_one_transaction(temp_db):
    """Test 101(a): crash BEFORE recovery runs — boot finds intent + orphan checkpoint and
    finishes the discard completely. ONE transaction covering outbox insertion, checkpoint,
    skeleton, candidate cleanup, protection release and intent deletion."""
    await _checkpoint(temp_db, parent="parent-1", mode="incremental")
    await temp_db.commit_crawl_page_async(
        team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
        skeleton_rows=[_event("100.0")], checkpoint_patch={})
    candidate = await temp_db.insert_compaction_candidate_async(
        snapshot=_snapshot(),
        manifest_rows=[{"artifact_namespace": "image_analysis", "row_id": "7",
                        "source_ts": "5.0", "captured_render_version": "v1",
                        "content_hash": "h", "status_at_capture": "analysis"}],
        anchor_rows=[{"root_ts": "0.5", "status": "available", "projection_sha256": "p",
                      "observation_frontier": 0, "receipt_proof": "not_self"}])
    await temp_db.write_cancellation_intent_async(
        {"team_id": TEAM, "channel_id": CH, "namespace": NS, "crawl_id": CRAWL,
         "obligated_snapshot_id": "s1", "reason": "profile_changed",
         "created_ts": "2026-01-01"})

    assert await temp_db.finish_cancellation_discard_async(
        team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
        outbox_rows=[_row(_body(status="discarded", reason="obsolete_profile"))],
        candidate_id=candidate)

    assert _intents(temp_db) == []
    assert await temp_db.load_crawl_checkpoint_async(TEAM, CH, NS) is None
    assert await temp_db.skeleton_count_async(CRAWL) == 0
    assert await temp_db.get_snapshot_async(candidate) is None
    assert await temp_db.snapshot_manifest_async(candidate) == []
    assert await temp_db.snapshot_anchor_provenance_async(candidate) == []
    assert await temp_db.live_checkpoint_parent_ids_async() == []
    assert len(_outbox(temp_db)) == 1


async def test_a_failed_recovery_applies_nothing(temp_db):
    """Test 101(b): the transaction is atomic, so nothing is half-applied and the replay
    completes it. A poisoned outbox row is the failure injected here because it is the one
    that can genuinely arrive."""
    await _checkpoint(temp_db)
    await temp_db.write_cancellation_intent_async(
        {"team_id": TEAM, "channel_id": CH, "namespace": NS, "crawl_id": CRAWL,
         "obligated_snapshot_id": "s1", "reason": "profile_changed",
         "created_ts": "2026-01-01"})

    with pytest.raises(ValueError):
        await temp_db.finish_cancellation_discard_async(
            team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
            outbox_rows=[_row(_body("publish", event_seq=0))], candidate_id=None)

    assert len(_intents(temp_db)) == 1
    assert await temp_db.load_crawl_checkpoint_async(TEAM, CH, NS) is not None

    assert await temp_db.finish_cancellation_discard_async(
        team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
        outbox_rows=[_row(_body(status="discarded", reason="obsolete_profile"))],
        candidate_id=None)
    assert _intents(temp_db) == []
    assert await temp_db.load_crawl_checkpoint_async(TEAM, CH, NS) is None


async def test_both_cleanup_paths_use_the_same_atomic_accessor(temp_db, monkeypatch):
    """Test 101(c), the DIVERGENCE test. Live chunk-boundary cleanup and boot recovery call the
    SAME transactional accessor, which is what makes intent-without-checkpoint impossible
    rather than merely rare. Monkeypatching the shared accessor to skip the checkpoint delete
    must break BOTH callers — if one had its own implementation, only one would break.
    """
    calls = []
    real = DatabaseManager._delete_crawl_state

    async def skipping(db, team_id, channel_id, namespace, crawl_id):
        calls.append(crawl_id)
        # Deliberately skip the checkpoint delete, keeping only the skeleton half.
        await db.execute("DELETE FROM compaction_event_skeleton WHERE crawl_id = ?",
                         (crawl_id,))

    monkeypatch.setattr(DatabaseManager, "_delete_crawl_state", staticmethod(skipping))
    for crawl_id in ("live-cleanup", "boot-recovery"):
        await _checkpoint(temp_db, crawl_id=crawl_id)
        await temp_db.write_cancellation_intent_async(
            {"team_id": TEAM, "channel_id": CH, "namespace": NS, "crawl_id": crawl_id,
             "obligated_snapshot_id": "s1", "reason": "r", "created_ts": "2026-01-01"})
        await temp_db.finish_cancellation_discard_async(
            team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=crawl_id, candidate_id=None)
        assert await temp_db.load_crawl_checkpoint_async(TEAM, CH, NS) is not None

    assert calls == ["live-cleanup", "boot-recovery"]
    monkeypatch.setattr(DatabaseManager, "_delete_crawl_state", real)


# =========================================================== namespace rebuild (§3c, test 56)

LEGACY_SNAPSHOT_DDL = """
    CREATE TABLE channel_snapshots (
        snapshot_id TEXT PRIMARY KEY,
        team_id TEXT NOT NULL,
        channel_id TEXT NOT NULL,
        serializer_version INTEGER NOT NULL,
        generation INTEGER,
        boundary_ts TEXT NOT NULL,
        summary_text TEXT NOT NULL,
        root_anchors_json TEXT,
        created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        invalidated_at TIMESTAMP
    )
"""
LEGACY_POINTER_DDL = """
    CREATE TABLE channel_snapshot_pointer (
        team_id TEXT NOT NULL,
        channel_id TEXT NOT NULL,
        serializer_version INTEGER NOT NULL,
        active_snapshot_id TEXT,
        updated_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (team_id, channel_id, serializer_version)
    )
"""


def _make_p1_database(tmp_path, monkeypatch, *, rows=2):
    """A POPULATED P1 database: the landed shape, with real snapshot and pointer rows.

    Test 56 is explicit that the migration must work on a populated database, not an empty
    one — a rebuild that silently drops rows passes an empty-table test.
    """
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    db = DatabaseManager(platform="slack")
    db.conn.execute("DROP TABLE channel_snapshots")
    db.conn.execute("DROP TABLE channel_snapshot_pointer")
    db.conn.execute(LEGACY_SNAPSHOT_DDL)
    db.conn.execute(LEGACY_POINTER_DDL)
    db.conn.execute(
        "CREATE UNIQUE INDEX idx_channel_snapshot_generation "
        "ON channel_snapshots (team_id, channel_id, serializer_version, generation)")
    for i in range(rows):
        db.conn.execute(
            "INSERT INTO channel_snapshots (snapshot_id, team_id, channel_id, "
            "serializer_version, generation, boundary_ts, summary_text) "
            "VALUES (?, ?, ?, 1, ?, ?, ?)",
            (f"legacy-{i}", TEAM, CH, i + 1, f"{100 + i}.0", f"v1 summary {i}"))
    db.conn.execute(
        "INSERT INTO channel_snapshot_pointer (team_id, channel_id, serializer_version, "
        "active_snapshot_id) VALUES (?, ?, 1, ?)", (TEAM, CH, f"legacy-{rows - 1}"))
    db.conn.execute("DELETE FROM bot_meta WHERE key = 'snapshot_v1_pointers_retired_at'")
    db.conn.commit()
    db.conn.close()


def test_a_populated_p1_database_is_rebuilt_with_namespace_in_its_keys(tmp_path, monkeypatch):
    _make_p1_database(tmp_path, monkeypatch)
    db = DatabaseManager(platform="slack")
    try:
        rows = [dict(r) for r in db.conn.execute(
            "SELECT * FROM channel_snapshots ORDER BY generation")]
        assert [r["snapshot_id"] for r in rows] == ["legacy-0", "legacy-1"]
        assert {r["namespace"] for r in rows} == {PROD_NAMESPACE}
        assert [r["summary_text"] for r in rows] == ["v1 summary 0", "v1 summary 1"]

        key = [r["name"] for r in
               db.conn.execute("PRAGMA table_info(channel_snapshot_pointer)") if r["pk"]]
        assert set(key) == {"team_id", "channel_id", "namespace", "serializer_version"}

        index_cols = [r["name"] for r in
                      db.conn.execute("PRAGMA index_info(idx_channel_snapshot_generation)")]
        assert "namespace" in index_cols
        assert db._snapshot_schema_mismatch() is None
    finally:
        db.conn.close()


def test_the_rebuild_backfills_null_sizing_evidence_on_legacy_rows(tmp_path, monkeypatch):
    """Test 56: NULL is a NEVER-DOMINATING sentinel. A backfill INVENTING plausible sizing
    evidence must fail this — a fabricated `under_target` would discharge an obligation on the
    strength of a measurement nobody made."""
    _make_p1_database(tmp_path, monkeypatch)
    db = DatabaseManager(platform="slack")
    try:
        row = dict(db.conn.execute(
            "SELECT * FROM channel_snapshots WHERE snapshot_id = 'legacy-0'").fetchone())
        for column in ("headroom_source", "headroom_tokens", "effective_window",
                       "sizing_profile", "fit_result", "source_floor_ts", "payload_hash",
                       "payload_bytes", "mutation_frontier"):
            assert row[column] is None, column
        assert row["status"] == "published"
    finally:
        db.conn.close()


async def test_a_null_sizing_row_never_satisfies_an_obligation(temp_db):
    """Test 56's second half, at the accessor: NULL sizing evidence is a NEVER-DOMINATING
    sentinel, so a generation carrying it discharges nothing however new it is."""
    await temp_db.merge_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, profile_key=PROFILE,
        required_headroom=1000, obligated_snapshot_id="s1", obligated_generation=1,
        reason="fixed")
    sid = await temp_db.insert_compaction_candidate_async(snapshot=_snapshot())
    temp_db.conn.execute(
        "UPDATE channel_snapshots SET headroom_tokens = NULL, fit_result = NULL "
        "WHERE snapshot_id = ?", (sid,))

    result = await temp_db.publish_compaction_candidate_async(
        team_id=TEAM, channel_id=CH, namespace=NS, serializer_version=SV, snapshot_id=sid,
        expected_previous_id=None, source_floor_ts="1.0", boundary_ts="1000.000100",
        mutation_frontier=0, current_profile=PROFILE, satisfy={})
    assert result["won"]
    assert (await temp_db.load_pending_recompaction_async(
        TEAM, CH, NS))["requirements"] == {PROFILE: 1000}


def test_the_v1_pointer_is_retired_so_its_generations_become_sweepable(tmp_path, monkeypatch):
    """§1h: the sweep protects active pointer targets, so without deleting the v1 pointer the
    v1 generations would persist indefinitely. The ROWS survive; only the pointer goes."""
    _make_p1_database(tmp_path, monkeypatch)
    db = DatabaseManager(platform="slack")
    try:
        assert db.conn.execute(
            "SELECT COUNT(*) FROM channel_snapshot_pointer").fetchone()[0] == 0
        assert db.conn.execute(
            "SELECT COUNT(*) FROM channel_snapshots").fetchone()[0] == 2
        assert db.get_meta("snapshot_v1_pointers_retired_at")
    finally:
        db.conn.close()


def test_a_mismatched_rebuilt_schema_fails_startup(tmp_path, monkeypatch):
    """Test 56: the §3c validation is REQUIRED-CRITICAL. It FAILS STARTUP rather than serving a
    database whose uniqueness constraints cannot enforce one production pointer per channel —
    unlike `_migration_step`, which logs and continues."""
    import database as database_module

    _make_p1_database(tmp_path, monkeypatch)

    def namespaceless(table):
        suffix = "" if table == "channel_snapshots" else "_new"
        return (
            f"CREATE UNIQUE INDEX IF NOT EXISTS idx_channel_snapshot_generation{suffix} "
            f"ON {table} (team_id, channel_id, serializer_version, generation)",
            f"CREATE INDEX IF NOT EXISTS idx_channel_snapshot_scope{suffix} "
            f"ON {table} (team_id, channel_id, serializer_version)",
        )

    monkeypatch.setattr(database_module, "_snapshot_index_sql", namespaceless)
    with pytest.raises(RuntimeError, match="namespace"):
        DatabaseManager(platform="slack")


def test_a_second_boot_leaves_the_rebuilt_schema_alone(tmp_path, monkeypatch):
    """The rebuild is guarded on the mismatch check, so it runs once and a healthy database is
    never dropped and recreated on every boot."""
    _make_p1_database(tmp_path, monkeypatch)
    first = DatabaseManager(platform="slack")
    first.conn.close()
    second = DatabaseManager(platform="slack")
    try:
        assert second._snapshot_schema_mismatch() is None
        assert second.conn.execute(
            "SELECT COUNT(*) FROM channel_snapshots").fetchone()[0] == 2
    finally:
        second.conn.close()


def test_a_namespaceless_unique_index_is_reported_as_a_mismatch(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    db = DatabaseManager(platform="slack")
    try:
        assert db._snapshot_schema_mismatch() is None
        db.conn.execute("DROP INDEX idx_channel_snapshot_generation")
        db.conn.execute(
            "CREATE UNIQUE INDEX idx_channel_snapshot_generation "
            "ON channel_snapshots (team_id, channel_id, serializer_version, generation)")
        assert "namespace" in db._snapshot_schema_mismatch()
    finally:
        db.conn.close()


async def test_two_namespaces_hold_independent_pointers(temp_db):
    """§3c: the non-null namespace is what lets a fenced test epoch and production coexist
    without contending for one pointer row."""
    prod = await temp_db.insert_compaction_candidate_async(snapshot=_snapshot())
    fenced = await temp_db.insert_compaction_candidate_async(
        snapshot=_snapshot(namespace="epoch-7"))
    for sid, namespace in ((prod, NS), (fenced, "epoch-7")):
        result = await temp_db.publish_compaction_candidate_async(
            team_id=TEAM, channel_id=CH, namespace=namespace, serializer_version=SV,
            snapshot_id=sid, expected_previous_id=None, source_floor_ts="1.0",
            boundary_ts="1000.000100", mutation_frontier=0, current_profile=PROFILE)
        assert result["won"], (namespace, result)

    assert (await temp_db.select_snapshot_for_pin_async(
        TEAM, CH, NS, SV, None))["snapshot_id"] == prod
    assert (await temp_db.select_snapshot_for_pin_async(
        TEAM, CH, "epoch-7", SV, None))["snapshot_id"] == fenced


# =========================================== publish-nothing terminal (§1m, accessor contract)

async def _terminal_setup(temp_db, *, profile=PROFILE, state="active"):
    """A live crawl (checkpoint + skeleton + candidate) behind an obligation row."""
    await _checkpoint(temp_db, parent="parent-1", mode="incremental")
    await temp_db.commit_crawl_page_async(
        team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
        skeleton_rows=[_event("100.0")], checkpoint_patch={})
    candidate = await temp_db.insert_compaction_candidate_async(
        snapshot=_snapshot(),
        manifest_rows=[{"artifact_namespace": "image_analysis", "row_id": "7",
                        "source_ts": "5.0", "captured_render_version": "v1",
                        "content_hash": "h", "status_at_capture": "analysis"}],
        anchor_rows=[{"root_ts": "0.5", "status": "available", "projection_sha256": "p",
                      "observation_frontier": 0, "receipt_proof": "not_self"}])
    await temp_db.merge_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, profile_key=profile, required_headroom=1,
        obligated_snapshot_id="s1", obligated_generation=1, reason="fixed")
    if state == "dormant":
        assert await temp_db.cas_pending_recompaction_async(
            team_id=TEAM, channel_id=CH, namespace=NS, expect_state="active",
            new_state="dormant", dormant_profile_key=profile,
            next_attempt_after="2030-06-01T00:00:00")
    return candidate


async def _assert_crawl_state_gone(temp_db, candidate):
    """The load-bearing half: the crawl state must DIE WITH THE ATTEMPT, because a revived
    attempt starts a NEW crawl with a FRESH H. A surviving checkpoint still pins the OLD H."""
    assert await temp_db.load_crawl_checkpoint_async(TEAM, CH, NS) is None
    assert await temp_db.skeleton_count_async(CRAWL) == 0
    assert temp_db.conn.execute(
        "SELECT COUNT(*) FROM compaction_event_skeleton").fetchone()[0] == 0
    assert await temp_db.get_snapshot_async(candidate) is None
    assert await temp_db.snapshot_manifest_async(candidate) == []
    assert await temp_db.snapshot_anchor_provenance_async(candidate) == []
    # Deleting the checkpoint IS the parent-sweep protection release.
    assert await temp_db.live_checkpoint_parent_ids_async() == []


async def test_all_three_predicates_passing_marks_the_row_dormant(temp_db):
    candidate = await _terminal_setup(temp_db)
    result = await temp_db.terminal_publish_nothing_async(
        team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
        expect_state="active", expect_pair=("s1", 1), expect_profile_key=PROFILE,
        dormant_profile_key=PROFILE, next_attempt_after="2030-01-01T00:00:00",
        outbox_rows=[_row(_body(status="discarded", reason="publish_nothing"))],
        candidate_id=candidate)

    assert result == {"ok": True, "mismatch": None}
    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert (row["state"], row["dormant_profile_key"]) == ("dormant", PROFILE)
    assert row["next_attempt_after"] == "2030-01-01T00:00:00"
    assert "malformed" not in row
    assert len(_outbox(temp_db)) == 1
    await _assert_crawl_state_gone(temp_db, candidate)


async def test_a_profile_mismatch_routes_to_obsolescence(temp_db):
    """PROFILE mismatch means the task really is sized for a configuration that no longer
    exists — a different class from an older attempt finishing late."""
    candidate = await _terminal_setup(temp_db, profile=OTHER_PROFILE)
    result = await temp_db.terminal_publish_nothing_async(
        team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
        expect_profile_key=PROFILE, dormant_profile_key=PROFILE,
        next_attempt_after="2030-01-01T00:00:00", candidate_id=candidate)

    assert result == {"ok": False, "mismatch": "profile"}
    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert row["state"] == "active"                      # the newer obligation is UNTOUCHED
    await _assert_crawl_state_gone(temp_db, candidate)


async def test_a_state_mismatch_discards_only_the_stale_attempt(temp_db):
    """SAME-PROFILE state supersession: the newer obligation is left SCHEDULED AND UNTOUCHED.
    Treating this as obsolescence would retire a live, current-profile obligation."""
    candidate = await _terminal_setup(temp_db, state="dormant")
    result = await temp_db.terminal_publish_nothing_async(
        team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
        expect_state="active", expect_profile_key=PROFILE, dormant_profile_key=PROFILE,
        next_attempt_after="2030-01-01T00:00:00", candidate_id=candidate)

    assert result == {"ok": False, "mismatch": "state"}
    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert row["next_attempt_after"] == "2030-06-01T00:00:00"   # its OWN deadline survives
    await _assert_crawl_state_gone(temp_db, candidate)


async def test_an_obligated_pair_mismatch_discards_only_the_stale_attempt(temp_db):
    """A newer obligation superseded the pair while the old attempt ground toward
    publish-nothing; putting a never-attempted obligation into an hour's backoff is exactly
    what the predicate prevents."""
    candidate = await _terminal_setup(temp_db)
    await temp_db.merge_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, profile_key=PROFILE, required_headroom=1,
        obligated_snapshot_id="s9", obligated_generation=9, reason="fixed")

    result = await temp_db.terminal_publish_nothing_async(
        team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
        expect_state="active", expect_pair=("s1", 1), expect_profile_key=PROFILE,
        dormant_profile_key=PROFILE, next_attempt_after="2030-01-01T00:00:00",
        candidate_id=candidate)

    assert result == {"ok": False, "mismatch": "pair"}
    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert (row["state"], row["obligated_snapshot_id"]) == ("active", "s9")
    assert row["next_attempt_after"] is None
    await _assert_crawl_state_gone(temp_db, candidate)


async def test_the_three_mismatch_classes_are_never_collapsed(temp_db):
    """They ROUTE DIFFERENTLY in §1m, so a test accepting any non-None tag misses the point."""
    seen = set()
    for label, kwargs in (
            ("profile", {"profile": OTHER_PROFILE}),
            ("state", {"state": "dormant"}),
    ):
        temp_db.conn.execute("DELETE FROM pending_recompaction")
        temp_db.conn.execute("DELETE FROM compaction_crawl_checkpoints")
        candidate = await _terminal_setup(temp_db, **kwargs)
        result = await temp_db.terminal_publish_nothing_async(
            team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
            expect_state="active", expect_profile_key=PROFILE, dormant_profile_key=PROFILE,
            next_attempt_after="2030-01-01T00:00:00", candidate_id=candidate)
        assert result["mismatch"] == label
        seen.add(result["mismatch"])
    assert seen == {"profile", "state"}


async def test_cleanup_survives_a_restart_after_a_mismatch(temp_db, tmp_path, monkeypatch):
    """Test 95f's shape at this layer: NO checkpoint survives, asserted ACROSS A RESTART."""
    candidate = await _terminal_setup(temp_db, profile=OTHER_PROFILE)
    await temp_db.terminal_publish_nothing_async(
        team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
        expect_profile_key=PROFILE, dormant_profile_key=PROFILE,
        next_attempt_after="2030-01-01T00:00:00", candidate_id=candidate)
    temp_db.conn.close()

    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    rebooted = DatabaseManager(platform="slack")
    try:
        assert await rebooted.load_crawl_checkpoint_async(TEAM, CH, NS) is None
        assert await rebooted.skeleton_count_async(CRAWL) == 0
        assert await rebooted.live_checkpoint_parent_ids_async() == []
    finally:
        rebooted.conn.close()


async def test_an_absent_obligation_row_is_a_state_mismatch_not_a_profile_one(temp_db):
    """There is no obligation left to retire and none to reschedule, so the harmless
    stale-attempt route is the honest one — routing it to obsolescence would be a claim about
    a configuration change that never happened."""
    await _checkpoint(temp_db)
    result = await temp_db.terminal_publish_nothing_async(
        team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
        expect_profile_key=PROFILE, dormant_profile_key=PROFILE,
        next_attempt_after="2030-01-01T00:00:00")
    assert result == {"ok": False, "mismatch": "state"}
    assert await temp_db.load_crawl_checkpoint_async(TEAM, CH, NS) is None


async def test_a_dormancy_key_absent_from_requirements_is_refused_as_a_profile_mismatch(
        temp_db):
    """Marking it dormant anyway would write exactly the row the dormancy validator has to
    fail closed on, so the accessor refuses instead of creating one."""
    candidate = await _terminal_setup(temp_db)
    result = await temp_db.terminal_publish_nothing_async(
        team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
        expect_state="active", dormant_profile_key="a-profile-nobody-uses",
        next_attempt_after="2030-01-01T00:00:00", candidate_id=candidate)

    assert result == {"ok": False, "mismatch": "profile"}
    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert row["state"] == "active" and "malformed" not in row


@pytest.mark.parametrize("missing", ["dormant_profile_key", "next_attempt_after"])
async def test_a_dormancy_field_missing_is_refused_before_anything_is_cleaned_up(temp_db,
                                                                                missing):
    """The check precedes the transaction, so a caller bug cannot half-apply the cleanup."""
    candidate = await _terminal_setup(temp_db)
    kwargs = {"dormant_profile_key": PROFILE, "next_attempt_after": "2030-01-01T00:00:00"}
    kwargs[missing] = ""

    with pytest.raises(ValueError):
        await temp_db.terminal_publish_nothing_async(
            team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
            candidate_id=candidate, **kwargs)

    assert await temp_db.load_crawl_checkpoint_async(TEAM, CH, NS) is not None
    assert await temp_db.get_snapshot_async(candidate) is not None


async def test_a_poisoned_outbox_row_rolls_the_whole_terminal_back(temp_db):
    """The outbox insert is INSIDE the transaction, so a payload that fails its six-clause
    validation takes the cleanup down with it rather than half-applying."""
    candidate = await _terminal_setup(temp_db)
    with pytest.raises(ValueError):
        await temp_db.terminal_publish_nothing_async(
            team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
            expect_state="active", dormant_profile_key=PROFILE,
            next_attempt_after="2030-01-01T00:00:00",
            outbox_rows=[_row(_body("publish", event_seq=0))], candidate_id=candidate)

    assert await temp_db.load_crawl_checkpoint_async(TEAM, CH, NS) is not None
    assert await temp_db.get_snapshot_async(candidate) is not None
    assert (await temp_db.load_pending_recompaction_async(TEAM, CH, NS))["state"] == "active"


async def test_profile_outranks_state_when_both_predicates_fail(temp_db):
    """PROFILE IS EVALUATED FIRST, and only a row failing BOTH predicates can prove it.

    §1m defines the state/pair classes as SAME-PROFILE supersession, so a profile disagreement
    can never be honestly reported as one of them. Reporting `state` here would send a task
    sized for a configuration that no longer exists down the leave-it-scheduled route, and the
    obsolescence cleanup would never run.
    """
    candidate = await _terminal_setup(temp_db, profile=OTHER_PROFILE, state="dormant")
    result = await temp_db.terminal_publish_nothing_async(
        team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
        expect_state="active", expect_profile_key=PROFILE, dormant_profile_key=PROFILE,
        next_attempt_after="2030-01-01T00:00:00", candidate_id=candidate)

    assert result == {"ok": False, "mismatch": "profile"}
    await _assert_crawl_state_gone(temp_db, candidate)


async def test_profile_outranks_pair_when_both_predicates_fail(temp_db):
    candidate = await _terminal_setup(temp_db, profile=OTHER_PROFILE)
    result = await temp_db.terminal_publish_nothing_async(
        team_id=TEAM, channel_id=CH, namespace=NS, crawl_id=CRAWL,
        expect_state="active", expect_pair=("nope", 99), expect_profile_key=PROFILE,
        dormant_profile_key=PROFILE, next_attempt_after="2030-01-01T00:00:00",
        candidate_id=candidate)

    assert result == {"ok": False, "mismatch": "profile"}
