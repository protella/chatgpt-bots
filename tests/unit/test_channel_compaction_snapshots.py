"""Channel compaction snapshot store + coordinator (single-stream P1, spec §7).

Publication is a compare-and-swap on a pointer row, not a write to a "current" column, so the
tests that matter are the racing ones: two turns compacting the same channel must produce one
winner, one deleted candidate, and one generation number — never two active snapshots and never
a pointer aimed at a row that got swept.
"""
import asyncio
import sqlite3
import uuid

import pytest

from database import DatabaseManager
from message_processor.channel_snapshots import ChannelSnapshotCoordinator

TEAM = "T1"
CH = "C1"
SV = 1


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    db = DatabaseManager(platform="slack")
    yield db
    db.conn.close()


async def _candidate(db, boundary="1000.000100", team=TEAM, channel=CH, sv=SV, anchors=None):
    snapshot_id = uuid.uuid4().hex
    await db.insert_channel_snapshot_async(snapshot_id, team, channel, sv, boundary,
                                           "summary text", anchors)
    return snapshot_id


async def _publish_chain(db, count):
    """Publish `count` generations in sequence; returns their ids oldest-first."""
    ids = []
    previous = None
    for _ in range(count):
        sid = await _candidate(db)
        assert await db.publish_channel_snapshot_async(TEAM, CH, SV, sid, previous)
        ids.append(sid)
        previous = sid
    return ids


def _backdate(db, snapshot_id, days):
    db.conn.execute(
        "UPDATE channel_snapshots SET created_ts = datetime('now', ?) WHERE snapshot_id = ?",
        (f"-{days} days", snapshot_id))


# --------------------------------------------------------------------- candidates

async def test_candidate_is_unpublished(temp_db):
    sid = await _candidate(temp_db, anchors=[{"root_ts": "1.0", "text": "hi"}])
    row = await temp_db.get_snapshot_async(sid)
    assert row["generation"] is None
    assert row["boundary_ts"] == "1000.000100"
    assert row["root_anchors"] == [{"root_ts": "1.0", "text": "hi"}]
    assert await temp_db.get_active_snapshot_async(TEAM, CH, SV) is None


async def test_many_candidates_coexist(temp_db):
    """The UNIQUE index counts published generations; NULLs are distinct in SQLite."""
    for _ in range(3):
        await _candidate(temp_db)
    assert await temp_db.get_active_snapshot_async(TEAM, CH, SV) is None


# --------------------------------------------------------------------- publication

async def test_genesis_publish_assigns_generation_one(temp_db):
    sid = await _candidate(temp_db)
    assert await temp_db.publish_channel_snapshot_async(TEAM, CH, SV, sid, None)
    active = await temp_db.get_active_snapshot_async(TEAM, CH, SV)
    assert active["snapshot_id"] == sid
    assert active["generation"] == 1


async def test_genesis_double_publish_race_has_one_winner(temp_db):
    first, second = await _candidate(temp_db), await _candidate(temp_db)
    assert await temp_db.publish_channel_snapshot_async(TEAM, CH, SV, first, None)
    assert not await temp_db.publish_channel_snapshot_async(TEAM, CH, SV, second, None)

    assert (await temp_db.get_active_snapshot_async(TEAM, CH, SV))["snapshot_id"] == first
    assert (await temp_db.get_snapshot_async(second))["generation"] is None


async def test_concurrent_genesis_publishes_serialize(temp_db):
    """Both racers really run at once — BEGIN IMMEDIATE is what makes one of them lose."""
    first, second = await _candidate(temp_db), await _candidate(temp_db)
    results = await asyncio.gather(
        temp_db.publish_channel_snapshot_async(TEAM, CH, SV, first, None),
        temp_db.publish_channel_snapshot_async(TEAM, CH, SV, second, None))

    assert sorted(results) == [False, True]
    winner = first if results[0] else second
    assert (await temp_db.get_active_snapshot_async(TEAM, CH, SV))["snapshot_id"] == winner
    generations = [row[0] for row in temp_db.conn.execute(
        "SELECT generation FROM channel_snapshots WHERE generation IS NOT NULL")]
    assert generations == [1]


async def test_generations_increment_and_stale_expectations_lose(temp_db):
    first, second = await _publish_chain(temp_db, 2)
    assert (await temp_db.get_snapshot_async(second))["generation"] == 2

    loser = await _candidate(temp_db)
    assert not await temp_db.publish_channel_snapshot_async(TEAM, CH, SV, loser, first)
    assert (await temp_db.get_active_snapshot_async(TEAM, CH, SV))["snapshot_id"] == second


async def test_republishing_the_active_id_keeps_its_generation(temp_db):
    sid = (await _publish_chain(temp_db, 1))[0]
    assert await temp_db.publish_channel_snapshot_async(TEAM, CH, SV, sid, sid)
    assert (await temp_db.get_snapshot_async(sid))["generation"] == 1


async def test_an_old_generation_cannot_be_republished_over_the_pointer(temp_db):
    """Supplying an already-published snapshot with the live pointer as the expectation would
    otherwise win the CAS and roll the active stream back to a stale summary."""
    first, second = await _publish_chain(temp_db, 2)

    with pytest.raises(ValueError):
        await temp_db.publish_channel_snapshot_async(TEAM, CH, SV, first, second)

    assert (await temp_db.get_active_snapshot_async(TEAM, CH, SV))["snapshot_id"] == second
    assert (await temp_db.get_snapshot_async(first))["generation"] == 1
    assert (await temp_db.get_snapshot_async(second))["generation"] == 2


async def test_a_published_snapshot_is_not_a_candidate_at_genesis_either(temp_db):
    published = (await _publish_chain(temp_db, 1))[0]
    other = await _candidate(temp_db)
    assert await temp_db.publish_channel_snapshot_async(TEAM, CH, SV, other, published)

    with pytest.raises(ValueError):
        await temp_db.publish_channel_snapshot_async(TEAM, CH, SV, published, None)


async def test_unknown_id_is_a_caller_bug(temp_db):
    with pytest.raises(ValueError):
        await temp_db.publish_channel_snapshot_async(TEAM, CH, SV, "nosuchid", None)


async def test_cross_scope_id_is_refused(temp_db):
    """Returning False here would invite the caller to delete another channel's snapshot."""
    other_channel = await _candidate(temp_db, channel="C2")
    with pytest.raises(ValueError):
        await temp_db.publish_channel_snapshot_async(TEAM, CH, SV, other_channel, None)

    other_version = await _candidate(temp_db, sv=2)
    with pytest.raises(ValueError):
        await temp_db.publish_channel_snapshot_async(TEAM, CH, SV, other_version, None)

    other_team = await _candidate(temp_db, team="T2")
    with pytest.raises(ValueError):
        await temp_db.publish_channel_snapshot_async(TEAM, CH, SV, other_team, None)


async def test_invalidated_candidate_loses_instead_of_raising(temp_db):
    sid = await _candidate(temp_db)
    assert await temp_db.invalidate_snapshot_async(sid)
    assert not await temp_db.publish_channel_snapshot_async(TEAM, CH, SV, sid, None)
    assert await temp_db.get_active_snapshot_async(TEAM, CH, SV) is None


async def test_serializer_versions_have_independent_pointers(temp_db):
    v1 = await _candidate(temp_db, sv=1)
    v2 = await _candidate(temp_db, sv=2)
    assert await temp_db.publish_channel_snapshot_async(TEAM, CH, 1, v1, None)
    assert await temp_db.publish_channel_snapshot_async(TEAM, CH, 2, v2, None)
    assert (await temp_db.get_active_snapshot_async(TEAM, CH, 1))["snapshot_id"] == v1
    assert (await temp_db.get_active_snapshot_async(TEAM, CH, 2))["snapshot_id"] == v2
    assert (await temp_db.get_snapshot_async(v2))["generation"] == 1


async def test_generation_uniqueness_is_enforced_by_the_index(temp_db):
    ids = await _publish_chain(temp_db, 2)
    with pytest.raises(sqlite3.IntegrityError):
        temp_db.conn.execute(
            "UPDATE channel_snapshots SET generation = 1 WHERE snapshot_id = ?", (ids[1],))


# --------------------------------------------------------------------- invalidate / delete

async def test_invalidate_is_idempotent_and_visible_to_readers(temp_db):
    sid = (await _publish_chain(temp_db, 1))[0]
    assert await temp_db.invalidate_snapshot_async(sid)
    assert not await temp_db.invalidate_snapshot_async(sid)
    active = await temp_db.get_active_snapshot_async(TEAM, CH, SV)
    assert active["snapshot_id"] == sid
    assert active["invalidated_at"] is not None


async def test_delete_refuses_the_active_snapshot(temp_db):
    sid = (await _publish_chain(temp_db, 1))[0]
    assert not await temp_db.delete_snapshot_async(sid)
    assert await temp_db.get_snapshot_async(sid) is not None


async def test_delete_removes_a_loser_candidate(temp_db):
    sid = await _candidate(temp_db)
    assert await temp_db.delete_snapshot_async(sid)
    assert await temp_db.get_snapshot_async(sid) is None


# --------------------------------------------------------------------- retention sweep

async def test_sweep_keeps_the_newest_generations_and_the_active_row(temp_db):
    ids = await _publish_chain(temp_db, 5)
    for sid in ids:
        _backdate(temp_db, sid, 30)

    assert await temp_db.sweep_snapshots_async([], retain_generations=3, retain_days=7) == 2
    survivors = [sid for sid in ids if await temp_db.get_snapshot_async(sid)]
    assert survivors == ids[2:]


async def test_sweep_keeps_young_snapshots_past_the_generation_cap(temp_db):
    ids = await _publish_chain(temp_db, 5)
    assert await temp_db.sweep_snapshots_async([], retain_generations=3, retain_days=7) == 0
    survivors = [sid for sid in ids if await temp_db.get_snapshot_async(sid)]
    assert survivors == ids


async def test_sweep_spares_pinned_snapshots(temp_db):
    ids = await _publish_chain(temp_db, 5)
    for sid in ids:
        _backdate(temp_db, sid, 30)

    assert await temp_db.sweep_snapshots_async([ids[0]], retain_generations=3,
                                               retain_days=7) == 1
    assert await temp_db.get_snapshot_async(ids[0]) is not None
    assert await temp_db.get_snapshot_async(ids[1]) is None


async def test_sweep_drops_old_candidates_but_not_fresh_ones(temp_db):
    stale, fresh = await _candidate(temp_db), await _candidate(temp_db)
    _backdate(temp_db, stale, 2)
    assert await temp_db.sweep_snapshots_async([]) == 1
    assert await temp_db.get_snapshot_async(stale) is None
    assert await temp_db.get_snapshot_async(fresh) is not None


async def test_sweep_counts_generations_per_scope(temp_db):
    """Another channel's generation numbers must not age this channel's out."""
    mine = await _publish_chain(temp_db, 1)
    _backdate(temp_db, mine[0], 30)
    for _ in range(5):
        sid = uuid.uuid4().hex
        await temp_db.insert_channel_snapshot_async(sid, TEAM, "C2", SV, "1.0", "s")
        assert await temp_db.publish_channel_snapshot_async(
            TEAM, "C2", SV, sid,
            (await temp_db.get_active_snapshot_async(TEAM, "C2", SV) or {}).get("snapshot_id"))

    await temp_db.sweep_snapshots_async([], retain_generations=3, retain_days=7)
    assert await temp_db.get_snapshot_async(mine[0]) is not None


# --------------------------------------------------------------------- coordinator

def _coordinator(db):
    return ChannelSnapshotCoordinator(db, retain_generations=3, retain_days=7)


async def _coord_candidate(coord, db, text="summary", *, boundary="1000.0"):
    """A candidate carrying the channel's CURRENT effective sizing profile.

    The final-profile predicate compares the candidate's `sizing_profile` against the live one
    inside the publication transaction, so a candidate that never recorded one can never be
    published — see the test below, which pins that deliberately.
    """
    profile = (await coord.sizing_for(TEAM, CH))["sizing_profile"]
    payload = text.encode("utf-8")
    return await db.insert_compaction_candidate_async(snapshot={
        "team_id": TEAM, "channel_id": CH, "namespace": PROD_NAMESPACE,
        "serializer_version": SV, "boundary_ts": boundary, "source_floor_ts": "1.0",
        "prompt_version": "v1", "model": "gpt-5.6-luna", "source_hash": "sh",
        "payload_bytes": payload, "mutation_frontier": 0, "headroom_source": "measured",
        "headroom_tokens": 1000, "effective_window": 400000, "sizing_profile": profile,
        "fit_result": "under_target"})


async def _coord_publish(coord, sid, previous, *, boundary="1000.0"):
    """`publish()` is keyword-only and returns the §1d result dict, not a bool."""
    return await coord.publish(
        team_id=TEAM, channel_id=CH, serializer_version=SV, snapshot_id=sid,
        expected_previous_id=previous, source_floor_ts="1.0", boundary_ts=boundary,
        mutation_frontier=0)


async def test_coordinator_publishes_and_keeps_the_winner(temp_db):
    coord = _coordinator(temp_db)
    sid = await _coord_candidate(coord, temp_db)
    assert (await _coord_publish(coord, sid, None))["won"]
    assert (await coord.get_active(TEAM, CH, SV))["snapshot_id"] == sid


async def test_coordinator_deletes_the_loser_candidate(temp_db):
    coord = _coordinator(temp_db)
    winner = await _coord_candidate(coord, temp_db, "won")
    loser = await _coord_candidate(coord, temp_db, "lost")

    assert (await _coord_publish(coord, winner, None))["won"]
    assert not (await _coord_publish(coord, loser, None))["won"]
    assert await temp_db.get_snapshot_async(loser) is None
    assert await temp_db.get_snapshot_async(winner) is not None


async def test_a_candidate_with_no_sizing_profile_cannot_be_published(temp_db):
    """`insert_candidate` builds the P1 v1-shaped row, which records no sizing evidence. The
    final-profile predicate therefore refuses it — a v2 generation published without its
    evidence would be permanently undominating for no reason anyone intended."""
    coord = _coordinator(temp_db)
    sid = await coord.insert_candidate(TEAM, CH, SV, "1000.0", "summary")
    result = await _coord_publish(coord, sid, None)
    assert result == {"won": False, "reason": "profile_changed", "generation": None}
    assert await temp_db.get_snapshot_async(sid) is None


async def test_coordinator_channel_lock_is_stable_per_channel(temp_db):
    coord = _coordinator(temp_db)
    assert coord.channel_lock(TEAM, CH) is coord.channel_lock(TEAM, CH)
    assert coord.channel_lock(TEAM, CH) is not coord.channel_lock(TEAM, "C2")


async def test_coordinator_pins_survive_the_sweep(temp_db):
    coord = _coordinator(temp_db)
    ids = await _publish_chain(temp_db, 5)
    for sid in ids:
        _backdate(temp_db, sid, 30)

    async with coord.pinned(ids[0]):
        assert coord.pinned_ids() == [ids[0]]
        assert await coord.sweep() == 1
        assert await temp_db.get_snapshot_async(ids[0]) is not None

    assert coord.pinned_ids() == []
    assert await coord.sweep() == 1
    assert await temp_db.get_snapshot_async(ids[0]) is None


async def test_coordinator_pin_refcount_holds_until_the_last_reader(temp_db):
    coord = _coordinator(temp_db)
    await coord.pin("abc")
    await coord.pin("abc")
    await coord.unpin("abc")
    assert coord.pinned_ids() == ["abc"]
    await coord.unpin("abc")
    assert coord.pinned_ids() == []
    # Unbalanced unpin is a no-op, never a negative refcount.
    await coord.unpin("abc")
    assert coord.pinned_ids() == []


async def test_coordinator_invalidate_delegates(temp_db):
    coord = _coordinator(temp_db)
    sid = (await _publish_chain(temp_db, 1))[0]
    assert await coord.invalidate(sid)
    assert (await coord.get_active(TEAM, CH, SV))["invalidated_at"] is not None


# ------------------------------------------------- select_and_pin (P4-ready, P2 unit-only)
# P2 callers never reach this: ANY active pointer fails a P2 turn closed. It lands
# API-complete now because the selection and the pin have to be ONE step — a coordinator that
# chose a snapshot and then pinned it would let the sweep delete the row in between, and that
# is not a bug a P4 test would find reliably.

async def test_select_and_pin_returns_the_active_snapshot_and_pins_it(temp_db):
    coord = _coordinator(temp_db)
    sid = (await _publish_chain(temp_db, 1))[0]
    selected = await coord.select_and_pin(TEAM, CH, SV)
    assert selected["snapshot_id"] == sid
    assert coord.pinned_ids() == [sid]


async def test_select_and_pin_reports_genesis_rather_than_none(temp_db):
    """The §1b result dict ALWAYS carries a tag. `None` was the P1 shape and is retired: it
    could not tell genesis apart from an invalidated lineage, which is the whole distinction
    §1b exists to make."""
    coord = _coordinator(temp_db)
    result = await coord.select_and_pin(TEAM, CH, SV)
    assert result["result"] == "genesis"
    assert result["snapshot_id"] is None
    assert coord.pinned_ids() == []


async def test_select_and_pin_refuses_an_invalidated_pointer(temp_db):
    """An invalidated generation with no valid ancestor is raw_rebuild_required, NOT genesis —
    and it is never pinned, because nothing will ever render it."""
    coord = _coordinator(temp_db)
    sid = (await _publish_chain(temp_db, 1))[0]
    await coord.invalidate(sid)
    result = await coord.select_and_pin(TEAM, CH, SV)
    assert result["result"] == "raw_rebuild_required"
    assert result["snapshot_id"] == sid
    assert coord.pinned_ids() == []


async def test_select_and_pin_honours_max_boundary(temp_db):
    coord = _coordinator(temp_db)
    sid = await _candidate(temp_db, boundary="1000.000100")
    assert await temp_db.publish_channel_snapshot_async(TEAM, CH, SV, sid, None)
    early = await coord.select_and_pin(TEAM, CH, SV, max_boundary="999.000000")
    assert early["result"] == "no_eligible_generation"
    assert coord.pinned_ids() == []
    assert (await coord.select_and_pin(TEAM, CH, SV,
                                       max_boundary="1000.000100"))["snapshot_id"] == sid


async def test_select_and_pin_compares_boundaries_numerically(temp_db):
    """"1000.5" > "1000.10" numerically and the other way round as strings."""
    coord = _coordinator(temp_db)
    sid = await _candidate(temp_db, boundary="1000.5")
    assert await temp_db.publish_channel_snapshot_async(TEAM, CH, SV, sid, None)
    assert (await coord.select_and_pin(
        TEAM, CH, SV, max_boundary="1000.10"))["result"] == "no_eligible_generation"


async def test_select_and_pin_cannot_lose_its_row_to_the_sweep(temp_db):
    coord = _coordinator(temp_db)
    ids = await _publish_chain(temp_db, 4)
    for sid in ids[:-1]:
        _backdate(temp_db, sid, 30)
    selected = await coord.select_and_pin(TEAM, CH, SV)
    await coord.sweep()
    assert await temp_db.get_snapshot_async(selected["snapshot_id"]) is not None
    assert coord.pinned_ids() == [selected["snapshot_id"]]


# =====================================================================================
# P4 §1b/§1d/§1f/§1j — selection, publication, retirement, and the SIX physical-delete
# sites. Everything below runs against serializer v2 in the production namespace.
# =====================================================================================

import hashlib                                                          # noqa: E402

from database import PROD_NAMESPACE, SNAPSHOT_SIZING_FIELDS             # noqa: E402

NS = PROD_NAMESPACE
V2 = 2
PROFILE = "gpt-5.6-luna:400000:320000:280000"
OTHER_PROFILE = "gpt-5.6-luna:400000:300000:260000"
ANCHOR_ROOT = "0.500000"          # deliberately BELOW source_floor_ts


def _v2(**kw):
    body = kw.pop("payload_text", "the summary").encode("utf-8")
    base = dict(team_id=TEAM, channel_id=CH, namespace=NS, serializer_version=V2,
                boundary_ts="1000.000100", source_floor_ts="10.000000",
                parent_snapshot_id=None, prompt_version="v1", model="gpt-5.6-luna",
                source_hash="sh", payload_bytes=body, anchor_payload_bytes=b"anchors",
                mutation_frontier=0, headroom_source="measured", headroom_tokens=90000,
                effective_window=400000, sizing_profile=PROFILE, fit_result="under_target")
    base.update(kw)
    return base


def _manifest(row_id="7", status="ready"):
    return [{"artifact_namespace": "image_analysis", "row_id": row_id, "source_ts": "50.0",
             "captured_render_version": "v1", "content_hash": "content-hash",
             "status_at_capture": status}]


def _anchors(root_ts=ANCHOR_ROOT, status="available", frontier=0, proof="not_self"):
    return [{"root_ts": root_ts, "status": status, "projection_sha256": "rendered-bytes",
             "observation_frontier": frontier, "receipt_proof": proof}]


async def _candidate_v2(db, *, manifest=None, anchors=None, **kw):
    return await db.insert_compaction_candidate_async(
        snapshot=_v2(**kw),
        manifest_rows=_manifest() if manifest is None else manifest,
        anchor_rows=_anchors() if anchors is None else anchors)


async def _publish_v2(db, sid, previous=None, *, profile=PROFILE, frontier=0, **kw):
    return await db.publish_compaction_candidate_async(
        team_id=TEAM, channel_id=CH, namespace=NS, serializer_version=V2, snapshot_id=sid,
        expected_previous_id=previous, source_floor_ts="10.000000",
        boundary_ts="1000.000100", mutation_frontier=frontier, current_profile=profile, **kw)


async def _observe(db, subject_ts, identity="E1", kind="edit"):
    await db.record_activity_and_mutation_async(
        observation=None,
        mutation={"team_id": TEAM, "channel_id": CH, "subject_ts": subject_ts, "kind": kind,
                  "observation_identity": identity, "observed_at": "1.0"})


async def _chain_v2(db, count):
    """Publish `count` v2 generations in sequence; returns their ids oldest-first."""
    ids = []
    previous = None
    for i in range(count):
        sid = await _candidate_v2(db, boundary_ts=f"{1000 + i}.000100",
                                  payload_text=f"summary {i}",
                                  parent_snapshot_id=previous)
        assert (await _publish_v2(db, sid, previous))["won"]
        ids.append(sid)
        previous = sid
    return ids


# --------------------------------------------------------------------- selection (§1b)

async def test_selection_reports_genesis_only_when_nothing_was_ever_published(temp_db):
    result = await temp_db.select_snapshot_for_pin_async(TEAM, CH, NS, V2, None)
    assert result["result"] == "genesis"
    assert result["snapshot"] is None and result["snapshot_id"] is None


async def test_selection_pins_the_newest_valid_generation(temp_db):
    ids = await _chain_v2(temp_db, 3)
    result = await temp_db.select_snapshot_for_pin_async(TEAM, CH, NS, V2, None)
    assert result["result"] == "pinned"
    assert result["snapshot_id"] == ids[-1]
    assert result["generation"] == 3
    assert result["snapshot"]["payload_bytes"] == b"summary 2"


async def test_a_stale_generation_is_distinguishable_from_a_valid_one(temp_db):
    sid = await _candidate_v2(temp_db)
    assert (await _publish_v2(temp_db, sid, None, status="published_stale"))["won"]
    result = await temp_db.select_snapshot_for_pin_async(TEAM, CH, NS, V2, None)
    assert result["result"] == "pinned_stale"
    assert result["snapshot_id"] == sid


async def test_an_invalidated_generation_with_no_ancestor_is_never_reported_as_genesis(temp_db):
    """§1b: raw_rebuild_required RETAINS the identity, for telemetry and for the publication
    CAS. Reporting genesis would claim nothing was ever published, which is false."""
    sid = (await _chain_v2(temp_db, 1))[0]
    await temp_db.invalidate_snapshot_async(sid)

    result = await temp_db.select_snapshot_for_pin_async(TEAM, CH, NS, V2, None)
    assert result["result"] == "raw_rebuild_required"
    assert result["snapshot_id"] == sid
    assert result["generation"] == 1


async def test_an_invalidated_descendant_falls_back_to_its_valid_ancestor(temp_db):
    first, second = await _chain_v2(temp_db, 2)
    await temp_db.invalidate_snapshot_async(second)
    result = await temp_db.select_snapshot_for_pin_async(TEAM, CH, NS, V2, None)
    assert (result["result"], result["snapshot_id"]) == ("pinned", first)


async def test_refused_lineage_is_excluded_from_selection(temp_db):
    """Phase 1 of retirement marks the lineage retirement-pending; new selection REFUSES it
    from that moment, before the deletion transaction runs."""
    first, second = await _chain_v2(temp_db, 2)
    result = await temp_db.select_snapshot_for_pin_async(
        TEAM, CH, NS, V2, None, refused_lineage=[second])
    assert (result["result"], result["snapshot_id"]) == ("pinned", first)

    result = await temp_db.select_snapshot_for_pin_async(
        TEAM, CH, NS, V2, None, refused_lineage=[first, second])
    assert result["result"] == "raw_rebuild_required"


async def test_selection_respects_max_boundary_numerically(temp_db):
    """"1000.5" > "1000.10" numerically and the other way round as strings."""
    sid = await _candidate_v2(temp_db, boundary_ts="1000.5")
    assert (await temp_db.publish_compaction_candidate_async(
        team_id=TEAM, channel_id=CH, namespace=NS, serializer_version=V2, snapshot_id=sid,
        expected_previous_id=None, source_floor_ts="10.0", boundary_ts="1000.5",
        mutation_frontier=0, current_profile=PROFILE))["won"]

    assert (await temp_db.select_snapshot_for_pin_async(
        TEAM, CH, NS, V2, "1000.10"))["result"] == "no_eligible_generation"
    assert (await temp_db.select_snapshot_for_pin_async(
        TEAM, CH, NS, V2, "1000.5"))["result"] == "pinned"


async def test_an_ineligible_generation_is_not_reported_as_genesis(temp_db):
    """OWNER RULING: a channel holding a summary that is merely NOT YET ELIGIBLE under this
    turn's H is a different CONTROL state from one that has never been compacted at all, and
    collapsing them would make the first invisible to telemetry.

    A test that accepts either tag here fails the point of the ruling, so the two are asserted
    to be DIFFERENT — and the ineligible one RETAINS the identity telemetry has to name.
    """
    never_compacted = await temp_db.select_snapshot_for_pin_async(TEAM, CH, NS, V2, None)
    assert never_compacted["result"] == "genesis"
    assert never_compacted["snapshot_id"] is None

    sid = (await _chain_v2(temp_db, 1))[0]
    ineligible = await temp_db.select_snapshot_for_pin_async(TEAM, CH, NS, V2, "500.000000")

    assert ineligible["result"] == "no_eligible_generation"
    assert ineligible["result"] != never_compacted["result"]
    assert ineligible["snapshot_id"] == sid
    assert ineligible["generation"] == 1
    assert ineligible["snapshot"] is not None


async def test_an_ineligible_generation_is_not_a_refuse_to_render_state(temp_db):
    """It RENDERS, exactly as genesis does — no summary block, honest coverage floor. Only
    raw_rebuild_required and payload_corrupt are the results the caller must never hand to the
    builder, and this is not a third."""
    await _chain_v2(temp_db, 1)
    result = await temp_db.select_snapshot_for_pin_async(TEAM, CH, NS, V2, "500.000000")
    assert result["result"] not in ("raw_rebuild_required", "payload_corrupt")


async def test_raw_rebuild_required_outranks_the_ineligible_tag(temp_db):
    """The tag is for "every generation is VALID, just not yet eligible" ONLY.

    The discriminating case needs BOTH conditions live at once: one generation INVALIDATED, and
    another VALID but above the ceiling. A recompaction really is owed, so raw_rebuild_required
    must win — a turn told everything is merely early would never repair the invalidation.
    Invalidating every generation does NOT test this: the ceiling branch is then never reached.
    """
    first, second = await _chain_v2(temp_db, 2)          # boundaries 1000.000100 / 1001.000100
    await temp_db.invalidate_snapshot_async(first)

    ceiling = "1000.500000"                              # excludes `second`, not `first`
    assert (await temp_db.get_snapshot_row_async(second))["status"] == "published"

    result = await temp_db.select_snapshot_for_pin_async(TEAM, CH, NS, V2, ceiling)
    assert result["result"] == "raw_rebuild_required"

    # And with the invalidation lifted, the SAME ceiling yields the ineligible tag — proving
    # the two branches are distinguished by validity, not by the ceiling alone.
    temp_db.conn.execute(
        "UPDATE channel_snapshots SET status = 'published', invalidated_at = NULL "
        "WHERE snapshot_id = ?", (first,))
    unblocked = await temp_db.select_snapshot_for_pin_async(TEAM, CH, NS, V2, ceiling)
    assert unblocked["result"] == "pinned" and unblocked["snapshot_id"] == first


async def test_a_newer_boundary_loses_to_an_older_H(temp_db):
    """§7b: newer-boundary/older-H selection. A turn whose H predates the newest generation's
    boundary pins the older generation, not the newer one."""
    first, second = await _chain_v2(temp_db, 2)
    result = await temp_db.select_snapshot_for_pin_async(TEAM, CH, NS, V2, "1000.000100")
    assert result["snapshot_id"] == first


async def test_corrupted_payload_bytes_return_payload_corrupt(temp_db, caplog):
    """Test 20: a snapshot whose persisted bytes no longer match `payload_hash` returns
    payload_corrupt, is excluded from selection, logs CRITICAL, and is NEVER rendered."""
    sid = (await _chain_v2(temp_db, 1))[0]
    temp_db.conn.execute(
        "UPDATE channel_snapshots SET payload_bytes = ? WHERE snapshot_id = ?",
        (b"bytes nobody generated", sid))

    with caplog.at_level("ERROR"):
        result = await temp_db.select_snapshot_for_pin_async(TEAM, CH, NS, V2, None)
    assert result["result"] == "payload_corrupt"
    assert result["snapshot_id"] == sid                       # identity RETAINED
    assert any("CRITICAL" in r.message and sid in r.message for r in caplog.records)


async def test_a_corrupt_generation_does_not_masquerade_as_stale_or_pinned(temp_db):
    """payload_corrupt is a FIFTH distinguishable result, not an error folded into another."""
    sid = (await _chain_v2(temp_db, 1))[0]
    temp_db.conn.execute(
        "UPDATE channel_snapshots SET payload_hash = 'not-the-hash' WHERE snapshot_id = ?",
        (sid,))
    result = await temp_db.select_snapshot_for_pin_async(TEAM, CH, NS, V2, None)
    assert result["result"] == "payload_corrupt"


async def test_selection_verifies_the_hash_it_stored(temp_db):
    sid = (await _chain_v2(temp_db, 1))[0]
    row = await temp_db.get_snapshot_row_async(sid)
    assert row["payload_hash"] == hashlib.sha256(row["payload_bytes"]).hexdigest()


# --------------------------------------------------------------- candidate insert (§1g)

@pytest.mark.parametrize("field", SNAPSHOT_SIZING_FIELDS)
async def test_a_v2_candidate_missing_any_sizing_field_is_rejected(temp_db, field):
    """Test 99: the columns are nullable FOR MIGRATION, not for new writes. A test that
    publishes a v2 generation with NULL fit_result and expects success is asserting the
    retired rule."""
    with pytest.raises(ValueError, match="sizing evidence"):
        await temp_db.insert_compaction_candidate_async(snapshot=_v2(**{field: None}))
    assert temp_db.conn.execute(
        "SELECT COUNT(*) FROM channel_snapshots").fetchone()[0] == 0


async def test_a_v1_candidate_may_still_carry_no_sizing_evidence(temp_db):
    """NULL stays legal on the legacy serializer — those rows predate the evidence."""
    sid = await temp_db.insert_compaction_candidate_async(
        snapshot=_v2(serializer_version=1, headroom_source=None, headroom_tokens=None,
                     effective_window=None, sizing_profile=None, fit_result=None))
    assert (await temp_db.get_snapshot_row_async(sid))["fit_result"] is None


async def test_candidate_manifest_and_anchors_commit_as_one_transaction(temp_db):
    """Test 14: nothing is inserted until validation passes, and then all three row types go
    in together — which is why an ordinary discarded candidate has no rows to clean up."""
    sid = await _candidate_v2(temp_db)
    assert (await temp_db.get_snapshot_row_async(sid))["status"] == "candidate"
    assert len(await temp_db.snapshot_manifest_async(sid)) == 1
    assert len(await temp_db.snapshot_anchor_provenance_async(sid)) == 1


async def test_a_rejected_candidate_writes_no_manifest_rows(temp_db):
    with pytest.raises(ValueError):
        await temp_db.insert_compaction_candidate_async(
            snapshot=_v2(fit_result=None), manifest_rows=_manifest(),
            anchor_rows=_anchors())
    assert temp_db.conn.execute(
        "SELECT COUNT(*) FROM snapshot_capture_manifest").fetchone()[0] == 0
    assert temp_db.conn.execute(
        "SELECT COUNT(*) FROM snapshot_anchor_provenance").fetchone()[0] == 0


# --------------------------------------------------------------- anchor provenance (§1j)

async def test_anchor_rows_record_status_fingerprint_and_frontier(temp_db):
    """Test 89: every anchored root gets a row, INCLUDING the ones that rendered
    [root unavailable] — a missing row would mean "this snapshot never anchored that thread",
    which is false and would let a later mutation of that root go unnoticed."""
    sid = await _candidate_v2(temp_db, anchors=[
        {"root_ts": "0.5", "status": "available", "projection_sha256": "quoted-text-bytes",
         "observation_frontier": 3, "receipt_proof": "not_self"},
        {"root_ts": "0.6", "status": "unavailable",
         "projection_sha256": "root-unavailable-bytes", "observation_frontier": 3,
         "receipt_proof": None},
        {"root_ts": "0.7", "status": "unsafe", "projection_sha256": "root-unavailable-bytes",
         "observation_frontier": 3, "receipt_proof": None},
        {"root_ts": "0.8", "status": "refused", "projection_sha256": "root-unavailable-bytes",
         "observation_frontier": 3, "receipt_proof": None}])

    rows = {r["root_ts"]: r for r in await temp_db.snapshot_anchor_provenance_async(sid)}
    assert set(rows) == {"0.5", "0.6", "0.7", "0.8"}
    assert rows["0.5"]["receipt_proof"] == "not_self"
    # NULL means exactly one thing: no proof because the root was never admitted.
    assert rows["0.6"]["receipt_proof"] is None
    assert rows["0.6"]["projection_sha256"] == "root-unavailable-bytes"


async def test_a_fetched_tombstone_is_available_not_unavailable(temp_db):
    """Test 89: Slack RETURNED the root, the normalizer represents it, and the serializer
    renders its [root deleted] marker — that is a SUCCESSFUL anchor carrying real evidence.
    `unavailable` means something narrower: Slack will not give us the root at all."""
    sid = await _candidate_v2(temp_db, anchors=[
        {"root_ts": "0.5", "status": "available",
         "projection_sha256": "root-deleted-marker-bytes", "observation_frontier": 0,
         "receipt_proof": "not_self"}])
    row = (await temp_db.snapshot_anchor_provenance_async(sid))[0]
    assert row["status"] == "available"
    assert row["projection_sha256"] == "root-deleted-marker-bytes"


async def test_an_unknown_anchor_status_is_refused_by_the_schema(temp_db):
    with pytest.raises(sqlite3.IntegrityError):
        await temp_db.insert_compaction_candidate_async(
            snapshot=_v2(), anchor_rows=[
                {"root_ts": "0.5", "status": "maybe", "projection_sha256": "x",
                 "observation_frontier": 0}])


async def test_an_anchor_root_mutation_invalidates_a_generation_that_predates_it(temp_db):
    """§1c + §1j: the scope predicate is EXTENDED. An anchor root may sit BELOW
    source_floor_ts, so the span predicate structurally cannot reach it — but the anchor is
    evidence the snapshot RENDERED, so it is inside the correctness envelope."""
    sid = (await _chain_v2(temp_db, 1))[0]
    assert float(ANCHOR_ROOT) < float("10.000000")            # genuinely pre-floor

    affected = await temp_db.affected_snapshot_ids_async(TEAM, CH, NS, ANCHOR_ROOT)
    assert affected == [sid]

    in_span = await temp_db.affected_snapshot_ids_async(TEAM, CH, NS, "500.0")
    assert in_span == [sid]
    assert await temp_db.affected_snapshot_ids_async(TEAM, CH, NS, "5000.0") == []


async def test_every_generation_covering_a_mutation_is_affected_not_only_the_active_one(
        temp_db):
    """Test 1: an ancestor and its descendant are BOTH excluded after one in-span mutation.
    Falling back to an ancestor that summarized the same source would silently restore the
    lie."""
    ids = await _chain_v2(temp_db, 2)
    assert await temp_db.affected_snapshot_ids_async(TEAM, CH, NS, "500.0") == ids


# --------------------------------------------------------------- publication (§1d, §1m)

async def test_the_final_profile_predicate_fails_closed(temp_db):
    """§1m: a profile can change during the final map/reduce, after the last chunk boundary has
    already passed — at which point no further boundary exists to cancel at."""
    sid = await _candidate_v2(temp_db)
    result = await _publish_v2(temp_db, sid, None, profile=OTHER_PROFILE)
    assert result == {"won": False, "reason": "profile_changed", "generation": None}
    assert await temp_db.select_snapshot_for_pin_async(
        TEAM, CH, NS, V2, None) == {"result": "genesis", "snapshot": None,
                                    "snapshot_id": None, "generation": None}


async def test_an_in_span_mutation_above_the_frontier_discards_the_candidate(temp_db):
    await _observe(temp_db, "500.0")
    sid = await _candidate_v2(temp_db)
    result = await _publish_v2(temp_db, sid, None, frontier=0)
    assert result["reason"] == "frontier"


async def test_a_mutation_below_the_frontier_does_not_discard(temp_db):
    await _observe(temp_db, "500.0")
    frontier = await temp_db.max_mutation_observation_id_async(TEAM, CH)
    sid = await _candidate_v2(temp_db)
    assert (await _publish_v2(temp_db, sid, None, frontier=frontier))["won"]


async def test_an_out_of_span_mutation_does_not_discard(temp_db):
    await _observe(temp_db, "5000.0")
    sid = await _candidate_v2(temp_db)
    assert (await _publish_v2(temp_db, sid, None, frontier=0))["won"]


async def test_a_pre_floor_anchor_mutation_the_span_predicate_cannot_see_still_discards(
        temp_db):
    """§1d predicate 3. The anchor root PREDATES source_floor_ts, so predicate 2 passes and
    only the anchor predicate can catch it. This is the whole reason there are two."""
    await _observe(temp_db, ANCHOR_ROOT)
    sid = await _candidate_v2(temp_db, anchors=_anchors(frontier=0))
    result = await _publish_v2(temp_db, sid, None, frontier=0)
    assert result["reason"] == "anchor"


async def test_an_anchor_mutation_below_that_roots_own_frontier_does_not_discard(temp_db):
    await _observe(temp_db, ANCHOR_ROOT)
    frontier = await temp_db.max_mutation_observation_id_async(TEAM, CH)
    sid = await _candidate_v2(temp_db, anchors=_anchors(frontier=frontier))
    assert (await _publish_v2(temp_db, sid, None, frontier=0))["won"]


async def test_a_lost_cas_reports_cas_and_leaves_the_winner_alone(temp_db):
    winner = await _candidate_v2(temp_db)
    loser = await _candidate_v2(temp_db, payload_text="lost")
    assert (await _publish_v2(temp_db, winner, None))["won"]

    result = await _publish_v2(temp_db, loser, None)
    assert result == {"won": False, "reason": "cas", "generation": None}
    assert (await temp_db.select_snapshot_for_pin_async(
        TEAM, CH, NS, V2, None))["snapshot_id"] == winner


async def test_concurrent_publications_produce_exactly_one_generation(temp_db):
    first = await _candidate_v2(temp_db)
    second = await _candidate_v2(temp_db, payload_text="other")
    results = await asyncio.gather(_publish_v2(temp_db, first, None),
                                   _publish_v2(temp_db, second, None))
    assert sorted(r["won"] for r in results) == [False, True]
    generations = [row[0] for row in temp_db.conn.execute(
        "SELECT generation FROM channel_snapshots WHERE generation IS NOT NULL")]
    assert generations == [1]


async def test_publication_writes_its_outbox_rows_in_the_same_commit(temp_db):
    """§1l: the state change and its telemetry commit as ONE SQLite transaction."""
    sid = await _candidate_v2(temp_db)
    at = 1700000000.5
    build = {"event": "compaction_snapshot", "crawl_id": "c9", "attempt_seq": 0,
             "event_seq": 0, "team_id": TEAM, "channel_id": CH, "namespace": NS, "at": at,
             "op": "build", "model": "m", "tokens_in": 1, "tokens_out": 2,
             "cached_input_tokens": 0, "call_count": 1, "status": "ok"}
    publish = {"event": "compaction_snapshot", "crawl_id": "c9", "attempt_seq": 0,
               "event_seq": 1, "team_id": TEAM, "channel_id": CH, "namespace": NS, "at": at,
               "op": "publish", "snapshot_id": sid, "generation": 1,
               "boundary_ts": "1000.000100", "fit_result": "under_target",
               "serializer_version": V2}
    assert (await _publish_v2(temp_db, sid, None, outbox_rows=[
        {"crawl_id": "c9", "attempt_seq": 0, "event_seq": 0, "body": build},
        {"crawl_id": "c9", "attempt_seq": 0, "event_seq": 1, "body": publish}]))["won"]

    batch = await temp_db.read_outbox_batch_async()
    assert [r["body"]["op"] for r in batch] == ["build", "publish"]


async def test_a_conflicting_outbox_body_rolls_the_publication_back(temp_db):
    """§1l: two distinct events claiming one identity is a defect at its source, and the
    enclosing state change must not commit on top of it."""
    at = 1700000000.5
    build = {"event": "compaction_snapshot", "crawl_id": "c9", "attempt_seq": 0,
             "event_seq": 0, "team_id": TEAM, "channel_id": CH, "namespace": NS, "at": at,
             "op": "build", "model": "m", "tokens_in": 1, "tokens_out": 2,
             "cached_input_tokens": 0, "call_count": 1, "status": "ok"}
    await temp_db.insert_outbox_rows_async(
        [{"crawl_id": "c9", "attempt_seq": 0, "event_seq": 0, "body": build}])

    sid = await _candidate_v2(temp_db)
    with pytest.raises(ValueError):
        await _publish_v2(temp_db, sid, None, outbox_rows=[
            {"crawl_id": "c9", "attempt_seq": 0, "event_seq": 0,
             "body": {**build, "tokens_in": 999}}])

    assert (await temp_db.get_snapshot_row_async(sid))["generation"] is None
    assert (await temp_db.select_snapshot_for_pin_async(
        TEAM, CH, NS, V2, None))["result"] == "genesis"


# ---------------------------------------------------- profile-bound satisfaction (§1m)

async def _owed_generation(db):
    """Publish generation 1 and raise an obligation against it, so the SATISFYING publication
    is generation 2 — satisfaction requires a STRICTLY GREATER generation."""
    return (await _chain_v2(db, 1))[0]


async def _owe(db, *, profile=PROFILE, headroom=90000, obligated="s1", generation=1):
    await db.merge_pending_recompaction_async(
        team_id=TEAM, channel_id=CH, namespace=NS, profile_key=profile,
        required_headroom=headroom, obligated_snapshot_id=obligated,
        obligated_generation=generation, reason="fixed_headroom")


async def test_a_publication_meeting_the_requirement_discharges_it(temp_db):
    prior = await _owed_generation(temp_db)
    await _owe(temp_db, headroom=50000)
    sid = await _candidate_v2(temp_db, boundary_ts="2000.0", headroom_tokens=90000)
    assert (await _publish_v2(temp_db, sid, prior, satisfy={}))["won"]
    assert await temp_db.load_pending_recompaction_async(TEAM, CH, NS) is None


async def test_a_smaller_fixed_publication_does_not_discharge_a_measured_requirement(temp_db):
    """Test 91: superseding the generation without meeting the sizing standard is exactly the
    case the obligation exists to catch."""
    prior = await _owed_generation(temp_db)
    await _owe(temp_db, headroom=90000)
    sid = await _candidate_v2(temp_db, boundary_ts="2000.0", headroom_source="fixed",
                              headroom_tokens=40000)
    assert (await _publish_v2(temp_db, sid, prior, satisfy={}))["won"]
    assert (await temp_db.load_pending_recompaction_async(
        TEAM, CH, NS))["requirements"] == {PROFILE: 90000}


async def test_a_different_profile_key_discharges_nothing(temp_db):
    """Test 91: including when only the trigger/target THRESHOLDS changed — the key is the
    full four-part profile, because a publication proved to fit under a LOOSER target says
    nothing about fit under the stricter one that produced the obligation."""
    prior = await _owed_generation(temp_db)
    await _owe(temp_db, profile=OTHER_PROFILE, headroom=10)
    sid = await _candidate_v2(temp_db, boundary_ts="2000.0", sizing_profile=PROFILE,
                              headroom_tokens=900000)
    assert (await _publish_v2(temp_db, sid, prior, satisfy={}))["won"]
    assert (await temp_db.load_pending_recompaction_async(
        TEAM, CH, NS))["requirements"] == {OTHER_PROFILE: 10}


async def test_an_under_trigger_publication_discharges_nothing(temp_db):
    """Test 91: the under-trigger fallback is the escape hatch for a channel that cannot get
    under target AT ALL; the obligation exists precisely to restore under-target fit."""
    prior = await _owed_generation(temp_db)
    await _owe(temp_db, headroom=10)
    sid = await _candidate_v2(temp_db, boundary_ts="2000.0", fit_result="under_trigger",
                              headroom_tokens=900000)
    assert (await _publish_v2(temp_db, sid, prior, satisfy={}))["won"]
    assert (await temp_db.load_pending_recompaction_async(
        TEAM, CH, NS))["requirements"] == {PROFILE: 10}


async def test_only_the_dominated_entries_go_and_the_row_survives(temp_db):
    """Test 85: entries under other keys SURVIVE, and the ROW disappears only when the map
    empties."""
    prior = await _owed_generation(temp_db)
    await _owe(temp_db, profile=PROFILE, headroom=10)
    await _owe(temp_db, profile=OTHER_PROFILE, headroom=10)
    sid = await _candidate_v2(temp_db, boundary_ts="2000.0", headroom_tokens=90000)
    assert (await _publish_v2(temp_db, sid, prior, satisfy={}))["won"]

    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert row["requirements"] == {OTHER_PROFILE: 10}


async def test_a_generation_not_greater_than_the_obligated_one_discharges_nothing(temp_db):
    """A publication cannot discharge an obligation raised by a LATER generation."""
    prior = await _owed_generation(temp_db)
    await _owe(temp_db, headroom=10, generation=5)
    sid = await _candidate_v2(temp_db, boundary_ts="2000.0", headroom_tokens=90000)
    assert (await _publish_v2(temp_db, sid, prior, satisfy={}))["won"]
    assert (await temp_db.load_pending_recompaction_async(
        TEAM, CH, NS))["requirements"] == {PROFILE: 10}


async def test_an_undominated_publication_goes_dormant_in_the_same_commit(temp_db):
    """§1m CRASH-ATOMIC: writing them separately would let a crash leave an ACTIVE row behind
    a publication that already failed to dominate, and boot would retry it with the backoff
    unwritten."""
    prior = await _owed_generation(temp_db)
    await _owe(temp_db, headroom=900000)
    sid = await _candidate_v2(temp_db, boundary_ts="2000.0", fit_result="under_trigger")
    assert (await _publish_v2(temp_db, sid, prior, satisfy={}, dormancy={
        "profile_key": PROFILE, "next_attempt_after": "2030-01-01T00:00:00"}))["won"]

    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert (row["state"], row["dormant_profile_key"]) == ("dormant", PROFILE)
    assert row["next_attempt_after"] == "2030-01-01T00:00:00"


# --------------------------------------------------- retirement and rollback (§1f, R0-5)

async def test_rollback_aborts_wholesale_when_the_pointer_moved(temp_db):
    """Test 30: separate statements would let a CONCURRENT PUBLICATION be retired by
    accident, so the expected-pointer check aborts the whole transaction."""
    first, second = await _chain_v2(temp_db, 2)
    result = await temp_db.rollback_published_generation_async(
        team_id=TEAM, channel_id=CH, namespace=NS, serializer_version=V2,
        expected_snapshot_id=first)
    assert result == {"ok": False, "restored": None, "reason": "pointer"}
    assert await temp_db.get_snapshot_row_async(first) is not None
    assert (await temp_db.select_snapshot_for_pin_async(
        TEAM, CH, NS, V2, None))["snapshot_id"] == second


async def test_rollback_with_no_prior_generation_returns_the_channel_to_genesis(temp_db):
    """Test 30/60: with no prior generation the pointer is REMOVED and selection returns
    genesis — never raw_rebuild_required, which would immediately restart the very compaction
    the owner rejected."""
    sid = (await _chain_v2(temp_db, 1))[0]
    result = await temp_db.rollback_published_generation_async(
        team_id=TEAM, channel_id=CH, namespace=NS, serializer_version=V2,
        expected_snapshot_id=sid)
    assert result["ok"] and result["restored"] is None

    assert await temp_db.get_snapshot_row_async(sid) is None
    assert await temp_db.snapshot_manifest_async(sid) == []
    assert (await temp_db.select_snapshot_for_pin_async(
        TEAM, CH, NS, V2, None))["result"] == "genesis"
    assert temp_db.conn.execute(
        "SELECT COUNT(*) FROM channel_snapshot_pointer").fetchone()[0] == 0


async def test_rollback_with_a_prior_generation_restores_it(temp_db):
    """Test 60: an earlier VALID v2 generation becomes active again."""
    first, second = await _chain_v2(temp_db, 2)
    result = await temp_db.rollback_published_generation_async(
        team_id=TEAM, channel_id=CH, namespace=NS, serializer_version=V2,
        expected_snapshot_id=second)
    assert result["ok"] and result["restored"] == first

    selected = await temp_db.select_snapshot_for_pin_async(TEAM, CH, NS, V2, None)
    assert (selected["result"], selected["snapshot_id"]) == ("pinned", first)
    assert await temp_db.get_snapshot_row_async(second) is None


async def test_corrupt_lineage_retirement_restores_the_newest_valid_ancestor(temp_db):
    """Test 61: a corrupt INCREMENTAL generation frequently has an earlier valid ancestor
    still physically present and still readable; retiring the lineage and claiming genesis
    would discard a perfectly good summary."""
    first, second, third = await _chain_v2(temp_db, 3)
    result = await temp_db.retire_snapshot_lineage_async(
        team_id=TEAM, channel_id=CH, namespace=NS, serializer_version=V2,
        lineage_ids=[second, third], expected_active_id=third, expected_generation=3)
    assert result["ok"] and result["restored"] == first

    for gone in (second, third):
        assert await temp_db.get_snapshot_row_async(gone) is None
        assert await temp_db.snapshot_manifest_async(gone) == []
        assert await temp_db.snapshot_anchor_provenance_async(gone) == []
    selected = await temp_db.select_snapshot_for_pin_async(TEAM, CH, NS, V2, None)
    assert (selected["result"], selected["snapshot_id"]) == ("pinned", first)


async def test_retiring_the_whole_lineage_returns_genesis(temp_db):
    """Test 61's other branch: when there GENUINELY is no ancestor, the state is genuinely
    genesis — no generation exists, and there is no state below it."""
    ids = await _chain_v2(temp_db, 2)
    result = await temp_db.retire_snapshot_lineage_async(
        team_id=TEAM, channel_id=CH, namespace=NS, serializer_version=V2,
        lineage_ids=ids, expected_active_id=ids[-1], expected_generation=2)
    assert result["ok"] and result["restored"] is None
    assert (await temp_db.select_snapshot_for_pin_async(
        TEAM, CH, NS, V2, None))["result"] == "genesis"


async def test_retirement_aborts_on_a_generation_mismatch(temp_db):
    sid = (await _chain_v2(temp_db, 1))[0]
    result = await temp_db.retire_snapshot_lineage_async(
        team_id=TEAM, channel_id=CH, namespace=NS, serializer_version=V2,
        lineage_ids=[sid], expected_active_id=sid, expected_generation=99)
    assert result == {"ok": False, "restored": None, "reason": "generation"}
    assert await temp_db.get_snapshot_row_async(sid) is not None


# ------------------------------------------------------------------ status machine (§1g)

def test_an_illegal_status_value_is_rejected(temp_db):
    """Test 7: the vocabulary is a CLOSED enum — candidate | published | published_stale |
    invalidated, and nothing else."""
    with pytest.raises(sqlite3.IntegrityError):
        temp_db.conn.execute(
            "INSERT INTO channel_snapshots (snapshot_id, team_id, channel_id, namespace, "
            "serializer_version, boundary_ts, status) "
            "VALUES ('x', 'T', 'C', 'prod', 2, '1.0', 'retired')")


async def test_invalidation_only_moves_a_published_row(temp_db):
    """Test 7: candidate -> invalidated is NOT a legal transition. Losers and discarded
    candidates are DELETED, never statused."""
    candidate = await _candidate_v2(temp_db)
    await temp_db.invalidate_snapshot_async(candidate)
    assert (await temp_db.get_snapshot_row_async(candidate))["status"] == "candidate"

    published = await _candidate_v2(temp_db, boundary_ts="2000.0")
    assert (await _publish_v2(temp_db, published))["won"]
    await temp_db.invalidate_snapshot_async(published)
    assert (await temp_db.get_snapshot_row_async(published))["status"] == "invalidated"


async def test_a_stale_generation_may_also_be_invalidated(temp_db):
    sid = await _candidate_v2(temp_db)
    assert (await _publish_v2(temp_db, sid, None, status="published_stale"))["won"]
    await temp_db.invalidate_snapshot_async(sid)
    assert (await temp_db.get_snapshot_row_async(sid))["status"] == "invalidated"


# ------------------------------------------- the SIX physical-delete sites (§1j / tests 14, 89)

async def _site_cas_loss(db):
    winner = await _candidate_v2(db)
    assert (await _publish_v2(db, winner))["won"]
    doomed = await _candidate_v2(db, payload_text="lost")
    assert (await _publish_v2(db, doomed, None))["reason"] == "cas"
    return doomed


async def _site_frontier_failure(db):
    await _observe(db, "500.0", identity="frontier-site")
    doomed = await _candidate_v2(db)
    assert (await _publish_v2(db, doomed, None, frontier=0))["reason"] == "frontier"
    return doomed


async def _site_anchor_failure(db):
    await _observe(db, ANCHOR_ROOT, identity="anchor-site")
    doomed = await _candidate_v2(db, anchors=_anchors(frontier=0))
    assert (await _publish_v2(db, doomed, None, frontier=0))["reason"] == "anchor"
    return doomed


async def _site_final_profile_failure(db):
    doomed = await _candidate_v2(db)
    assert (await _publish_v2(db, doomed, None,
                              profile=OTHER_PROFILE))["reason"] == "profile_changed"
    return doomed


async def _site_sweep(db):
    ids = await _chain_v2(db, 5)
    doomed = ids[0]
    for sid in ids:
        _backdate(db, sid, 30)
    assert await db.sweep_snapshots_async([], retain_generations=3, retain_days=7) >= 1
    return doomed


async def _site_corrupt_retirement(db):
    first, second = await _chain_v2(db, 2)
    result = await db.retire_snapshot_lineage_async(
        team_id=TEAM, channel_id=CH, namespace=NS, serializer_version=V2,
        lineage_ids=[second], expected_active_id=second, expected_generation=2)
    assert result["ok"]
    return second


async def _site_c05_rollback(db):
    sid = (await _chain_v2(db, 1))[0]
    assert (await db.rollback_published_generation_async(
        team_id=TEAM, channel_id=CH, namespace=NS, serializer_version=V2,
        expected_snapshot_id=sid))["ok"]
    return sid


# §1j names SIX sites; `sweep` and `corrupt-lineage retirement` are two DISTINCT code paths
# listed under one number, so they are exercised SEPARATELY here. A test covering only CAS
# loss and sweep leaves the newer paths unguarded, which is how orphan rows get missed.
PHYSICAL_DELETE_SITES = [
    ("i_publication_cas_loss", _site_cas_loss),
    ("ii_failed_frontier_verification", _site_frontier_failure),
    ("iii_failed_anchor_verification", _site_anchor_failure),
    ("iv_failed_final_profile_predicate", _site_final_profile_failure),
    ("v_sweep", _site_sweep),
    ("v_corrupt_lineage_retirement", _site_corrupt_retirement),
    ("vi_c05_rollback", _site_c05_rollback),
]


def test_the_delete_site_list_is_the_whole_contract():
    """Tests 14 and 89 are PARAMETERIZED OVER ONE LIST so the count and the list cannot
    drift. Six §1j sites, with (v) covering two distinct code paths."""
    assert len(PHYSICAL_DELETE_SITES) == 7
    assert len({name.split("_")[0] for name, _ in PHYSICAL_DELETE_SITES}) == 6


@pytest.mark.parametrize("name,site", PHYSICAL_DELETE_SITES,
                         ids=[n for n, _ in PHYSICAL_DELETE_SITES])
async def test_manifest_and_anchor_rows_die_with_their_snapshot(temp_db, name, site):
    """Tests 14 + 89: SQLite foreign keys are OFF in this database, so nothing cascades and
    every site must delete explicitly. An orphan row surviving its snapshot can fail a LATER
    publication's anchor check for a generation that no longer exists."""
    doomed = await site(temp_db)

    assert await temp_db.get_snapshot_row_async(doomed) is None, name
    assert await temp_db.snapshot_manifest_async(doomed) == [], name
    assert await temp_db.snapshot_anchor_provenance_async(doomed) == [], name


@pytest.mark.parametrize("name,site", PHYSICAL_DELETE_SITES,
                         ids=[n for n, _ in PHYSICAL_DELETE_SITES])
async def test_no_orphan_manifest_or_anchor_row_survives_anywhere(temp_db, name, site):
    """The stronger form: no manifest or anchor row anywhere names a snapshot that is gone."""
    await site(temp_db)
    orphan_manifest = temp_db.conn.execute(
        "SELECT COUNT(*) FROM snapshot_capture_manifest WHERE snapshot_id NOT IN "
        "(SELECT snapshot_id FROM channel_snapshots)").fetchone()[0]
    orphan_anchors = temp_db.conn.execute(
        "SELECT COUNT(*) FROM snapshot_anchor_provenance WHERE snapshot_id NOT IN "
        "(SELECT snapshot_id FROM channel_snapshots)").fetchone()[0]
    assert (orphan_manifest, orphan_anchors) == (0, 0), name


async def test_an_ordinary_discarded_candidate_is_not_on_the_list(temp_db):
    """§1j: validation PRECEDES insertion, so a discarded candidate has no rows at all — which
    is why it is deliberately absent from the physical-delete list."""
    with pytest.raises(ValueError):
        await temp_db.insert_compaction_candidate_async(
            snapshot=_v2(sizing_profile=None), manifest_rows=_manifest(),
            anchor_rows=_anchors())
    assert temp_db.conn.execute(
        "SELECT COUNT(*) FROM channel_snapshots").fetchone()[0] == 0


# ----------------------------------------------------------------------- sweep (§1n, §3.9)

async def test_the_sweep_protects_a_live_checkpoints_parent(temp_db):
    """Test 71: retiring the one lineage an in-progress incremental crawl needs would force it
    back to raw — which, on a channel whose Slack retention is shallow, destroys the only
    usable lineage for no reason."""
    ids = await _chain_v2(temp_db, 5)
    for sid in ids:
        _backdate(temp_db, sid, 30)

    await temp_db.upsert_crawl_checkpoint_async({
        "team_id": TEAM, "channel_id": CH, "namespace": NS, "crawl_id": "live-crawl",
        "crawl_mode": "incremental", "phase": 2, "pinned_H": "9000.0",
        "mutation_frontier": 0, "source_floor_ts": "1.0", "input_floor_ts": "1000.0",
        "input_floor_inclusive": 0, "parent_snapshot_id": ids[0], "serializer_version": V2,
        "serializer_config_hash": "sc", "prompt_version": "v1", "sizing_profile": PROFILE,
        "headroom_source": "measured", "headroom_tokens": 1, "profile_version": "measured:1",
        "actor_snapshot_hash": "ah", "chunk_index": 0, "attempt_seq": 0,
        "attempt_tokens_in": 0, "attempt_tokens_out": 0, "attempt_cached_input_tokens": 0,
        "attempt_call_count": 0, "event_count": 0, "consecutive_discards": 0})

    await temp_db.sweep_snapshots_async([], retain_generations=3, retain_days=7)
    assert await temp_db.get_snapshot_row_async(ids[0]) is not None

    await temp_db.delete_crawl_state_async(TEAM, CH, NS, "live-crawl")
    await temp_db.sweep_snapshots_async([], retain_generations=3, retain_days=7)
    assert await temp_db.get_snapshot_row_async(ids[0]) is None


async def test_the_sweep_honours_explicit_protected_ids(temp_db):
    ids = await _chain_v2(temp_db, 5)
    for sid in ids:
        _backdate(temp_db, sid, 30)
    await temp_db.sweep_snapshots_async([], retain_generations=3, retain_days=7,
                                        protected_ids=[ids[0]])
    assert await temp_db.get_snapshot_row_async(ids[0]) is not None


async def test_the_sweep_counts_generations_per_namespace(temp_db):
    """A fenced test epoch's generation numbers must not age production's out."""
    mine = (await _chain_v2(temp_db, 1))[0]
    _backdate(temp_db, mine, 30)
    previous = None
    for i in range(5):
        sid = await temp_db.insert_compaction_candidate_async(
            snapshot=_v2(namespace="epoch-1", boundary_ts=f"{2000 + i}.0"))
        assert (await temp_db.publish_compaction_candidate_async(
            team_id=TEAM, channel_id=CH, namespace="epoch-1", serializer_version=V2,
            snapshot_id=sid, expected_previous_id=previous, source_floor_ts="10.0",
            boundary_ts=f"{2000 + i}.0", mutation_frontier=0,
            current_profile=PROFILE))["won"]
        previous = sid

    await temp_db.sweep_snapshots_async([], retain_generations=3, retain_days=7)
    assert await temp_db.get_snapshot_row_async(mine) is not None


# --------------------------------------------------------------- late artifact evidence

async def test_late_evidence_is_computed_against_the_pinned_manifest(temp_db):
    """§1i: a turn's late evidence is ITS PINNED snapshot's business — an overlapping turn may
    pin S1 after S2 became active."""
    temp_db.conn.execute(
        "INSERT INTO images (thread_id, message_ts, url, image_type, analysis) "
        "VALUES (?, ?, ?, ?, ?)", (f"{CH}:100.0", "100.0", "u1", "analysis", "a chart"))
    row_id = str(temp_db.conn.execute("SELECT id FROM images").fetchone()[0])

    captured = await _candidate_v2(temp_db, manifest=[
        {"artifact_namespace": "image_analysis", "row_id": row_id, "source_ts": "100.0",
         "captured_render_version": "v1", "content_hash": "h", "status_at_capture":
             "analysis"}])
    uncaptured = await _candidate_v2(temp_db, boundary_ts="2000.0", manifest=[])

    assert await temp_db.late_artifact_evidence_async(
        TEAM, CH, captured, boundary_ts="1000.000100", high_ts="9000.0") == []

    late = await temp_db.late_artifact_evidence_async(
        TEAM, CH, uncaptured, boundary_ts="1000.000100", high_ts="9000.0")
    assert [(e["artifact_namespace"], e["row_id"]) for e in late] == [
        ("image_analysis", row_id)]


async def test_late_evidence_is_keyed_by_the_full_tuple(temp_db):
    """§1i: (source_ts, snapshot_id) alone COLLIDES whenever one message carries several
    artifacts, so the key is the full four-part tuple and the order is
    (source_ts, artifact_namespace, row_id)."""
    for i in range(2):
        temp_db.conn.execute(
            "INSERT INTO images (thread_id, message_ts, url, image_type, analysis) "
            "VALUES (?, ?, ?, 'analysis', ?)",
            (f"{CH}:100.0", "100.0", f"u{i}", f"chart {i}"))
    temp_db.conn.execute(
        "INSERT INTO documents (thread_id, message_ts, filename, mime_type, summary) "
        "VALUES (?, '100.0', 'a.pdf', 'application/pdf', 'sum')", (f"{CH}:100.0",))

    sid = await _candidate_v2(temp_db, manifest=[])
    late = await temp_db.late_artifact_evidence_async(
        TEAM, CH, sid, boundary_ts="1000.000100", high_ts="9000.0")
    assert len(late) == 3
    assert [e["artifact_namespace"] for e in late] == [
        "document_extraction", "image_analysis", "image_analysis"]
    assert len({(e["source_ts"], e["snapshot_id"], e["artifact_namespace"], e["row_id"])
                for e in late}) == 3


# ------------------------------------------------------------------ pre-boundary sidecars

async def test_preboundary_receipts_are_pinned_only_when_asked_for(temp_db):
    """Interfaces §3.9: the landed sidecar read retrieves receipts only inside (boundary, H],
    which §1k rehydration needs more than."""
    await temp_db.seed_channel_coverage_async(TEAM, CH, "1.0")
    await temp_db.register_receipt_async(TEAM, CH, "50.0", "turn-old", "finalized")
    await temp_db.register_receipt_async(TEAM, CH, "500.0", "turn-new", "finalized")

    plain = await temp_db.read_channel_sidecars_async(
        TEAM, CH, "9000.0", window=("100.0", False))
    assert [r["message_ts"] for r in plain["receipts"]] == ["500.0"]
    assert plain["preboundary_receipts"] == []

    pinned_read = await temp_db.read_channel_sidecars_async(
        TEAM, CH, "9000.0", window=("100.0", False), preboundary_receipts=True)
    assert [r["message_ts"] for r in pinned_read["receipts"]] == ["500.0"]
    assert [r["message_ts"] for r in pinned_read["preboundary_receipts"]] == ["50.0"]
    assert pinned_read["versions_hash"] != plain["versions_hash"]


async def test_a_receipt_is_in_exactly_one_of_the_two_lists(temp_db):
    """The pre-boundary read is the STRICT COMPLEMENT of the window below the floor, so the
    boundary message's own receipt is never counted twice."""
    await temp_db.seed_channel_coverage_async(TEAM, CH, "1.0")
    await temp_db.register_receipt_async(TEAM, CH, "100.0", "turn-boundary", "finalized")

    exclusive = await temp_db.read_channel_sidecars_async(
        TEAM, CH, "9000.0", window=("100.0", False), preboundary_receipts=True)
    assert exclusive["receipts"] == []
    assert [r["message_ts"] for r in exclusive["preboundary_receipts"]] == ["100.0"]

    inclusive = await temp_db.read_channel_sidecars_async(
        TEAM, CH, "9000.0", window=("100.0", True), preboundary_receipts=True)
    assert [r["message_ts"] for r in inclusive["receipts"]] == ["100.0"]
    assert inclusive["preboundary_receipts"] == []


# =====================================================================================
# P4a-C1 — the COORDINATOR half. The DB half above is A1's; everything below drives the
# coordinator's own surface (plan §1m/§1f). The broader coordinator behaviour lives in
# tests/unit/test_compaction_coordinator.py and tests/unit/test_compaction_dormancy.py.
# =====================================================================================

async def test_publish_does_not_deadlock_against_a_caller_holding_the_channel_lock(temp_db):
    """§1m's DEADLOCK FIX, asserted directly.

    `channel_lock()` is NOT reentrant. The landed shape had `publish()` take it AND a compaction
    path hold it across the call — which wedges the moment both are true. The fix scopes the lock
    strictly inside `publish()`, and `lock_held=True` is how the ONE caller that already holds it
    says so.
    """
    coord = _coordinator(temp_db)
    sid = await _coord_candidate(coord, temp_db)
    async with coord.channel_lock(TEAM, CH):
        result = await asyncio.wait_for(
            coord.publish(team_id=TEAM, channel_id=CH, serializer_version=SV, snapshot_id=sid,
                          expected_previous_id=None, source_floor_ts="1.0",
                          boundary_ts="1000.0", mutation_frontier=0, lock_held=True),
            timeout=2.0)
    assert result["won"]


async def test_publish_takes_the_lock_itself_when_the_caller_does_not_hold_it(temp_db):
    """The other half: NOBODY holds the channel lock across a call into `publish()`, so the
    default path must serialize on its own."""
    coord = _coordinator(temp_db)
    first = await _coord_candidate(coord, temp_db, "one")
    second = await _coord_candidate(coord, temp_db, "two")
    results = await asyncio.gather(_coord_publish(coord, first, None),
                                   _coord_publish(coord, second, None))
    assert sum(1 for r in results if r["won"]) == 1


async def test_a_winning_publication_clears_a_compaction_stall(temp_db):
    """A stalled channel resumes on evidence, not on hope: the mark is what stops the coordinator
    attempting anything further, so a successful publication has to lift it."""
    coord = _coordinator(temp_db)
    coord._stalled.add((TEAM, CH, PROD_NAMESPACE))
    sid = await _coord_candidate(coord, temp_db)
    assert (await _coord_publish(coord, sid, None))["won"]
    assert (TEAM, CH, PROD_NAMESPACE) not in coord._stalled


async def test_select_and_pin_pins_only_what_it_renders(temp_db):
    """The two refused results retain identity for telemetry and the CAS but are never read, so
    pinning them would hold rows against the sweep that nothing will ever render."""
    coord = _coordinator(temp_db)
    sid = (await _publish_chain(temp_db, 1))[0]
    await coord.invalidate(sid)
    result = await coord.select_and_pin(TEAM, CH, SV)
    assert result["result"] == "raw_rebuild_required" and result["snapshot_id"] == sid
    assert coord.pinned_ids() == []


async def test_pending_invalidation_is_resolved_before_the_pointer_is_read(temp_db):
    """§1a step 3. An observation that landed while the drain was waiting must reach the pointer
    BEFORE the turn selects, or the turn renders from a generation a durable decision has already
    condemned."""
    coord = _coordinator(temp_db)
    sid = (await _chain_v2(temp_db, 1))[0]
    await _observe(temp_db, "500.000000")            # inside [source_floor_ts, boundary_ts]
    assert await coord.resolve_pending_invalidation(TEAM, CH) == [sid]
    assert (await coord.select_and_pin(TEAM, CH, V2,
                                       namespace=NS))["result"] == "raw_rebuild_required"


async def test_resolving_twice_does_not_re_invalidate(temp_db):
    """The resolution point advances, so a busy channel does not re-walk its whole journal on
    every turn."""
    coord = _coordinator(temp_db)
    sid = (await _chain_v2(temp_db, 1))[0]
    await _observe(temp_db, "500.000000")
    assert await coord.resolve_pending_invalidation(TEAM, CH) == [sid]
    assert await coord.resolve_pending_invalidation(TEAM, CH) == []
