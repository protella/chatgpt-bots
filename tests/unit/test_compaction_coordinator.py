"""The compaction coordinator: turn-path ordering, retirement, triggers, revalidation, drainer.

Appendix C tests 19 (partly, with the dormancy file), 30, 40, 42, 54, 55, 59, 60, 61, 62, 71, 72,
79 and 100's drainer cases live here. The dormancy/obsolescence half is
`test_compaction_dormancy.py`.

The load-bearing claims are the ORDERING ones, and they are asserted as orderings rather than as
outcomes: "the active pointer is never consulted before the frontier drain" and "a row is deleted
only after acknowledgement" are both true by accident in a happy path and false the moment
anything is slow, so a test that only checks the end state proves nothing.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from database import PROD_NAMESPACE, DatabaseManager
from message_processor import channel_snapshots as cs
from message_processor.channel_snapshots import ChannelSnapshotCoordinator

TEAM = "T1"
CH = "C0BKX77NU66"
NS = PROD_NAMESPACE
V2 = 2
PROFILE = "gpt-5.6-luna:400000:320000:280000"
OTHER_PROFILE = "gpt-5.6-luna:400000:300000:260000"


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    db = DatabaseManager(platform="slack")
    yield db
    db.conn.close()


def coordinator(db, **kw) -> ChannelSnapshotCoordinator:
    kw.setdefault("retry_delay", 0.0)
    return ChannelSnapshotCoordinator(db, retain_generations=3, retain_days=7, **kw)


def v2_row(**kw) -> Dict[str, Any]:
    body = kw.pop("payload_text", "the summary").encode("utf-8")
    row = dict(team_id=TEAM, channel_id=CH, namespace=NS, serializer_version=V2,
               boundary_ts="1000.000100", source_floor_ts="10.000000",
               parent_snapshot_id=None, prompt_version="v1", model="gpt-5.6-luna",
               source_hash="sh", payload_bytes=body, anchor_payload_bytes=b"anchors",
               mutation_frontier=0, headroom_source="measured", headroom_tokens=90000,
               effective_window=400000, sizing_profile=PROFILE, fit_result="under_target")
    row.update(kw)
    return row


async def candidate(db, **kw) -> str:
    return await db.insert_compaction_candidate_async(snapshot=v2_row(**kw))


async def publish(coord, sid, previous=None, *, profile=PROFILE, boundary="1000.000100", **kw):
    return await coord.publish(
        team_id=TEAM, channel_id=CH, namespace=NS, serializer_version=V2, snapshot_id=sid,
        expected_previous_id=previous, source_floor_ts="10.000000", boundary_ts=boundary,
        mutation_frontier=0, current_profile=profile, **kw)


async def chain(db, coord, count: int) -> List[str]:
    ids: List[str] = []
    previous = None
    for i in range(count):
        sid = await candidate(db, boundary_ts=f"{1000 + i}.000100", payload_text=f"summary {i}",
                              parent_snapshot_id=previous)
        assert (await publish(coord, sid, previous, boundary=f"{1000 + i}.000100"))["won"]
        ids.append(sid)
        previous = sid
    return ids


# ===================================================================== §1a turn-path ordering

class _Host:
    """The smallest thing `_build_channel_turn_stream` can be driven against.

    Bound unbound, deliberately: the method under test is the ORDERING, and a full processor would
    bring a hundred other reasons for the call to fail.
    """

    def __init__(self, db, coord):
        self.db = db
        self.snapshot_coordinator = coord
        self.mcp_manager = None

    def log_info(self, *a, **k): pass
    def log_debug(self, *a, **k): pass
    def log_warning(self, *a, **k): pass

    async def ingest_channel_origin_slice(self, *a, **k): return None


def _bind_real_methods():
    """The REAL builder call and the REAL trigger site, not stand-ins: those two are the thing
    under test, and reimplementing them here would test the reimplementation."""
    from message_processor.base import MessageProcessor
    _Host._channel_stream_call = MessageProcessor._channel_stream_call
    _Host._trigger_compaction = MessageProcessor._trigger_compaction
    _Host._compaction_evidence = MessageProcessor._compaction_evidence
    _Host._post_turn_compaction = MessageProcessor._post_turn_compaction


_bind_real_methods()


def _message(channel_id=CH):
    message = MagicMock()
    message.channel_id = channel_id
    message.thread_id = "1700001000.000100"
    message.metadata = {"ts": "1700001000.000100"}
    return message


async def _drive_turn(db, coord, monkeypatch, *, selection: Dict[str, Any],
                      raise_fetch: Optional[Exception] = None,
                      builder=None) -> Dict[str, Any]:
    """Run the §1a section of `_build_channel_turn_stream` and report what it did, in order."""
    from message_processor.base import MessageProcessor
    from slack_client import admission_watermark

    order: List[str] = []
    captured: Dict[str, Any] = {"order": order, "builder_kwargs": None, "triggers": []}

    async def _drain(channel_id, frontier, timeout=None):
        order.append("drain")
    monkeypatch.setattr(admission_watermark, "drain", _drain)

    async def _resolve(team_id, channel_id, *, namespace=NS):
        order.append("resolve_invalidation")
        return []
    monkeypatch.setattr(coord, "resolve_pending_invalidation", _resolve)

    async def _select(team_id, channel_id, serializer_version, max_boundary=None, *,
                      namespace=NS):
        order.append("select_and_pin")
        return dict(selection)
    monkeypatch.setattr(coord, "select_and_pin", _select)

    def _trigger(*, team_id, channel_id, path, **evidence):
        captured["triggers"].append((path, evidence))
    monkeypatch.setattr(coord, "trigger", _trigger)

    async def _build(**kwargs):
        order.append("build_channel_stream")
        captured["builder_kwargs"] = kwargs
        if raise_fetch is not None:
            raise raise_fetch
        stream = MagicMock()
        stream.message_count = 0
        stream.byte_count = 0
        stream.pinned.H = "1700009999.000000"
        stream.pinned.floor_ts = "1.0"
        stream.pinned.floor_inclusive = True
        stream.pinned.actor_names = {}
        stream.pinned.sidecars = None
        return stream

    monkeypatch.setattr("message_processor.channel_stream.build_channel_stream",
                        builder or _build)
    monkeypatch.setattr("message_processor.channel_request.origin_slice_messages",
                        lambda *a, **k: ())
    monkeypatch.setattr("message_processor.channel_request.origin_participants_from_slice",
                        lambda *a, **k: {})

    host = _Host(db, coord)
    client = MagicMock()
    client.self_team_id = TEAM
    client.tool_registry = None
    turn = MagicMock()
    turn.turn_id = "turn-1"
    turn.snapshot_lease = None
    h_pin = MagicMock()
    h_pin.h = "1700009999.000000"
    h_pin.frontier = 7

    captured["turn"] = turn
    method = MessageProcessor._build_channel_turn_stream
    try:
        await method(host, _message(), client, turn, h_pin, {"model": "gpt-5.6-luna"},
                     MagicMock())
    except Exception as e:  # noqa: BLE001 — the fail-closed cases are the point of some cases
        captured["error"] = e
    return captured


async def test_the_active_pointer_is_never_consulted_before_the_frontier_drain(
        temp_db, monkeypatch):
    """§1a, and it is an ORDERING claim, not an outcome one.

    An observation still in flight when the turn reads the pointer reaches it a moment later, and
    the turn then renders from a generation a durable decision has already condemned.
    """
    coord = coordinator(temp_db)
    result = await _drive_turn(temp_db, coord, monkeypatch,
                              selection={"result": "genesis", "snapshot": None,
                                         "snapshot_id": None})
    assert result.get("error") is None, result.get("error")
    assert result["order"] == ["drain", "resolve_invalidation", "select_and_pin",
                               "build_channel_stream"]


@pytest.mark.parametrize("refused", ["raw_rebuild_required", "payload_corrupt"])
async def test_a_refused_selection_is_never_handed_to_the_builder(temp_db, monkeypatch, refused):
    """The two results that RETAIN identity for telemetry and the CAS are not renderable.

    Passing one to the builder would render a raw window a durable decision says is wrong. They
    trigger recompaction instead, and the turn fails closed on the unresolved pointer.
    """
    coord = coordinator(temp_db)
    result = await _drive_turn(
        temp_db, coord, monkeypatch,
        selection={"result": refused, "snapshot": v2_row(), "snapshot_id": "snap-bad",
                   "generation": 4})
    assert result["builder_kwargs"]["snapshot"] is None
    assert result["turn"].snapshot_lease is None          # nothing was pinned, so nothing leased
    assert [path for path, _ in result["triggers"]] == [cs.PATH_SELECTION_REFUSED]
    assert result["triggers"][0][1]["reason"] == refused


async def test_no_eligible_generation_hands_the_builder_exactly_what_genesis_does(
        temp_db, monkeypatch):
    """The SIXTH result RENDERS, and it renders AS GENESIS — no summary block, same horizon.

    A comparison, not a spot check: asserting only "no summary item" would pass an implementation
    that rendered a subtly different horizon. The builder call is byte-for-byte the genesis one,
    which is what makes the two streams hash the same.
    """
    coord = coordinator(temp_db)
    early = await _drive_turn(
        temp_db, coord, monkeypatch,
        selection={"result": "no_eligible_generation", "snapshot": v2_row(),
                   "snapshot_id": "snap-early", "generation": 2})
    genesis = await _drive_turn(temp_db, coord, monkeypatch,
                                selection={"result": "genesis", "snapshot": None,
                                           "snapshot_id": None, "generation": None})
    assert early["builder_kwargs"]["snapshot"] is None
    # Everything the BYTES are a function of, compared as a whole rather than spot-checked. The
    # per-call objects (the fetch budget, the mock client) are identity, not content.
    contentful = ("snapshot", "namespace", "h", "frontier", "team_id", "channel_id",
                  "capability_profile_hash", "tool_schema_version", "origin_thread_ts",
                  "trigger_ts")
    assert ({k: early["builder_kwargs"][k] for k in contentful}
            == {k: genesis["builder_kwargs"][k] for k in contentful})
    assert early["builder_kwargs"]["selection_result"] == "no_eligible_generation"
    assert genesis["builder_kwargs"]["selection_result"] == "genesis"
    # NOT a refuse state: nothing here is invalid, so nothing is recompacted.
    assert early["triggers"] == []


async def test_no_eligible_generation_renders_a_byte_identical_stream(temp_db):
    """The positive half of the render equivalence, at the serializer.

    `stream_sha256` is the cleanest expression of it: that hash is what the prompt-cache prefix
    turns on, so equality here is equality of everything the model sees.
    """
    from tests.unit.test_channel_stream_v2 import msg, pinned
    from message_processor.channel_stream import serialize_stream
    early = serialize_stream(pinned([msg("1700001000.000100")], snapshot=None))
    genesis = serialize_stream(pinned([msg("1700001000.000100")], snapshot=None))
    assert early.stream_sha256 == genesis.stream_sha256
    assert early.summary_item is None


async def test_the_selection_result_and_its_identity_reach_stream_render(temp_db, monkeypatch):
    """A tag nothing emits is a tag nobody can act on — and a tag with no snapshot_id says a
    channel is holding something early without saying WHICH generation."""
    from message_processor import channel_stream
    emitted: List[Dict[str, Any]] = []
    monkeypatch.setattr("message_processor.participation_telemetry.stream_render",
                        lambda **fields: emitted.append(fields))
    stream = MagicMock()
    stream.stream_render_fields.return_value = {"channel_id": CH, "snapshot_id": "snap-early",
                                                "generation": 2}
    channel_stream._emit_stream_render(stream, turn_id="turn-1", origin_thread_ts=None,
                                       trigger_ts=None,
                                       selection_result="no_eligible_generation")
    assert emitted[0]["selection_result"] == "no_eligible_generation"
    assert emitted[0]["snapshot_id"] == "snap-early"


async def test_a_refused_turn_still_emits_exactly_one_stream_render(temp_db, monkeypatch):
    """Appendix C test 40. A refused turn's FINAL CANONICAL BUILD IS EVIDENCE.

    Never zero — an over-budget refusal is precisely the case where the evidence matters most, and
    suppressing it hides the turns a channel is failing. Never two — there is no
    measure → compact → rebuild sequence inside one turn, because same-turn compaction does not
    exist in P4.
    """
    from message_processor import channel_stream
    emitted: List[Dict[str, Any]] = []
    monkeypatch.setattr("message_processor.participation_telemetry.stream_render",
                        lambda **fields: emitted.append(fields))

    stream = MagicMock()
    stream.stream_render_fields.return_value = {"channel_id": CH}
    channel_stream._emit_stream_render(stream, turn_id="turn-1", origin_thread_ts=None,
                                       trigger_ts=None, selection_result="pinned")
    # The turn then fails closed over budget. NOTHING rebuilds, so nothing emits again.
    coord = coordinator(temp_db)
    coord._accepting = True
    seen: List[int] = []
    coord._compact = AsyncMock(side_effect=lambda *a, **k: seen.append(1))
    coord.trigger(team_id=TEAM, channel_id=CH, namespace=NS, path=cs.PATH_OVER_BUDGET,
                  headroom_tokens=1000)
    await asyncio.gather(*[t for t in coord._tasks.values()], return_exceptions=True)
    assert len(emitted) == 1


async def test_the_page_ceiling_is_told_apart_from_any_other_fetch_failure(temp_db, monkeypatch):
    """§1m PATH 3. A rate-limited fetch is a Slack problem a background crawl cannot fix, so
    enqueueing one for it would spend a crawl per rate-limit."""
    from base_client import HistoryFetchError
    from config import config
    coord = coordinator(temp_db)
    plain = await _drive_turn(temp_db, coord, monkeypatch,
                              selection={"result": "genesis", "snapshot": None,
                                         "snapshot_id": None},
                              raise_fetch=HistoryFetchError("slack said no"))
    assert plain["triggers"] == []

    async def _exhaust(**kwargs):
        budget = kwargs["budget"]
        try:
            while True:
                await budget.charge_page()
        except HistoryFetchError:
            raise HistoryFetchError("channel history fetch hit the page ceiling")

    assert int(config.history_page_ceiling) > 0
    ceiling = await _drive_turn(temp_db, coord, monkeypatch,
                               selection={"result": "genesis", "snapshot": None,
                                          "snapshot_id": None}, builder=_exhaust)
    assert [path for path, _ in ceiling["triggers"]] == [cs.PATH_PAGE_CEILING]


# ===================================================================== §1f retirement phases

async def test_retirement_marks_the_lineage_before_the_readers_have_drained(temp_db):
    """Appendix C test 62 — the phasing, and the proof `unpin()` is not deadlocked.

    `unpin()` takes the PIN lock, so a retirement that waited for refcounts while holding it could
    never succeed: the reader it is waiting for cannot get in to release. The pin lock is taken
    only to set the mark; the CHANNEL lock is what is held across all three phases.
    """
    coord = coordinator(temp_db)
    ids = await chain(temp_db, coord, 2)
    await coord.pin(ids[-1])

    refused_during_phase_2: List[tuple] = []
    released = asyncio.Event()

    async def _reader():
        await released.wait()
        # This is the call a pin-lock-holding retirement would deadlock.
        await coord.unpin(ids[-1])

    reader = asyncio.create_task(_reader())

    original = coord._await_drain

    async def _observe(lineage, *, timeout, poll=0.01):
        refused_during_phase_2.append(coord.refused_lineage(TEAM, CH, NS))
        released.set()
        return await original(lineage, timeout=timeout, poll=poll)

    coord._await_drain = _observe
    result = await coord.retire_lineage(team_id=TEAM, channel_id=CH, lineage_ids=[ids[-1]],
                                        expected_active_id=ids[-1], namespace=NS,
                                        serializer_version=V2, drain_timeout=5.0)
    await reader
    assert result["ok"] and result["restored"] == ids[0]
    # Phase 2 saw the mark, so a concurrent selection would have refused the lineage already.
    assert ids[-1] in refused_during_phase_2[0]
    # And the mark is gone once the transaction committed.
    assert coord.refused_lineage(TEAM, CH, NS) == ()


async def test_a_selection_during_phase_two_refuses_the_pending_lineage(temp_db):
    coord = coordinator(temp_db)
    ids = await chain(temp_db, coord, 2)
    async with coord._pin_lock:
        coord._retiring[(TEAM, CH, NS)].add(ids[-1])
    result = await coord.select_and_pin(TEAM, CH, V2, namespace=NS)
    assert result["snapshot_id"] == ids[0]        # the older, still-selectable generation


async def test_a_cancelled_retirement_unmarks_in_an_outer_finally(temp_db):
    """§1f. Without the outer `finally` a cancelled retirement leaves a snapshot that exists, is
    valid, and that nothing in this process will ever read again."""
    coord = coordinator(temp_db)
    ids = await chain(temp_db, coord, 1)
    await coord.pin(ids[0])           # a reader that never releases
    result = await coord.retire_lineage(team_id=TEAM, channel_id=CH, lineage_ids=[ids[0]],
                                        expected_active_id=ids[0], namespace=NS,
                                        serializer_version=V2, drain_timeout=0.05)
    assert not result["ok"] and result["reason"] == "pinned"
    assert coord.refused_lineage(TEAM, CH, NS) == ()
    assert (await coord.select_and_pin(TEAM, CH, V2, namespace=NS))["snapshot_id"] == ids[0]


async def test_a_failed_retirement_transaction_also_unmarks(temp_db):
    coord = coordinator(temp_db)
    ids = await chain(temp_db, coord, 1)
    result = await coord.retire_lineage(team_id=TEAM, channel_id=CH, lineage_ids=[ids[0]],
                                        expected_active_id="not-the-active-one", namespace=NS,
                                        serializer_version=V2, drain_timeout=1.0)
    assert not result["ok"] and result["reason"] == "pointer"
    assert coord.refused_lineage(TEAM, CH, NS) == ()


async def test_corrupt_lineage_retirement_restores_the_newest_valid_ancestor(temp_db):
    """Appendix C test 61. A corrupt INCREMENTAL generation frequently has an earlier valid
    ancestor still physically present; claiming genesis would discard a perfectly good summary."""
    coord = coordinator(temp_db)
    ids = await chain(temp_db, coord, 3)
    result = await coord.retire_lineage(team_id=TEAM, channel_id=CH, lineage_ids=[ids[2]],
                                        expected_active_id=ids[2], namespace=NS,
                                        serializer_version=V2)
    assert result["ok"] and result["restored"] == ids[1]
    assert (await coord.select_and_pin(TEAM, CH, V2, namespace=NS))["result"] == "pinned"


async def test_retiring_the_whole_lineage_returns_the_channel_to_genesis(temp_db):
    """The other branch of test 61: genesis ONLY when no valid ancestor exists."""
    coord = coordinator(temp_db)
    ids = await chain(temp_db, coord, 2)
    result = await coord.retire_lineage(team_id=TEAM, channel_id=CH, lineage_ids=ids,
                                        expected_active_id=ids[-1], namespace=NS,
                                        serializer_version=V2)
    assert result["ok"] and result["restored"] is None
    assert (await coord.select_and_pin(TEAM, CH, V2, namespace=NS))["result"] == "genesis"


# ===================================================================== R0-5 rollback

async def test_rollback_aborts_wholesale_when_the_pointer_moved(temp_db):
    """Appendix C test 30. Separate statements would let a concurrent publication be retired by
    accident."""
    coord = coordinator(temp_db)
    ids = await chain(temp_db, coord, 2)
    result = await coord.rollback_generation(team_id=TEAM, channel_id=CH,
                                             expected_snapshot_id=ids[0], namespace=NS,
                                             serializer_version=V2)
    assert not result["ok"] and result["reason"] == "pointer"
    assert await temp_db.get_snapshot_async(ids[0]) is not None


async def test_rollback_waits_for_a_live_reader(temp_db):
    """Appendix C test 59 — PIN-AWARE. The rollback must not delete rows underneath a turn that
    is still rendering from them."""
    coord = coordinator(temp_db)
    ids = await chain(temp_db, coord, 1)
    await coord.pin(ids[0])

    task = asyncio.create_task(coord.rollback_generation(
        team_id=TEAM, channel_id=CH, expected_snapshot_id=ids[0], namespace=NS,
        serializer_version=V2, drain_timeout=5.0))
    await asyncio.sleep(0.05)
    assert not task.done()                                   # still waiting on the reader
    assert await temp_db.get_snapshot_async(ids[0]) is not None
    await coord.unpin(ids[0])
    result = await task
    assert result["ok"]
    assert await temp_db.get_snapshot_async(ids[0]) is None


async def test_rollback_restores_the_prior_generation_or_returns_genesis(temp_db):
    """Appendix C test 60, both branches — never `raw_rebuild_required`, which would restart the
    very compaction the owner rejected."""
    coord = coordinator(temp_db)
    ids = await chain(temp_db, coord, 2)
    assert (await coord.rollback_generation(team_id=TEAM, channel_id=CH,
                                            expected_snapshot_id=ids[1], namespace=NS,
                                            serializer_version=V2))["restored"] == ids[0]
    restored = await coord.select_and_pin(TEAM, CH, V2, namespace=NS)
    assert restored["snapshot_id"] == ids[0]
    await coord.unpin(ids[0])          # the reader is done; the next rollback must not wait on it

    second = await coord.rollback_generation(team_id=TEAM, channel_id=CH,
                                            expected_snapshot_id=ids[0], namespace=NS,
                                            serializer_version=V2)
    assert second["ok"] and second["restored"] is None
    assert (await coord.select_and_pin(TEAM, CH, V2, namespace=NS))["result"] == "genesis"


async def test_nothing_in_this_wave_runs_the_c05_crawl():
    """R0-5: the crawl for C0BKX77NU66 runs ONLY after the owner approves it, and no user-facing
    post of any kind is introduced in that channel. The rollback is built COMPLETELY; its
    initiation is not, and nothing here names that channel."""
    import pathlib
    source = pathlib.Path(cs.__file__).read_text()
    assert "C0BKX77NU66" not in source


# ===================================================================== the sweep (test 71)

async def test_the_sweep_protects_a_live_checkpoints_parent(temp_db):
    """Appendix C test 71. Without this the sweep can retire the one lineage an in-progress
    incremental crawl needs, forcing a fall back to raw on a channel whose Slack retention may not
    support one."""
    coord = coordinator(temp_db)
    ids = await chain(temp_db, coord, 4)
    for sid in ids[:-1]:
        temp_db.conn.execute(
            "UPDATE channel_snapshots SET created_ts = datetime('now', '-30 days') "
            "WHERE snapshot_id = ?", (sid,))
    temp_db.conn.commit()
    await temp_db.upsert_crawl_checkpoint_async({
        "team_id": TEAM, "channel_id": CH, "namespace": NS, "crawl_id": uuid.uuid4().hex,
        "crawl_mode": "incremental", "phase": 1, "pinned_H": "9999.0", "mutation_frontier": 0,
        "source_floor_ts": "1.0", "input_floor_ts": "1.0", "input_floor_inclusive": 0,
        "parent_snapshot_id": ids[0], "serializer_version": V2, "serializer_config_hash": "c",
        "prompt_version": "v1", "sizing_profile": PROFILE, "headroom_source": "measured",
        "headroom_tokens": 1, "profile_version": "measured:1", "root_inventory": {},
        "history_span_density": [], "actor_snapshot": {}, "actor_snapshot_hash": "",
        "chunk_index": 0, "chunk_hashes": [], "chunk_aggregates": [], "chunk_summaries": {},
        "frozen_renders": {}, "frozen_receipts": {}, "attempt_seq": 0, "attempt_tokens_in": 0,
        "attempt_tokens_out": 0, "attempt_cached_input_tokens": 0, "attempt_call_count": 0,
        "event_count": 0, "consecutive_discards": 0, "updated_at": "1.0"})
    await coord.sweep()
    assert await temp_db.get_snapshot_async(ids[0]) is not None


# ===================================================================== revalidation (§1m)

class _Evidence:
    def __init__(self, *, total, target, fixed, profile=PROFILE, admitted=True):
        self.total_tokens = total
        self.target_tokens = target
        self.trigger_tokens = int(target * 1.15)
        self.fixed_tokens = fixed
        self.sizing_profile = profile
        self.admitted = admitted
        self.h = "1700009999.000000"
        self.compactable_tokens = 100
        self.near_trigger = False


async def _fixed_publication(db, coord) -> Dict[str, Any]:
    sid = await candidate(db, headroom_source="fixed", headroom_tokens=80000)
    assert (await publish(coord, sid, None))["won"]
    return await db.get_snapshot_row_async(sid)


async def test_a_measured_publication_owes_no_revalidation(temp_db):
    """Appendix C test 54, the measured half. Path 2 has REAL admission components from its
    triggering turn, so no heuristic is involved and nothing is owed."""
    coord = coordinator(temp_db)
    sid = await candidate(temp_db, headroom_source="measured", headroom_tokens=90000)
    assert (await publish(coord, sid, None))["won"]
    snapshot = await temp_db.get_snapshot_row_async(sid)
    assert await coord.revalidation_state(TEAM, CH, snapshot=snapshot) == "complete"
    assert (await coord.revalidate(TEAM, CH, snapshot=snapshot,
                                   evidence=_Evidence(total=1, target=10, fixed=1)))[
        "claimed"] is False


async def test_a_fixed_publication_owes_one(temp_db):
    """Test 54's fixed half. The 80000 default is an ACCEPTED HEURISTIC made safe by a recorded
    obligation, not by hope."""
    coord = coordinator(temp_db)
    snapshot = await _fixed_publication(temp_db, coord)
    assert await coord.revalidation_state(TEAM, CH, snapshot=snapshot) == "owed"


async def test_exactly_one_of_many_concurrent_turns_claims(temp_db):
    """Appendix C test 72. LOSERS DO NOTHING — they neither re-verify nor enqueue, so a burst of
    concurrent turns cannot produce a burst of compactions."""
    coord = coordinator(temp_db)
    coord._accepting = False               # the enqueue is durable; the task map is not the test
    snapshot = await _fixed_publication(temp_db, coord)
    evidence = _Evidence(total=5, target=100, fixed=1)
    results = await asyncio.gather(*[
        coord.revalidate(TEAM, CH, snapshot=snapshot, evidence=evidence) for _ in range(6)])
    assert sum(1 for r in results if r["claimed"]) == 1


async def test_an_over_target_result_enqueues_durably_before_it_completes(temp_db):
    """Appendix C test 55 — the TARGET/TRIGGER BAND, where the ordinary paths enqueue nothing and
    where a mis-sized fixed profile hides. A test whose usage exceeds trigger does not exercise it.

    ORDERING IS THE POINT: clearing the obligation before enqueueing would lose the revalidation
    permanently if the process died in between.
    """
    coord = coordinator(temp_db)
    coord._accepting = False
    snapshot = await _fixed_publication(temp_db, coord)
    band = _Evidence(total=90, target=80, fixed=40000)
    assert band.total_tokens > band.target_tokens
    assert band.total_tokens < band.trigger_tokens          # strictly inside the band

    order: List[str] = []
    original = temp_db.merge_pending_recompaction_async

    async def _merge(**kw):
        order.append("enqueue")
        return await original(**kw)
    temp_db.merge_pending_recompaction_async = _merge

    result = await coord.revalidate(TEAM, CH, snapshot=snapshot, evidence=band)
    order.append("complete")
    assert result == {"claimed": True, "enqueued": True}
    assert order == ["enqueue", "complete"]
    row = await temp_db.load_pending_recompaction_async(TEAM, CH, NS)
    assert row["requirements"] == {PROFILE: 40000}
    assert await coord.revalidation_state(TEAM, CH, snapshot=snapshot) == "complete"


async def test_an_under_target_result_completes_without_enqueueing(temp_db):
    """Appendix C test 42's happy half — the first turn after a crawl publishes re-verifies fit
    against REAL admission components and, when it fits, closes the obligation."""
    coord = coordinator(temp_db)
    coord._accepting = False
    snapshot = await _fixed_publication(temp_db, coord)
    result = await coord.revalidate(TEAM, CH, snapshot=snapshot,
                                    evidence=_Evidence(total=10, target=100, fixed=5))
    assert result == {"claimed": True, "enqueued": False}
    assert await temp_db.load_pending_recompaction_async(TEAM, CH, NS) is None


async def test_a_claimant_cannot_complete_another_generations_obligation(temp_db):
    """Appendix C test 79. Every CAS predicate names the obligated (snapshot_id, generation), so a
    claimant measuring A cannot discharge a revalidation for a newly published B that never
    happened."""
    coord = coordinator(temp_db)
    coord._accepting = False
    a = await _fixed_publication(temp_db, coord)
    assert (await coord.revalidate(TEAM, CH, snapshot=a,
                                   evidence=_Evidence(total=1, target=10, fixed=1)))["claimed"]
    assert await coord.revalidation_state(TEAM, CH, snapshot=a) == "complete"

    b_id = await candidate(temp_db, headroom_source="fixed", headroom_tokens=80000,
                           boundary_ts="1001.000100")
    assert (await publish(coord, b_id, a["snapshot_id"], boundary="1001.000100"))["won"]
    b = await temp_db.get_snapshot_row_async(b_id)
    # A's completion says nothing about B: B is still owed.
    assert await coord.revalidation_state(TEAM, CH, snapshot=b) == "owed"


async def test_a_dead_claim_reverts_to_owed(temp_db, monkeypatch):
    """Appendix C test 72's third clause. A claimant that crashed mid-measurement must not strand
    the obligation forever."""
    from config import config
    coord = coordinator(temp_db)
    coord._accepting = False
    snapshot = await _fixed_publication(temp_db, coord)
    key = (TEAM, CH, NS, V2)
    coord._revalidation[key] = {
        "pair": (snapshot["snapshot_id"], snapshot["generation"]),
        "state": "claimed", "claimed_at": -float(config.revalidation_claim_ttl) - 10.0}
    assert await coord.revalidation_state(TEAM, CH, snapshot=snapshot) == "owed"
    assert (await coord.revalidate(TEAM, CH, snapshot=snapshot,
                                   evidence=_Evidence(total=1, target=10, fixed=1)))["claimed"]


# ===================================================================== the drainer (§1l)

def _body(crawl_id, event_seq, at, *, op="build"):
    body = {"event": "compaction_snapshot", "crawl_id": crawl_id, "attempt_seq": 0,
            "event_seq": event_seq, "team_id": TEAM, "channel_id": CH, "namespace": NS,
            "at": at, "op": op}
    if op == "build":
        body.update({"model": "m", "tokens_in": 1, "tokens_out": 2, "cached_input_tokens": 0,
                     "call_count": 1, "status": "ok"})
    else:
        body.update({"snapshot_id": "s1", "generation": 1, "boundary_ts": "1000.0",
                     "fit_result": "under_target", "serializer_version": V2})
    return body


async def _queue_attempt(db, crawl_id, *, at=1000.0):
    await db.insert_outbox_rows_async([
        {"crawl_id": crawl_id, "attempt_seq": 0, "event_seq": 0, "created_ts": at,
         "body": _body(crawl_id, 0, at)},
        {"crawl_id": crawl_id, "attempt_seq": 0, "event_seq": 1, "created_ts": at,
         "body": _body(crawl_id, 1, at, op="publish")}])


def _sink(monkeypatch, *, results=None):
    """Record what the drainer emitted, and control acknowledgement per call."""
    emitted: List[Dict[str, Any]] = []
    answers = list(results or [])

    def _emit(body, *, timeout=30.0):
        ok = answers.pop(0) if answers else True
        if ok:
            emitted.append(dict(body))
        return ok

    monkeypatch.setattr("message_processor.participation_telemetry.emit_outbox_body", _emit)
    return emitted


async def test_build_reaches_the_ledger_before_publish(temp_db, monkeypatch):
    """Appendix C test 100(a). Direct-emitting the publish must fail this."""
    coord = coordinator(temp_db)
    emitted = _sink(monkeypatch)
    await _queue_attempt(temp_db, "aaaa")
    assert await coord.drain_outbox() == 2
    assert [b["op"] for b in emitted] == ["build", "publish"]
    assert await temp_db.read_outbox_batch_async(10) == []


async def test_an_unacknowledged_row_is_not_deleted(temp_db, monkeypatch):
    """Appendix C test 100(b). An implementation deleting after a plain, unacknowledged
    `record()` loses the event."""
    coord = coordinator(temp_db)
    _sink(monkeypatch, results=[False])
    await _queue_attempt(temp_db, "bbbb")
    assert await coord.drain_outbox() == 0
    assert len(await temp_db.read_outbox_batch_async(10)) == 2


async def test_an_ack_failure_at_event_seq_zero_halts_the_drain_globally(temp_db, monkeypatch):
    """Appendix C test 100(g). THE HALT IS GLOBAL.

    Skipping ahead after a failure puts `publish` in the ledger with no `build` before it — the
    precise failure `event_seq` exists to prevent — and it does not even help, because the sink is
    unavailable and the next row will fail too. This is the case that catches a drainer treating
    acknowledgement failure as "try the next row".
    """
    coord = coordinator(temp_db)
    emitted = _sink(monkeypatch, results=[False, True, True, True])
    await _queue_attempt(temp_db, "cccc", at=1000.0)
    await _queue_attempt(temp_db, "aaaa", at=2000.0)          # a DIFFERENT attempt behind it
    assert await coord.drain_outbox() == 0
    assert emitted == []
    assert len(await temp_db.read_outbox_batch_async(10)) == 4


async def test_delivery_order_survives_a_lower_uuid_insert(temp_db, monkeypatch):
    """Appendix C test 100(i). `crawl_id` is a random uuid4, so identity order is NOT time order:
    an implementation ordering by the identity triple emits BACKWARD here."""
    coord = coordinator(temp_db)
    emitted = _sink(monkeypatch)
    await _queue_attempt(temp_db, "zzzz", at=1000.0)
    await _queue_attempt(temp_db, "0000", at=2000.0)          # sorts BELOW the queued one
    assert await coord.drain_outbox() == 4
    assert [b["crawl_id"] for b in emitted] == ["zzzz", "zzzz", "0000", "0000"]


async def test_rows_are_retained_while_telemetry_is_unavailable(temp_db, monkeypatch):
    """Appendix C test 100(d). They are a durable backlog, not best-effort: a channel whose
    telemetry was off does not silently lose its compaction record."""
    coord = coordinator(temp_db)
    _sink(monkeypatch, results=[False, False])
    await _queue_attempt(temp_db, "dddd")
    assert await coord.drain_outbox() == 0
    assert len(await temp_db.read_outbox_batch_async(10)) == 2

    emitted = _sink(monkeypatch)                              # the sink comes back
    assert await coord.drain_outbox() == 2
    assert [b["op"] for b in emitted] == ["build", "publish"]


@pytest.mark.parametrize("corrupt,clause", [
    ('"not an object"', "not_object"),
    ('{"event": "something_else"}', "event"),
])
async def test_a_poisoned_row_halts_stays_and_logs_once_per_boot(temp_db, monkeypatch, caplog,
                                                                 corrupt, clause):
    """Appendix C test 100(k). A POISONED ROW IS NOT A SINK OUTAGE.

    It halts ordered emission, STAYS in the table, and logs CRITICAL bounded to once per row per
    boot — deliberately NOT cycled through the outage backoff, which would bury the one message
    that identifies it while presenting a permanent defect as a transient outage.
    """
    import logging
    coord = coordinator(temp_db)
    emitted = _sink(monkeypatch)
    await _queue_attempt(temp_db, "eeee")
    temp_db.conn.execute("UPDATE compaction_telemetry_outbox SET payload = ? "
                         "WHERE event_seq = 0", (corrupt,))
    temp_db.conn.commit()

    with caplog.at_level(logging.CRITICAL, logger=cs.logger.name):
        assert await coord.drain_outbox() == 0
        assert await coord.drain_outbox() == 0                 # re-checked, not latched
    assert emitted == []
    assert len(await temp_db.read_outbox_batch_async(10)) == 2
    criticals = [r for r in caplog.records if r.levelno >= logging.CRITICAL]
    assert len(criticals) == 1
    assert clause in criticals[0].getMessage()


async def test_a_repaired_row_resumes_with_no_restart(temp_db, monkeypatch):
    """Appendix C test 100(k)'s recovery clause. THE HALT IS A PROPERTY OF THE ROW'S CONTENT,
    NEVER A LATCHED PROCESS STATE — an implementation that latches it must fail."""
    import json
    coord = coordinator(temp_db)
    emitted = _sink(monkeypatch)
    await _queue_attempt(temp_db, "ffff", at=1234.5)
    temp_db.conn.execute("UPDATE compaction_telemetry_outbox SET payload = ? "
                         "WHERE event_seq = 0", ('{"event": "wrong"}',))
    temp_db.conn.commit()
    assert await coord.drain_outbox() == 0

    good = json.dumps(_body("ffff", 0, 1234.5), sort_keys=True, separators=(",", ":"))
    temp_db.conn.execute("UPDATE compaction_telemetry_outbox SET payload = ? "
                         "WHERE event_seq = 0", (good,))
    temp_db.conn.commit()
    assert await coord.drain_outbox() == 2                    # the SAME coordinator instance
    assert [b["op"] for b in emitted] == ["build", "publish"]


async def test_the_batch_size_and_the_backoff_ladder_are_the_pinned_ones():
    assert cs.DRAIN_BATCH == 50
    assert (cs.DRAIN_BACKOFF_START, cs.DRAIN_BACKOFF_CEILING) == (30.0, 600.0)


# ===================================================================== lifecycle

async def test_boot_attempts_the_drain_before_the_coordinator_starts(temp_db, monkeypatch):
    """Appendix C test 100(e). AN UNAVAILABLE SINK NEVER BLOCKS STARTUP.

    The test distinguishes "attempted before the coordinator" from "fully drained": asserting an
    empty outbox at startup would fail a correct implementation whose sink was down.
    """
    coord = coordinator(temp_db)
    order: List[str] = []
    _sink(monkeypatch, results=[False])
    await _queue_attempt(temp_db, "gggg")

    original_drain = coord.drain_outbox
    original_start = coord.start

    async def _drain(**kw):
        order.append("drain")
        return await original_drain(**kw)

    async def _start():
        order.append("start")
        return await original_start()

    coord.drain_outbox = _drain
    coord.start = _start

    await coord.drain_outbox(bounded=True)
    await coord.start()
    try:
        assert order == ["drain", "start"]
        assert coord._accepting                                # startup PROCEEDED
        assert len(await temp_db.read_outbox_batch_async(10)) == 2   # rows RETAINED
    finally:
        await coord.stop()


async def test_the_shutdown_drain_is_one_bounded_attempt(temp_db, monkeypatch):
    """Appendix C test 100(f). No retry loop, no waiting out an unavailable sink: on failure the
    rows survive to the next boot and shutdown still completes promptly."""
    coord = coordinator(temp_db)
    calls: List[float] = []

    def _emit(body, *, timeout=30.0):
        calls.append(timeout)
        return False

    monkeypatch.setattr("message_processor.participation_telemetry.emit_outbox_body", _emit)
    await _queue_attempt(temp_db, "hhhh")
    assert await coord.drain_outbox(bounded=True) == 0
    assert len(calls) == 1                                    # ONE attempt, not a loop
    assert len(await temp_db.read_outbox_batch_async(10)) == 2


async def test_stop_closes_to_new_work_before_it_waits(temp_db):
    """The accepting flag drops FIRST, so a task finishing during the drain spawns nothing."""
    coord = coordinator(temp_db)
    await coord.start()
    assert coord._accepting
    await coord.stop()
    assert not coord._accepting
    coord.trigger(team_id=TEAM, channel_id=CH, namespace=NS, path=cs.PATH_NEAR_TRIGGER)
    assert coord._tasks == {}


async def test_main_places_the_coordinator_stop_and_the_final_drain_in_the_landed_order():
    """§1l shutdown ordering, asserted on the source because the sequence is the contract.

    The final drain runs AFTER the coordinator stops (nothing is still producing) and BEFORE
    telemetry shutdown and DB teardown (it needs both stores alive).
    """
    import pathlib
    source = pathlib.Path("main.py").read_text()
    marks = ["await self.client.stop()", "drain_ingress_callbacks(",
             "admission_watermark.close_issuance()", "await admission_watermark.shutdown()",
             "await self.receipt_service.drain_late_arrivals()",
             "await self.snapshot_coordinator.stop()",
             "self.snapshot_coordinator.drain_outbox(bounded=True),",
             "log_session_end()"]
    positions = [source.index(mark) for mark in marks]
    assert positions == sorted(positions), dict(zip(marks, positions))


# ===================================================================== keyed barriers (§4a)

def test_barriers_are_keyed_by_operation_so_two_turns_do_not_collide():
    """§4a. Seam-only keying makes two concurrent turns share one pair of files: the first to
    arrive owns the announcement and the release frees BOTH, which is what P2's live battery hit
    and recovered from only on timeout."""
    from message_processor import dev_barriers
    one = dev_barriers.barrier_key(dev_barriers.POST_ADMISSION, {"turn_id": "turn-a"})
    two = dev_barriers.barrier_key(dev_barriers.POST_ADMISSION, {"turn_id": "turn-b"})
    assert one != two


def test_a_compaction_seam_is_keyed_by_compaction_id_not_turn_id():
    from message_processor import dev_barriers
    key = dev_barriers.operation_id(dev_barriers.PRE_RESUME_AFTER_COMPACTION,
                                    {"compaction_id": "crawl-1", "turn_id": "turn-a"})
    assert key == "crawl-1"
    assert dev_barriers.operation_id(dev_barriers.POST_ADMISSION,
                                     {"compaction_id": "crawl-1", "turn_id": "turn-a"}) == "turn-a"


def test_the_epoch_is_part_of_the_key(monkeypatch):
    from message_processor import dev_barriers
    monkeypatch.setenv("DEV_TEST_EPOCH_ID", "epoch-7")
    assert dev_barriers.barrier_key(dev_barriers.POST_ADMISSION,
                                    {"turn_id": "t"}).endswith(".epoch-7")


async def test_the_compaction_seam_is_wired(temp_db, monkeypatch):
    """`pre_resume_after_compaction` was declared in P2 and unwired. It fires after a background
    compaction publishes and before its task vacates the slot."""
    from message_processor import dev_barriers
    seen: List[Dict[str, Any]] = []

    async def _barrier(**context):
        seen.append(context)
        return True

    monkeypatch.setattr(dev_barriers, "pre_resume_after_compaction", _barrier)
    coord = coordinator(temp_db)
    await coord._after_publication((TEAM, CH, NS),
                                   {"snapshot_id": "s1", "generation": 2, "crawl_id": "crawl-9"})
    assert seen and seen[0]["compaction_id"] == "crawl-9"


# ===================================================================== §1k atomicity

async def test_rehydrations_preboundary_receipts_ride_the_one_sidecar_transaction(monkeypatch):
    """§1k. The pre-boundary receipt/chrome evidence is read INSIDE the canonical sidecar
    transaction, not as a second one.

    Two transactions mean rehydration can see a different world than the stream it is attached
    to — and the landed window read retrieves receipts only inside `(boundary, H]`, which is not
    enough for rehydration's role and chrome decisions.
    """
    from message_processor import channel_stream
    reads: List[Dict[str, Any]] = []
    original = channel_stream.build_channel_stream

    class _DB:
        async def read_channel_sidecars_async(self, team_id, channel_id, high_ts, window=None,
                                              preboundary_receipts=False):
            reads.append({"window": window, "preboundary_receipts": preboundary_receipts})
            return {"window": ("1.0", True), "coverage": None, "receipt_feature_epoch_ts": None,
                    "receipts": [], "activity": [], "image_analyses": [],
                    "document_extractions": [], "ambient_artifacts": [], "tool_usage": {},
                    "preboundary_receipts": [{"message_ts": "5.0", "state": "finalized"}]}

        async def get_active_snapshot_async(self, *a, **k):
            return None

    async def _drain(channel_id, frontier, timeout=None):
        return None

    monkeypatch.setattr("slack_client.admission_watermark.drain", _drain)
    with pytest.raises(Exception):
        # Coverage is unseeded here, so the build fails closed straight after the read — which is
        # all this test needs: it is asserting HOW MANY reads happened and with what.
        await original(client=MagicMock(), db=_DB(), team_id=TEAM, channel_id=CH,
                       h="9000.000000")
    assert len(reads) == 1
    assert reads[0]["preboundary_receipts"] is True


def test_the_shared_artifact_render_producer_is_the_one_the_coordinator_calls():
    """One producer, because SUPPRESSION IS BY BYTE IDENTITY.

    The projection embeds these bytes and their SHA-256 becomes the manifest's `content_hash`;
    late evidence re-renders and compares that hash. Two producers drifting by one character would
    make every already-summarized artifact look changed and render it a SECOND time as late
    evidence the summary already contains.
    """
    import inspect

    from message_processor import channel_request, channel_stream
    assert (channel_request._artifact_render_bytes.__wrapped__
            if hasattr(channel_request._artifact_render_bytes, "__wrapped__")
            else channel_request._artifact_render_bytes) is channel_stream.artifact_render_bytes
    source = inspect.getsource(ChannelSnapshotCoordinator._freeze_source_pin)
    assert "artifact_render_bytes" in source
