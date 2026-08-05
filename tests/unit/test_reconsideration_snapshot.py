"""The pure reconsideration snapshot (STALE_RECONSIDERATION §4c) and the fresh-context rule.

The seam rebuilds one channel window for a reconsideration pass. Its whole contract is
NEGATIVE: no durable anchor write, no dirty-state clearing, no actor-tail writes, no telemetry,
no dev barrier — the one permitted side effect is the in-memory username-cache fill. The
fresh-context half is the memo split: pinned evidence rides byte-identical, stream-derived
entries (`stream_actors`, and `roster` transitively) recompute from the fresh stream.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from message_processor import channel_request, channel_stream, participation_telemetry
from slack_client import actor_tail as actor_tail_module
from slack_client import admission_watermark

TEAM = "T1"
CH = "C0BKX77NU66"
FLOOR = "1700000000.000000"
H = "1700009999.000000"

_ANCHOR = {"floor_ts": FLOOR, "selection_version": 1}


class _Client:
    def __init__(self):
        self.self_team_id = TEAM
        self.bot_user_id = "U_BOT"
        self.bot_id = "B_BOT"
        self.app_id = "A_BOT"
        self.bot_handle = "chatgpt-dev"
        self.app = MagicMock()
        self.app.client.conversations_history = AsyncMock(
            return_value={"ok": True, "messages": []})
        self.app.client.conversations_replies = AsyncMock(
            return_value={"ok": True, "messages": []})
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
        # The in-memory cache fill: the ONE side effect §4c permits the snapshot.
        self.resolved.append(list(ids))
        return {uid: f"name-{uid}" for uid in ids}


def _db(*, activity_roots=None):
    db = MagicMock()
    db.read_channel_window_anchor_async = AsyncMock(return_value={
        "anchor": _ANCHOR,
        "inventory": {"inventory_start_ts": FLOOR, "bootstrap_status": "complete",
                      "reason": "genesis"},
    })
    db.read_channel_discovery_roots_async = AsyncMock(return_value={
        "activity_roots": dict(activity_roots or {}),
        "receipt_roots": (),
    })
    db.read_channel_sidecars_for_async = AsyncMock(return_value={
        "ids": [], "receipt_feature_epoch_ts": None, "receipts": [],
        "image_analyses": [], "document_extractions": [], "ambient_artifacts": [],
        "tool_usage": {}, "versions_hash": "h",
    })
    db.advance_channel_window_anchor_async = AsyncMock(return_value=True)
    db.clear_thread_dirty_async = AsyncMock(return_value=True)
    db.delete_thread_activity_if_unchanged_async = AsyncMock(return_value=True)
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


# ------------------------------------------------------------------ snapshot purity


async def test_snapshot_writes_nothing_durable_and_emits_no_telemetry(client, monkeypatch):
    """§8: no durable/anchor/actor-tail/telemetry mutation; the username fill is permitted.

    The window carries a threaded root known dirty to the index, so the fan-out runs the very
    branch that clears dirty state on a production build — and must not here (probe=True).
    """
    root, reply = "1700000100.000000", "1700000200.000000"
    client.app.client.conversations_history.return_value = {
        "ok": True, "messages": [raw(root, reply_count=1, latest_reply=reply)]}
    client.app.client.conversations_replies.return_value = {
        "ok": True, "messages": [raw(root, reply_count=1), raw(reply, root=root)]}
    db = _db(activity_roots={root: "evt-1"})

    tail_writes = []
    monkeypatch.setattr(channel_stream.actor_tail_module, "reconcile_window",
                        lambda *a, **k: tail_writes.append((a, k)))
    telemetry_rows = []
    monkeypatch.setattr(participation_telemetry, "stream_render",
                        lambda **k: telemetry_rows.append(("stream_render", k)))
    monkeypatch.setattr(participation_telemetry, "record",
                        lambda *a, **k: telemetry_rows.append((a, k)))

    result = await channel_stream.build_reconsideration_snapshot(
        client=client, db=db, team_id=TEAM, channel_id=CH, trigger_ts=H)

    # The build genuinely ran: the rebuilt stream contains the thread.
    assert {m.ts for m in result.stream.pinned.fetch_snapshot} == {root, reply}
    assert result.anchor_advanced is False
    # Forbidden side effects, each by name.
    db.advance_channel_window_anchor_async.assert_not_called()
    db.clear_thread_dirty_async.assert_not_called()
    db.delete_thread_activity_if_unchanged_async.assert_not_called()
    assert tail_writes == []
    assert telemetry_rows == []
    # The one permitted side effect happened: names were resolved into the in-memory cache.
    assert client.resolved


async def test_dev_barrier_skipped_even_when_enabled(client, monkeypatch):
    """§4c: `skip_dev_barrier=True` skips `post_admission` outright — the barrier writes files
    and can block, and omitting `barrier_context` does not skip it. The full builder, under the
    same (enabled) barrier, still reaches it — proving the flag is what does the skipping."""
    barrier_calls = []

    async def _post_admission(**context):
        barrier_calls.append(context)
        return True

    monkeypatch.setattr(channel_stream.dev_barriers, "post_admission", _post_admission)

    await channel_stream.build_reconsideration_snapshot(
        client=client, db=_db(), team_id=TEAM, channel_id=CH, trigger_ts=H)
    assert barrier_calls == []

    await channel_stream.build_channel_stream(
        client=client, db=_db(), team_id=TEAM, channel_id=CH, h=H)
    assert len(barrier_calls) == 1


async def test_one_shared_deadline_and_concurrent_periphery_origin(client, monkeypatch):
    """§4c: composed like the full builder — one absolute deadline shared by both components,
    periphery and origin CONCURRENT (each waits for the other to start; sequential composition
    would deadlock and fail the timeout), and the periphery phase runs probe=True."""
    root, reply = "1700000100.000000", "1700000200.000000"
    client.app.client.conversations_history.return_value = {
        "ok": True, "messages": [raw(root, reply_count=1, latest_reply=reply)]}
    client.app.client.conversations_replies.return_value = {
        "ok": True, "messages": [raw(root, reply_count=1), raw(reply, root=root)]}

    started = {}
    both_started = asyncio.Event()
    real_pin = channel_stream.build_channel_pin
    real_origin = channel_stream.fetch_origin_thread

    async def _pin(prepared, **kwargs):
        started["periphery"] = (kwargs["deadline_at"], kwargs.get("probe"))
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), 2)
        return await real_pin(prepared, **kwargs)

    async def _origin(c, channel_id, origin_root_ts, h, budget, trigger_ts):
        started["origin"] = (budget.deadline_at, None)
        if len(started) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), 2)
        return await real_origin(c, channel_id, origin_root_ts, h, budget, trigger_ts)

    monkeypatch.setattr(channel_stream, "build_channel_pin", _pin)
    monkeypatch.setattr(channel_stream, "fetch_origin_thread", _origin)

    result = await channel_stream.build_reconsideration_snapshot(
        client=client, db=_db(), team_id=TEAM, channel_id=CH,
        trigger_ts=H, origin_root_ts=root)

    periphery_deadline, probe = started["periphery"]
    origin_deadline, _ = started["origin"]
    assert periphery_deadline == origin_deadline
    assert probe is True
    assert result.stream.pinned.origin_root_ts == root


# ------------------------------------------------------------------ the memo split


def test_memo_tripwire_every_written_key_is_classified():
    """§4c tripwire: every memo key written anywhere in channel_request.py appears in exactly
    one of the two sets, so a future memo entry cannot silently pick a side."""
    tree = ast.parse(inspect.getsource(channel_request))
    written = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if not isinstance(target, ast.Subscript):
                continue
            base = target.value
            is_memo = ((isinstance(base, ast.Name) and base.id == "memo")
                       or (isinstance(base, ast.Attribute) and base.attr == "memo"))
            if not is_memo:
                continue
            key = target.slice
            assert isinstance(key, ast.Constant) and isinstance(key.value, str), (
                "memo keys must be literal strings so the tripwire can classify them")
            written.add(key.value)

    pinned = {"time_suffix", "job_state_notes", "memory", "topic", "requester", "custom"}
    stream_derived = channel_request.STREAM_DERIVED_MEMO_KEYS
    assert written, "the scan found no memo writes; the tripwire itself is broken"
    assert not pinned & stream_derived
    unclassified = written - (pinned | stream_derived)
    assert not unclassified, f"memo keys written but classified in neither set: {unclassified}"
    # Neither set names a phantom key nothing writes.
    assert stream_derived <= written
    assert pinned <= written


def _context_with_populated_memo():
    from tests.unit.channel_turn_harness import build_stream, normalized, thread_config

    stream = build_stream([normalized("10.0", sender_id="U1")])
    ctx = channel_request.ChannelTurnContext(
        stream=stream, steering=SimpleNamespace(developer_policy=None, user_facts=None),
        thread_config=thread_config(), channel_id=CH, team_id=TEAM,
        trigger_ts="10.0", origin_thread_ts=None,
        requester=channel_request.RequesterFacts(user_id="U1", real_name="Alice",
                                                 sender_type="human"))
    processor = MagicMock()
    processor._build_time_suffix_context.return_value = "[time: pinned]"
    processor._build_generation_inflight_note.return_value = None
    processor._build_research_inflight_note.return_value = "[research running]"
    client = SimpleNamespace(bot_user_id="U_BOT")

    ctx.time_suffix(processor)
    ctx.job_state_notes(processor)
    channel_request.build_evidence_items(ctx, client=client, request_config={}, registry=None)
    return ctx, client


def test_fresh_context_pins_memo_bytes_and_recomputes_stream_derived_keys():
    """§4c: `fresh_turn_context` replaces stream and memo; the copied memo keeps the pinned
    entries as the SAME objects (byte-identical evidence) and drops the stream-derived ones, so
    a new actor in the fresh stream reaches the roster evidence."""
    from tests.unit.channel_turn_harness import build_stream, normalized

    ctx, client = _context_with_populated_memo()
    assert "stream_actors" in ctx.memo and "roster" in ctx.memo
    original_memo = dict(ctx.memo)

    fresh_stream = build_stream([normalized("10.0", sender_id="U1"),
                                 normalized("20.0", sender_id="U_NEW")])
    fresh = channel_request.fresh_turn_context(ctx, fresh_stream)

    assert fresh.stream is fresh_stream
    assert fresh.memo is not ctx.memo
    # Stream-derived entries are ABSENT and will lazily recompute.
    assert "stream_actors" not in fresh.memo
    assert "roster" not in fresh.memo
    # Every pinned entry rides as the same object — byte-identical across the pass.
    for key in ("time_suffix", "job_state_notes", "memory", "topic", "requester", "custom"):
        assert key in fresh.memo
        assert fresh.memo[key] is original_memo[key]

    # The recomputation is real: the new actor reaches the roster evidence of the fresh
    # context, and the ORIGINAL context's memo is untouched by it.
    fresh_items = channel_request.build_evidence_items(
        fresh, client=client, request_config={}, registry=None)
    fresh_texts = [item["content"] for item in fresh_items]
    assert any("user-U_NEW" in text for text in fresh_texts)
    assert {a.user_id for a in fresh.stream_actors} == {"U1", "U_NEW"}
    assert ctx.memo["roster"] is original_memo["roster"]
    assert ctx.memo["stream_actors"] is original_memo["stream_actors"]
    assert "user-U_NEW" not in (original_memo["roster"] or "")
