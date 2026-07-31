"""Sidecar artifacts around the window boundary.

An image analysis, a document summary, an ambient artifact and a tool-provenance record are all
derived from a MESSAGE. When that message is outside the window — or was excluded from the stream
as chrome or as a reply still being written — its marker has nothing to attach to. A marker that
rendered anyway would be a line of context with no message above it: the model would read it as
belonging to whatever message happened to precede it.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from message_processor.channel_stream import (
    CoveragePin,
    PinnedTuple,
    ReceiptRec,
    SERIALIZER_VERSION,
    SidecarPin,
    build_channel_stream,
    serialize_stream,
)
from slack_client import actor_tail as actor_tail_module
from slack_client import admission_watermark
from slack_client.normalizer import ORIGIN_HISTORY, ORIGIN_REPLIES, NormalizedMessage

TEAM = "T1"
CH = "C0BKX77NU66"
FLOOR = "1700000000.000000"
H = "1700009999.000000"
PRE = "1699990000.000000"          # before the floor
IN = "1700000500.000000"
COVERAGE = CoveragePin(start_ts=FLOOR, status="complete", reason="genesis")


def msg(ts, *, text="hello", sender="U1", sender_type="human", root=None,
        origin=ORIGIN_HISTORY, tombstone=False) -> NormalizedMessage:
    return NormalizedMessage(
        team_id=TEAM, channel_id=CH, ts=ts, thread_root_ts=root, subtype=None,
        sender_id=sender, sender_type=sender_type, raw_bot_name=None, text=text,
        files=(), reactions=(), edited_ts=None, is_broadcast=False, is_tombstone=tombstone,
        reply_count=None, latest_reply=None, mention_ids=(), origin=origin)


def cards(**kw) -> SidecarPin:
    base = dict(window=(FLOOR, True), receipts=(), receipt_feature_epoch_ts=None,
                coverage=COVERAGE, activity_roots=(), activity_event_ts=(),
                image_analyses=(), document_extractions=(), ambient_artifacts=(),
                tool_usage=(), versions_hash="h")
    base.update({k: tuple(v) if isinstance(v, list) else v for k, v in kw.items()})
    return SidecarPin(**base)


def pinned(messages, sidecars) -> PinnedTuple:
    return PinnedTuple(
        team_id=TEAM, channel_id=CH, snapshot=None, window=sidecars.window, H=H,
        fetch_snapshot=tuple(messages), sidecar_versions_hash=sidecars.versions_hash,
        actor_map=(("U1", "alice"),), actor_map_hash="a",
        serializer_version=SERIALIZER_VERSION, serializer_config_hash="c",
        capability_profile_hash="p", tool_schema_version="v",
        coverage=COVERAGE, receipt_feature_epoch_ts=sidecars.receipt_feature_epoch_ts,
        receipt_map=(), sidecars=sidecars)


def rendered(stream):
    return "\n".join(item.content for item in stream.items)


# ------------------------------------------------------------------ orphaned markers

def test_a_document_marker_for_a_message_outside_the_snapshot_renders_nowhere():
    sidecars = cards(document_extractions=[
        {"message_ts": PRE, "filename": "old.pdf", "summary": "an old summary"},
        {"message_ts": IN, "filename": "new.pdf", "summary": "a new summary"}])
    out = rendered(serialize_stream(pinned([msg(IN)], sidecars)))
    assert "[document (new.pdf): summary available]" in out
    assert "old.pdf" not in out


def test_an_image_analysis_for_a_pre_floor_message_renders_nowhere():
    sidecars = cards(image_analyses=[
        {"message_ts": PRE, "url": "u1", "analysis": "an old picture", "metadata":
            {"filename": "old.png"}}])
    assert "image analysis" not in rendered(serialize_stream(pinned([msg(IN)], sidecars)))


def test_an_ambient_artifact_for_a_pre_floor_message_renders_nowhere():
    sidecars = cards(ambient_artifacts=[
        {"source_ts": PRE, "kind": "link", "ref": "https://old", "title": "Old",
         "summary": "old", "status": "ready", "derivation_source": "fetch"}])
    assert "[link" not in rendered(serialize_stream(pinned([msg(IN)], sidecars)))


def test_tool_provenance_for_a_pre_floor_reply_renders_nowhere():
    sidecars = cards(tool_usage=[(PRE, ({"tool_name": "web_search", "gist": "q=x"},))])
    assert "used tools" not in rendered(serialize_stream(pinned([msg(IN)], sidecars)))


def test_markers_vanish_with_the_message_they_describe():
    """An in-flight reply is excluded from the stream; its provenance marker must go with it,
    not float up onto somebody else's message."""
    sidecars = cards(
        receipts=[ReceiptRec(IN, "in_flight", "turn-1", None)],
        tool_usage=[(IN, ({"tool_name": "web_search", "gist": "q=x"},))])
    stream = serialize_stream(pinned(
        [msg("1700000100.000000", text="a human line"),
         msg(IN, sender="B0", sender_type="self", text="half-written")], sidecars))
    assert stream.message_count == 1
    assert "used tools" not in rendered(stream)


def test_a_chrome_message_takes_its_markers_with_it():
    sidecars = cards(receipts=[ReceiptRec(IN, "chrome", "turn-1", None)],
                     document_extractions=[{"message_ts": IN, "filename": "x.pdf",
                                            "summary": "real"}])
    stream = serialize_stream(pinned(
        [msg(IN, sender="B0", sender_type="self", text=":hourglass: Thinking...")], sidecars))
    assert stream.message_items == ()
    assert "x.pdf" not in rendered(stream)


# ------------------------------------------------------------------ straddling threads

def test_a_pre_floor_root_does_not_render_but_its_in_window_reply_does():
    root = msg(PRE, text="the old root")
    reply = msg(IN, text="a fresh reply", root=PRE, origin=ORIGIN_REPLIES)
    # The root is outside the window, so the fetch never admitted it; only the reply is pinned.
    stream = serialize_stream(pinned([reply], cards()))
    assert stream.message_count == 1
    header = stream.message_items[0].content.split("\n")[0]
    assert header.endswith(f"thread={PRE}]")            # no snippet: the root is absent
    assert root.text not in rendered(stream)


def test_an_in_window_root_lends_its_snippet_to_the_reply():
    root = msg(FLOOR, text="deploy plan for friday")
    reply = msg(IN, text="ack", root=FLOOR, origin=ORIGIN_REPLIES)
    header = serialize_stream(pinned([root, reply], cards())).message_items[1].content
    assert '~"deploy plan for friday"' in header.split("\n")[0]


def test_a_tombstoned_in_window_root_says_so_rather_than_quoting_itself():
    root = msg(FLOOR, text="This message was deleted.", tombstone=True)
    reply = msg(IN, text="ack", root=FLOOR, origin=ORIGIN_REPLIES)
    header = serialize_stream(pinned([root, reply], cards())).message_items[1].content
    assert '~"[deleted]"' in header.split("\n")[0]


# ------------------------------------------------------------------ end to end

class _Client:
    def __init__(self, messages):
        self.self_team_id = TEAM
        self.bot_user_id = "U_BOT"
        self.bot_id = "B_BOT"
        self.app_id = "A_BOT"
        self.bot_handle = "bot"
        self.app = MagicMock()
        self.app.client.conversations_history = AsyncMock(
            return_value={"ok": True, "messages": messages})
        self.app.client.conversations_replies = AsyncMock(
            return_value={"ok": True, "messages": []})

    def is_own_message(self, msg):
        return bool(msg) and msg.get("bot_id") == self.bot_id

    def classify_sender(self, msg):
        return "self" if self.is_own_message(msg) else "human"

    async def resolve_usernames(self, ids, api_client, max_remote_lookups=25):
        return {uid: "alice" for uid in ids}


@pytest.fixture(autouse=True)
def _clean():
    admission_watermark.watermark.reset()
    actor_tail_module.actor_tail.reset()
    yield
    admission_watermark.watermark.reset()
    actor_tail_module.actor_tail.reset()


async def test_a_full_build_renders_only_in_window_artifacts():
    client = _Client([{"ts": IN, "user": "U1", "text": "here it is"}])
    db = MagicMock()
    db.get_active_snapshot_async = AsyncMock(return_value=None)
    db.clear_thread_dirty_async = AsyncMock(return_value=True)
    db.read_channel_sidecars_async = AsyncMock(return_value={
        "window": (FLOOR, True),
        "coverage": {"coverage_start_ts": FLOOR, "bootstrap_status": "complete",
                     "reason": "genesis"},
        "receipt_feature_epoch_ts": None, "receipts": [], "activity": [],
        "image_analyses": [{"message_ts": IN, "url": "https://x/files-pri/T-F9/a.png",
                            "analysis": "a chart", "metadata": None},
                           {"message_ts": PRE, "url": "u2", "analysis": "old", "metadata": None}],
        "document_extractions": [], "ambient_artifacts": [], "tool_usage": {},
        "versions_hash": "h"})
    stream = await build_channel_stream(client=client, db=db, team_id=TEAM, channel_id=CH, h=H)
    out = rendered(stream)
    assert "[image analysis (F9): a chart]" in out
    assert "old" not in out


# =====================================================================================
# P4 §1i — the FROZEN capture manifest, against a real database.
#
# The manifest is what stops an artifact the summary already saw from reappearing as "late"
# evidence forever. It is frozen at the source pin and inserted with the candidate; a live
# publication-time query is forbidden, because an artifact that completed after the pin but
# before publication would otherwise be marked captured though the summary never saw it.
# =====================================================================================

import pytest_asyncio  # noqa: E402,F401  (the suite's async plugin, imported for parity)

from database import DatabaseManager, PROD_NAMESPACE  # noqa: E402

NS = PROD_NAMESPACE
V2 = 2
MANIFEST_PROFILE = "gpt-5.6-luna:400000:320000:280000"
BOUNDARY = "1700005000.000000"
SOURCE = "1700000500.000000"          # a pre-boundary message
DB_CH = "C1"


@pytest.fixture
def manifest_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    db = DatabaseManager(platform="slack")
    yield db
    db.conn.close()


def _v2_snapshot(**kw):
    base = dict(team_id=TEAM, channel_id=DB_CH, namespace=NS, serializer_version=V2,
                boundary_ts=BOUNDARY, source_floor_ts="1.000000", parent_snapshot_id=None,
                prompt_version="v1", model="gpt-5.6-luna", source_hash="sh",
                payload_bytes=b"summary", anchor_payload_bytes=b"", mutation_frontier=0,
                headroom_source="measured", headroom_tokens=90000, effective_window=400000,
                sizing_profile=MANIFEST_PROFILE, fit_result="under_target")
    base.update(kw)
    return base


async def _publish(db, sid, previous=None, *, boundary=BOUNDARY):
    return await db.publish_compaction_candidate_async(
        team_id=TEAM, channel_id=DB_CH, namespace=NS, serializer_version=V2, snapshot_id=sid,
        expected_previous_id=previous, source_floor_ts="1.000000", boundary_ts=boundary,
        mutation_frontier=0, current_profile=MANIFEST_PROFILE)


def _ambient(db, status="pending"):
    db.conn.execute(
        "INSERT INTO ambient_artifacts (channel_id, source_ts, conversation_ts, kind, ref, "
        "title, summary, status, derivation_source) "
        "VALUES (?, ?, ?, 'link', 'https://x', 'Title', 'a summary', ?, 'fetch')",
        (DB_CH, SOURCE, SOURCE, status))
    return str(db.conn.execute(
        "SELECT id FROM ambient_artifacts ORDER BY id DESC LIMIT 1").fetchone()[0])


async def test_a_same_row_completion_is_detected_by_the_hash_and_status(manifest_db):
    """§7b: artifacts complete by MUTATING the same row id, so typed identity alone cannot see
    it. A pending capture must never suppress the later ready summary — the hash/status change
    is what surfaces it as late evidence."""
    row_id = _ambient(manifest_db, status="pending")
    sid = await manifest_db.insert_compaction_candidate_async(
        snapshot=_v2_snapshot(),
        manifest_rows=[{"artifact_namespace": "ambient_artifact", "row_id": row_id,
                        "source_ts": SOURCE, "captured_render_version": "v1",
                        "content_hash": "hash-of-the-pending-render",
                        "status_at_capture": "pending"}])
    assert (await _publish(manifest_db, sid))["won"]

    # Unchanged since capture: the summary already carries it, so it is not late evidence.
    assert await manifest_db.late_artifact_evidence_async(
        TEAM, DB_CH, sid, boundary_ts=BOUNDARY, high_ts="1700009999.000000") == []

    manifest_db.conn.execute("UPDATE ambient_artifacts SET status = 'ready'")
    late = await manifest_db.late_artifact_evidence_async(
        TEAM, DB_CH, sid, boundary_ts=BOUNDARY, high_ts="1700009999.000000")
    assert [(e["artifact_namespace"], e["row_id"]) for e in late] == [
        ("ambient_artifact", row_id)]
    assert late[0]["manifest_status_at_capture"] == "pending"
    assert late[0]["status"] == "ready"
    # The frozen capture hash rides along so the caller can finish the comparison for the
    # statusless namespaces, where only the render bytes can say whether anything changed.
    assert late[0]["manifest_content_hash"] == "hash-of-the-pending-render"


async def test_a_statusless_namespace_always_offers_its_capture_hash(manifest_db):
    """§1i: `document_extraction` and `tool_provenance` have NO native status and store the
    literal "complete", so the content hash alone carries the change signal."""
    manifest_db.conn.execute(
        "INSERT INTO documents (thread_id, message_ts, filename, mime_type, summary) "
        "VALUES (?, ?, 'plan.pdf', 'application/pdf', 'a summary')",
        (f"{DB_CH}:{SOURCE}", SOURCE))
    row_id = str(manifest_db.conn.execute("SELECT id FROM documents").fetchone()[0])
    sid = await manifest_db.insert_compaction_candidate_async(
        snapshot=_v2_snapshot(),
        manifest_rows=[{"artifact_namespace": "document_extraction", "row_id": row_id,
                        "source_ts": SOURCE, "captured_render_version": "v1",
                        "content_hash": "frozen-render-hash",
                        "status_at_capture": "complete"}])
    assert (await _publish(manifest_db, sid))["won"]

    late = await manifest_db.late_artifact_evidence_async(
        TEAM, DB_CH, sid, boundary_ts=BOUNDARY, high_ts="1700009999.000000")
    assert [e["manifest_content_hash"] for e in late] == ["frozen-render-hash"]


async def test_a_pinned_old_manifest_survives_supersession(manifest_db):
    """§7b: a turn's late evidence is computed against ITS PINNED snapshot's manifest, never
    the active one — an overlapping turn may pin S1 after S2 became active, and its evidence
    is S1's business. Supersession must therefore never remove a pinned old manifest."""
    row_id = _ambient(manifest_db, status="ready")
    old = await manifest_db.insert_compaction_candidate_async(
        snapshot=_v2_snapshot(),
        manifest_rows=[{"artifact_namespace": "ambient_artifact", "row_id": row_id,
                        "source_ts": SOURCE, "captured_render_version": "v1",
                        "content_hash": "old-hash", "status_at_capture": "ready"}])
    assert (await _publish(manifest_db, old))["won"]

    new = await manifest_db.insert_compaction_candidate_async(
        snapshot=_v2_snapshot(boundary_ts="1700006000.000000"), manifest_rows=[])
    assert (await _publish(manifest_db, new, old, boundary="1700006000.000000"))["won"]

    assert [r["row_id"] for r in await manifest_db.snapshot_manifest_async(old)] == [row_id]
    assert await manifest_db.snapshot_manifest_async(new) == []
    # And the still-pinned old snapshot still suppresses the artifact it captured.
    assert await manifest_db.late_artifact_evidence_async(
        TEAM, DB_CH, old, boundary_ts=BOUNDARY, high_ts="1700009999.000000") == []
    assert len(await manifest_db.late_artifact_evidence_async(
        TEAM, DB_CH, new, boundary_ts="1700006000.000000",
        high_ts="1700009999.000000")) == 1


async def test_an_inherited_manifest_row_keeps_the_parents_capture_state(manifest_db):
    """§1i: a descendant INHERITS the parent manifest, retaining the PARENT's content_hash and
    status_at_capture so a later same-row change is still detected as a change. Without the
    inheritance every artifact already inside the parent summary would reappear as late
    evidence forever."""
    row_id = _ambient(manifest_db, status="pending")
    parent_rows = [{"artifact_namespace": "ambient_artifact", "row_id": row_id,
                    "source_ts": SOURCE, "captured_render_version": "v1",
                    "content_hash": "parent-hash", "status_at_capture": "pending"}]
    parent = await manifest_db.insert_compaction_candidate_async(
        snapshot=_v2_snapshot(), manifest_rows=parent_rows)
    assert (await _publish(manifest_db, parent))["won"]

    child = await manifest_db.insert_compaction_candidate_async(
        snapshot=_v2_snapshot(boundary_ts="1700006000.000000", parent_snapshot_id=parent),
        manifest_rows=parent_rows)
    assert (await _publish(manifest_db, child, parent, boundary="1700006000.000000"))["won"]

    inherited = (await manifest_db.snapshot_manifest_async(child))[0]
    assert (inherited["content_hash"], inherited["status_at_capture"]) == (
        "parent-hash", "pending")
    assert await manifest_db.late_artifact_evidence_async(
        TEAM, DB_CH, child, boundary_ts="1700006000.000000",
        high_ts="1700009999.000000") == []

    manifest_db.conn.execute("UPDATE ambient_artifacts SET status = 'ready'")
    assert len(await manifest_db.late_artifact_evidence_async(
        TEAM, DB_CH, child, boundary_ts="1700006000.000000",
        high_ts="1700009999.000000")) == 1
