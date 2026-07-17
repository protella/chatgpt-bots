"""F51c — late-artifact addenda: derived ambient context that completes AFTER (or is folded
in DURING) a thread's compaction survives into the rebuilt context via the summary head.

The race: an ambient artifact (a slow link fetch, a deferred vision job) finishes after its
source message was already folded into a thread's compaction summary — or a message with a
long-ready artifact is compacted later. Either way the derived note would vanish (it never
lived in thread_state.messages, the summary was written without it, and the compacted message
no longer returns in the rebuilt tail). These tests exercise both trigger paths, the no-double
guarantee, the cap, restart durability (real temp SQLite), and the deletion lifecycle.
"""
import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from base_client import Message
from message_processor.ambient_memory import AmbientArtifactService, _Job
from message_processor.thread_management import ThreadManagementMixin
from message_processor.utilities import MessageUtilitiesMixin
from thread_manager import AsyncThreadStateManager, ThreadState

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


# --------------------------------------------------------------------------- harness

class _Proc(ThreadManagementMixin, MessageUtilitiesMixin):
    def __init__(self, db=None, openai_client=None):
        self.db = db
        self.thread_manager = AsyncThreadStateManager(db=db)
        self.openai_client = openai_client
        self.document_handler = None

    def log_info(self, *a, **k): pass
    def log_debug(self, *a, **k): pass
    def log_warning(self, *a, **k): pass
    def log_error(self, *a, **k): pass
    def _update_status(self, *a, **k): pass


def _hist(ts, text):
    return Message(text=text, user_id="U1", channel_id="C1", thread_id="100.0",
                   attachments=[],
                   metadata={"ts": ts, "is_bot": False, "sender_type": "human",
                             "bot_name": None, "username": "Peter", "reactions": None})


def _incoming(ts="200.0", text="latest question"):
    return Message(text=text, user_id="U1", channel_id="C1", thread_id="100.0",
                   attachments=[], metadata={"ts": ts})


def _client_with_history(history):
    client = MagicMock()
    client.get_thread_history = AsyncMock(return_value=history)
    client.name = "slack"
    client.user_cache = {}
    client.bot_user_id = "UBOT"
    return client


def _mock_openai(summary="ROLLED-UP SUMMARY"):
    oc = MagicMock()
    oc.create_text_response = AsyncMock(return_value=summary)
    return oc


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    from database import DatabaseManager
    db = DatabaseManager(platform="slack")
    yield db
    db.conn.close()


def _svc(db):
    return AmbientArtifactService(db=db, openai_client=_mock_openai(), channel_pulse=None)


def _link_art(summary="S", src="fetch"):
    return {"kind": "link", "title": "T", "summary": summary, "derivation_source": src}


async def _ready_artifact(db, source_ts, ref, *, kind="link", summary="S"):
    """Persist a ready ambient artifact directly (bypasses the workers)."""
    await db.insert_pending_ambient_artifact(
        channel_id="C1", source_ts=source_ts, conversation_ts="100.0", kind=kind, ref=ref)
    await db.set_ambient_artifact_ready(
        channel_id="C1", source_ts=source_ts, kind=kind, ref=ref,
        title="T", summary=summary, model="m", derivation_source="fetch")


# --------------------------------------------------------- completion-time trigger

async def test_late_completion_behind_boundary_records_addendum(temp_db):
    """Artifact completes AFTER its source message was compacted → addendum recorded."""
    temp_db.get_or_create_thread("C1:100.0", "C1")
    temp_db.save_thread_summary("C1:100.0", "Earlier stuff.", "101.5")

    svc = _svc(temp_db)
    job = _Job(kind="link", channel_id="C1", source_ts="101.0", conversation_ts="100.0",
               ref="https://x/a")
    await svc._record_late_addendum(job, _link_art(summary="derived body"))

    rows = await temp_db.get_thread_summary_addenda_async("C1:100.0")
    assert len(rows) == 1
    assert "derived body" in rows[0]["note"]
    assert rows[0]["source_ts"] == "101.0"


async def test_completion_still_in_tail_records_nothing(temp_db):
    """Artifact completes while its message is still in the live tail (source_ts > boundary):
    the normal batch-load renders it there, so NO addendum (no double-inject)."""
    temp_db.get_or_create_thread("C1:100.0", "C1")
    temp_db.save_thread_summary("C1:100.0", "Earlier stuff.", "101.5")

    svc = _svc(temp_db)
    job = _Job(kind="link", channel_id="C1", source_ts="102.0", conversation_ts="100.0",
               ref="https://x/b")
    await svc._record_late_addendum(job, _link_art())

    rows = await temp_db.get_thread_summary_addenda_async("C1:100.0")
    assert rows == []


async def test_no_summary_records_nothing(temp_db):
    """A thread that never compacted has no summary to be behind — no addendum."""
    temp_db.get_or_create_thread("C1:100.0", "C1")
    svc = _svc(temp_db)
    job = _Job(kind="link", channel_id="C1", source_ts="101.0", conversation_ts="100.0",
               ref="https://x/c")
    await svc._record_late_addendum(job, _link_art())
    assert await temp_db.get_thread_summary_addenda_async("C1:100.0") == []


async def test_unfurl_source_skipped(temp_db):
    """Parity with pulse/batch-load: an unfurl note is F48's preview, never re-described."""
    temp_db.get_or_create_thread("C1:100.0", "C1")
    temp_db.save_thread_summary("C1:100.0", "Earlier.", "101.5")
    svc = _svc(temp_db)
    job = _Job(kind="link", channel_id="C1", source_ts="101.0", conversation_ts="100.0",
               ref="https://x/d")
    await svc._record_late_addendum(job, _link_art(src="unfurl"))
    assert await temp_db.get_thread_summary_addenda_async("C1:100.0") == []


async def test_ready_funnel_records_addendum_end_to_end(temp_db):
    """Drive the real _ready funnel (set_ready + pulse + addendum) to confirm the wiring."""
    temp_db.get_or_create_thread("C1:100.0", "C1")
    temp_db.save_thread_summary("C1:100.0", "Earlier.", "101.5")
    svc = _svc(temp_db)
    job = _Job(kind="link", channel_id="C1", source_ts="101.0", conversation_ts="100.0",
               ref="https://x/e")
    await svc._ready(job, "link", title="Title", summary="the derived summary",
                     model="m", derivation_source="fetch")
    rows = await temp_db.get_thread_summary_addenda_async("C1:100.0")
    assert len(rows) == 1 and "the derived summary" in rows[0]["note"]


# --------------------------------------------------------- compaction-time capture

async def test_compaction_captures_already_ready_artifact(temp_db):
    """A message with a long-ready artifact gets compacted later → its derived note is
    captured into an addendum at compaction time (it was never in thread_state.messages)."""
    temp_db.get_or_create_thread("C1:100.0", "C1")
    await _ready_artifact(temp_db, "101.0", "https://x/f", summary="ready-before-compaction")

    proc = _Proc(db=temp_db, openai_client=_mock_openai("first summary"))
    state = ThreadState(thread_ts="100.0", channel_id="C1")
    dropped = [{"role": "user", "content": "see https://x/f", "metadata": {"ts": "101.0"}}]
    await proc._write_thread_summary(state, "C1:100.0", dropped)

    rows = await temp_db.get_thread_summary_addenda_async("C1:100.0")
    assert len(rows) == 1 and "ready-before-compaction" in rows[0]["note"]

    # And the live summary head already carries it.
    head = next(m for m in state.messages
                if (m.get("metadata") or {}).get("type") == "thread_summary")
    assert "ready-before-compaction" in head["content"]
    assert "Context that arrived after this summary was written:" in head["content"]


async def test_completion_and_compaction_paths_do_not_double_record(temp_db):
    """The completion path and the compaction path are idempotent on (thread, source, kind,
    ref): recording via both yields exactly one row."""
    temp_db.get_or_create_thread("C1:100.0", "C1")
    temp_db.save_thread_summary("C1:100.0", "Earlier.", "101.5")
    await _ready_artifact(temp_db, "101.0", "https://x/g", summary="once")

    # Completion path first.
    svc = _svc(temp_db)
    job = _Job(kind="link", channel_id="C1", source_ts="101.0", conversation_ts="100.0",
               ref="https://x/g")
    await svc._record_late_addendum(job, _link_art(summary="once"))

    # Compaction path second — same artifact.
    proc = _Proc(db=temp_db, openai_client=_mock_openai())
    state = ThreadState(thread_ts="100.0", channel_id="C1")
    await proc._capture_late_artifacts_for_span(
        state, "C1:100.0",
        [{"role": "user", "content": "x", "metadata": {"ts": "101.0"}}])

    rows = await temp_db.get_thread_summary_addenda_async("C1:100.0")
    assert len(rows) == 1


# ------------------------------------------------------------------ cap + ordering

async def test_addenda_cap_respected(temp_db):
    temp_db.get_or_create_thread("C1:100.0", "C1")
    temp_db.save_thread_summary("C1:100.0", "Earlier.", "9999.0")  # all sources are behind it
    svc = _svc(temp_db)
    from database import _MAX_SUMMARY_ADDENDA_PER_THREAD as CAP
    for i in range(CAP + 5):
        job = _Job(kind="link", channel_id="C1", source_ts=f"1{i:03d}.0",
                   conversation_ts="100.0", ref=f"https://x/{i}")
        await svc._record_late_addendum(job, _link_art(summary=f"s{i}"))
    rows = await temp_db.get_thread_summary_addenda_async("C1:100.0")
    assert len(rows) == CAP


async def test_addenda_ordered_numerically(temp_db):
    """source_ts is TEXT but numeric — ordering must be by REAL value, not string collation."""
    temp_db.get_or_create_thread("C1:100.0", "C1")
    temp_db.save_thread_summary("C1:100.0", "Earlier.", "2000.0")
    svc = _svc(temp_db)
    for ts, ref in [("999.0", "a"), ("1000.0", "b"), ("101.0", "c")]:
        job = _Job(kind="link", channel_id="C1", source_ts=ts, conversation_ts="100.0",
                   ref=f"https://x/{ref}")
        await svc._record_late_addendum(job, _link_art(summary=ref))
    rows = await temp_db.get_thread_summary_addenda_async("C1:100.0")
    assert [r["source_ts"] for r in rows] == ["101.0", "999.0", "1000.0"]


# ---------------------------------------------------------- restart / rebuild path

async def test_addendum_visible_in_cold_rebuild(temp_db):
    """Restart-safe: a DB-backed addendum surfaces in the summary head after a cold rebuild,
    while the compacted-away source message stays out of the tail."""
    temp_db.get_or_create_thread("C1:100.0", "C1")
    temp_db.save_thread_summary("C1:100.0", "Earlier: they planned the launch.", "101.5")
    # Record a late addendum for a behind-boundary message.
    svc = _svc(temp_db)
    job = _Job(kind="link", channel_id="C1", source_ts="101.0", conversation_ts="100.0",
               ref="https://x/late")
    await svc._record_late_addendum(job, _link_art(summary="LATE DERIVED CONTEXT"))

    # Fresh process (in-memory state gone) does a cold rebuild from Slack + DB.
    proc = _Proc(db=temp_db)
    history = [_hist("101.0", "old covered message"),   # behind boundary: excluded from tail
               _hist("102.0", "fresh tail message")]    # after boundary: in tail
    state = await proc._get_or_rebuild_thread_state(_incoming(), _client_with_history(history))

    head = state.messages[0]
    assert (head.get("metadata") or {}).get("type") == "thread_summary"
    assert "LATE DERIVED CONTEXT" in head["content"]

    joined = json.dumps([m.get("content") for m in state.messages])
    assert "fresh tail message" in joined
    assert "old covered message" not in joined   # source message not resurrected in the tail


# ------------------------------------------------------------- deletion lifecycle

async def test_delete_by_source_removes_addendum(temp_db):
    temp_db.get_or_create_thread("C1:100.0", "C1")
    temp_db.save_thread_summary("C1:100.0", "Earlier.", "101.5")
    svc = _svc(temp_db)
    job = _Job(kind="link", channel_id="C1", source_ts="101.0", conversation_ts="100.0",
               ref="https://x/del")
    await svc._record_late_addendum(job, _link_art())
    assert len(await temp_db.get_thread_summary_addenda_async("C1:100.0")) == 1

    await temp_db.delete_ambient_artifacts_by_source("C1", "101.0")
    assert await temp_db.get_thread_summary_addenda_async("C1:100.0") == []


async def test_delete_by_file_id_removes_image_addendum(temp_db):
    temp_db.get_or_create_thread("C1:100.0", "C1")
    temp_db.save_thread_summary("C1:100.0", "Earlier.", "101.5")
    svc = _svc(temp_db)
    job = _Job(kind="image", channel_id="C1", source_ts="101.0", conversation_ts="100.0",
               ref="FILEID123")
    await svc._record_late_addendum(
        job, {"kind": "image", "title": None, "summary": "a chart",
              "derivation_source": "vision_worker"})
    assert len(await temp_db.get_thread_summary_addenda_async("C1:100.0")) == 1

    await temp_db.delete_ambient_artifacts_by_file_id("FILEID123")
    assert await temp_db.get_thread_summary_addenda_async("C1:100.0") == []


async def test_delete_thread_summary_cascades_addenda(temp_db):
    temp_db.get_or_create_thread("C1:100.0", "C1")
    temp_db.save_thread_summary("C1:100.0", "Earlier.", "101.5")
    svc = _svc(temp_db)
    job = _Job(kind="link", channel_id="C1", source_ts="101.0", conversation_ts="100.0",
               ref="https://x/cascade")
    await svc._record_late_addendum(job, _link_art())
    temp_db.delete_thread_summary("C1:100.0")
    assert await temp_db.get_thread_summary_addenda_async("C1:100.0") == []


async def test_retention_sweep_cascades_addenda(temp_db):
    """Finding 8: when an ambient artifact ages out of the retention sweep, its late-artifact
    addendum must die in the SAME operation. Otherwise the derived note lingers indefinitely in
    the summary head and keeps occupying one of the per-thread addenda cap slots."""
    temp_db.get_or_create_thread("C1:100.0", "C1")
    temp_db.save_thread_summary("C1:100.0", "Earlier.", "101.5")
    svc = _svc(temp_db)

    # An EXPIRED artifact with a matching addendum (behind the compaction boundary).
    await temp_db.insert_pending_ambient_artifact(
        channel_id="C1", source_ts="101.0", conversation_ts="100.0", kind="link",
        ref="https://x/expired", expires_at="2000-01-01 00:00:00")
    await svc._record_late_addendum(
        _Job(kind="link", channel_id="C1", source_ts="101.0", conversation_ts="100.0",
             ref="https://x/expired"), _link_art(summary="aged out"))

    # A FRESH artifact + addendum that must SURVIVE the sweep.
    await temp_db.insert_pending_ambient_artifact(
        channel_id="C1", source_ts="101.2", conversation_ts="100.0", kind="link",
        ref="https://x/fresh")
    await svc._record_late_addendum(
        _Job(kind="link", channel_id="C1", source_ts="101.2", conversation_ts="100.0",
             ref="https://x/fresh"), _link_art(summary="still fresh"))
    assert len(await temp_db.get_thread_summary_addenda_async("C1:100.0")) == 2

    swept = temp_db.delete_expired_ambient_artifacts(days=30)
    # The sweep returns the DISTINCT thread keys whose addenda it deleted (for warm-thread refresh).
    assert swept == ["C1:100.0"]
    rows = await temp_db.get_thread_summary_addenda_async("C1:100.0")
    assert [r["ref"] for r in rows] == ["https://x/fresh"]  # expired cascaded, fresh untouched


async def test_sweep_result_marks_warm_threads_for_refresh(temp_db):
    """F51d: the cleanup worker feeds the sweep's returned thread keys into mark_needs_refresh, so
    an active warm thread stops sending an expired note from its in-memory summary head. Mirrors
    the worker loop directly (sweep → mark each returned key)."""
    temp_db.get_or_create_thread("C1:100.0", "C1")
    temp_db.save_thread_summary("C1:100.0", "Earlier.", "101.5")
    tm = AsyncThreadStateManager(db=temp_db)
    svc = _svc(temp_db)

    await temp_db.insert_pending_ambient_artifact(
        channel_id="C1", source_ts="101.0", conversation_ts="100.0", kind="link",
        ref="https://x/expired", expires_at="2000-01-01 00:00:00")
    await svc._record_late_addendum(
        _Job(kind="link", channel_id="C1", source_ts="101.0", conversation_ts="100.0",
             ref="https://x/expired"), _link_art(summary="aged out"))
    tm.consume_needs_refresh("C1:100.0")  # clear the flag the insert itself set

    swept_keys = temp_db.delete_expired_ambient_artifacts(days=30)
    assert swept_keys == ["C1:100.0"]
    # The worker loop marks each returned key, fail-soft per thread.
    for thread_key in swept_keys:
        tm.mark_needs_refresh(thread_key)

    assert tm.consume_needs_refresh("C1:100.0") is True


# ------------------------------------------------------- warm-thread refresh (Finding 9)

def _client_with_thread_manager(tm):
    client = MagicMock()
    client.processor.thread_manager = tm
    return client


async def test_addendum_insert_marks_warm_thread_refresh(temp_db):
    """Finding 9: a late addendum landing behind the compaction boundary marks the warm thread
    for refresh, so its next turn rebuilds from Slack and folds the note in — instead of answering
    from the stale summary head until a cold rebuild or the next compaction."""
    temp_db.get_or_create_thread("C1:100.0", "C1")
    temp_db.save_thread_summary("C1:100.0", "Earlier.", "101.5")
    tm = AsyncThreadStateManager(db=temp_db)
    svc = _svc(temp_db)
    svc._client = _client_with_thread_manager(tm)

    job = _Job(kind="link", channel_id="C1", source_ts="101.0", conversation_ts="100.0",
               ref="https://x/refresh")
    await svc._record_late_addendum(job, _link_art(summary="late body"))

    assert len(await temp_db.get_thread_summary_addenda_async("C1:100.0")) == 1
    assert tm.consume_needs_refresh("C1:100.0") is True


async def test_no_insert_leaves_thread_unmarked(temp_db):
    """Only a REAL insert marks the thread. A message still in the live tail records no addendum,
    so it must not spuriously flag a rebuild."""
    temp_db.get_or_create_thread("C1:100.0", "C1")
    temp_db.save_thread_summary("C1:100.0", "Earlier.", "101.5")
    tm = AsyncThreadStateManager(db=temp_db)
    svc = _svc(temp_db)
    svc._client = _client_with_thread_manager(tm)

    job = _Job(kind="link", channel_id="C1", source_ts="102.0", conversation_ts="100.0",
               ref="https://x/tail")  # source_ts > boundary → no addendum
    await svc._record_late_addendum(job, _link_art())

    assert await temp_db.get_thread_summary_addenda_async("C1:100.0") == []
    assert tm.consume_needs_refresh("C1:100.0") is False


async def test_refresh_marking_failure_is_fail_soft(temp_db):
    """Fail-soft: a broken thread-manager handle must never affect the addendum or the artifact —
    the note is still durably recorded."""
    temp_db.get_or_create_thread("C1:100.0", "C1")
    temp_db.save_thread_summary("C1:100.0", "Earlier.", "101.5")
    boom_tm = MagicMock()
    boom_tm.mark_needs_refresh.side_effect = RuntimeError("thread manager is down")
    svc = _svc(temp_db)
    svc._client = _client_with_thread_manager(boom_tm)

    job = _Job(kind="link", channel_id="C1", source_ts="101.0", conversation_ts="100.0",
               ref="https://x/soft")
    await svc._record_late_addendum(job, _link_art(summary="still recorded"))

    rows = await temp_db.get_thread_summary_addenda_async("C1:100.0")
    assert len(rows) == 1 and "still recorded" in rows[0]["note"]
