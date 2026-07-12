"""F32: thread-scoped code-interpreter containers.

The behaviours worth defending, in rough order of how much they'd hurt to get wrong:

1. A dead container id must never reach the API — it 404s and costs the user their whole turn.
2. Nothing here may disable code interpreter. Every failure degrades to `auto` (an ephemeral
   container), not to a missing tool.
3. A reused container's listing is CUMULATIVE. Without a durable record of what we already
   uploaded, a restart mid-conversation re-posts turn 1's chart on turn 2.
4. Staleness is computed in SQL (UTC), never against Python's local clock.
"""
import sqlite3
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from database import DatabaseManager
from message_processor.containers import AUTO_CONTAINER, ContainerManager, is_container_gone


def _api_error(status: int, message: str) -> Exception:
    exc = Exception(message)
    exc.status_code = status
    return exc


CONTAINER_GONE = _api_error(404, "Container with id 'cntr_dead' not found.")


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = DatabaseManager("test")
        db.db_path = f"{tmpdir}/test.db"
        db.conn = sqlite3.connect(db.db_path, check_same_thread=False, isolation_level=None)
        db.conn.row_factory = sqlite3.Row
        db.init_schema()
        yield db
        db.conn.close()


def _openai(container_id="cntr_new", *, status="running", create_error=None,
            retrieve_error=None):
    """A stand-in OpenAI client exposing just the container surface we touch."""
    raw = MagicMock()
    raw.containers.create = AsyncMock(
        side_effect=create_error) if create_error else AsyncMock(
        return_value=MagicMock(id=container_id))
    raw.containers.retrieve = AsyncMock(
        side_effect=retrieve_error) if retrieve_error else AsyncMock(
        return_value=MagicMock(status=status))
    raw.containers.delete = AsyncMock(return_value=None)
    client = MagicMock()
    client.client = raw
    return client, raw


class TestIsContainerGone:
    """Only a container 404 may unbind a thread — misfiring here nukes a healthy container."""

    def test_container_404_is_gone(self):
        assert is_container_gone(CONTAINER_GONE) is True

    def test_unrelated_404_is_not_a_container_death(self):
        # e.g. a bad model name. Unbinding on this would trash a perfectly good container.
        assert is_container_gone(_api_error(404, "The model 'gpt-nope' does not exist")) is False

    def test_server_error_is_not_gone(self):
        assert is_container_gone(_api_error(500, "Container backend exploded")) is False

    def test_plain_exception_is_not_gone(self):
        assert is_container_gone(RuntimeError("boom")) is False


@pytest.mark.asyncio
class TestGetOrCreate:

    async def test_creates_and_persists_when_thread_has_none(self, temp_db):
        client, raw = _openai("cntr_a")
        cm = ContainerManager(client, db=temp_db)

        got = await cm.get_or_create("C1:111.1")

        assert got == "cntr_a"
        raw.containers.create.assert_awaited_once()
        # Persisted, so the NEXT turn can find it.
        row = temp_db.get_fresh_thread_container("C1:111.1", 15)
        assert row["container_id"] == "cntr_a"

    async def test_reuses_a_live_container(self, temp_db):
        client, raw = _openai("cntr_new")
        temp_db.save_thread_container("C1:111.1", "cntr_existing")
        cm = ContainerManager(client, db=temp_db)

        got = await cm.get_or_create("C1:111.1")

        assert got == "cntr_existing"          # continuity: same sandbox as last turn
        raw.containers.create.assert_not_awaited()
        raw.containers.retrieve.assert_awaited_once_with("cntr_existing")

    async def test_expired_container_is_replaced_not_reused(self, temp_db):
        """The whole point: never hand the API an id it will 404 on."""
        client, raw = _openai("cntr_fresh", retrieve_error=CONTAINER_GONE)
        temp_db.save_thread_container("C1:111.1", "cntr_dead")
        cm = ContainerManager(client, db=temp_db)

        got = await cm.get_or_create("C1:111.1")

        assert got == "cntr_fresh"
        assert temp_db.get_fresh_thread_container("C1:111.1", 15)["container_id"] == "cntr_fresh"

    async def test_expired_container_drops_its_published_file_record(self, temp_db):
        """A stale cfile id must not suppress a NEW artifact in the replacement container."""
        temp_db.save_thread_container("C1:111.1", "cntr_dead")
        temp_db.add_published_container_files("C1:111.1", "cntr_dead", ["cfile_old"])
        client, _ = _openai("cntr_fresh", retrieve_error=CONTAINER_GONE)
        cm = ContainerManager(client, db=temp_db)

        await cm.get_or_create("C1:111.1")

        assert await cm.get_published_files("C1:111.1", "cntr_dead") == []

    async def test_unverifiable_container_is_treated_as_dead(self, temp_db):
        """A timeout/5xx means we don't KNOW it's alive. A wasted create costs one API call;
        a wrong 'alive' costs the user's turn."""
        client, raw = _openai("cntr_fresh", retrieve_error=_api_error(503, "upstream timeout"))
        temp_db.save_thread_container("C1:111.1", "cntr_unknown")
        cm = ContainerManager(client, db=temp_db)

        assert await cm.get_or_create("C1:111.1") == "cntr_fresh"

    async def test_non_running_status_is_not_reused(self, temp_db):
        client, _ = _openai("cntr_fresh", status="expired")
        temp_db.save_thread_container("C1:111.1", "cntr_stopped")
        cm = ContainerManager(client, db=temp_db)

        assert await cm.get_or_create("C1:111.1") == "cntr_fresh"

    async def test_stale_row_outside_reuse_window_is_not_reused(self, temp_db):
        """Row exists but we last used it too long ago — the DB won't even hand it back."""
        temp_db.save_thread_container("C1:111.1", "cntr_old")
        temp_db.conn.execute(
            "UPDATE thread_containers SET last_used_at = datetime('now', '-30 minutes')")
        temp_db.conn.commit()
        client, raw = _openai("cntr_fresh")
        cm = ContainerManager(client, db=temp_db)

        assert await cm.get_or_create("C1:111.1") == "cntr_fresh"
        raw.containers.retrieve.assert_not_awaited()   # never even asked about the dead one

    async def test_create_failure_falls_back_to_auto(self, temp_db):
        """Degrade to an ephemeral container — code interpreter must still WORK."""
        client, _ = _openai(create_error=_api_error(500, "no capacity"))
        cm = ContainerManager(client, db=temp_db)

        assert await cm.get_or_create("C1:111.1") == AUTO_CONTAINER

    async def test_no_db_still_yields_a_working_container(self):
        client, raw = _openai("cntr_a")
        cm = ContainerManager(client, db=None)

        assert await cm.get_or_create("C1:111.1") == "cntr_a"
        raw.containers.create.assert_awaited_once()

    async def test_ttl_request_is_within_the_api_ceiling(self, temp_db):
        """20 minutes is the API's hard max — 60 is an HTTP 400, verified live."""
        client, raw = _openai("cntr_a")
        cm = ContainerManager(client, db=temp_db)

        await cm.get_or_create("C1:111.1")

        expires = raw.containers.create.await_args.kwargs["expires_after"]
        assert expires["anchor"] == "last_active_at"
        assert 1 <= expires["minutes"] <= 20

    async def test_container_name_is_traceable_to_the_thread(self, temp_db):
        client, raw = _openai("cntr_a")
        cm = ContainerManager(client, db=temp_db)

        await cm.get_or_create("C1:111.1")

        assert "C1:111.1" in raw.containers.create.await_args.kwargs["name"]

    async def test_distinct_threads_get_distinct_containers(self, temp_db):
        """Scope is the thread. One thread's sandbox must never be visible to another."""
        client, raw = _openai()
        raw.containers.create = AsyncMock(
            side_effect=[MagicMock(id="cntr_1"), MagicMock(id="cntr_2")])
        cm = ContainerManager(client, db=temp_db)

        assert await cm.get_or_create("C1:111.1") == "cntr_1"
        assert await cm.get_or_create("C1:222.2") == "cntr_2"


@pytest.mark.asyncio
class TestPublishedFileRecord:
    """Durable dedupe — the thing that stops a restart re-posting an earlier turn's chart."""

    async def test_roundtrip(self, temp_db):
        client, _ = _openai()
        cm = ContainerManager(client, db=temp_db)
        await cm.get_or_create("C1:111.1")

        await cm.remember_published("C1:111.1", "cntr_new", ["cfile_1", "cfile_2"])

        assert set(await cm.get_published_files("C1:111.1", "cntr_new")) == {"cfile_1", "cfile_2"}

    async def test_survives_a_new_process(self, temp_db):
        """The in-memory dedupe dies with the process; this must not."""
        client, _ = _openai()
        await ContainerManager(client, db=temp_db).get_or_create("C1:111.1")
        await ContainerManager(client, db=temp_db).remember_published("C1:111.1", "cntr_new", ["cfile_1"])

        reborn = ContainerManager(_openai()[0], db=temp_db)
        assert await reborn.get_published_files("C1:111.1", "cntr_new") == ["cfile_1"]

    async def test_no_row_means_no_record(self, temp_db):
        client, _ = _openai()
        cm = ContainerManager(client, db=temp_db)
        assert await cm.get_published_files("C1:nope", "cntr_x") == []

    async def test_db_failure_is_swallowed(self, temp_db):
        client, _ = _openai()
        cm = ContainerManager(client, db=MagicMock(
            get_thread_container_async=AsyncMock(side_effect=RuntimeError("db down"))))
        assert await cm.get_published_files("C1:111.1", "cntr_x") == []


@pytest.mark.asyncio
class TestReap:

    async def test_deletes_expired_containers_and_their_rows(self, temp_db):
        temp_db.save_thread_container("C1:111.1", "cntr_old")
        temp_db.conn.execute(
            "UPDATE thread_containers SET last_used_at = datetime('now', '-3 hours')")
        temp_db.conn.commit()
        client, raw = _openai()
        cm = ContainerManager(client, db=temp_db)

        assert await cm.reap() == 1
        raw.containers.delete.assert_awaited_once_with("cntr_old")
        assert temp_db.get_expired_thread_containers(0) == []

    async def test_leaves_active_containers_alone(self, temp_db):
        temp_db.save_thread_container("C1:111.1", "cntr_live")
        client, raw = _openai()
        cm = ContainerManager(client, db=temp_db)

        assert await cm.reap() == 0
        raw.containers.delete.assert_not_awaited()
        assert temp_db.get_fresh_thread_container("C1:111.1", 15)["container_id"] == "cntr_live"

    async def test_already_expired_container_still_drops_its_row(self, temp_db):
        """A 404 on delete is the EXPECTED case — it idle-expired hours ago."""
        temp_db.save_thread_container("C1:111.1", "cntr_gone")
        temp_db.conn.execute(
            "UPDATE thread_containers SET last_used_at = datetime('now', '-3 hours')")
        temp_db.conn.commit()
        client, raw = _openai()
        raw.containers.delete = AsyncMock(side_effect=CONTAINER_GONE)
        cm = ContainerManager(client, db=temp_db)

        assert await cm.reap() == 1
        assert temp_db.get_fresh_thread_container("C1:111.1", 999) is None

    async def test_no_db_is_a_noop(self):
        client, _ = _openai()
        assert await ContainerManager(client, db=None).reap() == 0


class TestDatabaseContainerRows:
    """The SQL contract. Every cutoff is datetime('now', …) — CURRENT_TIMESTAMP is UTC, and a
    Python local-time cutoff would judge every container hours fresher than it is."""

    def test_reuse_window_excludes_an_old_row(self, temp_db):
        temp_db.save_thread_container("C1:111.1", "cntr_a")
        temp_db.conn.execute(
            "UPDATE thread_containers SET last_used_at = datetime('now', '-16 minutes')")
        temp_db.conn.commit()

        assert temp_db.get_fresh_thread_container("C1:111.1", 15) is None
        assert temp_db.get_fresh_thread_container("C1:111.1", 20)["container_id"] == "cntr_a"

    def test_utc_not_local_time(self, temp_db):
        """Guards the exact trap called out in delete_old_tool_usage: on a UTC-4 host, a
        local-time cutoff would see a just-written row as 4 hours old and drop it."""
        temp_db.save_thread_container("C1:111.1", "cntr_a")
        assert temp_db.get_fresh_thread_container("C1:111.1", 1) is not None

    def test_touch_refreshes_the_window(self, temp_db):
        temp_db.save_thread_container("C1:111.1", "cntr_a")
        temp_db.conn.execute(
            "UPDATE thread_containers SET last_used_at = datetime('now', '-16 minutes')")
        temp_db.conn.commit()
        assert temp_db.get_fresh_thread_container("C1:111.1", 15) is None

        temp_db.touch_thread_container("C1:111.1", "cntr_a")

        assert temp_db.get_fresh_thread_container("C1:111.1", 15)["container_id"] == "cntr_a"

    def test_rebinding_clears_the_published_record(self, temp_db):
        temp_db.save_thread_container("C1:111.1", "cntr_a")
        temp_db.add_published_container_files("C1:111.1", "cntr_a", ["cfile_1"])

        temp_db.save_thread_container("C1:111.1", "cntr_b")   # new container, empty sandbox

        row = temp_db.get_fresh_thread_container("C1:111.1", 15)
        assert row["container_id"] == "cntr_b"
        assert row["published_files"] == []

    def test_published_files_merge_without_duplicates(self, temp_db):
        temp_db.save_thread_container("C1:111.1", "cntr_a")
        temp_db.add_published_container_files("C1:111.1", "cntr_a", ["cfile_1"])
        temp_db.add_published_container_files("C1:111.1", "cntr_a", ["cfile_1", "cfile_2"])

        assert temp_db.get_fresh_thread_container("C1:111.1", 15)["published_files"] == [
            "cfile_1", "cfile_2"]

    def test_published_files_are_bounded(self, temp_db):
        temp_db.save_thread_container("C1:111.1", "cntr_a")
        cap = DatabaseManager._CONTAINER_PUBLISHED_CAP
        temp_db.add_published_container_files(
            "C1:111.1", "cntr_a", [f"cfile_{i}" for i in range(cap + 50)])

        stored = temp_db.get_fresh_thread_container("C1:111.1", 15)["published_files"]
        assert len(stored) == cap
        assert stored[-1] == f"cfile_{cap + 49}"   # newest kept

    def test_published_files_for_unknown_thread_is_a_noop(self, temp_db):
        temp_db.add_published_container_files("C1:ghost", "cntr_x", ["cfile_1"])   # must not raise
        assert temp_db.get_fresh_thread_container("C1:ghost", 15) is None

    def test_corrupt_json_degrades_to_empty(self, temp_db):
        temp_db.save_thread_container("C1:111.1", "cntr_a")
        temp_db.conn.execute("UPDATE thread_containers SET published_files_json = 'not json'")
        temp_db.conn.commit()

        assert temp_db.get_fresh_thread_container("C1:111.1", 15)["published_files"] == []

    def test_delete_removes_the_binding(self, temp_db):
        temp_db.save_thread_container("C1:111.1", "cntr_a")
        temp_db.delete_thread_container("C1:111.1")
        assert temp_db.get_fresh_thread_container("C1:111.1", 15) is None

    def test_threads_are_isolated(self, temp_db):
        temp_db.save_thread_container("C1:111.1", "cntr_a")
        temp_db.save_thread_container("C1:222.2", "cntr_b")

        assert temp_db.get_fresh_thread_container("C1:111.1", 15)["container_id"] == "cntr_a"
        assert temp_db.get_fresh_thread_container("C1:222.2", 15)["container_id"] == "cntr_b"


# ------------------------------------------------------- codex round-2 regressions

@pytest.mark.asyncio
class TestPublicationLatch:
    """The thread lock is RELEASED before artifacts publish (main.py runs after
    process_message returns). Without a latch, turn A is still listing/uploading out of the
    persistent container while turn B is already writing new files into it — A posts B's
    half-finished work under A's answer, and both turns can upload the same file."""

    async def test_next_turn_waits_for_the_previous_publication(self, temp_db):
        import asyncio as aio
        from message_processor.containers import publication_lock, release_publication_lock

        temp_db.save_thread_container("C1:111.1", "cntr_a")
        client, _ = _openai("cntr_a")
        cm = ContainerManager(client, db=temp_db)

        lock = publication_lock("C1:111.1")
        await lock.acquire()                      # a publication is in flight
        resolved = aio.create_task(cm.get_or_create("C1:111.1"))
        await aio.sleep(0.05)

        assert not resolved.done(), "container resolution must WAIT for publication to finish"

        lock.release()
        release_publication_lock("C1:111.1")
        assert await aio.wait_for(resolved, timeout=2) == "cntr_a"

    async def test_uncontended_resolution_does_not_block(self, temp_db):
        temp_db.save_thread_container("C1:111.1", "cntr_a")
        client, _ = _openai("cntr_a")
        cm = ContainerManager(client, db=temp_db)
        assert await cm.get_or_create("C1:111.1") == "cntr_a"   # no latch held -> immediate

    async def test_latch_registry_does_not_leak(self):
        from message_processor import containers as cmod
        from message_processor.containers import publication_lock, release_publication_lock

        before = len(cmod._publication_locks)
        for i in range(50):
            key = f"C1:{i}"
            lock = publication_lock(key)
            async with lock:
                pass
            release_publication_lock(key)
        assert len(cmod._publication_locks) == before, "one Lock per thread would grow forever"


@pytest.mark.asyncio
class TestBaselineSnapshot:
    """A reused container's listing is CUMULATIVE. Anything already in it at turn start is by
    definition NOT this turn's output — and if it stays eligible, leftovers from turn 1 consume
    turn 2's publication cap and the chart the user just asked for is silently dropped."""

    def _with_files(self, client, raw, *file_ids, source="assistant"):
        files = [MagicMock(id=f, source=source, path=f"/mnt/data/{f}.png") for f in file_ids]

        async def _aiter():
            for f in files:
                yield f

        raw.containers.files.list = MagicMock(side_effect=lambda **k: _aiter())
        return client

    async def test_preexisting_files_are_marked_not_publishable(self, temp_db):
        temp_db.save_thread_container("C1:111.1", "cntr_a")
        client, raw = _openai("cntr_a")
        self._with_files(client, raw, "cfile_turn1_a", "cfile_turn1_b")
        cm = ContainerManager(client, db=temp_db)

        assert await cm.get_or_create("C1:111.1") == "cntr_a"

        recorded = await cm.get_published_files("C1:111.1", "cntr_a")
        assert set(recorded) == {"cfile_turn1_a", "cfile_turn1_b"}

    async def test_a_users_own_mounted_file_is_not_baselined(self, temp_db):
        """Only assistant output is ours to suppress. A user's attachment was never publishable
        in the first place (the source filter blocks it), so it does not belong in this record."""
        temp_db.save_thread_container("C1:111.1", "cntr_a")
        client, raw = _openai("cntr_a")
        self._with_files(client, raw, "cfile_user_csv", source="user")
        cm = ContainerManager(client, db=temp_db)

        await cm.get_or_create("C1:111.1")

        assert await cm.get_published_files("C1:111.1", "cntr_a") == []

    async def test_a_fresh_container_needs_no_baseline(self, temp_db):
        client, raw = _openai("cntr_new")
        cm = ContainerManager(client, db=temp_db)

        await cm.get_or_create("C1:111.1")   # no prior row -> create, nothing to baseline

        raw.containers.files.list.assert_not_called()

    async def test_baseline_failure_does_not_break_the_turn(self, temp_db):
        temp_db.save_thread_container("C1:111.1", "cntr_a")
        client, raw = _openai("cntr_a")
        raw.containers.files.list = MagicMock(side_effect=RuntimeError("listing down"))
        cm = ContainerManager(client, db=temp_db)

        assert await cm.get_or_create("C1:111.1") == "cntr_a"   # still usable


@pytest.mark.asyncio
class TestDedupeSurvivesLongTurns:
    """get_published_files must NOT be age-filtered. A single turn can outlive the reuse window
    (a tool loop with slow tools); if publication can no longer read its own dedupe list, it
    re-posts every earlier artifact still sitting in the container."""

    async def test_record_readable_after_the_reuse_window_lapses(self, temp_db):
        temp_db.save_thread_container("C1:111.1", "cntr_a")
        temp_db.add_published_container_files("C1:111.1", "cntr_a", ["cfile_turn1"])
        # The turn has now been running longer than the reuse window.
        temp_db.conn.execute(
            "UPDATE thread_containers SET last_used_at = datetime('now', '-45 minutes')")
        temp_db.conn.commit()
        client, _ = _openai()
        cm = ContainerManager(client, db=temp_db)

        assert await cm.get_published_files("C1:111.1", "cntr_a") == ["cfile_turn1"]

    async def test_record_for_a_rebound_container_is_not_returned(self, temp_db):
        """These ids describe a sandbox that no longer backs this thread."""
        temp_db.save_thread_container("C1:111.1", "cntr_a")
        temp_db.add_published_container_files("C1:111.1", "cntr_a", ["cfile_old"])
        temp_db.save_thread_container("C1:111.1", "cntr_b")   # rebound
        client, _ = _openai()
        cm = ContainerManager(client, db=temp_db)

        assert await cm.get_published_files("C1:111.1", "cntr_a") == []


class TestContainerScopedWrites:
    """Every mutation is conditional on container_id. A row can be rebound at any moment, and a
    thread_id-only write then lands on the WRONG container."""

    def test_reaper_cannot_delete_a_replacement_binding(self, temp_db):
        """Reaper selects stale X; a live turn rebinds the thread to Y; the reaper's delete must
        NOT take Y with it — that would leave the live container untracked."""
        temp_db.save_thread_container("C1:111.1", "cntr_X")
        temp_db.save_thread_container("C1:111.1", "cntr_Y")   # rebound mid-reap

        temp_db.delete_thread_container("C1:111.1", "cntr_X")

        assert temp_db.get_thread_container("C1:111.1")["container_id"] == "cntr_Y"

    def test_late_publication_cannot_poison_a_new_containers_record(self, temp_db):
        """A publication for old container X finishing after the row was rebound to Y must not
        write X's ids into Y's dedupe list — they would suppress Y's real artifacts."""
        temp_db.save_thread_container("C1:111.1", "cntr_X")
        temp_db.save_thread_container("C1:111.1", "cntr_Y")

        temp_db.add_published_container_files("C1:111.1", "cntr_X", ["cfile_from_X"])

        assert temp_db.get_thread_container("C1:111.1")["published_files"] == []

    def test_touch_is_scoped_to_its_container(self, temp_db):
        temp_db.save_thread_container("C1:111.1", "cntr_Y")
        temp_db.conn.execute(
            "UPDATE thread_containers SET last_used_at = datetime('now', '-30 minutes')")
        temp_db.conn.commit()

        temp_db.touch_thread_container("C1:111.1", "cntr_X")   # stale container, wrong id

        assert temp_db.get_fresh_thread_container("C1:111.1", 15) is None

    def test_get_thread_container_ignores_age(self, temp_db):
        temp_db.save_thread_container("C1:111.1", "cntr_a")
        temp_db.conn.execute(
            "UPDATE thread_containers SET last_used_at = datetime('now', '-99 hours')")
        temp_db.conn.commit()

        assert temp_db.get_fresh_thread_container("C1:111.1", 15) is None      # selection: no
        assert temp_db.get_thread_container("C1:111.1")["container_id"] == "cntr_a"  # dedupe: yes

class TestStreamingContainerDeath:
    """The shape that actually bit us live.

    A container that dies mid-STREAM does not arrive as NotFoundError(status_code=404). The SSE
    iterator raises a bare `openai.APIError` with NO status_code, so the original
    `status_code == 404` gate returned False, the designed recovery never fired, and the turn
    survived only by luck (the generic non-streaming fallback) — leaving an ERROR traceback and a
    Slack streaming_state_conflict behind. Every mock in the first round of tests set status_code,
    which is exactly why they all passed against a broken detector.
    """

    def test_bare_api_error_from_the_stream_is_detected(self):
        exc = Exception("Container with id 'cntr_6a5327db55d4' not found.")
        assert not hasattr(exc, "status_code")          # the real streaming shape
        assert is_container_gone(exc) is True

    def test_real_openai_api_error_class_is_detected(self):
        from openai import APIError
        exc = APIError("Container with id 'cntr_abc' not found.", request=None, body=None)
        assert getattr(exc, "status_code", None) is None
        assert is_container_gone(exc) is True

    def test_unrelated_stream_error_is_not_a_container_death(self):
        assert is_container_gone(Exception("Stream interrupted: connection reset")) is False

    def test_unrelated_not_found_is_not_a_container_death(self):
        # Must not unbind a healthy container just because something else 404'd.
        assert is_container_gone(Exception("Model with id 'gpt-nope' not found.")) is False
