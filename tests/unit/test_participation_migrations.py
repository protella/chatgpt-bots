"""The binary gate's two one-way migrations: the retired levels, and the retired preference rows.

Both exist because the rich participation gate is gone, and both are about NOT LOSING what a
person actually said:

  * levels — `judicious` and `active` described how much value was enough to speak, a question the
    binary gate does not ask. They collapse to `on`. A channel left on a name the new gate cannot
    read falls back to the global default, which for an operator who deliberately turned the
    channel ON can mean silence.
  * preference rows — "react less in here" was stored as a channel_memory row authored
    `participation_engine:pref:<dimension>` and rendered to the model as an instruction. The writer
    AND the section that rendered it are gone, so that text has to move into the reserved policy
    row or it becomes an instruction nothing obeys.

The tests below therefore care about three things above correctness of the happy path: that a
failure anywhere leaves the legacy data exactly where it was (it is the only copy), that one bad
channel cannot cost another channel its preferences, and that the merged policy text is
byte-deterministic — the rendered steering block is prompt-cached on those bytes.

All local SQLite, no live bot.
"""
from __future__ import annotations

import sqlite3
import tempfile
from unittest.mock import patch

import aiosqlite
import pytest

from database import DatabaseManager, merged_policy_text

# One marker per (channel, dimension) — the partial unique index enforces it, so a channel with
# several preferences has several dimensions, never several rows of one.
MARK_REACTIONS = "participation_engine:pref:reactions"
MARK_REPLIES = "participation_engine:pref:replies"
MARK_VERBOSITY = "participation_engine:pref:verbosity"

MIGRATION_AUTHOR = "migration:participation_prefs"


@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("os.makedirs"):
            db = DatabaseManager("test")
            db.db_path = f"{tmpdir}/test.db"
            if getattr(db, "conn", None):
                db.conn.close()
            db.conn = sqlite3.connect(db.db_path, check_same_thread=False, isolation_level=None)
            db.conn.row_factory = sqlite3.Row
            db.conn.execute("PRAGMA journal_mode=WAL")
            db.init_schema()
            yield db
            if getattr(db, "conn", None):
                db.conn.close()


def _settings(db, channel_id, level, response_mode="tag_only"):
    db.conn.execute(
        "INSERT INTO channel_settings (channel_id, participation_level, response_mode) "
        "VALUES (?, ?, ?)", (channel_id, level, response_mode))


def _level_of(db, channel_id):
    row = db.conn.execute(
        "SELECT participation_level FROM channel_settings WHERE channel_id = ?",
        (channel_id,)).fetchone()
    return row["participation_level"] if row else None


def _mode_of(db, channel_id):
    row = db.conn.execute(
        "SELECT response_mode FROM channel_settings WHERE channel_id = ?",
        (channel_id,)).fetchone()
    return row["response_mode"] if row else None


def _policy_rows(db, channel_id="C1"):
    return [dict(r) for r in db.conn.execute(
        "SELECT * FROM channel_memory WHERE scope = 'policy' AND channel_id = ?", (channel_id,))]


def _policy_text(db, channel_id="C1"):
    rows = _policy_rows(db, channel_id)
    return rows[0]["content"] if rows else None


def _pref_rows(db, channel_id=None):
    sql = ("SELECT * FROM channel_memory "
           "WHERE author LIKE 'participation_engine:pref:%' ")
    params: tuple = ()
    if channel_id:
        sql += "AND channel_id = ? "
        params = (channel_id,)
    return [dict(r) for r in db.conn.execute(sql + "ORDER BY id", params)]


# --------------------------------------------------------------------- the merge rule itself
#
# The prefs migration folds several rows into ONE policy, one after another, so by the second fold
# the "current" text is multi-line. These pin the line-level dedup that makes that safe without
# changing what the single-line callers (the directives migration, the legacy modal) already got.

class TestMergedPolicyText:
    def test_single_line_callers_are_unchanged(self):
        assert merged_policy_text("", "only deploys") == "only deploys"
        assert merged_policy_text("only deploys", "") == "only deploys"
        assert merged_policy_text("", "") is None
        assert merged_policy_text("a", "b") == "a\nb"
        assert merged_policy_text("only deploys", "only  deploys") == "only deploys"

    def test_a_line_already_present_in_a_multiline_policy_is_not_repeated(self):
        assert merged_policy_text("a\nb", "b") == "a\nb"
        assert merged_policy_text("a\nb", "  B ") == "a\nb\nB"      # dedup is not case folding
        assert merged_policy_text("a\nb", "c") == "a\nb\nc"

    def test_an_incoming_text_does_not_duplicate_itself(self):
        assert merged_policy_text("a", "b\nb\nc") == "a\nb\nc"

    def test_the_existing_policy_is_kept_verbatim(self):
        # Whatever a human last typed, including their own blank line, survives untouched.
        assert merged_policy_text("a\n\n  b", "c") == "a\n\n  b\nc"


# --------------------------------------------------------------------- levels → binary

class TestParticipationLevelMigration:
    async def test_both_retired_names_collapse_to_on(self, temp_db):
        _settings(temp_db, "C1", "judicious")
        _settings(temp_db, "C2", "active")
        assert await temp_db.migrate_participation_levels_to_binary_async() == (2, 0)
        assert _level_of(temp_db, "C1") == "on"
        assert _level_of(temp_db, "C2") == "on"

    async def test_the_surviving_levels_and_inherit_are_untouched(self, temp_db):
        # NULL is "inherit the global default" and is a real choice, not a missing value.
        _settings(temp_db, "C1", "off")
        _settings(temp_db, "C2", "mentions_only")
        _settings(temp_db, "C3", None)
        assert await temp_db.migrate_participation_levels_to_binary_async() == (0, 0)
        assert _level_of(temp_db, "C1") == "off"
        assert _level_of(temp_db, "C2") == "mentions_only"
        assert _level_of(temp_db, "C3") is None

    async def test_rerun_is_a_noop(self, temp_db):
        _settings(temp_db, "C1", "judicious")
        await temp_db.migrate_participation_levels_to_binary_async()
        assert await temp_db.migrate_participation_levels_to_binary_async() == (0, 0)
        assert _level_of(temp_db, "C1") == "on"

    async def test_response_mode_is_left_alone(self, temp_db):
        # It is dual-written, its legacy mapping already reads correctly, and it is the one column
        # a rollback to the previous release still consults.
        _settings(temp_db, "C1", "judicious", response_mode="auto_respond")
        await temp_db.migrate_participation_levels_to_binary_async()
        assert _mode_of(temp_db, "C1") == "auto_respond"

    async def test_a_hand_edited_value_still_matches(self, temp_db):
        # The modal only ever wrote these lowercase; a value typed straight into the DB is exactly
        # the row that would otherwise be left behind to fall back to the global default.
        _settings(temp_db, "C1", " Judicious ")
        assert await temp_db.migrate_participation_levels_to_binary_async() == (1, 0)
        assert _level_of(temp_db, "C1") == "on"

    async def test_one_bad_channel_does_not_block_the_others(self, temp_db, monkeypatch):
        _settings(temp_db, "C1", "judicious")
        _settings(temp_db, "C2", "active")
        real_execute = aiosqlite.Connection.execute

        def _fail_for_c2(self, sql, *a, **kw):
            # NOT async: aiosqlite's execute returns an awaitable that is ALSO an async context
            # manager, and wrapping it in a coroutine would break every `async with` here rather
            # than the one statement this test is about.
            if "UPDATE channel_settings" in sql and a and "C2" in str(a[0]):
                raise sqlite3.OperationalError("disk I/O error")
            return real_execute(self, sql, *a, **kw)

        monkeypatch.setattr(aiosqlite.Connection, "execute", _fail_for_c2)
        assert await temp_db.migrate_participation_levels_to_binary_async() == (1, 1)
        monkeypatch.undo()
        assert _level_of(temp_db, "C1") == "on"
        assert _level_of(temp_db, "C2") == "active"      # untouched, and reported as failed


# --------------------------------------------------------------------- pref rows → policy

class TestParticipationPrefMigration:
    async def test_a_lone_pref_row_becomes_the_policy(self, temp_db):
        temp_db.add_channel_memory("C1", "react less here", author=MARK_REACTIONS)
        assert await temp_db.migrate_participation_prefs_to_policy_async() == (1, 0)
        assert _policy_text(temp_db) == "react less here"
        assert _pref_rows(temp_db) == []

    async def test_the_migration_records_its_own_provenance(self, temp_db):
        temp_db.add_channel_memory("C1", "react less here", author=MARK_REACTIONS)
        await temp_db.migrate_participation_prefs_to_policy_async()
        assert _policy_rows(temp_db)[0]["author"] == MIGRATION_AUTHOR

    async def test_an_existing_policy_keeps_its_author_and_gains_a_line(self, temp_db):
        # We appended to the operator's policy; we did not write it. Authorship stays theirs.
        await temp_db.set_channel_policy_async("C1", "only jump in on deploys", author="U1")
        temp_db.add_channel_memory("C1", "react less here", author=MARK_REACTIONS)
        assert await temp_db.migrate_participation_prefs_to_policy_async() == (1, 0)
        row = _policy_rows(temp_db)[0]
        assert row["content"] == "only jump in on deploys\nreact less here"
        assert row["author"] == "U1"

    async def test_a_line_already_in_the_policy_is_not_duplicated(self, temp_db):
        await temp_db.set_channel_policy_async("C1", "react less here", author="U1")
        temp_db.add_channel_memory("C1", "react  less  here", author=MARK_REACTIONS)
        assert await temp_db.migrate_participation_prefs_to_policy_async() == (1, 0)
        assert _policy_text(temp_db) == "react less here"      # exactly, not twice
        assert _pref_rows(temp_db) == []                       # still moved: it is represented

    async def test_several_prefs_merge_in_id_order(self, temp_db):
        # Exact text, because the steering block is prompt-cached on these bytes: the same
        # database has to render the same policy on every startup.
        temp_db.add_channel_memory("C1", "react less here", author=MARK_REACTIONS)
        temp_db.add_channel_memory("C1", "keep replies short here", author=MARK_VERBOSITY)
        temp_db.add_channel_memory("C1", "do not reply in threads here", author=MARK_REPLIES)
        assert await temp_db.migrate_participation_prefs_to_policy_async() == (1, 0)
        assert _policy_text(temp_db) == ("react less here\n"
                                         "keep replies short here\n"
                                         "do not reply in threads here")

    async def test_several_prefs_append_to_an_existing_policy_in_order(self, temp_db):
        await temp_db.set_channel_policy_async("C1", "only deploys", author="U1")
        temp_db.add_channel_memory("C1", "react less here", author=MARK_REACTIONS)
        temp_db.add_channel_memory("C1", "keep replies short here", author=MARK_VERBOSITY)
        await temp_db.migrate_participation_prefs_to_policy_async()
        assert _policy_text(temp_db) == ("only deploys\n"
                                         "react less here\n"
                                         "keep replies short here")

    async def test_channels_are_independent(self, temp_db):
        temp_db.add_channel_memory("C1", "react less here", author=MARK_REACTIONS)
        temp_db.add_channel_memory("C2", "no threads here", author=MARK_REPLIES)
        assert await temp_db.migrate_participation_prefs_to_policy_async() == (2, 0)
        assert _policy_text(temp_db, "C1") == "react less here"
        assert _policy_text(temp_db, "C2") == "no threads here"

    async def test_a_blank_pref_row_is_dropped_without_a_policy_write(self, temp_db):
        # Nothing to preserve, and leaving it behind would make every rerun report work.
        temp_db.add_channel_memory("C1", "   ", author=MARK_REACTIONS)
        assert await temp_db.migrate_participation_prefs_to_policy_async() == (1, 0)
        assert _policy_rows(temp_db) == []
        assert _pref_rows(temp_db) == []

    async def test_rerun_is_a_noop(self, temp_db):
        temp_db.add_channel_memory("C1", "react less here", author=MARK_REACTIONS)
        await temp_db.migrate_participation_prefs_to_policy_async()
        assert await temp_db.migrate_participation_prefs_to_policy_async() == (0, 0)
        assert _policy_text(temp_db) == "react less here"       # not folded onto itself
        assert len(_policy_rows(temp_db)) == 1

    async def test_ordinary_facts_and_workspace_rows_are_never_touched(self, temp_db):
        temp_db.add_channel_memory("C1", "Pat owns billing", author="U1")
        temp_db.add_channel_memory("C1", "the deploy channel is #ops", scope="workspace")
        temp_db.add_channel_memory("C1", "react less here", author=MARK_REACTIONS)
        await temp_db.migrate_participation_prefs_to_policy_async()
        rest = [dict(r) for r in temp_db.conn.execute(
            "SELECT content, scope FROM channel_memory WHERE scope != 'policy' ORDER BY id")]
        assert rest == [{"content": "Pat owns billing", "scope": "channel"},
                        {"content": "the deploy channel is #ops", "scope": "workspace"}]
        assert _policy_text(temp_db) == "react less here"       # only the pref moved

    async def test_a_failure_before_the_delete_leaves_everything_intact(self, temp_db,
                                                                       monkeypatch):
        # The legacy rows are the ONLY copy of this text until the policy write commits, so an
        # interruption between copy and delete must leave both sides exactly as they were.
        await temp_db.set_channel_policy_async("C1", "only deploys", author="U1")
        temp_db.add_channel_memory("C1", "react less here", author=MARK_REACTIONS)
        real_execute = aiosqlite.Connection.execute

        def _fail_on_the_delete(self, sql, *a, **kw):
            # NOT async — see the note in test_channel_steering.py: aiosqlite's execute returns an
            # awaitable that is also an async context manager, and a coroutine wrapper would break
            # every `async with` in the migration instead of the one statement under test.
            if "DELETE FROM channel_memory" in sql:
                raise sqlite3.OperationalError("disk I/O error")
            return real_execute(self, sql, *a, **kw)

        monkeypatch.setattr(aiosqlite.Connection, "execute", _fail_on_the_delete)
        # REPORTED, not swallowed: nothing renders these rows any more, so startup aborts on it.
        assert await temp_db.migrate_participation_prefs_to_policy_async() == (0, 1)
        monkeypatch.undo()
        assert [r["content"] for r in _pref_rows(temp_db, "C1")] == ["react less here"]
        assert _policy_text(temp_db) == "only deploys"          # rolled back, not half-applied
        assert _policy_rows(temp_db)[0]["author"] == "U1"

    async def test_a_failure_on_the_policy_write_keeps_the_pref_rows(self, temp_db, monkeypatch):
        temp_db.add_channel_memory("C1", "react less here", author=MARK_REACTIONS)
        real_execute = aiosqlite.Connection.execute

        def _fail_on_the_policy_write(self, sql, *a, **kw):
            if "INSERT INTO channel_memory" in sql:
                raise sqlite3.OperationalError("disk I/O error")
            return real_execute(self, sql, *a, **kw)

        monkeypatch.setattr(aiosqlite.Connection, "execute", _fail_on_the_policy_write)
        assert await temp_db.migrate_participation_prefs_to_policy_async() == (0, 1)
        monkeypatch.undo()
        assert [r["content"] for r in _pref_rows(temp_db, "C1")] == ["react less here"]
        assert _policy_rows(temp_db) == []

    async def test_one_bad_channel_does_not_block_the_others(self, temp_db, monkeypatch):
        # Grouped per channel precisely so an unwritable channel cannot discard another's
        # preferences along with its own.
        temp_db.add_channel_memory("C1", "react less here", author=MARK_REACTIONS)
        temp_db.add_channel_memory("C2", "poison", author=MARK_REACTIONS)
        real_execute = aiosqlite.Connection.execute

        def _fail_for_c2(self, sql, *a, **kw):
            if a and a[0] and "poison" in str(a[0]):
                raise sqlite3.OperationalError("disk I/O error")
            return real_execute(self, sql, *a, **kw)

        monkeypatch.setattr(aiosqlite.Connection, "execute", _fail_for_c2)
        assert await temp_db.migrate_participation_prefs_to_policy_async() == (1, 1)
        monkeypatch.undo()
        assert _policy_text(temp_db, "C1") == "react less here"
        assert _pref_rows(temp_db, "C1") == []
        assert _policy_rows(temp_db, "C2") == []
        assert [r["content"] for r in _pref_rows(temp_db, "C2")] == ["poison"]
