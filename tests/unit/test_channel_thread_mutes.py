"""Participation redesign — Layer 0 (DB foundation).

Covers the atomic/inheriting channel_settings setters (partial writes, no authorship bump on
non-structural writes, NULL inheritance), the pref-marker upsert + memory cleanup migration, and
the drop of the retired channel_thread_mutes table (the per-thread mute mechanism was removed:
its methods and table are gone, the inert JSON muted_threads column is kept but cleared). All
against throwaway temp DBs — no live bot, no live data/slack.db.
"""
from __future__ import annotations

import asyncio
import sqlite3
import tempfile
from unittest.mock import patch

import pytest

from config import config
from database import DatabaseManager, _build_channel_settings_write
from message_processor.participation import resolve_participation_level


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


# --------------------------------------------------------------------------- atomic setter

class TestAtomicSetter:
    def test_partial_update_preserves_untouched_fields(self, temp_db):
        temp_db.set_channel_settings("C1", response_mode="auto_respond",
                                     directives="rule A", participation_level="active")
        temp_db.set_channel_settings("C1", directives="rule B")  # only directives
        row = temp_db.get_channel_settings("C1")
        assert row["response_mode"] == "auto_respond"
        assert row["participation_level"] == "active"
        assert row["directives"] == "rule B"

    def test_non_structural_write_does_not_bump_authorship(self, temp_db):
        temp_db.set_channel_settings("C1", response_mode="auto_respond", updated_by="alice")
        # Pin updated_ts to a detectable sentinel so any bump is visible.
        temp_db.conn.execute(
            "UPDATE channel_settings SET updated_ts = '2000-01-01 00:00:00' WHERE channel_id = 'C1'")
        # A snoozed_until-only write is NOT structural → authorship + timestamp must not move.
        temp_db.set_channel_settings("C1", snoozed_until="2026-07-09T20:00:00+00:00",
                                     updated_by="housekeeping")
        row = temp_db.get_channel_settings("C1")
        assert row["snoozed_until"] == "2026-07-09T20:00:00+00:00"  # write DID land
        assert row["updated_by"] == "alice"                        # not stolen
        assert row["updated_ts"] == "2000-01-01 00:00:00"          # not bumped

    def test_structural_write_bumps_authorship(self, temp_db):
        temp_db.set_channel_settings("C1", response_mode="auto_respond", updated_by="alice")
        temp_db.conn.execute(
            "UPDATE channel_settings SET updated_ts = '2000-01-01 00:00:00' WHERE channel_id = 'C1'")
        temp_db.set_channel_settings("C1", directives="new rule", updated_by="bob")
        row = temp_db.get_channel_settings("C1")
        assert row["updated_by"] == "bob"
        assert row["updated_ts"] != "2000-01-01 00:00:00"

    async def test_async_setter_matches_sync(self, temp_db):
        await temp_db.set_channel_settings_async("C1", response_mode="off", directives="quiet",
                                                 updated_by="alice")
        await temp_db.set_channel_settings_async("C1", directives="quieter")
        row = await temp_db.get_channel_settings_async("C1")
        assert row["response_mode"] == "off"
        assert row["directives"] == "quieter"
        assert row["updated_by"] == "alice"

    def test_no_fields_is_noop(self, temp_db):
        temp_db.set_channel_settings("C1")  # nothing provided
        assert temp_db.get_channel_settings("C1") is None

    def test_same_value_structural_write_preserves_attribution(self, temp_db):
        # #9: an idempotent structural write (SAME value) must NOT rewrite updated_ts/updated_by —
        # "field supplied" is not "value changed". A re-save of unchanged settings by a different
        # actor must not make that actor look like the last editor.
        temp_db.set_channel_settings("C1", participation_level="judicious", updated_by="alice")
        temp_db.conn.execute(
            "UPDATE channel_settings SET updated_ts = '2000-01-01 00:00:00' WHERE channel_id = 'C1'")
        temp_db.set_channel_settings("C1", participation_level="judicious", updated_by="bob")
        row = temp_db.get_channel_settings("C1")
        assert row["participation_level"] == "judicious"   # value intact
        assert row["updated_by"] == "alice"                # NOT stolen by the no-op write
        assert row["updated_ts"] == "2000-01-01 00:00:00"  # NOT bumped

    def test_mixed_write_with_a_real_change_still_bumps(self, temp_db):
        # A batch where one provided structural field is unchanged but another genuinely changes
        # still counts as a real edit → attribution bumps.
        temp_db.set_channel_settings("C1", participation_level="judicious", reply_in_channel=True,
                                     updated_by="alice")
        temp_db.conn.execute(
            "UPDATE channel_settings SET updated_ts = '2000-01-01 00:00:00' WHERE channel_id = 'C1'")
        temp_db.set_channel_settings("C1", participation_level="judicious", reply_in_channel=False,
                                     updated_by="bob")
        row = temp_db.get_channel_settings("C1")
        assert row["reply_in_channel"] is False
        assert row["updated_by"] == "bob"
        assert row["updated_ts"] != "2000-01-01 00:00:00"

    async def test_async_same_value_write_preserves_attribution(self, temp_db):
        # The fix lives in the shared builder, so the async setter inherits it too.
        await temp_db.set_channel_settings_async("C1", response_mode="auto_respond", updated_by="alice")
        temp_db.conn.execute(
            "UPDATE channel_settings SET updated_ts = '2000-01-01 00:00:00' WHERE channel_id = 'C1'")
        await temp_db.set_channel_settings_async("C1", response_mode="auto_respond", updated_by="bob")
        row = await temp_db.get_channel_settings_async("C1")
        assert row["updated_by"] == "alice"
        assert row["updated_ts"] == "2000-01-01 00:00:00"


# --------------------------------------------------------------------------- NULL inheritance

class TestNullInheritance:
    def test_unset_reply_in_channel_is_none_not_false(self, temp_db):
        # A partial insert that never mentions reply_in_channel must leave it NULL (inherit),
        # not materialize an explicit False that would override the global default.
        temp_db.set_channel_settings("C1", directives="x")
        row = temp_db.get_channel_settings("C1")
        assert row["reply_in_channel"] is None
        assert row["response_mode"] is None          # inherit, not 'tag_only'
        assert row["participation_level"] is None

    def test_reply_in_channel_explicit_values(self, temp_db):
        temp_db.set_channel_settings("C1", reply_in_channel=True)
        assert temp_db.get_channel_settings("C1")["reply_in_channel"] is True
        temp_db.set_channel_settings("C1", reply_in_channel=False)
        assert temp_db.get_channel_settings("C1")["reply_in_channel"] is False
        temp_db.set_channel_settings("C1", reply_in_channel=None)  # cleared → inherit
        assert temp_db.get_channel_settings("C1")["reply_in_channel"] is None

    def test_resolve_participation_level_with_nulls(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "channel_response_mode", "auto_respond", raising=False)
        temp_db.set_channel_settings("C1", directives="x")  # response_mode + level both NULL
        cs = temp_db.get_channel_settings("C1")
        # NULL participation_level + NULL response_mode → falls back to the global default mode.
        assert resolve_participation_level(cs) == "judicious"

    def test_explicit_level_still_wins_over_null_mode(self, temp_db):
        temp_db.set_channel_settings("C1", participation_level="active")  # response_mode stays NULL
        cs = temp_db.get_channel_settings("C1")
        assert cs["response_mode"] is None
        assert resolve_participation_level(cs) == "active"


# --------------------------------------------------------------------------- pure builder unit

class TestBuilderNoOp:
    def test_builder_returns_none_when_nothing_provided(self):
        assert _build_channel_settings_write("C1") is None
        assert _build_channel_settings_write("C1", updated_by="x") is None

    def test_builder_pins_inheritance_cols_null_on_partial_insert(self):
        sql, params = _build_channel_settings_write("C1", directives="x", updated_by="u")
        # response_mode + reply_in_channel are explicitly present (pinned NULL), so a fresh insert
        # cannot pick up the table's 'tag_only'/0 defaults.
        assert "response_mode" in sql and "reply_in_channel" in sql
        assert params[0] == "C1"
        assert params[1] is None and params[2] is None  # response_mode, reply_in_channel pinned NULL


# --------------------------------------------------------------------------- migration

class TestMigration:
    def _seed(self, db):
        db.conn.execute(
            "INSERT INTO channel_settings (channel_id, response_mode, reply_in_channel, "
            "participation_level, muted_threads, updated_by) VALUES (?, ?, ?, ?, ?, ?)",
            ("C1", "auto_respond", 1, "judicious", '["10.0", "20.0"]', "human"))
        db.conn.execute(
            "INSERT INTO channel_settings (channel_id, muted_threads) VALUES (?, ?)",
            ("C2", '["30.0"]'))
        # a stale participation-engine fact (delete) + a normal fact (keep)
        db.conn.execute(
            "INSERT INTO channel_memory (channel_id, scope, content, author) VALUES (?, ?, ?, ?)",
            ("C1", "channel", "old butt-out fact", "participation_engine:10.0"))
        db.conn.execute(
            "INSERT INTO channel_memory (channel_id, scope, content, author) VALUES (?, ?, ?, ?)",
            ("C1", "channel", "keep this one", "someuser"))
        db.conn.commit()

    def test_migration_cleans_memory_and_leaves_structural(self, temp_db):
        self._seed(temp_db)
        temp_db._migrate_participation_redesign()

        # severe participation-engine memory fact deleted; the control fact survives
        mem = temp_db.get_channel_memory("C1")
        authors = {m["author"] for m in mem}
        assert "participation_engine:10.0" not in authors
        assert "someuser" in authors

        # structural columns untouched by the migration
        row = temp_db.get_channel_settings("C1")
        assert row["response_mode"] == "auto_respond"
        assert row["participation_level"] == "judicious"
        assert row["reply_in_channel"] is True
        assert row["updated_by"] == "human"

    def test_migration_is_idempotent(self, temp_db):
        self._seed(temp_db)
        temp_db._migrate_participation_redesign()
        temp_db._migrate_participation_redesign()  # second run — no double effect, no crash
        mem = temp_db.get_channel_memory("C1")
        assert {m["author"] for m in mem} == {"someuser"}

    def test_migration_clears_json_muted_threads(self, temp_db):
        # The mute mechanism was removed; the inert JSON column is nulled so no stale blob lingers
        # (nothing reads it anymore, but a re-run must converge on the same empty state).
        self._seed(temp_db)
        temp_db._migrate_participation_redesign()
        for ch in ("C1", "C2"):
            row = temp_db.get_channel_settings(ch)
            assert row["muted_threads"] == []   # decoded empty (column is NULL)

    def test_fresh_db_has_no_mute_table(self, temp_db):
        # init_schema (run by the fixture) no longer CREATEs channel_thread_mutes, and the
        # separately-keyed drop step is a no-op on a DB that never had it → the table is absent.
        names = {r[0] for r in temp_db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "channel_thread_mutes" not in names

    def test_legacy_db_drops_mute_table_and_clears_json(self):
        # A live/legacy DB still carries the channel_thread_mutes table (Layer 0 created it) plus a
        # muted_threads JSON blob. Running the migrations (via init_schema) must DROP the table and
        # CLEAR the JSON — end state matches a fresh DB.
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/legacy.db"
            conn = sqlite3.connect(path)
            conn.execute(
                "CREATE TABLE channel_settings (channel_id TEXT PRIMARY KEY, "
                "response_mode TEXT DEFAULT 'tag_only', directives TEXT, "
                "reply_in_channel BOOLEAN DEFAULT 0, muted_threads TEXT, "
                "updated_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_by TEXT)")
            conn.execute(
                "CREATE TABLE channel_thread_mutes (channel_id TEXT, thread_ts TEXT, "
                "created_ts TIMESTAMP, created_by TEXT, reason TEXT)")
            conn.execute(
                "INSERT INTO channel_settings (channel_id, muted_threads) VALUES (?, ?)",
                ("C1", '["10.0", "20.0"]'))
            conn.execute(
                "INSERT INTO channel_thread_mutes (channel_id, thread_ts) VALUES (?, ?)",
                ("C1", "10.0"))
            conn.commit()
            conn.close()

            with patch("os.makedirs"):
                db = DatabaseManager("test")
                if getattr(db, "conn", None):
                    db.conn.close()
                db.db_path = path
                db.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
                db.conn.row_factory = sqlite3.Row
                db.init_schema()  # runs migrations, incl. the drop step
                names = {r[0] for r in db.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")}
                assert "channel_thread_mutes" not in names        # dropped
                row = db.get_channel_settings("C1")
                assert row["muted_threads"] == []                 # JSON cleared
                db.conn.close()

    def test_pref_marker_survives_migration_and_rerun(self, temp_db):
        # BLOCKER #1: a per-dimension preference marker is LIVE redesign state and must survive the
        # migration's memory cleanup — on the first run AND every re-run (the migration runs on
        # every init). Only the old generic participation_engine facts are purged.
        temp_db.conn.execute(
            "INSERT INTO channel_memory (channel_id, scope, content, author) VALUES (?, ?, ?, ?)",
            ("C1", "channel", "react less here", "participation_engine:pref:reactions"))
        temp_db.conn.execute(
            "INSERT INTO channel_memory (channel_id, scope, content, author) VALUES (?, ?, ?, ?)",
            ("C1", "channel", "old butt-out fact", "participation_engine:99.0"))
        temp_db.conn.commit()
        temp_db._migrate_participation_redesign()
        temp_db._migrate_participation_redesign()
        authors = {m["author"] for m in temp_db.get_channel_memory("C1")}
        assert "participation_engine:pref:reactions" in authors   # preference preserved
        assert "participation_engine:99.0" not in authors         # generic fact purged

    def test_migration_dedupes_pref_markers_and_enforces_uniqueness(self, temp_db):
        # SHOULD-FIX #8: any duplicate per-(channel,dimension) markers collapse to the newest, and
        # the partial UNIQUE index the migration creates then makes a second duplicate impossible.
        # init_schema already ran the migration (and created the index); drop it to reconstruct a
        # pre-upgrade DB that still holds duplicates, then re-run the migration to dedupe + rebuild.
        temp_db.conn.execute("DROP INDEX IF EXISTS idx_channel_memory_pref_marker")
        for content in ("first", "second", "third"):
            temp_db.conn.execute(
                "INSERT INTO channel_memory (channel_id, scope, content, author) VALUES (?, ?, ?, ?)",
                ("C1", "channel", content, "participation_engine:pref:reactions"))
        temp_db.conn.commit()
        temp_db._migrate_participation_redesign()
        rows = [m for m in temp_db.get_channel_memory("C1")
                if m["author"] == "participation_engine:pref:reactions"]
        assert len(rows) == 1 and rows[0]["content"] == "third"  # newest kept
        # the UNIQUE index now forbids a duplicate marker for the same (channel, dimension)
        with pytest.raises(sqlite3.IntegrityError):
            temp_db.conn.execute(
                "INSERT INTO channel_memory (channel_id, scope, content, author) VALUES (?, ?, ?, ?)",
                ("C1", "channel", "dup", "participation_engine:pref:reactions"))
        # a DIFFERENT dimension in the same channel is still allowed
        temp_db.conn.execute(
            "INSERT INTO channel_memory (channel_id, scope, content, author) VALUES (?, ?, ?, ?)",
            ("C1", "channel", "be brief", "participation_engine:pref:verbosity"))

    def test_migration_dedupe_keeps_freshest_updated_ts_not_highest_id(self, temp_db):
        # SHOULD-FIX 2: the dedupe keeps the row with the greatest updated_ts (id only as the
        # tie-breaker), NOT merely the highest id. The upsert refreshes a marker IN PLACE with a
        # fresh updated_ts, so a later-refreshed but LOWER-id duplicate holds the current content —
        # the old MAX(id) rule would have discarded it in favor of a stale higher-id row.
        temp_db.conn.execute("DROP INDEX IF EXISTS idx_channel_memory_pref_marker")
        temp_db.conn.execute(
            "INSERT INTO channel_memory (id, channel_id, scope, content, author, updated_ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (200, "C1", "channel", "stale-high-id", "participation_engine:pref:reactions",
             "2026-07-15 09:00:00"))
        temp_db.conn.execute(
            "INSERT INTO channel_memory (id, channel_id, scope, content, author, updated_ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (100, "C1", "channel", "fresh-low-id", "participation_engine:pref:reactions",
             "2026-07-15 12:00:00"))
        temp_db.conn.commit()
        temp_db._migrate_participation_redesign()
        rows = [m for m in temp_db.get_channel_memory("C1")
                if m["author"] == "participation_engine:pref:reactions"]
        assert len(rows) == 1
        assert rows[0]["content"] == "fresh-low-id"   # greatest updated_ts wins over greatest id
        assert rows[0]["id"] == 100

    def test_migration_dedupe_ignores_workspace_pref_rows(self, temp_db):
        # SHOULD-FIX 2: the marker is a CHANNEL-scope row. A workspace-scope row that happens to
        # share the pref author prefix must be left ALONE by the scope-filtered dedupe, while the
        # channel-scope duplicates still collapse to one.
        temp_db.conn.execute("DROP INDEX IF EXISTS idx_channel_memory_pref_marker")
        temp_db.conn.execute(
            "INSERT INTO channel_memory (channel_id, scope, content, author) VALUES (?, ?, ?, ?)",
            ("C1", "workspace", "workspace pref", "participation_engine:pref:reactions"))
        for content in ("chan-a", "chan-b"):
            temp_db.conn.execute(
                "INSERT INTO channel_memory (channel_id, scope, content, author) VALUES (?, ?, ?, ?)",
                ("C1", "channel", content, "participation_engine:pref:reactions"))
        temp_db.conn.commit()
        temp_db._migrate_participation_redesign()
        marker_rows = [m for m in temp_db.get_channel_memory("C1")
                       if m["author"] == "participation_engine:pref:reactions"]
        # the workspace row survives untouched; the channel duplicates collapse to exactly one
        assert sorted(r["scope"] for r in marker_rows) == ["channel", "workspace"]

    def test_channel_marker_index_does_not_collide_with_workspace_pref(self, temp_db):
        # SHOULD-FIX 2: the partial UNIQUE index is scope-filtered, so a workspace pref-prefixed row
        # sharing (channel_id, author) does NOT block inserting the channel marker — under the old
        # scope-agnostic predicate the two would have collided. init_schema already (re)built the
        # index with the new predicate.
        temp_db.conn.execute(
            "INSERT INTO channel_memory (channel_id, scope, content, author) VALUES (?, ?, ?, ?)",
            ("C1", "workspace", "workspace pref", "participation_engine:pref:reactions"))
        # the CHANNEL marker for the same (channel, author) inserts fine alongside the workspace row
        temp_db.conn.execute(
            "INSERT INTO channel_memory (channel_id, scope, content, author) VALUES (?, ?, ?, ?)",
            ("C1", "channel", "channel pref", "participation_engine:pref:reactions"))
        temp_db.conn.commit()
        # but a SECOND channel duplicate for that (channel, dimension) is still forbidden
        with pytest.raises(sqlite3.IntegrityError):
            temp_db.conn.execute(
                "INSERT INTO channel_memory (channel_id, scope, content, author) VALUES (?, ?, ?, ?)",
                ("C1", "channel", "dup", "participation_engine:pref:reactions"))


# --------------------------------------------------------------------------- pref marker upsert

class TestPrefMarkerUpsert:
    """SHOULD-FIX #8 — the atomic per-dimension preference upsert."""

    MARK = "participation_engine:pref:reactions"

    async def test_first_write_inserts_then_refresh_keeps_same_row(self, temp_db):
        rid = await temp_db.upsert_channel_pref_memory("C1", self.MARK, "react less")
        assert isinstance(rid, int)
        again = await temp_db.upsert_channel_pref_memory("C1", self.MARK, "react even less")
        assert again == rid                                   # SAME row refreshed, not a duplicate
        rows = [m for m in temp_db.get_channel_memory("C1") if m["author"] == self.MARK]
        assert len(rows) == 1
        assert rows[0]["content"] == "react even less"        # content updated
        assert rows[0]["author"] == self.MARK                 # author IS the marker

    async def test_cap_declines_new_marker_but_always_refreshes_existing(self, temp_db):
        # Fill the channel to cap with human facts; a NEW marker is declined (returns None) rather
        # than evicting a human's memory.
        for i in range(3):
            await temp_db.add_channel_memory_async("C1", f"human fact {i}", author="U9")
        assert await temp_db.upsert_channel_pref_memory(
            "C1", self.MARK, "react less", max_rows=3) is None
        assert [m for m in temp_db.get_channel_memory("C1") if m["author"] == self.MARK] == []
        # But an EXISTING marker is always refreshed even at/over cap (no new row, frees no slot).
        rid = await temp_db.upsert_channel_pref_memory("C1", self.MARK, "seed", max_rows=99)
        assert await temp_db.upsert_channel_pref_memory(
            "C1", self.MARK, "refreshed", max_rows=1) == rid

    async def test_concurrent_writes_converge_on_one_row(self, temp_db):
        # The BEGIN IMMEDIATE transaction serializes racing writers → exactly one marker row.
        await asyncio.gather(*[
            temp_db.upsert_channel_pref_memory("C1", self.MARK, f"v{i}") for i in range(12)])
        rows = [m for m in temp_db.get_channel_memory("C1") if m["author"] == self.MARK]
        assert len(rows) == 1

    async def test_blank_args_are_safe_noops(self, temp_db):
        assert await temp_db.upsert_channel_pref_memory("", self.MARK, "x") is None
        assert await temp_db.upsert_channel_pref_memory("C1", "", "x") is None
