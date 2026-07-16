"""Participation redesign — channel-memory textarea reconcile (DB foundation).

The settings modal renders channel-scope memory as ONE multiline textarea (one note per line).
On submit the handler passes the open-time seed (``[id, content_hash]`` for the rows the user
actually saw) plus the edited lines, and ``reconcile_channel_memory_from_textarea_async`` diffs
them in one atomic transaction: keep / delete / add, with a cap and a concurrent-edit guard.

These pin that contract directly against a throwaway SQLite file (no live bot, no data/slack.db),
plus the shared ``normalize_memory_line`` / ``memory_content_hash`` identity the modal builder,
the submit handler, and this method all route content through.
"""
from __future__ import annotations

import sqlite3
import tempfile
from unittest.mock import patch

import pytest

from database import (
    DatabaseManager,
    memory_content_hash,
    normalize_memory_line,
)

MARK = "participation_engine:pref:reactions"


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


# --------------------------------------------------------------------------- helpers
def _seed_rows(db, *contents, author="U1", channel="C1"):
    """Insert channel-scope rows, return the open-time seed [[id, hash], ...] for exactly them."""
    seed = []
    for content in contents:
        rid = db.add_channel_memory(channel, content, author=author)
        seed.append([rid, memory_content_hash(content)])
    return seed


def _channel_rows(db, channel="C1"):
    return [m for m in db.get_channel_memory(channel) if m["scope"] == "channel"]


def _contents(db, channel="C1"):
    return sorted(m["content"] for m in _channel_rows(db, channel))


# --------------------------------------------------------------------------- normalize / hash contract
class TestNormalizeAndHash:
    def test_collapses_all_whitespace_runs(self):
        assert normalize_memory_line("deploys are\nThursday   mornings\t— ping") == \
            "deploys are Thursday mornings — ping"

    def test_blank_and_none_normalize_to_empty(self):
        for blank in ("", "   ", "\n\t ", None):
            assert normalize_memory_line(blank) == ""

    def test_hash_is_normalization_invariant(self):
        # A legacy multi-line fact and its single-line textarea rendering hash identically — the
        # identity the whole keep/delete/conflict diff rests on.
        assert memory_content_hash("react   less\nhere") == memory_content_hash("react less here")

    def test_hash_is_short_hex(self):
        h = memory_content_hash("anything")
        assert len(h) == 16 and all(c in "0123456789abcdef" for c in h)


# --------------------------------------------------------------------------- keep / delete / add
class TestReconcileCore:
    async def test_keep_all_untouched(self, temp_db):
        seed = _seed_rows(temp_db, "alpha", "beta")
        res = await temp_db.reconcile_channel_memory_from_textarea_async(
            "C1", seed, ["alpha", "beta"], author="U2", max_rows=50)
        assert res == {"deleted": [], "added": [], "conflicts": 0, "over_cap": 0}
        assert _contents(temp_db) == ["alpha", "beta"]
        # a kept row is untouched — original author preserved, not re-stamped by the editor
        assert all(m["author"] == "U1" for m in _channel_rows(temp_db))

    async def test_delete_line_removed_from_textarea(self, temp_db):
        seed = _seed_rows(temp_db, "alpha", "beta")
        beta_id = seed[1][0]
        res = await temp_db.reconcile_channel_memory_from_textarea_async(
            "C1", seed, ["alpha"], author="U2", max_rows=50)
        assert res["deleted"] == [beta_id] and res["added"] == []
        assert _contents(temp_db) == ["alpha"]

    async def test_edit_line_is_delete_plus_add(self, temp_db):
        seed = _seed_rows(temp_db, "alpha")
        alpha_id = seed[0][0]
        res = await temp_db.reconcile_channel_memory_from_textarea_async(
            "C1", seed, ["alpha edited"], author="U2", max_rows=50)
        assert res["deleted"] == [alpha_id]
        assert res["added"] == ["alpha edited"]
        rows = _channel_rows(temp_db)
        assert len(rows) == 1 and rows[0]["content"] == "alpha edited"
        assert rows[0]["author"] == "U2"           # the added line is authored by the editor

    async def test_blank_all_deletes_everything(self, temp_db):
        seed = _seed_rows(temp_db, "alpha", "beta")
        res = await temp_db.reconcile_channel_memory_from_textarea_async(
            "C1", seed, [], author="U2", max_rows=50)
        assert sorted(res["deleted"]) == sorted(sid for sid, _ in seed)
        assert res["added"] == []
        assert _channel_rows(temp_db) == []

    async def test_dedup_lines_add_once(self, temp_db):
        seed = _seed_rows(temp_db, "alpha")
        res = await temp_db.reconcile_channel_memory_from_textarea_async(
            "C1", seed, ["alpha", "new note", "new note"], author="U2", max_rows=50)
        assert res["added"] == ["new note"]        # duplicate collapsed
        assert res["deleted"] == []
        assert _contents(temp_db) == ["alpha", "new note"]


# --------------------------------------------------------------------------- unseeded rows / cap
class TestReconcileScopeAndCap:
    async def test_unseeded_row_never_deleted_or_duplicated(self, temp_db):
        # A row that exists but was NOT seeded (e.g. beyond the textarea budget, "+N more"): the
        # reconcile only ever deletes SEEDED rows, so it survives; and a typed line equal to it is
        # deduped against surviving content rather than duplicated.
        seed = _seed_rows(temp_db, "alpha")
        temp_db.add_channel_memory("C1", "hidden fact", author="U9")  # unseeded
        res = await temp_db.reconcile_channel_memory_from_textarea_async(
            "C1", seed, ["alpha", "hidden fact"], author="U2", max_rows=50)
        assert res["deleted"] == [] and res["added"] == []
        assert _contents(temp_db) == ["alpha", "hidden fact"]

    async def test_cap_counts_all_remaining_rows_not_just_adds(self, temp_db):
        # Two UNSEEDED rows already fill 2 of a 3-row cap. Only ONE new line fits; the rest overflow
        # and are reported in over_cap. Proves the cap counts ALL surviving channel rows.
        temp_db.add_channel_memory("C1", "u1", author="U9")
        temp_db.add_channel_memory("C1", "u2", author="U9")
        res = await temp_db.reconcile_channel_memory_from_textarea_async(
            "C1", [], ["n1", "n2", "n3"], author="U2", max_rows=3)
        assert res["added"] == ["n1"]
        assert res["over_cap"] == 2
        assert sorted(_contents(temp_db)) == ["n1", "u1", "u2"]

    async def test_deletes_free_slots_for_adds_under_cap(self, temp_db):
        # Deleting a seeded row frees a slot: at cap=2 with 2 seeded rows, dropping one and adding
        # one lands the add (current − deletes + adds ≤ cap), no over_cap.
        seed = _seed_rows(temp_db, "old1", "old2")
        res = await temp_db.reconcile_channel_memory_from_textarea_async(
            "C1", seed, ["old1", "fresh"], author="U2", max_rows=2)
        assert res["added"] == ["fresh"]
        assert res["over_cap"] == 0
        assert _contents(temp_db) == ["fresh", "old1"]


# --------------------------------------------------------------------------- concurrent-edit guard
class TestReconcileConflicts:
    async def test_conflict_when_row_changed_since_open(self, temp_db):
        # Seed captured hash("alpha"); a concurrent editor changed the row before submit. Dropping
        # it from the textarea must NOT clobber the concurrent edit → counted as a conflict, skipped.
        seed = _seed_rows(temp_db, "alpha")
        alpha_id = seed[0][0]
        temp_db.conn.execute(
            "UPDATE channel_memory SET content = ? WHERE id = ?", ("alpha CHANGED", alpha_id))
        temp_db.conn.commit()
        res = await temp_db.reconcile_channel_memory_from_textarea_async(
            "C1", seed, [], author="U2", max_rows=50)
        assert res["conflicts"] == 1
        assert res["deleted"] == []
        rows = _channel_rows(temp_db)
        assert len(rows) == 1 and rows[0]["content"] == "alpha CHANGED"  # not clobbered

    async def test_already_deleted_seed_is_silent_no_conflict(self, temp_db):
        # The seeded row was removed elsewhere before submit. Dropping it from the textarea is a
        # no-op — never a conflict (conflicts counts only a STILL-PRESENT row that changed) and
        # never a resurrection.
        seed = _seed_rows(temp_db, "alpha")
        alpha_id = seed[0][0]
        temp_db.conn.execute("DELETE FROM channel_memory WHERE id = ?", (alpha_id,))
        temp_db.conn.commit()
        res = await temp_db.reconcile_channel_memory_from_textarea_async(
            "C1", seed, [], author="U2", max_rows=50)
        assert res["conflicts"] == 0
        assert res["deleted"] == []
        assert _channel_rows(temp_db) == []


# --------------------------------------------------------------------------- legacy multiline + pref markers
class TestReconcileLegacyAndMarkers:
    async def test_multiline_legacy_fact_collapses_and_roundtrips_lossless_when_kept(self, temp_db):
        # A legacy fact stored with embedded newlines renders as ONE normalized textarea line; its
        # seed hash is computed on the normalized text (what the user sees). Saving untouched must
        # KEEP the row — leaving the original multi-line content byte-for-byte (lossless), never
        # rewriting it to the collapsed form.
        original = "deploys are\nThursday   mornings"
        rid = temp_db.add_channel_memory("C1", original, author="U9")
        displayed = normalize_memory_line(original)
        assert "\n" not in displayed                       # collapses to one displayed line
        seed = [[rid, memory_content_hash(original)]]      # modal seeds on the normalized hash
        res = await temp_db.reconcile_channel_memory_from_textarea_async(
            "C1", seed, [displayed], author="U2", max_rows=50)
        assert res == {"deleted": [], "added": [], "conflicts": 0, "over_cap": 0}
        rows = _channel_rows(temp_db)
        assert len(rows) == 1
        assert rows[0]["content"] == original              # lossless — kept, not collapsed
        assert rows[0]["author"] == "U9"

    async def test_pref_marker_kept_preserves_author(self, temp_db):
        # A per-dimension preference marker shown in the box and saved UNCHANGED keeps its marker
        # author — the engine still owns it (it is not demoted to a human fact by a no-op save).
        rid = temp_db.add_channel_memory("C1", "react less here", author=MARK)
        seed = [[rid, memory_content_hash("react less here")]]
        res = await temp_db.reconcile_channel_memory_from_textarea_async(
            "C1", seed, ["react less here"], author="U2", max_rows=50)
        assert res == {"deleted": [], "added": [], "conflicts": 0, "over_cap": 0}
        rows = _channel_rows(temp_db)
        assert len(rows) == 1 and rows[0]["author"] == MARK

    async def test_pref_marker_edited_demotes_author(self, temp_db):
        # Documented behavior: EDITING a marker line in the textarea is a delete+add — the engine's
        # marker row is deleted and the new text is re-added as a human-authored channel fact. The
        # preference loses its marker author (demoted); the engine can re-establish it later.
        rid = temp_db.add_channel_memory("C1", "react less here", author=MARK)
        seed = [[rid, memory_content_hash("react less here")]]
        res = await temp_db.reconcile_channel_memory_from_textarea_async(
            "C1", seed, ["react a lot less here"], author="U2", max_rows=50)
        assert res["deleted"] == [rid]
        assert res["added"] == ["react a lot less here"]
        rows = _channel_rows(temp_db)
        assert len(rows) == 1
        assert rows[0]["content"] == "react a lot less here"
        assert rows[0]["author"] == "U2"                   # demoted from marker to human author
        assert not rows[0]["author"].startswith("participation_engine:pref:")


# --------------------------------------------------------------------------- degenerate inputs
class TestReconcileDegenerate:
    async def test_empty_channel_id_is_safe_noop(self, temp_db):
        res = await temp_db.reconcile_channel_memory_from_textarea_async(
            "", [], ["x"], author="U2", max_rows=50)
        assert res == {"deleted": [], "added": [], "conflicts": 0, "over_cap": 0}

    async def test_add_into_empty_channel(self, temp_db):
        # The section doubles as an add surface: with no prior rows and an empty seed, typed lines
        # are inserted fresh.
        res = await temp_db.reconcile_channel_memory_from_textarea_async(
            "C1", [], ["first note", "second note"], author="U2", max_rows=50)
        assert res["added"] == ["first note", "second note"]
        assert _contents(temp_db) == ["first note", "second note"]
