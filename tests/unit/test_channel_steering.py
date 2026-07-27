"""Shared memory steering — ONE snapshot per turn, read once, obeyed by both halves.

The invariant under test: the participation gate and the responder see the SAME BYTES of a
channel's steering. Everything else here exists to protect that, or to protect the reserved
policy row the steering is built around:

  * lifecycle — who reads, when, how many times, and what a redispatch/retry/failure does;
  * rendering — deterministic bytes, policy always first, ids exposed only where a tool may act;
  * storage — the reserved row's uniqueness, replace-or-delete semantics, authorship;
  * authorization — the only two writers, and every generic path that must NOT reach it;
  * migration — the legacy directives column, moved once and never read again.

All stubbed I/O — no live bot.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import aiosqlite
import pytest

from base_client import Message
from config import config
from database import DatabaseManager, memory_content_hash
from settings_modal import SettingsModal
from message_processor import channel_steering
from message_processor.channel_steering import (
    CHANNEL_FACT_HEADING, EMPTY_SNAPSHOT, POLICY_HEADING, POLICY_MAX_CHARS,
    WORKSPACE_FACT_HEADING, ChannelSteeringSnapshot, is_ordinary_fact, is_policy_row, is_pref_row,
    load_snapshot, render_snapshot, stamp, stamped,
)

MARK = "participation_engine:pref:reactions"


def _fact(mid, content, scope="channel", author="U1"):
    return {"id": mid, "content": content, "scope": scope, "author": author}


def _steering_db(rows=None, policy=None, memory_error=None, policy_error=None):
    """A db stub whose ONLY steering surface is the two reads load_snapshot performs, so a test
    can count them."""
    db = MagicMock()
    db.get_channel_memory_async = AsyncMock(
        return_value=list(rows or []), side_effect=memory_error)
    db.get_channel_policy_async = AsyncMock(return_value=policy, side_effect=policy_error)
    return db


# --------------------------------------------------------------------------- row classification

class TestRowClassification:
    def test_policy_row_identified_by_scope(self):
        assert is_policy_row({"scope": "policy"})
        assert not is_policy_row({"scope": "channel"})
        assert not is_policy_row(None)

    def test_pref_row_identified_by_author_marker(self):
        assert is_pref_row({"author": MARK})
        assert is_pref_row({"author": MARK + ":extra"})
        assert not is_pref_row({"author": "U1"})
        assert not is_pref_row({"author": None})

    def test_ordinary_fact_is_neither(self):
        assert is_ordinary_fact(_fact(1, "x"))
        assert not is_ordinary_fact({"scope": "policy", "content": "p"})
        assert not is_ordinary_fact({"scope": "channel", "author": MARK})


# --------------------------------------------------------------------------- rendering

class TestRendering:
    def test_policy_renders_first_even_with_the_highest_id(self):
        # Order is FIXED, not chronological: an incidental fact written after the operator set
        # the policy must not outrank it just because it is newer or has a bigger id.
        text = render_snapshot({"id": 999, "content": "only deploys"},
                               [_fact(1, "Pat owns billing")]).text
        assert text.index(POLICY_HEADING) < text.index(CHANNEL_FACT_HEADING)
        assert text.startswith(POLICY_HEADING)

    def test_policy_id_is_never_rendered(self):
        # No model may address the policy row, so it never learns the row exists as a row.
        text = render_snapshot({"id": 42, "content": "only deploys"}, []).text
        assert "[#42]" not in text and "#42" not in text

    def test_fact_ordering_is_by_id_not_input_order(self):
        rows = [_fact(3, "third"), _fact(1, "first"), _fact(2, "second")]
        text = render_snapshot(None, rows).text
        assert "- [#1] first\n- [#2] second\n- [#3] third" in text

    def test_identical_state_renders_identical_bytes(self):
        rows = [_fact(2, "b"), _fact(1, "a")]
        first = render_snapshot({"content": "p"}, rows).text
        second = render_snapshot({"content": "p"}, list(reversed(rows))).text
        assert first == second     # determinism is what makes prompt caching work at all

    def test_instructions_and_background_are_separately_labelled(self):
        text = render_snapshot(
            {"content": "only deploys"},
            [_fact(2, "Pat owns billing"),
             _fact(3, "the company ships on Thursdays", scope="workspace")]).text
        # Each heading states its KIND — the one thing a model cannot recover from the content.
        assert "instructions" in POLICY_HEADING
        assert "background" in CHANNEL_FACT_HEADING and "background" in WORKSPACE_FACT_HEADING
        assert (text.index(POLICY_HEADING) < text.index(CHANNEL_FACT_HEADING)
                < text.index(WORKSPACE_FACT_HEADING))

    def test_a_stray_preference_row_is_inert_rather_than_rendered(self):
        # Nothing writes these any more and the startup migration deletes every survivor (a
        # process that fails that migration refuses to start), so one cannot exist during a live
        # run. If one somehow did, it stays OUT of the prompt rather than arriving as an ordinary
        # fact with an editable [#id] pointing at an operator instruction.
        text = render_snapshot(None, [_fact(9, "a fact"), _fact(4, "react less", author=MARK)]).text
        assert "react less" not in text
        assert "[#9] a fact" in text

    def test_empty_sections_are_omitted_entirely(self):
        text = render_snapshot({"content": "only deploys"}, []).text
        assert CHANNEL_FACT_HEADING not in text and WORKSPACE_FACT_HEADING not in text

    def test_blank_rows_and_blank_policy_are_dropped(self):
        snap = render_snapshot({"content": "   "}, [_fact(1, ""), _fact(2, "  ")])
        assert snap.text is None and snap.is_empty and not snap.policy_present

    def test_policy_hash_tracks_the_policy_text(self):
        a = render_snapshot({"content": "only deploys"}, [])
        b = render_snapshot({"content": "only deploys"}, [_fact(1, "unrelated")])
        c = render_snapshot({"content": "everything"}, [])
        assert a.policy_present and a.policy_hash == b.policy_hash != c.policy_hash

    def test_a_policy_row_in_the_fact_list_is_never_rendered_as_a_fact(self):
        # Defence in depth: the getter excludes it, and so does the renderer.
        text = render_snapshot(None, [{"id": 5, "scope": "policy", "content": "sneaky"}]).text
        assert text is None


# --------------------------------------------------------------------------- loading

class TestLoadSnapshot:
    async def test_one_read_each_of_policy_and_memory(self):
        db = _steering_db([_fact(1, "a fact")], policy={"content": "only deploys"})
        snap = await load_snapshot(db, "C1")
        db.get_channel_policy_async.assert_awaited_once()
        db.get_channel_memory_async.assert_awaited_once()
        assert POLICY_HEADING in snap.text and "a fact" in snap.text

    async def test_memory_disabled_still_renders_the_policy(self):
        # ENABLE_CHANNEL_MEMORY=false means "stop remembering things", never "stop obeying the
        # rules I set" — if it silenced the policy, the directives migration would quietly turn
        # off every live operator rule in the workspace.
        db = _steering_db([_fact(1, "a fact")], policy={"content": "only deploys"})
        snap = await load_snapshot(db, "C1", memory_enabled=False)
        assert "only deploys" in snap.text
        assert "a fact" not in snap.text

    async def test_a_failed_read_yields_the_ready_but_empty_snapshot(self):
        db = _steering_db(memory_error=RuntimeError("db gone"),
                          policy_error=RuntimeError("db gone"))
        snap = await load_snapshot(db, "C1")
        assert snap.text is None and snap.is_empty     # never raises, never partial

    async def test_no_db_or_no_channel_is_empty(self):
        assert (await load_snapshot(None, "C1")) is EMPTY_SNAPSHOT
        assert (await load_snapshot(_steering_db(), None)) is EMPTY_SNAPSHOT


class TestStamp:
    def test_stamp_then_read_back(self):
        msg = SimpleNamespace(metadata={})
        snap = ChannelSteeringSnapshot(text="x")
        assert stamp(msg, snap) is snap
        assert stamped(msg) is snap

    def test_unstamped_message_reads_none(self):
        assert stamped(SimpleNamespace(metadata={})) is None
        assert stamped(SimpleNamespace(metadata=None)) is None

    def test_a_second_stamp_overwrites(self):
        # A new gate attempt judges a new moment; it must not inherit the previous attempt's view.
        msg = SimpleNamespace(metadata={})
        stamp(msg, ChannelSteeringSnapshot(text="old"))
        stamp(msg, ChannelSteeringSnapshot(text="new"))
        assert stamped(msg).text == "new"

    def test_the_snapshot_is_frozen(self):
        with pytest.raises(Exception):
            ChannelSteeringSnapshot(text="x").text = "y"   # type: ignore[misc]

    def test_a_foreign_value_on_the_key_is_ignored(self):
        msg = SimpleNamespace(metadata={channel_steering.STEERING_KEY: "not a snapshot"})
        assert stamped(msg) is None


# --------------------------------------------------------------------------- the reserved row

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


def _policy_rows(db, channel_id="C1"):
    return [dict(r) for r in db.conn.execute(
        "SELECT * FROM channel_memory WHERE scope = 'policy' AND channel_id = ?", (channel_id,))]


class TestPolicyRowStorage:
    async def test_set_get_replace_updates_one_row(self, temp_db):
        await temp_db.set_channel_policy_async("C1", "only deploys", author="U1")
        await temp_db.set_channel_policy_async("C1", "everything", author="U2")
        rows = _policy_rows(temp_db)
        assert len(rows) == 1                       # REPLACE, never append or accumulate
        assert rows[0]["content"] == "everything"
        assert rows[0]["author"] == "U2"            # authorship follows the writer
        got = await temp_db.get_channel_policy_async("C1")
        assert got["content"] == "everything"

    async def test_blank_deletes_the_row(self, temp_db):
        await temp_db.set_channel_policy_async("C1", "only deploys", author="U1")
        await temp_db.set_channel_policy_async("C1", "   ", author="U1")
        assert _policy_rows(temp_db) == []          # an empty row would render an empty heading
        assert await temp_db.get_channel_policy_async("C1") is None

    async def test_delete_of_a_missing_policy_is_a_noop(self, temp_db):
        await temp_db.set_channel_policy_async("C1", None)
        assert await temp_db.get_channel_policy_async("C1") is None

    async def test_policy_is_per_channel(self, temp_db):
        await temp_db.set_channel_policy_async("C1", "only deploys")
        await temp_db.set_channel_policy_async("C2", "anything goes")
        assert (await temp_db.get_channel_policy_async("C1"))["content"] == "only deploys"
        assert (await temp_db.get_channel_policy_async("C2"))["content"] == "anything goes"

    async def test_unique_index_holds_under_concurrent_upserts(self, temp_db):
        # "Reserved" has to be true in the STORAGE, not only in the code that writes it.
        await asyncio.gather(*[
            temp_db.set_channel_policy_async("C1", f"policy {i}", author=f"U{i}")
            for i in range(8)])
        assert len(_policy_rows(temp_db)) == 1

    async def test_the_generic_fact_api_cannot_see_the_policy(self, temp_db):
        await temp_db.set_channel_policy_async("C1", "only deploys")
        temp_db.add_channel_memory("C1", "a real fact")
        rows = await temp_db.get_channel_memory_async("C1")
        assert [r["content"] for r in rows] == ["a real fact"]
        assert temp_db.get_channel_policy("C1")["content"] == "only deploys"   # sync twin agrees

    async def test_policy_does_not_consume_fact_capacity(self, temp_db):
        # MEMORY_MAX_ROWS protects the prompt from a pile of remembered FACTS. The policy is not
        # a fact, and a channel whose steering filled the cap could remember nothing at all.
        await temp_db.set_channel_policy_async("C1", "only deploys")
        temp_db.add_channel_memory("C1", "react less", author=MARK)
        res = await temp_db.reconcile_channel_memory_from_textarea_async(
            "C1", [], ["fact one", "fact two"], author="U2", max_rows=2)
        assert res["over_cap"] == 0 and len(res["added"]) == 2

    async def test_pref_marker_upsert_ignores_steering_rows_in_its_cap(self, temp_db):
        await temp_db.set_channel_policy_async("C1", "only deploys")
        temp_db.add_channel_memory("C1", "verbosity: shorter", author=MARK + "2")
        temp_db.add_channel_memory("C1", "the one real fact")
        # cap=1 and one ordinary fact exists → a NEW marker is declined, and the policy plus the
        # other marker are not what pushed it over.
        assert await temp_db.upsert_channel_pref_memory("C1", MARK, "react less", max_rows=2) \
            is not None
        assert await temp_db.upsert_channel_pref_memory(
            "C1", MARK + "3", "shorter threads", max_rows=1) is None

    async def test_settings_and_policy_land_in_one_transaction(self, temp_db):
        await temp_db.set_channel_settings_and_policy_async(
            "C1", "only deploys", author="U1",
            participation_level="mentions_only", reply_in_channel=False)
        assert (await temp_db.get_channel_policy_async("C1"))["content"] == "only deploys"
        row = await temp_db.get_channel_settings_async("C1")
        assert row["participation_level"] == "mentions_only"
        assert row["reply_in_channel"] is False
        assert row["updated_by"] == "U1"

    async def test_policy_only_write_touches_no_settings_row(self, temp_db):
        await temp_db.set_channel_settings_and_policy_async("C1", "only deploys", author="U1")
        assert (await temp_db.get_channel_policy_async("C1"))["content"] == "only deploys"
        assert await temp_db.get_channel_settings_async("C1") is None


# --------------------------------------------------------------------------- migration

def _legacy_directives(db, channel_id, text):
    db.conn.execute(
        "INSERT INTO channel_settings (channel_id, directives) VALUES (?, ?)", (channel_id, text))


def _directives_of(db, channel_id):
    row = db.conn.execute(
        "SELECT directives FROM channel_settings WHERE channel_id = ?", (channel_id,)).fetchone()
    return row["directives"] if row else None


class TestDirectivesMigration:
    async def test_directives_move_once_and_the_column_is_nulled(self, temp_db):
        _legacy_directives(temp_db, "C1", "only jump in on deploy failures")
        assert await temp_db.migrate_channel_directives_to_policy_async() == (1, 0)
        assert (await temp_db.get_channel_policy_async("C1"))["content"] \
            == "only jump in on deploy failures"
        assert _directives_of(temp_db, "C1") is None

    async def test_rerun_is_a_noop(self, temp_db):
        _legacy_directives(temp_db, "C1", "only deploys")
        await temp_db.migrate_channel_directives_to_policy_async()
        assert await temp_db.migrate_channel_directives_to_policy_async() == (0, 0)
        assert len(_policy_rows(temp_db)) == 1

    async def test_migration_records_its_own_provenance(self, temp_db):
        _legacy_directives(temp_db, "C1", "only deploys")
        await temp_db.migrate_channel_directives_to_policy_async()
        assert _policy_rows(temp_db)[0]["author"] == "migration:channel_directives"

    async def test_a_channel_with_both_keeps_both_texts(self, temp_db):
        # Guessing which one the operator meant is not the migration's call, and silently
        # dropping either is how a live rule disappears.
        await temp_db.set_channel_policy_async("C1", "answer in threads")
        _legacy_directives(temp_db, "C1", "only deploys")
        await temp_db.migrate_channel_directives_to_policy_async()
        content = (await temp_db.get_channel_policy_async("C1"))["content"]
        assert "answer in threads" in content and "only deploys" in content
        assert len(_policy_rows(temp_db)) == 1

    async def test_an_identical_duplicate_is_not_doubled(self, temp_db):
        await temp_db.set_channel_policy_async("C1", "only deploys")
        _legacy_directives(temp_db, "C1", "only  deploys")     # same rule after normalization
        await temp_db.migrate_channel_directives_to_policy_async()
        assert (await temp_db.get_channel_policy_async("C1"))["content"] == "only deploys"

    async def test_blank_directives_are_left_alone(self, temp_db):
        _legacy_directives(temp_db, "C1", "   ")
        assert await temp_db.migrate_channel_directives_to_policy_async() == (0, 0)
        assert _policy_rows(temp_db) == []

    async def test_a_failure_before_commit_leaves_the_directives_intact(self, temp_db, monkeypatch):
        # The column is the fallback until it isn't: the policy row is written FIRST and the
        # column nulled only after, so an interruption anywhere leaves the rules where they were.
        _legacy_directives(temp_db, "C1", "only deploys")
        real_execute = aiosqlite.Connection.execute

        def _fail_on_the_delete(self, sql, *a, **kw):
            # NOT async: aiosqlite's execute returns an awaitable that is ALSO an async context
            # manager, and wrapping it in a coroutine would break every `async with` in the
            # migration rather than the one statement this test is about.
            if "SET directives = NULL" in sql:
                raise sqlite3.OperationalError("disk I/O error")
            return real_execute(self, sql, *a, **kw)

        monkeypatch.setattr(aiosqlite.Connection, "execute", _fail_on_the_delete)
        # The failure is REPORTED, not swallowed: nothing reads the legacy column any more, so a
        # channel left behind has an operator rule that will not be obeyed. Startup aborts on it.
        assert await temp_db.migrate_channel_directives_to_policy_async() == (0, 1)
        monkeypatch.undo()
        assert _directives_of(temp_db, "C1") == "only deploys"
        assert _policy_rows(temp_db) == []              # rolled back, not half-applied

    async def test_one_bad_channel_does_not_block_the_others(self, temp_db, monkeypatch):
        _legacy_directives(temp_db, "C1", "only deploys")
        _legacy_directives(temp_db, "C2", "poison")
        real_execute = aiosqlite.Connection.execute

        def _fail_for_c2(self, sql, *a, **kw):
            if a and a[0] and "poison" in str(a[0]):
                raise sqlite3.OperationalError("disk I/O error")
            return real_execute(self, sql, *a, **kw)

        monkeypatch.setattr(aiosqlite.Connection, "execute", _fail_for_c2)
        assert await temp_db.migrate_channel_directives_to_policy_async() == (1, 1)
        monkeypatch.undo()
        assert _directives_of(temp_db, "C1") is None
        assert _directives_of(temp_db, "C2") == "poison"

    async def test_every_pref_row_survives_and_still_renders_before_facts(self, temp_db):
        # The backoff writer lives until the binary-gate commit; its rows are not this
        # migration's business, and must come out the other side rendering as instructions.
        temp_db.add_channel_memory("C1", "react less here", author=MARK)
        temp_db.add_channel_memory("C1", "Pat owns billing")
        _legacy_directives(temp_db, "C1", "only deploys")
        await temp_db.migrate_channel_directives_to_policy_async()
        snap = await load_snapshot(temp_db, "C1")
        assert snap.text.index(POLICY_HEADING) < snap.text.index(CHANNEL_FACT_HEADING)
        assert "Pat owns billing" in snap.text
        # The row itself survives THIS migration untouched — it is the participation-preference
        # migration's job to move it, and that one runs at startup too.
        assert any(r["author"] == MARK for r in temp_db.get_channel_memory("C1"))


# --------------------------------------------------------------------------- the length bound

def test_the_policy_bound_is_the_modals_own_bound():
    """No new number: the tool and the modal share the bound the ground-rules box always had."""
    modal = SettingsModal.__new__(SettingsModal)
    view = modal.build_channel_settings_modal("C1", None, "tag_only")
    block = next(b for b in view["blocks"] if b.get("block_id") == "policy_block")
    assert block["element"]["max_length"] == POLICY_MAX_CHARS


# --------------------------------------------------------------------------- the gate's read
#
# From here on the tests are about WHO reads, WHEN, and how many times — the lifecycle that makes
# the same-bytes invariant true rather than merely intended.

@pytest.fixture(autouse=True)
def _no_canvas_catalog(monkeypatch):
    """The gate asks Slack for the channel's canvases. Every test here drives it with a mock
    client, whose canvas titles would be mock objects — and the real classifier prompt joins
    those titles into a string. Empty is the honest answer for a channel nobody stubbed."""
    from message_processor import canvas_tools
    monkeypatch.setattr(canvas_tools, "build_catalog", AsyncMock(return_value=[]))


def _gate_app(db, evaluate):
    """main.ChatBotV2 wired down to just the gate path (the harness test_custom_emoji uses)."""
    from main import ChatBotV2

    app = ChatBotV2.__new__(ChatBotV2)
    app.processor = MagicMock()
    app.processor.db = db
    app.processor.channel_summary_service = None
    app.processor.mcp_manager = None
    app.participation_engine = MagicMock()
    app.participation_engine.evaluate = evaluate
    app.participation_engine.note_arrival = MagicMock()

    client = MagicMock()
    client.channel_pulse = None
    client.get_channel_context = AsyncMock(return_value={})
    client.bot_handle = "ChatGPT"
    return app, client


def _gate_msg(**meta):
    md = {"ts": "10.0", "gate_required": True, "participation_level": "judicious"}
    md.update(meta)
    return Message(text="deploy failed", user_id="U1", channel_id="C1", thread_id="10.0",
                   metadata=md)


def _sleeping_engine(captured):
    """An engine that records what it was handed and decides not to wake."""
    from message_processor.participation import GateEvaluation, WakeDecision

    async def _eval(**kw):
        captured.update(kw)
        return GateEvaluation(decision=WakeDecision(wake=False))
    return _eval


class TestGateReads:
    async def test_the_gate_reads_once_stamps_and_hands_the_engine_the_text(self):
        db = _steering_db([_fact(1, "Pat owns billing")], policy={"content": "only deploys"})
        captured = {}
        app, client = _gate_app(db, _sleeping_engine(captured))
        msg = _gate_msg()

        assert await app._gate_verdict(msg, client) is None      # wake=false → stays silent
        db.get_channel_memory_async.assert_awaited_once()
        db.get_channel_policy_async.assert_awaited_once()

        snap = stamped(msg)
        assert snap is not None and "only deploys" in snap.text
        # The engine gets the rendered block and nothing else — not the raw rows, not a separate
        # directives string, which is what let the gate render its own version of the channel.
        assert captured["channel_steering_text"] == snap.text
        assert "directives" not in captured and "memory_facts" not in captured

    async def test_a_second_gate_attempt_reads_afresh(self):
        # A queued redispatch (or an edit re-evaluation) is a NEW attempt judging a NEW moment:
        # it must not inherit the first attempt's view of the channel.
        db = _steering_db([], policy={"content": "only deploys"})
        captured = {}
        app, client = _gate_app(db, _sleeping_engine(captured))
        msg = _gate_msg()

        await app._gate_verdict(msg, client)
        first = stamped(msg).text
        db.get_channel_policy_async.return_value = {"content": "anything goes"}
        await app._gate_verdict(msg, client)
        second = stamped(msg).text

        assert "only deploys" in first and "anything goes" in second
        assert captured["channel_steering_text"] == second
        assert db.get_channel_memory_async.await_count == 2

    async def test_a_failed_read_stamps_the_empty_snapshot(self):
        db = _steering_db(memory_error=RuntimeError("db gone"),
                          policy_error=RuntimeError("db gone"))
        captured = {}
        app, client = _gate_app(db, _sleeping_engine(captured))
        msg = _gate_msg()

        await app._gate_verdict(msg, client)
        assert stamped(msg) is not None                  # stamped, so the responder won't retry
        assert stamped(msg).is_empty
        assert captured["channel_steering_text"] is None


# --------------------------------------------------------------------------- the responder's read

def _responder(db, handler=None):
    """A real MessageProcessor with everything but the steering wiring stubbed out. The handler
    is a sentinel: it records its kwargs and ends the turn, so nothing downstream can mask a
    regression in what base.py passed."""
    from message_processor.base import MessageProcessor

    with patch("message_processor.base.AsyncThreadStateManager"), \
         patch("message_processor.base.OpenAIClient"):
        p = MessageProcessor()
    p.db = db
    p.thread_manager.acquire_thread_lock = AsyncMock(return_value=True)
    p.thread_manager.release_thread_lock = AsyncMock()
    p.thread_manager._token_counter.count_thread_tokens = MagicMock(return_value=0)
    p.thread_manager._token_counter.count_message_tokens = MagicMock(return_value=0)

    state = SimpleNamespace(had_timeout=False, messages=[], thread_ts="10.0", channel_id="C1",
                            root_author=("U1", "human"), config_overrides={}, participants={},
                            current_model=None, has_trimmed_messages=False)

    async def _state_for(*a, **k):
        return state

    p._get_or_rebuild_thread_state = _state_for
    p._process_attachments = AsyncMock(return_value=([], [], []))
    p._handle_text_response = handler or AsyncMock(side_effect=_Stop())
    return p


class _Stop(Exception):
    """Ends a turn at a known point, so a test never catches bare Exception to get there."""


def _turn_msg(**meta):
    md = {"ts": "10.0"}
    md.update(meta)
    return Message(text="hi", user_id="U1", channel_id="C1", thread_id="10.0", metadata=md)


async def _run_turn(p, msg):
    client = MagicMock()
    client.send_message = AsyncMock()
    with patch.object(config, "enable_channel_memory", True):
        return await p.process_message(msg, client, None)


def _passed_steering(p):
    return p._handle_text_response.await_args.kwargs.get("channel_steering_text")


class TestResponderReads:
    async def test_an_ungated_turn_reads_at_the_point_of_no_return(self):
        db = _steering_db([_fact(1, "Pat owns billing")], policy={"content": "only deploys"})
        p = _responder(db)
        await _run_turn(p, _turn_msg())
        db.get_channel_memory_async.assert_awaited_once()
        assert "only deploys" in _passed_steering(p)

    async def test_a_queued_turn_reads_nothing(self):
        # The read costs a query per admitted turn, not per arriving message: a message that
        # queues never reaches the point of no return, and its redispatch reads afresh later.
        db = _steering_db([_fact(1, "a fact")])
        p = _responder(db)
        p.thread_manager.acquire_thread_lock = AsyncMock(return_value=False)
        p.thread_manager.enqueue_pending = MagicMock()
        response = await _run_turn(p, _turn_msg())
        assert response.type == "queued"
        db.get_channel_memory_async.assert_not_awaited()
        db.get_channel_policy_async.assert_not_awaited()

    async def test_a_gated_turn_reuses_the_stamp_and_never_reads_again(self):
        db = _steering_db([_fact(1, "a fact")], policy={"content": "only deploys"})
        p = _responder(db)
        msg = _turn_msg(gate_required=True)
        stamp(msg, ChannelSteeringSnapshot(text="what the gate saw"))
        await _run_turn(p, msg)
        assert _passed_steering(p) == "what the gate saw"
        db.get_channel_memory_async.assert_not_awaited()
        db.get_channel_policy_async.assert_not_awaited()

    async def test_a_policy_change_after_the_gate_does_not_reach_the_responder(self):
        # The whole bug, in one test: the gate judged the message against one version of the
        # channel's rules, so the reply obeys THAT version — not whatever landed since.
        db = _steering_db([], policy={"content": "the policy the responder must not see"})
        p = _responder(db)
        msg = _turn_msg(gate_required=True)
        stamp(msg, ChannelSteeringSnapshot(text="the policy the gate saw"))
        await _run_turn(p, msg)
        assert _passed_steering(p) == "the policy the gate saw"

    async def test_a_gated_turn_with_no_stamp_answers_without_steering(self):
        # An invariant failure, not a cue to go and read: reading here is exactly the divergence
        # this commit removes. It is logged loudly and the turn proceeds with nothing.
        db = _steering_db([_fact(1, "a fact")], policy={"content": "only deploys"})
        p = _responder(db)
        warnings = []
        p.log_warning = lambda m, *a, **k: warnings.append(m)
        await _run_turn(p, _turn_msg(gate_required=True))
        assert _passed_steering(p) is None
        db.get_channel_memory_async.assert_not_awaited()
        assert any("steering" in w for w in warnings)

    async def test_a_failed_ungated_read_is_not_retried_downstream(self):
        db = _steering_db(memory_error=RuntimeError("db gone"),
                          policy_error=RuntimeError("db gone"))
        p = _responder(db)
        await _run_turn(p, _turn_msg())
        assert _passed_steering(p) is None
        # ONE attempt, then live with it. (The policy read fails first and the snapshot is
        # all-or-nothing, so the facts are never even asked for — see TestSnapshotIsAllOrNothing.)
        assert db.get_channel_policy_async.await_count == 1
        assert db.get_channel_memory_async.await_count == 0


# ------------------------------------------------------- the same bytes, as the API receives them

class _GateSpy:
    """The OpenAI client the wake classifier is bound to: runs the REAL prompt builder and keeps
    the payload it would have sent."""

    def __init__(self, wake=True):
        self.client = MagicMock()
        self.payloads = []
        self._wake = wake

    async def classify_wake(self, *, sources, channel_steering_text=None):
        from openai_client.api import responses as responses_api
        return await responses_api.classify_wake(
            self, sources=sources, channel_steering_text=channel_steering_text)

    async def _safe_api_call(self, *a, **k):
        self.payloads.append(k.get("input"))
        item = SimpleNamespace(content=[SimpleNamespace(text=json.dumps({"wake": self._wake}))])
        return SimpleNamespace(output=[item])

    def log_debug(self, *a, **k):
        pass

    def log_warning(self, *a, **k):
        pass

    def log_error(self, *a, **k):
        pass


class _ResponderSpy:
    """Records the system prompt handed to the responding model."""

    def __init__(self):
        self.prompts = []

    async def create_text_response(self, messages=None, system_prompt=None, **kw):
        self.prompts.append(system_prompt)
        return "ok"

    async def _create_text_response_with_timeout(self, **kw):
        return await self.create_text_response(**kw)


class _FakeSlack:
    name = "Slack"
    MAX_MESSAGE_LENGTH = 3900

    def __init__(self):
        self.posted = []
        self.channel_pulse = None

    def supports_streaming(self):
        return False

    def supports_native_streaming(self):
        return False

    def format_text(self, t):
        return t

    async def send_message_get_ts(self, channel, thread, text, lease=None, surface=None):
        self.posted.append(text)
        return {"success": True, "ts": f"ts{len(self.posted)}"}

    async def send_message(self, channel, thread=None, text="", **kw):
        self.posted.append(text)
        return f"ts{len(self.posted)}"

    async def update_message(self, channel, ts, text):
        return True

    async def delete_message(self, channel, ts):
        return True

    async def set_assistant_status(self, channel, thread, status=""):
        return None

    def _record_own_reply_pulse(self, *a, **k):
        pass


def _api_spy_processor(db, openai):
    """A real MessageProcessor that reaches the API with the REAL system prompt."""
    p = _responder(db, handler=None)
    p.openai_client = openai
    p._handle_text_response = p.__class__._handle_text_response.__get__(p)

    async def _passthru(m, *a, **k):
        return m

    async def _none(*a, **k):
        return None

    p._add_message_with_token_management = MagicMock()
    p._inject_image_analyses = _passthru
    p._pre_trim_messages_for_api = _passthru
    p._build_participant_roster = MagicMock(return_value="")
    p._build_suffix_context = MagicMock(return_value="")
    p._build_pulse_envelope = MagicMock(return_value=None)
    p._build_tools_array = MagicMock(return_value=[])
    p._materialize_request_tools = MagicMock(return_value=(None, {}, False, ""))
    p._persist_tool_provenance = MagicMock()
    p._update_status = MagicMock()
    p._build_channel_info = _none
    p._build_channel_summary_block = _none
    p._async_post_response_cleanup = _none
    p._drop_dead_containers = _none
    p._resolve_ci_container = _none

    def _discard(coro, *a, **k):
        if hasattr(coro, "close"):
            coro.close()

    p._schedule_async_call = MagicMock(side_effect=_discard)
    return p


async def test_the_gate_and_the_responder_receive_the_IDENTICAL_snapshot_bytes(monkeypatch):
    """THE test. Two models, two prompts, one string.

    Both payloads are captured where they would leave for the API, and the canonical block must
    be an exact substring of each. Anything that re-renders, reorders, re-prefixes or refetches
    between the two halves breaks this and nothing else — which is precisely why the bug it
    replaces survived so long: each half was individually correct.
    """
    from message_processor.participation import ParticipationEngine

    monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
    rows = [_fact(2, "demos are Fridays"), _fact(1, "Pat owns billing"),
            _fact(3, "react less here", author=MARK)]
    policy = {"id": 77, "content": "only jump in on deploy failures"}
    db = _steering_db(rows, policy=policy)

    # What the bytes must be, derived independently of either consumer.
    expected = render_snapshot(policy, rows).text
    assert expected and POLICY_HEADING in expected

    # --- the gate half: the real engine, the real classifier prompt builder.
    gate_spy = _GateSpy(wake=True)
    app, client = _gate_app(db, None)
    # The REAL engine, including its own arrival bookkeeping — a stubbed note_arrival makes every
    # message look superseded, and the gate would never wake.
    app.participation_engine = ParticipationEngine(gate_spy)
    msg = _gate_msg()
    verdict = await app._gate_verdict(msg, client)
    assert verdict is not None, "the gate must wake for the responder half to run"
    assert gate_spy.payloads, "the classifier must have reached the API"
    gate_prompt = gate_spy.payloads[0][1]["content"]

    # --- the responder half: the same Message object, carrying the same stamp.
    responder_spy = _ResponderSpy()
    p = _api_spy_processor(db, responder_spy)
    with patch.object(config, "enable_channel_memory", True), \
         patch.object(config, "get_thread_config_async",
                      side_effect=AsyncMock(return_value={
                          "model": "gpt-5.6-sol", "temperature": 1.0, "max_tokens": 4096,
                          "enable_streaming": False, "enable_web_search": False,
                          "enable_code_interpreter": False, "reasoning_effort": "low",
                          "verbosity": "medium", "custom_instructions": None})):
        await p.process_message(msg, _FakeSlack(), None)
    assert responder_spy.prompts, "the responder must have reached the API"

    assert expected in gate_prompt, "the gate's payload lost the canonical block"
    assert expected in responder_spy.prompts[0], "the responder's payload lost the canonical block"
    # One read for the whole turn, both halves.
    assert db.get_channel_memory_async.await_count == 1
    assert db.get_channel_policy_async.await_count == 1


def test_every_reentry_path_threads_the_snapshot_rather_than_refetching():
    """A source-level backstop for the five re-entry paths (context retry, streaming fallback,
    MCP retry, timeout retry, non-streaming fallback). Each re-enters a handler, and a handler
    that fetched for itself would hand two attempts of ONE turn different rules."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    missing = []
    for rel in ("message_processor/base.py", "message_processor/handlers/text.py"):
        tree = ast.parse((root / rel).read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            if name not in ("_handle_text_response", "_handle_streaming_text_response"):
                continue
            if not any(kw.arg == "channel_steering_text" for kw in node.keywords):
                missing.append(f"{rel}:{node.lineno}")
    assert not missing, "handler re-entry without the turn's snapshot at: " + ", ".join(missing)


# --------------------------------------------------------------------------- writer 1: the modal

class _FakeApp:
    """Captures handlers registered via @app.action / @app.view (others are no-ops)."""

    def __init__(self):
        self.actions = {}
        self.views = {}

    def action(self, action_id):
        def deco(fn):
            self.actions[action_id] = fn
            return fn
        return deco

    def view(self, callback_id):
        def deco(fn):
            self.views[callback_id] = fn
            return fn
        return deco

    def command(self, *_a, **_k):
        return lambda fn: fn

    def shortcut(self, *_a, **_k):
        return lambda fn: fn

    def event(self, *_a, **_k):
        return lambda fn: fn


def _settings_host(db):
    from slack_client.event_handlers.settings import SlackSettingsHandlersMixin

    host = SlackSettingsHandlersMixin.__new__(SlackSettingsHandlersMixin)
    host.app = _FakeApp()
    host.db = db
    host.settings_modal = SettingsModal.__new__(SettingsModal)
    host.log_info = host.log_error = host.log_debug = host.log_warning = lambda *a, **k: None
    host._register_settings_handlers()
    return host


def _submit_state(policy=None, legacy=None, memory=""):
    state = {"channel_memory_block": {"channel_memory": {"value": memory}}}
    if policy is not None:
        state["policy_block"] = {"standing_policy": {"value": policy}}
    if legacy is not None:
        state["directives_block"] = {"directives": {"value": legacy}}
    return state


async def _submit(host, state, *, policy_seed="", channel_id="C1", user="U1"):
    ephemeral = []
    client = SimpleNamespace(
        chat_postEphemeral=AsyncMock(side_effect=lambda **kw: ephemeral.append(kw.get("text"))))
    view = {"private_metadata": json.dumps({"channel_id": channel_id, "mem_seed": [],
                                            "policy_seed": policy_seed}),
            "state": {"values": state}}
    await host.app.views["channel_settings_modal"](
        ack=AsyncMock(), body={"user": {"id": user}}, view=view, client=client)
    return ephemeral


class TestModalWritesThePolicy:
    async def test_submit_replaces_the_policy_row(self, temp_db):
        await temp_db.set_channel_policy_async("C1", "old rule", author="U0")
        host = _settings_host(temp_db)
        await _submit(host, _submit_state(policy="new rule"),
                      policy_seed=memory_content_hash("old rule"))
        rows = _policy_rows(temp_db)
        assert len(rows) == 1
        assert rows[0]["content"] == "new rule"
        assert rows[0]["author"] == "U1"          # attributed to whoever saved the modal

    async def test_an_emptied_box_clears_the_policy(self, temp_db):
        await temp_db.set_channel_policy_async("C1", "old rule")
        host = _settings_host(temp_db)
        await _submit(host, _submit_state(policy=""),
                      policy_seed=memory_content_hash("old rule"))
        assert _policy_rows(temp_db) == []

    async def test_a_policy_changed_since_open_is_never_overwritten(self, temp_db):
        # This box REPLACES the policy wholesale, so a save from a modal opened before someone
        # else edited it would silently revert their rule — and an empty box would revert it to
        # nothing at all.
        await temp_db.set_channel_policy_async("C1", "changed elsewhere", author="U9")
        host = _settings_host(temp_db)
        notices = await _submit(host, _submit_state(policy="stale edit"),
                                policy_seed=memory_content_hash("what the modal showed"))
        assert (await temp_db.get_channel_policy_async("C1"))["content"] == "changed elsewhere"
        assert any("changed elsewhere while this was open" in (n or "") for n in notices)

    async def test_a_pre_deploy_modal_submits_its_old_field_into_the_policy_row(self, temp_db):
        # A modal opened before this shipped is still on someone's screen; its Save must land
        # somewhere real. It writes the policy ROW — never the legacy column.
        host = _settings_host(temp_db)
        await _submit(host, _submit_state(legacy="only jump in on deploy failures"))
        assert (await temp_db.get_channel_policy_async("C1"))["content"] \
            == "only jump in on deploy failures"
        assert _directives_of(temp_db, "C1") is None

    async def test_an_unchanged_box_writes_nothing(self, temp_db):
        await temp_db.set_channel_policy_async("C1", "same rule", author="U0")
        host = _settings_host(temp_db)
        await _submit(host, _submit_state(policy="same rule"),
                      policy_seed=memory_content_hash("same rule"))
        assert _policy_rows(temp_db)[0]["author"] == "U0"      # authorship not stolen by a no-op

    async def test_the_settings_write_no_longer_carries_directives(self, temp_db):
        host = _settings_host(temp_db)
        await _submit(host, _submit_state(policy="only deploys"))
        row = await temp_db.get_channel_settings_async("C1")
        assert row is not None and "directives" not in row


# ------------------------------------------------------ writer 2: set_channel_participation

def _tool_ctx(db, **kw):
    defaults = dict(channel_id="C1", thread_ts="1.0", trigger_ts="1.0", user_id="U1", db=db,
                    is_dm=False, structural_change_authorized=True,
                    message=SimpleNamespace(metadata={"sender_type": "human"}))
    defaults.update(kw)
    return SimpleNamespace(**defaults)


async def _set_participation(db, args, **ctx_kw):
    from message_processor.participation_tools import execute_set_channel_participation
    return await execute_set_channel_participation(_tool_ctx(db, **ctx_kw), args)


class TestParticipationToolPolicy:
    async def test_standing_policy_alone_is_a_complete_call(self, temp_db):
        res = await _set_participation(temp_db, {"standing_policy": "only jump in on deploys"})
        assert res["ok"] and "standing policy replaced" in res["confirmation"]
        assert (await temp_db.get_channel_policy_async("C1"))["content"] == "only jump in on deploys"

    async def test_an_empty_string_clears_the_policy(self, temp_db):
        await temp_db.set_channel_policy_async("C1", "old rule")
        res = await _set_participation(temp_db, {"standing_policy": ""})
        assert res["ok"] and "cleared" in res["confirmation"]
        assert await temp_db.get_channel_policy_async("C1") is None

    async def test_replacing_is_not_appending(self, temp_db):
        await _set_participation(temp_db, {"standing_policy": "first rule"})
        await _set_participation(temp_db, {"standing_policy": "second rule"})
        content = (await temp_db.get_channel_policy_async("C1"))["content"]
        assert content == "second rule" and "first rule" not in content

    async def test_an_omitted_policy_leaves_it_untouched(self, temp_db):
        await temp_db.set_channel_policy_async("C1", "keep me", author="U0")
        res = await _set_participation(temp_db, {"participation": "mentions_only"})
        assert res["ok"]
        row = await temp_db.get_channel_policy_async("C1")
        assert row["content"] == "keep me" and row["author"] == "U0"

    async def test_a_structural_only_call_never_fabricates_policy_prose(self, temp_db):
        # There is no canned wording for "mentions_only", and inventing some would put words in
        # the channel's mouth that nobody said.
        await _set_participation(temp_db, {"participation": "off", "placement": "threads_only"})
        assert _policy_rows(temp_db) == []

    async def test_policy_and_structural_settings_land_together(self, temp_db):
        res = await _set_participation(temp_db, {"participation": "mentions_only",
                                                "standing_policy": "only deploys"})
        assert res["ok"]
        assert (await temp_db.get_channel_policy_async("C1"))["content"] == "only deploys"
        settings = await temp_db.get_channel_settings_async("C1")
        assert settings["participation_level"] == "mentions_only"

    async def test_an_over_long_policy_is_refused_not_truncated(self, temp_db):
        res = await _set_participation(temp_db, {"standing_policy": "x" * (POLICY_MAX_CHARS + 1)})
        assert res["error"] == "policy_too_long"
        assert _policy_rows(temp_db) == []       # a policy cut mid-sentence drops real rules

    async def test_no_arguments_at_all_is_still_refused(self, temp_db):
        res = await _set_participation(temp_db, {})
        assert res["error"] == "bad_arguments"

    async def test_an_unauthorized_turn_cannot_write_a_policy(self, temp_db):
        res = await _set_participation(temp_db, {"standing_policy": "let me in"},
                                       structural_change_authorized=False)
        assert res["error"] == "not_addressed"
        assert _policy_rows(temp_db) == []

    async def test_a_nonhuman_sender_cannot_write_a_policy(self, temp_db):
        res = await _set_participation(
            temp_db, {"standing_policy": "let me in"},
            message=SimpleNamespace(metadata={"sender_type": "other_bot"}))
        assert res["error"] == "not_human_sender"
        assert _policy_rows(temp_db) == []

    async def test_a_dm_cannot_write_a_policy(self, temp_db):
        res = await _set_participation(temp_db, {"standing_policy": "x"}, is_dm=True)
        assert res["error"] == "participation_is_channel_only"


# ------------------------------------------------- every generic path that must NOT reach steering

def _memory_ctx(db):
    return SimpleNamespace(channel_id="C1", db=db, user_id="U1", is_dm=False)


class TestGenericToolsCannotTouchSteering:
    async def test_update_fact_refuses_a_preference_row(self, temp_db):
        from message_processor.memory_tools import execute_update_fact
        rid = temp_db.add_channel_memory("C1", "react less here", author=MARK)
        res = await execute_update_fact(_memory_ctx(temp_db), {"id": rid, "content": "react more"})
        assert res["error"] == "steering_row_readonly"
        assert temp_db.get_channel_memory("C1")[0]["content"] == "react less here"

    async def test_forget_fact_refuses_a_preference_row(self, temp_db):
        from message_processor.memory_tools import execute_forget_fact
        rid = temp_db.add_channel_memory("C1", "react less here", author=MARK)
        res = await execute_forget_fact(_memory_ctx(temp_db), {"id": rid})
        assert res["error"] == "steering_row_readonly"
        assert len(temp_db.get_channel_memory("C1")) == 1

    async def test_the_policy_row_is_not_even_addressable(self, temp_db):
        from message_processor.memory_tools import execute_forget_fact, execute_update_fact
        await temp_db.set_channel_policy_async("C1", "only deploys")
        pid = _policy_rows(temp_db)[0]["id"]
        assert (await execute_update_fact(
            _memory_ctx(temp_db), {"id": pid, "content": "anything"}))["error"] == "not_found"
        assert (await execute_forget_fact(
            _memory_ctx(temp_db), {"id": pid}))["error"] == "not_found"
        assert (await temp_db.get_channel_policy_async("C1"))["content"] == "only deploys"

    async def test_remember_fact_capacity_ignores_steering_rows(self, temp_db, monkeypatch):
        from message_processor.memory_tools import execute_remember_fact
        monkeypatch.setattr(config, "memory_max_rows", 2, raising=False)
        await temp_db.set_channel_policy_async("C1", "only deploys")
        temp_db.add_channel_memory("C1", "react less here", author=MARK)
        temp_db.add_channel_memory("C1", "one real fact")
        res = await execute_remember_fact(_memory_ctx(temp_db), {"content": "a second real fact"})
        assert res["ok"], "steering rows must not consume the fact cap"
        res = await execute_remember_fact(_memory_ctx(temp_db), {"content": "a third real fact"})
        assert res["error"] == "memory_full"        # the cap still applies to FACTS

    async def test_the_fallback_extractor_cannot_evict_or_revise_steering(self, temp_db,
                                                                         monkeypatch):
        from message_processor.thread_management import ThreadManagementMixin
        monkeypatch.setattr(config, "memory_max_rows", 1, raising=False)
        await temp_db.set_channel_policy_async("C1", "only deploys")
        pref_id = temp_db.add_channel_memory("C1", "react less here", author=MARK)

        proc = ThreadManagementMixin.__new__(type("P", (ThreadManagementMixin,), {}))
        proc.db = temp_db
        proc.log_info = proc.log_debug = proc.log_error = lambda *a, **k: None
        proc.openai_client = MagicMock()
        # The extractor names the preference row — the only ids it was shown are ordinary facts,
        # so this is either a hallucination or a stale id, and either way it is refused.
        proc.openai_client.extract_memory = AsyncMock(
            return_value={"action": "update", "id": pref_id, "content": "react constantly"})
        state = SimpleNamespace(channel_id="C1", messages=[
            {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
        with patch.object(config, "enable_channel_memory", True):
            await proc._async_extract_channel_memory(state)
        rows = {r["id"]: r["content"] for r in temp_db.get_channel_memory("C1")}
        assert rows[pref_id] == "react less here"
        assert (await temp_db.get_channel_policy_async("C1"))["content"] == "only deploys"

    async def test_the_extractor_evicts_only_ordinary_facts(self, temp_db, monkeypatch):
        from message_processor.thread_management import ThreadManagementMixin
        monkeypatch.setattr(config, "memory_max_rows", 1, raising=False)
        pref_id = temp_db.add_channel_memory("C1", "react less here", author=MARK)
        old_id = temp_db.add_channel_memory("C1", "a stale fact")

        proc = ThreadManagementMixin.__new__(type("P", (ThreadManagementMixin,), {}))
        proc.db = temp_db
        proc.log_info = proc.log_debug = proc.log_error = lambda *a, **k: None
        proc.openai_client = MagicMock()
        proc.openai_client.extract_memory = AsyncMock(
            return_value={"action": "add", "content": "a fresh fact"})
        state = SimpleNamespace(channel_id="C1", messages=[
            {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
        with patch.object(config, "enable_channel_memory", True):
            await proc._async_extract_channel_memory(state)
        contents = {r["id"]: r["content"] for r in temp_db.get_channel_memory("C1")}
        assert old_id not in contents                 # the stale FACT was evicted
        assert contents[pref_id] == "react less here"  # the preference was not


# --------------------------------------------------------------------------- the gate's prompt

# --------------------------------------------------------------------------- the removal guard

def test_no_active_reader_or_writer_of_channel_settings_directives_remains():
    """The legacy column survives in SQLite so the migration has something to read. Any OTHER
    live path to it is a bug in waiting: the migration NULLs the column, so a straggler reader
    would silently start seeing nothing, and a straggler writer would store rules nobody reads.

    Two mentions are legitimate and named here, so adding a third has to be deliberate:
    database.py (the migration and the schema it reads) and the settings submit handler's
    acceptance of the OLD modal field id from a pre-deploy modal still open on someone's screen.
    """
    import pathlib

    # Producers, consumers, prompt arguments and settings writers — the four shapes the removal
    # was about. Bare prose mentions are not interesting; these are.
    patterns = ("channel_directives", "directives=", '["directives"]', "['directives']",
                'get("directives")', "get('directives')", "directives_block")
    allowed = {
        "database.py",                                  # the migration + the column it reads
        "slack_client/event_handlers/settings.py",      # accepts the pre-deploy modal's old field
    }
    root = pathlib.Path(__file__).resolve().parents[2]
    skip_dirs = {"tests", "venv", ".venv", "node_modules", "build", "dist", "__pycache__",
                 "site-packages", "Docs"}
    offenders = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if set(rel.parts) & skip_dirs or any(p.startswith(".") for p in rel.parts):
            continue
        if str(rel) in allowed:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "migrate_channel_directives_to_policy_async" in line:
                continue                                # the migration's caller, by design
            if any(p in line for p in patterns):
                offenders.append(f"{rel}:{lineno}")
    assert not offenders, "live channel_directives paths remain at: " + ", ".join(offenders)


def test_the_two_allowed_directives_mentions_are_still_what_the_guard_thinks_they_are():
    """The allow-list above is only honest if those two files use the column for exactly the two
    reasons named. Pinned so a future edit cannot quietly hide a third path behind the exemption.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    settings = (root / "slack_client/event_handlers/settings.py").read_text(encoding="utf-8")
    # Read as INPUT for the policy row; never written back to the column.
    assert "directives_block" in settings
    assert "directives=" not in settings

    db = (root / "database.py").read_text(encoding="utf-8")
    # The only writes are the migration's own NULL-out; nothing else sets the column.
    assert db.count("SET directives = NULL") == 1
    assert "provided[\"directives\"]" not in db


# --------------------------------------------------------------------------- the responder's prompt

def test_the_responder_is_told_to_consider_a_durable_write_at_the_end_of_the_turn():
    """A prompt STEP, not a mandatory call: the default is to write nothing, and the two kinds of
    write are pointed at different tools so a behavioural rule never lands as a fact."""
    from prompts import LOCAL_TOOLS_GUIDANCE

    assert ("Consider whether there is anything durable to write. The default is nothing. "
            "Store only stable facts or explicitly stated preferences, never a transcript. "
            "Replace standing behavioral policy through the policy operation; use ordinary "
            "memory tools only for background facts.") in LOCAL_TOOLS_GUIDANCE


def test_the_memory_tools_point_behavioural_rules_at_the_policy_operation():
    from message_processor.memory_tools import (get_forget_fact_schema, get_remember_fact_schema,
                                                get_update_fact_schema)

    for schema in (get_remember_fact_schema(), get_update_fact_schema(), get_forget_fact_schema()):
        assert "standing_policy" in schema["description"], schema["name"]


class TestMemoryOffStillSteers:
    """ENABLE_CHANNEL_MEMORY governs remembered FACTS. If it also disabled the policy, the
    directives migration would quietly switch off every live operator rule in the workspace."""

    async def test_the_tool_still_writes_a_policy_with_memory_off(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "enable_channel_memory", False, raising=False)
        res = await _set_participation(temp_db, {"standing_policy": "only deploys"})
        assert res["ok"]
        assert (await temp_db.get_channel_policy_async("C1"))["content"] == "only deploys"

    async def test_the_modal_still_writes_a_policy_with_memory_off(self, temp_db, monkeypatch):
        monkeypatch.setattr(config, "enable_channel_memory", False, raising=False)
        host = _settings_host(temp_db)
        await _submit(host, _submit_state(policy="only deploys"))
        assert (await temp_db.get_channel_policy_async("C1"))["content"] == "only deploys"

    async def test_a_turn_with_memory_off_still_carries_the_policy_to_the_responder(self, temp_db):
        temp_db.add_channel_memory("C1", "a remembered fact")
        await temp_db.set_channel_policy_async("C1", "only deploys")
        p = _responder(temp_db)
        with patch.object(config, "enable_channel_memory", False):
            client = MagicMock()
            client.send_message = AsyncMock()
            await p.process_message(_turn_msg(), client, None)
        passed = _passed_steering(p)
        assert "only deploys" in passed
        assert "a remembered fact" not in passed


# ------------------------------------------------- the six review findings, pinned individually

class TestPolicyCompareAndSwap:
    """Finding 1. The modal REPLACES the policy wholesale, so its conflict check has to be part
    of the same transaction as the write. Read-then-write in two statements narrows the race
    without closing it: a save landing between them is silently overwritten."""

    async def test_a_matching_hash_applies(self, temp_db):
        await temp_db.set_channel_policy_async("C1", "old rule", author="U0")
        applied = await temp_db.set_channel_policy_if_unchanged_async(
            "C1", memory_content_hash("old rule"), "new rule", author="U1")
        assert applied is True
        assert (await temp_db.get_channel_policy_async("C1"))["content"] == "new rule"

    async def test_a_stale_hash_is_refused_and_changes_nothing(self, temp_db):
        await temp_db.set_channel_policy_async("C1", "someone else's rule", author="U9")
        applied = await temp_db.set_channel_policy_if_unchanged_async(
            "C1", memory_content_hash("what my modal showed"), "my rule", author="U1")
        assert applied is False
        row = await temp_db.get_channel_policy_async("C1")
        assert row["content"] == "someone else's rule" and row["author"] == "U9"

    async def test_an_empty_expectation_matches_only_an_absent_policy(self, temp_db):
        assert await temp_db.set_channel_policy_if_unchanged_async(
            "C1", "", "first rule", author="U1") is True
        assert await temp_db.set_channel_policy_if_unchanged_async(
            "C1", "", "second rule", author="U2") is False       # no longer absent
        assert (await temp_db.get_channel_policy_async("C1"))["content"] == "first rule"

    async def test_a_matching_hash_with_blank_content_clears(self, temp_db):
        await temp_db.set_channel_policy_async("C1", "old rule")
        assert await temp_db.set_channel_policy_if_unchanged_async(
            "C1", memory_content_hash("old rule"), "") is True
        assert _policy_rows(temp_db) == []

    async def test_concurrent_swaps_from_the_same_seed_leave_exactly_one_winner(self, temp_db):
        # The point of the whole finding: both callers were shown "old rule", both try to replace
        # it. One wins; the other is told it lost, rather than clobbering the winner.
        await temp_db.set_channel_policy_async("C1", "old rule", author="U0")
        seed = memory_content_hash("old rule")
        results = await asyncio.gather(*[
            temp_db.set_channel_policy_if_unchanged_async("C1", seed, f"rule {i}", author=f"U{i}")
            for i in range(6)])
        assert sum(1 for r in results if r) == 1
        assert len(_policy_rows(temp_db)) == 1
        assert _policy_rows(temp_db)[0]["content"] != "old rule"

    async def test_the_submit_handler_goes_through_the_swap(self, temp_db):
        # And the handler must actually USE it — a handler that still read-then-wrote would pass
        # every test above while leaving the race exactly where it was.
        host = _settings_host(temp_db)
        await temp_db.set_channel_policy_async("C1", "changed elsewhere", author="U9")
        calls = []
        real = temp_db.set_channel_policy_if_unchanged_async

        async def _spy(*a, **kw):
            calls.append(a)
            return await real(*a, **kw)

        temp_db.set_channel_policy_if_unchanged_async = _spy
        notices = await _submit(host, _submit_state(policy="stale edit"),
                                policy_seed=memory_content_hash("what the modal showed"))
        assert calls, "the submit handler must use the compare-and-swap"
        assert (await temp_db.get_channel_policy_async("C1"))["content"] == "changed elsewhere"
        assert any("changed elsewhere while this was open" in (n or "") for n in notices)


class TestMergeRuleIsShared:
    """Findings 3/4 lean on one rule — preserve both texts — used by the migration and by a
    legacy modal. Both go through ``merged_policy_text`` so they cannot drift apart."""

    def test_incoming_into_nothing(self):
        from database import merged_policy_text
        assert merged_policy_text("", "only deploys") == "only deploys"

    def test_nothing_incoming_never_clears(self):
        from database import merged_policy_text
        assert merged_policy_text("only deploys", "") == "only deploys"
        assert merged_policy_text("", "") is None

    def test_identical_after_normalization_is_a_noop(self):
        from database import merged_policy_text
        assert merged_policy_text("only  deploys", "only deploys") == "only  deploys"

    def test_different_texts_are_both_preserved_in_order(self):
        from database import merged_policy_text
        assert merged_policy_text("threads only", "only deploys") == "threads only\nonly deploys"

    async def test_merge_never_deletes_on_a_blank(self, temp_db):
        await temp_db.set_channel_policy_async("C1", "only deploys", author="U0")
        await temp_db.merge_channel_policy_async("C1", "", author="U1")
        row = await temp_db.get_channel_policy_async("C1")
        assert row["content"] == "only deploys" and row["author"] == "U0"   # untouched entirely

    async def test_merge_preserves_both(self, temp_db):
        await temp_db.set_channel_policy_async("C1", "threads only")
        await temp_db.merge_channel_policy_async("C1", "only deploys", author="U1")
        content = (await temp_db.get_channel_policy_async("C1"))["content"]
        assert "threads only" in content and "only deploys" in content


class TestLegacyModalAfterTheMigration:
    """Finding 4, as the deployed sequence actually runs it: startup migrates the directives into
    the policy row, and only THEN does someone hit Save on a modal that was already open."""

    async def test_the_deployed_sequence(self, temp_db):
        _legacy_directives(temp_db, "C1", "only jump in on deploy failures")
        assert await temp_db.migrate_channel_directives_to_policy_async() == (1, 0)

        # The old modal has no policy_seed at all (its private_metadata predates the field) and
        # submits the old directives box. Under a hash check it would be refused every time —
        # the migration it is racing has already made the stored policy nonempty.
        host = _settings_host(temp_db)
        view = {"private_metadata": json.dumps({"channel_id": "C1", "mem_seed": []}),
                "state": {"values": _submit_state(legacy="only jump in on deploy failures")}}
        notices = []
        client = SimpleNamespace(chat_postEphemeral=AsyncMock(
            side_effect=lambda **kw: notices.append(kw.get("text"))))
        await host.app.views["channel_settings_modal"](
            ack=AsyncMock(), body={"user": {"id": "U1"}}, view=view, client=client)

        assert (await temp_db.get_channel_policy_async("C1"))["content"] \
            == "only jump in on deploy failures"
        assert len(_policy_rows(temp_db)) == 1          # deduped, not doubled
        assert not any("was changed elsewhere" in (n or "") for n in notices)

    async def test_a_legacy_modal_with_new_text_preserves_both(self, temp_db):
        await temp_db.set_channel_policy_async("C1", "migrated rule")
        host = _settings_host(temp_db)
        view = {"private_metadata": json.dumps({"channel_id": "C1"}),
                "state": {"values": _submit_state(legacy="a different rule")}}
        await host.app.views["channel_settings_modal"](
            ack=AsyncMock(), body={"user": {"id": "U1"}}, view=view,
            client=SimpleNamespace(chat_postEphemeral=AsyncMock()))
        content = (await temp_db.get_channel_policy_async("C1"))["content"]
        assert "migrated rule" in content and "a different rule" in content

    async def test_an_empty_legacy_box_never_clears_a_policy(self, temp_db):
        # A modal that cannot see the policy cannot mean "delete it".
        await temp_db.set_channel_policy_async("C1", "migrated rule", author="U0")
        host = _settings_host(temp_db)
        view = {"private_metadata": json.dumps({"channel_id": "C1"}),
                "state": {"values": _submit_state(legacy="")}}
        await host.app.views["channel_settings_modal"](
            ack=AsyncMock(), body={"user": {"id": "U1"}}, view=view,
            client=SimpleNamespace(chat_postEphemeral=AsyncMock()))
        assert (await temp_db.get_channel_policy_async("C1"))["content"] == "migrated rule"

    async def test_a_current_modal_CAN_still_clear(self, temp_db):
        # The distinction is the seed's presence, not its value: a modal that shows the policy is
        # allowed to empty it.
        await temp_db.set_channel_policy_async("C1", "migrated rule")
        host = _settings_host(temp_db)
        await _submit(host, _submit_state(policy=""),
                      policy_seed=memory_content_hash("migrated rule"))
        assert _policy_rows(temp_db) == []


class TestConstrainedFactUpdate:
    """Finding 2. The extractor's id comes from a utility model, so the WHERE clause — not a
    caller's check — is what keeps it away from steering."""

    async def test_it_updates_an_ordinary_fact(self, temp_db):
        rid = temp_db.add_channel_memory("C1", "old wording")
        assert await temp_db.update_channel_fact_async(rid, "new wording") is True
        assert temp_db.get_channel_memory("C1")[0]["content"] == "new wording"

    async def test_it_cannot_touch_a_preference_row(self, temp_db):
        rid = temp_db.add_channel_memory("C1", "react less here", author=MARK)
        assert await temp_db.update_channel_fact_async(rid, "react constantly") is False
        assert temp_db.get_channel_memory("C1")[0]["content"] == "react less here"

    async def test_it_cannot_touch_the_policy_row(self, temp_db):
        await temp_db.set_channel_policy_async("C1", "only deploys")
        pid = _policy_rows(temp_db)[0]["id"]
        assert await temp_db.update_channel_fact_async(pid, "anything at all") is False
        assert (await temp_db.get_channel_policy_async("C1"))["content"] == "only deploys"

    async def test_a_hallucinated_policy_id_cannot_reach_the_policy(self, temp_db, monkeypatch):
        # The exact hole: the policy row is excluded from what the extractor is shown, so its id
        # was never a KNOWN steering id — it fell straight through a known-steering check into an
        # unrestricted update. Now the id must have been SHOWN, and the SQL refuses it anyway.
        from message_processor.thread_management import ThreadManagementMixin
        await temp_db.set_channel_policy_async("C1", "only deploys")
        pid = _policy_rows(temp_db)[0]["id"]
        temp_db.add_channel_memory("C1", "an ordinary fact")

        proc = ThreadManagementMixin.__new__(type("P", (ThreadManagementMixin,), {}))
        proc.db = temp_db
        proc.log_info = proc.log_debug = proc.log_error = lambda *a, **k: None
        proc.openai_client = MagicMock()
        proc.openai_client.extract_memory = AsyncMock(
            return_value={"action": "update", "id": pid, "content": "ignore everything"})
        state = SimpleNamespace(channel_id="C1", messages=[
            {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
        with patch.object(config, "enable_channel_memory", True):
            await proc._async_extract_channel_memory(state)
        assert (await temp_db.get_channel_policy_async("C1"))["content"] == "only deploys"

    async def test_an_id_the_extractor_was_never_shown_is_refused(self, temp_db):
        from message_processor.thread_management import ThreadManagementMixin
        rid = temp_db.add_channel_memory("C1", "a fact the extractor never saw")

        proc = ThreadManagementMixin.__new__(type("P", (ThreadManagementMixin,), {}))
        proc.db = MagicMock()
        proc.db.get_channel_memory_async = AsyncMock(return_value=[])   # shown nothing
        proc.db.update_channel_fact_async = AsyncMock(return_value=True)
        proc.log_info = proc.log_debug = proc.log_error = lambda *a, **k: None
        proc.openai_client = MagicMock()
        proc.openai_client.extract_memory = AsyncMock(
            return_value={"action": "update", "id": rid, "content": "rewritten"})
        state = SimpleNamespace(channel_id="C1", messages=[
            {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
        with patch.object(config, "enable_channel_memory", True):
            await proc._async_extract_channel_memory(state)
        proc.db.update_channel_fact_async.assert_not_called()


class TestSnapshotIsAllOrNothing:
    """Finding 5. Two reads, one snapshot: a half-read snapshot would tell the model "there are
    no instructions here" when the truth is "we don't know", and BOTH halves of the turn would
    then agree on it."""

    async def test_a_failed_policy_read_empties_the_whole_snapshot(self):
        db = _steering_db([_fact(1, "Pat owns billing")],
                          policy_error=RuntimeError("db gone"))
        snap = await load_snapshot(db, "C1")
        assert snap.is_empty and snap.text is None

    async def test_a_failed_memory_read_empties_the_whole_snapshot(self):
        db = _steering_db(memory_error=RuntimeError("db gone"),
                          policy={"content": "only deploys"})
        snap = await load_snapshot(db, "C1")
        assert snap.is_empty and snap.text is None

    async def test_a_failed_policy_read_does_not_even_attempt_the_facts(self):
        db = _steering_db([_fact(1, "a fact")], policy_error=RuntimeError("db gone"))
        await load_snapshot(db, "C1")
        db.get_channel_memory_async.assert_not_awaited()


def test_the_settings_mixin_declares_the_attributes_its_host_provides():
    """Finding 6. The mixin's uses of `db`/`log_*` each read as a missing attribute to the type
    checker, which drowned out anything real the file might say. Declared the way the other
    mixins in slack_client/ declare theirs."""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    src = (root / "slack_client/event_handlers/settings.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "SlackSettingsHandlersMixin")
    guarded = [n for n in cls.body
               if isinstance(n, ast.If) and getattr(n.test, "id", "") == "TYPE_CHECKING"]
    assert guarded, "the host-provided attributes are not declared"
    declared = {t.target.id for t in guarded[0].body if isinstance(t, ast.AnnAssign)}
    assert {"db", "app", "settings_modal", "log_info", "log_warning", "log_error"} <= declared


class TestStartupRefusesToRunWithUnmigratedRules:
    """Finding 3. The alternative — log the failure and carry on — is the worst outcome available
    here: the bot comes up looking healthy in a channel whose "only speak up about deploys" is now
    in a column nothing reads, and behaves as though nobody ever set a rule. Nothing downstream
    can detect that. A process that refuses to start is visible in the first place anyone looks.
    """

    def _app(self, monkeypatch, migrate):
        import main as main_mod

        app = main_mod.ChatBotV2.__new__(main_mod.ChatBotV2)
        app.platform = "slack"
        app.participation_engine = None
        # Startup runs THREE state migrations, each fatal on failure for the same reason. This
        # class drives the directives one; the other two must succeed so the abort under test is
        # unambiguously the one being injected.
        fake_db = SimpleNamespace(
            migrate_channel_directives_to_policy_async=migrate,
            migrate_participation_levels_to_binary_async=AsyncMock(return_value=(0, 0)),
            migrate_participation_prefs_to_policy_async=AsyncMock(return_value=(0, 0)))
        fake_client = SimpleNamespace(db=fake_db, processor=None)

        import slack_client
        monkeypatch.setattr(slack_client, "SlackBot", lambda **kw: fake_client)
        monkeypatch.setattr(main_mod, "MessageProcessor", lambda **kw: MagicMock())
        monkeypatch.setattr(main_mod, "ParticipationEngine", lambda *a, **kw: MagicMock())
        monkeypatch.setattr(main_mod.participation_telemetry, "initialize", lambda *a, **k: None)
        monkeypatch.setattr(main_mod.config, "validate", lambda: None)
        monkeypatch.setattr(main_mod.signal, "signal", lambda *a, **k: None)
        return app

    async def test_a_failed_channel_aborts_startup(self, monkeypatch):
        app = self._app(monkeypatch, AsyncMock(return_value=(3, 1)))
        with pytest.raises(SystemExit) as exit_info:
            await app.initialize()
        assert exit_info.value.code == 1

    async def test_a_migration_that_cannot_run_at_all_aborts_startup(self, monkeypatch):
        app = self._app(monkeypatch, AsyncMock(side_effect=RuntimeError("database is locked")))
        with pytest.raises(SystemExit) as exit_info:
            await app.initialize()
        assert exit_info.value.code == 1

    async def test_a_clean_migration_starts_normally(self, monkeypatch):
        migrate = AsyncMock(return_value=(2, 0))
        app = self._app(monkeypatch, migrate)
        await app.initialize()                    # no SystemExit
        migrate.assert_awaited_once()
