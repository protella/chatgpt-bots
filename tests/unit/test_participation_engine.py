"""Phase F — ParticipationEngine: verdict validation, level/mode mapping, debounce,
throttle rails, snooze lifecycle, backoff writes, placement wiring, modal dual-write,
DB columns/migration, and the busy-rejection needs_refresh fix.

All stubbed I/O — no live bot, no legacy suite.
"""
from __future__ import annotations

import asyncio
import datetime
import inspect
import sqlite3
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from base_client import Message
from config import config
from database import DatabaseManager
from message_processor.participation import (
    LEVEL_TO_MODE, MODE_TO_LEVEL, ParticipationEngine, ParticipationVerdict,
    is_snoozed, resolve_participation_level, snooze_expiry_iso,
)


# ------------------------------------------------------------------ level resolution

class TestLevelResolution:
    def test_mode_mapping_round_trip(self):
        assert MODE_TO_LEVEL == {"off": "off", "tag_only": "mentions_only", "auto_respond": "judicious"}
        for level, mode in LEVEL_TO_MODE.items():
            if level != "active":  # active has no distinct legacy mode
                assert MODE_TO_LEVEL[mode] in (level, "judicious")

    def test_participation_level_wins_over_mode(self):
        cs = {"participation_level": "active", "response_mode": "off"}
        assert resolve_participation_level(cs) == "active"

    def test_falls_back_to_row_mode(self):
        assert resolve_participation_level({"response_mode": "auto_respond"}) == "judicious"
        assert resolve_participation_level({"response_mode": "off"}) == "off"

    def test_falls_back_to_global_default(self, monkeypatch):
        monkeypatch.setattr(config, "channel_response_mode", "tag_only", raising=False)
        assert resolve_participation_level(None) == "mentions_only"
        monkeypatch.setattr(config, "channel_response_mode", "auto_respond", raising=False)
        assert resolve_participation_level({}) == "judicious"

    def test_garbage_degrades_safe(self, monkeypatch):
        monkeypatch.setattr(config, "channel_response_mode", "banana", raising=False)
        assert resolve_participation_level({"participation_level": "loud"}) == "mentions_only"


# ------------------------------------------------------------------------ snooze

class TestSnooze:
    def test_future_snooze_active(self):
        cs = {"snoozed_until": snooze_expiry_iso(hours=1)}
        assert is_snoozed(cs) is True

    def test_past_snooze_expired(self):
        past = (datetime.datetime.now(datetime.timezone.utc)
                - datetime.timedelta(hours=1)).isoformat(timespec="seconds")
        assert is_snoozed({"snoozed_until": past}) is False

    def test_absent_and_malformed_not_snoozed(self):
        assert is_snoozed(None) is False
        assert is_snoozed({}) is False
        assert is_snoozed({"snoozed_until": "not-a-date"}) is False

    def test_expiry_deterministic_given_now(self):
        now = datetime.datetime(2026, 7, 9, 12, 0, tzinfo=datetime.timezone.utc)
        assert snooze_expiry_iso(hours=4, now=now) == snooze_expiry_iso(hours=4, now=now)
        assert "16:00" in snooze_expiry_iso(hours=4, now=now)


# ----------------------------------------------------------------- verdict validation

class TestVerdictValidation:
    def test_malformed_and_invalid_action_ignore(self):
        assert ParticipationEngine.validate_verdict(None).action == "ignore"
        assert ParticipationEngine.validate_verdict("respond").action == "ignore"
        assert ParticipationEngine.validate_verdict({"action": "shout"}).action == "ignore"

    def test_respond_defaults(self):
        v = ParticipationEngine.validate_verdict({"action": "respond"})
        assert (v.action, v.placement, v.emoji) == ("respond", "thread", None)

    def test_react_allowlist_and_colon_strip(self, monkeypatch):
        monkeypatch.setattr(config, "reaction_emojis", ["thumbsup", "eyes"], raising=False)
        assert ParticipationEngine.validate_verdict(
            {"action": "react", "emoji": ":eyes:"}).emoji == "eyes"
        # off-allowlist → first allowlisted emoji
        assert ParticipationEngine.validate_verdict(
            {"action": "react", "emoji": "middle_finger"}).emoji == "thumbsup"

    def test_bad_placement_coerced_to_thread(self):
        v = ParticipationEngine.validate_verdict({"action": "respond", "placement": "everywhere"})
        assert v.placement == "thread"

    def test_reason_truncated(self):
        v = ParticipationEngine.validate_verdict({"action": "ignore", "reason": "x" * 999})
        assert len(v.reason) == 300


# ------------------------------------------------------------------- debounce + rails

class _FakeClient:
    def __init__(self, verdict):
        self._verdict = verdict
        self.calls = 0

    async def classify_participation(self, text, signals=None):
        self.calls += 1
        return self._verdict


class _FakePulse:
    def __init__(self, count=0):
        self._count = count

    def unprompted_count_last_hour(self, channel_id):
        return self._count

    def render_envelope(self, *a, **k):
        return "[Recent channel activity]\n- Peter (top-level): hi"


class TestDebounceAndRails:
    @pytest.mark.asyncio
    async def test_rapid_fire_collapses_to_latest(self, monkeypatch):
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
        fake = _FakeClient({"action": "respond"})
        engine = ParticipationEngine(fake)
        first = asyncio.create_task(engine.evaluate(channel_id="C1", ts="1.0", text="line one"))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(engine.evaluate(channel_id="C1", ts="2.0", text="line two"))
        r1, r2 = await asyncio.gather(first, second)
        assert r1 is None            # superseded — silent
        assert r2.action == "respond"
        assert fake.calls == 1       # ONE engine call for the burst

    @pytest.mark.asyncio
    async def test_channels_debounce_independently(self, monkeypatch):
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.02, raising=False)
        fake = _FakeClient({"action": "ignore"})
        engine = ParticipationEngine(fake)
        r1, r2 = await asyncio.gather(
            engine.evaluate(channel_id="C1", ts="1.0", text="a"),
            engine.evaluate(channel_id="C2", ts="1.0", text="b"),
        )
        assert r1 is not None and r2 is not None
        assert fake.calls == 2

    @pytest.mark.asyncio
    async def test_engine_api_failure_is_silent_ignore(self, monkeypatch):
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)

        class _Boom:
            async def classify_participation(self, *a, **k):
                raise RuntimeError("api down")

        v = await ParticipationEngine(_Boom()).evaluate(channel_id="C1", ts="1.0", text="x")
        assert v.action == "ignore"

    def test_hourly_cap_and_active_multiplier(self, monkeypatch):
        monkeypatch.setattr(config, "max_unprompted_replies_per_hour", 6, raising=False)
        engine = ParticipationEngine(MagicMock())
        assert engine.hourly_cap("judicious") == 6
        assert engine.hourly_cap("active") == 12
        assert engine.over_throttle(_FakePulse(6), "C1", "judicious") is True
        assert engine.over_throttle(_FakePulse(6), "C1", "active") is False
        assert engine.over_throttle(None, "C1", "judicious") is False


# --------------------------------------------------------------- main.py gate wiring

def _make_app(verdict, pulse=None, engine_enabled=True, monkeypatch=None):
    from main import ChatBotV2
    app = ChatBotV2.__new__(ChatBotV2)
    fake = _FakeClient(verdict)
    app.participation_engine = ParticipationEngine(fake)
    app.processor = MagicMock()
    app.processor.db = MagicMock()
    app.processor.db.get_channel_memory_async = AsyncMock(return_value=[])
    app.processor.db.set_channel_settings_async = AsyncMock()
    app.processor.db.add_channel_memory_async = AsyncMock()
    client = MagicMock()
    client.channel_pulse = pulse
    client.react = AsyncMock()
    # F6 addendum: the gate's react verdict routes through _reserve_and_react (guard-aware).
    client._reserve_and_react = AsyncMock(return_value={"ok": True})
    return app, client, fake


def _channel_msg(**meta):
    m = {"ts": "10.0", "participation_check": True, "participation_level": "judicious"}
    m.update(meta)
    return Message(text="anyone know the deploy status?", user_id="U1",
                   channel_id="C1", thread_id="10.0", metadata=m)


class TestGateWiring:
    @pytest.mark.asyncio
    async def test_engine_disabled_stays_silent(self, monkeypatch):
        monkeypatch.setattr(config, "enable_participation_engine", False, raising=False)
        app, client, fake = _make_app({"action": "respond"})
        assert await app._run_participation_gate(_channel_msg(), client) is None
        assert fake.calls == 0

    @pytest.mark.asyncio
    async def test_throttle_skips_engine_call(self, monkeypatch):
        monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
        monkeypatch.setattr(config, "max_unprompted_replies_per_hour", 2, raising=False)
        app, client, fake = _make_app({"action": "respond"}, pulse=_FakePulse(2))
        assert await app._run_participation_gate(_channel_msg(), client) is None
        assert fake.calls == 0  # rail fires BEFORE the model

    @pytest.mark.asyncio
    async def test_name_hit_bypasses_throttle_rail(self, monkeypatch):
        # F14: a name-addressed message must reach the engine even when the hourly
        # runaway brake is at/over cap — only the classifier decides if it's a summons.
        monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        monkeypatch.setattr(config, "max_unprompted_replies_per_hour", 2, raising=False)
        app, client, fake = _make_app({"action": "respond"}, pulse=_FakePulse(2))
        verdict = await app._run_participation_gate(
            _channel_msg(participation_name_hit=True), client)
        assert verdict is not None and verdict.action == "respond"
        assert fake.calls == 1  # rail skipped — the engine still judged

    @pytest.mark.asyncio
    async def test_respond_verdict_passes_through(self, monkeypatch):
        monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        app, client, _ = _make_app({"action": "respond", "placement": "thread"})
        verdict = await app._run_participation_gate(_channel_msg(), client)
        assert verdict is not None and verdict.action == "respond"

    def test_is_unprompted_turn_excludes_name_hit(self):
        # F14: a name-hit respond is prompted in spirit — it must NOT burn the
        # unprompted runaway-brake budget; an ambient participation reply still does.
        from main import ChatBotV2
        assert ChatBotV2._is_unprompted_turn(_channel_msg()) is True
        assert ChatBotV2._is_unprompted_turn(
            _channel_msg(participation_name_hit=True)) is False
        # A non-gated (e.g. @-mention / DM) turn is never unprompted.
        assert ChatBotV2._is_unprompted_turn(
            Message(text="hi", user_id="U1", channel_id="C1", thread_id="10.0",
                    metadata={"ts": "10.0"})) is False

    @pytest.mark.asyncio
    async def test_react_verdict_reacts_and_stays_silent(self, monkeypatch):
        monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        monkeypatch.setattr(config, "reaction_emojis", ["eyes"], raising=False)
        app, client, _ = _make_app({"action": "react", "emoji": "eyes"})
        assert await app._run_participation_gate(_channel_msg(), client) is None
        # F6 addendum: routed through the guard-aware reservation, not the raw react.
        client._reserve_and_react.assert_awaited_once_with("C1", "10.0", "eyes")

    @pytest.mark.asyncio
    async def test_backoff_snoozes_reacts_and_writes_memory(self, monkeypatch):
        monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        monkeypatch.setattr(config, "snooze_ack_emoji", "zipper_mouth_face", raising=False)
        monkeypatch.setattr(config, "enable_channel_memory", True, raising=False)
        app, client, _ = _make_app({"action": "backoff"})
        assert await app._run_participation_gate(_channel_msg(), client) is None
        client.react.assert_awaited_once_with("C1", "10.0", "zipper_mouth_face")
        snooze_call = app.processor.db.set_channel_settings_async.await_args
        assert snooze_call.args[0] == "C1"
        assert snooze_call.kwargs["snoozed_until"]  # a future ISO stamp
        mem_call = app.processor.db.add_channel_memory_async.await_args
        assert "less unprompted participation" in mem_call.args[1]

    @pytest.mark.asyncio
    async def test_gate_exception_is_silent(self, monkeypatch):
        monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
        app, client, _ = _make_app({"action": "respond"})
        app.participation_engine.evaluate = AsyncMock(side_effect=RuntimeError("boom"))
        assert await app._run_participation_gate(_channel_msg(), client) is None


# ------------------------------------------------------------------ placement wiring

class TestPlacement:
    def _app_with_processor(self, response):
        from main import ChatBotV2
        app = ChatBotV2.__new__(ChatBotV2)
        app.participation_engine = None
        app.processor = MagicMock()
        app.processor.process_message = AsyncMock(return_value=response)
        app.processor.thread_manager = MagicMock(spec=[])  # no upload latch attrs
        client = MagicMock()
        client.channel_pulse = None
        client.send_thinking_indicator = AsyncMock(return_value="think.1")
        client.delete_message = AsyncMock()
        client.send_message = AsyncMock()
        client.format_text = lambda t: t
        client.maybe_post_response_footer = AsyncMock()
        return app, client

    def _resp(self, text="answer"):
        r = MagicMock()
        r.type = "text"
        r.content = text
        r.metadata = {}
        return r

    @pytest.mark.asyncio
    async def test_reply_in_channel_setting_posts_top_level_and_skips_footer(self):
        app, client = self._app_with_processor(self._resp())
        msg = Message(text="q", user_id="U1", channel_id="C1", thread_id="10.0",
                      metadata={"ts": "10.0", "reply_in_channel": True})
        await app.handle_message(msg, client)
        client.send_thinking_indicator.assert_awaited_once_with("C1", None)
        assert client.send_message.await_args.args[1] is None  # top-level post
        client.maybe_post_response_footer.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_default_threads_and_footer_posts(self):
        app, client = self._app_with_processor(self._resp())
        msg = Message(text="q", user_id="U1", channel_id="C1", thread_id="10.0",
                      metadata={"ts": "10.0"})
        await app.handle_message(msg, client)
        client.send_thinking_indicator.assert_awaited_once_with("C1", "10.0")
        assert client.send_message.await_args.args[1] == "10.0"
        client.maybe_post_response_footer.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_engine_thread_verdict_overrides_reply_in_channel(self):
        # reply_in_channel is an ALLOWANCE: when the engine judges the answer is
        # worth a thread, the reply threads even with the setting on.
        app, client = self._app_with_processor(self._resp())
        app.participation_engine = MagicMock()
        verdict = ParticipationVerdict(action="respond", emoji="", placement="thread", reason="long answer")
        app._run_participation_gate = AsyncMock(return_value=verdict)
        msg = Message(text="q", user_id="U1", channel_id="C1", thread_id="10.0",
                      metadata={"ts": "10.0", "reply_in_channel": True, "participation_check": True})
        await app.handle_message(msg, client)
        assert client.send_message.await_args.args[1] == "10.0"  # threaded

    @pytest.mark.asyncio
    async def test_engine_channel_verdict_honored_with_setting_on(self):
        app, client = self._app_with_processor(self._resp())
        app.participation_engine = MagicMock()
        verdict = ParticipationVerdict(action="respond", emoji="", placement="channel", reason="quick answer")
        app._run_participation_gate = AsyncMock(return_value=verdict)
        msg = Message(text="q", user_id="U1", channel_id="C1", thread_id="10.0",
                      metadata={"ts": "10.0", "reply_in_channel": True, "participation_check": True})
        await app.handle_message(msg, client)
        assert client.send_message.await_args.args[1] is None  # top-level

    @pytest.mark.asyncio
    async def test_thread_reply_never_moves_top_level_despite_setting(self):
        app, client = self._app_with_processor(self._resp())
        msg = Message(text="q", user_id="U1", channel_id="C1", thread_id="10.0",
                      metadata={"ts": "11.0", "reply_in_channel": True})  # reply inside thread
        await app.handle_message(msg, client)
        assert client.send_message.await_args.args[1] == "10.0"


# ------------------------------------------------------- modal dual-write + DB columns

class TestDBAndModal:
    @pytest.fixture
    def temp_db(self):
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

    def test_new_columns_set_get_preserve_clear(self, temp_db):
        temp_db.set_channel_settings("C1", participation_level="active",
                                     snoozed_until="2026-07-09T20:00:00+00:00")
        row = temp_db.get_channel_settings("C1")
        assert row["participation_level"] == "active"
        assert row["snoozed_until"] == "2026-07-09T20:00:00+00:00"
        # omitted fields preserved
        temp_db.set_channel_settings("C1", directives="rule")
        row = temp_db.get_channel_settings("C1")
        assert row["participation_level"] == "active"
        assert row["snoozed_until"] == "2026-07-09T20:00:00+00:00"
        # explicit None clears
        temp_db.set_channel_settings("C1", participation_level=None, snoozed_until=None)
        row = temp_db.get_channel_settings("C1")
        assert row["participation_level"] is None
        assert row["snoozed_until"] is None

    def test_migration_adds_columns_to_legacy_table(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/legacy.db"
            conn = sqlite3.connect(path)
            conn.execute("""
                CREATE TABLE channel_settings (
                    channel_id TEXT PRIMARY KEY, response_mode TEXT DEFAULT 'tag_only',
                    directives TEXT, reply_in_channel BOOLEAN DEFAULT 0,
                    updated_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_by TEXT)
            """)
            conn.commit()
            conn.close()
            with patch("os.makedirs"):
                db = DatabaseManager("test")
                if getattr(db, "conn", None):
                    db.conn.close()
                db.db_path = path
                db.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
                db.conn.row_factory = sqlite3.Row
                db.init_schema()  # runs migrations
                cols = [c[1] for c in db.conn.execute("PRAGMA table_info(channel_settings)")]
                assert "participation_level" in cols and "snoozed_until" in cols
                db.conn.close()

    def test_modal_participation_select_and_snooze_block(self):
        from settings_modal import SettingsModal
        builder = SettingsModal.__new__(SettingsModal)
        view = builder.build_channel_settings_modal(
            "C1", {"participation_level": "active",
                   "snoozed_until": snooze_expiry_iso(hours=1)}, "tag_only")
        blocks = {b.get("block_id"): b for b in view["blocks"] if b.get("block_id")}
        sel = blocks["participation_block"]["element"]
        assert sel["action_id"] == "participation_level"
        assert sel["initial_option"]["value"] == "active"
        values = [o["value"] for o in sel["options"]]
        assert values == ["inherit", "mentions_only", "judicious", "active", "off"]
        assert "snooze_block" in blocks  # snoozed → early-resume control rendered

    def test_modal_legacy_mode_row_maps_and_no_snooze_block(self):
        from settings_modal import SettingsModal
        builder = SettingsModal.__new__(SettingsModal)
        view = builder.build_channel_settings_modal("C1", {"response_mode": "auto_respond"}, "tag_only")
        blocks = {b.get("block_id"): b for b in view["blocks"] if b.get("block_id")}
        assert blocks["participation_block"]["element"]["initial_option"]["value"] == "judicious"
        assert "snooze_block" not in blocks


# ----------------------------------------------------- busy rejection → needs_refresh

class TestNeedsRefresh:
    def test_mark_and_consume_semantics(self):
        from thread_manager import AsyncThreadStateManager
        mgr = AsyncThreadStateManager.__new__(AsyncThreadStateManager)
        mgr._needs_refresh = set()
        AsyncThreadStateManager.mark_needs_refresh(mgr, "C1:10.0")
        assert AsyncThreadStateManager.consume_needs_refresh(mgr, "C1:10.0") is True
        assert AsyncThreadStateManager.consume_needs_refresh(mgr, "C1:10.0") is False  # cleared
        assert AsyncThreadStateManager.consume_needs_refresh(mgr, "C2:1.0") is False  # cold thread unaffected

    def test_contention_branch_queues_not_rejects(self):
        """Phase Q: lock contention enqueues (no busy rejection); needs_refresh is
        reserved for the loss paths (queue overflow / enqueue failure)."""
        from message_processor import base as mp_base
        src = inspect.getsource(mp_base.MessageProcessor.process_message)
        assert 'type="busy"' not in src
        queued_idx = src.index('type="queued"')
        assert "enqueue_pending" in src[:queued_idx]
        # overflow/failure still flags a transcript refetch
        from thread_manager import AsyncThreadStateManager
        assert "mark_needs_refresh" in inspect.getsource(AsyncThreadStateManager.enqueue_pending)

    def test_rebuild_consumes_refresh_flag(self):
        from message_processor import thread_management as tm
        src = inspect.getsource(tm.ThreadManagementMixin._get_or_rebuild_thread_state)
        assert "consume_needs_refresh" in src
        # flag check must be able to flip should_rebuild for WARM threads
        assert src.index("consume_needs_refresh") < src.index("if should_rebuild")
