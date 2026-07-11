"""Phase F — ParticipationEngine: verdict validation, level/mode mapping, debounce,
uncapped participation (F17: no hourly-cap rail), backoff thread-mute + memory writes
(F15), placement wiring, modal dual-write, DB columns/migration, and the busy-rejection
needs_refresh fix.

All stubbed I/O — no live bot, no legacy suite.
"""
from __future__ import annotations

import asyncio
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
    resolve_participation_level,
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

    # F19: acknowledgment flag
    def test_ack_absent_defaults_false(self):
        assert ParticipationEngine.validate_verdict({"action": "respond"}).ack is False

    def test_ack_true_on_respond(self):
        assert ParticipationEngine.validate_verdict(
            {"action": "respond", "ack": True}).ack is True

    def test_ack_malformed_coerced_false(self):
        # Only a literal True flips it; strings/ints/None never do.
        for bad in ("true", 1, "yes", None, "ack"):
            assert ParticipationEngine.validate_verdict(
                {"action": "respond", "ack": bad}).ack is False

    def test_ack_ignored_on_non_respond_actions(self):
        # ack is meaningful only with respond — react/ignore/backoff never carry it.
        assert ParticipationEngine.validate_verdict(
            {"action": "ignore", "ack": True}).ack is False
        assert ParticipationEngine.validate_verdict(
            {"action": "react", "emoji": "eyes", "ack": True}).ack is False
        assert ParticipationEngine.validate_verdict(
            {"action": "backoff", "ack": True}).ack is False


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

    def test_hourly_cap_rail_removed(self):
        # F17: the hourly-cap hard rail is gone entirely — no hourly_cap/over_throttle
        # methods remain on the engine (pacing is the classifier's judgment now).
        engine = ParticipationEngine(MagicMock())
        assert not hasattr(engine, "hourly_cap")
        assert not hasattr(engine, "over_throttle")


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
    app.processor.db.update_channel_memory_async = AsyncMock()
    app.processor.db.add_muted_thread_async = AsyncMock(return_value=True)
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
    async def test_high_unprompted_count_still_reaches_engine(self, monkeypatch):
        # F17: no hourly-cap rail — even a very high recorded unprompted count never
        # silences a turn before the model sees it. The classifier alone decides.
        monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        app, client, fake = _make_app({"action": "respond"}, pulse=_FakePulse(40))
        verdict = await app._run_participation_gate(_channel_msg(), client)
        assert verdict is not None and verdict.action == "respond"
        assert fake.calls == 1  # engine judged despite 40 unprompted replies on record

    @pytest.mark.asyncio
    async def test_name_hit_still_reaches_engine(self, monkeypatch):
        # F17: a name-addressed message reaches the engine like any other — only the
        # classifier decides if it's a genuine summons.
        monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        app, client, fake = _make_app({"action": "respond"}, pulse=_FakePulse(40))
        verdict = await app._run_participation_gate(
            _channel_msg(participation_name_hit=True), client)
        assert verdict is not None and verdict.action == "respond"
        assert fake.calls == 1

    @pytest.mark.asyncio
    async def test_respond_verdict_passes_through(self, monkeypatch):
        monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        app, client, _ = _make_app({"action": "respond", "placement": "thread"})
        verdict = await app._run_participation_gate(_channel_msg(), client)
        assert verdict is not None and verdict.action == "respond"

    # F19: acknowledgment reaction on respond+ack
    @pytest.mark.asyncio
    async def test_respond_ack_reacts_before_dispatch(self, monkeypatch):
        monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        monkeypatch.setattr(config, "enable_ack_reaction", True, raising=False)
        monkeypatch.setattr(config, "ack_reaction_emoji", "eyes", raising=False)
        app, client, _ = _make_app({"action": "respond", "ack": True})
        verdict = await app._run_participation_gate(_channel_msg(), client)
        assert verdict is not None and verdict.action == "respond" and verdict.ack is True
        # reaction placed on the triggering message through the F6 reservation guard
        client._reserve_and_react.assert_awaited_once_with("C1", "10.0", "eyes")

    @pytest.mark.asyncio
    async def test_respond_without_ack_no_reaction(self, monkeypatch):
        monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        monkeypatch.setattr(config, "enable_ack_reaction", True, raising=False)
        app, client, _ = _make_app({"action": "respond"})
        verdict = await app._run_participation_gate(_channel_msg(), client)
        assert verdict is not None and verdict.action == "respond"
        client._reserve_and_react.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ack_disabled_skips_reaction(self, monkeypatch):
        monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        monkeypatch.setattr(config, "enable_ack_reaction", False, raising=False)
        app, client, _ = _make_app({"action": "respond", "ack": True})
        verdict = await app._run_participation_gate(_channel_msg(), client)
        assert verdict is not None and verdict.action == "respond"
        client._reserve_and_react.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ack_emoji_configurable(self, monkeypatch):
        monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        monkeypatch.setattr(config, "enable_ack_reaction", True, raising=False)
        monkeypatch.setattr(config, "ack_reaction_emoji", "hourglass_flowing_sand", raising=False)
        app, client, _ = _make_app({"action": "respond", "ack": True})
        await app._run_participation_gate(_channel_msg(), client)
        client._reserve_and_react.assert_awaited_once_with("C1", "10.0", "hourglass_flowing_sand")

    @pytest.mark.asyncio
    async def test_react_verdict_ignores_ack_field(self, monkeypatch):
        # A react verdict never acks even if the model leaks an ack field — only the
        # react emoji is placed, once.
        monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        monkeypatch.setattr(config, "enable_ack_reaction", True, raising=False)
        monkeypatch.setattr(config, "ack_reaction_emoji", "eyes", raising=False)
        monkeypatch.setattr(config, "reaction_emojis", ["thumbsup"], raising=False)
        app, client, _ = _make_app({"action": "react", "emoji": "thumbsup", "ack": True})
        assert await app._run_participation_gate(_channel_msg(), client) is None
        client._reserve_and_react.assert_awaited_once_with("C1", "10.0", "thumbsup")

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
    async def test_backoff_mutes_thread_reacts_and_writes_memory(self, monkeypatch):
        # F15: backoff acks with the emoji, MUTES THE THREAD (not a channel-wide timer),
        # and writes a dated butt-out memory fact. snoozed_until is never written.
        monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        monkeypatch.setattr(config, "snooze_ack_emoji", "zipper_mouth_face", raising=False)
        monkeypatch.setattr(config, "enable_channel_memory", True, raising=False)
        app, client, _ = _make_app({"action": "backoff"})
        assert await app._run_participation_gate(_channel_msg(), client) is None
        client.react.assert_awaited_once_with("C1", "10.0", "zipper_mouth_face")
        # thread muted (DB-backed), keyed by the thread root ts
        mute_call = app.processor.db.add_muted_thread_async.await_args
        assert mute_call.args[0] == "C1" and mute_call.args[1] == "10.0"
        # no snooze timer written
        app.processor.db.set_channel_settings_async.assert_not_awaited()
        # dated butt-out fact, authored with the thread marker (for dedup on repeat)
        mem_call = app.processor.db.add_channel_memory_async.await_args
        assert "butt out" in mem_call.args[1]
        assert "raise the bar for unprompted replies" in mem_call.args[1]
        assert mem_call.kwargs["author"] == "participation_engine:10.0"

    @pytest.mark.asyncio
    async def test_backoff_updates_existing_memory_fact_not_duplicate(self, monkeypatch):
        # F15: a repeat butt-out on the SAME thread UPDATES the prior fact (matched by its
        # participation_engine:<thread> author) instead of adding a duplicate row.
        monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        monkeypatch.setattr(config, "enable_channel_memory", True, raising=False)
        app, client, _ = _make_app({"action": "backoff"})
        app.processor.db.get_channel_memory_async = AsyncMock(return_value=[
            {"id": 7, "author": "participation_engine:10.0", "content": "old fact"},
        ])
        assert await app._run_participation_gate(_channel_msg(), client) is None
        app.processor.db.update_channel_memory_async.assert_awaited_once()
        assert app.processor.db.update_channel_memory_async.await_args.args[0] == 7
        app.processor.db.add_channel_memory_async.assert_not_awaited()

    def test_participation_prompt_has_butt_out_memory_line(self):
        # F15: the classifier learns about butt-out feedback through channel memory —
        # the prompt must tell it how to weigh recorded/repeated butt-out facts.
        from prompts import PARTICIPATION_SYSTEM_PROMPT
        assert "butt-out feedback in the channel memory" in PARTICIPATION_SYSTEM_PROMPT
        assert "observe-only" in PARTICIPATION_SYSTEM_PROMPT

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

    def test_muted_threads_set_get_preserve_clear(self, temp_db):
        # F15: muted_threads round-trips as a Python list (stored as JSON), is preserved
        # when omitted, and clears on None/[].
        temp_db.set_channel_settings("C1", muted_threads=["10.0", "20.5"])
        assert temp_db.get_channel_settings("C1")["muted_threads"] == ["10.0", "20.5"]
        temp_db.set_channel_settings("C1", directives="rule")  # omitted → preserved
        assert temp_db.get_channel_settings("C1")["muted_threads"] == ["10.0", "20.5"]
        temp_db.set_channel_settings("C1", muted_threads=None)  # cleared
        assert temp_db.get_channel_settings("C1")["muted_threads"] == []
        # no row → empty list, never a crash
        assert temp_db.get_channel_settings("C2") is None

    def test_add_muted_thread_persists_and_dedupes(self, temp_db):
        # F15: mute is DB-backed (survives restart — a fresh get sees it) and idempotent.
        assert asyncio.run(temp_db.add_muted_thread_async("C1", "10.0")) is True
        assert asyncio.run(temp_db.add_muted_thread_async("C1", "10.0")) is False  # dedup
        assert asyncio.run(temp_db.add_muted_thread_async("C1", "30.0")) is True
        # read back through a fresh async connection (simulates restart — state is durable)
        row = asyncio.run(temp_db.get_channel_settings_async("C1"))
        assert row["muted_threads"] == ["10.0", "30.0"]

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
                assert "muted_threads" in cols  # F15 migration
                db.conn.close()

    def test_modal_participation_select_no_snooze_block(self):
        # F15: the snooze early-resume control is retired — the modal never renders it.
        from settings_modal import SettingsModal
        builder = SettingsModal.__new__(SettingsModal)
        view = builder.build_channel_settings_modal(
            "C1", {"participation_level": "active"}, "tag_only")
        blocks = {b.get("block_id"): b for b in view["blocks"] if b.get("block_id")}
        sel = blocks["participation_block"]["element"]
        assert sel["action_id"] == "participation_level"
        assert sel["initial_option"]["value"] == "active"
        values = [o["value"] for o in sel["options"]]
        assert values == ["inherit", "mentions_only", "judicious", "active", "off"]
        assert "snooze_block" not in blocks

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
