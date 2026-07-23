"""Phase F — ParticipationEngine: verdict validation, level/mode mapping, debounce,
uncapped participation (F17: no hourly-cap rail), backoff pref-memory writes (thread-scope
now persists nothing; the mute mechanism was removed), placement wiring, modal dual-write,
DB columns/migration, and the busy-rejection needs_refresh fix.

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
    _MAX_PENDING_KEYS, LEVEL_TO_MODE, MODE_TO_LEVEL, ParticipationEngine,
    ParticipationVerdict, resolve_participation_level,
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

    def test_react_and_respond_keeps_valid_emoji_and_placement(self, monkeypatch):
        # react_and_respond reacts AND replies in one turn: a valid emoji is kept and placement
        # is coerced exactly like a respond verdict.
        monkeypatch.setattr(config, "reaction_emojis", [], raising=False)
        v = ParticipationEngine.validate_verdict(
            {"action": "react_and_respond", "emoji": ":tada:", "placement": "channel"})
        assert (v.action, v.emoji, v.placement) == ("react_and_respond", "tada", "channel")

    def test_react_and_respond_no_allowlist_invalid_emoji_downgrades_to_respond(self, monkeypatch):
        # (a) With NO allowlist, an unresolvable emoji drops the reaction but KEEPS the reply:
        # downgrade to a plain respond, NEVER to ignore (react→None ignores; this must not).
        monkeypatch.setattr(config, "reaction_emojis", [], raising=False)
        v = ParticipationEngine.validate_verdict(
            {"action": "react_and_respond", "emoji": "bad name!"})
        assert v.action == "respond" and v.emoji is None

    def test_react_and_respond_allowlist_offlist_falls_back_and_stays(self, monkeypatch):
        # (b) With an allowlist set, an off-list emoji falls back to the first allowed emoji and the
        # action STAYS react_and_respond (the emoji resolved, so there is no downgrade).
        monkeypatch.setattr(config, "reaction_emojis", ["thumbsup", "eyes"], raising=False)
        v = ParticipationEngine.validate_verdict(
            {"action": "react_and_respond", "emoji": "middle_finger"})
        assert v.action == "react_and_respond" and v.emoji == "thumbsup"

    def test_bad_placement_coerced_to_thread(self):
        v = ParticipationEngine.validate_verdict({"action": "respond", "placement": "everywhere"})
        assert v.placement == "thread"

    def test_reason_truncated(self):
        v = ParticipationEngine.validate_verdict({"action": "ignore", "reason": "x" * 999})
        assert len(v.reason) == 300

    # F38: the classifier's `ack` bit is GONE — it predicted "real work ahead" before the
    # model had done anything, and the gate dropped a 👀 on that guess. The verdict must
    # carry no such field, and a stale `ack` key from an old prompt must be inert.
    def test_verdict_has_no_ack_field(self):
        assert not hasattr(ParticipationEngine.validate_verdict({"action": "respond"}), "ack")

    def test_stale_ack_key_is_ignored(self):
        v = ParticipationEngine.validate_verdict({"action": "respond", "ack": True})
        assert v.action == "respond"
        assert not hasattr(v, "ack")


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
    async def test_thread_message_survives_newer_message_in_other_thread(self, monkeypatch):
        """F21: supersession is conversation-scoped. A pending evaluation in thread A
        must NOT be dropped because thread B (or another conversation) posted something
        newer in the same channel during the debounce window."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
        fake = _FakeClient({"action": "respond"})
        engine = ParticipationEngine(fake)
        a = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="10.5", text="question in thread A", thread_root_ts="10.0"))
        await asyncio.sleep(0.01)
        b = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="20.5", text="chatter in thread B", thread_root_ts="20.0"))
        ra, rb = await asyncio.gather(a, b)
        assert ra is not None and ra.action == "respond"   # thread A still judged
        assert rb is not None and rb.action == "respond"
        assert fake.calls == 2                             # both conversations evaluated

    @pytest.mark.asyncio
    async def test_thread_message_survives_newer_top_level(self, monkeypatch):
        """F21: a newer TOP-LEVEL message must not supersede a pending thread reply."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
        fake = _FakeClient({"action": "respond"})
        engine = ParticipationEngine(fake)
        a = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="10.5", text="thread question", thread_root_ts="10.0"))
        await asyncio.sleep(0.01)
        b = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="30.0", text="unrelated top-level"))  # roots key as |top
        ra, rb = await asyncio.gather(a, b)
        assert ra is not None and rb is not None
        assert fake.calls == 2

    @pytest.mark.asyncio
    async def test_same_thread_burst_still_collapses(self, monkeypatch):
        """F21: within ONE thread the old behavior holds — the newest message of a
        rapid burst supersedes the older ones (its tail covers the batch)."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
        fake = _FakeClient({"action": "respond"})
        engine = ParticipationEngine(fake)
        first = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="10.5", text="line one", thread_root_ts="10.0"))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="10.6", text="line two", thread_root_ts="10.0"))
        r1, r2 = await asyncio.gather(first, second)
        assert r1 is None            # superseded within the conversation
        assert r2.action == "respond"
        assert fake.calls == 1

    def test_conv_key_root_vs_reply(self):
        """A thread ROOT keys as top-level (thread_root == ts); its replies key by root.
        F27: top-level keys are per-sender; a thread reply key ignores sender."""
        assert ParticipationEngine._conv_key("C1", "10.0", "10.0", "U1") == "C1|top|U1"
        assert ParticipationEngine._conv_key("C1", "10.5", "10.0", "U1") == "C1|10.0"
        assert ParticipationEngine._conv_key("C1", "30.0", None, "U2") == "C1|top|U2"
        # no sender_id → "unknown" (back-compat default)
        assert ParticipationEngine._conv_key("C1", "30.0", None) == "C1|top|unknown"

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


# ---------------------------------------------------------- F27 same-author burst carry


class _CapturingClient:
    """Records the signals of the LAST classify call (survivors only reach here)."""
    def __init__(self, verdict):
        self._verdict = verdict
        self.calls = 0
        self.signals = None

    async def classify_participation(self, text, signals=None):
        self.calls += 1
        self.signals = signals
        return self._verdict


class TestBurstCarryForward:
    @pytest.mark.asyncio
    async def test_different_authors_top_level_both_survive(self, monkeypatch):
        """F27: two DIFFERENT users' unrelated top-level messages within the debounce no
        longer collapse — each is the newest in its own per-sender stream, both answered."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
        fake = _FakeClient({"action": "respond"})
        engine = ParticipationEngine(fake)
        a = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="1.0", text="alice question", sender_id="U1"))
        await asyncio.sleep(0.01)
        b = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="2.0", text="bob question", sender_id="U2"))
        ra, rb = await asyncio.gather(a, b)
        assert ra is not None and ra.action == "respond"
        assert rb is not None and rb.action == "respond"
        assert fake.calls == 2                       # both evaluated independently
        assert ra.burst_earlier is None and rb.burst_earlier is None

    @pytest.mark.asyncio
    async def test_same_author_top_level_burst_carries_earlier(self, monkeypatch):
        """F27: a same-author fast-follow supersedes the first message, but the survivor
        carries the earlier text so ONE reply covers the whole burst."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
        fake = _FakeClient({"action": "respond"})
        engine = ParticipationEngine(fake)
        first = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="1.0", text="first thought", sender_id="U1"))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="2.0", text="actually also this", sender_id="U1"))
        r1, r2 = await asyncio.gather(first, second)
        assert r1 is None                            # superseded — silent
        assert r2.action == "respond"
        assert r2.burst_earlier == ["first thought"]
        assert fake.calls == 1                        # ONE reply for the burst
        # pending bucket drained after the survivor collected it
        assert not engine._pending.get("C1|top|U1")

    @pytest.mark.asyncio
    async def test_burst_signal_reaches_classifier(self, monkeypatch):
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
        cap = _CapturingClient({"action": "respond"})
        engine = ParticipationEngine(cap)
        first = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="1.0", text="one", sender_id="U1"))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="2.0", text="two", sender_id="U1"))
        await asyncio.gather(first, second)
        assert cap.signals["burst_earlier"] == ["one"]

    @pytest.mark.asyncio
    async def test_channel_people_signal_reaches_classifier(self, monkeypatch):
        # F29: the people summary passed to evaluate is forwarded in the classifier signals.
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.0, raising=False)
        cap = _CapturingClient({"action": "ignore"})
        engine = ParticipationEngine(cap)
        await engine.evaluate(channel_id="C1", ts="1.0", text="hi", sender_id="U1",
                              channel_people="~5 members; recently active: Alice")
        assert cap.signals["channel_people"] == "~5 members; recently active: Alice"

    def test_burst_of_five_keeps_newest_three(self):
        """F27: a same-author burst of 5 carries only the newest 3 earlier messages."""
        eng = ParticipationEngine(MagicMock())
        key = "C1|top|U1"
        for i in range(1, 6):
            eng._register_pending(key, f"{i}.0", f"m{i}")
        eng._latest[key] = "5.0"                     # 5.0 is the survivor
        carried = eng._collect_burst(key, "5.0", 0.05)
        assert carried == ["m2", "m3", "m4"]         # newest 3 strictly-older, oldest-first
        assert key not in eng._pending               # bucket drained

    def test_stale_pending_entry_not_carried(self):
        """F27: an entry far older than the survivor (a leftover from a crashed evaluation)
        is dropped, never leaked into a fresh burst minutes later."""
        eng = ParticipationEngine(MagicMock())
        key = "C1|top|U1"
        eng._register_pending(key, "100.0", "ancient")    # >15s before survivor → stale
        eng._register_pending(key, "1000.0", "recent")    # within freshness window
        eng._register_pending(key, "1001.0", "survivor")
        eng._latest[key] = "1001.0"
        carried = eng._collect_burst(key, "1001.0", 0.05)  # window = max(15, 0.25) = 15s
        assert carried == ["recent"]
        assert key not in eng._pending

    def test_pending_map_is_bounded(self):
        """F27: the pending map can't grow unbounded over the process lifetime."""
        eng = ParticipationEngine(MagicMock())
        for i in range(_MAX_PENDING_KEYS + 50):
            eng._register_pending(f"C1|top|U{i}", "1.0", "x")
        assert len(eng._pending) <= _MAX_PENDING_KEYS

    @pytest.mark.asyncio
    async def test_thread_burst_still_collapses_cross_author(self, monkeypatch):
        """F27 leaves F21 thread behavior intact: within one thread a cross-author burst
        still collapses to the newest (its in-thread reply has full history)."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
        fake = _FakeClient({"action": "respond"})
        engine = ParticipationEngine(fake)
        first = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="10.5", text="alice in thread",
            sender_id="U1", thread_root_ts="10.0"))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="10.6", text="bob in thread",
            sender_id="U2", thread_root_ts="10.0"))   # different author, SAME thread
        r1, r2 = await asyncio.gather(first, second)
        assert r1 is None                              # still collapses in-thread
        assert r2.action == "respond"
        assert fake.calls == 1

    @pytest.mark.asyncio
    async def test_thread_cross_author_burst_not_carried(self, monkeypatch):
        """F27: an in-thread cross-author burst collapses (F21) but the survivor must NOT
        carry the superseded — possibly different-author — text: the render sites label it
        "the same sender", so carrying it would misattribute. Carry is top-level-only."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
        cap = _CapturingClient({"action": "respond"})
        engine = ParticipationEngine(cap)
        first = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="10.5", text="alice's words",
            sender_id="U1", thread_root_ts="10.0"))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="10.6", text="bob's reply",
            sender_id="U2", thread_root_ts="10.0"))   # different author, same thread
        r1, r2 = await asyncio.gather(first, second)
        assert r1 is None
        assert r2.action == "respond"
        assert not r2.burst_earlier                    # None/empty — no cross-author carry
        assert cap.signals["burst_earlier"] == []      # classifier sees no burst texts

    @pytest.mark.asyncio
    async def test_thread_survivor_drains_bucket_despite_discard(self, monkeypatch):
        """F27: a thread survivor discards the carry but must STILL drain its pending bucket
        (memory hygiene) so a busy thread's bucket can't grow unbounded."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
        fake = _FakeClient({"action": "respond"})
        engine = ParticipationEngine(fake)
        first = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="10.5", text="one",
            sender_id="U1", thread_root_ts="10.0"))
        await asyncio.sleep(0.01)
        second = asyncio.create_task(engine.evaluate(
            channel_id="C1", ts="10.6", text="two",
            sender_id="U1", thread_root_ts="10.0"))
        await asyncio.gather(first, second)
        assert not engine._pending.get("C1|10.0")      # bucket emptied


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
    app.processor.db.delete_channel_memory_async = AsyncMock()
    # Redesign SHOULD-FIX #8: the pref add/refresh path routes through the atomic marker upsert.
    app.processor.db.upsert_channel_pref_memory = AsyncMock(return_value=7)
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

    # F38: the gate NEVER reacts on a respond verdict. The 👀 is a claim on work, staked by
    # the work itself once a slow tool really starts — not a prediction the classifier makes
    # before the model has looked at anything. A gate that acks a passing comment and then
    # says nothing is exactly the misleading behavior this removed.
    @pytest.mark.asyncio
    async def test_respond_never_reacts(self, monkeypatch):
        monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        monkeypatch.setattr(config, "enable_ack_reaction", True, raising=False)
        monkeypatch.setattr(config, "ack_reaction_emoji", "eyes", raising=False)
        app, client, _ = _make_app({"action": "respond"})
        verdict = await app._run_participation_gate(_channel_msg(), client)
        assert verdict is not None and verdict.action == "respond"
        client._reserve_and_react.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_respond_never_reacts_even_with_a_stale_ack_bit(self, monkeypatch):
        # An old prompt (or a model reciting the old contract) can still emit "ack": true.
        # It must be inert — no reaction, no crash.
        monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        monkeypatch.setattr(config, "enable_ack_reaction", True, raising=False)
        monkeypatch.setattr(config, "ack_reaction_emoji", "eyes", raising=False)
        app, client, _ = _make_app({"action": "respond", "ack": True})
        verdict = await app._run_participation_gate(_channel_msg(), client)
        assert verdict is not None and verdict.action == "respond"
        client._reserve_and_react.assert_not_awaited()

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
    async def test_react_and_respond_reacts_and_falls_through(self, monkeypatch):
        # react_and_respond places the gate reaction AND returns the verdict so the response loop
        # runs — and it stamps the emoji so the response turn's suffix can tell the model.
        monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        monkeypatch.setattr(config, "reaction_emojis", ["tada"], raising=False)
        app, client, _ = _make_app({"action": "react_and_respond", "emoji": "tada"})
        msg = _channel_msg()
        verdict = await app._run_participation_gate(msg, client)
        assert verdict is not None and verdict.action == "react_and_respond"
        client._reserve_and_react.assert_awaited_once_with("C1", "10.0", "tada")
        assert msg.metadata["participation_reaction_emoji"] == "tada"

    @pytest.mark.asyncio
    async def test_queue_redispatch_react_and_respond_twice_places_one(self, monkeypatch):
        # Queue dedup (i): the SAME Message object is re-run through the gate on redispatch, and the
        # fresh pass picks a DIFFERENT emoji but the same react_and_respond verdict. The stamp from
        # the first placement makes the second pass a no-op — exactly ONE reaction total.
        monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        monkeypatch.setattr(config, "reaction_emojis", ["tada", "fire"], raising=False)
        app, client, fake = _make_app({"action": "react_and_respond", "emoji": "tada"})
        msg = _channel_msg()
        v1 = await app._run_participation_gate(msg, client)
        assert v1 is not None and v1.action == "react_and_respond"
        fake._verdict = {"action": "react_and_respond", "emoji": "fire"}  # redispatch, new emoji
        v2 = await app._run_participation_gate(msg, client)
        assert v2 is not None and v2.action == "react_and_respond"
        assert client._reserve_and_react.await_count == 1
        client._reserve_and_react.assert_awaited_with("C1", "10.0", "tada")
        assert msg.metadata["participation_reaction_emoji"] == "tada"

    @pytest.mark.asyncio
    async def test_queue_redispatch_react_and_respond_then_react_places_one(self, monkeypatch):
        # Queue dedup (ii): react_and_respond on the first pass, then a redispatch flips to a PLAIN
        # react with a different emoji. Both branches route through the shared helper, so the stamp
        # still blocks the second reaction — one reaction total, and the plain react is terminal.
        monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        monkeypatch.setattr(config, "reaction_emojis", ["tada", "fire"], raising=False)
        app, client, fake = _make_app({"action": "react_and_respond", "emoji": "tada"})
        msg = _channel_msg()
        v1 = await app._run_participation_gate(msg, client)
        assert v1 is not None and v1.action == "react_and_respond"
        fake._verdict = {"action": "react", "emoji": "fire"}  # redispatch flips to plain react
        v2 = await app._run_participation_gate(msg, client)
        assert v2 is None  # a plain react is terminal
        assert client._reserve_and_react.await_count == 1
        client._reserve_and_react.assert_awaited_with("C1", "10.0", "tada")

    @pytest.mark.asyncio
    async def test_backoff_thread_exit_persists_nothing_via_gate(self, monkeypatch):
        # Redesign: an explicit "stay out of THIS thread" (standing, thread-scoped) backoff routes
        # through the gate into _apply_backoff. The per-thread mute mechanism was removed, so it is
        # guidance for the current message only — it writes NOTHING durable (no structural
        # channel_settings — the clobber this redesign fixes — no channel memory, no marker upsert).
        # The gate stays silent (returns None).
        monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        monkeypatch.setattr(config, "enable_channel_memory", True, raising=False)
        app, client, _ = _make_app({
            "action": "backoff", "durability": "standing", "scope": "thread",
            "dimension": "thread_participation", "guidance": "stay out of this thread",
            "memory_op": "add"})
        assert await app._run_participation_gate(_channel_msg(), client) is None
        app.processor.db.set_channel_settings_async.assert_not_awaited()
        app.processor.db.add_channel_memory_async.assert_not_awaited()
        app.processor.db.upsert_channel_pref_memory.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_backoff_standing_channel_pref_updates_marker_via_gate(self, monkeypatch):
        # Redesign: a standing, channel-scoped soft preference records ONE per-channel/
        # per-dimension memory keyed by the stable marker author `participation_engine:pref:
        # <dimension>`, written through the atomic upsert so a repeat converges on that single
        # row instead of piling up duplicates (the false "REPEATED = observe-only" escalation is
        # gone). No mute, no structural write, and never the raw add.
        monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        monkeypatch.setattr(config, "enable_channel_memory", True, raising=False)
        app, client, _ = _make_app({
            "action": "backoff", "durability": "standing", "scope": "channel",
            "dimension": "reactions", "guidance": "react less here", "memory_op": "add"})
        assert await app._run_participation_gate(_channel_msg(), client) is None
        app.processor.db.upsert_channel_pref_memory.assert_awaited_once()
        args = app.processor.db.upsert_channel_pref_memory.await_args
        assert args.args[1] == "participation_engine:pref:reactions"   # stable marker author
        assert "react less" in args.args[2]                           # normalized content
        app.processor.db.add_channel_memory_async.assert_not_awaited()
        app.processor.db.set_channel_settings_async.assert_not_awaited()

    def test_participation_prompt_has_butt_out_memory_line(self):
        # F15: the classifier learns about butt-out feedback through channel memory —
        # the prompt must tell it how to weigh recorded/repeated butt-out facts.
        from prompts import PARTICIPATION_SYSTEM_PROMPT
        assert "butt-out feedback in the channel memory" in PARTICIPATION_SYSTEM_PROMPT
        assert "observe-only" in PARTICIPATION_SYSTEM_PROMPT

    def test_participation_prompt_documents_burst_signal(self):
        # F27: the prompt must tell the classifier to judge a same-author burst as ONE
        # combined request.
        from prompts import PARTICIPATION_SYSTEM_PROMPT
        assert "Same-author burst" in PARTICIPATION_SYSTEM_PROMPT
        assert "ONE combined request" in PARTICIPATION_SYSTEM_PROMPT

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
        """F39: a top-level channel reply gets NO placeholder — it is posted once, finished.

        It used to get one, and that was the "(edited)" bug: Slack can only stream into a
        thread (chat.startStream REQUIRES thread_ts), so a top-level placeholder could only
        become the answer by being chat.update-d — which brands the message "(edited)" forever.
        A human teammate posts once; so do we.
        """
        app, client = self._app_with_processor(self._resp())
        msg = Message(text="q", user_id="U1", channel_id="C1", thread_id="10.0",
                      metadata={"ts": "10.0", "reply_in_channel": True})
        await app.handle_message(msg, client)
        client.send_thinking_indicator.assert_not_awaited()
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

    @pytest.mark.asyncio
    async def test_respond_verdict_burst_earlier_lands_in_metadata(self):
        # F27: a respond verdict carrying earlier same-author burst messages stamps them
        # onto message.metadata so the wake envelope can tell the reply to cover them all.
        app, client = self._app_with_processor(self._resp())
        app.participation_engine = MagicMock()
        verdict = ParticipationVerdict(
            action="respond", emoji="", placement="thread", reason="combined ask",
            burst_earlier=["first bit", "second bit"])
        app._run_participation_gate = AsyncMock(return_value=verdict)
        msg = Message(text="q", user_id="U1", channel_id="C1", thread_id="10.0",
                      metadata={"ts": "10.0", "participation_check": True})
        await app.handle_message(msg, client)
        assert msg.metadata["participation_burst_earlier"] == ["first bit", "second bit"]
        assert msg.metadata["participation_reason"] == "combined ask"

    @pytest.mark.asyncio
    async def test_no_burst_earlier_leaves_metadata_unset(self):
        app, client = self._app_with_processor(self._resp())
        app.participation_engine = MagicMock()
        verdict = ParticipationVerdict(action="respond", emoji="", placement="thread",
                                       reason="plain")
        app._run_participation_gate = AsyncMock(return_value=verdict)
        msg = Message(text="q", user_id="U1", channel_id="C1", thread_id="10.0",
                      metadata={"ts": "10.0", "participation_check": True})
        await app.handle_message(msg, client)
        assert "participation_burst_earlier" not in msg.metadata


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
