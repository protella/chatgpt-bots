"""Phase E — ChannelPulse ambient awareness + response envelope.

Covers the ring buffer, once-only backfill, DM exclusion, deterministic envelope
rendering with current-thread exclusion, suffix (not system prompt) placement,
participation stats, feed-before-gates wiring, and flag gating.
"""
import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from slack_client.channel_pulse import ChannelPulse
from config import config


def _entry(ts, text="hello", thread_ts=None, name="Alice", sender="human"):
    return dict(ts=ts, thread_ts=thread_ts, user_id="U1", display_name=name,
                sender_type=sender, text=text, is_bot=sender != "human")


# ------------------------------------------------------------------ ring core

def test_ring_eviction_keeps_newest():
    p = ChannelPulse(size=3)
    for i in range(5):
        p.record("C1", **_entry(f"{i}.0", text=f"m{i}"))
    env = p.render_envelope("C1")
    assert "m2" in env and "m3" in env and "m4" in env
    assert "m0" not in env and "m1" not in env


def test_dm_and_malformed_excluded():
    p = ChannelPulse(size=5)
    p.record("D123", **_entry("1.0"))          # DM
    p.record(None, **_entry("2.0"))            # no channel
    p.record("C1", **_entry(None))             # no ts
    assert p.render_envelope("D123") == ""
    assert p.render_envelope("C1") == ""


def test_own_and_ignored_messages_still_recorded():
    # The buffer takes everything it's fed — gating happens in the event handler,
    # which feeds BEFORE the own-message/mode gates (see wiring test below).
    p = ChannelPulse(size=5)
    p.record("C1", **_entry("1.0", text="bot said", name="ChatGPT", sender="self"))
    p.record("C1", **_entry("2.0", text="claude said", name="Claude", sender="other_bot"))
    env = p.render_envelope("C1")
    assert "bot said" in env and "claude said" in env


# ------------------------------------------------------------------- backfill

@pytest.mark.asyncio
async def test_backfill_once_per_channel():
    p = ChannelPulse(size=10)
    client = MagicMock()
    client.conversations_history = AsyncMock(return_value={"messages": [
        {"ts": "2.0", "text": "second", "user": "U2"},
        {"ts": "1.0", "text": "first", "user": "U1"},  # Slack returns newest first
    ]})
    bot = MagicMock()
    bot.classify_sender = MagicMock(return_value="human")
    bot.user_cache = {}

    await p.ensure_backfill("C1", client, bot)
    await p.ensure_backfill("C1", client, bot)  # second call must not re-fetch
    assert client.conversations_history.await_count == 1

    env = p.render_envelope("C1")
    # oldest -> newest ordering after backfill
    assert env.index("first") < env.index("second")


@pytest.mark.asyncio
async def test_backfill_failure_is_silent_and_nonfatal():
    p = ChannelPulse(size=10)
    client = MagicMock()
    client.conversations_history = AsyncMock(side_effect=RuntimeError("boom"))
    await p.ensure_backfill("C1", client, MagicMock())
    assert p.render_envelope("C1") == ""  # empty, no raise


@pytest.mark.asyncio
async def test_backfill_skips_dms():
    p = ChannelPulse(size=10)
    client = MagicMock()
    client.conversations_history = AsyncMock()
    await p.ensure_backfill("D999", client, MagicMock())
    client.conversations_history.assert_not_awaited()


# ------------------------------------------------------------------- envelope

def _seeded_pulse():
    p = ChannelPulse(size=10)
    p.record("C1", **_entry("100.0", text="deploy question here"))
    p.record("C1", **_entry("101.0", text="reply inside deploy thread",
                            thread_ts="100.0", name="Bob"))
    p.record("C1", **_entry("102.0", text="unrelated top level", name="Cara"))
    return p


def test_envelope_thread_vs_top_level_lines():
    env = _seeded_pulse().render_envelope("C1")
    assert '- Alice (top-level): deploy question here' in env
    assert '- Bob (in thread "deploy question here…"): reply inside deploy thread' in env
    assert '- Cara (top-level): unrelated top level' in env


def test_envelope_excludes_current_thread():
    env = _seeded_pulse().render_envelope("C1", exclude_thread_ts="100.0")
    # the thread root AND its replies are the model's full context already
    assert "deploy question" not in env and "reply inside" not in env
    assert "unrelated top level" in env


def test_envelope_cap_and_zero():
    p = ChannelPulse(size=10)
    for i in range(8):
        p.record("C1", **_entry(f"{i}.0", text=f"m{i}"))
    capped = p.render_envelope("C1", max_lines=3)
    assert len([l for l in capped.splitlines() if l.startswith("- ")]) == 3
    assert "m7" in capped and "m0" not in capped  # newest kept
    assert p.render_envelope("C1", max_lines=0) == ""


def test_envelope_deterministic_given_same_state():
    a, b = _seeded_pulse(), _seeded_pulse()
    assert a.render_envelope("C1") == b.render_envelope("C1")
    assert a.render_envelope("C1") == a.render_envelope("C1")


# --------------------------------------------------------- participation stats

def test_participation_stat_math():
    p = ChannelPulse()
    now = 10_000.0
    p.record_bot_reply("C1", "1.0", unprompted=True, now=now - 3599)   # inside window
    p.record_bot_reply("C1", "2.0", unprompted=True, now=now - 3601)   # outside
    p.record_bot_reply("C1", "3.0", unprompted=False, now=now)         # prompted: not counted
    p.record_bot_reply("D1", "4.0", unprompted=True, now=now)          # DM: not counted
    assert p.unprompted_count_last_hour("C1", now=now) == 1
    assert p.unprompted_count_last_hour("C2", now=now) == 0


# ------------------------------------------------- idempotence / dedup (F5)

def test_record_idempotent_by_ts():
    # Dual delivery (app_mention + message) / retries: same ts recorded once.
    p = ChannelPulse(size=10)
    p.record("C1", **_entry("1.0", text="hi"))
    p.record("C1", **_entry("1.0", text="hi"))
    assert len(p._buffers["C1"]) == 1


def test_seen_ts_outer_map_bounded(monkeypatch):
    # The OUTER channel map must be a bounded whole-channel LRU (no unbounded growth).
    monkeypatch.setattr(config, "pulse_thread_tail_channels_max", 3)
    p = ChannelPulse(size=5)
    for i in range(10):
        p.record(f"C{i}", **_entry("1.0", text="hi"))
    assert len(p._seen_ts) <= 3
    assert "C9" in p._seen_ts          # newest channels retained
    assert "C0" not in p._seen_ts      # oldest evicted


def test_seen_ts_resurrection_guard_uses_live_ring():
    # A ts that aged out of the dedup window but is STILL in the live ring must be
    # treated as already-recorded — a delayed retry can't resurrect it (F5).
    p = ChannelPulse(size=30)
    p.record("C1", **_entry("100.1", text="original"))
    assert len(p._buffers["C1"]) == 1
    # Simulate the ts falling out of the bounded _seen_ts window while it lives on in
    # the buffer (and the thread-tail ring).
    p._seen_ts["C1"].clear()
    p.record("C1", **_entry("100.1", text="original"))   # delayed retry
    assert len(p._buffers["C1"]) == 1                     # NOT re-appended
    # A genuinely new ts (not in any ring) still records normally.
    p.record("C1", **_entry("200.2", text="new"))
    assert len(p._buffers["C1"]) == 2


# ------------------------------------------------------------------ wiring

def _mixin_host(pulse):
    """Minimal object binding the real _feed_channel_pulse."""
    from slack_client.event_handlers.message_events import SlackMessageEventsMixin

    class Host(SlackMessageEventsMixin):
        def __init__(self):
            self.channel_pulse = pulse
            self.user_cache = {"U7": {"real_name": "Peter"}}
            self.bot_user_id = "UBOT"

        def is_own_message(self, e):
            return e.get("user") == "UBOT"

        def classify_sender(self, e):
            return "self" if e.get("user") == "UBOT" else "human"

        def log_debug(self, *a, **k): pass

    return Host()


@pytest.mark.asyncio
async def test_feed_skips_own_records_others():
    # F5 fix (a): the event feed skips OUR OWN posts (echoed placeholders/footers/edits
    # are chrome) — the bot's own final reply is recorded cleanly at the messaging layer.
    # Other senders (incl. bot_message subtype) are recorded.
    p = ChannelPulse(size=5)
    host = _mixin_host(p)
    await host._feed_channel_pulse({"channel": "C1", "ts": "1.0", "user": "UBOT",
                                    "text": "my own post"})
    await host._feed_channel_pulse({"channel": "C1", "ts": "2.0", "user": "U7",
                                    "text": "human post"})
    env = p.render_envelope("C1")
    assert "human post" in env and "Peter" in env
    assert "my own post" not in env
    # ...but the bot's own reply DOES enter the ring via the messaging-layer recorder.
    p.record_own_reply("C1", thread_ts=None, ts="3.0", text="my own post")
    assert "my own post" in p.render_envelope("C1")


@pytest.mark.asyncio
async def test_feed_noop_when_pulse_disabled():
    host = _mixin_host(None)
    # must not raise
    await host._feed_channel_pulse({"channel": "C1", "ts": "1.0", "text": "x"})


# -------------------------------------------------- suffix placement (not prefix)

def _bind_utils():
    from message_processor.utilities import MessageUtilitiesMixin

    class P(MessageUtilitiesMixin):
        def log_debug(self, *a, **k): pass

    return P.__new__(P)


def test_envelope_rides_suffix_not_system_prompt():
    proc = _bind_utils()
    client = MagicMock()
    client.channel_pulse = _seeded_pulse()
    suffix = proc._build_suffix_context(client, "C1", None)
    assert "[Recent channel activity" in suffix
    assert "[Current date and time:" in suffix
    # DM: envelope absent, time still present
    dm_suffix = proc._build_suffix_context(client, "D1", None)
    assert "[Recent channel activity" not in dm_suffix
    # And the envelope never appears in the system prompt builder's output — the
    # system prompt has no pulse/client access at all; assert the source contract.
    import inspect
    from message_processor import utilities as u
    src = inspect.getsource(u.MessageUtilitiesMixin._get_system_prompt)
    assert "pulse" not in src and "Recent channel activity" not in src


def test_envelope_disabled_when_pulse_none():
    proc = _bind_utils()
    client = MagicMock(spec=[])  # no channel_pulse attribute at all
    assert proc._build_pulse_envelope(client, "C1", None) is None


def test_envelope_respects_env_cap(monkeypatch):
    proc = _bind_utils()
    client = MagicMock()
    client.channel_pulse = _seeded_pulse()
    monkeypatch.setattr(config, "channel_pulse_envelope_max", 0)
    assert proc._build_pulse_envelope(client, "C1", None) is None
