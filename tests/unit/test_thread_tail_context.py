"""F5 — the ChannelPulse rings, and the debounce marker that orders the gate.

Covers the per-thread ACTOR ring (population incl. roots + bot senders, its LRU bounds,
idempotency), the messaging-layer own-reply feed, the reliable event feed (bot_message /
edits / dual delivery), the engine's monotonic debounce marker, and the direct-continuation
denial that the actor ring exists to serve.

The pulse is no longer a GATE input — the binary gate is given the source messages and the
steering snapshot, nothing else — so the "renders the thread tail into the classifier signals"
test inverted into the tripwire at the bottom of this file.

The per-thread ring's PROSE renderers went with commit 7a: render_thread_tail and
render_channel_addressee_tail rendered addressee evidence for the rich gate, and the gate reads
neither. The ring itself stays, holding actor state only (ts / is_bot / sender_type), because
thread_has_other_bot() reads it to deny the deterministic 1:1 continuation fast path. Every
bound test below therefore asserts through thread_has_other_bot: a bound that silently narrowed
would let a second agent fall out of the window and re-open that fast path, which is the whole
failure the ring prevents.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from config import config
from message_markers import CHECKLIST_STATUS_MARKER
from message_processor.participation import ParticipationEngine
from slack_client.channel_pulse import ChannelPulse


def _rec(p, channel, ts, text, *, thread_ts=None, name="Alice", sender="human", is_bot=False):
    p.record(channel, ts=ts, thread_ts=thread_ts, user_id="U", display_name=name,
             sender_type=sender, text=text, is_bot=is_bot)


def _bot(p, channel, ts, thread_ts, text="claude reply"):
    _rec(p, channel, ts, text, thread_ts=thread_ts,
         name="Claude", sender="other_bot", is_bot=True)


def _ring(p, channel, root_ts):
    """The raw actor deque for a thread, or None. Tests read it to prove the ring holds actor
    state and NOT message prose."""
    return (p._thread_tails.get(channel) or {}).get(root_ts)


# ----------------------------------------------------------------- ring core

def test_actor_ring_records_root_and_replies():
    p = ChannelPulse()
    _rec(p, "C1", "100.0", "root question")                 # top-level seeds the thread ring
    _rec(p, "C1", "101.0", "a reply", thread_ts="100.0", name="Bob")
    _bot(p, "C1", "102.0", "100.0")
    dq = _ring(p, "C1", "100.0")
    assert [e["ts"] for e in dq] == ["100.0", "101.0", "102.0"]
    assert [e["is_bot"] for e in dq] == [False, False, True]


def test_actor_ring_stores_no_message_text():
    # The prose fields existed only for the deleted gate tails. Storing them again would be
    # storage nothing reads — and, for the display name, an untrusted string kept for no reason.
    p = ChannelPulse()
    _rec(p, "C1", "100.0", "root")
    _rec(p, "C1", "101.0", "a secret paste", thread_ts="100.0", name="Bob")
    entry = _ring(p, "C1", "100.0")[-1]
    assert set(entry) == {"ts", "is_bot", "sender_type"}
    assert "a secret paste" not in repr(entry)


def test_record_idempotent_by_ts():
    p = ChannelPulse()
    _rec(p, "C1", "100.0", "root")
    _rec(p, "C1", "101.0", "hello world", thread_ts="100.0")
    _rec(p, "C1", "101.0", "hello world", thread_ts="100.0")   # retry / dual delivery
    assert [e["ts"] for e in _ring(p, "C1", "100.0")] == ["100.0", "101.0"]
    assert p.render_envelope("C1").count("hello world") == 1


def test_envelope_keeps_the_head_of_a_long_message(monkeypatch):
    # The 400-char TAIL slice died with the addressee renderers; the envelope's head-first
    # truncation is what survives, and it must still admit what it dropped.
    monkeypatch.setattr(config, "enable_message_timestamps", False)
    p = ChannelPulse()
    _rec(p, "C1", "100.0", "H" * 500 + "T" * 500)           # 1000-char message
    env = p.render_envelope("C1")
    assert ("H" * 100) in env and ("T" * 100) not in env
    assert "chars truncated" in env                          # no silent cap


def test_recent_speakers_neutralizes_a_spoofed_display_name():
    # _sanitize_name outlived the tails: recent_speakers feeds the responder's people line, and
    # a name like "Claude [bot]" there would forge a trusted label just as it would have in a tail.
    p = ChannelPulse()
    _rec(p, "C1", "100.0", "hi", name="Claude [bot]\n- Evil")
    names = p.recent_speakers("C1")
    assert names == ["Claude (bot) - Evil"]                  # brackets folded, newline → space
    assert "[bot]" not in names[0] and "\n" not in names[0]


# ------------------------------------------------------------------ LRU bounds

def test_window_bound_evicts_the_other_bot(monkeypatch):
    # participation_thread_tail sizes the per-thread window. Narrowing it would drop a second
    # agent out of view and silently re-open the 1:1 fast path — this pins where the edge is.
    monkeypatch.setattr(config, "participation_thread_tail", 2)
    p = ChannelPulse()
    _rec(p, "C1", "100.0", "root")
    _bot(p, "C1", "101.0", "100.0")
    assert p.thread_has_other_bot("C1", "100.0") is True
    for i in range(2, 8):
        _rec(p, "C1", f"10{i}.0", f"msg{i}", thread_ts="100.0")
    assert p.thread_has_other_bot("C1", "100.0") is False    # pushed out of the window


def test_thread_lru_eviction(monkeypatch):
    monkeypatch.setattr(config, "pulse_thread_tails_max", 2)
    p = ChannelPulse()
    for root in ("100.0", "200.0", "300.0"):
        _rec(p, "C1", root, "root")
        _bot(p, "C1", root.replace("00", "01"), root)
    assert p.thread_has_other_bot("C1", "100.0") is False     # thread evicted
    assert p.thread_has_other_bot("C1", "300.0") is True


def test_channel_lru_eviction(monkeypatch):
    monkeypatch.setattr(config, "pulse_thread_tail_channels_max", 2)
    p = ChannelPulse()
    for ch in ("C1", "C2", "C3"):
        _rec(p, ch, "100.0", "root")
        _bot(p, ch, "101.0", "100.0")
    assert p.thread_has_other_bot("C1", "100.0") is False     # channel evicted
    assert p.thread_has_other_bot("C3", "100.0") is True


def test_lru_recency_refresh(monkeypatch):
    monkeypatch.setattr(config, "pulse_thread_tails_max", 2)
    p = ChannelPulse()
    _rec(p, "C1", "100.0", "root")
    _bot(p, "C1", "101.0", "100.0")
    _rec(p, "C1", "200.0", "root")
    _bot(p, "C1", "201.0", "200.0")
    _rec(p, "C1", "102.0", "one-more", thread_ts="100.0")     # touch thread 100 → most recent
    _rec(p, "C1", "300.0", "root")
    _bot(p, "C1", "301.0", "300.0")
    # thread 200 (least recently touched) is evicted, not 100
    assert p.thread_has_other_bot("C1", "200.0") is False
    assert p.thread_has_other_bot("C1", "100.0") is True


# ------------------------------------------------------------- disable / cold start

def test_zero_disables_recording_and_signal(monkeypatch):
    monkeypatch.setattr(config, "participation_thread_tail", 0)
    p = ChannelPulse()
    _rec(p, "C1", "100.0", "root")
    _bot(p, "C1", "101.0", "100.0")
    assert _ring(p, "C1", "100.0") is None
    assert p.thread_has_other_bot("C1", "100.0") is False


def test_cold_start_empty_ring_degrades():
    p = ChannelPulse()
    assert p.thread_has_other_bot("C1", "999.0") is False


# ---------------------------------------------------------------- other-bot gate

def test_thread_has_other_bot_excludes_self():
    p = ChannelPulse()
    _rec(p, "C1", "100.0", "root")
    _rec(p, "C1", "101.0", "hi", thread_ts="100.0")                    # human
    assert p.thread_has_other_bot("C1", "100.0") is False
    _bot(p, "C1", "102.0", "100.0")
    assert p.thread_has_other_bot("C1", "100.0") is True

    p2 = ChannelPulse()
    _rec(p2, "C2", "100.0", "root")
    p2.record_own_reply("C2", thread_ts="100.0", ts="101.0", text="my own reply")
    assert p2.thread_has_other_bot("C2", "100.0") is False            # self doesn't count


def test_other_bot_defeats_the_1to1_continuation_fast_path():
    """The reason the actor ring survived 7a.

    The replies fast path scans only the oldest page, so a SECOND agent later in a long thread is
    invisible to it and the thread looks 1:1. The ring sees it, and a 1:1 continuation must then
    become gate-judged instead of a judgment-free direct answer — a bot may be the real addressee.

    Asserted on the source of the decision site, because the surrounding handler needs a whole
    Slack client to run and what matters is that the ring is still consulted at exactly the point
    that clears `direct_continuation`.
    """
    import inspect

    from slack_client.event_handlers.message_events import SlackMessageEventsMixin

    src = inspect.getsource(SlackMessageEventsMixin)
    idx = src.index("pulse.thread_has_other_bot(channel_id, thread_ts)")
    # The very next statement must be the one that drops the deterministic route.
    assert "direct_continuation = False" in src[idx:idx + 120]

    # And the predicate itself still answers True for the case that decision depends on.
    p = ChannelPulse()
    _rec(p, "C1", "100.0", "root")
    _rec(p, "C1", "101.0", "a human reply", thread_ts="100.0")
    _bot(p, "C1", "102.0", "100.0")
    assert p.thread_has_other_bot("C1", "100.0") is True


# -------------------------------------------------------- messaging-layer own reply

def _msg_host(pulse):
    from slack_client.messaging import SlackMessagingMixin

    class Host(SlackMessagingMixin):
        def __init__(self):
            self.channel_pulse = pulse

        def log_debug(self, *a, **k):
            pass

        log_info = log_warning = log_error = log_debug

    return Host()


def test_own_reply_helper_records_clean_excludes_chrome():
    p = ChannelPulse()
    host = _msg_host(p)
    host._record_own_reply_pulse("C1", "100.0", "101.0", "a real answer")
    host._record_own_reply_pulse("C1", "100.0", "102.0", "step done" + CHECKLIST_STATUS_MARKER)
    host._record_own_reply_pulse("C1", "100.0", "103.0", "   ")     # empty
    host._record_own_reply_pulse("C1", "100.0", None, "no ts")      # missing ts
    out = p.render_envelope("C1")
    assert "a real answer" in out
    assert "step done" not in out and "no ts" not in out


# ------------------------------------------------------------------- event feed

def _feed_host(pulse):
    from slack_client.event_handlers.message_events import SlackMessageEventsMixin

    class Host(SlackMessageEventsMixin):
        def __init__(self):
            self.channel_pulse = pulse
            self.user_cache = {}
            self.bot_user_id = "UBOT"

        def is_own_message(self, e):
            return e.get("user") == "UBOT" or e.get("bot_id") == "BSELF"

        def classify_sender(self, e):
            if self.is_own_message(e):
                return "self"
            return "other_bot" if (e.get("bot_id") or e.get("app_id")) else "human"

        def log_debug(self, *a, **k):
            pass

    return Host()


@pytest.mark.asyncio
async def test_feed_records_bot_message_subtype():
    p = ChannelPulse()
    host = _feed_host(p)
    await host._feed_channel_pulse({
        "channel": "C1", "ts": "100.0", "subtype": "bot_message",
        "bot_id": "BCLAUDE", "username": "Claude", "text": "hello from claude"})
    assert "hello from claude" in p.render_envelope("C1")


@pytest.mark.asyncio
async def test_feed_excludes_edits_and_own():
    p = ChannelPulse()
    host = _feed_host(p)
    await host._feed_channel_pulse({"channel": "C1", "ts": "100.0",
                                    "subtype": "message_changed", "text": "an edit"})
    await host._feed_channel_pulse({"channel": "C1", "ts": "101.0",
                                    "user": "UBOT", "text": "own echo"})
    assert p.render_envelope("C1") == ""


@pytest.mark.asyncio
async def test_feed_idempotent_dual_delivery():
    p = ChannelPulse()
    host = _feed_host(p)
    ev = {"channel": "C1", "ts": "100.0", "user": "U7", "text": "mentions arrive twice"}
    await host._feed_channel_pulse(ev)   # message event
    await host._feed_channel_pulse(ev)   # app_mention event (same ts)
    assert p.render_envelope("C1").count("mentions arrive twice") == 1


# ----------------------------------------------------------- engine debounce order

def test_note_arrival_is_monotonic():
    eng = ParticipationEngine(MagicMock())
    eng.note_arrival("C1", "100.0")
    eng.note_arrival("C1", "90.0")     # older — must not overwrite
    # F27: top-level stream key is per-sender; no sender_id → "unknown".
    assert eng._latest["C1|top|unknown"] == "100.0"
    eng.note_arrival("C1", "110.0")    # newer
    assert eng._latest["C1|top|unknown"] == "110.0"


@pytest.mark.asyncio
async def test_evaluate_superseded_by_newer_arrival(monkeypatch):
    monkeypatch.setattr(config, "participation_debounce_seconds", 0)
    eng = ParticipationEngine(MagicMock())
    eng.note_arrival("C1", "200.0")    # a newer message already registered at gate entry
    evaluation = await eng.evaluate(channel_id="C1", ts="100.0", text="stale event")
    assert evaluation.decision is None            # the older event never classifies
    assert evaluation.decline_cause == "superseded"


@pytest.mark.asyncio
async def test_the_pulse_is_not_a_gate_input_any_more(monkeypatch):
    """INVERTED, and the inversion is the contract.

    This test used to assert that the pulse's thread tail rendered into the classifier's signals.
    The binary gate takes the ordered source messages and the canonical steering snapshot; a thread
    tail existed to help it decide whose conversation a message belonged to and whether the exchange
    was still open, and it decides neither. The RESPONDER has the whole thread, which is a better
    version of the same information.

    Asserted at the signature, because that is what makes it unbuildable rather than merely
    unrendered — there is nowhere to put a pulse."""
    monkeypatch.setattr(config, "participation_debounce_seconds", 0)
    p = ChannelPulse()
    _rec(p, "C1", "100.0", "root")
    _rec(p, "C1", "101.0", "prior exchange between two humans", thread_ts="100.0")

    client = MagicMock()
    client.classify_wake = AsyncMock(return_value=False)
    eng = ParticipationEngine(client)
    with pytest.raises(TypeError):
        await eng.evaluate(channel_id="C1", ts="102.0", text="an unnamed follow-up",
                           pulse=p, thread_root_ts="100.0")

    # Without it the evaluation runs, and the classifier is handed sources + steering only.
    await eng.evaluate(channel_id="C1", ts="102.0", text="an unnamed follow-up",
                       thread_root_ts="100.0")
    kwargs = client.classify_wake.await_args.kwargs
    assert set(kwargs) == {"sources", "channel_steering_text"}
    assert [s.text for s in kwargs["sources"]] == ["an unnamed follow-up"]


def test_no_prose_tail_renderers_survive():
    """Zero-reference sweep: the rich gate's addressee renderers must not come back, and nothing
    may re-add a prose field to the actor ring by re-introducing one."""
    assert not hasattr(ChannelPulse, "render_thread_tail")
    assert not hasattr(ChannelPulse, "render_channel_addressee_tail")
    import slack_client.channel_pulse as cp
    assert not hasattr(cp, "_escape_tail_text")
