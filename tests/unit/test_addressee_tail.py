"""F47 — channel addressee tail: the structural fix for the top-level attribution bug.

A TOP-LEVEL trigger has an EMPTY thread tail (render_thread_tail excludes the root it
sits on), so before this fix the participation classifier had no authoritative record of
who the sender had been addressing. A bare "you" that continued the user's exchange with
ANOTHER assistant ("Claude") therefore got wrongly claimed. ChannelPulse.render_channel_
addressee_tail rebuilds that record from the channel ring, and classify_participation
renders it ABOVE the peripheral channel-activity envelope.

These are BEHAVIORAL tests (seed the exact incident flow, assert what the tail contains and
where it renders), not word-presence checks on the prompt.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from slack_client.channel_pulse import ChannelPulse
from openai_client.api.responses import classify_participation


def _rec(p, ts, text, *, thread_ts=None, name, sender, is_bot):
    p.record("C", ts=ts, thread_ts=thread_ts, user_id=("U" if not is_bot else None),
             display_name=name, sender_type=sender, text=text, is_bot=is_bot)


def _seed_incident(p):
    """The 'do you stream messages?' flow: Peter directs Claude, Claude answers, twice."""
    # Control line from OUR OWN bot in an unrelated exchange (labels [self]).
    _rec(p, "999.0", "heads up, standup moved to 10", name="ChatGPT", sender="self", is_bot=True)
    _rec(p, "1000.0", "hey claude, write me a story", name="Peter", sender="human", is_bot=False)
    _rec(p, "1001.0", "Once upon a time there was a bot", name="Claude", sender="other_bot", is_bot=True)
    _rec(p, "1002.0", "do it again in a thread", name="Peter", sender="human", is_bot=False)
    _rec(p, "1003.5", "A second tale about a fox", thread_ts="1002.0",
         name="Claude", sender="other_bot", is_bot=True)
    # The trigger itself (a bare top-level "you" follow-up) — recorded, but must be excluded.
    _rec(p, "1004.0", "do you stream messages?", name="Peter", sender="human", is_bot=False)


def _line_with(tail, needle):
    for ln in tail.splitlines():
        if ln.startswith("- ") and needle in ln:
            return ln
    raise AssertionError(f"no tail line contains {needle!r}\n{tail}")


def test_addressee_tail_captures_the_other_assistants_exchange():
    p = ChannelPulse(size=30)
    _seed_incident(p)
    tail = p.render_channel_addressee_tail("C", before_ts="1004.0")

    # Peter's Claude-directed request AND Claude's replies are in the record.
    assert "hey claude, write me a story" in tail
    assert "Once upon a time there was a bot" in tail
    assert "A second tale about a fox" in tail

    # The trigger is strictly-before-excluded — it must NOT appear.
    assert "do you stream messages" not in tail


def test_addressee_tail_labels_sender_types():
    p = ChannelPulse(size=30)
    _seed_incident(p)
    tail = p.render_channel_addressee_tail("C", before_ts="1004.0")

    # Claude → [bot]; Peter → [human]; our own bot → [self]. The trusted type, not the name.
    assert "[bot]" in _line_with(tail, "Once upon a time there was a bot")
    assert "[human]" in _line_with(tail, "hey claude, write me a story")
    assert "[self]" in _line_with(tail, "standup moved to 10")

    # The where-marker distinguishes the threaded reply from the top-level ones.
    assert "(in a thread)" in _line_with(tail, "A second tale about a fox")
    assert "(top-level)" in _line_with(tail, "hey claude, write me a story")

    # FIX 4 / F47b: the header frames this around the SENDER's continuity + the label legend, is
    # explicitly NOT a blanket "authoritative record" of the whole channel, and says an exchange
    # not involving the sender is "someone else's — not yours to answer, and not a reason for
    # silence" (so a busy unrelated thread can't bias a clearly-new ask toward silence). It also
    # carries the topic-shift rule: a bare 'you' after the sender addressed another assistant
    # continues with that assistant even on a NEW TOPIC.
    header = tail.splitlines()[0]
    assert "authoritative" not in header.lower()
    assert "sender" in header.lower() and "someone else" in header.lower()
    assert "not a reason for silence" in header.lower()
    assert "new topic" in header.lower()
    assert "[self] is you" in header and "[bot] is another assistant" in header


def test_addressee_tail_keeps_sender_exchange_amid_unrelated_chatter():
    # FIX 4: unrelated third-party chatter in the ring must not crowd the sender's real exchange
    # out of the tail — the sender's Claude-directed request still renders.
    p = ChannelPulse(size=30)
    _seed_incident(p)
    # Interleave unrelated humans talking amongst themselves (still before the trigger).
    _rec(p, "1002.5", "did anyone catch the game last night", name="Dana", sender="human", is_bot=False)
    _rec(p, "1003.7", "yeah wild finish", name="Sam", sender="human", is_bot=False)
    tail = p.render_channel_addressee_tail("C", before_ts="1004.0")
    assert "hey claude, write me a story" in tail        # the sender's exchange survives
    assert "A second tale about a fox" in tail


def test_addressee_tail_keeps_a_trailing_address_in_a_long_message():
    # FIX 3: the address often sits at the END of a long paste; the ring's 300-char HEAD `text`
    # drops it, so the tail must render the sanitized full-text tail instead.
    p = ChannelPulse(size=30)
    long_paste = "x " * 400  # ~800 chars, well past the 300 head cap
    _rec(p, "2000.0", f"{long_paste}Claude, thoughts?", name="Peter", sender="human", is_bot=False)
    tail = p.render_channel_addressee_tail("C", before_ts="2001.0")
    assert "Claude, thoughts?" in tail


def test_addressee_tail_explicit_zero_disables(monkeypatch):
    # FIX 5: an explicit max_entries=0 DISABLES (is-None semantics), never falls back to default.
    from config import config
    monkeypatch.setattr(config, "participation_addressee_tail", 8)
    p = ChannelPulse(size=30)
    _seed_incident(p)
    assert p.render_channel_addressee_tail("C", before_ts="1004.0", max_entries=0) == ""
    # None still uses the configured default (non-empty here).
    assert p.render_channel_addressee_tail("C", before_ts="1004.0", max_entries=None) != ""


def test_addressee_tail_empty_without_before_ts_or_ring():
    p = ChannelPulse(size=30)
    # No before_ts → nothing to bound against.
    _seed_incident(p)
    assert p.render_channel_addressee_tail("C", before_ts=None) == ""
    # Unknown channel → empty, never raises.
    assert p.render_channel_addressee_tail("C-nope", before_ts="1004.0") == ""


def test_addressee_tail_disabled_when_config_zero(monkeypatch):
    from config import config
    monkeypatch.setattr(config, "participation_addressee_tail", 0)
    p = ChannelPulse(size=30)
    _seed_incident(p)
    assert p.render_channel_addressee_tail("C", before_ts="1004.0") == ""


# ------------------------------------------------------------------ render ordering


class _FakeContent:
    def __init__(self, text):
        self.text = text


class _FakeItem:
    def __init__(self, text):
        self.content = [_FakeContent(text)]


class _FakeResp:
    def __init__(self, text):
        self.output = [_FakeItem(text)]


class _FakeLLM:
    def __init__(self, text='{"action": "ignore"}'):
        self._text = text
        self.client = MagicMock()
        self.captured_input = None

    async def _safe_api_call(self, *a, **k):
        self.captured_input = k.get("input")
        return _FakeResp(self._text)

    def log_debug(self, *a, **k):
        pass

    def log_warning(self, *a, **k):
        pass


@pytest.mark.asyncio
async def test_addressee_tail_renders_above_channel_activity():
    # The addressee tail is authoritative; the channel-activity envelope is peripheral — so the
    # tail must render BEFORE the "[Recent channel activity" block in the classifier prompt.
    llm = _FakeLLM()
    await classify_participation(llm, "do you stream messages?", signals={
        "channel_addressee_tail":
            "[Recent channel exchange just before this message — who the sender has been talking to]\n"
            "- Peter [human] (top-level): \"hey claude, write me a story\"",
        "channel_activity":
            "[Recent channel activity — peripheral context from OTHER conversations]\n"
            "- Dana (top-level): unrelated chatter",
    })
    prompt = llm.captured_input[1]["content"]
    assert (prompt.index("[Recent channel exchange just before this message")
            < prompt.index("[Recent channel activity"))
