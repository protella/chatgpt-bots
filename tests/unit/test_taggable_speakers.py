"""AREA A — taggable recent channel speakers: the SUFFIX block + roster non-contamination.

The pulse accessor (ChannelPulse.recent_taggable_speakers) is covered in test_channel_pulse.py.
Here we cover the response-surface wiring:

- MessageUtilitiesMixin._build_taggable_speakers_block renders a clearly-labeled suffix block of
  channel peers the model can @-mention, EXCLUDING anyone already in the thread participant roster,
  channels only, empty→None; it passes bot_user_id through and is wired into _build_suffix_context.
- build_roster_text and the multi-user-thread detection (utilities.py: participant_roster.count
  ("→ <@") >= 2) stay UNCHANGED — ambient speakers are NOT merged into the thread roster, so a
  crowded ambient block can never inflate the multi-user count.
- The prompt nudge (LOCAL_TOOLS_GUIDANCE) tells the model how to reach a taggable id.

All in-memory; no network/DB.
"""
from __future__ import annotations

from types import SimpleNamespace

from message_processor.utilities import MessageUtilitiesMixin, build_roster_text

HEADER = "RECENT CHANNEL SPEAKERS you can @-mention"


class _StubPulse:
    """Records how recent_taggable_speakers was called and returns a fixed list."""
    def __init__(self, speakers):
        self._speakers = speakers
        self.calls = []

    def recent_taggable_speakers(self, channel_id, *, bot_user_id=None, **kw):
        self.calls.append({"channel_id": channel_id, "bot_user_id": bot_user_id})
        return list(self._speakers)


class _Utils(MessageUtilitiesMixin):
    def log_debug(self, *a, **k):
        pass


def _host():
    return _Utils.__new__(_Utils)


def _client(pulse, bot_user_id="UBOT"):
    return SimpleNamespace(channel_pulse=pulse, bot_user_id=bot_user_id)


def _tstate(participants=None):
    return SimpleNamespace(participants=participants or {})


# ------------------------------------------------------------------ suffix block

def test_taggable_block_renders_mentionable_ids():
    pulse = _StubPulse([{"user_id": "UA", "name": "Alice"},
                        {"user_id": "UB", "name": "Bob"}])
    block = _host()._build_taggable_speakers_block(_client(pulse), "C1", _tstate())
    assert HEADER in block
    assert "Alice → <@UA>" in block
    assert "Bob → <@UB>" in block


def test_taggable_block_excludes_thread_participants():
    pulse = _StubPulse([{"user_id": "UA", "name": "Alice"},
                        {"user_id": "UB", "name": "Bob"}])
    # Alice is already in the thread roster (system prompt) — the block is for peers who AREN'T.
    block = _host()._build_taggable_speakers_block(
        _client(pulse), "C1", _tstate({"UA": "Alice"}))
    assert "<@UB>" in block
    assert "<@UA>" not in block


def test_taggable_block_none_when_every_speaker_in_thread():
    pulse = _StubPulse([{"user_id": "UA", "name": "Alice"}])
    assert _host()._build_taggable_speakers_block(
        _client(pulse), "C1", _tstate({"UA": "Alice"})) is None


def test_taggable_block_none_when_pulse_empty():
    assert _host()._build_taggable_speakers_block(
        _client(_StubPulse([])), "C1", _tstate()) is None


def test_taggable_block_none_for_dm_and_no_channel_without_touching_ring():
    pulse = _StubPulse([{"user_id": "UA", "name": "Alice"}])
    host = _host()
    assert host._build_taggable_speakers_block(_client(pulse), "D1", _tstate()) is None
    assert host._build_taggable_speakers_block(_client(pulse), None, _tstate()) is None
    assert pulse.calls == []   # short-circuited before consulting the pulse


def test_taggable_block_none_when_no_pulse_on_client():
    client = SimpleNamespace(channel_pulse=None, bot_user_id="UBOT")
    assert _host()._build_taggable_speakers_block(client, "C1", _tstate()) is None


def test_taggable_block_passes_bot_user_id_to_pulse():
    pulse = _StubPulse([{"user_id": "UA", "name": "Alice"}])
    _host()._build_taggable_speakers_block(
        _client(pulse, bot_user_id="UZZZ"), "C1", _tstate())
    assert pulse.calls[0]["bot_user_id"] == "UZZZ"


def test_taggable_block_wired_into_suffix_context():
    pulse = _StubPulse([{"user_id": "UA", "name": "Alice"}])
    host = _host()
    # Neutralize the OTHER suffix builders so this isolates the taggable-block wiring.
    host._build_time_suffix_context = lambda *a, **k: "[Current date and time: X]"
    host._build_pulse_envelope = lambda *a, **k: None
    host._build_channel_people_line = lambda *a, **k: None
    host._build_wake_envelope = lambda *a, **k: None
    host._build_generation_inflight_note = lambda *a, **k: None
    host._build_research_inflight_note = lambda *a, **k: None
    suffix = host._build_suffix_context(_client(pulse), "C1", None, thread_state=_tstate())
    assert HEADER in suffix
    assert "Alice → <@UA>" in suffix


# ---------------------------------------- roster / multi-user detection UNCHANGED

def test_multiuser_detection_reads_only_thread_roster():
    # The detection expression (utilities.py:1035) is participant_roster.count("→ <@") >= 2.
    # It reads ONLY the thread roster, so one participant stays single-user even when the
    # ambient taggable block (a SEPARATE string) is crowded with arrows.
    roster = build_roster_text({"UPETER": "Peter"})
    assert roster.count("→ <@") == 1                       # single-user → line kept
    two = build_roster_text({"UPETER": "Peter", "UDANA": "Dana"})
    assert two.count("→ <@") == 2                          # genuine multi-user thread

    pulse = _StubPulse([{"user_id": f"U{i}", "name": f"N{i}"} for i in range(3)])
    block = _host()._build_taggable_speakers_block(_client(pulse), "C1", _tstate())
    assert block.count("→ <@") == 3                        # ambient arrows live here…
    assert block not in roster and roster not in block     # …a different string entirely


def test_build_roster_text_signature_and_behavior_unchanged():
    # build_roster_text takes ONLY participants (no pulse/ambient param) — ambient speakers
    # cannot leak in. Its skip/self-guard behavior is preserved.
    assert build_roster_text({}) == ""
    assert build_roster_text({"bot": "x", "unknown": "y"}) == ""
    out = build_roster_text({"UPETER": "Peter", "UBOT": "ChatGPT"}, bot_user_id="UBOT")
    assert "<@UPETER>" in out and "UBOT" not in out


# ---------------------------------------------------------------- prompt nudge (A4)

def test_prompt_nudge_points_at_recent_speakers_and_list_members():
    from prompts import LOCAL_TOOLS_GUIDANCE
    g = LOCAL_TOOLS_GUIDANCE
    assert "RECENT CHANNEL SPEAKERS" in g          # you may @-mention someone listed
    assert "list_channel_members" in g             # …or fetch an id for someone who isn't
    assert "<@id>" in g
