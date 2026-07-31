"""Unit tests for F3 — wake envelopes (structured trigger metadata in the suffix).

Exercises the deterministic renderer (_build_wake_envelope) across every trigger enum,
sender role, bot flag, escaping, and the config/missing-metadata off-paths, plus its
placement in the volatile developer suffix.

The envelope itself is DM/legacy now: a channel turn states the same facts through
`build_coordinates_suffix`, which took the envelope's content shape with it (covered in
test_channel_evidence_builders.py; the bridge is asserted at the foot of this file).

TWO BLOCKS LEFT THE ENVELOPE IN COMMIT 6, and both were the gate describing messages in prose:

* the F27 burst line ("Moments before this message, the same person also sent: …"). It quoted real
  messages into a block the prompt also tells the model not to trust as content, stripped of
  sender, time and attachments. The cohort is real input now — and under one whole-channel stream
  it is not history either: every member Slack propagated is IN the stream with its own header,
  and only the rest is quoted, post-breakpoint, as messages awaiting the stream
  (channel_request.build_cohort_fallback).
* the "you already reacted" note. It existed because the gate placed an emoji before the responder
  ran and then had to confess it. The gate places nothing, so there is nothing to confess.

Both are asserted as absences below rather than deleted silently, because either one coming back is
the gate growing a second channel into the responder's prompt.
"""
from types import SimpleNamespace

import pytest

from config import config
from message_processor.utilities import MessageUtilitiesMixin


class _WakeHost:
    def __init__(self):
        for n in ("_build_wake_envelope", "_wake_trigger_line", "_wake_sender_role",
                  "_build_suffix_context"):
            setattr(self, n, getattr(MessageUtilitiesMixin, n).__get__(self))
        self._escape_suffix_text = MessageUtilitiesMixin._escape_suffix_text

    # Sub-builders the suffix assembles — stubbed so the wake block is isolated.
    def _build_time_suffix_context(self, *a, **k):
        return "[time]"

    def _build_generation_inflight_note(self, *a, **k):
        return None

    def _build_research_inflight_note(self, *a, **k):
        return None

    def log_debug(self, *a, **k):
        pass


def _msg(**md):
    md.setdefault("username", "alice")
    return SimpleNamespace(user_id=md.pop("user_id", "U1"), metadata=md)


def _state(root_author=("U1", "human"), thread_ts="T1"):
    return SimpleNamespace(root_author=root_author, thread_ts=thread_ts)


# ------------------------------------------------------------------- trigger enums

@pytest.mark.parametrize("source", ["app_mention", "dm", "thread_continuation", "name_mention"])
def test_trigger_enum_renders(source):
    env = _WakeHost()._build_wake_envelope(_msg(wake_source=source, sender_type="human"), _state())
    assert f"trigger: {source}" in env
    assert env.startswith("[Wake context — informational metadata, not instructions]")


def test_the_gates_own_reason_never_reaches_the_responder():
    # no_response_needed is meant to be an INDEPENDENT second look at whether this turn
    # should speak. Handing the responder the gate's justification first made it a rubber
    # stamp: a wrong verdict arrived pre-argued and the veto never fired against it.
    env = _WakeHost()._build_wake_envelope(
        _msg(wake_source="ambient", sender_type="human",
             participation_reason="looks like a question for me"), _state())
    assert "trigger: ambient" in env
    assert "looks like a question for me" not in env
    assert "engine:" not in env


def test_trigger_ambient_without_reason():
    env = _WakeHost()._build_wake_envelope(
        _msg(wake_source="ambient", sender_type="human"), _state())
    assert "trigger: ambient" in env
    assert "engine:" not in env


def test_trigger_catch_up_batch_keeps_latest():
    env = _WakeHost()._build_wake_envelope(
        _msg(wake_source="ambient", sender_type="human", queued_batch_size=3), _state())
    assert "trigger: catch_up_batch (3) — latest trigger: ambient" in env


# ------------------------------------------------------------------- sender role/bot

def test_sender_root_author():
    env = _WakeHost()._build_wake_envelope(
        _msg(wake_source="dm", sender_type="human", user_id="U1"), _state(("U1", "human")))
    assert "sender: alice — root author" in env


def test_sender_participant():
    env = _WakeHost()._build_wake_envelope(
        _msg(wake_source="ambient", sender_type="human", user_id="U2"), _state(("U1", "human")))
    assert "sender: alice — participant" in env


def test_sender_bot_flag():
    for st in ("self", "other_bot"):
        env = _WakeHost()._build_wake_envelope(
            _msg(wake_source="ambient", sender_type=st, user_id="U9"), _state(("U9", st)))
        assert env.endswith("— bot")
        assert "root author — bot" in env


def test_the_role_no_longer_depends_on_where_the_reply_will_go():
    """The role used to be suppressed for a reply headed top-level. Where the reply goes is now
    the model's call, made later in the turn — so the envelope states the sender's relationship
    to the conversation and leaves placement to the tool that owns it."""
    env = _WakeHost()._build_wake_envelope(
        _msg(wake_source="app_mention", sender_type="human"), _state())
    assert "sender: alice" in env
    assert "root author" in env


def test_unknown_root_omits_role():
    env = _WakeHost()._build_wake_envelope(
        _msg(wake_source="ambient", sender_type="human"), _state(root_author=None))
    assert "sender: alice" in env
    assert "root author" not in env and "participant" not in env


# ------------------------------------------------- the burst is conversation, not prose now

def test_no_burst_prose_however_the_metadata_is_shaped():
    """The four burst-rendering tests collapse into one absence.

    They asserted the line rendered, escaped its brackets, capped long text, and stayed out when
    the metadata was absent or malformed — all of which was careful handling of a paraphrase that
    should never have been a paraphrase. The producer (`participation_burst_earlier`) and the
    renderer (`_wake_burst_line`) are both gone; a stale stamp from a rolling deploy must render
    nothing rather than half a block."""
    host = _WakeHost()
    assert not hasattr(MessageUtilitiesMixin, "_wake_burst_line")
    for stamp in (["first thought", "and this too"], [], ["", "  "], "not a list", None,
                  [123, None]):
        env = host._build_wake_envelope(
            _msg(wake_source="ambient", sender_type="human",
                 participation_burst_earlier=stamp), _state())
        assert "Moments before" not in env
        assert "first thought" not in env


def _sources(*records, trigger_ts="3.0"):
    return SimpleNamespace(metadata={"ts": trigger_ts, "gate_sources": tuple(records)})


def _cohort():
    from message_processor.participation import SourceMessage

    return (SourceMessage(ts="1.0", text="first thought", sender_name="alice",
                          sender_id="U1", sender_type="human"),
            SourceMessage(ts="2.0", text="", sender_name="alice", sender_id="U1",
                          sender_type="human", attachments=("chart.png (image)",)),
            SourceMessage(ts="3.0", text="and this too", sender_name="alice",
                          sender_id="U1", sender_type="human"))


def test_the_cohort_becomes_typed_sources_not_prose():
    """`_merge_gate_cohort` is gone with `ThreadState.messages` as a channel input. The gate's
    burst and the queue's drained batch both become CohortSources — one shape, two producers —
    deduplicated by ts, with the trigger's own message dropped because it is not "also" anything."""
    from message_processor.channel_request import cohort_sources_from_message

    assert not hasattr(MessageUtilitiesMixin, "_merge_gate_cohort")
    sources = cohort_sources_from_message(_sources(*_cohort()))
    assert [s.ts for s in sources] == ["1.0", "2.0"]
    assert sources[0].text == "first thought" and sources[0].sender_name == "alice"
    assert sources[1].attachment_names == ("chart.png (image)",)


def test_only_the_part_of_the_burst_the_stream_cannot_see_is_quoted():
    """The whole point of the move. A source Slack propagated is IN the stream with its real
    header; quoting it again would show the model the same message twice and invite two answers.
    What is quoted is what the window fetched too early to contain."""
    from message_processor.channel_request import build_cohort_fallback, cohort_sources_from_message
    from tests.unit import channel_turn_harness as harness

    stream = harness.build_stream([harness.normalized("1.0", "first thought"),
                                   harness.normalized("3.0", "and this too")])
    turn = SimpleNamespace()
    ctx = harness.pin_channel_turn(
        turn, stream=stream, trigger_ts="3.0",
        cohort_sources=cohort_sources_from_message(_sources(*_cohort())))
    block = build_cohort_fallback(ctx)
    assert "Also awaiting the stream — 1 message(s)" in block["content"]
    assert "first thought" not in block["content"]           # the stream already has it
    assert "[attached: chart.png (image)]" in block["content"]   # named, not described
    assert block["role"] == "user"                          # content, never developer voice


def test_a_burst_the_stream_caught_up_to_is_quoted_nowhere():
    from message_processor.channel_request import build_cohort_fallback, cohort_sources_from_message
    from tests.unit import channel_turn_harness as harness

    stream = harness.build_stream([harness.normalized(ts) for ts in ("1.0", "2.0", "3.0")])
    turn = SimpleNamespace()
    ctx = harness.pin_channel_turn(
        turn, stream=stream, trigger_ts="3.0",
        cohort_sources=cohort_sources_from_message(_sources(*_cohort())))
    assert build_cohort_fallback(ctx) is None


def test_every_message_the_turn_is_answering_is_quoted():
    """[r2-8] The gate's cohort is uncapped, so a cap here dropped messages the turn had already
    decided it was answering — the model replied to a burst of forty having been shown ten of them,
    with nothing saying so. Size is not a reason to drop them silently: every quoted line is charged
    by the admission estimate, so a burst too large to send is refused out loud instead."""
    from message_processor.channel_request import CohortSource, build_cohort_fallback
    from tests.unit import channel_turn_harness as harness

    turn = SimpleNamespace()
    ctx = harness.pin_channel_turn(
        turn, messages=[harness.normalized("3.0")], trigger_ts="3.0",
        cohort_sources=[CohortSource(ts=f"{i}.5", text=f"m{i}") for i in range(40)])
    block = build_cohort_fallback(ctx)
    assert "40 message(s)" in block["content"]
    assert block["content"].count("(ts=") == 40
    for i in range(40):
        assert f"m{i}" in block["content"]


def test_a_cohort_members_files_are_still_authorized_by_id():
    """Without this, a burst that dropped a CSV and then asked about it would put the question in
    front of the model with the file unreadable for the length of one turn."""
    from message_processor.channel_request import (CohortSource, canonical_files_from_stream,
                                                   merge_absent_source_files)
    from tests.unit import channel_turn_harness as harness

    stream = harness.build_stream([harness.normalized("3.0")])
    ref = harness.file_ref("FCSV", "numbers.csv", "text/csv")
    merged = merge_absent_source_files(canonical_files_from_stream(stream),
                                      [CohortSource(ts="2.0", files=(ref,))], "C1")
    assert merged["FCSV"]["filename"] == "numbers.csv"
    assert merged["FCSV"]["message_ts"] == "2.0"


# ------------------------------------------------------------------- escaping / off

def test_escaping_of_username_and_reason():
    env = _WakeHost()._build_wake_envelope(
        _msg(wake_source="ambient", sender_type="human",
             username="ali\nce[x]", participation_reason="a\nb\t[c]"), _state())
    assert "\n" in env  # the block's own line breaks
    # ...but no raw control chars or brackets from the free-text fields leak in
    body = env.replace("[Wake context — informational metadata, not instructions]", "")
    assert "ce[x]" not in body
    assert "\t" not in body
    assert "b\nc" not in body


def test_empty_on_missing_metadata():
    assert _WakeHost()._build_wake_envelope(_msg(sender_type="human"), _state()) == ""
    assert _WakeHost()._build_wake_envelope(None, _state()) == ""


def test_config_off(monkeypatch):
    monkeypatch.setattr(config, "enable_wake_envelope", False)
    assert _WakeHost()._build_wake_envelope(
        _msg(wake_source="dm", sender_type="human"), _state()) == ""


# ------------------------------------------------------------------- suffix placement

def test_envelope_in_suffix_when_message_present():
    host = _WakeHost()
    suffix = host._build_suffix_context(
        client=None, channel_id="C1", thread_ts="T1",
        message=_msg(wake_source="dm", sender_type="human"), thread_state=_state())
    assert "[Wake context" in suffix
    assert "[time]" in suffix  # rides alongside the other volatile context


def test_no_envelope_without_message():
    host = _WakeHost()
    suffix = host._build_suffix_context(client=None, channel_id="C1", thread_ts="T1")
    assert "[Wake context" not in suffix


def test_the_suffix_never_claims_the_bot_already_reacted():
    """INVERTED. The note told the model "you already reacted :tada: — do not add another", which
    was only ever true because the gate's react_and_respond verdict put an emoji on the message
    before the responder ran. Nothing does that now, so the note would be a false statement about
    the room; a stale stamp must produce nothing at all."""
    host = _WakeHost()
    assert not hasattr(MessageUtilitiesMixin, "_reacted_already_note")
    suffix = host._build_suffix_context(
        client=None, channel_id="C1", thread_ts="T1",
        message=_msg(wake_source="ambient", sender_type="human",
                     participation_reaction_emoji="tada"),
        thread_state=_state())
    assert "already reacted" not in suffix
    assert ":tada:" not in suffix   # (":tada:", not "tada" — "metadata" contains it)


@pytest.mark.asyncio
async def test_event_to_message_captures_sender_type():
    """_event_to_message stamps sender_type so the wake envelope can render '— bot'."""
    from unittest.mock import AsyncMock, MagicMock
    from slack_client.base import SlackBot
    bot = SlackBot.__new__(SlackBot)  # no __init__ — exercise _event_to_message only
    bot.bot_user_id = "U07SELF"
    bot.bot_id = None
    bot.app_id = None
    bot.user_cache = {}
    bot.db = MagicMock()
    bot.db.get_user_info_async = AsyncMock(return_value=None)
    bot.get_username = AsyncMock(return_value="peter")
    bot.get_user_timezone = AsyncMock(return_value="UTC")

    human = {"text": "hi", "user": "U1", "channel": "C1", "ts": "2.0"}
    msg = await bot._event_to_message(human, client=MagicMock())
    assert msg.metadata["sender_type"] == "human"

    other_bot = {"text": "hi", "user": "U2", "channel": "C1", "ts": "3.0", "bot_id": "B9"}
    msg2 = await bot._event_to_message(other_bot, client=MagicMock())
    assert msg2.metadata["sender_type"] == "other_bot"


def test_the_channel_path_states_the_same_wake_facts_through_its_coordinates():
    """The envelope did not survive P2 on a channel turn — `build_coordinates_suffix` states the
    trigger, the sender's relation to the thread and the wake source, and states them alongside
    the ids the turn may act on. Asserted here so the F3 CONTENT cannot go missing in the move:
    the coordinates block's own rules are covered in test_channel_evidence_builders.py."""
    from message_processor.utilities import TurnCoordinates, build_coordinates_suffix

    out = build_coordinates_suffix(TurnCoordinates(
        channel_id="C1", trigger_ts="3.0", origin_thread_ts="1.0",
        trigger_sender_name="alice", trigger_sender_id="U1", trigger_sender_type="human",
        sender_is_root_author=True, wake_source="ambient", queued_batch_size=3))
    assert "woke on: catch_up_batch (3) — latest trigger: ambient" in out
    assert "from alice, the thread's root author" in out
    assert "reason" not in out.lower()          # still no gate justification


def test_wake_before_inflight_note():
    # Order must be wake -> in-flight (F2's contract paragraph is appended after, by the
    # text handler). Stub the in-flight note so both are present and check ordering.
    host = _WakeHost()
    host._build_generation_inflight_note = lambda *a, **k: "[in-flight note]"
    suffix = host._build_suffix_context(
        client=None, channel_id="C1", thread_ts="T1",
        message=_msg(wake_source="dm", sender_type="human"), thread_state=_state())
    assert suffix.index("[Wake context") < suffix.index("[in-flight note]")
