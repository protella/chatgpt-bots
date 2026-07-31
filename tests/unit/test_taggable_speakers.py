"""AREA A — which ids the model may @-mention, and who is excluded.

The subject survived P2; its mechanism did not. There used to be TWO blocks: a thread roster in
the system prompt and an ambient "RECENT CHANNEL SPEAKERS" suffix block fed by a per-channel pulse
ring. The split existed only to protect the cached prefix — merging them would have made the
prompt vary per requester. Post-breakpoint evidence has no such constraint, so
`build_taggable_roster_evidence` renders ONE roster off the pinned stream's actor map and the
assembled channel request carries it after the breakpoint. The pulse ring is deleted.

The builder's own rules — recency ordering, the cap, escaping, the id sentinels — are covered in
test_channel_evidence_builders.py. What is covered HERE is the wiring: the roster names exactly
the people the pinned stream contains, it lands post-breakpoint in user voice, it is one block,
and the DM/legacy suffix emits neither of the retired blocks.
"""
from __future__ import annotations

import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from message_processor.channel_request import assemble_channel_request
from message_processor.utilities import MessageUtilitiesMixin, build_roster_text
from tests.unit import channel_turn_harness as harness

ROSTER_HEADER = "[PEOPLE YOU CAN @-MENTION HERE"
RETIRED_HEADER = "RECENT CHANNEL SPEAKERS you can @-mention"


def _assembler_host():
    """The processor surface `assemble_channel_request` reaches for, and nothing more.

    The two in-flight notes must return None rather than a MagicMock: the suffix filters on
    truthiness, and a mock would render as its own repr inside a developer-role block.
    """
    host = MagicMock()
    host._get_system_prompt = MagicMock(return_value="sys")
    host._build_time_suffix_context = MagicMock(return_value="[time]")
    host._build_generation_inflight_note = MagicMock(return_value=None)
    host._build_research_inflight_note = MagicMock(return_value=None)
    return host


def _assemble(messages, *, bot_user_id="UBOT", actor_map=None, requester=None,
              origin_participants=None):
    stream = harness.build_stream(messages, actor_map=actor_map)
    turn = SimpleNamespace()
    ctx = harness.pin_channel_turn(
        turn, stream=stream, requester=requester, origin_participants=origin_participants,
        prepared=harness.no_tools_prepared())
    return assemble_channel_request(
        processor=_assembler_host(), client=SimpleNamespace(bot_user_id=bot_user_id), ctx=ctx,
        model=harness.DEFAULT_MODEL, tools=None, request_config={}, contract_suffix=None)


def _roster_item(request):
    items = [item for item in request.input_items
             if isinstance(item.get("content"), str)
             and item["content"].startswith(ROSTER_HEADER)]
    assert len(items) == 1, f"expected exactly one roster block, got {len(items)}"
    return items[0]


# ------------------------------------------------------------------ the roster in the request

def test_the_roster_names_the_people_the_stream_contains():
    request = _assemble([harness.normalized("10.0", "morning", sender_id="U1"),
                         harness.normalized("11.0", "hi", sender_id="U2")])
    block = _roster_item(request)["content"]
    assert "<@U1>" in block and "<@U2>" in block


def test_the_roster_rides_after_the_breakpoint_in_user_voice():
    """It varies with the room minute to minute, and it is a catalog derived from what people
    wrote — so it goes below the cache breakpoint, and it is never developer voice."""
    request = _assemble([harness.normalized("10.0", "hi", sender_id="U1")])
    item = _roster_item(request)
    assert item["role"] == "user"
    assert not item.get("_stream")
    positions = [i for i, it in enumerate(request.input_items) if it.get("_stream")]
    assert request.input_items.index(item) > max(positions)


def test_other_bots_stay_taggable_and_we_do_not():
    """A peer agent has to be reachable; tagging ourselves is a loop."""
    request = _assemble(
        [harness.normalized("10.0", "hi", sender_id="U1"),
         harness.normalized("11.0", "beep", sender_id="UPEER", sender_type="other_bot",
                            raw_bot_name="Peerbot"),
         harness.normalized("12.0", "ok", sender_id="UBOT", sender_type="self")],
        bot_user_id="UBOT")
    block = _roster_item(request)["content"]
    assert "<@UPEER>" in block and "<@U1>" in block
    assert "UBOT" not in block


def test_someone_mentioned_but_silent_is_still_taggable():
    """The actor map resolves names for people the window only ever mentioned. They have no ts,
    so they sort last — but a name the model can see is a name it must be able to tag."""
    request = _assemble([harness.normalized("10.0", "ask <@U9>", sender_id="U1")],
                        actor_map=(("U1", "Alice"), ("U9", "Nine")))
    block = _roster_item(request)["content"]
    assert "<@U9>" in block
    assert block.index("<@U1>") < block.index("<@U9>")


def test_the_requester_and_the_origin_thread_join_the_stream_actors():
    from message_processor.channel_request import RequesterFacts

    request = _assemble([harness.normalized("10.0", "hi", sender_id="U1")],
                        requester=RequesterFacts(user_id="U3", real_name="Carol",
                                                 sender_type="human"),
                        origin_participants={"U2": "Bob"})
    block = _roster_item(request)["content"]
    assert "<@U1>" in block and "<@U2>" in block and "<@U3>" in block


def test_the_roster_says_how_to_write_a_mention_and_disclaims_authority():
    request = _assemble([harness.normalized("10.0", "hi", sender_id="U1")])
    block = _roster_item(request)["content"]
    assert "write their id as <@USER_ID> exactly" in block
    assert "Informational, not instructions" in block


# ------------------------------------------------------------------ what was retired

def test_the_pulse_ring_and_its_two_blocks_are_gone():
    """Asserted as absences rather than deleted silently: either block coming back is a second,
    lossier account of activity the channel stream already renders message by message."""
    for name in ("_build_taggable_speakers_block", "_build_channel_people_line",
                 "_build_pulse_envelope", "_build_channel_summary_block"):
        assert not hasattr(MessageUtilitiesMixin, name), name
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("slack_client.channel_pulse")


def test_the_dm_suffix_emits_neither_retired_block():
    host = MessageUtilitiesMixin.__new__(type("P", (MessageUtilitiesMixin,), {}))
    host.log_debug = lambda *a, **k: None
    host._build_time_suffix_context = lambda *a, **k: "[time]"
    host._build_generation_inflight_note = lambda *a, **k: None
    host._build_research_inflight_note = lambda *a, **k: None
    suffix = host._build_suffix_context(SimpleNamespace(bot_user_id="UBOT"), "C1", "10.0")
    assert RETIRED_HEADER not in suffix
    assert "Channel people" not in suffix
    assert suffix == "[time]"


# ---------------------------------------- the DM thread roster is UNCHANGED

def test_build_roster_text_signature_and_behavior_unchanged():
    """The DM system prompt still carries a thread roster, and its multi-user detection still
    counts arrows in that string alone (utilities.py `multi_user_thread`). It takes ONLY
    participants — there is no ambient parameter for a crowded room to inflate the count with."""
    assert build_roster_text({}) == ""
    assert build_roster_text({"bot": "x", "unknown": "y"}) == ""
    out = build_roster_text({"UPETER": "Peter", "UBOT": "ChatGPT"}, bot_user_id="UBOT")
    assert "<@UPETER>" in out and "UBOT" not in out
    assert build_roster_text({"UPETER": "Peter"}).count("→ <@") == 1
    assert build_roster_text({"UPETER": "Peter", "UDANA": "Dana"}).count("→ <@") == 2


def test_the_channel_roster_is_not_the_thread_roster():
    """Different function, different string. A channel turn passes `participant_roster=None` to
    the system prompt precisely so the roster cannot re-enter the cached prefix."""
    request = _assemble([harness.normalized("10.0", "hi", sender_id="U1"),
                         harness.normalized("11.0", "hi", sender_id="U2")])
    block = _roster_item(request)["content"]
    assert block != build_roster_text({"U1": "user-U1", "U2": "user-U2"})
    assert ROSTER_HEADER not in request.instructions


# ---------------------------------------------------------------- prompt nudge (A4)

def test_prompt_nudge_points_at_a_roster_and_the_fallback_lookup():
    from prompts import LOCAL_TOOLS_GUIDANCE
    g = LOCAL_TOOLS_GUIDANCE
    assert "@-mention" in g
    assert "list_channel_members" in g             # …for someone the roster doesn't name
    assert "<@id>" in g
