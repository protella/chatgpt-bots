"""The restraint contract, rewritten for full channel visibility (spec §9, P2).

The model now reads ONE stream containing every thread in the channel. Everything that used to be
enforced by what it couldn't see is now enforced only by these paragraphs, so each clause below is
pinned individually — and each is pinned because something went wrong once without it:

  * F47 — it stepped into an exchange between two other people. That protection used to be
    structural (it never saw the exchange). Now it is prose.
  * the value floor — silence is free, a reply that adds nothing is not.
  * the trigger — "the latest message" read against a whole-channel stream means "whatever is
    newest in the channel", which is usually somebody else's message.
  * the terminal contract — exactly one ending, and silence that doesn't cancel work in flight.
  * the sticky hand-off — thread-scoped, or it would follow a person across the whole room.

The scenario suite (tests/integration/test_participation_scenarios.py) proves the BEHAVIOUR; this
file proves the words are still there to prove it with.

P3 added the second half: cross-thread conduct and let-the-exchange-end (spec §9). Those words are
what makes the channel surface DIFFER from the DM surface, so the tests below are as much about
what the channel text no longer says — the origin-acknowledgment instruction that contradicts the
channel schema — as about what it now does.
"""
from __future__ import annotations

import pytest

import prompts
from config import config

from message_processor.utilities import (SURFACE_CHANNEL,
                                        local_tools_guidance_for,
                                        reach_tools_for)
from prompts import (CHANNEL_ACTIVITY_NO_REPLY_SUFFIX, CHANNEL_CROSS_THREAD_CONDUCT_SUFFIX,
                     CHANNEL_LOCAL_TOOLS_GUIDANCE, CHANNEL_POST_TO_THREAD_DESCRIPTION,
                     CHANNEL_POST_TO_THREAD_TARGET_DESCRIPTION, DESTINATION_CONTRACT_SUFFIX,
                     LOCAL_TOOLS_GUIDANCE, TAGGABLE_ROSTER_HEADING,
                     THREAD_ACTIVITY_NO_REPLY_SUFFIX, TURN_COORDINATES_HEADING)

BOTH = (CHANNEL_ACTIVITY_NO_REPLY_SUFFIX, THREAD_ACTIVITY_NO_REPLY_SUFFIX)


# --------------------------------------------------------------- the stream is not an invitation

def test_the_channel_paragraph_says_the_stream_is_the_room_not_an_invitation():
    s = CHANNEL_ACTIVITY_NO_REPLY_SUFFIX
    assert "The stream is the room, not an invitation." in s
    assert "whole channel" in s and "every thread in it" in s


def test_the_channel_paragraph_keeps_the_F47_dont_step_into_their_exchange_rule():
    """The scar: two people working something out, and the bot answering into it because it could
    see it. Nothing structural stops that any more."""
    s = CHANNEL_ACTIVITY_NO_REPLY_SUFFIX
    assert "reading their exchange is not being asked to join it" in s
    assert "stepping in because you happened to see it" in s


def test_the_thread_paragraph_says_visibility_elsewhere_is_context_not_licence():
    s = THREAD_ACTIVITY_NO_REPLY_SUFFIX
    assert "You can read the rest of the channel too" in s
    assert "does not make an exchange between other people elsewhere your business" in s


def test_neither_paragraph_apologizes_for_being_there():
    """The old opening — "You joined this conversation uninvited" — framed every unaddressed turn
    as a social intrusion. The honest bar is not apology, it is worth; it stays deleted."""
    for s in BOTH:
        assert "uninvited" not in s


# ----------------------------------------------------------------------------- the value floor

def test_the_value_floor_survives_in_both_paragraphs():
    assert "Silence is the DEFAULT here" in CHANNEL_ACTIVITY_NO_REPLY_SUFFIX
    for s in BOTH:
        assert "could not easily get themselves" in s


def test_the_honesty_escape_survives_in_the_channel_paragraph():
    """C1: a turn whose only honest answer is "I don't know" stays silent — but a real answer is
    not suppressed just because it admits a limitation, and a named person gets an answer."""
    s = CHANNEL_ACTIVITY_NO_REPLY_SUFFIX
    assert 'consist only of "I haven\'t tried it,"' in s
    assert "do not suppress a substantive answer merely because it includes a limitation" in s
    assert "addressed by name, prefer a brief honest answer over silence" in s


# ----------------------------------------------------- the trigger, not "whatever is newest"

def test_both_paragraphs_identify_the_trigger_through_the_coordinates_block():
    for s in BOTH:
        assert "coordinates block" in s
    assert "the trigger identified in the coordinates block" in CHANNEL_ACTIVITY_NO_REPLY_SUFFIX
    assert "the trigger identified there" in THREAD_ACTIVITY_NO_REPLY_SUFFIX


def test_the_channel_paragraph_denies_the_globally_newest_reading():
    s = CHANNEL_ACTIVITY_NO_REPLY_SUFFIX
    assert "This turn is about ONE message" in s
    assert '"the latest message" mean here' in s
    assert "never whatever happens to be newest in the channel" in s


def test_the_thread_paragraph_names_the_thread_rather_than_assuming_one():
    s = THREAD_ACTIVITY_NO_REPLY_SUFFIX
    assert "the thread identified in the coordinates block" in s
    assert "check its addressee yourself" in s


def test_the_coordinates_pointer_names_the_block_the_builder_actually_emits():
    """A cross-reference to a heading that no longer exists is worse than none: the model is told
    to consult something absent. The paragraphs and the builder share one constant."""
    from message_processor.utilities import TurnCoordinates, build_coordinates_suffix

    block = build_coordinates_suffix(TurnCoordinates(channel_id="C1", trigger_ts="1.0"))
    assert block.startswith(TURN_COORDINATES_HEADING)
    assert "coordinates" in TURN_COORDINATES_HEADING.lower()


def test_the_tools_etiquette_names_the_roster_block_that_actually_exists():
    """Same failure, one file over: the etiquette told the model its taggable ids came from
    "RECENT CHANNEL SPEAKERS" and the "Channel people" line for a while after P2 retired both
    blocks — a pointer at nothing. The prose and the builder now share one constant."""
    from message_processor.utilities import StreamActor, build_taggable_roster_evidence

    label = TAGGABLE_ROSTER_HEADING.lstrip("[")
    assert label in LOCAL_TOOLS_GUIDANCE
    block = build_taggable_roster_evidence([StreamActor("U1", "Alice", "human", "1.0")])
    assert block.startswith(TAGGABLE_ROSTER_HEADING)


def test_no_prompt_or_tool_description_points_at_a_retired_block():
    import pathlib

    from message_processor import people_tools

    root = pathlib.Path(__file__).resolve().parents[2]
    sources = {
        "prompts.py": (root / "prompts.py").read_text(encoding="utf-8"),
        "people_tools.py": (root / "message_processor/people_tools.py").read_text(
            encoding="utf-8"),
        "lookup_user schema": str(people_tools.get_lookup_user_schema()),
    }
    for where, text in sources.items():
        for retired in ("RECENT CHANNEL SPEAKERS", "Channel people", "people line"):
            assert retired not in text, f"{where} still points at the retired {retired!r} block"


# ------------------------------------------------------------------- the sticky hand-off

def test_the_hand_off_sticks_and_sticks_only_inside_that_thread():
    """It exists because of a live failure: the model recognized a message was for someone else
    and said so out loud, which is words about not saying words. Under one whole-channel stream an
    unqualified hand-off would read as a fact about that person everywhere they speak."""
    s = THREAD_ACTIVITY_NO_REPLY_SUFFIX
    assert "That hand-off STICKS, and it sticks inside THAT thread" in s
    assert "an unnamed follow-up in the same thread" in s
    assert 'every bare "you" still means them' in s
    assert "names or @-mentions you again in that thread" in s


def test_the_no_placeholder_rule_survives():
    s = THREAD_ACTIVITY_NO_REPLY_SUFFIX
    assert "NEVER post a placeholder" in s
    assert "silence means silence" in s


# -------------------------------------------------------------- the terminal-action contract

def test_the_channel_paragraph_still_demands_exactly_one_ending():
    s = CHANNEL_ACTIVITY_NO_REPLY_SUFFIX
    assert "End your turn with exactly one of" in s
    assert "react_to_message with empty text" in s
    assert "no_response_needed" in s


def test_silence_ends_words_not_work_in_both_paragraphs():
    for s in BOTH:
        assert "ends your words, not your other actions" in s
        assert "anything else you do this round still happens" in s
        assert "wait for work you started yourself" in s


def test_both_paragraphs_are_one_bracketed_block():
    for s in BOTH + (DESTINATION_CONTRACT_SUFFIX,):
        assert s.startswith("[") and s.endswith("]")
        assert "[" not in s[1:] and "]" not in s[:-1]     # nothing can close the frame early


def test_the_two_paragraphs_are_not_the_same_text():
    assert CHANNEL_ACTIVITY_NO_REPLY_SUFFIX != THREAD_ACTIVITY_NO_REPLY_SUFFIX


def test_neither_paragraph_restates_the_silence_vocabulary():
    """The eight values live in the tool schema. Two copies of one vocabulary drift."""
    from message_processor.terminal_actions import SILENCE_REASONS

    for s in BOTH:
        assert not [v for v in SILENCE_REASONS if "_" in v and v in s]


# ------------------------------------------------------------------ the destination contract

def test_the_destination_contract_names_the_origin_thread():
    """"Under the message" needed an antecedent once the stream carries every thread."""
    s = DESTINATION_CONTRACT_SUFFIX
    assert "the trigger identified in the coordinates block" in s
    assert "the origin thread, where your reply lands by default" in s


def test_the_destination_contract_keeps_its_thread_default_and_single_call():
    s = DESTINATION_CONTRACT_SUFFIX
    assert "call set_reply_destination exactly once" in s
    assert "If it is a close call, choose thread" in s


def test_the_destination_contract_cannot_be_read_as_reaching_a_third_thread():
    s = DESTINATION_CONTRACT_SUFFIX
    assert "never sends your reply into a different thread" in s


# ------------------------------------------------------------------ let the exchange end

def test_both_restraint_paragraphs_let_the_exchange_end():
    """The field observation (Docs/internal/CLAUDE_TAG_WAKE_STUDY.md §d7): it keeps answering while
    it is the one being asked and stops the moment the thread is the room's again. A GENERAL
    principle — no rule about thanks, or musings, or any other shape of message."""
    for s in BOTH:
        assert "An exchange you were part of is allowed to end" in s
        assert "Keep answering while you are the one being asked" in s
        assert "the moment the thread is the room's again" in s


def test_the_concede_once_line_is_in_both_paragraphs():
    """§d9: it concedes in one line and does not chase the last word. The failure it guards is a
    correction answered, then defended, then defended again."""
    for s in BOTH:
        assert "concede once and go quiet" in s
        assert "Never work to keep the last word." in s


def test_the_two_paragraphs_cannot_say_it_differently():
    """One text, carried by both. Two copies of a principle this soft would diverge on the next
    edit and nothing would notice — the paragraphs read fine either way."""
    from prompts import _LET_THE_EXCHANGE_END

    for s in BOTH:
        assert _LET_THE_EXCHANGE_END in s


def test_letting_the_exchange_end_names_no_scenario():
    """OWNER RULING (2026-07-29): no scenario-specific rules in the prompts. The principle may
    describe what a landed exchange looks like; it may not enumerate cases and prescribe a reply
    for each, and it may never say silence is compulsory — "if the conversation ended, great; if
    more needs to be said, great"."""
    from prompts import _LET_THE_EXCHANGE_END

    lowered = _LET_THE_EXCHANGE_END.lower()
    for banned in ("no_response_needed", "must not", "never reply", "always react",
                   "praise", "banter", "compliment"):
        assert banned not in lowered, f"the principle prescribes a case: {banned!r}"
    # And it stays a principle about worth, not an instruction to go silent: the reaction-or-
    # nothing option is offered, never imposed.
    assert "lands fine with a reaction or with nothing" in _LET_THE_EXCHANGE_END


# ------------------------------------------------------------ the cross-thread conduct paragraph

CONDUCT = CHANNEL_CROSS_THREAD_CONDUCT_SUFFIX


def test_the_conduct_paragraph_says_closing_a_loop_is_legitimate():
    """The paragraph has to license the act before it constrains it. A page of prohibitions with
    no permission in it produces a model that will not use the tool at all — which was the state
    the canvas tools were in before F36."""
    assert "legitimate and needs no apology" in CONDUCT
    assert "closing a loop you were part of elsewhere" in CONDUCT


def test_the_conduct_paragraph_demands_one_post_in_one_thread():
    assert "Post it ONCE, in the ONE thread it belongs in." in CONDUCT
    assert "Never send the same answer into more than one thread" in CONDUCT
    assert "never post it there and then repeat it here as well" in CONDUCT


def test_the_conduct_paragraph_sources_the_target_from_the_stream_labels_only():
    """The executor's allowlist is the set of `thread=<ts>` labels the stream rendered, frozen at
    pin time. A prompt that pointed anywhere else — the coordinates block, a tool result, a ts in
    somebody's message — would be inviting a call the runtime refuses."""
    assert "thread=<ts> in a message header" in CONDUCT
    assert "those labels are the whole list of places you may post" in CONDUCT
    assert "a timestamp quoted inside somebody's message is not one" in CONDUCT


def test_the_conduct_paragraph_rules_out_the_origin_thread_as_a_target():
    """The executor rejects it as `same_thread`, so this is the prompt agreeing with the rail
    rather than letting the model discover it through a failed call."""
    assert "The thread you were triggered in is not a cross-thread target" in CONDUCT


def test_the_conduct_paragraph_forbids_a_preamble_in_the_origin():
    """A preamble cannot be retracted: on the streaming path it is already reaching the room while
    the tool call is still in flight, so "I'll answer her over there" survives a post that fails."""
    assert "Write NOTHING here before you post" in CONDUCT
    assert "cannot be taken back" in CONDUCT


def test_the_conduct_paragraph_says_an_empty_origin_is_a_valid_ending():
    """Both handlers were taught that empty prose after a delivered post is a real terminal rather
    than the bare-empty glitch. This is the half of that contract the MODEL reads."""
    assert "Saying nothing here is the normal ending, not a lapse." in CONDUCT
    assert "If something is owed to them, a reaction carries it" in CONDUCT


def test_the_origin_silence_rule_covers_a_one_word_ack():
    """MEASURED IN. The first draft said the words were spent and 1 of 3 trials posted correctly
    into the right thread and then wrote "Done." in a thread it had been asked not to speak in — it
    read the rule as being about long answers."""
    assert 'not a one-word "done"' in CONDUCT
    assert "no summary, no pointer to it" in CONDUCT
    # The stubborn half: it kept reading the rule as being about the ANSWER and treating "Done." as
    # a different speech act, so confirming the action is named separately.
    assert "Do not report the post either" in CONDUCT
    assert "confirming an action is still speaking in a thread you were asked to stay out of" in (
        CONDUCT)


def test_the_conduct_paragraph_closes_the_side_door():
    """MEASURED IN, and the reason the paragraph has a NEGATIVE half at all. The restraint
    paragraphs say reading a strangers' exchange is not being asked to join it; they were written
    before there was a tool that reaches into it, and 1 of 3 trials used the tool to settle two
    other people's open argument."""
    assert "Two cases, and they are the whole list" in CONDUCT
    assert "Nothing else licenses it" in CONDUCT
    assert "is not a loop of yours to close, however well you could settle it" in CONDUCT
    assert "posting into it reaches further in than speaking here would" in CONDUCT


def test_the_conduct_paragraph_sends_the_answer_where_the_question_is_open():
    """MEASURED IN. Permission on its own is not enough: with only "closing a loop is legitimate",
    2 of 3 trials thanked the person who handed over the missing fact and left the question they
    had asked in another thread unanswered."""
    assert "When the open question is over there, that is where the answer goes" in CONDUCT
    assert "not to whoever happened to hand you the missing piece" in CONDUCT


def test_the_conduct_paragraph_is_one_bracketed_block():
    assert CONDUCT.startswith("[") and CONDUCT.endswith("]")
    assert "[" not in CONDUCT[1:] and "]" not in CONDUCT[:-1]


def test_the_conduct_paragraph_names_no_scenario():
    """Same owner ruling as above. It describes the two general shapes the act takes (asked to
    answer elsewhere / closing your own loop) and never a particular room's situation."""
    lowered = CONDUCT.lower()
    for banned in ("dana", "her thread", "nightly", "for example", "e.g."):
        assert banned not in lowered


# -------------------------------------------------- no contradictory origin-ack on this surface

ORIGIN_ACK_PHRASES = ("acknowledge briefly here", "just acknowledge briefly",
                      "Acknowledge briefly in the current thread")


def test_the_channel_etiquette_drops_the_bullet_that_contradicts_the_schema():
    """THE CONTRADICTION (plan §1c): the DM etiquette says post there and acknowledge HERE, and the
    channel schema and conduct paragraph say post there and say nothing here. Two instructions, one
    tool, opposite endings — and the cached one wins arguments it should not."""
    assert "post_to_thread" in LOCAL_TOOLS_GUIDANCE, "the DM bullet is the thing being removed"
    assert "post_to_thread" not in CHANNEL_LOCAL_TOOLS_GUIDANCE


def test_the_channel_etiquette_is_the_dm_etiquette_minus_exactly_that_bullet():
    """Derived, not copied — and this is the assertion that keeps the derivation honest. Renaming
    the DM bullet would make the filter a no-op and quietly restore the contradiction; a divergence
    anywhere else means somebody edited one surface's etiquette and not the other's."""
    dm_lines = LOCAL_TOOLS_GUIDANCE.split("\n")
    channel_lines = CHANNEL_LOCAL_TOOLS_GUIDANCE.split("\n")
    removed = [line for line in dm_lines if line not in channel_lines]
    assert len(removed) == 1, f"expected exactly one bullet to differ, got {len(removed)}"
    assert removed[0].startswith("- post_to_thread:")
    assert channel_lines == [line for line in dm_lines if line != removed[0]]


def test_no_origin_ack_instruction_survives_anywhere_on_the_channel_surface():
    """All three places the channel surface can speak about the tool: the cached etiquette, the
    post-breakpoint conduct paragraph, and the tool schema."""
    surfaces = {
        "channel etiquette": CHANNEL_LOCAL_TOOLS_GUIDANCE,
        "conduct paragraph": CONDUCT,
        "channel schema description": CHANNEL_POST_TO_THREAD_DESCRIPTION,
        "channel target description": CHANNEL_POST_TO_THREAD_TARGET_DESCRIPTION,
    }
    for where, text in surfaces.items():
        for phrase in ORIGIN_ACK_PHRASES:
            assert phrase not in text, f"{where} still asks for an origin acknowledgment"


def test_the_dm_etiquette_keeps_its_origin_ack_bullet():
    """The DM surface is not being fixed here. It has one thread, `post_to_thread` reaches a
    different conversation entirely, and acknowledging where you were asked is right there."""
    assert "just acknowledge briefly here" in LOCAL_TOOLS_GUIDANCE


# ------------------------------------------------------------------ the channel tool schema

def _post_to_thread_schema(surface: str) -> dict:
    from unittest.mock import MagicMock

    from slack_client.messaging import SlackMessagingMixin

    host = MagicMock()
    host._POST_TO_THREAD_DESCRIPTION = SlackMessagingMixin._POST_TO_THREAD_DESCRIPTION
    host._POST_TO_THREAD_TARGET_DESCRIPTION = (
        SlackMessagingMixin._POST_TO_THREAD_TARGET_DESCRIPTION)
    host._post_to_thread_schema = SlackMessagingMixin._post_to_thread_schema
    getter = (SlackMessagingMixin.get_post_to_thread_channel_schema if surface == "channel"
              else SlackMessagingMixin.get_post_to_thread_tool_schema)
    return getter.__get__(host)()


def test_the_channel_schema_now_carries_the_channel_words():
    channel = _post_to_thread_schema("channel")
    assert channel["description"] == CHANNEL_POST_TO_THREAD_DESCRIPTION
    assert (channel["parameters"]["properties"]["thread_ts"]["description"]
            == CHANNEL_POST_TO_THREAD_TARGET_DESCRIPTION)


def test_the_dm_schema_is_untouched_by_the_channel_wording():
    from slack_client.messaging import SlackMessagingMixin

    dm = _post_to_thread_schema("dm")
    assert dm["description"] == SlackMessagingMixin._POST_TO_THREAD_DESCRIPTION
    assert (dm["parameters"]["properties"]["thread_ts"]["description"]
            == SlackMessagingMixin._POST_TO_THREAD_TARGET_DESCRIPTION)
    assert dm != _post_to_thread_schema("channel")


def test_the_channel_schema_promises_only_what_the_executor_allows():
    """The DM text says a target may come "from a tool". On a channel turn the allowlist is the
    stream's own labels and a tool-supplied root is REFUSED, so the DM promise would teach the
    model that a working tool is broken. NARROWED (plan §1d)."""
    from slack_client.messaging import SlackMessagingMixin

    assert "or from a tool" in SlackMessagingMixin._POST_TO_THREAD_TARGET_DESCRIPTION
    assert "from a tool" not in CHANNEL_POST_TO_THREAD_TARGET_DESCRIPTION
    assert "this channel's stream labels it" in CHANNEL_POST_TO_THREAD_TARGET_DESCRIPTION


def test_the_channel_schema_and_the_conduct_paragraph_agree_about_the_origin():
    """One rule, two places it is said, and they have to end the same way: the answer goes in the
    target ONCE and is not repeated where the turn started."""
    assert "is not repeated where you are now" in CHANNEL_POST_TO_THREAD_DESCRIPTION
    assert "The thread you were triggered in is not a target" in (
        CHANNEL_POST_TO_THREAD_TARGET_DESCRIPTION)


def test_the_allowlist_the_prompt_describes_is_the_one_the_stream_computes():
    """The paragraph tells the model the labels are the list. This asserts the list really IS the
    labels — a real serialized stream, and the roots it authorizes are exactly the `thread=` labels
    it rendered. If `trusted_thread_roots` ever widened, the prompt would be describing a stricter
    tool than the one the model has."""
    from tests.unit.channel_turn_harness import build_stream, normalized

    stream = build_stream([
        normalized("10.0", "a question"),
        normalized("11.0", "an answer", thread_root_ts="10.0"),
        normalized("30.0", "a top-level line nobody replied to"),
    ])
    rendered = "\n".join(item.content for item in stream.message_items)
    labelled = {chunk.split()[0].split("~")[0].rstrip("]")
                for chunk in rendered.split("thread=")[1:]}
    assert labelled == {"10.0"}
    assert stream.trusted_thread_roots == frozenset({"10.0"})


# --------------------------------------------------- where the paragraph rides, and what it costs

def _materialize(surface, *, silence_capable=True, tools_disabled=False, registry=None,
                 destination_open=True):
    """The real `_materialize_request_tools`, which is what decides whether a paragraph rides.

    Driven with a REAL turn and the destination tool registered, because the paragraph the order
    is about only exists on a turn that still has a destination to choose. The earlier version of
    this helper passed `turn=None`, which made `destination_open` false, dropped
    DESTINATION_CONTRACT out of every composition, and left the order tests grading a two-item
    list — the one arrangement in which the arrangement could not be wrong.
    """
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock, patch

    from config import config
    from message_processor.destination_tools import register_destination_tools
    from message_processor.handlers.text import TextHandlerMixin
    from message_processor.turn_runtime import TurnRuntime
    from tool_registry import SURFACE_CHANNEL, ToolRegistry

    if registry is None:
        registry = ToolRegistry()
        registry.register({"type": "function", "name": "post_to_thread", "parameters": {}},
                          AsyncMock())
        registry.register({"type": "function", "name": "no_response_needed", "parameters": {}},
                          AsyncMock(),
                          enabled=lambda cfg: bool(cfg.get("_silence_capable_turn")),
                          channel_enabled=lambda cfg: True)
        register_destination_tools(registry)
    host = MagicMock()
    host._materialize_request_tools = TextHandlerMixin._materialize_request_tools.__get__(host)
    host._get_tool_registry = TextHandlerMixin._get_tool_registry.__get__(host)
    client = MagicMock()
    client.tool_registry = registry
    message = SimpleNamespace(
        channel_id="C1" if surface == SURFACE_CHANNEL else "D1",
        metadata={"ts": "10.0", "sender_type": "human", "silence_capable": silence_capable})
    turn = TurnRuntime()
    turn.destination_selected = not destination_open
    with patch.object(config, "enable_tool_loop", True), \
         patch.object(config, "enable_no_reply_tool", True):
        _reg, _cfg, _no_reply, contract = host._materialize_request_tools(
            client, {"model": "m"}, message, tools_disabled=tools_disabled, turn=turn,
            surface=surface)
    return contract or ""


def test_the_conduct_paragraph_reaches_an_addressed_channel_turn():
    """The case it is about — "answer her over there, not here" — arrives ADDRESSED as often as
    not, and an addressed turn gets NO restraint suffix (there is nothing to restrain). So the
    paragraph is keyed to the tool, not to a posture, and this is the half that would be lost if
    somebody moved it into the restraint suffixes where it looks like it belongs."""
    from tool_registry import SURFACE_CHANNEL

    contract = _materialize(SURFACE_CHANNEL, silence_capable=False)
    assert CONDUCT in contract
    assert CHANNEL_ACTIVITY_NO_REPLY_SUFFIX not in contract
    assert THREAD_ACTIVITY_NO_REPLY_SUFFIX not in contract
    # ADDRESSED and cross-thread: no restraint paragraph exists, so conduct is what the request
    # ends on — and it must not be followed by the destination contract telling the model to
    # answer HERE after conduct has just said to answer over there.
    assert DESTINATION_CONTRACT_SUFFIX in contract
    assert contract.index(DESTINATION_CONTRACT_SUFFIX) < contract.index(CONDUCT)
    assert contract.endswith(CONDUCT)


def test_the_conduct_paragraph_also_reaches_a_silence_capable_channel_turn():
    """And the three paragraphs narrow, in this order: destination (where my own reply goes),
    conduct (if the answer belongs in another thread, how it gets there), restraint (whether to
    speak at all).

    The pairing that fixed the order: DESTINATION_CONTRACT ends "call set_reply_destination, then
    answer", and conduct says to post over there and write nothing here. Whichever is last is the
    one the model reads as the standing instruction, and the contract was answering a question
    conduct had already closed. Restraint keeps the closing position wherever it rides."""
    from tool_registry import SURFACE_CHANNEL

    contract = _materialize(SURFACE_CHANNEL, silence_capable=True)
    assert CHANNEL_ACTIVITY_NO_REPLY_SUFFIX in contract and CONDUCT in contract
    assert DESTINATION_CONTRACT_SUFFIX in contract
    assert (contract.index(DESTINATION_CONTRACT_SUFFIX) < contract.index(CONDUCT)
            < contract.index(CHANNEL_ACTIVITY_NO_REPLY_SUFFIX))
    assert contract.endswith(CHANNEL_ACTIVITY_NO_REPLY_SUFFIX)
    assert CONDUCT + "\n\n" in contract


def test_a_turn_that_cannot_post_cross_thread_is_never_told_how_to():
    """Two ways the tool goes missing, and neither may leave the instruction behind — an attempt
    that names an unavailable tool is how a timeout retry ends in a refused tool call. The cached
    etiquette cannot make this mistake at all: it no longer mentions the tool."""
    from unittest.mock import AsyncMock

    from tool_registry import SURFACE_CHANNEL, ToolRegistry

    empty = ToolRegistry()
    empty.register({"type": "function", "name": "react_to_message", "parameters": {}}, AsyncMock())
    assert CONDUCT not in _materialize(SURFACE_CHANNEL, tools_disabled=True)
    assert CONDUCT not in _materialize(SURFACE_CHANNEL, registry=empty)
    assert "post_to_thread" not in CHANNEL_LOCAL_TOOLS_GUIDANCE


def test_the_conduct_paragraph_never_rides_a_dm_turn():
    """The tool IS in this registry's DM schema set, so what is under test is the SURFACE check and
    nothing else. A DM has no other thread of its own, and the channel-wide conduct paragraph has
    no business in its prompt."""
    from tool_registry import SURFACE_DM

    assert CONDUCT not in _materialize(SURFACE_DM)


def _assembled(contract_suffix):
    from unittest.mock import MagicMock

    from message_processor import channel_request
    from tests.unit.channel_turn_harness import (build_stream, normalized, thread_config)
    from message_processor.utilities import MessageUtilitiesMixin

    host = MagicMock()
    host._build_time_suffix_context = (
        MessageUtilitiesMixin._build_time_suffix_context.__get__(host))
    host._build_generation_inflight_note = MagicMock(return_value=None)
    host._build_research_inflight_note = MagicMock(return_value=None)
    host._get_system_prompt = MagicMock(return_value="SYSTEM")
    stream = build_stream([normalized("10.0", "a question"),
                           normalized("11.0", "an answer", thread_root_ts="10.0")])
    ctx = channel_request.ChannelTurnContext(
        stream=stream, steering=None, thread_config=thread_config(), channel_id="C1",
        team_id="T1", trigger_ts="11.0", origin_thread_ts="10.0", trigger_text="a question",
        requester=channel_request.RequesterFacts(user_id="U1", real_name="Alice",
                                                sender_type="human"))
    return channel_request.assemble_channel_request(
        processor=host, client=MagicMock(), ctx=ctx, model="gpt-5.6-sol", tools=None,
        request_config=thread_config(), contract_suffix=contract_suffix, with_estimate=True)


def test_the_conduct_paragraph_lands_after_the_breakpoint():
    """It is VOLATILE — present only on the turns that expose the tool — so it must never sit in
    the shared prefix. One paragraph above the line and every channel's prefix cache splits by
    whether the tool happened to be on."""
    request = _assembled(CONDUCT)
    items = request.input_items
    breakpoint_index = next(
        index for index, item in enumerate(items)
        if isinstance(item.get("content"), list)
        and any(isinstance(p, dict) and p.get("prompt_cache_breakpoint")
                for p in item["content"]))
    carriers = [index for index, item in enumerate(items)
                if isinstance(item.get("content"), str) and CONDUCT in item["content"]]
    assert carriers, "the conduct paragraph reached no item at all"
    assert min(carriers) > breakpoint_index


def test_the_conduct_paragraph_is_charged_by_admission():
    """Every byte a turn sends is measured before the turn is allowed to run. A paragraph the
    estimate did not see is a paragraph that can push a request past the window after it was
    admitted — the one failure mode admission exists to prevent."""
    with_conduct = _assembled(CONDUCT)
    without = _assembled(None)
    assert with_conduct.estimate.total_tokens > without.estimate.total_tokens
    assert CONDUCT in with_conduct.input_items[-1]["content"]


# -------------------------------------------------------------- the two byte-identity guarantees

def test_the_dm_system_prompt_carries_the_dm_etiquette_verbatim():
    """The DM surface must not move by a byte. Its etiquette is the whole DM block, the channel
    text is nowhere in it, and none of the four channel constants leak across."""
    from unittest.mock import MagicMock

    from message_processor.utilities import MessageUtilitiesMixin
    from tool_registry import SURFACE_CHANNEL, SURFACE_DM

    proc = MessageUtilitiesMixin.__new__(type("P", (MessageUtilitiesMixin,), {}))
    client = MagicMock()
    client.name = "Slack"
    dm = proc._get_system_prompt(client, tool_surface=SURFACE_DM, tools_available=True,
                                 channel_steering=None)
    channel = proc._get_system_prompt(client, tool_surface=SURFACE_CHANNEL, tools_available=True,
                                      channel_steering=None)
    assert LOCAL_TOOLS_GUIDANCE in dm
    assert "post_to_thread" in dm
    assert CHANNEL_LOCAL_TOOLS_GUIDANCE in channel and "post_to_thread" not in channel
    for volatile in (CONDUCT, CHANNEL_POST_TO_THREAD_DESCRIPTION,
                     CHANNEL_POST_TO_THREAD_TARGET_DESCRIPTION):
        assert volatile not in dm and volatile not in channel


def test_the_gate_reads_exactly_what_it_read_before():
    """The gate decides one bit and P3 changed nothing about it. Two ways that could have slipped:
    a word of cross-thread vocabulary in the classifier prompt, or the steering snapshot's
    `gate_text` drifting from the flattened text the responder gets. Both are asserted rather than
    assumed, because the gate is the one surface in this file nobody would think to re-record."""
    from message_processor.channel_steering import render_snapshot
    from prompts import WAKE_CLASSIFIER_SYSTEM_PROMPT

    for word in ("post_to_thread", "cross-thread", "thread=<ts>", "last word"):
        assert word not in WAKE_CLASSIFIER_SYSTEM_PROMPT

    snapshot = render_snapshot({"content": "Only reply when tagged."},
                               [{"id": 1, "content": "Deploys go through #ops.",
                                 "scope": "channel"}])
    assert snapshot.gate_text == snapshot.text
    assert "Only reply when tagged." in snapshot.gate_text


# ================================================= the window guidance (§2i, Appendix A6)
#
# A shallow window is only safe if the model knows it is looking at one. The bytes declare the
# edges; these words are what the model is told to DO about them — and every requirement here is
# channel-stable, so all of it rides the cached pre-breakpoint instructions and costs nothing per
# turn.

REACH_SUBSETS = [
    (),
    ("search_slack",),
    ("fetch_channel_history", "fetch_thread_messages"),
    ("search_slack", "fetch_channel_history", "fetch_thread_messages"),
]


def test_the_verification_rule_states_both_halves():
    """T60. The two halves do DIFFERENT jobs and a prompt carrying only the first is not
    compliant.

    The first stops a confident wrong quote. The second stops the far more likely failure — "I
    don't see any discussion of X here" said about a channel that discussed X at length last
    week, which is a false statement about the world delivered in the voice of someone who
    checked. Grepping for one and calling it done is exactly how the half that matters gets
    dropped in a later edit, which is why both are asserted separately.
    """
    for reach in REACH_SUBSETS:
        text = prompts.render_window_guidance(reach)
        assert "be able to point at the message that says it" in text, reach
        assert "never evidence that it did not happen" in text, reach
        assert "a claim about your view, not about the room" in text, reach


def test_the_middle_sentence_needs_a_tool_to_be_honest():
    """The go-and-read sentence is omitted when there is nothing to read WITH. Instructing the
    model to reach for a tool it does not have is worse than naming none: it then reports a
    failed tool call as an answer."""
    assert prompts.WINDOW_GUIDANCE_VERIFY_FETCH in prompts.render_window_guidance(
        ("search_slack",))
    assert prompts.WINDOW_GUIDANCE_VERIFY_FETCH not in prompts.render_window_guidance(())
    # …but the two halves that need no tool still ride.
    empty = prompts.render_window_guidance(())
    assert prompts.WINDOW_GUIDANCE_VERIFY_HEAD in empty
    assert prompts.WINDOW_GUIDANCE_VERIFY_TAIL in empty


@pytest.mark.parametrize("reach", REACH_SUBSETS)
def test_the_channel_guidance_is_derived_plus_rendered_window_text(reach):
    """T61. The tripwire, EXTENDED to the composition: channel guidance is (the DM text minus
    exactly one bullet) PLUS the rendered window text, byte for byte, for every reach subset.

    There is no static `CHANNEL_WINDOW_GUIDANCE` string to append — a test asserting one fails by
    construction, which is the point: the words and the interpolation arrive together, so there
    is no empty-constant no-op state to land the plumbing behind.
    """
    assert not hasattr(prompts, "CHANNEL_WINDOW_GUIDANCE")
    composed = local_tools_guidance_for(SURFACE_CHANNEL, reach)
    assert composed == (CHANNEL_LOCAL_TOOLS_GUIDANCE + "\n\n"
                        + prompts.render_window_guidance(reach))
    # The derived constant itself is UNCHANGED by the composition.
    assert CHANNEL_LOCAL_TOOLS_GUIDANCE in composed
    assert "post_to_thread" not in CHANNEL_LOCAL_TOOLS_GUIDANCE


def test_the_dm_surface_gains_nothing():
    """RULING-8. A DM has no window and no periphery; adding window guidance there would change
    DM bytes for no reason. The default argument exists so the signature stays call-compatible,
    not so the DM branch renders anything new."""
    assert local_tools_guidance_for("dm") == LOCAL_TOOLS_GUIDANCE
    assert local_tools_guidance_for("dm", ()) == LOCAL_TOOLS_GUIDANCE
    assert "window, not the whole room" not in local_tools_guidance_for("dm")


@pytest.mark.parametrize("search_on,history_on,expected", [
    (True, True, ("search_slack", "fetch_channel_history", "fetch_thread_messages")),
    (True, False, ("search_slack",)),
    (False, True, ("fetch_channel_history", "fetch_thread_messages")),
    (False, False, ()),
])
def test_the_named_reach_tools_are_the_exposed_ones(monkeypatch, search_on, history_on,
                                                    expected):
    """T62. The names are DERIVED from the same global switches the registry reads, never
    hard-coded: both default true but both can be off, and a hard-coded list would promise a tool
    the model cannot call.

    With NONE exposed, the reach paragraph and the search-or-fetch clause are both ABSENT.
    """
    monkeypatch.setattr(config, "enable_search_tool", search_on, raising=False)
    monkeypatch.setattr(config, "enable_history_tools", history_on, raising=False)
    assert reach_tools_for() == expected

    text = prompts.render_window_guidance(reach_tools_for())
    for name in ("search_slack", "fetch_channel_history", "fetch_thread_messages"):
        assert (name in text) is (name in expected), name
    if not expected:
        assert "To see past the window" not in text
        assert prompts.WINDOW_GUIDANCE_VERIFY_FETCH not in text
    else:
        assert "To see past the window" in text


def test_the_reach_list_renders_in_a_fixed_order_whatever_the_caller_passed():
    """The rendered order is REACH_TOOLS', not the caller's. Two callers with the same set must
    produce the same cached bytes, or the prefix forks on argument order alone."""
    forward = prompts.render_window_guidance(
        ("search_slack", "fetch_channel_history", "fetch_thread_messages"))
    backward = prompts.render_window_guidance(
        ("fetch_thread_messages", "fetch_channel_history", "search_slack"))
    assert forward == backward
    body = forward.split("To see past the window: ")[1]
    assert body.index("search_slack") < body.index("fetch_channel_history")
    assert body.index("fetch_channel_history") < body.index("fetch_thread_messages")


def test_every_tool_the_prompt_names_is_callable(monkeypatch):
    """T63. For each configuration, every reach name the guidance renders is a name the same
    attempt's schema set actually contains. "The names exist" could not prove this — the point is
    that the name and the SCHEMA agree, and they are decided by two different modules."""
    from slack_client.history_tool import SlackHistoryToolMixin

    class _Host(SlackHistoryToolMixin):
        """The narrowest host the schema builder needs — the REAL method, not a copy of it."""
        bot_user_id = "UBOT"

    host = _Host()

    for search_on, history_on in ((True, True), (True, False), (False, True), (False, False)):
        monkeypatch.setattr(config, "enable_search_tool", search_on, raising=False)
        monkeypatch.setattr(config, "enable_history_tools", history_on, raising=False)

        # The registry's own two sources: `slack_client/base.py` guards the search schema on
        # enable_search_tool, and registers whatever get_history_tools_for_openai returns.
        exposed = set()
        if config.enable_search_tool:
            exposed.add("search_slack")
        exposed.update(schema["name"] for schema in host.get_history_tools_for_openai()
                       if isinstance(schema, dict) and schema.get("name"))

        named = reach_tools_for()
        assert set(named) <= exposed, (
            f"the prompt names {set(named) - exposed} which the registry does not expose "
            f"(search={search_on}, history={history_on})")
        rendered = prompts.render_window_guidance(named)
        for name in ("search_slack", "fetch_channel_history", "fetch_thread_messages"):
            if name in rendered:
                assert name in exposed, f"{name} is named in the prompt but not exposed"

        # AND THROUGH THE COMPOSED INSTRUCTIONS, which is the text the model actually reads.
        # Asserting only on `render_window_guidance(reach_tools_for())` proves the helper agrees
        # with itself: the call site is free to pass nothing and silently take the full default,
        # which is exactly the defect this half exists to catch — the guidance promised all three
        # tools with both switches off, and no test noticed.
        composed = local_tools_guidance_for(SURFACE_CHANNEL, reach_tools_for())
        # SCOPED TO THE WINDOW SEGMENT, deliberately. The base etiquette block names the same
        # tools in CONDITIONAL phrasing ("When search_slack is available…"), which is honest
        # whatever the switches say, and T61 pins that text byte-for-byte as the DM text minus
        # one bullet. The window guidance is the half that makes an UNCONDITIONAL promise —
        # "To see past the window: <tool> does X" — so it is the half that must be derived.
        window_segment = composed[len(CHANNEL_LOCAL_TOOLS_GUIDANCE):]
        for name in ("search_slack", "fetch_channel_history", "fetch_thread_messages"):
            if name in window_segment:
                assert name in exposed, (
                    f"the composed window guidance names {name}, which the registry does not "
                    f"expose (search={search_on}, history={history_on})")
        if not exposed:
            assert "To see past the window" not in window_segment


def test_the_system_prompt_passes_the_derived_reach_tools(monkeypatch):
    """The CALL SITE, not the helper. `_get_system_prompt` builds the channel instructions, and
    it must hand the materializer the DERIVED tuple rather than letting the default ride.

    Read at the source because the composition above cannot see which argument its caller used:
    with both switches on, the derived tuple and the default are the same value, so a call site
    that passes nothing is indistinguishable from a correct one in every assertion but this.
    """
    import inspect

    from message_processor.utilities import MessageUtilitiesMixin

    source = inspect.getsource(MessageUtilitiesMixin._get_system_prompt)
    assert "local_tools_guidance_for(tool_surface, reach_tools_for())" in source, (
        "the system prompt must pass the derived reach tuple; passing only the surface takes "
        "the full default and promises tools the registry may not expose")
