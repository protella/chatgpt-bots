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

import message_processor.prompts as prompts
from config import config

from message_processor.destination_tools import destination_marker
from message_processor.turn_runtime import SELECTABLE_DESTINATIONS
from message_processor.utilities import (SURFACE_CHANNEL,
                                        local_tools_guidance_for,
                                        reach_tools_for)
from message_processor.prompts import (CHANNEL_ACTIVITY_NO_REPLY_SUFFIX, CHANNEL_CROSS_THREAD_CONDUCT_SUFFIX,
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
        assert "Speak only when you are offering something the room does not already have" in s


def test_the_value_floor_is_retrieved_facts_not_capability_in_both_paragraphs():
    """Quiet-by-default, 2026-08-20: the old floor asked only for something the people here
    "could not easily get themselves", which being able to answer already satisfies — so the bot
    raced the room to general questions. The floor is now what it actually holds that they don't."""
    for s in BOTH:
        assert "a fact you actually retrieved or verified with your tools this turn" in s
        assert "an established fact about your own prior words, actions, or tool use" in s
        assert "Being able to answer is never by itself a reason to answer" in s
        assert "an informed colleague could supply is the room's to give, not yours" in s
        # the capability license it replaces is gone from both
        assert "could not easily get" not in s


def test_the_honesty_escape_survives_in_the_channel_paragraph():
    """C1: a turn whose only honest answer is "I don't know" stays silent — but a real answer is
    not suppressed just because it admits a limitation, and a named person gets an answer."""
    s = CHANNEL_ACTIVITY_NO_REPLY_SUFFIX
    assert 'consist only of "I haven\'t tried it,"' in s
    assert "do not suppress a substantive answer merely because it includes a limitation" in s
    assert "being addressed by name outranks this whole test" in s
    assert 'deserves a brief honest answer even when that answer is "I don\'t know."' in s


def test_the_value_floor_is_earned_by_new_grounded_information():
    """The incident the tightening is for: an open ownership question answered with the asker's
    own guess handed back. A non-answer dressed as advice is still a non-answer, and speaking is
    earned by grounding rather than by having a way to respond."""
    s = CHANNEL_ACTIVITY_NO_REPLY_SUFFIX
    assert "a non-answer dressed as advice is still that answer" in s
    assert "new grounded information" in s
    # 2026-08-20: "knowledge you actually hold with confidence" was the same license in the list —
    # a model always holds its general knowledge with confidence. Grounding has to be an act.
    assert "a fact you actually retrieved or verified" in s
    assert "knowledge you actually hold with confidence" not in s
    # 2026-08-20 (again): the clarification route went with it — an inference any informed
    # colleague could draw from the stream is not new grounded information.
    assert "a clarifying connection supported by the stream in front of you" not in s
    assert "a check you made this turn, or a fact you actually retrieved or verified" in s
    # The live loophole the first §7 probe found: an empty search reported as if it were a fact.
    assert "reporting that you searched and found nothing" in s


# ------------------------------------------------------- a reaction is a position, not a comment

def test_the_reaction_endorsement_rule_reaches_both_paragraphs():
    """One constant, both suffixes — the channel variant and the thread variant cannot say it
    differently, and the shape that earned it was a thread reply."""
    for s in BOTH:
        assert prompts._BANTER_RESTRAINT in s


def test_the_open_question_standing_limit_reaches_both_paragraphs():
    """An open question earns the turn, not the ruling. One constant, both suffixes — a thread
    reply can walk into the same judgment the channel variant's exception invited."""
    for s in BOTH:
        assert prompts._OPEN_QUESTION_STANDING in s


def test_the_open_question_standing_limit_states_the_principle():
    """Pinned as a principle, not a topic list: the judgment belongs to whoever is accountable,
    unseen state is not to be guessed, and deferring out loud is still speaking."""
    s = prompts._OPEN_QUESTION_STANDING
    assert "An open question does not confer authority" in s
    assert "belongs to the people accountable for the decision" in s
    assert "current state you cannot see or verify" in s
    assert "do not guess the unseen state or prescribe the ruling" in s
    assert "rather than announcing your deference" in s


def test_the_open_question_exception_is_not_a_capability_license():
    """The exception survives — an unaddressed question is still answerable — but "if you can
    answer it accurately" made capability the bar, which is the whole failure mode. What earns the
    turn is the same retrieved-or-verified fact the value floor asks for."""
    s = CHANNEL_ACTIVITY_NO_REPLY_SUFFIX
    assert "A genuine question put to the room is the exception" in s
    assert "if you can answer it accurately" not in s
    assert "Settle it with a fact you actually retrieved or verified this turn" in s
    assert "what any informed colleague here could say, the answer is theirs to give" in s
    assert "being first with it adds nothing" in s
    # the retrieved-or-verified fact is the ONLY route now — the clarification license is gone
    assert "materially advance it with one useful clarification" not in s
    assert "do that — briefly" in s


def test_the_open_question_standing_limit_wants_a_fact_that_was_actually_retrieved():
    """"A verifiable fact" reads as "a fact I could verify" — general knowledge passes it. The
    contribution clause now names the act, matching the floor in both paragraphs."""
    s = prompts._OPEN_QUESTION_STANDING
    assert "You may contribute a fact you actually retrieved or verified" in s
    assert "verifiable fact" not in s


def test_the_thread_bar_never_overrides_the_addressee_rules_or_an_answer_owed():
    """The mirrored floor sits in a paragraph whose whole first half is about being the one asked.
    Read flat it would silence an addressed follow-up, so it says what it is for."""
    s = THREAD_ACTIVITY_NO_REPLY_SUFFIX
    assert "That bar is for what is genuinely the room's" in s
    assert "never overrides the addressee rules above" in s
    assert "never cuts short an answer or a correction you already owe" in s
    # and the rules it must not outrank are still there, on both sides of it
    assert "check its addressee yourself" in s
    assert "Keep answering while you are the one being asked" in s


def test_the_open_question_standing_limit_separates_product_knowledge_from_deployment_state():
    """Knowing a product well is not seeing how this workplace runs it: configuration, licensing
    and administration are unseen state, and the carve-out keeps what the conversation shows or a
    tool returned usable as evidence."""
    s = prompts._OPEN_QUESTION_STANDING
    assert "General product knowledge is not evidence" in s
    assert "do not state unseen workplace state as fact" not in s
    assert "Do not state unseen workplace state as fact" in s
    assert "your familiarity is not" in s
    # retrieving it this turn does not convert it into evidence about this workplace
    assert "whether you remember it or just retrieved it this turn from public documentation" in s
    assert "CAN be configured is not a fact about how it IS configured here" in s
    assert "a tool actually returned about this workplace" in s
    # the contribution has to advance the question, not the product
    assert "materially advances the question itself" in s
    assert "a fact about the product in general is not that" in s
    # and an unaddressed turn does not get to summon anyone or assign homework
    assert "never the place to @-mention a person into the thread" in s
    assert "hand someone a next step to go check" in s


def test_the_reaction_endorsement_rule_states_the_mechanism_and_its_limbs():
    """The incident: an agreement emoji on a config tip framed as an opinion about a product.
    Both limbs are pinned — factual business in the same message launders nothing, and a
    judgment the bot has no standing to make is not one to take a side on at all."""
    s = prompts._BANTER_RESTRAINT
    assert "it is you taking the message's position in public" in s
    assert "not yours to react to just because part of it is factual" in s
    assert "do not take a side by emoji at all" in s
    assert "agreeing with the insult" in s


def test_the_voice_paragraph_carries_the_standing_rule():
    """It rides the system prompt rather than the restraint paragraph because standing governs
    every surface and every turn, addressed or not."""
    s = prompts.SLACK_SYSTEM_PROMPT
    assert "An opinion needs standing" in s
    assert "never from experiences you do not have" in s
    assert "not yours to endorse or dispute, in words or by reaction" in s


def test_the_endorsement_sentence_survives_the_channel_derivation():
    """CHANNEL_LOCAL_TOOLS_GUIDANCE is a line filter over the DM text; a dropped line would take
    the rule off the channel surface silently."""
    sentence = ("An agreement-shaped reaction on someone's opinion is an endorsement, not an "
                "acknowledgment")
    assert sentence in LOCAL_TOOLS_GUIDANCE
    assert sentence in CHANNEL_LOCAL_TOOLS_GUIDANCE


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
        "prompts.py": (root / "message_processor/prompts.py").read_text(encoding="utf-8"),
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
    for s in BOTH:
        assert s.startswith("[") and s.endswith("]")
        assert "[" not in s[1:] and "]" not in s[:-1]     # nothing can close the frame early


def test_the_destination_paragraph_is_one_block_around_two_markers():
    """Same frame, one licensed exception: W4's markers are literally brackets, so the paragraph
    that teaches them cannot be bracket-free. Nothing ELSE may add one — with the two markers
    removed, the old no-inner-bracket rule still holds exactly."""
    s = DESTINATION_CONTRACT_SUFFIX
    assert s.startswith("[") and s.endswith("]")
    for destination in SELECTABLE_DESTINATIONS:
        s = s.replace(destination_marker(destination), "MARKER")
    assert "[" not in s[1:] and "]" not in s[:-1]


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


def test_the_destination_contract_keeps_its_thread_default_and_asks_for_one_marker():
    """W4 changed HOW the choice is stated — a marker in the first tokens instead of a tool call
    that cost a whole round — and nothing else about the contract. The rubric is the same."""
    s = DESTINATION_CONTRACT_SUFFIX
    assert "begin your reply with exactly" in s
    assert "set_reply_destination" not in s, "the tool is retired on the surface this rides"
    assert "If it is a close call, choose thread" in s


def test_the_destination_contract_excuses_a_turn_with_no_words():
    """A marker belongs to a reply. A turn that ends in silence or a reaction has nothing to
    place, and the paragraph has to say so or the model will invent words to carry one."""
    assert "a turn that ends without any needs none" in DESTINATION_CONTRACT_SUFFIX


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


def test_a_closing_addressed_to_us_is_still_acknowledged():
    """OWNER RULING (2026-08-09): a correction aimed at us closed a bot-to-bot exchange and the
    model answered with nothing at all — the value floor read as license to ignore. Letting the
    exchange end may not become leaving a directed message hanging, and the floor itself is
    unchanged: what nobody put to us still gets silence."""
    for s in BOTH:
        assert "when the message that closes it was put to you, it still earns" in s
        assert "the acknowledgment a teammate would give, and a reaction is enough of one" in s
        assert "Silence is for what nobody put to you." in s


def test_the_two_paragraphs_cannot_say_it_differently():
    """One text, carried by both. Two copies of a principle this soft would diverge on the next
    edit and nothing would notice — the paragraphs read fine either way."""
    from message_processor.prompts import _LET_THE_EXCHANGE_END

    for s in BOTH:
        assert _LET_THE_EXCHANGE_END in s


def test_letting_the_exchange_end_names_no_scenario():
    """OWNER RULING (2026-07-29): no scenario-specific rules in the prompts. The principle may
    describe what a landed exchange looks like; it may not enumerate cases and prescribe a reply
    for each, and it may never say silence is compulsory — "if the conversation ended, great; if
    more needs to be said, great"."""
    from message_processor.prompts import _LET_THE_EXCHANGE_END

    lowered = _LET_THE_EXCHANGE_END.lower()
    for banned in ("no_response_needed", "must not", "never reply", "always react",
                   "praise", "banter", "compliment"):
        assert banned not in lowered, f"the principle prescribes a case: {banned!r}"
    # And it stays a principle about worth, not an instruction to go silent: the reaction-or-
    # nothing option is offered, never imposed.
    assert "lands fine with a reaction or with nothing" in _LET_THE_EXCHANGE_END


def test_a_fulfilled_request_is_spent():
    """A live 2026-08-10 turn re-answered a pin request whose work was already done, announcing
    "It's already pinned." — it learned the fulfillment from its own tool result, so the room
    could not teach it. Being the addressee keeps earning an answer only while an answer is left
    to give. The rule rides the shared constant because that turn was a thread reply."""
    from message_processor.prompts import _LET_THE_EXCHANGE_END

    for fragment in ("turns out to be already delivered",
                     "your own tool comes back telling you it is already so",
                     "the request is spent",
                     "closes a spent request"):
        assert fragment in _LET_THE_EXCHANGE_END, f"the spent-request rule lost {fragment!r}"
    # And it reaches the thread path as well as the channel one — the incident was a thread turn.
    for s in BOTH:
        assert "the request is spent" in s


# ------------------------------------------------------------ the cross-thread conduct paragraph

CONDUCT = CHANNEL_CROSS_THREAD_CONDUCT_SUFFIX


def test_the_conduct_paragraph_licenses_the_post_before_it_constrains_it():
    """The paragraph has to license the act before it constrains it. A page of prohibitions with
    no permission in it produces a model that will not use the tool at all — which was the state
    the canvas tools were in before F36.

    The license is SUBSTANTIVE (owner ruling, 2026-08-04): what puts the answer in another thread
    is a responsibility living there, never the wording of the request that reached this one."""
    assert "legitimate and needs no apology" in CONDUCT
    assert "a concrete responsibility living in that thread" in CONDUCT
    assert "resolves, corrects or carries forward" in CONDUCT
    assert "a question left open there, an answer you owed there and can now give" in CONDUCT


def test_the_conduct_paragraph_demands_one_post_in_one_thread():
    assert "Post it ONCE, in the ONE thread it belongs in." in CONDUCT
    assert "Never send the same answer into more than one thread" in CONDUCT
    assert "never post it there and then repeat it here as well" in CONDUCT


def test_the_prompt_names_exactly_the_two_legal_sources():
    """T87. EXTENDED from "the stream labels only" (W3, §5.6): the executor's allowlist is now
    the `thread=<ts>` labels the stream rendered PLUS any thread in this channel a search or
    history tool returned this turn. Both places the rule is stated have to name those two and
    nothing else — a prompt pointing anywhere further (the coordinates block, a ts in somebody's
    message) invites a call the runtime refuses, and one pointing at only the first describes a
    stricter tool than the model has.

    W3 adds REACH, NOT PERMISSION, so the four clauses the P3b/FXP3 scenario rows measured in are
    asserted VERBATIM here rather than merely surviving nearby."""
    both = (CONDUCT, CHANNEL_POST_TO_THREAD_TARGET_DESCRIPTION)
    for text in both:
        assert "thread=<ts> in a message header" in text
        # RETURNED, not "shown": the executor refuses a hit whose workspace provenance is absent
        # or contradictory even though the model can read it, so "every thread you were shown"
        # would promise more than the executor grants. The prompt is the weaker claim on purpose.
        assert "a search or history tool returned to you this turn" in text
        assert "already shown you this turn" not in text
        assert "if a target is refused, open it with fetch_thread_messages" in text
        assert "two are the whole list of places you may post" in text or (
            "Those two are the only valid targets" in text)
    assert "a timestamp quoted inside somebody's message is not one" in CONDUCT
    assert "never a ts read out of a message body" in CHANNEL_POST_TO_THREAD_TARGET_DESCRIPTION

    for clause in ("not a loop of yours to close, however well you could settle it",
                   "When the open question is over there, that is where the answer goes",
                   "Do not report the post either"):
        assert clause in CONDUCT


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
    assert "a reaction usually carries it" in CONDUCT


def test_the_conduct_paragraph_licenses_a_brief_non_reporting_ack():
    """OWNER RULING 2026-08-03. The measured trials thanked the person who handed over the news
    ("Got it — thanks.") and the owner called that the model being a teammate, not a lapse — so
    the ack is licensed, and the license itself carries the boundary: the words must not depend
    on the post existing, must carry no figures, and must not point anywhere. The oracle-side
    twin of this line is `origin_ack_violation` (tests/live/battery_harness.py)."""
    assert "a brief human word to the person in front of you" in CONDUCT
    assert "would read exactly the same if the post had never happened" in CONDUCT
    assert "no figures, no mention of where anything went, nothing standing in for the answer" in (
        CONDUCT)


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
    other people's open argument.

    The closed two-case whitelist that used to carry this is gone (owner ruling, 2026-08-04) and
    the boundary is not: with the license now stated as "something is owed there", the sentences
    that say a strangers' exchange is not owed to you are the whole of what keeps the door shut."""
    assert "is not a loop of yours to close, however well you could settle it" in CONDUCT
    assert "posting into it reaches further in than speaking here would" in CONDUCT
    # The other half of the boundary, and the one the substantive license makes load-bearing:
    # prior participation and topical relevance are each explicitly refused as license.
    assert "Having been in a thread once is not a reason to return to it" in CONDUCT
    assert "a thread being about the same subject as this one is not a reason either" in CONDUCT


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
    """Same owner ruling as above. It describes what makes another thread the right place — a
    responsibility open there that this turn settles — and never a particular room's situation."""
    lowered = CONDUCT.lower()
    for banned in ("dana", "her thread", "nightly", "for example", "e.g."):
        assert banned not in lowered


# ------------------------------------------------------- the general thinking clause (§4)
#
# It is the positive half of the same ruling: with the placement frame gone, what tells the model
# to move work at all is one paragraph about deciding what a turn owes. It rides OUTSIDE every tool
# bullet, ahead of them, so it reaches the channel surface too — `CHANNEL_LOCAL_TOOLS_GUIDANCE`
# strips the post_to_thread line and nothing else. Every other test here would stay green if the
# paragraph were deleted from both blocks, which is exactly why it needs a pin of its own.

CLAUSE_OPENING = "First, what does this turn actually owe?"
CLAUSE_FRAGMENTS = (CLAUSE_OPENING,
                    "a reaction, or no words at all, is a complete reply",
                    "do the smallest sufficient thing",
                    "putting the work where it belongs rather than where you happen to be standing",
                    "going back to something of your own once you know it needs correcting")

# THE CHECK (2026-08-04, measured in). The scenario row news-settles-buried-loop reproduced the
# live failure 3 of 3: handed news that settled a question this channel had left open out of
# sight, the turn acknowledged it where it stood, stored a fact, and never looked. Nothing in the
# corpus had caught it because every other cross-thread row shows the owed thread in the stream.
# The principle rides in TWO places — the general clause, which is about what a turn owes, and the
# search bullet, which is where the tool that does the checking is described — and both halves
# reach both surfaces. Its second half is load-bearing in the other direction: without "when the
# connection is plain" and "turning up nothing is a perfectly good answer" this reads as a
# standing instruction to search, which is the failure the value floor exists to prevent.
CHECK_FRAGMENTS = ("what you can see is not the whole room",
                   "filing a fact away is not the same as closing the loop it belongs to",
                   "Look when the connection is plain, not on the chance that something might "
                   "turn up",
                   "turning up nothing is a perfectly good answer",
                   "It is also how you check what you cannot see",
                   "before you treat the turn as finished")


def test_the_check_principle_rides_both_shared_blocks_exactly_once():
    """Deleting it would leave every other test here green, and the row that measures it costs
    real API calls — so the wording that moved it is pinned where a unit run can see it."""
    for label, block in (("the DM etiquette", LOCAL_TOOLS_GUIDANCE),
                         ("the channel etiquette", CHANNEL_LOCAL_TOOLS_GUIDANCE)):
        for fragment in CHECK_FRAGMENTS:
            assert block.count(fragment) == 1, (
                f"{label} carries {fragment!r} {block.count(fragment)} times, expected once")


def test_the_check_principle_names_no_occasion():
    """Standing owner rule: prompts state principles, and the enumerated cases live in the
    scenario corpus. The three shapes this was written from — an open question, a pending
    decision, its own answer gone stale — must not appear as a list of things to go looking for,
    or the model learns to match situations instead of reading one."""
    for block in (LOCAL_TOOLS_GUIDANCE, CHANNEL_LOCAL_TOOLS_GUIDANCE):
        lowered = block.lower()
        for banned in ("for example", "e.g.", "such as when", "in these cases"):
            assert banned not in lowered


def test_the_thinking_clause_rides_both_shared_blocks_exactly_once():
    """One paragraph, both surfaces, one copy each — the same rule as every other shared text
    here. Two copies on one surface is a drift waiting to happen; zero on the channel surface
    would mean the ruling's positive half never reached the room the tool lives in."""
    for label, block in (("the DM etiquette", LOCAL_TOOLS_GUIDANCE),
                         ("the channel etiquette", CHANNEL_LOCAL_TOOLS_GUIDANCE)):
        for fragment in CLAUSE_FRAGMENTS:
            assert block.count(fragment) == 1, (
                f"{label} carries {fragment!r} {block.count(fragment)} times, expected once")


def test_the_thinking_clause_precedes_every_tool_bullet():
    """It is general guidance, not a bullet about a tool. Below the list it would read as a note
    on whichever tool it followed, and the placement is the whole reason it applies to the turn
    rather than to a call."""
    for block in (LOCAL_TOOLS_GUIDANCE, CHANNEL_LOCAL_TOOLS_GUIDANCE):
        lines = block.split("\n")
        clause = next(i for i, line in enumerate(lines) if line.startswith(CLAUSE_OPENING))
        first_bullet = next(i for i, line in enumerate(lines) if line.startswith("- "))
        assert clause < first_bullet
        # And it is its own line, so the channel derivation cannot take it out with a bullet.
        assert not lines[clause].startswith("- ")


def test_neither_shared_block_advertises_the_terminal_silence_tool():
    """CACHE HYGIENE, and the reason the clause says "no words at all" rather than naming the
    tool. `no_response_needed` is exposed only on silence-capable turns and lives in the volatile
    suffix (prompts.py:300-302); naming it in the always-cached etiquette would promise a tool
    that half the turns do not have, and a model that calls a tool it was not given reports the
    failure as its answer."""
    for block in (LOCAL_TOOLS_GUIDANCE, CHANNEL_LOCAL_TOOLS_GUIDANCE):
        assert "no_response_needed" not in block


# ------------------------------------------------------ the request-parsing frame is GONE (§5)
#
# The owner's 2026-08-04 ruling: "no special wording... if those exist in our code base, we did it
# wrong." The frame that came out told the model to read a trigger for an INSTRUCTION about where
# its answer should land, which turned an NLP assistant's judgment into a phrase it had to spot.
# Deleting prose is the one edit no positive assertion can protect — every pin above still passes
# with the whitelist quietly restored beside the new license — so the fragments are named here and
# asserted ABSENT from every surface that ever carried them.

FRAME_FRAGMENTS = ("placement request", "asks the answer to go", "answer in that thread",
                   "Two cases", "Nothing else licenses it", "someone asked you to answer")


def _frame_free(label: str, text: str) -> None:
    for fragment in FRAME_FRAGMENTS:
        assert fragment.lower() not in text.lower(), (
            f"{label} still carries the request-parsing frame: {fragment!r}")


def test_no_surface_asks_the_model_to_parse_a_request_for_placement():
    """All five prompt surfaces the frame was written across: the DM etiquette bullet (which is
    also the channel bullet's source), the conduct paragraph, and both schemas' descriptions."""
    from slack_client.messaging import SlackMessagingMixin

    bullet = next(line for line in LOCAL_TOOLS_GUIDANCE.split("\n")
                  if line.startswith("- post_to_thread:"))
    _frame_free("the post_to_thread bullet", bullet)
    _frame_free("the conduct paragraph", CONDUCT)
    _frame_free("the channel schema description", CHANNEL_POST_TO_THREAD_DESCRIPTION)
    _frame_free("the channel target description", CHANNEL_POST_TO_THREAD_TARGET_DESCRIPTION)
    _frame_free("the DM schema description", SlackMessagingMixin._POST_TO_THREAD_DESCRIPTION)
    _frame_free("the DM target description",
                SlackMessagingMixin._POST_TO_THREAD_TARGET_DESCRIPTION)


def test_the_live_battery_no_longer_triggers_the_cross_thread_row_with_a_magic_phrase():
    """The prompts can be clean while the thing that MEASURES them still asks for the old
    behavior. Row 4's trigger used to end "answer in that thread" — the exact wording the ruling
    removed — so a redesigned row is part of the same change, and the row's own source is where
    the phrase would come back. No Slack, no network: this reads the module."""
    import inspect

    from tests.live import battery_rows

    row = next(r for r in battery_rows.REGISTRY if r.name == "search-to-action")
    _frame_free("row 4's documented trigger", row.trigger_template)
    _frame_free("row 4's source", inspect.getsource(row.run))


# -------------------------------------------------- no contradictory origin-ack on this surface

ORIGIN_ACK_PHRASES = ("acknowledge briefly here", "just acknowledge briefly",
                      "Acknowledge briefly in the current thread")


def test_the_channel_etiquette_drops_the_bullet_that_contradicts_the_schema():
    """THE CONTRADICTION (plan §1c): the DM etiquette says post there and acknowledge HERE, and the
    channel schema and conduct paragraph say post there and say nothing here. Two instructions, one
    tool, opposite endings — and the cached one wins arguments it should not."""
    assert "post_to_thread" in LOCAL_TOOLS_GUIDANCE, "the DM bullet is the thing being removed"
    assert "post_to_thread" not in CHANNEL_LOCAL_TOOLS_GUIDANCE


def test_the_channel_etiquette_is_the_dm_etiquette_minus_and_plus_exactly_the_declared_bullets():
    """Derived, not copied — and this is the assertion that keeps the derivation honest. Renaming
    a DM bullet would make the filter a no-op and quietly restore the contradiction; a divergence
    anywhere else means somebody edited one surface's etiquette and not the other's.

    The toolbelt round (2026-08-12) made the difference two-way. It is still DECLARED, in the two
    prompts.py tuples read here rather than in a literal pinned to this test: a DM-only bullet is
    filtered out (post_to_thread, whose channel instruction is the opposite; personal memory, which
    a channel turn has no store for) and a channel-only bullet is spliced in (delete_own_message
    and the topic/purpose pair, none of which the DM surface is registered for at all). Every other
    line must still be identical on both surfaces."""
    dm_lines = LOCAL_TOOLS_GUIDANCE.split("\n")
    channel_lines = CHANNEL_LOCAL_TOOLS_GUIDANCE.split("\n")

    removed = [line for line in dm_lines if line not in channel_lines]
    assert len(removed) == 2, f"expected exactly two DM-only bullets, got {len(removed)}"
    assert [line for line in removed
            if line.startswith(prompts.DM_ONLY_TOOL_BULLET_PREFIXES)] == removed
    assert any(line.startswith("- post_to_thread:") for line in removed)

    added = [line for line in channel_lines if line not in dm_lines]
    assert added == list(prompts.CHANNEL_ONLY_TOOL_BULLETS)

    # And nothing else moved: strip each surface's own bullets and the two texts are the same.
    assert ([line for line in channel_lines if line not in added]
            == [line for line in dm_lines if line not in removed])


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
    """The DM schema reads its OWN constants, whatever either surface's words happen to be.

    Its description moved once, in the 2026-08-04 wave, because the request-parsing frame it
    shared with the channel text was removed everywhere it was written — a licence that depends on
    how somebody phrased their request was wrong on both surfaces. What this test holds is the
    thing that did not change: the two surfaces are two constants, the channel's words never reach
    the DM schema, and the two schemas are not equal."""
    from slack_client.messaging import SlackMessagingMixin

    dm = _post_to_thread_schema("dm")
    assert dm["description"] == SlackMessagingMixin._POST_TO_THREAD_DESCRIPTION
    assert (dm["parameters"]["properties"]["thread_ts"]["description"]
            == SlackMessagingMixin._POST_TO_THREAD_TARGET_DESCRIPTION)
    assert dm != _post_to_thread_schema("channel")


def test_the_channel_schema_promises_only_what_the_executor_allows():
    """The DM text says a target may come "from a tool", flatly. The channel executor is
    narrower on BOTH sides and the schema has to match it, or the model learns that a working
    tool is broken.

    W3 widened the allowlist to include a root a search or history tool RETURNED THIS TURN — so
    the old "the stream's labels and nothing else" is gone. It did NOT widen to every ts a tool
    ever showed: a hit whose workspace provenance is absent or contradictory is still refused,
    which is why the channel wording promises a returned thread rather than any seen one, and
    names the recovery instead of implying there is never a refusal."""
    from slack_client.messaging import SlackMessagingMixin

    assert "or from a tool" in SlackMessagingMixin._POST_TO_THREAD_TARGET_DESCRIPTION
    assert "or from a tool" not in CHANNEL_POST_TO_THREAD_TARGET_DESCRIPTION
    assert "this channel's stream labels it" in CHANNEL_POST_TO_THREAD_TARGET_DESCRIPTION
    assert ("a search or history tool returned to you this turn"
            in CHANNEL_POST_TO_THREAD_TARGET_DESCRIPTION)
    # The prompt must not promise more than the executor grants: refusals exist, and the way
    # through one is named rather than left for the model to discover.
    assert ("if a target is refused, open it with fetch_thread_messages"
            in CHANNEL_POST_TO_THREAD_TARGET_DESCRIPTION)


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
    from message_processor.tool_registry import SURFACE_CHANNEL, ToolRegistry

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
    from message_processor.tool_registry import SURFACE_CHANNEL

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
    from message_processor.tool_registry import SURFACE_CHANNEL

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

    from message_processor.tool_registry import SURFACE_CHANNEL, ToolRegistry

    empty = ToolRegistry()
    empty.register({"type": "function", "name": "react_to_message", "parameters": {}}, AsyncMock())
    assert CONDUCT not in _materialize(SURFACE_CHANNEL, tools_disabled=True)
    assert CONDUCT not in _materialize(SURFACE_CHANNEL, registry=empty)
    assert "post_to_thread" not in CHANNEL_LOCAL_TOOLS_GUIDANCE


def test_the_conduct_paragraph_never_rides_a_dm_turn():
    """The tool IS in this registry's DM schema set, so what is under test is the SURFACE check and
    nothing else. A DM has no other thread of its own, and the channel-wide conduct paragraph has
    no business in its prompt."""
    from message_processor.tool_registry import SURFACE_DM

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
    """The DM prompt is the DM block and nothing else. Its etiquette is the whole DM block, the
    channel text is nowhere in it, and none of the four channel constants leak across.

    What is guaranteed is SEPARATION, not immobility: the DM etiquette's own words may be edited
    (the 2026-08-04 wave rewrote the post_to_thread bullet on both surfaces), and this test still
    fails the moment channel wording crosses over or the derivation stops removing the bullet."""
    from unittest.mock import MagicMock

    from message_processor.utilities import MessageUtilitiesMixin
    from message_processor.tool_registry import SURFACE_CHANNEL, SURFACE_DM

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
    from message_processor.prompts import WAKE_CLASSIFIER_SYSTEM_PROMPT

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
