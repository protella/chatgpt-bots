"""Prompt modernization (frontier-model trim) — behavioral contracts.

Covers: the multi-user prefix-cache fix in _get_system_prompt, the ack reaction, and the
presence of the teammate/batch/brevity guidance. (The intent classifier and the vision
enhancement hop this file also used to cover were deleted with the legacy image path — F34
made image work a set of TOOLS, so nothing pre-routes a turn any more.)
"""
import asyncio

from unittest.mock import MagicMock

from message_processor.utilities import MessageUtilitiesMixin
from message_processor.prompts import (
    SLACK_SYSTEM_PROMPT,
    WAKE_CLASSIFIER_SYSTEM_PROMPT,
)


# --------------------------------------------------------------------------- harness

class _Proc(MessageUtilitiesMixin):
    def __init__(self, openai_client=None):
        self.db = None
        self.openai_client = openai_client

    def log_info(self, *a, **k): pass
    def log_debug(self, *a, **k): pass
    def log_warning(self, *a, **k): pass
    def log_error(self, *a, **k): pass


def _slack_client():
    client = MagicMock()
    client.name = "slack"
    client.tool_registry = None
    return client


ROSTER_TWO_HUMANS = (
    "\n\nTHREAD PARTICIPANTS — to mention or tag someone, write their Slack ID in the form "
    "<@USER_ID> (exactly, with the angle brackets). Never put a person's plain name inside "
    "angle brackets. Known participants:\n- Peter → <@U1AAA>\n- Dana → <@U2BBB>"
)
ROSTER_ONE_HUMAN = ROSTER_TWO_HUMANS.rsplit("\n", 1)[0]  # only Peter


def _sys_prompt(proc, user_real_name=None, user_email=None, roster=None):
    return proc._get_system_prompt(
        _slack_client(), "UTC", None, user_real_name, user_email,
        "gpt-5.5", False, False, None, participant_roster=roster,
    )


# ------------------------------------------------- multi-user prefix-cache fix

def test_channel_prefix_stable_across_triggering_users():
    """In a multi-user thread (roster >= 2 humans) the system prompt must be
    byte-identical regardless of who triggered the response — otherwise every
    speaker change busts the OpenAI prefix cache for the whole thread."""
    proc = _Proc()
    p1 = _sys_prompt(proc, "Erin Evans", "peter@example.com", ROSTER_TWO_HUMANS)
    p2 = _sys_prompt(proc, "Dana Smith", "dana@example.com", ROSTER_TWO_HUMANS)
    assert p1 == p2
    assert "You're speaking with" not in p1


def test_dm_prompt_keeps_user_context():
    """DMs (no roster) keep the stable 'You're speaking with' line."""
    proc = _Proc()
    p = _sys_prompt(proc, "Erin Evans", "peter@example.com", roster=None)
    assert "You're speaking with Erin Evans (email: peter@example.com)" in p


def test_single_user_thread_keeps_user_context():
    """One human in the roster -> the sender can never change -> keeping the
    line is cache-safe and preserves identity context."""
    proc = _Proc()
    p = _sys_prompt(proc, "Erin Evans", None, roster=ROSTER_ONE_HUMAN)
    assert "You're speaking with Erin Evans" in p


def _run(coro):
    return asyncio.run(coro)


# F38: `_place_ack_reaction` is gone, and with it the tests that pinned it. It fired on the
# first tool EVENT — before a call's arguments were validated, and for fast lookups that were
# over before the eye rendered. The work claim now lives on TurnRuntime (see
# tests/unit/test_ack_lifecycle.py): staked only by work that is slow and really happening,
# and taken back if that work produces nothing.


# ------------------------------------------------- new guidance present

def test_teammate_batch_brevity_lines_present():
    assert "teammate" in SLACK_SYSTEM_PROMPT
    assert "use a thread when the request calls for the detail" in SLACK_SYSTEM_PROMPT
    assert "several queued messages" in SLACK_SYSTEM_PROMPT
    assert "emoji reaction is your entire response" in SLACK_SYSTEM_PROMPT


def test_f17_voice_banter_clause_present():
    # F17: the Voice paragraph adopts a personable-teammate register — banter/teasing
    # aimed at the bot gets answered in kind, with self-aware humor, but never forced
    # and never at the expense of real help; playful register never licenses fabrication.
    assert "teasing pointed straight at you" in SLACK_SYSTEM_PROMPT
    assert "self-aware humor about being a bot" in SLACK_SYSTEM_PROMPT
    assert "never force a joke" in SLACK_SYSTEM_PROMPT
    assert "never do bits when someone actually needs help" in SLACK_SYSTEM_PROMPT
    assert "never licenses making things up" in SLACK_SYSTEM_PROMPT
    # channel-level brevity for banter lives in the Participation paragraph
    assert "one good line beats three" in SLACK_SYSTEM_PROMPT



def test_f17_participation_banter_clause_is_gone():
    """Superseded 2026-07-26 — see test_b_banter_rule_replaced for the full story. F17 added a
    clause making teasing aimed at the assistant participation-worthy; it turned out to be the
    mechanism behind a real misfire (the bot replying to a jab 52s after being told to hush) and
    was removed. Kept as an inverted guard so the clause cannot quietly return.

    Re-baselined for the binary gate: the prompt it was removed from (PARTICIPATION_SYSTEM_PROMPT)
    no longer exists, so the guard now points at the ONE gate prompt that replaced it. The two
    positive assertions that used to live here pinned the rich prompt's staged addressee reasoning
    ("STAGE 1 — WHOSE MESSAGE IS THIS?", relation=about_assistant); the gate does not decide
    addressee at all now, so there is nothing left to assert and asserting the successor would be
    inventing a contract."""
    assert "Playful banter or teasing genuinely aimed AT the assistant is participation-worthy" \
        not in WAKE_CLASSIFIER_SYSTEM_PROMPT
    assert "not marginal-value noise" not in WAKE_CLASSIFIER_SYSTEM_PROMPT


def test_tool_provenance_ground_truth_instruction_present():
    # M2: the model must treat its own "[used tools: …]" annotations as authoritative
    # ground truth about its past actions and never deny them (it was denying its own
    # verified tool use even with the annotation in context).
    assert "[used tools:" in SLACK_SYSTEM_PROMPT
    assert "ground truth" in SLACK_SYSTEM_PROMPT
    lowered = SLACK_SYSTEM_PROMPT.lower()
    assert "authoritative" in lowered
    # Absence of an annotation must be interpreted as "no local tools ran".
    assert "no such line means you used no local tools" in SLACK_SYSTEM_PROMPT


def test_grounding_rule_governs_binding_not_just_sourcing():
    """The live failure that motivated the rule: asked why a nightly job slowed down, the bot
    answered "you just called it a minute ago: replica warmup was the culprit" — lifting a cause
    from an unrelated message someone had sent a colleague three minutes earlier.

    The pre-existing Truthfulness rule did not catch it and could not: it governs SOURCING, and the
    bot had genuinely called fetch_channel_history. What was unsupported was the LINK it drew
    between two records.

    REWRITTEN as evidence-not-proof. The first version was categorical — it told the bot what it
    could not conclude, and a rule phrased that way invites the opposite failure, where an empty
    lookup gets reported as a fact about the world ("that never happened") because absence felt
    like a finding. This version says what the material IS: evidence about the room, incomplete by
    nature, with strength that has to be carried honestly in both directions."""
    assert "Grounding:" in SLACK_SYSTEM_PROMPT
    # The frame: what the bot can read is a partial record, not the events themselves.
    assert "evidence about the room, not proof of everything that happened in it" in SLACK_SYSTEM_PROMPT
    assert "partial, stale, mistaken, or missing the exchange that gives a line its meaning" \
        in SLACK_SYSTEM_PROMPT
    # THE new half: not finding something is a fact about the lookup, not about the world.
    assert "Absence of evidence is not evidence that something did not happen" in SLACK_SYSTEM_PROMPT
    assert "say only that you did not find it there" in SLACK_SYSTEM_PROMPT
    # Adjacency, topic and a shared system name are explicitly denied as links.
    assert "Do not invent links between records" in SLACK_SYSTEM_PROMPT
    assert "proximity, similar topics, or the same system name" in SLACK_SYSTEM_PROMPT
    # A related record is offered as a lead, not served as the answer.
    assert "report a related record as a lead, not as the answer" in SLACK_SYSTEM_PROMPT
    # Claim strength is preserved rather than hardened, and stays attributed.
    assert "Keep every claim exactly as strong as its source" in SLACK_SYSTEM_PROMPT
    assert "attribute it to whoever said it, about what they were actually discussing" \
        in SLACK_SYSTEM_PROMPT
    # And the anti-hedge clause, without which this rule turns the bot to mush.
    assert "do not hedge ceremonially" in SLACK_SYSTEM_PROMPT
    assert "when the evidence is sufficient, answer directly" in SLACK_SYSTEM_PROMPT


def test_the_categorical_phrasing_is_gone():
    """The sentences the rewrite replaced. Pinned as absences because the old wording reads as
    more decisive and is the natural thing to reach for when tightening this paragraph again."""
    for retired in ("not that some link you drew between two of them is real",
                    "several conversations interleaved",
                    "never because the topics rhyme",
                    "points back into their own conversation",
                    "when the support is there, say it plainly"):
        assert retired not in SLACK_SYSTEM_PROMPT, retired


def test_details_are_repeated_at_the_precision_they_were_given():
    """Found by the frozen holdout corpus: told "board review moved to the 14th", the bot answered
    "August 14". No message named a month — it resolved the 14th against today's date and served
    the inference as if it were the record. The pre-existing "never fabricate details" rule did not
    catch it, because sharpening a detail doesn't feel like inventing one."""
    assert "don't quietly sharpen one either" in SLACK_SYSTEM_PROMPT
    assert "repeat a detail at the precision it was given" in SLACK_SYSTEM_PROMPT
    # The three ways it actually over-resolves: month from today's date, year, surname.
    assert "which month from today's date, which year, or whose surname" in SLACK_SYSTEM_PROMPT
    # And why it matters more than plain invention.
    assert "harder to catch" in SLACK_SYSTEM_PROMPT


def test_voice_carries_no_word_blacklist():
    """A previous revision added a list of banned tells to the voice paragraph (em dashes, the
    "not just X, it's Y" reversal, delve/leverage/robust/seamless, tidy three-item lists). It was
    added without authorization, never measured, and the owner rejected it. Codex's objection was
    substantive too: banning useful words can make an answer worse, which is the opposite of the
    goal. Tone work is deferred; if it returns, the mechanism to copy counts named defects and
    sends at zero rather than blacklisting vocabulary."""
    for banned_list_marker in ("The loudest tell is the em dash", "delve, leverage, robust",
                               "it's not just X, it's Y", "a testament to"):
        assert banned_list_marker not in SLACK_SYSTEM_PROMPT
    # The voice paragraph itself survives — only the blacklist came out.
    assert "write the way a sharp coworker writes in Slack" in SLACK_SYSTEM_PROMPT
    assert "If one line covers it, send one line." in SLACK_SYSTEM_PROMPT
