"""Prompt modernization (frontier-model trim) — behavioral contracts.

Covers: the multi-user prefix-cache fix in _get_system_prompt, the ack reaction, and the
presence of the teammate/batch/brevity guidance. (The intent classifier and the vision
enhancement hop this file also used to cover were deleted with the legacy image path — F34
made image work a set of TOOLS, so nothing pre-routes a turn any more.)
"""
import asyncio

from unittest.mock import MagicMock

from message_processor.utilities import MessageUtilitiesMixin
from prompts import (
    PARTICIPATION_SYSTEM_PROMPT,
    SLACK_SYSTEM_PROMPT,
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
    assert "offer to expand in a thread" in SLACK_SYSTEM_PROMPT
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


def test_f17_participation_banter_clause_present():
    # F17: playful banter/teasing genuinely AT the assistant is a respond (a short quip)
    # or react case — not "marginal value" to ignore — but addressee rules still dominate.
    assert "Playful banter or teasing aimed genuinely AT the assistant is a respond case" \
        in PARTICIPATION_SYSTEM_PROMPT
    assert "not marginal-value noise to ignore" in PARTICIPATION_SYSTEM_PROMPT
    assert "never overrides the addressee rules" in PARTICIPATION_SYSTEM_PROMPT


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
