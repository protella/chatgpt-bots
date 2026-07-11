"""Prompt modernization (frontier-model trim) — behavioral contracts.

Covers: the multi-user prefix-cache fix in _get_system_prompt, the vision
enhancement-hop retirement (flag default off + default question), the intent
classifier's five-label contract on the trimmed prompt, and the presence of
the new teammate/batch/brevity guidance.
"""
import asyncio
import types

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from config import config
from openai_client.api.responses import (classify_intent as _classify_intent,
                                          _parse_intent_and_ack)
from message_processor.utilities import MessageUtilitiesMixin
from message_processor.handlers.vision import VisionHandlerMixin, _VAGUE_VISION_ASKS
from prompts import (
    INTENT_CLASSIFIER_PROMPT,
    PARTICIPATION_SYSTEM_PROMPT,
    SLACK_SYSTEM_PROMPT,
    VISION_DEFAULT_QUESTION,
)


# --------------------------------------------------------------------------- harness

class _Proc(VisionHandlerMixin, MessageUtilitiesMixin):
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


# ------------------------------------------------- vision enhancement retirement

def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro) if False else asyncio.run(coro)


def test_enhancement_skipped_when_flag_off():
    openai_client = MagicMock()
    openai_client._enhance_vision_prompt = AsyncMock(return_value="ENHANCED")
    proc = _Proc(openai_client=openai_client)
    with patch.object(config, "enable_vision_enhancement", False):
        out = _run(proc._build_vision_question("what breed is the dog on the left?", []))
    assert out == "what breed is the dog on the left?"
    openai_client._enhance_vision_prompt.assert_not_awaited()


def test_enhancement_runs_when_flag_on():
    openai_client = MagicMock()
    openai_client._enhance_vision_prompt = AsyncMock(return_value="ENHANCED")
    proc = _Proc(openai_client=openai_client)
    with patch.object(config, "enable_vision_enhancement", True):
        out = _run(proc._build_vision_question("what breed is the dog?", [{"role": "user"}]))
    assert out == "ENHANCED"
    openai_client._enhance_vision_prompt.assert_awaited_once()


@pytest.mark.parametrize("ask", ["", "   ", "describe this", "What is this?", "describe this image."])
def test_default_question_for_empty_or_vague_asks(ask):
    """Empty/vague asks get the standard default question — flag on OR off."""
    openai_client = MagicMock()
    openai_client._enhance_vision_prompt = AsyncMock(return_value="ENHANCED")
    proc = _Proc(openai_client=openai_client)
    for flag in (False, True):
        with patch.object(config, "enable_vision_enhancement", flag):
            out = _run(proc._build_vision_question(ask, []))
        assert out == VISION_DEFAULT_QUESTION
    openai_client._enhance_vision_prompt.assert_not_awaited()


def test_vague_set_is_lowercase_normalized():
    assert all(p == p.lower() for p in _VAGUE_VISION_ASKS)


# ------------------------------------------------- intent classifier contract

class _Classifier:
    """Binds the real classify_intent with a mocked API."""
    from openai_client.api.responses import classify_intent  # bound below

    def __init__(self, word):
        content = types.SimpleNamespace(text=word)
        item = types.SimpleNamespace(content=[content])
        self._response = types.SimpleNamespace(output=[item], usage=None)
        self.client = MagicMock()
        self.client.timeout = 30
        self.captured_params = None

    async def _safe_api_call(self, fn, operation_type=None, timeout_seconds=None, **params):
        self.captured_params = params
        return self._response

    def log_debug(self, *a, **k): pass
    def log_warning(self, *a, **k): pass
    def log_error(self, *a, **k): pass


_Classifier.classify_intent = _classify_intent


@pytest.mark.parametrize("word,expected", [
    ("new", "new_image"),
    ("edit", "edit_image"),
    ("vision", "vision"),
    ("ambiguous", "ambiguous_image"),
    ("none", "text_only"),
    ("garbage sentence with spaces", "text_only"),  # invalid -> safe default
])
def test_classifier_five_label_contract(word, expected):
    c = _Classifier(word)
    intent = _run(c.classify_intent([], "some message"))
    assert intent == expected
    # The trimmed prompt is what actually gets sent
    dev_msgs = [m for m in c.captured_params["input"] if m.get("role") == "developer"]
    assert dev_msgs and dev_msgs[0]["content"] == INTENT_CLASSIFIER_PROMPT


def test_classifier_prompt_token_budget():
    """Fires on every responded message and sits below the 1024-token prompt-cache
    threshold — must stay small. chars/4 proxy."""
    assert len(INTENT_CLASSIFIER_PROMPT) / 4 < 350


# ------------------------------------------------- F19 intent ack parse

@pytest.mark.parametrize("raw,intent,ack", [
    ("vision ack", "vision", True),          # two tokens: intent + ack
    ("new ack", "new_image", True),
    ("edit noack", "edit_image", False),     # explicit noack
    ("none", "text_only", False),            # one-word fallback -> noack
    ("vision", "vision", False),
    ("new banana", "new_image", False),      # garbage 2nd token never breaks routing
    ("none whatever extra", "text_only", False),
    ("garbage sentence here", "text_only", False),  # unmapped 1st token -> safe default
    ("", "text_only", False),                # empty -> safe default
    ("  vision   ACK  ", "vision", True),    # whitespace + case tolerant
])
def test_parse_intent_and_ack(raw, intent, ack):
    assert _parse_intent_and_ack(raw) == (intent, ack)


def test_classifier_return_ack_two_token(monkeypatch):
    """return_ack=True unpacks (intent, ack) from the two-token model output."""
    c = _Classifier("vision ack")
    intent, ack = _run(c.classify_intent([], "look at this", return_ack=True))
    assert intent == "vision" and ack is True


def test_classifier_return_ack_defaults_noack_on_one_word():
    """A one-word (legacy-shaped) output still parses; ack defaults False."""
    c = _Classifier("none")
    intent, ack = _run(c.classify_intent([], "hi", return_ack=True))
    assert intent == "text_only" and ack is False


def test_classifier_bare_return_unchanged_without_return_ack():
    """Without return_ack the call still returns a bare intent string (back-compat)."""
    c = _Classifier("new ack")
    assert _run(c.classify_intent([], "draw a cat")) == "new_image"


# ------------------------------------------------- F19 intent-path ack reaction

def _ack_msg():
    from base_client import Message
    return Message(text="analyze this deck", user_id="U1", channel_id="C1",
                   thread_id="99.0", metadata={"ts": "99.0"})


def test_place_ack_reaction_uses_reservation_guard(monkeypatch):
    """The intent-path ack routes through the F6 reservation guard on the triggering ts."""
    monkeypatch.setattr(config, "ack_reaction_emoji", "eyes", raising=False)
    proc = _Proc()
    client = MagicMock()
    client._reserve_and_react = AsyncMock(return_value={"ok": True})
    client.react = AsyncMock()
    _run(proc._place_ack_reaction(client, _ack_msg()))
    client._reserve_and_react.assert_awaited_once_with("C1", "99.0", "eyes")
    client.react.assert_not_awaited()


def test_place_ack_reaction_falls_back_to_react(monkeypatch):
    """A client without the guard still gets the reaction via plain react()."""
    monkeypatch.setattr(config, "ack_reaction_emoji", "hourglass", raising=False)
    proc = _Proc()
    client = MagicMock(spec=["react"])
    client.react = AsyncMock()
    _run(proc._place_ack_reaction(client, _ack_msg()))
    client.react.assert_awaited_once_with("C1", "99.0", "hourglass")


def test_place_ack_reaction_is_silent_on_failure(monkeypatch):
    """A wedged/failing Slack call never propagates — the turn continues."""
    monkeypatch.setattr(config, "ack_reaction_emoji", "eyes", raising=False)
    proc = _Proc()
    client = MagicMock()
    client._reserve_and_react = AsyncMock(side_effect=RuntimeError("slack down"))
    _run(proc._place_ack_reaction(client, _ack_msg()))  # does not raise


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


def test_f17_classifier_banter_clause_present():
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
