"""F20 — human-style reactions on others' posts.

Covers the five design points: broadened react-verdict prompt guidance + softened
main-model etiquette; unrestricted standard-emoji judgment at all four enforcement
points (schema enum, executor, classifier signal line, validate_verdict) with the
optional REACTION_EMOJIS allowlist still honored; and the pulse-ring social-proof
signal (reaction_added/removed accumulation + envelope/tail summary rendering).
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import config, valid_emoji_name
from tool_registry import ToolContext
from slack_client.channel_pulse import ChannelPulse
from slack_client.messaging import SlackMessagingMixin
from slack_client.event_handlers import feedback as feedback_handlers
from message_processor.participation import ParticipationEngine
import prompts


# ------------------------------------------------------------------- point 1+4: prompts

def test_participation_prompt_broadened_guidance():
    p = prompts.PARTICIPATION_SYSTEM_PROMPT
    # social-proof: the room already reacting lowers the bar
    assert "already reacted" in p and "LOWERS the bar" in p
    # taste rails preserved
    assert "heated, sensitive, or personal" in p
    assert "when unsure, ignore" in p
    # any-standard-emoji wording (no curated palette)
    assert "any standard Slack emoji name" in p


def test_local_tools_guidance_softened():
    g = prompts.LOCAL_TOOLS_GUIDANCE
    assert "react the way a teammate does" in g
    assert "when the room is already reacting" in g
    # absolutism removed
    assert "Most messages deserve NO reaction" not in g
    # still one-per-message rail
    assert "one emoji per target message" in g


# ------------------------------------------------------------------ F24: reaction-preference

def test_f24_participation_prompt_prefers_react():
    p = prompts.PARTICIPATION_SYSTEM_PROMPT
    # preference wording: react over respond when an emoji fully carries the reply
    assert 'PREFER "react" over "respond"' in p
    # delegation/FYI example present
    assert "instruction or delegation" in p and "FYI" in p
    # redundant-acknowledgment rule present
    assert "ALREADY acknowledged with a reaction" in p


def test_f24_local_tools_guidance_broadened():
    g = prompts.LOCAL_TOOLS_GUIDANCE
    # broadened beyond "thanks!" to acknowledgments/delegations/FYIs
    assert "got it" in g and "delegation" in g and "FYI" in g
    assert "while I'm out" in g


# ------------------------------------------------------------------ point 2: schema enum

def _mixin_host():
    s = MagicMock()
    s.get_react_tool_schema = SlackMessagingMixin.get_react_tool_schema.__get__(s)
    return s


def test_schema_has_no_enum_by_default(monkeypatch):
    monkeypatch.setattr(config, "reaction_emojis", [])
    schema = _mixin_host().get_react_tool_schema()
    emoji = schema["parameters"]["properties"]["emoji"]
    assert "enum" not in emoji
    assert "standard Slack emoji shorthand name" in emoji["description"]


def test_schema_gains_enum_when_configured(monkeypatch):
    monkeypatch.setattr(config, "reaction_emojis", ["thumbsup", ":eyes:"])
    schema = _mixin_host().get_react_tool_schema()
    assert schema["parameters"]["properties"]["emoji"]["enum"] == ["thumbsup", "eyes"]


# ------------------------------------------------------------------- point 2: executor

def _react_self():
    s = MagicMock()
    s.react = AsyncMock(return_value=True)
    s.execute_react_tool = SlackMessagingMixin.execute_react_tool.__get__(s)
    s._reserve_and_react = SlackMessagingMixin._reserve_and_react.__get__(s)
    s._trim_reaction_guard = SlackMessagingMixin._trim_reaction_guard.__get__(s)
    s._REACTION_GUARD_MAX = SlackMessagingMixin._REACTION_GUARD_MAX
    s._REACTION_GUARD_RECENCY_S = SlackMessagingMixin._REACTION_GUARD_RECENCY_S
    s._reaction_guard = None
    s._reaction_guard_ts = None
    return s


class TestExecutorUnrestricted:
    def setup_method(self):
        self.ctx = ToolContext(channel_id="C1", thread_ts="100.0", trigger_ts="123.4")

    @pytest.mark.asyncio
    async def test_off_list_name_accepted_by_default(self, monkeypatch):
        monkeypatch.setattr(config, "enable_reactions", True)
        monkeypatch.setattr(config, "enable_react_tool", True)
        monkeypatch.setattr(config, "reaction_emojis", [])
        s = _react_self()
        out = await s.execute_react_tool(self.ctx, {"emoji": "joy"})
        assert out["ok"] is True
        s.react.assert_awaited_once_with("C1", "123.4", "joy")

    @pytest.mark.asyncio
    async def test_malformed_name_rejected_syntactically(self, monkeypatch):
        monkeypatch.setattr(config, "enable_reactions", True)
        monkeypatch.setattr(config, "enable_react_tool", True)
        monkeypatch.setattr(config, "reaction_emojis", [])
        s = _react_self()
        out = await s.execute_react_tool(self.ctx, {"emoji": "NOT valid!"})
        assert out["ok"] is False and out["error"] == "invalid_emoji"
        s.react.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_allowlist_enforced_when_configured(self, monkeypatch):
        monkeypatch.setattr(config, "enable_reactions", True)
        monkeypatch.setattr(config, "enable_react_tool", True)
        monkeypatch.setattr(config, "reaction_emojis", ["thumbsup"])
        s = _react_self()
        out = await s.execute_react_tool(self.ctx, {"emoji": "joy"})
        assert out["ok"] is False and out["error"] == "emoji_not_allowed"
        s.react.assert_not_awaited()


def test_valid_emoji_name_matrix():
    assert valid_emoji_name("joy") and valid_emoji_name("+1") and valid_emoji_name("-1")
    assert valid_emoji_name("white_check_mark") and valid_emoji_name("thumbsup")
    assert not valid_emoji_name("")
    assert not valid_emoji_name("NOT valid!")
    assert not valid_emoji_name("has space")
    assert not valid_emoji_name("x" * 65)


# ------------------------------------------------------------- point 2: validate_verdict

class TestValidateVerdictUnrestricted:
    def test_off_list_name_accepted_by_default(self, monkeypatch):
        monkeypatch.setattr(config, "reaction_emojis", [], raising=False)
        v = ParticipationEngine.validate_verdict({"action": "react", "emoji": ":joy:"})
        assert v.action == "react" and v.emoji == "joy"

    def test_malformed_name_downgrades_to_ignore(self, monkeypatch):
        monkeypatch.setattr(config, "reaction_emojis", [], raising=False)
        v = ParticipationEngine.validate_verdict({"action": "react", "emoji": "bad name!"})
        assert v.action == "ignore"

    def test_allowlist_enforced_when_configured(self, monkeypatch):
        monkeypatch.setattr(config, "reaction_emojis", ["thumbsup", "eyes"], raising=False)
        # off-list falls back to first allowlisted emoji
        v = ParticipationEngine.validate_verdict({"action": "react", "emoji": "joy"})
        assert v.action == "react" and v.emoji == "thumbsup"


# -------------------------------------------------------------- point 2: tool-enabled gate

def test_gate_registers_react_with_empty_default(monkeypatch):
    from slack_client.base import SlackBot
    monkeypatch.setattr(config, "enable_history_tools", False)
    monkeypatch.setattr(config, "enable_reactions", True)
    monkeypatch.setattr(config, "enable_react_tool", True)
    monkeypatch.setattr(config, "reaction_emojis", [])  # empty default = unrestricted
    monkeypatch.setattr(config, "enable_search_tool", False)
    monkeypatch.setattr(config, "enable_channel_memory", False)
    monkeypatch.setattr(config, "enable_read_document_tool", False)
    s = MagicMock()
    s.get_history_tools_for_openai.return_value = []
    s.get_react_tool_schema.return_value = {
        "type": "function", "name": "react_to_message", "parameters": {}}
    registry = SlackBot._build_tool_registry(s)
    assert "react_to_message" in {t["name"] for t in registry.schemas()}


# ----------------------------------------------------------------- point 2: classifier line

async def _capture_classifier_prompt(monkeypatch):
    """Run classify_participation with a stubbed API call; return the user-message text
    (which carries the rendered signal lines)."""
    from openai_client.api import responses as responses_api
    captured = {}

    async def _fake_safe_api_call(self, fn, *, operation_type, **params):
        captured["input"] = params["input"]
        return SimpleNamespace(output=[])

    host = MagicMock()
    host._safe_api_call = _fake_safe_api_call.__get__(host)
    host.classify_participation = responses_api.classify_participation.__get__(host)
    await host.classify_participation(text="hi", signals={})
    return captured["input"][1]["content"]


@pytest.mark.asyncio
async def test_classifier_signal_line_any_emoji_by_default(monkeypatch):
    monkeypatch.setattr(config, "reaction_emojis", [], raising=False)
    prompt = await _capture_classifier_prompt(monkeypatch)
    assert "any standard Slack emoji name (shorthand, no colons)" in prompt


@pytest.mark.asyncio
async def test_classifier_signal_line_allowlist_when_configured(monkeypatch):
    monkeypatch.setattr(config, "reaction_emojis", ["thumbsup", "eyes"], raising=False)
    prompt = await _capture_classifier_prompt(monkeypatch)
    assert "Allowed reaction emoji (choose one): thumbsup, eyes" in prompt


# --------------------------------------------------------------- point 3: pulse social proof

def _entry(ts, text="hello", thread_ts=None, name="Alice", sender="human"):
    return dict(ts=ts, thread_ts=thread_ts, user_id="U1", display_name=name,
                sender_type=sender, text=text, is_bot=sender != "human")


def test_pulse_accumulates_and_decrements_keyed_by_ts():
    p = ChannelPulse(size=5)
    p.add_reaction("C1", "1.0", "joy")
    p.add_reaction("C1", "1.0", "joy")
    p.add_reaction("C1", "1.0", "fire")
    p.add_reaction("C1", "2.0", "tada")
    assert p.render_reactions("C1", "1.0") == "[reactions: 2× joy, 1× fire]"
    assert p.render_reactions("C1", "2.0") == "[reactions: 1× tada]"
    # decrement on removed, keyed by ts
    p.remove_reaction("C1", "1.0", "fire")
    assert p.render_reactions("C1", "1.0") == "[reactions: 2× joy]"
    p.remove_reaction("C1", "1.0", "joy")
    p.remove_reaction("C1", "1.0", "joy")
    assert p.render_reactions("C1", "1.0") == ""  # pruned when empty


def test_pulse_reactions_dm_excluded_and_colon_stripped():
    p = ChannelPulse(size=5)
    p.add_reaction("D1", "1.0", "joy")  # DM excluded
    assert p.render_reactions("D1", "1.0") == ""
    p.add_reaction("C1", "1.0", ":thumbsup::skin-tone-2:")  # folded to base
    assert p.render_reactions("C1", "1.0") == "[reactions: 1× thumbsup]"


def test_pulse_render_top2_deterministic():
    p = ChannelPulse(size=5)
    # insertion order shouldn't affect output (sorted by count desc then name)
    for e in ["b", "b", "b", "a", "a", "a", "c"]:
        p.add_reaction("C1", "1.0", e)
    first = p.render_reactions("C1", "1.0")
    assert first == "[reactions: 3× a, 3× b]"  # tie broken by name; top 2 only
    p2 = ChannelPulse(size=5)
    for e in ["c", "a", "b", "a", "b", "a", "b"]:
        p2.add_reaction("C1", "1.0", e)
    assert p2.render_reactions("C1", "1.0") == "[reactions: 3× a, 3× b]"


def test_envelope_appends_reaction_summary_and_omits_when_none():
    p = ChannelPulse(size=5)
    p.record("C1", **_entry("1.0", text="landed a big win"))
    p.record("C1", **_entry("2.0", text="quiet message"))
    p.add_reaction("C1", "1.0", "tada")
    p.add_reaction("C1", "1.0", "tada")
    env = p.render_envelope("C1")
    assert "landed a big win [reactions: 2× tada]" in env
    assert "quiet message" in env and "quiet message [reactions" not in env


def test_thread_tail_appends_reaction_summary():
    p = ChannelPulse(size=5)
    root = "10.0"
    p.record("C1", **_entry(root, text="root msg"))
    p.record("C1", **_entry("11.0", text="reply one", thread_ts=root))
    p.add_reaction("C1", "11.0", "fire")
    tail = p.render_thread_tail("C1", root, before_ts="99.0")
    assert '[reactions: 1× fire]' in tail


# ------------------------------------------------------- point 3: own-message feedback intact

class _Host:
    def __init__(self, pulse):
        self.channel_pulse = pulse
        self.bot_user_id = "UBOT"
        self.db = SimpleNamespace(record_response_feedback_async=AsyncMock())

    def log_debug(self, *a, **k):
        pass


def _reaction_event(reaction="tada", item_user="UBOT", user="U1", channel="C1", ts="9.9"):
    return {"type": "reaction_added", "reaction": reaction, "user": user,
            "item_user": item_user, "item": {"type": "message", "channel": channel, "ts": ts}}


@pytest.mark.asyncio
async def test_own_message_reaction_still_reaches_feedback_sink():
    # F20 pulse update is additive; the feedback path for the bot's OWN messages is intact.
    p = ChannelPulse(size=5)
    host = _Host(p)
    event = _reaction_event(reaction="+1", item_user="UBOT")  # +1 maps to a feedback signal
    await feedback_handlers.ingest_reaction(host, event)
    host.db.record_response_feedback_async.assert_awaited_once()
    # and the additive pulse update records the same reaction in-memory
    feedback_handlers.note_reaction_pulse(host, event, added=True)
    assert p.render_reactions("C1", "9.9") == "[reactions: 1× +1]"


def test_note_reaction_pulse_removed_decrements():
    p = ChannelPulse(size=5)
    host = _Host(p)
    ev = _reaction_event(reaction="joy", ts="9.9")
    feedback_handlers.note_reaction_pulse(host, ev, added=True)
    assert p.render_reactions("C1", "9.9") == "[reactions: 1× joy]"
    feedback_handlers.note_reaction_pulse(host, ev, added=False)
    assert p.render_reactions("C1", "9.9") == ""
