"""F20 — human-style reactions on others' posts.

Covers what remains of the five design points: the main model's react etiquette
(LOCAL_TOOLS_GUIDANCE), unrestricted standard-emoji judgment at the two enforcement points that
still exist (schema enum, executor) with the optional REACTION_EMOJIS allowlist still honored, and
tool registration.

TWO OF THE FOUR ENFORCEMENT POINTS ARE GONE, along with the prompt they served. The gate used to
CHOOSE the emoji and place it — hence a react verdict, `validate_verdict`'s emoji coercion, and a
rendered palette/allowlist line in the gate prompt. The binary gate returns one bit and places
nothing, so the choosing moved wholly to the responder, where the schema enum and the executor
already enforced the same rules. What is deleted here is the duplicate enforcement on a chooser
that no longer exists; what survives is the enforcement on the one that does.

The F20/F24 prompt tests went with PARTICIPATION_SYSTEM_PROMPT. Their content — a reaction clears a
lower bar than words, ownership still governs, prefer an emoji when it fully carries the reply, do
not strain for a joke — was guidance for CHOOSING an emoji, which the responder's own prompt and
tool guidance now own; the surviving assertions are on LOCAL_TOOLS_GUIDANCE below.

The pulse-ring social-proof signal (reaction accumulation, envelope rendering, the workspace
emoji tally, and the bot's own-reaction bookkeeping) is gone with ChannelPulse itself (retired
2026-07-27) — reactions are no longer remembered in memory at all, so there is nothing left here
to test.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import config, valid_emoji_name
from tool_registry import ToolContext
from slack_client.messaging import SlackMessagingMixin
from message_processor.participation import ParticipationEngine
import prompts


# --------------------------------------------------- point 1+4: the responder's etiquette

def test_the_gate_prompt_carries_no_reaction_guidance_at_all():
    """The inverted remains of two deleted tests.

    They asserted a dozen sentences of the rich gate's reaction rubric: the lower bar, the ownership
    rule, the social-proof gate, the taste rails, the words > emoji > nothing ladder, "Apt beats
    safe", the any-standard-emoji wording, and F24's preference for reacting when an emoji fully
    carries the reply. Every one of them was instruction for CHOOSING an emoji, and the gate chooses
    nothing — so there is no successor sentence, and asserting one would invent a contract.

    The measurement that produced those sentences is not lost: it argued that reactions should clear
    a lower bar than words, which is now the responder's judgment to make with the whole thread in
    front of it rather than a classifier's with one message."""
    p = prompts.WAKE_CLASSIFIER_SYSTEM_PROMPT
    for retired in ("A reaction clears a lower bar", "does not take the floor",
                    "Most messages get nothing", "the emoji is the cheaper mistake",
                    "prefer nothing", "Apt beats safe", "do not strain for a joke",
                    "any standard Slack emoji name", 'prefer "react" over "respond"',
                    "ALREADY acknowledged with a reaction"):
        assert retired not in p, retired
    # The one thing it DOES say about reacting: that the assistant it wakes may react instead of
    # speaking. That is context for the wake decision, not instruction about which emoji.
    assert "add an emoji reaction instead of speaking" in p


def test_local_tools_guidance_softened():
    g = prompts.LOCAL_TOOLS_GUIDANCE
    assert "react the way a teammate does" in g
    assert "when the room is already reacting" in g
    # absolutism removed
    assert "Most messages deserve NO reaction" not in g
    # still one-per-message rail
    assert "one emoji per target message" in g


# ------------------------------------------------------------------ F24: reaction-preference

def test_f24_local_tools_guidance_broadened():
    g = prompts.LOCAL_TOOLS_GUIDANCE
    # broadened beyond "thanks!" to acknowledgments/delegations/FYIs
    assert "got it" in g and "delegation" in g and "FYI" in g
    assert "while I'm out" in g


# ------------------------------------------------------------------ point 2: schema enum

def _mixin_host():
    s = MagicMock()
    s._react_tool_schema = SlackMessagingMixin._react_tool_schema.__get__(s)
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

    # F38: the guard calls _react_add now — delegate to the `react` stub these tests drive.
    async def _react_add(channel_id, ts, emoji):
        ok = await s.react(channel_id, ts, emoji)
        return ok, ok
    s._react_add = _react_add
    s.execute_react_tool = SlackMessagingMixin.execute_react_tool.__get__(s)
    s._reserve_and_react = SlackMessagingMixin._reserve_and_react.__get__(s)
    s._reserve_and_react_owned = SlackMessagingMixin._reserve_and_react_owned.__get__(s)
    s._reserve_once = SlackMessagingMixin._reserve_once.__get__(s)
    s.settle_reaction_lease = SlackMessagingMixin.settle_reaction_lease.__get__(s)
    s._is_committed = SlackMessagingMixin._is_committed
    s._REMOVING = SlackMessagingMixin._REMOVING
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


# ------------------------------------- point 2: the verdict-side enforcement is gone

def test_the_verdict_side_emoji_enforcement_is_gone():
    """Five tests collapse here, and nothing they covered is unprotected.

    They exercised `validate_verdict`'s emoji handling: an off-list name accepted by default, a
    malformed one downgrading react → ignore, an allowlist coercing to its first entry, and the two
    react_and_respond variants (off-list falls back and stays; unresolvable drops the emoji but
    keeps the reply). All of it repaired an emoji the GATE had chosen so the gate could place it.

    There is no verdict and no gate-placed reaction. The identical rules are enforced where the
    emoji is actually chosen — the schema enum above and the executor above that — and the executor
    tests cover the same three cases against the live path."""
    assert not hasattr(ParticipationEngine, "validate_verdict")
    assert not hasattr(ParticipationEngine, "_coerce_emoji")
    assert not hasattr(ParticipationEngine, "_apply_invariants")
    import message_processor.participation as participation
    assert not hasattr(participation, "ParticipationVerdict")
    # No action vocabulary survives anywhere in the module either — `react` and
    # `react_and_respond` were the only reasons an emoji ever reached it.
    assert not hasattr(participation, "VALID_ACTIONS")
    assert not hasattr(participation, "SPEAKING_ACTIONS")


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
    # Every schema the builder registers unconditionally must be a real dict: a bare MagicMock
    # is CALLABLE, and register() reads a callable as a schema factory (F34) — which is a
    # registration error, not a react_to_message failure.
    s.get_post_to_thread_tool_schema.return_value = {
        "type": "function", "name": "post_to_thread", "parameters": {}}
    s.get_edit_own_message_tool_schema.return_value = {
        "type": "function", "name": "edit_own_message", "parameters": {}}
    # T5: the toolbelt round's two new client-side schemas.
    s.get_delete_own_message_tool_schema.return_value = {
        "type": "function", "name": "delete_own_message", "parameters": {}}
    s.get_remove_reaction_tool_schema.return_value = {
        "type": "function", "name": "remove_reaction", "parameters": {}}
    s.get_pin_message_tool_schema.return_value = {
        "type": "function", "name": "pin_message", "parameters": {}}
    s.get_no_reply_tool_schema.return_value = {
        "type": "function", "name": "no_response_needed", "parameters": {}}
    s.get_emoji_search_tool_schema.return_value = {
        "type": "function", "name": "search_workspace_emoji", "parameters": {}}
    registry = SlackBot._build_tool_registry(s)
    assert "react_to_message" in {t["name"] for t in registry.schemas()}


# ------------------------------------------- point 2: the gate is told about no palette

@pytest.mark.parametrize("allowlist", [[], ["thumbsup", "eyes"]])
def test_the_gate_prompt_renders_neither_a_palette_nor_an_allowlist(monkeypatch, allowlist):
    """INVERTED from two tests that asserted the rendered signal line, one per allowlist state.

    The line existed so the gate could pick from a legal set. With no picking, rendering it would
    describe an ability the gate does not have — and the allowlist itself is still honoured, at the
    schema and the executor, which is where a name the model proposes actually gets checked."""
    monkeypatch.setattr(config, "reaction_emojis", allowlist, raising=False)
    import inspect
    from openai_client.api import responses

    src = inspect.getsource(responses.classify_wake)
    assert "reaction_emojis" not in src
    assert "Allowed reaction emoji" not in src
    # ...and nothing in the rendered source block either: it describes the message, not the palette.
    from message_processor.participation import SourceMessage
    block = responses._render_wake_source(
        SourceMessage(ts="1.0", text="hi", sender_name="Peter", sender_type="human"),
        index=0, total=1)
    assert "emoji" not in block.lower()
