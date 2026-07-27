"""F20 — human-style reactions on others' posts.

Covers what remains of the five design points: the main model's react etiquette
(LOCAL_TOOLS_GUIDANCE), unrestricted standard-emoji judgment at the two enforcement points that
still exist (schema enum, executor) with the optional REACTION_EMOJIS allowlist still honored, tool
registration, and the pulse-ring social-proof signal (reaction_added/removed accumulation +
envelope/tail summary rendering).

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
    s.get_no_reply_tool_schema.return_value = {
        "type": "function", "name": "no_response_needed", "parameters": {}}
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


# --- the palette must learn from the ROOM, not from the bot (2026-07-26) ---

def test_own_reactions_never_teach_the_emoji_palette():
    """Slack emits reaction_added for the bot's OWN reactions, so without a filter the usage
    tally learns from the bot: an emoji it picked once outranks the rest next turn, gets picked
    again, and climbs. Observed live — :dumpster-fire: went 0 -> 2 on the bot's own two uses and
    was then the top custom emoji offered for anything negative, in a workspace with 1,397 custom
    emoji. The tally answers "what does THIS WORKSPACE react with"; the bot is not the workspace."""
    from slack_client.channel_pulse import ChannelPulse
    pulse = ChannelPulse(size=10)
    pulse.add_reaction("C1", "1.0", "dumpster-fire", from_self=True)
    pulse.add_reaction("C1", "1.0", "tada")                       # a human's
    vocab = pulse.reaction_vocab_snapshot()
    assert "dumpster-fire" not in vocab, "the bot taught itself its own preference"
    assert vocab.get("tada") == 1
    # Social proof on the specific message is different: our reaction really IS on it.
    assert "dumpster-fire" in pulse.render_reactions("C1", "1.0")


def test_removing_our_own_reaction_does_not_dock_the_room_s_tally():
    """Ours never incremented the tally, so un-reacting must not decrement it — otherwise the bot
    reacting then removing would push a genuinely popular emoji BELOW what the room earned it."""
    from slack_client.channel_pulse import ChannelPulse
    pulse = ChannelPulse(size=10)
    for _ in range(3):
        pulse.add_reaction("C1", "1.0", "tada")                   # the room's three
    pulse.add_reaction("C1", "1.0", "tada", from_self=True)       # ours on top
    pulse.remove_reaction("C1", "1.0", "tada", from_self=True)    # ours taken back
    assert pulse.reaction_vocab_snapshot().get("tada") == 3
