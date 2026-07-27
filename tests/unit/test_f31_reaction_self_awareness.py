"""F31 — reaction self-awareness.

The bot must remember every reaction IT places so "did you react to that?" is
answerable from context. Covers the ChannelPulse.record_own_reaction bookkeeping
(attribution + excerpt + truncation, generic form, thread landing, DM exclusion),
the _reserve_and_react choke-point hook (fires on commit, not on failure/duplicate),
and the RESPONDER's react path end-to-end.

The gate used to have a react verdict of its own, and the end-to-end test at the bottom drove it.
That verdict is gone — the gate returns one bit and places nothing in the room — so the same
end-to-end property is now asserted through the reaction the responder actually makes, plus a
tripwire that the gate leaves the pulse alone on every outcome.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from config import config
from main import ChatBotV2
from message_processor.participation import GateEvaluation, WakeDecision
from slack_client.channel_pulse import ChannelPulse
from slack_client.messaging import SlackMessagingMixin


def _human(ts, text="hello", thread_ts=None, name="Alice"):
    return dict(ts=ts, thread_ts=thread_ts, user_id="U1", display_name=name,
                sender_type="human", text=text, is_bot=False)


# ---------------------------------------------------------- record_own_reaction

def test_record_own_reaction_attribution_and_excerpt():
    p = ChannelPulse(size=10)
    p.record("C1", **_human("100.0", text="Fable limit has been removed for everyone",
                            name="Kousha Mazloumi"))
    p.record_own_reaction("C1", message_ts="100.0", emoji="tada")
    env = p.render_envelope("C1")
    assert "reacted :tada: to Kousha Mazloumi's message:" in env
    assert "Fable limit has been removed" in env


def test_record_own_reaction_truncates_long_excerpt():
    p = ChannelPulse(size=10)
    long_text = "word " * 60  # ~300 chars
    p.record("C1", **_human("100.0", text=long_text, name="Bob"))
    p.record_own_reaction("C1", message_ts="100.0", emoji="eyes")
    env = p.render_envelope("C1")
    # Excerpt head-truncated to ~80 chars with an ellipsis.
    assert "…\"]" in env
    excerpt = env.split('message: "', 1)[1].split('"]', 1)[0]
    assert len(excerpt) <= 81


def test_record_own_reaction_generic_when_target_missing():
    p = ChannelPulse(size=10)
    p.record_own_reaction("C1", message_ts="999.0", emoji="tada")
    env = p.render_envelope("C1")
    assert "reacted :tada: to an earlier message" in env
    assert "'s message" not in env


def test_record_own_reaction_thread_target_lands_in_thread_tail():
    p = ChannelPulse(size=10)
    # A reply inside a thread rooted at 50.0.
    p.record("C1", **_human("60.0", text="a reply in the thread", thread_ts="50.0",
                            name="Carol"))
    p.record_own_reaction("C1", message_ts="60.0", emoji="fire")
    # The synthetic entry must appear in that thread's classifier tail.
    tail = p.render_thread_tail("C1", "50.0", before_ts=None)
    assert "reacted :fire:" in tail


def test_record_own_reaction_dm_excluded():
    p = ChannelPulse(size=10)
    p.record_own_reaction("D123", message_ts="100.0", emoji="tada")
    assert p.render_envelope("D123") == ""


def test_record_own_reaction_uses_bot_alias_name(monkeypatch):
    monkeypatch.setattr(config, "bot_name_aliases", ["Sol", "ChatGPT"])
    p = ChannelPulse(size=10)
    p.record("C1", **_human("100.0", text="hi", name="Dan"))
    p.record_own_reaction("C1", message_ts="100.0", emoji="wave")
    # Rendered as a self entry under the first alias, landing in the TARGET's thread
    # (a root target IS its own thread root — Codex review fix, never a bogus top-level).
    env = p.render_envelope("C1")
    assert 'Sol (in thread "hi…"): [reacted :wave:' in env


# --------------------------------------------- _reserve_and_react choke-point hook

class _ReactHost(SlackMessagingMixin):
    def __init__(self, reactions_add, pulse):
        self.app = SimpleNamespace(client=SimpleNamespace(reactions_add=reactions_add))
        self.channel_pulse = pulse

    def log_debug(self, *a, **k):
        pass

    log_info = log_warning = log_error = log_debug


@pytest.mark.asyncio
async def test_reserve_and_react_hook_fires_on_commit(monkeypatch):
    monkeypatch.setattr(config, "enable_reactions", True)
    pulse = ChannelPulse(size=10)
    pulse.record("C1", **_human("100.0", text="great point", name="Eve"))
    host = _ReactHost(AsyncMock(), pulse)
    res = await host._reserve_and_react("C1", "100.0", "tada")
    assert res["ok"] is True
    assert "reacted :tada: to Eve's message" in pulse.render_envelope("C1")


@pytest.mark.asyncio
async def test_reserve_and_react_hook_not_on_failure(monkeypatch):
    from slack_sdk.errors import SlackApiError
    monkeypatch.setattr(config, "enable_reactions", True)
    pulse = ChannelPulse(size=10)
    pulse.record("C1", **_human("100.0", text="nope", name="Eve"))
    fail = AsyncMock(side_effect=SlackApiError("message_not_found",
                                              {"error": "message_not_found"}))
    host = _ReactHost(fail, pulse)
    res = await host._reserve_and_react("C1", "100.0", "tada")
    assert res["ok"] is False
    assert "reacted" not in pulse.render_envelope("C1")


@pytest.mark.asyncio
async def test_reserve_and_react_hook_records_once_on_duplicate(monkeypatch):
    monkeypatch.setattr(config, "enable_reactions", True)
    pulse = ChannelPulse(size=10)
    pulse.record("C1", **_human("100.0", text="great point", name="Eve"))
    pulse.record_own_reaction = Mock(wraps=pulse.record_own_reaction)
    host = _ReactHost(AsyncMock(), pulse)
    await host._reserve_and_react("C1", "100.0", "tada")
    await host._reserve_and_react("C1", "100.0", "tada")  # duplicate — idempotent, no slot
    assert pulse.record_own_reaction.call_count == 1


# --------------------------------------- the reaction the RESPONDER makes, end to end

@pytest.mark.asyncio
async def test_the_responders_react_tool_records_its_own_reaction(monkeypatch):
    """RE-BASELINED to the actor that still exists.

    This was the gate's react verdict end-to-end: `evaluate` returned action=react with an emoji,
    `_run_participation_gate` placed it and returned None, and the pulse learned about it. The
    verdict is gone, so the same property — an emoji WE placed becomes part of what the bot can
    later say it did — is asserted through `react_to_message`, which is how every reaction reaches
    the room now. The choke point (`_reserve_and_react`) and the bookkeeping are unchanged, which
    is why nothing about the assertion needed weakening."""
    monkeypatch.setattr(config, "enable_reactions", True)
    monkeypatch.setattr(config, "enable_react_tool", True)
    monkeypatch.setattr(config, "reaction_emojis", ["tada"], raising=False)
    pulse = ChannelPulse(size=10)
    pulse.record("C1", **_human("100.0", text="Fable limit removed for everyone?",
                                name="Kousha"))
    host = _ReactHost(AsyncMock(), pulse)
    host.execute_react_tool = SlackMessagingMixin.execute_react_tool.__get__(host)

    out = await host.execute_react_tool(
        SimpleNamespace(channel_id="C1", trigger_ts="100.0", thread_ts="100.0",
                        attempt_id=None),
        {"emoji": "tada"})
    assert out["ok"] is True
    assert "reacted :tada: to Kousha's message" in pulse.render_envelope("C1")


@pytest.mark.asyncio
@pytest.mark.parametrize("wake", [True, False])
async def test_the_gate_leaves_the_pulse_untouched(monkeypatch, wake):
    """The tripwire the deleted test becomes: a gate pass must add nothing to the pulse, because it
    puts nothing in the room to add. A reaction recorded here without one on the message would make
    the bot claim, later and in good faith, to have reacted to something it never touched."""
    monkeypatch.setattr(config, "enable_reactions", True)
    monkeypatch.setattr(config, "enable_participation_engine", True)
    monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
    pulse = ChannelPulse(size=10)
    pulse.record("C1", **_human("100.0", text="Fable limit removed for everyone?",
                                name="Kousha"))
    host = _ReactHost(AsyncMock(), pulse)

    bot = ChatBotV2(platform="slack")
    bot.processor = Mock()
    bot.processor.mcp_manager = None
    engine = Mock()
    engine.note_arrival = Mock()
    engine.evaluate = AsyncMock(return_value=GateEvaluation(decision=WakeDecision(wake=wake)))
    bot.participation_engine = engine

    message = SimpleNamespace(
        channel_id="C1", thread_id="100.0", user_id="U9",
        text="Fable limit removed for everyone?",
        attachments=[],  # real Message defaults this to [] in __post_init__; the gate reads it
        metadata={"ts": "100.0", "participation_level": "on"})

    before = pulse.render_envelope("C1")
    decision = await bot._run_participation_gate(message, host)
    assert (decision is not None) is wake
    assert pulse.render_envelope("C1") == before
    assert "reacted" not in pulse.render_envelope("C1")
