"""F31 — reaction self-awareness.

The bot must remember every reaction IT places so "did you react to that?" is
answerable from context. Covers the ChannelPulse.record_own_reaction bookkeeping
(attribution + excerpt + truncation, generic form, thread landing, DM exclusion),
the _reserve_and_react choke-point hook (fires on commit, not on failure/duplicate),
and the main.py verdict-react path end-to-end.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from config import config
from main import ChatBotV2
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


# ----------------------------------------------- main.py verdict-react integration

@pytest.mark.asyncio
async def test_verdict_react_path_records_own_reaction(monkeypatch):
    monkeypatch.setattr(config, "enable_reactions", True)
    monkeypatch.setattr(config, "enable_participation_engine", True)
    pulse = ChannelPulse(size=10)
    pulse.record("C1", **_human("100.0", text="Fable limit removed for everyone?",
                                name="Kousha"))
    host = _ReactHost(AsyncMock(), pulse)

    bot = ChatBotV2(platform="slack")
    bot.processor = Mock()
    bot.processor.mcp_manager = None
    engine = Mock()
    engine.note_arrival = Mock()
    engine.evaluate = AsyncMock(
        return_value=SimpleNamespace(action="react", emoji="tada", reason="fun"))
    bot.participation_engine = engine

    message = SimpleNamespace(
        channel_id="C1", thread_id="100.0", user_id="U9",
        text="Fable limit removed for everyone?",
        attachments=[],  # real Message defaults this to [] in __post_init__; the gate reads it
        metadata={"ts": "100.0", "participation_level": "judicious"})

    result = await bot._run_participation_gate(message, host)
    assert result is None  # react verdicts stay silent
    assert "reacted :tada: to Kousha's message" in pulse.render_envelope("C1")
