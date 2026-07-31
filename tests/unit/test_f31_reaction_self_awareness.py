"""F31 — reaction self-awareness.

Everything this file used to cover lived on ChannelPulse: `record_own_reaction`'s bookkeeping
(attribution + excerpt + truncation, generic form, thread landing, DM exclusion), the
`_reserve_and_react` choke-point hook that fed it, and the end-to-end proof that a reaction the
RESPONDER places becomes something the bot can later say it did. ChannelPulse is retired
(2026-07-27) and `_reserve_and_react` calls no such hook any more — reactions are no longer
remembered in memory at all, so none of that has a successor to test here.

What survives is the one property that was never about remembering — it was about the GATE never
placing anything in the first place. That used to be checked by diffing the pulse's rendered
envelope before and after a gate pass; with no pulse to diff, it is checked directly against the
only place a reaction could have landed: Slack's reactions.add.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from config import config
from main import ChatBotV2
from message_processor.participation import GateEvaluation, WakeDecision
from slack_client.messaging import SlackMessagingMixin


class _ReactHost(SlackMessagingMixin):
    """A real messaging mixin over a fake Slack client, so the assertion below is about the
    call the gate does or doesn't make — not about a memory structure that no longer exists."""

    def __init__(self):
        self.added = []

        async def reactions_add(channel, name, timestamp):
            self.added.append((channel, name, timestamp))

        self.app = SimpleNamespace(client=SimpleNamespace(reactions_add=reactions_add))

    def log_debug(self, *a, **k):
        pass

    log_info = log_warning = log_error = log_debug


@pytest.mark.asyncio
@pytest.mark.parametrize("wake", [True, False])
async def test_the_gate_places_no_reaction(monkeypatch, wake):
    """The tripwire the deleted test becomes: a gate pass must never touch Slack's reaction API,
    on either outcome, because the binary gate returns one bit and places nothing in the room. A
    reaction landing here would make the bot claim, later and in good faith, to have reacted to
    something it never touched."""
    monkeypatch.setattr(config, "enable_reactions", True)
    monkeypatch.setattr(config, "enable_participation_engine", True)
    monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
    host = _ReactHost()

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

    decision = await bot._run_participation_gate(message, host)
    assert (decision is not None) is wake
    assert host.added == []
