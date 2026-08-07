"""The four routing facts: stamped once at dispatch, read everywhere else.

`gate_required`, `gate_woke`, `routing_posture` and `silence_capable` replace one overloaded
boolean that four different consumers each re-derived their own way. These tests hold the
replacement to the contract that made it worth doing:

1. EVERY dispatch route stamps ALL four — including the routes where three of them are False.
   An absent fact and a False fact must never be the same thing downstream.
2. The values match the route truth table, so the rename cannot quietly change who may stay
   silent or who must pass the gate.
3. `routing_posture` comes from addressing and topology ONLY. A message that says our name is
   discussing us, not addressing us, and it keeps the posture it would have had anyway.
4. `gate_woke` cannot exist without `gate_required` — the illegal state is unconstructable
   rather than merely undocumented.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from base_client import Message
from config import config
from message_processor import routing_facts
from message_processor.turn_runtime import TurnRuntime
from slack_client.event_handlers.message_events import SlackMessageEventsMixin
from slack_client.utilities import SlackUtilitiesMixin


class _Bot(SlackMessageEventsMixin, SlackUtilitiesMixin):
    def log_debug(self, *a, **k):
        pass

    def log_info(self, *a, **k):
        pass

    def log_warning(self, *a, **k):
        pass

    def log_error(self, *a, **k):
        pass


async def _e2m(event, client):
    return Message(text=event.get("text", ""), user_id=event.get("user"),
                   channel_id=event.get("channel"),
                   thread_id=event.get("thread_ts") or event.get("ts"),
                   attachments=[], metadata={"ts": event.get("ts")})


def _make_bot(cs=None, sender_type="human"):
    bot = _Bot.__new__(_Bot)
    bot.bot_user_id = "UBOT"
    bot.bot_id = "BBOT"
    bot.app_id = None
    bot.message_handler = AsyncMock()
    bot.app = MagicMock()
    bot.app.client = MagicMock()
    bot.db = MagicMock()
    bot.db.get_user_preferences_async = AsyncMock(return_value={"settings_completed": True})
    bot._get_channel_settings = AsyncMock(return_value=cs)
    bot._event_to_message = _e2m
    bot._thread_participation = AsyncMock(return_value=(False, 1, 0))
    bot._post_settings_button_if_new_thread = AsyncMock()
    bot._maybe_set_assistant_thread_title = AsyncMock()
    bot._stash_edit_context = MagicMock()
    bot._register_edit_dispatch = MagicMock()
    return bot


def _evt(**kw):
    e = {"channel": "C1", "ts": "100.1", "user": "UHUMAN", "text": "hello there",
         "channel_type": "channel"}
    e.update(kw)
    return e


def _dispatched(bot):
    return bot.message_handler.await_args.args[0].metadata


@pytest.fixture
def listening(monkeypatch):
    """A channel at the talkative level. Named for what it means now: `judicious` was one of two
    restraint dials on the rich gate and both collapsed into `on` when the gate became one bit."""
    monkeypatch.setattr(config, "channel_response_mode", "auto_respond", raising=False)
    monkeypatch.setattr(config, "bot_name_aliases", ["ChatGPT"], raising=False)
    monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
    monkeypatch.setattr(config, "enable_channel_listening", False, raising=False)


# ------------------------------------------------------------- the route truth table

@pytest.mark.asyncio
async def test_ambient_channel_message_is_gated_and_may_stay_silent(listening):
    bot = _make_bot()
    await bot._handle_channel_message(_evt(text="anyone know the q3 numbers?"), bot.app.client)
    md = _dispatched(bot)
    assert md["gate_required"] is True
    assert md["silence_capable"] is True
    assert md["routing_posture"] == "channel_activity"
    assert md["gate_woke"] is False          # nothing has run the gate yet


@pytest.mark.asyncio
async def test_direct_thread_continuation_skips_the_gate_but_keeps_the_silence_option(
        listening, monkeypatch):
    """The strict 1:1 continuation runs no gate, so the model is the only decider — and it may
    still decide there is nothing to add. Strict means strict: the bot, one human, no other
    agents, and it holds at every participation level."""
    bot = _make_bot()
    bot._thread_participation = AsyncMock(return_value=(True, 1, 0))  # bot + one human
    await bot._handle_channel_message(
        _evt(text="and what about friday?", thread_ts="50.0", ts="60.0"), bot.app.client)
    md = _dispatched(bot)
    assert md["gate_required"] is False
    assert md["silence_capable"] is True
    assert md["routing_posture"] == "thread_activity"
    assert md["wake_source"] == "thread_continuation"   # provenance is untouched


@pytest.mark.asyncio
async def test_engine_off_legacy_name_wake_owes_an_answer(listening, monkeypatch):
    """No engine means no judgment: the deterministic wake answers, and it does not get the
    silence option a judged turn gets."""
    monkeypatch.setattr(config, "enable_participation_engine", False, raising=False)
    bot = _make_bot()
    await bot._handle_channel_message(_evt(text="ChatGPT, what's the weather?"), bot.app.client)
    md = _dispatched(bot)
    assert md["gate_required"] is False
    assert md["silence_capable"] is False
    assert md["routing_posture"] == "channel_activity"


@pytest.mark.asyncio
async def test_app_mention_is_addressed_and_owes_an_answer(listening):
    bot = _make_bot({"response_mode": "auto_respond"})
    await bot._handle_slack_message(_evt(text="<@UBOT> hi"), bot.app.client,
                                    wake_source="app_mention")
    md = _dispatched(bot)
    assert md["gate_required"] is False
    assert md["silence_capable"] is False
    assert md["routing_posture"] == "addressed_to_assistant"


@pytest.mark.asyncio
async def test_dm_is_addressed_and_owes_an_answer(listening):
    bot = _make_bot()
    await bot._handle_slack_message(_evt(channel="D1", text="hey"), bot.app.client,
                                    wake_source="dm")
    md = _dispatched(bot)
    assert md["gate_required"] is False
    assert md["silence_capable"] is False
    assert md["routing_posture"] == "addressed_to_assistant"


@pytest.mark.asyncio
async def test_a_dm_thread_reply_is_still_addressed_not_thread_activity(listening):
    """Topology never overrides explicit addressing: every message in a DM is for us."""
    bot = _make_bot()
    await bot._handle_slack_message(_evt(channel="D1", ts="60.0", thread_ts="50.0"),
                                    bot.app.client, wake_source="dm")
    assert _dispatched(bot)["routing_posture"] == "addressed_to_assistant"


@pytest.mark.asyncio
async def test_edit_redispatch_is_gated_ambient_traffic(listening, monkeypatch):
    bot = _make_bot({"participation_level": "on"})
    synthetic = {"channel": "C1", "ts": "200.0", "user": "UHUMAN", "text": "the numbers again"}
    await bot._dispatch_edit_to_engine(bot.app.client, synthetic, "C1", "200.0",
                                       "the numbers", "the numbers again")
    md = _dispatched(bot)
    assert md["gate_required"] is True
    assert md["silence_capable"] is True
    assert md["routing_posture"] == "channel_activity"


@pytest.mark.asyncio
async def test_a_gated_thread_message_is_thread_activity(listening):
    """A thread reply the gate judges is thread activity, and it is still gated.

    The shape had to change with the membership widening. "Another bot is in the thread" used to
    be enough to send a reply to the gate; in an `on` channel it no longer is, because we have
    posted there and that is now the wake signal. What remains gated is a thread we are NOT part
    of — no continuation rule reaches it, so only the gate can."""
    bot = _make_bot()
    bot._thread_participation = AsyncMock(return_value=(False, 2, 1))  # a thread we never joined
    await bot._handle_channel_message(
        _evt(text="what do we think?", thread_ts="50.0", ts="60.0"), bot.app.client)
    md = _dispatched(bot)
    assert md["gate_required"] is True
    assert md["routing_posture"] == "thread_activity"


@pytest.mark.asyncio
async def test_a_name_hit_does_not_make_a_message_addressed(listening):
    """The whole reason posture exists: "ChatGPT was wrong earlier" matches the name regex and
    is NOT addressed to us. The narrower name provenance is unchanged."""
    bot = _make_bot()
    await bot._handle_channel_message(_evt(text="ChatGPT was wrong earlier"), bot.app.client)
    md = _dispatched(bot)
    assert md["routing_posture"] == "channel_activity"
    assert md["participation_name_hit"] is True
    assert md["wake_source"] == "name_mention"


# -------------------------------------------------- thread membership is the wake signal
#
# The second rule that skips the gate. In a channel set to `on`, an untagged human reply in a
# thread we have ALREADY POSTED IN goes straight to the responder, whoever else is in that
# thread. Participation in a thread is itself the wake signal: a thread we have posted in is one
# we are already part of, and the responder — which can see the thread, where the gate sees only
# the trigger text — decides what the turn owes, including nothing.
#
# Two things it deliberately is not. It does not apply at `mentions_only`, where we tell the user
# verbatim that nothing but a mention or a bare name wakes us (ruling 1A). And it carries no
# structural authority, which `membership_wake` is stamped to record (ruling 2A) — see
# test_structural_authorization.py.

@pytest.mark.asyncio
async def test_membership_wakes_us_ungated_when_another_bot_is_in_the_thread(listening):
    """THE INCIDENT. In a thread where we had generated an image and named the dish, a person
    said "thanks guys" and we did nothing: another assistant was in the thread, so the strict 1:1
    rule failed, and the gate — which sees two words and no thread — could only say no. The
    reaction rule lives in the responder, which never ran."""
    bot = _make_bot()
    bot._thread_participation = AsyncMock(return_value=(True, 1, 1))  # us + a human + another bot
    await bot._handle_channel_message(
        _evt(text="thanks guys", thread_ts="50.0", ts="60.0"), bot.app.client)
    md = _dispatched(bot)
    assert md["gate_required"] is False
    assert md["silence_capable"] is True                 # reaction and silence stay terminal
    assert md["routing_posture"] == "thread_activity"
    assert md["wake_source"] == "thread_continuation"    # provenance does not split by rule
    assert md["membership_wake"] is True                 # but this fact does


@pytest.mark.asyncio
async def test_membership_wakes_us_ungated_in_a_crowded_thread(listening):
    """Several humans is the other way out of strict: same thread, same answer. Whether the reply
    was meant for us is the responder's question now, not the gate's."""
    bot = _make_bot()
    bot._thread_participation = AsyncMock(return_value=(True, 3, 0))
    await bot._handle_channel_message(
        _evt(text="and what about friday?", thread_ts="50.0", ts="60.0"), bot.app.client)
    md = _dispatched(bot)
    assert md["gate_required"] is False
    assert md["silence_capable"] is True
    assert md["wake_source"] == "thread_continuation"
    assert md["membership_wake"] is True


@pytest.mark.asyncio
async def test_mentions_only_does_not_widen_to_thread_membership(listening):
    """RULING 1A. `mentions_only` tells the user, in the responder's own words, that an explicit
    @-mention always reaches us, a bare name is weighed first, and nothing else wakes us. Waking
    on every thread reply there would make that a lie about our own configuration."""
    bot = _make_bot({"participation_level": "mentions_only"})
    bot._thread_participation = AsyncMock(return_value=(True, 1, 1))  # the incident's thread
    await bot._handle_channel_message(
        _evt(text="thanks guys", thread_ts="50.0", ts="60.0"), bot.app.client)
    bot.message_handler.assert_not_called()              # exactly as today: nothing wakes

    # And the one thing that DID reach the gate here still does, still gated, still not a
    # membership wake.
    await bot._handle_channel_message(
        _evt(text="ChatGPT what do we think?", thread_ts="50.0", ts="61.0"), bot.app.client)
    md = _dispatched(bot)
    assert md["gate_required"] is True
    assert md["membership_wake"] is False


@pytest.mark.asyncio
async def test_mentions_only_still_honours_a_strict_one_to_one_continuation(listening):
    """The narrow rule is level-INDEPENDENT and untouched by the widening: us, one human, no
    other agents. It skips the gate at `mentions_only` exactly as it always did."""
    bot = _make_bot({"participation_level": "mentions_only"})
    bot._thread_participation = AsyncMock(return_value=(True, 1, 0))
    await bot._handle_channel_message(
        _evt(text="and what about friday?", thread_ts="50.0", ts="60.0"), bot.app.client)
    md = _dispatched(bot)
    assert md["gate_required"] is False
    assert md["silence_capable"] is True
    assert md["wake_source"] == "thread_continuation"
    assert md["membership_wake"] is False               # strict, so authority is not withheld


@pytest.mark.asyncio
async def test_a_thread_we_never_posted_in_still_goes_to_the_gate(listening):
    """The accepted limitation, stated as a test. Membership is what widened; a foreign thread
    reaches neither rule and still gets only the gate, with only the trigger text."""
    bot = _make_bot()
    bot._thread_participation = AsyncMock(return_value=(False, 2, 1))
    await bot._handle_channel_message(
        _evt(text="thanks guys", thread_ts="50.0", ts="60.0"), bot.app.client)
    md = _dispatched(bot)
    assert md["gate_required"] is True
    assert md["membership_wake"] is False

    # Same at `mentions_only`, where a name hit is what buys a foreign thread its gate run.
    quiet = _make_bot({"participation_level": "mentions_only"})
    quiet._thread_participation = AsyncMock(return_value=(False, 2, 1))
    await quiet._handle_channel_message(
        _evt(text="ChatGPT any idea?", thread_ts="50.0", ts="61.0"), quiet.app.client)
    md = _dispatched(quiet)
    assert md["gate_required"] is True
    assert md["membership_wake"] is False


@pytest.mark.asyncio
async def test_another_bots_reply_in_our_thread_is_ungated_in_an_on_channel(listening):
    """Owner decision, UNCAPPED: two assistants discussing something is a thing people set up
    deliberately, and the gate — one message, no thread — is the wrong judge of whether the
    exchange is worth continuing. So a bot's reply in a thread we are part of takes the
    membership route like anyone else's.

    Nothing in code bounds the exchange. The only brake is each side deciding it has nothing to
    add, which is why `silence_capable` on this turn is the load-bearing half of the decision."""
    bot = _make_bot()
    bot._thread_participation = AsyncMock(return_value=(True, 1, 1))
    await bot._handle_channel_message(
        _evt(user="UCLAUDE", bot_id="BCLAUDE", text="I agree with the plan.",
             thread_ts="50.0", ts="60.0"), bot.app.client)
    md = _dispatched(bot)
    assert md["gate_required"] is False
    assert md["silence_capable"] is True                 # the only brake there is
    assert md["wake_source"] == "thread_continuation"
    assert md["membership_wake"] is True                 # never strict, so never authoritative


@pytest.mark.asyncio
async def test_a_bot_sender_can_never_reach_the_strict_rule(listening):
    """Strict is the route that answers with no gate AND full structural authority, so it stays
    human-only however 1:1 the thread looks. A bot in an otherwise-strict thread gets the
    membership route instead — `on`-only, and authority-free."""
    bot = _make_bot()
    bot._thread_participation = AsyncMock(return_value=(True, 1, 0))   # textbook strict shape
    await bot._handle_channel_message(
        _evt(user="UCLAUDE", bot_id="BCLAUDE", text="here's what I found.",
             thread_ts="50.0", ts="60.0"), bot.app.client)
    md = _dispatched(bot)
    assert md["gate_required"] is False
    assert md["membership_wake"] is True                 # NOT strict, despite the 1:1 counts

    # `mentions_only` is where that distinction bites: membership does not apply there (1A), and
    # strict is closed to a bot sender, so the reply falls through to the gate exactly as before.
    quiet = _make_bot({"participation_level": "mentions_only"})
    quiet._thread_participation = AsyncMock(return_value=(True, 1, 0))
    await quiet._handle_channel_message(
        _evt(user="UCLAUDE", bot_id="BCLAUDE", text="ChatGPT, what does the data say?",
             thread_ts="50.0", ts="61.0"), quiet.app.client)
    md = _dispatched(quiet)
    assert md["gate_required"] is True
    assert md["membership_wake"] is False


@pytest.mark.asyncio
async def test_a_bots_message_outside_our_threads_is_still_gated(listening):
    """The widening is about threads we are IN. A bot at top level, or in a thread we never
    posted in, still reaches the gate — that is where the loop-seed guard still lives."""
    bot = _make_bot()
    bot._thread_participation = AsyncMock(return_value=(False, 2, 1))  # a thread we never joined
    await bot._handle_channel_message(
        _evt(user="UCLAUDE", bot_id="BCLAUDE", text="I agree with the plan.",
             thread_ts="50.0", ts="60.0"), bot.app.client)
    md = _dispatched(bot)
    assert md["gate_required"] is True
    assert md["membership_wake"] is False

    top = _make_bot()
    top._thread_participation = AsyncMock(return_value=(True, 1, 1))
    await top._handle_channel_message(
        _evt(user="UCLAUDE", bot_id="BCLAUDE", text="deploy finished.", ts="70.0"),
        top.app.client)
    md = _dispatched(top)
    assert md["gate_required"] is True
    assert md["membership_wake"] is False
    top._thread_participation.assert_not_called()        # no thread, nothing to probe


@pytest.mark.asyncio
async def test_membership_wake_never_reaches_a_dm_or_a_mention(listening, mock_env, monkeypatch):
    """The fact is stamped on the CHANNEL path only, and the predicate that reads it uses
    `is True` — so every other dispatch site is byte-identical: absent is not True."""
    monkeypatch.setattr(config, "enable_no_reply_tool", True, raising=False)
    for channel, source in (("D1", "dm"), ("C1", "app_mention")):
        bot = _make_bot()
        await bot._handle_slack_message(_evt(channel=channel, text="<@UBOT> hi"),
                                        bot.app.client, wake_source=source)
        md = _dispatched(bot)
        assert "membership_wake" not in md, source

        host = _MatHost(_registry())
        _, request_config, _, _ = host._materialize_request_tools(
            host._client, {"model": "m"},
            _msg(sender_type="human", **{k: v for k, v in md.items() if k != "ts"}),
            tools_disabled=True)
        assert request_config["_structural_change_authorized"] is True, source


# ------------------------------------------------------------------ the helpers themselves

def test_posture_is_derived_from_addressing_and_topology_only():
    assert routing_facts.derive_posture(
        addressed=True, ts="1", thread_ts="1") == routing_facts.POSTURE_ADDRESSED
    assert routing_facts.derive_posture(
        addressed=False, ts="2", thread_ts="1") == routing_facts.POSTURE_THREAD
    assert routing_facts.derive_posture(
        addressed=False, ts="1", thread_ts="1") == routing_facts.POSTURE_CHANNEL
    # A top-level message keys as its own thread root; that is not thread activity.
    assert routing_facts.derive_posture(
        addressed=False, ts="1", thread_ts=None) == routing_facts.POSTURE_CHANNEL


def test_every_stamp_writes_all_four_facts():
    message = Message(text="x", user_id="U1", channel_id="C1", thread_id="1", metadata={})
    routing_facts.stamp_routing_facts(message, gate_required=False, silence_capable=False,
                                      addressed=True, ts="1", thread_ts=None)
    assert set(message.metadata) == {"gate_required", "gate_woke", "routing_posture",
                                     "silence_capable"}


def test_gate_woke_cannot_be_set_without_a_required_gate():
    """The illegal state, refused at the only place that could create it."""
    ungated = Message(text="x", user_id="U1", channel_id="C1", thread_id="1", metadata={})
    routing_facts.stamp_routing_facts(ungated, gate_required=False, silence_capable=False,
                                      addressed=True, ts="1", thread_ts=None)
    routing_facts.set_gate_woke(ungated, True)
    assert ungated.metadata["gate_woke"] is False

    gated = Message(text="x", user_id="U1", channel_id="C1", thread_id="1", metadata={})
    routing_facts.stamp_routing_facts(gated, gate_required=True, silence_capable=True,
                                      addressed=False, ts="1", thread_ts=None)
    routing_facts.set_gate_woke(gated, True)
    assert gated.metadata["gate_woke"] is True
    # A second gate run on the same object (Phase Q redispatch) overwrites, never inherits.
    routing_facts.set_gate_woke(gated, False)
    assert gated.metadata["gate_woke"] is False


# --------------------------------------------------------------- consumers read the fact

def _msg(**meta):
    return SimpleNamespace(metadata={"ts": "1.1", **meta}, channel_id="C1", thread_id="1.1")


class _MatHost:
    def __init__(self, registry):
        from message_processor.handlers.text import TextHandlerMixin
        for n in ("_materialize_request_tools", "_get_tool_registry"):
            setattr(self, n, getattr(TextHandlerMixin, n).__get__(self))
        self._client = SimpleNamespace(tool_registry=registry)


def _registry():
    from tool_registry import ToolRegistry
    reg = ToolRegistry()
    reg.register(
        {"type": "function", "name": "no_response_needed",
         "parameters": {"type": "object", "properties": {}}},
        AsyncMock(return_value={"ok": True}),
        enabled=lambda cfg: (config.enable_no_reply_tool
                             and bool(cfg.get("_silence_capable_turn"))))
    return reg


@pytest.mark.parametrize("meta, expect_silence", [
    ({"gate_required": True, "silence_capable": True}, True),
    ({"gate_required": False, "silence_capable": True}, True),      # continuation
    ({"gate_required": False, "silence_capable": False}, False),    # DM / mention
])
def test_turn_runtime_and_tool_exposure_agree_with_the_fact(mock_env, monkeypatch, meta,
                                                            expect_silence):
    """One value drives both, so the UI policy and the tool can never drift apart."""
    monkeypatch.setattr(config, "enable_no_reply_tool", True, raising=False)
    turn = TurnRuntime.for_message(_msg(**meta), channel_post_allowed=False)
    assert turn.silence_capable is expect_silence

    host = _MatHost(_registry())
    _, request_config, available, _ = host._materialize_request_tools(
        host._client, {"model": "m"}, _msg(**meta), tools_disabled=False)
    assert available is expect_silence
    assert bool(request_config.get("_silence_capable_turn")) is expect_silence


def test_the_config_switch_still_closes_the_silence_option(mock_env, monkeypatch):
    """The route says silence is ALLOWED; the flag says the tool that performs it exists."""
    monkeypatch.setattr(config, "enable_no_reply_tool", False, raising=False)
    turn = TurnRuntime.for_message(_msg(gate_required=True, silence_capable=True), channel_post_allowed=False)
    assert turn.silence_capable is False


def test_the_suffix_wording_follows_posture_not_the_gate(mock_env, monkeypatch):
    """The paragraphs describe why the message is in front of the model, so posture picks them.
    A thread message the gate judged and an ungated thread continuation raise the same question —
    is this still mine? — and now read the same instruction."""
    from prompts import (CHANNEL_ACTIVITY_NO_REPLY_SUFFIX,
                         THREAD_ACTIVITY_NO_REPLY_SUFFIX)
    monkeypatch.setattr(config, "enable_no_reply_tool", True, raising=False)
    host = _MatHost(_registry())
    _, _, _, gated_thread = host._materialize_request_tools(
        host._client, {"model": "m"},
        _msg(gate_required=True, silence_capable=True, routing_posture="thread_activity"),
        tools_disabled=False)
    _, _, _, continuation = host._materialize_request_tools(
        host._client, {"model": "m"},
        _msg(gate_required=False, silence_capable=True, routing_posture="thread_activity"),
        tools_disabled=False)
    _, _, _, ambient = host._materialize_request_tools(
        host._client, {"model": "m"},
        _msg(gate_required=True, silence_capable=True, routing_posture="channel_activity"),
        tools_disabled=False)
    assert gated_thread == THREAD_ACTIVITY_NO_REPLY_SUFFIX
    assert continuation == THREAD_ACTIVITY_NO_REPLY_SUFFIX
    assert ambient == CHANNEL_ACTIVITY_NO_REPLY_SUFFIX
