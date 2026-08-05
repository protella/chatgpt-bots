"""EDIT §8 — an edit_own_message overwrite must not wake the bot on its own echo.

A chat.update re-fires `message_changed`, and — observed live 2026-07-16 — MAY also emit a
genuine `app_mention` when the edit's text carries the bot's mention. Every guard on that path
already exists; what this file proves is that they hold THROUGH THE REAL EVENT PATH, not on a
listener called directly: the host is the real registration + message-events + utilities mixins,
`_register_handlers` runs against a capturing app, and the captured DECORATED callbacks are
driven with a realistic own `message_changed` (nested edited message with our identity, an
`edited.ts`, the target ts, a thread root, and `<@BOT_USER_ID>` in the text) AND its own
`app_mention` twin — parametrized so each of the three identity fields (bot user id, bot_id,
app_id) carries the self-check alone.

Asserted, per §8's own list: the message handler never runs; `_run_edit_triggered_reply` is
never scheduled; nothing posts, updates or reacts; the admission watermark never moves; the
activity index writes nothing; no ambient offer is made; the actor tail is unchanged; and
ingress returns to zero.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import config
from slack_client import actor_tail, admission_watermark
from slack_client.event_handlers.message_events import SlackMessageEventsMixin
from slack_client.event_handlers.registration import SlackRegistrationMixin, ingress
from slack_client.formatting.text import SlackFormattingMixin
from slack_client.utilities import SlackUtilitiesMixin

BOT_USER = "UBOTSELF"
BOT_ID = "BBOTSELF"
APP_ID = "ABOTSELF"
TEAM = "T1"
ROOT = "1700000000.000100"
TARGET = "1700000060.000200"
EDITED_AT = "1700000100.000000"


class _FakeApp:
    """Bolt's registration surface, capturing the ACTUAL decorated callbacks."""

    def __init__(self):
        self.events = {}
        self.actions = {}
        self.client = SimpleNamespace(
            chat_postMessage=AsyncMock(),
            chat_update=AsyncMock(),
            reactions_add=AsyncMock(),
            conversations_replies=AsyncMock(return_value={"messages": []}),
        )

    def event(self, name):
        def deco(fn):
            self.events[name] = fn
            return fn
        return deco

    def action(self, action_id):
        def deco(fn):
            self.actions[str(action_id)] = fn
            return fn
        return deco


class _Host(SlackRegistrationMixin, SlackMessageEventsMixin, SlackFormattingMixin,
            SlackUtilitiesMixin):
    """The REAL mixins — identity checks, tail feed, ambient ingest, edit gating — with only
    the settings-handler registration (outside §8's path) stubbed out. The formatting mixin
    rides along because `_event_to_message` resolves mentions through it."""

    def log_debug(self, *args, **kwargs):
        pass

    log_info = log_warning = log_error = log_debug

    def _register_settings_handlers(self):
        pass


@pytest.fixture
def wired(monkeypatch):
    """A registered host, plus the sinks every §8 assertion reads."""
    monkeypatch.setattr(config, "enable_edit_triggered_replies", True, raising=False)
    monkeypatch.setattr(config, "enable_channel_listening", True, raising=False)
    monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
    monkeypatch.setattr(config, "edit_reply_window_minutes", 0, raising=False)  # no age gate
    ingress.reset()

    bot = _Host.__new__(_Host)
    bot.app = _FakeApp()
    bot.bot_user_id = BOT_USER
    bot.bot_id = BOT_ID
    bot.app_id = APP_ID
    bot.self_team_id = TEAM
    bot.message_handler = AsyncMock()
    bot.user_cache = {BOT_USER: {"username": "chatgpt", "real_name": "ChatGPT",
                                 "email": None, "timezone": "UTC", "tz_label": "UTC",
                                 "tz_offset": 0}}
    bot.db = MagicMock()
    bot.db.get_user_info_async = AsyncMock(return_value=None)
    bot.db.get_or_create_user_async = AsyncMock(return_value={})
    bot.db.get_user_timezone_async = AsyncMock(return_value=None)
    bot.db.get_channel_settings_async = AsyncMock(return_value=None)
    bot.db.get_user_preferences_async = AsyncMock(return_value={"settings_completed": True})
    bot.db.delete_ambient_artifacts_by_source = AsyncMock()

    offers = []
    bot.processor = SimpleNamespace(
        ambient_service=SimpleNamespace(
            offer_event=lambda event, facade: offers.append(event)),
        participation_engine=None,
        thread_manager=None,
    )
    scheduled = []
    bot._run_edit_triggered_reply = MagicMock(return_value="CORO")
    bot._schedule_edit_reply = lambda coro: scheduled.append(coro)

    bot._register_handlers()
    try:
        yield bot, offers, scheduled
    finally:
        ingress.reset()


# The three identity spellings, each carried ALONE so each self-check does the whole job.
IDENTITIES = [
    pytest.param({"user": BOT_USER}, id="bot-user-ID-only"),
    pytest.param({"bot_id": BOT_ID, "subtype": "bot_message"}, id="bot-ID-only"),
    pytest.param({"app_id": APP_ID}, id="app-ID-only"),
]


def _own_message_changed(channel: str, identity: dict) -> dict:
    """The echo Slack fires for our own chat.update: nested edited message with our identity,
    `edited.ts`, the target's own ts, its thread root, and the bot's mention in the text."""
    inner = {"type": "message", "ts": TARGET, "thread_ts": ROOT, "team": TEAM,
             "text": f"<@{BOT_USER}> Correction: the cap is 48000.",
             "edited": {"ts": EDITED_AT}, **identity}
    previous = {"type": "message", "ts": TARGET, "thread_ts": ROOT, "team": TEAM,
                "text": "the cap is 36000", **identity}
    return {"type": "message", "subtype": "message_changed", "channel": channel,
            "channel_type": "channel", "hidden": True, "team": TEAM,
            "ts": "1700000100.000001", "event_ts": "1700000100.000001",
            "message": inner, "previous_message": previous}


def _own_app_mention(channel: str, identity: dict) -> dict:
    """The twin: editing our mention into a message makes Slack deliver a REAL app_mention for
    the same ts (registration.py's F52 note; observed in production)."""
    event = {"type": "app_mention", "channel": channel, "ts": TARGET, "thread_ts": ROOT,
             "team": TEAM, "event_ts": "1700000100.000002",
             "text": f"<@{BOT_USER}> Correction: the cap is 48000.", **identity}
    event.pop("subtype", None)  # app_mention events carry no message subtype
    return event


def _assert_nothing_happened(bot, offers, scheduled, channel):
    # The message handler never runs, and no edit-triggered reply is ever scheduled.
    bot.message_handler.assert_not_called()
    bot._run_edit_triggered_reply.assert_not_called()
    assert scheduled == []
    # No post, no update, no reaction.
    bot.app.client.chat_postMessage.assert_not_awaited()
    bot.app.client.chat_update.assert_not_awaited()
    bot.app.client.reactions_add.assert_not_awaited()
    # The watermark never moved: our own posts must not advance H (a turn must never be able
    # to wait on its own reply). And the channel was not DEGRADED either — the index path
    # declined the self event cleanly rather than failing an observation about it.
    assert admission_watermark.current(channel) is None
    assert not admission_watermark.is_degraded(channel)
    # The activity index wrote nothing (a self sender resolves to no observation at all).
    assert not bot.db.record_thread_activity_async.called
    assert not bot.db.seed_channel_coverage_async.called
    # No ambient offer, and the actor tail is untouched.
    assert offers == []
    assert actor_tail.generation(channel) == 0
    assert not getattr(bot, "_actor_tail_seen_map", None)
    # Ingress returned to zero — the callbacks were counted and released.
    assert ingress.in_flight == 0


@pytest.mark.parametrize("identity", IDENTITIES)
async def test_own_edit_echo_wakes_nothing_through_the_real_registration(wired, identity):
    bot, offers, scheduled = wired
    channel = f"C8NOSELFWAKE{abs(hash(tuple(sorted(identity)))) % 1000}"
    handle_message = bot.app.events["message"]
    await handle_message(_own_message_changed(channel, identity), MagicMock(),
                         bot.app.client)
    _assert_nothing_happened(bot, offers, scheduled, channel)


@pytest.mark.parametrize("identity", IDENTITIES)
async def test_own_app_mention_twin_wakes_nothing_either(wired, identity):
    bot, offers, scheduled = wired
    channel = f"C8NOSELFTWIN{abs(hash(tuple(sorted(identity)))) % 1000}"
    handle_app_mention = bot.app.events["app_mention"]
    await handle_app_mention(_own_app_mention(channel, identity), MagicMock(),
                             bot.app.client)
    _assert_nothing_happened(bot, offers, scheduled, channel)


@pytest.mark.parametrize("identity", IDENTITIES)
async def test_the_full_echo_pair_back_to_back_still_wakes_nothing(wired, identity):
    """Production order for a mention-bearing edit: the message_changed echo AND the genuine
    app_mention for the same ts, one after the other — the §8 shape in full."""
    bot, offers, scheduled = wired
    channel = f"C8NOSELFPAIR{abs(hash(tuple(sorted(identity)))) % 1000}"
    await bot.app.events["message"](_own_message_changed(channel, identity), MagicMock(),
                                    bot.app.client)
    await bot.app.events["app_mention"](_own_app_mention(channel, identity), MagicMock(),
                                        bot.app.client)
    _assert_nothing_happened(bot, offers, scheduled, channel)
