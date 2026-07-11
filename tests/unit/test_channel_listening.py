"""Phase 5/6 + 2.5 — channel-listening decision logic, reply placement, and bot-in-roster.

These exercise the real decision code in SlackMessageEventsMixin with stubbed I/O, so they
assert the SAFE-by-default behavior the keystone is supposed to ship.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from base_client import Message
from config import config
from message_processor.utilities import build_roster_text
from slack_client.event_handlers.message_events import SlackMessageEventsMixin
from slack_client.utilities import SlackUtilitiesMixin


class _Bot(SlackMessageEventsMixin, SlackUtilitiesMixin):
    """Minimal harness exposing the real channel-decision logic with stubbed logging/I/O."""

    def log_debug(self, *a, **k):
        pass

    def log_info(self, *a, **k):
        pass

    def log_warning(self, *a, **k):
        pass

    def log_error(self, *a, **k):
        pass


def _make_bot():
    bot = _Bot.__new__(_Bot)
    bot.bot_user_id = "UBOT"
    bot.bot_id = "BBOT"
    bot.app_id = None
    bot.message_handler = AsyncMock()
    bot.app = MagicMock()
    bot.app.client = MagicMock()

    async def _fake_event_to_message(event, client):
        # Mirror the real _event_to_message file plumbing (both the @-mention and
        # channel-listening paths share it), so gate tests can assert the dispatched
        # Message carries files from a file_share event.
        attachments = []
        for f in event.get("files", []) or []:
            mimetype = f.get("mimetype", "")
            attachments.append({
                "type": "image" if mimetype.startswith("image/") else "file",
                "url": f.get("url_private"),
                "id": f.get("id"),
                "name": f.get("name"),
                "mimetype": mimetype,
            })
        return Message(
            text=event.get("text", ""),
            user_id=event.get("user"),
            channel_id=event.get("channel"),
            thread_id=event.get("thread_ts") or event.get("ts"),
            attachments=attachments,
            metadata={"ts": event.get("ts")},
        )

    bot._event_to_message = _fake_event_to_message
    return bot


def _evt(**kw):
    e = {"channel": "C1", "ts": "100.1", "user": "UHUMAN", "text": "hello there", "channel_type": "channel"}
    e.update(kw)
    return e


@pytest.fixture
def tag_only(monkeypatch):
    monkeypatch.setattr(config, "channel_response_mode", "tag_only", raising=False)
    monkeypatch.setattr(config, "bot_name_aliases", ["ChatGPT", "ChatGPT-Dev"], raising=False)


@pytest.mark.asyncio
async def test_own_message_by_user_id_short_circuits(tag_only):
    bot = _make_bot()
    await bot._handle_channel_message(_evt(user="UBOT", text="ChatGPT hi"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_own_message_by_bot_id_short_circuits(tag_only):
    bot = _make_bot()
    await bot._handle_channel_message(_evt(bot_id="BBOT", user=None, text="ChatGPT hi"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_subtype_skipped(tag_only):
    bot = _make_bot()
    await bot._handle_channel_message(_evt(subtype="channel_join", text="ChatGPT hi"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_message_changed_subtype_excluded_from_gate(tag_only):
    # F14: non-content subtypes (edits/deletes) still never drive a response.
    bot = _make_bot()
    await bot._handle_channel_message(
        _evt(subtype="message_changed", text="ChatGPT hi"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_file_share_reaches_gate_and_plumbs_files(tag_only):
    # F14: an image/file upload arrives as subtype 'file_share' — it must proceed
    # through the response gate (was dropped before) AND carry its files onto the
    # dispatched Message so intent classification can route the vision/document flow.
    bot = _make_bot()
    file_meta = {
        "id": "F123", "name": "poster.png", "mimetype": "image/png",
        "url_private": "https://files.slack.com/poster.png",
    }
    await bot._handle_channel_message(
        _evt(subtype="file_share", text="ChatGPT good marketing material?",
             files=[file_meta]),
        bot.app.client)
    bot.message_handler.assert_called_once()
    msg = bot.message_handler.call_args[0][0]
    assert msg.metadata.get("participation_name_hit") is True
    assert msg.attachments and msg.attachments[0]["type"] == "image"
    assert msg.attachments[0]["id"] == "F123"


@pytest.mark.asyncio
async def test_thread_broadcast_subtype_reaches_gate(tag_only):
    # F14: a thread reply also broadcast to channel arrives as 'thread_broadcast' —
    # real content, so it reaches the gate (engine judges the name-hit).
    bot = _make_bot()
    await bot._handle_channel_message(
        _evt(subtype="thread_broadcast", text="ChatGPT what do you think?",
             thread_ts="50.0", ts="60.0"),
        bot.app.client)
    bot.message_handler.assert_called_once()
    msg = bot.message_handler.call_args[0][0]
    assert msg.metadata.get("participation_name_hit") is True


@pytest.mark.asyncio
async def test_real_event_to_message_extracts_files(tag_only):
    # F14: the SHARED _event_to_message (both the @-mention and channel paths call it)
    # extracts event files into attachments — proving the channel path plumbs files
    # identically to the mention path. Image mimetypes classify as 'image', others 'file'.
    bot = _Bot.__new__(_Bot)
    bot.bot_user_id = "UBOT"
    bot.user_cache = {}
    bot.db = MagicMock()
    bot.db.get_user_info_async = AsyncMock(return_value=None)
    bot._clean_mentions = lambda t: t
    bot.get_username = AsyncMock(return_value="Human")
    bot.get_user_timezone = AsyncMock(return_value="UTC")
    bot.classify_sender = lambda e: "human"
    event = _evt(
        subtype="file_share", text="what do we think?",
        files=[
            {"id": "F1", "name": "poster.png", "mimetype": "image/png",
             "url_private": "https://files.slack.com/poster.png"},
            {"id": "F2", "name": "brief.pdf", "mimetype": "application/pdf",
             "url_private": "https://files.slack.com/brief.pdf"},
        ])
    msg = await bot._event_to_message(event, bot.app.client if hasattr(bot, "app") else MagicMock())
    assert [a["type"] for a in msg.attachments] == ["image", "file"]
    assert [a["id"] for a in msg.attachments] == ["F1", "F2"]


@pytest.mark.asyncio
async def test_off_mode_never_responds(monkeypatch):
    monkeypatch.setattr(config, "channel_response_mode", "off", raising=False)
    monkeypatch.setattr(config, "bot_name_aliases", ["ChatGPT"], raising=False)
    bot = _make_bot()
    await bot._handle_channel_message(_evt(text="ChatGPT help me"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_tag_only_unaddressed_ignored(tag_only):
    bot = _make_bot()
    await bot._handle_channel_message(_evt(text="lunch anyone?"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_tag_only_name_hit_routes_to_engine(tag_only):
    # Revised contract: a name-in-text hit is a SIGNAL, not a verdict — the engine
    # decides addressed vs merely-discussed vs same-named public product.
    bot = _make_bot()
    await bot._handle_channel_message(_evt(text="ChatGPT, what's the weather?"), bot.app.client)
    bot.message_handler.assert_called_once()
    msg = bot.message_handler.call_args[0][0]
    assert msg.metadata.get("channel_listen") is True
    assert msg.metadata.get("participation_check") is True
    assert msg.metadata.get("participation_name_hit") is True


@pytest.mark.asyncio
async def test_tag_only_name_hit_engine_disabled_falls_back_deterministic(tag_only, monkeypatch):
    # With the engine off, the legacy deterministic name wake keeps working.
    monkeypatch.setattr(config, "enable_participation_engine", False, raising=False)
    bot = _make_bot()
    await bot._handle_channel_message(_evt(text="ChatGPT, what's the weather?"), bot.app.client)
    bot.message_handler.assert_called_once()
    msg = bot.message_handler.call_args[0][0]
    assert msg.metadata.get("participation_check") is not True


@pytest.mark.asyncio
async def test_explicit_mention_is_deduped(tag_only):
    # An <@UBOT> mention is already delivered via the app_mention event; channel path must skip.
    bot = _make_bot()
    await bot._handle_channel_message(_evt(text="<@UBOT> hello"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_auto_respond_sets_participation_check_for_unaddressed(monkeypatch):
    monkeypatch.setattr(config, "channel_response_mode", "auto_respond", raising=False)
    monkeypatch.setattr(config, "bot_name_aliases", ["ChatGPT"], raising=False)
    monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
    bot = _make_bot()
    await bot._handle_channel_message(_evt(text="anyone know the q3 numbers?"), bot.app.client)
    bot.message_handler.assert_called_once()
    msg = bot.message_handler.call_args[0][0]
    assert msg.metadata.get("participation_check") is True
    assert msg.metadata.get("participation_level") == "judicious"  # auto_respond ≡ judicious


@pytest.mark.asyncio
async def test_engine_disabled_makes_auto_respond_mentions_only(monkeypatch):
    monkeypatch.setattr(config, "channel_response_mode", "auto_respond", raising=False)
    monkeypatch.setattr(config, "bot_name_aliases", ["ChatGPT"], raising=False)
    monkeypatch.setattr(config, "enable_participation_engine", False, raising=False)
    bot = _make_bot()
    # unaddressed → ignored with zero model cost
    await bot._handle_channel_message(_evt(text="anyone know the q3 numbers?"), bot.app.client)
    bot.message_handler.assert_not_called()
    # addressed by name → still responds directly (pre-F tag_only behavior preserved)
    await bot._handle_channel_message(_evt(ts="101.1", text="ChatGPT what's up?"), bot.app.client)
    bot.message_handler.assert_called_once()


@pytest.mark.asyncio
async def test_snoozed_channel_skips_unprompted_but_not_addressed(monkeypatch):
    from message_processor.participation import snooze_expiry_iso
    monkeypatch.setattr(config, "channel_response_mode", "auto_respond", raising=False)
    monkeypatch.setattr(config, "bot_name_aliases", ["ChatGPT"], raising=False)
    monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
    bot = _make_bot()
    snoozed = {"response_mode": "auto_respond", "snoozed_until": snooze_expiry_iso(hours=1)}

    async def _cs(channel_id):
        return snoozed

    bot._get_channel_settings = _cs
    # unprompted while snoozed → silent, no dispatch
    await bot._handle_channel_message(_evt(text="anyone know the q3 numbers?"), bot.app.client)
    bot.message_handler.assert_not_called()
    # name-bearing while snoozed → still reaches the engine (told to be quiet ≠ deaf),
    # carrying both signals so the model can require a genuine summons.
    await bot._handle_channel_message(_evt(ts="102.1", text="ChatGPT you there?"), bot.app.client)
    bot.message_handler.assert_called_once()
    msg = bot.message_handler.call_args[0][0]
    assert msg.metadata.get("participation_check") is True
    assert msg.metadata.get("participation_name_hit") is True
    assert msg.metadata.get("participation_snoozed") is True


@pytest.mark.asyncio
async def test_participation_level_off_row_silences_channel(monkeypatch):
    monkeypatch.setattr(config, "channel_response_mode", "auto_respond", raising=False)
    monkeypatch.setattr(config, "bot_name_aliases", ["ChatGPT"], raising=False)
    bot = _make_bot()

    async def _cs(channel_id):
        return {"response_mode": "auto_respond", "participation_level": "off"}

    bot._get_channel_settings = _cs
    await bot._handle_channel_message(_evt(text="ChatGPT help"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_thread_reply_one_on_one_responds(tag_only):
    bot = _make_bot()
    bot._thread_participation = AsyncMock(return_value=(True, 1, 0))
    await bot._handle_channel_message(_evt(text="and what about friday?", thread_ts="50.0", ts="60.0"), bot.app.client)
    bot.message_handler.assert_called_once()


@pytest.mark.asyncio
async def test_thread_reply_multiparty_unaddressed_ignored(tag_only):
    bot = _make_bot()
    bot._thread_participation = AsyncMock(return_value=(True, 3, 0))
    await bot._handle_channel_message(_evt(text="sounds good to me", thread_ts="50.0", ts="60.0"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_thread_with_other_agent_is_not_a_continuation(tag_only):
    # A second bot/agent in the thread means untagged replies may be for IT —
    # no deterministic continuation (this is the Claude-in-the-test-channel bug:
    # one human + two agents looked "1:1" when only humans were counted).
    bot = _make_bot()
    bot._thread_participation = AsyncMock(return_value=(True, 1, 1))
    await bot._handle_channel_message(_evt(text="sounds good", thread_ts="50.0", ts="60.0"), bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_bot_sender_never_direct_continuation(tag_only):
    # Another bot replying in our 1:1 thread must not get a judgment-free response.
    bot = _make_bot()
    bot._thread_participation = AsyncMock(return_value=(True, 1, 0))
    evt = _evt(user="UCLAUDE", bot_id="BCLAUDE", text="I agree with the plan.",
               thread_ts="50.0", ts="60.0")
    await bot._handle_channel_message(evt, bot.app.client)
    bot.message_handler.assert_not_called()
    bot._thread_participation.assert_not_called()  # not even consulted for bot senders


@pytest.mark.asyncio
async def test_bot_sender_name_hit_routes_to_engine_with_signal(tag_only):
    # Bot-to-bot is allowed — but only via the engine's judgment, with the
    # sender-is-bot signal attached.
    bot = _make_bot()
    evt = _evt(user="UCLAUDE", bot_id="BCLAUDE", text="ChatGPT, what does the data say?")
    await bot._handle_channel_message(evt, bot.app.client)
    bot.message_handler.assert_called_once()
    msg = bot.message_handler.call_args[0][0]
    assert msg.metadata.get("participation_check") is True
    assert msg.metadata.get("participation_sender_bot") is True


@pytest.mark.asyncio
async def test_bot_sender_name_hit_engine_disabled_stays_silent(tag_only, monkeypatch):
    # With the engine off there is no judgment available, so a bot naming us
    # must not trigger the legacy deterministic wake (loop seed).
    monkeypatch.setattr(config, "enable_participation_engine", False, raising=False)
    bot = _make_bot()
    evt = _evt(user="UCLAUDE", bot_id="BCLAUDE", text="ChatGPT, ping")
    await bot._handle_channel_message(evt, bot.app.client)
    bot.message_handler.assert_not_called()


@pytest.mark.asyncio
async def test_reply_placed_in_thread(tag_only):
    bot = _make_bot()
    await bot._handle_channel_message(_evt(text="ChatGPT ping", ts="77.7"), bot.app.client)
    msg = bot.message_handler.call_args[0][0]
    assert msg.thread_id == "77.7"  # top-level wake → reply in a thread rooted at the message


def test_text_mentions_bot_name_whole_word(tag_only):
    bot = _make_bot()
    assert bot._text_mentions_bot_name("hey ChatGPT can you help")
    assert bot._text_mentions_bot_name("CHATGPT-DEV go")
    assert not bot._text_mentions_bot_name("the chatgptithon event")  # whole-word match only
    assert not bot._text_mentions_bot_name("no name here")


@pytest.mark.asyncio
async def test_thread_participation_counts_humans_and_bot(tag_only):
    bot = _make_bot()
    bot.app.client.conversations_replies = AsyncMock(return_value={"messages": [
        {"user": "UBOT", "bot_id": "BBOT"},  # self
        {"user": "UHUMAN1"},
        {"user": "UHUMAN2"},
        {"user": "UHUMAN1"},  # dup human
        {"user": "UCLAUDE", "bot_id": "BCLAUDE"},  # another agent
        {"user": "UCLAUDE", "bot_id": "BCLAUDE"},  # dup agent
    ]})
    bot_present, humans, other_bots = await bot._thread_participation("C1", "50.0")
    assert bot_present is True
    assert humans == 2
    assert other_bots == 1


@pytest.mark.asyncio
async def test_thread_participation_handles_api_error(tag_only):
    bot = _make_bot()
    bot.app.client.conversations_replies = AsyncMock(side_effect=RuntimeError("boom"))
    assert await bot._thread_participation("C1", "50.0") == (False, 0, 0)


def test_default_config_is_safe(monkeypatch):
    # OUT OF THE BOX: the bot must not auto-listen, and the default channel mode is
    # tag_only. Build a fresh config with the env keys absent — the module singleton
    # may reflect a real .env (e.g. the dev box enables listening for live testing).
    monkeypatch.delenv("ENABLE_CHANNEL_LISTENING", raising=False)
    monkeypatch.delenv("CHANNEL_RESPONSE_MODE", raising=False)
    from config import BotConfig
    fresh = BotConfig()
    assert fresh.enable_channel_listening is False
    assert fresh.channel_response_mode == "tag_only"


def test_bot_with_real_user_id_lands_in_roster():
    # Phase 2.5: another bot that posts with a real user_id can be tagged via the roster.
    txt = build_roster_text({"U123": "Peter", "U999": "Claude"}, user_cache={}, bot_user_id="UBOT")
    assert "<@U999>" in txt
    # The "bot"/"unknown" placeholder ids are excluded (cannot <@>-tag a bot_id).
    txt2 = build_roster_text({"bot": "Bot", "U123": "Peter"}, bot_user_id="UBOT")
    assert "<@bot>" not in txt2
    assert "<@U123>" in txt2
