"""Channel-teammate onboarding: a first-time user's @mention in a CHANNEL is answered with
default settings — NO public "I've DM'd you" chrome, NO blocking modal gate — and the settings
button is instead DM'd silently, exactly once (durable). DMs keep the full onboarding flow.

Three layers:
  A. DB (claim/clear): the durable "nudged once" test-and-set survives restarts.
  B. Mention-path routing (_handle_slack_message): channel new-user → helper + dispatch;
     DM new-user → the legacy onboarding block, never the channel shortcut.
  C. The helper itself (_welcome_new_channel_user_via_dm): claim-gated, DMs the user (not the
     channel), carries NO original-message context, and releases the claim if the send fails.

Real decision code; stubbed I/O (Part A uses a real temp DB so the claim is genuinely exercised).
"""
from __future__ import annotations

import sqlite3
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from slack_sdk.errors import SlackApiError

from base_client import Message
from config import config
from database import DatabaseManager
from slack_client.event_handlers.message_events import SlackMessageEventsMixin


# =========================================================== Part A: durable claim/clear (real DB)

@pytest.fixture
def temp_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("os.makedirs"):
            db = DatabaseManager("test")
            db.db_path = f"{tmpdir}/test.db"
            if getattr(db, "conn", None):
                db.conn.close()
            db.conn = sqlite3.connect(db.db_path, check_same_thread=False, isolation_level=None)
            db.conn.row_factory = sqlite3.Row
            db.conn.execute("PRAGMA journal_mode=WAL")
            db.init_schema()
            yield db
            if getattr(db, "conn", None):
                db.conn.close()


async def test_claim_is_true_exactly_once(temp_db):
    assert await temp_db.claim_channel_onboarding_nudge_async("UHUMAN") is True
    # Every later interaction (incl. a fresh process reading the same durable row) loses.
    assert await temp_db.claim_channel_onboarding_nudge_async("UHUMAN") is False
    assert await temp_db.claim_channel_onboarding_nudge_async("UHUMAN") is False


async def test_claim_is_per_user(temp_db):
    assert await temp_db.claim_channel_onboarding_nudge_async("UA") is True
    assert await temp_db.claim_channel_onboarding_nudge_async("UB") is True


async def test_clear_allows_a_single_retry(temp_db):
    assert await temp_db.claim_channel_onboarding_nudge_async("UHUMAN") is True
    await temp_db.clear_channel_onboarding_nudge_async("UHUMAN")   # send failed → release
    assert await temp_db.claim_channel_onboarding_nudge_async("UHUMAN") is True   # retry wins
    assert await temp_db.claim_channel_onboarding_nudge_async("UHUMAN") is False  # then done


async def test_clear_missing_row_is_noop(temp_db):
    await temp_db.clear_channel_onboarding_nudge_async("UGHOST")  # must not raise


# ==================================================== Part B + C: mention-path routing + the helper

class _Bot(SlackMessageEventsMixin):
    def log_debug(self, *a, **k): pass
    def log_info(self, *a, **k): pass
    def log_warning(self, *a, **k): pass
    def log_error(self, *a, **k): pass


def _ok_resp():
    return SimpleNamespace(get=lambda k, d=None: {"ok": True, "ts": "1.1"}.get(k, d))


def _make_bot(*, settings_completed=False, claim_result=True):
    bot = _Bot.__new__(_Bot)
    bot.bot_user_id = "UBOT"
    bot.bot_id = "BBOT"
    bot.app_id = None
    bot.message_handler = AsyncMock()
    bot.app = MagicMock()
    bot.app.client = MagicMock()
    bot.app.client.chat_postMessage = AsyncMock(return_value=_ok_resp())
    bot.db = MagicMock()
    bot.db.get_user_preferences_async = AsyncMock(
        return_value={"settings_completed": settings_completed})
    bot.db.claim_channel_onboarding_nudge_async = AsyncMock(return_value=claim_result)
    bot.db.clear_channel_onboarding_nudge_async = AsyncMock()
    bot._get_channel_settings = AsyncMock(return_value={})
    bot.classify_sender = lambda e: "human"
    bot._post_settings_button_if_new_thread = AsyncMock()
    bot._maybe_set_assistant_thread_title = AsyncMock()

    async def _e2m(event, client):
        return Message(
            text=event.get("text", ""), user_id=event.get("user"),
            channel_id=event.get("channel"),
            thread_id=event.get("thread_ts") or event.get("ts"),
            attachments=[], metadata={"ts": event.get("ts")})

    bot._event_to_message = _e2m
    return bot


def _evt(**kw):
    e = {"channel": "C1", "ts": "100.1", "user": "UHUMAN", "text": "<@UBOT> hi"}
    e.update(kw)
    return e


@pytest.fixture(autouse=True)
def _listening(monkeypatch):
    # Keep the app_mention path off the "off" rail (off drops @mentions outright). Named for the
    # level it actually sets: `auto_respond` resolves to `on` now, and `judicious` is not a level.
    monkeypatch.setattr(config, "channel_response_mode", "auto_respond", raising=False)
    monkeypatch.setattr(config, "enable_participation_engine", True, raising=False)
    monkeypatch.setattr(config, "enable_channel_listening", False, raising=False)


def _button_value(blocks):
    for b in blocks or []:
        if b.get("type") == "actions":
            for el in b.get("elements", []):
                if el.get("type") == "button":
                    return el.get("value")
    return None


# ---- Part B: routing

async def test_channel_new_user_answers_and_routes_to_silent_dm():
    bot = _make_bot(settings_completed=False)
    bot._welcome_new_channel_user_via_dm = AsyncMock()

    await bot._handle_slack_message(_evt(), bot.app.client, wake_source="app_mention")

    # The mention IS answered (no blocking gate) ...
    bot.message_handler.assert_awaited_once()
    # ... via the silent-DM helper (not the legacy public onboarding) ...
    bot._welcome_new_channel_user_via_dm.assert_awaited_once()
    assert bot._welcome_new_channel_user_via_dm.await_args.args[0] == "UHUMAN"
    # ... and NOTHING was posted into the channel.
    bot.app.client.chat_postMessage.assert_not_called()


async def test_dm_new_user_keeps_legacy_onboarding_not_channel_shortcut():
    bot = _make_bot(settings_completed=False)
    bot._welcome_new_channel_user_via_dm = AsyncMock()

    # A DM (channel id starts with 'D') must NOT take the channel shortcut.
    await bot._handle_slack_message(_evt(channel="D1", ts="200.2"),
                                    bot.app.client, wake_source="dm")

    bot._welcome_new_channel_user_via_dm.assert_not_called()
    bot.db.claim_channel_onboarding_nudge_async.assert_not_called()
    # The legacy DM onboarding blocks until settings are saved → the turn is NOT dispatched.
    bot.message_handler.assert_not_called()


async def test_completed_user_channel_mention_skips_nudge_entirely():
    bot = _make_bot(settings_completed=True)
    bot._welcome_new_channel_user_via_dm = AsyncMock()

    await bot._handle_slack_message(_evt(), bot.app.client, wake_source="app_mention")

    bot._welcome_new_channel_user_via_dm.assert_not_called()
    bot.message_handler.assert_awaited_once()


# ---- Part C: the helper

async def test_helper_dms_the_user_with_contextless_button():
    bot = _make_bot(claim_result=True)
    client = bot.app.client

    await bot._welcome_new_channel_user_via_dm("UHUMAN", client)

    bot.db.claim_channel_onboarding_nudge_async.assert_awaited_once_with("UHUMAN")
    client.chat_postMessage.assert_awaited_once()
    kwargs = client.chat_postMessage.await_args.kwargs
    # DM'd to the USER (routes to their IM), never the channel.
    assert kwargs["channel"] == "UHUMAN"
    # The button carries NO original-message context → no post-configure replay/double-answer.
    assert _button_value(kwargs["blocks"]) == "{}"
    bot.db.clear_channel_onboarding_nudge_async.assert_not_called()


async def test_helper_no_dm_when_claim_lost():
    bot = _make_bot(claim_result=False)   # already nudged / concurrent mention won
    client = bot.app.client

    await bot._welcome_new_channel_user_via_dm("UHUMAN", client)

    client.chat_postMessage.assert_not_called()
    bot.db.clear_channel_onboarding_nudge_async.assert_not_called()


async def test_helper_releases_claim_when_send_fails():
    bot = _make_bot(claim_result=True)
    client = bot.app.client
    client.chat_postMessage = AsyncMock(
        side_effect=SlackApiError("boom", {"ok": False, "error": "cannot_dm_bot"}))

    await bot._welcome_new_channel_user_via_dm("UHUMAN", client)

    # Claim rolled back so a later interaction can retry the one-time nudge.
    bot.db.clear_channel_onboarding_nudge_async.assert_awaited_once_with("UHUMAN")
