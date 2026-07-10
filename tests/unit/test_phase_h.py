"""Phase H — response feedback: native feedback buttons + passive reaction ingestion.

Covers: the response_feedback sink (upsert semantics, sources, ratio helper, async
parity), emoji→signal mapping, passive reaction_added ingestion (own-message gate,
no-LLM/no-reply contract), the block_actions handler, the DM feedback strip vs the
channel footer in maybe_post_response_footer, and the history-rebuild skip for
UI-helper messages. All stubbed I/O — no live bot.
"""
from __future__ import annotations

import sqlite3
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from config import config
from database import DatabaseManager
from slack_client.event_handlers import feedback
from slack_client.messaging import SlackMessagingMixin, _is_ui_helper_message


# --------------------------------------------------------------------------- fixtures

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


class _Host:
    """Minimal stand-in for the SlackBot pieces the feedback handlers touch."""

    def __init__(self, db=None, bot_user_id="UBOT"):
        self.db = db or SimpleNamespace(record_response_feedback_async=AsyncMock())
        self.bot_user_id = bot_user_id
        self.app = SimpleNamespace(client=SimpleNamespace(chat_postEphemeral=AsyncMock()))
        self.debug_lines = []

    def log_debug(self, msg):
        self.debug_lines.append(msg)


def _button_body(value="good", channel="D123", user="U1", ts="111.222", thread_ts=None):
    msg = {"ts": ts}
    if thread_ts:
        msg["thread_ts"] = thread_ts
    return {
        "actions": [{"action_id": feedback.FEEDBACK_ACTION_ID, "value": value}],
        "channel": {"id": channel},
        "user": {"id": user},
        "message": msg,
        "container": {"message_ts": ts},
    }


def _reaction_event(reaction="+1", item_user="UBOT", user="U1",
                    channel="C1", ts="123.456", item_type="message"):
    return {
        "type": "reaction_added",
        "reaction": reaction,
        "user": user,
        "item_user": item_user,
        "item": {"type": item_type, "channel": channel, "ts": ts},
    }


# --------------------------------------------------------------------------- DB sink

class TestFeedbackSink:
    def test_record_and_ratio(self, temp_db):
        temp_db.record_response_feedback("C1", "100.0", "100.1", "U1", 1, "button")
        temp_db.record_response_feedback("C1", "100.0", "100.2", "U2", -1, "reaction")
        pos, neg, ratio = temp_db.get_channel_feedback_ratio("C1")
        assert (pos, neg) == (1, 1)
        assert ratio == 0.5

    def test_upsert_same_message_user_source_updates(self, temp_db):
        temp_db.record_response_feedback("C1", None, "100.1", "U1", 1, "button")
        temp_db.record_response_feedback("C1", None, "100.1", "U1", -1, "button")
        pos, neg, ratio = temp_db.get_channel_feedback_ratio("C1")
        assert (pos, neg) == (0, 1)  # one row, thumb flipped — not two votes

    def test_sources_are_independent_rows(self, temp_db):
        temp_db.record_response_feedback("C1", None, "100.1", "U1", 1, "button")
        temp_db.record_response_feedback("C1", None, "100.1", "U1", 1, "reaction")
        pos, neg, _ = temp_db.get_channel_feedback_ratio("C1")
        assert pos == 2

    def test_ratio_none_when_no_feedback(self, temp_db):
        assert temp_db.get_channel_feedback_ratio("C_EMPTY") == (0, 0, None)

    def test_ratio_scoped_to_channel(self, temp_db):
        temp_db.record_response_feedback("C1", None, "1.1", "U1", 1, "button")
        temp_db.record_response_feedback("C2", None, "2.1", "U1", -1, "button")
        assert temp_db.get_channel_feedback_ratio("C1")[:2] == (1, 0)
        assert temp_db.get_channel_feedback_ratio("C2")[:2] == (0, 1)

    def test_ratio_days_window_excludes_old_rows(self, temp_db):
        temp_db.record_response_feedback("C1", None, "1.1", "U1", 1, "button")
        temp_db.conn.execute(
            "UPDATE response_feedback SET created_ts = datetime('now', '-60 days')"
        )
        assert temp_db.get_channel_feedback_ratio("C1", days=30) == (0, 0, None)
        assert temp_db.get_channel_feedback_ratio("C1", days=90)[:2] == (1, 0)

    def test_signal_constraint(self, temp_db):
        with pytest.raises(sqlite3.IntegrityError):
            temp_db.conn.execute(
                "INSERT INTO response_feedback (channel_id, message_ts, user_id, signal, source) "
                "VALUES ('C1', '1.1', 'U1', 5, 'button')"
            )

    def test_delete_row(self, temp_db):
        temp_db.record_response_feedback("C1", None, "1.1", "U1", 1, "reaction")
        temp_db.delete_response_feedback("1.1", "U1", "reaction")
        assert temp_db.get_channel_feedback_ratio("C1") == (0, 0, None)

    @pytest.mark.asyncio
    async def test_async_record_and_ratio(self, temp_db):
        await temp_db.record_response_feedback_async("C1", "1.0", "1.1", "U1", 1, "button")
        await temp_db.record_response_feedback_async("C1", "1.0", "1.1", "U1", -1, "button")
        pos, neg, ratio = await temp_db.get_channel_feedback_ratio_async("C1")
        assert (pos, neg, ratio) == (0, 1, 0.0)


# --------------------------------------------------------------------------- emoji map

class TestReactionSignal:
    @pytest.mark.parametrize("name,expected", [
        ("+1", 1), ("thumbsup", 1), ("thumbsup_all", 1),
        ("+1::skin-tone-4", 1), ("thumbsdown::skin-tone-2", -1),
        ("-1", -1), ("thumbsdown", -1),
        ("eyes", None), ("joy", None), ("", None),
    ])
    def test_mapping(self, name, expected):
        assert feedback.reaction_signal(name) == expected


# --------------------------------------------------------------------------- ingestion

@pytest.mark.asyncio
class TestReactionIngestion:
    async def test_thumb_on_own_message_recorded(self):
        host = _Host()
        await feedback.ingest_reaction(host, _reaction_event())
        host.db.record_response_feedback_async.assert_awaited_once_with(
            channel_id="C1", thread_ts=None, message_ts="123.456",
            user_id="U1", signal=1, source="reaction",
        )

    async def test_thumb_on_someone_elses_message_ignored(self):
        host = _Host()
        await feedback.ingest_reaction(host, _reaction_event(item_user="U_OTHER"))
        host.db.record_response_feedback_async.assert_not_awaited()

    async def test_non_thumb_reaction_ignored(self):
        host = _Host()
        await feedback.ingest_reaction(host, _reaction_event(reaction="eyes"))
        host.db.record_response_feedback_async.assert_not_awaited()

    async def test_non_message_item_ignored(self):
        host = _Host()
        await feedback.ingest_reaction(host, _reaction_event(item_type="file"))
        host.db.record_response_feedback_async.assert_not_awaited()

    async def test_unresolved_self_identity_ignores(self):
        host = _Host(bot_user_id=None)
        await feedback.ingest_reaction(host, _reaction_event())
        host.db.record_response_feedback_async.assert_not_awaited()

    async def test_malformed_event_never_raises(self):
        host = _Host()
        await feedback.ingest_reaction(host, {})
        await feedback.ingest_reaction(host, {"reaction": "+1"})  # no item/user

    async def test_db_failure_swallowed(self):
        host = _Host()
        host.db.record_response_feedback_async = AsyncMock(side_effect=RuntimeError("db down"))
        await feedback.ingest_reaction(host, _reaction_event())  # must not raise
        assert any("failed" in line for line in host.debug_lines)

    async def test_purely_passive_no_replies(self):
        # The no-LLM / no-reply contract: ingestion touches the DB and nothing else.
        host = _Host()
        await feedback.ingest_reaction(host, _reaction_event())
        host.app.client.chat_postEphemeral.assert_not_awaited()


# --------------------------------------------------------------------------- button handler

@pytest.mark.asyncio
class TestFeedbackAction:
    async def test_good_click_recorded_and_acked(self):
        host = _Host()
        ack = AsyncMock()
        await feedback.handle_feedback_action(host, ack, _button_body("good"))
        ack.assert_awaited_once()
        host.db.record_response_feedback_async.assert_awaited_once_with(
            channel_id="D123", thread_ts="111.222", message_ts="111.222",
            user_id="U1", signal=1, source="button",
        )
        host.app.client.chat_postEphemeral.assert_awaited_once()

    async def test_bad_click_records_negative(self):
        host = _Host()
        await feedback.handle_feedback_action(host, AsyncMock(), _button_body("bad"))
        kwargs = host.db.record_response_feedback_async.await_args.kwargs
        assert kwargs["signal"] == -1

    async def test_thread_ts_preferred_over_message_ts(self):
        host = _Host()
        body = _button_body("good", ts="222.333", thread_ts="111.000")
        await feedback.handle_feedback_action(host, AsyncMock(), body)
        kwargs = host.db.record_response_feedback_async.await_args.kwargs
        assert kwargs["thread_ts"] == "111.000"
        assert kwargs["message_ts"] == "222.333"

    async def test_unknown_value_ignored(self):
        host = _Host()
        await feedback.handle_feedback_action(host, AsyncMock(), _button_body("meh"))
        host.db.record_response_feedback_async.assert_not_awaited()

    async def test_db_failure_swallowed_and_no_raise(self):
        host = _Host()
        host.db.record_response_feedback_async = AsyncMock(side_effect=RuntimeError("boom"))
        await feedback.handle_feedback_action(host, AsyncMock(), _button_body("good"))

    async def test_ephemeral_failure_swallowed(self):
        host = _Host()
        host.app.client.chat_postEphemeral = AsyncMock(side_effect=RuntimeError("no ephemeral here"))
        await feedback.handle_feedback_action(host, AsyncMock(), _button_body("good"))
        host.db.record_response_feedback_async.assert_awaited_once()


# --------------------------------------------------------------------------- blocks + skip

class TestFeedbackBlocks:
    def test_shape_is_native_context_actions(self):
        blocks = feedback.build_feedback_blocks("gpt-5.6-sol")
        assert len(blocks) == 2
        block = blocks[0]
        assert block["type"] == "context_actions"
        el = block["elements"][0]
        assert el["type"] == "feedback_buttons"
        assert el["action_id"] == feedback.FEEDBACK_ACTION_ID
        assert el["positive_button"]["value"] == "good"
        assert el["negative_button"]["value"] == "bad"

    def test_model_button_present_with_user_settings_action(self):
        blocks = feedback.build_feedback_blocks("gpt-5.6-sol")
        btn = blocks[1]["elements"][0]
        assert blocks[1]["type"] == "actions"
        assert btn["type"] == "button"
        assert btn["action_id"] == feedback.USER_SETTINGS_ACTION_ID
        assert btn["text"]["text"] == "⚙️ gpt-5.6-sol"

    def test_model_button_falls_back_to_config_model(self):
        blocks = feedback.build_feedback_blocks(None)
        assert blocks[1]["elements"][0]["text"]["text"] == f"⚙️ {config.gpt_model}"

    def test_rebuild_skips_feedback_strip(self):
        msg = {"blocks": feedback.build_feedback_blocks("gpt-5.6-sol"), "text": "Rate this response"}
        assert _is_ui_helper_message(msg) is True

    def test_rebuild_skips_user_settings_button_alone(self):
        # Robustness: even a strip variant carrying ONLY the model button is chrome.
        msg = {"blocks": [{"type": "actions", "elements": [
            {"type": "button", "action_id": "open_user_settings"}]}]}
        assert _is_ui_helper_message(msg) is True

    def test_rebuild_skips_channel_footer(self):
        msg = {"blocks": [{"type": "actions", "elements": [
            {"type": "button", "action_id": "open_channel_settings"}]}]}
        assert _is_ui_helper_message(msg) is True

    def test_real_messages_not_skipped(self):
        assert _is_ui_helper_message({"text": "plain reply"}) is False
        assert _is_ui_helper_message({"blocks": [
            {"type": "section", "text": {"type": "mrkdwn", "text": "hello"}}]}) is False
        assert _is_ui_helper_message({"blocks": [{"type": "actions", "elements": [
            {"type": "button", "action_id": "some_other_action"}]}]}) is False


# --------------------------------------------------------------------------- strip posting

class _MsgHost(SlackMessagingMixin):
    def __init__(self):
        self.app = SimpleNamespace(client=SimpleNamespace(chat_postMessage=AsyncMock()))
        self.debug_lines = []

    def log_debug(self, msg):
        self.debug_lines.append(msg)


def _msg(channel_id="D123", thread_id="1.0"):
    return SimpleNamespace(channel_id=channel_id, thread_id=thread_id)


def _resp(content="an answer", type_="text", model=None):
    return SimpleNamespace(type=type_, content=content, metadata={"model": model} if model else {})


@pytest.mark.asyncio
class TestFeedbackStripPosting:
    @pytest.fixture(autouse=True)
    def _fresh_offers(self, monkeypatch):
        # Pin the feature ON regardless of the local .env (the dev .env disables it),
        # and reset the once-per-thread offer cache between tests.
        from config import config
        monkeypatch.setattr(config, "enable_feedback_buttons", True)
        feedback._reset_feedback_offers()
        yield
        feedback._reset_feedback_offers()

    async def test_dm_gets_feedback_strip(self):
        host = _MsgHost()
        await host.maybe_post_response_footer(_msg("D123"), _resp())
        kwargs = host.app.client.chat_postMessage.await_args.kwargs
        assert kwargs["channel"] == "D123"
        assert kwargs["blocks"][0]["type"] == "context_actions"
        assert kwargs["blocks"][0]["elements"][0]["action_id"] == feedback.FEEDBACK_ACTION_ID

    async def test_strip_posts_once_per_thread(self):
        # The ENTIRE strip (thumbs + settings button) posts under the FIRST reply
        # of a thread only; later replies get no trailing chrome (reactions stay
        # the always-available feedback signal). A different thread starts fresh.
        host = _MsgHost()
        await host.maybe_post_response_footer(_msg("D123"), _resp(model="gpt-5.6-sol"))
        assert host.app.client.chat_postMessage.await_count == 1
        first = host.app.client.chat_postMessage.await_args.kwargs["blocks"]
        assert any(b["type"] == "context_actions" for b in first)
        assert any(
            el.get("action_id") == feedback.USER_SETTINGS_ACTION_ID
            for b in first for el in (b.get("elements") or [])
        )

        # Same thread again: nothing posted.
        await host.maybe_post_response_footer(_msg("D123"), _resp(model="gpt-5.6-sol"))
        assert host.app.client.chat_postMessage.await_count == 1

        # New thread: strip offered again.
        await host.maybe_post_response_footer(_msg("D123", thread_id="2.0"), _resp(model="gpt-5.6-sol"))
        assert host.app.client.chat_postMessage.await_count == 2

    async def test_dm_strip_carries_responding_model_button(self):
        host = _MsgHost()
        await host.maybe_post_response_footer(_msg("D123"), _resp(model="gpt-5.6-terra"))
        blocks = host.app.client.chat_postMessage.await_args.kwargs["blocks"]
        btn = blocks[1]["elements"][0]
        assert btn["action_id"] == feedback.USER_SETTINGS_ACTION_ID
        assert btn["text"]["text"] == "⚙️ gpt-5.6-terra"

    async def test_dm_flag_off_posts_nothing(self, monkeypatch):
        monkeypatch.setenv("ENABLE_FEEDBACK_BUTTONS", "false")
        monkeypatch.setattr(feedback, "config", SimpleNamespace(enable_feedback_buttons=None))
        host = _MsgHost()
        await host.maybe_post_response_footer(_msg("D123"), _resp())
        host.app.client.chat_postMessage.assert_not_awaited()

    async def test_channel_keeps_configure_footer(self):
        host = _MsgHost()
        await host.maybe_post_response_footer(_msg("C777"), _resp(model="gpt-5.5"))
        kwargs = host.app.client.chat_postMessage.await_args.kwargs
        assert kwargs["blocks"][0]["type"] == "actions"
        assert kwargs["blocks"][0]["elements"][0]["action_id"] == "open_channel_settings"

    async def test_reaction_only_turn_posts_nothing(self):
        host = _MsgHost()
        await host.maybe_post_response_footer(_msg("D123"), _resp(content=""))
        host.app.client.chat_postMessage.assert_not_awaited()

    async def test_non_text_response_posts_nothing(self):
        host = _MsgHost()
        await host.maybe_post_response_footer(_msg("D123"), _resp(type_="image"))
        host.app.client.chat_postMessage.assert_not_awaited()

    async def test_post_failure_never_raises(self):
        host = _MsgHost()
        host.app.client.chat_postMessage = AsyncMock(side_effect=RuntimeError("slack down"))
        await host.maybe_post_response_footer(_msg("D123"), _resp())


# --------------------------------------------------------------------------- flag default

class TestFeedbackFlag:
    def test_default_on_without_config_or_env(self, monkeypatch):
        monkeypatch.delenv("ENABLE_FEEDBACK_BUTTONS", raising=False)
        monkeypatch.setattr(feedback, "config", SimpleNamespace(enable_feedback_buttons=None))
        assert feedback.feedback_enabled() is True

    def test_config_attr_wins(self, monkeypatch):
        monkeypatch.setenv("ENABLE_FEEDBACK_BUTTONS", "true")
        monkeypatch.setattr(feedback, "config", SimpleNamespace(enable_feedback_buttons=False))
        assert feedback.feedback_enabled() is False

    def test_env_fallback(self, monkeypatch):
        monkeypatch.setenv("ENABLE_FEEDBACK_BUTTONS", "false")
        monkeypatch.setattr(feedback, "config", SimpleNamespace(enable_feedback_buttons=None))
        assert feedback.feedback_enabled() is False


# --------------------------------------------------------------------------- open_user_settings handler

class _FakeApp:
    """Captures handlers registered via @app.action/@app.command/@app.view/@app.shortcut."""
    def __init__(self):
        self.actions = {}

    def action(self, action_id):
        def deco(fn):
            self.actions[action_id] = fn
            return fn
        return deco

    def command(self, *_a, **_k):
        return lambda fn: fn

    def view(self, *_a, **_k):
        return lambda fn: fn

    def shortcut(self, *_a, **_k):
        return lambda fn: fn

    def event(self, *_a, **_k):
        return lambda fn: fn


def _settings_host():
    from slack_client.event_handlers.settings import SlackSettingsHandlersMixin

    host = SlackSettingsHandlersMixin.__new__(SlackSettingsHandlersMixin)
    host.app = _FakeApp()
    host.db = SimpleNamespace(
        get_user_preferences_async=AsyncMock(return_value={"model": "gpt-5.6-sol"}),
        get_or_create_user_async=AsyncMock(return_value={"email": "u@example.com"}),
        create_default_user_preferences_async=AsyncMock(return_value={"model": "gpt-5.6-sol"}),
        get_channel_settings_async=AsyncMock(return_value=None),
    )
    host.settings_modal = SimpleNamespace(
        build_settings_modal=AsyncMock(return_value={"type": "modal"}),
        build_channel_settings_modal=lambda *a, **k: {"type": "modal"},
    )
    host.log_info = lambda *a, **k: None
    host.log_error = lambda *a, **k: None
    host.log_debug = lambda *a, **k: None
    host._register_settings_handlers()
    return host


def _action_body(action_id="open_user_settings"):
    return {
        "trigger_id": "trig-1",
        "user": {"id": "U1"},
        "container": {"channel_id": "D123"},
        "actions": [{"action_id": action_id}],
    }


@pytest.mark.asyncio
class TestOpenUserSettingsHandler:
    async def test_opens_user_settings_modal(self):
        host = _settings_host()
        handler = host.app.actions["open_user_settings"]
        client = SimpleNamespace(views_open=AsyncMock(return_value={"ok": True}),
                                 chat_postEphemeral=AsyncMock())
        await handler(ack=AsyncMock(), body=_action_body(), client=client)
        client.views_open.assert_awaited_once()
        kwargs = host.settings_modal.build_settings_modal.await_args.kwargs
        assert kwargs["user_id"] == "U1"
        assert kwargs["thread_id"] is None and kwargs["in_thread"] is False
        client.chat_postEphemeral.assert_not_awaited()

    async def test_failure_sends_ephemeral(self):
        host = _settings_host()
        handler = host.app.actions["open_user_settings"]
        client = SimpleNamespace(views_open=AsyncMock(side_effect=RuntimeError("boom")),
                                 chat_postEphemeral=AsyncMock())
        await handler(ack=AsyncMock(), body=_action_body(), client=client)
        eph = client.chat_postEphemeral.await_args.kwargs
        assert eph["channel"] == "D123" and eph["user"] == "U1"
        assert "Couldn't open settings" in eph["text"]

    async def test_missing_trigger_id_noops(self):
        host = _settings_host()
        handler = host.app.actions["open_user_settings"]
        client = SimpleNamespace(views_open=AsyncMock(), chat_postEphemeral=AsyncMock())
        body = _action_body(); body.pop("trigger_id")
        await handler(ack=AsyncMock(), body=body, client=client)
        client.views_open.assert_not_awaited()
        client.chat_postEphemeral.assert_not_awaited()
