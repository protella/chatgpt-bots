"""Phase 7 entry point — .env-backed config lists, the channel-settings modal, and the
Configure-button footer. No slash command. All stubbed I/O — no live bot, no legacy suite.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import config, _env_list
from database import DatabaseManager
from settings_modal import SettingsModal
from slack_client.messaging import SlackMessagingMixin
from base_client import Message, Response


# --------------------------------------------------------------------------- env lists
class TestEnvList:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("XLIST", raising=False)
        assert _env_list("XLIST", ["a", "b"]) == ["a", "b"]

    def test_comma_split_and_trim(self, monkeypatch):
        monkeypatch.setenv("XLIST", " a, b ,c,  ")
        assert _env_list("XLIST", ["z"]) == ["a", "b", "c"]

    def test_empty_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("XLIST", "  , ,")
        assert _env_list("XLIST", ["z"]) == ["z"]


# --------------------------------------------------------------------------- modal builder
class TestChannelSettingsModal:
    @pytest.fixture
    def modal(self):
        return SettingsModal(db=MagicMock())

    def _block(self, view, block_id):
        return next(b for b in view["blocks"] if b.get("block_id") == block_id)

    def test_no_row_inherits(self, modal):
        view = modal.build_channel_settings_modal("C1", None, "tag_only")
        assert view["callback_id"] == "channel_settings_modal"
        assert json.loads(view["private_metadata"])["channel_id"] == "C1"
        assert self._block(view, "participation_block")["element"]["initial_option"]["value"] == "inherit"
        assert self._block(view, "directives_block")["element"]["initial_value"] == ""
        # reply-in-channel unchecked → no initial_options
        assert "initial_options" not in self._block(view, "reply_in_channel_block")["element"]

    def test_prefill_from_row(self, modal):
        # Legacy row (response_mode only) maps to its participation-level equivalent.
        cs = {"response_mode": "auto_respond", "directives": "only deploys", "reply_in_channel": True}
        view = modal.build_channel_settings_modal("C2", cs, "tag_only")
        assert self._block(view, "participation_block")["element"]["initial_option"]["value"] == "judicious"
        assert self._block(view, "directives_block")["element"]["initial_value"] == "only deploys"
        assert self._block(view, "reply_in_channel_block")["element"].get("initial_options")

    def test_null_mode_treated_as_inherit(self, modal):
        view = modal.build_channel_settings_modal("C3", {"response_mode": None}, "off")
        assert self._block(view, "participation_block")["element"]["initial_option"]["value"] == "inherit"


# --------------------------------------------------------------------------- footer blocks
class TestFooterBlocks:
    def test_single_compact_row(self):
        """One actions block only — a single button carrying the model name (no context row)."""
        blocks = SlackMessagingMixin._build_response_footer_blocks(MagicMock(), "gpt-5.5")
        assert len(blocks) == 1
        assert blocks[0]["type"] == "actions"
        button = blocks[0]["elements"][0]
        assert button["action_id"] == "open_channel_settings"
        assert "gpt-5.5" in button["text"]["text"]

    def test_model_fallback(self):
        blocks = SlackMessagingMixin._build_response_footer_blocks(MagicMock(), None)
        assert config.gpt_model in blocks[0]["elements"][0]["text"]["text"]


# --------------------------------------------------------------------------- footer gating
class TestFooterPosting:
    def _fake_self(self):
        s = MagicMock()
        s.app.client.chat_postMessage = AsyncMock()
        s.log_debug = MagicMock()
        s._build_response_footer_blocks = SlackMessagingMixin._build_response_footer_blocks.__get__(s)
        return s

    @pytest.mark.asyncio
    async def test_posts_for_channel_text(self, monkeypatch):
        monkeypatch.setattr(config, "enable_response_footer", True)
        s = self._fake_self()
        msg = Message(text="hi", user_id="U1", channel_id="C1", thread_id="T1")
        resp = Response(type="text", content="hello", metadata={"model": "gpt-5.5"})
        await SlackMessagingMixin.maybe_post_response_footer(s, msg, resp)
        s.app.client.chat_postMessage.assert_awaited_once()
        kwargs = s.app.client.chat_postMessage.await_args.kwargs
        assert kwargs["channel"] == "C1"
        assert kwargs["blocks"][0]["elements"][0]["action_id"] == "open_channel_settings"

    @pytest.mark.asyncio
    async def test_skips_empty_content(self, monkeypatch):
        """Reaction-only turns (empty text) post no message, so no footer either."""
        monkeypatch.setattr(config, "enable_response_footer", True)
        s = self._fake_self()
        msg = Message(text="hi", user_id="U1", channel_id="C1", thread_id="T1")
        resp = Response(type="text", content="", metadata={"reaction_only": True})
        await SlackMessagingMixin.maybe_post_response_footer(s, msg, resp)
        s.app.client.chat_postMessage.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_dm(self, monkeypatch):
        monkeypatch.setattr(config, "enable_response_footer", True)
        s = self._fake_self()
        msg = Message(text="hi", user_id="U1", channel_id="D1", thread_id="T1")
        await SlackMessagingMixin.maybe_post_response_footer(s, msg, Response(type="text", content="x"))
        s.app.client.chat_postMessage.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_when_flag_off(self, monkeypatch):
        monkeypatch.setattr(config, "enable_response_footer", False)
        s = self._fake_self()
        msg = Message(text="hi", user_id="U1", channel_id="C1", thread_id="T1")
        await SlackMessagingMixin.maybe_post_response_footer(s, msg, Response(type="text", content="x"))
        s.app.client.chat_postMessage.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_skips_non_text(self, monkeypatch):
        monkeypatch.setattr(config, "enable_response_footer", True)
        s = self._fake_self()
        msg = Message(text="hi", user_id="U1", channel_id="C1", thread_id="T1")
        await SlackMessagingMixin.maybe_post_response_footer(s, msg, Response(type="error", content="boom"))
        s.app.client.chat_postMessage.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_never_raises(self, monkeypatch):
        monkeypatch.setattr(config, "enable_response_footer", True)
        s = self._fake_self()
        s.app.client.chat_postMessage = AsyncMock(side_effect=Exception("slack down"))
        msg = Message(text="hi", user_id="U1", channel_id="C1", thread_id="T1")
        await SlackMessagingMixin.maybe_post_response_footer(s, msg, Response(type="text", content="x"))  # no raise


# --------------------------------------------------------------------------- inherit → clears (what the modal submit does)
class TestInheritClears:
    @pytest.fixture
    def temp_db(self):
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

    def test_omitted_arg_preserves(self, temp_db):
        temp_db.set_channel_settings("C1", response_mode="auto_respond", directives="rule")
        temp_db.set_channel_settings("C1", directives="rule2")  # omit mode → preserved
        assert temp_db.get_channel_settings("C1")["response_mode"] == "auto_respond"

    def test_explicit_none_clears_mode(self, temp_db):
        temp_db.set_channel_settings("C1", response_mode="auto_respond")
        temp_db.set_channel_settings("C1", response_mode=None)  # "inherit"
        assert temp_db.get_channel_settings("C1")["response_mode"] is None

    def test_explicit_none_clears_directives(self, temp_db):
        temp_db.set_channel_settings("C1", directives="only deploys")
        temp_db.set_channel_settings("C1", directives=None)
        assert temp_db.get_channel_settings("C1")["directives"] is None

    @pytest.mark.asyncio
    async def test_async_inherit_clears(self, temp_db):
        await temp_db.set_channel_settings_async("C2", response_mode="off")
        await temp_db.set_channel_settings_async("C2", response_mode=None)
        row = await temp_db.get_channel_settings_async("C2")
        assert row["response_mode"] is None
