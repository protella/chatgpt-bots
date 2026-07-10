"""F8 — settings-footer chrome attached to NON-streamed replies.

Covers the extended send seam (single message attaches blocks + returns ts; split
ignores blocks), maybe_post_response_footer standing down when footer_attached is set,
and the main.py non-streamed branch: footer blocks attached + footer_attached metadata
on a normal reply, suppression on top-level channel placement, and degradation to the
separate-footer fallback when block-building yields nothing.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from base_client import Message, Response
from config import config
from slack_client.formatting.text import SlackFormattingMixin
from slack_client.messaging import SlackMessagingMixin
from slack_client.utilities import SlackUtilitiesMixin


class _Bot(SlackMessagingMixin, SlackFormattingMixin, SlackUtilitiesMixin):
    MAX_MESSAGE_LENGTH = 3900

    def __init__(self):
        self.app = MagicMock()
        self.markdown_converter = MagicMock()

    def log_info(self, *a, **k): pass
    log_debug = log_warning = log_error = log_info

    def format_text(self, text):
        return text


# ------------------------------------------------------- send seam: blocks + ts

@pytest.mark.asyncio
async def test_single_message_attaches_blocks_and_returns_ts():
    # F8 fix: action-only blocks would render INSTEAD of the reply text (hiding the
    # answer), so the reply rides a leading section block, then the footer actions.
    b = _Bot()
    b.app.client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "111.0"})
    blocks = [{"type": "actions", "elements": []}]
    meta = {}
    ts = await b.send_message("C1", "T1", "short reply", blocks=blocks, meta_out=meta)
    assert ts == "111.0"
    kwargs = b.app.client.chat_postMessage.await_args.kwargs
    assert kwargs["blocks"][0]["type"] == "section"
    assert kwargs["blocks"][0]["text"]["text"] == "short reply"   # answer visible
    assert kwargs["blocks"][1:] == blocks                          # footer follows
    assert kwargs["text"] == "short reply"                         # notification fallback
    assert meta["footer_attached"] is True


@pytest.mark.asyncio
async def test_too_long_for_section_does_not_attach_footer():
    # A reply that doesn't fit one section block can't carry the footer as blocks — it
    # posts as plain text and reports footer_attached False so the separate footer posts.
    b = _Bot()
    b.app.client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "111.0"})
    long_reply = "x" * 3200  # > section limit (2900), still <= MAX_MESSAGE_LENGTH (3900)
    meta = {}
    await b.send_message("C1", "T1", long_reply, blocks=[{"type": "actions"}], meta_out=meta)
    kwargs = b.app.client.chat_postMessage.await_args.kwargs
    assert "blocks" not in kwargs
    assert kwargs["text"] == long_reply
    assert meta["footer_attached"] is False


@pytest.mark.asyncio
async def test_no_blocks_when_none_passed():
    b = _Bot()
    b.app.client.chat_postMessage = AsyncMock(return_value={"ok": True, "ts": "111.0"})
    await b.send_message("C1", "T1", "short reply")
    assert "blocks" not in b.app.client.chat_postMessage.await_args.kwargs


@pytest.mark.asyncio
async def test_split_message_ignores_blocks():
    # A split reply falls back to the separate footer post — attaching chrome to an
    # arbitrary chunk would misplace it. No chunk carries blocks.
    b = _Bot()
    b.app.client.chat_postMessage = AsyncMock(
        side_effect=lambda **kw: {"ok": True, "ts": "1"})
    long_text = ("word " * 400 + "\n\n") * 6  # forces a split
    await b.send_message("C1", "T1", long_text, blocks=[{"type": "actions"}])
    assert b.app.client.chat_postMessage.await_count >= 2
    for call in b.app.client.chat_postMessage.await_args_list:
        assert "blocks" not in call.kwargs


# ------------------------------------------- maybe_post_response_footer stand-down

@pytest.mark.asyncio
async def test_footer_stands_down_when_already_attached(monkeypatch):
    monkeypatch.setattr(config, "enable_response_footer", True)
    s = MagicMock()
    s.app.client.chat_postMessage = AsyncMock()
    s.log_debug = MagicMock()
    s._build_response_footer_blocks = SlackMessagingMixin._build_response_footer_blocks.__get__(s)
    msg = Message(text="hi", user_id="U1", channel_id="C1", thread_id="T1")
    resp = Response(type="text", content="hello", metadata={"model": "m", "footer_attached": True})
    await SlackMessagingMixin.maybe_post_response_footer(s, msg, resp)
    s.app.client.chat_postMessage.assert_not_awaited()  # chrome already rode the message


# ------------------------------------------------------- main.py non-streamed branch

def _make_bot():
    from main import ChatBotV2
    bot = ChatBotV2(platform="slack")
    bot.processor = MagicMock()
    bot.processor._persist_tool_provenance = MagicMock()
    return bot


def _client():
    c = MagicMock()
    c.send_thinking_indicator = AsyncMock(return_value=None)
    c.delete_message = AsyncMock()

    # Mirror the real send seam: report footer_attached via meta_out when blocks ride the
    # message (the composed section+actions path), so main.py sets its flag from reality.
    async def _send(channel_id, thread_id, text, blocks=None, meta_out=None):
        if meta_out is not None:
            meta_out["footer_attached"] = bool(blocks)
        return "posted.1"
    c.send_message = AsyncMock(side_effect=_send)
    c.format_text = MagicMock(side_effect=lambda t: t)
    c.attachable_footer_blocks = MagicMock(return_value=[{"type": "actions"}])
    c.maybe_post_response_footer = AsyncMock()
    c.channel_pulse = None
    return c


@pytest.mark.asyncio
async def test_main_nonstreamed_attaches_footer_blocks_and_sets_metadata():
    bot = _make_bot()
    client = _client()
    md = {"streamed": False, "model": "gpt-5.6-sol"}
    resp = Response(type="text", content="here you go", metadata=md)
    bot.processor.process_message = AsyncMock(return_value=resp)
    # DM → never top-level placement, so the footer path is active.
    message = Message(text="q", user_id="U1", channel_id="D123", thread_id="T1",
                      metadata={"ts": "200.0"})

    await bot.handle_message(message, client)

    # Blocks rode the reply message itself…
    assert client.send_message.await_args.kwargs.get("blocks") == [{"type": "actions"}]
    # …and the metadata tells the separate footer to stand down.
    assert resp.metadata.get("footer_attached") is True
    assert resp.metadata.get("posted") is True


@pytest.mark.asyncio
async def test_main_top_level_channel_placement_suppresses_footer():
    bot = _make_bot()
    client = _client()
    resp = Response(type="text", content="quick answer", metadata={"streamed": False, "model": "m"})
    bot.processor.process_message = AsyncMock(return_value=resp)
    # Top-level channel trigger with reply_in_channel → place_in_channel True.
    message = Message(text="q", user_id="U1", channel_id="C1", thread_id="200.0",
                      metadata={"ts": "200.0", "reply_in_channel": True})

    await bot.handle_message(message, client)

    client.attachable_footer_blocks.assert_not_called()
    assert client.send_message.await_args.kwargs.get("blocks") is None
    assert resp.metadata.get("footer_attached") is not True


@pytest.mark.asyncio
async def test_main_degrades_to_fallback_when_no_blocks_built():
    bot = _make_bot()
    client = _client()
    client.attachable_footer_blocks = MagicMock(return_value=None)  # footer disabled/unbuildable
    resp = Response(type="text", content="hello", metadata={"streamed": False, "model": "m"})
    bot.processor.process_message = AsyncMock(return_value=resp)
    message = Message(text="q", user_id="U1", channel_id="D123", thread_id="T1",
                      metadata={"ts": "200.0"})

    await bot.handle_message(message, client)

    assert client.send_message.await_args.kwargs.get("blocks") is None
    assert resp.metadata.get("footer_attached") is not True
    # Falls back to the separate trailing footer post.
    client.maybe_post_response_footer.assert_awaited_once()
