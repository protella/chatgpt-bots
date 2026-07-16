"""F23 — cross-thread reply tool (post_to_thread).

Covers the design points: schema + required params, markdown-converted post into the
TARGET thread via the standard messaging layer, own-reply pulse recording keyed on the
target thread, the refusal rails (current-thread double-post, empty text, missing
thread_ts, disabled), provenance listing, and the never-raises contract on a Slack API
failure. The per-thread mute mechanism was removed, so the tool posts with NO mute lookup.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import config
from tool_registry import ToolContext
from markdown_converter import MarkdownConverter
from slack_client.messaging import SlackMessagingMixin
from slack_client.formatting.text import SlackFormattingMixin
import prompts


def _ctx(channel="C1", thread="root.1", trigger="msg.1"):
    # The per-thread mute mechanism was removed: post_to_thread no longer consults any mute
    # state. The db still exposes is_thread_muted_async so a test can prove it is NEVER called.
    db = MagicMock()
    db.is_thread_muted_async = AsyncMock(return_value=True)
    return ToolContext(channel_id=channel, thread_ts=thread, trigger_ts=trigger,
                       client=MagicMock(), db=db)


def _light_host():
    """Host with the executor bound and send_message MOCKED (refusal / routing tests)."""
    s = MagicMock()
    s.execute_post_to_thread = SlackMessagingMixin.execute_post_to_thread.__get__(s)
    s.send_message = AsyncMock(return_value="900.0")
    return s


def _real_send_host():
    """Host with the REAL send_message + format_text so markdown conversion and the
    own-reply pulse actually run (proves the standard messaging path is used)."""
    s = MagicMock()
    s.execute_post_to_thread = SlackMessagingMixin.execute_post_to_thread.__get__(s)
    s.send_message = SlackMessagingMixin.send_message.__get__(s)
    s.format_text = SlackFormattingMixin.format_text.__get__(s)
    s._encode_mentions = lambda t: t
    s.markdown_converter = MarkdownConverter(platform="slack")
    s._record_own_reply_pulse = SlackMessagingMixin._record_own_reply_pulse.__get__(s)
    s._compose_reply_with_footer = SlackMessagingMixin._compose_reply_with_footer.__get__(s)
    s._SECTION_TEXT_LIMIT = SlackMessagingMixin._SECTION_TEXT_LIMIT
    s.MAX_MESSAGE_LENGTH = 3900
    s.app.client.chat_postMessage = AsyncMock(return_value={"ts": "900.0"})
    pulse = MagicMock()
    pulse.record_own_reply = MagicMock()
    s.channel_pulse = pulse
    return s


# ------------------------------------------------------------------- schema

def test_schema_registered_and_required_params():
    s = MagicMock()
    schema = SlackMessagingMixin.get_post_to_thread_tool_schema.__get__(s)()
    assert schema["name"] == "post_to_thread"
    props = schema["parameters"]["properties"]
    assert set(schema["parameters"]["required"]) == {"thread_ts", "text"}
    assert "thread_ts" in props and "text" in props
    # current-channel-only: no channel_id parameter is exposed
    assert "channel_id" not in props


def test_registered_in_tool_registry(monkeypatch):
    from tool_registry import ToolRegistry
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    reg = ToolRegistry()
    reg.register(
        SlackMessagingMixin.get_post_to_thread_tool_schema.__get__(MagicMock())(),
        AsyncMock(),
    )
    assert "post_to_thread" in [sc["name"] for sc in reg.schemas()]


# ------------------------------------------------------------------- happy path

@pytest.mark.asyncio
async def test_posts_markdown_converted_text_to_target(monkeypatch):
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _real_send_host()
    out = await host.execute_post_to_thread(
        _ctx(thread="root.1", trigger="msg.1"),
        {"thread_ts": "OTHER.9", "text": "Answer: **done**"},
    )
    assert out["ok"] is True and out["thread_ts"] == "OTHER.9"
    call = host.app.client.chat_postMessage.await_args
    assert call.kwargs["thread_ts"] == "OTHER.9"
    # markdown converted to Slack mrkdwn (**bold** -> *bold*)
    assert "*done*" in call.kwargs["text"]


@pytest.mark.asyncio
async def test_records_own_reply_on_target_thread(monkeypatch):
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _real_send_host()
    await host.execute_post_to_thread(
        _ctx(thread="root.1"), {"thread_ts": "OTHER.9", "text": "hi there"}
    )
    host.channel_pulse.record_own_reply.assert_called_once()
    kwargs = host.channel_pulse.record_own_reply.call_args.kwargs
    assert kwargs["thread_ts"] == "OTHER.9" and kwargs["ts"] == "900.0"


@pytest.mark.asyncio
async def test_routes_through_send_message(monkeypatch):
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()
    out = await host.execute_post_to_thread(
        _ctx(), {"thread_ts": "OTHER.9", "text": "hello"}
    )
    assert out["ok"] is True
    host.send_message.assert_awaited_once_with("C1", "OTHER.9", "hello")


# ------------------------------------------------------------------- refusals

@pytest.mark.asyncio
async def test_posts_with_no_mute_lookup(monkeypatch):
    # Even against a target the old code would have refused as "muted", the tool now posts and
    # NEVER consults is_thread_muted_async — the mute mechanism is gone.
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()
    ctx = _ctx()
    out = await host.execute_post_to_thread(
        ctx, {"thread_ts": "OTHER.9", "text": "hello"}
    )
    assert out["ok"] is True and out["posted_ts"] == "900.0"
    ctx.db.is_thread_muted_async.assert_not_awaited()
    host.send_message.assert_awaited_once_with("C1", "OTHER.9", "hello")


@pytest.mark.asyncio
async def test_current_thread_refused(monkeypatch):
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()
    out = await host.execute_post_to_thread(
        _ctx(thread="root.1"), {"thread_ts": "root.1", "text": "hello"}
    )
    assert out["ok"] is False and out["error"] == "same_thread"
    host.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_trigger_ts_also_counts_as_current(monkeypatch):
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()
    out = await host.execute_post_to_thread(
        _ctx(thread="root.1", trigger="msg.1"), {"thread_ts": "msg.1", "text": "hello"}
    )
    assert out["ok"] is False and out["error"] == "same_thread"


@pytest.mark.asyncio
async def test_empty_text_refused(monkeypatch):
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()
    out = await host.execute_post_to_thread(
        _ctx(), {"thread_ts": "OTHER.9", "text": "   "}
    )
    assert out["ok"] is False and out["error"] == "empty_text"


@pytest.mark.asyncio
async def test_missing_thread_ts_refused(monkeypatch):
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()
    out = await host.execute_post_to_thread(_ctx(), {"text": "hello"})
    assert out["ok"] is False and out["error"] == "missing_thread_ts"


@pytest.mark.asyncio
async def test_disabled_refused(monkeypatch):
    monkeypatch.setattr(config, "enable_post_to_thread_tool", False)
    host = _light_host()
    out = await host.execute_post_to_thread(
        _ctx(), {"thread_ts": "OTHER.9", "text": "hello"}
    )
    assert out["ok"] is False and out["error"] == "disabled"


# ------------------------------------------------------------------- never raises

@pytest.mark.asyncio
async def test_never_raises_on_slack_failure(monkeypatch):
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()
    host.send_message = AsyncMock(side_effect=RuntimeError("slack down"))
    out = await host.execute_post_to_thread(
        _ctx(), {"thread_ts": "OTHER.9", "text": "hello"}
    )
    assert out["ok"] is False and out["error"] == "post_failed"


@pytest.mark.asyncio
async def test_send_returns_none_is_failure(monkeypatch):
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _light_host()
    host.send_message = AsyncMock(return_value=None)
    out = await host.execute_post_to_thread(
        _ctx(), {"thread_ts": "OTHER.9", "text": "hello"}
    )
    assert out["ok"] is False and out["error"] == "post_failed"


# ------------------------------------------------------------------- provenance + guidance

def test_provenance_line_includes_post_to_thread():
    from message_processor.tool_provenance import build_provenance, render_used_tools_annotation
    tools = build_provenance([{"name": "post_to_thread", "ok": True, "gist": ""}], [])
    line = render_used_tools_annotation(tools)
    assert "post_to_thread" in line


def test_local_tools_guidance_has_post_to_thread_bullet():
    assert "post_to_thread" in prompts.LOCAL_TOOLS_GUIDANCE
