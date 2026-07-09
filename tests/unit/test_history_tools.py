"""Phase 8 — on-demand Slack history-fetch tool: schema, bounded limit, and the privacy gate."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from slack_sdk.errors import SlackApiError

from slack_client.history_tool import SlackHistoryToolMixin
from config import config


class _Harness(SlackHistoryToolMixin):
    """Minimal object exposing the mixin with a mocked async Slack client + no-op logging."""

    def __init__(self):
        self.app = MagicMock()
        self.app.client = MagicMock()
        self.app.client.conversations_info = AsyncMock()
        self.app.client.conversations_history = AsyncMock()
        self.app.client.conversations_replies = AsyncMock()

    def log_debug(self, *a, **k):
        pass

    def log_info(self, *a, **k):
        pass

    def log_warning(self, *a, **k):
        pass

    def log_error(self, *a, **k):
        pass


@pytest.fixture
def bot():
    return _Harness()


@pytest.fixture(autouse=True)
def _enable(monkeypatch):
    # Deterministic defaults regardless of env.
    monkeypatch.setattr(config, "enable_history_tools", True)
    monkeypatch.setattr(config, "history_tool_max_messages", 50)


# --- schema / feature flag ---

def test_tools_exposed_when_enabled(bot):
    tools = bot.get_history_tools_for_openai()
    names = {t["name"] for t in tools}
    assert names == {"fetch_channel_history", "fetch_thread_messages"}
    assert all(t["type"] == "function" for t in tools)
    assert all("channel_id" in t["parameters"]["properties"] for t in tools)


def test_no_tools_when_disabled(bot, monkeypatch):
    monkeypatch.setattr(config, "enable_history_tools", False)
    assert bot.get_history_tools_for_openai() == []


# --- bounded limit ---

def test_limit_clamped(bot):
    assert bot._clamp_limit(None) == 50
    assert bot._clamp_limit(9999) == 50      # capped
    assert bot._clamp_limit(0) == 50         # falsy -> default cap
    assert bot._clamp_limit(-5) == 1         # floored
    assert bot._clamp_limit(10) == 10
    assert bot._clamp_limit("bad") == 50     # non-int -> default cap


@pytest.mark.asyncio
async def test_fetch_passes_clamped_limit(bot):
    bot.app.client.conversations_info.return_value = {"channel": {"is_private": False}}
    bot.app.client.conversations_history.return_value = {"messages": []}
    await bot.fetch_history_tool("C1", limit=9999)
    assert bot.app.client.conversations_history.call_args.kwargs["limit"] == 50


# --- privacy gate ---

@pytest.mark.asyncio
async def test_public_channel_allowed(bot):
    bot.app.client.conversations_info.return_value = {"channel": {"is_private": False, "is_member": False}}
    bot.app.client.conversations_history.return_value = {
        "messages": [{"user": "U1", "ts": "1.1", "text": "hi"}]
    }
    res = await bot.fetch_history_tool("C_PUBLIC")
    assert res["ok"] is True
    assert res["count"] == 1
    assert res["messages"][0]["text"] == "hi"


@pytest.mark.asyncio
async def test_private_member_allowed(bot):
    bot.app.client.conversations_info.return_value = {"channel": {"is_private": True, "is_member": True}}
    bot.app.client.conversations_history.return_value = {"messages": [{"user": "U1", "ts": "1.1", "text": "secret-ok"}]}
    res = await bot.fetch_history_tool("C_PRIV_MEMBER")
    assert res["ok"] is True
    assert res["messages"][0]["text"] == "secret-ok"


@pytest.mark.asyncio
async def test_private_non_member_refused_no_content(bot):
    bot.app.client.conversations_info.return_value = {"channel": {"is_private": True, "is_member": False}}
    res = await bot.fetch_history_tool("C_PRIV_FOREIGN")
    assert res["ok"] is False
    assert res["error"] == "not_accessible"
    assert "messages" not in res            # critical: NO content leaked
    # and we never even attempted to read the channel
    bot.app.client.conversations_history.assert_not_called()


@pytest.mark.asyncio
async def test_channel_info_error_refused_no_content(bot):
    bot.app.client.conversations_info.side_effect = SlackApiError("boom", {"error": "channel_not_found"})
    res = await bot.fetch_history_tool("C_MISSING")
    assert res["ok"] is False
    assert res["error"] == "not_accessible"
    assert "messages" not in res
    bot.app.client.conversations_history.assert_not_called()


@pytest.mark.asyncio
async def test_missing_channel_refused(bot):
    res = await bot.fetch_history_tool("")
    assert res["ok"] is False and res["error"] == "not_accessible"


# --- routing: thread vs channel ---

@pytest.mark.asyncio
async def test_thread_uses_replies(bot):
    bot.app.client.conversations_info.return_value = {"channel": {"is_private": False}}
    bot.app.client.conversations_replies.return_value = {"messages": [{"user": "U1", "ts": "1.1", "text": "t"}]}
    res = await bot.fetch_history_tool("C1", thread_ts="1.0")
    assert res["ok"] is True
    bot.app.client.conversations_replies.assert_awaited_once()
    bot.app.client.conversations_history.assert_not_called()


@pytest.mark.asyncio
async def test_channel_uses_history(bot):
    bot.app.client.conversations_info.return_value = {"channel": {"is_private": False}}
    bot.app.client.conversations_history.return_value = {"messages": []}
    await bot.fetch_history_tool("C1")
    bot.app.client.conversations_history.assert_awaited_once()
    bot.app.client.conversations_replies.assert_not_called()


@pytest.mark.asyncio
async def test_fetch_api_error_returns_no_content(bot):
    bot.app.client.conversations_info.return_value = {"channel": {"is_private": False}}
    bot.app.client.conversations_history.side_effect = SlackApiError("x", {"error": "not_in_channel"})
    res = await bot.fetch_history_tool("C1")
    assert res["ok"] is False
    assert res["error"] == "not_in_channel"
    assert "messages" not in res


# --- dispatch ---

@pytest.mark.asyncio
async def test_dispatch_channel(bot):
    bot.app.client.conversations_info.return_value = {"channel": {"is_private": False}}
    bot.app.client.conversations_history.return_value = {"messages": []}
    res = await bot.dispatch_history_tool_call("fetch_channel_history", {"channel_id": "C1", "limit": 5})
    assert res["ok"] is True
    bot.app.client.conversations_history.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatch_thread_with_json_string_args(bot):
    bot.app.client.conversations_info.return_value = {"channel": {"is_private": False}}
    bot.app.client.conversations_replies.return_value = {"messages": []}
    res = await bot.dispatch_history_tool_call("fetch_thread_messages", '{"channel_id": "C1", "thread_ts": "1.0"}')
    assert res["ok"] is True
    bot.app.client.conversations_replies.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispatch_unknown_tool(bot):
    res = await bot.dispatch_history_tool_call("nope", {})
    assert res["ok"] is False and res["error"] == "unknown_tool"


@pytest.mark.asyncio
async def test_dispatch_bad_json(bot):
    res = await bot.dispatch_history_tool_call("fetch_channel_history", "{not json")
    assert res["ok"] is False and res["error"] == "bad_arguments"
