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
        self.app.client.chat_getPermalink = AsyncMock()
        self.app.client.pins_list = AsyncMock()
        self.app.client.users_info = AsyncMock()

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
    assert names == {
        "fetch_channel_history", "fetch_thread_messages", "get_message_permalink",
        "fetch_channel_info", "fetch_pinned_messages", "fetch_user_profile",
    }
    assert all(t["type"] == "function" for t in tools)


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
async def test_dispatch_defaults_channel_from_ctx(bot):
    # Regression (2026-07-10): requiring channel_id made the model fabricate IDs
    # (channel_not_found). Omitted channel_id falls back to the current channel.
    from types import SimpleNamespace
    bot.app.client.conversations_info.return_value = {"channel": {"is_private": False}}
    bot.app.client.conversations_history.return_value = {"messages": []}
    ctx = SimpleNamespace(channel_id="C_CUR", thread_ts="9.0")
    res = await bot.dispatch_history_tool_call("fetch_channel_history", {}, ctx)
    assert res["ok"] is True
    assert bot.app.client.conversations_history.call_args.kwargs["channel"] == "C_CUR"


@pytest.mark.asyncio
async def test_dispatch_defaults_thread_from_ctx(bot):
    from types import SimpleNamespace
    bot.app.client.conversations_info.return_value = {"channel": {"is_private": False}}
    bot.app.client.conversations_replies.return_value = {"messages": []}
    ctx = SimpleNamespace(channel_id="C_CUR", thread_ts="9.0")
    res = await bot.dispatch_history_tool_call("fetch_thread_messages", {}, ctx)
    assert res["ok"] is True
    kw = bot.app.client.conversations_replies.call_args.kwargs
    assert kw["channel"] == "C_CUR" and kw["ts"] == "9.0"


@pytest.mark.asyncio
async def test_dispatch_thread_without_ts_anywhere_is_bad_arguments(bot):
    from types import SimpleNamespace
    ctx = SimpleNamespace(channel_id="C_CUR", thread_ts=None)
    res = await bot.dispatch_history_tool_call("fetch_thread_messages", {}, ctx)
    assert res["ok"] is False and res["error"] == "bad_arguments"


def test_schemas_do_not_require_channel_id(bot):
    # channel_id must stay optional in every schema — required IDs get hallucinated.
    for schema in bot.get_history_tools_for_openai():
        assert "channel_id" not in schema["parameters"].get("required", []), schema["name"]


@pytest.mark.asyncio
async def test_dispatch_unknown_tool(bot):
    res = await bot.dispatch_history_tool_call("nope", {})
    assert res["ok"] is False and res["error"] == "unknown_tool"


@pytest.mark.asyncio
async def test_dispatch_bad_json(bot):
    res = await bot.dispatch_history_tool_call("fetch_channel_history", "{not json")
    assert res["ok"] is False and res["error"] == "bad_arguments"


# --- reactions in history payloads ---

@pytest.mark.asyncio
async def test_history_includes_current_reactions(bot):
    bot.app.client.conversations_info.return_value = {"channel": {"is_private": False}}
    bot.app.client.conversations_history.return_value = {"messages": [
        {"user": "U1", "ts": "1.1", "text": "hi",
         "reactions": [{"name": "thumbsup", "count": 2, "users": ["U2", "U3"]}]},
        {"user": "U2", "ts": "1.2", "text": "plain"},
    ]}
    res = await bot.fetch_history_tool("C_PUBLIC")
    assert res["messages"][0]["reactions"] == [{"emoji": "thumbsup", "count": 2, "users": ["U2", "U3"]}]
    assert "reactions" not in res["messages"][1]  # absent when a message has none


# --- message permalinks ---

@pytest.mark.asyncio
async def test_permalink_returned_for_accessible_channel(bot):
    bot.app.client.conversations_info.return_value = {"channel": {"is_private": False}}
    bot.app.client.chat_getPermalink.return_value = {
        "permalink": "https://acme.slack.com/archives/C1/p1720500000123456"
    }
    res = await bot.get_message_permalink_tool("C1", "1720500000.123456")
    assert res["ok"] is True
    assert res["permalink"].startswith("https://")
    kwargs = bot.app.client.chat_getPermalink.await_args.kwargs
    assert kwargs == {"channel": "C1", "message_ts": "1720500000.123456"}


@pytest.mark.asyncio
async def test_permalink_refused_for_private_non_member(bot):
    bot.app.client.conversations_info.return_value = {"channel": {"is_private": True, "is_member": False}}
    res = await bot.get_message_permalink_tool("C_PRIV", "1.0")
    assert res["ok"] is False and res["error"] == "not_accessible"
    assert "permalink" not in res
    bot.app.client.chat_getPermalink.assert_not_called()


@pytest.mark.asyncio
async def test_permalink_api_error_graceful(bot):
    bot.app.client.conversations_info.return_value = {"channel": {"is_private": False}}
    bot.app.client.chat_getPermalink.side_effect = SlackApiError("x", {"error": "message_not_found"})
    res = await bot.get_message_permalink_tool("C1", "9.9")
    assert res["ok"] is False and res["error"] == "message_not_found"


# --- channel info ---

@pytest.mark.asyncio
async def test_channel_info_returns_facts(bot):
    bot.app.client.conversations_info.return_value = {"channel": {
        "is_private": False, "name": "menu-insights",
        "topic": {"value": "menus"}, "purpose": {"value": "menu data"}, "num_members": 42,
    }}
    res = await bot.fetch_channel_info_tool("C1")
    assert res == {"ok": True, "channel": "C1", "name": "menu-insights", "topic": "menus",
                   "purpose": "menu data", "num_members": 42, "is_private": False}


# --- pinned messages ---

@pytest.mark.asyncio
async def test_pins_listed_messages_only(bot):
    bot.app.client.conversations_info.return_value = {"channel": {"is_private": False}}
    bot.app.client.pins_list.return_value = {"items": [
        {"message": {"user": "U1", "ts": "1.1", "text": "release checklist",
                     "permalink": "https://acme.slack.com/archives/C1/p11"}},
        {"file": {"id": "F1"}},  # pinned file: skipped
    ]}
    res = await bot.fetch_pinned_messages_tool("C1")
    assert res["ok"] is True and res["count"] == 1
    assert res["pins"][0]["text"] == "release checklist"


@pytest.mark.asyncio
async def test_pins_missing_scope_names_the_fix(bot):
    bot.app.client.conversations_info.return_value = {"channel": {"is_private": False}}
    bot.app.client.pins_list.side_effect = SlackApiError("x", {"error": "missing_scope"})
    res = await bot.fetch_pinned_messages_tool("C1")
    assert res["ok"] is False and res["error"] == "missing_scope"
    assert "pins:read" in res["message"]


@pytest.mark.asyncio
async def test_pins_refused_for_private_non_member(bot):
    bot.app.client.conversations_info.return_value = {"channel": {"is_private": True, "is_member": False}}
    res = await bot.fetch_pinned_messages_tool("C_PRIV")
    assert res["ok"] is False and res["error"] == "not_accessible"
    bot.app.client.pins_list.assert_not_called()


# --- user profiles ---

@pytest.mark.asyncio
async def test_user_profile_returns_card_facts_no_email(bot):
    bot.app.client.users_info.return_value = {"user": {
        "tz": "America/New_York", "tz_label": "Eastern Daylight Time", "is_bot": False,
        "profile": {"real_name": "Erin Evans", "display_name": "peter",
                    "title": "Slackbot Wrangler", "email": "should-not-leak@x.com"},
    }}
    res = await bot.fetch_user_profile_tool("U1")
    assert res["ok"] is True
    assert res["real_name"] == "Erin Evans"
    assert res["timezone"] == "America/New_York"
    assert "email" not in res  # profile-card facts only


@pytest.mark.asyncio
async def test_user_profile_error_graceful(bot):
    bot.app.client.users_info.side_effect = SlackApiError("x", {"error": "user_not_found"})
    res = await bot.fetch_user_profile_tool("UNOPE")
    assert res["ok"] is False and res["error"] == "user_not_found"


# --- dispatch routing for the new tools ---

@pytest.mark.asyncio
@pytest.mark.parametrize("name,args,client_attr", [
    ("get_message_permalink", {"channel_id": "C1", "message_ts": "1.0"}, "chat_getPermalink"),
    ("fetch_channel_info", {"channel_id": "C1"}, "conversations_info"),
    ("fetch_pinned_messages", {"channel_id": "C1"}, "pins_list"),
    ("fetch_user_profile", {"user_id": "U1"}, "users_info"),
])
async def test_dispatch_routes_new_tools(bot, name, args, client_attr):
    bot.app.client.conversations_info.return_value = {"channel": {"is_private": False}}
    bot.app.client.chat_getPermalink.return_value = {"permalink": "https://x"}
    bot.app.client.pins_list.return_value = {"items": []}
    bot.app.client.users_info.return_value = {"user": {"profile": {}}}
    res = await bot.dispatch_history_tool_call(name, args)
    assert res["ok"] is True
    getattr(bot.app.client, client_attr).assert_called()
