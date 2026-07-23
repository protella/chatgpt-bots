"""Phase B — search_slack tool (assistant.search.context).

Covers: schema shape, action_token plumbing (event → metadata → ToolContext),
executor success mapping, missing-token fallback, the SEARCH_CHANNEL_TYPES code
gate (im excluded by default, included when configured), API-error wrapping,
registry gating, and prompt guidance presence.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from slack_sdk.errors import SlackApiError

from config import config
from slack_client.history_tool import SlackHistoryToolMixin
from slack_client.search_tool import SlackSearchToolMixin
from tool_registry import ToolContext, ToolRegistry


class _Bot(SlackSearchToolMixin, SlackHistoryToolMixin):
    # The real SlackBot mixes search + history on ONE instance; the delivery-audience gate that
    # search now consults (self._source_is_public / self._bot_team_id) lives in the history mixin.
    def __init__(self):
        self.app = MagicMock()
        self.self_team_id = "T_TEST"
        self.bot_user_id = "U_BOT"

    def log_debug(self, *a, **k): pass
    def log_info(self, *a, **k): pass
    def log_warning(self, *a, **k): pass
    def log_error(self, *a, **k): pass


def _ctx(**kw):
    defaults = dict(channel_id="C04QDHE8W8M", thread_ts="1.0", trigger_ts="1.0",
                    action_token="tok-123")
    defaults.update(kw)
    return ToolContext(**defaults)


def _api_response(messages):
    return {"ok": True, "results": {"messages": messages}}


# --- schema ---

def test_schema_shape():
    schema = _Bot().get_search_tool_schema()
    assert schema["type"] == "function"
    assert schema["name"] == "search_slack"
    props = schema["parameters"]["properties"]
    assert set(schema["parameters"]["required"]) == {"query"}
    assert set(props["scope"]["enum"]) == {"channel", "workspace"}
    assert "limit" in props


# --- action_token plumbing ---

@pytest.mark.asyncio
async def test_event_to_message_captures_action_token():
    """_event_to_message must copy the event's action_token into metadata."""
    from slack_client.base import SlackBot
    bot = SlackBot.__new__(SlackBot)  # no __init__ — we only exercise _event_to_message
    bot.bot_user_id = "U07SELF"
    bot.user_cache = {}
    bot.db = MagicMock()
    bot.db.get_user_info_async = AsyncMock(return_value=None)
    bot.get_username = AsyncMock(return_value="peter")
    bot.get_user_timezone = AsyncMock(return_value="UTC")

    event = {"text": "find that thread", "user": "U1", "channel": "C1", "ts": "2.0",
             "action_token": "tok-evt"}
    msg = await bot._event_to_message(event, client=MagicMock())
    assert msg.metadata["action_token"] == "tok-evt"

    event_without = {"text": "hi", "user": "U1", "channel": "C1", "ts": "3.0"}
    msg2 = await bot._event_to_message(event_without, client=MagicMock())
    assert msg2.metadata["action_token"] is None


def test_tool_context_built_from_metadata():
    """The Phase A ToolContext builder passes metadata['action_token'] through."""
    from message_processor.handlers.text import TextHandlerMixin  # noqa: F401 — import proves wiring exists
    import inspect
    import message_processor.handlers.text as text_mod
    src = inspect.getsource(text_mod)
    assert 'action_token=meta.get("action_token")' in src


# --- executor ---

@pytest.mark.asyncio
async def test_search_success_maps_results():
    bot = _Bot()
    bot.app.client.api_call = AsyncMock(return_value=_api_response([
        {"channel_id": "C09", "message_ts": "100.1", "author_user_id": "U9",
         "content": "we decided fridays", "permalink": "https://x/p1"},
        {"channel": {"id": "C08"}, "ts": "90.2", "user": "U8", "text": "older note"},
    ]))
    # DM surface: full reach, so the field-mapping + request plumbing this test covers isn't
    # touched by the delivery-audience filter (which is exercised in test_channel_scope_guard.py).
    out = await bot.execute_search_tool(_ctx(is_dm=True), {"query": "demo day"})
    assert out["ok"] is True and out["count"] == 2
    first = out["results"][0]
    assert first == {"channel": "C09", "ts": "100.1", "author": "U9",
                     "text": "we decided fridays", "permalink": "https://x/p1"}
    # request carried token + context channel + configured channel_types
    kwargs = bot.app.client.api_call.call_args
    assert kwargs.args[0] == "assistant.search.context"
    data = kwargs.kwargs["data"]
    assert data["action_token"] == "tok-123"
    assert data["context_channel_id"] == "C04QDHE8W8M"
    assert data["channel_types"] == "public_channel,private_channel"


@pytest.mark.asyncio
async def test_scope_channel_filters_other_channels():
    bot = _Bot()
    bot.app.client.api_call = AsyncMock(return_value=_api_response([
        {"channel_id": "C04QDHE8W8M", "message_ts": "1.1", "content": "here"},
        {"channel_id": "C_OTHER", "message_ts": "1.2", "content": "elsewhere"},
    ]))
    out = await bot.execute_search_tool(_ctx(), {"query": "x", "scope": "channel"})
    assert out["count"] == 1 and out["results"][0]["channel"] == "C04QDHE8W8M"


@pytest.mark.asyncio
async def test_scope_channel_constrains_query_at_api(monkeypatch):
    """F22: channel scope must append `in:<#CHANNEL_ID>` so the API constrains at the
    source, not just the post-filter (a workspace-wide top-N could miss the channel)."""
    bot = _Bot()
    bot.app.client.api_call = AsyncMock(return_value=_api_response([]))
    await bot.execute_search_tool(_ctx(), {"query": "budget", "scope": "channel"})
    sent = bot.app.client.api_call.call_args.kwargs["data"]["query"]
    assert sent == "budget in:<#C04QDHE8W8M>"


@pytest.mark.asyncio
async def test_scope_workspace_query_unmodified():
    """F22: workspace scope must NOT inject an in:<#...> operator."""
    bot = _Bot()
    bot.app.client.api_call = AsyncMock(return_value=_api_response([]))
    await bot.execute_search_tool(_ctx(), {"query": "budget", "scope": "workspace"})
    sent = bot.app.client.api_call.call_args.kwargs["data"]["query"]
    assert sent == "budget"
    # and the human-facing echo keeps the original query
    out = await bot.execute_search_tool(_ctx(), {"query": "budget"})
    assert out["query"] == "budget"


@pytest.mark.asyncio
async def test_missing_token_falls_back():
    bot = _Bot()
    bot.app.client.api_call = AsyncMock()
    out = await bot.execute_search_tool(_ctx(action_token=None), {"query": "x"})
    assert out["ok"] is False and out["error"] == "search_unavailable"
    assert "fetch_channel_history" in out["hint"]
    bot.app.client.api_call.assert_not_called()


@pytest.mark.asyncio
async def test_empty_query_rejected():
    out = await _Bot().execute_search_tool(_ctx(), {"query": "  "})
    assert out["ok"] is False and out["error"] == "bad_arguments"


# --- channel-type gate ---

@pytest.mark.asyncio
async def test_channel_types_gate_excludes_im_by_default():
    bot = _Bot()
    bot.app.client.api_call = AsyncMock(return_value=_api_response([]))
    await bot.execute_search_tool(_ctx(), {"query": "x"})
    sent = bot.app.client.api_call.call_args.kwargs["data"]["channel_types"]
    assert "im" not in sent.split(",") and "mpim" not in sent.split(",")


@pytest.mark.asyncio
async def test_channel_types_gate_env_widening_and_validation():
    bot = _Bot()
    bot.app.client.api_call = AsyncMock(return_value=_api_response([]))
    with patch.object(config, "search_channel_types", ["public_channel", "im", "bogus_type"]):
        await bot.execute_search_tool(_ctx(), {"query": "x"})
    sent = bot.app.client.api_call.call_args.kwargs["data"]["channel_types"]
    assert sent == "public_channel,im"  # bogus filtered, im honored when configured


@pytest.mark.asyncio
async def test_no_valid_channel_types_disables_search():
    bot = _Bot()
    bot.app.client.api_call = AsyncMock()
    with patch.object(config, "search_channel_types", ["bogus"]):
        out = await bot.execute_search_tool(_ctx(), {"query": "x"})
    assert out["ok"] is False and out["error"] == "search_disabled"
    bot.app.client.api_call.assert_not_called()


# --- error wrapping ---

def _slack_error(err):
    resp = MagicMock()
    resp.get = lambda k, d=None: {"error": err}.get(k, d)
    return SlackApiError(message=err, response=resp)


@pytest.mark.asyncio
async def test_token_error_becomes_search_unavailable():
    bot = _Bot()
    bot.app.client.api_call = AsyncMock(side_effect=_slack_error("invalid_action_token"))
    out = await bot.execute_search_tool(_ctx(), {"query": "x"})
    assert out["ok"] is False and out["error"] == "search_unavailable"


@pytest.mark.asyncio
async def test_api_error_wrapped_never_raises():
    bot = _Bot()
    bot.app.client.api_call = AsyncMock(side_effect=_slack_error("ratelimited"))
    out = await bot.execute_search_tool(_ctx(), {"query": "x"})
    assert out["ok"] is False and out["error"] == "ratelimited"

    bot.app.client.api_call = AsyncMock(side_effect=RuntimeError("boom"))
    out2 = await bot.execute_search_tool(_ctx(), {"query": "x"})
    assert out2["ok"] is False and out2["error"] == "exception"


# --- registry gating + guidance ---

def test_registry_gating():
    bot = _Bot()
    registry = ToolRegistry()
    if config.enable_search_tool:
        registry.register(bot.get_search_tool_schema(), bot.execute_search_tool)
    names = [s["name"] for s in registry.schemas()]
    assert ("search_slack" in names) == config.enable_search_tool

    # default is enabled
    assert config.enable_search_tool is True


def test_limit_clamping():
    clamp = SlackSearchToolMixin._clamp_search_limit
    assert clamp(None) == 10
    assert clamp("nope") == 10
    assert clamp(0) == 1
    assert clamp(500) == 20


def test_guidance_mentions_search():
    from prompts import LOCAL_TOOLS_GUIDANCE
    assert "search_slack" in LOCAL_TOOLS_GUIDANCE
    # BF1: the guidance is availability-conditional now — when search_slack is not among the
    # available tools (no action_token), fall to the fetch tools without comment.
    assert "When search_slack is available" in LOCAL_TOOLS_GUIDANCE
    assert "not among the available tools" in LOCAL_TOOLS_GUIDANCE
