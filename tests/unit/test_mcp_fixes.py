"""MCP hardening fixes — env interpolation, per-server enable, failure detection
accumulation, discovery-cache wiring, approval warning, health probe."""
import asyncio
import json
import re

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch


from openai_client.mcp_manager import MCPManager
from openai_client.api.responses import _collect_mcp_list_tools


# ---------- helpers ----------

def _manager_with_config(tmp_path, monkeypatch, config_data):
    """Build an MCPManager loading the given config dict from a temp file."""
    path = tmp_path / "mcp_config.json"
    path.write_text(json.dumps(config_data))
    from config import config as bot_config
    monkeypatch.setattr(bot_config, "mcp_config_path", str(path))
    mgr = MCPManager(db=None)
    mgr.initialize()
    return mgr


def _server(url="https://example.com/mcp", **extra):
    base = {"server_url": url, "require_approval": "never"}
    base.update(extra)
    return base


# ---------- 1. env-var interpolation ----------

def test_interpolation_resolves_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_MCP_KEY", "secret-value")
    mgr = _manager_with_config(tmp_path, monkeypatch, {"mcpServers": {
        "s1": _server(headers={"X-API-Key": "${TEST_MCP_KEY}"}),
    }})
    assert mgr.servers["s1"]["headers"]["X-API-Key"] == "secret-value"


def test_interpolation_unresolved_skips_server_keeps_others(tmp_path, monkeypatch):
    monkeypatch.delenv("DEFINITELY_NOT_SET_VAR", raising=False)
    mgr = _manager_with_config(tmp_path, monkeypatch, {"mcpServers": {
        "broken": _server(headers={"X-API-Key": "${DEFINITELY_NOT_SET_VAR}"}),
        "fine": _server(),
    }})
    assert "broken" not in mgr.servers
    assert "fine" in mgr.servers


def test_interpolation_passthrough_without_placeholder(tmp_path, monkeypatch):
    mgr = _manager_with_config(tmp_path, monkeypatch, {"mcpServers": {
        "s1": _server(headers={"X-API-Key": "literal-key"}),
    }})
    assert mgr.servers["s1"]["headers"]["X-API-Key"] == "literal-key"


# ---------- 2. per-server enabled flag ----------

def test_disabled_server_skipped(tmp_path, monkeypatch):
    mgr = _manager_with_config(tmp_path, monkeypatch, {"mcpServers": {
        "off": _server(enabled=False),
        "on": _server(enabled=True),
        "default": _server(),
    }})
    assert set(mgr.servers) == {"on", "default"}


# ---------- 6. approval warning ----------

def test_non_never_approval_warns_but_loads(tmp_path, monkeypatch):
    with patch.object(MCPManager, "log_warning") as warn:
        mgr = _manager_with_config(tmp_path, monkeypatch, {"mcpServers": {
            "s1": _server(require_approval="always"),
        }})
    assert "s1" in mgr.servers
    assert any("forcing 'never'" in str(c) for c in warn.call_args_list)
    # And the built tool def still forces never
    tools = mgr.get_tools_for_openai()
    assert tools[0]["require_approval"] == "never"


# ---------- 4. failure detection + accumulation ----------

def _text_mixin():
    from message_processor.handlers.text import TextHandlerMixin
    obj = TextHandlerMixin.__new__(TextHandlerMixin)
    obj.log_warning = lambda *a, **k: None
    obj.log_info = lambda *a, **k: None
    obj.log_error = lambda *a, **k: None
    obj.log_debug = lambda *a, **k: None
    return obj


def test_exclusion_set_normalization():
    h = _text_mixin()
    assert h._as_mcp_exclusion_set(None) == set()
    assert h._as_mcp_exclusion_set("a") == {"a"}
    assert h._as_mcp_exclusion_set({"a", "b"}) == {"a", "b"}
    assert h._as_mcp_exclusion_set(["a", "a"]) == {"a"}


def test_extract_failed_server_structured_first():
    h = _text_mixin()
    e = Exception("boring text")
    e.status_code = 424
    e.body = {"error": {"message": "Error retrieving tool list from MCP server: 'context7'"}}
    assert h._extract_failed_mcp_server(e) == "context7"


def test_extract_failed_server_regex_fallback():
    h = _text_mixin()
    e = Exception("Error retrieving tool list from MCP server: 'aws_knowledge' (http 500)")
    assert h._extract_failed_mcp_server(e) == "aws_knowledge"


def test_extract_failed_server_none_for_generic():
    h = _text_mixin()
    assert h._extract_failed_mcp_server(Exception("rate limit exceeded")) is None


def test_build_tools_array_excludes_multiple_servers():
    h = _text_mixin()
    h.mcp_manager = MagicMock()
    h.mcp_manager.has_mcp_servers.return_value = True
    h.mcp_manager.get_tools_for_openai.return_value = [
        {"type": "mcp", "server_label": "a"},
        {"type": "mcp", "server_label": "b"},
        {"type": "mcp", "server_label": "c"},
    ]
    with patch("message_processor.handlers.text.config") as cfg:
        cfg.enable_web_search = False
        cfg.mcp_enabled_default = True
        # F32: this test is about MCP exclusion — keep code_interpreter out of the array so
        # it stays focused (a MagicMock config would otherwise read as enabled).
        cfg.enable_code_interpreter = False
        tools = h._build_tools_array({"enable_web_search": False, "enable_mcp": True},
                                     "gpt-5.5", exclude_mcp_server={"a", "c"})
    labels = [t["server_label"] for t in tools]
    assert labels == ["b"]


def test_streaming_retry_accumulates_exclusions_source_gate():
    """The streaming except block must union prior exclusions with the new failure
    and bound the failover by server count (no two-server ping-pong)."""
    import inspect
    from message_processor.handlers import text as text_mod
    src = inspect.getsource(text_mod)
    assert "already_excluded | {failed_mcp_server}" in src
    assert "failed_mcp_server in already_excluded or len(already_excluded) >= total_servers" in src
    assert "failed_mcp_server=failed_mcp_servers" in src


def _mcp_424(label):
    e = Exception(f"Error code: 424 - Error retrieving tool list from MCP server: '{label}'")
    e.status_code = 424
    e.body = {"error": {"message": f"Error retrieving tool list from MCP server: '{label}'"}}
    return e


def _nonstreaming_mcp_host(loop_side_effect):
    """The REAL non-streaming `_handle_text_response` with the real MCP failover helpers."""
    from message_processor.handlers.text import TextHandlerMixin
    from tests.unit.test_context_meter import _handler_host

    host = _handler_host(loop_side_effect)
    for name in ("_recoverable_mcp_failure", "_extract_failed_mcp_server"):
        setattr(host, name, getattr(TextHandlerMixin, name).__get__(host))
    host._as_mcp_exclusion_set = TextHandlerMixin._as_mcp_exclusion_set
    host.mcp_manager.get_server_labels.return_value = ["example-mcp", "other-mcp"]
    return host


async def _nonstreaming_dm_turn(host, state, failed_mcp_server=None, retry_count=0,
                                enable_streaming=False, turn=None):
    from config import config
    from message_processor.client_contract import Message
    from message_processor.turn_runtime import TurnRuntime

    message = Message(text="hi", user_id="U1", channel_id="D1", thread_id="10.0",
                      metadata={"ts": "10.0", "username": "Dana Whitfield"})

    async def fake_config(**kw):
        return {"model": "gpt-5.6-sol", "temperature": 1.0, "max_tokens": 100,
                "enable_streaming": enable_streaming, "enable_code_interpreter": False}

    with patch.object(config, "get_thread_config_async", side_effect=fake_config):
        return await host._handle_text_response("hi", state, MagicMock(), message,
                                                thinking_id=None, turn=turn or TurnRuntime(),
                                                retry_count=retry_count,
                                                failed_mcp_server=failed_mcp_server)


async def test_nonstreaming_mcp_424_retries_once_without_that_server():
    from tests.unit.test_context_meter import _state

    ok = {"text": "answer", "tools_used": [], "local_tool_calls": []}
    host = _nonstreaming_mcp_host([_mcp_424("example-mcp"), ok])
    host._handle_streaming_text_response = AsyncMock()
    state = _state()
    # The buffered fallback of a failed stream (retry_count=1, streaming on): its MCP retry
    # must stay buffered, or a second stream failure would cycle back here forever.
    response = await _nonstreaming_dm_turn(host, state, retry_count=1, enable_streaming=True)
    assert response.content.startswith("answer")
    host._handle_streaming_text_response.assert_not_awaited()
    assert host.openai_client.create_text_response_with_tool_loop.await_count == 2
    excluded = [c.kwargs.get("exclude_mcp_server") for c in host._build_tools_array.call_args_list]
    assert excluded == [None, {"example-mcp"}]
    # The retry re-adds the turn's own message; it must not be in history twice.
    assert [m["content"] for m in state.messages if m.get("role") == "user"].count("hi") == 1


@pytest.mark.parametrize("case", ["already_excluded", "local_tool_ran"])
async def test_nonstreaming_mcp_424_for_an_excluded_server_still_surfaces(case):
    from message_processor.turn_runtime import TurnRuntime
    from tests.unit.test_context_meter import _state

    turn = TurnRuntime()
    excluded = {"example-mcp"} if case == "already_excluded" else None

    async def _loop(**_kw):
        if case == "local_tool_ran":
            # A tool with a visible effect ran before the later round's 424: no replay.
            turn.note_tool_call({"name": "edit_image", "ok": True})
        raise _mcp_424("example-mcp")

    host = _nonstreaming_mcp_host(_loop)
    with pytest.raises(Exception, match="424"):
        await _nonstreaming_dm_turn(host, _state(), failed_mcp_server=excluded, turn=turn)
    assert host.openai_client.create_text_response_with_tool_loop.await_count == 1


# ---------- 5. discovery cache ----------

def test_collect_mcp_list_tools_from_objects_and_dicts():
    sink = {}
    item = SimpleNamespace(
        type="mcp_list_tools", server_label="context7",
        tools=[SimpleNamespace(name="resolve-library-id", description="d1", input_schema={"a": 1}),
               {"name": "query-docs", "description": "d2", "input_schema": None},
               {"description": "nameless — skipped"}])
    _collect_mcp_list_tools(sink, item)
    assert list(sink) == ["context7"]
    assert [t["name"] for t in sink["context7"]] == ["resolve-library-id", "query-docs"]


def test_collect_mcp_list_tools_never_raises():
    sink = {}
    _collect_mcp_list_tools(sink, object())  # no attrs at all
    assert sink == {}


def test_cache_discovered_tools_payload_persists():
    mgr = MCPManager(db=MagicMock())
    mgr.cache_discovered_tools_payload("srv", [
        {"name": "t1", "description": "d", "input_schema": {"x": 1}},
        {"name": "t2", "description": None, "input_schema": None},
    ])
    assert mgr.db.save_mcp_tool.call_count == 2
    # schema serialized to a JSON string for the DB
    first_call = mgr.db.save_mcp_tool.call_args_list[0]
    assert isinstance(first_call.args[3], str) and json.loads(first_call.args[3]) == {"x": 1}
    assert [t["tool_name"] for t in mgr.get_cached_tools("srv")] == ["t1", "t2"]


# ---------- 3. health probe ----------

def _probe_session(get_side_effect=None, status=200):
    """Build a mock aiohttp module whose session.get returns an async CM."""
    resp = MagicMock()
    resp.status = status

    class _GetCM:
        async def __aenter__(self):
            if get_side_effect:
                raise get_side_effect
            return resp
        async def __aexit__(self, *a):
            return False

    session = MagicMock()
    session.get = MagicMock(return_value=_GetCM())

    class _SessionCM:
        async def __aenter__(self):
            return session
        async def __aexit__(self, *a):
            return False

    aiohttp_mock = MagicMock()
    aiohttp_mock.ClientTimeout = MagicMock(return_value=None)
    aiohttp_mock.ClientSession = MagicMock(return_value=_SessionCM())
    return aiohttp_mock, session


def test_health_probe_logs_reachable(tmp_path, monkeypatch):
    mgr = _manager_with_config(tmp_path, monkeypatch, {"mcpServers": {
        "s1": _server(headers={"X-API-Key": "k"}),
    }})
    aiohttp_mock, session = _probe_session(status=406)
    with patch.dict("sys.modules", {"aiohttp": aiohttp_mock}), \
         patch.object(MCPManager, "log_info") as info:
        asyncio.run(mgr.health_probe())
    assert any("reachable (HTTP 406)" in str(c) for c in info.call_args_list)
    # auth headers were attached to the probe
    assert session.get.call_args.kwargs["headers"] == {"X-API-Key": "k"}


def test_health_probe_logs_unreachable_never_raises(tmp_path, monkeypatch):
    mgr = _manager_with_config(tmp_path, monkeypatch, {"mcpServers": {"s1": _server()}})
    aiohttp_mock, _ = _probe_session(get_side_effect=OSError("connection refused"))
    with patch.dict("sys.modules", {"aiohttp": aiohttp_mock}), \
         patch.object(MCPManager, "log_warning") as warn:
        asyncio.run(mgr.health_probe())  # must not raise
    assert any("unreachable" in str(c) for c in warn.call_args_list)
    # probe never disables the server
    assert "s1" in mgr.servers


def test_health_probe_noop_without_servers():
    mgr = MCPManager(db=None)
    asyncio.run(mgr.health_probe())  # no servers, no aiohttp import needed, no raise


# ---------- live config sanity ----------

def test_live_config_parses_with_interpolated_keys(monkeypatch):
    """The real mcp_config.json (when present) must load with its ${VAR} keys resolved from env."""
    try:
        raw = open("mcp_config.json").read()
    except FileNotFoundError:
        pytest.skip("no local mcp_config.json on this clone")
    for var in sorted(set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", raw))):
        monkeypatch.setenv(var, "test-key-value")
    from config import config as bot_config
    monkeypatch.setattr(bot_config, "mcp_config_path", "mcp_config.json")
    mgr = MCPManager(db=None)
    mgr.initialize()
    enabled = {name for name, server in json.loads(raw)["mcpServers"].items()
               if server.get("enabled", True)}
    assert set(mgr.servers) == enabled
    # no literal secrets remain in the config file
    assert not re.search(r'"X-API-Key":\s*"[^$]', raw)
