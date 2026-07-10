"""Unit tests for F2 — explicit no-reply terminal-action contract.

Covers the tool-exposure gate (unprompted-only, config-off), the once-materialized
request config (shared dict never mutated), the tool-loop terminal semantics (ends the
loop, executes only sibling react, suppresses other siblings, sanitizes the reason), and
the deferred-output routing that keeps unprompted turns off the streaming path.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from config import config
from tool_registry import ToolRegistry


def _no_reply_schema():
    return {
        "type": "function",
        "name": "no_response_needed",
        "parameters": {"type": "object",
                       "properties": {"reason": {"type": "string"}},
                       "required": ["reason"]},
    }


def _registry_with_no_reply():
    reg = ToolRegistry()
    reg.register(
        _no_reply_schema(), AsyncMock(return_value={"ok": True}),
        enabled=lambda cfg: config.enable_no_reply_tool and bool(cfg.get("_unprompted_turn")))
    return reg


# --------------------------------------------------------------------- exposure gate

def test_no_reply_hidden_without_unprompted_flag():
    reg = _registry_with_no_reply()
    assert reg.schemas({}) == []
    assert reg.schemas({"_unprompted_turn": True})[0]["name"] == "no_response_needed"


def test_no_reply_hidden_when_config_off(monkeypatch):
    monkeypatch.setattr(config, "enable_no_reply_tool", False)
    reg = _registry_with_no_reply()
    assert reg.schemas({"_unprompted_turn": True}) == []


# --------------------------------------------------- _materialize_request_tools

class _MatHost:
    def __init__(self, registry):
        from message_processor.handlers.text import TextHandlerMixin
        for n in ("_materialize_request_tools", "_get_tool_registry"):
            setattr(self, n, getattr(TextHandlerMixin, n).__get__(self))
        self._client = SimpleNamespace(tool_registry=registry)


def _msg(participation=False):
    md = {"ts": "1.1"}
    if participation:
        md["participation_check"] = True
    return SimpleNamespace(metadata=md, channel_id="C1")


def test_materialize_unprompted_exposes_tool_without_mutating_shared(mock_env):
    host = _MatHost(_registry_with_no_reply())
    shared = {"model": "gpt-5"}
    registry, request_config, available = host._materialize_request_tools(
        host._client, shared, _msg(participation=True), tools_disabled=False)
    assert available is True
    assert request_config["_unprompted_turn"] is True
    assert "_unprompted_turn" not in shared  # copied, never mutated
    assert registry is not None


def test_materialize_prompted_turn_no_tool(mock_env):
    host = _MatHost(_registry_with_no_reply())
    registry, request_config, available = host._materialize_request_tools(
        host._client, {"model": "gpt-5"}, _msg(participation=False), tools_disabled=False)
    assert available is False
    assert "_unprompted_turn" not in request_config


def test_materialize_timeout_retry_drops_tool_and_paragraph(mock_env):
    host = _MatHost(_registry_with_no_reply())
    registry, request_config, available = host._materialize_request_tools(
        host._client, {"model": "gpt-5"}, _msg(participation=True), tools_disabled=True)
    # Retry disables the registry — so the tool AND the suffix paragraph fall away.
    assert registry is None
    assert available is False


# ------------------------------------------------------------- tool-loop terminal

class _LoopSelf:
    def log_info(self, *a, **k):
        pass

    log_warning = log_debug = log_error = log_info


class _FakeRegistry:
    """Records dispatched call names; returns ok for each."""
    def __init__(self, dispatched):
        self.dispatched = dispatched

    async def dispatch_all(self, ctx, calls):
        results = []
        for c in calls:
            self.dispatched.append(c.get("name"))
            results.append({"ok": True})
        return results


def _fc(name, call_id, arguments="{}"):
    return {"type": "function_call", "name": name, "call_id": call_id, "arguments": arguments}


@pytest.mark.asyncio
async def test_tool_loop_no_reply_ends_and_suppresses_siblings(monkeypatch):
    from openai_client.api import tool_loop
    scripts = [[
        _fc("no_response_needed", "1", '{"reason": "not really for me"}'),
        _fc("react_to_message", "2", '{"emoji": "eyes"}'),
        _fc("remember_fact", "3", "{}"),
    ]]

    async def fake_create(self, messages, tools, return_metadata, function_call_sink,
                          tool_choice=None, **kw):
        function_call_sink.extend(scripts.pop(0))
        return {"text": "should be suppressed", "tools_used": []}

    monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools", fake_create)
    dispatched = []
    result = await tool_loop.create_text_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=_FakeRegistry(dispatched),
        tool_context=None)

    assert result["terminal_action"] == "no_reply"
    assert result["reason"] == "not really for me"
    assert result["text"] == ""
    # Only the terminal + sibling react ran; the memory write was suppressed.
    assert set(dispatched) == {"no_response_needed", "react_to_message"}
    assert result["local_tool_calls"] == [{"name": "react_to_message", "ok": True}]


@pytest.mark.asyncio
async def test_tool_loop_no_reply_reason_sanitized(monkeypatch):
    from openai_client.api import tool_loop
    dirty = "line one\nline two\t" + "x" * 400
    scripts = [[_fc("no_response_needed", "1", '{"reason": %r}' % dirty)]]

    async def fake_create(self, messages, tools, return_metadata, function_call_sink,
                          tool_choice=None, **kw):
        function_call_sink.extend(scripts.pop(0))
        return {"text": "", "tools_used": []}

    monkeypatch.setattr(tool_loop.responses_api, "create_text_response_with_tools", fake_create)
    result = await tool_loop.create_text_response_with_tool_loop(
        _LoopSelf(), messages=[], tools=[], registry=_FakeRegistry([]), tool_context=None)
    reason = result["reason"]
    assert "\n" not in reason and "\t" not in reason
    assert len(reason) <= 300


# --------------------------------------------------------------- contract paragraph

def test_contract_paragraph_wording():
    from prompts import NO_REPLY_CONTRACT_SUFFIX
    assert NO_REPLY_CONTRACT_SUFFIX.startswith("[You joined this conversation uninvited.")
    assert "no_response_needed" in NO_REPLY_CONTRACT_SUFFIX
    assert NO_REPLY_CONTRACT_SUFFIX.endswith("over filler.]")
