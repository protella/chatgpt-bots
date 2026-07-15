"""F11 — capability manifest for the participation classifier.

Covers `render_capabilities_line` (pure composition from config + mcp_manager),
its forwarding through `ParticipationEngine.evaluate()` into the signals dict, the
`classify_participation` payload carrying (or omitting) the line at its fixed
position after the alias identity line, and the new prompt judgment rule.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from config import config
from message_processor.participation import (ParticipationEngine,
                                             render_capabilities_line)


class _FakeMCP:
    """Minimal stand-in for MCPManager: an insertion-ordered `servers` dict and
    `has_mcp_servers()`."""

    def __init__(self, servers):
        self.servers = servers

    def has_mcp_servers(self):
        return len(self.servers) > 0


# ------------------------------------------------------------ render_capabilities_line

class TestRenderCapabilitiesLine:
    def test_web_search_flag_on(self, monkeypatch):
        monkeypatch.setattr(config, "enable_web_search", True, raising=False)
        monkeypatch.setattr(config, "mcp_enabled_default", False, raising=False)
        line = render_capabilities_line(None)
        assert "web search" in line
        assert "image generation and editing" in line

    def test_web_search_flag_off(self, monkeypatch):
        monkeypatch.setattr(config, "enable_web_search", False, raising=False)
        monkeypatch.setattr(config, "mcp_enabled_default", False, raising=False)
        line = render_capabilities_line(None)
        assert "web search" not in line
        assert line == ("image generation and editing; "
                        "analyzing images and documents shared in chat")

    def test_mcp_servers_present_render_descriptions(self, monkeypatch):
        monkeypatch.setattr(config, "enable_web_search", True, raising=False)
        monkeypatch.setattr(config, "mcp_enabled_default", True, raising=False)
        mcp = _FakeMCP({
            "datassential": {"server_description": "Food & beverage market data"},
            "weather": {"server_description": "Live weather lookups"},
        })
        line = render_capabilities_line(mcp)
        assert "Food & beverage market data" in line
        assert "Live weather lookups" in line
        # semicolon-joined
        assert "; " in line

    def test_mcp_absent_when_manager_none(self, monkeypatch):
        monkeypatch.setattr(config, "enable_web_search", True, raising=False)
        monkeypatch.setattr(config, "mcp_enabled_default", True, raising=False)
        line = render_capabilities_line(None)
        assert line == ("web search; image generation and editing; "
                        "analyzing images and documents shared in chat")

    def test_mcp_omitted_when_default_off(self, monkeypatch):
        monkeypatch.setattr(config, "enable_web_search", False, raising=False)
        monkeypatch.setattr(config, "mcp_enabled_default", False, raising=False)
        mcp = _FakeMCP({"datassential": {"server_description": "Food & bev data"}})
        line = render_capabilities_line(mcp)
        assert "Food & bev data" not in line
        assert line == ("image generation and editing; "
                        "analyzing images and documents shared in chat")

    def test_mcp_omitted_when_no_servers(self, monkeypatch):
        monkeypatch.setattr(config, "enable_web_search", False, raising=False)
        monkeypatch.setattr(config, "mcp_enabled_default", True, raising=False)
        line = render_capabilities_line(_FakeMCP({}))
        assert line == ("image generation and editing; "
                        "analyzing images and documents shared in chat")

    def test_description_falls_back_to_label(self, monkeypatch):
        monkeypatch.setattr(config, "enable_web_search", False, raising=False)
        monkeypatch.setattr(config, "mcp_enabled_default", True, raising=False)
        # No server_description → the label is used.
        mcp = _FakeMCP({"my_server": {"server_url": "https://x"}})
        line = render_capabilities_line(mcp)
        assert "my_server" in line

    def test_insertion_order_preserved(self, monkeypatch):
        monkeypatch.setattr(config, "enable_web_search", False, raising=False)
        monkeypatch.setattr(config, "mcp_enabled_default", True, raising=False)
        mcp = _FakeMCP({"alpha": {}, "beta": {}, "gamma": {}})
        line = render_capabilities_line(mcp)
        assert line.index("alpha") < line.index("beta") < line.index("gamma")

    def test_deterministic_across_calls(self, monkeypatch):
        monkeypatch.setattr(config, "enable_web_search", True, raising=False)
        monkeypatch.setattr(config, "mcp_enabled_default", True, raising=False)
        mcp = _FakeMCP({"a": {"server_description": "aaa"}, "b": {"server_description": "bbb"}})
        assert render_capabilities_line(mcp) == render_capabilities_line(mcp)

    def test_never_empty_string_image_gen_unconditional(self, monkeypatch):
        # Everything off, no MCP → still non-None because image gen is unconditional;
        # the None-when-empty guard is defensive (the list is never actually empty).
        monkeypatch.setattr(config, "enable_web_search", False, raising=False)
        monkeypatch.setattr(config, "mcp_enabled_default", False, raising=False)
        line = render_capabilities_line(None)
        assert line and isinstance(line, str)

    def test_none_when_caps_empty_guard(self, monkeypatch):
        # Directly exercise the guard: an mcp_manager whose has_mcp_servers raises is
        # treated as absent, and with everything else off only image gen remains — the
        # guard's None branch is unreachable while image gen is unconditional, so this
        # asserts the guard never yields an empty string.
        monkeypatch.setattr(config, "enable_web_search", False, raising=False)
        monkeypatch.setattr(config, "mcp_enabled_default", True, raising=False)

        class _Boom:
            servers = {"x": {}}

            def has_mcp_servers(self):
                raise RuntimeError("boom")

        line = render_capabilities_line(_Boom())
        assert line == ("image generation and editing; "
                        "analyzing images and documents shared in chat")


# ------------------------------------------------------- evaluate() forwards capabilities

class _CapturingClient:
    def __init__(self, verdict=None):
        self._verdict = verdict or {"action": "ignore"}
        self.captured = None

    async def classify_participation(self, text, signals=None):
        self.captured = signals
        return self._verdict


class TestEvaluateForwardsCapabilities:
    @pytest.mark.asyncio
    async def test_capabilities_copied_into_signals(self, monkeypatch):
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        client = _CapturingClient()
        engine = ParticipationEngine(client)
        await engine.evaluate(channel_id="C1", ts="1.0", text="hi",
                              capabilities="web search; image generation and editing")
        assert client.captured["capabilities"] == "web search; image generation and editing"

    @pytest.mark.asyncio
    async def test_capabilities_defaults_none(self, monkeypatch):
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        client = _CapturingClient()
        engine = ParticipationEngine(client)
        await engine.evaluate(channel_id="C1", ts="1.0", text="hi")
        assert client.captured["capabilities"] is None


# ------------------------------------------------- classify_participation payload

class _FakeContent:
    def __init__(self, text):
        self.text = text


class _FakeItem:
    def __init__(self, text):
        self.content = [_FakeContent(text)]


class _FakeResp:
    def __init__(self, text):
        self.output = [_FakeItem(text)]


class _FakeLLM:
    def __init__(self, text='{"action": "ignore"}'):
        self._text = text
        self.client = MagicMock()
        self.captured_input = None

    async def _safe_api_call(self, *a, **k):
        self.captured_input = k.get("input")
        return _FakeResp(self._text)

    def log_debug(self, *a, **k):
        pass

    def log_warning(self, *a, **k):
        pass


class TestClassifyParticipationPayload:
    @pytest.mark.asyncio
    async def test_capabilities_line_present_when_set(self):
        from openai_client.api.responses import classify_participation
        llm = _FakeLLM()
        await classify_participation(
            llm, "anyone know gen z ice cream trends?",
            signals={"capabilities": "web search; menu & flavor trend data"})
        prompt = llm.captured_input[1]["content"]
        assert "The assistant's own tools/data sources" in prompt
        assert "menu & flavor trend data" in prompt

    @pytest.mark.asyncio
    async def test_capabilities_line_omitted_when_absent(self):
        from openai_client.api.responses import classify_participation
        llm = _FakeLLM()
        await classify_participation(llm, "msg", signals={})
        assert "tools/data sources" not in llm.captured_input[1]["content"]

    @pytest.mark.asyncio
    async def test_capabilities_line_follows_alias_line(self, monkeypatch):
        # Fixed position: immediately after the alias identity line, before the sender.
        monkeypatch.setattr(config, "bot_name_aliases", ["chatgpt"], raising=False)
        from openai_client.api.responses import classify_participation
        llm = _FakeLLM()
        await classify_participation(
            llm, "msg",
            signals={"capabilities": "image generation and editing",
                     "sender_name": "Peter"})
        prompt = llm.captured_input[1]["content"]
        alias_pos = prompt.index("The assistant's name in this workspace")
        cap_pos = prompt.index("The assistant's own tools/data sources")
        sender_pos = prompt.index("Sender: Peter")
        assert alias_pos < cap_pos < sender_pos


class TestPromptRule:
    def test_open_question_rule_present(self):
        from prompts import PARTICIPATION_SYSTEM_PROMPT
        assert "OPEN question to the room" in PARTICIPATION_SYSTEM_PROMPT
        assert "tools/data sources" in PARTICIPATION_SYSTEM_PROMPT


# ------------------------------------------------------ F14b attachment signals

class TestF14bCapabilityEntry:
    def test_analyzing_entry_unconditional(self, monkeypatch):
        # Present regardless of web-search / MCP config — vision & document flows are core.
        monkeypatch.setattr(config, "enable_web_search", False, raising=False)
        monkeypatch.setattr(config, "mcp_enabled_default", False, raising=False)
        assert "analyzing images and documents shared in chat" in render_capabilities_line(None)


class TestF14bEvaluateForwardsAttachments:
    @pytest.mark.asyncio
    async def test_attachments_copied_into_signals(self, monkeypatch):
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        client = _CapturingClient()
        engine = ParticipationEngine(client)
        await engine.evaluate(channel_id="C1", ts="1.0", text="what do we think?",
                              attachments="1 image (food.png)")
        assert client.captured["attachments"] == "1 image (food.png)"

    @pytest.mark.asyncio
    async def test_attachments_defaults_none(self, monkeypatch):
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        client = _CapturingClient()
        engine = ParticipationEngine(client)
        await engine.evaluate(channel_id="C1", ts="1.0", text="hi")
        assert client.captured["attachments"] is None


class TestF14bClassifyPayload:
    @pytest.mark.asyncio
    async def test_attachment_line_present_when_set(self):
        from openai_client.api.responses import classify_participation
        llm = _FakeLLM()
        await classify_participation(
            llm, "what do we think? good marketing material?",
            signals={"sender_name": "Peter", "attachments": "1 image (food.png)"})
        prompt = llm.captured_input[1]["content"]
        assert "Attached to the message: 1 image (food.png)." in prompt
        # F40: the old line here said "The assistant can view and analyze attachments" — a claim
        # about the ANSWERING model, fed unconditionally to a classifier that could see nothing.
        # The model took it as licence to opine on a picture it had never seen (the :dogkek:
        # reaction). With no image_status signal, the honest line is: you see the name, not the
        # contents — but the assistant can open it if it decides to respond.
        assert "Only the filename and type" in prompt
        assert "can view and analyze attachments" not in prompt

    @pytest.mark.asyncio
    async def test_attachment_line_omitted_when_absent(self):
        from openai_client.api.responses import classify_participation
        llm = _FakeLLM()
        await classify_participation(llm, "msg", signals={"sender_name": "Peter"})
        assert "Attached to the message" not in llm.captured_input[1]["content"]
