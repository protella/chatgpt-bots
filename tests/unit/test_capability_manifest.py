"""F11 — the capability manifest, now with nothing to feed.

`render_capabilities_line` is a pure function of already-loaded config + mcp_manager, and its whole
composition contract is still tested below: web-search flag, MCP descriptions, label fallback,
insertion order, determinism, the never-empty guard. That part is untouched and worth keeping.

WHAT IT FED IS GONE. The inventory existed so the rich gate could judge ANSWERABILITY — "could the
assistant actually supply what is being asked?" — which is a question the binary gate does not ask.
It decides whether the responder runs, and the responder knows its own tools by having them. So the
`capabilities=` parameter, the signals key, and the prompt's "The assistant's own tools/data
sources" line are all deleted, and the tests that asserted the hop are inverted into tripwires.

The function itself has zero callers now and is scheduled for deletion in the cleanup commit
(spec §8). Until then its unit tests are the only thing keeping it honest, so they stay.
"""
from __future__ import annotations

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


# ------------------------------------------- the inventory no longer reaches the gate

class _CapturingClient:
    """Records the cohort and steering the gate actually sends."""

    def __init__(self, wake=False):
        self._wake = wake
        self.sources = None
        self.steering = None

    async def classify_wake(self, *, sources, channel_steering_text=None):
        self.sources = tuple(sources)
        self.steering = channel_steering_text
        return self._wake


class TestTheInventoryIsNotAGateInput:
    @pytest.mark.asyncio
    async def test_evaluate_will_not_accept_a_capability_line(self, monkeypatch):
        """INVERTED from "capabilities are copied into the signals dict".

        Asserted as a TypeError rather than as an absent signal, because a silently-swallowed kwarg
        would let a caller believe the gate had been told what the bot can do."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        engine = ParticipationEngine(_CapturingClient())
        with pytest.raises(TypeError):
            await engine.evaluate(channel_id="C1", ts="1.0", text="hi",
                                  capabilities=render_capabilities_line(None))

    @pytest.mark.asyncio
    async def test_the_classifier_gets_the_message_and_the_steering_and_nothing_else(
            self, monkeypatch):
        """The positive half: what the gate DOES send. Two arguments, so there is no signals dict
        for an inventory (or a people line, or a pulse) to be added back into."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        client = _CapturingClient()
        await ParticipationEngine(client).evaluate(
            channel_id="C1", ts="1.0", text="anyone know gen z ice cream trends?")
        assert [s.text for s in client.sources] == ["anyone know gen z ice cream trends?"]
        assert client.steering is None

    def test_the_inventory_has_no_runtime_caller_left(self):
        """The other half of the deletion: nothing builds the line for the gate any more.

        `render_capabilities_line` survives as a function with no callers (cleanup commit 7), so the
        check is over the modules that used to call it — main.py built it on the gate's hot path,
        which is where half a dozen API reads and cache lookups went to support a judgment the gate
        stopped making."""
        import inspect
        import main
        from message_processor import participation
        from openai_client.api import responses

        assert "render_capabilities_line" not in inspect.getsource(main)
        assert "capabilities" not in inspect.getsource(participation.ParticipationEngine.evaluate)
        assert "capabilities" not in inspect.getsource(responses.classify_wake)


class TestPromptRule:
    def test_the_answerability_rule_went_with_the_stage_that_asked_it(self):
        """The rule this file's last test asserted ("a question genuinely open to the channel at
        large... judged against the tools as they are described to it") was Stage 3 of the rich
        prompt — the value floor, which is exactly what the inventory was FOR. The binary prompt
        does not weigh whether an answer is available; when it is unsure it wakes, and the responder
        (which has the tools rather than a description of them) decides.

        Asserted as an absence so the floor cannot creep back in without its inputs."""
        from prompts import WAKE_CLASSIFIER_SYSTEM_PROMPT as p
        for retired in ("tools as they are described to it",
                        "genuinely open to the channel at large", "own tools/data sources"):
            assert retired not in p, retired
        # What replaced it, in one line: generosity, because the responder can still say nothing.
        assert "When you are unsure, wake it" in p


# ------------------------------------------------------ F14b attachment signals

class TestF14bCapabilityEntry:
    def test_analyzing_entry_unconditional(self, monkeypatch):
        # Present regardless of web-search / MCP config — vision & document flows are core.
        monkeypatch.setattr(config, "enable_web_search", False, raising=False)
        monkeypatch.setattr(config, "mcp_enabled_default", False, raising=False)
        assert "analyzing images and documents shared in chat" in render_capabilities_line(None)


class TestF14bAttachmentsAreDescriptorsOnTheSourceRecord:
    @pytest.mark.asyncio
    async def test_attachments_ride_the_source_record_as_name_and_type(self, monkeypatch):
        """RE-BASELINED in shape, not in substance: the gate still learns that something was
        attached and what it is called.

        It used to arrive as one prose sentence assembled in the event handler ("1 image
        (food.png)") and pasted into the prompt — an event handler writing part of a prompt. It is
        now a tuple of "name (kind)" descriptors on the typed SourceMessage, and the renderer
        renders it."""
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        client = _CapturingClient(wake=True)
        await ParticipationEngine(client).evaluate(
            channel_id="C1", ts="1.0", text="what do we think?",
            attachments=["food.png (image)", "brief.pdf (file)"])
        assert client.sources[0].attachments == ("food.png (image)", "brief.pdf (file)")

    @pytest.mark.asyncio
    async def test_attachments_default_to_an_empty_tuple(self, monkeypatch):
        # Empty tuple rather than None: callers iterate unconditionally, and "no files" is not a
        # missing value.
        monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
        client = _CapturingClient(wake=True)
        await ParticipationEngine(client).evaluate(channel_id="C1", ts="1.0", text="hi")
        assert client.sources[0].attachments == ()


class TestF14bTheRenderedLineIsStillHonest:
    def test_the_gate_is_told_it_cannot_see_the_contents(self):
        """F40's honesty fix, carried across the rewrite.

        The old line claimed "The assistant can view and analyze attachments" — a fact about the
        ANSWERING model, fed unconditionally to a classifier that could see nothing, and the model
        took it as licence to opine on a picture it had never seen (the :dogkek: reaction). The
        binary gate never looks at images at all, so the rendered block says so outright."""
        from message_processor.participation import SourceMessage
        from openai_client.api.responses import _render_wake_source

        block = _render_wake_source(SourceMessage(
            ts="1.0", text="what do we think? good marketing material?", sender_name="Peter",
            sender_type="human", attachments=("food.png (image)",)), index=0, total=1)
        assert "food.png (image)" in block
        assert "contents not shown to you" in block
        assert "can view and analyze attachments" not in block

    def test_no_attachment_line_when_there_are_no_files(self):
        from message_processor.participation import SourceMessage
        from openai_client.api.responses import _render_wake_source

        block = _render_wake_source(SourceMessage(
            ts="1.0", text="msg", sender_name="Peter", sender_type="human"), index=0, total=1)
        assert "Attached" not in block
