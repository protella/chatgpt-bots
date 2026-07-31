"""Request-builder unification: DM byte-identity proof + channel-layout rules.

The legacy half of this suite is the proof that P1 changed nothing for existing callers. The
fixtures in tests/fixtures/legacy_request_params.json were captured by driving the SIX
pre-refactor request blocks in openai_client/api/responses.py with a mocked API client, before
`_build_request_params` existed. Every legacy scenario below re-drives the same path through the
refactored code and asserts the kwargs handed to `responses.create` are identical.

They are FROZEN. A legacy failure means a caller's request shape moved, not that the fixture is
stale — never regenerate them to make this suite pass.
"""
from __future__ import annotations

import copy
import json
import os
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from config import config

FIXTURE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "fixtures", "legacy_request_params.json")

SYSTEM_PROMPT = "You are a helpful assistant.\nBe brief."

PLAIN_MESSAGES: List[Dict[str, Any]] = [
    {"role": "user", "content": "what's the total?"},
    # Metadata riding along on our own dicts must never reach the API.
    {"role": "assistant", "content": "56,088", "ts": "1750000000.000100", "source": "slack"},
]

TOOL_MESSAGES: List[Dict[str, Any]] = [
    {"role": "user", "content": [
        {"type": "input_text", "text": "look at this"},
        {"type": "input_image", "image_url": "data:image/png;base64,AAAA", "detail": "high"},
    ]},
    {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc", "summary": []},
    {"type": "function_call", "call_id": "call_1", "name": "mount_file", "arguments": "{}"},
    {"type": "function_call_output", "call_id": "call_1", "output": "mounted"},
    {"role": "assistant", "content": "Mounting that now."},
]

TOOLS: List[Dict[str, Any]] = [
    {"type": "web_search"},
    {"type": "function", "name": "mount_file",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
]

PATHS = ("plain", "tools", "plain_stream", "tools_stream", "plain_timeout", "tools_timeout")
_TOOL_PATHS = ("tools", "tools_stream", "tools_timeout")
_PLAIN_PATHS = ("plain", "plain_stream", "plain_timeout")


# --- capture harness -------------------------------------------------------------------------

class _CapturingClient:
    """Stands in for OpenAIClient: records the kwargs each path hands to responses.create."""

    def __init__(self):
        self.captured: Dict[str, Any] = {}
        self.client = MagicMock()
        self.debug_logs: List[str] = []

    def log_debug(self, message="", **kwargs):
        self.debug_logs.append(str(message))

    def log_info(self, message="", **kwargs):
        pass

    def log_warning(self, message="", **kwargs):
        pass

    def log_error(self, message="", **kwargs):
        pass

    async def _safe_api_call(self, api_method, operation_type=None, timeout_seconds=None,
                             **params):
        self.captured = params
        return SimpleNamespace(output=[], usage=None)

    async def _safe_stream_iteration(self, stream, operation_type="streaming"):
        # One terminal event, then STOP. A mock stream that never ends is how this repo once
        # OOM-killed a machine (CLAUDE.md pitfall 6).
        yield SimpleNamespace(type="response.completed",
                              response=SimpleNamespace(usage=None))


@contextmanager
def pinned_config():
    """Freeze every config default the builders read, so fixtures are reproducible."""
    pins = {
        "gpt_model": "gpt-5.6-sol",
        "default_temperature": 0.8,
        "default_max_tokens": 32768,
        "default_top_p": 1.0,
        "default_reasoning_effort": "medium",
        "default_verbosity": "medium",
    }
    saved = {k: getattr(config, k) for k in pins}
    for k, v in pins.items():
        setattr(config, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(config, k, v)


def _scenarios() -> Dict[str, Dict[str, Any]]:
    """The legacy matrix: every path × model × effort, plus tool-loop and default variants."""
    out: Dict[str, Dict[str, Any]] = {}
    for path in PATHS:
        for model in ("gpt-5.6-sol", "gpt-5.5"):
            for effort in ("medium", "none"):
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "reasoning_effort": effort,
                    "verbosity": "low",
                    "temperature": 0.4,
                    "top_p": 0.85,
                    "max_tokens": 1234,
                    "system_prompt": SYSTEM_PROMPT,
                }
                # The plain timeout twin has never accepted a cache key — the pre-refactor
                # signature is the fixture's baseline, so it must not be passed here.
                if path != "plain_timeout":
                    kwargs["prompt_cache_key"] = "T123:C456"
                out[f"{path}__{model}__{effort}"] = {"path": path, "kwargs": kwargs}
            if path in _TOOL_PATHS:
                out[f"{path}__{model}__loop"] = {"path": path, "kwargs": {
                    "model": model,
                    "reasoning_effort": "high",
                    "system_prompt": SYSTEM_PROMPT,
                    "prompt_cache_key": "T123:C456",
                    "function_call_sink": [],
                    "tool_choice": "none",
                }}
    for path in ("plain", "tools"):
        out[f"{path}__no_system_prompt"] = {"path": path, "kwargs": {"model": "gpt-5.6-sol"}}
        out[f"{path}__defaults"] = {"path": path, "kwargs": {}}
    return out


SCENARIOS = _scenarios()


async def drive(path: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Run one request path against the capturing client and return the captured kwargs."""
    from openai_client.api import responses as R

    fake = _CapturingClient()
    messages = copy.deepcopy(TOOL_MESSAGES if path in _TOOL_PATHS else PLAIN_MESSAGES)

    async def _cb(_chunk):
        return None

    if path == "plain":
        await R.create_text_response(fake, messages=messages, **kwargs)
    elif path == "tools":
        await R.create_text_response_with_tools(fake, messages=messages, tools=TOOLS, **kwargs)
    elif path == "plain_stream":
        await R.create_streaming_response(fake, messages=messages, stream_callback=_cb, **kwargs)
    elif path == "tools_stream":
        await R.create_streaming_response_with_tools(
            fake, messages=messages, tools=TOOLS, stream_callback=_cb, **kwargs)
    elif path == "plain_timeout":
        await R._create_text_response_with_timeout(fake, messages=messages, **kwargs)
    elif path == "tools_timeout":
        await R._create_text_response_with_tools_with_timeout(
            fake, messages=messages, tools=TOOLS, **kwargs)
    else:
        raise AssertionError(f"unknown path {path}")
    return fake.captured


def canonical(params: Dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, indent=2, default=str)


def load_fixtures() -> Dict[str, Any]:
    with open(FIXTURE_PATH, encoding="utf-8") as fh:
        return json.load(fh)["scenarios"]


# --- legacy byte identity --------------------------------------------------------------------

@pytest.mark.critical
class TestLegacyByteIdentity:
    def test_every_scenario_has_a_frozen_fixture(self):
        assert set(load_fixtures()) == set(SCENARIOS)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", sorted(SCENARIOS))
    async def test_request_matches_prerefactor_capture(self, name):
        fixtures = load_fixtures()
        scenario = SCENARIOS[name]
        with pinned_config():
            captured = await drive(scenario["path"], dict(scenario["kwargs"]))
        assert canonical(captured) == canonical(fixtures[name])

    @pytest.mark.asyncio
    async def test_plain_timeout_twin_keeps_its_missing_cache_params(self):
        """The legacy twin sets NO cache params — not even for 5.5, whose retention every other
        path sets. That gap is load-bearing for byte identity; the fix is channel-layout only."""
        with pinned_config():
            captured = await drive("plain_timeout", {
                "model": "gpt-5.5", "prompt_cache_key": "T123:C456"})
        assert "prompt_cache_retention" not in captured
        assert "prompt_cache_key" not in captured

    @pytest.mark.asyncio
    async def test_legacy_plain_prepends_developer_item_and_sets_no_instructions(self):
        with pinned_config():
            captured = await drive("plain", {"system_prompt": SYSTEM_PROMPT})
        assert captured["input"][0] == {"role": "developer", "content": SYSTEM_PROMPT}
        assert "instructions" not in captured
        assert all(set(item) == {"role", "content"} for item in captured["input"])

    @pytest.mark.asyncio
    async def test_legacy_tools_promotes_system_prompt_to_instructions(self):
        with pinned_config():
            captured = await drive("tools", {"system_prompt": SYSTEM_PROMPT})
        assert captured["instructions"] == SYSTEM_PROMPT
        assert captured["input"][0]["role"] == "user"


# --- the builder itself ----------------------------------------------------------------------

def build(**kwargs):
    from openai_client.base import _build_request_params

    params = {"model": "gpt-5.6-sol", "input_items": [], "system_prompt": SYSTEM_PROMPT}
    params.update(kwargs)
    with pinned_config():
        return _build_request_params(**params)


class TestChannelLayout:
    def test_instructions_on_every_channel_path(self):
        for legacy_kind in ("plain", "tools"):
            params = build(layout="channel", legacy_kind=legacy_kind,
                           input_items=[{"role": "user", "content": "hi"}])
            assert params["instructions"] == SYSTEM_PROMPT
            assert params["input"] == [{"role": "user", "content": "hi"}]

    def test_typed_item_allowlist_passthrough(self):
        items = [
            {"type": "function_call", "call_id": "c1", "name": "n", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc"},
        ]
        params = build(layout="channel", input_items=copy.deepcopy(items))
        assert params["input"] == items

    def test_unsupported_typed_item_is_dropped(self):
        params = build(layout="channel", input_items=[
            {"type": "item_reference", "id": "msg_1"},
            {"role": "user", "content": "kept"},
        ])
        assert params["input"] == [{"role": "user", "content": "kept"}]

    def test_streaming_tool_round_assistant_string_replays(self):
        """The streaming tool loop appends {"role": "assistant", "content": <str>} for the
        pre-tool preamble. A parts-only allowlist would silently drop the bot's own words."""
        items = [
            {"role": "user", "content": [{"type": "input_text", "text": "make a chart"}]},
            {"role": "assistant", "content": "Making that now."},
            {"type": "function_call", "call_id": "c1", "name": "create_image_asset",
             "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "done"},
        ]
        params = build(layout="channel", input_items=copy.deepcopy(items))
        assert params["input"] == items

    def test_message_metadata_is_stripped(self):
        params = build(layout="channel", input_items=[
            {"role": "user", "content": "hi", "ts": "1.1", "source": "slack"}])
        assert params["input"] == [{"role": "user", "content": "hi"}]

    def test_foreign_content_parts_are_dropped(self):
        params = build(layout="channel", input_items=[{"role": "user", "content": [
            {"type": "input_text", "text": "keep"},
            {"type": "output_text", "text": "drop"},
            {"type": "input_file", "filename": "a.pdf", "file_data": "data:..."},
            {"type": "input_image", "image_url": "data:image/png;base64,AA"},
        ]}])
        assert [p["type"] for p in params["input"][0]["content"]] == [
            "input_text", "input_file", "input_image"]

    def test_message_with_no_surviving_parts_is_dropped(self):
        params = build(layout="channel", input_items=[
            {"role": "user", "content": [{"type": "output_text", "text": "drop"}]},
            {"role": "user", "content": "kept"},
        ])
        assert params["input"] == [{"role": "user", "content": "kept"}]

    def test_roleless_and_non_dict_items_are_dropped(self):
        params = build(layout="channel", input_items=[
            {"content": "no role"}, "raw string", {"role": "user", "content": 17},
            {"role": "user", "content": "kept"},
        ])
        assert params["input"] == [{"role": "user", "content": "kept"}]

    def test_no_developer_item_is_prepended(self):
        params = build(layout="channel", input_items=[{"role": "user", "content": "hi"}])
        assert all(item.get("role") != "developer" for item in params["input"])


class TestContentPartKeys:
    """Pitfall 5: our part dicts do double duty (API part AND DB metadata). Checking the part
    TYPE and forwarding the dict lets `source` / `url` / Slack's `file_id` through, and each of
    those is a 400 for the whole turn — so the keys are picked off, not the parts waved past."""

    def _dirty(self):
        return [{"role": "user", "content": [
            {"type": "input_text", "text": "look",
             "prompt_cache_breakpoint": {"mode": "explicit"},
             "ts": "1.1", "source": "slack"},
            {"type": "input_image", "image_url": "data:image/png;base64,AA", "detail": "high",
             "source": "upload", "filename": "shot.png", "url": "https://files.slack/x",
             "original_url": "https://files.slack/x", "file_id": "F0BGSHE3JGJ"},
            {"type": "input_file", "filename": "a.pdf", "file_data": "data:...",
             "source": "upload", "url": "https://files.slack/y", "file_id": "F0BGSHE3JGK",
             "mimetype": "application/pdf"},
        ]}]

    def test_dirty_keys_are_stripped_from_every_part_type(self):
        params = build(model="gpt-5.6-sol", layout="channel", input_items=self._dirty())
        text, image, file_part = params["input"][0]["content"]
        assert text == {"type": "input_text", "text": "look",
                        "prompt_cache_breakpoint": {"mode": "explicit"}}
        assert image == {"type": "input_image", "image_url": "data:image/png;base64,AA",
                         "detail": "high"}
        assert file_part == {"type": "input_file", "filename": "a.pdf", "file_data": "data:..."}

    def test_slack_file_id_never_reaches_the_api(self):
        """The API's `file_id` names an OPENAI file; ours is Slack's, and it earns its own 400."""
        params = build(model="gpt-5.6-sol", layout="channel", input_items=self._dirty())
        assert all("file_id" not in p for p in params["input"][0]["content"])

    def test_the_breakpoint_survives_the_strip(self):
        params = build(model="gpt-5.6-sol", layout="channel", input_items=self._dirty())
        assert params["input"][0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}

    def test_stripping_a_dirty_part_never_mutates_the_caller(self):
        items = self._dirty()
        before = copy.deepcopy(items)
        for model in ("gpt-5.5", "gpt-5.6-sol"):
            build(model=model, layout="channel", input_items=items)
        assert items == before

    def test_none_valued_keys_are_dropped(self):
        params = build(model="gpt-5.6-sol", layout="channel", input_items=[
            {"role": "user", "content": [
                {"type": "input_image", "image_url": "data:image/png;base64,AA",
                 "detail": None, "file_id": None}]}])
        assert params["input"][0]["content"] == [
            {"type": "input_image", "image_url": "data:image/png;base64,AA"}]


class TestCacheBreakpoints:
    def _marked_items(self):
        return [{"role": "user", "content": [
            {"type": "input_text", "text": "end of stream",
             "prompt_cache_breakpoint": {"mode": "explicit"}},
            {"type": "input_text", "text": "plain"},
        ]}]

    def test_breakpoints_survive_on_56(self):
        params = build(model="gpt-5.6-sol", layout="channel", input_items=self._marked_items())
        assert params["input"][0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}

    def test_breakpoints_stripped_off_56(self):
        params = build(model="gpt-5.5", layout="channel", input_items=self._marked_items())
        parts = params["input"][0]["content"]
        assert "prompt_cache_breakpoint" not in parts[0]
        assert parts[0]["text"] == "end of stream"
        assert parts[1] == {"type": "input_text", "text": "plain"}

    def test_stripping_is_copy_on_write(self):
        """A 5.5 retry must not strip a marker out of a pinned input that a 5.6 retry reuses."""
        items = self._marked_items()
        before = copy.deepcopy(items)
        build(model="gpt-5.5", layout="channel", input_items=items)
        assert items == before

    def test_builder_never_mutates_caller_input(self):
        typed = copy.deepcopy(TOOL_MESSAGES) + self._marked_items()
        plain = copy.deepcopy(PLAIN_MESSAGES)
        before_typed, before_plain = copy.deepcopy(typed), copy.deepcopy(plain)
        for model in ("gpt-5.5", "gpt-5.6-sol"):
            build(model=model, layout="channel", input_items=typed)
            build(model=model, layout="legacy", legacy_kind="tools", input_items=typed)
            build(model=model, layout="legacy", legacy_kind="plain", input_items=plain)
        assert typed == before_typed
        assert plain == before_plain

    def test_attach_cache_breakpoint_marks_on_56(self):
        from openai_client.base import attach_cache_breakpoint

        part = {"type": "input_text", "text": "marker"}
        marked = attach_cache_breakpoint(part, "gpt-5.6-terra")
        assert marked == {"type": "input_text", "text": "marker",
                          "prompt_cache_breakpoint": {"mode": "explicit"}}
        assert part == {"type": "input_text", "text": "marker"}

    def test_attach_cache_breakpoint_is_a_noop_elsewhere(self):
        from openai_client.base import attach_cache_breakpoint

        part = {"type": "input_text", "text": "marker"}
        assert attach_cache_breakpoint(part, "gpt-5.5") == part
        assert attach_cache_breakpoint(part, None) == part


class TestCachePolicy:
    def test_retention_is_55_only(self):
        assert build(model="gpt-5.5")["prompt_cache_retention"] == "24h"
        assert "prompt_cache_retention" not in build(model="gpt-5.6-sol")

    def test_cache_key_needs_a_supported_model(self):
        assert build(prompt_cache_key="k")["prompt_cache_key"] == "k"
        assert build(model="gpt-5.5", prompt_cache_key="k")["prompt_cache_key"] == "k"
        assert "prompt_cache_key" not in build(model="gpt-4.1", prompt_cache_key="k")
        assert "prompt_cache_key" not in build()

    def test_cache_options_are_56_only(self):
        options = {"mode": "explicit", "ttl": "30m"}
        assert build(prompt_cache_options=options)["prompt_cache_options"] == options
        assert "prompt_cache_options" not in build(model="gpt-5.5",
                                                  prompt_cache_options=options)

    def test_legacy_cache_suppression_is_legacy_only(self):
        suppressed = build(model="gpt-5.5", prompt_cache_key="k", legacy_cache_params=False)
        assert "prompt_cache_retention" not in suppressed
        assert "prompt_cache_key" not in suppressed
        fixed = build(model="gpt-5.5", prompt_cache_key="k", legacy_cache_params=False,
                      layout="channel")
        assert fixed["prompt_cache_retention"] == "24h"
        assert fixed["prompt_cache_key"] == "k"


class TestSamplingAndPassthrough:
    def test_temperature_forced_when_reasoning(self):
        params = build(reasoning_effort="high", temperature=0.4, top_p=0.85)
        assert params["temperature"] == 1.0
        assert "top_p" not in params

    def test_explicit_temperature_and_top_p_survive_at_effort_none(self):
        params = build(reasoning_effort="none", temperature=0.4, top_p=0.85)
        assert params["temperature"] == 0.4
        assert params["top_p"] == 0.85

    def test_effort_none_still_forces_temperature_on_other_models(self):
        params = build(model="gpt-4.1", reasoning_effort="none", temperature=0.4)
        assert params["temperature"] == 1.0
        assert "top_p" not in params

    def test_effort_is_clamped_per_model(self):
        assert build(model="gpt-5.6-sol", reasoning_effort="minimal")["reasoning"] == {
            "effort": "none"}
        assert build(model="gpt-5.5", reasoning_effort="max")["reasoning"] == {"effort": "xhigh"}

    def test_config_defaults_fill_the_gaps(self):
        params = build(model=None, temperature=None, top_p=None, max_output_tokens=None,
                       reasoning_effort=None, verbosity=None)
        assert params["model"] == "gpt-5.6-sol"
        assert params["max_output_tokens"] == 32768
        assert params["text"] == {"verbosity": "medium"}
        assert params["reasoning"] == {"effort": "medium"}

    def test_optional_request_fields_are_omitted_when_unset(self):
        params = build()
        for key in ("tools", "tool_choice", "include", "parallel_tool_calls", "stream"):
            assert key not in params

    def test_optional_request_fields_pass_through(self):
        params = build(tools=TOOLS, tool_choice="none", parallel_tool_calls=True,
                       include=["reasoning.encrypted_content"], stream=True, store=True)
        assert params["tools"] == TOOLS
        assert params["tool_choice"] == "none"
        assert params["parallel_tool_calls"] is True
        assert params["include"] == ["reasoning.encrypted_content"]
        assert params["stream"] is True
        assert params["store"] is True

    def test_legacy_tools_kind_always_sends_a_tools_key(self):
        assert build(legacy_kind="tools")["tools"] is None

    def test_system_prompt_absent_means_no_instructions_and_no_developer_item(self):
        params = build(system_prompt=None, layout="channel",
                       input_items=[{"role": "user", "content": "hi"}])
        assert "instructions" not in params
        assert params["input"] == [{"role": "user", "content": "hi"}]
