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
import logging
import os
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from config import config

FIXTURE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "fixtures", "legacy_request_params.json")

_capture_log = logging.getLogger("tests.capturing_client")

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
    """Stands in for OpenAIClient: records the kwargs each path hands to responses.create.

    ``served_tier`` is what the fake API echoes back — the `service_tier` on the created and
    completed response objects. None (the default) means the field is absent entirely, which is
    what a standard-tier response looks like on the wire.
    """

    def __init__(self, served_tier: Optional[str] = None):
        self.captured: Dict[str, Any] = {}
        self.client = MagicMock()
        self.debug_logs: List[str] = []
        self.served_tier = served_tier

    def _served(self, **fields: Any) -> SimpleNamespace:
        """A response object shaped like the API's: `service_tier` present only when served."""
        if self.served_tier is not None:
            fields["service_tier"] = self.served_tier
        return SimpleNamespace(**fields)

    def log_debug(self, message="", **kwargs):
        self.debug_logs.append(str(message))

    def log_info(self, message="", **kwargs):
        # Forwarded to a REAL logger rather than discarded: the service-tier echo is only ever
        # observable as a log line, so a fake that swallows log_info would let the echo break
        # while every test stayed green.
        _capture_log.info(str(message))

    def log_warning(self, message="", **kwargs):
        _capture_log.warning(str(message))

    def log_error(self, message="", **kwargs):
        _capture_log.error(str(message))

    async def _safe_api_call(self, api_method, operation_type=None, timeout_seconds=None,
                             **params):
        self.captured = params
        return self._served(output=[], usage=None)

    async def _safe_stream_iteration(self, stream, operation_type="streaming"):
        # Created then terminal, then STOP. A mock stream that never ends is how this repo once
        # OOM-killed a machine (CLAUDE.md pitfall 6).
        yield SimpleNamespace(type="response.created", response=self._served(usage=None))
        yield SimpleNamespace(type="response.completed", response=self._served(usage=None))


@contextmanager
def pinned_config(service_tier: str = "standard"):
    """Freeze every config default the builders read, so fixtures are reproducible.

    ``service_tier`` is a pin like the rest — the frozen legacy fixtures must not depend on
    whether the machine running them has OPENAI_SERVICE_TIER set.
    """
    pins = {
        "gpt_model": "gpt-5.6-sol",
        "default_temperature": 0.8,
        "default_max_tokens": 32768,
        "default_top_p": 1.0,
        "default_reasoning_effort": "medium",
        "default_verbosity": "medium",
        # The fast tier is a request-shape input like any other: an operator running with
        # OPENAI_SERVICE_TIER=fast must not make the frozen legacy fixtures fail.
        "openai_service_tier": service_tier,
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


async def drive(path: str, kwargs: Dict[str, Any],
                served_tier: Optional[str] = None) -> Dict[str, Any]:
    """Run one request path against the capturing client and return the captured kwargs."""
    from openai_client.api import responses as R

    fake = _CapturingClient(served_tier)
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

def build(service_tier: str = "standard", **kwargs: Any) -> Dict[str, Any]:
    from openai_client.base import _build_request_params

    params: Dict[str, Any] = {"model": "gpt-5.6-sol", "input_items": [],
                              "system_prompt": SYSTEM_PROMPT}
    params.update(kwargs)
    with pinned_config(service_tier):
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


# --- W2: the fast service tier ----------------------------------------------------------------

class TestServiceTierAttachment:
    """Three legs, all required. The builder serves reconsideration, background and research
    calls as well as the user's turn, so config alone must never be enough to buy the fast tier
    — and `standard` means the parameter is ABSENT, not sent with a "standard" value the API
    would reject.
    """

    def test_eligible_and_fast_and_sol_attaches(self):
        params = build(service_tier="fast", model="gpt-5.6-sol", service_tier_eligible=True)
        assert params["service_tier"] == "fast"

    def test_not_eligible_omits_it(self):
        """Reconsideration, background and research calls come through here without the flag."""
        params = build(service_tier="fast", model="gpt-5.6-sol", service_tier_eligible=False)
        assert "service_tier" not in params

    def test_the_flag_defaults_to_off(self):
        """A caller that says nothing buys nothing — the default is what protects every path
        that was never told about the fast tier."""
        params = build(service_tier="fast", model="gpt-5.6-sol")
        assert "service_tier" not in params

    def test_standard_config_omits_it_even_when_eligible(self):
        params = build(service_tier="standard", model="gpt-5.6-sol",
                       service_tier_eligible=True)
        assert "service_tier" not in params

    @pytest.mark.parametrize("model", ["gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5"])
    def test_another_model_omits_it(self, model):
        """Only Sol honors the fast pool; anywhere else the 2x price buys nothing."""
        params = build(service_tier="fast", model=model, service_tier_eligible=True)
        assert "service_tier" not in params

    def test_the_resolved_model_decides_not_the_argument(self):
        """model=None resolves to config.gpt_model, and THAT is what has to be judged."""
        params = build(service_tier="fast", model=None, service_tier_eligible=True)
        assert params["model"] == "gpt-5.6-sol"
        assert params["service_tier"] == "fast"

    def test_the_literal_standard_is_never_sent(self):
        """The API's own default is "default" and it rejects "standard" — the only two request
        shapes this builder may produce are `service_tier: "fast"` and no key at all."""
        for tier in ("standard", "fast"):
            for eligible in (True, False):
                params = build(service_tier=tier, model="gpt-5.6-sol",
                               service_tier_eligible=eligible)
                assert params.get("service_tier") in (None, "fast")


@pytest.mark.asyncio
class TestServiceTierThroughTheCallPaths:
    """The flag has to survive the whole wrapper chain, and the calls that never set it have to
    stay byte-identical to what they send today."""

    @pytest.mark.parametrize("path", PATHS)
    async def test_an_eligible_path_sends_fast(self, path):
        with pinned_config("fast"):
            captured = await drive(path, {"model": "gpt-5.6-sol",
                                          "service_tier_eligible": True})
        assert captured["service_tier"] == "fast"

    @pytest.mark.parametrize("path", PATHS)
    async def test_the_same_path_sends_nothing_without_the_flag(self, path):
        with pinned_config("fast"):
            captured = await drive(path, {"model": "gpt-5.6-sol"})
        assert "service_tier" not in captured

    async def test_the_reconsideration_call_never_asks_for_fast(self):
        """It shares the builder with the responder and must not be swept along: it is
        housekeeping, and it is not what the owner agreed to pay double for."""
        from openai_client.api import responses as R

        fake = _CapturingClient()
        with pinned_config("fast"):
            with pytest.raises(R.ReconsiderationDecisionError):
                await R.create_reconsideration_decision(
                    fake, input_items=[{"role": "user", "content": "still relevant?"}],
                    model="gpt-5.6-sol")
        assert "service_tier" not in fake.captured


# --- W2: the echo, and where eligibility is allowed to come from -------------------------------

@pytest.mark.asyncio
class TestServiceTierEcho:
    """The downgrade is invisible unless we log it.

    A fast request the API declines to honor comes back tagged `"default"` and costs the same
    2x — there is no error, no exception, and no other signal anywhere in the stack. These
    assert the LOG, because the log is the entire feature.
    """

    @pytest.mark.parametrize("path", PATHS)
    @pytest.mark.parametrize("served,verdict", [("priority", "honored"),
                                                ("default", "downgraded")])
    async def test_the_served_tier_is_logged_on_every_path(self, path, served, verdict, caplog):
        with caplog.at_level("INFO", logger="tests.capturing_client"):
            with pinned_config("fast"):
                await drive(path, {"model": "gpt-5.6-sol", "service_tier_eligible": True},
                            served_tier=served)
        echoes = [r.getMessage() for r in caplog.records if "[service_tier]" in r.getMessage()]
        assert len(echoes) == 1, f"expected exactly one echo line, got {echoes}"
        assert f"echoed={served!r}" in echoes[0] and verdict in echoes[0]

    @pytest.mark.parametrize("path", PATHS)
    async def test_a_standard_turn_logs_no_echo_at_all(self, path, caplog):
        """Every turn on a standard deployment goes through this code. It must say nothing."""
        with caplog.at_level("INFO", logger="tests.capturing_client"):
            with pinned_config("standard"):
                await drive(path, {"model": "gpt-5.6-sol", "service_tier_eligible": True},
                            served_tier="default")
        assert not [r for r in caplog.records if "[service_tier]" in r.getMessage()]

    @pytest.mark.parametrize("path", ("plain_stream", "tools_stream"))
    async def test_a_stream_that_never_reaches_its_terminal_still_logs(self, path, caplog):
        """The reason the echo is read at `response.created` rather than at the end: a stream
        can be interrupted, suppressed by the stale-send guard, or ended by a raising callback,
        and the tier we were billed for would go unrecorded on every one of those."""
        from openai_client.api import responses as R

        fake = _CapturingClient("default")

        async def _only_created(stream, operation_type="streaming"):
            yield SimpleNamespace(type="response.created",
                                  response=SimpleNamespace(service_tier="default", usage=None))
            raise RuntimeError("connection dropped mid-stream")

        fake._safe_stream_iteration = _only_created  # type: ignore[method-assign]  # the fake IS the seam

        async def _cb(_chunk):
            return None

        with caplog.at_level("INFO", logger="tests.capturing_client"):
            with pinned_config("fast"):
                with pytest.raises(RuntimeError):
                    if path == "plain_stream":
                        await R.create_streaming_response(
                            fake, messages=copy.deepcopy(PLAIN_MESSAGES), stream_callback=_cb,
                            model="gpt-5.6-sol", service_tier_eligible=True)
                    else:
                        await R.create_streaming_response_with_tools(
                            fake, messages=copy.deepcopy(TOOL_MESSAGES), tools=TOOLS,
                            stream_callback=_cb, model="gpt-5.6-sol",
                            service_tier_eligible=True)
        echoes = [r.getMessage() for r in caplog.records if "[service_tier]" in r.getMessage()]
        assert len(echoes) == 1 and "downgraded" in echoes[0]


# --- W2: the eligibility boundary, pinned at the real call sites -------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The three files that are ALLOWED to say `service_tier_eligible`: two carry the parameter down
# the plumbing, one is the caller that turns it on. Any fourth is a new path buying the fast tier.
_PLUMBING = ("openai_client/base.py", "openai_client/api/responses.py")
_THE_ONLY_CALLER = "message_processor/handlers/text.py"

# Every responder entry point a handler can call. A new one added to the caller below without
# the flag is a user-facing turn silently dropped back to the standard pool.
_RESPONDER_CALLS = frozenset({
    "create_text_response", "create_text_response_with_tools",
    "create_text_response_with_tool_loop", "create_streaming_response",
    "create_streaming_response_with_tools", "create_streaming_response_with_tool_loop",
    "_create_text_response_with_timeout", "_create_text_response_with_tools_with_timeout",
})

_SKIP_DIRS = {"tests", ".venv", "venv", "Docs", "data", "htmlcov", "logs", "metrics_reports",
              "__pycache__", ".git", "status_messages"}


def _production_files() -> List[str]:
    """Every .py file that ships, as a repo-relative path."""
    found: List[str] = []
    for dirpath, dirnames, filenames in os.walk(_REPO_ROOT):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if name.endswith(".py"):
                found.append(os.path.relpath(os.path.join(dirpath, name), _REPO_ROOT))
    return sorted(found)


class TestEligibilityBoundary:
    """WHO may ask for the fast tier — asserted over the shipping source, not over a fixture.

    The wrapper-level tests above prove the flag works. They cannot prove that the right
    callers set it: a handler that quietly loses `service_tier_eligible=True` still passes
    every one of them, and so does `research_tools.py` growing a copy of it and putting the
    2x meter on a background job nobody is waiting for. Only the call sites can say that, so
    this reads them.
    """

    def test_only_three_shipping_files_mention_eligibility(self):
        mentions = [f for f in _production_files()
                    if "service_tier_eligible" in open(os.path.join(_REPO_ROOT, f),
                                                       encoding="utf-8").read()]
        assert sorted(mentions) == sorted([*_PLUMBING, _THE_ONLY_CALLER]), (
            "a new file asks for the fast tier. Eligibility belongs to the user-facing "
            f"responder call and nothing else — see spec W2. Got: {mentions}")

    def test_no_background_research_or_utility_path_asks_for_it(self):
        """Named explicitly, because these are the ones it would be tempting to speed up: they
        are the calls nobody is watching, and the spec says they stay on the standard pool."""
        for path in ("message_processor/research_tools.py",
                     "message_processor/ambient_memory.py",
                     "message_processor/thread_management.py",
                     "message_processor/channel_summary.py",
                     "message_processor/utilities.py",
                     "message_processor/image_delivery.py",
                     "message_processor/participation.py",
                     "slack_client/event_handlers/channel_join.py"):
            with open(os.path.join(_REPO_ROOT, path), encoding="utf-8") as fh:
                assert "service_tier_eligible" not in fh.read(), (
                    f"{path} must not ask for the fast tier")

    def test_every_responder_call_in_the_handler_is_marked_eligible(self):
        """The other direction: a responder call site added to the handler and NOT marked would
        run the user's turn on the standard pool while the operator believes they bought fast."""
        import ast

        with open(os.path.join(_REPO_ROOT, _THE_ONLY_CALLER), encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        found: Dict[str, List[bool]] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            name = node.func.attr
            if name not in _RESPONDER_CALLS:
                continue
            eligible = any(kw.arg == "service_tier_eligible"
                           and isinstance(kw.value, ast.Constant) and kw.value.value is True
                           for kw in node.keywords)
            found.setdefault(name, []).append(eligible)

        assert found, "found no responder calls at all — this test has lost its target"
        unmarked = {n: flags for n, flags in found.items() if not all(flags)}
        assert not unmarked, (
            f"responder calls in {_THE_ONLY_CALLER} missing service_tier_eligible=True: "
            f"{sorted(unmarked)}")
        assert sum(len(v) for v in found.values()) == 8, (
            f"the handler's responder call sites moved; recount and re-read them: {found}")


def _recording_openai_client() -> MagicMock:
    """An openai_client that records the kwargs of whichever responder the handler picks."""
    from unittest.mock import AsyncMock

    loop_result: Dict[str, Any] = {"text": "here you go", "tools_used": [],
                                   "local_tool_calls": [], "terminal_action": None,
                                   "silence_reason": None}
    client = MagicMock()
    client.create_text_response_with_tool_loop = AsyncMock(return_value=loop_result)
    client.create_streaming_response_with_tool_loop = AsyncMock(return_value=loop_result)
    return client


def _handler_host(streaming: bool) -> MagicMock:
    """The REAL handler method on a mock host — the same shape tests/unit/test_search_to_action
    uses. Everything stubbed here is scenery; the method under test is production code."""
    from message_processor.handlers.text import TextHandlerMixin

    host = MagicMock()
    method = (TextHandlerMixin._handle_streaming_text_response if streaming
              else TextHandlerMixin._handle_text_response)
    host.handler = method.__get__(host)
    host._background_tasks = set()
    host._is_reaction_only = TextHandlerMixin._is_reaction_only

    async def _passthru(m, *a, **k):
        return m

    async def _none(*a, **k):
        return None

    async def _empty(*a, **k):
        return ""

    host._inject_image_analyses = _passthru
    host._pre_trim_messages_for_api = _passthru
    host._build_channel_info = _empty
    host._drop_dead_containers = _none
    host._resolve_ci_container = _none
    host._prepare_sandbox_tools = _none
    host._cleanup_silent_stream = _none
    host._get_system_prompt = MagicMock(return_value="sys")
    host._build_suffix_context = MagicMock(return_value="")
    host._build_participant_roster = MagicMock(return_value="")
    host._build_tools_array = MagicMock(return_value=[{"type": "function", "name": "t"}])
    host._materialize_request_tools = MagicMock(
        return_value=(MagicMock(), {"model": "gpt-5.6-sol"}, True, None))
    host._build_tool_context = MagicMock(return_value=SimpleNamespace(
        background_job_started=False, sandbox_image_assets=[], mounted_files=[]))
    host._current_image_urls = MagicMock(return_value=[])
    host._add_message_with_token_management = MagicMock()
    host.mcp_manager = MagicMock()
    host.openai_client = _recording_openai_client()
    return host


async def _drive_handler(host, *, streaming: bool):
    from unittest.mock import patch

    from base_client import Message

    message = Message(text="what's the total?", user_id="U1", channel_id="C1",
                      thread_id="9000.0", metadata={"ts": "9000.0"})
    thread_state = SimpleNamespace(
        messages=[{"role": "user", "content": "hi"}], channel_id="C1", thread_ts="9000.0",
        current_model="gpt-5.6-sol", config_overrides={}, has_summary_head=False,
        channel_directives=None, record_usage=MagicMock(), last_usage=None, participants={})

    async def fake_config(**kw):
        return {"model": "gpt-5.6-sol", "temperature": 1.0, "max_tokens": 100,
                "enable_streaming": streaming, "enable_code_interpreter": False}

    client = MagicMock()
    client.name = "Slack"
    client.supports_streaming = MagicMock(return_value=True)
    client.supports_native_streaming = MagicMock(return_value=False)
    client.get_streaming_config = MagicMock(
        return_value={"update_interval": 0.0, "buffer_size": 1, "min_interval": 0.0})
    with patch.object(config, "get_thread_config_async", side_effect=fake_config):
        return await host.handler("what's the total?", thread_state, client, message,
                                  thinking_id=None, turn=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [True, False], ids=["streaming", "non-streaming"])
async def test_the_real_handler_marks_its_responder_call_eligible(streaming):
    """The link the wrapper tests cannot make: the production handler, driven for real, is what
    turns eligibility on. Delete `service_tier_eligible=True` from the call site and this fails
    — which is the whole point, because nothing else notices."""
    host = _handler_host(streaming)
    await _drive_handler(host, streaming=streaming)

    call = (host.openai_client.create_streaming_response_with_tool_loop if streaming
            else host.openai_client.create_text_response_with_tool_loop)
    assert call.await_count == 1, "the handler did not reach the responder call"
    assert call.await_args.kwargs.get("service_tier_eligible") is True
