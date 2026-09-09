"""GPT-5.6 upgrade — effort clamp, per-model modal ladders, request-param shapes.

Live-verified facts these tests encode (probed against the real API 2026-07-09):
- `max` returns 200 on ALL three 5.6 tiers -> offered everywhere on 5.6
- `minimal` 400s on every 5.6 model -> must never reach the API
- effort=none allows temperature/top_p on 5.6 (same hybrid shape as 5.5)
- 5.6 uses implicit prompt caching -> no prompt_cache_retention param

Extended for GPT-6 Astra (probed 2026-09-08): no `none` on the ladder, no sampling parameters
at any effort, `prompt_cache_options={"ttl":"30m"}` and never `prompt_cache_retention`, and a
trailing `configuration_update` input item as the cache-preserving way to raise one call's
effort. The 5.5/5.6 request shapes must stay byte-identical through all of it.
"""
import json
from unittest.mock import MagicMock

import pytest

from config import (GPT6_EFFORTS, GPT55_EFFORTS, GPT56_EFFORTS, SUPPORTED_CHAT_MODELS,
                    clamp_effort, config)
from slack_client.settings_modal import SettingsModal


@pytest.fixture
def modal():
    return SettingsModal(db=MagicMock())


# --- clamp_effort ---

@pytest.mark.critical
class TestClampEffort:
    def test_minimal_maps_to_none_on_56(self):
        for model in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
            assert clamp_effort(model, "minimal") == "none"

    def test_full_ladder_passes_through_on_56(self):
        for effort in GPT56_EFFORTS:
            assert clamp_effort("gpt-5.6-sol", effort) == effort
            assert clamp_effort("gpt-5.6-luna", effort) == effort

    def test_max_maps_to_xhigh_on_55(self):
        assert clamp_effort("gpt-5.5", "max") == "xhigh"

    def test_minimal_maps_to_low_on_55(self):
        assert clamp_effort("gpt-5.5", "minimal") == "low"

    def test_minimal_stays_on_mini(self):
        # gpt-5-mini (legacy utility) still accepts minimal
        assert clamp_effort("gpt-5-mini", "minimal") == "minimal"

    def test_unknown_and_none_fall_back_to_medium(self):
        assert clamp_effort("gpt-5.6-sol", "turbo") == "medium"
        assert clamp_effort("gpt-5.6-sol", None) == "medium"
        assert clamp_effort("gpt-5.5", "bogus") == "medium"

    def test_case_insensitive(self):
        assert clamp_effort("gpt-5.6-sol", "Minimal") == "none"
        assert clamp_effort("gpt-5.6-sol", "MAX") == "max"


# --- defaults ---

class TestDefaults:
    def test_supported_lineup(self):
        # Astra leads: the list order is the modal's display order, and the workspace default
        # belongs at the top of it.
        assert SUPPORTED_CHAT_MODELS == [
            "gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5"
        ]

    def test_ladders(self):
        assert GPT6_EFFORTS == ["low", "medium", "high", "xhigh", "max"]
        assert GPT56_EFFORTS == ["none", "low", "medium", "high", "xhigh", "max"]
        assert GPT55_EFFORTS == ["none", "low", "medium", "high", "xhigh"]

    def test_config_defaults(self, monkeypatch):
        from config import BotConfig
        monkeypatch.delenv("GPT_MODEL", raising=False)
        monkeypatch.delenv("UTILITY_MODEL", raising=False)
        monkeypatch.delenv("UTILITY_REASONING_EFFORT", raising=False)
        fresh = BotConfig()
        assert fresh.gpt_model == "gpt-6-astra"
        # The utility model stays on luna: `none` is legal there, and utility work whose whole
        # point is being cheap does not move to a model an order of magnitude more expensive.
        assert fresh.utility_model == "gpt-5.6-luna"
        assert fresh.utility_reasoning_effort == "none"


# --- modal effort ladder per model ---

class TestModalLadders:
    def _effort_values(self, modal, model, settings=None):
        blocks = modal._add_gpt55_settings(settings or {}, model)
        reasoning_block = next(b for b in blocks if b.get("block_id") == "reasoning_block_gpt54")
        return [o["value"] for o in reasoning_block["accessory"]["options"]]

    def test_56_models_offer_max(self, modal):
        for model in ("gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"):
            assert self._effort_values(modal, model) == GPT56_EFFORTS

    def test_55_has_no_max(self, modal):
        assert self._effort_values(modal, "gpt-5.5") == GPT55_EFFORTS

    def test_stale_max_clamps_when_switching_to_55(self, modal):
        """User had max on Sol, switches to 5.5 — initial_option must be valid."""
        blocks = modal._add_gpt55_settings({"reasoning_effort": "max"}, "gpt-5.5")
        reasoning_block = next(b for b in blocks if b.get("block_id") == "reasoning_block_gpt54")
        initial = reasoning_block["accessory"]["initial_option"]["value"]
        assert initial == "xhigh"

    def test_stale_minimal_clamps_on_56(self, modal):
        blocks = modal._add_gpt55_settings({"reasoning_effort": "minimal"}, "gpt-5.6-sol")
        reasoning_block = next(b for b in blocks if b.get("block_id") == "reasoning_block_gpt54")
        initial = reasoning_block["accessory"]["initial_option"]["value"]
        assert initial == "none"

    def test_stale_model_coerces_to_sol_in_full_modal(self, modal):
        blocks = modal._build_modal_blocks(
            settings={"model": "gpt-4o"}, selected_model="gpt-5.6-sol",
            is_new_user=False, in_thread=False, scope="global",
        )
        model_block = next(b for b in blocks if b.get("block_id") == "model_block")
        assert model_block["accessory"]["initial_option"]["value"] == "gpt-5.6-sol"


# --- validate_settings clamps ---

class TestValidateSettings:
    def test_minimal_clamped_on_56(self, modal):
        validated = modal.validate_settings(
            {"model": "gpt-5.6-sol", "reasoning_effort": "minimal"})
        assert validated["reasoning_effort"] == "none"

    def test_max_clamped_on_55(self, modal):
        validated = modal.validate_settings(
            {"model": "gpt-5.5", "reasoning_effort": "max"})
        assert validated["reasoning_effort"] == "xhigh"

    def test_max_kept_on_56(self, modal):
        validated = modal.validate_settings(
            {"model": "gpt-5.6-terra", "reasoning_effort": "max"})
        assert validated["reasoning_effort"] == "max"


# --- request-param shapes (responses.py builders) ---

class _FakeClient:
    """Capture request_params without a network call."""
    def __init__(self):
        self.captured = {}
        self.client = MagicMock()
        self.client.timeout = 30

    def log_debug(self, *a, **k): pass
    def log_info(self, *a, **k): pass
    def log_warning(self, *a, **k): pass
    def log_error(self, *a, **k): pass

    async def _safe_api_call(self, fn, operation_type=None, timeout_seconds=None, **params):
        self.captured = params
        resp = MagicMock()
        resp.output = []
        resp.usage = None
        return resp


@pytest.mark.asyncio
class TestRequestParams:
    async def _call(self, model, effort, **kwargs):
        from openai_client.api import responses as R
        fake = _FakeClient()
        await R.create_text_response(
            fake, messages=[{"role": "user", "content": "hi"}],
            model=model, reasoning_effort=effort,
            prompt_cache_key="thread-key", **kwargs,
        )
        return fake.captured

    async def test_56_no_prompt_cache_retention(self):
        params = await self._call("gpt-5.6-sol", "medium")
        assert "prompt_cache_retention" not in params
        assert params["prompt_cache_key"] == "thread-key"
        assert params["reasoning"] == {"effort": "medium"}

    async def test_55_keeps_prompt_cache_retention(self):
        params = await self._call("gpt-5.5", "medium")
        assert params["prompt_cache_retention"] == "24h"
        assert params["prompt_cache_key"] == "thread-key"

    async def test_56_minimal_clamped_before_api(self):
        params = await self._call("gpt-5.6-luna", "minimal")
        assert params["reasoning"] == {"effort": "none"}

    async def test_55_max_clamped_before_api(self):
        params = await self._call("gpt-5.5", "max")
        assert params["reasoning"] == {"effort": "xhigh"}

    async def test_56_temp_top_p_at_none(self):
        params = await self._call("gpt-5.6-sol", "none", temperature=0.7, top_p=0.9)
        assert params["top_p"] == 0.9
        assert params["temperature"] == 0.7

    async def test_56_temp_forced_when_reasoning(self):
        params = await self._call("gpt-5.6-sol", "high", temperature=0.7)
        assert params["temperature"] == 1.0
        assert "top_p" not in params

    async def test_utility_paths_clamp(self):
        """Utility call sites route the configured effort through the clamp."""
        from openai_client.api import responses as R
        fake = _FakeClient()
        orig_model, orig_effort = config.utility_model, config.utility_reasoning_effort
        try:
            config.utility_model = "gpt-5.6-luna"
            config.utility_reasoning_effort = "minimal"  # stale .env value
            await R.extract_memory(fake, exchange_text="hello")
            assert fake.captured["reasoning"] == {"effort": "none"}
        finally:
            config.utility_model = orig_model
            config.utility_reasoning_effort = orig_effort


# --- GPT-6 request shape + configuration_update -----------------------------------------
#
# Probed live 2026-09-08 against our own key:
#   temperature=0.8 -> 400 "not supported with this model"; top_p -> 400 at every value
#   prompt_cache_options={"ttl":"30m"} -> 200; {"ttl":"24h"} -> 400 (30m is the only value)
#   prompt_cache_retention="24h" -> 200, but deprecated and a no-op, so it must not be sent
#   a trailing {"type":"configuration_update","reasoning":{"effort":...}} input item -> 200,
#   and it PRESERVES the prompt cache (4016 cached tokens kept) where changing the top-level
#   effort is a full miss.

def _build(**kwargs):
    from openai_client.base import _build_request_params
    kwargs.setdefault("input_items", [{"role": "user", "content": "hi"}])
    return _build_request_params(**kwargs)


class TestGpt6RequestShape:
    def test_gpt6_sends_neither_sampling_key(self):
        params = _build(model="gpt-6-astra", reasoning_effort="medium",
                        temperature=0.7, top_p=0.9)
        assert "temperature" not in params
        assert "top_p" not in params

    def test_gpt6_sends_neither_sampling_key_at_the_lowest_effort_either(self):
        # There is no `none` on this family, so there is no branch where sampling comes back.
        params = _build(model="gpt-6-astra", reasoning_effort="none", temperature=0.7)
        assert "temperature" not in params and "top_p" not in params
        assert params["reasoning"] == {"effort": "low"}      # clamped, not passed through

    def test_gpt6_cache_shape(self):
        params = _build(model="gpt-6-astra", reasoning_effort="medium",
                        prompt_cache_key="thread-key", layout="channel")
        assert "prompt_cache_retention" not in params
        assert params["prompt_cache_key"] == "thread-key"
        assert params["prompt_cache_options"] == {"ttl": "30m"}

    def test_a_caller_supplied_cache_option_still_wins_on_gpt6(self):
        params = _build(model="gpt-6-astra", reasoning_effort="medium",
                        prompt_cache_key="k", layout="channel",
                        prompt_cache_options={"ttl": "30m", "scope": "x"})
        assert params["prompt_cache_options"] == {"ttl": "30m", "scope": "x"}

    def test_gpt6_always_caches_even_off_the_legacy_cache_path(self):
        """Spec 2.4: GPT-6's cache shape is how the family caches, not an opt-in.

         is the plain timeout twin's shipped gap on 5.x. Letting
        GPT-6 inherit it dropped both the key and the ttl on every call down that path.
        """
        params = _build(model="gpt-6-astra", reasoning_effort="medium",
                        prompt_cache_key="thread-key", legacy_cache_params=False)
        assert params["prompt_cache_key"] == "thread-key"
        assert params["prompt_cache_options"] == {"ttl": "30m"}
        assert "prompt_cache_retention" not in params

    def test_the_5_6_gap_on_that_same_path_is_left_exactly_where_it_was(self):
        # The exclusion GPT-6 steps around must still hold for 5.6: a shipped shape.
        params = _build(model="gpt-5.6-sol", reasoning_effort="medium",
                        prompt_cache_key="thread-key", legacy_cache_params=False)
        assert "prompt_cache_key" not in params
        assert "prompt_cache_options" not in params
        assert "prompt_cache_retention" not in params


class TestEffortOverride:
    """One call's departure from the thread's baseline effort.

    On GPT-6 a changed TOP-LEVEL effort is a full prompt-cache miss, so the baseline stays put
    and the override rides a trailing input item instead. Everywhere else the override simply
    replaces the top-level value, which is what the code has always done.
    """

    def test_gpt6_keeps_the_baseline_and_appends_the_item_last(self):
        params = _build(model="gpt-6-astra", reasoning_effort="low", effort_override="high")
        assert params["reasoning"] == {"effort": "low"}          # the baseline, untouched
        assert params["input"][-1] == {"type": "configuration_update",
                                       "reasoning": {"effort": "high"}}
        assert len(params["input"]) == 2

    def test_the_overridden_value_is_clamped_too(self):
        # A stored `none` reaching the override path must not become a 400 in an input item.
        params = _build(model="gpt-6-astra", reasoning_effort="medium", effort_override="none")
        assert params["input"][-1]["reasoning"] == {"effort": "low"}

    def test_the_item_survives_the_channel_layout(self):
        # `_CHANNEL_TYPED_ITEMS` does not carry `configuration_update`, so an item appended
        # BEFORE the allowlist runs is silently dropped. It must be appended after.
        params = _build(model="gpt-6-astra", reasoning_effort="low", effort_override="high",
                        system_prompt="sys", layout="channel", prompt_cache_key="k")
        assert params["input"][-1]["type"] == "configuration_update"
        assert params["reasoning"] == {"effort": "low"}

    def test_5_6_replaces_the_top_level_effort_and_appends_nothing(self):
        params = _build(model="gpt-5.6-sol", reasoning_effort="low", effort_override="high")
        assert params["reasoning"] == {"effort": "high"}
        assert all(item.get("type") != "configuration_update" for item in params["input"])
        assert len(params["input"]) == 1

    def test_no_override_appends_nothing_on_gpt6(self):
        params = _build(model="gpt-6-astra", reasoning_effort="low")
        assert len(params["input"]) == 1
        assert params["reasoning"] == {"effort": "low"}

    @pytest.mark.asyncio
    async def test_the_streaming_wrapper_forwards_the_override(self, monkeypatch):
        """The builder is only half of it: `create_streaming_response_with_tools` is the
        signature deep research actually calls, and an intervening hop that dropped the kwarg
        would leave the cache-preserving transport unused with nothing failing."""
        from openai_client.api import responses as R

        captured = {}

        class _Stop(Exception):
            pass

        def _spy(**kwargs):
            captured.update(kwargs)
            raise _Stop()

        monkeypatch.setattr(R, "_build", _spy)
        with pytest.raises(_Stop):
            await R.create_streaming_response_with_tools(
                MagicMock(), messages=[{"role": "user", "content": "hi"}],
                tools=[], stream_callback=None,
                model="gpt-6-astra", reasoning_effort="low", effort_override="high")

        assert captured["effort_override"] == "high"
        assert captured["reasoning_effort"] == "low"
_SHIPPED_5X_CALLS = {
    "gpt56_channel": dict(
        model="gpt-5.6-sol", layout="channel",
        system_prompt="You are a helpful assistant.",
        input_items=[{"role": "user", "content": "What is the plan?"},
                     {"role": "assistant", "content": "Here it is."}],
        reasoning_effort="high", verbosity="low",
        prompt_cache_key="C_TEST:1234.5678",
        max_output_tokens=4096, temperature=0.7, top_p=0.9,
        stream=True, store=False),
    "gpt56_legacy": dict(
        model="gpt-5.6-sol", layout="legacy", legacy_kind="plain",
        system_prompt="You are a helpful assistant.",
        input_items=[{"role": "user", "content": "What is the plan?"}],
        reasoning_effort="none", verbosity="medium",
        prompt_cache_key="C_TEST:1234.5678",
        max_output_tokens=2048, temperature=0.3, top_p=0.8,
        stream=False, store=False),
    "gpt55_channel": dict(
        model="gpt-5.5", layout="channel",
        system_prompt="You are a helpful assistant.",
        input_items=[{"role": "user", "content": "What is the plan?"},
                     {"role": "assistant", "content": "Here it is."}],
        reasoning_effort="xhigh", verbosity="high",
        prompt_cache_key="C_TEST:1234.5678",
        max_output_tokens=4096, temperature=0.7, top_p=0.9,
        stream=True, store=False),
    "gpt55_legacy": dict(
        model="gpt-5.5", layout="legacy", legacy_kind="tools",
        system_prompt="You are a helpful assistant.",
        input_items=[{"role": "user", "content": "What is the plan?"}],
        reasoning_effort="none", verbosity="medium",
        prompt_cache_key="C_TEST:1234.5678", tools=[],
        max_output_tokens=2048, temperature=0.3, top_p=0.8,
        stream=False, store=False),
    "gpt56_no_cache_params": dict(
        model="gpt-5.6-sol", layout="legacy", legacy_kind="plain", legacy_cache_params=False,
        system_prompt="You are a helpful assistant.",
        input_items=[{"role": "user", "content": "What is the plan?"}],
        reasoning_effort="medium", verbosity="medium",
        prompt_cache_key="C_TEST:1234.5678",
        max_output_tokens=2048, temperature=0.3, top_p=0.8,
        stream=False, store=False),
    "gpt55_no_cache_params": dict(
        model="gpt-5.5", layout="legacy", legacy_kind="plain", legacy_cache_params=False,
        system_prompt="You are a helpful assistant.",
        input_items=[{"role": "user", "content": "What is the plan?"}],
        reasoning_effort="medium", verbosity="medium",
        prompt_cache_key="C_TEST:1234.5678",
        max_output_tokens=2048, temperature=0.3, top_p=0.8,
        stream=False, store=False),
}


# The COMPLETE serialized request each shipped 5.x call produced at 85451fb, derived by
# running that commit's own builder. Not a key list and not a spot-check of three
# fields: the whole payload, so a key added, dropped, renamed, reordered or revalued
# anywhere in the 5.x path fails here. GPT-6 was threaded through the EXISTING slots
# rather than appended, and this is what says so.
SHIPPED_5X_REQUESTS = {
    # the shipped channel turn
    "gpt56_channel":
        '{"model": "gpt-5.6-sol", "input": [{"role": "user", "content": "What is the plan?"}, {"role": "assistant", "content": "Here it is."}], "temperature": 1.0, "max_output_tokens": 4096, "store": false, "stream": true, "instructions": "You are a helpful assistant.", "reasoning": {"effort": "high"}, "text": {"verbosity": "low"}, "prompt_cache_key": "C_TEST:1234.5678"}',
    # a DM turn at effort=none, where sampling comes back
    "gpt56_legacy":
        '{"model": "gpt-5.6-sol", "input": [{"role": "developer", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the plan?"}], "temperature": 0.3, "max_output_tokens": 2048, "store": false, "reasoning": {"effort": "none"}, "text": {"verbosity": "medium"}, "top_p": 0.8, "prompt_cache_key": "C_TEST:1234.5678"}',
    # 5.5 keeps its explicit 24h retention
    "gpt55_channel":
        '{"model": "gpt-5.5", "input": [{"role": "user", "content": "What is the plan?"}, {"role": "assistant", "content": "Here it is."}], "temperature": 1.0, "max_output_tokens": 4096, "store": false, "stream": true, "instructions": "You are a helpful assistant.", "reasoning": {"effort": "xhigh"}, "text": {"verbosity": "high"}, "prompt_cache_retention": "24h", "prompt_cache_key": "C_TEST:1234.5678"}',
    # the tools layout, which promotes the prompt to top-level instructions
    "gpt55_legacy":
        '{"model": "gpt-5.5", "input": [{"role": "user", "content": "What is the plan?"}], "tools": [], "temperature": 0.3, "max_output_tokens": 2048, "store": false, "instructions": "You are a helpful assistant.", "reasoning": {"effort": "none"}, "text": {"verbosity": "medium"}, "top_p": 0.8, "prompt_cache_retention": "24h", "prompt_cache_key": "C_TEST:1234.5678"}',
    # the plain timeout twin: no cache params at all
    "gpt56_no_cache_params":
        '{"model": "gpt-5.6-sol", "input": [{"role": "developer", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the plan?"}], "temperature": 1.0, "max_output_tokens": 2048, "store": false, "reasoning": {"effort": "medium"}, "text": {"verbosity": "medium"}}',
    # same gap on 5.5 — a shipped bug, but a shipped shape
    "gpt55_no_cache_params":
        '{"model": "gpt-5.5", "input": [{"role": "developer", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the plan?"}], "temperature": 1.0, "max_output_tokens": 2048, "store": false, "reasoning": {"effort": "medium"}, "text": {"verbosity": "medium"}}',
}


@pytest.mark.parametrize("case", list(SHIPPED_5X_REQUESTS))
def test_the_5x_request_shapes_are_byte_identical_to_the_shipped_ones(case):
    """Every 5.5/5.6 request must serialize exactly as it did before GPT-6 arrived."""
    params = _build(**_SHIPPED_5X_CALLS[case])
    assert json.dumps(params, sort_keys=False) == SHIPPED_5X_REQUESTS[case]
