"""GPT-5.6 upgrade — effort clamp, per-model modal ladders, request-param shapes.

Live-verified facts these tests encode (probed against the real API 2026-07-09,
see Docs/GPT_5_6_UPGRADE_PLAN.md "Verification results"):
- `max` returns 200 on ALL three 5.6 tiers -> offered everywhere on 5.6
- `minimal` 400s on every 5.6 model -> must never reach the API
- effort=none allows temperature/top_p on 5.6 (same hybrid shape as 5.5)
- 5.6 uses implicit prompt caching -> no prompt_cache_retention param
"""
from unittest.mock import MagicMock

import pytest

from config import (GPT55_EFFORTS, GPT56_EFFORTS, SUPPORTED_CHAT_MODELS,
                    clamp_effort, config)
from settings_modal import SettingsModal


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
        assert SUPPORTED_CHAT_MODELS == [
            "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5"
        ]

    def test_ladders(self):
        assert GPT56_EFFORTS == ["none", "low", "medium", "high", "xhigh", "max"]
        assert GPT55_EFFORTS == ["none", "low", "medium", "high", "xhigh"]

    def test_config_defaults(self, monkeypatch):
        from config import BotConfig
        monkeypatch.delenv("GPT_MODEL", raising=False)
        monkeypatch.delenv("UTILITY_MODEL", raising=False)
        monkeypatch.delenv("UTILITY_REASONING_EFFORT", raising=False)
        fresh = BotConfig()
        assert fresh.gpt_model == "gpt-5.6-sol"
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
