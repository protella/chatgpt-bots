"""Context-window budgets — verified 2026-07-09 against the official OpenAI model
pages: the GPT-5.6 family (sol/terra/luna) AND gpt-5.5 share a 1,050,000-token
context window with 128,000 max output tokens; gpt-5-mini is 400,000/128,000.
Prompts >272K input bill at 2x input / 1.5x output on 5.5 and the 5.6 family.

These tests pin the resolved per-model limits, the reserve formula (usable input
must leave room for the configured output cap plus estimator/tool/doc headroom),
and the long-context billing helper. If a future model changes the window, update
config.py's verified-comment block and these pins together."""
import pytest

from config import BotConfig, SUPPORTED_CHAT_MODELS


@pytest.fixture
def cfg(monkeypatch):
    # Pin the env-backed values to defaults so a developer's .env can't skew pins.
    for var in (
        "GPT54_MAX_TOKENS", "GPT5_MAX_TOKENS",
        "GPT54_TOKEN_BUFFER_PERCENTAGE", "TOKEN_BUFFER_PERCENTAGE",
        "TOKEN_CLEANUP_THRESHOLD", "TOKEN_COMPACTION_TARGET",
        "DEFAULT_MAX_TOKENS", "VISION_MAX_TOKENS", "UTILITY_MODEL",
    ):
        monkeypatch.delenv(var, raising=False)
    return BotConfig()


FULL_WINDOW_MODELS = ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5"]
FULL_WINDOW = 1_050_000
FALLBACK_WINDOW = 400_000
MAX_OUTPUT = 128_000  # API max output tokens (all five models)


class TestResolvedLimits:
    @pytest.mark.parametrize("model", FULL_WINDOW_MODELS)
    def test_full_window_models_get_full_budget(self, cfg, model):
        assert cfg.get_model_token_limit(model) == int(FULL_WINDOW * 0.876)  # 919,800

    @pytest.mark.parametrize("model", ["gpt-5-mini", "some-future-model", "gpt-4o"])
    def test_unknown_and_legacy_fall_back_conservatively(self, cfg, model):
        assert cfg.get_model_token_limit(model) == int(FALLBACK_WINDOW * 0.875)  # 350,000

    def test_every_selectable_model_gets_the_full_window(self, cfg):
        for model in SUPPORTED_CHAT_MODELS:
            assert cfg.get_model_token_limit(model) == int(FULL_WINDOW * 0.876), model

    def test_utility_model_budgets_against_its_real_window(self, cfg):
        # The utility model is gpt-5.6-luna (1.05M) — its budget must NOT be the
        # old gpt-5-mini 400k fallback (threads budgeted at ~920k flow into
        # utility calls with full context).
        assert cfg.utility_model == "gpt-5.6-luna"
        assert cfg.get_model_token_limit(cfg.utility_model) == int(FULL_WINDOW * 0.876)


class TestReserveFormula:
    def test_full_window_reserve_covers_output_cap_plus_headroom(self, cfg):
        usable = cfg.get_model_token_limit("gpt-5.6-sol")
        reserve = FULL_WINDOW - usable
        # Reserve must cover the configured output cap (reasoning bills inside it)...
        assert reserve >= cfg.default_max_tokens
        assert reserve >= cfg.vision_max_tokens
        # ...plus real headroom for chars/4 estimator error and tool/doc overhead
        # (at least ~5% of the window beyond the output cap).
        headroom = reserve - max(cfg.default_max_tokens, cfg.vision_max_tokens)
        assert headroom >= int(FULL_WINDOW * 0.05)

    def test_fallback_reserve_covers_output_cap(self, cfg):
        usable = cfg.get_model_token_limit("gpt-5-mini")
        reserve = FALLBACK_WINDOW - usable
        assert reserve >= cfg.default_max_tokens

    def test_output_caps_within_api_max(self, cfg):
        assert cfg.default_max_tokens <= MAX_OUTPUT
        assert cfg.vision_max_tokens <= MAX_OUTPUT

    def test_compaction_thresholds_ordered(self, cfg):
        # Compaction must target strictly below its own trigger, both within the budget.
        assert 0 < cfg.token_compaction_target < cfg.token_cleanup_threshold <= 1


class TestLongContextBilling:
    def test_threshold_value(self, cfg):
        assert cfg.LONG_CONTEXT_BILLING_THRESHOLD == 272_000

    def test_boundary(self, cfg):
        assert not cfg.is_long_context(272_000)
        assert cfg.is_long_context(272_001)
        assert not cfg.is_long_context(0)

    def test_threshold_below_usable_budget(self, cfg):
        # The billing tier sits well inside the usable window — crossing it is
        # normal operation (log-only), never a blocker.
        assert cfg.LONG_CONTEXT_BILLING_THRESHOLD < cfg.get_model_token_limit("gpt-5.6-sol")
