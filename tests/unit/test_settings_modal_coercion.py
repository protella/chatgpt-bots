"""Regression tests for two modal-rejection bugs.

F1  — the ambient-memory checkbox `description` must stay under Slack's option-description limit,
      or views.open fails with invalid_arguments and the whole modal never renders.
F23 — every static_select / radio_buttons initial_option value must be one of that element's
      options; a stale stored value (e.g. a retired gpt-image-1-mini image model) otherwise makes
      Slack reject the entire modal. All stored image/vision selects are coerced before render.

Pure builders — no live Slack, no API.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from settings_modal import SettingsModal
from slack_client.event_handlers.settings import SlackSettingsHandlersMixin


# Slack caps an option object's `description` (checkboxes/radio). The live failure was at 153
# chars; keep a margin under the reported 150 ceiling.
_OPTION_DESCRIPTION_LIMIT = 150


def _selects(blocks):
    """Yield (element, options) for every static_select / radio_buttons in the blocks that
    carries an initial_option — the elements Slack validates against their option list."""
    for block in blocks:
        for element in (block.get("accessory"), block.get("element")):
            if not isinstance(element, dict):
                continue
            if element.get("type") not in ("static_select", "radio_buttons"):
                continue
            if "initial_option" in element:
                yield element


class TestAmbientDescriptionLength:
    def test_description_under_slack_limit(self):
        block = SlackSettingsHandlersMixin._ambient_memory_block(None)
        desc = block["element"]["options"][0]["description"]["text"]
        assert len(desc) < _OPTION_DESCRIPTION_LIMIT, f"description is {len(desc)} chars"

    def test_description_still_conveys_meaning(self):
        block = SlackSettingsHandlersMixin._ambient_memory_block(None)
        desc = block["element"]["options"][0]["description"]["text"].lower()
        # Still says it takes notes and that they age out — the whole point of the opt-out.
        assert "note" in desc
        assert "age out" in desc


class TestStaleSelectCoercion:
    @pytest.fixture
    def modal(self):
        return SettingsModal(db=MagicMock())

    def _assert_all_initial_options_valid(self, blocks):
        for element in _selects(blocks):
            valid = {opt["value"] for opt in element["options"]}
            value = element["initial_option"]["value"]
            assert value in valid, (
                f"{element.get('action_id')} initial_option {value!r} not in {valid}")

    def test_stale_values_do_not_break_modal(self, modal):
        # Every image/vision select carries a value that is no longer an option.
        stale = {
            "image_model": "gpt-image-1-mini",   # retired — the reported trigger
            "image_size": "512x512",
            "image_quality": "ultra",
            "image_background": "rainbow",
            "input_fidelity": "medium",
            "vision_detail": "extreme",
        }
        blocks = modal._build_modal_blocks(stale, selected_model="gpt-5.6-sol")
        self._assert_all_initial_options_valid(blocks)

    def test_stale_image_model_falls_back_to_a_real_option(self, modal):
        blocks = modal._build_modal_blocks(
            {"image_model": "gpt-image-1-mini"}, selected_model="gpt-5.6-sol")
        element = next(e for e in _selects(blocks) if e.get("action_id") == "image_model")
        assert element["initial_option"]["value"] in {"gpt-image-2", "gpt-image-1"}

    def test_valid_values_pass_through_unchanged(self, modal):
        good = {
            "image_model": "gpt-image-1",
            "image_size": "1536x1024",
            "image_quality": "high",
            "vision_detail": "low",
        }
        blocks = modal._build_modal_blocks(good, selected_model="gpt-5.6-sol")
        by_id = {e["action_id"]: e["initial_option"]["value"] for e in _selects(blocks)}
        assert by_id["image_model"] == "gpt-image-1"
        assert by_id["image_size"] == "1536x1024"
        assert by_id["image_quality"] == "high"
        assert by_id["vision_detail"] == "low"

    def test_empty_settings_are_valid(self, modal):
        # A brand-new user with no stored image prefs must also render a valid modal.
        blocks = modal._build_modal_blocks({}, selected_model="gpt-5.6-sol")
        self._assert_all_initial_options_valid(blocks)
