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

from slack_client.settings_modal import SettingsModal
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
        from config import SUPPORTED_IMAGE_MODELS
        blocks = modal._build_modal_blocks(
            {"image_model": "gpt-image-1-mini"}, selected_model="gpt-5.6-sol")
        element = next(e for e in _selects(blocks) if e.get("action_id") == "image_model")
        assert element["initial_option"]["value"] in set(SUPPORTED_IMAGE_MODELS)

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
        # `image_size` is no longer a control: shape and tier are, and they resolve to the one
        # stored key. 1536x1024 is Landscape at the standard tier.
        assert by_id["image_ratio"] == "3:2"
        assert by_id["image_quality"] == "high"
        assert by_id["vision_detail"] == "low"

    def test_empty_settings_are_valid(self, modal):
        # A brand-new user with no stored image prefs must also render a valid modal.
        blocks = modal._build_modal_blocks({}, selected_model="gpt-5.6-sol")
        self._assert_all_initial_options_valid(blocks)


# ============================================================ shape × tier image sizing

def _legal_dimensions(size: str) -> bool:
    """The four envelope rules, spelled out here rather than imported from the image service,
    so a change to that module's own predicate cannot make these assertions vacuously true."""
    w, h = (int(v) for v in size.split("x"))
    return (w % 16 == 0 and h % 16 == 0
            and max(w, h) <= 2560
            and 655360 <= w * h <= 3686400
            and max(w / h, h / w) <= 3.0)


class TestShapeTierGrid:
    """People pick a SHAPE and a SIZE, never pixels. The grid behind those two controls is a
    table of measurements — every landscape/square cell was sent to the live API and returned
    200 — so what these tests defend is that no cell can drift outside what the API accepts."""

    @pytest.fixture
    def modal(self):
        return SettingsModal(db=MagicMock())

    def test_all_14_cells_are_legal_sizes(self):
        from slack_client.settings_modal import _SHAPE_TIER_SIZES
        cells = [(shape, tier, size)
                 for shape, row in _SHAPE_TIER_SIZES.items()
                 for tier, size in row.items()]
        # 7 shapes x 2 tiers; the `max` column was pulled 2026-09-09.
        assert len(cells) == 14
        for shape, tier, size in cells:
            assert _legal_dimensions(size), (shape, tier, size)

    def test_every_cell_is_the_shape_it_claims_to_be(self):
        from slack_client.settings_modal import _SHAPE_TIER_SIZES
        for shape, row in _SHAPE_TIER_SIZES.items():
            a, b = (int(v) for v in shape.split(":"))
            for tier, size in row.items():
                w, h = (int(v) for v in size.split("x"))
                assert abs((w / h) - (a / b)) < 0.06, (shape, tier, size)

    def test_shape_and_tier_round_trip_both_ways(self, modal):
        from slack_client.settings_modal import _IMAGE_SHAPES, _IMAGE_TIERS, _SHAPE_TIER_SIZES
        shapes = [s for s, _ in _IMAGE_SHAPES]
        tiers = [t for t, _ in _IMAGE_TIERS]
        for shape, row in _SHAPE_TIER_SIZES.items():
            for tier, size in row.items():
                assert modal.image_size_for(shape, tier) == size
                assert modal.shape_tier_for(size, shapes, tiers) == (shape, tier)

    def test_shape_auto_stores_the_literal_auto(self, modal):
        # No storage change: `image_size` has always been able to hold "auto", and the tier is
        # ignored while the shape is auto.
        for tier in ("standard", "large"):
            assert modal.image_size_for("auto", tier) == "auto"
        assert modal.shape_tier_for("auto", ["auto", "1:1"], ["standard"]) == ("auto", "standard")

    def test_a_saved_max_tier_renders_as_large_not_the_default(self, modal, monkeypatch):
        """The retired 4K tier maps to Large before the configured default is consulted, the
        same rule `image_service.user_defaults` applies. Under `DEFAULT_IMAGE_TIER=standard`
        the modal used to show Standard while the renderer used Large, so saving the view
        downgraded the render."""
        from slack_client import settings_modal as sm
        monkeypatch.setattr(sm.config, "default_image_tier", "standard")
        assert modal.shape_tier_for(
            "auto", ["auto", "1:1"], ["standard", "large"], "max") == ("auto", "large")
        # gpt-image-1 renders only Standard, so that is what the controls can offer.
        assert modal.shape_tier_for(
            "auto", ["auto", "1:1"], ["standard"], "max") == ("auto", "standard")

    def test_an_off_grid_stored_size_survives_the_modal_open(self, modal):
        # A size the model chose on an earlier turn, or a legacy value. Slack rejects the whole
        # view when initial_option is absent from options, so a synthetic option is injected.
        blocks = modal._build_modal_blocks(
            {"image_model": "gpt-image-2", "image_size": "1808x1008"},
            selected_model="gpt-5.6-sol")
        element = next(e for e in _selects(blocks) if e.get("action_id") == "image_ratio")
        assert element["initial_option"]["value"] == "1808x1008"
        assert "1808" in element["initial_option"]["text"]["text"]
        assert element["options"][0]["value"] == "1808x1008"

    def test_the_synthetic_option_is_the_same_object_in_both_places(self, modal):
        # Slack compares the option objects, so two equal-but-separate dicts is a real risk to
        # guard: it is one object here, not a copy.
        blocks = modal._build_modal_blocks(
            {"image_model": "gpt-image-2", "image_size": "1808x1008"},
            selected_model="gpt-5.6-sol")
        element = next(e for e in _selects(blocks) if e.get("action_id") == "image_ratio")
        assert element["initial_option"] is element["options"][0]

    def test_a_grid_size_carried_onto_gpt_image_1_also_gets_the_synthetic_option(self, modal):
        # gpt-image-1 takes only the three named sizes, so a Widescreen selection made on
        # another model is not producible by the controls it renders — same rule, third case.
        blocks = modal._build_modal_blocks(
            {"image_model": "gpt-image-1", "image_size": "1920x1088"},
            selected_model="gpt-5.6-sol")
        element = next(e for e in _selects(blocks) if e.get("action_id") == "image_ratio")
        assert element["initial_option"] is element["options"][0]
        assert element["initial_option"]["value"] == "1920x1088"

    def test_the_tier_select_is_absent_on_gpt_image_1(self, modal):
        blocks = modal._build_modal_blocks(
            {"image_model": "gpt-image-1"}, selected_model="gpt-5.6-sol")
        ids = {e.get("action_id") for e in _selects(blocks)}
        assert "image_ratio" in ids
        assert "image_tier" not in ids

    @pytest.mark.parametrize("model", ["gpt-image-2", "gpt-image-2.5-flare",
                                       "gpt-image-2.5-sunburst"])
    def test_the_tier_select_is_present_on_every_custom_size_model(self, modal, model):
        blocks = modal._build_modal_blocks({"image_model": model},
                                           selected_model="gpt-5.6-sol")
        ids = {e.get("action_id") for e in _selects(blocks)}
        assert "image_tier" in ids

    @pytest.mark.parametrize("model", ["gpt-image-1", "gpt-image-2", "gpt-image-2.5-flare",
                                       "gpt-image-2.5-sunburst"])
    def test_every_image_model_renders_a_valid_modal(self, modal, model):
        blocks = modal._build_modal_blocks({"image_model": model, "image_size": "auto"},
                                           selected_model="gpt-6-astra")
        for element in _selects(blocks):
            valid = {opt["value"] for opt in element["options"]}
            assert element["initial_option"]["value"] in valid, element.get("action_id")


class TestAstraModalLadder:
    @pytest.fixture
    def modal(self):
        return SettingsModal(db=MagicMock())

    def test_the_personal_effort_ladder_for_astra_excludes_none(self, modal):
        from config import GPT6_EFFORTS
        blocks = modal._add_gpt55_settings({}, "gpt-6-astra")
        block = next(b for b in blocks if b.get("block_id") == "reasoning_block_gpt54")
        values = [o["value"] for o in block["accessory"]["options"]]
        assert values == GPT6_EFFORTS
        assert "none" not in values

    def test_a_stored_none_clamps_to_a_value_the_ladder_offers(self, modal):
        # The submit-time clamp already stops the 400; what this stops is the modal advertising
        # a choice it will silently overwrite.
        blocks = modal._add_gpt55_settings({"reasoning_effort": "none"}, "gpt-6-astra")
        block = next(b for b in blocks if b.get("block_id") == "reasoning_block_gpt54")
        assert block["accessory"]["initial_option"]["value"] == "low"


class TestSolSamplingRoundTrip:
    def test_sol_at_none_keeps_temperature_and_top_p(self):
        """Render -> extract -> validate: Sol has `none`, so the sampling controls it renders
        there must survive the submit instead of being stripped as on Astra."""
        modal = SettingsModal(db=MagicMock())
        blocks = modal._add_gpt55_settings(
            {"reasoning_effort": "none", "temperature": 0.3, "top_p": 0.5}, "gpt-6-sol")
        by_id = {b.get("block_id"): b for b in blocks}
        reasoning = by_id["reasoning_block_gpt54"]["accessory"]["initial_option"]
        temperature = by_id["temperature_block"]["element"]["initial_value"]
        top_p = by_id["top_p_block"]["element"]["initial_value"]

        # Submit exactly what was rendered, so a renderer that dropped the saved values
        # (e.g. back to 1.0) fails here.
        extracted = modal.extract_form_values({"values": {
            "model_block": {"model_select": {"selected_option": {"value": "gpt-6-sol"}}},
            "reasoning_block_gpt54": {"reasoning_level_gpt54": {
                "selected_option": reasoning}},
            "temperature_block": {"temperature": {"value": temperature}},
            "top_p_block": {"top_p": {"value": top_p}},
        }})
        validated = modal.validate_settings(extracted)
        assert validated["reasoning_effort"] == "none"
        assert validated["temperature"] == 0.3
        assert validated["top_p"] == 0.5


class TestFastTierControl:
    """Slack has no disabled form control, so "greyed out" is rendered as no control at all."""

    @pytest.fixture
    def modal(self):
        return SettingsModal(db=MagicMock())

    def _blocks(self, modal, monkeypatch, *, tier, model, settings=None):
        from config import config
        monkeypatch.setattr(config, "openai_service_tier", tier)
        return modal._fast_tier_blocks(settings or {}, model, "global")

    def test_the_checkbox_renders_for_an_eligible_model_when_the_gate_is_open(
            self, modal, monkeypatch):
        blocks = self._blocks(modal, monkeypatch, tier="fast", model="gpt-6-astra")
        assert len(blocks) == 1
        accessory = blocks[0]["accessory"]
        assert blocks[0]["block_id"] == "service_tier_block"
        assert accessory["type"] == "checkboxes"
        assert accessory["action_id"] == "service_tier"
        assert "initial_options" not in accessory          # off by default, array omitted
        assert accessory["options"][0]["value"] == "fast"

    def test_a_saved_opt_in_is_the_same_option_object(self, modal, monkeypatch):
        blocks = self._blocks(modal, monkeypatch, tier="fast", model="gpt-5.6-sol",
                              settings={"service_tier": "fast"})
        accessory = blocks[0]["accessory"]
        assert accessory["initial_options"][0] is accessory["options"][0]

    def test_an_ineligible_model_gets_a_context_line_naming_it(self, modal, monkeypatch):
        blocks = self._blocks(modal, monkeypatch, tier="fast", model="gpt-5.6-luna")
        assert len(blocks) == 1 and blocks[0]["type"] == "context"
        text = blocks[0]["elements"][0]["text"]
        assert "not available on" in text
        # The display name, not the raw model id.
        assert "GPT-5.6 Luna (Fast and affordable)" in text
        assert "gpt-5.6-luna" not in text

    def test_the_admin_gate_wins_over_an_eligible_model(self, modal, monkeypatch):
        blocks = self._blocks(modal, monkeypatch, tier="standard", model="gpt-6-astra")
        assert len(blocks) == 1 and blocks[0]["type"] == "context"
        assert "Disabled by your system administrator." in blocks[0]["elements"][0]["text"]

    def test_the_control_never_appears_in_the_thread_scope(self, modal, monkeypatch):
        # Thread settings are whole-document replacements, so a hidden checkbox there would
        # DELETE a stored opt-in. `service_tier` is personal and never thread-scoped.
        from config import config
        monkeypatch.setattr(config, "openai_service_tier", "fast")
        assert modal._fast_tier_blocks({"service_tier": "fast"}, "gpt-6-astra", "thread") == []


class TestFastTierExtraction:
    @pytest.fixture
    def modal(self):
        return SettingsModal(db=MagicMock())

    def _values(self, selected_options):
        return {"values": {"service_tier_block": {"service_tier": {
            "selected_options": selected_options}}}}

    def test_a_ticked_box_extracts_fast(self, modal):
        extracted = modal.extract_form_values(self._values([{"value": "fast"}]))
        assert extracted["service_tier"] == "fast"

    def test_an_unticked_box_extracts_standard(self, modal):
        extracted = modal.extract_form_values(self._values([]))
        assert extracted["service_tier"] == "standard"

    def test_an_absent_block_writes_nothing(self, modal):
        # Global preference updates are partial, so omission preserves the stored column.
        assert "service_tier" not in modal.extract_form_values({"values": {}})

    def test_validate_drops_the_tier_on_an_ineligible_model(self, modal):
        assert "service_tier" not in modal.validate_settings(
            {"model": "gpt-5.5", "service_tier": "fast"})
        assert modal.validate_settings(
            {"model": "gpt-6-astra", "service_tier": "fast"})["service_tier"] == "fast"


class TestTierUnderAuto:
    """§4.8.5. Auto delegates the SHAPE to the model and nothing else — the size tier stays the
    person's choice, so it has to survive a render and come back out of the form."""

    @pytest.fixture
    def modal(self):
        return SettingsModal(db=MagicMock())

    def _size_selects(self, blocks):
        return {e["action_id"]: e for e in _selects(blocks)
                if e.get("action_id") in ("image_ratio", "image_tier")}

    def _resolution(self, blocks):
        return [el["text"] for b in blocks if b.get("type") == "context"
                for el in b.get("elements", []) if "Resolution:" in el.get("text", "")][0]

    def test_a_saved_tier_is_what_the_select_opens_on_under_auto(self, modal):
        blocks = modal._build_modal_blocks(
            {"image_model": "gpt-image-2.5-sunburst", "image_size": "auto",
             "image_tier": "standard"}, selected_model="gpt-5.6-sol")
        selects = self._size_selects(blocks)
        assert selects["image_ratio"]["initial_option"]["value"] == "auto"
        assert selects["image_tier"]["initial_option"]["value"] == "standard"

    def test_only_two_tiers_are_offered(self, modal):
        """The 4K tier was pulled 2026-09-09 — experimental per OpenAI's guide, and visible
        mesh artifacts at `high`."""
        blocks = modal._build_modal_blocks(
            {"image_model": "gpt-image-2.5-sunburst", "image_size": "auto"},
            selected_model="gpt-5.6-sol")
        values = [o["value"] for o in self._size_selects(blocks)["image_tier"]["options"]]
        assert values == ["standard", "large"]

    def test_a_saved_max_tier_renders_as_large(self, modal):
        # A legacy `max` is not an option any more; it must not become an initial_option
        # outside its own option list, which Slack rejects for the whole view.
        blocks = modal._build_modal_blocks(
            {"image_model": "gpt-image-2.5-sunburst", "image_size": "auto",
             "image_tier": "max"}, selected_model="gpt-5.6-sol")
        assert self._size_selects(blocks)["image_tier"]["initial_option"]["value"] == "large"

    def test_a_saved_4k_size_renders_as_the_synthetic_custom_option(self, modal):
        # 3840x2160 was the 16:9 max cell. With the column pulled it matches no rendered cell,
        # so it falls through to the "Custom: …" option rather than getting a special case.
        blocks = modal._build_modal_blocks(
            {"image_model": "gpt-image-2.5-sunburst", "image_size": "3840x2160",
             "image_tier": "max"}, selected_model="gpt-5.6-sol")
        selects = self._size_selects(blocks)
        assert selects["image_ratio"]["initial_option"]["value"] == "3840x2160"
        assert "Custom" in selects["image_ratio"]["initial_option"]["text"]["text"]
        assert selects["image_tier"]["initial_option"]["value"] == "large"

    def test_auto_with_no_saved_tier_defaults_to_large(self, modal):
        blocks = modal._build_modal_blocks(
            {"image_model": "gpt-image-2.5-sunburst", "image_size": "auto"},
            selected_model="gpt-5.6-sol")
        assert self._size_selects(blocks)["image_tier"]["initial_option"]["value"] == "large"

    def test_the_tier_round_trips_render_to_extract_under_auto(self, modal):
        blocks = modal._build_modal_blocks(
            {"image_model": "gpt-image-2.5-sunburst", "image_size": "auto",
             "image_tier": "standard"}, selected_model="gpt-5.6-sol")
        selects = self._size_selects(blocks)
        extracted = modal.extract_form_values({"values": {
            "image_ratio_block": {"image_ratio": {
                "selected_option": selects["image_ratio"]["initial_option"]}},
            "image_tier_block": {"image_tier": {
                "selected_option": selects["image_tier"]["initial_option"]}},
        }})
        # Storage is unchanged: Auto still stores the literal "auto" in `image_size`.
        assert extracted["image_size"] == "auto"
        assert extracted["image_tier"] == "standard"

    def test_extract_emits_the_tier_even_with_a_concrete_shape(self, modal):
        extracted = modal.extract_form_values({"values": {
            "image_ratio_block": {"image_ratio": {"selected_option": {"value": "16:9"}}},
            "image_tier_block": {"image_tier": {"selected_option": {"value": "large"}}},
        }})
        assert extracted["image_size"] == "1920x1088"
        assert extracted["image_tier"] == "large"

    def test_no_tier_block_writes_no_tier(self, modal):
        # gpt-image-1 renders no tier select; a partial update must not invent a value.
        extracted = modal.extract_form_values({"values": {
            "image_ratio_block": {"image_ratio": {"selected_option": {"value": "1:1"}}}}})
        assert "image_tier" not in extracted

    @pytest.mark.parametrize("tier,expected", [
        ("standard", "_Resolution: chosen per image at Standard (e.g. 1360 × 768 for 16:9)_"),
        ("large", "_Resolution: chosen per image at Large (e.g. 1920 × 1088 for 16:9)_"),
    ])
    def test_the_resolution_line_names_the_tier_under_auto(self, modal, tier, expected):
        blocks = modal._build_modal_blocks(
            {"image_model": "gpt-image-2.5-sunburst", "image_size": "auto",
             "image_tier": tier}, selected_model="gpt-5.6-sol")
        assert self._resolution(blocks) == expected

    def test_a_model_without_tiers_keeps_the_old_auto_line(self, modal):
        # gpt-image-1 has no tier select, so there is no tier to name.
        blocks = modal._build_modal_blocks(
            {"image_model": "gpt-image-1", "image_size": "auto", "image_tier": "max"},
            selected_model="gpt-5.6-sol")
        assert self._resolution(blocks) == "_Resolution: chosen by the model_"

    def test_an_off_grid_size_does_not_reset_the_saved_tier(self, modal):
        # The shape select shows the synthetic custom option; the tier select must still open on
        # what was saved, because submitting the form writes `image_tier` back.
        blocks = modal._build_modal_blocks(
            {"image_model": "gpt-image-2.5-sunburst", "image_size": "1808x1008",
             "image_tier": "standard"}, selected_model="gpt-5.6-sol")
        assert self._size_selects(blocks)["image_tier"]["initial_option"]["value"] \
            == "standard"


class TestImagePickerLabels:
    """The picker rows people read. Slack truncates a static_select row around 30 characters."""

    @pytest.fixture
    def modal(self):
        return SettingsModal(db=MagicMock())

    def test_the_image_model_labels_say_what_to_pick_and_fit(self, modal):
        assert modal._get_image_model_display_name(
            "gpt-image-2.5-sunburst") == "GPT-Image-2.5 Sunburst (Best)"
        assert modal._get_image_model_display_name(
            "gpt-image-2.5-flare") == "GPT-Image-2.5 Flare (Faster)"
        for model in ("gpt-image-2.5-sunburst", "gpt-image-2.5-flare"):
            assert len(modal._get_image_model_display_name(model)) <= 30

    def test_the_quality_labels_carry_the_measured_multipliers(self, modal):
        assert {q: modal._get_image_quality_display(q) for q in
                ("auto", "low", "medium", "high", "xhigh", "max")} == {
            "auto": "Auto (≈ Low)",
            "low": "Low (0.1× cost)",
            "medium": "Medium (0.25× cost)",
            "high": "High (1× · default)",
            "xhigh": "Extra High (2× cost)",
            "max": "Maximum (4× cost)",
        }

    def test_the_rendered_quality_options_use_those_labels(self, modal):
        blocks = modal._build_modal_blocks(
            {"image_model": "gpt-image-2.5-sunburst", "image_quality": "high"},
            selected_model="gpt-5.6-sol")
        element = next(e for e in _selects(blocks) if e.get("action_id") == "image_quality")
        assert element["initial_option"]["text"]["text"] == "High (1× · default)"
        assert [o["text"]["text"] for o in element["options"]] == [
            "Auto (≈ Low)", "Low (0.1× cost)", "Medium (0.25× cost)",
            "High (1× · default)", "Extra High (2× cost)", "Maximum (4× cost)"]
