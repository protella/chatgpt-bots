"""Image settings resolution (message_processor/image_service.py).

The rules this module exists to hold, and what breaks if it stops holding them:

1. **The user's saved settings are what runs.** Model, size, quality, background, format and
   fidelity all come from the person's preferences. No tool schema offers a way to depart from
   them, so ``resolve_settings`` has exactly one source and nothing to reject. The model used to
   be able to change any of them on any call, silently; that is withdrawn.
2. **The legal option space is READ FROM A TABLE, never inferred from the model id.**
   ``"gpt-image-2.5-flare".startswith("gpt-image-2")`` is True, so a prefix test hands 2.5 the
   legacy quality ladder and denies it `xhigh`/`max`, which it accepts. ``IMAGE_CAPS`` states
   each model's capabilities outright and an unknown id gets the most conservative row.
3. **A near-miss size is fitted, not rejected.** The API rejects any side not divisible by 16
   (verified live: 1920x1080 is a hard 400). Rejecting it would fall back to the user's
   default — handing back a SQUARE when a 16:9 image was asked for, a bigger lie than the one
   avoided. Snap to the grid, repair what the snap broke, and say so. An aspect beyond 3:1 is
   squared off to 3:1 rather than refused, for the same reason.
"""
import pytest

from message_processor import image_service as svc


def _cfg(**over):
    """A thread_config with every image preference set explicitly, so these tests assert on
    the resolution rules rather than on whatever the env defaults happen to be."""
    base = {
        "image_model": "gpt-image-2",
        "image_size": "1024x1024",
        "image_quality": "auto",
        "image_background": "auto",
        "image_format": "png",
        "image_compression": 100,
        "input_fidelity": "high",
    }
    base.update(over)
    return base


# ------------------------------------------------------------------ the capability table

def test_the_caps_table_covers_every_supported_image_model():
    from config import SUPPORTED_IMAGE_MODELS
    assert set(svc.IMAGE_CAPS) == set(SUPPORTED_IMAGE_MODELS)


def test_an_unknown_model_gets_the_most_conservative_row():
    # A model we have never probed must never be handed a parameter it rejects, so it inherits
    # gpt-image-1's row: named sizes only, legacy qualities.
    assert svc.caps_for("gpt-image-9-unreleased") == svc.IMAGE_CAPS["gpt-image-1"]
    assert svc.caps_for(None) == svc.IMAGE_CAPS["gpt-image-1"]


@pytest.mark.parametrize("model, expected", [
    ("gpt-image-2.5-flare", ["auto", "low", "medium", "high", "xhigh", "max"]),
    ("gpt-image-2.5-sunburst", ["auto", "low", "medium", "high", "xhigh", "max"]),
    ("gpt-image-2", ["auto", "low", "medium", "high"]),
    ("gpt-image-1", ["auto", "low", "medium", "high"]),
])
def test_qualities_for_returns_one_set_per_model(model, expected):
    assert svc.qualities_for(model) == expected
    # legal_options is the surface that reaches the model as evidence; it must agree.
    assert svc.legal_options(model)["quality"] == expected


@pytest.mark.parametrize("quality", ["xhigh", "max"])
def test_xhigh_and_max_are_legal_on_2_5_and_illegal_on_the_legacy_models(quality):
    assert quality in svc.qualities_for("gpt-image-2.5-flare")
    assert quality in svc.qualities_for("gpt-image-2.5-sunburst")
    assert quality not in svc.qualities_for("gpt-image-2")
    assert quality not in svc.qualities_for("gpt-image-1")


def test_transparent_is_legal_on_every_model():
    # Probed live 2026-09-08: transparent + png returns 200 on gpt-image-2 too. The belief that
    # it did not was wrong, and it cost users the option on what was then the default model.
    for model in svc.IMAGE_CAPS:
        assert svc.backgrounds_for(model) == ["auto", "transparent", "opaque"]


def test_input_fidelity_is_legal_only_on_gpt_image_1():
    assert svc.supports_input_fidelity("gpt-image-1")
    for model in ("gpt-image-2", "gpt-image-2.5-flare", "gpt-image-2.5-sunburst"):
        assert not svc.supports_input_fidelity(model)
        assert svc.legal_options(model)["input_fidelity"] == []


def test_custom_sizes_are_a_capability_read_not_a_prefix_test():
    # The bug this replaced: `.startswith("gpt-image-2")` also matches `gpt-image-2.5-*`.
    assert svc.supports_custom_sizes("gpt-image-2")
    assert svc.supports_custom_sizes("gpt-image-2.5-flare")
    assert svc.supports_custom_sizes("gpt-image-2.5-sunburst")
    assert not svc.supports_custom_sizes("gpt-image-1")


# ------------------------------------------------------------------ the size envelope

def _is_legal(size: str) -> bool:
    """The four envelope rules, spelled out here rather than imported, so a change to the
    module's own predicate cannot make these assertions vacuously true."""
    w, h = (int(v) for v in size.split("x"))
    return (w % 16 == 0 and h % 16 == 0
            and max(w, h) <= 2560
            and 655360 <= w * h <= 3686400
            and max(w / h, h / w) <= 3.0)


@pytest.mark.parametrize("asked, expected", [
    # codex's two snap counterexamples: the independent per-axis snap moved each of these back
    # across a boundary the scale had just satisfied, so the repair loop has to walk it back.
    ("763x344", "1216x544"),        # snapped to 1200x544 = 2,560 pixels UNDER the floor
    ("4000x3000", "2208x1664"),     # snapped to 2224x1664 = 14,336 pixels OVER the ceiling
    # the two regressions from the old min-edge envelope
    ("512x512", "816x816"),         # both edges clear any per-axis minimum and it is still a 400
    ("1920x1920", "1920x1920"),     # legal square: the old caller re-clamped a square to landscape
    # aspect repair: independent rounding pushes an exact 3:1 request past the limit
    ("3000x1000", "2560x864"),      # 2560x848 would be 3.02:1
    ("5000x100", "1408x480"),       # 50:1 is squared off to 3:1 rather than refused
    # shape-preserving fits
    ("3000x3000", "1920x1920"),     # square in, square out (this used to come back landscape)
    ("6497x4373", "2336x1568"),     # over the ceiling -> 1.486:1 in, 1.490:1 out
    ("4000x1400", "2560x896"),      # too wide -> 2.86:1 in, 2.86:1 out
    ("120x360", "480x1408"),        # under the pixel floor -> scaled up, still about 1:3
])
def test_fit_envelope_and_normalize_size_agree_and_land_inside_the_envelope(asked, expected):
    w, h = (int(v) for v in asked.split("x"))
    fw, fh, _fitted = svc._fit_envelope(w, h)
    assert f"{fw}x{fh}" == expected

    # …and through the caller, which is where the legal-square regression actually lived: it
    # used to re-snap _fit_envelope's output against a per-axis height cap.
    usable, _note = svc.normalize_size("gpt-image-2", asked)
    assert usable == expected
    assert _is_legal(expected)


def test_size_snaps_to_the_16px_grid_with_a_note():
    # The most obvious slide size in the world is a 400 from the API. Snap, don't reject.
    usable, note = svc.normalize_size("gpt-image-2", "1920x1080")
    assert usable == "1920x1088"
    assert "1920x1088" in note and "16" in note


def test_size_already_on_the_grid_passes_through_silently():
    usable, note = svc.normalize_size("gpt-image-2", "1536x864")   # 16:9, both sides ÷16
    assert usable == "1536x864" and note is None


def test_an_out_of_envelope_fit_explains_itself():
    _usable, note = svc.normalize_size("gpt-image-2", "6497x4373")
    assert "aspect ratio" in note and "6497x4373" in note


def test_custom_size_rejected_on_gpt_image_1():
    # v1 takes only the named sizes; a WxH there is a 400, so it never reaches the API.
    usable, reason = svc.normalize_size("gpt-image-1", "1536x864")
    assert usable is None and "gpt-image-1" in reason


def test_named_sizes_work_on_every_model():
    for model in svc.IMAGE_CAPS:
        assert svc.normalize_size(model, "1024x1536") == ("1024x1536", None)
        assert svc.normalize_size(model, "auto") == ("auto", None)


def test_garbage_size_is_rejected():
    usable, reason = svc.normalize_size("gpt-image-2", "enormous")
    assert usable is None and "WxH" in reason


# ------------------------------------------------------------------ resolution

def test_resolve_settings_is_exactly_the_user_defaults():
    cfg = _cfg(image_size="1024x1536", image_quality="high", image_format="webp",
               image_compression=80)
    effective, rejected = svc.resolve_settings(cfg)
    assert rejected == []
    assert effective == {"model": "gpt-image-2", "size": "1024x1536", "tier": "large",
                         "quality": "high", "background": "auto", "format": "webp",
                         "compression": 80, "input_fidelity": "high"}


def test_a_supplied_override_changes_nothing():
    # The argument survives only because an unowned caller unpacks the two-tuple. Whatever it
    # carries, the person's saved settings are what runs.
    cfg = _cfg(image_size="1024x1536", image_quality="high")
    assert svc.resolve_settings(cfg) == svc.resolve_settings(
        cfg, {"size": "1536x1024", "quality": "low", "background": "transparent"})


def test_image_model_falls_back_to_config_when_thread_has_none():
    from config import config
    effective, rejected = svc.resolve_settings({})
    assert effective["model"] == config.image_model
    assert rejected == []


def test_png_forces_full_compression():
    # PNG is lossless; carrying a 50 would be a lie in the log and in the tool result.
    effective, _ = svc.resolve_settings(_cfg(image_format="png", image_compression=50))
    assert effective["format"] == "png" and effective["compression"] == 100


def test_lossy_format_keeps_its_compression():
    effective, rejected = svc.resolve_settings(_cfg(image_format="jpeg", image_compression=60))
    assert effective["format"] == "jpeg" and effective["compression"] == 60
    assert rejected == []


# ------------------------------------------------------------------ user_defaults

def test_user_defaults_coerce_a_quality_the_selected_model_cannot_do():
    # A saved `max` from the 2.5 family meeting a legacy model: the ladders differ by model, so
    # the saved value is checked against THIS model's, not a global list.
    assert svc.user_defaults(_cfg(image_model="gpt-image-2", image_quality="max"))["quality"] \
        == "auto"
    assert svc.user_defaults(
        _cfg(image_model="gpt-image-2.5-flare", image_quality="max"))["quality"] == "max"


def test_user_defaults_keep_transparent_on_every_model():
    for model in svc.IMAGE_CAPS:
        assert svc.user_defaults(
            _cfg(image_model=model, image_background="transparent"))["background"] \
            == "transparent"


def test_user_defaults_repair_an_illegal_saved_size():
    # A saved custom size after a switch to gpt-image-1 (which has no custom sizes).
    assert svc.user_defaults(
        _cfg(image_model="gpt-image-1", image_size="1920x1088"))["size"] == "auto"


@pytest.mark.parametrize("saved,expected", [
    ("512x512", "816x816"),        # under the pixel floor: scaled up, aspect kept
    ("763x344", "1216x544"),       # under the floor and off the /16 grid
    ("6497x4373", "2336x1568"),    # over the ceiling: fitted, aspect kept
    ("1920x1920", "1920x1920"),    # already legal: untouched
])
def test_user_defaults_hand_back_the_repaired_size(saved, expected):
    """The repair `normalize_size` computed, not merely its verdict.

    A saved size the repair loop can fix used to be checked for validity and then sent on
    unrepaired — so 512x512 reached the API as 512x512 and 400d, which is the one case
    normalization exists for.
    """
    assert svc.user_defaults(
        _cfg(image_model="gpt-image-2.5-flare", image_size=saved))["size"] == expected


def test_defaults_sentence_names_the_settings_the_call_will_run_with():
    sentence = svc.defaults_sentence(_cfg(image_size="1024x1536", image_quality="high"))
    assert "size=1024x1536" in sentence and "quality=high" in sentence


# ------------------------------------------------------------------ shape x tier

@pytest.mark.parametrize("shape,tier,expected", [
    ("16:9", "standard", "1360x768"),
    ("16:9", "large", "1920x1088"),
    ("1:1", "large", "1440x1440"),
    ("1:3", "standard", "592x1776"),
    # The 4K tier was pulled 2026-09-09 (experimental per OpenAI, mesh artifacts at `high`).
    # A legacy saved `max` is an unknown tier now, so it renders at Large.
    ("16:9", "max", "1920x1088"),
    ("1:3", "max", "832x2496"),
])
def test_size_for_shape_reads_the_grid(shape, tier, expected):
    assert svc.size_for_shape(shape, tier) == expected


def test_size_for_shape_falls_back_on_junk():
    # An unknown tier still has a shape to render, so it takes the shipped tier. An unknown
    # shape has no cell at all, and `auto` hands the choice back to the API.
    assert svc.size_for_shape("16:9", "enormous") == "1920x1088"
    assert svc.size_for_shape("4:5", "large") == "auto"


def test_user_defaults_carry_the_saved_tier():
    assert svc.user_defaults(_cfg(image_tier="standard"))["tier"] == "standard"


def test_the_4k_tier_is_no_longer_offered():
    """Pulled 2026-09-09: experimental per OpenAI's guide, and visible mesh artifacts at
    `high`. The cells are kept, commented out, next to the table, plus LEGACY_MAX_SIZES so
    the migration can still map a stored 4K size onto its Large cell."""
    assert svc.TIERS == ("standard", "large")
    assert all("max" not in row for row in svc.SHAPE_TIER_SIZES.values())
    assert svc.LEGACY_MAX_SIZES["16:9"] == "3840x2160"
    assert set(svc.LEGACY_MAX_SIZES) == set(svc.SHAPE_TIER_SIZES)


def test_a_legacy_max_tier_coerces_to_large():
    assert svc.user_defaults(_cfg(image_tier="max"))["tier"] == "large"


def test_a_legacy_max_tier_beats_a_standard_default(monkeypatch):
    """`max` is the retired 4K tier, not a typo: it maps to `large` BEFORE the configured
    default is consulted. With `DEFAULT_IMAGE_TIER=standard` the old fallback collapsed it to
    Standard here while the settings modal still showed Large, and saving the modal wrote the
    downgrade back."""
    from config import config
    monkeypatch.setattr(config, "default_image_tier", "standard")
    assert svc.user_defaults(_cfg(image_tier="max"))["tier"] == "large"
    # A default of `max` takes the same mapping rather than falling through to the hard-coded
    # `large`, and an unusable tier still lands on the (coerced) default.
    monkeypatch.setattr(config, "default_image_tier", "max")
    assert svc.user_defaults(_cfg(image_tier="gigantic"))["tier"] == "large"


def test_a_4k_request_is_fitted_to_the_cap_with_its_aspect_kept():
    # 3840x2160 is 16:9; so is 2560x1440. The pull narrows the envelope; it does not refuse.
    usable, note = svc.normalize_size("gpt-image-2.5-sunburst", "3840x2160")
    assert usable == "2560x1440"
    assert note and "2560x1440" in note


def test_user_defaults_coerce_an_unusable_tier():
    from config import config
    assert svc.user_defaults(_cfg(image_tier="gigantic"))["tier"] \
        == config.default_image_tier
    assert svc.user_defaults(_cfg())["tier"] == config.default_image_tier


def test_the_settings_sentence_says_the_shape_is_delegated_under_auto():
    """`size=auto` in a prompt reads as "the API decides everything". Under a saved Auto the
    person delegated exactly one thing — the shape — so the line says that and names the tier
    it renders at."""
    sentence = svc.defaults_sentence(_cfg(image_size="auto", image_tier="standard"))
    assert "size=auto" not in sentence
    assert "shape=chosen by you per request via the aspect argument" in sentence
    assert "rendered at the standard tier" in sentence
    assert "quality=" in sentence
