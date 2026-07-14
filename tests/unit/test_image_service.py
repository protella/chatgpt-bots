"""F34 — image settings resolution (message_processor/image_service.py).

The rules this module exists to hold, and what breaks if it stops holding them:

1. **The image MODEL is the user's, always.** It is a saved preference, it is not in any tool
   schema, and an override naming a different one is ignored and REPORTED. A model that could
   pick its own image model could silently spend the user's money on a model they rejected.
2. **An override the selected model cannot honor is DROPPED, not coerced.** gpt-image-2 has no
   transparent background. Quietly turning `transparent` into `auto` and reporting success
   teaches the model it got a cutout it never received — it then builds a composite on that
   lie. The user's saved value stands and the attempt comes back in `rejected`.
3. **A near-miss size is snapped, not rejected.** The API rejects any side not divisible by 16
   (verified live: 1920x1080 is a hard 400). Rejecting it would fall back to the user's
   default — handing back a SQUARE when a 16:9 image was asked for, a bigger lie than the one
   avoided. Snap to the grid and say so. An impossible aspect ratio is still a real rejection.
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


def _mentions(rejected, needle):
    return any(needle in note for note in rejected)


# ------------------------------------------------------------------ the model is not selectable

@pytest.mark.parametrize("key", ["model", "image_model"])
def test_override_cannot_change_the_image_model(key):
    # The user chose gpt-image-1. The model asks for gpt-image-2 (by either spelling).
    effective, rejected = svc.resolve_settings(
        _cfg(image_model="gpt-image-1"), {key: "gpt-image-2", "quality": "high"})

    assert effective["model"] == "gpt-image-1"        # the user's choice, untouched
    assert _mentions(rejected, key)                    # …and the attempt is reported back
    assert _mentions(rejected, "fixed by the user's settings")
    assert effective["quality"] == "high"              # a legitimate override still lands


def test_image_model_falls_back_to_config_when_thread_has_none():
    from config import config
    effective, rejected = svc.resolve_settings({}, None)
    assert effective["model"] == config.image_model
    assert rejected == []


# ------------------------------------------------------------------ background: dropped, not coerced

def test_v2_transparent_override_is_rejected_and_user_default_stands():
    # gpt-image-2 cannot do transparent. The user's saved `opaque` must survive — coercing to
    # `auto` would be inventing a third value nobody asked for.
    effective, rejected = svc.resolve_settings(
        _cfg(image_model="gpt-image-2", image_background="opaque"),
        {"background": "transparent"})

    assert effective["background"] == "opaque"
    assert _mentions(rejected, "transparent")
    assert _mentions(rejected, "gpt-image-2")


def test_v1_transparent_override_is_accepted():
    effective, rejected = svc.resolve_settings(
        _cfg(image_model="gpt-image-1"), {"background": "transparent"})
    assert effective["background"] == "transparent"
    assert rejected == []


def test_backgrounds_offered_depend_on_the_model():
    assert svc.backgrounds_for("gpt-image-2") == ["auto", "opaque"]
    assert "transparent" in svc.backgrounds_for("gpt-image-1")
    assert svc.is_v2("gpt-image-2") and not svc.is_v2("gpt-image-1")


# ------------------------------------------------------------------ size snapping

def test_size_snaps_to_the_16px_grid_with_a_note():
    # The most obvious slide size in the world is a 400 from the API. Snap, don't reject.
    usable, note = svc.normalize_size("gpt-image-2", "1920x1080")
    assert usable == "1920x1088"
    assert "1920x1088" in note and "16" in note

    effective, rejected = svc.resolve_settings(_cfg(), {"size": "1920x1080"})
    assert effective["size"] == "1920x1088"       # honored, not dropped to the 1024 default
    assert _mentions(rejected, "1920x1088")       # and the snap is disclosed


def test_size_already_on_the_grid_passes_through_silently():
    usable, note = svc.normalize_size("gpt-image-2", "1536x864")   # 16:9, both sides ÷16
    assert usable == "1536x864" and note is None

    effective, rejected = svc.resolve_settings(_cfg(), {"size": "1536x864"})
    assert effective["size"] == "1536x864" and rejected == []


def test_extreme_aspect_ratio_is_a_real_rejection():
    # Snapping cannot fix 50:1 — this one has to fail, and the user's default must survive.
    usable, reason = svc.normalize_size("gpt-image-2", "5000x100")
    assert usable is None and "3:1" in reason

    effective, rejected = svc.resolve_settings(_cfg(image_size="1024x1536"), {"size": "5000x100"})
    assert effective["size"] == "1024x1536"
    assert _mentions(rejected, "3:1")


def test_custom_size_rejected_on_gpt_image_1():
    # v1 takes only the named sizes; a WxH there is a 400, so it never reaches the API.
    usable, reason = svc.normalize_size("gpt-image-1", "1536x864")
    assert usable is None and "gpt-image-1" in reason

    effective, rejected = svc.resolve_settings(
        _cfg(image_model="gpt-image-1"), {"size": "1536x864"})
    assert effective["size"] == "1024x1024"       # the user's saved size stands
    assert _mentions(rejected, "1536x864")


@pytest.mark.parametrize("asked, expected", [
    ("3000x3000", "2160x2160"),   # square, too tall → still SQUARE (was 3008x2160: landscape)
    ("4000x1400", "3840x1344"),   # too wide → 2.86:1 in, 2.86:1 out
    ("120x360", "256x768"),       # below the 256px minimum edge → scaled up, still 1:3
])
def test_out_of_envelope_sizes_keep_their_shape(asked, expected):
    # Clamping the sides independently would answer a square request with a landscape image —
    # the exact lie the snapping rule exists to avoid. Scale proportionally instead.
    usable, note = svc.normalize_size("gpt-image-2", asked)
    assert usable == expected

    aw, ah = (int(v) for v in asked.split("x"))
    ew, eh = (int(v) for v in expected.split("x"))
    assert abs((aw / ah) - (ew / eh)) < 0.02      # the shape survives the fit
    assert "aspect ratio" in note                  # …and the note explains the real reason


def test_named_sizes_work_on_both_families():
    for model in ("gpt-image-1", "gpt-image-2"):
        assert svc.normalize_size(model, "1024x1536") == ("1024x1536", None)
        assert svc.valid_size(model, "auto")


def test_garbage_size_is_rejected():
    usable, reason = svc.normalize_size("gpt-image-2", "enormous")
    assert usable is None and "WxH" in reason


# ------------------------------------------------------------------ format / compression

def test_png_forces_full_compression():
    # PNG is lossless; carrying a 50 would be a lie in the log and in the tool result.
    effective, _ = svc.resolve_settings(
        _cfg(image_format="jpeg", image_compression=60),
        {"format": "png", "compression": 50})
    assert effective["format"] == "png" and effective["compression"] == 100


def test_lossy_format_keeps_its_compression():
    effective, rejected = svc.resolve_settings(
        _cfg(), {"format": "jpeg", "compression": 60})
    assert effective["format"] == "jpeg" and effective["compression"] == 60
    assert rejected == []


@pytest.mark.parametrize("bad", [200, -5, "high"])
def test_out_of_range_compression_is_rejected(bad):
    effective, rejected = svc.resolve_settings(_cfg(image_format="jpeg"), {"compression": bad})
    assert effective["compression"] == 100        # the user's saved value
    assert _mentions(rejected, "compression")


def test_unknown_quality_and_format_are_rejected():
    effective, rejected = svc.resolve_settings(
        _cfg(), {"quality": "ultra", "format": "tiff"})
    assert effective["quality"] == "auto" and effective["format"] == "png"
    assert _mentions(rejected, "ultra") and _mentions(rejected, "tiff")


# ------------------------------------------------------------------ input_fidelity

def test_input_fidelity_rejected_on_v2_accepted_on_v1():
    _, rejected = svc.resolve_settings(_cfg(image_model="gpt-image-2"),
                                       {"input_fidelity": "low"})
    assert _mentions(rejected, "auto-handled")
    assert not svc.supports_input_fidelity("gpt-image-2")

    effective, rejected = svc.resolve_settings(_cfg(image_model="gpt-image-1"),
                                               {"input_fidelity": "low"})
    assert effective["input_fidelity"] == "low" and rejected == []


# ------------------------------------------------------------------ user_defaults

def test_user_defaults_coerce_saved_transparent_to_auto_on_v2():
    # The user saved `transparent` under gpt-image-1 and then switched models. The saved pref
    # is now illegal — honor the intent as far as the model allows instead of failing calls.
    assert svc.user_defaults(
        _cfg(image_model="gpt-image-2", image_background="transparent"))["background"] == "auto"
    # …and it survives untouched on the model that supports it.
    assert svc.user_defaults(
        _cfg(image_model="gpt-image-1", image_background="transparent"))["background"] == "transparent"


def test_user_defaults_repair_an_illegal_saved_size():
    # A saved custom size after a switch to gpt-image-1 (which has no custom sizes).
    assert svc.user_defaults(
        _cfg(image_model="gpt-image-1", image_size="1920x1088"))["size"] == "auto"


def test_no_overrides_returns_exactly_the_user_defaults():
    cfg = _cfg(image_size="1024x1536", image_quality="high", image_format="webp",
               image_compression=80)
    effective, rejected = svc.resolve_settings(cfg, None)
    assert rejected == []
    assert effective == {"model": "gpt-image-2", "size": "1024x1536", "quality": "high",
                         "background": "auto", "format": "webp", "compression": 80,
                         "input_fidelity": "high"}


def test_defaults_sentence_names_what_the_model_is_departing_from():
    sentence = svc.defaults_sentence(_cfg(image_size="1024x1536", image_quality="high"))
    assert "size=1024x1536" in sentence and "quality=high" in sentence
