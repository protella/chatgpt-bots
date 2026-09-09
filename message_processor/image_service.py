"""Image settings resolution and the option space the model is allowed to choose from (F34).

The old handlers each resolved image settings themselves, slightly differently (the
background job silently dropped format/compression, the edit path passed them). This module
is the single owner of "what settings does this image call actually run with", so the image
tools, the detached job, and any future background agent all answer that question the same way.

Two rules the rest of the system depends on:

1. **The user's saved image settings are what runs.** The image MODEL comes from the user's
   preference in ``thread_config``, and so does every other setting — size, quality,
   background, format, compression, fidelity. No tool schema offers the model a way to depart
   from them, so it cannot change what the person asked for, silently or otherwise.

2. **The legal option space depends on the selected model, and is READ FROM A TABLE.** It used
   to be inferred from the model id with ``startswith("gpt-image-2")``, which also matches
   ``gpt-image-2.5-*`` — so 2.5 inherited gpt-image-2's quality ladder and would have been
   denied ``xhigh``/``max``, which it accepts. ``IMAGE_CAPS`` states each model's capabilities
   outright; an id that is not in it falls back to the most conservative entry, so a model we
   have never probed can never be handed a parameter it rejects.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from config import config
from logger import setup_logger

logger = setup_logger(name="slack_bot.ImageService")

# Named sizes every image model accepts. Models with `custom_sizes` also take arbitrary WxH.
NAMED_SIZES: List[str] = ["1024x1024", "1024x1536", "1536x1024", "auto"]

# The shape × size-tier grid behind the settings modal's two selects AND the call-time resolution
# of a model-chosen shape (saved shape `auto` + saved tier). Every cell was sent to the live API
# and returned 200 (2026-09-08); the odd-looking ones (1088, 1184, 1776) are the nearest 16px grid
# points to the true ratio inside the pixel envelope. Measurements, not a formula — do not
# recompute.
SHAPE_TIER_SIZES: Dict[str, Dict[str, str]] = {
    "1:1":  {"standard": "1024x1024", "large": "1440x1440"},
    "3:2":  {"standard": "1536x1024", "large": "1776x1184"},
    "2:3":  {"standard": "1024x1536", "large": "1184x1776"},
    "16:9": {"standard": "1360x768",  "large": "1920x1088"},
    "9:16": {"standard": "768x1360",  "large": "1088x1920"},
    "3:1":  {"standard": "1776x592",  "large": "2496x832"},
    "1:3":  {"standard": "592x1776",  "large": "832x2496"},
}
# The `max` (4K) tier, PULLED 2026-09-09 by the owner. OpenAI's image guide says "Resolutions
# above 2560x1440 are experimental"
# (https://developers.openai.com/api/docs/guides/image-generation), and at those sizes
# `quality=high` renders visible mesh artifacts (measured 2026-09-09: 3840x2160 at `high` fills
# foliage and rock with a woven mesh; `xhigh` is mostly clean). Restore these cells — and
# `"max"` in TIERS, the modal row, and the custom-size envelope below — when the docs drop the
# experimental label. The values are exact and were all probed 200 on 2026-09-08:
#
#     "1:1":  {..., "max": "2880x2880"},
#     "3:2":  {..., "max": "3520x2352"},
#     "2:3":  {..., "max": "2352x3520"},
#     "16:9": {..., "max": "3840x2160"},
#     "9:16": {..., "max": "2160x3840"},
#     "3:1":  {..., "max": "3840x1280"},
#     "1:3":  {..., "max": "1280x3840"},
#
# Those cells are still stored in some users' `image_size`, so the one-time migration has to be
# able to map them to their Large cell. It reads them from here rather than from the live table.
LEGACY_MAX_SIZES: Dict[str, str] = {
    "1:1":  "2880x2880",
    "3:2":  "3520x2352",
    "2:3":  "2352x3520",
    "16:9": "3840x2160",
    "9:16": "2160x3840",
    "3:1":  "3840x1280",
    "1:3":  "1280x3840",
}
SHAPES: Tuple[str, ...] = tuple(SHAPE_TIER_SIZES)
TIERS: Tuple[str, ...] = ("standard", "large")
FORMATS: List[str] = ["png", "jpeg", "webp"]
FIDELITIES: List[str] = ["low", "high"]

QUALITIES_LEGACY: Tuple[str, ...] = ("auto", "low", "medium", "high")
QUALITIES_25: Tuple[str, ...] = ("auto", "low", "medium", "high", "xhigh", "max")
# Verified live 2026-09-08: transparent + png returns 200 on every supported model, including
# gpt-image-2. The "gpt-image-2 has no transparent background" belief that used to live here
# was wrong, and cost users the option on the default model.
ALL_BACKGROUNDS: Tuple[str, ...] = ("auto", "transparent", "opaque")


@dataclass(frozen=True)
class ImageCaps:
    """What one image model actually accepts, as probed against the live API."""

    qualities: Tuple[str, ...]
    backgrounds: Tuple[str, ...]
    supports_input_fidelity: bool
    custom_sizes: bool


IMAGE_CAPS: Dict[str, ImageCaps] = {
    "gpt-image-2.5-flare": ImageCaps(QUALITIES_25, ALL_BACKGROUNDS, False, True),
    "gpt-image-2.5-sunburst": ImageCaps(QUALITIES_25, ALL_BACKGROUNDS, False, True),
    "gpt-image-2": ImageCaps(QUALITIES_LEGACY, ALL_BACKGROUNDS, False, True),
    "gpt-image-1": ImageCaps(QUALITIES_LEGACY, ALL_BACKGROUNDS, True, False),
}

_FALLBACK_CAPS = IMAGE_CAPS["gpt-image-1"]

_WXH_RE = re.compile(r"^(\d{2,4})x(\d{2,4})$")

# Custom-size envelope, verified live 2026-09-08. Every rule is a whole-image rule: the edge
# cap is on the LONGEST edge (1920x1920 passes, 2880x1152 does not), and the floor and ceiling
# are on total pixels, not on either side. There is no minimum edge — 512x512 clears any
# per-axis minimum you could invent and is still rejected, for being under the pixel floor.
#
# The ceiling is the OWNER's cap, not the API's: OpenAI's image guide says "Resolutions above
# 2560x1440 are experimental"
# (https://developers.openai.com/api/docs/guides/image-generation), and at those sizes
# `quality=high` renders visible mesh artifacts (measured 2026-09-09). The API still accepts the
# old, larger envelope — `_MAX_EDGE = 3840`, `_MAX_PIXELS = 8294400` — so restore those two
# values, and the `max` tier above, when the docs drop the experimental label. Anything a caller
# asks for above the cap is fitted down with its aspect intact, not refused.
_SIZE_STEP = 16
_MAX_EDGE = 2560
_MIN_PIXELS = 655360
_MAX_PIXELS = 3686400  # 2560 * 1440
_MAX_ASPECT = 3.0
# The snap in step 3 can move the pixel count across a boundary by a few thousand pixels; the
# repair walks it back one grid step at a time. Eight steps is far more than any real request
# needs (the observed worst case is one), and it bounds a loop whose output is billed.
_REPAIR_STEPS = 8

_COMPRESSION_RULE = "an integer 0-100 (jpeg/webp only; png is always 100)"
_V1_SIZE_RULE = f"one of {', '.join(NAMED_SIZES)}"
_CUSTOM_SIZE_LIMITS = (
    f"both sides divisible by 16, the longest edge at most {_MAX_EDGE}px, between "
    f"{_MIN_PIXELS:,} and {_MAX_PIXELS:,} total pixels, and no more extreme than 3:1")
_V2_SIZE_RULE = f"one of {', '.join(NAMED_SIZES)}, or a custom WxH with {_CUSTOM_SIZE_LIMITS}"


def caps_for(model_id: Optional[str]) -> ImageCaps:
    """The capability row for an image model; the conservative row for anything unknown."""
    return IMAGE_CAPS.get(str(model_id or ""), _FALLBACK_CAPS)


def qualities_for(model_id: Optional[str]) -> List[str]:
    """Every quality this model accepts. The 2.5 family adds `xhigh` and `max`."""
    return list(caps_for(model_id).qualities)


def supports_custom_sizes(model_id: Optional[str]) -> bool:
    """True when the model takes arbitrary WxH sizes as well as the named ones.

    This was ``is_v2``, a prefix test — and `"gpt-image-2.5-flare".startswith("gpt-image-2")`
    is True, which is how 2.5 came to inherit gpt-image-2's limits. It is a capability read
    now, and named for the capability rather than for a family it no longer identifies.
    """
    return caps_for(model_id).custom_sizes


def backgrounds_for(model_id: Optional[str]) -> List[str]:
    """Every model accepts all three backgrounds, transparent included (probed 2026-09-08)."""
    return list(caps_for(model_id).backgrounds)


def supports_input_fidelity(model_id: Optional[str]) -> bool:
    """Only gpt-image-1 takes `input_fidelity`; the 2.x models 400 on it, so it is omitted."""
    return caps_for(model_id).supports_input_fidelity


def legal_options(model_id: Optional[str]) -> Dict[str, Any]:
    """The PINNED option allowlist for an image model.

    On the channel surface the schema is a static superset — every option is advertised for
    every model — so the option space is no longer expressed by the schema and this is the only
    place that says what a given model will actually accept. It reaches the model as this
    turn's evidence (``settings_evidence_lines``).
    """
    caps = caps_for(model_id)
    return {
        "size": list(NAMED_SIZES),
        "size_rule": _V2_SIZE_RULE if caps.custom_sizes else _V1_SIZE_RULE,
        "quality": list(caps.qualities),
        "background": list(caps.backgrounds),
        "format": list(FORMATS),
        "compression_rule": _COMPRESSION_RULE,
        "input_fidelity": list(FIDELITIES) if caps.supports_input_fidelity else [],
    }


def _legal_dimensions(w: int, h: int) -> bool:
    """All four envelope rules, together. This is what the API itself enforces."""
    if w <= 0 or h <= 0:
        return False
    if w % _SIZE_STEP or h % _SIZE_STEP:
        return False
    if max(w, h) > _MAX_EDGE:
        return False
    if not (_MIN_PIXELS <= w * h <= _MAX_PIXELS):
        return False
    return max(w / h, h / w) <= _MAX_ASPECT


def _named_fallback(w: int, h: int) -> Tuple[int, int]:
    """The named size matching a request's orientation — the guaranteed-legal last resort."""
    if w > h:
        return 1536, 1024
    if h > w:
        return 1024, 1536
    return 1024, 1024


def _fit_envelope(w: int, h: int) -> Tuple[int, int, bool]:
    """Fit a request into the API's envelope, PRESERVING its shape as far as the rules allow.

    Clamping the two sides independently changes the aspect ratio: 3000x3000 (a legal shape)
    used to come back 3008x2160 — a square request answered with a landscape image. That is
    the same lie this module exists to prevent, so the fit is proportional throughout.

    The five steps, in order:

    1. clamp an aspect beyond 3:1 by shortening the longer side (a shape that extreme cannot
       be honored at all, so it is squared off rather than refused);
    2. scale proportionally into the pixel budget and under the longest-edge cap;
    3. snap both sides to the 16px grid the API requires;
    4. repair, because snapping each axis independently moves both the pixel count and the
       aspect, and can push either back over a boundary — 763x344 snaps to 1200x544, which is
       2,560 pixels UNDER the floor; 4000x3000 snaps to 2224x1664, which is 14,336 OVER the
       ceiling; and 3000x1000 snaps to 2560x848, which is 3.02:1. Each step moves one axis by
       16 (so the grid rule survives), normally the longer one because that shifts the pixel
       count fastest, the shorter one where the shape is against the 3:1 limit;
    5. validate. This function's output goes straight to a paid API call, so a result that
       still breaks a rule falls back to a named size rather than being sent.

    Returns ``(w, h, fitted)`` where ``fitted`` says the request was outside the envelope, so
    the note can explain WHY rather than blaming the 16px grid for it.
    """
    fw, fh = float(w), float(h)

    # 1. aspect
    if fw / fh > _MAX_ASPECT:
        fw = fh * _MAX_ASPECT
    elif fh / fw > _MAX_ASPECT:
        fh = fw * _MAX_ASPECT

    # 2. pixel budget, then the longest-edge cap (which can only shrink further)
    pixels = fw * fh
    if pixels > _MAX_PIXELS:
        fw, fh = fw * math.sqrt(_MAX_PIXELS / pixels), fh * math.sqrt(_MAX_PIXELS / pixels)
    elif pixels < _MIN_PIXELS:
        fw, fh = fw * math.sqrt(_MIN_PIXELS / pixels), fh * math.sqrt(_MIN_PIXELS / pixels)
    longest = max(fw, fh)
    if longest > _MAX_EDGE:
        fw, fh = fw * (_MAX_EDGE / longest), fh * (_MAX_EDGE / longest)

    fitted = (round(fw), round(fh)) != (w, h)

    # 3. grid
    iw = max(_SIZE_STEP, int(round(fw / _SIZE_STEP)) * _SIZE_STEP)
    ih = max(_SIZE_STEP, int(round(fh / _SIZE_STEP)) * _SIZE_STEP)

    # 4. repair
    for _ in range(_REPAIR_STEPS):
        area = iw * ih
        aspect = max(iw / ih, ih / iw)
        if area > _MAX_PIXELS or max(iw, ih) > _MAX_EDGE:
            # Shrinking the LONGER axis sheds pixels fastest and moves the shape toward
            # square, so it can never break the aspect rule on the way.
            if iw >= ih:
                iw = max(_SIZE_STEP, iw - _SIZE_STEP)
            else:
                ih = max(_SIZE_STEP, ih - _SIZE_STEP)
        elif area < _MIN_PIXELS or aspect > _MAX_ASPECT:
            # Growing the LONGER axis buys pixels fastest, but at the 3:1 limit it is the
            # SHORTER axis that has to grow: it buys pixels AND pulls the shape back inside,
            # and it is the only move that can undo an over-3:1 snap (independent rounding of
            # 3000x1000 lands on 2560x848, which is 3.02:1 and a 400).
            grow_longer = (area < _MIN_PIXELS and aspect <= _MAX_ASPECT
                           and (max(iw, ih) + _SIZE_STEP) / min(iw, ih) <= _MAX_ASPECT)
            grow_w = (iw >= ih) if grow_longer else (iw < ih)
            if grow_w:
                iw += _SIZE_STEP
            else:
                ih += _SIZE_STEP
        else:
            break

    # 5. guarantee
    if not _legal_dimensions(iw, ih):
        logger.warning(f"Size {w}x{h} could not be fitted into the envelope "
                       f"(reached {iw}x{ih}); falling back to a named size")
        iw, ih = _named_fallback(w, h)
        fitted = True
    return iw, ih, fitted


def normalize_size(model_id: Optional[str], size: str) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(usable_size, note)``, or ``(None, reason)`` if the size cannot be honored.

    Verified live against the API (2026-07-12): "Width and height must both be divisible by
    16" — so 1920x1080, the most obvious slide size in the world, is a hard 400. Rather than
    reject it and fall back to the user's default (silently handing back a SQUARE when a
    16:9 image was asked for — a much bigger lie than the one we'd be avoiding), snap it to
    the nearest legal grid point: 1920x1080 -> 1920x1088, still 16:9 to within a pixel. The
    caller reports the snap, so the model is never misled about what it got.

    A request outside the size envelope is fitted with its aspect ratio intact
    (6497x4373 -> 2336x1568), for the same reason: the shape is the part of the ask that
    matters, and honoring the shape is what "adjusted" is allowed to mean. An aspect beyond
    3:1 is squared off to 3:1 by the same logic rather than refused.

    Nothing may re-clamp what ``_fit_envelope`` returns. It already snaps, repairs and
    validates against all four rules; a second pass over its output is how a legal square
    used to come back landscape (2880x2880 -> 2880x2160, under the older, larger envelope).
    """
    if size in NAMED_SIZES:
        return size, None
    m = _WXH_RE.match(size or "")
    if not m:
        return None, f"size={size!r} is not one of {NAMED_SIZES} or a WxH like 1536x864"
    if not supports_custom_sizes(model_id):
        return None, f"size={size!r}: {model_id} accepts only {NAMED_SIZES}"

    w, h = int(m.group(1)), int(m.group(2))
    if w <= 0 or h <= 0:
        return None, f"size={size!r} has a zero side"

    fw, fh, fitted = _fit_envelope(w, h)
    snapped = f"{fw}x{fh}"
    if snapped == size:
        return snapped, None
    if fitted:
        return snapped, (
            f"size {size} adjusted to {snapped}: it is outside what this model renders "
            f"({_CUSTOM_SIZE_LIMITS}), so it was fitted to the nearest legal size with the "
            "same aspect ratio")
    return snapped, (f"size {size} adjusted to {snapped} (each side must be divisible by 16)")


def size_for_shape(shape: str, tier: str) -> str:
    """The WxH one grid cell holds, for a model-chosen shape at the user's saved tier.

    This is the call-time half of shape × tier: the modal resolves the pair on submit, and
    when the saved shape is Auto there is nothing to resolve until the model names a shape in
    its ``aspect`` argument. An unknown tier falls back to ``large`` (the shipped default) —
    which is now where a legacy saved ``max`` lands, the 4K tier having been pulled; an unknown
    shape has no cell at all, so it falls back to ``auto`` and the API picks.
    """
    row = SHAPE_TIER_SIZES.get(shape)
    if not row:
        return "auto"
    return row.get(tier) or row["large"]


def image_model_for(thread_config: Optional[Dict[str, Any]]) -> str:
    """The user's selected image model — the one hard constraint. Never model-supplied."""
    return (thread_config or {}).get("image_model") or config.image_model


def user_defaults(thread_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The user's saved image preferences, coerced to values the selected model accepts.

    These are what every image call runs with, and what the tool description advertises so the
    model knows what the picture it is asking for will look like.
    """
    cfg = thread_config or {}
    model = image_model_for(cfg)

    # The REPAIRED size, not merely a yes/no on the saved one. `normalize_size` snaps a
    # saved 512x512 to 816x816 and fits a saved 6497x4373 to 3488x2368; keeping only the
    # boolean sent the unrepaired string straight to the API, which is a 400 for exactly the
    # values normalization exists to rescue. Only a size it cannot repair falls back.
    size = cfg.get("image_size") or config.default_image_size
    size = normalize_size(model, size)[0] or "auto"

    quality = cfg.get("image_quality") or config.default_image_quality
    if quality not in qualities_for(model):
        # A saved `xhigh`/`max` meeting a legacy model, for instance: the ladders differ by
        # model, so the saved value is checked against THIS model's, not a global list.
        quality = "auto"

    background = cfg.get("image_background") or config.default_image_background
    if background not in backgrounds_for(model):
        background = "auto"

    fmt = cfg.get("image_format") or config.default_image_format
    if fmt not in FORMATS:
        fmt = "png"

    compression = cfg.get("image_compression")
    if compression is None:
        compression = config.default_image_compression
    try:
        compression = max(0, min(100, int(compression)))
    except (TypeError, ValueError):
        compression = 100

    fidelity = cfg.get("input_fidelity") or config.default_input_fidelity
    if fidelity not in FIDELITIES:
        fidelity = "high"

    # The size TIER, which only matters while `size` is "auto": the shape is chosen per
    # request (by the model, via `aspect`) and rendered at this tier. It is never sent to the
    # API on its own — `size_for_shape` turns the pair into a WxH at call time.
    # A saved tier outside TIERS falls back, but `max` is NOT a typo — it is the retired 4K
    # tier, and the person who saved it asked for the largest render there was. It maps to
    # `large` BEFORE any default is consulted, so a `DEFAULT_IMAGE_TIER=standard` deployment
    # cannot silently downgrade it (the settings modal applies the same rule, and the two
    # disagreeing is what made a save downgrade the render). Anything else outside TIERS takes
    # the configured default, itself put through the same `max` mapping.
    tier = cfg.get("image_tier") or config.default_image_tier
    if tier == "max":
        tier = "large"
    if tier not in TIERS:
        default_tier = config.default_image_tier
        if default_tier == "max":
            default_tier = "large"
        tier = default_tier if default_tier in TIERS else "large"

    return {
        "size": size,
        "tier": tier,
        "quality": quality,
        "background": background,
        "format": fmt,
        "compression": compression,
        "input_fidelity": fidelity,
    }


def resolve_settings(
    thread_config: Optional[Dict[str, Any]],
    overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """The settings an image call runs with: the user's saved values, and nothing else.

    Returns ``(effective, rejected)``. ``rejected`` is retained as an empty list because
    callers unpack a two-tuple; there is nothing left to reject. The model used to be able to
    depart from the user's saved size/quality/background/format/fidelity on any call, silently
    — that is withdrawn. No tool schema carries settings any more, so ``overrides`` is accepted
    for signature compatibility and ignored: a caller that supplies one changes nothing.
    """
    del overrides  # settings come from the user, not from a caller
    effective = user_defaults(thread_config)
    effective["model"] = image_model_for(thread_config)

    # PNG is always full-quality; carrying a lower number would be a lie in the log.
    if effective["format"] == "png":
        effective["compression"] = 100

    return effective, []


def _size_phrase(d: Dict[str, Any]) -> str:
    """How the size half of a settings sentence reads.

    A saved WxH is stated outright. A saved `auto` is the one thing the person HAS delegated:
    the model names the shape per request and we render it at their saved tier, so the line
    says that rather than the word "auto", which reads as "the API decides everything".
    """
    if d["size"] == "auto":
        return ("shape=chosen by you per request via the aspect argument, rendered at the "
                f"{d['tier']} tier")
    return f"size={d['size']}"


def defaults_sentence(thread_config: Optional[Dict[str, Any]]) -> str:
    """One line naming the user's saved settings, for the tool description.

    The model cannot change them, but it does need to know what the image it is about to ask
    for will look like — a 3:2 print at low quality is a different answer than a 16:9 render.
    """
    d = user_defaults(thread_config)
    return (f"{_size_phrase(d)}, quality={d['quality']}, "
            f"background={d['background']}, format={d['format']}")


SETTINGS_EVIDENCE_HEADER = "Image settings in force:"


def settings_evidence_lines(thread_config: Optional[Dict[str, Any]] = None) -> List[str]:
    """The image-settings half of the channel turn's tool-evidence block.

    The static schema no longer carries the model's option space or the requester's saved
    settings, so both have to reach the model as evidence instead: what model will run, what it
    will legally accept, and the settings the call will actually use. None of it is selectable
    — it is what the picture will be.
    """
    model = image_model_for(thread_config)
    legal = legal_options(model)
    d = user_defaults(thread_config)
    fidelity = ", ".join(legal["input_fidelity"]) or "auto-handled (not selectable)"
    return [
        SETTINGS_EVIDENCE_HEADER,
        f"image model: {model} (fixed by settings — not selectable in a tool call)",
        f"legal size: {legal['size_rule']}",
        f"legal quality: {', '.join(legal['quality'])}",
        f"legal background: {', '.join(legal['background'])}",
        f"legal format: {', '.join(legal['format'])}",
        f"legal compression: {legal['compression_rule']}",
        f"legal input_fidelity: {fidelity}",
        (f"settings this image call will run with: {_size_phrase(d)}, "
         f"quality={d['quality']}, background={d['background']}, format={d['format']}, "
         f"compression={d['compression']}, input_fidelity={d['input_fidelity']}"),
    ]
