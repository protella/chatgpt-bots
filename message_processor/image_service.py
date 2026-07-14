"""Image settings resolution and the option space the model is allowed to choose from (F34).

The old handlers each resolved image settings themselves, slightly differently (the
background job silently dropped format/compression, the edit path passed them). This module
is the single owner of "what settings does this image call actually run with", so the image
tools, the detached job, and any future background agent all answer that question the same way.

Two rules the rest of the system depends on:

1. **The image MODEL is a hard constraint, never model-selectable.** It comes from the user's
   saved preference in ``thread_config`` and is not present in any tool schema, so the model
   cannot express a different one even by accident. Everything else is a *preference*: the
   user's saved value is the default, and the model may override it when the task genuinely
   calls for something else (a 16:9 title slide, an opaque background for a deck).

2. **The legal option space depends on the selected model.** gpt-image-2 has no transparent
   background and auto-handles input fidelity, so those values must not be advertised when
   it is selected — a schema that offers `transparent` and then silently coerces it to `auto`
   teaches the model a lie about what it asked for.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from config import config
from logger import setup_logger

logger = setup_logger(name="slack_bot.ImageService")

# Named sizes both model families accept. gpt-image-2 additionally accepts arbitrary WxH.
NAMED_SIZES: List[str] = ["1024x1024", "1024x1536", "1536x1024", "auto"]
QUALITIES: List[str] = ["auto", "low", "medium", "high"]
FORMATS: List[str] = ["png", "jpeg", "webp"]
FIDELITIES: List[str] = ["low", "high"]

_WXH_RE = re.compile(r"^(\d{2,4})x(\d{2,4})$")

# gpt-image-2 free-size envelope (from the API's own limits).
_V2_MAX_W, _V2_MAX_H = 3840, 2160
_V2_MIN_EDGE = 256
_V2_STEP = 16
_V2_MAX_ASPECT = 3.0


def is_v2(model_id: Optional[str]) -> bool:
    """gpt-image-2 family — a different param surface than gpt-image-1."""
    return bool(model_id) and str(model_id).startswith("gpt-image-2")


def backgrounds_for(model_id: Optional[str]) -> List[str]:
    """gpt-image-2 cannot do transparent, so it must not be offered."""
    return ["auto", "opaque"] if is_v2(model_id) else ["auto", "transparent", "opaque"]


def supports_input_fidelity(model_id: Optional[str]) -> bool:
    """gpt-image-2 auto-handles fidelity; the param is omitted from its edit calls."""
    return not is_v2(model_id)


def _snap(value: int, lo: int, hi: int) -> int:
    """Nearest legal grid point: the API rejects any side not divisible by 16."""
    value = max(lo, min(hi, value))
    snapped = int(round(value / _V2_STEP)) * _V2_STEP
    return max(lo, min(hi, snapped))


def _fit_envelope(w: int, h: int) -> Tuple[int, int, bool]:
    """Scale a too-big / too-small request into the API's envelope, PRESERVING its shape.

    Clamping the two sides independently changes the aspect ratio: 3000x3000 (legal shape,
    illegal height) would come back 3008x2160 — a square request answered with a landscape
    image. That is the same lie this module exists to prevent, so scale proportionally
    instead and let the grid snap take it from there (a ≤8px nudge per side).

    Returns ``(w, h, scaled)`` where ``scaled`` says whether the request was outside the
    envelope, so the note can explain WHY rather than blaming the 16px grid for it.
    """
    scale = 1.0
    if w > _V2_MAX_W or h > _V2_MAX_H:
        scale = min(_V2_MAX_W / w, _V2_MAX_H / h)
    elif w < _V2_MIN_EDGE or h < _V2_MIN_EDGE:
        scale = max(_V2_MIN_EDGE / w, _V2_MIN_EDGE / h)
    if scale == 1.0:
        return w, h, False
    return max(1, int(round(w * scale))), max(1, int(round(h * scale))), True


def normalize_size(model_id: Optional[str], size: str) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(usable_size, note)``, or ``(None, reason)`` if the size cannot be honored.

    Verified live against the API (2026-07-12): "Width and height must both be divisible by
    16" — so 1920x1080, the most obvious slide size in the world, is a hard 400. Rather than
    reject it and fall back to the user's default (silently handing back a SQUARE when a
    16:9 image was asked for — a much bigger lie than the one we'd be avoiding), snap it to
    the nearest legal grid point: 1920x1080 -> 1920x1088, still 16:9 to within a pixel. The
    caller reports the snap, so the model is never misled about what it got.

    A request outside the size envelope is scaled to fit with its aspect ratio intact
    (3000x3000 -> 2160x2160), for the same reason: the shape is the part of the ask that
    matters, and honoring the shape is what "adjusted" is allowed to mean.

    An aspect ratio beyond 3:1 cannot be fixed by scaling or snapping, so that is a real
    rejection.
    """
    if size in NAMED_SIZES:
        return size, None
    m = _WXH_RE.match(size or "")
    if not m:
        return None, f"size={size!r} is not one of {NAMED_SIZES} or a WxH like 1536x864"
    if not is_v2(model_id):
        return None, f"size={size!r}: {model_id} accepts only {NAMED_SIZES}"

    w, h = int(m.group(1)), int(m.group(2))
    if max(w / h, h / w) > _V2_MAX_ASPECT:
        return None, f"size={size!r} is more extreme than the 3:1 aspect-ratio limit"

    fw, fh, scaled = _fit_envelope(w, h)
    sw = _snap(fw, _V2_MIN_EDGE, _V2_MAX_W)
    sh = _snap(fh, _V2_MIN_EDGE, _V2_MAX_H)
    snapped = f"{sw}x{sh}"
    if snapped == size:
        return snapped, None
    if scaled:
        return snapped, (
            f"size {size} adjusted to {snapped}: it is outside the {_V2_MIN_EDGE}px-"
            f"{_V2_MAX_W}x{_V2_MAX_H} range, so it was scaled to fit with the same aspect "
            "ratio (each side must also be divisible by 16)")
    return snapped, (f"size {size} adjusted to {snapped} (each side must be divisible by 16)")


def valid_size(model_id: Optional[str], size: str) -> bool:
    """True when the size is usable as-is or after snapping."""
    return normalize_size(model_id, size)[0] is not None


def image_model_for(thread_config: Optional[Dict[str, Any]]) -> str:
    """The user's selected image model — the one hard constraint. Never model-supplied."""
    return (thread_config or {}).get("image_model") or config.image_model


def user_defaults(thread_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The user's saved image preferences, coerced to values the selected model accepts.

    These are what a tool call runs with when the model supplies no overrides — and what the
    tool description advertises, so the model knows what it is departing from.
    """
    cfg = thread_config or {}
    model = image_model_for(cfg)

    size = cfg.get("image_size") or config.default_image_size
    if not valid_size(model, size):
        size = "auto"

    quality = cfg.get("image_quality") or config.default_image_quality
    if quality not in QUALITIES:
        quality = "auto"

    background = cfg.get("image_background") or config.default_image_background
    if background not in backgrounds_for(model):
        # The user saved `transparent` and then switched to gpt-image-2: honor the intent as
        # far as the model allows rather than failing the call.
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

    return {
        "size": size,
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
    """Fold the model's overrides onto the user's defaults.

    Returns ``(effective, rejected)``. An override the selected model cannot honor is
    DROPPED (the user's default stands) and named in ``rejected`` so the caller can tell the
    model what it did not get — silently coercing it would leave the model believing it got
    a transparent background it never received.

    ``model`` in the result is always the user's, whatever the overrides say.
    """
    model = image_model_for(thread_config)
    effective = user_defaults(thread_config)
    effective["model"] = model

    rejected: List[str] = []
    if not isinstance(overrides, dict):
        return effective, rejected

    # A model that tries to pick the image model is ignored, loudly. The schema does not
    # offer the field, so this only fires on a model going off-script.
    for forbidden in ("model", "image_model"):
        if forbidden in overrides:
            rejected.append(
                f"{forbidden}: the image model is fixed by the user's settings ({model})")

    if "size" in overrides:
        usable, note = normalize_size(model, str(overrides["size"]))
        if usable:
            effective["size"] = usable
            if note:  # snapped to the 16px grid — say so rather than quietly resizing
                rejected.append(note)
        else:
            rejected.append(note or "size is not valid")

    if "quality" in overrides:
        quality = str(overrides["quality"])
        if quality in QUALITIES:
            effective["quality"] = quality
        else:
            rejected.append(f"quality={quality!r} is not one of {QUALITIES}")

    if "background" in overrides:
        background = str(overrides["background"])
        allowed = backgrounds_for(model)
        if background in allowed:
            effective["background"] = background
        else:
            rejected.append(f"background={background!r} is not supported by {model}")

    if "format" in overrides:
        fmt = str(overrides["format"])
        if fmt in FORMATS:
            effective["format"] = fmt
        else:
            rejected.append(f"format={fmt!r} is not one of {FORMATS}")

    if "compression" in overrides:
        try:
            comp = int(overrides["compression"])
        except (TypeError, ValueError):
            rejected.append("compression must be an integer 0-100")
        else:
            if 0 <= comp <= 100:
                effective["compression"] = comp
            else:
                rejected.append("compression must be between 0 and 100")

    if "input_fidelity" in overrides:
        fidelity = str(overrides["input_fidelity"])
        if not supports_input_fidelity(model):
            rejected.append(f"input_fidelity is auto-handled by {model}")
        elif fidelity in FIDELITIES:
            effective["input_fidelity"] = fidelity
        else:
            rejected.append(f"input_fidelity={fidelity!r} is not one of {FIDELITIES}")

    # PNG is always full-quality; carrying a lower number would be a lie in the log.
    if effective["format"] == "png":
        effective["compression"] = 100

    if rejected:
        logger.info(f"Image overrides rejected: {rejected}")
    return effective, rejected


def defaults_sentence(thread_config: Optional[Dict[str, Any]]) -> str:
    """One line naming the user's saved defaults, for the tool description.

    The model needs to know what it is departing from before it can decide the task warrants
    departing — a bare "overrides" object with no stated baseline invites gratuitous choices.
    """
    d = user_defaults(thread_config)
    return (f"size={d['size']}, quality={d['quality']}, "
            f"background={d['background']}, format={d['format']}")
