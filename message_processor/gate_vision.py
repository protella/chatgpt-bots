"""F40 — the wake gate looks at the picture.

The gate is a cheap text classifier that decides whether the bot wakes at all. It never saw
attached images: all it got was a filename. So a meme captioned ":dogkek:" earned a :joy:
reaction that the model had reasoned its way to from the emoji SHORTCODE — "A laughing reaction
fits the playful meme post" — without ever looking at the picture. Reacting to an image you
haven't seen is the same dishonesty as an 👀 with no work behind it: the caption is not the post.

Design notes worth keeping:

* NOT gated on "thin text". A long caption ("this is exactly what prod does every Friday") is
  just as meaningless without the image, and a text-only first pass would preserve the very bug
  we're fixing — that pass answered *confidently*. If an image is there and we can safely show
  it, the gate sees it.
* Nothing downloads until the message SURVIVES the debounce. A superseded burst must not spend
  bandwidth fetching pictures for a verdict that gets thrown away.
* Cost is controlled by CAPS, not cleverness: `detail: low`, a couple of images, a hard byte
  ceiling, a short timeout. Raising reasoning effort would buy no visual resolution at all.
* Failure NEVER means silence. Every failure path degrades to a text-only judgment that is TOLD
  the image was unavailable — because a gate that goes quiet on an unreadable file is a gate
  that drops wakes.
* Bytes live in memory and die here. Nothing touches disk; no base64 is ever persisted.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple

from config import config
from image_validation import API_IMAGE_MIMETYPES, sniff_image_mimetype
from logger import setup_logger

logger = setup_logger(name="slack_bot.GateVision")

# This runs on the DEBOUNCE HOT PATH, in front of every ambient message that carries a picture.
# The byte ceiling below is a cap on what we SEND, not on what we fetch: the downloader buffers
# the whole body before we ever see its length, and it carries a 30s timeout of its own. Two
# images downloaded in sequence could therefore sit on a gate for ~a minute. So: an explicit
# deadline per image, and a hard ceiling on how many gates may be fetching at once — base64
# inflates bytes by 4/3 and Bolt happily runs channels concurrently, so "a few big memes at
# once" is a real memory shape, not a hypothetical.
_DOWNLOAD_TIMEOUT_S = 8.0
_MAX_CONCURRENT_GATE_FETCHES = 2
_fetch_slots = asyncio.Semaphore(_MAX_CONCURRENT_GATE_FETCHES)

# What we are willing to put in front of the model: what the API accepts, MINUS gif.
#
# F50 unified the API-supported list into image_validation, but deliberately did NOT unify this
# one away. The gate's narrowness is a choice, not an oversight: the API takes a static gif
# happily, but telling static from animated means parsing the file, and this runs on the DEBOUNCE
# HOT PATH in front of every ambient message that carries a picture. A wake verdict is not worth
# that, and a gif skipped here is not a dropped wake — it degrades to `unavailable`, and the
# classifier is TOLD it couldn't see the image rather than left to invent it from the filename.
# (SVG and HEIC need no exclusion here; the API doesn't take them either.)
_SAFE_MIMETYPES = API_IMAGE_MIMETYPES - {"image/gif"}

VISIBLE = "visible"
UNAVAILABLE = "unavailable"
NONE = "none"


_VALID_DETAIL = {"low", "high", "auto"}


def _detail() -> str:
    """A typo in GATE_VISION_DETAIL would 400 every image call — and then trip the text-only
    retry on every single one. Normalize here; an unrecognized value is just `low`."""
    d = str(getattr(config, "gate_vision_detail", "low") or "low").strip().lower()
    return d if d in _VALID_DETAIL else "low"


def _sniff(raw: bytes) -> Optional[str]:
    """What arrived, or None if it isn't something this gate will show.

    Slack serves a LOGIN PAGE (HTML, HTTP 200) when auth is wrong, so "it downloaded fine"
    proves nothing — the shared sniffer answers what the bytes ACTUALLY are. Re-checking the
    answer against _SAFE_MIMETYPES is what keeps the gif exclusion honest: `eligible()` screened
    on the DECLARED mimetype, so a gif labelled image/png reaches this point.
    """
    mime = sniff_image_mimetype(raw)
    return mime if mime in _SAFE_MIMETYPES else None


def eligible(descriptors: Any) -> List[Dict]:
    """The images we would be willing to show the model, before any download."""
    out: List[Dict] = []
    for d in (descriptors or []):
        d = d or {}
        mime = str(d.get("mimetype") or "").lower()
        if mime not in _SAFE_MIMETYPES or not d.get("url"):
            continue
        # The DECLARED size, checked before a single byte is fetched.
        size = d.get("size")
        if isinstance(size, int) and size > config.gate_vision_max_bytes:
            logger.debug(f"Gate vision: skipping {d.get('name')} — {size}B over the cap")
            continue
        out.append(d)
    return out


async def load_for_gate(client: Any, descriptors: Any) -> Tuple[List[Dict], str, List[Dict]]:
    """Fetch up to N images and return (input_image parts, status, shown descriptors).

    status is one of: `visible` (the model can see them), `unavailable` (images are attached but
    we could not show them — the prompt must say so, so the model doesn't invent their content),
    `none` (no images at all). Never raises: a gate that throws is a gate that goes silent.

    `shown` is the ORIGINAL descriptor for each image that actually reached the model, in the same
    order as `parts` — the API parts are anonymized to {type, image_url, detail}, so this parallel
    list is how a caller maps a per-image observation back to its Slack file id (F51b piggyback).
    """
    if not getattr(config, "enable_multimodal_gate", True):
        return [], NONE, []
    candidates = eligible(descriptors)
    if not candidates:
        # Attached-but-unshowable (a PDF, an SVG, an oversized photo) still has to be declared:
        # the model must not fill the gap from the filename, which is exactly what it did.
        return [], (UNAVAILABLE if descriptors else NONE), []

    import base64

    cap = max(1, int(config.gate_vision_max_images or 1))
    budget = int(config.gate_vision_max_bytes)     # a TOTAL ceiling, not just a per-image one
    parts: List[Dict] = []
    shown: List[Dict] = []                          # descriptor per part, same order (F51b)

    async with _fetch_slots:
        for d in candidates[:cap]:
            raw = None
            try:
                raw = await asyncio.wait_for(
                    client.download_file(d.get("url"), d.get("id")),
                    timeout=_DOWNLOAD_TIMEOUT_S)
            except asyncio.TimeoutError:
                logger.debug(f"Gate vision: {d.get('name')} took too long — judging without it")
            except Exception as e:  # noqa: BLE001 — never fatal; degrade to text-only
                logger.debug(f"Gate vision: download failed for {d.get('name')}: {e}")
            if not raw:
                continue
            if len(raw) > budget:
                logger.debug(f"Gate vision: {d.get('name')} is over the remaining budget "
                             f"({len(raw)}B > {budget}B)")
                continue
            mime = _sniff(raw)
            if not mime:
                # An HTML login page dressed as a PNG, or something we don't recognise.
                logger.debug(f"Gate vision: {d.get('name')} is not a recognisable image — skipped")
                continue
            budget -= len(raw)
            parts.append({
                "type": "input_image",
                "image_url": f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}",
                "detail": _detail(),
            })
            shown.append(d)

    if not parts:
        return [], UNAVAILABLE, []
    logger.debug(f"Gate vision: showing the classifier {len(parts)} image(s)")
    return parts, VISIBLE, shown
