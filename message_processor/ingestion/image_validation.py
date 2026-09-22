"""One source of truth for what the OpenAI API will accept as an image.

Three lists used to disagree: `image_url_handler.SUPPORTED_IMAGE_MIMETYPES` (consulted only on
the URL path), the wake gate's own narrower allowlist, and — on the attachment path —
nothing at all. So an `image/heic` dragged into Slack was base64'd straight into the request and
took the WHOLE turn down with a 400: the user's message just failed, with no notice explaining
why. Anything that decides "can this picture ride an API call?" asks this module now.

The declared mimetype is NOT evidence. Slack labels a file from its name, browsers lie, and Slack
itself serves an HTML login page (HTTP 200) when auth is wrong — so "it downloaded and says
image/png" tells you nothing about what the bytes are. We sniff the bytes and use what we find:
that both rejects the unsupported and CORRECTS the merely mislabeled, so a JPEG named .png stops
being a 400 and starts being a picture.

The API's supported set, quoted from its own 400, is exactly:
    ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
BMP, TIFF, HEIC and SVG are hard 400s (verified live against gpt-5.6-luna, 2026-07) — but the ones
Pillow can decode (BMP, TIFF, ICO, ...) are no longer refused: `ensure_api_compatible` transcodes
them to PNG in memory (F50b), so only genuinely undecodable bytes (corrupt, or HEIC/SVG with no
decoder installed) reach a rejection. `validate_image_bytes` still answers the narrower question of
what the API accepts AS-IS, unchanged. GIF is
accepted — and, verified across gpt-5.6-sol/terra/luna and gpt-5.5, an ANIMATED gif is accepted
too: the model renders its first frame. That contradicts the F50 spec, which asserted animated
gifs 400 the turn; they do not, so we accept them by default rather than falsely refuse an image
the bot can actually read. The detection is retained behind `REJECT_ANIMATED_GIFS` (a one-line
flip) for whoever wants first-frame-only gifs turned away instead. Bytes stay in memory and die
here — nothing touches disk.
"""

from __future__ import annotations

import math
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

from logger import setup_logger

# Under the app's `slack_bot.*` hierarchy so a resize actually reaches app.log; a bare
# `getLogger(__name__)` has no handler there and its INFO line is silently dropped.
logger = setup_logger(name="slack_bot.ImageValidation")

# The mimetypes the API accepts, as DECLARED labels — for cheap pre-download screening only
# (URL content-type checks and the like). `image/jpg` is not a real mimetype, but
# Slack and half the web send it anyway, so it earns its place as an alias here. Never treat
# membership as proof of anything; only `validate_image_bytes` proves.
API_IMAGE_MIMETYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/gif",
    "image/webp",
}

# File extensions that MAY be one of the above. Used for URL/path guesses before any bytes
# exist; the bytes still get the final word.
API_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

# The CANONICAL mimetypes the Responses *vision* API accepts as-is (what `sniff_image_mimetype`
# can ever return — `image/jpg` is never produced). This is the acceptance set for
# `ensure_api_compatible`.
VISION_MIMETYPES = frozenset({"image/png", "image/jpeg", "image/gif", "image/webp"})

# The Images *edit* endpoint is STRICTER than vision: it takes png/jpeg/webp only — NOT gif
# (verified against its own 400). A GIF source therefore has to be transcoded to PNG (first
# frame) before it can be edited, even though vision would have read it directly.
IMAGE_EDIT_MIMETYPES = frozenset({"image/png", "image/jpeg", "image/webp"})

# The API renders an animated gif's first frame rather than rejecting it (verified live across
# the whole model family), so the default is to let it through. Flip to True to turn animated
# gifs away — e.g. if first-frame-only is judged more confusing than helpful. When True, a gif
# whose animation state can't be determined is ALSO rejected (as UNREADABLE), because the point
# of turning it away is to avoid a surprise, and an undetermined gif is a surprise.
REJECT_ANIMATED_GIFS = False

# Magic bytes -> the CANONICAL mimetype we will actually send. Every value here is a member of
# API_IMAGE_MIMETYPES, and `image/jpg` is deliberately never produced.
_MAGIC: Tuple[Tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"RIFF", "image/webp"),        # RIFF....WEBP — confirmed below
)

# Why a picture was turned away. These are the keys; `rejection_text` renders them for humans.
UNREADABLE = "unreadable_image"
ANIMATED_GIF = "animated_gif"
# A source that decoded fine but ballooned past the byte ceiling once transcoded (e.g. a highly
# compressed TIFF/GIF expanding into a huge PNG). Enforced by callers, not by ensure_compatible.
TOO_LARGE_AFTER_CONVERSION = "too_large_after_conversion"
# A vision image over the API's patch budget whose frame is ALSO past `_MAX_TRANSCODE_PIXELS`, so
# it is refused from the header rather than decoded to be shrunk.
TOO_LARGE_TO_RESIZE = "too_large_to_resize"

_REJECTION_TEXT = {
    UNREADABLE: ("isn't in a format I can read — I can look at PNG, JPEG, GIF and WebP images. "
                 "(HEIC photos straight from an iPhone are a common culprit; re-saving as PNG "
                 "or JPEG works.)"),
    ANIMATED_GIF: "is an animated GIF, which I can't look at — a static image works.",
    TOO_LARGE_AFTER_CONVERSION: ("is too large to edit once converted to a supported format — "
                                 "a smaller or already-PNG/JPEG image works."),
    TOO_LARGE_TO_RESIZE: ("is too large for me to look at — its pixel dimensions are far over "
                          "the limit. A smaller version works."),
}


def rejection_text(reason: Optional[str]) -> str:
    """Human-readable half of a rejection, for the failed-files notice."""
    return _REJECTION_TEXT.get(reason or "", _REJECTION_TEXT[UNREADABLE])


def sniff_image_mimetype(raw: bytes) -> Optional[str]:
    """The canonical mimetype these bytes actually are, or None if we don't recognise them.

    Format only — says nothing about whether a GIF is animated. Callers sending to the API want
    `validate_image_bytes` instead.
    """
    if not raw:
        return None
    for magic, mime in _MAGIC:
        if raw.startswith(magic):
            # RIFF is a container: it fronts WAV and AVI too, so the WEBP tag has to be there.
            if mime == "image/webp" and raw[8:12] != b"WEBP":
                return None
            return mime
    return None


def _gif_is_animated(raw: bytes) -> Optional[bool]:
    """True/False, or None when we genuinely cannot tell.

    Pillow's `is_animated` seeks exactly ONE frame ahead rather than counting them all, so this
    stays cheap enough for a per-upload check and never decodes pixel data. None means the file
    defeated the parser — callers must treat that as a reject, because the alternative is
    gambling the user's whole turn on a 400.
    """
    try:
        from PIL import Image

        with Image.open(BytesIO(raw)) as img:
            return bool(getattr(img, "is_animated", False))
    except Exception:  # noqa: BLE001 — a truncated/hostile GIF is a reject, not a crash
        return None


def _decodes_as_image(raw: bytes, max_pixels: Optional[int] = None) -> bool:
    """True only if Pillow can actually PARSE these bytes as an image.

    A magic-byte prefix is not proof: "PNG signature + junk" matches the prefix but is not a
    real PNG, and sending it still 400s the whole turn — the exact failure this module exists to
    prevent. Pillow's `verify()` checks structural integrity without a full decode, but it leaves
    the file object spent, so the image must be REOPENED before `load()` (Pillow's documented
    verify-then-reopen requirement). We do both: verify catches truncation/corruption cheaply,
    load forces the decoder far enough to reject signature-plus-junk.

    `max_pixels`, when set, is a decompression-bomb ceiling read from the HEADER — `im.size` is
    known as soon as the file is opened, so an oversized frame is refused before `verify()` or
    `load()` can spend the memory to decode it. None (the default) checks nothing.
    """
    try:
        from PIL import Image

        with Image.open(BytesIO(raw)) as im:
            if max_pixels is not None:
                width, height = im.size
                if width * height > max_pixels:
                    return False
            im.verify()
        with Image.open(BytesIO(raw)) as im2:
            im2.load()
        return True
    except Exception:  # noqa: BLE001 — any parse failure means these bytes are not a real image
        return False


def validate_image_bytes(raw: bytes,
                         max_pixels: Optional[int] = None) -> Tuple[Optional[str], Optional[str]]:
    """Decide whether these bytes may ride an API call.

    Returns `(mimetype, None)` on success — the SNIFFED mimetype, which the caller should send
    in place of whatever was declared — or `(None, reason)` on rejection, where reason is one of
    UNREADABLE / ANIMATED_GIF.

    Two gates: the magic-byte sniff decides the FORMAT (and that it is one the API accepts), then
    Pillow PARSES the bytes so a valid-looking prefix followed by garbage is rejected here rather
    than by a 400 mid-turn.

    `max_pixels` bounds the DECODED frame for callers holding bytes from somewhere hostile (a
    web import): a sub-10MB PNG can decode to hundreds of megabytes, and this refuses it from
    the header before any decode happens. Omitting it leaves every existing caller unchanged.
    """
    mime = sniff_image_mimetype(raw)
    if not mime:
        return None, UNREADABLE
    if mime == "image/gif" and REJECT_ANIMATED_GIFS:
        # Off by default — the API accepts animated gifs. See REJECT_ANIMATED_GIFS.
        animated = _gif_is_animated(raw)
        if animated is None:
            return None, UNREADABLE
        if animated:
            return None, ANIMATED_GIF
    if not _decodes_as_image(raw, max_pixels):
        return None, UNREADABLE
    return mime, None


# A decoded frame past this many pixels is refused rather than transcoded: re-encoding a
# decompression-bomb-sized image spends memory to produce something the API would reject anyway.
# Stricter than Pillow's own DecompressionBombError ceiling, which fires far higher.
_MAX_TRANSCODE_PIXELS = 50_000_000


def _transcode_to_png(raw: bytes) -> Optional[bytes]:
    """Decode `raw` with Pillow and re-encode it as PNG in memory, or None on any failure.

    For formats the API will not take but Pillow can still fully read (BMP, TIFF, ICO, PPM, PCX,
    TGA — whatever the installed Pillow supports; no new deps). Modes are coerced to something PNG
    can hold: palette/`LA`/`PA` with transparency and native `RGBA` become RGBA (alpha preserved),
    everything else (`CMYK`, `YCbCr`, `I;16`, `L`, `1`, palette without transparency, ...) becomes
    RGB. Multi-frame files (TIFF/ICO) contribute their FIRST frame only. Bytes stay in memory.
    """
    try:
        from PIL import Image

        with Image.open(BytesIO(raw)) as im:
            im.seek(0)  # first frame of a multi-frame TIFF/ICO; a no-op for single-frame files
            width, height = im.size
            if width * height > _MAX_TRANSCODE_PIXELS:
                return None
            mode = im.mode
            has_alpha = mode in ("RGBA", "LA", "PA") or (
                mode == "P" and "transparency" in im.info)
            if has_alpha:
                converted = im.convert("RGBA")
            elif mode == "RGB":
                converted = im
            else:
                converted = im.convert("RGB")
            out = BytesIO()
            converted.save(out, format="PNG")
        return out.getvalue()
    except Exception:  # noqa: BLE001 — any decode/convert/encode failure is a graceful rejection
        return None


def ensure_compatible(
    raw: bytes, *, allowed: "frozenset[str]" = VISION_MIMETYPES
) -> Tuple[Optional[bytes], Optional[str]]:
    """Return image bytes a given endpoint will accept, transcoding in memory when it has to.

    `allowed` is the set of CANONICAL sniffed mimetypes that endpoint takes as-is (a subset of
    what `sniff_image_mimetype` can return). Vision takes all four (`VISION_MIMETYPES`, the
    default); the Images edit endpoint takes only png/jpeg/webp (`IMAGE_EDIT_MIMETYPES`), so a
    GIF handed here for editing is transcoded rather than passed through.

    Success is `(bytes, mimetype)`:
      - bytes ALREADY in an `allowed` format -> the ORIGINAL bytes, untouched (no re-encode, no
        copy), with their sniffed mimetype.
      - bytes Pillow can fully decode but the endpoint won't take as-is (a recognized format
        outside `allowed` such as GIF for editing, or an unrecognized-signature BMP/TIFF/ICO/...)
        -> a freshly encoded PNG (first frame) and "image/png".

    Failure is `(None, reason)` — UNREADABLE for corrupt/truncated/undecodable bytes, or
    ANIMATED_GIF when REJECT_ANIMATED_GIFS turned a real animated gif away and gif is in `allowed`.
    This never raises and never 400s the endpoint; the caller runs its own graceful rejection.
    """
    sniffed = sniff_image_mimetype(raw)
    if sniffed is not None and sniffed in allowed:
        # In-format candidate. It still has to actually PARSE — a valid signature followed by
        # junk matches the prefix but 400s the endpoint, so `validate_image_bytes` is the gate.
        mime, reason = validate_image_bytes(raw)
        if mime:
            return raw, mime
        # Recognized signature but broken (junk after a PNG header) or a deliberately-rejected
        # member (an animated gif under the opt-in flag). Transcoding those would be wrong, so
        # the rejection stands with its original reason.
        return None, reason
    if sniffed is not None:
        # A recognized format the endpoint won't take as-is (e.g. GIF for editing). Pillow reads
        # it fine — transcode its first frame to PNG.
        png = _transcode_to_png(raw)
        return (png, "image/png") if png is not None else (None, UNREADABLE)
    # Unrecognised signature: Pillow may still decode it (BMP/TIFF/ICO/...). Transcode to PNG, or
    # fall through to the same honest rejection as before on any failure.
    png = _transcode_to_png(raw)
    return (png, "image/png") if png is not None else (None, UNREADABLE)


# The vision API's own pixel budget, quoted from its 400: "The image you provided requires 31570
# patches after processing, exceeding the limit of 30000. Please resize the image and try again."
# A patch is a 32x32 tile, and at our detail setting (auto/omitted) the gpt-6/gpt-5.6 models keep an
# image's pixel dimensions, so the count is ceil(w/32) * ceil(h/32) — the incident's 6560x4928
# phone photo is 205 * 154 = 31570, exactly the number in that 400. These are the API's numbers,
# not ours. (gpt-5.5 downscales on its own, so shrinking before it is harmless.)
_VISION_PATCH_SIZE = 32
_VISION_MAX_PATCHES = 30_000


def _vision_patches(width: int, height: int) -> int:
    """How many 32-px patches the vision API counts for a frame of this size."""
    return math.ceil(width / _VISION_PATCH_SIZE) * math.ceil(height / _VISION_PATCH_SIZE)


def _fit_to_patch_budget(width: int, height: int) -> Tuple[int, int]:
    """The LARGEST aspect-preserving (w, h) whose patch count fits `_VISION_MAX_PATCHES`.

    A plain sqrt-ratio scale is not enough: flooring 6560x4928 by sqrt(30000/31570) gives
    6394x4803, which is 200 * 151 = 30200 patches — still over, because the ceilings round up. So
    the sqrt scale is only the starting guess for the long edge; the short edge is always derived
    from it (rounded down, so the aspect never drifts past a pixel) and the long edge then steps
    until the integer check is exact. Patches never decrease as the long edge grows, so stepping
    up while the next size fits and down while this one does not lands on the maximum.
    """
    long_edge, short_edge = max(width, height), min(width, height)

    def dims(long_px: int) -> Tuple[int, int]:
        return long_px, max(1, long_px * short_edge // long_edge)

    def fits(long_px: int) -> bool:
        return _vision_patches(*dims(long_px)) <= _VISION_MAX_PATCHES

    scale = math.sqrt(_VISION_MAX_PATCHES / _vision_patches(width, height))
    candidate = min(long_edge, max(1, int(long_edge * scale)))
    while candidate + 1 <= long_edge and fits(candidate + 1):
        candidate += 1
    while candidate > 1 and not fits(candidate):
        candidate -= 1
    new_long, new_short = dims(candidate)
    return (new_long, new_short) if width >= height else (new_short, new_long)


def _shrink_to_patch_budget(raw: bytes,
                            mime: str) -> Tuple[Optional[bytes], Optional[str]]:
    """Downscale a decodable image that is over the patch budget, in memory. Never raises.

    EXIF orientation is applied FIRST: the re-encoded bytes carry no EXIF, so a phone photo left
    untransposed would reach the model lying on its side. A JPEG is re-encoded with the source's
    own quantization tables and chroma subsampling — its own compression level, not a quality
    number made up here; anything else (PNG, WebP, a BMP/TIFF already transcoded to PNG) is
    encoded as PNG with the same mode rules as `_transcode_to_png`, so alpha survives.
    """
    try:
        from PIL import Image, ImageOps, JpegImagePlugin

        with Image.open(BytesIO(raw)) as im:
            is_jpeg = mime == "image/jpeg"
            qtables = getattr(im, "quantization", None) if is_jpeg else None
            subsampling = JpegImagePlugin.get_sampling(im) if is_jpeg else -1
            src_w, src_h = im.size
            # Carried onto the output: dropping it re-reads a Display-P3 phone photo as sRGB.
            icc_profile = im.info.get("icc_profile")
            upright = ImageOps.exif_transpose(im)
        width, height = upright.size
        before = _vision_patches(width, height)
        new_w, new_h = _fit_to_patch_budget(width, height)
        after = _vision_patches(new_w, new_h)
        if after > _VISION_MAX_PATCHES:  # the explicit integer check, before any encode
            return None, UNREADABLE

        if is_jpeg:
            resized = upright.resize((new_w, new_h), Image.Resampling.LANCZOS)
            save_kwargs: Dict[str, Any] = {}
            if qtables:
                save_kwargs["qtables"] = qtables
            if subsampling is not None and subsampling >= 0:
                save_kwargs["subsampling"] = subsampling
            if icc_profile:
                save_kwargs["icc_profile"] = icc_profile
            out = BytesIO()
            resized.save(out, format="JPEG", **save_kwargs)
            out_mime = "image/jpeg"
        else:
            mode = upright.mode
            # Any `transparency` key counts, not just a palette one: an RGB/L PNG can carry a
            # tRNS colour key, and resampling it without converting first bleeds the hidden
            # colour into its opaque neighbours.
            has_alpha = mode in ("RGBA", "LA", "PA") or "transparency" in upright.info
            if has_alpha:
                converted = upright.convert("RGBA")
            elif mode == "RGB":
                converted = upright
            else:
                converted = upright.convert("RGB")
            resized = converted.resize((new_w, new_h), Image.Resampling.LANCZOS)
            png_kwargs: Dict[str, Any] = {"icc_profile": icc_profile} if icc_profile else {}
            out = BytesIO()
            resized.save(out, format="PNG", **png_kwargs)
            out_mime = "image/png"
        logger.info(f"Downscaled image over the vision patch budget: {src_w}x{src_h} -> "
                    f"{new_w}x{new_h} ({before} -> {after} patches)")
        return out.getvalue(), out_mime
    except Exception:  # noqa: BLE001 — any decode/resize/encode failure is a graceful rejection
        return None, UNREADABLE


def ensure_api_compatible(raw: bytes) -> Tuple[Optional[bytes], Optional[str]]:
    """Bytes the Responses *vision* API will accept, transcoding in memory when it has to.

    `ensure_compatible` pinned to `VISION_MIMETYPES` (jpeg/png/gif/webp), plus one vision-only
    rule: an image whose pixel dimensions are over the API's patch budget (`_VISION_MAX_PATCHES`)
    comes back SMALLER — the largest aspect-preserving size that fits, upright per its EXIF
    orientation, as JPEG for a JPEG source and PNG otherwise. Without that, a full-resolution phone
    photo 400s the whole turn. Only a confirmed-oversize image is touched: the size is read from the
    header, and anything within budget comes back exactly as `ensure_compatible` returned it (the
    SAME bytes object for an already-compatible source). A frame past `_MAX_TRANSCODE_PIXELS` is
    refused (TOO_LARGE_TO_RESIZE) rather than decoded. GIFs are never re-encoded here — not even an
    oversize one (see REJECT_ANIMATED_GIFS). The Images edit endpoint does not go through this, so
    edit sources keep their pixels. See `ensure_compatible` for the rest of the contract.
    """
    out, mime = ensure_compatible(raw, allowed=VISION_MIMETYPES)
    if out is None or mime is None or mime == "image/gif":
        return out, mime
    try:
        from PIL import Image

        with Image.open(BytesIO(out)) as im:  # lazy: reads the header, decodes nothing
            width, height = im.size
    except Exception:  # noqa: BLE001 — ensure_compatible just parsed these bytes; be safe anyway
        return None, UNREADABLE
    if _vision_patches(width, height) <= _VISION_MAX_PATCHES:
        return out, mime
    if width * height > _MAX_TRANSCODE_PIXELS:
        return None, TOO_LARGE_TO_RESIZE
    return _shrink_to_patch_budget(out, mime)
