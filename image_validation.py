"""One source of truth for what the OpenAI API will accept as an image.

Three lists used to disagree: `image_url_handler.SUPPORTED_IMAGE_MIMETYPES` (consulted only on
the URL path), `gate_vision._SAFE_MIMETYPES` (narrower still), and — on the attachment path —
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

from io import BytesIO
from typing import Optional, Tuple

# The mimetypes the API accepts, as DECLARED labels — for cheap pre-download screening only
# (`gate_vision.eligible`, URL content-type checks). `image/jpg` is not a real mimetype, but
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

_REJECTION_TEXT = {
    UNREADABLE: ("isn't in a format I can read — I can look at PNG, JPEG, GIF and WebP images. "
                 "(HEIC photos straight from an iPhone are a common culprit; re-saving as PNG "
                 "or JPEG works.)"),
    ANIMATED_GIF: "is an animated GIF, which I can't look at — a static image works.",
    TOO_LARGE_AFTER_CONVERSION: ("is too large to edit once converted to a supported format — "
                                 "a smaller or already-PNG/JPEG image works."),
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


def _decodes_as_image(raw: bytes) -> bool:
    """True only if Pillow can actually PARSE these bytes as an image.

    A magic-byte prefix is not proof: "PNG signature + junk" matches the prefix but is not a
    real PNG, and sending it still 400s the whole turn — the exact failure this module exists to
    prevent. Pillow's `verify()` checks structural integrity without a full decode, but it leaves
    the file object spent, so the image must be REOPENED before `load()` (Pillow's documented
    verify-then-reopen requirement). We do both: verify catches truncation/corruption cheaply,
    load forces the decoder far enough to reject signature-plus-junk.
    """
    try:
        from PIL import Image

        with Image.open(BytesIO(raw)) as im:
            im.verify()
        with Image.open(BytesIO(raw)) as im2:
            im2.load()
        return True
    except Exception:  # noqa: BLE001 — any parse failure means these bytes are not a real image
        return False


def validate_image_bytes(raw: bytes) -> Tuple[Optional[str], Optional[str]]:
    """Decide whether these bytes may ride an API call.

    Returns `(mimetype, None)` on success — the SNIFFED mimetype, which the caller should send
    in place of whatever was declared — or `(None, reason)` on rejection, where reason is one of
    UNREADABLE / ANIMATED_GIF.

    Two gates: the magic-byte sniff decides the FORMAT (and that it is one the API accepts), then
    Pillow PARSES the bytes so a valid-looking prefix followed by garbage is rejected here rather
    than by a 400 mid-turn.
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
    if not _decodes_as_image(raw):
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


def ensure_api_compatible(raw: bytes) -> Tuple[Optional[bytes], Optional[str]]:
    """Bytes the Responses *vision* API will accept, transcoding in memory when it has to.

    Thin wrapper over `ensure_compatible` pinned to `VISION_MIMETYPES` (jpeg/png/gif/webp).
    Animated GIFs ride through unchanged — GIFs are never re-encoded here (see
    REJECT_ANIMATED_GIFS). See `ensure_compatible` for the full contract.
    """
    return ensure_compatible(raw, allowed=VISION_MIMETYPES)
