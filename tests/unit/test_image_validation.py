"""F50 — images are validated by the BYTES before they can ride an API call.

The bug this pins down: nothing validated image attachments. Slack types any `image/*` as an
image, and `_process_attachments` base64'd it straight into the request — so a single image/heic
400'd the ENTIRE turn and the user's message just failed, silently. These tests are about
behavior at that boundary: what gets sent, what gets turned away, and what the user is told.
"""

import base64
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

import image_validation
from image_validation import (ANIMATED_GIF, UNREADABLE, API_IMAGE_MIMETYPES,
                              IMAGE_EDIT_MIMETYPES, ensure_api_compatible,
                              ensure_compatible, sniff_image_mimetype,
                              validate_image_bytes)
from message_processor.base import MessageProcessor
from message_processor.utilities import MessageUtilitiesMixin

pytestmark = pytest.mark.unit


# --------------------------------------------------------------- real bytes, not fixtures
# Pillow is already a dependency, so these are REAL encoded images rather than magic-byte
# stubs — a stub can't tell us whether the animation check actually works.

def _png() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (2, 2), "red").save(buf, format="PNG")
    return buf.getvalue()


def _jpeg() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (2, 2), "blue").save(buf, format="JPEG")
    return buf.getvalue()


def _webp() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (2, 2), "green").save(buf, format="WEBP")
    return buf.getvalue()


def _static_gif() -> bytes:
    buf = BytesIO()
    Image.new("P", (2, 2), 0).save(buf, format="GIF")
    return buf.getvalue()


def _animated_gif() -> bytes:
    # The frames must genuinely DIFFER: Pillow's GIF writer drops a frame identical to its
    # predecessor, which quietly collapses a naive fixture to one frame — and then the thing
    # under test correctly accepts it and the test proves nothing. Guarded below.
    buf = BytesIO()
    frames = [Image.new("RGB", (4, 4), "red").convert("P"),
              Image.new("RGB", (4, 4), "blue").convert("P")]
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=100)
    return buf.getvalue()


def _bmp() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (2, 2), "red").save(buf, format="BMP")
    return buf.getvalue()


def _tiff() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (3, 3), "red").save(buf, format="TIFF")
    return buf.getvalue()


def _multiframe_tiff() -> bytes:
    buf = BytesIO()
    frames = [Image.new("RGB", (3, 3), "red"), Image.new("RGB", (3, 3), "blue")]
    frames[0].save(buf, format="TIFF", save_all=True, append_images=frames[1:])
    return buf.getvalue()


def _rgba_tiff() -> bytes:
    # A NON-accepted format (TIFF) carrying a real alpha channel: pixel (0,0) fully transparent.
    # The transcode must land on RGBA and keep that transparency.
    img = Image.new("RGBA", (4, 4), (255, 0, 0, 255))
    img.putpixel((0, 0), (0, 0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="TIFF")
    return buf.getvalue()


def _palette_png_with_transparency() -> bytes:
    # A P-mode image whose info carries a `transparency` index. PNG is API-accepted, so this is fed
    # to `_transcode_to_png` DIRECTLY (not via ensure_api_compatible, which would pass it through)
    # to prove the palette->RGBA branch keeps the transparent index as alpha.
    img = Image.new("P", (4, 4), 1)
    img.putpixel((0, 0), 0)
    img.info["transparency"] = 0
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestSupportedFormatsStillWork:
    """The whole point of a gate is that it lets the right things through."""

    @pytest.mark.parametrize("raw,expected", [
        (_png(), "image/png"),
        (_jpeg(), "image/jpeg"),
        (_webp(), "image/webp"),
        (_static_gif(), "image/gif"),
    ])
    def test_api_supported_images_pass_with_their_real_mimetype(self, raw, expected):
        assert validate_image_bytes(raw) == (expected, None)

    def test_every_sniffable_type_is_in_the_shared_allowlist(self):
        # The allowlist screens DECLARED mimetypes before download; the sniffer names what
        # actually arrived. If the sniffer could ever produce something the allowlist rejects,
        # the two halves of this module would disagree — which is the bug F50 exists to kill.
        for raw in (_png(), _jpeg(), _webp(), _static_gif()):
            assert sniff_image_mimetype(raw) in API_IMAGE_MIMETYPES


class TestGifAnimation:
    """The API accepts GIF — including animated (first frame). The spec claimed otherwise; these
    pin the ACTUAL behavior (accept by default) plus the retained opt-in rejection."""

    def test_the_fixtures_are_what_they_claim_to_be(self):
        # Everything below is worthless if these two are secretly the same file.
        assert Image.open(BytesIO(_animated_gif())).is_animated is True
        assert Image.open(BytesIO(_static_gif())).is_animated is False

    def test_static_gif_is_accepted(self):
        assert validate_image_bytes(_static_gif()) == ("image/gif", None)

    def test_animated_gif_is_accepted_by_default(self):
        # Verified live: the API renders an animated gif's first frame across the whole model
        # family. Refusing it would decline an image the bot can actually read.
        assert validate_image_bytes(_animated_gif()) == ("image/gif", None)

    def test_opt_in_flag_rejects_animated_gif_with_its_own_reason(self, monkeypatch):
        # The detection is retained behind REJECT_ANIMATED_GIFS for whoever wants first-frame
        # gifs turned away. When on, the user is told it was the animation, not just "unreadable".
        monkeypatch.setattr(image_validation, "REJECT_ANIMATED_GIFS", True)
        assert validate_image_bytes(_animated_gif()) == (None, ANIMATED_GIF)
        # A static gif still passes under the flag.
        assert validate_image_bytes(_static_gif()) == ("image/gif", None)

    def test_opt_in_flag_rejects_undetermined_gif_rather_than_gambling(self, monkeypatch):
        # With rejection on, a gif we can't parse is a reject, not a 400 gamble.
        monkeypatch.setattr(image_validation, "REJECT_ANIMATED_GIFS", True)
        mime, reason = validate_image_bytes(_animated_gif()[:12])
        assert mime is None and reason == UNREADABLE


class TestBytesBeatDeclaredMimetype:
    @pytest.mark.parametrize("raw", [_bmp(), b"<html>login page</html>", b"", b"RIFFxxxxAVI "])
    def test_unsupported_and_junk_bytes_are_rejected(self, raw):
        assert validate_image_bytes(raw)[0] is None

    def test_riff_container_that_is_not_webp_is_rejected(self):
        # RIFF fronts WAV/AVI too; the WEBP tag has to actually be there.
        assert sniff_image_mimetype(b"RIFF____WAVEfmt ") is None

    @pytest.mark.parametrize("raw", [
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 128,          # PNG signature + junk
        b"\xff\xd8\xff" + b"\x00" * 128,               # JPEG SOI + junk
        b"GIF89a" + b"\x00" * 32,                       # GIF header + junk
        b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 32,     # WEBP container + junk
    ])
    def test_signature_plus_junk_is_rejected_by_parse(self, raw):
        # A valid magic prefix is NOT proof: these sniff as a supported format but do not DECODE,
        # and sending them still 400s the whole turn. validate_image_bytes must parse (Pillow) and
        # reject, not merely sniff the prefix.
        mime, reason = validate_image_bytes(raw)
        assert mime is None and reason == UNREADABLE

    def test_mislabeled_but_supported_image_is_corrected_not_rejected(self):
        # A JPEG named .png is a picture we can read. Sending the DECLARED type would 400;
        # rejecting it would be needlessly obtuse. We send what the bytes are.
        assert validate_image_bytes(_jpeg()) == ("image/jpeg", None)


# --------------------------------------------------- F50b: transcode instead of reject

class TestEnsureApiCompatible:
    """`ensure_api_compatible` is the layer the call sites use: pass the accepted formats through
    untouched, transcode the decodable-but-unsupported ones to PNG, reject only the undecodable."""

    @pytest.mark.parametrize("raw,mime", [
        (_png(), "image/png"),
        (_jpeg(), "image/jpeg"),
        (_webp(), "image/webp"),
        (_static_gif(), "image/gif"),
        (_animated_gif(), "image/gif"),
    ])
    def test_accepted_formats_pass_through_byte_identical(self, raw, mime):
        out_bytes, out_mime = ensure_api_compatible(raw)
        assert out_mime == mime
        # The SAME object, not a re-encode or even a copy — the happy path must not touch bytes.
        assert out_bytes is raw

    def test_bmp_transcodes_to_a_real_png(self):
        out_bytes, out_mime = ensure_api_compatible(_bmp())
        assert out_mime == "image/png"
        assert out_bytes is not None and out_bytes != _bmp()
        # The result must itself be an API-acceptable PNG.
        assert validate_image_bytes(out_bytes) == ("image/png", None)

    def test_tiff_transcodes_to_png(self):
        out_bytes, out_mime = ensure_api_compatible(_tiff())
        assert out_mime == "image/png"
        assert validate_image_bytes(out_bytes) == ("image/png", None)

    def test_multiframe_tiff_transcodes_first_frame_only(self):
        out_bytes, out_mime = ensure_api_compatible(_multiframe_tiff())
        assert out_mime == "image/png"
        result = Image.open(BytesIO(out_bytes))
        result.load()
        # One frame survives — the first — not an animated PNG.
        assert getattr(result, "n_frames", 1) == 1
        assert result.getpixel((0, 0))[:3] == (255, 0, 0)  # red, the first frame

    def test_rgba_tiff_keeps_its_alpha_through_the_transcode(self):
        out_bytes, out_mime = ensure_api_compatible(_rgba_tiff())
        assert out_mime == "image/png"
        result = Image.open(BytesIO(out_bytes)).convert("RGBA")
        assert result.getpixel((0, 0))[3] == 0        # the transparent pixel stayed transparent
        assert result.getpixel((1, 1))[3] == 255      # an opaque pixel stayed opaque

    def test_palette_transparency_becomes_rgba_alpha(self):
        # PNG is API-accepted, so drive the transcode helper directly to exercise the P->RGBA path.
        out = image_validation._transcode_to_png(_palette_png_with_transparency())
        assert out is not None
        result = Image.open(BytesIO(out))
        assert result.mode == "RGBA"
        assert result.getpixel((0, 0))[3] == 0

    @pytest.mark.parametrize("raw", [
        b"",                                            # nothing at all
        b"\x00\x00\x00 ftypheic",                       # HEIC — no decoder installed
        _png()[:12],                                    # truncated PNG
        _bmp()[:8],                                     # truncated BMP
    ])
    def test_undecodable_bytes_are_rejected(self, raw):
        assert ensure_api_compatible(raw) == (None, UNREADABLE)

    @pytest.mark.parametrize("raw", [
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 128,           # PNG signature + junk
        b"\xff\xd8\xff" + b"\x00" * 128,                # JPEG SOI + junk
    ])
    def test_signature_plus_junk_is_rejected_not_transcoded(self, raw):
        # A valid API-format signature over garbage is a broken member of that format, not a
        # transcode candidate — reject it, never re-encode it into a "valid" PNG.
        assert ensure_api_compatible(raw) == (None, UNREADABLE)

    def test_absurdly_large_image_is_refused_rather_than_transcoded(self, monkeypatch):
        # The pixel cap keeps a decompression-bomb-sized frame from being re-encoded. A 2x2 BMP
        # trips a cap of 3 px, standing in for the real 50-megapixel ceiling without the memory.
        monkeypatch.setattr(image_validation, "_MAX_TRANSCODE_PIXELS", 3)
        assert ensure_api_compatible(_bmp()) == (None, UNREADABLE)

    def test_animated_gif_under_the_reject_flag_is_not_transcoded(self, monkeypatch):
        # GIFs are never re-encoded. With the opt-in flag on, a rejected animated gif keeps its
        # ANIMATED_GIF reason rather than being silently transcoded to a first-frame PNG.
        monkeypatch.setattr(image_validation, "REJECT_ANIMATED_GIFS", True)
        assert ensure_api_compatible(_animated_gif()) == (None, ANIMATED_GIF)


class TestEnsureCompatibleEditSet:
    """`ensure_compatible(allowed=IMAGE_EDIT_MIMETYPES)` is the edit endpoint's gate: png/jpeg/webp
    only. GIF — which vision reads directly — is NOT accepted here and must be transcoded."""

    @pytest.mark.parametrize("raw,mime", [
        (_png(), "image/png"),
        (_jpeg(), "image/jpeg"),
        (_webp(), "image/webp"),
    ])
    def test_edit_accepted_formats_pass_through_byte_identical(self, raw, mime):
        out_bytes, out_mime = ensure_compatible(raw, allowed=IMAGE_EDIT_MIMETYPES)
        assert out_mime == mime
        assert out_bytes is raw          # no re-encode, not even a copy

    def test_static_gif_transcodes_to_png_for_editing(self):
        # Vision accepts GIF; the edit endpoint does not. A recognized-but-unallowed format is
        # transcoded rather than passed through or rejected.
        out_bytes, out_mime = ensure_compatible(_static_gif(), allowed=IMAGE_EDIT_MIMETYPES)
        assert out_mime == "image/png"
        assert out_bytes is not None
        assert validate_image_bytes(out_bytes) == ("image/png", None)

    def test_animated_gif_transcodes_first_frame_to_png_for_editing(self):
        out_bytes, out_mime = ensure_compatible(_animated_gif(), allowed=IMAGE_EDIT_MIMETYPES)
        assert out_mime == "image/png"
        result = Image.open(BytesIO(out_bytes))
        result.load()
        assert result.format == "PNG"
        assert getattr(result, "n_frames", 1) == 1     # first frame only
        assert result.getpixel((0, 0))[:3] == (255, 0, 0)   # red, the first frame

    def test_bmp_still_transcodes_to_png_for_editing(self):
        out_bytes, out_mime = ensure_compatible(_bmp(), allowed=IMAGE_EDIT_MIMETYPES)
        assert out_mime == "image/png"
        assert validate_image_bytes(out_bytes) == ("image/png", None)

    @pytest.mark.parametrize("raw", [
        b"",
        b"this is not an image",
        b"\x89PNG\r\n\x1a\n" + b"\x00" * 128,           # signature + junk: broken, not transcoded
    ])
    def test_undecodable_edit_source_is_rejected(self, raw):
        assert ensure_compatible(raw, allowed=IMAGE_EDIT_MIMETYPES) == (None, UNREADABLE)

    def test_default_allowed_matches_the_vision_wrapper(self):
        # `ensure_compatible` with no `allowed` == `ensure_api_compatible`: gif passes through.
        for raw in (_png(), _jpeg(), _webp(), _static_gif(), _animated_gif()):
            assert ensure_compatible(raw) == ensure_api_compatible(raw)


# ------------------------------------------------------- the attachment path (the live bug)

class _Proc(MessageUtilitiesMixin):
    """Minimal harness around the mixin — _process_attachments is what actually shipped the bug."""

    def __init__(self):
        self.db = None
        self.document_handler = None
        self.image_url_handler = MagicMock()
        # Real int so the F5 pre-download gate / F19 post-transcode ceiling can compare.
        self.image_url_handler.max_image_size = 20 * 1024 * 1024
        self.thread_manager = MagicMock()
        for name in ("log_info", "log_debug", "log_warning", "log_error"):
            setattr(self, name, MagicMock())


def _message(attachments):
    return SimpleNamespace(attachments=attachments, text="", channel_id="C1",
                           thread_id="123.456", metadata={"ts": "123.456"})


def _client(payload: bytes):
    client = MagicMock()
    client.download_file = AsyncMock(return_value=payload)
    return client


def _attachment(name, mimetype):
    return {"type": "image", "name": name, "mimetype": mimetype,
            "url": f"https://files.slack.com/{name}", "id": "F123"}


class TestProcessAttachments:
    @pytest.mark.asyncio
    async def test_supported_image_still_rides_the_turn(self):
        proc = _Proc()
        images, _docs, unsupported = await proc._process_attachments(
            _message([_attachment("ok.png", "image/png")]), _client(_png()))

        assert unsupported == []
        assert len(images) == 1
        assert images[0]["image_url"].startswith("data:image/png;base64,")
        # And it's the real bytes, base64'd once.
        assert images[0]["image_url"].endswith(base64.b64encode(_png()).decode())

    @pytest.mark.asyncio
    async def test_heic_degrades_to_a_notice_instead_of_400ing_the_turn(self):
        # THE BUG. HEIC is what an iPhone actually uploads, and it used to be base64'd into the
        # request unchallenged, taking the user's whole message down with it.
        proc = _Proc()
        images, _docs, unsupported = await proc._process_attachments(
            _message([_attachment("photo.heic", "image/heic")]), _client(b"\x00\x00\x00 ftypheic"))

        assert images == []
        assert len(unsupported) == 1
        assert unsupported[0]["name"] == "photo.heic"
        assert unsupported[0]["reason"] == UNREADABLE

    @pytest.mark.asyncio
    async def test_both_static_and_animated_gif_ride_the_turn(self):
        # The API accepts both (animated -> first frame), so both reach the model rather than
        # bouncing off a false refusal.
        proc = _Proc()
        for raw in (_animated_gif(), _static_gif()):
            images, _docs, unsupported = await proc._process_attachments(
                _message([_attachment("g.gif", "image/gif")]), _client(raw))
            assert unsupported == []
            assert images[0]["image_url"].startswith("data:image/gif;base64,")

    @pytest.mark.asyncio
    async def test_a_bmp_lying_as_png_is_transcoded_and_rides(self):
        # Slack says image/png; it is a BMP — a format the API rejects but Pillow decodes.
        # F50b transcodes it to PNG in memory rather than turning it away, so it rides the turn
        # as a genuine image/png (not the BMP bytes the API would 400 on).
        proc = _Proc()
        images, _docs, unsupported = await proc._process_attachments(
            _message([_attachment("sneaky.png", "image/png")]), _client(_bmp()))

        assert unsupported == []
        assert len(images) == 1
        assert images[0]["image_url"].startswith("data:image/png;base64,")
        payload = base64.b64decode(images[0]["image_url"].split(",", 1)[1])
        assert sniff_image_mimetype(payload) == "image/png"

    @pytest.mark.asyncio
    async def test_a_mislabeled_jpeg_is_sent_as_a_jpeg(self):
        # The data URL must carry the SNIFFED type — handing the API `data:image/png` with JPEG
        # bytes is the same 400 by a different route.
        proc = _Proc()
        images, _docs, unsupported = await proc._process_attachments(
            _message([_attachment("mislabeled.png", "image/png")]), _client(_jpeg()))

        assert unsupported == []
        assert images[0]["image_url"].startswith("data:image/jpeg;base64,")

    @pytest.mark.asyncio
    async def test_one_bad_image_does_not_cost_the_good_one(self):
        # The bad one is genuinely undecodable (HEIC, no decoder) — a BMP would now transcode and
        # ride, so it can no longer stand in for "bad".
        proc = _Proc()
        client = MagicMock()
        client.download_file = AsyncMock(side_effect=[b"\x00\x00\x00 ftypheic", _png()])
        images, _docs, unsupported = await proc._process_attachments(
            _message([_attachment("bad.png", "image/png"), _attachment("good.png", "image/png")]),
            client)

        assert len(images) == 1 and len(unsupported) == 1
        assert unsupported[0]["name"] == "bad.png"

    @pytest.mark.asyncio
    async def test_rejected_image_is_never_recorded_as_an_image_we_have(self):
        # A row in the image ledger for a picture we never looked at would let a later turn
        # claim it saw one.
        proc = _Proc()
        proc.db = MagicMock()
        proc.db.save_image_metadata_async = AsyncMock()
        await proc._process_attachments(
            _message([_attachment("photo.heic", "image/heic")]), _client(b"\x00\x00\x00 ftypheic"))

        proc.db.save_image_metadata_async.assert_not_called()


# ------------------------------------------------------------------ what the user is told

class TestFailedFilesNotice:
    build = staticmethod(MessageProcessor._build_failed_files_notice)

    def test_fetch_failure_and_rejection_stay_distinct(self):
        # "I couldn't download it" and "I can't read it" are different problems with different
        # fixes; collapsing them tells the user to retry something that will never work.
        out = self.build([
            {"name": "gone.png", "type": "image", "mimetype": "image/png",
             "error": "download_failed"},
            {"name": "photo.heic", "type": "image", "mimetype": "image/heic",
             "reason": UNREADABLE},
        ])
        assert "Couldn't Download File" in out and "gone.png" in out
        assert "Couldn't Read Image" in out and "photo.heic" in out
        assert "try re-uploading" in out

    def test_animated_gif_is_not_told_that_gif_is_supported(self):
        # Routing a rejected image through the generic explainer would print
        # "• Images (JPEG, PNG, GIF, WebP)" directly under a rejected GIF.
        out = self.build([{"name": "spin.gif", "type": "image", "mimetype": "image/gif",
                           "reason": ANIMATED_GIF}])
        assert "animated GIF" in out
        assert "Currently supported" not in out

    def test_unreadable_image_names_the_formats_it_can_read(self):
        out = self.build([{"name": "photo.heic", "type": "image", "mimetype": "image/heic",
                           "reason": UNREADABLE}])
        assert "photo.heic" in out
        assert "PNG, JPEG, GIF and WebP" in out

    def test_unrelated_unsupported_files_keep_the_old_explainer(self):
        # No `reason` -> the pre-F50 path, untouched.
        out = self.build([{"name": "x.bin", "type": "file", "mimetype": "application/octet-stream"}])
        assert "Unsupported File Type" in out and "Currently supported" in out
        assert "Couldn't Read Image" not in out


# ------------------------------------------------------------------------- the wake gate

class TestGateVisionStaysNarrow:
    def test_gate_still_refuses_gif_on_the_hot_path(self):
        # Deliberate, not an oversight: telling static from animated costs a parse, and the gate
        # runs in front of every ambient message. Unifying this away would be a regression.
        from message_processor import gate_vision

        assert "image/gif" not in gate_vision._SAFE_MIMETYPES
        assert gate_vision._sniff(_static_gif()) is None

    def test_gate_shows_what_the_api_accepts_otherwise(self):
        from message_processor import gate_vision

        assert gate_vision._sniff(_png()) == "image/png"
        assert gate_vision._sniff(_jpeg()) == "image/jpeg"
        assert gate_vision._sniff(_webp()) == "image/webp"
        assert gate_vision._sniff(b"<html>login</html>") is None
