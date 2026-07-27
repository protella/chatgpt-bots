"""F40 — the image-loading module, kept; the gate that used it, gone.

THE ORIGINAL BUG, live (2026-07-13): a meme was posted with the caption ":dogkek:" and nothing
else, and the rich gate returned `{"action":"react","emoji":"joy","reason":"A laughing reaction
fits the playful meme post."}` — a confident opinion about an image it had never seen, reasoned
from the emoji SHORTCODE in the caption. F40's answer was to show it the picture: reacting to
something you have not looked at is the same dishonesty as an 👀 with no work behind it.

COMMIT 6 ANSWERS IT THE OTHER WAY. The gate no longer reacts, so it has no opinion to be wrong
about. It decides one thing — whether the responder runs — and a picture cannot change that answer
in a way its filename does not, so the gate does not look at images at all. Everything that made
the pictures reach it is deleted: the `images=` parameter, the prompt's image_status line, the
observation harvest, the gate→ambient piggyback, and the ambient hold that waited for the gate's
decision. The ambient worker analyses images on its own schedule, immediately, whatever the gate
decides — which is strictly better, because nothing now waits on a resolver that might not come.

What SURVIVES here, and why the file survives with it:

* `message_processor/gate_vision.py` — the eligibility/fetch/validate module. It has no runtime
  caller now and is scheduled for deletion in the cleanup commit; until then its own tests are the
  only thing keeping it honest, so they stay (spec §8).
* the ONE tripwire section, asserting there is no way back in: the engine takes no images, the
  ambient predicate is a hard False, and the harvest is gone.
* the two vision-default tests at the bottom, which are about the ANSWERING path's image fidelity
  and never had anything to do with the gate.
"""

import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import config
from message_processor import gate_vision

def _real_image(fmt: str) -> bytes:
    from io import BytesIO

    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (4, 4), "red").save(buf, format=fmt)
    return buf.getvalue()


# Genuinely-decodable images: F38's validator PARSES the bytes with Pillow, so the gate's "good"
# fixtures must be real pictures, not a signature-plus-junk stub.
PNG = _real_image("PNG")
JPEG = _real_image("JPEG")
HTML = b"<!DOCTYPE html><html><body>Sign in to Slack"
# A valid PNG signature followed by junk: it SNIFFS as a PNG but does not decode. Before F38 this
# reached the gate's own vision call and 400'd it (a wasted round that degrades to a blind text
# retry); now it is turned away here.
CORRUPT_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def _img(name="meme.png", mime="image/png", size=1234, url="https://files.slack.com/x"):
    return {"type": "image", "url": url, "id": "F1", "name": name,
            "mimetype": mime, "size": size}


def _client(payload=PNG):
    c = MagicMock()
    c.download_file = AsyncMock(return_value=payload)
    return c


# --------------------------------------------------------------------- eligibility

def test_only_formats_the_model_actually_accepts_are_sent():
    """SVG is markup, not a picture (and an injection surface); HEIC isn't accepted; an animated
    GIF can be rejected outright. Gamble on the hot path and the gate 400s — so we skip them and
    say so, rather than guess."""
    assert gate_vision.eligible([_img(mime="image/png")])
    assert gate_vision.eligible([_img(mime="image/jpeg")])
    assert gate_vision.eligible([_img(mime="image/webp")])
    assert not gate_vision.eligible([_img(mime="image/svg+xml")])
    assert not gate_vision.eligible([_img(mime="image/heic")])
    assert not gate_vision.eligible([_img(mime="application/pdf")])


def test_an_oversized_image_is_rejected_before_a_single_byte_is_fetched(monkeypatch):
    monkeypatch.setattr(config, "gate_vision_max_bytes", 1000, raising=False)
    assert not gate_vision.eligible([_img(size=5_000_000)])
    assert gate_vision.eligible([_img(size=900)])


@pytest.mark.asyncio
async def test_the_image_reaches_the_model_as_an_input_image_part(monkeypatch):
    monkeypatch.setattr(config, "enable_multimodal_gate", True, raising=False)
    monkeypatch.setattr(config, "gate_vision_detail", "low", raising=False)

    parts, status, _shown = await gate_vision.load_for_gate(_client(PNG), [_img()])

    assert status == gate_vision.VISIBLE
    assert len(parts) == 1
    assert parts[0]["type"] == "input_image"
    assert parts[0]["detail"] == "low"
    assert parts[0]["image_url"].startswith("data:image/png;base64,")
    assert base64.b64decode(parts[0]["image_url"].split(",", 1)[1]) == PNG
    # api_part()'s whitelist is {type, image_url, detail} — anything else is a hard 400.
    assert set(parts[0]) == {"type", "image_url", "detail"}


@pytest.mark.asyncio
async def test_a_slack_login_page_is_not_mistaken_for_an_image():
    """Slack serves an HTML login page with HTTP 200 when auth is wrong, so "it downloaded fine"
    proves nothing. Sniff the bytes; never hand the model a web page dressed as a PNG."""
    parts, status, _shown = await gate_vision.load_for_gate(_client(HTML), [_img()])
    assert parts == []
    assert status == gate_vision.UNAVAILABLE


@pytest.mark.asyncio
async def test_structurally_corrupt_image_is_turned_away_not_sent_to_the_model():
    """F38: magic bytes are not proof. A valid PNG signature followed by junk sniffs as a PNG but
    does not decode, so it 400s the gate's vision call — a wasted round that then degrades to a
    blind text-only retry. The Pillow-parsing validator rejects it here, so the gate degrades to
    UNAVAILABLE up front and the model is honestly told it couldn't see the image."""
    parts, status, _shown = await gate_vision.load_for_gate(_client(CORRUPT_PNG), [_img()])
    assert parts == []
    assert status == gate_vision.UNAVAILABLE


@pytest.mark.asyncio
async def test_a_failed_download_degrades_to_unavailable_never_to_silence():
    c = MagicMock()
    c.download_file = AsyncMock(side_effect=RuntimeError("slack is down"))
    parts, status, _shown = await gate_vision.load_for_gate(c, [_img()])
    assert parts == []
    assert status == gate_vision.UNAVAILABLE


@pytest.mark.asyncio
async def test_the_image_cap_is_honored(monkeypatch):
    monkeypatch.setattr(config, "gate_vision_max_images", 2, raising=False)
    parts, _, _shown = await gate_vision.load_for_gate(
        _client(PNG), [_img(name=f"{i}.png") for i in range(5)])
    assert len(parts) == 2


@pytest.mark.asyncio
async def test_the_flag_turns_it_all_off(monkeypatch):
    monkeypatch.setattr(config, "enable_multimodal_gate", False, raising=False)
    parts, status, _shown = await gate_vision.load_for_gate(_client(PNG), [_img()])
    assert parts == [] and status == gate_vision.NONE


# --------------------------------------------------------------------- the way back in is shut

@pytest.mark.asyncio
async def test_the_engine_cannot_be_handed_an_image():
    """The `images=` parameter is gone from `evaluate`, which is what makes "the gate does not look
    at pictures" structural rather than a habit. Asserted as a TypeError because a silently-ignored
    kwarg would let a caller believe it had sent something."""
    from message_processor.participation import ParticipationEngine

    engine = ParticipationEngine(MagicMock())
    parts, status, _shown = await gate_vision.load_for_gate(_client(PNG), [_img()])
    assert parts, "fixture sanity: the module still loads an image"
    with pytest.raises(TypeError):
        await engine.evaluate(channel_id="C1", ts="10.0", text=":dogkek:", images=parts)
    # And the harvest that turned a verdict's `image_observations` into stored descriptions.
    assert not hasattr(ParticipationEngine, "_harvest_image_observations")
    assert not hasattr(ParticipationEngine, "_ambient_service")


@pytest.mark.asyncio
async def test_the_attachment_reaches_the_gate_as_a_name_and_a_type():
    """What the gate gets INSTEAD, so this is not merely a deletion.

    A descriptor — "meme.png (image)" — and the rendered prompt says outright that the contents are
    not shown. That is the honest version of F40's problem: rather than promising the model a look
    at the picture (or explaining why it cannot have one), it is told what was attached and nothing
    more, and it is deciding something a filename can support."""
    from message_processor.participation import SourceMessage
    from openai_client.api.responses import _render_wake_source

    block = _render_wake_source(SourceMessage(
        ts="10.0", text=":dogkek:", sender_name="Peter", sender_type="human",
        attachments=("meme.png (image)",)), index=0, total=1)
    assert "meme.png (image)" in block
    assert "contents not shown to you" in block
    for retired in ("image_status", "cannot see", "do not guess"):
        assert retired not in block


def test_nothing_holds_an_ambient_image_for_the_gate():
    """The hold was only safe while a resolver was guaranteed to release it, and the resolver WAS
    the gate's post-decision callback. With no callback, a held image would sit until a bounded
    timeout expired — analysis delayed for no benefit, on every image in every channel. The
    predicate stays as a named False rather than vanishing into its call site, because it is what a
    reader looks for when asking "does anything still block ambient vision"."""
    from slack_client.event_handlers.message_events import SlackMessageEventsMixin

    host = SlackMessageEventsMixin.__new__(SlackMessageEventsMixin)
    for event in ({"files": [{"mimetype": "image/png", "name": "x.png"}]},
                  {"files": []}, {}, {"subtype": "file_share"}):
        assert SlackMessageEventsMixin._gate_will_see_images(host, event) is False


@pytest.mark.asyncio
async def test_no_message_touches_the_ambient_service_from_the_gate(monkeypatch):
    """Widened from "a text-only message never piggybacks" to "nothing does".

    The piggyback existed so ONE look at an image served both the verdict and the stored
    description. The gate takes no look, so it has nothing to hand over — and with an IMAGE
    attached, which is the case that used to call `resolve_gate`, it still calls nothing. Note what
    is also asserted: no download. Nothing is fetched for a decision that cannot use it."""
    monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
    from message_processor.participation import ParticipationEngine

    class _Svc:
        def __init__(self):
            self.calls = []

        def resolve_gate(self, channel_id, source_ts, observations):
            self.calls.append((channel_id, source_ts, dict(observations)))

    class _Client:
        def __init__(self, svc):
            self._svc = svc
            self.download_file = AsyncMock(return_value=PNG)

        def _ambient_service(self):
            return self._svc

    svc = _Svc()
    client = _Client(svc)
    openai = MagicMock()
    openai.classify_wake = AsyncMock(return_value=True)
    engine = ParticipationEngine(openai)

    for text in ("just chatting", "what do we think?"):
        await engine.evaluate(channel_id="C1", ts="10.0", text=text, sender_id="U1",
                              attachments=["meme.png (image)"], client=client)
    assert svc.calls == []
    client.download_file.assert_not_awaited()


def test_api_part_now_stamps_the_configured_detail():
    """Regression: DEFAULT_DETAIL_LEVEL existed but never reached the main turn's image parts —
    the builders set no `detail`, so every attached image rode at the API default (`auto`, which
    downsamples) regardless of the setting. That is how a rollback token got transcribed with the
    wrong last character and then repeated as fact. api_part is the one choke point every content
    part crosses, so the default is stamped there; an explicit detail still wins."""
    from message_processor.utilities import api_part
    from config import config
    part = api_part({"type": "input_image", "image_url": "data:x",
                     "source": "attachment", "url": "u", "file_id": "F1"})
    assert part["detail"] == config.default_detail_level
    assert "source" not in part and "file_id" not in part      # whitelist still enforced
    assert api_part({"type": "input_image", "image_url": "d", "detail": "low"})["detail"] == "low"


def test_the_answering_path_defaults_to_full_fidelity():
    """`auto` on the 5.6 family means ORIGINAL dimensions with no resize — the maximum. Pinning it
    to `high` would cap large screenshots, not sharpen them.

    The companion assertion (GATE_VISION_DETAIL defaults to `high`) is dropped: it was justified by
    the gate's observations becoming the image's durable stored description, and the gate produces
    no observations. The setting still exists and is read by gate_vision, which has no runtime
    caller — pinning the default of a knob nothing turns would only make the cleanup commit look
    like a regression."""
    from config import BotConfig
    import os
    from unittest.mock import patch
    with patch.dict(os.environ, {}, clear=True):
        fresh = BotConfig()
    assert fresh.default_detail_level == "auto"
