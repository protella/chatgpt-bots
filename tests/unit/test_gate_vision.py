"""F40 — the wake gate has to look at the picture before it reacts to it.

The bug, live (2026-07-13): a meme was posted with the caption ":dogkek:" and nothing else. The
gate returned `{"action":"react","emoji":"joy","reason":"A laughing reaction fits the playful
meme post."}` — a confident opinion about an image it had never seen. It had reasoned from the
emoji SHORTCODE in the caption, and from a prompt line that told it, untruthfully, that it "can
view and analyze attachments".

Reacting to a picture you haven't looked at is the same dishonesty as an 👀 with no work behind
it. So: if an image is attached and we can safely show it, the gate sees it — and when we
CAN'T show it, the gate is told so in plain words instead of being left to guess.
"""

import asyncio
import base64
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import config
from message_processor import gate_vision
from message_processor.participation import ParticipationEngine

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff" + b"\x00" * 64
HTML = b"<!DOCTYPE html><html><body>Sign in to Slack"


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


# --------------------------------------------------------------------- the prompt tells the truth

class _FakeOpenAI:
    """Enough of the OpenAI client for the real classify_participation to run against."""

    def __init__(self, fail_first=False):
        self.calls = []
        self.fail_first = fail_first
        self.client = MagicMock()

    async def _safe_api_call(self, fn, operation_type=None, **params):
        self.calls.append(params)
        if self.fail_first and len(self.calls) == 1:
            raise RuntimeError("400 unsupported image format")
        out = MagicMock()
        content = MagicMock()
        content.text = '{"action":"react","emoji":"joy"}'
        item = MagicMock()
        item.content = [content]
        out.output = [item]
        return out

    def log_debug(self, *a, **k):
        pass

    def log_warning(self, *a, **k):
        pass


def _rendered(params) -> str:
    """The user block the model actually receives, however it was structured."""
    block = params["input"][-1]["content"]
    if isinstance(block, list):
        return " ".join(p.get("text", "") for p in block if p.get("type") == "input_text")
    return block


async def _classify(images, status, fail_first=False):
    from openai_client.api.responses import classify_participation
    fake = _FakeOpenAI(fail_first=fail_first)
    verdict = await classify_participation(
        fake, text=":dogkek:",
        signals={"attachments": "1 image (meme.png)", "image_status": status},
        images=images)
    return fake, verdict


@pytest.mark.asyncio
async def test_when_the_model_can_see_the_image_the_prompt_says_so():
    parts, _, _shown = await gate_vision.load_for_gate(_client(PNG), [_img()])
    fake, verdict = await _classify(parts, gate_vision.VISIBLE)

    prompt = _rendered(fake.calls[0])
    assert "shown to you below" in prompt
    assert "untrusted" in prompt, "text inside an image is content under discussion, not orders"
    # The image rides as its own content part — never interpolated into the prompt string.
    content = fake.calls[0]["input"][-1]["content"]
    assert isinstance(content, list)
    assert [p["type"] for p in content] == ["input_text", "input_image"]
    assert "base64" not in prompt
    assert verdict["action"] == "react"


@pytest.mark.asyncio
async def test_when_it_cannot_see_the_image_it_is_told_not_to_guess():
    """The old prompt said "The assistant can view and analyze attachments" unconditionally — a
    claim about the ANSWERING model, fed to a classifier that could see nothing. The model took
    it as licence to have an opinion about the picture. That IS the :dogkek: bug."""
    fake, _ = await _classify([], gate_vision.UNAVAILABLE)

    prompt = _rendered(fake.calls[0])
    assert "CANNOT see it" in prompt
    assert "filename" in prompt and "emoji in the caption" in prompt
    assert "can view and analyze attachments" not in prompt


@pytest.mark.asyncio
async def test_an_image_the_api_rejects_costs_us_the_picture_not_the_wake():
    """If the image itself 400s, judging on the text is a far better outcome than losing the
    wake entirely — but the retry must TELL the model it is now blind."""
    parts, _, _shown = await gate_vision.load_for_gate(_client(PNG), [_img()])
    fake, verdict = await _classify(parts, gate_vision.VISIBLE, fail_first=True)

    assert len(fake.calls) == 2, "it should have retried without the image"
    retry = fake.calls[1]["input"][-1]["content"]
    assert isinstance(retry, str), "the retry carries no image parts"
    assert "CANNOT see it" in retry
    # The retry must RE-RENDER the whole signal block, not bolt a correction onto the old one.
    # Appending "you can't see it" to a block that still said "shown to you below" left the
    # model holding two contradictory claims and no image to resolve them against.
    assert "shown to you below" not in retry, "the retry still claims the image is visible"
    assert verdict["action"] == "react", "the wake survived"


@pytest.mark.asyncio
async def test_an_outage_is_not_retried_as_if_the_image_were_at_fault():
    """Retrying on ANY exception meant a timeout or a 429 bought a second 30-second utility
    call — doubling the stall on the debounce hot path for a request that was never going to
    succeed. Only an image REJECTION earns the text-only retry."""
    from openai_client.api.responses import classify_participation
    parts, _, _shown = await gate_vision.load_for_gate(_client(PNG), [_img()])

    class _Down(_FakeOpenAI):
        async def _safe_api_call(self, fn, operation_type=None, **params):
            self.calls.append(params)
            raise TimeoutError("upstream connect timeout")

    fake = _Down()
    verdict = await classify_participation(
        fake, text="hi", signals={"attachments": "1 image (meme.png)",
                                  "image_status": gate_vision.VISIBLE}, images=parts)

    assert len(fake.calls) == 1, "an outage must not be retried as an image problem"
    assert verdict == {"action": "ignore"}, "and it still fails safe"


# --------------------------------------------------------------------- engine wiring

@pytest.mark.asyncio
async def test_nothing_downloads_until_the_message_survives_the_debounce(monkeypatch):
    """A superseded burst must not spend bandwidth fetching pictures for a verdict that is
    thrown away — so the download lives AFTER the supersession check, not before it."""
    monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
    openai = MagicMock()
    openai.classify_participation = AsyncMock(return_value={"action": "ignore"})
    engine = ParticipationEngine(openai)
    client = _client(PNG)

    # A newer message in the same stream lands during our debounce window.
    async def supersede():
        engine.note_arrival("C1", "20.0", None, "U1")

    import asyncio
    task = asyncio.create_task(supersede())
    verdict = await engine.evaluate(
        channel_id="C1", ts="10.0", text=":dogkek:", sender_id="U1",
        images=[_img()], client=client)
    await task

    assert verdict is None, "superseded"
    client.download_file.assert_not_awaited()
    openai.classify_participation.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_dogkek_case_the_gate_now_sees_the_meme(monkeypatch):
    """End to end: caption is a single emoji shortcode, all the meaning is in the picture."""
    monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
    monkeypatch.setattr(config, "enable_multimodal_gate", True, raising=False)
    openai = MagicMock()
    openai.classify_participation = AsyncMock(return_value={"action": "react", "emoji": "joy"})
    engine = ParticipationEngine(openai)

    await engine.evaluate(channel_id="C1", ts="10.0", text=":dogkek:", sender_id="U1",
                          images=[_img()], client=_client(PNG))

    kwargs = openai.classify_participation.await_args.kwargs
    assert kwargs["images"], "the gate reacted to a meme without being shown the meme"
    assert kwargs["images"][0]["type"] == "input_image"
    assert kwargs["signals"]["image_status"] == gate_vision.VISIBLE


@pytest.mark.asyncio
async def test_an_unreadable_image_still_gets_a_verdict_just_a_blind_one(monkeypatch):
    """The gate must never go silent because a file wouldn't load — that drops the wake."""
    monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
    openai = MagicMock()
    openai.classify_participation = AsyncMock(return_value={"action": "ignore"})
    engine = ParticipationEngine(openai)
    c = MagicMock()
    c.download_file = AsyncMock(side_effect=RuntimeError("nope"))

    verdict = await engine.evaluate(channel_id="C1", ts="10.0", text="look at this",
                                    sender_id="U1", images=[_img()], client=c)

    assert verdict is not None, "a broken download must not swallow the wake"
    kwargs = openai.classify_participation.await_args.kwargs
    assert not kwargs.get("images")
    assert kwargs["signals"]["image_status"] == gate_vision.UNAVAILABLE


@pytest.mark.asyncio
async def test_a_text_only_message_is_completely_unaffected(monkeypatch):
    monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
    openai = MagicMock()
    openai.classify_participation = AsyncMock(return_value={"action": "ignore"})
    engine = ParticipationEngine(openai)

    await engine.evaluate(channel_id="C1", ts="10.0", text="just chatting", sender_id="U1")

    kwargs = openai.classify_participation.await_args.kwargs
    assert "images" not in kwargs, (
        "a text-only judgment must keep its exact old call shape — not one token different")
    assert kwargs["signals"]["image_status"] == gate_vision.NONE


# ------------------------------------------------ F51b gate/ambient piggyback (engine seam)

class _RecordingService:
    """Records resolve_gate calls — stands in for the AmbientArtifactService the gate reaches
    through the Slack facade."""

    def __init__(self):
        self.calls = []

    def resolve_gate(self, channel_id, source_ts, observations):
        self.calls.append((channel_id, source_ts, dict(observations)))


class _GateClient:
    def __init__(self, payload=PNG, svc=None):
        self._svc = svc
        self.download_file = AsyncMock(return_value=payload)

    def _ambient_service(self):
        return self._svc


def test_harvest_image_observations_maps_and_drops_defensively():
    h = ParticipationEngine._harvest_image_observations
    shown = [{"id": "F1"}, {"id": "F2"}]
    # exact count, both usable → mapped by file id, in order
    assert h({"image_observations": ["a", "b"]}, shown) == {"F1": "a", "F2": "b"}
    # wrong count → ALL dropped (order can't be trusted); the worker covers them
    assert h({"image_observations": ["only one"]}, shown) == {}
    # missing / non-list / non-dict → {}
    assert h({"action": "ignore"}, shown) == {}
    assert h({"image_observations": "nope"}, shown) == {}
    assert h(None, shown) == {}
    # correctly-sized but a blank / non-string entry is skipped; the valid one is kept
    assert h({"image_observations": ["  ", "keep"]}, shown) == {"F2": "keep"}
    assert h({"image_observations": ["a", 5]}, shown) == {"F1": "a"}
    # no images shown → nothing to harvest
    assert h({"image_observations": ["a"]}, []) == {}


@pytest.mark.asyncio
async def test_engine_piggybacks_observations_to_ambient(monkeypatch):
    """The gate looked at the image; its per-image observation is handed to ambient memory keyed
    by Slack file id — one look serving both the verdict and the stored observation."""
    monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
    monkeypatch.setattr(config, "enable_multimodal_gate", True, raising=False)
    svc = _RecordingService()
    client = _GateClient(PNG, svc)
    openai = MagicMock()
    openai.classify_participation = AsyncMock(return_value={
        "action": "ignore",
        "image_observations": ["A screenshot of a terminal showing a Python stack trace."]})
    engine = ParticipationEngine(openai)

    verdict = await engine.evaluate(channel_id="C1", ts="10.0", text="see this log",
                                    sender_id="U1", images=[_img()], client=client)

    assert verdict.action == "ignore"
    assert svc.calls == [("C1", "10.0",
                          {"F1": "A screenshot of a terminal showing a Python stack trace."})]


@pytest.mark.asyncio
async def test_engine_verdict_is_unaffected_by_a_broken_piggyback(monkeypatch):
    """Verdict safety is absolute: an exploding ambient service can never alter the verdict."""
    monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
    monkeypatch.setattr(config, "enable_multimodal_gate", True, raising=False)

    class _Boom:
        def resolve_gate(self, *a, **k):
            raise RuntimeError("ambient is down")

    client = _GateClient(PNG, _Boom())
    openai = MagicMock()
    openai.classify_participation = AsyncMock(return_value={"action": "react", "emoji": "eyes"})
    engine = ParticipationEngine(openai)

    verdict = await engine.evaluate(channel_id="C1", ts="10.0", text="x", sender_id="U1",
                                    images=[_img()], client=client)
    assert verdict.action == "react" and verdict.emoji == "eyes"


@pytest.mark.asyncio
async def test_engine_releases_when_the_gate_is_blind(monkeypatch):
    """Images attached but unshowable → the gate produces no observations and resolve_gate is
    called with {}, releasing the held jobs to the ordinary vision worker."""
    monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
    svc = _RecordingService()
    c = MagicMock()
    c.download_file = AsyncMock(side_effect=RuntimeError("nope"))
    c._ambient_service = lambda: svc
    openai = MagicMock()
    openai.classify_participation = AsyncMock(return_value={"action": "ignore"})
    engine = ParticipationEngine(openai)

    await engine.evaluate(channel_id="C1", ts="10.0", text="x", sender_id="U1",
                          images=[_img()], client=c)
    assert svc.calls == [("C1", "10.0", {})]


@pytest.mark.asyncio
async def test_engine_releases_held_images_on_supersession(monkeypatch):
    """A superseded burst releases its held images promptly (not left to the hold timeout), and
    still never downloads a picture for a verdict it throws away."""
    monkeypatch.setattr(config, "participation_debounce_seconds", 0.05, raising=False)
    svc = _RecordingService()
    client = _GateClient(PNG, svc)
    openai = MagicMock()
    openai.classify_participation = AsyncMock(return_value={"action": "ignore"})
    engine = ParticipationEngine(openai)

    async def supersede():
        engine.note_arrival("C1", "20.0", None, "U1")

    task = asyncio.create_task(supersede())
    verdict = await engine.evaluate(channel_id="C1", ts="10.0", text=":x:", sender_id="U1",
                                    images=[_img()], client=client)
    await task

    assert verdict is None
    assert svc.calls == [("C1", "10.0", {})]
    client.download_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_text_only_message_never_touches_the_ambient_service(monkeypatch):
    """No images → no piggyback at all; the ambient service is never even fetched or called."""
    monkeypatch.setattr(config, "participation_debounce_seconds", 0, raising=False)
    svc = _RecordingService()
    client = _GateClient(PNG, svc)
    openai = MagicMock()
    openai.classify_participation = AsyncMock(return_value={"action": "ignore"})
    engine = ParticipationEngine(openai)

    await engine.evaluate(channel_id="C1", ts="10.0", text="just chatting",
                          sender_id="U1", client=client)
    assert svc.calls == []
