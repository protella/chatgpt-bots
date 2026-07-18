"""The image-analysis 400, caught live on 2026-07-13.

`analyze_images` was written expecting bare base64 STRINGS:

    content.append({"type": "input_image", "image_url": f"data:image/png;base64,{image_data}"})

but its only caller (image_catalog) holds images as the pipeline's attachment PARTS — dicts of
{type, image_url, source, filename, url, file_id}. Interpolating a dict into that f-string
produced `data:image/png;base64,{'type': 'input_image', ...}` and the API answered:

    400 Invalid 'input[0].content[1].image_url'. Expected a base64-encoded data URL with an
    image MIME type, but got an invalid base64-encoded value.

`analyze_images` swallows its own exceptions, so this never surfaced: it just meant that every
uploaded image's analysis silently failed and the thread quietly lost that context later. This
is CLAUDE.md pitfall #5 ("never send our attachment dicts straight to the API") in a new dress.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from openai_client.api.vision import analyze_images

B64 = "aVZCT1J3MEtHZ29BQUFBTlNVaEVVZw=="


class _FakeClient:
    def __init__(self):
        self.captured = None
        self.client = MagicMock()

    async def _safe_api_call(self, fn, operation_type=None, **params):
        self.captured = params
        out = MagicMock()
        content = MagicMock()
        content.text = "a chart"
        item = MagicMock()
        item.content = [content]
        out.output = [item]
        out.output_text = "a chart"
        return out

    def log_warning(self, *a, **k):
        pass

    def log_debug(self, *a, **k):
        pass

    def log_info(self, *a, **k):
        pass

    def log_error(self, *a, **k):
        pass


def _image_parts(params):
    for msg in params["input"]:
        content = msg.get("content")
        if isinstance(content, list):
            return [p for p in content if p.get("type") == "input_image"]
    return []


@pytest.mark.asyncio
async def test_the_attachment_part_shape_the_pipeline_actually_passes():
    """THE BUG. This is exactly what image_catalog hands it."""
    c = _FakeClient()
    part = {"type": "input_image", "image_url": f"data:image/png;base64,{B64}",
            "source": "attachment", "filename": "chart.png",
            "url": "https://files.slack.com/x", "file_id": "F1"}

    await analyze_images(c, images=[part], question="what is this?")

    got = _image_parts(c.captured)[0]["image_url"]
    assert got == f"data:image/png;base64,{B64}", (
        f"a dict was stringified into the data URL — the API rejects this: {got[:60]!r}")
    assert "'type'" not in got and "{" not in got


@pytest.mark.asyncio
async def test_a_jpeg_is_not_announced_to_the_api_as_a_png():
    """The quieter half of the same bug: the mimetype was hardcoded, so a JPEG went out
    labelled image/png. The part carries its real type — honour it."""
    c = _FakeClient()
    part = {"type": "input_image", "image_url": f"data:image/jpeg;base64,{B64}"}

    await analyze_images(c, images=[part], question="?")

    assert _image_parts(c.captured)[0]["image_url"].startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_bare_base64_still_works():
    """The shape the function was originally written for must keep working."""
    c = _FakeClient()
    await analyze_images(c, images=[B64], question="?")
    assert _image_parts(c.captured)[0]["image_url"] == f"data:image/png;base64,{B64}"


@pytest.mark.asyncio
async def test_with_no_usable_image_it_refuses_to_ask_rather_than_invent_one():
    """The nastiest failure mode of the bug above. If every part is unusable we used to send the
    question ANYWAY — "describe this image" with no image — and the model, being helpful, would
    describe one. `catalog_uploads` then persisted that as a genuine analysis: a confident
    description, in the database, of a picture nobody ever sent."""
    c = _FakeClient()

    out = await analyze_images(c, images=[{"type": "input_image"}, {}], question="describe this")

    assert c.captured is None, "it asked the model to imagine an image"
    assert out == ""


@pytest.mark.asyncio
async def test_the_parts_detail_is_honoured():
    """`detail` was computed and then never sent — every analysis silently ran at the default."""
    c = _FakeClient()
    await analyze_images(
        c, images=[{"type": "input_image", "image_url": f"data:image/png;base64,{B64}",
                    "detail": "high"}], question="?")
    assert _image_parts(c.captured)[0].get("detail") == "high"


@pytest.mark.asyncio
@pytest.mark.parametrize("model,effort,expected", [
    ("gpt-5.6-sol", "minimal", "none"),   # F27: 'minimal' is a hard 400 on 5.6 -> none
    ("gpt-5.6-sol", "high", "high"),      # a legal effort passes through untouched
    ("gpt-5.5", "max", "xhigh"),          # 5.5 has no 'max' -> xhigh
])
async def test_reasoning_effort_is_clamped_for_the_target_model(model, effort, expected):
    """F27: this was the ONLY vision call site that skipped clamp_effort, so a stored/legacy
    'minimal' would reach the API and 400 the whole analysis. The effort must be clamped for the
    model actually being called, in both the streaming and non-streaming request builders (they
    share the one clamped value)."""
    c = _FakeClient()
    await analyze_images(c, images=[f"data:image/png;base64,{B64}"], question="?",
                         model=model, reasoning_effort=effort)
    assert c.captured["reasoning"]["effort"] == expected


@pytest.mark.asyncio
async def test_a_ready_made_data_url_is_not_double_wrapped():
    c = _FakeClient()
    await analyze_images(c, images=[f"data:image/webp;base64,{B64}"], question="?")
    got = _image_parts(c.captured)[0]["image_url"]
    assert got == f"data:image/webp;base64,{B64}"
    assert got.count("data:") == 1


# ---------------------------------------------------------------- F9: streaming terminal states
#
# Pitfall #6: every mock stream below is a FINITE generator yielding real strings, so a stale
# side_effect can never spin an unbounded async iterator.


class _StreamClient:
    """Drives analyze_images' streaming branch off a fixed list of events."""

    def __init__(self, events):
        self._events = events
        self.client = MagicMock()

    async def _safe_api_call(self, fn, operation_type=None, **params):
        return SimpleNamespace()  # the stream handle; our fake iterator ignores it

    def _safe_stream_iteration(self, stream, operation_type="streaming"):
        async def _iter():
            for e in self._events:
                yield e
        return _iter()

    def log_debug(self, *a, **k):
        pass

    def log_warning(self, *a, **k):
        pass

    def log_info(self, *a, **k):
        pass

    def log_error(self, *a, **k):
        pass


class _Recorder:
    """Records streamed chunks and whether the terminal flush (stream_callback(None)) fired."""

    def __init__(self):
        self.chunks = []
        self.flushed = False

    def __call__(self, chunk):
        if chunk is None:
            self.flushed = True
        else:
            self.chunks.append(chunk)


def _delta(text):
    return SimpleNamespace(type="response.output_text.delta", delta=text)


@pytest.mark.asyncio
async def test_streaming_completed_flushes_and_returns_text():
    """The baseline terminal state must keep working after the branch was widened."""
    c = _StreamClient([_delta("hello "), _delta("world"),
                       SimpleNamespace(type="response.completed")])
    rec = _Recorder()
    out = await analyze_images(c, images=[f"data:image/png;base64,{B64}"], question="?",
                               stream_callback=rec)
    assert out == "hello world"
    assert rec.flushed is True


@pytest.mark.asyncio
async def test_streaming_incomplete_flushes_and_returns_partial():
    """F9: response.incomplete (e.g. hit max_output_tokens) is terminal too — it must still fire
    the final flush and return the partial text, not hang the consumer waiting for a signal that
    matches only done/completed."""
    c = _StreamClient([_delta("partial "), _delta("answer"),
                       SimpleNamespace(type="response.incomplete")])
    rec = _Recorder()
    out = await analyze_images(c, images=[f"data:image/png;base64,{B64}"], question="?",
                               stream_callback=rec)
    assert out == "partial answer"
    assert rec.flushed is True


@pytest.mark.asyncio
async def test_streaming_failed_flushes_then_raises():
    """F9: response.failed is terminal and an ERROR — it fires the flush (so the consumer isn't
    left hanging) and then propagates, consistent with analyze_images' outer error handling. A
    silent "" would masquerade as a clean empty answer."""
    c = _StreamClient([_delta("half "), SimpleNamespace(type="response.failed")])
    rec = _Recorder()
    with pytest.raises(RuntimeError, match="response.failed"):
        await analyze_images(c, images=[f"data:image/png;base64,{B64}"], question="?",
                             stream_callback=rec)
    assert rec.flushed is True          # the consumer still got its terminal signal
