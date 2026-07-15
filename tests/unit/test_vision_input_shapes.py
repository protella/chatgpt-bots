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
async def test_a_ready_made_data_url_is_not_double_wrapped():
    c = _FakeClient()
    await analyze_images(c, images=[f"data:image/webp;base64,{B64}"], question="?")
    got = _image_parts(c.captured)[0]["image_url"]
    assert got == f"data:image/webp;base64,{B64}"
    assert got.count("data:") == 1
