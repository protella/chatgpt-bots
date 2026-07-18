"""F13 — catalog_uploads stores a DISTINCT visual description per uploaded image.

The old path made a single aggregate analyze_images call over ALL uploaded images and then saved
that one blurb as the `analysis` of every url. Three uploaded screenshots became three identical
catalog entries, so "edit the second one" had nothing to disambiguate on — and editing the wrong
image is exactly the expensive, irreversible mistake the catalog exists to prevent.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from message_processor import image_catalog

pytestmark = pytest.mark.unit


def _part(url, filename):
    return {"type": "input_image", "image_url": "data:image/png;base64,AAAA",
            "source": "attachment", "filename": filename, "url": url, "file_id": filename}


def _attach(url):
    return {"type": "image", "url": url}


class _DB:
    def __init__(self):
        self.saved = []

    async def save_image_metadata_async(self, **kwargs):
        self.saved.append(kwargs)


def _processor(analyze):
    return SimpleNamespace(db=_DB(), openai_client=SimpleNamespace(analyze_images=analyze))


@pytest.mark.asyncio
async def test_each_image_gets_its_own_description_keyed_by_its_url():
    # One analyze call PER image, each seeing exactly that image, saved under that image's url.
    descriptions = {"https://files.slack.com/a": "A red bar chart",
                    "https://files.slack.com/b": "A blue line graph"}

    async def _analyze(images, question, enhance_prompt=False):
        assert len(images) == 1, "each image is described on its own, not in aggregate"
        return descriptions[images[0]["url"]]

    proc = _processor(AsyncMock(side_effect=_analyze))
    parts = [_part("https://files.slack.com/a", "a.png"),
             _part("https://files.slack.com/b", "b.png")]
    attachments = [_attach("https://files.slack.com/a"), _attach("https://files.slack.com/b")]

    await image_catalog.catalog_uploads(proc, "C1:100.0", attachments, parts, message_ts="100.5")

    by_url = {row["url"]: row["analysis"] for row in proc.db.saved}
    assert by_url == {
        "https://files.slack.com/a": "A red bar chart",
        "https://files.slack.com/b": "A blue line graph",
    }
    # The whole point: the two entries are NOT the same blurb.
    assert by_url["https://files.slack.com/a"] != by_url["https://files.slack.com/b"]


@pytest.mark.asyncio
async def test_one_images_failed_description_does_not_sink_the_others():
    async def _analyze(images, question, enhance_prompt=False):
        if images[0]["url"].endswith("bad"):
            raise RuntimeError("vision 500")
        return "A good description"

    proc = _processor(AsyncMock(side_effect=_analyze))
    parts = [_part("https://files.slack.com/bad", "bad.png"),
             _part("https://files.slack.com/good", "good.png")]
    attachments = [_attach("https://files.slack.com/bad"), _attach("https://files.slack.com/good")]

    await image_catalog.catalog_uploads(proc, "C1:100.0", attachments, parts)

    assert [row["url"] for row in proc.db.saved] == ["https://files.slack.com/good"]
    assert proc.db.saved[0]["analysis"] == "A good description"


@pytest.mark.asyncio
async def test_an_empty_description_is_not_persisted():
    proc = _processor(AsyncMock(return_value=""))
    parts = [_part("https://files.slack.com/a", "a.png")]
    await image_catalog.catalog_uploads(proc, "C1:100.0", [_attach("https://files.slack.com/a")],
                                        parts)
    assert proc.db.saved == []
