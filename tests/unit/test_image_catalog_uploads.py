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
    def __init__(self, existing=None):
        self.saved = []
        # Rows already in the catalog, as find_thread_images_async returns them.
        self.existing = list(existing or [])

    async def save_image_metadata_async(self, **kwargs):
        self.saved.append(kwargs)

    async def find_thread_images_async(self, thread_id, image_type=None):
        return self.existing


def _processor(analyze, existing=None):
    return SimpleNamespace(db=_DB(existing),
                           openai_client=SimpleNamespace(analyze_images=analyze))


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

    await image_catalog.catalog_uploads(proc, "C1:100.0", parts, message_ts="100.5")

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

    await image_catalog.catalog_uploads(proc, "C1:100.0", parts)

    assert [row["url"] for row in proc.db.saved] == ["https://files.slack.com/good"]
    assert proc.db.saved[0]["analysis"] == "A good description"


@pytest.mark.asyncio
async def test_an_empty_description_is_not_persisted():
    proc = _processor(AsyncMock(return_value=""))
    parts = [_part("https://files.slack.com/a", "a.png")]
    await image_catalog.catalog_uploads(proc, "C1:100.0", parts)
    assert proc.db.saved == []


@pytest.mark.asyncio
async def test_an_already_described_image_costs_no_vision_call():
    """The participation gate's observation lands in images.analysis first, and the upsert is
    merge-preserving — so describing it again computed a primary-model description that the DB
    then threw away. Read the catalog first and spend nothing on what is already described."""
    analyze = AsyncMock(return_value="a fresh description")
    proc = _processor(analyze, existing=[
        {"url": "https://files.slack.com/a", "analysis": "the gate already described this"},
    ])

    await image_catalog.catalog_uploads(
        proc, "C1:100.0", [_part("https://files.slack.com/a", "a.png")])

    analyze.assert_not_awaited()
    assert proc.db.saved == [], "no write either — the existing description stands"


@pytest.mark.asyncio
async def test_a_blank_existing_analysis_is_not_a_description():
    """catalog_unattended and the URL-borne path both pre-create the row with an EMPTY analysis.
    Treating that as 'already described' would leave those images permanently undescribed."""
    analyze = AsyncMock(return_value="the real description")
    proc = _processor(analyze, existing=[
        {"url": "https://files.slack.com/a", "analysis": "   "},
        {"url": "https://files.slack.com/b", "analysis": None},
    ])

    await image_catalog.catalog_uploads(
        proc, "C1:100.0", [_part("https://files.slack.com/a", "a.png"),
                           _part("https://files.slack.com/b", "b.png")])

    assert analyze.await_count == 2
    assert {row["url"] for row in proc.db.saved} == {"https://files.slack.com/a",
                                                     "https://files.slack.com/b"}


@pytest.mark.asyncio
async def test_only_the_undescribed_images_are_described():
    analyze = AsyncMock(return_value="new")
    proc = _processor(analyze, existing=[
        {"url": "https://files.slack.com/a", "analysis": "already known"},
    ])

    await image_catalog.catalog_uploads(
        proc, "C1:100.0", [_part("https://files.slack.com/a", "a.png"),
                           _part("https://files.slack.com/b", "b.png")])

    assert analyze.await_count == 1
    assert [row["url"] for row in proc.db.saved] == ["https://files.slack.com/b"]


@pytest.mark.asyncio
async def test_a_link_borne_image_is_described_too():
    """An image pulled from a URL carries `original_url`, not `url`. Reading only `url` skipped
    every one of them, so a linked image entered the catalog with no description at all."""
    analyze = AsyncMock(return_value="the linked picture")
    proc = _processor(analyze)
    part = {"type": "input_image", "image_url": "data:image/png;base64,AAAA",
            "source": "url", "original_url": "https://example.com/chart.png"}

    await image_catalog.catalog_uploads(proc, "C1:100.0", [part])

    assert analyze.await_count == 1
    assert proc.db.saved[0]["url"] == "https://example.com/chart.png"
    assert proc.db.saved[0]["analysis"] == "the linked picture"


@pytest.mark.asyncio
async def test_the_same_url_twice_in_one_turn_is_described_once():
    analyze = AsyncMock(return_value="once")
    proc = _processor(analyze)
    parts = [_part("https://files.slack.com/a", "a.png"),
             {"type": "input_image", "source": "url",
              "original_url": "https://files.slack.com/a"}]

    await image_catalog.catalog_uploads(proc, "C1:100.0", parts)

    assert analyze.await_count == 1
    assert len(proc.db.saved) == 1


@pytest.mark.asyncio
async def test_an_unreadable_catalog_still_describes():
    """A DB failure reading existing descriptions must not silently stop cataloging."""
    class _BrokenDB(_DB):
        async def find_thread_images_async(self, thread_id, image_type=None):
            raise RuntimeError("database is locked")

    analyze = AsyncMock(return_value="described anyway")
    proc = SimpleNamespace(db=_BrokenDB(),
                           openai_client=SimpleNamespace(analyze_images=analyze))

    await image_catalog.catalog_uploads(
        proc, "C1:100.0", [_part("https://files.slack.com/a", "a.png")])

    assert analyze.await_count == 1
    assert proc.db.saved[0]["analysis"] == "described anyway"
