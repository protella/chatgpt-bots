"""A DM is one conversation; Slack just splits it into roots.

Every top-level DM message carries no `thread_ts`, so `thread_id = ts` and each message becomes
its own thread key. Send a picture as one message and ask about it in the next, and the picture
lives under a different key than the request — so the strictly-scoped image catalog came back
EMPTY, `edit_image` and `view_image` (both schema factories that return None on an empty catalog)
were never offered, and the model, left holding only `generate_image`, re-imagined the picture
from scratch instead of editing the one it was asked about.

The widening is DM-only and one-DM-only. Channels stay strict: there a thread is a real
conversation boundary, not an accident of the surface.
"""
import pytest

from message_processor import image_catalog

pytestmark = pytest.mark.unit


def _row(row_id, url, analysis="a screenshot", image_type="uploaded"):
    return {"id": row_id, "url": url, "analysis": analysis, "prompt": "",
            "image_type": image_type, "created_at": "2026-07-24 10:00:00"}


class _DB:
    """Records how it was asked, so scope can be asserted, not assumed."""

    def __init__(self, thread_rows=None, channel_rows=None):
        self.thread_rows = list(thread_rows or [])
        self.channel_rows = list(channel_rows or [])
        self.channel_calls = []

    async def find_thread_images_async(self, thread_id, image_type=None):
        return self.thread_rows

    async def find_channel_images_async(self, channel_id, within_hours=None, limit=50):
        self.channel_calls.append({"channel_id": channel_id, "within_hours": within_hours,
                                   "limit": limit})
        return self.channel_rows


@pytest.mark.asyncio
async def test_a_dm_finds_an_image_sent_in_the_previous_message():
    # The exact live failure: image in message A, "edit that" in message B.
    db = _DB(thread_rows=[], channel_rows=[_row(7, "https://files.slack.com/deploy.png")])

    entries = await image_catalog.build_catalog(db, "D08EDPS3QMC:1784925818.611379")

    assert [e["image_id"] for e in entries] == ["img_7"]
    assert entries[0]["origin"] == "earlier in this DM"
    assert image_catalog.valid_ids(entries) == ["img_7"], "so edit_image/view_image are OFFERED"


@pytest.mark.asyncio
async def test_the_widening_is_scoped_to_this_one_dm():
    db = _DB(thread_rows=[], channel_rows=[])
    await image_catalog.build_catalog(db, "D08EDPS3QMC:1784925818.611379")

    assert db.channel_calls == [{"channel_id": "D08EDPS3QMC",
                                 "within_hours": image_catalog.DM_LOOKBACK_HOURS,
                                 "limit": image_catalog.MAX_CATALOG * 2}]


@pytest.mark.asyncio
async def test_channels_stay_strict():
    """A channel thread IS a conversation boundary. Widening there would be a scope change."""
    db = _DB(thread_rows=[], channel_rows=[_row(7, "https://files.slack.com/other.png")])

    entries = await image_catalog.build_catalog(db, "C0BKX77NU66:1784921906.654579")

    assert entries == []
    assert db.channel_calls == [], "the channel-wide lookup is never even issued"


@pytest.mark.asyncio
async def test_this_threads_own_images_come_first_and_are_not_duplicated():
    same = "https://files.slack.com/same.png"
    db = _DB(thread_rows=[_row(1, same)],
             channel_rows=[_row(1, same), _row(2, "https://files.slack.com/older.png")])

    entries = await image_catalog.build_catalog(db, "D1:100.0")

    assert [e["image_id"] for e in entries] == ["img_1", "img_2"]
    assert "origin" not in entries[0], "this message's own image is not 'earlier'"
    assert entries[1]["origin"] == "earlier in this DM"


@pytest.mark.asyncio
async def test_the_widening_respects_the_catalog_cap():
    db = _DB(thread_rows=[_row(i, f"https://files.slack.com/{i}.png")
                          for i in range(image_catalog.MAX_CATALOG)],
             channel_rows=[_row(99, "https://files.slack.com/extra.png")])

    entries = await image_catalog.build_catalog(db, "D1:100.0")

    assert len(entries) == image_catalog.MAX_CATALOG
    assert "img_99" not in [e["image_id"] for e in entries]
    assert db.channel_calls == [], "no room left — the lookup is skipped entirely"


@pytest.mark.asyncio
async def test_a_thread_key_with_colons_on_both_sides_resolves_the_channel():
    # CLAUDE.md pitfall #3: thread keys are "channel:thread_ts" and the ts contains a dot, but a
    # naive split on ":" from the right would hand us the timestamp, not the channel.
    db = _DB(thread_rows=[], channel_rows=[_row(3, "https://files.slack.com/a.png")])
    await image_catalog.build_catalog(db, "D08EDPS3QMC:1784925818.611379")
    assert db.channel_calls[0]["channel_id"] == "D08EDPS3QMC"


@pytest.mark.asyncio
async def test_a_failed_widening_still_returns_the_strict_catalog():
    class _Broken(_DB):
        async def find_channel_images_async(self, channel_id, within_hours=None, limit=50):
            raise RuntimeError("database is locked")

    db = _Broken(thread_rows=[_row(1, "https://files.slack.com/a.png")])
    entries = await image_catalog.build_catalog(db, "D1:100.0")
    assert [e["image_id"] for e in entries] == ["img_1"]


@pytest.mark.asyncio
async def test_a_db_without_the_lookup_degrades_quietly():
    class _Old:
        async def find_thread_images_async(self, thread_id, image_type=None):
            return [_row(1, "https://files.slack.com/a.png")]

    entries = await image_catalog.build_catalog(_Old(), "D1:100.0")
    assert [e["image_id"] for e in entries] == ["img_1"]


def test_a_widened_entry_is_labelled_in_the_tool_description():
    """The model has to be able to tell 'the image I just sent' from 'that one from earlier'."""
    lines = image_catalog.catalog_lines([
        {"image_id": "img_9", "kind": "uploaded", "analysis": "this turn's picture",
         "prompt": ""},
        {"image_id": "img_7", "kind": "uploaded", "analysis": "yesterday's picture",
         "prompt": "", "origin": "earlier in this DM"},
    ])
    assert "img_9 (most recent) — uploaded: this turn's picture" in lines
    assert "img_7 [earlier in this DM] — uploaded: yesterday's picture" in lines
