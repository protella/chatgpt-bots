"""The image catalog reaches as far as the conversation does.

Two surfaces, two reasons the current thread root was never the whole story.

DM: every top-level DM message carries no `thread_ts`, so `thread_id = ts` and each message
becomes its own thread key. Send a picture as one message and ask about it in the next, and the
picture lives under a different key than the request — so the strictly-scoped catalog came back
EMPTY, `edit_image` and `view_image` (both schema factories that return None on an empty catalog)
were never offered, and the model, left holding only `generate_image`, re-imagined the picture
from scratch instead of editing it.

CHANNEL: a screenshot pasted as its own channel message is captured by ambient memory and shows
up in the channel stream as `[image analysis (ref): …]`. Asked about it from another root, the
model could SEE that the image existed and `view_image` could not resolve it — so the bot said it
could not reopen the screenshot and web-searched for the source instead. The channel widening
fixes the reach; the entries it adds are VIEW-ONLY, because capturing ambient context is not
permission to post a derivative of a colleague's image.

The surface is passed by the caller, never guessed from the key's first letter.
"""
import pytest

from message_processor import image_catalog
from message_processor.tool_registry import SURFACE_CHANNEL, SURFACE_DM

pytestmark = pytest.mark.unit


def _row(row_id, url, analysis="a screenshot", image_type="uploaded",
         created_at="2026-07-24 10:00:00"):
    return {"id": row_id, "url": url, "analysis": analysis, "prompt": "",
            "image_type": image_type, "created_at": created_at}


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

    entries = await image_catalog.build_catalog(db, "D08EDPS3QMC:1784925818.611379",
                                                surface=SURFACE_DM)

    assert [e["image_id"] for e in entries] == ["img_7"]
    assert entries[0]["origin"] == "earlier in this DM"
    assert image_catalog.valid_ids(entries) == ["img_7"], "so edit_image/view_image are OFFERED"


@pytest.mark.asyncio
async def test_the_widening_is_scoped_to_this_one_dm():
    db = _DB(thread_rows=[], channel_rows=[])
    await image_catalog.build_catalog(db, "D08EDPS3QMC:1784925818.611379", surface=SURFACE_DM)

    assert db.channel_calls == [{"channel_id": "D08EDPS3QMC",
                                 "within_hours": image_catalog.DM_LOOKBACK_HOURS,
                                 "limit": image_catalog.MAX_CATALOG * 2}]


# --- the channel widening (spec rulings 1-5) ---------------------------------------------


@pytest.mark.asyncio
async def test_a_channel_reaches_the_whole_channel_with_no_time_bound():
    """The live failure: a screenshot pasted in another root, asked about here.

    Replaces the old `test_channels_stay_strict`. Channel strictness was never an owner
    decision — it contradicted the shipped per-channel `ambient_memory` setting, which promises
    exactly this ("quietly note links, images, and files shared here for later context").
    """
    db = _DB(thread_rows=[], channel_rows=[_row(7, "https://files.slack.com/other.png")])

    entries = await image_catalog.build_catalog(db, "C0BKX77NU66:1784921906.654579",
                                                surface=SURFACE_CHANNEL)

    assert [e["image_id"] for e in entries] == ["img_7"], "the lookup is issued and it resolves"
    assert db.channel_calls == [{"channel_id": "C0BKX77NU66",
                                 "within_hours": None,
                                 "limit": image_catalog.MAX_CATALOG * 2}], (
        "no time bound: MAX_CATALOG is the only bound a channel gets")


@pytest.mark.asyncio
async def test_a_widened_channel_image_is_an_ordinary_entry_editable_like_any_other():
    """[OWNER 2026-09-16] A channel is a shared workspace: "edit that chart someone posted" is
    ordinary use of one, so a widened entry carries no reduced authority — only a label."""
    db = _DB(thread_rows=[_row(4, "https://files.slack.com/ours.png",
                               created_at="2026-07-24 09:00:00")],
             channel_rows=[_row(7, "https://files.slack.com/theirs.png",
                                created_at="2026-07-24 08:00:00")])

    entries = await image_catalog.build_catalog(db, "C0BKX77NU66:1784921906.654579",
                                                surface=SURFACE_CHANNEL)

    assert image_catalog.valid_ids(entries) == ["img_4", "img_7"]
    assert image_catalog.resolve(entries, "img_7") is not None
    assert all("editable" not in e for e in entries), "the flag is gone, not merely True"


@pytest.mark.asyncio
async def test_origin_labels_say_where_each_image_came_from():
    db = _DB(thread_rows=[_row(4, "https://files.slack.com/ours.png",
                               analysis="this thread's chart",
                               created_at="2026-07-24 09:00:00")],
             channel_rows=[_row(7, "https://files.slack.com/theirs.png",
                                analysis="a pasted screenshot",
                                created_at="2026-07-24 08:00:00")])

    entries = await image_catalog.build_catalog(db, "C0BKX77NU66:1784921906.654579",
                                                surface=SURFACE_CHANNEL)
    lines = image_catalog.catalog_lines(entries)

    assert "origin" not in entries[0], "the current root gets no qualifier at all"
    assert entries[1]["origin"] == "elsewhere in this channel"
    assert "img_4 (most recent) — uploaded: this thread's chart" in lines
    assert "img_7 [elsewhere in this channel] — uploaded: a pasted screenshot" in lines


@pytest.mark.asyncio
async def test_the_catalog_is_one_global_recency_order_not_ours_then_theirs():
    """Ruling 3. Index 0 is labelled "(most recent)", so it had better be."""
    db = _DB(thread_rows=[_row(4, "https://files.slack.com/ours.png",
                               created_at="2026-07-24 08:00:00")],
             channel_rows=[_row(7, "https://files.slack.com/theirs.png",
                                created_at="2026-07-24 09:00:00")])

    entries = await image_catalog.build_catalog(db, "C0BKX77NU66:1784921906.654579",
                                                surface=SURFACE_CHANNEL)

    assert [e["image_id"] for e in entries] == ["img_7", "img_4"], (
        "the channel's newer image outranks this root's older one")
    assert "(most recent)" in image_catalog.catalog_lines(entries).split("\n")[0]
    assert entries[0]["image_id"] == "img_7"


@pytest.mark.asyncio
async def test_images_saved_in_the_same_second_are_ordered_by_id():
    """`images.created_at` is second-precision, so the tie-break is the row id — which is why
    both DB queries carry one. Without it "the last one" is whatever order the merge produced."""
    same_second = "2026-07-24 10:00:00"
    db = _DB(thread_rows=[_row(4, "https://files.slack.com/a.png", created_at=same_second),
                          _row(9, "https://files.slack.com/b.png", created_at=same_second)],
             channel_rows=[_row(6, "https://files.slack.com/c.png", created_at=same_second)])

    entries = await image_catalog.build_catalog(db, "C0BKX77NU66:1784921906.654579",
                                                surface=SURFACE_CHANNEL)

    assert [e["image_id"] for e in entries] == ["img_9", "img_6", "img_4"]


@pytest.mark.asyncio
async def test_this_threads_own_images_come_first_and_are_not_duplicated():
    same = "https://files.slack.com/same.png"
    db = _DB(thread_rows=[_row(1, same, created_at="2026-07-24 10:00:00")],
             channel_rows=[_row(1, same, created_at="2026-07-24 10:00:00"),
                           _row(2, "https://files.slack.com/older.png",
                                created_at="2026-07-24 09:00:00")])

    entries = await image_catalog.build_catalog(db, "D1:100.0", surface=SURFACE_DM)

    assert [e["image_id"] for e in entries] == ["img_1", "img_2"]
    assert "origin" not in entries[0], "this message's own image is not 'earlier'"
    assert entries[1]["origin"] == "earlier in this DM"


@pytest.mark.asyncio
async def test_the_widening_respects_the_catalog_cap():
    """MAX_CATALOG is the only bound, and it is applied to the MERGED list.

    The lookup is no longer skipped when this root already holds MAX_CATALOG images: the cap
    comes after the merge, so an image from elsewhere that is genuinely newer has to be able to
    displace an older one. Here it is older, so it does not.
    """
    db = _DB(thread_rows=[_row(i, f"https://files.slack.com/{i}.png",
                               created_at=f"2026-07-24 10:00:{i:02d}")
                          for i in range(image_catalog.MAX_CATALOG)],
             channel_rows=[_row(99, "https://files.slack.com/extra.png",
                                created_at="2026-07-20 10:00:00")])

    entries = await image_catalog.build_catalog(db, "D1:100.0", surface=SURFACE_DM)

    assert len(entries) == image_catalog.MAX_CATALOG
    assert "img_99" not in [e["image_id"] for e in entries], "too old to make the cut"
    assert entries[0]["image_id"] == f"img_{image_catalog.MAX_CATALOG - 1}"


@pytest.mark.asyncio
async def test_a_newer_widened_image_displaces_an_older_one_at_the_cap():
    db = _DB(thread_rows=[_row(i, f"https://files.slack.com/{i}.png",
                               created_at=f"2026-07-24 10:00:{i:02d}")
                          for i in range(image_catalog.MAX_CATALOG)],
             channel_rows=[_row(99, "https://files.slack.com/newest.png",
                                created_at="2026-07-25 10:00:00")])

    entries = await image_catalog.build_catalog(db, "C1:100.0", surface=SURFACE_CHANNEL)

    assert len(entries) == image_catalog.MAX_CATALOG, "the cap still holds"
    assert entries[0]["image_id"] == "img_99", "and the newest image is genuinely first"
    assert "img_0" not in [e["image_id"] for e in entries], "the oldest fell off the end"


@pytest.mark.asyncio
async def test_a_thread_key_with_colons_on_both_sides_resolves_the_channel():
    # CLAUDE.md pitfall #3: thread keys are "channel:thread_ts" and the ts contains a dot, but a
    # naive split on ":" from the right would hand us the timestamp, not the channel.
    db = _DB(thread_rows=[], channel_rows=[_row(3, "https://files.slack.com/a.png")])
    await image_catalog.build_catalog(db, "D08EDPS3QMC:1784925818.611379", surface=SURFACE_DM)
    assert db.channel_calls[0]["channel_id"] == "D08EDPS3QMC"


@pytest.mark.asyncio
async def test_a_failed_widening_still_returns_the_strict_catalog():
    class _Broken(_DB):
        async def find_channel_images_async(self, channel_id, within_hours=None, limit=50):
            raise RuntimeError("database is locked")

    db = _Broken(thread_rows=[_row(1, "https://files.slack.com/a.png")])
    entries = await image_catalog.build_catalog(db, "D1:100.0", surface=SURFACE_DM)
    assert [e["image_id"] for e in entries] == ["img_1"]


@pytest.mark.asyncio
async def test_a_db_without_the_lookup_degrades_quietly():
    class _Old:
        async def find_thread_images_async(self, thread_id, image_type=None):
            return [_row(1, "https://files.slack.com/a.png")]

    entries = await image_catalog.build_catalog(_Old(), "D1:100.0", surface=SURFACE_DM)
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


# --- the reusable generation prompt (F34 follow-up) --------------------------------------
#
# "Generate that again with the same prompt" could not be honoured: the enhanced prompt is in
# `images.prompt`, rebuilt history redacts tool arguments, and the catalog line carried only a
# 110-char description. The prompt now rides the line in full, on ONE line.


def test_a_generated_image_carries_its_full_prompt_not_a_110_char_stub():
    prompt = "a cinematic wide shot of a lighthouse at dusk, " * 20  # ~940 chars
    lines = image_catalog.catalog_lines([
        {"image_id": "img_843", "kind": "generated", "analysis": "a lighthouse at dusk",
         "prompt": prompt},
    ])

    assert len(prompt.strip()) > 900, "the fixture must exercise the long-prompt case"
    assert prompt.strip() in lines, "the enhanced prompt is reusable only if it is verbatim"
    assert "generation prompt" in lines
    assert "a lighthouse at dusk" in lines, "the description half is unchanged"
    assert "…" not in lines, "nothing is truncated at this length"


def test_a_long_prompt_is_carried_whole_with_no_character_ceiling():
    """There is no cap. A truncated prompt is not the same prompt: "generate that again"
    would re-enhance the missing tail and hand back a different picture."""
    prompt = "an isometric cutaway of a lighthouse, every deck labelled, " * 51  # ~3000 chars
    lines = image_catalog.catalog_lines([
        {"image_id": "img_844", "kind": "generated", "analysis": "a lighthouse cutaway",
         "prompt": prompt},
    ])

    assert len(prompt.strip()) > 3000, "the fixture must exceed the old 2000-char ceiling"
    assert prompt.strip() in lines, "the whole prompt rides the line, verbatim"
    assert "…" not in lines, "nothing is truncated at any length"


def test_an_uploaded_image_renders_exactly_as_before():
    """No prompt of ours to reuse — the line must not grow a 'generation prompt' clause."""
    lines = image_catalog.catalog_lines([
        {"image_id": "img_9", "kind": "uploaded", "analysis": "a screenshot of a dashboard",
         "prompt": ""},
    ])

    assert lines == "img_9 (most recent) — uploaded: a screenshot of a dashboard"
    assert "generation prompt" not in lines


def test_a_multiline_prompt_is_collapsed_so_evidence_stays_one_line_per_image():
    # catalog_evidence_lines splits on newlines: an embedded newline would scatter one image
    # across several entries and break the id-per-line contract.
    entries = [
        {"image_id": "img_5", "kind": "edited", "analysis": "a chart",
         "prompt": "make the bars blue\n\nand widen the axis labels"},
        {"image_id": "img_4", "kind": "uploaded", "analysis": "the original chart", "prompt": ""},
    ]

    lines = image_catalog.catalog_evidence_lines(entries)

    assert lines[0] == image_catalog.EVIDENCE_HEADER
    assert len(lines) == 4, "a header, exactly one line per image, and the index note"
    assert "make the bars blue and widen the axis labels" in lines[1]
    assert lines[-1] == image_catalog.INDEX_NOTE
