"""`view_image` — re-attach an EARLIER thread image as real pixels, without the sandbox.

The gap it closes (observed live): only the answered message's attachments become `input_image`
parts; every earlier image reaches the model as TEXT. Asked whether a screenshot posted two
messages earlier was genuine, the model had no pixels, went hunting in the code-interpreter
sandbox (thread attachments auto-mount there and the container persists), rendered a matplotlib
contact sheet to pull them into its own vision — and that debug figure auto-published.

Covered here: the schema (always offered, listing the turn's catalog without being bounded by
it), the executor (the turn's catalog then a channel-scoped lookup, the channel boundary,
download, transcode, dedupe, per-turn cap, honest failures), and the tool-loop drain (a USER-role
message, `_`-prefixed bookkeeping stripped, placed after the call/output pairs, replayed exactly
once).

Real decision code, stubbed I/O — no network, no DB, no container.
"""
from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


from message_processor.image_view import (
    CATALOG_KEY,
    execute_view_image,
    get_view_image_schema,
)

# A 1x1 PNG — real bytes, so ensure_api_compatible genuinely validates rather than being mocked.
_PNG = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _catalog():
    """Exactly the shape image_catalog.build_catalog emits — note there is NO file_id, which is
    why the executor lets download_file extract it from the URL."""
    return [
        {"image_id": "img_9", "url": "https://files.slack.com/a.png", "kind": "uploaded",
         "prompt": "", "analysis": "a model pricing table", "created_at": "2026-07-24 17:41:05"},
        {"image_id": "img_8", "url": "https://files.slack.com/b.png", "kind": "uploaded",
         "prompt": "", "analysis": "a terminal model picker", "created_at": "2026-07-24 17:41:43"},
        {"image_id": "img_7", "url": "https://files.slack.com/c.png", "kind": "generated",
         "prompt": "", "analysis": "", "created_at": "2026-07-24 17:38:47"},
    ]


def _img(data=_PNG):
    """What ImageURLHandler.download_image returns (it already transcodes + size-caps)."""
    if data is None:
        return None
    return {"url": "u", "mimetype": "image/png", "size": len(data),
            "base64_data": base64.b64encode(data).decode(), "data": data}


def _ctx(*, catalog=None, download=_PNG, current_urls=None, handler=None):
    h = handler or SimpleNamespace(download_image=AsyncMock(return_value=_img(download)))
    return SimpleNamespace(
        image_catalog=_catalog() if catalog is None else catalog,
        pending_vision_parts=[],
        current_image_urls=list(current_urls or []),
        processor=SimpleNamespace(image_url_handler=h),
    )


# ------------------------------------------------------------------ schema

def test_schema_is_offered_even_with_no_catalog():
    """[OWNER 2026-09-16] The turn's catalog is a shortlist, not the boundary. It used to return
    None here — and a DM whose images are all older than the catalog's reach could then be
    SEARCHED for a screenshot and handed an id with no tool to open it with."""
    for cfg in (None, {}, {CATALOG_KEY: []}):
        assert get_view_image_schema(cfg)["name"] == "view_image"


def test_schema_lists_the_catalog_but_does_not_enum_it():
    schema = get_view_image_schema({CATALOG_KEY: _catalog()})
    assert schema["name"] == "view_image"
    # The descriptions ride along so the model can pick the right one…
    assert "a model pricing table" in schema["description"]
    assert "img_9" in schema["description"]
    # …but they do not BOUND it: an id from a search_stored_knowledge hit has to be emittable,
    # and the executor is what authorizes any of them.
    assert "enum" not in schema["parameters"]["properties"]["image_id"]
    assert "search_stored_knowledge" in schema["description"]


def test_every_schema_that_shows_the_list_says_it_is_only_the_recent_ones():
    """[OWNER 2026-09-16] A list presented as complete is how a model decides an unlisted image
    does not exist — and then web-searches for a screenshot sitting in the channel. MAX_CATALOG
    may bound the advertised list only on the condition that the list says so, everywhere it is
    shown."""
    from message_processor import image_catalog, image_tools
    from message_processor.image_view import get_view_image_schema_static

    cfg = {CATALOG_KEY: _catalog()}
    showing_the_list = (
        get_view_image_schema(cfg),
        get_view_image_schema_static(cfg),
        image_tools.get_edit_image_schema(cfg),
        image_tools.get_edit_image_asset_schema(cfg),
        image_tools.get_edit_image_schema_static(cfg),
    )
    for schema in showing_the_list:
        assert image_catalog.INDEX_NOTE in schema["description"], schema["name"]
    # And the evidence block, which is where the channel surface's ids actually live.
    assert image_catalog.INDEX_NOTE in image_catalog.catalog_evidence_lines(_catalog())
    assert image_catalog.INDEX_NOTE in image_catalog.catalog_evidence_lines([]), (
        "an empty shortlist is exactly when 'search for older ones' matters most")


def test_schema_routes_other_intents_elsewhere():
    """Looking is all this tool does: changing a picture is edit_image, computing on one is
    mount_file, and neither is 'render it in the sandbox so I can see it'."""
    desc = get_view_image_schema({CATALOG_KEY: _catalog()})["description"]
    assert "edit_image" in desc and "mount_file" in desc
    assert "Never render an image in the sandbox merely to see it." in desc
    # It must steer OFF the current message's images — those are already visible.
    assert "do NOT call this for those" in desc


# ------------------------------------------------------------------ executor: scope guard

async def test_unknown_id_is_refused():
    """The catalog is thread-scoped, so an id from another thread simply isn't in it. An
    invented id lands in the same place: refused, never a guess at 'the recent one'."""
    res = await execute_view_image(_ctx(), {"image_id": "img_999"})
    assert res["ok"] is False and res["error"] == "unknown_image"


async def test_missing_id_is_refused():
    res = await execute_view_image(_ctx(), {"image_id": "  "})
    assert res["ok"] is False and res["error"] == "missing_image_id"


# ------------------------------------------------- executor: reach past the turn's catalog
#
# [OWNER 2026-09-16] MAX_CATALOG bounds the ADVERTISED list, never what is reachable.
# search_stored_knowledge indexes every image description in the channel with no time bound, so
# the id it hands back has to open regardless of the picture's age or which thread it was in —
# otherwise it is a handle that always breaks, which is why it used to be withheld entirely.


class _ChannelDB:
    """Records how it was asked, so the privacy boundary can be asserted rather than assumed."""

    def __init__(self, rows=None):
        self.rows = dict(rows or {})
        self.calls = []

    async def get_channel_image_by_id_async(self, channel_id, image_id):
        self.calls.append((channel_id, image_id))
        return self.rows.get((channel_id, image_id))


def _channel_ctx(db, *, channel_id="C0BKX77NU66", catalog=None, download=_PNG):
    h = SimpleNamespace(download_image=AsyncMock(return_value=_img(download)))
    return SimpleNamespace(
        image_catalog=_catalog() if catalog is None else catalog,
        pending_vision_parts=[],
        current_image_urls=[],
        channel_id=channel_id,
        processor=SimpleNamespace(image_url_handler=h, db=db),
    )


async def test_an_id_outside_the_turns_catalog_resolves_through_the_channel():
    """The live case: a three-week-old screenshot found by search, in another thread entirely."""
    row = {"id": 500, "url": "https://files.slack.com/old-screenshot.png",
           "image_type": "uploaded", "prompt": "",
           "analysis": "a devtools panel showing a 500 on checkout",
           "created_at": "2026-07-01 09:00:00"}
    db = _ChannelDB({("C0BKX77NU66", 500): row})
    ctx = _channel_ctx(db)

    assert "img_500" not in [e["image_id"] for e in ctx.image_catalog], "not in the shortlist"

    res = await execute_view_image(ctx, {"image_id": "img_500"})

    assert res["ok"] is True
    assert db.calls == [("C0BKX77NU66", 500)], "looked up by row id, inside this channel"
    ctx.processor.image_url_handler.download_image.assert_awaited_once()
    assert ctx.processor.image_url_handler.download_image.await_args[0][0] == row["url"]
    staged = ctx.pending_vision_parts[0]
    assert staged["_image_id"] == "img_500" and staged["_ready"] is True


async def test_the_fallback_works_with_no_advertised_catalog_at_all():
    row = {"id": 500, "url": "https://files.slack.com/old.png", "image_type": "uploaded",
           "prompt": "", "analysis": "an old chart", "created_at": "2026-07-01 09:00:00"}
    ctx = _channel_ctx(_ChannelDB({("C0BKX77NU66", 500): row}), catalog=[])

    res = await execute_view_image(ctx, {"image_id": "img_500"})

    assert res["ok"] is True, "an empty shortlist is not an empty channel"


async def test_an_id_in_another_channel_stays_unresolvable():
    """The boundary is unchanged: the DB lookup is scoped to THIS channel, so a row that exists
    but lives elsewhere is the same answer as one that never existed."""
    row = {"id": 500, "url": "https://files.slack.com/theirs.png", "image_type": "uploaded",
           "prompt": "", "analysis": "someone else's channel", "created_at": "2026-07-01 09:00:00"}
    db = _ChannelDB({("C0ELSEWHERE", 500): row})
    ctx = _channel_ctx(db)

    res = await execute_view_image(ctx, {"image_id": "img_500"})

    assert res["ok"] is False and res["error"] == "unknown_image"
    assert db.calls == [("C0BKX77NU66", 500)], "asked about THIS channel and got nothing"
    ctx.processor.image_url_handler.download_image.assert_not_called()


@pytest.mark.parametrize("bad_id", ["img_3x", "IMG_500", "../img_500", "img_", "img_-1"])
async def test_a_malformed_id_never_reaches_the_database(bad_id):
    # The handle is parsed strictly before it becomes a lookup: anything that is not
    # `img_<digits>` is refused without a query.
    db = _ChannelDB()
    ctx = _channel_ctx(db)

    res = await execute_view_image(ctx, {"image_id": bad_id})

    assert res["ok"] is False and res["error"] == "unknown_image"
    assert db.calls == []


async def test_a_failed_channel_lookup_is_an_unresolved_id_not_a_failed_turn():
    class _Broken(_ChannelDB):
        async def get_channel_image_by_id_async(self, channel_id, image_id):
            raise RuntimeError("database is locked")

    res = await execute_view_image(_channel_ctx(_Broken()), {"image_id": "img_500"})
    assert res["ok"] is False and res["error"] == "unknown_image"


async def test_does_not_download_for_unknown_id():
    """Scope guard runs BEFORE any fetch — an unresolvable id must not hit the network."""
    ctx = _ctx()
    await execute_view_image(ctx, {"image_id": "img_999"})
    ctx.processor.image_url_handler.download_image.assert_not_called()


# ------------------------------------------------------------------ executor: happy path

async def test_stages_a_vision_part():
    ctx = _ctx()
    res = await execute_view_image(ctx, {"image_id": "img_9"})
    assert res["ok"] is True
    assert len(ctx.pending_vision_parts) == 1
    res = ctx.pending_vision_parts[0]
    assert res["_image_id"] == "img_9" and res["_ready"] is True
    label, image = res["parts"]
    # Each image is LABELLED with its id immediately before its pixels: sibling calls complete
    # out of order, so unlabelled images can be attributed to the wrong id on a comparison.
    assert label["type"] == "input_text" and "img_9" in label["text"]
    assert image["type"] == "input_image"
    assert image["image_url"].startswith("data:image/png;base64,")


async def test_downloads_the_id_the_model_named_not_the_newest():
    """The whole point of opaque ids: img_8 fetches img_8's url, never a 'most recent' guess."""
    ctx = _ctx()
    await execute_view_image(ctx, {"image_id": "img_8"})
    args = ctx.processor.image_url_handler.download_image.await_args
    assert args.args[0] == "https://files.slack.com/b.png"


# ------------------------------------------------------------------ executor: cost guards

async def test_second_call_for_same_image_does_not_refetch():
    """A model that asks twice already has it. Re-staging would repeat full-resolution base64 in
    every remaining round of the turn."""
    ctx = _ctx()
    await execute_view_image(ctx, {"image_id": "img_9"})
    res = await execute_view_image(ctx, {"image_id": "img_9"})
    assert res["ok"] is True and res["already_visible"] is True
    assert len(ctx.pending_vision_parts) == 1
    assert ctx.processor.image_url_handler.download_image.await_count == 1


async def test_there_is_no_per_turn_ceiling():
    """[OWNER 2026-09-16] A two-image cap used to refuse the third call. It was a token lever
    plus a guess that a model wanting several screenshots is rummaging — and "compare these three
    charts" is work, not rummaging. The dedupe above is what survives."""
    ctx = _ctx()
    for img in ("img_9", "img_8", "img_7"):
        assert (await execute_view_image(ctx, {"image_id": img}))["ok"] is True
    assert [r["_image_id"] for r in ctx.pending_vision_parts] == ["img_9", "img_8", "img_7"]
    assert ctx.processor.image_url_handler.download_image.await_count == 3


# ------------------------------------------------------------------ executor: honest failures

async def test_deleted_slack_file_reports_honestly():
    """A deleted file is indistinguishable from one that never existed — say so rather than
    letting the model narrate an image it never saw."""
    ctx = _ctx(download=None)
    res = await execute_view_image(ctx, {"image_id": "img_9"})
    assert res["ok"] is False and res["error"] == "unavailable_source"
    assert not ctx.pending_vision_parts


async def test_undecodable_bytes_are_refused_not_attached():
    """ImageURLHandler returns None when ensure_api_compatible rejects the bytes."""
    ctx = _ctx(download=None)
    res = await execute_view_image(ctx, {"image_id": "img_9"})
    assert res["ok"] is False and res["error"] == "unavailable_source"
    assert not ctx.pending_vision_parts


async def test_download_exception_never_raises_into_the_loop():
    ctx = _ctx(handler=SimpleNamespace(
        download_image=AsyncMock(side_effect=RuntimeError("slack exploded"))))
    res = await execute_view_image(ctx, {"image_id": "img_9"})
    assert res["ok"] is False and res["error"] == "unavailable_source"


async def test_missing_handler_degrades_gracefully():
    ctx = _ctx()
    ctx.processor = None
    res = await execute_view_image(ctx, {"image_id": "img_9"})
    assert res["ok"] is False and res["error"] == "unavailable"


# ------------------------------------------------------------------ the tool-loop drain

def _drain(tool_context, input_items):
    """Mirror of the drain in tool_loop._run_tool_round (kept in lockstep by the assertions
    below, which pin the shape the API actually requires)."""
    from openai_client.api import tool_loop  # noqa: F401  (import proves the module loads)
    staged = getattr(tool_context, "pending_vision_parts", None) or []
    fresh = [r for r in staged if r.get("_ready") and not r.get("_replayed")]
    if fresh:
        content = []
        for reservation in fresh:
            reservation["_replayed"] = True
            content.extend(reservation.get("parts") or [])
        if content:
            input_items.append({"role": "user", "content": content})


async def test_drained_as_user_role_with_bookkeeping_stripped():
    """USER role (untrusted user-supplied bytes, matching the boundary the stored descriptions
    already respect), and no `_`-prefixed keys — the API 400s on unknown keys in a content part.
    """
    ctx = _ctx()
    await execute_view_image(ctx, {"image_id": "img_9"})
    items = []
    _drain(ctx, items)
    assert len(items) == 1
    assert items[0]["role"] == "user"
    parts = items[0]["content"]
    assert [p["type"] for p in parts] == ["input_text", "input_image"]
    for p in parts:
        assert not [k for k in p if k.startswith("_")]


async def test_drain_is_idempotent_across_rounds():
    """input_items persists across rounds, so the already-appended message keeps the image
    visible. Re-draining it would duplicate the payload every round."""
    ctx = _ctx()
    await execute_view_image(ctx, {"image_id": "img_9"})
    items = []
    _drain(ctx, items)
    _drain(ctx, items)
    assert len(items) == 1


async def test_nothing_appended_when_nothing_staged():
    items = []
    _drain(_ctx(), items)
    assert items == []


# ------------------------------------------------ codex review: current-turn images excluded

async def test_image_already_on_this_turn_is_not_reattached():
    """The catalog is built AFTER the answered message's attachments are persisted, so the image
    the user just posted IS in it. Re-attaching it would bill the same pixels twice per round to
    show the model something already in front of it."""
    ctx = _ctx(current_urls=["https://files.slack.com/a.png"])
    res = await execute_view_image(ctx, {"image_id": "img_9"})
    assert res["ok"] is True and res["already_visible"] is True
    assert not ctx.pending_vision_parts
    ctx.processor.image_url_handler.download_image.assert_not_called()


async def test_other_images_still_viewable_when_one_is_current():
    ctx = _ctx(current_urls=["https://files.slack.com/a.png"])
    res = await execute_view_image(ctx, {"image_id": "img_8"})
    assert res["ok"] is True and not res.get("already_visible")
    assert len(ctx.pending_vision_parts) == 1


# ------------------------------------------------ codex review: SSRF / token-leak safety

async def test_fetches_through_the_guarded_handler_not_the_slack_downloader():
    """The catalog also holds images harvested from EXTERNAL urls (F18 persists them so
    edit_image can name them), and Slack's downloader falls back to a direct GET carrying the bot
    token when it can't parse a Slack file id — handing that token to any host someone can get
    linked into a channel. ImageURLHandler authenticates verified Slack hosts only and fetches
    everything else under SSRF guards, so the fetch MUST go through it."""
    ctx = _ctx(catalog=[{"image_id": "img_5", "url": "https://evil.example/x.png",
                         "kind": "uploaded", "prompt": "", "analysis": "external",
                         "created_at": "2026-07-24 10:00:00"}])
    # A client that would leak the token is present but must never be reached.
    ctx.client = SimpleNamespace(download_file=AsyncMock(return_value=_PNG))
    await execute_view_image(ctx, {"image_id": "img_5"})
    ctx.client.download_file.assert_not_called()
    ctx.processor.image_url_handler.download_image.assert_awaited_once()


# ------------------------------------------------ codex review: parallel dispatch race

def _slow_handler():
    """A downloader every concurrent call parks inside, so siblings genuinely interleave."""
    import asyncio

    async def _slow(url, auth_token=None):
        await asyncio.sleep(0.01)
        return _img()

    return SimpleNamespace(download_image=AsyncMock(side_effect=_slow))


async def test_concurrent_siblings_on_one_image_stage_and_download_it_once():
    """A round's calls run under asyncio.gather (tool_registry.dispatch_all), so siblings
    interleave at every await. Reserving the slot BEFORE the fetch is what stops three
    simultaneous calls for one picture from all seeing an empty list and all appending — which
    would download it three times and repeat its pixels three times in every later round.

    This is the half of that block that is NOT a cap, and it has to keep working now the cap is
    gone."""
    import asyncio

    ctx = _ctx(handler=_slow_handler())
    results = await asyncio.gather(*[
        execute_view_image(ctx, {"image_id": "img_9"}) for _ in range(3)
    ])
    assert all(r["ok"] for r in results)
    assert len(ctx.pending_vision_parts) == 1
    assert ctx.processor.image_url_handler.download_image.await_count == 1


async def test_concurrent_siblings_on_different_images_all_succeed():
    import asyncio

    ctx = _ctx(handler=_slow_handler())
    results = await asyncio.gather(*[
        execute_view_image(ctx, {"image_id": i}) for i in ("img_9", "img_8", "img_7")
    ])
    assert all(r["ok"] for r in results), "no sibling is refused for the others' sake"
    assert len(ctx.pending_vision_parts) == 3


async def test_failed_fetch_frees_its_slot_for_a_retry():
    """A reservation that never fills must be removed, or one dead fetch permanently burns a
    slot the model could have spent on an image that IS retrievable."""
    calls = {"n": 0}

    async def _first_fails(url, auth_token=None):
        calls["n"] += 1
        return None if calls["n"] == 1 else _img()

    ctx = _ctx(handler=SimpleNamespace(download_image=AsyncMock(side_effect=_first_fails)))
    bad = await execute_view_image(ctx, {"image_id": "img_9"})
    assert bad["ok"] is False
    assert ctx.pending_vision_parts == []          # slot released
    good = await execute_view_image(ctx, {"image_id": "img_8"})
    assert good["ok"] is True and len(ctx.pending_vision_parts) == 1


async def test_unready_reservation_is_never_drained():
    """A half-built reservation must not reach the API as an empty/partial content block."""
    ctx = _ctx()
    ctx.pending_vision_parts.append({"_image_id": "img_9", "_ready": False, "parts": []})
    items = []
    _drain(ctx, items)
    assert items == []


# ------------------------------------------------ produced images are shown back to the model

def _produced(fmt="png"):
    return SimpleNamespace(base64_data=base64.b64encode(_PNG).decode(), format=fmt, prompt="p")


def test_produced_image_is_staged_for_the_model():
    """The model used to get only the string "the edited image has been posted" — it never saw
    its own output, so it could not confirm the edit landed or act on "make the text bigger"."""
    from message_processor.image_view import stage_produced_image
    ctx = _ctx()
    assert stage_produced_image(ctx, _produced(), label="Your edited image") is True
    res = ctx.pending_vision_parts[0]
    assert res["_ready"] is True and res["_image_id"] == "produced:1"
    label, image = res["parts"]
    assert "Your edited image" in label["text"]
    assert image["image_url"].startswith("data:image/png;base64,")
    # Follows the configured detail rather than a literal: on the 5.6 family `auto` is FULL
    # resolution (equivalent to `original`), so pinning "high" here would assert a downgrade.
    from config import config as _cfg
    assert image["detail"] == _cfg.default_detail_level


def test_produced_jpeg_gets_the_right_mimetype():
    from message_processor.image_view import stage_produced_image
    ctx = _ctx()
    stage_produced_image(ctx, _produced(fmt="jpg"), label="x")
    assert ctx.pending_vision_parts[0]["parts"][1]["image_url"].startswith("data:image/jpeg;base64,")


def test_viewed_images_do_not_spend_the_produced_ceiling():
    """The produced ceiling counts `produced:` entries only, so images the turn looked BACK at
    share the staging list without eating into it."""
    from message_processor.image_view import stage_produced_image
    ctx = _ctx()
    for i in range(5):
        ctx.pending_vision_parts.append({"_image_id": f"img_{i}", "_ready": True, "parts": []})
    assert stage_produced_image(ctx, _produced(), label="y") is True


def test_every_produced_image_is_staged_however_many_there_are():
    """[OWNER 2026-09-16] There was a ceiling of two, and it failed SILENTLY — the third picture
    the model had just made was never shown to it and nothing said why, so "generate five variants
    and pick the best" could not work. The count shown is the count made."""
    from message_processor.image_view import stage_produced_image
    ctx = _ctx()
    for _ in range(5):
        assert stage_produced_image(ctx, _produced(), label="y") is True
    produced = [r for r in ctx.pending_vision_parts
                if str(r["_image_id"]).startswith("produced:")]
    assert len(produced) == 5
    # The ids stay distinct, so five pictures are five labelled things and not one overwritten.
    assert [r["_image_id"] for r in produced] == [f"produced:{i}" for i in range(1, 6)]


def test_staging_never_raises_on_a_junk_image_object():
    """Showing the model its own output is an enrichment; the image is already posted, so a
    failure here must never take down the turn."""
    from message_processor.image_view import stage_produced_image
    assert stage_produced_image(_ctx(), object(), label="y") is False
    assert stage_produced_image(_ctx(), None, label="y") is False


def test_produced_image_drains_into_the_user_message():
    from message_processor.image_view import stage_produced_image
    ctx = _ctx()
    stage_produced_image(ctx, _produced(), label="Your edited image")
    items = []
    _drain(ctx, items)
    assert items[0]["role"] == "user"
    assert [p["type"] for p in items[0]["content"]] == ["input_text", "input_image"]


# ------------------------------------------------------------------ static channel schema

def test_static_schema_is_byte_stable_across_catalogs_and_requesters():
    import json
    from message_processor.image_view import get_view_image_schema_static

    a = json.dumps(get_view_image_schema_static(
        {CATALOG_KEY: _catalog(), "user_id": "U_A", "model": "gpt-5.6-sol"}), sort_keys=True)
    b = json.dumps(get_view_image_schema_static(
        {CATALOG_KEY: [], "user_id": "U_B", "model": "gpt-5.5"}), sort_keys=True)
    c = json.dumps(get_view_image_schema_static(), sort_keys=True)
    assert a == b == c


def test_static_schema_carries_no_ids_and_no_enum():
    import json
    from message_processor.image_view import get_view_image_schema_static

    schema = get_view_image_schema_static({CATALOG_KEY: _catalog()})
    assert "enum" not in schema["parameters"]["properties"]["image_id"]
    blob = json.dumps(schema)
    assert "img_9" not in blob and "a model pricing table" not in blob
    # It still has to steer the model the same way the factory did.
    assert "do NOT call this for those" in schema["description"]
    assert "Never render an image in the sandbox merely to see it." in schema["description"]
    assert "evidence" in schema["description"]


def test_static_schema_is_never_hidden():
    """The factory returned None with no catalog. The static variant cannot — the executor's
    honest refusal is what covers an empty catalog on the channel surface."""
    from message_processor.image_view import get_view_image_schema_static
    for cfg in (None, {}, {CATALOG_KEY: []}):
        assert get_view_image_schema_static(cfg)["name"] == "view_image"


def test_both_factories_now_agree_that_the_catalog_is_not_the_boundary():
    assert get_view_image_schema({CATALOG_KEY: []})["name"] == "view_image"
    assert "enum" not in get_view_image_schema(
        {CATALOG_KEY: _catalog()})["parameters"]["properties"]["image_id"]


# ------------------------------------------------------------------ executor: honest empty

async def test_an_empty_catalog_is_answered_honestly():
    """With no enum to stop it, the model can name an id on a thread that has no images at all.
    The refusal has to say THAT, not imply the id was merely wrong."""
    res = await execute_view_image(_ctx(catalog=[]), {"image_id": "img_9"})

    assert res["ok"] is False and res["error"] == "unknown_image"
    assert res["valid_image_ids"] == []
    assert "no images" in res["message"]


async def test_an_unknown_id_lists_the_ids_that_would_have_worked():
    res = await execute_view_image(_ctx(), {"image_id": "img_999"})

    assert res["error"] == "unknown_image"
    assert res["valid_image_ids"] == ["img_9", "img_8", "img_7"]
