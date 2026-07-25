"""`view_image` — re-attach an EARLIER thread image as real pixels, without the sandbox.

The gap it closes (observed live): only the answered message's attachments become `input_image`
parts; every earlier image reaches the model as TEXT. Asked whether a screenshot posted two
messages earlier was genuine, the model had no pixels, went hunting in the code-interpreter
sandbox (thread attachments auto-mount there and the container persists), rendered a matplotlib
contact sheet to pull them into its own vision — and that debug figure auto-published.

Covered here: the schema (offered only with a catalog; ids as a literal enum), the executor
(scope guard, download, transcode, dedupe, per-turn cap, honest failures), and the tool-loop
drain (a USER-role message, `_`-prefixed bookkeeping stripped, placed after the call/output
pairs, replayed exactly once).

Real decision code, stubbed I/O — no network, no DB, no container.
"""
from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock


from message_processor.image_view import (
    CATALOG_KEY,
    MAX_VIEWS_PER_TURN,
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

def test_schema_absent_without_catalog():
    """No earlier images → no tool. Offering it would guarantee a failed call."""
    assert get_view_image_schema({}) is None
    assert get_view_image_schema({CATALOG_KEY: []}) is None


def test_schema_pins_ids_as_enum_and_lists_them():
    schema = get_view_image_schema({CATALOG_KEY: _catalog()})
    assert schema["name"] == "view_image"
    assert schema["parameters"]["properties"]["image_id"]["enum"] == ["img_9", "img_8", "img_7"]
    # The descriptions ride along so the model can pick the right one.
    assert "a model pricing table" in schema["description"]


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


async def test_per_turn_cap_is_enforced():
    ctx = _ctx()
    for img in ("img_9", "img_8"):
        assert (await execute_view_image(ctx, {"image_id": img}))["ok"] is True
    res = await execute_view_image(ctx, {"image_id": "img_7"})
    assert res["ok"] is False and res["error"] == "limit_reached"
    assert len(ctx.pending_vision_parts) == MAX_VIEWS_PER_TURN


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

async def test_concurrent_siblings_cannot_exceed_the_cap():
    """A round's calls run under asyncio.gather (tool_registry.dispatch_all), so siblings
    interleave at every await. Reserving the slot BEFORE the fetch is what keeps three
    simultaneous calls from all passing the cap check and all appending."""
    import asyncio

    started = asyncio.Event()

    async def _slow(url, auth_token=None):
        started.set()
        await asyncio.sleep(0.01)      # every call parks here at the same time
        return _img()

    ctx = _ctx(handler=SimpleNamespace(download_image=AsyncMock(side_effect=_slow)))
    results = await asyncio.gather(*[
        execute_view_image(ctx, {"image_id": i}) for i in ("img_9", "img_8", "img_7")
    ])
    assert sum(1 for r in results if r["ok"]) == MAX_VIEWS_PER_TURN
    assert sum(1 for r in results if r.get("error") == "limit_reached") == 1
    assert len(ctx.pending_vision_parts) == MAX_VIEWS_PER_TURN


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


def test_produced_images_do_not_spend_the_view_budget():
    """Looking BACK at thread images is rationed; seeing what you just MADE is not the same
    thing and must not be starved by it."""
    from message_processor.image_view import stage_produced_image
    ctx = _ctx()
    for _ in range(MAX_VIEWS_PER_TURN):
        ctx.pending_vision_parts.append({"_image_id": "img_x", "_ready": True, "parts": []})
    assert stage_produced_image(ctx, _produced(), label="y") is True


def test_produced_images_have_their_own_ceiling():
    from message_processor.image_view import MAX_PRODUCED_PER_TURN, stage_produced_image
    ctx = _ctx()
    for _ in range(MAX_PRODUCED_PER_TURN):
        assert stage_produced_image(ctx, _produced(), label="y") is True
    assert stage_produced_image(ctx, _produced(), label="y") is False


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
