"""T6 — bookmarks: list, add, remove.

Two things in this module are worth testing and the rest is plumbing:

* `bookmarks.add` needs FOUR fields (`channel_id`, `title`, `type="link"`, `link`), and the one
  that gets forgotten is `type` — so the add tests assert the whole call, not just the URL.
* `remove_bookmark` proves its id against a FRESH `bookmarks.list` rather than trusting anything
  the model saw earlier, because a same-round listing cannot reach it and an older one is a
  snapshot. So the remove tests care about the ORDER of the calls and about what happens when
  the listing fails: nothing is removed.

Everything runs the real executors against a mocked Slack transport.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from slack_sdk.errors import SlackApiError

from message_processor import bookmark_tools as bt
from tool_registry import ToolContext, ToolRegistry

CH = "C1000000000"
USER = "U_REQUESTER"
BM = "Bk01234567"
URL = "https://example.com/runbook"


def _listing(*entries):
    return {"ok": True, "bookmarks": list(entries)}


def _bookmark(bid=BM, title="Deploy runbook", link=URL):
    return {"id": bid, "title": title, "link": link, "type": "link",
            "channel_id": CH, "created": 1700000000}


def _ctx(channel=CH, *, listing=None, add=None, remove=None):
    """A ToolContext whose `client.app.client` is the mocked Slack web client."""
    web = SimpleNamespace(
        bookmarks_list=AsyncMock(return_value=listing if listing is not None
                                 else _listing(_bookmark())),
        bookmarks_add=AsyncMock(return_value=add if add is not None
                                else {"ok": True, "bookmark": _bookmark()}),
        bookmarks_remove=AsyncMock(return_value=remove if remove is not None else {"ok": True}),
    )
    client = SimpleNamespace(app=SimpleNamespace(client=web))
    return ToolContext(channel_id=channel, thread_ts="1700000000.000100",
                       user_id=USER, client=client)


def _web(ctx):
    return ctx.client.app.client


def _api_error(error="channel_not_found"):
    return SlackApiError("boom", {"ok": False, "error": error})


# ================================================================================ schemas

def test_schemas_are_well_formed():
    for schema in (bt.get_list_bookmarks_schema(), bt.get_add_bookmark_schema(),
                   bt.get_remove_bookmark_schema()):
        assert schema["type"] == "function"
        assert schema["name"]
        assert schema["description"]
        assert schema["parameters"]["type"] == "object"


def test_list_takes_no_arguments():
    params = bt.get_list_bookmarks_schema()["parameters"]
    assert params["properties"] == {}
    assert params["required"] == []


def test_write_schemas_say_request_only():
    """The whole policy for add/remove is social — so it has to actually be in the text."""
    for schema in (bt.get_add_bookmark_schema(), bt.get_remove_bookmark_schema()):
        text = schema["description"].lower()
        assert "asks you to" in text
        assert "own initiative" in text


def test_no_channel_argument_anywhere():
    """The conversation comes from the ToolContext; a channel argument would be a way out of it."""
    for schema in (bt.get_list_bookmarks_schema(), bt.get_add_bookmark_schema(),
                   bt.get_remove_bookmark_schema()):
        assert "channel_id" not in schema["parameters"]["properties"]
        assert "channel" not in schema["parameters"]["properties"]


def test_registration_registers_all_three():
    registry = ToolRegistry()
    bt.register_bookmark_tools(registry)
    names = {s["name"] for s in registry.schemas({})}
    assert names == {"list_bookmarks", "add_bookmark", "remove_bookmark"}


# ================================================================================ list

@pytest.mark.asyncio
async def test_list_returns_entries_scoped_to_this_channel():
    ctx = _ctx(listing=_listing(_bookmark(), _bookmark("Bk9", "Dashboard", "https://example.com/d")))
    out = await bt.execute_list_bookmarks(ctx, {})
    assert out["ok"] is True
    assert out["count"] == 2
    assert out["bookmarks"][0] == {"bookmark_id": BM, "title": "Deploy runbook",
                                  "link": URL, "type": "link"}
    _web(ctx).bookmarks_list.assert_awaited_once_with(channel_id=CH)


@pytest.mark.asyncio
async def test_list_skips_entries_with_no_id():
    """One junk entry must not turn a readable answer into a failure."""
    ctx = _ctx(listing={"ok": True, "bookmarks": [{"title": "nameless"}, _bookmark()]})
    out = await bt.execute_list_bookmarks(ctx, {})
    assert out["ok"] is True
    assert [b["bookmark_id"] for b in out["bookmarks"]] == [BM]


@pytest.mark.asyncio
async def test_list_empty_channel():
    out = await bt.execute_list_bookmarks(_ctx(listing=_listing()), {})
    assert out == {"ok": True, "bookmarks": [], "count": 0}


@pytest.mark.asyncio
async def test_list_surfaces_slacks_own_error_name():
    ctx = _ctx()
    _web(ctx).bookmarks_list.side_effect = _api_error("not_in_channel")
    out = await bt.execute_list_bookmarks(ctx, {})
    assert out["ok"] is False and out["error"] == "list_failed"
    assert "not_in_channel" in out["message"]


@pytest.mark.asyncio
async def test_list_without_a_channel_is_refused():
    out = await bt.execute_list_bookmarks(_ctx(channel=None), {})
    assert out["ok"] is False and out["error"] == "unavailable"


# ================================================================================ add

@pytest.mark.asyncio
async def test_add_sends_all_four_required_fields():
    """`type="link"` has no default: omit it and Slack refuses the whole call."""
    ctx = _ctx()
    out = await bt.execute_add_bookmark(ctx, {"title": "Deploy runbook", "url": URL})
    assert out["ok"] is True
    assert out["bookmark_id"] == BM
    _web(ctx).bookmarks_add.assert_awaited_once_with(
        channel_id=CH, title="Deploy runbook", type="link", link=URL)


@pytest.mark.asyncio
async def test_add_trims_its_arguments():
    ctx = _ctx()
    await bt.execute_add_bookmark(ctx, {"title": "  Deploy runbook \n", "url": f" {URL} "})
    assert _web(ctx).bookmarks_add.await_args.kwargs["title"] == "Deploy runbook"
    assert _web(ctx).bookmarks_add.await_args.kwargs["link"] == URL


@pytest.mark.parametrize("args,error", [
    ({"url": URL}, "missing_title"),
    ({"title": "   ", "url": URL}, "missing_title"),
    ({"title": 7, "url": URL}, "missing_title"),
    ({"title": "Runbook"}, "missing_url"),
    ({"title": "Runbook", "url": ""}, "missing_url"),
    ({"title": "Runbook", "url": ["nope"]}, "missing_url"),
])
@pytest.mark.asyncio
async def test_add_refuses_bad_arguments_without_calling_slack(args, error):
    ctx = _ctx()
    out = await bt.execute_add_bookmark(ctx, args)
    assert out["ok"] is False and out["error"] == error
    assert _web(ctx).bookmarks_add.await_count == 0


@pytest.mark.asyncio
async def test_add_surfaces_slacks_own_error_name():
    ctx = _ctx()
    _web(ctx).bookmarks_add.side_effect = _api_error("invalid_arguments")
    out = await bt.execute_add_bookmark(ctx, {"title": "Runbook", "url": URL})
    assert out["ok"] is False and out["error"] == "add_failed"
    assert "invalid_arguments" in out["message"]


@pytest.mark.asyncio
async def test_add_survives_a_response_with_no_bookmark_record():
    """A landed write with an unreadable echo is still a landed write."""
    ctx = _ctx(add={"ok": True})
    out = await bt.execute_add_bookmark(ctx, {"title": "Runbook", "url": URL})
    assert out["ok"] is True and out["bookmark_id"] is None
    assert out["title"] == "Runbook" and out["link"] == URL


# ================================================================================ remove

@pytest.mark.asyncio
async def test_remove_lists_first_then_removes():
    ctx = _ctx()
    out = await bt.execute_remove_bookmark(ctx, {"bookmark_id": BM})
    assert out["ok"] is True and out["removed"] is True
    assert out["title"] == "Deploy runbook"
    _web(ctx).bookmarks_list.assert_awaited_once_with(channel_id=CH)
    _web(ctx).bookmarks_remove.assert_awaited_once_with(channel_id=CH, bookmark_id=BM)


@pytest.mark.asyncio
async def test_remove_refuses_an_id_the_live_listing_does_not_contain():
    """The whole point of the fresh listing: a remembered or invented id removes nothing."""
    ctx = _ctx(listing=_listing(_bookmark("Bk_OTHER", "Dashboard")))
    out = await bt.execute_remove_bookmark(ctx, {"bookmark_id": BM})
    assert out["ok"] is False and out["error"] == "unknown_bookmark"
    assert [b["bookmark_id"] for b in out["bookmarks"]] == ["Bk_OTHER"]
    assert _web(ctx).bookmarks_remove.await_count == 0


@pytest.mark.asyncio
async def test_remove_fails_closed_when_the_listing_cannot_be_read():
    ctx = _ctx()
    _web(ctx).bookmarks_list.side_effect = _api_error("ratelimited")
    out = await bt.execute_remove_bookmark(ctx, {"bookmark_id": BM})
    assert out["ok"] is False and out["error"] == "list_failed"
    assert "ratelimited" in out["message"]
    assert _web(ctx).bookmarks_remove.await_count == 0


@pytest.mark.parametrize("args", [{}, {"bookmark_id": "  "}, {"bookmark_id": 12}])
@pytest.mark.asyncio
async def test_remove_refuses_bad_arguments_without_calling_slack(args):
    ctx = _ctx()
    out = await bt.execute_remove_bookmark(ctx, args)
    assert out["ok"] is False and out["error"] == "missing_bookmark_id"
    assert _web(ctx).bookmarks_list.await_count == 0
    assert _web(ctx).bookmarks_remove.await_count == 0


@pytest.mark.asyncio
async def test_remove_surfaces_slacks_own_error_name():
    ctx = _ctx()
    _web(ctx).bookmarks_remove.side_effect = _api_error("invalid_bookmark_type")
    out = await bt.execute_remove_bookmark(ctx, {"bookmark_id": BM})
    assert out["ok"] is False and out["error"] == "remove_failed"
    assert "invalid_bookmark_type" in out["message"]


# ================================================================================ contract

@pytest.mark.asyncio
async def test_no_executor_ever_raises():
    """A client with no Slack transport at all is the harshest shape a context can take."""
    ctx = ToolContext(channel_id=CH, client=None)
    for execute, args in ((bt.execute_list_bookmarks, {}),
                          (bt.execute_add_bookmark, {"title": "x", "url": URL}),
                          (bt.execute_remove_bookmark, {"bookmark_id": BM})):
        out = await execute(ctx, args)
        assert out["ok"] is False and out["error"] == "unavailable"
