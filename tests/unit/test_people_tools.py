"""F29 — people awareness tools + shared summary formatter.

Covers: format_people_summary (both pieces, either missing, none); lookup_user
(id path, pasted <@id> mention, name path via cache, name path via DB rows,
ambiguous → candidates, unknown → hint, non-Slack platform); list_channel_members
(pagination, name resolution, name cap + LOUD truncation note, DM refusal); and
registry gating on ENABLE_PEOPLE_TOOLS.
"""
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from config import config
from message_processor.people_tools import (
    execute_list_channel_members,
    execute_lookup_user,
    format_people_summary,
    get_list_channel_members_schema,
    get_lookup_user_schema,
    register_people_tools,
    MEMBERS_NAME_CAP,
)
from tool_registry import ToolContext, ToolRegistry

CHANNEL = "C04QDHE8W8M"


# ---------------------------------------------------- format_people_summary

def test_summary_both_pieces():
    assert format_people_summary(12, ["Erin Evans", "Claude"]) == \
        "~12 members; recently active: Erin Evans, Claude"


def test_summary_singular_member():
    assert format_people_summary(1, []) == "~1 member"


def test_summary_only_speakers_when_count_missing():
    assert format_people_summary(None, ["Alice"]) == "recently active: Alice"


def test_summary_only_count_when_no_speakers():
    assert format_people_summary(9, []) == "~9 members"


def test_summary_none_when_empty():
    assert format_people_summary(None, []) is None
    assert format_people_summary(0, []) is None          # zero is not a useful count
    assert format_people_summary("bad", None) is None    # non-int count degrades to None


def test_summary_drops_blank_names():
    assert format_people_summary(None, ["", "  ", "Bob"]) == "recently active: Bob"


# --------------------------------------------------------------- schemas

def test_schema_shapes():
    lu = get_lookup_user_schema()
    assert lu["type"] == "function" and lu["name"] == "lookup_user"
    assert lu["parameters"]["required"] == ["user"]
    # Discoverability lesson (F25): teaches that any seen name is enough, no id needed.
    assert "don't need a slack id" in lu["description"].lower() or \
        "do not need a slack id" in lu["description"].lower()
    lm = get_list_channel_members_schema()
    assert lm["name"] == "list_channel_members"
    assert lm["parameters"]["required"] == []


# ----------------------------------------------------------- test doubles

def _client(users_info=None, members_pages=None, user_cache=None, get_username=None):
    api = SimpleNamespace()
    if users_info is not None:
        api.users_info = AsyncMock(return_value=users_info)
    if members_pages is not None:
        api.conversations_members = AsyncMock(side_effect=list(members_pages))
    client = SimpleNamespace(app=SimpleNamespace(client=api),
                             user_cache=user_cache or {})
    client.get_username = get_username or AsyncMock(side_effect=lambda uid, c: f"name-of-{uid}")
    return client, api


def _users_info(user_id="U07PETER", display="peter", real="Erin Evans",
                title="Engineer", tz_label="Eastern", is_bot=False, email="peter@x.com",
                status_text="", status_emoji=""):
    return {"ok": True, "user": {
        "id": user_id, "name": display, "is_bot": is_bot, "tz_label": tz_label,
        "profile": {"display_name": display, "real_name": real, "title": title,
                    "email": email, "status_text": status_text, "status_emoji": status_emoji}}}


def _ctx(client, db=None, **kw):
    defaults = dict(channel_id=CHANNEL, thread_ts="1.0", user_id="U07PETER",
                    client=client, db=db, is_dm=False)
    defaults.update(kw)
    return ToolContext(**defaults)


# --------------------------------------------------------- lookup_user

@pytest.mark.asyncio
async def test_lookup_by_bare_id():
    client, api = _client(users_info=_users_info())
    res = await execute_lookup_user(_ctx(client), {"user": "U07PETER"})
    assert res["ok"] and res["id"] == "U07PETER"
    assert res["real_name"] == "Erin Evans" and res["title"] == "Engineer"
    assert res["timezone"] == "Eastern"
    # Profile-CARD facts only: the retired fetch_user_profile's no-email rail carries over.
    assert "email" not in res
    api.users_info.assert_awaited_once_with(user="U07PETER")


@pytest.mark.asyncio
async def test_lookup_strips_pasted_mention():
    client, api = _client(users_info=_users_info())
    await execute_lookup_user(_ctx(client), {"user": "<@U07PETER|peter>"})
    api.users_info.assert_awaited_once_with(user="U07PETER")


@pytest.mark.asyncio
async def test_lookup_by_name_via_cache_uses_fresh_users_info():
    client, api = _client(
        users_info=_users_info(),
        user_cache={"U07PETER": {"username": "peter", "real_name": "Erin Evans"}})
    db = MagicMock()
    db.get_all_users_async = AsyncMock(return_value=[])
    # Case-insensitive real-name match resolves to the id, then FRESH users.info is fetched.
    res = await execute_lookup_user(_ctx(client, db=db), {"user": "erin evans"})
    assert res["ok"] and res["id"] == "U07PETER"
    api.users_info.assert_awaited_once_with(user="U07PETER")


@pytest.mark.asyncio
async def test_lookup_by_name_via_db_rows():
    client, api = _client(users_info=_users_info(user_id="U9", display="dana", real="Dana Lee"))
    db = MagicMock()
    db.get_all_users_async = AsyncMock(return_value=[
        {"user_id": "U9", "username": "dana", "real_name": "Dana Lee"}])
    res = await execute_lookup_user(_ctx(client, db=db), {"user": "@dana"})
    assert res["ok"] and res["id"] == "U9"


@pytest.mark.asyncio
async def test_lookup_ambiguous_returns_candidates():
    client, api = _client(user_cache={
        "U1": {"username": "sam", "real_name": "Sam One"},
        "U2": {"username": "sammy", "real_name": "Sam One"}})
    db = MagicMock()
    db.get_all_users_async = AsyncMock(return_value=[])
    res = await execute_lookup_user(_ctx(client, db=db), {"user": "Sam One"})
    assert res["ok"] is False and res["error"] == "ambiguous"
    ids = {c["id"] for c in res["candidates"]}
    assert ids == {"U1", "U2"}


@pytest.mark.asyncio
async def test_lookup_unknown_name_gives_hint():
    client, _ = _client()
    db = MagicMock()
    db.get_all_users_async = AsyncMock(return_value=[])
    res = await execute_lookup_user(_ctx(client, db=db), {"user": "Nobody Here"})
    assert res["ok"] is False and res["error"] == "not_found" and res["hint"]


@pytest.mark.asyncio
async def test_lookup_blank_user_arg():
    client, _ = _client()
    res = await execute_lookup_user(_ctx(client), {"user": "  "})
    assert res["ok"] is False and res["error"] == "bad_arguments"


@pytest.mark.asyncio
async def test_lookup_non_slack_platform():
    client = SimpleNamespace(app=None, user_cache={})
    res = await execute_lookup_user(_ctx(client), {"user": "U07PETER"})
    assert res["ok"] is False and res["error"] == "unavailable"


# ------------------------------------------------- list_channel_members

def _members_page(members, cursor=""):
    return {"ok": True, "members": members,
            "response_metadata": {"next_cursor": cursor}}


@pytest.mark.asyncio
async def test_list_members_paginates_and_resolves_names():
    pages = [_members_page(["U1", "U2"], cursor="c1"), _members_page(["U3"])]
    client, api = _client(members_pages=pages)
    res = await execute_list_channel_members(_ctx(client), {})
    assert res["ok"] and res["total_members"] == 3
    assert res["members"] == ["name-of-U1", "name-of-U2", "name-of-U3"]
    assert "truncated" not in res
    assert api.conversations_members.await_count == 2


@pytest.mark.asyncio
async def test_list_members_caps_names_with_loud_note():
    ids = [f"U{i}" for i in range(MEMBERS_NAME_CAP + 5)]
    client, api = _client(members_pages=[_members_page(ids)])
    res = await execute_list_channel_members(_ctx(client), {})
    assert res["ok"] and res["total_members"] == MEMBERS_NAME_CAP + 5
    assert len(res["members"]) == MEMBERS_NAME_CAP           # names capped
    assert res["truncated"] is True
    assert "PARTIAL" in res["note"] and str(MEMBERS_NAME_CAP) in res["note"]


@pytest.mark.asyncio
async def test_list_members_refuses_dm():
    client, _ = _client(members_pages=[_members_page([])])
    res = await execute_list_channel_members(_ctx(client, is_dm=True), {})
    assert res["ok"] is False and res["error"] == "not_a_channel"


@pytest.mark.asyncio
async def test_list_members_api_error():
    client, api = _client()
    api.conversations_members = AsyncMock(return_value={"ok": False, "error": "channel_not_found"})
    res = await execute_list_channel_members(_ctx(client), {})
    assert res["ok"] is False and res["error"] == "channel_not_found"


# --------------------------------------------------------- registry gating

def test_registry_gating_on_enable_people_tools():
    from slack_client.base import SlackBot

    def build(flag):
        bot = SlackBot.__new__(SlackBot)
        with patch.object(config, "enable_people_tools", flag), \
             patch.object(config, "enable_channel_memory", False), \
             patch.object(config, "enable_history_tools", False), \
             patch.object(config, "enable_reactions", False), \
             patch.object(config, "enable_search_tool", False), \
             patch.object(config, "enable_post_to_thread_tool", False), \
             patch.object(config, "enable_read_document_tool", False):
            with patch.object(SlackBot, "get_history_tools_for_openai", return_value=[], create=True):
                registry = SlackBot._build_tool_registry(bot)
        return {s["name"] for s in registry.schemas()}

    assert {"lookup_user", "list_channel_members"} <= build(True)
    assert build(False) == set()


def test_register_people_tools_adds_both():
    reg = ToolRegistry()
    register_people_tools(reg)
    assert {"lookup_user", "list_channel_members"} <= {s["name"] for s in reg.schemas()}
