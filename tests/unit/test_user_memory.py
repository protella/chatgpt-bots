"""Toolbelt T2 — per-user memory: the store, the DM tool surface, the DM injection, the modal.

The channel half is tests/unit/test_memory_tools.py. This file owns everything that is new:
`user_memory` and its accessors, the DM routing that replaced the old flat refusal, `list_facts`
on both surfaces, the "personal facts never load in a channel" ruling, and the settings modal's
view/delete section including the explicit forget-everything checkbox.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import config
from database import DatabaseManager, memory_content_hash
from message_processor import channel_steering
from message_processor.memory_tools import (
    execute_forget_fact,
    execute_list_facts,
    execute_remember_fact,
    execute_update_fact,
    get_list_facts_dm_schema,
    get_remember_fact_dm_schema,
    get_remember_fact_schema,
    register_memory_tools,
)
from settings_modal import SettingsModal
from tool_registry import SURFACE_CHANNEL, SURFACE_DM, ToolContext, ToolRegistry

DM = "D0BKX77NU66"
CHANNEL = "C0BKX77NU66"
USER = "U07DANA"
OTHER = "U07RILEY"


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    manager = DatabaseManager(platform="user_memory_tests")
    yield manager
    manager.conn.close()


def _ctx(store, **kw):
    defaults = dict(channel_id=DM, thread_ts="1.0", trigger_ts="1.0",
                    user_id=USER, db=store, is_dm=True)
    defaults.update(kw)
    return ToolContext(**defaults)


def _mock_db(user_rows=None, new_id=11):
    store = MagicMock()
    store.get_user_memory_async = AsyncMock(return_value=list(user_rows or []))
    store.add_user_memory_async = AsyncMock(return_value=new_id)
    store.update_user_fact_async = AsyncMock(return_value=True)
    store.delete_user_memory_async = AsyncMock(return_value=True)
    store.get_channel_memory_async = AsyncMock(return_value=[])
    store.add_channel_memory_async = AsyncMock(return_value=99)
    store.get_channel_policy_async = AsyncMock(return_value=None)
    return store


def _user_row(row_id, content, author=USER, updated_ts="2026-08-01"):
    return {"id": row_id, "user_id": USER, "content": content, "author": author,
            "created_ts": updated_ts, "updated_ts": updated_ts}


# --------------------------------------------------------------------------- the store

@pytest.mark.asyncio
async def test_table_exists_and_the_migration_step_is_idempotent(db):
    cursor = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='user_memory'")
    assert cursor.fetchone() is not None
    # Re-running the whole chain must converge, not raise or duplicate.
    db._run_migrations()
    db._run_migrations()
    cursor = db.conn.execute("SELECT COUNT(*) FROM user_memory")
    assert cursor.fetchone()[0] == 0


@pytest.mark.asyncio
async def test_round_trip_and_owner_scoping(db):
    mine = await db.add_user_memory_async(USER, "prefers the code first", author=USER)
    theirs = await db.add_user_memory_async(OTHER, "works from the Lisbon office", author=OTHER)

    rows = await db.get_user_memory_async(USER)
    assert [r["content"] for r in rows] == ["prefers the code first"]
    assert rows[0]["author"] == USER

    # Another person's row is not reachable through my id — for either write.
    assert await db.update_user_fact_async(USER, theirs, "hijacked") is False
    assert await db.delete_user_memory_async(USER, theirs) is False
    assert [r["content"] for r in await db.get_user_memory_async(OTHER)] == \
        ["works from the Lisbon office"]

    assert await db.update_user_fact_async(USER, mine, "prefers short answers") is True
    assert (await db.get_user_memory_async(USER))[0]["content"] == "prefers short answers"
    assert await db.delete_user_memory_async(USER, mine) is True
    assert await db.get_user_memory_async(USER) == []


@pytest.mark.asyncio
async def test_delete_all_is_one_persons_store_only(db):
    for text in ("first", "second", "third"):
        await db.add_user_memory_async(USER, text, author=USER)
    await db.add_user_memory_async(OTHER, "untouched", author=OTHER)

    assert await db.delete_all_user_memory_async(USER) == 3
    assert await db.get_user_memory_async(USER) == []
    assert len(await db.get_user_memory_async(OTHER)) == 1


@pytest.mark.asyncio
async def test_reconcile_keeps_deletes_and_adds(db):
    keep = await db.add_user_memory_async(USER, "keep me", author=USER)
    drop = await db.add_user_memory_async(USER, "drop me", author=USER)
    seed = [[keep, memory_content_hash("keep me")], [drop, memory_content_hash("drop me")]]

    result = await db.reconcile_user_memory_from_textarea_async(
        USER, seed, ["keep me", "brand new"], author=USER, max_rows=25)

    assert result["deleted"] == [drop]
    assert result["added"] == ["brand new"]
    assert result["conflicts"] == 0 and result["over_cap"] == 0
    assert sorted(r["content"] for r in await db.get_user_memory_async(USER)) == \
        ["brand new", "keep me"]


@pytest.mark.asyncio
async def test_reconcile_refuses_to_clobber_a_row_changed_since_open(db):
    row = await db.add_user_memory_async(USER, "as it was at open", author=USER)
    seed = [[row, memory_content_hash("as it was at open")]]
    await db.update_user_fact_async(USER, row, "changed elsewhere")

    result = await db.reconcile_user_memory_from_textarea_async(
        USER, seed, [], author=USER, max_rows=25)

    assert result["conflicts"] == 1 and result["deleted"] == []
    assert (await db.get_user_memory_async(USER))[0]["content"] == "changed elsewhere"


@pytest.mark.asyncio
async def test_reconcile_honours_the_row_cap(db):
    seed: list = []
    result = await db.reconcile_user_memory_from_textarea_async(
        USER, seed, ["one", "two", "three"], author=USER, max_rows=2)
    assert result["added"] == ["one", "two"]
    assert result["over_cap"] == 1


@pytest.mark.asyncio
async def test_blanking_the_box_leaves_rows_it_never_showed(db):
    """The reason the forget-everything checkbox exists: an unseeded row survives a blanked box."""
    shown = await db.add_user_memory_async(USER, "shown in the box", author=USER)
    await db.add_user_memory_async(USER, "past the textarea budget", author=USER)
    seed = [[shown, memory_content_hash("shown in the box")]]

    await db.reconcile_user_memory_from_textarea_async(USER, seed, [], author=USER, max_rows=25)

    assert [r["content"] for r in await db.get_user_memory_async(USER)] == \
        ["past the textarea budget"]
    # …and the full-store delete is what actually forgets it.
    assert await db.delete_all_user_memory_async(USER) == 1


# --------------------------------------------------------------------------- the DM tools

@pytest.mark.asyncio
async def test_dm_remember_writes_the_user_store(db):
    result = await execute_remember_fact(_ctx(db), {"content": "Dana Whitfield deploys on Thursdays."})
    assert result["ok"] is True
    rows = await db.get_user_memory_async(USER)
    assert [r["content"] for r in rows] == ["Dana Whitfield deploys on Thursdays."]
    assert rows[0]["author"] == USER


@pytest.mark.asyncio
async def test_dm_remember_cap_counts_user_rows_and_lists_the_stalest(db):
    for i in range(1, 4):
        await db.add_user_memory_async(USER, f"fact {i}", author=USER)
    with patch.object(config, "memory_max_rows", 3):
        result = await execute_remember_fact(_ctx(db), {"content": "one more"})
    assert result["ok"] is False and result["error"] == "memory_full"
    assert [r["content"] for r in result["oldest"]] == ["fact 1", "fact 2", "fact 3"]
    assert len(await db.get_user_memory_async(USER)) == 3


@pytest.mark.asyncio
async def test_dm_remember_truncates_at_the_shared_limit(db):
    from message_processor.memory_tools import MAX_FACT_CHARS

    await execute_remember_fact(_ctx(db), {"content": "x" * (MAX_FACT_CHARS + 50)})
    assert len((await db.get_user_memory_async(USER))[0]["content"]) == MAX_FACT_CHARS


@pytest.mark.asyncio
async def test_dm_update_and_forget_resolve_against_my_own_rows(db):
    mine = await db.add_user_memory_async(USER, "old wording", author=USER)

    updated = await execute_update_fact(_ctx(db), {"id": mine, "content": "new wording"})
    assert updated == {"ok": True, "id": mine, "content": "new wording"}

    forgotten = await execute_forget_fact(_ctx(db), {"id": mine})
    assert forgotten == {"ok": True, "id": mine, "forgot": "new wording"}
    assert await db.get_user_memory_async(USER) == []


@pytest.mark.asyncio
async def test_dm_tools_cannot_reach_another_persons_row(db):
    theirs = await db.add_user_memory_async(OTHER, "their private fact", author=OTHER)

    assert (await execute_update_fact(_ctx(db), {"id": theirs, "content": "x"}))["error"] == "not_found"
    assert (await execute_forget_fact(_ctx(db), {"id": theirs}))["error"] == "not_found"
    assert (await execute_forget_fact(_ctx(db), {"id": "abc"}))["error"] == "bad_arguments"
    assert len(await db.get_user_memory_async(OTHER)) == 1


@pytest.mark.asyncio
async def test_a_dm_call_never_touches_the_channel_store():
    store = _mock_db()
    await execute_remember_fact(_ctx(store), {"content": "personal"})
    await execute_list_facts(_ctx(store), {})
    store.get_channel_memory_async.assert_not_awaited()
    store.add_channel_memory_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_executors_refuse_when_the_surface_flag_is_off():
    store = _mock_db()
    with patch.object(config, "enable_user_memory", False):
        result = await execute_remember_fact(_ctx(store), {"content": "x"})
    assert result["ok"] is False and result["error"] == "memory_disabled"
    store.add_user_memory_async.assert_not_awaited()

    with patch.object(config, "enable_channel_memory", False):
        result = await execute_remember_fact(
            _ctx(store, is_dm=False, channel_id=CHANNEL), {"content": "x"})
    assert result["ok"] is False and result["error"] == "memory_disabled"


@pytest.mark.asyncio
async def test_a_dm_without_a_requester_is_refused():
    result = await execute_remember_fact(_ctx(_mock_db(), user_id=None), {"content": "x"})
    assert result["ok"] is False and result["error"] == "no_user"


# --------------------------------------------------------------------------- list_facts

@pytest.mark.asyncio
async def test_list_facts_in_a_dm_returns_id_sorted_facts(db):
    for text in ("alpha", "beta", "gamma"):
        await db.add_user_memory_async(USER, text, author=USER)

    result = await execute_list_facts(_ctx(db), {})
    assert result["ok"] is True and result["scope"] == "user"
    assert [f["content"] for f in result["facts"]] == ["alpha", "beta", "gamma"]
    assert [f["id"] for f in result["facts"]] == sorted(f["id"] for f in result["facts"])
    assert result["facts"][0]["author"] == USER and result["facts"][0]["updated"]


@pytest.mark.asyncio
async def test_list_facts_filters_case_insensitively(db):
    await db.add_user_memory_async(USER, "Prefers Thursday deploys", author=USER)
    await db.add_user_memory_async(USER, "Owns the billing service", author=USER)

    result = await execute_list_facts(_ctx(db), {"query": "thursday"})
    assert [f["content"] for f in result["facts"]] == ["Prefers Thursday deploys"]
    assert result["count"] == 1


@pytest.mark.asyncio
async def test_list_facts_on_a_channel_shows_the_ids_the_prompt_shows():
    """The blind-edit fix: listed ids must equal rendered ids, workspace rows flagged read-only
    and the gate's preference markers absent — exactly what render_snapshot does."""
    rows = [
        {"id": 5, "channel_id": CHANNEL, "scope": "channel", "content": "beta",
         "author": None, "created_ts": "2026-08-02", "updated_ts": "2026-08-02"},
        {"id": 2, "channel_id": CHANNEL, "scope": "channel", "content": "alpha",
         "author": None, "created_ts": "2026-08-01", "updated_ts": "2026-08-01"},
        {"id": 8, "channel_id": CHANNEL, "scope": "workspace", "content": "shared",
         "author": None, "created_ts": "2026-08-03", "updated_ts": "2026-08-03"},
        {"id": 9, "channel_id": CHANNEL, "scope": "channel", "content": "a preference",
         "author": channel_steering.PREF_AUTHOR_PREFIX + "reactions",
         "created_ts": "2026-08-04", "updated_ts": "2026-08-04"},
    ]
    store = _mock_db()
    store.get_channel_memory_async = AsyncMock(return_value=rows)

    result = await execute_list_facts(_ctx(store, is_dm=False, channel_id=CHANNEL), {})
    assert result["scope"] == "channel"
    assert [f["id"] for f in result["facts"]] == [2, 5, 8]
    assert result["facts"][2]["read_only"] is True
    assert all("read_only" not in f for f in result["facts"][:2])

    rendered = channel_steering.render_snapshot(None, rows).text
    for fact in result["facts"]:
        assert f"[#{fact['id']}]" in rendered
    assert "[#9]" not in rendered


# --------------------------------------------------------------------------- registration

def test_the_two_surfaces_carry_different_flags_and_different_words():
    registry = ToolRegistry()
    register_memory_tools(registry)
    names = {"remember_fact", "update_fact", "forget_fact", "list_facts"}

    with patch.object(config, "enable_user_memory", True), \
         patch.object(config, "enable_channel_memory", True):
        dm = {s["name"]: s for s in registry.schemas({}, surface=SURFACE_DM)}
        channel = {s["name"]: s for s in registry.schemas({}, surface=SURFACE_CHANNEL)}
    assert names <= set(dm) and names <= set(channel)
    # The DM wording is about a PERSON, the channel wording about a CHANNEL — the split codex
    # required, since the channel schemas describe channel memory in so many words. (The DM text
    # does mention channels, once, to promise a personal fact is never shown in one.)
    for tool in names:
        assert dm[tool]["description"] != channel[tool]["description"]
    assert "person" in dm["remember_fact"]["description"].lower()
    assert "this channel's long-term memory" in channel["remember_fact"]["description"].lower()
    # A DM write takes no scope argument: there is only one user store.
    assert "scope" not in get_remember_fact_dm_schema()["parameters"]["properties"]
    assert "scope" in get_remember_fact_schema()["parameters"]["properties"]
    assert get_list_facts_dm_schema()["name"] == "list_facts"

    with patch.object(config, "enable_user_memory", False), \
         patch.object(config, "enable_channel_memory", True):
        assert names.isdisjoint({s["name"] for s in registry.schemas({}, surface=SURFACE_DM)})
        assert names <= {s["name"] for s in registry.schemas({}, surface=SURFACE_CHANNEL)}

    with patch.object(config, "enable_user_memory", True), \
         patch.object(config, "enable_channel_memory", False):
        assert names <= {s["name"] for s in registry.schemas({}, surface=SURFACE_DM)}
        assert names.isdisjoint({s["name"] for s in registry.schemas({}, surface=SURFACE_CHANNEL)})


# --------------------------------------------------------------------------- DM injection

@pytest.mark.asyncio
async def test_dm_snapshot_carries_user_facts_under_their_own_heading(db):
    await db.add_user_memory_async(USER, "beta", author=USER)
    await db.add_user_memory_async(USER, "alpha", author=USER)

    snap = await channel_steering.load_snapshot(db, DM, user_id=USER)

    assert channel_steering.USER_FACT_HEADING in snap.text
    ids = [r["id"] for r in await db.get_user_memory_async(USER)]
    assert f"- [#{min(ids)}] beta\n- [#{max(ids)}] alpha" in snap.text
    # Deterministic: the same state renders the same bytes (prompt-cache stability).
    assert (await channel_steering.load_snapshot(db, DM, user_id=USER)).text == snap.text


@pytest.mark.asyncio
async def test_user_facts_never_load_on_a_channel_surface():
    """Owner ruling, made structural: even a caller that passes a user_id gets nothing."""
    store = _mock_db(user_rows=[_user_row(1, "a personal detail")])

    snap = await channel_steering.load_snapshot(store, CHANNEL, user_id=USER)

    store.get_user_memory_async.assert_not_awaited()
    assert channel_steering.USER_FACT_HEADING not in (snap.text or "")


@pytest.mark.asyncio
async def test_user_facts_are_off_when_the_flag_is_off():
    store = _mock_db(user_rows=[_user_row(1, "a personal detail")])
    snap = await channel_steering.load_snapshot(store, DM, user_id=USER,
                                                user_memory_enabled=False)
    store.get_user_memory_async.assert_not_awaited()
    assert channel_steering.USER_FACT_HEADING not in (snap.text or "")


@pytest.mark.asyncio
async def test_a_dm_with_no_requester_reads_nothing():
    store = _mock_db(user_rows=[_user_row(1, "a personal detail")])
    await channel_steering.load_snapshot(store, DM, user_id=None)
    store.get_user_memory_async.assert_not_awaited()


# --------------------------------------------------------------------------- the settings modal

def _modal(store):
    modal = SettingsModal.__new__(SettingsModal)
    modal.db = store
    modal.logger_name = "SettingsModal"
    return modal


def _blocks_by_id(view):
    return {b.get("block_id"): b for b in view["blocks"] if b.get("block_id")}


@pytest.mark.asyncio
async def test_the_user_modal_seeds_the_memory_box_and_offers_the_forget_checkbox(db):
    await db.add_user_memory_async(USER, "prefers short answers", author=USER)
    await db.add_user_memory_async(USER, "owns the billing service", author=USER)
    store = _modal(db)
    store.db.create_modal_session_async = AsyncMock()

    view = await store.build_settings_modal(USER, "trigger-1", current_settings={"model": "gpt-5.6-sol"})

    blocks = _blocks_by_id(view)
    box = blocks[SettingsModal.USER_MEMORY_BLOCK]["element"]
    assert box["initial_value"] == "prefers short answers\nowns the billing service"
    forget = blocks[SettingsModal.USER_MEMORY_FORGET_BLOCK]["element"]
    assert forget["type"] == "checkboxes"
    assert "initial_options" not in forget
    assert forget["options"][0]["value"] == SettingsModal.USER_MEMORY_FORGET_VALUE
    # The seed rides the session row, not private_metadata (which holds only the session id).
    session_state = store.db.create_modal_session_async.await_args.args[2]
    assert [pair[1] for pair in session_state["user_mem_seed"]] == [
        memory_content_hash("prefers short answers"),
        memory_content_hash("owns the billing service")]
    assert "user_mem_seed" not in json.loads(view["private_metadata"])


@pytest.mark.asyncio
async def test_rows_past_the_textarea_budget_are_counted_not_seeded(db):
    long_fact = "x" * 1500
    for _ in range(3):
        await db.add_user_memory_async(USER, long_fact, author=USER)
    store = _modal(db)
    store.db.create_modal_session_async = AsyncMock()

    view = await store.build_settings_modal(USER, "trigger-1", current_settings={"model": "gpt-5.6-sol"})

    session_state = store.db.create_modal_session_async.await_args.args[2]
    assert len(session_state["user_mem_seed"]) == 1
    contexts = [b for b in view["blocks"] if b.get("type") == "context"]
    assert any("+2 more not shown" in b["elements"][0]["text"] for b in contexts)


@pytest.mark.asyncio
async def test_the_memory_section_disappears_with_the_flag(db):
    await db.add_user_memory_async(USER, "prefers short answers", author=USER)
    store = _modal(db)
    store.db.create_modal_session_async = AsyncMock()

    with patch.object(config, "enable_user_memory", False):
        view = await store.build_settings_modal(
            USER, "trigger-1", current_settings={"model": "gpt-5.6-sol"})

    assert SettingsModal.USER_MEMORY_BLOCK not in _blocks_by_id(view)
    assert SettingsModal.USER_MEMORY_FORGET_BLOCK not in _blocks_by_id(view)


def test_extract_user_memory_tells_absent_apart_from_blank():
    modal = SettingsModal.__new__(SettingsModal)
    # A modal opened before this section existed: nothing to reconcile, NOT "forget everything".
    assert modal.extract_user_memory({"values": {"custom_instructions_block": {}}}) is None

    submitted = modal.extract_user_memory({"values": {
        SettingsModal.USER_MEMORY_BLOCK: {
            SettingsModal.USER_MEMORY_ACTION: {"value": " keeps this \n\n keeps this \nand this "}},
        SettingsModal.USER_MEMORY_FORGET_BLOCK: {
            SettingsModal.USER_MEMORY_FORGET_ACTION: {"selected_options": [
                {"value": SettingsModal.USER_MEMORY_FORGET_VALUE}]}},
    }})
    assert submitted is not None
    assert submitted["lines"] == ["keeps this", "and this"]   # normalized, deduped, in order
    assert submitted["forget_all"] is True

    blank = modal.extract_user_memory({"values": {
        SettingsModal.USER_MEMORY_BLOCK: {SettingsModal.USER_MEMORY_ACTION: {"value": ""}},
        SettingsModal.USER_MEMORY_FORGET_BLOCK: {
            SettingsModal.USER_MEMORY_FORGET_ACTION: {"selected_options": []}},
    }})
    assert blank == {"raw": "", "lines": [], "forget_all": False}


# --------------------------------------------------------------------------- the save path

class _FakeApp:
    def __init__(self):
        self.views = {}

    def view(self, callback_id):
        def deco(fn):
            self.views[callback_id] = fn
            return fn
        return deco

    def action(self, *_a, **_k):
        return lambda fn: fn

    def command(self, *_a, **_k):
        return lambda fn: fn

    def shortcut(self, *_a, **_k):
        return lambda fn: fn

    def event(self, *_a, **_k):
        return lambda fn: fn


def _settings_host(store):
    from slack_client.event_handlers.settings import SlackSettingsHandlersMixin

    host = SlackSettingsHandlersMixin.__new__(SlackSettingsHandlersMixin)
    host.app = _FakeApp()
    host.db = store
    host.settings_modal = SettingsModal.__new__(SettingsModal)
    host.log_info = host.log_error = host.log_debug = host.log_warning = lambda *a, **k: None
    host._register_settings_handlers()
    return host


async def _submit(host, session_id, memory_state):
    await host.app.views["settings_modal"](
        ack=AsyncMock(),
        body={"user": {"id": USER},
              "view": {"private_metadata": json.dumps({"session_id": session_id})}},
        view={"id": "V1", "callback_id": "settings_modal",
              "private_metadata": json.dumps({"session_id": session_id}),
              "state": {"values": memory_state}},
        client=AsyncMock(),
    )


def _memory_state(text, forget=False):
    return {
        SettingsModal.USER_MEMORY_BLOCK: {SettingsModal.USER_MEMORY_ACTION: {"value": text}},
        SettingsModal.USER_MEMORY_FORGET_BLOCK: {
            SettingsModal.USER_MEMORY_FORGET_ACTION: {
                "selected_options": ([{"value": SettingsModal.USER_MEMORY_FORGET_VALUE}]
                                     if forget else [])}},
    }


@pytest.mark.asyncio
async def test_saving_the_modal_reconciles_the_edited_box(db):
    keep = await db.add_user_memory_async(USER, "keep me", author=USER)
    drop = await db.add_user_memory_async(USER, "drop me", author=USER)
    session_id = "session-1"
    await db.create_modal_session_async(session_id, USER, {
        "settings": {}, "scope": "global",
        "user_mem_seed": [[keep, memory_content_hash("keep me")],
                          [drop, memory_content_hash("drop me")]]})

    await _submit(_settings_host(db), session_id, _memory_state("keep me\nadded by hand"))

    assert sorted(r["content"] for r in await db.get_user_memory_async(USER)) == \
        ["added by hand", "keep me"]


@pytest.mark.asyncio
async def test_the_checkbox_forgets_rows_the_box_never_showed(db):
    """The binding correction: blanking the textarea cannot reach an unseeded row; this can."""
    shown = await db.add_user_memory_async(USER, "shown in the box", author=USER)
    await db.add_user_memory_async(USER, "never shown", author=USER)
    session_id = "session-2"
    await db.create_modal_session_async(session_id, USER, {
        "settings": {}, "scope": "global",
        "user_mem_seed": [[shown, memory_content_hash("shown in the box")]]})

    # Text still in the box AND the checkbox ticked: the explicit request wins outright.
    await _submit(_settings_host(db), session_id, _memory_state("shown in the box", forget=True))

    assert await db.get_user_memory_async(USER) == []


@pytest.mark.asyncio
async def test_a_modal_without_the_section_leaves_memory_alone(db):
    await db.add_user_memory_async(USER, "survives an old modal", author=USER)
    session_id = "session-3"
    await db.create_modal_session_async(session_id, USER, {"settings": {}, "scope": "global"})

    await _submit(_settings_host(db), session_id, {"custom_instructions_block": {}})

    assert [r["content"] for r in await db.get_user_memory_async(USER)] == \
        ["survives an old modal"]


# --------------------------------------------------------------------------- argument hygiene

@pytest.mark.asyncio
async def test_list_facts_returns_the_whole_store():
    """No result cap: `list_facts` is the full-view surface the modal's '+N more' points at, and
    memory_max_rows is the only thing that bounds a store."""
    import message_processor.memory_tools as memory_tools

    assert not hasattr(memory_tools, "LIST_FACTS_MAX")
    rows = [_user_row(i, f"fact {i}") for i in range(1, 121)]
    result = await execute_list_facts(_ctx(_mock_db(user_rows=rows)), {})
    assert result["count"] == 120
    assert [f["id"] for f in result["facts"]] == list(range(1, 121))


@pytest.mark.asyncio
async def test_non_string_arguments_are_refused_not_raised():
    """The never-raise invariant: the registry guarantees a dict, not what is inside it."""
    store = _mock_db()
    for bad in (123, {"a": 1}, ["x"], True, None):
        result = await execute_remember_fact(_ctx(store), {"content": bad})
        assert result["ok"] is False and result["error"] == "bad_arguments"
        result = await execute_update_fact(_ctx(store), {"id": 1, "content": bad})
        assert result["ok"] is False and result["error"] == "bad_arguments"
    store.add_user_memory_async.assert_not_awaited()
    store.update_user_fact_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_non_string_query_is_refused_not_treated_as_no_filter():
    """A present-but-unusable filter must not answer with the whole store."""
    store = _mock_db(user_rows=[_user_row(1, "alpha"), _user_row(2, "beta")])
    for bad in (42, {"a": 1}, ["x"], True):
        result = await execute_list_facts(_ctx(store), {"query": bad})
        assert result["ok"] is False and result["error"] == "bad_arguments"
        assert "facts" not in result
    store.get_user_memory_async.assert_not_awaited()
    # An absent query still means "everything".
    result = await execute_list_facts(_ctx(store), {})
    assert result["ok"] is True and result["count"] == 2


@pytest.mark.asyncio
async def test_a_non_integer_id_is_refused_not_raised():
    store = _mock_db()
    for bad in ({"a": 1}, ["x"], None, "not-a-number"):
        result = await execute_forget_fact(_ctx(store), {"id": bad})
        assert result["ok"] is False and result["error"] == "bad_arguments"
    store.delete_user_memory_async.assert_not_awaited()


# --------------------------------------------------------------------------- confirmation path

async def _confirm(host, session_id, client):
    """Drive the pushed custom-instructions confirmation modal's submission."""
    metadata = json.dumps({"session_id": session_id, "confirmed": True,
                           "form_values": {"custom_instructions": "be brief"}})
    await host.app.views["confirm_global_custom_instructions"](
        ack=MagicMock(),
        body={"user": {"id": USER}, "view": {"private_metadata": metadata}},
        view={"id": "V2", "private_metadata": metadata},
        client=client,
    )


@pytest.mark.asyncio
async def test_the_confirmation_path_applies_the_stashed_forget(db):
    await db.add_user_memory_async(USER, "should be forgotten", author=USER)
    session_id = "session-4"
    await db.create_modal_session_async(session_id, USER, {
        "settings": {}, "scope": "global",
        "user_mem_seed": [],
        "user_memory_pending": {"raw": "", "lines": [], "forget_all": True}})

    client = AsyncMock()
    await _confirm(_settings_host(db), session_id, client)

    assert await db.get_user_memory_async(USER) == []
    assert "Personal memory cleared" in client.chat_postMessage.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_the_confirmation_path_never_reports_a_failed_forget_as_done(db):
    """Codex finding: the confirmation used to discard the reconcile result and confirm anyway."""
    await db.add_user_memory_async(USER, "still here", author=USER)
    session_id = "session-5"
    await db.create_modal_session_async(session_id, USER, {
        "settings": {}, "scope": "global", "user_mem_seed": [],
        "user_memory_pending": {"raw": "", "lines": [], "forget_all": True}})

    host = _settings_host(db)
    client = AsyncMock()
    with patch.object(host.db, "delete_all_user_memory_async",
                      AsyncMock(side_effect=RuntimeError("db down"))):
        await _confirm(host, session_id, client)

    text = client.chat_postMessage.await_args.kwargs["text"]
    assert "Couldn't fully update personal memory" in text
    assert [r["content"] for r in await db.get_user_memory_async(USER)] == ["still here"]


@pytest.mark.asyncio
async def test_a_dropped_stash_warns_instead_of_vanishing(db):
    """The session row is the only copy of the edit at push time — a failed write must be loud."""
    # The confirmation is only pushed when a global custom-instructions value already exists.
    await db.create_default_user_preferences_async(USER, None)
    await db.update_user_preferences_async(USER, {"custom_instructions": "existing global"})
    session_id = "session-6"
    await db.create_modal_session_async(session_id, USER, {
        "settings": {}, "scope": "global", "thread_id": f"{CHANNEL}:1.0", "in_thread": True,
        "user_mem_seed": []})

    host = _settings_host(db)
    client = AsyncMock()
    state = _memory_state("a note I typed")
    state["custom_instructions_block"] = {"custom_instructions": {"value": "from this thread"}}
    with patch.object(host, "_update_session_data", AsyncMock(return_value=False)):
        await host.app.views["settings_modal"](
            ack=AsyncMock(),
            body={"user": {"id": USER},
                  "view": {"private_metadata": json.dumps({"session_id": session_id})}},
            view={"id": "V1", "callback_id": "settings_modal",
                  "private_metadata": json.dumps({"session_id": session_id}),
                  "state": {"values": state}},
            client=client,
        )

    assert "was NOT saved" in client.chat_postMessage.await_args.kwargs["text"]
