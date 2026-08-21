"""Phase C — model-invoked memory tools, CHANNEL surface.

Covers: the executors on a channel (happy paths, cap-hit with oldest-3 listing,
wrong-channel not_found, workspace-scope write refusal), author attribution,
[#id]-prefixed deterministic injection rendering, extractor fallback gating,
registry gating on ENABLE_CHANNEL_MEMORY, and guidance text.

The DM surface — where the same tool names reach the per-user store instead —
lives in tests/unit/test_user_memory.py, including the DM behaviour that used to
be a flat refusal here.
"""
import os

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from config import config
from message_processor.memory_tools import (
    execute_forget_fact,
    execute_remember_fact,
    execute_update_fact,
    get_forget_fact_schema,
    get_list_facts_schema,
    get_remember_fact_schema,
    get_update_fact_schema,
    register_memory_tools,
)
from message_processor.tool_registry import ToolContext, ToolRegistry

CHANNEL = "C0BKX77NU66"


def _row(id, content, scope="channel", channel_id=CHANNEL, updated_ts="2026-07-01"):
    return {"id": id, "channel_id": channel_id, "scope": scope, "content": content,
            "author": None, "created_ts": updated_ts, "updated_ts": updated_ts}


def _db(rows=None, new_id=42):
    db = MagicMock()
    db.get_channel_memory_async = AsyncMock(return_value=list(rows or []))
    db.add_channel_memory_async = AsyncMock(return_value=new_id)
    db.update_channel_memory_async = AsyncMock()
    db.update_channel_fact_async = AsyncMock(return_value=True)
    db.delete_channel_memory_async = AsyncMock()
    db.get_channel_policy_async = AsyncMock(return_value=None)
    return db


def _ctx(db, **kw):
    defaults = dict(channel_id=CHANNEL, thread_ts="1.0", trigger_ts="1.0",
                    user_id="U07PETER", db=db, is_dm=False)
    defaults.update(kw)
    return ToolContext(**defaults)


# --- schemas ---

def test_schema_shapes():
    for schema, name, required in [
        (get_remember_fact_schema(), "remember_fact", {"content"}),
        (get_update_fact_schema(), "update_fact", {"id", "content"}),
        (get_forget_fact_schema(), "forget_fact", {"id"}),
        (get_list_facts_schema(), "list_facts", set()),
    ]:
        assert schema["type"] == "function"
        assert schema["name"] == name
        assert set(schema["parameters"]["required"]) == required
    # writes are channel-scope only, enforced at the schema level too
    assert get_remember_fact_schema()["parameters"]["properties"]["scope"]["enum"] == ["channel"]


# --- remember_fact ---

@pytest.mark.asyncio
async def test_remember_happy_path_attributes_author():
    db = _db(rows=[], new_id=7)
    result = await execute_remember_fact(_ctx(db), {"content": "Sprint demos are on Fridays."})
    assert result == {"ok": True, "id": 7, "content": "Sprint demos are on Fridays."}
    db.add_channel_memory_async.assert_awaited_once_with(
        CHANNEL, "Sprint demos are on Fridays.", scope="channel", author="U07PETER"
    )


@pytest.mark.asyncio
async def test_over_budget_write_is_refused_with_the_whole_store_and_how_to_fix_it():
    """A store is bounded by CHARACTERS as of 2026-08-20, and a refusal has to be actionable:
    consolidating is the only way forward, and the model cannot merge notes it was not shown."""
    rows = [_row(i, f"fact {i} " + "z" * 20, updated_ts=f"2026-06-{i:02d}") for i in range(1, 6)]
    db = _db(rows=rows)
    with patch.object(config, "memory_store_max_chars", 140):
        result = await execute_remember_fact(_ctx(db), {"content": "one more"})
    assert result["ok"] is False and result["error"] == "memory_full"
    # the FULL store comes back, ids included, not a sample of the oldest
    assert [f["id"] for f in result["facts"]] == [1, 2, 3, 4, 5]
    assert [f["content"] for f in result["facts"]] == [r["content"] for r in rows]
    assert result["budget_chars"] == 140 and result["would_be_chars"] > 140
    assert "consolidate" in result["hint"]
    assert "NOTHING WAS SAVED" in result["message"]
    assert "update_fact" in result["message"] and "forget_fact" in result["message"]
    db.add_channel_memory_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_budget_is_the_serialized_store_newlines_included():
    """The budget measures what the settings modal puts in its one textarea — one fact per line —
    so the joining newlines count. Contents that sum to exactly the budget do not fit."""
    rows = [_row(1, "a" * 40), _row(2, "b" * 39)]
    db = _db(rows=rows, new_id=3)
    # 40 + 39 + 20 = 99 characters of content, but 101 as two newline-joined lines plus a third.
    with patch.object(config, "memory_store_max_chars", 100):
        refused = await execute_remember_fact(_ctx(db), {"content": "c" * 20})
    assert refused["ok"] is False and refused["error"] == "memory_full"
    assert refused["would_be_chars"] == 101
    with patch.object(config, "memory_store_max_chars", 101):
        fits = await execute_remember_fact(_ctx(db), {"content": "c" * 20})
    assert fits["ok"] is True


@pytest.mark.asyncio
async def test_row_count_alone_never_blocks_a_write():
    """The row cap is gone from this path: many short notes that fit the modal's box are fine.
    MEMORY_MAX_ROWS is left parsed for .env compatibility and must gate nothing here."""
    rows = [_row(i, f"note {i}") for i in range(1, 61)]
    db = _db(rows=rows, new_id=61)
    with patch.object(config, "memory_max_rows", 5):
        result = await execute_remember_fact(_ctx(db), {"content": "note 61"})
    assert result["ok"] is True
    db.add_channel_memory_async.assert_awaited_once()


@pytest.mark.asyncio
async def test_budget_counts_only_this_channels_own_facts():
    """Workspace-scope rows are visible but read-only from here, so they cannot spend a budget the
    model has no way to free."""
    rows = [_row(1, "chan fact"), _row(2, "x" * 400, scope="workspace")]
    db = _db(rows=rows, new_id=9)
    with patch.object(config, "memory_store_max_chars", 100):
        result = await execute_remember_fact(_ctx(db), {"content": "fits"})
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_an_update_does_not_have_to_fit_beside_the_row_it_replaces():
    """Otherwise a full store could never be consolidated: every merge would be refused by the
    very note it is merging."""
    rows = [_row(1, "a" * 45), _row(2, "b" * 45)]
    db = _db(rows=rows)
    with patch.object(config, "memory_store_max_chars", 95):
        result = await execute_update_fact(_ctx(db), {"id": 2, "content": "c" * 49})
    assert result["ok"] is True
    db.update_channel_fact_async.assert_awaited_once_with(2, "c" * 49)


@pytest.mark.asyncio
async def test_remember_truncates_at_the_configured_fact_cap():
    """Owner's ruling 2026-08-20: a fact's length limit is an operator setting, not a literal in
    memory_tools. Truncation itself is unchanged — it just reads the cap from config now."""
    db = _db(rows=[], new_id=11)
    with patch.object(config, "memory_fact_max_chars", 20):
        result = await execute_remember_fact(_ctx(db), {"content": "y" * 90})
    assert result["content"] == "y" * 20
    db.add_channel_memory_async.assert_awaited_once_with(
        CHANNEL, "y" * 20, scope="channel", author="U07PETER"
    )


def test_both_memory_limits_come_from_the_environment():
    """Neither limit may be hardcoded: MEMORY_FACT_MAX_CHARS defaults to 500 and
    MEMORY_STORE_MAX_CHARS to 2900 (the modal's textarea budget), and both follow the env."""
    from config import BotConfig

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("MEMORY_FACT_MAX_CHARS", None)
        os.environ.pop("MEMORY_STORE_MAX_CHARS", None)
        defaults = BotConfig()
        assert defaults.memory_fact_max_chars == 500
        assert defaults.memory_store_max_chars == 2900

        os.environ["MEMORY_FACT_MAX_CHARS"] = "42"
        os.environ["MEMORY_STORE_MAX_CHARS"] = "1200"
        tuned = BotConfig()
        assert tuned.memory_fact_max_chars == 42
        assert tuned.memory_store_max_chars == 1200


def test_the_store_budget_and_the_modal_textarea_are_one_number():
    """Drift guard. The tools refuse a write that would not fit the modal's box, so if these two
    ever disagree the store either hides notes from the person or refuses writes that would fit.
    The modal clamps at Slack's element limit, which is the one direction they may differ."""
    from slack_client.settings_modal import SettingsModal

    modal = SettingsModal(db=None)
    assert modal._MEMORY_TEXTAREA_MAX == min(config.memory_store_max_chars, 2900)
    with patch.object(config, "memory_store_max_chars", 1500):
        assert modal._MEMORY_TEXTAREA_MAX == 1500
    with patch.object(config, "memory_store_max_chars", 99999):
        assert modal._MEMORY_TEXTAREA_MAX == 2900   # Slack caps the element at 3000


@pytest.mark.asyncio
async def test_remember_empty_content_refused():
    db = _db()
    result = await execute_remember_fact(_ctx(db), {"content": "   "})
    assert result["ok"] is False and result["error"] == "bad_arguments"


@pytest.mark.asyncio
async def test_remember_in_a_dm_never_touches_the_channel_store():
    """A DM call is not refused any more — it is ROUTED. The channel store stays untouched."""
    db = _db()
    db.get_user_memory_async = AsyncMock(return_value=[])
    db.add_user_memory_async = AsyncMock(return_value=3)
    result = await execute_remember_fact(_ctx(db, is_dm=True, channel_id="D1"), {"content": "x"})
    assert result["ok"] is True
    db.get_channel_memory_async.assert_not_awaited()
    db.add_channel_memory_async.assert_not_awaited()


# --- update_fact ---

@pytest.mark.asyncio
async def test_update_happy_path():
    db = _db(rows=[_row(3, "old wording")])
    result = await execute_update_fact(_ctx(db), {"id": 3, "content": "new wording"})
    assert result == {"ok": True, "id": 3, "content": "new wording"}
    db.update_channel_fact_async.assert_awaited_once_with(3, "new wording")


@pytest.mark.asyncio
async def test_update_wrong_channel_id_not_found():
    """An id belonging to another channel isn't visible here → not_found, no write."""
    db = _db(rows=[_row(3, "mine")])  # visible set contains only id 3
    result = await execute_update_fact(_ctx(db), {"id": 99, "content": "x"})
    assert result["ok"] is False and result["error"] == "not_found"
    db.update_channel_fact_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_workspace_scope_refused():
    db = _db(rows=[_row(4, "shared", scope="workspace")])
    result = await execute_update_fact(_ctx(db), {"id": 4, "content": "x"})
    assert result["ok"] is False and result["error"] == "workspace_scope_readonly"
    db.update_channel_fact_async.assert_not_awaited()


# --- forget_fact ---

@pytest.mark.asyncio
async def test_forget_happy_path_returns_content():
    db = _db(rows=[_row(5, "obsolete fact")])
    result = await execute_forget_fact(_ctx(db), {"id": 5})
    assert result == {"ok": True, "id": 5, "forgot": "obsolete fact"}
    db.delete_channel_memory_async.assert_awaited_once_with(5)


@pytest.mark.asyncio
async def test_forget_not_found_and_bad_id():
    db = _db(rows=[])
    assert (await execute_forget_fact(_ctx(db), {"id": 8}))["error"] == "not_found"
    assert (await execute_forget_fact(_ctx(db), {"id": "abc"}))["error"] == "bad_arguments"
    db.delete_channel_memory_async.assert_not_awaited()


# --- injection rendering ---

@pytest.mark.asyncio
async def test_memory_rendering_id_prefixed_and_sorted_by_id():
    """Rendering must be [#id]-prefixed and deterministic (sorted by id, not updated_ts)."""
    from message_processor import channel_steering

    # updated_ts order (2 newest-first) differs from id order — id order must win
    rows = [_row(2, "beta", updated_ts="2026-07-09"), _row(1, "alpha", updated_ts="2026-07-01")]
    db = _db(rows=rows)
    snap = await channel_steering.load_snapshot(db, CHANNEL)
    assert "- [#1] alpha\n- [#2] beta" in snap.text
    # determinism: identical inputs → identical rendering
    assert (await channel_steering.load_snapshot(db, CHANNEL)).text == snap.text


# --- extractor fallback gating ---

def _processor_for_cleanup():
    from message_processor.thread_management import ThreadManagementMixin

    class _P(ThreadManagementMixin):
        def __init__(self):
            self.db = MagicMock()
            self.extract_called = False
        async def _async_extract_channel_memory(self, thread_state):
            self.extract_called = True
        def log_debug(self, *a, **k): pass
        def log_info(self, *a, **k): pass
        def log_warning(self, *a, **k): pass
        def log_error(self, *a, **k): pass

    return _P()


@pytest.mark.asyncio
async def test_extractor_skipped_when_fallback_off():
    p = _processor_for_cleanup()
    thread_state = MagicMock(current_model="gpt-5.5", messages=[])
    with patch.object(config, "enable_memory_extraction_fallback", False):
        try:
            await p._async_post_response_cleanup(thread_state, "C1:1.0")
        except Exception:
            pass  # token-cleanup half may fail on the bare mock; extraction gate already ran
    assert p.extract_called is False


@pytest.mark.asyncio
async def test_extractor_runs_when_fallback_on():
    p = _processor_for_cleanup()
    thread_state = MagicMock(current_model="gpt-5.5", messages=[])
    with patch.object(config, "enable_memory_extraction_fallback", True):
        try:
            await p._async_post_response_cleanup(thread_state, "C1:1.0")
        except Exception:
            pass
    assert p.extract_called is True


# --- registry gating ---

def test_register_memory_tools_registers_every_tool():
    from message_processor.tool_registry import SURFACE_CHANNEL

    registry = ToolRegistry()
    register_memory_tools(registry)
    names = {s["name"] for s in registry.schemas(surface=SURFACE_CHANNEL)}
    assert {"remember_fact", "update_fact", "forget_fact", "list_facts"} <= names


def test_registry_gating_on_enable_channel_memory():
    """SlackBot._build_tool_registry must include the tools iff ENABLE_CHANNEL_MEMORY."""
    from slack_client.base import SlackBot

    def build(flag):
        bot = SlackBot.__new__(SlackBot)
        with patch.object(config, "enable_channel_memory", flag), \
             patch.object(config, "enable_history_tools", False), \
             patch.object(config, "enable_reactions", False), \
             patch.object(config, "enable_search_tool", False), \
             patch.object(config, "enable_post_to_thread_tool", False), \
             patch.object(config, "enable_read_document_tool", False), \
             patch.object(config, "enable_people_tools", False), \
             patch.object(config, "enable_deep_research", False):
            with patch.object(SlackBot, "get_history_tools_for_openai", return_value=[], create=True):
                registry = SlackBot._build_tool_registry(bot)
            # The CHANNEL surface is the one ENABLE_CHANNEL_MEMORY governs. (The DM surface
            # answers to ENABLE_USER_MEMORY and a different store; test_user_memory.py owns that
            # half.) Read INSIDE the patch: the client now registers the memory tools
            # unconditionally and `channel_enabled` reads the flag per request, so a listing taken
            # after the patch lifts would answer to the real config rather than to `flag`.
            from message_processor.tool_registry import SURFACE_CHANNEL
            return {s["name"] for s in registry.schemas(surface=SURFACE_CHANNEL)}

    # Assert about the memory tools themselves, not the whole registry: tools registered
    # unconditionally by other features (F34's generate_image) are legitimately present in
    # both builds and say nothing about this gate.
    memory_tools = {"remember_fact", "update_fact", "forget_fact", "list_facts"}
    assert memory_tools <= build(True)
    assert memory_tools.isdisjoint(build(False))


# --- ToolContext plumbing + guidance ---

def test_tool_context_carries_user_id():
    from message_processor.handlers.text import TextHandlerMixin
    from message_processor.client_contract import Message

    class _P(TextHandlerMixin):
        def __init__(self): self.db = MagicMock()

    msg = Message(text="hi", user_id="U07PETER", channel_id=CHANNEL, thread_id="1.0",
                  metadata={"ts": "1.0"})
    ctx = _P()._build_tool_context(msg, client=MagicMock())
    assert ctx.user_id == "U07PETER"
    assert ctx.is_dm is False


def test_guidance_mentions_memory_tools():
    from message_processor.prompts import LOCAL_TOOLS_GUIDANCE
    for needle in ("remember_fact", "update_fact", "forget_fact", "[#id]", "forget"):
        assert needle in LOCAL_TOOLS_GUIDANCE
