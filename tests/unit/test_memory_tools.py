"""Phase C — model-invoked channel-memory tools.

Covers: the three executors (happy paths, cap-hit with oldest-3 listing,
wrong-channel not_found, workspace-scope write refusal, DM refusal), author
attribution, [#id]-prefixed deterministic injection rendering, extractor
fallback gating, registry gating on ENABLE_CHANNEL_MEMORY, and guidance text.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from config import config
from message_processor.memory_tools import (
    execute_forget_fact,
    execute_remember_fact,
    execute_update_fact,
    get_forget_fact_schema,
    get_remember_fact_schema,
    get_update_fact_schema,
    register_memory_tools,
)
from tool_registry import ToolContext, ToolRegistry

CHANNEL = "C04QDHE8W8M"


def _row(id, content, scope="channel", channel_id=CHANNEL, updated_ts="2026-07-01"):
    return {"id": id, "channel_id": channel_id, "scope": scope, "content": content,
            "author": None, "created_ts": updated_ts, "updated_ts": updated_ts}


def _db(rows=None, new_id=42):
    db = MagicMock()
    db.get_channel_memory_async = AsyncMock(return_value=list(rows or []))
    db.add_channel_memory_async = AsyncMock(return_value=new_id)
    db.update_channel_memory_async = AsyncMock()
    db.delete_channel_memory_async = AsyncMock()
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
async def test_remember_cap_hit_lists_oldest_three():
    rows = [_row(i, f"fact {i}", updated_ts=f"2026-06-{i:02d}") for i in range(1, 6)]
    db = _db(rows=rows)
    with patch.object(config, "memory_max_rows", 5):
        result = await execute_remember_fact(_ctx(db), {"content": "one more"})
    assert result["ok"] is False
    assert result["error"] == "memory_full"
    assert result["hint"] == "forget or update something"
    assert [r["id"] for r in result["oldest"]] == [1, 2, 3]
    db.add_channel_memory_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_remember_cap_counts_only_channel_scope():
    """Workspace-scope rows are visible but must not consume the channel's cap."""
    rows = [_row(1, "chan fact"), _row(2, "shared fact", scope="workspace")]
    db = _db(rows=rows, new_id=9)
    with patch.object(config, "memory_max_rows", 2):
        result = await execute_remember_fact(_ctx(db), {"content": "fits"})
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_remember_empty_content_refused():
    db = _db()
    result = await execute_remember_fact(_ctx(db), {"content": "   "})
    assert result["ok"] is False and result["error"] == "bad_arguments"


@pytest.mark.asyncio
async def test_remember_dm_refused():
    db = _db()
    result = await execute_remember_fact(_ctx(db, is_dm=True, channel_id="D1"), {"content": "x"})
    assert result == {"ok": False, "error": "memory_is_channel_only",
                      "message": "Channel memory is not available in DMs."}
    db.get_channel_memory_async.assert_not_awaited()


# --- update_fact ---

@pytest.mark.asyncio
async def test_update_happy_path():
    db = _db(rows=[_row(3, "old wording")])
    result = await execute_update_fact(_ctx(db), {"id": 3, "content": "new wording"})
    assert result == {"ok": True, "id": 3, "content": "new wording"}
    db.update_channel_memory_async.assert_awaited_once_with(3, "new wording")


@pytest.mark.asyncio
async def test_update_wrong_channel_id_not_found():
    """An id belonging to another channel isn't visible here → not_found, no write."""
    db = _db(rows=[_row(3, "mine")])  # visible set contains only id 3
    result = await execute_update_fact(_ctx(db), {"id": 99, "content": "x"})
    assert result["ok"] is False and result["error"] == "not_found"
    db.update_channel_memory_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_workspace_scope_refused():
    db = _db(rows=[_row(4, "shared", scope="workspace")])
    result = await execute_update_fact(_ctx(db), {"id": 4, "content": "x"})
    assert result["ok"] is False and result["error"] == "workspace_scope_readonly"
    db.update_channel_memory_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_update_dm_refused():
    result = await execute_update_fact(_ctx(_db(), is_dm=True), {"id": 1, "content": "x"})
    assert result["error"] == "memory_is_channel_only"


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


@pytest.mark.asyncio
async def test_forget_dm_refused():
    result = await execute_forget_fact(_ctx(_db(), is_dm=True), {"id": 1})
    assert result["error"] == "memory_is_channel_only"


# --- injection rendering ---

@pytest.mark.asyncio
async def test_memory_rendering_id_prefixed_and_sorted_by_id():
    """Rendering must be [#id]-prefixed and deterministic (sorted by id, not updated_ts)."""
    from message_processor.utilities import MessageUtilitiesMixin

    class _P(MessageUtilitiesMixin):
        def __init__(self, db): self.db = db
        def log_debug(self, *a, **k): pass

    # updated_ts order (2 newest-first) differs from id order — id order must win
    rows = [_row(2, "beta", updated_ts="2026-07-09"), _row(1, "alpha", updated_ts="2026-07-01")]
    p = _P(_db(rows=rows))
    with patch.object(config, "enable_channel_memory", True):
        text = await p._build_channel_memory_text(CHANNEL)
    assert text == "- [#1] alpha\n- [#2] beta"
    # determinism: identical inputs → identical rendering
    with patch.object(config, "enable_channel_memory", True):
        assert await p._build_channel_memory_text(CHANNEL) == text


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

def test_register_memory_tools_registers_all_three():
    registry = ToolRegistry()
    register_memory_tools(registry)
    names = {s["name"] for s in registry.schemas()}
    assert {"remember_fact", "update_fact", "forget_fact"} <= names


def test_registry_gating_on_enable_channel_memory():
    """SlackBot._build_tool_registry must include the tools iff ENABLE_CHANNEL_MEMORY."""
    from slack_client.base import SlackBot

    def build(flag):
        bot = SlackBot.__new__(SlackBot)
        with patch.object(config, "enable_channel_memory", flag), \
             patch.object(config, "enable_history_tools", False), \
             patch.object(config, "enable_reactions", False), \
             patch.object(config, "enable_search_tool", False), \
             patch.object(config, "enable_read_document_tool", False):
            with patch.object(SlackBot, "get_history_tools_for_openai", return_value=[], create=True):
                registry = SlackBot._build_tool_registry(bot)
        return {s["name"] for s in registry.schemas()}

    assert {"remember_fact", "update_fact", "forget_fact"} <= build(True)
    assert build(False) == set()


# --- ToolContext plumbing + guidance ---

def test_tool_context_carries_user_id():
    from message_processor.handlers.text import TextHandlerMixin
    from base_client import Message

    class _P(TextHandlerMixin):
        def __init__(self): self.db = MagicMock()

    msg = Message(text="hi", user_id="U07PETER", channel_id=CHANNEL, thread_id="1.0",
                  metadata={"ts": "1.0"})
    ctx = _P()._build_tool_context(msg, client=MagicMock())
    assert ctx.user_id == "U07PETER"
    assert ctx.is_dm is False


def test_guidance_mentions_memory_tools():
    from prompts import LOCAL_TOOLS_GUIDANCE
    for needle in ("remember_fact", "update_fact", "forget_fact", "[#id]", "forget"):
        assert needle in LOCAL_TOOLS_GUIDANCE
