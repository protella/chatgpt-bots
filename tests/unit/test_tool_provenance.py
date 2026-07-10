"""F7 — tool-use provenance.

Covers the pure helpers (gist/build/render/strip incl. the anti-shielding strip), the
DB layer (roundtrip, idempotent upsert, age sweep), the processor persistence seam
(enabled/disabled/empty/no-ts), and deterministic rebuild reinjection with the pinned
footer-strip → used-tools → reactions ordering, compaction-boundary skipping, config-off,
and silent DB failure.
"""
import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock

from base_client import Message
from config import config
from message_processor import tool_provenance as tp
from message_processor.thread_management import ThreadManagementMixin
from message_processor.utilities import MessageUtilitiesMixin
from message_processor.handlers.vision import VisionHandlerMixin
from thread_manager import AsyncThreadStateManager


# --------------------------------------------------------------------------- harness

class _Proc(ThreadManagementMixin, VisionHandlerMixin, MessageUtilitiesMixin):
    def __init__(self, db=None):
        self.db = db
        self.thread_manager = AsyncThreadStateManager(db=db)
        self.openai_client = None
        self.document_handler = None

    def log_info(self, *a, **k): pass
    log_debug = log_warning = log_error = log_info

    def _update_status(self, *a, **k): pass


def _hist(ts, text, sender="human", reactions=None):
    return Message(
        text=text, user_id="U1", channel_id="C1", thread_id="100.0", attachments=[],
        metadata={"ts": ts, "is_bot": sender == "self", "sender_type": sender,
                  "bot_name": None, "username": "Peter", "reactions": reactions},
    )


def _incoming(ts="200.0", text="latest"):
    return Message(text=text, user_id="U1", channel_id="C1", thread_id="100.0",
                   attachments=[], metadata={"ts": ts})


def _client(history):
    c = MagicMock()
    c.get_thread_history = AsyncMock(return_value=history)
    c.name = "slack"
    c.user_cache = {}
    c.bot_user_id = "UBOT"
    return c


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    from database import DatabaseManager
    db = DatabaseManager(platform="slack")
    yield db
    db.conn.close()


@pytest.fixture(autouse=True)
def _provenance_on(monkeypatch):
    monkeypatch.setattr(config, "enable_tool_provenance", True)


# ------------------------------------------------------------------ pure helpers

def test_gist_from_arguments_scalars_and_containers():
    assert tp.gist_from_arguments('{"limit": 50, "before": "abc"}') == "limit=50, before=abc"
    assert tp.gist_from_arguments('{"q": "hi", "opts": {"a": 1}}') == "q=hi, opts={1}"
    assert tp.gist_from_arguments('{"ids": [1, 2, 3]}') == "ids=[3]"
    assert tp.gist_from_arguments("{}") == ""
    assert tp.gist_from_arguments("not json") == ""
    assert tp.gist_from_arguments(None) == ""


def test_gist_is_length_capped():
    big = '{"prompt": "' + "x" * 500 + '"}'
    assert len(tp.gist_from_arguments(big)) <= tp.MAX_GIST_CHARS


def test_build_provenance_combines_local_then_external_and_caps():
    local = [{"name": "fetch_channel_history", "ok": True, "gist": "limit=50"},
             {"name": "react_to_message", "ok": True, "gist": "emoji=eyes"}]
    out = tp.build_provenance(local, ["web_search"])
    assert out == [
        {"tool_name": "fetch_channel_history", "gist": "limit=50"},
        {"tool_name": "react_to_message", "gist": "emoji=eyes"},
        {"tool_name": "web_search", "gist": ""},
    ]
    many = [{"name": f"t{i}", "ok": True, "gist": ""} for i in range(20)]
    assert len(tp.build_provenance(many, ["a", "b", "c"])) == tp.MAX_PROVENANCE_ENTRIES


def test_render_prefers_gists_then_degrades_to_names():
    short = [{"tool_name": "fetch_channel_history", "gist": "limit=50"},
             {"tool_name": "web_search", "gist": ""}]
    assert tp.render_used_tools_annotation(short) == \
        "[used tools: fetch_channel_history(limit=50), web_search]"
    # Long gists blow the budget → names only.
    long = [{"tool_name": f"tool_{i}", "gist": "x" * 40} for i in range(6)]
    rendered = tp.render_used_tools_annotation(long)
    assert "(" not in rendered and rendered.startswith("[used tools: tool_0, ")
    assert tp.render_used_tools_annotation([]) == ""


def test_strip_footer_and_anti_shielding():
    assert tp.strip_used_tools_footer("hi\n\n_Used Tools: web_search_") == "hi"
    # A trailing [used tools:]/[reactions:] annotation must NOT shield the footer.
    shielded = "hi\n\n_Used Tools: web_search_\n[used tools: web_search]\n[reactions: :eyes: x1]"
    assert tp.strip_used_tools_footer(shielded) == "hi\n[used tools: web_search]\n[reactions: :eyes: x1]"
    assert tp.strip_used_tools_footer("no footer here") == "no footer here"
    assert tp.strip_used_tools_footer(None) is None


# ------------------------------------------------------------------ DB layer

@pytest.mark.asyncio
async def test_db_save_and_get_roundtrip(temp_db):
    tools = [{"tool_name": "fetch_channel_history", "gist": "limit=50"}]
    await temp_db.save_tool_usage_async("C1", "101.0", "C1:100.0", tools)
    got = await temp_db.get_thread_tool_usage_async("C1:100.0")
    assert got == {"101.0": tools}


@pytest.mark.asyncio
async def test_db_save_is_idempotent_upsert(temp_db):
    await temp_db.save_tool_usage_async("C1", "101.0", "C1:100.0", [{"tool_name": "a", "gist": ""}])
    await temp_db.save_tool_usage_async("C1", "101.0", "C1:100.0", [{"tool_name": "b", "gist": ""}])
    got = await temp_db.get_thread_tool_usage_async("C1:100.0")
    assert got == {"101.0": [{"tool_name": "b", "gist": ""}]}  # last write wins, one row


@pytest.mark.asyncio
async def test_db_age_sweep_deletes_old_rows(temp_db):
    await temp_db.save_tool_usage_async("C1", "101.0", "C1:100.0", [{"tool_name": "a", "gist": ""}])
    # Backdate the row well past the retention window.
    temp_db.conn.execute(
        "UPDATE message_tool_usage SET created_at = datetime('now', '-200 days')")
    temp_db.conn.commit()
    temp_db.delete_old_tool_usage(days=90)
    assert await temp_db.get_thread_tool_usage_async("C1:100.0") == {}


@pytest.mark.asyncio
async def test_db_get_is_silent_on_missing_table(temp_db):
    temp_db.conn.execute("DROP TABLE message_tool_usage")
    temp_db.conn.commit()
    assert await temp_db.get_thread_tool_usage_async("C1:100.0") == {}  # no raise


# ------------------------------------------------------------------ persistence seam

@pytest.mark.asyncio
async def test_persist_schedules_save_when_enabled():
    proc = _Proc(db=MagicMock(save_tool_usage_async=AsyncMock()))
    prov = [{"tool_name": "web_search", "gist": ""}]
    proc._persist_tool_provenance("C1", "101.0", "C1:100.0", prov)
    await asyncio.sleep(0)
    proc.db.save_tool_usage_async.assert_awaited_once_with("C1", "101.0", "C1:100.0", prov)


@pytest.mark.asyncio
async def test_persist_noop_when_disabled_empty_or_no_ts(monkeypatch):
    proc = _Proc(db=MagicMock(save_tool_usage_async=AsyncMock()))
    prov = [{"tool_name": "web_search", "gist": ""}]
    # disabled
    monkeypatch.setattr(config, "enable_tool_provenance", False)
    proc._persist_tool_provenance("C1", "101.0", "C1:100.0", prov)
    monkeypatch.setattr(config, "enable_tool_provenance", True)
    # empty provenance (no tools ran)
    proc._persist_tool_provenance("C1", "101.0", "C1:100.0", [])
    # no ts (reaction-only / statusless turn)
    proc._persist_tool_provenance("C1", None, "C1:100.0", prov)
    await asyncio.sleep(0)
    proc.db.save_tool_usage_async.assert_not_awaited()


# ------------------------------------------------------------------ rebuild reinjection

@pytest.mark.asyncio
async def test_rebuild_annotates_matching_bot_message(temp_db):
    await temp_db.save_tool_usage_async(
        "C1", "101.0", "C1:100.0",
        [{"tool_name": "fetch_channel_history", "gist": "limit=50"}])
    proc = _Proc(db=temp_db)
    history = [_hist("101.0", "Here are the threads.", sender="self")]
    state = await proc._get_or_rebuild_thread_state(_incoming(), _client(history))
    bot_msg = next(m for m in state.messages if m["role"] == "assistant")
    assert "[used tools: fetch_channel_history(limit=50)]" in bot_msg["content"]


@pytest.mark.asyncio
async def test_rebuild_ordering_footer_stripped_then_used_then_reactions(temp_db):
    await temp_db.save_tool_usage_async(
        "C1", "101.0", "C1:100.0", [{"tool_name": "web_search", "gist": ""}])
    proc = _Proc(db=temp_db)
    history = [_hist("101.0", "Answer.\n\n_Used Tools: web_search_", sender="self",
                     reactions=[{"name": "eyes", "count": 1, "users": ["U9"]}])]
    state = await proc._get_or_rebuild_thread_state(_incoming(), _client(history))
    content = next(m for m in state.messages if m["role"] == "assistant")["content"]
    assert "_Used Tools:" not in content  # external chrome stripped, not in model context
    assert content == "Answer.\n[used tools: web_search]\n[reactions: :eyes: x1 (<@U9>)]"


@pytest.mark.asyncio
async def test_rebuild_is_deterministic_across_repeats(temp_db):
    await temp_db.save_tool_usage_async(
        "C1", "101.0", "C1:100.0", [{"tool_name": "web_search", "gist": ""}])
    history = [_hist("101.0", "Answer.", sender="self")]
    first = await _Proc(db=temp_db)._get_or_rebuild_thread_state(_incoming(), _client(history))
    second = await _Proc(db=temp_db)._get_or_rebuild_thread_state(_incoming(), _client(history))
    a = next(m for m in first.messages if m["role"] == "assistant")["content"]
    b = next(m for m in second.messages if m["role"] == "assistant")["content"]
    assert a == b


@pytest.mark.asyncio
async def test_rebuild_skips_rows_behind_compaction_boundary(temp_db):
    thread_key = "C1:100.0"
    temp_db.get_or_create_thread(thread_key, "C1")
    temp_db.save_thread_summary(thread_key, "Earlier stuff.", "101.5", refs=[])
    # A row for a message AT/behind the boundary — its message is excluded, never annotated.
    await temp_db.save_tool_usage_async("C1", "101.0", thread_key,
                                        [{"tool_name": "web_search", "gist": ""}])
    proc = _Proc(db=temp_db)
    history = [_hist("101.0", "old", sender="self"), _hist("102.0", "fresh tail")]
    state = await proc._get_or_rebuild_thread_state(_incoming(), _client(history))
    joined = " ".join(str(m.get("content")) for m in state.messages)
    assert "used tools" not in joined


@pytest.mark.asyncio
async def test_rebuild_config_off_leaves_content_untouched(temp_db, monkeypatch):
    await temp_db.save_tool_usage_async(
        "C1", "101.0", "C1:100.0", [{"tool_name": "web_search", "gist": ""}])
    monkeypatch.setattr(config, "enable_tool_provenance", False)
    proc = _Proc(db=temp_db)
    history = [_hist("101.0", "Answer.\n\n_Used Tools: web_search_", sender="self")]
    state = await proc._get_or_rebuild_thread_state(_incoming(), _client(history))
    content = next(m for m in state.messages if m["role"] == "assistant")["content"]
    assert "used tools:" not in content  # no F7 annotation
    assert "_Used Tools: web_search_" in content  # footer untouched at rebuild (as today)


@pytest.mark.asyncio
async def test_rebuild_survives_db_provenance_read_failure(temp_db, monkeypatch):
    proc = _Proc(db=temp_db)
    monkeypatch.setattr(temp_db, "get_thread_tool_usage_async",
                        AsyncMock(side_effect=RuntimeError("boom")))
    history = [_hist("101.0", "Answer.", sender="self")]
    state = await proc._get_or_rebuild_thread_state(_incoming(), _client(history))
    # No annotation, but the rebuild completes and the message is present.
    assert any("Answer." in str(m.get("content")) for m in state.messages)
