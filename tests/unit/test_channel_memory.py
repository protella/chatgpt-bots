"""Phase 9 — per-channel memory (context-injection read + post-response extraction write).

Covers: channel_memory CRUD (sync + async), scope partitioning (a channel never sees another
channel's private rows; workspace rows shared), CHANNEL MEMORY system-prompt injection + the
_build_channel_memory_text helper, the post-response extraction logic (add / update / none / cap
eviction / flag-off / missing-exchange), _content_to_text flattening, and extract_memory's
defensive JSON parsing. All with stubbed I/O — no live bot, no legacy suite.
"""
from __future__ import annotations

import sqlite3
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import config
from database import DatabaseManager
from message_processor.thread_management import ThreadManagementMixin
from message_processor.utilities import MessageUtilitiesMixin
from openai_client.api import responses as responses_api


# --------------------------------------------------------------------------- DB CRUD + partitioning

class TestChannelMemoryDB:
    @pytest.fixture
    def temp_db(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("os.makedirs"):
                db = DatabaseManager("test")
                db.db_path = f"{tmpdir}/test.db"
                if getattr(db, "conn", None):
                    db.conn.close()
                db.conn = sqlite3.connect(db.db_path, check_same_thread=False, isolation_level=None)
                db.conn.row_factory = sqlite3.Row
                db.conn.execute("PRAGMA journal_mode=WAL")
                db.init_schema()
                yield db
                if getattr(db, "conn", None):
                    db.conn.close()

    def test_add_and_get(self, temp_db):
        mid = temp_db.add_channel_memory("C1", "they prefer terse answers")
        assert isinstance(mid, int)
        rows = temp_db.get_channel_memory("C1")
        assert len(rows) == 1
        assert rows[0]["content"] == "they prefer terse answers"
        assert rows[0]["scope"] == "channel"

    def test_unset_channel_returns_empty(self, temp_db):
        assert temp_db.get_channel_memory("C_NONE") == []

    def test_update(self, temp_db):
        mid = temp_db.add_channel_memory("C1", "old fact")
        temp_db.update_channel_memory(mid, "new fact")
        rows = temp_db.get_channel_memory("C1")
        assert rows[0]["content"] == "new fact"

    def test_delete(self, temp_db):
        mid = temp_db.add_channel_memory("C1", "fact")
        temp_db.delete_channel_memory(mid)
        assert temp_db.get_channel_memory("C1") == []

    def test_channel_scope_is_private_to_its_channel(self, temp_db):
        temp_db.add_channel_memory("C_A", "A-only fact")
        temp_db.add_channel_memory("C_B", "B-only fact")
        a_rows = [r["content"] for r in temp_db.get_channel_memory("C_A")]
        b_rows = [r["content"] for r in temp_db.get_channel_memory("C_B")]
        assert "A-only fact" in a_rows and "B-only fact" not in a_rows
        assert "B-only fact" in b_rows and "A-only fact" not in b_rows

    def test_workspace_scope_is_shared_across_channels(self, temp_db):
        temp_db.add_channel_memory("C_WS", "shared workspace fact", scope="workspace")
        temp_db.add_channel_memory("C_A", "A private")
        a_contents = [r["content"] for r in temp_db.get_channel_memory("C_A")]
        # C_A sees the workspace row plus its own, but not another channel's private row
        assert "shared workspace fact" in a_contents
        assert "A private" in a_contents
        other = [r["content"] for r in temp_db.get_channel_memory("C_OTHER")]
        assert other == ["shared workspace fact"]

    async def test_async_roundtrip(self, temp_db):
        await temp_db.add_channel_memory_async("C9", "async fact")
        rows = await temp_db.get_channel_memory_async("C9")
        assert rows[0]["content"] == "async fact"
        await temp_db.update_channel_memory_async(rows[0]["id"], "async fact v2")
        rows2 = await temp_db.get_channel_memory_async("C9")
        assert rows2[0]["content"] == "async fact v2"
        await temp_db.delete_channel_memory_async(rows[0]["id"])
        assert await temp_db.get_channel_memory_async("C9") == []


# --------------------------------------------------------------------------- prompt injection

def _utils():
    return MessageUtilitiesMixin.__new__(type("P", (MessageUtilitiesMixin,), {}))


def test_system_prompt_injects_memory_block():
    proc = _utils()
    out = proc._get_system_prompt(MagicMock(), channel_memory="- they deploy via #ops\n- Pat owns billing")
    assert "CHANNEL MEMORY" in out
    assert "they deploy via #ops" in out


def test_system_prompt_no_memory_block_when_absent():
    proc = _utils()
    out = proc._get_system_prompt(MagicMock())
    assert "CHANNEL MEMORY" not in out


async def test_build_channel_memory_text_formats_rows():
    proc = _utils()
    proc.db = MagicMock()
    proc.db.get_channel_memory_async = AsyncMock(
        return_value=[{"id": 1, "content": "fact one"}, {"id": 2, "content": "fact two"}]
    )
    with patch.object(config, "enable_channel_memory", True):
        text = await proc._build_channel_memory_text("C1")
    # Phase C: [#id] prefixes so the model can target update_fact/forget_fact
    assert text == "- [#1] fact one\n- [#2] fact two"


async def test_build_channel_memory_text_none_when_empty():
    proc = _utils()
    proc.db = MagicMock()
    proc.db.get_channel_memory_async = AsyncMock(return_value=[])
    with patch.object(config, "enable_channel_memory", True):
        assert await proc._build_channel_memory_text("C1") is None


async def test_build_channel_memory_text_none_when_flag_off():
    proc = _utils()
    proc.db = MagicMock()
    proc.db.get_channel_memory_async = AsyncMock(return_value=[{"content": "fact"}])
    with patch.object(config, "enable_channel_memory", False):
        assert await proc._build_channel_memory_text("C1") is None


async def test_build_channel_memory_text_none_without_db():
    proc = _utils()
    proc.db = None
    with patch.object(config, "enable_channel_memory", True):
        assert await proc._build_channel_memory_text("C1") is None


# --------------------------------------------------------------------------- post-response extraction

def _proc(decision):
    proc = ThreadManagementMixin.__new__(type("P", (ThreadManagementMixin,), {}))
    proc.db = MagicMock()
    proc.db.get_channel_memory_async = AsyncMock(return_value=[])
    proc.db.add_channel_memory_async = AsyncMock()
    proc.db.update_channel_memory_async = AsyncMock()
    proc.db.delete_channel_memory_async = AsyncMock()
    proc.openai_client = MagicMock()
    proc.openai_client.extract_memory = AsyncMock(return_value=decision)
    proc.log_info = MagicMock()
    proc.log_debug = MagicMock()
    return proc


def _state():
    return SimpleNamespace(
        channel_id="C1",
        messages=[
            {"role": "user", "content": "we always ship from the release branch"},
            {"role": "assistant", "content": "Got it."},
        ],
    )


def test_content_to_text_variants():
    f = ThreadManagementMixin._content_to_text
    assert f("hello") == "hello"
    assert f([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]) == "a b"
    assert f(["x", "y"]) == "x y"
    assert f(None) == ""


async def test_extraction_add_writes_row():
    proc = _proc({"action": "add", "content": "ships from release branch"})
    with patch.object(config, "enable_channel_memory", True), patch.object(config, "memory_max_rows", 25):
        await proc._async_extract_channel_memory(_state())
    proc.db.add_channel_memory_async.assert_called_once()
    args, kwargs = proc.db.add_channel_memory_async.call_args
    assert args[0] == "C1" and args[1] == "ships from release branch"
    assert kwargs.get("scope") == "channel"


async def test_extraction_none_writes_nothing():
    proc = _proc({"action": "none"})
    with patch.object(config, "enable_channel_memory", True):
        await proc._async_extract_channel_memory(_state())
    proc.db.add_channel_memory_async.assert_not_called()
    proc.db.update_channel_memory_async.assert_not_called()


async def test_extraction_update_updates_row():
    proc = _proc({"action": "update", "id": 7, "content": "revised"})
    with patch.object(config, "enable_channel_memory", True):
        await proc._async_extract_channel_memory(_state())
    proc.db.update_channel_memory_async.assert_called_once_with(7, "revised")


async def test_extraction_none_decision_object_safe():
    proc = _proc(None)  # extractor returned None
    with patch.object(config, "enable_channel_memory", True):
        await proc._async_extract_channel_memory(_state())
    proc.db.add_channel_memory_async.assert_not_called()


async def test_extraction_cap_evicts_oldest():
    proc = _proc({"action": "add", "content": "newest fact"})
    proc.db.get_channel_memory_async = AsyncMock(return_value=[
        {"id": 1, "content": "old", "scope": "channel", "updated_ts": "2026-06-01"},
        {"id": 2, "content": "mid", "scope": "channel", "updated_ts": "2026-06-02"},
        {"id": 3, "content": "recent", "scope": "channel", "updated_ts": "2026-06-03"},
    ])
    with patch.object(config, "enable_channel_memory", True), patch.object(config, "memory_max_rows", 3):
        await proc._async_extract_channel_memory(_state())
    proc.db.delete_channel_memory_async.assert_called_once_with(1)  # oldest evicted
    proc.db.add_channel_memory_async.assert_called_once()


async def test_extraction_flag_off_short_circuits():
    proc = _proc({"action": "add", "content": "x"})
    with patch.object(config, "enable_channel_memory", False):
        await proc._async_extract_channel_memory(_state())
    proc.openai_client.extract_memory.assert_not_called()


async def test_extraction_requires_full_exchange():
    proc = _proc({"action": "add", "content": "x"})
    state = SimpleNamespace(channel_id="C1", messages=[{"role": "user", "content": "hi"}])  # no assistant turn
    with patch.object(config, "enable_channel_memory", True):
        await proc._async_extract_channel_memory(state)
    proc.openai_client.extract_memory.assert_not_called()


# --------------------------------------------------------------------------- extract_memory JSON parsing

class _FakeOAI:
    def __init__(self, text):
        self._text = text
        self.client = MagicMock()
        self.log_warning = MagicMock()

    async def _safe_api_call(self, *a, **k):
        content = SimpleNamespace(text=self._text)
        item = SimpleNamespace(content=[content])
        return SimpleNamespace(output=[item])


async def test_extract_memory_parses_add():
    out = await responses_api.extract_memory(_FakeOAI('{"action":"add","content":"a fact"}'), "exchange")
    assert out == {"action": "add", "content": "a fact"}


async def test_extract_memory_parses_prose_wrapped_json():
    out = await responses_api.extract_memory(
        _FakeOAI('Sure, here:\n```json\n{"action":"update","id":4,"content":"b"}\n```'), "exchange")
    assert out == {"action": "update", "id": 4, "content": "b"}


async def test_extract_memory_malformed_returns_none():
    out = await responses_api.extract_memory(_FakeOAI("not json at all"), "exchange")
    assert out == {"action": "none"}
