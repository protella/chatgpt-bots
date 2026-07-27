"""Phase 9 — per-channel memory (context-injection read + post-response extraction write).

Covers: channel_memory CRUD (sync + async), scope partitioning (a channel never sees another
channel's private rows; workspace rows shared), CHANNEL STEERING system-prompt injection and the
snapshot that feeds it, the post-response extraction logic (add / update / none / cap eviction /
flag-off / missing-exchange), _content_to_text flattening, and extract_memory's defensive JSON
parsing. All with stubbed I/O — no live bot, no legacy suite.
"""
from __future__ import annotations

import sqlite3
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import config
from database import DatabaseManager
from message_processor import channel_steering
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
    out = proc._get_system_prompt(MagicMock(), channel_steering="- they deploy via #ops\n- Pat owns billing")
    assert "CHANNEL STEERING" in out
    assert "they deploy via #ops" in out


def test_system_prompt_no_memory_block_when_absent():
    proc = _utils()
    out = proc._get_system_prompt(MagicMock())
    assert "CHANNEL STEERING" not in out


def test_every_system_prompt_call_site_passes_channel_memory():
    """The injection above is worthless if a caller forgets the kwarg.

    That is exactly what happened: base.py built the memory text into
    `thread_state.system_prompt`, and NO code path ever read that attribute. The handlers build
    the whole prompt themselves and called `_get_system_prompt` without the kwarg, so durable
    facts reached no ordinary model call at all — not the opening turn, not any turn. Nothing
    failed loudly, because the parameter just defaults to None.

    A source-level assertion, since the defect is a missing argument at a call site rather than
    wrong behaviour inside the function. It is a backstop and not the real protection: it only
    proves the keyword is PRESENT, never that the value reaches the API. The behavioural
    API-spy tests below are what prove the snapshot propagates.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    # Only OUR source. rglob over the whole repo walked hidden trees and any vendored/generated
    # directory that happened to be sitting there — parsing files nobody here wrote, and paying
    # for it on every run.
    skip_dirs = {"tests", "venv", "node_modules", "build", "dist", "__pycache__",
                 "site-packages"}
    missing = []
    for path in root.rglob("*.py"):
        parts = set(path.relative_to(root).parts)
        if parts & skip_dirs or any(p.startswith(".") for p in path.relative_to(root).parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute) and fn.attr == "_get_system_prompt"):
                continue
            if not any(kw.arg == "channel_steering" for kw in node.keywords):
                missing.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not missing, "_get_system_prompt called without channel_steering at: " + ", ".join(missing)


def _steering_db(rows, policy=None):
    """A db stub whose two steering reads are the only ones load_snapshot performs."""
    db = MagicMock()
    db.get_channel_memory_async = AsyncMock(return_value=rows)
    db.get_channel_policy_async = AsyncMock(return_value=policy)
    return db


async def test_snapshot_formats_fact_rows():
    db = _steering_db([{"id": 1, "content": "fact one"}, {"id": 2, "content": "fact two"}])
    snap = await channel_steering.load_snapshot(db, "C1")
    # [#id] prefixes so the model can target update_fact/forget_fact, under a heading that says
    # these are background rather than instructions.
    assert channel_steering.CHANNEL_FACT_HEADING in snap.text
    assert "- [#1] fact one\n- [#2] fact two" in snap.text


async def test_snapshot_empty_when_no_rows():
    snap = await channel_steering.load_snapshot(_steering_db([]), "C1")
    assert snap.text is None and snap.is_empty


async def test_snapshot_drops_facts_when_memory_disabled():
    db = _steering_db([{"id": 1, "content": "fact", "scope": "channel"}])
    snap = await channel_steering.load_snapshot(db, "C1", memory_enabled=False)
    assert snap.text is None


async def test_snapshot_empty_without_db():
    snap = await channel_steering.load_snapshot(None, "C1")
    assert snap.text is None


# ------------------------------------------------------------- what the API actually receives
#
# The call-site guard above is a source-level check, and it passed for months while the block
# reached no model call at all: base.py built the memory text into `thread_state.system_prompt`,
# which nothing ever read, and the handlers built the real prompt without it. Only a spy on the
# prompt handed to the API can tell you the facts are actually in front of the model.

MEMORY_ROWS = [{"id": 3, "content": "deploys go out from #ops on Thursdays"}]
MEMORY_FACT = "deploys go out from #ops on Thursdays"


class _SpyOpenAI:
    """Records the system prompt handed to every API call this turn."""

    def __init__(self, text="ok", stream_error=None):
        self.prompts = []
        self._text = text
        self._stream_error = stream_error
        self.stream_attempts = 0

    async def create_text_response(self, messages=None, system_prompt=None, **kw):
        self.prompts.append(system_prompt)
        return self._text

    async def _create_text_response_with_timeout(self, **kw):
        return await self.create_text_response(**kw)

    async def create_streaming_response(self, messages=None, stream_callback=None,
                                        system_prompt=None, **kw):
        self.prompts.append(system_prompt)
        self.stream_attempts += 1
        if self._stream_error is not None and self.stream_attempts == 1:
            raise self._stream_error
        await stream_callback(self._text)
        await stream_callback(None)
        return self._text


class _FakeSlack:
    """Just enough Slack for a no-tools turn. `name` matters: the real _get_system_prompt
    branches on it to pick the Slack prompt."""

    name = "Slack"
    MAX_MESSAGE_LENGTH = 3900

    def __init__(self):
        self.posted = []
        self.channel_pulse = None

    def supports_streaming(self):
        return True

    def supports_native_streaming(self):
        return False

    def get_streaming_config(self):
        return {"update_interval": 0.0, "buffer_size": 1, "min_interval": 0.0}

    def format_text(self, t):
        return t

    async def send_message_get_ts(self, channel, thread, text, lease=None, surface=None):
        self.posted.append(text)
        return {"success": True, "ts": f"ts{len(self.posted)}"}

    async def send_message(self, channel, thread, text, **kw):
        self.posted.append(text)
        return f"ts{len(self.posted)}"

    async def update_message(self, channel, ts, text):
        return True

    async def update_message_streaming(self, channel, ts, text, lease=None, surface=None):
        return {"success": True}

    async def delete_message(self, channel, ts):
        return True

    async def set_assistant_status(self, channel, thread, status=""):
        return None

    def _record_own_reply_pulse(self, *a, **k):
        pass


def _thread_config(**over):
    cfg = {"model": "gpt-5.6-sol", "temperature": 1.0, "max_tokens": 4096,
           "enable_streaming": True, "enable_web_search": False,
           "enable_code_interpreter": False, "reasoning_effort": "low",
           "verbosity": "medium", "custom_instructions": None}
    cfg.update(over)
    return cfg


def _spy_processor(openai):
    """A real MessageProcessor wired to stub I/O — but keeping the REAL _get_system_prompt,
    which is the thing under test."""
    from message_processor.base import MessageProcessor

    with patch("message_processor.base.AsyncThreadStateManager"), \
         patch("message_processor.base.OpenAIClient"):
        p = MessageProcessor()
    p.openai_client = openai
    p.db = MagicMock()
    p.db.get_channel_memory_async = AsyncMock(return_value=MEMORY_ROWS)
    p.db.get_channel_policy_async = AsyncMock(return_value=None)

    async def _passthru(m, *a, **k):
        return m

    async def _none(*a, **k):
        return None

    p._add_message_with_token_management = MagicMock()
    p._inject_image_analyses = _passthru
    p._pre_trim_messages_for_api = _passthru
    p._build_participant_roster = MagicMock(return_value="")
    p._build_suffix_context = MagicMock(return_value="")
    p._build_pulse_envelope = MagicMock(return_value=None)
    p._build_tools_array = MagicMock(return_value=[])            # no tools -> plain API call
    p._materialize_request_tools = MagicMock(return_value=(None, {}, False, ""))
    p._persist_tool_provenance = MagicMock()

    def _discard(coro, *a, **k):
        # Detached cleanup is out of scope here; close the coroutine rather than leaving an
        # "was never awaited" warning that would mask a real one.
        if hasattr(coro, "close"):
            coro.close()

    p._schedule_async_call = MagicMock(side_effect=_discard)
    p._update_status = MagicMock()
    p._build_channel_info = _none
    p._build_channel_summary_block = _none
    p._async_post_response_cleanup = _none
    p._drop_dead_containers = _none
    p._resolve_ci_container = _none
    return p


def _spy_state(channel="C1", thread="10.0"):
    return SimpleNamespace(
        messages=[{"role": "user", "content": "hi"}],
        channel_id=channel, thread_ts=thread, current_model="gpt-5.6-sol",
        config_overrides={}, has_summary_head=False, channel_directives=None,
        record_usage=MagicMock(), last_usage=None,
    )


def _msg(channel="C1", thread="10.0"):
    return SimpleNamespace(
        channel_id=channel, thread_id=thread, user_id="U1", text="hi",
        attachments=None, metadata={"ts": "10.0"},
    )


async def test_non_streaming_turn_puts_channel_memory_in_front_of_the_model():
    openai = _SpyOpenAI()
    p = _spy_processor(openai)
    with patch.object(config, "enable_channel_memory", True), \
         patch.object(config, "get_thread_config_async",
                      side_effect=AsyncMock(return_value=_thread_config(enable_streaming=False))):
        await p._handle_text_response(
            "hi", _spy_state(), _FakeSlack(), _msg(),
            channel_steering_text=(await channel_steering.load_snapshot(p.db, "C1")).text)

    assert openai.prompts, "expected the non-streaming path to reach the API"
    assert "CHANNEL STEERING" in openai.prompts[0]
    assert MEMORY_FACT in openai.prompts[0]


async def test_streaming_turn_puts_channel_memory_in_front_of_the_model():
    openai = _SpyOpenAI()
    p = _spy_processor(openai)
    with patch.object(config, "enable_channel_memory", True), \
         patch.object(config, "get_thread_config_async",
                      side_effect=AsyncMock(return_value=_thread_config())):
        await p._handle_streaming_text_response(
            "hi", _spy_state(), _FakeSlack(), _msg(),
            channel_steering_text=(await channel_steering.load_snapshot(p.db, "C1")).text)

    assert openai.prompts, "expected the streaming path to reach the API"
    assert "CHANNEL STEERING" in openai.prompts[0]
    assert MEMORY_FACT in openai.prompts[0]


async def test_a_retry_reuses_the_turns_snapshot_instead_of_refetching():
    """One logical turn, one read. Retries re-enter the handlers (this one fails the stream and
    falls back to non-streaming), and a per-handler fetch would hit the memory table again —
    and could hand two attempts of the same turn different facts."""
    openai = _SpyOpenAI(stream_error=RuntimeError("stream died"))
    p = _spy_processor(openai)
    with patch.object(config, "enable_channel_memory", True), \
         patch.object(config, "get_thread_config_async",
                      side_effect=AsyncMock(return_value=_thread_config())):
        # the one read base.py performs
        snapshot = (await channel_steering.load_snapshot(p.db, "C1")).text
        await p._handle_streaming_text_response(
            "hi", _spy_state(), _FakeSlack(), _msg(), channel_steering_text=snapshot)

    assert len(openai.prompts) == 2, "expected the failed stream to fall back to non-streaming"
    for prompt in openai.prompts:
        assert MEMORY_FACT in prompt
    p.db.get_channel_memory_async.assert_awaited_once()   # the snapshot, and nothing more


class _StopHere(Exception):
    """Ends a turn at a known point so a test never has to catch `Exception` to get there."""


async def test_base_reads_the_memory_once_and_passes_it_to_the_handler():
    """The other half of the contract: base.py owns the read. It used to build the block into
    `thread_state.system_prompt`, which no code path ever read."""
    from message_processor.base import MessageProcessor

    with patch("message_processor.base.AsyncThreadStateManager"), \
         patch("message_processor.base.OpenAIClient"):
        p = MessageProcessor()

    state = SimpleNamespace(had_timeout=False, messages=[], thread_ts="10.0", channel_id="C1",
                            root_author=("U1", "human"), config_overrides={}, participants={},
                            current_model=None, has_trimmed_messages=False,
                            channel_directives=None)
    p.db = MagicMock()
    p.db.get_channel_memory_async = AsyncMock(return_value=MEMORY_ROWS)
    p.db.get_channel_policy_async = AsyncMock(return_value=None)
    p.thread_manager.acquire_thread_lock = AsyncMock(return_value=True)
    p.thread_manager.release_thread_lock = AsyncMock()
    p.thread_manager._token_counter.count_thread_tokens = MagicMock(return_value=0)
    p.thread_manager._token_counter.count_message_tokens = MagicMock(return_value=0)

    async def _state_for(*a, **k):
        return state

    p._get_or_rebuild_thread_state = _state_for
    # A SENTINEL, not a return value: the handler records its kwargs and then stops the turn
    # right there. Letting it return and catching whatever came next would also have swallowed
    # a regression anywhere downstream — this test would keep passing while base.py broke.
    p._handle_text_response = AsyncMock(side_effect=_StopHere())
    p._process_attachments = AsyncMock(return_value=([], [], []))

    from base_client import Message
    msg = Message(text="hi", user_id="U1", channel_id="C1", thread_id="10.0",
                  metadata={"ts": "10.0"})

    client = MagicMock()
    client.send_message = AsyncMock()

    with patch.object(config, "enable_channel_memory", True):
        # No try/except here: the sentinel ends the turn INSIDE the handler and base.py's own
        # error path returns a Response rather than raising. The old broad catch also hid
        # anything that broke on the way TO the handler, which is the half this test is about —
        # now that shows up as `assert_awaited` failing instead of as a pass.
        await p.process_message(msg, client, None)

    p.db.get_channel_memory_async.assert_awaited_once()
    p._handle_text_response.assert_awaited()
    passed = p._handle_text_response.await_args.kwargs.get("channel_steering_text")
    assert passed and MEMORY_FACT in passed, (
        "base.py must hand its one memory snapshot to the handler — the handlers build the "
        "real prompt and no longer read it themselves")


async def test_the_timeout_retry_carries_the_same_snapshot():
    """base.py's own retry path, which is separate from the handler-level retries: a text turn
    that times out is re-dispatched with a shorter timeout. It re-enters the handler, so without
    the snapshot riding along the second attempt would either re-read the table or run with no
    memory at all — and the user would see a reply that forgot facts the first attempt had."""
    from base_client import Response
    from message_processor.base import MessageProcessor

    with patch("message_processor.base.AsyncThreadStateManager"), \
         patch("message_processor.base.OpenAIClient"):
        p = MessageProcessor()

    state = SimpleNamespace(had_timeout=False, messages=[], thread_ts="10.0", channel_id="C1",
                            root_author=("U1", "human"), config_overrides={}, participants={},
                            current_model=None, has_trimmed_messages=False,
                            channel_directives=None)
    p.db = MagicMock()
    p.db.get_channel_memory_async = AsyncMock(return_value=MEMORY_ROWS)
    p.db.get_channel_policy_async = AsyncMock(return_value=None)
    p.thread_manager.acquire_thread_lock = AsyncMock(return_value=True)
    p.thread_manager.release_thread_lock = AsyncMock()
    p.thread_manager._token_counter.count_thread_tokens = MagicMock(return_value=0)
    p.thread_manager._token_counter.count_message_tokens = MagicMock(return_value=0)

    async def _state_for(*a, **k):
        return state

    p._get_or_rebuild_thread_state = _state_for
    p._process_attachments = AsyncMock(return_value=([], [], []))

    timeout = TimeoutError("upstream")
    timeout.operation_type = "text_normal"          # the only kind base.py retries
    p._handle_text_response = AsyncMock(
        side_effect=[timeout, Response(type="text", content="ok")])

    from base_client import Message
    msg = Message(text="hi", user_id="U1", channel_id="C1", thread_id="10.0",
                  metadata={"ts": "10.0"})
    client = MagicMock()
    client.send_message = AsyncMock()

    with patch.object(config, "enable_channel_memory", True):
        response = await p.process_message(msg, client, None)

    assert response.content == "ok"
    assert p._handle_text_response.await_count == 2
    first, retry = [c.kwargs.get("channel_steering_text")
                    for c in p._handle_text_response.await_args_list]
    assert first and MEMORY_FACT in first
    assert retry == first                                  # the SAME snapshot, not a re-read
    p.db.get_channel_memory_async.assert_awaited_once()


# --------------------------------------------------------------------------- post-response extraction

def _proc(decision):
    proc = ThreadManagementMixin.__new__(type("P", (ThreadManagementMixin,), {}))
    proc.db = MagicMock()
    proc.db.get_channel_memory_async = AsyncMock(return_value=[])
    proc.db.add_channel_memory_async = AsyncMock()
    proc.db.update_channel_memory_async = AsyncMock()
    proc.db.update_channel_fact_async = AsyncMock(return_value=True)
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
    proc.db.update_channel_fact_async.assert_not_called()


async def test_extraction_update_updates_row():
    proc = _proc({"action": "update", "id": 7, "content": "revised"})
    # An id the extractor was actually SHOWN. One it wasn't is refused (see
    # test_channel_steering.py) — an unknown id used to reach an unrestricted update.
    proc.db.get_channel_memory_async = AsyncMock(
        return_value=[{"id": 7, "content": "old wording", "scope": "channel", "author": "U1"}])
    with patch.object(config, "enable_channel_memory", True):
        await proc._async_extract_channel_memory(_state())
    proc.db.update_channel_fact_async.assert_called_once_with(7, "revised")


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
