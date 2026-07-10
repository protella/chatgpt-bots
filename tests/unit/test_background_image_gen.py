"""Unit tests for F1 — background image generation (release the thread lock).

Covers the generation registry, the dedicated image timeout, the shared delivery seam
(publish_image), the inline→background handoff, the in-flight suffix note, the
merge-preserving DB upsert, and the checklist history-filter marker.
"""
import asyncio
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from config import config
from message_processor.image_delivery import publish_image
from message_processor.progress import ProgressChecklist
from thread_manager import AsyncThreadStateManager


# --------------------------------------------------------------------------- registry

@pytest.mark.asyncio
async def test_registry_register_and_finish_id_conditional():
    tm = AsyncThreadStateManager(db=None)
    tm.register_generation("C:T", "gen1", "a cat")
    assert tm.generation_in_flight("C:T")["generation_id"] == "gen1"

    # A stale id must not clear the current entry.
    assert tm.finish_generation("C:T", "OLD") is False
    assert tm.generation_in_flight("C:T") is not None

    # The matching id clears it.
    assert tm.finish_generation("C:T", "gen1") is True
    assert tm.generation_in_flight("C:T") is None


@pytest.mark.asyncio
async def test_registry_watchdog_clears_stale(monkeypatch):
    tm = AsyncThreadStateManager(db=None)
    tm.register_generation("C:T", "gen1", "a cat")
    # Backdate the entry beyond api_timeout_image + 30s.
    tm._active_generations["C:T"]["started_at"] -= (config.api_timeout_image + 60)
    assert tm.generation_in_flight("C:T") is None
    assert "C:T" not in tm._active_generations


@pytest.mark.asyncio
async def test_registry_cancel_generations():
    tm = AsyncThreadStateManager(db=None)

    async def _never():
        await asyncio.sleep(100)

    task = asyncio.create_task(_never())
    tm.register_generation("C:T", "gen1", "a cat", task=task)
    await tm.cancel_generations(timeout=1.0)
    assert task.cancelled()
    assert tm._active_generations == {}


# --------------------------------------------------------------------------- timeout

def test_image_operation_timeout_uses_image_budget(mock_env, monkeypatch):
    from openai_client.base import OpenAIClient
    # Distinct values so the routing is unambiguous (both default to 300 in this env).
    monkeypatch.setattr(config, "api_timeout_image", 999.0)
    monkeypatch.setattr(config, "api_timeout_read", 180.0)
    # Bind just the method — avoid constructing the whole client.
    getter = OpenAIClient._get_operation_timeout.__get__(
        SimpleNamespace(log_debug=lambda *a, **k: None))
    assert getter("image_generation") == 999.0
    assert getter("image_edit") == 999.0
    assert getter("image_generation") > config.api_timeout_read
    # Vision stays on the general read timeout.
    assert getter("vision_analysis") == 180.0


# --------------------------------------------------------------------------- delivery seam

def _image_data():
    return SimpleNamespace(base64_data="b64", prompt="an enhanced prompt",
                           format="png", to_bytes=lambda: b"imgbytes")


def _delivery_client(**overrides):
    client = SimpleNamespace(
        send_image=AsyncMock(return_value="https://files.slack.com/img1"),
        update_message=AsyncMock(return_value=True),
        clear_assistant_status=AsyncMock(return_value=True),
        delete_message=AsyncMock(return_value=True),
        send_thinking_indicator=AsyncMock(return_value="stat1"),
        channel_pulse=SimpleNamespace(record_bot_reply=MagicMock()),
    )
    for k, v in overrides.items():
        setattr(client, k, v)
    return client


def _processor():
    return SimpleNamespace(
        log_debug=lambda *a, **k: None,
        log_error=lambda *a, **k: None,
        update_last_image_url=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_publish_image_background_success_writes_db_and_accounts():
    client = _delivery_client()
    db = SimpleNamespace(save_image_metadata_async=AsyncMock())
    tm = AsyncThreadStateManager(db=None)
    proc = _processor()

    url = await publish_image(
        processor=proc, client=client, channel_id="C1", thread_id="T1",
        thread_key="C1:T1", image_data=_image_data(), checklist=None,
        generation_id="gen1", prompt="an enhanced prompt", db=db,
        thread_manager=tm, unprompted=True, message_ts="1.1")

    assert url == "https://files.slack.com/img1"
    client.send_image.assert_awaited_once()
    # DB row written directly, carrying the generation_id (no breadcrumb dependency).
    db.save_image_metadata_async.assert_awaited_once()
    kwargs = db.save_image_metadata_async.await_args.kwargs
    assert kwargs["url"] == url and kwargs["prompt"] == "an enhanced prompt"
    assert kwargs["metadata"]["generation_id"] == "gen1"
    # Ledger updated in memory only.
    assert tm.get_or_create_asset_ledger("T1").images[-1]["slack_url"] == url
    # Unprompted channel turn is accounted here, not before delivery.
    client.channel_pulse.record_bot_reply.assert_called_once()
    # Sync-path persistence helper never used on the background path.
    proc.update_last_image_url.assert_not_awaited()


@pytest.mark.asyncio
async def test_publish_image_falsey_url_is_failure():
    client = _delivery_client(send_image=AsyncMock(return_value=None))
    db = SimpleNamespace(save_image_metadata_async=AsyncMock())
    checklist = ProgressChecklist(client, "C1", "T1", message_id="m1", min_edit_interval=0)

    url = await publish_image(
        processor=_processor(), client=client, channel_id="C1", thread_id="T1",
        thread_key="C1:T1", image_data=_image_data(), checklist=checklist,
        generation_id="gen1", prompt="p", db=db, thread_manager=AsyncThreadStateManager(db=None),
        unprompted=False, message_ts=None)

    assert url is None
    db.save_image_metadata_async.assert_not_awaited()
    assert checklist._terminal is True  # checklist failed


@pytest.mark.asyncio
async def test_publish_image_sync_path_uses_breadcrumb_helper():
    client = _delivery_client()
    db = SimpleNamespace(save_image_metadata_async=AsyncMock())
    proc = _processor()

    url = await publish_image(
        processor=proc, client=client, channel_id="C1", thread_id="T1",
        thread_key="C1:T1", image_data=_image_data(), checklist=None,
        generation_id=None, prompt="", db=db,
        thread_manager=AsyncThreadStateManager(db=None), unprompted=False)

    assert url == "https://files.slack.com/img1"
    # Legacy sync path persists via the breadcrumb helper, not a direct DB write.
    proc.update_last_image_url.assert_awaited_once()
    db.save_image_metadata_async.assert_not_awaited()


# --------------------------------------------------------------------------- handoff

class _BgHost:
    """Minimal MessageProcessor-ish host exposing the real image-gen handler + a real
    thread manager (so the registry/latch are exercised end to end)."""

    def __init__(self, client, openai_client, db=None):
        from message_processor.handlers.image_gen import ImageGenerationMixin
        for name in ("_handle_image_generation", "_start_background_generation",
                     "_finish_image_generation_background", "_abort_checklist",
                     "_create_prompt_message", "_lazy_create_prompt_ref",
                     "_resolve_prompt_message"):
            setattr(self, name, getattr(ImageGenerationMixin, name).__get__(self))
        self.openai_client = openai_client
        self.db = db
        self.thread_manager = AsyncThreadStateManager(db=None)
        self._tasks = []

    def _schedule_async_call(self, coro):
        task = asyncio.create_task(coro)
        self._tasks.append(task)
        return task

    async def _inject_image_analyses(self, messages, state):
        return messages

    async def _pre_trim_messages_for_api(self, messages, model=None):
        return messages

    def _update_status(self, *a, **k):
        pass

    def _update_message_streaming_sync(self, *a, **k):
        return {"success": True, "rate_limited": False, "retry_after": None}

    def _add_message_with_token_management(self, thread_state, role, content, **k):
        thread_state.messages.append({"role": role, "content": content,
                                      "metadata": k.get("metadata")})

    def _format_user_content_with_username(self, text, message):
        return text

    async def update_last_image_url(self, *a, **k):
        pass

    def log_info(self, *a, **k):
        pass

    log_debug = log_warning = log_error = log_info


def _bg_client():
    return SimpleNamespace(
        supports_streaming=lambda: False,  # exercise the simpler non-streaming path
        get_streaming_config=lambda: {},
        send_thinking_indicator=AsyncMock(return_value="stat1"),
        update_message=AsyncMock(return_value=True),
        delete_message=AsyncMock(return_value=True),
        send_image=AsyncMock(return_value="https://files.slack.com/img1"),
        clear_assistant_status=AsyncMock(return_value=True),
        handle_error=AsyncMock(),
        send_message=AsyncMock(),
        channel_pulse=SimpleNamespace(record_bot_reply=MagicMock()),
    )


def _bg_openai(**overrides):
    oc = SimpleNamespace(
        generate_image=AsyncMock(return_value=_image_data()),
        _enhance_image_prompt=AsyncMock(return_value="an enhanced prompt"),
    )
    for k, v in overrides.items():
        setattr(oc, k, v)
    return oc


def _thread_state():
    return SimpleNamespace(messages=[], config_overrides={}, thread_ts="T1",
                           channel_id="C1", current_model="gpt-5")


def _message(participation=False):
    md = {"ts": "1.1"}
    if participation:
        md["participation_check"] = True
    return SimpleNamespace(user_id="U1", channel_id="C1", metadata=md)


def _thread_config(**over):
    base = {"enable_streaming": True, "image_model": "gpt-image-1",
            "image_size": "1024x1024", "image_quality": "auto", "image_background": "auto"}
    base.update(over)
    return base


@pytest.mark.asyncio
async def test_handoff_returns_background_without_awaiting_generation(mock_env, monkeypatch):
    monkeypatch.setattr(config, "enable_background_image_gen", True)
    monkeypatch.setattr(config, "get_thread_config_async",
                        AsyncMock(return_value=_thread_config()))
    client, oc = _bg_client(), _bg_openai()
    host = _BgHost(client, oc, db=SimpleNamespace(save_image_metadata_async=AsyncMock()))
    ts = _thread_state()

    resp = await host._handle_image_generation(
        "a cat", ts, client, "C1", "think1", _message(), allow_background=True)

    assert resp.type == "background"
    assert resp.metadata["background_owns_status"] is True
    gid = resp.metadata["generation_id"]
    # The slow call has NOT run yet — the turn returned immediately (lock releases).
    oc.generate_image.assert_not_awaited()
    # Registered, and only the USER message was appended (no assistant breadcrumb).
    assert host.thread_manager.generation_in_flight("C1:T1")["generation_id"] == gid
    assert [m["role"] for m in ts.messages] == ["user"]

    # Drain the background job.
    await host.thread_manager._active_generations["C1:T1"]["task"]
    oc.generate_image.assert_awaited_once()
    client.send_image.assert_awaited_once()
    host.db.save_image_metadata_async.assert_awaited_once()
    # Registry cleared and a rebuild flagged for the next turn.
    assert host.thread_manager.generation_in_flight("C1:T1") is None
    assert host.thread_manager.consume_needs_refresh("C1:T1") is True


@pytest.mark.asyncio
async def test_config_off_runs_inline(mock_env, monkeypatch):
    monkeypatch.setattr(config, "enable_background_image_gen", False)
    monkeypatch.setattr(config, "get_thread_config_async",
                        AsyncMock(return_value=_thread_config()))
    client, oc = _bg_client(), _bg_openai()
    host = _BgHost(client, oc, db=None)

    resp = await host._handle_image_generation(
        "a cat", _thread_state(), client, "C1", "think1", _message(), allow_background=True)

    # Inline: generation ran during the turn and a normal image response came back.
    assert resp.type == "image"
    oc.generate_image.assert_awaited_once()
    assert host.thread_manager.generation_in_flight("C1:T1") is None


@pytest.mark.asyncio
async def test_background_moderation_block(mock_env, monkeypatch):
    monkeypatch.setattr(config, "enable_background_image_gen", True)
    monkeypatch.setattr(config, "get_thread_config_async",
                        AsyncMock(return_value=_thread_config()))
    client = _bg_client()
    oc = _bg_openai(generate_image=AsyncMock(side_effect=Exception("moderation_blocked")))
    host = _BgHost(client, oc, db=SimpleNamespace(save_image_metadata_async=AsyncMock()))

    resp = await host._handle_image_generation(
        "a banned thing", _thread_state(), client, "C1", "think1", _message(),
        allow_background=True)
    assert resp.type == "background"
    await host.thread_manager._active_generations["C1:T1"]["task"]

    # Friendly moderation notice posted; no image; registry cleared.
    client.send_message.assert_awaited_once()
    client.send_image.assert_not_awaited()
    assert host.thread_manager.generation_in_flight("C1:T1") is None


# --------------------------------------------------------------------------- suffix note

class _SuffixHost:
    def __init__(self, tm):
        from message_processor.utilities import MessageUtilitiesMixin
        self._build_generation_inflight_note = (
            MessageUtilitiesMixin._build_generation_inflight_note.__get__(self))
        self._escape_suffix_text = MessageUtilitiesMixin._escape_suffix_text
        self.thread_manager = tm

    def log_debug(self, *a, **k):
        pass


def test_inflight_suffix_note_present_then_absent():
    tm = AsyncThreadStateManager(db=None)
    host = _SuffixHost(tm)
    assert host._build_generation_inflight_note("C1", "T1") is None

    tm.register_generation("C1:T1", "gen1", 'a cat with [brackets]\nand newline')
    note = host._build_generation_inflight_note("C1", "T1")
    assert note is not None
    assert "currently being generated" in note
    # Free text is escaped: no raw brackets/newlines leak into the block.
    assert "[brackets]" not in note
    assert "\n" not in note

    tm.finish_generation("C1:T1", "gen1")
    assert host._build_generation_inflight_note("C1", "T1") is None


# --------------------------------------------------------------------------- DB merge upsert

@pytest.mark.asyncio
async def test_merge_preserving_upsert(tmp_path):
    from database import DatabaseManager
    db = DatabaseManager("test")
    db.db_path = f"{tmp_path}/test.db"
    db.conn = sqlite3.connect(db.db_path, check_same_thread=False, isolation_level=None)
    db.conn.row_factory = sqlite3.Row
    db.init_schema()

    # publish_image's write: full metadata.
    await db.save_image_metadata_async(
        thread_id="C:T", url="https://img/1", image_type="generated",
        prompt="a red cat", analysis="", metadata={"generation_id": "abc123"})
    # A later rebuild write over the SAME url with empty prompt/generic type + a ts.
    await db.save_image_metadata_async(
        thread_id="C:T", url="https://img/1", image_type="assistant",
        prompt="", analysis="", metadata=None, message_ts="9.9")

    row = db.conn.execute(
        "SELECT prompt, image_type, metadata_json, message_ts FROM images WHERE url=?",
        ("https://img/1",)).fetchone()
    db.conn.close()
    import json
    assert row["prompt"] == "a red cat"          # preserved over the empty write
    assert row["image_type"] == "generated"      # preserved over "assistant"
    assert json.loads(row["metadata_json"])["generation_id"] == "abc123"  # preserved
    assert row["message_ts"] == "9.9"            # newly filled in


# --------------------------------------------------------------------------- history marker

def test_checklist_marker_recognized():
    from message_markers import CHECKLIST_STATUS_MARKER, is_checklist_status_text
    assert is_checklist_status_text(f"✓ Generated image{CHECKLIST_STATUS_MARKER}")
    assert not is_checklist_status_text("✓ Generated image")
    assert not is_checklist_status_text("")


# ----------------------------------------------------------- enhanced-prompt fallback

class _PromptHost:
    def __init__(self):
        from message_processor.handlers.image_gen import ImageGenerationMixin
        for name in ("_create_prompt_message", "_lazy_create_prompt_ref",
                     "_resolve_prompt_message"):
            setattr(self, name, getattr(ImageGenerationMixin, name).__get__(self))
        self.edits = []

    def _update_message_streaming_sync(self, client, channel_id, mid, text):
        self.edits.append((mid, text))
        return {"success": True, "rate_limited": False, "retry_after": None}

    def log_debug(self, *a, **k):
        pass

    log_info = log_warning = log_error = log_debug


def _prompt_client(ts="new1"):
    return SimpleNamespace(
        send_message_get_ts=AsyncMock(return_value={"success": True, "ts": ts}))


@pytest.mark.asyncio
async def test_resolve_prompt_edits_existing_id():
    host = _PromptHost()
    client = _prompt_client()
    pid = await host._resolve_prompt_message(client, "C1", "T1", {"id": "think1"}, "final")
    assert pid == "think1"
    client.send_message_get_ts.assert_not_awaited()  # no new message on the normal surface
    assert host.edits == [("think1", "final")]


@pytest.mark.asyncio
async def test_resolve_prompt_creates_new_message_when_none():
    host = _PromptHost()
    client = _prompt_client(ts="new1")
    ref = {"id": None, "creating": False}
    pid = await host._resolve_prompt_message(client, "C1", "T1", ref, "the enhanced prompt")
    # Status-only surface: enhanced prompt posted as its OWN new message.
    assert pid == "new1"
    client.send_message_get_ts.assert_awaited_once_with("C1", "T1", "the enhanced prompt")
    assert ref["id"] == "new1"


@pytest.mark.asyncio
async def test_lazy_create_prompt_ref_populates_id():
    host = _PromptHost()
    client = _prompt_client(ts="new2")
    ref = {"id": None, "creating": True}
    await host._lazy_create_prompt_ref(client, "C1", "T1", "first chunk", ref)
    assert ref["id"] == "new2"
    assert ref["creating"] is False


@pytest.mark.asyncio
async def test_streaming_status_only_posts_enhanced_prompt_as_new_message(mock_env, monkeypatch):
    # thinking_id=None (status-only) + streaming enhancement: the enhanced prompt must
    # still appear — as its own new message — not silently no-op on a None id.
    monkeypatch.setattr(config, "enable_background_image_gen", False)  # stay inline
    monkeypatch.setattr(config, "get_thread_config_async",
                        AsyncMock(return_value=_thread_config()))
    client = _bg_client()
    client.supports_streaming = lambda: True
    client.send_thinking_indicator = AsyncMock(return_value=None)  # status-only
    client.send_message_get_ts = AsyncMock(return_value={"success": True, "ts": "prompt_msg"})
    host = _BgHost(client, _bg_openai(), db=None)

    resp = await host._handle_image_generation(
        "a cat", _thread_state(), client, "C1", None, _message(), allow_background=True)

    assert resp.type == "image"
    # The enhanced prompt was posted as its own message and marked do-not-touch.
    client.send_message_get_ts.assert_awaited()
    assert resp.metadata.get("prompt_message_id") == "prompt_msg"


@pytest.mark.asyncio
async def test_checklist_message_carries_marker():
    from message_markers import CHECKLIST_STATUS_MARKER
    client = _delivery_client()
    c = ProgressChecklist(client, "C1", "T1", message_id="m1", min_edit_interval=0)
    await c.step("Generating image…", done_text="Generated image")
    sent = client.update_message.await_args.args[2]
    assert CHECKLIST_STATUS_MARKER in sent
