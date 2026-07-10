"""Unit tests for the edit-in-place progress checklist (spec F4).

The checklist renders completed steps with a check and the active step with the
loader emoji, editing a single Slack message (or the composer status where that is
the only surface) in place. All edits serialize on an internal lock; non-terminal
edits inside the min-edit interval coalesce; terminal states are sticky.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from config import config
from message_markers import CHECKLIST_STATUS_MARKER
from message_processor.progress import ProgressChecklist

LOADER = config.circle_loader_emoji


def _client(**overrides):
    client = SimpleNamespace(
        send_thinking_indicator=AsyncMock(return_value="msg1"),
        update_message=AsyncMock(return_value=True),
        set_assistant_status=AsyncMock(return_value=True),
        clear_assistant_status=AsyncMock(return_value=True),
        delete_message=AsyncMock(return_value=True),
    )
    for key, value in overrides.items():
        setattr(client, key, value)
    return client


def _last_text(client):
    # Checklist message writes carry an invisible history-filter marker; strip it so
    # assertions compare the visible rendering.
    return client.update_message.await_args.args[2].replace(CHECKLIST_STATUS_MARKER, "")


# ---------------- step accumulation & rendering ----------------

@pytest.mark.asyncio
async def test_step_accumulates_and_renders():
    client = _client()
    c = ProgressChecklist(client, "C1", "T1", message_id="m1", min_edit_interval=0)

    await c.step("Enhancing prompt…", done_text="Enhanced prompt")
    assert _last_text(client) == f"{LOADER} Enhancing prompt…"

    await c.step("Generating image…", done_text="Generated image")
    assert _last_text(client) == f"✓ Enhanced prompt\n{LOADER} Generating image…"

    await c.step("Uploading…")
    assert _last_text(client) == (
        f"✓ Enhanced prompt\n✓ Generated image\n{LOADER} Uploading…")


@pytest.mark.asyncio
async def test_done_text_defaults_to_active_minus_ellipsis():
    client = _client()
    c = ProgressChecklist(client, "C1", "T1", message_id="m1", min_edit_interval=0)
    await c.step("Analyzing…")
    await c.step("Editing…")
    assert _last_text(client) == f"✓ Analyzing\n{LOADER} Editing…"


# ---------------- first-call message creation ----------------

@pytest.mark.asyncio
async def test_first_call_creates_message():
    client = _client()
    c = ProgressChecklist(client, "C1", "T1", min_edit_interval=0)
    assert c.surface == "none"  # undetermined until first step
    await c.step("Working…")
    client.send_thinking_indicator.assert_awaited_once_with("C1", "T1")
    assert c.surface == "message"
    assert c.message_id == "msg1"
    client.update_message.assert_awaited_once()
    assert client.update_message.await_args.args[1] == "msg1"


# ---------------- status-only surface degradation + terminal clear ----------------

@pytest.mark.asyncio
async def test_status_only_surface_degrades_to_set_status():
    client = _client(send_thinking_indicator=AsyncMock(return_value=None))
    c = ProgressChecklist(client, "C1", "T1", min_edit_interval=0)
    await c.step("Generating image…", done_text="Generated image")
    assert c.surface == "assistant_status"
    assert c.message_id is None
    client.update_message.assert_not_awaited()
    # Only the active step's text goes to the composer status.
    client.set_assistant_status.assert_awaited_once_with(
        "C1", "T1", status="Generating image…")

    await c.complete()
    client.clear_assistant_status.assert_awaited_once_with("C1", "T1")


# ---------------- fail keeps the message visible ----------------

@pytest.mark.asyncio
async def test_fail_keeps_message_with_cross():
    client = _client()
    c = ProgressChecklist(client, "C1", "T1", message_id="m1", min_edit_interval=0)
    await c.step("Enhancing prompt…", done_text="Enhanced prompt")
    await c.step("Generating image…")
    await c.fail("Image generation failed")
    assert _last_text(client) == "✓ Enhanced prompt\n✗ Image generation failed"
    client.delete_message.assert_not_awaited()


# ---------------- complete + delete_after ----------------

@pytest.mark.asyncio
async def test_complete_delete_after_deletes():
    client = _client()
    c = ProgressChecklist(client, "C1", "T1", message_id="m1", min_edit_interval=0)
    await c.step("Generating image…", done_text="Generated image")
    await c.complete(delete_after=0.01)
    client.delete_message.assert_not_awaited()  # not yet
    await asyncio.sleep(0.05)
    client.delete_message.assert_awaited_once_with("C1", "m1")


@pytest.mark.asyncio
async def test_complete_final_text_appended():
    client = _client()
    c = ProgressChecklist(client, "C1", "T1", message_id="m1", min_edit_interval=0)
    await c.step("Generating image…", done_text="Generated image")
    await c.complete(final_text="Uploaded")
    assert _last_text(client) == "✓ Generated image\n✓ Uploaded"


# ---------------- coalescing ----------------

@pytest.mark.asyncio
async def test_coalescing_lands_latest_state():
    client = _client()
    c = ProgressChecklist(client, "C1", "T1", message_id="m1", min_edit_interval=0.1)

    await c.step("A…")           # immediate flush
    await c.step("B…")           # within interval -> schedules deferred flush
    await c.step("C…")           # coalesces into the pending flush
    await asyncio.sleep(0.2)     # let the deferred flush fire

    # Two edits total: the immediate "A" and one coalesced final render.
    assert client.update_message.await_count == 2
    assert _last_text(client) == f"✓ A\n✓ B\n{LOADER} C…"


# ---------------- concurrency ----------------

@pytest.mark.asyncio
async def test_concurrent_step_and_complete_reach_terminal():
    client = _client()
    c = ProgressChecklist(client, "C1", "T1", message_id="m1", min_edit_interval=0)
    await asyncio.gather(c.step("Generating image…"), c.complete())
    assert c._terminal is True
    # Post-terminal calls no-op.
    before = client.update_message.await_count
    await c.step("late")
    await c.fail("late fail")
    assert client.update_message.await_count == before


# ---------------- cancellation during delete_after ----------------

@pytest.mark.asyncio
async def test_cancel_during_delete_after_does_not_delete():
    client = _client()
    c = ProgressChecklist(client, "C1", "T1", message_id="m1", min_edit_interval=0)
    await c.step("Generating image…")
    await c.complete(delete_after=10)
    c._delete_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await c._delete_task
    client.delete_message.assert_not_awaited()


# ---------------- false-returning client methods ----------------

@pytest.mark.asyncio
async def test_false_returning_update_message_keeps_state():
    client = _client(update_message=AsyncMock(return_value=False))
    c = ProgressChecklist(client, "C1", "T1", message_id="m1", min_edit_interval=0)
    # Nothing raises; the failed edit is swallowed and state is retained.
    await c.step("Enhancing prompt…", done_text="Enhanced prompt")
    await c.step("Generating image…")
    assert client.update_message.await_count == 2
    # State still carries the completed step for a later retry.
    assert c._done == ["Enhanced prompt"]


# ---------------- terminal idempotency ----------------

@pytest.mark.asyncio
async def test_terminal_is_idempotent():
    client = _client()
    c = ProgressChecklist(client, "C1", "T1", message_id="m1", min_edit_interval=0)
    await c.step("Generating image…", done_text="Generated image")
    await c.complete()
    count_after_first = client.update_message.await_count
    await c.complete()       # second complete no-ops
    await c.fail("nope")     # fail after complete no-ops
    assert client.update_message.await_count == count_after_first


# ---------------- rotator not started when checklist active ----------------

class _ImgHost:
    """Minimal MessageProcessor-ish host exposing the real image-gen handler."""

    def __init__(self, client, openai_client):
        from message_processor.handlers.image_gen import ImageGenerationMixin
        for name in ("_handle_image_generation", "_resolve_prompt_message",
                     "_create_prompt_message", "_lazy_create_prompt_ref"):
            setattr(self, name, getattr(ImageGenerationMixin, name).__get__(self))
        self.openai_client = openai_client
        self.db = None
        self.thread_manager = SimpleNamespace(
            get_or_create_asset_ledger=lambda *a, **k: SimpleNamespace(
                add_image=lambda *a, **k: None))
        self._start_progress_updater_async = AsyncMock(return_value=None)
        self._update_status = lambda *a, **k: None

    async def _inject_image_analyses(self, messages, state):
        return messages

    async def _pre_trim_messages_for_api(self, messages, model=None):
        return messages

    def _update_message_streaming_sync(self, *a, **k):
        return {"success": True, "rate_limited": False, "retry_after": None}

    def _add_message_with_token_management(self, *a, **k):
        pass

    def _format_user_content_with_username(self, text, message):
        return text

    def log_info(self, *a, **k):
        pass

    log_debug = log_warning = log_error = log_info


def _img_client():
    return SimpleNamespace(
        supports_streaming=lambda: True,
        get_streaming_config=lambda: {},
        send_thinking_indicator=AsyncMock(return_value="gen1"),
        update_message=AsyncMock(return_value=True),
        delete_message=AsyncMock(return_value=True),
        set_assistant_status=AsyncMock(return_value=True),
        clear_assistant_status=AsyncMock(return_value=True),
    )


def _img_openai():
    return SimpleNamespace(
        _enhance_image_prompt=AsyncMock(return_value="an enhanced prompt"),
        generate_image=AsyncMock(return_value=SimpleNamespace(
            base64_data="b64", prompt="an enhanced prompt", format="png")),
    )


def _thread_state():
    return SimpleNamespace(
        messages=[], config_overrides={}, thread_ts="T1",
        channel_id="C1", current_model="gpt-5")


def _message():
    return SimpleNamespace(user_id="U1", channel_id="C1", metadata={"ts": "1.1"})


@pytest.mark.asyncio
async def test_rotator_not_started_when_checklist_active(mock_env, monkeypatch):
    monkeypatch.setattr(config, "enable_progress_checklist", True)
    monkeypatch.setattr(
        config, "get_thread_config_async",
        AsyncMock(return_value={
            "enable_streaming": True, "image_model": "gpt-image-1",
            "image_size": "1024x1024", "image_quality": "auto",
            "image_background": "auto"}))
    client = _img_client()
    host = _ImgHost(client, _img_openai())

    resp = await host._handle_image_generation(
        "a cat", _thread_state(), client, "C1", "think1", _message())

    host._start_progress_updater_async.assert_not_awaited()
    # The checklist owned the generating-status message and edited it in place.
    assert resp.metadata.get("status_message_id") == "gen1"
    assert any(call.args[1] == "gen1" for call in client.update_message.await_args_list)


@pytest.mark.asyncio
async def test_rotator_started_when_checklist_disabled(mock_env, monkeypatch):
    monkeypatch.setattr(config, "enable_progress_checklist", False)
    monkeypatch.setattr(
        config, "get_thread_config_async",
        AsyncMock(return_value={
            "enable_streaming": True, "image_model": "gpt-image-1",
            "image_size": "1024x1024", "image_quality": "auto",
            "image_background": "auto"}))
    client = _img_client()
    host = _ImgHost(client, _img_openai())

    await host._handle_image_generation(
        "a cat", _thread_state(), client, "C1", "think1", _message())

    host._start_progress_updater_async.assert_awaited_once()
