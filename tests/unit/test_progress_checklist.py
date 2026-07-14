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
        send_message_get_ts=AsyncMock(return_value={"success": True, "ts": "posted1"}),
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


# ---------------- prefer_message: force a visible message on status-only surfaces ----------------

@pytest.mark.asyncio
async def test_prefer_message_posts_real_message_when_status_only():
    # send_thinking_indicator returns None (status-only surface), but prefer_message
    # forces a real thread message via send_message_get_ts instead of degrading.
    client = _client(send_thinking_indicator=AsyncMock(return_value=None))
    c = ProgressChecklist(client, "C1", "T1", min_edit_interval=0, prefer_message=True)
    await c.step("Generating image…", done_text="Generated image")

    assert c.surface == "message"
    assert c.message_id == "posted1"
    assert c.mirrors_status is True
    # The message was created (not degraded to composer-status-only).
    client.send_message_get_ts.assert_awaited_once()
    # The created message carries the invisible history-filter marker.
    created_text = client.send_message_get_ts.await_args.args[2]
    assert CHECKLIST_STATUS_MARKER in created_text
    assert "Generating image…" in created_text


@pytest.mark.asyncio
async def test_prefer_message_mirrors_active_step_to_status_and_clears():
    client = _client(send_thinking_indicator=AsyncMock(return_value=None))
    c = ProgressChecklist(client, "C1", "T1", min_edit_interval=0, prefer_message=True)

    await c.step("Generating image…", done_text="Generated image")
    # Dual display: active step mirrored into the composer status too.
    client.set_assistant_status.assert_awaited_with("C1", "T1", status="Generating image…")

    await c.step("Uploading…")
    client.set_assistant_status.assert_awaited_with("C1", "T1", status="Uploading…")

    await c.complete()
    # The checklist owns clearing the mirrored status on terminal.
    client.clear_assistant_status.assert_awaited_once_with("C1", "T1")


@pytest.mark.asyncio
async def test_prefer_message_delete_after_still_deletes():
    client = _client(send_thinking_indicator=AsyncMock(return_value=None))
    c = ProgressChecklist(client, "C1", "T1", min_edit_interval=0, prefer_message=True)
    await c.step("Generating image…", done_text="Generated image")
    await c.complete(delete_after=0.01)
    client.delete_message.assert_not_awaited()  # not yet
    await asyncio.sleep(0.05)
    client.delete_message.assert_awaited_once_with("C1", "posted1")


@pytest.mark.asyncio
async def test_prefer_message_off_reverts_to_status_degradation():
    # Even with send_message_get_ts available, prefer_message=False (config off) keeps
    # today's degradation: no message posted, composer status carries the active step.
    client = _client(send_thinking_indicator=AsyncMock(return_value=None))
    c = ProgressChecklist(client, "C1", "T1", min_edit_interval=0, prefer_message=False)
    await c.step("Generating image…", done_text="Generated image")

    assert c.surface == "assistant_status"
    assert c.message_id is None
    assert c.mirrors_status is False
    client.send_message_get_ts.assert_not_awaited()
    client.update_message.assert_not_awaited()
    client.set_assistant_status.assert_awaited_once_with("C1", "T1", status="Generating image…")


@pytest.mark.asyncio
async def test_prefer_message_with_real_thinking_ts_does_not_mirror():
    # send_thinking_indicator returns a real ts (setStatus failed → no status surface):
    # normal message surface, no mirror even though prefer_message is on.
    client = _client()  # send_thinking_indicator → "msg1"
    c = ProgressChecklist(client, "C1", "T1", min_edit_interval=0, prefer_message=True)
    await c.step("Generating image…")
    assert c.surface == "message"
    assert c.message_id == "msg1"
    assert c.mirrors_status is False
    client.send_message_get_ts.assert_not_awaited()
    client.set_assistant_status.assert_not_awaited()


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
