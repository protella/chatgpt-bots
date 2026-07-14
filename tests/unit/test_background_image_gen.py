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
    assert tm.generations_in_flight("C:T")[0]["generation_id"] == "gen1"

    # A stale id must not clear the current entry.
    assert tm.finish_generation("C:T", "OLD") is False
    assert len(tm.generations_in_flight("C:T")) == 1

    # The matching id clears it.
    assert tm.finish_generation("C:T", "gen1") is True
    assert tm.generations_in_flight("C:T") == []


@pytest.mark.asyncio
async def test_registry_watchdog_clears_stale(monkeypatch):
    tm = AsyncThreadStateManager(db=None)
    tm.register_generation("C:T", "gen1", "a cat")
    # Backdate the entry beyond api_timeout_image + 30s.
    tm._active_generations["C:T"]["gen1"]["started_at"] -= (config.api_timeout_image + 60)
    assert tm.generations_in_flight("C:T") == []
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
    # Ledger updated in memory only — metadata, never the base64 payload (pitfall 6).
    row = tm.get_or_create_asset_ledger("T1").images[-1]
    assert row["slack_url"] == url
    assert row["data"] is None
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
async def test_publish_image_sync_path_persists_directly_and_via_breadcrumb():
    client = _delivery_client()
    db = SimpleNamespace(save_image_metadata_async=AsyncMock())
    proc = _processor()

    url = await publish_image(
        processor=proc, client=client, channel_id="C1", thread_id="T1",
        thread_key="C1:T1", image_data=_image_data(), checklist=None,
        generation_id=None, prompt="", db=db,
        thread_manager=AsyncThreadStateManager(db=None), unprompted=False,
        image_type="edited")

    assert url == "https://files.slack.com/img1"
    # Sync path persists DIRECTLY (breadcrumb-independent; falls back to image_data.prompt)
    # AND refreshes the warm breadcrumb — a mid-flight refresh can't lose the DB row.
    db.save_image_metadata_async.assert_awaited_once()
    kwargs = db.save_image_metadata_async.await_args.kwargs
    assert kwargs["prompt"] == "an enhanced prompt" and kwargs["image_type"] == "edited"
    assert "generation_id" not in kwargs["metadata"]  # no id on the sync path
    proc.update_last_image_url.assert_awaited_once()


@pytest.mark.asyncio
async def test_publish_image_persist_failure_still_reports_posted():
    # A DB failure after a successful upload must NOT un-post the image or fail the checklist.
    client = _delivery_client()
    db = SimpleNamespace(save_image_metadata_async=AsyncMock(side_effect=RuntimeError("db down")))
    checklist = ProgressChecklist(client, "C1", "T1", message_id="m1", min_edit_interval=0)

    url = await publish_image(
        processor=_processor(), client=client, channel_id="C1", thread_id="T1",
        thread_key="C1:T1", image_data=_image_data(), checklist=checklist,
        generation_id="gen1", prompt="p", db=db,
        thread_manager=AsyncThreadStateManager(db=None), unprompted=False)

    assert url == "https://files.slack.com/img1"  # still posted
    assert checklist._failed_note is None  # checklist NOT failed
    assert checklist._terminal is True     # it completed


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


@pytest.mark.asyncio
async def test_checklist_message_carries_marker():
    from message_markers import CHECKLIST_STATUS_MARKER
    client = _delivery_client()
    c = ProgressChecklist(client, "C1", "T1", message_id="m1", min_edit_interval=0)
    await c.step("Generating image…", done_text="Generated image")
    sent = client.update_message.await_args.args[2]
    assert CHECKLIST_STATUS_MARKER in sent


@pytest.mark.asyncio
async def test_abort_checklist_clears_mirrored_status_and_deletes_message():
    # A force-message checklist deletes its message AND clears the mirrored composer
    # status on abort (moderation/cancel), so no status bubble lingers.
    client = _delivery_client(
        send_thinking_indicator=AsyncMock(return_value=None),
        send_message_get_ts=AsyncMock(return_value={"success": True, "ts": "posted1"}),
        set_assistant_status=AsyncMock(return_value=True),
    )
    from message_processor.handlers.image_gen import ImageJobMixin
    host = SimpleNamespace(log_warning=lambda *a, **k: None)
    host._abort_checklist = ImageJobMixin._abort_checklist.__get__(host)
    c = ProgressChecklist(client, "C1", "T1", min_edit_interval=0, prefer_message=True)
    await c.step("Generating image…", done_text="Generated image")
    assert c.surface == "message" and c.mirrors_status is True

    await host._abort_checklist(c, client, "C1", "T1")
    client.delete_message.assert_awaited_once_with("C1", "posted1")
    client.clear_assistant_status.assert_awaited_once_with("C1", "T1")


# ============================================================ F13: parallel generations

# --------------------------------------------------------------------- multi-entry registry

@pytest.mark.asyncio
async def test_registry_multi_entry_id_conditional_finish():
    tm = AsyncThreadStateManager(db=None)
    tm.register_generation("C:T", "genA", "a cat")
    tm.register_generation("C:T", "genB", "a dog")
    # Both live concurrently on the same thread (F13).
    assert {e["generation_id"] for e in tm.generations_in_flight("C:T")} == {"genA", "genB"}

    # Finishing ONE leaves the sibling untouched.
    assert tm.finish_generation("C:T", "genA") is True
    assert [e["generation_id"] for e in tm.generations_in_flight("C:T")] == ["genB"]
    # An already-finished id is a no-op (never disturbs the sibling).
    assert tm.finish_generation("C:T", "genA") is False
    assert [e["generation_id"] for e in tm.generations_in_flight("C:T")] == ["genB"]

    # Last one out drops the thread bucket entirely.
    assert tm.finish_generation("C:T", "genB") is True
    assert tm.generations_in_flight("C:T") == []
    assert "C:T" not in tm._active_generations


@pytest.mark.asyncio
async def test_watchdog_clears_only_stale_sibling():
    tm = AsyncThreadStateManager(db=None)
    tm.register_generation("C:T", "old", "a cat")
    tm.register_generation("C:T", "fresh", "a dog")
    # Backdate ONLY the old entry beyond the watchdog horizon.
    tm._active_generations["C:T"]["old"]["started_at"] -= (config.api_timeout_image + 60)
    remaining = tm.generations_in_flight("C:T")
    assert [e["generation_id"] for e in remaining] == ["fresh"]
    assert "old" not in tm._active_generations["C:T"]


# ------------------------------------------------------------- multi-generation upload latch

def test_upload_latch_sync_token_is_serial_and_idempotent():
    # A caller with no generation_id shares one "__sync__" token that releases idempotently.
    tm = AsyncThreadStateManager(db=None)
    key = "C:T"
    tm.mark_upload_started(key)
    tm.mark_upload_started(key)          # same token, deduped
    assert tm._upload_pending[key] == {"__sync__"}
    tm.mark_upload_finished(key)
    assert not tm._upload_pending.get(key)
    tm.mark_upload_finished(key)         # idempotent second release


# ------------------------------------------------------------------- suffix lists all entries

def test_inflight_suffix_lists_every_entry():
    tm = AsyncThreadStateManager(db=None)
    host = _SuffixHost(tm)
    tm.register_generation("C1:T1", "genA", "a red cat")
    tm.register_generation("C1:T1", "genB", "a blue dog")
    note = host._build_generation_inflight_note("C1", "T1")
    assert note is not None
    # Every in-flight summary is named, with plural phrasing.
    assert "a red cat" in note and "a blue dog" in note
    assert "2 images" in note
    assert "they are" in note


# ------------------------------------------------------------------------- wiring / config

def test_max_concurrent_config_default(monkeypatch):
    from config import BotConfig
    monkeypatch.delenv("MAX_CONCURRENT_IMAGE_GENERATIONS", raising=False)
    assert BotConfig().max_concurrent_image_generations == 5
