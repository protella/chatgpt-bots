"""Fix-wave regression tests: bug-hunt + async-audit fixes.

Covers: the upload-in-flight latch (race: "edit it" right after a generation),
analysis persistence for live-generated/edited images, safe fire-and-forget task
scheduling, the un-dead-ed text-timeout retry condition, and CI-checkable source
scans (no time.sleep in async code, no sync DB calls inside async defs, no bare
print in runtime modules).
"""
from __future__ import annotations

import ast
import asyncio
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from thread_manager import AsyncThreadStateManager
from message_processor.utilities import MessageUtilitiesMixin

REPO = Path(__file__).resolve().parents[2]

RUNTIME_FILES = [
    "main.py", "base_client.py", "config.py", "database.py", "thread_manager.py",
    "token_counter.py", "document_handler.py", "image_url_handler.py",
    "markdown_converter.py",
] + [str(p.relative_to(REPO)) for p in (REPO / "message_processor").rglob("*.py")] \
  + [str(p.relative_to(REPO)) for p in (REPO / "slack_client").rglob("*.py")] \
  + [str(p.relative_to(REPO)) for p in (REPO / "openai_client").rglob("*.py")]


# --------------------------------------------------------------------- upload latch

class TestUploadLatch:
    async def test_wait_returns_immediately_when_no_upload(self):
        mgr = AsyncThreadStateManager()
        await asyncio.wait_for(mgr.wait_for_uploads("C1:1"), timeout=1)

    async def test_wait_blocks_until_finished(self):
        mgr = AsyncThreadStateManager()
        mgr.mark_upload_started("C1:1")
        order = []

        async def editor():
            await mgr.wait_for_uploads("C1:1")
            order.append("edit")

        async def uploader():
            await asyncio.sleep(0.05)
            order.append("upload")
            mgr.mark_upload_finished("C1:1")

        await asyncio.gather(editor(), uploader())
        assert order == ["upload", "edit"]  # edit resolved only after upload landed

    async def test_wait_times_out_and_proceeds(self):
        mgr = AsyncThreadStateManager()
        mgr.mark_upload_started("C1:1")
        # never finished — bounded wait must return, not hang
        await asyncio.wait_for(mgr.wait_for_uploads("C1:1", timeout=0.1), timeout=1)

    async def test_latch_is_per_thread(self):
        mgr = AsyncThreadStateManager()
        mgr.mark_upload_started("C1:1")
        await asyncio.wait_for(mgr.wait_for_uploads("C2:2"), timeout=1)  # other thread unaffected


# ------------------------------------------------- analysis persisted for live images

class _Proc(MessageUtilitiesMixin):
    def log_debug(self, *a, **k): pass
    def log_info(self, *a, **k): pass
    def log_warning(self, *a, **k): pass
    def log_error(self, *a, **k): pass


async def test_update_last_image_url_persists_analysis_from_metadata():
    proc = _Proc.__new__(_Proc)
    proc.db = MagicMock()
    proc.db.save_image_metadata_async = AsyncMock()

    state = SimpleNamespace(
        thread_ts="1", channel_id="C1",
        messages=[{
            "role": "assistant",
            "content": "edited",
            "metadata": {"type": "image_edit", "prompt": "a red dog",
                         "original_analysis": "a dog wearing a red collar", "url": None},
        }],
    )
    proc.thread_manager = MagicMock()
    proc.thread_manager.get_or_create_thread_async = AsyncMock(return_value=state)
    proc.thread_manager.get_asset_ledger = MagicMock(return_value=None)

    await proc.update_last_image_url("C1", "1", "https://files.slack.com/x.png")

    kwargs = proc.db.save_image_metadata_async.call_args.kwargs
    assert kwargs["analysis"] == "a dog wearing a red collar"  # was hardcoded ""
    assert kwargs["original_analysis"] == "a dog wearing a red collar"


async def test_update_last_image_url_falls_back_to_newest_ledger_entry():
    proc = _Proc.__new__(_Proc)
    proc.db = MagicMock()
    proc.db.save_image_metadata_async = AsyncMock()

    state = SimpleNamespace(
        thread_ts="1", channel_id="C1",
        messages=[{"role": "assistant", "content": "made it",
                   "metadata": {"type": "image_generation", "prompt": "a cat", "url": None}}],
    )
    ledger = SimpleNamespace(images=[
        {"analysis": "OLD image analysis"},
        {"analysis": "a fluffy cat on a rug"},
    ])
    proc.thread_manager = MagicMock()
    proc.thread_manager.get_or_create_thread_async = AsyncMock(return_value=state)
    proc.thread_manager.get_asset_ledger = MagicMock(return_value=ledger)

    await proc.update_last_image_url("C1", "1", "https://files.slack.com/y.png")

    kwargs = proc.db.save_image_metadata_async.call_args.kwargs
    assert kwargs["analysis"] == "a fluffy cat on a rug"  # newest entry only, never older ones


# ------------------------------------------------------------- safe task scheduling

class _Sched(MessageUtilitiesMixin):
    def __init__(self):
        self.errors = []
    def log_error(self, msg, **k): self.errors.append(msg)
    def log_debug(self, *a, **k): pass


async def test_schedule_async_call_logs_background_exception():
    s = _Sched()

    async def boom():
        raise RuntimeError("memory extraction failed")

    task = s._schedule_async_call(boom())
    await asyncio.sleep(0.05)
    assert task.done()
    assert any("memory extraction failed" in e for e in s.errors)  # was silently swallowed


async def test_schedule_async_call_keeps_strong_reference():
    s = _Sched()

    async def slow():
        await asyncio.sleep(0.02)

    task = s._schedule_async_call(slow())
    assert task in s._background_tasks  # not GC-able mid-flight
    await task
    await asyncio.sleep(0.01)
    assert task not in s._background_tasks  # cleaned up on completion


# ------------------------------------------------------------- retry condition (fix 3)

def test_text_timeout_retry_condition_reachable():
    """Mirror of the fixed should_retry expression: text_normal retries even though
    `intent` is always assigned by routing; intent_classification keeps its guard."""
    def should_retry(operation_type, already_retried, intent_in_locals):
        return (
            operation_type in ['text_normal', 'intent_classification']
            and not already_retried
            and (operation_type != 'intent_classification' or not intent_in_locals)
        )

    # The dead-code bug: text_normal + intent assigned -> must now retry
    assert should_retry('text_normal', False, True) is True
    assert should_retry('text_normal', True, True) is False       # no infinite loops
    assert should_retry('intent_classification', False, False) is True
    assert should_retry('intent_classification', False, True) is False  # guard kept
    assert should_retry('vision', False, False) is False

    # And the source must actually carry the scoped guard
    src = (REPO / "message_processor" / "base.py").read_text()
    assert "operation_type != 'intent_classification' or 'intent' not in locals()" in src


# ------------------------------------------------------------------ source-scan gates

def _async_function_spans(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            yield node


def test_no_time_sleep_in_async_functions():
    offenders = []
    for rel in RUNTIME_FILES:
        p = REPO / rel
        if not p.exists():
            continue
        tree = ast.parse(p.read_text())
        for fn in _async_function_spans(tree):
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "sleep"
                        and isinstance(node.func.value, ast.Name) and node.func.value.id == "time"):
                    offenders.append(f"{rel}:{node.lineno}")
    assert offenders == [], f"time.sleep inside async def: {offenders}"


_SYNC_DB_METHODS = {
    "get_or_create_thread", "save_thread_config", "get_thread_config",
    "get_channel_settings", "set_channel_settings", "get_channel_memory",
    "add_channel_memory", "update_channel_memory", "delete_channel_memory",
    "get_thread_summary", "save_thread_summary", "save_image_metadata",
    "get_images_by_message", "find_thread_images", "get_user_preferences",
    "get_thread_documents", "cleanup_old_modal_sessions",
}


def test_no_sync_db_calls_inside_async_defs():
    """Every DB call inside an async def must use an *_async twin (audit fix #9)."""
    offenders = []
    for rel in RUNTIME_FILES:
        p = REPO / rel
        if not p.exists() or p.name == "database.py":
            continue
        tree = ast.parse(p.read_text())
        for fn in _async_function_spans(tree):
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in _SYNC_DB_METHODS):
                    # only flag calls on something named/ending in 'db' or config-with-db pattern
                    target = node.func.value
                    name = getattr(target, "attr", getattr(target, "id", ""))
                    if name == "db":
                        offenders.append(f"{rel}:{node.lineno} .{node.func.attr}()")
    assert offenders == [], f"sync DB calls on the event loop: {offenders}"


def test_no_bare_print_in_runtime_modules():
    offenders = []
    for rel in RUNTIME_FILES:
        p = REPO / rel
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if re.match(r"^\s*print\(", line):
                offenders.append(f"{rel}:{i}")
    assert offenders == [], f"bare print() in runtime modules: {offenders}"
