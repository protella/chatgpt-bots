"""F32 — channel-canvas creation must be serialized and fail closed.

A channel canvas is not idempotent: a second `conversations.canvases.create` means a second
permanent tab that can never be removed. A channel may hold several, so the guard is the TITLE.
Sibling tool calls in one round run concurrently (tool_registry gathers them) on shallow copies
of one context, so two create_channel_canvas calls with the same title could both pass the
duplicate check and each create a canvas. The fix is an asyncio.Lock around the check-then-create
with a per-turn record shared through the copies, plus a fail-CLOSED pre-check: if we cannot list
the channel's canvases, we refuse rather than risk a duplicate.
"""
import asyncio
import copy
from unittest.mock import AsyncMock, MagicMock

import pytest

from message_processor import canvas_tools as ct
from message_processor.tool_registry import ToolContext


def _ctx():
    web = MagicMock()
    web.files_list = AsyncMock(return_value={"files": []})
    web.conversations_info = AsyncMock(return_value={"channel": {"properties": {}}})
    web.files_info = AsyncMock(return_value={
        "file": {"id": "F123", "permalink": "https://slack.com/docs/F123"}})
    client = MagicMock()
    client.app = MagicMock()
    client.app.client = web
    return ToolContext(channel_id="C1", thread_ts="1.0", client=client), web


@pytest.mark.unit
class TestSerializedCreate:
    async def test_concurrent_duplicate_title_creates_make_only_one_canvas(self):
        # files.list and the tabs both lag the create, so only the per-turn record — shared by the
        # sibling calls' per-call copies — can see the first canvas when the second checks.
        ctx, web = _ctx()
        create_calls = 0

        async def fake_create(**kwargs):
            nonlocal create_calls
            create_calls += 1
            await asyncio.sleep(0)  # yield, so a sibling gets a chance to interleave the check
            return {"canvas_id": "F123"}

        web.conversations_canvases_create = fake_create

        out1, out2 = await asyncio.gather(
            ct.execute_create_channel_canvas(copy.copy(ctx), {"title": "P", "markdown": "x"}),
            ct.execute_create_channel_canvas(copy.copy(ctx), {"title": " p ", "markdown": "x"}),
        )

        results = [out1, out2]
        oks = [r for r in results if r.get("ok")]
        dupes = [r for r in results if r.get("error") == "duplicate_title"]
        assert create_calls == 1, "the lock must let exactly one create through"
        assert len(oks) == 1
        assert len(dupes) == 1

    async def test_precheck_failure_fails_closed(self, monkeypatch):
        # If we cannot verify whether a canvas exists, refuse — never create a possible duplicate.
        ctx, web = _ctx()
        web.files_list = AsyncMock(side_effect=RuntimeError("slack down"))
        web.conversations_canvases_create = AsyncMock(return_value={"canvas_id": "F999"})

        out = await ct.execute_create_channel_canvas(ctx, {"title": "P", "markdown": "x"})

        assert out["ok"] is False
        assert out["error"] == "check_failed"
        web.conversations_canvases_create.assert_not_awaited()
