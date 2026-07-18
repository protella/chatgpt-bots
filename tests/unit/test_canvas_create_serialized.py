"""F32 — channel-canvas creation must be serialized and fail closed.

A channel canvas is not idempotent: a second `conversations.canvases.create` means a second
permanent tab that can never be removed. Sibling tool calls in one round run concurrently
(tool_registry gathers them), so two create_channel_canvas calls could both pass the
"does one already exist?" check and each create a canvas. The fix is an asyncio.Lock around the
check-then-create, plus a fail-CLOSED pre-check: if we cannot verify no canvas exists, we refuse
rather than risk a duplicate.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from message_processor import canvas_tools as ct
from tool_registry import ToolContext


def _ctx():
    web = MagicMock()
    web.files_list = AsyncMock(return_value={"files": []})
    web.files_info = AsyncMock(return_value={
        "file": {"id": "F123", "permalink": "https://slack.com/docs/F123"}})
    client = MagicMock()
    client.app = MagicMock()
    client.app.client = web
    return ToolContext(channel_id="C1", thread_ts="1.0", client=client), web


@pytest.mark.unit
class TestSerializedCreate:
    async def test_concurrent_creates_make_only_one_canvas(self, monkeypatch):
        ctx, web = _ctx()
        created: list[str] = []
        create_calls = 0

        async def fake_check(web_, channel_id, live):
            # Reflects a canvas made earlier in THIS run — the real check reads live Slack state.
            return created[0] if created else None

        async def fake_create(**kwargs):
            nonlocal create_calls
            create_calls += 1
            await asyncio.sleep(0)  # yield, so a sibling gets a chance to interleave the check
            created.append("F123")
            return {"canvas_id": "F123"}

        monkeypatch.setattr(ct, "_channel_canvas_id", fake_check)
        web.conversations_canvases_create = fake_create

        out1, out2 = await asyncio.gather(
            ct.execute_create_channel_canvas(ctx, {"title": "P", "markdown": "x"}),
            ct.execute_create_channel_canvas(ctx, {"title": "P", "markdown": "x"}),
        )

        results = [out1, out2]
        oks = [r for r in results if r.get("ok")]
        dupes = [r for r in results if r.get("error") == "already_exists"]
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
