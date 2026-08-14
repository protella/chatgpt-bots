"""slack_client.base.SlackBot.download_file_async must accept and forward `max_bytes`.

The abstract (message_processor/client_contract.py) declares `max_bytes: Optional[int] = None`;
the concrete override had dropped it, so a caller passing a streaming byte cap would hit a
TypeError (or silently lose the cap on a looser call site). This pins the signature +
pass-through.

`allow_html` (canvases) rides the same wrapper and is pinned the same way — see
tests/unit/test_canvas_document.py for the canvas-side coverage.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from slack_client.base import SlackBot

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_download_file_async_forwards_max_bytes():
    download_file = AsyncMock(return_value=b"bytes")
    stub = SimpleNamespace(download_file=download_file)

    # Call the concrete method unbound against a stub self — it only touches self.download_file.
    out = await SlackBot.download_file_async(stub, "https://files.slack.com/x", "F1", max_bytes=4096)

    assert out == b"bytes"
    download_file.assert_awaited_once_with("https://files.slack.com/x", "F1",
                                           allow_html=False, max_bytes=4096)


@pytest.mark.asyncio
async def test_download_file_async_defaults_max_bytes_to_none():
    download_file = AsyncMock(return_value=b"bytes")
    stub = SimpleNamespace(download_file=download_file)

    await SlackBot.download_file_async(stub, "https://files.slack.com/x")

    download_file.assert_awaited_once_with("https://files.slack.com/x", None,
                                           allow_html=False, max_bytes=None)
