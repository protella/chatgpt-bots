"""Phase 4 — emoji reactions as a response.

Covers: the client react() capability (success / already_reacted / error / disabled / colon
stripping), the Response.reaction() helper, and the handle_response 'reaction' branch.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from slack_sdk.errors import SlackApiError

from base_client import Response
from config import config
from slack_client.messaging import SlackMessagingMixin


class _MsgClient(SlackMessagingMixin):
    def __init__(self, client):
        self.app = SimpleNamespace(client=client)

    def log_debug(self, *a, **k):
        pass

    log_info = log_warning = log_error = log_debug


def _api_error(code):
    return SlackApiError(code, {"error": code})


@pytest.mark.asyncio
async def test_react_success(monkeypatch):
    monkeypatch.setattr(config, "enable_reactions", True)
    client = SimpleNamespace(reactions_add=AsyncMock())
    host = _MsgClient(client)
    assert await host.react("C1", "123.45", "eyes") is True
    client.reactions_add.assert_awaited_with(channel="C1", name="eyes", timestamp="123.45")


@pytest.mark.asyncio
async def test_react_strips_colons(monkeypatch):
    monkeypatch.setattr(config, "enable_reactions", True)
    client = SimpleNamespace(reactions_add=AsyncMock())
    host = _MsgClient(client)
    await host.react("C1", "1", ":thumbsup:")
    _, kwargs = client.reactions_add.call_args
    assert kwargs["name"] == "thumbsup"


@pytest.mark.asyncio
async def test_react_already_reacted_is_success(monkeypatch):
    monkeypatch.setattr(config, "enable_reactions", True)
    client = SimpleNamespace(reactions_add=AsyncMock(side_effect=_api_error("already_reacted")))
    host = _MsgClient(client)
    assert await host.react("C1", "1", "eyes") is True  # idempotent


@pytest.mark.asyncio
async def test_react_other_error_returns_false(monkeypatch):
    monkeypatch.setattr(config, "enable_reactions", True)
    client = SimpleNamespace(reactions_add=AsyncMock(side_effect=_api_error("message_not_found")))
    host = _MsgClient(client)
    assert await host.react("C1", "1", "eyes") is False


@pytest.mark.asyncio
async def test_react_disabled(monkeypatch):
    monkeypatch.setattr(config, "enable_reactions", False)
    client = SimpleNamespace(reactions_add=AsyncMock())
    host = _MsgClient(client)
    assert await host.react("C1", "1", "eyes") is False
    client.reactions_add.assert_not_called()


@pytest.mark.asyncio
async def test_react_empty_emoji_returns_false(monkeypatch):
    monkeypatch.setattr(config, "enable_reactions", True)
    client = SimpleNamespace(reactions_add=AsyncMock())
    host = _MsgClient(client)
    assert await host.react("C1", "1", "::") is False
    client.reactions_add.assert_not_called()


def test_response_reaction_helper():
    r = Response.reaction("eyes", target_ts="55.5")
    assert r.type == "reaction"
    assert r.content == "eyes"
    assert r.metadata["react_ts"] == "55.5"

    r2 = Response.reaction(["tada", "thumbsup"])
    assert r2.type == "reaction" and r2.content == ["tada", "thumbsup"]
    assert r2.metadata == {}


@pytest.mark.asyncio
async def test_handle_response_reaction_branch(monkeypatch):
    monkeypatch.setattr(config, "enable_reactions", True)
    host = _MsgClient(SimpleNamespace())
    host.react = AsyncMock(return_value=True)
    await host.handle_response("C1", "T1", Response.reaction(["eyes", "tada"], target_ts="55.5"))
    assert host.react.await_count == 2
    host.react.assert_any_await("C1", "55.5", "eyes")
    host.react.assert_any_await("C1", "55.5", "tada")


@pytest.mark.asyncio
async def test_handle_response_reaction_defaults_to_thread_root(monkeypatch):
    monkeypatch.setattr(config, "enable_reactions", True)
    host = _MsgClient(SimpleNamespace())
    host.react = AsyncMock(return_value=True)
    await host.handle_response("C1", "T1", Response.reaction("eyes"))  # no target_ts
    host.react.assert_any_await("C1", "T1", "eyes")
