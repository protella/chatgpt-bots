"""Phase 5 — wake classifier (classify_wake) output mapping + conservative failure default."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from openai_client.api.responses import classify_wake


class _FakeContent:
    def __init__(self, text):
        self.text = text


class _FakeItem:
    def __init__(self, text):
        self.content = [_FakeContent(text)]


class _FakeResp:
    def __init__(self, text):
        self.output = [_FakeItem(text)]


class _FakeLLM:
    """Stands in for the OpenAIClient `self` that classify_wake is bound to."""

    def __init__(self, text=None, exc=None):
        self._text = text
        self._exc = exc
        self.client = MagicMock()

    async def _safe_api_call(self, *a, **k):
        if self._exc:
            raise self._exc
        return _FakeResp(self._text)

    def log_debug(self, *a, **k):
        pass

    def log_warning(self, *a, **k):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,expected", [
    ("respond", "respond"),
    ("react", "react"),
    ("ignore", "ignore"),
    ("RESPOND", "respond"),
    ("  Ignore  ", "ignore"),
    ("I would respond to this", "respond"),
    ("", "ignore"),
    ("banana", "ignore"),  # unrecognized → conservative ignore
])
async def test_classify_wake_mapping(raw, expected):
    llm = _FakeLLM(text=raw)
    assert await classify_wake(llm, "some channel message") == expected


@pytest.mark.asyncio
async def test_classify_wake_defaults_to_ignore_on_error():
    llm = _FakeLLM(exc=RuntimeError("api down"))
    assert await classify_wake(llm, "anything") == "ignore"


@pytest.mark.asyncio
async def test_classify_wake_accepts_thread_signal():
    llm = _FakeLLM(text="respond")
    assert await classify_wake(llm, "hi", signals={"is_thread_reply": True}) == "respond"
