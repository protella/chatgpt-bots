"""Phase F — classify_participation contract: strict-JSON verdict parsing and the
conservative fail-safe (any failure → {"action": "ignore"}).

Rewritten from the Phase-5 classify_wake tests (that classifier is deprecated and has
no runtime call sites; engine-level behavior lives in test_participation_engine.py).
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from openai_client.api.responses import classify_participation


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
    """Stands in for the OpenAIClient `self` that classify_participation is bound to."""

    def __init__(self, text=None, exc=None):
        self._text = text
        self._exc = exc
        self.client = MagicMock()
        self.captured_input = None

    async def _safe_api_call(self, *a, **k):
        if self._exc:
            raise self._exc
        self.captured_input = k.get("input")
        return _FakeResp(self._text)

    def log_debug(self, *a, **k):
        pass

    def log_warning(self, *a, **k):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,expected_action", [
    ('{"action": "respond", "placement": "thread", "reason": "asked us"}', "respond"),
    ('{"action": "react", "emoji": "thumbsup"}', "react"),
    ('{"action": "ignore"}', "ignore"),
    ('{"action": "backoff", "reason": "told to chill"}', "backoff"),
    # code fences / surrounding prose are tolerated
    ('```json\n{"action": "respond"}\n```', "respond"),
    ('Sure! Here is the verdict: {"action": "react", "emoji": "eyes"} hope that helps', "react"),
])
async def test_classify_participation_json_parsing(raw, expected_action):
    llm = _FakeLLM(text=raw)
    verdict = await classify_participation(llm, "some channel message")
    assert verdict["action"] == expected_action


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ["", "banana", "respond", "{not json}", "[]"])
async def test_classify_participation_garbage_defaults_ignore(raw):
    llm = _FakeLLM(text=raw)
    verdict = await classify_participation(llm, "anything")
    assert verdict == {"action": "ignore"}


@pytest.mark.asyncio
async def test_classify_participation_api_error_defaults_ignore():
    llm = _FakeLLM(exc=RuntimeError("api down"))
    assert await classify_participation(llm, "anything") == {"action": "ignore"}


@pytest.mark.asyncio
async def test_signals_render_into_prompt_deterministically():
    llm = _FakeLLM(text='{"action": "ignore"}')
    signals = {
        "sender_name": "Peter", "is_thread_reply": True, "strictness": "active",
        "directives": "only deploys", "unprompted_last_hour": 3, "hourly_cap": 12,
        "memory_facts": [{"id": 2, "content": "demos are Fridays"},
                         {"id": 1, "content": "Peter owns deploys"}],
        "channel_activity": "[Recent channel activity]\n- Peter (top-level): hi",
    }
    await classify_participation(llm, "msg", signals=dict(signals))
    first = llm.captured_input[1]["content"]
    await classify_participation(llm, "msg", signals=dict(signals))
    assert llm.captured_input[1]["content"] == first  # deterministic given same inputs
    assert "Sender: Peter" in first
    assert "Strictness: active" in first
    assert "only deploys" in first
    assert "[#1] Peter owns deploys; [#2] demos are Fridays" in first  # id-sorted
    assert "[Recent channel activity]" in first


@pytest.mark.asyncio
async def test_sender_is_bot_signal_renders_judgment_line():
    llm = _FakeLLM(text='{"action": "ignore"}')
    await classify_participation(llm, "msg", signals={"sender_is_bot": True})
    prompt = llm.captured_input[1]["content"]
    assert "another bot/agent" in prompt
    assert "use judgment" in prompt  # allowed, not banned
    # And absent the signal, the line stays out.
    llm2 = _FakeLLM(text='{"action": "ignore"}')
    await classify_participation(llm2, "msg", signals={})
    assert "another bot/agent" not in llm2.captured_input[1]["content"]


@pytest.mark.asyncio
async def test_deprecated_classify_wake_still_importable():
    # Kept one release for rollback; no runtime call sites.
    from openai_client.api.responses import classify_wake
    llm = _FakeLLM(exc=RuntimeError("api down"))
    assert await classify_wake(llm, "anything") == "ignore"


def test_prompt_carries_addressed_to_someone_else_rule():
    # Regression guard for the "hey claude, ..." bug (2026-07-10): a message that
    # names ANOTHER party is never for the assistant, however helpful it could be.
    from prompts import PARTICIPATION_SYSTEM_PROMPT
    assert "addressed to SOMEONE ELSE" in PARTICIPATION_SYSTEM_PROMPT
    assert "hey claude" in PARTICIPATION_SYSTEM_PROMPT


def test_prompt_carries_addressee_precedence_over_name_hit():
    # Regression guard (2026-07-10): "claude, do you still have the chatgpt bot's
    # repo checked out?" — the alias prefilter flags "chatgpt" (a possessive topic
    # ref) as a name hit, and the model must not let that outrank the opener
    # naming another party. Both the rule and the name_hit signal carry it.
    from prompts import PARTICIPATION_SYSTEM_PROMPT
    assert "the chatgpt bot's repo" in PARTICIPATION_SYSTEM_PROMPT
    import inspect
    from openai_client.api import responses
    src = inspect.getsource(responses.classify_participation)
    assert "DIFFERENT party" in src


def test_prompt_carries_second_person_continuity_rule():
    # Regression guard for the "how does your background workspace work?" bug
    # (2026-07-10): an unnamed "you"-follow-up mid-exchange with another
    # participant continues THAT exchange — it is not an invitation to jump in.
    from prompts import PARTICIPATION_SYSTEM_PROMPT
    assert '"You" belongs to whoever the sender has been talking to' in PARTICIPATION_SYSTEM_PROMPT
    assert "helpful third voice" in PARTICIPATION_SYSTEM_PROMPT


def test_participation_uses_dedicated_reasoning_effort():
    # Referent resolution fails at effort=none (verified live 2026-07-10), so the
    # participation call has its own knob, defaulting to "low" — without dragging
    # the rest of the utility calls (intent classification) up with it.
    import config as config_module
    assert config_module.config.participation_reasoning_effort  # field exists, non-empty
    import inspect
    from openai_client.api import responses
    src = inspect.getsource(responses.classify_participation)
    assert "participation_reasoning_effort" in src
    assert "utility_reasoning_effort" not in src
