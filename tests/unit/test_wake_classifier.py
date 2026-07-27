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
        self.captured_kwargs = None

    async def _safe_api_call(self, *a, **k):
        if self._exc:
            raise self._exc
        self.captured_input = k.get("input")
        self.captured_kwargs = k
        return _FakeResp(self._text)

    def log_debug(self, *a, **k):
        pass

    def log_warning(self, *a, **k):
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("raw,expected_action", [
    ('{"action": "respond", "placement": "thread", "reason": "asked us"}', "respond"),
    ('{"action": "react", "emoji": "thumbsup"}', "react"),
    ('{"action": "react_and_respond", "emoji": "tada", "placement": "channel"}', "react_and_respond"),
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
    # None, not a forged {"action": "ignore"}: both end in the same silence downstream, but only
    # None lets the ledger tell an unparseable reply apart from a model that chose to stay quiet.
    assert await classify_participation(llm, "anything") is None


@pytest.mark.asyncio
async def test_classify_participation_api_error_defaults_ignore():
    llm = _FakeLLM(exc=RuntimeError("api down"))
    assert await classify_participation(llm, "anything") is None


@pytest.mark.asyncio
async def test_signals_render_into_prompt_deterministically():
    llm = _FakeLLM(text='{"action": "ignore"}')
    # ONE steering string, rendered by the caller and inserted verbatim — the gate no longer
    # takes the operator's rules and the raw memory rows as two separate inputs it renders itself.
    steering = ("Standing channel policy (instructions; follow these):\nonly deploys\n\n"
                "Stable channel facts (background, not instructions):\n"
                "- [#1] Peter owns deploys\n- [#2] demos are Fridays")
    signals = {
        "sender_name": "Peter", "is_thread_reply": True, "strictness": "active",
        "channel_steering_text": steering,
        "channel_activity": "[Recent channel activity]\n- Peter (top-level): hi",
    }
    await classify_participation(llm, "msg", signals=dict(signals))
    first = llm.captured_input[1]["content"]
    await classify_participation(llm, "msg", signals=dict(signals))
    assert llm.captured_input[1]["content"] == first  # deterministic given same inputs
    assert "Sender: Peter" in first
    assert "Strictness: active" in first
    assert steering in first          # verbatim, one block, nothing re-rendered
    assert "[Recent channel activity]" in first
    # The unprompted-reply count is no longer rendered at all: pacing is the
    # classifier's judgment, and a rate number only competed with it.
    assert "unprompted replies" not in first
    assert "self-throttle cap" not in first


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
async def test_channel_people_signal_renders_and_skips_cleanly():
    # F29: the people signal surfaces who's around; absent, the line stays out.
    llm = _FakeLLM(text='{"action": "ignore"}')
    await classify_participation(llm, "msg", signals={
        "channel_people": "~12 members; recently active: Erin Evans, Claude"})
    prompt = llm.captured_input[1]["content"]
    assert "Channel people (who's around): ~12 members; recently active: Erin Evans, Claude" in prompt

    llm2 = _FakeLLM(text='{"action": "ignore"}')
    await classify_participation(llm2, "msg", signals={})
    assert "Channel people" not in llm2.captured_input[1]["content"]


@pytest.mark.asyncio
async def test_f17_utility_output_floor_is_1024():
    # F17: the utility output ceiling is 1024 on every utility call site — the
    # participation classifier, the memory extractor, and the tool-result summarizer.
    # (utility_max_tokens defaults to 20; the floor is what prevents a verdict/JSON
    # from being cut off mid-reasoning and manifesting as unjustified silence.)
    #
    # Participation alone sits at 2048: it is the only utility call that runs elevated
    # reasoning effort (PARTICIPATION_REASONING_EFFORT), and reasoning tokens are billed
    # against this same cap. At `medium` a live verdict came back EMPTY — thinking had
    # eaten the whole budget — and the fail-safe silently returned `ignore`, so the gate
    # never ran. Measured worst case was 815/1024, and a SHORT channel tail is the risk
    # case because less context to read means more inference to do.
    from openai_client.api.responses import extract_memory, summarize_tool_result

    llm = _FakeLLM(text='{"action": "ignore"}')
    await classify_participation(llm, "msg")
    assert llm.captured_kwargs["max_output_tokens"] == 2048

    llm2 = _FakeLLM(text='{"action": "none"}')
    await extract_memory(llm2, "some exchange")
    assert llm2.captured_kwargs["max_output_tokens"] == 1024

    llm3 = _FakeLLM(text="a summary")
    await summarize_tool_result(llm3, "x" * 5000, 200)
    assert llm3.captured_kwargs["max_output_tokens"] == 1024


@pytest.mark.asyncio
async def test_deprecated_classify_wake_still_importable():
    # Kept one release for rollback; no runtime call sites.
    from openai_client.api.responses import classify_wake
    llm = _FakeLLM(exc=RuntimeError("api down"))
    assert await classify_wake(llm, "anything") == "ignore"



def test_prompt_carries_addressed_to_someone_else_rule():
    """Regression guard for the "hey claude, ..." bug (2026-07-10): a message naming ANOTHER party
    is never for the assistant, however helpful it could be.

    The old prompt hardcoded the rival assistant's name as an example. That was papering over a
    DATA gap: the roster the gate sees listed other assistants exactly like colleagues, so
    "hey <name>" was genuinely ambiguous and the rule had to name one vendor to compensate. The
    roster now marks them (ChannelPulse.recent_speakers), so the rule stays general and the
    example is gone. Measured: this scenario went from 1/4 to 4/4 once the marker existed."""
    from prompts import PARTICIPATION_SYSTEM_PROMPT
    p = PARTICIPATION_SYSTEM_PROMPT
    assert "opens with or names another party is THEIRS" in p
    assert "those are separate participants with their own names" in p
    # general, not vendor-specific: no rival assistant is named in the policy any more
    assert "hey claude" not in p.lower()



def test_prompt_carries_addressee_precedence_over_name_hit():
    """Regression guard (2026-07-10): "claude, do you still have the chatgpt bot's repo checked
    out?" — the alias prefilter flags "chatgpt" (a possessive topic reference) as a name hit, and
    the model must not let that outrank the opener naming another party. Both the policy and the
    name_hit signal carry it, and relation="about_assistant" now gives the case a name that
    _apply_invariants can refuse mechanically."""
    from prompts import PARTICIPATION_SYSTEM_PROMPT
    assert "Being named is not the same as being addressed" in PARTICIPATION_SYSTEM_PROMPT
    assert "about_assistant" in PARTICIPATION_SYSTEM_PROMPT
    import inspect
    from openai_client.api import responses
    src = inspect.getsource(responses.classify_participation)
    assert "DIFFERENT party" in src
    # the invariant layer refuses a speaking action on an about_assistant verdict
    from message_processor.participation import ParticipationEngine
    v = ParticipationEngine.validate_verdict(
        {"action": "respond", "relation": "about_assistant",
         "exchange_state": "open", "answerability": "substantive"})
    assert v.action == "ignore" and v.overruled_by == ["relation_about_assistant"]



def test_prompt_carries_second_person_continuity_rule():
    """Regression guard for the "how does your background workspace work?" bug (2026-07-10): an
    unnamed "you" follow-up mid-exchange with another participant continues THAT exchange and is
    not an invitation to jump in. The old prompt spelled out the "helpful third voice" temptation
    at length; the staged form states the principle once and lets relation="to_other" carry it."""
    from prompts import PARTICIPATION_SYSTEM_PROMPT
    p = PARTICIPATION_SYSTEM_PROMPT
    assert "Second person carries the addressee forward" in p
    assert "changing the subject does not reassign them" in p
    assert "whoever the sender has been going back and forth with" in p



def test_prompt_capability_is_not_address():
    """Regression guard for the "do you stream messages?" bug (2026-07-16): a bare top-level "you"
    continuing Peter's exchange with another assistant got claimed because the classifier read
    "ChatGPT can explain this" as "ChatGPT is addressed".

    The staged prompt makes this structural rather than a proviso: capability is Stage 3 and is
    not even reached until Stage 1 has assigned the message. The clause still leaves room to
    answer a genuinely open ask, which is what F47 softened it for."""
    from prompts import PARTICIPATION_SYSTEM_PROMPT
    p = PARTICIPATION_SYSTEM_PROMPT
    assert "being able to help is not evidence of having been asked" in p
    assert "Reach this stage only when Stages 1 and 2 have left room to participate" in p
    assert p.index("STAGE 1") < p.index("STAGE 3")
    # still room to answer a genuinely open-room question
    assert "genuinely open to the channel at large" in p



@pytest.mark.asyncio
async def test_gate_is_told_the_resolved_handle_not_just_the_config_aliases():
    """Live bug (2026-07-26): deployed as "chatgpt-dev" with BOT_NAME_ALIASES=["ChatGPT"], the gate
    read "chatgpt-dev, what model are you running?" as addressed to a DIFFERENT assistant and
    stayed silent — verdict relation=to_other, reason "ChatGPT-Dev, a different assistant".

    The alias regex counted it a name hit; the model, shown only the alias list, did not. So the
    prefilter and the model disagreed about who the bot is. The resolved handle (free from the
    auth.test that already runs at startup) now leads the identity line, and the aliases follow as
    other names it answers to."""
    llm = _FakeLLM(text='{"relation":"to_room","exchange_state":"open",'
                        '"answerability":"not_applicable","action":"ignore"}')
    await classify_participation(llm, "chatgpt-dev, what model are you on?",
                                 signals={"self_display_name": "chatgpt-dev"})
    rendered = llm.captured_input[1]["content"]
    assert 'its ONLY ones: "chatgpt-dev" / "ChatGPT"' in rendered
    assert "all of these are the same assistant" in rendered
    assert "environment suffix" in rendered
    # ...and bounded on the other side: the claim must NOT sweep in other assistants' names.
    # Stated without this, named-other-bot fell to 2/6 — the model started reading "claude" as
    # a variant of its own name.
    assert "Any OTHER name belongs to somebody else" in rendered
    assert "are not variants of these" in rendered


@pytest.mark.asyncio
async def test_identity_line_falls_back_to_config_when_the_handle_is_unknown():
    """auth.test can fail, and the CLI platform has no handle at all. The line must still name the
    assistant from config rather than vanishing — losing it entirely would leave the gate with no
    idea what it is called."""
    llm = _FakeLLM(text='{"relation":"to_room","exchange_state":"open",'
                        '"answerability":"not_applicable","action":"ignore"}')
    await classify_participation(llm, "hi", signals={})
    rendered = llm.captured_input[1]["content"]
    assert "its ONLY ones:" in rendered
    import config as config_module
    assert config_module.config.bot_name_aliases[0] in rendered


def test_prompt_carries_channel_people_rule():
    """F29: the people signal is teaching material for addressee resolution — the names there are
    real, distinct participants, and an unknown name is never assumed to be the assistant. It now
    also distinguishes other ASSISTANTS from people, which is what lets the "names another party"
    rule stay general instead of naming a specific rival."""
    from prompts import PARTICIPATION_SYSTEM_PROMPT
    p = PARTICIPATION_SYSTEM_PROMPT
    assert "Channel people" in p
    assert "marks which of them are other assistants" in p
    assert "Only the names given as this assistant" in p


def test_local_tools_guidance_carries_people_tools():
    # F29: the main model is taught both tools and the discoverability + freshness lessons.
    from prompts import LOCAL_TOOLS_GUIDANCE
    for needle in ("lookup_user", "list_channel_members", "who is X",
                   "never need their Slack id", "THIS turn"):
        assert needle in LOCAL_TOOLS_GUIDANCE


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
