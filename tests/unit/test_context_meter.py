"""The context meter (CONTEXT_METER_SPEC §3.1-§3.5, §4 tests 1-4).

OpenAI's own input-token count is the only number a size decision is made on. These pin the
counter's request body, the thread meter's ordering rules, when preflight awaits a count, the
wrapper order it rides in, and the one recovery a turn gets when a request is over the window.
"""
from __future__ import annotations

import asyncio
import pathlib
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import config
from message_processor.client_contract import Message, Response
from message_processor.context_meter import ContextIrreducible, ContextOverLimit, MeterHook
from message_processor.thread_manager import ThreadState
from message_processor.turn_runtime import TurnRuntime
from openai_client.api import responses as R
from openai_client.api.token_count import COUNT_KEYS, CountResult, count_body, count_input_tokens

MODEL = "gpt-5.6-sol"


# ------------------------------------------------------------------ 1. the counter's request

def _create_kwargs() -> Dict[str, Any]:
    return {
        "model": MODEL, "input": [{"role": "user", "content": "hi"}], "instructions": "sys",
        "tools": [{"type": "web_search"}], "tool_choice": "auto", "parallel_tool_calls": True,
        "reasoning": {"effort": "low"}, "text": {"verbosity": "low"},
        # create-only keys the counter refuses
        "temperature": 1.0, "top_p": 1.0, "max_output_tokens": 100, "store": False,
        "stream": True, "include": ["reasoning.encrypted_content"],
        "prompt_cache_retention": "24h", "prompt_cache_key": "C1", "service_tier": "fast",
        "prompt_cache_options": {"ttl": "30m"},
    }


def test_count_body_keeps_exactly_the_sdk_keys_and_never_mutates():
    kwargs = _create_kwargs()
    snapshot = {k: v for k, v in kwargs.items()}
    body = count_body(kwargs)
    assert set(body) == {"model", "input", "instructions", "tools", "tool_choice",
                         "parallel_tool_calls", "reasoning", "text"}
    assert set(body) <= set(COUNT_KEYS)
    assert kwargs == snapshot
    # A new input list: the tool loop's later appends cannot reach a count already taken.
    assert body["input"] == kwargs["input"] and body["input"] is not kwargs["input"]


def _counter_client(side_effect: Any) -> Any:
    client = MagicMock()
    client.log_warning = MagicMock()

    async def _safe(method: Any, *, operation_type: str, **kw: Any) -> Any:
        assert operation_type == "token_count"
        return await method(**kw)

    client._safe_api_call = _safe
    client.client.responses.input_tokens.count = AsyncMock(side_effect=side_effect)
    return client


async def test_a_count_returns_the_official_number_complete():
    client = _counter_client([SimpleNamespace(input_tokens=1234)])
    assert await count_input_tokens(client, _create_kwargs()) == CountResult(1234, True)


async def test_an_mcp_failure_is_recounted_once_without_mcp_and_marked_incomplete():
    kwargs = {**_create_kwargs(), "tools": [{"type": "mcp", "server_label": "x"},
                                            {"type": "web_search"}]}
    client = _counter_client([RuntimeError("424 mcp"), SimpleNamespace(input_tokens=900)])
    assert await count_input_tokens(client, kwargs) == CountResult(900, False)
    retry_tools = client.client.responses.input_tokens.count.await_args_list[1].kwargs["tools"]
    assert retry_tools == [{"type": "web_search"}]


async def test_a_failed_count_is_none_and_never_raises():
    client = _counter_client([RuntimeError("boom")])
    assert await count_input_tokens(client, _create_kwargs()) == CountResult(None, False)


async def test_a_cancelled_count_propagates():
    client = _counter_client([asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await count_input_tokens(client, _create_kwargs())


# ------------------------------------------------------------------ 2. the thread meter

def _state() -> ThreadState:
    return ThreadState(thread_ts="10.0", channel_id="C1", current_model=MODEL)


def test_a_stale_seq_is_rejected():
    state = _state()
    s1, s2 = state.allocate_dispatch_seq(), state.allocate_dispatch_seq()
    assert state.record_measure(500, True, s2, state.meter_generation, MODEL, "usage")
    assert not state.record_measure(400, True, s1, state.meter_generation, MODEL, "count")
    assert state.current_measure(MODEL) == (500, True)


def test_usage_beats_count_at_the_same_seq_but_not_the_reverse():
    state = _state()
    seq = state.allocate_dispatch_seq()
    assert state.record_measure(480, False, seq, state.meter_generation, MODEL, "count")
    assert state.record_measure(500, True, seq, state.meter_generation, MODEL, "usage")
    assert not state.record_measure(470, True, seq, state.meter_generation, MODEL, "count")
    assert state.current_measure(MODEL) == (500, True)


def test_an_old_generation_is_rejected():
    state = _state()
    seq = state.allocate_dispatch_seq()
    old = state.meter_generation
    state.invalidate_measure()
    assert not state.record_measure(500, True, seq, old, MODEL, "usage")
    assert state.current_measure(MODEL) is None


def test_another_models_measure_is_not_current():
    state = _state()
    state.record_measure(500, True, state.allocate_dispatch_seq(), state.meter_generation,
                         "gpt-5.5", "usage")
    assert state.current_measure(MODEL) is None
    assert state.current_measure("gpt-5.5") == (500, True)


# ------------------------------------------------------------------ 3. preflight

class _Scheduled:
    """Collects what the hook schedules; closes the coroutines it never runs."""

    def __init__(self) -> None:
        self.coros: List[Any] = []

    def __call__(self, coro: Any) -> None:
        self.coros.append(coro)

    def close(self) -> None:
        for coro in self.coros:
            coro.close()


def _hook(state: ThreadState, *, tokens: Any = 100, heavy: bool = False) -> Any:
    client = MagicMock()
    client.count_input_tokens = AsyncMock(return_value=CountResult(tokens, True))
    scheduled = _Scheduled()
    hook = MeterHook(client=client, schedule=scheduled, thread_state=state, key="C1:10.0",
                     heavy=heavy)
    return hook, client, scheduled


def _measured(tokens: int, complete: bool = True) -> ThreadState:
    state = _state()
    state.record_measure(tokens, complete, state.allocate_dispatch_seq(),
                         state.meter_generation, MODEL, "usage")
    return state


@pytest.mark.parametrize("case", ["no_measure", "heavy", "over_threshold", "incomplete"])
async def test_preflight_awaits_a_count_when_the_margin_may_be_gone(case):
    limit = 1000
    state = {"no_measure": _state(), "heavy": _measured(100),
             "over_threshold": _measured(900), "incomplete": _measured(100, False)}[case]
    hook, client, scheduled = _hook(state, heavy=(case == "heavy"))
    with patch.object(config, "get_model_token_limit", return_value=limit), \
         patch.object(config, "token_cleanup_threshold", 0.5):
        await hook.preflight({"model": MODEL}, 0)
    client.count_input_tokens.assert_awaited_once()
    scheduled.close()


async def test_a_healthy_margin_awaits_nothing_and_counts_in_parallel():
    state = _measured(100)
    hook, client, scheduled = _hook(state)
    kwargs = {"model": MODEL, "input": []}
    with patch.object(config, "get_model_token_limit", return_value=1000), \
         patch.object(config, "token_cleanup_threshold", 0.5):
        await hook.preflight(kwargs, 0)
        client.count_input_tokens.assert_not_awaited()
        hook.dispatched(kwargs, 0)
        assert len(scheduled.coros) == 1
        await scheduled.coros.pop()
    client.count_input_tokens.assert_awaited_once()


async def test_preflight_runs_on_round_zero_only():
    hook, client, scheduled = _hook(_state())
    await hook.preflight({"model": MODEL}, 1)
    client.count_input_tokens.assert_not_awaited()
    scheduled.close()


def _wrapper_host(events: List[str]) -> Any:
    host = MagicMock()

    async def _safe(method: Any, **kw: Any) -> Any:
        events.append("create")
        return SimpleNamespace(output=[], usage=SimpleNamespace(
            input_tokens=321, output_tokens=5, input_tokens_details=None,
            output_tokens_details=None))

    host._safe_api_call = _safe
    return host


class _Sink:
    def __init__(self, events: List[str]) -> None:
        self.events = events

    def open(self, model: Any) -> Any:
        self.events.append("open")
        return SimpleNamespace(attempt_seq=1)

    def close(self, *a: Any, **k: Any) -> None:
        self.events.append("close")


async def test_the_wrapper_order_is_preflight_open_dispatched_create_usage():
    events: List[str] = []
    state = _state()
    hook, client, scheduled = _hook(state, tokens=100)
    real_dispatched, real_usage = hook.dispatched, hook.usage

    async def _preflight(kwargs: Any, round_index: int) -> None:
        events.append("preflight")

    def _dispatched(kwargs: Any, round_index: int) -> int:
        events.append("dispatched")
        return real_dispatched(kwargs, round_index)

    def _usage(seq: int, usage: Any) -> None:
        events.append("usage")
        real_usage(seq, usage)

    hook.preflight, hook.dispatched, hook.usage = _preflight, _dispatched, _usage
    await R.create_text_response(_wrapper_host(events), messages=[{"role": "user", "content": "x"}],
                                 model=MODEL, attempt_sink=_Sink(events), meter=hook)
    assert events == ["preflight", "open", "dispatched", "create", "usage", "close"]
    assert state.current_measure(MODEL) == (321, True)
    scheduled.close()


async def test_an_overflow_raises_before_any_attempt_is_opened():
    events: List[str] = []
    hook, _client, scheduled = _hook(_state(), tokens=5_000)
    with patch.object(config, "get_model_token_limit", return_value=1000):
        with pytest.raises(ContextOverLimit) as raised:
            await R.create_text_response(
                _wrapper_host(events), messages=[{"role": "user", "content": "x"}],
                model=MODEL, attempt_sink=_Sink(events), meter=hook)
    assert events == []
    assert raised.value.tokens == 5_000 and raised.value.kwargs["model"] == MODEL
    scheduled.close()


async def test_a_real_context_400_carries_the_wrappers_final_kwargs():
    hook, _client, scheduled = _hook(_measured(100))
    host = MagicMock()
    host._safe_api_call = AsyncMock(side_effect=RuntimeError(
        "Error code: 400 - context_length_exceeded"))
    with patch.object(config, "get_model_token_limit", return_value=1000):
        with pytest.raises(ContextOverLimit) as raised:
            await R.create_text_response(host, messages=[{"role": "user", "content": "x"}],
                                         model=MODEL, meter=hook, round_index=2)
    assert raised.value.tokens is None and raised.value.round_index == 2
    assert raised.value.kwargs["input"][-1]["content"] == "x"
    scheduled.close()


# ------------------------------------------------------------------ 4. the one recovery

def _recovery_host(outcome: str = "committed") -> Any:
    from message_processor.handlers.text import TextHandlerMixin

    host = MagicMock()
    host._recover_context_overflow = TextHandlerMixin._recover_context_overflow.__get__(host)
    host._compact_dm_for_recovery = AsyncMock(return_value=outcome)
    host._compact_channel_origin_for_recovery = AsyncMock(return_value="summary_failed")
    host.openai_client.count_input_tokens = AsyncMock(return_value=CountResult(None, False))
    return host


def _overflow(round_index: int = 0) -> ContextOverLimit:
    return ContextOverLimit(2_000, 1_000, kwargs={"model": MODEL}, round_index=round_index)


async def _recover(host: Any, overflow: ContextOverLimit, turn: Any, *,
                   committed: bool = False, channel: bool = False) -> None:
    await host._recover_context_overflow(
        overflow, turn=turn, thread_state=_state(), thread_key="D1:10.0",
        channel_turn=channel, committed=committed, own_messages=[])


async def test_the_budget_is_one_recovery_per_turn_shared_by_every_entry():
    host, turn = _recovery_host(), TurnRuntime()
    await _recover(host, _overflow(), turn)
    assert turn.context_recovery_used is True
    host._compact_dm_for_recovery.assert_awaited_once()
    # The buffered fallback (or any other re-entry) finds the budget spent on the TURN.
    with pytest.raises(ContextOverLimit):
        await _recover(host, _overflow(), turn)
    host._compact_dm_for_recovery.assert_awaited_once()


@pytest.mark.parametrize("round_index,committed", [(1, False), (0, True)])
async def test_no_recovery_after_round_zero_or_committed_text(round_index, committed):
    host, turn = _recovery_host(), TurnRuntime()
    with pytest.raises(ContextOverLimit):
        await _recover(host, _overflow(round_index), turn, committed=committed)
    host._compact_dm_for_recovery.assert_not_awaited()
    assert turn.context_recovery_used is False


async def test_an_irreducible_overflow_is_message_too_long():
    from message_processor.base import MessageProcessor

    host = _recovery_host("irreducible")
    with pytest.raises(ContextIrreducible) as raised:
        await _recover(host, _overflow(), TurnRuntime())
    msg = Message(text="x", user_id="U1", channel_id="D1", thread_id="10.0", metadata={})
    assert "Message Too Long" in MessageProcessor._turn_error_message(raised.value, msg)


async def test_a_failed_channel_summary_still_earns_the_one_retry():
    host = _recovery_host()
    await _recover(host, _overflow(), TurnRuntime(), channel=True)
    host._compact_channel_origin_for_recovery.assert_awaited_once()
    host._compact_dm_for_recovery.assert_not_awaited()


async def test_no_measure_anywhere_apportions_against_the_limit():
    host = _recovery_host()
    overflow = ContextOverLimit(None, 1_000, kwargs={"model": MODEL}, source="api")
    await _recover(host, overflow, TurnRuntime())
    host.openai_client.count_input_tokens.assert_awaited_once_with({"model": MODEL})
    assert host._compact_dm_for_recovery.await_args.kwargs["measure"] == 1_000


async def test_the_dm_adapter_sets_the_turns_own_message_aside_and_restores_it_once():
    from message_processor.thread_management import ThreadManagementMixin

    host = MagicMock()
    host._compact_dm_for_recovery = ThreadManagementMixin._compact_dm_for_recovery.__get__(host)
    state = _state()
    history = [{"role": "user", "content": f"m{i}"} for i in range(3)]
    own = {"role": "user", "content": "the question"}
    state.messages.extend([*history, own])
    seen: List[Any] = []

    async def _compact(thread_state: Any, key: str, measure: Any = None) -> int:
        seen.append(list(thread_state.messages))
        return 0

    host._compact_thread_to_target = _compact
    assert await host._compact_dm_for_recovery(state, "D1:10.0", [own], measure=5) == "irreducible"
    assert own not in seen[0]
    assert state.messages.count(own) == 1 and state.messages[-1] is own

    host._compact_thread_to_target = AsyncMock(side_effect=RuntimeError("summarizer down"))
    assert await host._compact_dm_for_recovery(state, "D1:10.0", [own]) == "summary_failed"
    assert state.messages.count(own) == 1


def _handler_host(loop_side_effect: Any) -> Any:
    """The REAL non-streaming `_handle_text_response` + recovery on a minimal host."""
    from message_processor.handlers.text import TextHandlerMixin

    host = _recovery_host()
    host._handle_text_response = TextHandlerMixin._handle_text_response.__get__(host)
    host._is_reaction_only = MagicMock(return_value=False)
    host._channel_request_too_large = TextHandlerMixin._channel_request_too_large
    host.db = None

    async def _passthru(m: Any, *a: Any, **k: Any) -> Any:
        return m

    async def _none(*a: Any, **k: Any) -> None:
        return None

    async def _empty(*a: Any, **k: Any) -> str:
        return ""

    host._inject_image_analyses = _passthru
    host._build_channel_info = _empty
    host._drop_dead_containers = _none
    host._resolve_ci_container = _none
    host._prepare_sandbox_tools = _none
    host._get_system_prompt = MagicMock(return_value="sys")
    host._build_suffix_context = MagicMock(return_value="")
    host._build_participant_roster = MagicMock(return_value="")
    host._build_tools_array = MagicMock(return_value=[{"type": "function", "name": "t"}])
    host._materialize_request_tools = MagicMock(
        return_value=(MagicMock(), {"model": MODEL}, True, None))
    host._build_tool_context = MagicMock(return_value=SimpleNamespace(
        background_job_started=False, sandbox_image_assets=[], mounted_files=[]))
    host._add_message_with_token_management = (
        lambda state, role, content, **kw: state.add_message(role, content, **kw))
    host.openai_client.create_text_response_with_tool_loop = AsyncMock(
        side_effect=loop_side_effect)
    return host


async def _run_dm_turn(host: Any, turn: Any) -> Response:
    message = Message(text="hi", user_id="U1", channel_id="D1", thread_id="10.0",
                      metadata={"ts": "10.0", "username": "Dana Whitfield"})

    async def fake_config(**kw: Any) -> Dict[str, Any]:
        return {"model": MODEL, "temperature": 1.0, "max_tokens": 100,
                "enable_streaming": False, "enable_code_interpreter": False}

    with patch.object(config, "get_thread_config_async", side_effect=fake_config):
        return await host._handle_text_response("hi", _state(), MagicMock(), message,
                                                thinking_id=None, turn=turn)


async def test_a_preflight_overflow_compacts_once_and_retries_once():
    ok = {"text": "answer", "tools_used": [], "local_tool_calls": []}
    host = _handler_host([_overflow(), ok])
    response = await _run_dm_turn(host, TurnRuntime())
    assert response.content.startswith("answer")
    host._compact_dm_for_recovery.assert_awaited_once()
    assert host.openai_client.create_text_response_with_tool_loop.await_count == 2


async def test_a_second_overflow_is_the_generic_error():
    from message_processor.base import MessageProcessor

    host = _handler_host([_overflow(), _overflow()])
    with pytest.raises(ContextOverLimit) as raised:
        await _run_dm_turn(host, TurnRuntime())
    host._compact_dm_for_recovery.assert_awaited_once()
    msg = Message(text="x", user_id="U1", channel_id="D1", thread_id="10.0", metadata={})
    assert "Something Went Wrong" in MessageProcessor._turn_error_message(raised.value, msg)


def test_the_too_much_for_one_request_card_is_gone():
    root = pathlib.Path(__file__).resolve().parents[2]
    for folder in ("message_processor", "openai_client", "slack_client"):
        for path in (root / folder).rglob("*.py"):
            assert "Too Much For One Request" not in path.read_text(encoding="utf-8"), path


# ------------------------------------------------------------------ fix round regressions

async def test_recovery_eligibility_is_turn_scoped_across_loop_reentry():
    """[codex 2] A re-entered loop starts at its own round 0; once a request of the TURN has
    completed, an overflow there is not recoverable and preflight no longer runs."""
    state = _measured(100)
    hook, client, scheduled = _hook(state)
    hook.request_completed()
    await hook.preflight({"model": MODEL}, 0)
    client.count_input_tokens.assert_not_awaited()
    overflow = hook.overflow_from(RuntimeError("400 context_length_exceeded"), {"model": MODEL}, 0)
    assert isinstance(overflow, ContextOverLimit) and overflow.recoverable is False

    host, turn = _recovery_host(), TurnRuntime()
    turn.context_meter = hook
    with pytest.raises(ContextOverLimit):
        await _recover(host, _overflow(), turn)
    host._compact_dm_for_recovery.assert_not_awaited()
    scheduled.close()


async def test_a_real_rejection_never_apportions_below_the_limit():
    """[codex 4] A stored measure (or recount) lower than the window the API just rejected must
    not turn recovery into a no-op: the limit is the floor."""
    host = _recovery_host()
    host.openai_client.count_input_tokens = AsyncMock(return_value=CountResult(300, False))
    state = _measured(200)
    overflow = ContextOverLimit(None, 1_000, kwargs={"model": MODEL}, source="api")
    await host._recover_context_overflow(overflow, turn=TurnRuntime(), thread_state=state,
                                         thread_key="D1:10.0", channel_turn=False,
                                         committed=False, own_messages=[])
    host.openai_client.count_input_tokens.assert_awaited_once_with({"model": MODEL})
    assert host._compact_dm_for_recovery.await_args.kwargs["measure"] == 1_000


def _streaming_host(loop: Any) -> Any:
    from message_processor.handlers.text import TextHandlerMixin

    host = _handler_host(None)
    host.handler = TextHandlerMixin._handle_streaming_text_response.__get__(host)
    for name in ("_as_mcp_exclusion_set", "_extract_failed_mcp_server", "_suspected_wedge"):
        setattr(host, name, getattr(TextHandlerMixin, name).__get__(host)
                if not isinstance(TextHandlerMixin.__dict__[name], staticmethod)
                else getattr(TextHandlerMixin, name))
    host._cleanup_silent_stream = AsyncMock()
    host._handle_text_response = AsyncMock(return_value=Response(type="text", content="retry"))
    host.openai_client.create_streaming_response_with_tool_loop = AsyncMock(side_effect=loop)
    return host


async def test_an_unrecoverable_streaming_overflow_still_cleans_up_its_partial():
    """[codex 1] The partial the attempt streamed is reconciled like any failed stream's — the
    seed it minted is deleted — and the overflow is raised instead of replayed."""
    async def _loop(*, stream_callback: Any, **_kw: Any) -> Dict[str, Any]:
        await stream_callback("a partial answer that reached the room ")
        raise ContextOverLimit(None, 1_000, kwargs={"model": MODEL}, round_index=1,
                               source="api", recoverable=False)

    host = _streaming_host(_loop)
    client = MagicMock()
    client.name = "Slack"
    client.supports_streaming = MagicMock(return_value=True)
    client.supports_native_streaming = MagicMock(return_value=False)
    client.get_streaming_config = MagicMock(
        return_value={"update_interval": 0.0, "buffer_size": 1, "min_interval": 0.0})
    client.send_message_get_ts = AsyncMock(return_value={"success": True, "ts": "SEED"})
    client.update_message_streaming = AsyncMock(return_value={"success": True})
    client.update_message = AsyncMock(return_value=True)
    client.delete_message = AsyncMock(return_value=True)
    message = Message(text="hi", user_id="U1", channel_id="D1", thread_id="10.0",
                      metadata={"ts": "10.0", "username": "Dana Whitfield"})

    async def fake_config(**kw: Any) -> Dict[str, Any]:
        return {"model": MODEL, "temperature": 1.0, "max_tokens": 100,
                "enable_streaming": True, "enable_code_interpreter": False}

    with patch.object(config, "get_thread_config_async", side_effect=fake_config):
        with pytest.raises(ContextOverLimit):
            await host.handler("hi", _state(), client, message, thinking_id=None,
                               turn=TurnRuntime())
    deleted = [c.args[1] for c in client.delete_message.await_args_list]
    assert "SEED" in deleted
    host._handle_text_response.assert_not_called()
