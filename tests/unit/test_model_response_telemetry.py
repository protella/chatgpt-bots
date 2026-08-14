"""CV8 `model_response` — one ledger row per Responses API ATTEMPT on a channel turn.

WHY THESE AND NOT OTHERS. The event exists to answer "how many API calls did this turn cost, and
which of them were re-runs of work already paid for". That makes CARDINALITY the whole contract:
one row per request, never two for one call and never zero for a call that raised. So the tests
here pin the count first — across a tool loop's rounds, across a container retry's second request,
across a failure — and only then the fields.

The other half is that none of this may cost a turn. A DM passes `attempt_sink=None` and must
produce nothing at all; a sink that blows up must not change what the API layer raises.

Pitfall #6: every mock stream is a finite generator yielding real strings, so a stale side_effect
can never spin an unbounded async iterator.
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import config
from message_processor import participation_telemetry as pt
from message_processor.handlers import text as text_handler
from message_processor.turn_runtime import TurnRuntime
from openai_client.api import responses as R
from openai_client.api import tool_loop
from message_processor.tool_registry import ToolContext, ToolRegistry


# --------------------------------------------------------------------------------- fixtures

@pytest.fixture
def rows(tmp_path):
    """The ledger, pointed at a temp dir, with a reader for its `model_response` lines.

    Same reset discipline as test_participation_telemetry: the sink logger comes out of logging's
    global registry, so a handler left behind by an earlier test would write this test's rows into
    that test's file. Only OUR handlers are touched.
    """
    named = logging.getLogger(pt._SINK_LOGGER_NAME)
    saved = named.handlers[:]
    pt.shutdown()
    with patch.object(config, "log_directory", str(tmp_path)), \
            patch.object(config, "enable_participation_telemetry", True):
        pt.initialize()

        def read(event="model_response"):
            pt._drain()   # the listener owns the file; reading without this races the thread
            path = tmp_path / pt.LOG_NAME
            if not path.exists():
                return []
            lines = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
            return [r for r in lines if event is None or r["event"] == event]

        try:
            yield read
        finally:
            pt.shutdown()
    named.handlers = saved


def _fake_client():
    fake = MagicMock()
    fake.log_info = fake.log_debug = fake.log_warning = fake.log_error = lambda *a, **k: None
    fake._safe_api_call = AsyncMock(return_value=SimpleNamespace(output=[], usage=None))
    return fake


def _stream(events):
    async def _iter(response, op):
        for e in events:
            yield e
    return _iter


def _usage(inp, out, cached=None):
    """Provider usage. `cached is None` means the provider sent NO `input_tokens_details` at all —
    which is the shape every pre-existing usage_sink assertion is written against."""
    if cached is None:
        return SimpleNamespace(input_tokens=inp, output_tokens=out)
    return SimpleNamespace(input_tokens=inp, output_tokens=out,
                           input_tokens_details=SimpleNamespace(cached_tokens=cached))


def _sink(fork_reason=None):
    turn = TurnRuntime()
    return turn, pt.ModelAttemptSink(turn=turn, fork_reason=fork_reason)


def _swallow(chunk):
    """A stream_callback that keeps nothing: these tests are about the ledger, not the text."""
    return None


# ------------------------------------------------------------------- one row per attempt, ok

@pytest.mark.asyncio
async def test_successive_calls_on_one_turn_sequence_1_2_3(rows):
    """attempt_seq is per TURN, which is the only counter that can answer "what did this turn
    cost". A process-wide counter would make two rows from two turns look like one expensive one.
    """
    fake = _fake_client()
    fake._safe_api_call = AsyncMock(
        return_value=SimpleNamespace(output=[], usage=_usage(10, 2)))
    turn, sink = _sink()

    for _ in range(3):
        await R.create_text_response(
            fake, messages=[{"role": "user", "content": "hi"}],
            model="gpt-5.6-sol", attempt_sink=sink)

    written = rows()
    assert [r["attempt_seq"] for r in written] == [1, 2, 3]
    assert {r["turn_id"] for r in written} == {turn.turn_id}
    assert {r["status"] for r in written} == {"ok"}
    assert {r["model"] for r in written} == {"gpt-5.6-sol"}


@pytest.mark.asyncio
async def test_row_carries_the_usage_numbers_including_cached(rows):
    """`cached_input_tokens` is the ONLY evidence the pinned-prefix cache key is doing anything,
    so it has to reach the row rather than only the budgeter."""
    fake = _fake_client()
    fake._safe_api_call = AsyncMock(
        return_value=SimpleNamespace(output=[], usage=_usage(900, 40, cached=768)))
    _turn, sink = _sink()
    usage_sink = {}

    await R.create_text_response(
        fake, messages=[{"role": "user", "content": "hi"}], model="gpt-5.6-sol",
        usage_sink=usage_sink, attempt_sink=sink)

    row = rows()[0]
    assert (row["input_tokens"], row["output_tokens"], row["cached_input_tokens"]) == (900, 40, 768)
    # The caller's sink gains the same key — budgeting and the ledger read one capture.
    assert usage_sink == {"input_tokens": 900, "output_tokens": 40, "cached_input_tokens": 768}


@pytest.mark.asyncio
async def test_cached_tokens_absent_when_the_provider_sent_no_details(rows):
    """An absent key and a zero are different facts: one says the provider reported nothing about
    the cache, the other says it reported a miss. `record()` omits None, so absence is absence."""
    fake = _fake_client()
    fake._safe_api_call = AsyncMock(
        return_value=SimpleNamespace(output=[], usage=_usage(11, 7)))
    _turn, sink = _sink()
    usage_sink = {}

    await R.create_text_response(
        fake, messages=[{"role": "user", "content": "hi"}], model="gpt-5.6-sol",
        usage_sink=usage_sink, attempt_sink=sink)

    row = rows()[0]
    assert "cached_input_tokens" not in row
    assert usage_sink == {"input_tokens": 11, "output_tokens": 7}


@pytest.mark.asyncio
async def test_zero_cached_tokens_is_written_as_a_reported_miss(rows):
    fake = _fake_client()
    fake._safe_api_call = AsyncMock(
        return_value=SimpleNamespace(output=[], usage=_usage(11, 7, cached=0)))
    _turn, sink = _sink()

    await R.create_text_response(
        fake, messages=[{"role": "user", "content": "hi"}], model="gpt-5.6-sol",
        attempt_sink=sink)

    # 0 is falsy but not None, and `record()` only omits None — a reported miss stays on the row.
    assert rows()[0]["cached_input_tokens"] == 0


@pytest.mark.asyncio
async def test_model_named_is_the_one_actually_sent(rows):
    """The caller may pass model=None and let config resolve it. The row has to say what was
    asked, not what the call site happened to hold."""
    fake = _fake_client()
    _turn, sink = _sink()

    await R.create_text_response(
        fake, messages=[{"role": "user", "content": "hi"}], model=None, attempt_sink=sink)

    assert rows()[0]["model"] == config.gpt_model


# -------------------------------------------------------------------------------- failures

@pytest.mark.asyncio
async def test_api_failure_writes_error_and_reraises_unchanged(rows):
    """The expensive attempts are exactly the ones that raised, so a failure must be ON the
    record — and the exception must reach the caller untouched, because the container-recovery
    and MCP-failover paths above this layer dispatch on its type."""
    fake = _fake_client()
    boom = RuntimeError("upstream boom")
    fake._safe_api_call = AsyncMock(side_effect=boom)
    _turn, sink = _sink()

    with pytest.raises(RuntimeError) as caught:
        await R.create_text_response(
            fake, messages=[{"role": "user", "content": "hi"}],
            model="gpt-5.6-sol", attempt_sink=sink)
    assert caught.value is boom

    row = rows()[0]
    assert row["status"] == "error"
    assert row["detail"] == "RuntimeError"
    assert row["attempt_seq"] == 1
    assert "input_tokens" not in row     # nothing came back to count


@pytest.mark.asyncio
async def test_failed_stream_is_an_error_row_carrying_its_usage(rows):
    """A `response.failed` terminal is a failure that still reports usage. Both halves belong on
    the row: it cost tokens AND it did not answer."""
    fake = _fake_client()
    fake._safe_stream_iteration = _stream([
        SimpleNamespace(type="response.output_text.delta", delta="half"),
        SimpleNamespace(type="response.failed", response=SimpleNamespace(
            usage=_usage(30, 4, cached=16),
            error=SimpleNamespace(code="server_error", message="kaput"))),
    ])
    _turn, sink = _sink()

    with pytest.raises(RuntimeError, match="kaput"):
        await R.create_streaming_response(
            fake, messages=[{"role": "user", "content": "hi"}],
            stream_callback=_swallow, model="gpt-5.6-sol", attempt_sink=sink)

    row = rows()[0]
    assert row["status"] == "error" and row["detail"] == "RuntimeError"
    assert (row["input_tokens"], row["cached_input_tokens"]) == (30, 16)


@pytest.mark.asyncio
async def test_incomplete_stream_is_a_successful_attempt(rows):
    """An incomplete response is a truncation, not a failed request: the call came back and its
    partial text is real. Filing it as an error would make every long answer look like an outage.
    """
    fake = _fake_client()
    fake._safe_stream_iteration = _stream([
        SimpleNamespace(type="response.output_text.delta", delta="partial"),
        SimpleNamespace(type="response.incomplete", response=SimpleNamespace(
            usage=_usage(5, 2),
            incomplete_details=SimpleNamespace(reason="max_output_tokens"))),
    ])
    _turn, sink = _sink()

    text = await R.create_streaming_response(
        fake, messages=[{"role": "user", "content": "hi"}],
        stream_callback=_swallow, model="gpt-5.6-sol", attempt_sink=sink)

    assert text == "partial"
    assert [(r["status"], r["output_tokens"]) for r in rows()] == [("ok", 2)]


@pytest.mark.asyncio
async def test_exactly_one_row_when_the_stream_succeeds_then_something_after_it_raises(rows):
    """The row describes the REQUEST. Once the call came back ok, a later failure in the same
    wrapper must not rewrite it as an error, and must not add a second row for one call."""
    fake = _fake_client()
    fake._safe_stream_iteration = _stream([
        SimpleNamespace(type="response.completed", response=SimpleNamespace(usage=_usage(8, 1))),
    ])
    _turn, sink = _sink()

    def _explode(chunk):
        if chunk is None:
            raise KeyError("late")     # swallowed by the wrapper's callback guard
        return None

    await R.create_streaming_response(
        fake, messages=[{"role": "user", "content": "hi"}],
        stream_callback=_explode, model="gpt-5.6-sol", attempt_sink=sink)

    assert [r["status"] for r in rows()] == ["ok"]


@pytest.mark.asyncio
async def test_a_successful_call_made_from_inside_an_except_block_is_still_ok(rows):
    """The `finally` reads the propagating exception rather than being handed one, and these
    wrappers ARE called from inside except clauses (the MCP failover, the context retry). An
    unrelated outer exception must not repaint a successful attempt as an error."""
    fake = _fake_client()
    fake._safe_api_call = AsyncMock(
        return_value=SimpleNamespace(output=[], usage=_usage(9, 3)))
    _turn, sink = _sink()

    try:
        raise KeyError("the failure that caused the retry")
    except KeyError:
        await R.create_text_response(
            fake, messages=[{"role": "user", "content": "hi"}],
            model="gpt-5.6-sol", attempt_sink=sink)

    assert [(r["status"], r.get("detail")) for r in rows()] == [("ok", None)]


# ------------------------------------------------------------------------ the no-op contract

@pytest.mark.asyncio
async def test_no_sink_writes_nothing(rows):
    """DMs and every direct caller pass None. Nothing may be written, on success or on failure."""
    fake = _fake_client()
    fake._safe_api_call = AsyncMock(
        return_value=SimpleNamespace(output=[], usage=_usage(10, 2)))
    await R.create_text_response(
        fake, messages=[{"role": "user", "content": "hi"}], model="gpt-5.6-sol")

    fake._safe_api_call = AsyncMock(side_effect=RuntimeError("x"))
    with pytest.raises(RuntimeError):
        await R.create_text_response(
            fake, messages=[{"role": "user", "content": "hi"}], model="gpt-5.6-sol")

    assert rows() == []


@pytest.mark.asyncio
async def test_a_broken_sink_never_changes_what_the_api_layer_does(rows):
    """A telemetry failure must cost a line, never a turn — including on the error path, where the
    exception the caller sees decides whether the turn recovers."""
    class _Hostile:
        def open(self, model=None):
            raise ValueError("no")

        def close(self, *a, **k):
            raise ValueError("no")

    fake = _fake_client()
    fake._safe_api_call = AsyncMock(
        return_value=SimpleNamespace(output=[], usage=_usage(4, 1)))
    assert await R.create_text_response(
        fake, messages=[{"role": "user", "content": "hi"}],
        model="gpt-5.6-sol", attempt_sink=_Hostile()) == ""

    boom = TimeoutError("slow")
    fake._safe_api_call = AsyncMock(side_effect=boom)
    with pytest.raises(TimeoutError) as caught:
        await R.create_text_response(
            fake, messages=[{"role": "user", "content": "hi"}],
            model="gpt-5.6-sol", attempt_sink=_Hostile())
    assert caught.value is boom
    assert rows() == []


# ------------------------------------------------------------------ one row per tool-loop round

def _fn_call_item(call_id="c1", name="echo"):
    return SimpleNamespace(type="function_call", call_id=call_id, name=name, arguments="{}")


def _registry():
    reg = ToolRegistry()

    async def echo(ctx, args):
        return {"ok": True}

    reg.register({"type": "function", "name": "echo", "description": "t",
                  "parameters": {"type": "object"}}, echo)
    return reg


@pytest.mark.asyncio
async def test_three_round_tool_loop_writes_three_rows(rows):
    """Each round of the loop is its OWN Responses call, so it is its own attempt. The loop
    forwards `**params`, which is how the sink reaches every round without the loop knowing what
    a turn is — and a single row here would under-report a tool-heavy turn by its whole cost."""
    fake = _fake_client()
    fake._safe_api_call = AsyncMock(side_effect=[
        SimpleNamespace(output=[_fn_call_item("c1")], usage=_usage(100, 5)),
        SimpleNamespace(output=[_fn_call_item("c2")], usage=_usage(200, 6)),
        SimpleNamespace(output=[], usage=_usage(300, 7)),
    ])
    _turn, sink = _sink()

    result = await tool_loop.create_text_response_with_tool_loop(
        fake, messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "name": "echo"}],
        registry=_registry(), tool_context=ToolContext(channel_id="C1"),
        model="gpt-5.6-sol", attempt_sink=sink)

    assert [(c["name"], c["ok"]) for c in result["local_tool_calls"]] == [
        ("echo", True), ("echo", True)]
    written = rows()
    assert [(r["attempt_seq"], r["input_tokens"]) for r in written] == [
        (1, 100), (2, 200), (3, 300)]
    assert {r["status"] for r in written} == {"ok"}


@pytest.mark.asyncio
async def test_a_round_that_fails_is_the_last_row_and_is_an_error(rows):
    fake = _fake_client()
    fake._safe_api_call = AsyncMock(side_effect=[
        SimpleNamespace(output=[_fn_call_item("c1")], usage=_usage(100, 5)),
        ValueError("round two died"),
    ])
    _turn, sink = _sink()

    with pytest.raises(ValueError, match="round two died"):
        await tool_loop.create_text_response_with_tool_loop(
            fake, messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "name": "echo"}],
            registry=_registry(), tool_context=ToolContext(channel_id="C1"),
            model="gpt-5.6-sol", attempt_sink=sink)

    assert [(r["attempt_seq"], r["status"], r.get("detail")) for r in rows()] == [
        (1, "ok", None), (2, "error", "ValueError")]


# ------------------------------------------------------- the container retry is a second attempt

@pytest.mark.asyncio
async def test_container_retry_writes_two_rows_one_per_request(rows):
    """`_create_with_container_recovery` retries the SAME call once with demoted tools. That is a
    second HTTP request and therefore a second attempt: the first is closed as the error it was,
    the second as the success it became. Folding them into one row would report a turn as costing
    one call when the provider was paid for two."""
    fake = _fake_client()
    dead = RuntimeError("Container with id 'ci_dead' not found.")
    fake._safe_api_call = AsyncMock(side_effect=[
        dead, SimpleNamespace(output=[], usage=_usage(50, 3))])
    _turn, sink = _sink()
    gone = []

    await R.create_text_response_with_tools(
        fake, messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "code_interpreter", "container": "ci_dead"}],
        model="gpt-5.6-sol", container_gone_sink=gone, attempt_sink=sink)

    assert gone == ["ci_dead"]
    assert [(r["attempt_seq"], r["status"], r.get("detail")) for r in rows()] == [
        (1, "error", "RuntimeError"), (2, "ok", None)]
    assert rows()[1]["input_tokens"] == 50


@pytest.mark.asyncio
async def test_the_container_retry_names_its_own_fork(rows):
    """[f21] The demoted retry exists BECAUSE the sandbox container died — one of the four
    documented cache-fork exceptions. Inheriting the entry's reason (usually none at all) left the
    attempt that lost the container and the attempt that replaced it indistinguishable, so the one
    fork this layer can see was the one nothing could count."""
    fake = _fake_client()
    fake._safe_api_call = AsyncMock(side_effect=[
        RuntimeError("Container with id 'ci_dead' not found."),
        SimpleNamespace(output=[], usage=_usage(50, 3))])
    _turn, sink = _sink()

    await R.create_text_response_with_tools(
        fake, messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "code_interpreter", "container": "ci_dead"}],
        model="gpt-5.6-sol", attempt_sink=sink)

    first, second = rows()
    assert "fork_reason" not in first          # the attempt that HAD the container forked nothing
    assert second["fork_reason"] == R.FORK_CONTAINER_RECOVERY == "container_recovery"


@pytest.mark.asyncio
async def test_a_fork_named_by_the_layer_does_not_overwrite_the_entrys_reason(rows):
    """The entry-wide reason still rides every other attempt: a container recovery INSIDE an MCP
    retry is both, and only the one request it describes may carry the narrower name."""
    fake = _fake_client()
    fake._safe_api_call = AsyncMock(side_effect=[
        RuntimeError("Container with id 'ci_dead' not found."),
        SimpleNamespace(output=[], usage=_usage(5, 1)),
        SimpleNamespace(output=[], usage=_usage(5, 1))])
    _turn, sink = _sink(fork_reason=text_handler.FORK_MCP_RETRY)

    await R.create_text_response_with_tools(
        fake, messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "code_interpreter", "container": "ci_dead"}],
        model="gpt-5.6-sol", attempt_sink=sink)
    await R.create_text_response(fake, messages=[{"role": "user", "content": "again"}],
                                 model="gpt-5.6-sol", attempt_sink=sink)

    assert [r["fork_reason"] for r in rows()] == [
        "mcp_retry", "container_recovery", "mcp_retry"]


# --------------------------------------------------- attach-time utility calls are attempts too

@pytest.mark.asyncio
async def test_an_attached_documents_summary_is_an_attempt_on_the_turn(rows):
    """[f20] The attach-time summarizer is a Responses API call the turn pays for, so the CV8
    contract — one `model_response` per attempt on a channel turn — is only true if it writes a row.
    It ran unrecorded, which under-reported spend on exactly the turns that spend the most: a turn
    with three documents made four calls and reported one.
    """
    from message_processor.utilities import MessageUtilitiesMixin as U

    fake = _fake_client()
    fake._safe_api_call = AsyncMock(
        return_value=SimpleNamespace(output=[], usage=_usage(4000, 120)))

    class _OpenAI:
        async def create_text_response(self, **kwargs):
            return await R.create_text_response(fake, **kwargs)

    host = MagicMock()
    host.openai_client = _OpenAI()
    host.log_warning = host.log_info = host.log_debug = lambda *a, **k: None
    host.finalize_deferred_documents = U.finalize_deferred_documents.__get__(host)
    host._finalize_document_summary = U._finalize_document_summary.__get__(host)
    host._summarize_document_for_attach = U._summarize_document_for_attach.__get__(host)

    turn = TurnRuntime()
    entries = [{"filename": f"q{i}.pdf", "mimetype": "application/pdf",
                "_persist": {"extracted": {"content": "revenue " * 50}, "thread_id": "C1:10.0",
                             "message_ts": "10.0", "url_private": "u", "size_bytes": 40}}
               for i in range(3)]
    await host.finalize_deferred_documents(
        entries, MagicMock(), SimpleNamespace(channel_id="C1", thread_id="10.0"), None,
        reserves=(), turn=turn)

    written = rows()
    assert [r["attempt_seq"] for r in written] == [1, 2, 3]
    assert {r["turn_id"] for r in written} == {turn.turn_id}
    # Named by the UTILITY model, which is how the ledger tells three documents from three retries.
    assert {r["model"] for r in written} == {config.utility_model}
    assert {r["status"] for r in written} == {"ok"}
    # Not a fork: this is work the turn owes, not a re-run of work already paid for.
    assert not any("fork_reason" in r for r in written)


@pytest.mark.asyncio
async def test_a_dm_document_summary_carries_no_sink_at_all(rows):
    """The DM half of the same contract. `turn=None` means nothing to sequence against, and a DM
    request that grew a keyword would stop being byte-identical to the recorded baseline."""
    from message_processor.utilities import MessageUtilitiesMixin as U, attach_summary_attempt_sink

    assert attach_summary_attempt_sink(None) is None

    host = MagicMock()
    host.openai_client.create_text_response = AsyncMock(return_value="a summary")
    host.log_warning = lambda *a, **k: None
    host._summarize_document_for_attach = U._summarize_document_for_attach.__get__(host)
    assert await host._summarize_document_for_attach(
        {"content": "prose " * 50}, "a.pdf", "application/pdf") == "a summary"
    assert host.openai_client.create_text_response.await_args.kwargs["attempt_sink"] is None
    assert rows() == []


# ------------------------------------------------------------------------------ fork_reason

@pytest.mark.asyncio
async def test_fork_reason_rides_every_attempt_of_a_forked_entry(rows):
    """It says why THIS handler entry is issuing calls at all, so it is fixed for the entry and
    identical on every row it produces — a per-call reason could not be grouped."""
    fake = _fake_client()
    fake._safe_api_call = AsyncMock(side_effect=[
        SimpleNamespace(output=[_fn_call_item("c1")], usage=_usage(10, 1)),
        SimpleNamespace(output=[], usage=_usage(20, 2)),
    ])
    _turn, sink = _sink(fork_reason=text_handler.FORK_MCP_RETRY)

    await tool_loop.create_text_response_with_tool_loop(
        fake, messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "name": "echo"}],
        registry=_registry(), tool_context=ToolContext(channel_id="C1"),
        model="gpt-5.6-sol", attempt_sink=sink)

    assert [r["fork_reason"] for r in rows()] == ["mcp_retry", "mcp_retry"]


@pytest.mark.asyncio
async def test_first_ordinary_attempt_has_no_fork_reason(rows):
    fake = _fake_client()
    _turn, sink = _sink()
    await R.create_text_response(
        fake, messages=[{"role": "user", "content": "hi"}],
        model="gpt-5.6-sol", attempt_sink=sink)
    # None is omitted rather than written as null, so a group-by never gets two empty buckets.
    assert "fork_reason" not in rows()[0]


class TestForkReasonPrecedence:
    """A deterministic order, pinned: a named failure beats the shape of the re-entry, which beats
    a bare retry. Two entries that disagree about which name a fork carries would split one rate
    across two buckets."""

    def test_ordinary_entry(self):
        assert text_handler._fork_reason() is None

    def test_mcp_failure_outranks_everything(self):
        assert text_handler._fork_reason(
            retry_count=1, retry_timeout=60.0, failed_mcp_server="reportpro",
            context_retry=True, nonstreaming_fallback=True) == "mcp_retry"

    def test_context_retry_outranks_the_fallback_shape(self):
        assert text_handler._fork_reason(
            retry_count=1, retry_timeout=60.0, context_retry=True,
            nonstreaming_fallback=True) == "context_retry"

    def test_fallback_shape_outranks_a_bare_retry(self):
        assert text_handler._fork_reason(
            retry_count=1, retry_timeout=60.0, nonstreaming_fallback=True
        ) == "nonstreaming_fallback"

    def test_timeout_retry_is_the_unnamed_buffered_retry(self):
        assert text_handler._fork_reason(retry_count=1, retry_timeout=60.0) == "timeout_retry"

    def test_retry_covers_a_retry_that_armed_no_ceiling(self):
        assert text_handler._fork_reason(retry_count=1) == "retry"


class TestSinkConstruction:
    """DM turns get NO sink. That is what keeps the DM request byte-identical: there is nothing to
    forward, so nothing about the request can differ."""

    def test_dm_turn_gets_none(self):
        assert text_handler._model_attempt_sink(TurnRuntime(), False, None) is None

    def test_channel_turn_without_a_turn_gets_none(self):
        assert text_handler._model_attempt_sink(None, True, None) is None

    def test_channel_turn_gets_a_sink_bound_to_the_turn(self):
        turn = TurnRuntime()
        sink = text_handler._model_attempt_sink(turn, True, "retry")
        assert sink.turn is turn and sink.fork_reason == "retry"


# ----------------------------------------------------------- the paths that carry the keyword

@pytest.mark.asyncio
async def test_every_request_issuing_path_writes_a_row(rows):
    """One row per path, on one turn, so the sequence numbers double as a checklist: every wrapper
    a channel turn can reach must be able to carry the sink. A path that silently dropped it would
    make a turn's cost look smaller than it was, and nothing else would notice."""
    fake = _fake_client()
    fake._safe_api_call = AsyncMock(
        return_value=SimpleNamespace(output=[], usage=_usage(1, 1)))
    _turn, sink = _sink()
    msgs = [{"role": "user", "content": "hi"}]

    await R.create_text_response(fake, messages=msgs, model="m", attempt_sink=sink)
    await R.create_text_response_with_tools(
        fake, messages=msgs, tools=[{"type": "web_search"}], model="m", attempt_sink=sink)
    await R._create_text_response_with_tools_with_timeout(
        fake, messages=msgs, tools=[{"type": "web_search"}], model="m",
        timeout_seconds=30.0, attempt_sink=sink)
    # The no-tools timeout twin has never had a usage_sink; its row still carries the numbers,
    # read for the ledger alone so the caller's budgeting is untouched.
    await R._create_text_response_with_timeout(
        fake, messages=msgs, model="m", timeout_seconds=30.0, attempt_sink=sink)

    fake._safe_stream_iteration = _stream([
        SimpleNamespace(type="response.completed", response=SimpleNamespace(usage=_usage(1, 1))),
    ])
    await R.create_streaming_response(
        fake, messages=msgs, stream_callback=_swallow, model="m", attempt_sink=sink)
    await R.create_streaming_response_with_tools(
        fake, messages=msgs, tools=[{"type": "web_search"}], stream_callback=_swallow,
        model="m", attempt_sink=sink)

    written = rows()
    assert [r["attempt_seq"] for r in written] == [1, 2, 3, 4, 5, 6]
    assert {r["status"] for r in written} == {"ok"}
    assert {r["input_tokens"] for r in written} == {1}
