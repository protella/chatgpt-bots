"""F9: streaming terminal handling for response.incomplete / response.failed, and F33: MCP
sink parity on the custom-timeout tools path.

Pitfall #6: every mock stream here is a finite generator yielding real strings, so a stale
side_effect can never spin an unbounded async iterator.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from openai_client.api import responses as R


def _fake_client():
    fake = MagicMock()
    fake.log_info = fake.log_debug = fake.log_warning = fake.log_error = lambda *a, **k: None
    fake._safe_api_call = AsyncMock(return_value=SimpleNamespace())
    return fake


def _stream(events):
    async def _iter(response, op):
        for e in events:
            yield e
    return _iter


class _Recorder:
    """Records streamed chunks and whether the final flush (stream_callback(None)) fired."""
    def __init__(self):
        self.chunks = []
        self.flushed = False

    def __call__(self, chunk):
        if chunk is None:
            self.flushed = True
        else:
            self.chunks.append(chunk)


def _usage(inp, out):
    return SimpleNamespace(input_tokens=inp, output_tokens=out)


# ---------------------------------------------------------------- create_streaming_response

@pytest.mark.asyncio
async def test_incomplete_flushes_and_returns_partial_no_tools():
    fake = _fake_client()
    fake._safe_stream_iteration = _stream([
        SimpleNamespace(type="response.output_text.delta", delta="partial "),
        SimpleNamespace(type="response.output_text.delta", delta="answer"),
        SimpleNamespace(type="response.incomplete", response=SimpleNamespace(
            usage=_usage(11, 7),
            incomplete_details=SimpleNamespace(reason="max_output_tokens"))),
    ])

    rec = _Recorder()
    usage_sink = {}
    text = await R.create_streaming_response(
        fake, messages=[{"role": "user", "content": "hi"}],
        stream_callback=rec, model="gpt-5.6-sol", usage_sink=usage_sink)

    assert text == "partial answer"
    assert rec.flushed is True                     # final flush fired despite non-success
    assert usage_sink == {"input_tokens": 11, "output_tokens": 7}  # usage captured


@pytest.mark.asyncio
async def test_failed_flushes_then_raises_no_tools():
    fake = _fake_client()
    fake._safe_stream_iteration = _stream([
        SimpleNamespace(type="response.output_text.delta", delta="oops"),
        SimpleNamespace(type="response.failed", response=SimpleNamespace(
            usage=_usage(3, 1),
            error=SimpleNamespace(code="server_error", message="upstream boom"))),
    ])

    rec = _Recorder()
    usage_sink = {}
    with pytest.raises(RuntimeError, match="upstream boom"):
        await R.create_streaming_response(
            fake, messages=[{"role": "user", "content": "hi"}],
            stream_callback=rec, model="gpt-5.6-sol", usage_sink=usage_sink)

    assert rec.flushed is True                     # buffer flushed even on failure
    assert usage_sink == {"input_tokens": 3, "output_tokens": 1}


# --------------------------------------------------- create_streaming_response_with_tools

@pytest.mark.asyncio
async def test_incomplete_flushes_and_captures_usage_with_tools():
    fake = _fake_client()
    fake._safe_stream_iteration = _stream([
        SimpleNamespace(type="response.output_text.delta", delta="draft"),
        SimpleNamespace(type="response.incomplete", response=SimpleNamespace(
            usage=_usage(20, 9),
            incomplete_details=SimpleNamespace(reason="max_output_tokens"))),
    ])

    rec = _Recorder()
    usage_sink = {}
    text = await R.create_streaming_response_with_tools(
        fake, messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "web_search"}], stream_callback=rec,
        model="gpt-5.6-sol", usage_sink=usage_sink)

    assert text == "draft"
    assert rec.flushed is True
    assert usage_sink == {"input_tokens": 20, "output_tokens": 9}


@pytest.mark.asyncio
async def test_failed_flushes_then_raises_with_tools():
    fake = _fake_client()
    fake._safe_stream_iteration = _stream([
        SimpleNamespace(type="response.output_text.delta", delta="x"),
        SimpleNamespace(type="response.failed", response=SimpleNamespace(
            usage=None,
            error={"code": "rate_limit", "message": "slow down"})),
    ])

    rec = _Recorder()
    with pytest.raises(RuntimeError, match="slow down"):
        await R.create_streaming_response_with_tools(
            fake, messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "web_search"}], stream_callback=rec, model="gpt-5.6-sol")

    assert rec.flushed is True


def _fn_call_done(call_id="c1", name="foo"):
    return SimpleNamespace(
        type="response.output_item.done",
        item=SimpleNamespace(type="function_call", call_id=call_id, name=name, arguments="{}"))


@pytest.mark.asyncio
async def test_normal_completion_with_function_call_defers_flush():
    # Baseline semantics preserved: a NORMAL completion carrying a local function call does not
    # flush — the tool loop will run another round, so buffered text isn't final yet.
    fake = _fake_client()
    fake._safe_stream_iteration = _stream([
        _fn_call_done(),
        SimpleNamespace(type="response.completed", response=SimpleNamespace(usage=None)),
    ])
    rec = _Recorder()
    sink = []
    await R.create_streaming_response_with_tools(
        fake, messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "web_search"}], stream_callback=rec,
        model="gpt-5.6-sol", function_call_sink=sink)

    assert rec.flushed is False          # deferred — another round is coming
    assert len(sink) == 1                # the function call was collected


@pytest.mark.asyncio
async def test_incomplete_with_function_call_still_flushes():
    # T1-9: an INCOMPLETE terminal has no next round, so it must flush even when a function
    # call was seen this round — otherwise the buffer stays stuck.
    fake = _fake_client()
    fake._safe_stream_iteration = _stream([
        _fn_call_done(),
        SimpleNamespace(type="response.incomplete", response=SimpleNamespace(
            usage=_usage(5, 2),
            incomplete_details=SimpleNamespace(reason="max_output_tokens"))),
    ])
    rec = _Recorder()
    usage_sink = {}
    sink = []
    await R.create_streaming_response_with_tools(
        fake, messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "web_search"}], stream_callback=rec,
        model="gpt-5.6-sol", function_call_sink=sink, usage_sink=usage_sink)

    assert rec.flushed is True           # flushed despite the function call
    assert usage_sink == {"input_tokens": 5, "output_tokens": 2}


@pytest.mark.asyncio
async def test_failed_with_function_call_still_flushes_and_raises():
    fake = _fake_client()
    fake._safe_stream_iteration = _stream([
        _fn_call_done(),
        SimpleNamespace(type="response.failed", response=SimpleNamespace(
            usage=None, error=SimpleNamespace(code="server_error", message="kaput"))),
    ])
    rec = _Recorder()
    with pytest.raises(RuntimeError, match="kaput"):
        await R.create_streaming_response_with_tools(
            fake, messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "web_search"}], stream_callback=rec,
            model="gpt-5.6-sol", function_call_sink=[])
    assert rec.flushed is True


@pytest.mark.asyncio
async def test_failed_with_no_error_detail_still_raises():
    fake = _fake_client()
    fake._safe_stream_iteration = _stream([
        SimpleNamespace(type="response.failed", response=SimpleNamespace(usage=None, error=None)),
    ])
    rec = _Recorder()
    with pytest.raises(RuntimeError, match="no error detail"):
        await R.create_streaming_response(
            fake, messages=[{"role": "user", "content": "hi"}],
            stream_callback=rec, model="gpt-5.6-sol")
    assert rec.flushed is True


# ------------------------------------------------- F33: timeout tools twin MCP sink parity

def _mcp_item(server_label, output, error=None):
    return SimpleNamespace(type="mcp_call", server_label=server_label,
                           output=output, error=error, content=None)


@pytest.mark.asyncio
async def test_timeout_twin_captures_mcp_result_and_discovery():
    """The custom-timeout retry path must harvest MCP results AND tool discovery into the same
    sinks the non-timeout twin fills — otherwise a retry silently loses tool-result memory."""
    fake = _fake_client()
    mcp_list = SimpleNamespace(
        type="mcp_list_tools", server_label="reportpro",
        tools=[{"name": "search", "description": "d", "input_schema": {}}])
    resp = SimpleNamespace(
        output=[_mcp_item("reportpro", "Ice Cream p.25 link=x"), mcp_list], usage=None)
    fake._safe_api_call = AsyncMock(return_value=resp)

    results_sink = []
    tools_sink = {}
    result = await R._create_text_response_with_tools_with_timeout(
        fake, messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "mcp"}], model="gpt-5.6-sol", timeout_seconds=30.0,
        return_metadata=True, mcp_results_sink=results_sink, mcp_tools_sink=tools_sink)

    assert results_sink == [{"tool_name": "reportpro", "output": "Ice Cream p.25 link=x"}]
    assert tools_sink["reportpro"][0]["name"] == "search"
    assert "reportpro" in result["tools_used"]


@pytest.mark.asyncio
async def test_timeout_twin_captures_usage_and_threads_cache_key():
    """Parity: the retry path must budget tokens (usage_sink) and route to the same cache shard
    (prompt_cache_key) — otherwise a retried turn corrupts context accounting on exactly the
    path that most needs it."""
    fake = _fake_client()
    resp = SimpleNamespace(output=[], usage=_usage(42, 8))
    fake._safe_api_call = AsyncMock(return_value=resp)

    usage_sink = {}
    await R._create_text_response_with_tools_with_timeout(
        fake, messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "web_search"}], model="gpt-5.6-sol", timeout_seconds=30.0,
        usage_sink=usage_sink, prompt_cache_key="C1:123.45")

    assert usage_sink == {"input_tokens": 42, "output_tokens": 8}
    # The cache key rides the request params into the actual API call.
    assert fake._safe_api_call.call_args.kwargs.get("prompt_cache_key") == "C1:123.45"
