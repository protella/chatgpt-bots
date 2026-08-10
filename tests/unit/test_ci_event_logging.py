"""The sandbox code a background job ran, surfaced to the internal observer.

Revision jobs regressed built documents by re-deriving sections instead of editing them, and
nothing in the logs said what the sandbox had actually executed — the container is gone long
before anyone looks. The streaming path already hands completed web_search and mcp_call items
to `tool_event_callback`; a completed `code_interpreter_call` now goes out the same channel as
`{"kind": "code_interpreter", "code": ..., "container_id": ...}`, and the consumer logs it at
DEBUG only (it is job content, not telemetry).

The kind is new, so the last test here pins the other half of the contract: the one production
consumer of these events may log it, but must not let it disturb the `tools_used` it rebuilds
from the same stream.
"""
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def _ci(code: Optional[str] = "print(1)",
        container_id: Optional[str] = "cntr_1") -> SimpleNamespace:
    return SimpleNamespace(type="code_interpreter_call", code=code,
                           container_id=container_id)


def _done(item: Any) -> SimpleNamespace:
    return SimpleNamespace(type="response.output_item.done", item=item)


def _events(*items: Any) -> List[SimpleNamespace]:
    return [SimpleNamespace(type="response.created"),
            *(_done(i) for i in items),
            SimpleNamespace(type="response.completed", response=None)]


@pytest.fixture
def client():
    with patch("openai_client.base.AsyncOpenAI"), \
            patch("openai_client.base.aiohttp.ClientSession"):
        from openai_client import OpenAIClient
        return OpenAIClient()


async def _run(client, events: List[Any]) -> List[Dict[str, Any]]:
    """Drive the real streaming-with-tools path over a scripted event sequence and return
    every observer payload it emitted."""
    async def _stream(*a, **k):
        for e in events:
            yield e

    client._safe_stream_iteration = _stream
    client._safe_api_call = AsyncMock(return_value=object())

    emitted: List[Dict[str, Any]] = []

    async def _on_event(payload):
        emitted.append(payload)

    await client.create_streaming_response_with_tools(
        messages=[{"role": "user", "content": "chart it"}],
        tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
        stream_callback=lambda chunk: None,
        tool_event_callback=_on_event)
    return emitted


class TestCodeInterpreterObserverEvent:

    async def test_a_completed_call_reports_its_code_and_container(self, client):
        emitted = await _run(client, _events(_ci(code="df.head()", container_id="cntr_a")))

        assert emitted == [{"kind": "code_interpreter", "code": "df.head()",
                            "container_id": "cntr_a"}]

    async def test_every_completed_call_reports_separately(self, client):
        """One event per call, in order — a job's sandbox history is the sequence of runs, and
        collapsing them would hide the run that broke the document."""
        emitted = await _run(client, _events(
            _ci(code="step one", container_id="cntr_a"),
            _ci(code="step two", container_id="cntr_a"),
        ))

        assert [e["code"] for e in emitted] == ["step one", "step two"]
        assert {e["kind"] for e in emitted} == {"code_interpreter"}

    async def test_an_item_carrying_neither_field_still_reports(self, client):
        """Both fields are the API's to send. A call we can't describe is still a call that
        happened, and silence would read as "the model ran no code"."""
        emitted = await _run(client, _events(_ci(code=None, container_id=None)))

        assert emitted == [{"kind": "code_interpreter", "code": None, "container_id": None}]

    async def test_other_completed_items_stay_silent(self, client):
        emitted = await _run(client, _events(
            SimpleNamespace(type="reasoning"),
            SimpleNamespace(type="message"),
            SimpleNamespace(type="mcp_list_tools"),
        ))

        assert emitted == []

    async def test_the_web_search_and_mcp_events_are_untouched(self, client):
        emitted = await _run(client, _events(
            SimpleNamespace(type="web_search_call", action={"query": "unit margins"}),
            _ci(code="chart(df)", container_id="cntr_a"),
            SimpleNamespace(type="mcp_call", server_label="acmedata", error=None),
        ))

        assert emitted == [
            {"kind": "web_search", "query": "unit margins"},
            {"kind": "code_interpreter", "code": "chart(df)", "container_id": "cntr_a"},
            {"kind": "mcp", "server_label": "acmedata"},
        ]


class TestTheConsumerToleratesTheNewKind:
    """`_consume_research_stream._on_event` is the only production consumer of these events.
    It recognises the new kind and logs it at DEBUG — but logging is all it may do: the same
    function rebuilds the job's `tools_used` provenance trailer from these events, and sandbox
    code is a diagnostic, not a research source. The invariant pinned here is that the
    code_interpreter event leaves `observed`/`tools_used` untouched."""

    async def test_a_code_interpreter_event_is_not_a_research_source(self):
        from message_processor import research_tools

        class _StreamStub:
            async def __call__(self, **kwargs):
                cb = kwargs["tool_event_callback"]
                for ev in ({"kind": "web_search", "query": "unit margins"},
                           {"kind": "code_interpreter", "code": "chart(df)",
                            "container_id": "cntr_a"}):
                    r = cb(ev)
                    if r is not None and hasattr(r, "__await__"):
                        await r
                return {"text": "done", "tools_used": [], "local_tool_calls": []}

        processor = SimpleNamespace(
            openai_client=SimpleNamespace(
                create_streaming_response_with_tool_loop=_StreamStub()),
            # The consumer logs the snippet here; the stub only has to survive the call.
            log_debug=lambda *a, **k: None)

        result = await research_tools._consume_research_stream(
            processor, messages=[], tools=[], registry=None, tool_context=None,
            model="gpt-5.6-sol", system_prompt=None, effort="medium", verbosity="medium",
            card=None)

        assert result["tools_used"] == ["web_search"]
