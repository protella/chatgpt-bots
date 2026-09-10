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

The second half of this file covers the stream's other observer, `progress_callback` (F38):
reasoning summaries and hosted-call STARTS, for a surface that shows what is happening between
milestones. Its separation from the completion channel above is the property under test —
progress is never counted, and a caller that registered only one of the two must be unaffected
by the other's existence.
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


# ================================================================ F38 — the live progress channel
#
# A SECOND observer on the same stream, and the separation is the whole contract: this one says
# "this is happening right now" (a line of the model's reasoning summary, a hosted call
# starting), while `tool_event_callback` above says "this finished" and is what the consumer's
# counters bill. Everything here is read off SDK event objects whose shape moves between
# versions, so an unrecognised one must cost a progress line and nothing else.


def _sum_delta(text: str, item_id: str = "rs_1", index: int = 0) -> SimpleNamespace:
    return SimpleNamespace(type="response.reasoning_summary_text.delta", item_id=item_id,
                           summary_index=index, delta=text)


def _sum_done(text: str, item_id: str = "rs_1", index: int = 0) -> SimpleNamespace:
    return SimpleNamespace(type="response.reasoning_summary_text.done", item_id=item_id,
                           summary_index=index, text=text)


def _text_delta(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="response.output_text.delta", delta=text, output_index=0)


def _stream(*events: Any) -> List[Any]:
    return [SimpleNamespace(type="response.created"), *events,
            SimpleNamespace(type="response.completed", response=None)]


def _shape(payloads: List[Dict[str, Any]], *keys: str) -> List[tuple]:
    """The fields a test actually cares about, out of the full payload. `request_seq` is a
    process-wide counter, so nothing can assert on its value — only that it is shared."""
    return [tuple(p[k] for k in keys) for p in payloads]


async def _run_progress(client, events: List[Any]):
    """Drive the real streaming path and return (report_text, progress_events, tool_events)."""
    async def _iter(*a, **k):
        for e in events:
            yield e

    client._safe_stream_iteration = _iter
    client._safe_api_call = AsyncMock(return_value=object())

    progress: List[Dict[str, Any]] = []
    tool_events: List[Dict[str, Any]] = []

    text = await client.create_streaming_response_with_tools(
        messages=[{"role": "user", "content": "dig into it"}],
        tools=[{"type": "web_search"}],
        stream_callback=lambda chunk: None,
        reasoning_summary="auto",
        progress_callback=progress.append,
        tool_event_callback=tool_events.append)
    return text, progress, tool_events


class TestReasoningSummaryProgress:

    async def test_deltas_accumulate_and_done_replaces_them(self, client):
        _text, progress, _tools = await _run_progress(client, _stream(
            _sum_delta("Checking the "),
            _sum_delta("filings for"),
            _sum_done("Checking the filings for the 2019 restatement."),
        ))

        assert [p["text"] for p in progress] == [
            "Checking the ",
            "Checking the filings for",
            "Checking the filings for the 2019 restatement.",
        ]
        assert {p["kind"] for p in progress} == {"summary"}
        assert _shape(progress, "state") == [("delta",), ("delta",), ("done",)]
        # One request, one seq, and the dedup key names the part it belongs to.
        assert len({p["request_seq"] for p in progress}) == 1
        assert {(p["item_id"], p["summary_index"]) for p in progress} == {("rs_1", 0)}

    async def test_two_summaries_on_one_item_never_splice_together(self, client):
        """A response can carry several summary parts, streamed interleaved. Accumulating on
        anything coarser than (item, index) glues two narrations into one sentence."""
        _text, progress, _tools = await _run_progress(client, _stream(
            _sum_delta("First part", index=0),
            _sum_delta("Second part", index=1),
            _sum_delta(" continues", index=1),
        ))

        assert [p["text"] for p in progress] == [
            "First part", "Second part", "Second part continues"]

    async def test_the_part_events_carry_the_same_narration(self, client):
        """The complementary path: some SDK versions report the assembled part rather than
        streaming its text."""
        _text, progress, _tools = await _run_progress(client, _stream(
            SimpleNamespace(type="response.reasoning_summary_part.added", item_id="rs_1",
                            summary_index=0, part=SimpleNamespace(text="")),
            SimpleNamespace(type="response.reasoning_summary_part.done", item_id="rs_1",
                            summary_index=0, part={"text": "Reading the 10-K."}),
        ))

        assert [p["text"] for p in progress] == ["Reading the 10-K."]

    async def test_a_completed_reasoning_item_reports_a_summary_the_deltas_never_sent(
            self, client):
        item = SimpleNamespace(type="reasoning", id="rs_9",
                               summary=[SimpleNamespace(text="Comparing the two filings.")])
        _text, progress, _tools = await _run_progress(client, _stream(_done(item)))

        assert [p["text"] for p in progress] == ["Comparing the two filings."]

    async def test_a_summary_both_paths_reported_is_emitted_once(self, client):
        """Deltas, the part's done event and the completed item can all describe the same part.
        The accumulator is what collapses them — three identical lines on a status card read as
        a stuck job."""
        item = SimpleNamespace(type="reasoning", id="rs_1",
                               summary=[SimpleNamespace(text="Reading the 10-K.")])
        _text, progress, _tools = await _run_progress(client, _stream(
            _sum_done("Reading the 10-K."), _done(item)))

        assert [p["text"] for p in progress] == ["Reading the 10-K."]

    async def test_the_summary_never_becomes_part_of_the_report(self, client):
        """The one failure that would be visible to the user: a job publishing the model's
        scratch reasoning as its findings."""
        text, progress, _tools = await _run_progress(client, _stream(
            _sum_delta("Thinking about margins."),
            _text_delta("Margins rose 4% in 2024."),
            _sum_done("Thinking about margins, done."),
        ))

        assert text == "Margins rose 4% in 2024."
        assert progress                      # ...and the narration still went to the observer

    async def test_an_unreadable_summary_event_costs_a_line_and_nothing_else(self, client):
        text, progress, _tools = await _run_progress(client, _stream(
            SimpleNamespace(type="response.reasoning_summary_text.delta"),   # no fields at all
            SimpleNamespace(type="response.reasoning_summary_part.done", part=None),
            _text_delta("Still fine."),
        ))

        assert text == "Still fine."
        assert progress == []


class TestHostedProgressIsNotACompletion:

    async def test_a_search_reports_its_start_as_progress_and_its_query_at_completion(
            self, client):
        item = SimpleNamespace(type="web_search_call", action={"queries": ["unit margins"]})
        _text, progress, tools = await _run_progress(client, _stream(
            SimpleNamespace(type="response.web_search_call.searching", item_id="ws_1"),
            _done(item),
        ))

        assert _shape(progress, "kind", "tool", "state", "label") == [
            ("hosted", "web_search", "start", None),
            ("hosted", "web_search", "complete", "unit margins"),
        ]
        # ONE completion, which is what the card counts. The start event bumps nothing.
        assert tools == [{"kind": "web_search", "query": "unit margins"}]

    async def test_a_repeated_lifecycle_event_is_reported_once(self, client):
        _text, progress, _tools = await _run_progress(client, _stream(
            SimpleNamespace(type="response.web_search_call.searching", item_id="ws_1"),
            SimpleNamespace(type="response.web_search_call.searching", item_id="ws_1"),
            SimpleNamespace(type="response.web_search_call.searching", item_id="ws_2"),
        ))

        assert _shape(progress, "tool", "state", "item_id") == [
            ("web_search", "start", "ws_1"), ("web_search", "start", "ws_2")]

    async def test_the_sandbox_and_mcp_report_their_starts(self, client):
        _text, progress, _tools = await _run_progress(client, _stream(
            SimpleNamespace(type="response.code_interpreter_call.interpreting", item_id="ci_1"),
            SimpleNamespace(type="response.mcp_call.in_progress", item_id="mcp_1",
                            server_label="acmedata", name="get_trends"),
        ))

        assert _shape(progress, "tool", "state", "label") == [
            ("code_interpreter", "start", None),
            ("mcp", "start", "get_trends"),      # the TOOL name, not the server it lives on
        ]

    async def test_a_caller_with_no_progress_observer_is_completely_unaffected(self, client):
        """The chat turn. It registers only the completion callback, and what it receives must
        not change by one event."""
        emitted = await _run(client, [
            SimpleNamespace(type="response.created"),
            _sum_delta("narrating"),
            SimpleNamespace(type="response.web_search_call.searching", item_id="ws_1"),
            _done(SimpleNamespace(type="web_search_call", action={"query": "unit margins"})),
            SimpleNamespace(type="response.completed", response=None),
        ])

        assert emitted == [{"kind": "web_search", "query": "unit margins"}]

    async def test_a_failed_hosted_call_says_so_instead_of_reporting_success(self, client):
        """"Called acmedata" over a call that errored is the card lying about the work. The
        completion CHANNEL still behaves as it always did — an errored MCP call is not counted
        and not credited as a source."""
        item = SimpleNamespace(type="mcp_call", server_label="acmedata", name="get_trends",
                               error="upstream 502", output=None)
        _text, progress, tools = await _run_progress(client, _stream(_done(item)))

        assert _shape(progress, "tool", "state", "label") == [("mcp", "failed", "get_trends")]
        assert tools == []

    async def test_a_completed_sandbox_run_reports_itself_without_its_code(self, client):
        """The live line says a run happened; the code itself goes to the DEBUG log through the
        completion channel, never to a surface the room can read."""
        _text, progress, tools = await _run_progress(client, _stream(_done(_ci(code="df()"))))

        assert _shape(progress, "tool", "state", "label") == [
            ("code_interpreter", "complete", None)]
        assert tools == [{"kind": "code_interpreter", "code": "df()",
                          "container_id": "cntr_1"}]

    async def test_a_lifecycle_completion_does_not_duplicate_the_items_own_report(self, client):
        """Some SDK versions send both. The item knows the query, so it is the one that speaks;
        the dedup key is what stops the pair reading as two searches."""
        item = SimpleNamespace(type="web_search_call", action={"queries": ["a", "b"]},
                               id="ws_1")
        _text, progress, _tools = await _run_progress(client, _stream(
            SimpleNamespace(type="response.web_search_call.searching", item_id="ws_1"),
            SimpleNamespace(type="response.web_search_call.completed", item_id="ws_1"),
            _done(item),
        ))

        assert _shape(progress, "state", "label") == [
            ("start", None), ("complete", "a · b")]     # a fan-out of two is ONE line

    async def test_a_re_reported_summary_never_overwrites_newer_hosted_activity(self, client):
        """The paths disagree about trailing whitespace, and that difference used to be enough
        to re-emit. By the time the `.done` arrives the surface may have moved on to a hosted
        call — so the re-report would replace what is happening now with what the model said a
        minute ago, and reset its age while doing it."""
        _text, progress, _tools = await _run_progress(client, _stream(
            _sum_delta("I opened the 10-K."),
            _done(_ci(code="df()")),
            _sum_done("I opened the 10-K.\n"),      # same narration, different bytes
        ))

        assert _shape(progress, "kind", "state") == [
            ("summary", "delta"), ("hosted", "complete")]

    async def test_the_accumulator_still_keeps_the_raw_text(self, client):
        """Dedup compares trimmed text but must STORE it raw: deltas concatenate, so trimming
        what is kept would glue the next word onto the last one."""
        _text, progress, _tools = await _run_progress(client, _stream(
            _sum_delta("Checking "),
            _sum_delta(" "),                        # nothing readable moved
            _sum_delta("margins now."),
        ))

        assert [p["text"] for p in progress] == ["Checking ", "Checking  margins now."]
