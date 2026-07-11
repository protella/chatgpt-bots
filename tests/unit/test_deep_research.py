"""F30 — background deep-dive research jobs.

Covers the start_deep_research schema + registry gating, the per-thread cap, context
snapshotting by copy, the happy-path findings delivery (with trailer), error/timeout failure
notes, the research-label fallback + process-lifetime memory, the thread_manager research
registry / shutdown cancellation, and config defaults.
"""
import asyncio
from types import SimpleNamespace

import pytest

from config import clamp_effort, config
from thread_manager import AsyncThreadStateManager
from tool_registry import ToolContext, ToolRegistry
import message_processor.research_tools as rt


# --------------------------------------------------------------- fakes

class _FakeClient:
    def __init__(self, fail_username=False):
        self.sent = []  # (channel, thread, text, username)
        self.fail_username = fail_username

    async def send_message(self, channel_id, thread_id, text, blocks=None,
                           meta_out=None, username=None):
        if username and self.fail_username:
            return None  # simulate missing_scope → send_message returns None
        self.sent.append((channel_id, thread_id, text, username))
        return "9999.000001"


class _FakeProcessor:
    def __init__(self, openai_client=None, tm=None):
        self.openai_client = openai_client
        self.thread_manager = tm
        self.scheduled = []

    def _schedule_async_call(self, coro):
        # Capture the coroutine instead of running it, so tests drive the job explicitly.
        self.scheduled.append(coro)
        return SimpleNamespace(done=lambda: False, cancel=lambda: None)

    def _build_tools_array(self, cfg, model, registry=None):
        return [{"type": "web_search"}]

    def log_info(self, *a, **k):
        pass

    log_error = log_warning = log_debug = log_info


def _ctx(processor, client, *, current_input=None, thread_ts="100.0", trigger_ts="100.0",
         channel_id="C1"):
    return ToolContext(
        channel_id=channel_id, thread_ts=thread_ts, trigger_ts=trigger_ts,
        client=client, processor=processor, current_input=current_input or [],
        system_prompt="DEV PROMPT", model="gpt-5.6-sol")


class _StreamStub:
    """Stand-in for openai_client.create_streaming_response_with_tool_loop (F30.2): records
    the call kwargs, optionally fires tool_event_callback with synthetic web_search/mcp
    events, optionally dispatches report_progress milestones through the REAL job registry
    (exercising the executor→card wiring), then returns the loop-shaped result dict — or
    raises / stalls, to exercise the failure paths."""
    def __init__(self, text="", events=None, raises=None, slow=False, milestones=None):
        self.text = text
        self.events = events or []
        self.raises = raises
        self.slow = slow
        self.milestones = milestones or []
        self.kwargs = None

    async def __call__(self, **kwargs):
        self.kwargs = kwargs
        if self.slow:
            await asyncio.sleep(5)
        if self.raises is not None:
            raise self.raises
        cb = kwargs.get("tool_event_callback")
        for ev in self.events:
            if cb is not None:
                r = cb(ev)
                if r is not None and hasattr(r, "__await__"):
                    await r
        registry = kwargs.get("registry")
        for m in self.milestones:
            assert registry is not None, "milestones need the job registry"
            res = await registry.dispatch(kwargs.get("tool_context"), "report_progress",
                                          {"milestone": m})
            assert res.get("ok") is True
        return {"text": self.text, "tools_used": [], "local_tool_calls": []}


class _CardClient(_FakeClient):
    """_FakeClient + the F30.1 status-card primitives, recording every card post/update."""
    def __init__(self, fail_username=False, card_username_fails=False, card_ts="CARD.1"):
        super().__init__(fail_username=fail_username)
        self.card_posts = []    # (channel, thread, text, blocks, username)
        self.card_updates = []  # (channel, ts, text, blocks)
        self.card_username_fails = card_username_fails
        self._card_ts = card_ts

    async def post_status_card(self, channel_id, thread_id, text, blocks, username=None):
        if username and self.card_username_fails:
            return None
        self.card_posts.append((channel_id, thread_id, text, blocks, username))
        return self._card_ts

    async def update_status_card(self, channel_id, ts, text, blocks):
        self.card_updates.append((channel_id, ts, text, blocks))
        return True


def _card_body(update_or_post):
    """The section block's mrkdwn text from a recorded card post/update tuple."""
    blocks = update_or_post[3]
    return blocks[0]["text"]["text"]


# --------------------------------------------------------------- schema + gating

def test_schema_shape():
    schema = rt.get_start_deep_research_schema()
    assert schema["name"] == "start_deep_research"
    assert schema["type"] == "function"
    props = schema["parameters"]["properties"]
    assert "task" in props and props["task"]["type"] == "string"
    # Optional short byline tag (F30.2 follow-up) — task stays the only required param.
    assert "label" in props and props["label"]["type"] == "string"
    assert schema["parameters"]["required"] == ["task"]


def test_registry_gating_on_flag(monkeypatch):
    monkeypatch.setattr(config, "enable_deep_research", True)
    reg = ToolRegistry()
    rt.register_research_tools(reg)
    assert any(s["name"] == "start_deep_research" for s in reg.schemas({}))


# --------------------------------------------------------------- executor: cap + snapshot

@pytest.mark.asyncio
async def test_executor_disabled_returns_error(monkeypatch):
    monkeypatch.setattr(config, "enable_deep_research", False)
    res = await rt.execute_start_deep_research(_ctx(_FakeProcessor(), _FakeClient()),
                                              {"task": "dig into X"})
    assert res["ok"] is False and res["error"] == "disabled"


@pytest.mark.asyncio
async def test_executor_missing_task(monkeypatch):
    monkeypatch.setattr(config, "enable_deep_research", True)
    res = await rt.execute_start_deep_research(_ctx(_FakeProcessor(), _FakeClient()),
                                              {"task": "   "})
    assert res["ok"] is False and res["error"] == "missing_task"


@pytest.mark.asyncio
async def test_per_thread_cap_rejection(monkeypatch):
    monkeypatch.setattr(config, "enable_deep_research", True)
    monkeypatch.setattr(config, "deep_research_max_per_thread", 2)
    tm = AsyncThreadStateManager()
    proc = _FakeProcessor(openai_client=SimpleNamespace(), tm=tm)
    ctx = _ctx(proc, _FakeClient())
    # Pre-fill the thread to the cap.
    tm.register_research("C1:100.0", "j1", "a")
    tm.register_research("C1:100.0", "j2", "b")
    res = await rt.execute_start_deep_research(ctx, {"task": "dig into X"})
    assert res["ok"] is False and res["error"] == "too_many_research_jobs"
    assert "max 2" in res["message"]
    assert proc.scheduled == []  # nothing detached


@pytest.mark.asyncio
async def test_executor_starts_and_returns_ack(monkeypatch):
    monkeypatch.setattr(config, "enable_deep_research", True)
    tm = AsyncThreadStateManager()
    proc = _FakeProcessor(openai_client=SimpleNamespace(), tm=tm)
    res = await rt.execute_start_deep_research(_ctx(proc, _FakeClient()),
                                              {"task": "validate the claim about X"})
    assert res["ok"] is True and res["status"] == "started"
    assert res["task"]  # gist for the model to relay
    assert len(proc.scheduled) == 1
    assert tm.research_in_flight_count("C1:100.0") == 1
    proc.scheduled[0].close()  # don't leave the captured coro un-awaited


@pytest.mark.asyncio
async def test_snapshot_is_by_copy(monkeypatch):
    monkeypatch.setattr(config, "enable_deep_research", True)
    live_input = [{"role": "user", "content": "original question"}]
    stub = _StreamStub(text="findings", events=[{"kind": "web_search", "query": "q"}])
    openai = SimpleNamespace(create_streaming_response_with_tool_loop=stub)
    proc = _FakeProcessor(openai_client=openai, tm=AsyncThreadStateManager())
    client = _FakeClient()
    ctx = _ctx(proc, client, current_input=live_input)
    await rt.execute_start_deep_research(ctx, {"task": "research the thing"})
    # Mutate the LIVE input after the call — the job's snapshot must be unaffected.
    live_input.append({"role": "user", "content": "LATE addition"})
    await proc.scheduled[0]  # run the captured job
    job_input = stub.kwargs["messages"]
    contents = [m.get("content") for m in job_input]
    assert "original question" in contents
    assert not any("LATE addition" == c for c in contents)
    # The appended developer instruction carries the task.
    assert any(m.get("role") == "developer" and "research the thing" in (m.get("content") or "")
               for m in job_input)


# --------------------------------------------------------------- job delivery

@pytest.mark.asyncio
async def test_happy_path_posts_findings_with_trailer(monkeypatch):
    monkeypatch.setattr(config, "enable_deep_research", True)
    monkeypatch.setattr(config, "enable_research_label", False)  # isolate the plain path
    openai = SimpleNamespace(create_streaming_response_with_tool_loop=_StreamStub(
        text="# Findings\nThe answer is 42.", events=[{"kind": "web_search", "query": "the claim"}]))
    proc = _FakeProcessor(openai_client=openai, tm=AsyncThreadStateManager())
    client = _FakeClient()
    await rt._run_deep_research_job(
        processor=proc, client=client, channel_id="C1", thread_root="100.0",
        thread_key="C1:100.0", job_id="j1", task="the task",
        snapshot=[{"role": "user", "content": "q"}], system_prompt="DEV", model="gpt-5.6-sol")
    assert len(client.sent) == 1
    _, thread, text, username = client.sent[0]
    assert thread == "100.0"
    assert "The answer is 42." in text
    assert "_deep research ·" in text and "effort" in text  # trailer
    assert "tools: web_search" in text  # visible tool attribution in the trailer
    assert username is None


@pytest.mark.asyncio
async def test_error_path_posts_failure_note(monkeypatch):
    monkeypatch.setattr(config, "enable_deep_research", True)
    openai = SimpleNamespace(create_streaming_response_with_tool_loop=_StreamStub(
        raises=RuntimeError("boom")))
    proc = _FakeProcessor(openai_client=openai, tm=AsyncThreadStateManager())
    client = _FakeClient()
    await rt._run_deep_research_job(
        processor=proc, client=client, channel_id="C1", thread_root="100.0",
        thread_key="C1:100.0", job_id="j1", task="the task",
        snapshot=[], system_prompt=None, model="gpt-5.6-sol")
    assert len(client.sent) == 1
    assert "hit a wall" in client.sent[0][2]


@pytest.mark.asyncio
async def test_timeout_path_posts_failure_note(monkeypatch):
    monkeypatch.setattr(config, "enable_deep_research", True)
    monkeypatch.setattr(config, "deep_research_timeout", 0.01)

    openai = SimpleNamespace(create_streaming_response_with_tool_loop=_StreamStub(slow=True))
    proc = _FakeProcessor(openai_client=openai, tm=AsyncThreadStateManager())
    client = _FakeClient()
    await rt._run_deep_research_job(
        processor=proc, client=client, channel_id="C1", thread_root="100.0",
        thread_key="C1:100.0", job_id="j1", task="the task",
        snapshot=[], system_prompt=None, model="gpt-5.6-sol")
    assert len(client.sent) == 1
    assert "time limit" in client.sent[0][2]


@pytest.mark.asyncio
async def test_job_clears_registry_on_finish(monkeypatch):
    monkeypatch.setattr(config, "enable_deep_research", True)
    monkeypatch.setattr(config, "enable_research_label", False)
    openai = SimpleNamespace(create_streaming_response_with_tool_loop=_StreamStub(text="done"))
    tm = AsyncThreadStateManager()
    tm.register_research("C1:100.0", "j1", "t")
    proc = _FakeProcessor(openai_client=openai, tm=tm)
    await rt._run_deep_research_job(
        processor=proc, client=_FakeClient(), channel_id="C1", thread_root="100.0",
        thread_key="C1:100.0", job_id="j1", task="t", snapshot=[],
        system_prompt=None, model="gpt-5.6-sol")
    assert tm.research_in_flight_count("C1:100.0") == 0


# --------------------------------------------------------------- research label fallback

@pytest.mark.asyncio
async def test_label_attempt_falls_back_and_remembers(monkeypatch):
    monkeypatch.setattr(config, "enable_deep_research", True)
    monkeypatch.setattr(config, "enable_research_label", True)
    monkeypatch.setattr(config, "bot_name_aliases", ["Sol"])
    proc = _FakeProcessor(tm=AsyncThreadStateManager())
    client = _FakeClient(fail_username=True)  # username override always fails (no scope)
    await rt._deliver_findings(proc, client, "C1", "100.0", "findings body", "the task")
    # It fell back to a plain post (no username) and posted exactly once.
    assert len(client.sent) == 1 and client.sent[0][3] is None
    # And it remembered the failure for the rest of the process.
    assert getattr(proc, rt._RESEARCH_LABEL_DISABLED_ATTR) is True
    # A second delivery skips the labelled attempt entirely (still plain).
    await rt._deliver_findings(proc, client, "C1", "100.0", "more findings", "another task")
    assert all(s[3] is None for s in client.sent)


@pytest.mark.asyncio
async def test_label_success_uses_username(monkeypatch):
    monkeypatch.setattr(config, "enable_deep_research", True)
    monkeypatch.setattr(config, "enable_research_label", True)
    monkeypatch.setattr(config, "bot_name_aliases", ["Sol"])
    proc = _FakeProcessor(tm=AsyncThreadStateManager())
    client = _FakeClient(fail_username=False)
    await rt._deliver_findings(proc, client, "C1", "100.0", "findings", "map the market")
    assert len(client.sent) == 1
    username = client.sent[0][3]
    assert username and username.startswith("Sol [research:")


# --------------------------------------------------------------- thread_manager registry

@pytest.mark.asyncio
async def test_thread_manager_research_registry_and_cancel():
    tm = AsyncThreadStateManager()

    async def _job():
        await asyncio.sleep(30)

    task = asyncio.create_task(_job())
    tm.register_research("C1:1.0", "j1", "gist", task=task)
    assert tm.research_in_flight_count("C1:1.0") == 1
    await tm.cancel_research_jobs(timeout=2.0)
    assert task.cancelled()
    assert tm.research_in_flight_count("C1:1.0") == 0


def test_finish_research_only_own_entry():
    tm = AsyncThreadStateManager()
    tm.register_research("C1:1.0", "j1", "a")
    tm.register_research("C1:1.0", "j2", "b")
    assert tm.finish_research("C1:1.0", "j1") is True
    assert tm.research_in_flight_count("C1:1.0") == 1
    assert tm.finish_research("C1:1.0", "nope") is False


# --------------------------------------------------------------- config defaults

def test_config_defaults():
    assert config.enable_deep_research is True
    assert config.deep_research_reasoning_effort == "high"
    assert config.deep_research_verbosity == "medium"
    assert float(config.deep_research_timeout) == 600.0
    assert config.deep_research_max_per_thread == 2
    assert config.deep_research_max_tool_rounds == 10
    assert config.enable_research_label is True
    # Effort routes through clamp_effort against a real model (never rejected).
    assert clamp_effort("gpt-5.6-sol", config.deep_research_reasoning_effort) in {
        "none", "low", "medium", "high", "xhigh", "max"}


# --------------------------------------------- Codex review fixes (round 1)

@pytest.mark.asyncio
async def test_web_search_forced_into_job_tools(monkeypatch):
    """An MCP-only tools array (global web search off) must still get web_search —
    the tool IS web research; a truthy array must not silently strip it."""
    monkeypatch.setattr(config, "enable_deep_research", True)
    monkeypatch.setattr(config, "enable_research_label", False)
    stub = _StreamStub(text="report")
    openai = SimpleNamespace(create_streaming_response_with_tool_loop=stub)
    proc = _FakeProcessor(openai_client=openai, tm=AsyncThreadStateManager())
    proc._build_tools_array = lambda cfg, model, registry=None: [
        {"type": "mcp", "server_label": "x"}]  # truthy, no web_search
    await rt._run_deep_research_job(
        processor=proc, client=_FakeClient(), channel_id="C1", thread_root="100.0",
        thread_key="C1:100.0", job_id="j1", task="t",
        snapshot=[], system_prompt="DEV", model="gpt-5.6-sol")
    tools = stub.kwargs["tools"]
    assert {"type": "web_search"} in tools
    assert {"type": "mcp", "server_label": "x"} in tools


@pytest.mark.asyncio
async def test_failed_findings_post_posts_failure_note(monkeypatch):
    """send_message swallowing a Slack error into None must NOT read as job success:
    the job posts an honest failure note instead of logging 'completed'."""
    monkeypatch.setattr(config, "enable_deep_research", True)
    monkeypatch.setattr(config, "enable_research_label", False)
    openai = SimpleNamespace(create_streaming_response_with_tool_loop=_StreamStub(text="report"))
    proc = _FakeProcessor(openai_client=openai, tm=AsyncThreadStateManager())

    class _FailingThenNoteClient(_FakeClient):
        async def send_message(self, channel_id, thread_id, text, blocks=None,
                               meta_out=None, username=None):
            if "hit a wall" not in text:
                return None  # the findings post itself fails
            return await super().send_message(channel_id, thread_id, text,
                                              blocks=blocks, meta_out=meta_out,
                                              username=username)

    client = _FailingThenNoteClient()
    await rt._run_deep_research_job(
        processor=proc, client=client, channel_id="C1", thread_root="100.0",
        thread_key="C1:100.0", job_id="j1", task="t",
        snapshot=[], system_prompt="DEV", model="gpt-5.6-sol")
    assert len(client.sent) == 1
    assert "hit a wall" in client.sent[0][2]
    assert "posting them to Slack failed" in client.sent[0][2]


@pytest.mark.asyncio
async def test_schedule_failure_closes_coroutine(monkeypatch):
    """A never-scheduled job coroutine is close()d so it can't warn as unawaited."""
    monkeypatch.setattr(config, "enable_deep_research", True)
    proc = _FakeProcessor(openai_client=SimpleNamespace(), tm=AsyncThreadStateManager())

    def _boom(coro):
        raise RuntimeError("loop is closed")
    proc._schedule_async_call = _boom
    res = await rt.execute_start_deep_research(
        _ctx(proc, _FakeClient()), {"task": "t"})
    assert res["ok"] is False and res["error"] == "schedule_failed"
    # Registry cleared — nothing left in flight for the thread.
    assert proc.thread_manager.research_in_flight_count("C1:100.0") == 0


# ============================================================ F30.1 — status card + suppression

# --------------------------------------------- ack suppression signal

@pytest.mark.asyncio
async def test_ack_suppression_flag_set_on_success(monkeypatch):
    monkeypatch.setattr(config, "enable_deep_research", True)
    proc = _FakeProcessor(openai_client=SimpleNamespace(), tm=AsyncThreadStateManager())
    ctx = _ctx(proc, _FakeClient())
    res = await rt.execute_start_deep_research(ctx, {"task": "validate the claim about X"})
    assert res["ok"] is True
    assert ctx.deep_research_started is True  # the turn's finalizer drops the ack reply
    proc.scheduled[0].close()


@pytest.mark.asyncio
async def test_ack_suppression_flag_not_set_on_cap(monkeypatch):
    monkeypatch.setattr(config, "enable_deep_research", True)
    monkeypatch.setattr(config, "deep_research_max_per_thread", 1)
    tm = AsyncThreadStateManager()
    tm.register_research("C1:100.0", "j0", "x")  # already at the cap
    proc = _FakeProcessor(openai_client=SimpleNamespace(), tm=tm)
    ctx = _ctx(proc, _FakeClient())
    res = await rt.execute_start_deep_research(ctx, {"task": "validate X"})
    assert res["ok"] is False
    assert ctx.deep_research_started is False  # rejection must NOT suppress a normal reply


@pytest.mark.asyncio
async def test_ack_suppression_flag_not_set_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "enable_deep_research", False)
    ctx = _ctx(_FakeProcessor(), _FakeClient())
    res = await rt.execute_start_deep_research(ctx, {"task": "X"})
    assert res["ok"] is False
    assert ctx.deep_research_started is False


def test_prompts_bullet_updated():
    from prompts import LOCAL_TOOLS_GUIDANCE
    assert "will NOT be posted" in LOCAL_TOOLS_GUIDANCE
    assert "write NOTHING after the call" in LOCAL_TOOLS_GUIDANCE


# --------------------------------------------- streaming consumption

@pytest.mark.asyncio
async def test_consume_stream_returns_text_and_tools_equivalent():
    """The internal streaming consumption accumulates the text and rebuilds tools_used from
    observed events — deduped, in order — matching the old non-streaming metadata."""
    stub = _StreamStub(text="the report", events=[
        {"kind": "web_search", "query": "q1"},
        {"kind": "mcp", "server_label": "datassential"},
        {"kind": "web_search", "query": "q2"},  # duplicate web_search collapses to one
    ])
    proc = _FakeProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=stub))
    out = await rt._consume_research_stream(
        proc, messages=[{"role": "user", "content": "q"}], tools=[{"type": "web_search"}],
        registry=ToolRegistry(), tool_context=ToolContext(),
        model="gpt-5.6-sol", system_prompt="DEV", effort="high", verbosity="medium", card=None)
    assert out["text"] == "the report"
    assert out["tools_used"] == ["web_search", "datassential"]
    # store=False and no Slack streaming (tool_callback not used for status).
    assert stub.kwargs["store"] is False
    # F30.2: the job's OWN round budget rides the call — the 4-round chat cap would
    # strangle milestone reporting.
    assert stub.kwargs["max_tool_rounds"] == config.deep_research_max_tool_rounds
    assert stub.kwargs["max_tool_calls"] == config.deep_research_max_tool_rounds


# --------------------------------------------- card lifecycle on the job

@pytest.mark.asyncio
async def test_card_posted_on_job_start_with_label(monkeypatch):
    monkeypatch.setattr(config, "enable_deep_research", True)
    monkeypatch.setattr(config, "enable_research_label", True)
    monkeypatch.setattr(config, "bot_name_aliases", ["Sol"])
    openai = SimpleNamespace(create_streaming_response_with_tool_loop=_StreamStub(
        text="findings body", events=[{"kind": "web_search", "query": "the claim"}]))
    proc = _FakeProcessor(openai_client=openai, tm=AsyncThreadStateManager())
    client = _CardClient()
    await rt._run_deep_research_job(
        processor=proc, client=client, channel_id="C1", thread_root="100.0",
        thread_key="C1:100.0", job_id="j1", task="map the market",
        snapshot=[{"role": "user", "content": "q"}], system_prompt="DEV", model="gpt-5.6-sol")
    assert len(client.card_posts) == 1
    _, thread, text, blocks, username = client.card_posts[0]
    assert thread == "100.0"
    assert text == rt._CARD_FALLBACK_TEXT                 # constant notification fallback
    assert username and username.startswith("Sol [research:")  # same label as findings
    assert blocks[0]["type"] == "section" and blocks[1]["type"] == "context"
    assert blocks[1]["elements"][0]["text"].startswith("todos as of ")
    # Final card update reads "Reported findings below." and lands BEFORE the report post.
    assert client.card_updates
    assert "✓ Reported findings below." in _card_body(client.card_updates[-1])
    # Findings posted under the SAME label.
    assert client.sent and client.sent[0][3] and client.sent[0][3].startswith("Sol [research:")
    # Fallback text stays constant across every card update (no "(edited)" badge).
    assert all(u[2] == rt._CARD_FALLBACK_TEXT for u in client.card_updates)


@pytest.mark.asyncio
async def test_card_unlabeled_when_label_disabled(monkeypatch):
    monkeypatch.setattr(config, "enable_deep_research", True)
    monkeypatch.setattr(config, "enable_research_label", False)  # label off → card still posts
    openai = SimpleNamespace(create_streaming_response_with_tool_loop=_StreamStub(text="body"))
    proc = _FakeProcessor(openai_client=openai, tm=AsyncThreadStateManager())
    client = _CardClient()
    await rt._run_deep_research_job(
        processor=proc, client=client, channel_id="C1", thread_root="100.0",
        thread_key="C1:100.0", job_id="j1", task="t", snapshot=[],
        system_prompt="DEV", model="gpt-5.6-sol")
    assert len(client.card_posts) == 1
    assert client.card_posts[0][4] is None  # unlabeled


@pytest.mark.asyncio
async def test_card_failure_does_not_break_findings(monkeypatch):
    monkeypatch.setattr(config, "enable_deep_research", True)
    monkeypatch.setattr(config, "enable_research_label", False)
    openai = SimpleNamespace(create_streaming_response_with_tool_loop=_StreamStub(
        text="body", events=[{"kind": "web_search", "query": "q"}]))
    proc = _FakeProcessor(openai_client=openai, tm=AsyncThreadStateManager())

    class _BoomCardClient(_CardClient):
        async def post_status_card(self, *a, **k):
            raise RuntimeError("card down")
        async def update_status_card(self, *a, **k):
            raise RuntimeError("card down")

    client = _BoomCardClient()
    await rt._run_deep_research_job(
        processor=proc, client=client, channel_id="C1", thread_root="100.0",
        thread_key="C1:100.0", job_id="j1", task="t", snapshot=[],
        system_prompt="DEV", model="gpt-5.6-sol")
    assert len(client.sent) == 1 and "body" in client.sent[0][2]  # findings still posted


@pytest.mark.asyncio
async def test_card_label_failure_falls_back_unlabeled_and_remembers(monkeypatch):
    monkeypatch.setattr(config, "enable_deep_research", True)
    monkeypatch.setattr(config, "enable_research_label", True)
    monkeypatch.setattr(config, "bot_name_aliases", ["Sol"])
    openai = SimpleNamespace(create_streaming_response_with_tool_loop=_StreamStub(text="body"))
    proc = _FakeProcessor(openai_client=openai, tm=AsyncThreadStateManager())
    client = _CardClient(card_username_fails=True)  # labelled card post rejected (no scope)
    await rt._run_deep_research_job(
        processor=proc, client=client, channel_id="C1", thread_root="100.0",
        thread_key="C1:100.0", job_id="j1", task="t", snapshot=[],
        system_prompt="DEV", model="gpt-5.6-sol")
    # Card fell back to an unlabeled post and remembered the failure process-wide.
    assert len(client.card_posts) == 1 and client.card_posts[0][4] is None
    assert getattr(proc, rt._RESEARCH_LABEL_DISABLED_ATTR) is True
    # Findings also skip the label (plain post).
    assert client.sent and client.sent[0][3] is None


# --------------------------------------------- card rendering + throttle (isolated)

def _bare_card(client, *, task="dig into the thing", label=None,
               clock=None, sleep=None, now_label=None):
    return rt._ResearchCard(
        processor=_FakeProcessor(), client=client, channel_id="C1", thread_root="100.0",
        task=task, label=label,
        clock=clock or (lambda: 0.0),
        sleep=sleep or (lambda d: asyncio.sleep(0)),
        now_label=now_label or (lambda: "3:00 PM"))


@pytest.mark.asyncio
async def test_card_update_throttle_coalesces_with_trailing_flush():
    clock = [0.0]
    slept = []

    async def _fake_sleep(d):
        slept.append(d)

    client = _CardClient()
    card = _bare_card(client, clock=lambda: clock[0], sleep=_fake_sleep)
    card.ts = "CARD.1"
    await card.add_milestone("first")     # last_update None → immediate flush (#1)
    assert len(client.card_updates) == 1
    await card.add_milestone("second")    # within window → schedule ONE trailing flush
    await card.note_mcp("srv")            # still within window → no extra task/update
    assert len(client.card_updates) == 1
    assert card._flush_task is not None
    clock[0] = 100.0
    await card._flush_task                # trailing flush → update #2 with the LATEST state
    assert len(client.card_updates) == 2
    body = _card_body(client.card_updates[-1])
    context = client.card_updates[-1][3][1]["elements"][0]["text"]
    assert "second" in body and "1 srv call" in context   # coalesced, not stale
    assert slept and slept[0] <= rt._card_throttle_s()
    assert all(u[2] == rt._CARD_FALLBACK_TEXT for u in client.card_updates)


def test_card_todo_line_cap_is_loud_and_counted():
    card = _bare_card(_CardClient())
    for i in range(15):
        card._milestones.append(f"✓ milestone {i}")
    lines = card._visible_lines()
    # header + 15 milestones = 16 logical lines → collapsed to exactly 10 visible.
    assert len(lines) == rt._CARD_MAX_TODO_LINES
    assert "… +7 more milestones" in lines  # 16 - (10 - 1) = 7 hidden, counted


@pytest.mark.asyncio
async def test_card_final_states():
    """Terminal line appended AND the headline ⏳ flips to the overall outcome emoji —
    the card never ends still showing an hourglass."""
    for finalize, needle, emoji in (
        (lambda c: c.finalize_success(), "✓ Reported findings below.", "✅"),
        (lambda c: c.finalize_failure("everything broke"), "✗ hit a wall: everything broke",
         "❌"),
        (lambda c: c.finalize_cancelled(), "✗ cancelled (bot shutting down)", "❌"),
    ):
        client = _CardClient()
        card = _bare_card(client)
        card.ts = "CARD.1"
        await finalize(card)
        body = _card_body(client.card_updates[-1])
        assert needle in body
        assert body.startswith(f"{emoji} ")  # headline flipped from ⏳
        # Closed: later notes are no-ops (no stale update after the terminal line).
        before = len(client.card_updates)
        await card.add_milestone("late")
        await card.note_web_search()
        assert len(client.card_updates) == before


def test_card_headline_starts_with_hourglass():
    card = _bare_card(_CardClient())
    assert card._visible_lines()[0].startswith("⏳ ")


@pytest.mark.asyncio
async def test_card_counters_in_context_line_not_body():
    """F30.2: raw tool events bump the context-line counters (with pluralization); the
    body stays model-authored milestones only — never a per-search log."""
    client = _CardClient()
    # Advancing clock so each note clears the throttle window and flushes immediately.
    t = [0.0]

    def _clk():
        t[0] += rt._card_throttle_s() + 1.0
        return t[0]

    card = _bare_card(client, clock=_clk)
    card.ts = "CARD.1"
    await card.note_web_search()
    await card.note_web_search()
    await card.note_mcp("datassential")
    await card.note_mcp(None)  # unlabeled server → generic "MCP" bucket
    await card.add_milestone("Searched  regulatory\ndockets — found the rule")  # ws collapsed
    body = _card_body(client.card_updates[-1])
    context = client.card_updates[-1][3][1]["elements"][0]["text"]
    assert "✓ Searched regulatory dockets — found the rule" in body
    assert "searched the web" not in body          # no mechanical body lines
    assert context.startswith("todos as of ")
    assert "2 web searches" in context
    assert "1 datassential call" in context
    assert "1 MCP call" in context


def test_card_throttle_derives_from_streaming_min_interval(monkeypatch):
    """Not a magic number: the card update floor is STREAMING_MIN_INTERVAL (Slack's
    ~1/sec message-update guidance), floored at 1.0."""
    assert rt._card_throttle_s() == max(1.0, float(config.streaming_min_interval))
    monkeypatch.setattr(config, "streaming_min_interval", 2.5)
    assert rt._card_throttle_s() == 2.5
    monkeypatch.setattr(config, "streaming_min_interval", 0.2)  # below Slack's floor
    assert rt._card_throttle_s() == 1.0


def test_base_wrapper_accepts_tool_event_callback():
    """Regression (live find): the job calls through OpenAIClient's wrapper, which must
    accept and forward tool_event_callback — the module function alone isn't enough."""
    import inspect
    from openai_client.base import OpenAIClient
    params = inspect.signature(OpenAIClient.create_streaming_response_with_tools).parameters
    assert "tool_event_callback" in params


def test_tool_loop_accepts_round_budget_overrides():
    """F30.2 wrapper-drift guard: the streaming tool loop takes the research job's round
    budget, and the OpenAIClient wrapper's **params passes it through (VAR_KEYWORD)."""
    import inspect
    from openai_client.api import tool_loop
    from openai_client.base import OpenAIClient
    loop_params = inspect.signature(tool_loop.create_streaming_response_with_tool_loop).parameters
    assert "max_tool_rounds" in loop_params and "max_tool_calls" in loop_params
    wrapper_params = inspect.signature(
        OpenAIClient.create_streaming_response_with_tool_loop).parameters
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in wrapper_params.values())


# ============================================================ F30.2 — model-authored milestones

def test_report_progress_schema_shape():
    schema = rt.get_report_progress_schema()
    assert schema["name"] == "report_progress"
    assert schema["type"] == "function"
    assert schema["parameters"]["required"] == ["milestone"]
    # The description steers goal-level milestones, not per-search spam.
    assert "NOT once per search" in schema["description"]


def test_job_instruction_mentions_milestones():
    assert "report_progress" in rt._RESEARCH_JOB_INSTRUCTION
    assert "BEFORE you start writing the findings report" in rt._RESEARCH_JOB_INSTRUCTION


@pytest.mark.asyncio
async def test_job_wires_report_progress_milestones_to_card(monkeypatch):
    """End-to-end through the real job: the stub 'model' calls report_progress via the
    job registry, and the milestone lands on the card body; the tools array carries the
    report_progress schema alongside the server tools; the trailer excludes it."""
    monkeypatch.setattr(config, "enable_deep_research", True)
    monkeypatch.setattr(config, "enable_research_label", False)
    stub = _StreamStub(text="findings body",
                       events=[{"kind": "web_search", "query": "q"}],
                       milestones=["Searched dockets — found the final rule."])
    proc = _FakeProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=stub))
    proc.thread_manager = AsyncThreadStateManager()
    client = _CardClient()
    await rt._run_deep_research_job(
        processor=proc, client=client, channel_id="C1", thread_root="100.0",
        thread_key="C1:100.0", job_id="j1", task="the task",
        snapshot=[{"role": "user", "content": "q"}], system_prompt="DEV", model="gpt-5.6-sol")
    # Milestone (model-authored) reached the card body.
    final_body = _card_body(client.card_updates[-1])
    assert "✓ Searched dockets — found the final rule." in final_body
    assert final_body.startswith("✅ ")  # headline flipped on success
    # The job's tools include the server tools AND report_progress.
    tools = stub.kwargs["tools"]
    assert {"type": "web_search"} in tools
    assert any(t.get("name") == "report_progress" for t in tools)
    # The registry passed to the loop dispatches report_progress (and only that).
    assert stub.kwargs["registry"].has_tools({})
    # Trailer attributes research sources only — card bookkeeping excluded.
    assert "tools: web_search" in client.sent[0][2]
    assert "report_progress" not in client.sent[0][2]


# --------------------------------------------- short byline tag (label param)

@pytest.mark.asyncio
async def test_executor_passes_label_hint_to_job(monkeypatch):
    """The model's short topic tag rides the detach into the job (whitespace-collapsed)."""
    monkeypatch.setattr(config, "enable_deep_research", True)
    seen = {}

    def _fake_job(**kwargs):
        seen.update(kwargs)

        async def _noop():
            return None
        return _noop()

    monkeypatch.setattr(rt, "_run_deep_research_job", _fake_job)
    proc = _FakeProcessor(openai_client=SimpleNamespace(), tm=AsyncThreadStateManager())
    res = await rt.execute_start_deep_research(
        _ctx(proc, _FakeClient()),
        {"task": "long fully-restated task", "label": "  fast-casual\n2026 performance "})
    assert res["ok"] is True
    assert seen["label_hint"] == "fast-casual 2026 performance"
    await proc.scheduled[0]  # run the captured no-op job


@pytest.mark.asyncio
async def test_short_label_hint_survives_untruncated_on_card_and_findings(monkeypatch):
    """A short topic tag renders whole inside Slack's 50-char username cap — same byline on
    the card and the findings; the long task no longer forces a truncated gist."""
    monkeypatch.setattr(config, "enable_deep_research", True)
    monkeypatch.setattr(config, "enable_research_label", True)
    monkeypatch.setattr(config, "bot_name_aliases", ["ChatGPT"])
    stub = _StreamStub(text="findings body")
    proc = _FakeProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=stub))
    proc.thread_manager = AsyncThreadStateManager()
    client = _CardClient()
    await rt._run_deep_research_job(
        processor=proc, client=client, channel_id="C1", thread_root="100.0",
        thread_key="C1:100.0", job_id="j1",
        task="Produce a sourced, cross-checked report on the three major US fast-casual "
             "chains' 2026 performance including traffic, comps, and expansion plans",
        label_hint="fast-casual 2026 outlook",
        snapshot=[], system_prompt="DEV", model="gpt-5.6-sol")
    expected = "ChatGPT [research: fast-casual 2026 outlook]"
    assert len(expected) <= 50
    assert client.card_posts[0][4] == expected      # untruncated, bracket intact
    assert client.sent[0][3] == expected            # findings byline identical
    # No hint → falls back to the (gisted) task text, still bracket-safe.
    label = rt._research_label(proc, "fast-casual 2026 outlook")
    assert label == expected


@pytest.mark.asyncio
async def test_report_progress_rejects_empty_milestone(monkeypatch):
    """The REAL job executor: a blank milestone is a structured error and adds nothing
    to the card body."""
    monkeypatch.setattr(config, "enable_deep_research", True)
    monkeypatch.setattr(config, "enable_research_label", False)

    class _BlankMilestoneStub(_StreamStub):
        async def __call__(self, **kwargs):
            res = await kwargs["registry"].dispatch(
                kwargs.get("tool_context"), "report_progress", {"milestone": "   "})
            assert res["ok"] is False and res["error"] == "missing_milestone"
            return {"text": "findings", "tools_used": [], "local_tool_calls": []}

    stub = _BlankMilestoneStub()
    proc = _FakeProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=stub))
    proc.thread_manager = AsyncThreadStateManager()
    client = _CardClient()
    await rt._run_deep_research_job(
        processor=proc, client=client, channel_id="C1", thread_root="100.0",
        thread_key="C1:100.0", job_id="j1", task="the task", snapshot=[],
        system_prompt="DEV", model="gpt-5.6-sol")
    # No milestone line landed — only headline + terminal in the final body.
    final_body = _card_body(client.card_updates[-1])
    assert "milestone" not in final_body.lower()
    assert final_body.startswith("✅ ") and "✓ Reported findings below." in final_body
