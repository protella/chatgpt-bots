"""F30 — background deep-dive research jobs.

Covers the start_deep_research schema + registry gating, the per-thread cap, context
snapshotting by copy, the happy-path findings delivery (with trailer), error/timeout failure
notes, the research-label fallback + process-lifetime memory, the thread_manager research
registry / shutdown cancellation, and config defaults.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

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


# --------------------------------------------------------------- schema + gating

def test_schema_shape():
    schema = rt.get_start_deep_research_schema()
    assert schema["name"] == "start_deep_research"
    assert schema["type"] == "function"
    props = schema["parameters"]["properties"]
    assert "task" in props and props["task"]["type"] == "string"
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
    openai = SimpleNamespace(create_text_response_with_tools=AsyncMock(
        return_value={"text": "findings", "tools_used": ["web_search"]}))
    proc = _FakeProcessor(openai_client=openai, tm=AsyncThreadStateManager())
    client = _FakeClient()
    ctx = _ctx(proc, client, current_input=live_input)
    await rt.execute_start_deep_research(ctx, {"task": "research the thing"})
    # Mutate the LIVE input after the call — the job's snapshot must be unaffected.
    live_input.append({"role": "user", "content": "LATE addition"})
    await proc.scheduled[0]  # run the captured job
    job_input = openai.create_text_response_with_tools.call_args.kwargs["messages"]
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
    openai = SimpleNamespace(create_text_response_with_tools=AsyncMock(
        return_value={"text": "# Findings\nThe answer is 42.", "tools_used": ["web_search"]}))
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
    assert username is None


@pytest.mark.asyncio
async def test_error_path_posts_failure_note(monkeypatch):
    monkeypatch.setattr(config, "enable_deep_research", True)
    openai = SimpleNamespace(create_text_response_with_tools=AsyncMock(
        side_effect=RuntimeError("boom")))
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

    async def _slow(*a, **k):
        await asyncio.sleep(5)

    openai = SimpleNamespace(create_text_response_with_tools=_slow)
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
    openai = SimpleNamespace(create_text_response_with_tools=AsyncMock(
        return_value={"text": "done", "tools_used": []}))
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
    openai = SimpleNamespace(create_text_response_with_tools=AsyncMock(
        return_value={"text": "report", "tools_used": []}))
    proc = _FakeProcessor(openai_client=openai, tm=AsyncThreadStateManager())
    proc._build_tools_array = lambda cfg, model, registry=None: [
        {"type": "mcp", "server_label": "x"}]  # truthy, no web_search
    await rt._run_deep_research_job(
        processor=proc, client=_FakeClient(), channel_id="C1", thread_root="100.0",
        thread_key="C1:100.0", job_id="j1", task="t",
        snapshot=[], system_prompt="DEV", model="gpt-5.6-sol")
    tools = openai.create_text_response_with_tools.await_args.kwargs["tools"]
    assert {"type": "web_search"} in tools
    assert {"type": "mcp", "server_label": "x"} in tools


@pytest.mark.asyncio
async def test_failed_findings_post_posts_failure_note(monkeypatch):
    """send_message swallowing a Slack error into None must NOT read as job success:
    the job posts an honest failure note instead of logging 'completed'."""
    monkeypatch.setattr(config, "enable_deep_research", True)
    monkeypatch.setattr(config, "enable_research_label", False)
    openai = SimpleNamespace(create_text_response_with_tools=AsyncMock(
        return_value={"text": "report", "tools_used": []}))
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
