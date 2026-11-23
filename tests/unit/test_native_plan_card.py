"""§B: the background-job status card as a native Slack plan block (SLACK_NATIVE_PLAN_CARD)."""

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from slack_sdk.errors import SlackApiError

import message_processor.research_tools as rt
from config import config
from slack_client.messaging import CardWriteResult, SlackMessagingMixin

# The probed, accepted payload shape (dev app, 2026-11-19): nested task objects with task_id.
EXPECTED_PLAN = {
    "type": "plan", "title": "Checking the probe (1 of 3 done)",
    "tasks": [
        {"type": "task_card", "task_id": "t20aab940a0", "title": "Read the inputs",
         "status": "complete"},
        {"type": "task_card", "task_id": "t5242418169", "title": "Compute the result",
         "status": "in_progress"},
        {"type": "task_card", "task_id": "t94e0d83732", "title": "Post the answer",
         "status": "pending"},
    ],
}


class _Proc:
    def __init__(self) -> None:
        self.infos: List[str] = []

    def log_info(self, msg: str, *a: Any, **k: Any) -> None:
        self.infos.append(msg)

    def log_debug(self, *a: Any, **k: Any) -> None:
        pass


class _Client:
    """Records card posts/updates; rejects the first `reject_posts` / `reject_updates` writes
    that carry a plan block with `error`."""

    def __init__(self, *, reject_posts: int = 0, reject_updates: int = 0,
                 error: str = "invalid_blocks") -> None:
        self.posts: List[Dict[str, Any]] = []
        self.updates: List[List[Dict[str, Any]]] = []
        self.reject_posts = reject_posts
        self.reject_updates = reject_updates
        self.error = error

    async def post_status_card(self, channel_id: str, thread_id: str, text: str,
                               blocks: list, username: Optional[str] = None,
                               receipts: Any = None, *, receipt_class: Optional[str],
                               **kw: Any) -> Optional[str]:
        self.posts.append({"blocks": blocks, "username": username, "kw": kw})
        if blocks[0]["type"] == "plan" and self.reject_posts:
            self.reject_posts -= 1
            if kw.get("error_out") is not None:
                kw["error_out"]["error"] = self.error
            return None
        return "CARD.1"

    async def update_status_card(self, channel_id: str, ts: str, text: str, blocks: list,
                                 receipts: Any = None) -> CardWriteResult:
        self.updates.append(blocks)
        if blocks[0]["type"] == "plan" and self.reject_updates:
            self.reject_updates -= 1
            return CardWriteResult(ok=False, error=self.error)
        return CardWriteResult(ok=True)


def _card(client: Any, *, task: str = "For a teammate, check the probe and report back in detail",
          short_name: Optional[str] = "Checking the probe", label: Optional[str] = None,
          plan: Optional[List[str]] = None) -> Any:
    proc = _Proc()
    return rt._ResearchCard(
        processor=proc, client=client, channel_id="C1", thread_root="100.0", task=task,
        label=label, short_name=short_name, todos=rt._TodoState(plan or ["Read the inputs", "Compute the result",
                                                  "Post the answer"]),
        clock=lambda: 0.0, sleep=lambda d: asyncio.sleep(0), now_label=lambda: "3:00 PM")


def _todos(*pairs: Any) -> List[Dict[str, str]]:
    return [{"text": t, "status": s} for t, s in pairs]


_PROBE_TODOS = _todos(("Read the inputs", "done"), ("Compute the result", "in_progress"),
                      ("Post the answer", "pending"))


@pytest.fixture
def native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "slack_native_plan_card", True)


@pytest.mark.asyncio
async def test_running_card_renders_the_probed_plan_payload(native: None) -> None:
    card = _card(_Client())
    await card.set_todos(_PROBE_TODOS)
    card.note_activity("Searching the web")
    blocks = card._blocks()
    assert blocks[0] == EXPECTED_PLAN
    assert blocks[1] == {"type": "context", "elements": [
        {"type": "plain_text", "text": "Searching the web\ntodos as of 3:00 PM", "emoji": True}]}


def test_task_ids_are_stable_hashes_with_duplicate_suffixes() -> None:
    a = rt._plan_task_ids(["Read the inputs", "Post the answer"])
    b = rt._plan_task_ids(["Post the answer", "Read  the INPUTS"])
    assert a == [b[1], b[0]]                                # reorder + normalization keep ids
    assert rt._plan_task_ids(["Read the outputs"])[0] != a[0]   # rename ⇒ new task
    d = rt._plan_task_ids(["Same", "Other", "same", "Same"])
    assert d[2] == d[0] + "-2" and d[3] == d[0] + "-3" and d[1] != d[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("ending,expected,verdict", [
    ("success", ["complete", "complete", "complete"], "✅ Reported findings below."),
    ("failure", ["complete", "error", "error"], "❌ hit a wall: boom"),
    ("cancelled", ["complete", "error", "error"], "❌ cancelled — stop"),
    ("partial", ["complete", "error", "error"], "⚠️ file missing"),
])
async def test_terminal_states_pin_every_task(native: None, ending: str,
                                              expected: List[str], verdict: str) -> None:
    client = _Client()
    card = _card(client)
    await card.start()
    await card.set_todos(_PROBE_TODOS)
    finalize = {"success": lambda: card.finalize_success(),
                "failure": lambda: card.finalize_failure("boom"),
                "cancelled": lambda: card.finalize_cancelled("stop"),
                "partial": lambda: card.finalize_partial("file missing")}[ending]
    await finalize()
    plan, ctx = client.updates[-1]
    done = expected.count("complete")
    assert plan["title"] == f"Checking the probe ({done} of 3 done)"
    assert [t["status"] for t in plan["tasks"]] == expected
    assert ctx["elements"][0]["text"].split("\n") == [verdict, "todos as of 3:00 PM"]


def test_title_falls_back_to_the_task_gist_and_drops_the_count_without_tasks(
        native: None) -> None:
    card = _card(_Client(), short_name=None)
    card.todos = rt._TodoState()
    assert card._blocks()[0]["title"] == rt._gist(card._task, 60)


@pytest.mark.asyncio
async def test_rejected_labelled_post_reposts_legacy_and_keeps_the_label(native: None) -> None:
    client = _Client(reject_posts=1)
    card = _card(client, label="Sol [research: probe]")
    await card.start()
    assert card.ts == "CARD.1" and card.label == "Sol [research: probe]"
    assert [p["blocks"][0]["type"] for p in client.posts] == ["plan", "section"]
    assert [p["username"] for p in client.posts] == ["Sol [research: probe]"] * 2
    assert not getattr(card.processor, rt._RESEARCH_LABEL_DISABLED_ATTR, False)
    assert card._acked_blocks == client.posts[1]["blocks"]
    await card.set_todos(_PROBE_TODOS)
    card._next_write_at = 0.0
    await card._writer_tick()
    assert client.updates[-1][0]["type"] == "section"       # legacy for the card's lifetime


@pytest.mark.asyncio
async def test_rejected_update_rewrites_the_same_state_as_legacy_now(native: None) -> None:
    client = _Client(reject_updates=1, error="invalid_arguments")
    card = _card(client)
    await card.start()
    assert client.posts[0]["blocks"][0]["type"] == "plan"
    await card.set_todos(_PROBE_TODOS)
    card._next_write_at = 0.0
    await card._writer_tick()
    assert [u[0]["type"] for u in client.updates] == ["plan", "section"]
    assert card._acked_blocks == client.updates[1]
    assert card._acked_revision == card._revision
    assert card._cooldown_until == 0.0 and not card._disabled


@pytest.mark.asyncio
async def test_other_post_errors_do_not_fall_back(native: None) -> None:
    client = _Client(reject_posts=1, error="ratelimited")
    card = _card(client)
    await card.start()
    assert card.ts is None and len(client.posts) == 1 and card._native_plan


@pytest.mark.asyncio
async def test_flag_off_renders_and_posts_legacy_unchanged() -> None:
    assert config.slack_native_plan_card is False
    client = _Client()
    card = _card(client)
    await card.start()
    assert client.posts[0]["kw"] == {}                      # no error_out on the legacy call
    assert client.posts[0]["blocks"] == card._legacy_blocks() == card._blocks()
    assert client.posts[0]["blocks"][0] == {
        "type": "section", "text": {"type": "mrkdwn", "text": (
            "◦ Read the inputs\n◦ Compute the result\n◦ Post the answer\n"
            f"{card._loader_emoji} Researching…")}}


@pytest.mark.asyncio
async def test_post_status_card_reports_the_slack_error_code() -> None:
    host = MagicMock()
    host.app.client.chat_postMessage = AsyncMock(
        side_effect=SlackApiError("bad", {"ok": False, "error": "invalid_blocks"}))
    out: Dict[str, str] = {}
    ts = await SlackMessagingMixin.post_status_card.__get__(host)(
        "C1", "1.0", "fallback", [], receipt_class="background_job", error_out=out)
    assert ts is None and out == {"error": "invalid_blocks"}


@pytest.mark.asyncio
@pytest.mark.parametrize("turn,claims,expected", [
    (SimpleNamespace(job_ack_sink=lambda: True, claim_work=AsyncMock()), True, False),
    (None, False, True),                                         # no sink: job posts the ack
])
async def test_job_ack_sink_decides_whether_the_job_posts_its_ack(
        monkeypatch: pytest.MonkeyPatch, turn: Any, claims: bool, expected: bool) -> None:
    from message_processor.thread_manager import AsyncThreadStateManager
    from message_processor.tool_registry import ToolContext

    monkeypatch.setattr(config, "enable_deep_research", True)
    scheduled: List[Any] = []
    proc = MagicMock()
    proc.thread_manager = AsyncThreadStateManager()

    def _schedule(coro: Any) -> Any:
        scheduled.append(coro)
        return MagicMock()
    proc._schedule_async_call = _schedule
    ctx = ToolContext(channel_id="C1", thread_ts="100.0", trigger_ts="100.0", client=MagicMock(),
                      processor=proc, current_input=[], system_prompt="DEV",
                      model="gpt-5.6-sol", turn=turn)
    res = await rt.execute_start_background_job(ctx, {"task": "t", "plan": ["Do the work"]})
    assert res["ok"] is True
    assert scheduled[0].cr_frame.f_locals["post_ack"] is expected
    scheduled[0].close()
