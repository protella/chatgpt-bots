"""cancel_background_job — the stop button a background job never had (+ the spinner it leaves).

Live 2026-08-09: asked to stand down on a doc job, the bot agreed in words and the job ran on
for eight more minutes, then delivered the document anyway. Agreeing was literally the only
thing it could do. These cover the tool that makes the agreement real — resolution across BOTH
detached registries (background jobs and image generations) and every refusal, the delivery
seam past which cancelling would be a lie, first-cancel-wins under a parallel tool round — and
Fix 4: a terminal card must not leave an animated loader spinning over work that stopped.
"""
import asyncio
from types import SimpleNamespace

import pytest

from config import config
from thread_manager import AsyncThreadStateManager
from tool_registry import ToolContext, ToolRegistry
import message_processor.research_tools as rt


# --------------------------------------------------------------- fakes

class _FakeClient:
    def __init__(self):
        self.sent = []          # (channel, thread, text, username)
        self.card_posts = []    # (channel, thread, text, blocks, username)
        self.card_updates = []  # (channel, ts, text, blocks)

    async def send_message(self, channel_id, thread_id, text, blocks=None, lease=None,
                           surface=None, receipts=None, receipt_kind=None, meta_out=None,
                           username=None, receipt_class=None):
        self.sent.append((channel_id, thread_id, text, username))
        return "9999.000001"

    async def post_status_card(self, channel_id, thread_id, text, blocks, username=None,
                               receipts=None, *, receipt_class):
        self.card_posts.append((channel_id, thread_id, text, blocks, username))
        return "CARD.1"

    async def update_status_card(self, channel_id, ts, text, blocks, receipts=None):
        self.card_updates.append((channel_id, ts, text, blocks))
        return True


class _SlowCardClient(_FakeClient):
    """The card post parked mid-flight, so a cancel can land between "Slack accepted it" and
    "self.ts assigned" — the window the shield in _run_background_job exists to cover."""

    def __init__(self):
        super().__init__()
        self.posting = asyncio.Event()
        self.release = asyncio.Event()

    async def post_status_card(self, channel_id, thread_id, text, blocks, username=None,
                               receipts=None, *, receipt_class):
        self.posting.set()
        await self.release.wait()
        return await super().post_status_card(channel_id, thread_id, text, blocks,
                                              username=username, receipts=receipts,
                                              receipt_class=receipt_class)


class _FakeProcessor:
    def __init__(self, openai_client=None, tm=None):
        self.openai_client = openai_client
        self.thread_manager = tm

    def _build_tools_array(self, cfg, model, registry=None):
        return [{"type": "web_search"}]

    def log_info(self, *a, **k):
        pass

    log_error = log_warning = log_debug = log_info


class _StallingStream:
    """The job's model call, parked until released. It YIELDS and it TERMINATES (CLAUDE.md
    pitfall 6): a stub that spins or never ends wedges the whole suite."""

    def __init__(self, text="the findings"):
        self.text = text
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, **kwargs):
        self.started.set()
        await self.release.wait()
        return {"text": self.text, "tools_used": [], "local_tool_calls": []}


class _ReturningStream:
    def __init__(self, text="the findings", raises=None, slow=False):
        self.text = text
        self.raises = raises
        self.slow = slow

    async def __call__(self, **kwargs):
        if self.slow:
            await asyncio.sleep(5)
        if self.raises is not None:
            raise self.raises
        return {"text": self.text, "tools_used": [], "local_tool_calls": []}


class _SeamTM(AsyncThreadStateManager):
    """Records the delivery-started mark onto the SAME timeline as the delivery calls, so a test
    can assert the flag lands before anything is posted rather than merely that it lands."""

    def __init__(self, events):
        super().__init__(db=None)
        self.events = events

    def mark_research_delivery_started(self, thread_key, job_id):
        self.events.append("mark")
        super().mark_research_delivery_started(thread_key, job_id)


class _SeamClient(_FakeClient):
    """Records every card render onto the shared timeline, so the tests can pin that the
    delivery mark lands before the card is finalized — not merely before the failure note."""

    def __init__(self, events):
        super().__init__()
        self.events = events

    async def update_status_card(self, channel_id, ts, text, blocks, receipts=None):
        self.events.append("card")
        return await super().update_status_card(channel_id, ts, text, blocks,
                                                receipts=receipts)


def _ctx(processor, client, *, thread_ts="100.0", trigger_ts="100.0", channel_id="C1"):
    return ToolContext(channel_id=channel_id, thread_ts=thread_ts, trigger_ts=trigger_ts,
                       client=client, processor=processor)


def _card_body(update_or_post):
    """The section block's mrkdwn text from a recorded card post/update tuple."""
    return update_or_post[3][0]["text"]["text"]


def _job_task(proc, client, *, job_id="j1", thread_key="C1:100.0",
              task="map the Q3 pricing shifts"):
    return asyncio.ensure_future(rt._run_background_job(
        processor=proc, client=client, channel_id="C1", thread_root="100.0",
        thread_key=thread_key, job_id=job_id, task=task, snapshot=[],
        system_prompt=None, model="gpt-5.6-sol"))


async def _live_task():
    """A task that is genuinely in flight, for the registry to cancel."""
    await asyncio.sleep(30)


# --------------------------------------------------------------- schema + registration

def test_schema_shape_and_registration():
    schema = rt.get_cancel_background_job_schema()
    assert schema["type"] == "function"
    assert schema["name"] == "cancel_background_job"
    props = schema["parameters"]["properties"]
    assert props["job_id"]["type"] == "string"
    assert props["reason"]["type"] == "string"
    # job_id is omittable — one job in flight is the common case. The reason never is: it
    # becomes the card's last line.
    assert schema["parameters"]["required"] == ["reason"]
    # Agreeing to stop is not stopping — the description has to say so, or the model keeps
    # doing what it did live.
    assert "only this call does" in schema["description"]

    reg = ToolRegistry()
    rt.register_research_tools(reg)
    names = {s["name"] for s in reg.schemas({})}
    assert {"start_background_job", "cancel_background_job"} <= names


# --------------------------------------------------------------- executor: refusals

@pytest.mark.asyncio
async def test_no_job_running():
    proc = _FakeProcessor(tm=AsyncThreadStateManager(db=None))
    res = await rt.execute_cancel_background_job(_ctx(proc, _FakeClient()), {"reason": "stop"})
    assert res == {"ok": False, "error": "no_job_running"}


@pytest.mark.asyncio
async def test_unavailable_without_a_processor():
    res = await rt.execute_cancel_background_job(
        _ctx(None, _FakeClient()), {"reason": "stop"})
    # Exact result: these dicts are the model's whole view of what happened, so an added key
    # or reworded message is a contract change, not a detail.
    assert res == {"ok": False, "error": "unavailable",
                   "message": "background jobs are not available here"}


@pytest.mark.asyncio
async def test_unknown_job_id_hands_back_the_roster():
    tm = AsyncThreadStateManager(db=None)
    tm.register_research("C1:100.0", "aaa", "map the Q3 pricing shifts", mode="research",
                         deliverables=["pricing.pdf"])
    proc = _FakeProcessor(tm=tm)
    res = await rt.execute_cancel_background_job(
        _ctx(proc, _FakeClient()), {"job_id": "zzz", "reason": "stop"})
    assert res["ok"] is False and res["error"] == "job_not_found"
    # id + gist + kind, nothing else: the model is picking WHICH work to stop, not reading a
    # status report — but it does need to know what kind of thing it is stopping.
    assert res["in_flight"] == [{"job_id": "aaa", "task_summary": "map the Q3 pricing shifts",
                                 "kind": "job"}]


@pytest.mark.asyncio
async def test_a_non_string_job_id_names_no_job():
    """Review r4 ruling: only an actual WRONG TYPE falls here. A number or a list is not a
    half-given id the way null and blank are — it names nothing, so the model gets the roster
    back rather than having one job picked for it."""
    tm = AsyncThreadStateManager(db=None)
    tm.register_research("C1:100.0", "aaa", "map the Q3 pricing shifts")
    proc = _FakeProcessor(tm=tm)
    for bad_id in (7, ["aaa"]):
        res = await rt.execute_cancel_background_job(
            _ctx(proc, _FakeClient()), {"job_id": bad_id, "reason": "stop"})
        assert res == {
            "ok": False, "error": "job_not_found",
            "in_flight": [{"job_id": "aaa", "task_summary": "map the Q3 pricing shifts",
                           "kind": "job"}],
        }


@pytest.mark.asyncio
async def test_an_explicit_null_job_id_reads_as_omitted():
    """Review r4 ruling: models serialize an unset optional as null, so an explicit null is
    the same intent as leaving the field out — with one candidate it resolves to that one."""
    tm = AsyncThreadStateManager(db=None)
    task = asyncio.ensure_future(_live_task())
    tm.register_research("C1:100.0", "aaa", "map the Q3 pricing shifts", task=task)
    proc = _FakeProcessor(tm=tm)
    res = await rt.execute_cancel_background_job(
        _ctx(proc, _FakeClient()), {"job_id": None, "reason": "stop"})
    assert res == {"ok": True, "kind": "job", "job_id": "aaa",
                   "task_summary": "map the Q3 pricing shifts"}
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_an_explicit_null_job_id_is_still_ambiguous_with_two():
    """The other half of the same ruling: reading null as omitted must not become a licence to
    pick one — with two candidates it is refused exactly as an omitted id is."""
    tm = AsyncThreadStateManager(db=None)
    tm.register_research("C1:100.0", "aaa", "map the Q3 pricing shifts")
    tm.register_research("C1:100.0", "bbb", "build the onboarding deck")
    proc = _FakeProcessor(tm=tm)
    res = await rt.execute_cancel_background_job(
        _ctx(proc, _FakeClient()), {"job_id": None, "reason": "stop"})
    assert res == {
        "ok": False, "error": "job_id_required",
        "in_flight": [
            {"job_id": "aaa", "task_summary": "map the Q3 pricing shifts", "kind": "job"},
            {"job_id": "bbb", "task_summary": "build the onboarding deck", "kind": "job"},
        ],
    }


@pytest.mark.asyncio
async def test_two_jobs_need_an_explicit_id():
    tm = AsyncThreadStateManager(db=None)
    tm.register_research("C1:100.0", "aaa", "map the Q3 pricing shifts")
    tm.register_research("C1:100.0", "bbb", "build the onboarding deck")
    proc = _FakeProcessor(tm=tm)
    res = await rt.execute_cancel_background_job(_ctx(proc, _FakeClient()), {"reason": "stop"})
    # Exact, including the complete projections — added keys or reworded gists are drift.
    assert res == {
        "ok": False, "error": "job_id_required",
        "in_flight": [
            {"job_id": "aaa", "task_summary": "map the Q3 pricing shifts", "kind": "job"},
            {"job_id": "bbb", "task_summary": "build the onboarding deck", "kind": "job"},
        ],
    }


@pytest.mark.asyncio
async def test_a_blank_job_id_reads_as_omitted():
    """Review r3 ruling: a blank/whitespace id is not an id, so it is treated as omission —
    which means with two entries in flight it is refused rather than resolving to one of
    them. A stripped-to-nothing string must never pick a job by luck."""
    tm = AsyncThreadStateManager(db=None)
    tm.register_research("C1:100.0", "aaa", "map the Q3 pricing shifts")
    tm.register_research("C1:100.0", "bbb", "build the onboarding deck")
    proc = _FakeProcessor(tm=tm)
    res = await rt.execute_cancel_background_job(
        _ctx(proc, _FakeClient()), {"job_id": "   ", "reason": "stop"})
    assert res["ok"] is False and res["error"] == "job_id_required"
    assert [j["job_id"] for j in res["in_flight"]] == ["aaa", "bbb"]


@pytest.mark.asyncio
async def test_cancelling_an_in_flight_image_generation():
    """Fix 1b: the requirement is stopping unwanted WORK, and a detached image generation is
    the other registry. Its own CancelledError handler clears the progress surface."""
    tm = AsyncThreadStateManager(db=None)
    task = asyncio.ensure_future(_live_task())
    tm.register_generation("C1:100.0", "gen777", "a cat wearing sunglasses", task=task)
    proc = _FakeProcessor(tm=tm)
    res = await rt.execute_cancel_background_job(
        _ctx(proc, _FakeClient()),
        {"job_id": "gen777", "reason": "the user changed their mind"})
    assert res["ok"] is True and res["kind"] == "image_generation"
    assert res["job_id"] == "gen777" and res["task_summary"] == "a cat wearing sunglasses"
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_a_cancelled_generation_clears_its_registry_entry_and_upload_token():
    """A generation cancelled before its coroutine's first tick runs no except and no finally,
    so the entry would claim work is running here forever and its upload token would never
    drain — every later turn then waits on an upload that will never land. The done callback
    on attach_generation_task is the belt to the coroutine's braces.

    NOT covered, deliberately: the progress surface. Clearing a checklist is an await and a
    done callback cannot await — the same accepted class as the research pre-body edge, and a
    task cancelled this early posted no surface to clear."""
    tm = AsyncThreadStateManager(db=None)
    task = asyncio.ensure_future(_live_task())
    tm.register_generation("C1:100.0", "gen777", "a cat wearing sunglasses")
    tm.attach_generation_task("C1:100.0", "gen777", task)
    tm.mark_upload_started("C1:100.0", "gen777")
    assert tm._upload_pending["C1:100.0"] == {"gen777"}

    proc = _FakeProcessor(tm=tm)
    res = await rt.execute_cancel_background_job(
        _ctx(proc, _FakeClient()), {"job_id": "gen777", "reason": "stop"})
    assert res["ok"] is True
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0)          # let the done callback run

    assert tm.generations_in_flight("C1:100.0") == []
    assert not tm._upload_pending.get("C1:100.0")


@pytest.mark.asyncio
async def test_a_second_generation_cancel_is_refused():
    """First-cancel-wins for generations too — and the second call must not re-cancel the
    task, or a retry would keep poking a task that is already on its way out."""
    cancels = []

    class _CountingTask:
        def done(self):
            return False

        def cancel(self):
            cancels.append(1)
            return True

    tm = AsyncThreadStateManager(db=None)
    tm.register_generation("C1:100.0", "gen777", "a cat wearing sunglasses",
                           task=_CountingTask())
    proc = _FakeProcessor(tm=tm)
    first = await rt.execute_cancel_background_job(
        _ctx(proc, _FakeClient()), {"job_id": "gen777", "reason": "stop"})
    assert first["ok"] is True
    second = await rt.execute_cancel_background_job(
        _ctx(proc, _FakeClient()), {"job_id": "gen777", "reason": "stop again"})
    assert second == {"ok": False, "error": "already_cancelling"}
    assert cancels == [1]


@pytest.mark.asyncio
async def test_a_job_and_a_generation_together_need_an_explicit_id():
    """An omitted id is only unambiguous across BOTH registries: with a deck building and an
    image rendering, "stop it" names neither."""
    tm = AsyncThreadStateManager(db=None)
    job_task = asyncio.ensure_future(_live_task())
    gen_task = asyncio.ensure_future(_live_task())
    tm.register_research("C1:100.0", "aaa", "map the Q3 pricing shifts", task=job_task)
    tm.register_generation("C1:100.0", "gen777", "a cat wearing sunglasses", task=gen_task)
    proc = _FakeProcessor(tm=tm)
    res = await rt.execute_cancel_background_job(_ctx(proc, _FakeClient()), {"reason": "stop"})
    assert res["ok"] is False and res["error"] == "job_id_required"
    assert [j["job_id"] for j in res["in_flight"]] == ["aaa", "gen777"]
    # Symmetric: BOTH kinds are named, so the model can say what it stopped rather than
    # inferring it from which registry the id happened to come out of.
    assert [j["kind"] for j in res["in_flight"]] == ["job", "image_generation"]
    # Ambiguity cancels NOTHING — a refusal that stopped one of them would be worse than none.
    assert not job_task.cancelled() and not gen_task.cancelled()
    for task in (job_task, gen_task):
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_a_finished_task_cannot_be_cancelled():
    tm = AsyncThreadStateManager(db=None)
    done = asyncio.ensure_future(asyncio.sleep(0))
    await done
    tm.register_research("C1:100.0", "aaa", "map the Q3 pricing shifts", task=done)
    proc = _FakeProcessor(tm=tm)
    res = await rt.execute_cancel_background_job(_ctx(proc, _FakeClient()), {"reason": "stop"})
    assert res == {"ok": False, "error": "job_already_finished"}


@pytest.mark.asyncio
async def test_a_refused_cancel_is_not_reported_as_stopped():
    """`Task.cancel()` returning False means the job is already past cancelling. Reporting ok
    there would have the model announce a stand-down over work that is still delivering — and
    the reason must not stick, or the retry would come back `already_cancelling`."""
    tm = AsyncThreadStateManager(db=None)
    tm.register_research("C1:100.0", "aaa", "map the Q3 pricing shifts",
                         task=SimpleNamespace(done=lambda: False, cancel=lambda: False))
    proc = _FakeProcessor(tm=tm)
    res = await rt.execute_cancel_background_job(_ctx(proc, _FakeClient()), {"reason": "stop"})
    assert res == {"ok": False, "error": "job_already_finished"}
    assert tm.research_cancel_reason("C1:100.0", "aaa") is None


@pytest.mark.asyncio
async def test_first_cancel_wins():
    """A round's tool calls dispatch under asyncio.gather, so two cancels of one job are a real
    shape. The second is refused rather than allowed to overwrite the first one's reason."""
    tm = AsyncThreadStateManager(db=None)
    task = asyncio.ensure_future(_live_task())
    tm.register_research("C1:100.0", "aaa", "map the Q3 pricing shifts", task=task)
    proc = _FakeProcessor(tm=tm)
    first = await rt.execute_cancel_background_job(
        _ctx(proc, _FakeClient()), {"reason": "the user asked me to stand down"})
    assert first["ok"] is True
    second = await rt.execute_cancel_background_job(
        _ctx(proc, _FakeClient()), {"reason": "some other reason"})
    assert second == {"ok": False, "error": "already_cancelling"}
    assert tm.research_cancel_reason("C1:100.0", "aaa") == "the user asked me to stand down"
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_a_job_that_is_already_posting_is_too_late_to_cancel():
    tm = AsyncThreadStateManager(db=None)
    task = asyncio.ensure_future(_live_task())
    tm.register_research("C1:100.0", "aaa", "map the Q3 pricing shifts", task=task)
    tm.mark_research_delivery_started("C1:100.0", "aaa")
    proc = _FakeProcessor(tm=tm)
    res = await rt.execute_cancel_background_job(_ctx(proc, _FakeClient()), {"reason": "stop"})
    # Exact, hint included: the hint is what stops the model claiming it stood the job down.
    assert res == {"ok": False, "error": "delivery_in_progress",
                   "hint": "the job is already posting its results; too late to cancel"}
    assert not task.cancelled()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_a_blank_reason_becomes_the_default():
    tm = AsyncThreadStateManager(db=None)
    task = asyncio.ensure_future(_live_task())
    tm.register_research("C1:100.0", "aaa", "map the Q3 pricing shifts", task=task)
    proc = _FakeProcessor(tm=tm)
    res = await rt.execute_cancel_background_job(_ctx(proc, _FakeClient()), {"reason": "   "})
    assert res["ok"] is True
    assert tm.research_cancel_reason("C1:100.0", "aaa") == "cancelled by request"
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_thread_ts_none_falls_back_to_trigger_ts():
    """Same key the start tool builds — a job dispatched from a top-level message lives under
    its trigger_ts, and a cancel that keyed differently would never find it."""
    tm = AsyncThreadStateManager(db=None)
    task = asyncio.ensure_future(_live_task())
    tm.register_research("C1:200.0", "aaa", "map the Q3 pricing shifts", task=task)
    proc = _FakeProcessor(tm=tm)
    res = await rt.execute_cancel_background_job(
        _ctx(proc, _FakeClient(), thread_ts=None, trigger_ts="200.0"), {"reason": "stop"})
    assert res["ok"] is True and res["job_id"] == "aaa"
    with pytest.raises(asyncio.CancelledError):
        await task


# --------------------------------------------------------------- the job side

@pytest.mark.asyncio
async def test_cancelling_the_one_running_job_ends_its_card(monkeypatch):
    monkeypatch.setattr(config, "enable_research_label", False)
    tm = AsyncThreadStateManager(db=None)
    stream = _StallingStream()
    proc = _FakeProcessor(
        openai_client=SimpleNamespace(create_streaming_response_with_tool_loop=stream), tm=tm)
    client = _FakeClient()
    tm.register_research("C1:100.0", "j1", "map the Q3 pricing shifts")
    task = _job_task(proc, client)
    tm.attach_research_task("C1:100.0", "j1", task)
    await asyncio.wait_for(stream.started.wait(), timeout=2)

    res = await rt.execute_cancel_background_job(
        _ctx(proc, client), {"reason": "  the user   asked me to stand down "})
    assert res["ok"] is True and res["kind"] == "job"
    assert res["job_id"] == "j1" and res["task_summary"] == "map the Q3 pricing shifts"

    with pytest.raises(asyncio.CancelledError):
        await task
    # Whitespace-normalized on the way in, and the card says who stopped it and why.
    assert _card_body(client.card_updates[-1]) == \
        "❌ cancelled — the user asked me to stand down"
    # A cancelled job delivers nothing: no findings, no failure note.
    assert client.sent == []
    assert tm.research_in_flight_count("C1:100.0") == 0


@pytest.mark.asyncio
async def test_cancel_during_the_card_post_still_finalizes_the_card(monkeypatch):
    """The narrow window: Slack has accepted the card but `self.ts` is not assigned yet, and
    `_finalize` is a no-op while ts is None. Without the shield the job dies and leaves a card
    spinning forever."""
    monkeypatch.setattr(config, "enable_research_label", False)
    tm = AsyncThreadStateManager(db=None)
    stream = _StallingStream()
    proc = _FakeProcessor(
        openai_client=SimpleNamespace(create_streaming_response_with_tool_loop=stream), tm=tm)
    client = _SlowCardClient()
    tm.register_research("C1:100.0", "j1", "map the Q3 pricing shifts")
    task = _job_task(proc, client)
    tm.attach_research_task("C1:100.0", "j1", task)
    await asyncio.wait_for(client.posting.wait(), timeout=2)

    res = tm.request_background_cancel("C1:100.0", None, "a teammate is already writing it")
    assert res["ok"] is True
    client.release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(client.card_posts) == 1
    assert _card_body(client.card_updates[-1]) == \
        "❌ cancelled — a teammate is already writing it"


@pytest.mark.asyncio
async def test_a_cancel_racing_the_jobs_own_failure_leaves_one_terminal_render(monkeypatch):
    """The real race, driven through the real job: the job is failing (its stream raised) and
    a cancel arrives in the same window.

    The delivery marker is what resolves it. Once the failure handler has marked delivery the
    cancel is REFUSED — the task is never cancelled, the failure handler owns the ending, and
    the card carries exactly ONE terminal render saying what actually happened."""
    monkeypatch.setattr(config, "enable_research_label", False)
    tm = AsyncThreadStateManager(db=None)
    proc = _FakeProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=_ReturningStream(
            raises=RuntimeError("boom"))), tm=tm)
    client = _FakeClient()
    tm.register_research("C1:100.0", "j1", "map the Q3 pricing shifts")

    failing = asyncio.Event()
    release = asyncio.Event()

    async def _fail(_client, _channel_id, _thread_root, _reason, receipts=None):
        # Parked INSIDE the failure path, after the card has been finalized — the exact
        # instant a user's cancel would land on a job that is already ending.
        failing.set()
        await release.wait()

    monkeypatch.setattr(rt, "_deliver_failure", _fail)
    task = _job_task(proc, client)
    tm.attach_research_task("C1:100.0", "j1", task)
    await asyncio.wait_for(failing.wait(), timeout=2)

    res = await rt.execute_cancel_background_job(
        _ctx(proc, client), {"reason": "the user asked me to stand down"})
    assert res["ok"] is False and res["error"] == "delivery_in_progress"
    assert not task.cancelled()

    release.set()
    await task
    # ONE render, and it tells the truth: the job died of its own error, not of the cancel.
    assert len(client.card_updates) == 1
    assert "hit a wall" in _card_body(client.card_updates[0])
    assert "cancelled" not in _card_body(client.card_updates[0])


@pytest.mark.asyncio
async def test_the_shutdown_path_keeps_its_own_wording():
    client = _FakeClient()
    card = rt._ResearchCard(processor=_FakeProcessor(), client=client, channel_id="C1",
                            thread_root="100.0", task="map the Q3 pricing shifts", label=None)
    card.ts = "CARD.1"
    await card.finalize_cancelled()
    assert _card_body(client.card_updates[-1]) == "❌ cancelled (bot shutting down)"


@pytest.mark.asyncio
async def test_a_cancel_during_the_build_releases_the_container_binding(monkeypatch):
    """The build phase binds its OWN container (`{thread_key}#job:{job_id}`) and the success
    path releases it. A cancel never returns from the phase, so before this the binding was
    stranded in the manager until it expired on its own — and a user-triggered cancel makes
    that a normal ending rather than a shutdown curiosity."""
    monkeypatch.setattr(config, "enable_research_label", False)
    invalidated = []

    class _Manager:
        async def create_explicit(self, ledger_key):
            return "cntr_build_1"

        async def invalidate(self, ledger_key, container_id=None):
            invalidated.append(ledger_key)

    building = asyncio.Event()

    async def _build(*a, **k):
        building.set()
        await asyncio.sleep(30)      # parked inside the build, holding the binding

    tm = AsyncThreadStateManager(db=None)
    proc = _FakeProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=_ReturningStream("the findings")), tm=tm)
    proc.container_manager = _Manager()
    monkeypatch.setattr(rt, "_run_build_phase", _build)

    tm.register_research("C1:100.0", "j1", "build the onboarding deck")
    task = asyncio.ensure_future(rt._run_background_job(
        processor=proc, client=_FakeClient(), channel_id="C1", thread_root="100.0",
        thread_key="C1:100.0", job_id="j1", task="build the onboarding deck", snapshot=[],
        system_prompt=None, model="gpt-5.6-sol", mode="research_and_build",
        deliverables=[{"type": "powerpoint", "description": "the deck",
                       "filename": "onboarding.pptx"}]))
    tm.attach_research_task("C1:100.0", "j1", task)
    await asyncio.wait_for(building.wait(), timeout=2)

    res = await rt.execute_cancel_background_job(
        _ctx(proc, _FakeClient()), {"reason": "the user asked me to stand down"})
    assert res["ok"] is True
    with pytest.raises(asyncio.CancelledError):
        await task
    assert invalidated == ["C1:100.0#job:j1"]


# --------------------------------------------------------------- the delivery seam

@pytest.mark.asyncio
async def test_delivery_flag_is_set_before_the_delivery_transacts(monkeypatch):
    monkeypatch.setattr(config, "enable_research_label", False)
    events = []
    tm = _SeamTM(events)
    tm.register_research("C1:100.0", "j1", "map the Q3 pricing shifts")
    proc = _FakeProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=_ReturningStream("the findings")), tm=tm)

    async def _plan(*a, **k):
        events.append("plan")
        return {"reply": "here it is", "publish": [], "post_report": True}

    async def _transact(*a, **k):
        events.append("transact")
        return True

    monkeypatch.setattr(rt, "_plan_delivery", _plan)
    monkeypatch.setattr(rt, "_transact_delivery", _transact)
    await _job_task(proc, _FakeClient())
    # Planning is still cancellable work; transacting is not.
    assert events == ["plan", "mark", "transact"]


@pytest.mark.asyncio
async def test_delivery_flag_is_set_before_the_empty_research_failure(monkeypatch):
    monkeypatch.setattr(config, "enable_research_label", False)
    events = []
    tm = _SeamTM(events)
    tm.register_research("C1:100.0", "j1", "map the Q3 pricing shifts")
    proc = _FakeProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=_ReturningStream("")), tm=tm)

    async def _fail(client, channel_id, thread_root, reason, receipts=None):
        events.append("failure_note")

    monkeypatch.setattr(rt, "_deliver_failure", _fail)
    await _job_task(proc, _SeamClient(events))
    # The CARD is terminal output too — the mark has to precede finalize_failure, not merely
    # the failure note. Recording only the note would pass with the mark moved between them.
    assert events == ["mark", "card", "failure_note"]


@pytest.mark.asyncio
async def test_delivery_flag_is_set_before_the_timeout_note(monkeypatch):
    monkeypatch.setattr(config, "enable_research_label", False)
    monkeypatch.setattr(config, "deep_research_timeout", 0.01)
    events = []
    tm = _SeamTM(events)
    tm.register_research("C1:100.0", "j1", "map the Q3 pricing shifts")
    proc = _FakeProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=_ReturningStream(slow=True)), tm=tm)

    async def _fail(client, channel_id, thread_root, reason, receipts=None):
        events.append("failure_note")

    monkeypatch.setattr(rt, "_deliver_failure", _fail)
    await _job_task(proc, _SeamClient(events))
    assert events == ["mark", "card", "failure_note"]


@pytest.mark.asyncio
async def test_delivery_flag_is_set_before_the_error_note(monkeypatch):
    monkeypatch.setattr(config, "enable_research_label", False)
    events = []
    tm = _SeamTM(events)
    tm.register_research("C1:100.0", "j1", "map the Q3 pricing shifts")
    proc = _FakeProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=_ReturningStream(
            raises=RuntimeError("boom"))), tm=tm)

    async def _fail(client, channel_id, thread_root, reason, receipts=None):
        events.append("failure_note")

    monkeypatch.setattr(rt, "_deliver_failure", _fail)
    await _job_task(proc, _SeamClient(events))
    assert events == ["mark", "card", "failure_note"]


# --------------------------------------------------------------- Fix 4: the terminal spinner

@pytest.mark.asyncio
async def test_a_terminal_card_stops_the_spinner(monkeypatch):
    """The retained in-flight item is the most important line on a failed card — it says where
    the job stopped. Rendered with the workspace's animated loader it says the opposite."""
    monkeypatch.setattr(config, "circle_loader_emoji", ":circle-loader:")
    client = _FakeClient()
    card = rt._ResearchCard(processor=_FakeProcessor(), client=client, channel_id="C1",
                            thread_root="100.0", task="map the Q3 pricing shifts", label=None)
    card.ts = "CARD.1"
    assert card.todos.set([{"text": "Pull the pricing per vendor", "status": "done"},
                           {"text": "Draft the summary", "status": "in_progress"}]) is None
    # While it runs, the loader is exactly what that line should show.
    assert ":circle-loader: Draft the summary" in "\n".join(card._visible_lines())

    await card.finalize_cancelled("the user asked me to stand down")
    body = _card_body(client.card_updates[-1])
    assert ":circle-loader:" not in body
    assert "⏹ Draft the summary" in body
    assert "✓ Pull the pricing per vendor" in body
    assert body.endswith("❌ cancelled — the user asked me to stand down")


@pytest.mark.asyncio
async def test_the_spinner_stops_on_every_failed_ending(monkeypatch):
    monkeypatch.setattr(config, "circle_loader_emoji", ":circle-loader:")
    client = _FakeClient()
    card = rt._ResearchCard(processor=_FakeProcessor(), client=client, channel_id="C1",
                            thread_root="100.0", task="map the Q3 pricing shifts", label=None)
    card.ts = "CARD.1"
    assert card.todos.set([{"text": "Build the deck", "status": "in_progress"}]) is None
    await card.finalize_failure("it ran past the time limit before finishing")
    body = _card_body(client.card_updates[-1])
    assert ":circle-loader:" not in body
    assert "⏹ Build the deck" in body


# --------------------------------------------------------------- the in-flight note

class _SuffixHost:
    def __init__(self, tm):
        from message_processor.utilities import MessageUtilitiesMixin
        self._build_research_inflight_note = (
            MessageUtilitiesMixin._build_research_inflight_note.__get__(self))
        self._build_generation_inflight_note = (
            MessageUtilitiesMixin._build_generation_inflight_note.__get__(self))
        self._escape_suffix_text = MessageUtilitiesMixin._escape_suffix_text
        self.thread_manager = tm

    def log_debug(self, *a, **k):
        pass


def test_the_inflight_note_carries_a_job_id_the_model_can_cancel_with():
    tm = AsyncThreadStateManager(db=None)
    tm.register_research("C1:T1", "aaa111bbb222", "map the Q3 pricing shifts", mode="research")
    tm.register_research("C1:T1", "ccc333ddd444", "build the onboarding deck",
                         mode="research_and_build", deliverables=["onboarding.pptx"])
    note = _SuffixHost(tm)._build_research_inflight_note("C1", "T1")
    assert note is not None
    # Every line, not just the first — with two jobs running the id is the only way to say
    # which one to stop.
    assert '- research [job aaa111bbb222]: "map the Q3 pricing shifts"' in note
    assert ('- research_and_build [job ccc333ddd444]: "build the onboarding deck" '
            '→ onboarding.pptx') in note
    assert "cancel_background_job(job_id, reason)" in note


def test_the_generation_note_carries_ids_the_model_can_cancel_with():
    tm = AsyncThreadStateManager(db=None)
    host = _SuffixHost(tm)
    tm.register_generation("C1:T1", "gen777", "a cat wearing sunglasses")
    note = host._build_generation_inflight_note("C1", "T1")
    assert note is not None
    assert '"a cat wearing sunglasses" [gen gen777]' in note
    assert "cancel_background_job(job_id, reason)" in note

    tm.register_generation("C1:T1", "gen888", "a dog on a skateboard")
    note = host._build_generation_inflight_note("C1", "T1")
    assert note is not None
    assert "[gen gen777]" in note and "[gen gen888]" in note
    # This note has always been ONE sentence in the volatile suffix block — a newline here
    # would break the block's shape (and test_background_image_gen pins it).
    assert "\n" not in note
