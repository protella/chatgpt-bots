"""update_background_job — steering a job that has already left the station.

Live 2026-08-09, the same thread that produced the cancel button: the bot's doc job was mid-run
when five corrections arrived "before you finish it", and it replied "Got it — I'll fold all
five into the draft." It could not. A job runs on a snapshot deep-copied at dispatch and never
re-reads the thread, so the only two things the model could actually do were let it finish wrong
or kill it outright — and neither was what was asked for, because the work was wanted and only
the current version of it wasn't.

These cover the tool that makes the agreement real, and — just as much — the honesty of what it
claims. A note is only ever "passed along": accepted at a gate that closes when the working
rounds end, injected at the next round, and, when it lands too late for any round, surfaced at
delivery as explicitly NOT applied rather than quietly folded into a cheerful summary.
"""
import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, cast

import pytest

from config import config
from thread_manager import AsyncThreadStateManager
from tool_registry import ToolContext, ToolRegistry
import message_processor.research_tools as rt


THREAD = "C1:100.0"


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


class _FakeProcessor:
    def __init__(self, openai_client=None, tm=None):
        self.openai_client = openai_client
        self.thread_manager = tm

    def _build_tools_array(self, cfg, model, registry=None):
        return [{"type": "web_search"}]

    def log_info(self, *a, **k):
        pass

    log_error = log_warning = log_debug = log_info


class _LoggingProcessor(_FakeProcessor):
    """Keeps the warnings, because on the failure endings the log IS the contract — a note the
    tool accepted and the system then dropped in silence is the one outcome this round exists
    to eliminate."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.warnings: List[str] = []

    def log_warning(self, msg, *a, **k):
        self.warnings.append(str(msg))


class _FakeTask:
    """The job's task, as the refusal ladder actually reads it: three predicates and nothing
    else. A real task here would only add a scheduler and a cleanup burden — the two rungs that
    genuinely depend on asyncio semantics (a task cancelled but not yet finished) use a real one."""

    def __init__(self, done: bool = False, cancelling: int = 0):
        self._done = done
        self._cancelling = cancelling

    def done(self) -> bool:
        return self._done

    def cancelling(self) -> int:
        return self._cancelling


class _RecordingTM(AsyncThreadStateManager):
    """Puts the steering gate, the drains and the delivery mark on ONE timeline, so a test can
    assert the ORDER they happen in rather than merely that they all happened. Order is the
    whole contract here: a gate that closes after the mark, or a drain that runs before the
    gate, is a note silently lost."""

    def __init__(self, events: Optional[List[str]] = None):
        super().__init__(db=None)
        self.events: List[str] = events if events is not None else []

    def close_job_steering(self, thread_key, job_id):
        self.events.append("close")
        super().close_job_steering(thread_key, job_id)

    def drain_job_notes(self, thread_key, job_id):
        notes = super().drain_job_notes(thread_key, job_id)
        self.events.append(f"drain:{len(notes)}")
        return notes

    def mark_research_delivery_started(self, thread_key, job_id):
        self.events.append("mark")
        super().mark_research_delivery_started(thread_key, job_id)


class _ReturningStream:
    """A model call that YIELDS a real string and TERMINATES (CLAUDE.md pitfall 6)."""

    def __init__(self, text="the findings", raises=None, slow=False):
        self.text = text
        self.raises = raises
        self.slow = slow
        self.calls: List[Dict[str, Any]] = []

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.slow:
            await asyncio.sleep(5)
        if self.raises is not None:
            raise self.raises
        return {"text": self.text, "tools_used": [], "local_tool_calls": []}


class _StallingStream(_ReturningStream):
    """The job's model call, parked mid-round. This is the window a note has to be able to land
    in: the tool says `ok`, and the round it would have ridden is already in flight."""

    def __init__(self, text="the findings"):
        super().__init__(text=text)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.started.set()
        await self.release.wait()
        return {"text": self.text, "tools_used": [], "local_tool_calls": []}


class _PlanCapturingStream(_ReturningStream):
    """Two-phase: the working round behaves normally, the DELIVERY round (recognised by its
    `deliver` tool) is captured and answered with a canned plan."""

    def __init__(self, text="the findings", plan=None, stall=False):
        super().__init__(text=text)
        self.plan = plan
        self.delivery_kwargs: Optional[Dict[str, Any]] = None
        self.started = asyncio.Event()
        self.release = asyncio.Event() if stall else None

    async def __call__(self, **kwargs):
        registry = kwargs.get("registry")
        if registry is not None and any(s.get("name") == "deliver"
                                        for s in registry.schemas({})):
            self.delivery_kwargs = kwargs
            if self.plan is not None:
                await registry.dispatch(kwargs.get("tool_context"), "deliver", self.plan)
            return {"text": "", "tools_used": [], "local_tool_calls": []}
        self.calls.append(kwargs)
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        return {"text": self.text, "tools_used": [], "local_tool_calls": []}


# --------------------------------------------------------------- helpers

def _ctx(processor, client=None, *, thread_ts="100.0", trigger_ts="100.0", channel_id="C1"):
    return ToolContext(channel_id=channel_id, thread_ts=thread_ts, trigger_ts=trigger_ts,
                       client=client, processor=processor)


def _running(tm, job_id="j1", *, thread_key=THREAD, summary="map the Q3 pricing shifts",
             task=None, **extra):
    """Register a job that the ladder will consider alive, plus any entry state under test."""
    tm.register_research(thread_key, job_id, summary, task=task or _FakeTask())
    entry = tm._active_research[thread_key][job_id]
    entry.update(extra)
    return entry


def _job_task(proc, client, *, job_id="j1", thread_key=THREAD, deliverables=None,
              mode="research", task="map the Q3 pricing shifts"):
    return asyncio.ensure_future(rt._run_background_job(
        processor=proc, client=client, channel_id="C1", thread_root="100.0",
        thread_key=thread_key, job_id=job_id, task=task, snapshot=[],
        system_prompt=None, model="gpt-5.6-sol", mode=mode, deliverables=deliverables))


def _context_text(update_or_post):
    """The context block's mrkdwn from a recorded card post/update tuple."""
    return update_or_post[3][1]["elements"][0]["text"]


def _delivery_messages(stub) -> List[Dict[str, Any]]:
    return list((stub.delivery_kwargs or {}).get("messages") or [])


def _record_failures(monkeypatch) -> List[tuple]:
    """Capture the visible failure post instead of silencing it. Every failure ending is
    REQUIRED to attempt one — a no-op stub would let deleting the post entirely keep this whole
    matrix green, which is exactly the regression the endings exist to prevent."""
    calls: List[tuple] = []

    async def _fail(client, channel_id, thread_root, reason, receipts=None):
        calls.append((channel_id, thread_root, reason, receipts))

    monkeypatch.setattr(rt, "_deliver_failure", _fail)
    return calls


def _late_item(stub) -> Optional[Dict[str, Any]]:
    return next((m for m in _delivery_messages(stub)
                 if "Updates that arrived after the final working round"
                 in str(m.get("content") or "")), None)


# --------------------------------------------------------------- schema + registration

def test_schema_shape_and_registration():
    schema = rt.get_update_background_job_schema()
    assert schema["type"] == "function"
    assert schema["name"] == "update_background_job"
    props = schema["parameters"]["properties"]
    assert props["job_id"]["type"] == "string"
    assert props["note"]["type"] == "string"
    # The note is the whole call; the id is omittable because one job in flight is the norm.
    assert schema["parameters"]["required"] == ["note"]
    # The line that keeps this from becoming a second cancel button.
    assert "work that should CONTINUE" in schema["description"]
    assert "cancel_background_job instead" in schema["description"]
    # ...and the honesty the tool cannot enforce from the executor.
    assert "never that it has been applied" in schema["description"]

    reg = ToolRegistry()
    rt.register_research_tools(reg)
    names = {s["name"] for s in reg.schemas({})}
    assert {"start_background_job", "cancel_background_job",
            "update_background_job"} <= names


# --------------------------------------------------------------- the refusal ladder

@pytest.mark.asyncio
async def test_unavailable_without_a_processor():
    res = await rt.execute_update_background_job(_ctx(None), {"note": "drop section 3"})
    # Exact result: this dict is the model's whole view of what happened.
    assert res == {"ok": False, "error": "unavailable",
                   "message": "background jobs are not available here"}


@pytest.mark.asyncio
async def test_an_empty_note_is_refused_before_anything_else_is_considered():
    """Rung 1, and first for a reason: with no note there is nothing to queue, so answering
    `no_job_running` would send the model off to check the roster over a call that could never
    have worked whatever the roster said."""
    tm = AsyncThreadStateManager(db=None)
    proc = _FakeProcessor(tm=tm)
    for bad in ("", "   ", "\n\t ", None, 7, ["drop section 3"], {"note": "x"}):
        res = await rt.execute_update_background_job(_ctx(proc), {"note": bad})
        assert res == {"ok": False, "error": "empty_note"}, bad
    # Still refused for the note, not for the registry, once a job IS running.
    _running(tm)
    assert await rt.execute_update_background_job(
        _ctx(proc), {"note": "  "}) == {"ok": False, "error": "empty_note"}


@pytest.mark.asyncio
async def test_no_job_running():
    proc = _FakeProcessor(tm=AsyncThreadStateManager(db=None))
    res = await rt.execute_update_background_job(_ctx(proc), {"note": "drop section 3"})
    assert res == {"ok": False, "error": "no_job_running"}


@pytest.mark.asyncio
async def test_an_image_only_registry_is_no_job_running():
    """Jobs only. An image generation is seconds long and has no tool loop to inject a note
    into, so it is not a candidate — and must not be offered as one."""
    tm = AsyncThreadStateManager(db=None)
    tm.register_generation(THREAD, "gen777", "a cat wearing sunglasses")
    proc = _FakeProcessor(tm=tm)
    res = await rt.execute_update_background_job(_ctx(proc), {"note": "make it a dog"})
    assert res == {"ok": False, "error": "no_job_running"}


@pytest.mark.asyncio
async def test_an_omitted_id_resolves_the_job_even_with_an_image_in_flight():
    """The counterpart: because generations are not candidates, they cannot make an omitted id
    ambiguous either. One job plus one image is still exactly one steerable thing."""
    tm = AsyncThreadStateManager(db=None)
    _running(tm, "aaa")
    tm.register_generation(THREAD, "gen777", "a cat wearing sunglasses")
    proc = _FakeProcessor(tm=tm)
    res = await rt.execute_update_background_job(_ctx(proc), {"note": "drop section 3"})
    assert res == {"ok": True, "job_id": "aaa", "queued": 1}


@pytest.mark.asyncio
async def test_a_blank_or_null_id_means_omitted():
    tm = AsyncThreadStateManager(db=None)
    _running(tm, "aaa")
    proc = _FakeProcessor(tm=tm)
    for blank in (None, "", "   "):
        res = await rt.execute_update_background_job(
            _ctx(proc), {"job_id": blank, "note": "drop section 3"})
        assert res["ok"] is True and res["job_id"] == "aaa", blank


@pytest.mark.asyncio
async def test_an_unknown_id_hands_back_the_roster():
    tm = AsyncThreadStateManager(db=None)
    _running(tm, "aaa")
    proc = _FakeProcessor(tm=tm)
    res = await rt.execute_update_background_job(
        _ctx(proc), {"job_id": "zzz", "note": "drop section 3"})
    assert res["ok"] is False and res["error"] == "job_not_found"
    assert res["in_flight"] == [{"job_id": "aaa", "task_summary": "map the Q3 pricing shifts",
                                 "kind": "job"}]


@pytest.mark.asyncio
async def test_a_non_string_id_names_no_job_and_the_roster_lists_jobs_only():
    """A wrong type names nothing — same answer as an id that doesn't exist. And the roster it
    gets back must not contain the image, or the retry fails for a reason this very list
    implied was fine."""
    tm = AsyncThreadStateManager(db=None)
    _running(tm, "aaa")
    tm.register_generation(THREAD, "gen777", "a cat wearing sunglasses")
    proc = _FakeProcessor(tm=tm)
    for bad_id in (7, ["aaa"], {"job_id": "aaa"}, True):
        res = await rt.execute_update_background_job(
            _ctx(proc), {"job_id": bad_id, "note": "drop section 3"})
        assert res["ok"] is False and res["error"] == "job_not_found", bad_id
        assert res["in_flight"] == [{"job_id": "aaa",
                                     "task_summary": "map the Q3 pricing shifts",
                                     "kind": "job"}], bad_id


@pytest.mark.asyncio
async def test_an_omitted_id_with_two_jobs_is_ambiguous():
    tm = AsyncThreadStateManager(db=None)
    _running(tm, "aaa")
    _running(tm, "bbb", summary="build the onboarding deck")
    proc = _FakeProcessor(tm=tm)
    res = await rt.execute_update_background_job(_ctx(proc), {"note": "drop section 3"})
    assert res["ok"] is False and res["error"] == "job_id_required"
    assert [j["job_id"] for j in res["in_flight"]] == ["aaa", "bbb"]
    assert all(j["kind"] == "job" for j in res["in_flight"])


@pytest.mark.asyncio
async def test_a_job_already_cancelling_takes_no_notes():
    tm = AsyncThreadStateManager(db=None)
    _running(tm, cancel_reason="the user asked me to stand down")
    proc = _FakeProcessor(tm=tm)
    res = await rt.execute_update_background_job(_ctx(proc), {"note": "drop section 3"})
    assert res == {"ok": False, "error": "already_cancelling"}


@pytest.mark.asyncio
async def test_job_already_finished_covers_a_job_that_died_before_it_started():
    """No task on the entry: the coroutine was killed on its first tick, or never got one. The
    registry still says a job is running here — the ladder must not believe it."""
    tm = AsyncThreadStateManager(db=None)
    tm.register_research(THREAD, "j1", "map the Q3 pricing shifts")     # task=None
    proc = _FakeProcessor(tm=tm)
    res = await rt.execute_update_background_job(_ctx(proc), {"note": "drop section 3"})
    assert res == {"ok": False, "error": "job_already_finished"}

    # And the ordinary case: the task has finished.
    _running(tm, "done1", task=_FakeTask(done=True))
    res = await rt.execute_update_background_job(
        _ctx(proc), {"job_id": "done1", "note": "drop section 3"})
    assert res == {"ok": False, "error": "job_already_finished"}


@pytest.mark.asyncio
async def test_job_already_finished_covers_the_reason_less_shutdown_cancel():
    """Shutdown cancels every job WITHOUT setting a cancel_reason, so `already_cancelling`
    never fires and the task has not finished yet either — between those two the entry reads
    as perfectly healthy. `cancelling()` is what actually catches it, which is why this one
    uses a real task rather than a stand-in."""
    tm = AsyncThreadStateManager(db=None)
    task = asyncio.ensure_future(asyncio.sleep(30))
    await asyncio.sleep(0)
    _running(tm, task=task)
    task.cancel()
    assert task.done() is False and tm.research_cancel_reason(THREAD, "j1") is None

    proc = _FakeProcessor(tm=tm)
    res = await rt.execute_update_background_job(_ctx(proc), {"note": "drop section 3"})
    assert res == {"ok": False, "error": "job_already_finished"}

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_a_job_that_is_posting_is_past_changing():
    tm = AsyncThreadStateManager(db=None)
    _running(tm, delivery_started=True)
    proc = _FakeProcessor(tm=tm)
    res = await rt.execute_update_background_job(_ctx(proc), {"note": "drop section 3"})
    assert res == {"ok": False, "error": "delivery_in_progress",
                   "hint": "the job is already posting its results; too late to change it"}


@pytest.mark.asyncio
async def test_a_job_past_its_working_rounds_is_job_finishing():
    tm = AsyncThreadStateManager(db=None)
    _running(tm, steering_closed=True)
    proc = _FakeProcessor(tm=tm)
    res = await rt.execute_update_background_job(_ctx(proc), {"note": "drop section 3"})
    assert res["ok"] is False and res["error"] == "job_finishing"
    # The hint promises cancelling still works — which is true HERE, and only here.
    assert res["hint"] == ("the job's working rounds are over; the output can no longer change "
                           "— you can still cancel until it starts posting")


@pytest.mark.asyncio
async def test_delivery_outranks_finishing_so_the_hint_never_lies():
    """Both gates set is the normal end state of every job, and the order of these two rungs is
    the difference between an honest refusal and a false promise: `job_finishing` tells the
    model it can still cancel, which stops being true the moment delivery starts."""
    tm = AsyncThreadStateManager(db=None)
    _running(tm, steering_closed=True, delivery_started=True)
    proc = _FakeProcessor(tm=tm)
    res = await rt.execute_update_background_job(_ctx(proc), {"note": "drop section 3"})
    assert res["error"] == "delivery_in_progress"
    assert "you can still cancel" not in res["hint"]


@pytest.mark.asyncio
async def test_the_ladder_order_holds_when_two_rungs_are_true_at_once():
    """Isolated states cannot pin an ORDER — every one of these would still pass with the rungs
    swapped. Each case below sets BOTH conditions and asserts which one answers, so the pinned
    sequence is actually locked rather than merely implemented.

    The order is the honest one: the earlier answer is the more fundamental fact about the job,
    and each rung is only reached once everything above it has been ruled out."""
    tm = AsyncThreadStateManager(db=None)
    proc = _FakeProcessor(tm=tm)

    async def _update(job_id):
        return await rt.execute_update_background_job(
            _ctx(proc), {"job_id": job_id, "note": "drop section 3"})

    # A job that is stopping outranks a job that has stopped: someone ASKED for this one, and
    # `already_cancelling` is the answer that says so.
    _running(tm, "a", cancel_reason="stood down", task=_FakeTask(done=True))
    assert (await _update("a"))["error"] == "already_cancelling"

    # A dead task outranks a delivery flag it could never have set.
    _running(tm, "b", task=_FakeTask(done=True), delivery_started=True)
    assert (await _update("b"))["error"] == "job_already_finished"

    # Posting outranks winding down — the rung that follows would promise a cancel that is no
    # longer available.
    _running(tm, "c", delivery_started=True, steering_closed=True)
    assert (await _update("c"))["error"] == "delivery_in_progress"

    # A closed window outranks a full queue: `queue_full` invites a retry after a drain, and
    # there will be no drain.
    _running(tm, "d", steering_closed=True, notes_accepted=10)
    assert (await _update("d"))["error"] == "job_finishing"

    # ...and the cap still answers once the window is genuinely open.
    _running(tm, "e", notes_accepted=10)
    assert (await _update("e"))["error"] == "queue_full"

    # An unknown id outranks every state rung, because no state was ever consulted.
    assert (await _update("nope"))["error"] == "job_not_found"


@pytest.mark.asyncio
async def test_the_cap_is_a_lifetime_not_a_backlog():
    """Ten per JOB, counted on the entry and never decremented. Capping the pending backlog
    instead would let a job absorb ten notes per round and hundreds across a run — the runaway
    this number exists to stop. So: fill it, drain it empty, and try again."""
    tm = AsyncThreadStateManager(db=None)
    _running(tm)
    proc = _FakeProcessor(tm=tm)
    for i in range(10):
        res = await rt.execute_update_background_job(_ctx(proc), {"note": f"update {i}"})
        assert res == {"ok": True, "job_id": "j1", "queued": i + 1}

    assert await rt.execute_update_background_job(
        _ctx(proc), {"note": "one more"}) == {"ok": False, "error": "queue_full"}

    # The queue is now empty — the backlog reading would re-open the gate here.
    assert len(tm.drain_job_notes(THREAD, "j1")) == 10
    assert tm.drain_job_notes(THREAD, "j1") == []
    assert await rt.execute_update_background_job(
        _ctx(proc), {"note": "one more"}) == {"ok": False, "error": "queue_full"}


@pytest.mark.asyncio
async def test_a_long_note_is_normalized_then_truncated():
    tm = AsyncThreadStateManager(db=None)
    _running(tm)
    proc = _FakeProcessor(tm=tm)

    # Whitespace first, so the cap counts real characters rather than the model's line breaks.
    await rt.execute_update_background_job(
        _ctx(proc), {"note": "  drop\n\n  section   3  \t"})
    assert tm.drain_job_notes(THREAD, "j1") == ["drop section 3"]

    res = await rt.execute_update_background_job(_ctx(proc), {"note": "x" * 5000})
    assert res["ok"] is True
    stored = tm.drain_job_notes(THREAD, "j1")[0]
    assert len(stored) == 1500
    assert stored == "x" * 1499 + "…"

    # Exactly at the cap is not over it.
    await rt.execute_update_background_job(_ctx(proc), {"note": "y" * 1500})
    kept = tm.drain_job_notes(THREAD, "j1")[0]
    assert kept == "y" * 1500 and "…" not in kept


@pytest.mark.asyncio
async def test_the_success_shape_reports_the_pending_count():
    tm = AsyncThreadStateManager(db=None)
    _running(tm, "aaa")
    proc = _FakeProcessor(tm=tm)
    assert await rt.execute_update_background_job(
        _ctx(proc), {"job_id": "aaa", "note": "drop section 3"}) == {
        "ok": True, "job_id": "aaa", "queued": 1}
    # `queued` is what is WAITING, not what has been accepted over the job's life.
    assert await rt.execute_update_background_job(
        _ctx(proc), {"job_id": "aaa", "note": "and add a risks slide"}) == {
        "ok": True, "job_id": "aaa", "queued": 2}
    tm.drain_job_notes(THREAD, "aaa")
    assert await rt.execute_update_background_job(
        _ctx(proc), {"job_id": "aaa", "note": "third"}) == {
        "ok": True, "job_id": "aaa", "queued": 1}


# --------------------------------------------------------------- cancel × update, same round

@pytest.mark.asyncio
async def test_cancel_then_update_in_one_round_refuses_the_update():
    """A round's tool calls dispatch under gather. Whichever order they land in, the pair has to
    reach a defensible end state — and a job that is stopping does not take instructions."""
    tm = AsyncThreadStateManager(db=None)
    task = asyncio.ensure_future(asyncio.sleep(30))
    await asyncio.sleep(0)
    _running(tm, task=task)
    proc = _FakeProcessor(tm=tm)

    cancelled, updated = await asyncio.gather(
        rt.execute_cancel_background_job(
            _ctx(proc), {"reason": "the user asked me to stand down"}),
        rt.execute_update_background_job(_ctx(proc), {"note": "drop section 3"}),
    )
    assert cancelled["ok"] is True
    assert updated == {"ok": False, "error": "already_cancelling"}
    assert tm.drain_job_notes(THREAD, "j1") == []

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_update_then_cancel_discards_the_queued_note():
    """The other order, and the ruling it encodes: cancel SUPERSEDES an accepted note. The
    update is answered `ok` — it genuinely was queued — and then the whole job goes away, note
    and all. Which is why the guidance says "passed along" and never "guaranteed"."""
    tm = AsyncThreadStateManager(db=None)
    task = asyncio.ensure_future(asyncio.sleep(30))
    await asyncio.sleep(0)
    _running(tm, task=task)
    proc = _FakeProcessor(tm=tm)

    updated, cancelled = await asyncio.gather(
        rt.execute_update_background_job(_ctx(proc), {"note": "drop section 3"}),
        rt.execute_cancel_background_job(
            _ctx(proc), {"reason": "the user asked me to stand down"}),
    )
    assert updated["ok"] is True and updated["queued"] == 1
    assert cancelled["ok"] is True
    # The note is still sitting in an entry nobody will ever drain again.
    assert tm.research_cancel_reason(THREAD, "j1") == "the user asked me to stand down"

    with pytest.raises(asyncio.CancelledError):
        await task


# --------------------------------------------------------------- the injection seam

@pytest.mark.asyncio
async def test_the_callback_wraps_each_note_as_a_developer_user_pair():
    """The split is an authority boundary, not formatting. A follow-up typed into a Slack thread
    is conversation content; letting it ride at developer authority would let anyone in the
    channel issue system instructions to a running job."""
    tm = AsyncThreadStateManager(db=None)
    _running(tm)
    proc = _FakeProcessor(tm=tm)
    card = rt._ResearchCard(processor=proc, client=_FakeClient(), channel_id="C1",
                            thread_root="100.0", task="t", label=None)
    cb = rt._make_steering_callback(proc, tm, thread_key=THREAD, job_id="j1", card=card)
    assert cb is not None
    assert await cb() == []                             # nothing queued: nothing injected

    tm.queue_job_note(THREAD, "j1", "drop section 3")
    tm.queue_job_note(THREAD, "j1", "the Q3 figure is 4.2M")
    items = await cb()
    assert [i["role"] for i in items] == ["developer", "user", "developer", "user"]
    assert items[0]["content"] == rt._STEERING_DEV_INSTRUCTION
    assert "revise the todo list with update_todos" in items[0]["content"]
    # The note itself is verbatim — normalization and truncation already happened at accept.
    assert items[1]["content"] == "drop section 3"
    assert items[3]["content"] == "the Q3 figure is 4.2M"
    # Popped, not copied: a second round must not re-inject the same correction.
    assert await cb() == []
    assert card._steering_notes == 2


@pytest.mark.asyncio
async def test_bookkeeping_failure_never_costs_the_drained_notes():
    """The queue is emptied before the card is touched, so from that instant the ONLY copy of
    these notes is the list being returned. The tool loop deliberately swallows anything raised
    out of this callback and sends the round unsteered — so a card or logger that throws would
    destroy accepted corrections outright. Bookkeeping is not allowed to prevent the return."""
    tm = AsyncThreadStateManager(db=None)
    _running(tm)
    proc = _FakeProcessor(tm=tm)

    class _BrokenCard:
        def bump_steering(self, n):
            raise RuntimeError("the card is wedged")

        def request_steering_flush(self):
            raise RuntimeError("and so is its flush")

    def _explode(*a, **k):
        raise RuntimeError("and the logger too")

    proc.log_info = _explode
    cb = rt._make_steering_callback(proc, tm, thread_key=THREAD, job_id="j1",
                                    card=cast(Any, _BrokenCard()))
    assert cb is not None
    tm.queue_job_note(THREAD, "j1", "drop section 3")

    items = await cb()
    assert [i["content"] for i in items] == [rt._STEERING_DEV_INSTRUCTION, "drop section 3"]


@pytest.mark.asyncio
async def test_the_callbacks_body_never_yields_to_the_event_loop():
    """The atomicity guarantee, asserted rather than assumed. The callback is awaitable because
    the tool loop awaits it — but if anything inside it ever suspends, a cancellation landing in
    that gap takes notes the tool already answered `ok` to, and no round ever sees them. A
    coroutine whose body never yields completes on its first `send`."""
    tm = AsyncThreadStateManager(db=None)
    _running(tm)
    proc = _FakeProcessor(tm=tm)
    card = rt._ResearchCard(processor=proc, client=_FakeClient(), channel_id="C1",
                            thread_root="100.0", task="t", label=None)
    cb = rt._make_steering_callback(proc, tm, thread_key=THREAD, job_id="j1", card=card)
    assert cb is not None
    tm.queue_job_note(THREAD, "j1", "drop section 3")

    coro = cb()
    with pytest.raises(StopIteration) as stopped:
        coro.send(None)     # a suspension here would return a future instead of stopping
    assert stopped.value.value[1] == {"role": "user", "content": "drop section 3"}


def test_the_callback_is_none_without_a_manager_to_drain():
    proc = _FakeProcessor(tm=None)
    card = rt._ResearchCard(processor=proc, client=_FakeClient(), channel_id="C1",
                            thread_root="100.0", task="t", label=None)
    assert rt._make_steering_callback(proc, None, thread_key=THREAD, job_id="j1",
                                      card=card) is None


@pytest.mark.asyncio
async def test_the_research_loop_is_handed_the_drain_callback():
    monkey_stream = _PlanCapturingStream(plan={"reply": "here it is", "post_report": True})
    tm = AsyncThreadStateManager(db=None)
    tm.register_research(THREAD, "j1", "map the Q3 pricing shifts")
    proc = _FakeProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=monkey_stream), tm=tm)
    await _job_task(proc, _FakeClient())
    assert callable(monkey_stream.calls[0]["pre_round_input_callback"])


@pytest.mark.asyncio
async def test_the_build_boundary_drains_before_the_build_input_goes_out(monkeypatch):
    """Seam 2. A note that lands during the research phase's final round has no research round
    left to reach — but the build has not started, so it is exactly the moment "drop the
    competitor section" can still change the deck."""
    monkeypatch.setattr(config, "enable_research_label", False)
    tm = AsyncThreadStateManager(db=None)
    _running(tm)
    proc = _FakeProcessor(tm=tm)
    card = rt._ResearchCard(processor=proc, client=_FakeClient(), channel_id="C1",
                            thread_root="100.0", task="t", label=None)
    cb = rt._make_steering_callback(proc, tm, thread_key=THREAD, job_id="j1", card=card)
    tm.queue_job_note(THREAD, "j1", "drop the competitor section")

    captured: Dict[str, Any] = {}

    async def _fake_consume(_processor, **kwargs):
        captured.update(kwargs)
        return {"text": "built it", "tools_used": []}

    monkeypatch.setattr(rt, "_consume_research_stream", _fake_consume)
    manager = SimpleNamespace(create_explicit=lambda key: _async_value("cntr_1"))
    proc.container_manager = manager
    monkeypatch.setattr(rt, "_thread_file_catalog", lambda *a, **k: _async_value([]))

    await rt._run_build_phase(
        processor=proc, client=_FakeClient(), channel_id="C1", thread_root="100.0",
        thread_key=THREAD, job_id="j1", task="build the deck", findings="the findings",
        deliverables=[{"type": "pdf", "description": "d", "filename": "report.pdf"}],
        snapshot=[], thread_config={}, system_prompt=None, model="gpt-5.6-sol", card=card,
        steering_callback=cb)

    messages = captured["messages"]
    # AFTER the build instruction, which is what makes it a correction TO the build rather
    # than part of the brief the instruction then overrides.
    assert messages[-2] == {"role": "developer", "content": rt._STEERING_DEV_INSTRUCTION}
    assert messages[-1] == {"role": "user", "content": "drop the competitor section"}
    # ...and the same callback still rides into the loop for every round after this one.
    assert captured["pre_round_input_callback"] is cb
    assert card._steering_notes == 1


@pytest.mark.asyncio
async def test_a_correction_applied_during_research_is_repeated_to_the_build_model(monkeypatch):
    """The two-phase hole. The drain is DESTRUCTIVE, and the build phase is a different model
    call assembled fresh from the original snapshot, the original task and the findings — it
    never saw the round where the correction landed. The findings and a four-item todo list are
    a lossy channel for a scope change, so without an explicit carry-over the build reads the
    ORIGINAL request and puts back the very section that was dropped."""
    monkeypatch.setattr(config, "enable_research_label", False)
    tm = AsyncThreadStateManager(db=None)
    _running(tm)
    proc = _FakeProcessor(tm=tm)
    card = rt._ResearchCard(processor=proc, client=_FakeClient(), channel_id="C1",
                            thread_root="100.0", task="t", label=None)
    applied: List[str] = []
    cb = rt._make_steering_callback(proc, tm, thread_key=THREAD, job_id="j1", card=card,
                                    applied_notes=applied)
    assert cb is not None

    # A note arrives and is drained by a RESEARCH round — gone from the queue for good.
    tm.queue_job_note(THREAD, "j1", "drop the competitor section")
    await cb()
    assert applied == ["drop the competitor section"]
    assert tm.drain_job_notes(THREAD, "j1") == []

    captured: Dict[str, Any] = {}

    async def _fake_consume(_processor, **kwargs):
        captured.update(kwargs)
        return {"text": "built it", "tools_used": []}

    monkeypatch.setattr(rt, "_consume_research_stream", _fake_consume)
    proc.container_manager = SimpleNamespace(
        create_explicit=lambda key: _async_value("cntr_1"))
    monkeypatch.setattr(rt, "_thread_file_catalog", lambda *a, **k: _async_value([]))

    # A SECOND note arrives after the research rounds are over — the boundary drain's business.
    tm.queue_job_note(THREAD, "j1", "and add a risks slide")

    await rt._run_build_phase(
        processor=proc, client=_FakeClient(), channel_id="C1", thread_root="100.0",
        thread_key=THREAD, job_id="j1", task="build the deck", findings="the findings",
        deliverables=[{"type": "pdf", "description": "d", "filename": "report.pdf"}],
        snapshot=[], thread_config={}, system_prompt=None, model="gpt-5.6-sol", card=card,
        steering_callback=cb, applied_notes=applied)

    messages = captured["messages"]
    carried_all = [m for m in messages
                   if "already applied during the research phase" in str(m.get("content") or "")]
    assert len(carried_all) == 1
    carried = carried_all[0]
    assert carried["role"] == "user"
    assert "1. drop the competitor section" in carried["content"]
    assert "keep them applied" in carried["content"]
    # The already-applied item names ONLY the research-phase note; the new one arrives as a
    # fresh correction through the boundary drain, not as something already honoured.
    assert "and add a risks slide" not in carried["content"]
    assert messages[-1] == {"role": "user", "content": "and add a risks slide"}
    # ...and it sits after the build instruction that carries the findings.
    instruction_idx = [i for i, m in enumerate(messages)
                       if "RESEARCH FINDINGS" in str(m.get("content") or "")]
    assert instruction_idx and instruction_idx[0] < messages.index(carried)


@pytest.mark.asyncio
async def test_a_build_with_no_research_corrections_gains_no_carry_over_item(monkeypatch):
    monkeypatch.setattr(config, "enable_research_label", False)
    tm = AsyncThreadStateManager(db=None)
    _running(tm)
    proc = _FakeProcessor(tm=tm)
    card = rt._ResearchCard(processor=proc, client=_FakeClient(), channel_id="C1",
                            thread_root="100.0", task="t", label=None)
    captured: Dict[str, Any] = {}

    async def _fake_consume(_processor, **kwargs):
        captured.update(kwargs)
        return {"text": "built it", "tools_used": []}

    monkeypatch.setattr(rt, "_consume_research_stream", _fake_consume)
    proc.container_manager = SimpleNamespace(
        create_explicit=lambda key: _async_value("cntr_1"))
    monkeypatch.setattr(rt, "_thread_file_catalog", lambda *a, **k: _async_value([]))

    await rt._run_build_phase(
        processor=proc, client=_FakeClient(), channel_id="C1", thread_root="100.0",
        thread_key=THREAD, job_id="j1", task="build the deck", findings="the findings",
        deliverables=[{"type": "pdf", "description": "d", "filename": "report.pdf"}],
        snapshot=[], thread_config={}, system_prompt=None, model="gpt-5.6-sol", card=card,
        steering_callback=None, applied_notes=[])

    assert not [m for m in captured["messages"]
                if "already applied" in str(m.get("content") or "")]


def _async_value(value):
    async def _coro():
        return value
    return _coro()


# --------------------------------------------------------------- the closing gate

@pytest.mark.asyncio
async def test_the_gate_closes_before_delivery_is_marked_on_the_research_path(monkeypatch):
    monkeypatch.setattr(config, "enable_research_label", False)
    tm = _RecordingTM()
    tm.register_research(THREAD, "j1", "map the Q3 pricing shifts")
    proc = _FakeProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=_ReturningStream("the findings")), tm=tm)

    async def _plan(*a, **k):
        tm.events.append("plan")
        return {"reply": "here it is", "publish": [], "post_report": True}

    async def _transact(*a, **k):
        tm.events.append("transact")
        return True

    monkeypatch.setattr(rt, "_plan_delivery", _plan)
    monkeypatch.setattr(rt, "_transact_delivery", _transact)
    await _job_task(proc, _FakeClient())

    # Steering shuts the moment the report exists; delivery is marked later, and cancelling
    # stays honest across that whole gap. Two gates, two moments.
    assert tm.events.index("close") < tm.events.index("plan") < tm.events.index("mark")
    # ...and the sweep runs immediately after the close, with nothing awaited in between.
    assert tm.events[tm.events.index("close") + 1].startswith("drain:")


@pytest.mark.asyncio
async def test_the_gate_stays_open_across_the_research_to_build_handover(monkeypatch):
    """The research-only close would be premature here: with a deck still to build, a
    correction arriving now can genuinely change the deliverable."""
    monkeypatch.setattr(config, "enable_research_label", False)
    tm = _RecordingTM()
    tm.register_research(THREAD, "j1", "build the onboarding deck")
    proc = _FakeProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=_ReturningStream("the findings")), tm=tm)

    async def _build(**kw):
        tm.events.append("build")
        return {"ledger_key": "k", "container_ids": ["c"], "notes": "",
                "suppress_digests": set(), "expect_filenames": ["deck.pptx"]}

    async def _stage(processor, *, job_id, build):
        tm.events.append("stage")
        return []

    async def _release(processor, *, ledger_key):
        return None

    async def _plan(*a, **k):
        tm.events.append("plan")
        return {"reply": "here it is", "publish": [], "post_report": True}

    async def _transact(*a, **k):
        return True

    monkeypatch.setattr(rt, "_run_build_phase", _build)
    monkeypatch.setattr(rt, "_stage_build", _stage)
    monkeypatch.setattr(rt, "_release_build_container", _release)
    monkeypatch.setattr(rt, "_plan_delivery", _plan)
    monkeypatch.setattr(rt, "_transact_delivery", _transact)
    await _job_task(proc, _FakeClient(), mode="research_and_build",
                    deliverables=[{"type": "powerpoint", "description": "d",
                                   "filename": "deck.pptx"}])

    # The close lands after the build phase and before its output is even staged.
    assert tm.events.index("build") < tm.events.index("close") < tm.events.index("stage")
    assert tm.events.index("close") < tm.events.index("mark")


@pytest.mark.asyncio
async def test_a_build_that_dies_before_any_stream_still_closes_the_gate(monkeypatch):
    """The helper returns None when it cannot get an addressable container — before a stream
    ever exists. A gate that closed "after the build stream" would stay open forever over a job
    with no rounds left, and every note accepted after this point would be unreachable."""
    monkeypatch.setattr(config, "enable_research_label", False)
    tm = _RecordingTM()
    tm.register_research(THREAD, "j1", "build the onboarding deck")
    stream = _PlanCapturingStream(text="the findings",
                                  plan={"reply": "here it is", "post_report": True})
    proc = _FakeProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=stream), tm=tm)
    tm._active_research[THREAD]["j1"]["task"] = _FakeTask()

    async def _build(**kw):
        # The note lands while the build phase is deciding it cannot run at all.
        assert tm.queue_job_note(THREAD, "j1", "make it two pages, not ten")["ok"] is True
        return None

    async def _release(processor, *, ledger_key):
        return None

    monkeypatch.setattr(rt, "_run_build_phase", _build)
    monkeypatch.setattr(rt, "_release_build_container", _release)
    await _job_task(proc, _FakeClient(), mode="research_and_build",
                    deliverables=[{"type": "powerpoint", "description": "d",
                                   "filename": "deck.pptx"}])

    assert "close" in tm.events
    # Swept, not stranded: it reaches delivery as an explicit not-applied update.
    late = _late_item(stream)
    assert late is not None and "1. make it two pages, not ten" in late["content"]


# --------------------------------------------------------------- the closing sweep

@pytest.mark.asyncio
async def test_a_note_accepted_mid_final_round_reaches_delivery_as_not_applied(monkeypatch):
    """The gap the sweep exists for. The tool answered `ok` while the last round's API call was
    already in flight, so no round will ever read it — and the delivering model has to be told
    both halves: the update is real, and it is not in the work."""
    monkeypatch.setattr(config, "enable_research_label", False)
    tm = AsyncThreadStateManager(db=None)
    stream = _PlanCapturingStream(text="the findings", stall=True,
                                  plan={"reply": "here it is", "post_report": True})
    proc = _FakeProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=stream), tm=tm)
    tm.register_research(THREAD, "j1", "map the Q3 pricing shifts")
    job = _job_task(proc, _FakeClient())
    tm.attach_research_task(THREAD, "j1", job)

    await asyncio.wait_for(stream.started.wait(), timeout=2)
    accepted = await rt.execute_update_background_job(
        _ctx(proc), {"note": "drop the competitor section"})
    assert accepted["ok"] is True
    stream.release.set()
    await asyncio.wait_for(job, timeout=5)

    messages = _delivery_messages(stream)
    late = _late_item(stream)
    assert late is not None
    assert late["role"] == "user"                       # data, never a system instruction
    assert "NOT reflected in the findings or files" in late["content"]
    assert "1. drop the competitor section" in late["content"]
    # Position is the contract: the developer instruction still gets the last word...
    assert messages[-1]["role"] == "developer"
    assert messages.index(late) == len(messages) - 2
    # ...and it is the developer item, not the user wrapper, that carries the obligation.
    assert ("Late updates listed above are NOT applied: acknowledge them in the reply and "
            "offer a follow-up revision; never claim they were applied."
            in messages[-1]["content"])


@pytest.mark.asyncio
async def test_the_late_updates_item_sits_after_the_build_notes(monkeypatch):
    monkeypatch.setattr(config, "enable_research_label", False)
    tm = AsyncThreadStateManager(db=None)
    stream = _PlanCapturingStream(text="the findings",
                                  plan={"reply": "here it is", "post_report": True})
    proc = _FakeProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=stream), tm=tm)
    tm.register_research(THREAD, "j1", "build the onboarding deck", task=_FakeTask())

    async def _build(**kw):
        assert tm.queue_job_note(THREAD, "j1", "drop the competitor section")["ok"] is True
        return {"ledger_key": "k", "container_ids": ["c"], "notes": "Wrote the deck.",
                "suppress_digests": set(), "expect_filenames": ["deck.pptx"]}

    async def _stage(processor, *, job_id, build):
        return []

    async def _release(processor, *, ledger_key):
        return None

    monkeypatch.setattr(rt, "_run_build_phase", _build)
    monkeypatch.setattr(rt, "_stage_build", _stage)
    monkeypatch.setattr(rt, "_release_build_container", _release)
    await _job_task(proc, _FakeClient(), mode="research_and_build",
                    deliverables=[{"type": "powerpoint", "description": "d",
                                   "filename": "deck.pptx"}])

    messages = _delivery_messages(stream)
    notes_idx = next(i for i, m in enumerate(messages)
                     if "BUILD NOTES" in str(m.get("content") or ""))
    late_idx = messages.index(_late_item(stream))
    assert notes_idx < late_idx < len(messages) - 1
    assert messages[-1]["role"] == "developer"


@pytest.mark.asyncio
async def test_a_steered_builds_delivery_input_carries_what_was_applied(monkeypatch):
    """Found live, and the most expensive failure of the round: the build obeyed "drop light,
    add repotting" and the delivery planner then REFUSED to ship it — reading the original
    snapshot, where light was still the brief, it saw a build that had ignored its instructions
    and declined to post "a knowingly wrong draft". Steering that produces correct work and
    then throws it away is worse than no steering. The planner has to be told the task moved."""
    monkeypatch.setattr(config, "enable_research_label", False)
    tm = AsyncThreadStateManager(db=None)
    stream = _PlanCapturingStream(text="the findings",
                                  plan={"reply": "here it is", "publish": [],
                                        "post_report": True})
    proc = _FakeProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=stream), tm=tm)
    tm.register_research(THREAD, "j1", "build the plant care deck", task=_FakeTask())

    async def _build(*, steering_callback=None, applied_notes=None, **kw):
        # A correction lands mid-build and IS applied — the drain is what records it.
        assert tm.queue_job_note(THREAD, "j1", "drop the light section, add repotting")[
            "ok"] is True
        if steering_callback is not None:
            await steering_callback()
        return {"ledger_key": "k", "container_ids": ["c"], "notes": "Wrote the deck.",
                "suppress_digests": set(), "expect_filenames": ["care.pptx"]}

    async def _stage(processor, *, job_id, build):
        return []

    async def _release(processor, *, ledger_key):
        return None

    monkeypatch.setattr(rt, "_run_build_phase", _build)
    monkeypatch.setattr(rt, "_stage_build", _stage)
    monkeypatch.setattr(rt, "_release_build_container", _release)
    await _job_task(proc, _FakeClient(), mode="research_and_build",
                    deliverables=[{"type": "powerpoint", "description": "d",
                                   "filename": "care.pptx"}])

    messages = _delivery_messages(stream)
    found = [m for m in messages if "applied DURING the run" in str(m.get("content") or "")]
    assert len(found) == 1
    applied = found[0]
    assert applied["role"] == "user"
    assert "1. drop the light section, add repotting" in applied["content"]
    # The sentence that stops the planner reading obedience as deviation.
    assert "judge the work against the updated task, not the original" in applied["content"]
    # Positioned with the late-updates item: after the build notes, before the last word.
    notes_idx = [i for i, m in enumerate(messages)
                 if "BUILD NOTES" in str(m.get("content") or "")]
    assert notes_idx and notes_idx[0] < messages.index(applied) < len(messages) - 1
    assert messages[-1]["role"] == "developer"
    # It was applied, so it must NOT also be reported as an unapplied late update.
    assert _late_item(stream) is None


@pytest.mark.asyncio
async def test_no_late_notes_leaves_the_delivery_input_exactly_as_it_was(monkeypatch):
    """The pinned invariant works both ways: an ordinary job must not grow a stray user item or
    an instruction about updates that never arrived."""
    monkeypatch.setattr(config, "enable_research_label", False)
    tm = AsyncThreadStateManager(db=None)
    stream = _PlanCapturingStream(text="the findings",
                                  plan={"reply": "here it is", "post_report": True})
    proc = _FakeProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=stream), tm=tm)
    tm.register_research(THREAD, "j1", "map the Q3 pricing shifts")
    await _job_task(proc, _FakeClient())

    messages = _delivery_messages(stream)
    assert _late_item(stream) is None
    assert messages[-1]["role"] == "developer"
    assert "Late updates listed above" not in messages[-1]["content"]


@pytest.mark.asyncio
async def test_a_failed_planning_call_still_owns_the_late_notes():
    """Planning returned None — two failures, or no `deliver` call — so the fallback posts
    everything with no reply of its own. Silence over a report would imply the updates were
    honoured, so the application says it itself."""
    proc = _FakeProcessor(tm=AsyncThreadStateManager(db=None))
    client = _FakeClient()
    card = rt._ResearchCard(processor=proc, client=client, channel_id="C1",
                            thread_root="100.0", task="t", label=None)
    card.ts = "CARD.1"

    delivered = await rt._transact_delivery(
        proc, client, channel_id="C1", thread_root="100.0", thread_key=THREAD, job_id="j1",
        plan=None, report="the findings", staged=[], label_source="pricing",
        deliverables=[], card=card, ledger_key=THREAD, elapsed=12.0, effort="high",
        tools_used=[], late_notes=["drop the competitor section"])

    assert delivered is True
    texts = [s[2] for s in client.sent]
    assert texts[0] == ("Note: updates sent while this was finishing are not reflected — say "
                        "the word and I'll do a follow-up revision.")
    # The report still goes out underneath it; the line is an addition, not a replacement.
    assert any("the findings" in t for t in texts)


@pytest.mark.asyncio
async def test_the_fallback_reply_stays_empty_without_late_notes():
    proc = _FakeProcessor(tm=AsyncThreadStateManager(db=None))
    client = _FakeClient()
    card = rt._ResearchCard(processor=proc, client=client, channel_id="C1",
                            thread_root="100.0", task="t", label=None)
    card.ts = "CARD.1"

    await rt._transact_delivery(
        proc, client, channel_id="C1", thread_root="100.0", thread_key=THREAD, job_id="j1",
        plan=None, report="the findings", staged=[], label_source="pricing",
        deliverables=[], card=card, ledger_key=THREAD, elapsed=12.0, effort="high",
        tools_used=[], late_notes=[])

    assert all("follow-up revision" not in s[2] for s in client.sent)


# --------------------------------------------------------------- failure endings

@pytest.mark.asyncio
async def test_an_empty_research_ending_logs_the_discarded_notes(monkeypatch):
    monkeypatch.setattr(config, "enable_research_label", False)
    tm = AsyncThreadStateManager(db=None)
    stream = _StallingStream(text="")
    proc = _LoggingProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=stream), tm=tm)
    tm.register_research(THREAD, "j1", "map the Q3 pricing shifts")
    failures = _record_failures(monkeypatch)
    job = _job_task(proc, _FakeClient())
    tm.attach_research_task(THREAD, "j1", job)

    await asyncio.wait_for(stream.started.wait(), timeout=2)
    assert (await rt.execute_update_background_job(
        _ctx(proc), {"note": "drop the competitor section"}))["ok"] is True
    stream.release.set()
    await asyncio.wait_for(job, timeout=5)

    discards = [w for w in proc.warnings if "discarding 1 mid-run update" in w]
    assert len(discards) == 1
    assert "drop the competitor section" in discards[0]
    # Discarding is only half the ruling — the ending still has to be VISIBLE in the thread.
    assert [r for _c, _t, r, _rc in failures] == ["the research came back empty"]


@pytest.mark.asyncio
async def test_the_empty_ending_closes_and_sweeps_exactly_once(monkeypatch):
    """A single close+sweep point, pinned by call count. Both manager methods are idempotent, so
    a second pass is not data loss today — but it is a second place for the ordering to drift,
    and the whole gate rests on the close being awaitless and unrepeated."""
    monkeypatch.setattr(config, "enable_research_label", False)
    tm = _RecordingTM()
    proc = _LoggingProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=_ReturningStream(text="")), tm=tm)
    tm.register_research(THREAD, "j1", "map the Q3 pricing shifts", task=_FakeTask())
    _record_failures(monkeypatch)
    await _job_task(proc, _FakeClient())

    # Exactly one close, and exactly one sweep alongside it. (The pre-round drain the callback
    # performs is a separate, expected drain — hence counting closes, and drains AFTER it.)
    assert tm.events.count("close") == 1
    assert tm.events[tm.events.index("close") + 1].startswith("drain:")
    assert tm.events.count("close") == len(
        [e for e in tm.events[tm.events.index("close"):] if e.startswith("drain:")])


@pytest.mark.asyncio
async def test_a_timeout_ending_closes_the_gate_and_logs_the_discard(monkeypatch):
    monkeypatch.setattr(config, "enable_research_label", False)
    monkeypatch.setattr(config, "deep_research_timeout", 0.05)
    tm = _RecordingTM()
    proc = _LoggingProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=_ReturningStream(slow=True)), tm=tm)
    tm.register_research(THREAD, "j1", "map the Q3 pricing shifts", task=_FakeTask())
    assert tm.queue_job_note(THREAD, "j1", "drop the competitor section")["ok"] is True
    failures = _record_failures(monkeypatch)

    await _job_task(proc, _FakeClient())

    # The gate shuts FIRST, before delivery is marked — a note accepted after a timeout has
    # nowhere at all to go, and `job_finishing` is the honest refusal.
    assert tm.events.index("close") < tm.events.index("mark")
    assert any("discarding 1 mid-run update" in w for w in proc.warnings)
    assert [r for _c, _t, r, _rc in failures] == [
        "it ran past the time limit before finishing"]


@pytest.mark.asyncio
async def test_an_exception_ending_closes_the_gate_and_logs_the_discard(monkeypatch):
    monkeypatch.setattr(config, "enable_research_label", False)
    tm = _RecordingTM()
    proc = _LoggingProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=_ReturningStream(
            raises=RuntimeError("the model call fell over"))), tm=tm)
    tm.register_research(THREAD, "j1", "map the Q3 pricing shifts", task=_FakeTask())
    assert tm.queue_job_note(THREAD, "j1", "drop the competitor section")["ok"] is True
    failures = _record_failures(monkeypatch)

    await _job_task(proc, _FakeClient())

    assert tm.events.index("close") < tm.events.index("mark")
    assert any("discarding 1 mid-run update" in w for w in proc.warnings)
    assert [r for _c, _t, r, _rc in failures] == ["the model call fell over"]


@pytest.mark.asyncio
async def test_notes_swept_before_a_failing_delivery_are_still_owned(monkeypatch):
    """The loss path the sweep alone does not close. Once the queue has been emptied into
    `late_notes`, an exception thrown by any later await leaves a handler whose FRESH sweep
    finds nothing — and the notes the tool answered `ok` to would go unmentioned by anyone.
    Realistically: the first Slack send in the delivery raising before the acknowledgement
    lands."""
    monkeypatch.setattr(config, "enable_research_label", False)
    tm = _RecordingTM()
    stream = _StallingStream(text="the findings")
    proc = _LoggingProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=stream), tm=tm)
    tm.register_research(THREAD, "j1", "map the Q3 pricing shifts")
    job = _job_task(proc, _FakeClient())
    tm.attach_research_task(THREAD, "j1", job)

    async def _plan(*a, **k):
        return {"reply": "here it is", "publish": [], "post_report": True}

    async def _boom(*a, **k):
        raise RuntimeError("Slack refused the post")

    monkeypatch.setattr(rt, "_plan_delivery", _plan)
    monkeypatch.setattr(rt, "_transact_delivery", _boom)
    failures = _record_failures(monkeypatch)

    await asyncio.wait_for(stream.started.wait(), timeout=2)
    assert (await rt.execute_update_background_job(
        _ctx(proc), {"note": "drop the competitor section"}))["ok"] is True
    stream.release.set()
    await asyncio.wait_for(job, timeout=5)

    # The queue was already empty by the time the handler ran, so only the hoisted list can
    # account for this note.
    assert tm.drain_job_notes(THREAD, "j1") == []
    discards = [w for w in proc.warnings if "discarding 1 mid-run update" in w]
    assert len(discards) == 1 and "drop the competitor section" in discards[0]
    assert [r for _c, _t, r, _rc in failures] == ["Slack refused the post"]


@pytest.mark.asyncio
async def test_late_notes_carried_by_a_successful_delivery_are_not_logged_as_lost(monkeypatch):
    """The other side of the same hoist: once the reply acknowledging them has posted, they are
    surfaced. A handler firing later must not report them as discarded."""
    monkeypatch.setattr(config, "enable_research_label", False)
    tm = AsyncThreadStateManager(db=None)
    stream = _StallingStream(text="the findings")
    proc = _LoggingProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=stream), tm=tm)
    tm.register_research(THREAD, "j1", "map the Q3 pricing shifts")
    job = _job_task(proc, _FakeClient())
    tm.attach_research_task(THREAD, "j1", job)

    async def _plan(*a, **k):
        return {"reply": "here it is", "publish": [], "post_report": True}

    async def _transact(*a, ack_out=None, **k):
        # The real function reports the reply landing; a stub that didn't would be asserting
        # against a contract nothing implements.
        if ack_out is not None:
            ack_out["reply_posted"] = True
        return True

    monkeypatch.setattr(rt, "_plan_delivery", _plan)
    monkeypatch.setattr(rt, "_transact_delivery", _transact)

    await asyncio.wait_for(stream.started.wait(), timeout=2)
    assert (await rt.execute_update_background_job(
        _ctx(proc), {"note": "drop the competitor section"}))["ok"] is True
    stream.release.set()
    await asyncio.wait_for(job, timeout=5)

    assert not [w for w in proc.warnings if "discarding" in w]


@pytest.mark.asyncio
async def test_a_partial_delivery_that_lost_the_reply_discards_its_late_notes(monkeypatch):
    """The narrow gap between "the delivery succeeded" and "they were told". A failed reply is
    deliberately FORGIVEN once the report posts — the findings are what must not be lost — so
    the delivery returns True and looks like a clean ending. But the reply is the only post
    carrying the not-applied acknowledgement, so the accepted update was neither applied, nor
    surfaced, nor logged. Codex reproduced exactly this: both reply attempts failed, the report
    landed, and the note vanished."""
    monkeypatch.setattr(config, "enable_research_label", False)

    class _ReplyRefusingClient(_FakeClient):
        """Slack takes the report (it carries the provenance trailer) and refuses the reply.
        Everything else about the ending looks perfectly healthy."""

        async def send_message(self, channel_id, thread_id, text, **kw):
            if "deep research ·" not in text:
                self.sent.append((channel_id, thread_id, text, kw.get("username")))
                return None
            return await super().send_message(channel_id, thread_id, text, **kw)

    proc = _LoggingProcessor(tm=AsyncThreadStateManager(db=None))
    client = _ReplyRefusingClient()
    card = rt._ResearchCard(processor=proc, client=client, channel_id="C1",
                            thread_root="100.0", task="t", label=None)
    card.ts = "CARD.1"
    ack: Dict[str, bool] = {}

    delivered = await rt._transact_delivery(
        proc, client, channel_id="C1", thread_root="100.0", thread_key=THREAD, job_id="j1",
        plan={"reply": "here are the findings", "publish": [], "post_report": True},
        report="the findings", staged=[], label_source="pricing", deliverables=[], card=card,
        ledger_key=THREAD, elapsed=12.0, effort="high", tools_used=[],
        late_notes=["drop the competitor section"], ack_out=ack)

    # The findings DID land, so the delivery is a success by its own (correct) standard...
    assert delivered is True
    assert any("the findings" in s[2] for s in client.sent)
    # ...and yet the acknowledgement did not, which is what the caller has to be told.
    assert ack["reply_posted"] is False


@pytest.mark.asyncio
async def test_the_job_discards_late_notes_whose_acknowledgement_never_posted(monkeypatch):
    """The same hole seen from the job: a delivery that returns True must not be read as
    "surfaced" when the reply carrying the acknowledgement failed."""
    monkeypatch.setattr(config, "enable_research_label", False)
    tm = AsyncThreadStateManager(db=None)
    stream = _StallingStream(text="the findings")
    proc = _LoggingProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=stream), tm=tm)
    tm.register_research(THREAD, "j1", "map the Q3 pricing shifts")
    job = _job_task(proc, _FakeClient())
    tm.attach_research_task(THREAD, "j1", job)

    async def _plan(*a, **k):
        return {"reply": "here it is", "publish": [], "post_report": True}

    async def _transact(*a, ack_out=None, **k):
        # The report saved the ending; the reply did not land.
        if ack_out is not None:
            ack_out["reply_posted"] = False
        return True

    monkeypatch.setattr(rt, "_plan_delivery", _plan)
    monkeypatch.setattr(rt, "_transact_delivery", _transact)

    await asyncio.wait_for(stream.started.wait(), timeout=2)
    assert (await rt.execute_update_background_job(
        _ctx(proc), {"note": "drop the competitor section"}))["ok"] is True
    stream.release.set()
    await asyncio.wait_for(job, timeout=5)

    discards = [w for w in proc.warnings if "discarding 1 mid-run update" in w]
    assert len(discards) == 1 and "drop the competitor section" in discards[0]


@pytest.mark.asyncio
async def test_a_delivery_that_reached_nobody_discards_its_late_notes(monkeypatch):
    """`_transact_delivery` returning False means it posted a failure note INSTEAD of the reply,
    so the acknowledgement never reached anyone either. No exception is raised, so only this
    branch can own them."""
    monkeypatch.setattr(config, "enable_research_label", False)
    tm = AsyncThreadStateManager(db=None)
    stream = _StallingStream(text="the findings")
    proc = _LoggingProcessor(openai_client=SimpleNamespace(
        create_streaming_response_with_tool_loop=stream), tm=tm)
    tm.register_research(THREAD, "j1", "map the Q3 pricing shifts")
    job = _job_task(proc, _FakeClient())
    tm.attach_research_task(THREAD, "j1", job)

    async def _plan(*a, **k):
        return {"reply": "here it is", "publish": [], "post_report": True}

    async def _transact(*a, **k):
        return False

    monkeypatch.setattr(rt, "_plan_delivery", _plan)
    monkeypatch.setattr(rt, "_transact_delivery", _transact)

    await asyncio.wait_for(stream.started.wait(), timeout=2)
    assert (await rt.execute_update_background_job(
        _ctx(proc), {"note": "drop the competitor section"}))["ok"] is True
    stream.release.set()
    await asyncio.wait_for(job, timeout=5)

    discards = [w for w in proc.warnings if "discarding 1 mid-run update" in w]
    assert len(discards) == 1 and "drop the competitor section" in discards[0]


# --------------------------------------------------------------- the card

@pytest.mark.asyncio
async def test_the_card_counts_updates_as_passed_along_never_as_applied():
    client = _FakeClient()
    proc = _FakeProcessor()
    card = rt._ResearchCard(processor=proc, client=client, channel_id="C1",
                            thread_root="100.0", task="t", label=None)
    card.ts = "CARD.1"
    await card.note_steering(1)
    assert "1 update passed along" in _context_text(client.card_updates[-1])
    # The second write is coalesced by the card's throttle (that is the point of the throttle),
    # so read the render itself rather than waiting a second for Slack.
    await card.note_steering(2)
    line = card._context_line()
    assert "3 updates passed along" in line
    # The wording is the ruling: the job was HANDED the update. Whether it acted on it is the
    # job's own output to show, and a card cannot know.
    assert "folded in" not in line


@pytest.mark.asyncio
async def test_a_card_finalized_before_the_flush_still_renders_the_bumped_count():
    """The drain must not await, so the count lands synchronously and only the Slack write is
    scheduled. If the job ends in that window the scheduled write never happens — the terminal
    render has to carry the number anyway, which it does because finalize renders CURRENT
    state rather than a queued snapshot."""
    client = _FakeClient()
    proc = _FakeProcessor()
    card = rt._ResearchCard(processor=proc, client=client, channel_id="C1",
                            thread_root="100.0", task="t", label=None)
    card.ts = "CARD.1"
    card._last_update = card._clock()       # inside the throttle window: the flush is deferred

    card.bump_steering(2)
    card.request_steering_flush()
    assert client.card_updates == []        # nothing written yet — the race this test is about
    await card.finalize_success("Reported findings below.")

    assert "2 updates passed along" in _context_text(client.card_updates[-1])
    # The deferred writes are no-ops against a closed card rather than a resurrection of it.
    await asyncio.gather(*list(card._steering_flush_tasks), return_exceptions=True)
    assert "✅ Reported findings below." in client.card_updates[-1][3][0]["text"]["text"]


@pytest.mark.asyncio
async def test_an_update_arriving_during_a_slack_write_still_reaches_the_live_card():
    """The stale-card race. `_flush` clears the dirty flag and renders its blocks BEFORE
    awaiting Slack, so a bump landing during that await is not in the message being sent. If the
    second request is dropped merely because a flush is still in the air, the running card sits
    a note behind for as long as the job keeps going — the terminal render eventually corrects
    it, which is no comfort to someone watching a ten-minute job."""

    class _SlowCardClient(_FakeClient):
        def __init__(self):
            super().__init__()
            self.writing = asyncio.Event()
            self.release = asyncio.Event()
            self.first = True

        async def update_status_card(self, channel_id, ts, text, blocks, receipts=None):
            if self.first:
                self.first = False
                self.writing.set()
                await self.release.wait()
            return await super().update_status_card(channel_id, ts, text, blocks,
                                                    receipts=receipts)

    client = _SlowCardClient()
    card = rt._ResearchCard(processor=_FakeProcessor(), client=client, channel_id="C1",
                            thread_root="100.0", task="t", label=None)
    card.ts = "CARD.1"

    # BOTH updates go through the drain's own path — that is the whole point. The first
    # scheduled flush is what used to make the second one get skipped.
    card.bump_steering(1)
    card.request_steering_flush()
    await asyncio.wait_for(client.writing.wait(), timeout=2)

    # The write for note 1 is mid-flight and has already taken its snapshot. Note 2 lands now,
    # while that first flush task is very much still in the air.
    assert any(not t.done() for t in card._steering_flush_tasks)
    card.bump_steering(1)
    card.request_steering_flush()
    client.release.set()

    for _ in range(3):
        await asyncio.gather(*list(card._steering_flush_tasks), return_exceptions=True)
        # The second request lands inside the throttle window, so it defers to a trailing flush.
        if card._flush_task is not None:
            await asyncio.gather(card._flush_task, return_exceptions=True)

    assert "2 updates passed along" in _context_text(client.card_updates[-1])


@pytest.mark.asyncio
async def test_an_update_arriving_during_a_delayed_flush_still_reaches_the_live_card():
    """The other half of the same race, and the one the throttle actually makes common. Here the
    in-flight write is a TRAILING flush — `_flush_task` is set and not done — so the arriving
    update is correctly refused a second trailing flush (one is the whole point) and correctly
    marks the card dirty. Nobody is then left to act on it: the scheduler declined, and the
    running flush had already rendered its blocks. Only the flush that is finishing can see
    that state, which is why the re-check lives at the end of the flush and not in the
    scheduler."""

    class _SlowDelayedClient(_FakeClient):
        def __init__(self):
            super().__init__()
            self.writing = asyncio.Event()
            self.release = asyncio.Event()
            self.first = True

        async def update_status_card(self, channel_id, ts, text, blocks, receipts=None):
            if self.first:
                self.first = False
                self.writing.set()
                await self.release.wait()
            return await super().update_status_card(channel_id, ts, text, blocks,
                                                    receipts=receipts)

    async def _instant(_delay):
        await asyncio.sleep(0)

    client = _SlowDelayedClient()
    card = rt._ResearchCard(processor=_FakeProcessor(), client=client, channel_id="C1",
                            thread_root="100.0", task="t", label=None, sleep=_instant)
    card.ts = "CARD.1"
    # Inside the throttle window, so the first request DEFERS to a trailing flush rather than
    # writing immediately — that is what makes this the delayed variant.
    card._last_update = card._clock()

    card.bump_steering(1)
    card.request_steering_flush()
    await asyncio.wait_for(client.writing.wait(), timeout=2)
    assert card._flush_task is not None and not card._flush_task.done()

    # Update 2 lands while that trailing flush is parked in Slack, holding its own snapshot.
    card.bump_steering(1)
    card.request_steering_flush()
    client.release.set()

    for _ in range(6):
        await asyncio.gather(*list(card._steering_flush_tasks), return_exceptions=True)
        if card._flush_task is not None:
            await asyncio.gather(card._flush_task, return_exceptions=True)
        await asyncio.sleep(0)

    assert "2 updates passed along" in _context_text(client.card_updates[-1])
    assert card._dirty is False          # nothing left holding unsent state


def test_the_context_line_says_nothing_when_no_update_was_passed_along():
    card = rt._ResearchCard(processor=_FakeProcessor(), client=_FakeClient(), channel_id="C1",
                            thread_root="100.0", task="t", label=None)
    assert "passed along" not in card._context_line()
    card.bump_steering(0)
    assert "passed along" not in card._context_line()


# --------------------------------------------------------------- the in-flight note

class _SuffixHost:
    def __init__(self, tm):
        from message_processor.utilities import MessageUtilitiesMixin
        self._build_research_inflight_note = (
            MessageUtilitiesMixin._build_research_inflight_note.__get__(self))
        self._escape_suffix_text = MessageUtilitiesMixin._escape_suffix_text
        self.thread_manager = tm

    def log_debug(self, *a, **k):
        pass


def test_the_inflight_note_offers_the_steering_call_beside_the_stop_button():
    """The note is where the model learns the job exists at all. Live, it agreed to fold five
    corrections into a running job — so the note has to say that agreeing is not the same as
    sending, and that "passed along" is the most it can claim."""
    tm = AsyncThreadStateManager(db=None)
    tm.register_research("C1:T1", "aaa111bbb222", "map the Q3 pricing shifts", mode="research")
    note = _SuffixHost(tm)._build_research_inflight_note("C1", "T1")
    assert note is not None
    assert "update_background_job(job_id, note)" in note
    assert "cancel_background_job(job_id, reason)" in note
    # Continue vs. abandon, and the honesty rule.
    assert "work that should CONTINUE" in note
    assert "passed along" in note
    assert "only once the job's own output shows it" in note
