"""The in-turn retry key: one dispatch per (turn, tool_call_id), and one effect per dispatch.

WHY THIS EXISTS AT ALL. Nothing in production replays a tool round today — the loop dispatches
each call once, container recovery retries the API request rather than the tools, and the timeout
fallback drops the registry entirely. So this is not a bug fix; it is the guard rail that makes the
first mechanism which DOES re-present a call id harmless. For the image tools a second dispatch is
a second picture, a second bill and a second post into somebody's thread, and there is no way to
take any of those back.

What is defended here, in the order it would hurt:

1. **ONE LAUNCH.** The same call id dispatched twice — sequentially or concurrently, through the
   REAL registry seam — runs the executor once. A duplicate receives the first call's exact result.
2. **AND ONLY WHEN IT IS THE SAME CALL.** An id re-used for different arguments or a different tool
   is refused outright: serving the first result would answer a question nobody asked.
3. **NO SILENT COLLAPSE, BUT NO ORPHANS EITHER.** A call with no id is never deduped — distinct
   calls must not merge onto one key just because neither carried an id — and is still OWNED by
   the turn, under an anonymous flight, so it is drained, cancelled and revoked with the rest.
4. **CANCELLATION IS ESTABLISHED, NOT REQUESTED.** A dispatch that stops waiting does not stop the
   effect; the turn cancels stragglers in its finally, waits for the cancellation to land, and
   revokes whatever refuses.
5. **AN ACCEPTED EFFECT IS ALWAYS ACCOUNTED FOR.** Each of the three effect paths (a message, an
   upload, a container write) is leased across the effect AND its receipt mechanism, settlement
   waits for a held lease, and a revoked turn causes no effect at all.
"""
from __future__ import annotations

import asyncio
import base64
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config import config
from message_processor import image_tools as it
from message_processor import outbound_receipts
from message_processor.turn_runtime import EffectRevoked, TurnRuntime
from openai_client.utilities import ImageData
from thread_manager import AsyncThreadStateManager
from tool_registry import ToolContext, ToolRegistry

pytestmark = pytest.mark.asyncio


# --------------------------------------------------------------------------------- the seam

def _registry(executor, *, name="probe", timeout=None):
    reg = ToolRegistry()
    reg.register({"type": "function", "name": name, "parameters": {}}, executor,
                 timeout=timeout)
    return reg


def _ctx(turn, **over):
    fields = dict(channel_id="C1", thread_ts="10.0", trigger_ts="10.0", turn=turn)
    fields.update(over)
    return ToolContext(**fields)


def _call(name="probe", args="{}", call_id="call_1"):
    return {"name": name, "arguments": args, "call_id": call_id}


async def _let_the_loop_run(turns: int = 5) -> None:
    """Yield to the event loop a bounded number of times.

    Used ONLY where a test asserts that something has NOT happened yet. Deliberately not a sleep:
    every ordering in this file is forced by an Event, and a wall-clock window is a coin flip that
    passes on a fast machine. The real proof in each of those tests is the recorded ORDER at the
    end — if settlement had not waited, "settled" would sit before the effect's own last step."""
    for _ in range(turns):
        await asyncio.sleep(0)


async def test_the_same_call_id_dispatched_twice_in_sequence_runs_once():
    runs = []

    async def _executor(ctx, args):
        runs.append(args)
        return {"ok": True, "n": len(runs)}

    reg, turn = _registry(_executor), TurnRuntime()
    ctx = _ctx(turn)
    first = await reg.dispatch_all(ctx, [_call()])
    second = await reg.dispatch_all(ctx, [_call()])

    assert len(runs) == 1
    # The EXACT original result object, not an equal one: a duplicate is told what the first call
    # was told, so the model cannot read two different accounts of one action.
    assert second[0] is first[0]


async def test_the_same_call_id_twice_in_one_round_runs_once():
    """dispatch_all gathers a round's calls, so the two arrive interleaved rather than in order —
    which is the case a check-then-launch would lose."""
    runs = []
    started = asyncio.Event()

    async def _executor(ctx, args):
        runs.append(1)
        started.set()
        await asyncio.sleep(0)
        return {"ok": True}

    reg, turn = _registry(_executor), TurnRuntime()
    results = await reg.dispatch_all(_ctx(turn), [_call(), _call()])

    assert len(runs) == 1
    assert results[0] is results[1]


async def test_distinct_call_ids_each_run():
    runs = []

    async def _executor(ctx, args):
        runs.append(getattr(ctx, "tool_call_id", None))
        return {"ok": True}

    reg, turn = _registry(_executor), TurnRuntime()
    await reg.dispatch_all(_ctx(turn), [_call(call_id="a"), _call(call_id="b")])

    assert runs == ["a", "b"]


@pytest.mark.parametrize("call_id", [None, ""])
async def test_a_call_with_no_id_is_never_deduped(call_id):
    """(turn_id, None) is not a key — collapsing distinct calls onto it would drop real work."""
    runs = []

    async def _executor(ctx, args):
        runs.append(1)
        return {"ok": True}

    reg, turn = _registry(_executor), TurnRuntime()
    ctx = _ctx(turn)
    await reg.dispatch_all(ctx, [_call(call_id=call_id), _call(call_id=call_id)])
    await reg.dispatch_all(ctx, [_call(call_id=call_id)])

    assert len(runs) == 3


async def test_a_reused_id_with_different_arguments_is_refused_not_served():
    runs = []

    async def _executor(ctx, args):
        runs.append(args)
        return {"ok": True, "for": args.get("q")}

    reg, turn = _registry(_executor), TurnRuntime()
    ctx = _ctx(turn)
    await reg.dispatch_all(ctx, [_call(args='{"q": "one"}')])
    out = await reg.dispatch_all(ctx, [_call(args='{"q": "two"}')])

    assert len(runs) == 1, "the second call must not run under a key already spent"
    assert out[0]["ok"] is False and out[0]["error"] == "duplicate_call_id"
    assert "one" not in str(out[0]), "the first call's answer must not be handed to the second"


async def test_a_reused_id_for_a_different_tool_is_refused():
    reg = ToolRegistry()
    for name in ("probe", "other"):
        reg.register({"type": "function", "name": name, "parameters": {}},
                     AsyncMock(return_value={"ok": True, "name": name}))
    turn = TurnRuntime()
    ctx = _ctx(turn)
    await reg.dispatch_all(ctx, [_call(name="probe")])
    out = await reg.dispatch_all(ctx, [_call(name="other")])

    assert out[0]["error"] == "duplicate_call_id"


async def test_a_plain_failure_result_is_a_completed_call_and_is_cached():
    """{"ok": false} is an ANSWER — the model asked and was told. Re-running it under the same id
    would be the double dispatch, and for an image tool a "failed" call can still have posted."""
    runs = []

    async def _executor(ctx, args):
        runs.append(1)
        return {"ok": False, "error": "moderation_blocked"}

    reg, turn = _registry(_executor), TurnRuntime()
    ctx = _ctx(turn)
    first = await reg.dispatch_all(ctx, [_call()])
    second = await reg.dispatch_all(ctx, [_call()])

    assert len(runs) == 1 and second[0] is first[0]


async def test_a_raise_before_the_launch_clears_the_key_so_a_retry_can_run():
    runs = []

    async def _executor(ctx, args):
        runs.append(1)
        if len(runs) == 1:
            raise RuntimeError("fell over before doing anything")
        return {"ok": True}

    reg, turn = _registry(_executor), TurnRuntime()
    ctx = _ctx(turn)
    first = await reg.dispatch_all(ctx, [_call()])
    second = await reg.dispatch_all(ctx, [_call()])

    assert first[0]["error"] == "execution_error"
    assert len(runs) == 2 and second[0]["ok"] is True


async def test_a_raise_after_the_launch_keeps_the_key_owned():
    """The effect was issued and then something downstream broke. A relaunch would issue it
    again — so the key stays spent and the duplicate is told what the first call was told."""
    runs = []

    async def _executor(ctx, args):
        runs.append(1)
        ctx.tool_flight.mark_launched()
        raise RuntimeError("posted, then the bookkeeping blew up")

    reg, turn = _registry(_executor), TurnRuntime()
    ctx = _ctx(turn)
    first = await reg.dispatch_all(ctx, [_call()])
    second = await reg.dispatch_all(ctx, [_call()])

    assert len(runs) == 1
    assert first[0]["error"] == "execution_error" and second[0]["error"] == "execution_error"


async def _cancel_a_flight_mid_call(*, launched: bool):
    """Drive one call to its first await, then cancel the FLIGHT the way the turn's finally does.
    Returns (registry, ctx, runs) so the caller can re-present the same id."""
    runs, hold = [], asyncio.Event()

    started = asyncio.Event()

    async def _executor(ctx, args):
        runs.append(1)
        if launched:
            ctx.tool_flight.mark_launched()
        started.set()
        await hold.wait()
        return {"ok": True, "run": len(runs)}

    reg, turn = _registry(_executor), TurnRuntime()
    ctx = _ctx(turn)
    dispatch = asyncio.ensure_future(reg.dispatch_all(ctx, [_call()]))
    await started.wait()
    flight = turn.pending_tool_flights[0]
    flight.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await flight.task
    dispatch.cancel()
    with pytest.raises(asyncio.CancelledError):
        await dispatch
    hold.set()
    return reg, ctx, runs


async def test_a_call_cancelled_before_its_launch_can_be_made_again():
    reg, ctx, runs = await _cancel_a_flight_mid_call(launched=False)

    out = await reg.dispatch_all(ctx, [_call()])
    assert len(runs) == 2, "nothing had happened yet, so the key is free"
    assert out[0]["ok"] is True


async def test_a_call_cancelled_after_its_launch_can_never_be_made_again():
    """The effect was already issued. A relaunch here is exactly the doubled picture."""
    reg, ctx, runs = await _cancel_a_flight_mid_call(launched=True)

    out = await reg.dispatch_all(ctx, [_call()])
    assert len(runs) == 1
    assert out[0]["ok"] is False, "the duplicate is refused, not re-run"


async def test_a_duplicate_inherits_the_first_calls_deadline():
    async def _executor(ctx, args):
        return {"ok": True}

    reg, turn = _registry(_executor, timeout=30.0), TurnRuntime()
    ctx = _ctx(turn)
    await reg.dispatch_all(ctx, [_call()])
    flight = list(turn._tool_flights.values())[0]
    stamped = flight.deadline
    await reg.dispatch_all(ctx, [_call()])

    assert flight.deadline == stamped, "a duplicate can neither extend nor overwrite the bound"


# ------------------------------------------------------------------ the per-call context copy

async def test_the_per_call_context_carries_the_identity_and_the_shared_one_does_not():
    seen = {}

    async def _executor(ctx, args):
        seen["id"] = ctx.tool_call_id
        seen["flight"] = ctx.tool_flight
        seen["timeout"] = ctx.tool_timeout
        seen["deadline"] = ctx.tool_deadline
        seen["vision"] = ctx.pending_vision_parts
        return {"ok": True}

    reg, turn = _registry(_executor, timeout=12.0), TurnRuntime()
    ctx = _ctx(turn)
    await reg.dispatch_all(ctx, [_call(call_id="xyz")])

    assert seen["id"] == "xyz" and seen["timeout"] == 12.0
    assert seen["deadline"] == seen["flight"].deadline
    # The SHARED context stays clean: a round's siblings run concurrently, and one mutable
    # "current call" on the shared object would name whichever wrote last.
    assert ctx.tool_call_id is None and ctx.tool_flight is None
    # …while the shared containers are shared BY REFERENCE, so staged vision parts and mounted
    # files are still one record per turn.
    assert seen["vision"] is ctx.pending_vision_parts


async def test_a_flag_set_on_the_copy_reaches_the_handler():
    """`image_generation_started` / `background_job_started` are how a detached producer tells the
    handler to drop the ack reply. Written on the copy, they have to be adopted back — including
    when the call ends by RAISING, which is what a suppressed post does after starting an image."""

    async def _executor(ctx, args):
        ctx.image_generation_started = True
        raise RuntimeError("after the fact")

    reg, turn = _registry(_executor), TurnRuntime()
    ctx = _ctx(turn)
    await reg.dispatch_all(ctx, [_call()])

    assert ctx.image_generation_started is True


# --------------------------------------------------------------- settling what outran its bound

async def test_a_flight_that_outruns_its_bound_is_not_cancelled_by_the_dispatch():
    """The dispatch stops WAITING; the tool keeps going. That is the whole point: a create_image_
    asset mid-mount or an edit_image mid-post must not be torn off at an arbitrary await."""
    state = {"cancelled": False, "finished": False}
    hold = asyncio.Event()

    async def _executor(ctx, args):
        try:
            await hold.wait()
            state["finished"] = True
            return {"ok": True}
        except asyncio.CancelledError:
            state["cancelled"] = True
            raise

    reg, turn = _registry(_executor, timeout=0.02), TurnRuntime()
    out = await reg.dispatch_all(_ctx(turn), [_call()])

    assert out[0]["error"] == "timeout"
    assert state == {"cancelled": False, "finished": False}
    assert len(turn.pending_tool_flights) == 1

    flight = turn.pending_tool_flights[0]
    hold.set()
    await flight.task
    assert state["finished"] is True, "the effect finished on its own after the bound expired"


async def test_the_finally_cancels_a_straggler_and_waits_for_the_cancellation_to_land():
    landed = asyncio.Event()

    async def _executor(ctx, args):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            landed.set()
            raise

    reg, turn = _registry(_executor, timeout=0.02), TurnRuntime()
    await reg.dispatch_all(_ctx(turn), [_call()])

    survivors = await turn.finish_tool_flights()
    assert survivors == ()
    assert landed.is_set(), "cancellation is established, not merely requested"
    assert turn.pending_tool_flights == []


async def test_a_straggler_that_suppresses_cancellation_is_named_and_revoked(caplog):
    hold = asyncio.Event()

    async def _executor(ctx, args):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            await hold.wait()      # the pathological case: refuses to die on request
        return {"ok": True}

    reg, turn = _registry(_executor, name="stubborn", timeout=0.02), TurnRuntime()
    await reg.dispatch_all(_ctx(turn), [_call(name="stubborn")])

    with caplog.at_level("CRITICAL"):
        survivors = await turn.finish_tool_flights(grace=0.05)

    assert survivors == ("stubborn",)
    assert "stubborn" in caplog.text
    # …and it can cause nothing further: the turn is about to settle receipts it could not cover.
    assert turn.effects_revoked is True

    hold.set()                     # let the test's own straggler finish
    await asyncio.gather(*(f.task for f in turn.pending_tool_flights),
                         return_exceptions=True)


async def test_a_cancelled_dispatch_leaves_the_effect_running_and_the_drain_waits_for_it():
    """A turn-level cancellation lands on the dispatch, not on the effect. The handler's
    pre-extraction drain is what then waits for it — up to the SAME stamped deadline."""
    entered, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def _executor(ctx, args):
        entered.set()
        await release.wait()
        finished.set()
        return {"ok": True}

    reg, turn = _registry(_executor, timeout=5.0), TurnRuntime()
    task = asyncio.ensure_future(reg.dispatch_all(_ctx(turn), [_call()]))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not finished.is_set()

    release.set()
    await turn.await_tool_flights()
    assert finished.is_set()


# ============================================================ the three image effect paths

@pytest.fixture(autouse=True)
def _image_tools_on(monkeypatch):
    monkeypatch.setattr(config, "enable_image_tools", True, raising=False)
    monkeypatch.setattr(config, "enable_code_interpreter", True)
    it._reset_semaphore_for_tests()
    yield
    it._reset_semaphore_for_tests()


@pytest.fixture(autouse=True)
def _stub_checklist(monkeypatch):
    class _Checklist:
        def __init__(self, *a, **k):
            self.steps = []

        async def step(self, text, done_text=None):
            self.steps.append(text)

    monkeypatch.setattr("message_processor.progress.ProgressChecklist", _Checklist)


def _image_data():
    return ImageData(base64_data=base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24).decode(),
                     format="png", prompt="an enhanced prompt")


def _image_cfg(**over):
    cfg = {"image_model": "gpt-image-2", "image_size": "1024x1024", "image_quality": "auto",
           "image_background": "auto", "image_format": "png", "image_compression": 100,
           "input_fidelity": "high"}
    cfg.update(over)
    return cfg


class _FakeProcessor:
    def __init__(self, generate=None, edit=None):
        raw = MagicMock()
        raw.containers.files.create = AsyncMock(
            return_value=SimpleNamespace(path="/mnt/data/cover.png"))
        self.openai_client = SimpleNamespace(
            client=raw,
            generate_image=generate or AsyncMock(return_value=_image_data()),
            edit_image=edit or AsyncMock(return_value=_image_data()))
        self.thread_manager = AsyncThreadStateManager(db=None)
        self.scheduled = []
        self.aborted = 0
        self._finish_image_generation_background = AsyncMock()

    def _schedule_async_call(self, coro):
        self.scheduled.append(coro)
        coro.close()
        return SimpleNamespace(done=lambda: False, cancel=lambda: None)

    async def _abort_checklist(self, *a, **k):
        self.aborted += 1


CATALOG = [{"image_id": "img_7", "url": "https://files.slack.com/red-cat.png",
            "kind": "generated", "prompt": "p", "analysis": "A red cat"}]


def _image_ctx(turn, processor, **over):
    fields = dict(channel_id="C1", thread_ts="100.0", trigger_ts="100.5",
                  client=MagicMock(), processor=processor, db=SimpleNamespace(),
                  thread_config=_image_cfg(), turn=turn, sandbox_image_assets=[])
    fields.update(over)
    return ToolContext(**fields)


def _image_registry(processor):
    reg = ToolRegistry()
    reg.register({"type": "function", "name": "generate_image", "parameters": {}},
                 it.execute_generate_image)
    reg.register({"type": "function", "name": "create_image_asset", "parameters": {}},
                 it.execute_create_image_asset)
    reg.register({"type": "function", "name": "edit_image", "parameters": {}},
                 it.execute_edit_image)
    return reg


@pytest.mark.parametrize("replay", ["sequential", "concurrent"])
async def test_a_replayed_generate_image_dispatches_exactly_one_job(replay):
    processor = _FakeProcessor()
    turn = TurnRuntime()
    ctx = _image_ctx(turn, processor)
    reg = _image_registry(processor)
    call = _call(name="generate_image", args='{"prompt": "a red cat"}', call_id="gen_1")

    if replay == "concurrent":
        results = await reg.dispatch_all(ctx, [call, dict(call)])
    else:
        first = await reg.dispatch_all(ctx, [call])
        second = await reg.dispatch_all(ctx, [dict(call)])
        results = [first[0], second[0]]

    assert results[0]["ok"] is True
    assert results[1] is results[0], "the duplicate gets the original payload, not a new job"
    assert len(processor.scheduled) == 1
    assert len(processor.thread_manager.generations_in_flight("C1:100.0")) == 1


@pytest.mark.parametrize("replay", ["sequential", "concurrent"])
async def test_a_replayed_create_image_asset_generates_and_mounts_once(replay):
    processor = _FakeProcessor()
    turn = TurnRuntime()
    ctx = _image_ctx(turn, processor, container_id="cntr_1")
    reg = _image_registry(processor)
    call = _call(name="create_image_asset",
                 args='{"prompt": "a cover", "filename": "cover.png"}', call_id="asset_1")

    if replay == "concurrent":
        results = await reg.dispatch_all(ctx, [call, dict(call)])
    else:
        first = await reg.dispatch_all(ctx, [call])
        second = await reg.dispatch_all(ctx, [dict(call)])
        results = [first[0], second[0]]

    assert results[0]["ok"] is True and results[1] is results[0]
    assert processor.openai_client.generate_image.await_count == 1
    assert processor.openai_client.client.containers.files.create.await_count == 1
    assert len(ctx.sandbox_image_assets) == 1


@pytest.mark.parametrize("replay", ["sequential", "concurrent"])
async def test_a_replayed_edit_image_edits_and_posts_once(replay, monkeypatch):
    posts = []

    async def _publish(**kwargs):
        posts.append(kwargs["thread_id"])
        return "https://files.slack.com/edited.png"

    monkeypatch.setattr("message_processor.image_delivery.publish_image", _publish)
    monkeypatch.setattr(it, "_download_edit_source",
                        AsyncMock(return_value=("Zm9v", "image/png")))
    processor = _FakeProcessor()
    turn = TurnRuntime()
    ctx = _image_ctx(turn, processor, image_catalog=CATALOG)
    reg = _image_registry(processor)
    call = _call(name="edit_image",
                 args='{"source_image_ids": ["img_7"], "prompt": "make it blue"}',
                 call_id="edit_1")

    if replay == "concurrent":
        results = await reg.dispatch_all(ctx, [call, dict(call)])
    else:
        first = await reg.dispatch_all(ctx, [call])
        second = await reg.dispatch_all(ctx, [dict(call)])
        results = [first[0], second[0]]

    assert results[0]["ok"] is True and results[1] is results[0]
    assert processor.openai_client.edit_image.await_count == 1
    assert len(posts) == 1, "one edit, one post"


async def test_a_generation_cancelled_before_its_launch_leaves_no_slot_and_no_checklist():
    """The one visible residue of a call that did nothing: a status card promising an image no job
    will make. Compensating cleanup is allowed precisely because taking something back is never
    the effect the lease withholds."""
    processor = _FakeProcessor()
    turn = TurnRuntime()
    ctx = _image_ctx(turn, processor)
    reg = _image_registry(processor)

    async def _slow_schedule(coro):
        coro.close()
        await asyncio.sleep(30)

    hold, stepped = asyncio.Event(), asyncio.Event()

    async def _blocking_step(self, *a, **k):
        stepped.set()
        await hold.wait()

    # Wedge the turn INSIDE the checklist step, which is after the card exists and before the job
    # is scheduled — the precise window the cleanup is for.
    from message_processor import progress
    original = progress.ProgressChecklist.step
    progress.ProgressChecklist.step = _blocking_step
    try:
        task = asyncio.ensure_future(reg.dispatch_all(
            ctx, [_call(name="generate_image", args='{"prompt": "x"}', call_id="g")]))
        await stepped.wait()
        flight = turn.pending_tool_flights[0]
        flight.task.cancel()
        task.cancel()
        for pending in (task, flight.task):
            with pytest.raises(asyncio.CancelledError):
                await pending
    finally:
        progress.ProgressChecklist.step = original
        hold.set()

    assert processor.scheduled == [], "nothing was ever launched"
    assert processor.aborted == 1, "the checklist that was already up came down"
    assert processor.thread_manager.generations_in_flight("C1:100.0") == []
    assert turn.pending_tool_flights == []


# =================================================== effect leases vs settlement + revocation
#
# Both orderings, on each of the three effect paths. These are the tests that make "an accepted
# effect is always accounted for" a property rather than a hope: settlement WAITS for a lease that
# is already held, and revocation WINS when it arrives before one is taken.


class _StubLedger:
    """Enough of a ReceiptLedger for settle_ledger to drive, recording the ORDER of events."""

    def __init__(self, order):
        self.order = order
        self.channel_id = "C1"
        self.service = None

    async def settle(self):
        self.order.append("settled")
        return 0


async def _drive_settlement(turn, order):
    settle = asyncio.ensure_future(
        outbound_receipts.settle_ledger(_StubLedger(order), turn=turn))
    await _let_the_loop_run()
    assert "settled" not in order, "settlement ran while an effect lease was still held"
    return settle


async def test_settlement_waits_for_a_held_message_lease(monkeypatch):
    from slack_client.messaging import SlackMessagingMixin

    order, release, started = [], asyncio.Event(), asyncio.Event()
    host = MagicMock()
    host.execute_post_to_thread = SlackMessagingMixin.execute_post_to_thread.__get__(host)

    async def _send(*a, **k):
        order.append("slack_accepted")
        started.set()
        await release.wait()
        order.append("receipt_registered")     # note_post, inside the same critical section
        return "900.0"

    host.send_message = _send
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    turn = TurnRuntime()
    ctx = ToolContext(channel_id="C1", thread_ts="10.0", trigger_ts="10.0", turn=turn)
    effect = asyncio.ensure_future(
        host.execute_post_to_thread(ctx, {"thread_ts": "OTHER.9", "text": "over here"}))
    await started.wait()
    assert turn.held_effect_leases == ("post_to_thread",)

    settle = await _drive_settlement(turn, order)
    release.set()
    await effect
    await settle

    assert order == ["slack_accepted", "receipt_registered", "settled"]


async def test_settlement_waits_for_a_held_upload_lease(monkeypatch):
    order, release, started = [], asyncio.Event(), asyncio.Event()

    async def _publish(**kwargs):
        order.append("upload_accepted")
        started.set()
        await release.wait()
        order.append("pending_share_recorded")
        return "https://files.slack.com/edited.png"

    monkeypatch.setattr("message_processor.image_delivery.publish_image", _publish)
    monkeypatch.setattr(it, "_download_edit_source",
                        AsyncMock(return_value=("Zm9v", "image/png")))
    processor = _FakeProcessor()
    turn = TurnRuntime()
    ctx = _image_ctx(turn, processor, image_catalog=CATALOG)
    effect = asyncio.ensure_future(it.execute_edit_image(
        ctx, {"source_image_ids": ["img_7"], "prompt": "make it blue"}))
    await started.wait()
    assert turn.held_effect_leases == ("edit_image.publish",)

    settle = await _drive_settlement(turn, order)
    release.set()
    await effect
    await settle

    assert order == ["upload_accepted", "pending_share_recorded", "settled"]


async def test_settlement_waits_for_a_held_container_mount(monkeypatch):
    order, release, started = [], asyncio.Event(), asyncio.Event()

    async def _mount(*a, **k):
        order.append("mount_started")
        started.set()
        await release.wait()
        order.append("mount_finished")
        return "/mnt/data/cover.png"

    monkeypatch.setattr(it, "mount_image_in_container", _mount)
    processor = _FakeProcessor()
    turn = TurnRuntime()
    ctx = _image_ctx(turn, processor, container_id="cntr_1")
    effect = asyncio.ensure_future(it.execute_create_image_asset(
        ctx, {"prompt": "a cover", "filename": "cover.png"}))
    await started.wait()
    assert turn.held_effect_leases == ("create_image_asset.mount",)

    settle = await _drive_settlement(turn, order)
    release.set()
    await effect
    await settle

    assert order == ["mount_started", "mount_finished", "settled"]


async def test_revocation_wins_before_a_message_lease_is_taken(monkeypatch):
    from slack_client.messaging import SlackMessagingMixin

    host = MagicMock()
    host.execute_post_to_thread = SlackMessagingMixin.execute_post_to_thread.__get__(host)
    host.send_message = AsyncMock(return_value="900.0")
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    turn = TurnRuntime()
    turn.revoke_effects("straggler refused cancellation")
    ctx = ToolContext(channel_id="C1", thread_ts="10.0", trigger_ts="10.0", turn=turn)

    out = await host.execute_post_to_thread(ctx, {"thread_ts": "OTHER.9", "text": "hi"})

    assert out["ok"] is False and out["error"] == "turn_cancelled"
    host.send_message.assert_not_awaited()          # no effect
    assert turn.destinations == []                   # …so no receipt and no record
    assert turn.visible_action_committed is False


async def test_revocation_wins_before_an_upload_lease_is_taken(monkeypatch):
    published = []

    async def _publish(**kwargs):
        published.append(1)
        return "url"

    monkeypatch.setattr("message_processor.image_delivery.publish_image", _publish)
    monkeypatch.setattr(it, "_download_edit_source",
                        AsyncMock(return_value=("Zm9v", "image/png")))
    processor = _FakeProcessor()
    turn = TurnRuntime()
    ctx = _image_ctx(turn, processor, image_catalog=CATALOG)

    # Revoked AFTER the edit ran, so the refusal has to happen at the lease rather than at entry.
    async def _edit(**kwargs):
        turn.revoke_effects("straggler refused cancellation")
        return _image_data()

    processor.openai_client.edit_image = _edit
    out = await it.execute_edit_image(ctx, {"source_image_ids": ["img_7"],
                                            "prompt": "make it blue"})

    assert out["ok"] is False and out["error"] == "turn_cancelled"
    assert published == []
    assert turn.visible_action_committed is False


async def test_revocation_wins_before_a_container_mount(monkeypatch):
    mounts = []

    async def _mount(*a, **k):
        mounts.append(1)
        return "/mnt/data/cover.png"

    monkeypatch.setattr(it, "mount_image_in_container", _mount)
    processor = _FakeProcessor()
    turn = TurnRuntime()
    ctx = _image_ctx(turn, processor, container_id="cntr_1")

    async def _generate(**kwargs):
        turn.revoke_effects("straggler refused cancellation")
        return _image_data()

    processor.openai_client.generate_image = _generate
    out = await it.execute_create_image_asset(ctx, {"prompt": "a cover",
                                                    "filename": "cover.png"})

    assert out["ok"] is False and out["error"] == "turn_cancelled"
    assert mounts == []
    assert ctx.sandbox_image_assets == [], "the reservation is released, not left dangling"


async def test_a_revoked_turn_starts_no_detached_generation():
    processor = _FakeProcessor()
    turn = TurnRuntime()
    turn.revoke_effects("straggler refused cancellation")
    ctx = _image_ctx(turn, processor)

    out = await it.execute_generate_image(ctx, {"prompt": "a red cat"})

    assert out["ok"] is False and out["error"] == "turn_cancelled"
    assert processor.scheduled == []
    assert processor.thread_manager.generations_in_flight("C1:100.0") == []


async def test_a_lease_already_held_is_never_interrupted_by_revocation():
    """Revocation stops the NEXT effect. Interrupting one in flight is the very failure the lease
    exists to prevent — a post accepted by Slack with its receipt half-written."""
    turn = TurnRuntime()
    release, done = asyncio.Event(), []

    async def _effect():
        await release.wait()
        done.append("finished")
        return "ok"

    task = asyncio.ensure_future(turn.run_leased_effect("probe", _effect))
    await _let_the_loop_run()
    turn.revoke_effects("straggler refused cancellation")
    release.set()
    assert await task == "ok"
    assert done == ["finished"]

    with pytest.raises(EffectRevoked):
        await turn.run_leased_effect("probe", _effect)


async def test_a_caller_that_stops_waiting_does_not_stop_the_effect():
    """The other half of the same rule: the awaiter is cancelled, the critical section is not."""
    turn = TurnRuntime()
    release, done = asyncio.Event(), []

    async def _effect():
        await release.wait()
        done.append("finished")
        return "ok"

    task = asyncio.ensure_future(turn.run_leased_effect("probe", _effect))
    await _let_the_loop_run()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert turn.held_effect_leases == ("probe",), "the lease is still held by the live effect"
    release.set()
    assert await turn.wait_for_effects() == ()
    assert done == ["finished"]


async def test_settlement_never_proceeds_while_a_lease_is_held():
    """THE invariant, and the reason there is no clock on this wait.

    There used to be one — a flat 30 seconds — and it was shorter than a legitimate Slack
    multipart send, so settlement could finalize this turn's receipts while its own post was
    still being accepted and the message would be permanently unclaimed. The wait terminates
    because the lease body is already bounded by its own transport, not because we imposed a
    second, shorter deadline on top of it."""
    turn = TurnRuntime()
    release, order = asyncio.Event(), []

    async def _effect():
        # Longer than any bound we might have been tempted to put on the wait.
        await release.wait()
        order.append("effect_finished")

    task = asyncio.ensure_future(turn.run_leased_effect("probe", _effect))
    settle = await _drive_settlement(turn, order)
    for _ in range(10):
        await _let_the_loop_run()
    assert order == [], "settlement proceeded through a held lease"

    release.set()
    await settle
    assert order == ["effect_finished", "settled"]
    await task


async def test_a_failed_lease_body_does_not_hold_settlement_open():
    """A lease body that RAISES is over: the effect is not in flight any more, so settlement
    proceeds — and the failure belongs to the executor that ran it, not to the settle."""
    turn = TurnRuntime()
    order = []

    async def _effect():
        raise RuntimeError("transport blew up")

    task = asyncio.ensure_future(turn.run_leased_effect("probe", _effect))
    with pytest.raises(RuntimeError):
        await task
    await outbound_receipts.settle_ledger(_StubLedger(order), turn=turn)
    assert order == ["settled"]


async def test_a_broken_effect_wait_stops_the_settle_rather_than_being_stepped_over():
    """If the wait for live effects cannot be performed, there is no state in which settling
    anyway is the right answer — the whole point of the wait is that what it protects is
    invisible afterwards. The failure is surfaced, not logged and passed."""
    order = []
    turn = SimpleNamespace(
        wait_for_effects=MagicMock(side_effect=RuntimeError("lease bookkeeping is broken")))
    with pytest.raises(RuntimeError):
        await outbound_receipts.settle_ledger(_StubLedger(order), turn=turn)
    assert order == [], "the ledger settled through a wait that never happened"


async def test_an_abandoned_lease_failure_is_consumed():
    """The caller walked away and the body then failed. Nobody is left to retrieve that
    exception, and an unretrieved one surfaces as a stray asyncio warning at exit."""
    turn = TurnRuntime()
    release = asyncio.Event()

    async def _effect():
        await release.wait()
        raise RuntimeError("failed after the awaiter gave up")

    waiter = asyncio.ensure_future(turn.run_leased_effect("probe", _effect))
    await _let_the_loop_run()
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    release.set()
    assert await turn.wait_for_effects() == ()
    assert turn.held_effect_leases == ()


# ======================================================= the launch transition on the post path
#
# post_to_thread goes through the generic registry seam, so it has a flight like any other call —
# and it is the one non-image tool whose effect cannot be taken back. Without the transition its
# flight looked pre-launch forever, so a cancellation dropped the key and a replay of the same
# call id was free to post the message a second time.


def _post_host(send):
    from slack_client.messaging import SlackMessagingMixin

    host = MagicMock()
    host.execute_post_to_thread = SlackMessagingMixin.execute_post_to_thread.__get__(host)
    host.send_message = send
    return host


def _post_registry(host):
    reg = ToolRegistry()
    reg.register({"type": "function", "name": "post_to_thread", "parameters": {}},
                 host.execute_post_to_thread)
    return reg


def _post_call(call_id="post_1", target="OTHER.9"):
    return _call(name="post_to_thread",
                 args='{"thread_ts": "%s", "text": "over here"}' % target, call_id=call_id)


async def test_the_post_is_marked_launched_before_slack_sees_it(monkeypatch):
    launched_at_send = []
    release = asyncio.Event()
    started = asyncio.Event()

    async def _send(*a, **k):
        launched_at_send.append(turn.pending_tool_flights[0].launched)
        started.set()
        await release.wait()
        return "900.0"

    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    turn = TurnRuntime()
    reg = _post_registry(_post_host(_send))
    task = asyncio.ensure_future(reg.dispatch_all(_ctx(turn), [_post_call()]))
    await started.wait()

    assert launched_at_send == [True], "the transition happened before the irreversible step"
    release.set()
    await task


async def test_a_cancelled_post_keeps_its_key_and_a_replay_cannot_post_twice(monkeypatch):
    """The double-post window: the executor's flight is cancelled while the shielded send is still
    going. If the key were dropped there, the same call id presented again would post again."""
    sends, release, started = [], asyncio.Event(), asyncio.Event()

    async def _send(*a, **k):
        sends.append(1)
        started.set()
        await release.wait()
        return "900.0"

    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    turn = TurnRuntime()
    reg = _post_registry(_post_host(_send))
    ctx = _ctx(turn)
    first = asyncio.ensure_future(reg.dispatch_all(ctx, [_post_call()]))
    await started.wait()
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    flight = turn.pending_tool_flights[0]
    assert flight.launched is True, "a send in flight is a launched call"

    replay = asyncio.ensure_future(reg.dispatch_all(ctx, [_post_call()]))
    await _let_the_loop_run()
    release.set()
    out = await replay

    assert sends == [1], "the replay joined the first post instead of making a second one"
    assert out[0]["ok"] is True and out[0]["posted_ts"] == "900.0"


async def test_the_foreign_post_completes_its_own_bookkeeping_after_first_accept(monkeypatch):
    """Cancelled after Slack took the first part. The shielded body owns delivery AND everything
    that describes it — without that the room has our words in another thread while the turn
    reports an observation with no commitment and no visible action."""
    release, accepted = asyncio.Event(), asyncio.Event()

    async def _send(channel_id, target, text, **k):
        k["on_first_accept"]("900.0")
        accepted.set()
        await release.wait()
        return "900.0"

    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    turn = TurnRuntime()
    host = _post_host(_send)
    ctx = ToolContext(channel_id="C1", thread_ts="10.0", trigger_ts="10.0", turn=turn)
    effect = asyncio.ensure_future(
        host.execute_post_to_thread(ctx, {"thread_ts": "OTHER.9", "text": "over here"}))
    await accepted.wait()
    effect.cancel()
    with pytest.raises(asyncio.CancelledError):
        await effect

    assert turn.visible_action_committed is False, "nothing is claimed before the send returns"
    release.set()
    assert await turn.wait_for_effects() == ()

    assert turn.visible_action_committed is True
    committed = turn.committed_destinations
    assert len(committed) == 1
    record = committed[0]
    assert (record.kind, record.thread_root_ts, record.first_ts) == (
        "post_to_thread", "OTHER.9", "900.0")


# ============================================ final state, not transport ordering (findings 4/5)


async def test_a_cancelled_mount_leaves_the_image_MOUNTED_AND_RESERVED(monkeypatch):
    """The state the lease protects is not "the upload happened", it is "the sandbox and this
    turn's accounting agree". A caller cancelled mid-mount used to leave bytes in the shared
    container with the reservation removed as uncommitted — an asset the model can open and the
    turn denies having made."""
    release, started = asyncio.Event(), asyncio.Event()

    async def _mount(*a, **k):
        started.set()
        await release.wait()
        return "/mnt/data/cover.png"

    monkeypatch.setattr(it, "mount_image_in_container", _mount)
    processor = _FakeProcessor()
    turn = TurnRuntime()
    ctx = _image_ctx(turn, processor, container_id="cntr_1")
    effect = asyncio.ensure_future(it.execute_create_image_asset(
        ctx, {"prompt": "a cover", "filename": "cover.png"}))
    await started.wait()
    effect.cancel()
    with pytest.raises(asyncio.CancelledError):
        await effect

    release.set()
    assert await turn.wait_for_effects() == ()

    assert len(ctx.sandbox_image_assets) == 1
    asset = ctx.sandbox_image_assets[0]
    assert asset["path"] == "/mnt/data/cover.png"
    assert asset["image_data"] is not None, "mounted and reserved, or neither"


async def test_a_failed_mount_leaves_NEITHER(monkeypatch):
    """The other half of the same rule, and the reason the release moved inside the body too."""
    monkeypatch.setattr(it, "mount_image_in_container", AsyncMock(return_value=None))
    processor = _FakeProcessor()
    turn = TurnRuntime()
    ctx = _image_ctx(turn, processor, container_id="cntr_1")

    out = await it.execute_create_image_asset(ctx, {"prompt": "a cover", "filename": "cover.png"})

    assert out["ok"] is False and out["error"] == "mount_failed"
    assert ctx.sandbox_image_assets == []


async def test_a_cancelled_edit_still_records_the_picture_it_posted(monkeypatch):
    """Posted and signalled together. The signal is what stops the end-of-turn settle reading a
    turn that delivered a picture as a silence and retracting its 👀."""
    release, started = asyncio.Event(), asyncio.Event()

    async def _publish(**kwargs):
        started.set()
        await release.wait()
        return "https://files.slack.com/edited.png"

    monkeypatch.setattr("message_processor.image_delivery.publish_image", _publish)
    monkeypatch.setattr(it, "_download_edit_source",
                        AsyncMock(return_value=("Zm9v", "image/png")))
    processor = _FakeProcessor()
    turn = TurnRuntime()
    ctx = _image_ctx(turn, processor, image_catalog=CATALOG)
    effect = asyncio.ensure_future(it.execute_edit_image(
        ctx, {"source_image_ids": ["img_7"], "prompt": "make it blue"}))
    await started.wait()
    effect.cancel()
    with pytest.raises(asyncio.CancelledError):
        await effect

    assert turn.visible_action_committed is False, "nothing is claimed before the upload lands"
    release.set()
    assert await turn.wait_for_effects() == ()
    assert turn.visible_action_committed is True


# ================================================= revocation at every launch boundary (finding 6)
#
# The checks that matter are the LAST ones before the irreversible step, not the first ones after
# entry: every await in between is a window in which the turn can give up. Each test below revokes
# during one of those intermediate awaits and asserts nothing was launched and nothing was paid for.


async def test_revoking_during_the_checklist_post_stops_the_detached_job():
    processor = _FakeProcessor()
    turn = TurnRuntime()
    ctx = _image_ctx(turn, processor)

    class _RevokingChecklist:
        def __init__(self, *a, **k):
            pass

        async def step(self, text, done_text=None):
            turn.revoke_effects("the turn was cancelled while the card went up")

    with patch("message_processor.progress.ProgressChecklist", _RevokingChecklist):
        out = await it.execute_generate_image(ctx, {"prompt": "a red cat"})

    assert out["ok"] is False and out["error"] == "turn_cancelled"
    assert processor.scheduled == [], "no job"
    assert processor.thread_manager.generations_in_flight("C1:100.0") == [], "no leaked slot"
    assert processor.aborted == 1, "the card that was already up came down"


def _occupy_the_semaphore(monkeypatch) -> asyncio.Semaphore:
    """Hold the ONE image slot, so the next executor genuinely queues on it."""
    monkeypatch.setattr(config, "max_concurrent_image_generations", 1)
    it._reset_semaphore_for_tests()
    return it._semaphore()


async def test_revoking_while_a_creation_waits_for_the_semaphore_pays_for_nothing(monkeypatch):
    """The semaphore wait is unbounded — a busy bot can hold a call here for the length of
    another generation — so it is the longest window between the entry check and the request."""
    processor = _FakeProcessor()
    turn = TurnRuntime()
    ctx = _image_ctx(turn, processor, container_id="cntr_1")

    semaphore = _occupy_the_semaphore(monkeypatch)
    await semaphore.acquire()
    try:
        effect = asyncio.ensure_future(it.execute_create_image_asset(
            ctx, {"prompt": "a cover", "filename": "cover.png"}))
        await _let_the_loop_run()
        turn.revoke_effects("the turn was cancelled while we queued")
    finally:
        semaphore.release()
    out = await effect

    assert out["ok"] is False and out["error"] == "turn_cancelled"
    processor.openai_client.generate_image.assert_not_awaited()
    assert ctx.sandbox_image_assets == []


async def test_revoking_while_an_edit_waits_for_the_semaphore_pays_for_nothing(monkeypatch):
    published = []

    async def _publish(**kwargs):
        published.append(1)
        return "url"

    monkeypatch.setattr("message_processor.image_delivery.publish_image", _publish)
    monkeypatch.setattr(it, "_download_edit_source",
                        AsyncMock(return_value=("Zm9v", "image/png")))
    processor = _FakeProcessor()
    turn = TurnRuntime()
    ctx = _image_ctx(turn, processor, image_catalog=CATALOG)

    semaphore = _occupy_the_semaphore(monkeypatch)
    await semaphore.acquire()
    try:
        effect = asyncio.ensure_future(it.execute_edit_image(
            ctx, {"source_image_ids": ["img_7"], "prompt": "make it blue"}))
        await _let_the_loop_run()
        turn.revoke_effects("the turn was cancelled while we queued")
    finally:
        semaphore.release()
    out = await effect

    assert out["ok"] is False and out["error"] == "turn_cancelled"
    processor.openai_client.edit_image.assert_not_awaited()
    assert published == []
    assert processor.aborted == 1, "the card that was already up came down"


async def test_revoking_during_an_edits_downloads_pays_for_nothing(monkeypatch):
    """The other intermediate await on the edit path: fetching the sources from Slack."""
    processor = _FakeProcessor()
    turn = TurnRuntime()

    async def _download(client, url):
        turn.revoke_effects("the turn was cancelled mid-download")
        return "Zm9v", "image/png"

    monkeypatch.setattr(it, "_download_edit_source", _download)
    ctx = _image_ctx(turn, processor, image_catalog=CATALOG)

    out = await it.execute_edit_image(ctx, {"source_image_ids": ["img_7"],
                                            "prompt": "make it blue"})

    assert out["ok"] is False and out["error"] == "turn_cancelled"
    processor.openai_client.edit_image.assert_not_awaited()


# ============================================== the plumbing fails CLOSED (finding 7)
#
# Every one of these is the single-flight mechanism itself breaking, on a turn that HAS a real
# call id. Running the tool anyway would be choosing unprotected execution of exactly the calls
# that must happen once, at the moment the bookkeeping is known to be broken.


class _BrokenFlights(TurnRuntime):
    def open_tool_flight(self, **kwargs):
        raise RuntimeError("the flight table is broken")


class _BrokenLaunch(TurnRuntime):
    """Fails exactly as a real launch failure would: the coroutine is left where it was handed
    over. The double used to close it, which quietly did the caller's job and hid whether the
    caller does it."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.handed_over = None

    def launch_tool_flight(self, flight, coro):
        self.handed_over = coro
        raise RuntimeError("could not own the work")


async def test_a_call_whose_flight_cannot_be_opened_is_refused_not_run():
    runs = []

    async def _executor(ctx, args):
        runs.append(1)
        return {"ok": True}

    reg = _registry(_executor)
    out = await reg.dispatch_all(_ctx(_BrokenFlights()), [_call()])

    assert out[0]["ok"] is False and out[0]["error"] == "flight_unavailable"
    assert runs == [], "nothing ran unprotected"


async def test_a_call_whose_flight_cannot_be_launched_is_refused_not_run():
    runs = []

    async def _executor(ctx, args):
        runs.append(1)
        return {"ok": True}

    turn = _BrokenLaunch()
    reg = _registry(_executor)
    out = await reg.dispatch_all(_ctx(turn), [_call()])

    assert out[0]["ok"] is False and out[0]["error"] == "flight_unavailable"
    assert runs == []
    # The key is handed back: nothing was launched under it, so a fresh call may still try.
    assert turn._tool_flights == {}
    # And the executor coroutine nobody took is CLOSED by the caller that made it — left open it
    # is a "coroutine was never awaited" warning hanging off an otherwise honest refusal.
    assert inspect.getcoroutinestate(turn.handed_over) == inspect.CORO_CLOSED


async def test_a_call_that_cannot_be_stamped_with_its_identity_is_refused_not_run(monkeypatch):
    """An unstamped context is a call running with its duplicate protection silently off — the
    executor has no flight to mark launched. That is the one state this must never produce."""
    runs = []

    async def _executor(ctx, args):
        runs.append(1)
        return {"ok": True}

    import tool_registry as tr
    monkeypatch.setattr(tr.copy, "copy", MagicMock(side_effect=RuntimeError("no copy")))
    turn = TurnRuntime()
    reg = _registry(_executor)
    out = await reg.dispatch_all(_ctx(turn), [_call()])

    assert out[0]["ok"] is False and out[0]["error"] == "flight_unavailable"
    assert runs == []
    assert turn._tool_flights == {}


async def test_a_generation_whose_launch_cannot_be_recorded_is_never_scheduled():
    """`mark_launched` is what stops a replay dispatching a second job. Swallowing its failure
    and scheduling anyway is the double-fire this whole mechanism exists to prevent."""
    processor = _FakeProcessor()
    turn = TurnRuntime()
    broken = MagicMock()
    broken.mark_launched.side_effect = RuntimeError("flight is broken")
    ctx = _image_ctx(turn, processor, tool_flight=broken)

    out = await it.execute_generate_image(ctx, {"prompt": "a red cat"})

    assert out["ok"] is False and out["error"] == "launch_not_recorded"
    assert processor.scheduled == []
    assert processor.thread_manager.generations_in_flight("C1:100.0") == []
    assert processor.aborted == 1


async def test_a_creation_whose_launch_cannot_be_recorded_never_pays_for_an_image():
    processor = _FakeProcessor()
    turn = TurnRuntime()
    broken = MagicMock()
    broken.mark_launched.side_effect = RuntimeError("flight is broken")
    ctx = _image_ctx(turn, processor, container_id="cntr_1", tool_flight=broken)

    out = await it.execute_create_image_asset(ctx, {"prompt": "a cover", "filename": "cover.png"})

    assert out["ok"] is False and out["error"] == "launch_not_recorded"
    processor.openai_client.generate_image.assert_not_awaited()
    assert ctx.sandbox_image_assets == []


async def test_an_edit_whose_launch_cannot_be_recorded_never_pays_for_an_image(monkeypatch):
    monkeypatch.setattr(it, "_download_edit_source",
                        AsyncMock(return_value=("Zm9v", "image/png")))
    processor = _FakeProcessor()
    turn = TurnRuntime()
    broken = MagicMock()
    broken.mark_launched.side_effect = RuntimeError("flight is broken")
    ctx = _image_ctx(turn, processor, image_catalog=CATALOG, tool_flight=broken)

    out = await it.execute_edit_image(ctx, {"source_image_ids": ["img_7"],
                                            "prompt": "make it blue"})

    assert out["ok"] is False and out["error"] == "launch_not_recorded"
    processor.openai_client.edit_image.assert_not_awaited()
    assert processor.aborted == 1


async def test_a_post_whose_launch_cannot_be_recorded_is_never_sent(monkeypatch):
    monkeypatch.setattr(config, "enable_post_to_thread_tool", True)
    host = _post_host(AsyncMock(return_value="900.0"))
    turn = TurnRuntime()
    broken = MagicMock()
    broken.mark_launched.side_effect = RuntimeError("flight is broken")
    ctx = ToolContext(channel_id="C1", thread_ts="10.0", trigger_ts="10.0", turn=turn,
                      tool_flight=broken)

    out = await host.execute_post_to_thread(ctx, {"thread_ts": "OTHER.9", "text": "over here"})

    assert out["ok"] is False and out["error"] == "launch_not_recorded"
    host.send_message.assert_not_awaited()
    assert turn.destinations == []


# ------------------------------------------------- the fences fail CLOSED, and own every sibling
#
# Both of these are the same failure in two places: the turn can no longer PROVE what is still
# running. Neither is a tool failing — those are absorbed by the drain and belong to their own
# dispatch. What is left when the bookkeeping itself breaks is unknown state, and the only honest
# move is to make the unknown harmless before anything settles.


async def test_a_broken_drain_revokes_before_the_round_s_results_are_read():
    """The handler's fence (text.py). It cannot refuse to answer — the model has already spoken —
    so it does the one thing that still holds: nothing new may be caused behind it."""
    from message_processor.handlers.text import _settle_tool_flights

    class _BrokenDrain(TurnRuntime):
        async def await_tool_flights(self):
            raise RuntimeError("the flight table is broken")

    turn = _BrokenDrain()
    await _settle_tool_flights(turn)

    assert turn.effects_revoked is True


async def test_a_broken_drain_waits_out_the_lease_revocation_cannot_interrupt():
    """Revoking stops the NEXT effect and touches nothing already in flight. So the fence that
    only revoked still returned while a post was mid-acceptance, and the round's results were
    read around it. It now waits the held leases out — completion-bound — before extraction."""
    from message_processor.handlers.text import _settle_tool_flights
    from message_processor.turn_runtime import run_effect

    order, release = [], asyncio.Event()

    class _BrokenDrain(TurnRuntime):
        async def await_tool_flights(self):
            raise RuntimeError("the flight table is broken")

    turn = _BrokenDrain()

    async def _slow_effect():
        await release.wait()
        order.append("effect finished")

    effect = asyncio.ensure_future(run_effect(turn, "probe", _slow_effect))
    await _let_the_loop_run()
    assert turn.held_effect_leases == ("probe",), "the lease is held when the fence is entered"

    fence = asyncio.ensure_future(_settle_tool_flights(turn))
    await _let_the_loop_run(10)
    assert turn.effects_revoked is True, "nothing NEW may be caused behind the extraction"
    assert not fence.done(), "the fence returned with a lease still held"

    release.set()
    await fence
    order.append("results extracted")
    await effect

    assert order == ["effect finished", "results extracted"]
    assert turn.held_effect_leases == ()


async def test_a_fence_whose_last_resorts_also_fail_aborts_extraction():
    """[codex r4] Both booleans exist so the fence can act on them: a drain that broke AND a
    revocation that also failed leaves unknown, unrevoked state — returning to result extraction
    over that state is the one move left that must not happen."""
    from message_processor.handlers.text import _settle_tool_flights
    from message_processor.turn_runtime import TurnEffectsUnsettled

    class _EverythingBroken(TurnRuntime):
        async def await_tool_flights(self):
            raise RuntimeError("the flight table is broken")

        def revoke_effects(self, reason):
            raise RuntimeError("and so is the revocation")

    with pytest.raises(TurnEffectsUnsettled):
        await _settle_tool_flights(_EverythingBroken())


async def test_a_fence_that_cannot_wait_out_held_leases_aborts_extraction():
    """[codex r4] The await half of the fence carries the same authority as the revoke half."""
    from message_processor.handlers.text import _settle_tool_flights
    from message_processor.turn_runtime import TurnEffectsUnsettled

    class _UnwaitableLeases(TurnRuntime):
        async def await_tool_flights(self):
            raise RuntimeError("the flight table is broken")

        def wait_for_effects(self):
            raise RuntimeError("and the leases cannot be counted")

    with pytest.raises(TurnEffectsUnsettled):
        await _settle_tool_flights(_UnwaitableLeases())


@pytest.mark.parametrize("answer", [None, "not a flight"])
async def test_a_turn_that_cannot_mint_an_anonymous_flight_runs_nothing(answer):
    """The PRESENCE of the lifecycle API is the promise. A turn that has it and answers with
    something that is not a flight has failed at it exactly as a raise does — and the legacy path
    would run the id-less call with nothing to drain, cancel or revoke it."""
    ran = []

    class _Malformed(TurnRuntime):
        def open_anonymous_tool_flight(self, *, tool_name, timeout):
            return answer

    async def _executor(ctx, args):
        ran.append(True)
        return {"ok": True}

    turn = _Malformed()
    result = await _registry(_executor).dispatch(_ctx(turn), "probe", {}, call_id=None)

    assert result["error"] == "flight_unavailable"
    assert ran == [], "nothing ran untracked"
    assert turn.pending_tool_flights == []


async def test_an_id_less_sibling_cannot_effect_after_the_ledger_has_settled():
    """Codex's case, end to end. One sibling declines to post (a stale suppression, which
    `gather` propagates out of the round) while an id-less sibling is still short of its lease.

    Untracked, that sibling was invisible to the finalizer: nothing drained it, nothing cancelled
    it, nothing revoked it, and when it finally took its lease the receipts had settled — a
    message in a thread with `note_post` refusing to claim it. Owned by an anonymous flight, the
    same run ends with the effect REFUSED before it happens.
    """
    from message_processor.stale_send_guard import StaleSendSuppressed
    from message_processor.turn_runtime import run_effect

    order, stuck = [], asyncio.Event()
    turn = TurnRuntime()

    async def _guarded(ctx, args):
        raise StaleSendSuppressed(surface="post_to_thread", last_seen_ts="10.0",
                                  observed_latest_ts="11.0")

    async def _id_less(ctx, args):
        try:
            await asyncio.sleep(30)          # pre-lease when the round blows up
        except asyncio.CancelledError:
            # Outlives the cancel grace, but never past this test: a sibling that could suppress
            # cancellation forever would hang the loop's teardown on any failure above.
            try:
                await asyncio.wait_for(stuck.wait(), timeout=5)
            except asyncio.TimeoutError:
                pass
        try:
            await run_effect(turn, "post_to_thread", _post)
        except EffectRevoked:
            order.append("effect refused")
        return {"ok": True}

    async def _post():
        order.append("POSTED AFTER SETTLEMENT")
        return "900.0"

    reg = ToolRegistry()
    reg.register({"type": "function", "name": "guarded", "parameters": {}}, _guarded)
    reg.register({"type": "function", "name": "id_less", "parameters": {}}, _id_less,
                 timeout=0.02)

    with pytest.raises(StaleSendSuppressed):
        await reg.dispatch_all(_ctx(turn), [_call(name="guarded", call_id="c1"),
                                            _call(name="id_less", call_id=None)])

    # The sibling `gather` left running is the turn's, keyed where no lookup can reach it.
    assert [f.tool_name for f in turn.pending_tool_flights] == ["id_less"]
    assert [k for k in turn._tool_flights if k[1] == ("anonymous", 1)]

    survivors = await turn.finish_tool_flights(grace=0.05)
    assert survivors == ("id_less",)
    await outbound_receipts.settle_ledger(_StubLedger(order), turn=turn)

    stuck.set()
    await asyncio.gather(*(f.task for f in turn._tool_flights.values() if f.task is not None),
                         return_exceptions=True)

    assert order == ["settled", "effect refused"], "the effect never happened, before or after"
