"""F7 provenance on the bot's own IMAGE message.

The bot's text reply has always carried a provenance row keyed on the reply's ts; the image
never did, because files_upload_v2 returns no share ts. On a silent image turn (model calls
generate_image, says nothing) that left the image with no record of the tool that made it,
and the model would later deny its own verified tool use.

All stubbed I/O: no live Slack, no real waiting (the `clock` fixture below owns time wherever
a poll loop runs — the resolver's real bound is 15s).
"""
import asyncio
import time as real_time
from unittest.mock import AsyncMock, MagicMock

import pytest
from slack_sdk.errors import SlackApiError

from config import config
from message_processor import image_delivery
from message_processor.utilities import MessageUtilitiesMixin
from slack_client import messaging as messaging_module
from slack_client.messaging import SlackMessagingMixin


# --------------------------------------------------------------------------- helpers

class _Messaging(SlackMessagingMixin):
    """The mixin with only what these two methods touch: an app client and the loggers."""
    def __init__(self, files_info=None, upload_result=None):
        self.app = MagicMock()
        self.app.client.files_info = files_info or AsyncMock()
        self.app.client.files_upload_v2 = AsyncMock(return_value=upload_result)

    def log_info(self, *a, **k): pass
    def log_debug(self, *a, **k): pass
    def log_warning(self, *a, **k): pass
    def log_error(self, *a, **k): pass


class _FakeTime:
    """Stands in for the `time` module inside slack_client.messaging ONLY — patching the real
    time.monotonic would drag the event loop's own clock along with it. Everything other than
    monotonic proxies through, so the module keeps working."""
    def __init__(self, state):
        self._state = state

    def monotonic(self):
        return self._state["now"]

    def __getattr__(self, name):
        return getattr(real_time, name)


@pytest.fixture
def clock(monkeypatch):
    """A clock the test owns: sleeping ADVANCES time instead of spending it, so a test can
    exhaust the resolver's real 15s budget instantly. Yields the list of delays the resolver
    asked for, which is what the poll schedule is asserted against."""
    state = {"now": 0.0}
    slept = []

    async def fake_sleep(seconds):
        slept.append(seconds)
        state["now"] += seconds

    monkeypatch.setattr(messaging_module, "time", _FakeTime(state))
    monkeypatch.setattr("asyncio.sleep", fake_sleep)
    return slept


def _shares(scope, channel_id="C1", ts="1700000000.000100"):
    return {"file": {"shares": {scope: {channel_id: [{"ts": ts}]}}}}


def _rate_limited(retry_after=None):
    """Slack's 429 as the SDK raises it: `error` of "ratelimited", plus the Retry-After header
    it usually (not always) sends."""
    response = MagicMock()
    response.get = lambda key, default=None: {"error": "ratelimited"}.get(key, default)
    response.status_code = 429
    response.headers = {"Retry-After": str(retry_after)} if retry_after is not None else {}
    return SlackApiError("ratelimited", response)


class _ImageData:
    format = "png"
    prompt = ""

    def to_bytes(self):
        return b"\x89PNG"


class _Processor:
    """Collects what publish_image schedules instead of running it, so tests drive the
    detached coroutine themselves (no event-loop races, no orphaned tasks)."""
    def __init__(self):
        self.scheduled = []
        self.persisted = []

    def _schedule_async_call(self, coro):
        self.scheduled.append(coro)
        return coro

    def _persist_tool_provenance(self, channel_id, message_ts, thread_key, provenance):
        self.persisted.append((channel_id, message_ts, thread_key, provenance))

    async def update_last_image_url(self, *a, **k): pass

    def log_debug(self, *a, **k): pass
    def log_info(self, *a, **k): pass
    def log_error(self, *a, **k): pass


class _Checklist:
    """Only what publish_image touches. ``order`` is shared with the fake resolver so a test
    can assert WHEN the indicator came down relative to the share landing — that ordering is
    the entire point of the hold."""
    def __init__(self, order=None, surface="assistant_status"):
        self.surface = surface
        self.completed = False
        self.delete_after = None
        self.order = order if order is not None else []

    async def step(self, active_text, done_text=None):
        self.order.append(f"step:{active_text}")

    async def complete(self, final_text=None, delete_after=None):
        self.completed = True
        self.delete_after = delete_after
        self.order.append("indicator down")


async def _publish(processor, client, provenance_tool=None, checklist=None):
    return await image_delivery.publish_image(
        processor=processor, client=client, channel_id="C1", thread_id="1.0",
        thread_key="C1:1.0", image_data=_ImageData(), checklist=checklist, generation_id=None,
        prompt="a cat", db=None, thread_manager=MagicMock(), unprompted=False,
        provenance_tool=provenance_tool)


def _client(resolve=None, file_id="F123"):
    c = MagicMock()

    async def send_image(channel_id, thread_id, data, filename, caption, meta_out=None):
        if meta_out is not None and file_id is not None:
            meta_out["file_id"] = file_id
        return "https://files.slack.com/img.png"

    c.send_image = send_image
    if resolve is None:
        del c.resolve_file_share_ts  # a client that cannot resolve (non-Slack / test double)
    else:
        c.resolve_file_share_ts = resolve
    return c


# --------------------------------------------------------------------------- resolve_file_share_ts

class TestResolveFileShareTs:
    @pytest.mark.asyncio
    async def test_hit_under_public_scope(self):
        m = _Messaging(files_info=AsyncMock(return_value=_shares("public")))
        assert await m.resolve_file_share_ts("C1", "F1") == "1700000000.000100"

    @pytest.mark.asyncio
    async def test_hit_under_private_scope(self):
        """Private channels AND DMs both report under `private` — measured live."""
        m = _Messaging(files_info=AsyncMock(return_value=_shares("private", "D08EDPS3QMC")))
        assert await m.resolve_file_share_ts("D08EDPS3QMC", "F1") == "1700000000.000100"

    @pytest.mark.asyncio
    async def test_share_appearing_on_a_later_poll_is_found(self, monkeypatch):
        """`shares` is {} at upload time and filled asynchronously — the whole reason this
        polls at all. Also pins the mild backoff: a slow DM must not cost ~30 calls."""
        sleeps = []

        async def fake_sleep(d):
            sleeps.append(d)

        monkeypatch.setattr("asyncio.sleep", fake_sleep)
        empty = {"file": {"shares": {}}}
        m = _Messaging(files_info=AsyncMock(
            side_effect=[empty, empty, empty, empty, _shares("private")]))
        assert await m.resolve_file_share_ts("C1", "F1") == "1700000000.000100"
        assert sleeps == [0.5, 0.5, 1.0, 1.0]

    @pytest.mark.asyncio
    async def test_timeout_returns_none(self, monkeypatch):
        """A zero budget still polls once, then gives up without sleeping: the share is often
        already there, so the first poll is always worth making."""
        monkeypatch.setattr(config, "image_share_ts_timeout_seconds", 0.0)
        info = AsyncMock(return_value={"file": {"shares": {}}})
        m = _Messaging(files_info=info)
        assert await m.resolve_file_share_ts("C1", "F1") is None
        assert info.await_count == 1

    @pytest.mark.asyncio
    async def test_a_failing_lookup_returns_none_never_raises(self, clock):
        m = _Messaging(files_info=AsyncMock(
            side_effect=SlackApiError("nope", {"error": "file_not_found"})))
        assert await m.resolve_file_share_ts("C1", "F1") is None

    @pytest.mark.asyncio
    async def test_unexpected_error_returns_none_never_raises(self, clock):
        m = _Messaging(files_info=AsyncMock(side_effect=RuntimeError("transport blew up")))
        assert await m.resolve_file_share_ts("C1", "F1") is None

    @pytest.mark.asyncio
    async def test_a_rate_limit_is_retried_not_surrendered(self, clock):
        """A 429 is not an answer. Bailing on the first one silently costs the row that the
        very next poll would have had — and provenance lost here is never recomputed."""
        info = AsyncMock(side_effect=[_rate_limited(), _shares("private")])
        m = _Messaging(files_info=info)
        assert await m.resolve_file_share_ts("C1", "F1") == "1700000000.000100"
        assert info.await_count == 2

    @pytest.mark.asyncio
    async def test_retry_after_outranks_the_backoff(self, clock):
        """Slack said when to come back; coming back sooner just earns another 429."""
        info = AsyncMock(side_effect=[_rate_limited(retry_after=3), _shares("private")])
        m = _Messaging(files_info=info)
        assert await m.resolve_file_share_ts("C1", "F1") == "1700000000.000100"
        assert clock == [3.0]  # not the 0.5s the backoff schedule wanted

    @pytest.mark.asyncio
    async def test_retry_after_is_clamped_to_the_budget(self, clock, monkeypatch):
        """A Retry-After longer than the whole budget must not outlive it."""
        monkeypatch.setattr(config, "image_share_ts_timeout_seconds", 2.0)
        m = _Messaging(files_info=AsyncMock(side_effect=_rate_limited(retry_after=600)))
        assert await m.resolve_file_share_ts("C1", "F1") is None
        assert clock == [2.0]

    @pytest.mark.asyncio
    async def test_file_not_found_is_transient_here(self, clock):
        """It means the upload's own eventual consistency hasn't caught up — the exact race
        this poll exists to paper over, not a verdict that the file isn't real."""
        info = AsyncMock(side_effect=[SlackApiError("nope", {"error": "file_not_found"}),
                                      _shares("private")])
        m = _Messaging(files_info=info)
        assert await m.resolve_file_share_ts("C1", "F1") == "1700000000.000100"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("code", ["missing_scope", "invalid_auth"])
    async def test_a_permanent_error_gives_up_without_burning_the_budget(self, clock, code):
        """Auth and scope failures will not come right inside 15s, so polling on is waste."""
        info = AsyncMock(side_effect=SlackApiError(code, {"error": code}))
        m = _Messaging(files_info=info)
        assert await m.resolve_file_share_ts("C1", "F1") is None
        assert info.await_count == 1
        assert clock == []

    @pytest.mark.asyncio
    async def test_no_request_is_issued_past_the_deadline(self, clock, monkeypatch):
        """The bound is wall-clock: the loop must stop AT it rather than wake up on it and
        poll once more, which is what sleeping to the deadline and re-looping used to do."""
        monkeypatch.setattr(config, "image_share_ts_timeout_seconds", 2.0)
        info = AsyncMock(return_value={"file": {"shares": {}}})
        m = _Messaging(files_info=info)
        assert await m.resolve_file_share_ts("C1", "F1") is None
        # 2s of budget buys polls at t=0, 0.5 and 1.0; t=2.0 is the wall, so there is no 4th.
        assert clock == [0.5, 0.5, 1.0]
        assert info.await_count == 3

    @pytest.mark.asyncio
    async def test_a_hung_request_cannot_outlive_the_budget(self, monkeypatch):
        """Each call is bounded by what's left of the budget. Without that, slack_sdk's own
        default timeout is the only ceiling and one hung files.info sails past ours.

        Real clock on purpose (the budget is 0.05s) — a faked one can't catch a real await.
        """
        monkeypatch.setattr(config, "image_share_ts_timeout_seconds", 0.05)

        async def never_returns(**kwargs):
            await asyncio.sleep(30)

        m = _Messaging(files_info=AsyncMock(side_effect=never_returns))
        started = real_time.monotonic()
        assert await m.resolve_file_share_ts("C1", "F1") is None
        assert real_time.monotonic() - started < 5


# --------------------------------------------------------------------------- send_image meta_out

class TestSendImageMetaOut:
    @pytest.mark.asyncio
    async def test_meta_out_receives_the_file_id(self):
        m = _Messaging(upload_result={"files": [{"id": "F123", "url_private": "u"}]})
        meta = {}
        assert await m.send_image("C1", "1.0", b"x", "a.png", "", meta_out=meta) == "u"
        assert meta["file_id"] == "F123"

    @pytest.mark.asyncio
    async def test_without_meta_out_the_upload_still_works(self):
        """The return contract is the URL and nothing else — every legacy caller omits meta_out."""
        m = _Messaging(upload_result={"files": [{"id": "F123", "url_private": "u"}]})
        assert await m.send_image("C1", "1.0", b"x", "a.png") == "u"


# --------------------------------------------------------------------------- publish_image

class TestPublishImageProvenance:
    @pytest.mark.asyncio
    async def test_row_is_keyed_on_the_resolved_share_ts(self):
        proc = _Processor()
        client = _client(resolve=AsyncMock(return_value="1700000000.000100"))
        assert await _publish(proc, client, provenance_tool="generate_image")
        await proc.scheduled[0]
        channel_id, ts, thread_key, provenance = proc.persisted[0]
        # The IMAGE message's own ts — not the file id, not the trigger ts.
        assert (channel_id, ts, thread_key) == ("C1", "1700000000.000100", "C1:1.0")
        assert provenance == [{"tool_name": "generate_image", "gist": ""}]

    @pytest.mark.asyncio
    async def test_no_provenance_tool_never_resolves(self):
        """The legacy classifier-routed path: no tool ran, so attributing one would be a lie."""
        proc = _Processor()
        resolve = AsyncMock(return_value="1.1")
        assert await _publish(proc, _client(resolve=resolve), provenance_tool=None)
        resolve.assert_not_awaited()
        assert proc.scheduled == []

    @pytest.mark.asyncio
    async def test_provenance_disabled_costs_zero_api_calls(self, monkeypatch):
        monkeypatch.setattr(config, "enable_tool_provenance", False)
        proc = _Processor()
        resolve = AsyncMock(return_value="1.1")
        assert await _publish(proc, _client(resolve=resolve), provenance_tool="generate_image")
        resolve.assert_not_awaited()
        assert proc.scheduled == []

    @pytest.mark.asyncio
    async def test_client_that_cannot_resolve_is_skipped(self):
        proc = _Processor()
        assert await _publish(proc, _client(resolve=None), provenance_tool="generate_image")
        assert proc.scheduled == []

    @pytest.mark.asyncio
    async def test_unresolved_ts_writes_no_row_and_does_not_raise(self):
        proc = _Processor()
        client = _client(resolve=AsyncMock(return_value=None))
        assert await _publish(proc, client, provenance_tool="generate_image")
        await proc.scheduled[0]
        assert proc.persisted == []

    @pytest.mark.asyncio
    async def test_upload_with_no_file_id_is_skipped(self):
        proc = _Processor()
        resolve = AsyncMock(return_value="1.1")
        client = _client(resolve=resolve, file_id=None)
        assert await _publish(proc, client, provenance_tool="generate_image")
        resolve.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_raising_resolver_never_un_posts_the_image(self):
        """The image is already in the thread by the time any of this runs."""
        proc = _Processor()
        client = _client(resolve=AsyncMock(side_effect=RuntimeError("slack down")))
        url = await _publish(proc, client, provenance_tool="generate_image")
        assert url == "https://files.slack.com/img.png"
        with pytest.raises(RuntimeError):
            await proc.scheduled[0]  # surfaces to _schedule_async_call's logger, not the caller
        assert proc.persisted == []

    @pytest.mark.asyncio
    async def test_a_scheduler_that_refuses_leaves_no_un_awaited_coroutine(self):
        """The coroutine is built before it is handed over, so a scheduler that raises must
        close it — otherwise it dies at GC with a "never awaited" warning from nowhere."""
        proc = _Processor()
        proc._schedule_async_call = MagicMock(side_effect=RuntimeError("no loop"))
        assert await _publish(proc, _client(resolve=AsyncMock(return_value="1.1")),
                              provenance_tool="generate_image")
        coro = proc._schedule_async_call.call_args[0][0]
        with pytest.raises(RuntimeError, match="cannot reuse"):
            await coro  # already closed; a live coroutine would run here instead


# --------------------------------------------------------------------------- indicator hold

class TestIndicatorHold:
    """The "Uploading…" indicator must outlive the upload call.

    files_upload_v2 returns ~0.25s but the image only becomes visible ~2s later, so completing
    the checklist on the upload left the user watching every progress indicator disappear with
    no image on screen yet. The share record is what makes it visible, so the indicator waits
    on that instead of on a hardcoded cushion.
    """

    @pytest.mark.asyncio
    async def test_the_indicator_stays_up_until_the_share_lands(self, monkeypatch):
        """The ordering IS the fix: the upload has already returned by this point, and
        completing here is exactly the gap being closed."""
        monkeypatch.setattr(config, "image_indicator_hold_seconds", 5.0)
        order = []
        started = asyncio.Event()
        gate = asyncio.Event()

        async def resolve(channel_id, file_id):
            started.set()
            await gate.wait()
            order.append("image visible")
            return "1700000000.000100"

        proc = _Processor()
        chk = _Checklist(order)
        pub = asyncio.create_task(
            _publish(proc, _client(resolve=AsyncMock(side_effect=resolve)),
                     provenance_tool="generate_image", checklist=chk))
        await asyncio.wait_for(started.wait(), timeout=1)  # we are now inside the hold
        assert not chk.completed, "indicator came down before the image was visible"

        gate.set()
        assert await asyncio.wait_for(pub, timeout=1)
        assert order == ["step:Uploading…", "image visible", "indicator down"]
        await proc.scheduled[0]  # drive the detached row so it isn't GC'd un-awaited

    @pytest.mark.asyncio
    async def test_the_hold_expires_without_killing_the_resolve(self, monkeypatch):
        """The indicator's bound is its own: it stops WATCHING, it does not cancel the poll.
        Provenance is invisible, keeps the longer budget, and must still get its row.

        Real clock on purpose (the bound under test is 0.05s) — a faked one can't catch a real
        await sailing past it.
        """
        monkeypatch.setattr(config, "image_indicator_hold_seconds", 0.05)

        async def slow_resolve(channel_id, file_id):
            await asyncio.sleep(0.3)  # outlives the hold, not the 15s provenance budget
            return "1700000000.000100"

        proc = _Processor()
        chk = _Checklist()
        began = real_time.monotonic()
        assert await _publish(proc, _client(resolve=AsyncMock(side_effect=slow_resolve)),
                              provenance_tool="generate_image", checklist=chk)
        # Gave up on the indicator rather than hang it on a signal that never came...
        assert real_time.monotonic() - began < 0.25
        assert chk.completed
        # ...but the resolve lived on (shielded), so provenance still lands.
        await proc.scheduled[0]
        assert proc.persisted[0][1] == "1700000000.000100"

    @pytest.mark.asyncio
    async def test_one_resolve_answers_both_consumers(self):
        """The indicator and provenance ask the same question; asking Slack twice would double
        the polling to learn the same fact."""
        resolve = AsyncMock(return_value="1700000000.000100")
        proc = _Processor()
        chk = _Checklist()
        assert await _publish(proc, _client(resolve=resolve),
                              provenance_tool="generate_image", checklist=chk)
        await proc.scheduled[0]
        assert resolve.await_count == 1
        assert chk.completed and proc.persisted

    @pytest.mark.asyncio
    async def test_the_indicator_holds_even_with_provenance_off(self, monkeypatch):
        """The gap is a UX bug, not a provenance feature — turning F7 off must not reopen it."""
        monkeypatch.setattr(config, "enable_tool_provenance", False)
        resolve = AsyncMock(return_value="1700000000.000100")
        proc = _Processor()
        chk = _Checklist()
        assert await _publish(proc, _client(resolve=resolve),
                              provenance_tool="generate_image", checklist=chk)
        resolve.assert_awaited_once()  # the indicator still needed the answer
        assert chk.completed
        assert proc.scheduled == []  # but no row was written

    @pytest.mark.asyncio
    async def test_a_client_that_cannot_resolve_still_completes_the_indicator(self):
        """Non-Slack clients have no resolve_file_share_ts. The indicator must not wait on a
        signal that will never arrive — a stuck spinner is worse than the gap."""
        proc = _Processor()
        chk = _Checklist()
        assert await _publish(proc, _client(resolve=None),
                              provenance_tool="generate_image", checklist=chk)
        assert chk.completed

    @pytest.mark.asyncio
    async def test_a_checklist_with_no_surface_is_never_held(self, monkeypatch):
        """Nothing is on screen to keep up, so there is nothing worth delaying delivery for."""
        monkeypatch.setattr(config, "image_indicator_hold_seconds", 30.0)

        async def slow_resolve(channel_id, file_id):
            await asyncio.sleep(0.3)
            return "1700000000.000100"

        proc = _Processor()
        chk = _Checklist(surface="none")
        began = real_time.monotonic()
        assert await _publish(proc, _client(resolve=AsyncMock(side_effect=slow_resolve)),
                              provenance_tool="generate_image", checklist=chk)
        assert real_time.monotonic() - began < 0.25
        await proc.scheduled[0]

    @pytest.mark.asyncio
    async def test_a_cancel_during_completion_does_not_orphan_the_resolve(self, monkeypatch):
        """Shutdown landing in complete() must still hand the resolve over to someone.

        Nothing else can stop it: it is shielded, so the hold's own cancellation path won't
        reach it, and a poll left running with no owner is exactly what the shield makes
        possible.
        """
        monkeypatch.setattr(config, "enable_tool_provenance", False)  # → the cancel branch
        # The hold must EXPIRE first, so the resolve is still in flight when the cancel lands —
        # a resolve that already finished would be "not orphaned" for the wrong reason.
        monkeypatch.setattr(config, "image_indicator_hold_seconds", 0.05)
        resolve_task = {}

        async def resolve(channel_id, file_id):
            await asyncio.sleep(5)
            return "1700000000.000100"

        class _CancellingChecklist(_Checklist):
            async def complete(self, final_text=None, delete_after=None):
                raise asyncio.CancelledError()

        proc = _Processor()
        client = _client(resolve=AsyncMock(side_effect=resolve))
        real_start = image_delivery._start_share_resolve

        def spy(*a, **k):
            resolve_task["task"] = real_start(*a, **k)
            return resolve_task["task"]

        monkeypatch.setattr(image_delivery, "_start_share_resolve", spy)
        with pytest.raises(asyncio.CancelledError):
            await _publish(proc, client, provenance_tool="generate_image",
                           checklist=_CancellingChecklist())
        assert not resolve_task["task"].done(), "test bug: the resolve was never left in flight"
        await asyncio.sleep(0.05)  # let the cancellation actually land
        assert resolve_task["task"].cancelled(), \
            "the shielded resolve was left running with nobody to own or stop it"

    @pytest.mark.asyncio
    async def test_the_sandbox_rescue_path_has_no_indicator_to_hold(self, monkeypatch):
        """main.py's F38 rescue posts with checklist=None. It must not inherit the wait."""
        monkeypatch.setattr(config, "image_indicator_hold_seconds", 30.0)

        async def slow_resolve(channel_id, file_id):
            await asyncio.sleep(0.3)
            return "1700000000.000100"

        proc = _Processor()
        began = real_time.monotonic()
        assert await _publish(proc, _client(resolve=AsyncMock(side_effect=slow_resolve)),
                              provenance_tool="create_image_asset")
        assert real_time.monotonic() - began < 0.25
        await proc.scheduled[0]


# --------------------------------------------------------------------------- end to end

class _RealProcessor(MessageUtilitiesMixin):
    """The PRODUCTION scheduler and provenance-persist, over a real temp DB. The _Processor
    fake above proves publish_image's wiring; only this proves the row survives the DB hop and
    comes back the way the rebuild reads it.

    update_last_image_url is the one stub: it is publish_image's warm-state hop, not part of
    the provenance path under test.
    """
    def __init__(self, db):
        self.db = db

    async def update_last_image_url(self, *a, **k): pass

    def log_debug(self, *a, **k): pass
    def log_error(self, *a, **k): pass
    def log_info(self, *a, **k): pass


async def _drain(processor):
    """Await every task the real scheduler spawned, deterministically. One gather isn't
    enough: the resolve task schedules the DB write from INSIDE itself, so the second task
    only exists once the first has run."""
    for _ in range(10):
        pending = [t for t in list(getattr(processor, "_background_tasks", ())) if not t.done()]
        if not pending:
            return
        await asyncio.gather(*pending)
    raise AssertionError("background tasks never settled")


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_DIR", str(tmp_path))
    from database import DatabaseManager
    db = DatabaseManager(platform="slack")
    yield db
    db.conn.close()


class TestImageProvenanceEndToEnd:
    @pytest.mark.asyncio
    async def test_the_row_is_durable_and_reads_back_the_way_the_rebuild_reads_it(self, temp_db):
        """The whole path with only Slack's HTTP faked: upload → file id via meta_out →
        resolved share ts → real _schedule_async_call → real _persist_tool_provenance → real
        sqlite → the exact query thread_management.py makes to render `[used tools: …]`.
        """
        client = _Messaging(
            files_info=AsyncMock(return_value=_shares("private")),
            upload_result={"files": [{"id": "F123", "url_private": "https://slack/img.png"}]})
        proc = _RealProcessor(temp_db)

        assert await _publish(proc, client, provenance_tool="generate_image")
        await _drain(proc)

        # Keyed on the IMAGE message's own ts — the thing this feature exists to find.
        assert await temp_db.get_thread_tool_usage_async("C1:1.0") == {
            "1700000000.000100": [{"tool_name": "generate_image", "gist": ""}]}

    @pytest.mark.asyncio
    async def test_an_unresolvable_ts_leaves_the_db_untouched(self, temp_db):
        """No ts, no key to hang a row on — and a row under a wrong key would teach the model
        it used a tool on some other message."""
        client = _Messaging(
            files_info=AsyncMock(return_value={"file": {"shares": {}}}),
            upload_result={"files": [{"id": "F123", "url_private": "https://slack/img.png"}]})
        proc = _RealProcessor(temp_db)
        # A spent budget keeps it to the single guaranteed poll instead of the real 15s.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(config, "image_share_ts_timeout_seconds", 0.0)
            assert await _publish(proc, client, provenance_tool="generate_image")
            await _drain(proc)

        assert await temp_db.get_thread_tool_usage_async("C1:1.0") == {}
