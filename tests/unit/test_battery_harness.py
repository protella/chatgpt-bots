"""T112–T114, T116–T122, T125–T127 — the live battery's harness, NETWORK-FREE.

THESE LIVE UNDER `tests/unit/` DELIBERATELY. `make test` runs `pytest tests/unit` and nothing
else, so a harness test anywhere else would sit outside the capped gate and rot unnoticed.
Nothing here touches Slack, the bot, or the real database.
"""
import ast
import asyncio
import inspect
import json
import pathlib
import re
import subprocess
import sys
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import pytest
from slack_sdk.errors import SlackApiError
from slack_sdk.http_retry.builtin_async_handlers import AsyncRateLimitErrorRetryHandler

from message_processor.dev_barriers import POST_ADMISSION, POST_PARTIAL_POST
from tests.live import battery_harness as bh
from tests.live import battery_rows as rows
from tests.live import classify_probe
from tests.live import run_battery as runner

OURS = bh.PartyIdentity(bot_id="B_OURS", user_id="U_BOT")
OPERATOR = bh.PartyIdentity(bot_id="", user_id="U_HUMAN")
CLAUDE = bh.PartyIdentity(bot_id="B_CLAUDE", user_id="U_CLAUDE")

# ONE APP OWNS TWO BOT RECORDS. `OURS.bot_id` is the bot-token record that `auth.test` returns;
# `SEED_RECORD` is the user-token posting record, distinguished by having NO user_id, and it is
# the id every seed the harness posts actually carries. Modelling both is the whole point of the
# fixture below — a fake with only one of them cannot express the 2026-08-01 misconfiguration.
OUR_APP_ID = "A_OURS"
SEED_RECORD = "B_OURS_USER"
CLAUDE_APP_ID = "A_CLAUDE"

_BOT_RECORDS = {
    OURS.bot_id: {"id": OURS.bot_id, "app_id": OUR_APP_ID, "user_id": OURS.user_id},
    SEED_RECORD: {"id": SEED_RECORD, "app_id": OUR_APP_ID},          # no user_id — deliberate
    CLAUDE.bot_id: {"id": CLAUDE.bot_id, "app_id": CLAUDE_APP_ID, "user_id": CLAUDE.user_id},
    # A THIRD app whose record carries no bot-USER id. It cannot be mentioned, so it cannot be the
    # party this battery grades, and the preflight has to say so rather than return half a pair.
    "B_MUTE": {"id": "B_MUTE", "app_id": "A_MUTE"},
}


# --------------------------------------------------------------------------------- fake Slack

class _Resp:
    """The `.data` carrier a real `SlackResponse` exposes, and nothing else."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.data = data

    def get(self, key, default=None):
        return self.data.get(key, default)


def slack_error(code: str) -> SlackApiError:
    return SlackApiError(f"the request failed: {code}", response=_Resp({"ok": False,
                                                                       "error": code}))


class FakeSlack:
    """Records every call and answers from `handlers`. An unregistered method is a test bug."""

    def __init__(self, **handlers: Any) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self.handlers: Dict[str, Any] = dict(handlers)
        self.retry_handlers: List[Any] = []

    def method_calls(self, name: str) -> List[Dict[str, Any]]:
        return [kwargs for called, kwargs in self.calls if called == name]

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        async def call(**kwargs: Any) -> Any:
            self.calls.append((name, kwargs))
            handler = self.handlers.get(name)
            if handler is None:
                raise AssertionError(f"unexpected Slack call {name}({kwargs})")
            result = handler(**kwargs) if callable(handler) else handler
            if inspect.isawaitable(result):
                result = await result
            if isinstance(result, BaseException):
                raise result
            return result

        return call


class FakeClock:
    """Time the poll loops read through, so a 180-second deadline costs no wall time."""

    def __init__(self) -> None:
        self.t = 1000.0
        self.sleeps: List[float] = []

    def now(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


@pytest.fixture(autouse=True)
def _no_leaked_cleanups():
    """`run_battery._PENDING_CLEANUPS` is module-global. A task left in it by one test makes the
    NEXT `run_with_drain` wait on something that will never finish — which is exactly how a full
    suite hangs while each file passes on its own."""
    runner._PENDING_CLEANUPS.clear()
    yield
    for task in list(runner._PENDING_CLEANUPS):
        task.cancel()
    runner._PENDING_CLEANUPS.clear()


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(bh, "_now", fake.now)
    monkeypatch.setattr(bh, "_sleep", fake.sleep)
    return fake


@pytest.fixture
def install(monkeypatch):
    """Install a client pair and clear the identity caches around the test."""
    def _install(user=None, bot=None) -> bh.Clients:
        # FakeSlack stands in for the AsyncWebClient the real pair carries.
        pair = bh.Clients(user=cast(Any, user or FakeSlack()),
                          bot=cast(Any, bot or user or FakeSlack()))
        monkeypatch.setattr(bh, "_clients", pair)
        monkeypatch.setattr(bh, "_bot_auth_data", None)
        monkeypatch.setattr(bh, "_harness_user_identity", None)
        return pair
    return _install


def replies_page(messages: List[Dict[str, Any]], cursor: Optional[str] = None) -> Dict[str, Any]:
    page: Dict[str, Any] = {"ok": True, "messages": messages}
    if cursor:
        page["response_metadata"] = {"next_cursor": cursor}
    return page


def bot_msg(ts: str, text: str = "hi", **extra: Any) -> Dict[str, Any]:
    return {"ts": ts, "text": text, "user": OURS.user_id, "bot_id": OURS.bot_id, **extra}


# ------------------------------------------------------------------------------------- T112

async def test_the_poller_raise_taxonomy(clock, install):
    """T112. A battery must die loudly rather than report a false negative.

    An invalid token, a missing scope or a malformed response read as "the bot didn't answer" if
    the poller swallowed them, and the battery would report a behavioural failure that is really
    an operator error. Every condition of §7.1a's table, each with the Slack code attached.
    """
    assert (bh.REPLY_DEADLINE_SECONDS, bh.REPLY_POLL_SECONDS) == (180.0, 5.0)
    assert (bh.SLOW_TURN_DEADLINE_SECONDS, bh.SEED_PACE_SECONDS) == (200.0, 1.0)

    for code, expected in (("invalid_auth", bh.HarnessAuthError),
                           ("not_authed", bh.HarnessAuthError),
                           ("account_inactive", bh.HarnessAuthError),
                           ("missing_scope", bh.HarnessScopeError),
                           ("not_in_channel", bh.HarnessScopeError),
                           ("channel_not_found", bh.HarnessScopeError),
                           ("fatal_error", bh.HarnessApiError),
                           ("ratelimited", bh.HarnessApiError)):
        install(bot=FakeSlack(conversations_replies=slack_error(code)))
        with pytest.raises(expected) as caught:
            await bh.wait_bot_reply("C1", "1.1", "1.1", author=OURS, deadline=0)
        assert caught.value.code == code

    for malformed in ({"ok": True, "messages": "not a list"},
                      # A message with no ts would coerce to "" and fall out of every `_newer`
                      # compare — the message would VANISH and the row would report silence.
                      {"ok": True, "messages": [{"text": "no ts here"}]},
                      # Malformed cursor metadata read as end-of-walk is a silent truncation.
                      {"ok": True, "messages": [], "response_metadata": "not an object"},
                      {"ok": True, "messages": [], "response_metadata": {"next_cursor": 7}}):
        install(bot=FakeSlack(conversations_replies=malformed))
        with pytest.raises(bh.HarnessProtocolError):
            await bh.wait_bot_reply("C1", "1.1", "1.1", author=OURS, deadline=0)

    # THE ONLY SILENT OUTCOME: Slack answered normally and there genuinely was nothing.
    install(bot=FakeSlack(conversations_replies=replies_page([])))
    assert await bh.wait_bot_reply("C1", "1.1", "1.1", author=OURS, deadline=0) == []


# ------------------------------------------------------------------------------------- T113

async def _stub_row(name: str, run) -> rows.BatteryRow:
    return rows.BatteryRow(name=name, trigger_template="t", assertions=("a",), run=run)


async def test_the_restore_runs_even_when_a_row_fails(clock, install, monkeypatch, tmp_path):
    """T113. A failed assertion is the interesting news; losing it to a restore error would be
    the harness hiding the result it exists to produce.

    AND THE ROW STILL LISTS EVERY TS IT PUT IN THE ROOM. Nothing is deleted (owner ruling), so the
    report's `seeded_ts` is the only index a reader has back to the messages a run posted."""
    install()
    applied: List[str] = []
    real_apply = bh.apply_restore

    def _apply(restore, channel, team_id=""):
        applied.append(f"{restore.kind}:{restore.key}")

    monkeypatch.setattr(bh, "apply_restore", _apply)

    async def _explodes(ctx: bh.RowContext) -> None:
        ctx.seeded_ts.extend(["1.1", "1.2", "1.3"])
        ctx.restores.append(bh.Restore(kind="window_anchor", key="500.0|3",
                                       prior=("100.0", 2), existed=True))
        raise bh.HarnessProtocolError("the row broke mid-flight")

    report = await runner.run_row(await _stub_row("boom", _explodes), "C1")

    assert applied == ["window_anchor:500.0|3"], "the finally never reached the restore"
    assert report["status"] == "error"
    assert report["seeded_ts"] == ["1.1", "1.2", "1.3"]
    assert report["cleanup"]["restored"] == ["window_anchor:500.0|3"]
    assert "the row broke mid-flight" in report["notes"]

    # A restore that cannot be applied is recorded, and a row's own FAILURE outranks it. The REAL
    # `apply_restore` against a database that is not there is the honest way to break one.
    monkeypatch.setattr(bh, "apply_restore", real_apply)
    monkeypatch.setattr(bh, "db_path", lambda: tmp_path / "missing.db")

    async def _fails_an_assertion(ctx: bh.RowContext) -> None:
        ctx.assert_that("the thing held", False)
        ctx.restores.append(bh.Restore(kind="channel_setting", key="model",
                                       prior="gpt-5.6-sol", existed=True))

    failed = await runner.run_row(await _stub_row("nope", _fails_an_assertion), "C1")
    assert failed["status"] == "fail"
    assert failed["cleanup"]["restore_failures"] == ["channel_setting:model"]


def test_the_restore_finishes_during_runner_shutdown(install, monkeypatch):
    """T113, the BaseException half — and it models the REAL teardown.

    A caught exception proves nothing about this guarantee, and neither does a cancellation in a
    loop the test keeps alive afterwards. `asyncio.run` CANCELS every remaining task when it
    closes, so a shielded restore its row abandoned is killed on the way out — at the exact moment
    row 8 has the window anchor sitting on a floor the battery chose.

    So this drives `run_with_drain`, which is the runner's real entry path: it owns the loop, and
    drains the abandoned work while that loop is still alive. SYNCHRONOUS, because
    `run_with_drain` calls `run_until_complete` and would raise inside a running loop.

    IT CANCELS TWICE. One `cancel()` delivers one `CancelledError`, which an ordinary `finally`
    survives — so a single-cancel test cannot tell a shielded restore from an unshielded one. The
    second, landing while the restore is suspended, is what abandons the task and hands it to the
    drain.
    """
    async def _yields(seconds):
        await asyncio.sleep(0)

    monkeypatch.setattr(bh, "_sleep", _yields)
    gate = threading.Event()
    restored: List[str] = []

    def _apply(restore, channel, team_id=""):
        # The FIRST restore blocks in its worker thread, so the second cancel lands while the
        # shielded task is genuinely mid-flight rather than merely scheduled.
        if not restored:
            restored.append(restore.key)
            gate.wait(10)
            return
        restored.append(restore.key)

    monkeypatch.setattr(bh, "apply_restore", _apply)
    install()
    started = asyncio.Event()

    async def _hangs(ctx: bh.RowContext) -> None:
        ctx.restores.extend([bh.Restore(kind="channel_setting", key=key, prior="x", existed=True)
                             for key in ("model", "verbosity", "effort")])
        started.set()
        await asyncio.Event().wait()   # never completes; the driver cancels it

    async def _driver():
        task = asyncio.create_task(runner.run_row(await _stub_row("shutdown", _hangs), "C1"))
        await started.wait()
        task.cancel()                  # first: the row's own await
        for _ in range(200):           # let the `finally` reach the first restore
            if restored:
                break
            await asyncio.sleep(0.01)
        assert restored == ["model"], "the restore did not start from the finally"
        task.cancel()                  # second: abandons the shielded task to the drain
        gate.set()
        await task

    with pytest.raises(asyncio.CancelledError):
        runner.run_with_drain(_driver())

    # The DRAIN finished them, after the run had already unwound. Without it the loop would have
    # closed with the channel still on state the battery wrote.
    assert restored == ["model", "verbosity", "effort"], "the abandoned restore was not drained"
    assert runner._PENDING_CLEANUPS == []


async def test_an_interrupt_during_the_drain_keeps_the_retry_inventory(clock, install,
                                                                      monkeypatch):
    """T113. The inventory IS the retry list, so an entry leaves it only once its task is DONE.

    Clearing the list and then awaiting the copy means an interrupt landing during the gather
    erases the only record of what is still outstanding — the retry has nothing to retry, and the
    channel keeps whatever durable state the battery left on it.
    """
    gate = threading.Event()
    deleted: List[str] = []

    def _apply(restore, channel, team_id=""):
        gate.wait(10)
        deleted.append(restore.key)

    monkeypatch.setattr(bh, "apply_restore", _apply)
    install()
    ctx = bh.RowContext(row="r", nonce="n", channel="C1")
    ctx.restores.append(bh.Restore(kind="channel_setting", key="1.1", prior="x", existed=True))

    task = asyncio.ensure_future(bh.cleanup_row(ctx))
    runner._PENDING_CLEANUPS.append(task)
    try:
        drain = asyncio.ensure_future(runner.drain_cleanups())
        for _ in range(10):            # let the drain reach its await
            await asyncio.sleep(0)
        drain.cancel()                 # THE INTERRUPT, landing during the drain
        with pytest.raises(asyncio.CancelledError):
            await drain

        assert runner._PENDING_CLEANUPS == [task], "the interrupt erased the retry inventory"
        assert deleted == [], "the shielded cleanup should still be pending"

        gate.set()
        await runner.drain_cleanups()  # the retry finishes what the interrupt left
        assert deleted == ["1.1"]
        assert runner._PENDING_CLEANUPS == []
    finally:
        runner._PENDING_CLEANUPS.clear()
        gate.set()
        if not task.done():
            await task


_SIGINT_SCENARIO = r"""
import asyncio, os, signal, sys, threading, time
sys.path.insert(0, {root!r})
from tests.live import battery_harness as bh
from tests.live import run_battery as runner
from tests.live.battery_rows import BatteryRow

DONE = []
SIGNALS = {signals!r}
RELEASE = [0.0]


def _apply(restore, channel, team_id=""):
    # Blocks in its worker thread until RELEASE, so the interrupts land while the shielded
    # restore is genuinely in flight.
    while time.monotonic() < RELEASE[0]:
        time.sleep(0.01)
    DONE.append(restore.key)


async def _yields(seconds):
    await asyncio.sleep(0)


async def _row(ctx):
    ctx.restores.extend([bh.Restore(kind="channel_setting", key=k, prior="x", existed=True)
                         for k in ("1.1", "1.2")])
    RELEASE[0] = time.monotonic() + {hold!r}
    for delay in SIGNALS:
        threading.Timer(delay, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    await asyncio.sleep(30)


async def _driver():
    row = BatteryRow(name="sig", trigger_template="t", assertions=("a",), run=_row)
    return await runner.run_row(row, "C1")


bh.apply_restore = _apply
bh._sleep = _yields
try:
    runner.run_with_drain(_driver())
except BaseException as e:
    print("RAISED", type(e).__name__)
print("RESTORED", ",".join(DONE))
print("PENDING", len(runner._PENDING_CLEANUPS))
"""


def _run_sigint_scenario(tmp_path, signals, hold):
    """Drive the real signal path in a CHILD PROCESS.

    Raising SIGINT inside the shared pytest process is not a test, it is a hazard: the signal is
    process-wide, its disposition depends on whatever else touched the handler, and a timer that
    fires a millisecond after the test ends lands in pytest's own machinery. An earlier version of
    these two tests did exactly that and hung the full suite while passing on its own. The child
    keeps the real semantics — a genuine SIGINT into a genuine `asyncio.Runner` — and cannot take
    the suite with it.
    """
    script = tmp_path / "sigint_scenario.py"
    script.write_text(_SIGINT_SCENARIO.format(
        root=str(pathlib.Path(__file__).resolve().parents[2]), signals=signals, hold=hold),
        encoding="utf-8")
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                          timeout=120)
    out = proc.stdout
    restored = next((line.split(" ", 1)[1] for line in out.splitlines()
                     if line.startswith("RESTORED")), "")
    pending = next((line.split(" ", 1)[1] for line in out.splitlines()
                    if line.startswith("PENDING")), "?")
    return [t for t in restored.split(",") if t], pending, out, proc.stderr


def test_a_real_sigint_still_reaches_the_rows_finally(tmp_path):
    """T113. `asyncio.Runner`, not a bare `loop.run_until_complete` — and a REAL signal proves it.

    The Runner reproduces `asyncio.run`'s first-SIGINT behaviour: the interrupt CANCELS THE MAIN
    TASK. A bare `run_until_complete` lets the `KeyboardInterrupt` unwind the loop out from under
    the running task instead, so `run_row`'s `finally` never executes, nothing is registered, and
    the drain has nothing to drain.

    Delivered while the loop is IDLE, which is where the two differ: a signal raised inline
    propagates out of the coroutine either way and proves nothing.
    """
    restored, pending, out, err = _run_sigint_scenario(tmp_path, (0.10,), 1.0)
    assert restored == ["1.1", "1.2"], f"the row's finally never registered it\n{out}{err}"
    assert pending == "0", out


def test_a_second_interrupt_during_the_drain_still_finishes_the_restore(tmp_path):
    """T113. THE COMPOSED GUARANTEE — the halves passing separately is not enough.

    Three real signals: one during the ROW, one abandoning the shielded restore to the drain, and
    one while the runner's OWN drain is blocked on it. Removing `run_with_drain`'s retry loop
    fails this and nothing else.
    """
    restored, pending, out, err = _run_sigint_scenario(tmp_path, (0.10, 0.45, 0.80), 1.2)
    assert restored == ["1.1", "1.2"], f"the retry after the second interrupt never finished\n{out}{err}"
    assert pending == "0", out


async def test_a_broken_cleanup_never_reports_pass(clock, install, monkeypatch):
    """T113. `status_for` grades what cleanup MEASURED; a cleanup that never finished measured
    nothing, so passing assertions prove the row worked, not that it left nothing behind."""
    install()

    async def _boom(ctx):
        raise RuntimeError("cleanup itself broke")

    monkeypatch.setattr(runner, "cleanup_row", _boom)

    async def _passes(ctx: bh.RowContext) -> None:
        ctx.assert_that("everything held", True)

    report = await runner.run_row(await _stub_row("clean-row", _passes), "C1")
    assert report["status"] == "error"
    assert "cleanup raised" in report["notes"]


# ------------------------------------------------------------------------------------- T114

class FakeRowStore:
    """A provenance store whose row APPEARS and then GROWS, which is what the writer really does."""

    def __init__(self, timeline: List[Optional[List[str]]]) -> None:
        self.timeline = timeline
        self.reads = 0

    async def read(self, channel: str, message_ts: str) -> Optional[str]:
        index = min(self.reads, len(self.timeline) - 1)
        self.reads += 1
        names = self.timeline[index]
        if names is None:
            return None
        return json.dumps([{"tool_name": name, "gist": ""} for name in names])


async def _drive_row_one(monkeypatch, provenance: bh.ProvenanceRead) -> bh.RowContext:
    """Row 1 against fixtures: the same provenance read, graded in both directions."""
    counter = {"n": 0}

    async def _post(channel, text, thread_ts=None, ctx=None):
        counter["n"] += 1
        ts = f"10.{counter['n']:04d}"
        if ctx is not None:
            ctx.seeded_ts.append(ts)
        return ts

    ctx = bh.RowContext(row="cross-thread-awareness", nonce="n1", channel="C1", team_id="T1")

    async def _output(team_id, channel, turn_id, *, deadline=0, poll=0, ctx=None):
        # The row's answer is read through the TURN's own receipts, so the fixture answers as the
        # correlated reader does — including recording the surface for cleanup.
        if ctx is not None:
            ctx.observed_ts.append("99.0001")
        # WRITTEN THE WAY A BOT WRITES A NUMBER — with the separator the seeded fact does not
        # have. The row grades digit-normalized, and a verbatim compare fails right here.
        answer = f"{int(bh.digits_of(bh.quantity('n1', 'crates', digits=4))):,} crates."
        return [bh.Observed(ts="99.0001", text=answer,
                            thread_ts="10.0003", channel=channel,
                            user=OURS.user_id, bot_id=OURS.bot_id)]

    monkeypatch.setattr(rows, "post_seed", _post)
    monkeypatch.setattr(rows, "observe_turn_output", _output)
    monkeypatch.setattr(rows, "bot_identity", lambda: _returns(OURS))
    monkeypatch.setattr(rows, "harness_user_identity", lambda: _returns(OPERATOR))
    monkeypatch.setattr(rows, "find_turn_id", lambda *a, **k: _returns("T1"))
    monkeypatch.setattr(rows, "wait_for_telemetry",
                        lambda *a, **k: _returns({"H": "99.9999"}))
    monkeypatch.setattr(rows, "await_tools_used_for", lambda *a, **k: _returns(provenance))

    await rows.row_cross_thread_awareness(ctx)
    return ctx


async def _returns(value):
    return value


async def test_the_report_schema_is_pinned(clock, install, monkeypatch):
    """T114. The pinned object, the two-halves grading, and the poll contract."""
    install()

    # --- the object, and an observation that cannot change a status -------------------------
    ctx = bh.RowContext(row="value-floor-holds", nonce="n-1", channel="C1")
    ctx.assert_that("the turn stayed quiet", True)
    ctx.observe("tools recorded for the reply", [], False)
    ctx.seeded_ts.append("1.1")
    ctx.observed_ts.append("2.2")
    ctx.external_ts.append("3.3")
    cleanup = bh.CleanupResult(restored=(), restore_failures=())
    report = bh.build_report(ctx, status=bh.status_for(ctx, cleanup), started_at=1.0,
                             finished_at=2.0, cleanup=cleanup)
    assert set(report) >= {"row", "status", "nonce", "started_at", "finished_at", "seeded_ts",
                           "observed_ts", "external_ts", "evidence", "assertions",
                           "observations", "cleanup", "notes"}
    assert set(report["cleanup"]) == {"restored", "restore_failures"}
    assert report["status"] == "pass" and report["status"] in bh.ROW_STATUSES
    assert report["observations"] == [{"name": "tools recorded for the reply", "value": [],
                                       "observed": False}]
    # Everything the row put in the room, by author — the only index back to it, since none of
    # it is ever deleted.
    assert report["seeded_ts"] == ["1.1"] and report["observed_ts"] == ["2.2"]
    assert report["external_ts"] == ["3.3"]

    # --- ONE read, TWO outcomes: an assertion when it finds something, an observation when not
    absent = await _drive_row_one(monkeypatch, bh.ProvenanceRead(row_present=False, names=()))
    assert all(ok for _, ok in absent.assertions)
    assert absent.observations == [("tools recorded for the reply", [], False)]

    named = await _drive_row_one(
        monkeypatch, bh.ProvenanceRead(row_present=True, names=("search_slack",)))
    failing = bh.build_report(named, status=bh.status_for(named, cleanup), started_at=1.0,
                              finished_at=2.0, cleanup=cleanup)
    assert failing["status"] == "fail"

    # --- the poll contract, four terminal states --------------------------------------------
    def _store(timeline):
        store = FakeRowStore(timeline)
        monkeypatch.setattr(bh, "read_tool_provenance_row", store.read)
        return store

    # (a) the partial-row race: a row EXISTS while still missing the awaited name.
    _store([["fetch_channel_info"], ["fetch_channel_info"],
            ["fetch_channel_info", "search_slack"]])
    got = await bh.await_tools_used_for("C1", "9.9", required_name=lambda n: n == "search_slack",
                                        deadline=60, poll=5)
    assert got == bh.ProvenanceRead(True, ("fetch_channel_info", "search_slack"))

    # (b) predicate expiry: present throughout, never matching.
    _store([["fetch_channel_info"]])
    assert await bh.await_tools_used_for("C1", "9.9", required_name=lambda n: n == "search_slack",
                                         deadline=10, poll=5) == \
        bh.ProvenanceRead(True, ("fetch_channel_info",))
    # (c) absent, and (d) present-but-empty — the two states a bare tuple could not tell apart.
    _store([None])
    assert await bh.await_tools_used_for("C1", "9.9", deadline=0) == bh.ProvenanceRead(False, ())
    _store([[]])
    assert await bh.await_tools_used_for("C1", "9.9", deadline=0) == bh.ProvenanceRead(True, ())

    # --- a full run is THIRTEEN objects with no duplicate names ------------------------------
    reports = await _run_all_rows(monkeypatch)
    assert len(reports) == 13
    assert len({r["row"] for r in reports}) == 13


async def _run_all_rows(monkeypatch, run=None) -> List[Dict[str, Any]]:
    async def _noop(ctx: bh.RowContext) -> None:
        ctx.assert_that("ran", True)

    stubs = tuple(rows.BatteryRow(name=row.name, trigger_template=row.trigger_template,
                                 assertions=row.assertions, run=run or _noop)
                  for row in rows.REGISTRY)
    monkeypatch.setattr(runner, "REGISTRY", stubs)
    monkeypatch.setattr(runner, "bot_identity", lambda: _returns(OURS))
    monkeypatch.setattr(runner, "harness_user_identity", lambda: _returns(OPERATOR))
    monkeypatch.setattr(runner, "bot_team_id", lambda: _returns("T_DEV"))
    monkeypatch.setattr(runner, "assert_channel_unfenced", lambda *a: _returns(None))
    monkeypatch.setattr(runner, "assert_claude_tag_allowlisted", lambda: _returns(CLAUDE))
    return await runner.run_battery("C1", [row.name for row in rows.REGISTRY])


# ------------------------------------------------------------------------------------- T116

def _write_ledger(path: Path, events: List[Dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")


async def test_correlation_is_unique_or_it_errors(clock, monkeypatch, tmp_path):
    """T116. Two turns claiming one trigger breaks the row's premise; zero is its own answer."""
    monkeypatch.setattr(bh, "ledger_dir", lambda: tmp_path)
    start = {"event": "turn_start", "channel_id": "C1", "trigger_ts": "5.5", "turn_id": "T-A"}

    _write_ledger(tmp_path / "participation.jsonl", [start])
    assert await bh.find_turn_id("C1", "5.5", deadline=0) == "T-A"

    # ROTATION: a long battery rotates mid-run, and a reader that only opened the live file
    # would lose the rows it is waiting for.
    _write_ledger(tmp_path / "participation.jsonl", [{"event": "turn_outcome"}])
    _write_ledger(tmp_path / "participation.jsonl.1", [start])
    assert await bh.find_turn_id("C1", "5.5", deadline=0) == "T-A"

    # ZERO at the deadline: reported as an error rather than silently waited out.
    _write_ledger(tmp_path / "participation.jsonl.1", [])
    with pytest.raises(bh.HarnessCorrelationError):
        await bh.find_turn_id("C1", "5.5", deadline=10, poll=5)

    # MORE THAN ONE: immediately, without spending a single poll interval.
    _write_ledger(tmp_path / "participation.jsonl",
                  [start, dict(start, turn_id="T-B")])
    before = len(clock.sleeps)
    with pytest.raises(bh.HarnessCorrelationError) as caught:
        await bh.find_turn_id("C1", "5.5", deadline=600)
    assert len(clock.sleeps) == before
    assert "T-A" in str(caught.value) and "T-B" in str(caught.value)


# ------------------------------------------------------------------------------------- T117

async def test_the_barrier_key_is_the_frozen_three_part_form(monkeypatch, tmp_path):
    """T117. `<seam>.<turn_id>.0.release` — the operation id THEN the epoch component.

    A test asserting a two-part `(seam, turn_id)` key would be asserting a contract §3 does not
    have, and a two-part path names a file no waiter is watching.
    """
    monkeypatch.setenv("DEV_TURN_BARRIERS", POST_ADMISSION)
    monkeypatch.setenv("DEV_TURN_BARRIERS_DIR", str(tmp_path))
    monkeypatch.setenv("DEV_TURN_BARRIERS_TIMEOUT", "30")
    monkeypatch.delenv("DEV_TEST_EPOCH_ID", raising=False)

    assert bh.barrier_release_path(POST_ADMISSION, "T-1") == \
        tmp_path / f"{POST_ADMISSION}.T-1.0.release"

    from message_processor import dev_barriers
    task = asyncio.create_task(dev_barriers.post_admission(turn_id="T-1"))
    await bh.wait_barrier_reached(POST_ADMISSION, "T-1", deadline=10, poll=0.02)
    assert bh.barrier_waiting_path(POST_ADMISSION, "T-1").exists()

    bh.release_barrier(POST_ADMISSION, "T-1")
    assert await asyncio.wait_for(task, timeout=10) is True


def test_each_seam_keys_on_what_its_real_call_site_supplies():
    """T117. The two seams key on DIFFERENT things, and a live run proved it the hard way.

    `post_partial_post`'s call site passes `channel_id`/`message_ts`/`owner` and NO `turn_id`, so
    `dev_barriers.operation_id` falls through to the message ts. A row releasing on a turn id
    names a path no waiter is watching, and the frozen turn then holds a message no token can
    delete until its own timeout expires.

    Driven through the REAL `dev_barriers.barrier_key` against the REAL context shapes, so a
    change to either the call sites or the key format moves this test with it.
    """
    from message_processor import dev_barriers

    assert bh.SEAM_KEYED_BY == {POST_ADMISSION: "turn_id", POST_PARTIAL_POST: "message_ts"}

    # The contexts production actually hands each seam.
    admission_ctx = {"turn_id": "abc123:1", "channel_id": "C1", "H": "9.1", "floor_ts": "1.0"}
    partial_ctx = {"channel_id": "C1", "message_ts": "1785579847.267779", "owner": "o"}

    # What the BOT will key on, from its own function...
    assert dev_barriers.barrier_key(POST_ADMISSION, admission_ctx) == "abc123_1.0"
    assert dev_barriers.barrier_key(POST_PARTIAL_POST, partial_ctx) == "1785579847.267779.0"

    # ...and what the HARNESS picks must produce the same key.
    assert bh.barrier_operation(POST_ADMISSION, turn_id="abc123:1") == "abc123:1"
    assert bh.barrier_operation(POST_PARTIAL_POST,
                                message_ts="1785579847.267779") == "1785579847.267779"

    # Supplying the WRONG field for a seam raises rather than building a dead path.
    with pytest.raises(bh.HarnessError, match="keys on message_ts"):
        bh.barrier_operation(POST_PARTIAL_POST, turn_id="abc123:1")
    with pytest.raises(bh.HarnessError, match="keys on turn_id"):
        bh.barrier_operation(POST_ADMISSION, message_ts="1785579847.267779")


def test_a_trigger_verdict_reads_both_ledger_shapes():
    """T114. MEASURED on the live bot: a woken message and a declined one are different rows.

    A declined message emits `gate_start` → `gate_decision` → `visible_action(kind=silence)` and
    **no `turn_start`**. Rows 9a–9d grade "what did this make the bot do", which spans both — and
    reading only `turn_outcome` reports `error` for the very restraint being measured.
    """
    woken = [{"event": "turn_start", "turn_id": "T1"},
             {"event": "stream_render", "turn_id": "T1"},
             {"event": "turn_outcome", "turn_id": "T1", "kind": "reply"}]
    v = bh.classify_trigger(woken)
    assert (v.kind, v.woke, v.turn_id, v.source) == ("reply", True, "T1", "turn_outcome")

    declined = [{"event": "gate_start"}, {"event": "gate_decision"},
                {"event": "visible_action", "kind": "silence"}]
    v = bh.classify_trigger(declined)
    assert (v.kind, v.woke, v.turn_id, v.source) == ("silence", False, None, "visible_action")

    # A gate REACTION is the same shape, and row 9c depends on it.
    v = bh.classify_trigger([{"event": "visible_action", "kind": "reaction_only"}])
    assert (v.kind, v.woke, v.source) == ("reaction_only", False, "visible_action")

    # BOTH present: the turn's own outcome is the more complete account and wins.
    v = bh.classify_trigger([{"event": "turn_start", "turn_id": "T2"},
                             {"event": "visible_action", "kind": "silence"},
                             {"event": "turn_outcome", "turn_id": "T2", "kind": "reaction_only"}])
    assert (v.kind, v.source) == ("reaction_only", "turn_outcome")

    # Nothing judged it at all -> no verdict, which the async wrapper turns into an error.
    assert bh.classify_trigger([{"event": "gate_start"}]) is None


# ------------------------------------------------------------------------------------- T118

def _deep_thread_pages(total: int = 120, per_page: int = 50):
    pages = [[{"ts": f"1.{i:04d}", "text": f"m{i}"} for i in range(start, min(start + per_page,
                                                                             total))]
             for start in range(0, total, per_page)]

    def handler(**kwargs):
        cursor = kwargs.get("cursor")
        index = int(cursor) if cursor else 0
        nxt = str(index + 1) if index + 1 < len(pages) else None
        return replies_page(pages[index], nxt)

    return handler


async def test_pagination_reaches_the_whole_thread(clock, install):
    """T118. A thread of 120 returns 120. One page returns 100 and is a silent truncation —
    exactly what row 5 would otherwise assert against."""
    pair = install(bot=FakeSlack(conversations_replies=_deep_thread_pages(120, 50)))
    walked = await bh.fetch_thread_complete("C1", "1.0000")
    assert len(walked) == 120
    assert len(pair.bot.method_calls("conversations_replies")) == 3

    # A REPEATED cursor is an infinite loop that, from the outside, is indistinguishable from a
    # bot that never answered.
    install(bot=FakeSlack(conversations_replies=replies_page([bot_msg("1.0001")], "stuck")))
    with pytest.raises(bh.HarnessProtocolError, match="not advancing"):
        await bh.fetch_thread_complete("C1", "1.0000")


# ------------------------------------------------------------------------------------- T119

async def test_rate_limit_handling_uses_the_async_handler(clock, monkeypatch):
    """T119. The ASYNC handler by type, the seed pace on a fake clock, and the named startup
    failure — `AsyncWebClient`'s pipeline calls `can_retry_async`, which the SYNC handler does not
    implement, so installing it would silently never retry."""
    monkeypatch.setenv("SLACK_TEST_USER_TOKEN", "xoxp-not-a-real-token")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-not-a-real-token")
    pair = bh.build_clients()
    for client in (pair.user, pair.bot):
        assert any(isinstance(h, AsyncRateLimitErrorRetryHandler) for h in client.retry_handlers)

    # Missing either token is a NAMED startup failure, not a row failure fifty seconds later.
    monkeypatch.delenv("SLACK_TEST_USER_TOKEN")
    with pytest.raises(bh.HarnessStartupError) as caught:
        bh.build_clients()
    assert "SLACK_TEST_USER_TOKEN" in str(caught.value)

    counter = {"n": 0}

    def _post(**kwargs):
        counter["n"] += 1
        return {"ok": True, "ts": f"7.{counter['n']:04d}"}

    monkeypatch.setattr(bh, "_clients", bh.Clients(user=FakeSlack(chat_postMessage=_post),
                                                   bot=FakeSlack()))
    clock_sleeps: List[float] = []
    monkeypatch.setattr(bh, "_sleep", lambda s: _record(clock_sleeps, s))
    await bh.seed_messages("C1", [f"m{i}" for i in range(120)])
    assert counter["n"] == 120
    assert clock_sleeps == [bh.SEED_PACE_SECONDS] * 119


async def _record(sink: List[float], seconds: float) -> None:
    sink.append(seconds)


# ------------------------------------------------------------------------------------- T120

def test_an_unrestored_row_does_not_report_pass(clock, install, monkeypatch, tmp_path):
    """T120. Durable state the battery could not put back downgrades a row; messages do not.

    SYNCHRONOUS on purpose: it drives `run_battery.main`, which calls `asyncio.run` and would
    raise inside a running loop. The async pieces get their own `asyncio.run`.
    """
    ctx = bh.RowContext(row="r", nonce="n", channel="C1")
    ctx.assert_that("everything held", True)
    unrestored = bh.CleanupResult(restored=(), restore_failures=("window_anchor:5.0|2",))
    assert bh.status_for(ctx, unrestored) == "unrestored"

    # MESSAGES LEFT IN THE ROOM ARE NOT A DOWNGRADE. The battery is not supposed to remove
    # them, so a row that seeded a hundred and observed five is a clean pass.
    ctx.seeded_ts.extend(["1.9", "2.9"])
    ctx.external_ts.append("3.9")
    assert bh.status_for(ctx, bh.CleanupResult(restored=(), restore_failures=())) == "pass"

    # `skipped` is legal ONLY via an explicit --rows; a row that cannot RUN is an `error`. Driven
    # BEFORE `run_battery` is stubbed out below, so this exercises the real selection.
    install()
    assert all(r["status"] != "skipped" for r in asyncio.run(_run_all_rows(monkeypatch)))
    partial = asyncio.run(runner.run_battery("C1", ["value-floor-holds"]))
    assert [r["status"] for r in partial].count("skipped") == 12

    async def _cannot_run(ctx: bh.RowContext) -> None:
        raise bh.HarnessScopeError("not_in_channel", code="not_in_channel")

    assert asyncio.run(_run_one("nope", _cannot_run))["status"] == "error"

    # THE AUTHORIZED-SCOPE RULE IS ENFORCED, not documented. One typo would seed 101 messages
    # into a real conversation, and no override flag exists because none was ever authorized.
    with pytest.raises(SystemExit, match="C0BKX77NU66"):
        runner.main(["--channel", "C_PRODUCTION", "--out", str(tmp_path / "x.json")])

    def _reports(*statuses: str) -> List[Dict[str, Any]]:
        return [{"row": f"r{i}", "status": s} for i, s in enumerate(statuses)]

    monkeypatch.setattr(runner, "run_battery",
                        lambda *a, **k: _returns(_reports("pass", "unrestored")))
    assert runner.main(["--out", str(tmp_path / "a.json")]) == 1
    monkeypatch.setattr(runner, "run_battery", lambda *a, **k: _returns(_reports("pass", "pass")))
    assert runner.main(["--out", str(tmp_path / "b.json")]) == 0
    # A selection that SKIPS rows still exits 0 when the ones it ran passed.
    monkeypatch.setattr(runner, "run_battery",
                        lambda *a, **k: _returns(_reports("pass", "skipped")))
    assert runner.main(["--out", str(tmp_path / "c.json")]) == 0


async def _run_one(name: str, run) -> Dict[str, Any]:
    return await runner.run_row(await _stub_row(name, run), "C1")


# ------------------------------------------------------------------------------------- T121

async def test_the_battery_never_deletes_a_message(clock, install, monkeypatch):
    """T121, under the owner's ruling of 2026-08-02: the run leaves the room exactly as it is.

    The owner watches these runs live and reads the channel afterwards, so a harness that
    removed its own seeds — or the bot's replies to them — would be erasing what they are
    reading. This drives the FULL row path, because the deleting version called `chat.delete`
    from `cleanup_row`'s `finally`, which is where a reinstated one would come back.
    """
    pair = install(user=FakeSlack(chat_delete={"ok": True}),
                   bot=FakeSlack(chat_delete={"ok": True}))
    applied: List[str] = []
    monkeypatch.setattr(bh, "apply_restore",
                        lambda restore, channel, team_id="": applied.append(restore.key))

    async def _row(ctx: bh.RowContext) -> None:
        ctx.seeded_ts.extend(["1.1", "1.2", "1.3"])
        ctx.observed_ts.extend(["2.1", "2.2"])
        ctx.external_ts.append("3.1")
        ctx.restores.append(bh.Restore(kind="window_anchor", key="5.0|2", prior=None,
                                       existed=False))
        ctx.assert_that("everything held", True)

    report = await runner.run_row(await _stub_row("keeps-everything", _row), "C1")

    assert pair.user.method_calls("chat_delete") == []
    assert pair.bot.method_calls("chat_delete") == []
    # Durable state IS still put back — that is bot configuration, not conversation.
    assert applied == ["5.0|2"]
    # And every ts stays in the report, which is now the only index back to the messages.
    assert report["status"] == "pass"
    assert report["seeded_ts"] == ["1.1", "1.2", "1.3"]
    assert report["observed_ts"] == ["2.1", "2.2"]
    assert report["external_ts"] == ["3.1"]


async def test_cleanup_enumerates_every_surface_a_turn_owns(clock, install, bot_db, monkeypatch):
    """T121. Receipts behind the turn's OWN completion fence — correlation AND completeness.

    THE RACE IS THE POINT. "A receipt exists and none is in_flight" is satisfied by a chrome-only
    snapshot before a word has landed, and by the first finalized part of a split reply before
    its later parts register. So the fixture starts INCOMPLETE and grows across the fence, the
    way a real turn does.
    """
    # What a poller would see first: chrome only, nothing in flight — and utterly premature.
    _insert_receipts(bot_db, [("9.0003", "chrome", None)])

    def _fence(turn_id, event, **kwargs):
        # `turn_outcome` lands only after the effects drained and the receipts settled, so the
        # later surfaces appear as it does — including one in a THREAD.
        _insert_receipts(bot_db, [("9.0001", "finalized", None),
                                  ("9.0002", "finalized", "8.0000")])
        return _returns({"kind": "reply",
                         "destinations": [{"first_ts": "9.0001", "kind": "reply"}]})

    monkeypatch.setattr(bh, "wait_for_telemetry", _fence)
    # An unrelated turn's surface, newer than ours. A time window would sweep it up.
    _insert_receipts(bot_db, [("9.0009", "finalized", None)], turn_id="SOMEONE-ELSE")

    top = {"ok": True, "messages": [bot_msg("9.0001", "part one"), bot_msg("9.0003", "thinking…"),
                                    bot_msg("9.0009", "theirs")]}
    threaded = {"ok": True, "messages": [bot_msg("9.0002", "part two", thread_ts="8.0000")]}
    pair = install(bot=FakeSlack(conversations_history=top, conversations_replies=threaded))
    ctx = bh.RowContext(row="r", nonce="n", channel="C1", team_id="T1")
    prose = await bh.observe_turn_output("T1", "C1", "TURN-A", deadline=0, ctx=ctx)

    # Chrome is RECORDED (the bot created it) but is not an ANSWER (it is not words).
    assert ctx.observed_ts == ["9.0001", "9.0002", "9.0003"]
    assert [o.ts for o in prose] == ["9.0001", "9.0002"]
    assert "9.0009" not in ctx.observed_ts

    # A MESSAGE WITH NO TS IS BROKEN EVIDENCE, not a missing message. Without the guard this
    # compares `"" == ts`, returns None, and a real surface reads as absent — so the row grades a
    # reply it never saw.
    install(bot=FakeSlack(conversations_history={"ok": True, "messages": [{"text": "no ts"}]}))
    with pytest.raises(bh.HarnessProtocolError, match="no usable ts"):
        await bh.fetch_message("C1", "9.0001")

    # A NON-NUMERIC ts passes an emptiness check and then flows into an equality compare that
    # never reaches the parser — so it would read as "not our message" instead of as damage.
    install(bot=FakeSlack(conversations_history={"ok": True,
                                                 "messages": [{"ts": "abc", "text": "junk"}]}))
    with pytest.raises(bh.HarnessProtocolError, match="message ts"):
        await bh.fetch_message("C1", "9.0001")

    install(bot=FakeSlack(conversations_history=top, conversations_replies=threaded))
    # THE THREADED READ IS INCLUSIVE. Slack's bounds exclude the boundary by default, so an
    # exact-ts read returns nothing and a real reply reads as missing despite its receipt.
    threaded_calls = pair.bot.method_calls("conversations_replies")
    assert threaded_calls and threaded_calls[0]["inclusive"] is True
    assert threaded_calls[0]["oldest"] == threaded_calls[0]["latest"] == "9.0002"


async def test_partial_settlement_after_the_outcome_is_a_harness_error(clock, install, bot_db,
                                                                        monkeypatch):
    """T121. `turn_outcome` is a NECESSARY fence, not a sufficient one — two production paths
    emit it without settled receipts.

    `_finalize_turn_effects` deliberately returns without settling when a flight drain fails AND
    the revocation fails, and `settle_ledger` catches its own 10s timeout and hands the settle to
    the drain worker. Both then emit `turn_outcome`. Receipts still `in_flight` at the bound are
    therefore BROKEN EVIDENCE about this turn: grading them would score the bot on half a reply
    and leave the other half in the channel.
    """
    _insert_receipts(bot_db, [("9.0001", "finalized", None), ("9.0002", "in_flight", None)])
    monkeypatch.setattr(bh, "wait_for_telemetry", lambda *a, **k: _returns(
        {"kind": "reply", "destinations": [{"first_ts": "9.0001", "kind": "reply"}]}))
    install()

    with pytest.raises(bh.HarnessProtocolError, match="still in_flight"):
        await bh.await_turn_settled("T1", "C1", "TURN-A", deadline=0)

    # ONE BOUND TOTAL, measured on the clock. Two full deadlines in sequence would silently
    # double a row's declared ceiling, so the assertion is on elapsed time, not on the argument.
    def _fence(turn_id, event, *, deadline=0, poll=0):
        clock.t += 12.0                      # the outcome took 12s of the 30s bound
        return _returns({"kind": "reply",
                         "destinations": [{"first_ts": "9.0001", "kind": "reply"}]})

    monkeypatch.setattr(bh, "wait_for_telemetry", _fence)
    started_at = clock.t
    with pytest.raises(bh.HarnessProtocolError):
        await bh.await_turn_settled("T1", "C1", "TURN-A", deadline=30, poll=5)
    assert clock.t - started_at <= 30, "the receipt wait started a fresh deadline"


async def test_a_delivered_outcome_waits_for_its_receipt(clock, install, monkeypatch):
    """T121. An outcome can precede the receipt ENTIRELY — and "no rows" must not read as settled.

    `ReceiptService.apply` QUEUES a failed registration, and a queued op merges per key, so a
    later finalize absorbs it. `turn_outcome` can therefore be emitted while the row has not been
    inserted at all. Returning `()` there would grade a turn that posted as SILENT and leave the
    late surface in the channel forever, with nothing tracking it.
    """
    install()
    monkeypatch.setattr(bh, "wait_for_telemetry", lambda *a, **k: _returns(
        {"kind": "reply", "destinations": [{"first_ts": "9.0001", "kind": "reply"}]}))

    # ZERO rows at the outcome, then the absorbing finalize lands two polls later.
    late = [(), (), ((("9.0001", "finalized"),)), ((("9.0001", "finalized"),))]
    reads = {"n": 0}

    async def _read(team_id, channel, turn_id):
        index = min(reads["n"], len(late) - 1)
        reads["n"] += 1
        return tuple(bh.Receipt(message_ts=ts, state=state, thread_root_ts=None)
                     for ts, state in late[index])

    monkeypatch.setattr(bh, "read_turn_receipts", _read)
    settled = await bh.await_turn_settled("T1", "C1", "TURN-A", deadline=60, poll=5)
    assert [r.message_ts for r in settled] == ["9.0001"]

    # ...and if it NEVER lands, that is broken evidence, not silence.
    reads["n"] = 0
    monkeypatch.setattr(bh, "read_turn_receipts", lambda *a, **k: _returns(()))
    with pytest.raises(bh.HarnessProtocolError, match="no receipt at all"):
        await bh.await_turn_settled("T1", "C1", "TURN-A", deadline=10, poll=5)

    # A SILENT turn names no destination, so zero receipts is the honest answer for it.
    monkeypatch.setattr(bh, "wait_for_telemetry",
                        lambda *a, **k: _returns({"kind": "silence", "destinations": []}))
    assert await bh.await_turn_settled("T1", "C1", "TURN-B", deadline=10, poll=5) == ()


async def test_a_late_unnamed_surface_is_caught_by_stability(clock, install, monkeypatch):
    """T121. The outcome names the FIRST ts of each destination, not a split reply's later parts
    or its chrome — so a set that agrees with the outcome can still be growing."""
    install()
    monkeypatch.setattr(bh, "wait_for_telemetry", lambda *a, **k: _returns(
        {"kind": "reply", "destinations": [{"first_ts": "9.0001", "kind": "reply"}]}))
    growing = [((("9.0001", "finalized"),)),
               ((("9.0001", "finalized"), ("9.0002", "finalized"))),
               ((("9.0001", "finalized"), ("9.0002", "finalized")))]
    reads = {"n": 0}

    async def _read(team_id, channel, turn_id):
        index = min(reads["n"], len(growing) - 1)
        reads["n"] += 1
        return tuple(bh.Receipt(message_ts=ts, state=state, thread_root_ts=None)
                     for ts, state in growing[index])

    monkeypatch.setattr(bh, "read_turn_receipts", _read)
    settled = await bh.await_turn_settled("T1", "C1", "TURN-A", deadline=60, poll=5)
    assert [r.message_ts for r in settled] == ["9.0001", "9.0002"]


async def test_a_post_window_surface_is_undetectable_and_says_so(clock, install, monkeypatch,
                                                                bot_db):
    """T121. THE HONEST BEHAVIOUR, which is not a detection.

    The receipt drain worker retries every 2 seconds INDEFINITELY, so a queued registration can
    fail through both reads of the stability window and succeed after it. No finite number of
    identical polls can prove an unobservable queue has drained — only a durable, production-owned
    completion fact could, and that would mean changing the shipped receipts subsystem.

    So this asserts what CAN be asserted: the late-third-state sequence completes without raising
    (the harness does not pretend to catch it), AND the row's report carries the limitation
    verbatim, so no green row implies a completeness guarantee this cannot give.
    """
    install()
    monkeypatch.setattr(bh, "wait_for_telemetry", lambda *a, **k: _returns(
        {"kind": "reply", "destinations": [{"first_ts": "9.0001", "kind": "reply"}]}))

    # complete-looking set -> IDENTICAL set (the window closes here) -> a late chrome receipt.
    sequence = [((("9.0001", "finalized"),)),
                ((("9.0001", "finalized"),)),
                ((("9.0001", "finalized"), ("9.0002", "chrome")))]
    reads = {"n": 0}

    async def _read(team_id, channel, turn_id):
        index = min(reads["n"], len(sequence) - 1)
        reads["n"] += 1
        return tuple(bh.Receipt(message_ts=ts, state=state, thread_root_ts=None)
                     for ts, state in sequence[index])

    monkeypatch.setattr(bh, "read_turn_receipts", _read)
    monkeypatch.setattr(bh, "fetch_message", lambda channel, ts, **k: _returns(
        bh.Observed(ts=ts, text="hi", thread_ts=None, channel=channel,
                    user=OURS.user_id, bot_id=OURS.bot_id)))

    ctx = bh.RowContext(row="r", nonce="n-abc", channel="C1", team_id="T1")
    prose = await bh.observe_turn_output("T1", "C1", "TURN-A", deadline=60, ctx=ctx)

    # It returns the set it could see. The late surface is NOT tracked — that is the bound.
    assert [o.ts for o in prose] == ["9.0001"]
    assert ctx.observed_ts == ["9.0001"]

    report = bh.build_report(ctx, status="pass", started_at=1.0, finished_at=2.0)
    limitation = [o for o in report["observations"] if o["name"] == bh.SETTLE_LIMITATION]
    assert len(limitation) == 1 and limitation[0]["observed"] is False
    # It states the bound HONESTLY: what the row could not see, and what that costs the grading.
    assert "bounded by the stability window" in bh.SETTLE_LIMITATION
    assert "observed_ts" in bh.SETTLE_LIMITATION and "never graded" in bh.SETTLE_LIMITATION
    # An observation can never change a status.
    assert report["status"] == "pass"

    # Once per ROW, not once per turn: a row reading three turns has one limitation, not three.
    await bh.observe_turn_output("T1", "C1", "TURN-A", deadline=60, ctx=ctx)
    assert sum(1 for name, _, _ in ctx.observations if name == bh.SETTLE_LIMITATION) == 1


def test_a_delivering_outcome_with_no_destinations_is_broken_evidence():
    """T121. Reading a malformed destination list as EMPTY makes a `reply` indistinguishable from
    a `silence` — so a turn that posted would be graded quiet and its surface never cleaned."""
    assert bh.delivered_surfaces({"kind": "silence", "destinations": []}) == set()
    assert bh.delivered_surfaces({"kind": "silence"}) == set()
    assert bh.delivered_surfaces(
        {"kind": "reply", "destinations": [{"first_ts": "9.1"}]}) == {"9.1"}

    for broken in ({"kind": "reply"},
                   {"kind": "reply", "destinations": []},
                   {"kind": "reply", "destinations": [{"first_ts": None}]},
                   {"kind": "reply", "destinations": "nope"},
                   {"kind": "reply", "destinations": ["not an object"]},
                   {"kind": "reply", "destinations": [{"first_ts": 7}]},
                   # A MALFORMED SHAPE IS BROKEN EVIDENCE WHATEVER THE KIND. Skipping the bad
                   # entry would let a silent-looking outcome hide a destination that failed to
                   # serialize — and a `reply` only raises here because a SECOND guard catches
                   # the empty result, which this kind does not have.
                   {"kind": "silence", "destinations": ["not an object"]},
                   {"kind": "silence", "destinations": [{"first_ts": 7}]},
                   {"kind": "silence", "destinations": 42},
                   # A non-numeric ts would compare unequal against every receipt and read as a
                   # destination whose receipt never arrived — damaged evidence wearing the
                   # costume of a missing one.
                   {"kind": "reply", "destinations": [{"first_ts": "abc"}]},
                   {"kind": "silence", "destinations": [{"first_ts": "not-a-ts"}]}):
        with pytest.raises(bh.HarnessProtocolError):
            bh.delivered_surfaces(broken)


async def test_a_failed_settle_still_reports_the_limitation(clock, install, monkeypatch):
    """T121. A row whose settle FAILED is exactly the one whose report must not look like it
    carried a completeness guarantee — so the observation is recorded BEFORE the read."""
    install()
    monkeypatch.setattr(bh, "wait_for_telemetry", lambda *a, **k: _returns(
        {"kind": "reply", "destinations": [{"first_ts": "9.0001", "kind": "reply"}]}))
    monkeypatch.setattr(bh, "read_turn_receipts", lambda *a, **k: _returns(()))

    ctx = bh.RowContext(row="r", nonce="n-abc", channel="C1", team_id="T1")
    with pytest.raises(bh.HarnessProtocolError):
        await bh.observe_turn_output("T1", "C1", "TURN-A", deadline=0, ctx=ctx)

    report = bh.build_report(ctx, status="error", started_at=1.0, finished_at=2.0)
    assert [o["name"] for o in report["observations"]] == [bh.SETTLE_LIMITATION]


async def test_receipts_are_ordered_by_the_canonical_key(clock, install, bot_db, monkeypatch):
    """T121. SQL's `ORDER BY message_ts` is a LEXICAL sort, so "1000.000001" would come back
    before "999.999999" and a row would grade the wrong surface as "the first"."""
    _insert_receipts(bot_db, [("1000.000001", "finalized", None),
                              ("999.999999", "finalized", None)])
    assert [r.message_ts for r in await bh.read_turn_receipts("T1", "C1", "TURN-A")] == \
        ["999.999999", "1000.000001"]

    # A corrupt receipt is broken evidence, not a silently mis-ordered one.
    _insert_receipts(bot_db, [("not-a-ts", "finalized", None)], turn_id="TURN-B")
    with pytest.raises(bh.HarnessProtocolError, match="receipt message_ts"):
        await bh.read_turn_receipts("T1", "C1", "TURN-B")


async def test_the_observed_router_sorts_by_author(clock, install):
    """The routing rule the report depends on: ours, the operator's, and a third app's."""
    ctx = bh.RowContext(row="r", nonce="n", channel="C1")
    await bh.record_observed(ctx, [
        bh.Observed("2.1", "ours", None, "C1", OURS.user_id, OURS.bot_id),
        bh.Observed("1.1", "seed", None, "C1", OPERATOR.user_id, None),
        bh.Observed("3.1", "theirs", None, "C1", CLAUDE.user_id, CLAUDE.bot_id),
    ], ours=OURS, operator=OPERATOR)
    assert (ctx.observed_ts, ctx.seeded_ts, ctx.external_ts) == (["2.1"], ["1.1"], ["3.1"])


# ------------------------------------------- the owner's ruling of 2026-08-02, in two tests

# What a synthetic test marker looks like. The first is the shape the old nonces had
# (`searchlag1785728257`, `hx6p2v`-style hex); the rest are the words that gave a message away as
# machinery rather than conversation.
_MARKER_HEX = re.compile(r"\b[0-9a-f]{8,}\b")
_MARKER_WORDS = ("filler", "nonce", "probe", "marker", "battery", "harness", "fixture",
                 "assertion", "seeded")
_SHOUTED = re.compile(r"\b[A-Z]{5,}\b")


def _reads_as_a_person(text: str) -> Optional[str]:
    """The first reason `text` would give itself away as a test, or None."""
    if _MARKER_HEX.search(text):
        return "carries a token-shaped id"
    if _SHOUTED.search(text):
        return "shouts a marker word"
    lowered = text.lower()
    for word in _MARKER_WORDS:
        if word in lowered:
            return f"says {word!r}"
    return None


def _posted_literals(module=None) -> List[str]:
    """Every string literal a live module hands to `post_seed` / `seed_messages`.

    READ OFF THE AST, not off a list someone maintains: a row added next year posts its seed the
    same way, so it is scanned without anybody remembering to add it here. F-strings contribute
    their literal parts; the substituted values come from the generators, which the other test
    covers.
    """
    # A live module always has a __file__; the Optional is the general module-type declaration.
    tree = ast.parse(pathlib.Path(cast(str, (module or rows).__file__)).read_text(encoding="utf-8"))
    found: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name not in ("post_seed", "seed_messages"):
            continue
        for argument in node.args[1:2]:
            if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                found.append(argument.value)
            elif isinstance(argument, ast.JoinedStr):
                found.append("".join(part.value for part in argument.values
                                     if isinstance(part, ast.Constant)
                                     and isinstance(part.value, str)))
    return found


def test_nothing_the_battery_posts_looks_like_a_test():
    """The owner's ruling, enforced rather than remembered.

    A channel full of `searchlag1785728257` and `TANGERINE` is a channel the owner cannot read,
    and they read this one live. So every literal a row posts, every bait and aside, and every
    line of bulk chatter has to pass for something a coworker typed.
    """
    material = list(_posted_literals())
    assert len(material) >= 20, "the AST scan found almost nothing; it stopped matching the rows"
    # The classify probe posts into the same channel and is part of the harness contract now, so
    # its remarks live under the same rule as the rows'.
    material.extend(_posted_literals(classify_probe))
    material.extend(classify_probe.ASIDES)
    material.extend(rows._BAIT_VARIANTS)
    material.extend(rows._ASIDE_VARIANTS)
    material.extend(bh.chatter_lines("re-anchor-observable-4f2a1b9c", 40))
    material.extend([bh.vendor_name("x"), bh.person_name("x"), bh.quantity("x"), bh.money("x"),
                     bh.date_phrase("x"), bh.weekday("x")])

    for text in material:
        reason = _reads_as_a_person(text)
        assert reason is None, f"{text!r} {reason}"

    # And the nonce itself is never posted: it is the seed for the facts, not one of them.
    nonce = bh.mint_nonce("cross-thread-awareness")
    assert nonce not in " ".join(material)


def test_the_seeded_facts_are_run_unique_and_the_chatter_stays_clear_of_them():
    """Unguessability moved from the text's SHAPE into its CONTENT, so the content must carry it.

    Two runs must not mint the same supplier and quantity — a stale answer from last week's run
    would satisfy this week's assertion — and the hundred chatter lines that bury a fact must not
    state the number the row grades on, which would leave the graded fact ABOVE the floor and let
    the row pass without the bot searching for anything.
    """
    runs = [bh.mint_nonce("verification-rule") for _ in range(50)]
    assert len({bh.vendor_name(n) for n in runs}) > 40
    # THE PAIR, not either half. A five-digit quantity has ninety thousand values, so fifty draws
    # collide about one run in seventy — asserting the number alone is unique makes this test
    # flaky, which it duly was. What the row actually relies on is the supplier AND the figure
    # together, and that space is three thousand times larger.
    assert len({(bh.vendor_name(n), bh.quantity(n)) for n in runs}) == 50
    # Deterministic from the nonce, which is what makes a report reproduce its own run.
    assert bh.vendor_name(runs[0]) == bh.vendor_name(runs[0])

    # THE AVOIDED NUMBER IS ONE THE CHATTER REALLY WOULD HAVE SAID. Handing it a quantity the
    # word banks could never produce exercises nothing: the guard's removal is invisible, and the
    # first version of this test passed with it deleted.
    # THE NAME GUARD, on a supplier the generator REALLY produces (codex): row 4 concludes the bot
    # read the buried reply because its supplier exists in exactly one message, and chatter mints
    # suppliers from the same space. Derived, not invented — an avoided name the banks could never
    # produce would leave the guard's removal invisible, which is the trap the number guard
    # already sprang once.
    name_seed = "search-to-action-4f2a1b9c|chatter"
    plain = bh.chatter_lines(name_seed, 101)
    collided = next(name for index in range(101)
                    for name in [bh.vendor_name(name_seed, f"v-{index}-0")]
                    if any(bh.states_phrase(line, name) for line in plain))
    guarded = bh.chatter_lines(name_seed, 101, avoid_names=[collided])
    assert [line for line in plain if bh.states_phrase(line, collided)]
    assert not [line for line in guarded if bh.states_phrase(line, collided)]

    # REPRODUCIBLE FROM THE REPORT. Rows seed their chatter from `ctx.nonce`, so the same nonce
    # replays the whole run — a fresh random seed at the call site was a run nobody could rebuild.
    assert bh.chatter_lines("row-nonce|chatter", 20) == bh.chatter_lines("row-nonce|chatter", 20)
    assert bh.chatter_lines("row-nonce|chatter", 20) != bh.chatter_lines("other|chatter", 20)

    seed = "verification-rule-4f2a1b9c-chatter"
    unguarded = bh.chatter_lines(seed, 101)
    collision = next(number for line in unguarded for number in re.findall(r"\d+", line))
    assert any(bh.states_number(line, collision) for line in unguarded)

    lines = bh.chatter_lines(seed, 101, avoid=[collision, bh.quantity(seed, "quote")])
    assert len(lines) == len(set(lines)) == 101
    assert not [line for line in lines if bh.states_number(line, collision)]
    # Statements, never questions and never mentions: the gate must decline all hundred of them,
    # or a bulk row spends a hundred turns instead of a hundred posts.
    assert not [line for line in lines if "?" in line or "<@" in line]


def test_a_number_is_compared_by_its_digits_and_a_name_by_its_letters():
    """The 2026-08-02 defect, pinned in both directions.

    The bot answered "847,800 crates." to a seeded `847800` and the row failed a correct answer
    over the comma. Loosening it is only safe if a DIFFERENT number still fails, which is why the
    match is bounded rather than a bare substring.
    """
    assert bh.states_number("847,800 crates.", "847800")
    assert bh.states_number("the tally was 12 480 cases", "12,480")
    assert bh.states_number("we paid $41,770 in the end", "$41770")
    assert not bh.states_number("1847800 crates", "847800")     # a longer number is another number
    assert not bh.states_number("847801 crates", "847800")
    assert not bh.states_number("no numbers here", "847800")
    assert not bh.states_number("847800", "")

    # ZERO CENTS ARE THE SAME NUMBER; other cents are not (codex). A model writing a price out in
    # full was failing a row whose own docstring promised it would pass.
    assert bh.states_number("the quote was $41,770.00", "$41,770")
    assert bh.states_number("the quote was 41770.0", "41770")
    assert not bh.states_number("the quote was $41,770.50", "$41,770")
    assert bh.states_number("1,000 boxes turned up", "1000")   # a thousands group is not cents

    assert bh.states_phrase("kestwood freight are holding it up", "Kestwood Freight")
    assert bh.states_phrase("blame  Kestwood   Freight", "Kestwood Freight")
    assert not bh.states_phrase("Kestwood are holding it up", "Kestwood Freight")
    assert not bh.states_phrase("anything", "")
    # BOUNDED, or "naming a supplier" is plain substring matching and row 4's proof evaporates.
    assert not bh.states_phrase("NotKestwood Freightening are late", "Kestwood Freight")
    assert not bh.states_phrase("Kestwood Freightening", "Kestwood Freight")
    assert bh.states_phrase("(Kestwood Freight), finally.", "Kestwood Freight")
    assert bh.states_phrase("blame Kestwood Freight-the-younger", "Kestwood Freight")


async def test_the_probe_waits_for_its_own_thread_to_exist(clock, install):
    """Row 10's 2026-08-02 error: `chat.postMessage` returned a ts the next read could not see.

    The probe launched immediately and died on `OriginFetchError: origin thread … came back empty
    for a reply-triggered turn`. Three things had to be true before that wait meant anything, and
    each was learned the hard way:

    * it asks in production's shape (`latest=<horizon>, inclusive=True`) — an unbounded read saw
      both messages while the probe half a second later still read nothing;
    * it wants the complete answer TWICE, because one complete-looking read is what the failing
      run already had;
    * it waits for the TS VALUES the caller posted, not for a count — any unrelated reply in the
      same thread satisfies a count while our own message is still invisible (codex).
    """
    theirs = bot_msg("8.0009", "someone else's reply")
    pages = [replies_page([bot_msg("8.0000"), theirs]),          # count is met, ours is missing
             replies_page([bot_msg("8.0000"), theirs, bot_msg("8.0001")]),
             replies_page([bot_msg("8.0000"), theirs, bot_msg("8.0001")])]
    reads = {"n": 0}

    def _replies(**kwargs):
        page = pages[min(reads["n"], len(pages) - 1)]
        reads["n"] += 1
        return page

    pair = install(bot=FakeSlack(conversations_replies=_replies))
    messages = await bh.await_thread_visible("C1", "8.0000", expected_ts=("8.0000", "8.0001"),
                                             deadline=60)
    assert [m["ts"] for m in messages] == ["8.0000", "8.0009", "8.0001"]
    assert reads["n"] == 3, "a count another thread's reply satisfies is not visibility"
    asked = pair.bot.method_calls("conversations_replies")[-1]
    assert asked["inclusive"] is True and asked["latest"], "not the read production performs"

    install(bot=FakeSlack(conversations_replies=replies_page([bot_msg("8.0000"), theirs])))
    with pytest.raises(bh.HarnessError, match="8.0001"):
        await bh.await_thread_visible("C1", "8.0000", expected_ts=("8.0000", "8.0001"),
                                      deadline=0)

    # A wait with nothing named cannot be satisfied by anything, so it refuses rather than
    # returning the first page it sees.
    with pytest.raises(bh.HarnessError, match="ts values"):
        await bh.await_thread_visible("C1", "8.0000", expected_ts=(), deadline=0)


async def test_holding_one_turn_at_a_seam_frees_every_other_one(clock, monkeypatch, tmp_path):
    """Row 7's 2026-08-03 error, twice over: `error: turn C never replied`.

    THE SEAM IS PROCESS-GLOBAL. Row 7 freezes turn A at `post_partial_post` on purpose — and its
    own B and C turns then froze on the same seam at their first streamed chunk, with nobody
    watching those keys. B sat until the bot's own timeout, C never produced words, and the row
    left a frozen streaming message in the channel that no token could delete.

    So the row holds ONE key and frees the rest. Freeing A's too would destroy the experiment,
    which is why that name is the one thing this never touches.
    """
    monkeypatch.setenv("DEV_TURN_BARRIERS_DIR", str(tmp_path))
    held = bh.barrier_waiting_path(POST_PARTIAL_POST, "9.0001")
    other = bh.barrier_waiting_path(POST_PARTIAL_POST, "9.0002")
    for path in (held, other):
        path.write_text("waiting", encoding="utf-8")

    async def _sleep_once(seconds):
        await asyncio.sleep(0)

    monkeypatch.setattr(bh, "_sleep", _sleep_once)
    async with bh.freeing_other_turns(POST_PARTIAL_POST, "9.0001") as freed:
        for _ in range(20):
            await asyncio.sleep(0)
            if freed:
                break

    assert freed == [other.name]
    assert bh.barrier_release_path(POST_PARTIAL_POST, "9.0002").exists()
    assert not bh.barrier_release_path(POST_PARTIAL_POST, "9.0001").exists(), \
        "it released the very turn the row is holding"


# ------------------------------------------------------------------------------------- T122

REGISTRY_NAMES = ("cross-thread-awareness", "verification-rule", "cross-thread-action",
                  "search-to-action", "full-origin-fidelity", "stream-currency",
                  "in-flight-exclusion", "re-anchor-observable", "foreign-exchange-bait",
                  "directed-banter-answered", "thanks-response-choice", "value-floor-holds",
                  "render-equality-probe")


async def _drive_response_row(monkeypatch, row, *, kind, messages, reactions):
    """One of the two model-choice rows against a fixed outcome, network-free.

    The harness seams are replaced rather than the rows' logic: the row still asks the verdict
    reader what happened and still routes surfaces the way it does live.
    """
    counter = {"n": 0}

    async def _post(channel, text, thread_ts=None, ctx=None):
        counter["n"] += 1
        ts = f"10.{counter['n']:04d}"
        if ctx is not None:
            ctx.seeded_ts.append(ts)
        return ts

    async def _output(team_id, channel, turn_id, *, deadline=0, poll=0, ctx=None):
        out = []
        for index in range(messages):
            ts = f"99.{index:04d}"
            if ctx is not None:
                ctx.observed_ts.append(ts)
            out.append(bh.Observed(ts=ts, text="sure thing", thread_ts="10.0001", channel=channel,
                                   user=OURS.user_id, bot_id=OURS.bot_id))
        if ctx is not None and out:
            ctx.evidence.setdefault("observed_text", {})[turn_id] = [
                {"ts": o.ts, "thread_ts": o.thread_ts, "text": o.text} for o in out]
        return out

    verdict = bh.TriggerVerdict(kind=kind, woke=kind != "silence", turn_id="T-1",
                                source="turn_outcome")
    monkeypatch.setattr(rows, "post_seed", _post)
    monkeypatch.setattr(rows, "observe_turn_output", _output)
    monkeypatch.setattr(rows, "bot_identity", lambda: _returns(OURS))
    monkeypatch.setattr(rows, "await_trigger_verdict", lambda *a, **k: _returns(verdict))
    monkeypatch.setattr(rows, "wait_bot_reaction", lambda *a, **k: _returns(tuple(reactions)))

    ctx = bh.RowContext(row=row.name, nonce="n-1", channel="C1", team_id="T1")
    await row.run(ctx)
    return ctx


async def test_every_buried_fact_row_keeps_both_halves_out_of_its_chatter(clock, monkeypatch):
    """The guard has to cover whatever the row GRADES, and the graded fact is a pair now.

    Guarding the supplier while the chatter is free to state the figure puts half the graded fact
    back inside the window — the row could then be satisfied without the buried sentence ever
    being read. This asserts the ARGUMENTS each row hands the generator, because the guard itself
    is tested elsewhere and what regresses is the call site.
    """
    calls: List[Dict[str, Any]] = []
    counter = {"n": 0}

    async def _post(channel, text, thread_ts=None, ctx=None):
        counter["n"] += 1
        return f"10.{counter['n']:04d}"

    async def _seed(channel, texts, thread_ts=None, ctx=None):
        return [f"20.{index:04d}" for index, _ in enumerate(texts)]

    def _chatter(seed, count, *, avoid=(), avoid_names=()):
        calls.append({"seed": seed, "avoid": list(avoid), "avoid_names": list(avoid_names)})
        return [f"chatter {index}" for index in range(count)]

    async def _output(team_id, channel, turn_id, *, deadline=0, poll=0, ctx=None):
        return [bh.Observed(ts="99.0001", text="answered", thread_ts="10.0001", channel=channel,
                            user=OURS.user_id, bot_id=OURS.bot_id)]

    monkeypatch.setattr(rows, "post_seed", _post)
    monkeypatch.setattr(rows, "seed_messages", _seed)
    monkeypatch.setattr(rows, "chatter_lines", _chatter)
    monkeypatch.setattr(rows, "observe_turn_output", _output)
    monkeypatch.setattr(rows, "bot_identity", lambda: _returns(OURS))
    monkeypatch.setattr(rows, "find_turn_id", lambda *a, **k: _returns("T-1"))
    # search-to-action seeds its buried obligation as a MENTION and reads the verdict that mention
    # earns before it buries it, exactly as the cross-thread row reads its premise.
    monkeypatch.setattr(rows, "await_trigger_verdict", lambda *a, **k: _returns(
        bh.TriggerVerdict(kind="in_thread_reply", woke=True, turn_id="T-0",
                          source="turn_outcome")))
    monkeypatch.setattr(rows, "await_tools_used_for",
                        lambda *a, **k: _returns(bh.ProvenanceRead(True, ("search_slack",))))
    monkeypatch.setattr(rows, "wait_for_telemetry", lambda *a, **k: _returns(
        {"kind": "detached", "destinations": [], "origin_count": 122}))

    for row_name, run, figure_of, name_of in [
            ("verification-rule", rows.row_verification_rule,
             lambda n: rows.verification_decision(n), lambda n: bh.vendor_name(n)),
            ("search-to-action", rows.row_search_to_action,
             lambda n: bh.money(n, "reissue"), lambda n: bh.vendor_name(n, "issuer")),
            ("full-origin-fidelity", rows.row_full_origin_fidelity,
             lambda n: bh.quantity(n, "tally", digits=5), lambda n: bh.vendor_name(n, "count"))]:
        calls.clear()
        ctx = bh.RowContext(row=row_name, nonce=f"{row_name}-n1", channel="C1", team_id="T1")
        await run(ctx)
        assert len(calls) == 1, f"{row_name} did not mint its chatter exactly once"
        seeded = calls[0]
        assert seeded["seed"] == f"{ctx.nonce}|chatter", f"{row_name}'s chatter is not replayable"
        assert figure_of(ctx.nonce) in seeded["avoid"], f"{row_name} left its figure in the window"
        assert name_of(ctx.nonce) in seeded["avoid_names"], f"{row_name} left its supplier there"


def test_every_seeded_count_shrinks_with_the_window(monkeypatch):
    """The owner's 2026-08-03 ruling, enforced: no row may hardcode its bulk.

    The seeding exists only to push a fact below the rendered floor, and the floor is env-tunable —
    so a bot launched at `CHANNEL_WINDOW_CEILING=12` must cost the room ~13 messages where the
    shipped 100 costs ~101. Every count is derived at RUNTIME from the resolved config, which is
    what makes small-window mode a launch-environment decision rather than a code change.
    """
    def _at(ceiling: int, target: int):
        monkeypatch.setattr(rows, "window_ceiling", lambda: ceiling)
        monkeypatch.setattr(rows, "window_target", lambda: target)
        return rows.window_ceiling() + 1, rows.origin_reply_count()

    fillers_small, origin_small = _at(12, 8)
    assert (fillers_small, origin_small) == (13, 24)

    fillers_shipped, origin_shipped = _at(100, 50)
    assert (fillers_shipped, origin_shipped) == (101, 120), \
        "the shipped window must still seed what it always did"

    # The cap is what stops the derivation growing the origin thread past what the row has ever
    # proven, and the floor stops a tiny window producing a thread too shallow to bury anything.
    assert _at(400, 200)[1] == rows.ORIGIN_REPLY_CAP
    assert _at(1, 1)[1] == 4


async def test_the_report_records_the_window_it_ran_against(clock, install, monkeypatch):
    """A pass at 8/12 is a pass at 8/12 — never a pass at the shipped 50/100.

    The harness cannot read the BOT's environment, so the window it computed against is a fact the
    report has to carry; without it a small-window green is indistinguishable from a full-window
    one to anybody reading the file later.
    """
    install()
    monkeypatch.setattr(runner, "resolved_window",
                        lambda: {"channel_window_target": 8, "channel_window_ceiling": 12})

    async def _passes(ctx: bh.RowContext) -> None:
        ctx.assert_that("everything held", True)

    report = await runner.run_row(await _stub_row("windowed", _passes), "C1")
    assert report["evidence"]["window"] == {"channel_window_target": 8,
                                            "channel_window_ceiling": 12}


async def test_the_deep_origin_row_grades_both_halves_of_its_buried_fact(clock, monkeypatch):
    """Codex verify-7: a buried fact is a supplier AND a figure, and one half is not proof.

    Row 5's fact sits in the root, 121 messages above the question, and the trigger asks for both
    halves — so requiring both is asking whether the oldest message survived the render, not
    whether the bot phrases things the way the harness likes.
    """
    answer = {"text": ""}
    counter = {"n": 0}

    async def _post(channel, text, thread_ts=None, ctx=None):
        counter["n"] += 1
        ts = f"10.{counter['n']:04d}"
        if ctx is not None:
            ctx.seeded_ts.append(ts)
        return ts

    async def _seed(channel, texts, thread_ts=None, ctx=None):
        return [await _post(channel, text, thread_ts=thread_ts, ctx=ctx) for text in texts]

    async def _output(team_id, channel, turn_id, *, deadline=0, poll=0, ctx=None):
        return [bh.Observed(ts="99.0001", text=answer["text"], thread_ts="10.0001",
                            channel=channel, user=OURS.user_id, bot_id=OURS.bot_id)]

    monkeypatch.setattr(rows, "post_seed", _post)
    monkeypatch.setattr(rows, "seed_messages", _seed)
    monkeypatch.setattr(rows, "observe_turn_output", _output)
    monkeypatch.setattr(rows, "find_turn_id", lambda *a, **k: _returns("T-1"))
    monkeypatch.setattr(rows, "wait_for_telemetry",
                        lambda *a, **k: _returns({"origin_count": 122}))

    async def _run(template):
        counter["n"] = 0
        ctx = bh.RowContext(row="full-origin-fidelity", nonce="n-1", channel="C1", team_id="T1")
        supplier = bh.vendor_name("n-1", "count")
        tally = bh.quantity("n-1", "tally", digits=5)
        answer["text"] = template.format(supplier=supplier, tally=tally)
        await rows.row_full_origin_fidelity(ctx)
        return dict(ctx.assertions)

    both = await _run("{tally} cases, on {supplier}'s count.")
    assert both["the reply carries this run's opening tally"] is True
    assert both["the reply names whose count it was"] is True

    figure_only = await _run("{tally} cases.")
    assert figure_only["the reply names whose count it was"] is False
    name_only = await _run("it was {supplier}'s count.")
    assert name_only["the reply carries this run's opening tally"] is False
    # The render assertion is untouched in every case — the pair is what changed.
    assert both["origin_count is 122"] is True and name_only["origin_count is 122"] is True


async def test_cross_thread_action_grades_the_answer_that_landed(clock, monkeypatch):
    """Codex: destination + absence-from-A + `kind="detached"` are all satisfied by a post under C
    that says "I don't know" — the row passing while the information never moved.

    The seeded capacity is what the thread asked for, so requiring the post to state it is
    information FLOW, not expression: this row exists to prove the number reached the thread.
    """
    posts = {"text": ""}
    counter = {"n": 0}

    async def _post(channel, text, thread_ts=None, ctx=None):
        counter["n"] += 1
        ts = f"10.{counter['n']:04d}"
        if ctx is not None:
            ctx.seeded_ts.append(ts)
        return ts

    _WOKEN = bh.TriggerVerdict(kind="detached", woke=True, turn_id="T-1", source="turn_outcome")

    async def _output(team_id, channel, turn_id, *, deadline=0, poll=0, ctx=None):
        if turn_id == "T-0":
            return []             # the premise turn, in the runs where it says nothing
        return [bh.Observed(ts="99.0001", text=posts["text"], thread_ts="10.0001",
                            channel=channel, user=OURS.user_id, bot_id=OURS.bot_id)]

    # SEQUENTIAL VERDICTS, one per trigger. Handing the row the same woken verdict twice would
    # leave the premise's independence untested: a regression that returns early when the FIRST
    # trigger sleeps — which is exactly what the row used to do — passes a same-verdict fixture,
    # because the first trigger never sleeps in it.
    verdicts: List[Any] = []

    async def _verdict(*args, **kwargs):
        return verdicts.pop(0) if verdicts else _WOKEN

    monkeypatch.setattr(rows, "post_seed", _post)
    monkeypatch.setattr(rows, "observe_turn_output", _output)
    monkeypatch.setattr(rows, "bot_identity", lambda: _returns(OURS))
    monkeypatch.setattr(rows, "await_trigger_verdict", _verdict)
    monkeypatch.setattr(rows, "wait_for_telemetry", lambda *a, **k: _returns(
        {"kind": "detached",
         "destinations": [{"kind": "post_to_thread", "thread_root_ts": "10.0001"}]}))

    async def _run(answer_template, premise=None):
        counter["n"] = 0          # the fixture's destination is `10.0001`, so C must be minted
        verdicts[:] = [premise, _WOKEN] if premise is not None else []
        ctx = bh.RowContext(row="cross-thread-action", nonce="n-1", channel="C1", team_id="T1")
        capacity = bh.quantity("n-1", "pallet-capacity", digits=2)
        posts["text"] = answer_template.format(capacity=capacity)
        await rows.row_cross_thread_action(ctx)
        return ctx

    carried = dict((await _run("{capacity} crates a pallet, going by what was just posted.")
                    ).assertions)
    assert carried["the post under C states the capacity we just supplied"] is True
    # THE PREMISE IS AN OBSERVATION NOW (owner ruling, 2026-08-03). `{C}` is unanswerable when it
    # is asked, so grading the bot for answering it asked for the one thing the tuned prompts
    # still refuse — and the row died there twice without ever reaching the half it exists for.
    assert not any(name.startswith("the bot answered the open pallet question")
                   for name in carried), "the premise is graded again"
    assert rows.row_by_name("cross-thread-action").assertions[0] == "the fyi opened a turn"

    dodged = dict((await _run("I don't know the pallet size, sorry.")).assertions)
    assert dodged["the post under C states the capacity we just supplied"] is False
    # The older assertions still hold in BOTH runs, which is exactly why this one was needed.
    assert dodged["exactly one post_to_thread landed under C"] is True
    assert dodged["A heard nothing, or only a brief non-reporting acknowledgment"] is True

    # AND THE INDEPENDENCE ITSELF: a premise the gate DECLINED, followed by a woken fyi. Every
    # graded claim the row owns still has to run, and the choice has to reach the report.
    for premise in (bh.TriggerVerdict(kind="silence", woke=True, turn_id="T-0",
                                      source="turn_outcome"),
                    bh.TriggerVerdict(kind="declined", woke=False, turn_id=None,
                                      source="visible_action")):
        ctx = await _run("{capacity} crates a pallet.", premise=premise)
        graded = dict(ctx.assertions)
        assert set(graded) == {"the fyi opened a turn",
                               "exactly one post_to_thread landed under C",
                               "the post under C states the capacity we just supplied",
                               "A heard nothing, or only a brief non-reporting acknowledgment"}, (
            f"the row stopped short after a {premise.kind} premise: {sorted(graded)}")
        assert all(graded.values()), graded
        recorded = [value for name, value, _ in ctx.observations
                    if name == "what the bot did with the open pallet question"]
        assert recorded and recorded[0]["choice"] == "silence"
        assert recorded[0]["woke"] is premise.woke


async def test_the_thanks_row_records_the_choice_and_grades_nothing(clock, monkeypatch):
    """The owner's ruling of 2026-08-03, pinned: an emoji, a short line and silence are all fine.

    Which one the responder picks is not the harness's business, so the row asserts NOTHING and
    always passes — what it owes the report is the choice itself, and the words when there were
    words. A row that grades this is a row that fails the bot for using judgement it is meant to
    have.
    """
    row = rows.row_by_name("thanks-response-choice")
    assert row is not None and row.observation_only and row.assertions == ()

    for kind, messages, reactions, expected in [
            ("reaction_only", 0, ["pray"], "reaction"),
            ("reply", 1, [], "message"),
            ("silence", 0, [], "silence"),
            ("reply", 1, ["pray"], "message and reaction")]:
        ctx = await _drive_response_row(monkeypatch, row, kind=kind, messages=messages,
                                        reactions=reactions)
        assert ctx.assertions == [], f"{kind} was graded; the row must only observe"
        assert bh.status_for(ctx, bh.CleanupResult(restored=(), restore_failures=())) == "pass"
        assert ctx.evidence["choice"] == expected
        recorded = [value for name, value, _ in ctx.observations
                    if name == "what the bot chose in answer to thanks"]
        assert recorded and recorded[0]["choice"] == expected
        if messages:
            # The WORDS, not just the shape — the report has to show what it actually said.
            assert ctx.evidence["observed_text"]["T-1"][0]["text"] == "sure thing"


async def test_directed_banter_accepts_any_visible_response(clock, monkeypatch):
    """Same ruling, the other half: being addressed and responding is MACHINERY and stays graded.

    The FORM is not. An earlier version asserted `turn_outcome.kind == "reply"`, so a bot that
    answered a joke with a single 🌭 — a perfectly good answer — would have failed the row.
    """
    row = rows.row_by_name("directed-banter-answered")
    assert row is not None and not row.observation_only

    words = await _drive_response_row(monkeypatch, row, kind="reply", messages=1, reactions=[])
    emoji = await _drive_response_row(monkeypatch, row, kind="reaction_only", messages=0,
                                      reactions=["hotdog"])
    for ctx in (words, emoji):
        assert ctx.assertions == [("the bot responded when directly addressed", True)]

    quiet = await _drive_response_row(monkeypatch, row, kind="silence", messages=0, reactions=[])
    assert quiet.assertions == [("the bot responded when directly addressed", False)]


def test_the_row_registry_is_complete_and_unique():
    """T122, first half. A row in §9's table with no implementation, or vice versa, fails."""
    assert rows.ROW_NAMES == REGISTRY_NAMES
    assert len(set(rows.ROW_NAMES)) == 13
    for row in rows.REGISTRY:
        assert callable(row.run) and asyncio.iscoroutinefunction(row.run)
        assert row.trigger_template.strip()
        # AN EMPTY `assertions` IS LEGAL ONLY WHEN THE ROW SAYS SO. Observation-only rows grade
        # nothing by contract (owner ruling, 2026-08-03); every other empty one is an
        # implementation that would report `pass` for having done nothing.
        assert row.assertions or row.observation_only

    # And the observation-only set is PINNED, so a row cannot quietly stop grading.
    assert {row.name for row in rows.REGISTRY if row.observation_only} == {
        "thanks-response-choice"}

    # THE MAPPING, PINNED. "Every name has some coroutine" still passes when two rows are wired
    # to each other's implementation — the registry would look complete while the battery graded
    # the wrong behaviour under every affected name.
    assert {row.name: row.run.__name__ for row in rows.REGISTRY} == {
        "cross-thread-awareness": "row_cross_thread_awareness",
        "verification-rule": "row_verification_rule",
        "cross-thread-action": "row_cross_thread_action",
        "search-to-action": "row_search_to_action",
        "full-origin-fidelity": "row_full_origin_fidelity",
        "stream-currency": "row_stream_currency",
        "in-flight-exclusion": "row_in_flight_exclusion",
        "re-anchor-observable": "row_re_anchor_observable",
        "foreign-exchange-bait": "row_foreign_exchange_bait",
        "directed-banter-answered": "row_directed_banter_answered",
        "thanks-response-choice": "row_thanks_response_choice",
        "value-floor-holds": "row_value_floor_holds",
        "render-equality-probe": "row_render_equality_probe",
    }


def test_row_four_varies_its_subject_per_run():
    """THE CROSS-RUN CONTAMINATION (2026-08-04). The channel deletes nothing, so a question worded
    identically every run is answerable from the last run's material — which is exactly what
    happened: the seed-time turn answered one trial's question with the previous trial's figure.

    Two properties make the fix real. The draw is DETERMINISTIC, or the report's nonce stops
    reproducing the sentences the run posted and a failure cannot be read back. And it genuinely
    VARIES across nonces, or the repair is a comment. Nothing here touches Slack."""
    assert rows.row4_subject("n-1") == rows.row4_subject("n-1")
    drawn = {rows.row4_subject(f"nonce-{i}") for i in range(200)}
    assert len(drawn) > 1, "the subject never moves — a stored answer still answers next run"
    assert drawn <= set(rows.ROW4_SUBJECTS)
    # The pair moves TOGETHER: a run asking about the storage rate asks it about the warehouse,
    # in the buried question and in the news that settles it.
    for subject, work in rows.ROW4_SUBJECTS:
        assert subject and work and subject != work


@pytest.mark.parametrize("reply,offending", [
    ("I don't know — nobody has given us that number here.", None),
    # THE ONE FIGURE THAT IS ALLOWED. The quote is stated in the question, so repeating it is an
    # echo of the ask, not an answer to it — and the honest seed-time reply often does exactly
    # this. Failing it would turn every good premise into an invalid one.
    ("I can't see a cap for the year. They quoted $9,400 for it, but that's all I have.", None),
    # THE SHAPE THAT CONTAMINATED THE ROW: a price from SOMEWHERE ELSE — the previous run's
    # trigger, still sitting in a channel that deletes nothing — closing the obligation before
    # the graded trigger ever runs. The old guard compared against this run's cap only and let it
    # straight through.
    ("Looks like $79,822 a year, from what was posted earlier.", "$79,822"),
    ("The cap is $79,822; the $9,400 quote is inside it.", "$79,822"),
    ("It's $41770.00 for the year.", "$41770.00"),
    # Not cap-shaped: the row must not error on ordinary prose that happens to contain digits.
    ("Nothing on record as of 2026 — ask finance?", None),
    ("", None),
])
def test_row_four_premise_guard_catches_any_price_but_the_quote(reply, offending):
    assert rows.stale_cap_figure(reply, quoted="$9,400") == offending


@pytest.mark.parametrize("case,expected", [
    ("positive", True),
    ("a-wrong-root", False),
    ("b-natural-in-thread-answer", True),
    ("c-no-tool", False),
    ("h-second-post-elsewhere", False),
    ("i-words-under-the-trigger", False),
    ("j-words-in-the-channel", False),
    ("k-post-never-committed", False),
    ("l-no-cap", False),
    ("m-reaction-in-the-origin", True),
    ("n-brief-ack-in-the-origin", True),
    ("o-ack-that-reports-the-post", False),
    ("p-ack-carrying-the-figure", False),
    ("q-ack-grown-into-a-message", False),
])
def test_row_four_grader_against_fixtures(case, expected):
    """T122, second half. Row 4's predicate is the one most easily satisfied by an implementation
    that only LOOKS like it works, so each mutation must bite.

    THE PAIR CLAUSES ARE GONE (first live run, 2026-08-03): the landing under a root reachable
    only through this turn's search is the read-proof, and case b pins the natural in-thread
    answer — the exact reply the first live sample posted, cap and nothing else — as a PASS. The
    old pair cases (name-only, figure-only, half-the-name) graded restatement and died with the
    clauses that wanted it.

    CASES h–l ARE THE 2026-08-04 ORACLE (codex r2 #1/#2). The predicate used to count
    destinations already filtered to `{C}`, so h and i and j all scored as passes: one post in the
    right thread said nothing about a second post elsewhere or a running commentary in the origin.
    And l is the redesign's own clause — the cap is the only thing in a correct post that can have
    come from the trigger, so a post without it has repeated the question rather than answered it.

    CASES n–q ARE THE ACK RULING (owner, 2026-08-03): a brief non-reporting acknowledgment in the
    origin passes, and reporting stems, digits, and the full-match receipt grammar are each shown
    to bite on their own.
    """
    fact = bh.vendor_name("search-to-action-4f2a1b9c", "issuer")
    cap = bh.money("search-to-action-4f2a1b9c", "renewals-cap")
    target = "100.0001"
    origin = "300.0003"
    answer = f"finance settled it — the renewals cap is {cap} a year, effective now"
    kwargs: Dict[str, Any] = {
        "fact": fact,
        "cap": cap,
        "target_root": target,
        "origin_root": origin,
        "outcome_kind": "detached",
        "destinations": [{"kind": "post_to_thread", "thread_root_ts": target,
                          "state": "committed"}],
        "provenance": bh.ProvenanceRead(True, ("fetch_channel_info", "search_slack",
                                               "create_image_asset")),
        "posted_text": answer,
    }
    if case == "a-wrong-root":
        # A SIBLING thread, not {C}: `destinations[]` is inspected, never merely non-empty.
        kwargs["destinations"] = [{"kind": "post_to_thread", "thread_root_ts": "200.0002",
                                   "state": "committed"}]
    elif case == "b-natural-in-thread-answer":
        # THE FIRST LIVE SAMPLE'S SHAPE, verbatim in structure: the number the thread was waiting
        # on and not one word of the asker's own supplier or quote restated back at them.
        kwargs["posted_text"] = f"Finance has now set the annual rate at {cap}, effective now."
    elif case == "c-no-tool":
        kwargs["provenance"] = bh.ProvenanceRead(True, ("create_image_asset",))
    elif case == "h-second-post-elsewhere":
        # FAN-OUT THAT LANDED. The right thread got the answer and so did a stranger's.
        kwargs["destinations"] = [
            {"kind": "post_to_thread", "thread_root_ts": target, "state": "committed"},
            {"kind": "post_to_thread", "thread_root_ts": "200.0002", "state": "committed"}]
    elif case == "i-words-under-the-trigger":
        kwargs["outcome_kind"] = "reply"
        kwargs["destinations"] = kwargs["destinations"] + [
            {"kind": "reply", "thread_root_ts": origin, "state": "committed"}]
    elif case == "j-words-in-the-channel":
        # Origin prose that is NOT under the trigger's root — a top-level reply carries no
        # `thread_root_ts` to sweep, which is why the turn's own kind is graded beside it.
        kwargs["outcome_kind"] = "reply"
    elif case == "k-post-never-committed":
        kwargs["destinations"] = [{"kind": "post_to_thread", "thread_root_ts": target,
                                   "state": "observed"}]
    elif case == "l-no-cap":
        # An answer-shaped post with the one thing that can only have come from the trigger
        # missing — the obligation was not discharged, however right the thread.
        kwargs["posted_text"] = "finance settled it, number to follow"
    elif case == "m-reaction-in-the-origin":
        # A PASS, and the reason the clause is about WORDS. The conduct paragraph offers a
        # reaction as the ending, so an empty response that posts and reacts is a correct turn —
        # and it reports `reaction_only`, because that branch is taken before the committed
        # detached surface is consulted. Requiring `detached` alone would have failed the very
        # ending the prompt recommends.
        kwargs["outcome_kind"] = "reaction_only"
    elif case == "n-brief-ack-in-the-origin":
        # THE OWNER'S 2026-08-03 RULING, as a fixture: the measured "Got it — thanks." is the
        # model being a teammate, and the grader passes it. This is the exact shape case i used
        # to fail — a `reply` with words at the origin — with the one difference that the words
        # survive `origin_ack_violation`.
        kwargs["outcome_kind"] = "reply"
        kwargs["destinations"] = kwargs["destinations"] + [
            {"kind": "reply", "thread_root_ts": origin, "state": "committed"}]
        kwargs["origin_text"] = "Got it — thanks."
    elif case == "o-ack-that-reports-the-post":
        # The boundary of the ruling: words that depend on the post existing are reporting, and
        # the license is exactly for words that would read the same if it never happened.
        kwargs["outcome_kind"] = "reply"
        kwargs["destinations"] = kwargs["destinations"] + [
            {"kind": "reply", "thread_root_ts": origin, "state": "committed"}]
        kwargs["origin_text"] = "Thanks — posted it in the infra thread."
    elif case == "p-ack-carrying-the-figure":
        # A figure in the origin is content, not acknowledgment — the answer leaking back.
        kwargs["outcome_kind"] = "reply"
        kwargs["destinations"] = kwargs["destinations"] + [
            {"kind": "reply", "thread_root_ts": origin, "state": "committed"}]
        kwargs["origin_text"] = f"Got it — {cap} noted, thanks."
    elif case == "q-ack-grown-into-a-message":
        # No stems and no digits, but free prose all the same — the full-match grammar is what
        # keeps "brief" from meaning "anything without a keyword in it".
        kwargs["outcome_kind"] = "reply"
        kwargs["destinations"] = kwargs["destinations"] + [
            {"kind": "reply", "thread_root_ts": origin, "state": "committed"}]
        kwargs["origin_text"] = ("Really appreciate you flagging this so quickly — it helps a "
                                 "lot to know the finance side has finally settled on what we "
                                 "were all waiting for, and I will make sure the right people "
                                 "hear about it as soon as possible.")

    graded = rows.grade_search_to_action(**kwargs)
    assert len(graded) == 5
    assert all(ok for _, ok in graded) is expected


@pytest.mark.parametrize("text,clean", [
    # The two acks the live trials actually produced — the ruling's own examples.
    ("Got it — thanks.", True),
    ("👍", True),                                   # no letters: an emoji line IS a receipt
    ("", True),                                     # silence is not a violation, it is the default
    ("thanks for the update!", True),               # about the NEWS — "update" is not a stem
    ("appreciate it, good to know", True),
    ("Done.", False),                               # the measured regression, by name
    ("Answered in the other thread.", False),
    ("thanks — I replied over there", False),
    ("posted!", False),
    ("noted, the cap is $84,200", False),           # a digit is content, wherever it hides
    # Codex's ack-round-1 counterexamples: reporting that names no post, no thread, no reply —
    # first-person past action and "it's handled" shapes have to fail on their own stems.
    ("Sent it along — thanks.", False),
    ("All sorted — thanks.", False),
    ("It's with them now.", False),
    ("Took care of it — appreciate the update.", False),
    # Codex's ack-round-2 counterexamples: a receipt WORD wrapped around free prose. Token
    # presence is not a grammar — every word has to be part of a receipt phrase.
    ("Got it — I put it where it belongs.", False),
    ("Thanks — they have it now.", False),
    ("Got it — all set.", False),
    ("Appreciate the update — that's been dealt with.", False),
    # Non-Latin prose is words, not an emoji line (codex ack-round-2): Unicode word extraction
    # sends it through the grammar, where it fails like any other free prose.
    ("Спасибо, всё готово.", False),
    ("🙏", True),
    # A @-mention is address, not content — stripped before the grammar, but only in ADDRESS
    # positions (codex ack #13): after a finished receipt phrase or at a vocative boundary, never
    # inside a clause, where it would launder the very content the checks exist to catch.
    ("thanks <@U123ABCDE> — got it", True),
    ("Thanks, <@U123ABCDE>!", True),
    ("Thanks for <@U123ABCDE> closing the loop.", False),
    ("Thanks for checking <@U123ABCDE>.", False),
    # Codex ack-round-3's four neighbors: wordless text passes only when an emoji is actually
    # there to do the acknowledging — Slack transports emoji as :codes:, and :+1:'s digit is the
    # emoji's, not the answer's.
    (":wave:", True),
    (":+1:", True),
    ("...", False),
    ("<@U123ABCDE>", False),
    # Codex ack-round-4: a colon costume is not an emoji. Only vouched aliases count, so the
    # laundering shapes fail before anything is stripped…
    (":done:", False),
    (":posted:", False),
    (":90742:", False),
    # …and emoji are counted: a wall of :wave: is noise, not a receipt.
    (":wave: :wave: :wave: :wave:", False),
    # Codex ack-round-5: category So is not an emoji test. ℗ is So and acknowledges nothing;
    # ❤️ (heart + VS16, two codepoints) is ONE vouched emoji; a multi-person ZWJ cluster is
    # unvouched So characters and fails on the strict side.
    ("℗", False),
    ("❤️", True),
    ("👍🏽", True),                                  # skin-tone modifier is Sk, rides along
    ("👨‍👩‍👧‍👦", False),
    ("thanks ⌘", False),
    # Codex ack-round-6: representation parity — the same acknowledgment grades the same as an
    # alias and as its rendered character.
    ("♥️", True),
    ("✔️", True),
    ("🙇", True),
    # Codex ack-round-7's external golden: Slack's :ok: renders 🆗 (and :ok_hand: is 👌) — a
    # parity test over the mapping cannot see a wrong mapping, so the real renderings are pinned.
    ("🆗", True),
    ("👌", True),
    # The pair is a structure, not a count of two: different phrases compose, repetition fails.
    ("got it thanks", True),
    ("thanks thanks", False),
    ("finally finally", False),
    ("thanks thanks thanks thanks", False),
    # MEASURED IN (the first strict-predicate recording run failed all three of these real
    # trials): gratitude with a reason is a receipt, because a "thanks for" tail describes the
    # OTHER person's act and a report of the bot's own post cannot take that shape. The stems
    # still screen every tail word.
    ("Got it — thanks for closing the loop.", True),
    ("Perfect, thanks for checking.", True),
    ("thanks for the quick turnaround", True),
    ("thanks for posting it over there", False),    # the tail is screened — stem fires
    ("thanks for going through every single one of those old piles for me", False),  # a message,
                                                    # not a modifier phrase — the tail is bounded
    # Codex ack-rounds-9/10: no grammatical screen proves a free tail is not about the post —
    # embedded actors, passives, and nominals all smuggle event-dependence in — so the reason is
    # a vouched set and everything outside it fails.
    ("Thanks for letting me put that there.", False),
    ("Perfect, thanks for letting me handle that.", False),
    ("Thanks for what we did.", False),
    ("Thanks for work we completed.", False),
    ("Thanks for getting it placed.", False),
    ("Thanks for successful placement.", False),
    ("Thanks for the completed handoff.", False),
    ("thanks for letting me know", True),           # the exact idiom stays a phrase, untouched
    # THE STRICT SIDE, ACCEPTED ON PURPOSE: plausibly innocent, but a full-match grammar cannot
    # tell it from reporting, and a false failure is reviewable where a false pass is invisible.
    ("Thanks for answering!", False),
    ("Thanks, Priya!", False),                      # plain-name address WITHOUT the addressees
                                                    # evidence; see the addressees test below
    ("Happy to help with the renewals question.", False),
])
def test_origin_ack_violation_draws_the_ruled_line(text, clean):
    """The shared predicate behind scenario-oracle clause 5 and live row 4's origin clause. Both
    surfaces import THIS function, so this is the one place the ack line is drawn."""
    assert (bh.origin_ack_violation(text) is None) is clean


def test_ack_emoji_aliases_and_characters_are_one_mapping():
    """Codex ack-round-6: the alias set and the rendered set derive from ONE mapping, and every
    entry acknowledges in BOTH representations — `:hearts:` and ♥️ are the same receipt, and a
    vouched base character must actually be reachable (category So), or it would be invisible to
    the counter and fail as 'no words and no emoji'."""
    for alias, char in bh._ACK_EMOJI.items():
        assert bh.origin_ack_violation(f":{alias}:") is None, alias
        assert bh.origin_ack_violation(char) is None, (alias, char)
        assert bh.origin_ack_violation(char + "️") is None, (alias, char)


def test_ack_addressee_names_are_address_not_content():
    """MEASURED IN: a real trial wrote "Thanks, Tessa." and the grammar called the name free
    prose. With the caller's addressees evidence the name strips like a mention — full display
    names and their word parts, word-bounded — and without it the words still fail (the predicate
    cannot invent who is in the room)."""
    assert bh.origin_ack_violation("Thanks, Tessa.", addressees=("Tessa Tran",)) is None
    assert bh.origin_ack_violation("Thanks, Tessa Tran — got it.",
                                   addressees=("Tessa Tran",)) is None
    assert bh.origin_ack_violation("Thanks, Tessa.") is not None
    # A name is address only — it cannot vouch for the free prose beside it.
    assert bh.origin_ack_violation("Tessa handled it.", addressees=("Tessa Tran",)) is not None
    # And only in VOCATIVE position (codex ack #11/#12): a name inside a clause is content, so
    # event-dependent praise of the bot cannot strip itself into a receipt, and a sentence-final
    # direct object is not a vocative either.
    assert bh.origin_ack_violation("Thanks for ChatGPT closing the loop.",
                                   addressees=("ChatGPT", "Tessa Tran")) is not None
    assert bh.origin_ack_violation("Thanks for checking Tessa.",
                                   addressees=("Tessa Tran",)) is not None
    assert bh.origin_ack_violation("Tessa, thanks.", addressees=("Tessa Tran",)) is None
    # Mentions follow the same rules, plus caller-vouched IDS (codex ack #13): with ids given,
    # a mention of anyone else — the bot included — is a violation even in vocative position.
    assert bh.origin_ack_violation("Thanks, <@U111HUMAN>!",
                                   addressee_ids=("U111HUMAN",)) is None
    assert bh.origin_ack_violation("Thanks, <@UBOT99999>!",
                                   addressee_ids=("U111HUMAN",)) is not None
    # The scenario corpus's readable synthetic ids are mentions too (codex ack #14) — the
    # predicate must recognize its callers' own fixture shape, not only production Slack ids.
    assert bh.origin_ack_violation("Thanks, <@U-tessa>!",
                                   addressee_ids=("U-tessa",)) is None
    assert bh.origin_ack_violation("thanks <@U-tessa> — got it",
                                   addressee_ids=("U-tessa",)) is None


def test_reply_chrome_strip_is_structural_not_gear_adjacent():
    """Codex ack-round-3 #3: the chrome strip must remove exactly the transported settings-button
    suffix (gear + model label + 'button', at end of text) and never rewrite an answer that
    happens to mention a gear or a button."""
    def _obs(text):
        return bh.Observed(ts="1.0", text=text, thread_ts="1.0", channel="C1",
                           user="U1", bot_id="B1")

    assert rows._text_of([_obs("Got it, thanks. :gear: gpt-5.6-sol button")]) == "Got it, thanks."
    assert rows._text_of([_obs("Finance set it at $75,798. :gear: gpt-5.5 button")]) == (
        "Finance set it at $75,798.")
    for untouched in ("Use the :gear: settings button to change models",
                      # End-position, gpt-prefixed, and still not chrome: the label shape is
                      # gpt-<digit>… (codex ack #4), so a prose "gpt-settings" survives.
                      "Use the :gear: gpt-settings button"):
        assert rows._text_of([_obs(untouched)]) == untouched


def test_origin_ack_violation_catches_the_answer_fragments_it_is_told_about():
    """The digit rule cannot see a supplier's NAME travel, so the caller hands the predicate the
    non-numeric halves of its seeded answer (codex, ack round #1). The full-match grammar refuses
    the same words anyway; the fragment guard's job is to make the finding NAME the leak — and it
    is word-bounded via `states_phrase`, so a name inside another word does not fire it."""
    supplier = "Kestwood Freight"
    named = bh.origin_ack_violation(f"thanks — {supplier} confirmed it", fragments=(supplier,))
    assert named is not None and supplier in named
    assert bh.origin_ack_violation("thanks, got it", fragments=(supplier,)) is None
    # Without the caller's evidence the words still fail — as free prose, not as a named leak.
    unnamed = bh.origin_ack_violation(f"thanks — {supplier} confirmed it")
    assert unnamed is not None and supplier not in unnamed
    # Word-bounded: the fragment does not fire inside other words; the grammar still refuses the
    # free prose around it.
    bounded = bh.origin_ack_violation("thanks, NotKestwood Freightening again",
                                      fragments=(supplier,))
    assert bounded is not None and supplier not in bounded


@pytest.mark.parametrize("case,expected", [
    ("positive", True),
    ("a-no-tool", False),
    ("b-neither-half", False),
    ("c-stale-figure", False),
    ("d-punctuated-differently", True),
    ("e-zero-cents", True),
    ("f-non-zero-cents", False),
    ("g-longer-number", False),
    ("h-denial-sentence-about-something-else", True),
    ("i-supplier-only", False),
    ("j-figure-only", False),
])
def test_row_two_grader_against_fixtures(case, expected):
    """T122. Row 2's contract, exercised network-free — "a tool ran and it said something" is
    what the row used to accept, and it is not the contract.

    IT GRADES THE PAIR (codex, verify-7): which supplier, and what figure. The trigger names
    neither — it asks about the WORK — so both have to come back from the one sentence below the
    floor, and a single half that surfaced by coincidence out of an accumulating channel is not
    enough.

    THE FIGURE IS DIGIT-NORMALIZED, including a zero-cents tail: a model writing `$41,770.00` has
    named the seeded decision. `41,770.50` has not — that is a different figure — and a number the
    seeded digits merely sit inside is a different number too.

    AND NO PROSE IS SCANNED. The retired denial-marker list failed a correct answer that said "no
    mention of" about something the question did not ask for."""
    decision = rows.verification_decision("verification-rule-4f2a1b9c")
    supplier = bh.vendor_name("verification-rule-4f2a1b9c")
    digits = bh.digits_of(decision)
    kwargs = {
        "decision": decision,
        "supplier": supplier,
        "reply_text": f"we went with {supplier} at {decision} in the end",
        "provenance": bh.ProvenanceRead(True, ("search_slack",)),
    }
    if case == "a-no-tool":
        kwargs["provenance"] = bh.ProvenanceRead(True, ("create_image_asset",))
    elif case == "b-neither-half":
        kwargs["reply_text"] = "I had a look — you went with the cheaper of the two quotes"
    elif case == "c-stale-figure":
        # A PREVIOUS run's value, still sitting in the channel's history.
        kwargs["reply_text"] = f"{supplier} quoted {rows.verification_decision('x')}"
    elif case == "d-punctuated-differently":
        kwargs["reply_text"] = f"{supplier}, and the agreed figure was {digits} even"
    elif case == "e-zero-cents":
        kwargs["reply_text"] = f"{supplier} — the agreed figure was {decision}.00"
    elif case == "f-non-zero-cents":
        kwargs["reply_text"] = f"{supplier} — the agreed figure was {decision}.50"
    elif case == "g-longer-number":
        kwargs["reply_text"] = f"{supplier} — the agreed figure was 9{digits}"
    elif case == "h-denial-sentence-about-something-else":
        # THE CODEX CASE. True, helpful, and the retired prose scan failed it.
        kwargs["reply_text"] = (f"No mention of a delivery date, but the accepted quote was "
                                f"{decision} from {supplier}")
    elif case == "i-supplier-only":
        kwargs["reply_text"] = f"we went with {supplier} in the end"
    elif case == "j-figure-only":
        kwargs["reply_text"] = f"the agreed figure was {decision}"

    graded = rows.grade_verification(**kwargs)
    assert len(graded) == 3
    assert all(ok for _, ok in graded) is expected


def test_the_verification_decision_is_run_unique():
    """T122. A permanent literal is guessable AND survives in history from every previous run."""
    assert rows.verification_decision("a") != rows.verification_decision("b")
    assert rows.verification_decision("a") == rows.verification_decision("a")
    # It reads as a price a coworker would write, not as a marker: currency, thousands separator,
    # and digits that carry the uniqueness.
    assert rows.verification_decision("a").startswith("$")
    assert len(bh.digits_of(rows.verification_decision("a"))) >= 5


@pytest.mark.parametrize("render,expected", [
    ({"periphery_floor_ts": "500.0", "anchor_advanced": True, "selection_version": 3},
     "500.0|3"),
    # RESELECTED BUT DID NOT WIN THE CAS: another builder's write landed, and undoing it would be
    # the battery corrupting live selection state while tidying up after itself.
    ({"periphery_floor_ts": "500.0", "anchor_advanced": False, "selection_version": 3}, None),
    ({"periphery_floor_ts": "", "anchor_advanced": True, "selection_version": 3}, None),
])
def test_row_eight_registers_a_restore_only_when_its_write_landed(render, expected):
    """T126. `reselected` is a CHOICE; `anchor_advanced` is the write that landed."""
    restore = rows.anchor_restore_for(render, ("100.0", 2))
    assert (restore.key if restore else None) == expected
    if restore is not None:
        assert restore.kind == "window_anchor" and restore.prior == ("100.0", 2)


# ------------------------------------------------------------------------------------- T125

async def test_the_pollers_use_the_complete_walk(clock, install, monkeypatch):
    """T125. Asserted by CALL-THROUGH against a fake, not by testing the helper in isolation."""
    seen: List[Dict[str, Any]] = []

    async def _complete(channel, root_ts, *, oldest=None, latest=None):
        seen.append({"channel": channel, "root_ts": root_ts, "oldest": oldest})
        return [bot_msg("9.0001")] if len(seen) >= 2 else []

    # The helper's signature ACCEPTS `oldest` and FORWARDS it to conversations.replies — checked
    # against the REAL helper, before the call-through fake replaces it.
    pair = install(bot=FakeSlack(conversations_replies=replies_page([])))
    await bh.fetch_thread_complete("C1", "1.0000", oldest="1.0000")
    assert pair.bot.method_calls("conversations_replies")[0]["oldest"] == "1.0000"

    monkeypatch.setattr(bh, "fetch_thread_complete", _complete)
    found = await bh.wait_bot_reply("C1", "1.0000", "1.0000", author=OURS, deadline=60, poll=5)
    assert [o.ts for o in found] == ["9.0001"]
    assert len(seen) == 2 and all(call["oldest"] == "1.0000" for call in seen)

    # A BROADCAST of a thread reply appears in the top-level feed but is NOT a top-level message,
    # so counting it would pass the wrong row.
    page = {"ok": True, "messages": [bot_msg("9.0002", subtype="thread_broadcast"),
                                     bot_msg("9.0003")],
            "response_metadata": {"next_cursor": ""}}
    install(bot=FakeSlack(conversations_history=page))
    top = await bh.wait_bot_reply_channel("C1", "1.0000", author=OURS, deadline=0)
    assert [o.ts for o in top] == ["9.0003"]

    # LEXICAL ORDERING IS NOT TIMESTAMP ORDERING. "1000.000001" sorts BEFORE "999.999999" as a
    # string, so a lexical poller drops the newer message and reports that the bot said nothing.
    # The production contract rejects string ordering for exactly this reason.
    assert bh.ts_lt("999.999999", "1000.000001")
    assert not bh.ts_lt("1000.000001", "999.999999")
    install(bot=FakeSlack(conversations_history={"ok": True,
                                                 "messages": [bot_msg("1000.000001")]}))
    crossing = await bh.wait_bot_reply_channel("C1", "999.999999", author=OURS, deadline=0)
    assert [o.ts for o in crossing] == ["1000.000001"]

    only_broadcast = {"ok": True, "messages": [bot_msg("9.0002", subtype="thread_broadcast")]}
    install(bot=FakeSlack(conversations_history=only_broadcast))
    assert await bh.wait_bot_reply_channel("C1", "1.0000", author=OURS, deadline=0) == []


# ------------------------------------------------------------------------------------- T126

@pytest.fixture
def bot_db(monkeypatch, tmp_path):
    """A stand-in for the bot's own database, with the tables the harness reads."""
    path = tmp_path / "slack.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE outbound_receipts (team_id TEXT, channel_id TEXT, "
                 "message_ts TEXT, turn_id TEXT, state TEXT, thread_root_ts TEXT)")
    conn.execute("CREATE TABLE channel_window_anchor (team_id TEXT, channel_id TEXT, "
                 "floor_ts TEXT, selection_version INTEGER)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(bh, "db_path", lambda: path)
    return path


def _insert_receipts(path, rows_, turn_id="TURN-A", team="T1", channel="C1"):
    conn = sqlite3.connect(str(path))
    conn.executemany("INSERT INTO outbound_receipts VALUES (?, ?, ?, ?, ?, ?)",
                     [(team, channel, ts, turn_id, state, root) for ts, state, root in rows_])
    conn.commit()
    conn.close()


@pytest.fixture
def settings_db(monkeypatch, tmp_path):
    path = tmp_path / "slack.db"
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE channel_settings (channel_id TEXT PRIMARY KEY, model TEXT, "
                 "verbosity TEXT)")
    conn.execute("INSERT INTO channel_settings VALUES ('C1', 'gpt-5.6-terra', 'high')")
    conn.commit()
    conn.close()
    monkeypatch.setattr(bh, "db_path", lambda: path)
    return path


def _read_settings(path: Path):
    conn = sqlite3.connect(str(path))
    try:
        return conn.execute("SELECT model, verbosity FROM channel_settings "
                            "WHERE channel_id = 'C1'").fetchone()
    finally:
        conn.close()


async def test_restore_distinguishes_null_from_absent(clock, install, settings_db):
    """T126. `channel_settings` columns are nullable and NULL MEANS "inherit the global default"
    — a real, intended value. Restoring that by DELETING the row would erase every OTHER setting
    on the channel."""
    bh.apply_restore(bh.Restore(kind="channel_setting", key="model", prior=None, existed=True),
                     "C1")
    # NULL was written, and the SIBLING COLUMN SURVIVED — which is what a delete would have lost.
    assert _read_settings(settings_db) == (None, "high")

    bh.apply_restore(bh.Restore(kind="channel_setting", key="model", prior="x", existed=False),
                     "C1")
    assert _read_settings(settings_db) is None

    # A restore that RAISES lands in restore_failures and downgrades the row to `unrestored`.
    install()
    ctx = bh.RowContext(row="r", nonce="n", channel="C1")
    ctx.assert_that("everything held", True)
    ctx.restores.append(bh.Restore(kind="channel_setting", key="model", prior="x", existed=True))
    result = await bh.cleanup_row(ctx)
    assert result.restore_failures == ("channel_setting:model",)
    assert bh.status_for(ctx, result) == "unrestored"


async def test_the_window_anchor_restore_compares_before_it_writes(clock, install, bot_db):
    """T126, the fourth kind. Row 8 advances durable selection state, so it MUST restore — but
    never over a legitimate advance, and never on a partial match.

    THE COMPARE IS OVER THE COMPLETE TUPLE. Two builds can land the same `floor_ts` under
    different selection versions, so a floor-only compare would mistake another build's anchor
    for its own and overwrite it.
    """
    conn = sqlite3.connect(str(bot_db))
    conn.execute("INSERT INTO channel_window_anchor VALUES ('T1', 'C1', '500.0000', 3)")
    conn.commit()
    conn.close()

    install()

    async def _restore(key):
        ctx = bh.RowContext(row="re-anchor-observable", nonce="n", channel="C1", team_id="T1")
        ctx.assert_that("everything held", True)
        ctx.restores.append(bh.Restore(kind="window_anchor", key=key,
                                       prior=("100.0000", 2), existed=True))
        return ctx, await bh.cleanup_row(ctx)

    # SAME FLOOR, DIFFERENT VERSION: another build's anchor, and we must not touch it.
    ctx, result = await _restore("500.0000|9")
    assert result.restore_failures == ("window_anchor:500.0000|9",)
    assert _anchor(bot_db) == ("500.0000", 3)

    ctx, result = await _restore("500.0000|3")
    assert result.restored == ("window_anchor:500.0000|3",)
    assert _anchor(bot_db) == ("100.0000", 2)

    # THE COMPARE LOSES: a legitimate turn advanced the anchor after us. We do NOT overwrite it,
    # and the row reports `unrestored` rather than claiming a restore it did not make.
    ctx2, result2 = await _restore("500.0000|3")
    assert result2.restore_failures == ("window_anchor:500.0000|3",)
    assert bh.status_for(ctx2, result2) == "unrestored"
    assert _anchor(bot_db) == ("100.0000", 2)


def _anchor(path):
    conn = sqlite3.connect(str(path))
    try:
        row = conn.execute("SELECT floor_ts, selection_version FROM channel_window_anchor "
                           "WHERE team_id = 'T1' AND channel_id = 'C1'").fetchone()
    finally:
        conn.close()
    return (row[0], row[1]) if row else None


# ------------------------------------------------------------------------------------- T127

@pytest.fixture
def preflight(monkeypatch, install):
    """Our own identity from `auth.test`, and a `bots.info` that answers PER BOT ID.

    Dispatching on the `bot=` argument is required, not convenience: the preflight now partitions
    the allowlist by app, so a fixture returning one canned record for every lookup would make our
    own bot-token record and Claude Tag indistinguishable and the partition untestable.
    """
    def _setup(allowlist: str, bots_info=None):
        monkeypatch.setenv("DEV_TREAT_BOT_IDS_AS_HUMAN", allowlist)

        def _lookup(bot: str, **_: Any) -> Dict[str, Any]:
            record = _BOT_RECORDS.get(bot)
            return {"ok": True, "bot": dict(record)} if record else {"ok": True}

        fake = FakeSlack(auth_test={"ok": True, "bot_id": OURS.bot_id, "user_id": OURS.user_id,
                                    "team_id": "T_DEV"},
                         bots_info=bots_info if bots_info is not None else _lookup)
        install(user=fake, bot=fake)
        return fake
    return _setup


async def test_the_allowlist_preflight_reads_and_asserts(clock, preflight, monkeypatch):
    """T127. The allowlist is PRECONFIGURED bot state; the preflight reads and asserts it.

    CONTRACT CHANGED 2026-08-02, and the change is the point of the test. The old version required
    `auth.test`'s bot_id — the BOT-TOKEN record — to be allowlisted, on the belief that seeds carry
    it. They carry the app's USER-TOKEN record instead. Satisfying the old check meant removing the
    id the carve-out needs; every seed then classified as `other_bot` and a whole battery pass was
    graded against a bot-classified operator while the preflight reported clean. So: the partition
    is now BY APP, the required entry is the user-token record, and the bot-token record is not
    required at all (it is inert in the allowlist — `is_own_message` matches it first).
    """
    # (1) THE CORRECT CONFIGURATION PASSES — and it does NOT name the bot-token record.
    fake = preflight(f"{SEED_RECORD},{CLAUDE.bot_id}")
    identity = await bh.assert_claude_tag_allowlisted()
    assert identity == CLAUDE

    # COST AND NON-REPETITION: one lookup to learn our app_id, then one per unique entry. The
    # survivor is NOT resolved a second time — an implementation that re-derives it after the
    # partition would spend four.
    looked_up = [call["bot"] for call in fake.method_calls("bots_info")]
    assert looked_up == [OURS.bot_id, SEED_RECORD, CLAUDE.bot_id]

    # (2) THE REGRESSION, NAMED. This is the exact 2026-08-01 misconfiguration: the bot-token id in
    # place of the user-token one. It must raise, and the message must say which id is missing —
    # this is the single check whose absence corrupts every row rather than failing one.
    preflight(f"{OURS.bot_id},{CLAUDE.bot_id}")
    with pytest.raises(bh.HarnessPreflightError, match="user-token posting record") as caught:
        await bh.assert_claude_tag_allowlisted()
    assert "auth.test returns" in str(caught.value)

    # (3) THE PAIR: the allowlist supplies the bot_id, bots.info supplies the bot-USER id, and a
    # fixture where the two differ catches any code that swaps them.
    assert identity.bot_id != identity.user_id

    # …and `author=` matching accepts a raw `user` match OR a raw `bot_id` match, both directions,
    # since Slack fills the two fields differently depending on how a message was posted.
    assert bh.identity_matches(CLAUDE.user_id, None, CLAUDE)
    assert bh.identity_matches(None, CLAUDE.bot_id, CLAUDE)
    assert not bh.identity_matches(OURS.user_id, OURS.bot_id, CLAUDE)

    # (4) CARDINALITY, counted AFTER the partition: zero and two foreign entries each raise, naming
    # them. A DUPLICATE is a typo in a comma-separated env var, not a second party — and neither is
    # our own second record, which is why listing BOTH of ours still leaves exactly one.
    preflight(SEED_RECORD)
    with pytest.raises(bh.HarnessPreflightError, match="EXACTLY ONE"):
        await bh.assert_claude_tag_allowlisted()
    preflight(f"{SEED_RECORD},{CLAUDE.bot_id},B_THIRD")
    with pytest.raises(bh.HarnessPreflightError) as caught:
        await bh.assert_claude_tag_allowlisted()
    assert "B_THIRD" in str(caught.value)
    preflight(f"{SEED_RECORD},{CLAUDE.bot_id},{CLAUDE.bot_id}")
    assert await bh.assert_claude_tag_allowlisted() == CLAUDE
    preflight(f"{OURS.bot_id},{SEED_RECORD},{CLAUDE.bot_id}")
    assert await bh.assert_claude_tag_allowlisted() == CLAUDE

    # (5) AN EMPTY ALLOWLIST is the same corruption stated the shortest way, and costs zero calls.
    fake = preflight("")
    with pytest.raises(bh.HarnessPreflightError, match="empty"):
        await bh.assert_claude_tag_allowlisted()
    assert fake.method_calls("bots_info") == []

    # (6) TWO WAYS AN ENTRY CAN BE UNUSABLE, and they are different failures. An id this workspace
    # does not know at all has no record to partition by; a real record with no bot-USER id cannot
    # be mentioned, and the literal text '@Claude' mentions nobody. Neither may return half a pair.
    preflight(f"{SEED_RECORD},B_UNKNOWN")
    with pytest.raises(bh.HarnessPreflightError, match="no bot record"):
        await bh.assert_claude_tag_allowlisted()
    preflight(f"{SEED_RECORD},B_MUTE")
    with pytest.raises(bh.HarnessPreflightError, match="no user_id"):
        await bh.assert_claude_tag_allowlisted()

    # (7) THE SYNTHETIC NEGATIVE, kept deliberately. Because the id is DERIVED from the allowlist,
    # "derived id not in allowlist" is unreachable in production — so it is forced by mutating the
    # list between the derive and the verify. It guards the derive-then-verify seam: an
    # implementation that derives and then TRUSTS passes every reachable case and fails this one.
    # DO NOT "simplify" this away as dead.
    preflight(f"{SEED_RECORD},{CLAUDE.bot_id}")
    answers = iter([[SEED_RECORD, CLAUDE.bot_id], [SEED_RECORD]])
    monkeypatch.setattr(bh, "allowlisted_bot_ids", lambda: next(answers))
    with pytest.raises(bh.HarnessPreflightError, match="not in DEV_TREAT_BOT_IDS_AS_HUMAN"):
        await bh.assert_claude_tag_allowlisted()

    # (8) IT WRITES NOTHING — no environment assignment, no settings write, no Restore entry. A
    # boot-loaded env var cannot be changed from this process, so a preflight that "set" the
    # allowlist here would pass a restore test while the bot never saw the value at all.
    preflight(f"{SEED_RECORD},{CLAUDE.bot_id}")
    monkeypatch.setattr(bh, "allowlisted_bot_ids", lambda: [SEED_RECORD, CLAUDE.bot_id])
    monkeypatch.setattr(bh.sqlite3, "connect",
                        lambda *a, **k: pytest.fail("the preflight wrote to the database"))
    before = dict(__import__("os").environ)
    ctx = bh.RowContext(row="foreign-exchange-bait", nonce="n", channel="C1")
    await bh.assert_claude_tag_allowlisted()
    assert dict(__import__("os").environ) == before
    assert ctx.restores == []


async def test_the_preflight_refuses_a_fenced_channel(clock, install, bot_db):
    """T127. "Unfenced" is a documented premise; the preflight is what makes it a FACT.

    THE LEASE MUST BE READ FROM THE DURABLE TABLE. `epoch_fence._FENCES` is a dict in the BOT's
    process — this harness is a different process and would read an empty one every time and
    conclude "no fence", the same mistake as a harness that "sets" a boot-loaded env var and then
    asserts it worked.
    """
    # No table at all: the fence watcher creates it, so its absence proves none has ever run.
    assert await bh.read_epoch_fence_lease("T1", "C1") is None
    await bh.assert_channel_unfenced("T1", "C1")

    conn = sqlite3.connect(str(bot_db))
    conn.execute("CREATE TABLE epoch_fence_lease (team_id TEXT, channel_id TEXT, lease_id TEXT, "
                 "state TEXT, expiry_ts TEXT, test_epoch_id TEXT)")
    conn.commit()
    conn.close()

    future = f"{time.time() + 600:.6f}"
    past = f"{time.time() - 600:.6f}"

    def _lease(state, expiry):
        c = sqlite3.connect(str(bot_db))
        c.execute("DELETE FROM epoch_fence_lease")
        c.execute("INSERT INTO epoch_fence_lease VALUES ('T1', 'C1', 'L-1', ?, ?, NULL)",
                  (state, expiry))
        c.commit()
        c.close()

    for state in ("armed", "active", "closing"):
        _lease(state, future)
        with pytest.raises(bh.HarnessPreflightError, match="fenced"):
            await bh.assert_channel_unfenced("T1", "C1")
        # EXPIRED, and therefore dead: a stale lease must not lock the channel out forever.
        _lease(state, past)
        await bh.assert_channel_unfenced("T1", "C1")

    # `invalidated` refuses WHATEVER its expiry says — it is cleared only by a human.
    _lease("invalidated", past)
    with pytest.raises(bh.HarnessPreflightError, match="INVALIDATED"):
        await bh.assert_channel_unfenced("T1", "C1")

    _lease("released", future)
    await bh.assert_channel_unfenced("T1", "C1")

    # IT FAILS CLOSED. A malformed lease is DAMAGED EVIDENCE, not proof the channel is free —
    # reading an unparseable expiry as "expired", or waving through a state this harness does not
    # recognise, turns a broken row into permission and silently grades an overlay.
    for state in ("armed", "active", "closing"):
        _lease(state, "not-a-timestamp")
        with pytest.raises(bh.HarnessPreflightError, match="unparseable"):
            await bh.assert_channel_unfenced("T1", "C1")
    _lease("some_future_state", past)
    with pytest.raises(bh.HarnessPreflightError, match="UNRECOGNISED"):
        await bh.assert_channel_unfenced("T1", "C1")


async def test_the_two_tokens_must_share_one_workspace(clock, install, monkeypatch):
    """T127. The battery's premise is that our seeds carry OUR app's allowlisted identity.

    Two tokens in two workspaces cannot satisfy it, and an empty user identity is worse than
    useless: it matches nothing, so every seed the harness posted would fall through the router
    into the third-party bucket and the report would attribute the run's own messages to
    somebody else.
    """
    install(user=FakeSlack(auth_test={"ok": True, "user_id": "U_HUMAN", "team_id": "T_OTHER"}),
            bot=FakeSlack(auth_test={"ok": True, "bot_id": OURS.bot_id,
                                     "user_id": OURS.user_id, "team_id": "T_DEV"}))
    with pytest.raises(bh.HarnessPreflightError, match="workspace"):
        await bh.harness_user_identity()

    install(user=FakeSlack(auth_test={"ok": True, "user_id": "", "team_id": "T_DEV"}),
            bot=FakeSlack(auth_test={"ok": True, "bot_id": OURS.bot_id,
                                     "user_id": OURS.user_id, "team_id": "T_DEV"}))
    with pytest.raises(bh.HarnessPreflightError, match="no user_id"):
        await bh.harness_user_identity()

