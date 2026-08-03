"""The live battery's entry point (SHALLOW_STREAM_RESPEC §7.1a, §9).

    python3 -m tests.live.run_battery [--rows a,b] [--out report.json]

It runs against REAL Slack with the dev bot, and only in the authorized test channel. Read
`tests/live/README.md` first — in particular the `DEV_TREAT_BOT_IDS_AS_HUMAN` prerequisite, which
must be set in the BOT's `.env` before the bot process starts. The harness cannot set it; a
separate process cannot change a boot-loaded env var.

THE EXIT CODE IS THE VERDICT. Nonzero unless every EXECUTED row is a clean `pass`:
`unrestored`, `fail` and `error` all exit 1 — the first because a row that changed durable bot
state and could not put it back has left the channel configured by the battery rather than by its
owner. A selection that skips rows still exits 0 if the ones it ran all passed; the operator asked
for a subset.

THE BATTERY DELETES NOTHING. Its messages stay in the channel by owner ruling — they are what the
owner watches and reads afterwards — so a run's only footprint outside the room is durable state,
and that is the one thing a row puts back.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Coroutine, Dict, List, Optional, Sequence

from tests.live.battery_harness import (CleanupResult, HarnessError, PartyIdentity, RowContext,
                                        assert_channel_unfenced, assert_claude_tag_allowlisted,
                                        bot_identity, bot_team_id, build_report, cleanup_row,
                                        harness_user_identity, mint_nonce, status_for,
                                        write_report)
from tests.live.battery_rows import REGISTRY, ROW_NAMES, BatteryRow, window_ceiling, window_target


def resolved_window() -> Dict[str, int]:
    """The window the HARNESS resolved, which is what every seeded count was computed from.

    It cannot read the bot's environment, so this is a record rather than a check: if the two
    processes were launched with different values the rows will disagree with the bot's renders —
    row 8's `root_count` assertion is the one that notices first.
    """
    return {"channel_window_target": window_target(),
            "channel_window_ceiling": window_ceiling()}

# The ONLY channel live testing is authorized in. Prod remains hands-off, and a battery that can
# be pointed anywhere by a flag with no default is a battery one typo away from seeding 101
# messages into a real conversation.
DEFAULT_CHANNEL = "C0BKX77NU66"

_EMPTY_CLEANUP = CleanupResult(restored=(), restore_failures=())


async def run_row(row: BatteryRow, channel: str, *, team_id: str = "",
                  claude: Optional[PartyIdentity] = None,
                  window: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """One row, with its restore in a TRUE `finally` — one that survives `BaseException` too.

    An interrupt during row 8 is exactly when the restore matters most: that is the moment the
    window anchor is sitting on a floor the battery chose. So it is started inside `finally` and
    SHIELDED, which means a cancellation delivered to this task stops us WAITING for the restore
    without stopping the restore itself. The cancellation is then re-raised, because a cancelled
    row has no result to report; the abandoned task is drained by the runner before the loop
    closes.

    A raise from the poller taxonomy or a correlation failure is an `error` — the harness broke,
    or the row's premise did not hold — and is NEVER downgraded to a failed assertion, which
    would report an operator's misconfiguration as the bot misbehaving.
    """
    ctx = RowContext(row=row.name, nonce=mint_nonce(row.name), channel=channel,
                     team_id=team_id, claude=claude)
    # THE WINDOW THIS RUN COMPUTED AGAINST, on every row's evidence. The battery is run in
    # small-window mode (`CHANNEL_WINDOW_TARGET=8`, `CHANNEL_WINDOW_CEILING=12` in the bot's and
    # the harness's launch environment), and a pass at 8/12 must never be read later as a pass at
    # the shipped 50/100 — so the number rides with the result rather than in someone's memory.
    ctx.evidence["window"] = dict(window or resolved_window())
    started_at = time.time()
    status: Optional[str] = None
    try:
        try:
            await row.run(ctx)
        except HarnessError as e:
            status = "error"
            ctx.notes = f"{type(e).__name__}: {e}"
        except Exception as e:  # noqa: BLE001 — an unexpected raise is still the harness breaking
            status = "error"
            ctx.notes = f"unhandled {type(e).__name__}: {e}"
    finally:
        cleanup = await _cleanup_guarded(ctx)

    if status is None:
        # A restore step that BROKE cannot report a clean pass. `status_for` grades what it
        # measured, and one that never finished measured nothing — so the row's own assertions
        # holding proves only that the row worked, not that the channel is back as it was.
        status = status_for(ctx, cleanup) if cleanup is not None else "error"
    return build_report(ctx, status=status, started_at=started_at, finished_at=time.time(),
                        cleanup=cleanup)


# Cleanup tasks whose row was cancelled before it could await them. THE RUNNER OWNS THIS, and
# drains it while the loop is still alive — see `run_with_drain`.
_PENDING_CLEANUPS: List["asyncio.Task[Any]"] = []


async def _cleanup_guarded(ctx: RowContext) -> Optional[CleanupResult]:
    """Run `cleanup_row` so that neither an ordinary failure nor a cancellation can skip it.

    Returns None when cleanup itself broke, which the caller turns into `error` rather than
    letting the row's passing assertions carry it to `pass`.

    On cancellation the task is left in `_PENDING_CLEANUPS` rather than abandoned: the shield
    stops the cancellation propagating from THIS await, but nothing here can stop the loop from
    cancelling a stray task on the way out. The drain is what finishes the job.
    """
    task = asyncio.ensure_future(cleanup_row(ctx))
    _PENDING_CLEANUPS.append(task)
    try:
        result = await asyncio.shield(task)
    except asyncio.CancelledError:
        raise                      # the drain owns it now
    except BaseException as e:  # noqa: BLE001 — cleanup_row is contracted not to raise at all
        _discard(task)
        ctx.notes = f"{ctx.notes} | cleanup raised: {type(e).__name__}: {e}".strip(" |")
        return None
    _discard(task)
    return result


def _discard(task: "asyncio.Task[Any]") -> None:
    if task in _PENDING_CLEANUPS:
        _PENDING_CLEANUPS.remove(task)


# How long the drain will wait for the cleanups a cancelled row left behind. BOUNDED, because a
# cleanup that can never finish must not hang the runner at exit — the battery would sit there
# forever with the operator watching a blank terminal, which is worse than an honest report of
# what could not be drained.
CLEANUP_DRAIN_SECONDS = 60.0


async def drain_cleanups(timeout: float = CLEANUP_DRAIN_SECONDS) -> List[str]:
    """Finish every cleanup a cancelled row left behind. Returns what it could NOT finish.

    **THE INVENTORY IS THE RETRY LIST, so an entry leaves it only once its task is DONE.** An
    earlier version cleared the list and then awaited the copy, which meant an interrupt landing
    during the gather erased the only record of what was still outstanding — the retry had
    nothing to retry.

    **AND THE WAIT IS BOUNDED.** A wedged cleanup task would otherwise block the drain forever,
    and with it the runner's exit. Anything still unfinished at the bound is reported rather than
    waited on; it stays in the inventory, so a later drain can still pick it up.
    """
    unfinished: List[str] = []
    bound = time.monotonic() + max(0.0, timeout)
    while True:
        for task in list(_PENDING_CLEANUPS):
            if task.done():
                _PENDING_CLEANUPS.remove(task)
        pending = list(_PENDING_CLEANUPS)
        if not pending:
            return unfinished
        remaining = bound - time.monotonic()
        if remaining <= 0:
            return [repr(t) for t in pending]
        try:
            # Shielded so a cancellation delivered to the DRAIN does not propagate into the
            # restores themselves; whatever is unfinished stays in the inventory for the retry.
            await asyncio.wait_for(
                asyncio.gather(*(asyncio.shield(task) for task in pending),
                               return_exceptions=True),
                timeout=remaining)
        except asyncio.TimeoutError:
            return [repr(t) for t in _PENDING_CLEANUPS]


def run_with_drain(coro: "Coroutine[Any, Any, Any]") -> Any:
    """Own the loop through `asyncio.Runner`, so a cancelled run can still finish its restores.

    **`asyncio.Runner`, NOT a bare `loop.run_until_complete`.** The Runner reproduces
    `asyncio.run`'s first-SIGINT behaviour: the interrupt CANCELS THE MAIN TASK rather than
    unwinding the loop out from under it. That difference is the whole guarantee — a raw
    `run_until_complete` can unwind before `run_row` ever receives its cancellation, so the
    `finally` that registers the cleanup never runs at all and there is nothing to drain.

    The Runner also cancels every remaining task when it closes, which is exactly what would kill
    a restore its row abandoned — at the precise moment the anchor is on the battery's own floor.
    The drain runs inside the Runner, before that close.

    A SECOND interrupt during the drain is retried ONCE, because the inventory survived it. If
    that attempt is interrupted too the run gives up and says so; pretending otherwise would be
    the harness claiming a restore it did not perform.
    """
    with asyncio.Runner() as runner:
        try:
            return runner.run(coro)
        finally:
            for _ in range(2):
                try:
                    left = runner.run(drain_cleanups())
                    if left:
                        print(f"cleanup did not finish for: {left}", file=sys.stderr)
                    break
                except BaseException:  # noqa: BLE001 — an interrupt mid-drain earns one retry
                    if not _PENDING_CLEANUPS:
                        break


def skipped_report(row: BatteryRow) -> Dict[str, Any]:
    """A row the operator's `--rows` selection left out.

    `skipped` is LEGAL ONLY HERE. A row never skips itself: a row that cannot run is an `error`,
    because "could not run" and "was not asked to run" are different facts and only the second is
    the operator's choice.
    """
    now = time.time()
    ctx = RowContext(row=row.name, nonce="", channel="")
    ctx.notes = "not selected by --rows"
    return build_report(ctx, status="skipped", started_at=now, finished_at=now)


def select(names: Optional[str]) -> List[str]:
    if not names:
        return list(ROW_NAMES)
    chosen = [part.strip() for part in names.split(",") if part.strip()]
    unknown = [name for name in chosen if name not in ROW_NAMES]
    if unknown:
        raise SystemExit(f"unknown row(s): {unknown}. Known rows: {list(ROW_NAMES)}")
    return chosen


async def run_battery(channel: str, chosen: Sequence[str]) -> List[Dict[str, Any]]:
    """Preflight, then every row in registry order. The preflight ABORTS the run, never a row.

    An allowlist that leaves the wrong number of entries, a second party that will not resolve, a
    user token in another workspace, or an epoch fence holding the channel — each means the
    battery would be measuring something other than what it claims, and finding that out after
    101 seeded messages means a hundred meaningless messages in the owner's channel.
    """
    await bot_identity()
    await harness_user_identity()
    team_id = await bot_team_id()
    window = resolved_window()
    print(f"window: target={window['channel_window_target']} "
          f"ceiling={window['channel_window_ceiling']} "
          f"(the BOT must have been launched with the same two variables)")
    await assert_channel_unfenced(team_id, channel)
    claude = await assert_claude_tag_allowlisted()

    reports: List[Dict[str, Any]] = []
    for row in REGISTRY:
        if row.name not in chosen:
            reports.append(skipped_report(row))
            continue
        # The verified pair is handed DOWN. Row 9a resolving Claude Tag again would spend a second
        # `bots.info` to re-derive an identity this preflight already proved.
        reports.append(await run_row(row, channel, team_id=team_id, claude=claude,
                                     window=window))
    return reports


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--channel", default=DEFAULT_CHANNEL,
                        help=f"must be {DEFAULT_CHANNEL}; no other channel is authorized")
    parser.add_argument("--rows", default=None,
                        help="comma-separated row names; everything else reports `skipped`")
    parser.add_argument("--out", default="battery_report.json", help="report path")
    args = parser.parse_args(argv)

    if args.channel != DEFAULT_CHANNEL:
        # THE AUTHORIZED-SCOPE RULE, ENFORCED RATHER THAN DOCUMENTED. The bulk rows seed over a
        # hundred messages; one typo would put them into a real conversation, and no flag exists
        # to override this because no override was ever authorized.
        raise SystemExit(
            f"refusing to run in {args.channel!r}: live testing is authorized only in "
            f"{DEFAULT_CHANNEL} (#chatgpt-bot-test). Prod is hands-off.")
    chosen = select(args.rows)
    reports = run_with_drain(run_battery(args.channel, chosen))
    write_report(reports, Path(args.out))

    executed = [r for r in reports if r["status"] != "skipped"]
    for report in reports:
        print(f"{report['status']:>8}  {report['row']}")
    failed = [r for r in executed if r["status"] != "pass"]
    print(f"\n{len(executed) - len(failed)}/{len(executed)} clean; report at {args.out}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
