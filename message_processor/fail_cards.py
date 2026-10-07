"""When Slack throttles us, we owe the answer anyway — the bot owns the recovery.

A channel turn whose history fetch Slack throttled (429, or a retry budget spent waiting out
Retry-After) cannot see its room and fails closed. That is transient: the same request works once
Slack lets up. So when somebody asked — an addressed wake, or a message the participation gate
woke — the turn posts one short NOTE in the conversation's voice, and the bot re-runs the failed
trigger itself:

  * after the Retry-After Slack gave, or, failing that, as soon as the next history fetch for
    that channel succeeds — whichever comes first;
  * as the NORMAL turn for that trigger, through the normal dispatch (`client.message_handler`)
    and the normal lease/ownership — the gate is not re-asked, it already woke;
  * failing transiently again just re-registers it for the next signal. No attempt counter: each
    attempt is bounded by the existing fetch budget.

A newer turn from the SAME sender, in a conversation scope that overlaps the failed trigger's (the
stale guard's own scopes — a thread reply under the failed root covers it, and so does the same
person's next top-level message), that delivers a reply first COVERS it, exactly as an early
stand-down would have: a re-run not yet admitted is rejected before admission, an admitted one is
asked to stand down through the normal path (same eligibility, same cleanup, deferred if it is
mid-mutation — or left to the stale guard if it is ineligible), and the note comes down. When the
re-run (or the covering reply) delivers, the note is deleted; a deletion that fails stays
registered for the next success.

IN MEMORY, DELIBERATELY. A restart forgets the registry; a note still up then stays up.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from logger import setup_logger
from message_processor import participation_telemetry
from message_processor.stale_send_guard import Scope, is_newer, request_stand_down

logger = setup_logger(name="slack_bot.FailCards")

# The note, verbatim (owner wording). No "try again": the bot owns the retry.
TRANSIENT_NOTE = "Waiting on Slack… I'll answer as soon as it catches up."

# Stamped on the re-run's message: its own transient failure is still owed a retry, even though
# the re-run itself runs ungated.
FAIL_RETRY_RERUN = "fail_retry_rerun"
# Stamped on a dispatched re-run that a newer reply covered before it was admitted: main.py
# rejects it before admission. The value is the covering trigger's ts.
FAIL_RETRY_COVERED = "fail_retry_covered"

FAIL_CARD_POSTED = "posted"
FAIL_CARD_SILENT = "silent"

RETRY_SCHEDULED = "scheduled"
RETRY_STARTED = "started"
RETRY_COVERED = "covered"


def scope_key(channel_id: Optional[str], thread_root: Optional[str],
              trigger_ts: Optional[str]) -> str:
    """One conversation scope: a thread (`channel:root`) or the channel's top level."""
    if thread_root and trigger_ts and str(thread_root) != str(trigger_ts):
        return f"{channel_id}:{thread_root}"
    return f"{channel_id}:top"


@dataclass
class PendingFailure:
    scope: str
    channel_id: str
    trigger_ts: str
    sender_id: Optional[str]
    cause: Optional[str]
    card_ts: Optional[str]
    # Dispatches the re-run; returns (re-run message, its task) when it could.
    start: Callable[[], Optional[Tuple[Any, Any]]] = field(repr=False)
    # The failed trigger's stale-guard scopes — what a covering reply must overlap.
    scopes: Tuple[Scope, ...] = ()
    at: float = field(default_factory=time.time)
    timer: Optional["asyncio.Task[Any]"] = field(default=None, repr=False)
    # True from the moment the re-run is dispatched until it fails transiently again.
    started: bool = False
    # The retry is over (delivered or covered); only the note's deletion may still be owed.
    resolved: bool = False
    # The note is being posted: its owner finishes the bookkeeping once it knows the ts.
    posting: bool = False
    rerun_message: Any = field(default=None, repr=False)
    rerun_task: Any = field(default=None, repr=False)


class FailureRegistry:
    def __init__(self) -> None:
        self._items: Dict[Tuple[str, str], PendingFailure] = {}

    def find(self, channel_id: str, trigger_ts: Optional[str]) -> Optional[PendingFailure]:
        return self._items.get((str(channel_id), str(trigger_ts)))

    def items(self) -> List[PendingFailure]:
        return list(self._items.values())

    def register(self, failure: PendingFailure, *, retry_after: Optional[float]) -> None:
        """Wait for the next signal to re-run this failure: the Retry-After timer, when Slack gave
        one, and the channel's next successful fetch (`fetch_succeeded`) either way."""
        key = (failure.channel_id, failure.trigger_ts)
        previous = self._items.get(key)
        if previous is not None and previous is not failure:
            _cancel(previous)
        self._items[key] = failure
        failure.started = False
        _cancel(failure)
        if retry_after is not None:
            failure.timer = asyncio.ensure_future(self._after(failure, retry_after))
        participation_telemetry.fail_retry(failure.channel_id, failure.trigger_ts,
                                           state=RETRY_SCHEDULED, card_ts=failure.card_ts,
                                           cause=failure.cause, retry_after=retry_after)

    async def _after(self, failure: PendingFailure, delay: float) -> None:
        await asyncio.sleep(max(0.0, float(delay)))
        failure.timer = None
        self.fire(failure)

    def fire(self, failure: PendingFailure) -> None:
        """Dispatch the re-run, once per signal. Never raises."""
        if failure.started or failure.resolved:
            return
        if self._items.get((failure.channel_id, failure.trigger_ts)) is not failure:
            return
        failure.started = True
        _cancel(failure)
        participation_telemetry.fail_retry(failure.channel_id, failure.trigger_ts,
                                           state=RETRY_STARTED, card_ts=failure.card_ts,
                                           cause=failure.cause)
        try:
            dispatched = failure.start()
        except Exception as e:  # noqa: BLE001 — a re-run that cannot start waits for the next
            failure.started = False
            logger.warning(f"re-run of {failure.trigger_ts} could not start: {e}")
            return
        if dispatched is not None:
            failure.rerun_message, failure.rerun_task = dispatched

    def fetch_succeeded(self, channel_id: Optional[str]) -> None:
        """A history fetch for this channel just worked: Slack has let up, so every failure here
        still waiting re-runs now."""
        if not channel_id:
            return
        for failure in self.items():
            if failure.channel_id == str(channel_id) and not failure.started:
                self.fire(failure)

    def drop(self, failure: PendingFailure) -> None:
        _cancel(failure)
        key = (failure.channel_id, failure.trigger_ts)
        if self._items.get(key) is failure:
            self._items.pop(key, None)

    def reset(self) -> None:
        for failure in self._items.values():
            _cancel(failure)
        self._items.clear()


def _cancel(failure: PendingFailure) -> None:
    if failure.timer is not None and not failure.timer.done():
        failure.timer.cancel()
    failure.timer = None


registry = FailureRegistry()


def fetch_succeeded(channel_id: Optional[str]) -> None:
    """Hook for the stream builders: a channel history fetch completed. Never raises."""
    try:
        registry.fetch_succeeded(channel_id)
    except Exception as e:  # noqa: BLE001
        logger.debug(f"fail retry signal skipped: {e}")


async def settle_delivery(client: Any, *, channel_id: str, scopes: Iterable[Scope],
                          trigger_ts: Optional[str], sender_id: Optional[str]) -> None:
    """A turn delivered a reply. The failure it re-ran is done; a same-sender failure it is newer
    than, in an OVERLAPPING stale-guard scope, is covered (its re-run rejected or stood down).
    Either way the note comes down — and any note still owed a deletion that this reply could
    settle gets another try. Never raises."""
    covering = set(scopes)
    for failure in registry.items():
        if failure.channel_id != str(channel_id):
            continue
        own = failure.trigger_ts == str(trigger_ts)
        overlaps = bool(covering & set(failure.scopes))
        if not failure.resolved:
            if own:
                failure.resolved = True
            elif (sender_id and failure.sender_id == sender_id and overlaps
                  and is_newer(trigger_ts, failure.trigger_ts)):
                failure.resolved = True
                _cancel(failure)
                participation_telemetry.fail_retry(failure.channel_id, failure.trigger_ts,
                                                   state=RETRY_COVERED, card_ts=failure.card_ts,
                                                   cause=failure.cause, covered_by=trigger_ts)
                _cover_rerun(failure, str(trigger_ts))
            else:
                continue
        elif not (own or overlaps):
            continue
        if failure.posting:
            continue        # its poster deletes the note the moment it knows the ts
        if failure.card_ts is None or await _delete_note(client, failure):
            if failure.card_ts is not None:
                participation_telemetry.fail_card_cleared(
                    failure.channel_id, trigger_ts, card_ts=failure.card_ts,
                    card_trigger_ts=failure.trigger_ts, cause=failure.cause)
            registry.drop(failure)


def _cover_rerun(failure: PendingFailure, by: str) -> None:
    """A newer reply covered a re-run already dispatched. Not yet admitted: stamped, and
    rejected before admission. Admitted: the normal stand-down is requested (an ineligible turn
    is left to the stale guard and reconsideration)."""
    if not failure.started or failure.rerun_message is None:
        return
    task = failure.rerun_task
    if task is not None and request_stand_down(task, by):
        return
    meta = getattr(failure.rerun_message, "metadata", None)
    if isinstance(meta, dict):
        meta[FAIL_RETRY_COVERED] = by


async def finish_post(client: Any, failure: PendingFailure, card_ts: Optional[str]) -> None:
    """The note's post returned. If delivery or a cover resolved the failure meanwhile (or it was
    dropped), the accepted note comes straight back down rather than standing unowned."""
    failure.posting = False
    if card_ts:
        failure.card_ts = str(card_ts)
    owned = registry.find(failure.channel_id, failure.trigger_ts) is failure
    if owned and not failure.resolved:
        return
    if failure.card_ts is None:
        registry.drop(failure)
        return
    if await _delete_note(client, failure):
        participation_telemetry.fail_card_cleared(
            failure.channel_id, failure.trigger_ts, card_ts=failure.card_ts,
            card_trigger_ts=failure.trigger_ts, cause=failure.cause, outcome="resolved_mid_post")
        registry.drop(failure)
    elif not owned:
        # Kept for the next success: re-own the record it was dropped from.
        registry._items[(failure.channel_id, failure.trigger_ts)] = failure


async def end_rerun(client: Any, *, channel_id: str, trigger_ts: Optional[str],
                    outcome: str) -> None:
    """The bot's re-run of this trigger ended WITHOUT a delivered reply and without being
    throttled again — an error, a silence, a stand-down nobody covered, a cancellation. The note
    promised an answer that is not coming from this re-run, so it never outlives it: taken down
    now (kept registered only if the deletion fails), logged with how the re-run ended. Never
    raises."""
    failure = registry.find(channel_id, trigger_ts)
    if failure is None or not failure.started or failure.resolved:
        return      # delivered and settled, re-registered for another signal, or covered
    failure.resolved = True
    _cancel(failure)
    if failure.posting:
        return      # its poster takes the note down the moment it knows the ts
    if failure.card_ts is None:
        registry.drop(failure)
        return
    if await _delete_note(client, failure):
        participation_telemetry.fail_card_cleared(
            failure.channel_id, trigger_ts, card_ts=failure.card_ts,
            card_trigger_ts=failure.trigger_ts, cause=failure.cause, outcome=outcome)
        registry.drop(failure)


async def _delete_note(client: Any, failure: PendingFailure) -> bool:
    try:
        gone = await client.delete_message(failure.channel_id, failure.card_ts)  # unleased-ok: teardown of our own stale note — removing a surface is never a stale answer
    except Exception as e:  # noqa: BLE001 — kept for the next success
        logger.debug(f"fail note {failure.card_ts} not deleted: {e}")
        return False
    if not gone:
        return False
    drop_receipt = getattr(client, "_drop_own_message_receipt", None)
    if callable(drop_receipt):
        try:
            await drop_receipt(failure.channel_id, failure.card_ts)
        except Exception as e:  # noqa: BLE001 — the message is gone; the row is best-effort
            logger.debug(f"fail note {failure.card_ts} receipt not dropped: {e}")
    return True


__all__ = ["FAIL_CARD_POSTED", "FAIL_CARD_SILENT", "FAIL_RETRY_COVERED", "FAIL_RETRY_RERUN",
           "FailureRegistry", "finish_post",
           "PendingFailure", "RETRY_COVERED", "RETRY_SCHEDULED", "RETRY_STARTED",
           "TRANSIENT_NOTE", "end_rerun", "fetch_succeeded", "registry", "scope_key",
           "settle_delivery"]
