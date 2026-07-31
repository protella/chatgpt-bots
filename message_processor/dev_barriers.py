"""Dev-only turn barriers (spec §13 live checks, plan §4a).

Named seams where a live battery can freeze a turn — or a background compaction — and look at the
world before it moves on. Some claims are only checkable mid-flight: "the stream a turn renders is
current as of admission" cannot be observed from the outside once the reply has landed, and neither
can "an in-flight receipt is excluded from the very stream being built".

HARD no-op unless DEV_TURN_BARRIERS names the seam. Not a flag read behind a filesystem probe —
nothing is touched at all, because this code sits on the production turn path.

BARRIERS ARE KEYED `(seam, operation_id, test_epoch_id)` (§4a). Seam-only keying makes two
concurrent turns collide on one pair of files: the first to arrive owns the `.waiting` announcement
and the release frees BOTH, which is how P2's live battery lost a case and recovered only on
timeout. The operation id is the `turn_id` at a turn seam and the `compaction_id` at a compaction
seam, so two operations at one seam are two independent barriers.

Protocol: the barrier writes `<dir>/<seam>.<key>.waiting` (its context, one JSON object, key
included) and waits for `<dir>/<seam>.<key>.release` — or for the unkeyed `<dir>/<seam>.release`,
which releases every operation at that seam and is what a harness that does not care about keys
touches. A bounded wait means a forgotten barrier costs one slow turn, not a wedged bot.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any, Dict, Optional

from logger import setup_logger

logger = setup_logger(name="slack_bot.DevBarriers")

POST_ADMISSION = "post_admission"
POST_PARTIAL_POST = "post_partial_post"
PRE_RESUME_AFTER_COMPACTION = "pre_resume_after_compaction"

SEAMS = (POST_ADMISSION, POST_PARTIAL_POST, PRE_RESUME_AFTER_COMPACTION)
# Which id names the operation at each seam. A compaction has no turn.
COMPACTION_SEAMS = (PRE_RESUME_AFTER_COMPACTION,)

_ENV_SEAMS = "DEV_TURN_BARRIERS"
_ENV_DIR = "DEV_TURN_BARRIERS_DIR"
_ENV_TIMEOUT = "DEV_TURN_BARRIERS_TIMEOUT"
_ENV_EPOCH = "DEV_TEST_EPOCH_ID"
_DEFAULT_TIMEOUT = 120.0
_POLL_SECONDS = 0.1
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _enabled(seam: str) -> bool:
    raw = (os.environ.get(_ENV_SEAMS) or "").strip()
    if not raw:
        return False
    names = {part.strip().lower() for part in raw.split(",") if part.strip()}
    return bool(names & {"1", "all", "true", seam})


def _barrier_dir() -> Optional[str]:
    path = (os.environ.get(_ENV_DIR) or "").strip()
    return path or None


def _timeout() -> float:
    try:
        return max(0.0, float(os.environ.get(_ENV_TIMEOUT) or _DEFAULT_TIMEOUT))
    except ValueError:
        return _DEFAULT_TIMEOUT


def _slug(value: Any) -> str:
    text = str(value or "").strip()
    return _UNSAFE.sub("_", text)[:64]


def operation_id(seam: str, context: Optional[Dict[str, Any]] = None) -> str:
    """WHICH operation this barrier belongs to (§4a).

    `compaction_id` at a compaction seam, `turn_id` at a turn seam. The remaining fallbacks are
    for seams whose caller carries neither: a per-message ts still identifies ONE operation, and
    only a caller with no identifying field at all shares the unkeyed barrier.
    """
    ctx = context or {}
    names = (("compaction_id", "crawl_id", "turn_id") if seam in COMPACTION_SEAMS
             else ("turn_id", "compaction_id", "crawl_id"))
    for name in (*names, "message_ts", "attempt_id"):
        value = ctx.get(name)
        if value:
            return _slug(value)
    return "shared"


def barrier_key(seam: str, context: Optional[Dict[str, Any]] = None) -> str:
    """The `(operation_id, test_epoch_id)` half of the §4a key, as one filename fragment."""
    epoch = (context or {}).get("test_epoch_id") or os.environ.get(_ENV_EPOCH) or "0"
    return f"{operation_id(seam, context)}.{_slug(epoch)}"


async def barrier(seam: str, context: Optional[Dict[str, Any]] = None) -> bool:
    """Pause at `seam` until released. Returns True only when it actually waited."""
    if seam not in SEAMS or not _enabled(seam):
        return False
    directory = _barrier_dir()
    if not directory:
        logger.warning(f"dev barrier {seam} is enabled but {_ENV_DIR} is unset; not pausing")
        return False
    key = barrier_key(seam, context)
    waiting = os.path.join(directory, f"{seam}.{key}.waiting")
    release = os.path.join(directory, f"{seam}.{key}.release")
    # The unkeyed release frees EVERY operation at this seam — what a harness touches when it
    # does not care which turn it caught. Never removed here: one operation's release must not
    # consume the signal another is still waiting on.
    release_all = os.path.join(directory, f"{seam}.release")
    try:
        os.makedirs(directory, exist_ok=True)
        with open(waiting, "w", encoding="utf-8") as handle:
            json.dump({"seam": seam, "key": key, "at": time.time(), **(context or {})}, handle,
                      default=str)
    except OSError as e:
        logger.warning(f"dev barrier {seam} could not announce itself: {e}")
        return False
    logger.warning(f"dev barrier {seam} [{key}] waiting for {release}")
    deadline = time.monotonic() + _timeout()
    released = False
    while time.monotonic() < deadline:
        if os.path.exists(release) or os.path.exists(release_all):
            released = True
            break
        await asyncio.sleep(_POLL_SECONDS)
    for path in (waiting, release):
        try:
            os.unlink(path)
        except OSError:
            pass
    if not released:
        logger.warning(f"dev barrier {seam} [{key}] timed out; continuing")
    return True


async def post_admission(**context: Any) -> bool:
    """After H is pinned, the snapshot pointer is checked and the sidecars are read — and
    BEFORE any Slack fetch. Freezing here lets a battery post a message and prove the stream
    that follows does NOT contain it."""
    return await barrier(POST_ADMISSION, context)


async def post_partial_post(**context: Any) -> bool:
    """At the FIRST conversational in_flight transition on a barrier-eligible turn ledger,
    before any finalize. Freezing here lets a battery prove an in-flight surface is excluded
    from the stream a concurrent turn builds."""
    return await barrier(POST_PARTIAL_POST, context)


async def pre_resume_after_compaction(**context: Any) -> bool:
    """After a background compaction has published and BEFORE its task vacates the channel's
    single-flight slot — the only window in which "the next turn selects the new generation" is
    observable as a before/after rather than as a race. Keyed by `compaction_id`."""
    return await barrier(PRE_RESUME_AFTER_COMPACTION, context)
