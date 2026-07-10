from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from config import config
from message_markers import CHECKLIST_STATUS_MARKER

logger = logging.getLogger(__name__)

_DONE_MARK = "✓"
_FAIL_MARK = "✗"


def _strip_ellipsis(text: str) -> str:
    """Derive a done-label from an active label by dropping a trailing ellipsis."""
    stripped = text.rstrip()
    for suffix in ("…", "..."):
        if stripped.endswith(suffix):
            return stripped[: -len(suffix)].rstrip()
    return stripped


class ProgressChecklist:
    """Accumulating checklist rendered into a single edited-in-place Slack message.

    Completed steps render with a check; the active step renders with the loader
    emoji. One message (or, where Slack's assistant status is the only surface, the
    composer status line) is edited in place as steps advance.

    All methods serialize on an internal lock. Non-terminal edits within
    ``min_edit_interval`` coalesce into a single scheduled flush so the latest state
    always lands while intermediate states may skip. Terminal methods
    (``complete``/``fail``) are sticky and idempotent.
    """

    def __init__(self, client, channel_id: str, thread_id: Optional[str],
                 message_id: Optional[str] = None,
                 min_edit_interval: float = 0.8,
                 prefer_message: bool = False):
        self._client = client
        self._channel_id = channel_id
        self._thread_id = thread_id
        self._message_id = message_id
        self._min_interval = min_edit_interval
        # When True, a status-only surface still gets a real visible checklist message
        # (posted lazily via send_message_get_ts) instead of degrading to the composer
        # status, and the active step is mirrored into that status too (dual display).
        self._prefer_message = prefer_message
        # Set only when the force-message path fires: the message is the primary surface
        # AND the composer status is mirrored, so terminal/abort must clear the status.
        self._mirror_status = False

        # A caller-supplied message id fixes the surface up front; otherwise it is
        # resolved on the first flush via send_thinking_indicator.
        self._surface: Optional[str] = "message" if message_id is not None else None

        self._done: List[str] = []
        self._active: Optional[str] = None
        self._active_done: Optional[str] = None
        self._failed_note: Optional[str] = None

        self._terminal = False
        self._lock = asyncio.Lock()
        self._last_edit_time = float("-inf")
        self._pending_flush: Optional[asyncio.Task] = None
        self._delete_task: Optional[asyncio.Task] = None

    @property
    def message_id(self) -> Optional[str]:
        return self._message_id

    @property
    def surface(self) -> str:
        return self._surface or "none"

    @property
    def mirrors_status(self) -> bool:
        """True when the force-message surface is also mirroring the composer status."""
        return self._mirror_status

    async def step(self, active_text: str, done_text: Optional[str] = None) -> None:
        """Complete the current active step and start a new one."""
        async with self._lock:
            if self._terminal:
                logger.debug("checklist terminal; step(%r) ignored", active_text)
                return
            await self._ensure_surface()
            if self._active is not None:
                self._done.append(self._active_done or _strip_ellipsis(self._active))
            self._active = active_text
            self._active_done = done_text
            await self._flush_or_schedule()

    async def complete(self, final_text: Optional[str] = None,
                       delete_after: Optional[float] = None) -> None:
        """Mark every step done (sticky). Optionally delete the message after a delay."""
        async with self._lock:
            if self._terminal:
                logger.debug("checklist already terminal; complete() ignored")
                return
            self._terminal = True
            self._cancel_pending_flush()
            await self._ensure_surface()
            if self._active is not None:
                self._done.append(self._active_done or _strip_ellipsis(self._active))
                self._active = None
            if final_text:
                self._done.append(final_text)
            await self._terminal_flush()
            if delete_after is not None and self._surface == "message" and self._message_id:
                self._delete_task = asyncio.create_task(self._delete_after(delete_after))

    async def fail(self, note: str) -> None:
        """Mark the active step failed (sticky); completed steps stay visible."""
        async with self._lock:
            if self._terminal:
                logger.debug("checklist already terminal; fail() ignored")
                return
            self._terminal = True
            self._cancel_pending_flush()
            await self._ensure_surface()
            self._active = None
            self._failed_note = note
            await self._terminal_flush()

    # --- internal ---

    def _now(self) -> float:
        return asyncio.get_running_loop().time()

    async def _ensure_surface(self) -> None:
        if self._surface is not None:
            return
        msg_id = None
        if hasattr(self._client, "send_thinking_indicator"):
            msg_id = await self._client.send_thinking_indicator(self._channel_id, self._thread_id)
        if msg_id:
            self._message_id = msg_id
            self._surface = "message"
        elif (self._prefer_message and self._channel_id and self._thread_id
              and hasattr(self._client, "send_message_get_ts")):
            # Force a visible checklist message even though the assistant-status surface
            # is available (send_thinking_indicator returned None because setStatus
            # succeeded). The message is posted lazily on the first flush so its first
            # write already carries the rendered steps; the active step is ALSO mirrored
            # into the composer status (dual display).
            self._surface = "message"
            self._mirror_status = hasattr(self._client, "set_assistant_status")
        elif (hasattr(self._client, "set_assistant_status")
              and self._channel_id and self._thread_id):
            self._surface = "assistant_status"
        else:
            self._surface = "none"

    def _render(self) -> str:
        lines = [f"{_DONE_MARK} {d}" for d in self._done]
        if self._failed_note is not None:
            lines.append(f"{_FAIL_MARK} {self._failed_note}")
        elif self._active is not None:
            lines.append(f"{config.circle_loader_emoji} {self._active}")
        return "\n".join(lines)

    def _message_body(self) -> str:
        """Rendered checklist plus the invisible marker that keeps it out of history."""
        return self._render() + CHECKLIST_STATUS_MARKER

    async def _flush_or_schedule(self) -> None:
        elapsed = self._now() - self._last_edit_time
        if elapsed >= self._min_interval and self._pending_flush is None:
            await self._edit()
        elif self._pending_flush is None:
            self._pending_flush = asyncio.create_task(
                self._deferred_flush(self._min_interval - elapsed)
            )
        # A pending flush already covers the latest state — nothing else to do.

    async def _deferred_flush(self, delay: float) -> None:
        try:
            if delay > 0:
                await asyncio.sleep(delay)
            async with self._lock:
                self._pending_flush = None
                if self._terminal:
                    return
                await self._edit()
        except asyncio.CancelledError:
            return

    async def _edit(self) -> None:
        """Render the current (non-terminal) state to the active surface."""
        if self._surface == "message":
            just_created = await self._ensure_message_posted()
            if not just_created and self._message_id and hasattr(self._client, "update_message"):
                ok = await self._client.update_message(
                    self._channel_id, self._message_id, self._message_body())
                if ok is False:
                    logger.debug("checklist edit failed; keeping state for next flush")
            await self._mirror_active_status()
        elif self._surface == "assistant_status":
            if self._active and hasattr(self._client, "set_assistant_status"):
                await self._client.set_assistant_status(
                    self._channel_id, self._thread_id, status=self._active)
        self._last_edit_time = self._now()

    async def _ensure_message_posted(self) -> bool:
        """Lazily post the force-message checklist message with the current rendering.

        Returns True only when it creates the message on this call (so the caller can
        skip a redundant immediate update — the create already carried the body). A
        no-op returning False once a message id exists or off the force-message path.
        A create failure is swallowed and retried on the next flush.
        """
        if self._message_id is not None or not self._prefer_message:
            return False
        if not hasattr(self._client, "send_message_get_ts"):
            return False
        try:
            res = await self._client.send_message_get_ts(
                self._channel_id, self._thread_id, self._message_body())
            if res and res.get("success") and res.get("ts"):
                self._message_id = res["ts"]
                return True
        except Exception:  # noqa: BLE001 — a UI failure must never break the pipeline
            logger.debug("checklist message create failed", exc_info=True)
        return False

    async def _mirror_active_status(self) -> None:
        """Best-effort dual display: reflect the active step in the composer status too.
        Only fires on the force-message surface; failures never raise."""
        if not self._mirror_status or not self._active:
            return
        if hasattr(self._client, "set_assistant_status"):
            try:
                await self._client.set_assistant_status(
                    self._channel_id, self._thread_id, status=self._active)
            except Exception:  # noqa: BLE001 — mirror is best-effort
                logger.debug("checklist status mirror failed", exc_info=True)

    async def _clear_mirror_status(self) -> None:
        """Clear the mirrored composer status (checklist owns this clear)."""
        if hasattr(self._client, "clear_assistant_status"):
            try:
                await self._client.clear_assistant_status(self._channel_id, self._thread_id)
            except Exception:  # noqa: BLE001 — best-effort
                logger.debug("checklist status mirror clear failed", exc_info=True)

    async def _terminal_flush(self) -> None:
        elapsed = self._now() - self._last_edit_time
        if elapsed < self._min_interval:
            await asyncio.sleep(self._min_interval - elapsed)
        if self._surface == "message":
            just_created = await self._ensure_message_posted()
            if not just_created and self._message_id and hasattr(self._client, "update_message"):
                await self._client.update_message(
                    self._channel_id, self._message_id, self._message_body())
            if self._mirror_status:
                # Dual-display: the checklist owns clearing the mirrored composer status.
                await self._clear_mirror_status()
        elif self._surface == "assistant_status":
            # The checklist owns the composer status clear on this surface.
            if hasattr(self._client, "clear_assistant_status"):
                await self._client.clear_assistant_status(self._channel_id, self._thread_id)
        self._last_edit_time = self._now()

    def _cancel_pending_flush(self) -> None:
        if self._pending_flush is not None:
            self._pending_flush.cancel()
            self._pending_flush = None

    async def _delete_after(self, delay: float) -> None:
        try:
            await asyncio.sleep(delay)
            if self._message_id and hasattr(self._client, "delete_message"):
                await self._client.delete_message(self._channel_id, self._message_id)
        except asyncio.CancelledError:
            raise
