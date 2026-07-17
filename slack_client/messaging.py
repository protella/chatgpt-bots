from __future__ import annotations

import asyncio
import random
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional
from uuid import uuid4

from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_sdk.errors import SlackApiError

from base_client import HistoryFetchError, Message, Response
from config import SUPPORTED_CHAT_MODELS, config, pipeline_status_markers, valid_emoji_name
from message_markers import (
    CONTINUATION_HEAD,
    continuation_trailer,
    fence_safe_chunks,
    is_checklist_status_text,
)
from slack_client.event_handlers.feedback import (
    FEEDBACK_ACTION_ID,
    USER_SETTINGS_ACTION_ID,
    build_feedback_blocks,
    feedback_enabled,
    should_offer_feedback,
)
from slack_client.formatting.blocks import extract_supplementary_text
from slack_client.utilities import strip_citations

import re as _re

# Block action_ids that mark a message as one of our UI helpers (channel footer's
# Configure button, Phase H feedback strip + its user-settings button). PURE-chrome
# messages are not conversation — the history rebuild must never feed them back into
# context. But a real response can now CARRY the Configure chrome (native streaming
# attaches it on stopStream), so helper action_ids alone no longer condemn a message:
# only the known chrome fallback texts do. Those fallbacks are exactly what the
# separate helper posts set as `text`: the model label (footer), "Rate this response"
# (feedback strip), and the legacy "Settings available".
_UI_HELPER_ACTION_IDS = frozenset({
    "open_channel_settings", FEEDBACK_ACTION_ID, USER_SETTINGS_ACTION_ID,
})
# Notification/accessibility fallback text for a standalone Configure-footer post. NOT a bare
# model name (that reads as a spurious message when Slack surfaces the fallback — the live
# "gpt-5.6-sol" post 2026-07-16); still recognized here so the history rebuild skips it as chrome.
RESPONSE_FOOTER_FALLBACK_TEXT = "Channel settings"
_UI_HELPER_FALLBACK_TEXTS = frozenset(
    {"Rate this response", "Settings available", RESPONSE_FOOTER_FALLBACK_TEXT}
    | set(SUPPORTED_CHAT_MODELS)
)


def _is_ui_helper_message(msg: dict) -> bool:
    """True when a message is PURE UI chrome: a helper action_id in its blocks AND no
    text beyond the helper's own notification fallback. A content-bearing response
    with the Configure chrome attached keeps its real text and is preserved."""
    has_helper = any(
        el.get("action_id") in _UI_HELPER_ACTION_IDS
        for b in (msg.get("blocks") or []) if b.get("type") in ("actions", "context_actions")
        for el in (b.get("elements") or [])
    )
    if not has_helper:
        return False
    text = (msg.get("text") or "").strip()
    return not text or text in _UI_HELPER_FALLBACK_TEXTS

# Own-message placeholder/status filters for history rebuild (R3): only messages
# shaped like ":emoji: Status text" AND carrying a known transient marker are
# skipped — a human message merely containing the word "Thinking" is kept.
_SELF_STATUS_RE = _re.compile(r"^:[a-z0-9_+\-]+:\s")

# assistant.threads.setStatus renders PLAIN TEXT — :shortcodes: appear literally
# (user screenshot 2026-07-09). Posted messages render them fine, so status strings
# are sanitized only at the setStatus boundary: known shortcodes become Unicode,
# unknown ones (incl. workspace custom emoji, which have no Unicode form) are
# stripped. One configured string thus renders correctly on both surfaces.
_SHORTCODE_TO_UNICODE = {
    "hourglass_flowing_sand": "⏳", "hourglass": "⌛", "mag": "🔍",
    "bar_chart": "📊", "brain": "🧠", "bulb": "💡", "gear": "⚙️",
    "robot_face": "🤖", "sparkles": "✨", "thinking_face": "🤔",
    "memo": "📝", "art": "🎨", "camera": "📷", "globe_with_meridians": "🌐",
}
_SHORTCODE_RE = _re.compile(r":([a-z0-9_+\-]+):")


def _status_plain_text(text: str) -> str:
    """Render a status string for the plain-text setStatus surface."""
    def sub(m):
        return _SHORTCODE_TO_UNICODE.get(m.group(1), "")
    return _SHORTCODE_RE.sub(sub, text or "").strip() or "working on it…"
_SELF_STATUS_MARKERS = (
    "Thinking...",
    "Rebuilding thread history",
    "Catching up on",
)


def is_self_chrome_message(text: str, msg: dict) -> bool:
    """True when a message is our OWN transient UI chrome — a status/placeholder line
    (":emoji: Thinking…"), a progress checklist ("✓ …"), the legacy processing notice, the
    "Settings available" button, or a pure UI-helper block (Configure button / feedback strip)
    with no real text. Such messages must never be replayed as an assistant turn (history
    rebuild) NOR recorded as authoritative `[self]` addressee evidence (F47 cold-start backfill).

    Content-bearing replies — even ones carrying the Configure chrome attached on stopStream —
    are NOT chrome and return False. The caller decides ownership (only pass our own messages for
    the self-status checks to be meaningful); this only classifies the shape. Fail-open: any
    error classifies as NOT chrome, so a real reply is never silently dropped."""
    try:
        text = text or ""
        # F1 progress-checklist ("✓ …") — carries an invisible marker, not the ":emoji:" shape.
        if is_checklist_status_text(text):
            return True
        # Transient placeholders/status lines: ":emoji: Thinking..." and same-shaped updates.
        if _SELF_STATUS_RE.match(text) and (
            any(marker in text for marker in _SELF_STATUS_MARKERS)
            or any(marker in text for marker in pipeline_status_markers())
        ):
            return True
        # Legacy busy/processing notice.
        if ":warning:" in text and "currently processing" in text:
            return True
        # Settings button message.
        if text == "Settings available":
            return True
        # Pure UI-helper block (Configure button / Phase H feedback strip) with no real text.
        if _is_ui_helper_message(msg):
            return True
    except Exception:
        return False
    return False


class NativeStreamSession:
    """Adapter over Slack's native streaming API (chat.startStream/appendStream/stopStream).

    The existing streaming handler thinks in CUMULATIVE text ("everything so far"), while the
    native API wants DELTAS. This session bridges the two: feed it the cumulative text each tick
    via ``update()`` and it appends only the new tail. ``finish()`` closes the stream (optionally
    with blocks, which Slack only allows on stop).

    Fully best-effort: if startStream fails the session is inert (``active`` False) and the caller
    falls back to the legacy ``update_message_streaming`` path. Any later failure also flips it
    inert so the caller can recover.
    """

    def __init__(self, client, channel_id: str, thread_id: str, logger=None,
                 team_id: Optional[str] = None, user_id: Optional[str] = None):
        self._client = client
        self._channel = channel_id
        self._thread = thread_id
        self._log = logger
        # chat.startStream requires BOTH recipient_team_id (workspace) and
        # recipient_user_id (the triggering user) for channel streaming — missing either
        # 400s (missing_recipient_team_id / missing_recipient_user_id). appendStream/
        # stopStream key off the returned ts and need neither.
        self._team_id = team_id
        self._user_id = user_id
        self.ts: Optional[str] = None
        self.active: bool = False
        self._sent: str = ""

    async def start(self, initial_text: str = "") -> bool:
        if not self._thread:
            # chat.startStream REQUIRES thread_ts (Slack: "missing required field"),
            # so top-level channel replies can never stream natively. Skip the
            # guaranteed-to-fail call; the caller falls back to legacy streaming.
            if self._log:
                self._log("native streaming requires a thread — top-level reply falls back to legacy")
            self.active = False
            return False
        if not self._team_id or not self._user_id:
            # chat.startStream now REQUIRES recipient_team_id AND recipient_user_id for
            # channel streaming (Slack: "missing_recipient_team_id" /
            # "missing_recipient_user_id"). Without both the call is guaranteed to fail —
            # skip it and let the caller fall back to legacy streaming (never crash).
            if self._log:
                missing = "team_id" if not self._team_id else "user_id"
                self._log(f"native streaming requires a {missing} — falling back to legacy")
            self.active = False
            return False
        try:
            resp = await self._client.chat_startStream(
                channel=self._channel,
                thread_ts=self._thread,
                markdown_text=initial_text or None,
                recipient_team_id=self._team_id,
                recipient_user_id=self._user_id,
            )
            self.ts = resp.get("ts")
            self._sent = initial_text or ""
            self.active = bool(self.ts)
            return self.active
        except Exception as e:  # noqa: BLE001 - best-effort, never fatal
            if self._log:
                self._log(f"native stream start failed, will fall back: {e}")
            self.active = False
            return False

    async def update(self, cumulative_text: str) -> bool:
        """Append the new tail of ``cumulative_text`` since the last update."""
        if not self.active or self.ts is None:
            return False
        delta = cumulative_text[len(self._sent):] if cumulative_text.startswith(self._sent) else cumulative_text
        if not delta:
            return True
        try:
            await self._client.chat_appendStream(channel=self._channel, ts=self.ts, markdown_text=delta)
            self._sent = cumulative_text
            return True
        except Exception as e:  # noqa: BLE001
            if self._log:
                self._log(f"native stream append failed: {e}")
            self.active = False
            return False

    async def finish(self, final_text: Optional[str] = None, blocks=None) -> bool:
        if self.ts is None:
            return False
        try:
            kwargs = {"channel": self._channel, "ts": self.ts}
            if final_text is not None and final_text.startswith(self._sent):
                tail = final_text[len(self._sent):]
                if tail:
                    kwargs["markdown_text"] = tail
            elif final_text is not None:
                kwargs["markdown_text"] = final_text
            if blocks is not None:
                kwargs["blocks"] = blocks
            await self._client.chat_stopStream(**kwargs)
            self.active = False
            return True
        except Exception as e:  # noqa: BLE001
            if self._log:
                self._log(f"native stream stop failed: {e}")
            self.active = False
            return False


class WorkspaceEmojiCache:
    """Process-lifetime cache of the workspace's CUSTOM emoji shorthand names (emoji.list).

    Reachable from BOTH the react_to_message tool-schema factory and the participation gate
    via ``client.workspace_emojis``. Holds a sorted/deduped tuple of names plus a monotonic
    expiry; ``refresh()`` is the only thing that hits Slack, and ``get_custom_emoji_names()``
    is a sync, stale-tolerant getter that schedules a background refresh when expired but
    never blocks a request.

    Fail-soft everywhere: any error (including the emoji:read scope being absent — which is
    also the de-facto off switch) RETAINS the last good tuple; an empty tuple only ever means
    we have never had a successful fetch. The model then simply sees no customs, and no turn
    fails on account of emoji.list.
    """

    def __init__(self, client):
        self._client = client
        self._names: tuple = ()
        self._expiry: float = 0.0        # monotonic deadline; 0 = never fetched
        self._lock = asyncio.Lock()
        self._refreshing: bool = False   # guards against scheduling overlapping refreshes
        self._refresh_task = None        # ref to the scheduled task (GC + lifecycle guard)

    def _log_debug(self, msg: str) -> None:
        log = getattr(self._client, "log_debug", None)
        if log:
            log(msg)

    async def refresh(self) -> tuple:
        """Fetch emoji.list and rebuild the name tuple.

        Parses resp["emoji"] KEYS (both real customs and ``alias:*`` entries — the KEY is the
        alias NAME reactions.add accepts, so aliases are kept), filters each through
        ``valid_emoji_name``, then sorts + dedupes. On ANY error the last good tuple is kept
        (empty only if never fetched). The TTL is reset either way, so a persistent failure
        (e.g. missing emoji:read) backs off instead of hammering the API on every getter call.
        """
        async with self._lock:
            try:
                resp = await self._client.app.client.emoji_list()
                emoji = (resp or {}).get("emoji") or {}
                names = {
                    name for name in ((k or "").strip().strip(":") for k in emoji.keys())
                    if valid_emoji_name(name)
                }
                self._names = tuple(sorted(names))
            except Exception as e:  # noqa: BLE001 — never fatal; keep the last good tuple
                self._log_debug(f"workspace emoji refresh failed, keeping last good: {e}")
            finally:
                self._expiry = time.monotonic() + max(
                    1.0, float(getattr(config, "workspace_emoji_ttl_seconds", 3600)))
            return self._names

    def get_custom_emoji_names(self) -> tuple:
        """Sync, stale-ok. Returns the current tuple immediately; if it has expired AND no
        refresh is already running, schedules a background refresh (fire-and-forget). Never
        awaits, never raises — a request path can call this freely."""
        if time.monotonic() >= self._expiry and not self._refreshing:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return self._names  # no running loop — just return what we have
            self._refreshing = True  # set synchronously so a burst schedules exactly one refresh
            try:
                task = loop.create_task(self.refresh())
                self._refresh_task = task  # keep a ref so the task isn't GC'd mid-flight
                # Own the guard's lifecycle HERE, not in refresh()'s finally: the callback fires on
                # completion, error, AND cancellation, so a task cancelled before refresh() runs can't
                # wedge _refreshing=True and block every future refresh. And since only this getter
                # sets the flag True (once, under the guard), there is no premature-clear race.
                task.add_done_callback(self._on_refresh_done)
            except Exception as e:  # noqa: BLE001
                self._refreshing = False
                self._log_debug(f"workspace emoji background refresh not scheduled: {e}")
        return self._names

    def _on_refresh_done(self, task) -> None:
        """Clear the refresh guard once the scheduled task settles (success, error, or cancel)."""
        self._refreshing = False
        self._refresh_task = None


class SlackMessagingMixin:
    async def start(self):
        """Start the Slack bot"""
        self.handler = AsyncSocketModeHandler(self.app, config.slack_app_token)
        self.log_info("Starting Slack bot in socket mode...")

        # F9: detection-only socket-liveness monitor. Hook every inbound envelope on the
        # async socket client and start the 60s watchdog (never touches the socket).
        try:
            from slack_client.socket_liveness import SocketLivenessMonitor
            self._socket_liveness = SocketLivenessMonitor(
                getattr(self.handler, "client", None),
                timeout=config.socket_liveness_timeout,
                log_info=self.log_info,
                log_warning=self.log_warning,
                log_error=self.log_error,
            )
            self._socket_liveness.attach()
            self._socket_liveness.start()
        except Exception as e:
            # Partial start (e.g. attach() installed the listener but start() failed): stop
            # the monitor so we never leave a dangling listener behind.
            self.log_warning(f"Could not start socket-liveness monitor: {e}")
            await self._stop_socket_liveness_quietly()

        # Resolve our own identity up front so we can tell our messages apart from other bots'
        await self._ensure_self_identity()

        # C1: warm the workspace custom-emoji cache once, now that identity is set. Fail-soft —
        # a missing emoji:read scope (or any API error) just leaves the cache empty and the
        # model sees no customs; the getter refreshes it lazily thereafter.
        cache = getattr(self, "workspace_emojis", None)
        if cache is not None:
            try:
                await cache.refresh()
            except Exception as e:  # noqa: BLE001 — startup must never break on emoji.list
                self.log_debug(f"initial workspace emoji refresh failed: {e}")

        # Create a task for start_async that can be cancelled
        self._start_task = asyncio.create_task(self.handler.start_async())

        try:
            await self._start_task
        except asyncio.CancelledError:
            self.log_info("Slack bot start task cancelled")
            await self._stop_socket_liveness_quietly()  # detach before propagating
            raise
        except Exception as e:
            self.log_error(f"Error in Slack bot start: {e}")
            await self._stop_socket_liveness_quietly()  # detach before propagating
            raise

    async def _stop_socket_liveness_quietly(self) -> None:
        """Stop + detach the socket-liveness monitor, swallowing errors and clearing the ref."""
        monitor = getattr(self, "_socket_liveness", None)
        self._socket_liveness = None
        if monitor is not None:
            try:
                await monitor.stop()
            except Exception as e:
                self.log_debug(f"Error stopping socket-liveness monitor: {e}")

    async def stop(self):
        """Stop the Slack bot"""
        # F9: stop the liveness monitor first (independent of the handler teardown below).
        monitor = getattr(self, "_socket_liveness", None)
        if monitor is not None:
            try:
                await monitor.stop()
            except Exception as e:
                self.log_debug(f"Error stopping socket-liveness monitor: {e}")
            self._socket_liveness = None

        if self.handler:
            self.log_info("Stopping Slack bot...")

            # Cancel the start task to break out of the blocking start_async call
            if hasattr(self, '_start_task') and not self._start_task.done():
                self.log_info("Cancelling start task...")
                self._start_task.cancel()
                try:
                    await self._start_task
                except asyncio.CancelledError:
                    self.log_info("Slack bot start task cancelled")

            # Try to close handler sessions first before calling handler.close_async()
            # Also try to close the socket client's session if it exists
            if hasattr(self.handler, 'client') and self.handler.client:
                if hasattr(self.handler.client, 'session') and self.handler.client.session:
                    if not self.handler.client.session.closed:
                        self.log_debug("Closing handler client session")
                        try:
                            await asyncio.wait_for(self.handler.client.session.close(), timeout=0.5)
                            self.log_debug("Handler client session closed")
                        except asyncio.TimeoutError:
                            self.log_warning("Timeout closing handler client session")
                        except Exception as e:
                            self.log_warning(f"Error closing handler client session: {e}")

                if hasattr(self.handler.client, 'aiohttp_client_session') and self.handler.client.aiohttp_client_session:
                    session = self.handler.client.aiohttp_client_session
                    if not session.closed:
                        # Don't call session.close() or connector.close() as they hang
                        # Just forcibly mark everything as closed
                        try:
                            # Mark the connector as closed without actually closing it
                            if hasattr(session, '_connector') and session._connector:
                                if hasattr(session._connector, '_closed'):
                                    session._connector._closed = True
                                # Clear any transports
                                if hasattr(session._connector, '_transports'):
                                    session._connector._transports = []
                                # Clear conns if it exists
                                if hasattr(session._connector, '_conns'):
                                    session._connector._conns = {}

                            # Also try the public connector attribute
                            if hasattr(session, 'connector') and session.connector:
                                if hasattr(session.connector, '_closed'):
                                    session.connector._closed = True

                            # Mark session as closed
                            if hasattr(session, '_closed'):
                                session._closed = True

                            # Try to detach from the event loop
                            if hasattr(session, '_loop'):
                                session._loop = None

                        except Exception as e:
                            self.log_warning(f"Error during force-close of aiohttp_client_session: {e}")

            # Now try to close the socket mode handler itself - but skip if it might hang
            # Check if we should even try - if we manually closed sessions, maybe skip handler close
            skip_handler_close = False
            if hasattr(self.handler, 'client') and self.handler.client:
                if hasattr(self.handler.client, 'aiohttp_client_session'):
                    # If we have the session and it's closed, we probably don't need handler.close_async
                    if self.handler.client.aiohttp_client_session.closed:
                        skip_handler_close = True

            if not skip_handler_close:
                try:
                    # Create a task for handler close so it doesn't block
                    close_task = asyncio.create_task(self.handler.close_async())

                    # Wait for it with a very short timeout since it tends to hang
                    try:
                        await asyncio.wait_for(asyncio.shield(close_task), timeout=0.1)
                        self.log_debug("Socket mode handler closed")
                    except asyncio.TimeoutError:
                        self.log_warning("Socket mode handler close timed out after 0.1 seconds, continuing...")
                        # Don't cancel the task, let it complete in background
                except Exception as e:
                    self.log_warning(f"Error closing socket mode handler: {e}")

        # Close the web client's aiohttp session if it exists
        if self.app:
            # Try the main client
            if self.app.client:
                try:
                    # The AsyncWebClient has a _session attribute that needs closing
                    if hasattr(self.app.client, '_session') and self.app.client._session:
                        if not self.app.client._session.closed:
                            await self.app.client._session.close()
                            self.log_info("Closed Slack web client session")
                except Exception as e:
                    self.log_warning(f"Error closing web client session: {e}")

            # Check for _async_client as well (some versions use this)
            if hasattr(self.app, '_async_client') and self.app._async_client:
                try:
                    if hasattr(self.app._async_client, '_session') and self.app._async_client._session:
                        if not self.app._async_client._session.closed:
                            await self.app._async_client._session.close()
                            self.log_info("Closed app._async_client session")
                except Exception as e:
                    self.log_warning(f"Error closing _async_client session: {e}")

        # Clean up utilities session if it exists
        if hasattr(self, '_cleanup_session'):
            try:
                await self._cleanup_session()
            except Exception as e:
                self.log_warning(f"Error cleaning up utilities session: {e}")

    def _record_own_reply_pulse(self, channel_id: str, thread_id: Optional[str],
                                ts: Optional[str], text: str) -> None:
        """F5 fix (a): record the bot's OWN final reply into the pulse at send time.

        The clean, complete reply text keyed on its real ts — the authoritative source for
        the assistant's own turns in the classifier's thread tail (echoed placeholders /
        footers / native-stream edits are unreliable). Chrome and empty text are skipped.
        Best-effort; never breaks sending."""
        pulse = getattr(self, "channel_pulse", None)
        if pulse is None or not ts:
            return
        if not (text or "").strip() or is_checklist_status_text(text):
            return
        try:
            pulse.record_own_reply(channel_id, thread_ts=thread_id, ts=ts, text=text)
        except Exception as e:
            self.log_debug(f"own-reply pulse record failed: {e}")

    def _record_own_reaction_pulse(self, channel_id: str, ts: Optional[str],
                                   emoji: str) -> Optional[dict]:
        """F31: record a reaction the bot itself just placed into the channel pulse, so the
        envelope and thread tails surface the bot's own reactions. All real Slack reaction
        paths (verdict, work-claim, react_to_message tool) commit through _reserve_and_react,
        so hooking that single choke point covers them once. Best-effort; never breaks
        reacting. Returns the pulse receipt (F38) so an owned reaction that gets taken back
        can take its synthetic history entry back with it."""
        pulse = getattr(self, "channel_pulse", None)
        if pulse is None or not ts:
            return None
        try:
            return pulse.record_own_reaction(channel_id, message_ts=ts, emoji=emoji)
        except Exception as e:
            self.log_debug(f"own-reaction pulse record failed: {e}")
            return None

    # Slack section-block text hard limit is 3000 chars; keep a margin so the reply text
    # fits one section when we attach the footer as blocks.
    _SECTION_TEXT_LIMIT = 2900

    def _compose_reply_with_footer(self, formatted_text: str, footer_blocks: list):
        """Blocks that render the REPLY TEXT plus the footer actions row.

        Attaching action blocks alone makes Slack render blocks INSTEAD of the top-level
        `text`, hiding the answer (only the ⚙️ button shows). So the reply rides a leading
        section block, then the footer actions. Returns None when the text is too long to
        fit a single section — the caller then posts plain text and lets the separate
        footer post happen instead."""
        if not footer_blocks or len(formatted_text) > self._SECTION_TEXT_LIMIT:
            return None
        return [{"type": "section", "text": {"type": "mrkdwn", "text": formatted_text}}] + list(footer_blocks)

    async def send_message(self, channel_id: str, thread_id: str, text: str,
                           blocks: Optional[list] = None,
                           meta_out: Optional[dict] = None,
                           username: Optional[str] = None) -> Optional[str]:
        """Send a text message to Slack, splitting if needed.

        Returns the posted message ts (the FIRST chunk's ts when split), or None on
        failure. Truthy-on-success, so legacy `if await send_message(...)` callers keep
        working while F7 can key tool-use provenance on the returned ts.

        `blocks` (F8): the settings-footer ACTIONS row. When provided AND the reply fits a
        single section block, the reply text + footer ride the message itself (composed via
        _compose_reply_with_footer) instead of a separate trailing post. When the text is
        too long for a section, or the message must split, the footer is NOT attached and
        the plain text posts — the caller's separate footer post covers it.

        `meta_out` (F8): optional dict the caller can read back — `meta_out["footer_attached"]`
        reports whether the footer ACTUALLY rode the message, so the caller sets its
        footer_attached flag from reality (a split/too-long reply must still get the
        separate footer)."""
        def _set_attached(v: bool) -> None:
            if meta_out is not None:
                meta_out["footer_attached"] = v
        try:
            # Strip MCP citations from text before sending to Slack
            text = strip_citations(text)
            # Format text for Slack
            formatted_text = self.format_text(text)

            # Check if we need to split the message
            if len(formatted_text) <= self.MAX_MESSAGE_LENGTH:
                # Single message. Link previews follow ENABLE_LINK_PREVIEWS (default off:
                # links stay inline; no unfurl cards, and no Slack-unfurler "(edited)" marks).
                unfurl = bool(getattr(config, "enable_link_previews", False))
                post_kwargs = dict(channel=channel_id, thread_ts=thread_id, text=formatted_text,
                                   unfurl_links=unfurl, unfurl_media=unfurl)
                composed = self._compose_reply_with_footer(formatted_text, blocks) if blocks else None
                if composed is not None:
                    # text stays as the notification fallback; blocks carry the visible reply.
                    post_kwargs["blocks"] = composed
                # F30: optional username override (labelled research findings). Needs the
                # chat:write.customize scope; without it Slack raises missing_scope and this
                # whole send returns None, so the caller retries the plain path.
                if username:
                    post_kwargs["username"] = username
                # Wall-clock instant just before the post. Reconcile anchors its lower bound
                # here: a message that landed can only carry a Slack ts at or after we tried,
                # so an older own-message (an identical earlier reply) can never be mistaken
                # for this post. Captured on the happy path too — one cheap time.time() — but
                # only ever READ in the ambiguous branch below.
                attempt_start = time.time()
                try:
                    result = await self.app.client.chat_postMessage(**post_kwargs)
                except SlackApiError:
                    # Definitive API rejection — nothing landed. Let the outer handler return
                    # None so the caller's single retry is correct; no reconcile (there is no
                    # ambiguity to resolve, and this path must add zero extra work).
                    raise
                except Exception as transport_error:
                    # AMBIGUOUS transport failure (timeout / connection reset raised AFTER the
                    # request may have already reached Slack). chat.postMessage has no server-side
                    # idempotency key, so before letting the caller re-post — which would double
                    # the reply — reconcile against recent history: if our own message with this
                    # text is already there, the post landed and we return its ts as success. If
                    # it is NOT found (or the reconcile query itself fails) we re-raise unchanged
                    # so the caller's existing single retry still runs (a missing answer is worse
                    # than a rare duplicate).
                    reconciled_ts = await self._reconcile_uncertain_post(
                        channel_id, thread_id, formatted_text, attempt_start)
                    if not reconciled_ts:
                        raise
                    self.log_warning(
                        f"Final post response timed out but the message landed "
                        f"(reconciled ts={reconciled_ts}); not re-posting: {transport_error}")
                    result = {"ts": reconciled_ts}
                posted_ts = result.get("ts")
                # Report footer attachment only AFTER Slack returns a ts — a post that never
                # landed hasn't attached anything, and the separate footer must still fire.
                _set_attached(composed is not None and bool(posted_ts))
                self._record_own_reply_pulse(channel_id, thread_id, posted_ts, text)
                return posted_ts
            else:
                _set_attached(False)  # split replies never attach the footer
                # Split into multiple messages, "Continued..." style (shared markers so
                # the rebuild-side stripper always recognizes them). A failed chunk is
                # retried once (honoring a rate-limit Retry-After); if it STILL fails the
                # remainder is ABORTED — later chunks around a hole read worse than an
                # honest cut — and a loud truncation note posts in its place. Silent
                # partial delivery is never acceptable (Codex review find).
                chunks = self._split_message(formatted_text)
                last = len(chunks) - 1
                first_ts = None
                for i, chunk in enumerate(chunks):
                    body = chunk
                    if i > 0:
                        body = f"{CONTINUATION_HEAD}\n\n{body}"
                    # No "Continued in next message..." trailer (user directive 2026-07-11):
                    # the "...continued" HEAD on the next chunk alone marks the seam, and the
                    # rebuild merger fires on EITHER marker (thread_management merge is OR).
                    unfurl = bool(getattr(config, "enable_link_previews", False))
                    chunk_kwargs = dict(channel=channel_id, thread_ts=thread_id, text=body,
                                        unfurl_links=unfurl, unfurl_media=unfurl)
                    if username:
                        chunk_kwargs["username"] = username  # F30: labelled findings
                    posted = False
                    for attempt in (1, 2):
                        try:
                            result = await self.app.client.chat_postMessage(**chunk_kwargs)
                            if first_ts is None:
                                first_ts = result.get("ts")
                            posted = True
                            break
                        except SlackApiError as chunk_error:
                            self.log_error(
                                f"Error sending message chunk {i + 1}/{last + 1} "
                                f"(attempt {attempt}/2): {chunk_error}")
                            if attempt == 2:
                                break
                            # Honor Slack's Retry-After on 429s; brief pause otherwise.
                            delay = 1.0
                            try:
                                delay = float(getattr(chunk_error, "response", None)
                                              .headers.get("Retry-After", 1))
                            except Exception:
                                pass
                            await asyncio.sleep(min(max(delay, 0.5), 30.0))
                    if not posted:
                        missing = last + 1 - i
                        self.log_error(
                            f"Aborting split post after chunk {i + 1}/{last + 1} failed twice — "
                            f"{missing} part(s) not delivered")
                        if meta_out is not None:
                            meta_out["split_truncated"] = True
                        try:
                            await self.app.client.chat_postMessage(
                                channel=channel_id, thread_ts=thread_id,
                                text=(f"⚠️ This message was cut off — the remaining {missing} "
                                      f"part(s) failed to post to Slack."))
                        except SlackApiError:
                            pass  # posting is broken; the ERROR log above stays loud
                        break
                # Record the full reply once, keyed on the first chunk's ts.
                self._record_own_reply_pulse(channel_id, thread_id, first_ts, text)
                return first_ts
        except SlackApiError as e:
            self.log_error(f"Error sending message: {e}")
            return None

    # Ambiguous-commit reconcile window: an own message this recent that matches what we tried
    # to send is treated as the post that just landed. Slack timestamps are epoch seconds. The
    # primary lower bound is the attempt-start instant (below); this 120s value is a secondary
    # ceiling only.
    _RECONCILE_WINDOW_SECS = 120
    # Slack `ts` is server-stamped while attempt_start is local; allow a little drift so a post
    # that truly landed isn't rejected for a ts a hair before our local clock read.
    _RECONCILE_CLOCK_SKEW_SECS = 5
    # Prefix matching (Slack may append/trim chrome around the fallback text) is only safe for
    # long messages: below this length a short reply can be a prefix of an unrelated longer one.
    _RECONCILE_PREFIX_MIN_LEN = 200
    # conversations.replies returns the EARLIEST in-window messages first, so the freshly-posted
    # tail can sit past the first page. Follow the cursor up to this many pages before giving up.
    _RECONCILE_MAX_PAGES = 3

    @staticmethod
    def _normalize_for_match(text: str) -> str:
        """Whitespace/entity-normalized text for comparing what we SENT against what Slack
        STORED. Slack collapses runs of whitespace and HTML-escapes &/</>, so undo both before
        comparing so a benign normalization can't defeat the match."""
        text = (text or "").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        return " ".join(text.split())

    async def _reconcile_uncertain_post(self, channel_id: str, thread_id: Optional[str],
                                        formatted_text: str, attempt_start: float) -> Optional[str]:
        """After an AMBIGUOUS transport failure on a single-message post, look for the message we
        just tried to send in recent history. Returns its ts when our OWN bot posted matching
        text at or after `attempt_start` (the instant just before this post), else None (the
        caller then reports failure so its single retry runs).

        `formatted_text` is the exact post-conversion payload text handed to chat.postMessage.
        `attempt_start` is a local wall-clock timestamp captured immediately before the post; a
        message that actually landed can only carry a Slack ts at/after it (minus a small skew
        for clock drift), so an identical EARLIER reply can never be mistaken for this one.

        Best-effort: any error querying history returns None — a missing answer is worse than a
        rare duplicate. Only the ambiguous branch calls this, so the happy path pays nothing."""
        target = self._normalize_for_match(formatted_text)
        if not target:
            return None
        # Lower bound anchored to the attempt (minus drift skew). The 120s window is a secondary
        # ceiling: whichever bound is more recent wins, and the attempt-anchored floor normally
        # does, so a stale identical reply is excluded outright.
        floor_ts = attempt_start - self._RECONCILE_CLOCK_SKEW_SECS
        cutoff = max(floor_ts, time.time() - self._RECONCILE_WINDOW_SECS)
        oldest = f"{floor_ts:.6f}"
        min_len = self._RECONCILE_PREFIX_MIN_LEN
        # `oldest` + `inclusive` scope the query to the attempt window. conversations.replies
        # returns the EARLIEST in-window messages first, so the freshly-posted tail can sit past
        # the first page when >100 replies fall in-window — page the cursor (bounded to
        # _RECONCILE_MAX_PAGES) and scan EVERY page. conversations.history returns newest-first so
        # it matches on page 1 in practice, but the loop shape is shared. Any query error anywhere
        # returns None — a missing answer is worse than a rare duplicate.
        cursor: Optional[str] = None
        for _page in range(self._RECONCILE_MAX_PAGES):
            try:
                if thread_id:
                    resp = await self.app.client.conversations_replies(
                        channel=channel_id, ts=thread_id, oldest=oldest, inclusive=True,
                        limit=100, cursor=cursor)
                else:
                    resp = await self.app.client.conversations_history(
                        channel=channel_id, oldest=oldest, inclusive=True,
                        limit=100, cursor=cursor)
                messages = resp.get("messages", []) if resp else []
            except Exception as e:
                self.log_warning(f"Reconcile query failed after uncertain post: {e}")
                return None
            for msg in messages:
                if not self.is_own_message(msg):
                    continue
                try:
                    if float(msg.get("ts", 0)) < cutoff:
                        continue
                except (TypeError, ValueError):
                    continue
                candidate = self._normalize_for_match(msg.get("text", ""))
                if not candidate:
                    continue
                if candidate == target:
                    return msg.get("ts")
                # Full-prefix match (Slack may append/trim chrome around the fallback text) only
                # when BOTH normalized strings are long — otherwise a short reply is a prefix of an
                # unrelated longer one ("OK" vs "OK, done") and a genuinely lost new post gets
                # swallowed. Compare the WHOLE shorter string, never a 200-char head: two long
                # replies sharing a 200-char boilerplate prefix then diverging must NOT match.
                if (len(candidate) >= min_len and len(target) >= min_len
                        and (candidate.startswith(target) or target.startswith(candidate))):
                    return msg.get("ts")
            cursor = (resp.get("response_metadata") or {}).get("next_cursor") if resp else None
            if not cursor:
                break
        return None

    def _split_message(self, text: str) -> List[str]:
        """Split a long message into chunks that fit within Slack's limit.

        Fence-aware (code blocks are closed and reopened across the seam) and
        entity-safe; oversized single fragments are hard-wrapped instead of
        being sent as-is (which used to msg_too_long and abort the remainder).
        Margin covers the continuation markers + fence reopen prefix.
        """
        return fence_safe_chunks(text, self.MAX_MESSAGE_LENGTH - 150)

    async def send_message_get_ts(self, channel_id: str, thread_id: str, text: str) -> Dict:
        """Send a message and return the response including timestamp"""
        try:
            # Strip MCP citations from text before sending to Slack
            text = strip_citations(text)
            # Format text for Slack
            formatted_text = self.format_text(text)
            
            # Safety check - this should never happen for continuation messages
            # but if somehow the text is too long, truncate it
            if len(formatted_text) > self.MAX_MESSAGE_LENGTH:
                formatted_text = formatted_text[:self.MAX_MESSAGE_LENGTH - 80] + "\n\n*[Message exceeded Slack limit]*"
            
            result = await self.app.client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_id,
                text=formatted_text
            )
            
            return {"success": True, "ts": result["ts"]}
        except SlackApiError as e:
            self.log_error(f"Error sending message: {e}")
            return {"success": False, "error": str(e)}

    async def send_image(self, channel_id: str, thread_id: str, image_data: bytes, filename: str,
                         caption: str = "", meta_out: Optional[dict] = None) -> Optional[str]:
        """Send an image to Slack and return the file URL.

        `meta_out` (F7): optional dict the caller reads back — `meta_out["file_id"]` is the
        uploaded file's id, set only once Slack accepts the upload. The RETURN stays the bare
        URL (base_client / slack_client.base declare that contract), so the file id rides a
        side channel rather than breaking every existing caller. It exists because
        files_upload_v2 hands back no share ts: the file id is the only handle from which the
        image message's ts can later be resolved (see resolve_file_share_ts).
        """
        try:
            # Use files_upload_v2 for image upload
            result = await self.app.client.files_upload_v2(
                channel=channel_id,  # Changed from channels to channel (singular)
                thread_ts=thread_id,
                file=image_data,
                filename=filename,
                initial_comment=caption
            )

            # Extract the file URL from the response
            if result and "files" in result and len(result["files"]) > 0:
                file_info = result["files"][0]
                file_url = file_info.get("url_private", file_info.get("permalink"))
                if meta_out is not None:
                    meta_out["file_id"] = file_info.get("id")
                self.log_info(f"Image uploaded: {filename} - URL: {file_url}")
                return file_url
            else:
                self.log_warning("Image uploaded but no URL found in response")
                return None

        except SlackApiError as e:
            self.log_error(f"Error uploading image: {e}")
            return None

    # Poll schedule for resolve_file_share_ts, in seconds. Mild backoff rather than a fixed
    # tight interval: the share lands ~1.8s (channel) to ~3.8s (DM) after upload, so a 0.4s
    # loop would burn ~10 calls on a DM to learn nothing the 5th poll wouldn't have.
    _SHARE_TS_BACKOFF = (0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 4.0)

    # Slack errors that will not come right inside the budget, so polling on until the
    # deadline is pure waste. Everything ELSE is worth another poll while budget remains —
    # including `file_not_found`, which here is the upload's own eventual consistency (the
    # very race this poll exists to paper over), not a verdict that the file isn't real.
    _SHARE_TS_PERMANENT_ERRORS = frozenset({
        "invalid_auth", "not_authed", "missing_scope", "access_denied"})

    @staticmethod
    def _share_ts_error_code(error: SlackApiError) -> str:
        """Slack's machine-readable `error` string, or "" when the shape isn't what we expect
        (an unrecognized code is treated as transient, which is the safe default here)."""
        response = getattr(error, "response", None)
        if response is None:
            return ""
        try:
            return str(response.get("error") or "")
        except (AttributeError, TypeError):
            return ""

    @staticmethod
    def _share_ts_retry_after(error: SlackApiError) -> Optional[float]:
        """Seconds Slack asked us to wait on a 429, if it said. Header lookup is
        case-insensitive because the casing varies with the transport underneath the SDK."""
        headers = getattr(getattr(error, "response", None), "headers", None)
        if not headers:
            return None
        try:
            for key, value in headers.items():
                if str(key).lower() == "retry-after":
                    return max(0.0, float(value))
        except (AttributeError, TypeError, ValueError):
            return None
        return None

    async def resolve_file_share_ts(self, channel_id: str, file_id: str) -> Optional[str]:
        """The ts of the message that shares an uploaded file, or None.

        Why this exists: files_upload_v2's response DOES carry a `shares` key, and it is
        always `{}` at upload time — Slack populates it asynchronously, so the share ts is
        only readable from a later files.info call. Measured live 2026-07-16: the entry
        appeared ~1.76s after upload in a private channel and ~3.81s in a DM (DMs are
        markedly slower).

        The entry sits at `shares["private"][channel_id][0]` for private channels AND DMs;
        public channels use `shares["public"][channel_id][0]`. Both scopes are checked — the
        caller has no way to know which applies. That entry's `ts` IS the file-share
        message's ts (cross-checked against conversations.replies / conversations.history).

        A transient failure is RETRIED within the budget rather than surrendering the row: a
        429 or a blip is not an answer, and giving up on the first one throws away provenance
        that a second poll would have had. Only clearly permanent errors bail early.

        Best-effort chrome: a timeout, a SlackApiError, or any other failure returns None and
        is logged, never raised. The image is already posted by the time anyone calls this,
        and provenance must never be able to touch it.
        """
        if not channel_id or not file_id:
            return None
        deadline = time.monotonic() + max(0.0, float(config.image_share_ts_timeout_seconds))
        attempt = 0
        while True:
            remaining = deadline - time.monotonic()
            # Budget is checked BEFORE the request, not after: waking exactly AT the deadline
            # and polling once more is how a "15s bound" quietly becomes 15s plus a request.
            # Attempt 0 is the deliberate exception — the share is often already there, so the
            # first poll is always worth making even on an exhausted budget.
            if attempt and remaining <= 0:
                self.log_debug(f"share ts for {file_id} did not appear before the timeout")
                return None

            retry_after: Optional[float] = None
            try:
                # The budget bounds the CALL too, not just the gaps between calls, or one hung
                # request sails past the deadline on its own (the SDK's default timeout being
                # the only other ceiling). The guaranteed first poll has no budget left to be
                # bounded by, so it keeps that SDK default.
                result = await asyncio.wait_for(
                    self.app.client.files_info(file=file_id),
                    timeout=remaining if remaining > 0 else None)
                shares = ((result or {}).get("file") or {}).get("shares") or {}
                for scope in ("public", "private"):
                    entries = (shares.get(scope) or {}).get(channel_id) or []
                    if entries and entries[0].get("ts"):
                        return entries[0]["ts"]
            except SlackApiError as e:
                if self._share_ts_error_code(e) in self._SHARE_TS_PERMANENT_ERRORS:
                    self.log_debug(f"files.info share-ts lookup gave up for {file_id}: {e}")
                    return None
                retry_after = self._share_ts_retry_after(e)
                self.log_debug(f"files.info share-ts lookup will retry for {file_id}: {e}")
            except Exception as e:  # noqa: BLE001 — never load-bearing; see docstring
                # Transport blips and our own call timeout above: transient like a 429, so
                # they buy another poll rather than costing the row.
                self.log_debug(f"share-ts resolve error for {file_id}: {e}")

            delay = self._SHARE_TS_BACKOFF[min(attempt, len(self._SHARE_TS_BACKOFF) - 1)]
            if retry_after is not None:
                # Slack said when to come back; polling sooner just earns another 429.
                delay = max(delay, retry_after)
            attempt += 1
            remaining = deadline - time.monotonic()
            if remaining > 0:
                await asyncio.sleep(min(delay, remaining))

    async def send_file(self, channel_id: str, thread_id: str, file_data,
                        filename: str, title: Optional[str] = None,
                        initial_comment: str = "") -> Optional[Dict[str, Any]]:
        """F32: upload an arbitrary file (BytesIO) and return its full Slack identity.

        Distinct from send_image, which returns a bare URL: an artifact has to be findable
        again, so callers get {"file_id", "url_private", "permalink"} to persist. The
        file_id is what `read_document` looks up, so without it the model could never
        re-read its own artifact.

        Returns None on any failure — the caller decides whether that's fatal (for an
        artifact it never is: the text answer already landed).
        """
        try:
            result = await self.app.client.files_upload_v2(
                channel=channel_id,
                thread_ts=thread_id,
                file=file_data,
                filename=filename,
                title=title or filename,
                initial_comment=initial_comment or "",
            )
            files = (result or {}).get("files") or []
            if not files:
                self.log_warning(f"File uploaded but no file info returned: {filename}")
                return None
            info = files[0]
            file_id = info.get("id")
            url = info.get("url_private") or info.get("permalink")
            if not file_id or not url:
                # Without an id we could never find this file again, and the caller would
                # persist a ref that points at nothing. A response we can't use is not success.
                self.log_warning(
                    f"File upload returned no usable identity (id={file_id!r}): {filename}")
                return None
            identity = {"file_id": file_id, "url_private": url,
                        "permalink": info.get("permalink")}
            self.log_info(f"File uploaded: {filename} (id={file_id})")
            return identity
        except SlackApiError as e:
            self.log_error(f"Error uploading file '{filename}': {e}")
            return None
        except Exception as e:  # noqa: BLE001 — an upload problem must never break the turn
            self.log_error(f"Unexpected error uploading file '{filename}': {e}")
            return None

    async def send_thinking_indicator(self, channel_id: str, thread_id: str) -> Optional[str]:
        """Show a progress indicator; returns the placeholder message ts, or None.

        Contract: assistant.threads.setStatus is the SOLE indicator wherever Slack
        accepts it — the June-2026 agent surface renders the composer status in DMs
        AND channel threads (verified live 2026-07-09: a channel thread showed both
        the status line and our redundant placeholder). Native status means no
        message, no "(edited)" churn, auto-clears on reply. Returns None; downstream
        consumers treat a None ts as "status-only" (streaming seeds its own message
        lazily, phase updates route to setStatus, deletes no-op).

        Only where setStatus FAILS (non-agent contexts, older surfaces) do we post
        the classic "Thinking..." placeholder and return its ts.
        """
        status_set = await self.set_assistant_status(channel_id, thread_id)
        if status_set:
            return None
        try:
            result = await self.app.client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_id,
                text=f"{config.circle_loader_emoji} {config.random_loading_message()}"
            )
            return result.get("ts")  # Return message timestamp for deletion
        except SlackApiError as e:
            self.log_error(f"Error sending thinking indicator: {e}")
            return None

    async def delete_message(self, channel_id: str, message_id: str) -> bool:
        """Delete a message from Slack"""
        try:
            await self.app.client.chat_delete(
                channel=channel_id,
                ts=message_id
            )
            return True
        except SlackApiError as e:
            self.log_debug(f"Could not delete message: {e}")
            return False

    async def update_message(self, channel_id: str, message_id: str, text: str) -> bool:
        """Update a message in Slack"""
        try:
            # Strip MCP citations from text before sending to Slack
            text = strip_citations(text)
            await self.app.client.chat_update(
                channel=channel_id,
                ts=message_id,
                text=text,
                mrkdwn=True  # Enable markdown parsing for italics/bold
            )
            return True
        except SlackApiError as e:
            self.log_error(f"Could not update message: {e}")
            return False

    async def post_status_card(self, channel_id: str, thread_id: str, text: str,
                               blocks: list, username: Optional[str] = None) -> Optional[str]:
        """F30.1: post a blocks status card (e.g. the deep-research todo card). Returns the
        posted ts, or None on failure. `text` is the CONSTANT notification fallback; `blocks`
        carry the visible card. `username` optionally labels the poster (needs the
        chat:write.customize scope; without it Slack raises missing_scope → None so the caller
        can retry unlabeled). Best-effort — never raises."""
        try:
            kwargs = dict(channel=channel_id, thread_ts=thread_id, text=text, blocks=blocks,
                          unfurl_links=False, unfurl_media=False)
            if username:
                kwargs["username"] = username
            result = await self.app.client.chat_postMessage(**kwargs)
            return result.get("ts")
        except SlackApiError as e:
            self.log_warning(f"Status card post failed: {e}")
            return None

    async def update_status_card(self, channel_id: str, ts: str, text: str,
                                 blocks: list) -> bool:
        """F30.1: update a blocks status card in place. `text` MUST stay CONSTANT across
        updates (Slack badges '(edited)' only when the top-level text changes; blocks-only
        edits don't badge). Best-effort — returns False on failure, never raises."""
        try:
            await self.app.client.chat_update(
                channel=channel_id, ts=ts, text=text, blocks=blocks)
            return True
        except SlackApiError as e:
            self.log_debug(f"Status card update failed: {e}")
            return False

    async def _replies_page_with_retry(self, kwargs: Dict, attempts: int = 3):
        """One conversations.replies page, honoring Retry-After on 429s (R1).

        Retries only rate-limit errors; anything else propagates immediately.
        """
        for attempt in range(attempts):
            try:
                return await self.app.client.conversations_replies(**kwargs)
            except SlackApiError as e:
                err = e.response.get("error") if getattr(e, "response", None) else None
                status = getattr(getattr(e, "response", None), "status_code", None)
                if (err == "ratelimited" or status == 429) and attempt < attempts - 1:
                    headers = getattr(getattr(e, "response", None), "headers", None) or {}
                    try:
                        delay = float(headers.get("Retry-After") or 1)
                    except (TypeError, ValueError):
                        delay = 1.0
                    delay = min(max(delay, 0.5), 30.0)
                    self.log_warning(
                        f"conversations.replies rate-limited (attempt {attempt + 1}/{attempts}), "
                        f"retrying in {delay}s"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

    async def get_thread_history(self, channel_id: str, thread_id: str, limit: int = None,
                                 oldest: str = None) -> List[Message]:
        """Get COMPLETE thread history from Slack - fetches ALL messages by default.

        `oldest` (Slack ts) fetches only messages strictly after it (Slack's default
        inclusive=false) — used by the Phase S summary-tail rebuild so a compacted
        1500-message thread doesn't refetch the summarized head.

        Raises HistoryFetchError on terminal fetch failure. Since Phase S, Slack is
        the ONLY transcript — returning [] here would make the bot answer with
        amnesia, silently. An actually-empty thread still returns [] normally.
        """
        messages = []

        try:
            # Fetch ALL messages using pagination
            cursor = None
            total_fetched = 0

            while True:
                # Slack's max per request is 1000
                per_request_limit = 1000
                if limit and limit - total_fetched < 1000:
                    per_request_limit = limit - total_fetched

                kwargs = {
                    "channel": channel_id,
                    "ts": thread_id,
                    "limit": per_request_limit
                }
                if oldest:
                    kwargs["oldest"] = oldest
                if cursor:
                    kwargs["cursor"] = cursor

                result = await self._replies_page_with_retry(kwargs)
                slack_messages = result.get("messages", [])
                
                if not slack_messages:
                    break
                    
                # Process messages from this batch
                for msg in slack_messages:
                    text = msg.get("text", "")
                    # Determine sender up front — the placeholder/status filters below
                    # must only ever skip OUR OWN messages (R3: a human saying
                    # "Thinking about the Q3 plan" is real context, not a placeholder).
                    sender_type = self.classify_sender(msg)
                    # Our OWN transient chrome (status/placeholder/checklist/"Settings
                    # available"/pure UI-helper block) must never replay as an assistant turn.
                    # Factored into is_self_chrome_message so this path and the F47 cold-start
                    # backfill (channel_pulse.ensure_backfill) can never drift.
                    if sender_type == "self" and is_self_chrome_message(text, msg):
                        continue
                    # A non-self message can still be a pure UI-helper block (its action_ids,
                    # never text — so a real reply can't false-positive); skip it too.
                    if sender_type != "self" and _is_ui_helper_message(msg):
                        continue

                    # sender_type computed above (drives both the self-only filters and metadata)
                    is_bot = bool(msg.get("bot_id"))  # kept for existing readers ("any bot")
                    # Display name for bot authors (used to name-prefix other bots like humans)
                    bot_name = msg.get("username") or (msg.get("bot_profile") or {}).get("name")

                    # Clean text
                    text = msg.get("text", "")
                    # F48 — the DURABILITY half. Slack is the ONLY transcript, so fixing
                    # only the live path buys exactly one turn: a table ingests Monday and
                    # Tuesday's rebuild re-drops it, leaving the bot with amnesia about
                    # content it already discussed. Rendered RAW and combined BEFORE the
                    # mention pass below, matching the live path so a message serializes
                    # identically live and rebuilt. Never for our OWN messages: our status
                    # and welcome cards live in these fields (F47 attribution bug).
                    if sender_type != "self":
                        supplementary = extract_supplementary_text(msg, primary_text=text)
                        if supplementary:
                            text = f"{text}\n\n{supplementary}" if text.strip() else supplementary
                    if not is_bot:
                        text = self._clean_mentions(text)
                    
                    # Check for files
                    attachments = []
                    files = msg.get("files", [])
                    for file in files:
                        # Determine file type based on mimetype
                        mimetype = file.get("mimetype", "")
                        file_type = "image" if mimetype.startswith("image/") else "file"
                        
                        attachments.append({
                            "type": file_type,
                            "name": file.get("name"),
                            "mimetype": mimetype,
                            "url": file.get("url_private", file.get("permalink")),
                            # Match the live path (message_events) so a rebuilt Message carries the
                            # same provenance: the file id and declared size enable later
                            # re-download / size-gate decisions.
                            "id": file.get("id"),
                            "size": file.get("size"),
                        })
                    
                    messages.append(Message(
                        text=text,
                        user_id=msg.get("user", "bot" if is_bot else "unknown"),
                        channel_id=channel_id,
                        thread_id=thread_id,
                        attachments=attachments,
                        metadata={
                            "ts": msg.get("ts"),
                            "is_bot": is_bot,
                            "sender_type": sender_type,
                            "bot_name": bot_name,
                            # Raw reactions from conversations.replies (name/users/count) —
                            # rendered into rebuilt context as a deterministic annotation
                            "reactions": msg.get("reactions") or None
                        }
                    ))
                
                total_fetched += len(slack_messages)
                
                # Check if we've hit our limit
                if limit and total_fetched >= limit:
                    break
                
                # Check for pagination
                response_metadata = result.get("response_metadata", {})
                next_cursor = response_metadata.get("next_cursor")
                
                if not next_cursor:
                    # No more messages
                    break
                    
                cursor = next_cursor
                # Continue to next iteration
            
            self.log_info(f"Fetched {len(messages)} messages from thread {thread_id}")
            return messages
            
        except SlackApiError as e:
            # Terminal failure (rate-limit retries exhausted or a hard API error).
            # Do NOT return [] — Slack is the only transcript, and an empty result
            # here would silently rebuild the thread with no context (R1).
            self.log_error(f"Error getting thread history: {e}")
            raise HistoryFetchError(
                f"Could not fetch thread history for {channel_id}:{thread_id}: {e}"
            ) from e

    def supports_streaming(self) -> bool:
        """Returns True if streaming is enabled for Slack"""
        return config.enable_streaming and config.slack_streaming

    def get_streaming_config(self) -> Dict:
        """Returns platform-specific streaming configuration"""
        return {
            "update_interval": config.streaming_update_interval,
            "min_interval": config.streaming_min_interval,
            "max_interval": config.streaming_max_interval,
            "buffer_size": config.streaming_buffer_size,
            "circuit_breaker_threshold": config.streaming_circuit_breaker_threshold,
            "circuit_breaker_cooldown": config.streaming_circuit_breaker_cooldown,
            "platform": "slack"
        }

    def supports_native_streaming(self) -> bool:
        """True if native Slack streaming (chat.startStream/appendStream/stopStream) is enabled
        and available on the SDK. Default OFF via config pending live dev-bot verification."""
        return (
            config.slack_native_streaming
            and self.supports_streaming()
            and hasattr(self.app.client, "chat_startStream")
        )

    def begin_native_stream(self, channel_id: str, thread_id: str,
                            user_id: Optional[str] = None) -> "NativeStreamSession":
        """Create a (not-yet-started) NativeStreamSession bound to this channel/thread.

        chat.startStream requires recipient_team_id (workspace, resolved once via auth.test
        in _ensure_self_identity) AND recipient_user_id (the triggering user, plumbed in by
        the handler) for channel streaming — both are threaded onto the session here."""
        return NativeStreamSession(
            self.app.client, channel_id, thread_id, logger=self.log_debug,
            team_id=getattr(self, "self_team_id", None), user_id=user_id)

    async def set_assistant_status(self, channel_id: str, thread_id: str,
                                   status: Optional[str] = None,
                                   loading_messages: Optional[List[str]] = None) -> bool:
        """Best-effort assistant.threads.setStatus (Phase 3.2).

        Shows a transient 'thinking/working' status on the assistant-thread surface with a
        rotating branded loading_messages set; auto-clears when the app replies. Degrades to a
        silent no-op in plain channels / non-assistant contexts — must never raise.

        GUARD (Phase G / agent_view): on the June-2026 surface setStatus AUTO-OPENS the
        thread for the user. Never call this speculatively for a channel message the
        participation engine might still ignore — the only caller is
        send_thinking_indicator, which main.py invokes strictly AFTER the engine's
        'respond' verdict. Keep it that way (regression-tested in test_phase_g.py).
        """
        if not config.enable_assistant_status:
            return False
        if not hasattr(self.app.client, "assistant_threads_setStatus"):
            return False
        # Slack API contract (verified live 2026-07-10): a NON-EMPTY `status`
        # string is what renders — status:"" is the CLEAR signal and hides the
        # indicator entirely, loading_messages never render without a status,
        # and an empty loading_messages array is rejected ("must provide at
        # least 1 items"). So every visible update sends ONE text in BOTH
        # fields, keeping the in-thread transient and the composer line
        # identical (mismatched texts read as two indicators; user screenshots
        # 2026-07-09/10). Variety comes from the pools: the initial call picks
        # a random loading message, phase updates pick a random stage variant.
        # An explicit status="" (clear_assistant_status) goes out bare.
        if status == "":
            msgs = []  # clear: bare empty status, never loading_messages
            status_text = ""
        elif status is not None:
            msgs = loading_messages if loading_messages is not None else [status]
            status_text = status
        elif loading_messages:
            msgs = loading_messages
            status_text = loading_messages[0]
        else:
            pool = config.get_loading_messages() or [config.status_loading_fallback]
            pick = random.choice(pool)
            msgs = [pick]
            status_text = pick
        try:
            kwargs = {"channel_id": channel_id, "thread_ts": thread_id,
                      "status": _status_plain_text(status_text) if status_text else ""}
            texts = [t for t in (_status_plain_text(m) for m in msgs) if t]
            if texts:
                kwargs["loading_messages"] = texts
            await self.app.client.assistant_threads_setStatus(**kwargs)
            return True
        except SlackApiError as e:
            # Most common in a plain channel: not an assistant thread -> just skip it.
            err = e.response.get("error") if getattr(e, "response", None) else e
            self.log_debug(f"assistant setStatus unavailable here ({err}); continuing without it")
            return False
        except Exception as e:  # noqa: BLE001
            self.log_debug(f"assistant setStatus error: {e}")
            return False

    async def clear_assistant_status(self, channel_id: str, thread_id: str) -> bool:
        """Clear the assistant status: bare status="" with NO loading_messages (the API
        rejects an empty array and treats "" as the clear signal). Needed explicitly for
        native-streamed replies — Slack's auto-clear keys on chat.postMessage only."""
        return await self.set_assistant_status(channel_id, thread_id, status="")

    async def _react_add(self, channel_id: str, message_ts: str, emoji: str) -> tuple:
        """The raw add, returning (ok, added).

        `added` is the bit `react()` throws away and F38 needs: True only when THIS call
        actually put the reaction there. Slack's `already_reacted` is still ok=True — the
        emoji is present, the caller's intent is satisfied — but added=False, because a
        reaction we did not place is not a reaction we may take away. (Slack scopes
        reactions per user, so `already_reacted` can only mean the BOT reacted before, never
        that a human did.)"""
        if not config.enable_reactions:
            return False, False
        name = (emoji or "").strip().strip(":")
        if not name:
            return False, False
        try:
            await self.app.client.reactions_add(channel=channel_id, name=name, timestamp=message_ts)
            return True, True
        except SlackApiError as e:
            err = e.response.get("error") if getattr(e, "response", None) else str(e)
            if err == "already_reacted":
                return True, False  # present, but not ours to remove
            self.log_warning(f"Could not add reaction :{name}: ({err})")
            return False, False
        except Exception as e:  # noqa: BLE001
            self.log_error(f"Unexpected error adding reaction :{name}: {e}")
            return False, False

    async def react(self, channel_id: str, message_ts: str, emoji: str) -> bool:
        """Add an emoji reaction to a message (Phase 4). ``emoji`` may include or omit colons.

        Best-effort: treats already_reacted as success, never raises.
        """
        ok, _added = await self._react_add(channel_id, message_ts, emoji)
        return ok

    async def unreact(self, channel_id: str, message_ts: str, emoji: str) -> bool:
        """Remove one of the BOT'S OWN reactions (F38). Slack scopes reactions.remove to the
        authenticated user, so this can never strip a human's emoji off a message.

        `no_reaction` counts as success — the goal state is "the emoji is not there", and
        something else having removed it already satisfies that. Never raises."""
        if not config.enable_reactions:
            return False
        name = (emoji or "").strip().strip(":")
        if not name:
            return False
        try:
            await self.app.client.reactions_remove(
                channel=channel_id, name=name, timestamp=message_ts)
            return True
        except SlackApiError as e:
            err = e.response.get("error") if getattr(e, "response", None) else str(e)
            if err == "no_reaction":
                return True  # already gone — the intended end state
            self.log_warning(f"Could not remove reaction :{name}: ({err})")
            return False
        except Exception as e:  # noqa: BLE001
            self.log_error(f"Unexpected error removing reaction :{name}: {e}")
            return False

    # --- react_to_message local tool (redesign Phase D) ---

    # ~600-char budget for the custom-emoji list injected into a schema/classifier description,
    # so surfacing customs never bloats every main-model request.
    _CUSTOM_EMOJI_CHAR_BUDGET = 600

    def _budgeted_custom_emoji_names(self, count_cap: int) -> list:
        """A deterministic, budgeted slice of the workspace custom-emoji names for a schema
        description: at most ``count_cap`` names AND within the ~600-char budget. Reads the
        sync, stale-ok cache getter; returns [] when there are no customs (or no cache)."""
        cache = getattr(self, "workspace_emojis", None)
        if cache is None:
            return []
        try:
            names = cache.get_custom_emoji_names()
        except Exception:  # noqa: BLE001 — a schema build must never fail the turn
            return []
        cap = max(0, int(count_cap or 0))
        capped = names[:cap]  # hard max: 0 → none, never "unlimited"
        out, used = [], 0
        for name in capped:
            cost = len(name) + 2  # +2 approximates the ", " separator between names
            if out and used + cost > self._CUSTOM_EMOJI_CHAR_BUDGET:
                break
            out.append(name)
            used += cost
        return out

    def get_react_tool_schema(self, cfg: Optional[dict] = None) -> dict:
        """Registry FACTORY (called per request as ``schema(cfg)``) for the react_to_message
        tool. By default the model may pick ANY standard Slack emoji shorthand name — choosing
        the right one IS the judgment. If REACTION_EMOJIS is configured, it constrains the
        choice to that allowlist via an enum (brand control), and customs are NOT injected. When
        no allowlist is set, the workspace's custom emoji are surfaced as EXTRA named choices in
        the field DESCRIPTION (never an enum — an enum would forbid every standard emoji)."""
        allowed = [e.strip().strip(":") for e in (config.reaction_emojis or []) if e and e.strip().strip(":")]
        emoji_schema = {"type": "string",
                        "description": "Any standard Slack emoji shorthand name (no colons), e.g. joy, tada, fire."}
        if allowed:
            emoji_schema["enum"] = allowed
        else:
            customs = self._budgeted_custom_emoji_names(
                getattr(config, "react_tool_custom_emoji_cap", 64))
            if customs:
                emoji_schema["description"] += (
                    " This workspace also has custom emoji you may use when one fits, e.g.: "
                    + ", ".join(customs) + "."
                )
        return {
            "type": "function",
            "name": "react_to_message",
            "description": (
                "Add an emoji reaction to a Slack message, the way a teammate would — when "
                "something lands, when you agree, when the room is already reacting, or to "
                "acknowledge a completed request (a ✅, a celebration). If a reaction alone "
                "fully answers the message, react and reply with empty text. Defaults to the "
                "message you are answering. Call once per emoji when asked for multiple."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "emoji": emoji_schema,
                    "ts": {"type": "string",
                           "description": "Optional ts of another recent message in this channel to react to."},
                },
                "required": ["emoji"],
            },
        }

    async def execute_react_tool(self, ctx, args: dict) -> dict:
        """Executor for react_to_message. Syntactic emoji validation (Slack's invalid_name
        is the semantic backstop) + optional REACTION_EMOJIS allowlist + per-message cap
        (REACTION_MAX_PER_MESSAGE distinct emoji); never raises (returns
        {"ok": False, ...} on any refusal/failure)."""
        if not (config.enable_reactions and config.enable_react_tool):
            return {"ok": False, "error": "disabled", "message": "Reactions are disabled."}
        emoji = (args.get("emoji") or "").strip().strip(":")
        if not valid_emoji_name(emoji):
            return {"ok": False, "error": "invalid_emoji", "message": "Not a valid emoji shorthand name."}
        allowed = {e.strip().strip(":") for e in (config.reaction_emojis or []) if e and e.strip().strip(":")}
        if allowed and emoji not in allowed:
            return {"ok": False, "error": "emoji_not_allowed", "allowed": sorted(allowed)}
        channel_id = getattr(ctx, "channel_id", None)
        ts = (args.get("ts") or "").strip() or getattr(ctx, "trigger_ts", None) or getattr(ctx, "thread_ts", None)
        if not channel_id or not ts:
            return {"ok": False, "error": "no_target", "message": "No message to react to."}
        return await self._reserve_and_react(channel_id, ts, emoji)

    # Reaction-guard eviction tuning. Entries touched within the recency window are PINNED
    # (never evicted) — this covers both the committed slots of an ACTIVE turn (so a burst
    # of 2000+ reactions on other messages can't resurrect a message's consumed slots) and
    # fresh pending reservations. A pending future untouched for the whole window is treated
    # as abandoned and becomes evictable (bounded expiry for a never-resolving Future).
    _REACTION_GUARD_MAX = 2000
    _REACTION_GUARD_RECENCY_S = 120.0

    def _trim_reaction_guard(self, guard, ts_map, now, keep=None) -> None:
        """Evict oldest guard entries beyond the cap, pinning anything touched within the
        recency window (and always ``keep``).

        F38: an entry holding a LIVE OWNED slot is pinned unconditionally. Ownership is what
        lets a turn take its 👀 back, and a long turn (a research job runs for minutes) would
        otherwise age out of the recency window and lose the right to clean up after itself."""
        if len(guard) <= self._REACTION_GUARD_MAX:
            return
        cutoff = now - self._REACTION_GUARD_RECENCY_S
        for k in list(guard.keys()):
            if len(guard) <= self._REACTION_GUARD_MAX:
                break
            entry = guard.get(k)
            if entry is keep:
                continue
            if any(isinstance(v, dict) for v in (entry or {}).values()):
                continue  # a live claim lives here — evicting it would strand the 👀
            if ts_map.get(k, 0.0) >= cutoff:
                continue  # recently touched → pinned (active-turn committed or fresh pending)
            del guard[k]
            ts_map.pop(k, None)

    # --- F38: reaction leases (a 👀 the turn can take back) ---
    #
    # Ownership is part of the GUARD, not a map beside it. That matters, and it took a review
    # round to see why: a parallel map cannot be kept honest, because Slack's `already_reacted`
    # is silent about WHO reacted. Sequence that breaks the parallel design —
    #
    #   1. turn A adds 👀 and records itself as owner
    #   2. A's guard entry is evicted (2000-entry LRU)
    #   3. turn B reserves the same emoji: no slot, so it calls Slack
    #   4. Slack: `already_reacted` (A's 👀 is still up there) → B gets no lease...
    #   5. ...and B never overwrites A's ownership record, because it never had one to write
    #   6. A ends silently, its token still "matches", and it rips the 👀 out from under B
    #
    # Holding ownership in the slot makes step 2 self-correcting: eviction destroys the claim,
    # so A can no longer prove the emoji is its own and declines to touch it. Losing the right
    # to clean up is the safe failure; removing someone else's reaction is not.
    #
    # A slot is therefore one of:
    #   Future              an add is in flight (a concurrent sibling reserved it first)
    #   True                committed, unowned — nobody may remove it
    #   {"token": ...}      committed and OWNED by the turn holding the matching lease
    #   {"token", "removing"}  that owner is mid-removal; no one else may touch it
    _REMOVING = "removing"

    @staticmethod
    def _is_committed(slot) -> bool:
        """True/owned/removing all mean 'the emoji is on the message'. A Future does not."""
        return slot is True or isinstance(slot, dict)

    def settle_reaction_lease(self, lease: Optional[dict]) -> None:
        """The turn produced something: the reaction has earned its place. Drop the claim —
        the emoji, the guard slot and the pulse entry all stay exactly as they are.

        Releasing matters even for reactions nobody intends to remove (a gate verdict, the
        model's own react tool): an owned slot is pinned against eviction, so never settling
        one would slowly fill the guard with unevictable entries."""
        if not lease:
            return
        slots = (getattr(self, "_reaction_guard", None) or {}).get(
            (lease.get("channel_id"), lease.get("ts")))
        if slots is None:
            return
        slot = slots.get(lease.get("emoji"))
        if isinstance(slot, dict) and slot.get("token") == lease.get("token"):
            slots[lease["emoji"]] = True  # committed, unowned, evictable again

    def _settle_removal_slot(self, channel_id: str, ts: str, emoji: str, token: str,
                             ok: bool, lease: dict) -> None:
        """Transition a `removing` slot to its final state. Runs from the removal TASK's
        `finally`, so it happens even if the turn that asked for the removal is cancelled —
        otherwise the slot would stay `removing` forever, and since owned slots are pinned
        against eviction, a run of cancelled turns would grow the guard without bound."""
        guard = getattr(self, "_reaction_guard", None)
        slots = (guard or {}).get((channel_id, ts))
        slot = slots.get(emoji) if slots is not None else None
        if not (isinstance(slot, dict) and slot.get("token") == token
                and slot.get(self._REMOVING) is not None):
            return  # someone else already resolved it; don't stomp their state
        if not ok:
            # The emoji may well still be up there. Demote to committed-unowned rather than
            # dropping the slot: a stale 👀 is survivable, a guard that thinks a live reaction
            # is gone is not (it would let the cap be exceeded and re-add over the top).
            slots[emoji] = True
            self.log_debug(f"Could not take back :{emoji}: — leaving the bookkeeping intact")
            return
        slots.pop(emoji, None)
        if not slots and guard is not None and guard.get((channel_id, ts)) is slots:
            guard.pop((channel_id, ts), None)
            ts_map = getattr(self, "_reaction_guard_ts", None)
            if ts_map is not None:
                ts_map.pop((channel_id, ts), None)
        pulse = getattr(self, "channel_pulse", None)
        if pulse is not None and lease.get("pulse_receipt"):
            try:
                pulse.remove_own_reaction(lease["pulse_receipt"])
            except Exception as e:  # noqa: BLE001 — history cleanup must never break a turn
                self.log_debug(f"own-reaction pulse removal failed: {e}")
        self.log_debug(f"Took back :{emoji}: — the turn produced nothing")

    async def _run_reaction_removal(self, channel_id: str, ts: str, emoji: str, token: str,
                                    lease: dict) -> bool:
        """The removal itself, as its own task. Bounded, and it ALWAYS settles the slot."""
        ok = False
        try:
            ok = await asyncio.wait_for(
                self.unreact(channel_id, ts, emoji),
                timeout=max(1.0, float(getattr(config, "tool_call_timeout", 20))))
        except asyncio.TimeoutError:
            self.log_debug(f"Reaction removal timed out for :{emoji}:")
        except Exception as e:  # noqa: BLE001
            self.log_debug(f"Reaction removal failed for :{emoji}: {e}")
        finally:
            # `finally`, not `except Exception` — a CancelledError is a BaseException and
            # would otherwise sail straight past, stranding the slot in `removing`.
            self._settle_removal_slot(channel_id, ts, emoji, token, ok, lease)
        return ok

    async def remove_owned_reaction(self, lease: Optional[dict]) -> bool:
        """The turn produced nothing: take the reaction back off.

        Refuses unless the slot still carries OUR token — if the entry was evicted and
        re-committed, or another turn claimed the emoji, it is no longer ours and we leave it.

        The removal runs as its own task and settles the guard from a `finally`, so a
        cancelled turn cannot strand the slot mid-removal. The caller merely waits for it."""
        if not lease:
            return False
        channel_id, ts = lease.get("channel_id"), lease.get("ts")
        emoji, token = lease.get("emoji"), lease.get("token")
        guard = getattr(self, "_reaction_guard", None)
        slots = (guard or {}).get((channel_id, ts))
        slot = slots.get(emoji) if slots is not None else None
        if not (isinstance(slot, dict) and slot.get("token") == token
                and slot.get(self._REMOVING) is None):
            self.log_debug(f"Reaction lease for :{emoji}: is stale — leaving it alone")
            return False
        # Publish the removal SYNCHRONOUSLY, before any await, so a concurrent remover bails
        # and a concurrent reserver can WAIT on the outcome rather than being told the emoji
        # is safely present when it is moments from disappearing.
        task = asyncio.ensure_future(
            self._run_reaction_removal(channel_id, ts, emoji, token, lease))
        slots[emoji] = {"token": token, self._REMOVING: task}
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            raise      # the task lives on and settles the slot itself
        except Exception:  # noqa: BLE001
            return False

    async def _reserve_and_react(self, channel_id: str, ts: str, emoji: str) -> dict:
        """F6 reservation for a PERMANENT reaction — one the caller never takes back.

        Settles the lease immediately: this reaction is staying, so it must not sit in the
        guard as an unevictable owned slot."""
        result, lease = await self._reserve_and_react_owned(channel_id, ts, emoji)
        self.settle_reaction_lease(lease)
        return result

    async def _reserve_and_react_owned(self, channel_id: str, ts: str, emoji: str) -> tuple:
        """F6 reservation + F38 lease. Returns (result, lease).

        Retries until the slot reaches a STABLE state, under one absolute deadline. Every
        await in `_reserve_once` — waiting on someone else's in-flight add, waiting on a
        removal — can be overtaken while we sleep: the owner of the add we waited for may
        start removing it, a removal we waited for may be followed by a fresh add and a second
        removal. So a pass that had to wait never trusts what it learned; it re-reads the slot
        and decides again.

        A fixed number of passes cannot express that (I tried two, and codex pointed out that
        two removal generations back-to-back fall straight through it and return nothing at
        all). The deadline can: we either converge on a real answer or say so honestly."""
        deadline = time.monotonic() + max(1.0, float(getattr(config, "tool_call_timeout", 20)))
        while True:
            result, lease, retry = await self._reserve_once(
                channel_id, ts, emoji, deadline)
            if not retry:
                return result, lease
            if time.monotonic() >= deadline:
                # Never fall through with a None result — the react tool subscripts it.
                return ({"ok": False, "error": "reaction_busy",
                         "message": f"Could not settle :{emoji}: — it is being changed "
                                    f"concurrently. Try again."}, None)

    async def _reserve_once(self, channel_id: str, ts: str, emoji: str,
                            deadline: Optional[float] = None) -> tuple:
        """One reservation pass. Returns (result, lease, retry).

        Guard: bounded LRU map (channel, ts) -> {emoji: Future(pending) | True(committed)
        | {"token"}(owned) | {"token","removing"}(being taken back)}, plus a parallel
        (channel, ts) -> monotonic touch time. Distinct emoji up to REACTION_MAX_PER_MESSAGE
        land; a duplicate emoji is idempotent success WITHOUT consuming a slot. Because
        dispatch_all runs sibling calls concurrently, the slot is reserved SYNCHRONOUSLY
        (before the first await) so N+1 distinct reactions can't all pass the cap; a
        failed/cancelled Slack call rolls the reservation back in `finally`. A duplicate whose
        in-flight owner FAILS must not report success (round-2 fix a). Eviction pins
        recently-touched entries and any entry holding a live claim; a duplicate's wait on an
        in-flight owner is time-bounded.

        F38 — the LEASE. Non-None only when THIS call genuinely added the reaction (not a
        duplicate, not `already_reacted`, not a wait on someone else's in-flight add). It is
        the receipt that lets `remove_owned_reaction` prove the emoji on screen is the one we
        put there, so a work-claim 👀 we take back can never strip a reaction that a
        concurrent turn — or the model's own react tool — has since made its own."""
        now = time.monotonic()
        cap = max(1, int(getattr(config, "reaction_max_per_message", 4)))
        guard = getattr(self, "_reaction_guard", None)
        if guard is None:
            guard = self._reaction_guard = OrderedDict()  # (channel, ts) -> {emoji: Future|True}
        ts_map = getattr(self, "_reaction_guard_ts", None)
        if ts_map is None:
            ts_map = self._reaction_guard_ts = {}       # (channel, ts) -> monotonic touch time
        key = (channel_id, ts)
        slots = guard.get(key)
        if slots is None:
            slots = guard[key] = {}
        guard.move_to_end(key)  # LRU recency refresh
        ts_map[key] = now
        self._trim_reaction_guard(guard, ts_map, now, keep=slots)

        # How long we may wait on someone else's in-flight operation: whatever is left of the
        # caller's overall deadline, so a slot that keeps churning can't outlast it.
        wait_bound = max(1.0, float(getattr(config, "tool_call_timeout", 20)))
        if deadline is not None:
            wait_bound = min(wait_bound, max(0.0, deadline - now))

        busy = ({"ok": False, "error": "reaction_busy",
                 "message": f"Could not settle :{emoji}: — it is being changed concurrently. "
                            f"Try again."}, None, False)

        existing = slots.get(emoji)
        if existing is not None:
            removal = existing.get(self._REMOVING) if isinstance(existing, dict) else None
            if removal is not None:
                # The emoji is being TAKEN BACK right now. Reporting "it's there" would be a
                # lie the moment the removal lands — and for the model's react tool that lie
                # becomes a reaction-only reply whose reaction does not exist. Wait for the
                # real outcome instead.
                try:
                    await asyncio.wait_for(asyncio.shield(removal), timeout=wait_bound)
                except asyncio.TimeoutError:
                    # Still running. We do NOT know how it ends, and guessing "still present"
                    # would be the same lie one step later. Say so honestly.
                    return busy
                except Exception:
                    pass
                # Resolved, one way or the other — and the task has already settled the slot
                # (popped on success, demoted to True on failure). Anything we remember about
                # it is stale, so decide again from what the guard says NOW.
                return None, None, True
            # Duplicate emoji — idempotent, no new slot consumed. No lease: we did not put
            # this one there, so it is not ours to take back.
            if self._is_committed(existing):
                return {"ok": True, "emoji": emoji, "ts": ts, "idempotent": True}, None, False
            # In-flight ADD: await the owner's real outcome (shield so our cancellation
            # doesn't cancel theirs), time-bounded so a never-resolving owner can't hang us.
            try:
                ok = await asyncio.wait_for(asyncio.shield(existing), timeout=wait_bound)
            except asyncio.TimeoutError:
                return busy
            except Exception:
                ok = False
            if not ok:
                return ({"ok": False, "error": "reaction_failed",
                         "message": f"Could not add :{emoji}:."}, None, False)
            # The add succeeded — but that was then. While we slept, its owner may already
            # have started taking it back (a work-claim turn that produced nothing). Reporting
            # success on the strength of a stale future would promise a reaction that is on
            # its way out. Re-read the slot and decide again.
            return None, None, True

        # New emoji — enforce the cap over committed + pending distinct emoji.
        if len(slots) >= cap:
            return ({"ok": False, "error": "reaction_cap",
                     "message": f"Already at the max of {cap} reactions on that message."},
                    None, False)

        # Reserve synchronously (before any await) so concurrent siblings see the slot.
        fut = asyncio.get_event_loop().create_future()
        slots[emoji] = fut
        committed = False
        try:
            ok, added = await self._react_add(channel_id, ts, emoji)
            if ok:
                committed = True
                if not fut.done():
                    fut.set_result(True)
                if not added:
                    # Slack said already_reacted: the emoji is present, but WE did not put it
                    # there this time — a previous turn did, and the guard entry proving it was
                    # evicted. Commit the slot UNOWNED and mint no lease: removing it would
                    # take back a reaction that is not ours.
                    slots[emoji] = True
                    return {"ok": True, "emoji": emoji, "ts": ts, "idempotent": True}, None, False
                # F31: a genuine new commit — record it as the bot's own reaction so it's
                # self-visible in the rings (idempotent duplicates below never reach here).
                receipt = self._record_own_reaction_pulse(channel_id, ts, emoji)
                token = uuid4().hex
                slots[emoji] = {"token": token}   # committed AND owned by this caller
                return ({"ok": True, "emoji": emoji, "ts": ts},
                        {"token": token, "channel_id": channel_id, "ts": ts,
                         "emoji": emoji, "pulse_receipt": receipt},
                        False)
            if not fut.done():
                fut.set_result(False)
            return ({"ok": False, "error": "reaction_failed",
                     "message": f"Could not add :{emoji}:."}, None, False)
        finally:
            if not committed:
                # Roll back the reservation — covers failure, timeout, and cancellation.
                if slots.get(emoji) is fut:
                    del slots[emoji]
                if not fut.done():
                    fut.set_result(False)
                # Identity-conditional cleanup: only drop the key when it STILL maps to
                # our own slots object (a concurrent recreate after eviction installs a
                # different dict, which we must not delete).
                if not slots and guard.get(key) is slots:
                    guard.pop(key, None)
                    ts_map.pop(key, None)
            # Retrim after settlement: a burst may have blown past the cap with everything
            # pending (nothing evictable then); now that this call resolved, sweep again.
            self._trim_reaction_guard(guard, ts_map, time.monotonic())

    def get_post_to_thread_tool_schema(self) -> dict:
        """F23: schema for the cross-thread reply tool. CURRENT CHANNEL ONLY — there is no
        channel_id param; cross-channel posting is out of scope (a write boundary, unlike the
        read tools that can reach other channels)."""
        return {
            "type": "function",
            "name": "post_to_thread",
            "description": (
                "Post a reply into a DIFFERENT thread in THIS channel. Use when a reply "
                "belongs somewhere other than the current conversation — someone asked you to "
                "answer a message over in another thread, or you're closing a loop you were "
                "part of elsewhere. Acknowledge briefly in the current thread rather than "
                "duplicating the whole answer in both places. Only targets threads in the "
                "current channel; there is no way to post to another channel."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "thread_ts": {
                        "type": "string",
                        "description": "Root ts of the target conversation (a top-level message's ts "
                                       "targets its thread). Must be a ts you have actually seen in "
                                       "context or from a tool — never guess one.",
                    },
                    "text": {
                        "type": "string",
                        "description": "The reply to post, in normal markdown (converted to Slack "
                                       "formatting automatically).",
                    },
                },
                "required": ["thread_ts", "text"],
            },
        }

    async def execute_post_to_thread(self, ctx, args: dict) -> dict:
        """Executor for post_to_thread (F23). Posts a markdown-converted reply into another
        thread of the CURRENT channel via the standard messaging layer (which also records the
        own-reply pulse, keeping the rings truthful). Never raises — every refusal/failure is an
        {"ok": False, ...} result. Runs inside an addressed/judged turn, so no unprompted
        accounting is added."""
        if not config.enable_post_to_thread_tool:
            return {"ok": False, "error": "disabled", "message": "Cross-thread posting is disabled."}
        channel_id = getattr(ctx, "channel_id", None)
        if not channel_id:
            return {"ok": False, "error": "no_channel", "message": "No channel to post into."}
        target = (args.get("thread_ts") or "").strip()
        text = (args.get("text") or "").strip()
        if not target:
            return {"ok": False, "error": "missing_thread_ts", "message": "A target thread_ts is required."}
        if not text:
            return {"ok": False, "error": "empty_text", "message": "Nothing to post — text was empty."}
        # Posting into the CURRENT conversation would double-post alongside the normal reply.
        current = getattr(ctx, "thread_ts", None)
        trigger = getattr(ctx, "trigger_ts", None)
        if target == current or target == trigger:
            return {"ok": False, "error": "same_thread",
                    "message": "That's the current thread — just reply normally instead."}
        try:
            posted_ts = await self.send_message(channel_id, target, text)
        except Exception as e:
            self.log_warning(f"post_to_thread: send failed for {channel_id}/{target}: {e}")
            return {"ok": False, "error": "post_failed", "message": "Could not post to that thread."}
        if not posted_ts:
            return {"ok": False, "error": "post_failed", "message": "Could not post to that thread."}
        return {"ok": True, "thread_ts": target, "posted_ts": posted_ts}

    def get_no_reply_tool_schema(self) -> dict:
        """Function-tool schema for the F2 terminal no-reply action (unprompted turns only)."""
        return {
            "type": "function",
            "name": "no_response_needed",
            "description": (
                "End this turn without posting anything. Call this when, after seeing the "
                "full conversation, you have nothing useful to add — the message wasn't "
                "really for you, someone else already answered, or silence is the socially "
                "right move. You may add an emoji reaction (react_to_message) in the same "
                "round; call this instead of replying, never after writing a reply."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string",
                               "description": "One short sentence: why silence is right."},
                },
                "required": ["reason"],
            },
        }

    async def execute_no_reply_tool(self, ctx, args: dict) -> dict:
        """Executor for no_response_needed. Terminal signal only — the tool loop stops the
        turn and the handler surfaces the outcome; nothing is posted here."""
        return {"ok": True}

    async def update_message_streaming(self, channel_id: str, message_id: str, text: str) -> Dict:
        """Updates a message with rate limit awareness"""
        try:
            # Strip MCP citations from text before sending to Slack
            # This is the single point of control for all streaming updates
            text = strip_citations(text)

            # For messages that already contain Slack mrkdwn (like enhanced prompts with _italics_),
            # skip the markdown conversion to avoid double-processing
            if text.startswith("✨") or text.startswith("*Enhanced Prompt:*") or text.startswith("Enhancing your prompt:"):
                # This is an enhanced prompt - it already has proper Slack formatting
                formatted_text = text
            else:
                # Format text for Slack using markdown conversion
                formatted_text = self.format_text(text)
            
            # More aggressive truncation for streaming to avoid msg_too_long errors
            # Account for Slack's markdown expansion and special characters
            safe_length = self.MAX_MESSAGE_LENGTH - 200  # More buffer for safety
            if len(formatted_text) > safe_length:
                # Try to truncate at a reasonable boundary (code block or paragraph)
                truncated = formatted_text[:safe_length]
                
                # If we're in the middle of a code block, close it
                if truncated.count('```') % 2 == 1:
                    truncated += '\n```'
                
                formatted_text = truncated + continuation_trailer()
            
            # Call Slack API's chat_update method
            result = await self.app.client.chat_update(
                channel=channel_id,
                ts=message_id,
                text=formatted_text,
                mrkdwn=True  # Enable markdown parsing for italics/bold
            )
            
            # Return success status
            return {
                "success": True,
                "rate_limited": False,
                "retry_after": None,
                "result": result
            }
            
        except SlackApiError as e:
            # Handle msg_too_long error specifically
            if e.response.get('error') == 'msg_too_long':
                self.log_warning("Message too long for Slack, truncating more aggressively")
                # Try with much shorter message
                very_short = formatted_text[:2000] + "\n\n*...continued in next message...*"
                if very_short.count('```') % 2 == 1:
                    very_short += '\n```'
                
                try:
                    result = await self.app.client.chat_update(
                        channel=channel_id,
                        ts=message_id,
                        text=very_short,
                        mrkdwn=True
                    )
                    return {
                        "success": True,
                        "rate_limited": False,
                        "retry_after": None,
                        "result": result
                    }
                except Exception:
                    # If even the short version fails, just acknowledge the error
                    self.log_error("Even truncated message failed to send")
                    raise
            
            # Handle 429 rate limit responses
            elif e.response.status_code == 429:
                # Extract retry-after header
                retry_after = None
                if hasattr(e.response, 'headers') and 'Retry-After' in e.response.headers:
                    try:
                        retry_after = int(e.response.headers['Retry-After'])
                    except (ValueError, KeyError):
                        retry_after = None
                
                self.log_warning("🚨🚨🚨 HIT RATE LIMIT 429 🚨🚨🚨")
                
                return {
                    "success": False,
                    "rate_limited": True,
                    "retry_after": retry_after,
                    "error": str(e)
                }
            else:
                # Handle other API errors
                self.log_error(f"Error updating message in streaming: {e}")
                return {
                    "success": False,
                    "rate_limited": False,
                    "retry_after": None,
                    "error": str(e)
                }
        except Exception as e:
            # Handle unexpected errors
            self.log_error(f"Unexpected error updating message in streaming: {e}")
            return {
                "success": False,
                "rate_limited": False,
                "retry_after": None,
                "error": str(e)
            }

    def _build_response_footer_blocks(self, model: str) -> list:
        """Footer: a single compact row — one small button carrying the model name that opens
        the per-channel settings modal (handled by the ``open_channel_settings`` action)."""
        model_label = model or config.gpt_model
        return [
            {"type": "actions", "elements": [
                {"type": "button",
                 "text": {"type": "plain_text", "text": f"⚙️ {model_label}"},
                 "action_id": "open_channel_settings"}
            ]},
        ]

    def attachable_footer_blocks(self, channel_id: Optional[str], model: Optional[str] = None):
        """Settings chrome to ATTACH to the final part of a native-streamed response
        (chat.stopStream accepts blocks), so the "⚙️ <model>" row rides the response
        message itself instead of a separate trailing post — on EVERY surface
        (user request 2026-07-10; matches Claude's per-message footer row).

        Surface routing: channels/channel threads get the per-channel settings button
        (open_channel_settings); DMs/assistant threads get the personal settings
        button (open_user_settings) since there are no channel settings there. This
        is independent of the feedback strip (ENABLE_FEEDBACK_BUTTONS), which stays
        off by the operator's choice.

        Returns None when ENABLE_RESPONSE_FOOTER is off. Fallback for paths that
        can't attach: channels may still post the separate footer message
        (maybe_post_response_footer); DMs simply get no gear — /chatgpt-settings
        always works."""
        if not channel_id:
            return None
        if not getattr(config, "enable_response_footer", True):
            return None
        if channel_id.startswith("D"):
            model_label = model or config.gpt_model
            return [
                {"type": "actions", "elements": [
                    {"type": "button",
                     "text": {"type": "plain_text", "text": f"⚙️ {model_label}"},
                     "action_id": USER_SETTINGS_ACTION_ID}
                ]},
            ]
        return self._build_response_footer_blocks(model)

    async def maybe_post_response_footer(self, message, response) -> None:
        """Trailing chrome under a final text response — surface-dependent:

        - Channels: the Phase 7 footer (model + Configure button), when
          ENABLE_RESPONSE_FOOTER is on.
        - DMs/assistant threads: the Phase H native feedback buttons strip, when
          ENABLE_FEEDBACK_BUTTONS is on (channels deliberately get no feedback strip —
          pixels matter there; reactions cover feedback in channels).

        Posted as a SEPARATE trailing message, so it never touches the text/split/streaming
        path and is inherently attached only after the final part (streamed included).
        Fires once, only for text responses. Never raises.
        """
        try:
            if not response or getattr(response, "type", None) != "text":
                return
            # Reaction-only turns post no message — nothing to hang chrome under
            if not (getattr(response, "content", None) or "").strip():
                return
            # The chrome already rode the response message itself (native streaming
            # attaches it on stopStream) — don't double up with a separate post.
            if (getattr(response, "metadata", None) or {}).get("footer_attached"):
                return
            channel_id = getattr(message, "channel_id", None)
            if not channel_id:
                return
            if channel_id.startswith("D"):
                # Phase H: feedback buttons + "⚙️ <model>" (user settings) on the
                # assistant/DM surface.
                if not feedback_enabled():
                    return
                # The whole strip (feedback thumbs + "⚙️ <model>" settings button)
                # posts ONCE, under the first reply of a thread — later replies get
                # no trailing chrome at all (user feedback 2026-07-09: per-message
                # buttons are bulky; a hyperlink can't open a modal — no trigger_id).
                thread_ts = getattr(message, "thread_id", None)
                if not should_offer_feedback(channel_id, thread_ts):
                    return
                model = (getattr(response, "metadata", None) or {}).get("model")
                await self.app.client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text="Rate this response",  # fallback text for notifications
                    blocks=build_feedback_blocks(model),
                )
                return
            # Channels: per-channel settings footer.
            if not getattr(config, "enable_response_footer", True):
                return
            model = (getattr(response, "metadata", None) or {}).get("model")
            blocks = self._build_response_footer_blocks(model)
            await self.app.client.chat_postMessage(
                channel=channel_id,
                thread_ts=getattr(message, "thread_id", None),
                # Describes the footer's purpose instead of showing a bare model name (which reads
                # as a spurious standalone message — the "gpt-5.6-sol" post seen live 2026-07-16).
                text=RESPONSE_FOOTER_FALLBACK_TEXT,
                blocks=blocks,
            )
        except Exception as e:
            self.log_debug(f"Could not post response footer: {e}")

    async def handle_response(self, channel_id: str, thread_id: str, response: Response):
        """Handle a Response object and send to Slack"""
        if response.type == "text":
            await self.send_message(channel_id, thread_id, response.content)
        elif response.type == "image":
            # response.content should be ImageData
            image_data = response.content
            file_url = await self.send_image(
                channel_id,
                thread_id,
                image_data.to_bytes(),
                f"generated_image.{image_data.format}",
                ""  # No caption - prompt already displayed via streaming
            )
            
            # Store the URL in the image data for tracking
            if file_url:
                image_data.slack_url = file_url
                
        elif response.type == "reaction":
            # Phase 4: respond with emoji reaction(s) instead of (or before) text.
            # content is an emoji name or list; metadata.react_ts is the target message
            # (falls back to the thread root if not provided).
            target_ts = (response.metadata or {}).get("react_ts") or thread_id
            emojis = response.content if isinstance(response.content, list) else [response.content]
            for emoji in emojis:
                await self.react(channel_id, target_ts, emoji)
        elif response.type == "error":
            formatted_error = self.format_error_message(response.content)
            await self.send_message(channel_id, thread_id, formatted_error)
