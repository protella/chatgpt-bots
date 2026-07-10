from __future__ import annotations

import asyncio
import random
from typing import Dict, List, Optional

from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_sdk.errors import SlackApiError

from base_client import HistoryFetchError, Message, Response
from config import SUPPORTED_CHAT_MODELS, config, pipeline_status_markers
from message_markers import (
    CONTINUATION_HEAD,
    continuation_trailer,
    fence_safe_chunks,
)
from slack_client.event_handlers.feedback import (
    FEEDBACK_ACTION_ID,
    USER_SETTINGS_ACTION_ID,
    build_feedback_blocks,
    feedback_enabled,
    should_offer_feedback,
)
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
_UI_HELPER_FALLBACK_TEXTS = frozenset(
    {"Rate this response", "Settings available"} | set(SUPPORTED_CHAT_MODELS)
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

    def __init__(self, client, channel_id: str, thread_id: str, logger=None):
        self._client = client
        self._channel = channel_id
        self._thread = thread_id
        self._log = logger
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
        try:
            resp = await self._client.chat_startStream(
                channel=self._channel,
                thread_ts=self._thread,
                markdown_text=initial_text or None,
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


class SlackMessagingMixin:
    async def start(self):
        """Start the Slack bot"""
        self.handler = AsyncSocketModeHandler(self.app, config.slack_app_token)
        self.log_info("Starting Slack bot in socket mode...")

        # Resolve our own identity up front so we can tell our messages apart from other bots'
        await self._ensure_self_identity()

        # Create a task for start_async that can be cancelled
        self._start_task = asyncio.create_task(self.handler.start_async())

        try:
            await self._start_task
        except asyncio.CancelledError:
            self.log_info("Slack bot start task cancelled")
            raise
        except Exception as e:
            self.log_error(f"Error in Slack bot start: {e}")
            raise

    async def stop(self):
        """Stop the Slack bot"""
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

    async def send_message(self, channel_id: str, thread_id: str, text: str) -> bool:
        """Send a text message to Slack, splitting if needed"""
        try:
            # Strip MCP citations from text before sending to Slack
            text = strip_citations(text)
            # Format text for Slack
            formatted_text = self.format_text(text)
            
            # Check if we need to split the message
            if len(formatted_text) <= self.MAX_MESSAGE_LENGTH:
                # Single message
                await self.app.client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_id,
                    text=formatted_text
                )
            else:
                # Split into multiple messages, "Continued..." style (shared markers so
                # the rebuild-side stripper always recognizes them). One chunk failing
                # must not abort the rest.
                chunks = self._split_message(formatted_text)
                last = len(chunks) - 1
                sent_any = False
                for i, chunk in enumerate(chunks):
                    body = chunk
                    if i > 0:
                        body = f"{CONTINUATION_HEAD}\n\n{body}"
                    if i < last:
                        body = f"{body}{continuation_trailer()}"
                    try:
                        await self.app.client.chat_postMessage(
                            channel=channel_id,
                            thread_ts=thread_id,
                            text=body
                        )
                        sent_any = True
                    except SlackApiError as chunk_error:
                        self.log_error(f"Error sending message chunk {i + 1}/{last + 1}: {chunk_error}")
                return sent_any if last > 0 else True
            return True
        except SlackApiError as e:
            self.log_error(f"Error sending message: {e}")
            return False

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

    async def send_image(self, channel_id: str, thread_id: str, image_data: bytes, filename: str, caption: str = "") -> Optional[str]:
        """Send an image to Slack and return the file URL"""
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
                self.log_info(f"Image uploaded: {filename} - URL: {file_url}")
                return file_url
            else:
                self.log_warning("Image uploaded but no URL found in response")
                return None
                
        except SlackApiError as e:
            self.log_error(f"Error uploading image: {e}")
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
                text=f"{config.thinking_emoji} Thinking..."
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
                    if sender_type == "self":
                        # Our transient placeholders/status lines: ":emoji: Thinking..."
                        # and status updates that share that shape. Precise-match only.
                        if _SELF_STATUS_RE.match(text) and (
                            any(marker in text for marker in _SELF_STATUS_MARKERS)
                            or any(marker in text for marker in pipeline_status_markers())
                        ):
                            continue
                        # Legacy busy/processing notices
                        if ":warning:" in text and "currently processing" in text:
                            continue
                        # Settings button messages
                        if text == "Settings available":
                            continue
                    # Skip our UI-helper messages (channel footer's Configure button,
                    # Phase H feedback strip) — detected by their block action_ids,
                    # so a real reply can't false-positive.
                    if _is_ui_helper_message(msg):
                        continue

                    # sender_type computed above (drives both the self-only filters and metadata)
                    is_bot = bool(msg.get("bot_id"))  # kept for existing readers ("any bot")
                    # Display name for bot authors (used to name-prefix other bots like humans)
                    bot_name = msg.get("username") or (msg.get("bot_profile") or {}).get("name")

                    # Clean text
                    text = msg.get("text", "")
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
                            "url": file.get("url_private", file.get("permalink"))
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

    def begin_native_stream(self, channel_id: str, thread_id: str) -> "NativeStreamSession":
        """Create a (not-yet-started) NativeStreamSession bound to this channel/thread."""
        return NativeStreamSession(self.app.client, channel_id, thread_id, logger=self.log_debug)

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

    async def react(self, channel_id: str, message_ts: str, emoji: str) -> bool:
        """Add an emoji reaction to a message (Phase 4). ``emoji`` may include or omit colons.

        Best-effort: treats already_reacted as success, never raises.
        """
        if not config.enable_reactions:
            return False
        name = (emoji or "").strip().strip(":")
        if not name:
            return False
        try:
            await self.app.client.reactions_add(channel=channel_id, name=name, timestamp=message_ts)
            return True
        except SlackApiError as e:
            err = e.response.get("error") if getattr(e, "response", None) else str(e)
            if err == "already_reacted":
                return True  # idempotent: the reaction is already present
            self.log_warning(f"Could not add reaction :{name}: ({err})")
            return False
        except Exception as e:  # noqa: BLE001
            self.log_error(f"Unexpected error adding reaction :{name}: {e}")
            return False

    # --- react_to_message local tool (redesign Phase D) ---

    def get_react_tool_schema(self) -> dict:
        """Function-tool schema for model-invoked reactions. The emoji enum is the
        REACTION_EMOJIS allowlist, so the model can only pick vetted (on-brand) emoji."""
        allowed = [e.strip().strip(":") for e in (config.reaction_emojis or []) if e and e.strip().strip(":")]
        return {
            "type": "function",
            "name": "react_to_message",
            "description": (
                "Add an emoji reaction to a Slack message, like a human colleague would — "
                "sparingly and tastefully (an acknowledgment, a ✅ on a completed request, a "
                "celebration). If a reaction alone fully answers the message, react and reply "
                "with empty text. Defaults to the message you are answering."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "emoji": {"type": "string", "enum": allowed,
                              "description": "Reaction emoji name (no colons)."},
                    "ts": {"type": "string",
                           "description": "Optional ts of another recent message in this channel to react to."},
                },
                "required": ["emoji"],
            },
        }

    async def execute_react_tool(self, ctx, args: dict) -> dict:
        """Executor for react_to_message. Allowlist + one-bot-reaction-per-message dedup;
        never raises (returns {"ok": False, ...} on any refusal/failure)."""
        if not (config.enable_reactions and config.enable_react_tool):
            return {"ok": False, "error": "disabled", "message": "Reactions are disabled."}
        emoji = (args.get("emoji") or "").strip().strip(":")
        allowed = {e.strip().strip(":") for e in (config.reaction_emojis or [])}
        if not emoji or emoji not in allowed:
            return {"ok": False, "error": "emoji_not_allowed", "allowed": sorted(allowed)}
        channel_id = getattr(ctx, "channel_id", None)
        ts = (args.get("ts") or "").strip() or getattr(ctx, "trigger_ts", None) or getattr(ctx, "thread_ts", None)
        if not channel_id or not ts:
            return {"ok": False, "error": "no_target", "message": "No message to react to."}

        # At most one bot reaction per message (bounded in-memory guard)
        reacted = getattr(self, "_tool_reacted_ts", None)
        if reacted is None:
            reacted = self._tool_reacted_ts = set()
        key = f"{channel_id}:{ts}"
        if key in reacted:
            return {"ok": False, "error": "already_reacted",
                    "message": "Already reacted to that message — one reaction per message."}

        ok = await self.react(channel_id, ts, emoji)
        if ok:
            reacted.add(key)
            if len(reacted) > 5000:
                reacted.clear()  # crude bound; dedup is best-effort, Slack dedups same-emoji anyway
            return {"ok": True, "emoji": emoji, "ts": ts}
        return {"ok": False, "error": "reaction_failed", "message": f"Could not add :{emoji}:."}

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
                text=(model or config.gpt_model),  # fallback text for notifications
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
