#!/usr/bin/env python3
"""
Multi-Platform Chat Bot V2 - Main Entry Point
Supports multiple chat platforms with shared AI capabilities
"""
import sys
import signal
import asyncio
import argparse
from typing import Optional
from config import config
from logger import log_session_start, log_session_end, main_logger
from message_processor.base import MessageProcessor
from message_processor.participation import (ParticipationEngine,
                                             render_capabilities_line)
from base_client import BaseClient, Message


class ChatBotV2:
    """Main application class for multi-platform chat bot"""
    
    def __init__(self, platform: str = "slack"):
        self.platform = platform.lower()
        self.client: Optional[BaseClient] = None
        self.processor = None  # Will be initialized after client
        self.participation_engine = None  # Phase F; set in initialize()
        self.cleanup_task = None
        self.running = False
        self.sigint_count = 0  # Track number of SIGINT received
        self.last_sigint_time = 0  # Track time of last SIGINT
        
    async def initialize(self):
        """Initialize the bot components"""
        main_logger.info(f"Initializing Chat Bot V2 for {self.platform}...")
        
        # Validate configuration
        try:
            config.validate()
        except ValueError as e:
            main_logger.error(f"Configuration error: {e}")
            sys.exit(1)
        
        # Initialize platform-specific client
        if self.platform == "slack":
            from slack_client import SlackBot
            self.client = SlackBot(message_handler=self.handle_message)
            # Initialize processor with database from client
            self.processor = MessageProcessor(db=self.client.db)
            # Give the client a reference to the processor for thread state updates
            self.client.processor = self.processor
            # Phase F: judgment layer for unprompted channel participation.
            self.participation_engine = ParticipationEngine(self.processor.openai_client)
        else:
            main_logger.error(f"Unknown platform: {self.platform}")
            sys.exit(1)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        main_logger.info("Initialization complete")
    
    @staticmethod
    def _is_unprompted_turn(message: Message) -> bool:
        """F14: whether a posted channel reply counts as UNPROMPTED for pulse pacing.

        A participation-gated turn is unprompted UNLESS it was woken by a name-hit — being
        called by name is prompted in spirit (like an @-mention), so its reply must not
        burn the runaway-brake budget."""
        md = message.metadata or {}
        return (md.get("participation_check") is True
                and md.get("participation_name_hit") is not True)

    async def _run_participation_gate(self, message: Message, client: BaseClient):
        """Phase F gate for UNPROMPTED channel messages: hard rails → debounce →
        ONE engine call → act. Returns a verdict only for action='respond';
        every other outcome (ignore / react / backoff / superseded / any failure)
        is handled here and returns None so the caller stays silent."""
        engine = self.participation_engine
        if engine is None or not getattr(config, "enable_participation_engine", True):
            return None  # engine off → unaddressed messages stay unanswered (mentions_only behavior)
        try:
            channel_id = message.channel_id
            ts = message.metadata.get("ts") or message.thread_id
            level = message.metadata.get("participation_level") or "judicious"
            pulse = getattr(client, "channel_pulse", None)

            # F5 fix (b): register this message's ts as the channel's newest BEFORE the
            # memory/topic awaits below — an older event delayed by that I/O must not
            # overwrite a newer event's debounce marker and win the race.
            engine.note_arrival(channel_id, ts)

            # F17: no hourly-cap hard rail — pacing is the classifier's judgment. The
            # unprompted-reply count is still tallied and fed to the engine as a signal
            # (below), but a high count never silences a turn before the model sees it.
            name_hit = message.metadata.get("participation_name_hit") is True

            channel_activity = None
            unprompted = 0
            if pulse is not None:
                channel_activity = pulse.render_envelope(
                    channel_id, exclude_thread_ts=None,
                    max_lines=config.channel_pulse_envelope_max,
                ) or None
                unprompted = pulse.unprompted_count_last_hour(channel_id)

            memory_facts = []
            try:
                if getattr(config, "enable_channel_memory", True) and self.processor.db:
                    memory_facts = await self.processor.db.get_channel_memory_async(channel_id)
            except Exception:
                memory_facts = []

            channel_topic = None
            fetch_ctx = getattr(client, "get_channel_context", None)
            if fetch_ctx:
                try:
                    ctx = await fetch_ctx(channel_id)
                    channel_topic = (ctx or {}).get("topic") or None
                except Exception:
                    channel_topic = None

            is_thread_reply = bool(ts and message.thread_id and message.thread_id != ts)
            # F11: inventory of the assistant's own tools/data sources so the classifier
            # can weigh whether it is well-suited to answer an open question to the room.
            capabilities = render_capabilities_line(getattr(self.processor, "mcp_manager", None))
            verdict = await engine.evaluate(
                channel_id=channel_id, ts=ts, text=message.text,
                sender_name=message.metadata.get("user_real_name") or message.metadata.get("username"),
                is_thread_reply=is_thread_reply, level=level,
                directives=message.metadata.get("channel_directives"),
                memory_facts=memory_facts, channel_activity=channel_activity,
                unprompted_last_hour=unprompted,
                name_hit=name_hit,
                sender_is_bot=message.metadata.get("participation_sender_bot") is True,
                channel_topic=channel_topic,
                capabilities=capabilities,
                pulse=pulse, thread_root_ts=message.thread_id,
            )
            if verdict is None:  # superseded by a newer message during debounce
                main_logger.debug("Participation gate: superseded during debounce — silent")
                return None
            main_logger.debug(f"Participation verdict: {verdict.action} ({verdict.reason})")

            if verdict.action == "react":
                react_ts = message.metadata.get("ts") or message.thread_id
                # F6: route the gate's own reaction through the reservation guard so a
                # later main-model turn on this message honestly sees the slot consumed
                # (and won't double-add the same emoji). Falls back to the raw react.
                try:
                    # Bound the gate's own react by the configured tool-call timeout so a
                    # wedged Slack call can't stall the turn (the model-invoked react tool
                    # is already timeout-guarded by the tool loop; this direct path wasn't).
                    if hasattr(client, "_reserve_and_react"):
                        await asyncio.wait_for(
                            client._reserve_and_react(channel_id, react_ts, verdict.emoji),
                            timeout=config.tool_call_timeout)
                    elif hasattr(client, "react"):
                        await asyncio.wait_for(
                            client.react(channel_id, react_ts, verdict.emoji),
                            timeout=config.tool_call_timeout)
                except asyncio.TimeoutError:
                    main_logger.debug("Participation react timed out")
                except Exception as e:
                    main_logger.debug(f"Participation react failed: {e}")
                return None
            if verdict.action == "backoff":
                await self._apply_backoff(message, client)
                return None
            if verdict.action == "respond":
                return verdict
            return None  # ignore
        except Exception as e:
            # Fail-safe stays silence: worst failure mode is a missed reply, never spam.
            main_logger.warning(f"Participation gate error: {e}; staying silent")
            return None

    async def _apply_backoff(self, message: Message, client: BaseClient):
        """'Butt out' loop (F15): ack with an emoji (no more words), permanently MUTE
        THIS THREAD for unprompted participation, and write/update a durable channel-memory
        fact so the classifier raises the bar channel-wide. No timer — nothing expires; the
        model forgets/updates the fact (and the mute lifts) when re-invited. @-mentions and
        name-hit summons in the muted thread still answer."""
        channel_id = message.channel_id
        react_ts = message.metadata.get("ts") or message.thread_id
        thread_root = message.thread_id or react_ts
        if hasattr(client, "react"):
            try:
                await client.react(channel_id, react_ts, config.snooze_ack_emoji)
            except Exception as e:
                main_logger.debug(f"Backoff ack react failed: {e}")
        try:
            newly = await self.processor.db.add_muted_thread_async(
                channel_id, thread_root, updated_by="participation_engine")
            main_logger.info(
                f"Participation backoff: muted thread {channel_id}:{thread_root} "
                f"(newly={newly})")
        except Exception as e:
            main_logger.warning(f"Backoff thread-mute write failed: {e}")
        try:
            if getattr(config, "enable_channel_memory", True) and self.processor.db:
                await self._record_backoff_memory(message, react_ts)
        except Exception as e:
            main_logger.debug(f"Backoff memory write failed: {e}")

    async def _record_backoff_memory(self, message: Message, react_ts: str):
        """Write or UPDATE (never duplicate) the channel-memory fact recording a butt-out.

        One fact per thread — keyed by author=`participation_engine:<thread_root>` so a repeat
        backoff on the same thread refreshes that row's date/text instead of piling up
        duplicates, while butt-outs in DIFFERENT threads accrue as distinct facts (the
        classifier reads REPEATED facts as observe-only)."""
        import datetime
        channel_id = message.channel_id
        thread_root = message.thread_id or react_ts
        try:
            day = datetime.datetime.fromtimestamp(
                float(react_ts), tz=datetime.timezone.utc).date().isoformat()
        except (ValueError, TypeError, OSError):
            day = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        who = (message.metadata.get("user_real_name")
               or message.metadata.get("username") or "a teammate")
        topic = self._backoff_topic(message.text)
        content = (f"{day}: {who} told the assistant to butt out of {topic} — "
                   f"raise the bar for unprompted replies in this channel.")
        marker = f"participation_engine:{thread_root}"
        existing = await self.processor.db.get_channel_memory_async(channel_id)
        prior = next((f for f in (existing or []) if f.get("author") == marker), None)
        if prior:
            await self.processor.db.update_channel_memory_async(prior["id"], content)
        else:
            await self.processor.db.add_channel_memory_async(
                channel_id, content, author=marker)

    @staticmethod
    def _backoff_topic(text: Optional[str]) -> str:
        """A short, single-line topic phrase for the butt-out memory fact, derived from the
        triggering message. Collapses whitespace and truncates so the stored fact stays a
        readable one-liner."""
        cleaned = " ".join((text or "").split())
        if not cleaned:
            return "this conversation"
        return (cleaned[:80] + "…") if len(cleaned) > 80 else cleaned

    async def handle_message(self, message: Message, client: BaseClient):
        """Handle incoming message from any platform"""
        # Phase F participation gate: for UNPROMPTED channel messages (judicious/active
        # levels) the engine decides respond/react/ignore/backoff BEFORE anything is
        # posted. Only action='respond' falls through.
        placement_verdict = None
        if message.metadata.get("participation_check") is True:
            verdict = await self._run_participation_gate(message, client)
            if verdict is None:
                return
            placement_verdict = verdict.placement
            # F3: the engine's reason rides the wake envelope for ambient wakes.
            if isinstance(message.metadata, dict) and getattr(verdict, "reason", None):
                message.metadata["participation_reason"] = verdict.reason

        # Phase F placement (plan §4a, revised 2026-07-10): the channel's
        # reply_in_channel setting is an ALLOWANCE, not a mandate — when it's ON and
        # the trigger was top-level, the engine's per-message placement verdict
        # decides ("channel" = quick top-level answer, "thread" = worth a thread).
        # Mentions/name-wakes carry no verdict (no engine call) and reply top-level:
        # the user summoned the bot at channel level. Setting OFF = everything
        # threads. Images always thread (enforced in the image branch, which keys
        # off message.thread_id regardless).
        is_top_level_trigger = message.metadata.get("ts") == message.thread_id
        place_in_channel = (
            bool(message.metadata.get("reply_in_channel")) and is_top_level_trigger
            and bool(message.channel_id) and not message.channel_id.startswith("D")
            and placement_verdict != "thread"
        )
        if placement_verdict:
            main_logger.debug(
                f"Placement: verdict={placement_verdict}, reply_in_channel_setting="
                f"{bool(message.metadata.get('reply_in_channel'))} → "
                f"{'channel' if place_in_channel else 'thread'}"
            )
        post_thread_id = None if place_in_channel else message.thread_id
        # Handlers key presentation chrome off this (e.g. the Used Tools attribution
        # line is suppressed on top-level channel replies).
        if isinstance(message.metadata, dict):
            message.metadata["place_in_channel"] = place_in_channel

        # Phase Q: if this conversation is mid-processing, the message is about to be
        # queued (not answered now) — skip the thinking indicator so nothing flashes.
        # Advisory peek only: losing the race just means a briefly-posted indicator
        # that the queued short-circuit below deletes.
        # `is True` (not truthiness): same hardening as the wake gate — mocked or
        # malformed managers must never silently suppress the indicator.
        thread_manager = getattr(self.processor, "thread_manager", None)
        already_processing = (
            thread_manager is not None
            and hasattr(thread_manager, "is_thread_processing")
            and thread_manager.is_thread_processing(message.thread_id, message.channel_id) is True
        )

        # Send initial thinking indicator (streamed replies grow inside this message,
        # so placement is decided here).
        thinking_id = None
        if not already_processing:
            thinking_id = await client.send_thinking_indicator(
                message.channel_id,
                post_thread_id
            )
            # Batched catch-up turn (drained queue): make the status say so.
            batch_size = message.metadata.get("queued_batch_size", 0)
            if isinstance(batch_size, int) and batch_size > 1:
                catch_up = f"Catching up on {batch_size} messages..."
                try:
                    if thinking_id and hasattr(client, "update_message"):
                        await client.update_message(
                            message.channel_id, thinking_id,
                            f"{config.circle_loader_emoji} {catch_up}"
                        )
                    elif thinking_id is None and hasattr(client, "set_assistant_status"):
                        # Status-only DM indicator: the composer status carries it.
                        await client.set_assistant_status(
                            message.channel_id, post_thread_id, status=catch_up
                        )
                except Exception as e:
                    main_logger.debug(f"Catch-up status update failed: {e}")

        response = None
        try:
            # Process the message and get intent
            response = await self.processor.process_message(message, client, thinking_id)
            
            # Delete thinking indicator (but not if streaming was used - it's already the
            # response, not for background image gen — the job owns that message/status —
            # and not when a ProgressChecklist owns the thinking message, F4).
            if (thinking_id and response
                    and not response.metadata.get("streamed")
                    and response.type != "background"
                    and response.metadata.get("checklist") is None):
                await client.delete_message(message.channel_id, thinking_id)
            elif thinking_id and not response:
                await client.delete_message(message.channel_id, thinking_id)

            # Handle the response
            if response:
                if response.type == "queued":
                    # Phase Q: the message joined its conversation's pending queue and
                    # will be answered by the in-flight turn's batched catch-up. Nothing
                    # to post (the indicator, if any, was already deleted above).
                    main_logger.debug(f"Message queued behind in-flight turn for {message.channel_id}:{message.thread_id}")
                elif response.type == "text":
                    # Reaction-only turns (react tool, empty text) post no message at all
                    if not (response.content or "").strip():
                        main_logger.debug("Empty text response (reaction-only) — nothing to post")
                    # If streaming was used, the message is already displayed
                    elif not response.metadata.get("streamed"):
                        # Format and send text (top-level when placement chose channel)
                        formatted_text = client.format_text(response.content)
                        # F8: attach the settings-footer chrome to the message itself (same
                        # as the native-streaming path's stopStream blocks) instead of a
                        # separate trailing post. Suppressed for top-level channel placement
                        # (same rule as the separate footer below) and when block-building is
                        # unavailable — those fall back to maybe_post_response_footer.
                        footer_blocks = None
                        if not place_in_channel and hasattr(client, "attachable_footer_blocks"):
                            try:
                                footer_blocks = client.attachable_footer_blocks(
                                    message.channel_id, response.metadata.get("model"))
                            except Exception as e:
                                main_logger.debug(f"Footer block build failed: {e}")
                                footer_blocks = None
                        send_meta = {}
                        sent_ts = await client.send_message(
                            message.channel_id,
                            post_thread_id,
                            formatted_text,
                            blocks=footer_blocks,
                            meta_out=send_meta,
                        )
                        # Honest accounting: the ACTUAL send result decides `posted` (a
                        # failed send must not burn the hourly unprompted quota).
                        if isinstance(response.metadata, dict):
                            response.metadata["posted"] = bool(sent_ts)
                            # Only stand the separate footer down when the chrome ACTUALLY
                            # rode the message (a split/too-long reply doesn't attach it, so
                            # the separate footer post must still happen).
                            if sent_ts and send_meta.get("footer_attached"):
                                response.metadata["footer_attached"] = True
                        # F7: persist tool-use provenance keyed on the reply's real ts.
                        if sent_ts:
                            self.processor._persist_tool_provenance(
                                message.channel_id, sent_ts,
                                f"{message.channel_id}:{message.thread_id}",
                                (response.metadata or {}).get("tool_provenance"))
                    # Phase 7: Configure footer under the response (channels only, any
                    # member can open settings). Native-streamed responses attach the
                    # chrome to the message itself on stopStream (footer_attached
                    # metadata makes this call a no-op); everything else falls back to
                    # this separate trailing message.
                    # Best-effort: a cosmetic footer must never break message handling.
                    # Skipped for top-level placement — it would land as ANOTHER top-level
                    # message and read as spam.
                    # No footer under an empty turn (F2 no_reply / reaction-only) — there is
                    # no message for it to sit under.
                    # Also skip when the reply didn't actually post (posted is explicitly
                    # False) — a footer under a message that never landed reads as orphaned.
                    if (hasattr(client, "maybe_post_response_footer") and not place_in_channel
                            and (response.content or "").strip()
                            and (response.metadata or {}).get("posted") is not False):
                        try:
                            await client.maybe_post_response_footer(message, response)
                        except Exception as e:
                            main_logger.debug(f"Response footer skipped: {e}")
                elif response.type == "image":
                    # Latch: the upload + DB row land after the thread lock released, so a
                    # fast follow-up "edit it" must wait for this to finish (or time out).
                    upload_thread_key = f"{message.channel_id}:{message.thread_id}"
                    upload_manager = getattr(self.processor, "thread_manager", None)
                    if upload_manager and hasattr(upload_manager, "mark_upload_started"):
                        upload_manager.mark_upload_started(upload_thread_key)

                    from message_processor.image_delivery import publish_image
                    unprompted = self._is_unprompted_turn(message)
                    # F4: if the handler threaded a live ProgressChecklist, it owns the
                    # status surface — publish_image does the "Uploading…" step + completes
                    # it in place (keeping the accumulated steps + history marker). Only the
                    # config-off (no-checklist) path gets the manual "Uploading…" message.
                    checklist = response.metadata.get("checklist")
                    upload_status_id = None
                    if checklist is None:
                        platform_name = client.name.replace("Bot", "") if hasattr(client, 'name') else "system"
                        upload_status = f"{config.circle_loader_emoji} Uploading image to {platform_name}..."
                        if response.metadata.get("streamed"):
                            status_msg_id = response.metadata.get("status_message_id")
                            if status_msg_id and hasattr(client, 'update_message'):
                                await client.update_message(message.channel_id, status_msg_id, upload_status)
                                upload_status_id = status_msg_id
                            else:
                                upload_status_id = await client.send_thinking_indicator(message.channel_id, message.thread_id)
                                if upload_status_id and hasattr(client, 'update_message'):
                                    await client.update_message(message.channel_id, upload_status_id, upload_status)
                        else:
                            upload_status_id = await client.send_thinking_indicator(message.channel_id, message.thread_id)
                            if upload_status_id and hasattr(client, 'update_message'):
                                await client.update_message(message.channel_id, upload_status_id, upload_status)

                    try:
                        # Delivery seam owns upload, falsey-URL = failure, breadcrumb-
                        # independent DB persistence + warm state, ledger, checklist
                        # completion, and unprompted accounting. generation_id is None (sync).
                        file_url = await publish_image(
                            processor=self.processor,
                            client=client,
                            channel_id=message.channel_id,
                            thread_id=message.thread_id,
                            thread_key=upload_thread_key,
                            image_data=response.content,
                            checklist=checklist,
                            generation_id=None,
                            prompt=(response.metadata.get("prompt") or ""),
                            db=getattr(self.processor, "db", None),
                            thread_manager=self.processor.thread_manager,
                            unprompted=unprompted,
                            message_ts=(message.metadata or {}).get("ts"),
                            image_type=response.metadata.get("image_type", "generated"),
                        )
                        if file_url is None:
                            await client.handle_error(
                                message.channel_id, message.thread_id,
                                "⚠️ I created the image but couldn't post it. Please try again.")
                    finally:
                        # Always release the latch — a wedged latch would stall the next
                        # edit for the full wait timeout (publish_image no longer releases it).
                        if upload_manager and hasattr(upload_manager, "mark_upload_finished"):
                            upload_manager.mark_upload_finished(upload_thread_key)

                    # Manual status cleanup (the checklist path deletes its own message via
                    # complete(delete_after=4)).
                    if upload_status_id:
                        await asyncio.sleep(4)
                        await client.delete_message(message.channel_id, upload_status_id)
                elif response.type == "background":
                    # F1: new-image generation detached to a background job. Like 'queued',
                    # nothing to post here — the job posts the image and owns its progress
                    # surface. The thinking indicator was already left in place above.
                    main_logger.debug(
                        f"Image generation running in background for {message.channel_id}:{message.thread_id}")
                elif response.type == "error":
                    # Send error message
                    await client.handle_error(
                        message.channel_id,
                        message.thread_id,
                        response.content
                    )

            # Phase E/F participation stats (F2: accounted AFTER delivery, honest posted).
            # Count a reply only when visible content actually went out on an unprompted
            # (wake-gate) channel turn. Image/background turns account in publish_image.
            if (response and message.channel_id and not message.channel_id.startswith("D")
                    and getattr(client, "channel_pulse", None) is not None):
                terminal = (response.metadata or {}).get("terminal_action")
                if terminal == "no_reply":
                    main_logger.info(
                        f"no_response_needed — no reply posted "
                        f"(reason: {response.metadata.get('reason')!r})")
                else:
                    posted = response.metadata.get("posted")
                    if posted is None:
                        # Non-streaming handlers can't know; derive from the outcome.
                        posted = bool(
                            response.type == "text"
                            and (response.metadata.get("streamed")
                                 or (response.content or "").strip()))
                    if posted:
                        try:
                            client.channel_pulse.record_bot_reply(
                                message.channel_id, message.metadata.get("ts"),
                                unprompted=self._is_unprompted_turn(message),
                            )
                        except Exception as e:
                            main_logger.debug(f"participation stat record failed: {e}")
                    elif (response.type == "text"
                          and not (response.content or "").strip()
                          and not response.metadata.get("reaction_only")):
                        # Bare empty text without the terminal tool: contract violation.
                        # Fail-safe silence, no quota burn, no re-prompt this phase.
                        main_logger.warning(
                            "Empty text response without a terminal action — posting nothing")

        except Exception as e:
            main_logger.error(f"Error handling message: {e}", exc_info=True)

            # Delete thinking indicator on error — best-effort; a failed delete
            # must never swallow the user-facing notice below.
            if thinking_id:
                try:
                    await client.delete_message(message.channel_id, thinking_id)
                except Exception as delete_error:
                    main_logger.error(f"Failed to delete thinking indicator: {delete_error}")

            # Fixed, friendly notice — the raw exception stays in the logs only.
            try:
                await client.handle_error(
                    message.channel_id,
                    message.thread_id,
                    "⚠️ **Something Went Wrong**\n\n"
                    "I hit a snag finishing that response. Please try again in a moment."
                )
            except Exception as notify_error:
                main_logger.error(f"Failed to send error notice: {notify_error}")
        finally:
            # Native-streamed replies don't trip Slack's "auto-clear status on reply"
            # (it keys on chat.postMessage, not chat.stopStream), so a status-only turn
            # left the working bubble spinning forever (user report 2026-07-10).
            # Explicit best-effort clear. Skipped for queued turns — their status
            # belongs to the in-flight request that will answer them.
            # Also skipped for background image gen (background_owns_status): the job owns
            # the status-only progress surface and clears it on completion — clearing here
            # would blank it the instant the turn returns (Codex finding 8).
            if (thinking_id is None
                    and not (response is not None and response.type == "queued")
                    and not (response is not None and response.metadata.get("background_owns_status"))
                    and hasattr(client, "clear_assistant_status")):
                try:
                    await client.clear_assistant_status(message.channel_id, post_thread_id)
                except Exception as clear_error:
                    main_logger.debug(f"Assistant status clear failed: {clear_error}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals - double Ctrl-C for force exit"""
        import os
        import time

        # Handle SIGINT (Ctrl-C) with double-press for force exit
        if signum == signal.SIGINT:
            current_time = time.time()
            
            # If second Ctrl-C within 2 seconds, force exit
            if self.sigint_count > 0 and (current_time - self.last_sigint_time) < 2.0:
                main_logger.warning("Force exit requested (double Ctrl-C) - terminating immediately!")
                
                # Show active threads for debugging
                import threading
                active_threads = threading.enumerate()
                if len(active_threads) > 1:
                    main_logger.warning(f"Active threads at force exit: {len(active_threads)}")
                    for thread in active_threads:
                        if thread.name != "MainThread":
                            main_logger.warning(f"  - {thread.name} (daemon={thread.daemon})")
                
                # Force exit without cleanup
                os._exit(1)
            
            self.sigint_count += 1
            self.last_sigint_time = current_time
            
            if self.sigint_count == 1:
                main_logger.info(f"Received signal {signum}, attempting graceful shutdown...")
                main_logger.info("Press Ctrl-C again within 2 seconds to force exit")
                # Schedule shutdown on the event loop
                asyncio.create_task(self.shutdown())
            else:
                main_logger.warning("Shutdown already in progress... Press Ctrl-C again to force exit")
        else:
            # Handle other signals normally
            main_logger.info(f"Received signal {signum}, shutting down...")
            # Schedule shutdown on the event loop
            asyncio.create_task(self.shutdown())
    
    async def start_cleanup_task(self):
        """Start background task for periodic cleanup"""
        async def cleanup_worker():
            from croniter import croniter
            import datetime

            try:
                # Validate cron expression
                cron = croniter(config.cleanup_schedule, datetime.datetime.now())
                main_logger.info(f"Cleanup schedule configured: {config.cleanup_schedule} (cron format)")
                main_logger.info(f"Cleanup will remove threads older than {config.cleanup_max_age_hours} hours")
            except Exception as e:
                main_logger.error(f"Invalid cron expression '{config.cleanup_schedule}': {e}")
                main_logger.info("Falling back to daily at midnight (0 0 * * *)")
                cron = croniter("0 0 * * *", datetime.datetime.now())

            while self.running:
                try:
                    # Calculate next run time
                    next_run = cron.get_next(datetime.datetime)
                    now = datetime.datetime.now()
                    seconds_until_next = (next_run - now).total_seconds()

                    # Log when next cleanup will occur
                    if seconds_until_next > 3600:
                        main_logger.info(f"Next cleanup scheduled for {next_run.strftime('%Y-%m-%d %H:%M:%S')} ({seconds_until_next/3600:.1f} hours from now)")
                    else:
                        main_logger.info(f"Next cleanup scheduled for {next_run.strftime('%Y-%m-%d %H:%M:%S')} ({seconds_until_next/60:.1f} minutes from now)")

                    # Sleep until next scheduled time
                    await asyncio.sleep(seconds_until_next)

                    if self.running:
                        main_logger.info(f"Running scheduled cleanup (removing threads older than {config.cleanup_max_age_hours} hours)...")
                        # Convert hours to seconds for the cleanup function
                        max_age_seconds = config.cleanup_max_age_hours * 3600
                        await self.processor.thread_manager.cleanup_old_threads(max_age=max_age_seconds)

                        # Also clean up old modal sessions (24 hours old)
                        if hasattr(self.processor, 'db') and self.processor.db:
                            await self.processor.db.cleanup_old_modal_sessions_async(hours=24)
                            main_logger.info("Cleaned up old modal sessions")

                            # F7: sweep aged tool-use provenance rows (no FK cascade —
                            # PRAGMA foreign_keys is never enabled, so these need their own
                            # age sweep, same as documents).
                            try:
                                self.processor.db.delete_old_tool_usage(
                                    days=config.tool_usage_retention_days)
                            except Exception as e:
                                main_logger.debug(f"Tool-usage sweep skipped: {e}")

                        stats = self.processor.get_stats()
                        main_logger.info(f"Cleanup complete. Stats: {stats}")
                except asyncio.CancelledError:
                    main_logger.info("Cleanup task cancelled")
                    break
                except Exception as e:
                    main_logger.error(f"Error in cleanup task: {e}")
                    # Wait 5 minutes before retrying on error
                    await asyncio.sleep(300)

        self.cleanup_task = asyncio.create_task(cleanup_worker())
        main_logger.info("Started cleanup task")
    
    async def run(self):
        """Run the bot"""
        log_session_start()

        try:
            await self.initialize()
            self.running = True

            # Start cleanup task
            await self.start_cleanup_task()

            # MCP startup health probe (informational; runs in the background so
            # a slow server can't delay boot). Strong ref so it can't be GC'd.
            if getattr(self.processor, "mcp_manager", None) and self.processor.mcp_manager.has_mcp_servers():
                self._mcp_probe_task = asyncio.create_task(self.processor.mcp_manager.health_probe())
                self._mcp_probe_task.add_done_callback(
                    lambda t: t.exception() and main_logger.warning(f"MCP health probe error: {t.exception()}"))

            # Start the client (blocks)
            main_logger.info(f"Starting {self.platform} bot...")
            if self.client:
                try:
                    await self.client.start()
                except asyncio.CancelledError:
                    main_logger.info("Bot client cancelled during shutdown")
                    pass

        except KeyboardInterrupt:
            main_logger.info("Received keyboard interrupt")
        except Exception as e:
            main_logger.error(f"Unexpected error: {e}", exc_info=True)
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Shutdown the bot gracefully"""
        if not self.running:
            return

        self.running = False
        main_logger.info(f"Shutting down {self.platform} bot...")

        # Cancel cleanup task
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        # F1: cancel/await in-flight background image generations BEFORE the Slack client
        # stops — otherwise the client tears down mid-upload and the jobs fail noisily.
        tm = getattr(self.processor, "thread_manager", None) if self.processor else None
        if tm is not None and hasattr(tm, "cancel_generations"):
            try:
                await tm.cancel_generations(timeout=5.0)
            except Exception as e:
                main_logger.warning(f"Error cancelling background generations: {e}")

        # Stop the client (this should interrupt any stuck operations)
        if self.client:
            try:
                await self.client.stop()
            except Exception as e:
                main_logger.warning(f"Error stopping client: {e}")

        # Clean up resources
        try:
            if self.processor:
                stats = self.processor.get_stats()
                main_logger.info(f"Final stats: {stats}")
                # Clean up processor resources
                await self.processor.cleanup()
        except Exception as e:
            main_logger.warning(f"Error during processor cleanup: {e}")

        # Give aiohttp sessions and pending coroutines a moment to clean up
        await asyncio.sleep(0.5)

        # Cancel any remaining tasks that might be lingering
        tasks = [t for t in asyncio.all_tasks() if t != asyncio.current_task()]
        if tasks:
            main_logger.warning(f"Cancelling {len(tasks)} remaining tasks...")
            for task in tasks:
                task.cancel()
            # Wait briefly for cancellation
            await asyncio.gather(*tasks, return_exceptions=True)

        log_session_end()
        main_logger.info("Shutdown complete")


async def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ChatGPT Slack Bot")
    parser.add_argument(
        "--platform",
        choices=["slack"],
        default="slack",
        help="Chat platform to use (default: slack)"
    )

    args = parser.parse_args()

    # Create and run bot
    bot = ChatBotV2(platform=args.platform)
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())