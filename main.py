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
from message_processor.participation import (ParticipationEngine, resolve_participation_level,
                                             snooze_expiry_iso)
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

            # Hard rail BEFORE any model call. Mentions never route through this gate,
            # so the throttle can't silence an addressed message.
            if engine.over_throttle(pulse, channel_id, level):
                main_logger.debug(f"Participation gate: hourly cap reached in {channel_id} — silent")
                return None

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

            is_thread_reply = bool(ts and message.thread_id and message.thread_id != ts)
            verdict = await engine.evaluate(
                channel_id=channel_id, ts=ts, text=message.text,
                sender_name=message.metadata.get("user_real_name") or message.metadata.get("username"),
                is_thread_reply=is_thread_reply, level=level,
                directives=message.metadata.get("channel_directives"),
                memory_facts=memory_facts, channel_activity=channel_activity,
                unprompted_last_hour=unprompted,
                name_hit=message.metadata.get("participation_name_hit") is True,
                snoozed=message.metadata.get("participation_snoozed") is True,
            )
            if verdict is None:  # superseded by a newer message during debounce
                main_logger.debug("Participation gate: superseded during debounce — silent")
                return None
            main_logger.debug(f"Participation verdict: {verdict.action} ({verdict.reason})")

            if verdict.action == "react":
                react_ts = message.metadata.get("ts") or message.thread_id
                if hasattr(client, "react"):
                    try:
                        await client.react(channel_id, react_ts, verdict.emoji)
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
        """'Butt out' loop: ack with an emoji (no more words), snooze unprompted
        participation for PARTICIPATION_SNOOZE_HOURS, and leave a durable memory
        fact so the preference outlives the snooze. Mentions keep working."""
        channel_id = message.channel_id
        react_ts = message.metadata.get("ts") or message.thread_id
        if hasattr(client, "react"):
            try:
                await client.react(channel_id, react_ts, config.snooze_ack_emoji)
            except Exception as e:
                main_logger.debug(f"Backoff ack react failed: {e}")
        try:
            expiry = snooze_expiry_iso()
            await self.processor.db.set_channel_settings_async(
                channel_id, snoozed_until=expiry, updated_by="participation_engine")
            main_logger.info(f"Participation backoff: {channel_id} snoozed until {expiry}")
        except Exception as e:
            main_logger.warning(f"Backoff snooze write failed: {e}")
        try:
            if getattr(config, "enable_channel_memory", True) and self.processor.db:
                import datetime
                day = datetime.datetime.fromtimestamp(
                    float(react_ts), tz=datetime.timezone.utc).date().isoformat()
                await self.processor.db.add_channel_memory_async(
                    channel_id,
                    f"On {day} the channel asked for less unprompted participation from the assistant.",
                    author="participation_engine",
                )
        except Exception as e:
            main_logger.debug(f"Backoff memory write failed: {e}")

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

        # Phase F placement (plan §4a): a genuine top-level reply happens only when the
        # channel's reply_in_channel setting is ON (the setting wins over the engine's
        # placement verdict, which is logged but not authoritative) AND the trigger was
        # itself top-level. Everything else threads. Images always thread (enforced in
        # the image branch, which keys off message.thread_id regardless).
        is_top_level_trigger = message.metadata.get("ts") == message.thread_id
        place_in_channel = (
            bool(message.metadata.get("reply_in_channel")) and is_top_level_trigger
            and bool(message.channel_id) and not message.channel_id.startswith("D")
        )
        if placement_verdict:
            main_logger.debug(
                f"Placement: verdict={placement_verdict}, reply_in_channel_setting="
                f"{bool(message.metadata.get('reply_in_channel'))} → "
                f"{'channel' if place_in_channel else 'thread'}"
            )
        post_thread_id = None if place_in_channel else message.thread_id

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
            if thinking_id and isinstance(batch_size, int) and batch_size > 1 and hasattr(client, "update_message"):
                try:
                    await client.update_message(
                        message.channel_id, thinking_id,
                        f"{config.thinking_emoji} Catching up on {batch_size} messages..."
                    )
                except Exception as e:
                    main_logger.debug(f"Catch-up status update failed: {e}")

        try:
            # Process the message and get intent
            response = await self.processor.process_message(message, client, thinking_id)
            
            # Delete thinking indicator (but not if streaming was used - it's already the response)
            if thinking_id and not (response and response.metadata.get("streamed")):
                await client.delete_message(message.channel_id, thinking_id)
            
            # Phase E/F: participation stats — count replies the bot volunteered
            # (wake-gate 'respond' verdicts) so the engine can self-throttle later.
            if (response and message.channel_id and not message.channel_id.startswith("D")
                    and getattr(client, "channel_pulse", None) is not None):
                posted = (response.metadata.get("streamed")
                          or (response.type == "text" and (response.content or "").strip()))
                if posted:
                    try:
                        client.channel_pulse.record_bot_reply(
                            message.channel_id, message.metadata.get("ts"),
                            unprompted=message.metadata.get("participation_check") is True,
                        )
                    except Exception as e:
                        main_logger.debug(f"participation stat record failed: {e}")

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
                        await client.send_message(
                            message.channel_id,
                            post_thread_id,
                            formatted_text
                        )
                    # Phase 7: append a Configure footer under the response (channels only, any
                    # member can open settings). Separate trailing message → safe for streamed too.
                    # Best-effort: a cosmetic footer must never break message handling.
                    # Skipped for top-level placement — it would land as ANOTHER top-level
                    # message and read as spam.
                    if hasattr(client, "maybe_post_response_footer") and not place_in_channel:
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
                    # Show uploading status
                    upload_status_id = None
                    # Use the client's name (e.g., "SlackBot") or extract platform name
                    platform_name = client.name.replace("Bot", "") if hasattr(client, 'name') else "system"
                    upload_status = f"{config.circle_loader_emoji} Uploading image to {platform_name}..."
                    
                    if response.metadata.get("streamed"):
                        # For streamed cases, we have a separate status message - update that, NOT the prompt!
                        status_msg_id = response.metadata.get("status_message_id")
                        if status_msg_id and hasattr(client, 'update_message'):
                            await client.update_message(message.channel_id, status_msg_id, upload_status)
                            upload_status_id = status_msg_id
                        else:
                            # Fallback: create new status message if not provided
                            upload_status_id = await client.send_thinking_indicator(message.channel_id, message.thread_id)
                            if upload_status_id and hasattr(client, 'update_message'):
                                await client.update_message(message.channel_id, upload_status_id, upload_status)
                    else:
                        # Non-streaming case - create new status message
                        upload_status_id = await client.send_thinking_indicator(message.channel_id, message.thread_id)
                        if upload_status_id and hasattr(client, 'update_message'):
                            await client.update_message(message.channel_id, upload_status_id, upload_status)
                    
                    try:
                        # Send image
                        image_data = response.content
                        file_url = await client.send_image(
                            message.channel_id,
                            message.thread_id,
                            image_data.to_bytes(),
                            f"generated_image.{image_data.format}",
                            ""  # No caption - prompt already displayed via streaming
                        )

                        # Update thread state with the URL
                        if file_url:
                            await self.processor.update_last_image_url(
                                message.channel_id,
                                message.thread_id,
                                file_url
                            )
                    finally:
                        # Always release the latch — a wedged latch would stall the next
                        # edit for the full wait timeout.
                        if upload_manager and hasattr(upload_manager, "mark_upload_finished"):
                            upload_manager.mark_upload_finished(upload_thread_key)

                    # Wait 4 seconds then handle cleanup
                    if upload_status_id:
                        await asyncio.sleep(4)

                        # Delete the status message - the enhanced prompt message remains untouched
                        await client.delete_message(message.channel_id, upload_status_id)
                elif response.type == "error":
                    # Send error message
                    await client.handle_error(
                        message.channel_id,
                        message.thread_id,
                        response.content
                    )
        
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