#!/usr/bin/env python3
"""
Multi-Platform Chat Bot V2 - Main Entry Point
Supports multiple chat platforms with shared AI capabilities
"""
import sys
import signal
import time
import argparse
from threading import Thread
from typing import Optional
from config import config
from logger import log_session_start, log_session_end, main_logger
from message_processor import MessageProcessor
from base_client import BaseClient, Message


class ChatBotV2:
    """Main application class for multi-platform chat bot"""
    
    def __init__(self, platform: str = "slack"):
        self.platform = platform.lower()
        self.client: Optional[BaseClient] = None
        self.processor = None  # Will be initialized after client
        self.cleanup_thread = None
        self.running = False
        self.sigint_count = 0  # Track number of SIGINT received
        self.last_sigint_time = 0  # Track time of last SIGINT
        
    def initialize(self):
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
        elif self.platform == "discord":
            # Future: from discordbot import DiscordBot
            # self.client = DiscordBot(message_handler=self.handle_message)
            # self.processor = MessageProcessor(db=self.client.db)
            main_logger.error("Discord platform not yet implemented")
            sys.exit(1)
        else:
            main_logger.error(f"Unknown platform: {self.platform}")
            sys.exit(1)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        main_logger.info("Initialization complete")
    
    def handle_message(self, message: Message, client: BaseClient):
        """Handle incoming message from any platform"""
        # Send initial thinking indicator
        thinking_id = client.send_thinking_indicator(
            message.channel_id,
            message.thread_id
        )
        
        try:
            # Process the message and get intent
            response = self.processor.process_message(message, client, thinking_id)
            
            # Delete thinking indicator (but not if streaming was used - it's already the response)
            if thinking_id and not (response and response.metadata.get("streamed")):
                client.delete_message(message.channel_id, thinking_id)
            
            # Handle the response
            if response:
                if response.type == "busy":
                    # Special handling for busy state
                    if hasattr(client, 'send_busy_message'):
                        client.send_busy_message(message.channel_id, message.thread_id)
                    else:
                        client.send_message(
                            message.channel_id,
                            message.thread_id,
                            response.content
                        )
                elif response.type == "text":
                    # If streaming was used, the message is already displayed
                    if not response.metadata.get("streamed"):
                        # Format and send text
                        formatted_text = client.format_text(response.content)
                        client.send_message(
                            message.channel_id,
                            message.thread_id,
                            formatted_text
                        )
                elif response.type == "image":
                    # Show uploading status 
                    upload_status_id = None
                    # Use the client's name (e.g., "SlackBot") or extract platform name
                    platform_name = client.name.replace("Bot", "") if hasattr(client, 'name') else "system"
                    upload_status = f"{config.circle_loader_emoji} Uploading image to {platform_name}..."
                    
                    if response.metadata.get("streamed"):
                        # For streamed cases, we have a separate status message - update that, NOT the prompt!
                        status_msg_id = response.metadata.get("status_message_id")
                        if status_msg_id and hasattr(client, 'update_message'):
                            client.update_message(message.channel_id, status_msg_id, upload_status)
                            upload_status_id = status_msg_id
                        else:
                            # Fallback: create new status message if not provided
                            upload_status_id = client.send_thinking_indicator(message.channel_id, message.thread_id)
                            if upload_status_id and hasattr(client, 'update_message'):
                                client.update_message(message.channel_id, upload_status_id, upload_status)
                    else:
                        # Non-streaming case - create new status message
                        upload_status_id = client.send_thinking_indicator(message.channel_id, message.thread_id)
                        if upload_status_id and hasattr(client, 'update_message'):
                            client.update_message(message.channel_id, upload_status_id, upload_status)
                    
                    # Send image
                    image_data = response.content
                    file_url = client.send_image(
                        message.channel_id,
                        message.thread_id,
                        image_data.to_bytes(),
                        f"generated_image.{image_data.format}",
                        ""  # No caption - prompt already displayed via streaming
                    )
                    
                    # Update thread state with the URL
                    if file_url:
                        self.processor.update_last_image_url(
                            message.channel_id,
                            message.thread_id,
                            file_url
                        )
                    
                    # Wait 4 seconds then handle cleanup
                    if upload_status_id:
                        import time
                        time.sleep(4)
                        
                        # Delete the status message - the enhanced prompt message remains untouched
                        client.delete_message(message.channel_id, upload_status_id)
                elif response.type == "error":
                    # Send error message
                    client.handle_error(
                        message.channel_id,
                        message.thread_id,
                        response.content
                    )
        
        except Exception as e:
            main_logger.error(f"Error handling message: {e}", exc_info=True)
            
            # Delete thinking indicator on error
            if thinking_id:
                client.delete_message(message.channel_id, thinking_id)
            
            # Send error message
            client.handle_error(
                message.channel_id,
                message.thread_id,
                str(e)
            )
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals - double Ctrl-C for force exit"""
        import os
        
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
                self.shutdown()
            else:
                main_logger.warning("Shutdown already in progress... Press Ctrl-C again to force exit")
        else:
            # Handle other signals normally
            main_logger.info(f"Received signal {signum}, shutting down...")
            self.shutdown()
    
    def start_cleanup_thread(self):
        """Start background thread for periodic cleanup"""
        def cleanup_worker():
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
                    time.sleep(seconds_until_next)
                    
                    if self.running:
                        main_logger.info(f"Running scheduled cleanup (removing threads older than {config.cleanup_max_age_hours} hours)...")
                        # Convert hours to seconds for the cleanup function
                        max_age_seconds = config.cleanup_max_age_hours * 3600
                        self.processor.thread_manager.cleanup_old_threads(max_age=max_age_seconds)
                        stats = self.processor.get_stats()
                        main_logger.info(f"Cleanup complete. Stats: {stats}")
                except Exception as e:
                    main_logger.error(f"Error in cleanup thread: {e}")
                    # Wait 5 minutes before retrying on error
                    time.sleep(300)
        
        self.cleanup_thread = Thread(target=cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        main_logger.info("Started cleanup thread")
    
    def run(self):
        """Run the bot"""
        log_session_start()
        
        try:
            self.initialize()
            self.running = True
            
            # Start cleanup thread
            self.start_cleanup_thread()
            
            # Start the client (blocks)
            main_logger.info(f"Starting {self.platform} bot...")
            if self.client:
                self.client.start()
            
        except KeyboardInterrupt:
            main_logger.info("Received keyboard interrupt")
        except Exception as e:
            main_logger.error(f"Unexpected error: {e}", exc_info=True)
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown the bot gracefully"""
        if not self.running:
            return
        
        self.running = False
        main_logger.info(f"Shutting down {self.platform} bot...")
        
        # Stop the client (this should interrupt any stuck operations)
        if self.client:
            try:
                self.client.stop()
            except Exception as e:
                main_logger.warning(f"Error stopping client: {e}")
        
        # Clean up resources
        try:
            if self.processor:
                stats = self.processor.get_stats()
                main_logger.info(f"Final stats: {stats}")
        except Exception as e:
            main_logger.warning(f"Error getting final stats: {e}")
        
        log_session_end()
        sys.exit(0)


def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Multi-platform AI Chat Bot")
    parser.add_argument(
        "--platform",
        choices=["slack", "discord"],
        default="slack",
        help="Chat platform to use (default: slack)"
    )
    
    args = parser.parse_args()
    
    # Create and run bot
    bot = ChatBotV2(platform=args.platform)
    bot.run()


if __name__ == "__main__":
    main()