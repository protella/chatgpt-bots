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
from base_client import BaseClient, Message, Response


class ChatBotV2:
    """Main application class for multi-platform chat bot"""
    
    def __init__(self, platform: str = "slack"):
        self.platform = platform.lower()
        self.client: Optional[BaseClient] = None
        self.processor = MessageProcessor()
        self.cleanup_thread = None
        self.running = False
        
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
            from slackbot import SlackBot
            self.client = SlackBot(message_handler=self.handle_message)
        elif self.platform == "discord":
            # Future: from discordbot import DiscordBot
            # self.client = DiscordBot(message_handler=self.handle_message)
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
            
            # Delete thinking indicator
            if thinking_id:
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
                    # Format and send text
                    formatted_text = client.format_text(response.content)
                    client.send_message(
                        message.channel_id,
                        message.thread_id,
                        formatted_text
                    )
                elif response.type == "image":
                    # Send image
                    image_data = response.content
                    client.send_image(
                        message.channel_id,
                        message.thread_id,
                        image_data.to_bytes(),
                        f"generated_image.{image_data.format}",
                        f"Generated image: {image_data.prompt}"
                    )
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
        """Handle shutdown signals"""
        main_logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
    
    def start_cleanup_thread(self):
        """Start background thread for periodic cleanup"""
        def cleanup_worker():
            while self.running:
                try:
                    # Clean up old thread states every 30 minutes
                    time.sleep(1800)
                    if self.running:
                        self.processor.thread_manager.cleanup_old_threads()
                        stats = self.processor.get_stats()
                        main_logger.info(f"Cleanup complete. Stats: {stats}")
                except Exception as e:
                    main_logger.error(f"Error in cleanup thread: {e}")
        
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
        
        # Stop the client
        if self.client:
            self.client.stop()
        
        # Clean up resources
        stats = self.processor.get_stats()
        main_logger.info(f"Final stats: {stats}")
        
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