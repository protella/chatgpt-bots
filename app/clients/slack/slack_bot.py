#!/usr/bin/env python3
# slack_bot.py - Main entry point for Slack bot
import os
import sys
import logging
import json
from typing import Dict, List, Any, Optional
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv

# Import internal modules
from app.core.chatbot import ChatBot
from app.core.queue import QueueManager
from app.core.history import rebuild_thread_history, remove_slack_mentions, get_user_info
from app.core.logging import setup_logger
from app.core.config import ConfigService
from app.core.intent_service import is_image_request

# Load environment variables
load_dotenv()

# Set up logging
logger = setup_logger(__name__)

# Initialize the queue manager
queue_manager = QueueManager.get_instance()

# Initialize the config service
config_service = ConfigService()

# Initialize Slack app with bot token and socket mode
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN or not OPENAI_API_KEY:
    logger.error("Missing required environment variables (SLACK_BOT_TOKEN, SLACK_APP_TOKEN, or OPENAI_API_KEY)")
    sys.exit(1)

# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN)

# Initialize the ChatBot
chatbot = ChatBot(api_key=OPENAI_API_KEY)

# Cache to store temporary message timestamps for deletion
cleanup_messages: Dict[str, List[str]] = {}

def get_bot_user_id() -> str:
    """
    Get the bot's user ID from Slack API.
    
    Returns:
        The bot's user ID
    """
    try:
        auth_response = app.client.auth_test()
        return auth_response["user_id"]
    except Exception as e:
        logger.error(f"Error getting bot user ID: {str(e)}")
        sys.exit(1)
        
# Store bot user ID for later use
bot_user_id = get_bot_user_id()

def clean_temp_messages(channel_id: str, thread_ts: str) -> None:
    """
    Clean up temporary messages (e.g., "Thinking...").
    
    Args:
        channel_id: The Slack channel ID
        thread_ts: The thread timestamp
    """
    key = f"{channel_id}:{thread_ts}"
    if key in cleanup_messages and cleanup_messages[key]:
        for ts in cleanup_messages[key]:
            try:
                app.client.chat_delete(channel=channel_id, ts=ts)
            except Exception as e:
                logger.debug(f"Error deleting message: {str(e)}")
        
        # Clear the list
        cleanup_messages[key] = []

def process_and_respond(event: Dict[str, Any], say) -> None:
    """
    Process a Slack message and respond with GPT.
    
    Args:
        event: The Slack event
        say: The Slack say function to respond
    """
    # Extract key info from the event
    channel_id = event["channel"]
    is_thread = "thread_ts" in event
    thread_ts = event.get("thread_ts") if is_thread else event.get("ts")
    user_id = event.get("user")
    
    # Key for tracking cleanup messages
    cleanup_key = f"{channel_id}:{thread_ts}"
    
    logger.info(f"Processing message in thread {thread_ts}")
    
    # Check if this thread is already processing
    if queue_manager.is_processing_sync(thread_ts):
        logger.info(f"Thread {thread_ts} is already processing")
        busy_response = say(
            text="I'm busy processing another request in this thread. Please wait a moment.",
            thread_ts=thread_ts
        )
        if cleanup_key not in cleanup_messages:
            cleanup_messages[cleanup_key] = []
        cleanup_messages[cleanup_key].append(busy_response["ts"])
        return
    
    # Try to start processing this thread
    if not queue_manager.start_processing_sync(thread_ts):
        logger.info(f"Failed to start processing thread {thread_ts}")
        return  # Another call got here first
    
    try:
        # Get message text and remove mentions
        message_text = event.get("text", "")
        message_text = remove_slack_mentions(message_text)
        
        # Add username context if available
        user_first_name = get_user_info(app.client, user_id)
        if user_first_name:
            message_text = f"[username={user_first_name}] {message_text}"
        
        # Get thread configuration
        thread_config = config_service.get(thread_ts)
        
        # Check for config updates in the message
        extracted_config = config_service.extract_config_from_text(message_text)
        if extracted_config:
            config_service.update(thread_ts, extracted_config)
            thread_config = config_service.get(thread_ts)
            logger.info(f"Updated config from message: {extracted_config}")
        
        # Send "thinking" message
        thinking_response = say(
            text="Thinking...",
            thread_ts=thread_ts
        )
        if cleanup_key not in cleanup_messages:
            cleanup_messages[cleanup_key] = []
        cleanup_messages[cleanup_key].append(thinking_response["ts"])
        
        # Process attached images if any
        images = []
        if "files" in event:
            files = event.get("files", [])
            for file in files:
                if file.get("mimetype") in ["image/jpeg", "image/png", "image/webp", "image/gif"]:
                    file_url = file.get("url_private")
                    if file_url:
                        try:
                            from app.core.history import download_and_encode_image
                            encoded_image = download_and_encode_image(file_url, SLACK_BOT_TOKEN)
                            if encoded_image:
                                images.append(encoded_image)
                                logger.info(f"Added image to request: {file.get('name')}")
                        except Exception as e:
                            logger.error(f"Error processing image: {str(e)}")
        
        # Check if we need to rebuild conversation history for this thread
        if thread_ts not in chatbot.thread_responses:
            logger.info(f"New thread or restarted bot, rebuilding history for {thread_ts}")
            
            # If this is part of a thread, rebuild the conversation history
            if is_thread:
                # Don't use the rebuilt messages directly - instead just rebuild to get previous_response_id
                messages = rebuild_thread_history(app.client, channel_id, thread_ts, bot_user_id)
                
                # If we have messages beyond the system prompt, let's simulate a series of turns
                # to build up the OpenAI history properly
                if len(messages) > 1:
                    logger.info(f"Initializing thread with {len(messages) - 1} synthetic turns")
                    
                    # For each message, send it to OpenAI as if it were a new turn
                    # This initializes the conversation and sets up previous_response_id properly
                    previous_id = None
                    for i, msg in enumerate(messages[1:]):  # Skip system prompt
                        # If this message is from the assistant, we need to use it as a reference
                        # but not send it (since we're using OpenAI to track conversation instead)
                        if msg["role"] == "assistant":
                            continue
                        
                        # For user messages, extract just the text from content list
                        if msg["role"] == "user":
                            content_text = ""
                            image_list = []
                            
                            # Extract text and images from content
                            for item in msg["content"]:
                                if item["type"] == "text":
                                    content_text += item["text"] + " "
                                elif item["type"] == "image_url":
                                    # Extract the base64 part from the URL (after data:image/png;base64,)
                                    image_url = item["image_url"]["url"]
                                    if "base64," in image_url:
                                        base64_data = image_url.split("base64,")[1]
                                        image_list.append(base64_data)
                            
                            # Send the message to OpenAI to build history
                            synthetic_response = chatbot.get_response(
                                input_text=content_text.strip(),
                                thread_id=thread_ts,
                                images=image_list
                            )
                            
                            # Store the previous response ID for the next turn
                            previous_id = chatbot.thread_responses.get(thread_ts)
                            
                            # Log but don't send the synthetic response
                            if synthetic_response["success"]:
                                logger.info(f"Successfully initialized history turn {i+1}")
                            else:
                                logger.error(f"Error initializing history: {synthetic_response['error']}")
            else:
                # This is a new conversation, not part of a thread
                logger.info("Starting new conversation")
        
        # Determine if message is an image request
        is_image_gen = False
        if not images:  # Only check text intent if no images attached
            try:
                is_image_gen = is_image_request(message_text, thread_ts, thread_config)
                logger.info(f"Intent detection for thread {thread_ts}: image_request={is_image_gen}")
            except Exception as e:
                logger.error(f"Error in intent detection: {str(e)}")
        
        # Get response from OpenAI (will add image-specific routing in next phase)
        response = chatbot.get_response(
            input_text=message_text,
            thread_id=thread_ts,
            images=images,
            config=thread_config  # Pass thread config to chatbot
        )
        
        # Clean up temporary messages
        clean_temp_messages(channel_id, thread_ts)
        
        # Send the response
        if response["success"]:
            say(text=response["content"], thread_ts=thread_ts)
        else:
            logger.error(f"Error from OpenAI: {response['error']}")
            say(
                text=f"I'm sorry, I encountered an error: {response['error']}",
                thread_ts=thread_ts
            )
    
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        try:
            # Clean up temporary messages
            clean_temp_messages(channel_id, thread_ts)
            
            # Send error message
            say(
                text=f"Sorry, I encountered an unexpected error: {str(e)}",
                thread_ts=thread_ts
            )
        except Exception as inner_e:
            logger.error(f"Error sending error message: {str(inner_e)}")
    
    finally:
        # Always make sure to release the thread lock
        queue_manager.finish_processing_sync(thread_ts)

# Define event handlers

@app.event("app_mention")
def handle_app_mention(event, say):
    """Handle mentions in channels or threads."""
    logger.info(f"Received app_mention event from user {event.get('user')}")
    
    # Ignore messages from bots and messages with subtypes
    if event.get("bot_id") or "subtype" in event:
        logger.debug("Ignoring bot message or message with subtype")
        return
    
    process_and_respond(event, say)

@app.event("message")
def handle_message(event, say):
    """Handle direct messages."""
    # Only handle messages in direct message channels
    if event.get("channel_type") != "im":
        return
    
    # Ignore messages from bots and messages with subtypes
    if event.get("bot_id") or "subtype" in event:
        logger.debug("Ignoring bot message or message with subtype")
        return
    
    logger.info(f"Received DM from user {event.get('user')}")
    process_and_respond(event, say)

@app.command("/chatgpt-config-dev")
def handle_config_command(ack, body, respond):
    """Handle the configuration slash command."""
    logger.info(f"Received /chatgpt-config-dev command from user {body.get('user_id')}")
    
    # Acknowledge the command
    ack()
    
    # Extract thread ID from the context (if available)
    channel_id = body.get("channel_id")
    thread_ts = body.get("thread_ts")
    
    thread_id = thread_ts if thread_ts else f"user_{body.get('user_id')}"
    
    try:
        # Get current config for this thread
        current_config = config_service.get(thread_id)
        
        # Check for a reset request in the text
        command_text = body.get("text", "").strip().lower()
        if command_text == "reset":
            config_service.reset(thread_id)
            respond({
                "response_type": "ephemeral",
                "text": "Configuration has been reset to defaults for this thread."
            })
            return
        
        # Format config as a message block
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "Thread Configuration",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Thread ID:* {thread_id}\n*Status:* Active"
                }
            },
            {
                "type": "divider"
            }
        ]
        
        # Add config sections
        sections = [
            {
                "title": "Model Settings",
                "fields": [
                    {"key": "gpt_model", "label": "GPT Model"},
                    {"key": "temperature", "label": "Temperature"},
                    {"key": "top_p", "label": "Top P"},
                    {"key": "max_output_tokens", "label": "Max Tokens"}
                ]
            },
            {
                "title": "Image Generation Settings",
                "fields": [
                    {"key": "image_model", "label": "Image Model"},
                    {"key": "size", "label": "Size"},
                    {"key": "quality", "label": "Quality"},
                    {"key": "style", "label": "Style"},
                    {"key": "number", "label": "Number of Images"},
                    {"key": "detail", "label": "Vision Detail"}
                ]
            }
        ]
        
        for section in sections:
            # Add section header
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*{section['title']}*"
                }
            })
            
            # Add fields
            fields_text = []
            for field in section["fields"]:
                key = field["key"]
                value = current_config.get(key, "Not set")
                fields_text.append(f"*{field['label']}*: {value}")
            
            # Split into columns (max 10 fields per section)
            for i in range(0, len(fields_text), 10):
                blocks.append({
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": text
                        } for text in fields_text[i:i+10]
                    ]
                })
        
        # Add reset button
        blocks.append({
            "type": "actions",
            "elements": [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "Reset to Defaults",
                        "emoji": True
                    },
                    "value": "reset_config",
                    "action_id": "reset_config"
                }
            ]
        })
        
        # Respond with the config details
        respond({
            "response_type": "ephemeral",
            "blocks": blocks
        })
        
    except Exception as e:
        logger.error(f"Error handling config command: {str(e)}")
        respond({
            "response_type": "ephemeral",
            "text": f"Error retrieving configuration: {str(e)}"
        })

@app.action("reset_config")
def handle_reset_action(ack, body, respond):
    """Handle the reset config button action."""
    ack()
    
    user_id = body.get("user", {}).get("id")
    channel_id = body.get("channel", {}).get("id")
    message_ts = body.get("message", {}).get("ts")
    
    # Extract thread info from the original message if available
    original_text = body.get("message", {}).get("text", "")
    thread_id_match = None
    
    # If we can extract the thread ID from the text
    if "Thread ID:" in original_text:
        thread_id_parts = original_text.split("Thread ID:")[1].split("\n")[0].strip()
        thread_id_match = thread_id_parts
    
    thread_id = thread_id_match if thread_id_match else f"user_{user_id}"
    
    try:
        # Reset the config
        config_service.reset(thread_id)
        
        # Respond with confirmation
        respond({
            "response_type": "ephemeral",
            "replace_original": False,
            "text": f"Configuration for thread {thread_id} has been reset to defaults."
        })
        
    except Exception as e:
        logger.error(f"Error resetting config: {str(e)}")
        respond({
            "response_type": "ephemeral",
            "replace_original": False,
            "text": f"Error resetting configuration: {str(e)}"
        })

def main():
    """Main entry point for the Slack bot."""
    logger.info("Starting Slack bot (V2)...")
    
    # Check for required environment variables
    required_env_vars = ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        sys.exit(1)
    
    logger.info("Environment variables loaded successfully.")
    
    # Start the socket mode handler
    handler = SocketModeHandler(app, SLACK_APP_TOKEN)
    logger.info("Starting socket mode handler...")
    handler.start()

if __name__ == "__main__":
    main() 