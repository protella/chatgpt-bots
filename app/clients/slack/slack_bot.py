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

# Get configuration values from environment
THINKING_EMOJI = os.environ.get("THINKING_EMOJI", ":thinking_face:")

# Initialize Slack app with bot token and socket mode
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_APP_TOKEN = os.environ.get("SLACK_APP_TOKEN")
OPENAI_KEY = os.environ.get("OPENAI_KEY")

if not SLACK_BOT_TOKEN or not SLACK_APP_TOKEN or not OPENAI_KEY:
    logger.error("Missing required environment variables (SLACK_BOT_TOKEN, SLACK_APP_TOKEN, or OPENAI_KEY)")
    sys.exit(1)

# Initialize the Slack app
app = App(token=SLACK_BOT_TOKEN)

# Initialize the ChatBot
chatbot = ChatBot(api_key=OPENAI_KEY)

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
        logger.info(f"Cleaning up {len(cleanup_messages[key])} temporary messages in thread {thread_ts}")
        
        for ts in cleanup_messages[key]:
            try:
                app.client.chat_delete(channel=channel_id, ts=ts)
                logger.debug(f"Successfully deleted temporary message {ts}")
            except Exception as e:
                # This is not a critical error - just log and continue
                logger.debug(f"Error deleting temporary message {ts}: {str(e)}")
        
        # Clear the list after attempting to delete all messages
        cleanup_messages[key] = []
        logger.info(f"Cleanup completed for thread {thread_ts}")
    else:
        logger.debug(f"No temporary messages to clean up for thread {thread_ts}")

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
            text="I'm still working on your last request. Please wait a moment and try again.",
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
        
        # Get user's first name for personalization
        user_first_name = get_user_info(app.client, user_id)
        if user_first_name:
            # Inject personalization tag at the beginning of the message
            message_text = f"[username={user_first_name}] {message_text}"
            logger.info(f"Added personalization tag for user {user_first_name}")
        
        # Get thread configuration
        thread_config = config_service.get(thread_ts)
        
        # Check for config updates in the message
        extracted_config = config_service.extract_config_from_text(message_text)
        if extracted_config:
            config_service.update(thread_ts, extracted_config)
            thread_config = config_service.get(thread_ts)
            logger.info(f"Updated config from message: {extracted_config}")
        
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
        if thread_ts not in chatbot.conversations:
            logger.info(f"New thread or restarted bot, rebuilding history for {thread_ts}")
            
            # If this is part of a thread, rebuild the conversation history
            if is_thread:
                logger.info(f"Rebuilding history for existing Slack thread {thread_ts}")
                
                # Get the conversation history from Slack API
                messages = rebuild_thread_history(app.client, channel_id, thread_ts, bot_user_id)
                logger.info(f"Retrieved {len(messages)} messages from Slack thread history")
                
                # Initialize the ChatBot with the rebuilt history
                chatbot.initialize_from_history(thread_ts, messages)
            else:
                # This is a new conversation, not part of a thread
                logger.info("Starting new conversation")
                # ChatBot will initialize a new conversation with system prompt on first call
        
        # Determine if message is an image request
        is_image_gen = False
        if not images:  # Only check text intent if no images attached
            try:
                is_image_gen = is_image_request(message_text, thread_ts, thread_config)
                logger.info(f"Intent detection for thread {thread_ts}: image_request={is_image_gen}")
            except Exception as e:
                logger.error(f"Error in intent detection: {str(e)}")
        
        # Create temporary message for showing progress
        thinking_response = None
        if cleanup_key not in cleanup_messages:
            cleanup_messages[cleanup_key] = []
        
        # Handle image generation if detected
        if is_image_gen:
            from app.core.image_service import generate_image, create_optimized_prompt, generate_image_description
            
            # Replace thinking with generating message for better UX
            thinking_response = say(
                text=f"Generating image, please wait... {THINKING_EMOJI}",
                thread_ts=thread_ts
            )
            cleanup_messages[cleanup_key].append(thinking_response["ts"])
            
            try:
                # Get conversation history for context (if available)
                conversation_history = None
                if thread_ts in chatbot.conversations:
                    conversation_history = chatbot.conversations[thread_ts]["messages"]
                    logger.info(f"Using conversation history with {len(conversation_history)} messages for context")
                
                # Create an optimized prompt with conversation context
                optimized_prompt = create_optimized_prompt(
                    message_text, 
                    thread_ts, 
                    thread_config,
                    conversation_history
                )
                logger.info(f"Created optimized image prompt: {optimized_prompt[:50]}...")
                
                # Generate the image
                image_bytes, revised_prompt, is_error = generate_image(
                    optimized_prompt, thread_ts, thread_config
                )
                
                if is_error or not image_bytes:
                    logger.error(f"Image generation failed in thread {thread_ts}")
                    # Clean up temporary messages
                    clean_temp_messages(channel_id, thread_ts)
                    
                    # Get the configured image model for the error message
                    image_model = thread_config.get(
                        "image_model", 
                        os.environ.get("GPT_IMAGE_MODEL", "unknown")
                    )
                    
                    # More helpful error message with model info
                    say(
                        text=f":warning: I couldn't generate that image using the {image_model} model. This could be due to content policy restrictions or a temporary service issue. Please try a different description or try again later.",
                        thread_ts=thread_ts
                    )
                else:
                    # Generate a detailed description of the image for future context
                    image_description = generate_image_description(
                        optimized_prompt,
                        revised_prompt,
                        thread_ts,
                        thread_config
                    )
                    
                    # Add this description to the chatbot's conversation history
                    if thread_ts in chatbot.conversations:
                        # Create an assistant message with the image description
                        image_msg = {
                            "role": "assistant",
                            "content": image_description
                        }
                        # Add to conversation history
                        chatbot.conversations[thread_ts]["messages"].append(image_msg)
                        logger.info("Added image description to conversation history")
                    
                    # Clean up temporary messages
                    clean_temp_messages(channel_id, thread_ts)
                    
                    # Prepare image description if revised prompt is available
                    file_comment = ""
                    if revised_prompt and thread_config.get("d3_revised_prompt", False):
                        file_comment = f"DALL·E generated image from prompt: _{revised_prompt}_"
                    
                    # Upload the generated image
                    app.client.files_upload_v2(
                        channel=channel_id,
                        file=image_bytes,
                        filename="generated_image.png",
                        initial_comment=file_comment,
                        thread_ts=thread_ts
                    )
                    
                    logger.info(f"Successfully uploaded generated image to thread {thread_ts}")
            except Exception as img_error:
                error_msg = str(img_error)
                logger.error(f"Error in image generation flow for thread {thread_ts}: {error_msg}")
                
                # Create a user-friendly error message
                user_error_msg = ":warning: I encountered an error while generating your image."
                
                # Add more details for specific error types
                if "content policy" in error_msg.lower() or "safety" in error_msg.lower():
                    user_error_msg += " Your request may have triggered content policy restrictions."
                elif "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    user_error_msg += " We've hit a rate limit or quota with our image provider."
                elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
                    user_error_msg += " The request timed out. The image service might be experiencing heavy load."
                else:
                    # For general errors, add a suggestion
                    user_error_msg += " Please try again with a different description or try later."
                
                # Clean up temporary messages
                clean_temp_messages(channel_id, thread_ts)
                say(
                    text=user_error_msg,
                    thread_ts=thread_ts
                )
            
            # We handled the image generation, so skip the regular text processing
            return
        else:
            # Send "thinking" message with configurable emoji for non-image requests
            thinking_response = say(
                text=f"Thinking... {THINKING_EMOJI}", 
                thread_ts=thread_ts
            )
            cleanup_messages[cleanup_key].append(thinking_response["ts"])
        
        # Get response from OpenAI
        response = chatbot.get_response(
            input_text=message_text,
            thread_id=thread_ts,
            images=images,
            config=thread_config  # Pass thread config to chatbot
        )
        
        # Send the response
        if response["success"]:
            # Send the response first, then clean up temporary messages
            say(text=response["content"], thread_ts=thread_ts)
            # Only clean up temporary messages after response is sent
            clean_temp_messages(channel_id, thread_ts)
        else:
            logger.error(f"Error from OpenAI: {response['error']}")
            # Send error response first, then clean up temporary messages
            say(
                text=":warning: Something went wrong. Please try again or contact support.",
                thread_ts=thread_ts
            )
            # Clean up temporary messages after error response is sent
            clean_temp_messages(channel_id, thread_ts)
    
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        try:
            # Send error message first
            say(
                text=":warning: Something went wrong. Please try again or contact support.",
                thread_ts=thread_ts
            )
            # Clean up temporary messages after error response is sent
            clean_temp_messages(channel_id, thread_ts)
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
    required_env_vars = ["SLACK_BOT_TOKEN", "SLACK_APP_TOKEN", "OPENAI_KEY"]
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