"""Thread history reconstruction helpers for Slack.

This module provides functions to rebuild conversation history from Slack threads.
"""

import os
import base64
import logging
import requests
import re
from typing import Dict, List, Any, Optional

# Import system prompt
import prompts

# Import logging
from app.core.logging import setup_logger

logger = setup_logger(__name__)

# Allowed image MIME types that can be processed
ALLOWED_MIMETYPES = [
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
]

# Initialize constants
SLACK_BOT_THINKING_MESSAGES = ["_Thinking..._", "_Processing..._", "_Working on it..._"]

def remove_slack_mentions(text: str) -> str:
    """
    Remove Slack mention formatting from text (e.g., <@U123456>).
    
    Args:
        text: Text that may contain Slack mention formatting
        
    Returns:
        Text with Slack mentions removed
    """
    # This pattern matches Slack user mentions: <@U12345>
    user_id_pattern = r"<@[A-Z0-9]+>"
    # Replace mentions with empty string
    cleaned_text = re.sub(user_id_pattern, "", text)
    # Fix multiple spaces and normalize spacing
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    # Special case for punctuation with extra space
    cleaned_text = re.sub(r'\s+([,.!?])', r'\1', cleaned_text)
    return cleaned_text.strip()

def download_and_encode_image(url: str, token: str) -> str:
    """
    Download an image from Slack and encode it as base64.
    
    Args:
        url: The Slack file URL
        token: Slack API token
        
    Returns:
        Base64-encoded image data as string
    """
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return base64.b64encode(response.content).decode('utf-8')
    else:
        raise Exception(f"Failed to download image: {response.status_code}")

def rebuild_thread_history(
    client: Any, 
    channel_id: str, 
    thread_ts: str, 
    bot_user_id: str
) -> List[Dict[str, Any]]:
    """
    Rebuild a conversation history from a Slack thread.
    
    Args:
        client: Slack client
        channel_id: Channel ID
        thread_ts: Thread timestamp
        bot_user_id: Bot user ID
        
    Returns:
        List of messages in the OpenAI format
    """
    # Get thread messages
    result = client.conversations_replies(
        channel=channel_id,
        ts=thread_ts,
        limit=100
    )
    
    # Extract just the messages
    messages = result.get("messages", [])
    
    # Remove the current message (last message) to avoid processing it twice
    if len(messages) > 0:
        messages = messages[:-1]
    
    # Find the last user message and remove it - it's likely a "repeat conversation" request
    # or something we don't want in repeats
    last_user_message_index = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("user") != bot_user_id:
            last_user_message_index = i
            break
            
    if last_user_message_index is not None:
        logger.info(f"Removing last user message from history to prevent repetition")
        messages.pop(last_user_message_index)
    
    # Initialize OpenAI messages with system prompt
    openai_messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        }
    ]
    
    # Get the Slack token for downloading files
    slack_token = os.environ.get("SLACK_BOT_TOKEN")
    
    # Process messages
    for message in messages:
        # Skip bot thinking messages
        if (message.get("user") == bot_user_id and 
            any(thinking in message.get("text", "") for thinking in SLACK_BOT_THINKING_MESSAGES)):
            continue
        
        # Determine role
        if message.get("user") == bot_user_id:
            role = "assistant"
        else:
            role = "user"
        
        # Initialize content
        content = []
        
        # Add text content
        if message.get("text"):
            message_text = message.get("text")
            
            # Remove personalization tags from user messages
            # These are for internal use and shouldn't appear in the conversation history
            if role == "user":
                message_text = remove_personalization_tags(message_text)
            
            content.append({
                "type": "text",
                "text": message_text
            })
        
        # Add image content if any
        has_image_error = False
        if "files" in message and message["files"]:
            for file in message["files"]:
                if file.get("mimetype", "").startswith("image/"):
                    # Download and encode image
                    try:
                        image_data = download_and_encode_image(
                            file["url_private"], 
                            slack_token
                        )
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}",
                                "detail": "auto"
                            }
                        })
                    except Exception as e:
                        # Log error but continue processing
                        logger.error(f"Error processing image: {str(e)}")
                        has_image_error = True
        
        # Add the message to the OpenAI messages if it has content
        if content:
            openai_messages.append({
                "role": role,
                "content": content
            })
    
    return openai_messages

def get_user_info(client, user_id: str) -> Optional[str]:
    """
    Get the first name of a Slack user.
    
    Args:
        client: The Slack client
        user_id: The Slack user ID
        
    Returns:
        The user's first name or None if not found
    """
    try:
        response = client.users_info(user=user_id)
        user = response.get("user", {})
        
        # Try to get the first name from profile or real name if available
        profile = user.get("profile", {})
        first_name = profile.get("first_name")
        
        if not first_name:
            # Fall back to parsing the real name
            real_name = profile.get("real_name", "")
            if real_name:
                first_name = real_name.split(" ")[0]
        
        return first_name
    except Exception as e:
        logger.error(f"Error getting user info: {str(e)}")
        return None

def remove_personalization_tags(text: str) -> str:
    """
    Remove personalization tags from message text (e.g., [username=Peter]).
    
    Args:
        text: Text that may contain personalization tags
        
    Returns:
        Text with personalization tags removed
    """
    # This pattern matches personalization tags like [username=Peter]
    username_pattern = r"\[username=[^\]]+\]\s*"
    # Replace tags with empty string
    cleaned_text = re.sub(username_pattern, "", text)
    return cleaned_text.strip() 