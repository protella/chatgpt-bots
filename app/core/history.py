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

def download_and_encode_image(image_url: str, bot_token: str) -> Optional[str]:
    """
    Download an image from Slack and encode it as base64.
    
    Args:
        image_url: The Slack URL for the image
        bot_token: The Slack bot token for authentication
        
    Returns:
        Base64-encoded image string or None if download fails
    """
    logger.info(f"Downloading image from {image_url}")
    
    headers = {"Authorization": f"Bearer {bot_token}"}
    try:
        response = requests.get(image_url, headers=headers)
        
        if response.status_code == 200:
            encoded_image = base64.b64encode(response.content).decode("utf-8")
            logger.debug(f"Successfully encoded image (size: {len(encoded_image)} bytes)")
            return encoded_image
        else:
            logger.error(f"Failed to download image: status code {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error downloading and encoding image: {str(e)}")
        return None

def rebuild_thread_history(client, channel_id: str, thread_ts: str, bot_user_id: str) -> List[Dict[str, Any]]:
    """
    Rebuild conversation history for a Slack thread.
    
    Args:
        client: The Slack client
        channel_id: The Slack channel ID
        thread_ts: The thread timestamp
        bot_user_id: The bot's user ID
        
    Returns:
        List of messages for OpenAI API
    """
    logger.info(f"Rebuilding conversation history for thread {thread_ts}")
    
    # Start with system prompt
    messages = [prompts.SLACK_SYSTEM_PROMPT]
    
    # Bot commands and responses to ignore when rebuilding history
    skip_messages = [
        "Thinking...",
        "Generating image",
        "I'm busy processing",
        "!help",
        "!usage",
        "!config",
        "!reset"
    ]
    
    try:
        # Fetch conversation replies from Slack API
        response = client.conversations_replies(channel=channel_id, ts=thread_ts)
        thread_messages = response.get("messages", [])
        
        # Process each message in the thread except the most recent (it will be added separately)
        for msg in thread_messages[:-1]:
            text = msg.get("text", "").strip()
            
            # Skip empty messages, bot commands, and processing messages
            if not text or any(text.startswith(cmd) for cmd in skip_messages) or any(cmd in text for cmd in skip_messages):
                continue
            
            # Determine message role based on user ID
            role = "assistant" if msg.get("user") == bot_user_id else "user"
            content = []
            
            # Remove slack mentions and add text content
            cleaned_text = remove_slack_mentions(text)
            if cleaned_text:
                content.append({
                    "type": "text",
                    "text": cleaned_text
                })
            
            # Process file attachments (images)
            files = msg.get("files", [])
            for file in files:
                if file.get("mimetype") in ALLOWED_MIMETYPES:
                    image_url = file.get("url_private")
                    if image_url:
                        # Get bot token from environment
                        slack_bot_token = os.environ.get("SLACK_BOT_TOKEN")
                        if not slack_bot_token:
                            logger.error("SLACK_BOT_TOKEN not found in environment")
                            continue
                            
                        # Download and encode the image
                        encoded_image = download_and_encode_image(
                            image_url, slack_bot_token
                        )
                        
                        if encoded_image:
                            # OpenAI API doesn't support images from assistant, so force to user
                            if role == "assistant":
                                # We'll create a new message from the "user" to show this image
                                image_msg = {
                                    "role": "user",
                                    "content": [{
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{encoded_image}",
                                            "detail": "auto"
                                        }
                                    }]
                                }
                                if image_msg["content"]:
                                    messages.append(image_msg)
                            else:
                                # Add image to current user message
                                content.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{encoded_image}",
                                        "detail": "auto"
                                    }
                                })
            
            # Add message to history if it has content
            if content:
                messages.append({
                    "role": role,
                    "content": content
                })
        
        logger.info(f"Rebuilt conversation history with {len(messages) - 1} messages (excluding system prompt)")
        return messages
        
    except Exception as e:
        logger.error(f"Error rebuilding thread history: {str(e)}")
        # Return just the system prompt if we failed to rebuild history
        return [prompts.SLACK_SYSTEM_PROMPT]

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