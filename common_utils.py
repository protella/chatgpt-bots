import base64
import os
from copy import deepcopy
from prompts import IMAGE_CHECK_SYSTEM_PROMPT, IMAGE_GEN_SYSTEM_PROMPT
import requests
from dotenv import load_dotenv
from logger import setup_logger, get_log_level, get_logger

# Unset any existing log level environment variables to ensure .env values are used
if "UTILS_LOG_LEVEL" in os.environ:
    del os.environ["UTILS_LOG_LEVEL"]

# Load environment variables
load_dotenv()

# Read model configuration from environment
GPT_MODEL = os.environ.get("GPT_MODEL", "gpt-5")
UTILITY_MODEL = os.environ.get("UTILITY_MODEL", GPT_MODEL)

# Configure logging level from environment variable with fallback to INFO
LOG_LEVEL_NAME = os.environ.get("UTILS_LOG_LEVEL", "INFO").upper()
LOG_LEVEL = get_log_level(LOG_LEVEL_NAME)
# Initialize logger with the configured log level
logger = get_logger('common_utils', LOG_LEVEL)

def create_dalle3_prompt(message, gpt_Bot, thread_id):
    """
    Use ChatGPT to generate a DALL-E 3 prompt based on the message and chat history.
    
    This function creates a copy of the conversation history, adds the user's message,
    changes the system prompt to the image generation prompt, and gets a response from GPT
    that can be used as a DALL-E 3 prompt.
    
    Args:
        message (str): The user's message.
        gpt_Bot (ChatBot): The ChatBot instance.
        thread_id (str): The ID of the thread/conversation.
        
    Returns:
        object: The GPT response containing the DALL-E 3 prompt.
    """
    logger.info(f"Creating DALL-E 3 prompt from history for thread {thread_id}")
    
    # Create a deep copy of the conversation history to avoid modifying the original
    chat_history = deepcopy(gpt_Bot.conversations[thread_id]["messages"])
    
    # Replace base64 encoded images with descriptive placeholders to improve performance
    # while maintaining context about the images
    for message_obj in chat_history:
        if "content" in message_obj and isinstance(message_obj["content"], list):
            for i, content_item in enumerate(message_obj["content"]):
                if (isinstance(content_item, dict) and 
                    content_item.get("type") == "image_url" and 
                    "image_url" in content_item):
                    # For DALL-E 3 prompt generation, we want to preserve the context that an image was shown
                    # So we use a more descriptive placeholder than in the image check function
                    message_obj["content"][i] = {
                        "type": "text",
                        "text": "[An image was shared in the conversation]"
                    }
    
    # Add the user's message to the chat history
    chat_history.append(
        {"role": "user", "content": [{"type": "text", "text": message}]}
    )
    
    # Change the system prompt to the image generation prompt
    chat_history[0]['content'] = IMAGE_GEN_SYSTEM_PROMPT
    
    # Get a response from GPT to use as a DALL-E 3 prompt (use primary GPT model)
    dalle3_prompt = gpt_Bot.get_gpt_response(chat_history, GPT_MODEL)

    return dalle3_prompt


def check_for_image_generation(message, gpt_Bot, thread_id):
    """
    Check if a message is requesting image generation.
    
    Args:
        message (str): The message to check.
        gpt_Bot (ChatBot): The ChatBot instance.
        thread_id (str): The ID of the thread/conversation.
        
    Returns:
        bool: True if the user is requesting an image generation, False otherwise.
    """
    logger.info(f"Checking if message is requesting image generation: {message[:50]}...")
    
    # Create a deep copy of the conversation history to avoid modifying the original
    chat_history = deepcopy(gpt_Bot.conversations[thread_id]["messages"]) 
    
    # Log the original system prompt for debugging
    original_system_prompt = chat_history[0]['content']
    logger.debug(f"Original system prompt: {original_system_prompt[:100]}...")
    
    # Log the conversation history length for debugging
    history_length = len(chat_history)
    logger.debug(f"Conversation history length: {history_length}")
    
    # Replace base64 encoded images with placeholders to improve performance
    for message_obj in chat_history:
        if "content" in message_obj and isinstance(message_obj["content"], list):
            for i, content_item in enumerate(message_obj["content"]):
                if (isinstance(content_item, dict) and 
                    content_item.get("type") == "image_url" and 
                    "image_url" in content_item):
                    # Replace with a simple placeholder
                    message_obj["content"][i] = {
                        "type": "text",
                        "text": "[Image content not included for performance]"
                    }
    
    # Add the user's message to the chat history
    chat_history.append(
        {"role": "user", "content": [{"type": "text", "text": message}]}
    )
    
    # Add a final explicit instruction message to ensure True/False response
    chat_history.append(
        {"role": "user", "content": [{"type": "text", "text": "Based on my last message, am I requesting an image generation? Answer with ONLY the word 'True' or 'False'."}]}
    )
    
    # Change the system prompt to the image check prompt
    chat_history[0]['content'] = IMAGE_CHECK_SYSTEM_PROMPT
    
    # Log the modified system prompt for debugging
    logger.debug(f"Image check system prompt: {IMAGE_CHECK_SYSTEM_PROMPT[:100]}...")

    # Determine model for utility checks
    model_for_check = UTILITY_MODEL
    logger.info(f"Using model for image check: {model_for_check}")

    # Configure parameters based on model
    # Check if it's a GPT-5 reasoning model
    # Reasoning models: gpt-5, gpt-5-mini, gpt-5-nano (with dates)
    model_lower = model_for_check.lower()
    is_gpt5_reasoning = (
        model_lower.startswith("gpt-5") and 
        not "chat" in model_lower and
        any(x in model_lower for x in ["gpt-5-", "gpt-5-mini", "gpt-5-nano"])
    )
    
    if is_gpt5_reasoning:
        # GPT-5 reasoning models (nano, mini, full) only support temperature=1
        logger.debug("Configuring for GPT-5 reasoning model (temperature fixed at 1)")
        temperature = 1
        reasoning_effort = "minimal"  # Fastest reasoning for simple True/False
        verbosity = "low"  # Short responses
        max_completion_tokens = None  # Let model determine tokens needed
    else:
        # GPT-4, GPT-5-chat, and earlier models support temperature variations
        logger.debug("Configuring for non-reasoning model (temperature 0.0 for deterministic output)")
        temperature = 0.0  # Zero temperature for most deterministic True/False
        reasoning_effort = None  # Not supported
        verbosity = None  # Not supported
        max_completion_tokens = 10  # Works fine with 10 tokens

    is_image_request = gpt_Bot.get_gpt_response(
        chat_history, 
        model_for_check, 
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
        max_completion_tokens=max_completion_tokens
    )
    
    # Log the full response for debugging
    logger.info(f"Image check response: '{is_image_request.content}'")
    
    # Check for keywords in the response that indicate an image request
    response_text = is_image_request.content.strip().lower()
    
    # More robust checking for True/False responses
    if response_text == 'true' or 'true' in response_text:
        logger.info("Image check result: True (matched 'true')")
        return True
    elif 'image' in response_text and ('generat' in response_text or 'creat' in response_text):
        logger.info("Image check result: True (matched image generation keywords)")
        return True
    elif 'picture' in response_text and ('generat' in response_text or 'creat' in response_text):
        logger.info("Image check result: True (matched picture generation keywords)")
        return True
    elif 'dall' in response_text and 'e' in response_text:
        logger.info("Image check result: True (matched DALL-E reference)")
        return True
    else:
        logger.info("Image check result: False (no image generation intent detected)")
        return False


def download_and_encode_file(say, file_url, bot_token):
    """
    Download a file from Slack and encode it as base64.
    
    In order to download files from Slack, the bot's request needs to be authenticated
    to the workspace via the Slackbot token.
    
    Args:
        say (callable): A function to send messages to Slack.
        file_url (str): The URL of the file to download.
        bot_token (str): The Slackbot token for authentication.
        
    Returns:
        str or None: The base64-encoded file content, or None if an error occurred.
    """
    logger.info(f"Downloading and encoding file from {file_url}")
    
    headers = {"Authorization": f"Bearer {bot_token}"}
    response = requests.get(file_url, headers=headers)

    if response.status_code == 200:
        encoded_file = base64.b64encode(response.content).decode("utf-8")
        logger.debug(f"Successfully encoded file (size: {len(encoded_file)} bytes)")
        return encoded_file
    else:
        handle_error(say, response.status_code)
        return None


def handle_error(say, error, thread_ts=None):
    """
    Handle errors by sending an error message to Slack.
    
    Args:
        say (callable): A function to send messages to Slack.
        error (any): The error to handle.
        thread_ts (str, optional): The timestamp of the thread to reply to. Defaults to None.
    """
    logger.error(f"Handling error: {error}")
    
    try:
        say(
            f":no_entry: `An error occurred. Error details:` :no_entry:\n```{error}```",
            thread_ts=thread_ts,
        )
    except Exception as e:
        logger.error(f"Error sending error message: {e}", exc_info=True)


def format_message_for_debug(conversation_history):
    """
    Format a conversation history for debugging purposes.
    
    This function takes a conversation history and formats it as a string for debugging,
    replacing image data with placeholders.
    
    Args:
        conversation_history (dict): The conversation history to format.
        
    Returns:
        str: A formatted string representation of the conversation history.
    """
    formatted_output = []
    for message in conversation_history['messages']:
        role = message['role']
        content = message['content']
        
        message_texts = []  # To collect text and placeholders for each message

        # Check if content is a list (typically for 'user' or 'assistant' with mixed content)
        if isinstance(content, list):
            # Process each content item in the list
            for item in content:
                if item['type'] == 'text':
                    message_texts.append(item['text'])
                elif item['type'] == 'image_url':
                    # Add a placeholder for images
                    message_texts.append("[Image Data]")
        
        elif isinstance(content, str):
            # Directly append the content if it's a string
            message_texts.append(content)
                    
        # Join all parts of the message into a single string and append to the output
        formatted_message = ' '.join(message_texts)
        formatted_output.append(f"-- {role.capitalize()}: {formatted_message}")
    
    return "\n".join(formatted_output)


###############################