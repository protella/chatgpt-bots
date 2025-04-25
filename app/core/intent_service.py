"""Intent detection service for classifying user requests.

This module uses a lightweight OpenAI model to detect if a user message
is intended for image generation or text conversation.
"""

import logging
from typing import Dict, List, Any, Optional

from openai import OpenAI
import prompts

# Initialize logger
logger = logging.getLogger(__name__)


def is_image_request(input_text: str, thread_id: str, config: Dict[str, Any]) -> bool:
    """
    Determines whether the user message intends to trigger image generation.
    
    Args:
        input_text: The user's message text to analyze
        thread_id: The Slack thread ID (for logging purposes)
        config: Configuration dictionary containing model settings
    
    Returns:
        bool: True if the message is detected as an image request, False otherwise
    """
    try:
        client = OpenAI()
        
        # Prepare the messages for the Responses API
        messages = [
            {"role": "system", "content": prompts.IMAGE_CHECK_SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
            {"role": "user", "content": "Answer with ONLY the word 'True' or 'False'."}
        ]
        
        # Call the model with deterministic settings
        response = client.chat.completions.create(
            model="gpt-4.1-mini-2025-04-14",
            messages=messages,
            temperature=0.0,
            max_tokens=10,
            store=False
        )
        
        # Extract the text response
        result_text = response.choices[0].message.content.strip()
        
        # Convert to boolean, handling potential malformed outputs
        if result_text.lower() == "true":
            return True
        elif result_text.lower() == "false":
            return False
        else:
            # Log malformed outputs
            logger.warning(
                f"Malformed intent detection output in thread {thread_id}. "
                f"Expected 'True'/'False', got: '{result_text}'"
            )
            # Gracefully reject by returning False for malformed outputs
            return False
            
    except Exception as e:
        logger.error(f"Error in image intent detection for thread {thread_id}: {str(e)}")
        # Default to False on errors
        return False 