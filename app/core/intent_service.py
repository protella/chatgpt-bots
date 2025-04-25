"""Intent detection service for classifying user requests.

This module uses a lightweight OpenAI model to detect if a user message
is intended for image generation or text conversation.
"""

import os
import sys
import logging
from typing import Dict, Any

from openai import OpenAI
from prompts import IMAGE_CHECK_SYSTEM_PROMPT

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
    # Log the input text for easier debugging
    logger.debug(f"Analyzing text for image intent in thread {thread_id}: '{input_text}'")
    
    # Check for explicit image keywords in short messages
    # This helps with ambiguous follow-up requests
    input_lower = input_text.lower()
    if len(input_lower.split()) <= 5:  # Short message detection
        # Explicit image keywords that strongly indicate image intent
        explicit_keywords = [
            "image", "picture", "photo", "draw", "visualize", "create", 
            "generate", "show", "make", "dall-e"
        ]
        for keyword in explicit_keywords:
            if keyword in input_lower:
                logger.info(f"Short message with explicit image keyword '{keyword}' detected in thread {thread_id}")
                return True
    
    try:
        # Get the OpenAI API key from environment
        api_key = os.environ.get("OPENAI_KEY")
        if not api_key:
            logger.error("OpenAI API key not found in environment")
            return False
            
        client = OpenAI(api_key=api_key)
        
        # Get utility model from environment
        utility_model = os.environ.get("UTILITY_MODEL", "gpt-4o-mini")
        
        # Enhance system prompt with better handling for short or ambiguous messages
        enhanced_instruction = """
        IMPORTANT: Pay special attention to short or ambiguous follow-up messages like "how about now?" 
        or "try again" or "make it better" that might refer to previous image generation requests.
        These should be classified as 'True' when they appear to be continuing an image-related conversation.
        """
        
        # Prepare the messages for the Responses API
        messages = [
            {"role": "system", "content": IMAGE_CHECK_SYSTEM_PROMPT + enhanced_instruction},
            {"role": "user", "content": input_text},
            {"role": "user", "content": "Answer with ONLY the word 'True' or 'False'."}
        ]
        
        # Call the model with deterministic settings
        response = client.chat.completions.create(
            model=utility_model,
            messages=messages,
            temperature=0.0,
            max_tokens=10,
        )
        
        # Extract the text response
        result_text = response.choices[0].message.content.strip()
        logger.debug(f"Intent detection raw response: '{result_text}'")
        
        # Convert to boolean, handling potential malformed outputs
        if result_text.lower() == "true":
            logger.info(f"Image intent detected for message in thread {thread_id}")
            return True
        elif result_text.lower() == "false":
            logger.info(f"No image intent detected for message in thread {thread_id}")
            return False
        else:
            # Log malformed outputs
            logger.warning(
                f"Malformed intent detection output in thread {thread_id}. "
                f"Expected 'True'/'False', got: '{result_text}'"
            )
            # More aggressive fallback - if response contains "true" or image-related words
            # assume it's intended to be an image request
            if "true" in result_text.lower() or "image" in result_text.lower() or "picture" in result_text.lower():
                logger.info(f"Fallback image intent detected from malformed response in thread {thread_id}")
                return True
            # Gracefully reject by returning False for other malformed outputs
            return False
            
    except Exception as e:
        logger.error(f"Error in image intent detection for thread {thread_id}: {str(e)}")
        # Default to False on errors
        return False 