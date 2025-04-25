"""Image generation service for producing images based on text prompts.

This module interfaces with OpenAI's Images API to generate images based on text prompts,
supporting both GPT-Image-1 and DALL-E 3 models.
"""

import os
import base64
import logging
import sys
from typing import Dict, Tuple, Optional, Any, List

import openai

# Add the root directory to sys.path to allow importing prompts
sys.path.insert(0, '/app')

from prompts import IMAGE_GEN_SYSTEM_PROMPT

# Initialize logger
logger = logging.getLogger(__name__)

def create_optimized_prompt(
    input_text: str, 
    thread_id: str, 
    config: Dict[str, Any],
    conversation_history: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Create an optimized prompt for image generation using GPT, incorporating conversation history.
    
    Args:
        input_text: The user's original request text
        thread_id: The Slack thread ID (for logging/tracking)
        config: Configuration dictionary containing model settings
        conversation_history: Optional list of previous conversation messages for context
        
    Returns:
        str: Optimized prompt for image generation
    """
    try:
        # Create client with API key from environment
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_KEY"))
        
        # Get utility model from environment
        utility_model = os.environ.get("UTILITY_MODEL", "gpt-4o-mini")
        
        # Prepare the conversation context
        context_text = ""
        if conversation_history:         
            # Format the conversation messages into a readable format
            history_lines = []
            for msg in conversation_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                
                # Handle different content formats
                if isinstance(msg["content"], list):
                    text_parts = []
                    for item in msg["content"]:
                        if item["type"] == "text":
                            text_parts.append(item["text"])
                    msg_text = " ".join(text_parts)
                else:
                    msg_text = msg["content"]
                
                history_lines.append(f"{role}: {msg_text}")
            
            # Join the history into a context block
            if history_lines:
                context_text = "Previous conversation:\n" + "\n".join(history_lines) + "\n\n"
        
        # Construct the system prompt with context
        system_prompt = f"""You are an expert at creating detailed, optimized prompts for image generation.

{context_text}Based on the conversation history above and the current request below, create an optimized, detailed prompt for image generation.
Pay special attention to any feedback or requested changes to previous images.
Make the prompt detailed enough to create a high-quality, precise image.

Current request: {input_text}

Output ONLY the optimized prompt text with no additional explanation or commentary."""
        
        # Prepare the messages for the Responses API
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_text}
        ]
        
        # Call the model with specific settings for prompt generation
        response = client.chat.completions.create(
            model=utility_model,
            messages=messages,
            temperature=0.7,  # Allow some creativity
            max_tokens=300,   # Enough for a detailed prompt
        )
        
        # Extract the optimized prompt
        optimized_prompt = response.choices[0].message.content.strip()
        logger.info(f"Created optimized image prompt for thread {thread_id}")
        
        return optimized_prompt
        
    except Exception as e:
        logger.error(f"Error creating optimized prompt in thread {thread_id}: {str(e)}")
        # Fall back to original text on errors
        return input_text

def generate_image(
    prompt: str, 
    thread_id: str, 
    config: Dict[str, Any]
) -> Tuple[bytes, Optional[str], bool]:
    """
    Generate an image based on a text prompt using OpenAI's Images API.
    
    Args:
        prompt: The text description to generate an image from
        thread_id: The Slack thread ID (for logging/tracking)
        config: Configuration dictionary containing model settings
        
    Returns:
        Tuple containing:
        - image: The generated image as bytes
        - revised_prompt: The revised prompt (DALL-E 3 only) or None
        - is_error: Boolean indicating if an error occurred
    """
    try:
        # Create client with API key from environment
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_KEY"))
        
        # Get default image model from environment
        default_image_model = os.environ.get("GPT_IMAGE_MODEL", "gpt-image-1")
        dalle_model = os.environ.get("DALLE_MODEL", "dall-e-3")
        
        # Get image model from config, defaulting to environment variable
        image_model = config.get("image_model", default_image_model)
        
        # Set size parameter - common for all models
        size = config.get("size", "1024x1024")
        
        # Different handling based on model type
        if image_model == dalle_model:
            # DALL-E 3 only supports 'standard' or 'hd' quality
            dalle_quality = config.get("quality", "standard")
            if dalle_quality not in ["standard", "hd"]:
                dalle_quality = "standard"  # Default to standard if invalid
                
            # DALL-E 3 specific API call
            response = client.images.generate(
                model=image_model,
                prompt=prompt,
                n=1,
                size=size,
                quality=dalle_quality,
                style=config.get("style", "vivid"),
                response_format="b64_json"
            )
            
            # Extract base64 image data and decode
            image_base64 = response.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)
            
            # Get the revised prompt if enabled in config
            revised_prompt = None
            if config.get("d3_revised_prompt", False):
                revised_prompt = getattr(response.data[0], "revised_prompt", None)
                if revised_prompt:
                    logger.debug(f"DALL-E 3 revised prompt: {revised_prompt}")
            
        else:
            # GPT-Image-1 specific API call with appropriate parameters
            params = {
                "model": image_model,
                "prompt": prompt,
                "n": 2,
                "size": size,
                "moderation": config.get("moderation", "auto")  # Always include moderation parameter
            }
            
            # Handle quality parameter - gpt-image-1 supports 'auto', 'low', 'medium', 'high'
            gpt_quality = config.get("quality", "auto")
            valid_qualities = ["auto", "low", "medium", "high"]
            if gpt_quality in valid_qualities:
                params["quality"] = gpt_quality
            else:
                # Default to 'auto' for gpt-image-1 if invalid quality value
                params["quality"] = "auto"
                logger.warning(f"Invalid quality value '{gpt_quality}' for {image_model}. Using 'auto' instead.")
            
            # Handle output_format - gpt-image-1 supports 'png', 'jpeg', 'webp'
            if config.get("output_format"):
                output_format = config.get("output_format")
                valid_formats = ["png", "jpeg", "webp"]
                if output_format in valid_formats:
                    params["output_format"] = output_format
                else:
                    # Default to 'png' if invalid format
                    params["output_format"] = "png"
                    logger.warning(f"Invalid output_format '{output_format}' for {image_model}. Using 'png' instead.")
            
            # Handle output_compression - for jpeg/webp (0-100%)
            if config.get("output_compression") is not None:
                compression = config.get("output_compression")
                if isinstance(compression, int) and 0 <= compression <= 100:
                    params["output_compression"] = compression
                else:
                    logger.warning(f"Invalid output_compression value {compression}. Must be 0-100.")
            
            logger.info(f"Generating image with {image_model} in thread {thread_id} with params: {params}")
            
            # Call the API with appropriate parameters
            response = client.images.generate(**params)
            
            # For gpt-image-1, get the base64 data directly
            image_base64 = response.data[0].b64_json
            image_bytes = base64.b64decode(image_base64)
            revised_prompt = None
        
        logger.info(f"Successfully generated image in thread {thread_id}")
        return image_bytes, revised_prompt, False
        
    except Exception as e:
        error_details = str(e)
        logger.error(f"Error generating image in thread {thread_id}: {error_details}")
        
        # Extract and log additional details from OpenAI API errors
        if hasattr(e, 'response'):
            try:
                response_data = e.response
                status_code = getattr(response_data, 'status_code', 'unknown')
                error_message = getattr(response_data, 'text', str(e))
                logger.error(f"API error details - Status: {status_code}, Message: {error_message}")
            except Exception as detail_err:
                logger.error(f"Error extracting API error details: {str(detail_err)}")
        
        # Log model and prompt information for debugging
        try:
            image_model = config.get("image_model", os.environ.get("GPT_IMAGE_MODEL", "gpt-image-1"))
            logger.error(f"Failed image generation details - Model: {image_model}, Prompt length: {len(prompt)}")
            # Log a truncated version of the prompt for context
            truncated_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
            logger.error(f"Failed prompt: {truncated_prompt}")
        except Exception as log_err:
            logger.error(f"Error logging additional debugging info: {str(log_err)}")
        
        # Return empty bytes, no revised prompt, and error flag
        return bytes(), None, True

def generate_image_description(
    image_prompt: str,
    revised_prompt: Optional[str] = None,
    thread_id: str = "",
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a detailed description of an image to use for context in future iterations.
    
    Args:
        image_prompt: The original prompt used to generate the image
        revised_prompt: The DALL-E 3 revised prompt (if available)
        thread_id: The thread ID for logging
        config: Configuration options
        
    Returns:
        str: A detailed description of the image
    """
    try:
        # Create client with API key from environment
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_KEY"))
        
        # Get utility model from environment
        utility_model = os.environ.get("UTILITY_MODEL", "gpt-4o-mini")
        
        # Use the more detailed prompt (revised prompt from DALL-E if available)
        best_prompt = revised_prompt or image_prompt
        
        # Prepare the messages for the Responses API
        description_prompt = f"""
        Based on the following image generation prompt, create a detailed description of what 
        the image likely contains. Be specific about visual elements, colors, composition, 
        and any notable features. This description will be used for context in future 
        image iteration requests.
        
        PROMPT: {best_prompt}
        
        Provide only the description, without any introductory text like "Here's a description" or similar.
        """
        
        # Call the model with specific settings for description generation
        response = client.chat.completions.create(
            model=utility_model,
            messages=[{"role": "user", "content": description_prompt}],
            temperature=0.5,  # Balance between creativity and accuracy
            max_tokens=200,   # Enough for a detailed description
        )
        
        # Extract the image description
        description = response.choices[0].message.content.strip()
        logger.info(f"Generated image description for thread {thread_id}")
        
        # Format the description
        formatted_description = f"[GENERATED IMAGE DESCRIPTION: {description}]"
        return formatted_description
        
    except Exception as e:
        logger.error(f"Error generating image description in thread {thread_id}: {str(e)}")
        # Return a basic description on error
        return f"[GENERATED IMAGE based on prompt: {image_prompt}]" 