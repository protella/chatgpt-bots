"""Image generation service for producing images based on text prompts.

This module interfaces with OpenAI's Images API to generate images based on text prompts,
supporting both GPT-Image-1 and DALL-E 3 models.
"""

import os
import base64
import logging
from typing import Dict, Tuple, Optional, Any, Union
import openai

# Initialize logger
logger = logging.getLogger(__name__)

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
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Get image model from config
        image_model = config.get("image_model", "gpt-image-1")
        
        # Prepare API parameters
        params = {
            "model": image_model,
            "prompt": prompt,
            "n": 1,  # Only single image generation is supported
            "response_format": "b64_json"  # Get image as base64
        }
        
        # Handle model-specific parameters
        if image_model == "dall-e-3":
            # Add DALL-E 3 specific parameters
            params["size"] = config.get("size", "1024x1024")
            params["quality"] = config.get("quality", "standard")
            params["style"] = config.get("style", "vivid")
        elif image_model == "gpt-image-1":
            # GPT-Image-1 parameters
            # Only supports size, other DALL-E 3 parameters are ignored
            params["size"] = config.get("size", "1024x1024")
        
        logger.info(f"Generating image with {image_model} in thread {thread_id}")
        logger.debug(f"Image generation parameters: {params}")
        
        # Call the API
        response = client.images.generate(**params)
        
        # Extract base64 image data and decode
        image_base64 = response.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        
        # For DALL-E 3, get the revised prompt if enabled in config
        revised_prompt = None
        if image_model == "dall-e-3" and config.get("d3_revised_prompt", False):
            revised_prompt = getattr(response.data[0], "revised_prompt", None)
            if revised_prompt:
                logger.debug(f"DALL-E 3 revised prompt: {revised_prompt}")
        
        logger.info(f"Successfully generated image in thread {thread_id}")
        return image_bytes, revised_prompt, False
        
    except Exception as e:
        logger.error(f"Error generating image in thread {thread_id}: {str(e)}")
        # Return empty bytes, no revised prompt, and error flag
        return bytes(), None, True 