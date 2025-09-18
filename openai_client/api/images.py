from __future__ import annotations

import base64
from typing import Any, Callable, Dict, List, Optional

from config import config
from prompts import IMAGE_EDIT_SYSTEM_PROMPT, IMAGE_GEN_SYSTEM_PROMPT

from ..utilities import ImageData


async def generate_image(
    client,
    prompt: str,
    size: Optional[str] = None,
    quality: Optional[str] = None,
    background: Optional[str] = None,
    format: Optional[str] = None,
    compression: Optional[int] = None,
    enhance_prompt: bool = True,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> ImageData:
    """Generate an image using GPT-Image-1 model."""

    self = client
    # Default size for gpt-image-1
    size = size or config.default_image_size

    # Quality parameter is not used by gpt-image-1
    # Reserved for future DALL-E 3 support

    # Enhance prompt if requested
    enhanced_prompt = prompt
    if enhance_prompt:
        enhanced_prompt = await self._enhance_image_prompt(prompt, conversation_history)

    self.log_info(f"Generating image: {prompt[:100]}...")

    try:
        # Build parameters for images.generate
        # Default to gpt-image-1 parameters
        params = {
            "model": config.image_model,  # gpt-image-1
            "prompt": enhanced_prompt,  # Use the enhanced prompt
            "n": 1,  # Number of images to generate
        }

        # Add size if specified (gpt-image-1 supports size)
        if size:
            params["size"] = size

        # Note: gpt-image-1 doesn't support response_format or quality parameters
        # It returns URLs that we'll download and convert to base64

        # Future: When adding DALL-E 3 support, check model and add:
        # - response_format="b64_json"
        # - quality parameter
        # - style parameter

        # Use the images.generate API for image generation
        self.log_debug(f"Calling generate_image API with {config.api_timeout_read}s timeout")
        response = await self._safe_api_call(
            self.client.images.generate,
            operation_type="general",
            **params,
        )

        # Extract image data from response
        if response.data and len(response.data) > 0:
            # Check if we have base64 data
            if hasattr(response.data[0], "b64_json") and response.data[0].b64_json:
                image_data = response.data[0].b64_json
            # Otherwise, we might have a URL - need to download it
            elif hasattr(response.data[0], "url") and response.data[0].url:
                import requests

                url = response.data[0].url
                self.log_debug(f"Downloading image from URL: {url}")
                img_response = requests.get(url)
                if img_response.status_code == 200:
                    image_data = base64.b64encode(img_response.content).decode("utf-8")
                else:
                    raise ValueError(f"Failed to download image from URL: {url}")
            else:
                raise ValueError("No image data or URL in response")
        else:
            raise ValueError("No image data in response")

        self.log_info("Image generated successfully")

        return ImageData(
            base64_data=image_data,
            format="png",  # API always returns PNG for now
            prompt=enhanced_prompt,  # Store the enhanced prompt that was actually used
        )

    except Exception as e:
        self.log_error(f"Error generating image: {e}", exc_info=True)
        raise


async def _enhance_image_edit_prompt(
    client,
    user_request: str,
    image_description: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    stream_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """Enhance an image editing prompt using the analyzed image description."""

    self = client
    # Build context with image description and user request
    context = f"Image Description:\n{image_description}\n\nUser Edit Request:\n{user_request}"

    # Add conversation history if it exists and has messages
    if conversation_history and len(conversation_history) > 0:
        context = "Previous Conversation:\n"
        for msg in conversation_history:  # Full conversation history per CLAUDE.md
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle multi-part content
            if isinstance(content, list):
                text_parts = [c.get("text", "") for c in content if c.get("type") == "input_text"]
                content = " ".join(text_parts)

            # Include full message content per CLAUDE.md - no truncation
            context += f"{role}: {content}\n"

        context += f"\nImage Description:\n{image_description}\n\nUser Edit Request:\n{user_request}"

    # Log the enhancement input
    print("\n" + "=" * 80)
    print("DEBUG: IMAGE EDIT FLOW - STEP 4: EDIT PROMPT ENHANCEMENT")
    print("=" * 80)
    print(f"User Request: {user_request}")
    print(
        f"Image Description: {image_description[:200]}..."
        if len(image_description) > 200
        else f"Image Description: {image_description}"
    )
    if conversation_history and len(conversation_history) > 0:
        print(f"Including {len(conversation_history)} conversation messages for context")
    print("=" * 80)

    try:
        # Build request parameters with edit-specific system prompt
        request_params = {
            "model": config.utility_model,
            "input": [
                {"role": "developer", "content": IMAGE_EDIT_SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            "max_output_tokens": 500,
            "store": False,
        }

        # Check if we're using a GPT-5 reasoning model
        if config.utility_model.startswith("gpt-5") and "chat" not in config.utility_model.lower():
            request_params["temperature"] = 1.0
            request_params["reasoning"] = {"effort": config.utility_reasoning_effort}
            request_params["text"] = {"verbosity": config.utility_verbosity}
        else:
            request_params["temperature"] = 0.7

        # Check if streaming callback provided
        if stream_callback:
            # Create streaming response with timeout wrapper
            stream = await self._safe_api_call(
                self.client.responses.create,
                operation_type="streaming",
                stream=True,
                **request_params,
            )
            enhanced = ""

            async for event in stream:
                event_type = event.type if hasattr(event, "type") else None

                if event_type in ["response.output_item.delta", "response.output_text.delta"]:
                    # Extract text from delta event (same as in create_streaming_text_response)
                    text_chunk = None

                    # For response.output_text.delta, the text is directly in event.delta
                    if event_type == "response.output_text.delta" and hasattr(event, "delta"):
                        text_chunk = event.delta
                    # For response.output_item.delta, need to dig deeper
                    elif hasattr(event, "delta") and event.delta:
                        if hasattr(event.delta, "content") and event.delta.content:
                            for content in event.delta.content:
                                if hasattr(content, "text") and content.text:
                                    text_chunk = content.text
                                    break

                    if text_chunk:
                        enhanced += text_chunk
                        if stream_callback:
                            stream_callback(text_chunk)
        else:
            # Non-streaming fallback
            response = await self._safe_api_call(
                self.client.responses.create,
                operation_type="general",
                **request_params,
            )

            enhanced = ""
            if response.output:
                for item in response.output:
                    if hasattr(item, "content") and item.content:
                        for content in item.content:
                            if hasattr(content, "text"):
                                enhanced += content.text

        enhanced = enhanced.strip()

        # Log the enhanced result
        print("\n" + "=" * 80)
        print("DEBUG: IMAGE EDIT FLOW - STEP 5: ENHANCED EDIT PROMPT")
        print("=" * 80)
        print(f"Final Enhanced Edit Prompt:\n{enhanced}")
        print("=" * 80)

        if enhanced and len(enhanced) > 10:
            return enhanced
        else:
            # Fallback to simple combination
            # Return just the enhanced description without prefix
            return f"{image_description}. Change: {user_request}"

    except Exception as e:
        self.log_warning(f"Failed to enhance edit prompt: {e}")
        # Return just the enhanced description without prefix
        return f"{image_description}. Change: {user_request}"


async def _enhance_image_prompt(
    client,
    prompt: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    stream_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """Enhance an image generation prompt for better results."""

    self = client
    # Build conversation context
    context = "Conversation History:\n"

    if conversation_history:
        # Include full conversation history per CLAUDE.md
        for msg in conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle multi-part content
            if isinstance(content, list):
                text_parts = [c.get("text", "") for c in content if c.get("type") == "input_text"]
                content = " ".join(text_parts)

            # Include full message content per CLAUDE.md - no truncation
            context += f"{role}: {content}\n"

    context += f"\nCurrent User Request: {prompt}"

    # Log the prompt enhancement input
    print("\n" + "=" * 80)
    print("DEBUG: IMAGE EDIT FLOW - STEP 4: PROMPT ENHANCEMENT INPUT")
    print("=" * 80)
    print(f"Original Prompt to Enhance:\n{prompt}")
    print(f"\nFull Context Sent to Enhancer:\n{context}")
    print("=" * 80)

    try:
        # If streaming callback provided, use streaming response
        if stream_callback:
            # Use streaming version for real-time feedback
            enhanced = await self.create_streaming_response(
                messages=[{"role": "user", "content": context}],
                stream_callback=stream_callback,
                model=config.utility_model,
                temperature=
                0.7
                if "chat" in config.utility_model.lower() or not config.utility_model.startswith("gpt-5")
                else 1.0,
                max_tokens=500,
                system_prompt=IMAGE_GEN_SYSTEM_PROMPT,
                reasoning_effort=
                config.utility_reasoning_effort
                if config.utility_model.startswith("gpt-5") and "chat" not in config.utility_model.lower()
                else None,
                verbosity=
                config.utility_verbosity
                if config.utility_model.startswith("gpt-5") and "chat" not in config.utility_model.lower()
                else None,
                store=False,
            )
        else:
            # Use non-streaming version
            # Build request parameters
            request_params = {
                "model": config.utility_model,
                "input": [
                    {"role": "developer", "content": IMAGE_GEN_SYSTEM_PROMPT},
                    {"role": "user", "content": context},
                ],
                "max_output_tokens": 500,
                "store": False,
            }

            if config.utility_model.startswith("gpt-5") and "chat" not in config.utility_model.lower():
                request_params["temperature"] = 1.0
                request_params["reasoning"] = {"effort": config.utility_reasoning_effort}
                request_params["text"] = {"verbosity": config.utility_verbosity}
            else:
                request_params["temperature"] = 0.7

            response = await self._safe_api_call(
                self.client.responses.create,
                operation_type="general",
                **request_params,
            )

            enhanced = ""
            if response.output:
                for item in response.output:
                    if hasattr(item, "content") and item.content:
                        for content in item.content:
                            if hasattr(content, "text"):
                                enhanced += content.text

            enhanced = enhanced.strip()

        # Log the enhanced prompt result
        print("\n" + "=" * 80)
        print("DEBUG: IMAGE EDIT FLOW - STEP 5: ENHANCED PROMPT OUTPUT")
        print("=" * 80)
        print(f"Final Enhanced Prompt:\n{enhanced}")
        print("=" * 80)

        # Make sure we got a valid enhancement
        if enhanced and len(enhanced) > 10:
            self.log_debug(f"Enhanced prompt: {enhanced[:100]}...")
            return enhanced
        else:
            print("\n" + "=" * 80)
            print("DEBUG: Enhancement failed or too short, using original")
            print("=" * 80)
            return prompt

    except Exception as e:
        self.log_warning(f"Failed to enhance prompt: {e}")
        return prompt  # Return original on error


async def edit_image(
    client,
    input_images: List[str],
    prompt: str,
    input_mimetypes: Optional[List[str]] = None,
    image_description: Optional[str] = None,
    input_fidelity: str = "low",
    background: Optional[str] = None,
    mask: Optional[str] = None,
    output_format: str = "png",
    output_compression: int = 100,
    enhance_prompt: bool = True,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> ImageData:
    """Edit or combine images using GPT-Image-1 model."""

    self = client
    # Limit to 16 images
    if len(input_images) > 16:
        self.log_warning(f"Limiting to 16 images for editing (received {len(input_images)})")
        input_images = input_images[:16]

    # Default background
    background = background or config.default_image_background

    # Enhance prompt if requested
    enhanced_prompt = prompt
    if enhance_prompt:
        # Use the edit-specific enhancement for image editing
        if image_description:
            enhanced_prompt = await self._enhance_image_edit_prompt(
                user_request=prompt,
                image_description=image_description,
                conversation_history=conversation_history,
            )
        else:
            # Fallback to regular enhancement if no description
            enhanced_prompt = await self._enhance_image_prompt(prompt, conversation_history)

    self.log_info(f"Editing {len(input_images)} image(s): {prompt[:100]}...")

    try:
        # Convert base64 to BytesIO objects with proper file extension
        from io import BytesIO

        image_files = []

        # Default mimetypes if not provided
        if not input_mimetypes:
            input_mimetypes = ["image/png"] * len(input_images)

        for i, b64_data in enumerate(input_images):
            image_bytes = base64.b64decode(b64_data)
            bio = BytesIO(image_bytes)

            # Determine file extension from mimetype
            mimetype = input_mimetypes[i] if i < len(input_mimetypes) else "image/png"
            if mimetype == "image/jpeg":
                bio.name = f"image_{i}.jpg"
            elif mimetype == "image/webp":
                bio.name = f"image_{i}.webp"
            else:  # Default to PNG
                bio.name = f"image_{i}.png"

            image_files.append(bio)

        # Build parameters for images.edit
        params = {
            "model": config.image_model,  # gpt-image-1
            "image": image_files if len(image_files) > 1 else image_files[0],
            "prompt": enhanced_prompt,
            "input_fidelity": input_fidelity,
            "background": background,
            "output_format": output_format,
            "n": 1,
        }

        # Only add compression for JPEG/WebP (PNG must be 100)
        if output_format in ["jpeg", "webp"]:
            params["output_compression"] = output_compression
        elif output_format == "png" and output_compression != 100:
            self.log_debug(f"PNG format requires compression=100, ignoring {output_compression}")

        # Add mask if provided
        if mask:
            mask_bytes = base64.b64decode(mask)
            params["mask"] = BytesIO(mask_bytes)

        # Use the images.edit API
        self.log_debug(f"Calling edit_image API with {config.api_timeout_read}s timeout")
        response = await self._safe_api_call(
            self.client.images.edit,
            operation_type="general",
            **params,
        )

        # Extract image data from response
        if response.data and len(response.data) > 0:
            # Check if we have base64 data
            if hasattr(response.data[0], "b64_json") and response.data[0].b64_json:
                image_data = response.data[0].b64_json
            # Otherwise, we might have a URL - need to download it
            elif hasattr(response.data[0], "url") and response.data[0].url:
                import requests

                url = response.data[0].url
                self.log_debug(f"Downloading edited image from URL: {url}")
                img_response = requests.get(url)
                if img_response.status_code == 200:
                    image_data = base64.b64encode(img_response.content).decode("utf-8")
                else:
                    raise ValueError(f"Failed to download edited image from URL: {url}")
            else:
                raise ValueError("No image data or URL in response")
        else:
            raise ValueError("No image data in response")

        self.log_info("Image edited successfully")

        return ImageData(
            base64_data=image_data,
            format=output_format,
            prompt=enhanced_prompt,
        )

    except Exception as e:
        self.log_error(f"Error editing image: {e}", exc_info=True)
        raise
