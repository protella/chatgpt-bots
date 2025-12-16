from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from config import config
from prompts import VISION_ENHANCEMENT_PROMPT


async def _enhance_vision_prompt(
    client,
    user_question: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Enhance a vision analysis prompt based on conversation context."""

    self = client
    try:
        # Build enhancement messages with full context
        enhancement_messages = [{"role": "developer", "content": VISION_ENHANCEMENT_PROMPT}]

        # Include conversation history if provided (text only, no images)
        if conversation_history:
            for msg in conversation_history:
                # Only include text content, skip any image data
                content = msg.get("content", "")
                if isinstance(content, str):
                    enhancement_messages.append({"role": msg["role"], "content": content})

        # Add the current user message with image indicator
        enhancement_messages.append(
            {"role": "user", "content": f"[User has attached an image with this message]: {user_question}"}
        )

        # Ask for enhancement based on context
        enhancement_messages.append(
            {
                "role": "user",
                "content": "Based on the conversation above, create an appropriate prompt for analyzing the attached image.",
            }
        )

        # Build request parameters
        request_params = {
            "model": config.utility_model,
            "input": enhancement_messages,
            "max_output_tokens": 200,
            "store": False,
        }

        # Check if we're using a GPT-5 reasoning model
        if config.utility_model.startswith("gpt-5") and "chat" not in config.utility_model.lower():
            request_params["temperature"] = 1.0
            request_params["reasoning"] = {"effort": config.utility_reasoning_effort}
            request_params["text"] = {"verbosity": config.utility_verbosity}
        else:
            request_params["temperature"] = 0.7

        response = await self._safe_api_call(
            self.client.responses.create,
            operation_type="prompt_enhancement",
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

        if enhanced and len(enhanced) > 10:
            self.log_debug(f"Enhanced vision prompt: {enhanced[:100]}...")
            return enhanced
        else:
            return user_question  # Fallback to original

    except Exception as e:
        self.log_warning(f"Failed to enhance vision prompt: {e}")
        return user_question


async def analyze_images(
    client,
    images: List[str],
    question: str,
    detail: Optional[str] = None,
    enhance_prompt: bool = True,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    system_prompt: Optional[str] = None,
    stream_callback: Optional[Callable[[str], None]] = None,
) -> str:
    """Analyze one or more images with a question."""

    self = client
    detail = detail or config.default_detail_level

    # Limit to 10 images
    if len(images) > 10:
        self.log_warning(f"Limiting to 10 images (received {len(images)})")
        images = images[:10]

    # Enhance the question if requested
    enhanced_question = question
    if enhance_prompt:
        enhanced_question = await self._enhance_vision_prompt(question, conversation_history)
        self.log_info(f"Vision analysis with enhanced prompt: {enhanced_question[:100]}...")

    # Build content array with text and images
    content = [{"type": "input_text", "text": enhanced_question}]

    for image_data in images:
        # Use data URL format for base64 images
        content.append({"type": "input_image", "image_url": f"data:image/png;base64,{image_data}"})

    try:
        # Build request parameters with conversation history
        input_messages = []

        # Add platform system prompt if provided (for consistent personality/formatting)
        if system_prompt:
            input_messages.append({"role": "developer", "content": system_prompt})

        # Include FULL conversation history if provided (including all developer messages)
        # Filter out metadata to avoid API errors
        if conversation_history:
            for msg in conversation_history:
                # Only include role and content for API
                api_msg = {"role": msg["role"], "content": msg["content"]}
                input_messages.append(api_msg)

        # Add the current vision request
        input_messages.append({"role": "user", "content": content})

        # Check if streaming is requested
        if stream_callback:
            # Use streaming for vision analysis
            self.log_debug(f"Streaming vision analysis with {config.api_timeout_read}s timeout")

            request_params = {
                "model": config.gpt_model,
                "input": input_messages,
                "max_output_tokens": config.vision_max_tokens,
                "store": False,
                "stream": True,
            }

            if config.gpt_model.startswith("gpt-5") and "chat" not in config.gpt_model.lower():
                request_params["temperature"] = 1.0
                request_params["reasoning"] = {"effort": config.analysis_reasoning_effort}
                request_params["text"] = {"verbosity": config.analysis_verbosity}

                # Add prompt caching for GPT-5.1
                if config.gpt_model == "gpt-5.1":
                    request_params["prompt_cache_retention"] = "24h"

            # Stream the response
            output_text = ""
            stream = await self._safe_api_call(
                self.client.responses.create,
                operation_type="vision_analysis",
                **request_params,
            )

            # Process stream events with timeout protection
            async for event in self._safe_stream_iteration(stream, "vision_analysis"):
                try:
                    # Get event type
                    event_type = getattr(event, "type", "unknown")

                    if event_type == "response.created":
                        self.log_debug("Vision stream started")
                        continue
                    elif event_type in ["response.output_item.delta", "response.output_text.delta"]:
                        # Extract text from delta event
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
                            output_text += text_chunk
                            if stream_callback:
                                # Support both sync and async callbacks
                                result = stream_callback(text_chunk)
                                # If the callback returns a coroutine, await it
                                if hasattr(result, '__await__'):
                                    await result
                        continue
                    elif event_type == "response.output_item.done":
                        continue
                    elif event_type in ["response.done", "response.completed"]:
                        self.log_debug("Vision stream completed")
                        if stream_callback:
                            try:
                                # Support both sync and async callbacks
                                result = stream_callback(None)
                                # If the callback returns a coroutine, await it
                                if hasattr(result, '__await__'):
                                    await result
                            except Exception as callback_error:
                                self.log_warning(f"Stream completion callback error: {callback_error}")
                        break
                    else:
                        continue
                except Exception as event_error:
                    self.log_warning(f"Error processing vision stream event: {event_error}")
                    continue

            return output_text
        else:
            # Non-streaming version
            request_params = {
                "model": config.gpt_model,
                "input": input_messages,
                "max_output_tokens": config.vision_max_tokens,  # Use higher limit for vision with reasoning
                "store": False,
            }

            if config.gpt_model.startswith("gpt-5") and "chat" not in config.gpt_model.lower():
                request_params["temperature"] = 1.0
                request_params["reasoning"] = {"effort": config.analysis_reasoning_effort}
                request_params["text"] = {"verbosity": config.analysis_verbosity}

                # Add prompt caching for GPT-5.1
                if config.gpt_model == "gpt-5.1":
                    request_params["prompt_cache_retention"] = "24h"

            # API call with enforced timeout wrapper
            response = await self._safe_api_call(
                self.client.responses.create,
                operation_type="vision_analysis",
                **request_params,
            )

            # Extract response text
            output_text = ""
            if response.output:
                for item in response.output:
                    # Handle message/text type items
                    if hasattr(item, "content") and item.content:
                        for content in item.content:
                            # Handle output_text type for GPT-5 responses
                            if hasattr(content, "text"):
                                output_text += content.text
                            # Also check for type attribute
                            elif hasattr(content, "type") and content.type == "output_text" and hasattr(content, "text"):
                                output_text += content.text

            if not output_text:
                # Log the full response structure to debug
                self.log_warning("analyze_images returned empty response")
                self.log_debug(
                    f"Response output structure: {response.output if response.output else 'No output'}"
                )

                # Check if we only got reasoning tokens
                if response.usage and hasattr(response.usage, "output_tokens_details"):
                    details = response.usage.output_tokens_details
                    if hasattr(details, "reasoning_tokens") and details.reasoning_tokens > 0:
                        self.log_warning(
                            f"Response contained {details.reasoning_tokens} reasoning tokens but no text output"
                        )

            return output_text

    except Exception as e:
        self.log_error(f"Error analyzing images: {e}", exc_info=True)
        raise


async def analyze_image(
    client,
    image_data: str,
    question: str,
    detail: Optional[str] = None,
) -> str:
    """Analyze a single image (backward compatibility wrapper)."""

    self = client
    return await self.analyze_images([image_data], question, detail)
