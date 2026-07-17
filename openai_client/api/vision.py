from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from config import config

# Hard ceiling on one image analysis. A real description is a few hundred characters; this is
# orders of magnitude above any legitimate answer, so it can only be hit by a stream that is
# not terminating — in which case truncating is the only outcome that leaves a machine standing.
_MAX_ANALYSIS_CHARS = 200_000


async def analyze_images(
    client,
    images: List[str],
    question: str,
    detail: Optional[str] = None,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    system_prompt: Optional[str] = None,
    stream_callback: Optional[Callable[[str], None]] = None,
    model: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
) -> str:
    """Analyze one or more images with a question.

    `model`/`reasoning_effort`/`verbosity` override the analysis defaults (config.gpt_model +
    config.analysis_*) — the ambient vision worker routes through the utility model so a message
    the bot never answered doesn't spend primary-model reasoning. Omitted → original behavior."""

    self = client
    detail = detail or config.default_detail_level
    vision_model = model or config.gpt_model
    vision_effort = reasoning_effort or config.analysis_reasoning_effort
    vision_verbosity = verbosity or config.analysis_verbosity

    # Limit to 10 images
    if len(images) > 10:
        self.log_warning(f"Limiting to 10 images (received {len(images)})")
        images = images[:10]

    # Build content array with text and images
    content = [{"type": "input_text", "text": question}]

    for image_data in images:
        # Accept every shape a caller might hand us, because one of them was silently 400ing.
        #
        # Callers hold images as the pipeline's attachment PARTS — dicts of
        # {type, image_url, source, filename, url, file_id} — not as bare base64 (see
        # _process_attachments). Interpolating one of those straight into an f-string produced
        # `data:image/png;base64,{'type': 'input_image', ...}`, which the API rejects with
        # "invalid base64-encoded value". Image cataloguing never once succeeded for an uploaded
        # image: every analysis died in the except below and the thread lost that context.
        #
        # A dict also carries its REAL mimetype in image_url, so honouring it fixes a second,
        # quieter bug — a JPEG was being announced to the API as image/png.
        part_detail = detail
        if isinstance(image_data, dict):
            url = image_data.get("image_url")
            part_detail = image_data.get("detail") or detail
        elif isinstance(image_data, str) and image_data.startswith("data:"):
            url = image_data
        else:
            url = f"data:image/png;base64,{image_data}"
        if not url or not isinstance(url, str):
            self.log_warning("Skipping an image with no usable data")
            continue
        part = {"type": "input_image", "image_url": url}
        if part_detail:
            part["detail"] = part_detail
        content.append(part)

    # If NOTHING usable survived, do not ask anyway. The model would happily answer the question
    # ("describe this image") from thin air, and the caller persists whatever comes back as a
    # real analysis — a hallucinated description of a picture nobody ever sent.
    if not any(p.get("type") == "input_image" for p in content):
        self.log_warning("No usable images to analyze — refusing to ask the model to imagine one")
        return ""

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
                "model": vision_model,
                "input": input_messages,
                "max_output_tokens": config.vision_max_tokens,
                "store": False,
                "stream": True,
            }

            # Primary model is a GPT-5-series reasoning model (gpt-5.5)
            request_params["temperature"] = 1.0
            request_params["reasoning"] = {"effort": vision_effort}
            request_params["text"] = {"verbosity": vision_verbosity}
            if vision_model.startswith("gpt-5.5"):
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
                            # Two guards, because this loop trusts the stream to be finite and
                            # to yield strings, and a stream that is neither will take the whole
                            # process down with it. `output_text += <non-str>` does not raise:
                            # str.__add__ returns NotImplemented, Python falls back to the
                            # right operand's __radd__, and output_text silently becomes that
                            # object — each subsequent += building a new one that retains the
                            # last. That is unbounded, and it is not hypothetical: it once ate
                            # 30GB and OOM-killed a dev box.
                            if not isinstance(text_chunk, str):
                                self.log_warning(
                                    f"Vision stream yielded a non-text delta "
                                    f"({type(text_chunk).__name__}); ending the stream.")
                                break
                            if len(output_text) + len(text_chunk) > _MAX_ANALYSIS_CHARS:
                                self.log_warning(
                                    f"Vision analysis exceeded {_MAX_ANALYSIS_CHARS} chars; "
                                    f"truncating. The stream may not be terminating.")
                                output_text += text_chunk[:_MAX_ANALYSIS_CHARS - len(output_text)]
                                break
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
                "model": vision_model,
                "input": input_messages,
                "max_output_tokens": config.vision_max_tokens,  # Use higher limit for vision with reasoning
                "store": False,
            }

            # Primary model is a GPT-5-series reasoning model (gpt-5.5)
            request_params["temperature"] = 1.0
            request_params["reasoning"] = {"effort": vision_effort}
            request_params["text"] = {"verbosity": vision_verbosity}
            if vision_model.startswith("gpt-5.5"):
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
