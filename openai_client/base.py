from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional

import aiohttp
from openai import AsyncOpenAI

from config import config
from logger import LoggerMixin

from .api import images as image_api
from .api import responses as responses_api
from .api import vision as vision_api
from .utilities import ImageData


class OpenAIClient(LoggerMixin):
    """Async wrapper for OpenAI API using Responses API."""

    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a reusable aiohttp session"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _cleanup_session(self):
        """Clean up aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self.log_debug("OpenAI client aiohttp session closed")

    async def close(self):
        """Close OpenAI client and cleanup resources"""
        await self._cleanup_session()
        # Note: AsyncOpenAI client doesn't need explicit closing

    def __init__(self):
        # Initialize async OpenAI client with timeout directly
        # The OpenAI SDK accepts timeout as a parameter
        self.client = AsyncOpenAI(
            api_key=config.openai_api_key,
            timeout=config.api_timeout_read,  # Use read timeout as the overall timeout
            max_retries=0,  # Disable retries to fail fast on timeout
        )

        # Store streaming timeout for later use
        self.stream_timeout_seconds = config.api_timeout_streaming_chunk

        # Initialize aiohttp session for image downloads
        self._session = None

        self.log_info(
            f"Async OpenAI client initialized with timeout: {config.api_timeout_read}s, "
            f"streaming_chunk: {self.stream_timeout_seconds}s, max_retries: 0"
        )
        self.log_debug(
            f"Client timeout object: {self.client.timeout}, type: {type(self.client.timeout)}"
        )

    def _get_operation_timeout(self, operation_type: str) -> float:
        """Get timeout for specific operation type based on complexity and expected duration."""

        # Operation-specific timeouts based on real-world usage patterns
        operation_timeouts = {
            # Image operations - full API timeout
            "image_generation": config.api_timeout_read,  # Use configured timeout
            "image_edit": config.api_timeout_read,        # Use configured timeout
            "vision_analysis": config.api_timeout_read,   # Use configured timeout

            # All text operations - use streaming chunk timeout from config
            "text_with_tools": config.api_timeout_streaming_chunk,      # Use configured timeout
            "text_normal": config.api_timeout_streaming_chunk,          # Use configured timeout
            "intent_classification": config.api_timeout_streaming_chunk, # Use configured timeout
            "prompt_enhancement": config.api_timeout_streaming_chunk,    # Use configured timeout

            # Streaming operations
            "streaming_chunk": config.api_timeout_streaming_chunk,  # Time between chunks
            "streaming": config.api_timeout_read,  # Overall streaming timeout

            # Fallback - use full API timeout
            "general": config.api_timeout_read,
        }

        timeout = operation_timeouts.get(operation_type, config.api_timeout_read)

        self.log_debug(f"Operation '{operation_type}' using timeout: {timeout}s")
        return timeout

    async def _safe_stream_iteration(self, stream, operation_type: str = "streaming"):
        """
        Safely iterate over a stream with activity-based timeout protection.

        Args:
            stream: The async stream to iterate over
            operation_type: Type of operation for timeout determination

        Yields:
            Events from the stream

        Raises:
            asyncio.TimeoutError: If no activity for timeout period
        """
        start_time = asyncio.get_event_loop().time()

        # Get the appropriate timeout for this operation type
        # This will be API_TIMEOUT_STREAMING_CHUNK for text operations
        # or API_TIMEOUT_READ for image/vision operations
        activity_timeout = self._get_operation_timeout(operation_type)

        # Warning threshold for long gaps between chunks
        chunk_warning_threshold = 30.0  # Warn after 30 seconds of no activity

        last_activity_time = start_time
        first_event = True
        warned_about_delay = False  # Track if we've warned about this delay
        total_events = 0

        self.log_debug(f"Starting stream iteration with activity-based timeout={activity_timeout}s for {operation_type}")

        while True:
            try:
                # Try to get next event with activity timeout
                # This timeout resets on each activity
                time_since_activity = asyncio.get_event_loop().time() - last_activity_time
                remaining_timeout = max(activity_timeout - time_since_activity, 1.0)

                try:
                    event = await asyncio.wait_for(
                        stream.__anext__(),
                        timeout=remaining_timeout
                    )

                    # We got activity! Reset the activity timer
                    current_time = asyncio.get_event_loop().time()
                    time_since_last = current_time - last_activity_time

                    # Log if there was a significant gap
                    if time_since_last > chunk_warning_threshold:
                        self.log_debug(
                            f"Stream event received after {time_since_last:.1f}s gap (event #{total_events + 1})"
                        )
                    elif first_event:
                        self.log_debug("Received first streaming event")
                        first_event = False

                    last_activity_time = current_time
                    warned_about_delay = False  # Reset warning flag on activity
                    total_events += 1

                    yield event

                except asyncio.TimeoutError:
                    # Check if we've exceeded the activity timeout
                    time_since_activity = asyncio.get_event_loop().time() - last_activity_time

                    if time_since_activity >= activity_timeout:
                        # No activity for the full timeout period - this is a real timeout
                        elapsed_total = asyncio.get_event_loop().time() - start_time
                        error_msg = (
                            f"Stream timeout: No activity for {activity_timeout}s "
                            f"(total elapsed: {elapsed_total:.1f}s, events received: {total_events})"
                        )
                        self.log_error(error_msg)
                        # Create TimeoutError with operation_type for retry logic
                        timeout_error = asyncio.TimeoutError(error_msg)
                        timeout_error.operation_type = operation_type
                        raise timeout_error

                    # Still within timeout but worth a warning
                    if time_since_activity >= chunk_warning_threshold and not warned_about_delay:
                        elapsed_total = asyncio.get_event_loop().time() - start_time
                        self.log_warning(
                            f"Stream activity warning: No data for {time_since_activity:.1f}s "
                            f"(will timeout after {activity_timeout}s of inactivity, "
                            f"total elapsed: {elapsed_total:.1f}s, events so far: {total_events})"
                        )
                        warned_about_delay = True

                    # Continue waiting for more activity
                    continue

            except StopAsyncIteration:
                # Stream completed normally
                elapsed = asyncio.get_event_loop().time() - start_time
                self.log_debug(f"Stream completed normally after {elapsed:.2f}s")
                break
            except Exception as e:
                # Don't catch the TimeoutError we raise for max duration
                if isinstance(e, asyncio.TimeoutError):
                    raise
                elapsed = asyncio.get_event_loop().time() - start_time
                self.log_error(f"Stream error after {elapsed:.2f}s: {e}")
                raise

    async def _safe_api_call(
        self,
        api_method: Callable,
        *args,
        timeout_seconds: Optional[float] = None,
        operation_type: str = "general",
        **kwargs,
    ):
        """Async wrapper for OpenAI API calls with enforced timeout."""

        # Determine timeout based on operation type and .env settings
        if timeout_seconds:
            timeout = timeout_seconds
        else:
            timeout = self._get_operation_timeout(operation_type)

        self.log_debug(
            f"Using timeout: {timeout}s for {operation_type} operation (from .env: "
            f"read={config.api_timeout_read}s, chunk={config.api_timeout_streaming_chunk}s)"
        )

        api_name = getattr(api_method, "__name__", str(api_method))

        call_start = time.time()
        try:
            # Use asyncio.wait_for for proper async timeout handling
            result = await asyncio.wait_for(
                api_method(*args, **kwargs),
                timeout=timeout
            )
            call_duration = time.time() - call_start
            return result
        except asyncio.TimeoutError:
            call_duration = time.time() - call_start
            self.log_error(f"API call ({operation_type}) timed out after {timeout}s")
            # Create TimeoutError with operation_type attribute for smart retry logic
            timeout_error = TimeoutError(f"OpenAI API call timed out after {timeout} seconds")
            timeout_error.operation_type = operation_type
            raise timeout_error
        except Exception as e:
            call_duration = time.time() - call_start
            error_msg = str(e).lower()
            if "timeout" in error_msg or "timed out" in error_msg or "read timeout" in error_msg:
                self.log_error(f"API call ({operation_type}) timed out after {timeout}s: {e}")
                # Create TimeoutError with operation_type attribute for smart retry logic
                timeout_error = TimeoutError(f"OpenAI API call timed out after {timeout} seconds")
                timeout_error.operation_type = operation_type
                raise timeout_error
            raise

    async def create_text_response(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        system_prompt: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        store: bool = False,
    ) -> str:
        return await responses_api.create_text_response(
            self,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            system_prompt=system_prompt,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            store=store,
        )

    async def create_text_response_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        system_prompt: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        store: bool = False,
    ) -> str:
        return await responses_api.create_text_response_with_tools(
            self,
            messages=messages,
            tools=tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            system_prompt=system_prompt,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            store=store,
        )

    async def create_streaming_response(
        self,
        messages: List[Dict[str, Any]],
        stream_callback: Callable[[Optional[str]], Any],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        system_prompt: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        store: bool = False,
        tool_callback: Optional[Callable[[str, str], Any]] = None,
    ) -> str:
        return await responses_api.create_streaming_response(
            self,
            messages=messages,
            stream_callback=stream_callback,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            system_prompt=system_prompt,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            store=store,
            tool_callback=tool_callback,
        )

    async def create_streaming_response_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        stream_callback: Callable[[Optional[str]], Any],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        system_prompt: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        store: bool = False,
        tool_callback: Optional[Callable[[str, str], Any]] = None,
    ) -> str:
        return await responses_api.create_streaming_response_with_tools(
            self,
            messages=messages,
            tools=tools,
            stream_callback=stream_callback,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            system_prompt=system_prompt,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            store=store,
            tool_callback=tool_callback,
        )

    async def classify_intent(
        self,
        messages: List[Dict[str, Any]],
        last_user_message: str,
        has_attached_images: bool = False,
        max_retries: int = 2,
    ) -> str:
        return await responses_api.classify_intent(
            self,
            messages=messages,
            last_user_message=last_user_message,
            has_attached_images=has_attached_images,
            max_retries=max_retries,
        )

    async def _safe_api_call(
        self,
        api_method: Callable,
        *args,
        timeout_seconds: Optional[float] = None,
        operation_type: str = "general",
        **kwargs,
    ):
        """Async wrapper for OpenAI API calls with enforced timeout."""

        # Determine timeout based on operation type and .env settings
        if timeout_seconds:
            timeout = timeout_seconds
        else:
            timeout = self._get_operation_timeout(operation_type)

        self.log_debug(
            f"Using timeout: {timeout}s for {operation_type} operation (from .env: "
            f"read={config.api_timeout_read}s, chunk={config.api_timeout_streaming_chunk}s)"
        )

        api_name = getattr(api_method, "__name__", str(api_method))

        call_start = time.time()
        try:
            # Use asyncio.wait_for for proper async timeout handling
            result = await asyncio.wait_for(
                api_method(*args, **kwargs),
                timeout=timeout
            )
            call_duration = time.time() - call_start
            return result
        except asyncio.TimeoutError:
            call_duration = time.time() - call_start
            self.log_error(f"API call ({operation_type}) timed out after {timeout}s")
            # Create TimeoutError with operation_type attribute for smart retry logic
            timeout_error = TimeoutError(f"OpenAI API call timed out after {timeout} seconds")
            timeout_error.operation_type = operation_type
            raise timeout_error
        except Exception as e:
            call_duration = time.time() - call_start
            error_msg = str(e).lower()
            if "timeout" in error_msg or "timed out" in error_msg or "read timeout" in error_msg:
                self.log_error(f"API call ({operation_type}) timed out after {timeout}s: {e}")
                # Create TimeoutError with operation_type attribute for smart retry logic
                timeout_error = TimeoutError(f"OpenAI API call timed out after {timeout} seconds")
                timeout_error.operation_type = operation_type
                raise timeout_error
            raise

    async def generate_image(
        self,
        prompt: str,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        background: Optional[str] = None,
        format: Optional[str] = None,
        compression: Optional[int] = None,
        enhance_prompt: bool = True,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> ImageData:
        return await image_api.generate_image(
            self,
            prompt=prompt,
            size=size,
            quality=quality,
            background=background,
            format=format,
            compression=compression,
            enhance_prompt=enhance_prompt,
            conversation_history=conversation_history,
        )

    async def _enhance_image_edit_prompt(
        self,
        user_request: str,
        image_description: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        return await image_api._enhance_image_edit_prompt(
            self,
            user_request=user_request,
            image_description=image_description,
            conversation_history=conversation_history,
            stream_callback=stream_callback,
        )

    async def _enhance_image_prompt(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        return await image_api._enhance_image_prompt(
            self,
            prompt=prompt,
            conversation_history=conversation_history,
            stream_callback=stream_callback,
        )

    async def _enhance_vision_prompt(
        self,
        user_question: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        return await vision_api._enhance_vision_prompt(
            self,
            user_question=user_question,
            conversation_history=conversation_history,
        )

    async def analyze_images(
        self,
        images: List[str],
        question: str,
        detail: Optional[str] = None,
        enhance_prompt: bool = True,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        return await vision_api.analyze_images(
            self,
            images=images,
            question=question,
            detail=detail,
            enhance_prompt=enhance_prompt,
            conversation_history=conversation_history,
            system_prompt=system_prompt,
            stream_callback=stream_callback,
        )

    async def edit_image(
        self,
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
        return await image_api.edit_image(
            self,
            input_images=input_images,
            prompt=prompt,
            input_mimetypes=input_mimetypes,
            image_description=image_description,
            input_fidelity=input_fidelity,
            background=background,
            mask=mask,
            output_format=output_format,
            output_compression=output_compression,
            enhance_prompt=enhance_prompt,
            conversation_history=conversation_history,
        )

    async def analyze_image(
        self,
        image_data: str,
        question: str,
        detail: Optional[str] = None,
    ) -> str:
        return await vision_api.analyze_image(
            self,
            image_data=image_data,
            question=question,
            detail=detail,
        )

    async def _create_text_response_with_timeout(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        system_prompt: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        store: bool = False,
        timeout_seconds: float = 60.0,
    ) -> str:
        return await responses_api._create_text_response_with_timeout(
            self,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            system_prompt=system_prompt,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            store=store,
            timeout_seconds=timeout_seconds,
        )

    async def _create_text_response_with_tools_with_timeout(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        system_prompt: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        store: bool = False,
        timeout_seconds: float = 60.0
    ) -> str:
        return await responses_api._create_text_response_with_tools_with_timeout(
            self,
            messages=messages,
            tools=tools,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            system_prompt=system_prompt,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            store=store,
            timeout_seconds=timeout_seconds
        )

    async def close(self):
        """Close the OpenAI client and clean up resources."""
        if hasattr(self, 'client') and self.client:
            await self.client.close()
            self.log_debug("OpenAI client closed and resources cleaned up")


__all__ = ["OpenAIClient", "ImageData"]
