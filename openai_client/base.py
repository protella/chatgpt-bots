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
            "image_generation": 300.0,  # 5 minutes
            "image_edit": 300.0,        # 5 minutes
            "vision_analysis": 300.0,   # 5 minutes - large image analysis takes time

            # All text operations - 2.5 minutes regardless of complexity/tools/reasoning
            "text_high_reasoning": 150.0,  # 2.5 minutes (kept for compatibility)
            "text_with_tools": 150.0,      # 2.5 minutes
            "text_normal": 150.0,          # 2.5 minutes
            "intent_classification": 150.0, # 2.5 minutes
            "prompt_enhancement": 150.0,    # 2.5 minutes

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
        Safely iterate over a stream with proper timeout protection.

        Args:
            stream: The async stream to iterate over
            operation_type: Type of operation for timeout determination

        Yields:
            Events from the stream

        Raises:
            asyncio.TimeoutError: If stream times out (overall duration exceeded)
        """
        start_time = asyncio.get_event_loop().time()
        max_duration = self._get_operation_timeout(operation_type)
        # Use chunk timeout from config, but warn after 30s
        chunk_warning_threshold = 30.0  # Warn after 30 seconds of no chunks
        chunk_timeout = config.api_timeout_streaming_chunk  # Use config value from .env
        last_chunk_time = None
        first_event = True
        warned_about_delay = False  # Track if we've warned about this delay

        self.log_debug(f"Starting stream iteration with max_duration={max_duration}s, chunk_timeout={chunk_timeout}s")

        while True:
            try:
                # Check overall timeout
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > max_duration:
                    error_msg = f"Stream exceeded maximum duration of {max_duration}s (elapsed: {elapsed:.2f}s)"
                    self.log_error(error_msg)
                    raise asyncio.TimeoutError(error_msg)

                # Try to get next event with chunk timeout
                # All operations use the same chunk timeout (2.5 minutes)
                try:
                    event = await asyncio.wait_for(
                        stream.__anext__(),
                        timeout=chunk_timeout
                    )

                    # Track timing for monitoring
                    current_time = asyncio.get_event_loop().time()
                    if last_chunk_time is not None:
                        time_since_last_chunk = current_time - last_chunk_time
                        if time_since_last_chunk > chunk_warning_threshold:
                            self.log_debug(
                                f"Stream chunk received after {time_since_last_chunk:.1f}s"
                            )
                    elif first_event:
                        self.log_debug("Received first streaming event")
                        first_event = False

                    last_chunk_time = current_time
                    # Reset warning flag since we got a chunk
                    warned_about_delay = False
                    yield event

                except asyncio.TimeoutError:
                    # Chunk timeout - only warn, never fail
                    time_since_last = asyncio.get_event_loop().time() - (last_chunk_time or start_time)

                    # Warn after 30 seconds of no chunks (only once per delay)
                    if time_since_last >= chunk_warning_threshold and not warned_about_delay:
                        self.log_warning(
                            f"Stream chunk warning - no data received for {time_since_last:.1f}s. "
                            f"Continuing to wait (will timeout at {max_duration}s total)..."
                        )
                        warned_about_delay = True

                    # Always continue waiting - never fail on chunk timeout
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
