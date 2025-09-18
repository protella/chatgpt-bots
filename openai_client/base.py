from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional

from openai import AsyncOpenAI

from config import config
from logger import LoggerMixin

from .api import images as image_api
from .api import responses as responses_api
from .api import vision as vision_api
from .utilities import ImageData


class OpenAIClient(LoggerMixin):
    """Async wrapper for OpenAI API using Responses API."""

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

        self.log_info(
            f"Async OpenAI client initialized with timeout: {config.api_timeout_read}s, "
            f"streaming_chunk: {self.stream_timeout_seconds}s, max_retries: 0"
        )
        self.log_debug(
            f"Client timeout object: {self.client.timeout}, type: {type(self.client.timeout)}"
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
        elif operation_type == "intent":
            # Intent classification should be fast
            timeout = min(30.0, config.api_timeout_read)
        elif operation_type == "streaming":
            # Use streaming chunk timeout from .env
            timeout = config.api_timeout_streaming_chunk
        else:
            # Use general timeout from .env (API_TIMEOUT_READ)
            timeout = config.api_timeout_read

        self.log_debug(
            f"Using timeout: {timeout}s for {operation_type} operation (from .env: "
            f"read={config.api_timeout_read}s, chunk={config.api_timeout_streaming_chunk}s)"
        )

        # Log httpx client state if available
        try:
            if hasattr(self.client, "_client") and hasattr(self.client._client, "_transport"):
                transport = self.client._client._transport
                if hasattr(transport, "_pool"):
                    pool = transport._pool
                    self.log_debug(
                        "[HANG_DEBUG] httpx connection pool state - connections: "
                        f"{len(pool._connections) if hasattr(pool, '_connections') else 'unknown'}"
                    )
                else:
                    self.log_debug("[HANG_DEBUG] httpx transport has no pool attribute")
            else:
                self.log_debug("[HANG_DEBUG] Unable to inspect httpx client state")
        except Exception as e:
            self.log_debug(f"[HANG_DEBUG] Error inspecting httpx state: {e}")

        api_name = getattr(api_method, "__name__", str(api_method))
        self.log_debug(f"[HANG_DEBUG] About to call OpenAI API: {api_name} with timeout={timeout}s")

        call_start = time.time()
        try:
            # Use asyncio.wait_for for proper async timeout handling
            result = await asyncio.wait_for(
                api_method(*args, **kwargs),
                timeout=timeout
            )
            call_duration = time.time() - call_start
            self.log_debug(f"[HANG_DEBUG] API call {api_name} completed in {call_duration:.2f}s")
            return result
        except asyncio.TimeoutError:
            call_duration = time.time() - call_start
            self.log_error(f"API call ({operation_type}) timed out after {timeout}s")
            self.log_debug(f"[HANG_DEBUG] API call {api_name} timed out after {call_duration:.2f}s")
            raise TimeoutError(f"OpenAI API call timed out after {timeout} seconds")
        except Exception as e:
            call_duration = time.time() - call_start
            self.log_debug(
                f"[HANG_DEBUG] API call {api_name} failed after {call_duration:.2f}s with error: {e}"
            )
            error_msg = str(e).lower()
            if "timeout" in error_msg or "timed out" in error_msg or "read timeout" in error_msg:
                self.log_error(f"API call ({operation_type}) timed out after {timeout}s: {e}")
                raise TimeoutError(f"OpenAI API call timed out after {timeout} seconds")
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

    async def close(self):
        """Close the OpenAI client and clean up resources."""
        if hasattr(self, 'client') and self.client:
            await self.client.close()
            self.log_debug("OpenAI client closed and resources cleaned up")


__all__ = ["OpenAIClient", "ImageData"]
