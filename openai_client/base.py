from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from openai import OpenAI

from config import config
from logger import LoggerMixin

from .api import images as image_api
from .api import responses as responses_api
from .api import vision as vision_api
from .utilities import ImageData


class OpenAIClient(LoggerMixin):
    """Wrapper for OpenAI API using Responses API."""

    def __init__(self):
        # Initialize OpenAI client with timeout directly
        # The OpenAI SDK accepts timeout as a parameter
        self.client = OpenAI(
            api_key=config.openai_api_key,
            timeout=config.api_timeout_read,  # Use read timeout as the overall timeout
            max_retries=0,  # Disable retries to fail fast on timeout
        )

        # Store streaming timeout for later use
        self.stream_timeout_seconds = config.api_timeout_streaming_chunk

        self.log_info(
            f"OpenAI client initialized with timeout: {config.api_timeout_read}s, "
            f"streaming_chunk: {self.stream_timeout_seconds}s, max_retries: 0"
        )
        self.log_debug(
            f"Client timeout object: {self.client.timeout}, type: {type(self.client.timeout)}"
        )

    def _safe_api_call(
        self,
        api_method: Callable,
        *args,
        timeout_seconds: Optional[float] = None,
        operation_type: str = "general",
        **kwargs,
    ):
        """Wrapper for OpenAI API calls with enforced timeout."""

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
            result = api_method(*args, **kwargs)
            call_duration = time.time() - call_start
            self.log_debug(f"[HANG_DEBUG] API call {api_name} completed in {call_duration:.2f}s")
            return result
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

    def create_text_response(
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
        return responses_api.create_text_response(
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

    def create_text_response_with_tools(
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
        return responses_api.create_text_response_with_tools(
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

    def create_streaming_response(
        self,
        messages: List[Dict[str, Any]],
        stream_callback: Callable[[Optional[str]], None],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        system_prompt: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        store: bool = False,
        tool_callback: Optional[Callable[[str, str], None]] = None,
    ) -> str:
        return responses_api.create_streaming_response(
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

    def create_streaming_response_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        stream_callback: Callable[[Optional[str]], None],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        system_prompt: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
        store: bool = False,
        tool_callback: Optional[Callable[[str, str], None]] = None,
    ) -> str:
        return responses_api.create_streaming_response_with_tools(
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

    def classify_intent(
        self,
        messages: List[Dict[str, Any]],
        last_user_message: str,
        has_attached_images: bool = False,
        max_retries: int = 2,
    ) -> str:
        return responses_api.classify_intent(
            self,
            messages=messages,
            last_user_message=last_user_message,
            has_attached_images=has_attached_images,
            max_retries=max_retries,
        )

    def generate_image(
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
        return image_api.generate_image(
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

    def _enhance_image_edit_prompt(
        self,
        user_request: str,
        image_description: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        return image_api._enhance_image_edit_prompt(
            self,
            user_request=user_request,
            image_description=image_description,
            conversation_history=conversation_history,
            stream_callback=stream_callback,
        )

    def _enhance_image_prompt(
        self,
        prompt: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        return image_api._enhance_image_prompt(
            self,
            prompt=prompt,
            conversation_history=conversation_history,
            stream_callback=stream_callback,
        )

    def _enhance_vision_prompt(
        self,
        user_question: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        return vision_api._enhance_vision_prompt(
            self,
            user_question=user_question,
            conversation_history=conversation_history,
        )

    def analyze_images(
        self,
        images: List[str],
        question: str,
        detail: Optional[str] = None,
        enhance_prompt: bool = True,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        return vision_api.analyze_images(
            self,
            images=images,
            question=question,
            detail=detail,
            enhance_prompt=enhance_prompt,
            conversation_history=conversation_history,
            system_prompt=system_prompt,
            stream_callback=stream_callback,
        )

    def edit_image(
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
        return image_api.edit_image(
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

    def analyze_image(
        self,
        image_data: str,
        question: str,
        detail: Optional[str] = None,
    ) -> str:
        return vision_api.analyze_image(
            self,
            image_data=image_data,
            question=question,
            detail=detail,
        )


__all__ = ["OpenAIClient", "ImageData"]
