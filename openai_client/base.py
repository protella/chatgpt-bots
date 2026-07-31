from __future__ import annotations

import asyncio
from typing import Any, Callable, Dict, List, Optional

import aiohttp
from openai import AsyncOpenAI

from config import clamp_effort, config
from logger import LoggerMixin, setup_logger
from openai_client.container_errors import is_container_gone

from .api import images as image_api
from .api import responses as responses_api
from .api import tool_loop as tool_loop_api
from .api import vision as vision_api
from .utilities import ImageData

_request_log = setup_logger(name="slack_bot.request_builder")

# What a channel-layout request may carry. Typed items are the tool loop's own round-trip
# records; the content parts are the only ones the Responses API accepts inside a role message.
_CHANNEL_TYPED_ITEMS = ("function_call", "function_call_output", "reasoning")

# The keys each content part may keep, from the SDK's own param types
# (openai.types.responses.response_input_{text,image,file}_param). Our part dicts do double duty
# — API part AND DB metadata — so `source`, `url`, `original_url` and friends ride along and one
# of them is a 400 for the whole turn. Type alone is not enough: the keys have to be picked off.
#
# `file_id` is excluded on both image and file parts even though the SDK accepts it. It means an
# OPENAI file id and ours is Slack's (`F0BGSHE3JGJ`), which earns its own 400. Same rule as
# message_processor.utilities._API_PART_KEYS, which sanitizes upstream.
_CHANNEL_PART_KEYS: Dict[Any, tuple] = {
    "input_text": ("type", "text", "prompt_cache_breakpoint"),
    "input_image": ("type", "image_url", "detail", "prompt_cache_breakpoint"),
    "input_file": ("type", "filename", "file_data", "file_url", "detail",
                   "prompt_cache_breakpoint"),
}


def attach_cache_breakpoint(part: Dict[str, Any], model: Optional[str]) -> Dict[str, Any]:
    """Mark a content part as an explicit prompt-cache breakpoint.

    Explicit breakpoints exist only on the 5.6 family; on anything else the marker is an
    unknown parameter, so the part comes back unchanged and the miss is logged. Returns a NEW
    dict when it marks — never mutates the caller's part.
    """
    if not str(model or "").startswith("gpt-5.6"):
        _request_log.debug(
            f"prompt_cache_breakpoint unsupported on {model}; part left unmarked")
        return part
    return {**part, "prompt_cache_breakpoint": {"mode": "explicit"}}


def _channel_input_items(input_items: List[Dict[str, Any]],
                         model: str) -> List[Dict[str, Any]]:
    """The channel stream, allowlisted.

    Anything the API would reject is dropped rather than passed on: our own dicts do double
    duty (API part AND DB metadata), and one stray key is a 400 for the whole turn. Every part
    is rebuilt from `_CHANNEL_PART_KEYS` rather than forwarded, so the caller's dict is never
    mutated and a 5.5 retry never edits an input a later 5.6 retry reuses.
    """
    keep_breakpoints = model.startswith("gpt-5.6")
    items: List[Dict[str, Any]] = []
    for item in input_items or []:
        if not isinstance(item, dict):
            _request_log.debug("channel layout dropped a non-dict input item")
            continue
        item_type = item.get("type")
        if item_type is not None:
            if item_type in _CHANNEL_TYPED_ITEMS:
                items.append(item)
            else:
                _request_log.debug(f"channel layout dropped input item type {item_type!r}")
            continue
        role, content = item.get("role"), item.get("content")
        if not role:
            _request_log.debug("channel layout dropped an input item with no role")
            continue
        if isinstance(content, str):
            items.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            _request_log.debug(f"channel layout dropped {role} item with unusable content")
            continue
        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            allowed = _CHANNEL_PART_KEYS.get(part.get("type"))
            if allowed is None:
                continue
            clean = {k: v for k, v in part.items() if k in allowed and v is not None}
            if not keep_breakpoints:
                clean.pop("prompt_cache_breakpoint", None)
            parts.append(clean)
        if not parts:
            _request_log.debug(f"channel layout dropped {role} item with no usable parts")
            continue
        items.append({"role": role, "content": parts})
    return items


def _build_request_params(
    *,
    model: Optional[str],
    input_items: List[Dict[str, Any]],
    system_prompt: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stream: bool = False,
    store: bool = False,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    parallel_tool_calls: Optional[bool] = None,
    include: Optional[List[str]] = None,
    prompt_cache_key: Optional[str] = None,
    prompt_cache_options: Optional[Dict[str, Any]] = None,
    layout: str = "legacy",
    legacy_kind: str = "plain",
    legacy_cache_params: bool = True,
) -> Dict[str, Any]:
    """Assemble one Responses-API request.

    ``layout="legacy"`` reproduces what each caller built for itself: ``legacy_kind="plain"``
    prepends the system prompt as a developer input item, ``"tools"`` promotes it to top-level
    ``instructions``. DM turns stay on it, byte for byte, including the plain timeout twin's
    missing cache params (``legacy_cache_params=False``) — that gap is a bug, but it is a
    SHIPPED request shape and only the channel layout fixes it.

    ``layout="channel"`` is the one canonical shape (spec §3): instructions for every path,
    allowlisted input items, cache policy applied everywhere.
    """
    model = model or config.gpt_model
    temperature = temperature if temperature is not None else config.default_temperature
    max_output_tokens = max_output_tokens or config.default_max_tokens
    top_p = top_p if top_p is not None else config.default_top_p
    # Clamp guards against stored/legacy efforts the model rejects (`minimal` on 5.6,
    # `max` on 5.5).
    effort = clamp_effort(model, reasoning_effort or config.default_reasoning_effort)
    channel = layout == "channel"

    if channel:
        items = _channel_input_items(input_items, model)
    elif legacy_kind == "tools":
        # Raw Responses-API items (function_call / function_call_output from the tool loop)
        # carry a "type" and pass through as-is.
        items = [msg if "type" in msg else {"role": msg["role"], "content": msg["content"]}
                 for msg in input_items]
    else:
        items = [{"role": "developer", "content": system_prompt}] if system_prompt else []
        items += [{"role": msg["role"], "content": msg["content"]} for msg in input_items]

    params: Dict[str, Any] = {"model": model, "input": items}
    if tools is not None or legacy_kind == "tools":
        params["tools"] = tools
    params["temperature"] = temperature
    params["max_output_tokens"] = max_output_tokens
    params["store"] = store
    if stream:
        params["stream"] = True
    if parallel_tool_calls is not None:
        params["parallel_tool_calls"] = parallel_tool_calls
    if tool_choice is not None:
        params["tool_choice"] = tool_choice
    if include is not None:
        params["include"] = include
    if system_prompt and (channel or legacy_kind == "tools"):
        params["instructions"] = system_prompt

    params["reasoning"] = {"effort": effort}
    params["text"] = {"verbosity": verbosity or config.default_verbosity}

    # gpt-5.5 and the 5.6 family allow temperature/top_p when reasoning=none
    # (5.6 verified live 2026-07-09: effort=none + temperature/top_p -> 200)
    if (model.startswith("gpt-5.5") or model.startswith("gpt-5.6")) and effort == "none":
        params["top_p"] = top_p
    else:
        params["temperature"] = 1.0  # MUST be 1.0 for reasoning models

    if channel or legacy_cache_params:
        # gpt-5.5 keeps the explicit 24h retention param; the 5.6 family caches implicitly
        # (retention is deprecated there) and takes explicit breakpoints instead. The
        # per-thread key routes repeat calls to the same cache shard on both.
        if model.startswith("gpt-5.5"):
            params["prompt_cache_retention"] = "24h"
        if prompt_cache_key and (model.startswith("gpt-5.5") or model.startswith("gpt-5.6")):
            params["prompt_cache_key"] = prompt_cache_key
        if prompt_cache_options:
            if model.startswith("gpt-5.6"):
                params["prompt_cache_options"] = prompt_cache_options
            else:
                _request_log.debug(
                    f"prompt_cache_options dropped: unsupported on {model}")
    return params


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
        if hasattr(self, 'client') and self.client:
            await self.client.close()
            self.log_debug("OpenAI client closed and resources cleaned up")

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
            # Image operations - dedicated (longer) image budget; vision stays on read
            "image_generation": config.api_timeout_image,  # Dedicated image timeout
            "image_edit": config.api_timeout_image,        # Dedicated image timeout
            "vision_analysis": config.api_timeout_read,   # Use configured timeout

            # All text operations - use streaming chunk timeout from config
            "text_with_tools": config.api_timeout_streaming_chunk,      # Use configured timeout
            "text_normal": config.api_timeout_streaming_chunk,          # Use configured timeout
            "utility_call": config.api_timeout_streaming_chunk,          # Utility-model hops
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

                # Check if this is an MCP connection error (expected failure, handled gracefully)
                error_msg = str(e)
                is_mcp_error = "mcp server" in error_msg.lower() and ("404" in error_msg or "424" in error_msg)

                if is_mcp_error:
                    # MCP errors are handled gracefully by retry logic - log as WARNING
                    self.log_warning(f"MCP server connection failed after {elapsed:.2f}s: {error_msg}")
                elif is_container_gone(e):
                    # A code-interpreter container that idle-expires mid-stream is handled by the
                    # caller (it unbinds the container and re-runs without it), exactly like the
                    # MCP case above. Logging it as an ERROR-with-traceback made a recovered turn
                    # look like a crash in production.
                    self.log_warning(
                        f"Code-interpreter container expired mid-stream after {elapsed:.2f}s: {error_msg}")
                else:
                    # Unexpected errors - log as ERROR with stack trace
                    self.log_error(f"Stream error after {elapsed:.2f}s: {e}")
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
        prompt_cache_key: Optional[str] = None,
        prompt_cache_options: Optional[Dict[str, Any]] = None,
        usage_sink: Optional[Dict[str, Any]] = None,
        attempt_sink: Optional[Any] = None,
        layout: str = "legacy",
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
            prompt_cache_key=prompt_cache_key,
            prompt_cache_options=prompt_cache_options,
            usage_sink=usage_sink,
            attempt_sink=attempt_sink,
            layout=layout,
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
        return_metadata: bool = False,
        prompt_cache_key: Optional[str] = None,
        prompt_cache_options: Optional[Dict[str, Any]] = None,
        usage_sink: Optional[Dict[str, Any]] = None,
        attempt_sink: Optional[Any] = None,
        mcp_tools_sink: Optional[Dict[str, Any]] = None,
        # Pre-existing gap found while wiring F32: handlers/text.py has always passed
        # mcp_results_sink here, but this wrapper never accepted it — so the no-tool-loop
        # branch (ENABLE_TOOL_LOOP=false with web_search/MCP on) raised TypeError. The
        # tool-loop path forwards **params straight through and so never hit it.
        mcp_results_sink: Optional[List[Dict[str, Any]]] = None,
        artifacts_sink: Optional[List[Dict[str, Any]]] = None,
        container_gone_sink: Optional[List[str]] = None,
        layout: str = "legacy",
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
            return_metadata=return_metadata,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_options=prompt_cache_options,
            usage_sink=usage_sink,
            attempt_sink=attempt_sink,
            mcp_tools_sink=mcp_tools_sink,
            mcp_results_sink=mcp_results_sink,
            artifacts_sink=artifacts_sink,
            container_gone_sink=container_gone_sink,
            layout=layout,
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
        prompt_cache_key: Optional[str] = None,
        prompt_cache_options: Optional[Dict[str, Any]] = None,
        usage_sink: Optional[Dict[str, Any]] = None,
        attempt_sink: Optional[Any] = None,
        layout: str = "legacy",
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
            prompt_cache_key=prompt_cache_key,
            prompt_cache_options=prompt_cache_options,
            usage_sink=usage_sink,
            attempt_sink=attempt_sink,
            layout=layout,
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
        prompt_cache_key: Optional[str] = None,
        prompt_cache_options: Optional[Dict[str, Any]] = None,
        usage_sink: Optional[Dict[str, Any]] = None,
        attempt_sink: Optional[Any] = None,
        mcp_tools_sink: Optional[Dict[str, Any]] = None,
        # Same pre-existing gap as the non-streaming wrapper above — handlers/text.py passes
        # this on the no-tool-loop branch and it was never accepted here.
        mcp_results_sink: Optional[List[Dict[str, Any]]] = None,
        tool_event_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
        artifacts_sink: Optional[List[Dict[str, Any]]] = None,
        container_gone_sink: Optional[List[str]] = None,
        layout: str = "legacy",
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
            prompt_cache_key=prompt_cache_key,
            prompt_cache_options=prompt_cache_options,
            usage_sink=usage_sink,
            attempt_sink=attempt_sink,
            mcp_tools_sink=mcp_tools_sink,
            mcp_results_sink=mcp_results_sink,
            tool_event_callback=tool_event_callback,
            artifacts_sink=artifacts_sink,
            container_gone_sink=container_gone_sink,
            layout=layout,
        )

    async def create_text_response_with_tool_loop(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        registry: Any,
        tool_context: Any,
        **params: Any,
    ) -> Dict[str, Any]:
        """Phase A: non-streaming response with local function-call execution.
        Returns {"text", "tools_used", "local_tool_calls"}."""
        return await tool_loop_api.create_text_response_with_tool_loop(
            self,
            messages=messages,
            tools=tools,
            registry=registry,
            tool_context=tool_context,
            **params,
        )

    async def create_streaming_response_with_tool_loop(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        registry: Any,
        tool_context: Any,
        stream_callback: Callable[[Optional[str]], Any],
        tool_callback: Optional[Callable[[str, str], Any]] = None,
        **params: Any,
    ) -> Dict[str, Any]:
        """Phase A: streaming response with local function-call execution.
        Returns {"text", "tools_used", "local_tool_calls"}."""
        return await tool_loop_api.create_streaming_response_with_tool_loop(
            self,
            messages=messages,
            tools=tools,
            registry=registry,
            tool_context=tool_context,
            stream_callback=stream_callback,
            tool_callback=tool_callback,
            **params,
        )

    async def classify_wake(
        self,
        *,
        sources: Any,
        channel_steering_text: Optional[str] = None,
    ) -> Optional[bool]:
        """THE gate call: one bit. True/False as the model decided, or None when it produced
        nothing usable — which the engine turns into a decline, never into a decision."""
        return await responses_api.classify_wake(
            self, sources=sources, channel_steering_text=channel_steering_text)

    async def extract_memory(
        self,
        exchange_text: str,
        existing_memory: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Phase 9 post-response memory extraction. Returns {"action": "none"|"add"|"update", ...};
        conservative — any failure → {"action": "none"} (no write)."""
        return await responses_api.extract_memory(self, exchange_text=exchange_text, existing_memory=existing_memory)

    async def summarize_tool_result(self, text: str, max_chars: int) -> Optional[str]:
        """F16: compress ONE overlong MCP tool output to a single line under max_chars,
        preserving URLs/titles/dates/figures/IDs verbatim. Returns None on any failure
        (caller falls back to truncation); never raises."""
        return await responses_api.summarize_tool_result(self, text=text, max_chars=max_chars)

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

        try:
            # Use asyncio.wait_for for proper async timeout handling
            result = await asyncio.wait_for(
                api_method(*args, **kwargs),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            self.log_error(f"API call ({operation_type}) timed out after {timeout}s")
            # Create TimeoutError with operation_type attribute for smart retry logic
            timeout_error = TimeoutError(f"OpenAI API call timed out after {timeout} seconds")
            timeout_error.operation_type = operation_type
            raise timeout_error
        except Exception as e:
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
        model: Optional[str] = None,
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
            model=model,
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

    async def analyze_images(
        self,
        images: List[str],
        question: str,
        detail: Optional[str] = None,
        enhance_prompt: bool = False,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        stream_callback: Optional[Callable[[str], None]] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        verbosity: Optional[str] = None,
    ) -> str:
        # enhance_prompt is vestigial: the utility-model rewrite hop it used to gate is gone
        # (the only caller, image_catalog, always passed False). Kept solely so that explicit
        # `enhance_prompt=False` keyword keeps binding; drop it there and here together.
        #
        # model/reasoning_effort/verbosity override the analysis defaults (gpt_model +
        # analysis_*). The ambient vision worker passes the utility model + clamped utility
        # effort so a message the bot never answered doesn't cost primary-model spend, and the
        # recorded model matches the one that actually ran. Omitted → the original behavior.
        return await vision_api.analyze_images(
            self,
            images=images,
            question=question,
            detail=detail,
            conversation_history=conversation_history,
            system_prompt=system_prompt,
            stream_callback=stream_callback,
            model=model,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
        )

    async def edit_image(
        self,
        input_images: List[str],
        prompt: str,
        model: Optional[str] = None,
        input_mimetypes: Optional[List[str]] = None,
        image_description: Optional[str] = None,
        input_fidelity: str = "low",
        quality: Optional[str] = None,
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
            model=model,
            input_mimetypes=input_mimetypes,
            image_description=image_description,
            input_fidelity=input_fidelity,
            quality=quality,
            background=background,
            mask=mask,
            output_format=output_format,
            output_compression=output_compression,
            enhance_prompt=enhance_prompt,
            conversation_history=conversation_history,
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
        prompt_cache_key: Optional[str] = None,
        prompt_cache_options: Optional[Dict[str, Any]] = None,
        attempt_sink: Optional[Any] = None,
        layout: str = "legacy",
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
            prompt_cache_key=prompt_cache_key,
            prompt_cache_options=prompt_cache_options,
            attempt_sink=attempt_sink,
            layout=layout,
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
        timeout_seconds: float = 60.0,
        return_metadata: bool = False,
        prompt_cache_key: Optional[str] = None,
        prompt_cache_options: Optional[Dict[str, Any]] = None,
        usage_sink: Optional[Dict[str, Any]] = None,
        attempt_sink: Optional[Any] = None,
        mcp_tools_sink: Optional[Dict[str, Any]] = None,
        mcp_results_sink: Optional[List[Dict[str, Any]]] = None,
        artifacts_sink: Optional[List[Dict[str, Any]]] = None,
        container_gone_sink: Optional[List[str]] = None,
        layout: str = "legacy",
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
            timeout_seconds=timeout_seconds,
            return_metadata=return_metadata,
            prompt_cache_key=prompt_cache_key,
            prompt_cache_options=prompt_cache_options,
            usage_sink=usage_sink,
            attempt_sink=attempt_sink,
            mcp_tools_sink=mcp_tools_sink,
            mcp_results_sink=mcp_results_sink,
            artifacts_sink=artifacts_sink,
            container_gone_sink=container_gone_sink,
            layout=layout,
        )


__all__ = ["OpenAIClient", "ImageData", "attach_cache_breakpoint"]
