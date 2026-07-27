from __future__ import annotations

import asyncio
import json
from typing import Any, Callable, Dict, List, Optional

from config import config, clamp_effort
from openai_client.container_errors import (demote_container_tools, is_container_gone,
                                            persistent_container_ids)
from prompts import (MEMORY_EXTRACTION_SYSTEM_PROMPT, TOOL_RESULT_SUMMARIZE_PROMPT,
                     WAKE_CLASSIFIER_SYSTEM_PROMPT)


def _segment_separator(text_so_far: str) -> str:
    """The join between two text segments of ONE reply: paragraph break, space, or nothing.

    A hosted tool splits the model's answer into SEPARATE output items — some text, the call,
    then more text. Nothing inside either item knows the other exists, so concatenating them raw
    glues two sentences together with no gap: live 2026-07-24, a code_interpreter round produced
    "…same approach Claude described.Third version is built via HTML/CSS…". The item boundary is
    the only place that knows a seam is there, which is why the separator is inserted here rather
    than in the presentation buffer (the non-streaming paths glue items too).

    A completed sentence gets a paragraph break. Text that stopped MID-sentence — "the total is"
    → sandbox → "56,088" — gets a single space, because breaking there would be worse than the
    bug. Text the model already ended in whitespace is left alone.
    """
    if not text_so_far or text_so_far[-1].isspace():
        return ""
    return ("\n\n" if text_so_far.rstrip("\"'*`)]}”’").endswith((".", "!", "?", "…", ":", ";"))
            else " ")


def _join_output_text(response) -> str:
    """Concatenate every text item of a non-streaming response, seams included."""
    text = ""
    for item in (getattr(response, "output", None) or []):
        content = getattr(item, "content", None)
        if not content:
            continue
        chunk = "".join(c.text for c in content if getattr(c, "text", None))
        if not chunk:
            continue
        text += _segment_separator(text) + chunk
    return text


def _capture_usage(usage_sink, response):
    """Copy response.usage into the caller's sink (usage-driven context budgeting)."""
    if usage_sink is None or response is None:
        return
    usage = getattr(response, "usage", None)
    if not usage:
        return
    usage_sink["input_tokens"] = getattr(usage, "input_tokens", 0) or 0
    usage_sink["output_tokens"] = getattr(usage, "output_tokens", 0) or 0


def _incomplete_reason(response) -> str:
    """Best-effort reason string from a `response.incomplete` terminal event."""
    details = getattr(response, "incomplete_details", None)
    if details is None:
        return "unknown"
    reason = getattr(details, "reason", None)
    if reason is None and isinstance(details, dict):
        reason = details.get("reason")
    return reason or "unknown"


def _stream_failure_error(response) -> Exception:
    """Build the exception a `response.failed` terminal event should raise.

    A failed stream is a real error, so — like the non-streaming path — it propagates to the
    caller. We surface the API's own error code/message when it carries one."""
    error = getattr(response, "error", None)
    code = getattr(error, "code", None)
    message = getattr(error, "message", None)
    if isinstance(error, dict):
        code = error.get("code", code)
        message = error.get("message", message)
    detail = message or code or "response.failed with no error detail"
    return RuntimeError(f"OpenAI streaming response failed: {detail}")


async def _create_with_container_recovery(self, request_params: Dict[str, Any],
                                         operation_type: str,
                                         container_gone_sink: Optional[List[str]] = None,
                                         **safe_call_kwargs):
    """`responses.create`, surviving a container that died since we verified it.

    The tool loop makes one Responses call per round with minutes of tool work between them, so
    a container confirmed alive at turn start can idle-expire before round 3. That 404 would
    otherwise fail the whole turn — the user gets an error instead of an answer, which is never
    a fair price for a sandbox nicety.

    On a container 404 we demote the tools array to `{"type": "auto"}` (a fresh throwaway
    container) and retry the SAME call once. Local tools already executed this turn are not
    replayed: only this one API call repeats. The dead id lands in `container_gone_sink` so the
    caller can drop its DB binding.
    """
    try:
        return await self._safe_api_call(
            self.client.responses.create, operation_type=operation_type,
            **safe_call_kwargs, **request_params)
    except Exception as e:  # noqa: BLE001 — re-raised below unless it is a dead container
        if not is_container_gone(e):
            raise
        demoted, changed = demote_container_tools(request_params.get("tools"))
        if not changed:
            # A 404 mentioning "container" but no explicit container of ours to blame. Retrying
            # would fail identically.
            raise

        dead = persistent_container_ids(request_params.get("tools"))
        if container_gone_sink is not None:
            container_gone_sink.extend(dead)
        self.log_warning(
            f"Container {dead} died mid-turn — retrying this call with an ephemeral sandbox")

        retry_params = {**request_params, "tools": demoted}
        return await self._safe_api_call(
            self.client.responses.create, operation_type=operation_type,
            **safe_call_kwargs, **retry_params)


def _collect_mcp_list_tools(mcp_tools_sink, item):
    """
    Harvest an mcp_list_tools output item into the caller's sink:
    {server_label: [{"name","description","input_schema"}, ...]}.
    Informational only (feeds the discovery cache) — never raises.
    """
    try:
        server_label = getattr(item, "server_label", None)
        tools = getattr(item, "tools", None) or []
        if not server_label or not tools:
            return
        normalized = []
        for t in tools:
            if isinstance(t, dict):
                name = t.get("name")
                description = t.get("description")
                schema = t.get("input_schema")
            else:
                name = getattr(t, "name", None)
                description = getattr(t, "description", None)
                schema = getattr(t, "input_schema", None)
            if name:
                normalized.append({"name": name, "description": description,
                                   "input_schema": schema})
        if normalized:
            mcp_tools_sink[server_label] = normalized
    except Exception:
        # Discovery caching must never interfere with response processing
        pass


def _capture_mcp_result(mcp_results_sink, item, server_label):
    """F12: harvest a completed mcp_call's output text into the caller's sink as
    {"tool_name", "output"} (capture order). MCP outputs are external derived artifacts —
    safe to persist, unlike local Slack-fetch/document results. Errored or empty calls are
    skipped, and truncation/budgeting happen later in build_result_digests. Never raises."""
    if mcp_results_sink is None:
        return
    try:
        if getattr(item, "error", None):
            return  # a failed call's "output" isn't a usable result
        output = getattr(item, "output", None)
        if not output:
            return
        mcp_results_sink.append({"tool_name": server_label or "mcp", "output": str(output)})
    except Exception:
        # Result capture must never interfere with response processing
        pass


def _note_container(artifacts_sink, item):
    """F32: record the code-interpreter container so the caller can LIST the files it wrote.

    The container listing is the only artifact source. We deliberately do NOT harvest
    `container_file_citation` annotations: they appear only when the model writes a
    `sandbox:` link (which we forbid — dead in Slack), the listing is a strict superset of
    them anyway, and a citation could name the USER'S OWN mounted attachment, which the
    listing's `source == "assistant"` filter would otherwise have excluded.

    Never raises: losing a container costs files, not the response.
    """
    if artifacts_sink is None:
        return
    try:
        container_id = getattr(item, "container_id", None)
        if container_id:
            artifacts_sink.append({"container_id": container_id})
    except Exception:
        pass


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
    store: bool = False,  # Don't store by default for stateless operation
    prompt_cache_key: Optional[str] = None,
    usage_sink: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a text response using the Responses API
    
    Args:
        messages: List of message dictionaries
        model: Model to use (defaults to config)
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        top_p: Nucleus sampling parameter (not supported by GPT-5 reasoning models)
        system_prompt: System instructions
        reasoning_effort: For GPT-5 models (minimal, low, medium, high)
        verbosity: For GPT-5 models (low, medium, high)
        store: Whether to store the response (default False for stateless)
    
    Returns:
        Generated text response
    """
    model = model or config.gpt_model
    temperature = temperature if temperature is not None else config.default_temperature
    max_tokens = max_tokens or config.default_max_tokens
    top_p = top_p if top_p is not None else config.default_top_p
    
    # Build input for Responses API
    input_messages = []
    
    # Add system prompt if provided
    if system_prompt:
        input_messages.append({
            "role": "developer",
            "content": system_prompt
        })
    
    # Add conversation messages (filter out metadata - Responses API rejects unknown fields)
    for msg in messages:
        # Only include role and content for API
        api_msg = {"role": msg["role"], "content": msg["content"]}
        input_messages.append(api_msg)
    
    # Build request parameters
    request_params = {
        "model": model,
        "input": input_messages,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "store": store,
    }
    
    # Model-specific parameters — all supported models are GPT-5-series reasoning
    # models (gpt-5.5 primary, gpt-5-mini utility)
    # Clamp guards against stored/legacy efforts the model rejects (e.g. `minimal`
    # on 5.6, `max` on 5.5)
    reasoning_effort = clamp_effort(model, reasoning_effort or config.default_reasoning_effort)
    request_params["reasoning"] = {"effort": reasoning_effort}
    verbosity = verbosity or config.default_verbosity
    request_params["text"] = {"verbosity": verbosity}

    # gpt-5.5 and the 5.6 family allow temperature/top_p when reasoning=none
    # (5.6 verified live 2026-07-09: effort=none + temperature/top_p -> 200)
    if (model.startswith("gpt-5.5") or model.startswith("gpt-5.6")) and reasoning_effort == "none":
        request_params["top_p"] = top_p
    else:
        request_params["temperature"] = 1.0  # MUST be 1.0 for reasoning models

    # Prompt caching: gpt-5.5 keeps the explicit 24h retention param; the 5.6 family
    # uses implicit caching (verified live 2026-07-09: second identical call returned
    # cached_tokens>0 with NO cache params; prompt_cache_retention is deprecated on 5.6).
    # The per-thread cache key still helps route repeat calls to the same cache shard.
    if model.startswith("gpt-5.5"):
        request_params["prompt_cache_retention"] = "24h"
    if prompt_cache_key and (model.startswith("gpt-5.5") or model.startswith("gpt-5.6")):
        request_params["prompt_cache_key"] = prompt_cache_key

    self.log_debug(f"Creating text response with model {model}, temp {temperature}")

    try:
        # Determine operation type based on reasoning effort and context
        # All text operations use the same timeout regardless of reasoning level
        operation_type = "text_normal"

        # API call with enforced timeout wrapper
        response = await self._safe_api_call(
            self.client.responses.create,
            operation_type=operation_type,
            **request_params
        )
        
        _capture_usage(usage_sink, response)

        output_text = _join_output_text(response)

        self.log_info(f"Generated response: {len(output_text)} chars")
        return output_text
        
    except Exception as e:
        self.log_error(f"Error creating text response: {e}", exc_info=True)
        raise

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
    function_call_sink: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    prompt_cache_key: Optional[str] = None,
    usage_sink: Optional[Dict[str, Any]] = None,
    mcp_tools_sink: Optional[Dict[str, Any]] = None,
    mcp_results_sink: Optional[List[Dict[str, Any]]] = None,
    artifacts_sink: Optional[List[Dict[str, Any]]] = None,
    container_gone_sink: Optional[List[str]] = None
) -> str:
    """
    Create text response with tools (e.g., web search)

    Args:
        messages: Conversation messages
        tools: List of tools to enable (e.g., [{"type": "web_search"}])
        model: Model to use (defaults to config)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling
        system_prompt: System prompt to use
        reasoning_effort: Reasoning effort for GPT-5 reasoning models
        verbosity: Output verbosity for GPT-5 reasoning models
        store: Whether to store the response
        function_call_sink: Optional list; completed local function_call items
            ({"call_id","name","arguments"}) are appended for the tool loop
        tool_choice: Optional tool_choice override (e.g. "none" when the loop caps out)

    Returns:
        Generated text response
    """
    model = model or config.gpt_model
    temperature = temperature if temperature is not None else config.default_temperature
    max_tokens = max_tokens or config.default_max_tokens
    top_p = top_p if top_p is not None else config.default_top_p

    # Build request parameters. Raw Responses-API items (function_call /
    # function_call_output from the tool loop) carry a "type" and pass through as-is.
    request_params = {
        "model": model,
        "input": [msg if "type" in msg else {"role": msg["role"], "content": msg["content"]}
                  for msg in messages],
        "tools": tools,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "store": store,
    }
    if tool_choice is not None:
        request_params["tool_choice"] = tool_choice
    if function_call_sink is not None:
        # Stateless tool loop: reasoning items must round-trip between rounds, which
        # requires their encrypted content when store=False
        request_params["include"] = ["reasoning.encrypted_content"]

    # Add system prompt if provided
    if system_prompt:
        request_params["instructions"] = system_prompt
    
    # Model-specific parameters — all supported models are GPT-5-series reasoning
    # models (gpt-5.5 primary, gpt-5-mini utility)
    # Clamp guards against stored/legacy efforts the model rejects (e.g. `minimal`
    # on 5.6, `max` on 5.5)
    reasoning_effort = clamp_effort(model, reasoning_effort or config.default_reasoning_effort)
    request_params["reasoning"] = {"effort": reasoning_effort}
    verbosity = verbosity or config.default_verbosity
    request_params["text"] = {"verbosity": verbosity}

    # gpt-5.5 and the 5.6 family allow temperature/top_p when reasoning=none
    # (5.6 verified live 2026-07-09: effort=none + temperature/top_p -> 200)
    if (model.startswith("gpt-5.5") or model.startswith("gpt-5.6")) and reasoning_effort == "none":
        request_params["top_p"] = top_p
    else:
        request_params["temperature"] = 1.0  # MUST be 1.0 for reasoning models

    # Prompt caching: gpt-5.5 keeps the explicit 24h retention param; the 5.6 family
    # uses implicit caching (verified live 2026-07-09: second identical call returned
    # cached_tokens>0 with NO cache params; prompt_cache_retention is deprecated on 5.6).
    # The per-thread cache key still helps route repeat calls to the same cache shard.
    if model.startswith("gpt-5.5"):
        request_params["prompt_cache_retention"] = "24h"
    if prompt_cache_key and (model.startswith("gpt-5.5") or model.startswith("gpt-5.6")):
        request_params["prompt_cache_key"] = prompt_cache_key

    self.log_debug(f"Creating text response with tools using model {model}, tools: {tools}")

    try:
        # Determine operation type based on reasoning effort and context
        # All text operations use the same timeout regardless of reasoning level
        operation_type = "text_normal"

        # API call with enforced timeout wrapper
        response = await _create_with_container_recovery(
            self, request_params, operation_type,
            container_gone_sink=container_gone_sink,
        )

        _capture_usage(usage_sink, response)

        # Text first (seams and all — see _join_output_text); the loop below is tool bookkeeping.
        output_text = _join_output_text(response)
        tools_actually_used = []

        if response.output:
            for item in response.output:
                # Check for tool usage by examining output item types
                item_type = getattr(item, "type", None)
                if item_type == "mcp_call":
                    # Extract MCP server label for attribution
                    server_label = getattr(item, "server_label", None)
                    if server_label and server_label not in tools_actually_used:
                        tools_actually_used.append(server_label)
                    elif not server_label and "mcp" not in tools_actually_used:
                        tools_actually_used.append("mcp")
                    # F12: capture the completed call's output text (MCP results are external
                    # derived artifacts, safe to persist). Skip errored/empty calls.
                    _capture_mcp_result(mcp_results_sink, item, server_label)
                elif item_type == "web_search_call":
                    if "web_search" not in tools_actually_used:
                        tools_actually_used.append("web_search")
                elif item_type == "code_interpreter_call":
                    # F32: the model ran Python in the sandbox. Record the container so the
                    # caller can LIST the files it wrote.
                    #
                    # Why listing and not annotations: a `container_file_citation` annotation
                    # only appears when the model writes a `sandbox:` markdown link to the
                    # file — and we explicitly tell it not to (those links are dead in Slack).
                    # Verified live: prompt says "no links" -> 0 annotations, files still on
                    # disk in the container. The container listing is the source of truth;
                    # annotations are a bonus when the model happens to cite.
                    if "code_interpreter" not in tools_actually_used:
                        tools_actually_used.append("code_interpreter")
                    _note_container(artifacts_sink, item)
                elif item_type == "function_call" and function_call_sink is not None:
                    # Local function call — collected for the tool loop, not part of the text
                    function_call_sink.append({
                        "type": "function_call",
                        "call_id": getattr(item, "call_id", None),
                        "name": getattr(item, "name", None),
                        "arguments": getattr(item, "arguments", None) or "{}",
                    })
                elif item_type == "reasoning" and function_call_sink is not None:
                    # Reasoning items must be replayed with their function_call in the next
                    # round (stateless store=False requires encrypted reasoning round-trip)
                    function_call_sink.append({
                        "type": "reasoning",
                        "item": item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else None,
                    })

                elif item_type == "mcp_list_tools" and mcp_tools_sink is not None:
                    # Tool discovery payload — informational cache (server -> tools)
                    _collect_mcp_list_tools(mcp_tools_sink, item)

        if tools_actually_used:
            self.log_info(f"Generated response with tools: {len(output_text)} chars, used: {', '.join(tools_actually_used)}")
        else:
            self.log_info(f"Generated response with tools: {len(output_text)} chars (no tools invoked)")

        if return_metadata:
            return {"text": output_text, "tools_used": tools_actually_used}
        return output_text
        
    except Exception as e:
        self.log_error(f"Error creating response with tools: {e}", exc_info=True)
        raise

async def create_streaming_response(
    self,
    messages: List[Dict[str, Any]],
    stream_callback: Callable[[str], Any],
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
    usage_sink: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Create a streaming text response using the Responses API
    
    Args:
        messages: List of message dictionaries
        stream_callback: Function to call with text chunks as they arrive
        model: Model to use (defaults to config)
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        top_p: Nucleus sampling parameter (not supported by GPT-5 reasoning models)
        system_prompt: System instructions
        reasoning_effort: For GPT-5 models (minimal, low, medium, high)
        verbosity: For GPT-5 models (low, medium, high)
        store: Whether to store the response (default False for stateless)
        tool_callback: Optional callback for tool events (event_type, status)
    
    Returns:
        Complete generated text response
    """
    model = model or config.gpt_model
    temperature = temperature if temperature is not None else config.default_temperature
    max_tokens = max_tokens or config.default_max_tokens
    top_p = top_p if top_p is not None else config.default_top_p
    
    # Build input for Responses API
    input_messages = []
    
    # Add system prompt if provided
    if system_prompt:
        input_messages.append({
            "role": "developer",
            "content": system_prompt
        })
    
    # Add conversation messages (filter out metadata - Responses API rejects unknown fields)
    for msg in messages:
        # Only include role and content for API
        api_msg = {"role": msg["role"], "content": msg["content"]}
        input_messages.append(api_msg)
    
    # Build request parameters
    request_params = {
        "model": model,
        "input": input_messages,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "store": store,
        "stream": True,  # Enable streaming
    }
    
    # Model-specific parameters — all supported models are GPT-5-series reasoning
    # models (gpt-5.5 primary, gpt-5-mini utility)
    # Clamp guards against stored/legacy efforts the model rejects (e.g. `minimal`
    # on 5.6, `max` on 5.5)
    reasoning_effort = clamp_effort(model, reasoning_effort or config.default_reasoning_effort)
    request_params["reasoning"] = {"effort": reasoning_effort}
    verbosity = verbosity or config.default_verbosity
    request_params["text"] = {"verbosity": verbosity}

    # gpt-5.5 and the 5.6 family allow temperature/top_p when reasoning=none
    # (5.6 verified live 2026-07-09: effort=none + temperature/top_p -> 200)
    if (model.startswith("gpt-5.5") or model.startswith("gpt-5.6")) and reasoning_effort == "none":
        request_params["top_p"] = top_p
    else:
        request_params["temperature"] = 1.0  # MUST be 1.0 for reasoning models

    # Prompt caching: gpt-5.5 keeps the explicit 24h retention param; the 5.6 family
    # uses implicit caching (verified live 2026-07-09: second identical call returned
    # cached_tokens>0 with NO cache params; prompt_cache_retention is deprecated on 5.6).
    # The per-thread cache key still helps route repeat calls to the same cache shard.
    if model.startswith("gpt-5.5"):
        request_params["prompt_cache_retention"] = "24h"
    if prompt_cache_key and (model.startswith("gpt-5.5") or model.startswith("gpt-5.6")):
        request_params["prompt_cache_key"] = prompt_cache_key

    self.log_debug(f"Creating streaming response with model {model}, temp {temperature}")

    try:
        # Determine operation type based on reasoning effort and context
        # All text operations use the same timeout regardless of reasoning level
        operation_type = "text_normal"

        response = await self._safe_api_call(
            self.client.responses.create,
            operation_type=operation_type,
            **request_params
        )

        complete_text = ""
        # The output item the text deltas are currently arriving from. When it changes mid-reply
        # a hosted tool ran in between, and the two halves need a seam (_segment_separator).
        text_item_index: Optional[int] = None
        # A `response.failed` terminal event records its error here; we flush the callback
        # first (below) and raise it after the loop so it propagates like any other API error.
        stream_error: Optional[Exception] = None

        # Process streaming events with timeout protection
        async for event in self._safe_stream_iteration(response, operation_type):
            try:
                
                # Get event type
                event_type = getattr(event, 'type', 'unknown')
                
                if event_type == "response.created":
                    self.log_info("Stream started")
                    continue
                elif event_type == "response.output_item.added":
                    continue  # Skip without logging
                elif event_type in ["response.output_item.delta", "response.output_text.delta"]:
                    # Extract text from delta event
                    text_chunk = None
                    
                    # For response.output_text.delta, the text is directly in event.delta
                    if event_type == "response.output_text.delta" and hasattr(event, 'delta'):
                        text_chunk = event.delta
                    # For response.output_item.delta, need to dig deeper
                    elif hasattr(event, 'delta') and event.delta:
                        if hasattr(event.delta, 'content') and event.delta.content:
                            for content in event.delta.content:
                                if hasattr(content, 'text') and content.text:
                                    text_chunk = content.text
                                    break
                    
                    # If we found text, process it
                    if text_chunk:
                        # New output item after text already streamed = a hosted tool ran in the
                        # middle of the reply. Seam the halves, and send the separator through the
                        # callback too so Slack shows the same text we return.
                        item_index = getattr(event, 'output_index', None)
                        if text_item_index is not None and item_index != text_item_index:
                            text_chunk = _segment_separator(complete_text) + text_chunk
                        text_item_index = item_index
                        complete_text += text_chunk
                        try:
                            result = stream_callback(text_chunk)
                            # If the callback returns a coroutine, await it
                            if hasattr(result, '__await__'):
                                await result
                        except Exception as callback_error:
                            self.log_warning(f"Stream callback error: {callback_error}")
                    continue
                elif event_type == "response.output_item.done":
                    # Extract MCP server_label from completed items for attribution
                    if hasattr(event, 'item'):
                        item = event.item
                        item_type = getattr(item, 'type', None)
                        if item_type == 'mcp_call':
                            server_label = getattr(item, 'server_label', None)
                            tool_error = getattr(item, 'error', None)
                            if tool_error:
                                self.log_warning(f"MCP call error: {tool_error}")
                            if tool_callback and server_label:
                                tool_id = f"mcp:{server_label}"
                                try:
                                    result = tool_callback(tool_id, "completed")
                                    if result and hasattr(result, '__await__'):
                                        await result
                                except Exception as e:
                                    self.log_warning(f"Tool callback error for MCP completion: {e}")
                    continue
                elif event_type in ["response.done", "response.completed",
                                     "response.incomplete", "response.failed"]:
                    resp = getattr(event, "response", None)
                    # Usage rides the terminal event's response object on every outcome, not
                    # just success — capture it so token budgeting doesn't fall back to chars/4.
                    _capture_usage(usage_sink, resp)
                    if event_type == "response.failed":
                        stream_error = _stream_failure_error(resp)
                        self.log_error(
                            f"Stream failed after {len(complete_text)} chars: {stream_error}")
                    elif event_type == "response.incomplete":
                        # Truncated (e.g. max_output_tokens / content filter) but the partial
                        # text is real — return it, exactly as the non-streaming path returns
                        # whatever text a response carries regardless of status.
                        self.log_warning(
                            f"Stream incomplete ({_incomplete_reason(resp)}) after "
                            f"{len(complete_text)} chars")
                    else:
                        self.log_info("Stream completed")
                    # Always signal completion so the callback flushes any buffered text — a
                    # failed/incomplete stream that skips this leaves the buffer stuck forever.
                    try:
                        result = stream_callback(None)
                        # If the callback returns a coroutine, await it
                        if hasattr(result, '__await__'):
                            await result
                    except Exception as callback_error:
                        self.log_warning(f"Stream completion callback error: {callback_error}")
                    break
                elif event_type and ("call" in event_type or "tool" in event_type):
                    # Handle specific tool events
                    if tool_callback:
                        try:
                            result = None
                            if event_type == "response.web_search_call.in_progress":
                                result = tool_callback("web_search", "started")
                            elif event_type == "response.web_search_call.searching":
                                result = tool_callback("web_search", "searching")
                            elif event_type == "response.web_search_call.completed":
                                result = tool_callback("web_search", "completed")
                            elif event_type == "response.file_search_call.in_progress":
                                result = tool_callback("file_search", "started")
                            elif event_type == "response.file_search_call.searching":
                                result = tool_callback("file_search", "searching")
                            elif event_type == "response.file_search_call.completed":
                                result = tool_callback("file_search", "completed")
                            elif event_type == "response.image_generation_call.in_progress":
                                result = tool_callback("image_generation", "started")
                            elif event_type == "response.image_generation_call.generating":
                                result = tool_callback("image_generation", "generating")
                            elif event_type == "response.image_generation_call.completed":
                                result = tool_callback("image_generation", "completed")
                            elif event_type == "response.code_interpreter_call.in_progress":
                                result = tool_callback("code_interpreter", "started")
                            elif event_type == "response.code_interpreter_call.interpreting":
                                result = tool_callback("code_interpreter", "interpreting")
                            elif event_type == "response.code_interpreter_call.completed":
                                result = tool_callback("code_interpreter", "completed")
                            elif event_type == "response.mcp_list_tools.in_progress":
                                result = tool_callback("mcp", "discovering_tools")
                            elif event_type == "response.mcp_list_tools.completed":
                                result = tool_callback("mcp", "tools_discovered")
                            elif event_type == "response.mcp_call.in_progress":
                                # Extract server_label for attribution
                                event_data = getattr(event, "data", event)
                                server_label = getattr(event, "server_label", None) or getattr(event_data, "server_label", None)
                                tool_id = f"mcp:{server_label}" if server_label else "mcp"
                                result = tool_callback(tool_id, "calling")
                            elif event_type == "response.mcp_call.completed":
                                event_data = getattr(event, "data", event)
                                server_label = getattr(event, "server_label", None) or getattr(event_data, "server_label", None)
                                tool_id = f"mcp:{server_label}" if server_label else "mcp"
                                result = tool_callback(tool_id, "completed")

                            # If the tool callback returns a coroutine, await it
                            if result and hasattr(result, '__await__'):
                                await result
                        except Exception as tool_callback_error:
                            self.log_warning(f"Tool callback error: {tool_callback_error}")
                    continue
                else:
                    # Only log unhandled events for debugging
                    pass

            except Exception as event_error:
                self.log_warning(f"Error processing stream event: {event_error}")
                continue

        # A failed terminal event is raised here (outside the per-event try that would have
        # swallowed it) so it propagates like every other API error.
        if stream_error is not None:
            raise stream_error
        self.log_info(f"Generated streaming response: {len(complete_text)} chars")
        return complete_text
        
    except Exception as e:
        self.log_error(f"Error creating streaming response: {e}", exc_info=True)
        raise

async def create_streaming_response_with_tools(
    self,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    stream_callback: Callable[[str], Any],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    system_prompt: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
    store: bool = False,
    tool_callback: Optional[Callable[[str, str], Any]] = None,
    function_call_sink: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    prompt_cache_key: Optional[str] = None,
    usage_sink: Optional[Dict[str, Any]] = None,
    mcp_tools_sink: Optional[Dict[str, Any]] = None,
    mcp_results_sink: Optional[List[Dict[str, Any]]] = None,
    tool_event_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
    artifacts_sink: Optional[List[Dict[str, Any]]] = None,
    container_gone_sink: Optional[List[str]] = None
) -> str:
    """
    Create streaming text response with tools (e.g., web search)

    Args:
        messages: Conversation messages
        tools: List of tools to enable (e.g., [{"type": "web_search"}])
        stream_callback: Function to call with text chunks as they arrive
        model: Model to use (defaults to config)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling
        system_prompt: System prompt to use
        reasoning_effort: Reasoning effort for GPT-5 reasoning models
        verbosity: Output verbosity for GPT-5 reasoning models
        store: Whether to store the response
        tool_callback: Optional callback for tool events (event_type, status)
        function_call_sink: Optional list; completed local function_call items are
            appended for the tool loop. When the round contains function calls, its
            text deltas are suppressed (they're pre-tool preamble, not the answer)
            and the completion flush (stream_callback(None)) is skipped so the loop
            can run another round.
        tool_choice: Optional tool_choice override (e.g. "none" when the loop caps out)

    Returns:
        Complete generated text response
    """
    model = model or config.gpt_model
    temperature = temperature if temperature is not None else config.default_temperature
    max_tokens = max_tokens or config.default_max_tokens
    top_p = top_p if top_p is not None else config.default_top_p

    # Build request parameters. Raw Responses-API items (function_call /
    # function_call_output from the tool loop) carry a "type" and pass through as-is.
    request_params = {
        "model": model,
        "input": [msg if "type" in msg else {"role": msg["role"], "content": msg["content"]}
                  for msg in messages],
        "tools": tools,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "store": store,
        "stream": True,  # Enable streaming
        "parallel_tool_calls": True,  # Allow parallel tool execution
    }
    if tool_choice is not None:
        request_params["tool_choice"] = tool_choice
    if function_call_sink is not None:
        # Stateless tool loop: reasoning items must round-trip between rounds, which
        # requires their encrypted content when store=False
        request_params["include"] = ["reasoning.encrypted_content"]

    # Add system prompt if provided
    if system_prompt:
        request_params["instructions"] = system_prompt
    
    # Model-specific parameters — all supported models are GPT-5-series reasoning
    # models (gpt-5.5 primary, gpt-5-mini utility)
    # Clamp guards against stored/legacy efforts the model rejects (e.g. `minimal`
    # on 5.6, `max` on 5.5)
    reasoning_effort = clamp_effort(model, reasoning_effort or config.default_reasoning_effort)
    request_params["reasoning"] = {"effort": reasoning_effort}
    verbosity = verbosity or config.default_verbosity
    request_params["text"] = {"verbosity": verbosity}

    # gpt-5.5 and the 5.6 family allow temperature/top_p when reasoning=none
    # (5.6 verified live 2026-07-09: effort=none + temperature/top_p -> 200)
    if (model.startswith("gpt-5.5") or model.startswith("gpt-5.6")) and reasoning_effort == "none":
        request_params["top_p"] = top_p
    else:
        request_params["temperature"] = 1.0  # MUST be 1.0 for reasoning models

    # Prompt caching: gpt-5.5 keeps the explicit 24h retention param; the 5.6 family
    # uses implicit caching (verified live 2026-07-09: second identical call returned
    # cached_tokens>0 with NO cache params; prompt_cache_retention is deprecated on 5.6).
    # The per-thread cache key still helps route repeat calls to the same cache shard.
    if model.startswith("gpt-5.5"):
        request_params["prompt_cache_retention"] = "24h"
    if prompt_cache_key and (model.startswith("gpt-5.5") or model.startswith("gpt-5.6")):
        request_params["prompt_cache_key"] = prompt_cache_key

    self.log_debug(f"Creating streaming response with tools using model {model}")

    try:
        # Determine operation type based on reasoning effort and context
        # Determine operation type - all text operations use same timeout regardless of reasoning/tools
        if tools:
            operation_type = "text_with_tools"  # 2.5 minutes
        else:
            operation_type = "text_normal"  # 2.5 minutes

        response = await _create_with_container_recovery(
            self, request_params, operation_type,
            container_gone_sink=container_gone_sink,
        )

        complete_text = ""
        # The output item the text deltas are currently arriving from. When it changes mid-reply
        # a hosted tool ran in between, and the two halves need a seam (_segment_separator).
        text_item_index: Optional[int] = None
        # Tool-loop round state: once a local function_call appears in this round, further
        # text deltas are preamble ("let me check…") — don't stream them to the user.
        saw_function_call = False
        # A `response.failed` terminal event records its error here; it is raised after the
        # loop (a hard failure has no next round) so it propagates like any other API error.
        stream_error: Optional[Exception] = None

        async def _emit_tool_event(payload: Dict[str, Any]) -> None:
            """F30.1: hand a structured server-tool event to an internal observer (the deep
            research card consumes web_search/mcp completions here). Best-effort — observation
            must never break streaming; interactive callers pass no callback."""
            if not tool_event_callback:
                return
            try:
                r = tool_event_callback(payload)
                if r is not None and hasattr(r, "__await__"):
                    await r
            except Exception as e:  # noqa: BLE001
                self.log_warning(f"tool_event_callback error: {e}")

        # Process streaming events with timeout protection
        async for event in self._safe_stream_iteration(response, operation_type):
            try:

                # Get event type
                event_type = getattr(event, 'type', 'unknown')

                if event_type == "response.created":
                    self.log_info("Stream started")
                    continue
                elif event_type == "response.output_item.added":
                    if (function_call_sink is not None and hasattr(event, 'item')
                            and getattr(event.item, 'type', None) == 'function_call'):
                        saw_function_call = True
                    continue  # Skip without logging
                elif event_type in ["response.output_item.delta", "response.output_text.delta"]:
                    # Extract text from delta event
                    text_chunk = None

                    # For response.output_text.delta, the text is directly in event.delta
                    if event_type == "response.output_text.delta" and hasattr(event, 'delta'):
                        text_chunk = event.delta
                    # For response.output_item.delta, need to dig deeper
                    elif hasattr(event, 'delta') and event.delta:
                        if hasattr(event.delta, 'content') and event.delta.content:
                            for content in event.delta.content:
                                if hasattr(content, 'text') and content.text:
                                    text_chunk = content.text
                                    break

                    # If we found text, process it (unless this round is a tool round —
                    # then the text is pre-tool preamble and the loop discards it)
                    if text_chunk and saw_function_call:
                        continue
                    if text_chunk:
                        # New output item after text already streamed = a HOSTED tool (sandbox,
                        # web search) ran mid-reply. Seam the halves, and send the separator
                        # through the callback too so Slack shows the same text we return.
                        item_index = getattr(event, 'output_index', None)
                        if text_item_index is not None and item_index != text_item_index:
                            text_chunk = _segment_separator(complete_text) + text_chunk
                        text_item_index = item_index
                        complete_text += text_chunk
                        try:
                            result = stream_callback(text_chunk)
                            # If the callback returns a coroutine, await it
                            if hasattr(result, '__await__'):
                                await result
                        except Exception as callback_error:
                            self.log_warning(f"Stream callback error: {callback_error}")
                    continue
                elif event_type == "response.output_item.done":
                    # Extract MCP server_label from completed items for attribution
                    if hasattr(event, 'item'):
                        item = event.item
                        item_type = getattr(item, 'type', None)
                        if item_type == 'web_search_call':
                            # F30.1: surface the completed web search (with its query when
                            # available) to an internal observer. This mirrors the
                            # non-streaming path's web_search_call detection, so tools_used
                            # rebuilt from these events matches the create_*_with_tools result.
                            action = getattr(item, 'action', None)
                            query = None
                            if isinstance(action, dict):
                                query = action.get('query')
                            elif action is not None:
                                query = getattr(action, 'query', None)
                            await _emit_tool_event({"kind": "web_search", "query": query})
                        elif item_type == 'mcp_call':
                            server_label = getattr(item, 'server_label', None)
                            tool_error = getattr(item, 'error', None)
                            if tool_error:
                                self.log_warning(f"MCP call error: {tool_error}")
                            # F12: capture the completed call's output text (skips errored/
                            # empty calls internally) for tool-result memory.
                            _capture_mcp_result(mcp_results_sink, item, server_label)
                            # F30.1: surface the completed MCP call to the internal observer.
                            if not tool_error:
                                await _emit_tool_event({"kind": "mcp", "server_label": server_label})
                            if tool_callback and server_label:
                                tool_id = f"mcp:{server_label}"
                                try:
                                    result = tool_callback(tool_id, "completed")
                                    if result and hasattr(result, '__await__'):
                                        await result
                                except Exception as e:
                                    self.log_warning(f"Tool callback error for MCP completion: {e}")
                        elif item_type == 'function_call' and function_call_sink is not None:
                            # Completed local function call — hand to the tool loop
                            saw_function_call = True
                            function_call_sink.append({
                                "type": "function_call",
                                "call_id": getattr(item, 'call_id', None),
                                "name": getattr(item, 'name', None),
                                "arguments": getattr(item, 'arguments', None) or "{}",
                            })
                        elif item_type == 'reasoning' and function_call_sink is not None:
                            # Reasoning items must be replayed with their function_call in
                            # the next round (stateless store=False encrypted round-trip)
                            function_call_sink.append({
                                "type": "reasoning",
                                "item": item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else None,
                            })
                        elif item_type == 'mcp_list_tools' and mcp_tools_sink is not None:
                            # Tool discovery payload — informational cache (server -> tools)
                            _collect_mcp_list_tools(mcp_tools_sink, item)
                        elif item_type == 'code_interpreter_call':
                            # F32: record the container so its files can be listed after the
                            # stream. This — not the annotations below — is what actually
                            # surfaces artifacts, since we tell the model never to write the
                            # `sandbox:` links that would produce a citation.
                            _note_container(artifacts_sink, item)
                    continue
                elif event_type in ["response.done", "response.completed",
                                     "response.incomplete", "response.failed"]:
                    resp = getattr(event, "response", None)
                    # Usage rides the terminal event's response object on every outcome, not
                    # just success — capture it so token budgeting doesn't fall back to chars/4.
                    _capture_usage(usage_sink, resp)
                    if event_type == "response.failed":
                        stream_error = _stream_failure_error(resp)
                        self.log_error(
                            f"Stream failed after {len(complete_text)} chars: {stream_error}")
                    elif event_type == "response.incomplete":
                        self.log_warning(
                            f"Stream incomplete ({_incomplete_reason(resp)}) after "
                            f"{len(complete_text)} chars")
                    else:
                        self.log_info("Stream completed")
                    # Only a NORMAL completion with local function calls defers the flush — the
                    # tool loop will run another round, so the buffered text isn't final yet.
                    # An incomplete or failed terminal has no next round, so it must still flush
                    # (and, for failed, raise) even when a function call was seen this round —
                    # otherwise the buffer stays stuck. (Keyed on actual function calls;
                    # reasoning-only sink entries must not suppress the final flush.)
                    if saw_function_call and event_type in ("response.done", "response.completed"):
                        break
                    # Signal the callback that streaming is complete with None so it flushes
                    # any buffered text — a failed/incomplete stream that skips this leaves the
                    # buffer stuck forever.
                    try:
                        result = stream_callback(None)
                        # If the callback returns a coroutine, await it
                        if hasattr(result, '__await__'):
                            await result
                    except Exception as callback_error:
                        self.log_warning(f"Stream completion callback error: {callback_error}")
                    break
                elif event_type and ("call" in event_type or "tool" in event_type):
                    # Handle specific tool events
                    if tool_callback:
                        try:
                            result = None
                            if event_type == "response.web_search_call.in_progress":
                                result = tool_callback("web_search", "started")
                            elif event_type == "response.web_search_call.searching":
                                result = tool_callback("web_search", "searching")
                            elif event_type == "response.web_search_call.completed":
                                result = tool_callback("web_search", "completed")
                            elif event_type == "response.file_search_call.in_progress":
                                result = tool_callback("file_search", "started")
                            elif event_type == "response.file_search_call.searching":
                                result = tool_callback("file_search", "searching")
                            elif event_type == "response.file_search_call.completed":
                                result = tool_callback("file_search", "completed")
                            elif event_type == "response.image_generation_call.in_progress":
                                result = tool_callback("image_generation", "started")
                            elif event_type == "response.image_generation_call.generating":
                                result = tool_callback("image_generation", "generating")
                            elif event_type == "response.image_generation_call.completed":
                                result = tool_callback("image_generation", "completed")
                            elif event_type == "response.code_interpreter_call.in_progress":
                                result = tool_callback("code_interpreter", "started")
                            elif event_type == "response.code_interpreter_call.interpreting":
                                result = tool_callback("code_interpreter", "interpreting")
                            elif event_type == "response.code_interpreter_call.completed":
                                result = tool_callback("code_interpreter", "completed")
                            elif event_type == "response.mcp_list_tools.in_progress":
                                result = tool_callback("mcp", "discovering_tools")
                            elif event_type == "response.mcp_list_tools.completed":
                                result = tool_callback("mcp", "tools_discovered")
                            elif event_type == "response.mcp_call.in_progress":
                                # Extract server_label for attribution
                                event_data = getattr(event, "data", event)
                                server_label = getattr(event, "server_label", None) or getattr(event_data, "server_label", None)
                                tool_id = f"mcp:{server_label}" if server_label else "mcp"
                                result = tool_callback(tool_id, "calling")
                            elif event_type == "response.mcp_call.completed":
                                event_data = getattr(event, "data", event)
                                server_label = getattr(event, "server_label", None) or getattr(event_data, "server_label", None)
                                tool_id = f"mcp:{server_label}" if server_label else "mcp"
                                result = tool_callback(tool_id, "completed")

                            # If the tool callback returns a coroutine, await it
                            if result and hasattr(result, '__await__'):
                                await result
                        except Exception as tool_callback_error:
                            self.log_warning(f"Tool callback error: {tool_callback_error}")
                    continue
                else:
                    # Only log unhandled events for debugging
                    pass
                    
            except Exception as event_error:
                self.log_warning(f"Error processing stream event: {event_error}")
                continue

        # A failed terminal event is raised here (outside the per-event try that would have
        # swallowed it) so it reaches the outer handler and propagates like any other error.
        if stream_error is not None:
            raise stream_error
        self.log_info(f"Generated streaming response with tools: {len(complete_text)} chars")
        return complete_text
        
    except asyncio.TimeoutError as e:
        # Log timeout as warning without stack trace
        self.log_warning(f"Streaming response with tools timed out: {e}")
        raise
    except Exception as e:
        # Check if this is an MCP connection error (expected failure, handled gracefully)
        error_msg = str(e)
        is_mcp_error = "mcp server" in error_msg.lower() and ("404" in error_msg or "424" in error_msg)

        if is_mcp_error:
            # MCP errors are handled gracefully by retry logic - log as WARNING without stack trace
            self.log_warning(f"MCP connection failed during streaming (will retry without failed server): {error_msg}")
        elif is_container_gone(e):
            # Handled upstream (the binding is dropped and the turn re-runs without it), same as
            # the MCP case — a recovered turn must not leave a crash-shaped traceback behind.
            self.log_warning(
                f"Code-interpreter container expired during streaming (will retry without it): {error_msg}")
        else:
            # Unexpected errors - log as ERROR with stack trace
            self.log_error(f"Error creating streaming response with tools: {e}", exc_info=True)
        raise

async def classify_wake(self, *, sources: Any,
                        channel_steering_text: Optional[str] = None) -> Optional[bool]:
    """THE gate call: one utility-model request, one boolean out.

    Returns True/False as the model decided, or **None** when it produced nothing usable — an API
    exception, a refusal, a truncated response, or a payload without a real boolean. None is not a
    decision and the engine must not treat it as one: it becomes a `classifier_error` decline and
    a terminal `none`, so an outage at the provider is never scored as the model choosing silence.

    `sources` are the ordered SourceMessage records of this debounce cohort, oldest first.
    `channel_steering_text` is the canonical steering block from channel_steering.py, inserted
    VERBATIM — the responder's copy of this turn is the same string, byte for byte, and that
    identity is the invariant commit 5 exists to protect. Nothing else goes in. There is no pulse
    envelope, thread tail, people line, topic, canvas list, summary, capability inventory, emoji
    palette, strictness, or image here, because the gate no longer decides anything those inputs
    would inform.

    Structured Outputs with a strict schema, rather than "reply with JSON and we'll fish the
    object out of the prose". The old rich classifier parsed the first {...} it could find, which
    meant a truncated reply could still yield an action field with none of the checks around it.
    A boolean is exactly the kind of output a schema can guarantee.
    """
    blocks = [_render_wake_source(s, index=i, total=len(sources))
              for i, s in enumerate(sources)]
    prompt = "Messages to decide about, oldest first:\n\n" + "\n\n".join(blocks)
    steering = (channel_steering_text or "").strip()
    if steering:
        # Verbatim, as its own labelled section. Nothing may re-render or reorder this block: the
        # responder inserts the identical string, and a difference here would mean the two halves
        # of one turn obeyed different rules while each looked correct.
        prompt += ("\n\nWhat this channel has established (verbatim; instructions and background "
                   "are labelled):\n" + steering)

    request_params = {
        "model": config.utility_model,
        "input": [
            {"role": "developer", "content": WAKE_CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        # Reasoning tokens bill against this cap, so the floor has to cover the THINKING and not
        # just the answer. A live rich verdict once came back empty at `medium` effort because
        # reasoning had eaten the whole budget, and the fail-safe silently swallowed it — the gate
        # never actually ran. The output itself is now one boolean, but the reasoning is not
        # smaller for that. Unused tokens are not billed. Same policy as before; no new number.
        "max_output_tokens": max(2048, config.utility_max_tokens),
        "store": False,
        # A strict schema, so "did the model answer the question" stops being a parsing question.
        "text": {
            "format": {
                "type": "json_schema",
                "name": "wake_decision",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"wake": {"type": "boolean"}},
                    "required": ["wake"],
                    "additionalProperties": False,
                },
            },
        },
    }
    # 5.6-family hybrid rule: temperature is only legal at effort `none`, and this call reasons.
    request_params["temperature"] = 1.0
    # The gate keeps its own (higher) effort. Deciding whether a turn could be useful at all still
    # needs inference, and `none` collapses it into pattern-matching the last sentence.
    request_params["reasoning"] = {
        "effort": clamp_effort(config.utility_model, config.participation_reasoning_effort)}

    try:
        response = await self._safe_api_call(
            self.client.responses.create, operation_type="utility_call", **request_params)
        # An INCOMPLETE response can still carry parseable text — the model got partway through
        # and the budget ran out — and that text can even contain a valid-looking object. It is
        # not an answer: the run was cut off, so whatever is there is whatever had been emitted
        # when the lights went out. Treated as no decision, like a refusal or an exception.
        status = getattr(response, "status", None)
        if status and str(status) != "completed":
            reason = getattr(getattr(response, "incomplete_details", None), "reason", None)
            self.log_warning(
                f"Wake classifier response was {status}"
                f"{f' ({reason})' if reason else ''}; the engine will decline this attempt")
            return None
        out = ""
        for item in (response.output or []):
            # A refusal arrives as its own content part with no `text`, so it contributes nothing
            # and falls through to the no-boolean branch below — which is the correct outcome:
            # a refusal is not a decision either.
            for content in (getattr(item, "content", None) or []):
                if hasattr(content, "text") and content.text:
                    out += content.text
        wake = _parse_wake(out)
        if wake is None:
            self.log_warning(
                "Wake classifier produced no usable boolean; the engine will decline this attempt")
            return None
        self.log_debug(f"Wake decision: {wake} over {len(sources)} source message(s)")
        return wake
    except Exception as e:
        self.log_warning(f"Wake classification failed ({e}); the engine will decline this attempt")
        return None


def _parse_wake(raw: str) -> Optional[bool]:
    """The strict schema's payload, or None.

    Only a real JSON boolean counts. A string "true", a 1, or a missing key are all None rather
    than coerced: with a strict schema in force, anything else means the response is not the
    response we asked for, and guessing what it meant is how a gate starts waking on noise."""
    text = (raw or "").strip()
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        payload = json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return None
    wake = payload.get("wake") if isinstance(payload, dict) else None
    return wake if isinstance(wake, bool) else None


def _render_wake_source(source: Any, *, index: int, total: int) -> str:
    """One source message as labelled plain text for the gate prompt.

    Typed record in, flat block out — the gate sees WHO said it, where in the thread, and what was
    attached by name. Attachment names only: this gate never looks at pixels, and a description
    written by some other model is a claim about content it cannot check."""
    who = source.sender_name or source.sender_id or "someone"
    if source.sender_type in ("self", "other_bot"):
        who += " (a bot)"
    header = f"[{index + 1} of {total}] {who}"
    # WHEN, as well as who and where. This matters more now than it did to the rich gate: the
    # cohort has no freshness window any more (that window was a silent message-loss path), so a
    # cohort can legitimately hold a send from much earlier, and without the times the gate cannot
    # tell a three-second fast-follow — one thought split across sends — from a straggler that has
    # been sitting there for twenty minutes. Rendered through the shared pure helper, in UTC: the
    # gate does no per-sender timezone lookup, and elapsed time is what it needs, not local wall
    # clock. An unparseable ts renders as "" and is simply omitted.
    # Function-local: message_processor imports this package, so a module-level import here is a
    # cycle. The helper itself is pure (no config, no I/O) — see message_timestamps.py.
    from message_processor.message_timestamps import render_message_timestamp
    stamp = render_message_timestamp(source.ts, "UTC")
    if stamp:
        header += f" {stamp}"
    header += " — a reply inside a thread" if source.is_thread_reply else " — posted to the channel"
    lines = [header, (source.text or "").strip() or "(no text)"]
    if source.attachments:
        lines.append("Attached (names and types only, contents not shown to you): "
                     + ", ".join(source.attachments))
    edit = source.edit or {}
    if edit:
        old = str(edit.get("old_text") or "").strip()
        lines.append(
            f'This message was EDITED after it was posted. Before the edit it read: "{old}".'
            if old else "This message was EDITED after it was posted; it had no text before.")
        lines.append("The assistant already replied to it — wake it only if the edit changes what "
                     "is being asked." if edit.get("already_replied")
                     else "The assistant has not replied to it yet.")
    return "\n".join(lines)


async def extract_memory(self, exchange_text: str, existing_memory: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Post-response memory extraction (Phase 9). Given the latest exchange + current channel
    memory, decide whether to record a durable fact. Returns a dict:
        {"action": "none"} | {"action": "add", "content": str} | {"action": "update", "id": int, "content": str}
    Best-effort and CONSERVATIVE: any failure / unparseable output → {"action": "none"} (never write)."""
    existing_memory = existing_memory or []
    mem_lines = "\n".join(f"{m['id']}. {m['content']}" for m in existing_memory) or "(empty)"

    conversation_messages = [
        {"role": "developer", "content": MEMORY_EXTRACTION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Current memory:\n{mem_lines}\n\nLatest exchange:\n{exchange_text}\n\nRespond with ONLY the JSON object."},
    ]

    # Memory extraction emits a small JSON object (and reasoning models spend tokens before output),
    # so give it more room than the one-word wake classifier's tiny utility_max_tokens.
    request_params = {
        "model": config.utility_model,
        "input": conversation_messages,
        "max_output_tokens": max(1024, config.utility_max_tokens),
        "store": False,
    }
    # Utility model is a GPT-5-series reasoning model (gpt-5-mini)
    request_params["temperature"] = 1.0
    request_params["reasoning"] = {"effort": clamp_effort(config.utility_model, config.utility_reasoning_effort)}
    request_params["text"] = {"verbosity": config.utility_verbosity}

    try:
        response = await self._safe_api_call(
            self.client.responses.create,
            operation_type="utility_call",
            **request_params,
        )
        result = ""
        if response.output:
            for item in response.output:
                if hasattr(item, "content") and item.content:
                    for content in item.content:
                        if hasattr(content, "text"):
                            result += content.text
        result = result.strip()
        # Extract the JSON object defensively (model may wrap it in prose/fences).
        start, end = result.find("{"), result.rfind("}")
        if start == -1 or end == -1 or end < start:
            return {"action": "none"}
        parsed = json.loads(result[start:end + 1])
        action = str(parsed.get("action", "none")).lower()
        if action == "add" and parsed.get("content"):
            return {"action": "add", "content": str(parsed["content"]).strip()}
        if action == "update" and parsed.get("id") is not None and parsed.get("content"):
            return {"action": "update", "id": int(parsed["id"]), "content": str(parsed["content"]).strip()}
        return {"action": "none"}
    except Exception as e:
        self.log_warning(f"Memory extraction failed ({e}); skipping write")
        return {"action": "none"}


async def summarize_tool_result(self, text: str, max_chars: int) -> Optional[str]:
    """F16: compress ONE overlong MCP tool output to a single line under ``max_chars``,
    preserving URLs/titles/dates/figures/IDs verbatim (utility model, low effort).

    Best-effort and NON-BLOCKING for the reply pipeline: returns the summary string, or
    ``None`` on any error/timeout/empty output so the caller falls back to today's
    truncation. Never raises. The caller applies the input-char budget guard before
    calling, so ``text`` is already bounded."""
    conversation_messages = [
        {"role": "developer", "content": TOOL_RESULT_SUMMARIZE_PROMPT.format(max_chars=max_chars)},
        {"role": "user", "content": f"Tool output:\n{text}\n\nRespond with ONLY the single-line summary."},
    ]

    request_params = {
        "model": config.utility_model,
        "input": conversation_messages,
        "max_output_tokens": max(1024, config.utility_max_tokens),
        "store": False,
    }
    # Utility model is a GPT-5-series reasoning model; temperature fixed to 1.0. Low effort
    # per F16 — enough to summarize while preserving verbatim spans, without burning latency.
    request_params["temperature"] = 1.0
    request_params["reasoning"] = {"effort": clamp_effort(config.utility_model, "low")}
    request_params["text"] = {"verbosity": config.utility_verbosity}

    try:
        response = await self._safe_api_call(
            self.client.responses.create,
            operation_type="utility_call",
            **request_params,
        )
        result = ""
        if response.output:
            for item in response.output:
                if hasattr(item, "content") and item.content:
                    for content in item.content:
                        if hasattr(content, "text"):
                            result += content.text
        result = result.strip()
        return result or None
    except Exception as e:
        self.log_warning(f"Tool-result summarization failed ({e}); falling back to truncation")
        return None


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
    """
    Create a text response with custom timeout (for retry scenarios)

    Args:
        messages: List of message dictionaries
        model: Model to use (defaults to config)
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        top_p: Nucleus sampling parameter (not supported by GPT-5 reasoning models)
        system_prompt: System instructions
        reasoning_effort: For GPT-5 models (minimal, low, medium, high)
        verbosity: For GPT-5 models (low, medium, high)
        store: Whether to store the response (default False for stateless)
        timeout_seconds: Custom timeout for the API call

    Returns:
        Generated text response
    """
    model = model or config.gpt_model
    temperature = temperature if temperature is not None else config.default_temperature
    max_tokens = max_tokens or config.default_max_tokens
    top_p = top_p if top_p is not None else config.default_top_p

    # Build input for Responses API
    input_messages = []

    # Add system prompt if provided
    if system_prompt:
        input_messages.append({
            "role": "developer",
            "content": system_prompt
        })

    # Add conversation messages (filter out metadata - Responses API rejects unknown fields)
    for msg in messages:
        # Only include role and content for API
        api_msg = {"role": msg["role"], "content": msg["content"]}
        input_messages.append(api_msg)

    # Build request parameters
    request_params = {
        "model": model,
        "input": input_messages,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "store": store,
    }

    # Model-specific parameters — all supported models are GPT-5-series reasoning
    # models (gpt-5.5 primary, gpt-5-mini utility)
    # Clamp guards against stored/legacy efforts the model rejects (e.g. `minimal`
    # on 5.6, `max` on 5.5)
    reasoning_effort = clamp_effort(model, reasoning_effort or config.default_reasoning_effort)
    request_params["reasoning"] = {"effort": reasoning_effort}
    verbosity = verbosity or config.default_verbosity
    request_params["text"] = {"verbosity": verbosity}

    # gpt-5.5 and the 5.6 family allow temperature/top_p when reasoning=none
    # (5.6 verified live 2026-07-09: effort=none + temperature/top_p -> 200)
    if (model.startswith("gpt-5.5") or model.startswith("gpt-5.6")) and reasoning_effort == "none":
        request_params["top_p"] = top_p
    else:
        request_params["temperature"] = 1.0  # MUST be 1.0 for reasoning models

    self.log_debug(f"Creating text response with custom timeout {timeout_seconds}s, model {model}")

    try:
        # Determine operation type based on reasoning effort and context
        # All text operations use the same timeout regardless of reasoning level
        operation_type = "text_normal"

        # API call with custom timeout
        response = await self._safe_api_call(
            self.client.responses.create,
            operation_type=operation_type,
            timeout_seconds=timeout_seconds,
            **request_params
        )

        output_text = _join_output_text(response)

        self.log_info(f"Generated response with custom timeout: {len(output_text)} chars")
        return output_text

    except asyncio.TimeoutError as e:
        # Log timeout as warning without stack trace
        self.log_warning(f"Text response timed out: {e}")
        raise
    except Exception as e:
        self.log_error(f"Error creating text response with timeout: {e}", exc_info=True)
        raise

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
    function_call_sink: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
    prompt_cache_key: Optional[str] = None,
    usage_sink: Optional[Dict[str, Any]] = None,
    mcp_tools_sink: Optional[Dict[str, Any]] = None,
    mcp_results_sink: Optional[List[Dict[str, Any]]] = None,
    artifacts_sink: Optional[List[Dict[str, Any]]] = None,
    container_gone_sink: Optional[List[str]] = None
) -> str:
    """
    Create text response with tools and custom timeout (for retry scenarios)

    Args:
        messages: Conversation messages
        tools: List of tools to enable (e.g., [{"type": "web_search"}])
        model: Model to use (defaults to config)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling
        system_prompt: System prompt to use
        reasoning_effort: Reasoning effort for GPT-5 reasoning models
        verbosity: Output verbosity for GPT-5 reasoning models
        store: Whether to store the response
        timeout_seconds: Custom timeout for the API call

    Returns:
        Generated text response
    """
    model = model or config.gpt_model
    temperature = temperature if temperature is not None else config.default_temperature
    max_tokens = max_tokens or config.default_max_tokens
    top_p = top_p if top_p is not None else config.default_top_p

    # Build request parameters. Raw Responses-API items (function_call /
    # function_call_output from the tool loop) carry a "type" and pass through as-is.
    request_params = {
        "model": model,
        "input": [msg if "type" in msg else {"role": msg["role"], "content": msg["content"]}
                  for msg in messages],
        "tools": tools,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "store": store,
    }
    if tool_choice is not None:
        request_params["tool_choice"] = tool_choice
    if function_call_sink is not None:
        # Stateless tool loop: reasoning items must round-trip between rounds, which
        # requires their encrypted content when store=False
        request_params["include"] = ["reasoning.encrypted_content"]

    # Add system prompt if provided
    if system_prompt:
        request_params["instructions"] = system_prompt

    # Model-specific parameters — all supported models are GPT-5-series reasoning
    # models (gpt-5.5 primary, gpt-5-mini utility)
    # Clamp guards against stored/legacy efforts the model rejects (e.g. `minimal`
    # on 5.6, `max` on 5.5)
    reasoning_effort = clamp_effort(model, reasoning_effort or config.default_reasoning_effort)
    request_params["reasoning"] = {"effort": reasoning_effort}
    verbosity = verbosity or config.default_verbosity
    request_params["text"] = {"verbosity": verbosity}

    # gpt-5.5 and the 5.6 family allow temperature/top_p when reasoning=none
    # (5.6 verified live 2026-07-09: effort=none + temperature/top_p -> 200)
    if (model.startswith("gpt-5.5") or model.startswith("gpt-5.6")) and reasoning_effort == "none":
        request_params["top_p"] = top_p
    else:
        request_params["temperature"] = 1.0  # MUST be 1.0 for reasoning models

    # Prompt caching — parity with the non-timeout twin. gpt-5.5 keeps the explicit 24h
    # retention param; the 5.6 family uses implicit caching, and the per-thread cache key still
    # routes repeat calls to the same cache shard.
    if model.startswith("gpt-5.5"):
        request_params["prompt_cache_retention"] = "24h"
    if prompt_cache_key and (model.startswith("gpt-5.5") or model.startswith("gpt-5.6")):
        request_params["prompt_cache_key"] = prompt_cache_key

    self.log_debug(f"Creating text response with tools and custom timeout {timeout_seconds}s, model {model}, tools: {tools}")

    try:
        # Determine operation type based on reasoning effort and context
        # All text operations use the same timeout regardless of reasoning level
        operation_type = "text_normal"

        # API call with custom timeout
        response = await _create_with_container_recovery(
            self, request_params, operation_type,
            container_gone_sink=container_gone_sink,
            timeout_seconds=timeout_seconds,
        )

        # Usage-driven context budgeting must not degrade on the retry path — parity with the
        # non-timeout twin. Without this, a retried turn silently falls back to chars/4.
        _capture_usage(usage_sink, response)

        # Text first (seams and all — see _join_output_text); the loop below is tool bookkeeping.
        output_text = _join_output_text(response)
        tools_actually_used = []

        if response.output:
            for item in response.output:
                # Check for tool usage by examining output item types
                item_type = getattr(item, "type", None)
                if item_type == "mcp_call":
                    # Extract MCP server label for attribution
                    server_label = getattr(item, "server_label", None)
                    if server_label and server_label not in tools_actually_used:
                        tools_actually_used.append(server_label)
                    elif not server_label and "mcp" not in tools_actually_used:
                        tools_actually_used.append("mcp")
                    # F12: capture the completed call's output text (MCP results are external
                    # derived artifacts, safe to persist). Skip errored/empty calls. Parity with
                    # the non-timeout twin — without this, a retry silently drops tool-result
                    # memory.
                    _capture_mcp_result(mcp_results_sink, item, server_label)
                elif item_type == "web_search_call":
                    if "web_search" not in tools_actually_used:
                        tools_actually_used.append("web_search")
                elif item_type == "code_interpreter_call":
                    # F32: the model ran Python in the sandbox. Record the container so the
                    # caller can LIST the files it wrote.
                    #
                    # Why listing and not annotations: a `container_file_citation` annotation
                    # only appears when the model writes a `sandbox:` markdown link to the
                    # file — and we explicitly tell it not to (those links are dead in Slack).
                    # Verified live: prompt says "no links" -> 0 annotations, files still on
                    # disk in the container. The container listing is the source of truth;
                    # annotations are a bonus when the model happens to cite.
                    if "code_interpreter" not in tools_actually_used:
                        tools_actually_used.append("code_interpreter")
                    _note_container(artifacts_sink, item)
                elif item_type == "function_call" and function_call_sink is not None:
                    # Local function call — collected for the tool loop, not part of the text
                    function_call_sink.append({
                        "type": "function_call",
                        "call_id": getattr(item, "call_id", None),
                        "name": getattr(item, "name", None),
                        "arguments": getattr(item, "arguments", None) or "{}",
                    })
                elif item_type == "reasoning" and function_call_sink is not None:
                    # Reasoning items must be replayed with their function_call in the next
                    # round (stateless store=False requires encrypted reasoning round-trip)
                    function_call_sink.append({
                        "type": "reasoning",
                        "item": item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else None,
                    })
                elif item_type == "mcp_list_tools" and mcp_tools_sink is not None:
                    # Tool discovery payload — informational cache (server -> tools). Parity with
                    # the non-timeout twin — without this, a retry silently drops discovery.
                    _collect_mcp_list_tools(mcp_tools_sink, item)

        if tools_actually_used:
            self.log_info(f"Generated response with tools and custom timeout: {len(output_text)} chars, used: {', '.join(tools_actually_used)}")
        else:
            self.log_info(f"Generated response with tools and custom timeout: {len(output_text)} chars (no tools invoked)")

        if return_metadata:
            return {"text": output_text, "tools_used": tools_actually_used}
        return output_text

    except asyncio.TimeoutError as e:
        # Log timeout as warning without stack trace
        self.log_warning(f"Response with tools timed out: {e}")
        raise
    except Exception as e:
        self.log_error(f"Error creating response with tools and timeout: {e}", exc_info=True)
        raise
