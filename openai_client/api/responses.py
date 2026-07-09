from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable, Dict, List, Optional

from config import config
from prompts import IMAGE_INTENT_SYSTEM_PROMPT, MEMORY_EXTRACTION_SYSTEM_PROMPT, WAKE_CLASSIFIER_SYSTEM_PROMPT


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
    reasoning_effort = reasoning_effort or config.default_reasoning_effort
    request_params["reasoning"] = {"effort": reasoning_effort}
    verbosity = verbosity or config.default_verbosity
    request_params["text"] = {"verbosity": verbosity}

    # gpt-5.5 allows temperature/top_p when reasoning=none
    if model.startswith("gpt-5.5") and reasoning_effort == "none":
        request_params["top_p"] = top_p
    else:
        request_params["temperature"] = 1.0  # MUST be 1.0 for reasoning models

    # Prompt caching (gpt-5.5)
    if model.startswith("gpt-5.5"):
        request_params["prompt_cache_retention"] = "24h"

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
        
        # Extract text from response
        output_text = ""
        if response.output:
            for item in response.output:
                if hasattr(item, "content") and item.content:
                    for content in item.content:
                        if hasattr(content, "text"):
                            output_text += content.text

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
    tool_choice: Optional[str] = None
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
    reasoning_effort = reasoning_effort or config.default_reasoning_effort
    request_params["reasoning"] = {"effort": reasoning_effort}
    verbosity = verbosity or config.default_verbosity
    request_params["text"] = {"verbosity": verbosity}

    # gpt-5.5 allows temperature/top_p when reasoning=none
    if model.startswith("gpt-5.5") and reasoning_effort == "none":
        request_params["top_p"] = top_p
    else:
        request_params["temperature"] = 1.0  # MUST be 1.0 for reasoning models

    self.log_debug(f"Creating text response with tools using model {model}, tools: {tools}")

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
        
        # Extract text from response and detect tool usage
        output_text = ""
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
                elif item_type == "web_search_call":
                    if "web_search" not in tools_actually_used:
                        tools_actually_used.append("web_search")
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

                # Extract text content
                if hasattr(item, "content") and item.content:
                    for content in item.content:
                        if hasattr(content, "text"):
                            output_text += content.text

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
    reasoning_effort = reasoning_effort or config.default_reasoning_effort
    request_params["reasoning"] = {"effort": reasoning_effort}
    verbosity = verbosity or config.default_verbosity
    request_params["text"] = {"verbosity": verbosity}

    # gpt-5.5 allows temperature/top_p when reasoning=none
    if model.startswith("gpt-5.5") and reasoning_effort == "none":
        request_params["top_p"] = top_p
    else:
        request_params["temperature"] = 1.0  # MUST be 1.0 for reasoning models

    # Prompt caching (gpt-5.5)
    if model.startswith("gpt-5.5"):
        request_params["prompt_cache_retention"] = "24h"

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
                elif event_type in ["response.done", "response.completed"]:
                    self.log_info("Stream completed")
                    # Signal the callback that streaming is complete with None
                    # This allows it to flush any remaining buffered text
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
    tool_choice: Optional[str] = None
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
    reasoning_effort = reasoning_effort or config.default_reasoning_effort
    request_params["reasoning"] = {"effort": reasoning_effort}
    verbosity = verbosity or config.default_verbosity
    request_params["text"] = {"verbosity": verbosity}

    # gpt-5.5 allows temperature/top_p when reasoning=none
    if model.startswith("gpt-5.5") and reasoning_effort == "none":
        request_params["top_p"] = top_p
    else:
        request_params["temperature"] = 1.0  # MUST be 1.0 for reasoning models

    self.log_debug(f"Creating streaming response with tools using model {model}")

    try:
        # Determine operation type based on reasoning effort and context
        # Determine operation type - all text operations use same timeout regardless of reasoning/tools
        if tools:
            operation_type = "text_with_tools"  # 2.5 minutes
        else:
            operation_type = "text_normal"  # 2.5 minutes

        response = await self._safe_api_call(
            self.client.responses.create,
            operation_type=operation_type,
            **request_params
        )

        complete_text = ""
        # Tool-loop round state: once a local function_call appears in this round, further
        # text deltas are preamble ("let me check…") — don't stream them to the user.
        saw_function_call = False

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
                    continue
                elif event_type in ["response.done", "response.completed"]:
                    self.log_info("Stream completed")
                    # When the round produced local function calls, the tool loop will run
                    # another round — don't signal completion to the buffer yet.
                    # (Keyed on actual function calls; reasoning-only sink entries must
                    # not suppress the final flush.)
                    if saw_function_call:
                        break
                    # Signal the callback that streaming is complete with None
                    # This allows it to flush any remaining buffered text
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
        else:
            # Unexpected errors - log as ERROR with stack trace
            self.log_error(f"Error creating streaming response with tools: {e}", exc_info=True)
        raise

async def classify_wake(self, text: str, signals: Optional[Dict[str, Any]] = None) -> str:
    """Lightweight 'should the bot respond?' classifier for channel auto_respond mode.

    Returns 'respond' | 'react' | 'ignore'. Best-effort and CONSERVATIVE: any failure or
    unrecognized output defaults to 'ignore' (never spam a channel)."""
    signals = signals or {}
    signal_lines = []
    if signals.get("is_thread_reply"):
        signal_lines.append("- This is a reply inside a thread the assistant is part of.")
    if signals.get("directives"):
        signal_lines.append(f"- Operator-set ground rules for this channel (honor them): {signals['directives']}")
    signal_note = ("\n\nSignals:\n" + "\n".join(signal_lines)) if signal_lines else ""

    conversation_messages = [
        {"role": "developer", "content": WAKE_CLASSIFIER_SYSTEM_PROMPT},
        {"role": "user", "content": f"Message:\n{text}{signal_note}\n\nRespond with ONLY one word: respond, react, or ignore."},
    ]

    request_params = {
        "model": config.utility_model,
        "input": conversation_messages,
        "max_output_tokens": config.utility_max_tokens,
        "store": False,
    }
    # Utility model is a GPT-5-series reasoning model (gpt-5-mini)
    request_params["temperature"] = 1.0
    request_params["reasoning"] = {"effort": config.utility_reasoning_effort}
    request_params["text"] = {"verbosity": config.utility_verbosity}

    try:
        response = await self._safe_api_call(
            self.client.responses.create,
            operation_type="intent_classification",
            **request_params,
        )
        result = ""
        if response.output:
            for item in response.output:
                if hasattr(item, "content") and item.content:
                    for content in item.content:
                        if hasattr(content, "text"):
                            result += content.text
        result = result.strip().lower()
        self.log_debug(f"Wake classifier raw result: '{result}' for: '{text[:60]}...'")
        if "respond" in result:
            return "respond"
        if "react" in result:
            return "react"
        return "ignore"
    except Exception as e:
        self.log_warning(f"Wake classification failed ({e}); defaulting to ignore")
        return "ignore"


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
        "max_output_tokens": max(512, config.utility_max_tokens),
        "store": False,
    }
    # Utility model is a GPT-5-series reasoning model (gpt-5-mini)
    request_params["temperature"] = 1.0
    request_params["reasoning"] = {"effort": config.utility_reasoning_effort}
    request_params["text"] = {"verbosity": config.utility_verbosity}

    try:
        response = await self._safe_api_call(
            self.client.responses.create,
            operation_type="intent_classification",
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


async def classify_intent(
    self,
    messages: List[Dict[str, Any]],
    last_user_message: str,
    has_attached_images: bool = False,
    max_retries: int = 2
) -> str:
    """
    Classify user intent using a lightweight model with retry logic

    Args:
        messages: Recent conversation context (last 6-8 exchanges)
        last_user_message: The latest user message to classify
        has_attached_images: Whether the current message has images attached
        max_retries: Number of retry attempts on timeout (default: 2)

    Returns:
        Intent classification: 'new_image', 'modify_image', or 'text_only'
        Returns 'error' if classification fails after retries
    """
    # Build properly structured conversation
    conversation_messages = []
    
    # Add system prompt as developer message
    conversation_messages.append({
        "role": "developer",
        "content": IMAGE_INTENT_SYSTEM_PROMPT
    })
    
    # Track if we've seen recent images
    has_recent_image = False
    
    # Add historical messages with proper roles
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        # Handle multi-part content
        if isinstance(content, list):
            text_parts = [c.get("text", "") for c in content if c.get("type") == "input_text"]
            content = " ".join(text_parts)
        
        # Check if assistant recently generated an image
        if role == "assistant" and "generated image" in content.lower():
            has_recent_image = True
        
        # Pass full message content without truncation
        # This ensures the intent classifier has complete context
        
        # Add message with proper role
        conversation_messages.append({
            "role": role,
            "content": content
        })
    
    # Add the current message to classify with metadata
    current_msg_with_metadata = last_user_message
    if has_attached_images:
        current_msg_with_metadata += "\n[Note: User has attached images with this message]"
    
    conversation_messages.append({
        "role": "user",
        "content": current_msg_with_metadata
    })
    
    # Add classification instruction as final user message
    conversation_messages.append({
        "role": "user", 
        "content": "Based on this conversation, classify the user's latest message. Respond with ONLY one of: new, edit, vision, ambiguous, or none."
    })
    
    # Debug logging
    self.log_debug(f"Intent classification with {len(conversation_messages)} messages")
    self.log_debug(f"Historical messages: {len(messages)}, has_recent_image: {has_recent_image}")
    if hasattr(config, 'debug_intent_classification') and config.debug_intent_classification:
        self.log_debug(f"Messages structure: {conversation_messages[-3:]}")  # Last 3 messages
    
    try:
        # Build request parameters with properly structured conversation
        request_params = {
            "model": config.utility_model,
            "input": conversation_messages,
            "max_output_tokens": config.utility_max_tokens,  # Configurable for different reasoning efforts
            "store": False,  # Never store classification calls
        }
        
        # Utility model is a GPT-5-series reasoning model (gpt-5-mini)
        request_params["temperature"] = 1.0  # Fixed for reasoning models
        request_params["reasoning"] = {"effort": config.utility_reasoning_effort}  # Use utility config
        request_params["text"] = {"verbosity": config.utility_verbosity}  # Use utility config
        
        self.log_debug(f"About to call responses.create for intent classification at {time.strftime('%H:%M:%S')}")
        self.log_debug(f"Using model: {config.utility_model}, timeout: {self.client.timeout}s")
        
        # Use safe API call wrapper with intent-specific timeout
        response = await self._safe_api_call(
            self.client.responses.create,
            operation_type="intent_classification",  # Uses 30s timeout
            **request_params
        )

        self.log_debug(f"Response received from API at {time.strftime('%H:%M:%S')}")
        
        # Extract True/False response
        result = ""
        if response.output:
            for item in response.output:
                if hasattr(item, "content") and item.content:
                    for content in item.content:
                        if hasattr(content, "text"):
                            result += content.text
        
        result = result.strip().lower()
        
        # Validate that we got a single word response
        if ' ' in result or len(result) > 20:
            # Classifier returned a full sentence instead of a word
            self.log_error(f"Classifier returned invalid response (expected single word): '{result[:100]}...'")
            result = "none"  # Default to text_only for safety
        
        # Debug logging
        self.log_debug(f"Image check raw result: '{result}' for message: '{last_user_message[:50]}...'")
        
        # Map the 5-state classifier results to intent categories
        if result == "new":
            intent = "new_image"
        elif result == "edit":
            intent = "edit_image"
        elif result == "vision":
            intent = "vision"
        elif result == "ambiguous":
            intent = "ambiguous_image"
        elif result == "none":
            intent = "text_only"
        else:
            # Fallback for unexpected responses
            self.log_warning(f"Unexpected classifier result: '{result}', defaulting to text_only")
            intent = "text_only"
        
        self.log_debug(f"Classified intent: {intent}")
        return intent
        
    except TimeoutError:
        # On timeout, retry with exponential backoff
        for retry in range(1, max_retries + 1):
            wait_time = 2 ** (retry - 1)  # 1s, 2s, 4s...
            self.log_warning(f"Intent classification timeout (attempt {retry}/{max_retries}), retrying in {wait_time}s...")
            time.sleep(wait_time)

            try:
                # Retry the classification with shorter timeout
                response = await self._safe_api_call(
                    self.client.responses.create,
                    operation_type="intent_classification",
                    timeout_seconds=15,  # Shorter timeout for retries
                    **request_params
                )

                # Process response (same as above)
                result = ""
                if response.output:
                    for item in response.output:
                        if hasattr(item, "content") and item.content:
                            for content in item.content:
                                if hasattr(content, "text"):
                                    result += content.text

                result = result.strip().lower()

                # Validate and map result
                if ' ' in result or len(result) > 20:
                    result = "none"

                # Map to intent
                if result == "new":
                    intent = "new_image"
                elif result == "edit":
                    intent = "edit_image"
                elif result == "ambiguous":
                    intent = "ambiguous_image"
                elif result == "vision":
                    intent = "vision"
                else:
                    intent = "text_only"

                self.log_info(f"Intent classification succeeded on retry {retry}: {intent}")
                return intent

            except TimeoutError:
                if retry == max_retries:
                    self.log_error(f"Intent classification failed after {max_retries} retries")
                    return 'error'  # Return error to trigger proper error handling
                continue
            except Exception as retry_error:
                self.log_error(f"Retry {retry} failed with error: {retry_error}")
                if retry == max_retries:
                    return 'error'
                continue

        # Should not reach here, but failsafe
        return 'error'

    except Exception as e:
        self.log_error(f"Error classifying intent: {e}")
        self.log_error(f"Exception type: {type(e).__name__}")
        return 'error'  # Return error instead of defaulting to text

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
    reasoning_effort = reasoning_effort or config.default_reasoning_effort
    request_params["reasoning"] = {"effort": reasoning_effort}
    verbosity = verbosity or config.default_verbosity
    request_params["text"] = {"verbosity": verbosity}

    # gpt-5.5 allows temperature/top_p when reasoning=none
    if model.startswith("gpt-5.5") and reasoning_effort == "none":
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

        # Extract text from response
        output_text = ""
        if response.output:
            for item in response.output:
                if hasattr(item, "content") and item.content:
                    for content in item.content:
                        if hasattr(content, "text"):
                            output_text += content.text

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
    tool_choice: Optional[str] = None
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
    reasoning_effort = reasoning_effort or config.default_reasoning_effort
    request_params["reasoning"] = {"effort": reasoning_effort}
    verbosity = verbosity or config.default_verbosity
    request_params["text"] = {"verbosity": verbosity}

    # gpt-5.5 allows temperature/top_p when reasoning=none
    if model.startswith("gpt-5.5") and reasoning_effort == "none":
        request_params["top_p"] = top_p
    else:
        request_params["temperature"] = 1.0  # MUST be 1.0 for reasoning models

    self.log_debug(f"Creating text response with tools and custom timeout {timeout_seconds}s, model {model}, tools: {tools}")

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

        # Extract text from response and detect tool usage
        output_text = ""
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
                elif item_type == "web_search_call":
                    if "web_search" not in tools_actually_used:
                        tools_actually_used.append("web_search")
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

                # Extract text content
                if hasattr(item, "content") and item.content:
                    for content in item.content:
                        if hasattr(content, "text"):
                            output_text += content.text

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
