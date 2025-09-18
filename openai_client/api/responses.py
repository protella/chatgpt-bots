from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from config import config
from prompts import IMAGE_INTENT_SYSTEM_PROMPT

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
    
    # Handle model-specific parameters
    if model.startswith("gpt-5"):
        # Check if it's a reasoning model (not chat model)
        is_reasoning_model = "chat" not in model.lower()
        
        if is_reasoning_model:
            # GPT-5 reasoning models (nano, mini, full)
            # Fixed temperature, supports reasoning_effort and verbosity
            request_params["temperature"] = 1.0  # MUST be 1.0 for reasoning models
            reasoning_effort = reasoning_effort or config.default_reasoning_effort
            request_params["reasoning"] = {"effort": reasoning_effort}
            verbosity = verbosity or config.default_verbosity
            request_params["text"] = {"verbosity": verbosity}
        else:
            # GPT-5 chat models - standard parameters only
            # temperature and top_p work normally, no reasoning/verbosity
            request_params["top_p"] = top_p
    else:
        # GPT-4 and other models - include top_p
        request_params["top_p"] = top_p
    
    self.log_debug(f"Creating text response with model {model}, temp {temperature}")

    try:
        # API call with enforced timeout wrapper
        self.log_debug(f"[HANG_DEBUG] About to create text response with {len(input_messages)} messages")
        response = await self._safe_api_call(
            self.client.responses.create,
            operation_type="general",
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
    store: bool = False
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
    
    Returns:
        Generated text response
    """
    model = model or config.gpt_model
    temperature = temperature if temperature is not None else config.default_temperature
    max_tokens = max_tokens or config.default_max_tokens
    top_p = top_p if top_p is not None else config.default_top_p
    
    # Build request parameters
    request_params = {
        "model": model,
        "input": [{"role": msg["role"], "content": msg["content"]} for msg in messages],
        "tools": tools,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "store": store,
    }
    
    # Add system prompt if provided
    if system_prompt:
        request_params["instructions"] = system_prompt
    
    # Handle model-specific parameters
    if model.startswith("gpt-5"):
        # Check if it's a reasoning model (not chat model)
        is_reasoning_model = "chat" not in model.lower()
        
        if is_reasoning_model:
            # GPT-5 reasoning models (nano, mini, full)
            # Fixed temperature, supports reasoning_effort and verbosity
            request_params["temperature"] = 1.0  # MUST be 1.0 for reasoning models
            reasoning_effort = reasoning_effort or config.default_reasoning_effort
            request_params["reasoning"] = {"effort": reasoning_effort}
            verbosity = verbosity or config.default_verbosity
            request_params["text"] = {"verbosity": verbosity}
        else:
            # GPT-5 chat models - standard parameters only
            request_params["top_p"] = top_p
    else:
        # GPT-4 and other models - include top_p
        request_params["top_p"] = top_p
    
    self.log_debug(f"Creating text response with tools using model {model}, tools: {tools}")
    
    try:
        # API call with enforced timeout wrapper
        response = await self._safe_api_call(
            self.client.responses.create,
            operation_type="general",
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
        
        self.log_info(f"Generated response with tools: {len(output_text)} chars")
        return output_text
        
    except Exception as e:
        self.log_error(f"Error creating response with tools: {e}", exc_info=True)
        raise

async def create_streaming_response(
    self,
    messages: List[Dict[str, Any]],
    stream_callback: Callable[[str], None],
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
    
    # Handle model-specific parameters
    if model.startswith("gpt-5"):
        # Check if it's a reasoning model (not chat model)
        is_reasoning_model = "chat" not in model.lower()
        
        if is_reasoning_model:
            # GPT-5 reasoning models (nano, mini, full)
            # Fixed temperature, supports reasoning_effort and verbosity
            request_params["temperature"] = 1.0  # MUST be 1.0 for reasoning models
            reasoning_effort = reasoning_effort or config.default_reasoning_effort
            request_params["reasoning"] = {"effort": reasoning_effort}
            verbosity = verbosity or config.default_verbosity
            request_params["text"] = {"verbosity": verbosity}
        else:
            # GPT-5 chat models - standard parameters only
            # temperature and top_p work normally, no reasoning/verbosity
            request_params["top_p"] = top_p
    else:
        # GPT-4 and other models - include top_p
        request_params["top_p"] = top_p
    
    self.log_debug(f"Creating streaming response with model {model}, temp {temperature}")
    
    try:
        response = await self._safe_api_call(
            self.client.responses.create,
            operation_type="general",
            **request_params
        )
        
        complete_text = ""
        last_chunk_time = None  # Don't start timer until first event
        first_event = True
        
        # Process streaming events
        async for event in response:
            try:
                current_time = time.time()
                
                # Only check timeout after first event has been received
                if last_chunk_time is not None:
                    time_since_last_chunk = current_time - last_chunk_time
                    # Log if event took unusually long but don't error
                    if time_since_last_chunk > self.stream_timeout_seconds:
                        self.log_warning(f"Stream event took {time_since_last_chunk:.1f}s to arrive (timeout is {self.stream_timeout_seconds}s)")
                elif first_event:
                    # Log time to first event for monitoring
                    self.log_debug("Received first streaming event")
                    first_event = False
                
                # Update timer for next iteration
                last_chunk_time = current_time
                
                # Get event type without logging every single one
                event_type = getattr(event, 'type', 'unknown')
                
                # DEBUG: Log ALL events to see what's actually coming through
                if event_type not in ["response.output_item.added", "response.output_item.delta", 
                                     "response.output_text.delta", "response.output_item.done"]:
                    self.log_debug(f"Stream event received: {event_type}")
                
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
                        # Call the callback with the text chunk
                        try:
                            stream_callback(text_chunk)
                        except Exception as callback_error:
                            self.log_warning(f"Stream callback error: {callback_error}")
                    continue
                elif event_type == "response.output_item.done":
                    continue  # Skip without logging
                elif event_type in ["response.done", "response.completed"]:
                    self.log_info("Stream completed")
                    # Signal the callback that streaming is complete with None
                    # This allows it to flush any remaining buffered text
                    try:
                        stream_callback(None)
                    except Exception as callback_error:
                        self.log_warning(f"Stream completion callback error: {callback_error}")
                    break
                elif event_type and ("call" in event_type or "tool" in event_type):
                    # Handle specific tool events
                    if tool_callback:
                        if event_type == "response.web_search_call.in_progress":
                            tool_callback("web_search", "started")
                        elif event_type == "response.web_search_call.searching":
                            tool_callback("web_search", "searching")
                        elif event_type == "response.web_search_call.completed":
                            tool_callback("web_search", "completed")
                        elif event_type == "response.file_search_call.in_progress":
                            tool_callback("file_search", "started")
                        elif event_type == "response.file_search_call.searching":
                            tool_callback("file_search", "searching")
                        elif event_type == "response.file_search_call.completed":
                            tool_callback("file_search", "completed")
                        elif event_type == "response.image_generation_call.in_progress":
                            tool_callback("image_generation", "started")
                        elif event_type == "response.image_generation_call.generating":
                            tool_callback("image_generation", "generating")
                        elif event_type == "response.image_generation_call.completed":
                            tool_callback("image_generation", "completed")
                    # Log tool-related events for visibility
                    self.log_info(f"Tool event: {event_type}")
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
    stream_callback: Callable[[str], None],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    system_prompt: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    verbosity: Optional[str] = None,
    store: bool = False,
    tool_callback: Optional[Callable[[str, str], None]] = None
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
    
    Returns:
        Complete generated text response
    """
    model = model or config.gpt_model
    temperature = temperature if temperature is not None else config.default_temperature
    max_tokens = max_tokens or config.default_max_tokens
    top_p = top_p if top_p is not None else config.default_top_p
    
    # Build request parameters
    request_params = {
        "model": model,
        "input": [{"role": msg["role"], "content": msg["content"]} for msg in messages],
        "tools": tools,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
        "store": store,
        "stream": True,  # Enable streaming
        "parallel_tool_calls": True,  # Allow parallel tool execution
    }
    
    # Add system prompt if provided
    if system_prompt:
        request_params["instructions"] = system_prompt
    
    # Handle model-specific parameters
    if model.startswith("gpt-5"):
        # Check if it's a reasoning model (not chat model)
        is_reasoning_model = "chat" not in model.lower()
        
        if is_reasoning_model:
            # GPT-5 reasoning models (nano, mini, full)
            # Fixed temperature, supports reasoning_effort and verbosity
            request_params["temperature"] = 1.0  # MUST be 1.0 for reasoning models
            reasoning_effort = reasoning_effort or config.default_reasoning_effort
            request_params["reasoning"] = {"effort": reasoning_effort}
            verbosity = verbosity or config.default_verbosity
            request_params["text"] = {"verbosity": verbosity}
        else:
            # GPT-5 chat models - standard parameters only
            request_params["top_p"] = top_p
    else:
        # GPT-4 and other models - include top_p
        request_params["top_p"] = top_p
    
    self.log_debug(f"Creating streaming response with tools using model {model}, tools: {tools}")
    
    try:
        response = await self._safe_api_call(
            self.client.responses.create,
            operation_type="general",
            **request_params
        )
        
        complete_text = ""
        last_chunk_time = None  # Don't start timer until first event
        first_event = True
        
        # Process streaming events
        async for event in response:
            try:
                current_time = time.time()
                
                # Only check timeout after first event has been received
                if last_chunk_time is not None:
                    time_since_last_chunk = current_time - last_chunk_time
                    # Log if event took unusually long but don't error
                    if time_since_last_chunk > self.stream_timeout_seconds:
                        self.log_warning(f"Stream event took {time_since_last_chunk:.1f}s to arrive (timeout is {self.stream_timeout_seconds}s)")
                elif first_event:
                    # Log time to first event for monitoring
                    self.log_debug("Received first streaming event")
                    first_event = False
                
                # Update timer for next iteration
                last_chunk_time = current_time
                
                # Get event type without logging every single one
                event_type = getattr(event, 'type', 'unknown')
                
                # DEBUG: Log ALL events to see what's actually coming through
                if event_type not in ["response.output_item.added", "response.output_item.delta", 
                                     "response.output_text.delta", "response.output_item.done"]:
                    self.log_debug(f"Stream event received: {event_type}")
                
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
                        # Call the callback with the text chunk
                        try:
                            stream_callback(text_chunk)
                        except Exception as callback_error:
                            self.log_warning(f"Stream callback error: {callback_error}")
                    continue
                elif event_type == "response.output_item.done":
                    continue  # Skip without logging
                elif event_type in ["response.done", "response.completed"]:
                    self.log_info("Stream completed")
                    # Signal the callback that streaming is complete with None
                    # This allows it to flush any remaining buffered text
                    try:
                        stream_callback(None)
                    except Exception as callback_error:
                        self.log_warning(f"Stream completion callback error: {callback_error}")
                    break
                elif event_type and ("call" in event_type or "tool" in event_type):
                    # Handle specific tool events
                    if tool_callback:
                        if event_type == "response.web_search_call.in_progress":
                            tool_callback("web_search", "started")
                        elif event_type == "response.web_search_call.searching":
                            tool_callback("web_search", "searching")
                        elif event_type == "response.web_search_call.completed":
                            tool_callback("web_search", "completed")
                        elif event_type == "response.file_search_call.in_progress":
                            tool_callback("file_search", "started")
                        elif event_type == "response.file_search_call.searching":
                            tool_callback("file_search", "searching")
                        elif event_type == "response.file_search_call.completed":
                            tool_callback("file_search", "completed")
                        elif event_type == "response.image_generation_call.in_progress":
                            tool_callback("image_generation", "started")
                        elif event_type == "response.image_generation_call.generating":
                            tool_callback("image_generation", "generating")
                        elif event_type == "response.image_generation_call.completed":
                            tool_callback("image_generation", "completed")
                    # Log tool-related events for visibility
                    self.log_info(f"Tool event: {event_type}")
                    continue
                else:
                    # Only log unhandled events for debugging
                    pass
                    
            except Exception as event_error:
                self.log_warning(f"Error processing stream event: {event_error}")
                continue
        
        self.log_info(f"Generated streaming response with tools: {len(complete_text)} chars")
        return complete_text
        
    except Exception as e:
        self.log_error(f"Error creating streaming response with tools: {e}", exc_info=True)
        raise

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
            "max_output_tokens": 20,  # Only need one word response
            "store": False,  # Never store classification calls
        }
        
        # Check if we're using a GPT-5 reasoning model
        if config.utility_model.startswith("gpt-5") and "chat" not in config.utility_model.lower():
            # GPT-5 reasoning model - use fixed temperature and reasoning parameters
            request_params["temperature"] = 1.0  # Fixed for reasoning models
            request_params["reasoning"] = {"effort": config.utility_reasoning_effort}  # Use utility config
            request_params["text"] = {"verbosity": config.utility_verbosity}  # Use utility config
        else:
            # GPT-4 or other models - use standard parameters
            request_params["temperature"] = 0.3  # Low temperature for consistent classification
        
        self.log_debug(f"About to call responses.create for intent classification at {time.strftime('%H:%M:%S')}")
        self.log_debug(f"Using model: {config.utility_model}, timeout: {self.client.timeout}s")
        
        # Use safe API call wrapper with intent-specific timeout
        self.log_debug("[HANG_DEBUG] Starting intent classification API call")
        response = await self._safe_api_call(
            self.client.responses.create,
            operation_type="intent",  # Uses min(30s, API_TIMEOUT_READ from .env)
            **request_params
        )
        self.log_debug("[HANG_DEBUG] Intent classification completed")

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
                # Retry the classification
                response = await self._safe_api_call(
                    self.client.responses.create,
                    operation_type="intent",
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
