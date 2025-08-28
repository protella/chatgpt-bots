"""
OpenAI Client wrapper for Responses API
Handles all interactions with OpenAI's GPT and image generation models
"""
import base64
import signal
import time
import functools
from contextlib import contextmanager
from io import BytesIO
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass
from openai import OpenAI
from config import config
from logger import LoggerMixin
from prompts import IMAGE_INTENT_SYSTEM_PROMPT, IMAGE_GEN_SYSTEM_PROMPT, IMAGE_EDIT_SYSTEM_PROMPT, VISION_ENHANCEMENT_PROMPT


def timeout_wrapper(timeout_seconds: float):
    """
    Decorator to add timeout handling to OpenAI API calls.
    Uses threading-based timeout since Slack bot runs in worker threads.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import threading
            
            # Always use threading-based timeout since we're in worker threads
            result = [None]
            exception = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout_seconds)
            
            if thread.is_alive():
                # Thread is still running, timeout occurred
                # Note: The thread will continue running in background
                # but we return control to avoid blocking
                raise TimeoutError(f"OpenAI API call timed out after {timeout_seconds} seconds")
            
            if exception[0]:
                raise exception[0]
            
            return result[0]
        
        return wrapper
    return decorator


@dataclass
class ImageData:
    """Container for image data"""
    base64_data: str
    format: str = "png"
    prompt: str = ""
    timestamp: float = 0
    slack_url: Optional[str] = None
    
    def to_bytes(self) -> BytesIO:
        """Convert base64 to BytesIO"""
        return BytesIO(base64.b64decode(self.base64_data))


class OpenAIClient(LoggerMixin):
    """Wrapper for OpenAI API using Responses API"""
    
    def __init__(self):
        # Initialize OpenAI client with timeout directly
        # The OpenAI SDK accepts timeout as a parameter
        self.client = OpenAI(
            api_key=config.openai_api_key,
            timeout=config.api_timeout_read,  # Use read timeout as the overall timeout
            max_retries=0  # Disable retries to fail fast on timeout
        )
        
        # Store streaming timeout for later use
        self.stream_timeout_seconds = config.api_timeout_streaming_chunk
        
        self.log_info(f"OpenAI client initialized with timeout: {config.api_timeout_read}s, "
                     f"streaming_chunk: {self.stream_timeout_seconds}s, max_retries: 0")
        self.log_debug(f"Client timeout object: {self.client.timeout}, type: {type(self.client.timeout)}")
    
    def _safe_api_call(self, api_method: Callable, *args, timeout_seconds: Optional[float] = None, operation_type: str = "general", **kwargs):
        """
        Wrapper for OpenAI API calls with enforced timeout.
        Falls back to SDK timeout if our wrapper fails.
        
        Args:
            api_method: The API method to call
            timeout_seconds: Override timeout in seconds (uses .env values by default)
            operation_type: Type of operation for timeout selection:
                - "intent": Quick intent classification (30s)
                - "streaming": Streaming operations (uses streaming chunk timeout)
                - "general": General API calls (uses API_TIMEOUT_READ from .env)
        """
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
        
        self.log_debug(f"Using timeout: {timeout}s for {operation_type} operation (from .env: read={config.api_timeout_read}s, chunk={config.api_timeout_streaming_chunk}s)")
        
        @timeout_wrapper(timeout)
        def make_call():
            return api_method(*args, **kwargs)
        
        try:
            return make_call()
        except TimeoutError as e:
            self.log_error(f"API call ({operation_type}) timed out after {timeout}s: {e}")
            raise
        except Exception as e:
            # If it's already a timeout error from the SDK, re-raise as our TimeoutError
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                raise TimeoutError(f"OpenAI API call timed out: {e}")
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
            response = self._safe_api_call(
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
            response = self._safe_api_call(
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
    def create_streaming_response(
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
            response = self._safe_api_call(
                self.client.responses.create,
                operation_type="general",
                **request_params
            )
            
            complete_text = ""
            last_chunk_time = None  # Don't start timer until first event
            first_event = True
            
            # Process streaming events
            for event in response:
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
    def create_streaming_response_with_tools(
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
            response = self._safe_api_call(
                self.client.responses.create,
                operation_type="general",
                **request_params
            )
            
            complete_text = ""
            last_chunk_time = None  # Don't start timer until first event
            first_event = True
            
            # Process streaming events
            for event in response:
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
    def classify_intent(
        self,
        messages: List[Dict[str, Any]],
        last_user_message: str,
        has_attached_images: bool = False
    ) -> str:
        """
        Classify user intent using a lightweight model
        
        Args:
            messages: Recent conversation context (last 6-8 exchanges)
            last_user_message: The latest user message to classify
            has_attached_images: Whether the current message has images attached
        
        Returns:
            Intent classification: 'new_image', 'modify_image', or 'text_only'
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
            response = self._safe_api_call(
                self.client.responses.create,
                operation_type="intent",  # Uses min(30s, API_TIMEOUT_READ from .env)
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
            
        except TimeoutError as e:
            # Timeout is somewhat expected - log as warning, not error
            self.log_warning(f"Intent classification timed out after {30 if 'intent' in str(e) else self.client.timeout}s - defaulting to text_only")
            return 'text_only'  # Default to text on timeout
        except Exception as e:
            self.log_error(f"Error classifying intent: {e}")
            self.log_error(f"Exception type: {type(e).__name__}")
            self.log_error(f"Occurred at: {time.strftime('%H:%M:%S')}")
            import traceback
            self.log_error(f"Traceback: {traceback.format_exc()}")
            return 'text_only'  # Default to text on error
    def generate_image(
        self,
        prompt: str,
        size: Optional[str] = None,
        quality: Optional[str] = None,
        background: Optional[str] = None,
        format: Optional[str] = None,
        compression: Optional[int] = None,
        enhance_prompt: bool = True,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> ImageData:
        """
        Generate an image using GPT-Image-1 model
        
        Args:
            prompt: Image generation prompt
            size: Image size (1024x1024, etc. - check model capabilities)
            quality: Quality setting (reserved for future DALL-E 3 support)
            background: Background type (reserved for future use)
            format: Output format (always returns png for now)
            compression: Compression level (reserved for future use)
            enhance_prompt: Whether to enhance the prompt first
        
        Returns:
            ImageData object with generated image
        """
        # Default size for gpt-image-1
        size = size or config.default_image_size
        
        # Quality parameter is not used by gpt-image-1
        # Reserved for future DALL-E 3 support
        
        # Enhance prompt if requested
        enhanced_prompt = prompt
        if enhance_prompt:
            enhanced_prompt = self._enhance_image_prompt(prompt, conversation_history)
        
        self.log_info(f"Generating image: {prompt[:100]}...")
        
        try:
            # Build parameters for images.generate
            # Default to gpt-image-1 parameters
            params = {
                "model": config.image_model,  # gpt-image-1
                "prompt": enhanced_prompt,  # Use the enhanced prompt
                "n": 1  # Number of images to generate
            }
            
            # Add size if specified (gpt-image-1 supports size)
            if size:
                params["size"] = size
            
            # Note: gpt-image-1 doesn't support response_format or quality parameters
            # It returns URLs that we'll download and convert to base64
            
            # Future: When adding DALL-E 3 support, check model and add:
            # - response_format="b64_json"
            # - quality parameter
            # - style parameter
            
            # Use the images.generate API for image generation
            self.log_debug(f"Calling generate_image API with {config.api_timeout_read}s timeout")
            response = self.client.images.generate(**params)
            
            # Extract image data from response
            if response.data and len(response.data) > 0:
                # Check if we have base64 data
                if hasattr(response.data[0], 'b64_json') and response.data[0].b64_json:
                    image_data = response.data[0].b64_json
                # Otherwise, we might have a URL - need to download it
                elif hasattr(response.data[0], 'url') and response.data[0].url:
                    import requests
                    import base64
                    url = response.data[0].url
                    self.log_debug(f"Downloading image from URL: {url}")
                    img_response = requests.get(url)
                    if img_response.status_code == 200:
                        image_data = base64.b64encode(img_response.content).decode('utf-8')
                    else:
                        raise ValueError(f"Failed to download image from URL: {url}")
                else:
                    raise ValueError("No image data or URL in response")
            else:
                raise ValueError("No image data in response")
            
            self.log_info("Image generated successfully")
            
            return ImageData(
                base64_data=image_data,
                format="png",  # API always returns PNG for now
                prompt=enhanced_prompt,  # Store the enhanced prompt that was actually used
            )
            
        except Exception as e:
            self.log_error(f"Error generating image: {e}", exc_info=True)
            raise
    
    def _enhance_image_edit_prompt(
        self,
        user_request: str,
        image_description: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        stream_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Enhance an image editing prompt using the analyzed image description
        
        Args:
            user_request: User's edit request
            image_description: Description of the current image
            conversation_history: Recent conversation messages for additional context
        
        Returns:
            Enhanced edit prompt
        """
        # Build context with image description and user request
        context = f"Image Description:\n{image_description}\n\nUser Edit Request:\n{user_request}"
        
        # Add conversation history if it exists and has messages
        if conversation_history and len(conversation_history) > 0:
            context = "Previous Conversation:\n"
            for msg in conversation_history:  # Full conversation history per CLAUDE.md
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                # Handle multi-part content
                if isinstance(content, list):
                    text_parts = [c.get("text", "") for c in content if c.get("type") == "input_text"]
                    content = " ".join(text_parts)
                
                # Include full message content per CLAUDE.md - no truncation
                context += f"{role}: {content}\n"
            
            context += f"\nImage Description:\n{image_description}\n\nUser Edit Request:\n{user_request}"
        
        # Log the enhancement input
        print("\n" + "="*80)
        print("DEBUG: IMAGE EDIT FLOW - STEP 4: EDIT PROMPT ENHANCEMENT")
        print("="*80)
        print(f"User Request: {user_request}")
        print(f"Image Description: {image_description[:200]}..." if len(image_description) > 200 else f"Image Description: {image_description}")
        if conversation_history and len(conversation_history) > 0:
            print(f"Including {len(conversation_history)} conversation messages for context")
        print("="*80)
        
        try:
            # Build request parameters with edit-specific system prompt
            request_params = {
                "model": config.utility_model,
                "input": [
                    {"role": "developer", "content": IMAGE_EDIT_SYSTEM_PROMPT},
                    {"role": "user", "content": context}
                ],
                "max_output_tokens": 500,
                "store": False,
            }
            
            # Check if we're using a GPT-5 reasoning model
            if config.utility_model.startswith("gpt-5") and "chat" not in config.utility_model.lower():
                request_params["temperature"] = 1.0
                request_params["reasoning"] = {"effort": config.utility_reasoning_effort}
                request_params["text"] = {"verbosity": config.utility_verbosity}
            else:
                request_params["temperature"] = 0.7
            
            # Check if streaming callback provided
            if stream_callback:
                # Create streaming response with timeout wrapper
                stream = self._safe_api_call(
                    self.client.responses.create,
                    operation_type="streaming",
                    stream=True,
                    **request_params
                )
                enhanced = ""
                
                for event in stream:
                    event_type = event.type if hasattr(event, 'type') else None
                    
                    if event_type in ["response.output_item.delta", "response.output_text.delta"]:
                        # Extract text from delta event (same as in create_streaming_text_response)
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
                        
                        if text_chunk:
                            enhanced += text_chunk
                            if stream_callback:
                                stream_callback(text_chunk)
            else:
                # Non-streaming fallback
                response = self._safe_api_call(
                    self.client.responses.create,
                    operation_type="general",
                    **request_params
                )
                
                enhanced = ""
                if response.output:
                    for item in response.output:
                        if hasattr(item, "content") and item.content:
                            for content in item.content:
                                if hasattr(content, "text"):
                                    enhanced += content.text
            
            enhanced = enhanced.strip()
            
            # Log the enhanced result
            print("\n" + "="*80)
            print("DEBUG: IMAGE EDIT FLOW - STEP 5: ENHANCED EDIT PROMPT")
            print("="*80)
            print(f"Final Enhanced Edit Prompt:\n{enhanced}")
            print("="*80)
            
            if enhanced and len(enhanced) > 10:
                return enhanced
            else:
                # Fallback to simple combination
                # Return just the enhanced description without prefix
                return f"{image_description}. Change: {user_request}"
            
        except Exception as e:
            self.log_warning(f"Failed to enhance edit prompt: {e}")
            # Return just the enhanced description without prefix
            return f"{image_description}. Change: {user_request}"
    
    def _enhance_image_prompt(self, prompt: str, conversation_history: Optional[List[Dict[str, Any]]] = None, 
                             stream_callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Enhance an image generation prompt for better results
        
        Args:
            prompt: Original user prompt
            conversation_history: Recent conversation messages for context
            stream_callback: Optional callback for streaming the enhancement
        
        Returns:
            Enhanced prompt
        """
        # Build conversation context
        context = "Conversation History:\n"
        
        if conversation_history:
            # Include full conversation history per CLAUDE.md
            for msg in conversation_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                # Handle multi-part content
                if isinstance(content, list):
                    text_parts = [c.get("text", "") for c in content if c.get("type") == "input_text"]
                    content = " ".join(text_parts)
                
                # Include full message content per CLAUDE.md - no truncation
                context += f"{role}: {content}\n"
        
        context += f"\nCurrent User Request: {prompt}"
        
        # Log the prompt enhancement input
        print("\n" + "="*80)
        print("DEBUG: IMAGE EDIT FLOW - STEP 4: PROMPT ENHANCEMENT INPUT")
        print("="*80)
        print(f"Original Prompt to Enhance:\n{prompt}")
        print(f"\nFull Context Sent to Enhancer:\n{context}")
        print("="*80)
        
        try:
            # If streaming callback provided, use streaming response
            if stream_callback:
                # Use streaming version for real-time feedback
                enhanced = self.create_streaming_response(
                    messages=[{"role": "user", "content": context}],
                    stream_callback=stream_callback,
                    model=config.utility_model,
                    temperature=0.7 if "chat" in config.utility_model.lower() or not config.utility_model.startswith("gpt-5") else 1.0,
                    max_tokens=500,
                    system_prompt=IMAGE_GEN_SYSTEM_PROMPT,
                    reasoning_effort=config.utility_reasoning_effort if config.utility_model.startswith("gpt-5") and "chat" not in config.utility_model.lower() else None,
                    verbosity=config.utility_verbosity if config.utility_model.startswith("gpt-5") and "chat" not in config.utility_model.lower() else None,
                    store=False
                )
            else:
                # Use non-streaming version
                # Build request parameters
                request_params = {
                    "model": config.utility_model,
                    "input": [
                        {"role": "developer", "content": IMAGE_GEN_SYSTEM_PROMPT},
                        {"role": "user", "content": context}
                    ],
                    "max_output_tokens": 500,  # Increased for detailed image prompts
                    "store": False,
                }
                
                # Check if we're using a GPT-5 reasoning model
                if config.utility_model.startswith("gpt-5") and "chat" not in config.utility_model.lower():
                    # GPT-5 reasoning model - use fixed temperature and reasoning parameters
                    request_params["temperature"] = 1.0  # Fixed for reasoning models
                    request_params["reasoning"] = {"effort": config.utility_reasoning_effort}  # Use utility config
                    request_params["text"] = {"verbosity": config.utility_verbosity}  # Use utility config
                else:
                    # GPT-4 or other models - use standard parameters
                    request_params["temperature"] = 0.7  # Moderate temperature for creative prompts
                
                response = self._safe_api_call(
                    self.client.responses.create,
                    operation_type="general",
                    **request_params
                )
                
                enhanced = ""
                if response.output:
                    for item in response.output:
                        if hasattr(item, "content") and item.content:
                            for content in item.content:
                                if hasattr(content, "text"):
                                    enhanced += content.text
                
                enhanced = enhanced.strip()
            
            # Log the enhanced prompt result
            print("\n" + "="*80)
            print("DEBUG: IMAGE EDIT FLOW - STEP 5: ENHANCED PROMPT OUTPUT")
            print("="*80)
            print(f"Final Enhanced Prompt:\n{enhanced}")
            print("="*80)
            
            # Make sure we got a valid enhancement
            if enhanced and len(enhanced) > 10:
                self.log_debug(f"Enhanced prompt: {enhanced[:100]}...")
                return enhanced
            else:
                print("\n" + "="*80)
                print("DEBUG: Enhancement failed or too short, using original")
                print("="*80)
                return prompt
            
        except Exception as e:
            self.log_warning(f"Failed to enhance prompt: {e}")
            return prompt  # Return original on error
    
    def _enhance_vision_prompt(self, user_question: str) -> str:
        """
        Enhance a vision analysis prompt for more detailed responses
        
        Args:
            user_question: Original user question about the image
        
        Returns:
            Enhanced prompt for better vision analysis
        """
        try:
            # Build request parameters
            request_params = {
                "model": config.utility_model,
                "input": [
                    {"role": "developer", "content": VISION_ENHANCEMENT_PROMPT},
                    {"role": "user", "content": user_question}
                ],
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
            
            response = self._safe_api_call(
                self.client.responses.create,
                operation_type="general",
                **request_params
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
    
    def analyze_images(
        self,
        images: List[str],
        question: str,
        detail: Optional[str] = None,
        enhance_prompt: bool = True,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        stream_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Analyze one or more images with a question
        
        Args:
            images: List of base64 encoded image data (max 10)
            question: Question about the image(s)
            detail: Analysis detail level (auto, low, high)
            enhance_prompt: Whether to enhance the question for better analysis
            stream_callback: Optional callback for streaming the response
        
        Returns:
            Analysis response
        """
        detail = detail or config.default_detail_level
        
        # Limit to 10 images
        if len(images) > 10:
            self.log_warning(f"Limiting to 10 images (received {len(images)})")
            images = images[:10]
        
        # Enhance the question if requested
        enhanced_question = question
        if enhance_prompt:
            enhanced_question = self._enhance_vision_prompt(question)
            self.log_info(f"Vision analysis with enhanced prompt: {enhanced_question[:100]}...")
        
        # Build content array with text and images
        content = [{"type": "input_text", "text": enhanced_question}]
        
        for image_data in images:
            # Use data URL format for base64 images
            content.append({
                "type": "input_image",
                "image_url": f"data:image/png;base64,{image_data}"
            })
        
        try:
            # Build request parameters with conversation history
            input_messages = []
            
            # Add platform system prompt if provided (for consistent personality/formatting)
            if system_prompt:
                input_messages.append({
                    "role": "developer",
                    "content": system_prompt
                })
            
            # Include FULL conversation history if provided (including all developer messages)
            # Filter out metadata to avoid API errors
            if conversation_history:
                for msg in conversation_history:
                    # Only include role and content for API
                    api_msg = {"role": msg["role"], "content": msg["content"]}
                    input_messages.append(api_msg)
            
            # Add the current vision request
            input_messages.append({
                "role": "user",
                "content": content
            })
            
            # Check if streaming is requested
            if stream_callback:
                # Use streaming for vision analysis
                self.log_debug(f"Streaming vision analysis with {config.api_timeout_read}s timeout")
                
                request_params = {
                    "model": config.gpt_model,
                    "input": input_messages,
                    "max_output_tokens": config.vision_max_tokens,
                    "store": False,
                    "stream": True
                }
                
                # Add GPT-5 reasoning parameters if using a reasoning model
                if config.gpt_model.startswith("gpt-5") and "chat" not in config.gpt_model.lower():
                    request_params["temperature"] = 1.0
                    request_params["reasoning"] = {"effort": config.analysis_reasoning_effort}
                    request_params["text"] = {"verbosity": config.analysis_verbosity}
                
                # Stream the response
                output_text = ""
                stream = self._safe_api_call(
                    self.client.responses.create,
                    operation_type="general",
                    **request_params
                )
                
                # Process stream events (similar to create_streaming_response)
                for event in stream:
                    try:
                        # Get event type
                        event_type = getattr(event, 'type', 'unknown')
                        
                        if event_type == "response.created":
                            self.log_debug("Vision stream started")
                            continue
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
                                output_text += text_chunk
                                stream_callback(text_chunk)
                            continue
                        elif event_type in ["response.done", "response.completed"]:
                            self.log_debug("Vision stream completed")
                            # Signal completion to callback
                            try:
                                stream_callback(None)
                            except Exception as callback_error:
                                self.log_warning(f"Stream completion callback error: {callback_error}")
                            break
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
                    "store": False  # Don't store vision analysis calls
                }
                
                # Add GPT-5 reasoning parameters if using a reasoning model
                if config.gpt_model.startswith("gpt-5") and "chat" not in config.gpt_model.lower():
                    request_params["temperature"] = 1.0  # Fixed for reasoning models
                    request_params["reasoning"] = {"effort": config.analysis_reasoning_effort}  # Use analysis config
                    request_params["text"] = {"verbosity": config.analysis_verbosity}  # Use analysis config
                
                # API call with enforced timeout wrapper
                self.log_debug(f"Calling analyze_images API with {config.api_timeout_read}s timeout")
                response = self._safe_api_call(
                    self.client.responses.create,
                    operation_type="general",
                    **request_params
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
                    self.log_warning(f"analyze_images returned empty response")
                    self.log_debug(f"Response output structure: {response.output if response.output else 'No output'}")
                    
                    # Check if we only got reasoning tokens
                    if response.usage and hasattr(response.usage, "output_tokens_details"):
                        details = response.usage.output_tokens_details
                        if hasattr(details, "reasoning_tokens") and details.reasoning_tokens > 0:
                            self.log_warning(f"Response contained {details.reasoning_tokens} reasoning tokens but no text output")
                
                return output_text
            
        except Exception as e:
            self.log_error(f"Error analyzing images: {e}", exc_info=True)
            raise
    
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
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> ImageData:
        """
        Edit or combine images using GPT-Image-1 model
        
        Args:
            input_images: List of base64 encoded input images (max 16)
            prompt: Edit instructions (up to 32000 chars)
            input_fidelity: How closely to match input images (high/low)
            background: Background type (transparent/opaque/auto)
            mask: Optional base64 encoded PNG mask
            output_format: Format (png/jpeg/webp)
            output_compression: Compression level 0-100
            enhance_prompt: Whether to enhance the prompt first
            conversation_history: Recent conversation for context
        
        Returns:
            ImageData object with edited image
        """
        # Limit to 16 images
        if len(input_images) > 16:
            self.log_warning(f"Limiting to 16 images for editing (received {len(input_images)})")
            input_images = input_images[:16]
        
        # Default background
        background = background or config.default_image_background
        
        # Enhance prompt if requested
        enhanced_prompt = prompt
        if enhance_prompt:
            # Use the edit-specific enhancement for image editing
            if image_description:
                enhanced_prompt = self._enhance_image_edit_prompt(
                    user_request=prompt,
                    image_description=image_description,
                    conversation_history=conversation_history
                )
            else:
                # Fallback to regular enhancement if no description
                enhanced_prompt = self._enhance_image_prompt(prompt, conversation_history)
        
        self.log_info(f"Editing {len(input_images)} image(s): {prompt[:100]}...")
        
        try:
            # Convert base64 to BytesIO objects with proper file extension
            from io import BytesIO
            image_files = []
            
            # Default mimetypes if not provided
            if not input_mimetypes:
                input_mimetypes = ["image/png"] * len(input_images)
            
            for i, b64_data in enumerate(input_images):
                image_bytes = base64.b64decode(b64_data)
                bio = BytesIO(image_bytes)
                
                # Determine file extension from mimetype
                mimetype = input_mimetypes[i] if i < len(input_mimetypes) else "image/png"
                if mimetype == "image/jpeg":
                    bio.name = f"image_{i}.jpg"
                elif mimetype == "image/webp":
                    bio.name = f"image_{i}.webp"
                else:  # Default to PNG
                    bio.name = f"image_{i}.png"
                
                image_files.append(bio)
            
            # Build parameters for images.edit
            params = {
                "model": config.image_model,  # gpt-image-1
                "image": image_files if len(image_files) > 1 else image_files[0],
                "prompt": enhanced_prompt,
                "input_fidelity": input_fidelity,
                "background": background,
                "output_format": output_format,
                "n": 1
            }
            
            # Only add compression for JPEG/WebP (PNG must be 100)
            if output_format in ["jpeg", "webp"]:
                params["output_compression"] = output_compression
            elif output_format == "png" and output_compression != 100:
                self.log_debug(f"PNG format requires compression=100, ignoring {output_compression}")
            
            # Add mask if provided
            if mask:
                mask_bytes = base64.b64decode(mask)
                params["mask"] = BytesIO(mask_bytes)
            
            # Use the images.edit API
            self.log_debug(f"Calling edit_image API with {config.api_timeout_read}s timeout")
            response = self.client.images.edit(**params)
            
            # Extract image data from response
            if response.data and len(response.data) > 0:
                # Check if we have base64 data
                if hasattr(response.data[0], 'b64_json') and response.data[0].b64_json:
                    image_data = response.data[0].b64_json
                # Otherwise, we might have a URL - need to download it
                elif hasattr(response.data[0], 'url') and response.data[0].url:
                    import requests
                    url = response.data[0].url
                    self.log_debug(f"Downloading edited image from URL: {url}")
                    img_response = requests.get(url)
                    if img_response.status_code == 200:
                        image_data = base64.b64encode(img_response.content).decode('utf-8')
                    else:
                        raise ValueError(f"Failed to download edited image from URL: {url}")
                else:
                    raise ValueError("No image data or URL in response")
            else:
                raise ValueError("No image data in response")
            
            self.log_info("Image edited successfully")
            
            return ImageData(
                base64_data=image_data,
                format=output_format,
                prompt=enhanced_prompt,
            )
            
        except Exception as e:
            self.log_error(f"Error editing image: {e}", exc_info=True)
            raise
    
    def analyze_image(self, image_data: str, question: str, detail: Optional[str] = None) -> str:
        """
        Analyze a single image (backward compatibility wrapper)
        
        Args:
            image_data: Base64 encoded image data
            question: Question about the image
            detail: Analysis detail level (auto, low, high)
        
        Returns:
            Analysis response
        """
        return self.analyze_images([image_data], question, detail)