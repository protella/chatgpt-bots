from __future__ import annotations

import asyncio
import re
from typing import Any, List, Optional, Set
from openai import APIError, APIStatusError

from base_client import BaseClient, Message, Response
from config import config
from streaming import FenceHandler, RateLimitManager, StreamingBuffer


class TextHandlerMixin:
    async def _handle_text_response(self, user_content: Any, thread_state, client: BaseClient,
                              message: Message, thinking_id: Optional[str] = None,
                              attachment_urls: Optional[List[str]] = None,
                              retry_count: int = 0,
                              failed_mcp_server: Optional[str] = None) -> Response:
        """Handle text-only response generation"""
        # Get thread config (with user preferences)
        thread_config = config.get_thread_config(
            overrides=thread_state.config_overrides,
            user_id=message.user_id,
            db=self.db
        )
        
        # Check if streaming is enabled and supported (respecting user prefs)
        # CRITICAL: Don't retry streaming if we've already failed once
        streaming_enabled = thread_config.get('enable_streaming', config.enable_streaming)
        if (hasattr(client, 'supports_streaming') and client.supports_streaming() and
            streaming_enabled and thinking_id is not None and retry_count == 0):  # Streaming requires a message ID to update
            return await self._handle_streaming_text_response(user_content, thread_state, client, message, thinking_id, attachment_urls)
        
        # Fall back to non-streaming logic
        # For vision requests with images, store only a text breadcrumb with URLs, not the base64 data
        if isinstance(user_content, list):
            # Extract text and count images from the multi-part content
            text_parts = []
            image_count = 0
            for item in user_content:
                if item.get("type") == "input_text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "input_image":
                    image_count += 1
            
            # Create clean text for thread history (no URLs or counts)
            breadcrumb_text = " ".join(text_parts).strip()
            
            # Add simplified breadcrumb to thread state (no base64 data)
            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
            message_ts = message.metadata.get("ts") if message.metadata else None
            self._add_message_with_token_management(thread_state, "user", breadcrumb_text, db=self.db, thread_key=thread_key, message_ts=message_ts)
            
            # Use the full content with images for the actual API call
            messages_for_api = thread_state.messages[:-1] + [{"role": "user", "content": user_content}]
        else:
            # Simple text content - add as-is
            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
            message_ts = message.metadata.get("ts") if message.metadata else None
            
            # Check if this content contains documents and add metadata
            message_metadata = None
            if isinstance(user_content, str) and "=== DOCUMENT:" in user_content:
                # Don't mark as document_upload type - documents should be trimmable
                message_metadata = {"contains_document": True}
            
            self._add_message_with_token_management(thread_state, "user", user_content, db=self.db, thread_key=thread_key, message_ts=message_ts, metadata=message_metadata)
            messages_for_api = thread_state.messages
        
        # Inject stored image analyses into the conversation for full context
        messages_for_api = self._inject_image_analyses(messages_for_api, thread_state)

        # Strip tools attribution from assistant messages before sending to API
        # (keeps user-visible context clean while preventing metadata pollution)
        for msg in messages_for_api:
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                msg["content"] = re.sub(r'\n\n_Used Tools:.+?_$', '', msg["content"])

        # Pre-trim messages to fit within context window
        messages_for_api = await self._pre_trim_messages_for_api(messages_for_api, model=thread_state.current_model)
        
        # Get thread config (with user preferences)
        thread_config = config.get_thread_config(
            overrides=thread_state.config_overrides,
            user_id=message.user_id,
            db=self.db
        )
        
        # Use thread's system prompt (which is now platform-specific)
        # Always regenerate to get current time
        user_timezone = message.metadata.get("user_timezone", "UTC") if message.metadata else "UTC"
        user_tz_label = message.metadata.get("user_tz_label", None) if message.metadata else None
        user_real_name = message.metadata.get("user_real_name", None) if message.metadata else None
        user_email = message.metadata.get("user_email", None) if message.metadata else None
        # Pass the model for dynamic knowledge cutoff (respecting user prefs)
        web_search_enabled = thread_config.get('enable_web_search', config.enable_web_search)
        model = config.web_search_model or thread_config["model"] if web_search_enabled else thread_config["model"]
        system_prompt = self._get_system_prompt(client, user_timezone, user_tz_label, user_real_name, user_email, model, web_search_enabled, thread_state.has_trimmed_messages, thread_config.get('custom_instructions'))
        
        # Update status before generating
        self._update_status(client, message.channel_id, thinking_id, "Generating response...")
        
        # Determine timeout based on retry attempt
        retry_timeout = 60.0 if retry_count > 0 else None

        # Determine which model to use (web search model if web search enabled)
        web_search_enabled = thread_config.get('enable_web_search', config.enable_web_search)
        model = config.web_search_model or thread_config["model"] if web_search_enabled else thread_config["model"]

        # Build tools array (includes web_search and/or MCP tools based on config)
        # Exclude any MCP server that failed in a previous attempt
        tools = self._build_tools_array(thread_config, model, exclude_mcp_server=failed_mcp_server)

        # Generate response with or without tools
        tools_actually_used = []  # Track which tools were actually invoked
        if tools:
            # Generate response with tools
            if retry_timeout:
                # Use shorter timeout for retry via direct _safe_api_call
                result = await self.openai_client._create_text_response_with_tools_with_timeout(
                    messages=messages_for_api,
                    tools=tools,
                    model=model,
                    temperature=thread_config["temperature"],
                    max_tokens=thread_config["max_tokens"],
                    system_prompt=system_prompt,
                    reasoning_effort=thread_config.get("reasoning_effort"),
                    verbosity=thread_config.get("verbosity"),
                    store=False,
                    timeout_seconds=retry_timeout,
                    return_metadata=True
                )
                response_text = result["text"]
                tools_actually_used = result["tools_used"]
            else:
                result = await self.openai_client.create_text_response_with_tools(
                    messages=messages_for_api,
                    tools=tools,
                    model=model,
                    temperature=thread_config["temperature"],
                    max_tokens=thread_config["max_tokens"],
                    system_prompt=system_prompt,
                    reasoning_effort=thread_config.get("reasoning_effort"),
                    verbosity=thread_config.get("verbosity"),
                    store=False,  # Match the existing behavior
                    return_metadata=True
                )
                response_text = result["text"]
                tools_actually_used = result["tools_used"]
        else:
            # Generate response without tools
            if retry_timeout:
                # Use shorter timeout for retry via direct _safe_api_call
                response_text = await self.openai_client._create_text_response_with_timeout(
                    messages=messages_for_api,
                    model=model,
                    temperature=thread_config["temperature"],
                    max_tokens=thread_config["max_tokens"],
                    system_prompt=system_prompt,
                    reasoning_effort=thread_config.get("reasoning_effort"),
                    verbosity=thread_config.get("verbosity"),
                    timeout_seconds=retry_timeout
                )
            else:
                response_text = await self.openai_client.create_text_response(
                    messages=messages_for_api,
                    model=model,
                    temperature=thread_config["temperature"],
                    max_tokens=thread_config["max_tokens"],
                    system_prompt=system_prompt,
                    reasoning_effort=thread_config.get("reasoning_effort"),
                    verbosity=thread_config.get("verbosity")
                )
        
        # Build unified tools attribution at the end of response
        # Use the actual tools that were invoked (from response metadata)
        if tools_actually_used or failed_mcp_server:
            # Add unified tools note at the END
            if tools_actually_used:
                # Show successful tools
                if failed_mcp_server:
                    tools_note = f"\n\n_Used Tools: {', '.join(tools_actually_used)} (failed: {failed_mcp_server})_"
                else:
                    tools_note = f"\n\n_Used Tools: {', '.join(tools_actually_used)}_"
            else:
                # Only failed MCP, no successful tools
                tools_note = f"\n\n_MCP server '{failed_mcp_server}' could not be reached. Response generated without external tools._"

            response_text = response_text + tools_note
            self.log_info(f"Added tools attribution: {', '.join(tools_actually_used) if tools_actually_used else 'none'}{' with failure note' if failed_mcp_server else ''}")

        # Add assistant response to thread state
        thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
        self._add_message_with_token_management(thread_state, "assistant", response_text, db=self.db, thread_key=thread_key)
        
        # Schedule async cleanup after response
        cleanup_coro = self._async_post_response_cleanup(thread_state, thread_key)
        self._schedule_async_call(cleanup_coro)
        
        return Response(
            type="text",
            content=response_text
        )

    async def _handle_streaming_text_response(self, user_content: Any, thread_state, client: BaseClient,
                                      message: Message, thinking_id: Optional[str] = None,
                                      attachment_urls: Optional[List[str]] = None) -> Response:
        """Handle text-only response generation with streaming support"""
        # Check if client supports streaming
        if not hasattr(client, 'supports_streaming') or not client.supports_streaming():
            self.log_debug("Client doesn't support streaming, falling back to non-streaming")
            return await self._handle_text_response(user_content, thread_state, client, message, thinking_id, attachment_urls, retry_count=0)
        
        # Get streaming configuration from client
        streaming_config = client.get_streaming_config() if hasattr(client, 'get_streaming_config') else {}
        
        # Create streaming buffer and rate limit manager
        buffer = StreamingBuffer(
            update_interval=streaming_config.get("update_interval", 2.0),
            buffer_size_threshold=streaming_config.get("buffer_size", 500),
            min_update_interval=streaming_config.get("min_interval", 1.0)
        )
        
        rate_limiter = RateLimitManager(
            base_interval=streaming_config.get("update_interval", 2.0),
            min_interval=streaming_config.get("min_interval", 1.0),
            max_interval=streaming_config.get("max_interval", 30.0),
            failure_threshold=streaming_config.get("circuit_breaker_threshold", 5),
            cooldown_seconds=streaming_config.get("circuit_breaker_cooldown", 300)
        )
        
        self.log_info("Starting streaming response generation")
        
        # Process user content for thread state (same as non-streaming)
        if isinstance(user_content, list):
            # Extract text and count images from the multi-part content
            text_parts = []
            image_count = 0
            for item in user_content:
                if item.get("type") == "input_text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "input_image":
                    image_count += 1
            
            # Create clean text for thread history (no URLs or counts)
            breadcrumb_text = " ".join(text_parts).strip()
            
            # Add simplified breadcrumb to thread state (no base64 data)
            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
            message_ts = message.metadata.get("ts") if message.metadata else None
            self._add_message_with_token_management(thread_state, "user", breadcrumb_text, db=self.db, thread_key=thread_key, message_ts=message_ts)
            
            # Use the full content with images for the actual API call
            messages_for_api = thread_state.messages[:-1] + [{"role": "user", "content": user_content}]
        else:
            # Simple text content - add as-is
            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
            message_ts = message.metadata.get("ts") if message.metadata else None
            
            # Check if this content contains documents and add metadata
            message_metadata = None
            if isinstance(user_content, str) and "=== DOCUMENT:" in user_content:
                # Don't mark as document_upload type - documents should be trimmable
                message_metadata = {"contains_document": True}
            
            self._add_message_with_token_management(thread_state, "user", user_content, db=self.db, thread_key=thread_key, message_ts=message_ts, metadata=message_metadata)
            messages_for_api = thread_state.messages
        
        # Inject stored image analyses into the conversation for full context
        messages_for_api = self._inject_image_analyses(messages_for_api, thread_state)

        # Strip tools attribution from assistant messages before sending to API
        # (keeps user-visible context clean while preventing metadata pollution)
        for msg in messages_for_api:
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                msg["content"] = re.sub(r'\n\n_Used Tools:.+?_$', '', msg["content"])

        # Pre-trim messages to fit within context window
        messages_for_api = await self._pre_trim_messages_for_api(messages_for_api, model=thread_state.current_model)
        
        # Get thread config (with user preferences)
        thread_config = config.get_thread_config(
            overrides=thread_state.config_overrides,
            user_id=message.user_id,
            db=self.db
        )
        
        # Use thread's system prompt (which is now platform-specific)
        # Always regenerate to get current time
        user_timezone = message.metadata.get("user_timezone", "UTC") if message.metadata else "UTC"
        user_tz_label = message.metadata.get("user_tz_label", None) if message.metadata else None
        user_real_name = message.metadata.get("user_real_name", None) if message.metadata else None
        user_email = message.metadata.get("user_email", None) if message.metadata else None
        # Pass the model for dynamic knowledge cutoff (respecting user prefs)
        web_search_enabled = thread_config.get('enable_web_search', config.enable_web_search)
        model = config.web_search_model or thread_config["model"] if web_search_enabled else thread_config["model"]
        system_prompt = self._get_system_prompt(client, user_timezone, user_tz_label, user_real_name, user_email, model, web_search_enabled, thread_state.has_trimmed_messages, thread_config.get('custom_instructions'))
        
        # Post an initial message to get the message ID for streaming updates
        # For streaming with potential tools, start with "Working on it" 
        # (will be overridden if tools are used)
        initial_message = f"{config.thinking_emoji} Working on it..."
        if thinking_id:
            # Update existing thinking message
            message_id = thinking_id
            await client.update_message(message.channel_id, message_id, initial_message)
        else:
            # We need a way to post a message and get its ID - this would depend on client implementation
            self.log_warning("No thinking_id provided for streaming - falling back to non-streaming")
            return await self._handle_text_response(user_content, thread_state, client, message, thinking_id, attachment_urls, retry_count=0)
        
        # Track tool states for status updates
        tool_states = {
            "web_search": False,
            "file_search": False,
            "image_generation": False,
            "mcp": False
        }

        # Track search counts
        search_counts = {
            "web_search": 0,
            "file_search": 0,
            "mcp": 0
        }

        # Track which MCP servers were used
        mcp_servers_used = set()

        # Define tool event callback
        async def tool_callback(tool_type: str, status: str):
            """Handle tool events for status updates"""
            nonlocal progress_task

            if status == "started":
                # Cancel progress updater when tools start (web search takes over status)
                if progress_task and not progress_task.done():
                    progress_task.cancel()
                    self.log_debug("Cancelled progress updater - tool started")

                # Tool just started - update status with appropriate emoji
                if tool_type == "web_search":
                    if not tool_states["web_search"]:
                        tool_states["web_search"] = True
                    search_counts["web_search"] += 1
                    # Show search count consistently for all searches
                    status_msg = f"{config.web_search_emoji} Searching the web (query {search_counts['web_search']})..."
                    try:
                        # Use update_message_streaming for consistency with streaming flow
                        result = await client.update_message_streaming(message.channel_id, message_id, status_msg)
                        if result["success"]:
                            self.log_info(f"Web search #{search_counts['web_search']} started - updated status")
                        else:
                            self.log_warning(f"Failed to update web search status: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        self.log_error(f"Error updating web search status: {e}")
                elif tool_type == "file_search":
                    if not tool_states["file_search"]:
                        tool_states["file_search"] = True
                    search_counts["file_search"] += 1
                    # Show search count consistently for all searches
                    status_msg = f"{config.web_search_emoji} Searching files (query {search_counts['file_search']})..."
                    try:
                        result = await client.update_message_streaming(message.channel_id, message_id, status_msg)
                        if result["success"]:
                            self.log_info(f"File search #{search_counts['file_search']} started - updated status")
                        else:
                            self.log_warning(f"Failed to update file search status: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        self.log_error(f"Error updating file search status: {e}")
                elif tool_type == "image_generation" and not tool_states["image_generation"]:
                    tool_states["image_generation"] = True
                    status_msg = f"{config.circle_loader_emoji} Generating image. This may take a minute..."
                    try:
                        result = await client.update_message_streaming(message.channel_id, message_id, status_msg)
                        if result["success"]:
                            self.log_info("Image generation started - updated status")
                        else:
                            self.log_warning(f"Failed to update image gen status: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        self.log_error(f"Error updating image gen status: {e}")
            elif tool_type == "mcp":
                # MCP has its own status values (not "started")
                if progress_task and not progress_task.done():
                    progress_task.cancel()
                    self.log_debug("Cancelled progress updater - MCP tool started")

                if status == "discovering_tools" and not tool_states["mcp"]:
                    tool_states["mcp"] = True
                    # Discovery status message suppressed per user preference (logging only)
                    self.log_info("MCP tool discovery started (status message suppressed)")
                elif status == "calling":
                    search_counts["mcp"] += 1
                    # Build status message with call count if multiple calls
                    call_suffix = f" (call {search_counts['mcp']})" if search_counts['mcp'] > 1 else ""
                    status_msg = f"{config.web_search_emoji} Using MCP tools{call_suffix}..."
                    try:
                        result = await client.update_message_streaming(message.channel_id, message_id, status_msg)
                        if result["success"]:
                            self.log_info(f"MCP call #{search_counts['mcp']} started - updated status")
                        else:
                            self.log_warning(f"Failed to update MCP call status: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        self.log_error(f"Error updating MCP call status: {e}")
            elif status == "completed":
                # Tool completed - clear the status for that tool
                if tool_type in tool_states:
                    tool_states[tool_type] = False
                    # Don't update status here - let the next event (another tool or text streaming) handle it
                    self.log_info(f"{tool_type} completed")
        
        # Track current streaming message and overflow
        current_message_id = message_id
        current_part = 1
        overflow_buffer = ""
        continuation_msg = "\n\n*Continued in next message...*"
        # Reserve space for: continuation msg, part prefix (~30), tools attribution (~100), markdown expansion (~200)
        # This prevents silent truncation in update_message_streaming which has a hard limit at 3700 chars
        safety_margin = len(continuation_msg) + 330
        message_char_limit = 3700 - safety_margin  # Approximately 3335 chars
        streaming_aborted = False  # Track if we had to abort streaming due to failures

        # Start progress updater task (will be cancelled when streaming starts)
        progress_task = None
        first_chunk_received = False

        # Define the streaming callback
        async def stream_callback(text_chunk: str):
            """Callback function called with each text chunk from OpenAI"""
            nonlocal current_message_id, current_part, overflow_buffer, progress_task, first_chunk_received, streaming_aborted

            # If we've aborted, ignore further chunks
            if streaming_aborted:
                return

            # Cancel progress updater on first real chunk (not the None completion signal)
            if not first_chunk_received and text_chunk is not None:
                first_chunk_received = True
                if progress_task and not progress_task.done():
                    progress_task.cancel()
                    self.log_debug("Cancelled progress updater - streaming started")
            
            # Check if this is the completion signal (None)
            if text_chunk is None:
                # Stream is complete - flush any remaining buffered text WITHOUT loading indicator
                if buffer.has_pending_update() and rate_limiter.can_make_request():
                    self.log_info("Flushing final buffered text")
                    rate_limiter.record_request_attempt()
                    # Use raw text for final flush - no loading indicator since stream is complete
                    final_text = buffer.get_complete_text()  # No loading indicator on completion

                    # Preserve part number prefix for overflow messages in final flush
                    if current_part > 1:
                        final_text = f"*Part {current_part} (continued)*\n\n{final_text}"

                    try:
                        result = await client.update_message_streaming(message.channel_id, current_message_id, final_text)
                        if result["success"]:
                            rate_limiter.record_success()
                            buffer.mark_updated()
                    except Exception as e:
                        self.log_error(f"Error flushing final text: {e}")
                return
            
            if not text_chunk:
                return
                
            # Add chunk to buffer
            buffer.add_chunk(text_chunk)
            
            # Check if it's time to update
            if buffer.should_update() and rate_limiter.can_make_request():
                rate_limiter.record_request_attempt()
                
                # Check if we need to overflow based on RAW text (not display text)
                raw_text = buffer.get_complete_text()
                
                if len(raw_text) > message_char_limit:
                    # Find a good split point - look for paragraph or sentence breaks
                    # Start from the limit and work backwards
                    search_start = max(0, message_char_limit - 500)  # Look back up to 500 chars

                    # Priority 1: Try to find a paragraph break (double newline)
                    double_newline = raw_text.rfind('\n\n', search_start, message_char_limit)
                    if double_newline > 0:
                        split_point = double_newline + 2  # Keep the paragraph break in first part
                    else:
                        # Priority 2: Try to find end of sentence
                        last_period = raw_text.rfind('. ', search_start, message_char_limit)
                        if last_period > 0:
                            split_point = last_period + 2  # Include period and space
                        else:
                            # Priority 3: Try to find a single newline
                            last_newline = raw_text.rfind('\n', search_start, message_char_limit)
                            if last_newline > 0:
                                split_point = last_newline + 1
                            else:
                                # Priority 4: At least don't split a word
                                last_space = raw_text.rfind(' ', search_start, message_char_limit)
                                if last_space > 0:
                                    split_point = last_space + 1
                                else:
                                    # Last resort: hard cut at limit
                                    split_point = message_char_limit
                    
                    # Split the RAW text at the chosen point
                    first_part_raw = raw_text[:split_point]
                    overflow_raw = raw_text[split_point:]
                    
                    # Check if we're splitting inside a code block
                    fence_handler_temp = FenceHandler()
                    fence_handler_temp.update_text(first_part_raw)
                    was_in_code_block = fence_handler_temp.is_in_code_block()
                    language_hint = fence_handler_temp.get_current_language_hint()
                    
                    # Get display-safe version of first part (with closed fences if needed)
                    first_part_display = fence_handler_temp.get_display_safe_text()
                    
                    # Update current message with continuation indicator
                    final_first_part = f"{first_part_display}{continuation_msg}"
                    try:
                        result = await client.update_message_streaming(message.channel_id, current_message_id, final_first_part)
                        if not result["success"]:
                            # CRITICAL: Overflow update failed - retry immediately
                            self.log_warning(f"Overflow update failed: {result.get('error', 'Unknown')} - retrying")
                            await asyncio.sleep(1.0)  # Brief pause
                            result = await client.update_message_streaming(message.channel_id, current_message_id, final_first_part)
                            if not result["success"]:
                                self.log_error(f"Overflow retry failed: {result.get('error', 'Unknown')} - stopping stream")
                                # Cannot continue safely without losing data
                                streaming_aborted = True
                                # Show what we have with error notice
                                error_msg = f"{final_first_part}\n\n{config.error_emoji} *Streaming interrupted at message overflow. Partial response shown above.*"
                                try:
                                    await client.update_message_streaming(message.channel_id, current_message_id, error_msg)
                                except:
                                    pass
                                return  # Exit callback

                        if result["success"]:
                            # Prepare overflow text with proper fence opening if needed
                            if was_in_code_block:
                                # Re-open the code block on the new page
                                lang_str = language_hint if language_hint else ""
                                overflow_with_fence = f"```{lang_str}\n{overflow_raw}"
                            else:
                                overflow_with_fence = overflow_raw
                            
                            # Post a new message for overflow
                            current_part += 1
                            
                            # Create new fence handler for the continuation
                            fence_handler_continuation = FenceHandler()
                            fence_handler_continuation.update_text(overflow_with_fence)
                            continuation_display = fence_handler_continuation.get_display_safe_text()
                            
                            continuation_text = f"*Part {current_part} (continued)*\n\n{continuation_display} {config.loading_ellipse_emoji}"

                            # Send new message and get its ID
                            new_msg_result = await client.send_message_get_ts(message.channel_id, thinking_id, continuation_text)
                            if new_msg_result and new_msg_result.get("success") and "ts" in new_msg_result:
                                current_message_id = new_msg_result["ts"]
                                # Reset buffer with the properly fenced overflow content
                                buffer.reset()
                                buffer.add_chunk(overflow_with_fence)
                                buffer.mark_updated()
                                self.log_info(f"Created overflow message part {current_part}, reopened code block: {was_in_code_block}")
                            else:
                                # Couldn't get message ID due to async limitations
                                # Continue without overflow handling (message will be sent but we can't track it)
                                self.log_warning(f"Could not get message ID for overflow part {current_part} - continuing with current message")

                                # Clean up the thinking emoji from the current message before continuing
                                # The current message still has the thinking emoji and initial text,
                                # but we need to replace it with just the overflow content
                                try:
                                    clean_overflow_text = overflow_with_fence
                                    cleanup_result = await client.update_message_streaming(message.channel_id, current_message_id, f"{clean_overflow_text} {config.loading_ellipse_emoji}")
                                    if cleanup_result["success"]:
                                        self.log_info("Cleaned thinking emoji from current message after overflow failure")
                                    else:
                                        self.log_warning(f"Failed to clean thinking emoji after overflow failure: {cleanup_result.get('error', 'Unknown error')}")
                                except Exception as cleanup_error:
                                    self.log_error(f"Error cleaning thinking emoji after overflow failure: {cleanup_error}")

                                # Reset buffer but keep using current message ID
                                buffer.reset()
                                buffer.add_chunk(overflow_with_fence)
                                buffer.mark_updated()
                    except Exception as e:
                        self.log_error(f"Error handling message overflow: {e}")
                else:
                    # Normal update - get display-safe text with closed fences
                    display_text = buffer.get_display_text()

                    # Preserve part number prefix for overflow messages
                    if current_part > 1:
                        display_text_with_indicator = f"*Part {current_part} (continued)*\n\n{display_text} {config.loading_ellipse_emoji}"
                    else:
                        display_text_with_indicator = f"{display_text} {config.loading_ellipse_emoji}"

                    # Call client.update_message_streaming with indicator
                    try:
                        result = await client.update_message_streaming(message.channel_id, current_message_id, display_text_with_indicator)

                        if result["success"]:
                            rate_limiter.record_success()
                            buffer.mark_updated()
                            buffer.update_interval_setting(rate_limiter.get_current_interval())
                        else:
                            # Update failed - this is CRITICAL, we must not lose text!
                            if result["rate_limited"]:
                                # Handle rate limit response
                                if result["retry_after"]:
                                    rate_limiter.set_retry_after(result["retry_after"])
                                rate_limiter.record_failure(is_rate_limit=True)

                                # Wait and retry with the same accumulated text
                                retry_wait = result.get("retry_after", 2.0)
                                self.log_warning(f"Rate limited - waiting {retry_wait}s before retry")
                                await asyncio.sleep(retry_wait)

                                # Retry the update with the same text
                                try:
                                    retry_result = await client.update_message_streaming(message.channel_id, current_message_id, display_text_with_indicator)
                                    if retry_result["success"]:
                                        self.log_info("Retry successful after rate limit")
                                        buffer.mark_updated()
                                    else:
                                        self.log_error(f"Retry failed after rate limit: {retry_result.get('error', 'Unknown error')}")
                                        # Keep retrying with exponential backoff
                                        retry_count = 2
                                        while retry_count < 5:  # Max 5 total attempts
                                            wait_time = 2.0 * retry_count
                                            self.log_warning(f"Retry {retry_count} failed - waiting {wait_time}s before next attempt")
                                            await asyncio.sleep(wait_time)
                                            try:
                                                retry_result = await client.update_message_streaming(message.channel_id, current_message_id, display_text_with_indicator)
                                                if retry_result["success"]:
                                                    self.log_info(f"Retry {retry_count} successful")
                                                    buffer.mark_updated()
                                                    break
                                            except Exception as e:
                                                self.log_error(f"Retry {retry_count} exception: {e}")
                                            retry_count += 1

                                        if retry_count >= 5 and not retry_result.get("success"):
                                            # After 5 attempts, we really need to stop
                                            self.log_error("CRITICAL: Unable to update after 5 attempts - stopping stream")
                                            streaming_aborted = True
                                            return
                                except Exception as retry_error:
                                    self.log_error(f"Retry exception: {retry_error}")
                                    # Try a few more times with backoff
                                    retry_count = 2
                                    while retry_count < 5:
                                        wait_time = 2.0 * retry_count
                                        await asyncio.sleep(wait_time)
                                        try:
                                            retry_result = await client.update_message_streaming(message.channel_id, current_message_id, display_text_with_indicator)
                                            if retry_result["success"]:
                                                self.log_info(f"Retry {retry_count} successful after exception")
                                                buffer.mark_updated()
                                                break
                                        except:
                                            pass
                                        retry_count += 1
                            else:
                                # Non-rate-limit failure - try one immediate retry
                                rate_limiter.record_failure(is_rate_limit=False)
                                self.log_warning(f"Message update failed: {result.get('error', 'Unknown error')} - attempting retry")

                                # Immediate retry
                                try:
                                    retry_result = await client.update_message_streaming(message.channel_id, current_message_id, display_text_with_indicator)
                                    if retry_result["success"]:
                                        self.log_info("Immediate retry successful")
                                        buffer.mark_updated()
                                    else:
                                        self.log_error(f"Immediate retry failed: {retry_result.get('error', 'Unknown error')}")
                                        self.log_error(f"Immediate retry failed: {retry_result.get('error', 'Unknown error')}")
                                        # Keep retrying with exponential backoff
                                        retry_count = 2
                                        while retry_count < 5:  # Max 5 total attempts
                                            wait_time = 1.0 * retry_count  # Shorter waits for non-rate-limit
                                            self.log_warning(f"Retry {retry_count} - waiting {wait_time}s")
                                            await asyncio.sleep(wait_time)
                                            try:
                                                retry_result = await client.update_message_streaming(message.channel_id, current_message_id, display_text_with_indicator)
                                                if retry_result["success"]:
                                                    self.log_info(f"Retry {retry_count} successful")
                                                    buffer.mark_updated()
                                                    break
                                            except Exception as e:
                                                self.log_error(f"Retry {retry_count} exception: {e}")
                                            retry_count += 1

                                        if retry_count >= 5 and not retry_result.get("success"):
                                            # After 5 attempts, stop to prevent infinite loop
                                            self.log_error("CRITICAL: Unable to update after 5 attempts")
                                            streaming_aborted = True
                                            error_msg = f"{buffer.get_complete_text()}\n\n{config.error_emoji} *Streaming interrupted after multiple failures.*"
                                            try:
                                                await client.update_message_streaming(message.channel_id, current_message_id, error_msg)
                                            except:
                                                pass
                                            return
                                except Exception as retry_error:
                                    self.log_error(f"Retry exception: {retry_error}")
                                    # Try a few more times
                                    retry_count = 2
                                    while retry_count < 5:
                                        wait_time = 1.0 * retry_count
                                        await asyncio.sleep(wait_time)
                                        try:
                                            retry_result = await client.update_message_streaming(message.channel_id, current_message_id, display_text_with_indicator)
                                            if retry_result["success"]:
                                                self.log_info(f"Retry {retry_count} successful after exception")
                                                buffer.mark_updated()
                                                break
                                        except:
                                            pass
                                        retry_count += 1

                                    if retry_count >= 5:
                                        streaming_aborted = True
                                        return
                            
                    except Exception as e:
                        rate_limiter.record_failure(is_rate_limit=False)
                        self.log_error(f"Error updating streaming message: {e}")
        
        # Start progress updater before making API call
        try:
            progress_task = await self._start_progress_updater_async(
                client, message.channel_id, message_id, "request", emoji=config.thinking_emoji
            )
            self.log_debug("Started progress updater task")
        except Exception as e:
            self.log_warning(f"Failed to start progress updater: {e}")
            progress_task = None

        # Start streaming from OpenAI with the callback
        try:
            web_search_enabled = thread_config.get('enable_web_search', config.enable_web_search)
            # Determine which model to use (web search model if web search enabled)
            model = config.web_search_model or thread_config["model"] if web_search_enabled else thread_config["model"]

            # Build tools array (includes web_search and/or MCP tools based on config)
            tools = self._build_tools_array(thread_config, model)

            if tools:
                # Generate response with tools (web_search and/or MCP)
                response_text = await self.openai_client.create_streaming_response_with_tools(
                    messages=messages_for_api,
                    tools=tools,
                    stream_callback=stream_callback,
                    tool_callback=tool_callback,  # Add tool callback
                    model=model,
                    temperature=thread_config["temperature"],
                    max_tokens=thread_config["max_tokens"],
                    system_prompt=system_prompt,
                    reasoning_effort=thread_config.get("reasoning_effort"),
                    verbosity=thread_config.get("verbosity"),
                    store=False  # Match the existing behavior
                )
            else:
                # Generate response without tools
                response_text = await self.openai_client.create_streaming_response(
                    messages=messages_for_api,
                    stream_callback=stream_callback,
                    tool_callback=tool_callback,  # Add tool callback even without tools (in case of built-in tools)
                    model=thread_config["model"],
                    temperature=thread_config["temperature"],
                    max_tokens=thread_config["max_tokens"],
                    system_prompt=system_prompt,
                    reasoning_effort=thread_config.get("reasoning_effort"),
                    verbosity=thread_config.get("verbosity")
                )

            # Ensure progress updater is cancelled if still running
            if progress_task and not progress_task.done():
                progress_task.cancel()
                self.log_debug("Cancelled progress updater after API call completed")

            # Build list of tools used (unified attribution)
            tools_used = []
            if search_counts["web_search"] > 0:
                tools_used.append("web_search")
            if search_counts["mcp"] > 0:
                # Show generic "mcp" instead of listing all available servers
                # since we don't track which specific servers were actually invoked
                tools_used.append("mcp")

            # Add unified tools note at the END if any tools were used
            # This works for both paginated and non-paginated responses
            if tools_used:
                tools_note = f"\n\n_Used Tools: {', '.join(tools_used)}_"
                response_text = response_text + tools_note
                self.log_info(f"Added tools attribution: {', '.join(tools_used)}")

            # Check if streaming was aborted due to failures
            if streaming_aborted:
                self.log_error("Streaming was aborted due to update failures")
                # The error message was already shown in the callback
                # Return an error response to prevent saving incomplete data
                return Response(
                    type="error",
                    content=f"Streaming was interrupted. Partial response was shown but may be incomplete.",
                    metadata={"streaming_aborted": True}
                )

            # Safety check: ensure all text was sent AND remove loading indicator
            # Note: current_message_id might be different from message_id if we overflowed
            # We need to update the current message (which might be part 2, 3, etc)
            if current_part > 1:
                # We're on an overflow message - just remove the loading indicator
                self.log_debug(f"Removing loading indicator from part {current_part}")
                try:
                    # Get the current display text without loading indicator
                    final_part_text = buffer.get_complete_text()
                    if final_part_text:
                        # Add tools attribution to the final overflow message if tools were used
                        if tools_used:
                            tools_note = f"\n\n_Used Tools: {', '.join(tools_used)}_"
                            final_part_text = final_part_text + tools_note
                            self.log_debug(f"Added tools attribution to overflow part {current_part}")

                        # Add the part indicator
                        final_part_text = f"*Part {current_part} (continued)*\n\n{final_part_text}"
                        final_result = await client.update_message_streaming(message.channel_id, current_message_id, final_part_text)
                        if not final_result["success"]:
                            self.log_error(f"Failed to remove indicator from part {current_part}: {final_result.get('error', 'Unknown error')}")
                except Exception as e:
                    self.log_error(f"Error removing indicator from overflow message: {e}")
            else:
                # Original message - check if we need to handle any remaining text
                if response_text != buffer.last_sent_text or True:  # Always update to remove indicator
                    if response_text != buffer.last_sent_text:
                        # Calculate if mismatch is just from tools attribution being added
                        char_difference = len(response_text) - len(buffer.last_sent_text)
                        expected_attribution_length = len(tools_note) if tools_used else 0

                        # Allow ±5 char tolerance for minor formatting differences
                        is_attribution_only = abs(char_difference - expected_attribution_length) <= 5

                        if is_attribution_only:
                            # Expected mismatch from attribution - just debug log
                            self.log_debug(f"Final update includes tools attribution (+{char_difference} chars)")
                        else:
                            # Unexpected mismatch - warn about it
                            self.log_warning(f"Unexpected text mismatch after streaming - sending correction update "
                                           f"(sent: {len(buffer.last_sent_text)}, should be: {len(response_text)} chars, "
                                           f"difference: {char_difference}, expected attribution: {expected_attribution_length})")
                    else:
                        self.log_debug("Sending final update to ensure loading indicator is removed")
                    try:
                        # Handle empty response
                        if not response_text:
                            response_text = "I apologize, but I couldn't generate a response. OpenAI either didn't respond or returned an empty response. Please try again."
                            self.log_warning("Empty response detected, using fallback message")
                        
                        # Check if message is too long for a single update
                        if len(response_text) > 3900:  # Slack's approximate limit
                            # This shouldn't happen if streaming overflow worked correctly
                            # But handle it as a fallback
                            truncated_text = response_text[:3800] + "\n\n*Continued in next message...*"
                            final_result = await client.update_message_streaming(message.channel_id, message_id, truncated_text)

                            # Send the rest as new messages
                            overflow_text = response_text[3800:]
                            await client.send_message(message.channel_id, message.thread_id, f"*...continued*\n\n{overflow_text}")
                            
                            if not final_result["success"]:
                                self.log_error(f"Final truncated update failed: {final_result.get('error', 'Unknown error')}")
                        else:
                            final_result = await client.update_message_streaming(message.channel_id, current_message_id, response_text)
                            if not final_result["success"]:
                                self.log_error(f"Final correction update failed: {final_result.get('error', 'Unknown error')}")
                    except Exception as e:
                        self.log_error(f"Error in final correction update: {e}")
            
            # Note: To properly detect if web search was used, we'd need to track
            # tool events during streaming. The presence of URLs doesn't mean web search was used.
            
            # Add assistant response to thread state
            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
            self._add_message_with_token_management(thread_state, "assistant", response_text, db=self.db, thread_key=thread_key)
            
            # Schedule async cleanup after response
            cleanup_coro = self._async_post_response_cleanup(thread_state, thread_key)
            self._schedule_async_call(cleanup_coro)
            
            # Log streaming stats
            stats = rate_limiter.get_stats()
            buffer_stats = buffer.get_stats()
            self.log_info(f"Streaming completed: {stats['successful_requests']}/{stats['total_requests']} updates, "
                         f"final length: {buffer_stats['text_length']} chars")
            
            return Response(
                type="text",
                content=response_text,
                metadata={"streamed": True, "message_id": message_id}
            )
            
        except Exception as e:
            # Check if this is an MCP connection error first (before logging)
            # Streaming throws APIError, non-streaming throws APIStatusError with code 424
            failed_mcp_server = None
            error_msg = str(e)

            # Check for MCP server failure in error message
            if "MCP server" in error_msg and ("404" in error_msg or "424" in error_msg):
                # Extract MCP server name from error message pattern
                # Example: "Error retrieving tool list from MCP server: 'context7'"
                match = re.search(r"MCP server: '([^']+)'", error_msg)
                if match:
                    failed_mcp_server = match.group(1)
                    # Log MCP failures at INFO level - they're handled gracefully
                    self.log_info(f"MCP server '{failed_mcp_server}' unavailable - retrying request without it")
            else:
                # Unexpected errors - log as ERROR
                self.log_error(f"Error in streaming response generation: {e}")

            # Ensure progress updater is cancelled on error
            if progress_task and not progress_task.done():
                progress_task.cancel()
                self.log_debug("Cancelled progress updater due to error")

            # Try to remove the loading indicator if we had a message_id
            if message_id and hasattr(client, 'update_message_streaming'):
                try:
                    # Send whatever text we have without the loading indicator, or a formatted error message
                    if buffer.has_content():
                        error_text = buffer.get_complete_text()
                    else:
                        if failed_mcp_server:
                            error_text = f"{config.error_emoji} *MCP Connection Failed*\n\nCouldn't connect to MCP server '{failed_mcp_server}'. Retrying with other tools..."
                        else:
                            error_text = f"{config.error_emoji} *OpenAI Stream Interrupted*\n\nOpenAI's streaming response was interrupted. I'll try again without streaming..."
                    await client.update_message_streaming(message.channel_id, message_id, error_text)
                except Exception as cleanup_error:
                    self.log_debug(f"Could not remove loading indicator: {cleanup_error}")

            # Fall back to non-streaming on error
            self.log_info("Falling back to non-streaming due to error")

            # Remove the message that was just added by streaming attempt
            # to prevent duplicates when fallback adds it again
            if thread_state.messages and thread_state.messages[-1].get("role") == "user":
                thread_state.messages.pop()
                self.log_debug("Removed duplicate user message before fallback")

            # Pass retry_count=1 to prevent re-entering streaming after timeout
            # Also pass failed_mcp_server so fallback can exclude it from tools
            return await self._handle_text_response(
                user_content, thread_state, client, message, thinking_id,
                attachment_urls, retry_count=1, failed_mcp_server=failed_mcp_server
            )

    def _build_tools_array(self, thread_config: dict, model: str,
                           exclude_mcp_server: Optional[str] = None) -> Optional[List[dict]]:
        """
        Build tools array for OpenAI API based on user preferences and model.

        Includes:
        - web_search if enabled in user preferences
        - MCP tools if enabled AND model is GPT-5 AND MCP servers are configured

        Args:
            thread_config: Thread configuration with user preferences
            model: Model being used for the request
            exclude_mcp_server: Optional MCP server label to exclude (e.g., if it failed)

        Returns:
            List of tool definitions, or None if no tools enabled
        """
        tools = []

        # Add web_search if enabled
        web_search_enabled = thread_config.get('enable_web_search', config.enable_web_search)
        if web_search_enabled:
            tools.append({"type": "web_search"})
            self.log_debug("Added web_search to tools array")

        # Add MCP tools if enabled AND model is GPT-5 AND MCP servers configured
        mcp_enabled = thread_config.get('enable_mcp', config.mcp_enabled_default)
        if mcp_enabled and model.startswith('gpt-5') and self.mcp_manager.has_mcp_servers():
            mcp_tools = self.mcp_manager.get_tools_for_openai()

            # Filter out excluded MCP server if specified
            if exclude_mcp_server:
                mcp_tools = [tool for tool in mcp_tools
                           if tool.get("server_label") != exclude_mcp_server]
                self.log_info(f"Excluded failed MCP server '{exclude_mcp_server}' from tools array")

            tools.extend(mcp_tools)
            self.log_debug(f"Added {len(mcp_tools)} MCP server(s) to tools array")

        # Return None if no tools, otherwise return the list
        if not tools:
            return None

        return tools
