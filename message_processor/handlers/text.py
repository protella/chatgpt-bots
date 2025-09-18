from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from base_client import BaseClient, Message, Response
from config import config
from streaming import FenceHandler, RateLimitManager, StreamingBuffer


class TextHandlerMixin:
    async def _handle_text_response(self, user_content: Any, thread_state, client: BaseClient,
                              message: Message, thinking_id: Optional[str] = None,
                              attachment_urls: Optional[List[str]] = None,
                              retry_count: int = 0) -> Response:
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
        
        # Check if web search should be available (respecting user prefs)
        web_search_enabled = thread_config.get('enable_web_search', config.enable_web_search)
        if web_search_enabled:
            # Use web search model if specified, otherwise use thread config model
            model = config.web_search_model or thread_config["model"]
            
            # Generate response with web search tool available
            response_text = await self.openai_client.create_text_response_with_tools(
                messages=messages_for_api,
                tools=[{"type": "web_search"}],
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
            response_text = await self.openai_client.create_text_response(
                messages=messages_for_api,
                model=thread_config["model"],
                temperature=thread_config["temperature"],
                max_tokens=thread_config["max_tokens"],
                system_prompt=system_prompt,
                reasoning_effort=thread_config.get("reasoning_effort"),
                verbosity=thread_config.get("verbosity")
            )
        
        # Check if response used web search and add citation note
        if web_search_enabled:
            # Look for indicators that web search was used
            # OpenAI typically includes numbered citations [1], [2] or URLs when web search is used
            if any(marker in response_text for marker in ["[1]", "[2]", "[3]", "http://", "https://"]):
                self.log_info("Response includes web search results")
        
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
            "image_generation": False
        }
        
        # Track search counts
        search_counts = {
            "web_search": 0,
            "file_search": 0
        }

        # Define tool event callback
        async def tool_callback(tool_type: str, status: str):
            """Handle tool events for status updates"""
            if status == "started":
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
        message_char_limit = 3700  # Leave room for indicators
        
        # Define the streaming callback
        async def stream_callback(text_chunk: str):
            """Callback function called with each text chunk from OpenAI"""
            nonlocal current_message_id, current_part, overflow_buffer
            
            # Check if this is the completion signal (None)
            if text_chunk is None:
                # Stream is complete - flush any remaining buffered text WITHOUT loading indicator
                if buffer.has_pending_update() and rate_limiter.can_make_request():
                    self.log_info("Flushing final buffered text")
                    rate_limiter.record_request_attempt()
                    # Use raw text for final flush - no loading indicator since stream is complete
                    final_text = buffer.get_complete_text()  # No loading indicator on completion
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
                    # Find a good split point - preferably at a line break
                    split_point = message_char_limit
                    
                    # Look backwards for the last newline before the limit
                    last_newline = raw_text.rfind('\n', 0, message_char_limit)
                    
                    # If we found a newline within reasonable distance (not too far back)
                    # Use it as the split point to avoid breaking lines
                    if last_newline > message_char_limit - 500 and last_newline > 0:
                        split_point = last_newline + 1  # Include the newline in first part
                    else:
                        # No good newline found, try to at least avoid splitting words
                        # Look for last space before the limit
                        last_space = raw_text.rfind(' ', max(0, message_char_limit - 100), message_char_limit)
                        if last_space > 0:
                            split_point = last_space + 1
                    
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
                    final_first_part = f"{first_part_display}\n\n*Continued in next message...*"
                    try:
                        result = await client.update_message_streaming(message.channel_id, current_message_id, final_first_part)
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
                    display_text_with_indicator = f"{display_text} {config.loading_ellipse_emoji}"
                    
                    # Call client.update_message_streaming with indicator
                    try:
                        result = await client.update_message_streaming(message.channel_id, current_message_id, display_text_with_indicator)

                        if result["success"]:
                            rate_limiter.record_success()
                            buffer.mark_updated()
                            buffer.update_interval_setting(rate_limiter.get_current_interval())
                        else:
                            if result["rate_limited"]:
                                # Handle rate limit response
                                if result["retry_after"]:
                                    rate_limiter.set_retry_after(result["retry_after"])
                                rate_limiter.record_failure(is_rate_limit=True)
                                
                                # Check if we should fall back to non-streaming
                                if not rate_limiter.is_streaming_enabled():
                                    self.log_warning("Circuit breaker opened - will complete without further streaming")
                                    # Try to clear the thinking indicator
                                    try:
                                        clear_text = buffer.last_sent_text if buffer.last_sent_text else "Processing..."
                                        if len(clear_text) > 3900:
                                            clear_text = clear_text[:3800] + "\n\n*Response too long - see next message*"
                                        await client.update_message_streaming(message.channel_id, message_id, clear_text)
                                    except Exception as clear_error:
                                        self.log_error(f"Failed to clear indicator after circuit break: {clear_error}")
                            else:
                                rate_limiter.record_failure(is_rate_limit=False)
                                self.log_warning(f"Message update failed: {result.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        rate_limiter.record_failure(is_rate_limit=False)
                        self.log_error(f"Error updating streaming message: {e}")
        
        # Start streaming from OpenAI with the callback
        try:
            web_search_enabled = thread_config.get('enable_web_search', config.enable_web_search)
            if web_search_enabled:
                # Use web search model if specified, otherwise use thread config model
                model = config.web_search_model or thread_config["model"]
                
                # Generate response with web search tool available
                response_text = await self.openai_client.create_streaming_response_with_tools(
                    messages=messages_for_api,
                    tools=[{"type": "web_search"}],
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
                        self.log_warning(f"Text mismatch after streaming - sending correction update "
                                       f"(sent: {len(buffer.last_sent_text)}, should be: {len(response_text)} chars)")
                    else:
                        self.log_debug("Sending final update to ensure loading indicator is removed")
                    try:
                        # Handle empty response
                        if not response_text:
                            response_text = "I apologize, but I wasn't able to generate a response. Please try again."
                            self.log_warning("Empty response detected, using fallback message")
                        
                        # Check if message is too long for a single update
                        if len(response_text) > 3900:  # Slack's approximate limit
                            # This shouldn't happen if streaming overflow worked correctly
                            # But handle it as a fallback
                            truncated_text = response_text[:3800] + "\n\n*Continued in next message...*"
                            final_result = await client.update_message_streaming(message.channel_id, message_id, truncated_text)

                            # Send the rest as new messages
                            overflow_text = response_text[3800:]
                            await client.send_message(message.channel_id, thinking_id, f"*...continued*\n\n{overflow_text}")
                            
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
            self.log_error(f"Error in streaming response generation: {e}")
            
            # Try to remove the loading indicator if we had a message_id
            if message_id and hasattr(client, 'update_message_streaming'):
                try:
                    # Send whatever text we have without the loading indicator, or a formatted error message
                    if buffer.has_content():
                        error_text = buffer.get_complete_text()
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
            return await self._handle_text_response(user_content, thread_state, client, message, thinking_id, attachment_urls, retry_count=1)
