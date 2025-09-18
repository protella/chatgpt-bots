from __future__ import annotations

import asyncio
from typing import Optional

import time

from base_client import BaseClient, Message, Response
from config import config
from streaming import RateLimitManager, StreamingBuffer


class ImageGenerationMixin:
    async def _handle_image_generation(self, prompt: str, thread_state, client: BaseClient,
                                channel_id: str, thinking_id: Optional[str], message: Message,
                                skip_enhancement: bool = False) -> Response:
        """Handle image generation request with streaming enhancement
        
        Args:
            skip_enhancement: If True, skip prompt enhancement (used when falling back from failed edit)
        """
        self.log_info(f"Generating image for prompt: {prompt[:100]}...")
        
        # Initialize response metadata
        response_metadata = {}
        
        # Get thread config (with user preferences)
        thread_config = config.get_thread_config(
            overrides=thread_state.config_overrides,
            user_id=message.user_id,
            db=self.db
        )
        
        # Inject stored image analyses for style consistency
        enhanced_messages = self._inject_image_analyses(thread_state.messages, thread_state)
        
        # Pre-trim messages to fit within context window
        enhanced_messages = await self._pre_trim_messages_for_api(enhanced_messages)
        
        # Check if streaming is supported (respecting user prefs) and enhancement is needed
        streaming_enabled = thread_config.get('enable_streaming', config.enable_streaming)
        if hasattr(client, 'supports_streaming') and client.supports_streaming() and streaming_enabled and not skip_enhancement:
            # Stream the enhancement to user with proper rate limiting
            streaming_config = client.get_streaming_config() if hasattr(client, 'get_streaming_config') else {}
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
            
            
            def enhancement_callback(chunk: str):
                buffer.add_chunk(chunk)
                # Only update if both buffer says it's time AND rate limiter allows it
                if buffer.should_update() and rate_limiter.can_make_request():
                    rate_limiter.record_request_attempt()
                    display_text = f"*Enhanced Prompt:* ✨ _{buffer.get_complete_text()}_ {config.loading_ellipse_emoji}"
                    
                    result = self._update_message_streaming_sync(client, channel_id, thinking_id, display_text)
                    
                    if result["success"]:
                        rate_limiter.record_success()
                        buffer.mark_updated()
                        buffer.update_interval_setting(rate_limiter.get_current_interval())
                    else:
                        if result["rate_limited"]:
                            if result["retry_after"]:
                                rate_limiter.set_retry_after(result["retry_after"])
                            rate_limiter.record_failure(is_rate_limit=True)
                            
                            # Check if circuit breaker opened
                            if not rate_limiter.is_streaming_enabled():
                                self.log_warning("Image enhancement rate limited - circuit breaker opened")
                        else:
                            rate_limiter.record_failure(is_rate_limit=False)
            
            # Enhance prompt with streaming (returns the complete enhanced text)
            enhanced_prompt = await self.openai_client._enhance_image_prompt(
                prompt=prompt,
                conversation_history=enhanced_messages,
                stream_callback=enhancement_callback
            )
            
            # Show the final enhanced prompt
            if enhanced_prompt and thinking_id:
                enhanced_text = f"*Enhanced Prompt:* ✨ _{enhanced_prompt}_"
                self._update_message_streaming_sync(client, channel_id, thinking_id, enhanced_text)
                # Mark that we should NOT touch this message again
                response_metadata["prompt_message_id"] = thinking_id
            
            # Create a NEW message for generating status - don't touch the enhanced prompt!
            generating_id = await client.send_thinking_indicator(channel_id, thread_state.thread_ts)
            self._update_status(client, channel_id, generating_id, 
                              "Generating image. This may take a minute...", 
                              emoji=config.circle_loader_emoji)
            # Track the status message ID
            response_metadata["status_message_id"] = generating_id
            
            # Generate image with already-enhanced prompt
            try:
                image_data = await self.openai_client.generate_image(
                    prompt=enhanced_prompt,
                    size=thread_config.get("image_size"),
                    quality=thread_config.get("image_quality"),
                    enhance_prompt=False,  # Already enhanced!
                    conversation_history=None  # Not needed since we enhanced already
                )
            except Exception as e:
                error_str = str(e)
                if "moderation_blocked" in error_str or "safety system" in error_str or "content policy" in error_str.lower():
                    # Clean up status message
                    if "status_message_id" in response_metadata:
                        if hasattr(client, 'delete_message'):
                            client.delete_message(channel_id, response_metadata["status_message_id"])
                    
                    # Provide friendly message
                    moderation_msg = (
                        "I couldn't generate that image as it was flagged by content safety filters. "
                        "This can happen with certain brand names, people, or other protected content. "
                        "Try rephrasing your request or describing what you want without using specific names."
                    )
                    
                    # Add to thread for context
                    thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
                    message_ts = message.metadata.get("ts") if message.metadata else None
                    formatted_prompt = self._format_user_content_with_username(prompt, message)
                    self._add_message_with_token_management(thread_state, "user", formatted_prompt, db=self.db, thread_key=thread_key, message_ts=message_ts)
                    self._add_message_with_token_management(thread_state, "assistant", moderation_msg, db=self.db, thread_key=thread_key)
                    
                    return Response(
                        type="text",
                        content=moderation_msg
                    )
                else:
                    # Re-raise other errors
                    raise
            
            # After image is generated, update message to show just the enhanced prompt
            # (remove the "Generating image..." status) - but respect rate limits
            if enhanced_prompt and thinking_id and rate_limiter.can_make_request():
                result = self._update_message_streaming_sync(client, channel_id, thinking_id, f"*Enhanced Prompt:* ✨ _{enhanced_prompt}_")
                if result["success"]:
                    rate_limiter.record_success()
                else:
                    if result["rate_limited"]:
                        rate_limiter.record_failure(is_rate_limit=True)
                        self.log_debug("Couldn't clean up image gen status due to rate limit")
        else:
            # Non-streaming fallback or skip enhancement
            if not skip_enhancement:
                self._update_status(client, channel_id, thinking_id, "Enhancing your prompt...")
            
            # Generate image with or without enhancement
            self._update_status(client, channel_id, thinking_id, "Creating your image. This may take a minute...", emoji=config.circle_loader_emoji)
            
            try:
                image_data = await self.openai_client.generate_image(
                    prompt=prompt,
                    size=thread_config.get("image_size"),
                    quality=thread_config.get("image_quality"),
                    enhance_prompt=not skip_enhancement,  # Skip if fallback from failed edit
                    conversation_history=enhanced_messages if not skip_enhancement else None  # Only pass if enhancing
                )
            except Exception as e:
                error_str = str(e)
                if "moderation_blocked" in error_str or "safety system" in error_str or "content policy" in error_str.lower():
                    # Provide friendly message
                    moderation_msg = (
                        "I couldn't generate that image as it was flagged by content safety filters. "
                        "This can happen with certain brand names, people, or other protected content. "
                        "Try rephrasing your request or describing what you want without using specific names."
                    )
                    
                    # Add to thread for context
                    thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
                    message_ts = message.metadata.get("ts") if message.metadata else None
                    formatted_prompt = self._format_user_content_with_username(prompt, message)
                    self._add_message_with_token_management(thread_state, "user", formatted_prompt, db=self.db, thread_key=thread_key, message_ts=message_ts)
                    self._add_message_with_token_management(thread_state, "assistant", moderation_msg, db=self.db, thread_key=thread_key)
                    
                    return Response(
                        type="text",
                        content=moderation_msg
                    )
                else:
                    # Re-raise other errors
                    raise
        
        # Store in asset ledger
        asset_ledger = self.thread_manager.get_or_create_asset_ledger(thread_state.thread_ts)
        thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
        asset_ledger.add_image(
            image_data.base64_data,
            image_data.prompt,  # Use the enhanced prompt
            time.time(),
            db=self.db,
            thread_id=thread_key
        )
        
        # Add breadcrumb to thread state with metadata
        thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
        message_ts = message.metadata.get("ts") if message.metadata else None
        formatted_prompt = self._format_user_content_with_username(prompt, message)
        self._add_message_with_token_management(thread_state, "user", formatted_prompt, db=self.db, thread_key=thread_key, message_ts=message_ts)
        # Store enhanced prompt with metadata for new image tracking
        self._add_message_with_token_management(thread_state, 
            "assistant", 
            image_data.prompt,  # Just the enhanced prompt, no "Generated image:" prefix
            db=self.db, 
            thread_key=thread_key,
            metadata={
                "type": "image_generation",
                "prompt": image_data.prompt,
                "url": None  # Will be updated after upload
            }
        )
        
        # Mark as streamed if we used streaming for the enhancement
        # Note: response_metadata already initialized and populated above
        if hasattr(client, 'supports_streaming') and client.supports_streaming() and streaming_enabled:
            response_metadata["streamed"] = True
            
        return Response(
            type="image",
            content=image_data,
            metadata=response_metadata
        )

    def _update_thinking_for_image(self, client: BaseClient, channel_id: str, thinking_id: str):
        """Update the thinking indicator to show image generation message"""
        self._update_status(client, channel_id, thinking_id, 
                          "Generating image. This may take a minute, please wait...",
                          emoji=config.circle_loader_emoji)
