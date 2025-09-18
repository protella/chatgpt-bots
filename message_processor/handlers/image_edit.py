from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

import re
import time

from base_client import BaseClient, Message, Response
from config import config
from prompts import IMAGE_ANALYSIS_PROMPT


class ImageEditMixin:
    async def _find_target_image(self, user_text: str, thread_state, client: BaseClient) -> Optional[str]:
        """Find the target image URL based on user's reference using DB"""
        thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
        
        # Get all images from DB for this thread
        all_available_images = []
        if self.db:
            db_images = self.db.find_thread_images(thread_key)
            for img in db_images:
                all_available_images.append({
                    "url": img.get("url"),
                    "description": img.get("prompt", "") or "uploaded image",
                    "type": img.get("image_type", "uploaded"),
                    "analysis": img.get("analysis", "")  # Include analysis for natural language matching
                })
        
        # Fallback to asset ledger if no DB or no images found
        if not all_available_images:
            image_registry = self._extract_image_registry(thread_state)
            for img in image_registry:
                all_available_images.append({
                    "url": img["url"],
                    "description": img["description"],
                    "type": "generated",
                    "analysis": ""
                })
        
        if not all_available_images:
            return None
        
        # If only one image, use it
        if len(all_available_images) == 1:
            self.log_debug(f"Only one image found, using it: {all_available_images[0]['url']}")
            return all_available_images[0]["url"]
        
        # If ambiguous and multiple images, use utility model to match
        # Let natural language processing handle ALL references including ordinals
        if len(all_available_images) > 1:
            # Build context for matching using DB analyses - FULL context per CLAUDE.md
            context = "Available images:\n"
            for i, img in enumerate(all_available_images, 1):
                context += f"{i}. {img['type'].capitalize()} image"
                if img['description']:
                    context += f": {img['description']}"  # Full description, no truncation
                context += "\n"
                
                # Include analysis if available for natural language matching
                if img.get('analysis'):
                    # Include FULL visual analysis for better matching - no truncation per CLAUDE.md
                    context += f"   Visual details: {img['analysis']}\n"
            
            context += f"\nUser reference: '{user_text}'\n"
            context += "Which image number best matches the user's reference? Respond with just the number."
            
            try:
                # Use utility model to find best match
                match_response = await self.openai_client.create_text_response(
                    messages=[{"role": "user", "content": context}],
                    model=config.utility_model,
                    temperature=0.1,
                    max_tokens=50,  # Increased to handle reasoning tokens
                    reasoning_effort=config.utility_reasoning_effort,  # Use utility config
                    verbosity=config.utility_verbosity  # Use utility config
                )
                
                # Parse response to get index
                numbers = re.findall(r'\d+', match_response)
                if numbers:
                    index = int(numbers[0]) - 1  # Convert to 0-based
                    if 0 <= index < len(all_available_images):
                        url = all_available_images[index]["url"]
                        self.log_debug(f"Utility model matched to image {index + 1}: {url}")
                        return url
            except Exception as e:
                self.log_warning(f"Failed to match image reference: {e}")
        
        # Default to most recent image (last in list)
        url = all_available_images[-1]["url"]
        image_type = all_available_images[-1]["type"]
        self.log_debug(f"Using most recent {image_type} image by default: {url}")
        return url

    async def _handle_image_modification(
        self,
        text: str,
        thread_state,
        thread_id: str,
        client: BaseClient,
        channel_id: str,
        thinking_id: Optional[str],
        message: Message
    ) -> Response:
        """Handle image modification request by finding and editing the target image"""
        # Don't update status yet - we might fall back to generation
        
        # Initialize response metadata early to track status messages
        response_metadata = {}
        
        # Try to find target image URL from conversation
        target_url = await self._find_target_image(text, thread_state, client)
        
        if target_url:
            # Found an image to edit - update status
            self._update_status(client, channel_id, thinking_id, "Finding the image to edit...", emoji=config.web_search_emoji)
            
            # Download the image from Slack
            self.log_info(f"Found target image URL: {target_url}")
            self._update_status(client, channel_id, thinking_id, "Downloading the image...")
            
            try:
                # Download the image
                image_data = await client.download_file(target_url, None)
                
                if image_data:
                    # Convert to base64 for editing
                    import base64
                    base64_data = base64.b64encode(image_data).decode('utf-8')
                    
                    # Analyze the image first
                    self.log_debug("Analyzing image for context")
                    self._update_status(client, channel_id, thinking_id, "Analyzing the image...", emoji=config.analyze_emoji)
                    
                    image_description = await self.openai_client.analyze_images(
                        images=[base64_data],
                        question=IMAGE_ANALYSIS_PROMPT,
                        detail="high"
                    )
                    
                    # Prepare for edit
                    self.log_info(f"Editing existing image with request: {text}")
                    
                    # Get thread config (with user preferences)
                    thread_config = config.get_thread_config(
                        overrides=thread_state.config_overrides,
                        user_id=message.user_id,
                        db=self.db
                    )
                    
                    # Inject stored image analyses for style matching
                    enhanced_messages = self._inject_image_analyses(thread_state.messages, thread_state)
                    
                    # Pre-trim messages to fit within context window
                    enhanced_messages = await self._pre_trim_messages_for_api(enhanced_messages, model=thread_state.current_model)
                    
                    # Check if streaming is supported for enhancement
                    streaming_enabled = thread_config.get('enable_streaming', config.enable_streaming)
                    if hasattr(client, 'supports_streaming') and client.supports_streaming() and streaming_enabled:
                        # Stream the enhancement to user with proper rate limiting
                        from streaming.buffer import StreamingBuffer
                        from streaming import RateLimitManager
                        
                        streaming_config = client.get_streaming_config() if hasattr(client, 'get_streaming_config') else {}
                        buffer = StreamingBuffer(
                            update_interval=streaming_config.get("update_interval", 2.0),
                            buffer_size_threshold=streaming_config.get("buffer_size", 500),
                            min_update_interval=streaming_config.get("min_interval", 1.0)
                        )
                        
                        rate_limiter = RateLimitManager(
                            base_interval=streaming_config.get("update_interval", 2.0),
                            failure_threshold=streaming_config.get("circuit_breaker_threshold", 5),
                            cooldown_seconds=streaming_config.get("circuit_breaker_cooldown", 60)
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
                                            self.log_warning("Image edit enhancement rate limited - circuit breaker opened")
                                    else:
                                        rate_limiter.record_failure(is_rate_limit=False)
                        
                        # First enhance the prompt with streaming
                        enhanced_edit_prompt = await self.openai_client._enhance_image_edit_prompt(
                            user_request=text,
                            image_description=image_description,
                            conversation_history=enhanced_messages,
                            stream_callback=enhancement_callback
                        )
                        
                        # Show the final enhanced prompt
                        if enhanced_edit_prompt and thinking_id:
                            enhanced_text = f"*Enhanced Prompt:* ✨ _{enhanced_edit_prompt}_"
                            self._update_message_streaming_sync(client, channel_id, thinking_id, enhanced_text)
                            # Mark that we should NOT touch this message again
                            response_metadata["prompt_message_id"] = thinking_id
                        
                        # Create a NEW message for editing status - don't touch the enhanced prompt!
                        editing_id = await client.send_thinking_indicator(channel_id, thread_state.thread_ts)
                        self._update_status(client, channel_id, editing_id, 
                                          "Editing your image. This may take a minute...", 
                                          emoji=config.circle_loader_emoji)
                        # Track the status message ID
                        response_metadata["status_message_id"] = editing_id
                        
                        # Mark as streamed for main.py
                        response_metadata["streamed"] = True
                        
                        # Use the edit_image API with the pre-enhanced prompt
                        edited_image = await self.openai_client.edit_image(
                            input_images=[base64_data],
                            prompt=enhanced_edit_prompt,
                            image_description=None,  # Already used for enhancement
                            input_mimetypes=["image/png"],
                            input_fidelity=thread_config.get("input_fidelity", "high"),
                            background=thread_config.get("image_background", "auto"),
                            output_format=thread_config.get("image_format", "png"),
                            output_compression=thread_config.get("image_compression", 100),
                            enhance_prompt=False,  # Already enhanced!
                            conversation_history=None  # Not needed since we enhanced already
                        )
                    else:
                        # Non-streaming fallback
                        self._update_status(client, channel_id, thinking_id, "Enhancing your edit request...")
                        self._update_status(client, channel_id, thinking_id, "Editing your image. This may take a minute...", emoji=config.circle_loader_emoji)
                        
                        edited_image = await self.openai_client.edit_image(
                            input_images=[base64_data],
                            prompt=text,
                            image_description=image_description,
                            input_mimetypes=["image/png"],
                            input_fidelity=thread_config.get("input_fidelity", "high"),
                            background=thread_config.get("image_background", "auto"),
                            output_format=thread_config.get("image_format", "png"),
                            output_compression=thread_config.get("image_compression", 100),
                            enhance_prompt=True,
                            conversation_history=enhanced_messages  # Pass enhanced conversation with image analyses
                        )
                    
                    # Add breadcrumbs with metadata
                    thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
                    message_ts = message.metadata.get("ts") if message.metadata else None
                    formatted_text = self._format_user_content_with_username(text, message)
                    self._add_message_with_token_management(thread_state, "user", formatted_text, db=self.db, thread_key=thread_key, message_ts=message_ts)
                    self._add_message_with_token_management(thread_state, 
                        "assistant", 
                        edited_image.prompt,
                        db=self.db, 
                        thread_key=thread_key,
                        metadata={
                            "type": "image_edit",
                            "prompt": edited_image.prompt,
                            "url": None  # Will be updated after upload
                        }
                    )
                    
                    return Response(
                        type="image",
                        content=edited_image,
                        metadata=response_metadata
                    )
                else:
                    self.log_warning(f"Failed to download image from URL: {target_url}")
                    
            except Exception as e:
                self.log_error(f"Error editing image from URL: {e}")
                
                # Check if this is a moderation block
                error_str = str(e)
                if "moderation_blocked" in error_str or "safety system" in error_str or "content policy" in error_str.lower():
                    # Moderation block - provide friendly message and don't fall back
                    self.log_warning(f"Edit blocked by moderation: {text}")
                    
                    # Clean up any status messages
                    if "status_message_id" in response_metadata:
                        if hasattr(client, 'delete_message'):
                            client.delete_message(channel_id, response_metadata["status_message_id"])
                    
                    # Add user-friendly explanation to thread
                    moderation_msg = (
                        "I couldn't complete that edit request as it was flagged by content safety filters. "
                        "This can happen with certain brand names, people, or other protected content. "
                        "Try rephrasing your request or describing what you want without using specific names."
                    )
                    
                    # Add messages to thread for context
                    thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
                    message_ts = message.metadata.get("ts") if message.metadata else None
                    formatted_text = self._format_user_content_with_username(text, message)
                    self._add_message_with_token_management(thread_state, "user", formatted_text, db=self.db, thread_key=thread_key, message_ts=message_ts)
                    self._add_message_with_token_management(thread_state, "assistant", moderation_msg, db=self.db, thread_key=thread_key)
                    
                    return Response(
                        type="text",
                        content=moderation_msg
                    )
                
                # Other errors - clean up and fall back to generation
                if "status_message_id" in response_metadata:
                    # Delete the "Editing your image..." message
                    if hasattr(client, 'delete_message'):
                        client.delete_message(channel_id, response_metadata["status_message_id"])
                elif thinking_id and not response_metadata.get("prompt_message_id"):
                    # Update the thinking message to show we're generating instead
                    self._update_status(client, channel_id, thinking_id, 
                                      "Edit failed, generating new image instead...", 
                                      emoji=config.error_emoji)
        
        # Fallback to generation if edit failed or no URL found
        self.log_info("No image URL found or edit failed, falling back to generation based on description")
        
        # Look for image descriptions in history
        image_registry = self._extract_image_registry(thread_state)
        if image_registry:
            # Use the most recent image description without re-enhancement
            # The prompt already contains the edit request, just generate based on it
            return await self._handle_image_generation(text, thread_state, client, channel_id, thinking_id, message, 
                                                skip_enhancement=True)
        else:
            # No previous images, treat as new generation
            return await self._handle_image_generation(text, thread_state, client, channel_id, thinking_id, message)

    async def _handle_image_edit(
        self,
        text: str,
        image_inputs: List[Dict],
        thread_state,
        client: BaseClient,
        channel_id: str,
        thinking_id: Optional[str],
        attachment_urls: Optional[List[str]] = None,
        message: Message = None
    ) -> Response:
        """Handle image editing with uploaded images"""
        self._update_status(client, channel_id, thinking_id, "Processing uploaded images...")
        
        # Extract base64 data and mime types from image inputs
        input_images = []
        input_mimetypes = []
        for img_input in image_inputs:
            if img_input.get("type") == "input_image":
                # Extract from data URL format
                image_url = img_input.get("image_url", "")
                if image_url.startswith("data:"):
                    # Parse data URL: data:image/png;base64,xxxxx
                    parts = image_url.split(",", 1)
                    if len(parts) == 2:
                        header, base64_data = parts
                        # Extract mimetype from header
                        mimetype_part = header.split(";")[0].replace("data:", "")
                        mimetype = mimetype_part if mimetype_part else "image/png"
                        
                        # OpenAI doesn't support GIF for editing, convert to PNG
                        if mimetype == "image/gif":
                            self.log_warning("Converting GIF to PNG for image edit (GIF not supported)")
                            mimetype = "image/png"
                        
                        input_images.append(base64_data)
                        input_mimetypes.append(mimetype)
        
        if not input_images:
            # Shouldn't happen but fallback to generation
            return await self._handle_image_generation(text, thread_state, client, channel_id, thinking_id, message)
        
        self.log_info(f"Editing {len(input_images)} uploaded image(s)")
        
        # First, analyze the uploaded images to get context
        self.log_debug("Analyzing uploaded images for context")
        # Update status with proper pluralization
        if len(input_images) == 1:
            status_msg = "Analyzing your uploaded image..."
        else:
            status_msg = f"Analyzing {len(input_images)} uploaded images..."
        self._update_status(client, channel_id, thinking_id, status_msg, emoji=config.analyze_emoji)
        
        # Log the analysis prompt
        print("\n" + "="*100)
        print("DEBUG: IMAGE EDIT FLOW - STEP 1: ANALYZE IMAGE")
        print("="*100)
        print(f"Analysis Question: {IMAGE_ANALYSIS_PROMPT}")
        print("="*100)
        
        try:
            # Analyze the images to understand what's in them
            image_description = await self.openai_client.analyze_images(
                images=input_images,
                question=IMAGE_ANALYSIS_PROMPT,
                detail="high"
            )
            
            # Log the full analysis result
            print("\n" + "="*100)
            print("DEBUG: IMAGE EDIT FLOW - STEP 2: ANALYSIS RESULT")
            print("="*100)
            print(f"Image Description (Full):\n{image_description}")
            print("="*100)
            
            # Don't show the analysis - just update status to show we're editing
            # The analysis is only used internally for better edit quality
            if thinking_id:
                self._update_status(client, channel_id, thinking_id, "Editing your image. This may take a minute...", emoji=config.circle_loader_emoji)
            
            # Store the description and user request separately for clean enhancement
            image_analysis = image_description
            user_edit_request = text
            
        except Exception as e:
            self.log_warning(f"Failed to analyze images, continuing without context: {e}")
            image_analysis = None
            user_edit_request = text
            print("\n" + "="*100)
            print("DEBUG: IMAGE EDIT FLOW - ANALYSIS FAILED")
            print("="*100)
            print(f"Error: {e}")
            print(f"Falling back to user prompt only: {text}")
            print("="*100)
        
        # Get thread config for settings (with user preferences)
        thread_config = config.get_thread_config(
            overrides=thread_state.config_overrides,
            user_id=message.user_id,
                db=self.db
        )
        
        # Inject stored image analyses for style matching
        enhanced_messages = self._inject_image_analyses(thread_state.messages, thread_state) if thread_state.messages else None
        
        # Pre-trim messages to fit within context window if we have messages
        if enhanced_messages:
            enhanced_messages = await self._pre_trim_messages_for_api(enhanced_messages, model=thread_state.current_model)
        
        # Check if streaming is supported for enhancement
        response_metadata = {}
        streaming_enabled = thread_config.get('enable_streaming', config.enable_streaming)
        if hasattr(client, 'supports_streaming') and client.supports_streaming() and streaming_enabled:
            # Stream the enhancement to user with proper rate limiting
            from streaming.buffer import StreamingBuffer
            from streaming import RateLimitManager
            
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
                                self.log_warning("Image edit enhancement rate limited - circuit breaker opened")
                        else:
                            rate_limiter.record_failure(is_rate_limit=False)
            
            # First enhance the prompt with streaming
            enhanced_edit_prompt = await self.openai_client._enhance_image_edit_prompt(
                user_request=user_edit_request,
                image_description=image_analysis,
                conversation_history=enhanced_messages,
                stream_callback=enhancement_callback
            )
            
            # Show the final enhanced prompt
            if enhanced_edit_prompt and thinking_id:
                enhanced_text = f"Enhanced Prompt: ✨ _{enhanced_edit_prompt}_"
                self._update_message_streaming_sync(client, channel_id, thinking_id, enhanced_text)
                # Mark that we should NOT touch this message again
                response_metadata["prompt_message_id"] = thinking_id
            
            # Create a NEW message for editing status - don't touch the enhanced prompt!
            editing_id = await client.send_thinking_indicator(channel_id, thread_state.thread_ts)
            self._update_status(client, channel_id, editing_id, 
                              "Generating edited image. This may take a minute...", 
                              emoji=config.circle_loader_emoji)
            # Track the status message ID
            response_metadata["status_message_id"] = editing_id
            
            # Mark as streamed for main.py
            response_metadata["streamed"] = True
            
            # Use the edit_image API with the pre-enhanced prompt
            try:
                image_data = await self.openai_client.edit_image(
                    input_images=input_images,
                    input_mimetypes=input_mimetypes,
                    prompt=enhanced_edit_prompt,  # Use the pre-enhanced prompt
                    image_description=None,  # Don't pass description since we already enhanced
                    input_fidelity=thread_config.get("input_fidelity", "high"),
                    background=thread_config.get("image_background", "auto"),
                    output_format=thread_config.get("image_format", "png"),
                    output_compression=thread_config.get("image_compression", 100),
                    enhance_prompt=False,  # Already enhanced!
                    conversation_history=None  # Not needed since already enhanced
                )
                
                # After edit is complete, update message to show just the enhanced prompt
                # (remove the "Generating edited image..." status) - but respect rate limits
                if enhanced_edit_prompt and thinking_id and rate_limiter.can_make_request():
                    result = self._update_message_streaming_sync(client, channel_id, thinking_id, f"*Enhanced Prompt:* ✨ _{enhanced_edit_prompt}_")
                    if result["success"]:
                        rate_limiter.record_success()
                    else:
                        if result["rate_limited"]:
                            rate_limiter.record_failure(is_rate_limit=True)
                            self.log_debug("Couldn't clean up image edit status due to rate limit")
                    
            except Exception as e:
                self.log_error(f"Error editing image: {e}")
                return Response(
                    type="error",
                    content=f"Failed to edit image: {str(e)}"
                )
        else:
            # Non-streaming fallback
            # Use the edit_image API with separated inputs
            try:
                image_data = await self.openai_client.edit_image(
                    input_images=input_images,
                    input_mimetypes=input_mimetypes,
                    prompt=user_edit_request,  # Just the user's request
                    image_description=image_analysis,  # The analyzed description
                    input_fidelity=thread_config.get("input_fidelity", "high"),
                    background=thread_config.get("image_background", "auto"),
                    output_format=thread_config.get("image_format", "png"),
                    output_compression=thread_config.get("image_compression", 100),
                    enhance_prompt=True,
                    conversation_history=enhanced_messages  # Pass enhanced conversation with image analyses
                )
            except Exception as e:
                self.log_error(f"Error editing image: {e}")
                return Response(
                    type="error",
                    content=f"Failed to edit image: {str(e)}"
                )
        
        # Add clean message to thread state (no URLs or counts)
        user_breadcrumb = text or ""
        thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
        message_ts = message.metadata.get("ts") if message.metadata else None
        formatted_breadcrumb = self._format_user_content_with_username(user_breadcrumb, message)
        self._add_message_with_token_management(thread_state, "user", formatted_breadcrumb, db=self.db, thread_key=thread_key, message_ts=message_ts)
        
        # Store edited image in asset ledger
        asset_ledger = self.thread_manager.get_or_create_asset_ledger(thread_state.thread_ts)
        thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
        asset_ledger.add_image(
            image_data.base64_data,
            image_data.prompt,  # The enhanced edit prompt
            time.time(),
            source="edited",
            db=self.db,
            thread_id=thread_key,
            analysis=image_analysis  # Store the full analysis
        )
        
        # Store the edit prompt with metadata for tracking
        # Include the original analysis as part of the content for context
        if image_analysis:
            # Include analysis for future vision questions
            content = f"{image_data.prompt}\n\n[Original: {image_analysis}]"
        else:
            content = image_data.prompt
            
        self._add_message_with_token_management(thread_state, 
            "assistant", 
            content,
            db=self.db, 
            thread_key=thread_key,
            metadata={
                "type": "image_edit",
                "prompt": image_data.prompt,
                "original_analysis": image_analysis if image_analysis else None,
                "url": None  # Will be updated after upload
            }
        )
        
        return Response(
            type="image",
            content=image_data,
            metadata=response_metadata
        )
