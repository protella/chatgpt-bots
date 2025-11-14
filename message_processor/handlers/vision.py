from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

import time

from base_client import BaseClient, Message, Response
from config import config
from prompts import IMAGE_ANALYSIS_PROMPT


class VisionHandlerMixin:
    def _inject_image_analyses(self, messages: List[Dict], thread_state) -> List[Dict]:
        """Inject stored image analyses into conversation for context"""
        if not self.db:
            return messages
            
        thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
        
        # Get all messages from DB with their timestamps
        cached_messages = self.db.get_cached_messages(thread_key)
        
        enhanced_messages = []
        
        # Match messages by position and content to get timestamps
        for i, msg in enumerate(messages):
            # Add the original message
            enhanced_messages.append(msg)
            
            # Only inject after user messages
            if msg.get("role") == "user":
                # Find corresponding cached message to get timestamp
                # Match by position and content (messages are in order)
                if i < len(cached_messages):
                    cached_msg = cached_messages[i]
                    msg_ts = cached_msg.get("message_ts")
                    
                    if msg_ts:
                        # Get images associated with this specific message
                        images_for_message = self.db.get_images_by_message(thread_key, msg_ts)
                        
                        for img_data in images_for_message:
                            analysis = img_data.get("analysis")
                            url = img_data.get("url")
                            image_type = img_data.get("image_type", "image")
                            
                            # Inject image context - either analysis or just URL info
                            if analysis:
                                # Full analysis available
                                enhanced_messages.append({
                                    "role": "developer",
                                    "content": f"[Visual context for {image_type}]:\n{analysis}\n[End of visual context]"
                                })
                                self.log_debug(f"Injected analysis for message at position {i}")
                            elif url:
                                # No analysis but we have the URL - inject basic info
                                context_msg = f"[Image context: {image_type} at {url}]"
                                if image_type == "generated":
                                    context_msg = f"[Bot generated an image and posted it at: {url}]"
                                elif image_type == "uploaded":
                                    context_msg = f"[User uploaded an image at: {url}]"
                                elif image_type == "edited":
                                    context_msg = f"[Bot edited an image and posted it at: {url}]"
                                    
                                enhanced_messages.append({
                                    "role": "developer",
                                    "content": context_msg
                                })
                                self.log_debug(f"Injected URL context for {image_type} at position {i}")
        
        if len(enhanced_messages) > len(messages):
            self.log_info(f"Enhanced conversation with {len(enhanced_messages) - len(messages)} image context entries")
        
        return enhanced_messages

    async def _handle_vision_analysis(self, user_text: str, image_inputs: List[Dict], thread_state, attachments: List[Dict],
                               client: BaseClient, channel_id: str, thinking_id: Optional[str], message: Message) -> Response:
        """Handle vision analysis of uploaded images"""
        if not image_inputs:
            return Response(
                type="error",
                content="No images found to analyze"
            )
        
        # Extract base64 data from image inputs and track URL images
        images_to_analyze = []
        url_images = []  # Track URL-sourced images
        
        for img_input in image_inputs:
            if img_input.get("type") == "input_image":
                # Extract from data URL format
                image_url = img_input.get("image_url", "")
                if image_url.startswith("data:"):
                    parts = image_url.split(",", 1)
                    if len(parts) == 2:
                        _, base64_data = parts
                        images_to_analyze.append(base64_data)
                        
                        # Track URL-sourced images in AssetLedger (including Slack URLs)
                        if img_input.get("source") in ["url", "slack_url"]:
                            url_images.append({
                                "base64_data": base64_data,
                                "original_url": img_input.get("original_url", "")
                            })
        
        if not images_to_analyze:
            return Response(
                type="error", 
                content="Could not process uploaded images"
            )
        
        self.log_info(f"Analyzing {len(images_to_analyze)} image(s) with prompt: {user_text[:100]}...")
        
        # Single vision API call with both analysis context and user question
        analysis_result = None
        
        try:
            self.log_info("Starting vision analysis with user question")
            
            # Get platform system prompt for consistent personality/formatting
            user_timezone = message.metadata.get("user_timezone", "UTC") if message.metadata else "UTC"
            user_tz_label = message.metadata.get("user_tz_label", None) if message.metadata else None
            user_real_name = message.metadata.get("user_real_name", None) if message.metadata else None
            user_email = message.metadata.get("user_email", None) if message.metadata else None
            # Use thread config model for vision analysis (with user preferences)
            thread_config = config.get_thread_config(
                overrides=thread_state.config_overrides,
                user_id=message.user_id,
                db=self.db
            )
            web_search_enabled = thread_config.get('enable_web_search', config.enable_web_search)
            system_prompt = self._get_system_prompt(client, user_timezone, user_tz_label, user_real_name, user_email, thread_config["model"], web_search_enabled, thread_state.has_trimmed_messages, thread_config.get('custom_instructions'))
            
            # Use the user's question directly - it will be enhanced for natural conversation
            # If no text provided with image, let the model infer from full conversation context
            if not user_text:
                # Pass empty question - the full conversation history will provide context
                user_question = ""
                self.log_debug("No text with image - relying on conversation context")
            else:
                user_question = user_text
            
            # Inject stored image analyses for better context
            enhanced_messages = self._inject_image_analyses(thread_state.messages, thread_state)
            
            # Pre-trim messages to fit within context window
            enhanced_messages = await self._pre_trim_messages_for_api(enhanced_messages, model=thread_state.current_model)
            
            # Update status to show we're preparing the analysis
            self._update_status(client, channel_id, thinking_id, "Preparing analysis...", emoji=config.analyze_emoji)
            
            # Extract filenames from attachments for context
            filenames = []
            if attachments and len(images_to_analyze) > 0:
                # Get filenames for images that were actually processed
                for i, img in enumerate(images_to_analyze[:len(attachments)]):
                    if i < len(attachments):
                        filenames.append(attachments[i].get("name", f"image{i+1}"))
            
            # Enhance the vision prompt with conversation context
            enhanced_question = user_question
            if user_question:  # Only enhance if there's actually a question
                enhanced_question = await self.openai_client._enhance_vision_prompt(
                    user_question,
                    conversation_history=enhanced_messages  # Pass the full conversation context
                )
                self.log_debug(f"Enhanced vision prompt: {enhanced_question[:100]}...")
            
            # Add filename context to the enhanced question for the API
            if filenames:
                filename_context = f"[Image files: {', '.join(filenames)}]\n"
                enhanced_question = filename_context + enhanced_question
                self.log_debug(f"Added filename context for {len(filenames)} images")
            
            # Update status to show we're analyzing the image(s)
            if len(images_to_analyze) == 1:
                status_msg = "Analyzing your image..."
            else:
                status_msg = f"Analyzing {len(images_to_analyze)} images..."
            self._update_status(client, channel_id, thinking_id, status_msg, emoji=config.analyze_emoji)
            
            # Check if streaming is supported and enabled (respecting user prefs)
            streaming_enabled = thread_config.get('enable_streaming', config.enable_streaming)
            if (hasattr(client, 'supports_streaming') and client.supports_streaming() and 
                streaming_enabled and thinking_id is not None):
                # Stream the vision analysis response
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
                
                message_id = thinking_id  # Start with the thinking message

                # Start progress updater task (will be cancelled when streaming starts)
                progress_task = None
                first_chunk_received = False

                def stream_callback(chunk: str):
                    nonlocal message_id, progress_task, first_chunk_received

                    # Cancel progress updater on first real chunk
                    if not first_chunk_received and chunk is not None:
                        first_chunk_received = True
                        if progress_task and not progress_task.done():
                            # Note: We're in a sync callback, so we can't await cancel
                            # The task will be cancelled from the async context below
                            pass
                    # Handle completion signal (None chunk)
                    if chunk is None:
                        # Final update without loading indicator
                        if message_id:
                            final_text = buffer.get_complete_text()
                            self._update_message_streaming_sync(client, channel_id, message_id, final_text)
                        return
                    
                    buffer.add_chunk(chunk)
                    
                    # Only update if both buffer says it's time AND rate limiter allows it
                    if buffer.should_update() and rate_limiter.can_make_request():
                        rate_limiter.record_request_attempt()
                        
                        # Build display text with loading indicator
                        display_text = buffer.get_complete_text() + " " + config.loading_ellipse_emoji
                        
                        # Try to update the message
                        result = self._update_message_streaming_sync(client, channel_id, message_id, display_text)
                        
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
                                    self.log_warning("Vision streaming circuit breaker opened")
                            else:
                                rate_limiter.record_failure(is_rate_limit=False)
                            self.log_debug(f"Failed to update message: {result.get('error', 'unknown error')}")

                # Start progress updater before making API call
                try:
                    progress_task = await self._start_progress_updater_async(
                        client, channel_id, message_id, "image analysis", emoji=config.analyze_emoji
                    )
                    self.log_debug("Started progress updater for vision analysis")
                except Exception as e:
                    self.log_warning(f"Failed to start progress updater: {e}")
                    progress_task = None

                # Call analyze_images with streaming callback
                self.log_info("Streaming vision analysis")
                analysis_result = await self.openai_client.analyze_images(
                    images=images_to_analyze,
                    question=enhanced_question,
                    detail="high",
                    enhance_prompt=False,  # Already enhanced
                    conversation_history=enhanced_messages,  # Pass enhanced conversation with image analyses
                    system_prompt=system_prompt,  # Pass platform system prompt
                    stream_callback=stream_callback
                )

                # Ensure progress updater is cancelled if still running
                if progress_task and not progress_task.done():
                    progress_task.cancel()
                    self.log_debug("Cancelled progress updater after vision analysis")

                # Log streaming stats
                stats = rate_limiter.get_stats()
                buffer_stats = buffer.get_stats()
                self.log_info(f"Vision streaming completed: {stats['successful_requests']}/{stats['total_requests']} updates, "
                             f"final length: {buffer_stats['text_length']} chars")
                
                self.log_debug(f"Vision analysis completed: {len(analysis_result)} chars")
            else:
                # Non-streaming version
                # Start progress updater for non-streaming vision analysis
                progress_task = None
                try:
                    progress_task = await self._start_progress_updater_async(
                        client, channel_id, thinking_id, "image analysis", emoji=config.analyze_emoji
                    )
                    self.log_debug("Started progress updater for non-streaming vision analysis")
                except Exception as e:
                    self.log_warning(f"Failed to start progress updater: {e}")
                    progress_task = None

                analysis_result = await self.openai_client.analyze_images(
                    images=images_to_analyze,
                    question=enhanced_question,
                    detail="high",
                    enhance_prompt=False,  # Already enhanced
                    conversation_history=enhanced_messages,  # Pass enhanced conversation with image analyses
                    system_prompt=system_prompt  # Pass platform system prompt
                )

                # Cancel progress updater after completion
                if progress_task and not progress_task.done():
                    progress_task.cancel()
                    self.log_debug("Cancelled progress updater after non-streaming vision analysis")

                self.log_debug(f"Vision analysis completed: {len(analysis_result)} chars")


        except TimeoutError as e:
            # Cancel progress updater on timeout
            if 'progress_task' in locals() and progress_task and not progress_task.done():
                progress_task.cancel()
                self.log_debug("Cancelled progress updater due to timeout")
            self.log_error(f"Vision analysis timed out: {e}")
            return Response(
                type="error",
                content=f"OpenAI's vision API timed out after {int(config.api_timeout_read)} seconds.\n\nThis is an OpenAI service issue. Please try again."
            )
            
        except Exception as e:
            self.log_error(f"Vision analysis failed: {e}", exc_info=True)
            return Response(
                type="error",
                content=f"Failed to analyze image: {str(e)}"
            )
        
        # Track all analyzed images in AssetLedger and database
        asset_ledger = self.thread_manager.get_or_create_asset_ledger(thread_state.thread_ts)
        thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
        
        # Save to database with error handling
        try:
            # Track URL images and save analysis
            if url_images:
                for url_img in url_images:
                    asset_ledger.add_url_image(
                        image_data=url_img["base64_data"],
                        url=url_img["original_url"],
                        timestamp=time.time()
                    )
                    self.log_debug(f"Added URL image to AssetLedger: {url_img['original_url']}")
                    
                    # Save comprehensive vision analysis for URL images
                    if self.db:
                        self.db.save_image_metadata(
                            thread_id=thread_key,
                            url=url_img["original_url"],
                            image_type="url",
                            prompt=user_text if user_text else "Vision analysis",
                            analysis=analysis_result,  # Store the vision analysis result
                            metadata={"timestamp": time.time()},
                            message_ts=message.metadata.get("ts") if message.metadata else None
                        )
                        self.log_debug(f"Saved comprehensive vision analysis for URL image: {url_img['original_url']}")
            
            # Save analysis for image attachments only (not documents)
            if attachments and self.db:
                for att in attachments:
                    # Only save actual images, not PDFs or other documents
                    if att.get("url") and att.get("type") == "image":
                        # Save the comprehensive vision analysis to database
                        self.db.save_image_metadata(
                            thread_id=thread_key,
                            url=att["url"],
                            image_type="uploaded",
                            prompt=user_text if user_text else "Vision analysis",
                            analysis=analysis_result,  # Store the vision analysis result
                            metadata={"timestamp": time.time()},
                            message_ts=message.metadata.get("ts") if message.metadata else None
                        )
                        self.log_debug(f"Saved comprehensive vision analysis for uploaded image: {att['url']}")
                        
        except Exception as e:
            self.log_error(f"Failed to save image metadata to database: {e}", exc_info=True)
            # Continue anyway - don't fail the whole request due to DB issues
        
        # Create breadcrumb with filenames for backend history
        if filenames:
            if user_text:
                breadcrumb_text = f"[Uploaded: {', '.join(filenames)}] {user_text}"
            else:
                breadcrumb_text = f"[Uploaded: {', '.join(filenames)}]"
        else:
            breadcrumb_text = user_text if user_text else "[User uploaded image(s) for analysis]"
        
        # Add URL if it was a URL-based image
        if img_input.get("source") == "url" and img_input.get("original_url"):
            breadcrumb_text += f" <{img_input['original_url']}>"
        
        # Add to thread state with error handling
        thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
        try:
            message_ts = message.metadata.get("ts") if message.metadata else None
            # Count total images for token estimation
            total_images = len(image_inputs)
            # Build metadata with filenames
            metadata = {
                "type": "vision_analysis", 
                "image_count": total_images
            }
            if filenames:
                metadata["filenames"] = filenames
            
            self._add_message_with_token_management(
                thread_state, "user", breadcrumb_text, 
                db=self.db, thread_key=thread_key, message_ts=message_ts,
                metadata=metadata
            )
            self._add_message_with_token_management(thread_state, "assistant", analysis_result, db=self.db, thread_key=thread_key)
        except Exception as e:
            self.log_error(f"Failed to save messages to database: {e}", exc_info=True)
            # Continue anyway - messages are in memory at least
        
        # Check if we streamed the response
        response_metadata = {}
        streaming_enabled = thread_config.get('enable_streaming', config.enable_streaming)
        if (hasattr(client, 'supports_streaming') and client.supports_streaming() and 
            streaming_enabled and thinking_id is not None):
            response_metadata["streamed"] = True
        
        return Response(
            type="text",
            content=analysis_result,
            metadata=response_metadata
        )

    async def _handle_mixed_content_analysis(
        self,
        user_text: str,
        image_inputs: List[Dict],
        document_inputs: List[Dict],
        thread_state,
        client: BaseClient,
        channel_id: str,
        thinking_id: Optional[str],
        message: Message
    ) -> Response:
        """Handle mixed content (images + documents) with two-call approach
        
        1. First call: Analyze images with vision model using technical prompt
        2. Second call: Combine image analysis + document content for final answer
        """
        try:
            # Step 1: Analyze images with vision model using IMAGE_ANALYSIS_PROMPT
            self._update_status(client, channel_id, thinking_id, "Analyzing images...", emoji=config.analyze_emoji)
            
            # Extract base64 data from image inputs
            images_to_analyze = []
            for img_input in image_inputs:
                if img_input.get("type") == "input_image":
                    image_url = img_input.get("image_url", "")
                    if image_url.startswith("data:"):
                        parts = image_url.split(",", 1)
                        if len(parts) == 2:
                            _, base64_data = parts
                            images_to_analyze.append(base64_data)
            
            # Analyze images with technical prompt (no enhancement needed)
            image_analysis = await self.openai_client.analyze_images(
                images=images_to_analyze,
                question=IMAGE_ANALYSIS_PROMPT,
                detail="high",
                enhance_prompt=False  # Use technical prompt as-is
            )
            
            self.log_info(f"Image analysis completed: {len(image_analysis)} chars")
            
            # Step 2: Build comprehensive context with image analysis and documents
            self._update_status(client, channel_id, thinking_id, "Combining analysis with documents...", emoji=config.analyze_emoji)
            
            # Build enhanced message with all context
            context_parts = []
            
            # Add image analysis results
            if image_analysis:
                context_parts.append("=== IMAGE ANALYSIS ===")
                context_parts.append(image_analysis)
                context_parts.append("")
            
            # Add document content directly from document_inputs
            # The user message hasn't been added to thread_state yet, so we must use document_inputs
            if document_inputs:
                for doc in document_inputs:
                    filename = doc.get("filename", "Unknown")
                    mimetype = doc.get("mimetype", "unknown")
                    content = doc.get("content", "")
                    pages = doc.get("pages")

                    # Build document header (same format as in utilities.py)
                    doc_header = f"\n\n=== DOCUMENT: {filename} ==="
                    if pages:
                        doc_header += f" ({pages} pages)"
                    doc_header += f"\nMIME Type: {mimetype}\n"

                    # Add document with its content
                    context_parts.append(doc_header)
                    context_parts.append(content)
                    context_parts.append("\n=== DOCUMENT END ===\n")
            
            # Add user question
            context_parts.append("")  # Empty line before question
            context_parts.append("USER QUESTION:")
            context_parts.append(user_text if user_text else "Please analyze the relationship between these files.")
            
            # Ensure all parts are strings before joining
            str_context_parts = []
            for i, part in enumerate(context_parts):
                if not isinstance(part, str):
                    self.log_warning(f"context_parts[{i}] is not a string: {type(part)}, value: {part}")
                    str_context_parts.append(str(part) if part is not None else "")
                else:
                    str_context_parts.append(part)
            
            combined_context = "\n".join(str_context_parts)
            
            # Step 3: Send to text model for final analysis
            self._update_status(client, channel_id, thinking_id, "Generating comprehensive response...", emoji=config.thinking_emoji)
            
            # Use text response handler with the combined context
            return await self._handle_text_response(combined_context, thread_state, client, message, thinking_id, retry_count=0)
            
        except Exception as e:
            self.log_error(f"Mixed content analysis failed: {e}", exc_info=True)
            return Response(
                type="error",
                content=f"Failed to analyze mixed content: {str(e)}"
            )

    async def _handle_vision_without_upload(
        self,
        text: str,
        thread_state,
        client: BaseClient,
        channel_id: str,
        thinking_id: Optional[str],
        message: Message
    ) -> Response:
        """Handle vision request when no images are uploaded - use text response with context"""
        
        # Check if we have any images in the conversation that provide context
        image_registry = self._extract_image_registry(thread_state)
        has_images = bool(image_registry) or self._has_recent_image(thread_state)
        
        # Check if we have documents in the conversation
        document_ledger = self.thread_manager.get_document_ledger(thread_state.thread_ts)
        has_documents = document_ledger and len(document_ledger.documents) > 0
        
        if has_images or has_documents:
            # We have image/document context in the conversation - let the model use that
            self.log_info(f"Vision intent with {'image' if has_images else ''}{' and ' if has_images and has_documents else ''}{'document' if has_documents else ''} context in history - using text response with context")
        else:
            self.log_info("Vision intent but no images or documents found in conversation")
        
        # Don't attach documents to the message - they're already in the thread history
        # The model will have access to them from the conversation context
        return await self._handle_text_response(text, thread_state, client, message, thinking_id, retry_count=0)
