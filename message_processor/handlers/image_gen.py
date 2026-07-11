from __future__ import annotations

import asyncio
from typing import Optional
from uuid import uuid4

import time

from base_client import BaseClient, Message, Response
from config import config, pipeline_status
from message_processor.progress import ProgressChecklist
from streaming import RateLimitManager, StreamingBuffer


class ImageGenerationMixin:
    async def _handle_image_generation(self, prompt: str, thread_state, client: BaseClient,
                                channel_id: str, thinking_id: Optional[str], message: Message,
                                skip_enhancement: bool = False, allow_background: bool = False) -> Response:
        """Handle image generation request with streaming enhancement

        Args:
            skip_enhancement: If True, skip prompt enhancement (used when falling back from failed edit)
            allow_background: If True (only the new_image intent path), the slow
                generate_image call detaches into a background job so the thread lock
                releases — the fast enhancement/setup stays inline. Edit fallbacks and
                timeout retries leave this False and run fully synchronous (F1).
        """
        self.log_info(f"Generating image for prompt: {prompt[:100]}...")

        # Initialize response metadata
        response_metadata = {}
        # Owned by whichever branch sets it up; referenced by the background handoff.
        checklist = None
        
        # Get thread config (with user preferences)
        thread_config = await config.get_thread_config_async(
            overrides=thread_state.config_overrides,
            user_id=message.user_id,
            db=self.db,
            channel_id=message.channel_id
        )
        
        # Inject stored image analyses for style consistency
        enhanced_messages = await self._inject_image_analyses(thread_state.messages, thread_state)
        
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
            
            
            # Enhanced-prompt display id: normally the thinking message, but since the
            # native-status refactor thinking_id is None on almost every surface (setStatus
            # wins), which silently swallowed the enhanced-prompt display. When it's None we
            # lazily create a real message (on the first streamed chunk) and edit that
            # instead (regression fix 2026-07-10).
            prompt_ref = {"id": thinking_id, "creating": False}

            def enhancement_callback(chunk: str):
                buffer.add_chunk(chunk)
                # Only update if both buffer says it's time AND rate limiter allows it
                if buffer.should_update() and rate_limiter.can_make_request():
                    rate_limiter.record_request_attempt()
                    display_text = f"*Enhanced Prompt:* ✨ _{buffer.get_complete_text()}_ {config.loading_ellipse_emoji}"

                    pid = prompt_ref["id"]
                    if pid is None:
                        # Status-only surface: lazily create the prompt message once; skip
                        # editing until its ts lands (a chunk or two may be dropped).
                        if not prompt_ref["creating"]:
                            prompt_ref["creating"] = True
                            prompt_ref["task"] = self._schedule_async_call(self._lazy_create_prompt_ref(
                                client, channel_id, thread_state.thread_ts, display_text, prompt_ref))
                        return
                    result = self._update_message_streaming_sync(client, channel_id, pid, display_text)

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
            
            # Show the final enhanced prompt (without loading indicator)
            if enhanced_prompt:
                enhanced_text = f"*Enhanced Prompt:* ✨ _{enhanced_prompt}_"
                pid = await self._resolve_prompt_message(
                    client, channel_id, thread_state.thread_ts, prompt_ref, enhanced_text)
                if pid:
                    # Mark that we should NOT touch this message again
                    response_metadata["prompt_message_id"] = pid
            
            # Create a NEW message for generating status - don't touch the enhanced prompt!
            generating_id = await client.send_thinking_indicator(channel_id, thread_state.thread_ts)
            progress_task = None
            if config.enable_progress_checklist:
                # The checklist owns the generating-status message; the legacy rotator
                # would overwrite its accumulated steps, so it is NOT started here (F4).
                checklist = ProgressChecklist(client, channel_id, thread_state.thread_ts,
                                              message_id=generating_id,
                                              prefer_message=config.progress_checklist_prefer_message)
                await checklist.step(
                    pipeline_status("generating_image", "Generating image. This may take a minute…"),
                    done_text="Generated image",
                )
                if checklist.message_id:  # status-only surface exposes no ts — never store a None id
                    response_metadata["status_message_id"] = checklist.message_id
            else:
                self._update_status(client, channel_id, generating_id,
                                  pipeline_status("generating_image", "Generating image. This may take a minute…"),
                                  emoji=config.circle_loader_emoji, thread_id=thread_state.thread_ts)
                # Track the status message ID
                if generating_id:  # status-only DMs return no ts — never store a None id
                    response_metadata["status_message_id"] = generating_id

                # Start progress updater for image generation
                try:
                    progress_task = await self._start_progress_updater_async(
                        client, channel_id, generating_id, "image generation", emoji=config.circle_loader_emoji
                    )
                    self.log_debug("Started progress updater for image generation")
                except Exception as e:
                    self.log_warning(f"Failed to start progress updater: {e}")
                    progress_task = None

            # F1: detach the slow generation into a background job (lock releases).
            if config.enable_background_image_gen and allow_background:
                # Stop the legacy rotator before detaching — otherwise (checklist off) it
                # keeps editing the generating-status message forever after we return.
                if progress_task and not progress_task.done():
                    progress_task.cancel()
                return await self._start_background_generation(
                    thread_state=thread_state, message=message, client=client,
                    channel_id=channel_id, prompt=prompt, final_prompt=enhanced_prompt,
                    enhance=False, conversation_history=None, thread_config=thread_config,
                    checklist=checklist, generating_id=generating_id,
                )

            # Generate image with already-enhanced prompt
            try:
                image_data = await self.openai_client.generate_image(
                    prompt=enhanced_prompt,
                    model=thread_config.get("image_model"),
                    size=thread_config.get("image_size"),
                    quality=thread_config.get("image_quality"),
                    background=thread_config.get("image_background"),
                    enhance_prompt=False,  # Already enhanced!
                    conversation_history=None  # Not needed since we enhanced already
                )

                # Cancel progress updater if still running
                if progress_task and not progress_task.done():
                    progress_task.cancel()
                    self.log_debug("Cancelled progress updater after image generation")

            except Exception as e:
                # Cancel progress updater on error
                if progress_task and not progress_task.done():
                    progress_task.cancel()
                    self.log_debug("Cancelled progress updater due to error")

                error_str = str(e)
                if "moderation_blocked" in error_str or "safety system" in error_str or "content policy" in error_str.lower():
                    # Clean up status message — best-effort; the friendly
                    # moderation reply below must still go out if this fails.
                    if "status_message_id" in response_metadata:
                        if hasattr(client, 'delete_message'):
                            try:
                                await client.delete_message(channel_id, response_metadata["status_message_id"])
                            except Exception as delete_error:
                                self.log_warning(f"Failed to delete status message: {delete_error}")
                    
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
            if enhanced_prompt and prompt_ref["id"] and rate_limiter.can_make_request():
                result = self._update_message_streaming_sync(client, channel_id, prompt_ref["id"], f"*Enhanced Prompt:* ✨ _{enhanced_prompt}_")
                if result["success"]:
                    rate_limiter.record_success()
                else:
                    if result["rate_limited"]:
                        rate_limiter.record_failure(is_rate_limit=True)
                        self.log_debug("Couldn't clean up image gen status due to rate limit")
        else:
            # Non-streaming fallback or skip enhancement
            if config.enable_progress_checklist:
                checklist = ProgressChecklist(client, channel_id, thread_state.thread_ts,
                                              message_id=thinking_id,
                                              prefer_message=config.progress_checklist_prefer_message)
                if not skip_enhancement:
                    await checklist.step(
                        pipeline_status("enhancing_prompt", "Enhancing your prompt…"),
                        done_text="Enhanced prompt",
                    )
                await checklist.step(
                    pipeline_status("generating_image", "Creating your image. This may take a minute…"),
                    done_text="Generated image",
                )
            else:
                if not skip_enhancement:
                    self._update_status(client, channel_id, thinking_id, pipeline_status("enhancing_prompt", "Enhancing your prompt…"), thread_id=thread_state.thread_ts)

                # Generate image with or without enhancement
                self._update_status(client, channel_id, thinking_id, pipeline_status("generating_image", "Creating your image. This may take a minute…"), emoji=config.circle_loader_emoji, thread_id=thread_state.thread_ts)

            # F1: detach the slow generation into a background job (lock releases).
            if config.enable_background_image_gen and allow_background:
                return await self._start_background_generation(
                    thread_state=thread_state, message=message, client=client,
                    channel_id=channel_id, prompt=prompt, final_prompt=prompt,
                    enhance=not skip_enhancement,
                    conversation_history=(enhanced_messages if not skip_enhancement else None),
                    thread_config=thread_config, checklist=checklist, generating_id=thinking_id,
                )

            try:
                image_data = await self.openai_client.generate_image(
                    prompt=prompt,
                    model=thread_config.get("image_model"),
                    size=thread_config.get("image_size"),
                    quality=thread_config.get("image_quality"),
                    background=thread_config.get("image_background"),
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

        # Hand the live checklist to the delivery seam (main.py) so F4's status message
        # is completed in place instead of overwritten. image_type drives the DB row.
        response_metadata["image_type"] = "generated"
        if checklist is not None:
            response_metadata["checklist"] = checklist

        return Response(
            type="image",
            content=image_data,
            metadata=response_metadata
        )

    async def _start_background_generation(self, *, thread_state, message, client,
            channel_id, prompt, final_prompt, enhance, conversation_history,
            thread_config, checklist, generating_id) -> Response:
        """Inline handoff (F1): append only the user turn, register + schedule the job,
        and return a 'background' response so the turn returns and the lock releases."""
        thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
        message_ts = message.metadata.get("ts") if message.metadata else None
        formatted_prompt = self._format_user_content_with_username(prompt, message)
        # ONLY the user message — no assistant breadcrumb. The Responses payload strips
        # metadata (a pending breadcrumb would read as a finished assistant turn) and the
        # no-DB-image rebuild check would wipe it before the image lands.
        self._add_message_with_token_management(
            thread_state, "user", formatted_prompt, db=self.db,
            thread_key=thread_key, message_ts=message_ts)

        generation_id = uuid4().hex[:12]
        self.thread_manager.register_generation(thread_key, generation_id, final_prompt[:200])
        try:
            task = self._schedule_async_call(self._finish_image_generation_background(
                client=client, channel_id=channel_id, thread_id=thread_state.thread_ts,
                thread_key=thread_key, prompt=final_prompt, enhance=enhance,
                conversation_history=conversation_history, thread_config=thread_config,
                checklist=checklist, generating_id=generating_id,
                generation_id=generation_id, message=message))
        except Exception as e:
            # Scheduling failed — the job will never run. Clear the registry (the latch
            # isn't started until process_message's return guard, so nothing to release)
            # and fall back to a friendly synchronous error.
            self.log_error(f"Failed to schedule background generation for {thread_key}: {e}", exc_info=True)
            self.thread_manager.finish_generation(thread_key, generation_id)
            await self._abort_checklist(checklist, client, channel_id, thread_state.thread_ts)
            return Response(type="text",
                            content="⚠️ I couldn't start that image generation. Please try again.")
        if task is not None:
            self.thread_manager.attach_generation_task(thread_key, generation_id, task)
        self.log_info(f"Detached image generation {generation_id} to background for {thread_key}")
        return Response(type="background", content="", metadata={
            "generation_id": generation_id, "background_owns_status": True})

    async def _finish_image_generation_background(self, *, client, channel_id, thread_id,
            thread_key, prompt, enhance, conversation_history, thread_config, checklist,
            generating_id, generation_id, message) -> None:
        """Background half of new-image generation (F1): the slow generate_image call plus
        delivery, run after the thread lock has already released."""
        from message_processor.image_delivery import publish_image
        status_only = checklist is not None and checklist.surface == "assistant_status"
        try:
            image_data = await self.openai_client.generate_image(
                prompt=prompt,
                model=thread_config.get("image_model"),
                size=thread_config.get("image_size"),
                quality=thread_config.get("image_quality"),
                background=thread_config.get("image_background"),
                enhance_prompt=enhance,
                conversation_history=conversation_history,
            )
            unprompted = bool(message.metadata.get("participation_check")) if message.metadata else False
            file_url = await publish_image(
                processor=self, client=client, channel_id=channel_id, thread_id=thread_id,
                thread_key=thread_key, image_data=image_data, checklist=checklist,
                generation_id=generation_id, prompt=image_data.prompt, db=self.db,
                thread_manager=self.thread_manager, unprompted=unprompted,
                message_ts=(message.metadata or {}).get("ts"),
            )
            if file_url is None:
                # publish_image already failed the checklist; surface a friendly notice
                # (recoverable via the post-refresh Slack rebuild).
                await client.handle_error(channel_id, thread_id,
                    "⚠️ I generated the image but couldn't post it. Please try again.")
        except asyncio.CancelledError:
            # Shutdown/cancel: clear the progress surface (message-surface checklists too,
            # which the finally's status-only clear wouldn't reach), then let finally run.
            await self._abort_checklist(checklist, client, channel_id, thread_id)
            raise
        except Exception as e:  # noqa: BLE001
            error_str = str(e)
            if ("moderation_blocked" in error_str or "safety system" in error_str
                    or "content policy" in error_str.lower()):
                await self._abort_checklist(checklist, client, channel_id, thread_id)
                try:
                    await client.send_message(channel_id, thread_id,
                        "I couldn't generate that image as it was flagged by content safety "
                        "filters. This can happen with certain brand names, people, or other "
                        "protected content. Try rephrasing your request or describing what you "
                        "want without using specific names.")
                except Exception:
                    pass
            else:
                self.log_error(f"Background image generation failed for {thread_key}: {e}", exc_info=True)
                if checklist is not None:
                    try:
                        await checklist.fail("Image generation failed")
                    except Exception:
                        pass
                try:
                    await client.handle_error(channel_id, thread_id,
                        "⚠️ I couldn't finish generating that image. Please try again.")
                except Exception:
                    pass
        finally:
            # Clear the progress surface on every path. With a checklist: only status-only
            # surfaces here (complete/fail already handled message surfaces). Without a
            # checklist (config-off background): delete the plain generating-status message,
            # or clear the composer status if the surface was status-only (generating_id None).
            try:
                if checklist is not None:
                    if status_only and hasattr(client, "clear_assistant_status"):
                        await client.clear_assistant_status(channel_id, thread_id)
                elif generating_id and hasattr(client, "delete_message"):
                    await client.delete_message(channel_id, generating_id)
                elif not generating_id and hasattr(client, "clear_assistant_status"):
                    await client.clear_assistant_status(channel_id, thread_id)
            except Exception:
                pass
            # ID-conditional clear (F1): finish_generation removes only THIS job's registry
            # entry, never a sibling's. F13: the upload latch is per-generation_id, so
            # releasing my own token is always safe (it can't drop a sibling's outstanding
            # upload) — release it unconditionally so a watchdog-cleared-but-still-running
            # job doesn't leak its latch token.
            self.thread_manager.finish_generation(thread_key, generation_id)
            try:
                self.thread_manager.mark_upload_finished(thread_key, generation_id)
            except Exception:
                pass
            # Next turn rebuilds the transcript from Slack (now has the posted image).
            self.thread_manager.mark_needs_refresh(thread_key)

    async def _abort_checklist(self, checklist, client, channel_id, thread_id) -> None:
        """Remove a checklist's progress surface entirely (moderation blocks — neither a
        completed ✓ nor a failed ✗ checklist should linger)."""
        if checklist is None:
            return
        try:
            if checklist.surface == "assistant_status":
                if hasattr(client, "clear_assistant_status"):
                    await client.clear_assistant_status(channel_id, thread_id)
            else:
                # A force-message checklist also mirrors the composer status — clear it
                # too so no status bubble lingers after the message is deleted.
                if checklist.mirrors_status and hasattr(client, "clear_assistant_status"):
                    await client.clear_assistant_status(channel_id, thread_id)
                if checklist.message_id and hasattr(client, "delete_message"):
                    await client.delete_message(channel_id, checklist.message_id)
        except Exception as e:  # noqa: BLE001
            self.log_warning(f"Failed to clear checklist surface: {e}")

    async def _create_prompt_message(self, client, channel_id, thread_ts, text) -> Optional[str]:
        """Post the enhanced-prompt display as its OWN message and return its ts (or None).
        Used when thinking_id is None (status-only surface) so the enhanced prompt still
        shows instead of silently no-oping on a None id."""
        try:
            if hasattr(client, "send_message_get_ts"):
                res = await client.send_message_get_ts(channel_id, thread_ts, text)
                if res and res.get("success"):
                    return res.get("ts")
        except Exception as e:  # noqa: BLE001
            self.log_warning(f"Failed to create enhanced-prompt message: {e}")
        return None

    async def _lazy_create_prompt_ref(self, client, channel_id, thread_ts, text, prompt_ref) -> None:
        """Background helper for the streaming callback: create the prompt message and
        record its id so subsequent chunks edit it."""
        try:
            pid = await self._create_prompt_message(client, channel_id, thread_ts, text)
            if pid:
                prompt_ref["id"] = pid
        finally:
            prompt_ref["creating"] = False

    async def _resolve_prompt_message(self, client, channel_id, thread_ts, prompt_ref, text) -> Optional[str]:
        """Final enhanced-prompt write: edit the existing message, or create one if none
        exists yet (status-only surface, or the lazy create lost the race)."""
        # If a lazy first-post is still in flight, wait for it — otherwise we'd post a
        # SECOND message and the pending task would then clobber prompt_ref["id"].
        pending = prompt_ref.get("task")
        if pending is not None:
            try:
                await pending
            except Exception:
                pass
        pid = prompt_ref.get("id")
        if pid:
            result = self._update_message_streaming_sync(client, channel_id, pid, text)
            if not result["success"] and result.get("rate_limited"):
                self.log_debug("Couldn't remove loading indicator from enhanced prompt due to rate limit")
            return pid
        pid = await self._create_prompt_message(client, channel_id, thread_ts, text)
        if pid:
            prompt_ref["id"] = pid
        return pid

    def _update_thinking_for_image(self, client: BaseClient, channel_id: str, thinking_id: Optional[str], thread_id: Optional[str] = None):
        """Update the progress indicator to show image generation message"""
        self._update_status(client, channel_id, thinking_id,
                          pipeline_status("generating_image", "Generating image. This may take a minute…"),
                          emoji=config.circle_loader_emoji, thread_id=thread_id)
