from __future__ import annotations

import asyncio


class ImageJobMixin:
    """The background half of an image generation job, plus its progress-surface teardown.

    Both methods are driven by the F34 image TOOLS (message_processor/image_tools.py) — the
    detached ``generate_image`` schedules the job from inside the tool loop, where there is no
    Message object, only a ToolContext. That is why the signature takes ``message_ts`` /
    ``unprompted`` rather than the triggering message.
    """

    async def _finish_image_generation_background(self, *, client, channel_id, thread_id,
            thread_key, prompt, enhance, conversation_history, thread_config, checklist,
            generating_id, generation_id, message_ts=None, unprompted=False) -> None:
        """The slow generate_image call plus delivery, run after the thread lock released."""
        from message_processor.image_delivery import publish_image
        from message_processor.image_service import resolve_settings
        status_only = checklist is not None and checklist.surface == "assistant_status"
        settings, _ = resolve_settings(thread_config)
        try:
            image_data = await self.openai_client.generate_image(
                prompt=prompt,
                model=settings["model"],
                size=settings["size"],
                quality=settings["quality"],
                background=settings["background"],
                format=settings["format"],
                compression=settings["compression"],
                enhance_prompt=enhance,
                conversation_history=conversation_history,
            )
            file_url = await publish_image(
                processor=self, client=client, channel_id=channel_id, thread_id=thread_id,
                thread_key=thread_key, image_data=image_data, checklist=checklist,
                generation_id=generation_id, prompt=image_data.prompt, db=self.db,
                thread_manager=self.thread_manager, unprompted=unprompted,
                message_ts=message_ts,
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
            # checklist: delete the plain generating-status message, or clear the composer
            # status if the surface was status-only (generating_id None).
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
