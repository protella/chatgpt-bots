from __future__ import annotations

import json
import time

from slack_sdk.errors import SlackApiError

from config import config


class SlackSettingsHandlersMixin:
    async def _get_session_data(self, body) -> dict:
        """
        Helper to get session data from database using session_id in metadata.

        Args:
            body: Slack modal interaction body

        Returns:
            Session data dict or None if not found
        """
        try:
            private_metadata = body.get('view', {}).get('private_metadata')
            if private_metadata:
                metadata = json.loads(private_metadata)
                session_id = metadata.get('session_id')
                if session_id:
                    session_data = await self.db.get_modal_session_async(session_id)
                    return session_data
        except Exception as e:
            self.log_error(f"Failed to get session data: {e}")
        return None

    async def _update_session_data(self, body, new_data) -> bool:
        """
        Helper to update session data in database.

        Args:
            body: Slack modal interaction body
            new_data: Updated session data

        Returns:
            True if updated successfully
        """
        try:
            private_metadata = body.get('view', {}).get('private_metadata')
            if private_metadata:
                metadata = json.loads(private_metadata)
                session_id = metadata.get('session_id')
                if session_id:
                    return await self.db.update_modal_session_async(session_id, new_data)
        except Exception as e:
            self.log_error(f"Failed to update session data: {e}")
        return False

    def _register_settings_handlers(self):
        # Register slash command handler
        @self.app.command(config.settings_slash_command)
        async def handle_settings_command(ack, body, client):
            """Handle the settings slash command"""
            await ack()  # Acknowledge command receipt immediately
            
            user_id = body.get('user_id')
            trigger_id = body.get('trigger_id')
            
            # Slash commands are always for global settings
            # Thread-specific settings use the message shortcut instead
            
            self.log_info(f"Settings command invoked by user {user_id}")
            
            # Get current settings or create defaults
            current_settings = await self.db.get_user_preferences_async(user_id)
            is_new_user = current_settings is None

            if is_new_user:
                # Get user's email for preferences
                user_data = await self.db.get_or_create_user_async(user_id)
                email = user_data.get('email') if user_data else None
                current_settings = await self.db.create_default_user_preferences_async(user_id, email)
            
            # Build and open modal for global settings
            try:
                modal = await self.settings_modal.build_settings_modal(
                    user_id=user_id,
                    trigger_id=trigger_id,
                    current_settings=current_settings,
                    is_new_user=is_new_user,
                    thread_id=None,  # Always None for slash commands
                    in_thread=False  # Always False for slash commands
                )
                # Keep default title "ChatGPT Settings (Dev)" from settings_modal.py
                
                # Open the modal
                response = await client.views_open(
                    trigger_id=trigger_id,
                    view=modal
                )
                
                if response.get('ok'):
                    self.log_info(f"Global settings modal opened for user {user_id}")
                else:
                    self.log_error(f"Failed to open modal: {response.get('error')}")
                    
            except SlackApiError as e:
                self.log_error(f"Error opening settings modal: {e}")
                # Fallback to ephemeral message
                try:
                    await client.chat_postEphemeral(
                        channel=body.get('channel_id'),
                        user=user_id,
                        text="❌ Sorry, I couldn't open the settings modal. Please try again."
                    )
                except Exception:
                    pass
        
        # --- Phase H+: USER settings modal, opened from the DM strip's "⚙️ <model>" button ---
        @self.app.action("open_user_settings")
        async def handle_open_user_settings(ack, body, client):
            """DM feedback strip's '⚙️ <model>' button → open the global user settings
            modal (same flow as the slash command).

            trigger_ids expire in ~3s, so ack and views_open immediately.
            """
            await ack()
            trigger_id = body.get('trigger_id')
            user_id = (body.get('user') or {}).get('id')
            channel_id = ((body.get('container') or {}).get('channel_id')
                          or (body.get('channel') or {}).get('id'))
            if not trigger_id or not user_id:
                return
            try:
                current_settings = await self.db.get_user_preferences_async(user_id)
                is_new_user = current_settings is None
                if is_new_user:
                    user_data = await self.db.get_or_create_user_async(user_id)
                    email = user_data.get('email') if user_data else None
                    current_settings = await self.db.create_default_user_preferences_async(user_id, email)
                modal = await self.settings_modal.build_settings_modal(
                    user_id=user_id,
                    trigger_id=trigger_id,
                    current_settings=current_settings,
                    is_new_user=is_new_user,
                    thread_id=None,   # strip button always edits global settings
                    in_thread=False,
                )
                await client.views_open(trigger_id=trigger_id, view=modal)
                self.log_info(f"User settings modal opened for {user_id} (DM strip button)")
            except Exception as e:
                self.log_error(f"Error opening user settings modal from DM strip: {e}")
                # The user clicked a button — silence reads as a dead button.
                try:
                    if channel_id:
                        await client.chat_postEphemeral(
                            channel=channel_id,
                            user=user_id,
                            text="⚠️ Couldn't open settings — please try again."
                        )
                except Exception:
                    pass

        # --- Phase 7: per-channel settings modal, opened from the response footer button ---
        # ANY channel member may open Configure and save — no admin gating, no membership lookups.
        @self.app.action("open_channel_settings")
        async def handle_open_channel_settings(ack, body, client):
            """Footer '⚙️ Configure' button → open the per-channel settings modal.

            trigger_ids expire in ~3s, so ack and views_open immediately with no slow work first.
            """
            await ack()
            trigger_id = body.get('trigger_id')
            channel_id = ((body.get('container') or {}).get('channel_id')
                          or (body.get('channel') or {}).get('id'))
            user_id = (body.get('user') or {}).get('id')
            if not trigger_id or not channel_id or channel_id.startswith('D'):
                return
            try:
                current = await self.db.get_channel_settings_async(channel_id)
                global_default = getattr(config, 'channel_response_mode', 'tag_only')
                modal = self.settings_modal.build_channel_settings_modal(channel_id, current, global_default)
                await client.views_open(trigger_id=trigger_id, view=modal)
                self.log_info(f"Channel settings modal opened for {channel_id} by {user_id} (footer button)")
            except Exception as e:
                self.log_error(f"Error opening channel settings modal from footer: {e}")
                # The user clicked a button — silence reads as a dead button.
                try:
                    await client.chat_postEphemeral(
                        channel=channel_id,
                        user=user_id,
                        text="⚠️ Couldn't open settings — please try again."
                    )
                except Exception:
                    pass

        # One-gear design: the channel modal carries a "My personal settings" button
        # that stacks the personal modal on top (views_push) — no second button in chat.
        @self.app.action("open_user_settings_push")
        async def handle_open_user_settings_push(ack, body, client):
            await ack()
            trigger_id = body.get('trigger_id')
            user_id = (body.get('user') or {}).get('id')
            if not trigger_id or not user_id:
                return
            try:
                current_settings = await self.db.get_user_preferences_async(user_id)
                is_new_user = current_settings is None
                if is_new_user:
                    user_data = await self.db.get_or_create_user_async(user_id)
                    email = user_data.get('email') if user_data else None
                    current_settings = await self.db.create_default_user_preferences_async(user_id, email)
                modal = await self.settings_modal.build_settings_modal(
                    user_id=user_id,
                    trigger_id=trigger_id,
                    current_settings=current_settings,
                    is_new_user=is_new_user,
                    thread_id=None,
                    in_thread=False,
                )
                await client.views_push(trigger_id=trigger_id, view=modal)
                self.log_info(f"User settings modal pushed for {user_id} (from channel modal)")
            except Exception as e:
                self.log_error(f"Error pushing user settings modal: {e}")

        @self.app.view("channel_settings_modal")
        async def handle_channel_settings_submission(ack, body, view, client):
            """Persist per-channel settings. 'inherit' clears the override (NULL → global default)."""
            await ack()
            user_id = body['user']['id']
            metadata = json.loads(view.get('private_metadata', '{}'))
            channel_id = metadata.get('channel_id')
            if not channel_id:
                return

            from message_processor.participation import LEVEL_TO_MODE, MODE_TO_LEVEL

            state = view.get('state', {}).get('values', {})
            sel = state.get('participation_block', {}).get('participation_level', {}).get('selected_option') or {}
            level_sel = sel.get('value', 'inherit')
            # Dual-write (Phase F): participation_level is authoritative; response_mode is
            # kept in lockstep so legacy readers stay consistent. 'inherit' clears BOTH.
            participation_level = None if level_sel == 'inherit' else level_sel
            response_mode = None if participation_level is None else LEVEL_TO_MODE.get(participation_level)

            dir_raw = state.get('directives_block', {}).get('directives', {}).get('value')
            directives = dir_raw.strip() if dir_raw and dir_raw.strip() else None

            ric_selected = state.get('reply_in_channel_block', {}).get('reply_in_channel', {}).get('selected_options') or []
            reply_in_channel = len(ric_selected) > 0

            # F15: the snooze early-resume control is retired (thread mutes replace the timer).

            # Shared response settings (model/effort/verbosity): 'inherit' clears to NULL
            # so the asker's personal preferences apply again.
            def _sel(block, action):
                opt = state.get(block, {}).get(action, {}).get('selected_option') or {}
                val = opt.get('value', 'inherit')
                return None if val == 'inherit' else val

            channel_model = _sel('channel_model_block', 'channel_model')
            channel_effort = _sel('channel_effort_block', 'channel_reasoning_effort')
            channel_verbosity = _sel('channel_verbosity_block', 'channel_verbosity')

            try:
                await self.db.set_channel_settings_async(
                    channel_id,
                    response_mode=response_mode,          # None → clears override
                    directives=directives,                # None → clears
                    reply_in_channel=reply_in_channel,
                    participation_level=participation_level,  # None → clears override
                    model=channel_model,                  # None → each person's own setting
                    reasoning_effort=channel_effort,
                    verbosity=channel_verbosity,
                    updated_by=user_id,
                )
                default_level = MODE_TO_LEVEL.get(getattr(config, 'channel_response_mode', 'tag_only'), 'mentions_only')
                effective = level_sel if level_sel != 'inherit' else f"inherit ({default_level})"
                self.log_info(f"Channel settings saved for {channel_id} by {user_id}: participation={level_sel}")
                try:
                    await client.chat_postEphemeral(
                        channel=channel_id, user=user_id,
                        text=f"✅ Channel settings saved. Response mode: *{effective}*."
                    )
                except Exception:
                    pass
            except Exception as e:
                self.log_error(f"Error saving channel settings for {channel_id}: {e}")

        @self.app.action("channel_model")
        async def handle_channel_model_change(ack, body, client):
            """Rebuild the channel modal when the model changes so the effort ladder
            matches the selected model (gpt-5.5 has no `max`). Live form selections
            are overlaid on the DB row so nothing the user set is lost."""
            await ack()
            view = body.get('view') or {}
            metadata = json.loads(view.get('private_metadata', '{}'))
            channel_id = metadata.get('channel_id')
            if not channel_id:
                return
            try:
                state = view.get('state', {}).get('values', {})

                def _sel(block, action):
                    opt = state.get(block, {}).get(action, {}).get('selected_option') or {}
                    val = opt.get('value', 'inherit')
                    return None if val == 'inherit' else val

                row = await self.db.get_channel_settings_async(channel_id) or {}
                cs = dict(row)
                cs['model'] = _sel('channel_model_block', 'channel_model')
                cs['reasoning_effort'] = _sel('channel_effort_block', 'channel_reasoning_effort')
                cs['verbosity'] = _sel('channel_verbosity_block', 'channel_verbosity')
                cs['participation_level'] = _sel('participation_block', 'participation_level')
                dir_raw = state.get('directives_block', {}).get('directives', {}).get('value')
                cs['directives'] = dir_raw if dir_raw else cs.get('directives')
                ric = state.get('reply_in_channel_block', {}).get('reply_in_channel', {}).get('selected_options') or []
                cs['reply_in_channel'] = len(ric) > 0

                global_default = getattr(config, 'channel_response_mode', 'tag_only')
                modal = self.settings_modal.build_channel_settings_modal(channel_id, cs, global_default)
                await client.views_update(view_id=view.get('id'), view=modal)
                self.log_debug(f"Channel modal rebuilt for model change in {channel_id}")
            except Exception as e:
                self.log_error(f"Error rebuilding channel modal on model change: {e}")

        # Register modal submission handlers
        @self.app.view("settings_modal")
        @self.app.view("welcome_settings_modal")
        async def handle_settings_submission(ack, body, view, client):
            """Handle settings modal submission"""

            user_id = body['user']['id']

            # Get session data from database
            metadata = json.loads(view.get('private_metadata', '{}'))  # Always parse for session_id
            session_data = await self._get_session_data(body)
            if not session_data:
                # Fallback for old modals - extract from metadata directly
                thread_id = metadata.get('thread_id')
                in_thread = metadata.get('in_thread', False)
                pending_message = metadata.get('pending_message')
            else:
                thread_id = session_data.get('thread_id')
                in_thread = session_data.get('in_thread', False)
                pending_message = session_data.get('pending_message')
            
            # Determine if this is a new user
            is_new_user = view.get('callback_id') == 'welcome_settings_modal'
            
            # For new users, always save to global regardless of scope selection
            if is_new_user:
                selected_scope = 'global'
                self.log_info("New user setup - forcing save to global settings")
            else:
                if session_data:
                    selected_scope = session_data.get('scope', 'thread' if in_thread else 'global')
                else:
                    selected_scope = metadata.get('scope', 'thread' if in_thread else 'global')  # Get selected scope

            # Debug logging
            self.log_debug(f"Modal submission session/metadata: session_id={metadata.get('session_id') if session_data else 'legacy'}")
            self.log_debug(f"Selected scope for save: {selected_scope} (is_new_user: {is_new_user})")
            if pending_message:
                self.log_info(f"Found pending message: {pending_message}")
            
            # Extract form values
            form_values = self.settings_modal.extract_form_values(view['state'])
            
            # Check if we need confirmation for global custom instructions from thread
            needs_confirmation = False
            if (in_thread and selected_scope == 'global' and
                form_values.get('custom_instructions') and
                not metadata.get('confirmed')):

                # Check if there are existing global custom instructions
                existing_prefs = await self.db.get_user_preferences_async(user_id)
                existing_custom = existing_prefs.get('custom_instructions', '') if existing_prefs else ''
                
                if existing_custom:
                    needs_confirmation = True
                    self.log_debug("Confirmation needed: saving thread custom instructions to global with existing global instructions")
            
            if needs_confirmation:
                # Push confirmation modal on top of settings modal (preserves settings modal underneath)
                confirmation_modal = {
                    "type": "modal",
                    "callback_id": "confirm_global_custom_instructions",
                    "title": {"type": "plain_text", "text": "Confirm Changes"},
                    "submit": {"type": "plain_text", "text": "Yes, Continue"},
                    "close": {"type": "plain_text", "text": "Go Back"},
                    "blocks": [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "⚠️ *You have existing global custom instructions*\n\nSaving these settings globally will replace your current global custom instructions with the ones from this thread.\n\nThis will affect all your future conversations."
                            }
                        }
                    ],
                    "private_metadata": json.dumps({
                        **metadata,
                        "confirmed": True,
                        "form_values": form_values
                    })
                }
                
                # Use 'push' to stack this modal on top, preserving the settings modal
                await ack(response_action="push", view=confirmation_modal)
                return
            
            # Normal flow - acknowledge immediately
            await ack()
            
            # Validate settings
            validated_settings = self.settings_modal.validate_settings(form_values)
            
            # Check for temperature/top_p warning
            warning_message = ""
            temp_changed = validated_settings.get('temperature', config.default_temperature) != config.default_temperature
            top_p_changed = validated_settings.get('top_p', config.default_top_p) != config.default_top_p

            if temp_changed and top_p_changed:
                warning_message = "\n⚠️ Note: You've changed both Temperature and Top P. OpenAI recommends using only one for best results."
            
            # Save to appropriate location based on selected scope
            if selected_scope == 'thread' and thread_id:
                # Save as thread config (don't include settings_completed flag for thread configs)
                thread_settings = {k: v for k, v in validated_settings.items() if k != 'settings_completed'}
                # Ensure thread exists
                channel_id, thread_ts = thread_id.split(':')
                await self.db.get_or_create_thread_async(thread_id, channel_id)
                await self.db.save_thread_config_async(thread_id, thread_settings)

                # Also flag the user as onboarded — saving any scope of settings is signal
                # enough that the user knows about /settings and doesn't need the welcome
                # warning anymore. Without this, users who only ever save thread-scope
                # configs keep seeing "Please configure your settings" on every DM.
                await self.db.update_user_preferences_async(user_id, {'settings_completed': True})

                # Update memory cache immediately if processor is available
                if hasattr(self, 'processor') and self.processor:
                    try:
                        # Update the thread state's config_overrides in memory
                        thread_state = self.processor.thread_manager.get_thread_state(channel_id, thread_ts)
                        if thread_state:
                            thread_state.config_overrides = thread_settings
                            self.log_debug(f"Thread config updated in memory for {thread_id}")
                    except Exception as e:
                        self.log_debug(f"Could not update thread config in memory: {e}")

                success = True
                save_location = "thread"
                self.log_info(f"Thread settings saved for {thread_id}: {thread_settings}")
            else:
                # Mark settings as completed for user preferences
                validated_settings['settings_completed'] = True
                # Update user preferences
                success = await self.db.update_user_preferences_async(user_id, validated_settings)
                save_location = "global"
                self.log_info(f"Global settings saved for user {user_id}: {validated_settings}")
            
            if success:
                # Clean up the session from database
                if session_data:
                    try:
                        session_id = json.loads(view.get('private_metadata', '{}')).get('session_id')
                        if session_id:
                            await self.db.delete_modal_session_async(session_id)
                            self.log_debug(f"Cleaned up modal session {session_id} after successful save")
                    except Exception as e:
                        self.log_warning(f"Failed to clean up session: {e}")

                # Send confirmation message
                try:
                    # Determine message based on save location
                    if save_location == "thread" and thread_id:
                        # Send ephemeral confirmation in the thread
                        channel_id, thread_ts = thread_id.split(':')
                        await client.chat_postEphemeral(
                            channel=channel_id,
                            thread_ts=thread_ts,
                            user=user_id,
                            text=f"✅ Thread settings updated successfully!{warning_message}\n_These settings will only apply to this conversation thread._"
                        )
                    else:
                        # Send DM for global settings with settings button
                        blocks = [
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"✅ Your global settings have been saved successfully!{warning_message}"
                                }
                            },
                            {
                                "type": "actions",
                                "elements": [
                                    {
                                        "type": "button",
                                        "text": {
                                            "type": "plain_text",
                                            "text": f"⚙️ {validated_settings.get('model') or config.gpt_model}"
                                        },
                                        "action_id": "open_global_settings_dm"
                                    }
                                ]
                            }
                        ]
                        
                        await client.chat_postMessage(
                            channel=user_id,  # Send to user's DM
                            text=f"✅ Your global settings have been saved successfully!{warning_message}",
                            blocks=blocks
                        )
                except SlackApiError as e:
                    self.log_error(f"Error sending confirmation: {e}")
                
                # Clean up reminder messages
                if hasattr(self, '_reminder_messages') and user_id in self._reminder_messages:
                    for msg_info in self._reminder_messages[user_id]:
                        try:
                            await client.chat_delete(
                                channel=msg_info['channel'],
                                ts=msg_info['ts']
                            )
                        except Exception:
                            pass  # Best effort
                    del self._reminder_messages[user_id]
                    self.log_debug(f"Cleaned up reminder messages for user {user_id}")
                
                # Update welcome message to compact settings button
                if hasattr(self, '_welcome_messages') and user_id in self._welcome_messages:
                    welcome_info = self._welcome_messages[user_id]
                    try:
                        # Update to compact settings button
                        await client.chat_update(
                            channel=welcome_info['channel'],
                            ts=welcome_info['ts'],
                            text="Settings available",
                            blocks=[
                                {
                                    "type": "actions",
                                    "elements": [
                                        {
                                            "type": "button",
                                            "text": {
                                                "type": "plain_text",
                                                "text": f"⚙️ {validated_settings.get('model') or config.gpt_model}"
                                            },
                                            "action_id": "open_welcome_settings",
                                            "value": json.dumps({
                                                "channel_id": welcome_info['channel'],
                                                "thread_id": welcome_info['thread_ts']
                                            })
                                        }
                                    ]
                                }
                            ]
                        )
                        self.log_debug(f"Updated welcome message to compact settings button for user {user_id}")
                    except Exception:
                        pass  # Best effort
                    del self._welcome_messages[user_id]
                
                # Remove from welcomed users set once they've configured settings
                # This ensures they won't get welcome messages anymore
                if hasattr(self, '_welcomed_users') and user_id in self._welcomed_users:
                    self._welcomed_users.remove(user_id)
                    
                # Process pending message if this was from welcome flow (only for global settings)
                if save_location == "global" and pending_message:
                    # Check if message was too long to store
                    if pending_message.get('too_long'):
                        self.log_info(f"Original message was too long for new user {user_id}")
                        # Notify the user to resend their message IN THE THREAD
                        try:
                            await client.chat_postMessage(
                                channel=pending_message.get('channel_id', user_id),
                                thread_ts=pending_message.get('thread_id'),  # Send to thread if it exists
                                text="✅ Settings saved! I couldn't save your initial message during setup (too long). Please send it again and I'll process it normally."
                            )
                        except Exception:
                            pass
                    elif pending_message.get('original_message'):
                        self.log_info(f"Processing pending message for new user {user_id}: {pending_message['original_message']}")
                        if pending_message.get('attachments'):
                            self.log_info(f"Found {len(pending_message['attachments'])} attachments in pending message")
                        try:
                            # Create a synthetic Slack event for the original message
                            synthetic_event = {
                                'type': 'message',
                                'text': pending_message['original_message'],
                                'user': user_id,
                                'channel': pending_message['channel_id'],
                                'thread_ts': pending_message.get('thread_id'),
                                'ts': pending_message.get('ts') or pending_message.get('thread_id') or str(time.time())
                            }
                        
                            # Add file attachments if they were present in original message
                            if pending_message.get('attachments'):
                                # Convert our internal format back to Slack format
                                files = []
                                for attachment in pending_message['attachments']:
                                    files.append({
                                        'id': attachment.get('id'),
                                        'name': attachment.get('name'),
                                        'mimetype': attachment.get('mimetype'),
                                        'url_private': attachment.get('url')
                                    })
                                synthetic_event['files'] = files

                            # Process the original message now that settings are configured
                            await self._handle_slack_message(synthetic_event, client)

                        except Exception as e:
                            self.log_error(f"Error processing pending message: {e}")
                            # Send error message to user
                            try:
                                await client.chat_postMessage(
                                    channel=pending_message['channel_id'],
                                    thread_ts=pending_message.get('thread_id'),
                                    text="I'm ready now! Could you please repeat your question?"
                                )
                            except Exception:
                                pass
                            
            else:
                self.log_error(f"Failed to save settings for user {user_id}")
                try:
                    await client.chat_postMessage(
                        channel=user_id,
                        text="❌ Sorry, there was an error saving your settings. Please try again."
                    )
                except Exception:
                    pass
        
        # Handler for custom instructions confirmation modal submission
        @self.app.view("confirm_global_custom_instructions")
        async def handle_custom_instructions_confirmation(ack, body, view, client):
            """Handle confirmation for overwriting global custom instructions"""
            # Clear all modals when confirmed
            ack(response_action="clear")
            
            user_id = body['user']['id']
            
            # Extract metadata with confirmed flag and form values
            metadata = json.loads(view.get('private_metadata', '{}'))
            form_values = metadata.get('form_values', {})
            
            # Now proceed with the normal save flow using the form values
            # This essentially continues the original submission with confirmed=True
            validated_settings = self.settings_modal.validate_settings(form_values)
            
            # Mark settings as completed for user preferences
            validated_settings['settings_completed'] = True
            
            # Update user preferences with the confirmed custom instructions
            success = await self.db.update_user_preferences_async(user_id, validated_settings)
            
            if success:
                self.log_info(f"Global settings saved after confirmation for user {user_id}: {validated_settings}")
                
                # Send confirmation message
                try:
                    blocks = [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "✅ Your global settings have been saved successfully!\n_Your custom instructions have been updated and will apply to all conversations._"
                            }
                        },
                        {
                            "type": "actions",
                            "elements": [
                                {
                                    "type": "button",
                                    "text": {
                                        "type": "plain_text",
                                        "text": f"⚙️ {validated_settings.get('model') or config.gpt_model}"
                                    },
                                    "action_id": "open_global_settings_dm"
                                }
                            ]
                        }
                    ]
                    
                    await client.chat_postMessage(
                        channel=user_id,
                        text="✅ Your global settings have been saved successfully!",
                        blocks=blocks
                    )
                except SlackApiError as e:
                    self.log_error(f"Error sending confirmation after custom instructions update: {e}")
            else:
                self.log_error(f"Failed to save settings after confirmation for user {user_id}")
        
        # Register modal action handlers (for dynamic updates)
        @self.app.action("model_select")
        @self.app.action("image_model")  # Reshapes modal (filters background options, hides input_fidelity on v2)
        async def handle_model_change(ack, body, client):
            """Handle model selection changes for dynamic modal updates"""
            await ack()

            user_id = body['user']['id']
            action_id = body['actions'][0].get('action_id', 'model_select')
            selected_model = body['actions'][0]['selected_option']['value']

            self.log_info(f"{action_id} changed to {selected_model} for user {user_id}")

            # Get session data from database
            session_data = await self._get_session_data(body)
            if not session_data:
                self.log_error("No session found for modal interaction")
                return

            stored_settings = session_data.get('settings', {})
            metadata_context = {
                'thread_id': session_data.get('thread_id'),
                'in_thread': session_data.get('in_thread', False),
                'scope': session_data.get('scope'),
                'pending_message': session_data.get('pending_message')
            }
            
            # Extract current form values (only gets visible fields)
            current_values = self.settings_modal.extract_form_values(body['view']['state'])
            
            # Merge: stored settings as base, current values override what's visible
            # This preserves all field values when switching models
            if isinstance(stored_settings, dict):
                merged_settings = stored_settings.copy()
            else:
                merged_settings = {}
            merged_settings.update(current_values)

            # Update session with merged settings
            session_data['settings'] = merged_settings
            await self._update_session_data(body, session_data)

            # Build updated modal with new model selection
            # Note: build_settings_modal will create a NEW session, but that's okay
            # because each modal update gets a fresh session_id
            is_new_user = body['view']['callback_id'] == 'welcome_settings_modal'
            updated_modal = await self.settings_modal.build_settings_modal(
                user_id=user_id,
                trigger_id=None,  # Not needed for update
                current_settings=merged_settings,
                is_new_user=is_new_user,
                thread_id=metadata_context.get('thread_id'),
                in_thread=metadata_context.get('in_thread', False),
                scope=metadata_context.get('scope'),  # Preserve selected scope
                pending_message=metadata_context.get('pending_message')  # Preserve pending message
            )
            
            # Update the modal view
            try:
                response = await client.views_update(
                    view_id=body['view']['id'],
                    view=updated_modal
                )
                
                if response.get('ok'):
                    self.log_debug(f"Modal updated for model change: {selected_model}")
                else:
                    self.log_error(f"Failed to update modal: {response.get('error')}")
                    
            except SlackApiError as e:
                self.log_error(f"Error updating modal for model change: {e}")
        
        # Register handler for features checkbox (needs modal rebuild for web search and MCP)
        @self.app.action("features_with_mcp")
        async def handle_features_change(ack, body, client):
            """Handle feature checkbox changes, especially web search and model-specific features"""
            await ack()
            
            user_id = body['user']['id']
            
            # Get session data from database
            session_data = await self._get_session_data(body)
            if not session_data:
                self.log_error("No session found for modal interaction")
                return

            stored_settings = session_data.get('settings', {})
            metadata_context = {
                'thread_id': session_data.get('thread_id'),
                'in_thread': session_data.get('in_thread', False),
                'scope': session_data.get('scope'),
                'pending_message': session_data.get('pending_message')
            }
            
            # Get current form values
            current_values = self.settings_modal.extract_form_values(body['view']['state'])
            
            # Merge settings - stored as base, current values override
            if isinstance(stored_settings, dict):
                merged_settings = stored_settings.copy()
            else:
                merged_settings = {}
            
            # Check if web search is being enabled
            web_search_enabled = current_values.get('enable_web_search', merged_settings.get('enable_web_search', False))
            
            # Check current reasoning value - could be from form or stored
            current_reasoning = current_values.get('reasoning_effort') or stored_settings.get('reasoning_effort', 'medium')
            
            # If web search is being enabled and reasoning is/was minimal, force upgrade
            if web_search_enabled and current_reasoning == 'minimal':
                current_values['reasoning_effort'] = 'low'
                self.log_info("Auto-upgraded reasoning from minimal to low due to web search")
            
            # Only update keys that are actually in current_values (not None)
            for key, value in current_values.items():
                if value is not None:
                    merged_settings[key] = value
            
            # Special handling: if reasoning_effort not in current values after our adjustment
            if 'reasoning_effort' not in current_values:
                if 'reasoning_effort' in stored_settings:
                    # Restore from stored settings
                    reasoning_from_stored = stored_settings.get('reasoning_effort', 'medium')
                    
                    # If web search is enabled and stored was minimal, upgrade to low
                    if web_search_enabled and reasoning_from_stored == 'minimal':
                        merged_settings['reasoning_effort'] = 'low'
                        self.log_debug("Restored reasoning as low (was minimal, web search on)")
                    # If web search is off, we can restore minimal
                    elif not web_search_enabled:
                        merged_settings['reasoning_effort'] = reasoning_from_stored
                        self.log_debug(f"Restored reasoning as {reasoning_from_stored} from stored settings")
                    else:
                        # Web search is on and wasn't minimal - keep the stored value
                        merged_settings['reasoning_effort'] = reasoning_from_stored
                else:
                    # No stored value either - use a safe default
                    model = merged_settings.get('model', config.gpt_model)
                    merged_settings['reasoning_effort'] = 'low' if web_search_enabled else 'none'
                    self.log_debug(f"No reasoning in form or stored, defaulting to {merged_settings['reasoning_effort']} for model {model}")

            # Final guard: the 5.6 family never accepts `minimal` (and 5.5 has no
            # `max`) — clamp whatever the merge/restore produced so a stale stored
            # value can never be re-stored or reach the API. With web search on,
            # `minimal` upgrades to `low` (handled above) before this maps it to `none`.
            from config import clamp_effort
            _model = merged_settings.get('model', config.gpt_model)
            _effort = merged_settings.get('reasoning_effort')
            if _effort:
                _clamped = clamp_effort(_model, _effort)
                if _clamped != _effort:
                    merged_settings['reasoning_effort'] = _clamped
                    self.log_debug(f"Clamped reasoning {_effort} -> {_clamped} for {_model}")

            # Debug logging
            self.log_debug(f"Features change - Web search: {merged_settings.get('enable_web_search')}, Reasoning: {merged_settings.get('reasoning_effort')}")

            # Update session with merged settings
            session_data['settings'] = merged_settings
            await self._update_session_data(body, session_data)

            # Rebuild modal
            is_new_user = body['view']['callback_id'] == 'welcome_settings_modal'
            updated_modal = await self.settings_modal.build_settings_modal(
                user_id=user_id,
                trigger_id=None,
                current_settings=merged_settings,
                is_new_user=is_new_user,
                thread_id=metadata_context.get('thread_id'),
                in_thread=metadata_context.get('in_thread', False),
                scope=metadata_context.get('scope'),  # Preserve selected scope
                pending_message=metadata_context.get('pending_message')  # Preserve pending message
            )
            
            # Validate the modal before sending (debug)
            for idx, block in enumerate(updated_modal.get('blocks', [])):
                if block.get('type') == 'section' and 'accessory' in block:
                    acc = block['accessory']
                    if 'initial_option' in acc and 'options' in acc:
                        initial_val = acc['initial_option'].get('value')
                        available_vals = [opt['value'] for opt in acc['options']]
                        if initial_val not in available_vals:
                            self.log_error(f"Block {idx} validation failed: initial '{initial_val}' not in options {available_vals}")
            
            # Update the modal
            try:
                # Special case: When minimal is selected and web search is enabled
                # Slack has a bug where it won't select 'low' when minimal is removed from options
                # We need to work around this by forcing the selection in the metadata
                if (stored_settings.get('reasoning_effort') == 'minimal' and
                    merged_settings.get('enable_web_search') and
                    merged_settings.get('reasoning_effort') == 'low'):
                    # Force the reasoning to be properly set in metadata
                    self.log_debug("Forcing reasoning selection from minimal to low due to web search")
                    # Update the metadata to force low reasoning
                    metadata_dict = json.loads(updated_modal.get("private_metadata", "{}"))
                    metadata_dict["settings"]["reasoning_effort"] = "low"  # Force low
                    updated_modal["private_metadata"] = json.dumps(metadata_dict)

                response = await client.views_update(
                    view_id=body['view']['id'],
                    view=updated_modal
                )
                if response.get('ok'):
                    self.log_debug("Modal updated after features change")
                    # Log if reasoning was changed
                    if stored_settings.get('reasoning_effort') != merged_settings.get('reasoning_effort'):
                        self.log_debug(f"Reasoning changed from {stored_settings.get('reasoning_effort')} to {merged_settings.get('reasoning_effort')}")
            except SlackApiError as e:
                self.log_error(f"Error updating modal for features change: {e}")
        
        # Register handler for settings scope toggle
        @self.app.action("settings_scope")
        async def handle_scope_change(ack, body, client):
            """Handle scope toggle between thread and global settings"""
            await ack()

            user_id = body['user']['id']
            selected_scope = body['actions'][0]['selected_option']['value']

            self.log_info(f"Settings scope changed to {selected_scope} for user {user_id}")

            # Get session data from database
            session_data = await self._get_session_data(body)
            if not session_data:
                self.log_error("No session found for modal interaction")
                return

            stored_settings = session_data.get('settings', {})
            metadata_context = {
                'thread_id': session_data.get('thread_id'),
                'in_thread': session_data.get('in_thread', False),
                'scope': selected_scope,  # Update the scope
                'pending_message': session_data.get('pending_message')
            }
            
            # Extract current form values to preserve user's changes
            current_values = self.settings_modal.extract_form_values(body['view']['state'])
            
            # Merge settings
            merged_settings = stored_settings.copy() if stored_settings else {}
            merged_settings.update(current_values)

            # Update session with merged settings and new scope
            session_data['settings'] = merged_settings
            session_data['scope'] = selected_scope
            await self._update_session_data(body, session_data)

            # Rebuild modal with new scope
            is_new_user = body['view']['callback_id'] == 'welcome_settings_modal'
            updated_modal = await self.settings_modal.build_settings_modal(
                user_id=user_id,
                trigger_id=None,
                current_settings=merged_settings,
                is_new_user=is_new_user,
                thread_id=metadata_context.get('thread_id'),
                in_thread=metadata_context.get('in_thread', False),
                scope=selected_scope,  # Pass the new scope
                pending_message=metadata_context.get('pending_message')  # Preserve pending message
            )
            
            # Update the modal
            try:
                response = await client.views_update(
                    view_id=body['view']['id'],
                    view=updated_modal
                )
                if response.get('ok'):
                    self.log_debug(f"Modal updated for scope change to: {selected_scope}")
            except SlackApiError as e:
                self.log_error(f"Error updating modal for scope change: {e}")
        
        # GPT-5.4 reasoning level change needs modal rebuild (to show/hide temp/top_p)
        @self.app.action("reasoning_level_gpt54")
        async def handle_reasoning_level_gpt54_change(ack, body, client):
            """Handle GPT-5.4 reasoning level change - rebuilds modal to show/hide temp/top_p"""
            await ack()

            user_id = body['user']['id']
            selected_reasoning = body['actions'][0]['selected_option']['value']

            self.log_info(f"GPT-5.4 reasoning changed to {selected_reasoning} for user {user_id}")

            # Get session data from database
            session_data = await self._get_session_data(body)
            if not session_data:
                self.log_error("No session found for modal interaction")
                return

            stored_settings = session_data.get('settings', {})
            metadata_context = {
                'thread_id': session_data.get('thread_id'),
                'in_thread': session_data.get('in_thread', False),
                'scope': session_data.get('scope'),
                'pending_message': session_data.get('pending_message')
            }

            # Extract current form values and merge with stored
            current_values = self.settings_modal.extract_form_values(body['view']['state'])
            if isinstance(stored_settings, dict):
                merged_settings = stored_settings.copy()
            else:
                merged_settings = {}
            merged_settings.update(current_values)

            # Ensure the new reasoning level is reflected
            merged_settings['reasoning_effort'] = selected_reasoning

            # Update session
            session_data['settings'] = merged_settings
            await self._update_session_data(body, session_data)

            # Rebuild modal
            is_new_user = body['view']['callback_id'] == 'welcome_settings_modal'
            updated_modal = await self.settings_modal.build_settings_modal(
                user_id=user_id,
                trigger_id=None,
                current_settings=merged_settings,
                is_new_user=is_new_user,
                thread_id=metadata_context.get('thread_id'),
                in_thread=metadata_context.get('in_thread', False),
                scope=metadata_context.get('scope'),
                pending_message=metadata_context.get('pending_message')
            )

            try:
                response = await client.views_update(
                    view_id=body['view']['id'],
                    view=updated_modal
                )
                if response.get('ok'):
                    self.log_debug(f"Modal updated for reasoning change: {selected_reasoning}")
                else:
                    self.log_error(f"Failed to update modal: {response.get('error')}")
            except SlackApiError as e:
                self.log_error(f"Error updating modal for reasoning change: {e}")

        # Register action handlers for other interactive components (just acknowledge)
        @self.app.action("verbosity")
        @self.app.action("input_fidelity")
        @self.app.action("vision_detail")
        @self.app.action("image_size")
        @self.app.action("image_quality")
        @self.app.action("image_background")
        async def handle_modal_actions(ack):
            """Acknowledge modal actions that don't need processing"""
            await ack()  # Just acknowledge - values are captured on submission
        
        # Handler for global settings button in DM
        @self.app.action("open_global_settings_dm")
        async def handle_open_global_settings_dm(ack, body, client):
            """Handle button click to open global settings from DM"""
            # ALWAYS acknowledge first, no matter what
            await ack()
            
            try:
                user_id = body['user']['id']
                trigger_id = body['trigger_id']
                
                # Get user preferences
                user_prefs = await self.db.get_user_preferences_async(user_id)
                if not user_prefs:
                    user_data = await self.db.get_or_create_user_async(user_id)
                    email = user_data.get('email') if user_data else None
                    user_prefs = await self.db.create_default_user_preferences_async(user_id, email)
                
                # Open the settings modal for global settings
                modal = await self.settings_modal.build_settings_modal(
                    user_id=user_id,
                    trigger_id=trigger_id,
                    current_settings=user_prefs,
                    is_new_user=False,  # Not a new user if they're clicking this
                    thread_id=None,
                    in_thread=False,  # Always global from DM button
                    scope='global'
                )

                response = await client.views_open(
                    trigger_id=trigger_id,
                    view=modal
                )
                
                if response.get('ok'):
                    self.log_info(f"Global settings modal opened from DM for user {user_id}")
                else:
                    self.log_warning(f"Failed to open global settings modal: {response}")
                    
            except Exception as e:
                self.log_error(f"Error in handle_open_global_settings_dm: {e}", exc_info=True)
        
        # Handler for welcome settings button
        @self.app.action("open_welcome_settings")
        async def handle_open_welcome_settings(ack, body, client):
            """Handle button click to open welcome settings modal"""
            # ALWAYS acknowledge first, no matter what
            await ack()
            
            user_id = body['user']['id']
            trigger_id = body['trigger_id']
            
            # Extract the original message details from the button value
            button_value = body['actions'][0].get('value', '{}')
            try:
                original_context = json.loads(button_value)
            except Exception:
                original_context = {}
            
            # Check if this was a truncated message that needs to be fetched
            if original_context.get('truncated'):
                # Fetch the original message from Slack using the timestamp
                channel_id = original_context.get('channel_id')
                ts = original_context.get('ts')
                self.log_info(f"Fetching truncated message from Slack: channel={channel_id}, ts={ts}")
                
                try:
                    # Get the original message from Slack
                    result = await client.conversations_history(
                        channel=channel_id,
                        latest=ts,
                        oldest=ts,
                        inclusive=True,
                        limit=1
                    )
                    
                    if result.get('ok') and result.get('messages'):
                        msg = result['messages'][0]
                        # Reconstruct the full context from the fetched message
                        original_context = {
                            "original_message": msg.get('text', ''),
                            "channel_id": channel_id,
                            "thread_id": original_context.get('thread_id'),
                            "attachments": []  # We'll process files if they exist
                        }
                        
                        # Process any file attachments
                        files = msg.get('files', [])
                        for file in files:
                            mimetype = file.get("mimetype", "")
                            file_type = "image" if mimetype.startswith("image/") else "file"
                            original_context['attachments'].append({
                                "type": file_type,
                                "url": file.get("url_private"),
                                "id": file.get("id"),
                                "name": file.get("name"),
                                "mimetype": mimetype
                            })
                        
                        self.log_info(f"Successfully fetched truncated message with {len(original_context['attachments'])} attachments")
                    else:
                        self.log_warning(f"Could not fetch truncated message: {result.get('error', 'Unknown error')}")
                        # Keep the truncated context as-is
                except Exception as e:
                    self.log_error(f"Error fetching truncated message: {e}", exc_info=True)
                    # Keep the truncated context as-is
            
            # Get or create user preferences
            user_data = await self.db.get_or_create_user_async(user_id)
            email = user_data.get('email') if user_data else None
            user_prefs = await self.db.get_user_preferences_async(user_id)

            if not user_prefs:
                # Create default preferences if they don't exist
                user_prefs = await self.db.create_default_user_preferences_async(user_id, email)
            
            # Track if this is a new user based on settings_completed flag
            is_new_user = not user_prefs.get('settings_completed', False)
            
            # Open the settings modal
            try:
                # Determine if we're in a thread based on the button context
                thread_id = original_context.get('thread_id')
                channel_id = original_context.get('channel_id')
                
                # Check if this is actually a thread (not just a main channel message)
                in_thread = False
                if thread_id and channel_id:
                    # In channels, if thread_id exists and is different from the channel, it's a thread
                    # In DMs, every message has a thread_id, so we always consider it a thread context
                    is_dm = channel_id.startswith('D')
                    in_thread = is_dm or (thread_id != channel_id)
                    
                    # Format thread_id properly if in thread
                    if in_thread and ':' not in thread_id:
                        thread_id = f"{channel_id}:{thread_id}"
                
                self.log_debug(f"Opening modal from button - thread_id: {thread_id}, channel_id: {channel_id}, in_thread: {in_thread}")
                
                # If we're in a thread, check for thread-specific settings
                thread_settings = None
                if in_thread and thread_id:
                    thread_config = await self.db.get_thread_config_async(thread_id)
                    if thread_config:
                        # Merge thread config with user prefs (thread overrides)
                        thread_settings = user_prefs.copy()
                        thread_settings.update(thread_config)
                        self.log_debug(f"Loaded thread config for {thread_id}: {thread_config}")
                    else:
                        self.log_debug(f"No thread config found for {thread_id}, using user prefs")
                
                # Use thread settings if available when in thread, otherwise user prefs
                current_settings = thread_settings if thread_settings else user_prefs

                # Prepare pending_message for new users
                pending_message = None
                if is_new_user and original_context and original_context.get('original_message'):
                    # Check if the message would fit in metadata
                    test_metadata = {
                        "settings": current_settings,
                        "thread_id": thread_id if in_thread else None,
                        "in_thread": in_thread,
                        "scope": "global",
                        "pending_message": original_context
                    }

                    if len(json.dumps(test_metadata)) > 3000:
                        # Message too long - use truncated version
                        self.log_info(f"Pending message too long for metadata ({len(json.dumps(test_metadata))} chars), will ask user to resend")
                        pending_message = {
                            'too_long': True,
                            'channel_id': original_context.get('channel_id'),
                            'thread_id': original_context.get('thread_id')
                        }
                    else:
                        # Message fits, use it normally
                        pending_message = original_context
                        self.log_info(f"Including pending message for new user: {original_context.get('original_message', '')[:50]}...")
                else:
                    self.log_debug(f"Not adding pending message - is_new_user: {is_new_user}, has_message: {bool(original_context and original_context.get('original_message'))}")

                # Build modal with pending_message
                modal = await self.settings_modal.build_settings_modal(
                    user_id=user_id,
                    trigger_id=trigger_id,
                    current_settings=current_settings,
                    is_new_user=is_new_user,
                    thread_id=thread_id if in_thread else None,
                    in_thread=in_thread,
                    pending_message=pending_message  # Pass the pending message directly
                )

                # Log the metadata size for debugging
                if is_new_user and pending_message:
                    self.log_debug(f"Metadata size for new user: {len(modal.get('private_metadata', '{}'))} chars")
                
                response = await client.views_open(
                    trigger_id=trigger_id,
                    view=modal
                )
                
                if response.get('ok'):
                    self.log_info(f"Welcome modal opened via button for user {user_id}")
                    
                    # Keep the button message for future access
                    # (removed deletion to allow persistent settings access)
                        
            except Exception as e:
                self.log_error(f"Error in handle_open_welcome_settings: {e}", exc_info=True)
        
        # Register message shortcut for thread-specific settings
        @self.app.shortcut("configure_thread_settings_dev")  # Dev callback ID
        @self.app.shortcut("configure_thread_settings")  # Prod callback ID (when configured)
        async def handle_thread_settings_shortcut(ack, shortcut, client):
            """Handle the thread settings message shortcut"""
            # ALWAYS acknowledge first, no matter what
            await ack()
            
            # Get thread context from the shortcut - this is reliable!
            channel_id = shortcut["channel"]["id"]
            message = shortcut["message"]
            thread_ts = message.get("thread_ts") or message["ts"]  # Use thread_ts if in thread, else message ts
            thread_id = f"{channel_id}:{thread_ts}"
            user_id = shortcut["user"]["id"]
            
            self.log_info(f"Thread settings shortcut invoked for thread {thread_id} by user {user_id}")
            
            # Load existing thread config if it exists
            thread_config = await self.db.get_thread_config_async(thread_id)

            # Get user preferences as base
            user_settings = await self.db.get_user_preferences_async(user_id)
            if not user_settings:
                # Create defaults if user has no settings yet
                user_data = await self.db.get_or_create_user_async(user_id)
                email = user_data.get('email') if user_data else None
                user_settings = await self.db.create_default_user_preferences_async(user_id, email)
            
            # Merge thread config over user settings for display
            current_settings = user_settings.copy()
            if thread_config:
                current_settings.update(thread_config)
            
            # Build modal specifically for thread settings
            try:
                modal = await self.settings_modal.build_settings_modal(
                    user_id=user_id,
                    trigger_id=shortcut["trigger_id"],
                    current_settings=current_settings,
                    is_new_user=False,
                    thread_id=thread_id,
                    in_thread=True  # Always true for message shortcuts
                )
                # Keep default title "ChatGPT Settings (Dev)" from settings_modal.py
                # The header inside will say "Configure Thread Preferences"
                
                # Open the modal
                response = await client.views_open(
                    trigger_id=shortcut["trigger_id"],
                    view=modal
                )
                
                if response.get('ok'):
                    self.log_info(f"Thread settings modal opened for thread {thread_id}")
                else:
                    self.log_error(f"Failed to open thread modal: {response.get('error')}")
                    
            except Exception as e:
                self.log_error(f"Error in handle_thread_settings_shortcut: {e}", exc_info=True)
