from __future__ import annotations

import json
import time
from typing import Any, Dict

from slack_sdk.errors import SlackApiError

from base_client import Message
from config import config


class SlackEventHandlersMixin:
    def _register_handlers(self):
        """Register Slack-specific event handlers"""
        
        @self.app.event("app_mention")
        def handle_app_mention(event, say, client):
            self.log_debug(f"App mention event: channel={event.get('channel')}, ts={event.get('ts')}")
            self._handle_slack_message(event, client)
        
        @self.app.event("message")
        def handle_message(event, say, client):
            # Only process DMs and non-bot messages
            if event.get("channel_type") == "im" and not event.get("bot_id"):
                self.log_debug(f"DM message event: channel={event.get('channel')}, ts={event.get('ts')}")
                self._handle_slack_message(event, client)
        
        # Register slash command handler
        @self.app.command(config.settings_slash_command)
        def handle_settings_command(ack, body, client):
            """Handle the settings slash command"""
            ack()  # Acknowledge command receipt immediately
            
            user_id = body.get('user_id')
            trigger_id = body.get('trigger_id')
            
            # Slash commands are always for global settings
            # Thread-specific settings use the message shortcut instead
            
            self.log_info(f"Settings command invoked by user {user_id}")
            
            # Get current settings or create defaults
            current_settings = self.db.get_user_preferences(user_id)
            is_new_user = current_settings is None
            
            if is_new_user:
                # Get user's email for preferences
                user_data = self.db.get_or_create_user(user_id)
                email = user_data.get('email') if user_data else None
                current_settings = self.db.create_default_user_preferences(user_id, email)
            
            # Build and open modal for global settings
            try:
                modal = self.settings_modal.build_settings_modal(
                    user_id=user_id,
                    trigger_id=trigger_id,
                    current_settings=current_settings,
                    is_new_user=is_new_user,
                    thread_id=None,  # Always None for slash commands
                    in_thread=False  # Always False for slash commands
                )
                # Keep default title "ChatGPT Settings (Dev)" from settings_modal.py
                
                # Open the modal
                response = client.views_open(
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
                    client.chat_postEphemeral(
                        channel=body.get('channel_id'),
                        user=user_id,
                        text="❌ Sorry, I couldn't open the settings modal. Please try again."
                    )
                except Exception:
                    pass
        
        # Register modal submission handlers
        @self.app.view("settings_modal")
        @self.app.view("welcome_settings_modal")
        def handle_settings_submission(ack, body, view, client):
            """Handle settings modal submission"""
            
            user_id = body['user']['id']
            
            # Extract metadata to determine context (thread vs global)
            metadata = json.loads(view.get('private_metadata', '{}'))
            thread_id = metadata.get('thread_id')
            in_thread = metadata.get('in_thread', False)
            pending_message = metadata.get('pending_message')  # Get any pending message to process
            
            # Determine if this is a new user
            is_new_user = view.get('callback_id') == 'welcome_settings_modal'
            
            # For new users, always save to global regardless of scope selection
            if is_new_user:
                selected_scope = 'global'
                self.log_info("New user setup - forcing save to global settings")
            else:
                selected_scope = metadata.get('scope', 'thread' if in_thread else 'global')  # Get selected scope
            
            # Debug logging
            self.log_debug(f"Modal submission metadata: {metadata}")
            self.log_debug(f"Selected scope for save: {selected_scope} (is_new_user: {is_new_user})")
            if pending_message:
                self.log_info(f"Found pending message in metadata: {pending_message}")
            
            # Extract form values
            form_values = self.settings_modal.extract_form_values(view['state'])
            
            # Check if we need confirmation for global custom instructions from thread
            needs_confirmation = False
            if (in_thread and selected_scope == 'global' and 
                form_values.get('custom_instructions') and 
                not metadata.get('confirmed')):
                
                # Check if there are existing global custom instructions
                existing_prefs = self.db.get_user_preferences(user_id)
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
                ack(response_action="push", view=confirmation_modal)
                return
            
            # Normal flow - acknowledge immediately
            ack()
            
            # Validate settings
            validated_settings = self.settings_modal.validate_settings(form_values)
            
            # Check for model switch warning (GPT-5 -> GPT-4)
            model_switch_warning = ""
            if selected_scope == 'thread' and thread_id:
                # For thread settings, get the current thread config from DB to check for model switch
                try:
                    current_thread_config = self.db.get_thread_config(thread_id)
                    if current_thread_config:
                        old_model = current_thread_config.get('model')
                        new_model = validated_settings.get('model')
                        
                        # Check if switching from GPT-5 family to GPT-4 family
                        if old_model and new_model:
                            if old_model.startswith('gpt-5') and new_model.startswith('gpt-4'):
                                model_switch_warning = "\n\n⚠️ **Important:** Switching to GPT-4 may require removing some older messages from long conversations due to its smaller context window (128k vs 400k tokens). Recent messages and important content like images will be preserved."
                except Exception as e:
                    self.log_debug(f"Could not check for model switch: {e}")
            else:
                # For global settings, check current default
                old_model = self.db.get_user_preferences(user_id).get('model') if self.db else config.gpt_model
                new_model = validated_settings.get('model')
                
                if old_model and new_model:
                    if old_model.startswith('gpt-5') and new_model.startswith('gpt-4'):
                        model_switch_warning = "\n\n⚠️ **Note:** You're switching to GPT-4 which has a smaller context window than GPT-5 (128k vs 400k tokens). Long conversations may have older messages automatically removed to fit within the limit."
            
            # Check for temperature/top_p warning
            warning_message = ""
            if validated_settings.get('model') not in ['gpt-5', 'gpt-5-mini', 'gpt-5-nano']:
                temp_changed = validated_settings.get('temperature', config.default_temperature) != config.default_temperature
                top_p_changed = validated_settings.get('top_p', config.default_top_p) != config.default_top_p
                
                if temp_changed and top_p_changed:
                    warning_message = "\n⚠️ Note: You've changed both Temperature and Top P. OpenAI recommends using only one for best results."
            
            # Combine warnings
            warning_message = warning_message + model_switch_warning
            
            # Save to appropriate location based on selected scope
            if selected_scope == 'thread' and thread_id:
                # Save as thread config (don't include settings_completed flag for thread configs)
                thread_settings = {k: v for k, v in validated_settings.items() if k != 'settings_completed'}
                # Ensure thread exists
                channel_id, thread_ts = thread_id.split(':')
                self.db.get_or_create_thread(thread_id, channel_id)
                self.db.save_thread_config(thread_id, thread_settings)
                
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
                success = self.db.update_user_preferences(user_id, validated_settings)
                save_location = "global"
                self.log_info(f"Global settings saved for user {user_id}: {validated_settings}")
            
            if success:
                # Send confirmation message
                try:
                    # Determine message based on save location
                    if save_location == "thread" and thread_id:
                        # Send ephemeral confirmation in the thread
                        channel_id, thread_ts = thread_id.split(':')
                        client.chat_postEphemeral(
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
                                            "text": "Open Settings"
                                        },
                                        "style": "primary",
                                        "action_id": "open_global_settings_dm"
                                    }
                                ]
                            }
                        ]
                        
                        client.chat_postMessage(
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
                            client.chat_delete(
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
                        client.chat_update(
                            channel=welcome_info['channel'],
                            ts=welcome_info['ts'],
                            text="Settings available",
                            blocks=[
                                {
                                    "type": "section",
                                    "text": {
                                        "type": "mrkdwn",
                                        "text": "⚙️ *Quick Settings Access*"
                                    }
                                },
                                {
                                    "type": "actions",
                                    "elements": [
                                        {
                                            "type": "button",
                                            "text": {
                                                "type": "plain_text",
                                                "text": "Settings"
                                            },
                                            "style": "primary",
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
                            client.chat_postMessage(
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
                            self._handle_slack_message(synthetic_event, client)

                        except Exception as e:
                            self.log_error(f"Error processing pending message: {e}")
                            # Send error message to user
                            try:
                                client.chat_postMessage(
                                    channel=pending_message['channel_id'],
                                    thread_ts=pending_message.get('thread_id'),
                                    text="I'm ready now! Could you please repeat your question?"
                                )
                            except Exception:
                                pass
                            
            else:
                self.log_error(f"Failed to save settings for user {user_id}")
                try:
                    client.chat_postMessage(
                        channel=user_id,
                        text="❌ Sorry, there was an error saving your settings. Please try again."
                    )
                except Exception:
                    pass
        
        # Handler for custom instructions confirmation modal submission
        @self.app.view("confirm_global_custom_instructions")
        def handle_custom_instructions_confirmation(ack, body, view, client):
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
            success = self.db.update_user_preferences(user_id, validated_settings)
            
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
                                        "text": "Open Settings"
                                    },
                                    "style": "primary",
                                    "action_id": "open_global_settings_dm"
                                }
                            ]
                        }
                    ]
                    
                    client.chat_postMessage(
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
        def handle_model_change(ack, body, client):
            """Handle model selection changes for dynamic modal updates"""
            ack()
            
            user_id = body['user']['id']
            selected_model = body['actions'][0]['selected_option']['value']
            
            self.log_info(f"Model selection changed to {selected_model} for user {user_id}")
            
            # Try to get stored settings from private_metadata first (preserves unsaved changes)
            stored_settings = {}
            metadata_context = {}
            try:
                private_metadata = body.get('view', {}).get('private_metadata')
                if private_metadata:
                    metadata = json.loads(private_metadata)
                    if isinstance(metadata, dict) and 'settings' in metadata:
                        # New format with context
                        stored_settings = metadata['settings']
                        metadata_context = {
                            'thread_id': metadata.get('thread_id'),
                            'in_thread': metadata.get('in_thread', False),
                            'scope': metadata.get('scope')  # Extract scope
                        }
                    else:
                        # Old format - just settings
                        stored_settings = metadata
            except Exception:
                pass
            
            # If no stored settings, get user's saved preferences
            if not stored_settings:
                saved_settings = self.db.get_user_preferences(user_id)
                if not saved_settings:
                    # If no saved settings, get defaults
                    user_data = self.db.get_or_create_user(user_id)
                    email = user_data.get('email') if user_data else None
                    saved_settings = self.db.create_default_user_preferences(user_id, email)
                stored_settings = saved_settings
            
            # Extract current form values (only gets visible fields)
            current_values = self.settings_modal.extract_form_values(body['view']['state'])
            
            # Merge: stored settings as base, current values override what's visible
            # This preserves all field values when switching models
            if isinstance(stored_settings, dict):
                merged_settings = stored_settings.copy()
            else:
                merged_settings = {}
            merged_settings.update(current_values)
            
            # Build updated modal with new model selection
            is_new_user = body['view']['callback_id'] == 'welcome_settings_modal'
            updated_modal = self.settings_modal.build_settings_modal(
                user_id=user_id,
                trigger_id=None,  # Not needed for update
                current_settings=merged_settings,
                is_new_user=is_new_user,
                thread_id=metadata_context.get('thread_id'),
                in_thread=metadata_context.get('in_thread', False),
                scope=metadata_context.get('scope')  # Preserve selected scope
            )
            
            # Update the modal view
            try:
                response = client.views_update(
                    view_id=body['view']['id'],
                    view=updated_modal
                )
                
                if response.get('ok'):
                    self.log_debug(f"Modal updated for model change: {selected_model}")
                else:
                    self.log_error(f"Failed to update modal: {response.get('error')}")
                    
            except SlackApiError as e:
                self.log_error(f"Error updating modal for model change: {e}")
        
        # Register handler for features checkbox (needs modal rebuild for web search)
        @self.app.action("features")
        def handle_features_change(ack, body, client):
            """Handle feature checkbox changes, especially web search"""
            ack()
            
            user_id = body['user']['id']
            
            # Extract current settings and context from metadata
            stored_settings = {}
            metadata_context = {}
            try:
                private_metadata = body.get('view', {}).get('private_metadata')
                if private_metadata:
                    metadata = json.loads(private_metadata)
                    if isinstance(metadata, dict) and 'settings' in metadata:
                        stored_settings = metadata['settings']
                        metadata_context = {
                            'thread_id': metadata.get('thread_id'),
                            'in_thread': metadata.get('in_thread', False),
                            'scope': metadata.get('scope')  # Extract scope
                        }
                    else:
                        stored_settings = metadata
            except Exception:
                pass
            
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
                    merged_settings['reasoning_effort'] = 'low' if web_search_enabled else 'medium'
                    self.log_debug(f"No reasoning in form or stored, defaulting to {merged_settings['reasoning_effort']}")
            
            # Debug logging
            self.log_debug(f"Features change - Web search: {merged_settings.get('enable_web_search')}, Reasoning: {merged_settings.get('reasoning_effort')}")
            
            # Rebuild modal
            is_new_user = body['view']['callback_id'] == 'welcome_settings_modal'
            updated_modal = self.settings_modal.build_settings_modal(
                user_id=user_id,
                trigger_id=None,
                current_settings=merged_settings,
                is_new_user=is_new_user,
                thread_id=metadata_context.get('thread_id'),
                in_thread=metadata_context.get('in_thread', False),
                scope=metadata_context.get('scope')  # Preserve selected scope
            )
            
            # Update private metadata
            updated_modal["private_metadata"] = json.dumps({
                "settings": merged_settings,
                "thread_id": metadata_context.get('thread_id'),
                "in_thread": metadata_context.get('in_thread', False),
                "scope": metadata_context.get('scope')  # Preserve selected scope
            })
            
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
                    # Make sure the metadata reflects the change
                    updated_modal["private_metadata"] = json.dumps({
                        "settings": {**merged_settings, 'reasoning_effort': 'low'},  # Force low
                        "thread_id": metadata_context.get('thread_id'),
                        "in_thread": metadata_context.get('in_thread', False),
                        "scope": metadata_context.get('scope')  # Preserve selected scope
                    })
                
                response = client.views_update(
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
        def handle_scope_change(ack, body, client):
            """Handle scope toggle between thread and global settings"""
            ack()
            
            user_id = body['user']['id']
            selected_scope = body['actions'][0]['selected_option']['value']
            
            self.log_info(f"Settings scope changed to {selected_scope} for user {user_id}")
            
            # Get current metadata context
            metadata_context = {}
            stored_settings = {}
            try:
                private_metadata = body.get('view', {}).get('private_metadata')
                if private_metadata:
                    metadata = json.loads(private_metadata)
                    if isinstance(metadata, dict):
                        stored_settings = metadata.get('settings', {})
                        metadata_context = {
                            'thread_id': metadata.get('thread_id'),
                            'in_thread': metadata.get('in_thread', False),
                            'scope': selected_scope  # Update the scope
                        }
            except Exception:
                pass
            
            # Extract current form values to preserve user's changes
            current_values = self.settings_modal.extract_form_values(body['view']['state'])
            
            # Merge settings
            merged_settings = stored_settings.copy() if stored_settings else {}
            merged_settings.update(current_values)
            
            # Rebuild modal with new scope
            is_new_user = body['view']['callback_id'] == 'welcome_settings_modal'
            updated_modal = self.settings_modal.build_settings_modal(
                user_id=user_id,
                trigger_id=None,
                current_settings=merged_settings,
                is_new_user=is_new_user,
                thread_id=metadata_context.get('thread_id'),
                in_thread=metadata_context.get('in_thread', False),
                scope=selected_scope  # Pass the new scope
            )
            
            # Update the modal
            try:
                response = client.views_update(
                    view_id=body['view']['id'],
                    view=updated_modal
                )
                if response.get('ok'):
                    self.log_debug(f"Modal updated for scope change to: {selected_scope}")
            except SlackApiError as e:
                self.log_error(f"Error updating modal for scope change: {e}")
        
        # Register action handlers for other interactive components (just acknowledge)
        @self.app.action("reasoning_level")
        @self.app.action("reasoning_level_no_minimal")  # Alternative action_id when minimal is hidden
        @self.app.action("verbosity")
        @self.app.action("input_fidelity")
        @self.app.action("vision_detail")
        @self.app.action("image_size")
        def handle_modal_actions(ack):
            """Acknowledge modal actions that don't need processing"""
            ack()  # Just acknowledge - values are captured on submission
        
        # Handler for global settings button in DM
        @self.app.action("open_global_settings_dm")
        def handle_open_global_settings_dm(ack, body, client):
            """Handle button click to open global settings from DM"""
            # ALWAYS acknowledge first, no matter what
            ack()
            
            try:
                user_id = body['user']['id']
                trigger_id = body['trigger_id']
                
                # Get user preferences
                user_prefs = self.db.get_user_preferences(user_id)
                if not user_prefs:
                    user_data = self.db.get_or_create_user(user_id)
                    email = user_data.get('email') if user_data else None
                    user_prefs = self.db.create_default_user_preferences(user_id, email)
                
                # Open the settings modal for global settings
                modal = self.settings_modal.build_settings_modal(
                    user_id=user_id,
                    trigger_id=trigger_id,
                    current_settings=user_prefs,
                    is_new_user=False,  # Not a new user if they're clicking this
                    thread_id=None,
                    in_thread=False,  # Always global from DM button
                    scope='global'
                )
                
                response = client.views_open(
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
        def handle_open_welcome_settings(ack, body, client):
            """Handle button click to open welcome settings modal"""
            # ALWAYS acknowledge first, no matter what
            ack()
            
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
                    result = client.conversations_history(
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
            user_data = self.db.get_or_create_user(user_id)
            email = user_data.get('email') if user_data else None
            user_prefs = self.db.get_user_preferences(user_id)
            
            if not user_prefs:
                # Create default preferences if they don't exist
                user_prefs = self.db.create_default_user_preferences(user_id, email)
            
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
                    thread_config = self.db.get_thread_config(thread_id)
                    if thread_config:
                        # Merge thread config with user prefs (thread overrides)
                        thread_settings = user_prefs.copy()
                        thread_settings.update(thread_config)
                        self.log_debug(f"Loaded thread config for {thread_id}: {thread_config}")
                    else:
                        self.log_debug(f"No thread config found for {thread_id}, using user prefs")
                
                # Use thread settings if available when in thread, otherwise user prefs
                current_settings = thread_settings if thread_settings else user_prefs
                
                modal = self.settings_modal.build_settings_modal(
                    user_id=user_id,
                    trigger_id=trigger_id,
                    current_settings=current_settings,
                    is_new_user=is_new_user,
                    thread_id=thread_id if in_thread else None,
                    in_thread=in_thread
                )
                
                # Only add the original message context for new users who need to reprocess their message
                if is_new_user and original_context and original_context.get('original_message'):
                    existing_metadata = json.loads(modal.get('private_metadata', '{}'))
                    # Check if adding the message would exceed Slack's limit
                    test_metadata = existing_metadata.copy()
                    test_metadata['pending_message'] = original_context

                    if len(json.dumps(test_metadata)) > 3000:
                        # Message too long - skip storing it and notify user later
                        self.log_info(f"Pending message too long for metadata ({len(json.dumps(test_metadata))} chars), will ask user to resend")
                        # Store a flag that we need to ask them to resend
                        existing_metadata['pending_message'] = {
                            'too_long': True,
                            'channel_id': original_context.get('channel_id'),
                            'thread_id': original_context.get('thread_id')
                        }
                    else:
                        # Message fits, store it normally
                        existing_metadata['pending_message'] = original_context

                    modal['private_metadata'] = json.dumps(existing_metadata)
                    self.log_debug(f"Metadata size for new user: {len(json.dumps(existing_metadata))} chars")
                else:
                    self.log_debug(f"Not adding pending message - is_new_user: {is_new_user}, has_message: {bool(original_context and original_context.get('original_message'))}")
                
                response = client.views_open(
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
        def handle_thread_settings_shortcut(ack, shortcut, client):
            """Handle the thread settings message shortcut"""
            # ALWAYS acknowledge first, no matter what
            ack()
            
            # Get thread context from the shortcut - this is reliable!
            channel_id = shortcut["channel"]["id"]
            message = shortcut["message"]
            thread_ts = message.get("thread_ts") or message["ts"]  # Use thread_ts if in thread, else message ts
            thread_id = f"{channel_id}:{thread_ts}"
            user_id = shortcut["user"]["id"]
            
            self.log_info(f"Thread settings shortcut invoked for thread {thread_id} by user {user_id}")
            
            # Load existing thread config if it exists
            thread_config = self.db.get_thread_config(thread_id)
            
            # Get user preferences as base
            user_settings = self.db.get_user_preferences(user_id)
            if not user_settings:
                # Create defaults if user has no settings yet
                user_data = self.db.get_or_create_user(user_id)
                email = user_data.get('email') if user_data else None
                user_settings = self.db.create_default_user_preferences(user_id, email)
            
            # Merge thread config over user settings for display
            current_settings = user_settings.copy()
            if thread_config:
                current_settings.update(thread_config)
            
            # Build modal specifically for thread settings
            try:
                modal = self.settings_modal.build_settings_modal(
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
                response = client.views_open(
                    trigger_id=shortcut["trigger_id"],
                    view=modal
                )
                
                if response.get('ok'):
                    self.log_info(f"Thread settings modal opened for thread {thread_id}")
                else:
                    self.log_error(f"Failed to open thread modal: {response.get('error')}")
                    
            except Exception as e:
                self.log_error(f"Error in handle_thread_settings_shortcut: {e}", exc_info=True)

    def _handle_slack_message(self, event: Dict[str, Any], client):
        """Convert Slack event to universal Message format"""
        
        # Skip message_changed events
        if event.get("subtype") == "message_changed":
            return
        
        # Extract and clean text
        text = event.get("text", "")
        text = self._clean_mentions(text)
        
        # Process attachments (files)
        attachments = []
        files = event.get("files", [])
        for file in files:
            mimetype = file.get("mimetype", "")
            # Determine file type based on mimetype
            file_type = "image" if mimetype.startswith("image/") else "file"
            
            attachments.append({
                "type": file_type,
                "url": file.get("url_private"),
                "id": file.get("id"),
                "name": file.get("name"),
                "mimetype": mimetype
            })
        
        # Get username and timezone for logging
        user_id = event.get("user")
        username = self.get_username(user_id, client) if user_id else "unknown"
        user_timezone = self.get_user_timezone(user_id, client) if user_id else "UTC"
        
        # Get timezone label (EST, PST, etc.), real name, and email if available
        user_tz_label = None
        user_real_name = None
        user_email = None
        if user_id in self.user_cache:
            user_tz_label = self.user_cache[user_id].get('tz_label')
            user_real_name = self.user_cache[user_id].get('real_name')
            user_email = self.user_cache[user_id].get('email')
            self.log_debug(f"User cache for {user_id}: email={user_email}, real_name={user_real_name}")
        else:
            # Try to get from database if not in cache
            user_info = self.db.get_user_info(user_id)
            if user_info:
                user_real_name = user_info.get('real_name')
                user_email = user_info.get('email')
                user_tz_label = user_info.get('tz_label')
                self.log_debug(f"User from DB for {user_id}: email={user_email}, real_name={user_real_name}")
        
        # Create universal message
        message = Message(
            text=text,
            user_id=user_id,
            channel_id=event.get("channel"),
            thread_id=event.get("thread_ts") or event.get("ts"),
            attachments=attachments,
            metadata={
                "ts": event.get("ts"),
                "slack_client": client,
                "username": username,  # Add username to metadata
                "user_real_name": user_real_name,  # Add real name to metadata
                "user_email": user_email,  # Add email to metadata
                "user_timezone": user_timezone,  # Add timezone to metadata
                "user_tz_label": user_tz_label  # Add timezone label (EST, PST, etc.)
            }
        )
        
        # Check if this is a new user (for auto-modal trigger)
        user_prefs = self.db.get_user_preferences(user_id)
        
        if not user_prefs:
            # Create default preferences for new user
            user_data = self.db.get_or_create_user(user_id)
            email = user_data.get('email') if user_data else None
            user_prefs = self.db.create_default_user_preferences(user_id, email)
            self.log_info(f"Created default preferences for new user {user_id}")
        
        # Check if user has completed settings
        if not user_prefs.get('settings_completed', False):
            # User hasn't completed settings - check if we've already sent welcome
            if not hasattr(self, '_welcomed_users'):
                self._welcomed_users = set()
            
            # Check if this is their first message this session
            is_first_message = user_id not in self._welcomed_users
            
            if is_first_message:
                # Mark as welcomed and send welcome button
                self._welcomed_users.add(user_id)
                
            # Check if we have a trigger_id for modal
            trigger_id = event.get('trigger_id')
            
            if trigger_id and is_first_message:
                # Create default preferences
                user_data = self.db.get_or_create_user(user_id)
                email = user_data.get('email') if user_data else None
                default_prefs = self.db.create_default_user_preferences(user_id, email)
                
                # Open welcome modal
                try:
                    modal = self.settings_modal.build_settings_modal(
                        user_id=user_id,
                        trigger_id=trigger_id,
                        current_settings=default_prefs,
                        is_new_user=True
                    )
                    
                    response = client.views_open(
                        trigger_id=trigger_id,
                        view=modal
                    )
                    
                    if response.get('ok'):
                        self.log_info(f"Welcome modal opened for new user {user_id}")
                        
                        # Send welcome message
                        client.chat_postMessage(
                            channel=message.channel_id,
                            thread_ts=message.thread_id,
                            text="👋 Welcome! I've opened your settings panel. Please configure your preferences and I'll be ready to help!"
                        )
                        return  # Don't process the message until settings are saved
                    
                except SlackApiError as e:
                    self.log_error(f"Error opening welcome modal for new user: {e}")
                    # Continue with processing using defaults
            elif is_first_message:
                # No trigger_id available, first message - send interactive message with button
                try:
                    # Prepare button value with size check
                    full_context = {
                        "original_message": message.text,
                        "channel_id": message.channel_id,
                        "thread_id": message.thread_id,
                        "attachments": message.attachments,  # Include file attachments
                        "ts": event.get("ts")  # Include timestamp for proper threading
                    }
                    
                    # Check if button value would exceed Slack's 2000 char limit (with buffer)
                    full_value = json.dumps(full_context)
                    if len(full_value) > 1900:  # Leave 100 char buffer
                        # Fallback: only store reference data
                        button_value = json.dumps({
                            "channel_id": message.channel_id,
                            "thread_id": message.thread_id,
                            "ts": event.get("ts"),  # Add timestamp to fetch message later
                            "has_attachments": bool(message.attachments),
                            "attachment_count": len(message.attachments),
                            "truncated": True
                        })
                        self.log_info(f"Welcome button value too large ({len(full_value)} chars), using truncated version")
                    else:
                        button_value = full_value
                    
                    # Check if we're in a channel/thread vs DM
                    is_dm = message.channel_id.startswith('D')
                    
                    if is_dm:
                        # For DMs, send the button in the same conversation
                        target_channel = message.channel_id
                        target_thread = message.thread_id
                    else:
                        # For channels/threads, send as a DM to the user
                        target_channel = user_id  # Send to user's DM
                        target_thread = None  # No thread in DM
                        
                        # Also send a brief message in the thread to acknowledge
                        client.chat_postMessage(
                            channel=message.channel_id,
                            thread_ts=message.thread_id,
                            text="👋 Welcome! I've sent you a direct message to configure your settings."
                        )
                    
                    # Send welcome button on first interaction
                    # On subsequent messages, the ephemeral will be sent from the outer check
                    
                    # Build blocks for welcome message
                    blocks = [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "👋 *Welcome to the AI Assistant!*\n\nI need you to configure your preferences before we begin. Click the button below to open your settings:"
                            }
                        },
                        {
                            "type": "actions",
                            "elements": [
                                {
                                    "type": "button",
                                    "text": {
                                        "type": "plain_text",
                                        "text": "⚙️ Configure Settings"
                                    },
                                    "style": "primary",
                                    "action_id": "open_welcome_settings",
                                    "value": button_value
                                }
                            ]
                        }
                    ]
                    
                    # No need to warn user - we handle truncation transparently
                    
                    response = client.chat_postMessage(
                        channel=target_channel,
                        thread_ts=target_thread,
                        text="👋 Welcome! Please configure your settings to get started.",
                        blocks=blocks
                    )
                    
                    # Track welcome message for updating after settings saved
                    if response.get('ok'):
                        if not hasattr(self, '_welcome_messages'):
                            self._welcome_messages = {}
                        self._welcome_messages[user_id] = {
                            'channel': target_channel,
                            'ts': response.get('ts'),
                            'thread_ts': target_thread
                        }
                    
                    return  # Don't process until settings are configured
                except SlackApiError as e:
                    self.log_error(f"Error sending welcome message: {e}")
            else:
                # Not first message - send regular reminder that we can delete later
                try:
                    response = client.chat_postMessage(
                        channel=message.channel_id,
                        thread_ts=message.thread_id,
                        text="⚠️ Please configure your settings before I can help you. Click the *Configure Settings* button above to get started."
                    )
                    # Track reminder message for cleanup
                    if response.get('ok'):
                        if not hasattr(self, '_reminder_messages'):
                            self._reminder_messages = {}
                        if user_id not in self._reminder_messages:
                            self._reminder_messages[user_id] = []
                        self._reminder_messages[user_id].append({
                            'channel': message.channel_id,
                            'ts': response.get('ts')
                        })
                except Exception as e:
                    self.log_debug(f"Could not send reminder: {e}")
                return  # Don't process until settings are configured
        else:
            # Existing user with preferences - check if this is a new thread that needs a settings button
            self._post_settings_button_if_new_thread(message, client, user_prefs)
        
        # Call the message handler if set
        if self.message_handler:
            self.message_handler(message, self)

    def _post_settings_button_if_new_thread(self, message: Message, client, user_prefs: dict):
        """Post a settings button at the start of a new thread"""
        try:
            # Check if this is the start of a new thread
            # For channels: thread_id != ts means it's a reply in a thread
            # For DMs: we want to check if there's any history
            
            is_dm = message.channel_id.startswith('D')
            self.log_debug(f"Checking for new thread: is_dm={is_dm}, channel={message.channel_id}, thread={message.thread_id}")
            
            # Get thread history to check if this is a new conversation
            if is_dm:
                # In DMs, every message is technically a new "thread" (unique timestamp)
                # Check if this specific thread already has messages
                history = client.conversations_replies(
                    channel=message.channel_id,
                    ts=message.thread_id
                )
                self.log_debug(f"DM thread history check: found {len(history.get('messages', []))} messages in thread {message.thread_id}")
                
                # If there's only 1 message (the current one), it's a new thread
                is_new_thread = len(history.get('messages', [])) <= 1
            else:
                # For channels, check if this is creating a new thread
                # When thread_id == ts, it's a new thread (first message)
                is_new_thread = (message.thread_id == message.metadata.get('ts'))
            
            self.log_info(f"New thread check result: is_new_thread={is_new_thread}")
            
            if is_new_thread:
                # Check if this is a new user who hasn't completed settings
                is_new_user = not user_prefs.get('settings_completed', False)
                
                if is_new_user:
                    # New user - need to store message for later processing
                    # Prepare button value with size check
                    full_context = {
                        "original_message": message.text,
                        "channel_id": message.channel_id,
                        "thread_id": message.thread_id,
                        "attachments": message.attachments
                    }
                    
                    # Check if button value would exceed Slack's 2000 char limit (with buffer)
                    full_value = json.dumps(full_context)
                    if len(full_value) > 1900:  # Leave 100 char buffer
                        # Fallback: only store reference data
                        button_value = json.dumps({
                            "channel_id": message.channel_id,
                            "thread_id": message.thread_id,
                            "ts": message.metadata.get('ts'),  # Add timestamp to fetch message later
                            "has_attachments": bool(message.attachments),
                            "attachment_count": len(message.attachments),
                            "truncated": True
                        })
                        self.log_info(f"Button value too large ({len(full_value)} chars), using truncated version")
                    else:
                        button_value = full_value
                    
                    # Full welcome message for new users
                    blocks = [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "*Welcome to the AI Assistant!* :wave:\n\nI need you to configure your preferences before we can start. You can accept the defaults or customize them."
                            }
                        },
                        {
                            "type": "actions",
                            "elements": [
                                {
                                    "type": "button",
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Configure Settings"
                                    },
                                    "style": "primary",
                                    "action_id": "open_welcome_settings",
                                    "value": button_value
                                }
                            ]
                        }
                    ]
                    
                    # No need to warn user - we handle truncation transparently
                else:
                    # Existing user - message is already being processed, just provide settings access
                    # Only store minimal context needed for settings modal
                    button_value = json.dumps({
                        "channel_id": message.channel_id,
                        "thread_id": message.thread_id
                    })
                    
                    # Compact settings button for existing users
                    blocks = [
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": "⚙️ *Quick Settings Access*"
                            }
                        },
                        {
                            "type": "actions",
                            "elements": [
                                {
                                    "type": "button",
                                    "text": {
                                        "type": "plain_text",
                                        "text": "Settings"
                                    },
                                    "style": "primary",
                                    "action_id": "open_welcome_settings",
                                    "value": button_value
                                }
                            ]
                        }
                    ]
                
                # Post the settings button as the first message in the thread
                client.chat_postMessage(
                    channel=message.channel_id,
                    thread_ts=message.thread_id,  # Always use thread_ts to post in the thread
                    text="Settings available",
                    blocks=blocks
                )
                
        except Exception as e:
            self.log_debug(f"Could not post settings button: {e}")
            # Don't block message processing if button posting fails
