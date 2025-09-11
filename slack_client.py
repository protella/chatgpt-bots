"""
Slack Bot Client Implementation
All Slack-specific functionality
"""
import re
import json
import time
import base64
from typing import Optional, List, Dict, Any
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError
from base_client import BaseClient, Message, Response
from config import config
from markdown_converter import MarkdownConverter
from database import DatabaseManager
from settings_modal import SettingsModal


class SlackBot(BaseClient):
    """Slack-specific bot implementation"""
    
    # Slack message limit (leaving buffer for formatting)
    MAX_MESSAGE_LENGTH = 3900
    
    def __init__(self, message_handler=None):
        super().__init__("SlackBot")
        self.app = App(token=config.slack_bot_token)
        self.handler = None
        self.message_handler = message_handler  # Callback for processing messages
        self.markdown_converter = MarkdownConverter(platform="slack")
        self.user_cache = {}  # Cache user info to avoid repeated API calls
        
        # Initialize database manager
        self.db = DatabaseManager(platform="slack")
        
        # Initialize settings modal handler
        self.settings_modal = SettingsModal(self.db)
        
        # Register Slack event handlers
        self._register_handlers()
    
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
                except:
                    pass
        
        # Register modal submission handlers
        @self.app.view("settings_modal")
        @self.app.view("welcome_settings_modal")
        def handle_settings_submission(ack, body, view, client):
            """Handle settings modal submission"""
            ack()
            
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
                self.log_info(f"New user setup - forcing save to global settings")
            else:
                selected_scope = metadata.get('scope', 'thread' if in_thread else 'global')  # Get selected scope
            
            # Debug logging
            self.log_debug(f"Modal submission metadata: {metadata}")
            self.log_debug(f"Selected scope for save: {selected_scope} (is_new_user: {is_new_user})")
            if pending_message:
                self.log_info(f"Found pending message in metadata: {pending_message}")
            
            # Extract form values
            form_values = self.settings_modal.extract_form_values(view['state'])
            
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
                
                # The thread state will be loaded from DB on next message
                self.log_debug(f"Thread config saved to DB for {thread_id} - will be loaded on next message")
                
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
                    # Get the channel from the original command or use user's DM
                    channel = body.get('view', {}).get('private_metadata', user_id)
                    
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
                        except:
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
                    except:
                        pass  # Best effort
                    del self._welcome_messages[user_id]
                
                # Remove from welcomed users set once they've configured settings
                # This ensures they won't get welcome messages anymore
                if hasattr(self, '_welcomed_users') and user_id in self._welcomed_users:
                    self._welcomed_users.remove(user_id)
                    
                # Process pending message if this was from welcome flow (only for global settings)
                if save_location == "global" and pending_message and pending_message.get('original_message'):
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
                        except:
                            pass
                            
            else:
                self.log_error(f"Failed to save settings for user {user_id}")
                try:
                    client.chat_postMessage(
                        channel=user_id,
                        text="❌ Sorry, there was an error saving your settings. Please try again."
                    )
                except:
                    pass
        
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
            except:
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
            except:
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
            except:
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
            ack()
            
            user_id = body['user']['id']
            trigger_id = body['trigger_id']
            
            # Get user preferences
            user_prefs = self.db.get_user_preferences(user_id)
            if not user_prefs:
                user_data = self.db.get_or_create_user(user_id)
                email = user_data.get('email') if user_data else None
                user_prefs = self.db.create_default_user_preferences(user_id, email)
            
            # Open the settings modal for global settings
            try:
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
                    
            except SlackApiError as e:
                self.log_error(f"Error opening global settings modal from DM: {e}")
        
        # Handler for welcome settings button
        @self.app.action("open_welcome_settings")
        def handle_open_welcome_settings(ack, body, client):
            """Handle button click to open welcome settings modal"""
            ack()
            
            user_id = body['user']['id']
            trigger_id = body['trigger_id']
            
            # Extract the original message details from the button value
            button_value = body['actions'][0].get('value', '{}')
            try:
                original_context = json.loads(button_value)
            except:
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
                    self.log_error(f"Error fetching truncated message: {e}")
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
                
                # Add the original message context to the modal's private_metadata
                existing_metadata = json.loads(modal.get('private_metadata', '{}'))
                existing_metadata['pending_message'] = original_context
                modal['private_metadata'] = json.dumps(existing_metadata)
                
                response = client.views_open(
                    trigger_id=trigger_id,
                    view=modal
                )
                
                if response.get('ok'):
                    self.log_info(f"Welcome modal opened via button for user {user_id}")
                    
                    # Keep the button message for future access
                    # (removed deletion to allow persistent settings access)
                        
            except SlackApiError as e:
                self.log_error(f"Error opening welcome modal via button: {e}")
        
        # Register message shortcut for thread-specific settings
        @self.app.shortcut("configure_thread_settings_dev")  # Dev callback ID
        @self.app.shortcut("configure_thread_settings")  # Prod callback ID (when configured)
        def handle_thread_settings_shortcut(ack, shortcut, client):
            """Handle the thread settings message shortcut"""
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
                self.log_error(f"Error opening thread settings modal: {e}")
    
    def get_username(self, user_id: str, client) -> str:
        """Get username from user ID, with caching"""
        # Check memory cache first
        if user_id in self.user_cache and 'username' in self.user_cache[user_id]:
            return self.user_cache[user_id]['username']
        
        # Check database for user
        user_data = self.db.get_or_create_user(user_id)
        if user_data.get('username'):
            # Load from DB to memory cache
            tz_info = self.db.get_user_timezone(user_id)
            if tz_info:
                self.user_cache[user_id] = {
                    'username': user_data['username'],
                    'timezone': tz_info[0],
                    'tz_label': tz_info[1],
                    'tz_offset': tz_info[2] or 0
                }
                return user_data['username']
        
        try:
            # Fetch user info from Slack API
            result = client.users_info(user=user_id)
            if result["ok"]:
                user_info = result["user"]
                # Get both display name and real name
                display_name = user_info.get("profile", {}).get("display_name")
                real_name = user_info.get("profile", {}).get("real_name")
                email = user_info.get("profile", {}).get("email")
                # Prefer display name, fall back to real name, then just the ID
                username = display_name or real_name or user_info.get("name") or user_id
                
                # Debug log for email
                self.log_debug(f"Fetched user info for {user_id}: email={email}, real_name={real_name}")
                
                # Cache both username and timezone info in memory
                self.user_cache[user_id] = {
                    'username': username,
                    'real_name': real_name,
                    'email': email,
                    'timezone': user_info.get('tz', 'UTC'),
                    'tz_label': user_info.get('tz_label', 'UTC'),
                    'tz_offset': user_info.get('tz_offset', 0)
                }
                
                # Save to database with all user info
                self.db.get_or_create_user(user_id, username)
                self.db.save_user_info(
                    user_id,
                    username=username,
                    real_name=real_name,
                    email=email,
                    timezone=user_info.get('tz', 'UTC'),
                    tz_label=user_info.get('tz_label', 'UTC'),
                    tz_offset=user_info.get('tz_offset', 0)
                )
                
                self.log_debug(f"Cached timezone info for {username}: tz={user_info.get('tz')}, tz_label={user_info.get('tz_label')}")
                return username
        except Exception as e:
            self.log_debug(f"Could not fetch username for {user_id}: {e}")
        
        return user_id  # Fallback to user ID if fetch fails
    
    def get_user_timezone(self, user_id: str, client) -> str:
        """Get user's timezone, fetching if necessary"""
        # Check memory cache first
        if user_id in self.user_cache and 'timezone' in self.user_cache[user_id]:
            return self.user_cache[user_id]['timezone']
        
        # Check database
        tz_info = self.db.get_user_timezone(user_id)
        if tz_info:
            # Load to memory cache
            if user_id not in self.user_cache:
                self.user_cache[user_id] = {}
            self.user_cache[user_id]['timezone'] = tz_info[0]
            self.user_cache[user_id]['tz_label'] = tz_info[1]
            self.user_cache[user_id]['tz_offset'] = tz_info[2] or 0
            return tz_info[0]
        
        # Fetch user info (which will also cache it)
        self.get_username(user_id, client)
        
        # Return timezone from cache or default to UTC
        if user_id in self.user_cache and 'timezone' in self.user_cache[user_id]:
            return self.user_cache[user_id]['timezone']
        
        return 'UTC'  # Default fallback
    
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
                        truncated = True
                        self.log_info(f"Welcome button value too large ({len(full_value)} chars), using truncated version")
                    else:
                        button_value = full_value
                        truncated = False
                    
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
                        truncated = True
                        self.log_info(f"Button value too large ({len(full_value)} chars), using truncated version")
                    else:
                        button_value = full_value
                        truncated = False
                    
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
    
    def _clean_mentions(self, text: str) -> str:
        """Remove Slack user mentions from text"""
        return re.sub(r'<@[A-Z0-9]+>', '', text).strip()
    
    def start(self):
        """Start the Slack bot"""
        self.handler = SocketModeHandler(self.app, config.slack_app_token)
        self.log_info("Starting Slack bot in socket mode...")
        self.handler.start()
    
    def stop(self):
        """Stop the Slack bot"""
        if self.handler:
            self.log_info("Stopping Slack bot...")
            self.handler.close()
    def send_message(self, channel_id: str, thread_id: str, text: str) -> bool:
        """Send a text message to Slack, splitting if needed"""
        try:
            # Format text for Slack
            formatted_text = self.format_text(text)
            
            # Check if we need to split the message
            if len(formatted_text) <= self.MAX_MESSAGE_LENGTH:
                # Single message
                self.app.client.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_id,
                    text=formatted_text
                )
            else:
                # Split into multiple messages
                chunks = self._split_message(formatted_text)
                for i, chunk in enumerate(chunks, 1):
                    # Add pagination indicator
                    paginated_chunk = f"*Part {i}/{len(chunks)}*\n\n{chunk}"
                    self.app.client.chat_postMessage(
                        channel=channel_id,
                        thread_ts=thread_id,
                        text=paginated_chunk
                    )
            return True
        except SlackApiError as e:
            self.log_error(f"Error sending message: {e}")
            return False
    
    def _split_message(self, text: str) -> List[str]:
        """Split a long message into chunks that fit within Slack's limit"""
        # Account for pagination indicator overhead (~20 chars)
        chunk_size = self.MAX_MESSAGE_LENGTH - 50
        chunks = []
        
        # Try to split on paragraph boundaries first
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for para in paragraphs:
            # If a single paragraph is too long, split it by sentences
            if len(para) > chunk_size:
                sentences = para.replace('. ', '.\n').split('\n')
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 2 <= chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "
            elif len(current_chunk) + len(para) + 2 <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def send_message_get_ts(self, channel_id: str, thread_id: str, text: str) -> Dict:
        """Send a message and return the response including timestamp"""
        try:
            # Format text for Slack
            formatted_text = self.format_text(text)
            
            # Ensure it fits in one message for streaming continuation
            if len(formatted_text) > self.MAX_MESSAGE_LENGTH:
                formatted_text = formatted_text[:self.MAX_MESSAGE_LENGTH - 50] + "\n\n*...truncated*"
            
            result = self.app.client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_id,
                text=formatted_text
            )
            
            return {"success": True, "ts": result["ts"]}
        except SlackApiError as e:
            self.log_error(f"Error sending message: {e}")
            return {"success": False, "error": str(e)}
    
    def send_image(self, channel_id: str, thread_id: str, image_data: bytes, filename: str, caption: str = "") -> Optional[str]:
        """Send an image to Slack and return the file URL"""
        try:
            # Use files_upload_v2 for image upload
            result = self.app.client.files_upload_v2(
                channel=channel_id,  # Changed from channels to channel (singular)
                thread_ts=thread_id,
                file=image_data,
                filename=filename,
                initial_comment=caption
            )
            
            # Extract the file URL from the response
            if result and "files" in result and len(result["files"]) > 0:
                file_info = result["files"][0]
                file_url = file_info.get("url_private", file_info.get("permalink"))
                self.log_info(f"Image uploaded: {filename} - URL: {file_url}")
                return file_url
            else:
                self.log_warning("Image uploaded but no URL found in response")
                return None
                
        except SlackApiError as e:
            self.log_error(f"Error uploading image: {e}")
            return None
    
    def send_thinking_indicator(self, channel_id: str, thread_id: str) -> Optional[str]:
        """Send thinking indicator to Slack"""
        try:
            result = self.app.client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_id,
                text=f"{config.thinking_emoji} Thinking..."
            )
            return result.get("ts")  # Return message timestamp for deletion
        except SlackApiError as e:
            self.log_error(f"Error sending thinking indicator: {e}")
            return None
    
    def delete_message(self, channel_id: str, message_id: str) -> bool:
        """Delete a message from Slack"""
        try:
            self.app.client.chat_delete(
                channel=channel_id,
                ts=message_id
            )
            return True
        except SlackApiError as e:
            self.log_debug(f"Could not delete message: {e}")
            return False
    def update_message(self, channel_id: str, message_id: str, text: str) -> bool:
        """Update a message in Slack"""
        try:
            self.app.client.chat_update(
                channel=channel_id,
                ts=message_id,
                text=text,
                mrkdwn=True  # Enable markdown parsing for italics/bold
            )
            return True
        except SlackApiError as e:
            self.log_error(f"Could not update message: {e}")
            return False
    
    def get_thread_history(self, channel_id: str, thread_id: str, limit: int = None) -> List[Message]:
        """Get COMPLETE thread history from Slack - fetches ALL messages by default"""
        messages = []
        
        try:
            # Fetch ALL messages using pagination
            cursor = None
            total_fetched = 0
            
            while True:
                # Slack's max per request is 1000
                per_request_limit = 1000
                if limit and limit - total_fetched < 1000:
                    per_request_limit = limit - total_fetched
                
                kwargs = {
                    "channel": channel_id,
                    "ts": thread_id,
                    "limit": per_request_limit
                }
                if cursor:
                    kwargs["cursor"] = cursor
                
                result = self.app.client.conversations_replies(**kwargs)
                slack_messages = result.get("messages", [])
                
                if not slack_messages:
                    break
                    
                # Process messages from this batch
                for msg in slack_messages:
                    # Skip loading indicators and system messages
                    text = msg.get("text", "")
                    if "Thinking" in text:
                        continue
                    # Skip busy/processing messages
                    if ":warning:" in text and "currently processing" in text:
                        continue
                    # Skip settings button messages
                    if text == "Settings available":
                        continue
                    
                    # Determine role
                    is_bot = bool(msg.get("bot_id"))
                    
                    # Clean text
                    text = msg.get("text", "")
                    if not is_bot:
                        text = self._clean_mentions(text)
                    
                    # Check for files
                    attachments = []
                    files = msg.get("files", [])
                    for file in files:
                        # Determine file type based on mimetype
                        mimetype = file.get("mimetype", "")
                        file_type = "image" if mimetype.startswith("image/") else "file"
                        
                        attachments.append({
                            "type": file_type,
                            "name": file.get("name"),
                            "mimetype": mimetype,
                            "url": file.get("url_private", file.get("permalink"))
                        })
                    
                    messages.append(Message(
                        text=text,
                        user_id=msg.get("user", "bot" if is_bot else "unknown"),
                        channel_id=channel_id,
                        thread_id=thread_id,
                        attachments=attachments,
                        metadata={
                            "ts": msg.get("ts"),
                            "is_bot": is_bot
                        }
                    ))
                
                total_fetched += len(slack_messages)
                
                # Check if we've hit our limit
                if limit and total_fetched >= limit:
                    break
                
                # Check for pagination
                response_metadata = result.get("response_metadata", {})
                next_cursor = response_metadata.get("next_cursor")
                
                if not next_cursor:
                    # No more messages
                    break
                    
                cursor = next_cursor
                # Continue to next iteration
            
            self.log_info(f"Fetched {len(messages)} messages from thread {thread_id}")
            return messages
            
        except SlackApiError as e:
            self.log_error(f"Error getting thread history: {e}")
            return []
    
    def extract_file_id_from_url(self, file_url: str) -> Optional[str]:
        """Extract file ID from a Slack file URL
        
        Args:
            file_url: The Slack file URL
            
        Returns:
            File ID if found, None otherwise
        """
        import re
        
        # Try to extract file ID from the URL
        patterns = [
            r'/files-pri/[^/]+-([^/]+)/',  # files-pri format
            r'/files/[^/]+/([^/]+)/',       # permalink format
        ]
        
        for pattern in patterns:
            match = re.search(pattern, file_url)
            if match:
                file_id = match.group(1)
                self.log_debug(f"Extracted file ID from URL: {file_id}")
                return file_id
        
        return None
    def download_file(self, file_url: str, file_id: Optional[str] = None) -> Optional[bytes]:
        """Download a file from Slack
        
        Args:
            file_url: The Slack file URL (can be url_private or permalink)
            file_id: Optional file ID (will be extracted from URL if not provided)
        """
        try:
            import requests
            
            # If file_id not provided, try to extract from URL
            if not file_id:
                # URL format: https://files.slack.com/files-pri/[TEAM]-[FILE_ID]/filename
                # or https://[team].slack.com/files/[USER]/[FILE_ID]/filename
                import re
                
                # Try to extract file ID from the URL
                patterns = [
                    r'/files-pri/[^/]+-([^/]+)/',  # files-pri format
                    r'/files/[^/]+/([^/]+)/',       # permalink format
                ]
                
                for pattern in patterns:
                    match = re.search(pattern, file_url)
                    if match:
                        file_id = match.group(1)
                        self.log_debug(f"Extracted file ID from URL: {file_id}")
                        break
                
                if not file_id:
                    # If we can't extract ID, try direct download with the URL
                    self.log_debug("Could not extract file ID, trying direct download")
                    headers = {"Authorization": f"Bearer {config.slack_bot_token}"}
                    response = requests.get(file_url, headers=headers)
                    
                    if response.status_code == 200:
                        return response.content
                    else:
                        self.log_error(f"Failed to download file directly: HTTP {response.status_code}")
                        return None
            
            # Get file info to get the private URL
            self.log_debug(f"Getting file info for file ID: {file_id}")
            file_info = self.app.client.files_info(file=file_id)
            
            # Check if file exists and is accessible
            if not file_info.get("ok"):
                self.log_error(f"Failed to get file info: {file_info.get('error', 'Unknown error')}")
                return None
            
            # Get the URL for downloading
            file_data = file_info.get("file", {})
            url_private = file_data.get("url_private") or file_data.get("url_private_download")
            
            if not url_private:
                self.log_error("No private URL found in file info")
                self.log_debug(f"File info keys: {file_data.keys()}")
                return None
            
            self.log_debug(f"Downloading from private URL: {url_private[:50]}...")
            
            # Download file using requests with auth header
            headers = {"Authorization": f"Bearer {config.slack_bot_token}"}
            response = requests.get(url_private, headers=headers)
            
            if response.status_code == 200:
                # Check if we got actual image data
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' in content_type:
                    self.log_error(f"Got HTML instead of image data from private URL")
                    self.log_debug(f"Response preview: {response.text[:200]}")
                    return None
                return response.content
            else:
                self.log_error(f"Failed to download file: HTTP {response.status_code}")
                return None
            
        except SlackApiError as e:
            self.log_error(f"Error getting file info: {e}")
            return None
        except Exception as e:
            self.log_error(f"Error downloading file: {e}")
            return None
    
    def format_text(self, text: str) -> str:
        """Format text for Slack using mrkdwn"""
        return self.markdown_converter.convert(text)
    
    def send_busy_message(self, channel_id: str, thread_id: str):
        """Send a busy message"""
        self.send_message(
            channel_id,
            thread_id,
            ":warning: `This thread is currently processing another request. Please wait a moment and try again.`"
        )
    
    def format_error_message(self, error: str) -> str:
        """Format error messages for Slack with emojis and code blocks"""
        import re
        
        # Extract error code if present
        error_code_match = re.search(r'Error code: (\d+)', error)
        error_code = error_code_match.group(1) if error_code_match else "Unknown"
        
        # Try to extract the actual error message
        if "{'error':" in error:
            # Parse OpenAI API error format
            try:
                import json
                error_dict_str = error[error.find("{'error':"):].replace("'", '"')
                error_dict = json.loads(error_dict_str)
                error_message = error_dict.get('error', {}).get('message', error)
                error_type = error_dict.get('error', {}).get('type', 'unknown_error')
            except:
                # Fallback to simpler extraction
                if "'message':" in error:
                    msg_start = error.find("'message': '") + len("'message': '")
                    msg_end = error.find("',", msg_start)
                    if msg_end > msg_start:
                        error_message = error[msg_start:msg_end]
                    else:
                        error_message = error
                else:
                    error_message = error
                error_type = "api_error"
        else:
            error_message = error
            error_type = "general_error"
        
        # Format the error message for Slack
        formatted = f":warning: *Oops! Something went wrong*\n\n"
        formatted += f"*Error Code:* `{error_code}`\n"
        formatted += f"*Type:* `{error_type}`\n\n"
        formatted += f"*Details:*\n```{error_message}```\n\n"
        formatted += f":bulb: *What you can do:*\n"
        
        # Add helpful suggestions based on error type
        if "rate_limit" in error_type.lower():
            formatted += "• Wait a moment and try again\n"
            formatted += "• The API rate limit has been reached"
        elif "invalid_request" in error_type.lower():
            formatted += "• Try rephrasing your request\n"
            formatted += "• The request format may be invalid"
        elif "context_length" in error_message.lower():
            formatted += "• Start a new thread\n"
            formatted += "• The conversation has become too long"
        else:
            formatted += "• Try again in a moment\n"
            formatted += "• If the problem persists, contact support"
        
        return formatted
    
    def supports_streaming(self) -> bool:
        """Returns True if streaming is enabled for Slack"""
        return config.enable_streaming and config.slack_streaming
    
    def get_streaming_config(self) -> Dict:
        """Returns platform-specific streaming configuration"""
        return {
            "update_interval": config.streaming_update_interval,
            "min_interval": config.streaming_min_interval,
            "max_interval": config.streaming_max_interval,
            "buffer_size": config.streaming_buffer_size,
            "circuit_breaker_threshold": config.streaming_circuit_breaker_threshold,
            "circuit_breaker_cooldown": config.streaming_circuit_breaker_cooldown,
            "platform": "slack"
        }
    
    def update_message_streaming(self, channel_id: str, message_id: str, text: str) -> Dict:
        """Updates a message with rate limit awareness"""
        try:
            # For messages that already contain Slack mrkdwn (like enhanced prompts with _italics_),
            # skip the markdown conversion to avoid double-processing
            if text.startswith("✨") or text.startswith("*Enhanced Prompt:*") or text.startswith("Enhancing your prompt:"):
                # This is an enhanced prompt - it already has proper Slack formatting
                formatted_text = text
            else:
                # Format text for Slack using markdown conversion
                formatted_text = self.format_text(text)
            
            # Truncate if too long during streaming
            if len(formatted_text) > self.MAX_MESSAGE_LENGTH:
                formatted_text = formatted_text[:self.MAX_MESSAGE_LENGTH - 100] + "\n\n*...continuing...*"
            
            # Call Slack API's chat_update method
            result = self.app.client.chat_update(
                channel=channel_id,
                ts=message_id,
                text=formatted_text,
                mrkdwn=True  # Enable markdown parsing for italics/bold
            )
            
            # Return success status
            return {
                "success": True,
                "rate_limited": False,
                "retry_after": None,
                "result": result
            }
            
        except SlackApiError as e:
            # Handle 429 rate limit responses
            if e.response.status_code == 429:
                # Extract retry-after header
                retry_after = None
                if hasattr(e.response, 'headers') and 'Retry-After' in e.response.headers:
                    try:
                        retry_after = int(e.response.headers['Retry-After'])
                    except (ValueError, KeyError):
                        retry_after = None
                
                self.log_warning(f"🚨🚨🚨 HIT RATE LIMIT 429 🚨🚨🚨")
                
                return {
                    "success": False,
                    "rate_limited": True,
                    "retry_after": retry_after,
                    "error": str(e)
                }
            else:
                # Handle other API errors
                self.log_error(f"Error updating message in streaming: {e}")
                return {
                    "success": False,
                    "rate_limited": False,
                    "retry_after": None,
                    "error": str(e)
                }
        except Exception as e:
            # Handle unexpected errors
            self.log_error(f"Unexpected error updating message in streaming: {e}")
            return {
                "success": False,
                "rate_limited": False,
                "retry_after": None,
                "error": str(e)
            }

    def handle_response(self, channel_id: str, thread_id: str, response: Response):
        """Handle a Response object and send to Slack"""
        if response.type == "text":
            self.send_message(channel_id, thread_id, response.content)
        elif response.type == "image":
            # response.content should be ImageData
            image_data = response.content
            file_url = self.send_image(
                channel_id,
                thread_id,
                image_data.to_bytes(),
                f"generated_image.{image_data.format}",
                ""  # No caption - prompt already displayed via streaming
            )
            
            # Store the URL in the image data for tracking
            if file_url:
                image_data.slack_url = file_url
                
        elif response.type == "error":
            formatted_error = self.format_error_message(response.content)
            self.send_message(channel_id, thread_id, formatted_error)