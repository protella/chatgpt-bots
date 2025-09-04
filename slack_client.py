"""
Slack Bot Client Implementation
All Slack-specific functionality
"""
import re
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
            self._handle_slack_message(event, client)
        
        @self.app.event("message")
        def handle_message(event, say, client):
            # Only process DMs and non-bot messages
            if event.get("channel_type") == "im" and not event.get("bot_id"):
                self._handle_slack_message(event, client)
        
        # Register slash command handler
        @self.app.command(config.settings_slash_command)
        def handle_settings_command(ack, body, client):
            """Handle the settings slash command"""
            ack()  # Acknowledge command receipt immediately
            
            user_id = body.get('user_id')
            trigger_id = body.get('trigger_id')
            
            self.log_info(f"Settings command invoked by user {user_id}")
            
            # Get current settings or create defaults
            current_settings = self.db.get_user_preferences(user_id)
            is_new_user = current_settings is None
            
            if is_new_user:
                # Get user's email for preferences
                user_data = self.db.get_or_create_user(user_id)
                email = user_data.get('email') if user_data else None
                current_settings = self.db.create_default_user_preferences(user_id, email)
            
            # Build and open modal
            try:
                modal = self.settings_modal.build_settings_modal(
                    user_id=user_id,
                    trigger_id=trigger_id,
                    current_settings=current_settings,
                    is_new_user=is_new_user
                )
                
                # Open the modal
                response = client.views_open(
                    trigger_id=trigger_id,
                    view=modal
                )
                
                if response.get('ok'):
                    self.log_info(f"Settings modal opened for user {user_id}")
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
            
            # Extract form values
            form_values = self.settings_modal.extract_form_values(view['state'])
            
            # Validate settings
            validated_settings = self.settings_modal.validate_settings(form_values)
            
            # Check for temperature/top_p warning
            warning_message = ""
            if validated_settings.get('model') not in ['gpt-5', 'gpt-5-mini', 'gpt-5-nano']:
                temp_changed = validated_settings.get('temperature', config.default_temperature) != config.default_temperature
                top_p_changed = validated_settings.get('top_p', config.default_top_p) != config.default_top_p
                
                if temp_changed and top_p_changed:
                    warning_message = "\n⚠️ Note: You've changed both Temperature and Top P. OpenAI recommends using only one for best results."
            
            # Mark settings as completed
            validated_settings['settings_completed'] = True
            
            # Update database
            success = self.db.update_user_preferences(user_id, validated_settings)
            
            if success:
                self.log_info(f"Settings saved for user {user_id}: {validated_settings}")
                
                # Send confirmation message
                try:
                    # Get the channel from the original command or use user's DM
                    channel = body.get('view', {}).get('private_metadata', user_id)
                    
                    client.chat_postMessage(
                        channel=user_id,  # Send to user's DM
                        text=f"✅ Your settings have been saved successfully!{warning_message}"
                    )
                except SlackApiError as e:
                    self.log_error(f"Error sending confirmation: {e}")
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
            try:
                import json
                private_metadata = body.get('view', {}).get('private_metadata')
                if private_metadata:
                    stored_settings = json.loads(private_metadata)
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
            merged_settings = stored_settings.copy()
            merged_settings.update(current_values)
            
            # Build updated modal with new model selection
            is_new_user = body['view']['callback_id'] == 'welcome_settings_modal'
            updated_modal = self.settings_modal.build_settings_modal(
                user_id=user_id,
                trigger_id=None,  # Not needed for update
                current_settings=merged_settings,
                is_new_user=is_new_user
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
        
        # Register action handlers for other interactive components (just acknowledge)
        @self.app.action("features")
        @self.app.action("reasoning_level")
        @self.app.action("verbosity")
        @self.app.action("input_fidelity")
        @self.app.action("vision_detail")
        def handle_modal_actions(ack):
            """Acknowledge modal actions that don't need processing"""
            ack()  # Just acknowledge - values are captured on submission
    
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
            # New user detected - check if we have a trigger_id for modal
            trigger_id = event.get('trigger_id')
            
            if trigger_id:
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
                            channel=channel_id,
                            thread_ts=thread_ts,
                            text="👋 Welcome! I've opened your settings panel. Please configure your preferences and I'll be ready to help!"
                        )
                        return  # Don't process the message until settings are saved
                    
                except SlackApiError as e:
                    self.log_error(f"Error opening welcome modal for new user: {e}")
                    # Continue with processing using defaults
            else:
                # No trigger_id available, send instructions
                try:
                    client.chat_postMessage(
                        channel=channel_id,
                        thread_ts=thread_ts, 
                        text=f"👋 Welcome! Please configure your settings by typing `{config.settings_slash_command}` to get started."
                    )
                    return  # Don't process until settings are configured
                except SlackApiError as e:
                    self.log_error(f"Error sending welcome message: {e}")
        
        # Call the message handler if set
        if self.message_handler:
            self.message_handler(message, self)
    
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