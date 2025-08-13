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
    
    def get_username(self, user_id: str, client) -> str:
        """Get username from user ID, with caching"""
        if user_id in self.user_cache and 'username' in self.user_cache[user_id]:
            return self.user_cache[user_id]['username']
        
        try:
            # Fetch user info from Slack API
            result = client.users_info(user=user_id)
            if result["ok"]:
                user_info = result["user"]
                # Prefer display name, fall back to real name, then just the ID
                username = (user_info.get("profile", {}).get("display_name") or 
                          user_info.get("profile", {}).get("real_name") or 
                          user_info.get("name") or 
                          user_id)
                
                # Cache both username and timezone info
                self.user_cache[user_id] = {
                    'username': username,
                    'timezone': user_info.get('tz', 'UTC'),
                    'tz_label': user_info.get('tz_label', 'UTC'),
                    'tz_offset': user_info.get('tz_offset', 0)
                }
                return username
        except Exception as e:
            self.log_debug(f"Could not fetch username for {user_id}: {e}")
        
        return user_id  # Fallback to user ID if fetch fails
    
    def get_user_timezone(self, user_id: str, client) -> str:
        """Get user's timezone, fetching if necessary"""
        # Check cache first
        if user_id in self.user_cache and 'timezone' in self.user_cache[user_id]:
            return self.user_cache[user_id]['timezone']
        
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
                "user_timezone": user_timezone  # Add timezone to metadata
            }
        )
        
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
                text=text
            )
            return True
        except SlackApiError as e:
            self.log_error(f"Could not update message: {e}")
            return False
    
    def get_thread_history(self, channel_id: str, thread_id: str, limit: int = 50) -> List[Message]:
        """Get thread history from Slack"""
        messages = []
        
        try:
            result = self.app.client.conversations_replies(
                channel=channel_id,
                ts=thread_id,
                limit=limit
            )
            
            slack_messages = result.get("messages", [])
            
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
            
            return messages
            
        except SlackApiError as e:
            self.log_error(f"Error getting thread history: {e}")
            return []
    
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
            file_info = self.app.client.files_info(file=file_id)
            url_private = file_info["file"]["url_private"]
            
            # Download file using requests with auth header
            headers = {"Authorization": f"Bearer {config.slack_bot_token}"}
            response = requests.get(url_private, headers=headers)
            
            if response.status_code == 200:
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
            # Format text for Slack using markdown conversion
            formatted_text = self.format_text(text)
            
            # Truncate if too long during streaming
            if len(formatted_text) > self.MAX_MESSAGE_LENGTH:
                formatted_text = formatted_text[:self.MAX_MESSAGE_LENGTH - 100] + "\n\n*...continuing...*"
            
            # Call Slack API's chat_update method
            result = self.app.client.chat_update(
                channel=channel_id,
                ts=message_id,
                text=formatted_text
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
                
                self.log_debug(f"Rate limited updating message in channel {channel_id}. Retry after: {retry_after}")
                
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
                f"Generated image: {image_data.prompt}"
            )
            
            # Store the URL in the image data for tracking
            if file_url:
                image_data.slack_url = file_url
                
        elif response.type == "error":
            formatted_error = self.format_error_message(response.content)
            self.send_message(channel_id, thread_id, formatted_error)