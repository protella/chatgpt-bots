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
    
    def __init__(self, message_handler=None):
        super().__init__("SlackBot")
        self.app = App(token=config.slack_bot_token)
        self.handler = None
        self.message_handler = message_handler  # Callback for processing messages
        self.markdown_converter = MarkdownConverter(platform="slack")
        
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
            if file.get("mimetype", "").startswith("image/"):
                attachments.append({
                    "type": "image",
                    "url": file.get("url_private"),
                    "id": file.get("id"),
                    "name": file.get("name"),
                    "mimetype": file.get("mimetype")
                })
        
        # Create universal message
        message = Message(
            text=text,
            user_id=event.get("user"),
            channel_id=event.get("channel"),
            thread_id=event.get("thread_ts") or event.get("ts"),
            attachments=attachments,
            metadata={
                "ts": event.get("ts"),
                "slack_client": client
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
        """Send a text message to Slack"""
        try:
            # Format text for Slack
            formatted_text = self.format_text(text)
            
            self.app.client.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_id,
                text=formatted_text
            )
            return True
        except SlackApiError as e:
            self.log_error(f"Error sending message: {e}")
            return False
    
    def send_image(self, channel_id: str, thread_id: str, image_data: bytes, filename: str, caption: str = "") -> bool:
        """Send an image to Slack"""
        try:
            # Use files_upload_v2 for image upload
            result = self.app.client.files_upload_v2(
                channel=channel_id,  # Changed from channels to channel (singular)
                thread_ts=thread_id,
                file=image_data,
                filename=filename,
                initial_comment=caption
            )
            self.log_info(f"Image uploaded: {filename}")
            return True
        except SlackApiError as e:
            self.log_error(f"Error uploading image: {e}")
            return False
    
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
                    attachments.append({
                        "type": "file",
                        "name": file.get("name"),
                        "mimetype": file.get("mimetype")
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
    
    def download_file(self, file_url: str, file_id: str) -> Optional[bytes]:
        """Download a file from Slack"""
        try:
            # Get file info
            file_info = self.app.client.files_info(file=file_id)
            url = file_info["file"]["url_private"]
            
            # Download file
            response = self.app.client.api_call(
                api_method="GET",
                url=url,
                headers={"Authorization": f"Bearer {config.slack_bot_token}"}
            )
            
            return response.data
            
        except SlackApiError as e:
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
    
    def handle_response(self, channel_id: str, thread_id: str, response: Response):
        """Handle a Response object and send to Slack"""
        if response.type == "text":
            self.send_message(channel_id, thread_id, response.content)
        elif response.type == "image":
            # response.content should be ImageData
            image_data = response.content
            self.send_image(
                channel_id,
                thread_id,
                image_data.to_bytes(),
                f"generated_image.{image_data.format}",
                f"Generated image: {image_data.prompt}"
            )
        elif response.type == "error":
            formatted_error = self.format_error_message(response.content)
            self.send_message(channel_id, thread_id, formatted_error)