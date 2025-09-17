from __future__ import annotations

from typing import Dict, List, Optional

from slack_bolt.adapter.socket_mode import SocketModeHandler
from slack_sdk.errors import SlackApiError

from base_client import Message, Response
from config import config


class SlackMessagingMixin:
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
            
            # Safety check - this should never happen for continuation messages
            # but if somehow the text is too long, truncate it
            if len(formatted_text) > self.MAX_MESSAGE_LENGTH:
                formatted_text = formatted_text[:self.MAX_MESSAGE_LENGTH - 80] + "\n\n*[Message exceeded Slack limit]*"
            
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

    def send_busy_message(self, channel_id: str, thread_id: str):
        """Send a busy message"""
        self.send_message(
            channel_id,
            thread_id,
            ":warning: `This thread is currently processing another request. Please wait a moment and try again.`"
        )

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
            
            # More aggressive truncation for streaming to avoid msg_too_long errors
            # Account for Slack's markdown expansion and special characters
            safe_length = self.MAX_MESSAGE_LENGTH - 200  # More buffer for safety
            if len(formatted_text) > safe_length:
                # Try to truncate at a reasonable boundary (code block or paragraph)
                truncated = formatted_text[:safe_length]
                
                # If we're in the middle of a code block, close it
                if truncated.count('```') % 2 == 1:
                    truncated += '\n```'
                
                formatted_text = truncated + "\n\n*...continued in next message...*"
            
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
            # Handle msg_too_long error specifically
            if e.response.get('error') == 'msg_too_long':
                self.log_warning("Message too long for Slack, truncating more aggressively")
                # Try with much shorter message
                very_short = formatted_text[:2000] + "\n\n*...continued in next message...*"
                if very_short.count('```') % 2 == 1:
                    very_short += '\n```'
                
                try:
                    result = self.app.client.chat_update(
                        channel=channel_id,
                        ts=message_id,
                        text=very_short,
                        mrkdwn=True
                    )
                    return {
                        "success": True,
                        "rate_limited": False,
                        "retry_after": None,
                        "result": result
                    }
                except Exception:
                    # If even the short version fails, just acknowledge the error
                    self.log_error("Even truncated message failed to send")
                    raise
            
            # Handle 429 rate limit responses
            elif e.response.status_code == 429:
                # Extract retry-after header
                retry_after = None
                if hasattr(e.response, 'headers') and 'Retry-After' in e.response.headers:
                    try:
                        retry_after = int(e.response.headers['Retry-After'])
                    except (ValueError, KeyError):
                        retry_after = None
                
                self.log_warning("🚨🚨🚨 HIT RATE LIMIT 429 🚨🚨🚨")
                
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
