"""
Shared Message Processor
Client-agnostic message processing logic
"""
import base64
import re
import time
from typing import Dict, Any, List, Optional, Tuple
from base_client import BaseClient, Message, Response
from thread_manager import ThreadStateManager
from openai_client import OpenAIClient, ImageData
from config import config
from logger import LoggerMixin
from prompts import SLACK_SYSTEM_PROMPT, DISCORD_SYSTEM_PROMPT, CLI_SYSTEM_PROMPT, IMAGE_ANALYSIS_PROMPT
from streaming import StreamingBuffer, RateLimitManager


class MessageProcessor(LoggerMixin):
    """Handles message processing logic independent of chat platform"""
    
    def __init__(self):
        self.thread_manager = ThreadStateManager()
        self.openai_client = OpenAIClient()
        self.log_info("MessageProcessor initialized")
    
    def process_message(self, message: Message, client: BaseClient, thinking_id: Optional[str] = None) -> Optional[Response]:
        """
        Process a message and return a response
        
        Args:
            message: Universal message object
            client: The client that received the message
            thinking_id: ID of the thinking indicator message to update
        
        Returns:
            Response object or None if unable to process
        """
        thread_key = f"{message.channel_id}:{message.thread_id}"
        
        # Log request start with clear markers
        username = message.metadata.get("username", message.user_id) if message.metadata else message.user_id
        self.log_info("="*80)
        self.log_info(f"REQUEST START | Thread: {thread_key} | User: {username}")
        self.log_info(f"Message: {message.text[:100] if message.text else 'No text'}{'...' if message.text and len(message.text) > 100 else ''}")
        self.log_info("="*80)
        
        request_start_time = time.time()
        
        # Check if thread is busy
        if not self.thread_manager.acquire_thread_lock(
            message.thread_id, 
            message.channel_id,
            timeout=0  # Don't wait, return immediately if busy
        ):
            elapsed = time.time() - request_start_time
            self.log_info("="*80)
            self.log_info(f"REQUEST END | Thread: {thread_key} | Status: BUSY | Time: {elapsed:.2f}s")
            self.log_info("="*80)
            return Response(
                type="busy",
                content="Thread is currently processing another request"
            )
        
        try:
            # Get or rebuild thread state
            thread_state = self._get_or_rebuild_thread_state(
                message,
                client
            )
            
            # Set platform-specific system prompt if not already set
            if not thread_state.system_prompt:
                thread_state.system_prompt = self._get_system_prompt(client)
            
            # Process any attachments (images and other files)
            image_inputs, unsupported_files = self._process_attachments(message, client)
            
            # Check for unsupported files and notify user
            if unsupported_files:
                file_types = set()
                file_names = []
                for file in unsupported_files:
                    file_types.add(file['mimetype'])
                    file_names.append(file['name'])
                
                types_str = ", ".join(sorted(file_types))
                files_str = ", ".join(f"*{name}*" for name in file_names)
                
                unsupported_msg = "⚠️ *Unsupported File Type*\n\n"
                unsupported_msg += f"I noticed you uploaded: {files_str}\n\n"
                unsupported_msg += f"*File type(s):* `{types_str}`\n\n"
                unsupported_msg += "───────────────\n"
                unsupported_msg += "*Currently supported:*\n"
                unsupported_msg += "• Images (JPEG, PNG, GIF, WebP)\n\n"
                unsupported_msg += "_Support for additional file types may be added in the future._"
                
                # If there's also text or images, continue processing those
                if (message.text and message.text.strip()) or image_inputs:
                    unsupported_msg += "\n\nI'll process your text/image request now."
                    # Add the unsupported files warning to conversation
                    thread_state.add_message("user", f"[Uploaded unsupported file(s): {files_str}]")
                    thread_state.add_message("assistant", unsupported_msg)
                    # Continue processing if we have text or images
                else:
                    # Only unsupported files were uploaded, nothing else to process
                    thread_state.add_message("user", f"[Uploaded unsupported file(s): {files_str}]")
                    thread_state.add_message("assistant", unsupported_msg)
                    elapsed = time.time() - request_start_time
                    self.log_info("="*80)
                    self.log_info(f"REQUEST END | Thread: {thread_key} | Status: UNSUPPORTED_FILE | Time: {elapsed:.2f}s")
                    self.log_info("="*80)
                    return Response(
                        type="text",
                        content=unsupported_msg
                    )
            
            # Build user content
            user_content = self._build_user_content(message.text, image_inputs)
            
            # Check if we're handling a clarification response
            if thread_state.pending_clarification:
                self.log_debug("Processing clarification response")
                # Re-classify with the clarification context
                original_request = thread_state.pending_clarification.get("original_request", "")
                combined_context = f"{original_request} - Clarification: {message.text}"
                
                intent = self.openai_client.classify_intent(
                    thread_state.get_recent_messages(),
                    combined_context,
                    has_attached_images=len(image_inputs) > 0
                )
                
                # Clear the pending clarification
                thread_state.pending_clarification = None
                
                # Use the original request text for processing
                message.text = original_request
                self.log_debug(f"Clarified intent: {intent}")
            
            # Determine intent based on context
            elif image_inputs:
                # User uploaded images - determine if it's vision or edit request
                if not message.text or message.text.strip() == "":
                    # No text with images - default to vision (analyze)
                    intent = "vision"
                    self.log_debug("No text with images - defaulting to vision analysis")
                else:
                    # Has text with images - classify if it's edit or vision
                    self._update_status(client, message.channel_id, thinking_id, 
                                      "Understanding your request...")
                    intent = self.openai_client.classify_intent(
                        thread_state.get_recent_messages(),
                        message.text,
                        has_attached_images=len(image_inputs) > 0
                    )
                    # Handle classification based on uploaded images
                    if intent == "vision":
                        # Already correctly classified as vision/analysis
                        pass
                    elif intent == "new_image":
                        # "new_image" with uploads means edit
                        intent = "edit_image"
                    elif intent == "ambiguous_image":
                        # Ambiguous with uploads - default to vision for things like "compare"
                        intent = "vision"
                    elif intent == "edit_image":
                        # Already correctly classified
                        pass
                    elif intent == "text_only":
                        # Not image-related but has images - default to vision
                        intent = "vision"
                    # else keep the intent as-is
            else:
                # No images uploaded - standard classification
                self._update_status(client, message.channel_id, thinking_id, 
                                  "Understanding your request...")
                intent = self.openai_client.classify_intent(
                    thread_state.get_recent_messages(),
                    message.text if message.text else "",
                    has_attached_images=False  # Already checked - no images here
                )
            
            self.log_debug(f"Classified intent: {intent}")
            
            # Handle ambiguous intent
            if intent == "ambiguous_image":
                # Check if there are recent images to clarify about
                has_recent_image = self._has_recent_image(thread_state)
                
                if has_recent_image:
                    # Store the pending clarification
                    thread_state.pending_clarification = {
                        "type": "image_intent",
                        "original_request": message.text
                    }
                    
                    # Add clarification to thread history
                    thread_state.add_message("user", message.text)
                    
                    # Check if it's an uploaded image or generated one
                    has_uploaded = any("files.slack.com" in msg.get("content", "") 
                                     for msg in thread_state.messages[-5:] 
                                     if msg.get("role") == "user")
                    
                    if has_uploaded:
                        clarification_msg = "Would you like me to edit the uploaded image, or create a new image based on your description?"
                    else:
                        clarification_msg = "Would you like me to modify the image I just created, or generate a completely new one?"
                    
                    thread_state.add_message("assistant", clarification_msg)
                    
                    elapsed = time.time() - request_start_time
                    self.log_info("="*80)
                    self.log_info(f"REQUEST END | Thread: {thread_key} | Status: CLARIFICATION | Time: {elapsed:.2f}s")
                    self.log_info("="*80)
                    return Response(
                        type="text",
                        content=clarification_msg
                    )
                else:
                    # No recent images, treat as new generation
                    intent = "new_image"
                    self.log_debug("No recent images found, treating ambiguous as new generation")
            
            # Update thinking indicator if generating/editing image
            if intent in ["new_image", "edit_image"] and thinking_id:
                self._update_thinking_for_image(client, message.channel_id, thinking_id)
            
            # Generate response based on intent
            if intent == "new_image":
                response = self._handle_image_generation(message.text, thread_state, client, message.channel_id, thinking_id)
            elif intent == "edit_image":
                # Check if we have uploaded images or need to find recent ones
                if image_inputs:
                    # User uploaded images with edit request
                    # Extract URLs from attachments for tracking
                    attachment_urls = [att.get("url") for att in message.attachments if att.get("type") == "image"]
                    response = self._handle_image_edit(
                        message.text,
                        image_inputs,
                        thread_state,
                        client,
                        message.channel_id,
                        thinking_id,
                        attachment_urls
                    )
                else:
                    # Try to find and edit recent image
                    response = self._handle_image_modification(
                        message.text, 
                        thread_state, 
                        message.thread_id,
                        client,
                        message.channel_id,
                        thinking_id
                    )
            elif intent == "vision":
                # Update status to show we're analyzing images (handle plural)
                if image_inputs:
                    image_count = len(image_inputs)
                    if image_count == 1:
                        status_msg = f"{config.analyze_emoji} Analyzing image..."
                    else:
                        status_msg = f"{config.analyze_emoji} Analyzing {image_count} images..."
                else:
                    status_msg = f"{config.analyze_emoji} Looking for images to analyze..."
                self._update_status(client, message.channel_id, thinking_id, status_msg)
                # Vision analysis - but check if we actually have images
                if image_inputs:
                    # User uploaded images for vision analysis
                    response = self._handle_vision_analysis(message.text, image_inputs, thread_state, message.attachments, 
                                                           client, message.channel_id, thinking_id)
                else:
                    # Vision-related question but no images - try to find previous images
                    self.log_debug("Vision intent detected but no images attached - searching for previous images")
                    response = self._handle_vision_without_upload(
                        message.text, 
                        thread_state, 
                        client, 
                        message.channel_id, 
                        thinking_id
                    )
            else:
                response = self._handle_text_response(user_content, thread_state, client, message.channel_id, thinking_id)
            
            # DEBUG: Print conversation history after processing
            import json
            print("\n" + "="*80)
            print("DEBUG: CONVERSATION HISTORY (RAW JSON)")
            print("="*80)
            print(json.dumps(thread_state.messages, indent=2))
            print("="*80 + "\n")
            
            elapsed = time.time() - request_start_time
            response_type = response.type if response else "None"
            self.log_info("="*80)
            self.log_info(f"REQUEST END | Thread: {thread_key} | Status: {response_type.upper()} | Time: {elapsed:.2f}s")
            self.log_info("="*80)
            return response
            
        except Exception as e:
            self.log_error(f"Error processing message: {e}", exc_info=True)
            elapsed = time.time() - request_start_time
            self.log_info("="*80)
            self.log_info(f"REQUEST END | Thread: {thread_key} | Status: ERROR | Time: {elapsed:.2f}s")
            self.log_info("="*80)
            return Response(
                type="error",
                content=str(e)
            )
        finally:
            self.thread_manager.release_thread_lock(
                message.thread_id,
                message.channel_id
            )
    
    def _get_or_rebuild_thread_state(
        self,
        message: Message,
        client: BaseClient
    ) -> Any:
        """Get existing thread state or rebuild from platform history"""
        thread_state = self.thread_manager.get_or_create_thread(
            message.thread_id,
            message.channel_id
        )
        
        # If thread has no messages, rebuild from platform
        if not thread_state.messages:
            self.log_info(f"Rebuilding thread state for {message.thread_id}")
            
            # Get history from platform
            history = client.get_thread_history(
                message.channel_id,
                message.thread_id
            )
            
            # Get current message timestamp to exclude it
            current_ts = message.metadata.get("ts")
            
            # Convert to thread state messages
            for hist_msg in history:
                # Skip the current message being processed
                if hist_msg.metadata.get("ts") == current_ts:
                    continue
                    
                # Determine role based on metadata
                is_bot = hist_msg.metadata.get("is_bot", False)
                role = "assistant" if is_bot else "user"
                
                # Build content with attachment info
                content = hist_msg.text
                
                # Track image URLs for bot messages
                if is_bot and hist_msg.attachments:
                    for attachment in hist_msg.attachments:
                        if attachment.get("type") == "image":
                            url = attachment.get("url")
                            # Check for both generated and edited image markers
                            if url and content and ("Generated image:" in content or "Edited image:" in content):
                                # Append URL to the breadcrumb if not already present
                                if "<" not in content:  # Don't add if URL already there
                                    content += f" <{url}>"
                                break  # Only add first image URL
                
                # Add user upload breadcrumbs with URLs
                if not is_bot and hist_msg.attachments:
                    att_count = len(hist_msg.attachments)
                    content += f" [Uploaded {att_count} file(s)]"
                    # Add URLs for uploaded images
                    for attachment in hist_msg.attachments:
                        if attachment.get("type") == "image" and attachment.get("url"):
                            content += f" <{attachment['url']}>"
                
                thread_state.add_message(role, content)
            
            self.log_info(f"Rebuilt thread with {len(thread_state.messages)} messages")
        
        return thread_state
    
    def _process_attachments(
        self,
        message: Message,
        client: BaseClient
    ) -> Tuple[List[Dict], List[Dict]]:
        """Process message attachments (mainly images)
        
        Returns:
            Tuple of (image_inputs, unsupported_files)
        """
        image_inputs = []
        unsupported_files = []
        image_count = 0
        max_images = 10
        
        for attachment in message.attachments:
            file_type = attachment.get("type", "unknown")
            file_name = attachment.get("name", "unnamed file")
            
            if file_type == "image":
                # Stop if we've reached the image limit
                if image_count >= max_images:
                    self.log_warning(f"Limiting to {max_images} images (user uploaded more)")
                    continue
                    
                try:
                    # Download the image
                    image_data = client.download_file(
                        attachment.get("url"),
                        attachment.get("id")
                    )
                    
                    if image_data:
                        # Convert to base64
                        base64_data = base64.b64encode(image_data).decode('utf-8')
                        
                        # Format for Responses API with base64
                        mimetype = attachment.get("mimetype", "image/png")
                        image_inputs.append({
                            "type": "input_image",
                            "image_url": f"data:{mimetype};base64,{base64_data}"
                        })
                        
                        image_count += 1
                        self.log_debug(f"Processed image {image_count}/{max_images}: {file_name}")
                
                except Exception as e:
                    self.log_error(f"Error processing attachment: {e}")
            else:
                # Track unsupported file types
                mimetype = attachment.get("mimetype", "unknown")
                unsupported_files.append({
                    "name": file_name,
                    "type": file_type,
                    "mimetype": mimetype
                })
                self.log_debug(f"Unsupported file type: {file_type} ({mimetype}) - {file_name}")
        
        return image_inputs, unsupported_files
    
    def _build_user_content(self, text: str, image_inputs: List[Dict]) -> Any:
        """Build user message content"""
        if image_inputs:
            # Multi-part content with text and images
            content = [{"type": "input_text", "text": text}]
            content.extend(image_inputs)
            return content
        else:
            # Simple text content
            return text
    
    def _extract_image_registry(self, thread_state) -> List[Dict[str, str]]:
        """Extract all image URLs and descriptions from thread state"""
        image_registry = []
        
        for msg in thread_state.messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    # Check for any image markers
                    image_markers = ["Generated image:", "Edited image:", "Analyzed uploaded image:"]
                    for marker in image_markers:
                        if marker in content:
                            # Extract URL if present
                            url = None
                            if "<" in content and ">" in content:
                                url_start = content.rfind("<")
                                url_end = content.rfind(">")
                                if url_start < url_end:
                                    url = content[url_start + 1:url_end]
                            
                            # Extract description based on marker type
                            if marker == "Analyzed uploaded image:":
                                # For analysis, we want to note this is the original uploaded image
                                desc_start = content.find(marker) + len(marker)
                                desc_end = content.find("<") if "<" in content else len(content)
                                description = f"[Original] {content[desc_start:desc_end].strip()}"
                            else:
                                # For generated/edited images
                                desc_start = content.find(marker) + len(marker)
                                desc_end = content.find("<") if "<" in content else len(content)
                                description = content[desc_start:desc_end].strip()
                            
                            if url or marker == "Image analysis:":
                                image_registry.append({
                                    "url": url if url else "[Uploaded image - URL pending]",
                                    "description": description,
                                    "type": marker.replace(":", "").lower().replace(" ", "_")
                                })
                            break  # Only process first marker found
        
        return image_registry
    
    def _has_recent_image(self, thread_state) -> bool:
        """Check if there are recent images in the conversation"""
        # Check last few messages for image generation breadcrumbs or uploaded images
        for msg in thread_state.messages[-5:]:  # Check last 5 messages
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if isinstance(content, str):
                    # Look for image generation markers
                    if any(marker in content.lower() for marker in [
                        "generated image:",
                        "here's the image",
                        "created an image",
                        "edited image:"
                    ]):
                        return True
            elif msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    # Look for uploaded image URLs (Slack format)
                    if "files.slack.com" in content or "[Uploaded" in content:
                        return True
        
        # Also check asset ledger if available
        asset_ledger = self.thread_manager.get_asset_ledger(thread_state.thread_ts)
        if asset_ledger and asset_ledger.images:
            # Check if any images were created in last 5 minutes
            current_time = time.time()
            for img in asset_ledger.get_recent_images(3):
                if current_time - img.get("timestamp", 0) < 300:  # 5 minutes
                    return True
        
        return False
    
    def _get_system_prompt(self, client: BaseClient) -> str:
        """Get the appropriate system prompt based on the client platform"""
        client_name = client.name.lower()
        
        if "slack" in client_name:
            return SLACK_SYSTEM_PROMPT
        elif "discord" in client_name:
            return DISCORD_SYSTEM_PROMPT
        else:
            # Default/CLI prompt
            return CLI_SYSTEM_PROMPT
    
    def _update_status(self, client: BaseClient, channel_id: str, thinking_id: Optional[str], message: str, emoji: Optional[str] = None):
        """Update the thinking indicator with a status message"""
        if thinking_id and hasattr(client, 'update_message'):
            status_emoji = emoji or config.thinking_emoji
            client.update_message(
                channel_id,
                thinking_id,
                f"{status_emoji} {message}"
            )
            self.log_debug(f"Status updated: {message}")
        elif not thinking_id:
            self.log_debug("No thinking_id provided for status update")
        else:
            self.log_debug("Client doesn't support message updates")
    
    def _update_thinking_for_image(self, client: BaseClient, channel_id: str, thinking_id: str):
        """Update the thinking indicator to show image generation message"""
        self._update_status(client, channel_id, thinking_id, 
                          "Generating image. This could take up to a minute, please wait...",
                          emoji=config.circle_loader_emoji)
    
    def _handle_text_response(self, user_content: Any, thread_state, client: BaseClient, 
                              channel_id: str = None, thinking_id: Optional[str] = None,
                              attachment_urls: Optional[List[str]] = None) -> Response:
        """Handle text-only response generation"""
        # Check if streaming is enabled and supported
        if (hasattr(client, 'supports_streaming') and client.supports_streaming() and 
            thinking_id is not None):  # Streaming requires a message ID to update
            return self._handle_streaming_text_response(user_content, thread_state, client, channel_id, thinking_id, attachment_urls)
        
        # Fall back to non-streaming logic
        # For vision requests with images, store only a text breadcrumb with URLs, not the base64 data
        if isinstance(user_content, list):
            # Extract text and count images from the multi-part content
            text_parts = []
            image_count = 0
            for item in user_content:
                if item.get("type") == "input_text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "input_image":
                    image_count += 1
            
            # Create breadcrumb text for thread history
            breadcrumb_text = " ".join(text_parts).strip()
            if image_count > 0:
                breadcrumb_text += f" [Uploaded {image_count} file(s)]"
                # Add URLs if we have them
                if attachment_urls:
                    for url in attachment_urls:
                        breadcrumb_text += f" <{url}>"
            
            # Add simplified breadcrumb to thread state (no base64 data)
            thread_state.add_message("user", breadcrumb_text)
            
            # Use the full content with images for the actual API call
            messages_for_api = thread_state.messages[:-1] + [{"role": "user", "content": user_content}]
        else:
            # Simple text content - add as-is
            thread_state.add_message("user", user_content)
            messages_for_api = thread_state.messages
        
        # Get thread config
        thread_config = config.get_thread_config(thread_state.config_overrides)
        
        # Use thread's system prompt (which is now platform-specific)
        system_prompt = thread_state.system_prompt or self._get_system_prompt(client)
        
        # Update status before generating
        self._update_status(client, channel_id, thinking_id, "Generating response...")
        
        # Check if web search should be available
        if config.enable_web_search:
            # Use web search model if specified, otherwise use thread config model
            model = config.web_search_model or thread_config["model"]
            
            # Generate response with web search tool available
            response_text = self.openai_client.create_text_response_with_tools(
                messages=messages_for_api,
                tools=[{"type": "web_search"}],
                model=model,
                temperature=thread_config["temperature"],
                max_tokens=thread_config["max_tokens"],
                system_prompt=system_prompt,
                reasoning_effort=thread_config.get("reasoning_effort"),
                verbosity=thread_config.get("verbosity"),
                store=False  # Match the existing behavior
            )
        else:
            # Generate response without tools
            response_text = self.openai_client.create_text_response(
                messages=messages_for_api,
                model=thread_config["model"],
                temperature=thread_config["temperature"],
                max_tokens=thread_config["max_tokens"],
                system_prompt=system_prompt,
                reasoning_effort=thread_config.get("reasoning_effort"),
                verbosity=thread_config.get("verbosity")
            )
        
        # Check if response used web search and add citation note
        if config.enable_web_search:
            # Look for indicators that web search was used
            # OpenAI typically includes numbered citations [1], [2] or URLs when web search is used
            if any(marker in response_text for marker in ["[1]", "[2]", "[3]", "http://", "https://"]):
                self.log_info("Response includes web search results")
        
        # Add assistant response to thread state
        thread_state.add_message("assistant", response_text)
        
        return Response(
            type="text",
            content=response_text
        )
    
    def _handle_streaming_text_response(self, user_content: Any, thread_state, client: BaseClient, 
                                      channel_id: str = None, thinking_id: Optional[str] = None,
                                      attachment_urls: Optional[List[str]] = None) -> Response:
        """Handle text-only response generation with streaming support"""
        # Check if client supports streaming
        if not hasattr(client, 'supports_streaming') or not client.supports_streaming():
            self.log_debug("Client doesn't support streaming, falling back to non-streaming")
            return self._handle_text_response(user_content, thread_state, client, channel_id, thinking_id, attachment_urls)
        
        # Get streaming configuration from client
        streaming_config = client.get_streaming_config() if hasattr(client, 'get_streaming_config') else {}
        
        # Create streaming buffer and rate limit manager
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
        
        self.log_info("Starting streaming response generation")
        
        # Process user content for thread state (same as non-streaming)
        if isinstance(user_content, list):
            # Extract text and count images from the multi-part content
            text_parts = []
            image_count = 0
            for item in user_content:
                if item.get("type") == "input_text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "input_image":
                    image_count += 1
            
            # Create breadcrumb text for thread history
            breadcrumb_text = " ".join(text_parts).strip()
            if image_count > 0:
                breadcrumb_text += f" [Uploaded {image_count} file(s)]"
                # Add URLs if we have them
                if attachment_urls:
                    for url in attachment_urls:
                        breadcrumb_text += f" <{url}>"
            
            # Add simplified breadcrumb to thread state (no base64 data)
            thread_state.add_message("user", breadcrumb_text)
            
            # Use the full content with images for the actual API call
            messages_for_api = thread_state.messages[:-1] + [{"role": "user", "content": user_content}]
        else:
            # Simple text content - add as-is
            thread_state.add_message("user", user_content)
            messages_for_api = thread_state.messages
        
        # Get thread config
        thread_config = config.get_thread_config(thread_state.config_overrides)
        
        # Use thread's system prompt (which is now platform-specific)
        system_prompt = thread_state.system_prompt or self._get_system_prompt(client)
        
        # Post an initial message to get the message ID for streaming updates
        # For streaming with potential tools, start with "Working on it" 
        # (will be overridden if tools are used)
        initial_message = f"{config.thinking_emoji} Working on it..."
        if thinking_id:
            # Update existing thinking message
            message_id = thinking_id
            client.update_message(channel_id, message_id, initial_message)
        else:
            # We need a way to post a message and get its ID - this would depend on client implementation
            self.log_warning("No thinking_id provided for streaming - falling back to non-streaming")
            return self._handle_text_response(user_content, thread_state, client, channel_id, thinking_id, attachment_urls)
        
        # Track tool states for status updates
        tool_states = {
            "web_search": False,
            "file_search": False,
            "image_generation": False
        }
        
        # Track search counts
        search_counts = {
            "web_search": 0,
            "file_search": 0
        }
        
        # Track if we've started streaming text yet
        text_streaming_started = False
        
        # Define tool event callback
        def tool_callback(tool_type: str, status: str):
            """Handle tool events for status updates"""
            if status == "started":
                # Tool just started - update status with appropriate emoji
                if tool_type == "web_search":
                    if not tool_states["web_search"]:
                        tool_states["web_search"] = True
                    search_counts["web_search"] += 1
                    # Show search count consistently for all searches
                    status_msg = f"{config.web_search_emoji} Searching the web (query {search_counts['web_search']})..."
                    try:
                        # Use update_message_streaming for consistency with streaming flow
                        result = client.update_message_streaming(channel_id, message_id, status_msg)
                        if result["success"]:
                            self.log_info(f"Web search #{search_counts['web_search']} started - updated status")
                        else:
                            self.log_warning(f"Failed to update web search status: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        self.log_error(f"Error updating web search status: {e}")
                elif tool_type == "file_search":
                    if not tool_states["file_search"]:
                        tool_states["file_search"] = True
                    search_counts["file_search"] += 1
                    # Show search count consistently for all searches
                    status_msg = f"{config.web_search_emoji} Searching files (query {search_counts['file_search']})..."
                    try:
                        result = client.update_message_streaming(channel_id, message_id, status_msg)
                        if result["success"]:
                            self.log_info(f"File search #{search_counts['file_search']} started - updated status")
                        else:
                            self.log_warning(f"Failed to update file search status: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        self.log_error(f"Error updating file search status: {e}")
                elif tool_type == "image_generation" and not tool_states["image_generation"]:
                    tool_states["image_generation"] = True
                    status_msg = f"{config.circle_loader_emoji} Generating image..."
                    try:
                        result = client.update_message_streaming(channel_id, message_id, status_msg)
                        if result["success"]:
                            self.log_info(f"Image generation started - updated status")
                        else:
                            self.log_warning(f"Failed to update image gen status: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        self.log_error(f"Error updating image gen status: {e}")
            elif status == "completed":
                # Tool completed - clear the status for that tool
                if tool_type in tool_states:
                    tool_states[tool_type] = False
                    # Don't update status here - let the next event (another tool or text streaming) handle it
                    self.log_info(f"{tool_type} completed")
        
        # Define the streaming callback
        def stream_callback(text_chunk: str):
            """Callback function called with each text chunk from OpenAI"""
            # Check if this is the completion signal (None)
            if text_chunk is None:
                # Stream is complete - flush any remaining buffered text WITHOUT loading indicator
                if buffer.has_pending_update() and rate_limiter.can_make_request():
                    self.log_info("Flushing final buffered text")
                    rate_limiter.record_request_attempt()
                    # Use raw text for final flush - no loading indicator since stream is complete
                    final_text = buffer.get_complete_text()  # No loading indicator on completion
                    try:
                        result = client.update_message_streaming(channel_id, message_id, final_text)
                        if result["success"]:
                            rate_limiter.record_success()
                            buffer.mark_updated()
                    except Exception as e:
                        self.log_error(f"Error flushing final text: {e}")
                return
            
            if not text_chunk:
                return
                
            # Add chunk to buffer
            buffer.add_chunk(text_chunk)
            
            # Check if it's time to update
            if buffer.should_update() and rate_limiter.can_make_request():
                rate_limiter.record_request_attempt()
                
                # Get display-safe text with closed fences
                display_text = buffer.get_display_text()
                # Add loading indicator at the end to show streaming is in progress
                # Use loading_ellipse_emoji for streaming content instead of thinking_emoji
                display_text_with_indicator = f"{display_text} {config.loading_ellipse_emoji}"
                
                # Call client.update_message_streaming with indicator
                try:
                    result = client.update_message_streaming(channel_id, message_id, display_text_with_indicator)
                    
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
                                self.log_warning("Circuit breaker opened - will complete without further streaming")
                        else:
                            rate_limiter.record_failure(is_rate_limit=False)
                            self.log_warning(f"Message update failed: {result.get('error', 'Unknown error')}")
                            
                except Exception as e:
                    rate_limiter.record_failure(is_rate_limit=False)
                    self.log_error(f"Error updating streaming message: {e}")
        
        # Start streaming from OpenAI with the callback
        try:
            if config.enable_web_search:
                # Use web search model if specified, otherwise use thread config model
                model = config.web_search_model or thread_config["model"]
                
                # Generate response with web search tool available
                response_text = self.openai_client.create_streaming_response_with_tools(
                    messages=messages_for_api,
                    tools=[{"type": "web_search"}],
                    stream_callback=stream_callback,
                    tool_callback=tool_callback,  # Add tool callback
                    model=model,
                    temperature=thread_config["temperature"],
                    max_tokens=thread_config["max_tokens"],
                    system_prompt=system_prompt,
                    reasoning_effort=thread_config.get("reasoning_effort"),
                    verbosity=thread_config.get("verbosity"),
                    store=False  # Match the existing behavior
                )
            else:
                # Generate response without tools
                response_text = self.openai_client.create_streaming_response(
                    messages=messages_for_api,
                    stream_callback=stream_callback,
                    tool_callback=tool_callback,  # Add tool callback even without tools (in case of built-in tools)
                    model=thread_config["model"],
                    temperature=thread_config["temperature"],
                    max_tokens=thread_config["max_tokens"],
                    system_prompt=system_prompt,
                    reasoning_effort=thread_config.get("reasoning_effort"),
                    verbosity=thread_config.get("verbosity")
                )
            
            # Safety check: ensure all text was sent AND remove loading indicator
            # Always send a final update to ensure the loading indicator is removed
            if response_text != buffer.last_sent_text or True:  # Always update to remove indicator
                if response_text != buffer.last_sent_text:
                    self.log_warning(f"Text mismatch after streaming - sending correction update "
                                   f"(sent: {len(buffer.last_sent_text)}, should be: {len(response_text)} chars)")
                else:
                    self.log_debug("Sending final update to ensure loading indicator is removed")
                try:
                    final_result = client.update_message_streaming(channel_id, message_id, response_text)
                    if not final_result["success"]:
                        self.log_error(f"Final correction update failed: {final_result.get('error', 'Unknown error')}")
                except Exception as e:
                    self.log_error(f"Error in final correction update: {e}")
            
            # Note: To properly detect if web search was used, we'd need to track
            # tool events during streaming. The presence of URLs doesn't mean web search was used.
            
            # Add assistant response to thread state
            thread_state.add_message("assistant", response_text)
            
            # Log streaming stats
            stats = rate_limiter.get_stats()
            buffer_stats = buffer.get_stats()
            self.log_info(f"Streaming completed: {stats['successful_requests']}/{stats['total_requests']} updates, "
                         f"final length: {buffer_stats['text_length']} chars")
            
            return Response(
                type="text",
                content=response_text,
                metadata={"streamed": True, "message_id": message_id}
            )
            
        except Exception as e:
            self.log_error(f"Error in streaming response generation: {e}")
            
            # Try to remove the loading indicator if we had a message_id
            if message_id and hasattr(client, 'update_message_streaming'):
                try:
                    # Send whatever text we have without the loading indicator
                    error_text = buffer.get_complete_text() if buffer.has_content() else "An error occurred during streaming."
                    client.update_message_streaming(channel_id, message_id, error_text)
                except Exception as cleanup_error:
                    self.log_debug(f"Could not remove loading indicator: {cleanup_error}")
            
            # Fall back to non-streaming on error
            self.log_info("Falling back to non-streaming due to error")
            return self._handle_text_response(user_content, thread_state, client, channel_id, thinking_id, attachment_urls)

    def _handle_vision_analysis(self, user_text: str, image_inputs: List[Dict], thread_state, attachments: List[Dict],
                               client: BaseClient, channel_id: str, thinking_id: Optional[str]) -> Response:
        """Handle vision analysis of uploaded images"""
        if not image_inputs:
            return Response(
                type="error",
                content="No images found to analyze"
            )
        
        # Extract base64 data from image inputs
        images_to_analyze = []
        for img_input in image_inputs:
            if img_input.get("type") == "input_image":
                # Extract from data URL format
                image_url = img_input.get("image_url", "")
                if image_url.startswith("data:"):
                    parts = image_url.split(",", 1)
                    if len(parts) == 2:
                        _, base64_data = parts
                        images_to_analyze.append(base64_data)
        
        if not images_to_analyze:
            return Response(
                type="error", 
                content="Could not process uploaded images"
            )
        
        self.log_info(f"Analyzing {len(images_to_analyze)} image(s) with prompt: {user_text[:100]}...")
        
        # Update status with proper pluralization
        if len(images_to_analyze) == 1:
            status_msg = "Analyzing your image..."
        else:
            status_msg = f"Analyzing {len(images_to_analyze)} images..."
        self._update_status(client, channel_id, thinking_id, status_msg, emoji=config.analyze_emoji)
        
        # Analyze images with enhanced prompt
        analysis_result = self.openai_client.analyze_images(
            images=images_to_analyze,
            question=user_text if user_text else "Please describe this image.",
            detail="high",
            enhance_prompt=True  # Enable prompt enhancement for detailed analysis
        )
        
        # Create breadcrumb for thread state with URLs
        breadcrumb_text = user_text if user_text else "Analyze image"
        breadcrumb_text += f" [Uploaded {len(images_to_analyze)} file(s)]"
        
        # Add URLs from attachments if available
        for att in attachments:
            if att.get("url"):
                breadcrumb_text += f" <{att['url']}>"
        
        # Add to thread state
        thread_state.add_message("user", breadcrumb_text)
        thread_state.add_message("assistant", analysis_result)
        
        return Response(
            type="text",
            content=analysis_result
        )
    
    def _handle_image_generation(self, prompt: str, thread_state, client: BaseClient, 
                                channel_id: str, thinking_id: Optional[str]) -> Response:
        """Handle image generation request"""
        self.log_info(f"Generating image for prompt: {prompt[:100]}...")
        
        self._update_status(client, channel_id, thinking_id, "Enhancing your prompt...")
        
        # Get thread config
        thread_config = config.get_thread_config(thread_state.config_overrides)
        
        # Generate image with conversation context for better prompt enhancement
        self._update_status(client, channel_id, thinking_id, "Creating your image...", emoji=config.circle_loader_emoji)
        
        image_data = self.openai_client.generate_image(
            prompt=prompt,
            size=thread_config.get("image_size"),
            quality=thread_config.get("image_quality"),
            enhance_prompt=True,
            conversation_history=thread_state.get_recent_messages()  # Pass conversation context
        )
        
        # Store in asset ledger
        asset_ledger = self.thread_manager.get_or_create_asset_ledger(thread_state.thread_ts)
        asset_ledger.add_image(
            image_data.base64_data,
            image_data.prompt,  # Use the enhanced prompt
            time.time()
        )
        
        # Add breadcrumb to thread state with the enhanced prompt used
        thread_state.add_message("user", prompt)
        # URL will be added after upload, for now just the prompt
        thread_state.add_message("assistant", f"Generated image: {image_data.prompt}")
        
        return Response(
            type="image",
            content=image_data
        )
    
    def _find_target_image(self, user_text: str, thread_state, client: BaseClient) -> Optional[str]:
        """Find the target image URL based on user's reference"""
        # First try to find generated images
        image_registry = self._extract_image_registry(thread_state)
        
        # Also check for uploaded images in user messages
        uploaded_image_urls = []
        for msg in thread_state.messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and "[Uploaded" in content and "<" in content:
                    # Extract URLs from upload breadcrumbs
                    urls = re.findall(r'<([^>]+)>', content)
                    uploaded_image_urls.extend(urls)
        
        # Combine both sources
        all_available_images = []
        
        # Add generated images with descriptions
        for img in image_registry:
            all_available_images.append({
                "url": img["url"],
                "description": img["description"],
                "type": "generated"
            })
        
        # Add uploaded images (most recent first)
        for url in reversed(uploaded_image_urls):
            all_available_images.append({
                "url": url,
                "description": "uploaded image",
                "type": "uploaded"
            })
        
        if not all_available_images:
            return None
        
        # If only one image, use it
        if len(all_available_images) == 1:
            self.log_debug(f"Only one image found, using it: {all_available_images[0]['url']}")
            return all_available_images[0]["url"]
        
        # Check for explicit references
        user_text_lower = user_text.lower()
        
        # Try ordinal references
        ordinals = {
            "first": 0, "1st": 0, "second": 1, "2nd": 1, 
            "third": 2, "3rd": 2, "last": -1, "latest": -1,
            "previous": -2, "recent": -1
        }
        
        for word, index in ordinals.items():
            if word in user_text_lower:
                try:
                    url = all_available_images[index]["url"]
                    self.log_debug(f"Found ordinal reference '{word}', using image: {url}")
                    return url
                except IndexError:
                    pass
        
        # If ambiguous and multiple images, use utility model to match
        if len(all_available_images) > 1:
            # Build context for matching
            context = "Available images:\n"
            for i, img in enumerate(all_available_images, 1):
                if img["type"] == "uploaded":
                    context += f"{i}. Uploaded image\n"
                else:
                    context += f"{i}. Generated: {img['description'][:100]}...\n"
            
            # Include vision analysis if available for better context
            # Search backwards for analysis that mentions these specific images
            for msg in reversed(thread_state.messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        # Check if this message contains analysis of our current images
                        # Either by checking for URLs or for "Image X:" pattern with multiple images
                        has_current_images = False
                        
                        # Check if any of our current image URLs are mentioned
                        if uploaded_image_urls:
                            has_current_images = any(url in content for url in uploaded_image_urls)
                        
                        # Or check if it's analyzing multiple images (likely our batch)
                        if not has_current_images and len(all_available_images) > 1:
                            has_current_images = ("Image 1:" in content and "Image 2:" in content)
                        
                        if has_current_images:
                            context += "\nAnalysis of these images:\n"
                            # Extract the first part of analysis that likely contains image descriptions
                            # Limit to reasonable length to avoid token overflow
                            analysis_excerpt = content[:1500]
                            context += analysis_excerpt + "\n"
                            break
            
            context += f"\nUser reference: '{user_text}'\n"
            context += "Which image number best matches the user's reference? Respond with just the number."
            
            try:
                # Use utility model to find best match
                match_response = self.openai_client.create_text_response(
                    messages=[{"role": "user", "content": context}],
                    model=config.utility_model,
                    temperature=0.1,
                    max_tokens=50,  # Increased to handle reasoning tokens
                    reasoning_effort=config.utility_reasoning_effort,  # Use utility config
                    verbosity=config.utility_verbosity  # Use utility config
                )
                
                # Parse response to get index
                numbers = re.findall(r'\d+', match_response)
                if numbers:
                    index = int(numbers[0]) - 1  # Convert to 0-based
                    if 0 <= index < len(all_available_images):
                        url = all_available_images[index]["url"]
                        self.log_debug(f"Utility model matched to image {index + 1}: {url}")
                        return url
            except Exception as e:
                self.log_warning(f"Failed to match image reference: {e}")
        
        # Default to most recent image (last in list)
        url = all_available_images[-1]["url"]
        image_type = all_available_images[-1]["type"]
        self.log_debug(f"Using most recent {image_type} image by default: {url}")
        return url
    
    def _handle_image_modification(
        self,
        text: str,
        thread_state,
        thread_id: str,
        client: BaseClient,
        channel_id: str,
        thinking_id: Optional[str]
    ) -> Response:
        """Handle image modification request by finding and editing the target image"""
        self._update_status(client, channel_id, thinking_id, "Finding the image to edit...", emoji=config.web_search_emoji)
        
        # Try to find target image URL from conversation
        target_url = self._find_target_image(text, thread_state, client)
        
        if target_url:
            # Download the image from Slack
            self.log_info(f"Found target image URL: {target_url}")
            self._update_status(client, channel_id, thinking_id, "Downloading the image...")
            
            try:
                # Download the image
                image_data = client.download_file(target_url, None)
                
                if image_data:
                    # Convert to base64 for editing
                    import base64
                    base64_data = base64.b64encode(image_data).decode('utf-8')
                    
                    # Analyze the image first
                    self.log_debug("Analyzing image for context")
                    self._update_status(client, channel_id, thinking_id, "Analyzing the image...", emoji=config.analyze_emoji)
                    
                    image_description = self.openai_client.analyze_images(
                        images=[base64_data],
                        question="Describe this image focusing on subject, colors, composition, and style.",
                        detail="high"
                    )
                    
                    # Prepare for edit
                    self.log_info(f"Editing existing image with request: {text}")
                    self._update_status(client, channel_id, thinking_id, "Enhancing your edit request...")
                    
                    # Get thread config
                    thread_config = config.get_thread_config(thread_state.config_overrides)
                    
                    # Edit the image
                    self._update_status(client, channel_id, thinking_id, "Editing your image...", emoji=config.circle_loader_emoji)
                    
                    edited_image = self.openai_client.edit_image(
                        input_images=[base64_data],
                        prompt=text,
                        image_description=image_description,
                        input_mimetypes=["image/png"],
                        input_fidelity=thread_config.get("input_fidelity", "high"),
                        background=thread_config.get("image_background", "auto"),
                        output_format=thread_config.get("image_format", "png"),
                        output_compression=thread_config.get("image_compression", 100),
                        enhance_prompt=True,
                        conversation_history=thread_state.messages
                    )
                    
                    # Add breadcrumbs
                    thread_state.add_message("user", text)
                    thread_state.add_message("assistant", f"Edited image: {edited_image.prompt}")
                    
                    return Response(
                        type="image",
                        content=edited_image
                    )
                else:
                    self.log_warning(f"Failed to download image from URL: {target_url}")
                    
            except Exception as e:
                self.log_error(f"Error editing image from URL: {e}")
        
        # Fallback to old behavior if no URL found
        self.log_info("No image URL found, falling back to generation based on description")
        
        # Look for image descriptions in history
        image_registry = self._extract_image_registry(thread_state)
        if image_registry:
            # Use the most recent image description
            previous_prompt = image_registry[-1]["description"]
            context_prompt = f"Previous image: {previous_prompt}\nModification request: {text}"
            return self._handle_image_generation(context_prompt, thread_state, client, channel_id, thinking_id)
        else:
            # No previous images, treat as new generation
            return self._handle_image_generation(text, thread_state, client, channel_id, thinking_id)
    
    def _handle_vision_without_upload(
        self,
        text: str,
        thread_state,
        client: BaseClient,
        channel_id: str,
        thinking_id: Optional[str]
    ) -> Response:
        """Handle vision request when no images are uploaded - use text response with context"""
        
        # Check if we have any images in the conversation that provide context
        image_registry = self._extract_image_registry(thread_state)
        has_images = bool(image_registry) or self._has_recent_image(thread_state)
        
        if has_images:
            # We have image context in the conversation - let the model use that
            self.log_info("Vision intent with image context in history - using text response with context")
        else:
            self.log_info("Vision intent but no images found in conversation")
        
        # Always use text response - the model will have context from the thread history
        # This avoids re-analysis and keeps responses concise
        return self._handle_text_response(text, thread_state, client, channel_id, thinking_id)
    
    def _handle_image_edit(
        self,
        text: str,
        image_inputs: List[Dict],
        thread_state,
        client: BaseClient,
        channel_id: str,
        thinking_id: Optional[str],
        attachment_urls: Optional[List[str]] = None
    ) -> Response:
        """Handle image editing with uploaded images"""
        self._update_status(client, channel_id, thinking_id, "Processing uploaded images...")
        
        # Extract base64 data and mime types from image inputs
        input_images = []
        input_mimetypes = []
        for img_input in image_inputs:
            if img_input.get("type") == "input_image":
                # Extract from data URL format
                image_url = img_input.get("image_url", "")
                if image_url.startswith("data:"):
                    # Parse data URL: data:image/png;base64,xxxxx
                    parts = image_url.split(",", 1)
                    if len(parts) == 2:
                        header, base64_data = parts
                        # Extract mimetype from header
                        mimetype_part = header.split(";")[0].replace("data:", "")
                        mimetype = mimetype_part if mimetype_part else "image/png"
                        
                        # OpenAI doesn't support GIF for editing, convert to PNG
                        if mimetype == "image/gif":
                            self.log_warning("Converting GIF to PNG for image edit (GIF not supported)")
                            mimetype = "image/png"
                        
                        input_images.append(base64_data)
                        input_mimetypes.append(mimetype)
        
        if not input_images:
            # Shouldn't happen but fallback to generation
            return self._handle_image_generation(text, thread_state, client, channel_id, thinking_id)
        
        self.log_info(f"Editing {len(input_images)} uploaded image(s)")
        
        # First, analyze the uploaded images to get context
        self.log_debug("Analyzing uploaded images for context")
        # Update status with proper pluralization
        if len(input_images) == 1:
            status_msg = "Analyzing your uploaded image..."
        else:
            status_msg = f"Analyzing {len(input_images)} uploaded images..."
        self._update_status(client, channel_id, thinking_id, status_msg, emoji=config.analyze_emoji)
        
        # Log the analysis prompt
        print("\n" + "="*80)
        print("DEBUG: IMAGE EDIT FLOW - STEP 1: ANALYZE IMAGE")
        print("="*80)
        print(f"Analysis Question: {IMAGE_ANALYSIS_PROMPT}")
        print("="*80)
        
        try:
            # Analyze the images to understand what's in them
            image_description = self.openai_client.analyze_images(
                images=input_images,
                question=IMAGE_ANALYSIS_PROMPT,
                detail="high"
            )
            
            # Log the full analysis result
            print("\n" + "="*80)
            print("DEBUG: IMAGE EDIT FLOW - STEP 2: ANALYSIS RESULT")
            print("="*80)
            print(f"Image Description (Full):\n{image_description}")
            print("="*80)
            
            # Don't show the analysis - just update status to show we're editing
            # The analysis is only used internally for better edit quality
            if thinking_id:
                self._update_status(client, channel_id, thinking_id, "Editing your image...", emoji=config.circle_loader_emoji)
            
            # Store the description and user request separately for clean enhancement
            image_analysis = image_description
            user_edit_request = text
            
        except Exception as e:
            self.log_warning(f"Failed to analyze images, continuing without context: {e}")
            image_analysis = None
            user_edit_request = text
            print("\n" + "="*80)
            print("DEBUG: IMAGE EDIT FLOW - ANALYSIS FAILED")
            print("="*80)
            print(f"Error: {e}")
            print(f"Falling back to user prompt only: {text}")
            print("="*80)
        
        # Get thread config for settings
        thread_config = config.get_thread_config(thread_state.config_overrides)
        
        # Use the edit_image API with separated inputs
        try:
            
            image_data = self.openai_client.edit_image(
                input_images=input_images,
                input_mimetypes=input_mimetypes,
                prompt=user_edit_request,  # Just the user's request
                image_description=image_analysis,  # The analyzed description
                input_fidelity=thread_config.get("input_fidelity", "high"),
                background=thread_config.get("image_background", "auto"),
                output_format=thread_config.get("image_format", "png"),
                output_compression=thread_config.get("image_compression", 100),
                enhance_prompt=True,
                conversation_history=thread_state.messages if thread_state.messages else None
            )
        except Exception as e:
            self.log_error(f"Error editing image: {e}")
            return Response(
                type="error",
                content=f"Failed to edit image: {str(e)}"
            )
        
        # Add breadcrumb to thread state with URLs for consistency with rebuild
        user_breadcrumb = text or "Edit uploaded image"
        if attachment_urls:
            user_breadcrumb += f" [Uploaded {len(attachment_urls)} file(s)]"
            for url in attachment_urls:
                user_breadcrumb += f" <{url}>"
        thread_state.add_message("user", user_breadcrumb)
        
        # Include brief context about what was uploaded and edited for future vision questions
        # This helps the model answer follow-up questions without re-analysis
        if image_analysis:
            # Add a brief note about the original image content (first 200 chars of analysis)
            brief_context = image_analysis[:200] + "..." if len(image_analysis) > 200 else image_analysis
            assistant_breadcrumb = f"Edited image: {image_data.prompt} [Original: {brief_context}]"
        else:
            assistant_breadcrumb = f"Edited image: {image_data.prompt}"
        thread_state.add_message("assistant", assistant_breadcrumb)
        
        return Response(
            type="image",
            content=image_data
        )
    
    def update_last_image_url(self, channel_id: str, thread_id: str, url: str):
        """Update the last assistant message with the image URL"""
        thread_state = self.thread_manager.get_or_create_thread(thread_id, channel_id)
        
        # Find the last assistant message with "Generated image:"
        for i in range(len(thread_state.messages) - 1, -1, -1):
            msg = thread_state.messages[i]
            if msg.get("role") == "assistant" and "Generated image:" in msg.get("content", ""):
                # Add URL if not already present
                if "<" not in msg["content"]:
                    msg["content"] += f" <{url}>"
                    self.log_debug(f"Updated message with URL: {url}")
                break
    
    def update_thread_config(
        self,
        channel_id: str,
        thread_id: str,
        config_updates: Dict[str, Any]
    ):
        """Update configuration for a specific thread"""
        self.thread_manager.update_thread_config(
            thread_id,
            channel_id,
            config_updates
        )
        
    def get_stats(self) -> Dict[str, int]:
        """Get processor statistics"""
        return self.thread_manager.get_stats()