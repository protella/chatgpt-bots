"""
Shared Message Processor
Client-agnostic message processing logic
"""
import asyncio
import time
from typing import Optional
from base_client import BaseClient, Message, Response
from thread_manager import AsyncThreadStateManager
from openai_client import OpenAIClient
from config import config
from logger import LoggerMixin
from .thread_management import ThreadManagementMixin
from .handlers.text import TextHandlerMixin
from .handlers.vision import VisionHandlerMixin
from .handlers.image_gen import ImageGenerationMixin
from .handlers.image_edit import ImageEditMixin
from .utilities import MessageUtilitiesMixin
from image_url_handler import ImageURLHandler
try:
    from document_handler import DocumentHandler
    DOCUMENT_HANDLER_AVAILABLE = True
except ImportError:
    DocumentHandler = None
    DOCUMENT_HANDLER_AVAILABLE = False


class MessageProcessor(ThreadManagementMixin,
                       TextHandlerMixin,
                       VisionHandlerMixin,
                       ImageGenerationMixin,
                       ImageEditMixin,
                       MessageUtilitiesMixin,
                       LoggerMixin):
    """Handles message processing logic independent of chat platform"""
    
    def __init__(self, db = None):
        self.thread_manager = AsyncThreadStateManager(db=db)
        self.openai_client = OpenAIClient()
        self.image_url_handler = ImageURLHandler()
        self.document_handler = DocumentHandler() if DOCUMENT_HANDLER_AVAILABLE else None
        self.db = db  # Database manager
        if not DOCUMENT_HANDLER_AVAILABLE:
            self.log_warning("DocumentHandler not available - document processing will be disabled")
        self.log_info(f"MessageProcessor initialized {'with' if db else 'without'} database")
    








    async def process_message(self, message: Message, client: BaseClient, thinking_id: Optional[str] = None) -> Optional[Response]:
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
        self.log_info("")
        self.log_info("="*100)
        self.log_info(f"REQUEST START | Thread: {thread_key} | User: {username}")
        self.log_info(f"Message: {message.text[:100] if message.text else 'No text'}{'...' if message.text and len(message.text) > 100 else ''}")
        self.log_info("="*100)
        self.log_info("")
        
        request_start_time = time.time()

        # Check if thread is busy
        lock_acquired = False
        try:
            lock_acquired = await self.thread_manager.acquire_thread_lock(
                message.thread_id,
                message.channel_id,
                timeout=0  # Don't wait, return immediately if busy
            )
        except Exception as lock_error:
            self.log_error(f"Lock acquisition failed with error: {lock_error}", exc_info=True)
            raise

        if not lock_acquired:
            elapsed = time.time() - request_start_time
            self.log_info("")
            self.log_info("="*100)
            self.log_info(f"REQUEST END | Thread: {thread_key} | Status: BUSY | Time: {elapsed:.2f}s")
            self.log_info("="*100)
            self.log_info("")
            return Response(
                type="busy",
                content="Thread is currently processing another request"
            )
        
        try:
            # Get or rebuild thread state
            thread_state = await self._get_or_rebuild_thread_state(
                message,
                client,
                thinking_id
            )
            
            # Check if this thread had a previous timeout
            if hasattr(thread_state, 'had_timeout') and thread_state.had_timeout:
                # Send timeout notification to user
                timeout_msg = f"⚠️ Your previous request timed out - OpenAI's API didn't respond within {int(config.api_timeout_read)} seconds."
                await client.send_message(
                    channel_id=message.channel_id,
                    text=timeout_msg,
                    thread_id=message.thread_id
                )
                # Clear the timeout flag
                thread_state.had_timeout = False
                self.log_info(f"Notified user about previous timeout in thread {thread_key}")
            
            # Note: 80% context warning moved to after response generation
            
            # Get thread config to determine model (with user preferences)
            thread_config = config.get_thread_config(
                overrides=thread_state.config_overrides,
                user_id=message.user_id,
                db=self.db
            )
            
            # Update thread state with current model for token limit calculations
            thread_state.current_model = thread_config["model"]
            
            # Always regenerate system prompt to get current time
            user_timezone = message.metadata.get("user_timezone", "UTC") if message.metadata else "UTC"
            user_tz_label = message.metadata.get("user_tz_label", None) if message.metadata else None
            user_real_name = message.metadata.get("user_real_name", None) if message.metadata else None
            user_email = message.metadata.get("user_email", None) if message.metadata else None
            web_search_enabled = thread_config.get('enable_web_search', config.enable_web_search)
            thread_state.system_prompt = self._get_system_prompt(client, user_timezone, user_tz_label, user_real_name, user_email, thread_config["model"], web_search_enabled, thread_state.has_trimmed_messages, thread_config.get('custom_instructions'))
            
            # Process any attachments (images, documents, and other files)
            image_inputs, document_inputs, unsupported_files = await self._process_attachments(message, client, thinking_id)
            
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
                unsupported_msg += "• Images (JPEG, PNG, GIF, WebP)\n"
                unsupported_msg += "• Documents (PDF, DOCX, XLSX, CSV, TXT, etc.)\n\n"
                unsupported_msg += "_Support for additional file types may be added in the future._"
                
                # If there's also text, images, or documents, continue processing those
                if (message.text and message.text.strip()) or image_inputs or document_inputs:
                    unsupported_msg += "\n\nI'll process your text/image/document request now."
                    # Add the unsupported files warning to conversation
                    thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
                    message_ts = message.metadata.get("ts") if message.metadata else None
                    formatted_content = self._format_user_content_with_username(f"[Uploaded unsupported file(s): {files_str}]", message)
                    self._add_message_with_token_management(thread_state, "user", formatted_content, db=self.db, thread_key=thread_key, message_ts=message_ts)
                    self._add_message_with_token_management(thread_state, "assistant", unsupported_msg, db=self.db, thread_key=thread_key)
                    # Continue processing if we have text or images
                else:
                    # Only unsupported files were uploaded, nothing else to process
                    thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
                    message_ts = message.metadata.get("ts") if message.metadata else None
                    formatted_content = self._format_user_content_with_username(f"[Uploaded unsupported file(s): {files_str}]", message)
                    self._add_message_with_token_management(thread_state, "user", formatted_content, db=self.db, thread_key=thread_key, message_ts=message_ts)
                    self._add_message_with_token_management(thread_state, "assistant", unsupported_msg, db=self.db, thread_key=thread_key)
                    elapsed = time.time() - request_start_time
                    self.log_info("")
                    self.log_info("="*100)
                    self.log_info(f"REQUEST END | Thread: {thread_key} | Status: UNSUPPORTED_FILE | Time: {elapsed:.2f}s")
                    self.log_info("="*100)
                    self.log_info("")
                    return Response(
                        type="text",
                        content=unsupported_msg
                    )
            
            # Build user content
            # First, format the base text with username
            username = message.metadata.get("username", "User") if message.metadata else "User"
            base_text_with_username = f"{username}: {message.text}" if message.text else f"{username}:"
            
            # If we have documents, enhance the text with document content
            enhanced_text = base_text_with_username
            if document_inputs:
                enhanced_text = self._build_message_with_documents(base_text_with_username, document_inputs)
            
            user_content = self._build_user_content(enhanced_text, image_inputs)
            
            # Check if adding this message would exceed limits and trim if needed
            # We temporarily add the message to check, then remove it
            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
            message_ts = message.metadata.get("ts") if message.metadata else None
            
            # Determine what content to use for checking
            content_to_check = enhanced_text if not image_inputs else (enhanced_text if enhanced_text else f"{username}: [uploaded image(s) for analysis]")
            
            # Temporarily add message to check total tokens
            temp_message = {"role": "user", "content": content_to_check}
            thread_state.messages.append(temp_message)
            
            # Check token count with the new message
            model = thread_state.current_model or config.gpt_model
            max_tokens = config.get_model_token_limit(model)
            current_tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages)
            
            # Apply smart trimming if needed - keep trimming until under limit
            if current_tokens > max_tokens:
                self.log_info(f"Thread would exceed limit with new message ({current_tokens}/{max_tokens} tokens), applying smart trim")
                total_trimmed = 0
                
                # Keep trimming until we're under the limit
                while current_tokens > max_tokens:
                    # Smart trim will work on all messages including the temp one
                    trimmed_count = await self._smart_trim_with_summarization(thread_state)
                    total_trimmed += trimmed_count
                    
                    if trimmed_count == 0:
                        # No more messages to trim, we've done all we can
                        self.log_warning(f"Cannot trim further - still at {current_tokens} tokens")
                        break
                    
                    # Recalculate tokens after trimming
                    current_tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages)
                    self.log_debug(f"After trimming {trimmed_count} messages, now at {current_tokens}/{max_tokens} tokens")
                
                if total_trimmed > 0:
                    self.log_info(f"Smart trim complete: {total_trimmed} total messages processed, final: {current_tokens}/{max_tokens} tokens")
            
            # Remove the temporary message - handlers will add it properly
            if thread_state.messages and thread_state.messages[-1] == temp_message:
                thread_state.messages.pop()
            
            # Check if this single message alone exceeds the model's context window
            message_tokens = self.thread_manager._token_counter.count_message_tokens(temp_message)
            max_model_tokens = config.thread_max_token_count  # Model's context limit
            
            # Check if this single message exceeds the model's context window
            if message_tokens > max_model_tokens:
                error_msg = (
                    f"❌ Your message is too large for the model to process.\n\n"
                    f"• Message size: {message_tokens:,} tokens\n"
                    f"• Model limit: {max_model_tokens:,} tokens\n\n"
                    f"Please reduce the size of your documents or split them into smaller requests."
                )
                
                # Log the issue
                self.log_error(f"Message exceeds context window: {message_tokens} > {max_model_tokens}")
                
                # Add minimal breadcrumb to history
                thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
                message_ts = message.metadata.get("ts") if message.metadata else None
                formatted_error_breadcrumb = self._format_user_content_with_username(
                    f"[Attempted to upload {len(document_inputs)} document(s) - exceeded context limit]", 
                    message
                )
                self._add_message_with_token_management(
                    thread_state, "user", 
                    formatted_error_breadcrumb,
                    db=self.db, thread_key=thread_key, message_ts=message_ts
                )
                self._add_message_with_token_management(
                    thread_state, "assistant", error_msg,
                    db=self.db, thread_key=thread_key
                )
                
                return Response(type="error", content=error_msg)
            
            # Check if we're handling a clarification response
            if thread_state.pending_clarification:
                self.log_debug("Processing clarification response")
                # Re-classify with the clarification context
                original_request = thread_state.pending_clarification.get("original_request", "")
                combined_context = f"{original_request} - Clarification: {message.text}"
                
                # Truncate documents in history for intent classification
                trimmed_history_for_intent = []
                for msg in thread_state.messages:
                    msg_copy = msg.copy()
                    content = str(msg_copy.get("content", ""))
                    if "=== DOCUMENT:" in content and len(content) > 500:
                        # Truncate document content
                        msg_copy["content"] = content[:200] + "...[document content truncated for classification]..."
                    trimmed_history_for_intent.append(msg_copy)
                
                # Use already-trimmed thread state for intent classification
                # Only mark as having attachments if there are actual image uploads
                intent = await self.openai_client.classify_intent(
                    trimmed_history_for_intent,  # Documents truncated for classification
                    combined_context,
                    has_attached_images=len(image_inputs) > 0
                )

                # Check if intent classification failed
                if intent == 'error':
                    # Update thinking message
                    if thinking_id:
                        self._update_status(client, message.channel_id, thinking_id,
                                          "Service temporarily unavailable.",
                                          emoji=config.error_emoji)

                    elapsed = time.time() - request_start_time
                    self.log_info("")
                    self.log_info("="*100)
                    self.log_info(f"REQUEST END | Thread: {thread_key} | Status: INTENT_ERROR | Time: {elapsed:.2f}s")
                    self.log_info("="*100)
                    self.log_info("")

                    return Response(
                        type="error",
                        content="⚠️ **OpenAI Not Responding**\n\n"
                                "OpenAI's API failed to respond after multiple attempts.\n\n"
                                "This is an OpenAI service issue. Please try again in a few moments."
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
                    # For intent classification, include the current message in trimming
                    # Temporarily add the current message
                    temp_intent_msg = {"role": "user", "content": enhanced_text}
                    thread_state.messages.append(temp_intent_msg)
                    
                    # Pre-trim messages INCLUDING the current message
                    trimmed_messages = await self._pre_trim_messages_for_api(thread_state.messages, model=thread_state.current_model, thread_state=thread_state)
                    
                    # Remove the temp message
                    if thread_state.messages and thread_state.messages[-1] == temp_intent_msg:
                        thread_state.messages.pop()
                    
                    # Use the actual current message (enhanced_text) for classification
                    intent_text = enhanced_text if enhanced_text else ""
                    
                    # Truncate document content for intent classification to avoid confusion
                    # Intent classifier should focus on the user's request, not document content
                    if "=== DOCUMENT:" in intent_text and len(intent_text) > 500:
                        # Keep just the document header and user's actual message
                        parts = intent_text.split("\n\n\n=== DOCUMENT:")
                        if len(parts) > 1:
                            user_msg = parts[0]  # User's actual message before document
                            # Add truncated document indicator
                            intent_text = user_msg + "\n[Document content truncated for classification]"
                        else:
                            # Document is at the start, truncate it
                            intent_text = intent_text[:200] + "...[document truncated]..."
                    
                    # Now pass the trimmed history WITHOUT the current message (since classify_intent expects it separately)
                    trimmed_history = [msg for msg in trimmed_messages if msg != trimmed_messages[-1]] if trimmed_messages else []
                    
                    # Also truncate documents in history for intent classification
                    trimmed_history_for_intent = []
                    for msg in trimmed_history:
                        msg_copy = msg.copy()
                        content = str(msg_copy.get("content", ""))
                        if "=== DOCUMENT:" in content and len(content) > 500:
                            # Truncate document content
                            msg_copy["content"] = content[:200] + "...[document content truncated for classification]..."
                        trimmed_history_for_intent.append(msg_copy)
                    
                    # Only mark as having attachments if there are actual image uploads
                    # Documents shouldn't affect image intent classification
                    intent = await self.openai_client.classify_intent(
                        trimmed_history_for_intent,  # Trimmed conversation history (without current)
                        intent_text,  # Current message (potentially trimmed/summarized)
                        has_attached_images=len(image_inputs) > 0
                    )
                    # If intent classification times out, it returns 'text_only' by default
                    # For uploaded images, override to vision if we got text_only (likely from timeout)
                    if intent == "text_only":
                        self.log_info("Intent unclear with uploaded images - defaulting to vision analysis")
                        intent = "vision"
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
                # For intent classification, include the current message in trimming
                # Temporarily add the current message
                temp_intent_msg = {"role": "user", "content": enhanced_text if enhanced_text else ""}
                thread_state.messages.append(temp_intent_msg)
                
                # Pre-trim messages INCLUDING the current message
                trimmed_messages = await self._pre_trim_messages_for_api(thread_state.messages, model=thread_state.current_model, thread_state=thread_state)
                
                # Remove the temp message
                if thread_state.messages and thread_state.messages[-1] == temp_intent_msg:
                    thread_state.messages.pop()
                
                # Use the actual current message (enhanced_text) for classification
                intent_text = enhanced_text if enhanced_text else ""
                
                # Truncate document content for intent classification to avoid confusion
                # Intent classifier should focus on the user's request, not document content
                if "=== DOCUMENT:" in intent_text and len(intent_text) > 500:
                    # Keep just the document header and user's actual message
                    parts = intent_text.split("\n\n\n=== DOCUMENT:")
                    if len(parts) > 1:
                        user_msg = parts[0]  # User's actual message before document
                        # Add truncated document indicator
                        intent_text = user_msg + "\n[Document content truncated for classification]"
                    else:
                        # Document is at the start, truncate it
                        intent_text = intent_text[:200] + "...[document truncated]..."
                
                # Now pass the trimmed history WITHOUT the current message (since classify_intent expects it separately)
                trimmed_history = [msg for msg in trimmed_messages if msg != trimmed_messages[-1]] if trimmed_messages else []
                
                # Also truncate documents in history for intent classification
                trimmed_history_for_intent = []
                for msg in trimmed_history:
                    msg_copy = msg.copy()
                    content = str(msg_copy.get("content", ""))
                    if "=== DOCUMENT:" in content and len(content) > 500:
                        # Truncate document content
                        msg_copy["content"] = content[:200] + "...[document content truncated for classification]..."
                    trimmed_history_for_intent.append(msg_copy)
                
                # Only mark as having attachments if there are actual image uploads
                intent = await self.openai_client.classify_intent(
                    trimmed_history_for_intent,  # Trimmed conversation history (without current)
                    intent_text,  # Current message (potentially trimmed/summarized)
                    has_attached_images=False  # Documents alone don't count as image attachments
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
                    thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
                    message_ts = message.metadata.get("ts") if message.metadata else None
                    formatted_text = self._format_user_content_with_username(message.text, message)
                    self._add_message_with_token_management(thread_state, "user", formatted_text, db=self.db, thread_key=thread_key, message_ts=message_ts)
                    
                    # Check if it's an uploaded image or generated one
                    has_uploaded = any("files.slack.com" in msg.get("content", "") 
                                     for msg in thread_state.messages[-5:] 
                                     if msg.get("role") == "user")
                    
                    if has_uploaded:
                        clarification_msg = "Would you like me to edit the uploaded image, or create a new image based on your description?"
                    else:
                        clarification_msg = "Would you like me to modify the image I just created, or generate a completely new one?"
                    
                    self._add_message_with_token_management(thread_state, "assistant", clarification_msg, db=self.db, thread_key=thread_key)
                    
                    elapsed = time.time() - request_start_time
                    self.log_info("")
                    self.log_info("="*100)
                    self.log_info(f"REQUEST END | Thread: {thread_key} | Status: CLARIFICATION | Time: {elapsed:.2f}s")
                    self.log_info("="*100)
                    self.log_info("")
                    return Response(
                        type="text",
                        content=clarification_msg
                    )
                else:
                    # No recent images, treat as new generation
                    intent = "new_image"
                    self.log_debug("No recent images found, treating ambiguous as new generation")
            
            # Update thinking indicator if generating/editing image (only for non-streaming)
            if intent in ["new_image", "edit_image"] and thinking_id:
                # Only show the image thinking message if we're not streaming
                # Note: We check global streaming here since we don't have thread_config yet
                if not (hasattr(client, 'supports_streaming') and client.supports_streaming() and config.enable_streaming):
                    self._update_thinking_for_image(client, message.channel_id, thinking_id)
            
            # Generate response based on intent
            if intent == "new_image":
                response = await self._handle_image_generation(message.text, thread_state, client, message.channel_id, thinking_id, message)
            elif intent == "edit_image":
                # Check if we have uploaded images or need to find recent ones
                if image_inputs:
                    # User uploaded images with edit request
                    # Extract URLs from attachments for tracking
                    attachment_urls = [att.get("url") for att in message.attachments if att.get("type") == "image"]
                    response = await self._handle_image_edit(
                        message.text,
                        image_inputs,
                        thread_state,
                        client,
                        message.channel_id,
                        thinking_id,
                        attachment_urls,
                        message
                    )
                else:
                    # Try to find and edit recent image
                    response = await self._handle_image_modification(
                        message.text,
                        thread_state,
                        message.thread_id,
                        client,
                        message.channel_id,
                        thinking_id,
                        message
                    )
            elif intent == "vision":
                # Vision analysis - but check if we actually have images or documents
                # Don't update status here, let _handle_vision_analysis manage the status flow
                if image_inputs or document_inputs:
                    # User uploaded images or documents for analysis
                    if document_inputs and not image_inputs:
                        # Documents only - show document-specific status
                        doc_count = len(document_inputs)
                        doc_names = ", ".join([d["filename"] for d in document_inputs[:3]])
                        if doc_count > 3:
                            doc_names += f" and {doc_count - 3} more"
                        status_msg = f"Analyzing {doc_count} document{'s' if doc_count > 1 else ''}: {doc_names}..."
                        self._update_status(client, message.channel_id, thinking_id, status_msg, emoji=config.analyze_emoji)
                        
                        # Documents are already in enhanced_text, just process as text with vision intent
                        response = await self._handle_text_response(user_content, thread_state, client, message, thinking_id, retry_count=0)
                    elif image_inputs and document_inputs:
                        # Both images and documents - use two-call approach
                        total_files = len(image_inputs) + len(document_inputs)
                        status_msg = f"Analyzing {len(image_inputs)} image{'s' if len(image_inputs) > 1 else ''} and {len(document_inputs)} document{'s' if len(document_inputs) > 1 else ''}..."
                        self._update_status(client, message.channel_id, thinking_id, status_msg, emoji=config.analyze_emoji)
                        
                        # Use new two-call approach for mixed content
                        response = await self._handle_mixed_content_analysis(
                            user_text=message.text,
                            image_inputs=image_inputs,
                            document_inputs=document_inputs,
                            thread_state=thread_state,
                            client=client,
                            channel_id=message.channel_id,
                            thinking_id=thinking_id,
                            message=message
                        )
                    else:
                        # Images only - use existing vision handler
                        response = await self._handle_vision_analysis(message.text, image_inputs, thread_state, message.attachments,
                                                               client, message.channel_id, thinking_id, message)
                else:
                    # Vision-related question but no images or documents - try to find previous images
                    self.log_debug("Vision intent detected but no files attached - searching for previous images")
                    response = await self._handle_vision_without_upload(
                        message.text,
                        thread_state,
                        client,
                        message.channel_id,
                        thinking_id,
                        message
                    )
            else:
                response = await self._handle_text_response(user_content, thread_state, client, message, thinking_id, retry_count=0)
            
            # DEBUG: Print conversation history after processing (with truncated content)
            import json
            print("\n" + "="*100)
            print("DEBUG: CONVERSATION HISTORY (TRUNCATED)")
            print("="*100)
            
            # Create a truncated version for debugging
            truncated_messages = []
            for msg in thread_state.messages:
                truncated_msg = msg.copy()
                content = str(truncated_msg.get("content", ""))
                if len(content) > 100:
                    truncated_msg["content"] = content[:100] + f"... [truncated {len(content) - 100} chars]"
                truncated_messages.append(truncated_msg)
            
            print(json.dumps(truncated_messages, indent=2))
            print("="*100 + "\n")
            
            elapsed = time.time() - request_start_time
            response_type = response.type if response else "None"
            
            # Calculate final token count
            final_tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages)
            
            # Check if we should show the 80% context warning AFTER the response
            if not thread_state.has_shown_80_percent_warning and response and response.type not in ["error", "busy"]:
                # Get current model's token limit and check usage
                model = thread_state.current_model or config.gpt_model
                max_tokens = config.get_model_token_limit(model)
                eighty_percent_threshold = int(max_tokens * config.token_cleanup_threshold)  # Using same threshold from .env
                
                if final_tokens > eighty_percent_threshold:
                    # Send one-time warning about context usage as a stylized message
                    warning_msg = (
                        f"```\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"📊 CONTEXT USAGE NOTIFICATION\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"\n"
                        f"Current Usage: {final_tokens:,} / {max_tokens:,} tokens\n"
                        f"({final_tokens/max_tokens:.0%} of available context)\n"
                        f"\n"
                        f"💡 Tips for optimal performance:\n"
                        f"   • Start new threads for unrelated topics\n"
                        f"   • Older messages may be auto-summarized\n"
                        f"   • Important context is always preserved\n"
                        f"\n"
                        f"✅ You can continue chatting normally\n"
                        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                        f"```"
                    )
                    
                    # Send as regular message so everyone in channel threads can see it
                    await client.send_message(
                        channel_id=message.channel_id,
                        text=warning_msg,
                        thread_id=message.thread_id
                    )
                    
                    # Mark that we've shown the warning
                    thread_state.has_shown_80_percent_warning = True
                    self.log_info(f"Sent 80% context warning for thread {thread_key}: {final_tokens}/{max_tokens} tokens")
            
            self.log_info("")
            self.log_info("="*100)
            self.log_info(f"REQUEST END | Thread: {thread_key} | Status: {response_type.upper()} | Time: {elapsed:.2f}s | Tokens: {final_tokens}")
            self.log_info("="*100)
            self.log_info("")
            return response
            
        except TimeoutError as e:
            # Handle timeout errors gracefully without stack trace
            elapsed = time.time() - request_start_time
            # Try to get token count even on error
            try:
                error_tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages) if 'thread_state' in locals() else 0
                token_info = f" | Tokens: {error_tokens}" if error_tokens > 0 else ""
            except Exception:
                token_info = ""

            # Get the operation type that timed out
            operation_type = getattr(e, 'operation_type', 'unknown')
            self.log_warning(f"Request timeout after {elapsed:.2f} seconds for thread {thread_key} (operation: {operation_type}): {e}")

            # Check if we should retry (only for text/intent operations, not image/vision operations)
            already_retried = getattr(e, 'already_retried', False)
            should_retry = (
                operation_type in ['text_normal', 'intent_classification']
                and not already_retried
                and 'intent' not in locals()  # Don't retry if we're already in intent classification
            )

            if should_retry:
                # Mark as retry attempt to prevent infinite loops
                e.already_retried = True

                # Update status to show retry
                if thinking_id and hasattr(client, 'update_message'):
                    retry_msg = "OpenAI is slow to respond. Retrying with shorter timeout..."
                    try:
                        self._update_status(client, message.channel_id, thinking_id, retry_msg, emoji="⏳")
                        self.log_debug("Updated thinking message to show retry attempt")
                    except Exception as update_error:
                        self.log_error(f"Failed to update thinking message for retry: {update_error}")

                self.log_info(f"Retrying {operation_type} operation with 60s timeout...")

                try:
                    # Retry the operation based on what failed
                    if operation_type == 'intent_classification':
                        # Re-run intent classification
                        if 'image_inputs' in locals() and image_inputs:
                            retry_intent = await self.openai_client.classify_intent(
                                messages=trimmed_history_for_intent if 'trimmed_history_for_intent' in locals() else [],
                                last_user_message=intent_text if 'intent_text' in locals() else str(user_content),
                                has_attached_images=len(image_inputs) > 0
                            )
                        else:
                            retry_intent = await self.openai_client.classify_intent(
                                messages=trimmed_history_for_intent if 'trimmed_history_for_intent' in locals() else [],
                                last_user_message=intent_text if 'intent_text' in locals() else str(user_content),
                                has_attached_images=False
                            )

                        # Continue with the classification result
                        if retry_intent == "error":
                            raise TimeoutError("Intent classification failed after retry")

                        # Use the retry result and continue processing
                        intent = retry_intent

                        # Process based on intent (re-enter the main flow)
                        if intent in ["new_image", "ambiguous_image"]:
                            if 'image_inputs' in locals() and image_inputs:
                                intent = "edit_image"

                        if intent == "new_image":
                            response = await self._handle_image_generation(
                                enhanced_text if 'enhanced_text' in locals() else user_content,
                                thread_state, client, message.channel_id, thinking_id, message
                            )
                        elif intent == "edit_image":
                            response = await self._handle_image_edit(
                                enhanced_text if 'enhanced_text' in locals() else user_content,
                                image_inputs if 'image_inputs' in locals() else [],
                                thread_state, client, message.channel_id, thinking_id,
                                attachment_urls if 'attachment_urls' in locals() else None,
                                message
                            )
                        elif intent == "vision":
                            response = await self._handle_vision_analysis(
                                enhanced_text if 'enhanced_text' in locals() else user_content,
                                image_inputs if 'image_inputs' in locals() else [],
                                thread_state, thread_state.attachments if hasattr(thread_state, 'attachments') else [],
                                client, message.channel_id, thinking_id, message
                            )
                        else:
                            response = await self._handle_text_response(
                                enhanced_text if 'enhanced_text' in locals() else user_content,
                                thread_state, client, message, thinking_id,
                                attachment_urls if 'attachment_urls' in locals() else None,
                                retry_count=1
                            )

                        self.log_info(f"Retry successful for {operation_type}")
                        return response

                    elif operation_type == 'text_normal':
                        # Retry text response with shorter timeout and retry_count=1
                        response = await self._handle_text_response(
                            enhanced_text if 'enhanced_text' in locals() else user_content,
                            thread_state, client, message, thinking_id,
                            attachment_urls if 'attachment_urls' in locals() else None,
                            retry_count=1
                        )
                        self.log_info(f"Retry successful for {operation_type}")
                        return response

                except TimeoutError as retry_error:
                    self.log_warning(f"Retry also failed for {operation_type}: {retry_error}")
                    # Continue to error handling below
                except Exception as retry_error:
                    self.log_error(f"Retry failed with unexpected error for {operation_type}: {retry_error}")
                    # Continue to error handling below

            self.log_info("")
            self.log_info("="*100)
            self.log_info(f"REQUEST END | Thread: {thread_key} | Status: TIMEOUT | Time: {elapsed:.2f}s{token_info}")
            self.log_info("="*100)
            self.log_info("")

            # Update thinking message to show final timeout
            if thinking_id and hasattr(client, 'update_message'):
                timeout_msg = "OpenAI is not responding. Try again shortly."
                try:
                    self._update_status(client, message.channel_id, thinking_id, timeout_msg, emoji=config.error_emoji)
                    self.log_debug("Updated thinking message to show timeout")
                except Exception as update_error:
                    self.log_error(f"Failed to update thinking message: {update_error}")

            # Mark thread as having a timeout for recovery
            if 'thread_state' in locals() and thread_state:
                thread_state.had_timeout = True

            error_message = (
                "⏱️ **Taking Too Long**\n\n"
                "OpenAI is being slow right now.\n\n"
                "Please try again in a moment."
            )

            return Response(
                type="error",
                content=error_message
            )
        except Exception as e:
            # Log full error details for non-timeout exceptions
            self.log_error(f"Error processing message: {e}", exc_info=True)
            elapsed = time.time() - request_start_time
            # Try to get token count even on error
            try:
                error_tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages) if 'thread_state' in locals() else 0
                token_info = f" | Tokens: {error_tokens}" if error_tokens > 0 else ""
            except Exception:
                token_info = ""

            self.log_info("")
            self.log_info("="*100)
            self.log_info(f"REQUEST END | Thread: {thread_key} | Status: ERROR | Time: {elapsed:.2f}s{token_info}")
            self.log_info("="*100)
            self.log_info("")

            # Check if this is a timeout error that wasn't caught as TimeoutError
            error_str = str(e)
            error_type = type(e).__name__

            # Check for various timeout error types that weren't caught as TimeoutError
            if any(timeout_indicator in error_str.lower() or timeout_indicator in error_type.lower()
                   for timeout_indicator in ['timeout', 'readtimeout', 'connecttimeout']):
                # Update thinking message to show timeout
                if thinking_id and hasattr(client, 'update_message'):
                    timeout_msg = f"{config.error_emoji} OpenAI is not responding. Try again shortly."
                    try:
                        await client.update_message(message.channel_id, thinking_id, timeout_msg)
                    except Exception:
                        pass  # Don't let update failure affect error handling

                # Mark as timeout for recovery
                if 'thread_state' in locals() and thread_state:
                    thread_state.had_timeout = True

                # Timeout-specific error message
                error_message = (
                    "⏱️ **Taking Too Long**\n\n"
                    "OpenAI is being slow right now.\n\n"
                    "Please try again in a moment."
                )
                self.log_warning(f"Request timeout (via string match) after {elapsed:.2f} seconds for thread {thread_key}")
            else:
                # Update thinking message to show error
                if thinking_id and hasattr(client, 'update_message'):
                    error_msg = f"{config.error_emoji} Something went wrong. Try again."
                    try:
                        await client.update_message(message.channel_id, thinking_id, error_msg)
                    except Exception:
                        pass  # Don't let update failure affect error handling

                # Generic error message - keep it simple for users
                # Log the actual error for debugging, but don't show technical details to user
                error_details = str(e)

                # Check for common error types and provide user-friendly messages
                if "rate" in error_details.lower() or "limit" in error_details.lower():
                    error_message = f"{config.error_emoji} **Too Many Requests**\n\nOpenAI is busy. Please wait a minute and try again."
                elif "context" in error_details.lower() or "token" in error_details.lower():
                    error_message = f"{config.error_emoji} **Message Too Long**\n\nYour message is too long. Please try a shorter request."
                elif "api" in error_details.lower() or "openai" in error_details.lower():
                    error_message = f"{config.error_emoji} **Service Issue**\n\nOpenAI is having problems. Please try again shortly."
                else:
                    # Generic fallback
                    error_message = f"{config.error_emoji} **Something Went Wrong**\n\nPlease try again. If this keeps happening, try later."

            return Response(
                type="error",
                content=error_message
            )
        finally:
            # Always release the thread lock, even on timeout
            try:
                await self.thread_manager.release_thread_lock(
                    message.thread_id,
                    message.channel_id
                )
            except Exception as lock_error:
                # Even if release fails, log it but don't crash
                self.log_error(f"Error releasing thread lock for {thread_key}: {lock_error}", exc_info=True)

    async def cleanup(self):
        """Clean up resources and close clients."""
        self.log_info("Cleaning up MessageProcessor resources...")
        if hasattr(self, 'openai_client') and self.openai_client:
            await self.openai_client.close()
        # Close thread manager resources if needed
        if hasattr(self.thread_manager, 'cleanup'):
            await self.thread_manager.cleanup()
        self.log_info("MessageProcessor cleanup completed")
    





















