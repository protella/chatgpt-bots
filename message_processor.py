"""
Shared Message Processor
Client-agnostic message processing logic
"""
import base64
import os
import re
import time
import datetime
import pytz
from typing import Dict, Any, List, Optional, Tuple
from base_client import BaseClient, Message, Response
from thread_manager import ThreadStateManager
from openai_client import OpenAIClient, ImageData
from config import config
from logger import LoggerMixin
from prompts import SLACK_SYSTEM_PROMPT, DISCORD_SYSTEM_PROMPT, CLI_SYSTEM_PROMPT, IMAGE_ANALYSIS_PROMPT
from streaming import StreamingBuffer, RateLimitManager, FenceHandler
from image_url_handler import ImageURLHandler
try:
    from document_handler import DocumentHandler
    DOCUMENT_HANDLER_AVAILABLE = True
except ImportError as e:
    DocumentHandler = None
    DOCUMENT_HANDLER_AVAILABLE = False


class MessageProcessor(LoggerMixin):
    """Handles message processing logic independent of chat platform"""
    
    def __init__(self, db = None):
        self.thread_manager = ThreadStateManager(db=db)
        self.openai_client = OpenAIClient()
        self.image_url_handler = ImageURLHandler()
        self.document_handler = DocumentHandler() if DOCUMENT_HANDLER_AVAILABLE else None
        self.db = db  # Database manager
        if not DOCUMENT_HANDLER_AVAILABLE:
            self.log_warning("DocumentHandler not available - document processing will be disabled")
        self.log_info(f"MessageProcessor initialized {'with' if db else 'without'} database")
    
    def _format_user_content_with_username(self, content: str, message: Message) -> str:
        """Format user content with username prefix for multi-user context
        
        Args:
            content: The message content to format
            message: The Message object containing metadata
            
        Returns:
            Content prefixed with username (e.g., "Alice: Hello")
        """
        username = message.metadata.get("username", "User") if message.metadata else "User"
        
        # Handle special content formats
        if not content or content.strip() == "":
            return f"{username}:"
        elif content.startswith("[") and content.endswith("]"):
            # Special bracketed content (e.g., "[uploaded image]")
            return f"{username}: {content}"
        else:
            # Normal text content
            return f"{username}: {content}"
    
    def _add_message_with_token_management(self, thread_state, role: str, content: Any, db=None, thread_key: str = None, message_ts: str = None, metadata: Dict[str, Any] = None, skip_auto_trim: bool = False):
        """Helper method to add messages with token management
        
        Args:
            skip_auto_trim: If True, skip the automatic simple trimming (used during rebuild to allow smart trimming later)
        """
        # Get dynamic token limit based on current model
        model = thread_state.current_model or config.gpt_model
        max_tokens = config.get_model_token_limit(model)
        
        # Count tokens for this message
        msg_tokens = self.thread_manager._token_counter.count_message_tokens({"role": role, "content": content})
        
        # Add the message
        # During rebuild, we skip auto-trim to allow smart trimming with summarization
        if skip_auto_trim:
            thread_state.add_message(
                role=role,
                content=content,
                db=db,
                thread_key=thread_key,
                message_ts=message_ts,
                metadata=metadata,
                token_counter=None,  # Skip automatic trimming
                max_tokens=None
            )
        else:
            thread_state.add_message(
                role=role,
                content=content,
                db=db,
                thread_key=thread_key,
                message_ts=message_ts,
                metadata=metadata,
                token_counter=self.thread_manager._token_counter,
                max_tokens=max_tokens
            )
        
        # Log token info in debug mode
        total_tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages)
        content_preview = str(content)[:50] + "..." if len(str(content)) > 50 else str(content)
        self.log_debug(f"MESSAGE ADDED | Role: {role} | Tokens: {msg_tokens} | Total: {total_tokens}/{max_tokens}")
    
    def _pre_trim_messages_for_api(self, messages: List[Dict[str, Any]], new_message_tokens: int = 0, model: str = None, thread_state=None) -> List[Dict[str, Any]]:
        """Pre-trim messages to fit within context window before sending to API
        
        Args:
            messages: List of messages to potentially trim
            new_message_tokens: Tokens that will be added (for pre-checks)
            model: Model name to get appropriate token limit
            thread_state: Optional thread state for smart trimming
            
        Returns:
            Trimmed list of messages that fits within context
        """
        # Get dynamic token limit based on model
        model = model or config.gpt_model
        max_tokens = config.get_model_token_limit(model)
        current_tokens = self.thread_manager._token_counter.count_thread_tokens(messages) + new_message_tokens
        
        if current_tokens <= max_tokens:
            return messages
        
        self.log_info(f"Pre-trimming messages: {current_tokens} tokens exceeds {max_tokens} limit")
        
        # If we have thread_state, use smart trimming
        if thread_state:
            # Apply smart trimming with document summarization
            trimmed_count = self._smart_trim_with_summarization(thread_state)
            if trimmed_count > 0:
                new_tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages)
                self.log_info(f"Smart trim complete: {current_tokens} → {new_tokens} tokens ({trimmed_count} messages processed)")
            return thread_state.messages
        
        # Fallback to basic trimming if no thread_state
        # Find first non-system message index
        start_index = 0
        for i, msg in enumerate(messages):
            if msg.get("role") not in ["system", "developer"]:
                start_index = i
                break
        
        # Create a copy to work with
        trimmed_messages = messages.copy()
        removed_count = 0
        
        # Remove messages from the beginning (after system messages)
        while current_tokens > max_tokens and len(trimmed_messages) > start_index + 1:
            if start_index < len(trimmed_messages) - 1:
                removed_msg = trimmed_messages.pop(start_index)
                removed_count += 1
                current_tokens = self.thread_manager._token_counter.count_thread_tokens(trimmed_messages) + new_message_tokens
                self.log_debug(f"Pre-trimmed message {removed_count}, tokens now: {current_tokens}")
            else:
                self.log_warning(f"Cannot trim further - would remove current message")
                break
        
        if removed_count > 0:
            self.log_info(f"Pre-trimmed {removed_count} messages to fit within context limit")
        
        return trimmed_messages
    
    def _should_preserve_message(self, msg: Dict[str, Any]) -> bool:
        """Determine if a message should be preserved during trimming
        
        Args:
            msg: Message dictionary to evaluate
            
        Returns:
            True if message should be preserved, False otherwise
        """
        # Never trim system messages
        if msg.get("role") in ["system", "developer"]:
            return True
        
        # Check metadata for special message types
        metadata = msg.get("metadata", {})
        # Note: document_upload is NOT preserved - documents can be summarized
        if metadata.get("type") in ["image_generation", "image_edit", "image_upload", "vision_analysis", "image_analysis"]:
            return True
        
        # Preserve summarized documents (they're already compressed)
        if metadata.get("summarized"):
            return True
        
        content = str(msg.get("content", ""))
        
        # Check for IMAGE URLs that might be needed for edits
        # Only preserve if it has image URLs (not document URLs)
        import re
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, content)
        for url in urls:
            # Only preserve if it's an image URL
            if any(ext in url.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg']):
                return True
            # Also preserve OpenAI generated image URLs
            if 'oaidalleapi' in url.lower() or 'dall-e' in url.lower():
                return True
        
        # Check for SUMMARIZED document markers - these are preserved
        # But full/unsummarized documents are NOT preserved (they can be trimmed/summarized)
        if "[SUMMARIZED" in content:
            return True
        
        # Check for injected analysis markers
        if "[Image Analysis:" in content or "[Vision Context:" in content:
            return True
        
        return False
    
    def _summarize_document_content(self, content: str) -> str:
        """Summarize document content to reduce token usage
        
        Args:
            content: Full document content with === DOCUMENT: markers
            
        Returns:
            Summarized version of the document content
        """
        try:
            # Extract document metadata
            import re
            doc_match = re.search(r'=== DOCUMENT: (.*?) ===.*?MIME Type: (.*?)\n', content, re.DOTALL)
            if not doc_match:
                return content  # Can't parse, return as-is
            
            filename = doc_match.group(1).strip()
            mimetype = doc_match.group(2).strip()
            
            # Extract page count if available
            pages_match = re.search(r'\((\d+) pages?\)', content)
            page_count = pages_match.group(1) if pages_match else "unknown"
            
            # Use OpenAI to summarize the document content
            # Extract just the document text (between headers)
            doc_text_match = re.search(r'=== DOCUMENT:.*?===\n(.*?)\n=== DOCUMENT END:', content, re.DOTALL)
            if not doc_text_match:
                return content
            
            doc_text = doc_text_match.group(1).strip()
            
            # Import the summarization prompt
            from prompts import DOCUMENT_SUMMARIZATION_PROMPT
            
            # Use proper role separation for document summarization
            # Developer message contains the instruction, user message contains the document
            summary = self.openai_client.create_text_response(
                messages=[
                    {"role": "developer", "content": DOCUMENT_SUMMARIZATION_PROMPT},
                    {"role": "user", "content": doc_text}  # Full document, no truncation
                ],
                model=config.utility_model,
                temperature=0.3,
                max_tokens=800,  # Increased for better summaries
                system_prompt=None  # Already using developer message above
            )
            
            # Format the summarized version
            summarized = f"""=== DOCUMENT: {filename} === ({page_count} pages)
MIME Type: {mimetype}
[SUMMARIZED - Original content reduced for context management]

{summary}

=== DOCUMENT END: {filename} ==="""
            
            self.log_info(f"Summarized document {filename}: {len(doc_text)} chars -> {len(summary)} chars")
            return summarized
            
        except Exception as e:
            self.log_error(f"Error summarizing document: {e}")
            return content  # Return original if summarization fails
    
    def _smart_trim_with_summarization(self, thread_state, trim_count: int = None) -> int:
        """Intelligently trim messages, summarizing documents only when they're in the trim list
        
        This method identifies the oldest N messages to be trimmed. If any contain
        unsummarized documents, it summarizes them in place (making them preserved).
        Then it trims any remaining non-preserved messages from the list.
        
        Args:
            thread_state: Thread state object to trim
            trim_count: Number of messages to trim (default from config)
            
        Returns:
            Number of messages actually trimmed or summarized
        """
        trim_count = trim_count or config.token_trim_message_count
        
        # Build list of ALL non-preserved message indices
        trimmable_indices = []
        for i, msg in enumerate(thread_state.messages):
            if not self._should_preserve_message(msg):
                trimmable_indices.append(i)
                # Debug: log if this is a document
                content = str(msg.get("content", ""))
                if "=== DOCUMENT:" in content and "[SUMMARIZED" not in content:
                    self.log_debug(f"Found unsummarized document at index {i} (trimmable)")
        
        self.log_debug(f"Found {len(trimmable_indices)} trimmable messages out of {len(thread_state.messages)} total")
        
        # Get the oldest N messages that would be trimmed
        indices_to_process = sorted(trimmable_indices)[:trim_count]
        self.log_debug(f"Processing oldest {len(indices_to_process)} messages: indices {indices_to_process}")
        
        if not indices_to_process:
            # No trimmable messages at all
            return 0
        
        # FIRST PASS: Check if any messages in the trim list contain unsummarized documents
        # If so, summarize them IN PLACE (they become preserved)
        documents_summarized = 0
        for idx in indices_to_process:
            if idx < len(thread_state.messages):
                msg = thread_state.messages[idx]
                content = str(msg.get("content", ""))
                
                # Check if this is an unsummarized document
                if "=== DOCUMENT:" in content and "[SUMMARIZED" not in content:
                    original_content = msg.get("content", "")
                    
                    self.log_info(f"Summarizing document at index {idx} (in trim list of {len(indices_to_process)} messages)")
                    
                    # Summarize the document
                    summarized_content = self._summarize_document_content(original_content)
                    
                    # Update the message IN PLACE with summarized content
                    thread_state.messages[idx]["content"] = summarized_content
                    
                    # Update metadata to indicate summarization
                    if "metadata" not in thread_state.messages[idx]:
                        thread_state.messages[idx]["metadata"] = {}
                    thread_state.messages[idx]["metadata"]["summarized"] = True
                    # Don't set type to document_upload - that would preserve it
                    thread_state.messages[idx]["metadata"]["original_length"] = len(original_content)
                    thread_state.messages[idx]["metadata"]["summarized_length"] = len(summarized_content)
                    
                    self.log_info(f"Summarized document: {len(original_content)} → {len(summarized_content)} chars")
                    documents_summarized += 1
        
        # If we summarized any documents, that's progress - return to recheck token count
        if documents_summarized > 0:
            return documents_summarized
        
        # SECOND PASS: No documents were summarized, so actually trim non-preserved messages
        # Re-check which messages are still trimmable (documents we just summarized are now preserved)
        messages_trimmed = 0
        still_trimmable = []
        
        for idx in indices_to_process:
            if idx < len(thread_state.messages):
                if not self._should_preserve_message(thread_state.messages[idx]):
                    still_trimmable.append(idx)
        
        # Remove messages in reverse order to maintain indices
        for idx in reversed(still_trimmable):
            if idx < len(thread_state.messages):
                removed_msg = thread_state.messages.pop(idx)
                messages_trimmed += 1
                self.log_debug(f"Trimmed message at index {idx}: {str(removed_msg.get('content', ''))[:50]}...")
        
        if messages_trimmed > 0:
            thread_state.has_trimmed_messages = True
            self.log_info(f"Smart-trimmed {messages_trimmed} messages from thread")
        
        return messages_trimmed
    
    def _smart_trim_oldest(self, thread_state, trim_count: int = None) -> int:
        """Intelligently trim oldest non-preserved messages from thread
        
        Args:
            thread_state: Thread state object to trim
            trim_count: Number of messages to trim (default from config)
            
        Returns:
            Number of messages actually trimmed
        """
        trim_count = trim_count or config.token_trim_message_count
        trimmed = 0
        
        # Build list of trimmable message indices
        trimmable_indices = []
        for i, msg in enumerate(thread_state.messages):
            if not self._should_preserve_message(msg):
                trimmable_indices.append(i)
        
        # Trim from oldest (lowest indices) first
        for i in sorted(trimmable_indices)[:trim_count]:
            # Adjust index for previously removed messages
            adjusted_index = i - trimmed
            if adjusted_index < len(thread_state.messages):
                removed_msg = thread_state.messages.pop(adjusted_index)
                trimmed += 1
                self.log_debug(f"Trimmed message at index {i}: {str(removed_msg.get('content', ''))[:50]}...")
        
        if trimmed > 0:
            thread_state.has_trimmed_messages = True
            self.log_info(f"Smart-trimmed {trimmed} messages from thread")
        
        return trimmed
    
    def _async_post_response_cleanup(self, thread_state, thread_key: str):
        """Asynchronously clean up thread after response is sent
        
        This runs after the response has been sent to Slack to proactively
        trim old messages before the next request. Will summarize documents
        before trimming them to preserve context.
        
        Args:
            thread_state: Thread state to potentially clean up
            thread_key: Thread identifier for database operations
        """
        try:
            # Get current model's token limit
            model = thread_state.current_model or config.gpt_model
            max_tokens = config.get_model_token_limit(model)
            cleanup_threshold = int(max_tokens * config.token_cleanup_threshold)
            
            # Check current token usage
            current_tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages)
            
            if current_tokens > cleanup_threshold:
                self.log_info(f"Thread at {current_tokens}/{max_tokens} tokens ({current_tokens/max_tokens:.1%}), triggering cleanup")
                
                # Use smart trim with summarization for documents
                trimmed = self._smart_trim_with_summarization(thread_state)
                
                if trimmed > 0:
                    # Update token count after trimming
                    new_tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages)
                    self.log_info(f"Cleanup complete: {current_tokens} → {new_tokens} tokens ({trimmed} messages removed)")
                    
                    # Always update database to maintain sync
                    if self.db:
                        # Clear and rebuild cache with current state
                        # This ensures DB reflects summarizations and removals
                        self.db.clear_thread_messages(thread_key)
                        for msg in thread_state.messages:
                            self.db.cache_message(
                                thread_key, 
                                msg.get("role"), 
                                msg.get("content"),
                                message_ts=None,  # Timestamp not needed for cache rebuild
                                metadata=msg.get("metadata")
                            )
                        self.log_debug(f"Database cache rebuilt for thread {thread_key} with {len(thread_state.messages)} messages")
                else:
                    self.log_warning("Cleanup triggered but no trimmable messages found")
            
        except Exception as e:
            self.log_error(f"Error during async cleanup: {e}")
            # Don't let cleanup errors affect the main flow
    
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
        self.log_info("")
        self.log_info("="*100)
        self.log_info(f"REQUEST START | Thread: {thread_key} | User: {username}")
        self.log_info(f"Message: {message.text[:100] if message.text else 'No text'}{'...' if message.text and len(message.text) > 100 else ''}")
        self.log_info("="*100)
        self.log_info("")
        
        request_start_time = time.time()
        
        # Check if thread is busy
        if not self.thread_manager.acquire_thread_lock(
            message.thread_id, 
            message.channel_id,
            timeout=0  # Don't wait, return immediately if busy
        ):
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
            thread_state = self._get_or_rebuild_thread_state(
                message,
                client,
                thinking_id
            )
            
            # Check if this thread had a previous timeout
            if hasattr(thread_state, 'had_timeout') and thread_state.had_timeout:
                # Send timeout notification to user
                timeout_msg = f"⚠️ Your previous request timed out after {int(config.api_timeout_read)} seconds. Please try again."
                client.post_message(
                    channel_id=message.channel_id,
                    text=timeout_msg,
                    thread_ts=message.thread_id
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
            image_inputs, document_inputs, unsupported_files = self._process_attachments(message, client, thinking_id)
            
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
                    trimmed_count = self._smart_trim_with_summarization(thread_state)
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
                intent = self.openai_client.classify_intent(
                    trimmed_history_for_intent,  # Documents truncated for classification
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
                    # For intent classification, include the current message in trimming
                    # Temporarily add the current message
                    temp_intent_msg = {"role": "user", "content": enhanced_text}
                    thread_state.messages.append(temp_intent_msg)
                    
                    # Pre-trim messages INCLUDING the current message
                    trimmed_messages = self._pre_trim_messages_for_api(thread_state.messages, model=thread_state.current_model, thread_state=thread_state)
                    
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
                    intent = self.openai_client.classify_intent(
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
                trimmed_messages = self._pre_trim_messages_for_api(thread_state.messages, model=thread_state.current_model, thread_state=thread_state)
                
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
                intent = self.openai_client.classify_intent(
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
                response = self._handle_image_generation(message.text, thread_state, client, message.channel_id, thinking_id, message)
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
                        attachment_urls,
                        message
                    )
                else:
                    # Try to find and edit recent image
                    response = self._handle_image_modification(
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
                        response = self._handle_text_response(user_content, thread_state, client, message, thinking_id)
                    elif image_inputs and document_inputs:
                        # Both images and documents - use two-call approach
                        total_files = len(image_inputs) + len(document_inputs)
                        status_msg = f"Analyzing {len(image_inputs)} image{'s' if len(image_inputs) > 1 else ''} and {len(document_inputs)} document{'s' if len(document_inputs) > 1 else ''}..."
                        self._update_status(client, message.channel_id, thinking_id, status_msg, emoji=config.analyze_emoji)
                        
                        # Use new two-call approach for mixed content
                        response = self._handle_mixed_content_analysis(
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
                        response = self._handle_vision_analysis(message.text, image_inputs, thread_state, message.attachments, 
                                                               client, message.channel_id, thinking_id, message)
                else:
                    # Vision-related question but no images or documents - try to find previous images
                    self.log_debug("Vision intent detected but no files attached - searching for previous images")
                    response = self._handle_vision_without_upload(
                        message.text, 
                        thread_state, 
                        client, 
                        message.channel_id, 
                        thinking_id,
                        message
                    )
            else:
                response = self._handle_text_response(user_content, thread_state, client, message, thinking_id)
            
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
                    client.post_message(
                        channel_id=message.channel_id,
                        text=warning_msg,
                        thread_ts=message.thread_id
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
            
        except Exception as e:
            self.log_error(f"Error processing message: {e}", exc_info=True)
            elapsed = time.time() - request_start_time
            # Try to get token count even on error
            try:
                error_tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages) if 'thread_state' in locals() else 0
                token_info = f" | Tokens: {error_tokens}" if error_tokens > 0 else ""
            except:
                token_info = ""
            
            self.log_info("")
            self.log_info("="*100)
            self.log_info(f"REQUEST END | Thread: {thread_key} | Status: ERROR | Time: {elapsed:.2f}s{token_info}")
            self.log_info("="*100)
            self.log_info("")
            
            # Check if this is a timeout error
            error_str = str(e)
            error_type = type(e).__name__
            
            # Check for various timeout error types
            if any(timeout_indicator in error_str.lower() or timeout_indicator in error_type.lower() 
                   for timeout_indicator in ['timeout', 'readtimeout', 'connecttimeout', 'timeouterror']):
                # Timeout-specific error message
                error_message = (
                    "The request timed out while waiting for a response. "
                    "This can happen with complex requests or when the service is busy. "
                    "Please try again in a moment."
                )
                self.log_warning(f"Request timeout after {elapsed:.2f} seconds for thread {thread_key}")
            else:
                # Generic error message with details
                error_message = str(e)
            
            return Response(
                type="error",
                content=error_message
            )
        finally:
            self.thread_manager.release_thread_lock(
                message.thread_id,
                message.channel_id
            )
    
    def _inject_image_analyses(self, messages: List[Dict], thread_state) -> List[Dict]:
        """Inject stored image analyses into conversation for context"""
        if not self.db:
            return messages
            
        thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
        
        # Get all messages from DB with their timestamps
        cached_messages = self.db.get_cached_messages(thread_key)
        
        enhanced_messages = []
        
        # Match messages by position and content to get timestamps
        for i, msg in enumerate(messages):
            # Add the original message
            enhanced_messages.append(msg)
            
            # Only inject after user messages
            if msg.get("role") == "user":
                # Find corresponding cached message to get timestamp
                # Match by position and content (messages are in order)
                if i < len(cached_messages):
                    cached_msg = cached_messages[i]
                    msg_ts = cached_msg.get("message_ts")
                    
                    if msg_ts:
                        # Get images associated with this specific message
                        images_for_message = self.db.get_images_by_message(thread_key, msg_ts)
                        
                        for img_data in images_for_message:
                            analysis = img_data.get("analysis")
                            url = img_data.get("url")
                            image_type = img_data.get("image_type", "image")
                            
                            # Inject image context - either analysis or just URL info
                            if analysis:
                                # Full analysis available
                                enhanced_messages.append({
                                    "role": "developer",
                                    "content": f"[Visual context for {image_type}]:\n{analysis}\n[End of visual context]"
                                })
                                self.log_debug(f"Injected analysis for message at position {i}")
                            elif url:
                                # No analysis but we have the URL - inject basic info
                                context_msg = f"[Image context: {image_type} at {url}]"
                                if image_type == "generated":
                                    context_msg = f"[Bot generated an image and posted it at: {url}]"
                                elif image_type == "uploaded":
                                    context_msg = f"[User uploaded an image at: {url}]"
                                elif image_type == "edited":
                                    context_msg = f"[Bot edited an image and posted it at: {url}]"
                                    
                                enhanced_messages.append({
                                    "role": "developer",
                                    "content": context_msg
                                })
                                self.log_debug(f"Injected URL context for {image_type} at position {i}")
        
        if len(enhanced_messages) > len(messages):
            self.log_info(f"Enhanced conversation with {len(enhanced_messages) - len(messages)} image context entries")
        
        return enhanced_messages
    def _is_error_or_busy_response(self, message_text: str) -> bool:
        """Check if a message is an error or busy response using consistent markers"""
        if not message_text:
            return False
            
        # Use the consistent error emoji marker
        if ":warning:" in message_text:
            return True
            
        # Also check config in case emoji was customized
        if config.error_emoji in message_text:
            return True
            
        return False
    
    def _get_or_rebuild_thread_state(
        self,
        message: Message,
        client: BaseClient,
        thinking_id: Optional[str] = None
    ) -> Any:
        """Get existing thread state or rebuild from platform history"""
        thread_state = self.thread_manager.get_or_create_thread(
            message.thread_id,
            message.channel_id
        )
        
        # If thread has no messages, rebuild from platform
        # Also rebuild if we have messages but no images in DB (to extract image URLs)
        should_rebuild = not thread_state.messages
        if not should_rebuild and self.db:
            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
            db_images = self.db.find_thread_images(thread_key)
            if not db_images and thread_state.messages:
                # We have messages but no images - check if there should be images
                for msg in thread_state.messages:
                    if msg.get("role") == "assistant":
                        metadata = msg.get("metadata", {})
                        if metadata.get("type") in ["image_generation", "image_edit"]:
                            should_rebuild = True
                            self.log_info("Found image generation messages without DB images - rebuilding to extract URLs")
                            break
        
        if should_rebuild:
            self.log_info(f"Checking thread history for {message.thread_id}")
            
            # Get history from platform first to see if there's anything to rebuild
            history = client.get_thread_history(
                message.channel_id,
                message.thread_id
            )
            
            # Only show rebuilding status if there's actual history (excluding current message)
            current_ts = message.metadata.get("ts")
            has_history = any(msg.metadata.get("ts") != current_ts for msg in history)
            
            if has_history:
                # Update status to show we're rebuilding
                if thinking_id:
                    self._update_status(
                        client, 
                        message.channel_id, 
                        thinking_id,
                        "Rebuilding thread history from Slack...",
                        emoji=config.circle_loader_emoji
                    )
                self.log_info(f"Rebuilding thread state for {message.thread_id} with {len(history)} messages")
            
            # Track pending image URLs for vision analysis association
            pending_image_urls = []
            pending_image_metadata = {}  # Store additional metadata per URL
            
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
                
                # For user messages, prefix with username
                if not is_bot:
                    # Get username from metadata (should be populated by client)
                    username = hist_msg.metadata.get("username") if hist_msg.metadata else None
                    
                    # If no username in metadata, fetch it from user_id
                    if not username and hist_msg.user_id:
                        # Use client's get_username method if available
                        if hasattr(client, 'get_username'):
                            # Get the slack_client from metadata if available
                            slack_client = hist_msg.metadata.get("slack_client") if hist_msg.metadata else None
                            if slack_client:
                                username = client.get_username(hist_msg.user_id, slack_client)
                            else:
                                # Try without slack_client
                                username = hist_msg.user_id  # Fallback to user_id
                        else:
                            username = hist_msg.user_id  # Fallback to user_id
                    
                    # Default to "User" if still no username
                    if not username:
                        username = "User"
                    
                    # Format content with username
                    content = f"{username}: {content}" if content else f"{username}:"
                
                # Track message metadata for preservation
                message_metadata = {}
                
                # Store bot image metadata in DB
                if is_bot and hist_msg.attachments:
                    for attachment in hist_msg.attachments:
                        if attachment.get("type") == "image":
                            url = attachment.get("url")
                            if url:
                                # Mark this message as containing an image for preservation
                                message_metadata["type"] = "image_generation"
                                message_metadata["url"] = url
                                
                                if self.db:
                                    # Determine image type from content
                                    image_type = "generated" if "Generated image:" in content else "edited" if "Edited image:" in content else "assistant"
                                    try:
                                        self.db.save_image_metadata(
                                            thread_id=f"{thread_state.channel_id}:{thread_state.thread_ts}",
                                            url=url,
                                            image_type=image_type,
                                            prompt=content,  # Store the generation/edit prompt
                                            analysis=None,
                                            metadata={"file_id": attachment.get("id")},
                                            message_ts=hist_msg.metadata.get("ts") if hist_msg.metadata else None
                                        )
                                    except Exception as e:
                                        self.log_warning(f"Failed to save bot image metadata: {e}")
                                break  # Only process first image
                
                # Store attachment metadata in DB instead of content
                if not is_bot and hist_msg.attachments:
                    # Track image URLs for potential vision analysis association
                    for attachment in hist_msg.attachments:
                        att_type = attachment.get("type")
                        att_url = attachment.get("url")
                        
                        # Handle both images and documents (files)
                        if att_url and (att_type == "image" or att_type == "file"):
                            # Check if this is actually a document based on mimetype
                            mimetype = attachment.get("mimetype", "")
                            filename = attachment.get("name", "")
                            
                            # Determine if it's an image or document
                            is_image = att_type == "image" or mimetype.startswith("image/")
                            is_document = (att_type == "file" and 
                                         self.document_handler and 
                                         self.document_handler.is_document_file(filename, mimetype))
                            
                            if is_image:
                                # Mark user messages with uploaded images
                                message_metadata["type"] = "image_upload"
                                message_metadata["url"] = att_url
                                
                                # Add to pending for vision analysis association
                                pending_image_urls.append(att_url)
                                pending_image_metadata[att_url] = {
                                    "file_id": attachment.get("id"),
                                    "message_ts": hist_msg.metadata.get("ts") if hist_msg.metadata else None,
                                    "user_text": hist_msg.text
                                }
                                
                                if self.db:
                                    try:
                                        self.db.save_image_metadata(
                                            thread_id=f"{thread_state.channel_id}:{thread_state.thread_ts}",
                                            url=att_url,
                                            image_type="uploaded",
                                            prompt=None,
                                            analysis=None,  # Will be updated when we find the vision analysis
                                            metadata={"file_id": attachment.get("id")},
                                            message_ts=hist_msg.metadata.get("ts") if hist_msg.metadata else None
                                        )
                                    except Exception as e:
                                        self.log_warning(f"Failed to save image metadata during rebuild: {e}")
                            
                            elif is_document:
                                # Handle document attachments during rebuild
                                self.log_info(f"Found document attachment during rebuild: {filename} ({mimetype})")
                                
                                # Store document metadata (but don't mark as preserved type)
                                # Documents should be trimmable/summarizable
                                message_metadata["filename"] = filename
                                message_metadata["mimetype"] = mimetype
                                
                                # Try to download and process the document
                                try:
                                    # Download the document using the client
                                    document_data = client.download_file(att_url, attachment.get("id"))
                                    
                                    if document_data and self.document_handler:
                                        # Extract document content
                                        extracted_content = self.document_handler.safe_extract_content(
                                            document_data, mimetype, filename
                                        )
                                        
                                        if extracted_content and extracted_content.get("content"):
                                            # Add document content to the message
                                            doc_content = self._build_message_with_documents(
                                                content,  # Original message content
                                                [{
                                                    "filename": filename,
                                                    "mimetype": mimetype,
                                                    "content": extracted_content["content"],
                                                    "metadata": extracted_content
                                                }]
                                            )
                                            content = doc_content
                                            self.log_info(f"Successfully extracted document content during rebuild: {filename}")
                                            
                                            # Store in document ledger
                                            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
                                            document_ledger = self.thread_manager.get_or_create_document_ledger(thread_state.thread_ts)
                                            document_ledger.add_document(
                                                content=extracted_content["content"],
                                                filename=filename,
                                                mime_type=mimetype,
                                                metadata=extracted_content
                                            )
                                        else:
                                            self.log_warning(f"Failed to extract content from document during rebuild: {filename}")
                                    else:
                                        self.log_warning(f"Failed to download document during rebuild: {filename}")
                                        
                                except Exception as e:
                                    self.log_error(f"Error processing document during rebuild: {e}")
                                    # Continue without the document content
                
                # Check if this is an assistant message that might be a vision analysis
                if is_bot and pending_image_urls:
                    self.log_debug(f"Assistant message with {len(pending_image_urls)} pending images")
                    self.log_debug(f"Content preview: {content[:100]}...")
                    
                    # Check if this is an error/busy response
                    if self._is_error_or_busy_response(content):
                        # Error response - clear pending as analysis failed
                        self.log_debug(f"Found error/busy response, clearing {len(pending_image_urls)} pending images")
                        pending_image_urls.clear()
                        pending_image_metadata.clear()
                    else:
                        # This is likely the vision analysis for the pending images
                        self.log_info(f"Found vision analysis for {len(pending_image_urls)} images")
                        self.log_debug(f"Pending URLs: {pending_image_urls}")
                        
                        # Store the analysis for all pending images
                        if self.db:
                            for image_url in pending_image_urls:
                                try:
                                    # Update the existing image metadata with the analysis
                                    self.log_debug(f"Storing analysis for: {image_url}")
                                    self.db.save_image_metadata(
                                        thread_id=f"{thread_state.channel_id}:{thread_state.thread_ts}",
                                        url=image_url,
                                        image_type="uploaded",
                                        prompt=pending_image_metadata.get(image_url, {}).get("user_text"),
                                        analysis=content,  # Store the full bot response as analysis
                                        metadata={
                                            "file_id": pending_image_metadata.get(image_url, {}).get("file_id"),
                                            "has_analysis": True
                                        },
                                        message_ts=pending_image_metadata.get(image_url, {}).get("message_ts")
                                    )
                                    self.log_info(f"Successfully stored vision analysis for image: {image_url[:60]}...")
                                except Exception as e:
                                    self.log_error(f"Failed to store vision analysis: {e}", exc_info=True)
                        else:
                            self.log_warning("No database available to store vision analysis")
                        
                        # Clear pending images after storing
                        pending_image_urls.clear()
                        pending_image_metadata.clear()
                
                thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
                message_ts = hist_msg.metadata.get("ts") if hist_msg.metadata else None
                # During rebuild, skip auto-trim to allow smart trimming with document summarization later
                self._add_message_with_token_management(thread_state, role, content, db=self.db, thread_key=thread_key, message_ts=message_ts, metadata=message_metadata, skip_auto_trim=True)
            
            self.log_info(f"Rebuilt thread with {len(thread_state.messages)} messages")
        
        # Apply smart trimming recursively if needed after rebuild
        model = thread_state.current_model or config.gpt_model
        max_tokens = config.get_model_token_limit(model)
        current_tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages)
        
        if current_tokens > max_tokens:
            self.log_info(f"Thread rebuilt over limit ({current_tokens}/{max_tokens} tokens), applying smart trim")
            
            # Update status to show we're trimming
            if thinking_id:
                self._update_status(
                    client,
                    message.channel_id,
                    thinking_id,
                    f"Optimizing conversation history ({current_tokens:,}/{max_tokens:,} tokens)...",
                    emoji=config.thinking_emoji
                )
            
            total_trimmed = 0
            
            # Keep trimming until we're under the limit
            while current_tokens > max_tokens:
                trimmed_count = self._smart_trim_with_summarization(thread_state)
                total_trimmed += trimmed_count
                
                if trimmed_count == 0:
                    # No more messages to trim
                    self.log_warning(f"Cannot trim further during rebuild - still at {current_tokens} tokens")
                    break
                
                # Recalculate tokens after trimming
                current_tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages)
                self.log_debug(f"After trimming {trimmed_count} messages during rebuild, now at {current_tokens}/{max_tokens} tokens")
            
            if total_trimmed > 0:
                self.log_info(f"Smart trim during rebuild complete: {total_trimmed} total messages processed, final: {current_tokens}/{max_tokens} tokens")
                
                # Update database with trimmed state
                if self.db:
                    thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
                    self.db.clear_thread_messages(thread_key)
                    for msg in thread_state.messages:
                        self.db.cache_message(
                            thread_id=thread_key,
                            role=msg.get("role"),
                            content=msg.get("content"),
                            metadata=msg.get("metadata"),
                            message_ts=msg.get("metadata", {}).get("ts")
                        )
                    self.log_info(f"Updated database with {len(thread_state.messages)} trimmed messages")
        
        # Log final token count
        final_tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages)
        self.log_info("="*100)
        self.log_info(f"THREAD STATE | Messages: {len(thread_state.messages)} | Tokens: {final_tokens}/{max_tokens}")
        self.log_info("="*100)
        
        return thread_state
    
    def _extract_slack_file_urls(self, text: str) -> List[str]:
        """Extract Slack file URLs from message text
        
        Args:
            text: Message text that may contain Slack file URLs
            
        Returns:
            List of Slack file URLs found
        """
        import re
        
        # Slack wraps URLs in angle brackets <URL>
        # Pattern to match Slack file URLs (both files.slack.com and workspace-specific URLs)
        # Examples:
        # - https://files.slack.com/files/...
        # - https://datassential.slack.com/files/...
        pattern = r'<(https?://(?:files\.slack\.com|[^/]+\.slack\.com/files)/[^>]+)>'
        
        urls = re.findall(pattern, text)
        
        # Also check for unwrapped Slack file URLs (but avoid capturing trailing >)
        pattern2 = r'(https?://(?:files\.slack\.com|[^/\s]+\.slack\.com/files)/[^\s>]+)'
        urls2 = re.findall(pattern2, text)
        
        # Combine and dedupe
        all_urls = list(set(urls + urls2))
        
        # Return ALL Slack file URLs, not just images
        # We'll determine the file type when processing them
        return all_urls
    def _process_attachments(
        self,
        message: Message,
        client: BaseClient,
        thinking_id: Optional[str] = None
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Process message attachments and extract images/documents from URLs in text
        
        Returns:
            Tuple of (image_inputs, document_inputs, unsupported_files)
        """
        image_inputs = []
        document_inputs = []
        unsupported_files = []
        image_count = 0
        max_images = 10
        processed_file_ids = set()  # Track processed file IDs to avoid duplicates
        
        # First, process regular attachments
        for attachment in message.attachments:
            file_type = attachment.get("type", "unknown")
            file_name = attachment.get("name", "unnamed file")
            
            if file_type == "image":
                # Stop if we've reached the image limit
                if image_count >= max_images:
                    self.log_warning(f"Limiting to {max_images} images (user uploaded more)")
                    continue
                    
                try:
                    # Track this file ID to avoid reprocessing
                    file_id = attachment.get("id")
                    if file_id:
                        processed_file_ids.add(file_id)
                    
                    # Download the image
                    image_data = client.download_file(
                        attachment.get("url"),
                        file_id
                    )
                    
                    if image_data:
                        # Convert to base64
                        base64_data = base64.b64encode(image_data).decode('utf-8')
                        
                        # Format for Responses API with base64
                        mimetype = attachment.get("mimetype", "image/png")
                        image_inputs.append({
                            "type": "input_image",
                            "image_url": f"data:{mimetype};base64,{base64_data}",
                            "source": "attachment",
                            "filename": file_name,
                            "url": attachment.get("url"),  # Keep URL for DB storage
                            "file_id": file_id
                        })
                        
                        # Store metadata in DB immediately
                        if self.db and attachment.get("url"):
                            thread_key = f"{message.channel_id}:{message.thread_id}"
                            try:
                                self.db.save_image_metadata(
                                    thread_id=thread_key,
                                    url=attachment.get("url"),
                                    image_type="uploaded",
                                    prompt=None,
                                    analysis=None,  # Will be added after vision analysis
                                    metadata={"file_id": file_id, "filename": file_name},
                                    message_ts=message.metadata.get("ts") if message.metadata else None
                                )
                                self.log_debug(f"Saved attachment metadata to DB: {file_name}")
                            except Exception as e:
                                self.log_warning(f"Failed to save attachment metadata: {e}")
                        
                        image_count += 1
                        self.log_debug(f"Processed image {image_count}/{max_images}: {file_name}")
                
                except Exception as e:
                    self.log_error(f"Error processing attachment: {e}")
            elif self.document_handler and self.document_handler.is_document_file(file_name, attachment.get("mimetype")):
                # Process document file
                mimetype = attachment.get("mimetype", "application/octet-stream")
                try:
                    # Track this file ID to avoid reprocessing
                    file_id = attachment.get("id")
                    if file_id:
                        processed_file_ids.add(file_id)
                    
                    # Update status to show we're processing the document
                    if thinking_id:
                        self._update_status(client, message.channel_id, thinking_id, 
                                          f"Processing {file_name}...", 
                                          emoji=config.analyze_emoji)
                    
                    # Download the document
                    document_data = client.download_file(
                        attachment.get("url"),
                        file_id
                    )
                    
                    if document_data:
                        # Update status to show we're extracting content
                        if thinking_id:
                            self._update_status(client, message.channel_id, thinking_id, 
                                              f"Extracting content from {file_name}...", 
                                              emoji=config.analyze_emoji)
                        
                        # Extract document content using DocumentHandler
                        extracted_content = self.document_handler.safe_extract_content(
                            document_data, mimetype, file_name
                        )
                        
                        if extracted_content and extracted_content.get("content"):
                            # Check if this is an image-based PDF that needs OCR
                            if extracted_content.get("is_image_based") and mimetype == "application/pdf":
                                self.log_info(f"PDF {file_name} appears to be image-based (scanned document)")
                                
                                # Check if we have page images for OCR
                                if extracted_content.get("page_images"):
                                    self.log_info(f"PDF has {len(extracted_content['page_images'])} page images for OCR")
                                    # Add page images to image_inputs for vision processing
                                    for page_img in extracted_content['page_images']:
                                        if image_count >= max_images:
                                            self.log_warning(f"Reached image limit, only processing first {image_count} PDF pages")
                                            break
                                        
                                        image_inputs.append({
                                            "type": "input_image",
                                            "image_url": f"data:{page_img['mimetype']};base64,{page_img['base64_data']}",
                                            "source": "pdf_page",
                                            "page_number": page_img['page'],
                                            "filename": file_name
                                        })
                                        image_count += 1
                                    
                                    # Update the document content to indicate OCR will be used
                                    extracted_content["content"] = (
                                        f"[PDF {file_name}: {extracted_content.get('total_pages', 'unknown')} pages total. "
                                        f"This appears to be a scanned document. "
                                        f"Using vision/OCR on {len(extracted_content['page_images'])} page(s) for text extraction.]"
                                    )
                                    extracted_content["ocr_processed"] = True
                                else:
                                    # No page images available
                                    extracted_content["warning"] = "This PDF appears to be a scanned document with minimal extractable text"
                            
                            document_inputs.append({
                                "filename": file_name,
                                "mimetype": mimetype,
                                "content": extracted_content["content"],
                                "page_structure": extracted_content.get("page_structure"),
                                "total_pages": extracted_content.get("total_pages"),
                                "summary": extracted_content.get("summary"),
                                "metadata": extracted_content.get("metadata", {}),
                                "url": attachment.get("url"),
                                "file_id": file_id,
                                "source": "attachment",
                                "is_image_based": extracted_content.get("is_image_based", False),
                                "requires_ocr": extracted_content.get("requires_ocr", False),
                                "ocr_processed": extracted_content.get("ocr_processed", False),
                                "warning": extracted_content.get("warning")
                            })
                            
                            # Store document in thread's DocumentLedger
                            thread_key = f"{message.channel_id}:{message.thread_id}"
                            document_ledger = self.thread_manager.get_or_create_document_ledger(message.thread_id)
                            document_ledger.add_document(
                                content=extracted_content["content"],
                                filename=file_name,
                                mime_type=mimetype,
                                page_structure=extracted_content.get("page_structure"),
                                total_pages=extracted_content.get("total_pages"),
                                summary=extracted_content.get("summary"),
                                metadata=extracted_content.get("metadata", {}),
                                db=self.db,
                                thread_id=thread_key,
                                message_ts=message.metadata.get("ts") if message.metadata else None
                            )
                            
                            if extracted_content.get("is_image_based"):
                                self.log_info(f"Processed image-based PDF: {file_name} ({extracted_content.get('total_pages', 'unknown')} pages)")
                            else:
                                self.log_info(f"Processed document: {file_name} ({extracted_content.get('total_pages', 'unknown')} pages)")
                        else:
                            self.log_warning(f"Failed to extract content from document: {file_name}")
                            # Update status to show extraction failed
                            if thinking_id:
                                error_msg = extracted_content.get("error", "Unable to extract content")
                                self._update_status(client, message.channel_id, thinking_id, 
                                                  f"⚠️ {file_name}: {error_msg}")
                            # Add to unsupported if extraction failed
                            unsupported_files.append({
                                "name": file_name,
                                "type": "file",
                                "mimetype": mimetype
                            })
                
                except Exception as e:
                    self.log_error(f"Error processing document attachment: {e}")
                    # Add to unsupported if processing failed
                    unsupported_files.append({
                        "name": file_name,
                        "type": "file",
                        "mimetype": mimetype
                    })
            else:
                # Track unsupported file types
                mimetype = attachment.get("mimetype", "unknown")
                unsupported_files.append({
                    "name": file_name,
                    "type": file_type,
                    "mimetype": mimetype
                })
                self.log_debug(f"Unsupported file type: {file_type} ({mimetype}) - {file_name}")
        
        # Second, check for image URLs in the message text
        if message.text and image_count < max_images:
            # First check for Slack file URLs and handle them specially
            slack_file_urls = self._extract_slack_file_urls(message.text)
            
            if slack_file_urls and hasattr(client, '__class__') and client.__class__.__name__ == 'SlackBot':
                self.log_debug(f"Found {len(slack_file_urls)} Slack file URL(s) to process")
                
                for url in slack_file_urls:
                    # Extract file ID from URL to check if already processed
                    file_id = None
                    if hasattr(client, 'extract_file_id_from_url'):
                        file_id = client.extract_file_id_from_url(url)
                    
                    # Skip if we already processed this file as an attachment
                    if file_id and file_id in processed_file_ids:
                        self.log_debug(f"Skipping duplicate Slack file {file_id} from URL")
                        continue
                    
                    # Determine file type from URL
                    url_lower = url.lower()
                    is_pdf = '.pdf' in url_lower
                    is_doc = any(ext in url_lower for ext in ['.docx', '.doc', '.xlsx', '.xls', '.csv', '.txt'])
                    is_image = any(ext in url_lower for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', 'image'])
                    
                    # Download the Slack file using the client's download_file method
                    self.log_info(f"Downloading Slack file from URL: {url}")
                    file_data = client.download_file(url)
                    
                    if file_data:
                        if is_pdf or is_doc:
                            # Process as document
                            # Extract filename from URL
                            import re
                            filename_match = re.search(r'/([^/]+\.(pdf|docx?|xlsx?|csv|txt))(\?|$)', url, re.IGNORECASE)
                            file_name = filename_match.group(1) if filename_match else "document"
                            
                            # Determine mimetype
                            if is_pdf:
                                mimetype = "application/pdf"
                            elif '.docx' in url_lower:
                                mimetype = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            elif '.doc' in url_lower:
                                mimetype = "application/msword"
                            elif '.xlsx' in url_lower:
                                mimetype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            elif '.xls' in url_lower:
                                mimetype = "application/vnd.ms-excel"
                            elif '.csv' in url_lower:
                                mimetype = "text/csv"
                            else:
                                mimetype = "text/plain"
                            
                            # Process document
                            if self.document_handler:
                                self.log_info(f"Processing Slack file URL as document: {file_name}")
                                
                                # Update status
                                if thinking_id:
                                    self._update_status(client, message.channel_id, thinking_id,
                                                      f"Extracting content from {file_name}...",
                                                      emoji=config.analyze_emoji)
                                
                                # Extract content
                                extracted_content = self.document_handler.safe_extract_content(
                                    file_data, mimetype, file_name
                                )
                                
                                if extracted_content and extracted_content.get("content"):
                                    # Check if this is an image-based PDF
                                    if extracted_content.get("is_image_based") and mimetype == "application/pdf":
                                        self.log_info(f"PDF {file_name} from URL appears to be image-based")
                                        
                                        # Check if we have page images for OCR
                                        if extracted_content.get("page_images"):
                                            self.log_info(f"PDF from URL has {len(extracted_content['page_images'])} page images for OCR")
                                            # Add page images to image_inputs for vision processing
                                            for page_img in extracted_content['page_images']:
                                                if image_count >= max_images:
                                                    self.log_warning(f"Reached image limit, only processing first {image_count} PDF pages")
                                                    break
                                                
                                                image_inputs.append({
                                                    "type": "input_image",
                                                    "image_url": f"data:{page_img['mimetype']};base64,{page_img['base64_data']}",
                                                    "source": "pdf_page_url",
                                                    "page_number": page_img['page'],
                                                    "filename": file_name
                                                })
                                                image_count += 1
                                            
                                            # Update content to indicate OCR will be used
                                            extracted_content["content"] = (
                                                f"[PDF {file_name} from URL: {extracted_content.get('total_pages', 'unknown')} pages. "
                                                f"Scanned document - using vision/OCR on {len(extracted_content['page_images'])} page(s).]"
                                            )
                                            extracted_content["ocr_processed"] = True
                                        else:
                                            extracted_content["warning"] = "This PDF appears to be a scanned document"
                                    
                                    document_inputs.append({
                                        "filename": file_name,
                                        "mimetype": mimetype,
                                        "content": extracted_content["content"],
                                        "page_structure": extracted_content.get("page_structure"),
                                        "total_pages": extracted_content.get("total_pages"),
                                        "url": url,
                                        "file_id": file_id,
                                        "source": "slack_url",
                                        "is_image_based": extracted_content.get("is_image_based", False),
                                        "requires_ocr": extracted_content.get("requires_ocr", False),
                                        "ocr_processed": extracted_content.get("ocr_processed", False),
                                        "warning": extracted_content.get("warning")
                                    })
                                    self.log_info(f"Successfully processed document from Slack URL: {file_name}")
                                else:
                                    self.log_warning(f"Failed to extract content from Slack file URL: {url}")
                                    unsupported_files.append({
                                        "name": file_name,
                                        "type": "document",
                                        "mimetype": mimetype,
                                        "error": "Content extraction failed"
                                    })
                            else:
                                self.log_warning("Document handler not available for Slack file URL")
                        elif is_image and image_count < max_images:
                            # Process as image
                            # Convert to base64
                            base64_data = base64.b64encode(file_data).decode('utf-8')
                            
                            # Determine mimetype from URL or default to PNG
                            mimetype = "image/png"
                            if '.jpg' in url_lower or '.jpeg' in url_lower:
                                mimetype = "image/jpeg"
                            elif '.gif' in url_lower:
                                mimetype = "image/gif"
                            elif '.webp' in url_lower:
                                mimetype = "image/webp"
                            
                            image_inputs.append({
                                "type": "input_image",
                                "image_url": f"data:{mimetype};base64,{base64_data}",
                                "source": "slack_url",
                                "original_url": url
                            })
                            
                            image_count += 1
                            self.log_info(f"Added Slack file image {image_count}/{max_images}: {url}")
                        else:
                            self.log_warning(f"Unknown file type or image limit reached for Slack URL: {url}")
                    else:
                        self.log_warning(f"Failed to download Slack file from URL: {url}")
            
            # Now check for external image URLs (excluding already-processed Slack URLs)
            # Create a modified text with Slack URLs removed to avoid double-processing
            text_for_url_processing = message.text
            for slack_url in slack_file_urls:
                text_for_url_processing = text_for_url_processing.replace(slack_url, "")
                # Also remove angle bracket wrapped versions
                text_for_url_processing = text_for_url_processing.replace(f"<{slack_url}>", "")
            
            # Get Slack token if this is a Slack client (for non-Slack URLs that might need auth)
            auth_token = None
            if hasattr(client, '__class__') and client.__class__.__name__ == 'SlackBot':
                auth_token = config.slack_bot_token
            
            downloaded_images, failed_urls = self.image_url_handler.process_urls_from_text(text_for_url_processing, auth_token)
            
            for img_data in downloaded_images:
                if image_count >= max_images:
                    self.log_warning(f"Limiting to {max_images} images (found more URLs)")
                    break
                
                # Format for Responses API
                image_inputs.append({
                    "type": "input_image",
                    "image_url": f"data:{img_data['mimetype']};base64,{img_data['base64_data']}",
                    "source": "url",
                    "original_url": img_data['url']
                })
                
                image_count += 1
                self.log_info(f"Added image from URL {image_count}/{max_images}: {img_data['url']}")
                
                # Store the image data for potential upload to Slack/Discord later
                # This will be handled by the AssetLedger tracking
                if hasattr(message, 'url_images'):
                    message.url_images.append(img_data)
                else:
                    message.url_images = [img_data]
            
            if failed_urls:
                self.log_warning(f"Failed to download images from URLs: {', '.join(failed_urls)}")
        
        return image_inputs, document_inputs, unsupported_files
    
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
    
    def _build_message_with_documents(self, text: str, document_inputs: List[Dict]) -> str:
        """Format documents with page/sheet structure for OpenAI context
        
        Args:
            text: Original user message text
            document_inputs: List of processed document dictionaries
            
        Returns:
            Formatted message text with document content and boundaries
        """
        if not document_inputs:
            return text
            
        # Ensure text is a string
        if not isinstance(text, str):
            self.log_warning(f"text parameter is not a string: {type(text)}")
            text = str(text) if text else ""
            
        # Start with the original message
        message_parts = [text] if text and text.strip() else []
        
        # Add document boundaries and content
        for doc in document_inputs:
            filename = doc.get("filename", "unknown_document")
            mimetype = doc.get("mimetype", "unknown")
            content = doc.get("content")
            # Ensure content is never None
            if content is None:
                content = "[Document content not available]"
            elif not content:
                content = "[Empty document]"
            page_structure = doc.get("page_structure")
            total_pages = doc.get("total_pages")
            
            # Build document header
            doc_header = f"\n\n=== DOCUMENT: {filename} ==="
            if total_pages:
                doc_header += f" ({total_pages} pages)"
            doc_header += f"\nMIME Type: {mimetype}\n"
            
            # Add page/sheet structure info if available
            if page_structure:
                if isinstance(page_structure, dict):
                    if "sheets" in page_structure:
                        # Excel/CSV with multiple sheets
                        sheet_names = list(page_structure["sheets"].keys())
                        doc_header += f"Sheets: {', '.join(sheet_names[:5])}"  # Limit displayed sheet names
                        if len(sheet_names) > 5:
                            doc_header += f" (and {len(sheet_names) - 5} more)"
                        doc_header += "\n"
                    elif "pages" in page_structure:
                        # PDF with page info
                        doc_header += f"Pages: {len(page_structure['pages'])}\n"
            
            doc_header += "=== CONTENT START ===\n"
            
            # Add the document content (ensure all parts are strings)
            if not isinstance(doc_header, str):
                self.log_warning(f"doc_header is not a string: {type(doc_header)}")
                doc_header = str(doc_header)
            if not isinstance(content, str):
                self.log_warning(f"content is not a string: {type(content)}")
                content = str(content)
                
            message_parts.append(doc_header)
            message_parts.append(content)
            message_parts.append(f"\n=== DOCUMENT END: {filename} ===")
        
        # Ensure all parts are strings before joining
        str_parts = []
        for i, part in enumerate(message_parts):
            if not isinstance(part, str):
                self.log_warning(f"message_parts[{i}] is not a string: {type(part)}")
                str_parts.append(str(part))
            else:
                str_parts.append(part)
        
        return "\n".join(str_parts)
    
    def _extract_image_registry(self, thread_state) -> List[Dict[str, str]]:
        """Extract all image URLs and descriptions from thread state"""
        image_registry = []
        
        for msg in thread_state.messages:
            if msg.get("role") == "assistant":
                metadata = msg.get("metadata", {})
                content = msg.get("content", "")
                
                # First check metadata (new approach)
                if metadata.get("type") in ["image_generation", "image_edit", "image_analysis"]:
                    url = metadata.get("url")
                    prompt = metadata.get("prompt", content)
                    image_type = metadata.get("type", "").replace("_", " ")
                    
                    image_registry.append({
                        "url": url if url else "[Pending upload]",
                        "description": prompt,
                        "type": image_type
                    })
                # Fallback to string matching for backward compatibility
                elif isinstance(content, str):
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
        # First check the database for ALL images in this thread (no limit)
        if self.db:
            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
            thread_images = self.db.find_thread_images(thread_key)
            if thread_images:
                self.log_debug(f"Found {len(thread_images)} images in DB for thread {thread_key}")
                return True
        
        # Fallback: Check last few messages for image generation breadcrumbs or uploaded images
        for msg in thread_state.messages[-5:]:  # Check last 5 messages
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Check metadata for image generation
                metadata = msg.get("metadata", {})
                if metadata.get("type") == "image_generation":
                    return True
                # Fallback to text markers
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
    
    def _get_system_prompt(self, client: BaseClient, user_timezone: str = "UTC", 
                          user_tz_label: Optional[str] = None, user_real_name: Optional[str] = None,
                          user_email: Optional[str] = None, model: Optional[str] = None,
                          web_search_enabled: bool = True, has_trimmed_messages: bool = False,
                          custom_instructions: Optional[str] = None) -> str:
        """Get the appropriate system prompt based on the client platform with user's timezone, name, email, model, web search capability, trimming status, and custom instructions"""
        client_name = client.name.lower()
        
        # Get base prompt for the platform
        if "slack" in client_name:
            base_prompt = SLACK_SYSTEM_PROMPT
            
            # Add company info for Slack if configured
            company_name = os.getenv("SLACK_COMPANY_NAME", "").strip()
            company_website = os.getenv("SLACK_COMPANY_WEBSITE", "").strip()
            
            if company_name and company_website:
                base_prompt += f"\n\nThe company's name is {company_name}."
                base_prompt += f"\nThe company's website is {company_website}."
            elif company_name:
                base_prompt += f"\n\nThe company's name is {company_name}."
            elif company_website:
                base_prompt += f"\n\nThe company's website is {company_website}."
        elif "discord" in client_name:
            base_prompt = DISCORD_SYSTEM_PROMPT
        else:
            # Default/CLI prompt
            base_prompt = CLI_SYSTEM_PROMPT
        
        # Get current time in user's timezone
        try:
            user_tz = pytz.timezone(user_timezone)
            current_time = datetime.datetime.now(pytz.UTC).astimezone(user_tz)
            
            # Use abbreviated timezone label if available (EST, PST, etc.), otherwise full name
            if user_tz_label:
                timezone_display = user_tz_label
            else:
                # Try to get the abbreviated name from the current time
                timezone_display = current_time.strftime('%Z')
                if not timezone_display or timezone_display == user_tz.zone:
                    # If strftime doesn't give us an abbreviation, use the full zone name
                    timezone_display = user_tz.zone
        except:
            # Fallback to UTC if timezone is invalid
            current_time = datetime.datetime.now(pytz.UTC)
            timezone_display = "UTC"
        
        # Format time and user context - emphasize "today's date" for clarity
        time_context = f"\n\nToday's date and current time: {current_time.strftime('%A, %B %d, %Y at %I:%M %p')} ({timezone_display})\nIMPORTANT: Always consider the current date and time (w/ timezone offset) and adjust your responses accordingly."
        
        # Add user's name and email if available
        user_context = ""
        if user_real_name and user_email:
            user_context = f"\nYou're speaking with {user_real_name} (email: {user_email})"
        elif user_real_name:
            user_context = f"\nYou're speaking with {user_real_name}"
        elif user_email:
            user_context = f"\nYou're speaking with user (email: {user_email})"
        
        # Add model and knowledge cutoff info
        model_context = ""
        if model:
            from config import MODEL_KNOWLEDGE_CUTOFFS
            cutoff_date = MODEL_KNOWLEDGE_CUTOFFS.get(model)
            if cutoff_date:
                model_context = f"\n\nYour current model is {model} and your knowledge cutoff is {cutoff_date}."
            else:
                # Fallback for unknown models
                model_context = f"\n\nYour current model is {model}."
        
        # Add web search capability context
        web_search_context = ""
        if web_search_enabled:
            web_search_context = "\n\nAdditional capability enabled: Web Search. You can search the web for current information when needed to provide up-to-date answers.  "
        else:
            # Get the settings command dynamically
            settings_command = config.settings_slash_command if hasattr(config, 'settings_slash_command') else '/chatgpt-settings'
            web_search_context = f"\n\nWeb search is currently disabled. If a user asks for current information or recent events beyond your knowledge cutoff, provide what you know but mention that web search is disabled in their user settings. They can enable it using `{settings_command}`."
        
        # Add trimming notification if messages have been removed
        trimming_context = ""
        if has_trimmed_messages:
            trimming_context = "\n\nNote: Some older messages have been removed from this conversation to manage context length."
        
        # Add custom instructions if provided
        custom_instructions_context = ""
        if custom_instructions:
            custom_instructions_context = f"\n\n--- USER CUSTOM INSTRUCTIONS ---\nThe following are custom instructions provided by the user. These should be followed and may supersede any conflicting default instructions (within legal and ethical boundaries):\n\n{custom_instructions}\n\n--- END OF USER CUSTOM INSTRUCTIONS ---"
        
        return base_prompt + time_context + user_context + model_context + web_search_context + trimming_context + custom_instructions_context
    
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
                          "Generating image. This may take a minute, please wait...",
                          emoji=config.circle_loader_emoji)
    def _handle_text_response(self, user_content: Any, thread_state, client: BaseClient, 
                              message: Message, thinking_id: Optional[str] = None,
                              attachment_urls: Optional[List[str]] = None) -> Response:
        """Handle text-only response generation"""
        # Get thread config (with user preferences)
        thread_config = config.get_thread_config(
            overrides=thread_state.config_overrides,
            user_id=message.user_id,
            db=self.db
        )
        
        # Check if streaming is enabled and supported (respecting user prefs)
        streaming_enabled = thread_config.get('enable_streaming', config.enable_streaming)
        if (hasattr(client, 'supports_streaming') and client.supports_streaming() and 
            streaming_enabled and thinking_id is not None):  # Streaming requires a message ID to update
            return self._handle_streaming_text_response(user_content, thread_state, client, message, thinking_id, attachment_urls)
        
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
            
            # Create clean text for thread history (no URLs or counts)
            breadcrumb_text = " ".join(text_parts).strip()
            
            # Add simplified breadcrumb to thread state (no base64 data)
            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
            message_ts = message.metadata.get("ts") if message.metadata else None
            self._add_message_with_token_management(thread_state, "user", breadcrumb_text, db=self.db, thread_key=thread_key, message_ts=message_ts)
            
            # Use the full content with images for the actual API call
            messages_for_api = thread_state.messages[:-1] + [{"role": "user", "content": user_content}]
        else:
            # Simple text content - add as-is
            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
            message_ts = message.metadata.get("ts") if message.metadata else None
            
            # Check if this content contains documents and add metadata
            message_metadata = None
            if isinstance(user_content, str) and "=== DOCUMENT:" in user_content:
                # Don't mark as document_upload type - documents should be trimmable
                message_metadata = {"contains_document": True}
            
            self._add_message_with_token_management(thread_state, "user", user_content, db=self.db, thread_key=thread_key, message_ts=message_ts, metadata=message_metadata)
            messages_for_api = thread_state.messages
        
        # Inject stored image analyses into the conversation for full context
        messages_for_api = self._inject_image_analyses(messages_for_api, thread_state)
        
        # Pre-trim messages to fit within context window
        messages_for_api = self._pre_trim_messages_for_api(messages_for_api, model=thread_state.current_model)
        
        # Get thread config (with user preferences)
        thread_config = config.get_thread_config(
            overrides=thread_state.config_overrides,
            user_id=message.user_id,
            db=self.db
        )
        
        # Use thread's system prompt (which is now platform-specific)
        # Always regenerate to get current time
        user_timezone = message.metadata.get("user_timezone", "UTC") if message.metadata else "UTC"
        user_tz_label = message.metadata.get("user_tz_label", None) if message.metadata else None
        user_real_name = message.metadata.get("user_real_name", None) if message.metadata else None
        user_email = message.metadata.get("user_email", None) if message.metadata else None
        # Pass the model for dynamic knowledge cutoff (respecting user prefs)
        web_search_enabled = thread_config.get('enable_web_search', config.enable_web_search)
        model = config.web_search_model or thread_config["model"] if web_search_enabled else thread_config["model"]
        system_prompt = self._get_system_prompt(client, user_timezone, user_tz_label, user_real_name, user_email, model, web_search_enabled, thread_state.has_trimmed_messages, thread_config.get('custom_instructions'))
        
        # Update status before generating
        self._update_status(client, message.channel_id, thinking_id, "Generating response...")
        
        # Check if web search should be available (respecting user prefs)
        web_search_enabled = thread_config.get('enable_web_search', config.enable_web_search)
        if web_search_enabled:
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
        if web_search_enabled:
            # Look for indicators that web search was used
            # OpenAI typically includes numbered citations [1], [2] or URLs when web search is used
            if any(marker in response_text for marker in ["[1]", "[2]", "[3]", "http://", "https://"]):
                self.log_info("Response includes web search results")
        
        # Add assistant response to thread state
        thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
        self._add_message_with_token_management(thread_state, "assistant", response_text, db=self.db, thread_key=thread_key)
        
        # Schedule async cleanup after response
        import threading
        cleanup_thread = threading.Thread(
            target=self._async_post_response_cleanup,
            args=(thread_state, thread_key),
            daemon=True
        )
        cleanup_thread.start()
        
        return Response(
            type="text",
            content=response_text
        )
    def _handle_streaming_text_response(self, user_content: Any, thread_state, client: BaseClient, 
                                      message: Message, thinking_id: Optional[str] = None,
                                      attachment_urls: Optional[List[str]] = None) -> Response:
        """Handle text-only response generation with streaming support"""
        # Check if client supports streaming
        if not hasattr(client, 'supports_streaming') or not client.supports_streaming():
            self.log_debug("Client doesn't support streaming, falling back to non-streaming")
            return self._handle_text_response(user_content, thread_state, client, message, thinking_id, attachment_urls)
        
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
            
            # Create clean text for thread history (no URLs or counts)
            breadcrumb_text = " ".join(text_parts).strip()
            
            # Add simplified breadcrumb to thread state (no base64 data)
            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
            message_ts = message.metadata.get("ts") if message.metadata else None
            self._add_message_with_token_management(thread_state, "user", breadcrumb_text, db=self.db, thread_key=thread_key, message_ts=message_ts)
            
            # Use the full content with images for the actual API call
            messages_for_api = thread_state.messages[:-1] + [{"role": "user", "content": user_content}]
        else:
            # Simple text content - add as-is
            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
            message_ts = message.metadata.get("ts") if message.metadata else None
            
            # Check if this content contains documents and add metadata
            message_metadata = None
            if isinstance(user_content, str) and "=== DOCUMENT:" in user_content:
                # Don't mark as document_upload type - documents should be trimmable
                message_metadata = {"contains_document": True}
            
            self._add_message_with_token_management(thread_state, "user", user_content, db=self.db, thread_key=thread_key, message_ts=message_ts, metadata=message_metadata)
            messages_for_api = thread_state.messages
        
        # Inject stored image analyses into the conversation for full context
        messages_for_api = self._inject_image_analyses(messages_for_api, thread_state)
        
        # Pre-trim messages to fit within context window
        messages_for_api = self._pre_trim_messages_for_api(messages_for_api, model=thread_state.current_model)
        
        # Get thread config (with user preferences)
        thread_config = config.get_thread_config(
            overrides=thread_state.config_overrides,
            user_id=message.user_id,
            db=self.db
        )
        
        # Use thread's system prompt (which is now platform-specific)
        # Always regenerate to get current time
        user_timezone = message.metadata.get("user_timezone", "UTC") if message.metadata else "UTC"
        user_tz_label = message.metadata.get("user_tz_label", None) if message.metadata else None
        user_real_name = message.metadata.get("user_real_name", None) if message.metadata else None
        user_email = message.metadata.get("user_email", None) if message.metadata else None
        # Pass the model for dynamic knowledge cutoff (respecting user prefs)
        web_search_enabled = thread_config.get('enable_web_search', config.enable_web_search)
        model = config.web_search_model or thread_config["model"] if web_search_enabled else thread_config["model"]
        system_prompt = self._get_system_prompt(client, user_timezone, user_tz_label, user_real_name, user_email, model, web_search_enabled, thread_state.has_trimmed_messages, thread_config.get('custom_instructions'))
        
        # Post an initial message to get the message ID for streaming updates
        # For streaming with potential tools, start with "Working on it" 
        # (will be overridden if tools are used)
        initial_message = f"{config.thinking_emoji} Working on it..."
        if thinking_id:
            # Update existing thinking message
            message_id = thinking_id
            client.update_message(message.channel_id, message_id, initial_message)
        else:
            # We need a way to post a message and get its ID - this would depend on client implementation
            self.log_warning("No thinking_id provided for streaming - falling back to non-streaming")
            return self._handle_text_response(user_content, thread_state, client, message, thinking_id, attachment_urls)
        
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
                        result = client.update_message_streaming(message.channel_id, message_id, status_msg)
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
                        result = client.update_message_streaming(message.channel_id, message_id, status_msg)
                        if result["success"]:
                            self.log_info(f"File search #{search_counts['file_search']} started - updated status")
                        else:
                            self.log_warning(f"Failed to update file search status: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        self.log_error(f"Error updating file search status: {e}")
                elif tool_type == "image_generation" and not tool_states["image_generation"]:
                    tool_states["image_generation"] = True
                    status_msg = f"{config.circle_loader_emoji} Generating image. This may take a minute..."
                    try:
                        result = client.update_message_streaming(message.channel_id, message_id, status_msg)
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
        
        # Track current streaming message and overflow
        current_message_id = message_id
        current_part = 1
        overflow_buffer = ""
        message_char_limit = 3700  # Leave room for indicators
        
        # Define the streaming callback
        def stream_callback(text_chunk: str):
            """Callback function called with each text chunk from OpenAI"""
            nonlocal current_message_id, current_part, overflow_buffer
            
            # Check if this is the completion signal (None)
            if text_chunk is None:
                # Stream is complete - flush any remaining buffered text WITHOUT loading indicator
                if buffer.has_pending_update() and rate_limiter.can_make_request():
                    self.log_info("Flushing final buffered text")
                    rate_limiter.record_request_attempt()
                    # Use raw text for final flush - no loading indicator since stream is complete
                    final_text = buffer.get_complete_text()  # No loading indicator on completion
                    try:
                        result = client.update_message_streaming(message.channel_id, current_message_id, final_text)
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
                
                # Check if we need to overflow based on RAW text (not display text)
                raw_text = buffer.get_complete_text()
                
                if len(raw_text) > message_char_limit:
                    # Find a good split point - preferably at a line break
                    split_point = message_char_limit
                    
                    # Look backwards for the last newline before the limit
                    last_newline = raw_text.rfind('\n', 0, message_char_limit)
                    
                    # If we found a newline within reasonable distance (not too far back)
                    # Use it as the split point to avoid breaking lines
                    if last_newline > message_char_limit - 500 and last_newline > 0:
                        split_point = last_newline + 1  # Include the newline in first part
                    else:
                        # No good newline found, try to at least avoid splitting words
                        # Look for last space before the limit
                        last_space = raw_text.rfind(' ', max(0, message_char_limit - 100), message_char_limit)
                        if last_space > 0:
                            split_point = last_space + 1
                    
                    # Split the RAW text at the chosen point
                    first_part_raw = raw_text[:split_point]
                    overflow_raw = raw_text[split_point:]
                    
                    # Check if we're splitting inside a code block
                    fence_handler_temp = FenceHandler()
                    fence_handler_temp.update_text(first_part_raw)
                    was_in_code_block = fence_handler_temp.is_in_code_block()
                    language_hint = fence_handler_temp.get_current_language_hint()
                    
                    # Get display-safe version of first part (with closed fences if needed)
                    first_part_display = fence_handler_temp.get_display_safe_text()
                    
                    # Update current message with continuation indicator
                    final_first_part = f"{first_part_display}\n\n*Continued in next message...*"
                    try:
                        result = client.update_message_streaming(message.channel_id, current_message_id, final_first_part)
                        if result["success"]:
                            # Prepare overflow text with proper fence opening if needed
                            if was_in_code_block:
                                # Re-open the code block on the new page
                                lang_str = language_hint if language_hint else ""
                                overflow_with_fence = f"```{lang_str}\n{overflow_raw}"
                            else:
                                overflow_with_fence = overflow_raw
                            
                            # Post a new message for overflow
                            current_part += 1
                            
                            # Create new fence handler for the continuation
                            fence_handler_continuation = FenceHandler()
                            fence_handler_continuation.update_text(overflow_with_fence)
                            continuation_display = fence_handler_continuation.get_display_safe_text()
                            
                            continuation_text = f"*Part {current_part} (continued)*\n\n{continuation_display} {config.loading_ellipse_emoji}"
                            
                            # Send new message and get its ID
                            new_msg_result = client.send_message_get_ts(message.channel_id, thinking_id, continuation_text)
                            if new_msg_result and "ts" in new_msg_result:
                                current_message_id = new_msg_result["ts"]
                                # Reset buffer with the properly fenced overflow content
                                buffer.reset()
                                buffer.add_chunk(overflow_with_fence)
                                buffer.mark_updated()
                                self.log_info(f"Created overflow message part {current_part}, reopened code block: {was_in_code_block}")
                    except Exception as e:
                        self.log_error(f"Error handling message overflow: {e}")
                else:
                    # Normal update - get display-safe text with closed fences
                    display_text = buffer.get_display_text()
                    display_text_with_indicator = f"{display_text} {config.loading_ellipse_emoji}"
                    
                    # Call client.update_message_streaming with indicator
                    try:
                        result = client.update_message_streaming(message.channel_id, current_message_id, display_text_with_indicator)
                        
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
                                    # Try to clear the thinking indicator
                                    try:
                                        clear_text = buffer.last_sent_text if buffer.last_sent_text else "Processing..."
                                        if len(clear_text) > 3900:
                                            clear_text = clear_text[:3800] + "\n\n*Response too long - see next message*"
                                        client.update_message_streaming(message.channel_id, message_id, clear_text)
                                    except Exception as clear_error:
                                        self.log_error(f"Failed to clear indicator after circuit break: {clear_error}")
                            else:
                                rate_limiter.record_failure(is_rate_limit=False)
                                self.log_warning(f"Message update failed: {result.get('error', 'Unknown error')}")
                            
                    except Exception as e:
                        rate_limiter.record_failure(is_rate_limit=False)
                        self.log_error(f"Error updating streaming message: {e}")
        
        # Start streaming from OpenAI with the callback
        try:
            web_search_enabled = thread_config.get('enable_web_search', config.enable_web_search)
            if web_search_enabled:
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
            # Note: current_message_id might be different from message_id if we overflowed
            # We need to update the current message (which might be part 2, 3, etc)
            if current_part > 1:
                # We're on an overflow message - just remove the loading indicator
                self.log_debug(f"Removing loading indicator from part {current_part}")
                try:
                    # Get the current display text without loading indicator
                    final_part_text = buffer.get_complete_text()
                    if final_part_text:
                        # Add the part indicator
                        final_part_text = f"*Part {current_part} (continued)*\n\n{final_part_text}"
                        final_result = client.update_message_streaming(message.channel_id, current_message_id, final_part_text)
                        if not final_result["success"]:
                            self.log_error(f"Failed to remove indicator from part {current_part}: {final_result.get('error', 'Unknown error')}")
                except Exception as e:
                    self.log_error(f"Error removing indicator from overflow message: {e}")
            else:
                # Original message - check if we need to handle any remaining text
                if response_text != buffer.last_sent_text or True:  # Always update to remove indicator
                    if response_text != buffer.last_sent_text:
                        self.log_warning(f"Text mismatch after streaming - sending correction update "
                                       f"(sent: {len(buffer.last_sent_text)}, should be: {len(response_text)} chars)")
                    else:
                        self.log_debug("Sending final update to ensure loading indicator is removed")
                    try:
                        # Handle empty response
                        if not response_text:
                            response_text = "I apologize, but I wasn't able to generate a response. Please try again."
                            self.log_warning("Empty response detected, using fallback message")
                        
                        # Check if message is too long for a single update
                        if len(response_text) > 3900:  # Slack's approximate limit
                            # This shouldn't happen if streaming overflow worked correctly
                            # But handle it as a fallback
                            truncated_text = response_text[:3800] + "\n\n*Continued in next message...*"
                            final_result = client.update_message_streaming(message.channel_id, message_id, truncated_text)
                            
                            # Send the rest as new messages
                            overflow_text = response_text[3800:]
                            client.send_message(message.channel_id, thinking_id, f"*...continued*\n\n{overflow_text}")
                            
                            if not final_result["success"]:
                                self.log_error(f"Final truncated update failed: {final_result.get('error', 'Unknown error')}")
                        else:
                            final_result = client.update_message_streaming(message.channel_id, current_message_id, response_text)
                            if not final_result["success"]:
                                self.log_error(f"Final correction update failed: {final_result.get('error', 'Unknown error')}")
                    except Exception as e:
                        self.log_error(f"Error in final correction update: {e}")
            
            # Note: To properly detect if web search was used, we'd need to track
            # tool events during streaming. The presence of URLs doesn't mean web search was used.
            
            # Add assistant response to thread state
            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
            self._add_message_with_token_management(thread_state, "assistant", response_text, db=self.db, thread_key=thread_key)
            
            # Schedule async cleanup after response
            import threading
            cleanup_thread = threading.Thread(
                target=self._async_post_response_cleanup,
                args=(thread_state, thread_key),
                daemon=True
            )
            cleanup_thread.start()
            
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
                    # Send whatever text we have without the loading indicator, or a formatted error message
                    if buffer.has_content():
                        error_text = buffer.get_complete_text()
                    else:
                        error_text = f"{config.error_emoji} *Streaming interrupted*\n\nThe response was interrupted. I'll try again without streaming..."
                    client.update_message_streaming(message.channel_id, message_id, error_text)
                except Exception as cleanup_error:
                    self.log_debug(f"Could not remove loading indicator: {cleanup_error}")
            
            # Fall back to non-streaming on error
            self.log_info("Falling back to non-streaming due to error")
            
            # Remove the message that was just added by streaming attempt
            # to prevent duplicates when fallback adds it again
            if thread_state.messages and thread_state.messages[-1].get("role") == "user":
                removed_msg = thread_state.messages.pop()
                self.log_debug("Removed duplicate user message before fallback")
            
            return self._handle_text_response(user_content, thread_state, client, message, thinking_id, attachment_urls)

    def _handle_vision_analysis(self, user_text: str, image_inputs: List[Dict], thread_state, attachments: List[Dict],
                               client: BaseClient, channel_id: str, thinking_id: Optional[str], message: Message) -> Response:
        """Handle vision analysis of uploaded images"""
        if not image_inputs:
            return Response(
                type="error",
                content="No images found to analyze"
            )
        
        # Extract base64 data from image inputs and track URL images
        images_to_analyze = []
        url_images = []  # Track URL-sourced images
        
        for img_input in image_inputs:
            if img_input.get("type") == "input_image":
                # Extract from data URL format
                image_url = img_input.get("image_url", "")
                if image_url.startswith("data:"):
                    parts = image_url.split(",", 1)
                    if len(parts) == 2:
                        _, base64_data = parts
                        images_to_analyze.append(base64_data)
                        
                        # Track URL-sourced images in AssetLedger (including Slack URLs)
                        if img_input.get("source") in ["url", "slack_url"]:
                            url_images.append({
                                "base64_data": base64_data,
                                "original_url": img_input.get("original_url", "")
                            })
        
        if not images_to_analyze:
            return Response(
                type="error", 
                content="Could not process uploaded images"
            )
        
        self.log_info(f"Analyzing {len(images_to_analyze)} image(s) with prompt: {user_text[:100]}...")
        
        # Single vision API call with both analysis context and user question
        analysis_result = None
        
        try:
            self.log_info("Starting vision analysis with user question")
            
            # Get platform system prompt for consistent personality/formatting
            user_timezone = message.metadata.get("user_timezone", "UTC") if message.metadata else "UTC"
            user_tz_label = message.metadata.get("user_tz_label", None) if message.metadata else None
            user_real_name = message.metadata.get("user_real_name", None) if message.metadata else None
            user_email = message.metadata.get("user_email", None) if message.metadata else None
            # Use thread config model for vision analysis (with user preferences)
            thread_config = config.get_thread_config(
                overrides=thread_state.config_overrides,
                user_id=message.user_id,
                db=self.db
            )
            web_search_enabled = thread_config.get('enable_web_search', config.enable_web_search)
            system_prompt = self._get_system_prompt(client, user_timezone, user_tz_label, user_real_name, user_email, thread_config["model"], web_search_enabled, thread_state.has_trimmed_messages, thread_config.get('custom_instructions'))
            
            # Use the user's question directly - it will be enhanced for natural conversation
            # If no text provided with image, let the model infer from full conversation context
            if not user_text:
                # Pass empty question - the full conversation history will provide context
                user_question = ""
                self.log_debug("No text with image - relying on conversation context")
            else:
                user_question = user_text
            
            # Inject stored image analyses for better context
            enhanced_messages = self._inject_image_analyses(thread_state.messages, thread_state)
            
            # Pre-trim messages to fit within context window
            enhanced_messages = self._pre_trim_messages_for_api(enhanced_messages, model=thread_state.current_model)
            
            # Update status to show we're preparing the analysis
            self._update_status(client, channel_id, thinking_id, "Preparing analysis...", emoji=config.analyze_emoji)
            
            # Extract filenames from attachments for context
            filenames = []
            if attachments and len(images_to_analyze) > 0:
                # Get filenames for images that were actually processed
                for i, img in enumerate(images_to_analyze[:len(attachments)]):
                    if i < len(attachments):
                        filenames.append(attachments[i].get("name", f"image{i+1}"))
            
            # Enhance the vision prompt with conversation context
            enhanced_question = user_question
            if user_question:  # Only enhance if there's actually a question
                enhanced_question = self.openai_client._enhance_vision_prompt(
                    user_question,
                    conversation_history=enhanced_messages  # Pass the full conversation context
                )
                self.log_debug(f"Enhanced vision prompt: {enhanced_question[:100]}...")
            
            # Add filename context to the enhanced question for the API
            if filenames:
                filename_context = f"[Image files: {', '.join(filenames)}]\n"
                enhanced_question = filename_context + enhanced_question
                self.log_debug(f"Added filename context for {len(filenames)} images")
            
            # Update status to show we're analyzing the image(s)
            if len(images_to_analyze) == 1:
                status_msg = "Analyzing your image..."
            else:
                status_msg = f"Analyzing {len(images_to_analyze)} images..."
            self._update_status(client, channel_id, thinking_id, status_msg, emoji=config.analyze_emoji)
            
            # Check if streaming is supported and enabled (respecting user prefs)
            streaming_enabled = thread_config.get('enable_streaming', config.enable_streaming)
            if (hasattr(client, 'supports_streaming') and client.supports_streaming() and 
                streaming_enabled and thinking_id is not None):
                # Stream the vision analysis response
                from streaming.buffer import StreamingBuffer
                from streaming import RateLimitManager
                
                streaming_config = client.get_streaming_config() if hasattr(client, 'get_streaming_config') else {}
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
                
                message_id = thinking_id  # Start with the thinking message
                
                def stream_callback(chunk: str):
                    nonlocal message_id
                    # Handle completion signal (None chunk)
                    if chunk is None:
                        # Final update without loading indicator
                        if message_id:
                            final_text = buffer.get_complete_text()
                            client.update_message_streaming(channel_id, message_id, final_text)
                        return
                    
                    buffer.add_chunk(chunk)
                    
                    # Only update if both buffer says it's time AND rate limiter allows it
                    if buffer.should_update() and rate_limiter.can_make_request():
                        rate_limiter.record_request_attempt()
                        
                        # Build display text with loading indicator
                        display_text = buffer.get_complete_text() + " " + config.loading_ellipse_emoji
                        
                        # Try to update the message
                        result = client.update_message_streaming(channel_id, message_id, display_text)
                        
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
                                    self.log_warning("Vision streaming circuit breaker opened")
                            else:
                                rate_limiter.record_failure(is_rate_limit=False)
                            self.log_debug(f"Failed to update message: {result.get('error', 'unknown error')}")
                
                # Call analyze_images with streaming callback
                self.log_info("Streaming vision analysis")
                analysis_result = self.openai_client.analyze_images(
                    images=images_to_analyze,
                    question=enhanced_question,
                    detail="high",
                    enhance_prompt=False,  # Already enhanced
                    conversation_history=enhanced_messages,  # Pass enhanced conversation with image analyses
                    system_prompt=system_prompt,  # Pass platform system prompt
                    stream_callback=stream_callback
                )
                
                # Log streaming stats
                stats = rate_limiter.get_stats()
                buffer_stats = buffer.get_stats()
                self.log_info(f"Vision streaming completed: {stats['successful_requests']}/{stats['total_requests']} updates, "
                             f"final length: {buffer_stats['text_length']} chars")
                
                self.log_debug(f"Vision analysis completed: {len(analysis_result)} chars")
            else:
                # Non-streaming version
                analysis_result = self.openai_client.analyze_images(
                    images=images_to_analyze,
                    question=enhanced_question,
                    detail="high",
                    enhance_prompt=False,  # Already enhanced
                    conversation_history=enhanced_messages,  # Pass enhanced conversation with image analyses
                    system_prompt=system_prompt  # Pass platform system prompt
                )
                self.log_debug(f"Vision analysis completed: {len(analysis_result)} chars")
            
            
        except TimeoutError as e:
            self.log_error(f"Vision analysis timed out: {e}")
            return Response(
                type="error",
                content=f"Image analysis timed out after {int(config.api_timeout_read)} seconds. Please try again."
            )
            
        except Exception as e:
            self.log_error(f"Vision analysis failed: {e}", exc_info=True)
            return Response(
                type="error",
                content=f"Failed to analyze image: {str(e)}"
            )
        
        # Track all analyzed images in AssetLedger and database
        asset_ledger = self.thread_manager.get_or_create_asset_ledger(thread_state.thread_ts)
        thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
        
        # Save to database with error handling
        try:
            # Track URL images and save analysis
            if url_images:
                for url_img in url_images:
                    asset_ledger.add_url_image(
                        image_data=url_img["base64_data"],
                        url=url_img["original_url"],
                        timestamp=time.time()
                    )
                    self.log_debug(f"Added URL image to AssetLedger: {url_img['original_url']}")
                    
                    # Save comprehensive vision analysis for URL images
                    if self.db:
                        self.db.save_image_metadata(
                            thread_id=thread_key,
                            url=url_img["original_url"],
                            image_type="url",
                            prompt=user_text if user_text else "Vision analysis",
                            analysis=analysis_result,  # Store the vision analysis result
                            metadata={"timestamp": time.time()},
                            message_ts=message.metadata.get("ts") if message.metadata else None
                        )
                        self.log_debug(f"Saved comprehensive vision analysis for URL image: {url_img['original_url']}")
            
            # Save analysis for image attachments only (not documents)
            if attachments and self.db:
                for att in attachments:
                    # Only save actual images, not PDFs or other documents
                    if att.get("url") and att.get("type") == "image":
                        # Save the comprehensive vision analysis to database
                        self.db.save_image_metadata(
                            thread_id=thread_key,
                            url=att["url"],
                            image_type="uploaded",
                            prompt=user_text if user_text else "Vision analysis",
                            analysis=analysis_result,  # Store the vision analysis result
                            metadata={"timestamp": time.time()},
                            message_ts=message.metadata.get("ts") if message.metadata else None
                        )
                        self.log_debug(f"Saved comprehensive vision analysis for uploaded image: {att['url']}")
                        
        except Exception as e:
            self.log_error(f"Failed to save image metadata to database: {e}", exc_info=True)
            # Continue anyway - don't fail the whole request due to DB issues
        
        # Create breadcrumb with filenames for backend history
        if filenames:
            if user_text:
                breadcrumb_text = f"[Uploaded: {', '.join(filenames)}] {user_text}"
            else:
                breadcrumb_text = f"[Uploaded: {', '.join(filenames)}]"
        else:
            breadcrumb_text = user_text if user_text else "[User uploaded image(s) for analysis]"
        
        # Add URL if it was a URL-based image
        if img_input.get("source") == "url" and img_input.get("original_url"):
            breadcrumb_text += f" <{img_input['original_url']}>"
        
        # Add to thread state with error handling
        thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
        try:
            message_ts = message.metadata.get("ts") if message.metadata else None
            # Count total images for token estimation
            total_images = len(image_inputs)
            # Build metadata with filenames
            metadata = {
                "type": "vision_analysis", 
                "image_count": total_images
            }
            if filenames:
                metadata["filenames"] = filenames
            
            self._add_message_with_token_management(
                thread_state, "user", breadcrumb_text, 
                db=self.db, thread_key=thread_key, message_ts=message_ts,
                metadata=metadata
            )
            self._add_message_with_token_management(thread_state, "assistant", analysis_result, db=self.db, thread_key=thread_key)
        except Exception as e:
            self.log_error(f"Failed to save messages to database: {e}", exc_info=True)
            # Continue anyway - messages are in memory at least
        
        # Check if we streamed the response
        response_metadata = {}
        streaming_enabled = thread_config.get('enable_streaming', config.enable_streaming)
        if (hasattr(client, 'supports_streaming') and client.supports_streaming() and 
            streaming_enabled and thinking_id is not None):
            response_metadata["streamed"] = True
        
        return Response(
            type="text",
            content=analysis_result,
            metadata=response_metadata
        )
    
    def _handle_image_generation(self, prompt: str, thread_state, client: BaseClient, 
                                channel_id: str, thinking_id: Optional[str], message: Message, 
                                skip_enhancement: bool = False) -> Response:
        """Handle image generation request with streaming enhancement
        
        Args:
            skip_enhancement: If True, skip prompt enhancement (used when falling back from failed edit)
        """
        self.log_info(f"Generating image for prompt: {prompt[:100]}...")
        
        # Initialize response metadata
        response_metadata = {}
        
        # Get thread config (with user preferences)
        thread_config = config.get_thread_config(
            overrides=thread_state.config_overrides,
            user_id=message.user_id,
            db=self.db
        )
        
        # Inject stored image analyses for style consistency
        enhanced_messages = self._inject_image_analyses(thread_state.messages, thread_state)
        
        # Pre-trim messages to fit within context window
        enhanced_messages = self._pre_trim_messages_for_api(enhanced_messages)
        
        # Check if streaming is supported (respecting user prefs) and enhancement is needed
        streaming_enabled = thread_config.get('enable_streaming', config.enable_streaming)
        if hasattr(client, 'supports_streaming') and client.supports_streaming() and streaming_enabled and not skip_enhancement:
            # Stream the enhancement to user with proper rate limiting
            streaming_config = client.get_streaming_config() if hasattr(client, 'get_streaming_config') else {}
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
            
            
            def enhancement_callback(chunk: str):
                buffer.add_chunk(chunk)
                # Only update if both buffer says it's time AND rate limiter allows it
                if buffer.should_update() and rate_limiter.can_make_request():
                    rate_limiter.record_request_attempt()
                    display_text = f"*Enhanced Prompt:* ✨ _{buffer.get_complete_text()}_ {config.loading_ellipse_emoji}"
                    
                    result = client.update_message_streaming(channel_id, thinking_id, display_text)
                    
                    if result["success"]:
                        rate_limiter.record_success()
                        buffer.mark_updated()
                        buffer.update_interval_setting(rate_limiter.get_current_interval())
                    else:
                        if result["rate_limited"]:
                            if result["retry_after"]:
                                rate_limiter.set_retry_after(result["retry_after"])
                            rate_limiter.record_failure(is_rate_limit=True)
                            
                            # Check if circuit breaker opened
                            if not rate_limiter.is_streaming_enabled():
                                self.log_warning("Image enhancement rate limited - circuit breaker opened")
                        else:
                            rate_limiter.record_failure(is_rate_limit=False)
            
            # Enhance prompt with streaming (returns the complete enhanced text)
            enhanced_prompt = self.openai_client._enhance_image_prompt(
                prompt=prompt,
                conversation_history=enhanced_messages,
                stream_callback=enhancement_callback
            )
            
            # Show the final enhanced prompt
            if enhanced_prompt and thinking_id:
                enhanced_text = f"*Enhanced Prompt:* ✨ _{enhanced_prompt}_"
                client.update_message_streaming(channel_id, thinking_id, enhanced_text)
                # Mark that we should NOT touch this message again
                response_metadata["prompt_message_id"] = thinking_id
            
            # Create a NEW message for generating status - don't touch the enhanced prompt!
            generating_id = client.send_thinking_indicator(channel_id, thread_state.thread_ts)
            self._update_status(client, channel_id, generating_id, 
                              "Generating image. This may take a minute...", 
                              emoji=config.circle_loader_emoji)
            # Track the status message ID
            response_metadata["status_message_id"] = generating_id
            
            # Generate image with already-enhanced prompt
            try:
                image_data = self.openai_client.generate_image(
                    prompt=enhanced_prompt,
                    size=thread_config.get("image_size"),
                    quality=thread_config.get("image_quality"),
                    enhance_prompt=False,  # Already enhanced!
                    conversation_history=None  # Not needed since we enhanced already
                )
            except Exception as e:
                error_str = str(e)
                if "moderation_blocked" in error_str or "safety system" in error_str or "content policy" in error_str.lower():
                    # Clean up status message
                    if "status_message_id" in response_metadata:
                        if hasattr(client, 'delete_message'):
                            client.delete_message(channel_id, response_metadata["status_message_id"])
                    
                    # Provide friendly message
                    moderation_msg = (
                        "I couldn't generate that image as it was flagged by content safety filters. "
                        "This can happen with certain brand names, people, or other protected content. "
                        "Try rephrasing your request or describing what you want without using specific names."
                    )
                    
                    # Add to thread for context
                    thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
                    message_ts = message.metadata.get("ts") if message.metadata else None
                    formatted_prompt = self._format_user_content_with_username(prompt, message)
                    self._add_message_with_token_management(thread_state, "user", formatted_prompt, db=self.db, thread_key=thread_key, message_ts=message_ts)
                    self._add_message_with_token_management(thread_state, "assistant", moderation_msg, db=self.db, thread_key=thread_key)
                    
                    return Response(
                        type="text",
                        content=moderation_msg
                    )
                else:
                    # Re-raise other errors
                    raise
            
            # After image is generated, update message to show just the enhanced prompt
            # (remove the "Generating image..." status) - but respect rate limits
            if enhanced_prompt and thinking_id and rate_limiter.can_make_request():
                result = client.update_message_streaming(channel_id, thinking_id, f"*Enhanced Prompt:* ✨ _{enhanced_prompt}_")
                if result["success"]:
                    rate_limiter.record_success()
                else:
                    if result["rate_limited"]:
                        rate_limiter.record_failure(is_rate_limit=True)
                        self.log_debug("Couldn't clean up image gen status due to rate limit")
        else:
            # Non-streaming fallback or skip enhancement
            if not skip_enhancement:
                self._update_status(client, channel_id, thinking_id, "Enhancing your prompt...")
            
            # Generate image with or without enhancement
            self._update_status(client, channel_id, thinking_id, "Creating your image. This may take a minute...", emoji=config.circle_loader_emoji)
            
            try:
                image_data = self.openai_client.generate_image(
                    prompt=prompt,
                    size=thread_config.get("image_size"),
                    quality=thread_config.get("image_quality"),
                    enhance_prompt=not skip_enhancement,  # Skip if fallback from failed edit
                    conversation_history=enhanced_messages if not skip_enhancement else None  # Only pass if enhancing
                )
            except Exception as e:
                error_str = str(e)
                if "moderation_blocked" in error_str or "safety system" in error_str or "content policy" in error_str.lower():
                    # Provide friendly message
                    moderation_msg = (
                        "I couldn't generate that image as it was flagged by content safety filters. "
                        "This can happen with certain brand names, people, or other protected content. "
                        "Try rephrasing your request or describing what you want without using specific names."
                    )
                    
                    # Add to thread for context
                    thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
                    message_ts = message.metadata.get("ts") if message.metadata else None
                    formatted_prompt = self._format_user_content_with_username(prompt, message)
                    self._add_message_with_token_management(thread_state, "user", formatted_prompt, db=self.db, thread_key=thread_key, message_ts=message_ts)
                    self._add_message_with_token_management(thread_state, "assistant", moderation_msg, db=self.db, thread_key=thread_key)
                    
                    return Response(
                        type="text",
                        content=moderation_msg
                    )
                else:
                    # Re-raise other errors
                    raise
        
        # Store in asset ledger
        asset_ledger = self.thread_manager.get_or_create_asset_ledger(thread_state.thread_ts)
        thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
        asset_ledger.add_image(
            image_data.base64_data,
            image_data.prompt,  # Use the enhanced prompt
            time.time(),
            db=self.db,
            thread_id=thread_key
        )
        
        # Add breadcrumb to thread state with metadata
        thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
        message_ts = message.metadata.get("ts") if message.metadata else None
        formatted_prompt = self._format_user_content_with_username(prompt, message)
        self._add_message_with_token_management(thread_state, "user", formatted_prompt, db=self.db, thread_key=thread_key, message_ts=message_ts)
        # Store enhanced prompt with metadata for new image tracking
        self._add_message_with_token_management(thread_state, 
            "assistant", 
            image_data.prompt,  # Just the enhanced prompt, no "Generated image:" prefix
            db=self.db, 
            thread_key=thread_key,
            metadata={
                "type": "image_generation",
                "prompt": image_data.prompt,
                "url": None  # Will be updated after upload
            }
        )
        
        # Mark as streamed if we used streaming for the enhancement
        # Note: response_metadata already initialized and populated above
        if hasattr(client, 'supports_streaming') and client.supports_streaming() and streaming_enabled:
            response_metadata["streamed"] = True
            
        return Response(
            type="image",
            content=image_data,
            metadata=response_metadata
        )
    
    def _find_target_image(self, user_text: str, thread_state, client: BaseClient) -> Optional[str]:
        """Find the target image URL based on user's reference using DB"""
        thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
        
        # Get all images from DB for this thread
        all_available_images = []
        if self.db:
            db_images = self.db.find_thread_images(thread_key)
            for img in db_images:
                all_available_images.append({
                    "url": img.get("url"),
                    "description": img.get("prompt", "") or "uploaded image",
                    "type": img.get("image_type", "uploaded"),
                    "analysis": img.get("analysis", "")  # Include analysis for natural language matching
                })
        
        # Fallback to asset ledger if no DB or no images found
        if not all_available_images:
            image_registry = self._extract_image_registry(thread_state)
            for img in image_registry:
                all_available_images.append({
                    "url": img["url"],
                    "description": img["description"],
                    "type": "generated",
                    "analysis": ""
                })
        
        if not all_available_images:
            return None
        
        # If only one image, use it
        if len(all_available_images) == 1:
            self.log_debug(f"Only one image found, using it: {all_available_images[0]['url']}")
            return all_available_images[0]["url"]
        
        # If ambiguous and multiple images, use utility model to match
        # Let natural language processing handle ALL references including ordinals
        if len(all_available_images) > 1:
            # Build context for matching using DB analyses - FULL context per CLAUDE.md
            context = "Available images:\n"
            for i, img in enumerate(all_available_images, 1):
                context += f"{i}. {img['type'].capitalize()} image"
                if img['description']:
                    context += f": {img['description']}"  # Full description, no truncation
                context += "\n"
                
                # Include analysis if available for natural language matching
                if img.get('analysis'):
                    # Include FULL visual analysis for better matching - no truncation per CLAUDE.md
                    context += f"   Visual details: {img['analysis']}\n"
            
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
        thinking_id: Optional[str],
        message: Message
    ) -> Response:
        """Handle image modification request by finding and editing the target image"""
        # Don't update status yet - we might fall back to generation
        
        # Initialize response metadata early to track status messages
        response_metadata = {}
        
        # Try to find target image URL from conversation
        target_url = self._find_target_image(text, thread_state, client)
        
        if target_url:
            # Found an image to edit - update status
            self._update_status(client, channel_id, thinking_id, "Finding the image to edit...", emoji=config.web_search_emoji)
            
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
                        question=IMAGE_ANALYSIS_PROMPT,
                        detail="high"
                    )
                    
                    # Prepare for edit
                    self.log_info(f"Editing existing image with request: {text}")
                    
                    # Get thread config (with user preferences)
                    thread_config = config.get_thread_config(
                        overrides=thread_state.config_overrides,
                        user_id=message.user_id,
                        db=self.db
                    )
                    
                    # Inject stored image analyses for style matching
                    enhanced_messages = self._inject_image_analyses(thread_state.messages, thread_state)
                    
                    # Pre-trim messages to fit within context window
                    enhanced_messages = self._pre_trim_messages_for_api(enhanced_messages, model=thread_state.current_model)
                    
                    # Check if streaming is supported for enhancement
                    streaming_enabled = thread_config.get('enable_streaming', config.enable_streaming)
                    if hasattr(client, 'supports_streaming') and client.supports_streaming() and streaming_enabled:
                        # Stream the enhancement to user with proper rate limiting
                        from streaming.buffer import StreamingBuffer
                        from streaming import RateLimitManager
                        
                        streaming_config = client.get_streaming_config() if hasattr(client, 'get_streaming_config') else {}
                        buffer = StreamingBuffer(
                            update_interval=streaming_config.get("update_interval", 2.0),
                            buffer_size_threshold=streaming_config.get("buffer_size", 500),
                            min_update_interval=streaming_config.get("min_interval", 1.0)
                        )
                        
                        rate_limiter = RateLimitManager(
                            base_interval=streaming_config.get("update_interval", 2.0),
                            failure_threshold=streaming_config.get("circuit_breaker_threshold", 5),
                            cooldown_seconds=streaming_config.get("circuit_breaker_cooldown", 60)
                        )
                        
                        def enhancement_callback(chunk: str):
                            buffer.add_chunk(chunk)
                            # Only update if both buffer says it's time AND rate limiter allows it
                            if buffer.should_update() and rate_limiter.can_make_request():
                                rate_limiter.record_request_attempt()
                                display_text = f"*Enhanced Prompt:* ✨ _{buffer.get_complete_text()}_ {config.loading_ellipse_emoji}"
                                
                                result = client.update_message_streaming(channel_id, thinking_id, display_text)
                                
                                if result["success"]:
                                    rate_limiter.record_success()
                                    buffer.mark_updated()
                                    buffer.update_interval_setting(rate_limiter.get_current_interval())
                                else:
                                    if result["rate_limited"]:
                                        if result["retry_after"]:
                                            rate_limiter.set_retry_after(result["retry_after"])
                                        rate_limiter.record_failure(is_rate_limit=True)
                                        
                                        # Check if circuit breaker opened
                                        if not rate_limiter.is_streaming_enabled():
                                            self.log_warning("Image edit enhancement rate limited - circuit breaker opened")
                                    else:
                                        rate_limiter.record_failure(is_rate_limit=False)
                        
                        # First enhance the prompt with streaming
                        enhanced_edit_prompt = self.openai_client._enhance_image_edit_prompt(
                            user_request=text,
                            image_description=image_description,
                            conversation_history=enhanced_messages,
                            stream_callback=enhancement_callback
                        )
                        
                        # Show the final enhanced prompt
                        if enhanced_edit_prompt and thinking_id:
                            enhanced_text = f"*Enhanced Prompt:* ✨ _{enhanced_edit_prompt}_"
                            client.update_message_streaming(channel_id, thinking_id, enhanced_text)
                            # Mark that we should NOT touch this message again
                            response_metadata["prompt_message_id"] = thinking_id
                        
                        # Create a NEW message for editing status - don't touch the enhanced prompt!
                        editing_id = client.send_thinking_indicator(channel_id, thread_state.thread_ts)
                        self._update_status(client, channel_id, editing_id, 
                                          "Editing your image. This may take a minute...", 
                                          emoji=config.circle_loader_emoji)
                        # Track the status message ID
                        response_metadata["status_message_id"] = editing_id
                        
                        # Mark as streamed for main.py
                        response_metadata["streamed"] = True
                        
                        # Use the edit_image API with the pre-enhanced prompt
                        edited_image = self.openai_client.edit_image(
                            input_images=[base64_data],
                            prompt=enhanced_edit_prompt,
                            image_description=None,  # Already used for enhancement
                            input_mimetypes=["image/png"],
                            input_fidelity=thread_config.get("input_fidelity", "high"),
                            background=thread_config.get("image_background", "auto"),
                            output_format=thread_config.get("image_format", "png"),
                            output_compression=thread_config.get("image_compression", 100),
                            enhance_prompt=False,  # Already enhanced!
                            conversation_history=None  # Not needed since we enhanced already
                        )
                    else:
                        # Non-streaming fallback
                        self._update_status(client, channel_id, thinking_id, "Enhancing your edit request...")
                        self._update_status(client, channel_id, thinking_id, "Editing your image. This may take a minute...", emoji=config.circle_loader_emoji)
                        
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
                            conversation_history=enhanced_messages  # Pass enhanced conversation with image analyses
                        )
                    
                    # Add breadcrumbs with metadata
                    thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
                    message_ts = message.metadata.get("ts") if message.metadata else None
                    formatted_text = self._format_user_content_with_username(text, message)
                    self._add_message_with_token_management(thread_state, "user", formatted_text, db=self.db, thread_key=thread_key, message_ts=message_ts)
                    self._add_message_with_token_management(thread_state, 
                        "assistant", 
                        edited_image.prompt,
                        db=self.db, 
                        thread_key=thread_key,
                        metadata={
                            "type": "image_edit",
                            "prompt": edited_image.prompt,
                            "url": None  # Will be updated after upload
                        }
                    )
                    
                    return Response(
                        type="image",
                        content=edited_image,
                        metadata=response_metadata
                    )
                else:
                    self.log_warning(f"Failed to download image from URL: {target_url}")
                    
            except Exception as e:
                self.log_error(f"Error editing image from URL: {e}")
                
                # Check if this is a moderation block
                error_str = str(e)
                if "moderation_blocked" in error_str or "safety system" in error_str or "content policy" in error_str.lower():
                    # Moderation block - provide friendly message and don't fall back
                    self.log_warning(f"Edit blocked by moderation: {text}")
                    
                    # Clean up any status messages
                    if "status_message_id" in response_metadata:
                        if hasattr(client, 'delete_message'):
                            client.delete_message(channel_id, response_metadata["status_message_id"])
                    
                    # Add user-friendly explanation to thread
                    moderation_msg = (
                        "I couldn't complete that edit request as it was flagged by content safety filters. "
                        "This can happen with certain brand names, people, or other protected content. "
                        "Try rephrasing your request or describing what you want without using specific names."
                    )
                    
                    # Add messages to thread for context
                    thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
                    message_ts = message.metadata.get("ts") if message.metadata else None
                    formatted_text = self._format_user_content_with_username(text, message)
                    self._add_message_with_token_management(thread_state, "user", formatted_text, db=self.db, thread_key=thread_key, message_ts=message_ts)
                    self._add_message_with_token_management(thread_state, "assistant", moderation_msg, db=self.db, thread_key=thread_key)
                    
                    return Response(
                        type="text",
                        content=moderation_msg
                    )
                
                # Other errors - clean up and fall back to generation
                if "status_message_id" in response_metadata:
                    # Delete the "Editing your image..." message
                    if hasattr(client, 'delete_message'):
                        client.delete_message(channel_id, response_metadata["status_message_id"])
                elif thinking_id and not response_metadata.get("prompt_message_id"):
                    # Update the thinking message to show we're generating instead
                    self._update_status(client, channel_id, thinking_id, 
                                      "Edit failed, generating new image instead...", 
                                      emoji=config.error_emoji)
        
        # Fallback to generation if edit failed or no URL found
        self.log_info("No image URL found or edit failed, falling back to generation based on description")
        
        # Look for image descriptions in history
        image_registry = self._extract_image_registry(thread_state)
        if image_registry:
            # Use the most recent image description without re-enhancement
            # The prompt already contains the edit request, just generate based on it
            return self._handle_image_generation(text, thread_state, client, channel_id, thinking_id, message, 
                                                skip_enhancement=True)
        else:
            # No previous images, treat as new generation
            return self._handle_image_generation(text, thread_state, client, channel_id, thinking_id, message)
    
    def _handle_mixed_content_analysis(
        self,
        user_text: str,
        image_inputs: List[Dict],
        document_inputs: List[Dict],
        thread_state,
        client: BaseClient,
        channel_id: str,
        thinking_id: Optional[str],
        message: Message
    ) -> Response:
        """Handle mixed content (images + documents) with two-call approach
        
        1. First call: Analyze images with vision model using technical prompt
        2. Second call: Combine image analysis + document content for final answer
        """
        try:
            # Step 1: Analyze images with vision model using IMAGE_ANALYSIS_PROMPT
            self._update_status(client, channel_id, thinking_id, "Analyzing images...", emoji=config.analyze_emoji)
            
            # Extract base64 data from image inputs
            images_to_analyze = []
            for img_input in image_inputs:
                if img_input.get("type") == "input_image":
                    image_url = img_input.get("image_url", "")
                    if image_url.startswith("data:"):
                        parts = image_url.split(",", 1)
                        if len(parts) == 2:
                            _, base64_data = parts
                            images_to_analyze.append(base64_data)
            
            # Analyze images with technical prompt (no enhancement needed)
            from prompts import IMAGE_ANALYSIS_PROMPT
            image_analysis = self.openai_client.analyze_images(
                images=images_to_analyze,
                question=IMAGE_ANALYSIS_PROMPT,
                detail="high",
                enhance_prompt=False  # Use technical prompt as-is
            )
            
            self.log_info(f"Image analysis completed: {len(image_analysis)} chars")
            
            # Step 2: Build comprehensive context with image analysis and documents
            self._update_status(client, channel_id, thinking_id, "Combining analysis with documents...", emoji=config.analyze_emoji)
            
            # Build enhanced message with all context
            context_parts = []
            
            # Add image analysis results
            if image_analysis:
                context_parts.append("=== IMAGE ANALYSIS ===")
                context_parts.append(image_analysis)
                context_parts.append("")
            
            # Add document content
            if document_inputs:
                doc_text = self._build_message_with_documents("", document_inputs)
                if doc_text and isinstance(doc_text, str):  # Ensure it's a string
                    context_parts.append(doc_text)
            
            # Add user question
            context_parts.append("")  # Empty line before question
            context_parts.append("USER QUESTION:")
            context_parts.append(user_text if user_text else "Please analyze the relationship between these files.")
            
            # Ensure all parts are strings before joining
            str_context_parts = []
            for i, part in enumerate(context_parts):
                if not isinstance(part, str):
                    self.log_warning(f"context_parts[{i}] is not a string: {type(part)}, value: {part}")
                    str_context_parts.append(str(part) if part is not None else "")
                else:
                    str_context_parts.append(part)
            
            combined_context = "\n".join(str_context_parts)
            
            # Step 3: Send to text model for final analysis
            self._update_status(client, channel_id, thinking_id, "Generating comprehensive response...", emoji=config.thinking_emoji)
            
            # Use text response handler with the combined context
            return self._handle_text_response(combined_context, thread_state, client, message, thinking_id)
            
        except Exception as e:
            self.log_error(f"Mixed content analysis failed: {e}", exc_info=True)
            return Response(
                type="error",
                content=f"Failed to analyze mixed content: {str(e)}"
            )
    
    def _handle_vision_without_upload(
        self,
        text: str,
        thread_state,
        client: BaseClient,
        channel_id: str,
        thinking_id: Optional[str],
        message: Message
    ) -> Response:
        """Handle vision request when no images are uploaded - use text response with context"""
        
        # Check if we have any images in the conversation that provide context
        image_registry = self._extract_image_registry(thread_state)
        has_images = bool(image_registry) or self._has_recent_image(thread_state)
        
        # Check if we have documents in the conversation
        document_ledger = self.thread_manager.get_document_ledger(thread_state.thread_ts)
        has_documents = document_ledger and len(document_ledger.documents) > 0
        
        if has_images or has_documents:
            # We have image/document context in the conversation - let the model use that
            self.log_info(f"Vision intent with {'image' if has_images else ''}{' and ' if has_images and has_documents else ''}{'document' if has_documents else ''} context in history - using text response with context")
        else:
            self.log_info("Vision intent but no images or documents found in conversation")
        
        # Don't attach documents to the message - they're already in the thread history
        # The model will have access to them from the conversation context
        return self._handle_text_response(text, thread_state, client, message, thinking_id)
    
    def _handle_image_edit(
        self,
        text: str,
        image_inputs: List[Dict],
        thread_state,
        client: BaseClient,
        channel_id: str,
        thinking_id: Optional[str],
        attachment_urls: Optional[List[str]] = None,
        message: Message = None
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
            return self._handle_image_generation(text, thread_state, client, channel_id, thinking_id, message)
        
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
        print("\n" + "="*100)
        print("DEBUG: IMAGE EDIT FLOW - STEP 1: ANALYZE IMAGE")
        print("="*100)
        print(f"Analysis Question: {IMAGE_ANALYSIS_PROMPT}")
        print("="*100)
        
        try:
            # Analyze the images to understand what's in them
            image_description = self.openai_client.analyze_images(
                images=input_images,
                question=IMAGE_ANALYSIS_PROMPT,
                detail="high"
            )
            
            # Log the full analysis result
            print("\n" + "="*100)
            print("DEBUG: IMAGE EDIT FLOW - STEP 2: ANALYSIS RESULT")
            print("="*100)
            print(f"Image Description (Full):\n{image_description}")
            print("="*100)
            
            # Don't show the analysis - just update status to show we're editing
            # The analysis is only used internally for better edit quality
            if thinking_id:
                self._update_status(client, channel_id, thinking_id, "Editing your image. This may take a minute...", emoji=config.circle_loader_emoji)
            
            # Store the description and user request separately for clean enhancement
            image_analysis = image_description
            user_edit_request = text
            
        except Exception as e:
            self.log_warning(f"Failed to analyze images, continuing without context: {e}")
            image_analysis = None
            user_edit_request = text
            print("\n" + "="*100)
            print("DEBUG: IMAGE EDIT FLOW - ANALYSIS FAILED")
            print("="*100)
            print(f"Error: {e}")
            print(f"Falling back to user prompt only: {text}")
            print("="*100)
        
        # Get thread config for settings (with user preferences)
        thread_config = config.get_thread_config(
            overrides=thread_state.config_overrides,
            user_id=message.user_id,
                db=self.db
        )
        
        # Inject stored image analyses for style matching
        enhanced_messages = self._inject_image_analyses(thread_state.messages, thread_state) if thread_state.messages else None
        
        # Pre-trim messages to fit within context window if we have messages
        if enhanced_messages:
            enhanced_messages = self._pre_trim_messages_for_api(enhanced_messages, model=thread_state.current_model)
        
        # Check if streaming is supported for enhancement
        response_metadata = {}
        streaming_enabled = thread_config.get('enable_streaming', config.enable_streaming)
        if hasattr(client, 'supports_streaming') and client.supports_streaming() and streaming_enabled:
            # Stream the enhancement to user with proper rate limiting
            from streaming.buffer import StreamingBuffer
            from streaming import RateLimitManager
            
            streaming_config = client.get_streaming_config() if hasattr(client, 'get_streaming_config') else {}
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
            
            
            def enhancement_callback(chunk: str):
                buffer.add_chunk(chunk)
                # Only update if both buffer says it's time AND rate limiter allows it
                if buffer.should_update() and rate_limiter.can_make_request():
                    rate_limiter.record_request_attempt()
                    display_text = f"*Enhanced Prompt:* ✨ _{buffer.get_complete_text()}_ {config.loading_ellipse_emoji}"
                    
                    result = client.update_message_streaming(channel_id, thinking_id, display_text)
                    
                    if result["success"]:
                        rate_limiter.record_success()
                        buffer.mark_updated()
                        buffer.update_interval_setting(rate_limiter.get_current_interval())
                    else:
                        if result["rate_limited"]:
                            if result["retry_after"]:
                                rate_limiter.set_retry_after(result["retry_after"])
                            rate_limiter.record_failure(is_rate_limit=True)
                            
                            # Check if circuit breaker opened
                            if not rate_limiter.is_streaming_enabled():
                                self.log_warning("Image edit enhancement rate limited - circuit breaker opened")
                        else:
                            rate_limiter.record_failure(is_rate_limit=False)
            
            # First enhance the prompt with streaming
            enhanced_edit_prompt = self.openai_client._enhance_image_edit_prompt(
                user_request=user_edit_request,
                image_description=image_analysis,
                conversation_history=enhanced_messages,
                stream_callback=enhancement_callback
            )
            
            # Show the final enhanced prompt
            if enhanced_edit_prompt and thinking_id:
                enhanced_text = f"Enhanced Prompt: ✨ _{enhanced_edit_prompt}_"
                client.update_message_streaming(channel_id, thinking_id, enhanced_text)
                # Mark that we should NOT touch this message again
                response_metadata["prompt_message_id"] = thinking_id
            
            # Create a NEW message for editing status - don't touch the enhanced prompt!
            editing_id = client.send_thinking_indicator(channel_id, thread_state.thread_ts)
            self._update_status(client, channel_id, editing_id, 
                              "Generating edited image. This may take a minute...", 
                              emoji=config.circle_loader_emoji)
            # Track the status message ID
            response_metadata["status_message_id"] = editing_id
            
            # Mark as streamed for main.py
            response_metadata["streamed"] = True
            
            # Use the edit_image API with the pre-enhanced prompt
            try:
                image_data = self.openai_client.edit_image(
                    input_images=input_images,
                    input_mimetypes=input_mimetypes,
                    prompt=enhanced_edit_prompt,  # Use the pre-enhanced prompt
                    image_description=None,  # Don't pass description since we already enhanced
                    input_fidelity=thread_config.get("input_fidelity", "high"),
                    background=thread_config.get("image_background", "auto"),
                    output_format=thread_config.get("image_format", "png"),
                    output_compression=thread_config.get("image_compression", 100),
                    enhance_prompt=False,  # Already enhanced!
                    conversation_history=None  # Not needed since already enhanced
                )
                
                # After edit is complete, update message to show just the enhanced prompt
                # (remove the "Generating edited image..." status) - but respect rate limits
                if enhanced_edit_prompt and thinking_id and rate_limiter.can_make_request():
                    result = client.update_message_streaming(channel_id, thinking_id, f"*Enhanced Prompt:* ✨ _{enhanced_edit_prompt}_")
                    if result["success"]:
                        rate_limiter.record_success()
                    else:
                        if result["rate_limited"]:
                            rate_limiter.record_failure(is_rate_limit=True)
                            self.log_debug("Couldn't clean up image edit status due to rate limit")
                    
            except Exception as e:
                self.log_error(f"Error editing image: {e}")
                return Response(
                    type="error",
                    content=f"Failed to edit image: {str(e)}"
                )
        else:
            # Non-streaming fallback
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
                    conversation_history=enhanced_messages  # Pass enhanced conversation with image analyses
                )
            except Exception as e:
                self.log_error(f"Error editing image: {e}")
                return Response(
                    type="error",
                    content=f"Failed to edit image: {str(e)}"
                )
        
        # Add clean message to thread state (no URLs or counts)
        user_breadcrumb = text or ""
        thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
        message_ts = message.metadata.get("ts") if message.metadata else None
        formatted_breadcrumb = self._format_user_content_with_username(user_breadcrumb, message)
        self._add_message_with_token_management(thread_state, "user", formatted_breadcrumb, db=self.db, thread_key=thread_key, message_ts=message_ts)
        
        # Store edited image in asset ledger
        asset_ledger = self.thread_manager.get_or_create_asset_ledger(thread_state.thread_ts)
        thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
        asset_ledger.add_image(
            image_data.base64_data,
            image_data.prompt,  # The enhanced edit prompt
            time.time(),
            source="edited",
            db=self.db,
            thread_id=thread_key,
            analysis=image_analysis  # Store the full analysis
        )
        
        # Store the edit prompt with metadata for tracking
        # Include the original analysis as part of the content for context
        if image_analysis:
            # Include analysis for future vision questions
            content = f"{image_data.prompt}\n\n[Original: {image_analysis}]"
        else:
            content = image_data.prompt
            
        self._add_message_with_token_management(thread_state, 
            "assistant", 
            content,
            db=self.db, 
            thread_key=thread_key,
            metadata={
                "type": "image_edit",
                "prompt": image_data.prompt,
                "original_analysis": image_analysis if image_analysis else None,
                "url": None  # Will be updated after upload
            }
        )
        
        return Response(
            type="image",
            content=image_data,
            metadata=response_metadata
        )
    
    def update_last_image_url(self, channel_id: str, thread_id: str, url: str):
        """Update the last assistant message with the image URL"""
        thread_state = self.thread_manager.get_or_create_thread(thread_id, channel_id)
        
        # Find the last assistant message with image metadata or legacy format
        for i in range(len(thread_state.messages) - 1, -1, -1):
            msg = thread_state.messages[i]
            if msg.get("role") == "assistant":
                metadata = msg.get("metadata", {})
                
                # Check metadata first (new approach)
                if metadata.get("type") in ["image_generation", "image_edit"]:
                    # Update metadata with URL
                    if "metadata" not in msg:
                        msg["metadata"] = {}
                    msg["metadata"]["url"] = url
                    self.log_debug(f"Updated message metadata with URL: {url}")
                    
                    # Save to database for persistence across restarts
                    if self.db:
                        thread_key = f"{channel_id}:{thread_id}"
                        image_type = "generated" if metadata.get("type") == "image_generation" else "edited"
                        prompt = metadata.get("prompt", "")
                        
                        # Save the image metadata to DB
                        self.db.save_image_metadata(
                            thread_id=thread_key,
                            url=url,
                            image_type=image_type,
                            prompt=prompt,
                            analysis="",  # No analysis for generated images
                            original_analysis=""
                        )
                        self.log_info(f"Saved {image_type} image to DB: {url}")
                    break
                    
                # Fallback to string matching for backward compatibility
                elif "Generated image:" in msg.get("content", "") or "Edited image:" in msg.get("content", ""):
                    # Add URL if not already present
                    if "<" not in msg["content"]:
                        msg["content"] += f" <{url}>"
                        self.log_debug(f"Updated message content with URL: {url}")
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