from __future__ import annotations

from typing import Any, Dict, List, Optional

from config import config


class ThreadManagementMixin:
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
                trimmed_messages.pop(start_index)
                removed_count += 1
                current_tokens = self.thread_manager._token_counter.count_thread_tokens(trimmed_messages) + new_message_tokens
                self.log_debug(f"Pre-trimmed message {removed_count}, tokens now: {current_tokens}")
            else:
                self.log_warning("Cannot trim further - would remove current message")
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
