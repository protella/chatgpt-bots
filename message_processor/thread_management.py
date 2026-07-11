from __future__ import annotations

from typing import Any, Dict, List, Optional

from base_client import BaseClient, Message
from config import config, pipeline_status
from message_markers import (
    ends_with_continuation,
    starts_as_continuation,
    strip_continuation_markers,
)
from message_processor.message_timestamps import sender_timezone, stamp_content
from message_processor.tool_provenance import (
    render_used_tools_annotation,
    strip_used_tools_footer,
)


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

    async def _pre_trim_messages_for_api(self, messages: List[Dict[str, Any]], new_message_tokens: int = 0, model: str = None, thread_state=None) -> List[Dict[str, Any]]:
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
            trimmed_count = await self._smart_trim_with_summarization(thread_state)
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

        # Phase D2 summary blocks are already-compressed (summary + ref, no full
        # content) — preserved; read_document covers depth, so re-summarizing
        # them would only destroy the pointer.
        if "=== DOCUMENT SUMMARY:" in content:
            return True
        
        # Check for injected analysis markers
        if "[Image Analysis:" in content or "[Vision Context:" in content:
            return True
        
        return False

    async def _summarize_document_content(self, content: str) -> str:
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
            summary = await self.openai_client.create_text_response(
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
            # Check if it's a context length error
            if "context_length_exceeded" in str(e) or "context window" in str(e):
                self.log_error("Document exceeds model context window for summarization")
                # Return truncated version with error marker
                return f"[ERROR: Document too large for model context window]\n{content[:1000]}..."
            return content  # Return original if summarization fails for other reasons

    async def _smart_trim_with_summarization(self, thread_state, trim_count: int = None,
                                             collector: Optional[List[Dict]] = None) -> int:
        """Intelligently trim messages, summarizing documents only when they're in the trim list

        This method identifies the oldest N messages to be trimmed. If any contain
        unsummarized documents, it summarizes them in place (making them preserved).
        Then it trims any remaining non-preserved messages from the list.

        Args:
            thread_state: Thread state object to trim
            trim_count: Number of messages to trim (default from config)
            collector: Optional list — dropped messages are appended so the caller can
                summarize the span into thread_summaries (Phase S compaction)

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

                    # Check if document would exceed context window for summarization
                    # Count tokens to see if it fits in the 350k limit
                    doc_tokens = self.thread_manager._token_counter.count_tokens(original_content)

                    # Get the model's token limit
                    model = thread_state.current_model or config.gpt_model
                    max_tokens = config.get_model_token_limit(model)

                    if doc_tokens > max_tokens:  # Model's token limit
                        self.log_warning(f"Document too large to summarize: {doc_tokens} tokens > {max_tokens} limit - dropping from context")

                        # Replace the message content with a placeholder
                        truncated_content = f"[Document removed - exceeded context window: {doc_tokens:,} tokens]\nFilename: "

                        # Try to extract filename for the placeholder
                        import re
                        doc_match = re.search(r'=== DOCUMENT: (.*?) ===', original_content)
                        if doc_match:
                            filename = doc_match.group(1).strip()
                            truncated_content += filename
                        else:
                            truncated_content += "Unknown"

                        # Update the message with placeholder
                        thread_state.messages[idx]["content"] = truncated_content

                        # Update metadata
                        if "metadata" not in thread_state.messages[idx]:
                            thread_state.messages[idx]["metadata"] = {}
                        thread_state.messages[idx]["metadata"]["document_removed"] = True
                        thread_state.messages[idx]["metadata"]["original_tokens"] = doc_tokens

                        self.log_info(f"Replaced oversized document with placeholder: {doc_tokens} tokens -> {len(truncated_content)} chars")
                        documents_summarized += 1  # Count as "processed" so we make progress
                        continue

                    self.log_info(f"Summarizing document at index {idx} (in trim list of {len(indices_to_process)} messages)")

                    # Summarize the document
                    summarized_content = await self._summarize_document_content(original_content)
                    
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
                if collector is not None:
                    collector.append(removed_msg)
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

    # --- Phase S: chunky compaction + rolling thread summary -------------------------

    SUMMARY_HEAD_MARKER = "thread_summary"

    async def _compact_thread_to_target(self, thread_state, thread_key: str) -> int:
        """Compact a thread to the configured target in ONE deliberate pass.

        Prompt-cache note: rewriting the head of the conversation is an expected,
        deliberate prefix-cache bust. That's why compaction is CHUNKY — compact down to
        token_compaction_target (well under the limit) in one pass, instead of trimming
        a few messages per turn, which would bust the OpenAI prefix cache every turn.

        Dropped messages are summarized (rolling, via the utility model) into the
        thread_summaries table, with structured refs preserved, and the summary head
        message in the live thread state is created/updated in place.

        Returns the number of messages dropped or summarized-in-place.
        """
        model = thread_state.current_model or config.gpt_model
        max_tokens = config.get_model_token_limit(model)
        target_tokens = int(max_tokens * config.token_compaction_target)

        dropped: List[Dict] = []
        total_processed = 0
        current_tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages)

        while current_tokens > target_tokens:
            processed = await self._smart_trim_with_summarization(thread_state, collector=dropped)
            if processed == 0:
                self.log_warning(
                    f"Compaction stalled at {current_tokens}/{target_tokens} tokens — "
                    f"no trimmable messages left"
                )
                break
            total_processed += processed
            current_tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages)

        if dropped:
            await self._write_thread_summary(thread_state, thread_key, dropped)

        # Re-baseline the tracked context size from the compacted messages; the next
        # API call's usage replaces this with the exact number.
        thread_state.reset_context_estimate(self.thread_manager._token_counter)

        if total_processed:
            self.log_info(
                f"Compacted thread {thread_key}: {total_processed} message(s) processed, "
                f"now at {current_tokens}/{target_tokens} target tokens"
            )
        return total_processed

    async def _write_thread_summary(self, thread_state, thread_key: str, dropped: List[Dict]):
        """Fold a dropped span into the rolling thread summary (DB row + in-state head).

        boundary_ts advances to the newest Slack ts in the dropped span. Messages without
        a known ts (some live assistant turns) are covered by the summary text but can
        reappear in a cold-rebuild tail — harmless duplication of meaning, never data loss.
        """
        if not self.db:
            return

        # Restore original order (collector receives popped messages newest-first per batch)
        def _ts_key(m):
            ts = (m.get("metadata") or {}).get("ts")
            try:
                return float(ts)
            except (TypeError, ValueError):
                return float("inf")
        ordered = sorted(dropped, key=_ts_key)

        prior = None
        try:
            prior = await self.db.get_thread_summary_async(thread_key)
        except Exception as e:
            self.log_warning(f"Could not load prior thread summary: {e}")

        # Boundary: newest known ts among dropped messages; fall back to prior boundary
        known_ts = [(m.get("metadata") or {}).get("ts") for m in ordered]
        known_ts = [t for t in known_ts if t]
        boundary_ts = max(known_ts, key=float) if known_ts else (prior or {}).get("boundary_ts")
        if not boundary_ts:
            self.log_warning(
                f"Compaction dropped {len(dropped)} message(s) with no known ts and no prior "
                f"summary for {thread_key} — skipping summary write (span lost, as pre-Phase-S)"
            )
            return

        span_lines = []
        for m in ordered:
            role = m.get("role", "user")
            text = self._content_to_text(m.get("content"))
            if len(text) > 2000:
                text = text[:2000] + " […truncated]"
            span_lines.append(f"{role}: {text}")
        span_text = "\n".join(span_lines)

        prior_text = (prior or {}).get("summary_text") or ""
        try:
            from prompts import CONVERSATION_SUMMARIZATION_PROMPT
            user_block = (
                (f"EXISTING SUMMARY OF OLDER MESSAGES:\n{prior_text}\n\n" if prior_text else "")
                + f"NEW MESSAGES TO FOLD IN:\n{span_text}"
            )
            summary_text = await self.openai_client.create_text_response(
                messages=[
                    {"role": "developer", "content": CONVERSATION_SUMMARIZATION_PROMPT},
                    {"role": "user", "content": user_block},
                ],
                model=config.utility_model,
                temperature=0.3,
                max_tokens=1200,
                system_prompt=None
            )
            summary_text = (summary_text or "").strip()
            if not summary_text:
                raise ValueError("empty summary")
        except Exception as e:
            self.log_warning(f"Span summarization failed ({e}) — using deterministic fallback")
            fallback = "(Earlier messages were removed to manage context length.)"
            summary_text = f"{prior_text}\n{fallback}".strip() if prior_text else fallback

        # Merge refs: prior refs + refs from the dropped span, deduped, deterministically ordered
        refs = {(r.get("kind"), r.get("value")): r for r in (prior or {}).get("refs", [])}
        for r in self._extract_refs_from_messages(ordered):
            refs.setdefault((r.get("kind"), r.get("value")), r)
        refs_list = sorted(refs.values(), key=lambda r: (r.get("kind") or "", r.get("value") or ""))

        try:
            await self.db.save_thread_summary_async(thread_key, summary_text, str(boundary_ts), refs_list)
        except Exception as e:
            self.log_error(f"Failed to persist thread summary for {thread_key}: {e}")
            return

        self._upsert_summary_head_in_state(thread_state, summary_text, refs_list)

    @classmethod
    def _build_summary_head_content(cls, summary_text: str, refs: Optional[List[Dict]]) -> str:
        """Render the summary head message. MUST be deterministic for a given DB row —
        no timestamps or counts — so rebuilds serialize identically (prompt-cache hygiene)."""
        parts = [
            "--- SUMMARY OF EARLIER CONVERSATION ---",
            summary_text,
        ]
        if refs:
            parts.append("References from the summarized span:")
            for r in refs:
                kind = r.get("kind") or "link"
                value = r.get("value") or ""
                name = r.get("name")
                parts.append(f"- [{kind}] {name + ': ' if name else ''}{value}")
        parts.append("--- END SUMMARY ---")
        return "\n".join(parts)

    def _upsert_summary_head_in_state(self, thread_state, summary_text: str, refs: Optional[List[Dict]]):
        """Create or update the summary head message at position 0 of the live state."""
        content = self._build_summary_head_content(summary_text, refs)
        for msg in thread_state.messages:
            if (msg.get("metadata") or {}).get("type") == self.SUMMARY_HEAD_MARKER:
                msg["content"] = content
                break
        else:
            thread_state.messages.insert(0, {
                "role": "developer",
                "content": content,
                "metadata": {"type": self.SUMMARY_HEAD_MARKER},
            })
        thread_state.has_summary_head = True

    def _tracked_context_tokens(self, thread_state):
        """The usage-tracked context size as an int, or None when unavailable
        (falls back to a chars/4 estimate at the call site)."""
        try:
            value = int(getattr(thread_state, "context_tokens", 0))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    @staticmethod
    def _is_context_length_error(e) -> bool:
        """Detect the API's context-window-exceeded error (backstop for the chars/4
        estimator — compact + retry once instead of failing the response)."""
        s = str(e).lower()
        return ("context_length_exceeded" in s
                or "maximum context length" in s
                or "context window" in s
                or getattr(e, "code", None) == "context_length_exceeded")

    @staticmethod
    def _render_reactions_annotation(reactions) -> str:
        """Render a message's reactions (from conversations.replies) as a compact,
        DETERMINISTIC annotation line: stable ordering (emoji name, then count), no
        timestamps — two rebuilds of the same history must serialize identically.
        Reactor IDs are rendered as <@UID> mentions; the roster maps them to names."""
        if not reactions:
            return ""
        entries = []
        for r in sorted(reactions, key=lambda r: (r.get("name") or "", r.get("count") or 0)):
            name = r.get("name")
            if not name:
                continue
            count = r.get("count") or len(r.get("users") or [])
            users = ", ".join(f"<@{u}>" for u in sorted(r.get("users") or []))
            entries.append(f":{name}: x{count}" + (f" ({users})" if users else ""))
        return f"[reactions: {'; '.join(entries)}]" if entries else ""

    @staticmethod
    def _extract_refs_from_messages(messages: List[Dict]) -> List[Dict]:
        """Pull structured refs (files/images/links) out of a span of messages."""
        import re
        refs = []
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        for m in messages:
            meta = m.get("metadata") or {}
            if meta.get("url"):
                kind = "image" if str(meta.get("type", "")).startswith("image") else "file"
                refs.append({"kind": kind, "value": meta["url"], "name": meta.get("filename")})
            if meta.get("filename") and not meta.get("url"):
                refs.append({"kind": "file", "value": meta["filename"], "name": meta.get("filename")})
            text = m.get("content") if isinstance(m.get("content"), str) else ""
            for url in re.findall(url_pattern, text or ""):
                kind = "image" if any(ext in url.lower() for ext in
                                      ('.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg')) else "link"
                refs.append({"kind": kind, "value": url, "name": None})
        return refs

    @staticmethod
    def _content_to_text(content) -> str:
        """Flatten a message content value (str or multimodal list of parts) to plain text."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    txt = part.get("text")
                    if txt:
                        parts.append(txt)
                elif isinstance(part, str):
                    parts.append(part)
            return " ".join(parts)
        return str(content) if content is not None else ""

    async def _async_extract_channel_memory(self, thread_state):
        """Phase 9: after a response is sent, run ONE lightweight utility-model call to decide whether
        the latest exchange holds a durable channel fact, and persist/update it. Best-effort, runs
        post-response (never blocks the reply), only writes channel-scope rows, enforces the row cap."""
        if not config.enable_channel_memory:
            return
        channel_id = getattr(thread_state, "channel_id", None)
        db = getattr(self, "db", None)
        openai_client = getattr(self, "openai_client", None)
        if not channel_id or not db or not openai_client:
            return

        msgs = getattr(thread_state, "messages", None) or []
        last_user = next((m for m in reversed(msgs) if m.get("role") == "user"), None)
        last_assistant = next((m for m in reversed(msgs) if m.get("role") == "assistant"), None)
        if not last_user or not last_assistant:
            return
        exchange = (
            f"User: {self._content_to_text(last_user.get('content'))}\n"
            f"Assistant: {self._content_to_text(last_assistant.get('content'))}"
        )

        try:
            existing = await db.get_channel_memory_async(channel_id)
        except Exception:
            existing = []
        existing_min = [{"id": r["id"], "content": r["content"]} for r in existing]

        decision = await openai_client.extract_memory(exchange, existing_min)
        action = (decision or {}).get("action", "none")

        if action == "add" and decision.get("content"):
            # Enforce per-channel cap by evicting oldest channel-scope rows first.
            chan_rows = [r for r in existing if r.get("scope") == "channel"]
            cap = max(1, config.memory_max_rows)
            while len(chan_rows) >= cap:
                oldest = min(chan_rows, key=lambda r: r.get("updated_ts") or "")
                await db.delete_channel_memory_async(oldest["id"])
                chan_rows.remove(oldest)
            await db.add_channel_memory_async(channel_id, decision["content"], scope="channel")
            self.log_info(f"Channel memory: recorded a durable fact for {channel_id}")
        elif action == "update" and decision.get("id") is not None and decision.get("content"):
            await db.update_channel_memory_async(decision["id"], decision["content"])
            self.log_info(f"Channel memory: updated fact {decision['id']} for {channel_id}")

    async def _async_post_response_cleanup(self, thread_state, thread_key: str):
        """Asynchronously clean up thread after response is sent
        
        This runs after the response has been sent to Slack to proactively
        trim old messages before the next request. Will summarize documents
        before trimming them to preserve context.
        
        Args:
            thread_state: Thread state to potentially clean up
            thread_key: Thread identifier for database operations
        """
        # Memory writes are model-invoked tools now (Phase C); the post-response extractor
        # survives one release behind ENABLE_MEMORY_EXTRACTION_FALLBACK (default off) in case
        # tool-driven writes under-perform. Best-effort; isolated from token cleanup.
        if config.enable_memory_extraction_fallback:
            try:
                await self._async_extract_channel_memory(thread_state)
            except Exception as e:
                self.log_debug(f"Channel memory extraction skipped: {e}")

        try:
            # Get current model's token limit
            model = thread_state.current_model or config.gpt_model
            max_tokens = config.get_model_token_limit(model)
            cleanup_threshold = int(max_tokens * config.token_cleanup_threshold)
            
            # Usage-driven budgeting: the tracked number (API usage + increment
            # estimates) is authoritative; fall back to a fresh estimate if unset.
            current_tokens = self._tracked_context_tokens(thread_state) or \
                self.thread_manager._token_counter.count_thread_tokens(thread_state.messages)

            # Cost visibility: >272K input bills at 2x input / 1.5x output on 5.5/5.6.
            # Log once per thread when it crosses the tier (log only — never block).
            if config.is_long_context(current_tokens) and \
                    not getattr(thread_state, "_long_context_logged", False):
                try:
                    thread_state._long_context_logged = True
                except Exception:
                    pass
                self.log_info(
                    f"Thread crossed the long-context billing tier: {current_tokens:,} input tokens "
                    f"> {config.LONG_CONTEXT_BILLING_THRESHOLD:,} (2x input / 1.5x output pricing applies)"
                )

            if current_tokens > cleanup_threshold:
                self.log_info(f"Thread at {current_tokens}/{max_tokens} tokens ({current_tokens/max_tokens:.1%}), triggering compaction")

                # Phase S: one chunky compaction down to the target (not a small per-turn
                # trim) — dropped span is folded into the rolling thread summary. No DB
                # message mirror to maintain anymore.
                processed = await self._compact_thread_to_target(thread_state, thread_key)

                if processed > 0:
                    new_tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages)
                    self.log_info(f"Compaction complete: {current_tokens} → {new_tokens} tokens ({processed} messages processed)")
                else:
                    self.log_warning("Compaction triggered but no trimmable messages found")
            
        except Exception as e:
            self.log_error(f"Error during async cleanup: {e}")
            # Don't let cleanup errors affect the main flow

    @staticmethod
    def _history_sender_type(msg: Message) -> str:
        """sender_type with the same back-compat fallback the rebuild loop uses."""
        st = (msg.metadata or {}).get("sender_type")
        if st is None:
            st = "self" if (msg.metadata or {}).get("is_bot") else "human"
        return st

    def _merge_continuation_history(self, history: List[Message]) -> List[Message]:
        """Collapse a split bot reply (part 1 + 'Continued...' parts) into ONE message.

        Consecutive own-bot messages are merged when the earlier ends with a
        continuation trailer or the later starts as a continuation part; markers
        are stripped everywhere on own-bot messages so rebuilt assistant turns
        never contain them (R2 — the model would imitate markers it sees itself
        emitting). The merged message keeps the FIRST part's ts; attachments and
        reactions from all parts are combined deterministically (part order).
        """
        merged: List[Message] = []
        for msg in history:
            if merged and self._history_sender_type(msg) == "self":
                prev = merged[-1]
                if self._history_sender_type(prev) == "self" and (
                    ends_with_continuation(prev.text) or starts_as_continuation(msg.text)
                ):
                    prev_text = strip_continuation_markers(prev.text)
                    cur_text = strip_continuation_markers(msg.text)
                    prev.text = f"{prev_text}\n\n{cur_text}".strip() if prev_text and cur_text \
                        else (prev_text or cur_text)
                    prev.attachments = (prev.attachments or []) + (msg.attachments or [])
                    prev_reactions = (prev.metadata or {}).get("reactions") or []
                    cur_reactions = (msg.metadata or {}).get("reactions") or []
                    if prev_reactions or cur_reactions:
                        prev.metadata["reactions"] = prev_reactions + cur_reactions
                    continue
            merged.append(msg)

        # Stray markers on unmerged own-bot messages (e.g. only the tail part is inside
        # the fetch window, or a part-1 whose continuation never posted) still get
        # stripped so they can't leak into assistant turns.
        for msg in merged:
            if self._history_sender_type(msg) == "self" and msg.text:
                stripped = strip_continuation_markers(msg.text)
                if stripped != msg.text:
                    msg.text = stripped
        return merged

    async def _fetch_thread_root(self, client, channel_id: str, thread_id: str):
        """F3: fetch a thread's ROOT message only (summary-tail rebuild, where the root is
        before the fetched window). A limit=1 replies page with no `oldest` returns the
        root. Best-effort — returns None on any failure (the wake role just gets omitted)."""
        if not channel_id or not thread_id or not hasattr(client, "get_thread_history"):
            return None
        try:
            page = await client.get_thread_history(channel_id, thread_id, limit=1)
            return page[0] if page else None
        except Exception as e:
            self.log_debug(f"root message fetch failed: {e}")
            return None

    async def _get_or_rebuild_thread_state(
        self,
        message: Message,
        client: BaseClient,
        thinking_id: Optional[str] = None
    ) -> Any:
        """Get existing thread state or rebuild from platform history"""
        thread_state = await self.thread_manager.get_or_create_thread_async(
            message.thread_id,
            message.channel_id
        )
        
        # If thread has no messages, rebuild from platform
        # Also rebuild if we have messages but no images in DB (to extract image URLs)
        should_rebuild = not thread_state.messages
        # A busy-rejected message never entered this warm state — Slack has it, we
        # don't. Full refetch (Slack is the transcript; same path as a cold rebuild,
        # so ts-dedup and summary-head composition behave identically).
        refresh_key = f"{message.channel_id}:{message.thread_id}"
        if not should_rebuild and self.thread_manager.consume_needs_refresh(refresh_key):
            self.log_info(f"Warm thread {refresh_key} flagged needs_refresh (busy-rejected "
                          f"message) — refetching transcript from Slack")
            should_rebuild = True
        if not should_rebuild and self.db:
            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
            db_images = await self.db.find_thread_images_async(thread_key)
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

            # Phase S: Slack is the only transcript. A rebuild always starts from a clean
            # slate (fresh fetch is authoritative — edited/deleted messages must not
            # survive from stale in-memory state) and composes:
            # stored summary head (compacted older span) + fresh conversations.replies tail.
            if thread_state.messages:
                thread_state.messages.clear()
                thread_state.has_summary_head = False
            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
            summary_row = None
            if self.db:
                try:
                    summary_row = await self.db.get_thread_summary_async(thread_key)
                except Exception as e:
                    self.log_warning(f"Could not load thread summary for rebuild: {e}")

            summary_boundary = None
            if summary_row:
                try:
                    summary_boundary = float(summary_row["boundary_ts"])
                except (TypeError, ValueError):
                    summary_boundary = None
                self._upsert_summary_head_in_state(
                    thread_state, summary_row["summary_text"], summary_row.get("refs")
                )

            # Get history from platform first to see if there's anything to rebuild.
            # With a valid summary boundary, fetch only the tail (strictly after the
            # boundary — Slack's default inclusive=false); the Python <= filter below
            # stays as belt-and-suspenders for the seam. A HistoryFetchError propagates
            # up so the turn fails loudly instead of answering with amnesia (R1).
            history = await client.get_thread_history(
                message.channel_id,
                message.thread_id,
                oldest=(summary_row["boundary_ts"] if summary_boundary is not None else None)
            )

            # Merge split bot replies ("Continued..." parts) back into single turns and
            # strip the markers — otherwise the model sees itself emitting continuation
            # markers and may imitate them (R2).
            history = self._merge_continuation_history(history)
            
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
                        pipeline_status("rebuilding_history", "Rebuilding thread history from Slack…"),
                        emoji=config.circle_loader_emoji
                    )
                self.log_info(f"Rebuilding thread state for {message.thread_id} with {len(history)} messages")
            
            # F3: capture the thread's ROOT author for the wake envelope.
            # Full-history rebuild (and a brand-new thread, whose only history IS the
            # current message): the earliest fetched message is the root. Summary-tail
            # rebuild: the root is before the fetched window — fetch it explicitly, once.
            # Guarded by the feature flag so a disabled envelope makes no extra API call.
            if config.enable_wake_envelope and thread_state.root_author is None:
                try:
                    if summary_boundary is None and history:
                        root_hist = history[0]
                        rt = root_hist.metadata.get("sender_type") or (
                            "self" if root_hist.metadata.get("is_bot") else "human")
                        thread_state.root_author = (root_hist.user_id, rt)
                    elif summary_boundary is not None:
                        root_msg = await self._fetch_thread_root(client, message.channel_id,
                                                                 message.thread_id)
                        if root_msg is not None:
                            rt = root_msg.metadata.get("sender_type") or (
                                "self" if root_msg.metadata.get("is_bot") else "human")
                            thread_state.root_author = (root_msg.user_id, rt)
                except Exception as e:
                    self.log_debug(f"root author capture failed: {e}")

            # Track pending image URLs for vision analysis association
            pending_image_urls = []
            pending_image_metadata = {}  # Store additional metadata per URL

            # F7: batch-fetch this thread's tool-use provenance once, keyed by reply ts, to
            # reinject "[used tools: …]" onto matching assistant turns during the loop below.
            # Messages at/behind the summary boundary are already skipped, so nothing behind
            # a compaction boundary is ever annotated.
            tool_usage_by_ts: Dict[str, list] = {}
            if config.enable_tool_provenance and self.db:
                try:
                    tool_usage_by_ts = await self.db.get_thread_tool_usage_async(thread_key)
                except Exception as e:
                    self.log_debug(f"tool-usage fetch for rebuild failed: {e}")

            # Convert to thread state messages
            for hist_msg in history:
                # Skip the current message being processed
                if hist_msg.metadata.get("ts") == current_ts:
                    continue

                # Skip messages already covered by the stored summary head (<= boundary).
                # Anything after the boundary composes the fresh tail — never duplicated.
                if summary_boundary is not None:
                    hist_ts = hist_msg.metadata.get("ts")
                    try:
                        if hist_ts is not None and float(hist_ts) <= summary_boundary:
                            continue
                    except (TypeError, ValueError):
                        pass
                    
                # Determine sender and role.
                # Only OUR OWN messages are assistant turns. Humans AND other bots (e.g. another
                # AI bot sharing the thread) are user turns — otherwise another bot's messages get
                # replayed to the model as if we had said them.
                is_bot = hist_msg.metadata.get("is_bot", False)
                sender_type = hist_msg.metadata.get("sender_type")
                if sender_type is None:
                    # Back-compat for history captured before sender_type existed
                    sender_type = "self" if is_bot else "human"
                is_self = (sender_type == "self")
                role = "assistant" if is_self else "user"

                # Build content with attachment info
                content = hist_msg.text

                # For non-self messages, prefix with a display name (humans by username,
                # other bots by their bot name) so the model knows who is speaking.
                if not is_self:
                    if sender_type == "other_bot":
                        username = (hist_msg.metadata.get("bot_name")
                                    or hist_msg.metadata.get("username")
                                    or "Bot")
                    else:
                        # Get username from metadata (should be populated by client)
                        username = hist_msg.metadata.get("username") if hist_msg.metadata else None

                        # If no username in metadata, fetch it from user_id
                        if not username and hist_msg.user_id:
                            # Use client's get_username method if available
                            if hasattr(client, 'get_username'):
                                # Get the slack_client from metadata if available
                                slack_client = hist_msg.metadata.get("slack_client") if hist_msg.metadata else None
                                if slack_client:
                                    username = await client.get_username(hist_msg.user_id, slack_client)
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

                    # Track real participants for the @mention roster (other bots use the
                    # placeholder "bot"/"unknown" user_id and are skipped here)
                    if hist_msg.user_id and hist_msg.user_id not in ("bot", "unknown"):
                        thread_state.participants[hist_msg.user_id] = username
                
                # F7: for OUR OWN turns, strip the external _Used Tools:_ footer (never model
                # context) then append the deterministic [used tools: …] annotation from the
                # persisted rows — pinned order: footer-strip → used-tools → reactions, so no
                # trailing annotation can shield the footer from stripping. Guarded by the
                # flag so config-off leaves rebuilt content exactly as today.
                if config.enable_tool_provenance and is_self:
                    content = strip_used_tools_footer(content)
                    used_note = render_used_tools_annotation(
                        tool_usage_by_ts.get(hist_msg.metadata.get("ts")))
                    if used_note:
                        content = f"{content}\n{used_note}" if content else used_note

                # Reactions on this message (from conversations.replies) — deterministic
                # annotation so the model knows who reacted with what
                reactions_note = self._render_reactions_annotation(hist_msg.metadata.get("reactions"))
                if reactions_note:
                    content = f"{content}\n{reactions_note}" if content else reactions_note

                # F10: prefix a deterministic per-message timestamp so the model can reason
                # about elapsed time between turns. A pure PREFIX — applied AFTER the pinned
                # end-anchored suffix annotations above, so it never disturbs the footer-strip
                # → [used tools:] → [reactions:] order. Rendered from the message's immutable
                # ts in the SENDER's cached timezone (self turns and unknown/other-bot senders
                # fall back to UTC — no per-message API lookup). Guarded so config-off leaves
                # rebuilt content byte-identical.
                if config.enable_message_timestamps:
                    stamp_tz = "UTC" if is_self else sender_timezone(
                        hist_msg.metadata, hist_msg.user_id, getattr(client, "user_cache", None))
                    content = stamp_content(content, hist_msg.metadata.get("ts"), stamp_tz)

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
                                        await self.db.save_image_metadata_async(
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
                                        await self.db.save_image_metadata_async(
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
                                # Handle document attachments during rebuild (Phase D2):
                                # the stored summary row is the fast path — no download,
                                # no re-extraction. Only legacy threads without a row
                                # fall back to download + extract (summary-only inject).
                                self.log_info(f"Found document attachment during rebuild: {filename} ({mimetype})")

                                message_metadata["filename"] = filename
                                message_metadata["mimetype"] = mimetype

                                try:
                                    thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
                                    doc_row = None
                                    try:
                                        doc_row = await self.db.get_document_by_filename_async(thread_key, filename)
                                    except Exception as row_err:
                                        self.log_debug(f"Doc row lookup failed for {filename}: {row_err}")

                                    if doc_row and doc_row.get("summary"):
                                        # Stored summary + ref — zero API calls
                                        content = self._build_message_with_documents(
                                            content,
                                            [{
                                                "filename": filename,
                                                "mimetype": mimetype,
                                                "summary": doc_row.get("summary"),
                                                "total_pages": doc_row.get("total_pages"),
                                                "size_bytes": doc_row.get("size_bytes"),
                                                "file_id": doc_row.get("file_id"),
                                            }]
                                        )
                                        self.log_info(f"Injected stored summary during rebuild: {filename}")
                                    else:
                                        # Legacy thread (no row): derive once, store
                                        # summary + ref, inject the summary block
                                        document_data = await client.download_file(att_url, attachment.get("id"))
                                        if document_data and self.document_handler:
                                            extracted_content = await self.document_handler.safe_extract_content_async(
                                                document_data, mimetype, filename,
                                                ocr_images=False  # summary-only injection; no page images needed
                                            )
                                            if extracted_content and extracted_content.get("content"):
                                                doc_summary = await self._summarize_document_for_attach(
                                                    extracted_content, filename, mimetype)
                                                content = self._build_message_with_documents(
                                                    content,
                                                    [{
                                                        "filename": filename,
                                                        "mimetype": mimetype,
                                                        "summary": doc_summary,
                                                        "total_pages": extracted_content.get("total_pages"),
                                                        "size_bytes": len(document_data),
                                                        "file_id": attachment.get("id"),
                                                    }]
                                                )
                                                document_ledger = self.thread_manager.get_or_create_document_ledger(thread_state.thread_ts)
                                                document_ledger.add_document(
                                                    content=extracted_content["content"],  # transient
                                                    filename=filename,
                                                    mime_type=mimetype,
                                                    summary=doc_summary,
                                                    total_pages=extracted_content.get("total_pages"),
                                                    page_structure=extracted_content.get("page_structure"),
                                                    metadata=None,
                                                    db=self.db,
                                                    thread_id=thread_key,
                                                    message_ts=message_metadata.get("ts"),
                                                    file_id=attachment.get("id"),
                                                    url_private=att_url,
                                                    size_bytes=len(document_data),
                                                )
                                                self.log_info(f"Derived + stored summary during rebuild: {filename}")
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
                                    await self.db.save_image_metadata_async(
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
        
        # Pre-flight compaction decision after rebuild. Cold rebuild has no usage
        # number yet — the whole assembled context is ESTIMATED at chars/4 (crude is
        # fine under TOKEN_BUFFER_PERCENTAGE headroom; the context_length_exceeded
        # backstop catches estimator edge cases like document-heavy rebuilds).
        model = thread_state.current_model or config.gpt_model
        max_tokens = config.get_model_token_limit(model)
        try:
            thread_state.reset_context_estimate(self.thread_manager._token_counter)
        except Exception:
            pass  # non-standard thread_state (tests); estimate below regardless
        current_tokens = self._tracked_context_tokens(thread_state) or \
            self.thread_manager._token_counter.count_thread_tokens(thread_state.messages)
        
        if current_tokens > max_tokens:
            self.log_info(f"Thread rebuilt over limit ({current_tokens}/{max_tokens} tokens), compacting")

            # Update status to show we're compacting
            if thinking_id:
                self._update_status(
                    client,
                    message.channel_id,
                    thinking_id,
                    pipeline_status("optimizing_history", f"Optimizing conversation history ({current_tokens:,}/{max_tokens:,} tokens)…"),
                    emoji=config.circle_loader_emoji
                )

            # Phase S: one chunky compaction to target; the dropped span rolls into the
            # thread summary (DB) and the summary head message updates in place.
            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
            total_trimmed = await self._compact_thread_to_target(thread_state, thread_key)
            current_tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages)

            if total_trimmed > 0:
                self.log_info(f"Compaction during rebuild complete: {total_trimmed} messages processed, final: {current_tokens}/{max_tokens} tokens")
        
        # Log final token count
        final_tokens = self.thread_manager._token_counter.count_thread_tokens(thread_state.messages)
        self.log_info("="*100)
        self.log_info(f"THREAD STATE | Messages: {len(thread_state.messages)} | Tokens: {final_tokens}/{max_tokens}")
        self.log_info("="*100)
        
        return thread_state
