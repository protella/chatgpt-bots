from __future__ import annotations

import asyncio
import datetime
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import base64
import os
import re
import pytz

from base_client import BaseClient, Message
from config import config, pipeline_status
from prompts import SLACK_SYSTEM_PROMPT, CLI_SYSTEM_PROMPT, LOCAL_TOOLS_GUIDANCE


def build_roster_text(participants, user_cache=None, bot_user_id=None):
    """Build a participant roster block mapping display name -> <@USER_ID> for the system prompt.

    participants: dict of user_id -> display name (thread participants). user_cache (optional)
    is used to improve names. Returns "" when there is no real participant to tag.
    """
    cache = user_cache or {}
    entries = []
    seen = set()
    for uid, name in (participants or {}).items():
        if not uid or uid in ("bot", "unknown"):
            continue
        if bot_user_id and uid == bot_user_id:
            continue
        if uid in seen:
            continue
        seen.add(uid)
        info = cache.get(uid)
        if isinstance(info, dict) and info.get("username"):
            name = info.get("username")
        entries.append((name or uid, uid))
    if not entries:
        return ""
    lines = "\n".join(f"- {name} → <@{uid}>" for name, uid in entries)
    return (
        "\n\nTHREAD PARTICIPANTS — to mention or tag someone, write their Slack ID in the form "
        "<@USER_ID> (exactly, with the angle brackets). Never put a person's plain name inside "
        "angle brackets. Known participants:\n" + lines
    )


class MessageUtilitiesMixin:
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

    def _build_user_content(self, text: str, image_inputs: List[Dict],
                            file_inputs: Optional[List[Dict]] = None) -> Any:
        """Build user message content.

        file_inputs are native input_file content parts (Phase D2): per-request
        base64 PDFs the model reads directly (text + rendered pages). They ride
        only the attach turn — thread state keeps the summary breadcrumb.
        """
        if image_inputs or file_inputs:
            content = [{"type": "input_text", "text": text}]
            content.extend(image_inputs or [])
            content.extend(file_inputs or [])
            return content
        else:
            # Simple text content
            return text

    def _native_file_eligible(self, mimetype: str, size_bytes: int,
                              total_pages: Optional[int]) -> bool:
        """Decide native input_file vs local extraction for the attach turn.

        Native (model sees text + rendered pages) requires ALL of:
        - ENABLE_NATIVE_FILE_INPUT on
        - PDF (the API renders pages only for PDFs; other types go local)
        - within the API request ceilings (<= NATIVE_FILE_MAX_MB, and
          <= NATIVE_FILE_MAX_PAGES when the page count is known)
        Everything else — non-PDF types, oversized PDFs, unknown-page scans
        over the limit, flag off — uses the local extraction path, which is a
        permanent first-class citizen (it also serves read_document).
        """
        if not config.enable_native_file_input:
            return False
        if mimetype != "application/pdf":
            return False
        if size_bytes > config.native_file_max_mb * 1024 * 1024:
            return False
        if total_pages is not None and total_pages > config.native_file_max_pages:
            return False
        return True

    def _build_spreadsheet_schema_block(self, extracted: Dict, filename: str) -> str:
        """Schema-first spreadsheet summary: sheets, columns, row counts, sample rows.

        Deterministic (no model call) — full data is reachable via read_document.
        """
        lines = []
        page_structure = extracted.get("page_structure") or {}
        sheets = page_structure.get("sheets") if isinstance(page_structure, dict) else None
        content = extracted.get("content") or ""
        if sheets:
            lines.append(f"Sheets ({len(sheets)}): {', '.join(list(sheets.keys())[:10])}")
            for name, info in list(sheets.items())[:10]:
                if isinstance(info, dict):
                    rows = info.get("rows") or info.get("row_count")
                    cols = info.get("columns") or info.get("column_names")
                    desc = []
                    if rows is not None:
                        desc.append(f"{rows} rows")
                    if isinstance(cols, list):
                        desc.append("columns: " + ", ".join(str(c) for c in cols[:15]))
                    if desc:
                        lines.append(f"- {name}: {'; '.join(desc)}")
        # Sample: first ~5 non-empty content lines (extraction renders markdown tables)
        sample = [ln for ln in content.splitlines() if ln.strip()][:7]
        if sample:
            lines.append("Sample (first rows):")
            lines.extend(sample)
        lines.append("(Schema and sample only — query full data via read_document.)")
        return "\n".join(lines)

    async def _summarize_document_for_attach(self, extracted: Dict, filename: str,
                                             mimetype: str) -> str:
        """Attach-time summary — the ONLY content-bearing field that persists.

        Spreadsheets get a deterministic schema-first block (no model call);
        other documents get a gap-honest utility-model summary. Any failure
        falls back to a labeled excerpt so the row is never contentless.
        """
        content = extracted.get("content") or ""
        page_structure = extracted.get("page_structure") or {}
        is_spreadsheet = isinstance(page_structure, dict) and "sheets" in page_structure
        if is_spreadsheet:
            try:
                return self._build_spreadsheet_schema_block(extracted, filename)
            except Exception as e:
                self.log_warning(f"Schema block failed for {filename}: {e}")
        try:
            from prompts import DOCUMENT_SUMMARIZATION_PROMPT
            # Bound the summarizer's input (utility window guard, chars/4 heuristic)
            summary = await self.openai_client.create_text_response(
                messages=[
                    {"role": "developer", "content": DOCUMENT_SUMMARIZATION_PROMPT},
                    {"role": "user", "content": content[:1_000_000]},
                ],
                model=config.utility_model,
                temperature=0.3,
                max_tokens=800,
                system_prompt=None,
            )
            if summary and summary.strip():
                return summary.strip()
        except Exception as e:
            self.log_warning(f"Attach-time summarization failed for {filename}: {e}")
        return ("[excerpt of original — full document available via read_document]\n"
                + content[:1500])

    def _apply_scanned_pdf_ocr(self, extracted_content: Dict, mimetype: str,
                               file_name: str, image_inputs: List[Dict],
                               image_count: int, max_images: int) -> int:
        """Legacy scanned-PDF OCR via page images (ISOLATED — local path only).

        Runs only when a scanned PDF does NOT ride the native input_file route
        (flag off, or PDF over the native size/page limits). Native page
        rendering covers scans without this. Slated for retirement once native
        input is validated in prod. Returns the updated image_count.
        """
        if not (extracted_content.get("is_image_based") and mimetype == "application/pdf"):
            return image_count
        self.log_info(f"PDF {file_name} appears to be image-based (scanned document)")
        if extracted_content.get("page_images"):
            self.log_info(f"PDF has {len(extracted_content['page_images'])} page images for OCR")
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
            extracted_content["content"] = (
                f"[PDF {file_name}: {extracted_content.get('total_pages', 'unknown')} pages total. "
                f"This appears to be a scanned document. "
                f"Using vision/OCR on {len(extracted_content['page_images'])} page(s) for text extraction.]"
            )
            extracted_content["ocr_processed"] = True
        else:
            extracted_content["warning"] = "This PDF appears to be a scanned document with minimal extractable text"
        return image_count

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

        # Phase D2: inject the labeled SUMMARY, never the full content — the
        # model reaches full fidelity via read_document (and, for eligible PDFs,
        # the native input_file part riding this same turn). Deterministic
        # rendering: summaries are stable once written (cache hygiene).
        for doc in document_inputs:
            filename = doc.get("filename", "unknown_document")
            mimetype = doc.get("mimetype", "unknown")
            summary = doc.get("summary")
            if not summary:
                # Never render full content; fall back to a labeled excerpt
                content = doc.get("content") or ""
                summary = ("[excerpt of original — full document available via read_document]\n"
                           + content[:1500]) if content else "[Document content not available]"
            total_pages = doc.get("total_pages")
            size_bytes = doc.get("size_bytes")
            file_id = doc.get("file_id")

            header = f"\n\n=== DOCUMENT SUMMARY: {filename} ==="
            details = [mimetype]
            if total_pages:
                details.append(f"{total_pages} pages")
            if size_bytes:
                details.append(f"{size_bytes:,} bytes")
            if file_id:
                details.append(f"file_id: {file_id}")
            header += f"\n({'; '.join(details)} — full content available via read_document)\n"

            message_parts.append(header)
            message_parts.append(str(summary))
            message_parts.append(f"=== END DOCUMENT SUMMARY: {filename} ===")
        
        # Ensure all parts are strings before joining
        str_parts = []
        for i, part in enumerate(message_parts):
            if not isinstance(part, str):
                self.log_warning(f"message_parts[{i}] is not a string: {type(part)}")
                str_parts.append(str(part))
            else:
                str_parts.append(part)
        
        return "\n".join(str_parts)

    def _extract_slack_file_urls(self, text: str) -> List[str]:
        """Extract Slack file URLs from message text
        
        Args:
            text: Message text that may contain Slack file URLs
            
        Returns:
            List of Slack file URLs found
        """
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

    async def _process_attachments(
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
                    image_data = await client.download_file(
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
                                await self.db.save_image_metadata_async(
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
                    else:
                        # Silently answering as if the file were never attached
                        # reads as the bot being obtuse — surface the failure.
                        self.log_warning(f"Failed to download image attachment: {file_name}")
                        unsupported_files.append({
                            "name": file_name,
                            "type": "image",
                            "mimetype": attachment.get("mimetype", "unknown"),
                            "error": "download_failed"
                        })

                except Exception as e:
                    self.log_error(f"Error processing attachment: {e}")
                    unsupported_files.append({
                        "name": file_name,
                        "type": "image",
                        "mimetype": attachment.get("mimetype", "unknown"),
                        "error": "download_failed"
                    })
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
                                          pipeline_status("processing_document", f"Processing {file_name}…", file_name=file_name), 
                                          emoji=config.analyze_emoji, thread_id=message.thread_id)
                    
                    # Download the document
                    document_data = await client.download_file(
                        attachment.get("url"),
                        file_id
                    )
                    
                    if document_data:
                        # Update status to show we're extracting content
                        if thinking_id:
                            self._update_status(client, message.channel_id, thinking_id, 
                                              pipeline_status("extracting_document", f"Extracting content from {file_name}…", file_name=file_name), 
                                              emoji=config.analyze_emoji, thread_id=message.thread_id)
                        
                        # Extract document content using DocumentHandler. Pre-extraction
                        # native screen (flag + PDF + size): when a PDF may ride the
                        # native input_file route, skip OCR page-image conversion —
                        # the model gets rendered pages from the API itself. (If the
                        # page count then disqualifies it, the rare oversized scan
                        # falls back to local extraction without page images.)
                        maybe_native = (
                            config.enable_native_file_input
                            and mimetype == "application/pdf"
                            and len(document_data) <= config.native_file_max_mb * 1024 * 1024
                        )
                        extracted_content = await self.document_handler.safe_extract_content_async(
                            document_data, mimetype, file_name,
                            ocr_images=not maybe_native
                        )
                        
                        if extracted_content and extracted_content.get("content"):
                            # Native-vs-local decision (one place, documented in
                            # _native_file_eligible). Local extraction already ran —
                            # it always does: it feeds the summary, page count,
                            # spreadsheet schema, and warms the read_document cache.
                            native = self._native_file_eligible(
                                mimetype, len(document_data),
                                extracted_content.get("total_pages"))

                            if native:
                                # Model reads the actual PDF this turn (text +
                                # rendered pages) — the legacy OCR path is not
                                # needed for scans on this route.
                                file_data_b64 = base64.b64encode(document_data).decode("ascii")
                            else:
                                file_data_b64 = None
                                # Legacy scanned-PDF OCR (flag-off / oversized PDFs
                                # only) — isolated here; slated for retirement.
                                image_count = self._apply_scanned_pdf_ocr(
                                    extracted_content, mimetype, file_name,
                                    image_inputs, image_count, max_images)

                            # Attach-time summary: the only content that persists.
                            if thinking_id:
                                self._update_status(client, message.channel_id, thinking_id,
                                                  pipeline_status("summarizing_document", f"Summarizing {file_name}…", file_name=file_name),
                                                  emoji=config.analyze_emoji, thread_id=message.thread_id)
                            doc_summary = await self._summarize_document_for_attach(
                                extracted_content, file_name, mimetype)

                            # Warm the read_document extraction LRU (in-memory only)
                            if file_id:
                                from message_processor.document_tools import _extraction_cache
                                _extraction_cache.put(file_id, extracted_content["content"])

                            document_inputs.append({
                                "filename": file_name,
                                "mimetype": mimetype,
                                # content is TRANSIENT (this turn's analysis only);
                                # it is never persisted or re-injected.
                                "content": extracted_content["content"],
                                "summary": doc_summary,
                                "native": native,
                                "file_data_b64": file_data_b64,
                                "size_bytes": len(document_data),
                                "page_structure": extracted_content.get("page_structure"),
                                "total_pages": extracted_content.get("total_pages"),
                                "metadata": extracted_content.get("metadata", {}),
                                "url": attachment.get("url"),
                                "file_id": file_id,
                                "source": "attachment",
                                "is_image_based": extracted_content.get("is_image_based", False),
                                "requires_ocr": extracted_content.get("requires_ocr", False),
                                "ocr_processed": extracted_content.get("ocr_processed", False),
                                "warning": extracted_content.get("warning")
                            })

                            # Store summary + metadata + Slack ref (never content)
                            thread_key = f"{message.channel_id}:{message.thread_id}"
                            document_ledger = self.thread_manager.get_or_create_document_ledger(message.thread_id)
                            document_ledger.add_document(
                                content=extracted_content["content"],  # transient; used only as summary fallback
                                filename=file_name,
                                mime_type=mimetype,
                                page_structure=extracted_content.get("page_structure"),
                                total_pages=extracted_content.get("total_pages"),
                                summary=doc_summary,
                                metadata=extracted_content.get("metadata", {}),
                                db=self.db,
                                thread_id=thread_key,
                                message_ts=message.metadata.get("ts") if message.metadata else None,
                                file_id=file_id,
                                url_private=attachment.get("url"),
                                size_bytes=len(document_data),
                            )

                            route = "native input_file" if native else "local extraction"
                            self.log_info(f"Processed document: {file_name} "
                                          f"({extracted_content.get('total_pages', 'unknown')} pages, {route})")
                        else:
                            self.log_warning(f"Failed to extract content from document: {file_name}")
                            # Update status to show extraction failed
                            if thinking_id:
                                error_msg = extracted_content.get("error", "Unable to extract content")
                                self._update_status(client, message.channel_id, thinking_id, 
                                                  f"⚠️ {file_name}: {error_msg}", thread_id=message.thread_id)
                            # Add to unsupported if extraction failed
                            unsupported_files.append({
                                "name": file_name,
                                "type": "file",
                                "mimetype": mimetype
                            })
                    else:
                        # Download failed — tell the user instead of answering
                        # as if the document were never attached.
                        self.log_warning(f"Failed to download document attachment: {file_name}")
                        unsupported_files.append({
                            "name": file_name,
                            "type": "file",
                            "mimetype": mimetype,
                            "error": "download_failed"
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
                    file_data = await client.download_file(url)
                    
                    if file_data:
                        if is_pdf or is_doc:
                            # Process as document
                            # Extract filename from URL
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
                                                      pipeline_status("extracting_document", f"Extracting content from {file_name}…", file_name=file_name),
                                                      emoji=config.analyze_emoji, thread_id=message.thread_id)
                                
                                # Extract content
                                extracted_content = await self.document_handler.safe_extract_content_async(
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
                        filename_match = re.search(r'/([^/?]+)(\?|$)', url)
                        unsupported_files.append({
                            "name": filename_match.group(1) if filename_match else url,
                            "type": "file",
                            "mimetype": "unknown",
                            "error": "download_failed"
                        })
            
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
            
            downloaded_images, failed_urls = await self.image_url_handler.process_urls_from_text(text_for_url_processing, auth_token)
            
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
                
                # Store the image data for potential upload to Slack later
                # This will be handled by the AssetLedger tracking
                if hasattr(message, 'url_images'):
                    message.url_images.append(img_data)
                else:
                    message.url_images = [img_data]
            
            if failed_urls:
                self.log_warning(f"Failed to download images from URLs: {', '.join(failed_urls)}")
                for failed_url in failed_urls:
                    unsupported_files.append({
                        "name": failed_url,
                        "type": "image",
                        "mimetype": "unknown",
                        "error": "download_failed"
                    })
        
        return image_inputs, document_inputs, unsupported_files

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

    async def _has_recent_image(self, thread_state) -> bool:
        """Check if there are recent images in the conversation"""
        # First check the database for ALL images in this thread (no limit)
        if self.db:
            thread_key = f"{thread_state.channel_id}:{thread_state.thread_ts}"
            thread_images = await self.db.find_thread_images_async(thread_key)
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

    def _build_participant_roster(self, thread_state, client) -> str:
        """Build the @mention roster text from thread participants + the client's user cache."""
        participants = getattr(thread_state, "participants", None) or {}
        return build_roster_text(
            participants,
            user_cache=getattr(client, "user_cache", None),
            bot_user_id=getattr(client, "bot_user_id", None),
        )

    async def _build_channel_memory_text(self, channel_id: Optional[str]) -> Optional[str]:
        """Build the CHANNEL MEMORY block from stored durable facts for this channel (Phase 9).
        Returns None when disabled, no db, or no rows — so the system prompt is unchanged."""
        if not config.enable_channel_memory or not channel_id:
            return None
        db = getattr(self, "db", None)
        if not db:
            return None
        try:
            rows = await db.get_channel_memory_async(channel_id)
        except Exception:
            return None
        if not rows:
            return None
        # [#id] prefixes let the model target update_fact/forget_fact; sorted by id
        # (not updated_ts) so the rendering is deterministic for prompt-cache hygiene.
        return "\n".join(f"- [#{r['id']}] {r['content']}" for r in sorted(rows, key=lambda r: r["id"]))

    async def _build_channel_info(self, client, channel_id: Optional[str]) -> Optional[dict]:
        """Fetch this channel's name/topic/purpose via the client's cached lookup.
        Returns None for DMs, non-Slack clients, or on any failure — prompt unchanged."""
        fetch = getattr(client, "get_channel_context", None)
        if not fetch or not channel_id:
            return None
        try:
            return await fetch(channel_id)
        except Exception:
            return None

    def _get_system_prompt(self, client: BaseClient, user_timezone: str = "UTC",
                          user_tz_label: Optional[str] = None, user_real_name: Optional[str] = None,
                          user_email: Optional[str] = None, model: Optional[str] = None,
                          web_search_enabled: bool = True, has_trimmed_messages: bool = False,
                          custom_instructions: Optional[str] = None,
                          participant_roster: Optional[str] = None,
                          channel_directives: Optional[str] = None,
                          channel_memory: Optional[str] = None,
                          channel_info: Optional[dict] = None) -> str:
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
        except Exception:
            # Fallback to UTC if timezone is invalid
            current_time = datetime.datetime.now(pytz.UTC)
            timezone_display = "UTC"
        
        # Add user's name and email if available.
        # Prompt-cache hygiene: in MULTI-USER threads (roster lists >=2 humans) this line
        # changes with every different sender, busting the prefix cache on each speaker
        # change. The roster + the "Username:" prefix on each message already identify who
        # is speaking there, so we omit it. DMs / single-user threads keep it — the sender
        # never changes within those, so the prefix stays stable.
        user_context = ""
        # Roster entries render as "- Name → <@UID>"; the instruction header also contains
        # a literal "<@USER_ID>", so count entry arrows, not raw mentions.
        multi_user_thread = (participant_roster or "").count("→ <@") >= 2
        if not multi_user_thread:
            if user_real_name and user_email:
                user_context = f"\n\nYou're speaking with {user_real_name} (email: {user_email})"
            elif user_real_name:
                user_context = f"\n\nYou're speaking with {user_real_name}"
            elif user_email:
                user_context = f"\n\nYou're speaking with user (email: {user_email})"

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

        # Phase S: summary-head note. Wording is deliberately stable/deterministic (no
        # counts, no timestamps) — this text lives in the cached prefix.
        trimming_context = ""
        if has_trimmed_messages:
            trimming_context = "\n\nNote: The beginning of this conversation has been summarized in a summary message above; file/image references from that summarized span remain available."

        # Add custom instructions if provided
        custom_instructions_context = ""
        if custom_instructions:
            custom_instructions_context = f"\n\n--- USER CUSTOM INSTRUCTIONS ---\nThe following are custom instructions provided by the user. These should be followed and may supersede any conflicting default instructions (within legal and ethical boundaries):\n\n{custom_instructions}\n\n--- END OF USER CUSTOM INSTRUCTIONS ---"

        # Where the conversation lives: channel name + topic + purpose (cached lookup;
        # None in DMs). Topics often carry load-bearing facts (links, owners, norms).
        channel_info_context = ""
        if channel_info and (channel_info.get("name") or channel_info.get("topic") or channel_info.get("purpose")):
            info_lines = []
            if channel_info.get("name"):
                info_lines.append(f"This conversation is in the #{channel_info['name']} channel.")
            if channel_info.get("topic"):
                info_lines.append(f"Channel topic: {channel_info['topic']}")
            if channel_info.get("purpose"):
                info_lines.append(f"Channel description: {channel_info['purpose']}")
            channel_info_context = "\n\n--- CHANNEL CONTEXT ---\n" + "\n".join(info_lines) + "\n--- END CHANNEL CONTEXT ---"

        # Phase 7: per-channel ground rules set by an operator (applied when present)
        channel_directives_context = ""
        if channel_directives:
            channel_directives_context = f"\n\n--- CHANNEL GROUND RULES ---\nAn operator has set ground rules for how you should behave in this channel. Follow them:\n\n{channel_directives}\n\n--- END CHANNEL GROUND RULES ---"

        # Phase 9: durable per-channel memory (facts the bot noted in earlier conversations)
        channel_memory_context = ""
        if channel_memory:
            channel_memory_context = f"\n\n--- CHANNEL MEMORY ---\nDurable facts you've noted for this channel in earlier conversations. Use them as background when relevant; do not recite them unprompted:\n\n{channel_memory}\n\n--- END CHANNEL MEMORY ---"

        # Phase A: local tool etiquette (static text — safe for prompt caching) when the
        # client exposes function tools through the loop
        local_tools_context = ""
        tool_registry = getattr(client, "tool_registry", None)
        if config.enable_tool_loop and tool_registry is not None and tool_registry.has_tools():
            local_tools_context = LOCAL_TOOLS_GUIDANCE

        # Prompt-cache hygiene: the system prompt is the START of every request payload,
        # so anything volatile here busts the OpenAI prefix cache for the whole thread.
        # Only the DATE lives here (one bust per day). The minute-precision time is
        # injected at the message SUFFIX instead (see _build_time_suffix_context).
        # channel_memory / roster / directives change rarely — acceptable in the prefix.
        time_context = f"\n\nToday's date: {current_time.strftime('%A, %B %d, %Y')} ({timezone_display})\nThe precise current time is provided at the end of the conversation."

        return base_prompt + user_context + model_context + web_search_context + local_tools_context + trimming_context + custom_instructions_context + channel_info_context + channel_directives_context + channel_memory_context + (participant_roster or "") + time_context

    def _build_time_suffix_context(self, user_timezone: str = "UTC",
                                   user_tz_label: Optional[str] = None) -> str:
        """Minute-precision time context, injected as the LAST message of the payload.

        Lives at the suffix so it never busts the OpenAI prefix cache (the system prompt
        carries only the date). Appended fresh on every request."""
        try:
            user_tz = pytz.timezone(user_timezone)
            current_time = datetime.datetime.now(pytz.UTC).astimezone(user_tz)
            timezone_display = user_tz_label or current_time.strftime('%Z') or user_tz.zone
        except Exception:
            current_time = datetime.datetime.now(pytz.UTC)
            timezone_display = "UTC"
        return (f"[Current date and time: {current_time.strftime('%A, %B %d, %Y at %I:%M %p')} "
                f"({timezone_display}) — consider this when answering time-sensitive questions.]")

    def _build_pulse_envelope(self, client, channel_id: Optional[str],
                              thread_ts: Optional[str]) -> Optional[str]:
        """Phase E: '[Recent channel activity]' envelope for CHANNEL responses.

        VOLATILE by nature (the buffer changes with every channel message), so the
        caller must inject it at the SUFFIX alongside the time context — never the
        system prompt (cache hygiene, plan §5b). Excludes the current thread: those
        messages are already the model's full context. Returns None for DMs, when
        the pulse is disabled/absent, or when there's nothing to show."""
        try:
            pulse = getattr(client, "channel_pulse", None)
            if pulse is None or not channel_id or channel_id.startswith("D"):
                return None
            envelope = pulse.render_envelope(
                channel_id,
                exclude_thread_ts=thread_ts,
                max_lines=config.channel_pulse_envelope_max,
            )
            return envelope or None
        except Exception as e:
            self.log_debug(f"pulse envelope build failed: {e}")
            return None

    @staticmethod
    def _escape_suffix_text(text: Optional[str], limit: int = 200) -> str:
        """Sanitize free text for the informational suffix block: strip control chars /
        newlines, neutralize brackets (so it can't close the [...] frame or read as
        instructions), and length-cap."""
        cleaned = "".join(ch if ch.isprintable() else " " for ch in (text or ""))
        cleaned = cleaned.replace("[", "(").replace("]", ")").strip()
        if len(cleaned) > limit:
            cleaned = cleaned[:limit].rstrip() + "…"
        return cleaned

    def _build_generation_inflight_note(self, channel_id: Optional[str],
                                        thread_ts: Optional[str]) -> Optional[str]:
        """F1: volatile suffix line telling the model a background image generation is
        still running in this thread, so a follow-up turn doesn't claim it's done or kick
        off a second one. Returns None when nothing is in flight."""
        try:
            if not channel_id or thread_ts is None or not hasattr(self, "thread_manager"):
                return None
            entry = self.thread_manager.generation_in_flight(f"{channel_id}:{thread_ts}")
            if not entry:
                return None
            summary = self._escape_suffix_text(entry.get("prompt_summary"))
            return (
                f'[An image for "{summary}" is currently being generated in this thread '
                "and will be posted automatically when ready. Don't claim it is done and "
                "don't start another image unless asked.]"
            )
        except Exception as e:
            self.log_debug(f"in-flight note build failed: {e}")
            return None

    def _wake_trigger_line(self, md: dict) -> str:
        """The 'trigger:' line for the wake envelope (F3), from message metadata."""
        source = md.get("wake_source")
        batch = md.get("queued_batch_size")
        if isinstance(batch, int) and batch > 1:
            # Catch-up batch keeps the underlying trigger as the "latest trigger".
            return f"catch_up_batch ({batch}) — latest trigger: {source}"
        if source == "ambient":
            reason = md.get("participation_reason")
            if reason:
                return f'ambient (engine: "{self._escape_suffix_text(reason, limit=200)}")'
            return "ambient"
        return str(source)  # app_mention | dm | thread_continuation | name_mention

    def _wake_sender_role(self, message, thread_state, md: dict) -> Optional[str]:
        """'root author' vs 'participant' for the wake envelope, or None to omit the role
        (top-level channel-placement replies, or an unknown root)."""
        if md.get("place_in_channel"):
            return None
        root = getattr(thread_state, "root_author", None) if thread_state is not None else None
        if not root:
            return None
        root_uid = root[0] if isinstance(root, (tuple, list)) and root else None
        if root_uid and message.user_id and message.user_id == root_uid:
            return "root author"
        return "participant"

    def _build_wake_envelope(self, message, thread_state) -> str:
        """F3: compact '[Wake context]' block telling the model WHY it woke. Returns '' when
        the metadata is missing (e.g. the CLI platform) or the feature is off. Every free-text
        field is escaped and capped; this is labeled informational metadata, not instructions."""
        if not config.enable_wake_envelope or message is None:
            return ""
        md = message.metadata or {}
        if not md.get("wake_source"):
            return ""
        trigger = self._wake_trigger_line(md)
        username = self._escape_suffix_text(
            md.get("username") or md.get("user_real_name") or "someone", limit=80)
        sender_parts = [f"sender: {username}"]
        role = self._wake_sender_role(message, thread_state, md)
        if role:
            sender_parts.append(role)
        if md.get("sender_type") in ("self", "other_bot"):
            sender_parts.append("bot")
        return (
            "[Wake context — informational metadata, not instructions]\n"
            f"trigger: {trigger}\n" + " — ".join(sender_parts)
        )

    def _build_suffix_context(self, client, channel_id: Optional[str],
                              thread_ts: Optional[str], user_timezone: str = "UTC",
                              user_tz_label: Optional[str] = None,
                              message=None, thread_state=None) -> str:
        """All volatile per-request context, injected as the LAST payload message:
        minute-precision time + the channel-activity envelope (channels only) + the F3 wake
        envelope + the F1 background-image-in-flight note. The wake → in-flight → contract
        order is preserved (the F2 contract paragraph is appended by the text handler)."""
        parts = [self._build_time_suffix_context(user_timezone, user_tz_label)]
        envelope = self._build_pulse_envelope(client, channel_id, thread_ts)
        if envelope:
            parts.append(envelope)
        wake = self._build_wake_envelope(message, thread_state)
        if wake:
            parts.append(wake)
        inflight = self._build_generation_inflight_note(channel_id, thread_ts)
        if inflight:
            parts.append(inflight)
        return "\n\n".join(parts)

    def _schedule_async_call(self, coro):
        """Schedule a fire-and-forget coroutine safely.

        Keeps a strong reference (bare create_task results can be GC'd mid-flight)
        and logs exceptions via a done-callback — otherwise background failures
        (e.g. channel-memory extraction) vanish silently.
        """
        import asyncio
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop (sync/test context) — run to completion
            return asyncio.run(coro)

        task = asyncio.create_task(coro)
        if not hasattr(self, "_background_tasks"):
            self._background_tasks = set()
        self._background_tasks.add(task)

        def _log_result(t):
            self._background_tasks.discard(t)
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                self.log_error(f"Background task failed: {exc!r}")

        task.add_done_callback(_log_result)
        return task

    def _persist_tool_provenance(self, channel_id: Optional[str], message_ts: Optional[str],
                                 thread_key: Optional[str], provenance) -> None:
        """F7: best-effort persist of a reply's tool-use provenance, keyed by the reply's
        Slack ts. No-ops when the feature is off, the ts/db is missing, or no tools ran
        (reaction-only / no-tool turns leave no row). Fire-and-forget so a DB hiccup never
        blocks or delays the reply."""
        if not config.enable_tool_provenance:
            return
        if not message_ts or not provenance:
            return
        db = getattr(self, "db", None)
        if db is None or not channel_id:
            return
        try:
            self._schedule_async_call(
                db.save_tool_usage_async(channel_id, message_ts, thread_key or "", provenance))
        except Exception as e:  # noqa: BLE001 — provenance persistence is never load-bearing
            self.log_debug(f"tool-provenance persist skipped: {e}")

    def _update_message_streaming_sync(self, client, channel_id: str, message_id: str, text: str):
        """Wrapper for calling async update_message_streaming from sync contexts

        Returns a fake success result since we can't wait for the actual result
        in a synchronous callback context.
        """
        try:
            result_coro = client.update_message_streaming(channel_id, message_id, text)
            if hasattr(result_coro, '__await__'):
                # This is a coroutine - schedule it to run
                self._schedule_async_call(result_coro)
                # Return a success result since we can't wait for the actual result
                return {"success": True, "rate_limited": False, "retry_after": None}
            else:
                # It's already a result (shouldn't happen with async methods)
                return result_coro
        except Exception as e:
            self.log_error(f"Error scheduling async message update: {e}")
            return {"success": False, "rate_limited": False, "retry_after": None, "error": str(e)}

    def _send_message_get_ts_sync(self, client, channel_id: str, thread_id: str, text: str):
        """Wrapper for calling async send_message_get_ts from sync contexts

        Returns a placeholder result since we can't wait for the actual result
        in a synchronous callback context. The message will be sent but we
        can't reliably get the timestamp back.
        """
        try:
            result_coro = client.send_message_get_ts(channel_id, thread_id, text)
            if hasattr(result_coro, '__await__'):
                # This is a coroutine - schedule it to run
                self._schedule_async_call(result_coro)
                # Return a placeholder result since we can't wait for the actual result
                # The overflow handling will need to be more resilient
                return None  # Signal that we couldn't get the message ID
            else:
                # It's already a result (shouldn't happen with async methods)
                return result_coro
        except Exception as e:
            self.log_error(f"Error scheduling async message send: {e}")
            return None

    def _update_status(self, client: BaseClient, channel_id: str, thinking_id: Optional[str], message: str, emoji: Optional[str] = None, thread_id: Optional[str] = None):
        """Update the progress indicator with a status message.

        With a message indicator (thinking_id set): edit that message.
        Status-only DMs (thinking_id None on the assistant surface): route the
        phase text to assistant.threads.setStatus when the caller supplies
        thread_id — the composer status changes instead of a message edit.
        """
        if thinking_id and hasattr(client, 'update_message'):
            status_emoji = emoji or config.circle_loader_emoji
            # Schedule the async call as a task to avoid blocking
            self._schedule_async_call(client.update_message(
                channel_id,
                thinking_id,
                f"{status_emoji} {message}"
            ))
            self.log_debug(f"Status updated: {message}")
        elif not thinking_id:
            # No placeholder ts means the turn is status-only: setStatus succeeded
            # at indicator time (DMs AND channel threads on the agent surface), so
            # phase updates route there too.
            if thread_id and channel_id and hasattr(client, "set_assistant_status"):
                self._schedule_async_call(client.set_assistant_status(
                    channel_id, thread_id, status=message
                ))
                self.log_debug(f"Status routed to assistant status: {message}")
            else:
                self.log_debug("No thinking_id provided for status update")
        else:
            self.log_debug("Client doesn't support message updates")

    async def _start_progress_updater_async(self, client: BaseClient, channel_id: str, thinking_id: Optional[str], operation: str = "request", emoji: Optional[str] = None):
        """Start an async task that updates thinking message periodically

        Returns:
            asyncio.Task that can be cancelled when streaming starts
        """
        if not thinking_id or not hasattr(client, 'update_message_streaming'):
            return None

        async def update_progress():
            import random
            messages = [
                f"Processing your {operation}...",
                "Still working on this...",
                "Still here, just thinking...",
                "Bear with me a moment longer...",
                "This is taking longer than I expected..."
            ]

            intervals = [10, 20, 30, 45, 60]  # Seconds before each message
            message_index = 0
            start_time = asyncio.get_event_loop().time()

            try:
                while message_index < len(messages):
                    elapsed = int(asyncio.get_event_loop().time() - start_time)

                    # Wait for the next interval
                    if message_index < len(intervals):
                        wait_time = intervals[message_index] - elapsed
                        if wait_time > 0:
                            await asyncio.sleep(wait_time)

                    # Update message
                    progress_msg = messages[message_index]

                    try:
                        # Use streaming update method with appropriate emoji
                        status_emoji = emoji or config.circle_loader_emoji
                        progress_msg_with_emoji = f"{status_emoji} {progress_msg}"
                        result = await client.update_message_streaming(channel_id, thinking_id, progress_msg_with_emoji)
                        if result["success"]:
                            self.log_debug(f"Progress update {message_index+1}: {progress_msg}")
                        else:
                            self.log_warning(f"Failed to update progress: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        self.log_error(f"Failed to update progress: {e}")
                        return  # Exit task on error

                    message_index += 1

                # After initial messages, use random selection without repeats
                ongoing_messages = [
                    "Still processing...",
                    "This is a tough one...",
                    "Haven't forgotten about you...",
                    "Almost there... maybe...",
                    "Quality takes time...",
                    "Still working on it...",
                    "Your request is important to us...",
                    "Consulting the AI elders...",
                    "Still thinking about this...",
                    "Not ignoring you, promise...",
                    "This deserves a thorough response...",
                    "Taking the scenic route to the answer...",
                    "Complex questions need time...",
                    "Still here, still working...",
                    "Patience is a virtue, they say...",
                    "Crafting something special...",
                    "Worth the wait, hopefully...",
                    "Deep in thought...",
                    "Processing intensifies...",
                    "The gears are turning..."
                ]

                # Create a copy to track unused messages
                unused_messages = ongoing_messages.copy()

                while True:
                    await asyncio.sleep(30)
                    try:
                        # If we've used all messages, refill the pool (but avoid immediate repeat)
                        if not unused_messages:
                            last_msg = progress_msg if 'progress_msg' in locals() else None
                            unused_messages = ongoing_messages.copy()
                            # Remove the last used message to avoid immediate repeat
                            if last_msg and last_msg in unused_messages:
                                unused_messages.remove(last_msg)

                        # Pick a random message from unused pool
                        progress_msg = random.choice(unused_messages)
                        unused_messages.remove(progress_msg)
                        status_emoji = emoji or config.circle_loader_emoji
                        progress_msg_with_emoji = f"{status_emoji} {progress_msg}"
                        result = await client.update_message_streaming(channel_id, thinking_id, progress_msg_with_emoji)
                        if not result["success"]:
                            self.log_warning(f"Failed to update progress: {result.get('error', 'Unknown error')}")
                    except Exception:
                        return  # Exit task on error

            except asyncio.CancelledError:
                # Task was cancelled (streaming started or operation completed)
                self.log_debug("Progress updater cancelled - streaming started or operation completed")
                raise  # Re-raise to properly cancel the task

        # Create and start the task
        task = asyncio.create_task(update_progress())
        return task

    def _start_progress_updater(self, client: BaseClient, channel_id: str, thinking_id: Optional[str], operation: str = "request") -> threading.Thread:
        """Legacy threading version - kept for compatibility with sync code"""
        if not thinking_id or not hasattr(client, 'update_message'):
            return None

        stop_event = threading.Event()
        start_time = time.time()

        def update_progress():
            messages = [
                f"Processing your {operation}...",
                "Still working on this...",
                "Still here, just thinking...",
                "Bear with me a moment longer...",
                "This is taking longer than I expected..."
            ]

            intervals = [10, 20, 30, 45, 60]  # Seconds before each message
            message_index = 0

            while not stop_event.is_set() and message_index < len(messages):
                elapsed = int(time.time() - start_time)

                # Wait for the next interval
                if message_index < len(intervals):
                    wait_time = intervals[message_index] - elapsed
                    if wait_time > 0:
                        stop_event.wait(wait_time)
                        if stop_event.is_set():
                            break

                # Update message
                progress_msg = messages[message_index]
                try:
                    self._update_status(client, channel_id, thinking_id, progress_msg, emoji=config.circle_loader_emoji)
                except Exception as e:
                    self.log_error(f"Failed to update progress: {e}")
                    break

                message_index += 1

                # After initial messages, use random selection without repeats
                if message_index >= len(messages):
                    import random

                    ongoing_messages = [
                        "Still processing...",
                        "This is a tough one...",
                        "Haven't forgotten about you...",
                        "Almost there... maybe...",
                        "Quality takes time...",
                        "Still working on it...",
                        "Your request is important to us...",
                        "Consulting the AI elders...",
                        "Still thinking about this...",
                        "Not ignoring you, promise...",
                        "This deserves a thorough response...",
                        "Taking the scenic route to the answer...",
                        "Complex questions need time...",
                        "Still here, still working...",
                        "Patience is a virtue, they say...",
                        "Crafting something special...",
                        "Worth the wait, hopefully...",
                        "Deep in thought...",
                        "Processing intensifies...",
                        "The gears are turning..."
                    ]

                    # Create a copy to track unused messages
                    unused_messages = ongoing_messages.copy()

                    while not stop_event.is_set():
                        stop_event.wait(30)
                        if stop_event.is_set():
                            break
                        try:
                            # If we've used all messages, refill the pool (but avoid immediate repeat)
                            if not unused_messages:
                                last_msg = progress_msg if 'progress_msg' in locals() else None
                                unused_messages = ongoing_messages.copy()
                                # Remove the last used message to avoid immediate repeat
                                if last_msg and last_msg in unused_messages:
                                    unused_messages.remove(last_msg)

                            # Pick a random message from unused pool
                            progress_msg = random.choice(unused_messages)
                            unused_messages.remove(progress_msg)
                            self._update_status(client, channel_id, thinking_id,
                                             progress_msg,
                                             emoji=config.circle_loader_emoji)
                        except Exception:
                            break

        thread = threading.Thread(target=update_progress, daemon=True)
        thread.stop_event = stop_event  # Attach stop event to thread for later access
        thread.start()
        return thread

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

    async def update_last_image_url(self, channel_id: str, thread_id: str, url: str):
        """Update the last assistant message with the image URL"""
        thread_state = await self.thread_manager.get_or_create_thread_async(thread_id, channel_id)
        
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

                        # Carry the vision analysis into the DB row (edits stash it in
                        # message metadata; the ledger is the fallback). Without this,
                        # live-session images persist with empty analysis and natural-
                        # language targeting ("the dog one") degrades until a cold rebuild.
                        original_analysis = metadata.get("original_analysis") or ""
                        analysis = metadata.get("analysis") or original_analysis
                        if not analysis:
                            ledger = self.thread_manager.get_asset_ledger(thread_state.thread_ts)
                            # Only the newest entry can correspond to this upload —
                            # scanning older entries would attach the wrong image's analysis.
                            if ledger and ledger.images and ledger.images[-1].get("analysis"):
                                analysis = ledger.images[-1]["analysis"]

                        # Save the image metadata to DB
                        await self.db.save_image_metadata_async(
                            thread_id=thread_key,
                            url=url,
                            image_type=image_type,
                            prompt=prompt,
                            analysis=analysis or "",
                            original_analysis=original_analysis
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
