from __future__ import annotations

import datetime
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import base64
import os
import re
import pytz

from base_client import BaseClient, Message
from config import config
from prompts import SLACK_SYSTEM_PROMPT, DISCORD_SYSTEM_PROMPT, CLI_SYSTEM_PROMPT


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
        except Exception:
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

    def _start_progress_updater(self, client: BaseClient, channel_id: str, thinking_id: Optional[str], operation: str = "request") -> threading.Thread:
        """Start a background thread that updates thinking message periodically"""
        if not thinking_id or not hasattr(client, 'update_message'):
            return None

        stop_event = threading.Event()
        start_time = time.time()

        def update_progress():
            messages = [
                f"Processing your {operation}...",
                f"Still working on your {operation}...",
                "This is taking longer than expected...",
                "Thank you for your patience...",
                f"Still processing your {operation}..."
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
                elapsed = int(time.time() - start_time)
                progress_msg = f"{messages[message_index]} ({elapsed}s)"
                try:
                    self._update_status(client, channel_id, thinking_id, progress_msg, emoji=config.thinking_emoji)
                except Exception as e:
                    self.log_error(f"Failed to update progress: {e}")
                    break

                message_index += 1

                # After all messages, just update the time every 30s
                if message_index >= len(messages):
                    while not stop_event.is_set():
                        stop_event.wait(30)
                        if stop_event.is_set():
                            break
                        elapsed = int(time.time() - start_time)
                        try:
                            self._update_status(client, channel_id, thinking_id,
                                             f"Still processing... ({elapsed}s)",
                                             emoji=config.thinking_emoji)
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
