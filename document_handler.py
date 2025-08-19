"""
Document Parsing and Content Extraction Handler

This module provides comprehensive document processing capabilities for the chatbot system,
supporting PDFs, Word documents, Excel spreadsheets, and other common file types.
Designed to maintain full document context without truncation and store complete content.
"""

import re
import base64
from io import BytesIO
from typing import Dict, List, Optional, Any, Union, Tuple
from urllib.parse import unquote
import logging

import pdfplumber
import PyPDF2
from pdf2image import convert_from_bytes
from docx import Document
import openpyxl
import pandas as pd

from logger import LoggerMixin

# Supported document MIME types
SUPPORTED_DOCUMENT_MIMETYPES = {
    # PDF documents
    "application/pdf",
    
    # Word documents
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "application/msword",  # .doc
    
    # Excel/Spreadsheet documents
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
    "application/vnd.ms-excel",  # .xls
    "text/csv",  # .csv
    
    # PowerPoint documents
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
    "application/vnd.ms-powerpoint",  # .ppt
    
    # Text documents
    "text/plain",  # .txt
    "text/markdown",  # .md
    "application/rtf",  # .rtf
    
    # Code files
    "text/x-python",  # .py
    "application/javascript",  # .js
    "application/json",  # .json
    "text/x-sql",  # .sql
    "application/x-yaml",  # .yaml/.yml
    "application/xml",  # .xml
    "text/html",  # .html
    
    # Log files
    "text/x-log",  # .log
}

# Document file extensions
DOCUMENT_EXTENSIONS = {
    '.pdf', '.docx', '.doc', '.xlsx', '.xls', '.csv', '.pptx', '.ppt',
    '.txt', '.md', '.rtf', '.py', '.js', '.json', '.sql', '.yaml', '.yml',
    '.xml', '.html', '.htm', '.log', '.out'
}

# MIME type routing handlers
MIME_TYPE_HANDLERS = {
    'application/pdf': 'parse_pdf_structured',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'parse_docx_structured',
    'application/msword': 'parse_docx_structured',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'parse_excel_adaptive',
    'application/vnd.ms-excel': 'parse_excel_adaptive',
    'text/csv': 'parse_excel_adaptive',
    'text/plain': 'parse_text',
    'text/markdown': 'parse_text',
    'application/rtf': 'parse_text',
    'text/x-python': 'parse_text',
    'application/javascript': 'parse_text',
    'application/json': 'parse_text',
    'text/x-sql': 'parse_text',
    'application/x-yaml': 'parse_text',
    'application/xml': 'parse_text',
    'text/html': 'parse_text',
    'text/x-log': 'parse_text',
}


class DocumentHandler(LoggerMixin):
    """Handles document parsing and content extraction with error recovery"""
    
    def __init__(self, max_document_size: int = 50 * 1024 * 1024):
        """
        Initialize the document handler
        
        Args:
            max_document_size: Maximum document size in bytes (default 50MB)
        """
        self.max_document_size = max_document_size
    
    def is_document_file(self, filename: str, mimetype: Optional[str] = None) -> bool:
        """
        Check if a file is a supported document type
        
        Args:
            filename: The filename to check
            mimetype: Optional MIME type
            
        Returns:
            True if the file is a supported document type
        """
        if mimetype and mimetype in SUPPORTED_DOCUMENT_MIMETYPES:
            return True
            
        # Check file extension
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in DOCUMENT_EXTENSIONS)
    
    def safe_extract_content(self, file_data: bytes, mime_type: str, filename: str) -> Dict[str, Any]:
        """
        Safely extract document content with comprehensive error recovery
        
        Args:
            file_data: The document file data as bytes
            mime_type: The MIME type of the document
            filename: The original filename
            
        Returns:
            Dict with extracted content, structure info, and any errors
        """
        
        # Validate file size
        if len(file_data) > self.max_document_size:
            return {
                'content': f'[Document {filename} too large: {len(file_data) / 1024 / 1024:.1f}MB (max: {self.max_document_size / 1024 / 1024:.1f}MB)]',
                'error': 'Document exceeds size limit',
                'format': 'error'
            }
        
        try:
            # Route to appropriate parser
            handler_name = MIME_TYPE_HANDLERS.get(mime_type)
            
            if not handler_name:
                # Try to determine handler from filename
                filename_lower = filename.lower()
                if filename_lower.endswith('.pdf'):
                    handler_name = 'parse_pdf_structured'
                elif filename_lower.endswith(('.docx', '.doc')):
                    handler_name = 'parse_docx_structured'
                elif filename_lower.endswith(('.xlsx', '.xls', '.csv')):
                    handler_name = 'parse_excel_adaptive'
                else:
                    handler_name = 'parse_text'
            
            # Get the parser method
            parser_method = getattr(self, handler_name)
            result = parser_method(file_data, filename)
            
            # Sanitize the content
            if 'content' in result:
                result['content'] = self.sanitize_content(result['content'])
            
            # Add filename to result
            result['filename'] = filename
            result['mime_type'] = mime_type
            result['size_bytes'] = len(file_data)
            
            return result
            
        except Exception as e:
            self.log_error(f"Failed to parse {filename}: {e}", exc_info=True)
            
            # Fallback: try basic text extraction
            try:
                raw_text = self.force_text_extraction(file_data, mime_type, filename)
                return {
                    'content': self.sanitize_content(raw_text),
                    'filename': filename,
                    'mime_type': mime_type,
                    'size_bytes': len(file_data),
                    'error': f'Partial extraction - document may be malformed: {str(e)}',
                    'format': 'text'
                }
            except Exception as fallback_error:
                self.log_error(f"Fallback extraction also failed for {filename}: {fallback_error}")
                
                # Last resort: mark as unparseable
                return {
                    'content': f'[Unable to parse {filename} - document appears malformed or corrupted]',
                    'filename': filename,
                    'mime_type': mime_type,
                    'size_bytes': len(file_data),
                    'error': f'Document could not be parsed: {str(e)}',
                    'format': 'error'
                }
    
    def sanitize_content(self, text: str) -> str:
        """
        Sanitize content to prevent format breaking and injection attacks
        
        Args:
            text: The text content to sanitize
            
        Returns:
            Sanitized text content
        """
        if not text:
            return ""
        
        sanitized = str(text)
        
        # Remove null bytes and control characters (except newlines, tabs, carriage returns)
        sanitized = sanitized.replace('\x00', '')
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\n\r\t')
        
        # Balance code blocks
        code_blocks = sanitized.count('```')
        if code_blocks % 2 != 0:
            sanitized += '\n```'  # Close unclosed code block
            
        # Escape document markers to prevent injection
        sanitized = sanitized.replace('[Document:', '[Document\\:')
        sanitized = sanitized.replace('[End Document]', '[End\\ Document]')
        sanitized = sanitized.replace('[Page ', '[Page\\ ')
        sanitized = sanitized.replace('[Sheet:', '[Sheet\\:')
        
        # Fix markdown tables
        sanitized = self.fix_markdown_tables(sanitized)
        
        # Limit consecutive newlines
        sanitized = re.sub(r'\n{4,}', '\n\n\n', sanitized)
        
        # No size limit - let the model's token limit handle it
        # Previously limited to 1MB
        
        return sanitized
    
    def parse_pdf_structured(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """
        Extract PDF content preserving tables and page structure
        
        Args:
            file_data: PDF file data as bytes
            filename: Original filename for error reporting
            
        Returns:
            Dict with pages, content, and structure info
        """
        # Try pdfplumber first, then PyPDF2 as fallback
        try:
            result = self._parse_pdf_with_pdfplumber(file_data, filename)
        except Exception as e:
            self.log_warning(f"pdfplumber failed, trying PyPDF2: {e}")
            result = self._parse_pdf_with_pypdf2(file_data, filename)
        
        # Check if PDF is likely image-based (scanned document)
        if result and self._is_image_based_pdf(result):
            result['is_image_based'] = True
            result['requires_ocr'] = True
            
            # Convert PDF pages to images for vision processing
            self.log_info(f"Converting image-based PDF {filename} to images for OCR")
            page_images = self.convert_pdf_to_images(file_data, max_pages=10)
            
            if page_images:
                result['page_images'] = page_images
                result['ocr_available'] = True
                self.log_info(f"Successfully converted {len(page_images)} PDF pages to images")
                
                # Update content to indicate OCR is available
                total_pages = result.get('total_pages', 0)
                if total_pages > len(page_images):
                    result['content'] = (
                        f"[Note: This PDF appears to be a scanned document with {total_pages} total pages. "
                        f"Due to API limits, converted first {len(page_images)} pages to images for vision/OCR analysis. "
                        f"Remaining {total_pages - len(page_images)} pages were not processed.]\n\n"
                        f"{result.get('content', '[No text extracted from PDF]')}"
                    )
                else:
                    result['content'] = (
                        f"[Note: This PDF appears to be a scanned document. "
                        f"Converted all {len(page_images)} page(s) to images for vision analysis. "
                        f"The bot will use vision/OCR to extract content.]\n\n"
                        f"{result.get('content', '[No text extracted from PDF]')}"
                    )
            else:
                # Conversion failed, use original note
                result['content'] = (
                    f"[Note: This PDF appears to be a scanned document or contains primarily images. "
                    f"Text extraction found minimal or no text content. "
                    f"PDF to image conversion failed - OCR not available.]\n\n"
                    f"{result.get('content', '[No text extracted from PDF]')}"
                )
        
        return result
    
    def _parse_pdf_with_pdfplumber(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Parse PDF using pdfplumber for advanced structure extraction"""
        
        pages = []
        total_pages = 0
        has_tables = False
        
        try:
            with pdfplumber.open(BytesIO(file_data)) as pdf:
                total_pages = len(pdf.pages)
                
                # Sanity limit for very large PDFs
                pages_to_process = min(total_pages, 1000)
                
                for i, page in enumerate(pdf.pages[:pages_to_process]):
                    page_data = {'page': i + 1}
                    
                    try:
                        # Extract tables if present
                        tables = page.extract_tables()
                        if tables:
                            page_data['tables'] = []
                            for table in tables:
                                if table and len(table) > 0:
                                    md_table = self.flexible_table_to_markdown(table)
                                    if md_table:
                                        page_data['tables'].append(md_table)
                                        has_tables = True
                        
                        # Extract text content
                        text = page.extract_text() or ""
                        page_data['content'] = text
                        
                        # Try to detect structure hints
                        page_data['structure_hints'] = self._detect_text_structure(text)
                        
                    except Exception as e:
                        self.log_warning(f"Error processing page {i+1} of {filename}: {e}")
                        page_data['content'] = f'[Page {i+1} extraction failed]'
                    
                    pages.append(page_data)
                
                if pages_to_process < total_pages:
                    pages.append({
                        'page': pages_to_process + 1,
                        'content': f'[Remaining {total_pages - pages_to_process} pages omitted - document too large]'
                    })
        
        except Exception as e:
            raise Exception(f"pdfplumber parsing failed: {e}")
        
        # Combine all content
        all_content = []
        for page in pages:
            all_content.append(f"[Page {page['page']}]")
            if page.get('tables'):
                all_content.extend(page['tables'])
            if page.get('content'):
                all_content.append(page['content'])
        
        return {
            'content': '\n'.join(all_content),
            'pages': pages,
            'total_pages': total_pages,
            'has_tables': has_tables,
            'format': 'pdf'
        }
    
    def _parse_pdf_with_pypdf2(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Parse PDF using PyPDF2 as fallback"""
        
        try:
            reader = PyPDF2.PdfReader(BytesIO(file_data))
            total_pages = len(reader.pages)
            
            pages = []
            content_parts = []
            
            # Limit pages for very large documents
            pages_to_process = min(total_pages, 100)
            
            for i, page in enumerate(reader.pages[:pages_to_process]):
                try:
                    text = page.extract_text()
                    page_data = {
                        'page': i + 1,
                        'content': text or f'[Page {i+1} - no text extracted]'
                    }
                    pages.append(page_data)
                    content_parts.append(f"[Page {i+1}]")
                    content_parts.append(page_data['content'])
                except Exception as e:
                    self.log_warning(f"Error extracting page {i+1}: {e}")
                    pages.append({
                        'page': i + 1,
                        'content': f'[Page {i+1} extraction failed]'
                    })
            
            if pages_to_process < total_pages:
                content_parts.append(f'[Remaining {total_pages - pages_to_process} pages omitted]')
            
            return {
                'content': '\n'.join(content_parts),
                'pages': pages,
                'total_pages': total_pages,
                'has_tables': False,
                'format': 'pdf',
                'extraction_method': 'PyPDF2_fallback'
            }
            
        except Exception as e:
            raise Exception(f"PyPDF2 parsing failed: {e}")
    
    def parse_docx_structured(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """
        Extract Word document content preserving formatting and structure
        
        Args:
            file_data: DOCX file data as bytes
            filename: Original filename for error reporting
            
        Returns:
            Dict with content, pages/sections, and structure info
        """
        
        try:
            doc = Document(BytesIO(file_data))
            content_blocks = []
            tables_found = False
            
            # Extract paragraphs and tables in document order
            for element in doc.element.body:
                if element.tag.endswith('p'):  # Paragraph
                    # Find the corresponding paragraph object
                    for para in doc.paragraphs:
                        if para._element == element:
                            text = para.text.strip()
                            if text:  # Only include non-empty paragraphs
                                # Detect if this looks like a header
                                style_name = para.style.name if para.style else ''
                                if 'Heading' in style_name or text.isupper() and len(text) < 100:
                                    content_blocks.append(f"## {text}")
                                else:
                                    content_blocks.append(text)
                            break
                
                elif element.tag.endswith('tbl'):  # Table
                    # Find the corresponding table object
                    for table in doc.tables:
                        if table._element == element:
                            table_md = self._extract_docx_table(table)
                            if table_md:
                                content_blocks.append(table_md)
                                tables_found = True
                            break
            
            # Combine content
            full_content = '\n\n'.join(content_blocks)
            
            # Split into sections if headers are detected
            sections = self._split_into_sections(full_content)
            
            return {
                'content': full_content,
                'sections': sections,
                'has_tables': tables_found,
                'total_sections': len(sections),
                'format': 'docx'
            }
            
        except Exception as e:
            # Try alternative extraction method for problematic DOCX files
            self.log_warning(f"Standard DOCX parsing failed, trying alternative method: {e}")
            return self.parse_docx_alternative(file_data, filename)
    
    def parse_docx_alternative(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """
        Alternative DOCX parsing using ZIP extraction for problematic files
        """
        import zipfile
        import xml.etree.ElementTree as ET
        
        try:
            # First check if this is actually a ZIP file
            if not file_data.startswith(b'PK'):  # ZIP files start with 'PK'
                self.log_error(f"File {filename} does not appear to be a ZIP/DOCX file. First bytes: {file_data[:4]}")
                # Could be an old .doc format
                return self.parse_doc_legacy(file_data, filename)
            
            # DOCX files are ZIP archives
            with zipfile.ZipFile(BytesIO(file_data)) as zip_file:
                # Debug: log what's in the ZIP file
                file_list = zip_file.namelist()
                self.log_info(f"DOCX ZIP contents for {filename}: {file_list[:10]}...")  # First 10 files
                
                # Check if it's a valid DOCX structure - try different possible paths
                # Handle both forward and backslash separators
                doc_path = None
                for possible_path in ['word/document.xml', 'word\\document.xml', 'Word/document.xml', 'document.xml']:
                    if possible_path in file_list:
                        doc_path = possible_path
                        break
                
                if not doc_path:
                    # Log more details about what we found
                    self.log_error(f"No document.xml found. ZIP contains: {file_list}")
                    raise ValueError(f"Not a valid DOCX file structure. Files in archive: {len(file_list)}")
                
                # Extract the main document XML
                with zip_file.open(doc_path) as xml_file:
                    tree = ET.parse(xml_file)
                    root = tree.getroot()
                    
                    # Define namespaces
                    namespaces = {
                        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
                    }
                    
                    # Extract all text from paragraphs
                    content_parts = []
                    for paragraph in root.findall('.//w:p', namespaces):
                        texts = []
                        for text_elem in paragraph.findall('.//w:t', namespaces):
                            if text_elem.text:
                                texts.append(text_elem.text)
                        
                        if texts:
                            para_text = ''.join(texts).strip()
                            if para_text:
                                content_parts.append(para_text)
                    
                    # Try to extract tables
                    for table in root.findall('.//w:tbl', namespaces):
                        table_rows = []
                        for row in table.findall('.//w:tr', namespaces):
                            cells = []
                            for cell in row.findall('.//w:tc', namespaces):
                                cell_texts = []
                                for text_elem in cell.findall('.//w:t', namespaces):
                                    if text_elem.text:
                                        cell_texts.append(text_elem.text)
                                cells.append(' '.join(cell_texts))
                            if cells:
                                table_rows.append(' | '.join(cells))
                        
                        if table_rows:
                            content_parts.append('\n[Table]\n' + '\n'.join(table_rows))
                    
                    content = '\n\n'.join(content_parts)
                    
                    return {
                        'content': content or '[No text content extracted]',
                        'format': 'docx',
                        'extraction_method': 'xml_parsing',
                        'warning': 'Document was parsed using alternative method - formatting may be simplified'
                    }
                    
        except Exception as e:
            self.log_error(f"Alternative DOCX parsing also failed: {e}")
            # Final fallback - try to extract any text using textract or similar
            return self.parse_docx_textract_fallback(file_data, filename)
    
    def parse_doc_legacy(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """
        Parse legacy .doc files (pre-2007 Word format)
        """
        try:
            # Check if it's actually a .doc file (OLE2 format)
            if file_data.startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):
                self.log_info(f"Detected legacy .doc format for {filename}")
                
                # Try using python-docx2txt or other libraries
                try:
                    import docx2txt
                    # Save to temp file since docx2txt needs a file path
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.doc', delete=False) as tmp:
                        tmp.write(file_data)
                        tmp_path = tmp.name
                    
                    try:
                        text = docx2txt.process(tmp_path)
                        return {
                            'content': text or '[No text content found]',
                            'format': 'doc',
                            'extraction_method': 'docx2txt',
                            'warning': 'Legacy .doc format - formatting simplified'
                        }
                    finally:
                        import os
                        os.unlink(tmp_path)
                        
                except ImportError:
                    self.log_warning("docx2txt not available for .doc parsing")
                
                # Fallback to basic text extraction
                return {
                    'content': '[Legacy .doc format detected - cannot extract content without additional tools]',
                    'format': 'doc',
                    'extraction_method': 'none',
                    'error': 'Legacy Word format requires additional tools like docx2txt or antiword'
                }
            else:
                # Not a recognized format
                return {
                    'content': f'[Unrecognized file format for {filename}]',
                    'format': 'unknown',
                    'error': f'File does not appear to be a valid Word document. First bytes: {file_data[:4].hex()}'
                }
                
        except Exception as e:
            self.log_error(f"Legacy doc parsing failed: {e}")
            return {
                'content': f'[Unable to parse {filename}]',
                'format': 'doc',
                'error': str(e)
            }
    
    def parse_docx_textract_fallback(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """
        Final fallback using system tools or basic extraction
        """
        try:
            # Save temporarily and use pandoc or other system tools if available
            import tempfile
            import subprocess
            
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
                tmp.write(file_data)
                tmp_path = tmp.name
            
            try:
                # Try pandoc if available
                result = subprocess.run(
                    ['pandoc', '-f', 'docx', '-t', 'plain', tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0 and result.stdout:
                    return {
                        'content': result.stdout,
                        'format': 'docx',
                        'extraction_method': 'pandoc',
                        'warning': 'Document extracted using pandoc - formatting simplified'
                    }
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
            finally:
                import os
                os.unlink(tmp_path)
                
        except Exception as e:
            self.log_error(f"Textract fallback failed: {e}")
        
        # Absolute final fallback
        return {
            'content': f'[Unable to extract content from {filename} - document format is not supported or file is corrupted]',
            'format': 'docx',
            'extraction_method': 'failed',
            'error': 'All extraction methods failed'
        }
    
    def parse_excel_adaptive(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """
        Extract Excel/CSV content with adaptive structure preservation
        
        Args:
            file_data: Excel/CSV file data as bytes
            filename: Original filename for error reporting
            
        Returns:
            Dict with sheets, content, and structure info
        """
        
        try:
            # Determine if this is CSV or Excel
            if filename.lower().endswith('.csv'):
                return self._parse_csv_with_pandas(file_data, filename, pd)
            else:
                return self._parse_excel_with_pandas(file_data, filename, pd)
                
        except Exception as e:
            # Try CSV interpretation as fallback
            try:
                self.log_warning(f"Excel parsing failed for {filename}, trying CSV fallback: {e}")
                return self._parse_csv_with_pandas(file_data, filename, pd)
            except Exception as csv_error:
                raise Exception(f"Spreadsheet parsing failed: {e}, CSV fallback: {csv_error}")
    
    def _parse_excel_with_pandas(self, file_data: bytes, filename: str, pd) -> Dict[str, Any]:
        """Parse Excel file using pandas"""
        # Read all sheets
        try:
            all_sheets = pd.read_excel(BytesIO(file_data), sheet_name=None, engine='openpyxl')
        except:
            # Try without specifying engine
            all_sheets = pd.read_excel(BytesIO(file_data), sheet_name=None)
        
        sheets = []
        content_parts = []
        
        # Limit number of sheets
        sheet_items = list(all_sheets.items())[:20]
        
        for sheet_name, df in sheet_items:
            sheet_data = {
                'name': str(sheet_name)[:50],  # Limit sheet name length
                'rows': len(df),
                'cols': len(df.columns)
            }
            
            # Handle different data structures
            if df.empty:
                sheet_content = "[Empty sheet]"
                sheet_data['format'] = 'empty'
            elif self._is_simple_table(df):
                # Convert to markdown table
                sheet_content = self._dataframe_to_markdown(df)
                sheet_data['format'] = 'table'
            elif self._is_list_data(df):
                # Simple list format
                sheet_content = self._format_as_list(df)
                sheet_data['format'] = 'list'
            else:
                # Fallback: structured text
                sheet_content = df.to_string(max_rows=1000, max_cols=50)
                sheet_data['format'] = 'raw'
            
            sheet_data['content'] = sheet_content
            
            # Add to overall content
            content_parts.append(f"[Sheet: {sheet_name}]")
            content_parts.append(sheet_content)
            
            sheets.append(sheet_data)
        
        return {
            'content': '\n\n'.join(content_parts),
            'sheets': sheets,
            'total_sheets': len(all_sheets),
            'format': 'excel'
        }
    
    def _parse_csv_with_pandas(self, file_data: bytes, filename: str, pd) -> Dict[str, Any]:
        """Parse CSV file using pandas"""
        try:
            # Try different encodings and separators
            text_data = file_data.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text_data = file_data.decode('latin-1')
            except UnicodeDecodeError:
                text_data = file_data.decode('utf-8', errors='ignore')
        
        # Try to read CSV with different separators
        for sep in [',', ';', '\t', '|']:
            try:
                df = pd.read_csv(BytesIO(text_data.encode('utf-8')), 
                               sep=sep, on_bad_lines='skip', nrows=10000)
                if len(df.columns) > 1:  # Found good separator
                    break
            except:
                continue
        else:
            # Fallback: assume comma separation
            df = pd.read_csv(BytesIO(text_data.encode('utf-8')), 
                           on_bad_lines='skip', nrows=10000)
        
        # Format the data
        if df.empty:
            content = "[Empty CSV file]"
        else:
            content = self._dataframe_to_markdown(df)
        
        return {
            'content': content,
            'sheets': [{'name': 'CSV Data', 'content': content, 'format': 'table'}],
            'total_sheets': 1,
            'rows': len(df),
            'cols': len(df.columns),
            'format': 'csv'
        }
    
    def parse_text(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """
        Parse plain text files with encoding detection
        
        Args:
            file_data: Text file data as bytes
            filename: Original filename for error reporting
            
        Returns:
            Dict with text content
        """
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                text = file_data.decode(encoding)
                
                # Detect file type from content or extension
                file_format = self._detect_text_format(text, filename)
                
                return {
                    'content': text,
                    'format': file_format,
                    'encoding': encoding,
                    'lines': len(text.splitlines())
                }
                
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, use utf-8 with error handling
        text = file_data.decode('utf-8', errors='ignore')
        return {
            'content': text,
            'format': 'text',
            'encoding': 'utf-8_with_errors',
            'lines': len(text.splitlines()),
            'warning': 'Some characters may not have been decoded correctly'
        }
    
    def flexible_table_to_markdown(self, table_data: List[List[str]]) -> str:
        """
        Convert variable table structures to markdown with error handling
        
        Args:
            table_data: 2D list representing table rows and columns
            
        Returns:
            Markdown table string or empty string if conversion fails
        """
        if not table_data or not table_data[0]:
            return ""
        
        try:
            # Handle irregular tables gracefully
            max_cols = max(len(row) if row else 0 for row in table_data)
            
            if max_cols == 0:
                return ""
            
            # Normalize rows to same width
            normalized = []
            for row in table_data:
                if row:
                    # Pad short rows, handle None values
                    norm_row = [(str(cell) if cell is not None else "") for cell in row]
                    norm_row.extend([""] * (max_cols - len(norm_row)))
                    normalized.append(norm_row)
            
            if not normalized:
                return ""
            
            # Build markdown table
            lines = []
            
            # Header row
            header = "| " + " | ".join(cell.replace("|", "\\|") for cell in normalized[0]) + " |"
            lines.append(header)
            
            # Separator
            separator = "|" + "|".join([" --- " for _ in range(max_cols)]) + "|"
            lines.append(separator)
            
            # Data rows (limit to reasonable number)
            for row in normalized[1:100]:  # Limit to 100 rows
                line = "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"
                lines.append(line)
            
            if len(normalized) > 101:
                lines.append(f"| ... | ({len(normalized) - 101} more rows) | ... |")
            
            return "\n".join(lines)
            
        except Exception as e:
            self.log_warning(f"Table conversion failed: {e}")
            return "[Table data could not be converted]"
    
    def fix_markdown_tables(self, text: str) -> str:
        """
        Fix common markdown table formatting issues
        
        Args:
            text: Text containing potentially malformed markdown tables
            
        Returns:
            Text with fixed table formatting
        """
        try:
            lines = text.split('\n')
            fixed_lines = []
            in_table = False
            table_col_count = 0
            
            for line in lines:
                if '|' in line and line.strip():
                    # This looks like a table row
                    stripped = line.strip()
                    
                    # Ensure proper table row format
                    if not stripped.startswith('|'):
                        stripped = '| ' + stripped
                    if not stripped.endswith('|'):
                        stripped = stripped + ' |'
                    
                    # Count columns
                    col_count = stripped.count('|') - 1
                    
                    if not in_table:
                        # Starting a new table
                        table_col_count = col_count
                        in_table = True
                    elif col_count != table_col_count:
                        # Column count mismatch - normalize
                        stripped = self._normalize_table_row(stripped, table_col_count)
                    
                    fixed_lines.append(stripped)
                    
                elif in_table and line.strip() and '|' not in line:
                    # Table ended
                    in_table = False
                    fixed_lines.append(line)
                else:
                    # Regular line
                    if in_table and not line.strip():
                        in_table = False  # Empty line ends table
                    fixed_lines.append(line)
            
            return '\n'.join(fixed_lines)
            
        except Exception as e:
            self.log_warning(f"Table fixing failed: {e}")
            return text
    
    def force_text_extraction(self, file_data: bytes, mime_type: str, filename: str) -> str:
        """
        Last resort text extraction for when structured parsing fails
        
        Args:
            file_data: File data as bytes
            mime_type: MIME type of the file
            filename: Original filename
            
        Returns:
            Extracted text content
        """
        # Don't try to decode binary formats as text
        binary_extensions = ('.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt', '.pdf', '.zip', '.rar')
        if filename.lower().endswith(binary_extensions):
            # These are binary formats that shouldn't be decoded as text
            return f"[Unable to extract text from corrupted {filename} - binary format]"
        
        # Try direct text decoding for text-based formats
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                text = file_data.decode(encoding, errors='ignore')
                # Clean up the text
                text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
                # Check if we got mostly readable text (not binary garbage)
                if text.strip() and len([c for c in text[:100] if c.isprintable()]) > 80:
                    return f"[Raw text extraction from {filename}]\n{text}"
            except:
                continue
        
        return f"[Unable to extract readable text from {filename}]"
    
    # Helper methods
    
    def _detect_text_structure(self, text: str) -> Dict[str, Any]:
        """Detect structural elements in text"""
        lines = text.split('\n')
        structure = {
            'has_headers': False,
            'has_lists': False,
            'has_tables': False,
            'line_count': len(lines)
        }
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
                
            # Check for headers (short lines, all caps, etc.)
            if len(stripped) < 100 and (stripped.isupper() or 
                                       stripped.startswith('#') or
                                       len(stripped.split()) <= 5):
                structure['has_headers'] = True
            
            # Check for lists
            if re.match(r'^\s*[-*•]\s+', line) or re.match(r'^\s*\d+\.\s+', line):
                structure['has_lists'] = True
            
            # Check for table-like structures
            if '|' in line and line.count('|') >= 2:
                structure['has_tables'] = True
        
        return structure
    
    def _extract_docx_table(self, table) -> str:
        """Extract table from docx document as markdown"""
        try:
            rows = []
            for row in table.rows:
                cells = []
                for cell in row.cells:
                    # Clean cell text
                    cell_text = cell.text.strip().replace('\n', ' ')
                    cells.append(cell_text)
                rows.append(cells)
            
            return self.flexible_table_to_markdown(rows)
        except Exception as e:
            self.log_warning(f"Failed to extract docx table: {e}")
            return "[Table extraction failed]"
    
    def _split_into_sections(self, content: str) -> List[Dict[str, str]]:
        """Split content into sections based on headers"""
        lines = content.split('\n')
        sections = []
        current_section = {'title': 'Document Start', 'content': []}
        
        for line in lines:
            if line.startswith('##'):
                # New section
                if current_section['content']:
                    current_section['content'] = '\n'.join(current_section['content']).strip()
                    sections.append(current_section)
                
                current_section = {
                    'title': line.replace('##', '').strip(),
                    'content': []
                }
            else:
                current_section['content'].append(line)
        
        # Add final section
        if current_section['content']:
            current_section['content'] = '\n'.join(current_section['content']).strip()
            sections.append(current_section)
        
        return sections
    
    def _is_simple_table(self, df) -> bool:
        """Check if DataFrame is a simple table suitable for markdown"""
        if df.empty or len(df.columns) < 2:
            return False
        
        # Check if column names look reasonable
        unnamed_cols = sum(1 for col in df.columns if 'Unnamed' in str(col))
        if unnamed_cols > len(df.columns) / 2:
            return False
        
        return True
    
    def _is_list_data(self, df) -> bool:
        """Check if DataFrame is better represented as a list"""
        return len(df.columns) <= 2 and len(df) > 10
    
    def _dataframe_to_markdown(self, df) -> str:
        """Convert DataFrame to markdown with size limits"""
        try:
            # Limit size
            if len(df) > 1000:
                df_limited = df.head(1000)
                truncated = True
            else:
                df_limited = df
                truncated = False
            
            if len(df_limited.columns) > 20:
                df_limited = df_limited.iloc[:, :20]
                cols_truncated = True
            else:
                cols_truncated = False
            
            # Clean column names
            df_limited.columns = [str(col)[:50] for col in df_limited.columns]
            
            # Convert to markdown
            md_table = df_limited.to_markdown(index=False, tablefmt='pipe')
            
            # Add truncation notices
            notices = []
            if truncated:
                notices.append(f"[Showing first 1000 of {len(df)} rows]")
            if cols_truncated:
                notices.append(f"[Showing first 20 of {len(df.columns)} columns]")
            
            if notices:
                md_table = '\n'.join(notices) + '\n\n' + md_table
            
            return md_table
            
        except Exception as e:
            self.log_warning(f"DataFrame to markdown conversion failed: {e}")
            return df.to_string(max_rows=100, max_cols=10)
    
    def _format_as_list(self, df) -> str:
        """Format DataFrame as a simple list"""
        try:
            lines = []
            for _, row in df.head(1000).iterrows():
                if len(df.columns) == 1:
                    lines.append(f"• {row.iloc[0]}")
                else:
                    lines.append(f"• {row.iloc[0]}: {row.iloc[1]}")
            
            if len(df) > 1000:
                lines.append(f"[... and {len(df) - 1000} more items]")
            
            return '\n'.join(lines)
        except Exception as e:
            self.log_warning(f"List formatting failed: {e}")
            return df.to_string()
    
    def _detect_text_format(self, text: str, filename: str) -> str:
        """Detect the format of a text file"""
        filename_lower = filename.lower()
        
        if filename_lower.endswith('.py'):
            return 'python'
        elif filename_lower.endswith('.js'):
            return 'javascript'
        elif filename_lower.endswith('.json'):
            return 'json'
        elif filename_lower.endswith('.sql'):
            return 'sql'
        elif filename_lower.endswith(('.yaml', '.yml')):
            return 'yaml'
        elif filename_lower.endswith('.xml'):
            return 'xml'
        elif filename_lower.endswith(('.html', '.htm')):
            return 'html'
        elif filename_lower.endswith('.md'):
            return 'markdown'
        elif filename_lower.endswith('.log'):
            return 'log'
        else:
            return 'text'
    
    def _normalize_table_row(self, row: str, target_cols: int) -> str:
        """Normalize table row to have target number of columns"""
        try:
            parts = row.split('|')
            # Remove first and last empty parts (from leading/trailing |)
            if parts and not parts[0].strip():
                parts = parts[1:]
            if parts and not parts[-1].strip():
                parts = parts[:-1]
            
            # Adjust to target column count
            if len(parts) < target_cols:
                parts.extend([''] * (target_cols - len(parts)))
            elif len(parts) > target_cols:
                parts = parts[:target_cols]
            
            return '| ' + ' | '.join(parts) + ' |'
        except:
            return row
    
    def convert_pdf_to_images(self, file_data: bytes, max_pages: int = 10) -> List[Dict[str, Any]]:
        """
        Convert PDF pages to images for vision/OCR processing
        
        Args:
            file_data: PDF file data as bytes
            max_pages: Maximum number of pages to convert (default 10 - OpenAI limit)
            
        Returns:
            List of dicts with page images as base64 and metadata
        """
        
        try:
            # Convert PDF pages to PIL images
            # Note: This requires poppler-utils to be installed on the system
            images = convert_from_bytes(file_data, dpi=150, fmt='png')
            
            converted_pages = []
            pages_to_process = min(len(images), max_pages)
            
            for i, image in enumerate(images[:pages_to_process]):
                try:
                    # Convert PIL image to base64
                    buffer = BytesIO()
                    image.save(buffer, format='PNG')
                    image_data = buffer.getvalue()
                    base64_data = base64.b64encode(image_data).decode('utf-8')
                    
                    converted_pages.append({
                        'page': i + 1,
                        'base64_data': base64_data,
                        'mimetype': 'image/png',
                        'width': image.width,
                        'height': image.height
                    })
                    
                    self.log_debug(f"Converted PDF page {i+1} to image ({image.width}x{image.height})")
                    
                except Exception as e:
                    self.log_warning(f"Failed to convert page {i+1} to image: {e}")
                    continue
            
            if len(images) > max_pages:
                self.log_info(f"Converted first {max_pages} of {len(images)} PDF pages to images")
            else:
                self.log_info(f"Converted all {len(images)} PDF pages to images")
            
            return converted_pages
            
        except Exception as e:
            self.log_error(f"Failed to convert PDF to images: {e}")
            # Check if it's a poppler issue
            if "poppler" in str(e).lower():
                self.log_error("poppler-utils may not be installed. Install with: apt-get install poppler-utils (Linux) or brew install poppler (Mac)")
            return []
    
    def _is_image_based_pdf(self, pdf_result: Dict[str, Any]) -> bool:
        """
        Detect if a PDF is likely image-based (scanned document)
        
        Args:
            pdf_result: Result from PDF parsing
            
        Returns:
            True if PDF appears to be image-based/scanned
        """
        # Check if we have pages data
        pages = pdf_result.get('pages', [])
        if not pages:
            return False
        
        # Count pages with meaningful text content
        pages_with_text = 0
        total_text_length = 0
        
        for page in pages:
            content = page.get('content', '')
            # Remove whitespace and check length
            clean_content = content.strip()
            
            # Skip page markers and extraction failure messages
            if clean_content and not clean_content.startswith('[') and len(clean_content) > 50:
                pages_with_text += 1
                total_text_length += len(clean_content)
        
        # Heuristics to determine if PDF is image-based:
        # 1. Less than 20% of pages have meaningful text
        # 2. Average text per page is very low (< 100 chars)
        total_pages = pdf_result.get('total_pages', len(pages))
        
        if total_pages == 0:
            return False
        
        text_page_ratio = pages_with_text / total_pages
        avg_text_per_page = total_text_length / total_pages if total_pages > 0 else 0
        
        # Consider it image-based if:
        # - Very few pages have text (< 20%)
        # - OR average text per page is very low (< 100 chars)
        # - OR the entire content is very short relative to page count
        is_likely_image_based = (
            text_page_ratio < 0.2 or 
            avg_text_per_page < 100 or
            (total_pages > 1 and total_text_length < 200)
        )
        
        if is_likely_image_based:
            self.log_info(f"PDF appears to be image-based: {pages_with_text}/{total_pages} pages with text, "
                         f"avg {avg_text_per_page:.0f} chars/page")
        
        return is_likely_image_based