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

from logger import LoggerMixin

# pandas will be imported dynamically when needed

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
        self._dependencies_checked = False
        self._available_parsers = {}
        
    def _check_dependencies(self):
        """Check which parsing libraries are available"""
        if self._dependencies_checked:
            return
            
        try:
            import pdfplumber
            self._available_parsers['pdfplumber'] = pdfplumber
            self.log_debug("pdfplumber library available")
        except ImportError:
            self.log_warning("pdfplumber not available - PDF parsing will use fallback")
            
        try:
            import PyPDF2
            self._available_parsers['PyPDF2'] = PyPDF2
            self.log_debug("PyPDF2 library available")
        except ImportError:
            self.log_warning("PyPDF2 not available - PDF fallback disabled")
            
        try:
            from docx import Document
            self._available_parsers['python-docx'] = Document
            self.log_debug("python-docx library available")
        except ImportError:
            self.log_warning("python-docx not available - Word document parsing disabled")
            
        try:
            import openpyxl
            self._available_parsers['openpyxl'] = openpyxl
            self.log_debug("openpyxl library available")
        except ImportError:
            self.log_warning("openpyxl not available - Excel parsing will use pandas only")
            
        # pandas should be available from requirements
        try:
            import pandas as pd
            self._available_parsers['pandas'] = pd
            self.log_debug("pandas library available")
        except ImportError:
            self.log_error("pandas not available - spreadsheet parsing severely limited")
            
        self._dependencies_checked = True
    
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
        self._check_dependencies()
        
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
        if 'pdfplumber' in self._available_parsers:
            return self._parse_pdf_with_pdfplumber(file_data, filename)
        elif 'PyPDF2' in self._available_parsers:
            return self._parse_pdf_with_pypdf2(file_data, filename)
        else:
            raise ImportError("No PDF parsing libraries available")
    
    def _parse_pdf_with_pdfplumber(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """Parse PDF using pdfplumber for advanced structure extraction"""
        pdfplumber = self._available_parsers['pdfplumber']
        
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
        PyPDF2 = self._available_parsers['PyPDF2']
        
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
        if 'python-docx' not in self._available_parsers:
            # Fallback to text extraction
            return self.parse_text(file_data, filename)
        
        Document = self._available_parsers['python-docx']
        
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
            raise Exception(f"Word document parsing failed: {e}")
    
    def parse_excel_adaptive(self, file_data: bytes, filename: str) -> Dict[str, Any]:
        """
        Extract Excel/CSV content with adaptive structure preservation
        
        Args:
            file_data: Excel/CSV file data as bytes
            filename: Original filename for error reporting
            
        Returns:
            Dict with sheets, content, and structure info
        """
        if 'pandas' not in self._available_parsers:
            raise ImportError("pandas not available for spreadsheet parsing")
        
        pd = self._available_parsers['pandas']
        
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
        # Try direct text decoding
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                text = file_data.decode(encoding, errors='ignore')
                # Clean up the text
                text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
                if text.strip():
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