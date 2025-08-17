# Document Support Implementation Plan

**Document Version:** 1.0  
**Created:** 2025-08-17  
**Purpose:** Add comprehensive document processing capabilities to the chatbot system

---

## Executive Summary

This plan outlines the implementation of full document support for the chatbot, enabling processing of PDFs, Word documents, Excel spreadsheets, and other common file types. The design maintains full document context without truncation, stores complete content in the database, and integrates seamlessly with existing bot architecture.

---

## Requirements

### Core Requirements
- **Full Context:** Never truncate documents - maintain complete content
- **Multiple Documents:** Support multiple documents per request for comparisons (max 10 per request due to slack limitation)
- **Database Storage:** Store full document content with metadata
- **Real-time Processing:** Response times similar to normal ChatGPT conversations
- **Client Agnostic:** Platform-independent implementation (Slack first, Discord compatible)

### Supported File Types

#### Tier 1 (Priority)
- **PDF** (.pdf) - Reports, contracts, mixed content
- **Word** (.docx, .doc) - Documents, proposals, specifications
- **Excel** (.xlsx, .xls, .csv) - Spreadsheets, data exports
- **Text** (.txt, .md) - Plain text, markdown documentation
- **PowerPoint** (.pptx, .ppt) - Presentations

#### Tier 2 (Secondary)
- **Code** (.py, .js, .sql, .json, .yaml, .xml) - Source code files
- **Logs** (.log, .out) - System logs
- **HTML** (.html, .htm) - Web pages, reports
- **RTF** (.rtf) - Rich text format
- **Email** (.msg, .eml) - Email messages

---

## Architecture Design

### 1. Document Processing Pipeline

```
Message Received → _process_attachments() → Document Detection
                                          ↓
                                    Document Parser
                                          ↓
                                    Content Extraction
                                          ↓
                                    DocumentLedger Storage
                                          ↓
                                    Database Persistence
                                          ↓
                                    Intent Classification
                                          ↓
                                    OpenAI Processing
```

### 2. Component Extensions

#### 2.1 Document Handler Module (`document_handler.py`)
New module for document processing, similar to `image_url_handler.py`:

```python
class DocumentHandler:
    """Handles document parsing and content extraction"""
    
    def extract_content(file_data, mime_type, filename):
        """Route to appropriate parser based on file type
        Returns: Dict with content, page_map, total_pages"""
        
    def parse_pdf(file_data):
        """Extract text with page boundaries preserved
        Returns: {
            'content': full_text,
            'pages': [{'page': 1, 'content': '...'}, ...],
            'total_pages': n
        }"""
        
    def parse_docx(file_data):
        """Extract Word document content with page breaks
        Returns similar structure to parse_pdf"""
        
    def parse_excel(file_data):
        """Extract spreadsheet data as structured format
        Returns: {
            'content': combined_text,
            'sheets': [{'name': 'Sheet1', 'content': '...'}, ...],
            'total_sheets': n
        }"""
        
    def parse_text(file_data):
        """Handle plain text files (no page concept)"""
```

#### 2.2 DocumentLedger Extension (`thread_manager.py`)
Extend ThreadState to include document tracking:

```python
@dataclass
class DocumentLedger:
    """Ledger for tracking documents per thread"""
    thread_ts: str
    documents: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_document(self, content, filename, mime_type, page_map=None, ...):
        """Add document to ledger with full content and page structure
        Args:
            content: Full document text
            filename: Original filename
            mime_type: File MIME type
            page_map: Dict with page/sheet structure info
        """
```

#### 2.3 Database Schema Extension (`database.py`)
New documents table for persistence:

```sql
CREATE TABLE documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    content TEXT NOT NULL,  -- Full document content
    page_structure TEXT,    -- JSON with page/sheet boundaries
    total_pages INTEGER,    -- Total page/sheet count
    summary TEXT,           -- AI-generated summary
    metadata_json TEXT,     -- Size, author, creation date, etc.
    message_ts TEXT,        -- Links to specific message
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE,
    INDEX idx_thread_docs (thread_id, created_at),
    INDEX idx_filename (filename)
);
```

### 3. Intent Classification Updates

#### 3.1 Extended Intent Categories
Modify `IMAGE_INTENT_SYSTEM_PROMPT` to include document operations:

```
6. **"document"** - User wants to analyze, query, or process documents
   - REQUIRES: Document files attached (PDF, DOCX, XLSX, etc.)
   - Examples: "review this contract", "analyze this data", "translate this document"
   - Document-specific operations and analysis

7. **"hybrid"** - Mixed content (documents + images) or cross-reference
   - Multiple attachment types requiring combined analysis
   - Examples: "compare these reports" (with multiple PDFs)
```

### 4. Message Processing Integration

#### 4.1 Modify `_process_attachments()` (`message_processor.py`)
Extend to handle documents alongside images:

```python
def _process_attachments(self, message, client):
    image_inputs = []
    document_inputs = []
    unsupported_files = []
    
    for attachment in message.attachments:
        if is_image(attachment):
            # Existing image handling
        elif is_document(attachment):
            # New document handling
            content = document_handler.extract_content(...)
            document_inputs.append({
                "filename": attachment["name"],
                "content": content,
                "type": attachment["type"]
            })
        else:
            unsupported_files.append(attachment)
    
    return image_inputs, document_inputs, unsupported_files
```

#### 4.2 Document Context Building
Add document content to conversation context with flexible structure:

```python
def _build_message_with_documents(self, text, document_inputs):
    """Build message content with adaptive document formatting"""
    content_parts = [{"type": "input_text", "text": text}]
    
    for doc in document_inputs:
        doc_text = format_document_for_context(doc)
        content_parts.append({
            "type": "input_text",
            "text": doc_text
        })
    
    return content_parts

def format_document_for_context(doc):
    """Flexibly format document based on its structure"""
    header = f"\n[Document: {doc['filename']}"
    
    # Add metadata if available
    if doc.get('total_pages'):
        header += f" | Pages: {doc['total_pages']}"
    elif doc.get('total_sheets'):
        header += f" | Sheets: {doc['total_sheets']}"
    header += "]\n"
    
    content = []
    
    # Handle different document structures
    if doc.get('pages'):
        # Page-based documents (PDF/Word)
        for page in doc['pages']:
            content.append(f"[Page {page['page']}]")
            
            # Include tables if present
            if page.get('tables'):
                for table in page['tables']:
                    content.append(table)
            
            # Include regular content
            if page.get('content'):
                content.append(page['content'])
                
    elif doc.get('sheets'):
        # Sheet-based documents (Excel)
        for sheet in doc['sheets']:
            content.append(f"[Sheet: {sheet['name']}]")
            content.append(sheet['content'])
            
    else:
        # Simple documents or fallback
        content.append(doc.get('content', doc.get('text', '')))
    
    return header + '\n'.join(content) + "\n[End Document]"
```

### 5. OpenAI Client Updates

#### 5.1 Document-Aware Processing (`openai_client.py`)
Extend to handle document content in messages:

```python
def create_completion_with_documents(self, messages, document_inputs=None):
    """Create completion with document context"""
    # Include full document content in messages
    # No truncation - send complete context
```

### 6. Parser Implementation Strategy

#### 6.1 Library Dependencies
Add to `requirements.txt`:
```
pypdf2>=3.0.0        # PDF text extraction
pdfplumber>=0.9.0    # Advanced PDF parsing
python-docx>=0.8.11  # Word documents
openpyxl>=3.1.0      # Excel files
pandas>=2.0.0        # Data manipulation
python-magic>=0.4.27 # MIME type detection
```

#### 6.2 Routing Logic
```python
MIME_TYPE_HANDLERS = {
    'application/pdf': parse_pdf,
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': parse_docx,
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': parse_excel,
    'text/plain': parse_text,
    'text/csv': parse_csv,
    # ... additional mappings
}
```

#### 6.3 Flexible Structure Extraction

```python
class StructurePreserver:
    """Flexible system for preserving document structure"""
    
    def extract_with_structure(self, file_data, mime_type):
        """Main entry point - routes to appropriate handler"""
        # Returns dict with:
        # - 'text': plain text fallback
        # - 'structured': best-effort structured representation
        # - 'format_hint': how to present (markdown, plain, table, etc.)
        
    def adaptive_table_extraction(self, data):
        """Convert tables to markdown or structured format"""
        # Handles varying table structures:
        # - Simple grids → markdown tables
        # - Complex nested → hierarchical JSON
        # - Irregular → best-effort key-value pairs
        
    def preserve_hierarchy(self, elements):
        """Maintain document hierarchy (headers, lists, etc.)"""
        # Preserves:
        # - Header levels (H1, H2, H3...)
        # - Nested lists with indentation
        # - Section relationships
```

##### PDF with Structure
```python
def parse_pdf_structured(file_data):
    """Extract PDF preserving tables and structure"""
    import pdfplumber
    
    pages = []
    with pdfplumber.open(BytesIO(file_data)) as pdf:
        for i, page in enumerate(pdf.pages):
            page_data = {'page': i + 1}
            
            # Extract tables if present
            tables = page.extract_tables()
            if tables:
                # Convert to markdown tables
                page_data['tables'] = []
                for table in tables:
                    if table and len(table) > 0:
                        # Flexible markdown conversion
                        md_table = flexible_table_to_markdown(table)
                        page_data['tables'].append(md_table)
            
            # Extract text with structure hints
            text = page.extract_text() or ""
            page_data['content'] = text
            
            # Try to detect structure (headers, lists, etc.)
            page_data['structure_hints'] = detect_structure(text)
            
            pages.append(page_data)
    
    return {
        'pages': pages,
        'total_pages': len(pages),
        'has_tables': any('tables' in p for p in pages)
    }

def flexible_table_to_markdown(table_data):
    """Convert variable table structures to markdown"""
    if not table_data or not table_data[0]:
        return ""
    
    # Handle irregular tables gracefully
    max_cols = max(len(row) if row else 0 for row in table_data)
    
    # Normalize rows to same width
    normalized = []
    for row in table_data:
        if row:
            # Pad short rows, handle None values
            norm_row = [(cell or "") for cell in row]
            norm_row.extend([""] * (max_cols - len(norm_row)))
            normalized.append(norm_row)
    
    if not normalized:
        return ""
    
    # Build markdown table
    lines = []
    
    # Header row
    header = "| " + " | ".join(str(cell) for cell in normalized[0]) + " |"
    lines.append(header)
    
    # Separator
    separator = "|" + "|".join([" --- " for _ in range(max_cols)]) + "|"
    lines.append(separator)
    
    # Data rows
    for row in normalized[1:]:
        line = "| " + " | ".join(str(cell) for cell in row) + " |"
        lines.append(line)
    
    return "\n".join(lines)
```

##### Word with Structure
```python
def parse_docx_structured(file_data):
    """Extract Word preserving formatting and structure"""
    from docx import Document
    
    doc = Document(BytesIO(file_data))
    content_blocks = []
    
    for element in doc.element.body:
        if element.tag.endswith('p'):  # Paragraph
            para = extract_paragraph_with_style(element)
            content_blocks.append(para)
            
        elif element.tag.endswith('tbl'):  # Table
            table = extract_table_flexible(element)
            content_blocks.append({
                'type': 'table',
                'content': table
            })
    
    # Group into pages if page breaks detected
    pages = group_by_page_breaks(content_blocks)
    
    return {
        'pages': pages,
        'structure': content_blocks,
        'has_tables': any(b['type'] == 'table' for b in content_blocks)
    }
```

##### Excel with Flexible Structure
```python
def parse_excel_adaptive(file_data):
    """Extract Excel with adaptive structure preservation"""
    import pandas as pd
    
    # Read all sheets
    all_sheets = pd.read_excel(BytesIO(file_data), sheet_name=None)
    
    sheets = []
    for sheet_name, df in all_sheets.items():
        sheet_data = {
            'name': sheet_name,
            'rows': len(df),
            'cols': len(df.columns)
        }
        
        # Detect data type/structure
        if df.empty:
            sheet_data['content'] = "[Empty sheet]"
        elif is_simple_table(df):
            # Convert to markdown table
            sheet_data['content'] = df.to_markdown(index=False)
            sheet_data['format'] = 'table'
        elif is_pivot_like(df):
            # Preserve pivot structure
            sheet_data['content'] = format_pivot_table(df)
            sheet_data['format'] = 'pivot'
        elif is_list_data(df):
            # Simple list format
            sheet_data['content'] = format_as_list(df)
            sheet_data['format'] = 'list'
        else:
            # Fallback: key-value pairs or JSON
            sheet_data['content'] = df.to_string()
            sheet_data['format'] = 'raw'
            
        sheets.append(sheet_data)
    
    return {
        'sheets': sheets,
        'total_sheets': len(sheets)
    }

def is_simple_table(df):
    """Detect if DataFrame is a regular table"""
    # Has column headers and regular rows
    return (not df.columns.str.contains('Unnamed').all() and 
            len(df) > 0 and len(df.columns) > 1)
```

#### 6.4 Error Handling and Sanitization

##### Safe Content Extraction
```python
class DocumentHandler:
    def safe_extract_content(self, file_data, mime_type, filename):
        """Safely extract with error recovery"""
        try:
            # Primary extraction attempt
            result = self.extract_content(file_data, mime_type, filename)
            
            # Sanitize the content
            result['content'] = self.sanitize_content(result['content'])
            return result
            
        except Exception as e:
            self.log_error(f"Failed to parse {filename}: {e}")
            
            # Fallback: try basic text extraction
            try:
                raw_text = self.force_text_extraction(file_data, mime_type)
                return {
                    'content': self.sanitize_content(raw_text),
                    'error': f'Partial extraction - document may be malformed',
                    'format': 'text'
                }
            except:
                # Last resort: mark as unparseable
                return {
                    'content': f'[Unable to parse {filename} - document appears malformed]',
                    'error': 'Document could not be parsed',
                    'format': 'error'
                }
    
    def sanitize_content(self, text):
        """Sanitize content to prevent format breaking"""
        if not text:
            return ""
        
        # Escape/fix problematic patterns
        sanitized = text
        
        # Balance code blocks
        code_blocks = sanitized.count('```')
        if code_blocks % 2 != 0:
            sanitized += '\n```'  # Close unclosed code block
        
        # Escape our document markers to prevent injection
        sanitized = sanitized.replace('[Document:', '[Document\\:')
        sanitized = sanitized.replace('[End Document]', '[End\\ Document]')
        sanitized = sanitized.replace('[Page ', '[Page\\ ')
        sanitized = sanitized.replace('[Sheet:', '[Sheet\\:')
        
        # Fix common table issues
        sanitized = self.fix_markdown_tables(sanitized)
        
        # Limit consecutive newlines
        sanitized = re.sub(r'\n{4,}', '\n\n\n', sanitized)
        
        # Remove null bytes and control characters
        sanitized = sanitized.replace('\x00', '')
        sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\n\r\t')
        
        # Size limit per document (1MB of text)
        if len(sanitized) > 1_000_000:
            sanitized = sanitized[:1_000_000] + '\n[Content truncated due to size]'
        
        return sanitized
    
    def fix_markdown_tables(self, text):
        """Fix common markdown table issues"""
        lines = text.split('\n')
        in_table = False
        fixed_lines = []
        
        for line in lines:
            if '|' in line:
                # Ensure proper table row format
                if not line.strip().startswith('|'):
                    line = '| ' + line
                if not line.strip().endswith('|'):
                    line = line + ' |'
                
                # Ensure consistent column count
                col_count = line.count('|') - 1
                if in_table and col_count != table_col_count:
                    # Pad or truncate to match
                    line = self.normalize_table_row(line, table_col_count)
                else:
                    table_col_count = col_count
                    in_table = True
            elif in_table and line.strip() and '|' not in line:
                # Table ended abruptly
                in_table = False
                
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
```

##### Parser-Specific Protection
```python
def parse_pdf_protected(file_data):
    """Parse PDF with protection against malformed content"""
    try:
        with pdfplumber.open(BytesIO(file_data)) as pdf:
            # Normal extraction with timeout
            pages = []
            for i, page in enumerate(pdf.pages):
                if i > 1000:  # Sanity limit
                    pages.append({'page': i+1, 'content': '[Remaining pages omitted - document too large]'})
                    break
                    
                try:
                    page_text = page.extract_text(timeout=30) or ""
                    pages.append({'page': i+1, 'content': page_text})
                except:
                    pages.append({'page': i+1, 'content': '[Page extraction failed]'})
                    
            return {'pages': pages, 'total_pages': len(pdf.pages)}
            
    except Exception as e:
        # Try PyPDF2 as fallback
        try:
            reader = PyPDF2.PdfReader(BytesIO(file_data))
            text = []
            for page in reader.pages[:100]:  # Limit pages
                text.append(page.extract_text())
            return {'content': '\n'.join(text), 'error': 'Fallback extraction used'}
        except:
            return {'content': '[PDF appears corrupted or password protected]', 'error': True}

def parse_excel_protected(file_data):
    """Parse Excel with protection against malformed data"""
    try:
        # Try with pandas first
        dfs = pd.read_excel(BytesIO(file_data), sheet_name=None)
        return parse_dataframes_safe(dfs)
    except:
        # Try CSV interpretation
        try:
            df = pd.read_csv(BytesIO(file_data), on_bad_lines='skip')
            return {'content': df.to_markdown(), 'format': 'csv_fallback'}
        except:
            # Try raw text extraction
            try:
                text = file_data.decode('utf-8', errors='ignore')
                return {'content': text[:50000], 'format': 'raw_text', 'error': 'Spreadsheet unreadable'}
            except:
                return {'content': '[Spreadsheet format unreadable]', 'error': True}

def parse_dataframes_safe(dfs):
    """Safely parse DataFrames with size limits"""
    sheets = []
    for name, df in list(dfs.items())[:20]:  # Max 20 sheets
        if len(df) > 10000:  # Row limit
            df = df.head(10000)
            truncated = True
        else:
            truncated = False
            
        # Sanitize column names
        df.columns = [str(col)[:50] for col in df.columns]
        
        sheet_data = {
            'name': str(name)[:50],
            'content': df.to_markdown(max_cols=50),
            'truncated': truncated
        }
        sheets.append(sheet_data)
        
    return {'sheets': sheets}
```

##### Content Validation
```python
def validate_document_content(doc):
    """Validate document before including in context"""
    # Check for suspicious patterns
    suspicious_patterns = [
        r'\[Document:.*\[Document:',  # Nested document markers
        r'```[\s\S]*```[\s\S]*```[\s\S]*```',  # Too many code blocks
        r'(\n\|[^\n]*){100,}',  # Extremely long tables
    ]
    
    content = doc.get('content', '')
    for pattern in suspicious_patterns:
        if re.search(pattern, content):
            doc['warning'] = 'Document structure appears irregular'
            doc['content'] = self.sanitize_aggressive(content)
    
    return doc

def format_document_with_safety(doc):
    """Format document with safety checks"""
    # Validate first
    doc = validate_document_content(doc)
    
    if doc.get('error'):
        # Include error notice
        header = f"\n[Document: {doc['filename']} - WARNING: {doc.get('error', 'Parse error')}]\n"
    else:
        header = f"\n[Document: {doc['filename']}]\n"
    
    content = doc.get('content', '[No content extracted]')
    
    # Final safety check on complete document
    full_doc = header + content + "\n[End Document]"
    
    # Ensure it won't break the message format
    if full_doc.count('[Document:') != full_doc.count('[End Document]'):
        # Fix document boundary issues
        full_doc = header + "[Content sanitized due to format issues]\n[End Document]"
    
    return full_doc
```

#### 6.5 Hybrid Detection
For PDFs with mixed content:
```python
def needs_vision_model(pdf_data):
    """Detect if PDF requires vision model"""
    # Check for:
    # - Low text extraction ratio
    # - Presence of images/diagrams  
    # - Scanned document indicators
    return has_images or is_scanned or low_text_ratio
```

---

## Implementation Steps

### Phase 1: Core Infrastructure (Week 1)
1. Create `document_handler.py` module
2. Add DocumentLedger to `thread_manager.py`
3. Create documents table in `database.py`
4. Add document parser dependencies to `requirements.txt`

### Phase 2: Parser Implementation (Week 1)
1. Implement PDF parser with page boundary preservation
2. Implement Word document parser with page break detection
3. Implement Excel/CSV parser with sheet structure
4. Implement plain text parser
5. Add MIME type detection and routing

### Phase 3: Integration (Week 2)
1. Extend `_process_attachments()` in `message_processor.py`
2. Update intent classification to include document intents
3. Modify OpenAI client to handle document content
4. Update thread state management for documents
5. Implement comprehensive error handling and sanitization

### Phase 4: Testing & Refinement (Week 2)
1. Write unit tests for document parsers
2. Write integration tests for document flow
3. Test with various document types and sizes
4. Performance optimization for large documents

---

## Testing Strategy

### Unit Tests
- `test_document_handler.py` - Parser functionality
- `test_document_ledger.py` - Document storage
- `test_document_intent.py` - Intent classification with documents

### Integration Tests
- End-to-end document processing flow
- Database persistence and retrieval
- Multiple document handling
- Mixed content (documents + images)

### Test Documents
Create test fixtures with:
- Simple text PDF
- PDF with tables and mixed structure
- Complex PDF with images
- Word document with formatting and tables
- Excel with multiple sheets and formulas
- Excel with pivot tables and irregular data
- CSV with various delimiters
- Large documents (>10MB)
- Non-English documents
- Documents with malformed/irregular structures

### Error Handling Tests
Test error recovery for:
- Corrupted PDFs
- Password-protected documents
- Malformed Excel files (invalid formulas, circular references)
- Documents with unclosed markdown syntax (```, tables)
- Files with control characters or null bytes
- Extremely large tables (1000+ rows)
- Nested document boundary markers
- Mixed encoding issues
- Files that appear to be one type but are another

---

## Key Design Decisions

### 1. Full Content Storage
- Store complete document content in database
- No summarization or truncation during storage
- Preserve structural relationships (tables, lists, hierarchies)
- Store both raw text and structured representations
- Summaries generated on-demand if needed

### 2. Synchronous Processing
- Parse documents synchronously during message processing
- No background jobs or queues (maintains real-time response)
- Leverage OpenAI's context window for processing

### 3. Client-Agnostic Design
- Document handling in platform-independent layer
- Platform clients only handle file download/upload
- Core processing logic shared across all platforms

### 4. Memory Management
- Document content stored in database, not memory
- ThreadState maintains document metadata only
- Full content loaded from DB when needed

### 5. Flexible Structure Preservation
- Adaptive parsing based on document type and content
- Best-effort structure extraction (tables, lists, hierarchies)
- Graceful degradation for malformed documents
- Always maintain plain text fallback

### 6. Robust Error Handling
- Multi-level fallback strategy for failed parsing
- Content sanitization to prevent format injection
- Protection against malformed markdown/tables
- Size limits to prevent memory issues (1MB text per doc, 10K rows per sheet)
- Validation of document boundaries and structure
- Clear error messages when documents cannot be parsed

---

## Migration & Rollback Plan

### Migration
1. Deploy new document tables without affecting existing functionality
2. Add document handler module without modifying existing code paths
3. Gradually enable document support with feature flag
4. Full rollout after testing

### Rollback
- Document support can be disabled by reverting `_process_attachments()`
- Database tables remain but unused
- No impact on existing image/text functionality

---

## Performance Considerations

### Context Window Management
- GPT-4/5 models support 128K+ tokens
- Average page ~500 tokens
- Can handle 200+ page documents
- Page markers add minimal overhead (~20 tokens per page)
- Monitor token usage and warn if approaching limits

### Database Performance
- Index on thread_id for fast retrieval
- Store page_structure as compressed JSON
- Consider compression for large documents
- Implement cleanup for old documents (configurable retention)

### Response Time Optimization
- Stream responses while processing large documents
- Cache parsed content for repeated access
- Parallel parsing for multiple documents

---

## Security & Privacy

### Data Handling
- Documents sent to OpenAI per existing business agreement
- No local persistent storage outside database
- Automatic cleanup based on retention policy

### Access Control
- Documents scoped to thread/channel
- No cross-thread document access
- Platform-level permissions enforced

---

## Success Metrics

1. **Functionality**
   - Successfully parse 95%+ of standard business documents
   - Support documents up to 50MB
   - Handle 10+ documents per request

2. **Performance**
   - Document processing < 5 seconds for typical files
   - No degradation in text/image response times
   - Memory usage stable with document processing

3. **Reliability**
   - All existing tests continue passing
   - No crashes with malformed documents
   - Graceful handling of unsupported formats

---

## Future Enhancements

1. **Vector Search** (v2)
   - Add embeddings for semantic search
   - Implement document chunks for large files
   - Cross-document knowledge base

2. **OCR Integration** (v2)
   - Add Tesseract for scanned documents
   - Integrate with cloud OCR services
   - Automatic language detection

3. **Advanced Analytics** (v3)
   - Document comparison tools
   - Change tracking between versions
   - Automated report generation

---

## Appendix: File Type Details

### PDF Processing
- Primary: pdfplumber for structure-aware extraction
- Extract tables as markdown or structured data
- Fallback: PyPDF2 for simpler PDFs
- Page boundaries preserved in extraction
- Vision: Route image-heavy PDFs to GPT-4V

### Excel Processing
- Use pandas for flexible data handling
- Adaptive formatting based on data structure:
  - Regular tables → Markdown tables
  - Pivot tables → Preserve pivot structure
  - Lists → Simple list format
  - Irregular data → Key-value pairs or raw text
- Each sheet extracted separately with name
- Preserve formulas and cell relationships where possible
- Support multi-sheet workbooks

### Word Processing
- python-docx for content extraction
- Preserve document structure:
  - Headers and hierarchy (H1, H2, H3...)
  - Tables as markdown
  - Lists with proper nesting
  - Formatting context (bold, italic for emphasis)
- Detect page breaks via XML parsing
- Handle embedded images/objects

### Structure Reference Examples
Users can reference documents naturally:
- "On page 5 of contract.pdf..."
- "In Sheet2 of the Excel file..."
- "The table on page 3 shows..."
- "What's the total in column B?"
- "Compare the Q1 data with Q2..."
- "In the Executive Summary section..."