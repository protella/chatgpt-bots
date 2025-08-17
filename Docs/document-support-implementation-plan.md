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
        """Route to appropriate parser based on file type"""
        
    def parse_pdf(file_data):
        """Extract text and detect if vision model needed"""
        
    def parse_docx(file_data):
        """Extract Word document content"""
        
    def parse_excel(file_data):
        """Extract spreadsheet data as structured format"""
        
    def parse_text(file_data):
        """Handle plain text files"""
```

#### 2.2 DocumentLedger Extension (`thread_manager.py`)
Extend ThreadState to include document tracking:

```python
@dataclass
class DocumentLedger:
    """Ledger for tracking documents per thread"""
    thread_ts: str
    documents: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_document(self, content, filename, mime_type, ...):
        """Add document to ledger with full content"""
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
    summary TEXT,           -- AI-generated summary
    metadata_json TEXT,     -- Size, pages, author, etc.
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
Add document content to conversation context:

```python
def _build_message_with_documents(self, text, document_inputs):
    """Build message content including document data"""
    content_parts = [{"type": "input_text", "text": text}]
    
    for doc in document_inputs:
        content_parts.append({
            "type": "input_text",
            "text": f"\n[Document: {doc['filename']}]\n{doc['content']}\n[End Document]"
        })
    
    return content_parts
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

#### 6.3 Hybrid Detection
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
1. Implement PDF parser with text extraction
2. Implement Word document parser
3. Implement Excel/CSV parser
4. Implement plain text parser
5. Add MIME type detection and routing

### Phase 3: Integration (Week 2)
1. Extend `_process_attachments()` in `message_processor.py`
2. Update intent classification to include document intents
3. Modify OpenAI client to handle document content
4. Update thread state management for documents

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
- Complex PDF with images
- Word document with formatting
- Excel with multiple sheets
- Large documents (>10MB)
- Non-English documents

---

## Key Design Decisions

### 1. Full Content Storage
- Store complete document content in database
- No summarization or truncation during storage
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
- Monitor token usage and warn if approaching limits

### Database Performance
- Index on thread_id for fast retrieval
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
- Primary: PyPDF2 for text extraction
- Fallback: pdfplumber for complex layouts
- Vision: Route image-heavy PDFs to GPT-4V

### Excel Processing
- Use pandas for data manipulation
- Preserve formulas and formatting metadata
- Support multi-sheet workbooks

### Word Processing
- python-docx for content extraction
- Preserve formatting for context
- Handle embedded images/objects