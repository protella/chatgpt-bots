"""Tests for new document handler functionality added for PDF OCR and DOCX alternative parsing"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
import io
from io import BytesIO
import base64
import zipfile
import xml.etree.ElementTree as ET

from document_handler import DocumentHandler


class TestPDFToImageConversion:
    """Test PDF to image conversion functionality"""
    
    @pytest.fixture
    def handler(self):
        """Create a DocumentHandler instance"""
        return DocumentHandler()
    
    @patch('document_handler.convert_from_bytes')
    def test_convert_pdf_to_images_success(self, mock_convert, handler):
        """Test successful PDF to image conversion"""
        # Create mock PIL images
        mock_image1 = Mock()
        mock_image1.width = 800
        mock_image1.height = 1000
        mock_image1.save = Mock()
        
        mock_image2 = Mock()
        mock_image2.width = 800
        mock_image2.height = 1000
        mock_image2.save = Mock()
        
        mock_convert.return_value = [mock_image1, mock_image2]
        
        # Mock base64 encoding
        with patch('base64.b64encode') as mock_b64:
            mock_b64.return_value = b'base64imagedata'
            
            result = handler.convert_pdf_to_images(b'pdf data', max_pages=10)
        
        assert len(result) == 2
        assert result[0]['page'] == 1
        assert result[0]['mimetype'] == 'image/png'
        assert result[0]['width'] == 800
        assert result[0]['height'] == 1000
        
        # Verify convert_from_bytes was called correctly
        mock_convert.assert_called_once_with(b'pdf data', dpi=150, fmt='png')
    
    @patch('document_handler.convert_from_bytes')
    def test_convert_pdf_to_images_max_pages_limit(self, mock_convert, handler):
        """Test PDF conversion respects max_pages limit"""
        # Create 15 mock images
        mock_images = []
        for i in range(15):
            mock_image = Mock()
            mock_image.width = 800
            mock_image.height = 1000
            mock_image.save = Mock()
            mock_images.append(mock_image)
        
        mock_convert.return_value = mock_images
        
        with patch('base64.b64encode') as mock_b64:
            mock_b64.return_value = b'base64imagedata'
            
            # Request max 10 pages
            result = handler.convert_pdf_to_images(b'pdf data', max_pages=10)
        
        # Should only convert 10 pages
        assert len(result) == 10
        assert result[0]['page'] == 1
        assert result[9]['page'] == 10
    
    @patch('document_handler.convert_from_bytes')
    def test_convert_pdf_to_images_failure(self, mock_convert, handler):
        """Test handling of PDF conversion failure"""
        mock_convert.side_effect = Exception("poppler-utils not installed")
        
        result = handler.convert_pdf_to_images(b'pdf data')
        
        assert result == []
    
    @patch('document_handler.convert_from_bytes')
    def test_convert_pdf_to_images_partial_failure(self, mock_convert, handler):
        """Test handling when some pages fail to convert"""
        mock_image1 = Mock()
        mock_image1.width = 800
        mock_image1.height = 1000
        mock_image1.save = Mock()
        
        mock_image2 = Mock()
        mock_image2.width = 800
        mock_image2.height = 1000
        # This image will fail to save
        mock_image2.save = Mock(side_effect=Exception("Save failed"))
        
        mock_convert.return_value = [mock_image1, mock_image2]
        
        with patch('base64.b64encode') as mock_b64:
            mock_b64.return_value = b'base64imagedata'
            
            result = handler.convert_pdf_to_images(b'pdf data')
        
        # Should only get 1 successful conversion
        assert len(result) == 1
        assert result[0]['page'] == 1


class TestImageBasedPDFDetection:
    """Test detection of image-based (scanned) PDFs"""
    
    @pytest.fixture
    def handler(self):
        """Create a DocumentHandler instance"""
        return DocumentHandler()
    
    def test_is_image_based_pdf_with_text(self, handler):
        """Test detection when PDF has good text content"""
        pdf_result = {
            'pages': [
                {'content': 'This is a page with lots of text content that is meaningful and long enough to be considered real text content. Adding more text here to ensure we have plenty of content on this page.'},
                {'content': 'Another page with substantial text content that shows this is a text-based PDF document. This page also has more text to ensure we exceed the minimum thresholds for text detection.'},
            ],
            'total_pages': 2
        }
        
        assert handler._is_image_based_pdf(pdf_result) is False
    
    def test_is_image_based_pdf_without_text(self, handler):
        """Test detection when PDF has minimal text"""
        pdf_result = {
            'pages': [
                {'content': ''},
                {'content': '1'},
                {'content': ''},
            ],
            'total_pages': 3
        }
        
        assert handler._is_image_based_pdf(pdf_result) is True
    
    def test_is_image_based_pdf_with_extraction_errors(self, handler):
        """Test detection when PDF has extraction error messages"""
        pdf_result = {
            'pages': [
                {'content': '[Text extraction failed]'},
                {'content': '[No text content found]'},
                {'content': ''},
            ],
            'total_pages': 3
        }
        
        assert handler._is_image_based_pdf(pdf_result) is True
    
    def test_is_image_based_pdf_mixed_content(self, handler):
        """Test detection with mixed text and image pages"""
        pdf_result = {
            'pages': [
                {'content': 'Page 1 has substantial text content here that is meaningful.'},
                {'content': ''},
                {'content': ''},
                {'content': ''},
                {'content': ''},
            ],
            'total_pages': 5
        }
        
        # Only 1 of 5 pages has text (20%), should be detected as image-based
        assert handler._is_image_based_pdf(pdf_result) is True
    
    def test_is_image_based_pdf_empty_result(self, handler):
        """Test detection with empty PDF result"""
        assert handler._is_image_based_pdf({}) is False
        assert handler._is_image_based_pdf({'pages': []}) is False


class TestDOCXAlternativeParsing:
    """Test alternative DOCX parsing methods"""
    
    @pytest.fixture
    def handler(self):
        """Create a DocumentHandler instance"""
        return DocumentHandler()
    
    def test_parse_docx_alternative_valid_structure(self, handler):
        """Test alternative DOCX parsing with valid ZIP structure"""
        # Create a mock DOCX file (ZIP with word/document.xml)
        xml_content = """<?xml version="1.0"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
            <w:body>
                <w:p>
                    <w:r><w:t>Hello World</w:t></w:r>
                </w:p>
                <w:p>
                    <w:r><w:t>This is a test document.</w:t></w:r>
                </w:p>
            </w:body>
        </w:document>"""
        
        # Create ZIP file in memory
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr('word/document.xml', xml_content)
        
        zip_data = zip_buffer.getvalue()
        
        result = handler.parse_docx_alternative(zip_data, 'test.docx')
        
        assert result['format'] == 'docx'
        assert result['extraction_method'] == 'xml_parsing'
        assert 'Hello World' in result['content']
        assert 'This is a test document' in result['content']
    
    def test_parse_docx_alternative_windows_path_separator(self, handler):
        """Test handling of Windows-style path separators in ZIP"""
        xml_content = """<?xml version="1.0"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
            <w:body>
                <w:p><w:r><w:t>Windows path test</w:t></w:r></w:p>
            </w:body>
        </w:document>"""
        
        # Create ZIP with Windows-style path
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            # Use backslash separator (Windows style)
            zip_file.writestr('word\\document.xml', xml_content)
        
        zip_data = zip_buffer.getvalue()
        
        result = handler.parse_docx_alternative(zip_data, 'test.docx')
        
        assert result['format'] == 'docx'
        assert 'Windows path test' in result['content']
    
    def test_parse_docx_alternative_with_tables(self, handler):
        """Test extraction of tables from DOCX"""
        xml_content = """<?xml version="1.0"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
            <w:body>
                <w:tbl>
                    <w:tr>
                        <w:tc><w:p><w:r><w:t>Header1</w:t></w:r></w:p></w:tc>
                        <w:tc><w:p><w:r><w:t>Header2</w:t></w:r></w:p></w:tc>
                    </w:tr>
                    <w:tr>
                        <w:tc><w:p><w:r><w:t>Data1</w:t></w:r></w:p></w:tc>
                        <w:tc><w:p><w:r><w:t>Data2</w:t></w:r></w:p></w:tc>
                    </w:tr>
                </w:tbl>
            </w:body>
        </w:document>"""
        
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr('word/document.xml', xml_content)
        
        zip_data = zip_buffer.getvalue()
        
        result = handler.parse_docx_alternative(zip_data, 'test.docx')
        
        assert '[Table]' in result['content']
        assert 'Header1 | Header2' in result['content']
        assert 'Data1 | Data2' in result['content']
    
    def test_parse_docx_alternative_invalid_zip(self, handler):
        """Test handling of non-ZIP data"""
        # Not a ZIP file (doesn't start with 'PK')
        invalid_data = b'This is not a ZIP file'
        
        result = handler.parse_docx_alternative(invalid_data, 'test.docx')
        
        # Should return an error for invalid format
        assert 'error' in result
        assert 'Unrecognized file format' in result['content'] or 'not appear to be a valid' in result['error']
    
    def test_parse_docx_alternative_missing_document_xml(self, handler):
        """Test handling of ZIP without document.xml"""
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            zip_file.writestr('some/other/file.txt', 'content')
        
        zip_data = zip_buffer.getvalue()
        
        result = handler.parse_docx_alternative(zip_data, 'test.docx')
        
        # Should return an error for invalid DOCX structure
        assert 'error' in result
        assert 'extraction methods failed' in result.get('error', '') or 'Unable to extract' in result.get('content', '')
    
    @patch('subprocess.run')
    def test_parse_docx_textract_fallback_pandoc_success(self, mock_run, handler):
        """Test pandoc fallback for DOCX extraction"""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Document content extracted by pandoc"
        )
        
        result = handler.parse_docx_textract_fallback(b'docx data', 'test.docx')
        
        assert result['extraction_method'] == 'pandoc'
        assert 'Document content extracted by pandoc' in result['content']
    
    @patch('subprocess.run')
    def test_parse_docx_textract_fallback_pandoc_failure(self, mock_run, handler):
        """Test handling when pandoc is not available"""
        mock_run.side_effect = FileNotFoundError("pandoc not found")
        
        result = handler.parse_docx_textract_fallback(b'docx data', 'test.docx')
        
        assert result['extraction_method'] == 'failed'
        assert 'Unable to extract content' in result['content']


class TestSlackFileURLExtraction:
    """Test updated Slack file URL extraction"""
    
    def test_extract_slack_file_urls_workspace_specific(self):
        """Test extraction of workspace-specific Slack file URLs"""
        from message_processor import MessageProcessor
        
        processor = MessageProcessor()
        
        text = "Check this file: <https://datassential.slack.com/files/U01AF99F3JR/F09BQ7Y1VDE/document.pdf>"
        
        urls = processor._extract_slack_file_urls(text)
        
        assert len(urls) == 1
        assert 'datassential.slack.com/files' in urls[0]
    
    def test_extract_slack_file_urls_generic(self):
        """Test extraction of generic Slack file URLs"""
        from message_processor import MessageProcessor
        
        processor = MessageProcessor()
        
        text = "File here: <https://files.slack.com/files-pri/T123-F456/image.png>"
        
        urls = processor._extract_slack_file_urls(text)
        
        assert len(urls) == 1
        assert 'files.slack.com' in urls[0]
    
    def test_extract_slack_file_urls_multiple(self):
        """Test extraction of multiple Slack file URLs"""
        from message_processor import MessageProcessor
        
        processor = MessageProcessor()
        
        text = """
        Files:
        <https://files.slack.com/files/U123/F456/doc1.pdf>
        <https://myworkspace.slack.com/files/U789/F012/doc2.docx>
        https://another.slack.com/files/U345/F678/image.jpg
        """
        
        urls = processor._extract_slack_file_urls(text)
        
        assert len(urls) == 3
    
    def test_extract_slack_file_urls_no_angle_brackets(self):
        """Test extraction without angle brackets"""
        from message_processor import MessageProcessor
        
        processor = MessageProcessor()
        
        text = "File: https://workspace.slack.com/files/U123/F456/document.pdf"
        
        urls = processor._extract_slack_file_urls(text)
        
        assert len(urls) == 1
        assert 'workspace.slack.com/files' in urls[0]
    
    def test_extract_slack_file_urls_dedupe(self):
        """Test deduplication of URLs"""
        from message_processor import MessageProcessor
        
        processor = MessageProcessor()
        
        text = """
        <https://files.slack.com/files/U123/F456/doc.pdf>
        https://files.slack.com/files/U123/F456/doc.pdf
        <https://files.slack.com/files/U123/F456/doc.pdf>
        """
        
        urls = processor._extract_slack_file_urls(text)
        
        # Should dedupe to 1 URL
        assert len(urls) == 1


class TestForceTextExtraction:
    """Test updated force text extraction with binary format detection"""
    
    @pytest.fixture
    def handler(self):
        """Create a DocumentHandler instance"""
        return DocumentHandler()
    
    def test_force_text_extraction_binary_format(self, handler):
        """Test that binary formats are not decoded as text"""
        # DOCX file data (starts with PK for ZIP)
        docx_data = b'PK\x03\x04' + b'\x00' * 100
        
        result = handler.force_text_extraction(docx_data, 'application/octet-stream', 'test.docx')
        
        assert 'Unable to extract text from corrupted test.docx' in result
        assert 'binary format' in result
    
    def test_force_text_extraction_text_format(self, handler):
        """Test text extraction from text-based formats"""
        # Need at least 80 printable chars for the method to consider it valid text
        text_data = ("This is plain text content that needs to be long enough to pass the threshold check. " +
                    "Adding more text here to ensure we have enough printable characters.").encode('utf-8')
        
        result = handler.force_text_extraction(text_data, 'text/plain', 'test.txt')
        
        assert 'This is plain text content' in result
        assert 'Raw text extraction' in result
    
    def test_force_text_extraction_json_format(self, handler):
        """Test text extraction from JSON files"""
        # Need at least 80 printable chars
        json_data = ('{"key": "value", "number": 123, "description": "This is a longer JSON string to pass the threshold", ' +
                    '"extra": "data to reach 80+ chars"}').encode('utf-8')
        
        result = handler.force_text_extraction(json_data, 'application/json', 'test.json')
        
        assert '"key": "value"' in result
        assert '"number": 123' in result
    
    def test_force_text_extraction_non_printable(self, handler):
        """Test handling of non-printable characters"""
        # Mix of printable and non-printable
        mixed_data = b'Hello\x00\x01\x02World\x03\x04\x05!'
        
        result = handler.force_text_extraction(mixed_data, 'text/plain', 'test.txt')
        
        # Should still extract the printable parts
        assert 'Hello' in result or '[Unable to extract' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])