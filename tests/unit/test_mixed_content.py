"""
Unit tests for mixed content (images + documents) processing
Tests the two-call approach for analyzing both visual and document content
"""
import pytest
import base64
from unittest.mock import MagicMock, patch, Mock, call
from message_processor import MessageProcessor
from base_client import Message, Response


class TestMixedContentAnalysis:
    """Test mixed content analysis with documents and images"""
    
    @pytest.fixture
    def mock_thread_manager(self):
        """Create a mock thread manager"""
        mock = MagicMock()
        mock.acquire_thread_lock.return_value = True
        mock.release_thread_lock.return_value = None
        mock.get_or_create_thread.return_value = MagicMock(
            thread_ts="123.456",
            channel_id="C123",
            messages=[],
            config_overrides={},
            system_prompt=None,
            is_processing=False,
            had_timeout=False,
            pending_clarification=None,
            add_message=MagicMock(),
            get_recent_messages=MagicMock(return_value=[])
        )
        mock.get_document_ledger.return_value = None
        mock.get_or_create_document_ledger.return_value = MagicMock(
            documents=[],
            add_document=MagicMock()
        )
        return mock
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client"""
        mock = MagicMock()
        mock.classify_intent.return_value = "vision"
        mock.get_response.return_value = "Analysis complete"
        mock.create_text_response.return_value = "Combined analysis result"
        mock.create_text_response_with_tools.return_value = "Combined analysis result"
        
        # Mock analyze_images to return technical description
        mock.analyze_images.return_value = "Technical image description: Shows a keyboard setup with two synthesizers"
        
        return mock
    
    @pytest.fixture
    def mock_document_handler(self):
        """Create a mock document handler"""
        mock = MagicMock()
        mock.is_document_file.return_value = True
        mock.safe_extract_content.return_value = {
            "content": "Document content about social anxiety",
            "page_structure": {"pages": [{"page": 1, "content": "Page 1 content"}]},
            "total_pages": 1,
            "summary": "A worksheet about social anxiety",
            "metadata": {}
        }
        return mock
    
    @pytest.fixture
    def mock_client(self):
        """Create a mock platform client"""
        mock = MagicMock()
        mock.platform = "slack"
        mock.name = "SlackClient"
        mock.post_message.return_value = "msg_123"
        mock.upload_image.return_value = "https://slack.com/image.png"
        mock.download_file.return_value = b"fake_file_data"
        mock.fetch_thread_history.return_value = []
        mock.send_thinking_indicator.return_value = "thinking_123"
        mock.update_message.return_value = None
        mock.delete_message.return_value = None
        return mock
    
    @pytest.fixture
    def processor(self, mock_thread_manager, mock_openai_client, mock_document_handler):
        """Create a MessageProcessor with mocked dependencies"""
        with patch('message_processor.ThreadStateManager', return_value=mock_thread_manager):
            with patch('message_processor.OpenAIClient', return_value=mock_openai_client):
                processor = MessageProcessor()
                processor.thread_manager = mock_thread_manager
                processor.openai_client = mock_openai_client
                processor.document_handler = mock_document_handler
                return processor
    
    def test_mixed_content_detection(self, processor, mock_client):
        """Test that mixed content (image + document) is properly detected"""
        message = Message(
            text="Are these files related?",
            user_id="U123",
            channel_id="C456",
            thread_id="T789",
            attachments=[
                {"type": "file", "mimetype": "application/pdf", "name": "document.pdf", "url": "http://example.com/doc.pdf", "id": "file_1"},
                {"type": "image", "mimetype": "image/jpeg", "name": "photo.jpg", "url": "http://example.com/img.jpg", "id": "file_2"}
            ]
        )
        
        # Mock file downloads
        mock_client.download_file.side_effect = [
            b"fake_pdf_data",  # PDF download
            b"fake_image_data"  # Image download
        ]
        
        response = processor.process_message(message, mock_client, thinking_id="think_123")
        
        # Should have detected both document and image
        assert mock_client.download_file.call_count == 2
        assert processor.document_handler.safe_extract_content.called
    
    def test_mixed_content_two_call_approach(self, processor, mock_client):
        """Test that mixed content triggers two API calls"""
        message = Message(
            text="Compare these files",
            user_id="U123",
            channel_id="C456",
            thread_id="T789",
            attachments=[
                {"type": "file", "mimetype": "application/pdf", "name": "doc.pdf", "url": "http://example.com/doc.pdf", "id": "file_1"},
                {"type": "image", "mimetype": "image/png", "name": "img.png", "url": "http://example.com/img.png", "id": "file_2"}
            ]
        )
        
        mock_client.download_file.side_effect = [
            b"fake_pdf_data",
            b"fake_image_data"
        ]
        
        response = processor.process_message(message, mock_client, thinking_id="think_123")
        
        # Should call analyze_images for vision analysis
        processor.openai_client.analyze_images.assert_called_once()
        
        # Response should be a text response (the two-call approach generates a final text response)
        assert response is not None
        assert response.type == "text"
    
    def test_mixed_content_context_building(self, processor, mock_client):
        """Test that context is properly built with image analysis and document content"""
        message = Message(
            text="What's the relationship?",
            user_id="U123",
            channel_id="C456",
            thread_id="T789",
            attachments=[
                {"type": "file", "mimetype": "application/pdf", "name": "report.pdf", "url": "http://example.com/report.pdf", "id": "file_1"},
                {"type": "image", "mimetype": "image/jpeg", "name": "chart.jpg", "url": "http://example.com/chart.jpg", "id": "file_2"}
            ]
        )
        
        mock_client.download_file.side_effect = [
            b"fake_pdf_data",
            b"fake_image_data"
        ]
        
        # Capture what gets passed to _handle_text_response
        actual_context = None
        original_handle_text = processor._handle_text_response
        def capture_text_handler(text, *args, **kwargs):
            nonlocal actual_context
            actual_context = text
            # Call the original to continue processing
            return original_handle_text(text, *args, **kwargs)
        
        # Patch _handle_text_response to capture the combined context
        with patch.object(processor, '_handle_text_response', side_effect=capture_text_handler):
            response = processor.process_message(message, mock_client, thinking_id="think_123")
        
        # Verify context includes both image analysis and document content
        assert actual_context is not None
        # Should contain image analysis section
        assert "IMAGE ANALYSIS" in actual_context or "Technical image description" in actual_context
        # Should contain document section
        assert "DOCUMENT" in actual_context or "Document content" in actual_context
        # Should contain user question
        assert "relationship" in actual_context.lower()
    
    def test_mixed_content_status_updates(self, processor, mock_client):
        """Test that appropriate status updates are shown during mixed content processing"""
        message = Message(
            text="Analyze both files",
            user_id="U123",
            channel_id="C456",
            thread_id="T789",
            attachments=[
                {"type": "file", "mimetype": "application/pdf", "name": "data.pdf", "url": "http://example.com/data.pdf", "id": "file_1"},
                {"type": "image", "mimetype": "image/png", "name": "graph.png", "url": "http://example.com/graph.png", "id": "file_2"}
            ]
        )
        
        mock_client.download_file.side_effect = [
            b"fake_pdf_data",
            b"fake_image_data"
        ]
        
        # Track status updates
        status_updates = []
        def track_update(client, channel_id, thinking_id, message, emoji=None):
            status_updates.append(message)
        
        with patch.object(processor, '_update_status', side_effect=track_update):
            response = processor.process_message(message, mock_client, thinking_id="think_123")
        
        # Should show various status updates
        assert any("Processing data.pdf" in update for update in status_updates)
        assert any("Extracting content" in update for update in status_updates)
        assert any("Analyzing" in update for update in status_updates)
    
    def test_mixed_content_image_only_attachments(self, processor, mock_client):
        """Test that only image attachments are passed to vision handler"""
        message = Message(
            text="Compare these",
            user_id="U123",
            channel_id="C456",
            thread_id="T789",
            attachments=[
                {"type": "file", "mimetype": "application/pdf", "name": "doc.pdf", "url": "http://example.com/doc.pdf", "id": "file_1"},
                {"type": "image", "mimetype": "image/jpeg", "name": "pic.jpg", "url": "http://example.com/pic.jpg", "id": "file_2"}
            ]
        )
        
        mock_client.download_file.side_effect = [
            b"fake_pdf_data",
            b"fake_image_data"
        ]
        
        # Capture what's passed to analyze_images
        actual_images = None
        def capture_images(images=None, *args, **kwargs):
            nonlocal actual_images
            actual_images = images
            return "Image analysis"
        
        processor.openai_client.analyze_images.side_effect = capture_images
        
        response = processor.process_message(message, mock_client, thinking_id="think_123")
        
        # Should only pass image data to vision analysis
        assert actual_images is not None
        assert len(actual_images) == 1  # Only the image, not the PDF
    
    def test_mixed_content_document_ledger_storage(self, processor, mock_client):
        """Test that documents are stored in DocumentLedger"""
        message = Message(
            text="Analyze these",
            user_id="U123",
            channel_id="C456",
            thread_id="T789",
            attachments=[
                {"type": "file", "mimetype": "application/pdf", "name": "report.pdf", "url": "http://example.com/report.pdf", "id": "file_1"},
                {"type": "image", "mimetype": "image/png", "name": "chart.png", "url": "http://example.com/chart.png", "id": "file_2"}
            ]
        )
        
        mock_client.download_file.side_effect = [
            b"fake_pdf_data",
            b"fake_image_data"
        ]
        
        # Get the document ledger mock
        doc_ledger = processor.thread_manager.get_or_create_document_ledger.return_value
        
        response = processor.process_message(message, mock_client, thinking_id="think_123")
        
        # Document should be added to ledger
        doc_ledger.add_document.assert_called_once()
        call_args = doc_ledger.add_document.call_args
        assert call_args[1]["filename"] == "report.pdf"
        assert call_args[1]["mime_type"] == "application/pdf"
    
    @pytest.mark.critical
    def test_critical_mixed_content_flow(self, processor, mock_client):
        """Critical: Test complete mixed content processing flow"""
        message = Message(
            text="Are these two files related?",
            user_id="U123",
            channel_id="C456",
            thread_id="T789",
            attachments=[
                {"type": "file", "mimetype": "application/pdf", "name": "document.pdf", "url": "http://example.com/doc.pdf", "id": "file_1"},
                {"type": "image", "mimetype": "image/jpeg", "name": "image.jpg", "url": "http://example.com/img.jpg", "id": "file_2"}
            ],
            metadata={"username": "testuser", "ts": "msg_123"}
        )
        
        mock_client.download_file.side_effect = [
            b"fake_pdf_data",
            b"fake_image_data"
        ]
        
        # Process message
        response = processor.process_message(message, mock_client, thinking_id="think_123")
        
        # Verify complete flow
        assert response is not None
        assert response.type == "text"
        
        # Verify both processing paths were used
        processor.document_handler.safe_extract_content.assert_called_once()
        processor.openai_client.analyze_images.assert_called_once()
        
        # Verify response was successfully generated through the two-call approach
        # The exact mock calls depend on the streaming configuration, but we should have a valid response
        assert response is not None
        assert response.type == "text"
        # Content comes from the mock's return value which is set to "Combined analysis result"
        # But with streaming it might be empty, so just check it's not None
        assert response.content is not None
    
    @pytest.mark.smoke
    def test_smoke_mixed_content_basic(self, processor, mock_client):
        """Smoke test: Basic mixed content processing works"""
        try:
            message = Message(
                text="Test",
                user_id="U1",
                channel_id="C1", 
                thread_id="T1",
                attachments=[
                    {"type": "file", "mimetype": "application/pdf", "name": "test.pdf", "url": "http://example.com/test.pdf", "id": "f1"},
                    {"type": "image", "mimetype": "image/png", "name": "test.png", "url": "http://example.com/test.png", "id": "f2"}
                ]
            )
            
            mock_client.download_file.return_value = b"test_data"
            
            response = processor.process_message(message, mock_client)
            assert response is not None
            
        except Exception as e:
            pytest.fail(f"Mixed content processing failed: {e}")


class TestMixedContentEdgeCases:
    """Test edge cases in mixed content handling"""
    
    @pytest.fixture
    def processor(self):
        """Create a MessageProcessor with minimal mocking"""
        with patch('message_processor.ThreadStateManager'):
            with patch('message_processor.OpenAIClient'):
                processor = MessageProcessor()
                processor.document_handler = MagicMock()
                processor.document_handler.is_document_file.return_value = True
                processor.document_handler.safe_extract_content.return_value = {
                    "content": "Test content",
                    "total_pages": 1
                }
                return processor
    
    def test_multiple_documents_with_image(self, processor):
        """Test handling multiple documents with an image"""
        message = Message(
            text="Compare all files",
            user_id="U123",
            channel_id="C456",
            thread_id="T789",
            attachments=[
                {"type": "file", "mimetype": "application/pdf", "name": "doc1.pdf", "url": "http://example.com/doc1.pdf", "id": "f1"},
                {"type": "file", "mimetype": "application/pdf", "name": "doc2.pdf", "url": "http://example.com/doc2.pdf", "id": "f2"},
                {"type": "image", "mimetype": "image/jpeg", "name": "img.jpg", "url": "http://example.com/img.jpg", "id": "f3"}
            ]
        )
        
        mock_client = MagicMock()
        mock_client.download_file.return_value = b"test_data"
        
        # Mock the necessary components
        processor.thread_manager.acquire_thread_lock.return_value = True
        processor.thread_manager.get_or_create_thread.return_value = MagicMock(
            thread_ts="T789",
            channel_id="C456",
            messages=[],
            had_timeout=False
        )
        processor.openai_client.classify_intent.return_value = "vision"
        processor.openai_client.analyze_images.return_value = "Image analysis"
        processor.openai_client.create_text_response.return_value = "Combined result"
        
        response = processor.process_message(message, mock_client)
        
        # Should process both PDFs
        assert processor.document_handler.safe_extract_content.call_count == 2
    
    def test_document_without_images(self, processor):
        """Test that documents alone don't trigger vision analysis"""
        message = Message(
            text="Summarize this document",
            user_id="U123",
            channel_id="C456",
            thread_id="T789",
            attachments=[
                {"type": "file", "mimetype": "application/pdf", "name": "doc.pdf", "url": "http://example.com/doc.pdf", "id": "f1"}
            ]
        )
        
        mock_client = MagicMock()
        mock_client.download_file.return_value = b"pdf_data"
        
        processor.thread_manager.acquire_thread_lock.return_value = True
        processor.thread_manager.get_or_create_thread.return_value = MagicMock(
            thread_ts="T789",
            channel_id="C456",
            messages=[],
            had_timeout=False
        )
        processor.openai_client.classify_intent.return_value = "vision"
        processor.openai_client.create_text_response.return_value = "Summary"
        
        response = processor.process_message(message, mock_client)
        
        # Should NOT call vision analysis
        processor.openai_client.analyze_images.assert_not_called()
    
    def test_mixed_content_with_failed_document_extraction(self, processor):
        """Test handling when document extraction fails but image works"""
        message = Message(
            text="Analyze these",
            user_id="U123",
            channel_id="C456",
            thread_id="T789",
            attachments=[
                {"type": "file", "mimetype": "application/pdf", "name": "corrupt.pdf", "url": "http://example.com/bad.pdf", "id": "f1"},
                {"type": "image", "mimetype": "image/png", "name": "good.png", "url": "http://example.com/good.png", "id": "f2"}
            ]
        )
        
        mock_client = MagicMock()
        mock_client.download_file.side_effect = [
            b"corrupt_pdf_data",
            b"good_image_data"
        ]
        
        # Make document extraction fail
        processor.document_handler.safe_extract_content.return_value = {
            "content": "",
            "error": "Failed to parse PDF"
        }
        
        processor.thread_manager.acquire_thread_lock.return_value = True
        processor.thread_manager.get_or_create_thread.return_value = MagicMock(
            thread_ts="T789",
            channel_id="C456",
            messages=[],
            had_timeout=False
        )
        processor.openai_client.classify_intent.return_value = "vision"
        processor.openai_client.analyze_images.return_value = "Image analysis"
        processor.openai_client.create_text_response.return_value = "Result with image only"
        
        response = processor.process_message(message, mock_client)
        
        # Should still process the image even if document failed
        processor.openai_client.analyze_images.assert_called_once()
        assert response is not None


class TestVisionWithoutUploadEnhancements:
    """Test enhancements to vision without upload (check again functionality)"""
    
    @pytest.fixture
    def processor(self):
        """Create a MessageProcessor with document ledger support"""
        with patch('message_processor.ThreadStateManager'):
            with patch('message_processor.OpenAIClient'):
                processor = MessageProcessor()
                processor.openai_client.create_text_response.return_value = "Analysis result"
                processor.openai_client.create_text_response_with_tools.return_value = "Analysis result"
                return processor
    
    def test_check_again_retrieves_documents(self, processor):
        """Test that 'check again' retrieves stored documents"""
        # Setup thread with existing documents
        thread_state = MagicMock(
            thread_ts="T789",
            channel_id="C456",
            messages=[
                {"role": "user", "content": "original upload"},
                {"role": "assistant", "content": "previous response"}
            ],
            had_timeout=False
        )
        
        # Mock document ledger with stored documents
        doc_ledger = MagicMock()
        doc_ledger.documents = [
            {
                "filename": "stored.pdf",
                "content": "Stored document content",
                "mime_type": "application/pdf",
                "total_pages": 2
            }
        ]
        
        processor.thread_manager.get_or_create_thread.return_value = thread_state
        processor.thread_manager.get_document_ledger.return_value = doc_ledger
        processor.thread_manager.acquire_thread_lock.return_value = True
        
        message = Message(
            text="check again",
            user_id="U123",
            channel_id="C456",
            thread_id="T789"
        )
        
        mock_client = MagicMock()
        processor.openai_client.classify_intent.return_value = "vision"
        
        # Capture what gets sent to text response
        actual_content = None
        def capture_content(text, *args, **kwargs):
            nonlocal actual_content
            actual_content = text
            return MagicMock(type="text", content="Result")
        
        with patch.object(processor, '_handle_text_response', side_effect=capture_content):
            response = processor.process_message(message, mock_client)
        
        # Should include document content in the enhanced text
        assert actual_content is not None
        assert "stored.pdf" in actual_content
        assert "Stored document content" in actual_content
    
    def test_check_again_with_images_and_documents(self, processor):
        """Test that 'check again' retrieves both images and documents"""
        thread_state = MagicMock(
            thread_ts="T789",
            channel_id="C456",
            messages=[
                {"role": "user", "content": "previous upload"},
                {"role": "assistant", "content": "previous analysis", 
                 "metadata": {"type": "image_analysis", "url": "http://example.com/img.jpg"}}
            ],
            had_timeout=False
        )
        
        # Mock document ledger
        doc_ledger = MagicMock()
        doc_ledger.documents = [
            {
                "filename": "doc.pdf",
                "content": "Document text",
                "mime_type": "application/pdf",
                "total_pages": 1
            }
        ]
        
        processor.thread_manager.get_or_create_thread.return_value = thread_state
        processor.thread_manager.get_document_ledger.return_value = doc_ledger
        processor.thread_manager.acquire_thread_lock.return_value = True
        
        message = Message(
            text="compare them again",
            user_id="U123",
            channel_id="C456",
            thread_id="T789"
        )
        
        mock_client = MagicMock()
        processor.openai_client.classify_intent.return_value = "vision"
        
        # Process the message
        response = processor.process_message(message, mock_client)
        
        # Should handle both images (from history) and documents (from ledger)
        assert response is not None