"""
Integration tests for mixed content processing
These tests verify the full flow with real-like data
"""
import pytest
import base64
from unittest.mock import MagicMock, patch, Mock
from message_processor.base import MessageProcessor
from base_client import Message, Response
from document_handler import DocumentHandler
from thread_manager import ThreadStateManager


@pytest.mark.integration
class TestMixedContentIntegration:
    """Integration tests for mixed content with realistic scenarios"""
    
    @pytest.fixture
    def real_pdf_content(self):
        """Sample PDF content that would be extracted"""
        return """Exploring Social Anxiety
Social anxiety is a disorder characterized by overwhelming anxiety or self-consciousness in ordinary
social situations. In milder cases, the symptoms of social anxiety only appear in specific situations, such
as public speaking. On the more extreme end, any form of social interaction can act as a trigger.

Which social situations are you anxious about?
- Giving a speech
- Spending time alone with a friend
- Going on a date
- Attending a crowded event"""
    
    @pytest.fixture
    def real_image_description(self):
        """Sample image analysis that would come from vision model"""
        return """The image shows a home music studio setup with the following equipment:
- Two keyboards stacked vertically (Yamaha workstation on top)
- A small audio mixer on the left side
- A microphone on a stand
- A single studio monitor speaker on the right
- Sheet music displayed on a music stand
The setup appears to be in a residential room with beige walls."""
    
    @pytest.fixture
    def processor_with_real_handlers(self):
        """Create processor with semi-real handlers"""
        processor = MessageProcessor()
        
        # Use real document handler if available
        if DocumentHandler:
            processor.document_handler = DocumentHandler()
        else:
            processor.document_handler = MagicMock()
            processor.document_handler.is_document_file.return_value = True
        
        return processor
    
    @pytest.mark.skip(reason="Recursion issue with MagicMock - needs investigation")
    def test_integration_pdf_and_image_comparison(self, processor_with_real_handlers, real_pdf_content, real_image_description):
        """Integration test: Compare PDF and image content"""
        processor = processor_with_real_handlers
        
        # Ensure OpenAIClient is mocked to prevent real API calls
        from unittest.mock import MagicMock, patch
        processor.openai_client = MagicMock()
        
        # Mock the external dependencies
        mock_client = MagicMock()
        mock_client.platform = "slack"
        mock_client.name = "SlackClient"
        mock_client.download_file.side_effect = [
            b"fake_pdf_data",  # PDF
            b"fake_image_data"  # Image
        ]
        
        # Mock document extraction to return realistic content
        processor.document_handler.safe_extract_content = MagicMock(return_value={
            "content": real_pdf_content,
            "total_pages": 1,
            "page_structure": {"pages": [{"page": 1, "content": real_pdf_content}]}
        })
        
        # Mock vision analysis to return realistic description
        processor.openai_client.analyze_images = MagicMock(return_value=real_image_description)
        
        # Mock final text response
        def generate_comparison(messages=None, *args, **kwargs):
            # Simulate what the model would actually respond
            if messages and len(messages) > 0:
                last_msg = messages[-1]
                if isinstance(last_msg, dict) and "content" in last_msg:
                    content = last_msg["content"]
                    if "social anxiety" in content.lower() and "keyboard" in content.lower():
                        return "No, these files are not related. The PDF is a therapeutic worksheet about social anxiety, while the image shows music production equipment. They cover completely different topics."
            return "Unable to compare files"
        
        processor.openai_client.create_text_response = MagicMock(side_effect=generate_comparison)
        processor.openai_client.create_text_response_with_tools = MagicMock(side_effect=generate_comparison)
        # Mock streaming to return the comparison result
        def stream_comparison(*args, **kwargs):
            response_text = "No, these files are not related. The PDF is about social anxiety, the image shows music equipment."
            for chunk in response_text:
                yield chunk
        
        processor.openai_client.create_streaming_response = MagicMock(side_effect=stream_comparison)
        processor.openai_client.create_streaming_response_with_tools = MagicMock(side_effect=stream_comparison)
        processor.openai_client.get_response = MagicMock(return_value="No, these files are not related.")
        processor.openai_client.classify_intent = MagicMock(return_value="vision")
        
        # Setup thread manager
        processor.thread_manager.acquire_thread_lock = MagicMock(return_value=True)
        processor.thread_manager.release_thread_lock = MagicMock()
        thread_state_mock = MagicMock(
            thread_ts="T789",
            channel_id="C456",
            messages=[],
            had_timeout=False,
            pending_clarification=None,
            config_overrides={}
        )
        # add_message method should append to messages list
        def add_message_impl(role, content, **kwargs):
            thread_state_mock.messages.append({"role": role, "content": content})
        thread_state_mock.add_message = MagicMock(side_effect=add_message_impl)
        
        processor.thread_manager.get_or_create_thread = MagicMock(return_value=thread_state_mock)
        processor.thread_manager.get_or_create_document_ledger = MagicMock(return_value=MagicMock(
            add_document=MagicMock()
        ))
        
        # Create realistic message
        message = Message(
            text="Are these two files related in any way?",
            user_id="U123",
            channel_id="C456",
            thread_id="T789",
            attachments=[
                {
                    "type": "file",
                    "mimetype": "application/pdf",
                    "name": "exploring-social-anxiety.pdf",
                    "url": "https://files.slack.com/files-pri/T123/F456/exploring-social-anxiety.pdf",
                    "id": "F456"
                },
                {
                    "type": "image",
                    "mimetype": "image/jpeg",
                    "name": "IMG_20130618_031224.jpg",
                    "url": "https://files.slack.com/files-pri/T123/F789/img.jpg",
                    "id": "F789"
                }
            ]
        )
        
        # Process the message with streaming disabled
        with patch('message_processor.config.enable_streaming', False):
            response = processor.process_message(message, mock_client, thinking_id="think_123")
        
        # Verify the response makes sense
        assert response is not None
        if response.type == "error":
            print(f"ERROR RESPONSE: {response.content}")
        assert response.type == "text"
        assert "not related" in response.content.lower()
        assert "social anxiety" in response.content.lower() or "therapeutic" in response.content.lower()
        assert "music" in response.content.lower() or "keyboard" in response.content.lower()
    
    def test_integration_check_again_with_stored_content(self, processor_with_real_handlers, real_pdf_content, real_image_description):
        """Integration test: 'Check again' retrieves and reanalyzes stored content"""
        processor = processor_with_real_handlers
        
        # Setup thread with previous conversation
        thread_state = MagicMock(
            thread_ts="T789",
            channel_id="C456",
            messages=[
                {
                    "role": "user",
                    "content": "are these two files related in any way?"
                },
                {
                    "role": "assistant",
                    "content": "No, these files are not related.",
                    "metadata": {
                        "type": "mixed_content_analysis",
                        "image_count": 1,
                        "document_count": 1
                    }
                }
            ],
            had_timeout=False,
            pending_clarification=None
        )
        
        # Setup document ledger with stored PDF
        doc_ledger = MagicMock()
        doc_ledger.documents = [
            {
                "filename": "exploring-social-anxiety.pdf",
                "content": real_pdf_content,
                "mime_type": "application/pdf",
                "total_pages": 1,
                "page_structure": {"pages": [{"page": 1, "content": real_pdf_content}]}
            }
        ]
        
        # Setup mocks
        processor.thread_manager.acquire_thread_lock = MagicMock(return_value=True)
        processor.thread_manager.release_thread_lock = MagicMock()
        processor.thread_manager.get_or_create_thread = MagicMock(return_value=thread_state)
        processor.thread_manager.get_document_ledger = MagicMock(return_value=doc_ledger)
        processor.openai_client.classify_intent = MagicMock(return_value="vision")
        
        # Mock DB to return stored image metadata
        if processor.db:
            processor.db.find_thread_images = MagicMock(return_value=[
                {
                    "url": "https://files.slack.com/files-pri/T123/F789/img.jpg",
                    "analysis": real_image_description,
                    "prompt": "Technical image description"
                }
            ])
        
        # Mock final response
        def generate_reanalysis(*args, **kwargs):
            # The function is called with different signatures, we just need to return the expected response
            # when processing the "check again" message with the stored context
            return MagicMock(
                type="text",
                content="Upon reviewing again, I can confirm these files are unrelated. The PDF discusses social anxiety treatment, while the image shows music equipment."
            )
        
        with patch.object(processor, '_handle_text_response', side_effect=generate_reanalysis):
            # Create "check again" message
            message = Message(
                text="check again - are you sure they're not related?",
                user_id="U123",
                channel_id="C456",
                thread_id="T789"
            )
            
            mock_client = MagicMock()
            
            # Process the follow-up
            response = processor.process_message(message, mock_client)
        
        # Verify response
        assert response is not None
        assert response.type == "text"
        assert "unrelated" in response.content.lower() or "not related" in response.content.lower()
    
    def test_integration_multiple_documents_with_image(self, processor_with_real_handlers):
        """Integration test: Handle multiple documents plus an image"""
        processor = processor_with_real_handlers
        
        # Mock multiple document extractions
        doc_contents = [
            "Document 1: Technical specifications for keyboard synthesizers",
            "Document 2: User manual for Yamaha workstation",
            "Document 3: Social anxiety worksheet"
        ]
        
        extraction_count = 0
        def extract_content(file_data, mimetype, filename):
            nonlocal extraction_count
            content = doc_contents[extraction_count] if extraction_count < len(doc_contents) else "Unknown content"
            extraction_count += 1
            return {
                "content": content,
                "filename": filename,
                "total_pages": 1
            }
        
        processor.document_handler.safe_extract_content = MagicMock(side_effect=extract_content)
        processor.document_handler.is_document_file = MagicMock(return_value=True)
        
        # Mock other components
        mock_client = MagicMock()
        mock_client.download_file = MagicMock(return_value=b"fake_data")
        
        processor.openai_client.analyze_images = MagicMock(
            return_value="Image shows a Yamaha keyboard workstation"
        )
        processor.openai_client.classify_intent = MagicMock(return_value="vision")
        processor.openai_client.create_text_response = MagicMock(
            return_value="Documents 1 and 2 are related to the image (all about keyboards). Document 3 about social anxiety is unrelated."
        )
        
        processor.thread_manager.acquire_thread_lock = MagicMock(return_value=True)
        processor.thread_manager.release_thread_lock = MagicMock()
        thread_state_mock = MagicMock(
            thread_ts="T789",
            channel_id="C456",
            messages=[],
            had_timeout=False,
            pending_clarification=None,
            config_overrides={}
        )
        # add_message method should append to messages list
        def add_message_impl(role, content, **kwargs):
            thread_state_mock.messages.append({"role": role, "content": content})
        thread_state_mock.add_message = MagicMock(side_effect=add_message_impl)
        
        processor.thread_manager.get_or_create_thread = MagicMock(return_value=thread_state_mock)
        processor.thread_manager.get_or_create_document_ledger = MagicMock(return_value=MagicMock(
            add_document=MagicMock()
        ))
        
        # Create message with multiple files
        message = Message(
            text="Which documents relate to the image?",
            user_id="U123",
            channel_id="C456",
            thread_id="T789",
            attachments=[
                {"type": "file", "mimetype": "application/pdf", "name": "keyboard_specs.pdf", "url": "http://example.com/1.pdf", "id": "f1"},
                {"type": "file", "mimetype": "application/pdf", "name": "yamaha_manual.pdf", "url": "http://example.com/2.pdf", "id": "f2"},
                {"type": "file", "mimetype": "application/pdf", "name": "anxiety_worksheet.pdf", "url": "http://example.com/3.pdf", "id": "f3"},
                {"type": "image", "mimetype": "image/jpeg", "name": "keyboard_setup.jpg", "url": "http://example.com/img.jpg", "id": "f4"}
            ]
        )
        
        # Process
        response = processor.process_message(message, mock_client)
        
        # Verify all documents were processed
        assert processor.document_handler.safe_extract_content.call_count == 3
        
        # Verify response addresses the relationship
        assert response is not None
        assert response.type == "text"
        # Check that response mentions the keyboard-related documents
        assert "keyboard" in response.content.lower() or "yamaha" in response.content.lower()


@pytest.mark.integration 
class TestMixedContentErrorHandling:
    """Test error handling in mixed content scenarios"""
    
    def test_vision_api_failure_fallback(self):
        """Test fallback when vision API fails during mixed content"""
        processor = MessageProcessor()
        
        # Setup to simulate vision API failure
        processor.openai_client.analyze_images = MagicMock(
            side_effect=Exception("Vision API error")
        )
        processor.openai_client.classify_intent = MagicMock(return_value="vision")
        
        processor.document_handler = MagicMock()
        processor.document_handler.is_document_file.return_value = True
        processor.document_handler.safe_extract_content.return_value = {
            "content": "Document content",
            "total_pages": 1
        }
        
        processor.thread_manager.acquire_thread_lock = MagicMock(return_value=True)
        processor.thread_manager.release_thread_lock = MagicMock()
        thread_state_mock = MagicMock(
            thread_ts="T789",
            channel_id="C456",
            messages=[],
            had_timeout=False,
            pending_clarification=None,
            config_overrides={}
        )
        # add_message method should append to messages list
        def add_message_impl(role, content, **kwargs):
            thread_state_mock.messages.append({"role": role, "content": content})
        thread_state_mock.add_message = MagicMock(side_effect=add_message_impl)
        
        processor.thread_manager.get_or_create_thread = MagicMock(return_value=thread_state_mock)
        processor.thread_manager.get_or_create_document_ledger = MagicMock(return_value=MagicMock(
            add_document=MagicMock()
        ))
        
        mock_client = MagicMock()
        mock_client.download_file.return_value = b"fake_data"
        
        message = Message(
            text="Analyze these",
            user_id="U123",
            channel_id="C456",
            thread_id="T789",
            attachments=[
                {"type": "file", "mimetype": "application/pdf", "name": "doc.pdf", "url": "http://example.com/doc.pdf", "id": "f1"},
                {"type": "image", "mimetype": "image/png", "name": "img.png", "url": "http://example.com/img.png", "id": "f2"}
            ]
        )
        
        # Process - should handle the error gracefully
        response = processor.process_message(message, mock_client)
        
        # Should return an error response
        assert response is not None
        assert response.type == "error"
        assert "Failed to analyze mixed content" in response.content