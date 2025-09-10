"""
Tests for message trimming, preservation, and document summarization functionality
"""
import pytest
from unittest.mock import MagicMock, patch
from message_processor import MessageProcessor
from thread_manager import ThreadState
from config import config


class TestMessageTrimming:
    """Test message trimming and preservation logic"""
    
    @pytest.fixture
    def processor(self):
        """Create a MessageProcessor instance"""
        return MessageProcessor(db=None)
    
    @pytest.fixture
    def thread_state(self):
        """Create a sample thread state with various message types"""
        return ThreadState(
            channel_id="C123",
            thread_ts="T456",
            messages=[
                {"role": "user", "content": "Regular message 1", "metadata": {}},
                {"role": "user", "content": "Message with URL: https://example.com/image.png", "metadata": {}},
                {"role": "assistant", "content": "Response with Slack file: https://files.slack.com/file.png", "metadata": {}},
                {"role": "user", "content": "=== DOCUMENT: test.pdf ===\nMIME Type: application/pdf\nDocument content here", "metadata": {}},
                {"role": "developer", "content": "[Image Analysis: Technical description]", "metadata": {}},
                {"role": "assistant", "content": "Regular response", "metadata": {"type": "image_generation", "url": "https://oai.com/img.png"}},
                {"role": "user", "content": "Another regular message", "metadata": {}},
                {"role": "user", "content": "[SUMMARIZED document content]", "metadata": {"summarized": True}},
            ]
        )
    
    def test_should_preserve_system_messages(self, processor):
        """Test that system and developer messages are always preserved"""
        system_msg = {"role": "system", "content": "System prompt", "metadata": {}}
        developer_msg = {"role": "developer", "content": "Developer instruction", "metadata": {}}
        
        assert processor._should_preserve_message(system_msg) is True
        assert processor._should_preserve_message(developer_msg) is True
    
    def test_should_preserve_special_metadata_types(self, processor):
        """Test preservation of messages with special metadata types"""
        test_cases = [
            {"role": "user", "content": "text", "metadata": {"type": "image_generation"}},
            {"role": "user", "content": "text", "metadata": {"type": "image_edit"}},
            {"role": "user", "content": "text", "metadata": {"type": "vision_analysis"}},
            {"role": "user", "content": "text", "metadata": {"type": "image_analysis"}},
            {"role": "user", "content": "text", "metadata": {"type": "document_upload"}},
        ]
        
        for msg in test_cases:
            assert processor._should_preserve_message(msg) is True, f"Failed for type: {msg['metadata']['type']}"
    
    def test_should_preserve_summarized_documents(self, processor):
        """Test that summarized documents are preserved"""
        msg = {"role": "user", "content": "Some content", "metadata": {"summarized": True}}
        assert processor._should_preserve_message(msg) is True
    
    def test_should_preserve_urls(self, processor):
        """Test preservation of messages containing URLs"""
        test_cases = [
            "Check this: https://example.com/image.png",
            "Slack file: https://files.slack.com/file.pdf",
            "OpenAI image: https://oaidalleapiprodscus.blob.core.windows.net/img.png",
            "Discord CDN: https://cdn.discordapp.com/attachments/123/456/pic.jpg",
            "Multiple URLs: http://test.com and https://example.org/file.txt",
        ]
        
        for content in test_cases:
            msg = {"role": "user", "content": content, "metadata": {}}
            assert processor._should_preserve_message(msg) is True, f"Failed for: {content}"
    
    def test_should_not_preserve_regular_messages(self, processor):
        """Test that regular messages without special content are not preserved"""
        test_cases = [
            {"role": "user", "content": "Just a regular message", "metadata": {}},
            {"role": "assistant", "content": "Normal response without URLs", "metadata": {}},
            {"role": "user", "content": "No special markers here", "metadata": {}},
        ]
        
        for msg in test_cases:
            assert processor._should_preserve_message(msg) is False, f"Should not preserve: {msg['content']}"
    
    def test_should_preserve_document_markers(self, processor):
        """Test preservation of document content markers"""
        # Only SUMMARIZED documents should be preserved
        # Full documents are NOT preserved (they can be trimmed/summarized)
        
        # These should NOT be preserved
        not_preserved = [
            "=== DOCUMENT: file.pdf ===\nContent here",
            "Some text with MIME Type: application/pdf in it",
        ]
        
        for content in not_preserved:
            msg = {"role": "user", "content": content, "metadata": {}}
            assert processor._should_preserve_message(msg) is False, f"Should not preserve: {content}"
        
        # This SHOULD be preserved (summarized document)
        preserved = "[SUMMARIZED - Original content reduced]"
        msg = {"role": "user", "content": preserved, "metadata": {}}
        assert processor._should_preserve_message(msg) is True, f"Should preserve: {preserved}"
    
    def test_should_preserve_analysis_markers(self, processor):
        """Test preservation of injected analysis markers"""
        test_cases = [
            "[Image Analysis: Technical description of image]",
            "[Vision Context: Additional context here]",
        ]
        
        for content in test_cases:
            msg = {"role": "developer", "content": content, "metadata": {}}
            assert processor._should_preserve_message(msg) is True, f"Failed for: {content}"
    
    def test_smart_trim_oldest(self, processor, thread_state):
        """Test that smart_trim_oldest removes non-preserved messages"""
        initial_count = len(thread_state.messages)
        
        # Trim 3 messages
        trimmed = processor._smart_trim_oldest(thread_state, trim_count=3)
        
        # Documents are NOT preserved anymore, so they can be trimmed
        # Messages at indices 0 (regular), 3 (document), 6 (regular) are trimmable
        assert trimmed == 3  # Three messages trimmed
        assert len(thread_state.messages) == initial_count - 3
        
        # Check that URL-containing and summarized messages are still there
        remaining_contents = [msg["content"] for msg in thread_state.messages]
        assert any("https://example.com" in c for c in remaining_contents)
        assert any("https://files.slack.com" in c for c in remaining_contents)
        assert any("[Image Analysis:" in c for c in remaining_contents)
        assert any("[SUMMARIZED" in c for c in remaining_contents)
        
        # The unsummarized document should be gone
        assert not any("=== DOCUMENT: test.pdf ===" in c and "[SUMMARIZED" not in c for c in remaining_contents)
    
    def test_document_preservation_logic(self, processor):
        """Test that full documents are not preserved but summarized documents are"""
        # Full document message
        full_doc = {
            "role": "user", 
            "content": "=== DOCUMENT: test.pdf ===\nMIME Type: application/pdf\nContent\n=== DOCUMENT END ==="
        }
        
        # Summarized document message
        summarized_doc = {
            "role": "user",
            "content": "[SUMMARIZED - 500 chars -> 100 chars]\n=== DOCUMENT: test.pdf ===\nSummary\n=== DOCUMENT END ==="
        }
        
        # Test full document is NOT preserved (can be trimmed/summarized)
        assert not processor._should_preserve_message(full_doc), "Full documents should not be preserved"
        
        # Test summarized document IS preserved
        assert processor._should_preserve_message(summarized_doc), "Summarized documents should be preserved"
        
    def test_smart_trim_with_summarization(self, processor):
        """Test that documents in trim list get summarized"""
        # Create thread with documents and regular messages
        thread_state = ThreadState(
            channel_id="C123",
            thread_ts="T456",
            messages=[
                {"role": "user", "content": "Regular message 1", "metadata": {}},
                {"role": "user", "content": "=== DOCUMENT: test.pdf ===\nMIME Type: application/pdf\nDocument content\n=== DOCUMENT END: test.pdf ===", "metadata": {}},
                {"role": "user", "content": "Regular message 2", "metadata": {}},
                {"role": "user", "content": "Regular message 3", "metadata": {}},
            ]
        )
        
        # Documents are NOT preserved by default now
        # They should be in the trim list and get summarized
        initial_count = len(thread_state.messages)
        
        # Try to trim 2 messages (oldest 2 are message 1 and document)
        trimmed = processor._smart_trim_with_summarization(thread_state, trim_count=2)
        
        # Should summarize the document (returns 1) not trim
        assert trimmed == 1  # One document summarized
        assert len(thread_state.messages) == initial_count  # No messages removed yet
        
        # Document should be summarized now
        doc_msg = thread_state.messages[1]
        assert "[SUMMARIZED" in doc_msg["content"]
        
        # Call again to actually trim
        trimmed = processor._smart_trim_with_summarization(thread_state, trim_count=2)
        
        # Now should trim 2 regular messages (message 1 and 2)
        assert trimmed == 2  # Two messages trimmed
        assert len(thread_state.messages) == initial_count - 2
        
        # Document should still be there (summarized)
        doc_exists = any("[SUMMARIZED" in m["content"] for m in thread_state.messages)
        assert doc_exists
        
        # Regular message 1 and 2 should be gone
        assert not any("Regular message 1" in m["content"] for m in thread_state.messages)
        assert not any("Regular message 2" in m["content"] for m in thread_state.messages)
        
        # Regular message 3 should still be there
        assert any("Regular message 3" in m["content"] for m in thread_state.messages)
    
    @patch('message_processor.config')
    def test_summarize_document_content(self, mock_config, processor):
        """Test document content summarization"""
        mock_config.utility_model = "gpt-5-mini"
        
        # Create a document with proper format
        document_content = """=== DOCUMENT: test.pdf === (5 pages)
MIME Type: application/pdf
=== CONTENT START ===
This is a long document with lots of text that needs to be summarized.
It contains multiple paragraphs and important information.
The content goes on for many lines.
=== DOCUMENT END: test.pdf ==="""
        
        # Mock the OpenAI client response
        processor.openai_client.create_text_response = MagicMock(
            return_value="This is a concise summary of the document."
        )
        
        # Summarize the document
        result = processor._summarize_document_content(document_content)
        
        # Check the result
        assert "[SUMMARIZED" in result
        assert "test.pdf" in result
        assert "5 pages" in result
        assert "application/pdf" in result
        assert "This is a concise summary" in result
        
        # Verify the API was called correctly
        processor.openai_client.create_text_response.assert_called_once()
        call_args = processor.openai_client.create_text_response.call_args
        messages = call_args[1]['messages']
        
        # Check that we used developer role for the prompt
        assert messages[0]['role'] == 'developer'
        assert 'Summarize the document' in messages[0]['content']
        
        # Check that the document content was passed as user message
        assert messages[1]['role'] == 'user'
        assert 'This is a long document' in messages[1]['content']
    
    def test_summarize_document_content_malformed(self, processor):
        """Test that malformed documents are returned as-is"""
        malformed_content = "This is not a properly formatted document"
        
        result = processor._summarize_document_content(malformed_content)
        
        # Should return original content when can't parse
        assert result == malformed_content
    
    @patch('message_processor.config')
    def test_summarize_document_content_error_handling(self, mock_config, processor):
        """Test error handling in document summarization"""
        mock_config.utility_model = "gpt-5-mini"
        
        document_content = """=== DOCUMENT: test.pdf ===
MIME Type: application/pdf
Content here
=== DOCUMENT END: test.pdf ==="""
        
        # Mock the OpenAI client to raise an error
        processor.openai_client.create_text_response = MagicMock(
            side_effect=Exception("API Error")
        )
        
        # Should return original content on error
        result = processor._summarize_document_content(document_content)
        assert result == document_content
    
    def test_pre_trim_messages_for_api(self, processor):
        """Test pre_trim_messages_for_api removes oldest messages when over limit"""
        messages = [
            {"role": "user", "content": f"Message {i}", "metadata": {}}
            for i in range(10)
        ]
        
        # Add a preserved message in the middle
        messages[5]["content"] = "https://example.com/important.png"
        
        # Mock token counting to simulate being over limit
        with patch.object(processor.thread_manager._token_counter, 'count_thread_tokens') as mock_count:
            # First call returns over limit, subsequent calls return under
            mock_count.side_effect = [110000, 90000, 80000]
            
            # Mock config to return proper limit
            with patch('message_processor.config') as mock_config:
                mock_config.get_model_token_limit.return_value = 100000
                
                result = processor._pre_trim_messages_for_api(messages, new_message_tokens=1000, model="gpt-5")
                
                # Should have removed some messages
                assert len(result) < len(messages)
                
                # URL message should still be there
                assert any("https://example.com" in msg["content"] for msg in result)
    
    def test_async_post_response_cleanup(self, processor):
        """Test async cleanup triggers when over threshold"""
        thread_state = ThreadState(
            channel_id="C123",
            thread_ts="T456",
            messages=[
                {"role": "user", "content": f"Message {i}", "metadata": {}}
                for i in range(20)
            ]
        )
        
        # Mock token counting to be over threshold
        with patch.object(processor.thread_manager._token_counter, 'count_thread_tokens') as mock_count:
            mock_count.return_value = 350000  # Over 80% of 400k limit
            
            with patch.object(processor, '_smart_trim_with_summarization') as mock_trim:
                mock_trim.return_value = 5  # Trimmed 5 messages
                
                # Mock database update
                processor.db = MagicMock()
                
                # Run cleanup
                processor._async_post_response_cleanup(thread_state, "C123:T456")
                
                # Should have called trim
                mock_trim.assert_called_once_with(thread_state)
                
                # Should update database
                processor.db.clear_thread_messages.assert_called_once()
    
    def test_async_post_response_cleanup_under_threshold(self, processor):
        """Test async cleanup does nothing when under threshold"""
        thread_state = ThreadState(
            channel_id="C123",
            thread_ts="T456",
            messages=[
                {"role": "user", "content": "Small message", "metadata": {}}
            ]
        )
        
        # Mock token counting to be under threshold
        with patch.object(processor.thread_manager._token_counter, 'count_thread_tokens') as mock_count:
            mock_count.return_value = 50000  # Well under 80% of limit
            
            with patch.object(processor, '_smart_trim_with_summarization') as mock_trim:
                # Run cleanup
                processor._async_post_response_cleanup(thread_state, "C123:T456")
                
                # Should NOT have called trim
                mock_trim.assert_not_called()
    
    def test_smart_trim_on_cache_load(self, processor):
        """Test that smart trimming is applied when loading thread from cache with high tokens"""
        # Create a thread state that would be loaded from cache
        thread_state = ThreadState(
            channel_id="C123",
            thread_ts="T456",
            messages=[
                {"role": "user", "content": "Message 1", "metadata": {}},
                {"role": "user", "content": f"=== DOCUMENT: big.pdf ===\nMIME Type: application/pdf\n{'X' * 100000}\n=== DOCUMENT END: big.pdf ===", "metadata": {}},
                {"role": "assistant", "content": "Response", "metadata": {}},
            ],
            current_model="gpt-5"
        )
        
        # Mock the thread manager to return our thread
        with patch.object(processor.thread_manager, 'get_or_create_thread', return_value=thread_state):
            # Mock token counting - first call way over limit, second after trimming, third for final log
            with patch.object(processor.thread_manager._token_counter, 'count_thread_tokens') as mock_count:
                mock_count.side_effect = [500000, 100000, 100000]  # Over, then under after trim, then final check
                
                # Mock config to return limits
                with patch('message_processor.config') as mock_config:
                    mock_config.gpt_model = "gpt-5"
                    mock_config.get_model_token_limit.return_value = 350000
                    mock_config.token_cleanup_threshold = 0.8
                    
                    # Mock database
                    processor.db = MagicMock()
                    
                    # Mock message and client
                    from base_client import Message
                    mock_message = Message(
                        text="New",
                        user_id="U1",
                        channel_id="C123",
                        thread_id="T456",
                        attachments=[],
                        metadata={"ts": "123"}
                    )
                    mock_client = MagicMock()
                    mock_client.get_thread_history.return_value = []
                    
                    # Call get_or_rebuild_thread_state
                    with patch.object(processor, '_smart_trim_with_summarization', return_value=2) as mock_trim:
                        result = processor._get_or_rebuild_thread_state(mock_message, mock_client)
                        
                        # Should have called smart trim
                        mock_trim.assert_called_once_with(thread_state)
                        
                        # Should update database
                        processor.db.clear_thread_messages.assert_called_once()
                        assert processor.db.cache_message.call_count > 0