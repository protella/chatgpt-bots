"""
Tests for message trimming, preservation, and document summarization functionality
"""
import pytest
from unittest.mock import MagicMock, patch
from message_processor.base import MessageProcessor
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
        ]

        for msg in test_cases:
            assert processor._should_preserve_message(msg) is True, f"Failed for type: {msg['metadata']['type']}"

        # document_upload is deliberately NOT preserved — documents are
        # summarized instead (summary rows + Slack CDN refs)
        doc_msg = {"role": "user", "content": "text", "metadata": {"type": "document_upload"}}
        assert processor._should_preserve_message(doc_msg) is False
    
    def test_should_preserve_summarized_documents(self, processor):
        """Test that summarized documents are preserved"""
        msg = {"role": "user", "content": "Some content", "metadata": {"summarized": True}}
        assert processor._should_preserve_message(msg) is True
    
    def test_should_preserve_urls(self, processor):
        """Test preservation of messages containing URLs"""
        # Only IMAGE URLs are preserved (needed for edits); document/plain
        # URLs are trimmable — documents live on as summaries + CDN refs
        preserved = [
            "Check this: https://example.com/image.png",
            "OpenAI image: https://oaidalleapiprodscus.blob.core.windows.net/img.png",
            "Discord CDN: https://cdn.discordapp.com/attachments/123/456/pic.jpg",
        ]
        for content in preserved:
            msg = {"role": "user", "content": content, "metadata": {}}
            assert processor._should_preserve_message(msg) is True, f"Failed for: {content}"

        not_preserved = [
            "Slack file: https://files.slack.com/file.pdf",
            "Multiple URLs: http://test.com and https://example.org/file.txt",
        ]
        for content in not_preserved:
            msg = {"role": "user", "content": content, "metadata": {}}
            assert processor._should_preserve_message(msg) is False, f"Failed for: {content}"
    
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
    
