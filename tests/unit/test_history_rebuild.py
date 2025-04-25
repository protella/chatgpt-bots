import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from app.core.history import rebuild_thread_history


class TestHistoryRebuild:
    """Tests for rebuilding conversation history."""

    def test_last_user_message_exclusion(self):
        """Test that the last user message is excluded when rebuilding history."""
        # Mock the Slack client and response
        mock_client = MagicMock()
        mock_client.conversations_replies.return_value = {
            "messages": [
                # Thread starter (ts matches thread_ts)
                {"user": "U1", "text": "First user message", "ts": "1000"},
                
                # Bot response
                {"user": "BOT_ID", "text": "First bot response", "ts": "1001"},
                
                # Second user message
                {"user": "U1", "text": "Second user message", "ts": "1002"},
                
                # Second bot response
                {"user": "BOT_ID", "text": "Second bot response", "ts": "1003"},
                
                # Last user message - should be excluded
                {"user": "U1", "text": "Repeat the conversation", "ts": "1004"},
                
                # Current message being processed - automatically excluded
                {"user": "BOT_ID", "text": "Current response", "ts": "1005"}
            ]
        }
        
        # Call the function
        history = rebuild_thread_history(
            client=mock_client,
            channel_id="C123",
            thread_ts="1000",
            bot_user_id="BOT_ID"
        )
        
        # Verify the result
        # Should contain: system prompt + 2 user messages + 2 bot responses = 5 messages
        assert len(history) == 5
        
        # Check message roles and content
        assert history[0]["role"] == "system"  # System prompt
        
        assert history[1]["role"] == "user" 
        assert history[1]["content"][0]["text"] == "First user message"
        
        assert history[2]["role"] == "assistant"
        assert history[2]["content"][0]["text"] == "First bot response"
        
        assert history[3]["role"] == "user"
        assert history[3]["content"][0]["text"] == "Second user message"
        
        assert history[4]["role"] == "assistant"
        assert history[4]["content"][0]["text"] == "Second bot response"
        
        # Verify the "Repeat the conversation" message was excluded
        for message in history:
            if message["role"] == "user":
                assert "Repeat the conversation" not in message["content"][0]["text"]
        
        # Verify the current message was also excluded
        for message in history:
            if message["role"] == "assistant":
                assert "Current response" not in message["content"][0]["text"] 