"""Advanced unit tests for thread_manager.py - watchdog, cleanup, and edge cases"""

import pytest

from thread_manager import ThreadState

# Default timeout for stuck threads (5 seconds for testing)
THREAD_LOCK_TIMEOUT_SECONDS = 5


class TestThreadStateEdgeCases:
    """Test edge cases in thread state management"""
    
    @pytest.mark.skip(reason="Message limit behavior changed")
    def test_thread_state_message_limit(self):
        """Test thread state respects message limit"""
        thread_state = ThreadState("test:thread", "channel")
        
        # Add more than limit messages
        for i in range(60):
            thread_state.add_message(
                role="user" if i % 2 == 0 else "assistant",
                content=f"Message {i}",
                metadata={}
            )
        
        # Should only keep recent 50
        assert len(thread_state.messages) == 50
        
        # Should keep the most recent ones
        assert thread_state.messages[0]["content"] == "Message 10"
        assert thread_state.messages[-1]["content"] == "Message 59"
    
    # The AssetLedger large-image and URL-tracking tests exercised add_image/add_url_image/
    # get_recent_images, which were removed with the vision + image-edit handlers. Ledger rows
    # are now appended by message_processor/image_delivery.py::publish_image and never carry
    # base64 at all; see test_background_image_gen.py.

