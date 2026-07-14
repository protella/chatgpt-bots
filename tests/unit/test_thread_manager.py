"""
Unit tests for thread_manager.py module
"""
from unittest.mock import MagicMock
from thread_manager import ThreadState, AssetLedger


class TestThreadState:
    """Test ThreadState class"""
    
    def test_initialization(self):
        """Test ThreadState initialization"""
        thread = ThreadState(thread_ts="123.456", channel_id="C123")
        
        assert thread.thread_ts == "123.456"
        assert thread.channel_id == "C123"
        assert thread.messages == []
        assert thread.config_overrides == {}
        assert thread.system_prompt is None
        assert thread.is_processing is False
        assert thread.had_timeout is False
    
    def test_add_message_without_db(self):
        """Test adding message without database"""
        thread = ThreadState(thread_ts="123.456", channel_id="C123")
        thread.add_message("user", "Hello bot")
        
        assert len(thread.messages) == 1
        assert thread.messages[0]["role"] == "user"
        assert thread.messages[0]["content"] == "Hello bot"
    
    def test_add_message_with_metadata(self):
        """Test adding message with metadata"""
        thread = ThreadState(thread_ts="123.456", channel_id="C123")
        metadata = {"source": "test", "timestamp": 12345}
        thread.add_message("assistant", "Hello user", metadata=metadata)
        
        assert len(thread.messages) == 1
        assert thread.messages[0]["metadata"] == metadata
    
    def test_add_message_with_db(self):
        """Phase S: messages are NOT persisted; the Slack ts is stamped into metadata"""
        mock_db = MagicMock()
        thread = ThreadState(thread_ts="123.456", channel_id="C123")
        thread.add_message("user", "Hello", db=mock_db, thread_key="C123:123.456", message_ts="789.012")

        assert thread.messages[0]["metadata"]["ts"] == "789.012"
        assert not mock_db.method_calls  # no DB writes from add_message
    
    def test_get_recent_messages(self):
        """Test getting recent messages"""
        thread = ThreadState(thread_ts="123.456", channel_id="C123")
        
        # Add multiple messages
        for i in range(10):
            thread.add_message("user", f"Message {i}")
        
        recent = thread.get_recent_messages(count=3)
        assert len(recent) == 3
        assert recent[0]["content"] == "Message 7"
        assert recent[2]["content"] == "Message 9"
    
    def test_get_recent_messages_empty(self):
        """Test getting recent messages from empty thread"""
        thread = ThreadState(thread_ts="123.456", channel_id="C123")
        recent = thread.get_recent_messages()
        assert recent == []
    
    def test_clear_old_messages_noop(self):
        """Test that clear_old_messages is now a no-op with DB"""
        thread = ThreadState(thread_ts="123.456", channel_id="C123")
        for i in range(30):
            thread.add_message("user", f"Message {i}")
        
        thread.clear_old_messages(_keep_last=5)
        # Should not limit messages anymore
        assert len(thread.messages) == 30


class TestAssetLedger:
    """Test AssetLedger class"""
    
    def test_initialization(self):
        """Test AssetLedger initialization"""
        ledger = AssetLedger(thread_ts="123.456")
        assert ledger.thread_ts == "123.456"
        assert ledger.images == []
    
    # The add_image/add_url_image/get_recent_images writers were removed with the vision and
    # image-edit handlers. Rows are now appended by the delivery seam
    # (message_processor/image_delivery.py::publish_image), which is covered — including the
    # "no base64 in the row" invariant — in test_background_image_gen.py.


