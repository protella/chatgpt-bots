"""
Unit tests for thread_manager.py module
"""
import time
import pytest
from unittest.mock import MagicMock, patch, call
from thread_manager import ThreadState, AssetLedger, ThreadLockManager, ThreadStateManager


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
        """Test adding message with database"""
        mock_db = MagicMock()
        thread = ThreadState(thread_ts="123.456", channel_id="C123")
        thread.add_message("user", "Hello", db=mock_db, thread_key="C123:123.456", message_ts="789.012")
        
        mock_db.cache_message.assert_called_once_with("C123:123.456", "user", "Hello", "789.012", None)
    
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
        
        thread.clear_old_messages(keep_last=5)
        # Should not limit messages anymore
        assert len(thread.messages) == 30


class TestAssetLedger:
    """Test AssetLedger class"""
    
    def test_initialization(self):
        """Test AssetLedger initialization"""
        ledger = AssetLedger(thread_ts="123.456")
        assert ledger.thread_ts == "123.456"
        assert ledger.images == []
    
    def test_add_image_without_db(self):
        """Test adding image without database"""
        ledger = AssetLedger(thread_ts="123.456")
        ledger.add_image(
            image_data="base64data",
            prompt="A beautiful sunset",
            timestamp=time.time(),
            slack_url="https://slack.com/image.png"
        )
        
        assert len(ledger.images) == 1
        assert ledger.images[0]["data"] == "base64data"
        assert ledger.images[0]["prompt"] == "A beautiful sunset"[:100]
        assert ledger.images[0]["slack_url"] == "https://slack.com/image.png"
        assert ledger.images[0]["source"] == "generated"
    
    def test_add_image_with_db(self):
        """Test adding image with database"""
        mock_db = MagicMock()
        ledger = AssetLedger(thread_ts="123.456")
        
        ledger.add_image(
            image_data="base64data",
            prompt="Test prompt",
            timestamp=time.time(),
            slack_url="https://slack.com/image.png",
            db=mock_db,
            thread_id="C123:123.456",
            analysis="Image contains a cat"
        )
        
        # With DB, base64 should not be stored in memory
        assert ledger.images[0]["data"] is None
        assert ledger.images[0]["prompt"] == "Test prompt"  # Full prompt with DB
        
        mock_db.save_image_metadata.assert_called_once()
    
    def test_add_url_image(self):
        """Test adding URL image"""
        ledger = AssetLedger(thread_ts="123.456")
        ledger.add_url_image(
            image_data="base64data",
            url="https://example.com/image.jpg",
            timestamp=time.time()
        )
        
        assert len(ledger.images) == 1
        assert ledger.images[0]["source"] == "url"
        assert ledger.images[0]["original_url"] == "https://example.com/image.jpg"
    
    def test_get_recent_images(self):
        """Test getting recent images"""
        ledger = AssetLedger(thread_ts="123.456")
        
        for i in range(10):
            ledger.add_image(f"data{i}", f"prompt{i}", time.time())
        
        recent = ledger.get_recent_images(count=3)
        assert len(recent) == 3
        assert recent[0]["prompt"] == "prompt7"[:100]


class TestThreadLockManager:
    """Test ThreadLockManager class"""
    
    def test_get_lock(self):
        """Test getting or creating a lock"""
        manager = ThreadLockManager()
        lock1 = manager.get_lock("thread1")
        lock2 = manager.get_lock("thread1")
        
        assert lock1 is lock2  # Same lock object
    
    def test_record_and_clear_acquisition(self):
        """Test recording and clearing lock acquisition"""
        manager = ThreadLockManager()
        thread_key = "thread1"
        
        manager.record_acquisition(thread_key)
        assert thread_key in manager._lock_acquisition_times
        
        manager.clear_acquisition(thread_key)
        assert thread_key not in manager._lock_acquisition_times
    
    def test_get_stuck_threads(self):
        """Test identifying stuck threads"""
        manager = ThreadLockManager()
        
        # Record acquisition for thread1 with old timestamp
        manager._lock_acquisition_times["thread1"] = time.time() - 400
        # Record acquisition for thread2 with recent timestamp
        manager._lock_acquisition_times["thread2"] = time.time() - 100
        
        stuck = manager.get_stuck_threads(max_duration=300)
        assert "thread1" in stuck
        assert "thread2" not in stuck
    
    def test_is_busy(self):
        """Test checking if thread is busy"""
        manager = ThreadLockManager()
        thread_key = "thread1"
        
        # Should not be busy initially
        assert manager.is_busy(thread_key) is False
        
        # Acquire the lock
        lock = manager.get_lock(thread_key)
        lock.acquire()
        
        # Now should be busy
        assert manager.is_busy(thread_key) is True
        
        lock.release()
        # Should not be busy after release
        assert manager.is_busy(thread_key) is False


class TestThreadStateManager:
    """Test ThreadStateManager class"""
    
    @patch('thread_manager.threading.Thread')
    def test_initialization_without_db(self, mock_thread):
        """Test ThreadStateManager initialization without database"""
        manager = ThreadStateManager()
        
        assert manager._threads == {}
        assert manager._assets == {}
        assert manager.db is None
        mock_thread.assert_called_once()  # Watchdog thread created
    
    @patch('thread_manager.threading.Thread')
    def test_initialization_with_db(self, mock_thread):
        """Test ThreadStateManager initialization with database"""
        mock_db = MagicMock()
        manager = ThreadStateManager(db=mock_db)
        
        assert manager.db is mock_db
        mock_thread.assert_called_once()
    
    @patch('thread_manager.threading.Thread')
    def test_get_or_create_thread_new(self, mock_thread):
        """Test creating a new thread"""
        manager = ThreadStateManager()
        thread = manager.get_or_create_thread("123.456", "C123", "U123")
        
        assert thread.thread_ts == "123.456"
        assert thread.channel_id == "C123"
        assert "C123:123.456" in manager._threads
    
    @patch('thread_manager.threading.Thread')
    def test_get_or_create_thread_existing(self, mock_thread):
        """Test getting existing thread"""
        manager = ThreadStateManager()
        thread1 = manager.get_or_create_thread("123.456", "C123")
        thread2 = manager.get_or_create_thread("123.456", "C123")
        
        assert thread1 is thread2
    
    @patch('thread_manager.threading.Thread')
    def test_get_or_create_thread_with_db(self, mock_thread):
        """Test creating thread with database"""
        mock_db = MagicMock()
        mock_db.get_or_create_thread.return_value = {"id": 1}
        mock_db.get_thread_config.return_value = {"model": "gpt-5"}
        mock_db.get_cached_messages.return_value = [
            {"role": "user", "content": "Hello"}
        ]
        
        manager = ThreadStateManager(db=mock_db)
        thread = manager.get_or_create_thread("123.456", "C123", "U123")
        
        assert thread.config_overrides == {"model": "gpt-5"}
        assert len(thread.messages) == 1
        mock_db.get_or_create_thread.assert_called_once_with("C123:123.456", "C123", "U123")
    
    @patch('thread_manager.threading.Thread')
    def test_acquire_and_release_thread_lock(self, mock_thread):
        """Test acquiring and releasing thread lock"""
        manager = ThreadStateManager()
        
        # Acquire lock
        acquired = manager.acquire_thread_lock("123.456", "C123", timeout=0)
        assert acquired is True
        
        thread = manager.get_thread("123.456", "C123")
        assert thread.is_processing is True
        
        # Try to acquire again (should fail)
        acquired2 = manager.acquire_thread_lock("123.456", "C123", timeout=0)
        assert acquired2 is False
        
        # Release lock
        manager.release_thread_lock("123.456", "C123")
        assert thread.is_processing is False
    
    @patch('thread_manager.threading.Thread')
    def test_is_thread_busy(self, mock_thread):
        """Test checking if thread is busy"""
        manager = ThreadStateManager()
        
        assert manager.is_thread_busy("123.456", "C123") is False
        
        manager.acquire_thread_lock("123.456", "C123")
        assert manager.is_thread_busy("123.456", "C123") is True
        
        manager.release_thread_lock("123.456", "C123")
        assert manager.is_thread_busy("123.456", "C123") is False
    
    @patch('thread_manager.threading.Thread')
    def test_update_thread_config(self, mock_thread):
        """Test updating thread configuration"""
        manager = ThreadStateManager()
        config = {"model": "gpt-5-nano", "temperature": 0.5}
        
        manager.update_thread_config("123.456", "C123", config)
        
        thread = manager.get_thread("123.456", "C123")
        assert thread.config_overrides == config
    
    @patch('thread_manager.threading.Thread')
    def test_cleanup_old_threads(self, mock_thread):
        """Test cleaning up old threads"""
        manager = ThreadStateManager()
        
        # Create threads with different ages
        thread1 = manager.get_or_create_thread("123.456", "C123")
        thread1.last_activity = time.time() - 100000  # Old
        
        thread2 = manager.get_or_create_thread("789.012", "C123")
        thread2.last_activity = time.time() - 10  # Recent
        
        manager.cleanup_old_threads(max_age=86400)
        
        assert "C123:123.456" not in manager._threads
        assert "C123:789.012" in manager._threads
    
    @patch('thread_manager.threading.Thread')
    def test_get_stats(self, mock_thread):
        """Test getting statistics"""
        manager = ThreadStateManager()
        
        manager.get_or_create_thread("123.456", "C123")
        manager.get_or_create_thread("789.012", "C456")
        manager.acquire_thread_lock("123.456", "C123")
        
        stats = manager.get_stats()
        assert stats["active_threads"] == 2
        assert stats["processing_threads"] == 1
    
    @pytest.mark.critical
    @patch('thread_manager.threading.Thread')
    def test_critical_thread_isolation(self, mock_thread):
        """Critical path test: Ensure thread isolation works correctly"""
        manager = ThreadStateManager()
        
        # Create two separate threads
        thread1 = manager.get_or_create_thread("123.456", "C123")
        thread2 = manager.get_or_create_thread("789.012", "C456")
        
        # Add messages to each
        thread1.add_message("user", "Thread 1 message")
        thread2.add_message("user", "Thread 2 message")
        
        # Verify isolation
        assert len(thread1.messages) == 1
        assert len(thread2.messages) == 1
        assert thread1.messages[0]["content"] == "Thread 1 message"
        assert thread2.messages[0]["content"] == "Thread 2 message"
        
        # Verify thread keys are different
        assert manager.get_thread("123.456", "C123") is thread1
        assert manager.get_thread("789.012", "C456") is thread2
        assert thread1 is not thread2
    
    @pytest.mark.smoke
    @patch('thread_manager.threading.Thread')  
    def test_smoke_basic_thread_operations(self, mock_thread):
        """Smoke test: Verify basic thread operations work"""
        try:
            manager = ThreadStateManager()
            
            # Can create thread
            thread = manager.get_or_create_thread("123.456", "C123")
            assert thread is not None
            
            # Can add message
            thread.add_message("user", "Test message")
            assert len(thread.messages) == 1
            
            # Can acquire/release lock
            acquired = manager.acquire_thread_lock("123.456", "C123", timeout=0)
            assert acquired is True
            manager.release_thread_lock("123.456", "C123")
            
        except Exception as e:
            pytest.fail(f"Basic thread operations failed: {e}")
    
    @patch('thread_manager.threading.Thread')
    def test_contract_thread_state_interface(self, mock_thread):
        """Contract test: Ensure ThreadState provides expected interface"""
        thread = ThreadState(thread_ts="123.456", channel_id="C123")
        
        # Verify required attributes
        assert hasattr(thread, 'thread_ts')
        assert hasattr(thread, 'channel_id')
        assert hasattr(thread, 'messages')
        assert hasattr(thread, 'config_overrides')
        assert hasattr(thread, 'system_prompt')
        assert hasattr(thread, 'is_processing')
        assert hasattr(thread, 'had_timeout')
        
        # Verify required methods
        assert callable(thread.add_message)
        assert callable(thread.get_recent_messages)
        assert callable(thread.clear_old_messages)
        
        # Test method signatures
        thread.add_message("user", "test")  # Should not raise
        recent = thread.get_recent_messages(count=5)  # Should accept count param
        assert isinstance(recent, list)
    
    @patch('thread_manager.threading.Thread')
    def test_state_persistence_with_db(self, mock_thread):
        """State test: Verify thread state persists with database"""
        mock_db = MagicMock()
        mock_db.get_or_create_thread.return_value = {"id": 1}
        mock_db.get_thread_config.return_value = {"model": "gpt-5"}
        mock_db.get_cached_messages.return_value = []
        
        manager = ThreadStateManager(db=mock_db)
        
        # Create thread and add messages
        thread = manager.get_or_create_thread("123.456", "C123", "U123")
        thread.add_message("user", "Message 1", db=mock_db, thread_key="C123:123.456")
        thread.add_message("assistant", "Response 1", db=mock_db, thread_key="C123:123.456")
        
        # Verify DB was called
        assert mock_db.cache_message.call_count == 2
        mock_db.cache_message.assert_any_call("C123:123.456", "user", "Message 1", None, None)
        mock_db.cache_message.assert_any_call("C123:123.456", "assistant", "Response 1", None, None)
    
    @patch('thread_manager.threading.Thread')
    def test_regression_message_limit(self, mock_thread):
        """Regression test: Ensure message history is not limited with DB"""
        manager = ThreadStateManager()
        thread = manager.get_or_create_thread("123.456", "C123")
        
        # Add many messages
        for i in range(100):
            thread.add_message("user", f"Message {i}")
        
        # Should keep all messages (no 20 message limit)
        assert len(thread.messages) == 100
        
        # clear_old_messages should be no-op
        thread.clear_old_messages(keep_last=5)
        assert len(thread.messages) == 100  # Still 100, not limited
    
    @patch('thread_manager.threading.Thread')
    def test_diagnostic_thread_state(self, mock_thread):
        """Diagnostic test: Log thread state for debugging"""
        manager = ThreadStateManager()
        thread = manager.get_or_create_thread("123.456", "C123")
        
        # Set up some state
        thread.add_message("user", "Hello")
        thread.add_message("assistant", "Hi there")
        thread.config_overrides = {"model": "gpt-5-nano"}
        thread.is_processing = True
        
        # Diagnostic info
        diagnostic_info = {
            "thread_key": f"{thread.channel_id}:{thread.thread_ts}",
            "message_count": len(thread.messages),
            "config_overrides": thread.config_overrides,
            "is_processing": thread.is_processing,
            "has_system_prompt": thread.system_prompt is not None,
            "last_message": thread.messages[-1] if thread.messages else None
        }
        
        print(f"\nDiagnostic Thread Info: {diagnostic_info}")
        
        # Verify state
        assert diagnostic_info["message_count"] == 2
        assert diagnostic_info["is_processing"] is True
        assert diagnostic_info["config_overrides"]["model"] == "gpt-5-nano"
    
    @patch('thread_manager.threading.Thread')
    def test_scenario_concurrent_thread_access(self, mock_thread):
        """Scenario test: Multiple users accessing different threads concurrently"""
        manager = ThreadStateManager()
        
        # User 1 in thread 1
        thread1_acquired = manager.acquire_thread_lock("123.456", "C123", timeout=0)
        assert thread1_acquired is True
        
        # User 2 in thread 2 (different thread, should succeed)
        thread2_acquired = manager.acquire_thread_lock("789.012", "C456", timeout=0)
        assert thread2_acquired is True
        
        # User 3 trying to access thread 1 (should fail - busy)
        thread1_again = manager.acquire_thread_lock("123.456", "C123", timeout=0)
        assert thread1_again is False
        
        # Release thread 1
        manager.release_thread_lock("123.456", "C123")
        
        # Now user 3 can access thread 1
        thread1_retry = manager.acquire_thread_lock("123.456", "C123", timeout=0)
        assert thread1_retry is True
        
        # Clean up
        manager.release_thread_lock("123.456", "C123")
        manager.release_thread_lock("789.012", "C456")