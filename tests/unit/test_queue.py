import asyncio
import concurrent.futures
import threading
import time
from unittest import mock

import pytest
import pytest_asyncio

# Mock the logger before importing QueueManager
with mock.patch('app.core.logging.setup_logger', return_value=mock.MagicMock()):
    from app.core.queue import QueueManager


@pytest.fixture
def queue_manager():
    """Return a fresh QueueManager instance for each test."""
    # Reset the singleton instance
    QueueManager._instance = None
    with mock.patch('app.core.queue.logger'):
        manager = QueueManager()
        yield manager


def test_singleton_pattern():
    """Test that QueueManager uses the singleton pattern."""
    # Reset the singleton instance
    QueueManager._instance = None
    
    # Get two instances
    with mock.patch('app.core.queue.logger'):
        manager1 = QueueManager.get_instance()
        manager2 = QueueManager.get_instance()
    
    # Verify they're the same instance
    assert manager1 is manager2


def test_start_processing_sync(queue_manager):
    """Test that start_processing_sync works correctly."""
    # First call should succeed
    assert queue_manager.start_processing_sync("thread1") is True
    
    # Second call for same thread should fail
    assert queue_manager.start_processing_sync("thread1") is False
    
    # Different thread should succeed
    assert queue_manager.start_processing_sync("thread2") is True


def test_finish_processing_sync(queue_manager):
    """Test that finish_processing_sync works correctly."""
    # Start processing
    queue_manager.start_processing_sync("thread1")
    
    # Finish processing
    queue_manager.finish_processing_sync("thread1")
    
    # Should be able to start processing again
    assert queue_manager.start_processing_sync("thread1") is True


def test_cleanup_thread_sync(queue_manager):
    """Test that cleanup_thread_sync works correctly."""
    # Start processing
    queue_manager.start_processing_sync("thread1")
    
    # Clean up thread
    queue_manager.cleanup_thread_sync("thread1")
    
    # Should be able to start processing again
    assert queue_manager.start_processing_sync("thread1") is True


def test_cleanup_all_threads_sync(queue_manager):
    """Test that cleanup_all_threads_sync works correctly."""
    # Start processing for multiple threads
    queue_manager.start_processing_sync("thread1")
    queue_manager.start_processing_sync("thread2")
    
    # Clean up all threads
    queue_manager.cleanup_all_threads_sync()
    
    # Should be able to start processing again for all threads
    assert queue_manager.start_processing_sync("thread1") is True
    assert queue_manager.start_processing_sync("thread2") is True


def test_concurrent_sync_access():
    """Test that synchronous methods handle concurrent access correctly."""
    # Reset the singleton instance
    QueueManager._instance = None
    with mock.patch('app.core.queue.logger'):
        manager = QueueManager.get_instance()
    
    # Set up a shared counter
    counter = 0
    
    def worker(thread_id):
        nonlocal counter
        # Try to start processing
        if manager.start_processing_sync(thread_id):
            # Simulate work
            time.sleep(0.01)
            # Critical section
            nonlocal counter
            temp = counter
            time.sleep(0.01)  # Increase chance of race condition
            counter = temp + 1
            # Finish processing
            manager.finish_processing_sync(thread_id)
            return True
        return False
    
    # Create multiple threads trying to process the same thread_id
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit 10 tasks that all try to process the same thread_id
        futures = [executor.submit(worker, "same_thread") for _ in range(10)]
        
        # Wait for all tasks to complete
        results = [f.result() for f in futures]
    
    # Only one worker should have processed successfully
    assert sum(results) == 1
    # Counter should have been incremented exactly once
    assert counter == 1


@pytest.mark.asyncio
async def test_start_processing_async(queue_manager):
    """Test that start_processing works correctly."""
    # First call should succeed
    assert await queue_manager.start_processing("thread1") is True
    
    # Second call for same thread should fail
    assert await queue_manager.start_processing("thread1") is False
    
    # Different thread should succeed
    assert await queue_manager.start_processing("thread2") is True


@pytest.mark.asyncio
async def test_finish_processing_async(queue_manager):
    """Test that finish_processing works correctly."""
    # Start processing
    await queue_manager.start_processing("thread1")
    
    # Finish processing
    await queue_manager.finish_processing("thread1")
    
    # Should be able to start processing again
    assert await queue_manager.start_processing("thread1") is True


@pytest.mark.asyncio
async def test_cleanup_thread_async(queue_manager):
    """Test that cleanup_thread works correctly."""
    # Start processing
    await queue_manager.start_processing("thread1")
    
    # Clean up thread
    await queue_manager.cleanup_thread("thread1")
    
    # Should be able to start processing again
    assert await queue_manager.start_processing("thread1") is True


@pytest.mark.asyncio
async def test_cleanup_all_threads_async(queue_manager):
    """Test that cleanup_all_threads works correctly."""
    # Start processing for multiple threads
    await queue_manager.start_processing("thread1")
    await queue_manager.start_processing("thread2")
    
    # Clean up all threads
    await queue_manager.cleanup_all_threads()
    
    # Should be able to start processing again for all threads
    assert await queue_manager.start_processing("thread1") is True
    assert await queue_manager.start_processing("thread2") is True


@pytest.mark.asyncio
async def test_concurrent_async_access(queue_manager):
    """Test that async methods handle concurrent access correctly."""
    # Set up a shared counter
    counter = 0
    
    async def worker(thread_id):
        nonlocal counter
        # Try to start processing
        if await queue_manager.start_processing(thread_id):
            # Simulate work
            await asyncio.sleep(0.01)
            # Critical section
            nonlocal counter
            temp = counter
            await asyncio.sleep(0.01)  # Increase chance of race condition
            counter = temp + 1
            # Finish processing
            await queue_manager.finish_processing(thread_id)
            return True
        return False
    
    # Create multiple tasks trying to process the same thread_id
    tasks = [worker("same_thread") for _ in range(10)]
    results = await asyncio.gather(*tasks)
    
    # Only one worker should have processed successfully
    assert sum(results) == 1
    # Counter should have been incremented exactly once
    assert counter == 1 