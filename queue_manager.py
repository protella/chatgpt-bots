import asyncio
import threading
from typing import Dict


class QueueManager:
    """
    Manages processing state for different threads.
    
    This class tracks which threads are currently processing a request
    to prevent multiple simultaneous requests in the same thread,
    while allowing different threads to process concurrently.
    """
    
    def __init__(self):
        """Initialize the manager with empty dictionaries for processing states and locks."""
        self._processing: Dict[str, bool] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._lock = asyncio.Lock()  # Global lock for thread dictionary access
        
        # For synchronous access
        self._sync_lock = threading.RLock()
    
    async def get_lock(self, thread_id: str) -> asyncio.Lock:
        """
        Get or create a lock for a specific thread.
        
        Args:
            thread_id: The ID of the thread/conversation
            
        Returns:
            An asyncio.Lock instance for the thread
        """
        async with self._lock:
            if thread_id not in self._locks:
                self._locks[thread_id] = asyncio.Lock()
            return self._locks[thread_id]
    
    async def is_processing(self, thread_id: str) -> bool:
        """
        Check if a thread is currently processing a request.
        
        Args:
            thread_id: The ID of the thread/conversation
            
        Returns:
            True if the thread is processing, False otherwise
        """
        async with self._lock:
            return self._processing.get(thread_id, False)
    
    async def start_processing(self, thread_id: str) -> bool:
        """
        Try to start processing for a thread.
        
        Args:
            thread_id: The ID of the thread/conversation
            
        Returns:
            True if processing was started, False if thread is already processing
        """
        thread_lock = await self.get_lock(thread_id)
        async with thread_lock:
            async with self._lock:
                if self._processing.get(thread_id, False):
                    return False
                self._processing[thread_id] = True
                return True
    
    async def finish_processing(self, thread_id: str) -> None:
        """
        Mark a thread as finished processing.
        
        Args:
            thread_id: The ID of the thread/conversation
        """
        thread_lock = await self.get_lock(thread_id)
        async with thread_lock:
            async with self._lock:
                self._processing[thread_id] = False
    
    async def cleanup_thread(self, thread_id: str) -> None:
        """
        Clean up resources associated with a thread.
        
        Args:
            thread_id: The ID of the thread/conversation
        """
        async with self._lock:
            self._processing.pop(thread_id, None)
            self._locks.pop(thread_id, None)
    
    async def cleanup_all_threads(self) -> None:
        """
        Clean up all resources associated with all threads.
        """
        async with self._lock:
            self._processing.clear()
            self._locks.clear() 
            
    # Synchronous versions of the methods for use with the Slack bot
    
    def is_processing_sync(self, thread_id: str) -> bool:
        """
        Check if a thread is currently processing a request (synchronous version).
        
        Args:
            thread_id: The ID of the thread/conversation
            
        Returns:
            True if the thread is processing, False otherwise
        """
        with self._sync_lock:
            return self._processing.get(thread_id, False)
    
    def start_processing_sync(self, thread_id: str) -> bool:
        """
        Try to start processing for a thread (synchronous version).
        
        Args:
            thread_id: The ID of the thread/conversation
            
        Returns:
            True if processing was started, False if thread is already processing
        """
        with self._sync_lock:
            if self._processing.get(thread_id, False):
                return False
            self._processing[thread_id] = True
            return True
    
    def finish_processing_sync(self, thread_id: str) -> None:
        """
        Mark a thread as finished processing (synchronous version).
        
        Args:
            thread_id: The ID of the thread/conversation
        """
        with self._sync_lock:
            self._processing[thread_id] = False
    
    def cleanup_thread_sync(self, thread_id: str) -> None:
        """
        Clean up resources associated with a thread (synchronous version).
        
        Args:
            thread_id: The ID of the thread/conversation
        """
        with self._sync_lock:
            self._processing.pop(thread_id, None)
    
    def cleanup_all_threads_sync(self) -> None:
        """
        Clean up all resources associated with all threads (synchronous version).
        """
        with self._sync_lock:
            self._processing.clear() 