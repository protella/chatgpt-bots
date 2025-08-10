"""
Thread State Management for Slack Bot V2
Manages conversation state, locks, and memory for each Slack thread
"""
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from threading import Lock
from logger import LoggerMixin


@dataclass
class ThreadState:
    """State for a single Slack thread"""
    thread_ts: str
    channel_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    system_prompt: Optional[str] = None
    last_activity: float = field(default_factory=time.time)
    is_processing: bool = False
    pending_clarification: Optional[Dict[str, Any]] = None
    
    def add_message(self, role: str, content: Any):
        """Add a message to the thread history"""
        self.messages.append({
            "role": role,
            "content": content
        })
        self.last_activity = time.time()
    
    def get_recent_messages(self, count: int = 6) -> List[Dict[str, Any]]:
        """Get the most recent messages for context"""
        return self.messages[-count:] if self.messages else []
    
    def clear_old_messages(self, keep_last: int = 20):
        """Keep only the most recent messages to manage memory"""
        if len(self.messages) > keep_last:
            self.messages = self.messages[-keep_last:]


@dataclass 
class AssetLedger:
    """Ledger for tracking generated images per thread"""
    thread_ts: str
    images: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_image(self, image_data: str, prompt: str, timestamp: float, slack_url: Optional[str] = None):
        """Add an image to the ledger"""
        self.images.append({
            "data": image_data,
            "prompt": prompt[:100],  # Store first 100 chars as breadcrumb
            "timestamp": timestamp,
            "slack_url": slack_url
        })
    
    def get_recent_images(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent images"""
        return self.images[-count:] if self.images else []
    
    def clear_old_images(self, keep_last: int = 10):
        """Keep only the most recent images to manage memory"""
        if len(self.images) > keep_last:
            self.images = self.images[-keep_last:]


class ThreadLockManager(LoggerMixin):
    """
    Manages thread locks and processing state
    (Renamed from QueueManager to better reflect actual behavior)
    """
    
    def __init__(self):
        self._locks: Dict[str, Lock] = {}
        self._global_lock = Lock()
        self.log_info("ThreadLockManager initialized")
    
    def get_lock(self, thread_key: str) -> Lock:
        """Get or create a lock for a specific thread"""
        with self._global_lock:
            if thread_key not in self._locks:
                self._locks[thread_key] = Lock()
                self.log_debug(f"Created new lock for thread {thread_key}")
            return self._locks[thread_key]
    
    def is_busy(self, thread_key: str) -> bool:
        """Check if a thread is currently processing"""
        lock = self.get_lock(thread_key)
        acquired = lock.acquire(blocking=False)
        if acquired:
            lock.release()
            return False
        return True
    
    def cleanup_old_locks(self, max_age: int = 3600):
        """Remove locks that haven't been used recently"""
        # This is a placeholder for potential cleanup logic
        # In practice, locks are lightweight and can persist
        pass


class ThreadStateManager(LoggerMixin):
    """Manages conversation state for all threads"""
    
    def __init__(self):
        self._threads: Dict[str, ThreadState] = {}
        self._assets: Dict[str, AssetLedger] = {}
        self._lock_manager = ThreadLockManager()
        self._state_lock = Lock()
        self.log_info("ThreadStateManager initialized")
    
    def get_or_create_thread(self, thread_ts: str, channel_id: str) -> ThreadState:
        """Get existing thread state or create new one"""
        thread_key = f"{channel_id}:{thread_ts}"
        
        with self._state_lock:
            if thread_key not in self._threads:
                self._threads[thread_key] = ThreadState(
                    thread_ts=thread_ts,
                    channel_id=channel_id
                )
                self.log_debug(f"Created new thread state for {thread_key}")
            
            thread = self._threads[thread_key]
            thread.last_activity = time.time()
            return thread
    
    def get_thread(self, thread_ts: str, channel_id: str) -> Optional[ThreadState]:
        """Get thread state if it exists"""
        thread_key = f"{channel_id}:{thread_ts}"
        return self._threads.get(thread_key)
    
    def get_or_create_asset_ledger(self, thread_ts: str) -> AssetLedger:
        """Get or create asset ledger for a thread"""
        with self._state_lock:
            if thread_ts not in self._assets:
                self._assets[thread_ts] = AssetLedger(thread_ts=thread_ts)
                self.log_debug(f"Created new asset ledger for thread {thread_ts}")
            return self._assets[thread_ts]
    
    def get_asset_ledger(self, thread_ts: str) -> Optional[AssetLedger]:
        """Get asset ledger if it exists"""
        return self._assets.get(thread_ts)
    
    def acquire_thread_lock(self, thread_ts: str, channel_id: str, timeout: float = 0) -> bool:
        """
        Try to acquire lock for thread processing
        
        Args:
            thread_ts: Thread timestamp
            channel_id: Channel ID
            timeout: How long to wait for lock (0 = don't wait)
        
        Returns:
            True if lock acquired, False if thread is busy
        """
        thread_key = f"{channel_id}:{thread_ts}"
        lock = self._lock_manager.get_lock(thread_key)
        
        # Handle None or 0 timeout - ensure it's a valid number
        if timeout is None or not isinstance(timeout, (int, float)):
            timeout = 0
        
        # Acquire lock with proper timeout handling
        # For non-blocking (timeout=0), use acquire(blocking=False) without timeout parameter
        if timeout > 0:
            acquired = lock.acquire(blocking=True, timeout=timeout)
        else:
            acquired = lock.acquire(blocking=False)
        
        if acquired:
            thread = self.get_or_create_thread(thread_ts, channel_id)
            thread.is_processing = True
            self.log_debug(f"Acquired lock for thread {thread_key}")
        
        return acquired
    
    def release_thread_lock(self, thread_ts: str, channel_id: str):
        """Release lock for thread processing"""
        thread_key = f"{channel_id}:{thread_ts}"
        lock = self._lock_manager.get_lock(thread_key)
        
        thread = self.get_thread(thread_ts, channel_id)
        if thread:
            thread.is_processing = False
        
        try:
            lock.release()
            self.log_debug(f"Released lock for thread {thread_key}")
        except RuntimeError:
            self.log_warning(f"Attempted to release unheld lock for {thread_key}")
    
    def is_thread_busy(self, thread_ts: str, channel_id: str) -> bool:
        """Check if a thread is currently processing"""
        thread_key = f"{channel_id}:{thread_ts}"
        return self._lock_manager.is_busy(thread_key)
    
    def update_thread_config(self, thread_ts: str, channel_id: str, config_overrides: Dict[str, Any]):
        """Update configuration for a specific thread"""
        thread = self.get_or_create_thread(thread_ts, channel_id)
        thread.config_overrides.update(config_overrides)
        self.log_info(f"Updated config for thread {thread_ts}: {config_overrides}")
    
    def cleanup_old_threads(self, max_age: int = 7200):
        """Remove thread states that haven't been active recently"""
        current_time = time.time()
        threads_to_remove = []
        
        with self._state_lock:
            for key, thread in self._threads.items():
                if current_time - thread.last_activity > max_age and not thread.is_processing:
                    threads_to_remove.append(key)
            
            for key in threads_to_remove:
                del self._threads[key]
                # Also clean up associated asset ledger
                thread_ts = key.split(":")[1]
                if thread_ts in self._assets:
                    del self._assets[thread_ts]
                self.log_debug(f"Cleaned up old thread state: {key}")
        
        if threads_to_remove:
            self.log_info(f"Cleaned up {len(threads_to_remove)} old thread states")
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about managed threads"""
        return {
            "active_threads": len(self._threads),
            "asset_ledgers": len(self._assets),
            "processing_threads": sum(1 for t in self._threads.values() if t.is_processing)
        }