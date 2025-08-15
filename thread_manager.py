"""
Thread State Management for Slack Bot V2
Manages conversation state, locks, and memory for each Slack thread
"""
import time
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from threading import Lock
from logger import LoggerMixin
from config import config


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
    had_timeout: bool = False  # Track if this thread had a timeout for user notification
    
    def add_message(self, role: str, content: Any, db = None, thread_key: str = None, message_ts: str = None):
        """Add a message to the thread history"""
        self.messages.append({
            "role": role,
            "content": content
        })
        self.last_activity = time.time()
        
        # Save to database if available
        if db and thread_key:
            db.cache_message(thread_key, role, content, message_ts)
    
    def get_recent_messages(self, count: int = 6) -> List[Dict[str, Any]]:
        """Get the most recent messages for context"""
        return self.messages[-count:] if self.messages else []
    
    def clear_old_messages(self, keep_last: int = 20):
        """Keep only the most recent messages to manage memory"""
        # With database, we don't need to limit messages
        # This method is kept for backward compatibility but does nothing
        pass


@dataclass 
class AssetLedger:
    """Ledger for tracking generated images per thread"""
    thread_ts: str
    images: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_image(self, image_data: str, prompt: str, timestamp: float, slack_url: Optional[str] = None, source: str = "generated", original_url: Optional[str] = None, db = None, thread_id: Optional[str] = None, analysis: Optional[str] = None):
        """Add an image to the ledger
        
        Args:
            image_data: Base64 encoded image data (NOT stored in DB)
            prompt: Description or prompt for the image
            timestamp: When the image was added
            slack_url: URL if uploaded to Slack
            source: Source of image - 'generated', 'attachment', 'url'
            original_url: Original URL if image was downloaded from web
            db: Optional database manager for persistence
            thread_id: Optional thread ID for database storage
            analysis: Optional vision analysis for database storage
        """
        # Store in memory (for backward compatibility, but without base64 if DB available)
        if db:
            # Don't store base64 in memory when DB is available
            self.images.append({
                "data": None,  # No base64 in memory when using DB
                "prompt": prompt,  # Full prompt in memory when DB available
                "timestamp": timestamp,
                "slack_url": slack_url,
                "source": source,
                "original_url": original_url
            })
            
            # Store metadata in database (no base64)
            if thread_id and (slack_url or original_url):
                db.save_image_metadata(
                    thread_id=thread_id,
                    url=slack_url or original_url,
                    image_type=source,
                    prompt=prompt,  # Full prompt to DB
                    analysis=analysis,
                    metadata={"timestamp": timestamp}
                )
        else:
            # Legacy behavior when no DB
            self.images.append({
                "data": image_data,
                "prompt": prompt[:100],  # Truncated without DB
                "timestamp": timestamp,
                "slack_url": slack_url,
                "source": source,
                "original_url": original_url
            })
    
    def add_url_image(self, image_data: str, url: str, timestamp: float, slack_url: Optional[str] = None):
        """Add an image downloaded from a URL"""
        self.add_image(
            image_data=image_data,
            prompt=f"Image from URL: {url}",
            timestamp=timestamp,
            slack_url=slack_url,
            source="url",
            original_url=url
        )
    
    def get_recent_images(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent images"""
        return self.images[-count:] if self.images else []
    
    def clear_old_images(self, keep_last: int = 10):
        """Keep only the most recent images to manage memory"""
        # With database, we don't need to limit images
        # This method is kept for backward compatibility but does nothing
        pass


class ThreadLockManager(LoggerMixin):
    """
    Manages thread locks and processing state
    (Renamed from QueueManager to better reflect actual behavior)
    """
    
    def __init__(self):
        self._locks: Dict[str, Lock] = {}
        self._lock_acquisition_times: Dict[str, float] = {}  # Track when locks were acquired
        self._global_lock = Lock()
        self.log_info("ThreadLockManager initialized")
    
    def get_lock(self, thread_key: str) -> Lock:
        """Get or create a lock for a specific thread"""
        with self._global_lock:
            if thread_key not in self._locks:
                self._locks[thread_key] = Lock()
                self.log_debug(f"Created new lock for thread {thread_key}")
            return self._locks[thread_key]
    
    def record_acquisition(self, thread_key: str):
        """Record when a lock was acquired"""
        with self._global_lock:
            self._lock_acquisition_times[thread_key] = time.time()
            self.log_debug(f"Lock acquired for thread {thread_key}")
    
    def clear_acquisition(self, thread_key: str):
        """Clear the acquisition time when lock is released"""
        with self._global_lock:
            if thread_key in self._lock_acquisition_times:
                del self._lock_acquisition_times[thread_key]
                self.log_debug(f"Lock released for thread {thread_key}")
    
    def get_stuck_threads(self, max_duration: int = 300) -> List[str]:
        """Get list of threads that have been locked too long"""
        stuck = []
        now = time.time()
        with self._global_lock:
            for thread_key, acquire_time in self._lock_acquisition_times.items():
                if now - acquire_time > max_duration:
                    stuck.append(thread_key)
        return stuck
    
    def force_release(self, thread_key: str) -> bool:
        """Force release a stuck lock"""
        with self._global_lock:
            if thread_key in self._locks:
                lock = self._locks[thread_key]
                # Try to release if locked
                if lock.locked():
                    try:
                        # This is risky but necessary for stuck threads
                        lock.release()
                        self.log_warning(f"Force-released lock for thread {thread_key}")
                        self.clear_acquisition(thread_key)
                        return True
                    except RuntimeError:
                        self.log_error(f"Failed to force-release lock for {thread_key}")
                        return False
        return False
    
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
    
    def __init__(self, db = None):
        self._threads: Dict[str, ThreadState] = {}
        self._assets: Dict[str, AssetLedger] = {}
        self._lock_manager = ThreadLockManager()
        self._state_lock = Lock()
        self.db = db  # Optional database manager
        self._watchdog_thread = None
        self._start_watchdog()
        self.log_info(f"ThreadStateManager initialized {'with' if db else 'without'} database")
    
    def _start_watchdog(self):
        """Start the background thread that monitors for stuck locks"""
        def watchdog():
            # Use same timeout as API calls for consistency
            max_lock_duration = int(config.api_timeout_read)
            self.log_info(f"Thread lock watchdog started with {max_lock_duration}s timeout")
            
            while True:
                try:
                    time.sleep(30)  # Check every 30 seconds
                    
                    # Get stuck threads (locked for more than API timeout duration)
                    stuck_threads = self._lock_manager.get_stuck_threads(max_duration=max_lock_duration)
                    
                    for thread_key in stuck_threads:
                        self.log_error(f"Detected stuck thread: {thread_key} - attempting force release after {max_lock_duration}s")
                        
                        # Mark thread as no longer processing
                        if thread_key in self._threads:
                            self._threads[thread_key].is_processing = False
                            # Store that this thread had a timeout for notification
                            self._threads[thread_key].had_timeout = True
                        
                        # Force release the lock
                        if self._lock_manager.force_release(thread_key):
                            self.log_warning(f"Successfully force-released stuck thread: {thread_key}")
                            # Note: We don't add a system message here
                            # The MessageProcessor will handle sending a timeout message to the user
                        else:
                            self.log_error(f"Failed to force-release stuck thread: {thread_key}")
                            
                except Exception as e:
                    self.log_error(f"Watchdog error: {e}", exc_info=True)
        
        self._watchdog_thread = threading.Thread(target=watchdog, daemon=True, name="ThreadLockWatchdog")
        self._watchdog_thread.start()
    
    def get_or_create_thread(self, thread_ts: str, channel_id: str, user_id: Optional[str] = None) -> ThreadState:
        """Get existing thread state or create new one"""
        thread_key = f"{channel_id}:{thread_ts}"
        
        with self._state_lock:
            if thread_key not in self._threads:
                # Create new thread state
                thread_state = ThreadState(
                    thread_ts=thread_ts,
                    channel_id=channel_id
                )
                
                # If database available, check for persisted state
                if self.db:
                    # Get or create in database
                    db_thread = self.db.get_or_create_thread(thread_key, channel_id, user_id)
                    
                    # Load config from database if exists
                    thread_config = self.db.get_thread_config(thread_key)
                    if thread_config:
                        thread_state.config_overrides = thread_config
                    
                    # Load cached messages from database
                    cached_messages = self.db.get_cached_messages(thread_key)
                    if cached_messages:
                        # Convert DB format to thread format
                        for msg in cached_messages:
                            thread_state.messages.append({
                                "role": msg["role"],
                                "content": msg["content"]
                            })
                        self.log_debug(f"Loaded {len(cached_messages)} cached messages for {thread_key}")
                
                self._threads[thread_key] = thread_state
                self.log_debug(f"Created new thread state for {thread_key}")
            
            thread = self._threads[thread_key]
            thread.last_activity = time.time()
            
            # Update database activity if available
            if self.db:
                self.db.update_thread_activity(thread_key)
            
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
    
    def acquire_thread_lock(self, thread_ts: str, channel_id: str, timeout: float = 0, user_id: Optional[str] = None) -> bool:
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
            thread = self.get_or_create_thread(thread_ts, channel_id, user_id)
            thread.is_processing = True
            # Record lock acquisition time for watchdog
            self._lock_manager.record_acquisition(thread_key)
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
            # Clear lock acquisition time for watchdog
            self._lock_manager.clear_acquisition(thread_key)
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
        
        # Save to database if available
        if self.db:
            thread_key = f"{channel_id}:{thread_ts}"
            self.db.save_thread_config(thread_key, thread.config_overrides)
        
        self.log_info(f"Updated config for thread {thread_ts}: {config_overrides}")
    
    def cleanup_old_threads(self, max_age: int = 86400):
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