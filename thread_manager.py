"""
Thread State Management for Slack Bot V2
Manages conversation state, locks, and memory for each Slack thread
"""
import time
import asyncio
from collections import deque
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from logger import LoggerMixin
from config import config
from token_counter import TokenCounter

# Shared stateless estimator for incremental context-size tracking
_ESTIMATOR = TokenCounter()


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
    has_trimmed_messages: bool = False  # Track if messages have been trimmed from this thread
    has_summary_head: bool = False  # Phase S: a compaction summary head message is present
    has_shown_80_percent_warning: bool = False  # Track if we've shown the 80% context warning
    current_model: str = field(default_factory=lambda: config.gpt_model)  # Track current model for token limits
    participants: Dict[str, str] = field(default_factory=dict)  # user_id -> display name, for the @mention roster
    # Usage-driven budgeting: authoritative context size from the API's response.usage
    # after each call, plus chars/4 estimates for messages added between calls.
    context_tokens: int = 0
    
    def add_message(self, role: str, content: Any, db = None, thread_key: str = None, message_ts: str = None, metadata: Dict[str, Any] = None, token_counter: Optional[TokenCounter] = None, max_tokens: int = None):
        """Add a message to the thread history with optional metadata and token management.

        Phase S: messages are NOT persisted — Slack is the only transcript and state is
        rebuilt from conversations.replies on cold load. The db/thread_key params are kept
        for signature compatibility. message_ts is stamped into the message metadata so
        hidden-context injection (image analyses) and summary boundaries can key on it.
        """
        msg = {
            "role": role,
            "content": content
        }

        # Add metadata if provided; stamp the Slack ts so in-memory state carries it
        if metadata or message_ts:
            msg["metadata"] = dict(metadata) if metadata else {}
            if message_ts and "ts" not in msg["metadata"]:
                msg["metadata"]["ts"] = message_ts

        self.messages.append(msg)
        self.last_activity = time.time()

        # Usage-driven budgeting: increment the tracked size with a cheap estimate;
        # the next record_usage() replaces it with the API's authoritative number.
        self.context_tokens += (token_counter or _ESTIMATOR).count_message_tokens(msg)

        # Check token limit and trim if necessary
        if token_counter and max_tokens:
            self._trim_to_token_limit(token_counter, max_tokens, db, thread_key)
    
    def _trim_to_token_limit(self, token_counter: TokenCounter, max_tokens: int, db = None, thread_key: str = None):
        """Trim messages to fit within token limit"""
        import logging
        logger = logging.getLogger(__name__)
        
        current_tokens = token_counter.count_thread_tokens(self.messages)
        
        if current_tokens <= max_tokens:
            return
        
        logger.info(f"Thread exceeds token limit ({current_tokens} > {max_tokens}), trimming oldest messages")
        
        # Find first non-system message index
        start_index = 0
        for i, msg in enumerate(self.messages):
            if msg.get("role") not in ["system", "developer"]:
                start_index = i
                break
        
        # Remove messages from the beginning (after system message)
        removed_count = 0
        messages_to_remove = []
        
        while current_tokens > max_tokens and len(self.messages) > start_index + 1:
            if start_index < len(self.messages) - 1:
                removed_msg = self.messages.pop(start_index)
                messages_to_remove.append(removed_msg)
                removed_count += 1
                
                current_tokens = token_counter.count_thread_tokens(self.messages)
                logger.debug(f"Removed message {removed_count}, tokens now: {current_tokens}")
            else:
                logger.warning("Cannot trim further - would remove current message")
                break
        
        if removed_count > 0:
            logger.info(f"Trimmed {removed_count} messages to fit token limit")
    
    def record_usage(self, input_tokens: int = 0, output_tokens: int = 0):
        """Record the API's authoritative usage for the last call. input+output is the
        true size of the context that rides into the next turn — REPLACES the
        accumulated estimates."""
        total = (input_tokens or 0) + (output_tokens or 0)
        if total > 0:
            self.context_tokens = total

    def reset_context_estimate(self, token_counter: Optional[TokenCounter] = None):
        """Re-estimate the tracked context size from current messages (after cold
        rebuilds and compaction, when no fresh usage number exists yet)."""
        self.context_tokens = (token_counter or _ESTIMATOR).count_thread_tokens(self.messages)

    def get_recent_messages(self, count: int = 6) -> List[Dict[str, Any]]:
        """Get the most recent messages for context"""
        return self.messages[-count:] if self.messages else []
    
    def clear_old_messages(self, _keep_last: int = 20):
        """Keep only the most recent messages to manage memory"""
        # With database, we don't need to limit messages
        # This method is kept for backward compatibility but does nothing
        pass


@dataclass
class DocumentLedger:
    """Ledger for tracking documents per thread"""
    thread_ts: str
    documents: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_document(self, content: str, filename: str, mime_type: str,
                    page_structure: Optional[Dict[str, Any]] = None,
                    total_pages: Optional[int] = None,
                    summary: Optional[str] = None,
                    metadata: Optional[Dict[str, Any]] = None,
                    timestamp: float = None,
                    db = None, thread_id: Optional[str] = None,
                    message_ts: Optional[str] = None,
                    file_id: Optional[str] = None,
                    url_private: Optional[str] = None,
                    size_bytes: Optional[int] = None):
        """Add a document to the ledger.

        Content is NEVER persisted (CLAUDE.md pitfall 6a): the DB row and the
        in-memory entry hold summary + metadata + the Slack CDN ref only. The
        ``content`` parameter is accepted for interface stability but only used
        to derive a fallback summary when none was provided.

        Args:
            content: Full document text (transient; not stored)
            filename: Original filename
            mime_type: Document MIME type
            page_structure: Optional page/sheet structure info as dict
            total_pages: Total page/sheet count
            summary: Attach-time summary (the only content-bearing field)
            metadata: Additional metadata (size, author, etc.)
            timestamp: When the document was added
            db: Optional database manager for persistence
            thread_id: Optional thread ID for database storage
            message_ts: Message timestamp to link document to specific message
            file_id: Slack file id (read_document lookup key)
            url_private: Slack CDN URL for authenticated re-download
            size_bytes: Original file size
        """
        if timestamp is None:
            timestamp = time.time()

        if not summary and content:
            # Fallback so a row is never contentless if summarization failed upstream
            summary = ("[excerpt of original — full document available via read_document]\n"
                       + content[:1500])

        entry = {
            "filename": filename,
            "mime_type": mime_type,
            "content": None,  # never held; re-derived on demand via read_document
            "page_structure": page_structure,
            "total_pages": total_pages,
            "summary": summary,
            "timestamp": timestamp,
            "metadata": metadata,
            "file_id": file_id,
            "url_private": url_private,
            "size_bytes": size_bytes,
        }
        self.documents.append(entry)

        if db and thread_id:
            db.save_document(
                thread_id=thread_id,
                filename=filename,
                mime_type=mime_type,
                summary=summary,
                file_id=file_id,
                url_private=url_private,
                size_bytes=size_bytes,
                page_structure=page_structure,
                total_pages=total_pages,
                metadata=metadata,
                message_ts=message_ts
            )
    
    def get_recent_documents(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent documents"""
        return self.documents[-count:] if self.documents else []
    
    def clear_old_documents(self, _keep_last: int = 10):
        """Keep only the most recent documents to manage memory"""
        # With database, we don't need to limit documents
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
    
    def clear_old_images(self, _keep_last: int = 10):
        """Keep only the most recent images to manage memory"""
        # With database, we don't need to limit images
        # This method is kept for backward compatibility but does nothing
        pass


class AsyncThreadLockManager(LoggerMixin):
    """
    Async version of ThreadLockManager using asyncio.Lock
    Manages thread locks and processing state without force-release corruption
    """

    def __init__(self):
        self._locks: Dict[str, asyncio.Lock] = {}
        self._lock_acquisition_times: Dict[str, float] = {}  # Track when locks were acquired
        self._global_lock = asyncio.Lock()
        self.log_info("AsyncThreadLockManager initialized")

    async def get_lock(self, thread_key: str) -> asyncio.Lock:
        """Get or create a lock for a specific thread"""
        async with self._global_lock:
            if thread_key not in self._locks:
                self._locks[thread_key] = asyncio.Lock()
                self.log_debug(f"Created new async lock for thread {thread_key}")
            return self._locks[thread_key]

    async def record_acquisition(self, thread_key: str):
        """Record when a lock was acquired"""
        async with self._global_lock:
            self._lock_acquisition_times[thread_key] = time.time()
            self.log_debug(f"Async lock acquired for thread {thread_key}")

    async def clear_acquisition(self, thread_key: str):
        """Clear the acquisition time when lock is released"""
        async with self._global_lock:
            if thread_key in self._lock_acquisition_times:
                del self._lock_acquisition_times[thread_key]
                # Don't log here - the caller already logs

    async def get_stuck_threads(self, max_duration: int = 300) -> List[str]:
        """Get list of threads that have been locked too long"""
        stuck = []
        now = time.time()
        async with self._global_lock:
            for thread_key, acquire_time in self._lock_acquisition_times.items():
                if now - acquire_time > max_duration:
                    stuck.append(thread_key)
        return stuck

    async def is_busy(self, thread_key: str) -> bool:
        """Check if a thread is currently processing"""
        lock = await self.get_lock(thread_key)
        # For asyncio.Lock, we can't check if it's locked without acquiring
        # So we try to acquire with timeout 0
        try:
            await asyncio.wait_for(lock.acquire(), timeout=0)
            lock.release()
            return False
        except asyncio.TimeoutError:
            return True

    def locked_sync(self, thread_key: str) -> bool:
        """Non-acquiring peek: is this conversation's lock currently held?
        Dict access on the single event loop — safe without the global lock.
        Advisory only (the answer can change right after); callers must not
        use it for correctness, only to skip cosmetic work (e.g. a thinking
        indicator that would be deleted immediately for a queued message)."""
        lock = self._locks.get(thread_key)
        return bool(lock and lock.locked())

    async def cleanup_old_locks(self, _max_age: int = 3600):
        """Remove locks that haven't been used recently"""
        # This is a placeholder for potential cleanup logic
        # In practice, async locks are lightweight and can persist
        pass


    # ASYNC VERSION OF ThreadStateManager
    # =============================

    def create_async_manager(self, db=None):
        """Factory method to create async version of this manager"""
        return AsyncThreadStateManager(db=db, existing_state=self)


class AsyncThreadStateManager(LoggerMixin):
    """Async version of ThreadStateManager - manages conversation state for all threads"""

    def __init__(self, db=None, existing_state=None):
        # Copy existing state if provided (for migration)
        if existing_state:
            self._threads = existing_state._threads.copy()
            self._assets = existing_state._assets.copy()
            self._documents = existing_state._documents.copy()
        else:
            self._threads: Dict[str, ThreadState] = {}
            self._assets: Dict[str, AssetLedger] = {}
            self._documents: Dict[str, DocumentLedger] = {}

        self._lock_manager = AsyncThreadLockManager()
        self._state_lock = asyncio.Lock()
        self._token_counter = TokenCounter(config.gpt_model)
        self.db = db  # Optional database manager
        self._watchdog_task = None
        self._watchdog_started = False
        # Upload-in-flight latches: image upload + DB row land AFTER the thread lock
        # releases, so a fast follow-up "edit it" could resolve its target before the
        # new image exists. Editors await the latch before resolving targets.
        self._upload_events: Dict[str, asyncio.Event] = {}
        # F1 — background image generation registry: at most one in-flight generation
        # per thread. Entry: {generation_id, task, started_at, prompt_summary}. Used by
        # follow-up turns (suffix in-flight note + image-intent rejection) and shutdown
        # (cancel/await the tasks). All access is synchronous == atomic on the loop.
        self._active_generations: Dict[str, dict] = {}
        # Threads whose warm in-memory state is missing at least one Slack message
        # (e.g. a message dropped from an overfull pending queue, or a crash mid-queue).
        # The next request on that thread refetches from Slack (the transcript) before
        # processing. Consumed by the rebuild path.
        self._needs_refresh: set = set()
        # Phase Q — conversational queueing: messages arriving while a conversation's
        # lock is held append here instead of being busy-rejected. The finishing turn
        # drains the queue into ONE batched catch-up turn. Enqueue and pop are plain
        # synchronous operations, so on the single event loop each is atomic — the
        # invariant is that nothing awaits between a lock-contention check and the
        # corresponding enqueue, and the drain pops while STILL HOLDING the lock, so
        # no message can slip between "queue looks empty" and "lock released".
        self._pending_queues: Dict[str, deque] = {}
        self.log_info(f"AsyncThreadStateManager initialized {'with' if db else 'without'} database")

    # --- Phase Q: pending-message queue (busy rejection retired) ---

    def is_thread_processing(self, thread_ts: str, channel_id: str) -> bool:
        """Advisory non-acquiring peek at the conversation lock (cosmetic uses only)."""
        return self._lock_manager.locked_sync(f"{channel_id}:{thread_ts}")

    def enqueue_pending(self, thread_key: str, message) -> bool:
        """Queue a message that arrived while its conversation was mid-processing.
        Returns False (and flags the thread for a transcript refetch) when the queue
        is at QUEUE_MAX_PENDING — the message is dropped from warm state but Slack
        still has it, so the refetch recovers it in context."""
        queue = self._pending_queues.setdefault(thread_key, deque())
        max_pending = int(getattr(config, "queue_max_pending", 25))
        if len(queue) >= max_pending:
            self.mark_needs_refresh(thread_key)
            self.log_warning(
                f"Pending queue full for {thread_key} ({max_pending}); dropping message "
                f"from warm state (transcript refetch flagged)"
            )
            return False
        queue.append(message)
        self.log_debug(f"Queued message for busy conversation {thread_key} (pending={len(queue)})")
        return True

    def pending_count(self, thread_key: str) -> int:
        return len(self._pending_queues.get(thread_key) or ())

    def pop_pending_batch(self, thread_key: str, max_batch: int) -> list:
        """Pop up to max_batch pending messages, FIFO. Synchronous == atomic on the
        event loop. Anything left behind drains on the following turn."""
        queue = self._pending_queues.get(thread_key)
        if not queue:
            return []
        batch = []
        while queue and len(batch) < max_batch:
            batch.append(queue.popleft())
        if not queue:
            self._pending_queues.pop(thread_key, None)
        return batch

    def mark_needs_refresh(self, thread_key: str):
        """Flag a thread whose warm state is now incomplete (e.g. a busy-rejected
        message that Slack has but our in-memory state never saw)."""
        self._needs_refresh.add(thread_key)

    def consume_needs_refresh(self, thread_key: str) -> bool:
        """Pop-and-return the refresh flag. True → caller must refetch from Slack."""
        if thread_key in self._needs_refresh:
            self._needs_refresh.discard(thread_key)
            return True
        return False

    def mark_upload_started(self, thread_key: str):
        """Signal that an asset upload for this thread is in flight."""
        event = self._upload_events.get(thread_key)
        if event is None:
            event = asyncio.Event()
            self._upload_events[thread_key] = event
        event.clear()

    def mark_upload_finished(self, thread_key: str):
        """Signal that the in-flight asset upload (incl. its DB row) has landed."""
        event = self._upload_events.get(thread_key)
        if event is not None:
            event.set()

    async def wait_for_uploads(self, thread_key: str, timeout: float = 10.0):
        """Wait (bounded) for any in-flight asset upload on this thread to land."""
        event = self._upload_events.get(thread_key)
        if event is None or event.is_set():
            return
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self.log_warning(f"Timed out waiting for in-flight upload on {thread_key}; proceeding")

    # --- F1: background image-generation registry ---

    def register_generation(self, thread_key: str, generation_id: str,
                            prompt_summary: str, task: Optional[asyncio.Task] = None):
        """Register an in-flight background image generation for this thread."""
        self._active_generations[thread_key] = {
            "generation_id": generation_id,
            "task": task,
            "started_at": time.monotonic(),
            "prompt_summary": prompt_summary,
        }

    def attach_generation_task(self, thread_key: str, generation_id: str, task: asyncio.Task):
        """Store the scheduled task handle (minted after the id, needed for shutdown).
        ID-conditional so a stale job can't overwrite a newer registration."""
        entry = self._active_generations.get(thread_key)
        if entry is not None and entry["generation_id"] == generation_id:
            entry["task"] = task

    def finish_generation(self, thread_key: str, generation_id: str) -> bool:
        """Clear the in-flight generation, but ONLY if the id still matches — a stale
        job (already superseded by a newer one) must never clear the newer entry."""
        entry = self._active_generations.get(thread_key)
        if entry is not None and entry["generation_id"] == generation_id:
            self._active_generations.pop(thread_key, None)
            return True
        return False

    def generation_in_flight(self, thread_key: str) -> Optional[dict]:
        """Advisory peek at the in-flight generation entry, or None. Force-clears (and
        logs) an entry older than api_timeout_image + 30s — a watchdog against a job
        that died without clearing itself."""
        entry = self._active_generations.get(thread_key)
        if entry is None:
            return None
        max_age = float(config.api_timeout_image) + 30.0
        if time.monotonic() - entry["started_at"] > max_age:
            self.log_warning(
                f"Force-clearing stale generation {entry['generation_id']} on "
                f"{thread_key} (age > {max_age:.0f}s)")
            self._active_generations.pop(thread_key, None)
            return None
        return entry

    async def cancel_generations(self, timeout: float = 5.0):
        """Cancel and await all registered generation tasks (shutdown). Bounded so a
        wedged job can't stall shutdown."""
        tasks = [e["task"] for e in self._active_generations.values() if e.get("task")]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=timeout)
            except asyncio.TimeoutError:
                self.log_warning("Timed out awaiting generation tasks during shutdown")
        self._active_generations.clear()

    def _start_async_watchdog(self):
        """Start the background task that monitors for stuck locks"""
        async def async_watchdog():
            api_timeout = int(config.api_timeout_read)

            # Calculate maximum possible timeout (for image operations which can take 5 minutes)
            max_possible_timeout = 300  # 5 minutes for image operations
            # Use a buffer after the longest possible operation
            max_lock_duration = max_possible_timeout + 10  # 10s buffer after longest operation

            self.log_info(f"Async thread lock watchdog started with {max_lock_duration}s timeout (base API timeout: {api_timeout}s, max operation timeout: {max_possible_timeout}s)")

            while True:
                try:
                    await asyncio.sleep(10)  # Check every 10 seconds

                    # Get stuck threads (locked for more than API timeout duration)
                    stuck_threads = await self._lock_manager.get_stuck_threads(max_duration=max_lock_duration)

                    for thread_key in stuck_threads:
                        self.log_error(f"Detected stuck async thread: {thread_key} - after {max_lock_duration}s")

                        # Mark thread as no longer processing
                        if thread_key in self._threads:
                            self._threads[thread_key].is_processing = False
                            # Store that this thread had a timeout for notification
                            self._threads[thread_key].had_timeout = True

                        # With async locks, we don't force-release - they timeout naturally
                        self.log_warning(f"Async thread {thread_key} will timeout naturally with asyncio.wait_for")

                except Exception as e:
                    self.log_error(f"Async watchdog error: {e}", exc_info=True)

        # Start the watchdog task
        self._watchdog_task = asyncio.create_task(async_watchdog())

    async def acquire_thread_lock(self, thread_ts: str, channel_id: str, timeout: float = 0, user_id: Optional[str] = None) -> bool:
        """
        Async version of acquire_thread_lock - try to acquire lock for thread processing

        Args:
            thread_ts: Thread timestamp
            channel_id: Channel ID
            timeout: How long to wait for lock (0 = don't wait)
            user_id: Optional user ID

        Returns:
            True if lock acquired, False if thread is busy
        """
        # Start watchdog on first use
        if not self._watchdog_started:
            self._start_async_watchdog()
            self._watchdog_started = True

        thread_key = f"{channel_id}:{thread_ts}"
        lock = await self._lock_manager.get_lock(thread_key)

        # Acquire lock with proper timeout handling
        try:
            if timeout > 0:
                await asyncio.wait_for(lock.acquire(), timeout=timeout)
            else:
                # Non-blocking attempt with immediate return if busy
                acquired = lock.locked()
                if acquired:
                    return False  # Lock is already held
                await lock.acquire()

            # Successfully acquired lock
            thread = await self.get_or_create_thread_async(thread_ts, channel_id, user_id)
            thread.is_processing = True
            # Record lock acquisition time for watchdog
            await self._lock_manager.record_acquisition(thread_key)
            self.log_debug(f"Acquired async lock for thread {thread_key}")
            return True

        except asyncio.TimeoutError:
            # Lock is busy
            return False

    async def release_thread_lock(self, thread_ts: str, channel_id: str):
        """Async version of release_thread_lock - release lock for thread processing"""
        thread_key = f"{channel_id}:{thread_ts}"
        lock = await self._lock_manager.get_lock(thread_key)

        thread = await self.get_thread_async(thread_ts, channel_id)
        if thread:
            thread.is_processing = False

        try:
            lock.release()
            # Clear lock acquisition time for watchdog
            await self._lock_manager.clear_acquisition(thread_key)
            self.log_debug(f"Released async lock for thread {thread_key}")
        except RuntimeError:
            self.log_warning(f"Attempted to release unheld async lock for {thread_key}")

    async def get_or_create_thread_async(self, thread_ts: str, channel_id: str, user_id: Optional[str] = None) -> ThreadState:
        """Async version of get_or_create_thread - get existing thread state or create new one"""
        thread_key = f"{channel_id}:{thread_ts}"

        async with self._state_lock:
            if thread_key not in self._threads:
                # Create new thread state
                thread_state = ThreadState(
                    thread_ts=thread_ts,
                    channel_id=channel_id
                )

                # If database available, load thread CONFIG only. Phase S: no message
                # cache — a fresh thread state starts empty and the processor rebuilds
                # the transcript from Slack (conversations.replies) on first use.
                if self.db:
                    # Get or create in database (async twin — sync sqlite here blocked
                    # the event loop on every new-thread message)
                    await self.db.get_or_create_thread_async(thread_key, channel_id, user_id)

                    # Load config from database if exists
                    thread_config = await self.db.get_thread_config_async(thread_key)
                    if thread_config:
                        thread_state.config_overrides = thread_config

                self._threads[thread_key] = thread_state
                self.log_debug(f"Created new async thread state for {thread_key}")

            thread = self._threads[thread_key]
            thread.last_activity = time.time()

            # Always refresh config from database to ensure we have latest settings
            if self.db:
                thread_config = await self.db.get_thread_config_async(thread_key)
                if thread_config:
                    thread.config_overrides = thread_config
                    self.log_debug(f"Refreshed thread config from database for {thread_key}")

                # Update database activity
                await self.db.update_thread_activity_async(thread_key)

            return thread

    async def get_thread_async(self, thread_ts: str, channel_id: str) -> Optional[ThreadState]:
        """Async version of get_thread - get thread state if it exists"""
        thread_key = f"{channel_id}:{thread_ts}"
        return self._threads.get(thread_key)

    async def is_thread_busy(self, thread_ts: str, channel_id: str) -> bool:
        """Async version of is_thread_busy - check if a thread is currently processing"""
        thread_key = f"{channel_id}:{thread_ts}"
        return await self._lock_manager.is_busy(thread_key)

    async def cleanup_old_threads(self, max_age: int = 86400):
        """Async version of cleanup_old_threads - remove thread states that haven't been active recently"""
        current_time = time.time()
        threads_to_remove = []

        async with self._state_lock:
            for key, thread in self._threads.items():
                if current_time - thread.last_activity > max_age and not thread.is_processing:
                    threads_to_remove.append(key)

            for key in threads_to_remove:
                del self._threads[key]
                # Also clean up associated asset and document ledgers
                thread_ts = key.split(":")[1]
                if thread_ts in self._assets:
                    del self._assets[thread_ts]
                if thread_ts in self._documents:
                    del self._documents[thread_ts]
                self.log_debug(f"Cleaned up old async thread state: {key}")

        if threads_to_remove:
            self.log_info(f"Cleaned up {len(threads_to_remove)} old async thread states")

    def get_or_create_asset_ledger(self, thread_ts: str) -> AssetLedger:
        """Get or create asset ledger for a thread"""
        if thread_ts not in self._assets:
            # Use a simple approach - just create if missing
            # This is safe for dict access in most cases
            self._assets[thread_ts] = AssetLedger(thread_ts=thread_ts)
            self.log_debug(f"Created new asset ledger for thread {thread_ts}")
        return self._assets[thread_ts]

    def get_asset_ledger(self, thread_ts: str) -> Optional[AssetLedger]:
        """Get asset ledger if it exists"""
        return self._assets.get(thread_ts)

    def get_or_create_document_ledger(self, thread_ts: str) -> DocumentLedger:
        """Get or create document ledger for a thread"""
        if thread_ts not in self._documents:
            # Use a simple approach - just create if missing
            # This is safe for dict access in most cases
            self._documents[thread_ts] = DocumentLedger(thread_ts=thread_ts)
            self.log_debug(f"Created new document ledger for thread {thread_ts}")
        return self._documents[thread_ts]

    def get_document_ledger(self, thread_ts: str) -> Optional[DocumentLedger]:
        """Get document ledger if it exists"""
        return self._documents.get(thread_ts)

    async def update_thread_documents(self, thread_ts: str, channel_id: str, documents: List[Dict[str, Any]]):
        """Update documents for a specific thread"""
        thread_key = f"{channel_id}:{thread_ts}"
        document_ledger = self.get_or_create_document_ledger(thread_ts)

        # Add documents to ledger
        for doc in documents:
            document_ledger.add_document(
                content=doc.get('content', ''),
                filename=doc.get('filename', 'unknown'),
                mime_type=doc.get('mime_type', 'text/plain'),
                page_structure=doc.get('page_structure'),
                total_pages=doc.get('total_pages'),
                summary=doc.get('summary'),
                metadata=doc.get('metadata'),
                timestamp=doc.get('timestamp'),
                db=self.db,
                thread_id=thread_key,
                message_ts=doc.get('message_ts'),
                file_id=doc.get('file_id'),
                url_private=doc.get('url_private') or doc.get('url'),
                size_bytes=doc.get('size_bytes'),
            )

        self.log_info(f"Updated documents for thread {thread_ts}: {len(documents)} documents")

    async def get_thread_documents(self, thread_ts: str, channel_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get documents for a specific thread"""
        thread_key = f"{channel_id}:{thread_ts}"

        # Try to get from database first
        if self.db:
            documents = await self.db.get_thread_documents_async(thread_key, limit=limit)
            if documents:
                self.log_debug(f"Retrieved {len(documents)} documents from database for {thread_key}")
                return documents

        # Fallback to in-memory ledger
        document_ledger = self.get_document_ledger(thread_ts)
        if document_ledger:
            recent_docs = document_ledger.get_recent_documents(count=limit or 10)
            self.log_debug(f"Retrieved {len(recent_docs)} documents from memory for {thread_ts}")
            return recent_docs

        return []

    async def update_thread_config(self, thread_ts: str, channel_id: str, config_overrides: Dict[str, Any]):
        """Update configuration for a specific thread"""
        thread = await self.get_or_create_thread_async(thread_ts, channel_id)
        thread.config_overrides.update(config_overrides)

        # Save to database if available
        if self.db:
            thread_key = f"{channel_id}:{thread_ts}"
            await self.db.save_thread_config_async(thread_key, thread.config_overrides)

        self.log_info(f"Updated config for thread {thread_ts}: {config_overrides}")

    def get_thread(self, thread_ts: str, channel_id: str) -> Optional[ThreadState]:
        """Get thread state if it exists (sync version for compatibility)"""
        thread_key = f"{channel_id}:{thread_ts}"
        return self._threads.get(thread_key)

    async def cleanup(self):
        """Cleanup method for graceful shutdown"""
        self.log_info("Cleaning up ThreadManager...")
        # Release all thread locks
        if hasattr(self, '_locks'):
            for thread_key in list(self._locks.keys()):
                lock = self._locks.get(thread_key)
                if lock and lock.locked():
                    self.log_debug(f"Releasing lock for thread {thread_key}")
            # Clear locks dictionary
            self._locks.clear()
        self.log_info("ThreadManager cleanup completed")

    def get_thread_if_exists(self, thread_key: str) -> Optional[ThreadState]:
        """Get thread state if it exists, without creating a new one"""
        return self._threads.get(thread_key)

    def get_thread_state(self, channel_id: str, thread_ts: str) -> Optional[ThreadState]:
        """Get thread state if it exists (alternative method signature)"""
        thread_key = f"{channel_id}:{thread_ts}"
        return self._threads.get(thread_key)

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about managed threads"""
        return {
            "active_threads": len(self._threads),
            "asset_ledgers": len(self._assets),
            "document_ledgers": len(self._documents),
            "processing_threads": sum(1 for t in self._threads.values() if t.is_processing)
        }