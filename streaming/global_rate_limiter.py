"""
Global rate limiter singleton with async-native design
Provides thread-safe, shared rate limiting across all handlers
"""

import asyncio
import time
from typing import Optional, Dict, Any
from enum import Enum
from logger import LoggerMixin


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit breaker active, streaming disabled
    HALF_OPEN = "half_open"  # Testing if service has recovered


class AsyncRateLimiter(LoggerMixin):
    """
    Async-native rate limiter with circuit breaker pattern
    Thread-safe implementation using asyncio locks
    """

    def __init__(
        self,
        base_interval: float = 2.0,
        min_interval: float = 2.0,  # Increased from 1.0 to respect Slack limits
        max_interval: float = 30.0,
        failure_threshold: int = 3,
        cooldown_seconds: int = 60,
        success_reduction_factor: float = 0.95,  # Slower reduction from 0.9
        failure_window_seconds: float = 10.0  # Track failures within this window
    ):
        """Initialize the async rate limiter"""
        self.base_interval = base_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.success_reduction_factor = success_reduction_factor
        self.failure_window_seconds = failure_window_seconds

        # Current state (protected by lock)
        self.current_interval = base_interval
        self.consecutive_successes = 0
        self.circuit_state = CircuitState.CLOSED
        self.circuit_open_time: Optional[float] = None
        self.last_429_time: Optional[float] = None
        self.retry_after: Optional[float] = None

        # Time-window based failure tracking
        self.recent_failures: list[float] = []  # Timestamps of recent failures
        self.recent_rate_limits: list[float] = []  # Timestamps of recent 429s

        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.rate_limited_requests = 0
        self.circuit_trips = 0

        # Async lock for thread safety
        self._lock = asyncio.Lock()

        self.log_info(f"AsyncRateLimiter initialized: base={base_interval}s, "
                     f"threshold={failure_threshold}, cooldown={cooldown_seconds}s")

    def _cleanup_old_failures(self, current_time: float) -> None:
        """Remove failures outside the time window (internal, already locked)"""
        cutoff_time = current_time - self.failure_window_seconds
        self.recent_failures = [t for t in self.recent_failures if t > cutoff_time]
        self.recent_rate_limits = [t for t in self.recent_rate_limits if t > cutoff_time]

    async def can_make_request(self) -> bool:
        """
        Check if a request can be made now (async-safe)

        Returns:
            True if request is allowed
        """
        async with self._lock:
            current_time = time.time()

            # Check circuit breaker state
            if self.circuit_state == CircuitState.OPEN:
                if current_time - self.circuit_open_time >= self.cooldown_seconds:
                    self.circuit_state = CircuitState.HALF_OPEN
                    self.log_info("Circuit breaker moved to HALF_OPEN state")
                else:
                    return False

            # Check if we're still within retry-after period
            if self.retry_after and current_time < self.retry_after:
                return False

            return True

    async def record_request_attempt(self) -> None:
        """Record that a request attempt was made (async-safe)"""
        async with self._lock:
            self.total_requests += 1

    async def record_success(self) -> None:
        """
        Record a successful request (async-safe)
        Gradually reduces interval but doesn't reset failure tracking
        """
        async with self._lock:
            current_time = time.time()
            self.successful_requests += 1
            self.consecutive_successes += 1
            self.retry_after = None

            # Clean up old failures
            self._cleanup_old_failures(current_time)

            # Only close circuit breaker if it was HALF_OPEN (testing recovery)
            # Do NOT close if it's OPEN and still in cooldown period
            if self.circuit_state == CircuitState.HALF_OPEN:
                self.circuit_state = CircuitState.CLOSED
                self.circuit_open_time = None
                self.log_info("Circuit breaker closed after successful request in HALF_OPEN state")

            # Gradually reduce interval on sustained success
            # But only if we haven't had recent rate limits
            if (self.consecutive_successes >= 5 and
                self.current_interval > self.min_interval and
                len(self.recent_rate_limits) == 0):
                old_interval = self.current_interval
                self.current_interval = max(
                    self.current_interval * self.success_reduction_factor,
                    self.min_interval  # Use min_interval not base_interval
                )

                if abs(old_interval - self.current_interval) > 0.1:
                    self.log_info(f"Interval reduced from {old_interval:.1f}s to {self.current_interval:.1f}s "
                                 f"after {self.consecutive_successes} successes")

    async def record_failure(self, is_rate_limit: bool = False) -> None:
        """
        Record a failed request (async-safe)
        Increases interval and may trip circuit breaker based on failures within time window

        Args:
            is_rate_limit: True if failure was due to rate limiting (429)
        """
        async with self._lock:
            current_time = time.time()

            # Clean up old failures first
            self._cleanup_old_failures(current_time)

            # Add this failure to the window
            self.recent_failures.append(current_time)

            # Reset consecutive_successes
            self.consecutive_successes = 0

            if is_rate_limit:
                self.rate_limited_requests += 1
                self.last_429_time = current_time
                self.recent_rate_limits.append(current_time)
                self.log_warning(f"Rate limit hit (429 response) - {len(self.recent_failures)} failures in window")
            else:
                self.log_warning(f"Request failed - {len(self.recent_failures)} failures in window")

            # More aggressive backoff for rate limits
            if is_rate_limit:
                # For 429s, jump straight to a higher interval
                old_interval = self.current_interval
                self.current_interval = min(
                    max(self.current_interval * 2, 10.0),  # At least 10s for 429s
                    self.max_interval
                )
            else:
                # Regular exponential backoff for other failures
                old_interval = self.current_interval
                self.current_interval = min(
                    self.current_interval * 1.5,
                    self.max_interval
                )

            if abs(old_interval - self.current_interval) > 0.1:
                self.log_warning(f"Interval increased from {old_interval:.1f}s to {self.current_interval:.1f}s "
                                f"due to failure")

            # Check if we should trip the circuit breaker based on failures in window
            if len(self.recent_failures) >= self.failure_threshold:
                await self._trip_circuit_breaker()

    async def set_retry_after(self, retry_after_seconds: float) -> None:
        """
        Set retry-after time from 429 response header (async-safe)

        Args:
            retry_after_seconds: Seconds to wait before retrying
        """
        async with self._lock:
            self.retry_after = time.time() + retry_after_seconds
            self.log_warning(f"Rate limit retry-after set to {retry_after_seconds}s")

            # Also update our interval to be at least the retry-after period
            if retry_after_seconds > self.current_interval:
                self.current_interval = min(retry_after_seconds, self.max_interval)
                self.log_warning(f"Interval adjusted to {self.current_interval:.1f}s based on retry-after")

    async def _trip_circuit_breaker(self) -> None:
        """Trip the circuit breaker to disable streaming (internal, already locked)"""
        if self.circuit_state != CircuitState.OPEN:
            self.circuit_state = CircuitState.OPEN
            self.circuit_open_time = time.time()
            self.circuit_trips += 1

            self.log_error(f"Circuit breaker OPENED after {len(self.recent_failures)} failures in {self.failure_window_seconds}s window. "
                          f"Streaming disabled for {self.cooldown_seconds}s")

    async def get_current_interval(self) -> float:
        """Get the current update interval (async-safe)"""
        async with self._lock:
            return self.current_interval

    async def is_streaming_enabled(self) -> bool:
        """Check if streaming is currently enabled (circuit breaker closed)"""
        async with self._lock:
            return self.circuit_state != CircuitState.OPEN

    async def get_time_until_retry(self) -> Optional[float]:
        """
        Get seconds until next retry is allowed (async-safe)

        Returns:
            Seconds to wait, or None if retry is allowed now
        """
        async with self._lock:
            current_time = time.time()

            # Check retry-after
            if self.retry_after and current_time < self.retry_after:
                return self.retry_after - current_time

            # Check circuit breaker cooldown
            if self.circuit_state == CircuitState.OPEN and self.circuit_open_time:
                cooldown_remaining = (self.circuit_open_time + self.cooldown_seconds) - current_time
                if cooldown_remaining > 0:
                    return cooldown_remaining

            return None

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get rate limiting statistics (async-safe)

        Returns:
            Dictionary with current stats
        """
        async with self._lock:
            current_time = time.time()
            self._cleanup_old_failures(current_time)
            success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0

            return {
                # Current state
                "current_interval": self.current_interval,
                "circuit_state": self.circuit_state.value,
                "streaming_enabled": self.circuit_state != CircuitState.OPEN,
                "failures_in_window": len(self.recent_failures),
                "rate_limits_in_window": len(self.recent_rate_limits),
                "consecutive_successes": self.consecutive_successes,

                # Timing
                "time_until_retry": await self.get_time_until_retry() if self._lock.locked() else None,
                "last_429_ago": (current_time - self.last_429_time) if self.last_429_time else None,

                # Counters
                "total_requests": self.total_requests,
                "successful_requests": self.successful_requests,
                "rate_limited_requests": self.rate_limited_requests,
                "circuit_trips": self.circuit_trips,
                "success_rate_percent": round(success_rate, 1),
            }


class GlobalRateLimiter(LoggerMixin):
    """
    Singleton global rate limiter shared across all threads and handlers
    Uses async-native patterns throughout
    """
    _instance: Optional['GlobalRateLimiter'] = None
    _init_lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    async def initialize(self):
        """Async initialization to ensure thread safety"""
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:  # Double-check pattern
                return

            # Single shared async rate limiter for ALL Slack API calls
            self.slack_limiter = AsyncRateLimiter(
                base_interval=2.0,
                min_interval=2.0,  # Respect Slack's rate limits
                max_interval=30.0,
                failure_threshold=3,  # Trip circuit after 3 failures in window
                cooldown_seconds=60,  # 60 second cooldown before re-enabling streaming
                success_reduction_factor=0.95,  # Slower reduction to prevent thrashing
                failure_window_seconds=10.0  # Track failures within 10 second window
            )

            self._initialized = True
            self.log_info("GlobalRateLimiter initialized - shared across all threads")

    @classmethod
    async def get_instance(cls) -> 'GlobalRateLimiter':
        """Get or create the singleton instance (async-safe)"""
        instance = cls()
        await instance.initialize()
        return instance

    def get_limiter(self) -> AsyncRateLimiter:
        """Get the shared async rate limiter instance"""
        if not self._initialized:
            raise RuntimeError("GlobalRateLimiter not initialized. Call get_instance() first")
        return self.slack_limiter

    async def is_streaming_allowed(self) -> bool:
        """Check if streaming is currently allowed globally"""
        if not self._initialized:
            await self.initialize()
        return await self.slack_limiter.is_streaming_enabled()

    async def get_cooldown_remaining(self) -> Optional[float]:
        """Get seconds until streaming is allowed again"""
        if not self._initialized:
            await self.initialize()
        return await self.slack_limiter.get_time_until_retry()

    async def log_global_stats(self) -> None:
        """Log current global rate limit statistics"""
        if not self._initialized:
            await self.initialize()

        stats = await self.slack_limiter.get_stats()
        self.log_info(f"Global rate limit stats: {stats['successful_requests']}/{stats['total_requests']} success "
                     f"({stats['success_rate_percent']}%), interval={stats['current_interval']:.1f}s, "
                     f"state={stats['circuit_state']}, failures_in_window={stats['failures_in_window']}, "
                     f"circuit_trips={stats['circuit_trips']}")