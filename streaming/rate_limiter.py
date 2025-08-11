"""
RateLimitManager class for handling rate limiting and circuit breaker patterns
Manages update intervals with exponential backoff and workspace protection
"""

import time
from typing import Optional, Dict, Any
from enum import Enum
from logger import LoggerMixin


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit breaker active, streaming disabled
    HALF_OPEN = "half_open"  # Testing if service has recovered


class RateLimitManager(LoggerMixin):
    """
    Manages rate limiting with exponential backoff and circuit breaker pattern
    Protects workspace from excessive API calls and handles 429 responses
    """
    
    def __init__(
        self,
        base_interval: float = 2.0,
        min_interval: float = 1.0,
        max_interval: float = 30.0,
        failure_threshold: int = 5,
        cooldown_seconds: int = 300,  # 5 minutes
        success_reduction_factor: float = 0.9
    ):
        """
        Initialize the rate limit manager
        
        Args:
            base_interval: Base update interval in seconds
            min_interval: Minimum allowed interval
            max_interval: Maximum backoff interval
            failure_threshold: Consecutive failures before circuit opens
            cooldown_seconds: Time to wait before attempting recovery
            success_reduction_factor: Factor to reduce interval on success
        """
        self.base_interval = base_interval
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.success_reduction_factor = success_reduction_factor
        
        # Current state
        self.current_interval = base_interval
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.circuit_state = CircuitState.CLOSED
        self.circuit_open_time: Optional[float] = None
        self.last_429_time: Optional[float] = None
        self.retry_after: Optional[float] = None
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.rate_limited_requests = 0
        self.circuit_trips = 0
        
        self.log_info(f"RateLimitManager initialized: base={base_interval}s, "
                     f"threshold={failure_threshold}, cooldown={cooldown_seconds}s")
    
    def can_make_request(self) -> bool:
        """
        Check if a request can be made now
        
        Returns:
            True if request is allowed
        """
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
    
    def record_request_attempt(self) -> None:
        """Record that a request attempt was made"""
        self.total_requests += 1
    
    def record_success(self) -> None:
        """
        Record a successful request
        Gradually reduces interval and resets failure counters
        """
        self.successful_requests += 1
        self.consecutive_failures = 0
        self.consecutive_successes += 1
        self.retry_after = None
        
        # Close circuit breaker if it was open or half-open
        if self.circuit_state != CircuitState.CLOSED:
            self.circuit_state = CircuitState.CLOSED
            self.circuit_open_time = None
            self.log_info("Circuit breaker closed after successful request")
        
        # Gradually reduce interval on sustained success
        if self.consecutive_successes >= 3 and self.current_interval > self.base_interval:
            old_interval = self.current_interval
            self.current_interval = max(
                self.current_interval * self.success_reduction_factor,
                self.base_interval
            )
            
            if abs(old_interval - self.current_interval) > 0.1:
                self.log_info(f"Interval reduced from {old_interval:.1f}s to {self.current_interval:.1f}s "
                             f"after {self.consecutive_successes} successes")
    
    def record_failure(self, is_rate_limit: bool = False) -> None:
        """
        Record a failed request
        Increases interval and may trip circuit breaker
        
        Args:
            is_rate_limit: True if failure was due to rate limiting (429)
        """
        self.consecutive_failures += 1
        self.consecutive_successes = 0
        
        if is_rate_limit:
            self.rate_limited_requests += 1
            self.last_429_time = time.time()
            self.log_warning(f"Rate limit hit (429 response) - failure #{self.consecutive_failures}")
        else:
            self.log_warning(f"Request failed - failure #{self.consecutive_failures}")
        
        # Exponential backoff
        old_interval = self.current_interval
        self.current_interval = min(
            self.current_interval * 2,
            self.max_interval
        )
        
        if abs(old_interval - self.current_interval) > 0.1:
            self.log_warning(f"Interval increased from {old_interval:.1f}s to {self.current_interval:.1f}s "
                            f"due to failure")
        
        # Check if we should trip the circuit breaker
        if self.consecutive_failures >= self.failure_threshold:
            self._trip_circuit_breaker()
    
    def set_retry_after(self, retry_after_seconds: float) -> None:
        """
        Set retry-after time from 429 response header
        
        Args:
            retry_after_seconds: Seconds to wait before retrying
        """
        self.retry_after = time.time() + retry_after_seconds
        self.log_warning(f"Rate limit retry-after set to {retry_after_seconds}s")
        
        # Also update our interval to be at least the retry-after period
        if retry_after_seconds > self.current_interval:
            old_interval = self.current_interval
            self.current_interval = min(retry_after_seconds, self.max_interval)
            self.log_warning(f"Interval adjusted to {self.current_interval:.1f}s based on retry-after")
    
    def _trip_circuit_breaker(self) -> None:
        """Trip the circuit breaker to disable streaming"""
        if self.circuit_state != CircuitState.OPEN:
            self.circuit_state = CircuitState.OPEN
            self.circuit_open_time = time.time()
            self.circuit_trips += 1
            
            self.log_error(f"Circuit breaker OPENED after {self.consecutive_failures} consecutive failures. "
                          f"Streaming disabled for {self.cooldown_seconds}s")
    
    def get_current_interval(self) -> float:
        """Get the current update interval"""
        return self.current_interval
    
    def is_streaming_enabled(self) -> bool:
        """Check if streaming is currently enabled (circuit breaker closed)"""
        return self.circuit_state != CircuitState.OPEN
    
    def get_time_until_retry(self) -> Optional[float]:
        """
        Get seconds until next retry is allowed
        
        Returns:
            Seconds to wait, or None if retry is allowed now
        """
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
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get rate limiting statistics
        
        Returns:
            Dictionary with current stats
        """
        current_time = time.time()
        success_rate = (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        
        stats = {
            # Current state
            "current_interval": self.current_interval,
            "circuit_state": self.circuit_state.value,
            "streaming_enabled": self.is_streaming_enabled(),
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            
            # Timing
            "time_until_retry": self.get_time_until_retry(),
            "last_429_ago": (current_time - self.last_429_time) if self.last_429_time else None,
            
            # Counters
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "rate_limited_requests": self.rate_limited_requests,
            "circuit_trips": self.circuit_trips,
            "success_rate_percent": round(success_rate, 1),
            
            # Configuration
            "base_interval": self.base_interval,
            "max_interval": self.max_interval,
            "failure_threshold": self.failure_threshold,
            "cooldown_seconds": self.cooldown_seconds,
        }
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset statistics counters (keeps current state)"""
        self.total_requests = 0
        self.successful_requests = 0
        self.rate_limited_requests = 0
        self.log_info("Rate limit statistics reset")
    
    def force_reset(self) -> None:
        """Force reset to initial state (emergency override)"""
        self.current_interval = self.base_interval
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.circuit_state = CircuitState.CLOSED
        self.circuit_open_time = None
        self.retry_after = None
        
        self.log_warning("RateLimitManager force reset to initial state")
    
    def log_periodic_stats(self) -> None:
        """Log current statistics (call periodically for monitoring)"""
        stats = self.get_stats()
        self.log_info(f"Rate limit stats: {stats['successful_requests']}/{stats['total_requests']} success "
                     f"({stats['success_rate_percent']}%), interval={stats['current_interval']:.1f}s, "
                     f"state={stats['circuit_state']}")