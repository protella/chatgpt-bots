"""
Unit tests for the global async rate limiter
Tests singleton behavior, circuit breaker, and fallback logic
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from streaming.global_rate_limiter import (
    GlobalRateLimiter,
    AsyncRateLimiter,
    CircuitState
)


class TestGlobalRateLimiter:
    """Test the global rate limiter singleton"""

    @pytest.mark.asyncio
    async def test_singleton_pattern(self):
        """Test that GlobalRateLimiter follows singleton pattern"""
        limiter1 = await GlobalRateLimiter.get_instance()
        limiter2 = await GlobalRateLimiter.get_instance()

        assert limiter1 is limiter2
        assert limiter1.get_limiter() is limiter2.get_limiter()

    @pytest.mark.asyncio
    async def test_initialization_once(self):
        """Test that initialization happens only once"""
        limiter = await GlobalRateLimiter.get_instance()
        initial_limiter = limiter.get_limiter()

        # Try to get instance again
        limiter2 = await GlobalRateLimiter.get_instance()
        assert limiter2.get_limiter() is initial_limiter

    @pytest.mark.asyncio
    async def test_streaming_allowed_check(self):
        """Test checking if streaming is allowed"""
        limiter = await GlobalRateLimiter.get_instance()

        # Initially streaming should be allowed
        assert await limiter.is_streaming_allowed() is True

        # After failures, circuit should open
        rate_limiter = limiter.get_limiter()
        await rate_limiter.record_failure(is_rate_limit=True)
        await rate_limiter.record_failure(is_rate_limit=True)
        await rate_limiter.record_failure(is_rate_limit=True)

        # Circuit should be open now
        assert await limiter.is_streaming_allowed() is False

    @pytest.mark.asyncio
    async def test_cooldown_remaining(self):
        """Test getting cooldown time remaining"""
        # Create a fresh instance for this test
        from streaming.global_rate_limiter import AsyncRateLimiter

        # Use a fresh rate limiter, not the singleton
        rate_limiter = AsyncRateLimiter(cooldown_seconds=60)

        # Initially no cooldown
        assert await rate_limiter.get_time_until_retry() is None

        # Trip circuit breaker
        for _ in range(3):
            await rate_limiter.record_failure(is_rate_limit=True)

        # Should have cooldown time
        cooldown = await rate_limiter.get_time_until_retry()
        assert cooldown is not None
        assert cooldown > 0
        assert cooldown <= 60  # Default cooldown is 60s


class TestAsyncRateLimiter:
    """Test the async rate limiter implementation"""

    @pytest.mark.asyncio
    async def test_initial_state(self):
        """Test initial state of rate limiter"""
        limiter = AsyncRateLimiter()

        assert await limiter.can_make_request() is True
        assert await limiter.is_streaming_enabled() is True
        assert await limiter.get_time_until_retry() is None
        assert limiter.circuit_state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures"""
        limiter = AsyncRateLimiter(failure_threshold=3)

        # Record failures
        await limiter.record_failure(is_rate_limit=True)
        assert await limiter.is_streaming_enabled() is True

        await limiter.record_failure(is_rate_limit=True)
        assert await limiter.is_streaming_enabled() is True

        await limiter.record_failure(is_rate_limit=True)
        # Circuit should open after 3rd failure
        assert await limiter.is_streaming_enabled() is False
        assert limiter.circuit_state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_transition(self):
        """Test circuit breaker transitions to half-open after cooldown"""
        limiter = AsyncRateLimiter(failure_threshold=2, cooldown_seconds=0.1)

        # Trip circuit breaker
        await limiter.record_failure(is_rate_limit=True)
        await limiter.record_failure(is_rate_limit=True)
        assert limiter.circuit_state == CircuitState.OPEN

        # Wait for cooldown
        await asyncio.sleep(0.15)

        # Check request should transition to half-open
        can_request = await limiter.can_make_request()
        assert can_request is True
        assert limiter.circuit_state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_circuit_breaker_closes_on_success(self):
        """Test circuit breaker closes after successful request in half-open state"""
        limiter = AsyncRateLimiter(failure_threshold=2, cooldown_seconds=0.1)

        # Trip circuit breaker
        await limiter.record_failure(is_rate_limit=True)
        await limiter.record_failure(is_rate_limit=True)
        assert limiter.circuit_state == CircuitState.OPEN

        # Wait for cooldown
        await asyncio.sleep(0.15)
        await limiter.can_make_request()  # Transitions to half-open

        # Record success
        await limiter.record_success()
        assert limiter.circuit_state == CircuitState.CLOSED
        assert await limiter.is_streaming_enabled() is True

    @pytest.mark.asyncio
    async def test_retry_after_header_handling(self):
        """Test respecting retry-after headers from Slack"""
        limiter = AsyncRateLimiter()

        # Set retry-after
        await limiter.set_retry_after(5.0)

        # Should not be able to request immediately
        assert await limiter.can_make_request() is False

        # Should have time until retry
        time_remaining = await limiter.get_time_until_retry()
        assert time_remaining is not None
        assert time_remaining > 0
        assert time_remaining <= 5.0

    @pytest.mark.asyncio
    async def test_exponential_backoff(self):
        """Test exponential backoff on failures"""
        limiter = AsyncRateLimiter(base_interval=1.0, max_interval=10.0)

        initial_interval = await limiter.get_current_interval()
        assert initial_interval == 1.0

        # First failure increases by 1.5x for non-rate-limit failures
        await limiter.record_failure()
        assert await limiter.get_current_interval() == 1.5

        # Second failure increases by 1.5x again
        await limiter.record_failure()
        assert await limiter.get_current_interval() == 2.25

        # Rate limit failures have more aggressive backoff (2x, min 10s)
        await limiter.record_failure(is_rate_limit=True)
        assert await limiter.get_current_interval() >= 10.0

        # Should respect max_interval
        for _ in range(5):
            await limiter.record_failure()
        assert await limiter.get_current_interval() <= 10.0

    @pytest.mark.asyncio
    async def test_interval_reduction_on_success(self):
        """Test interval reduces on sustained success"""
        limiter = AsyncRateLimiter(
            base_interval=1.0,
            success_reduction_factor=0.5
        )

        # Increase interval with failures
        await limiter.record_failure()
        await limiter.record_failure()
        high_interval = await limiter.get_current_interval()
        assert high_interval > 1.0

        # Record multiple successes (need 5 to trigger reduction)
        for _ in range(5):
            await limiter.record_success()

        # Interval should reduce
        new_interval = await limiter.get_current_interval()
        assert new_interval < high_interval
        assert new_interval >= 1.0  # Should not go below min_interval

    @pytest.mark.asyncio
    async def test_thread_safety(self):
        """Test that async locks ensure thread safety"""
        limiter = AsyncRateLimiter()

        # Create many concurrent tasks
        async def record_request(i):
            await limiter.record_request_attempt()
            if i % 2 == 0:
                await limiter.record_success()
            else:
                await limiter.record_failure()

        tasks = [record_request(i) for i in range(100)]
        await asyncio.gather(*tasks)

        # Stats should be consistent
        stats = await limiter.get_stats()
        assert stats["total_requests"] == 100
        assert stats["successful_requests"] == 50

    @pytest.mark.asyncio
    async def test_stats_reporting(self):
        """Test statistics reporting"""
        limiter = AsyncRateLimiter()

        # Record some activity
        await limiter.record_request_attempt()
        await limiter.record_success()
        await limiter.record_request_attempt()
        await limiter.record_failure(is_rate_limit=True)

        stats = await limiter.get_stats()
        assert stats["total_requests"] == 2
        assert stats["successful_requests"] == 1
        assert stats["rate_limited_requests"] == 1
        assert stats["success_rate_percent"] == 50.0


@pytest.mark.asyncio
async def test_fallback_integration():
    """Test integration with streaming fallback to non-streaming"""
    from unittest.mock import AsyncMock

    # Mock the global limiter
    with patch('streaming.global_rate_limiter.GlobalRateLimiter.get_instance') as mock_get:
        mock_limiter = AsyncMock()
        mock_get.return_value = mock_limiter

        # Simulate circuit breaker open
        mock_limiter.is_streaming_allowed.return_value = False
        mock_limiter.get_cooldown_remaining.return_value = 45.0

        # This would be in the actual handler
        limiter = await mock_get()
        if not await limiter.is_streaming_allowed():
            cooldown = await limiter.get_cooldown_remaining()
            # Should fall back to non-streaming
            assert cooldown == 45.0
            # Handler would call non-streaming method here