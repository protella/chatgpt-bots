"""Unit tests for streaming modules"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from streaming.buffer import StreamingBuffer
from streaming.fence_handler import FenceHandler
from streaming.rate_limiter import RateLimitManager, CircuitState


class TestFenceHandler:
    """Test FenceHandler class"""
    
    @pytest.fixture
    def handler(self):
        """Create a FenceHandler instance"""
        return FenceHandler()
    
    def test_initialization(self, handler):
        """Test handler initialization"""
        assert handler.current_text == ""
    
    def test_reset(self, handler):
        """Test resetting handler state"""
        handler.update_text("some text")
        handler.reset()
        assert handler.current_text == ""
    
    def test_update_text(self, handler):
        """Test updating text"""
        test_text = "test content"
        handler.update_text(test_text)
        assert handler.current_text == test_text
    
    def test_display_safe_empty_text(self, handler):
        """Test display safe with empty text"""
        assert handler.get_display_safe_text() == ""
    
    def test_close_triple_backticks(self, handler):
        """Test closing unclosed triple backticks"""
        handler.update_text("```python\nprint('hello')")
        safe_text = handler.get_display_safe_text()
        assert safe_text.endswith("\n```")
    
    def test_close_triple_backticks_with_newline(self, handler):
        """Test closing triple backticks when text ends with newline"""
        handler.update_text("```python\nprint('hello')\n")
        safe_text = handler.get_display_safe_text()
        assert safe_text.endswith("```")
    
    def test_paired_triple_backticks(self, handler):
        """Test that paired triple backticks are not modified"""
        text = "```python\ncode\n```"
        handler.update_text(text)
        assert handler.get_display_safe_text() == text
    
    def test_close_single_backticks(self, handler):
        """Test closing unclosed single backticks"""
        handler.update_text("This is `inline code")
        safe_text = handler.get_display_safe_text()
        assert safe_text.endswith("`")
    
    def test_paired_single_backticks(self, handler):
        """Test that paired single backticks are not modified"""
        text = "This is `inline` code"
        handler.update_text(text)
        assert handler.get_display_safe_text() == text
    
    def test_single_backticks_inside_code_block(self, handler):
        """Test single backticks inside code blocks are ignored"""
        text = "```\n`single` backticks inside\n```"
        handler.update_text(text)
        assert handler.get_display_safe_text() == text
    
    def test_unclosed_code_block_with_single_backticks(self, handler):
        """Test unclosed code block containing single backticks"""
        handler.update_text("```\n`single` backticks")
        safe_text = handler.get_display_safe_text()
        # Should close the code block but not add extra backtick for the paired singles
        assert safe_text.count("```") == 2
    
    def test_get_unclosed_triple_count(self, handler):
        """Test counting unclosed triple backticks"""
        handler.update_text("```")
        assert handler.get_unclosed_triple_count() == 1
        
        handler.update_text("```\n```")
        assert handler.get_unclosed_triple_count() == 0
        
        handler.update_text("```\n```\n```")
        assert handler.get_unclosed_triple_count() == 1
    
    def test_get_unclosed_single_count(self, handler):
        """Test counting unclosed single backticks"""
        handler.update_text("`")
        assert handler.get_unclosed_single_count() == 1
        
        handler.update_text("``")
        assert handler.get_unclosed_single_count() == 0
        
        handler.update_text("`inline`")
        assert handler.get_unclosed_single_count() == 0
    
    def test_is_in_code_block(self, handler):
        """Test checking if position is in code block"""
        text = "before\n```python\ninside\n```\nafter"
        handler.update_text(text)
        
        # Test various positions
        assert not handler.is_in_code_block(5)  # "before"
        assert handler.is_in_code_block(15)  # inside code block
        assert not handler.is_in_code_block(len(text))  # after closing
    
    def test_is_in_code_block_unclosed(self, handler):
        """Test is_in_code_block with unclosed block"""
        handler.update_text("```python\ncode here")
        assert handler.is_in_code_block()  # Default to end of text
    
    def test_get_current_language_hint(self, handler):
        """Test getting language hint for current code block"""
        handler.update_text("```python\ncode")
        assert handler.get_current_language_hint() == "python"
        
        handler.update_text("```\ncode")
        assert handler.get_current_language_hint() is None
        
        handler.update_text("normal text")
        assert handler.get_current_language_hint() is None
    
    def test_analyze_fences(self, handler):
        """Test fence analysis"""
        handler.update_text("```python\ncode")
        analysis = handler.analyze_fences()
        
        assert analysis["unclosed_triple_fences"] == 1
        assert analysis["unclosed_single_fences"] == 0
        assert analysis["in_code_block"] is True
        assert analysis["current_language"] == "python"
        assert analysis["text_length"] == len("```python\ncode")
    
    @pytest.mark.critical
    def test_critical_fence_safety(self, handler):
        """Critical test for fence safety in various scenarios"""
        # Unclosed triple should be closed
        handler.update_text("```")
        assert "```" in handler.get_display_safe_text()
        assert handler.get_display_safe_text().count("```") == 2
        
        # Unclosed single should be closed
        handler.update_text("`test")
        assert handler.get_display_safe_text().count("`") == 2
        
        # Mixed unclosed should handle both
        handler.update_text("```python\n`inline")
        safe = handler.get_display_safe_text()
        assert safe.count("```") == 2  # Closed triple
        # Single backtick inside code block doesn't need closing


class TestStreamingBuffer:
    """Test StreamingBuffer class"""
    
    @pytest.fixture
    def buffer(self):
        """Create a StreamingBuffer instance"""
        return StreamingBuffer(update_interval=2.0, buffer_size_threshold=500, min_update_interval=1.0)
    
    def test_initialization(self, buffer):
        """Test buffer initialization"""
        assert buffer.update_interval == 2.0
        assert buffer.buffer_size_threshold == 500
        assert buffer.min_update_interval == 1.0
        assert buffer.accumulated_text == ""
        assert buffer.last_update_time == 0.0
    
    def test_reset(self, buffer):
        """Test resetting buffer"""
        buffer.add_chunk("test")
        buffer.reset()
        assert buffer.accumulated_text == ""
        assert buffer.last_update_time == 0.0
        assert buffer.last_sent_text == ""
    
    def test_add_chunk(self, buffer):
        """Test adding text chunks"""
        buffer.add_chunk("hello")
        assert buffer.accumulated_text == "hello"
        
        buffer.add_chunk(" world")
        assert buffer.accumulated_text == "hello world"
    
    def test_add_empty_chunk(self, buffer):
        """Test adding empty chunk"""
        buffer.add_chunk("")
        assert buffer.accumulated_text == ""
        
        buffer.add_chunk("test")
        buffer.add_chunk("")
        assert buffer.accumulated_text == "test"
    
    @patch('time.time')
    def test_should_update_time_based(self, mock_time, buffer):
        """Test time-based update triggering"""
        mock_time.return_value = 0.0
        buffer.mark_updated()
        
        # Not enough time passed
        mock_time.return_value = 0.5
        assert not buffer.should_update()
        
        # Minimum interval passed but not update interval
        mock_time.return_value = 1.5
        assert not buffer.should_update()
        
        # Update interval passed
        mock_time.return_value = 2.5
        assert buffer.should_update()
    
    @patch('time.time')
    def test_should_update_size_threshold(self, mock_time, buffer):
        """Test size-based update triggering"""
        mock_time.return_value = 0.0
        buffer.mark_updated()
        
        # Add large chunk exceeding threshold
        mock_time.return_value = 1.5  # After min interval
        buffer.add_chunk("x" * 600)
        assert buffer.should_update()
    
    @patch('time.time')
    def test_should_update_min_interval(self, mock_time, buffer):
        """Test minimum interval enforcement"""
        mock_time.return_value = 0.0
        buffer.mark_updated()
        
        # Even with large buffer, should respect min interval
        buffer.add_chunk("x" * 600)
        mock_time.return_value = 0.5  # Before min interval
        assert not buffer.should_update()
    
    def test_get_display_text(self, buffer):
        """Test getting display-safe text"""
        buffer.add_chunk("```python\ncode")
        display_text = buffer.get_display_text()
        # Should have fence handler applied
        assert display_text.count("```") == 2
    
    def test_get_complete_text(self, buffer):
        """Test getting raw accumulated text"""
        text = "```python\ncode"
        buffer.add_chunk(text)
        assert buffer.get_complete_text() == text
    
    @patch('time.time')
    def test_mark_updated(self, mock_time, buffer):
        """Test marking buffer as updated"""
        mock_time.return_value = 5.0
        buffer.add_chunk("test")
        buffer.mark_updated()
        
        assert buffer.last_update_time == 5.0
        assert buffer.last_sent_text == "test"
    
    def test_get_stats(self, buffer):
        """Test getting buffer statistics"""
        buffer.add_chunk("```python\ntest")
        stats = buffer.get_stats()
        
        assert stats["text_length"] == len("```python\ntest")
        assert "time_since_last_update" in stats
        assert stats["update_interval"] == 2.0
        assert stats["unclosed_triple_fences"] == 1
        assert stats["unclosed_single_fences"] == 0
    
    def test_update_interval_setting(self, buffer):
        """Test updating the interval setting"""
        buffer.update_interval_setting(3.0)
        assert buffer.update_interval == 3.0
        
        # Should respect minimum
        buffer.update_interval_setting(0.5)
        assert buffer.update_interval == 1.0  # min_interval
    
    def test_has_content(self, buffer):
        """Test checking for content"""
        assert not buffer.has_content()
        
        buffer.add_chunk("test")
        assert buffer.has_content()
        
        buffer.reset()
        assert not buffer.has_content()
    
    def test_has_pending_update(self, buffer):
        """Test checking for pending updates"""
        assert not buffer.has_pending_update()
        
        buffer.add_chunk("test")
        assert buffer.has_pending_update()
        
        buffer.mark_updated()
        assert not buffer.has_pending_update()
        
        buffer.add_chunk(" more")
        assert buffer.has_pending_update()
    
    @pytest.mark.critical
    def test_critical_buffer_flow(self, buffer):
        """Critical test for buffer accumulation and update flow"""
        # Add chunks
        buffer.add_chunk("Hello")
        assert buffer.has_content()
        assert buffer.has_pending_update()
        
        # Mark as sent
        buffer.mark_updated()
        assert not buffer.has_pending_update()
        
        # Add more
        buffer.add_chunk(" World")
        assert buffer.has_pending_update()
        assert buffer.get_complete_text() == "Hello World"


class TestRateLimitManager:
    """Test RateLimitManager class"""
    
    @pytest.fixture
    def manager(self):
        """Create a RateLimitManager instance"""
        return RateLimitManager(
            base_interval=2.0,
            min_interval=1.0,
            max_interval=30.0,
            failure_threshold=5,
            cooldown_seconds=300,
            success_reduction_factor=0.9
        )
    
    def test_initialization(self, manager):
        """Test manager initialization"""
        assert manager.base_interval == 2.0
        assert manager.current_interval == 2.0
        assert manager.circuit_state == CircuitState.CLOSED
        assert manager.consecutive_failures == 0
        assert manager.total_requests == 0
    
    def test_can_make_request_normal(self, manager):
        """Test request permission in normal state"""
        assert manager.can_make_request()
    
    @patch('time.time')
    def test_can_make_request_circuit_open(self, mock_time, manager):
        """Test request permission with open circuit"""
        mock_time.return_value = 0.0
        
        # Trip the circuit
        for _ in range(5):
            manager.record_failure()
        
        assert manager.circuit_state == CircuitState.OPEN
        assert not manager.can_make_request()
        
        # After cooldown, should move to half-open
        mock_time.return_value = 301.0
        assert manager.can_make_request()
        assert manager.circuit_state == CircuitState.HALF_OPEN
    
    @patch('time.time')
    def test_can_make_request_retry_after(self, mock_time, manager):
        """Test request permission with retry-after"""
        mock_time.return_value = 0.0
        manager.set_retry_after(10.0)
        
        # During retry-after period
        mock_time.return_value = 5.0
        assert not manager.can_make_request()
        
        # After retry-after period
        mock_time.return_value = 11.0
        assert manager.can_make_request()
    
    def test_record_success(self, manager):
        """Test recording successful request"""
        manager.record_request_attempt()
        manager.record_success()
        
        assert manager.successful_requests == 1
        assert manager.consecutive_failures == 0
        assert manager.consecutive_successes == 1
    
    def test_record_success_closes_circuit(self, manager):
        """Test success closes open circuit"""
        # Open the circuit
        for _ in range(5):
            manager.record_failure()
        
        assert manager.circuit_state == CircuitState.OPEN
        
        # Success should close it
        manager.record_success()
        assert manager.circuit_state == CircuitState.CLOSED
    
    def test_record_success_reduces_interval(self, manager):
        """Test sustained success reduces interval"""
        manager.current_interval = 4.0
        
        # Need 3 consecutive successes
        for _ in range(3):
            manager.record_success()
        
        assert manager.current_interval < 4.0
        assert manager.current_interval >= manager.base_interval
    
    def test_record_failure(self, manager):
        """Test recording failed request"""
        manager.record_failure()
        
        assert manager.consecutive_failures == 1
        assert manager.consecutive_successes == 0
        assert manager.current_interval > manager.base_interval
    
    def test_record_failure_exponential_backoff(self, manager):
        """Test exponential backoff on failures"""
        initial = manager.current_interval
        
        manager.record_failure()
        first_increase = manager.current_interval
        assert first_increase == initial * 2
        
        manager.record_failure()
        second_increase = manager.current_interval
        assert second_increase == first_increase * 2
    
    def test_record_failure_max_interval(self, manager):
        """Test interval doesn't exceed maximum"""
        for _ in range(10):
            manager.record_failure()
        
        assert manager.current_interval == manager.max_interval
    
    def test_record_failure_rate_limit(self, manager):
        """Test recording rate limit failure"""
        manager.record_failure(is_rate_limit=True)
        
        assert manager.rate_limited_requests == 1
        assert manager.last_429_time is not None
    
    def test_circuit_breaker_trip(self, manager):
        """Test circuit breaker trips after threshold"""
        # Record failures up to threshold
        for _ in range(4):
            manager.record_failure()
            assert manager.circuit_state == CircuitState.CLOSED
        
        # One more should trip it
        manager.record_failure()
        assert manager.circuit_state == CircuitState.OPEN
        assert manager.circuit_trips == 1
    
    @patch('time.time')
    def test_set_retry_after(self, mock_time, manager):
        """Test setting retry-after from 429 response"""
        mock_time.return_value = 0.0
        manager.set_retry_after(15.0)
        
        assert manager.retry_after == 15.0
        
        # Should also update interval if larger
        assert manager.current_interval >= 15.0
    
    def test_get_current_interval(self, manager):
        """Test getting current interval"""
        assert manager.get_current_interval() == 2.0
        
        manager.record_failure()
        assert manager.get_current_interval() > 2.0
    
    def test_is_streaming_enabled(self, manager):
        """Test checking if streaming is enabled"""
        assert manager.is_streaming_enabled()
        
        # Trip circuit
        for _ in range(5):
            manager.record_failure()
        
        assert not manager.is_streaming_enabled()
    
    @patch('time.time')
    def test_get_time_until_retry(self, mock_time, manager):
        """Test getting time until retry"""
        mock_time.return_value = 0.0
        
        # No restrictions
        assert manager.get_time_until_retry() is None
        
        # With retry-after
        manager.set_retry_after(10.0)
        mock_time.return_value = 3.0
        assert manager.get_time_until_retry() == 7.0
        
        # With circuit breaker
        manager.retry_after = None
        for _ in range(5):
            manager.record_failure()
        
        mock_time.return_value = 100.0
        remaining = manager.get_time_until_retry()
        assert remaining is not None
        assert remaining > 0
    
    def test_get_stats(self, manager):
        """Test getting statistics"""
        manager.record_request_attempt()
        manager.record_success()
        manager.record_request_attempt()
        manager.record_failure(is_rate_limit=True)
        
        stats = manager.get_stats()
        
        assert stats["total_requests"] == 2
        assert stats["successful_requests"] == 1
        assert stats["rate_limited_requests"] == 1
        assert stats["success_rate_percent"] == 50.0
        assert stats["circuit_state"] == "closed"
        assert stats["streaming_enabled"] is True
    
    def test_reset_stats(self, manager):
        """Test resetting statistics"""
        manager.record_request_attempt()
        manager.record_success()
        manager.reset_stats()
        
        assert manager.total_requests == 0
        assert manager.successful_requests == 0
        # Current state should be preserved
        assert manager.current_interval == manager.base_interval
    
    def test_force_reset(self, manager):
        """Test force reset"""
        # Mess up the state
        for _ in range(5):
            manager.record_failure()
        
        manager.force_reset()
        
        assert manager.current_interval == manager.base_interval
        assert manager.consecutive_failures == 0
        assert manager.circuit_state == CircuitState.CLOSED
        assert manager.retry_after is None
    
    @pytest.mark.critical
    def test_critical_circuit_breaker_flow(self, manager):
        """Critical test for circuit breaker flow"""
        # Normal operation
        assert manager.circuit_state == CircuitState.CLOSED
        assert manager.can_make_request()
        
        # Accumulate failures
        for i in range(4):
            manager.record_failure()
            assert manager.circuit_state == CircuitState.CLOSED
        
        # Trip circuit
        manager.record_failure()
        assert manager.circuit_state == CircuitState.OPEN
        assert not manager.can_make_request()
        
        # Recovery with success
        manager.record_success()
        assert manager.circuit_state == CircuitState.CLOSED
        assert manager.can_make_request()
    
    @pytest.mark.smoke
    def test_smoke_basic_operations(self, manager):
        """Smoke test for basic operations"""
        assert manager.can_make_request()
        manager.record_request_attempt()
        manager.record_success()
        assert manager.is_streaming_enabled()
        stats = manager.get_stats()
        assert stats["total_requests"] == 1
        assert stats["successful_requests"] == 1


class TestStreamingIntegration:
    """Integration tests for streaming components"""
    
    @pytest.mark.integration
    def test_buffer_with_fence_handler(self):
        """Test buffer integration with fence handler"""
        buffer = StreamingBuffer()
        
        # Add unclosed code block
        buffer.add_chunk("```python\ndef hello():\n    print('world')")
        
        # Display text should be safe
        display = buffer.get_display_text()
        assert display.count("```") == 2
        
        # Complete text should be unchanged
        complete = buffer.get_complete_text()
        assert complete.count("```") == 1
    
    @pytest.mark.integration
    @patch('time.time')
    def test_buffer_update_timing(self, mock_time):
        """Test buffer update timing logic"""
        buffer = StreamingBuffer(update_interval=2.0, min_update_interval=1.0)
        
        mock_time.return_value = 0.0
        buffer.mark_updated()
        
        # Add content
        buffer.add_chunk("test")
        
        # Too soon
        mock_time.return_value = 0.5
        assert not buffer.should_update()
        
        # After min but before regular
        mock_time.return_value = 1.5
        assert not buffer.should_update()
        
        # After regular interval
        mock_time.return_value = 2.5
        assert buffer.should_update()
    
    @pytest.mark.critical
    def test_critical_streaming_components_interface(self):
        """Critical test that streaming components maintain expected interfaces"""
        # StreamingBuffer
        buffer = StreamingBuffer()
        assert hasattr(buffer, 'add_chunk')
        assert hasattr(buffer, 'should_update')
        assert hasattr(buffer, 'get_display_text')
        assert hasattr(buffer, 'reset')
        
        # FenceHandler
        fence = FenceHandler()
        assert hasattr(fence, 'update_text')
        assert hasattr(fence, 'get_display_safe_text')
        assert hasattr(fence, 'is_in_code_block')
        
        # RateLimitManager
        rate_limit = RateLimitManager()
        assert hasattr(rate_limit, 'can_make_request')
        assert hasattr(rate_limit, 'record_success')
        assert hasattr(rate_limit, 'record_failure')
        assert hasattr(rate_limit, 'is_streaming_enabled')