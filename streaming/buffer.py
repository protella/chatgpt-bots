"""
StreamingBuffer class for accumulating and managing text chunks during streaming
Provides time-based update triggering and display-safe text with fence closing
"""

import time
from typing import Optional
from logger import LoggerMixin
from .fence_handler import FenceHandler


class StreamingBuffer(LoggerMixin):
    """
    Accumulates text chunks and determines when to trigger updates
    Handles markdown fence closing for safe display of incomplete code blocks
    """
    
    def __init__(
        self,
        update_interval: float = 2.0,
        buffer_size_threshold: int = 500,
        min_update_interval: float = 1.0
    ):
        """
        Initialize the streaming buffer
        
        Args:
            update_interval: Base interval between updates in seconds
            buffer_size_threshold: Character count that triggers forced update
            min_update_interval: Minimum time between updates (rate limit floor)
        """
        self.update_interval = update_interval
        self.buffer_size_threshold = buffer_size_threshold
        self.min_update_interval = min_update_interval
        
        self.fence_handler = FenceHandler()
        self.reset()
        
        self.log_debug(f"StreamingBuffer initialized with {update_interval}s interval")
    
    def reset(self):
        """Reset the buffer state"""
        self.accumulated_text = ""
        self.last_update_time = 0.0
        self.fence_handler.reset()
        self.last_sent_text = ""  # Track what was last sent
        self.log_debug("StreamingBuffer reset")
    
    def add_chunk(self, text: str) -> None:
        """
        Add a text chunk to the buffer
        
        Args:
            text: Text chunk to add
        """
        if not text:
            return
            
        self.accumulated_text += text
        self.fence_handler.update_text(self.accumulated_text)
        
        # Only log chunk additions periodically or for significant chunks
        if len(text) > 100 or len(self.accumulated_text) % 500 == 0:
            self.log_debug(f"Added chunk: {len(text)} chars (total: {len(self.accumulated_text)} chars)")
    
    def should_update(self) -> bool:
        """
        Determine if an update should be triggered
        
        Returns:
            True if update should happen now
        """
        current_time = time.time()
        time_elapsed = current_time - self.last_update_time
        
        # Check minimum interval (rate limit floor)
        if time_elapsed < self.min_update_interval:
            return False
        
        # Force update if buffer is getting large
        if len(self.accumulated_text) >= self.buffer_size_threshold:
            return True
        
        # Regular time-based update
        if time_elapsed >= self.update_interval:
            return True
        
        return False
    
    def get_display_text(self) -> str:
        """
        Get text with closed fences for safe display
        
        Returns:
            Text with temporary closing fences added
        """
        return self.fence_handler.get_display_safe_text()
    
    def get_complete_text(self) -> str:
        """
        Get the accumulated text as-is
        
        Returns:
            Raw accumulated text without modifications
        """
        return self.accumulated_text
    
    def mark_updated(self) -> None:
        """Mark that an update was performed"""
        self.last_update_time = time.time()
        self.last_sent_text = self.accumulated_text  # Remember what we sent
    
    def get_stats(self) -> dict:
        """
        Get buffer statistics
        
        Returns:
            Dictionary with buffer stats
        """
        current_time = time.time()
        return {
            "text_length": len(self.accumulated_text),
            "time_since_last_update": current_time - self.last_update_time,
            "update_interval": self.update_interval,
            "unclosed_triple_fences": self.fence_handler.get_unclosed_triple_count(),
            "unclosed_single_fences": self.fence_handler.get_unclosed_single_count(),
        }
    
    def update_interval_setting(self, new_interval: float) -> None:
        """
        Update the update interval (for rate limiting adjustments)
        
        Args:
            new_interval: New interval between updates
        """
        old_interval = self.update_interval
        self.update_interval = max(new_interval, self.min_update_interval)
        
        if abs(old_interval - self.update_interval) > 0.1:
            self.log_info(f"Update interval changed from {old_interval}s to {self.update_interval}s")
    
    def has_content(self) -> bool:
        """Check if buffer has accumulated content"""
        return len(self.accumulated_text) > 0
    
    def has_pending_update(self) -> bool:
        """Check if there's text that hasn't been sent yet"""
        return self.accumulated_text != self.last_sent_text