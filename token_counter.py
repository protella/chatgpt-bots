"""
Token counting utility for managing thread context sizes
Uses tiktoken to count tokens and manage context windows
"""
import tiktoken
from typing import List, Dict, Any, Optional, Tuple
from logger import LoggerMixin


class TokenCounter(LoggerMixin):
    """Manages token counting for threads to prevent context limit errors"""
    
    def __init__(self, model: str = "gpt-4"):
        """
        Initialize token counter with specified model
        
        Args:
            model: Model name for encoding (defaults to gpt-4 as fallback)
        """
        self.model = model
        self._encoder = None
        self._init_encoder()
    
    def _init_encoder(self):
        """Initialize the tiktoken encoder for the model"""
        try:
            # Try to get encoder for the specific model
            if self.model.startswith("gpt-5"):
                # GPT-5 models likely use the same encoding as GPT-4
                # Fallback to cl100k_base which is used by GPT-4
                self._encoder = tiktoken.get_encoding("cl100k_base")
                self.log_debug(f"Using cl100k_base encoding for {self.model}")
            else:
                # Try to get model-specific encoding
                try:
                    self._encoder = tiktoken.encoding_for_model(self.model)
                    self.log_debug(f"Using model-specific encoding for {self.model}")
                except KeyError:
                    # Fallback to cl100k_base for unknown models
                    self._encoder = tiktoken.get_encoding("cl100k_base")
                    self.log_debug(f"Model {self.model} not found, using cl100k_base encoding")
        except Exception as e:
            self.log_error(f"Failed to initialize tiktoken encoder: {e}")
            # Create a simple fallback that estimates tokens
            self._encoder = None
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in a text string
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
            
        if self._encoder:
            try:
                return len(self._encoder.encode(text))
            except Exception as e:
                self.log_warning(f"Failed to count tokens with encoder: {e}")
                # Fallback to estimation
                return self._estimate_tokens(text)
        else:
            # Fallback estimation if encoder not available
            return self._estimate_tokens(text)
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count when encoder is not available
        Rough estimate: ~1 token per 4 characters
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        return len(text) // 4
    
    def count_message_tokens(self, message: Dict[str, Any]) -> int:
        """
        Count tokens in a message dict
        
        Args:
            message: Message dictionary with role and content
            
        Returns:
            Number of tokens including formatting overhead
        """
        # Account for message structure overhead (role, separators, etc)
        # Typically adds 3-4 tokens per message
        overhead = 4
        
        tokens = overhead
        
        # Count content tokens
        content = message.get("content", "")
        if content:
            # Check if content contains image data (base64 URLs)
            if isinstance(content, str):
                # Look for base64 image data URLs
                if "data:image" in content and ";base64," in content:
                    # Extract and count image tokens
                    # For simplicity, use high detail estimate: 170 base + 170 per tile
                    # Assuming average image is ~1024x1024 = 4 tiles
                    image_tokens = 170 + (4 * 170)  # ~850 tokens per image
                    tokens += image_tokens
                    
                    # Also count any text around the image
                    text_parts = content.split("data:image")
                    for part in text_parts:
                        if ";base64," not in part:
                            tokens += self.count_tokens(part)
                else:
                    tokens += self.count_tokens(str(content))
            else:
                tokens += self.count_tokens(str(content))
        
        # Count role tokens
        role = message.get("role", "")
        if role:
            tokens += self.count_tokens(role)
        
        # Check for metadata that indicates images
        metadata = message.get("metadata", {})
        if metadata:
            # Check if this is an image generation or vision message
            msg_type = metadata.get("type", "")
            if msg_type in ["image_generation", "image_edit", "vision_analysis"]:
                # Add estimated tokens for image processing
                # These messages often involve images but don't store the base64 in content
                if msg_type == "vision_analysis":
                    # Vision analysis typically processes 1+ images
                    image_count = metadata.get("image_count", 1)
                    tokens += 850 * image_count  # High detail estimate per image
            
            # Don't count other metadata as it's not sent to API
        
        return tokens
    
    def count_thread_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        Count total tokens in a thread
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Total number of tokens in the thread
        """
        total = 0
        for message in messages:
            total += self.count_message_tokens(message)
        
        # Add some overhead for conversation structure
        total += 3
        
        return total
    
    def trim_thread_to_limit(
        self, 
        messages: List[Dict[str, Any]], 
        max_tokens: int,
        preserve_system: bool = True
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Trim messages from the beginning of thread to fit within token limit
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum token limit
            preserve_system: If True, never remove system messages
            
        Returns:
            Tuple of (trimmed messages list, number of messages removed)
        """
        if not messages:
            return messages, 0
        
        current_tokens = self.count_thread_tokens(messages)
        
        if current_tokens <= max_tokens:
            return messages, 0
        
        # Create a copy to work with
        trimmed = messages.copy()
        removed_count = 0
        
        # Find first non-system message index if preserving system
        start_index = 0
        if preserve_system:
            for i, msg in enumerate(trimmed):
                if msg.get("role") != "system" and msg.get("role") != "developer":
                    start_index = i
                    break
        
        # Remove messages from the beginning (after system message if preserved)
        while current_tokens > max_tokens and len(trimmed) > start_index + 1:
            # Don't remove the last message (current user input)
            if start_index < len(trimmed) - 1:
                removed_msg = trimmed.pop(start_index)
                removed_count += 1
                current_tokens = self.count_thread_tokens(trimmed)
                self.log_info(f"Removed message from thread to fit token limit. Role: {removed_msg.get('role')}, "
                          f"Content preview: {str(removed_msg.get('content', ''))[:50]}...")
            else:
                self.log_warning(f"Cannot trim thread further - would remove current message. "
                             f"Current tokens: {current_tokens}, limit: {max_tokens}")
                break
        
        if current_tokens > max_tokens:
            self.log_warning(f"Thread still exceeds token limit after trimming. "
                         f"Current: {current_tokens}, limit: {max_tokens}")
        
        return trimmed, removed_count
    
    def estimate_remaining_tokens(
        self, 
        messages: List[Dict[str, Any]], 
        max_tokens: int
    ) -> int:
        """
        Estimate how many tokens are remaining in the context window
        
        Args:
            messages: List of message dictionaries
            max_tokens: Maximum token limit
            
        Returns:
            Number of tokens remaining
        """
        current = self.count_thread_tokens(messages)
        remaining = max_tokens - current
        return max(0, remaining)