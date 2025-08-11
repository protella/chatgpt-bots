"""
FenceHandler class for managing markdown code fence safety during streaming
Handles closing unclosed triple and single backticks for display safety
"""

import re
from typing import Optional
from logger import LoggerMixin


class FenceHandler(LoggerMixin):
    """
    Handles markdown fence closing logic for streaming text display
    Ensures code blocks appear correctly even when incomplete
    """
    
    def __init__(self):
        """Initialize the fence handler"""
        self.current_text = ""
        self.log_debug("FenceHandler initialized")
    
    def reset(self) -> None:
        """Reset the handler state"""
        self.current_text = ""
        self.log_debug("FenceHandler reset")
    
    def update_text(self, text: str) -> None:
        """
        Update the text being tracked
        
        Args:
            text: Current accumulated text
        """
        self.current_text = text
    
    def get_display_safe_text(self) -> str:
        """
        Get text with temporary closing fences added for safe display
        
        Returns:
            Text with appropriate closing fences
        """
        if not self.current_text:
            return ""
        
        # Get the raw text and analyze fences
        text = self.current_text
        
        # Handle triple backticks first (code blocks)
        text = self._close_triple_backticks(text)
        
        # Then handle single backticks (inline code)
        text = self._close_single_backticks(text)
        
        return text
    
    def _close_triple_backticks(self, text: str) -> str:
        """
        Close unclosed triple backtick code blocks
        
        Args:
            text: Text to process
            
        Returns:
            Text with closed triple backtick blocks
        """
        # Find all triple backtick occurrences with optional language hints
        triple_pattern = r'```(\w+)?'
        matches = list(re.finditer(triple_pattern, text))
        
        if len(matches) % 2 == 1:
            # Odd number means we have an unclosed block
            last_match = matches[-1]
            language = last_match.group(1) if last_match.group(1) else ""
            
            # Check if we're in the middle of a line or if the text ends abruptly
            # Add closing fence with a newline for proper formatting
            if text.endswith('\n'):
                text += "```"
            else:
                text += "\n```"
            
            self.log_debug(f"Closed unclosed triple backtick block (language: {language or 'none'})")
        
        return text
    
    def _close_single_backticks(self, text: str) -> str:
        """
        Close unclosed single backtick inline code
        
        Args:
            text: Text to process
            
        Returns:
            Text with closed single backtick code
        """
        # Count single backticks that aren't part of triple backticks
        # This is more complex because we need to ignore backticks in triple-backtick blocks
        
        # First, temporarily replace triple backticks to avoid counting them
        temp_text = re.sub(r'```.*?```', lambda m: '█' * len(m.group()), text, flags=re.DOTALL)
        
        # Also handle unclosed triple backtick blocks (they were closed in previous step)
        temp_text = re.sub(r'```[^\n]*(?:\n.*)?$', lambda m: '█' * len(m.group()), temp_text, flags=re.DOTALL)
        
        # Count single backticks
        single_backticks = temp_text.count('`')
        
        if single_backticks % 2 == 1:
            # Odd number means unclosed inline code
            text += "`"
            self.log_debug("Closed unclosed single backtick inline code")
        
        return text
    
    def get_unclosed_triple_count(self) -> int:
        """
        Get count of unclosed triple backtick blocks
        
        Returns:
            Number of unclosed triple backtick blocks
        """
        matches = re.findall(r'```', self.current_text)
        return len(matches) % 2
    
    def get_unclosed_single_count(self) -> int:
        """
        Get count of unclosed single backticks (outside of triple backtick blocks)
        
        Returns:
            Number of unclosed single backticks
        """
        # Remove triple backtick blocks first
        temp_text = re.sub(r'```.*?```', '', self.current_text, flags=re.DOTALL)
        # Remove unclosed triple backtick blocks
        temp_text = re.sub(r'```.*$', '', temp_text, flags=re.DOTALL)
        
        single_backticks = temp_text.count('`')
        return single_backticks % 2
    
    def is_in_code_block(self, position: Optional[int] = None) -> bool:
        """
        Check if the given position (or end of text) is inside a code block
        
        Args:
            position: Position to check, defaults to end of text
            
        Returns:
            True if position is inside a code block
        """
        if position is None:
            position = len(self.current_text)
        
        text_up_to_position = self.current_text[:position]
        
        # Count triple backticks before this position
        triple_matches = re.findall(r'```', text_up_to_position)
        
        # If odd number, we're inside a code block
        return len(triple_matches) % 2 == 1
    
    def get_current_language_hint(self) -> Optional[str]:
        """
        Get the language hint for the current unclosed code block
        
        Returns:
            Language hint string or None if not in a code block
        """
        if not self.is_in_code_block():
            return None
        
        # Find the last triple backtick with language hint
        matches = list(re.finditer(r'```(\w+)?', self.current_text))
        
        if matches and len(matches) % 2 == 1:
            # We're in an unclosed block
            language = matches[-1].group(1)
            return language if language else None
        
        return None
    
    def analyze_fences(self) -> dict:
        """
        Analyze the current fence state
        
        Returns:
            Dictionary with fence analysis
        """
        return {
            "unclosed_triple_fences": self.get_unclosed_triple_count(),
            "unclosed_single_fences": self.get_unclosed_single_count(),
            "in_code_block": self.is_in_code_block(),
            "current_language": self.get_current_language_hint(),
            "text_length": len(self.current_text)
        }