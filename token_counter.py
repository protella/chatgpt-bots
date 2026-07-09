"""
Token estimation utility for managing thread context sizes.

Phase S usage-driven budgeting: tiktoken is GONE. The authoritative context size
comes from the API's own `response.usage` after every call (tracked on ThreadState);
this module only provides the cheap chars/4 ESTIMATE used between calls and for
cold-rebuild pre-flight decisions. With TOKEN_BUFFER_PERCENTAGE headroom, crude is
fine — and the context_length_exceeded backstop (compact + retry once) guards the
estimator's edge cases.

The TokenCounter interface (count_tokens / count_message_tokens /
count_thread_tokens / trim_thread_to_limit / estimate_remaining_tokens) is kept so
call sites are unchanged.
"""
from typing import List, Dict, Any, Tuple
from logger import LoggerMixin


def estimate_tokens(text: str) -> int:
    """Crude token estimate: ~1 token per 4 characters."""
    if not text:
        return 0
    return len(text) // 4


class TokenCounter(LoggerMixin):
    """Estimates token counts for threads (chars/4 — no tokenizer dependency)."""

    def __init__(self, model: str = "gpt-4"):
        """
        Args:
            model: Accepted for signature compatibility; the estimate is model-agnostic.
        """
        self.model = model

    def count_tokens(self, text: str) -> int:
        """Estimate tokens in a text string (chars/4)."""
        if not text:
            return 0
        return estimate_tokens(str(text))

    def count_message_tokens(self, message: Dict[str, Any]) -> int:
        """
        Estimate tokens in a message dict, including a small structure overhead.

        Base64 image parts are NOT counted — images go to the vision API, and the
        text conversation only carries breadcrumbs.
        """
        overhead = 4
        tokens = overhead

        role = message.get("role", "")
        if role:
            tokens += self.count_tokens(role)

        content = message.get("content", "")
        if content:
            if isinstance(content, str):
                tokens += self.count_tokens(content)
            elif isinstance(content, list):
                # Multi-part content: count text parts only, skip image data
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "input_text":
                            tokens += self.count_tokens(part.get("text", ""))
                        # Skip input_image parts
                    else:
                        tokens += self.count_tokens(str(part))
            else:
                tokens += self.count_tokens(str(content))

        return tokens

    def count_thread_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate total tokens in a thread."""
        total = 0
        for message in messages:
            total += self.count_message_tokens(message)
        total += 3  # conversation structure overhead
        return total

    def trim_thread_to_limit(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        preserve_system: bool = True
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Trim messages from the beginning of thread to fit within the (estimated) limit.

        Returns:
            Tuple of (trimmed messages list, number of messages removed)
        """
        if not messages:
            return messages, 0

        current_tokens = self.count_thread_tokens(messages)

        if current_tokens <= max_tokens:
            return messages, 0

        trimmed = messages.copy()
        removed_count = 0

        start_index = 0
        if preserve_system:
            for i, msg in enumerate(trimmed):
                if msg.get("role") != "system" and msg.get("role") != "developer":
                    start_index = i
                    break

        while current_tokens > max_tokens and len(trimmed) > start_index + 1:
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
        """Estimate how many tokens are remaining in the context window."""
        current = self.count_thread_tokens(messages)
        remaining = max_tokens - current
        return max(0, remaining)
