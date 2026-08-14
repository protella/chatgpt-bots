"""
Token estimation utility for managing thread context sizes.

Phase S usage-driven budgeting: the authoritative context size comes from the API's own
`response.usage` after every call (tracked on ThreadState). Between calls, THREE questions get
three different instruments, because the cost of being wrong is different in each.

`estimate_tokens` (chars/4) answers "roughly how big is this thread" for DM trimming and
compaction, where being wrong costs one extra compaction. No tokenizer, deliberately — the DM
thresholds are calibrated against this average and swapping it would move every one of them.

`admission_charge` answers "may this channel request be sent at all". That decision has to HOLD, so
it is not an estimate: every OpenAI tokenizer since GPT-2 is byte-level BPE, and a byte-level BPE
token consumes at least one input byte, so tokens ≤ utf-8 bytes for any such tokenizer — including
gpt-5.6's, which is not published. Charging one token per byte is therefore an upper bound that
needs no vocabulary, which is why the "what if tiktoken cannot load" question does not arise for
admission at all. It over-charges English prose about 4.5x; that is the price of a bound rather
than a hope, and the cost of the alternative is a turn that pays for its document summaries and
THEN fails inside the API.

`estimate_tokens_conservative` answers "how big is this text REALLY", and nothing is decided from
it. Document reserves are the byte charge too (a reserve derived from a count could not cover what
was charged), so this counter's one production caller is the refusal diagnostic: when a channel
request is refused, the real o200k count is logged beside the charged bound, which is what tells an
operator "this window needs compacting" from "the bound refused a window that would have fit".
Accuracy instead of certainty, since being wrong here changes a log line: the o200k count plus
headroom for the merge table we cannot see.

The TokenCounter interface (count_tokens / count_message_tokens /
count_thread_tokens / trim_thread_to_limit / estimate_remaining_tokens) is kept so
call sites are unchanged.
"""
import math
import threading
from typing import List, Dict, Any, Optional, Tuple
from logger import LoggerMixin, setup_logger

logger = setup_logger(name="slack_bot.TokenCounter")

# The diagnostic counter's tokenizer. gpt-5.6's own is not published; o200k_base is the current
# public OpenAI encoding and the closest available proxy. Measured against it, the content that
# actually breaks a budget looks nothing like the chars/4 average: a hex digest runs 1.0 utf-8
# bytes per token, base64 1.7, dense CSV 1.4, minified JSON 1.9 — against 4.5 for English prose.
_ADMISSION_ENCODING = "o200k_base"

# Multiplied onto the exact o200k count, for the merge table we cannot see. The unknown is the
# table, not the alphabet, so a different vocabulary shifts counts without changing their order of
# magnitude. Accuracy insurance on a diagnostic, not a bound — bounds are `admission_charge`'s job,
# and nothing that must hold is decided from this number.
_TOKENIZER_HEADROOM = 1.15

# Charged per input item on top of its text: the role wrapper and message delimiters the API adds
# around content we never see. The Responses API does not document its framing; 8 is comfortably
# above the 3-4 tokens per message the older published accounting recipe charged, and an item is
# cheap to over-charge — a 60-item request pays 480 tokens for the guarantee.
ITEM_STRUCTURAL_OVERHEAD = 8

# The tokenizer is loaded once per process, in a daemon thread, because `get_encoding` fetches the
# vocabulary over the network on a cold cache. Nothing may wait on that: a blackholed egress would
# wedge whatever asked for the count. So a caller that finds a load in flight — its own, or boot's —
# gives it a grace long enough for a WARM cache (measured ~120ms) and otherwise proceeds without it,
# and the encoder is picked up by whichever call comes after it lands.
#
# A FAILED load is retryable, not final: the usual cause is a cold cache with momentarily no
# network, which the next turn may well not hit. Retries are counted, because a permanently
# offline box must not pay the grace on every request forever.
_ENCODER_GRACE_SECONDS = 1.0
_ENCODER_MAX_ATTEMPTS = 3
_ENCODER_LOCK = threading.Lock()
_ENCODER_SLOT: Dict[str, Any] = {}
_ENCODER_THREAD: Optional[threading.Thread] = None
_ENCODER_ATTEMPTS = 0


def _load_admission_encoder() -> None:
    global _ENCODER_THREAD
    encoder = None
    try:
        import tiktoken
        encoder = tiktoken.get_encoding(_ADMISSION_ENCODING)
    except Exception as exc:  # noqa: BLE001 — no tokenizer is a coarser log line, not an outage
        logger.warning(f"Diagnostic tokenizer {_ADMISSION_ENCODING} unavailable ({exc}); a refused "
                       f"request reports the byte bound alone until a retry lands")
    with _ENCODER_LOCK:
        if encoder is not None:
            _ENCODER_SLOT["encoder"] = encoder
        # Cleared either way: on success nothing will ever look again, and on failure the slot
        # stays empty so a later caller can start attempt N+1.
        _ENCODER_THREAD = None


def _start_encoder_load() -> Optional[threading.Thread]:
    """The thread currently loading the encoder, starting one if nobody has.

    Returns whatever is in flight — NOT only a thread this call started. Boot kicks the load off
    without waiting, and a first turn that arrived while that was still running used to skip the
    grace entirely and go straight to the fallback.
    """
    global _ENCODER_THREAD, _ENCODER_ATTEMPTS
    with _ENCODER_LOCK:
        if "encoder" in _ENCODER_SLOT:
            return None
        if _ENCODER_THREAD is not None:
            return _ENCODER_THREAD
        if _ENCODER_ATTEMPTS >= _ENCODER_MAX_ATTEMPTS:
            return None
        _ENCODER_ATTEMPTS += 1
        _ENCODER_THREAD = threading.Thread(target=_load_admission_encoder,
                                           name="admission-tokenizer", daemon=True)
        _ENCODER_THREAD.start()
        return _ENCODER_THREAD


def _admission_encoder() -> Any:
    """The o200k encoder, or None while it is still loading / until a retry lands."""
    if "encoder" in _ENCODER_SLOT:
        return _ENCODER_SLOT["encoder"]
    loading = _start_encoder_load()
    if loading is not None:
        loading.join(_ENCODER_GRACE_SECONDS)
    return _ENCODER_SLOT.get("encoder")


def wait_for_admission_encoder(timeout: float = 30.0) -> Any:
    """Wait out the load attempt in flight, and return what it produced.

    For the callers that want the exact count deterministically rather than whatever was ready in
    time — a boot warm-up (`timeout=0`, which starts the load and waits for nothing), and the tests
    that assert the counted path rather than the fallback.
    """
    if "encoder" in _ENCODER_SLOT:
        return _ENCODER_SLOT["encoder"]
    loading = _start_encoder_load()
    if loading is not None:
        loading.join(timeout)
    return _ENCODER_SLOT.get("encoder")


def estimate_tokens(text: str) -> int:
    """Crude token estimate: ~1 token per 4 characters."""
    if not text:
        return 0
    return len(text) // 4


def admission_charge(text: str) -> int:
    """The tokens this text can cost AT WORST: its utf-8 byte count.

    A byte-level BPE token consumes at least one input byte, so a byte-level BPE tokenizer cannot
    emit more tokens than the text has bytes. Every OpenAI tokenizer since GPT-2 is byte-level BPE,
    so this holds for gpt-5.6's unpublished table as surely as for a published one — and it holds
    without knowing any vocabulary at all, which is the whole reason admission is decided here and
    not from a count.
    """
    if not text:
        return 0
    return len(str(text).encode("utf-8"))


def estimate_tokens_conservative(text: str) -> int:
    """The o200k token count plus headroom: what this text probably costs in reality.

    Not an admission instrument, and not a reserve either — reserves are the byte charge. Its
    production caller is the refusal diagnostic, so being a little high or low moves a number in a
    log line, never whether a turn can be sent.

    `encode_ordinary`, not `encode`: the latter raises on text containing a special-token literal,
    and a user is entirely capable of typing `<|endoftext|>` into a channel.
    """
    if not text:
        return 0
    body = str(text)
    encoder = _admission_encoder()
    if encoder is not None:
        return math.ceil(len(encoder.encode_ordinary(body)) * _TOKENIZER_HEADROOM)
    # No vocabulary, no count. The byte bound is the honest answer, and a refusal then reports the
    # charge next to itself — which says "no real count was available", not a second opinion.
    return admission_charge(body)


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
