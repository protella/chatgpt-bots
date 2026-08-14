"""
Base Client Abstract Class
Defines the interface that all chat clients must implement
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Dict, Any, List, Optional
from dataclasses import dataclass
from logger import LoggerMixin


@dataclass
class Message:
    """Universal message format"""
    text: str
    user_id: str
    channel_id: str
    thread_id: str
    # `None` here is a placeholder __post_init__ replaces; the annotations are what every
    # reader actually gets, so they stay non-Optional.
    attachments: List[Dict[str, Any]] = None  # type: ignore[assignment]
    metadata: Dict[str, Any] = None  # type: ignore[assignment]
    
    def __post_init__(self):
        if self.attachments is None:
            self.attachments = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Response:
    """Universal response format"""
    type: str  # 'text', 'image', 'file', 'reaction', 'error', 'queued'
    content: Any
    metadata: Dict[str, Any] = None  # type: ignore[assignment]  # __post_init__ fills it

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @classmethod
    def reaction(cls, emoji: Any, target_ts: Optional[str] = None) -> "Response":
        """Build a reaction response (Phase 4).

        Args:
            emoji: an emoji name or list of names (colons optional)
            target_ts: the message timestamp to react to (defaults to the thread root)
        """
        return cls(type="reaction", content=emoji, metadata={"react_ts": target_ts} if target_ts else {})


class ChannelStreamError(Exception):
    """A turn cannot see the conversation it was asked about.

    The base of every FAIL-CLOSED context failure: the turn says honestly that it cannot see
    the room rather than answering from a partial view. It lives HERE, not beside its channel
    subclasses, because `HistoryFetchError` below is one of them and predates all of them —
    client_contract sits under message_processor in the import graph, so this is the only place a
    single hierarchy can be rooted (message_processor/channel_stream.py imports it).
    """


class HistoryFetchError(ChannelStreamError):
    """Platform history could not be fetched (e.g. persistent rate limiting).

    Distinct from an empty thread: raised so the processor can fail the turn
    loudly instead of answering with amnesia — since Phase S the platform is
    the ONLY transcript, so a failed fetch means we have no context at all.

    A `ChannelStreamError` since P2: on a channel turn a thread-activity index that has not
    caught up is the same class of failure as a missing coverage floor. DM callers catch it by
    its own name and are unaffected.
    """


def _require_receipt_class(site: str, receipts: Any, receipt_class: Optional[str]) -> None:
    """EDIT §4/§11.9/§11.13/§11.23: the ONE receipts-require-a-class guard.

    Every receipt-bearing base method body calls it, so the contract lives in the BASE, not
    only in concrete overrides: a receipted post that does not say its class is a programming
    error, refused with ValueError before any platform call — never laundered into a
    class-less (NULL, ineligible) row after the message is already in the room. MODULE-LEVEL
    deliberately: the base seams are also driven against `MagicMock` stand-ins, where a
    `self.`-resolved guard would be a mock that refuses nothing."""
    if receipts is not None and receipt_class is None:
        raise ValueError(
            f"{site}: receipts passed without receipt_class (EDIT §4/§11.9)")


class BaseClient(ABC, LoggerMixin):
    """Abstract base class for all chat clients"""

    if TYPE_CHECKING:
        # Streaming seams the concrete platform client owns (SlackBot, via its messaging
        # mixin), declared so callers holding a `BaseClient` are checked against the object
        # they actually receive. TYPE_CHECKING-only and typed as `Callable` on purpose,
        # exactly as slack_client/_host.py does it: nothing is added to the class at runtime
        # (an @abstractmethod here would newly refuse to construct any client that does not
        # implement them), and the real signatures stay the single source of truth.
        send_message_get_ts: Callable[..., Any]
        update_message_streaming: Callable[..., Any]

    def __init__(self, name: str):
        self.name = name
        self.log_info(f"{name} client initialized")

    @abstractmethod
    async def start(self):
        """Start the client and begin listening for events"""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop the client gracefully"""
        pass
    
    @abstractmethod
    async def send_message(self, channel_id: str, thread_id: str, text: str, *,
                           blocks: Optional[list] = None,
                           meta_out: Optional[dict] = None,
                           receipts: Any = None,
                           receipt_kind: Optional[str] = None,
                           receipt_class: Optional[str] = None) -> Optional[str]:
        """Send a text message. Returns the posted message ts (truthy on success) or None
        on failure.

        `blocks`: optional platform chrome (e.g. a settings-footer actions row) to attach
        to the message itself; implementations that don't support blocks may ignore the
        kwarg (it always defaults to None, so callers passing blocks=None never break a
        non-Slack impl).
        `meta_out`: optional caller-provided dict a platform MAY populate with delivery
        facts (e.g. meta_out["footer_attached"]); implementations that don't report any
        may ignore it. Both kwargs default to None so non-Slack impls stay compatible.

        Receipt contract (EDIT_OWN_MESSAGE §4/§11.9/§11.23): passing `receipts` REQUIRES
        `receipt_class` in the same call — the base seam itself enforces it (the shared
        `_require_receipt_class` guard), so a receipted post with no class raises ValueError
        even through `super()` (a programming error, never laundered into a class-less row)."""
        _require_receipt_class("send_message", receipts, receipt_class)
        return None

    @abstractmethod
    async def send_message_async(self, channel_id: str, thread_id: str, text: str, *,
                                 blocks: Optional[list] = None,
                                 meta_out: Optional[dict] = None,
                                 lease: Any = None,
                                 receipts: Any = None,
                                 receipt_kind: Optional[str] = None,
                                 receipt_class: Optional[str] = None) -> Optional[str]:
        """Send a text message (async version). See send_message for the blocks/meta_out
        contract, the receipts-require-a-class ValueError contract (§11.9/§11.23 — enforced
        by the base seam itself), and for `lease` (the stale-send guard)."""
        _require_receipt_class("send_message_async", receipts, receipt_class)
        return None

    @abstractmethod
    async def send_image(self, channel_id: str, thread_id: str, image_data: bytes, filename: str,
                         caption: str = "", meta_out: Optional[dict] = None,
                         receipts: Any = None, *,
                         receipt_class: Optional[str]) -> Optional[str]:
        """Send an image. Returns the posted image's URL (truthy on success) or None.

        `meta_out`: optional caller-provided dict a platform MAY populate with delivery facts
        — meta_out["file_id"] is the uploaded file's id (F7), the only handle from which the
        image message's own ts can later be resolved. Implementations that report none may
        ignore it; it defaults to None so non-Slack impls stay compatible.

        Receipt contract (EDIT §4/§11.9/§11.13/§11.23): `receipt_class` is REQUIRED —
        producers stamp the §4 inventory class (`artifact` for shares) — and the base seam
        itself raises ValueError on receipts-without-class, before any platform call."""
        _require_receipt_class("send_image", receipts, receipt_class)
        return None

    @abstractmethod
    async def send_image_async(self, channel_id: str, thread_id: str, image_data: bytes, filename: str,
                               caption: str = "", meta_out: Optional[dict] = None,
                               receipts: Any = None, *,
                               receipt_class: Optional[str]) -> Optional[str]:
        """Send an image (async version). See send_image for the return/meta_out contract and
        the receipts-require-a-class ValueError contract (§11.9/§11.13/§11.23 — enforced by
        the base seam itself)."""
        _require_receipt_class("send_image_async", receipts, receipt_class)
        return None

    async def send_file(self, channel_id: str, thread_id: str, file_data,
                        filename: str, title: Optional[str] = None,
                        initial_comment: str = "",
                        receipts: Any = None, *,
                        receipt_class: Optional[str]) -> Optional[Dict[str, Any]]:
        """F32: upload an arbitrary file, returning its platform identity
        ({"file_id", "url_private", "permalink"}) or None.

        Deliberately NOT abstract: a platform that can't take file uploads is a platform
        where artifacts simply don't publish, not one that fails to construct. The default
        declines honestly so the caller drops the artifact and still posts its answer.

        Receipt contract (EDIT §4/§11.9/§11.13): `receipt_class` is REQUIRED (producers
        stamp `artifact`), and receipts-without-class raises ValueError before any platform
        call — the default decline honors that contract too (§11.23 shared guard).
        """
        _require_receipt_class("send_file", receipts, receipt_class)
        self.log_warning(f"send_file not implemented for {self.__class__.__name__} — "
                         f"dropping artifact '{filename}'")
        return None

    @abstractmethod
    async def send_thinking_indicator(self, channel_id: str, thread_id: str,
                                      receipts: Any = None, *,
                                      receipt_class: Optional[str]) -> Optional[str]:
        """Send a thinking/processing indicator.

        Receipt contract (EDIT §4/§11.9/§11.13/§11.23): `receipt_class` is REQUIRED
        (producers stamp `chrome` — the placeholder is excluded scaffolding until promotion),
        and the base seam itself raises ValueError on receipts-without-class."""
        _require_receipt_class("send_thinking_indicator", receipts, receipt_class)
        return None

    @abstractmethod
    async def send_thinking_indicator_async(self, channel_id: str, thread_id: str,
                                            receipts: Any = None, *,
                                            receipt_class: Optional[str]) -> Optional[str]:
        """Send a thinking/processing indicator (async version). See send_thinking_indicator
        for the receipts-require-a-class ValueError contract (§11.9/§11.13/§11.23 — enforced
        by the base seam itself)."""
        _require_receipt_class("send_thinking_indicator_async", receipts, receipt_class)
        return None

    @abstractmethod
    async def delete_message(self, channel_id: str, message_id: str) -> bool:
        """Delete a message"""
        pass

    async def update_message(self, channel_id: str, message_id: str, text: str,
                             receipts: Any = None, receipt_kind: Optional[str] = None,
                             receipt_class: Optional[str] = None) -> bool:
        """Update a message (optional - not all platforms support this).

        Receipt contract (§11.9/§11.13/§11.23): receipts-without-class raises ValueError via
        the shared base guard — the default decline honors the contract too."""
        _require_receipt_class("update_message", receipts, receipt_class)
        return False

    async def update_message_async(self, channel_id: str, message_id: str, text: str,
                                   receipts: Any = None,
                                   receipt_kind: Optional[str] = None,
                                   receipt_class: Optional[str] = None) -> bool:
        """Update a message (async version - optional). Same §11.9/§11.13/§11.23 contract,
        via the shared base guard."""
        _require_receipt_class("update_message_async", receipts, receipt_class)
        return False

    async def react(self, channel_id: str, message_ts: str, emoji: str) -> bool:
        """Add an emoji reaction to a message (optional capability; default no-op)."""
        return False

    @abstractmethod
    async def get_thread_history(self, channel_id: str, thread_id: str, limit: Optional[int] = None,
                                 oldest: Optional[str] = None) -> List[Message]:
        """Get message history for a thread - fetches ALL messages by default.

        `oldest` (platform ts) fetches only messages strictly after it (Phase S
        summary-tail optimization). Raises HistoryFetchError on terminal fetch
        failure — an empty thread returns [] instead.
        """
        pass

    @abstractmethod
    async def download_file(self, file_url: str, file_id: Optional[str] = None,
                            allow_html: bool = False,
                            max_bytes: Optional[int] = None) -> Optional[bytes]:
        """Download a file/image from the platform, aborting past max_bytes when set.

        `allow_html` accepts an HTML body instead of treating it as the platform's sign-in
        page. Only a canvas (`application/vnd.slack-docs`) wants it: its content genuinely
        IS html, and the caller then has to tell a canvas apart from a login screen itself.
        """
        pass

    @abstractmethod
    async def download_file_async(self, file_url: str, file_id: Optional[str] = None,
                                  allow_html: bool = False,
                                  max_bytes: Optional[int] = None) -> Optional[bytes]:
        """Download a file/image from the platform (async version), aborting past max_bytes when set"""
        pass
    
    @abstractmethod
    def format_text(self, text: str) -> str:
        """Format text for the specific platform (markdown conversion)"""
        pass
    
    async def handle_error(self, channel_id: str, thread_id: str, error: str,
                           lease: Any = None, receipts: Any = None):
        """Default error handler.

        `lease` (stale guard): an error notice is TERMINAL — on a turn with no thinking surface
        (a silence-capable buffered turn has none by design) it is the room's first and only
        word from us. If the conversation has moved on, the honest outcome is nothing at all:
        the suppression terminal records what happened, and "something went wrong" would be a
        claim about a turn where nothing did."""
        # Check if this is a handled case (documents too large, etc) vs an actual error
        if "Documents Too Large" in error or "Message Too Long" in error:
            self.log_warning(f"Handled limit exceeded in {self.name}: {error[:100]}...")
        else:
            self.log_error(f"Error in {self.name}: {error}")

        # Format error message for better readability
        formatted_error = self.format_error_message(error)
        # Spec §4: an error notice is a system_notice, stamped here — the one producer of
        # this surface — whatever ledger it settles under.
        await self.send_message_async(channel_id, thread_id, formatted_error, lease=lease,
                                      receipts=receipts, receipt_class="system_notice")
    
    def format_error_message(self, error: str) -> str:
        """Format error messages for display (can be overridden by platform-specific clients)"""
        return f"Error: {error}"