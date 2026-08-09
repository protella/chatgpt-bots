"""What a `SlackBot` mixin is allowed to assume about the object it ends up inside.

`SlackBot` is assembled from eleven mixins plus `BaseClient`, and each mixin freely reaches for
state that some *other* member of that composition owns — `self.app`, `self.db`, the logging
methods, a sibling's `classify_sender`. That is a normal and deliberate arrangement at runtime,
but a mixin read on its own genuinely does not have those attributes, and mypy said so 768 times
across the codebase, drowning every real finding in the noise.

This class is the missing half of the contract: it declares what the composed object provides, so
a mixin can be checked against the thing it will actually be part of.

IT IS A TYPING CONSTRUCT AND NOTHING ELSE. Mixins pick it up through the `_Host` alias below,
which is this class only under `TYPE_CHECKING` and plain `object` at runtime — so no base class
is added to any MRO, no attribute is created, and nothing here executes. Adding a name here does
not implement it; `SlackBot.__init__` and the mixins remain the only implementations.

Methods are declared as `Callable` attributes rather than `def` stubs on purpose. A `def` would
make every mixin that defines the real method an override, and mypy would then police signature
compatibility between a stub written here and the truth written there — turning one class of
false error into another.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, Optional


class SlackBotHost:
    """The attributes and collaborators `SlackBot` guarantees to its mixins."""

    # --- owned by SlackBot.__init__ / BaseClient.__init__ -------------------------------------
    app: Any                                  # slack_bolt AsyncApp
    app_id: Optional[str]
    db: Any                                   # DatabaseManager | None
    user_cache: Dict[str, Any]
    markdown_converter: Any
    message_handler: Optional[Callable[..., Any]]
    settings_modal: Any

    # --- class-level configuration -------------------------------------------------------------
    MAX_MESSAGE_LENGTH: int

    # --- LoggerMixin ---------------------------------------------------------------------------
    log_debug: Callable[..., None]
    log_info: Callable[..., None]
    log_warning: Callable[..., None]
    log_error: Callable[..., None]

    # --- provided by sibling mixins ------------------------------------------------------------
    _clean_mentions: Callable[..., Any]
    _maybe_set_assistant_thread_title: Callable[..., Any]
    bot_user_id_for: Callable[..., Any]
    classify_sender: Callable[..., Any]
    format_text: Callable[..., Any]
    get_message_permalink_tool: Callable[..., Any]
    get_user_timezone: Callable[..., Any]
    get_username: Callable[..., Any]
    is_own_message: Callable[..., Any]
    resolve_usernames: Callable[..., Any]


if TYPE_CHECKING:
    _Host = SlackBotHost
else:  # runtime: contribute nothing to the MRO
    _Host = object
