"""What a `MessageProcessor` mixin is allowed to assume about the object it ends up inside.

The counterpart to `slack_client._host`, and it exists for the same reason: `MessageProcessor` is
assembled from four mixins plus `LoggerMixin`, and each of them reaches for state owned elsewhere
in the composition — `self.thread_manager`, `self.openai_client`, the logging methods, a sibling's
`_update_status`. Correct at runtime, invisible to a checker reading one mixin at a time.

Read the module docstring in `slack_client/_host.py` for the full rationale. The short version:
this is a typing construct, `_Host` is plain `object` at runtime, nothing here executes, and
declaring a name here does not implement it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable


class MessageProcessorHost:
    """The attributes and collaborators `MessageProcessor` guarantees to its mixins."""

    # --- owned by MessageProcessor.__init__ ----------------------------------------------------
    db: Any                                   # DatabaseManager | None
    document_handler: Any
    image_url_handler: Any
    mcp_manager: Any
    openai_client: Any
    thread_manager: Any

    # --- LoggerMixin ---------------------------------------------------------------------------
    log_debug: Callable[..., None]
    log_info: Callable[..., None]
    log_warning: Callable[..., None]
    log_error: Callable[..., None]

    # --- provided by sibling mixins ------------------------------------------------------------
    _add_message_with_token_management: Callable[..., Any]
    _async_post_response_cleanup: Callable[..., Any]
    _build_channel_info: Callable[..., Any]
    _build_message_with_documents: Callable[..., Any]
    _build_participant_roster: Callable[..., Any]
    _build_suffix_context: Callable[..., Any]
    _compact_thread_to_target: Callable[..., Any]
    _get_system_prompt: Callable[..., Any]
    _inject_image_analyses: Callable[..., Any]
    _is_context_length_error: Callable[..., Any]
    _persist_tool_provenance: Callable[..., Any]
    _pre_trim_messages_for_api: Callable[..., Any]
    _schedule_async_call: Callable[..., Any]
    _start_progress_updater_async: Callable[..., Any]
    _summarize_document_for_attach: Callable[..., Any]
    _update_status: Callable[..., Any]


if TYPE_CHECKING:
    _Host = MessageProcessorHost
else:  # runtime: contribute nothing to the MRO
    _Host = object
