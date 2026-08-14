"""Message processor package.

`MessageProcessor` is resolved lazily. `message_processor.base` pulls in the whole
processing graph — the Slack and OpenAI clients, the streaming package, the tool
handlers — and several of those reach back into this package for leaf modules
(`prompts`, `message_markers`, `tool_registry`, `token_counter`). Binding the class
eagerly here would make every one of those leaf imports execute the full graph
first, so whichever package was imported first would fail on a partially
initialized module.
"""
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import MessageProcessor

__all__ = ["MessageProcessor"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import base
        return getattr(base, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # Without this, dir() and help() miss the lazily-resolved exports entirely.
    return sorted(list(globals()) + __all__)
