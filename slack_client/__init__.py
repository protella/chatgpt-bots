"""Slack client package exports.

`SlackBot` is resolved lazily, for the same reason `message_processor` resolves
`MessageProcessor` lazily. `slack_client.base` reaches into `message_processor`
(`messaging.py` imports `turn_runtime`), while `message_processor.turn_runtime`
reaches back here for `slack_client.normalizer`. Binding the class eagerly would make
that leaf import execute the whole Slack transport first, so importing
`message_processor.turn_runtime` — or any of the dozen modules behind it — would fail
on a partially initialized module.
"""
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import SlackBot

__all__ = ["SlackBot"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from . import base
        return getattr(base, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # Without this, dir() and help() miss the lazily-resolved exports entirely.
    return sorted(list(globals()) + __all__)
