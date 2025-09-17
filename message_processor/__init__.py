"""Compatibility wrapper while message_processor is being modularized."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_monolith_path = Path(__file__).resolve().parent.parent / "message_processor.py"
_spec = importlib.util.spec_from_file_location("_message_processor_monolith", _monolith_path)
_monolith = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_monolith)  # type: ignore[misc]

MessageProcessor = _monolith.MessageProcessor

__all__ = ["MessageProcessor"]
