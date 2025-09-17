"""Compatibility wrapper while slack_client is being modularized."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_monolith_path = Path(__file__).resolve().parent.parent / "slack_client.py"
_spec = importlib.util.spec_from_file_location("_slack_client_monolith", _monolith_path)
_monolith = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_monolith)  # type: ignore[misc]

SlackBot = _monolith.SlackBot

__all__ = ["SlackBot"]
