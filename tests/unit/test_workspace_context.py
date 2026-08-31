"""Workspace context file — the admin-authored background injected into every Slack prompt.

Covers the read-once loader (off, missing, blank, oversize) and its injection into
`_get_system_prompt`, including the invariant that an unset feature adds nothing at all: the
channel prefix is contracted to be byte-stable, so a stray newline here busts the prompt cache.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from config import config
from message_processor.utilities import (
    WORKSPACE_CONTEXT_WARN_CHARS,
    MessageUtilitiesMixin,
    _load_workspace_context,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    """The loader is `lru_cache(maxsize=1)` — one test's file would otherwise be every test's."""
    _load_workspace_context.cache_clear()
    yield
    _load_workspace_context.cache_clear()


def _write(tmp_path, text: str) -> str:
    path = tmp_path / "workspace_context.md"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _prompt(proc) -> str:
    client = MagicMock()
    client.name = "slack"
    return proc._get_system_prompt(client, "UTC")


def _proc():
    return MessageUtilitiesMixin.__new__(type("P", (MessageUtilitiesMixin,), {}))


# --------------------------------------------------------------------------- loader

def test_unset_setting_returns_empty_without_touching_disk():
    with patch.object(config, "workspace_context_file", ""), \
         patch("message_processor.utilities.Path") as mock_path:
        assert _load_workspace_context() == ""
    mock_path.assert_not_called()


def test_missing_file_returns_empty_and_warns(tmp_path):
    missing = str(tmp_path / "nope.md")
    with patch.object(config, "workspace_context_file", missing), \
         patch("message_processor.utilities.logger") as mock_logger:
        assert _load_workspace_context() == ""
        assert mock_logger.warning.call_count == 1


def test_file_content_is_returned_stripped(tmp_path):
    path = _write(tmp_path, "\n\n  Example Corp sells widgets.\n\n")
    with patch.object(config, "workspace_context_file", path):
        assert _load_workspace_context() == "Example Corp sells widgets."


def test_whitespace_only_file_behaves_as_absent(tmp_path):
    path = _write(tmp_path, "   \n\t\n")
    with patch.object(config, "workspace_context_file", path):
        assert _load_workspace_context() == ""


def test_oversize_file_is_loaded_in_full_with_a_warning(tmp_path):
    body = "x" * (WORKSPACE_CONTEXT_WARN_CHARS + 10)
    path = _write(tmp_path, body)
    with patch.object(config, "workspace_context_file", path), \
         patch("message_processor.utilities.logger") as mock_logger:
        assert _load_workspace_context() == body
        assert mock_logger.warning.call_count == 1


# --------------------------------------------------------------------------- injection

def test_prompt_carries_the_labeled_section_and_the_content(tmp_path):
    path = _write(tmp_path, "Example Corp sells widgets. WGT = widget.")
    with patch.object(config, "workspace_context_file", path):
        out = _prompt(_proc())
    assert "--- WORKSPACE CONTEXT (provided by the workspace admin) ---" in out
    assert "--- END WORKSPACE CONTEXT ---" in out
    assert "Example Corp sells widgets. WGT = widget." in out


def test_prompt_is_untouched_when_the_feature_is_off():
    with patch.object(config, "workspace_context_file", ""):
        out = _prompt(_proc())
    assert "WORKSPACE CONTEXT" not in out
