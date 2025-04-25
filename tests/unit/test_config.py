"""Unit tests for the configuration service."""

import pytest
import os
import sqlite3
from unittest.mock import patch, MagicMock
import json

from app.core.config import ConfigService, DEFAULT_CONFIG
import prompts


class TestConfigService:
    """Test cases for the configuration service."""

    @pytest.fixture
    def memory_config_service(self):
        """Create a config service that uses memory storage."""
        return ConfigService(use_memory_store=True)

    @pytest.fixture
    def mock_sqlite(self):
        """Mock SQLite connection and cursor."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__.return_value = mock_conn
        mock_conn.__exit__.return_value = None
        
        return mock_conn

    def test_get_default_config_memory(self, memory_config_service):
        """Test getting default config from memory store."""
        config = memory_config_service.get("thread123")
        
        # Should return a copy of the default config
        assert config == DEFAULT_CONFIG
        assert config is not DEFAULT_CONFIG  # Should be a different object
        assert config["system_prompt"] == prompts.SLACK_SYSTEM_PROMPT["content"]

    def test_update_config_memory(self, memory_config_service):
        """Test updating config in memory store."""
        # First get default config
        config = memory_config_service.get("thread123")
        assert config["temperature"] == DEFAULT_CONFIG["temperature"]
        assert config["top_p"] == DEFAULT_CONFIG["top_p"]
        
        # Update temperature
        memory_config_service.update("thread123", {"temperature": 1.5})
        
        # Get updated config
        updated_config = memory_config_service.get("thread123")
        assert updated_config["temperature"] == 1.5
        assert updated_config["top_p"] == DEFAULT_CONFIG["top_p"]
        
        # Other values should remain unchanged
        assert updated_config["system_prompt"] == prompts.SLACK_SYSTEM_PROMPT["content"]

    def test_system_prompt_override(self, memory_config_service):
        """Test that system_prompt can be overridden."""
        # Define a custom system prompt
        custom_prompt = "This is a custom system prompt for testing."
        
        # Update the config with the custom prompt
        memory_config_service.update("thread123", {"system_prompt": custom_prompt})
        
        # Get the updated config
        config = memory_config_service.get("thread123")
        
        # Verify the system prompt was updated
        assert config["system_prompt"] == custom_prompt
        
        # Verify other config values remain the default
        assert config["temperature"] == DEFAULT_CONFIG["temperature"]
        assert config["top_p"] == DEFAULT_CONFIG["top_p"]

    def test_system_prompt_default_fallback(self, memory_config_service):
        """Test that system_prompt falls back to default when not overridden."""
        # Update a different config option
        memory_config_service.update("thread123", {"temperature": 1.2})
        
        # Get the config
        config = memory_config_service.get("thread123")
        
        # Verify the system prompt is still the default
        assert config["system_prompt"] == DEFAULT_CONFIG["system_prompt"]
        assert config["system_prompt"] == prompts.SLACK_SYSTEM_PROMPT["content"]

    def test_reset_config_memory(self, memory_config_service):
        """Test resetting config in memory store."""
        # First update config
        memory_config_service.update("thread123", {"temperature": 1.5})
        
        # Reset config
        memory_config_service.reset("thread123")
        
        # Get config after reset
        reset_config = memory_config_service.get("thread123")
        assert reset_config["temperature"] == DEFAULT_CONFIG["temperature"]
        assert reset_config["top_p"] == DEFAULT_CONFIG["top_p"]
        assert reset_config["system_prompt"] == prompts.SLACK_SYSTEM_PROMPT["content"]

    @patch('app.core.config.sqlite3.connect')
    def test_get_config_sqlite_new_thread(self, mock_connect, mock_sqlite):
        """Test getting config for a new thread from SQLite."""
        mock_connect.return_value = mock_sqlite
        
        # Create cursor that returns no results (new thread)
        mock_cursor = mock_sqlite.cursor.return_value
        mock_cursor.fetchone.return_value = None
        
        config_service = ConfigService()
        config = config_service.get("thread123")
        
        # Should store defaults for new thread
        assert mock_sqlite.execute.called
        # Should return defaults
        assert config == DEFAULT_CONFIG

    @patch('app.core.config.sqlite3.connect')
    def test_get_config_sqlite_existing_thread(self, mock_connect, mock_sqlite):
        """Test getting config for an existing thread from SQLite."""
        mock_connect.return_value = mock_sqlite
        
        # Create cursor that returns existing config
        mock_cursor = mock_sqlite.cursor.return_value
        mock_cursor.fetchone.return_value = ('{"temperature": 1.5, "top_p": 0.9}',)
        
        config_service = ConfigService()
        config = config_service.get("thread123")
        
        # Should query the database
        mock_sqlite.cursor.assert_called_once()
        mock_cursor.execute.assert_called_once()
        # Should return parsed config
        assert config["temperature"] == 1.5
        assert config["top_p"] == 0.9

    @patch('app.core.config.sqlite3.connect')
    def test_update_config_sqlite(self, mock_connect, mock_sqlite):
        """Test updating config in SQLite."""
        mock_connect.return_value = mock_sqlite
        
        # Create cursor that returns existing config
        mock_cursor = mock_sqlite.cursor.return_value
        mock_cursor.fetchone.return_value = (json.dumps(DEFAULT_CONFIG),)
        
        config_service = ConfigService()
        config_service.update("thread123", {"temperature": 1.7})
        
        # Should update the database
        mock_sqlite.execute.assert_called()

    @patch('app.core.config.sqlite3.connect')
    def test_reset_config_sqlite(self, mock_connect, mock_sqlite):
        """Test resetting config in SQLite."""
        mock_connect.return_value = mock_sqlite
        
        config_service = ConfigService()
        config_service.reset("thread123")
        
        # Should update the database with defaults
        assert mock_sqlite.execute.called
        assert mock_sqlite.commit.called

    def test_extract_config_from_text(self, memory_config_service):
        """Test extracting config values from text."""
        # Test number of images
        text = "Please generate 3 images of a mountain"
        config = memory_config_service.extract_config_from_text(text)
        assert config["number"] == 3
        
        # Test style
        text = "Create an image with vivid style"
        config = memory_config_service.extract_config_from_text(text)
        assert config["style"] == "vivid"
        
        # Test image model
        text = "Use gpt-image-1 to create a sunset"
        config = memory_config_service.extract_config_from_text(text)
        assert config["image_model"] == "gpt-image-1"
        
        # Test quality
        text = "Generate an hd quality image"
        config = memory_config_service.extract_config_from_text(text)
        assert config["quality"] == "hd"
        
        # Test size
        text = "Make a landscape format image"
        config = memory_config_service.extract_config_from_text(text)
        assert config["size"] == "1792x1024"
        
        # Test detail
        text = "Use high detail mode for this image"
        config = memory_config_service.extract_config_from_text(text)
        assert config["detail"] == "high"
        
        # Test temperature
        text = "Set temperature to 1.2 for more creativity"
        config = memory_config_service.extract_config_from_text(text)
        assert config["temperature"] == 1.2
        
        # Test multiple settings
        text = "Generate 2 images with vivid style and temperature 1.5"
        config = memory_config_service.extract_config_from_text(text)
        assert config["number"] == 2
        assert config["style"] == "vivid"
        assert config["temperature"] == 1.5 