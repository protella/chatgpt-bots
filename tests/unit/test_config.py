"""
Unit tests for config.py module
"""
import os
import pytest
from unittest.mock import patch
from config import BotConfig


class TestBotConfig:
    """Test BotConfig class"""
    
    def test_default_initialization(self, mock_env):
        """Test config initialization with default values"""
        config = BotConfig()
        
        # Check Slack credentials
        assert config.slack_bot_token == 'xoxb-test-token'
        assert config.slack_app_token == 'xapp-test-token'
        
        # Check Discord credentials
        assert config.discord_token == 'discord-test-token'
        
        # Check OpenAI credentials
        assert config.openai_api_key == 'sk-test-key'
        
        # Check model configuration
        assert config.gpt_model == 'gpt-5-chat-latest'
        assert config.default_reasoning_effort == 'medium'
        assert config.default_verbosity == '2'
    
    def test_temperature_float_conversion(self, mock_env):
        """Test that temperature is properly converted to float"""
        config = BotConfig()
        assert isinstance(config.default_temperature, float)
        assert config.default_temperature == 0.7
    
    def test_max_tokens_int_conversion(self, mock_env):
        """Test that max_tokens is properly converted to int"""
        config = BotConfig()
        assert isinstance(config.default_max_tokens, int)
        assert config.default_max_tokens == 4096
    
    def test_boolean_conversion(self, mock_env):
        """Test boolean environment variable conversion"""
        config = BotConfig()
        assert config.enable_web_search is True
        assert config.enable_streaming is True
        assert config.debug_mode is False
    
    def test_validate_success(self, mock_env):
        """Test successful validation with all required fields"""
        config = BotConfig()
        assert config.validate() is True
    
    def test_validate_missing_slack_bot_token(self, monkeypatch):
        """Test validation fails when SLACK_BOT_TOKEN is missing"""
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
        config = BotConfig()
        
        with pytest.raises(ValueError, match="SLACK_BOT_TOKEN is required"):
            config.validate()
    
    def test_validate_missing_slack_app_token(self, mock_env, monkeypatch):
        """Test validation fails when SLACK_APP_TOKEN is missing"""
        monkeypatch.delenv("SLACK_APP_TOKEN", raising=False)
        config = BotConfig()
        
        with pytest.raises(ValueError, match="SLACK_APP_TOKEN is required"):
            config.validate()
    
    def test_validate_missing_openai_key(self, mock_env, monkeypatch):
        """Test validation fails when OPENAI_KEY is missing"""
        monkeypatch.delenv("OPENAI_KEY", raising=False)
        config = BotConfig()
        
        with pytest.raises(ValueError, match="OPENAI_KEY is required"):
            config.validate()
    
    def test_get_thread_config_default(self, mock_env):
        """Test get_thread_config returns default configuration"""
        config = BotConfig()
        thread_config = config.get_thread_config()
        
        assert thread_config["model"] == 'gpt-5-chat-latest'
        assert thread_config["temperature"] == 0.7
        assert thread_config["max_tokens"] == 4096
        assert thread_config["reasoning_effort"] == 'medium'
        assert thread_config["verbosity"] == '2'
        assert thread_config["enable_streaming"] is True
        assert thread_config["image_size"] == "1024x1024"
        assert thread_config["image_quality"] == "hd"
    
    def test_get_thread_config_with_overrides(self, mock_env):
        """Test get_thread_config with overrides"""
        config = BotConfig()
        overrides = {
            "model": "gpt-5-nano",
            "temperature": 0.3,
            "max_tokens": 2048,
            "custom_param": "custom_value"
        }
        
        thread_config = config.get_thread_config(overrides)
        
        assert thread_config["model"] == "gpt-5-nano"
        assert thread_config["temperature"] == 0.3
        assert thread_config["max_tokens"] == 2048
        assert thread_config["custom_param"] == "custom_value"
        # Check that non-overridden values remain default
        assert thread_config["reasoning_effort"] == 'medium'
        assert thread_config["verbosity"] == '2'
    
    def test_utility_parameters(self, mock_env):
        """Test utility-specific parameters"""
        config = BotConfig()
        assert config.utility_reasoning_effort == 'low'
        assert config.utility_verbosity == '1'
    
    def test_analysis_parameters(self, mock_env):
        """Test analysis-specific parameters"""
        config = BotConfig()
        assert config.analysis_reasoning_effort == 'high'
        assert config.analysis_verbosity == '3'
    
    def test_streaming_configuration(self, mock_env):
        """Test streaming-related configuration"""
        config = BotConfig()
        assert config.enable_streaming is True
        assert config.slack_streaming is True
        assert config.streaming_update_interval == 2.0
        assert config.streaming_buffer_size == 500
        assert config.streaming_circuit_breaker_threshold == 5
    
    def test_image_generation_parameters(self, mock_env):
        """Test image generation parameters"""
        config = BotConfig()
        assert config.default_image_size == "1024x1024"
        assert config.default_image_quality == "hd"
        assert config.default_image_style == "natural"
        assert config.default_image_number == 1
        assert config.default_image_format == "png"
    
    def test_emoji_configuration(self, mock_env):
        """Test emoji settings"""
        config = BotConfig()
        # Just verify these properties exist and have values
        assert config.thinking_emoji
        assert config.web_search_emoji
        assert config.loading_ellipse_emoji
    
    @patch.dict(os.environ, {"DEFAULT_TEMPERATURE": "not_a_number"})
    def test_invalid_float_conversion(self):
        """Test handling of invalid float values"""
        with pytest.raises(ValueError):
            BotConfig()
    
    @patch.dict(os.environ, {"DEFAULT_MAX_TOKENS": "not_a_number"})
    def test_invalid_int_conversion(self):
        """Test handling of invalid int values"""
        with pytest.raises(ValueError):
            BotConfig()
    
    @pytest.mark.critical
    def test_config_contract_interface(self, mock_env):
        """Contract test: Ensure BotConfig provides expected interface for other components"""
        config = BotConfig()
        
        # Verify all required attributes exist for OpenAIClient
        assert hasattr(config, 'openai_api_key')
        assert hasattr(config, 'gpt_model')
        assert hasattr(config, 'default_temperature')
        assert hasattr(config, 'default_max_tokens')
        assert hasattr(config, 'api_timeout_read')
        assert hasattr(config, 'api_timeout_streaming_chunk')
        
        # Verify all required attributes exist for SlackClient
        assert hasattr(config, 'slack_bot_token')
        assert hasattr(config, 'slack_app_token')
        
        # Verify all required attributes exist for ThreadStateManager
        assert hasattr(config, 'default_reasoning_effort')
        assert hasattr(config, 'default_verbosity')
        
        # Verify get_thread_config method signature hasn't changed
        assert callable(config.get_thread_config)
        # Test it accepts optional overrides
        result = config.get_thread_config()
        assert isinstance(result, dict)
        result_with_overrides = config.get_thread_config({"test": "value"})
        assert isinstance(result_with_overrides, dict)
    
    def test_timeout_configuration(self, mock_env):
        """Test timeout configuration values are properly loaded"""
        config = BotConfig()
        
        # Check timeout values are floats and reasonable
        assert isinstance(config.api_timeout_read, float)
        assert isinstance(config.api_timeout_streaming_chunk, float)
        
        assert config.api_timeout_read == 180.0
        assert config.api_timeout_streaming_chunk == 30.0
        
        # Ensure read timeout is greater than streaming chunk timeout
        assert config.api_timeout_read > config.api_timeout_streaming_chunk
    
    def test_config_persistence_state(self, mock_env):
        """State test: Verify config values remain consistent across multiple instantiations"""
        config1 = BotConfig()
        initial_model = config1.gpt_model
        initial_temp = config1.default_temperature
        
        # Create new instance
        config2 = BotConfig()
        
        # Values should be identical (loaded from same env)
        assert config2.gpt_model == initial_model
        assert config2.default_temperature == initial_temp
        
        # Overrides should not affect base config
        overridden = config1.get_thread_config({"model": "different-model"})
        assert overridden["model"] == "different-model"
        assert config1.gpt_model == initial_model  # Original unchanged
    
    @pytest.mark.smoke
    def test_smoke_basic_config_load(self, mock_env):
        """Smoke test: Verify config can be instantiated and validated"""
        try:
            config = BotConfig()
            assert config.validate() is True
            assert config.openai_api_key is not None
            assert config.slack_bot_token is not None
        except Exception as e:
            pytest.fail(f"Basic config loading failed: {e}")
    
    def test_diagnostic_config_values(self, mock_env):
        """Diagnostic test: Log all config values for debugging"""
        config = BotConfig()
        
        # Capture important values for debugging
        diagnostic_info = {
            "model": config.gpt_model,
            "temperature": config.default_temperature,
            "max_tokens": config.default_max_tokens,
            "reasoning_effort": config.default_reasoning_effort,
            "verbosity": config.default_verbosity,
            "timeouts": {
                "read": config.api_timeout_read,
                "streaming_chunk": config.api_timeout_streaming_chunk
            },
            "features": {
                "web_search": config.enable_web_search,
                "streaming": config.enable_streaming,
                "debug": config.debug_mode
            }
        }
        
        # This would help diagnose config issues
        print(f"\nDiagnostic Config Info: {diagnostic_info}")
        
        # Verify critical values are present
        assert diagnostic_info["model"] is not None
        assert diagnostic_info["temperature"] > 0
        assert diagnostic_info["max_tokens"] > 0