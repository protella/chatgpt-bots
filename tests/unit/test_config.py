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
        
        
        # Check OpenAI credentials
        assert config.openai_api_key == 'sk-test-key'
        
        # Check model configuration
        assert config.gpt_model == 'gpt-5'
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
    
    def test_progress_checklist_default_enabled(self, mock_env):
        """ENABLE_PROGRESS_CHECKLIST defaults to True"""
        config = BotConfig()
        assert config.enable_progress_checklist is True

    @patch.dict(os.environ, {"ENABLE_PROGRESS_CHECKLIST": "false"})
    def test_progress_checklist_disabled(self, mock_env):
        """ENABLE_PROGRESS_CHECKLIST=false disables the checklist"""
        config = BotConfig()
        assert config.enable_progress_checklist is False

    def test_progress_checklist_prefer_message_default_enabled(self, mock_env):
        """PROGRESS_CHECKLIST_PREFER_MESSAGE defaults to False — where Slack's native
        assistant status exists it is the ONLY progress surface (a forced checklist
        message triple-displays the same text: message + in-thread status + composer)."""
        config = BotConfig()
        assert config.progress_checklist_prefer_message is False

    @patch.dict(os.environ, {"PROGRESS_CHECKLIST_PREFER_MESSAGE": "false"})
    def test_progress_checklist_prefer_message_disabled(self, mock_env):
        """PROGRESS_CHECKLIST_PREFER_MESSAGE=false reverts to status-only degradation"""
        config = BotConfig()
        assert config.progress_checklist_prefer_message is False

    def test_api_timeout_image_default(self, mock_env):
        """API_TIMEOUT_IMAGE defaults to 300 and exceeds the general read timeout"""
        config = BotConfig()
        assert config.api_timeout_image == 300.0
        assert config.api_timeout_image > config.api_timeout_read

    def test_thread_tail_defaults(self, mock_env):
        """F17: thread-tail default raised 6 → 15 (50 / 30 unchanged)."""
        config = BotConfig()
        assert config.participation_thread_tail == 15
        assert config.pulse_thread_tails_max == 50
        assert config.pulse_thread_tail_channels_max == 30

    @patch.dict(os.environ, {"PARTICIPATION_THREAD_TAIL": "9"})
    def test_thread_tail_env_override(self, mock_env):
        """PARTICIPATION_THREAD_TAIL honors env overrides."""
        config = BotConfig()
        assert config.participation_thread_tail == 9

    @patch.dict(os.environ, {"PARTICIPATION_THREAD_TAIL": "0"})
    def test_thread_tail_disabled(self, mock_env):
        """PARTICIPATION_THREAD_TAIL=0 disables the tail recording + signal."""
        config = BotConfig()
        assert config.participation_thread_tail == 0

    def test_reaction_max_per_message_default(self, mock_env):
        """F6 per-message reaction cap defaults to 4."""
        config = BotConfig()
        assert config.reaction_max_per_message == 4

    def test_f14_cap_overhaul_defaults(self, mock_env):
        """F14/F17 defaults: 5 / 60 / 500 / 500 / 20 / 80 / 300 / 90 (F17 raised the
        two pulse truncations to 500; the hourly cap knob is gone entirely)."""
        config = BotConfig()
        assert config.max_concurrent_image_generations == 5
        assert config.channel_pulse_size == 60
        assert config.pulse_text_truncate == 500
        assert config.pulse_tail_text_truncate == 500
        assert config.tool_provenance_max_entries == 20
        assert config.tool_provenance_gist_chars == 80
        assert config.tool_provenance_line_budget == 300
        assert config.tool_usage_retention_days == 90

    def test_hourly_cap_config_removed(self, mock_env):
        """F17: MAX_UNPROMPTED_REPLIES_PER_HOUR is retired — no config attribute."""
        config = BotConfig()
        assert not hasattr(config, "max_unprompted_replies_per_hour")

    @patch.dict(os.environ, {
        "CHANNEL_PULSE_SIZE": "12",
        "PULSE_TEXT_TRUNCATE": "111", "PULSE_TAIL_TEXT_TRUNCATE": "222",
        "TOOL_PROVENANCE_MAX_ENTRIES": "7", "TOOL_PROVENANCE_GIST_CHARS": "40",
        "TOOL_PROVENANCE_LINE_BUDGET": "150", "TOOL_USAGE_RETENTION_DAYS": "30",
    })
    def test_f14_env_overrides_respected(self, mock_env):
        """F14/F17 caps are env-backed and honor overrides."""
        config = BotConfig()
        assert config.channel_pulse_size == 12
        assert config.pulse_text_truncate == 111
        assert config.pulse_tail_text_truncate == 222
        assert config.tool_provenance_max_entries == 7
        assert config.tool_provenance_gist_chars == 40
        assert config.tool_provenance_line_budget == 150
        assert config.tool_usage_retention_days == 30

    def test_no_reply_tool_default_enabled(self, mock_env):
        """ENABLE_NO_REPLY_TOOL defaults to True"""
        config = BotConfig()
        assert config.enable_no_reply_tool is True

    @patch.dict(os.environ, {"ENABLE_NO_REPLY_TOOL": "false"})
    def test_no_reply_tool_disabled(self, mock_env):
        """ENABLE_NO_REPLY_TOOL=false hides the tool"""
        config = BotConfig()
        assert config.enable_no_reply_tool is False

    def test_wake_envelope_default_enabled(self, mock_env):
        """ENABLE_WAKE_ENVELOPE defaults to True"""
        config = BotConfig()
        assert config.enable_wake_envelope is True

    @patch.dict(os.environ, {"ENABLE_WAKE_ENVELOPE": "false"})
    def test_wake_envelope_disabled(self, mock_env):
        """ENABLE_WAKE_ENVELOPE=false disables the wake envelope"""
        config = BotConfig()
        assert config.enable_wake_envelope is False

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
        
        assert thread_config["model"] == 'gpt-5'
        assert thread_config["temperature"] == 0.7
        assert thread_config["max_tokens"] == 4096
        assert thread_config["reasoning_effort"] == 'medium'
        assert thread_config["verbosity"] == '2'
        assert thread_config["enable_streaming"] is True
        assert thread_config["image_size"] == "1024x1024"
        assert thread_config["image_quality"] == "auto"
        assert thread_config["image_background"] == "auto"
    
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
        assert config.default_image_quality == "auto"  # auto, low, medium, high
        assert config.default_image_background == "auto"  # auto, transparent, opaque
        assert config.default_image_number == 1
        assert config.default_image_format == "png"
    
    def test_emoji_configuration(self, mock_env):
        """Test emoji settings"""
        config = BotConfig()
        # Just verify these properties exist and have values
        assert config.thinking_emoji
        assert config.circle_loader_emoji
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
    
    def test_get_thread_config_with_user_preferences(self, mock_env):
        """Test get_thread_config with user preferences from database"""
        from unittest.mock import Mock
        
        config = BotConfig()
        mock_db = Mock()
        
        # Mock user preferences from database (legacy stored model — the
        # compose-time clamp coerces the effort against whatever model wins)
        mock_db.get_user_preferences.return_value = {
            'model': 'gpt-5-nano',
            'reasoning_effort': 'high',
            'verbosity': 'high',
            'temperature': 0.5,
            'top_p': 0.9,
            'enable_web_search': 0,  # False in DB
            'enable_streaming': 1,   # True in DB
            'image_size': '512x512',
            'input_fidelity': 'low',
            'vision_detail': 'low'
        }
        
        thread_config = config.get_thread_config(
            overrides={},
            db=mock_db,
            user_id='U123'
        )
        
        # Verify user preferences are applied ('high' is valid on every family,
        # so it survives the compose-time clamp)
        assert thread_config["model"] == 'gpt-5-nano'
        assert thread_config["reasoning_effort"] == 'high'
        assert thread_config["verbosity"] == 'high'
        assert thread_config["temperature"] == 0.5
        assert thread_config["top_p"] == 0.9
        assert thread_config["enable_web_search"] is False
        assert thread_config["enable_streaming"] is True
        assert thread_config["slack_streaming"] is True
        assert thread_config["image_size"] == '512x512'
        assert thread_config["input_fidelity"] == 'low'
        assert thread_config["detail_level"] == 'low'
        
        # Verify database was queried with correct user ID
        mock_db.get_user_preferences.assert_called_once_with('U123')
    
    def test_get_thread_config_with_user_preferences_error(self, mock_env):
        """Test get_thread_config handles database errors gracefully"""
        from unittest.mock import Mock
        
        config = BotConfig()
        mock_db = Mock()
        
        # Mock database error
        mock_db.get_user_preferences.side_effect = Exception("Database error")
        
        # Should not raise, but use defaults
        thread_config = config.get_thread_config(
            overrides={},
            db=mock_db,
            user_id='U123'
        )
        
        # Verify defaults are used
        assert thread_config["model"] == 'gpt-5'
        assert thread_config["reasoning_effort"] == 'medium'
        assert thread_config["verbosity"] == '2'
        
    def test_get_thread_config_priority_order(self, mock_env):
        """Test that override priority is: thread > user > system"""
        from unittest.mock import Mock
        
        config = BotConfig()
        mock_db = Mock()
        
        # Mock user preferences
        mock_db.get_user_preferences.return_value = {
            'model': 'gpt-5-nano',
            'temperature': 0.5
        }
        
        # Thread overrides should take precedence
        thread_config = config.get_thread_config(
            overrides={'model': 'gpt-5-mini', 'temperature': 0.3},
            db=mock_db,
            user_id='U123'
        )
        
        assert thread_config["model"] == 'gpt-5-mini'  # Thread override wins
        assert thread_config["temperature"] == 0.3  # Thread override wins
        assert thread_config["reasoning_effort"] == 'medium'  # System default

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
                "streaming": config.enable_streaming
            }
        }

        # This would help diagnose config issues
        print(f"\nDiagnostic Config Info: {diagnostic_info}")

        # Verify critical values are present
        assert diagnostic_info["model"] is not None
        assert diagnostic_info["temperature"] > 0
        assert diagnostic_info["max_tokens"] > 0

    def test_get_model_token_limit_gpt55(self, mock_env):
        """gpt-5.5 uses the 1.05M window with its dedicated buffer percentage"""
        config = BotConfig()

        limit = config.get_model_token_limit("gpt-5.5")
        expected = int(config.gpt54_max_tokens * config.gpt54_token_buffer_percentage)
        assert limit == expected

    def test_get_model_token_limit_gpt5_mini(self, mock_env):
        """gpt-5-mini (utility) uses the 400k window with the standard buffer"""
        config = BotConfig()

        limit = config.get_model_token_limit("gpt-5-mini")
        expected = int(config.gpt5_max_tokens * config.token_buffer_percentage)
        assert limit == expected

    def test_get_model_token_limit_unknown_model(self, mock_env):
        """Unknown models fall back to the conservative 400k window"""
        config = BotConfig()

        expected = int(config.gpt5_max_tokens * config.token_buffer_percentage)

        # Test unknown model
        limit = config.get_model_token_limit("unknown-model")
        assert limit == expected

        # Test empty model name
        limit = config.get_model_token_limit("")
        assert limit == expected

    def test_get_thread_config_with_custom_instructions(self, mock_env):
        """Test get_thread_config handles custom instructions from user preferences"""
        from unittest.mock import Mock

        config = BotConfig()
        mock_db = Mock()

        # Mock user preferences with custom instructions
        mock_db.get_user_preferences.return_value = {
            'model': 'gpt-5-mini',
            'custom_instructions': 'Always be helpful and concise'
        }

        thread_config = config.get_thread_config(
            overrides={},
            db=mock_db,
            user_id='U123'
        )

        # Verify custom instructions are included
        assert thread_config["custom_instructions"] == 'Always be helpful and concise'
        assert thread_config["model"] == 'gpt-5-mini'

    def test_get_thread_config_without_custom_instructions(self, mock_env):
        """Test get_thread_config when user has no custom instructions"""
        from unittest.mock import Mock

        config = BotConfig()
        mock_db = Mock()

        # Mock user preferences without custom instructions
        mock_db.get_user_preferences.return_value = {
            'model': 'gpt-5-mini',
            'temperature': 0.5
        }

        thread_config = config.get_thread_config(
            overrides={},
            db=mock_db,
            user_id='U123'
        )

        # Verify custom instructions are not included if not set
        assert "custom_instructions" not in thread_config
        assert thread_config["model"] == 'gpt-5-mini'
        assert thread_config["temperature"] == 0.5

class TestDocExtractionWorkers:
    """DOC_EXTRACTION_WORKERS sizes the shared extraction thread pool (user-exposed:
    OCR can hold a worker 1-2 min, and queue wait counts against READ_DOCUMENT_TIMEOUT)."""

    def test_default_is_five(self):
        config = BotConfig()
        assert config.doc_extraction_workers == 5

    @patch.dict(os.environ, {"DOC_EXTRACTION_WORKERS": "8"})
    def test_env_override(self):
        config = BotConfig()
        assert config.doc_extraction_workers == 8

    @patch.dict(os.environ, {"DOC_EXTRACTION_WORKERS": "0"})
    def test_floor_of_one(self):
        config = BotConfig()
        assert config.doc_extraction_workers == 1
