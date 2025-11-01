"""
Configuration module for Slack Bot V2
Handles all environment variables and default settings
"""
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

load_dotenv()

# Model knowledge cutoff dates
# Supported models: gpt-5, gpt-5-mini, gpt-4.1, gpt-4o
MODEL_KNOWLEDGE_CUTOFFS = {
    # GPT-5 series
    "gpt-5": "September 30, 2024",
    "gpt-5-mini": "September 30, 2024",
    
    # GPT-4 series
    "gpt-4.1": "June 1, 2024",
    "gpt-4o": "October 1, 2023",
    
    # Default fallback
    "default": "January 1, 2024"
}


@dataclass
class BotConfig:
    """Central configuration for the Slack bot"""
    
    # Slack credentials
    slack_bot_token: str = field(default_factory=lambda: os.getenv("SLACK_BOT_TOKEN", ""))
    slack_app_token: str = field(default_factory=lambda: os.getenv("SLACK_APP_TOKEN", ""))
    
    # Discord credentials
    discord_token: str = field(default_factory=lambda: os.getenv("DISCORD_TOKEN", ""))
    discord_allowed_channels: str = field(default_factory=lambda: os.getenv("DISCORD_ALLOWED_CHANNEL_IDS", ""))
    
    # OpenAI credentials
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_KEY", ""))
    
    # Model configuration
    gpt_model: str = field(default_factory=lambda: os.getenv("GPT_MODEL", "gpt-5"))
    utility_model: str = field(default_factory=lambda: os.getenv("UTILITY_MODEL", "gpt-5-mini"))
    image_model: str = field(default_factory=lambda: os.getenv("GPT_IMAGE_MODEL", "gpt-image-1"))
    
    # Default parameters for text generation
    default_temperature: float = field(default_factory=lambda: float(os.getenv("DEFAULT_TEMPERATURE", "0.8")))
    default_max_tokens: int = field(default_factory=lambda: int(os.getenv("DEFAULT_MAX_TOKENS", "4096")))
    default_top_p: float = field(default_factory=lambda: float(os.getenv("DEFAULT_TOP_P", "1.0")))
    
    # GPT-5 specific parameters
    default_reasoning_effort: str = field(default_factory=lambda: os.getenv("DEFAULT_REASONING_EFFORT", "medium"))
    default_verbosity: str = field(default_factory=lambda: os.getenv("DEFAULT_VERBOSITY", "medium"))
    
    # Utility function parameters (for quick checks, image intent, etc.)
    utility_reasoning_effort: str = field(default_factory=lambda: os.getenv("UTILITY_REASONING_EFFORT", "minimal"))
    utility_verbosity: str = field(default_factory=lambda: os.getenv("UTILITY_VERBOSITY", "low"))
    utility_max_tokens: int = field(default_factory=lambda: int(os.getenv("UTILITY_MAX_TOKENS", "20")))

    # Analysis function parameters (for vision analysis, complex tasks)
    analysis_reasoning_effort: str = field(default_factory=lambda: os.getenv("ANALYSIS_REASONING_EFFORT", "medium"))
    analysis_verbosity: str = field(default_factory=lambda: os.getenv("ANALYSIS_VERBOSITY", "medium"))
    vision_max_tokens: int = field(default_factory=lambda: int(os.getenv("VISION_MAX_TOKENS", "8192")))
    
    # Image generation parameters
    default_image_size: str = field(default_factory=lambda: os.getenv("DEFAULT_IMAGE_SIZE", "1024x1024"))
    default_image_quality: str = field(default_factory=lambda: os.getenv("DEFAULT_IMAGE_QUALITY", "hd"))  # standard or hd
    default_image_style: str = field(default_factory=lambda: os.getenv("DEFAULT_IMAGE_STYLE", "natural"))  # natural or vivid
    default_image_number: int = field(default_factory=lambda: int(os.getenv("DEFAULT_IMAGE_NUMBER", "1")))  # Number of images
    default_image_background: str = field(default_factory=lambda: os.getenv("DEFAULT_IMAGE_BACKGROUND", "auto"))
    default_image_format: str = field(default_factory=lambda: os.getenv("DEFAULT_IMAGE_FORMAT", "png"))
    default_image_compression: int = field(default_factory=lambda: int(os.getenv("DEFAULT_IMAGE_COMPRESSION", "100")))  # 100 for PNG, can be less for JPEG/WebP
    default_input_fidelity: str = field(default_factory=lambda: os.getenv("DEFAULT_INPUT_FIDELITY", "high"))  # high or low
    
    # Vision parameters
    default_detail_level: str = field(default_factory=lambda: os.getenv("DEFAULT_DETAIL_LEVEL", "auto"))
    
    # System behavior (will be overridden by platform-specific prompts)
    default_system_prompt: str = field(default_factory=lambda: os.getenv(
        "DEFAULT_SYSTEM_PROMPT",
        ""  # Empty default, will use prompts.py
    ))
    
    # UI Configuration
    thinking_emoji: str = field(default_factory=lambda: os.getenv("THINKING_EMOJI", ":hourglass_flowing_sand:"))
    web_search_emoji: str = field(default_factory=lambda: os.getenv("WEB_SEARCH_EMOJI", ":web_search:"))
    loading_ellipse_emoji: str = field(default_factory=lambda: os.getenv("LOADING_ELLIPSE_EMOJI", ":loading-ellipse:"))
    circle_loader_emoji: str = field(default_factory=lambda: os.getenv("CIRCLE_LOADER_EMOJI", ":circle-loader:"))
    analyze_emoji: str = field(default_factory=lambda: os.getenv("ANALYZE_EMOJI", ":analyze:"))
    error_emoji: str = field(default_factory=lambda: os.getenv("ERROR_EMOJI", ":warning:"))
    
    # Web search configuration
    enable_web_search: bool = field(default_factory=lambda: os.getenv("ENABLE_WEB_SEARCH", "true").lower() == "true")
    web_search_model: str = field(default_factory=lambda: os.getenv("WEB_SEARCH_MODEL", ""))  # Empty = use default model

    # MCP (Model Context Protocol) configuration
    mcp_enabled_default: bool = field(default_factory=lambda: os.getenv("MCP_ENABLED_DEFAULT", "true").lower() == "true")
    mcp_config_path: str = field(default_factory=lambda: os.getenv("MCP_CONFIG_PATH", "mcp_config.json"))

    # Slack settings configuration
    settings_slash_command: str = field(default_factory=lambda: os.getenv("SETTINGS_SLASH_COMMAND", "/chatgpt-settings"))
    
    # Database configuration
    database_dir: str = field(default_factory=lambda: os.getenv("DATABASE_DIR", "data"))

    # Logging configuration
    log_level: str = field(default_factory=lambda: os.getenv("BOT_LOG_LEVEL", "INFO"))
    slack_log_level: str = field(default_factory=lambda: os.getenv("SLACK_LOG_LEVEL", "INFO"))
    discord_log_level: str = field(default_factory=lambda: os.getenv("DISCORD_LOG_LEVEL", "INFO"))
    utils_log_level: str = field(default_factory=lambda: os.getenv("UTILS_LOG_LEVEL", "INFO"))
    console_logging_enabled: bool = field(default_factory=lambda: os.getenv("CONSOLE_LOGGING_ENABLED", "TRUE").upper() == "TRUE")
    log_directory: str = field(default_factory=lambda: os.getenv("LOG_DIRECTORY", "logs"))
    debug_mode: bool = field(default_factory=lambda: os.getenv("DEBUG_MODE", "false").lower() == "true")
    
    # Performance settings
    max_concurrent_threads: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_THREADS", "10")))
    message_timeout: int = field(default_factory=lambda: int(os.getenv("MESSAGE_TIMEOUT", "60")))
    
    # Cleanup settings
    cleanup_schedule: str = field(default_factory=lambda: os.getenv("CLEANUP_SCHEDULE", "0 0 * * *"))  # Default: midnight daily
    cleanup_max_age_hours: float = field(default_factory=lambda: float(os.getenv("CLEANUP_MAX_AGE_HOURS", "24")))
    
    # API Timeout settings (in seconds)
    api_timeout_read: float = field(default_factory=lambda: float(os.getenv("API_TIMEOUT_READ", "180")))  # Overall timeout for API requests
    api_timeout_streaming_chunk: float = field(default_factory=lambda: float(os.getenv("API_TIMEOUT_STREAMING_CHUNK", "30")))  # Max time between streaming chunks
    
    # Model token limits
    # GPT-5: 400k total context window (shared between input, output, and reasoning)
    # With max_output_tokens=32k, we can theoretically use up to 368k for input
    # BUT: We must also account for system prompt (~1k), tool definitions (~0.5k),
    # and API formatting overhead (~8.5k), so practical limit is ~358k
    # We use 67.5% (270k) to ensure we stay well under the actual limit
    gpt5_max_tokens: int = field(default_factory=lambda: int(os.getenv("GPT5_MAX_TOKENS", "400000")))  # Total context window
    gpt4_max_tokens: int = field(default_factory=lambda: int(os.getenv("GPT4_MAX_TOKENS", "128000")))
    
    # Token management configuration
    # Buffer to leave room for output/reasoning tokens and overhead
    token_buffer_percentage: float = field(default_factory=lambda: float(os.getenv("TOKEN_BUFFER_PERCENTAGE", "0.875")))
    token_cleanup_threshold: float = field(default_factory=lambda: float(os.getenv("TOKEN_CLEANUP_THRESHOLD", "0.8")))
    token_trim_message_count: int = field(default_factory=lambda: int(os.getenv("TOKEN_TRIM_MESSAGE_COUNT", "5")))
    
    # Legacy - kept for backward compatibility, will be calculated dynamically
    thread_max_token_count: int = field(default_factory=lambda: int(os.getenv("THREAD_MAX_TOKEN_COUNT", "350000")))
    
    # Streaming configuration
    enable_streaming: bool = field(default_factory=lambda: os.getenv("ENABLE_STREAMING", "true").lower() == "true")
    slack_streaming: bool = field(default_factory=lambda: os.getenv("SLACK_STREAMING", "true").lower() == "true")
    streaming_update_interval: float = field(default_factory=lambda: float(os.getenv("STREAMING_UPDATE_INTERVAL", "2.0")))
    streaming_min_interval: float = field(default_factory=lambda: float(os.getenv("STREAMING_MIN_INTERVAL", "1.0")))
    streaming_max_interval: float = field(default_factory=lambda: float(os.getenv("STREAMING_MAX_INTERVAL", "30.0")))
    streaming_buffer_size: int = field(default_factory=lambda: int(os.getenv("STREAMING_BUFFER_SIZE", "500")))
    streaming_circuit_breaker_threshold: int = field(default_factory=lambda: int(os.getenv("STREAMING_CIRCUIT_BREAKER_THRESHOLD", "5")))
    streaming_circuit_breaker_cooldown: int = field(default_factory=lambda: int(os.getenv("STREAMING_CIRCUIT_BREAKER_COOLDOWN", "300")))
    
    def get_model_token_limit(self, model: str) -> int:
        """Get the effective input token limit for a specific model

        This returns the maximum number of input tokens we should send.
        For GPT-5: 400k total - output reservation = ~350k with buffer
        For GPT-4: 128k total - output reservation = ~112k with buffer

        Args:
            model: Model name (e.g., 'gpt-5', 'gpt-4.1', 'gpt-4o')

        Returns:
            Buffered token limit for safe operation
        """
        # Determine base limit based on model family
        if model.startswith('gpt-5'):
            base_limit = self.gpt5_max_tokens
        elif model.startswith('gpt-4'):
            base_limit = self.gpt4_max_tokens
        else:
            # Default to GPT-4 limit for unknown models
            base_limit = self.gpt4_max_tokens
        
        # Apply buffer percentage
        return int(base_limit * self.token_buffer_percentage)
    
    def validate(self) -> bool:
        """Validate required configuration"""
        if not self.slack_bot_token:
            raise ValueError("SLACK_BOT_TOKEN is required")
        if not self.slack_app_token:
            raise ValueError("SLACK_APP_TOKEN is required")
        if not self.openai_api_key:
            raise ValueError("OPENAI_KEY is required")
        return True
    
    def get_thread_config(self, overrides: Optional[Dict[str, Any]] = None, user_id: Optional[str] = None, db = None) -> Dict[str, Any]:
        """Get configuration for a specific thread with settings hierarchy:
        1. System defaults (from .env)
        2. User preferences (from database)
        3. Thread overrides (passed as parameter)
        
        Args:
            overrides: Thread-specific overrides
            user_id: User ID to fetch preferences for
            db: Database connection to fetch user preferences
        """
        # Start with system defaults
        config = {
            # Text generation
            "model": self.gpt_model,
            "temperature": self.default_temperature,
            "max_tokens": self.default_max_tokens,
            "top_p": self.default_top_p,
            "system_prompt": self.default_system_prompt,
            
            # GPT-5 specific
            "reasoning_effort": self.default_reasoning_effort,
            "verbosity": self.default_verbosity,
            
            # Image generation
            "image_model": self.image_model,
            "image_size": self.default_image_size,
            "image_quality": self.default_image_quality,
            "image_style": self.default_image_style,
            "image_number": self.default_image_number,
            "image_background": self.default_image_background,
            "image_format": self.default_image_format,
            "image_compression": self.default_image_compression,
            "input_fidelity": self.default_input_fidelity,
            
            # Vision
            "detail_level": self.default_detail_level,
            
            # Streaming
            "enable_streaming": self.enable_streaming,
            "slack_streaming": self.slack_streaming,
            "streaming_update_interval": self.streaming_update_interval,
            "streaming_min_interval": self.streaming_min_interval,
            "streaming_max_interval": self.streaming_max_interval,
            "streaming_buffer_size": self.streaming_buffer_size,
            "streaming_circuit_breaker_threshold": self.streaming_circuit_breaker_threshold,
            "streaming_circuit_breaker_cooldown": self.streaming_circuit_breaker_cooldown,

            # MCP
            "enable_mcp": self.mcp_enabled_default,
        }
        
        # Apply user preferences if available
        if user_id and db:
            try:
                user_prefs = db.get_user_preferences(user_id)
                if user_prefs:
                    # Map database fields to config keys
                    user_config = {}
                    
                    # Model and generation settings
                    if user_prefs.get('model'):
                        user_config['model'] = user_prefs['model']
                    if user_prefs.get('reasoning_effort'):
                        user_config['reasoning_effort'] = user_prefs['reasoning_effort']
                    if user_prefs.get('verbosity'):
                        user_config['verbosity'] = user_prefs['verbosity']
                    if user_prefs.get('temperature') is not None:
                        user_config['temperature'] = user_prefs['temperature']
                    if user_prefs.get('top_p') is not None:
                        user_config['top_p'] = user_prefs['top_p']
                    
                    # Feature toggles
                    if user_prefs.get('enable_web_search') is not None:
                        user_config['enable_web_search'] = bool(user_prefs['enable_web_search'])
                    if user_prefs.get('enable_mcp') is not None:
                        user_config['enable_mcp'] = bool(user_prefs['enable_mcp'])
                    if user_prefs.get('enable_streaming') is not None:
                        user_config['enable_streaming'] = bool(user_prefs['enable_streaming'])
                        user_config['slack_streaming'] = bool(user_prefs['enable_streaming'])
                    
                    # Image settings
                    if user_prefs.get('image_size'):
                        user_config['image_size'] = user_prefs['image_size']
                    if user_prefs.get('input_fidelity'):
                        user_config['input_fidelity'] = user_prefs['input_fidelity']
                    if user_prefs.get('vision_detail'):
                        user_config['detail_level'] = user_prefs['vision_detail']
                    
                    # Custom instructions
                    if user_prefs.get('custom_instructions'):
                        user_config['custom_instructions'] = user_prefs['custom_instructions']
                    
                    # Apply user config over system defaults
                    config.update(user_config)
            except Exception as e:
                # Log error but continue with defaults
                print(f"Error fetching user preferences: {e}")
        
        # Finally apply thread overrides (highest priority)
        if overrides:
            config.update(overrides)
        
        return config


# Global config instance
config = BotConfig()