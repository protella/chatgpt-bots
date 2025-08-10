"""
Configuration module for Slack Bot V2
Handles all environment variables and default settings
"""
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

load_dotenv()


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
    utility_model: str = field(default_factory=lambda: os.getenv("UTILITY_MODEL", "gpt-5-nano"))
    image_model: str = field(default_factory=lambda: os.getenv("GPT_IMAGE_MODEL", "gpt-image-1"))
    dalle_model: str = field(default_factory=lambda: os.getenv("DALLE_MODEL", "dall-e-3"))
    
    # Default parameters for text generation
    default_temperature: float = field(default_factory=lambda: float(os.getenv("DEFAULT_TEMPERATURE", "1.0")))
    default_max_tokens: int = field(default_factory=lambda: int(os.getenv("DEFAULT_MAX_TOKENS", "4096")))
    default_top_p: float = field(default_factory=lambda: float(os.getenv("DEFAULT_TOP_P", "1.0")))
    
    # GPT-5 specific parameters
    default_reasoning_effort: str = field(default_factory=lambda: os.getenv("DEFAULT_REASONING_EFFORT", "medium"))
    default_verbosity: str = field(default_factory=lambda: os.getenv("DEFAULT_VERBOSITY", "medium"))
    
    # Image generation parameters
    default_image_size: str = field(default_factory=lambda: os.getenv("DEFAULT_IMAGE_SIZE", "1024x1024"))
    default_image_quality: str = field(default_factory=lambda: os.getenv("DEFAULT_IMAGE_QUALITY", "high"))
    default_image_background: str = field(default_factory=lambda: os.getenv("DEFAULT_IMAGE_BACKGROUND", "auto"))
    default_image_format: str = field(default_factory=lambda: os.getenv("DEFAULT_IMAGE_FORMAT", "png"))
    default_image_compression: int = field(default_factory=lambda: int(os.getenv("DEFAULT_IMAGE_COMPRESSION", "85")))
    
    # Vision parameters
    default_detail_level: str = field(default_factory=lambda: os.getenv("DEFAULT_DETAIL_LEVEL", "auto"))
    
    # System behavior (will be overridden by platform-specific prompts)
    default_system_prompt: str = field(default_factory=lambda: os.getenv(
        "DEFAULT_SYSTEM_PROMPT",
        ""  # Empty default, will use prompts.py
    ))
    
    # UI Configuration
    thinking_emoji: str = field(default_factory=lambda: os.getenv("THINKING_EMOJI", ":hourglass_flowing_sand:"))
    slack_slash_command: str = field(default_factory=lambda: os.getenv("SLACK_SLASH_COMMAND", "/chatgpt-config"))
    dalle3_command: str = field(default_factory=lambda: os.getenv("DALLE3_CMD", "/dalle-3"))
    
    # Logging configuration
    log_level: str = field(default_factory=lambda: os.getenv("BOT_LOG_LEVEL", "INFO"))
    slack_log_level: str = field(default_factory=lambda: os.getenv("SLACK_LOG_LEVEL", "INFO"))
    discord_log_level: str = field(default_factory=lambda: os.getenv("DISCORD_LOG_LEVEL", "INFO"))
    utils_log_level: str = field(default_factory=lambda: os.getenv("UTILS_LOG_LEVEL", "INFO"))
    console_logging_enabled: bool = field(default_factory=lambda: os.getenv("CONSOLE_LOGGING_ENABLED", "TRUE").upper() == "TRUE")
    debug_mode: bool = field(default_factory=lambda: os.getenv("DEBUG_MODE", "false").lower() == "true")
    
    # Performance settings
    max_concurrent_threads: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_THREADS", "10")))
    message_timeout: int = field(default_factory=lambda: int(os.getenv("MESSAGE_TIMEOUT", "60")))
    
    def validate(self) -> bool:
        """Validate required configuration"""
        if not self.slack_bot_token:
            raise ValueError("SLACK_BOT_TOKEN is required")
        if not self.slack_app_token:
            raise ValueError("SLACK_APP_TOKEN is required")
        if not self.openai_api_key:
            raise ValueError("OPENAI_KEY is required")
        return True
    
    def get_thread_config(self, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get configuration for a specific thread with optional overrides"""
        config = {
            "model": self.gpt_model,
            "temperature": self.default_temperature,
            "max_tokens": self.default_max_tokens,
            "top_p": self.default_top_p,
            "system_prompt": self.default_system_prompt,
            "reasoning_effort": self.default_reasoning_effort,
            "verbosity": self.default_verbosity,
            "image_size": self.default_image_size,
            "image_quality": self.default_image_quality,
            "detail_level": self.default_detail_level,
        }
        
        if overrides:
            config.update(overrides)
        
        return config


# Global config instance
config = BotConfig()