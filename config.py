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
    default_temperature: float = field(default_factory=lambda: float(os.getenv("DEFAULT_TEMPERATURE", "0.8")))
    default_max_tokens: int = field(default_factory=lambda: int(os.getenv("DEFAULT_MAX_TOKENS", "4096")))
    default_top_p: float = field(default_factory=lambda: float(os.getenv("DEFAULT_TOP_P", "1.0")))
    
    # GPT-5 specific parameters
    default_reasoning_effort: str = field(default_factory=lambda: os.getenv("DEFAULT_REASONING_EFFORT", "medium"))
    default_verbosity: str = field(default_factory=lambda: os.getenv("DEFAULT_VERBOSITY", "medium"))
    
    # Image generation parameters
    default_image_size: str = field(default_factory=lambda: os.getenv("DEFAULT_IMAGE_SIZE", "1024x1024"))
    default_image_quality: str = field(default_factory=lambda: os.getenv("DEFAULT_IMAGE_QUALITY", "hd"))  # standard or hd
    default_image_style: str = field(default_factory=lambda: os.getenv("DEFAULT_IMAGE_STYLE", "natural"))  # natural or vivid
    default_image_number: int = field(default_factory=lambda: int(os.getenv("DEFAULT_IMAGE_NUMBER", "1")))  # Number of images (1 for DALL-E 3)
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
            "dalle_model": self.dalle_model,
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
        }
        
        if overrides:
            config.update(overrides)
        
        return config


# Global config instance
config = BotConfig()