"""
Configuration module for Slack Bot V2
Handles all environment variables and default settings
"""
import logging
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

load_dotenv()


def _env_list(var_name: str, default: list, sep: str = ",") -> list:
    """Parse a comma-separated env var into a clean list, falling back to `default`.

    Trims whitespace and drops empty entries. If the var is unset or resolves to an
    empty list, the sane `default` is used.
    """
    raw = os.getenv(var_name)
    if raw is None:
        return list(default)
    items = [part.strip() for part in raw.split(sep)]
    items = [item for item in items if item]
    return items or list(default)


# Model knowledge cutoff dates
# Supported models: gpt-5.5 (primary), gpt-5-mini (utility functions only)
MODEL_KNOWLEDGE_CUTOFFS = {
    # GPT-5.5 (August 2025 cutoff, 1.05M context window, released April 23, 2026)
    "gpt-5.5": "August 31, 2025",

    # GPT-5 Mini (utility model only — not user-selectable)
    "gpt-5-mini": "September 30, 2024",

    # Default fallback
    "default": "January 1, 2024"
}


@dataclass
class BotConfig:
    """Central configuration for the Slack bot"""
    
    # Slack credentials
    slack_bot_token: str = field(default_factory=lambda: os.getenv("SLACK_BOT_TOKEN", ""))
    slack_app_token: str = field(default_factory=lambda: os.getenv("SLACK_APP_TOKEN", ""))
    
    # OpenAI credentials
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_KEY", ""))
    
    # Model configuration
    gpt_model: str = field(default_factory=lambda: os.getenv("GPT_MODEL", "gpt-5.5"))
    utility_model: str = field(default_factory=lambda: os.getenv("UTILITY_MODEL", "gpt-5-mini"))
    image_model: str = field(default_factory=lambda: os.getenv("GPT_IMAGE_MODEL", "gpt-image-2"))
    
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
    
    # Image generation parameters (gpt-image-1.5)
    default_image_size: str = field(default_factory=lambda: os.getenv("DEFAULT_IMAGE_SIZE", "1024x1024"))
    default_image_quality: str = field(default_factory=lambda: os.getenv("DEFAULT_IMAGE_QUALITY", "auto"))  # auto, low, medium, high
    default_image_background: str = field(default_factory=lambda: os.getenv("DEFAULT_IMAGE_BACKGROUND", "auto"))  # transparent, opaque, auto
    default_image_number: int = field(default_factory=lambda: int(os.getenv("DEFAULT_IMAGE_NUMBER", "1")))  # Number of images
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
    utils_log_level: str = field(default_factory=lambda: os.getenv("UTILS_LOG_LEVEL", "INFO"))
    console_logging_enabled: bool = field(default_factory=lambda: os.getenv("CONSOLE_LOGGING_ENABLED", "TRUE").upper() == "TRUE")
    log_directory: str = field(default_factory=lambda: os.getenv("LOG_DIRECTORY", "logs"))
    debug_mode: bool = field(default_factory=lambda: os.getenv("DEBUG_MODE", "false").lower() == "true")
    
    # Performance settings
    max_concurrent_threads: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_THREADS", "10")))
    message_timeout: int = field(default_factory=lambda: int(os.getenv("MESSAGE_TIMEOUT", "60")))
    
    # Cleanup settings
    cleanup_schedule: str = field(default_factory=lambda: os.getenv("CLEANUP_SCHEDULE", "0 0 * * 0"))  # Default: midnight Sunday (weekly)
    cleanup_max_age_hours: float = field(default_factory=lambda: float(os.getenv("CLEANUP_MAX_AGE_HOURS", "24")))
    
    # API Timeout settings (in seconds)
    api_timeout_read: float = field(default_factory=lambda: float(os.getenv("API_TIMEOUT_READ", "180")))  # Overall timeout for API requests
    api_timeout_streaming_chunk: float = field(default_factory=lambda: float(os.getenv("API_TIMEOUT_STREAMING_CHUNK", "30")))  # Max time between streaming chunks
    
    # Model token limits
    # GPT-5.5: 1.05M total context window (shared between input, output, and reasoning)
    # GPT-5 Mini (utility): 400k total context window
    # Reserved for output/reasoning/overhead is ~130k (static across models)
    # Buffer percentages are calculated so effective input = total - 130k reserved
    # (env var names kept as GPT54_*/GPT5_* so existing .env files keep working)
    gpt54_max_tokens: int = field(default_factory=lambda: int(os.getenv("GPT54_MAX_TOKENS", "1050000")))  # GPT-5.5 context window
    gpt5_max_tokens: int = field(default_factory=lambda: int(os.getenv("GPT5_MAX_TOKENS", "400000")))  # gpt-5-mini (utility) context window

    # Token management configuration
    # Buffer to leave room for output/reasoning tokens and overhead
    # GPT-5.5: 0.876 = ~920k usable of 1.05M (130k reserved)
    # GPT-5 Mini: 0.675 = ~270k usable of 400k (130k reserved)
    gpt54_token_buffer_percentage: float = field(default_factory=lambda: float(os.getenv("GPT54_TOKEN_BUFFER_PERCENTAGE", "0.876")))
    token_buffer_percentage: float = field(default_factory=lambda: float(os.getenv("TOKEN_BUFFER_PERCENTAGE", "0.875")))
    token_cleanup_threshold: float = field(default_factory=lambda: float(os.getenv("TOKEN_CLEANUP_THRESHOLD", "0.8")))
    token_trim_message_count: int = field(default_factory=lambda: int(os.getenv("TOKEN_TRIM_MESSAGE_COUNT", "5")))
    # Phase S: chunky compaction target — when over the cleanup threshold, compact down to
    # this fraction of the model limit in ONE pass (per-turn micro-trims would bust the
    # OpenAI prefix cache every turn)
    token_compaction_target: float = field(default_factory=lambda: float(os.getenv("TOKEN_COMPACTION_TARGET", "0.7")))


    # Streaming configuration
    enable_streaming: bool = field(default_factory=lambda: os.getenv("ENABLE_STREAMING", "true").lower() == "true")
    slack_streaming: bool = field(default_factory=lambda: os.getenv("SLACK_STREAMING", "true").lower() == "true")
    streaming_update_interval: float = field(default_factory=lambda: float(os.getenv("STREAMING_UPDATE_INTERVAL", "2.0")))
    streaming_min_interval: float = field(default_factory=lambda: float(os.getenv("STREAMING_MIN_INTERVAL", "1.0")))
    streaming_max_interval: float = field(default_factory=lambda: float(os.getenv("STREAMING_MAX_INTERVAL", "30.0")))
    streaming_buffer_size: int = field(default_factory=lambda: int(os.getenv("STREAMING_BUFFER_SIZE", "500")))
    streaming_circuit_breaker_threshold: int = field(default_factory=lambda: int(os.getenv("STREAMING_CIRCUIT_BREAKER_THRESHOLD", "5")))
    streaming_circuit_breaker_cooldown: int = field(default_factory=lambda: int(os.getenv("STREAMING_CIRCUIT_BREAKER_COOLDOWN", "300")))

    # --- Native Slack streaming (Phase 3.1): chat.startStream/appendStream/stopStream ---
    # Replaces the chat.update edit-loop (Tier-3 rate-limited org-wide) with Slack's native
    # streaming API. DEFAULT OFF: the capability is built + unit-tested, but the streamed UX
    # must be verified live on the dev bot before enabling. Flip to true after that.
    slack_native_streaming: bool = field(default_factory=lambda: os.getenv("SLACK_NATIVE_STREAMING", "false").lower() == "true")

    # --- Assistant status indicator (Phase 3.2): assistant.threads.setStatus ---
    # Transient "thinking/working" status on the assistant-thread surface. Additive + best-effort:
    # it no-ops gracefully in plain channels (where the visible progress comes from streaming text).
    enable_assistant_status: bool = field(default_factory=lambda: os.getenv("ENABLE_ASSISTANT_STATUS", "true").lower() == "true")
    # Rotating loading messages shown by setStatus (comma-separated env; safe standard-emoji default).
    # >>> To brand these, set STATUS_LOADING_MESSAGES in .env with REAL Datassential custom-emoji
    # >>> names, e.g.  STATUS_LOADING_MESSAGES=:datassential: crunching the data…,:datassential: digging in…
    status_loading_messages: list = field(default_factory=lambda: _env_list("STATUS_LOADING_MESSAGES", [
        ":mag: crunching the data…",
        ":bar_chart: digging through the menu…",
        ":hourglass_flowing_sand: pulling it together…",
    ]))
    # Safe standard-emoji status used when no rotating set is desired / as the single status string.
    status_loading_fallback: str = field(default_factory=lambda: os.getenv("STATUS_LOADING_FALLBACK", ":hourglass_flowing_sand: working on it…"))

    # --- Assistant surface (agent split-view) adapter ---
    # Greets the user + sets suggested prompts when they open the split view, and titles
    # assistant threads from the first message. Additive/best-effort; messages themselves
    # still flow through the normal DM path.
    enable_assistant_surface: bool = field(default_factory=lambda: os.getenv("ENABLE_ASSISTANT_SURFACE", "true").lower() == "true")
    assistant_greeting: str = field(default_factory=lambda: os.getenv(
        "ASSISTANT_GREETING",
        "👋 Hi! Ask me anything — or pick one of the suggestions below to get started."))
    # Starter prompts shown in the split view (comma-separated env).
    assistant_suggested_prompts: list = field(default_factory=lambda: _env_list("ASSISTANT_SUGGESTED_PROMPTS", [
        "What's trending in food & beverage right now?",
        "Summarize this document for me",
        "Generate an image of …",
    ]))

    # --- Emoji reactions as a response (Phase 4) ---
    enable_reactions: bool = field(default_factory=lambda: os.getenv("ENABLE_REACTIONS", "true").lower() == "true")
    # Vetted emoji the bot is allowed to use as a reaction-response (names, no colons).
    # Env: REACTION_EMOJIS (comma-separated).
    reaction_emojis: list = field(default_factory=lambda: _env_list("REACTION_EMOJIS", [
        "thumbsup", "eyes", "white_check_mark", "raised_hands", "tada", "thinking_face", "+1",
    ]))

    # --- Outbound self-prefix hygiene (Phase 3.4) ---
    # Leading "Name:" prefixes to strip from the model's reply so it never answers as "ChatGPT: …"
    # (other bots now appear as "Name:" user turns in history, which the model may try to mimic).
    # Env: SELF_PREFIX_NAMES (comma-separated).
    self_prefix_names: list = field(default_factory=lambda: _env_list("SELF_PREFIX_NAMES", ["ChatGPT", "ChatGPT-Dev", "Assistant", "Bot"]))

    # --- Channel listening + wake classifier (Phase 5) ---
    # MASTER SWITCH (default OFF): when False the bot only acts on @mentions (app_mention) and
    # DMs, exactly as before — restarting the bot changes nothing. Flip on to let the bot see
    # and (per channel_response_mode) respond to non-mention public/private channel messages.
    enable_channel_listening: bool = field(default_factory=lambda: os.getenv("ENABLE_CHANNEL_LISTENING", "false").lower() == "true")
    # Default response mode for channels (Phase 7 adds per-channel overrides):
    #   "tag_only"     - respond only when clearly addressed (name / reply in our thread); no LLM. DEFAULT.
    #   "auto_respond" - a lightweight classifier decides respond/react/ignore per message.
    #   "off"          - never respond in channels.
    channel_response_mode: str = field(default_factory=lambda: os.getenv("CHANNEL_RESPONSE_MODE", "tag_only").strip().lower())
    # Names the bot answers to without an @mention (case-insensitive whole-word match), so
    # "ChatGPT, can you…" wakes it. Keep in sync with the bot's display name(s).
    # Env: BOT_NAME_ALIASES (comma-separated) — SET THIS per environment (e.g. "ChatGPT-Dev" in dev).
    bot_name_aliases: list = field(default_factory=lambda: _env_list("BOT_NAME_ALIASES", ["ChatGPT"]))
    # Bounded recent-channel-window size for a bare top-level wake (0 = just the triggering
    # message, which already keys as a length-1 thread). Larger values are a documented follow-up.
    channel_context_window: int = field(default_factory=lambda: int(os.getenv("CHANNEL_CONTEXT_WINDOW", "0")))

    # --- Response footer (Phase 7 entry point): a small context line + "⚙️ Configure" button
    # appended under each channel response (any member can open the per-channel settings modal).
    # Posted as a separate trailing message, so it never touches the text/split/streaming path.
    enable_response_footer: bool = field(default_factory=lambda: os.getenv("ENABLE_RESPONSE_FOOTER", "true").lower() == "true")

    # --- Local function-call loop (redesign Phase A) ---
    # Master switch for model-invoked local tools (history fetch, reactions, later search/memory).
    # The loop composes with server-side tools (web_search/MCP) in the same request.
    enable_tool_loop: bool = field(default_factory=lambda: os.getenv("ENABLE_TOOL_LOOP", "true").lower() == "true")
    # Runaway caps: max loop rounds per response / max total local calls per response. On cap,
    # one final round runs with tool_choice="none" so the model answers with what it has.
    max_tool_rounds: int = field(default_factory=lambda: int(os.getenv("MAX_TOOL_ROUNDS", "4")))
    max_tool_calls_per_turn: int = field(default_factory=lambda: int(os.getenv("MAX_TOOL_CALLS_PER_TURN", "8")))
    # Per-executor timeout (seconds); a timed-out tool returns an error result to the model.
    tool_call_timeout: float = field(default_factory=lambda: float(os.getenv("TOOL_CALL_TIMEOUT", "20")))
    # Truncation cap on a single tool result fed back to the model (characters).
    tool_result_max_chars: int = field(default_factory=lambda: int(os.getenv("TOOL_RESULT_MAX_CHARS", "20000")))
    # Model-invoked emoji reactions (redesign Phase D) — allowlist still REACTION_EMOJIS.
    enable_react_tool: bool = field(default_factory=lambda: os.getenv("ENABLE_REACT_TOOL", "true").lower() == "true")

    # --- On-demand Slack history-fetch tools (Phase 8) ---
    # Read-only + privacy-scoped (public or bot-member channels only), so default ON. Wired to
    # the model through the local function-call loop (ENABLE_TOOL_LOOP).
    enable_history_tools: bool = field(default_factory=lambda: os.getenv("ENABLE_HISTORY_TOOLS", "true").lower() == "true")
    # Hard cap on messages returned by a single history-fetch tool call (the model's `limit` is
    # clamped to this regardless of what it asks for).
    history_tool_max_messages: int = field(default_factory=lambda: int(os.getenv("HISTORY_TOOL_MAX_MESSAGES", "50")))

    # --- Per-channel memory (Phase 9) ---
    # Read = inject the channel's durable facts into the system prompt on each response.
    # Write = one lightweight utility-model "is there a durable fact?" call AFTER each response
    # (no function-call loop needed). Off → no injection, no extraction (unchanged behavior).
    enable_channel_memory: bool = field(default_factory=lambda: os.getenv("ENABLE_CHANNEL_MEMORY", "true").lower() == "true")
    # Hard cap on channel-scope memory rows per channel (oldest evicted on overflow) — keeps the
    # injected block small and bounded; prefer update/supersede over unbounded growth.
    memory_max_rows: int = field(default_factory=lambda: int(os.getenv("MEMORY_MAX_ROWS", "25")))

    def get_model_token_limit(self, model: str) -> int:
        """Get the effective input token limit for a specific model

        This returns the maximum number of input tokens we should send.
        For GPT-5.5: 1.05M total - 130k reserved = ~920k usable
        For GPT-5 Mini (utility): 400k total - 130k reserved = ~270k usable

        Args:
            model: Model name (e.g., 'gpt-5.5', 'gpt-5-mini')

        Returns:
            Buffered token limit for safe operation
        """
        if model.startswith('gpt-5.5'):
            return int(self.gpt54_max_tokens * self.gpt54_token_buffer_percentage)
        # gpt-5-mini (utility) and any unknown model: use the conservative 400k window
        return int(self.gpt5_max_tokens * self.token_buffer_percentage)
    
    def validate(self) -> bool:
        """Validate required configuration"""
        if not self.slack_bot_token:
            raise ValueError("SLACK_BOT_TOKEN is required")
        if not self.slack_app_token:
            raise ValueError("SLACK_APP_TOKEN is required")
        if not self.openai_api_key:
            raise ValueError("OPENAI_KEY is required")
        return True
    
    def _default_thread_config(self) -> Dict[str, Any]:
        """System-default thread config (hierarchy level 1)."""
        return {
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
            "image_background": self.default_image_background,
            "image_number": self.default_image_number,
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
        
    @staticmethod
    def _map_user_prefs(user_prefs: Dict[str, Any]) -> Dict[str, Any]:
        """Map database user-preference fields to config keys (hierarchy level 2)."""
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
        if user_prefs.get('image_model'):
            user_config['image_model'] = user_prefs['image_model']
        if user_prefs.get('image_size'):
            user_config['image_size'] = user_prefs['image_size']
        if user_prefs.get('image_quality'):
            user_config['image_quality'] = user_prefs['image_quality']
        if user_prefs.get('image_background'):
            user_config['image_background'] = user_prefs['image_background']
        if user_prefs.get('input_fidelity'):
            user_config['input_fidelity'] = user_prefs['input_fidelity']
        if user_prefs.get('vision_detail'):
            user_config['detail_level'] = user_prefs['vision_detail']

        # Custom instructions
        if user_prefs.get('custom_instructions'):
            user_config['custom_instructions'] = user_prefs['custom_instructions']

        return user_config

    def _compose_thread_config(self, user_prefs: Optional[Dict[str, Any]],
                               overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Compose the config hierarchy: defaults <- user prefs <- thread overrides."""
        config = self._default_thread_config()
        if user_prefs:
            config.update(self._map_user_prefs(user_prefs))
        if overrides:
            config.update(overrides)
        return config

    def get_thread_config(self, overrides: Optional[Dict[str, Any]] = None, user_id: Optional[str] = None, db = None) -> Dict[str, Any]:
        """Get configuration for a specific thread with settings hierarchy:
        1. System defaults (from .env)
        2. User preferences (from database)
        3. Thread overrides (passed as parameter)
        """
        user_prefs = None
        if user_id and db:
            try:
                user_prefs = db.get_user_preferences(user_id)
            except Exception as e:
                logging.getLogger("bot.config").warning(f"Error fetching user preferences: {e}")
        return self._compose_thread_config(user_prefs, overrides)

    async def get_thread_config_async(self, overrides: Optional[Dict[str, Any]] = None,
                                      user_id: Optional[str] = None, db = None) -> Dict[str, Any]:
        """Async get_thread_config — awaits the aiosqlite preference read instead of
        blocking the event loop with sync sqlite on every message."""
        user_prefs = None
        if user_id and db:
            try:
                user_prefs = await db.get_user_preferences_async(user_id)
            except Exception as e:
                logging.getLogger("bot.config").warning(f"Error fetching user preferences: {e}")
        return self._compose_thread_config(user_prefs, overrides)


# Global config instance
config = BotConfig()