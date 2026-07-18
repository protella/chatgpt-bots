"""
Configuration module for Slack Bot V2
Handles all environment variables and default settings
"""
import logging
import os
import random
import re
from functools import lru_cache
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
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


# F20: syntactic gate for a Slack emoji shorthand name (no colons). Any name that passes
# is offered to Slack's reactions.add, whose own invalid_name error is the semantic
# backstop; this only rejects obvious garbage so a malformed model arg never hits the API.
_EMOJI_NAME_RE = re.compile(r"^[a-z0-9_+'-]{1,64}$")


def valid_emoji_name(name: str) -> bool:
    """True if `name` is a syntactically plausible Slack emoji shorthand (lowercase name
    charset, sane length; no colons)."""
    return bool(name) and bool(_EMOJI_NAME_RE.match(name))


def _resolve_repo_path(path: str) -> str:
    """Resolve a relative path against the repo root (this file's directory)."""
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


@lru_cache(maxsize=8)
def _load_message_file(path: str) -> tuple:
    """Load a message file: one message per line, '#' comments and blanks skipped.

    Returns a tuple (hashable for the cache); empty on any read problem — callers
    fall back to their defaults, a missing file must never break the bot.
    """
    try:
        with open(_resolve_repo_path(path), encoding="utf-8") as f:
            return tuple(
                line.strip() for line in f
                if line.strip() and not line.strip().startswith("#")
            )
    except OSError:
        return ()


@lru_cache(maxsize=8)
def _load_stage_map(path: str) -> dict:
    """Parse a [stage]-sectioned message file into {stage: (variants...)}."""
    stages: Dict[str, list] = {}
    current: Optional[str] = None
    for line in _load_message_file(path):
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            stages.setdefault(current, [])
        elif current:
            stages[current].append(line)
    return {k: tuple(v) for k, v in stages.items() if v}


def pipeline_status_markers() -> tuple:
    """Every pipeline variant text, for recognizing our own transient status lines
    when rebuilding history (templated variants truncated at their first
    placeholder; very short prefixes dropped — too generic to match on).
    """
    try:
        stage_map = _load_stage_map(config.pipeline_messages_file)
        markers = (
            v.split("{", 1)[0].strip() if "{" in v else v
            for variants in stage_map.values() for v in variants
        )
        return tuple(m for m in markers if len(m) >= 8)
    except Exception:
        return ()


def pipeline_status(stage: str, default: str, **fmt) -> str:
    """Pick a random status variant for a pipeline stage.

    Variants come from config.pipeline_messages_file ([stage] sections); `fmt`
    fills {placeholders} in the chosen variant. Any problem — unknown stage,
    unreadable file, placeholder mismatch — falls back to `default` (which the
    caller passes already formatted).
    """
    try:
        variants = _load_stage_map(config.pipeline_messages_file).get(stage)
        if not variants:
            return default
        text = random.choice(variants)
        return text.format(**fmt) if fmt else text
    except Exception:
        return default


# Model knowledge cutoff dates
# Supported models: gpt-5.6-sol (default), gpt-5.6-terra, gpt-5.6-luna, gpt-5.5
# (gpt-5.6-luna doubles as the utility model)
MODEL_KNOWLEDGE_CUTOFFS = {
    # GPT-5.6 family (Feb 2026 cutoff, 1.05M context window, released July 9, 2026)
    "gpt-5.6-sol": "February 16, 2026",
    "gpt-5.6-terra": "February 16, 2026",
    "gpt-5.6-luna": "February 16, 2026",

    # GPT-5.5 (August 2025 cutoff, 1.05M context window, released April 23, 2026)
    "gpt-5.5": "August 31, 2025",

    # Default fallback
    "default": "January 1, 2024"
}

# The full user-selectable model set (order = modal display order)
SUPPORTED_CHAT_MODELS = ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5"]

# Reasoning-effort ladders per model family (verified live 2026-07-09:
# `max` returns 200 on ALL three 5.6 tiers; `minimal` 400s on all of them)
GPT56_EFFORTS = ["none", "low", "medium", "high", "xhigh", "max"]
GPT55_EFFORTS = ["none", "low", "medium", "high", "xhigh"]


def clamp_effort(model: str, effort: Optional[str]) -> str:
    """Coerce a stored/legacy reasoning effort into one the model accepts.

    Guarantees bad stored settings can never reach the API:
    - 5.6 family: `minimal` is unsupported (400) -> `none`; full ladder incl. `max`.
    - gpt-5.5 / gpt-5-mini and anything else: `max` doesn't exist -> `xhigh`;
      `minimal` stays valid on gpt-5-mini and maps to `low` on gpt-5.5 (its modal
      never offered minimal).
    Unknown values fall back to `medium`.
    """
    effort = (effort or "medium").lower()
    if model.startswith("gpt-5.6"):
        if effort == "minimal":
            return "none"
        return effort if effort in GPT56_EFFORTS else "medium"
    if effort == "max":
        return "xhigh"
    if effort == "minimal" and model.startswith("gpt-5.5"):
        return "low"
    valid = GPT55_EFFORTS + ["minimal"]
    return effort if effort in valid else "medium"


@dataclass
class BotConfig:
    """Central configuration for the Slack bot"""
    
    # Slack credentials
    slack_bot_token: str = field(default_factory=lambda: os.getenv("SLACK_BOT_TOKEN", ""))
    slack_app_token: str = field(default_factory=lambda: os.getenv("SLACK_APP_TOKEN", ""))
    
    # OpenAI credentials
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_KEY", ""))
    
    # Model configuration
    gpt_model: str = field(default_factory=lambda: os.getenv("GPT_MODEL", "gpt-5.6-sol"))
    utility_model: str = field(default_factory=lambda: os.getenv("UTILITY_MODEL", "gpt-5.6-luna"))
    image_model: str = field(default_factory=lambda: os.getenv("GPT_IMAGE_MODEL", "gpt-image-2"))
    
    # Default parameters for text generation
    default_temperature: float = field(default_factory=lambda: float(os.getenv("DEFAULT_TEMPERATURE", "0.8")))
    default_max_tokens: int = field(default_factory=lambda: int(os.getenv("DEFAULT_MAX_TOKENS", "32768")))
    default_top_p: float = field(default_factory=lambda: float(os.getenv("DEFAULT_TOP_P", "1.0")))
    
    # GPT-5 specific parameters
    default_reasoning_effort: str = field(default_factory=lambda: os.getenv("DEFAULT_REASONING_EFFORT", "medium"))
    default_verbosity: str = field(default_factory=lambda: os.getenv("DEFAULT_VERBOSITY", "medium"))
    
    # Utility function parameters (for quick checks, image intent, etc.)
    # `none` = zero reasoning tokens — right default for classifiers (5.6 dropped `minimal`)
    utility_reasoning_effort: str = field(default_factory=lambda: os.getenv("UTILITY_REASONING_EFFORT", "none"))
    utility_verbosity: str = field(default_factory=lambda: os.getenv("UTILITY_VERBOSITY", "low"))
    utility_max_tokens: int = field(default_factory=lambda: int(os.getenv("UTILITY_MAX_TOKENS", "20")))
    # Participation judgment gets its own effort: referent resolution ("is 'you' me
    # or the other agent?") reliably fails at `none`, and this call sits behind the
    # debounce window — not on the response critical path — so `low` costs nothing
    # the user can feel. Verified live 2026-07-10: none = 0/3, low = 3/3 on the
    # mid-exchange follow-up case.
    participation_reasoning_effort: str = field(default_factory=lambda: os.getenv("PARTICIPATION_REASONING_EFFORT", "low"))

    # F40 — the wake gate SEES attached images (user report 2026-07-13: a meme captioned only
    # ":dogkek:" earned a :joy: reaction the gate had inferred from the emoji in the caption,
    # never having looked at the picture). All the context can live in the image, so the gate
    # gets the pixels, not a filename. Deliberately NOT gated on "thin text": a long caption
    # ("this is exactly what prod does every Friday") is just as meaningless without the image,
    # and a text-only first pass would keep the reported bug — that pass answered confidently.
    #
    # Cost control is by CAPS, not by guessing: low detail, few images, hard byte ceiling.
    # Raising reasoning effort would NOT buy visual resolution — that's what `detail` is for.
    enable_multimodal_gate: bool = field(default_factory=lambda: os.getenv("ENABLE_MULTIMODAL_GATE", "true").lower() == "true")
    gate_vision_max_images: int = field(default_factory=lambda: int(os.getenv("GATE_VISION_MAX_IMAGES", "2")))
    gate_vision_max_bytes: int = field(default_factory=lambda: int(os.getenv("GATE_VISION_MAX_BYTES", str(5 * 1024 * 1024))))
    gate_vision_detail: str = field(default_factory=lambda: os.getenv("GATE_VISION_DETAIL", "low"))

    # F51 — Ambient memory. Images/links/files posted in a channel or thread are looked at,
    # summarized, and kept as derived artifacts in the running context even when the bot does
    # NOT respond ("nothing seen is forgotten"). Master kill switch below; participation `off`
    # does NOT mean memory-off (they are distinct settings — see per-channel opt-out).
    enable_ambient_memory: bool = field(default_factory=lambda: os.getenv("ENABLE_AMBIENT_MEMORY", "true").lower() == "true")
    # Sub-switches: image capture rides the gate-vision call (near-zero cost) or a detached
    # worker; link fetch opens URLs (SSRF-hardened); file summaries run bounded extraction.
    enable_ambient_image_memory: bool = field(default_factory=lambda: os.getenv("ENABLE_AMBIENT_IMAGE_MEMORY", "true").lower() == "true")
    enable_link_fetch: bool = field(default_factory=lambda: os.getenv("ENABLE_LINK_FETCH", "true").lower() == "true")
    enable_ambient_file_memory: bool = field(default_factory=lambda: os.getenv("ENABLE_AMBIENT_FILE_MEMORY", "true").lower() == "true")
    # fetch_url is the model-callable half of the SAME hardened fetcher (a directly-asked
    # "read this link" opens it instead of relying on web_search luck). Not a free tool.
    enable_fetch_url_tool: bool = field(default_factory=lambda: os.getenv("ENABLE_FETCH_URL_TOOL", "true").lower() == "true")

    # Bounds — every cap that fires persists an honest `omitted`/`failed` status, never a silent drop.
    ambient_queue_capacity: int = field(default_factory=lambda: int(os.getenv("AMBIENT_QUEUE_CAPACITY", "256")))
    ambient_fetch_workers: int = field(default_factory=lambda: int(os.getenv("AMBIENT_FETCH_WORKERS", "2")))
    ambient_vision_workers: int = field(default_factory=lambda: int(os.getenv("AMBIENT_VISION_WORKERS", "1")))
    ambient_document_workers: int = field(default_factory=lambda: int(os.getenv("AMBIENT_DOCUMENT_WORKERS", "1")))
    ambient_max_links_per_message: int = field(default_factory=lambda: int(os.getenv("AMBIENT_MAX_LINKS_PER_MESSAGE", "2")))
    ambient_max_images_per_message: int = field(default_factory=lambda: int(os.getenv("AMBIENT_MAX_IMAGES_PER_MESSAGE", "4")))
    ambient_max_files_per_message: int = field(default_factory=lambda: int(os.getenv("AMBIENT_MAX_FILES_PER_MESSAGE", "3")))
    ambient_summary_max_chars: int = field(default_factory=lambda: int(os.getenv("AMBIENT_SUMMARY_MAX_CHARS", "600")))
    ambient_extract_max_chars: int = field(default_factory=lambda: int(os.getenv("AMBIENT_EXTRACT_MAX_CHARS", "16000")))
    # Link fetch caps. Bytes ceiling is SMALLER than the addressed-document 50MB ceiling.
    link_fetch_max_bytes: int = field(default_factory=lambda: int(os.getenv("LINK_FETCH_MAX_BYTES", str(2 * 1024 * 1024))))
    link_fetch_connect_timeout_s: float = field(default_factory=lambda: float(os.getenv("LINK_FETCH_CONNECT_TIMEOUT_S", "5")))
    link_fetch_read_timeout_s: float = field(default_factory=lambda: float(os.getenv("LINK_FETCH_READ_TIMEOUT_S", "8")))
    link_fetch_total_timeout_s: float = field(default_factory=lambda: float(os.getenv("LINK_FETCH_TOTAL_TIMEOUT_S", "12")))
    link_fetch_max_redirects: int = field(default_factory=lambda: int(os.getenv("LINK_FETCH_MAX_REDIRECTS", "5")))
    # Ambient file byte ceiling (streamed pre-download gate) — much smaller than addressed docs.
    ambient_file_max_bytes: int = field(default_factory=lambda: int(os.getenv("AMBIENT_FILE_MAX_BYTES", str(8 * 1024 * 1024))))
    ambient_artifact_retention_days: int = field(default_factory=lambda: int(os.getenv("AMBIENT_ARTIFACT_RETENTION_DAYS", "30")))
    # Link summaries older than this re-fetch on next sighting (staleness window).
    ambient_link_stale_days: int = field(default_factory=lambda: int(os.getenv("AMBIENT_LINK_STALE_DAYS", "7")))

    # Analysis function parameters (for vision analysis, complex tasks)
    analysis_reasoning_effort: str = field(default_factory=lambda: os.getenv("ANALYSIS_REASONING_EFFORT", "medium"))
    analysis_verbosity: str = field(default_factory=lambda: os.getenv("ANALYSIS_VERBOSITY", "medium"))
    vision_max_tokens: int = field(default_factory=lambda: int(os.getenv("VISION_MAX_TOKENS", "8192")))
    
    # Image generation parameters (gpt-image-2 / gpt-image-1)
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
    # Reserved auxiliary indicator (kept configurable for future use; core status
    # flow uses circle_loader_emoji)
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

    # Cleanup settings
    cleanup_schedule: str = field(default_factory=lambda: os.getenv("CLEANUP_SCHEDULE", "0 0 * * 0"))  # Default: midnight Sunday (weekly)
    cleanup_max_age_hours: float = field(default_factory=lambda: float(os.getenv("CLEANUP_MAX_AGE_HOURS", "24")))
    
    # API Timeout settings (in seconds)
    api_timeout_read: float = field(default_factory=lambda: float(os.getenv("API_TIMEOUT_READ", "180")))  # Overall timeout for API requests
    api_timeout_streaming_chunk: float = field(default_factory=lambda: float(os.getenv("API_TIMEOUT_STREAMING_CHUNK", "30")))  # Max time between streaming chunks
    # Image generation/edit can legitimately run past the general read timeout. Applied
    # BOTH as the outer asyncio.wait_for and as a per-request SDK timeout (the AsyncOpenAI
    # client is built with api_timeout_read, so wait_for alone can't extend past it).
    api_timeout_image: float = field(default_factory=lambda: float(os.getenv("API_TIMEOUT_IMAGE", "300")))
    # --- Socket-liveness monitor (F9, detection-only) ---
    # Seconds without ANY inbound Socket Mode envelope before the liveness monitor speaks
    # up: if slack_sdk's ping-pong is ALSO frozen for the same window it logs an ERROR
    # (unambiguous half-open death — restart likely required); if pings are still fresh it
    # logs ONE WARNING per drought episode (idle or half-open, passively indistinguishable).
    # Detection only — the monitor NEVER touches the socket. 0 disables the monitor.
    socket_liveness_timeout: int = field(default_factory=lambda: int(os.getenv("SOCKET_LIVENESS_TIMEOUT", "600")))

    # Model token limits (verified 2026-07-09 against developers.openai.com/api/docs/models/*)
    # GPT-5.6 family (sol/terra/luna) AND GPT-5.5: 1,050,000 total context window,
    #   128,000 max output tokens (window is shared between input, output, and reasoning)
    # gpt-5-mini (legacy/fallback only): 400,000 total / 128,000 max output
    # (env var names kept as GPT54_*/GPT5_* so existing .env files keep working)
    gpt54_max_tokens: int = field(default_factory=lambda: int(os.getenv("GPT54_MAX_TOKENS", "1050000")))  # 5.6-family + 5.5 context window
    gpt5_max_tokens: int = field(default_factory=lambda: int(os.getenv("GPT5_MAX_TOKENS", "400000")))  # fallback window for unknown/legacy models

    # Token management configuration
    # Reserve formula (documented here, enforced by tests/unit/test_context_windows.py):
    #   usable input = window * buffer_pct; reserve = window - usable input.
    #   The reserve must cover: the configured output cap (DEFAULT_MAX_TOKENS /
    #   VISION_MAX_TOKENS, 32,768 each — reasoning tokens bill inside that output
    #   budget), chars/4 estimator error on large threads (~5-8%), and per-request
    #   overhead (system prompt growth, tool schemas/results, doc summary expansion).
    # 1.05M window: 0.876 → ~919.8k usable, ~130.2k reserved (32.8k output + ~97k headroom)
    # 400k fallback: 0.875 → 350k usable, 50k reserved (32.8k output + ~17k headroom)
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
    # --- Edit-in-place progress checklist (F4) ---
    # Accumulating "✓ done / loader active" checklist on the image-pipeline status
    # message instead of a single replaced status line + rotating "still working"
    # strings. Off → today's single-line _update_status/rotator behavior.
    enable_progress_checklist: bool = field(default_factory=lambda: os.getenv("ENABLE_PROGRESS_CHECKLIST", "true").lower() == "true")
    # Image-pipeline checklists post a real visible thread message even where Slack's
    # assistant-status surface is available (dual display: persistent checklist message +
    # live composer status). Off → status-only surfaces degrade to the composer status line
    # alone (the pre-2026-07 behavior). Only meaningful when enable_progress_checklist is on.
    progress_checklist_prefer_message: bool = field(default_factory=lambda: os.getenv("PROGRESS_CHECKLIST_PREFER_MESSAGE", "false").lower() == "true")
    # --- Wake envelope (F3) ---
    # Compact "[Wake context]" block in the volatile developer suffix telling the model
    # WHY it woke (trigger reason, sender role, bot-vs-human). Text-handler turns only;
    # never in the system prompt or history. Off → suffix unchanged.
    enable_wake_envelope: bool = field(default_factory=lambda: os.getenv("ENABLE_WAKE_ENVELOPE", "true").lower() == "true")
    # F13 — how many background image generations may run concurrently in ONE thread.
    # Enforced by the generate_image tool (message_processor/image_tools.py), which returns an
    # "at_capacity" error the model relays in its own words. Per-thread; there is deliberately
    # NO global cap (F14).
    max_concurrent_image_generations: int = field(default_factory=lambda: int(os.getenv("MAX_CONCURRENT_IMAGE_GENERATIONS", "5")))
    # Show the model-rewritten ("enhanced") image prompt as a caption under the image. The
    # enhancement ALWAYS runs — this only decides whether the user has to read it. Off by
    # default: it used to be posted as its own block above every image, which was noise.
    show_enhanced_prompt: bool = field(default_factory=lambda: os.getenv("SHOW_ENHANCED_PROMPT", "false").lower() == "true")
    # F36: Slack canvases (create/read/edit/list). A canvas is the right home for something the
    # thread keeps returning to — a spec, a checklist — where a message gets buried and a file
    # forks into _final_v3. Needs canvases:read + canvases:write.
    enable_canvas_tools: bool = field(default_factory=lambda: os.getenv("ENABLE_CANVAS_TOOLS", "true").lower() == "true")
    # Canvas DELETION is irreversible and public. It is withheld on unprompted turns regardless
    # (the model must never tidy up a channel it was only listening in) — this flag turns it off
    # entirely, even when asked directly.
    enable_canvas_delete: bool = field(default_factory=lambda: os.getenv("ENABLE_CANVAS_DELETE", "true").lower() == "true")
    # Rotating loading messages shown in the assistant thread's transient bubble.
    # Sourced from a message file (one per line, # comments ok) — see
    # status_messages/loading_messages.generic.txt. Brand them by pointing
    # STATUS_LOADING_MESSAGES_FILE at your own file; an explicit inline
    # STATUS_LOADING_MESSAGES (comma-separated) takes precedence over the file.
    # NOTE: the surface renders PLAIN TEXT — no emoji/:shortcodes: (known
    # shortcodes auto-convert to Unicode, unknown ones are stripped).
    status_loading_messages_file: str = field(default_factory=lambda: os.getenv(
        "STATUS_LOADING_MESSAGES_FILE", "status_messages/loading_messages.generic.txt"))
    status_loading_messages_inline: bool = field(default_factory=lambda: bool(os.getenv("STATUS_LOADING_MESSAGES")))
    status_loading_messages: list = field(default_factory=lambda: _env_list("STATUS_LOADING_MESSAGES", [
        "crunching the data…",
        "connecting the dots…",
        "pulling it together…",
    ]))
    # Last-resort single status text when no pool resolves (plain text, no emoji).
    status_loading_fallback: str = field(default_factory=lambda: os.getenv("STATUS_LOADING_FALLBACK", "working on it…"))
    # Stage-keyed pipeline status variants ([stage] sections, one variant per line).
    pipeline_messages_file: str = field(default_factory=lambda: os.getenv(
        "PIPELINE_MESSAGES_FILE", "status_messages/pipeline_messages.txt"))

    def get_loading_messages(self) -> list:
        """Resolve the loading-message pool: inline env wins, then the file, then defaults."""
        if not self.status_loading_messages_inline and self.status_loading_messages_file:
            msgs = _load_message_file(self.status_loading_messages_file)
            if msgs:
                return list(msgs)
        return self.status_loading_messages

    def random_loading_message(self) -> str:
        """One random pick from the loading pool (fallback text if the pool is empty).
        Placeholder messages use this so every waiting surface draws from the same
        variance pool as the native status."""
        msgs = self.get_loading_messages()
        return random.choice(msgs) if msgs else self.status_loading_fallback

    # --- Assistant surface (agent split-view) adapter ---
    # Greets the user + sets suggested prompts when they open the split view, and titles
    # assistant threads from the first message. Additive/best-effort; messages themselves
    # still flow through the normal DM path.
    enable_assistant_surface: bool = field(default_factory=lambda: os.getenv("ENABLE_ASSISTANT_SURFACE", "true").lower() == "true")
    # Keep this copy honest: on the agent_view surface, suggested prompts only render
    # when the manifest's agent_view.suggested_prompts is filled (app_home_opened has
    # no thread_ts, so they can't be set dynamically there).
    assistant_greeting: str = field(default_factory=lambda: os.getenv(
        "ASSISTANT_GREETING",
        "👋 Hi! Ask me anything to get started."))
    # Starter prompts shown in the split view (comma-separated env).
    assistant_suggested_prompts: list = field(default_factory=lambda: _env_list("ASSISTANT_SUGGESTED_PROMPTS", [
        "What's trending in food & beverage right now?",
        "Summarize this document for me",
        "Generate an image of …",
    ]))

    # --- Emoji reactions as a response (Phase 4) ---
    enable_reactions: bool = field(default_factory=lambda: os.getenv("ENABLE_REACTIONS", "true").lower() == "true")
    # F20: OPTIONAL reaction allowlist (names, no colons). Default EMPTY = unrestricted —
    # the bot may pick any standard Slack emoji (picking the right one is the judgment).
    # When set via REACTION_EMOJIS, it is honored everywhere as an allowlist (tool-schema
    # enum, executor, classifier verdict) for workspaces wanting brand control.
    reaction_emojis: list = field(default_factory=lambda: _env_list("REACTION_EMOJIS", []))
    # F6: max distinct emoji the bot may place on a single message. Guards against
    # over-reaction while still letting a user who asks for several get several.
    reaction_max_per_message: int = field(default_factory=lambda: int(os.getenv("REACTION_MAX_PER_MESSAGE", "4")))
    # C1/C6: workspace CUSTOM-emoji surfacing. The bot fetches emoji.list once at startup and
    # refreshes lazily; the names become extra choices for the classifier and the react tool
    # (only when REACTION_EMOJIS is empty — a set allowlist is the exact hard constraint and
    # customs are never injected over it). No explicit off-switch: absent the emoji:read scope
    # the fetch fails soft and the name set simply stays empty.
    workspace_emoji_ttl_seconds: int = field(default_factory=lambda: int(os.getenv("WORKSPACE_EMOJI_TTL_SECONDS", "3600")))
    # Deterministic sorted cap of custom names fed to the participation classifier as signals.
    participation_custom_emoji_cap: int = field(default_factory=lambda: int(os.getenv("PARTICIPATION_CUSTOM_EMOJI_CAP", "32")))
    # Cap of custom names listed in the react_to_message tool-schema description (also budgeted
    # by a per-request char budget so surfacing customs never bloats every main-model request).
    react_tool_custom_emoji_cap: int = field(default_factory=lambda: int(os.getenv("REACT_TOOL_CUSTOM_EMOJI_CAP", "64")))

    # --- Outbound self-prefix hygiene (Phase 3.4) ---
    # Leading "Name:" prefixes to strip from the model's reply so it never answers as "ChatGPT: …"
    # (other bots now appear as "Name:" user turns in history, which the model may try to mimic).
    # Env: SELF_PREFIX_NAMES (comma-separated).
    self_prefix_names: list = field(default_factory=lambda: _env_list("SELF_PREFIX_NAMES", ["ChatGPT", "ChatGPT-Dev", "Assistant", "Bot"]))

    # --- Dev/test harness only ---
    # bot_ids whose messages classify as HUMAN. Posts made via a user token (xoxp) carry the
    # app's bot_id/app_id, so the live-test harness — which posts as a real user — reads as
    # a bot everywhere sender type matters (participation judgment, edit-triggered replies).
    # Listing that bot_id here restores the truth: those posts ARE a human's. NEVER set in prod.
    # Env: DEV_TREAT_BOT_IDS_AS_HUMAN (comma-separated).
    dev_treat_bot_ids_as_human: list = field(default_factory=lambda: _env_list("DEV_TREAT_BOT_IDS_AS_HUMAN", []))

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
    # Default for channels with no explicit setting: may the bot answer a top-level
    # message at channel level? (The engine still judges per message; a channel's
    # saved setting overrides this.)
    reply_in_channel_default: bool = field(default_factory=lambda: os.getenv("REPLY_IN_CHANNEL_DEFAULT", "true").lower() == "true")
    # F46: judgment-call placement for MENTIONS/name-wakes (which run no participation gate and
    # so carry no placement verdict). When on, a top-level public-channel mention that allows
    # top-level replies gets one lean utility-model call deciding thread vs channel — so a
    # deliberately-requested long-form deliverable ("write me a 3-paragraph story") threads even
    # when no tool runs, like the Claude bot. DEFAULT OFF: inert until validated live on the dev
    # bot; off ⇒ this block is skipped entirely (zero added latency/cost, zero behavior change).
    enable_mention_placement_model: bool = field(default_factory=lambda: os.getenv("ENABLE_MENTION_PLACEMENT_MODEL", "false").lower() == "true")
    # Names the bot answers to without an @mention (case-insensitive whole-word match), so
    # "ChatGPT, can you…" wakes it. Keep in sync with the bot's display name(s).
    # Env: BOT_NAME_ALIASES (comma-separated) — SET THIS per environment (e.g. "ChatGPT-Dev" in dev).
    bot_name_aliases: list = field(default_factory=lambda: _env_list("BOT_NAME_ALIASES", ["ChatGPT"]))
    # 👍/👎 feedback buttons under DM/assistant responses (Phase H; channels use reactions)
    enable_feedback_buttons: bool = field(default_factory=lambda: os.getenv("ENABLE_FEEDBACK_BUTTONS", "true").lower() == "true")
    # --- ChannelPulse ambient awareness (redesign Phase E) ---
    # Per-channel in-memory ring of recent messages (fed by every channel event, even ignored
    # ones). Powers the wake-classifier context signal and the response envelope. Inert while
    # ENABLE_CHANNEL_LISTENING is false (no channel events arrive). Supersedes the old
    # CHANNEL_CONTEXT_WINDOW follow-up.
    enable_channel_pulse: bool = field(default_factory=lambda: os.getenv("ENABLE_CHANNEL_PULSE", "true").lower() == "true")
    channel_pulse_size: int = field(default_factory=lambda: int(os.getenv("CHANNEL_PULSE_SIZE", "60")))
    # Head-first char cap for the channel-activity envelope + thread labels (F14).
    pulse_text_truncate: int = field(default_factory=lambda: int(os.getenv("PULSE_TEXT_TRUNCATE", "500")))
    # Tail-first char cap for the F5 per-thread participation-classifier context (F14).
    pulse_tail_text_truncate: int = field(default_factory=lambda: int(os.getenv("PULSE_TAIL_TEXT_TRUNCATE", "500")))
    # Max "[Recent channel activity]" lines injected (at the SUFFIX — volatile, cache hygiene)
    # when responding in a channel. 0 disables the envelope without disabling the buffer.
    channel_pulse_envelope_max: int = field(default_factory=lambda: int(os.getenv("CHANNEL_PULSE_ENVELOPE_MAX", "15")))
    # F5: per-thread tail ring for the participation classifier. The pulse keeps the last
    # N messages of each active thread (their last N chars, sender-typed) so the wake
    # judge can resolve who "you" addresses. 0 disables recording + the signal. F17: 15
    # (busy threads out-chatter 6 lines — match the envelope).
    participation_thread_tail: int = field(default_factory=lambda: int(os.getenv("PARTICIPATION_THREAD_TAIL", "15")))
    # F47: a classifier-only "channel addressee tail" for TOP-LEVEL triggers, which have an
    # empty thread tail and so no authoritative record of who was being addressed. Renders the
    # last N channel-ring messages (top-level AND threaded), sender-typed, so the wake judge can
    # resolve who a bare "you" continues an exchange with. 0 disables the signal.
    participation_addressee_tail: int = field(default_factory=lambda: int(os.getenv("PARTICIPATION_ADDRESSEE_TAIL", "8")))
    # Max distinct threads whose tails are retained per channel (whole-thread LRU eviction).
    pulse_thread_tails_max: int = field(default_factory=lambda: int(os.getenv("PULSE_THREAD_TAILS_MAX", "50")))
    # Global bound on how many channels retain thread-tail rings (outer-map LRU).
    pulse_thread_tail_channels_max: int = field(default_factory=lambda: int(os.getenv("PULSE_THREAD_TAIL_CHANNELS_MAX", "30")))

    # --- Response footer (Phase 7 entry point): a small context line + "⚙️ Configure" button
    # appended under each channel response (any member can open the per-channel settings modal).
    # Posted as a separate trailing message, so it never touches the text/split/streaming path.
    enable_response_footer: bool = field(default_factory=lambda: os.getenv("ENABLE_RESPONSE_FOOTER", "true").lower() == "true")

    # --- ParticipationEngine (redesign Phase F) ---
    # Judgment layer for UNPROMPTED channel participation (judicious/active channels).
    # Replaces the wake classifier. When false, unaddressed channel messages are ignored
    # without any model call (every channel behaves like mentions_only/tag_only);
    # @mentions, name-wakes, 1:1 threads, and DMs are unaffected either way.
    enable_participation_engine: bool = field(default_factory=lambda: os.getenv("ENABLE_PARTICIPATION_ENGINE", "true").lower() == "true")
    # Rapid-fire messages in the same channel within this window collapse into ONE engine
    # evaluation of the latest state (someone typing four short lines ≠ four verdicts).
    participation_debounce_seconds: float = field(default_factory=lambda: float(os.getenv("PARTICIPATION_DEBOUNCE_SECONDS", "3")))
    # F52: an EDIT to a recent human message can also drive a reply. A forgotten @mention ADDED
    # by an edit routes as an addressed wake (Slack fires no app_mention for edits); every other
    # channel edit goes through the participation engine's full typo-vs-meaning judgment, so a
    # spelling/format fix stays silent while a real content change (a question added, facts
    # changed, an ask reversed) can respond — with a correction when the bot already answered.
    # DEFAULT OFF (feature-flag convention): the operator env turns it on.
    enable_edit_triggered_replies: bool = field(default_factory=lambda: os.getenv("ENABLE_EDIT_TRIGGERED_REPLIES", "false").lower() == "true")
    # Only edits of messages younger than this (age from the ORIGINAL post time, not the edit
    # time) are ever considered — an edit to last week's message never re-triggers.
    edit_reply_window_minutes: int = field(default_factory=lambda: int(os.getenv("EDIT_REPLY_WINDOW_MINUTES", "60")))
    # F17: the hourly-cap hard rail is gone. Unprompted replies are still counted and fed to
    # the classifier as a signal, but pacing is the model's judgment, not a numeric ceiling —
    # MAX_UNPROMPTED_REPLIES_PER_HOUR is retired (frontier models don't run away unless asked).
    # Participation-backoff redesign: a "backoff" verdict is no longer one blunt action. The
    # taxonomy routes each case — a standing per-CHANNEL preference is a channel-memory marker;
    # a thread-scoped "stop replying here" is guidance for the current message only and persists
    # NOTHING (the per-thread mute table was removed); a momentary "not now" likewise persists
    # nothing; and an explicit channel-settings change goes through the gated
    # set_channel_participation tool. The acknowledgement reaction is CONDITIONAL now (driven by
    # the classifier, and never emitted when the feedback is about reactions), not an always-on
    # emoji. SNOOZE_ACK_EMOJI is a retained legacy default; the engine picks the ack per verdict.
    snooze_ack_emoji: str = field(default_factory=lambda: os.getenv("SNOOZE_ACK_EMOJI", "zipper_mouth_face").strip().strip(":"))

    # F19: "I'm looking at it" acknowledgment reaction. When a reply will take real work
    # (attachments, data/MCP lookups, multi-step tools, long-form output), the fast models
    # that already look at every message — the participation classifier (unprompted turns)
    # and the intent classifier (addressed turns) — flag it, and the bot drops this emoji
    # on the triggering message BEFORE the slow work, Claude-Tag style. No timers/thresholds;
    # purely additive (never the turn's response, no accounting); routed through the F6
    # reservation guard; stays after the reply; fails silent.
    enable_ack_reaction: bool = field(default_factory=lambda: os.getenv("ENABLE_ACK_REACTION", "true").lower() == "true")
    ack_reaction_emoji: str = field(default_factory=lambda: os.getenv("ACK_REACTION_EMOJI", "eyes").strip().strip(":"))

    # Phase Q — conversational queueing (busy rejection retired). Messages arriving while a
    # conversation is mid-processing queue and are answered in one batched catch-up turn.
    # How long the finishing turn lingers (still holding the conversation lock) so stragglers
    # join the same batch instead of triggering another turn seconds later.
    queue_drain_linger_seconds: float = field(default_factory=lambda: float(os.getenv("QUEUE_DRAIN_LINGER_SECONDS", "1.0")))
    # Max queued messages composed into ONE catch-up turn; the remainder drains next turn.
    queue_max_batch: int = field(default_factory=lambda: int(os.getenv("QUEUE_MAX_BATCH", "10")))
    # Hard bound on a conversation's pending queue; beyond this, messages are dropped with a
    # log (Slack still has them — the thread is flagged for a transcript refetch).
    queue_max_pending: int = field(default_factory=lambda: int(os.getenv("QUEUE_MAX_PENDING", "25")))

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
    # Model-invoked cross-thread reply tool (F23): post a reply into a DIFFERENT thread in the
    # CURRENT channel (never cross-channel). Default ON; a muted target thread is refused.
    enable_post_to_thread_tool: bool = field(default_factory=lambda: os.getenv("ENABLE_POST_TO_THREAD_TOOL", "true").lower() == "true")
    # --- Explicit no-reply terminal tool (F2) ---
    # Exposes a `no_response_needed` tool on UNPROMPTED (participation-gated) turns so the
    # model can end without posting instead of emitting filler. Off → tool hidden, suffix
    # paragraph absent, behavior as today (the honest-accounting fix is unconditional).
    enable_no_reply_tool: bool = field(default_factory=lambda: os.getenv("ENABLE_NO_REPLY_TOOL", "true").lower() == "true")

    # --- Tool-use provenance (F7) ---
    # Capture the tools the bot invoked on each turn (names + short arg-derived gists,
    # never results/content), persist them keyed by the reply's Slack ts, and reinject a
    # deterministic "[used tools: …]" annotation onto the matching assistant message on
    # warm append + rebuild — so the model can recall its own past tool use instead of
    # confabulating. Off → nothing captured, nothing persisted, no annotation.
    enable_tool_provenance: bool = field(default_factory=lambda: os.getenv("ENABLE_TOOL_PROVENANCE", "true").lower() == "true")

    # --- Tool-result memory for MCP calls (F12, extends F7) ---
    # Additionally capture the OUTPUT text of completed mcp_call items and replay it as a
    # "[tool results: <server> → <digest>]" block after "[used tools: …]", so the model can
    # reuse a prior MCP result (links, figures, report titles) instead of re-querying (and
    # retracting when the re-query misses). MCP outputs only — local Slack-fetch/read_document
    # results never persist. Effective only when enable_tool_provenance is also on (results
    # ride on provenance rows). Off → names-only provenance exactly as F7.
    enable_tool_result_memory: bool = field(default_factory=lambda: os.getenv("ENABLE_TOOL_RESULT_MEMORY", "true").lower() == "true")
    # Per-call digest cap (chars); the output is truncated with a "… [truncated]" marker.
    tool_result_digest_chars: int = field(default_factory=lambda: int(os.getenv("TOOL_RESULT_DIGEST_CHARS", "2000")))
    # Per-turn total digest budget (chars, first-come order); calls past the cap store none.
    tool_result_turn_chars: int = field(default_factory=lambda: int(os.getenv("TOOL_RESULT_TURN_CHARS", "6000")))
    # F16: instead of blindly truncating an overlong MCP output (which can amputate the URL/
    # figure that made it worth keeping), summarize it ONCE at capture time with the utility
    # model (low effort) — compress under tool_result_digest_chars, preserving URLs/titles/
    # dates/figures/IDs verbatim. Off → today's pure truncation. Any summarizer error/timeout/
    # overlong return also falls back to truncation, so the reply pipeline never blocks/raises.
    enable_tool_result_summarization: bool = field(default_factory=lambda: os.getenv("ENABLE_TOOL_RESULT_SUMMARIZATION", "true").lower() == "true")
    # Budget guard: feed the summarizer at most this many leading chars of an output so a
    # pathological (huge) result can't blow up the utility call.
    tool_result_summarize_input_chars: int = field(default_factory=lambda: int(os.getenv("TOOL_RESULT_SUMMARIZE_INPUT_CHARS", "20000")))
    # F14: provenance record shaping (env-backed; boot-constant so rebuild determinism holds).
    # Max tool entries recorded/replayed per turn — raised from 8: a late call may be the one
    # that mattered.
    tool_provenance_max_entries: int = field(default_factory=lambda: int(os.getenv("TOOL_PROVENANCE_MAX_ENTRIES", "20")))
    # Per-call arg-gist char cap.
    tool_provenance_gist_chars: int = field(default_factory=lambda: int(os.getenv("TOOL_PROVENANCE_GIST_CHARS", "80")))
    # "[used tools: …]" annotation line budget; over it the line degrades to names only.
    tool_provenance_line_budget: int = field(default_factory=lambda: int(os.getenv("TOOL_PROVENANCE_LINE_BUDGET", "300")))
    # F7 on a bot-posted IMAGE: files_upload_v2 returns no share ts, so the image message's ts
    # has to be polled out of files.info, which Slack populates asynchronously. Wall-clock
    # bound on that poll. Measured live 2026-07-16: the share appeared ~1.8s after upload in a
    # private channel and ~3.8s in a DM (the slow case), so 15s is ~4x headroom over the worst
    # observation. Provenance is invisible, so it can afford to keep polling: the image is
    # already posted, and on timeout it simply keeps today's (missing) provenance. Only read
    # when enable_tool_provenance is on.
    image_share_ts_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("IMAGE_SHARE_TS_TIMEOUT_SECONDS", "15.0")))
    # How long the "Uploading…" indicator may keep waiting for that SAME share record before
    # giving up and completing anyway. The share is what makes the image actually VISIBLE, so
    # this decides whether the user watches a spinner or an empty gap. It is deliberately NOT
    # image_share_ts_timeout_seconds: this bound is visible and that one guards an invisible
    # DB row, so they fail in different directions and want different numbers.
    # Measured live 2026-07-17, upload-return → visible: 2.9-3.4s for a 3MB image, 4.1-4.7s
    # for 7MB, 5.4s for a real generation. It tracks image SIZE, so it will drift with whatever
    # the image model emits — hence the per-image log in image_delivery rather than trust in
    # this constant. 12s is ~2.2x the worst of those.
    # Erring long is the cheaper mistake: the wait ends when the share lands, so this only
    # bites when Slack is genuinely misbehaving, whereas erring short quietly re-opens the very
    # gap it exists to close. Expiring is not a failure — it just restores the old behavior.
    image_indicator_hold_seconds: float = field(default_factory=lambda: float(os.getenv("IMAGE_INDICATOR_HOLD_SECONDS", "12.0")))
    # Age (days) after which tool-use provenance rows are swept by the cleanup worker (F7/F14).
    tool_usage_retention_days: int = field(default_factory=lambda: int(os.getenv("TOOL_USAGE_RETENTION_DAYS", "90")))
    # Age (days) after which stored document-extraction rows (metadata + summary) are swept.
    document_retention_days: int = field(default_factory=lambda: int(os.getenv("DOCUMENT_RETENTION_DAYS", "90")))

    # --- Per-message timestamps (F10) ---
    # Prefix every message in model-visible thread context with a deterministic local
    # timestamp ("[Fri 2026-07-10 9:17 PM EDT]") rendered from the message's Slack ts in
    # the sender's profile timezone (UTC fallback), so the model can reason about time
    # gaps between messages. Applied on warm append + rebuild (self turns on rebuild only)
    # and to the participation classifier's thread-tail / channel-activity lines. Off →
    # content is byte-identical to pre-F10 (helper returns "" at every guarded call site).
    enable_message_timestamps: bool = field(default_factory=lambda: os.getenv("ENABLE_MESSAGE_TIMESTAMPS", "true").lower() == "true")

    # --- Slack search tool (redesign Phase B) — assistant.search.context ---
    # Requires an action_token from the triggering event, so the bot can only search in
    # response to a user interaction. See slack_client/search_tool.py for the privacy model.
    enable_search_tool: bool = field(default_factory=lambda: os.getenv("ENABLE_SEARCH_TOOL", "true").lower() == "true")
    # Code-level bound on what the executor will request, regardless of manifest scopes.
    # Add "im"/"mpim" only if you want the bot searching DMs (needs search:read.im/mpim scopes).
    search_channel_types: list = field(default_factory=lambda: _env_list(
        "SEARCH_CHANNEL_TYPES", ["public_channel", "private_channel"]))

    # --- Document architecture (Phase D2) ---
    # Native file input: PDFs within the API limits ride the attach turn as an input_file
    # content part (base64 per-request — never the OpenAI Files API), so the model sees
    # text + rendered pages (tables/charts/scans readable). Oversized PDFs and all other
    # types use local extraction. Later turns always use summary + read_document.
    enable_native_file_input: bool = field(default_factory=lambda: os.getenv("ENABLE_NATIVE_FILE_INPUT", "true").lower() == "true")
    native_file_max_pages: int = field(default_factory=lambda: int(os.getenv("NATIVE_FILE_MAX_PAGES", "100")))
    native_file_max_mb: int = field(default_factory=lambda: int(os.getenv("NATIVE_FILE_MAX_MB", "32")))
    # read_document tool: on-demand document access (download from Slack CDN -> extract in
    # memory -> return the requested slice). The DB holds summary + metadata + ref only.
    enable_read_document_tool: bool = field(default_factory=lambda: os.getenv("ENABLE_READ_DOCUMENT_TOOL", "true").lower() == "true")
    # Per-tool timeout for read_document, overriding the generic tool_call_timeout (20s): it may
    # download + render + OCR a scanned PDF, whose worst case (ocr_max_pages x ~3-5 s/page at 300
    # DPI, plus poppler render) far exceeds 20s. Default 120s covers a 20-page scan comfortably.
    read_document_timeout: float = field(default_factory=lambda: float(os.getenv("READ_DOCUMENT_TIMEOUT", "120.0")))
    # Process-lifetime LRU of extracted document text (never persisted) so iterating on one
    # document doesn't re-download/re-extract per question.
    doc_extraction_cache_size: int = field(default_factory=lambda: int(os.getenv("DOC_EXTRACTION_CACHE_SIZE", "20")))
    # F29: people-awareness tools — lookup_user (profile by id/@name/display name) and
    # list_channel_members (current-channel roster). Workspace-visible profile data only.
    enable_people_tools: bool = field(default_factory=lambda: os.getenv("ENABLE_PEOPLE_TOOLS", "true").lower() == "true")
    # OCR text from image-only/scanned PDFs on later turns (read_document) and on the big-file
    # local path. Requires the tesseract-ocr + poppler-utils system packages; if absent the
    # handler logs a warning and falls back to the honest "scanned, not extractable" note.
    enable_pdf_ocr: bool = field(default_factory=lambda: os.getenv("ENABLE_PDF_OCR", "true").lower() == "true")
    # Max pages OCR'd per document. tesseract at 300 DPI runs ~1-3 s/page, so 20 pages bounds
    # worst-case OCR near the tool-call timeout; beyond this a loud truncation note is prepended.
    ocr_max_pages: int = field(default_factory=lambda: int(os.getenv("OCR_MAX_PAGES", "20")))
    # Render DPI for OCR. tesseract accuracy needs ~300 DPI — 150 was live-verified too low for
    # small text (it is fine for the vision page-image path, which uses its own lower DPI).
    ocr_dpi: int = field(default_factory=lambda: int(os.getenv("OCR_DPI", "300")))
    # Concurrent document extractions (thread pool size). Extraction work is subprocess/CPU
    # (pdfplumber, poppler render, tesseract) off the event loop; a worst-case 20-page OCR can
    # hold one worker for ~1-2 min, and queue wait counts against READ_DOCUMENT_TIMEOUT, so
    # size this above the number of scans you expect to land at once. Floor of 1 enforced.
    doc_extraction_workers: int = field(default_factory=lambda: max(1, int(os.getenv("DOC_EXTRACTION_WORKERS", "5"))))

    # --- F30: background deep-research jobs ---
    # A local tool (start_deep_research) that detaches a genuine multi-source research
    # question into a background job: the model acks in one line, the thread lock releases,
    # and a sourced findings report lands in the same thread minutes later (mirrors the
    # background image-gen pattern). Off → the tool is not registered.
    enable_deep_research: bool = field(default_factory=lambda: os.getenv("ENABLE_DEEP_RESEARCH", "true").lower() == "true")
    # The detached job runs at its own (higher) reasoning effort — a real report is worth the
    # spend. Routed through clamp_effort against the thread's model.
    deep_research_reasoning_effort: str = field(default_factory=lambda: os.getenv("DEEP_RESEARCH_REASONING_EFFORT", "high"))
    deep_research_verbosity: str = field(default_factory=lambda: os.getenv("DEEP_RESEARCH_VERBOSITY", "medium"))
    # Hard wall-clock bound on one research job (it makes one non-streaming Responses call with
    # web_search + MCP). On timeout the job posts an honest failure note — never silent.
    deep_research_timeout: float = field(default_factory=lambda: float(os.getenv("DEEP_RESEARCH_TIMEOUT", "600")))
    # Per-thread cap on concurrent research jobs (friendly structured rejection at the cap, which
    # the model relays). Deliberately per-thread, no global cap — mirrors image gen's choice.
    deep_research_max_per_thread: int = field(default_factory=lambda: max(1, int(os.getenv("DEEP_RESEARCH_MAX_PER_THREAD", "2"))))
    # The job's tool-loop budget for PRODUCTIVE work. It is a runaway guard, not a ration: the
    # thing that actually bounds a detached job's cost is DEEP_RESEARCH_TIMEOUT (wall clock).
    # F37: card bookkeeping (update_todos) is exempt — it is passed as a `free_tool`, so a
    # chatty todo list cannot eat the calls the build phase needs for mount_file /
    # create_image_asset. Research spends almost nothing here (web_search is server-side and
    # costs no round); the budget is for local tools. On cap the loop forces a final answer
    # (tool_choice="none"), never an error.
    deep_research_max_tool_rounds: int = field(default_factory=lambda: max(1, int(os.getenv("DEEP_RESEARCH_MAX_TOOL_ROUNDS", "10"))))
    # F35: the BUILD phase — a second loop that runs only when the job declared `deliverables`,
    # with a code sandbox + image/mount tools, to turn the findings into an actual file.
    # It needs a bigger round budget than the research phase: mount, write code, read the
    # traceback, fix, re-run, verify. Running out of rounds mid-build is the difference between
    # a deck and an apology.
    deep_research_build_timeout: float = field(default_factory=lambda: float(os.getenv("DEEP_RESEARCH_BUILD_TIMEOUT", "600")))
    deep_research_max_build_rounds: int = field(default_factory=lambda: max(1, int(os.getenv("DEEP_RESEARCH_MAX_BUILD_ROUNDS", "16"))))
    # Label the findings post with a chat.postMessage username override ("<bot> [research: …]").
    # Needs the chat:write.customize scope, which the app may not have — on the first failure the
    # process falls back to plain posts for the rest of its life. Never breaks delivery.
    enable_research_label: bool = field(default_factory=lambda: os.getenv("ENABLE_RESEARCH_LABEL", "true").lower() == "true")

    # --- F32: code interpreter + artifacts ---
    # OpenAI's server-side code_interpreter tool: the model writes and runs Python in an
    # OpenAI-hosted container. Two things this buys us, both verified live 2026-07-11:
    #   1. REAL computation over attached data. Files already riding the turn as `input_file`
    #      parts auto-materialize in the container's /mnt/data — no Files API objects are
    #      created, so nothing of the user's data persists on OpenAI's side (same ephemeral
    #      request-body boundary as the native-PDF path). A 5000-row CSV summed exactly.
    #   2. ARTIFACTS. Files the code writes are read back off the container LISTING; we download
    #      the bytes and upload them to Slack. The container is the scratch space, so the
    #      no-disk rule (CLAUDE.md pitfall 6a) holds with ZERO new local dependencies.
    # The sandbox has no network egress (pip install and exfiltration both impossible).
    enable_code_interpreter: bool = field(default_factory=lambda: os.getenv("ENABLE_CODE_INTERPRETER", "true").lower() == "true")
    # F34: image generation/editing as TOOLS the model calls in context (default ON). When OFF,
    # the legacy pre-flight intent classifier + vision/new_image/edit routing in base.py runs
    # instead (the escape hatch, not the intended path). See Docs/TOOL_SUBSYSTEMS.md.
    enable_image_tools: bool = field(default_factory=lambda: os.getenv("ENABLE_IMAGE_TOOLS", "true").lower() == "true")
    # Containers are THREAD-SCOPED and persisted: one container per channel/thread, its id kept
    # in `thread_containers` and reused across turns, so the model's working state survives the
    # turn boundary ("clean that up" -> "now chart it" lands in the same /mnt/data).
    #
    # HARD API CEILING (probed live 2026-07-12): expires_after.minutes must be <= 20 — asking for
    # 60 returns HTTP 400 "integer above maximum value". So a container CANNOT be held longer
    # than 20 minutes of idle, and "persistent" means warm-within-an-active-conversation. A
    # revived thread always gets a fresh, empty container; that is the API's rule, not a choice.
    code_interpreter_container_ttl_minutes: int = field(
        default_factory=lambda: min(20, max(1, int(os.getenv("CODE_INTERPRETER_CONTAINER_TTL_MINUTES", "20")))))
    # Only reuse a stored container if we used it this recently. Held under the TTL so we don't
    # hand the API an id that idle-expired in the gap. Liveness is still CONFIRMED with a
    # retrieve() before reuse — the DB records when *we* last used it, which is not the same as
    # when the container was last active (an API call that failed never touched it).
    code_interpreter_container_reuse_minutes: int = field(
        default_factory=lambda: max(1, int(os.getenv("CODE_INTERPRETER_CONTAINER_REUSE_MINUTES", "15"))))
    # --- F32: outbound artifacts ---
    # Max artifacts published per turn. The model can write many intermediate files; only the
    # ones it cites get published, and this bounds a runaway loop from flooding the thread.
    artifact_max_files: int = field(default_factory=lambda: max(1, int(os.getenv("ARTIFACT_MAX_FILES", "4"))))
    # Per-file size ceiling for an outbound artifact. Slack's own limit is far higher; this is
    # our guard against uploading something absurd. Oversized artifacts are dropped with a note.
    artifact_max_mb: int = field(default_factory=lambda: max(1, int(os.getenv("ARTIFACT_MAX_MB", "25"))))
    # Outbound allowlist by extension. Deliberately excludes executables and macro-enabled Office
    # formats (.xlsm/.docm) — the bot must not hand anyone active content. `zip` IS allowed: a
    # background build can declare an "archive" deliverable, and the size caps above bound it.
    # HTML/SVG are excluded by default too: Slack won't render them inline anyway, and they can
    # carry script. Set ARTIFACT_ALLOWED_EXTENSIONS to override (comma-separated, no dots).
    artifact_allowed_extensions: List[str] = field(default_factory=lambda: [
        e.strip().lower().lstrip(".")
        for e in os.getenv(
            "ARTIFACT_ALLOWED_EXTENSIONS",
            "png,jpg,jpeg,gif,webp,pdf,csv,tsv,json,txt,md,xlsx,docx,pptx,zip"
        ).split(",") if e.strip()
    ])
    # Whole-phase bound on downloading + uploading a turn's artifacts. The answer is already
    # posted by then, but the turn is still held open, so a wedged upload would stall the next
    # message in the thread.
    artifact_publish_timeout: float = field(default_factory=lambda: float(os.getenv("ARTIFACT_PUBLISH_TIMEOUT", "120")))
    # Status-line emoji while the sandbox is running code.
    code_interpreter_emoji: str = field(default_factory=lambda: os.getenv("CODE_INTERPRETER_EMOJI", "📊"))

    # Link previews (unfurl cards) on the bot's posted messages. Default OFF (user
    # preference 2026-07-11, matching Claude Tag): links stay inline, and Slack's
    # unfurler no longer stamps "(edited)" on link-bearing posts. Applies to the
    # send_message path (normal replies, split chunks, research findings).
    enable_link_previews: bool = field(default_factory=lambda: os.getenv("ENABLE_LINK_PREVIEWS", "false").lower() == "true")

    # --- On-demand Slack history-fetch tools (Phase 8) ---
    # Read-only + privacy-scoped (public or bot-member channels only), so default ON. Wired to
    # the model through the local function-call loop (ENABLE_TOOL_LOOP).
    enable_history_tools: bool = field(default_factory=lambda: os.getenv("ENABLE_HISTORY_TOOLS", "true").lower() == "true")
    # Hard cap on messages returned by a single history-fetch tool call (the model's `limit` is
    # clamped to this regardless of what it asks for).
    history_tool_max_messages: int = field(default_factory=lambda: int(os.getenv("HISTORY_TOOL_MAX_MESSAGES", "50")))

    # --- Per-channel memory (Phase 9) ---
    # Read = inject the channel's durable facts into the system prompt on each response.
    # Write = model-invoked remember/update/forget tools during the response (Phase C).
    # Off → no injection, no tools, no extraction (unchanged behavior).
    enable_channel_memory: bool = field(default_factory=lambda: os.getenv("ENABLE_CHANNEL_MEMORY", "true").lower() == "true")
    # Hard cap on channel-scope memory rows per channel — keeps the injected block small and
    # bounded; at the cap the remember tool refuses and points at the oldest rows instead.
    memory_max_rows: int = field(default_factory=lambda: int(os.getenv("MEMORY_MAX_ROWS", "25")))
    # Legacy post-response extraction pass (pre-Phase-C write path). Kept one release as a
    # fallback in case tool-driven memory writes under-perform; costs one utility-model call
    # per exchange when on. Default OFF now that the model writes memory via tools.
    enable_memory_extraction_fallback: bool = field(default_factory=lambda: os.getenv("ENABLE_MEMORY_EXTRACTION_FALLBACK", "false").lower() == "true")

    # Long-context billing threshold (verified 2026-07-09): on 5.6-family and 5.5,
    # prompts with >272K input tokens bill at 2x input / 1.5x output for the request.
    LONG_CONTEXT_BILLING_THRESHOLD: int = 272_000

    def is_long_context(self, tokens: int) -> bool:
        """True when an input of `tokens` crosses OpenAI's long-context billing tier
        (>272K input → 2x input / 1.5x output on 5.5 and the 5.6 family)."""
        return tokens > self.LONG_CONTEXT_BILLING_THRESHOLD

    def get_model_token_limit(self, model: str) -> int:
        """Get the effective input token limit for a specific model

        This returns the maximum number of input tokens we should send.
        GPT-5.6 family (sol/terra/luna) and GPT-5.5 all share the verified
        1.05M window: 1.05M total - ~130k reserved = ~920k usable.
        Anything else (unknown/legacy, e.g. gpt-5-mini): the conservative
        400k window with a 50k reserve.

        Args:
            model: Model name (e.g., 'gpt-5.6-sol', 'gpt-5.5')

        Returns:
            Buffered token limit for safe operation
        """
        if model.startswith('gpt-5.6') or model.startswith('gpt-5.5'):
            return int(self.gpt54_max_tokens * self.gpt54_token_buffer_percentage)
        # Unknown/legacy models: use the conservative 400k window
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

    # Channel-shared override keys: the only channel_settings columns that join the
    # generation-config hierarchy (level 2.5). Everything else in channel_settings is
    # participation/UX config consumed elsewhere.
    _CHANNEL_OVERRIDE_KEYS = ("model", "reasoning_effort", "verbosity")

    def _map_channel_settings(self, channel_settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract the shared generation overrides from a channel_settings row
        (NULL columns = inherit → omitted)."""
        if not channel_settings:
            return {}
        return {k: channel_settings[k] for k in self._CHANNEL_OVERRIDE_KEYS
                if channel_settings.get(k)}

    def _compose_thread_config(self, user_prefs: Optional[Dict[str, Any]],
                               overrides: Optional[Dict[str, Any]],
                               channel_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Compose the config hierarchy:
        defaults <- user prefs <- channel shared settings <- thread overrides.

        Channel settings beat personal prefs (a channel behaves the same for
        everyone in it); an explicit thread override still wins. The composed
        reasoning_effort is clamped against the composed model so a cross-layer
        mix (e.g. channel model gpt-5.5 + user effort max) can never 400."""
        config = self._default_thread_config()
        if user_prefs:
            config.update(self._map_user_prefs(user_prefs))
        config.update(self._map_channel_settings(channel_settings))
        if overrides:
            config.update(overrides)
        config["reasoning_effort"] = clamp_effort(config.get("model", self.gpt_model),
                                                  config.get("reasoning_effort"))
        return config

    def get_thread_config(self, overrides: Optional[Dict[str, Any]] = None, user_id: Optional[str] = None,
                          db = None, channel_id: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration for a specific thread with settings hierarchy:
        1. System defaults (from .env)
        2. User preferences (from database)
        3. Channel shared settings (model/effort/verbosity — anyone in the channel can set them)
        4. Thread overrides (passed as parameter)
        """
        user_prefs = None
        channel_settings = None
        if user_id and db:
            try:
                user_prefs = db.get_user_preferences(user_id)
            except Exception as e:
                logging.getLogger("bot.config").warning(f"Error fetching user preferences: {e}")
        if channel_id and db:
            try:
                channel_settings = db.get_channel_settings(channel_id)
            except Exception as e:
                logging.getLogger("bot.config").warning(f"Error fetching channel settings: {e}")
        return self._compose_thread_config(user_prefs, overrides, channel_settings)

    async def get_thread_config_async(self, overrides: Optional[Dict[str, Any]] = None,
                                      user_id: Optional[str] = None, db = None,
                                      channel_id: Optional[str] = None) -> Dict[str, Any]:
        """Async get_thread_config — awaits the aiosqlite reads instead of
        blocking the event loop with sync sqlite on every message."""
        user_prefs = None
        channel_settings = None
        if user_id and db:
            try:
                user_prefs = await db.get_user_preferences_async(user_id)
            except Exception as e:
                logging.getLogger("bot.config").warning(f"Error fetching user preferences: {e}")
        if channel_id and db:
            try:
                channel_settings = await db.get_channel_settings_async(channel_id)
            except Exception as e:
                logging.getLogger("bot.config").warning(f"Error fetching channel settings: {e}")
        return self._compose_thread_config(user_prefs, overrides, channel_settings)


# Global config instance
config = BotConfig()