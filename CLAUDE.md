# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python-based chatbot system supporting Slack and Discord platforms using OpenAI's Responses API (not Chat Completions). The architecture is stateless, with platforms as the source of truth, rebuilding context from platform history on demand.

## Key Commands

### Development Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -U -r requirements.txt

# Setup SQLite database (creates data/ and data/backups/ directories)
mkdir -p data data/backups
```

### Running the Bot
```bash
python3 slackbot.py          # Slack bot
python3 discordbot.py         # Discord bot (V2 under development)
python3 main.py --platform slack  # Alternative method
```

### Testing Commands
```bash
make test               # Run unit tests with coverage (default)
make test-all          # Run ALL tests (unit + integration with real API)
make test-unit         # Run only unit tests
make test-integration  # Run only integration tests
make test-fast         # Run tests without coverage (faster)
make test-verbose      # Verbose test output
make test-coverage     # Generate HTML coverage report
make test-report       # Open coverage report in browser

# Run specific test file
python3 -m pytest tests/unit/test_config.py -v

# Run specific test
python3 -m pytest tests/unit/test_config.py::TestBotConfig::test_default_initialization -v

# Run tests with specific marker
python3 -m pytest -m critical  # Run critical tests only
python3 -m pytest -m smoke     # Run smoke tests only
```

### Code Quality
```bash
make lint    # Run linting checks (requires ruff, mypy)
make format  # Auto-format code (requires black, isort)
make check   # Run all checks (lint + test)
make clean   # Remove test artifacts and cache
```

## Architecture & Key Design Decisions

### Responses API Implementation
- Uses OpenAI's Responses API with `store=False` for stateless operation
- Full message history passed in `input` parameter, not using `previous_response_id` chaining
- System prompts included in messages as "developer" role
- Model-specific parameter handling:
  - GPT-5 reasoning models (`gpt-5-mini`, `gpt-5-nano`): Fixed `temperature=1.0`, no `top_p`, uses `reasoning_effort` and `verbosity`
  - GPT-5 chat models (`gpt-5-chat-latest`): Standard temperature/top_p support
- See `Docs/RESPONSES_API_IMPLEMENTATION_DETAILS.md` for migration details if switching to `previous_response_id` chaining

### Threading & State Management
- `ThreadStateManager` maintains conversation state per thread in memory
- Thread key format: `channel_id:thread_ts` (critical: watch for colon delimiter issues in DB)
- Thread locks prevent concurrent processing of same thread
- State is rebuilt from Slack/Discord history after restarts
- `AssetLedger` tracks generated images per thread (base64 data, prompts, URLs)
- All state lost on restart - must rebuild from platform APIs

### SQLite Persistence Layer
- Separate databases: `data/slack.db`, `data/discord.db`
- WAL mode enabled for concurrency
- Schema includes: threads, messages, images, users tables
- Automatic backups to `data/backups/` with 7-day retention
- Image metadata stored in DB, NOT base64 data (memory optimization)
- Full implementation plan in `Docs/sqlite-integration-plan.md`

### Message Processing Pipeline
1. `BaseClient.handle_event()` receives platform event
2. `MessageProcessor.process_message()` handles core logic
3. Intent classification determines response type (chat/image/vision/edit)
4. Response sent back through platform client
5. Thread state updated with conversation

### Configuration Hierarchy
1. `.env` file defaults (BotConfig)
2. Thread-specific overrides (in memory/DB)
3. Utility functions use: `UTILITY_REASONING_EFFORT`, `UTILITY_VERBOSITY`
4. Analysis functions use: `ANALYSIS_REASONING_EFFORT`, `ANALYSIS_VERBOSITY`

## Critical Implementation Notes

### Model-Specific Parameters
**GPT-5 Reasoning Models** (gpt-5-mini, gpt-5-nano):
- Temperature MUST be 1.0
- No top_p support
- Uses reasoning_effort and verbosity parameters

**GPT-5 Chat Models** (gpt-5-chat-*):
- Standard temperature/top_p support
- No reasoning_effort/verbosity parameters

### Image Processing Flow
1. Intent classification determines image vs text response
2. Image generation uses `gpt-image-1` model
3. Vision analysis stores full context (not shown to user)
4. Edit operations use previous analysis for context

### Error Handling Patterns
- All errors formatted with emoji indicators
- Code blocks for technical errors
- Circuit breaker pattern for rate limiting
- Thread locks prevent race conditions

## Testing Strategy

### Test Categories
- **@pytest.mark.critical** - Core functionality that must work
- **@pytest.mark.smoke** - Basic operations verification
- **@pytest.mark.integration** - Tests requiring external resources
- **@pytest.mark.unit** - Fast, isolated unit tests

### Test File Organization
```
tests/
├── conftest.py           # Shared fixtures and configuration
├── unit/                 # Unit tests (run by default)
│   ├── test_config.py
│   ├── test_database.py
│   ├── test_message_processor.py
│   ├── test_thread_manager.py
│   └── ...
└── integration/          # Integration tests (require real APIs)
    └── test_message_flow.py
```

### Test Environment
- Uses mock environment variables from `conftest.py`
- Integration tests use real API keys from `.env` when `make test-all` is run
- SQLite tests use temporary databases in memory

## Common Pitfalls to Avoid

1. **Never use Chat Completions API** - This codebase uses Responses API exclusively
2. **Don't modify temperature for GPT-5 reasoning models** - Must be 1.0
3. **Thread IDs contain colons** - Format is `channel_id:thread_ts` - consider using "|" delimiter if implementing DB changes
4. **Message history is rebuilt from Slack** - Don't rely on in-memory state persisting
5. **Utility functions should use utility env vars** - Not default reasoning/verbosity
6. **Image data storage** - Never store base64 image data in DB, only URLs and metadata
7. **SQLite concurrency** - Use WAL mode and be careful with `check_same_thread=False`
8. **Never start the bot during testing** - User manages running state
9. **ALWAYS include full context** in API calls - never limit to "x" previous messages

## File Structure & Responsibilities

Core modules:
- `main.py` - Entry point with platform selection
- `base_client.py` - Abstract base for platform clients
- `slack_client.py` - Slack-specific implementation
- `message_processor.py` - Core message handling logic
- `openai_client.py` - OpenAI API wrapper for Responses API
- `thread_manager.py` - Thread state and locking
- `database.py` - SQLite persistence layer
- `config.py` - Environment variable management
- `prompts.py` - System prompts and intent classification
- `markdown_converter.py` - Platform-specific markdown conversion
- `image_url_handler.py` - Image URL processing and validation

Directories:
- `streaming/` - Experimental streaming support (not active)
- `legacy/` - Previous bot version (reference only)
- `Docs/` - Technical documentation and plans
- `tests/` - Test suite with unit and integration tests
- `data/` - SQLite databases (created at runtime)
- `logs/` - Application logs (created at runtime)

## Development Workflow

1. **Before making changes**: Run `make test` to ensure tests pass
2. **After making changes**: 
   - Run `make test` for unit tests
   - Run `make test-all` if changes affect API interactions
   - Run `make lint` to check code quality
3. **When debugging tests**: Use `-xvs` flags for detailed output
4. **For test coverage**: Check `htmlcov/index.html` after running tests

## Important Instructions

- Always prefer editing existing files over creating new ones
- Never create documentation files unless explicitly requested
- Don't break existing bot code - if fixes are needed, consult user first
- Always use absolute paths for file operations
- Full context must be included in all API calls - no message limits