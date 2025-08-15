# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based chatbot system supporting Slack and Discord platforms using OpenAI's Responses API (not Chat Completions). The architecture is stateless, with platforms as the source of truth, rebuilding context from platform history on demand.

## Key Architecture Decisions

### Responses API Usage
- Uses OpenAI's Responses API with `store=False` for stateless operation
- Full message history passed in `input` parameter, not using `previous_response_id` chaining
- System prompts included in messages as "developer" role
- Model-specific parameter handling for GPT-5 reasoning vs chat models

### Threading & State Management (Current - In Memory)
- `ThreadStateManager` maintains conversation state per thread in memory
- Thread key format: `channel_id:thread_ts` (watch for colon delimiter issues)
- Messages limited to last 20 in memory for context management
- Thread locks prevent concurrent processing of same thread
- State is rebuilt from Slack/Discord history after restarts
- `AssetLedger` tracks generated images per thread (base64 data, prompts, URLs)
- All state lost on restart - must rebuild from platform APIs

### Configuration Hierarchy
Current:
1. `.env` file defaults (BotConfig)
2. Thread-specific overrides (in memory)
3. Utility functions use separate env vars: `UTILITY_REASONING_EFFORT`, `UTILITY_VERBOSITY`
4. Analysis functions use: `ANALYSIS_REASONING_EFFORT`, `ANALYSIS_VERBOSITY`

Future (with SQLite):
1. BotConfig (.env) → User config (DB) → Thread config (DB)

## Running the Application

```bash
# Setup virtual environment
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -U -r requirements.txt

# Run bots
python3 slackbot.py          # Slack bot
python3 discordbot.py         # Discord bot (V2 under development)
python3 main.py --platform slack  # Alternative method
```

## Development Commands

```bash
# Install dependencies
python3 -m pip install -U -r requirements.txt

# Check logs
tail -f logs/app.log
tail -f logs/error.log

# Environment setup
# Edit .env with required tokens
```

## Critical Implementation Notes

### Model-Specific Parameters
- **GPT-5 Reasoning Models** (gpt-5-mini, gpt-5-nano):
  - Temperature MUST be 1.0
  - No top_p support
  - Uses reasoning_effort and verbosity parameters
  
- **GPT-5 Chat Models** (gpt-5-chat-*):
  - Standard temperature/top_p support
  - No reasoning_effort/verbosity parameters

### Image Processing Flow
1. Intent classification determines image vs text response
2. Image generation uses `gpt-image-1` model
3. Vision analysis stores full context (not shown to user)
4. Edit operations use previous analysis for context

### Message Processing Pipeline
1. `BaseClient.handle_event()` receives platform event
2. `MessageProcessor.process_message()` handles core logic
3. Intent classification determines response type
4. Response sent back through platform client
5. Thread state updated with conversation

### Error Handling
- All errors formatted with emoji indicators
- Code blocks for technical errors
- Circuit breaker pattern for rate limiting (in progress)
- Thread locks prevent race conditions

## File Structure & Responsibilities

- `main.py` - Entry point, platform selection
- `base_client.py` - Abstract base for platform clients
- `slack_client.py` - Slack-specific implementation
- `message_processor.py` - Core message handling logic
- `openai_client.py` - OpenAI API wrapper for Responses API
- `thread_manager.py` - Thread state and locking
- `config.py` - Environment variable management
- `prompts.py` - System prompts and intent classification
- `streaming/` - Experimental streaming support (not active)
- `legacy/` - Previous bot version (reference only)

## Common Pitfalls to Avoid

1. **Never use Chat Completions API** - This codebase uses Responses API exclusively
2. **Don't modify temperature for GPT-5 reasoning models** - Must be 1.0
3. **Thread IDs contain colons** - Format is `channel_id:thread_ts` - consider using "|" delimiter if implementing DB
4. **Message history is rebuilt from Slack** - Don't rely on in-memory state persisting
5. **Utility functions should use utility env vars** - Not default reasoning/verbosity
6. **Image data storage** - Never store base64 image data in DB, only URLs and metadata
7. **SQLite concurrency** - Use WAL mode and be careful with `check_same_thread=False`

## Future Implementation Plans

### SQLite Integration (Planned - Phase 1)
**Implementation Priority - Core functionality only, no new user commands**

Database Structure:
- Separate databases: `data/slack.db`, `data/discord.db`
- Backup directory: `data/backups/` with 7-day retention
- WAL mode for concurrency (creates `.db-wal` and `.db-shm` files)

Key Benefits When Implemented:
- **Unlimited message history** (currently limited to 20)
- **Full image analysis storage** (currently truncated to 100 chars)
- **Config persistence** across restarts
- **Instant thread recovery** without Slack API calls
- **3-month data retention** matching Slack workspace retention

Implementation Notes:
- `database.py` will handle all SQLite operations
- ThreadStateManager will integrate with DB for persistence
- Message cache remains optional (Slack/Discord still source of truth)
- Phase 2 (user commands like /config) requires explicit approval
- Full plan in `Docs/sqlite-integration-plan.md`

### Responses API with previous_response_id (Documented)
- Full implementation details in `Docs/RESPONSES_API_IMPLEMENTATION_DETAILS.md`
- Would enable server-side conversation management
- Currently not implemented, using stateless approach with full message history

## Testing

Currently no automated tests. Manual testing approach:
1. Test image generation with prompts like "draw a cat"
2. Test vision analysis by uploading images
3. Test image editing with "edit the last image"
4. Verify thread state recovery after restart
5. Check config overrides are applied correctly
- Never start the bot on your own. The user will manage running state. If you need the bot restarted, request the user to do so.
- ALWAYS include full context in any API calls. NEVER restrict or limit context/previous responses in history to "x" previous messages. Every request should be full, unlimited context.