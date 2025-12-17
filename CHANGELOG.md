# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [2.3.4] - 2025-12-16

### 🔧 Improvement - Image Quality Auto Option

#### Changed
- **Auto Quality Default**: Added 'auto' option for image quality and set as new default
  - Lets the model decide quality level based on prompt complexity
  - Available options now: auto, low, medium, high

## [2.3.3] - 2025-12-16

### 🚀 Feature - Image Quality & Background Settings

#### Added
- **Image Quality Setting**: User-configurable quality for image generation
  - Options: Auto, Low (faster/cheaper), Medium (balanced), High (best quality)
  - Exposed in `/chatgpt-settings` modal under Image Generation
- **Image Background Setting**: User-configurable background type
  - Options: Auto, Transparent, Opaque
  - Exposed in `/chatgpt-settings` modal under Image Generation
- **Database Migrations**: Automatic schema updates for existing users
  - New columns added with smart defaults on bot startup
  - No manual intervention required

#### Changed
- **Default Image Model**: Updated to `gpt-image-1.5` in `.env.example`
- **Documentation**: Updated README with GPT-5.2 model references

#### Removed
- **Deprecated Settings**: Removed `image_style` parameter (was DALL-E 3 only)

## [2.3.2] - 2025-12-15

### 🐛 Bug Fix - Streaming Blank Message & Pagination Orphan

#### Fixed
- **Vision Streaming Blank Updates**: Fixed race condition causing messages to briefly go blank during streaming
  - Root cause: `progress_task.cancel()` only requests cancellation, takes effect at next await point
  - Without awaiting, progress_task could overwrite streamed content with stale text
  - Now properly awaits cancellation before proceeding with streaming updates
- **Vision Pagination Orphan**: Fixed "Continued in next message..." appearing without Part 2
  - Vision handler had no overflow/pagination logic
  - Added full overflow handling matching text.py pattern with intelligent split points
- **Async Callback Support for Vision**: Added async callback support to vision API
  - Vision streaming callbacks can now properly await async operations
  - Matches pattern already used in responses.py for text streaming

#### Changed
- **Safety Margin Increase**: Increased overflow detection margin from 330 to 600 chars
  - Ensures overflow triggers before messaging layer's backup truncation at 3700 chars
  - Prevents orphaned "continued" messages from backup truncation

## [2.3.1] - 2025-12-15

### 🔧 Improvements - MCP Citation Stripping & Tool Attribution

#### Changed
- **MCP Citation Stripping**: Moved citation stripping from streaming layer to Slack messaging layer
  - Single point of control for all message types (streaming, non-streaming, updates)
  - Enhanced regex patterns to catch additional MCP citation formats
  - Properly handles tool-generated citations (`read_documentation`, `get_library`, etc.)
  - Web search citations preserved as clickable links
- **MCP Tool Attribution**: "Used Tools" footer now shows specific MCP server names
  - Format changed from `Used Tools: mcp` to `Used Tools: MCP (aws_knowledge, context7)`
  - Groups multiple MCP servers under single "MCP" label
  - Extracts server_label from `response.output_item.done` events for accurate attribution

#### Fixed
- **Citation Display**: Fixed MCP citations rendering as emoji + backend strings in Slack messages
- **Tool Attribution Accuracy**: Now correctly identifies which MCP servers were invoked during a response

## [2.3.0] - 2025-01-15

### 🚀 Feature - GPT-5.1 Model Support & Performance Optimizations

#### Added
- **GPT-5.1 Model Support**: Added GPT-5.1 as a new model option with enhanced capabilities
  - New "None" reasoning_effort option with adaptive reasoning
  - Automatic reasoning depth adjustment based on query complexity
  - 24-hour prompt caching for GPT-5.1 across all API calls (chat, vision, intent classification)
  - Web search now works with all reasoning levels including "none"
  - Separate settings UI for GPT-5.1 with dedicated reasoning options
  - Future-proof support for gpt-5.1-mini (not yet released)
- **Migration Script**: Created `scripts/migrate_users_to_gpt51.py` for automated user migration from GPT-5 to GPT-5.1
- **Configuration Updates**:
  - Added `gpt-5.1` to MODEL_KNOWLEDGE_CUTOFFS
  - Updated model dropdown in settings modal to include GPT-5.1 as top option
  - Added `_add_gpt51_settings()` method with new reasoning options
  - Changed default UTILITY_MODEL from gpt-4.1-mini to gpt-5-mini in .env.example

#### Changed
- **Reasoning Options**:
  - GPT-5.1 uses "none/low/medium/high" (replaces "minimal" with "none")
  - GPT-5 retains "minimal/low/medium/high" (backward compatible)
  - GPT-5.1 removes web_search + minimal reasoning constraint
- **API Integration**:
  - Added prompt caching (`prompt_cache_retention="24h"`) for GPT-5.1 in:
    - Main chat responses (streaming and non-streaming)
    - Vision analysis (streaming and non-streaming)
    - Intent classification (for future gpt-5.1-mini support)
  - Enhanced model detection logic in responses.py
  - Added `reasoning_level_gpt51` action handler for Slack modal interactions
- **System Prompt Optimization**: Moved date/time context to end of system prompt to maximize prompt caching effectiveness (90% cost savings on cached tokens)

#### Fixed
- **MCP Settings Preservation**: Fixed bug where MCP settings were lost when switching between GPT-4 and GPT-5 models
  - Validation no longer forces `enable_mcp=False` for GPT-4 users
  - Preserves user's MCP preference when switching back to GPT-5
  - Database now retains MCP setting even when using non-GPT-5 models

#### Notes
- GPT-5 model remains unchanged for backward compatibility
- Users can explicitly opt into GPT-5.1 via settings modal
- Run migration script manually to update existing GPT-5 users to GPT-5.1
- Reasoning effort preferences are model-specific and may need adjustment when switching models

## [2.2.3] - 2025-11-10

### 🐛 Bug Fix - MCP Settings Persistence

#### Fixed
- **MCP Toggle Persistence**: Fixed bug where MCP toggle changes in settings modal were not persisting to the database
  - Added `enable_mcp` to boolean fields list in `update_user_preferences()` (sync/async)
  - Added boolean conversion in `get_user_preferences()` (sync/async)
  - Added to thread config propagation in `get_or_create_thread_async()`
- MCP settings now correctly save and load across sessions for both global and thread-specific configurations

## [2.2.2] - 2025-11-07

### 🐛 Bug Fix - MCP Tool Attribution

#### Fixed
- **MCP Tool Attribution Accuracy**: Fixed bug where bot reported all available MCP servers in "Used Tools" footer instead of only servers actually invoked
  - Non-streaming: Detects tools via response.output inspection
  - Streaming: Detects tools via search_counts tracking
  - Both modes now show "Used Tools: mcp" only when MCP was actually invoked

#### Changed
- Simplified MCP attribution to show generic "mcp" label instead of individual server names
- Added `return_metadata` parameter to response API for tool usage tracking

## [2.2.1] - 2025-11-07

### 📝 Configuration & Documentation

#### Added
- **MCP Environment Variables**: Added MCP configuration to `.env.example`
  - `MCP_ENABLED_DEFAULT`: Enable MCP by default for new users
  - `MCP_CONFIG_PATH`: Path to MCP server configuration file
- MCP architecture documentation

## [2.2.0] - 2025-11-07

### 🎉 Major Feature - Model Context Protocol (MCP) Integration

#### Added
- **MCP Support (Beta)**: Full Model Context Protocol integration for GPT-5 models
  - Server configuration management via `mcp_config.json`
  - Database schema for caching MCP tool definitions
  - MCPManager handles server validation and tool discovery
  - Settings UI toggle for enabling/disabling MCP (GPT-5 only)
  - Dynamic MCP server inclusion in tools array
- **Citation & Attribution System**:
  - Strip MCP citations while preserving web_search citations (clickable links)
  - Unified tools attribution at end of responses
  - Clean API messages by removing attribution before OpenAI submission
- **Error Handling & Retry Logic**:
  - Graceful MCP connection failure handling with retry logic
  - Exclude failed MCP servers from retry attempts
  - User-friendly error messages for connection issues
  - Show failed servers in tools attribution
- **UI & Status Updates**:
  - MCP status messages during tool discovery and execution
  - Track MCP call counts with generic "MCP call #N" messages
  - Settings modal integration for GPT-5 models
  - Beta feature notice in documentation

#### Changed
- Updated README with MCP configuration instructions and Slack scope requirements
- Enhanced MCP config example with comprehensive documentation
- Added MCP metrics gathering for monitoring

## [2.1.5] - 2025-09-30

### 🐛 Bug Fix - Message Pagination

#### Fixed
- **Overflow Message Display**: Fixed continuation messages not appearing in thread when response exceeded Slack's message length limit
  - Changed thread_id parameter from thinking_id (status message timestamp) to message.thread_id (actual thread timestamp)
  - Continuation messages now properly appear in correct thread and trigger pagination if still too long
  - Full message content was always correctly stored in database - this was purely a display bug affecting Slack message delivery

## [2.1.4] - 2025-09-24

### 🎯 Configuration, Session Management & Licensing Update

#### Added
- **MIT License**: Added open source MIT license to the project
- **Database Directory Configuration**: New `DATABASE_DIR` environment variable for customizable database location
- **Modal Session Database Storage**: Settings modal sessions now stored in database instead of Slack metadata
- **Modal Session Cleanup**: Automatic cleanup of orphaned settings modal sessions during daily maintenance

#### Fixed
- **Hardcoded Timeouts Removed**: All text operations now respect configured `API_TIMEOUT_STREAMING_CHUNK` value instead of hardcoded 150s
- **Dead Code Cleanup**: Removed unused `text_high_reasoning` operation type that was never utilized
- **Slack Metadata Size Limits**: Resolved issues with oversized private_metadata by moving session data to database

#### Changed
- **Settings Modal Architecture**: Migrated from storing full session state in Slack's private_metadata to database-backed sessions with UUID references
- **Timeout Configuration**: Text operations (intent classification, prompt enhancement, normal text, text with tools) now use environment-configured timeouts
- **Database Path Flexibility**: Database and backup directories now use configurable path from `DATABASE_DIR` setting

## [2.1.3] - 2025-09-18

### 🐛 Settings & Configuration Fixes

#### Fixed
- **Default Values Correction**: Fixed incorrect default values for `reasoning_effort` and `verbosity` in user preferences
- **Settings Modal Defaults**: Ensured proper default values are applied when creating new user preferences

## [2.1.2] - 2025-09-17

### 🔧 Logging & Thread Safety Improvements

#### Fixed
- **Logger Thread Safety**: Updated logger implementation for async/thread safety paradigms after refactor
- **Log Rotation Issues**: Fixed problems with log file rotation under concurrent access
- **Import Errors**: Fixed missing imports in refactored modules

#### Changed
- **Message Processor Restoration**: Reverted accidental restoration of monolithic message processor, re-applied modular version

## [2.1.1] - 2025-09-16

### 🚀 Enhanced Streaming Reliability & UX Improvements

#### Fixed
- **User Context**: Fixed user timezone/context not being injected after async refactor
- **Settings Modal**: Fixed reasoning level being lost on mobile when toggling web search
- **Streaming Reliability**: Fixed text truncation when Slack API updates fail (17/18 success case)
- **Message Overflow**: Fixed transitions with proper continuation handling
- **Part Labels**: Fixed "Part X" labels disappearing during streaming updates
- **Loading Indicators**: Fixed enhanced prompt loading indicators not being removed properly

#### Changed
- **Timeout Adjustments**: Increased all text operation timeouts to 2.5 minutes minimum
- **Progress Feedback**: Added humorous progress messages after 30s and 60s+ for long operations
- **Image Analysis**: Added progress monitoring to image analysis operations
- **Timeout Handling**: Improved to only warn (never fail) on chunk timeouts

## [2.1.0] - 2025-09-16

### 🎉 Major Async/Await Refactor & Critical Stability Fixes

#### Changed
- **Async/Await Migration**: Migrated critical components to async/await pattern to fix concurrency issues
- **Thread Management**: Added AsyncThreadStateManager and AsyncThreadLockManager for proper synchronization
- **Database Operations**: Implemented async database methods running in parallel with sync versions

#### Fixed
- **Database Commits**: Fixed missing commits in async methods (save_thread_config_async, cache_message_async, etc.)
- **Settings Modal**: Fixed not preserving pending messages for new user flow
- **Web Search Persistence**: Fixed checkbox not persisting after save
- **Boolean Conversions**: Fixed issues in async database methods
- **Thread Config**: Fixed retrieval issues under concurrent load
- **Race Conditions**: Eliminated crashes under concurrent load

#### Added
- **Comprehensive Testing**: Expanded test coverage for async operations
- **Load Testing**: Verified stability under production workloads

## [2.0.4] - 2025-09-16

### 🐛 Critical Bug Fix - Bot Hanging Resolution

#### Fixed
- **Removed problematic `timeout_wrapper` that was causing zombie threads and bot hanging**
  - The wrapper was creating daemon threads that continued running after timeouts
  - These threads held HTTP connections, eventually exhausting the connection pool
  - Bot would become unresponsive after multiple timeouts, requiring manual restart
- Now using OpenAI SDK's native timeout handling via httpx
- Bot no longer hangs after consecutive timeout errors

#### Changed
- Improved timeout error messages to clearly indicate OpenAI as the source
  - "OpenAI Timeout" instead of generic "Taking Too Long"
  - "OpenAI's API is not responding" with specific timeout duration
  - All user-facing messages now explicitly mention OpenAI service issues
- Updated tests to remove references to deleted `timeout_wrapper`

#### Added
- Integration tests for intent classification model comparison
- Better timeout tracking and logging for diagnostics

## [2.0.3] - 2024-12-15

### 🔧 Code Quality & Reliability Improvements

#### Changed
- Refactored codebase to improve maintainability and reliability
- Cleaned up unused imports across all modules
- Fixed unused variables (`channel`, `truncated`, `content_preview`, `removed_msg`, etc.)
- Replaced bare except clauses with specific `Exception` handling
- Cleaned up f-string placeholders without variables
- Improved custom instructions handling in main prompt

#### Added
- Comprehensive timeout error handling test suite (`test_timeout_error_handling.py`)
- 586 new test cases covering various error scenarios
- Better error context and recovery strategies

#### Fixed
- All linting issues identified by ruff and pyright diagnostics
- Improved exception propagation throughout the codebase

## [2.0.2] - 2024-12-14

### 🐛 Bug Fixes

#### Fixed
- Prevented infinite retry loop on OpenAI timeout errors
- Reduced duplicate logging in error scenarios
- Improved timeout handling with proper circuit breaker implementation

## [2.0.1] - 2024-12-13

### ✨ Features & Documentation

#### Added
- Context-aware vision enhancement for better screenshot handling
- Slack app manifest file for easy app configuration
- Slack app commands documentation in README

#### Changed
- Made vision prompt enhancement more intelligent based on image context
- Improved handling of screenshot analysis

#### Developer
- Added debugging capabilities for Slack shortcut handlers (later reverted)

## [2.0.0] - 2024-09-12

### 🎉 Major Release - Complete V2 Rewrite

This release represents a complete rewrite of the ChatGPT Bots project, focusing on production stability, user experience, and advanced AI capabilities.

### ✨ New Features

#### Core Architecture
- **Responses API Migration**: Migrated from OpenAI's Chat Completions API to the new Responses API for advanced tool calling. The Chat Completions API is now deprecated.
- **Stateless Design**: Platform (Slack) as source of truth with dynamic context rebuilding
- **Abstract Base Client**: Modular architecture supporting multiple platforms
- **SQLite Persistence**: User preferences, thread settings, and message caching
- **Thread Management**: Concurrent request handling with proper locking mechanisms

#### User Experience
- **Interactive Settings Modal**: Configure preferences via `/chatgpt-settings` command
- **Thread-Specific Settings**: Different configurations per conversation
- **Custom Instructions**: Personalized AI behavior per user
- **Multi-User Context**: Proper handling of shared conversations with username tracking
- **Welcome Flow**: First-time user onboarding with guided setup

#### AI Capabilities
- **Intelligent Intent Classification**: Automatic detection of image/text/vision/edit requests
- **Image Generation & Editing**: Natural language image creation and modification
- **Vision Analysis**: Process uploaded images with detailed descriptions
- **Document Processing**: Extract and analyze PDFs, Office files, code files
- **Web Search Integration**: Current information retrieval (GPT-5 models)
- **Streaming Responses**: Real-time message updates with circuit breaker protection

#### Models & Configuration
- **Multi-Model Support**: GPT-5, GPT-5 Mini, GPT-4.1, GPT-4o
- **Dynamic Parameters**: Model-specific settings (reasoning_effort, verbosity for GPT-5)
- **Token Management**: Smart trimming with configurable thresholds
- **Utility Models**: Separate models for different tasks (analysis, utilities)

### 🔧 Technical Improvements

#### Performance
- Thread-safe operations with comprehensive locking
- SQLite WAL mode for concurrent database access
- Automatic message trimming at 80% token capacity
- Circuit breaker pattern for streaming failures

#### Testing
- 100+ unit tests with 80%+ coverage
- Integration tests for OpenAI API interactions
- Load testing verified with production workloads
- Comprehensive test fixtures and mocks

#### Developer Experience
- Makefile for common operations
- Structured logging with rotation
- Environment-based configuration
- Comprehensive error handling
- Type hints throughout codebase

### 📝 Configuration Changes

#### New Environment Variables
- `SETTINGS_SLASH_COMMAND`: Customizable settings command
- `DEFAULT_REASONING_EFFORT`: GPT-5 reasoning depth
- `DEFAULT_VERBOSITY`: Response detail level
- `UTILITY_REASONING_EFFORT`: For quick operations
- `ANALYSIS_REASONING_EFFORT`: For complex tasks
- `TOKEN_BUFFER_PERCENTAGE`: Dynamic token limits
- `ENABLE_WEB_SEARCH`: Web search capability
- `ENABLE_STREAMING`: Real-time responses
- Multiple streaming configuration options

#### New Slack Scopes
- `groups:history`: Private channel access
- `users:read`: Workspace member information
- `users:read.email`: Email address access

### 🐛 Bug Fixes
- Fixed race conditions in concurrent message processing
- Resolved settings persistence issues under load
- Fixed scope selection logic for new vs existing users
- Addressed oversized Slack message handling
- Fixed thread context mixing in shared conversations

### 📚 Documentation
- Comprehensive README with setup instructions
- Detailed CLAUDE.md for AI assistant guidance
- SQLite integration plan
- User settings modal design document
- Responses API implementation details
- Test documentation and templates

### ⚠️ Breaking Changes
- Discord support temporarily removed (V2 rewrite in progress)
- Changed from Chat Completions to Responses API
- New database schema
- Updated environment variable structure
- Modified logging configuration

### 🔄 Migration Guide

1. **Database Migration**: No migration path from V1 - fresh install
2. **Environment Variables**: Update `.env` using `.env.example` as template
3. **Slack App**: Add new required scopes in Slack App settings
4. **Model Selection**: Choose appropriate GPT model and defaults in configuration
5. **Custom Instructions**: Users should configure via `/chatgpt-settings`


### 🙏 Acknowledgments
Special thanks to all testers who participated in load testing and helped identify edge cases.

---

## Previous Versions

For changes prior to v2.0.0, please refer to git history.