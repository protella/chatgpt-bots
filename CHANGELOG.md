# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### 🔌 MCP Slash Command & Tool Selection Improvements

#### Added
- **MCP Slash Command**: Added `/reportpro-dev` slash command for direct MCP tool invocation
  - Bypasses intent classification and directly queries MCP tools
  - Creates thread with settings button for user configuration
  - Implements forced intent pattern matching ReportPro implementation
- **Force Invoke Pattern**: Added `force_invoke` parameter to MCP handler to skip LLM tool selection
- **Comprehensive Tool Selection Prompt**: Created detailed, generic prompt for MCP tool selection in `prompts.py`
  - Emphasizes using specialized tools over LLM training data
  - Generic design works with any MCP tool type
  - Provides clear guidance for ambiguous cases

#### Changed
- **MCP Tool Selection**: Updated tool selection to use proper utility model configuration
  - Now uses `config.utility_model`, `config.utility_max_tokens`, `config.utility_reasoning_effort`, and `config.utility_verbosity`
  - Follows established pattern for model-specific parameter handling
  - Removed hardcoded temperature and reasoning_effort values
- **Settings Button Filtering**: Settings button messages now filtered from thread history rebuild to prevent pollution
- **Meaningful History Check**: "Rebuilding history" status only shows when meaningful conversation history exists
- **Dedicated MCP Intent Handler**: Added `elif intent == "mcp":` block for forced MCP invocation without LLM tool selection

#### Fixed
- **Slash Command Intent Enforcement**: Slash commands with `force_intent: 'mcp'` now correctly invoke MCP tools instead of falling back to text responses
- **Thread History Rebuild**: Settings button UI elements no longer included in conversation context sent to LLM
- **Force Intent Passthrough**: Message metadata now properly passes `force_intent` through event handlers

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