# SQLite Integration Plan for ChatGPT Bots V2

## Executive Summary
This document outlines the integration of SQLite as a persistent storage layer for the ChatGPT Bots system. The database will complement the existing memory-based architecture, providing persistence across restarts while maintaining the stateless design philosophy where Slack/Discord remain the source of truth.

## Current State Analysis

### What's Currently in Memory (Lost on Restart)
1. **Thread State** (`ThreadState` class)
   - Message history (LIMITED TO 20 MESSAGES)
   - Config overrides per thread
   - System prompts per thread
   - Processing state/locks
   - Pending clarifications

2. **Asset Ledger** (`AssetLedger` class)
   - Generated image data (base64 - MEMORY INTENSIVE)
   - Image prompts (truncated to 100 chars)
   - Slack URLs for images
   - Timestamps
   - LIMITED TO 10 RECENT IMAGES

3. **Thread Locks** (`ThreadLockManager`)
   - Active processing locks
   - Thread busy state

4. **User Cache** (`SlackClient.user_cache`)
   - Username
   - Timezone (tz)
   - Timezone label (tz_label)
   - Timezone offset (tz_offset)
   - Must re-fetch from Slack API after restart

5. **Rate Limiting State** (`RateLimitManager`)
   - Circuit breaker state (resets every 60s)
   - Not needed for persistence

## Database Architecture

### Separate Databases Per Platform
- `data/slack.db` - All Slack-related data
- `data/discord.db` - All Discord-related data
- `data/backups/` - Automated backups directory
  - `slack_YYYYMMDD_HHMMSS.db` - Timestamped Slack backups
  - `discord_YYYYMMDD_HHMMSS.db` - Timestamped Discord backups
  - 7-day retention, older backups auto-deleted
- Clean isolation, no cross-platform concerns
- Simple IDs, no prefixing needed
- **NOTE**: Create `data/` and `data/backups/` directories in project root

## Proposed Database Schema

```sql
-- Core thread metadata and configuration
CREATE TABLE threads (
    thread_id TEXT PRIMARY KEY,  -- Format: "channel_id:thread_ts"
    channel_id TEXT NOT NULL,
    thread_ts TEXT NOT NULL,  -- Thread identifier (Slack ts, Discord message ID, etc.)
    config_json TEXT,  -- JSON: Same structure as user config_json
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP,
    INDEX idx_channel_thread (channel_id, thread_ts),
    INDEX idx_last_activity (last_activity)
);

-- Message cache (optional - for faster rebuilds)
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    role TEXT NOT NULL,  -- 'user' or 'assistant'
    content TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    message_ts TEXT,  -- Slack message timestamp
    metadata_json TEXT,  -- Additional message metadata
    FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE,
    INDEX idx_thread_messages (thread_id, timestamp)
);

-- Image metadata and analyses
CREATE TABLE images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    url TEXT NOT NULL UNIQUE,  -- Slack/Discord URL
    image_type TEXT,  -- 'uploaded', 'generated', 'edited'
    prompt TEXT,  -- Full generation/edit prompt
    analysis TEXT,  -- Full vision analysis - stored for ALL processed images:
                    -- - Direct vision requests
                    -- - Internal analysis for image generation enhancement
                    -- - Internal analysis for image edit enhancement
    original_analysis TEXT,  -- For edited images, store pre-edit analysis
    metadata_json TEXT,  -- Size, quality, style, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE,
    INDEX idx_thread_images (thread_id, created_at),
    INDEX idx_url (url)
);

-- User preferences and settings
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,  -- Slack user ID (U123...) or Discord user ID
    username TEXT,
    config_json TEXT,  -- User's configurable preferences:
                       -- {
                       --   "model": "gpt-5",
                       --   "temperature": 0.8,
                       --   "top_p": 1.0,
                       --   "reasoning_effort": "medium",
                       --   "verbosity": "medium",
                       --   "image_size": "1024x1024",
                       --   "image_quality": "hd",
                       --   "image_style": "natural",
                       --   "input_fidelity": "high",
                       --   "detail_level": "auto"
                       -- }
    -- Timezone caching (new)
    timezone TEXT DEFAULT 'UTC',  -- User's timezone
    tz_label TEXT,  -- Timezone label (e.g., "Pacific Standard Time")
    tz_offset INTEGER DEFAULT 0,  -- Offset in seconds from UTC
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP
);
```

## Integration Architecture

### Database Layer (`database.py`)
```python
class DatabaseManager:
    def __init__(self, platform="slack"):
        # Each platform gets its own database file in data directory
        os.makedirs("data", exist_ok=True)  # Create data dir if not exists
        os.makedirs("data/backups", exist_ok=True)  # Create backups dir
        db_path = f"data/{platform}.db"  # data/slack.db, data/discord.db, etc.
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.platform = platform
        
        # Enable WAL mode for better concurrency
        self.conn.execute("PRAGMA journal_mode=WAL")
        
        self.init_schema()
    
    def backup_database(self):
        # Checkpoint WAL file before backup to ensure consistency
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        
        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"data/backups/{self.platform}_{timestamp}.db"
        
        # Use SQLite backup API (will include checkpointed data)
        backup_conn = sqlite3.connect(backup_path)
        self.conn.backup(backup_conn)
        backup_conn.close()
        
        # Clean up old backups (keep last 7 days)
        self.cleanup_old_backups()
    
    def cleanup_old_backups(self):
        # Remove backups older than 7 days
        cutoff = datetime.now() - timedelta(days=7)
        for file in os.listdir("data/backups"):
            if file.startswith(f"{self.platform}_"):
                # Parse timestamp from filename
                # Format: platform_YYYYMMDD_HHMMSS.db
                try:
                    date_str = file.split('_')[1] + file.split('_')[2].split('.')[0]
                    file_date = datetime.strptime(date_str, "%Y%m%d%H%M%S")
                    if file_date < cutoff:
                        os.remove(f"data/backups/{file}")
                except:
                    pass  # Skip malformed filenames
    
    def cleanup_old_threads(self):
        # Remove threads older than 3 months (matching Slack retention)
        cutoff = datetime.now() - timedelta(days=90)
        cursor = self.conn.execute(
            "DELETE FROM threads WHERE last_activity < ?",
            (cutoff,)
        )
        if cursor.rowcount > 0:
            self.log_info(f"Cleaned up {cursor.rowcount} threads older than 3 months")
    
    # Thread operations (clean, no platform prefix needed)
    def save_thread_config(thread_id, config)
    def get_thread_config(thread_id)
    def update_thread_activity(thread_id)
    def get_or_create_thread(thread_id, channel_id, user_id)  # Creates with user defaults
    
    # Message operations
    def cache_message(thread_id, role, content, metadata)
    def get_cached_messages(thread_id)  # No limit - get all messages
    
    # Image operations (NO BASE64 STORAGE)
    def save_image_metadata(thread_id, url, type, prompt, analysis)  # Full prompt & analysis, no base64
    def get_image_analysis_by_url(thread_id, url)  # Thread-isolated lookup
    def find_thread_images(thread_id, image_type=None)
    
    # User operations (simple, clean)
    def get_or_create_user(user_id, username):
        # Creates new user with defaults from config.py BotConfig:
        default_config = {
            # Text generation
            "model": config.gpt_model,
            "temperature": config.default_temperature,
            "top_p": config.default_top_p,
            "reasoning_effort": config.default_reasoning_effort,
            "verbosity": config.default_verbosity,
            # Image generation (gpt-image-1)
            "image_size": config.default_image_size,
            "image_quality": config.default_image_quality,
            "image_style": config.default_image_style,
            "input_fidelity": config.default_input_fidelity,
            # Vision
            "detail_level": config.default_detail_level
        }
    
    def update_user_config(user_id, config)
    def get_user_config(user_id)
    
    # Config hierarchy resolution
    def get_effective_config(thread_id, user_id)  # Merges BotConfig → user → thread
```

## What Moves to DB vs Stays in Memory

### Moves to Database (Persistent)
- Thread configurations
- Image URLs and analyses (NO BASE64 DATA)
- User preferences (including timezone)
- Message cache (unlimited history)
- Full image prompts (not truncated)
- Full vision analyses (not truncated)

### Stays in Memory (Transient)
- Active thread locks
- Current processing state
- Streaming buffers
- Rate limiting state (resets every 60s anyway)
- Base64 image data (never stored in DB)

## Optimizations Enabled by Database

### 1. Instant Thread Recovery
- **Current**: Rebuild from Slack on every restart
- **With DB**: Load cached messages + config instantly
- **Cache Strategy**: 
  - Cache miss (new thread) → Full sync from Slack
  - Cache hit → Use DB cache
  - New messages → Update cache as they arrive

### 2. Complete Image Context
- **Current**: 100-char truncated analysis in AssetLedger
- **With DB**: Full analysis stored for EVERY processed image:
  - Direct vision requests
  - Internal analysis for image generation enhancement
  - Internal analysis for image edit enhancement
- **Clean Slack UI**: Analysis never shown to user, stays in DB
- **Enhanced context**: When building prompts, include full analysis from DB
  - Example: User says "edit the hat" → We send model the full analysis + edit request
  - Model has complete context without polluting Slack thread
- **Thread Isolation**: Each thread's images are isolated - no cross-thread sharing
- **No Base64 Storage**: Only URLs and metadata stored, not raw image data

### 3. Complete Message History
- **Current**: Keep last 20 messages in memory, only 6 for recent context
- **With DB**: Unlimited history, all messages always sent
- **No truncation**: Full conversation context every time
- **Instant Recovery**: No expensive Slack API rebuilds after restart

### 4. Configuration Persistence
- **Current**: Lost on restart
- **With DB**: Per-thread and per-user configs persist
- **Hierarchical**: BotConfig (.env) → User defaults (DB) → Thread overrides (DB)
- **Phase 1**: Backend support only, no user commands

## New Features Added Since Plan (Already Implemented)

### Features to Consider for DB Integration:
1. **URL Image Processing** (`image_url_handler.py`)
   - Downloads and processes external image URLs
   - Currently re-processes on each mention
   - With DB: Store analysis per thread (maintain isolation)

2. **User Timezone Injection**
   - Currently cached in `SlackClient.user_cache`
   - Lost on restart, requires Slack API call
   - With DB: Permanent timezone storage in users table

3. **Pagination Support**
   - Smart message splitting for Slack limits
   - No DB storage needed - rebuilds naturally

4. **Streaming Enhancements**
   - Circuit breaker resets every 60s
   - No DB storage needed

## New Features Enabled (FUTURE - NOT IN PHASE 1)

### 1. User Profiles & Preferences (PHASE 2 - REQUIRES APPROVAL)
```
/config set temperature 0.5  # Sets for current thread AND user default
/config set model gpt-5-mini
/config set reasoning_effort high
/config set image_quality standard
/config show  # Shows current thread settings
/config reset  # Reset thread to user defaults
/defaults set temperature 0.5  # Set user default only
/defaults show  # Show user defaults
```
- Configurable settings: model, temperature, top_p, reasoning_effort, verbosity, image_size, image_quality, image_style, input_fidelity, detail_level
- System-controlled: max_tokens, streaming, web_search, image_format, compression, system_prompt
- Config hierarchy: BotConfig (.env) → User defaults → Thread overrides

### 2. Advanced Image Management
```
/images list  # Show all images in thread
/images search "red car"  # Search by content
/analyze <image_number>  # Re-analyze specific image
```
- Full image history with descriptions
- Search across all thread images
- Reference images by number/description

### 3. Thread Management
```
/history export  # Export conversation
/history search "docker"  # Search messages
/stats  # Show thread statistics
/resume  # Continue from last conversation
```
- Export conversations
- Search across history
- Resume conversations seamlessly

### 4. Unlimited Context
- Full conversation history always available
- No artificial message limits
- Complete context across sessions

## Implementation Approach

### IMPORTANT: Initial Implementation Scope
**PHASE 1 - CORE FUNCTIONALITY ONLY (DO THIS FIRST):**
- Database layer for existing functionality
- Thread message caching (unlimited history)
- Image metadata and full analysis storage
- Thread and user config storage in DB (replaces memory storage)
- Config hierarchy: BotConfig (.env) → User config (DB) → Thread config (DB)
- NO NEW SLASH COMMANDS - existing functionality only
- NO CONFIG MANAGEMENT COMMANDS - configs work but no /config or /defaults commands yet

**PHASE 2 - CONFIG FEATURES (FUTURE - REQUIRES EXPLICIT APPROVAL):**
- User preference commands (/config, /defaults)
- Config management slash commands
- New user-facing features

### Build Order (PHASE 1 ONLY)
1. **Setup** - Create `data/` and `data/backups/` directories, add to .gitignore
2. **Database Layer** - Create `database.py` with all tables and backup functionality
3. **Thread Integration** - Hook ThreadStateManager for message caching and config storage
4. **User Integration** - Create users on first interaction with BotConfig defaults
5. **Config Hierarchy** - Implement config resolution (BotConfig → User → Thread)
6. **Image Management** - Store full analyses and metadata

### Key Integration Points
- `SlackClient.__init__()` - Create DatabaseManager with platform='slack'
- `DiscordClient.__init__()` - Create DatabaseManager with platform='discord'
- `SlackClient._get_or_cache_user_info()` - Store timezone in users table
- `ThreadStateManager` - Receives DB instance from client
- `ThreadStateManager.get_or_create_thread()` - Check DB first before creating new
- `AssetLedger.add_image()` - Store URL and metadata only, NO base64
- `MessageProcessor._handle_image_edit()` - Save full analysis to DB
- `MessageProcessor._handle_vision_without_upload()` - Query DB for analysis
- `MessageProcessor._build_messages()` - Inject DB analysis when image URLs mentioned
  - When preparing messages for model, check for image URLs
  - Pull full analysis from DB and include in context
  - User never sees the analysis in Slack
- `image_url_handler.py` - Store processed URL metadata per thread

## Performance Considerations

### Caching Strategy
- **L1 Cache**: In-memory (current ThreadState)
- **L2 Cache**: SQLite
- **L3 Source**: Slack/Discord API

### Query Optimization
- Indexes on frequently queried columns
- Prepared statements for common queries
- Connection pooling
- WAL mode for concurrent access

### Data Retention
- 3-month retention for threads (matching Slack's workspace retention)
- Daily cleanup removes threads > 90 days old
- 7-day backup retention
- Images metadata kept with parent thread (cascade delete)

## Monitoring & Maintenance

### Health Checks
- Database size monitoring
- Query performance tracking
- Connection pool status
- Cache hit rates

### Backup Strategy
- Daily automated backups during cleanup cycle
- WAL checkpoint before backup (ensures consistency)
- 7-day retention (automatic cleanup of old backups)
- Timestamped format: `platform_YYYYMMDD_HHMMSS.db`
- Stored in `data/backups/` directory
- Uses SQLite backup API for consistency
- Note: WAL mode creates `.db-wal` and `.db-shm` files alongside `.db`

### Debugging Tools
- Query logging
- Performance profiling
- Data integrity checks
- Migration rollback capability

## Key Implementation Principles

### Thread Isolation
- **CRITICAL**: All data must stay within thread boundaries
- No cross-thread data sharing or caching
- Each thread's images, messages, and analyses are isolated
- URL image analyses are stored per-thread, not globally

### Storage Guidelines
- **NEVER store base64 image data in DB** - memory intensive
- Store only URLs, metadata, prompts, and analyses
- Full prompts and analyses (not truncated)
- Message history without limits

### Phase 1 Scope (Current Implementation)
- Database backend for existing functionality only
- No new user-facing commands
- Config support in backend without /config or /defaults commands
- Focus on persistence and performance improvements

## Conclusion

The SQLite integration provides significant benefits:
- **Persistence**: Survive restarts without losing context
- **Performance**: Faster thread recovery, reduced API calls
- **Unlimited Context**: No message limits, full image analyses
- **Clean UI**: Rich metadata without polluting Slack display
- **Reliability**: Consistent experience across sessions

The implementation is designed to be incremental, maintaining backwards compatibility while gradually enhancing the system's capabilities. The database acts as a complement to the stateless architecture, not a replacement for Slack/Discord as the source of truth.