"""
SQLite Database Manager for ChatGPT Bots
Provides persistent storage for threads, messages, images, documents, and user preferences
"""

import sqlite3
import aiosqlite
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
import logging
import asyncio
from logger import LoggerMixin

logger = logging.getLogger(__name__)

# Sentinel distinguishing "argument omitted → preserve existing value" from an explicit
# None (→ clear the column to NULL). Used by the channel_settings setters so the settings
# modal's "inherit from global default" selection stores NULL rather than a literal string.
_UNSET = object()


class DatabaseManager(LoggerMixin):
    """
    Manages SQLite database operations for bot persistence.
    Each platform gets its own database file.
    """
    
    def __init__(self, platform: str = "slack"):
        """
        Initialize database connection for the specified platform.

        Args:
            platform: Platform name (e.g. "slack")
        """
        self.platform = platform

        # Get database directory from config
        from config import BotConfig
        config = BotConfig()
        self.db_dir = config.database_dir

        # Ensure directories exist
        os.makedirs(self.db_dir, exist_ok=True)
        os.makedirs(f"{self.db_dir}/backups", exist_ok=True)

        # Connect to platform-specific database
        self.db_path = f"{self.db_dir}/{platform}.db"
        self.conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,  # Allow multi-threaded access
            isolation_level=None  # Autocommit mode
        )
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        
        # Enable WAL mode for better concurrency
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA busy_timeout=5000")  # 5 second timeout
        
        # Initialize schema
        self.init_schema()
        
        logger.info(f"Database initialized for {platform} at {self.db_path}")

        # For async operations, we'll create connections as needed
        self._async_db_semaphore = asyncio.Semaphore(10)  # Limit concurrent async connections
    
    def init_schema(self):
        """Create database tables if they don't exist."""
        
        # Threads table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                channel_id TEXT NOT NULL,
                thread_ts TEXT NOT NULL,
                config_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for threads
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_channel_thread 
            ON threads(channel_id, thread_ts)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_last_activity 
            ON threads(last_activity)
        """)
        
        # NOTE (Phase S): there is deliberately NO messages table. Slack is the only
        # transcript — context is always rebuilt from conversations.replies. The DB keeps
        # only what Slack doesn't have: config, memory, derived artifacts (images/documents),
        # and thread_summaries (compaction state). See Docs/CHANNEL_TEAMMATE_REDESIGN_PLAN.md §5b.

        # Thread summaries table — rolling compaction store for long threads.
        # summary_text covers everything at or before boundary_ts; refs_json preserves
        # structured references (files/images/links) from the summarized span.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS thread_summaries (
                thread_id TEXT PRIMARY KEY,
                summary_text TEXT NOT NULL,
                boundary_ts TEXT NOT NULL,
                refs_json TEXT,
                updated_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
            )
        """)

        # Images table (no base64 storage)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE,
                message_ts TEXT,  -- Links image to specific message
                image_type TEXT,
                prompt TEXT,
                analysis TEXT,
                original_analysis TEXT,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for images
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_thread_images 
            ON images(thread_id, created_at)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_url 
            ON images(url)
        """)
        
        # Documents table — summary + metadata + Slack ref ONLY (user hard rule,
        # CLAUDE.md pitfall 6a): full content is never at rest. The file lives on
        # Slack's CDN (file_id/url_private) and is re-derived in memory on demand
        # via the read_document tool.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                summary TEXT,
                file_id TEXT,
                url_private TEXT,
                size_bytes INTEGER,
                page_structure TEXT,
                total_pages INTEGER,
                metadata_json TEXT,
                message_ts TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
            )
        """)
        
        # Create indexes for documents
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_thread_documents 
            ON documents(thread_id, created_at)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_filename 
            ON documents(filename)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_message 
            ON documents(message_ts)
        """)
        
        # Users table with timezone support
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT,
                real_name TEXT,
                email TEXT,
                config_json TEXT,
                timezone TEXT DEFAULT 'UTC',
                tz_label TEXT,
                tz_offset INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # User preferences table for settings modal
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                slack_user_id TEXT PRIMARY KEY,
                slack_email TEXT,

                -- Model settings
                model TEXT DEFAULT 'gpt-5.6-sol',
                reasoning_effort TEXT DEFAULT 'medium',
                verbosity TEXT DEFAULT 'low',
                temperature REAL DEFAULT 0.8,
                top_p REAL DEFAULT 1.0,

                -- Feature toggles
                enable_web_search BOOLEAN DEFAULT 1,
                enable_mcp BOOLEAN DEFAULT 1,
                enable_streaming BOOLEAN DEFAULT 1,

                -- Image settings
                image_model TEXT DEFAULT 'gpt-image-2',
                image_size TEXT DEFAULT '1024x1024',
                image_quality TEXT DEFAULT 'auto',
                image_background TEXT DEFAULT 'auto',
                input_fidelity TEXT DEFAULT 'high',
                vision_detail TEXT DEFAULT 'auto',

                -- Metadata
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                updated_at INTEGER DEFAULT (strftime('%s', 'now')),
                settings_completed BOOLEAN DEFAULT 0,

                FOREIGN KEY (slack_user_id) REFERENCES users(user_id)
            )
        """)
        
        # Create index for email lookups
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_prefs_email
            ON user_preferences(slack_email)
        """)

        # Modal sessions table for temporary modal state storage
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS modal_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                modal_type TEXT DEFAULT 'settings',
                state TEXT NOT NULL,
                created_at INTEGER DEFAULT (strftime('%s', 'now')),
                updated_at INTEGER DEFAULT (strftime('%s', 'now'))
            )
        """)

        # Create indexes for modal sessions
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_modal_session_user
            ON modal_sessions(user_id)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_modal_session_created
            ON modal_sessions(created_at)
        """)

        # MCP tools cache table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS mcp_tools (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                server_label TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                description TEXT,
                input_schema TEXT,
                discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_verified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(server_label, tool_name)
            )
        """)

        # Create index for mcp_tools
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_mcp_server_label
            ON mcp_tools(server_label)
        """)

        # Phase 7: per-channel response settings (mode + freeform directives).
        # No row for a channel => global defaults apply (no behavior change).
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS channel_settings (
                channel_id TEXT PRIMARY KEY,
                response_mode TEXT DEFAULT 'tag_only',
                directives TEXT,
                reply_in_channel BOOLEAN DEFAULT 0,
                participation_level TEXT,
                snoozed_until TEXT,
                model TEXT,
                reasoning_effort TEXT,
                verbosity TEXT,
                updated_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_by TEXT
            )
        """)

        # Per-channel durable memory (Phase 9). scope='channel' rows are private to that channel;
        # scope='workspace' rows are shared (read-mostly, admin/manual writes only).
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS channel_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                scope TEXT NOT NULL DEFAULT 'channel',
                content TEXT NOT NULL,
                author TEXT,
                created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_channel_memory_lookup ON channel_memory (scope, channel_id)"
        )

        # Response feedback (Phase H): thumbs signal from native feedback buttons and
        # from +1/-1 reactions on the bot's own messages. One row per
        # (message, user, source); a changed thumb updates the row in place.
        # The participation engine may read per-channel ratios later.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS response_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                thread_ts TEXT,
                message_ts TEXT NOT NULL,
                user_id TEXT NOT NULL,
                signal INTEGER NOT NULL CHECK (signal IN (-1, 1)),
                source TEXT NOT NULL CHECK (source IN ('button', 'reaction')),
                created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (message_ts, user_id, source)
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_response_feedback_channel "
            "ON response_feedback (channel_id, created_ts)"
        )

        self.conn.commit()

        # Run migrations for existing databases
        self._run_migrations()
    
    def _run_migrations(self):
        """Run database migrations to update schema for existing databases."""
        try:
            # Phase F: participation_level + snoozed_until on channel_settings
            cursor = self.conn.execute("PRAGMA table_info(channel_settings)")
            cs_columns = [col[1] for col in cursor.fetchall()]
            if cs_columns and 'participation_level' not in cs_columns:
                self.log_info("DB: Adding participation_level column to channel_settings")
                self.conn.execute("ALTER TABLE channel_settings ADD COLUMN participation_level TEXT")
                self.conn.commit()
            if cs_columns and 'snoozed_until' not in cs_columns:
                self.log_info("DB: Adding snoozed_until column to channel_settings")
                self.conn.execute("ALTER TABLE channel_settings ADD COLUMN snoozed_until TEXT")
                self.conn.commit()
            # Shared per-channel model/effort/verbosity overrides (NULL = inherit)
            for col in ("model", "reasoning_effort", "verbosity"):
                if cs_columns and col not in cs_columns:
                    self.log_info(f"DB: Adding {col} column to channel_settings")
                    self.conn.execute(f"ALTER TABLE channel_settings ADD COLUMN {col} TEXT")
                    self.conn.commit()

            # Check if message_ts column exists in images table
            cursor = self.conn.execute("PRAGMA table_info(images)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'message_ts' not in columns:
                self.log_info("DB: Adding message_ts column to images table")
                self.conn.execute("""
                    ALTER TABLE images 
                    ADD COLUMN message_ts TEXT
                """)
                self.conn.commit()
                self.log_info("DB: Successfully added message_ts column")
            
            # Check if real_name column exists in users table
            cursor = self.conn.execute("PRAGMA table_info(users)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'real_name' not in columns:
                self.log_info("DB: Adding real_name column to users table")
                self.conn.execute("""
                    ALTER TABLE users 
                    ADD COLUMN real_name TEXT
                """)
                self.conn.commit()
                self.log_info("DB: Successfully added real_name column")
            
            # Check if custom_instructions column exists in user_preferences table
            cursor = self.conn.execute("PRAGMA table_info(user_preferences)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'custom_instructions' not in columns:
                self.log_info("DB: Adding custom_instructions column to user_preferences table")
                self.conn.execute("""
                    ALTER TABLE user_preferences 
                    ADD COLUMN custom_instructions TEXT
                """)
                self.conn.commit()
                self.log_info("DB: Successfully added custom_instructions column")
            
            # Check if email column exists in users table
            cursor = self.conn.execute("PRAGMA table_info(users)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'email' not in columns:
                self.log_info("DB: Adding email column to users table")
                self.conn.execute("""
                    ALTER TABLE users 
                    ADD COLUMN email TEXT
                """)
                self.conn.commit()
                self.log_info("DB: Successfully added email column")
            
            # Check if user_preferences table exists
            cursor = self.conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='user_preferences'
            """)
            if not cursor.fetchone():
                self.log_info("DB: Creating user_preferences table")
                self.conn.execute("""
                    CREATE TABLE user_preferences (
                        slack_user_id TEXT PRIMARY KEY,
                        slack_email TEXT,
                        
                        -- Model settings
                        model TEXT DEFAULT 'gpt-5.6-sol',
                        reasoning_effort TEXT DEFAULT 'medium',
                        verbosity TEXT DEFAULT 'low',
                        temperature REAL DEFAULT 0.8,
                        top_p REAL DEFAULT 1.0,
                        
                        -- Feature toggles
                        enable_web_search BOOLEAN DEFAULT 1,
                        enable_streaming BOOLEAN DEFAULT 1,
                        
                        -- Image settings
                        image_model TEXT DEFAULT 'gpt-image-2',
                        image_size TEXT DEFAULT '1024x1024',
                        image_quality TEXT DEFAULT 'auto',
                        image_background TEXT DEFAULT 'auto',
                        input_fidelity TEXT DEFAULT 'high',
                        vision_detail TEXT DEFAULT 'auto',

                        -- Metadata
                        created_at INTEGER DEFAULT (strftime('%s', 'now')),
                        updated_at INTEGER DEFAULT (strftime('%s', 'now')),
                        settings_completed BOOLEAN DEFAULT 0,

                        FOREIGN KEY (slack_user_id) REFERENCES users(user_id)
                    )
                """)
                self.conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_user_prefs_email
                    ON user_preferences(slack_email)
                """)
                self.conn.commit()
                self.log_info("DB: Successfully created user_preferences table")

            # Check if image_quality column exists in user_preferences table
            cursor = self.conn.execute("PRAGMA table_info(user_preferences)")
            columns = [col[1] for col in cursor.fetchall()]

            if 'image_quality' not in columns:
                self.log_info("DB: Adding image_quality column to user_preferences table")
                self.conn.execute("""
                    ALTER TABLE user_preferences
                    ADD COLUMN image_quality TEXT DEFAULT 'auto'
                """)
                self.conn.commit()
                self.log_info("DB: Successfully added image_quality column")

            # Check if image_background column exists in user_preferences table
            cursor = self.conn.execute("PRAGMA table_info(user_preferences)")
            columns = [col[1] for col in cursor.fetchall()]

            if 'image_background' not in columns:
                self.log_info("DB: Adding image_background column to user_preferences table")
                self.conn.execute("""
                    ALTER TABLE user_preferences
                    ADD COLUMN image_background TEXT DEFAULT 'auto'
                """)
                self.conn.commit()
                self.log_info("DB: Successfully added image_background column")

            # Check if enable_mcp column exists in user_preferences table
            cursor = self.conn.execute("PRAGMA table_info(user_preferences)")
            columns = [col[1] for col in cursor.fetchall()]

            if 'enable_mcp' not in columns:
                self.log_info("DB: Adding enable_mcp column to user_preferences table")
                self.conn.execute("""
                    ALTER TABLE user_preferences
                    ADD COLUMN enable_mcp BOOLEAN DEFAULT 1
                """)
                self.conn.commit()
                self.log_info("DB: Successfully added enable_mcp column")

            # Check if image_model column exists in user_preferences table
            cursor = self.conn.execute("PRAGMA table_info(user_preferences)")
            columns = [col[1] for col in cursor.fetchall()]

            if 'image_model' not in columns:
                self.log_info("DB: Adding image_model column to user_preferences table")
                self.conn.execute("""
                    ALTER TABLE user_preferences
                    ADD COLUMN image_model TEXT DEFAULT 'gpt-image-2'
                """)
                # Explicitly set all existing rows to gpt-image-2. The DEFAULT clause
                # above already does this on SQLite, but make the one-time bulk swap
                # explicit so the intent is unambiguous and the row count gets logged.
                # This runs exactly once (the surrounding `if` block guarantees it).
                cursor = self.conn.execute(
                    "UPDATE user_preferences SET image_model = 'gpt-image-2'"
                )
                row_count = cursor.rowcount
                self.conn.commit()
                self.log_info(
                    f"DB: Successfully added image_model column and migrated "
                    f"{row_count} existing user(s) to gpt-image-2"
                )

            # One-time bulk swap: migrate every user still on a pre-5.5 model to gpt-5.5.
            # Gated by a sentinel migration marker so it ran exactly once back when older
            # models were still selectable. (Superseded by the normalizer below, kept so
            # the column exists on databases created between the two migrations.)
            cursor = self.conn.execute("PRAGMA table_info(user_preferences)")
            columns = [col[1] for col in cursor.fetchall()]
            if 'gpt55_migrated' not in columns:
                self.conn.execute("""
                    ALTER TABLE user_preferences
                    ADD COLUMN gpt55_migrated INTEGER DEFAULT 0
                """)
                cursor = self.conn.execute("""
                    UPDATE user_preferences
                    SET model = 'gpt-5.5', gpt55_migrated = 1
                    WHERE gpt55_migrated = 0
                """)
                swapped = cursor.rowcount
                self.conn.commit()
                self.log_info(
                    f"DB: One-time migration — swapped {swapped} user(s) to gpt-5.5"
                )

            self._migrate_gpt56()

            # One-time backfill: mark long-standing users as settings_completed.
            # Earlier versions of the bot only flipped settings_completed=True when
            # the user saved with "global" scope. Users who only ever saved thread-scope
            # configs kept getting the "Please configure your settings" warning on every
            # DM. Backfill anyone whose row was created more than 24h ago — if they've
            # been around that long, they know the bot exists and don't need the gate.
            cursor = self.conn.execute("""
                UPDATE user_preferences
                SET settings_completed = 1
                WHERE settings_completed = 0
                  AND created_at IS NOT NULL
                  AND created_at < (strftime('%s', 'now') - 86400)
            """)
            backfilled = cursor.rowcount
            if backfilled:
                self.conn.commit()
                self.log_info(
                    f"DB: Backfilled settings_completed=1 for {backfilled} pre-existing user(s)"
                )

            # Phase S one-time cleanup: drop the message mirror. Slack is the only
            # transcript now — context is always rebuilt from conversations.replies.
            # Guarded on table existence so it runs exactly once per database; the
            # tagged backup is the rollback path.
            cursor = self.conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='messages'
            """)
            if cursor.fetchone():
                self.backup_database(tag="pre-v3-mirror-drop")
                size_before = os.path.getsize(self.db_path)
                cursor = self.conn.execute("SELECT COUNT(*) FROM messages")
                row_count = cursor.fetchone()[0]
                self.conn.execute("DROP TABLE IF EXISTS messages")
                # Drop the never-read LEGACY documents.summary column (dead since
                # day one). Only on the legacy table shape (content column present)
                # — the D2 schema has a NEW, load-bearing summary column that the
                # D2 migration below (re)creates and populates.
                # ALTER ... DROP COLUMN needs SQLite 3.35+; degrade gracefully below it.
                try:
                    cursor = self.conn.execute("PRAGMA table_info(documents)")
                    doc_columns = [col[1] for col in cursor.fetchall()]
                    if 'summary' in doc_columns and 'content' in doc_columns:
                        self.conn.execute("ALTER TABLE documents DROP COLUMN summary")
                except Exception as col_err:
                    self.log_warning(f"DB: Could not drop documents.summary column: {col_err}")
                self.conn.execute("VACUUM")
                size_after = os.path.getsize(self.db_path)
                self.log_info(
                    f"DB: Mirror-drop migration complete — removed {row_count} cached message "
                    f"row(s), reclaimed {max(0, size_before - size_after):,} bytes "
                    f"(backup tagged pre-v3-mirror-drop in {self.db_dir}/backups)"
                )

            # Doc-architecture (D2) one-time cleanup: drop documents.content.
            # Same hard rule as the mirror drop — no file/document content at rest;
            # rows keep summary + metadata + the Slack CDN ref. Guarded on the
            # content column existing so it runs exactly once per database.
            cursor = self.conn.execute("PRAGMA table_info(documents)")
            doc_columns = [col[1] for col in cursor.fetchall()]
            if 'content' in doc_columns:
                self.backup_database(tag="pre-v3-doc-content-drop")
                size_before = os.path.getsize(self.db_path)
                # Ensure the new columns exist before synthesizing summaries
                for col_name, col_type in (("summary", "TEXT"), ("file_id", "TEXT"),
                                           ("url_private", "TEXT"), ("size_bytes", "INTEGER")):
                    if col_name not in doc_columns:
                        self.conn.execute(f"ALTER TABLE documents ADD COLUMN {col_name} {col_type}")
                # Mechanical summary synthesis for legacy rows: labeled excerpt of
                # the stored content (cheap, safe; rows are ≤30 days old).
                cursor = self.conn.execute("""
                    UPDATE documents
                    SET summary = '[excerpt of original — full document available via read_document]' || char(10)
                                  || substr(content, 1, 1500)
                    WHERE (summary IS NULL OR summary = '') AND content IS NOT NULL
                """)
                synthesized = cursor.rowcount
                try:
                    self.conn.execute("ALTER TABLE documents DROP COLUMN content")
                    self.conn.execute("VACUUM")
                    size_after = os.path.getsize(self.db_path)
                    self.log_info(
                        f"DB: Doc-content-drop migration complete — synthesized {synthesized} "
                        f"summary(ies), reclaimed {max(0, size_before - size_after):,} bytes "
                        f"(backup tagged pre-v3-doc-content-drop in {self.db_dir}/backups)"
                    )
                except Exception as col_err:
                    # SQLite < 3.35 can't DROP COLUMN; content stays but is never
                    # read or written again. Log loudly — this violates the
                    # no-content-at-rest rule until SQLite is upgraded.
                    self.log_warning(f"DB: Could not drop documents.content column: {col_err}")
                self.conn.commit()

            # Check if mcp_tools table exists
            cursor = self.conn.execute("""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='mcp_tools'
            """)
            if not cursor.fetchone():
                self.log_info("DB: Creating mcp_tools table")
                self.conn.execute("""
                    CREATE TABLE mcp_tools (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        server_label TEXT NOT NULL,
                        tool_name TEXT NOT NULL,
                        description TEXT,
                        input_schema TEXT,
                        discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_verified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(server_label, tool_name)
                    )
                """)
                self.conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_mcp_server_label
                    ON mcp_tools(server_label)
                """)
                self.conn.commit()
                self.log_info("DB: Successfully created mcp_tools table")
        except Exception as e:
            self.log_error(f"DB: Migration error: {e}", exc_info=True)

    def _migrate_gpt56(self):
        """GPT-5.6 model-lineup migration (2026-07-09).

        Two parts, both safe to run on every startup:
        1. ONE-TIME (sentinel `gpt56_migrated` column, same pattern as the
           gpt55/gpt-image-2 swaps): move EVERYONE's default model to
           gpt-5.6-sol with medium reasoning. Users can re-customize globally
           and per channel/thread afterward — this only resets the default.
        2. EVERY-STARTUP normalizer: only gpt-5.6-sol/terra/luna and gpt-5.5
           are selectable; any other stored model (user prefs or per-thread
           overrides) coerces to gpt-5.6-sol, and stored reasoning efforts a
           model rejects are clamped (`minimal` is a 400 on 5.6 -> none;
           `max` doesn't exist on 5.5 -> xhigh). Guarantees the API layer
           never receives a dropped model name or an unsupported effort.
        """
        cursor = self.conn.execute("PRAGMA table_info(user_preferences)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'gpt56_migrated' not in columns:
            self.conn.execute("""
                ALTER TABLE user_preferences
                ADD COLUMN gpt56_migrated INTEGER DEFAULT 0
            """)
            cursor = self.conn.execute("""
                UPDATE user_preferences
                SET model = 'gpt-5.6-sol', reasoning_effort = 'medium', gpt56_migrated = 1
                WHERE gpt56_migrated = 0
            """)
            swapped = cursor.rowcount
            self.conn.commit()
            self.log_info(
                f"DB: One-time GPT-5.6 migration — swapped {swapped} user(s) to "
                f"gpt-5.6-sol with medium reasoning"
            )

        supported = "('gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-5.6-luna', 'gpt-5.5')"
        cursor = self.conn.execute(f"""
            UPDATE user_preferences
            SET model = 'gpt-5.6-sol'
            WHERE model IS NOT NULL AND model NOT IN {supported}
        """)
        if cursor.rowcount:
            self.log_info(
                f"DB: Normalized {cursor.rowcount} user(s) from dropped models to gpt-5.6-sol"
            )
        cursor = self.conn.execute("""
            UPDATE user_preferences
            SET reasoning_effort = 'none'
            WHERE model LIKE 'gpt-5.6%' AND reasoning_effort = 'minimal'
        """)
        if cursor.rowcount:
            self.log_info(
                f"DB: Clamped reasoning minimal->none for {cursor.rowcount} user(s) on 5.6 models"
            )
        cursor = self.conn.execute("""
            UPDATE user_preferences
            SET reasoning_effort = 'xhigh'
            WHERE model = 'gpt-5.5' AND reasoning_effort = 'max'
        """)
        if cursor.rowcount:
            self.log_info(
                f"DB: Clamped reasoning max->xhigh for {cursor.rowcount} user(s) on gpt-5.5"
            )
        cursor = self.conn.execute(f"""
            UPDATE threads
            SET config_json = json_set(config_json, '$.model', 'gpt-5.6-sol')
            WHERE config_json IS NOT NULL
              AND json_extract(config_json, '$.model') IS NOT NULL
              AND json_extract(config_json, '$.model') NOT IN {supported}
        """)
        if cursor.rowcount:
            self.log_info(
                f"DB: Normalized {cursor.rowcount} thread override(s) to gpt-5.6-sol"
            )
        cursor = self.conn.execute("""
            UPDATE threads
            SET config_json = json_set(config_json, '$.reasoning_effort', 'none')
            WHERE config_json IS NOT NULL
              AND json_extract(config_json, '$.model') LIKE 'gpt-5.6%'
              AND json_extract(config_json, '$.reasoning_effort') = 'minimal'
        """)
        if cursor.rowcount:
            self.log_info(
                f"DB: Clamped {cursor.rowcount} thread override(s) minimal->none on 5.6 models"
            )
        cursor = self.conn.execute("""
            UPDATE threads
            SET config_json = json_set(config_json, '$.reasoning_effort', 'xhigh')
            WHERE config_json IS NOT NULL
              AND json_extract(config_json, '$.model') = 'gpt-5.5'
              AND json_extract(config_json, '$.reasoning_effort') = 'max'
        """)
        if cursor.rowcount:
            self.log_info(
                f"DB: Clamped {cursor.rowcount} thread override(s) max->xhigh on gpt-5.5"
            )
        self.conn.commit()

    # Thread operations
    def get_or_create_thread(self, thread_id: str, channel_id: str, user_id: Optional[str] = None) -> Dict:
        """
        Get existing thread or create new one with user defaults.
        
        Args:
            thread_id: Thread identifier (channel_id:thread_ts format)
            channel_id: Channel ID
            user_id: Optional user ID to copy defaults from
            
        Returns:
            Thread data dictionary
        """
        self.log_debug(f"DB: get_or_create_thread - thread={thread_id}, channel={channel_id}, user={user_id}")
        
        # Try to get existing thread
        cursor = self.conn.execute(
            "SELECT * FROM threads WHERE thread_id = ?",
            (thread_id,)
        )
        row = cursor.fetchone()
        
        if row:
            # Update last activity
            self.update_thread_activity(thread_id)
            self.log_debug(f"DB: Found existing thread {thread_id}")
            return dict(row)
        
        # Create new thread
        thread_ts = thread_id.split(":", 1)[1] if ":" in thread_id else thread_id
        
        # Get user config if user_id provided
        config = {}
        if user_id:
            user_config = self.get_user_config(user_id)
            if user_config:
                config = user_config
                self.log_debug(f"DB: Applied user config for {user_id} to new thread")
        
        try:
            self.conn.execute("""
                INSERT INTO threads (thread_id, channel_id, thread_ts, config_json)
                VALUES (?, ?, ?, ?)
            """, (thread_id, channel_id, thread_ts, json.dumps(config) if config else None))
            
            self.log_info(f"DB: Created new thread {thread_id}")
            
        except Exception as e:
            self.log_error(f"DB: Failed to create thread {thread_id} - {e}", exc_info=True)
            raise
        
        return self.get_or_create_thread(thread_id, channel_id)
    def save_thread_config(self, thread_id: str, config: Dict):
        """
        Save thread configuration.
        
        Args:
            thread_id: Thread identifier
            config: Configuration dictionary
        """
        self.conn.execute("""
            UPDATE threads 
            SET config_json = ?, updated_at = CURRENT_TIMESTAMP
            WHERE thread_id = ?
        """, (json.dumps(config), thread_id))
        
        logger.debug(f"Saved config for thread {thread_id}")
    
    def get_thread_config(self, thread_id: str) -> Optional[Dict]:
        """
        Get thread configuration.
        
        Args:
            thread_id: Thread identifier
            
        Returns:
            Configuration dictionary or None
        """
        cursor = self.conn.execute(
            "SELECT config_json FROM threads WHERE thread_id = ?",
            (thread_id,)
        )
        row = cursor.fetchone()
        
        if row and row["config_json"]:
            return json.loads(row["config_json"])
        
        return None
    
    def get_channel_settings(self, channel_id: str) -> Optional[Dict]:
        """Get per-channel settings (Phase 7). Returns a dict or None if the channel has no row."""
        cursor = self.conn.execute(
            "SELECT response_mode, directives, reply_in_channel, participation_level, "
            "snoozed_until, model, reasoning_effort, verbosity, updated_ts, updated_by "
            "FROM channel_settings WHERE channel_id = ?",
            (channel_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "response_mode": row["response_mode"],
            "directives": row["directives"],
            "reply_in_channel": bool(row["reply_in_channel"]),
            "participation_level": row["participation_level"],
            "snoozed_until": row["snoozed_until"],
            "model": row["model"],
            "reasoning_effort": row["reasoning_effort"],
            "verbosity": row["verbosity"],
            "updated_ts": row["updated_ts"],
            "updated_by": row["updated_by"],
        }

    def set_channel_settings(self, channel_id: str, response_mode=_UNSET,
                             directives=_UNSET, reply_in_channel=_UNSET,
                             participation_level=_UNSET, snoozed_until=_UNSET,
                             model=_UNSET, reasoning_effort=_UNSET, verbosity=_UNSET,
                             updated_by: Optional[str] = None):
        """Upsert per-channel settings (Phase 7; Phase F adds participation_level/snoozed_until).

        Omitted fields are preserved. An explicit value sets it; an explicit None CLEARS it
        (→ NULL) so the modal's "inherit from global default" stores NULL rather than a literal
        string (NULL then resolves to the global default at read time). snoozed_until=None
        clears an active snooze.
        """
        existing = self.get_channel_settings(channel_id) or {}
        new_mode = existing.get("response_mode", "tag_only") if response_mode is _UNSET else response_mode
        new_dir = existing.get("directives") if directives is _UNSET else directives
        new_ric = existing.get("reply_in_channel", False) if reply_in_channel is _UNSET else reply_in_channel
        new_level = existing.get("participation_level") if participation_level is _UNSET else participation_level
        new_snooze = existing.get("snoozed_until") if snoozed_until is _UNSET else snoozed_until
        new_model = existing.get("model") if model is _UNSET else model
        new_effort = existing.get("reasoning_effort") if reasoning_effort is _UNSET else reasoning_effort
        new_verb = existing.get("verbosity") if verbosity is _UNSET else verbosity
        self.conn.execute("""
            INSERT INTO channel_settings (channel_id, response_mode, directives, reply_in_channel,
                                          participation_level, snoozed_until, model, reasoning_effort,
                                          verbosity, updated_ts, updated_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
            ON CONFLICT(channel_id) DO UPDATE SET
                response_mode=excluded.response_mode,
                directives=excluded.directives,
                reply_in_channel=excluded.reply_in_channel,
                participation_level=excluded.participation_level,
                snoozed_until=excluded.snoozed_until,
                model=excluded.model,
                reasoning_effort=excluded.reasoning_effort,
                verbosity=excluded.verbosity,
                updated_ts=CURRENT_TIMESTAMP,
                updated_by=excluded.updated_by
        """, (channel_id, new_mode, new_dir, 1 if new_ric else 0, new_level, new_snooze,
              new_model, new_effort, new_verb, updated_by))
        self.conn.commit()
        logger.debug(f"Saved channel_settings for {channel_id}: mode={new_mode}, level={new_level}")

    # --- Per-channel memory (Phase 9) ---
    def get_channel_memory(self, channel_id: str) -> List[Dict]:
        """Return durable memory visible to a channel: its own channel-scope rows + shared
        workspace-scope rows. A channel NEVER sees another channel's channel-scope rows."""
        cursor = self.conn.execute(
            "SELECT id, channel_id, scope, content, author, created_ts, updated_ts "
            "FROM channel_memory WHERE (scope = 'channel' AND channel_id = ?) OR scope = 'workspace' "
            "ORDER BY updated_ts ASC",
            (channel_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def add_channel_memory(self, channel_id: str, content: str, scope: str = "channel",
                           author: Optional[str] = None) -> int:
        """Insert a memory row; returns the new id."""
        cursor = self.conn.execute(
            "INSERT INTO channel_memory (channel_id, scope, content, author) VALUES (?, ?, ?, ?)",
            (channel_id, scope, content, author)
        )
        self.conn.commit()
        logger.debug(f"Added channel_memory for {channel_id} (scope={scope})")
        return cursor.lastrowid

    def update_channel_memory(self, memory_id: int, content: str):
        """Update an existing memory row's content (and updated_ts)."""
        self.conn.execute(
            "UPDATE channel_memory SET content = ?, updated_ts = CURRENT_TIMESTAMP WHERE id = ?",
            (content, memory_id)
        )
        self.conn.commit()

    def delete_channel_memory(self, memory_id: int):
        """Delete a memory row (manual forget / cap eviction)."""
        self.conn.execute("DELETE FROM channel_memory WHERE id = ?", (memory_id,))
        self.conn.commit()

    # --- Response feedback (Phase H) ---
    def record_response_feedback(self, channel_id: str, thread_ts: Optional[str],
                                 message_ts: str, user_id: str, signal: int,
                                 source: str) -> None:
        """Upsert one feedback signal. A user changing their thumb (same message,
        same source) updates the existing row rather than adding a second vote."""
        self.conn.execute("""
            INSERT INTO response_feedback (channel_id, thread_ts, message_ts, user_id, signal, source)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(message_ts, user_id, source) DO UPDATE SET
                signal=excluded.signal,
                updated_ts=CURRENT_TIMESTAMP
        """, (channel_id, thread_ts, message_ts, user_id, signal, source))
        self.conn.commit()
        logger.debug(f"Recorded response feedback {signal:+d} ({source}) on {channel_id}:{message_ts}")

    def delete_response_feedback(self, message_ts: str, user_id: str, source: str) -> None:
        """Remove a feedback row (future reaction_removed handling)."""
        self.conn.execute(
            "DELETE FROM response_feedback WHERE message_ts = ? AND user_id = ? AND source = ?",
            (message_ts, user_id, source)
        )
        self.conn.commit()

    def get_channel_feedback_ratio(self, channel_id: str, days: int = 30):
        """(positive, negative, ratio) for a channel's recent feedback.

        ratio is positive/(positive+negative), or None when there's no feedback —
        callers must treat None as "no signal", not as zero. Read-only plumbing for
        the participation engine; not wired into decisions yet."""
        cursor = self.conn.execute(
            "SELECT "
            "  SUM(CASE WHEN signal > 0 THEN 1 ELSE 0 END) AS positive, "
            "  SUM(CASE WHEN signal < 0 THEN 1 ELSE 0 END) AS negative "
            "FROM response_feedback "
            "WHERE channel_id = ? AND created_ts >= datetime('now', ?)",
            (channel_id, f"-{int(days)} days")
        )
        row = cursor.fetchone()
        positive = row["positive"] or 0
        negative = row["negative"] or 0
        total = positive + negative
        return positive, negative, (positive / total if total else None)

    def update_thread_activity(self, thread_id: str):
        """
        Update thread's last activity timestamp.
        
        Args:
            thread_id: Thread identifier
        """
        self.conn.execute("""
            UPDATE threads 
            SET last_activity = CURRENT_TIMESTAMP
            WHERE thread_id = ?
        """, (thread_id,))
    
    # Message operations
    
    # Thread summary operations (Phase S — rolling compaction store)
    def get_thread_summary(self, thread_id: str) -> Optional[Dict]:
        """
        Get the compaction summary row for a thread, if one exists.

        Returns:
            Dict with summary_text, boundary_ts, refs (parsed list), updated_ts — or None.
        """
        cursor = self.conn.execute(
            "SELECT * FROM thread_summaries WHERE thread_id = ?", (thread_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        summary = dict(row)
        summary["refs"] = json.loads(summary["refs_json"]) if summary.get("refs_json") else []
        return summary

    def save_thread_summary(self, thread_id: str, summary_text: str, boundary_ts: str,
                            refs: Optional[List[Dict]] = None):
        """
        Upsert the compaction summary for a thread (one row per thread, rolling).

        Args:
            thread_id: Thread identifier
            summary_text: Summary covering everything at or before boundary_ts
            boundary_ts: Slack ts of the newest message covered by the summary
            refs: Structured refs (files/images/links) from the summarized span
        """
        self.conn.execute("""
            INSERT INTO thread_summaries (thread_id, summary_text, boundary_ts, refs_json, updated_ts)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(thread_id) DO UPDATE SET
                summary_text = excluded.summary_text,
                boundary_ts = excluded.boundary_ts,
                refs_json = excluded.refs_json,
                updated_ts = CURRENT_TIMESTAMP
        """, (thread_id, summary_text, boundary_ts,
              json.dumps(refs) if refs else None))
        self.log_info(f"DB: Saved thread summary for {thread_id} (boundary_ts={boundary_ts})")

    def delete_thread_summary(self, thread_id: str):
        """Delete the compaction summary for a thread."""
        self.conn.execute("DELETE FROM thread_summaries WHERE thread_id = ?", (thread_id,))

    # Image operations
    def save_image_metadata(self, thread_id: str, url: str, image_type: str,
                           prompt: Optional[str] = None, analysis: Optional[str] = None,
                           original_analysis: Optional[str] = None, metadata: Optional[Dict] = None,
                           message_ts: Optional[str] = None):
        """
        Save image metadata (NO base64 data).
        
        Args:
            thread_id: Thread identifier
            url: Image URL
            image_type: Type of image (uploaded/generated/edited)
            prompt: Full generation/edit prompt
            analysis: Full vision analysis
            original_analysis: For edited images, the pre-edit analysis
            metadata: Additional metadata
            message_ts: Message timestamp to link image to specific message
        """
        self.log_debug(f"DB: Saving image - thread={thread_id}, url={url[:100]}, "
                      f"type={image_type}, has_analysis={bool(analysis)}, "
                      f"analysis_len={len(analysis) if analysis else 0}, "
                      f"prompt_len={len(prompt) if prompt else 0}")
        
        try:
            # Merge-preserving upsert (F1): a later write over the same URL (the
            # post-refresh Slack rebuild saves the uploaded file with an empty caption,
            # and the ledger issues its own upsert) must NOT erase the non-empty
            # prompt/analysis/type/generation_id an earlier write already recorded.
            # Existing non-empty values win over incoming empties; message_ts fills in.
            self.conn.execute("""
                INSERT INTO images
                (thread_id, url, image_type, prompt, analysis, original_analysis, metadata_json, message_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    thread_id = excluded.thread_id,
                    image_type = COALESCE(NULLIF(images.image_type, ''), excluded.image_type),
                    prompt = COALESCE(NULLIF(images.prompt, ''), excluded.prompt),
                    analysis = COALESCE(NULLIF(images.analysis, ''), excluded.analysis),
                    original_analysis = COALESCE(NULLIF(images.original_analysis, ''), excluded.original_analysis),
                    metadata_json = COALESCE(NULLIF(images.metadata_json, ''), excluded.metadata_json),
                    message_ts = COALESCE(excluded.message_ts, images.message_ts)
            """, (thread_id, url, image_type, prompt, analysis, original_analysis,
                  json.dumps(metadata) if metadata else None, message_ts))

            self.log_info(f"DB: Successfully saved image metadata for {url[:50]}... in thread {thread_id}")

        except sqlite3.IntegrityError as e:
            self.log_warning(f"DB: Image metadata already exists for {url[:50]}...: {e}")
        except Exception as e:
            self.log_error(f"DB: Failed to save image metadata - {e}", exc_info=True)
            raise
    def get_image_analysis_by_url(self, thread_id: str, url: str) -> Optional[Dict]:
        """
        Get image analysis by URL (thread-isolated).
        
        Args:
            thread_id: Thread identifier
            url: Image URL
            
        Returns:
            Image metadata dictionary or None
        """
        cursor = self.conn.execute("""
            SELECT * FROM images 
            WHERE thread_id = ? AND url = ?
        """, (thread_id, url))
        
        row = cursor.fetchone()
        if row:
            img = dict(row)
            if img.get("metadata_json"):
                img["metadata"] = json.loads(img["metadata_json"])
                del img["metadata_json"]
            return img
        
        return None
    
    def get_images_by_message(self, thread_id: str, message_ts: str) -> List[Dict]:
        """
        Get images associated with a specific message.
        
        Args:
            thread_id: Thread identifier
            message_ts: Message timestamp
            
        Returns:
            List of image metadata dictionaries
        """
        cursor = self.conn.execute("""
            SELECT * FROM images 
            WHERE thread_id = ? AND message_ts = ?
            ORDER BY created_at ASC
        """, (thread_id, message_ts))
        
        images = []
        for row in cursor:
            img = dict(row)
            if img.get("metadata_json"):
                img["metadata"] = json.loads(img["metadata_json"])
                del img["metadata_json"]
            images.append(img)
        
        return images
    
    def find_thread_images(self, thread_id: str, image_type: Optional[str] = None) -> List[Dict]:
        """
        Find all images for a thread.
        
        Args:
            thread_id: Thread identifier
            image_type: Optional filter by image type
            
        Returns:
            List of image metadata dictionaries
        """
        if image_type:
            cursor = self.conn.execute("""
                SELECT * FROM images 
                WHERE thread_id = ? AND image_type = ?
                ORDER BY created_at ASC
            """, (thread_id, image_type))
        else:
            cursor = self.conn.execute("""
                SELECT * FROM images 
                WHERE thread_id = ?
                ORDER BY created_at ASC
            """, (thread_id,))
        
        images = []
        for row in cursor:
            img = dict(row)
            if img.get("metadata_json"):
                img["metadata"] = json.loads(img["metadata_json"])
                del img["metadata_json"]
            images.append(img)
        
        return images
    
    def get_latest_thread_image(self, thread_id: str) -> Optional[Dict]:
        """
        Get the most recent image for a thread.
        
        Args:
            thread_id: Thread identifier
            
        Returns:
            Image metadata dictionary or None
        """
        cursor = self.conn.execute("""
            SELECT * FROM images 
            WHERE thread_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (thread_id,))
        
        row = cursor.fetchone()
        if row:
            img = dict(row)
            if img.get("metadata_json"):
                img["metadata"] = json.loads(img["metadata_json"])
                del img["metadata_json"]
            return img
        
        return None
    
    # User preferences operations
    
    def get_user_preferences(self, user_id: str) -> Optional[Dict]:
        """
        Get user preferences for settings modal.
        
        Args:
            user_id: Slack user ID
            
        Returns:
            User preferences dictionary or None if not found
        """
        cursor = self.conn.execute(
            "SELECT * FROM user_preferences WHERE slack_user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        
        if row:
            prefs = dict(row)
            # Convert SQLite boolean (0/1) to Python boolean
            prefs['enable_web_search'] = bool(prefs.get('enable_web_search', 1))
            prefs['enable_streaming'] = bool(prefs.get('enable_streaming', 1))
            prefs['enable_mcp'] = bool(prefs.get('enable_mcp', 1))
            prefs['settings_completed'] = bool(prefs.get('settings_completed', 0))
            return prefs
        
        return None
    
    def create_default_user_preferences(self, user_id: str, email: Optional[str] = None) -> Dict:
        """
        Create default user preferences based on environment variables.
        
        Args:
            user_id: Slack user ID
            email: Optional user email
            
        Returns:
            Dictionary of default preferences
        """
        from config import config
        
        defaults = {
            'slack_user_id': user_id,
            'slack_email': email,
            'model': config.gpt_model,
            'reasoning_effort': config.default_reasoning_effort,
            'verbosity': config.default_verbosity,
            'temperature': config.default_temperature,
            'top_p': config.default_top_p,
            'enable_web_search': config.enable_web_search,
            'enable_streaming': config.enable_streaming,
            'image_model': config.image_model,
            'image_size': config.default_image_size,
            'image_quality': config.default_image_quality,
            'image_background': config.default_image_background,
            'input_fidelity': config.default_input_fidelity,
            'vision_detail': config.default_detail_level,
            'settings_completed': False
        }

        try:
            # Insert with defaults
            self.conn.execute("""
                INSERT INTO user_preferences
                (slack_user_id, slack_email, model, reasoning_effort, verbosity,
                 temperature, top_p, enable_web_search, enable_streaming,
                 image_model, image_size, image_quality, image_background, input_fidelity, vision_detail, settings_completed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, email, defaults['model'],
                defaults['reasoning_effort'], defaults['verbosity'],
                defaults['temperature'], defaults['top_p'],
                1 if defaults['enable_web_search'] else 0,
                1 if defaults['enable_streaming'] else 0,
                defaults['image_model'],
                defaults['image_size'], defaults['image_quality'],
                defaults['image_background'], defaults['input_fidelity'],
                defaults['vision_detail'], 0
            ))
            
            self.log_info(f"DB: Created default preferences for user {user_id}")
            
        except Exception as e:
            self.log_error(f"DB: Failed to create default preferences for {user_id} - {e}")
            
        return defaults
    
    def update_user_preferences(self, user_id: str, preferences: Dict) -> bool:
        """
        Update user preferences.
        
        Args:
            user_id: Slack user ID
            preferences: Dictionary of preferences to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Build dynamic UPDATE query based on provided fields
            updates = []
            values = []
            
            for field in ['model', 'reasoning_effort', 'verbosity', 'temperature',
                         'top_p', 'image_size', 'image_quality', 'image_background',
                         'input_fidelity', 'vision_detail',
                         'slack_email', 'settings_completed', 'custom_instructions']:
                if field in preferences:
                    updates.append(f"{field} = ?")
                    values.append(preferences[field])
            
            # Handle boolean fields
            for field in ['enable_web_search', 'enable_streaming', 'enable_mcp']:
                if field in preferences:
                    updates.append(f"{field} = ?")
                    values.append(1 if preferences[field] else 0)
            
            # Always update timestamp
            updates.append("updated_at = strftime('%s', 'now')")
            
            # Add user_id for WHERE clause
            values.append(user_id)
            
            query = f"""
                UPDATE user_preferences 
                SET {', '.join(updates)}
                WHERE slack_user_id = ?
            """
            
            self.conn.execute(query, values)
            self.log_info(f"DB: Updated preferences for user {user_id}")
            return True
            
        except Exception as e:
            self.log_error(f"DB: Failed to update preferences for {user_id} - {e}")
            return False
    
    # Document operations
    
    def save_document(self, thread_id: str, filename: str, mime_type: str,
                     summary: Optional[str] = None, file_id: Optional[str] = None,
                     url_private: Optional[str] = None, size_bytes: Optional[int] = None,
                     page_structure: Optional[Dict] = None,
                     total_pages: Optional[int] = None,
                     metadata: Optional[Dict] = None, message_ts: Optional[str] = None):
        """
        Save document summary + metadata + Slack CDN ref. Full content is NEVER
        persisted (CLAUDE.md pitfall 6a) — it is re-derived in memory on demand.

        Args:
            thread_id: Thread identifier
            filename: Original filename
            mime_type: Document MIME type
            summary: Attach-time summary (the only content-bearing field)
            file_id: Slack file id (read_document lookup key)
            url_private: Slack CDN URL for authenticated re-download
            size_bytes: Original file size
            page_structure: Optional page/sheet structure info as dict
            total_pages: Total page/sheet count
            metadata: Additional metadata (size, author, etc.)
            message_ts: Message timestamp to link document to specific message
        """
        self.log_debug(f"DB: Saving document - thread={thread_id}, filename={filename}, "
                      f"summary_len={len(summary) if summary else 0}, pages={total_pages}")

        try:
            self.conn.execute("""
                INSERT INTO documents
                (thread_id, filename, mime_type, summary, file_id, url_private,
                 size_bytes, page_structure, total_pages, metadata_json, message_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (thread_id, filename, mime_type, summary, file_id, url_private,
                  size_bytes,
                  json.dumps(page_structure) if page_structure else None,
                  total_pages,
                  json.dumps(metadata) if metadata else None, message_ts))
            
            # Update thread activity
            self.update_thread_activity(thread_id)
            
            self.log_info(f"DB: Successfully saved document {filename} for thread {thread_id}")
            
        except Exception as e:
            self.log_error(f"DB: Failed to save document {filename} - {e}", exc_info=True)
            raise
    
    def get_thread_documents(self, thread_id: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Get all documents for a thread.
        
        Args:
            thread_id: Thread identifier
            limit: Optional limit on number of documents returned
            
        Returns:
            List of document dictionaries
        """
        query = """
            SELECT * FROM documents 
            WHERE thread_id = ? 
            ORDER BY created_at ASC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor = self.conn.execute(query, (thread_id,))
        documents = []
        
        for row in cursor:
            doc = dict(row)
            if doc.get("page_structure"):
                doc["page_structure"] = json.loads(doc["page_structure"])
            if doc.get("metadata_json"):
                doc["metadata"] = json.loads(doc["metadata_json"])
                del doc["metadata_json"]
            documents.append(doc)
        
        return documents
    
    def get_document_by_filename(self, thread_id: str, filename: str) -> Optional[Dict]:
        """
        Get a specific document by filename within a thread.
        
        Args:
            thread_id: Thread identifier
            filename: Document filename
            
        Returns:
            Document dictionary or None
        """
        cursor = self.conn.execute("""
            SELECT * FROM documents 
            WHERE thread_id = ? AND filename = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (thread_id, filename))
        
        row = cursor.fetchone()
        if row:
            doc = dict(row)
            if doc.get("page_structure"):
                doc["page_structure"] = json.loads(doc["page_structure"])
            if doc.get("metadata_json"):
                doc["metadata"] = json.loads(doc["metadata_json"])
                del doc["metadata_json"]
            return doc
        
        return None
    
    def delete_old_documents(self, days: int = 90):
        """
        Delete documents older than specified days.
        
        Args:
            days: Number of days to retain documents (default 90)
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        cursor = self.conn.execute("""
            DELETE FROM documents 
            WHERE created_at < ?
        """, (cutoff,))
        
        if cursor.rowcount > 0:
            self.log_info(f"DB: Cleaned up {cursor.rowcount} documents older than {days} days")
    
    # User operations
    
    def get_or_create_user(self, user_id: str, username: Optional[str] = None) -> Dict:
        """
        Get existing user or create new one with defaults.
        
        Args:
            user_id: User identifier
            username: Optional username
            
        Returns:
            User data dictionary
        """
        # Try to get existing user
        cursor = self.conn.execute(
            "SELECT * FROM users WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        
        if row:
            # Update last seen
            self.conn.execute("""
                UPDATE users SET last_seen = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """, (user_id,))
            return dict(row)
        
        # Create new user with defaults from config
        from config import BotConfig
        config = BotConfig()
        
        default_config = {
            "model": config.gpt_model,
            "temperature": config.default_temperature,
            "top_p": config.default_top_p,
            "reasoning_effort": config.default_reasoning_effort,
            "verbosity": config.default_verbosity,
            "image_model": config.image_model,
            "image_size": config.default_image_size,
            "image_quality": config.default_image_quality,
            "image_background": config.default_image_background,
            "input_fidelity": config.default_input_fidelity,
            "detail_level": config.default_detail_level
        }
        
        self.conn.execute("""
            INSERT INTO users (user_id, username, config_json)
            VALUES (?, ?, ?)
        """, (user_id, username, json.dumps(default_config)))
        
        return self.get_or_create_user(user_id, username)
    
    def update_user_config(self, user_id: str, config: Dict):
        """
        Update user configuration.
        
        Args:
            user_id: User identifier
            config: Configuration dictionary
        """
        self.conn.execute("""
            UPDATE users 
            SET config_json = ?, last_seen = CURRENT_TIMESTAMP
            WHERE user_id = ?
        """, (json.dumps(config), user_id))
        
        logger.debug(f"Updated config for user {user_id}")
    
    def get_user_config(self, user_id: str) -> Optional[Dict]:
        """
        Get user configuration.
        
        Args:
            user_id: User identifier
            
        Returns:
            Configuration dictionary or None
        """
        cursor = self.conn.execute(
            "SELECT config_json FROM users WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        
        if row and row["config_json"]:
            return json.loads(row["config_json"])
        
        return None
    
    def save_user_info(self, user_id: str, username: str = None, real_name: str = None,
                       email: str = None, timezone: str = None, tz_label: str = None, tz_offset: int = None):
        """
        Save comprehensive user information.
        
        Args:
            user_id: User identifier
            username: Display/username
            real_name: User's real name
            email: User's email address
            timezone: Timezone string
            tz_label: Timezone label
            tz_offset: Offset in seconds from UTC
        """
        # Build update query dynamically based on provided fields
        updates = []
        params = []
        
        if username is not None:
            updates.append("username = ?")
            params.append(username)
        if real_name is not None:
            updates.append("real_name = ?")
            params.append(real_name)
        
        if email is not None:
            updates.append("email = ?")
            params.append(email)
        if timezone is not None:
            updates.append("timezone = ?")
            params.append(timezone)
        if tz_label is not None:
            updates.append("tz_label = ?")
            params.append(tz_label)
        if tz_offset is not None:
            updates.append("tz_offset = ?")
            params.append(tz_offset)
        
        if updates:
            updates.append("last_seen = CURRENT_TIMESTAMP")
            params.append(user_id)
            
            query = f"""
                UPDATE users 
                SET {', '.join(updates)}
                WHERE user_id = ?
            """
            self.conn.execute(query, params)
            
            logger.debug(f"Updated user info for {user_id}: username={username}, real_name={real_name}, tz={timezone}")
    
    def save_user_timezone(self, user_id: str, timezone: str, 
                          tz_label: Optional[str] = None, tz_offset: Optional[int] = None):
        """
        Save user timezone information (kept for compatibility).
        
        Args:
            user_id: User identifier
            timezone: Timezone string
            tz_label: Timezone label
            tz_offset: Offset in seconds from UTC
        """
        self.save_user_info(user_id, timezone=timezone, tz_label=tz_label, tz_offset=tz_offset)
    
    def get_user_timezone(self, user_id: str) -> Optional[Tuple[str, str, int]]:
        """
        Get user timezone information.
        
        Args:
            user_id: User identifier
            
        Returns:
            Tuple of (timezone, tz_label, tz_offset) or None
        """
        cursor = self.conn.execute(
            "SELECT timezone, tz_label, tz_offset FROM users WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        
        if row:
            return (row["timezone"], row["tz_label"], row["tz_offset"])
        
        return None
    
    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive user information including email.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dict with user info or None
        """
        cursor = self.conn.execute(
            "SELECT username, real_name, email, timezone, tz_label, tz_offset FROM users WHERE user_id = ?",
            (user_id,)
        )
        row = cursor.fetchone()
        
        if row:
            return {
                'username': row["username"],
                'real_name': row["real_name"],
                'email': row["email"],
                'timezone': row["timezone"],
                'tz_label': row["tz_label"],
                'tz_offset': row["tz_offset"]
            }
        
        return None
    
    # Config hierarchy resolution
    
    def get_effective_config(self, thread_id: str, user_id: str) -> Dict:
        """
        Get effective configuration merging BotConfig -> User -> Thread.
        
        Args:
            thread_id: Thread identifier
            user_id: User identifier
            
        Returns:
            Merged configuration dictionary
        """
        from config import BotConfig
        bot_config = BotConfig()
        
        # Start with bot defaults
        effective = {
            "model": bot_config.gpt_model,
            "temperature": bot_config.default_temperature,
            "top_p": bot_config.default_top_p,
            "reasoning_effort": bot_config.default_reasoning_effort,
            "verbosity": bot_config.default_verbosity,
            "image_model": bot_config.image_model,
            "image_size": bot_config.default_image_size,
            "image_quality": bot_config.default_image_quality,
            "image_background": bot_config.default_image_background,
            "input_fidelity": bot_config.default_input_fidelity,
            "detail_level": bot_config.default_detail_level
        }
        
        # Apply user config
        user_config = self.get_user_config(user_id)
        if user_config:
            effective.update(user_config)
        
        # Apply thread config
        thread_config = self.get_thread_config(thread_id)
        if thread_config:
            effective.update(thread_config)
        
        return effective
    
    # Maintenance operations
    
    # =============================
    # MODAL SESSION MANAGEMENT
    # =============================

    def create_modal_session(self, session_id: str, user_id: str, state: Dict, modal_type: str = 'settings') -> bool:
        """
        Create a new modal session.

        Args:
            session_id: Unique session identifier (UUID)
            user_id: User ID who owns this session
            state: Initial state dictionary
            modal_type: Type of modal (default 'settings')

        Returns:
            True if created successfully
        """
        try:
            self.conn.execute("""
                INSERT INTO modal_sessions (session_id, user_id, modal_type, state)
                VALUES (?, ?, ?, ?)
            """, (session_id, user_id, modal_type, json.dumps(state)))
            self.log_debug(f"Created modal session {session_id} for user {user_id}")
            return True
        except Exception as e:
            self.log_error(f"Failed to create modal session: {e}")
            return False

    def get_modal_session(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve modal session state.

        Args:
            session_id: Session identifier

        Returns:
            State dictionary or None if not found
        """
        try:
            cursor = self.conn.execute("""
                SELECT state FROM modal_sessions
                WHERE session_id = ?
            """, (session_id,))

            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
            return None
        except Exception as e:
            self.log_error(f"Failed to get modal session: {e}")
            return None

    def update_modal_session(self, session_id: str, state: Dict) -> bool:
        """
        Update modal session state.

        Args:
            session_id: Session identifier
            state: Updated state dictionary

        Returns:
            True if updated successfully
        """
        try:
            cursor = self.conn.execute("""
                UPDATE modal_sessions
                SET state = ?, updated_at = strftime('%s', 'now')
                WHERE session_id = ?
            """, (json.dumps(state), session_id))

            if cursor.rowcount > 0:
                self.log_debug(f"Updated modal session {session_id}")
                return True
            return False
        except Exception as e:
            self.log_error(f"Failed to update modal session: {e}")
            return False

    def delete_modal_session(self, session_id: str) -> bool:
        """
        Delete a modal session.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted successfully
        """
        try:
            cursor = self.conn.execute("""
                DELETE FROM modal_sessions
                WHERE session_id = ?
            """, (session_id,))

            if cursor.rowcount > 0:
                self.log_debug(f"Deleted modal session {session_id}")
                return True
            return False
        except Exception as e:
            self.log_error(f"Failed to delete modal session: {e}")
            return False

    def cleanup_old_modal_sessions(self, hours: int = 24):
        """
        Clean up modal sessions older than specified hours.

        Args:
            hours: Number of hours to retain sessions (default 24)
        """
        try:
            cutoff = int((datetime.now() - timedelta(hours=hours)).timestamp())

            cursor = self.conn.execute("""
                DELETE FROM modal_sessions
                WHERE created_at < ?
            """, (cutoff,))

            if cursor.rowcount > 0:
                self.log_info(f"Cleaned up {cursor.rowcount} modal sessions older than {hours} hours")
        except Exception as e:
            self.log_error(f"Failed to cleanup modal sessions: {e}")

    # Async versions for modal sessions
    async def create_modal_session_async(self, session_id: str, user_id: str, state: Dict, modal_type: str = 'settings') -> bool:
        """Async version of create_modal_session."""
        async with self._async_db_semaphore:
            async with aiosqlite.connect(self.db_path) as db:
                try:
                    await db.execute("""
                        INSERT INTO modal_sessions (session_id, user_id, modal_type, state)
                        VALUES (?, ?, ?, ?)
                    """, (session_id, user_id, modal_type, json.dumps(state)))
                    await db.commit()
                    self.log_debug(f"Created modal session {session_id} for user {user_id} (async)")
                    return True
                except Exception as e:
                    self.log_error(f"Failed to create modal session (async): {e}")
                    return False

    async def get_modal_session_async(self, session_id: str) -> Optional[Dict]:
        """Async version of get_modal_session."""
        async with self._async_db_semaphore:
            async with aiosqlite.connect(self.db_path) as db:
                try:
                    async with db.execute("""
                        SELECT state FROM modal_sessions
                        WHERE session_id = ?
                    """, (session_id,)) as cursor:
                        row = await cursor.fetchone()
                        if row:
                            return json.loads(row[0])
                    return None
                except Exception as e:
                    self.log_error(f"Failed to get modal session (async): {e}")
                    return None

    async def update_modal_session_async(self, session_id: str, state: Dict) -> bool:
        """Async version of update_modal_session."""
        async with self._async_db_semaphore:
            async with aiosqlite.connect(self.db_path) as db:
                try:
                    await db.execute("""
                        UPDATE modal_sessions
                        SET state = ?, updated_at = strftime('%s', 'now')
                        WHERE session_id = ?
                    """, (json.dumps(state), session_id))
                    await db.commit()
                    self.log_debug(f"Updated modal session {session_id} (async)")
                    return True
                except Exception as e:
                    self.log_error(f"Failed to update modal session (async): {e}")
                    return False

    async def delete_modal_session_async(self, session_id: str) -> bool:
        """Async version of delete_modal_session."""
        async with self._async_db_semaphore:
            async with aiosqlite.connect(self.db_path) as db:
                try:
                    await db.execute("""
                        DELETE FROM modal_sessions
                        WHERE session_id = ?
                    """, (session_id,))
                    await db.commit()
                    self.log_debug(f"Deleted modal session {session_id} (async)")
                    return True
                except Exception as e:
                    self.log_error(f"Failed to delete modal session (async): {e}")
                    return False

    def backup_database(self, tag: Optional[str] = None):
        """Create timestamped backup of database.

        Args:
            tag: Optional label inserted before the timestamp (e.g. a migration name),
                 producing {platform}_{tag}_{timestamp}.db. Kept before the timestamp so
                 cleanup_old_backups' date parsing (last two underscore parts) still works.
        """
        # Checkpoint WAL file before backup
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = f"{tag}_" if tag else ""
        backup_path = f"{self.db_dir}/backups/{self.platform}_{label}{timestamp}.db"
        
        # Use SQLite backup API
        backup_conn = sqlite3.connect(backup_path)
        with backup_conn:
            self.conn.backup(backup_conn)
        backup_conn.close()
        
        logger.info(f"Created backup: {backup_path}")
        
        # Clean up old backups
        self.cleanup_old_backups()
    
    def cleanup_old_backups(self):
        """Remove backups older than 7 days."""
        cutoff = datetime.now() - timedelta(days=7)

        for filename in os.listdir(f"{self.db_dir}/backups"):
            if filename.startswith(f"{self.platform}_") and filename.endswith(".db"):
                try:
                    # Parse timestamp from filename
                    parts = filename.replace(".db", "").split("_")
                    if len(parts) >= 3:
                        date_str = parts[-2] + parts[-1]
                        file_date = datetime.strptime(date_str, "%Y%m%d%H%M%S")
                        if file_date < cutoff:
                            os.remove(f"{self.db_dir}/backups/{filename}")
                            logger.info(f"Removed old backup: {filename}")
                            
                except Exception as e:
                    logger.warning(f"Error processing backup file {filename}: {e}")
    
    def cleanup_old_threads(self):
        """Remove threads older than 3 months."""
        cutoff = datetime.now() - timedelta(days=90)
        
        cursor = self.conn.execute("""
            DELETE FROM threads 
            WHERE last_activity < ?
        """, (cutoff,))
        
        if cursor.rowcount > 0:
            logger.info(f"Cleaned up {cursor.rowcount} threads older than 3 months")
    
    # =============================
    # ASYNC VERSIONS OF CORE METHODS
    # =============================

    async def _get_async_connection(self):
        """Get an async database connection with semaphore control."""
        await self._async_db_semaphore.acquire()
        try:
            conn = await aiosqlite.connect(
                self.db_path,
                isolation_level=None  # Autocommit mode
            )
            conn.row_factory = aiosqlite.Row  # Enable column access by name

            # Enable WAL mode for better concurrency
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute("PRAGMA busy_timeout=5000")  # 5 second timeout

            return conn
        finally:
            self._async_db_semaphore.release()

    async def get_thread_summary_async(self, thread_id: str) -> Optional[Dict]:
        """Async version of get_thread_summary."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            async with db.execute(
                "SELECT * FROM thread_summaries WHERE thread_id = ?", (thread_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
                summary = dict(row)
                summary["refs"] = json.loads(summary["refs_json"]) if summary.get("refs_json") else []
                return summary

    async def save_thread_summary_async(self, thread_id: str, summary_text: str, boundary_ts: str,
                                        refs: Optional[List[Dict]] = None):
        """Async version of save_thread_summary (upsert, rolling)."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            await db.execute("""
                INSERT INTO thread_summaries (thread_id, summary_text, boundary_ts, refs_json, updated_ts)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(thread_id) DO UPDATE SET
                    summary_text = excluded.summary_text,
                    boundary_ts = excluded.boundary_ts,
                    refs_json = excluded.refs_json,
                    updated_ts = CURRENT_TIMESTAMP
            """, (thread_id, summary_text, boundary_ts,
                  json.dumps(refs) if refs else None))
            await db.commit()
        self.log_info(f"DB: Saved thread summary for {thread_id} (boundary_ts={boundary_ts}, async)")

    async def update_thread_activity_async(self, thread_id: str):
        """
        Async version of update_thread_activity.

        Args:
            thread_id: Thread identifier
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            await db.execute("""
                UPDATE threads
                SET last_activity = CURRENT_TIMESTAMP
                WHERE thread_id = ?
            """, (thread_id,))

            await db.commit()

    async def save_image_metadata_async(self, thread_id: str, url: str, image_type: str,
                                       prompt: Optional[str] = None, analysis: Optional[str] = None,
                                       original_analysis: Optional[str] = None, metadata: Optional[Dict] = None,
                                       message_ts: Optional[str] = None):
        """
        Async version of save_image_metadata (NO base64 data).

        Args:
            thread_id: Thread identifier
            url: Image URL
            image_type: Type of image (uploaded/generated/edited)
            prompt: Full generation/edit prompt
            analysis: Full vision analysis
            original_analysis: For edited images, the pre-edit analysis
            metadata: Additional metadata
            message_ts: Message timestamp to link image to specific message
        """
        self.log_debug(f"DB: Async saving image - thread={thread_id}, url={url[:100]}, "
                      f"type={image_type}, has_analysis={bool(analysis)}, "
                      f"analysis_len={len(analysis) if analysis else 0}, "
                      f"prompt_len={len(prompt) if prompt else 0}")

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                await db.execute("PRAGMA journal_mode=WAL")

                # Merge-preserving upsert (F1): see save_image_metadata. A later empty
                # write (rebuild with empty caption, ledger upsert) must not erase the
                # prompt/analysis/type/generation_id an earlier write recorded.
                await db.execute("""
                    INSERT INTO images
                    (thread_id, url, image_type, prompt, analysis, original_analysis, metadata_json, message_ts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(url) DO UPDATE SET
                        thread_id = excluded.thread_id,
                        image_type = COALESCE(NULLIF(images.image_type, ''), excluded.image_type),
                        prompt = COALESCE(NULLIF(images.prompt, ''), excluded.prompt),
                        analysis = COALESCE(NULLIF(images.analysis, ''), excluded.analysis),
                        original_analysis = COALESCE(NULLIF(images.original_analysis, ''), excluded.original_analysis),
                        metadata_json = COALESCE(NULLIF(images.metadata_json, ''), excluded.metadata_json),
                        message_ts = COALESCE(excluded.message_ts, images.message_ts)
                """, (thread_id, url, image_type, prompt, analysis, original_analysis,
                      json.dumps(metadata) if metadata else None, message_ts))

                await db.commit()

                self.log_info(f"DB: Successfully saved image metadata for {url[:50]}... in thread {thread_id}")

        except Exception as e:
            self.log_error(f"DB: Failed to save image metadata async - {e}", exc_info=True)
            raise

    async def get_thread_config_async(self, thread_id: str) -> Optional[Dict]:
        """
        Async version of get_thread_config.

        Args:
            thread_id: Thread identifier

        Returns:
            Configuration dictionary or None
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            async with db.execute(
                "SELECT config_json FROM threads WHERE thread_id = ?",
                (thread_id,)
            ) as cursor:
                row = await cursor.fetchone()

                if row and row["config_json"]:
                    return json.loads(row["config_json"])

                return None

    async def save_thread_config_async(self, thread_id: str, config: Dict):
        """
        Async version of save_thread_config.

        Args:
            thread_id: Thread identifier
            config: Configuration dictionary
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            await db.execute("""
                UPDATE threads
                SET config_json = ?, updated_at = CURRENT_TIMESTAMP
                WHERE thread_id = ?
            """, (json.dumps(config), thread_id))

            await db.commit()
            logger.debug(f"Saved config for thread {thread_id} (async)")

    async def get_channel_settings_async(self, channel_id: str) -> Optional[Dict]:
        """Async version of get_channel_settings."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute(
                "SELECT response_mode, directives, reply_in_channel, participation_level, "
                "snoozed_until, model, reasoning_effort, verbosity, updated_ts, updated_by "
                "FROM channel_settings WHERE channel_id = ?",
                (channel_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
                return {
                    "response_mode": row["response_mode"],
                    "directives": row["directives"],
                    "reply_in_channel": bool(row["reply_in_channel"]),
                    "participation_level": row["participation_level"],
                    "snoozed_until": row["snoozed_until"],
                    "model": row["model"],
                    "reasoning_effort": row["reasoning_effort"],
                    "verbosity": row["verbosity"],
                    "updated_ts": row["updated_ts"],
                    "updated_by": row["updated_by"],
                }

    async def set_channel_settings_async(self, channel_id: str, response_mode=_UNSET,
                                         directives=_UNSET, reply_in_channel=_UNSET,
                                         participation_level=_UNSET, snoozed_until=_UNSET,
                                         model=_UNSET, reasoning_effort=_UNSET, verbosity=_UNSET,
                                         updated_by: Optional[str] = None):
        """Async version of set_channel_settings (Phase F adds participation_level/snoozed_until).

        Omitted fields are preserved; an explicit None CLEARS the column to NULL (so the modal's
        "inherit" selection resolves to the global default at read time; snoozed_until=None
        clears an active snooze).
        """
        existing = await self.get_channel_settings_async(channel_id) or {}
        new_mode = existing.get("response_mode", "tag_only") if response_mode is _UNSET else response_mode
        new_dir = existing.get("directives") if directives is _UNSET else directives
        new_ric = existing.get("reply_in_channel", False) if reply_in_channel is _UNSET else reply_in_channel
        new_level = existing.get("participation_level") if participation_level is _UNSET else participation_level
        new_snooze = existing.get("snoozed_until") if snoozed_until is _UNSET else snoozed_until
        new_model = existing.get("model") if model is _UNSET else model
        new_effort = existing.get("reasoning_effort") if reasoning_effort is _UNSET else reasoning_effort
        new_verb = existing.get("verbosity") if verbosity is _UNSET else verbosity
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("""
                INSERT INTO channel_settings (channel_id, response_mode, directives, reply_in_channel,
                                              participation_level, snoozed_until, model, reasoning_effort,
                                              verbosity, updated_ts, updated_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
                ON CONFLICT(channel_id) DO UPDATE SET
                    response_mode=excluded.response_mode,
                    directives=excluded.directives,
                    reply_in_channel=excluded.reply_in_channel,
                    participation_level=excluded.participation_level,
                    snoozed_until=excluded.snoozed_until,
                    model=excluded.model,
                    reasoning_effort=excluded.reasoning_effort,
                    verbosity=excluded.verbosity,
                    updated_ts=CURRENT_TIMESTAMP,
                    updated_by=excluded.updated_by
            """, (channel_id, new_mode, new_dir, 1 if new_ric else 0, new_level, new_snooze,
                  new_model, new_effort, new_verb, updated_by))
            await db.commit()
            logger.debug(f"Saved channel_settings for {channel_id} (async): mode={new_mode}, level={new_level}")

    # --- Per-channel memory (Phase 9), async variants ---
    async def get_channel_memory_async(self, channel_id: str) -> List[Dict]:
        """Async version of get_channel_memory (channel-scope for this channel + shared workspace)."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute(
                "SELECT id, channel_id, scope, content, author, created_ts, updated_ts "
                "FROM channel_memory WHERE (scope = 'channel' AND channel_id = ?) OR scope = 'workspace' "
                "ORDER BY updated_ts ASC",
                (channel_id,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]

    async def add_channel_memory_async(self, channel_id: str, content: str, scope: str = "channel",
                                       author: Optional[str] = None) -> int:
        """Async version of add_channel_memory; returns the new id."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            cursor = await db.execute(
                "INSERT INTO channel_memory (channel_id, scope, content, author) VALUES (?, ?, ?, ?)",
                (channel_id, scope, content, author)
            )
            await db.commit()
            return cursor.lastrowid

    async def update_channel_memory_async(self, memory_id: int, content: str):
        """Async version of update_channel_memory."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(
                "UPDATE channel_memory SET content = ?, updated_ts = CURRENT_TIMESTAMP WHERE id = ?",
                (content, memory_id)
            )
            await db.commit()

    async def delete_channel_memory_async(self, memory_id: int):
        """Async version of delete_channel_memory."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("DELETE FROM channel_memory WHERE id = ?", (memory_id,))
            await db.commit()

    # --- Response feedback (Phase H) ---
    async def record_response_feedback_async(self, channel_id: str, thread_ts: Optional[str],
                                             message_ts: str, user_id: str, signal: int,
                                             source: str) -> None:
        """Async version of record_response_feedback (upsert per message/user/source)."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("""
                INSERT INTO response_feedback (channel_id, thread_ts, message_ts, user_id, signal, source)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_ts, user_id, source) DO UPDATE SET
                    signal=excluded.signal,
                    updated_ts=CURRENT_TIMESTAMP
            """, (channel_id, thread_ts, message_ts, user_id, signal, source))
            await db.commit()

    async def delete_response_feedback_async(self, message_ts: str, user_id: str, source: str) -> None:
        """Async version of delete_response_feedback."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(
                "DELETE FROM response_feedback WHERE message_ts = ? AND user_id = ? AND source = ?",
                (message_ts, user_id, source)
            )
            await db.commit()

    async def get_channel_feedback_ratio_async(self, channel_id: str, days: int = 30):
        """Async version of get_channel_feedback_ratio."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute(
                "SELECT "
                "  SUM(CASE WHEN signal > 0 THEN 1 ELSE 0 END) AS positive, "
                "  SUM(CASE WHEN signal < 0 THEN 1 ELSE 0 END) AS negative "
                "FROM response_feedback "
                "WHERE channel_id = ? AND created_ts >= datetime('now', ?)",
                (channel_id, f"-{int(days)} days")
            ) as cursor:
                row = await cursor.fetchone()
        positive = row["positive"] or 0
        negative = row["negative"] or 0
        total = positive + negative
        return positive, negative, (positive / total if total else None)

    async def get_or_create_user_async(self, user_id: str, username: Optional[str] = None) -> Dict:
        """
        Async version of get_or_create_user.

        Args:
            user_id: User identifier
            username: Optional username

        Returns:
            User data dictionary
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            # Try to get existing user
            async with db.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,)
            ) as cursor:
                row = await cursor.fetchone()

            if row:
                # Update last seen
                await db.execute("""
                    UPDATE users SET last_seen = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                """, (user_id,))
                await db.commit()
                return dict(row)

            # Create new user with defaults from config
            from config import BotConfig
            config = BotConfig()

            await db.execute("""
                INSERT INTO users (user_id, username, created_at, last_seen)
                VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """, (user_id, username))

            await db.commit()

            # Return the created user
            async with db.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else {}

    async def get_user_info_async(self, user_id: str) -> Optional[Dict]:
        """
        Async version of get_user_info.

        Args:
            user_id: User identifier

        Returns:
            User info dictionary or None
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            async with db.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def get_user_preferences_async(self, user_id: str) -> Optional[Dict]:
        """
        Async version of get_user_preferences.

        Args:
            user_id: User identifier

        Returns:
            User preferences dictionary or None
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            async with db.execute(
                "SELECT * FROM user_preferences WHERE slack_user_id = ?",
                (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    prefs = dict(row)
                    # Convert SQLite boolean (0/1) to Python boolean
                    prefs['enable_web_search'] = bool(prefs.get('enable_web_search', 1))
                    prefs['enable_streaming'] = bool(prefs.get('enable_streaming', 1))
                    prefs['enable_mcp'] = bool(prefs.get('enable_mcp', 1))
                    prefs['settings_completed'] = bool(prefs.get('settings_completed', 0))
                    return prefs
                return None

    async def create_default_user_preferences_async(self, user_id: str, email: str) -> Dict:
        """
        Async version of create_default_user_preferences.

        Args:
            user_id: User identifier
            email: User email

        Returns:
            Created preferences dictionary
        """
        from config import BotConfig
        config = BotConfig()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            # Create default preferences
            await db.execute("""
                INSERT OR REPLACE INTO user_preferences (
                    slack_user_id, slack_email,
                    model, temperature, top_p,
                    enable_web_search, enable_streaming,
                    reasoning_effort, verbosity,
                    image_model, image_size, image_quality, image_background,
                    input_fidelity, vision_detail
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, email,
                config.gpt_model, config.default_temperature, config.default_top_p,
                1 if config.enable_web_search else 0,
                1 if config.enable_streaming else 0,
                config.default_reasoning_effort, config.default_verbosity,
                config.image_model,
                config.default_image_size, config.default_image_quality, config.default_image_background,
                config.default_input_fidelity, config.default_detail_level
            ))

            await db.commit()

            # Return the created preferences
            async with db.execute(
                "SELECT * FROM user_preferences WHERE slack_user_id = ?",
                (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    prefs = dict(row)
                    # Convert SQLite boolean (0/1) to Python boolean
                    prefs['enable_web_search'] = bool(prefs.get('enable_web_search', 1))
                    prefs['enable_streaming'] = bool(prefs.get('enable_streaming', 1))
                    prefs['enable_mcp'] = bool(prefs.get('enable_mcp', 1))
                    prefs['settings_completed'] = bool(prefs.get('settings_completed', 0))
                    return prefs
                return {}

    async def find_thread_images_async(self, thread_id: str, image_type: Optional[str] = None) -> List[Dict]:
        """Async version of find_thread_images."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            if image_type:
                query = ("SELECT * FROM images WHERE thread_id = ? AND image_type = ? "
                         "ORDER BY created_at ASC")
                params = (thread_id, image_type)
            else:
                query = "SELECT * FROM images WHERE thread_id = ? ORDER BY created_at ASC"
                params = (thread_id,)

            async with db.execute(query, params) as cursor:
                images = []
                async for row in cursor:
                    img = dict(row)
                    if img.get("metadata_json"):
                        img["metadata"] = json.loads(img["metadata_json"])
                        del img["metadata_json"]
                    images.append(img)
                return images

    async def get_images_by_message_async(self, thread_id: str, message_ts: str) -> List[Dict]:
        """Async version of get_images_by_message."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            async with db.execute(
                "SELECT * FROM images WHERE thread_id = ? AND message_ts = ? ORDER BY created_at ASC",
                (thread_id, message_ts)
            ) as cursor:
                images = []
                async for row in cursor:
                    img = dict(row)
                    if img.get("metadata_json"):
                        img["metadata"] = json.loads(img["metadata_json"])
                        del img["metadata_json"]
                    images.append(img)
                return images

    async def get_thread_documents_async(self, thread_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Async version of get_thread_documents."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            query = "SELECT * FROM documents WHERE thread_id = ? ORDER BY created_at ASC"
            if limit:
                query += f" LIMIT {int(limit)}"

            async with db.execute(query, (thread_id,)) as cursor:
                documents = []
                async for row in cursor:
                    doc = dict(row)
                    if doc.get("page_structure"):
                        doc["page_structure"] = json.loads(doc["page_structure"])
                    if doc.get("metadata_json"):
                        doc["metadata"] = json.loads(doc["metadata_json"])
                        del doc["metadata_json"]
                    documents.append(doc)
                return documents

    async def get_document_by_filename_async(self, thread_id: str, filename: str) -> Optional[Dict]:
        """Async version of get_document_by_filename (newest matching row)."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute(
                """SELECT * FROM documents
                   WHERE thread_id = ? AND filename = ?
                   ORDER BY created_at DESC LIMIT 1""",
                (thread_id, filename),
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
                doc = dict(row)
                if doc.get("page_structure"):
                    doc["page_structure"] = json.loads(doc["page_structure"])
                if doc.get("metadata_json"):
                    doc["metadata"] = json.loads(doc["metadata_json"])
                    del doc["metadata_json"]
                return doc

    async def get_or_create_thread_async(self, thread_id: str, channel_id: str,
                                         user_id: Optional[str] = None) -> Dict:
        """Async wrapper for get_or_create_thread.

        The sync method is multi-step (lookup, activity touch, user-config copy,
        insert, recursive re-read); duplicating it in aiosqlite risks divergence,
        so it runs unchanged on a worker thread (the shared connection is created
        with check_same_thread=False and WAL handles concurrency).
        """
        return await asyncio.to_thread(self.get_or_create_thread, thread_id, channel_id, user_id)

    async def cleanup_old_modal_sessions_async(self, hours: int = 24):
        """Async version of cleanup_old_modal_sessions."""
        try:
            cutoff = int((datetime.now() - timedelta(hours=hours)).timestamp())
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA journal_mode=WAL")
                cursor = await db.execute(
                    "DELETE FROM modal_sessions WHERE created_at < ?", (cutoff,)
                )
                await db.commit()
                if cursor.rowcount > 0:
                    self.log_info(f"Cleaned up {cursor.rowcount} modal sessions older than {hours} hours")
        except Exception as e:
            self.log_error(f"Failed to cleanup modal sessions: {e}")

    async def get_user_timezone_async(self, user_id: str) -> Optional[str]:
        """
        Async version of get_user_timezone.

        Args:
            user_id: User identifier

        Returns:
            Timezone string or None
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            async with db.execute(
                "SELECT timezone FROM users WHERE user_id = ?",
                (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return row["timezone"] if row and row["timezone"] else None

    async def save_user_info_async(self, user_id: str, username: str, real_name: str, email: str,
                                   timezone: str = None, tz_label: str = None, tz_offset: int = None):
        """
        Async version of save_user_info.

        Args:
            user_id: User identifier
            username: Username
            real_name: Real name
            email: Email address
            timezone: Optional timezone
            tz_label: Optional timezone label
            tz_offset: Optional timezone offset
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")

            await db.execute("""
                UPDATE users
                SET username = ?, real_name = ?, email = ?, timezone = ?, last_seen = CURRENT_TIMESTAMP
                WHERE user_id = ?
            """, (username, real_name, email, timezone, user_id))

            await db.commit()

    async def update_user_preferences_async(self, user_id: str, preferences: Dict) -> bool:
        """
        Async version of update_user_preferences.

        Args:
            user_id: User identifier
            preferences: Preferences to update

        Returns:
            True if update successful
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")

            # Build dynamic update query
            update_fields = []
            values = []

            # Handle regular fields
            for field in ['model', 'reasoning_effort', 'verbosity', 'temperature',
                         'top_p', 'image_model', 'image_size', 'image_quality', 'image_background',
                         'input_fidelity', 'vision_detail',
                         'slack_email', 'settings_completed', 'custom_instructions']:
                if field in preferences:
                    update_fields.append(f"{field} = ?")
                    values.append(preferences[field])

            # Handle boolean fields - convert to integers for SQLite
            for field in ['enable_web_search', 'enable_streaming', 'enable_mcp']:
                if field in preferences:
                    update_fields.append(f"{field} = ?")
                    values.append(1 if preferences[field] else 0)

            if not update_fields:
                return False

            values.append(user_id)
            query = f"""
                UPDATE user_preferences
                SET {', '.join(update_fields)}, updated_at = CURRENT_TIMESTAMP
                WHERE slack_user_id = ?
            """

            await db.execute(query, values)
            await db.commit()
            return True

    async def get_or_create_thread_async(self, thread_id: str, channel_id: str, user_id: Optional[str] = None) -> Dict:
        """
        Async version of get_or_create_thread.

        Args:
            thread_id: Thread identifier
            channel_id: Channel identifier
            user_id: Optional user identifier

        Returns:
            Thread data dictionary
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            # Try to get existing thread
            async with db.execute(
                "SELECT * FROM threads WHERE thread_id = ?",
                (thread_id,)
            ) as cursor:
                row = await cursor.fetchone()

            if row:
                return dict(row)

            # Create new thread
            thread_ts = thread_id.split(":", 1)[1] if ":" in thread_id else thread_id

            # Get user config if user_id provided
            config = {}
            if user_id:
                user_prefs = await self.get_user_preferences_async(user_id)
                if user_prefs:
                    # Extract relevant config from user preferences
                    config = {
                        'model': user_prefs.get('model'),
                        'reasoning_effort': user_prefs.get('reasoning_effort'),
                        'verbosity': user_prefs.get('verbosity'),
                        'temperature': user_prefs.get('temperature'),
                        'top_p': user_prefs.get('top_p'),
                        'enable_web_search': user_prefs.get('enable_web_search'),
                        'enable_streaming': user_prefs.get('enable_streaming'),
                        'enable_mcp': user_prefs.get('enable_mcp')
                    }
                    # Remove None values
                    config = {k: v for k, v in config.items() if v is not None}

            await db.execute("""
                INSERT INTO threads (thread_id, channel_id, thread_ts, config_json)
                VALUES (?, ?, ?, ?)
            """, (thread_id, channel_id, thread_ts, json.dumps(config) if config else None))

            await db.commit()

            # Return the created thread
            async with db.execute(
                "SELECT * FROM threads WHERE thread_id = ?",
                (thread_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else {}

    async def get_thread_config_async(self, thread_id: str) -> Optional[Dict]:
        """
        Async version of get_thread_config.

        Args:
            thread_id: Thread identifier

        Returns:
            Thread config dictionary or None
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            async with db.execute(
                "SELECT config_json FROM threads WHERE thread_id = ?",
                (thread_id,)
            ) as cursor:
                row = await cursor.fetchone()

            if row and row["config_json"]:
                return json.loads(row["config_json"])

            return None

    # MCP tool caching methods
    def save_mcp_tool(self, server_label: str, tool_name: str, description: Optional[str] = None, input_schema: Optional[str] = None):
        """
        Save or update an MCP tool in the cache.

        Args:
            server_label: MCP server label
            tool_name: Tool name
            description: Tool description
            input_schema: Tool input schema (JSON string)
        """
        try:
            self.conn.execute("""
                INSERT INTO mcp_tools (server_label, tool_name, description, input_schema)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(server_label, tool_name) DO UPDATE SET
                    description = excluded.description,
                    input_schema = excluded.input_schema,
                    last_verified = CURRENT_TIMESTAMP
            """, (server_label, tool_name, description, input_schema))
            self.conn.commit()
            self.log_debug(f"DB: Cached MCP tool {server_label}:{tool_name}")
        except Exception as e:
            self.log_error(f"DB: Error caching MCP tool: {e}", exc_info=True)

    def get_mcp_tools(self, server_label: Optional[str] = None) -> List[Dict]:
        """
        Get cached MCP tools, optionally filtered by server.

        Args:
            server_label: Optional server label to filter by

        Returns:
            List of tool dictionaries
        """
        try:
            if server_label:
                cursor = self.conn.execute("""
                    SELECT server_label, tool_name, description, input_schema,
                           discovered_at, last_verified
                    FROM mcp_tools
                    WHERE server_label = ?
                    ORDER BY server_label, tool_name
                """, (server_label,))
            else:
                cursor = self.conn.execute("""
                    SELECT server_label, tool_name, description, input_schema,
                           discovered_at, last_verified
                    FROM mcp_tools
                    ORDER BY server_label, tool_name
                """)

            tools = []
            for row in cursor.fetchall():
                tools.append({
                    'server_label': row[0],
                    'tool_name': row[1],
                    'description': row[2],
                    'input_schema': row[3],
                    'discovered_at': row[4],
                    'last_verified': row[5]
                })
            return tools
        except Exception as e:
            self.log_error(f"DB: Error retrieving MCP tools: {e}", exc_info=True)
            return []

    def clear_mcp_tools(self, server_label: Optional[str] = None):
        """
        Clear cached MCP tools, optionally for a specific server.

        Args:
            server_label: Optional server label to clear tools for (clears all if not provided)
        """
        try:
            if server_label:
                self.conn.execute("DELETE FROM mcp_tools WHERE server_label = ?", (server_label,))
                self.log_info(f"DB: Cleared MCP tools for server {server_label}")
            else:
                self.conn.execute("DELETE FROM mcp_tools")
                self.log_info("DB: Cleared all MCP tools")
            self.conn.commit()
        except Exception as e:
            self.log_error(f"DB: Error clearing MCP tools: {e}", exc_info=True)

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info(f"Database connection closed for {self.platform}")