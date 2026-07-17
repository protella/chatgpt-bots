"""
SQLite Database Manager for ChatGPT Bots
Provides persistent storage for threads, messages, images, documents, and user preferences
"""

import sqlite3
import aiosqlite
import hashlib
import json
import os
import re
from contextlib import contextmanager
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

# F51c — cap on late-artifact addenda folded onto ONE thread's compaction summary head, so a
# pathological channel can't bloat the head unboundedly. Each note is already length-capped at
# render time; this caps the COUNT. Single source of truth for both the completion-time
# (ambient service) and compaction-time (thread management) capture paths.
_MAX_SUMMARY_ADDENDA_PER_THREAD = 20


# Shared hash/normalize contract for channel-memory reconciliation. The settings modal builder,
# the submit handler, and reconcile_channel_memory_from_textarea_async ALL route content through
# these two functions so a content hash computed at modal-open matches one recomputed at submit —
# that identity is how a seeded row is matched (keep), missed (delete), or changed (conflict).
def normalize_memory_line(text: str) -> str:
    """Collapse every whitespace run (spaces, tabs, newlines) to a single space and strip ends.

    Blank or whitespace-only input (and None) returns "". This is the single normalization all
    three call sites share, so a legacy multi-line fact and its single-line textarea rendering
    hash to the same value.
    """
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def memory_content_hash(text: str) -> str:
    """Stable short identity for a memory line: sha256 hexdigest of the normalized text, [:16].

    Paired with normalize_memory_line so text that is equal after normalization hashes equal.
    """
    return hashlib.sha256(normalize_memory_line(text).encode()).hexdigest()[:16]


def _decode_muted_threads(raw) -> List[str]:
    """Parse the channel_settings.muted_threads JSON column into a list of thread ts
    strings. Malformed/absent → empty list (fail open — a bad blob never silences)."""
    if not raw:
        return []
    try:
        val = json.loads(raw)
        return [str(t) for t in val] if isinstance(val, list) else []
    except (ValueError, TypeError):
        return []


def _encode_muted_threads(val) -> Optional[str]:
    """Serialize a list of thread ts strings for storage. None/empty → NULL."""
    if not val:
        return None
    try:
        return json.dumps([str(t) for t in val])
    except (ValueError, TypeError):
        return None


# channel_settings columns whose write is a real, attributed structural edit. Touching any of
# them bumps updated_ts/updated_by; a write that touches only the non-structural columns
# (snoozed_until, the deprecated muted_threads) leaves authorship untouched — so a background
# housekeeping write never looks like the last human to edit the channel's response settings.
_CHANNEL_SETTINGS_STRUCTURAL = (
    "response_mode", "directives", "reply_in_channel",
    "participation_level", "model", "reasoning_effort", "verbosity",
    "ambient_memory",
)


def _build_channel_settings_write(channel_id, response_mode=_UNSET, directives=_UNSET,
                                  reply_in_channel=_UNSET, participation_level=_UNSET,
                                  snoozed_until=_UNSET, muted_threads=_UNSET,
                                  model=_UNSET, reasoning_effort=_UNSET, verbosity=_UNSET,
                                  ambient_memory=_UNSET, updated_by=None):
    """Build the atomic upsert for a partial channel_settings write.

    Returns ``(sql, params)`` or ``None`` when no field was provided (caller no-ops).
    Pure — no I/O — so both the sync and async setters share one implementation and one behavior.

    Design (fixes the mute-clobber incident):
    - Only explicitly-provided columns are written. Untouched columns are NEVER rewritten, so a
      partial write cannot clobber another field (no read-modify-write of the whole row → no race
      with a concurrent modal save).
    - Inheritance-capable columns that carry a non-NULL table default (response_mode → 'tag_only',
      reply_in_channel → 0) are pinned to NULL on a FRESH insert unless explicitly provided, so a
      partial write never materializes a downgraded default over the live global config. Cleared
      inheritance fields store NULL (never a copied runtime default), so global-config changes keep
      being inherited.
    - updated_ts/updated_by bump ONLY when a structural field changed
      (see ``_CHANNEL_SETTINGS_STRUCTURAL``).
    - reply_in_channel: explicit None → NULL (inherit); True/False → 1/0.
    """
    provided: Dict[str, Any] = {}
    if response_mode is not _UNSET:
        provided["response_mode"] = response_mode
    if directives is not _UNSET:
        provided["directives"] = directives
    if reply_in_channel is not _UNSET:
        provided["reply_in_channel"] = (
            None if reply_in_channel is None else (1 if reply_in_channel else 0))
    if participation_level is not _UNSET:
        provided["participation_level"] = participation_level
    if snoozed_until is not _UNSET:
        provided["snoozed_until"] = snoozed_until
    if muted_threads is not _UNSET:
        # Deprecated inert JSON column — nothing reads it anymore (the per-thread mute mechanism
        # was removed). Kept only so an explicit write can still clear it to NULL.
        provided["muted_threads"] = _encode_muted_threads(muted_threads)
    if model is not _UNSET:
        provided["model"] = model
    if reasoning_effort is not _UNSET:
        provided["reasoning_effort"] = reasoning_effort
    if verbosity is not _UNSET:
        provided["verbosity"] = verbosity
    if ambient_memory is not _UNSET:
        # F51 opt-out: explicit None → NULL (inherit config.enable_ambient_memory); True/False → 1/0.
        provided["ambient_memory"] = (
            None if ambient_memory is None else (1 if ambient_memory else 0))

    if not provided:
        return None

    structural_provided = [c for c in provided if c in _CHANNEL_SETTINGS_STRUCTURAL]
    changed_structural = bool(structural_provided)
    # On the UPDATE (conflict) branch, "structural change" means a real VALUE change, not merely
    # "a structural field was supplied": writing the SAME value must preserve updated_ts/updated_by
    # so an idempotent structural write (a re-save of unchanged settings, a mute-path no-op) never
    # rewrites who last edited the channel. `IS NOT` is SQLite's null-safe inequality, so an
    # inherit(NULL)→NULL write also reads as unchanged.
    change_cond = " OR ".join(
        f"channel_settings.{c} IS NOT excluded.{c}" for c in structural_provided)

    insert_cols = ["channel_id"]
    params: List[Any] = [channel_id]
    update_assignments: List[str] = []

    # Pin the non-NULL-default inheritance columns to NULL on a fresh insert unless provided,
    # so a partial insert inherits from global config instead of freezing a downgraded default.
    for col in ("response_mode", "reply_in_channel"):
        insert_cols.append(col)
        params.append(provided.get(col))
        if col in provided:
            update_assignments.append(f"{col}=excluded.{col}")

    for col, val in provided.items():
        if col in ("response_mode", "reply_in_channel"):
            continue
        insert_cols.append(col)
        params.append(val)
        update_assignments.append(f"{col}=excluded.{col}")

    # Attribute authorship only on a structural change. On a fresh insert updated_ts still gets
    # its column default (a row must have a created stamp); the "don't bump" rule guards UPDATEs.
    # An anonymous structural write (updated_by=None) still stamps the change time but preserves
    # the prior author rather than erasing it.
    insert_cols.append("updated_by")
    params.append(updated_by if changed_structural else None)
    if changed_structural:
        # Bump the stamp/author ONLY when a provided structural column's value actually differs
        # from the stored row (see change_cond) — a same-value write leaves attribution intact.
        update_assignments.append(
            f"updated_ts=CASE WHEN ({change_cond}) THEN CURRENT_TIMESTAMP ELSE updated_ts END")
        if updated_by is not None:
            update_assignments.append(
                f"updated_by=CASE WHEN ({change_cond}) THEN excluded.updated_by ELSE updated_by END")

    placeholders = ", ".join(["?"] * len(insert_cols))
    cols_sql = ", ".join(insert_cols)
    if update_assignments:
        conflict = f"ON CONFLICT(channel_id) DO UPDATE SET {', '.join(update_assignments)}"
    else:
        # Only non-structural columns AND the row already exists → nothing changes there;
        # but a fresh insert still needs to land, so keep the INSERT and no-op the conflict.
        conflict = "ON CONFLICT(channel_id) DO NOTHING"
    sql = f"INSERT INTO channel_settings ({cols_sql}) VALUES ({placeholders}) {conflict}"
    return sql, params


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

        # F51c — late-artifact addenda to the compaction summary. An ambient artifact (a slow
        # link fetch, a deferred vision job) can complete AFTER its source message has already
        # been folded into a thread's compaction summary — or a message with a long-ready
        # artifact can be compacted later. Either way the derived note would vanish: it never
        # lived in thread_state.messages (injection is transient per API call), the summary was
        # written without it, and the compacted message no longer returns in the rebuilt tail.
        # These rows carry that late/folded derivation forward — the rebuild concatenates them
        # onto the summary head. Bounded per thread (_MAX_SUMMARY_ADDENDA_PER_THREAD); UNIQUE
        # per (thread, source, kind, ref) so the completion path and the compaction path can't
        # double-record the same note. Deterministic order (source_ts, id) for cache stability.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS thread_summary_addenda (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                source_ts TEXT NOT NULL,
                kind TEXT NOT NULL,
                ref TEXT NOT NULL,
                note TEXT NOT NULL,
                created_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(thread_id, source_ts, kind, ref)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_summary_addenda_thread
            ON thread_summary_addenda(thread_id, source_ts, id)
        """)

        # F32: thread-scoped code-interpreter containers. One OpenAI container per thread, so
        # the model's sandbox state (files in /mnt/data, loaded dataframes) survives the turn
        # boundary within a conversation.
        #
        # `published_files_json` is NOT bookkeeping fluff — it is a correctness guard. A reused
        # container's listing still contains every file from earlier turns, so without a durable
        # record of what we already uploaded, a bot restart mid-conversation would re-post turn
        # 1's chart on turn 2 (the in-memory dedupe dies with the process). It lives here, next
        # to the container id, because it is meaningless once the container is gone.
        #
        # No FK: PRAGMA foreign_keys is never enabled, and a container can outlive its threads
        # row. Rows are swept by age in the daily cleanup instead.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS thread_containers (
                thread_id TEXT PRIMARY KEY,
                container_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                published_files_json TEXT
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_thread_containers_last_used
            ON thread_containers(last_used_at)
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

        # F51 — Ambient artifacts. Derived summaries for images/links/files posted ambiently
        # (or addressed) in a channel/thread, kept in the running context even when the bot
        # doesn't respond. CHANNEL + source-ts keyed (NOT the colon thread key that locked out
        # the incident lookup). conversation_ts is the thread root (= source_ts for a top-level
        # message, NOT nullable) so thread retrieval + compaction stay deterministic without
        # ever splitting a colon-composed key. summary/model are NULL for pending/failed rows.
        # Slack stays the only transcript — this holds ONLY derivations + refs, never message
        # text mirrors, never image bytes (CLAUDE.md pitfall 4). Reuse is SAME-CHANNEL only.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ambient_artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                source_ts TEXT NOT NULL,
                conversation_ts TEXT NOT NULL,
                kind TEXT NOT NULL,               -- 'image' | 'link' | 'file'
                ref TEXT NOT NULL,                -- Slack file id, or normalized URL
                title TEXT,
                summary TEXT,                     -- NULL until ready
                model TEXT,
                status TEXT NOT NULL DEFAULT 'pending',  -- pending|ready|failed|blocked|omitted
                derivation_source TEXT,           -- gate_vision|vision_worker|fetch|unfurl|document
                content_type TEXT,
                error_code TEXT,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                fetched_at TIMESTAMP,
                expires_at TIMESTAMP,
                UNIQUE(channel_id, source_ts, kind, ref)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ambient_source
            ON ambient_artifacts(channel_id, source_ts)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ambient_conversation
            ON ambient_artifacts(channel_id, conversation_ts, source_ts)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ambient_ref
            ON ambient_artifacts(channel_id, kind, ref, status, fetched_at)
        """)

        # Tool-use provenance (F7): compact per-reply record of the tools the bot invoked
        # (names + arg-derived gists only, NO results/content), keyed by the reply's Slack
        # ts. Reinjected as a "[used tools: …]" annotation on rebuild so the model can
        # recall its own past tool use. Deliberately NO foreign key: the ON DELETE CASCADE
        # path is dead (PRAGMA foreign_keys is never enabled) — rows are swept by age via
        # delete_old_tool_usage() instead. UNIQUE(channel, ts) makes re-persist idempotent.
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS message_tool_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id TEXT NOT NULL,
                message_ts TEXT NOT NULL,
                thread_key TEXT NOT NULL,
                tools_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(channel_id, message_ts)
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tool_usage_thread
            ON message_tool_usage(thread_key)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tool_usage_created
            ON message_tool_usage(created_at)
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
                muted_threads TEXT,
                model TEXT,
                reasoning_effort TEXT,
                verbosity TEXT,
                ambient_memory INTEGER,
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
    
    @contextmanager
    def _migration_step(self, name: str):
        """Run one migration phase in isolation.

        A raising phase is logged LOUDLY (named, with a traceback) and the
        remaining phases still run. Previously the whole migration body sat under
        a single try/except, so one bad step silently skipped every later step and
        the bot then served traffic on a half-migrated schema.
        """
        try:
            yield
        except Exception as e:
            self.log_error(f"DB: Migration step '{name}' FAILED: {e}", exc_info=True)

    def _is_pre_v3_database(self) -> bool:
        """True when this database still has the pre-v3 (v2.x) shape.

        Cheap, read-only, and deliberately conservative — it must NOT fire on a
        brand-new database (init_schema's CREATE TABLE IF NOT EXISTS block runs
        immediately before the migrations, so a fresh DB already has `documents`
        and `user_preferences`) nor on an already-migrated one (second boot).

        Legacy signals, any one of which is decisive:
        - the `messages` mirror table exists (dropped by the v3 mirror-drop)
        - `documents.content` exists (dropped by the v3 doc-content-drop)
        - `user_preferences` is missing the `gpt56_migrated` sentinel AND already
          holds rows. The sentinel is added by the migration, not by CREATE TABLE,
          so a fresh DB also lacks it — the row check is what distinguishes real
          user preferences (about to be bulk-overwritten) from an empty new table.
        """
        cursor = self.conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='messages'
        """)
        if cursor.fetchone():
            return True

        cursor = self.conn.execute("PRAGMA table_info(documents)")
        if any(col[1] == 'content' for col in cursor.fetchall()):
            return True

        cursor = self.conn.execute("PRAGMA table_info(user_preferences)")
        up_columns = [col[1] for col in cursor.fetchall()]
        if up_columns and 'gpt56_migrated' not in up_columns:
            cursor = self.conn.execute("SELECT 1 FROM user_preferences LIMIT 1")
            if cursor.fetchone():
                return True

        return False

    def _run_migrations(self):
        """Run database migrations to update schema for existing databases.

        Each phase is isolated by `_migration_step` so a failure is loud and
        contained instead of silently skipping every later phase.
        """
        # Rollback path FIRST: snapshot the database before any migration writes to
        # it. The gpt-5.6 swap below bulk-overwrites every user's model/effort, and
        # the two destructive drops each take their own tagged backup only AFTER
        # that swap has already run — so without this, no backup can restore what
        # users actually picked. Runs at most once per database (see detection).
        with self._migration_step("pre-v3 backup"):
            if self._is_pre_v3_database():
                self.log_info(
                    "DB: Pre-v3 database detected — backup tagged pre-v3-upgrade before migrating"
                )
                self.backup_database(tag="pre-v3-upgrade")

        with self._migration_step("channel_settings columns"):
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
            # F15: muted_threads (JSON list) — threads permanently opted out of unprompted
            # participation via a "butt out" backoff. Replaces the snoozed_until timer rail.
            if cs_columns and 'muted_threads' not in cs_columns:
                self.log_info("DB: Adding muted_threads column to channel_settings")
                self.conn.execute("ALTER TABLE channel_settings ADD COLUMN muted_threads TEXT")
                self.conn.commit()
            # Shared per-channel model/effort/verbosity overrides (NULL = inherit)
            for col in ("model", "reasoning_effort", "verbosity"):
                if cs_columns and col not in cs_columns:
                    self.log_info(f"DB: Adding {col} column to channel_settings")
                    self.conn.execute(f"ALTER TABLE channel_settings ADD COLUMN {col} TEXT")
                    self.conn.commit()
            # F51: per-channel ambient-memory opt-out (NULL = inherit ENABLE_AMBIENT_MEMORY;
            # 0 = memory off for this channel, distinct from participation `off`).
            if cs_columns and 'ambient_memory' not in cs_columns:
                self.log_info("DB: Adding ambient_memory column to channel_settings")
                self.conn.execute("ALTER TABLE channel_settings ADD COLUMN ambient_memory INTEGER")
                self.conn.commit()

        with self._migration_step("images.message_ts"):
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

        with self._migration_step("users.real_name"):
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

        with self._migration_step("user_preferences.custom_instructions"):
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

        with self._migration_step("users.email"):
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

        with self._migration_step("user_preferences table"):
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

        with self._migration_step("user_preferences.image_quality"):
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

        with self._migration_step("user_preferences.image_background"):
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

        with self._migration_step("user_preferences.enable_mcp"):
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

        with self._migration_step("user_preferences.image_model"):
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

        with self._migration_step("gpt-5.5 swap"):
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

        with self._migration_step("settings_completed backfill"):
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

        with self._migration_step("mirror drop"):
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

        with self._migration_step("doc-content drop"):
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

        with self._migration_step("mcp_tools table"):
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

        Both parts are individually isolated: a failure in the one-time swap must
        not take the every-startup normalizers down with it, since those are what
        keep the API layer from ever seeing a dropped model or a rejected effort.
        """
        with self._migration_step("gpt-5.6 migration"):
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

        with self._migration_step("gpt-5.6 normalizers"):
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

        with self._migration_step("participation redesign: memory cleanup"):
            self._migrate_participation_redesign()

        with self._migration_step("drop channel_thread_mutes table"):
            # The per-thread mute mechanism was removed entirely. Drop the normalized table Layer
            # 0 created. Separately keyed from the participation-redesign step above (not folded
            # into it) so it is isolated by its own try/except and runs on every boot — including
            # the already-migrated live DB, which still has the table. DROP TABLE IF EXISTS is
            # idempotent: a no-op on a fresh DB that never created it and on every re-run.
            self.conn.execute("DROP TABLE IF EXISTS channel_thread_mutes")
            self.conn.commit()

    def _migrate_participation_redesign(self):
        """Layer 0 of the participation-backoff redesign. Idempotent and re-runnable.

        Runs on EVERY init (``_migration_step`` is only a try/except, not a one-time guard), so
        every step below MUST converge to the same state on a re-run and MUST NOT clobber rows
        the running system now owns.

        1. Clear the legacy JSON channel_settings.muted_threads column. The per-thread mute
           mechanism was removed, so there is no longer any table to migrate the entries into —
           the column is dead weight. Nothing reads it anymore, so nulling it is inert for the
           running system, and clearing converges on the same state on every re-run. (The table
           itself is dropped by a separate, later migration step.)
        2. Delete the old auto-written "butt out / raise the bar" channel-memory facts
           (author LIKE 'participation_engine:%') — BUT NOT the new per-dimension preference
           markers (author LIKE 'participation_engine:pref:%'), which are live state the
           redesign writes and must survive every restart. The old generic facts kept the
           classifier suppressed channel-wide after the fix; the pref markers replace them.
        3. Collapse any duplicate preference markers to one row per (channel, dimension) and
           enforce that with a partial UNIQUE index — the invariant upsert_channel_pref_memory
           relies on. Done here (not in init_schema) so the dedup runs BEFORE the UNIQUE index
           is created and a re-run never trips over pre-existing duplicates.

        Structural channel_settings columns are deliberately NOT rewritten: the mute clobber
        overwrote updated_by, so we cannot prove the stored values were implicit, and the one
        affected channel was already reset by hand.
        """
        # 1. CLEAR the legacy JSON muted_threads column. The per-thread mute mechanism was
        #    removed, so there is nothing to copy anywhere — just null out the inert column so no
        #    stale blob lingers. This is the only writer of the column and it converges on the
        #    same state every re-run.
        cleared = self.conn.execute(
            "UPDATE channel_settings SET muted_threads = NULL "
            "WHERE muted_threads IS NOT NULL AND muted_threads != ''"
        )
        if cleared.rowcount:
            self.log_info(
                f"DB: Cleared legacy muted_threads JSON on {cleared.rowcount} channel(s)")

        # 2. Remove the stale severe participation-engine memory facts — but PRESERVE the
        #    per-dimension preference markers (participation_engine:pref:*), which are live
        #    redesign state, not stale suppression facts.
        deleted = self.conn.execute(
            "DELETE FROM channel_memory "
            "WHERE author LIKE 'participation_engine:%' "
            "AND author NOT LIKE 'participation_engine:pref:%'"
        )
        if deleted.rowcount:
            self.log_info(
                f"DB: Removed {deleted.rowcount} stale participation-engine memory fact(s)")

        # 3. One preference marker per (channel, dimension). Collapse any duplicates BEFORE
        #    creating the partial UNIQUE index the upsert relies on.
        #    SHOULD-FIX 2: the marker is a CHANNEL-scope row (the upsert only ever writes/reads
        #    scope='channel'), so both the dedupe and the index predicate must be scoped to
        #    'channel' — otherwise a same-named WORKSPACE row could be swept by the dedupe or
        #    collide with a valid channel marker in the (channel_id, author) unique index. And
        #    keep the FRESHEST row (greatest updated_ts, id as the tie-breaker), not merely the
        #    highest id: the upsert refreshes a marker in place with a new updated_ts, so a
        #    later-refreshed but lower-id row must win over a stale higher-id duplicate.
        #    COALESCE guards any legacy NULL updated_ts (falls back to created_ts, then '').
        self.conn.execute(
            "DELETE FROM channel_memory "
            "WHERE author LIKE 'participation_engine:pref:%' AND scope = 'channel' "
            "AND id NOT IN ("
            "  SELECT m.id FROM channel_memory m "
            "  WHERE m.author LIKE 'participation_engine:pref:%' AND m.scope = 'channel' "
            "  AND NOT EXISTS ("
            "    SELECT 1 FROM channel_memory m2 "
            "    WHERE m2.author LIKE 'participation_engine:pref:%' AND m2.scope = 'channel' "
            "    AND m2.channel_id = m.channel_id AND m2.author = m.author "
            "    AND (COALESCE(m2.updated_ts, m2.created_ts, '') > COALESCE(m.updated_ts, m.created_ts, '') "
            "      OR (COALESCE(m2.updated_ts, m2.created_ts, '') = COALESCE(m.updated_ts, m.created_ts, '') "
            "          AND m2.id > m.id))))"
        )
        # Drop first so a re-run REPLACES an index created under the old (scope-agnostic)
        # predicate; CREATE ... IF NOT EXISTS alone would silently keep the stale predicate.
        self.conn.execute("DROP INDEX IF EXISTS idx_channel_memory_pref_marker")
        self.conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_channel_memory_pref_marker "
            "ON channel_memory (channel_id, author) "
            "WHERE author LIKE 'participation_engine:pref:%' AND scope = 'channel'"
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
            "snoozed_until, muted_threads, model, reasoning_effort, verbosity, updated_ts, updated_by "
            "FROM channel_settings WHERE channel_id = ?",
            (channel_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {
            "response_mode": row["response_mode"],
            "directives": row["directives"],
            # NULL reply_in_channel stays None (inherit → resolves to config.reply_in_channel_default
            # at read time). Collapsing NULL to False here erased the inherit distinction.
            "reply_in_channel": (None if row["reply_in_channel"] is None
                                 else bool(row["reply_in_channel"])),
            "participation_level": row["participation_level"],
            "snoozed_until": row["snoozed_until"],
            "muted_threads": _decode_muted_threads(row["muted_threads"]),
            "model": row["model"],
            "reasoning_effort": row["reasoning_effort"],
            "verbosity": row["verbosity"],
            "updated_ts": row["updated_ts"],
            "updated_by": row["updated_by"],
        }

    def set_channel_settings(self, channel_id: str, response_mode=_UNSET,
                             directives=_UNSET, reply_in_channel=_UNSET,
                             participation_level=_UNSET, snoozed_until=_UNSET,
                             muted_threads=_UNSET,
                             model=_UNSET, reasoning_effort=_UNSET, verbosity=_UNSET,
                             ambient_memory=_UNSET, updated_by: Optional[str] = None):
        """Upsert per-channel settings (Phase 7; Phase F adds participation_level/snoozed_until).

        Atomic partial write: ONLY the explicitly-provided fields are written — omitted fields are
        preserved untouched (never rewritten, so no clobber and no race with a concurrent save).
        An explicit value sets a field; an explicit None CLEARS it (→ NULL) so the modal's "inherit
        from global default" stores NULL rather than a copied default (NULL then resolves to the
        global default at read time). updated_ts/updated_by bump only when a STRUCTURAL field
        changed. muted_threads is a deprecated inert JSON column (nothing reads it — the
        per-thread mute mechanism was removed); it takes a Python list, None/[] clears it.
        """
        built = _build_channel_settings_write(
            channel_id, response_mode=response_mode, directives=directives,
            reply_in_channel=reply_in_channel, participation_level=participation_level,
            snoozed_until=snoozed_until, muted_threads=muted_threads, model=model,
            reasoning_effort=reasoning_effort, verbosity=verbosity,
            ambient_memory=ambient_memory, updated_by=updated_by)
        if built is None:
            return
        sql, params = built
        self.conn.execute(sql, params)
        self.conn.commit()
        logger.debug(f"Saved channel_settings for {channel_id}")

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
        """Delete the compaction summary for a thread (and its late-artifact addenda —
        PRAGMA foreign_keys is never enabled, so the cascade is explicit)."""
        self.conn.execute("DELETE FROM thread_summaries WHERE thread_id = ?", (thread_id,))
        self.conn.execute("DELETE FROM thread_summary_addenda WHERE thread_id = ?", (thread_id,))

    # Thread summary addenda (F51c — late-artifact context folded onto the summary head)
    async def add_thread_summary_addendum_async(
        self, thread_id: str, channel_id: str, source_ts: str, kind: str, ref: str, note: str,
        *, cap: int = _MAX_SUMMARY_ADDENDA_PER_THREAD,
    ) -> bool:
        """Record a late/folded ambient-artifact note against a thread's compaction summary.

        Idempotent on (thread_id, source_ts, kind, ref) so the completion-time path and the
        compaction-time path can't double-record the same note. Bounded: at most `cap` addenda
        per thread (a pathological channel can't bloat the summary head). Returns True when a
        row was actually inserted."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute(
                "SELECT 1 FROM thread_summary_addenda "
                "WHERE thread_id = ? AND source_ts = ? AND kind = ? AND ref = ?",
                (thread_id, source_ts, kind, ref)) as cur:
                if await cur.fetchone():
                    return False  # already recorded (idempotent) — doesn't re-count against cap
            async with db.execute(
                "SELECT COUNT(*) FROM thread_summary_addenda WHERE thread_id = ?",
                (thread_id,)) as cur:
                row = await cur.fetchone()
                if row and int(row[0]) >= int(cap):
                    return False  # cap reached — drop silently rather than bloat the head
            await db.execute(
                "INSERT OR IGNORE INTO thread_summary_addenda "
                "(thread_id, channel_id, source_ts, kind, ref, note) VALUES (?, ?, ?, ?, ?, ?)",
                (thread_id, channel_id, source_ts, kind, ref, note))
            await db.commit()
        return True

    async def get_thread_summary_addenda_async(self, thread_id: str) -> List[Dict]:
        """Late-artifact addenda for a thread, deterministically ordered (source_ts, id) so
        every rebuild serializes the summary head identically (prompt-cache hygiene). source_ts
        is a numeric Slack ts stored as TEXT, so order by its REAL value, not string collation."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute(
                "SELECT * FROM thread_summary_addenda WHERE thread_id = ? "
                "ORDER BY CAST(source_ts AS REAL) ASC, id ASC", (thread_id,)) as cursor:
                return [dict(row) async for row in cursor]

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

    def restore_document_derived(self, thread_id: str, filename: str, *,
                                 summary: Optional[str] = None,
                                 page_structure: Optional[Dict] = None,
                                 total_pages: Optional[int] = None,
                                 size_bytes: Optional[int] = None,
                                 message_ts: Optional[str] = None) -> int:
        """Re-hydrate a SLIMMED document row IN PLACE and return the rows updated.

        Retention (delete_old_documents) nulls a row's derived bulk (summary/page_structure/
        metadata) but keeps the Slack ref. When a rebuild re-derives that content it must UPDATE
        the preserved row, not INSERT a second one: the `documents` table has no
        UNIQUE(thread_id, filename) constraint, so a fresh save_document each retention/rebuild
        cycle accumulates duplicate reference rows. Returns 0 when no matching row exists — the
        caller then falls back to inserting (a genuinely legacy, never-stored document)."""
        try:
            cursor = self.conn.execute("""
                UPDATE documents
                SET summary = ?, page_structure = ?, total_pages = ?,
                    size_bytes = COALESCE(?, size_bytes),
                    message_ts = COALESCE(?, message_ts)
                WHERE thread_id = ? AND filename = ?
            """, (summary,
                  json.dumps(page_structure) if page_structure else None,
                  total_pages, size_bytes, message_ts, thread_id, filename))
            if cursor.rowcount:
                self.update_thread_activity(thread_id)
                self.log_info(f"DB: Re-hydrated slimmed document {filename} for thread {thread_id}")
            return cursor.rowcount or 0
        except Exception as e:
            self.log_error(f"DB: Failed to restore document {filename} - {e}", exc_info=True)
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
        """SLIM document-extraction rows older than `days` (retention sweep) — do NOT delete them.

        The reference row (filename, thread/channel key, Slack file_id/url_private, mime_type,
        size, timestamps) is PRESERVED so read_document and thread rebuilds can always re-resolve
        and re-extract the file from Slack on demand. Only the bulky DERIVED fields (summary,
        page_structure, metadata) are nulled. This is the fix for the compaction-boundary gap:
        a document behind a compaction boundary is never recreated by a rebuild, so DELETING its
        row made a 100-day-old-but-still-in-Slack file unresolvable (`document_not_found`) even
        though the summary head still referenced it. Slimming ages out the bulk while keeping the
        row reachable indefinitely. created_at defaults to CURRENT_TIMESTAMP (UTC), so the cutoff
        is computed IN SQL with datetime('now', …) — a Python datetime.now() cutoff would be LOCAL
        time and skew the retention window on non-UTC hosts. Same trap as delete_old_tool_usage."""
        cursor = self.conn.execute("""
            UPDATE documents
            SET summary = NULL, page_structure = NULL, metadata_json = NULL
            WHERE created_at < datetime('now', ?)
              AND (summary IS NOT NULL OR page_structure IS NOT NULL OR metadata_json IS NOT NULL)
        """, (f"-{int(days)} days",))

        if cursor.rowcount > 0:
            self.log_info(f"DB: Slimmed {cursor.rowcount} documents older than {days} days "
                          "(refs kept for on-demand re-extraction, derived bulk cleared)")

    def delete_old_tool_usage(self, days: int = 90):
        """Delete tool-use provenance rows older than `days` (F7 retention sweep).

        The ON DELETE CASCADE path is dead (PRAGMA foreign_keys is never enabled), so
        message_tool_usage gets this explicit age sweep instead, wired into the scheduled
        cleanup worker. created_at defaults to CURRENT_TIMESTAMP (UTC), so the cutoff is
        computed in SQL with datetime('now', …) — a Python datetime.now() cutoff would be
        LOCAL time and skew the retention window on non-UTC hosts."""
        cursor = self.conn.execute("""
            DELETE FROM message_tool_usage
            WHERE created_at < datetime('now', ?)
        """, (f"-{int(days)} days",))

        if cursor.rowcount > 0:
            self.log_info(f"DB: Cleaned up {cursor.rowcount} tool-usage rows older than {days} days")

    # F32: thread-scoped code-interpreter containers
    #
    # Every staleness cutoff below is computed IN SQL with datetime('now', …). last_used_at
    # defaults to CURRENT_TIMESTAMP, which SQLite writes in UTC — a Python datetime.now()
    # cutoff would be LOCAL time and, on this host (UTC-4), would judge every container
    # four hours fresher than it is. Same trap as delete_old_tool_usage above.

    # EVERY mutation below is conditional on `container_id`, not just `thread_id`. A row can be
    # rebound to a NEW container at any moment (the old one expired, a turn recreated it), and a
    # thread_id-only write then lands on the wrong container: the daily reaper, having selected
    # stale container X, would delete the row for its live replacement Y; and a late publication
    # for X would write X's file ids into Y's dedupe list, suppressing Y's real artifacts.
    _CONTAINER_PUBLISHED_CAP = 2048  # ~8x what one 20-minute container could plausibly hold

    @staticmethod
    def _container_row(row) -> Optional[Dict]:
        if not row:
            return None
        result = dict(row)
        try:
            result["published_files"] = json.loads(result.get("published_files_json") or "[]")
        except (json.JSONDecodeError, TypeError):
            result["published_files"] = []
        return result

    def get_thread_container(self, thread_id: str) -> Optional[Dict]:
        """The thread's container binding, regardless of age.

        Deliberately NOT age-filtered. Age belongs to container *selection* only. The dedupe
        record must stay readable for as long as the binding exists: a single turn can run
        longer than the reuse window (a tool loop with slow tools), and if publication could no
        longer read its own published-file list it would re-post every earlier artifact still
        sitting in the container.
        """
        cursor = self.conn.execute("""
            SELECT thread_id, container_id, published_files_json, created_at, last_used_at
            FROM thread_containers WHERE thread_id = ?
        """, (thread_id,))
        return self._container_row(cursor.fetchone())

    def get_fresh_thread_container(self, thread_id: str, reuse_minutes: int) -> Optional[Dict]:
        """The thread's container, but ONLY if we used it within `reuse_minutes`.

        A row older than that is not returned: the container has almost certainly idle-expired
        (20-minute API ceiling), and handing OpenAI a dead id would fail the whole turn.
        Callers treat None as "create a fresh one".
        """
        cursor = self.conn.execute("""
            SELECT thread_id, container_id, published_files_json, created_at, last_used_at
            FROM thread_containers
            WHERE thread_id = ? AND last_used_at > datetime('now', ?)
        """, (thread_id, f"-{int(reuse_minutes)} minutes"))
        return self._container_row(cursor.fetchone())

    def save_thread_container(self, thread_id: str, container_id: str):
        """Bind a NEW container to a thread, clearing the published-file record.

        The reset is deliberate: a new container starts empty, so nothing has been published
        out of it yet. Carrying the old ids forward would let a stale id suppress a genuinely
        new artifact that happened to reuse it.
        """
        self.conn.execute("""
            INSERT INTO thread_containers (thread_id, container_id, created_at, last_used_at,
                                           published_files_json)
            VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, '[]')
            ON CONFLICT(thread_id) DO UPDATE SET
                container_id = excluded.container_id,
                created_at = CURRENT_TIMESTAMP,
                last_used_at = CURRENT_TIMESTAMP,
                published_files_json = '[]'
        """, (thread_id, container_id))
        self.conn.commit()

    def touch_thread_container(self, thread_id: str, container_id: str):
        """Mark this container as used now (keeps it inside the reuse window)."""
        self.conn.execute("""
            UPDATE thread_containers SET last_used_at = CURRENT_TIMESTAMP
            WHERE thread_id = ? AND container_id = ?
        """, (thread_id, container_id))
        self.conn.commit()

    def add_published_container_files(self, thread_id: str, container_id: str,
                                      file_ids: List[str]):
        """Durably record container file ids already handled, so they are never posted twice.

        Holds two kinds of id, and treats them identically because their effect is identical —
        "not eligible for publication": files we uploaded to Slack, and files already sitting in
        the container when a turn started (the baseline). Without this record a bot restart
        mid-conversation re-posts every earlier artifact still in the reused container.
        """
        if not file_ids:
            return
        cursor = self.conn.execute(
            "SELECT published_files_json FROM thread_containers "
            "WHERE thread_id = ? AND container_id = ?", (thread_id, container_id))
        row = cursor.fetchone()
        if not row:
            # The row was rebound to a different container while this turn ran. These ids belong
            # to a container that no longer backs this thread; writing them would corrupt the
            # new container's dedupe list.
            return
        try:
            existing = json.loads(row["published_files_json"] or "[]")
        except (json.JSONDecodeError, TypeError):
            existing = []
        merged = list(dict.fromkeys([*existing, *file_ids]))[-self._CONTAINER_PUBLISHED_CAP:]
        self.conn.execute("""
            UPDATE thread_containers SET published_files_json = ?
            WHERE thread_id = ? AND container_id = ?
        """, (json.dumps(merged), thread_id, container_id))
        self.conn.commit()

    def delete_thread_container(self, thread_id: str, container_id: Optional[str] = None):
        """Forget a container binding (it expired, or the API told us it is gone).

        `container_id` scopes the delete so a slow reaper cannot drop a binding that a live turn
        has already replaced. Omitted only where the caller genuinely means "whatever is bound".
        """
        if container_id is None:
            self.conn.execute("DELETE FROM thread_containers WHERE thread_id = ?", (thread_id,))
        else:
            self.conn.execute(
                "DELETE FROM thread_containers WHERE thread_id = ? AND container_id = ?",
                (thread_id, container_id))
        self.conn.commit()

    def get_expired_thread_containers(self, older_than_minutes: int) -> List[Dict]:
        """Rows whose container is certainly dead — for the daily reap."""
        cursor = self.conn.execute("""
            SELECT thread_id, container_id FROM thread_containers
            WHERE last_used_at <= datetime('now', ?)
        """, (f"-{int(older_than_minutes)} minutes",))
        return [dict(r) for r in cursor.fetchall()]

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
        """Remove SCHEDULED backups older than 7 days.

        Only untagged nightly backups ({platform}_{date}_{time}.db) are pruned.
        Tagged backups ({platform}_{tag}_{date}_{time}.db — pre-v3-upgrade and the
        two migration drops) are an operator's only rollback path out of an
        irreversible upgrade, and retention must never eat them: the nightly backup
        calls this on every run, so a 7-day sweep would delete the pre-upgrade
        snapshot exactly one week after the upgrade. They are removed by hand.
        """
        cutoff = datetime.now() - timedelta(days=7)
        # Untagged shape only: platform_YYYYMMDD_HHMMSS.db — anything with an extra
        # segment carries a tag and is kept.
        scheduled = re.compile(rf"^{re.escape(self.platform)}_(\d{{8}})_(\d{{6}})\.db$")

        for filename in os.listdir(f"{self.db_dir}/backups"):
            match = scheduled.match(filename)
            if not match:
                continue
            try:
                file_date = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
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

    @staticmethod
    def _merge_tool_provenance(existing: List[Dict], new: List[Dict]) -> List[Dict]:
        """Merge two provenance lists (existing first, then new), ORDER-preserving.

        Two entry classes coexist: F7 used-tools entries (name + gist) and F12 result-digest
        entries (name + result_digest). They never collapse into each other.

        Used-tools: a tool that genuinely ran more than once (same name, different gists) is
        kept as multiple entries; only EXACT duplicates (same name AND gist) are deduped; an
        empty-gist placeholder is UPGRADED in place by a later non-empty gist for the same
        tool. Capped at config.tool_provenance_max_entries (F14; default 20, was 8) so the
        persisted row honors the same budget build_provenance applies.

        Result-digests (F12): deduped by (name, digest) so re-persist is idempotent but two
        distinct outputs from the same server are both kept. Already char-bounded at capture,
        so NOT subject to the used-tools entry cap; appended AFTER the used-tools entries
        (matches the pinned [used tools:] → [tool results:] render order). Old rows (no
        result_digest) merge exactly as before."""
        used: List[Dict] = []
        results: List[Dict] = []
        seen_results = set()
        for entry in list(existing or []) + list(new or []):
            if not isinstance(entry, dict):
                continue
            name = entry.get("tool_name")
            if not name:
                continue
            digest = entry.get("result_digest")
            if digest:
                key = (name, digest)
                if key in seen_results:
                    continue  # exact (name, digest) duplicate — idempotent re-persist
                seen_results.add(key)
                results.append({"tool_name": name, "result_digest": digest})
                continue
            gist = entry.get("gist") or ""
            if any(m["tool_name"] == name and m["gist"] == gist for m in used):
                continue  # exact duplicate — dedupe
            if gist.strip():
                placeholder = next(
                    (m for m in used if m["tool_name"] == name and not m["gist"].strip()), None)
                if placeholder is not None:
                    placeholder["gist"] = gist  # upgrade empty placeholder in place
                    continue
            elif any(m["tool_name"] == name and m["gist"].strip() for m in used):
                continue  # empty gist already covered by a non-empty entry for this tool
            used.append({"tool_name": name, "gist": gist})
        from config import config
        return used[:int(getattr(config, "tool_provenance_max_entries", 20))] + results

    async def save_tool_usage_async(self, channel_id: str, message_ts: str,
                                    thread_key: str, tools: List[Dict]) -> None:
        """Persist a reply's tool-use provenance (F7), keyed by channel+ts.

        `tools` is the compact [{"tool_name","gist"}] record (names + arg gists), with
        optional F12 [{"tool_name","result_digest"}] MCP result-memory entries appended.
        Idempotent on (channel_id, message_ts): a re-persist MERGES with the existing row
        (union by tool_name, preferring a non-empty gist) rather than last-write-wins, so a
        second pass can't drop tools recorded by the first. Best-effort — the caller wraps
        this so a DB failure never blocks the reply."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("PRAGMA journal_mode=WAL")
                # BEGIN IMMEDIATE takes the write lock up front so the read-modify-write
                # (SELECT existing → merge → UPSERT) is atomic against a concurrent
                # persist for the same reply — otherwise two passes could each read the
                # old row and the second would clobber the first's merged tools.
                await db.execute("BEGIN IMMEDIATE")
                existing: List[Dict] = []
                async with db.execute(
                    "SELECT tools_json FROM message_tool_usage WHERE channel_id = ? AND message_ts = ?",
                    (channel_id, message_ts)
                ) as cur:
                    row = await cur.fetchone()
                if row and row[0]:
                    try:
                        parsed = json.loads(row[0])
                        if isinstance(parsed, list):
                            existing = parsed
                    except (json.JSONDecodeError, TypeError, ValueError):
                        existing = []
                merged = self._merge_tool_provenance(existing, tools)
                await db.execute("""
                    INSERT INTO message_tool_usage
                        (channel_id, message_ts, thread_key, tools_json)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(channel_id, message_ts) DO UPDATE SET
                        thread_key = excluded.thread_key,
                        tools_json = excluded.tools_json
                """, (channel_id, message_ts, thread_key, json.dumps(merged)))
                await db.commit()
        except Exception as e:
            self.log_debug(f"DB: save_tool_usage_async failed (non-fatal): {e}")

    async def get_thread_tool_usage_async(self, thread_key: str) -> Dict[str, List[Dict]]:
        """Batch-fetch a thread's tool-use provenance for rebuild reinjection (F7).

        Returns {message_ts: [{"tool_name","gist"}, …]}. Empty on any error so a missing
        table / read failure degrades to no annotations rather than breaking the rebuild."""
        result: Dict[str, List[Dict]] = {}
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                await db.execute("PRAGMA journal_mode=WAL")
                async with db.execute(
                    "SELECT message_ts, tools_json FROM message_tool_usage WHERE thread_key = ?",
                    (thread_key,)
                ) as cursor:
                    async for row in cursor:
                        try:
                            parsed = json.loads(row["tools_json"])
                        except (json.JSONDecodeError, TypeError, ValueError):
                            continue
                        if isinstance(parsed, list):
                            result[row["message_ts"]] = parsed
        except Exception as e:
            self.log_debug(f"DB: get_thread_tool_usage_async failed (non-fatal): {e}")
        return result

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
                "snoozed_until, muted_threads, model, reasoning_effort, verbosity, "
                "ambient_memory, updated_ts, updated_by "
                "FROM channel_settings WHERE channel_id = ?",
                (channel_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if not row:
                    return None
                return {
                    "response_mode": row["response_mode"],
                    "directives": row["directives"],
                    # F51 opt-out: NULL → None (inherit config.enable_ambient_memory at read time).
                    "ambient_memory": (None if row["ambient_memory"] is None
                                       else bool(row["ambient_memory"])),
                    # NULL stays None (inherit → config.reply_in_channel_default at read time).
                    "reply_in_channel": (None if row["reply_in_channel"] is None
                                         else bool(row["reply_in_channel"])),
                    "participation_level": row["participation_level"],
                    "snoozed_until": row["snoozed_until"],
                    "muted_threads": _decode_muted_threads(row["muted_threads"]),
                    "model": row["model"],
                    "reasoning_effort": row["reasoning_effort"],
                    "verbosity": row["verbosity"],
                    "updated_ts": row["updated_ts"],
                    "updated_by": row["updated_by"],
                }

    async def set_channel_settings_async(self, channel_id: str, response_mode=_UNSET,
                                         directives=_UNSET, reply_in_channel=_UNSET,
                                         participation_level=_UNSET, snoozed_until=_UNSET,
                                         muted_threads=_UNSET,
                                         model=_UNSET, reasoning_effort=_UNSET, verbosity=_UNSET,
                                         ambient_memory=_UNSET, updated_by: Optional[str] = None):
        """Async version of set_channel_settings (Phase F adds participation_level/snoozed_until).

        Atomic partial write — ONLY provided fields are written, omitted fields preserved (no
        read-modify-write of the whole row, so no race with a concurrent modal save). Explicit None
        CLEARS to NULL (inherit); updated_ts/updated_by bump only on a STRUCTURAL change.
        muted_threads is a deprecated inert JSON column (nothing reads it — the per-thread mute
        mechanism was removed).
        """
        built = _build_channel_settings_write(
            channel_id, response_mode=response_mode, directives=directives,
            reply_in_channel=reply_in_channel, participation_level=participation_level,
            snoozed_until=snoozed_until, muted_threads=muted_threads, model=model,
            reasoning_effort=reasoning_effort, verbosity=verbosity,
            ambient_memory=ambient_memory, updated_by=updated_by)
        if built is None:
            return
        sql, params = built
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(sql, params)
            await db.commit()
            logger.debug(f"Saved channel_settings for {channel_id} (async)")

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

    async def upsert_channel_pref_memory(self, channel_id: str, marker_author: str,
                                         content: str, max_rows: Optional[int] = None
                                         ) -> Optional[int]:
        """Atomically write-or-refresh the SINGLE per-channel participation preference marker
        (author == ``marker_author``, e.g. ``participation_engine:pref:reactions``), returning
        its row id — or ``None`` when a new marker is declined because the channel is at the
        memory-row cap.

        Why a bespoke helper (participation redesign, SHOULD-FIX #8): the old
        read-all-then-insert in _apply_pref_memory raced (two concurrent "react less" verdicts
        both saw "no marker" and both INSERTed a duplicate) and an ``update:<id>`` path could
        leave the row authored by something other than the marker. This does the existence
        check, the cap check, and the write inside ONE ``BEGIN IMMEDIATE`` transaction (the
        connection is opened in autocommit so the explicit transaction is ours to control), so
        concurrent callers serialize and converge on exactly one marker row per (channel,
        dimension) — an invariant also pinned by the partial UNIQUE index
        idx_channel_memory_pref_marker. The stored/updated row's author is ALWAYS the marker.

        The cap mirrors remember_fact: at MEMORY_MAX_ROWS with no marker yet, decline rather
        than evict a human's memory. An existing marker is always refreshed (a refresh frees no
        slot and adds no row, so the cap never blocks it).
        """
        if not channel_id or not marker_author:
            return None
        async with aiosqlite.connect(self.db_path, isolation_level=None) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA busy_timeout=5000")
            await db.execute("BEGIN IMMEDIATE")
            try:
                async with db.execute(
                    "SELECT id FROM channel_memory "
                    "WHERE channel_id = ? AND author = ? AND scope = 'channel' LIMIT 1",
                    (channel_id, marker_author),
                ) as cur:
                    existing = await cur.fetchone()
                if existing is not None:
                    row_id = existing[0]
                    await db.execute(
                        "UPDATE channel_memory SET content = ?, updated_ts = CURRENT_TIMESTAMP "
                        "WHERE id = ?",
                        (content, row_id),
                    )
                    await db.execute("COMMIT")
                    return row_id
                if max_rows is not None:
                    async with db.execute(
                        "SELECT COUNT(*) FROM channel_memory "
                        "WHERE scope = 'channel' AND channel_id = ?",
                        (channel_id,),
                    ) as cur:
                        (count,) = await cur.fetchone()
                    if count >= max(1, int(max_rows)):
                        await db.execute("ROLLBACK")
                        return None
                cur = await db.execute(
                    "INSERT INTO channel_memory (channel_id, scope, content, author) "
                    "VALUES (?, 'channel', ?, ?)",
                    (channel_id, content, marker_author),
                )
                await db.execute("COMMIT")
                return cur.lastrowid
            except Exception:
                await db.execute("ROLLBACK")
                raise

    async def reconcile_channel_memory_from_textarea_async(
        self, channel_id: str, seed: list, lines: list, author: str, max_rows: int
    ) -> dict:
        """Reconcile channel-scope memory against an edited textarea in ONE atomic transaction.

        The settings modal renders channel memory as a multiline textarea (one note per line). On
        submit the handler passes:
          - ``seed``: the ``[memory_id, content_hash]`` pairs captured at modal-OPEN (channel
            scope, non-blank rows only) — the exact snapshot the user edited.
          - ``lines``: the submitted textarea lines, already ``normalize_memory_line``-d with
            blanks dropped, deduped, order preserved (re-normalized defensively here).

        Keep / delete / add, all inside one ``BEGIN IMMEDIATE`` transaction (mirrors
        ``upsert_channel_pref_memory``'s autocommit-plus-explicit-txn style) so a concurrent modal
        save serializes and a partial failure rolls back — never a half-applied edit:
          - KEEP a seed whose hash still appears in the textarea → untouched (author preserved).
          - DELETE a seed whose hash left the textarea, but ONLY if the row still exists AND its
            current content still hashes to the seed hash (unchanged since open). If it changed
            since open, count a ``conflict`` and leave it (never clobber a concurrent edit). If it
            is already gone, silently skip (never resurrect a row deleted elsewhere).
          - ADD each textarea line matching NO seed hash AND no surviving channel row's content
            (dedup vs unseeded rows). ``max_rows`` counts ALL remaining channel rows (current −
            deletes + adds-so-far); overflow lines are skipped and counted in ``over_cap``.

        Returns ``{'deleted': [ids], 'added': [contents], 'conflicts': int, 'over_cap': int}``.
        Never raises on empty seed/lines: a fully-blanked box (``lines=[]``) deletes every
        still-unchanged seeded row and adds nothing.
        """
        result: Dict[str, Any] = {"deleted": [], "added": [], "conflicts": 0, "over_cap": 0}
        if not channel_id:
            return result

        # Defensive re-normalize + dedup by hash, order preserved (the handler already did this;
        # a second pass keeps the method correct when called directly, e.g. from tests).
        norm_lines: List[str] = []
        line_hashes: set = set()
        for raw in (lines or []):
            n = normalize_memory_line(raw)
            if not n:
                continue
            h = memory_content_hash(n)
            if h in line_hashes:
                continue
            line_hashes.add(h)
            norm_lines.append(n)

        cap = max(1, int(max_rows)) if max_rows is not None else None

        async with aiosqlite.connect(self.db_path, isolation_level=None) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA busy_timeout=5000")
            await db.execute("BEGIN IMMEDIATE")
            try:
                # 1. Snapshot all current channel-scope rows → {id: content}.
                async with db.execute(
                    "SELECT id, content FROM channel_memory "
                    "WHERE scope = 'channel' AND channel_id = ?",
                    (channel_id,),
                ) as cur:
                    current: Dict[Any, str] = {
                        row["id"]: row["content"] for row in await cur.fetchall()
                    }

                # 2-4. Walk the seed once: keep, delete-if-unchanged, or record a conflict.
                deleted_ids: List[Any] = []
                seed_hashes: set = set()
                for entry in (seed or []):
                    try:
                        mem_id, seed_hash = entry[0], entry[1]
                    except (TypeError, IndexError, KeyError):
                        continue
                    seed_hashes.add(seed_hash)
                    if seed_hash in line_hashes:
                        continue  # KEEP — still present in the textarea, leave untouched.
                    cur_content = current.get(mem_id)
                    if cur_content is None:
                        continue  # Already deleted elsewhere — nothing to do, no conflict.
                    if memory_content_hash(cur_content) == seed_hash:
                        await db.execute(
                            "DELETE FROM channel_memory WHERE id = ?", (mem_id,))
                        deleted_ids.append(mem_id)
                        current.pop(mem_id, None)
                    else:
                        # Changed elsewhere since open — never clobber the concurrent edit.
                        result["conflicts"] += 1

                # 5. ADD. Dedup a new line against every seed hash (its KEEP row already covers it)
                #    and against the content of every channel row that survived deletion.
                surviving_hashes: set = {
                    memory_content_hash(c) for c in current.values()
                }
                remaining = len(current)
                added: List[str] = []
                for n in norm_lines:
                    h = memory_content_hash(n)
                    if h in seed_hashes or h in surviving_hashes:
                        continue
                    if cap is not None and remaining >= cap:
                        result["over_cap"] += 1
                        continue
                    await db.execute(
                        "INSERT INTO channel_memory (channel_id, scope, content, author) "
                        "VALUES (?, 'channel', ?, ?)",
                        (channel_id, n, author),
                    )
                    added.append(n)
                    surviving_hashes.add(h)
                    remaining += 1

                await db.execute("COMMIT")
                result["deleted"] = deleted_ids
                result["added"] = added
                return result
            except Exception:
                await db.execute("ROLLBACK")
                raise

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

            # Create new user
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

    async def get_all_users_async(self) -> list:
        """F29: all persisted user_info rows (user_id/username/real_name/email/tz), for
        resolving a name → id when lookup_user is called with a name rather than a Slack id.
        Read-only; returns a list of dicts (empty on any failure)."""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                await db.execute("PRAGMA journal_mode=WAL")
                async with db.execute(
                    "SELECT user_id, username, real_name, email, timezone, tz_label "
                    "FROM users"
                ) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(r) for r in rows]
        except Exception as e:
            logger.debug(f"DB: get_all_users_async failed: {e}")
            return []

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

    # ---------------------------------------------------------------- F51 ambient artifacts
    #
    # Channel + source-ts keyed derivations for ambiently-seen images/links/files. Slack stays
    # the only transcript: these hold summaries + refs ONLY. Reuse is same-channel by design —
    # cross-channel reuse could leak private-channel/DM-derived content elsewhere.

    async def insert_pending_ambient_artifact(
        self, *, channel_id: str, source_ts: str, conversation_ts: str, kind: str, ref: str,
        content_type: Optional[str] = None, derivation_source: Optional[str] = None,
        expires_at: Optional[str] = None,
    ) -> Optional[Dict]:
        """Claim (or observe) an ambient artifact occurrence. Idempotent by
        (channel_id, source_ts, kind, ref): a re-offer of the same occurrence does NOT clobber an
        existing row (singleflight — a ready summary survives). Returns the row as it stands AFTER
        the call (dict), so the caller can see whether it is already `ready`/`pending`/etc."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("""
                INSERT INTO ambient_artifacts
                    (channel_id, source_ts, conversation_ts, kind, ref, status,
                     content_type, derivation_source, expires_at)
                VALUES (?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                ON CONFLICT(channel_id, source_ts, kind, ref) DO NOTHING
            """, (channel_id, source_ts, conversation_ts, kind, ref,
                  content_type, derivation_source, expires_at))
            await db.commit()
            async with db.execute("""
                SELECT * FROM ambient_artifacts
                WHERE channel_id = ? AND source_ts = ? AND kind = ? AND ref = ?
            """, (channel_id, source_ts, kind, ref)) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def set_ambient_artifact_ready(
        self, *, channel_id: str, source_ts: str, kind: str, ref: str,
        title: Optional[str], summary: str, model: Optional[str],
        derivation_source: str, content_type: Optional[str] = None,
        expires_at: Optional[str] = None,
    ) -> None:
        """Mark an artifact ready with its derived summary. Only writes a row that exists (the
        pending occurrence was claimed first) — status flips to `ready`, fetched_at stamped."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("""
                UPDATE ambient_artifacts
                SET status = 'ready', title = ?, summary = ?, model = ?,
                    derivation_source = ?, content_type = COALESCE(?, content_type),
                    error_code = NULL, updated_at = CURRENT_TIMESTAMP,
                    fetched_at = CURRENT_TIMESTAMP, expires_at = COALESCE(?, expires_at)
                WHERE channel_id = ? AND source_ts = ? AND kind = ? AND ref = ?
            """, (title, summary, model, derivation_source, content_type, expires_at,
                  channel_id, source_ts, kind, ref))
            await db.commit()

    async def set_ambient_artifact_status(
        self, *, channel_id: str, source_ts: str, kind: str, ref: str,
        status: str, error_code: Optional[str] = None, increment_attempt: bool = False,
        derivation_source: Optional[str] = None,
    ) -> None:
        """Persist a terminal/interim status (failed/blocked/omitted/pending) with an honest
        error_code — the house rule is no silent drops."""
        attempt_sql = "attempt_count = attempt_count + 1," if increment_attempt else ""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(f"""
                UPDATE ambient_artifacts
                SET status = ?, error_code = ?, {attempt_sql}
                    derivation_source = COALESCE(?, derivation_source),
                    updated_at = CURRENT_TIMESTAMP
                WHERE channel_id = ? AND source_ts = ? AND kind = ? AND ref = ?
            """, (status, error_code, derivation_source,
                  channel_id, source_ts, kind, ref))
            await db.commit()

    async def get_ambient_artifacts_for_messages(
        self, channel_id: str, source_ts_list: List[str],
        statuses: Optional[List[str]] = None,
    ) -> Dict[str, List[Dict]]:
        """ONE batched query (never N+1) mapping source_ts -> [artifact rows], for rendering a
        whole thread/page of history at once. Same-channel scoped. Deterministic order (id ASC)."""
        if not channel_id or not source_ts_list:
            return {}
        # De-dup and cap placeholders defensively.
        uniq = list(dict.fromkeys(str(t) for t in source_ts_list if t))
        if not uniq:
            return {}
        placeholders = ",".join("?" for _ in uniq)
        status_filter = ""
        params: List[Any] = [channel_id, *uniq]
        if statuses:
            status_filter = f" AND status IN ({','.join('?' for _ in statuses)})"
            params.extend(statuses)
        out: Dict[str, List[Dict]] = {}
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute(f"""
                SELECT * FROM ambient_artifacts
                WHERE channel_id = ? AND source_ts IN ({placeholders}){status_filter}
                ORDER BY source_ts ASC, id ASC
            """, params) as cursor:
                async for row in cursor:
                    r = dict(row)
                    out.setdefault(r["source_ts"], []).append(r)
        return out

    async def find_reusable_ambient_summary(
        self, channel_id: str, kind: str, ref: str, *, fresh_after: Optional[str] = None,
    ) -> Optional[Dict]:
        """A ready summary for the same ref IN THE SAME CHANNEL, optionally requiring
        fetched_at >= fresh_after (ISO/SQL datetime) so a stale link re-fetches. Newest first."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            query = ("SELECT * FROM ambient_artifacts WHERE channel_id = ? AND kind = ? "
                     "AND ref = ? AND status = 'ready' AND summary IS NOT NULL")
            params: List[Any] = [channel_id, kind, ref]
            if fresh_after:
                query += " AND (fetched_at IS NULL OR fetched_at >= ?)"
                params.append(fresh_after)
            query += " ORDER BY fetched_at DESC, id DESC LIMIT 1"
            async with db.execute(query, params) as cursor:
                row = await cursor.fetchone()
                return dict(row) if row else None

    async def delete_ambient_artifacts_by_source(self, channel_id: str, source_ts: str) -> int:
        """Purge all artifacts for a (channel, source message) — message_deleted/edit lifecycle.

        ALSO purges the ambient image analyses the vision worker dual-wrote into `images`
        (marked metadata `{"ambient": true}`, message_ts == source_ts, thread_id under this
        channel). Without this the deleted/edited image's description survives in the ledger and
        keeps being injected — the exact leak the retention/deletion path is meant to close."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            cursor = await db.execute(
                "DELETE FROM ambient_artifacts WHERE channel_id = ? AND source_ts = ?",
                (channel_id, source_ts))
            # Exact structured match: the dual-written ambient row carries metadata
            # {"ambient":true,"channel_id":...}. json_extract avoids matching {"ambient":false}
            # or {"description":"ambient"}; json_valid guards any legacy non-JSON metadata.
            await db.execute(
                "DELETE FROM images WHERE message_ts = ? AND metadata_json IS NOT NULL "
                "AND json_valid(metadata_json) "
                "AND json_extract(metadata_json, '$.ambient') = 1 "
                "AND json_extract(metadata_json, '$.channel_id') = ?",
                (source_ts, channel_id))
            # F51c: a late-artifact addendum for this source message must die with it, or a
            # deleted/edited message's derived note keeps riding the summary head.
            await db.execute(
                "DELETE FROM thread_summary_addenda WHERE channel_id = ? AND source_ts = ?",
                (channel_id, source_ts))
            await db.commit()
            return cursor.rowcount or 0

    async def delete_ambient_artifacts_by_ref(self, channel_id: str, kind: str, ref: str) -> int:
        """Purge artifacts for a specific ref (file_deleted lifecycle — a Slack file removed)."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            cursor = await db.execute(
                "DELETE FROM ambient_artifacts WHERE channel_id = ? AND kind = ? AND ref = ?",
                (channel_id, kind, ref))
            await db.execute(
                "DELETE FROM thread_summary_addenda WHERE channel_id = ? AND kind = ? AND ref = ?",
                (channel_id, kind, ref))
            await db.commit()
            return cursor.rowcount or 0

    async def delete_ambient_artifacts_by_file_id(self, file_id: str) -> int:
        """Purge image/file artifacts derived from a Slack file id, workspace-wide (file_deleted).
        A file id is globally unique, so no channel scope is needed."""
        if not file_id:
            return 0
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            cursor = await db.execute(
                "DELETE FROM ambient_artifacts WHERE ref = ? AND kind IN ('image','file')",
                (file_id,))
            # Exact structured match on the file id stored in metadata — NOT a url substring LIKE
            # (an id that is a substring of another url would cross-delete the wrong image).
            await db.execute(
                "DELETE FROM images WHERE metadata_json IS NOT NULL AND json_valid(metadata_json) "
                "AND json_extract(metadata_json, '$.ambient') = 1 "
                "AND json_extract(metadata_json, '$.file_id') = ?",
                (file_id,))
            # F51c: file id is the addendum ref for image/file kinds — purge those too.
            await db.execute(
                "DELETE FROM thread_summary_addenda WHERE ref = ? AND kind IN ('image','file')",
                (file_id,))
            await db.commit()
            return cursor.rowcount or 0

    async def get_pending_ambient_artifacts(self, limit: int = 200) -> List[Dict]:
        """Rows still `pending` — interrupted work to resume on restart. Oldest first."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute(
                "SELECT * FROM ambient_artifacts WHERE status = 'pending' "
                "ORDER BY created_at ASC LIMIT ?", (int(limit),)) as cursor:
                return [dict(row) async for row in cursor]

    def delete_expired_ambient_artifacts(self, days: int = 30) -> List[str]:
        """Retention sweep (sync, mirrors delete_old_tool_usage) — wired into the scheduled
        cleanup worker. Uses expires_at when set, else falls back to a created_at age cutoff.
        Cutoffs computed IN SQL (UTC) like the other sweeps, never a local Python datetime.

        Returns the DISTINCT thread keys (`channel_id:thread_ts`) whose late-artifact addenda were
        deleted. F51d: the sweep clears the addenda from the DB, but an ACTIVE warm thread still
        holds an in-memory summary head carrying the expired note and would keep sending it
        indefinitely; the cleanup worker marks each returned thread for refresh so its next turn
        rebuilds without the note."""
        # F51c: retire the aged artifacts' late-artifact addenda in the SAME operation. A row's
        # derived note otherwise lingers indefinitely in the summary head after its artifact ages
        # out — and keeps occupying one of the per-thread addenda cap slots. Match on the artifact
        # identity (channel_id + source_ts + kind + ref); the row-value IN subquery uses the SAME
        # cutoff predicate and MUST run BEFORE the artifacts are deleted (afterwards the subquery
        # would find nothing to match).
        # Capture the affected thread keys BEFORE the delete — the same identity subquery finds
        # nothing once the artifacts are gone. thread_summary_addenda.thread_id is already stored
        # as the full `channel_id:thread_ts` key, so it is the mark_needs_refresh key verbatim.
        affected = self.conn.execute("""
            SELECT DISTINCT thread_id FROM thread_summary_addenda
            WHERE (channel_id, source_ts, kind, ref) IN (
                SELECT channel_id, source_ts, kind, ref FROM ambient_artifacts
                WHERE (expires_at IS NOT NULL AND expires_at < datetime('now'))
                   OR (expires_at IS NULL AND created_at < datetime('now', ?))
            )
        """, (f"-{int(days)} days",)).fetchall()
        affected_thread_keys = [row[0] for row in affected]
        self.conn.execute("""
            DELETE FROM thread_summary_addenda
            WHERE (channel_id, source_ts, kind, ref) IN (
                SELECT channel_id, source_ts, kind, ref FROM ambient_artifacts
                WHERE (expires_at IS NOT NULL AND expires_at < datetime('now'))
                   OR (expires_at IS NULL AND created_at < datetime('now', ?))
            )
        """, (f"-{int(days)} days",))
        cursor = self.conn.execute("""
            DELETE FROM ambient_artifacts
            WHERE (expires_at IS NOT NULL AND expires_at < datetime('now'))
               OR (expires_at IS NULL AND created_at < datetime('now', ?))
        """, (f"-{int(days)} days",))
        # Retention must also reach the dual-written ambient image analyses (metadata
        # `{"ambient": true}`) — they have no expires_at column, so age them by created_at with
        # the same window. Addressed uploads (no ambient marker) are untouched.
        img_cursor = self.conn.execute("""
            DELETE FROM images
            WHERE metadata_json IS NOT NULL AND json_valid(metadata_json)
              AND json_extract(metadata_json, '$.ambient') = 1
              AND created_at < datetime('now', ?)
        """, (f"-{int(days)} days",))
        if cursor.rowcount > 0 or img_cursor.rowcount > 0:
            self.log_info(f"DB: Cleaned up {cursor.rowcount} ambient artifacts + "
                          f"{img_cursor.rowcount} ambient image analyses (retention {days}d)")
        return affected_thread_keys

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

    async def get_channel_documents_async(self, channel_id: str) -> List[Dict]:
        """All documents shared anywhere in a channel (F22 channel-wide access).

        thread_id is stored as "channel:thread"; a channel's documents are every row
        whose thread_id starts with ``channel_id + ':'``. Channel ids are alphanumeric
        (no LIKE metacharacters), so a plain prefix LIKE is safe and cannot escape the
        channel — the privacy boundary is same-channel-only. Same row shape and
        created_at ASC ordering as get_thread_documents_async."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")
            async with db.execute(
                "SELECT * FROM documents WHERE thread_id LIKE ? ORDER BY created_at ASC",
                (f"{channel_id}:%",),
            ) as cursor:
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

    # F32 container async wrappers. asyncio.to_thread over the sync methods, matching
    # get_or_create_thread_async — these run on the message path, so they must not block
    # the event loop on SQLite.

    async def get_thread_container_async(self, thread_id: str) -> Optional[Dict]:
        return await asyncio.to_thread(self.get_thread_container, thread_id)

    async def get_fresh_thread_container_async(self, thread_id: str,
                                               reuse_minutes: int) -> Optional[Dict]:
        return await asyncio.to_thread(self.get_fresh_thread_container, thread_id, reuse_minutes)

    async def save_thread_container_async(self, thread_id: str, container_id: str):
        return await asyncio.to_thread(self.save_thread_container, thread_id, container_id)

    async def touch_thread_container_async(self, thread_id: str, container_id: str):
        return await asyncio.to_thread(self.touch_thread_container, thread_id, container_id)

    async def add_published_container_files_async(self, thread_id: str, container_id: str,
                                                  file_ids: List[str]):
        return await asyncio.to_thread(
            self.add_published_container_files, thread_id, container_id, file_ids)

    async def delete_thread_container_async(self, thread_id: str,
                                            container_id: Optional[str] = None):
        return await asyncio.to_thread(self.delete_thread_container, thread_id, container_id)

    async def get_expired_thread_containers_async(self, older_than_minutes: int) -> List[Dict]:
        return await asyncio.to_thread(self.get_expired_thread_containers, older_than_minutes)

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