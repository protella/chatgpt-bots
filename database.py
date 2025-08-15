"""
SQLite Database Manager for ChatGPT Bots
Provides persistent storage for threads, messages, images, and user preferences
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
import logging
from logger import LoggerMixin

logger = logging.getLogger(__name__)


class DatabaseManager(LoggerMixin):
    """
    Manages SQLite database operations for bot persistence.
    Each platform gets its own database file.
    """
    
    def __init__(self, platform: str = "slack"):
        """
        Initialize database connection for the specified platform.
        
        Args:
            platform: Platform name (slack, discord, etc.)
        """
        self.platform = platform
        
        # Ensure directories exist
        os.makedirs("data", exist_ok=True)
        os.makedirs("data/backups", exist_ok=True)
        
        # Connect to platform-specific database
        self.db_path = f"data/{platform}.db"
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
        
        # Messages table (for caching)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message_ts TEXT,
                metadata_json TEXT,
                FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
            )
        """)
        
        # Create index for messages
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_thread_messages 
            ON messages(thread_id, timestamp)
        """)
        
        # Images table (no base64 storage)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                url TEXT NOT NULL UNIQUE,
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
        
        # Users table with timezone support
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT,
                config_json TEXT,
                timezone TEXT DEFAULT 'UTC',
                tz_label TEXT,
                tz_offset INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
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
    
    def cache_message(self, thread_id: str, role: str, content: str, 
                     message_ts: Optional[str] = None, metadata: Optional[Dict] = None):
        """
        Cache a message for a thread.
        
        Args:
            thread_id: Thread identifier
            role: Message role (user/assistant/developer)
            content: Message content
            message_ts: Optional message timestamp
            metadata: Optional metadata dictionary
        """
        self.log_debug(f"DB: Caching message - thread={thread_id}, role={role}, "
                      f"content_len={len(content) if content else 0}, has_ts={bool(message_ts)}")
        
        try:
            self.conn.execute("""
                INSERT INTO messages (thread_id, role, content, message_ts, metadata_json)
                VALUES (?, ?, ?, ?, ?)
            """, (thread_id, role, content, message_ts, 
                  json.dumps(metadata) if metadata else None))
            
            # Update thread activity
            self.update_thread_activity(thread_id)
            
            self.log_info(f"DB: Successfully cached {role} message for thread {thread_id}")
            
        except Exception as e:
            self.log_error(f"DB: Failed to cache message - {e}", exc_info=True)
            raise
    
    def get_cached_messages(self, thread_id: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Get cached messages for a thread.
        
        Args:
            thread_id: Thread identifier
            limit: Optional limit on number of messages (None = all)
            
        Returns:
            List of message dictionaries
        """
        query = """
            SELECT * FROM messages 
            WHERE thread_id = ? 
            ORDER BY timestamp ASC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        cursor = self.conn.execute(query, (thread_id,))
        messages = []
        
        for row in cursor:
            msg = dict(row)
            if msg.get("metadata_json"):
                msg["metadata"] = json.loads(msg["metadata_json"])
                del msg["metadata_json"]
            messages.append(msg)
        
        return messages
    
    def clear_thread_messages(self, thread_id: str):
        """
        Clear all cached messages for a thread.
        
        Args:
            thread_id: Thread identifier
        """
        self.conn.execute(
            "DELETE FROM messages WHERE thread_id = ?",
            (thread_id,)
        )
        logger.debug(f"Cleared messages for thread {thread_id}")
    
    # Image operations
    
    def save_image_metadata(self, thread_id: str, url: str, image_type: str,
                           prompt: Optional[str] = None, analysis: Optional[str] = None,
                           original_analysis: Optional[str] = None, metadata: Optional[Dict] = None):
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
        """
        self.log_debug(f"DB: Saving image - thread={thread_id}, url={url[:100]}, "
                      f"type={image_type}, has_analysis={bool(analysis)}, "
                      f"analysis_len={len(analysis) if analysis else 0}, "
                      f"prompt_len={len(prompt) if prompt else 0}")
        
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO images 
                (thread_id, url, image_type, prompt, analysis, original_analysis, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (thread_id, url, image_type, prompt, analysis, original_analysis,
                  json.dumps(metadata) if metadata else None))
            
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
                ORDER BY created_at DESC
            """, (thread_id, image_type))
        else:
            cursor = self.conn.execute("""
                SELECT * FROM images 
                WHERE thread_id = ?
                ORDER BY created_at DESC
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
            "image_size": config.default_image_size,
            "image_quality": config.default_image_quality,
            "image_style": config.default_image_style,
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
    
    def save_user_timezone(self, user_id: str, timezone: str, 
                          tz_label: Optional[str] = None, tz_offset: Optional[int] = None):
        """
        Save user timezone information.
        
        Args:
            user_id: User identifier
            timezone: Timezone string
            tz_label: Timezone label
            tz_offset: Offset in seconds from UTC
        """
        self.conn.execute("""
            UPDATE users 
            SET timezone = ?, tz_label = ?, tz_offset = ?, last_seen = CURRENT_TIMESTAMP
            WHERE user_id = ?
        """, (timezone, tz_label, tz_offset, user_id))
        
        logger.debug(f"Saved timezone for user {user_id}: {timezone}")
    
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
            "image_size": bot_config.default_image_size,
            "image_quality": bot_config.default_image_quality,
            "image_style": bot_config.default_image_style,
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
    
    def backup_database(self):
        """Create timestamped backup of database."""
        # Checkpoint WAL file before backup
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        
        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"data/backups/{self.platform}_{timestamp}.db"
        
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
        
        for filename in os.listdir("data/backups"):
            if filename.startswith(f"{self.platform}_") and filename.endswith(".db"):
                try:
                    # Parse timestamp from filename
                    parts = filename.replace(".db", "").split("_")
                    if len(parts) >= 3:
                        date_str = parts[-2] + parts[-1]
                        file_date = datetime.strptime(date_str, "%Y%m%d%H%M%S")
                        
                        if file_date < cutoff:
                            os.remove(f"data/backups/{filename}")
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
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info(f"Database connection closed for {self.platform}")