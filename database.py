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
        
        # Documents table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                content TEXT NOT NULL,
                page_structure TEXT,
                total_pages INTEGER,
                summary TEXT,
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
                model TEXT DEFAULT 'gpt-5',
                reasoning_effort TEXT DEFAULT 'low',
                verbosity TEXT DEFAULT 'low',
                temperature REAL DEFAULT 0.8,
                top_p REAL DEFAULT 1.0,
                
                -- Feature toggles
                enable_web_search BOOLEAN DEFAULT 1,
                enable_streaming BOOLEAN DEFAULT 1,
                
                -- Image settings
                image_size TEXT DEFAULT '1024x1024',
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

        self.conn.commit()

        # Run migrations for existing databases
        self._run_migrations()
    
    def _run_migrations(self):
        """Run database migrations to update schema for existing databases."""
        try:
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
                        model TEXT DEFAULT 'gpt-5',
                        reasoning_effort TEXT DEFAULT 'low',
                        verbosity TEXT DEFAULT 'low',
                        temperature REAL DEFAULT 0.8,
                        top_p REAL DEFAULT 1.0,
                        
                        -- Feature toggles
                        enable_web_search BOOLEAN DEFAULT 1,
                        enable_streaming BOOLEAN DEFAULT 1,
                        
                        -- Image settings
                        image_size TEXT DEFAULT '1024x1024',
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
        except Exception as e:
            self.log_error(f"DB: Migration error: {e}", exc_info=True)
    
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
    
    def delete_oldest_messages(self, thread_id: str, count: int):
        """
        Delete the oldest N messages from a thread, preserving system messages.
        
        Args:
            thread_id: Thread identifier
            count: Number of messages to delete
        """
        # First get the IDs of non-system messages ordered by timestamp
        cursor = self.conn.execute("""
            SELECT id FROM messages 
            WHERE thread_id = ? AND role NOT IN ('system', 'developer')
            ORDER BY timestamp ASC
            LIMIT ?
        """, (thread_id, count))
        
        ids_to_delete = [row['id'] for row in cursor]
        
        if ids_to_delete:
            placeholders = ','.join(['?' for _ in ids_to_delete])
            self.conn.execute(
                f"DELETE FROM messages WHERE id IN ({placeholders})",
                ids_to_delete
            )
            logger.debug(f"Deleted {len(ids_to_delete)} oldest messages from thread {thread_id}")
    
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
            self.conn.execute("""
                INSERT OR REPLACE INTO images 
                (thread_id, url, image_type, prompt, analysis, original_analysis, metadata_json, message_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
            'image_size': config.default_image_size,
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
                 image_size, input_fidelity, vision_detail, settings_completed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, email, defaults['model'],
                defaults['reasoning_effort'], defaults['verbosity'],
                defaults['temperature'], defaults['top_p'],
                1 if defaults['enable_web_search'] else 0,
                1 if defaults['enable_streaming'] else 0,
                defaults['image_size'], defaults['input_fidelity'],
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
                         'top_p', 'image_size', 'input_fidelity', 'vision_detail',
                         'slack_email', 'settings_completed', 'custom_instructions']:
                if field in preferences:
                    updates.append(f"{field} = ?")
                    values.append(preferences[field])
            
            # Handle boolean fields
            for field in ['enable_web_search', 'enable_streaming']:
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
                     content: str, page_structure: Optional[Dict] = None,
                     total_pages: Optional[int] = None, summary: Optional[str] = None,
                     metadata: Optional[Dict] = None, message_ts: Optional[str] = None):
        """
        Save document content and metadata.
        
        Args:
            thread_id: Thread identifier
            filename: Original filename
            mime_type: Document MIME type
            content: Full document text content
            page_structure: Optional page/sheet structure info as dict
            total_pages: Total page/sheet count
            summary: Optional AI-generated summary
            metadata: Additional metadata (size, author, etc.)
            message_ts: Message timestamp to link document to specific message
        """
        self.log_debug(f"DB: Saving document - thread={thread_id}, filename={filename}, "
                      f"content_len={len(content) if content else 0}, pages={total_pages}")
        
        try:
            self.conn.execute("""
                INSERT INTO documents 
                (thread_id, filename, mime_type, content, page_structure, total_pages, 
                 summary, metadata_json, message_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (thread_id, filename, mime_type, content,
                  json.dumps(page_structure) if page_structure else None,
                  total_pages, summary,
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

    def backup_database(self):
        """Create timestamped backup of database."""
        # Checkpoint WAL file before backup
        self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self.db_dir}/backups/{self.platform}_{timestamp}.db"
        
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

    async def cache_message_async(self, thread_id: str, role: str, content: str,
                                 message_ts: Optional[str] = None, metadata: Optional[Dict] = None):
        """
        Async version of cache_message.

        Args:
            thread_id: Thread identifier
            role: Message role (user/assistant/developer)
            content: Message content
            message_ts: Optional message timestamp
            metadata: Optional metadata dictionary
        """
        self.log_debug(f"DB: Async caching message - thread={thread_id}, role={role}, "
                      f"content_len={len(content) if content else 0}, has_ts={bool(message_ts)}")

        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                await db.execute("PRAGMA journal_mode=WAL")

                await db.execute("""
                    INSERT INTO messages (thread_id, role, content, message_ts, metadata_json)
                    VALUES (?, ?, ?, ?, ?)
                """, (thread_id, role, content, message_ts,
                      json.dumps(metadata) if metadata else None))

                await db.commit()

                # Update thread activity
                await self.update_thread_activity_async(thread_id)

                self.log_info(f"DB: Successfully cached {role} message for thread {thread_id}")

        except Exception as e:
            self.log_error(f"DB: Failed to cache message async - {e}", exc_info=True)
            raise

    async def get_cached_messages_async(self, thread_id: str, limit: Optional[int] = None) -> List[Dict]:
        """
        Async version of get_cached_messages.

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

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            await db.execute("PRAGMA journal_mode=WAL")

            async with db.execute(query, (thread_id,)) as cursor:
                messages = []
                async for row in cursor:
                    msg = dict(row)
                    if msg.get("metadata_json"):
                        msg["metadata"] = json.loads(msg["metadata_json"])
                        del msg["metadata_json"]
                    messages.append(msg)

        return messages

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

                await db.execute("""
                    INSERT OR REPLACE INTO images
                    (thread_id, url, image_type, prompt, analysis, original_analysis, metadata_json, message_ts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
                    image_size, input_fidelity, vision_detail
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, email,
                config.gpt_model, config.default_temperature, config.default_top_p,
                1 if config.enable_web_search else 0,
                1 if config.enable_streaming else 0,
                config.default_reasoning_effort, config.default_verbosity,
                config.default_image_size, config.default_input_fidelity, config.default_detail_level
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
                    prefs['settings_completed'] = bool(prefs.get('settings_completed', 0))
                    return prefs
                return {}

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
                         'top_p', 'image_size', 'input_fidelity', 'vision_detail',
                         'slack_email', 'settings_completed', 'custom_instructions']:
                if field in preferences:
                    update_fields.append(f"{field} = ?")
                    values.append(preferences[field])

            # Handle boolean fields - convert to integers for SQLite
            for field in ['enable_web_search', 'enable_streaming']:
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
                        'enable_streaming': user_prefs.get('enable_streaming')
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

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info(f"Database connection closed for {self.platform}")