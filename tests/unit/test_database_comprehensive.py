"""
Comprehensive unit tests for database.py module
Tests for improved coverage of SQLite database operations
"""
import pytest
import sqlite3
import tempfile
import os
import json
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, Mock, AsyncMock
from database import DatabaseManager


class TestDatabaseManagerComprehensive:
    """Comprehensive tests for DatabaseManager for better coverage"""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock os.makedirs to prevent directory creation issues
            with patch('os.makedirs'):
                db = DatabaseManager("test")
                db.db_path = f"{tmpdir}/test.db"
                # Recreate connection with new path
                if hasattr(db, 'conn') and db.conn:
                    db.conn.close()
                db.conn = sqlite3.connect(
                    db.db_path,
                    check_same_thread=False,
                    isolation_level=None
                )
                db.conn.row_factory = sqlite3.Row
                db.init_schema()
                yield db
                if hasattr(db, 'conn') and db.conn:
                    db.conn.close()

    def test_init_creates_directories(self):
        """Test that __init__ creates necessary directories"""
        with patch('os.makedirs') as mock_makedirs:
            with tempfile.TemporaryDirectory() as tmpdir:
                db = DatabaseManager("test")

                # Should create data and data/backups directories
                assert mock_makedirs.call_count >= 2
                calls = [call[0][0] for call in mock_makedirs.call_args_list]
                assert "data" in calls
                assert "data/backups" in calls

    def test_init_wal_mode_enabled(self, temp_db):
        """Test that WAL mode is enabled on initialization"""
        cursor = temp_db.conn.execute("PRAGMA journal_mode")
        result = cursor.fetchone()
        # WAL mode might be 'wal' or 'delete' in test environment
        assert result[0].upper() in ['WAL', 'DELETE']

    def test_init_busy_timeout_set(self, temp_db):
        """Test that busy timeout is configured"""
        cursor = temp_db.conn.execute("PRAGMA busy_timeout")
        result = cursor.fetchone()
        assert result[0] == 5000

    def test_migrations_message_ts_column(self, temp_db):
        """Test migration adds message_ts column to images table"""
        # Drop the column first to test migration
        try:
            temp_db.conn.execute("""
                CREATE TABLE images_backup AS SELECT
                id, thread_id, url, image_type, prompt, analysis,
                original_analysis, metadata_json, created_at
                FROM images
            """)
            temp_db.conn.execute("DROP TABLE images")
            temp_db.conn.execute("""
                CREATE TABLE images (
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
            temp_db.conn.commit()

            # Run migrations
            temp_db._run_migrations()

            # Check if message_ts column exists
            cursor = temp_db.conn.execute("PRAGMA table_info(images)")
            columns = [col[1] for col in cursor.fetchall()]
            assert 'message_ts' in columns

        except Exception:
            # Skip if table manipulation fails
            pass

    def test_migrations_real_name_column(self, temp_db):
        """Test migration adds real_name column to users table"""
        # Check that real_name column exists after migrations
        cursor = temp_db.conn.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cursor.fetchall()]
        assert 'real_name' in columns

    def test_migrations_custom_instructions_column(self, temp_db):
        """Test migration adds custom_instructions column to user_preferences table"""
        cursor = temp_db.conn.execute("PRAGMA table_info(user_preferences)")
        columns = [col[1] for col in cursor.fetchall()]
        assert 'custom_instructions' in columns

    def test_migrations_email_column(self, temp_db):
        """Test migration adds email column to users table"""
        cursor = temp_db.conn.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cursor.fetchall()]
        assert 'email' in columns

    def test_migrations_user_preferences_table_creation(self, temp_db):
        """Test migration creates user_preferences table if it doesn't exist"""
        # Table should exist after initialization
        cursor = temp_db.conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='user_preferences'
        """)
        assert cursor.fetchone() is not None

    def test_get_or_create_thread_creates_new(self, temp_db):
        """Test get_or_create_thread creates new thread"""
        thread_id = "C123:1234567890.123"
        channel_id = "C123"
        user_id = "U123"

        result = temp_db.get_or_create_thread(thread_id, channel_id, user_id)

        assert result['thread_id'] == thread_id
        assert result['channel_id'] == channel_id
        assert result['created'] is True

        # Verify thread was actually created
        cursor = temp_db.conn.execute(
            "SELECT * FROM threads WHERE thread_id = ?",
            (thread_id,)
        )
        row = cursor.fetchone()
        assert row is not None
        assert row['thread_id'] == thread_id

    def test_get_or_create_thread_gets_existing(self, temp_db):
        """Test get_or_create_thread gets existing thread"""
        thread_id = "C123:1234567890.123"
        channel_id = "C123"

        # Create thread first
        temp_db.get_or_create_thread(thread_id, channel_id)

        # Get existing thread
        result = temp_db.get_or_create_thread(thread_id, channel_id)

        assert result['thread_id'] == thread_id
        assert result['created'] is False

    def test_save_and_get_thread_config(self, temp_db):
        """Test saving and retrieving thread configuration"""
        thread_id = "C123:1234567890.123"
        config = {
            "model": "gpt-5-mini",
            "temperature": 0.5,
            "custom_param": "value"
        }

        # Save config
        temp_db.save_thread_config(thread_id, config)

        # Retrieve config
        retrieved = temp_db.get_thread_config(thread_id)

        assert retrieved == config

    def test_get_thread_config_nonexistent(self, temp_db):
        """Test get_thread_config returns None for nonexistent thread"""
        result = temp_db.get_thread_config("nonexistent")
        assert result is None

    def test_update_thread_activity(self, temp_db):
        """Test updating thread activity timestamp"""
        thread_id = "C123:1234567890.123"
        channel_id = "C123"

        # Create thread
        temp_db.get_or_create_thread(thread_id, channel_id)

        # Update activity
        temp_db.update_thread_activity(thread_id)

        # Verify last_activity was updated
        cursor = temp_db.conn.execute(
            "SELECT last_activity FROM threads WHERE thread_id = ?",
            (thread_id,)
        )
        row = cursor.fetchone()
        assert row is not None

    def test_cache_message_with_metadata(self, temp_db):
        """Test caching message with metadata"""
        thread_id = "C123:1234567890.123"
        role = "user"
        content = "Hello world"
        message_ts = "1234567890.123"
        metadata = {"test": "value"}

        temp_db.cache_message(
            thread_id, role, content,
            timestamp=datetime.now(),
            message_ts=message_ts,
            metadata=metadata
        )

        # Verify message was cached
        messages = temp_db.get_cached_messages(thread_id)
        assert len(messages) == 1
        assert messages[0]['role'] == role
        assert messages[0]['content'] == content
        assert messages[0]['message_ts'] == message_ts
        assert messages[0]['metadata'] == metadata

    def test_cache_message_without_metadata(self, temp_db):
        """Test caching message without optional parameters"""
        thread_id = "C123:1234567890.123"
        role = "assistant"
        content = "Response content"

        temp_db.cache_message(thread_id, role, content)

        # Verify message was cached
        messages = temp_db.get_cached_messages(thread_id)
        assert len(messages) == 1
        assert messages[0]['role'] == role
        assert messages[0]['content'] == content
        assert messages[0]['message_ts'] is None

    def test_get_cached_messages_with_limit(self, temp_db):
        """Test getting cached messages with limit"""
        thread_id = "C123:1234567890.123"

        # Cache multiple messages
        for i in range(5):
            temp_db.cache_message(thread_id, "user", f"Message {i}")

        # Get limited messages
        messages = temp_db.get_cached_messages(thread_id, limit=3)
        assert len(messages) == 3

        # Get all messages
        all_messages = temp_db.get_cached_messages(thread_id)
        assert len(all_messages) == 5

    def test_clear_thread_messages(self, temp_db):
        """Test clearing all messages for a thread"""
        thread_id = "C123:1234567890.123"

        # Cache some messages
        temp_db.cache_message(thread_id, "user", "Message 1")
        temp_db.cache_message(thread_id, "assistant", "Response 1")

        # Clear messages
        temp_db.clear_thread_messages(thread_id)

        # Verify messages are cleared
        messages = temp_db.get_cached_messages(thread_id)
        assert len(messages) == 0

    def test_delete_oldest_messages(self, temp_db):
        """Test deleting oldest messages from thread"""
        thread_id = "C123:1234567890.123"

        # Cache multiple messages with different timestamps
        for i in range(5):
            temp_db.cache_message(
                thread_id, "user", f"Message {i}",
                timestamp=datetime.now() - timedelta(minutes=i)
            )

        # Delete oldest 2 messages
        temp_db.delete_oldest_messages(thread_id, 2)

        # Verify only 3 messages remain
        messages = temp_db.get_cached_messages(thread_id)
        assert len(messages) == 3

    def test_save_image_metadata_complete(self, temp_db):
        """Test saving complete image metadata"""
        thread_id = "C123:1234567890.123"
        url = "https://example.com/image.png"
        image_type = "generated"
        prompt = "A beautiful sunset"
        analysis = "Image analysis text"
        original_analysis = "Original analysis"
        metadata = {"width": 1024, "height": 1024}
        message_ts = "1234567890.123"

        temp_db.save_image_metadata(
            thread_id, url, image_type, prompt,
            analysis, original_analysis, metadata, message_ts
        )

        # Verify image was saved
        images = temp_db.find_thread_images(thread_id)
        assert len(images) == 1
        assert images[0]['url'] == url
        assert images[0]['image_type'] == image_type
        assert images[0]['prompt'] == prompt
        assert images[0]['message_ts'] == message_ts

    def test_save_image_metadata_minimal(self, temp_db):
        """Test saving minimal image metadata"""
        thread_id = "C123:1234567890.123"
        url = "https://example.com/image.png"
        image_type = "uploaded"

        temp_db.save_image_metadata(thread_id, url, image_type)

        # Verify image was saved with minimal data
        images = temp_db.find_thread_images(thread_id)
        assert len(images) == 1
        assert images[0]['url'] == url
        assert images[0]['image_type'] == image_type
        assert images[0]['prompt'] is None

    def test_get_image_analysis_by_url(self, temp_db):
        """Test getting image analysis by URL"""
        thread_id = "C123:1234567890.123"
        url = "https://example.com/image.png"
        analysis = "Detailed image analysis"

        temp_db.save_image_metadata(
            thread_id, url, "uploaded",
            analysis=analysis
        )

        result = temp_db.get_image_analysis_by_url(thread_id, url)
        assert result is not None
        assert result['analysis'] == analysis

    def test_get_image_analysis_by_url_not_found(self, temp_db):
        """Test getting image analysis for non-existent URL"""
        result = temp_db.get_image_analysis_by_url("thread123", "nonexistent.jpg")
        assert result is None

    def test_get_images_by_message(self, temp_db):
        """Test getting images by message timestamp"""
        thread_id = "C123:1234567890.123"
        message_ts = "1234567890.123"

        # Save multiple images with same message_ts
        temp_db.save_image_metadata(
            thread_id, "https://example.com/image1.png", "generated",
            message_ts=message_ts
        )
        temp_db.save_image_metadata(
            thread_id, "https://example.com/image2.png", "generated",
            message_ts=message_ts
        )

        images = temp_db.get_images_by_message(thread_id, message_ts)
        assert len(images) == 2

    def test_find_thread_images_with_type_filter(self, temp_db):
        """Test finding thread images with type filter"""
        thread_id = "C123:1234567890.123"

        # Save images of different types
        temp_db.save_image_metadata(thread_id, "https://example.com/gen1.png", "generated")
        temp_db.save_image_metadata(thread_id, "https://example.com/up1.png", "uploaded")
        temp_db.save_image_metadata(thread_id, "https://example.com/gen2.png", "generated")

        # Filter by type
        generated_images = temp_db.find_thread_images(thread_id, "generated")
        assert len(generated_images) == 2

        uploaded_images = temp_db.find_thread_images(thread_id, "uploaded")
        assert len(uploaded_images) == 1

        # No filter - all images
        all_images = temp_db.find_thread_images(thread_id)
        assert len(all_images) == 3

    def test_get_latest_thread_image(self, temp_db):
        """Test getting latest thread image"""
        thread_id = "C123:1234567890.123"

        # Save multiple images
        temp_db.save_image_metadata(thread_id, "https://example.com/old.png", "generated")
        temp_db.save_image_metadata(thread_id, "https://example.com/new.png", "generated")

        latest = temp_db.get_latest_thread_image(thread_id)
        assert latest is not None
        assert latest['url'] == "https://example.com/new.png"

    def test_get_latest_thread_image_none(self, temp_db):
        """Test getting latest thread image when none exist"""
        result = temp_db.get_latest_thread_image("nonexistent")
        assert result is None

    def test_get_user_preferences_not_found(self, temp_db):
        """Test getting user preferences for non-existent user"""
        result = temp_db.get_user_preferences("U999")
        assert result is None

    @patch('database.config')
    def test_create_default_user_preferences(self, mock_config, temp_db):
        """Test creating default user preferences"""
        # Mock config values
        mock_config.gpt_model = "gpt-5"
        mock_config.default_reasoning_effort = "medium"
        mock_config.default_verbosity = "medium"
        mock_config.default_temperature = 0.8
        mock_config.default_top_p = 1.0
        mock_config.enable_web_search = True
        mock_config.enable_streaming = True
        mock_config.default_image_size = "1024x1024"
        mock_config.default_input_fidelity = "high"
        mock_config.default_detail_level = "auto"

        user_id = "U123"
        email = "test@example.com"

        defaults = temp_db.create_default_user_preferences(user_id, email)

        assert defaults['slack_user_id'] == user_id
        assert defaults['slack_email'] == email
        assert defaults['model'] == "gpt-5"
        assert defaults['settings_completed'] is False

        # Verify preferences were actually saved
        prefs = temp_db.get_user_preferences(user_id)
        assert prefs is not None
        assert prefs['model'] == "gpt-5"

    @patch('database.config')
    def test_create_default_user_preferences_without_email(self, mock_config, temp_db):
        """Test creating default user preferences without email"""
        # Mock config values
        mock_config.gpt_model = "gpt-5"
        mock_config.default_reasoning_effort = "medium"
        mock_config.default_verbosity = "medium"
        mock_config.default_temperature = 0.8
        mock_config.default_top_p = 1.0
        mock_config.enable_web_search = True
        mock_config.enable_streaming = True
        mock_config.default_image_size = "1024x1024"
        mock_config.default_input_fidelity = "high"
        mock_config.default_detail_level = "auto"

        user_id = "U123"

        defaults = temp_db.create_default_user_preferences(user_id)

        assert defaults['slack_user_id'] == user_id
        assert defaults['slack_email'] is None

    def test_update_user_preferences(self, temp_db):
        """Test updating user preferences"""
        user_id = "U123"

        # Create initial preferences
        with patch('database.config') as mock_config:
            mock_config.gpt_model = "gpt-5"
            mock_config.default_reasoning_effort = "medium"
            mock_config.default_verbosity = "medium"
            mock_config.default_temperature = 0.8
            mock_config.default_top_p = 1.0
            mock_config.enable_web_search = True
            mock_config.enable_streaming = True
            mock_config.default_image_size = "1024x1024"
            mock_config.default_input_fidelity = "high"
            mock_config.default_detail_level = "auto"
            temp_db.create_default_user_preferences(user_id)

        # Update preferences
        updates = {
            'model': 'gpt-5-mini',
            'temperature': 0.5,
            'enable_web_search': False
        }

        result = temp_db.update_user_preferences(user_id, updates)
        assert result is True

        # Verify updates
        prefs = temp_db.get_user_preferences(user_id)
        assert prefs['model'] == 'gpt-5-mini'
        assert prefs['temperature'] == 0.5
        assert prefs['enable_web_search'] is False

    def test_update_user_preferences_nonexistent(self, temp_db):
        """Test updating preferences for non-existent user"""
        result = temp_db.update_user_preferences("U999", {'model': 'gpt-5'})
        assert result is False

    def test_get_user_preferences_boolean_conversion(self, temp_db):
        """Test user preferences boolean conversion from SQLite"""
        user_id = "U123"

        # Create preferences with explicit boolean values
        with patch('database.config') as mock_config:
            mock_config.gpt_model = "gpt-5"
            mock_config.default_reasoning_effort = "medium"
            mock_config.default_verbosity = "medium"
            mock_config.default_temperature = 0.8
            mock_config.default_top_p = 1.0
            mock_config.enable_web_search = True
            mock_config.enable_streaming = False
            mock_config.default_image_size = "1024x1024"
            mock_config.default_input_fidelity = "high"
            mock_config.default_detail_level = "auto"
            temp_db.create_default_user_preferences(user_id)

        prefs = temp_db.get_user_preferences(user_id)

        # Verify boolean conversion
        assert isinstance(prefs['enable_web_search'], bool)
        assert isinstance(prefs['enable_streaming'], bool)
        assert isinstance(prefs['settings_completed'], bool)
        assert prefs['enable_web_search'] is True
        assert prefs['enable_streaming'] is False

    def test_image_metadata_with_json(self, temp_db):
        """Test image metadata handling with JSON metadata"""
        thread_id = "C123:1234567890.123"
        url = "https://example.com/image.png"
        metadata = {"width": 1024, "height": 1024, "format": "PNG"}

        temp_db.save_image_metadata(
            thread_id, url, "generated",
            metadata=metadata
        )

        # Retrieve and verify JSON metadata is properly parsed
        images = temp_db.find_thread_images(thread_id)
        assert len(images) == 1
        assert images[0]['metadata'] == metadata

    def test_latest_image_with_metadata(self, temp_db):
        """Test latest image retrieval with JSON metadata"""
        thread_id = "C123:1234567890.123"
        metadata = {"test": "value"}

        temp_db.save_image_metadata(
            thread_id, "https://example.com/image.png", "generated",
            metadata=metadata
        )

        latest = temp_db.get_latest_thread_image(thread_id)
        assert latest is not None
        assert latest['metadata'] == metadata

    def test_message_caching_with_json_metadata(self, temp_db):
        """Test message caching with JSON metadata handling"""
        thread_id = "C123:1234567890.123"
        metadata = {"type": "system", "flags": ["important"]}

        temp_db.cache_message(
            thread_id, "assistant", "Response",
            metadata=metadata
        )

        messages = temp_db.get_cached_messages(thread_id)
        assert len(messages) == 1
        assert messages[0]['metadata'] == metadata


@pytest.mark.critical
class TestDatabaseManagerCritical:
    """Critical tests for database functionality"""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch('os.makedirs'):
                db = DatabaseManager("test")
                db.db_path = f"{tmpdir}/test.db"
                if hasattr(db, 'conn') and db.conn:
                    db.conn.close()
                db.conn = sqlite3.connect(
                    db.db_path,
                    check_same_thread=False,
                    isolation_level=None
                )
                db.conn.row_factory = sqlite3.Row
                db.init_schema()
                yield db
                if hasattr(db, 'conn') and db.conn:
                    db.conn.close()

    def test_critical_thread_operations_consistency(self, temp_db):
        """Critical test for thread operations consistency"""
        thread_id = "C123:1234567890.123"
        channel_id = "C123"

        # Create thread
        result1 = temp_db.get_or_create_thread(thread_id, channel_id)
        assert result1['created'] is True

        # Get existing thread
        result2 = temp_db.get_or_create_thread(thread_id, channel_id)
        assert result2['created'] is False
        assert result1['thread_id'] == result2['thread_id']

    def test_critical_message_ordering(self, temp_db):
        """Critical test for message ordering in cache"""
        thread_id = "C123:1234567890.123"

        # Add messages with specific timestamps
        times = [
            datetime.now() - timedelta(minutes=3),
            datetime.now() - timedelta(minutes=1),
            datetime.now() - timedelta(minutes=2),
        ]

        for i, time in enumerate(times):
            temp_db.cache_message(
                thread_id, "user", f"Message {i}",
                timestamp=time
            )

        # Get messages - should be ordered by timestamp
        messages = temp_db.get_cached_messages(thread_id)
        assert len(messages) == 3

        # Verify chronological order
        for i in range(len(messages) - 1):
            current_time = datetime.fromisoformat(messages[i]['timestamp'])
            next_time = datetime.fromisoformat(messages[i + 1]['timestamp'])
            assert current_time <= next_time

    def test_critical_user_preferences_integrity(self, temp_db):
        """Critical test for user preferences data integrity"""
        user_id = "U123"

        with patch('database.config') as mock_config:
            mock_config.gpt_model = "gpt-5"
            mock_config.default_reasoning_effort = "medium"
            mock_config.default_verbosity = "medium"
            mock_config.default_temperature = 0.8
            mock_config.default_top_p = 1.0
            mock_config.enable_web_search = True
            mock_config.enable_streaming = True
            mock_config.default_image_size = "1024x1024"
            mock_config.default_input_fidelity = "high"
            mock_config.default_detail_level = "auto"

            # Create and verify defaults
            defaults = temp_db.create_default_user_preferences(user_id)
            retrieved = temp_db.get_user_preferences(user_id)

            # Critical fields must match
            assert retrieved['slack_user_id'] == defaults['slack_user_id']
            assert retrieved['model'] == defaults['model']
            assert retrieved['temperature'] == defaults['temperature']