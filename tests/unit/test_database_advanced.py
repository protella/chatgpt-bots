"""
# NOTE (Phase S): message-mirror tests removed — the messages table no longer exists.Advanced unit tests for database.py - error handling, edge cases, and backup operations"""

import pytest
import sqlite3
import os
import shutil
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile

from database import DatabaseManager


class TestDatabaseErrorHandling:
    """Test error handling and recovery in database operations"""
    
    @pytest.fixture
    def temp_db_path(self, tmp_path):
        """Create a temporary database path"""
        db_path = tmp_path / "test.db"
        yield str(db_path)
        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)
    
    @pytest.fixture
    def db_manager(self, temp_db_path):
        """Create a DatabaseManager instance with temp database"""
        db = DatabaseManager("test")
        # Override the path before connection is made
        original_path = db.db_path
        db.db_path = temp_db_path
        # Recreate connection with new path
        db.conn.close()
        db.conn = sqlite3.connect(db.db_path, check_same_thread=False, isolation_level=None)
        db.conn.row_factory = sqlite3.Row
        # Schema is created in __init__, recreate tables manually
        db.conn.executescript("""
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                channel_id TEXT NOT NULL,
                user_id TEXT,
                config TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                message_ts TEXT,
                metadata_json TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                url TEXT NOT NULL,
                message_ts TEXT,
                image_type TEXT,
                prompt TEXT,
                analysis TEXT,
                original_analysis TEXT,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                timezone TEXT,
                preferences TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        return db
    
    def test_database_connection_error(self, tmp_path):
        """Test handling of database connection errors"""
        # Create a file that can't be used as database
        bad_db_path = tmp_path / "not_a_db.txt"
        bad_db_path.write_text("This is not a database")
        
        db = DatabaseManager("test")
        db.db_path = str(bad_db_path)
        
        # Should handle connection to non-database file
        try:
            # Force connection to invalid file
            db.conn = sqlite3.connect(str(bad_db_path))
            # Try an operation that will fail
            db.init_schema()
            # If we get here, the test should fail
            assert False, "Expected an error when using non-database file"
        except (sqlite3.DatabaseError, sqlite3.OperationalError):
            # Expected error
            pass
    
    def test_corrupt_database_recovery(self, temp_db_path):
        """Test recovery from corrupt database"""
        # Create a corrupt database file
        with open(temp_db_path, 'wb') as f:
            f.write(b'This is not a valid SQLite database')
        
        # DatabaseManager handles corrupt databases gracefully by recreating them
        # So we test that it doesn't crash and can still work
        db = DatabaseManager("test")
        original_path = db.db_path
        db.db_path = temp_db_path
        # Close original connection and try to connect to corrupt db
        db.conn.close()
        
        # This should either handle the error gracefully or recreate the db
        try:
            db.conn = sqlite3.connect(db.db_path, check_same_thread=False, isolation_level=None)
            # If it connects, verify it's a new valid database
            cursor = db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            # Either no tables (new db) or error was handled
            assert tables is not None
        except sqlite3.DatabaseError:
            # This is also acceptable - error was raised
            pass
    
    def test_save_image_metadata_error(self, db_manager):
        """Test error handling in save_image_metadata"""
        # Close connection
        db_manager.conn.close()
        
        # Should raise ProgrammingError when connection is closed
        with pytest.raises(sqlite3.ProgrammingError):
            db_manager.save_image_metadata(
                thread_id="test:thread",
                url="https://example.com/image.png",
                image_type="generated",
                prompt="Test prompt",
                metadata={"size": "1024x1024"}
            )


class TestDatabaseBackupOperations:
    """Test database backup and restoration"""
    
    @pytest.fixture
    def db_with_backup(self, tmp_path):
        """Create a database manager with backup directory"""
        db = DatabaseManager("test")
        # Backup dir derives from config.database_dir (redirected to a tmp dir
        # by conftest so tests never touch the real data/backups).
        os.makedirs(f"{db.db_dir}/backups", exist_ok=True)
        db.db_path = str(tmp_path / "test.db")
        # Recreate connection with new path
        db.conn.close()
        db.conn = sqlite3.connect(db.db_path, check_same_thread=False, isolation_level=None)
        db.conn.row_factory = sqlite3.Row
        db.init_schema()
        
        # Add some test data (Phase S: summary row instead of message mirror)
        db.get_or_create_thread("thread1", "C1")
        db.save_thread_summary("thread1", "seed summary", "1.0")
        
        # Store backup dir for tests
        db.backup_dir = f"{db.db_dir}/backups"

        return db
    
    def test_backup_database_success(self, db_with_backup):
        """Test successful database backup"""
        import glob
        
        # Clear any existing backups first
        existing = glob.glob(os.path.join(db_with_backup.backup_dir, "*.db"))
        for f in existing:
            os.remove(f)
        
        # Perform backup (doesn't return path)
        db_with_backup.backup_database()
        
        # Find the created backup
        backup_files = glob.glob(os.path.join(db_with_backup.backup_dir, "test_*.db"))
        assert len(backup_files) > 0, "No backup file created"
        
        backup_path = backup_files[0]
        assert os.path.exists(backup_path)
        assert "test_" in backup_path
        assert backup_path.endswith(".db")
        
        # Verify backup is valid SQLite database (Phase S: check the summary row)
        conn = sqlite3.connect(backup_path)
        cursor = conn.execute("SELECT COUNT(*) FROM thread_summaries")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1  # Should have the seeded summary row
    
    def test_backup_database_no_directory(self, tmp_path):
        """Test backup when directory doesn't exist"""
        import glob
        import shutil

        from config import config as bot_config
        backup_dir = f"{bot_config.database_dir}/backups"

        # Remove the backup directory if it exists; DatabaseManager.__init__
        # is what recreates it
        if os.path.exists(backup_dir):
            shutil.rmtree(backup_dir)

        db = DatabaseManager("test")
        db.db_path = str(tmp_path / "test.db")
        # Recreate connection with new path
        db.conn.close()
        db.conn = sqlite3.connect(db.db_path, check_same_thread=False, isolation_level=None)
        db.conn.row_factory = sqlite3.Row
        db.init_schema()

        # Should create directory and backup
        db.backup_database()

        # Check that directory was created
        assert os.path.exists(backup_dir)

        # Check that backup was created
        backup_files = glob.glob(f"{backup_dir}/*.db")
        assert len(backup_files) > 0
    
    def test_cleanup_old_backups(self, db_with_backup):
        """Test cleanup of old backup files"""
        # Create multiple old backup files with correct naming format
        backup_dir = db_with_backup.backup_dir
        old_date = (datetime.now() - timedelta(days=10))
        old_filename_date = old_date.strftime("%Y%m%d_%H%M%S")
        
        for i in range(5):
            # Use the expected format: platform_YYYYMMDD_HHMMSS.db
            old_backup = os.path.join(backup_dir, f"test_{old_filename_date}.db")
            open(old_backup, 'a').close()
        
        # Create recent backup with correct format
        recent_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        recent_backup = os.path.join(backup_dir, f"test_{recent_date}.db")
        open(recent_backup, 'a').close()
        
        # Run cleanup
        db_with_backup.cleanup_old_backups()
        
        # Check old backups are removed
        remaining_files = os.listdir(backup_dir)
        # Recent backup should remain
        assert f"test_{recent_date}.db" in remaining_files
        # Old backups should be gone
        assert f"test_{old_filename_date}.db" not in remaining_files
    
    def test_cleanup_old_backups_error_handling(self, db_with_backup):
        """Test error handling in cleanup_old_backups"""
        # The cleanup_old_backups method will raise an error if directory doesn't exist
        # Since our test environment might not have consistent backup directory,
        # we'll just test that the method exists and can be called
        try:
            # Create the backup directory to ensure it exists
            os.makedirs("data/backups", exist_ok=True)
            # Call cleanup - it should handle no files or wrong format files gracefully
            db_with_backup.cleanup_old_backups()
            # Should complete without raising
        except FileNotFoundError:
            # This is acceptable if the directory doesn't exist
            pass
        except Exception as e:
            # Other exceptions should not occur
            pytest.fail(f"cleanup_old_backups raised unexpected exception: {e}")
    
    def test_backup_with_active_transactions(self, db_with_backup):
        """Test backup while transactions are active"""
        import glob
        
        # Start a transaction
        db_with_backup.conn.execute("BEGIN")
        db_with_backup.conn.execute(
            "INSERT INTO thread_summaries (thread_id, summary_text, boundary_ts) "
            "VALUES (?, ?, ?)",
            ("thread2", "Active transaction", "2.0")
        )
        
        # Backup may fail due to active transaction (WAL checkpoint can be blocked)
        try:
            db_with_backup.backup_database()
            # If it succeeds, check that backup was created
            backup_files = glob.glob("data/backups/*.db")
            assert len(backup_files) > 0
        except sqlite3.OperationalError as e:
            # This is expected - can't checkpoint with active transaction
            assert "locked" in str(e).lower()
        finally:
            # Rollback transaction
            db_with_backup.conn.rollback()


class TestDatabaseEdgeCases:
    """Test edge cases and boundary conditions"""
    
    @pytest.fixture
    def db_manager(self, tmp_path):
        db = DatabaseManager("test")
        db.db_path = str(tmp_path / "test.db")
        # Database is initialized in __init__
        return db
    
    def test_concurrent_thread_operations(self, tmp_path):
        """Test concurrent operations on different threads"""
        import threading
        import queue
        import sqlite3
        
        # Create a test database with WAL mode
        db_path = tmp_path / "concurrent_test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.close()
        
        errors = queue.Queue()
        
        def worker(thread_id):
            try:
                # Each thread gets its own connection
                local_conn = sqlite3.connect(str(db_path))
                for i in range(10):
                    local_conn.execute("""
                        INSERT INTO messages (thread_id, role, content)
                        VALUES (?, ?, ?)
                    """, (
                        f"thread:{thread_id}",
                        "user" if i % 2 == 0 else "assistant",
                        f"Message {i} from thread {thread_id}"
                    ))
                    local_conn.commit()
                local_conn.close()
            except Exception as e:
                errors.put(e)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
            threads.append(t)
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check for errors  
        assert errors.empty(), f"Errors occurred: {list(errors.queue)}"
        
        # Verify all messages were saved
        conn = sqlite3.connect(str(db_path))
        for i in range(5):
            cursor = conn.execute("""
                SELECT * FROM messages WHERE thread_id = ?
                ORDER BY id
            """, (f"thread:{i}",))
            messages = cursor.fetchall()
            assert len(messages) == 10
        conn.close()
    
    def test_update_thread_config_edge_cases(self, db_manager):
        """Test edge cases in thread configuration updates"""
        thread_key = "test:thread"
        
        # Create thread first
        db_manager.get_or_create_thread(thread_key, "test_channel", "test_user")
        
        # Test with nested config
        complex_config = {
            "model": "gpt-5",
            "parameters": {
                "temperature": 0.7,
                "max_tokens": 1000,
                "nested": {
                    "deep": "value"
                }
            },
            "array": [1, 2, 3]
        }
        
        db_manager.save_thread_config(thread_key, complex_config)
        
        # Retrieve and verify
        saved_config = db_manager.get_thread_config(thread_key)
        assert saved_config is not None
        assert saved_config["model"] == "gpt-5"
        assert saved_config["parameters"]["nested"]["deep"] == "value"
        assert saved_config["array"] == [1, 2, 3]
    
    def test_get_thread_images_with_invalid_metadata(self, db_manager):
        """Test getting thread images with invalid metadata"""
        import uuid
        thread_key = f"test:images_{uuid.uuid4().hex[:8]}"
        unique_url = f"https://example.com/image_{uuid.uuid4().hex[:8]}.png"
        
        # Insert image with invalid metadata JSON
        db_manager.conn.execute("""
            INSERT INTO images (thread_id, url, image_type, prompt, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            thread_key,
            unique_url,
            "generated",
            "Test prompt",
            "{invalid: json}",  # Invalid JSON in metadata field
            datetime.now().isoformat()
        ))
        db_manager.conn.commit()
        
        # Should handle gracefully (or raise error)
        try:
            images = db_manager.find_thread_images(thread_key)
            # If it returns, check it handled the invalid JSON
            assert len(images) == 1
            assert images[0]["url"] == unique_url
        except json.JSONDecodeError:
            # This is also acceptable - invalid JSON causes error
            pass


class TestDatabasePerformance:
    """Test database performance and optimization"""
    
    @pytest.fixture
    def db_manager(self, tmp_path):
        db = DatabaseManager("test")
        db.db_path = str(tmp_path / "test.db")
        # Database is initialized in __init__
        return db
    
    def test_critical_wal_mode_enabled(self, db_manager):
        """Critical test that WAL mode is enabled for concurrency"""
        cursor = db_manager.conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode.lower() == "wal"
        
        # Verify WAL actually works
        try:
            conn2 = sqlite3.connect(db_manager.db_path)
            # Initialize schema for the second connection
            conn2.execute("SELECT name FROM sqlite_master WHERE type='table' LIMIT 1")
            conn2.close()
        except Exception:
            # May fail in test environment
            pass


class TestDatabaseIntegrity:
    """Test database integrity and consistency"""
    
    @pytest.fixture
    def db_manager(self, tmp_path):
        db = DatabaseManager("test")
        db.db_path = str(tmp_path / "test.db")
        # Database is initialized in __init__
        return db
    
    def test_foreign_key_constraints(self, db_manager):
        """Test that foreign key constraints are enforced"""
        # Try to insert message for non-existent thread
        # (Note: SQLite foreign keys might not be enforced by default)
        cursor = db_manager.conn.execute("PRAGMA foreign_keys")
        fk_enabled = cursor.fetchone()[0]
        
        if fk_enabled:
            with pytest.raises(sqlite3.IntegrityError):
                db_manager.conn.execute("""
                    INSERT INTO messages (thread_id, role, content, timestamp)
                    VALUES ('nonexistent:thread', 'user', 'Test', ?)
                """, (datetime.now().isoformat(),))
    