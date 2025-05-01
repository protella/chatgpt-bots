import os
import pytest
import psycopg2
from db.connection import get_connection, release_connection, init_connection_pool
from db.models import Conversation, UserConfig, StatusMessage

# Database connection parameters
DB_HOST = os.environ.get("DB_HOST", "db")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("POSTGRES_DB", "slackbot")
DB_USER = os.environ.get("POSTGRES_USER", "postgres")
DB_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")

def test_database_connection():
    """Test that we can connect to the database"""
    try:
        # Initialize connection pool
        assert init_connection_pool() == True
        
        # Get a connection
        conn = get_connection()
        assert conn is not None
        
        # Create a cursor
        cursor = conn.cursor()
        
        # Execute a simple query
        cursor.execute("SELECT 1")
        result = cursor.fetchone()
        
        # Close cursor
        cursor.close()
        
        # Release connection
        release_connection(conn)
        
        # Check the result
        assert result[0] == 1
        
    except Exception as e:
        pytest.fail(f"Database connection failed: {e}")

def test_tables_exist():
    """Test that the expected tables exist in the database"""
    try:
        # Get a connection
        conn = get_connection()
        assert conn is not None
        
        # Create a cursor
        cursor = conn.cursor()
        
        # Query to check if tables exist
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            AND table_name IN ('conversations', 'user_configs', 'status_messages')
        """)
        
        tables = cursor.fetchall()
        table_names = [table[0] for table in tables]
        
        # Close cursor
        cursor.close()
        
        # Release connection
        release_connection(conn)
        
        # Check that all expected tables exist
        assert 'conversations' in table_names
        assert 'user_configs' in table_names
        assert 'status_messages' in table_names
        
    except Exception as e:
        pytest.fail(f"Table existence test failed: {e}")

def test_conversation_model():
    """Test the Conversation model"""
    try:
        # Create a test conversation
        thread_ts = "test_thread_1"
        channel_id = "test_channel_1"
        previous_response_id = "test_response_1"
        
        # Create the conversation
        result = Conversation.create(thread_ts, channel_id, previous_response_id)
        assert result == thread_ts
        
        # Get the conversation
        conversation = Conversation.get(thread_ts)
        assert conversation is not None
        assert conversation[0] == thread_ts
        assert conversation[1] == previous_response_id
        assert conversation[2] == channel_id
        
        # Update the conversation
        new_response_id = "test_response_2"
        result = Conversation.update_response_id(thread_ts, new_response_id)
        assert result == thread_ts
        
        # Get the updated conversation
        conversation = Conversation.get(thread_ts)
        assert conversation is not None
        assert conversation[0] == thread_ts
        assert conversation[1] == new_response_id
        
        # Delete the conversation
        result = Conversation.delete(thread_ts)
        assert result == thread_ts
        
        # Verify it's gone
        conversation = Conversation.get(thread_ts)
        assert conversation is None
        
    except Exception as e:
        pytest.fail(f"Conversation model test failed: {e}")

def test_user_config_model():
    """Test the UserConfig model"""
    try:
        # Create a test user config
        user_id = "test_user_1"
        config_data = {
            "temperature": 0.7,
            "max_completion_tokens": 1000,
            "gpt_model": "test-model"
        }
        
        # Create the user config
        result = UserConfig.create_or_update(user_id, config_data)
        assert result == user_id
        
        # Get the user config
        user_config = UserConfig.get(user_id)
        assert user_config is not None
        assert user_config[0] == user_id
        assert float(user_config[1]) == config_data["temperature"]
        assert int(user_config[3]) == config_data["max_completion_tokens"]
        assert user_config[5] == config_data["gpt_model"]
        
        # Update a single setting
        new_temp = 0.9
        result = UserConfig.update_single_setting(user_id, "temperature", new_temp)
        assert result == user_id
        
        # Get the updated user config
        user_config = UserConfig.get(user_id)
        assert user_config is not None
        assert float(user_config[1]) == new_temp
        
        # Clean up test data using direct SQL since we don't have a delete method
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_configs WHERE user_id = %s", (user_id,))
        cursor.close()
        release_connection(conn)
        
    except Exception as e:
        pytest.fail(f"User config model test failed: {e}")

def test_status_message_model():
    """Test the StatusMessage model"""
    try:
        # First create a test conversation (required for foreign key)
        thread_ts = "test_thread_2"
        channel_id = "test_channel_2"
        Conversation.create(thread_ts, channel_id)
        
        # Create test status messages
        message_id_1 = "test_message_1"
        message_id_2 = "test_message_2"
        message_type = "thinking"
        
        # Create the status messages
        result_1 = StatusMessage.create(message_id_1, thread_ts, channel_id, message_type)
        assert result_1 == message_id_1
        
        result_2 = StatusMessage.create(message_id_2, thread_ts, channel_id, message_type)
        assert result_2 == message_id_2
        
        # Get messages by thread
        messages = StatusMessage.get_by_thread(thread_ts)
        assert messages is not None
        assert len(messages) == 2
        message_ids = [message[0] for message in messages]
        assert message_id_1 in message_ids
        assert message_id_2 in message_ids
        
        # Delete one message
        result = StatusMessage.delete(message_id_1)
        assert result == message_id_1
        
        # Verify it's gone
        messages = StatusMessage.get_by_thread(thread_ts)
        assert len(messages) == 1
        assert messages[0][0] == message_id_2
        
        # Delete all messages for thread
        results = StatusMessage.delete_by_thread(thread_ts)
        assert results is not None
        assert len(results) == 1
        assert results[0][0] == message_id_2
        
        # Verify all gone
        messages = StatusMessage.get_by_thread(thread_ts)
        assert messages == []
        
        # Clean up conversation
        Conversation.delete(thread_ts)
        
    except Exception as e:
        pytest.fail(f"Status message model test failed: {e}") 