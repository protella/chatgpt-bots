"""
Database models for the Slackbot application
"""
import logging
from db.connection import execute_query, get_connection, release_connection

# Configure logging
logger = logging.getLogger("DB_MODELS")

class Conversation:
    """Conversation model for interacting with the conversations table"""
    
    @staticmethod
    def create(thread_ts, channel_id, previous_response_id=None):
        """Create a new conversation record"""
        try:
            query = """
                INSERT INTO conversations (thread_ts, channel_id, previous_response_id)
                VALUES (%s, %s, %s)
                ON CONFLICT (thread_ts) DO UPDATE
                SET previous_response_id = %s, updated_at = CURRENT_TIMESTAMP
                RETURNING thread_ts
            """
            result = execute_query(
                query, 
                (thread_ts, channel_id, previous_response_id, previous_response_id),
                fetchone=True
            )
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            return None
    
    @staticmethod
    def get(thread_ts):
        """Get a conversation by thread_ts"""
        try:
            query = "SELECT * FROM conversations WHERE thread_ts = %s"
            return execute_query(query, (thread_ts,), fetchone=True)
        except Exception as e:
            logger.error(f"Failed to get conversation: {e}")
            return None
    
    @staticmethod
    def update_response_id(thread_ts, previous_response_id):
        """Update the previous_response_id for a conversation"""
        try:
            query = """
                UPDATE conversations
                SET previous_response_id = %s, updated_at = CURRENT_TIMESTAMP
                WHERE thread_ts = %s
                RETURNING thread_ts
            """
            result = execute_query(query, (previous_response_id, thread_ts), fetchone=True)
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Failed to update conversation response ID: {e}")
            return None
    
    @staticmethod
    def delete(thread_ts):
        """Delete a conversation"""
        try:
            query = "DELETE FROM conversations WHERE thread_ts = %s RETURNING thread_ts"
            result = execute_query(query, (thread_ts,), fetchone=True)
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Failed to delete conversation: {e}")
            return None


class UserConfig:
    """User configuration model for interacting with the user_configs table"""
    
    @staticmethod
    def get(user_id):
        """Get a user's configuration"""
        try:
            query = "SELECT * FROM user_configs WHERE user_id = %s"
            return execute_query(query, (user_id,), fetchone=True)
        except Exception as e:
            logger.error(f"Failed to get user config: {e}")
            return None
    
    @staticmethod
    def create_or_update(user_id, config_data):
        """Create or update a user's configuration"""
        try:
            # Get existing config first
            existing = UserConfig.get(user_id)
            
            # If it doesn't exist, create it with defaults and our updates
            if not existing:
                columns = ", ".join(config_data.keys())
                placeholders = ", ".join(["%s"] * len(config_data))
                params = list(config_data.values())
                
                query = f"""
                    INSERT INTO user_configs (user_id, {columns})
                    VALUES (%s, {placeholders})
                    RETURNING user_id
                """
                result = execute_query(query, [user_id] + params, fetchone=True)
                return result[0] if result else None
            
            # Otherwise update the existing config
            else:
                set_clause = ", ".join([f"{key} = %s" for key in config_data.keys()])
                params = list(config_data.values())
                
                query = f"""
                    UPDATE user_configs
                    SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = %s
                    RETURNING user_id
                """
                result = execute_query(query, params + [user_id], fetchone=True)
                return result[0] if result else None
                
        except Exception as e:
            logger.error(f"Failed to update user config: {e}")
            return None
    
    @staticmethod
    def update_single_setting(user_id, setting_name, setting_value):
        """Update a single setting for a user"""
        try:
            return UserConfig.create_or_update(user_id, {setting_name: setting_value})
        except Exception as e:
            logger.error(f"Failed to update single setting: {e}")
            return None


class StatusMessage:
    """Status message model for interacting with the status_messages table"""
    
    @staticmethod
    def create(message_id, thread_ts, channel_id, message_type):
        """Create a new status message record"""
        try:
            query = """
                INSERT INTO status_messages (message_id, thread_ts, channel_id, message_type)
                VALUES (%s, %s, %s, %s)
                RETURNING message_id
            """
            result = execute_query(
                query, 
                (message_id, thread_ts, channel_id, message_type),
                fetchone=True
            )
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Failed to create status message: {e}")
            return None
    
    @staticmethod
    def get_by_thread(thread_ts):
        """Get all status messages for a thread"""
        try:
            query = "SELECT * FROM status_messages WHERE thread_ts = %s"
            return execute_query(query, (thread_ts,), fetchall=True)
        except Exception as e:
            logger.error(f"Failed to get status messages: {e}")
            return None
    
    @staticmethod
    def delete(message_id):
        """Delete a status message"""
        try:
            query = "DELETE FROM status_messages WHERE message_id = %s RETURNING message_id"
            result = execute_query(query, (message_id,), fetchone=True)
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Failed to delete status message: {e}")
            return None
    
    @staticmethod
    def delete_by_thread(thread_ts):
        """Delete all status messages for a thread"""
        try:
            query = "DELETE FROM status_messages WHERE thread_ts = %s RETURNING message_id"
            return execute_query(query, (thread_ts,), fetchall=True)
        except Exception as e:
            logger.error(f"Failed to delete status messages by thread: {e}")
            return None 