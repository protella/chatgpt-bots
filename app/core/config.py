"""Configuration management service for per-thread settings.

This module provides a SQLite-backed configuration store for maintaining
thread-specific settings that persist across bot restarts.
"""

import os
import json
import sqlite3
import logging
from typing import Dict, Any, Optional
import re

import prompts

# Initialize logger
logger = logging.getLogger(__name__)

# Default config values
DEFAULT_CONFIG = {
    "temperature": 0.8,
    "top_p": 1.0,
    "max_output_tokens": 2048,
    "custom_init": "",
    "gpt_model": "gpt-4.1-2025-04-14",
    "image_model": "gpt-image-1",
    "size": "1024x1024",
    "quality": "hd",
    "style": "natural",
    "number": 1,
    "detail": "auto",
    "d3_revised_prompt": False,
    "system_prompt": prompts.SLACK_SYSTEM_PROMPT["content"]
}

# Data directory for SQLite storage
# Use /data in production (Docker volume) or local directory for dev/tests
if os.path.exists("/data") and os.access("/data", os.W_OK):
    DATA_DIR = "/data"
else:
    # Use a local directory for development and testing
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "config.db")

logger.info(f"Using config database at: {DB_PATH}")

class ConfigService:
    """Service for managing thread-specific configuration."""
    
    def __init__(self, use_memory_store: bool = False):
        """
        Initialize the config service.
        
        Args:
            use_memory_store: If True, use in-memory dict instead of SQLite (for testing)
        """
        self.use_memory_store = use_memory_store
        self.memory_store = {}
        
        if not use_memory_store:
            self._init_db()
    
    def _init_db(self) -> None:
        """Initialize the SQLite database with the required table."""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                CREATE TABLE IF NOT EXISTS thread_config (
                    thread_id TEXT PRIMARY KEY,
                    config TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                conn.commit()
            logger.info(f"Initialized config database at {DB_PATH}")
        except Exception as e:
            logger.error(f"Failed to initialize config database: {str(e)}")
            raise
    
    def get(self, thread_id: str) -> Dict[str, Any]:
        """
        Get configuration for a specific thread.
        
        Args:
            thread_id: The Slack thread ID
            
        Returns:
            Dict containing thread configuration (defaults applied if not found)
        """
        if self.use_memory_store:
            return self.memory_store.get(thread_id, DEFAULT_CONFIG.copy())
        
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT config FROM thread_config WHERE thread_id = ?",
                    (thread_id,)
                )
                result = cursor.fetchone()
                
                if result:
                    return json.loads(result[0])
                else:
                    # Store defaults for new thread
                    config = DEFAULT_CONFIG.copy()
                    self._store_config(thread_id, config)
                    return config
                    
        except Exception as e:
            logger.error(f"Error retrieving config for thread {thread_id}: {str(e)}")
            return DEFAULT_CONFIG.copy()
    
    def update(self, thread_id: str, updates: Dict[str, Any]) -> None:
        """
        Update configuration for a specific thread.
        
        Args:
            thread_id: The Slack thread ID
            updates: Dictionary of config values to update
        """
        try:
            # Get current config
            current_config = self.get(thread_id)
            
            # Apply updates
            for key, value in updates.items():
                if key in current_config:
                    current_config[key] = value
            
            # Store updated config
            self._store_config(thread_id, current_config)
            logger.info(f"Updated config for thread {thread_id}: {updates}")
            
        except Exception as e:
            logger.error(f"Error updating config for thread {thread_id}: {str(e)}")
            raise
    
    def reset(self, thread_id: str) -> None:
        """
        Reset configuration for a specific thread to defaults.
        
        Args:
            thread_id: The Slack thread ID
        """
        try:
            self._store_config(thread_id, DEFAULT_CONFIG.copy())
            logger.info(f"Reset config for thread {thread_id} to defaults")
        except Exception as e:
            logger.error(f"Error resetting config for thread {thread_id}: {str(e)}")
            raise
    
    def _store_config(self, thread_id: str, config: Dict[str, Any]) -> None:
        """
        Store configuration in the backing store.
        
        Args:
            thread_id: The Slack thread ID
            config: Full configuration dictionary to store
        """
        if self.use_memory_store:
            self.memory_store[thread_id] = config
            return
            
        try:
            config_json = json.dumps(config)
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                INSERT INTO thread_config (thread_id, config, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT (thread_id) 
                DO UPDATE SET config = excluded.config, updated_at = CURRENT_TIMESTAMP
                """, (thread_id, config_json))
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing config for thread {thread_id}: {str(e)}")
            raise

    def extract_config_from_text(self, text: str) -> Dict[str, Any]:
        """
        Extract configuration values from natural language text.
        
        Args:
            text: User's message text
            
        Returns:
            Dict of extracted config values (empty if none found)
        """
        extracted_config = {}
        
        # Number of images
        number_match = re.search(r'generate\s+(\d+)\s+images?', text.lower())
        if number_match:
            try:
                num = int(number_match.group(1))
                if 1 <= num <= 4:  # Validate range
                    extracted_config["number"] = num
            except ValueError:
                pass
        
        # Image style
        if re.search(r'\b(vivid|natural)\s+style\b', text.lower()):
            style = re.search(r'\b(vivid|natural)\s+style\b', text.lower()).group(1)
            extracted_config["style"] = style
        
        # Image model
        if re.search(r'\b(dall-?e-?3|gpt-image-1)\b', text.lower()):
            if re.search(r'\bdall-?e-?3\b', text.lower()):
                extracted_config["image_model"] = "dall-e-3"
            elif re.search(r'\bgpt-image-1\b', text.lower()):
                extracted_config["image_model"] = "gpt-image-1"
        
        # Image quality (DALL-E 3)
        if re.search(r'\b(standard|hd)\s+quality\b', text.lower()):
            quality = re.search(r'\b(standard|hd)\s+quality\b', text.lower()).group(1)
            extracted_config["quality"] = quality
        
        # Image size
        size_patterns = {
            r'\b1024x1024\b': "1024x1024",
            r'\b1792x1024\b': "1792x1024",
            r'\b1024x1792\b': "1024x1792",
            r'\bsquare\b': "1024x1024",
            r'\blandscape\b': "1792x1024",
            r'\bportrait\b': "1024x1792"
        }
        
        for pattern, size in size_patterns.items():
            if re.search(pattern, text.lower()):
                extracted_config["size"] = size
                break
        
        # Vision detail level
        if re.search(r'\b(low|high|auto)\s+detail\b', text.lower()):
            detail = re.search(r'\b(low|high|auto)\s+detail\b', text.lower()).group(1)
            extracted_config["detail"] = detail
            
        # Temperature
        temp_match = re.search(r'(?:set\s+)?temperature\s+(?:to\s+)?(\d+(?:\.\d+)?)', text.lower())
        if temp_match:
            try:
                temp = float(temp_match.group(1))
                if 0.0 <= temp <= 2.0:  # Validate range
                    extracted_config["temperature"] = temp
            except ValueError:
                pass
        
        return extracted_config 