"""Configuration management service for per-thread settings.

This module provides a SQLite-backed configuration store for maintaining
thread-specific settings that persist across bot restarts.
"""

import os
import json
import sqlite3
import logging
from typing import Dict, Any
import re

import prompts

# Initialize logger
logger = logging.getLogger(__name__)

# Get model configuration from environment variables
GPT_MODEL = os.environ.get("GPT_MODEL", "gpt-4.1-2025-04-14")
GPT_IMAGE_MODEL = os.environ.get("GPT_IMAGE_MODEL", "gpt-image-1")
DALLE_MODEL = os.environ.get("DALLE_MODEL", "dall-e-3")

# Default config values
DEFAULT_CONFIG = {
    "temperature": 0.8,
    "top_p": 1.0,
    "max_output_tokens": 2048,
    "custom_init": "",
    "gpt_model": GPT_MODEL,
    "image_model": GPT_IMAGE_MODEL,
    "size": "auto", # 1024x1024, 1536x1024 (landscape), 1024x1536 (portrait), or auto
    "quality": "auto",
    "style": "natural",
    "number": 1,
    "detail": "auto",
    "d3_revised_prompt": False,
    "moderation": "low", # "auto" or "low"
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
        dalle_pattern = r'\bdall-?e-?3\b'
        gpt_image_pattern = r'\bgpt-image-1\b'
        
        model_detected = None
        if re.search(rf'\b({dalle_pattern}|{gpt_image_pattern})\b', text.lower()):
            if re.search(dalle_pattern, text.lower()):
                extracted_config["image_model"] = DALLE_MODEL
                model_detected = DALLE_MODEL
            elif re.search(gpt_image_pattern, text.lower()):
                extracted_config["image_model"] = GPT_IMAGE_MODEL
                model_detected = GPT_IMAGE_MODEL
        
        # Image quality - model specific validation
        if re.search(r'\b(standard|hd|auto|low|medium|high)\s+quality\b', text.lower()):
            quality = re.search(r'\b(standard|hd|auto|low|medium|high)\s+quality\b', text.lower()).group(1)
            
            # Validate quality based on the detected or default model
            if model_detected is None:
                model_detected = DEFAULT_CONFIG["image_model"]
            
            if model_detected == DALLE_MODEL and quality in ["standard", "hd"]:
                # Valid DALL-E 3 quality
                extracted_config["quality"] = quality
            elif model_detected == GPT_IMAGE_MODEL and quality in ["auto", "low", "medium", "high"]:
                # Valid GPT-Image-1 quality
                extracted_config["quality"] = quality
            else:
                # Use appropriate default for the model
                if model_detected == DALLE_MODEL:
                    extracted_config["quality"] = "standard"
                    logger.warning(f"Invalid quality '{quality}' for {model_detected}, using 'standard'")
                else:
                    extracted_config["quality"] = "auto"
                    logger.warning(f"Invalid quality '{quality}' for {model_detected}, using 'auto'")
        
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