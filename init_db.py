#!/usr/bin/env python3
"""
Database initialization script for the Slackbot application
"""
import os
import logging
import time
from db.init_db import init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DB_INIT")

if __name__ == "__main__":
    logger.info("Starting database initialization...")
    if init_db():
        logger.info("Database initialization completed successfully!")
    else:
        logger.error("Database initialization failed!")
        exit(1) 