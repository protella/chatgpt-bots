"""
Database initialization script for the Slackbot application
"""
import os
import logging
import psycopg2
import time

# Configure logging
logger = logging.getLogger("DB_INIT")

# Database connection parameters
DB_HOST = os.environ.get("DB_HOST", "db")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("POSTGRES_DB", "slackbot")
DB_USER = os.environ.get("POSTGRES_USER", "postgres")
DB_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")

# Maximum number of retry attempts
MAX_RETRIES = 10
RETRY_DELAY = 3  # seconds

def create_database():
    """Create the database if it doesn't exist"""
    try:
        # Connect to the default postgres database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname="postgres",
            user=DB_USER,
            password=DB_PASSWORD
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if our database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
        if not cursor.fetchone():
            logger.info(f"Creating database {DB_NAME}...")
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
            logger.info(f"Database {DB_NAME} created.")
        else:
            logger.info(f"Database {DB_NAME} already exists.")
            
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        return False

def apply_schema():
    """Apply the database schema"""
    try:
        # Connect to the database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        conn.autocommit = True
        
        # Create a cursor
        cursor = conn.cursor()
        
        # Read and execute the schema file
        logger.info("Connected to database. Initializing schema...")
        schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
        with open(schema_path, 'r') as schema_file:
            schema_sql = schema_file.read()
            cursor.execute(schema_sql)
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        logger.info("Schema applied successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to apply schema: {e}")
        return False

def init_db():
    """Initialize the database with the schema"""
    retry_count = 0
    
    # First, try to create the database
    while retry_count < MAX_RETRIES:
        logger.info(f"Attempting to connect to PostgreSQL (attempt {retry_count + 1}/{MAX_RETRIES})...")
        if create_database():
            break
        retry_count += 1
        if retry_count < MAX_RETRIES:
            logger.info(f"Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
        else:
            logger.error("Max retries exceeded. Could not create database.")
            return False
    
    # Then initialize the schema
    retry_count = 0
    while retry_count < MAX_RETRIES:
        logger.info(f"Attempting to apply schema (attempt {retry_count + 1}/{MAX_RETRIES})...")
        if apply_schema():
            logger.info("Database initialization complete!")
            return True
        
        retry_count += 1
        if retry_count < MAX_RETRIES:
            logger.info(f"Retrying in {RETRY_DELAY} seconds...")
            time.sleep(RETRY_DELAY)
        else:
            logger.error("Max retries exceeded. Could not apply schema.")
            return False
    
    return False 