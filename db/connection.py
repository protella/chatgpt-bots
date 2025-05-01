"""
Database connection manager for the Slackbot application
"""
import os
import logging
import psycopg2
from psycopg2 import pool

# Configure logging
logger = logging.getLogger("DB_CONNECTION")

# Database connection parameters
DB_HOST = os.environ.get("DB_HOST", "db")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("POSTGRES_DB", "slackbot")
DB_USER = os.environ.get("POSTGRES_USER", "postgres")
DB_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")

# Connection pool
connection_pool = None

def init_connection_pool(min_conn=1, max_conn=10):
    """Initialize the database connection pool"""
    global connection_pool
    
    try:
        if connection_pool is None:
            logger.info("Initializing database connection pool...")
            connection_pool = psycopg2.pool.ThreadedConnectionPool(
                min_conn,
                max_conn,
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            logger.info("Database connection pool initialized")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize connection pool: {e}")
        return False

def get_connection():
    """Get a connection from the pool"""
    global connection_pool
    
    if connection_pool is None:
        init_connection_pool()
    
    try:
        connection = connection_pool.getconn()
        connection.autocommit = True
        return connection
    except Exception as e:
        logger.error(f"Failed to get connection from pool: {e}")
        return None

def release_connection(connection):
    """Release a connection back to the pool"""
    global connection_pool
    
    if connection_pool is not None and connection is not None:
        connection_pool.putconn(connection)

def close_all_connections():
    """Close all connections in the pool"""
    global connection_pool
    
    if connection_pool is not None:
        connection_pool.closeall()
        connection_pool = None
        logger.info("All database connections closed")

def execute_query(query, params=None, fetchone=False, fetchall=False):
    """Execute a query and optionally return results"""
    connection = None
    cursor = None
    result = None
    
    try:
        connection = get_connection()
        cursor = connection.cursor()
        
        cursor.execute(query, params)
        
        if fetchone:
            result = cursor.fetchone()
        elif fetchall:
            result = cursor.fetchall()
        
        return result
    except Exception as e:
        logger.error(f"Query execution error: {e}")
        logger.error(f"Query: {query}")
        logger.error(f"Params: {params}")
        return None
    finally:
        if cursor:
            cursor.close()
        if connection:
            release_connection(connection) 