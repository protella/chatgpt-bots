#!/usr/bin/env python3
"""
Production metrics extraction script for Slack bot usage.
Generates CSV reports with user-friendly names and comprehensive analytics.
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os
from pathlib import Path

# Database path
DB_PATH = "data/slack.db"
OUTPUT_DIR = "metrics_reports"

def get_connection():
    """Create database connection with read-only mode for safety."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    return conn

def resolve_user_names(conn):
    """Get user ID to friendly name mapping from both users table and message content."""
    # First get names from users table
    query = """
    SELECT
        u.user_id,
        COALESCE(u.real_name, u.username, u.user_id) as display_name,
        u.email
    FROM users u
    """
    user_table_names = pd.read_sql_query(query, conn).set_index('user_id')['display_name'].to_dict()

    # Then get channel to user name mapping from message content
    # Messages often start with "User Name: message content"
    query = """
    SELECT DISTINCT
        t.channel_id,
        SUBSTR(m.content, 1, INSTR(m.content, ':') - 1) as extracted_name
    FROM threads t
    JOIN messages m ON m.thread_id = t.thread_id
    WHERE m.role = 'user'
        AND m.content LIKE '%:%'
        AND LENGTH(SUBSTR(m.content, 1, INSTR(m.content, ':') - 1)) < 50
        AND SUBSTR(m.content, 1, INSTR(m.content, ':') - 1) NOT LIKE '[%'
    """

    channel_names = {}
    try:
        df = pd.read_sql_query(query, conn)
        # Group by channel and get most common name for each channel
        for channel_id in df['channel_id'].unique():
            channel_df = df[df['channel_id'] == channel_id]
            # Get the most frequently appearing name for this channel
            name_counts = channel_df['extracted_name'].value_counts()
            if len(name_counts) > 0:
                most_common_name = name_counts.index[0]
                if most_common_name and len(most_common_name.strip()) > 0:
                    channel_names[channel_id] = most_common_name.strip()
    except Exception as e:
        print(f"Warning: Could not extract names from messages: {e}")

    # Combine both dictionaries
    all_names = {**user_table_names, **channel_names}
    print(f"Resolved {len(channel_names)} channel names from messages, {len(user_table_names)} from users table")
    return all_names

def get_per_user_metrics(conn, user_names):
    """Generate per-user activity metrics."""
    query = """
    WITH user_activity AS (
        SELECT
            t.channel_id as user_id,
            COUNT(DISTINCT t.thread_id) as total_threads,
            COUNT(DISTINCT DATE(t.last_activity)) as days_with_activity,
            COUNT(m.id) as total_messages,
            MIN(DATE(t.created_at)) as first_activity,
            MAX(DATE(t.last_activity)) as last_activity
        FROM threads t
        LEFT JOIN messages m ON m.thread_id = t.thread_id AND m.role = 'user'
        GROUP BY t.channel_id
    ),
    user_requests AS (
        SELECT
            t.channel_id as user_id,
            COUNT(DISTINCT i.id) as image_requests,
            COUNT(DISTINCT d.id) as document_requests,
            COUNT(DISTINCT CASE WHEN m.metadata_json LIKE '%vision%' THEN m.id END) as vision_requests
        FROM threads t
        LEFT JOIN images i ON i.thread_id = t.thread_id
        LEFT JOIN documents d ON d.thread_id = t.thread_id
        LEFT JOIN messages m ON m.thread_id = t.thread_id
        GROUP BY t.channel_id
    ),
    user_settings AS (
        SELECT
            up.slack_user_id as user_id,
            up.model,
            up.reasoning_effort,
            up.verbosity,
            up.temperature,
            CASE WHEN up.custom_instructions IS NOT NULL THEN 'Yes' ELSE 'No' END as has_custom_instructions,
            up.enable_web_search,
            up.enable_streaming
        FROM user_preferences up
    )
    SELECT
        ua.user_id,
        ua.total_threads,
        ua.total_messages,
        ua.days_with_activity,
        ROUND(ua.total_messages * 1.0 / NULLIF(ua.total_threads, 0), 2) as avg_messages_per_thread,
        ua.first_activity,
        ua.last_activity,
        ROUND(julianday(ua.last_activity) - julianday(ua.first_activity), 1) as total_days_span,
        COALESCE(ur.image_requests, 0) as image_requests,
        COALESCE(ur.document_requests, 0) as document_requests,
        COALESCE(ur.vision_requests, 0) as vision_requests,
        COALESCE(us.model, 'default') as model,
        COALESCE(us.reasoning_effort, 'medium') as reasoning_effort,
        COALESCE(us.verbosity, 'medium') as verbosity,
        COALESCE(us.temperature, 0.8) as temperature,
        COALESCE(us.has_custom_instructions, 'No') as has_custom_instructions,
        COALESCE(us.enable_web_search, 1) as web_search_enabled,
        COALESCE(us.enable_streaming, 1) as streaming_enabled
    FROM user_activity ua
    LEFT JOIN user_requests ur ON ur.user_id = ua.user_id
    LEFT JOIN user_settings us ON us.user_id = ua.user_id
    ORDER BY ua.total_messages DESC
    """

    df = pd.read_sql_query(query, conn)

    # Map channel IDs to user names if available
    df['user_name'] = df['user_id'].map(lambda x: user_names.get(x, f"User_{x[:8]}"))

    # Reorder and rename columns for clarity
    df = df.rename(columns={
        'days_with_activity': 'days_active',
        'total_days_span': 'account_age_days'
    })

    cols = ['user_name', 'user_id', 'total_threads', 'total_messages', 'days_active',
            'avg_messages_per_thread', 'first_activity', 'last_activity', 'account_age_days',
            'image_requests', 'document_requests', 'vision_requests', 'model',
            'reasoning_effort', 'verbosity', 'temperature', 'has_custom_instructions',
            'web_search_enabled', 'streaming_enabled']

    df = df[cols]

    # Add column descriptions as additional rows
    descriptions = pd.DataFrame([
        ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['COLUMN DESCRIPTIONS:', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['user_name', 'Slack display name', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['user_id', 'Slack DM channel ID', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['total_threads', 'Total conversations started', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['total_messages', 'Total messages sent by user', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['days_active', 'Number of distinct days with activity', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['avg_messages_per_thread', 'Average messages per conversation', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['first_activity', 'Date of first interaction', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['last_activity', 'Date of most recent interaction', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['account_age_days', 'Days between first and last activity', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['image_requests', 'Total image generations/edits', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['document_requests', 'Total documents uploaded', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['vision_requests', 'Total image analysis requests', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['model', 'AI model preference', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['reasoning_effort', 'Reasoning level (low/medium/high)', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['verbosity', 'Response detail level', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['temperature', 'Creativity setting (0.0-2.0)', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['has_custom_instructions', 'User provided custom instructions', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['web_search_enabled', 'Web search feature enabled (1/0)', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['streaming_enabled', 'Streaming responses enabled (1/0)', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['EXAMPLES:', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
        ['Peter Rotella', 'D04PW6LSHEW', '197', '314', '10', '1.59', '2025-09-13', '2025-09-22', '9.0', '80', '12', '31', 'gpt-5', 'medium', 'low', '0.8', 'No', '1', '1']
    ], columns=cols)

    return pd.concat([df, descriptions], ignore_index=True)

def get_user_retention(conn):
    """Calculate user retention metrics."""
    query = """
    WITH user_activity_dates AS (
        SELECT
            t.channel_id as user_id,
            DATE(t.created_at) as activity_date
        FROM threads t
        UNION
        SELECT
            t.channel_id as user_id,
            DATE(m.timestamp) as activity_date
        FROM messages m
        JOIN threads t ON t.thread_id = m.thread_id
        WHERE m.role = 'user'
    ),
    first_last_activity AS (
        SELECT
            user_id,
            MIN(activity_date) as first_seen,
            MAX(activity_date) as last_seen,
            COUNT(DISTINCT activity_date) as days_active
        FROM user_activity_dates
        GROUP BY user_id
    )
    SELECT
        CASE
            WHEN days_active = 1 THEN 'One-time users'
            WHEN julianday('now') - julianday(last_seen) > 7 THEN 'Churned (>7 days inactive)'
            WHEN julianday('now') - julianday(last_seen) <= 1 THEN 'Active (last 24h)'
            WHEN julianday('now') - julianday(last_seen) <= 7 THEN 'Active (last 7d)'
            ELSE 'Other'
        END as user_category,
        COUNT(*) as user_count,
        ROUND(AVG(days_active), 1) as avg_days_active
    FROM first_last_activity
    GROUP BY user_category
    ORDER BY user_count DESC
    """

    return pd.read_sql_query(query, conn)

def get_response_times(conn):
    """Analyze response time patterns."""
    query = """
    WITH message_pairs AS (
        SELECT
            m1.thread_id,
            m1.timestamp as user_time,
            m1.content as user_message,
            MIN(m2.timestamp) as assistant_time,
            m2.content as assistant_response
        FROM messages m1
        JOIN messages m2 ON m1.thread_id = m2.thread_id
            AND m2.timestamp > m1.timestamp
            AND m2.role = 'assistant'
        WHERE m1.role = 'user'
        GROUP BY m1.thread_id, m1.timestamp
    )
    SELECT
        DATE(user_time) as date,
        COUNT(*) as interactions,
        ROUND(AVG((julianday(assistant_time) - julianday(user_time)) * 24 * 60 * 60), 2) as avg_response_seconds,
        ROUND(MIN((julianday(assistant_time) - julianday(user_time)) * 24 * 60 * 60), 2) as min_response_seconds,
        ROUND(MAX((julianday(assistant_time) - julianday(user_time)) * 24 * 60 * 60), 2) as max_response_seconds,
        ROUND(AVG(LENGTH(assistant_response)), 0) as avg_response_length
    FROM message_pairs
    WHERE (julianday(assistant_time) - julianday(user_time)) * 24 * 60 * 60 < 300  -- Filter out outliers > 5 min
    GROUP BY DATE(user_time)
    ORDER BY date DESC
    LIMIT 30
    """

    return pd.read_sql_query(query, conn)

def get_error_rates(conn):
    """Extract error rates from messages."""
    query = """
    WITH error_messages AS (
        SELECT
            DATE(timestamp) as date,
            COUNT(*) as total_messages,
            SUM(CASE WHEN content LIKE '%error%' OR content LIKE '%Error%'
                     OR content LIKE '%failed%' OR content LIKE '%Failed%' THEN 1 ELSE 0 END) as error_count,
            SUM(CASE WHEN content LIKE '%rate limit%' OR content LIKE '%429%' THEN 1 ELSE 0 END) as rate_limit_errors,
            SUM(CASE WHEN content LIKE '%timeout%' OR content LIKE '%Timeout%' THEN 1 ELSE 0 END) as timeout_errors
        FROM messages
        WHERE role = 'assistant'
        GROUP BY DATE(timestamp)
    )
    SELECT
        date,
        total_messages,
        error_count,
        ROUND(error_count * 100.0 / total_messages, 2) as error_rate_pct,
        rate_limit_errors,
        timeout_errors
    FROM error_messages
    WHERE date >= date('now', '-30 days')
    ORDER BY date DESC
    """

    return pd.read_sql_query(query, conn)

def get_model_usage(conn):
    """Analyze model usage patterns."""
    query = """
    SELECT
        COALESCE(up.model, 'gpt-5') as model,
        COUNT(DISTINCT up.slack_user_id) as users,
        COUNT(DISTINCT t.thread_id) as threads_affected
    FROM user_preferences up
    LEFT JOIN threads t ON t.channel_id = up.slack_user_id
    GROUP BY up.model

    UNION ALL

    SELECT
        'threads_with_custom_config' as model,
        COUNT(DISTINCT thread_id) as users,
        0 as threads_affected
    FROM threads
    WHERE config_json IS NOT NULL AND config_json != ''
    """

    return pd.read_sql_query(query, conn)

def get_web_search_utilization(conn):
    """Analyze web search feature utilization."""
    query = """
    WITH search_usage AS (
        SELECT
            up.slack_user_id as user_id,
            up.enable_web_search,
            COUNT(DISTINCT t.thread_id) as thread_count,
            COUNT(m.id) as message_count,
            SUM(CASE WHEN m.content LIKE '%search%' OR m.content LIKE '%google%'
                     OR m.content LIKE '%web%' THEN 1 ELSE 0 END) as potential_search_requests
        FROM user_preferences up
        LEFT JOIN threads t ON t.channel_id = up.slack_user_id
        LEFT JOIN messages m ON m.thread_id = t.thread_id AND m.role = 'user'
        GROUP BY up.slack_user_id, up.enable_web_search
    )
    SELECT
        CASE enable_web_search
            WHEN 1 THEN 'Web Search Enabled'
            ELSE 'Web Search Disabled'
        END as setting,
        COUNT(*) as user_count,
        SUM(thread_count) as total_threads,
        SUM(message_count) as total_messages,
        SUM(potential_search_requests) as search_related_requests
    FROM search_usage
    GROUP BY enable_web_search
    """

    return pd.read_sql_query(query, conn)

def get_daily_summary(conn):
    """Generate daily activity summary."""
    query = """
    WITH daily_stats AS (
        SELECT
            DATE(m.timestamp) as date,
            COUNT(DISTINCT t.channel_id) as unique_users,
            COUNT(DISTINCT m.thread_id) as active_threads,
            COUNT(CASE WHEN m.role = 'user' THEN 1 END) as user_messages,
            COUNT(CASE WHEN m.role = 'assistant' THEN 1 END) as bot_responses
        FROM messages m
        JOIN threads t ON t.thread_id = m.thread_id
        WHERE m.timestamp >= date('now', '-30 days')
        GROUP BY DATE(m.timestamp)
    ),
    daily_requests AS (
        SELECT
            DATE(created_at) as date,
            COUNT(*) as images_generated
        FROM images
        WHERE created_at >= date('now', '-30 days')
        GROUP BY DATE(created_at)
    ),
    daily_docs AS (
        SELECT
            DATE(created_at) as date,
            COUNT(*) as documents_processed
        FROM documents
        WHERE created_at >= date('now', '-30 days')
        GROUP BY DATE(created_at)
    )
    SELECT
        ds.date,
        ds.unique_users,
        ds.active_threads as threads_with_activity,
        ds.user_messages,
        ds.bot_responses,
        COALESCE(dr.images_generated, 0) as images_generated,
        COALESCE(dd.documents_processed, 0) as documents_processed,
        ds.user_messages + COALESCE(dr.images_generated, 0) + COALESCE(dd.documents_processed, 0) as total_user_requests
    FROM daily_stats ds
    LEFT JOIN daily_requests dr ON dr.date = ds.date
    LEFT JOIN daily_docs dd ON dd.date = ds.date
    ORDER BY ds.date DESC
    """

    df = pd.read_sql_query(query, conn)

    # Add column descriptions
    descriptions = pd.DataFrame([
        ['', '', '', '', '', '', '', ''],
        ['COLUMN DESCRIPTIONS:', '', '', '', '', '', '', ''],
        ['date', 'Calendar date (YYYY-MM-DD)', '', '', '', '', '', ''],
        ['unique_users', 'Distinct users active that day', '', '', '', '', '', ''],
        ['threads_with_activity', 'Conversations with messages that day', '', '', '', '', '', ''],
        ['user_messages', 'Messages sent by users', '', '', '', '', '', ''],
        ['bot_responses', 'Replies from the bot', '', '', '', '', '', ''],
        ['images_generated', 'Image generation/edit requests', '', '', '', '', '', ''],
        ['documents_processed', 'Files uploaded for analysis', '', '', '', '', '', ''],
        ['total_user_requests', 'Sum of user messages + images + docs', '', '', '', '', '', ''],
        ['', '', '', '', '', '', '', ''],
        ['EXAMPLE:', '', '', '', '', '', '', ''],
        ['2025-09-22', '26', '89', '154', '152', '5', '2', '161']
    ], columns=df.columns)

    return pd.concat([df, descriptions], ignore_index=True)

def main():
    """Main execution function."""
    # Create output directory
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Connect to database
    conn = get_connection()

    # Get user name mappings
    print("Resolving user names...")
    user_names = resolve_user_names(conn)

    # Generate only the two requested reports
    reports = {
        "per_user_metrics": get_per_user_metrics(conn, user_names),
        "daily_summary": get_daily_summary(conn)
    }

    # Save to CSV files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name, df in reports.items():
        filename = f"{OUTPUT_DIR}/slack_metrics_{name}_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"Saved: {filename} ({len(df)} rows)")

    # Create a combined summary (exclude description rows for calculations)
    print("\n=== QUICK SUMMARY ===")
    print(f"Total Users: {len(user_names)}")

    # Get only the data rows (exclude description rows which have empty first column)
    per_user_data = reports['per_user_metrics']
    data_rows = per_user_data[per_user_data['user_name'] != ''].copy()
    data_rows = data_rows[~data_rows['user_name'].str.contains('COLUMN|EXAMPLE', na=False)]

    # Convert numeric columns to proper types for calculations
    numeric_cols = ['total_threads', 'total_messages', 'image_requests', 'document_requests']
    for col in numeric_cols:
        data_rows[col] = pd.to_numeric(data_rows[col], errors='coerce')

    print(f"Total Threads: {data_rows['total_threads'].sum():.0f}")
    print(f"Total Messages: {data_rows['total_messages'].sum():.0f}")
    print(f"Total Images: {data_rows['image_requests'].sum():.0f}")
    print(f"Total Documents: {data_rows['document_requests'].sum():.0f}")

    conn.close()
    print(f"\n✅ All reports saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()