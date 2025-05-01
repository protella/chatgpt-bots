-- Database schema for the Slackbot application

-- Conversations table to track thread_ts and OpenAI previous_response_id
CREATE TABLE IF NOT EXISTS conversations (
    thread_ts VARCHAR(50) PRIMARY KEY,
    previous_response_id VARCHAR(100),
    channel_id VARCHAR(50) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- User configurations
CREATE TABLE IF NOT EXISTS user_configs (
    user_id VARCHAR(50) PRIMARY KEY,
    temperature FLOAT DEFAULT 0.8,
    top_p FLOAT DEFAULT 1.0,
    max_completion_tokens INTEGER DEFAULT 2048,
    custom_init TEXT DEFAULT '',
    gpt_model VARCHAR(50) DEFAULT 'gpt-4.1-2025-04-14',
    gpt_image_model VARCHAR(50) DEFAULT 'gpt-image-1',
    dalle_model VARCHAR(50) DEFAULT 'dall-e-3',
    image_size VARCHAR(20) DEFAULT '1024x1024',
    image_quality VARCHAR(10) DEFAULT 'hd',
    image_style VARCHAR(20) DEFAULT 'natural',
    image_number INTEGER DEFAULT 1,
    detail VARCHAR(10) DEFAULT 'auto',
    show_revised_prompt BOOLEAN DEFAULT true,
    system_prompt TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Status messages to track and clean up
CREATE TABLE IF NOT EXISTS status_messages (
    message_id VARCHAR(50) PRIMARY KEY,
    thread_ts VARCHAR(50) NOT NULL,
    channel_id VARCHAR(50) NOT NULL,
    message_type VARCHAR(20) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (thread_ts) REFERENCES conversations(thread_ts) ON DELETE CASCADE
);

-- Index for faster lookups
CREATE INDEX IF NOT EXISTS idx_conversations_thread_ts ON conversations(thread_ts);
CREATE INDEX IF NOT EXISTS idx_status_messages_thread_ts ON status_messages(thread_ts); 