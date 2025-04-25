FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN groupadd -r botuser && useradd -r -g botuser botuser

# Create data and cache directories and set permissions
RUN mkdir -p /data && \
    mkdir -p /app/.pytest_cache && \
    mkdir -p /app/logs && \
    chown -R botuser:botuser /data /app/.pytest_cache /app/logs

# Copy application code
COPY . .

# Create a start script
RUN echo '#!/bin/bash\npython app/clients/slack/slack_bot.py' > /app/start.sh && \
    chmod +x /app/start.sh

# Switch to non-root user
USER botuser

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command (can be overridden in docker-compose.yml)
CMD ["python", "app/clients/slack/slack_bot.py"] 