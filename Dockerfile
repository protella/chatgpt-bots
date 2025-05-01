FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Make scripts executable
RUN chmod +x init_db.py

# Command to run the application
CMD ["sh", "-c", "python init_db.py && python slackbot.py"] 