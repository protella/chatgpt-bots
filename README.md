# ChatGPT Bots
Python based ChatGPT bot integrations

## Description
ChatBot Integrations for Slack and Discord using Python and OpenAI's Responses API. This bot V2 is designed for GPT-5 models and supports image generation with gpt-image-1 (DALL-E 3 coming soon). The bot uses intelligent intent classification to determine when to generate images vs text responses. Conversations are stateless with Slack/Discord as the source of truth, rebuilding context from platform history on demand. Discord client V2 is under development.

## Recent Changes

For a detailed list of recent changes and improvements, please see the [CHANGELOG.md](CHANGELOG.md) file.

## Getting Started

Requires `Python 3.10+` as the script takes advantage of the new structural pattern matching (match/case) in this version.

### Model Support
**V2 Architecture**: Uses OpenAI's Responses API (not Chat Completions)

**Supported Models:**
- GPT-5 reasoning models (`gpt-5-mini`, `gpt-5-nano`) - Fixed `temperature=1.0`, no `top_p`
- GPT-5 chat models (`gpt-5-chat-latest`) - Standard temperature/top_p support
- Image generation: `gpt-image-1` (default), DALL-E 3 (coming soon)
- Utility model for intent classification: `gpt-5-mini` or `gpt-5-nano`  

The setup of a Slack or Discord App is out of scope of this README. There's plenty of documentation online detailing these processes.
  
### Slack quickstart guide: https://api.slack.com/start/quickstart
#### The Slack event subscriptions and scope are as follows:

| Event Name  	| Description                                                       	| Required Scope    	|
|-------------	|-------------------------------------------------------------------	|-------------------	|
| app_mention 	| Subscribe to only the message events that mention your app or bot 	| app_mentions:read 	|
| message.im  	| A message was posted in a direct message channel                  	| im:history        	|    
    
#### Slack OAuth & Permissions (Scopes):
| Scope                	|
|----------------------	|
| app_mentions:read    	|
| channels:history     	|
| channels:join        	|
| chat:write           	|
| chat:write.customize 	|
| commands             	|
| files:read           	|
| files:write          	|
| im:history           	|
| im:read              	|
| im:write             	|

---

### Discord OAuth2 
<img src="Docs/Discord_OAuth2.png" alt="image" width="40%" height="auto">

---

### Install `venv` module if you don't already have it
`python3 -m pip install --user venv`

### Clone the repository
`git clone https://github.com/protella/chatgpt-bots`

### Create and Activate the Virtual Environment
```
cd chatgpt-bots
python3 -m venv chatbots
source chatbots/bin/activate
```

### Installing Dependencies:
```python3 -m pip install -U -r requirements.txt```

- _Note: The included `requirements.txt` file includes all of the dependencies for all 3 clients in this repo._

### Setup `.env` file
- Aquire the necessary keys or tokens from the integration you're using. 
I.e., OpenAI, Slack and Discord tokens.
The only required token is the OPENAI_KEY. The others depend on which integration you're using.

- Create a `.env` file in the root of your venv folder and populate it with your keys, tokens, other vars as follows:

```
# Required
SLACK_BOT_TOKEN = 'YOURTOKENHERE'
SLACK_APP_TOKEN = 'YOURTOKENHERE'
OPENAI_KEY = 'YOURTOKENHERE'

# Model Configuration
GPT_MODEL = 'gpt-5-chat-latest'
GPT_IMAGE_MODEL = 'gpt-image-1'
UTILITY_MODEL = 'gpt-5-mini-2025-08-07'
DALLE_MODEL = 'dall-e-3'

# Default Generation Parameters
DEFAULT_TEMPERATURE = '0.8'  # 0.0-2.0
DEFAULT_MAX_TOKENS = '4096'
DEFAULT_TOP_P = '1.0'  # 0.0-1.0
DEFAULT_REASONING_EFFORT = 'medium'  # minimal/low/medium/high
DEFAULT_VERBOSITY = 'medium'  # low/medium/high

# Image Generation Defaults  
DEFAULT_IMAGE_SIZE = '1024x1024'  # 1024x1024/1024x1792/1792x1024
DEFAULT_IMAGE_QUALITY = 'hd'  # standard/hd
DEFAULT_IMAGE_STYLE = 'natural'  # natural/vivid
DEFAULT_IMAGE_NUMBER = '1'
DEFAULT_INPUT_FIDELITY = 'high'  # high/low - high preserves original

# Vision Defaults
DEFAULT_DETAIL_LEVEL = 'auto'  # auto/low/high

# UI Configuration  
THINKING_EMOJI = ':hourglass_flowing_sand:'

# Discord (V2 under development)
DISCORD_TOKEN = 'YOURTOKENHERE'
DISCORD_ALLOWED_CHANNEL_IDS = '1234567890, 1234567890'

# Logging Configuration
SLACK_LOG_LEVEL = "WARNING"
DISCORD_LOG_LEVEL = "WARNING"
BOT_LOG_LEVEL = "WARNING"
UTILS_LOG_LEVEL = "WARNING"
CONSOLE_LOGGING_ENABLED = "TRUE"
LOG_DIRECTORY = "logs"

# Cleanup Configuration (cron format)
CLEANUP_SCHEDULE = "0 0 * * *"  # Daily at midnight
CLEANUP_MAX_AGE_HOURS = "24"
```

### Logging Configuration
The bots support a comprehensive logging system that can be configured through the environment variables shown above. Logs are stored in the `logs` directory with automatic rotation when they reach 10MB in size.

### Configuration - Bot Prompt Tuning
The `prompts.py` file contains the various system prompts the script will use to set the tone for how the bot will respond. Telling it that it is a chatbot and with any specific style of responses will help with more appropriate responses.

### Configuration - Memory Cleanup
Thread cleanup runs on a schedule (cron format):
- `CLEANUP_SCHEDULE` - Cron expression (default: "0 0 * * *" for daily at midnight)
- `CLEANUP_MAX_AGE_HOURS` - Remove threads older than this (default: 24 hours)

### Running the bot
Run the py file for your chosen interface, e.g.:

- `python3 slackbot.py` - Run Slack bot
- `python3 discordbot.py` - (V2 under development)
- `python3 main.py --platform slack` - Alternative with platform parameter

### Running as a service/daemon
- You can run the script in the background with NOHUP on Linux so you can close the terminal and it will continue to run:
  - `nohup /path/to/venv/chatgpt-bots/bin/python3 slackbot.py &> /path/to/venv/environments/chatgpt-bots/slackbot.log &`
- Put it in your crontab to start on boot:
  - `@reboot cd /path/to/venv/chatgpt-bots && . bin/activate && /path/to/venv/chatgpt-bots/bin/python3 slackbot.py &`
- Use PM2 to manage the script (my pref):
  - `pm2 start /path/to/venv/chatgpt-bots/slackbot.py --name "SlackBot" --interpreter=/path/to/venv/chatgpt-bots/bin/python3 --output=/path/to/venv/chatgpt-bots/slackbot.log --error=/path/to/venv/chatgpt-bots/slackbot.err`
- You could also build a systemd service definition for it.

## V2 Implementation Status:

### Completed ✅
- Client-based architecture with abstract base class
- Stateless design with platform as source of truth
- Thread state rebuilding from Slack history
- Intent classification for image vs text responses
- Image generation and editing with context awareness
- Vision analysis for uploaded images
- Thread locking for concurrent request handling
- Configurable cleanup with cron scheduling
- Error formatting with emojis and code blocks
- Markdown to Slack mrkdwn conversion
- Unsupported file type notifications

### Pending Implementation 🚧
- Response streaming (experimental)
- Discord V2 client
- Multi-workspace support
- Rate limiting and usage tracking 


