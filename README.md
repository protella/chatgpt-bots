# ChatGPT Bots V2
Python-based AI assistant for Slack using OpenAI's Responses API

## Description
A production-ready Slack bot built with Python and OpenAI's Responses API (not Chat Completions). Features intelligent intent classification, image generation/editing, vision analysis, document processing, and user-specific settings with thread-level customization. The architecture is stateless with Slack as the source of truth, rebuilding context from platform history on demand.

**Note:** Discord support is temporarily unavailable while V2 development focuses on Slack. Discord V2 will be released in a future update.

## Recent Changes

For a detailed list of recent changes and improvements, please see the [CHANGELOG.md](CHANGELOG.md) file.

### ⚠️ Important: Timeout Configuration Update
**Breaking change for streaming responses:** The timeout behavior has been updated to improve reliability. If you previously had `OPENAI_STREAMING_CHUNK_TIMEOUT` set to a low value (e.g., 30 seconds), you must increase it to avoid premature stream termination. Check the updated `.env.example` for recommended values and update your `.env` file accordingly. Low timeout values will now cause responses to drop mid-stream.

## Getting Started

### Requirements
- `Python 3.10+` for structural pattern matching (match/case) support
- `SQLite 3.35+` for JSON support and WAL mode (usually included with Python)

### Model Support
**V2 Architecture**: Uses OpenAI's Responses API exclusively

**Supported Models:**
- **GPT-5** (`gpt-5`) - Reasoning model with web search capability
- **GPT-5 Mini** (`gpt-5-mini`) - Faster reasoning model
- **GPT-4.1** (`gpt-4.1`) - Latest GPT-4 variant
- **GPT-4o** (`gpt-4o`) - Optimized GPT-4 model
- **Image Generation**: `gpt-image-1`
- **Utility Model**: `gpt-5-mini` or `gpt-5-nano` for intent classification  

The setup of a Slack or Discord App is out of scope of this README. There's plenty of documentation online detailing these processes.
  
### Slack quickstart guide: https://api.slack.com/start/quickstart

You can use the included `slack_app_manifest.yml` file to quickly configure your Slack app with all required settings. Simply:
1. Create a new Slack app at https://api.slack.com/apps
2. Choose "From an app manifest" 
3. Select your workspace
4. Paste the contents of `slack_app_manifest.yml`
5. Review and create the app
6. Install to your workspace and copy the tokens to your `.env` file
#### The Slack event subscriptions and scope are as follows:

| Event Name  	| Description                                                       	| Required Scope    	|
|-------------	|-------------------------------------------------------------------	|-------------------	|
| app_mention 	| Subscribe to only the message events that mention your app or bot 	| app_mentions:read 	|
| message.im  	| A message was posted in a direct message channel                  	| im:history        	|    
    
#### Slack Bot Token Scopes:
| Scope                	| Description |
|----------------------	|------------- |
| app_mentions:read    	| Read messages that mention the bot |
| channels:history     	| View messages in public channels |
| channels:join        	| Join public channels |
| chat:write           	| Send messages as the bot |
| chat:write.customize 	| Send messages with custom username/avatar |
| commands             	| Add and respond to slash commands |
| files:read           	| Access files shared in channels |
| files:write          	| Upload and modify files |
| groups:history       	| View messages in private channels |
| im:history           	| View direct message history |
| im:read              	| View direct messages |
| im:write             	| Send direct messages |
| users:read           	| View people in workspace |
| users:read.email     	| View email addresses |

#### Slack Slash Commands:
Configure the following slash command in your Slack app:
- **Command**: `/chatgpt-settings` (production) or `/chatgpt-settings-dev` (development)
- **Request URL**: Your bot's URL endpoint
- **Short Description**: "Configure ChatGPT settings"
- **Usage Hint**: "Opens the settings modal"
- **Set in .env**: `SETTINGS_SLASH_COMMAND=/chatgpt-settings` (or `/chatgpt-settings-dev` for dev)

#### Slack App Shortcuts:
The bot includes a message shortcut for thread-specific settings:
- **Callback ID**: `configure_thread_settings` (production) or `configure_thread_settings_dev` (development)
- **Name**: "Configure Thread Settings" (or similar)
- **Description**: "Configure AI settings for this thread"
- **Where**: Messages -> Message shortcuts menu (three dots on any message)

#### Slack User Token Scopes:
| Scope                	| Description |
|----------------------	|------------- |
| chat:write           	| Send messages on user's behalf |
| users:read           	| View people in workspace |
| users:read.email     	| View email addresses |

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
- _For OCR support on scanned PDFs, install `poppler-utils`: `apt-get install poppler-utils` (Linux) or `brew install poppler` (Mac)_
- _For better DOCX support, optionally install `pandoc`: `apt-get install pandoc` (Linux) or `brew install pandoc` (Mac)_

### Setup `.env` file
- Aquire the necessary keys or tokens from the integration you're using. 
I.e., OpenAI, Slack and Discord tokens.
The only required token is the OPENAI_KEY. The others depend on which integration you're using.

- Create a `.env` file in the root of your venv folder and populate it with your keys, tokens, other vars as follows:

```
See `.env.example` for a complete configuration template with all available settings.
```

### Key Configuration Options

- **Models**: Configure GPT-5, GPT-5 Mini, GPT-4.1, or GPT-4o as your primary model
- **User Settings**: Users can customize their experience via `/chatgpt-settings` command
- **Thread Settings**: Different settings per conversation thread
- **Web Search**: Available with GPT-5 models (requires reasoning_effort >= low)
- **Streaming**: Real-time response streaming with configurable update intervals
- **Logging**: Comprehensive logging with rotation at 10MB, configurable levels per component

### Features

#### Core Capabilities
- **Intelligent Intent Classification**: Automatically determines whether to generate images, analyze uploads, or provide text responses
- **Image Generation & Editing**: Create and modify images with natural language
- **Vision Analysis**: Analyze uploaded images and compare multiple images
- **Document Processing**: Extract and analyze text from PDFs, Office files, and code
- **Web Search**: Current information retrieval (GPT-5 models only)
- **Streaming Responses**: Real-time message updates as responses generate

#### User Experience
- **Settings Modal**: Interactive configuration UI with `/chatgpt-settings`
- **Thread-Specific Settings**: Different configurations per conversation
- **Custom Instructions**: Personalized response styles per user
- **Multi-User Context**: Maintains separate contexts in shared conversations
- **Persistent Settings**: User preferences saved to SQLite database

### MCP (Model Context Protocol) Integration

> **⚠️ BETA FEATURE**: MCP integration is currently in beta. Not all features are fully implemented yet. Notably, the approval UI for tool calls is not available, so `require_approval` is currently ignored and always set to "never" internally. This field is preserved in the configuration for future implementation.

The bot supports OpenAI's native Model Context Protocol, allowing you to connect to specialized data sources and tools for enhanced capabilities.

#### What is MCP?

Model Context Protocol is a standardized way to connect AI applications to external data sources. With MCP, the bot can access:
- Library documentation (e.g., React, Python packages)
- Database queries
- API integrations
- Custom enterprise data sources
- And more...

#### Requirements

- **GPT-5 Model**: MCP tools only work with GPT-5 or GPT-5 Mini
- **HTTP/SSE Transport**: Bot uses OpenAI's native MCP support (stdio not supported)

#### Setup

1. **Create MCP Configuration File**

Copy the example template:
```bash
cp mcp_config.example.json mcp_config.json
```

2. **Configure Your MCP Servers**

Edit `mcp_config.json` with your MCP servers:
```json
{
  "mcpServers": {
    "context7": {
      "server_url": "https://mcp.context7.com/mcp",
      "server_description": "Library documentation and code examples",
      "require_approval": "never"
    },
    "my_database": {
      "server_url": "https://api.example.com/mcp",
      "server_description": "Company database access",
      "authorization": {
        "type": "bearer",
        "token": "YOUR_API_KEY"
      },
      "require_approval": "never",
      "allowed_tools": ["query_customers", "get_orders"]
    }
  }
}
```

3. **Required Fields**:
   - `server_url`: HTTPS endpoint for the MCP server

4. **Optional Fields**:
   - `server_description`: Helps the AI understand when to use this server
   - `authorization`: Authentication credentials (bearer token, API key, etc.)
   - `require_approval`: **IGNORED** - Always set to "never" internally. No approval UI is implemented yet, so other values would cause the bot to hang. This field is preserved in config for future feature development.
   - `allowed_tools`: Whitelist specific tools (omit to allow all)

5. **Restart the Bot**

The bot will automatically load and connect to your configured MCP servers on startup.

#### User Configuration

Users can enable/disable MCP access via the settings modal:
1. Type `/chatgpt-settings` in Slack
2. Check/uncheck "MCP Servers" in the Features section
3. Note: Enabling MCP requires selecting a GPT-5 model

#### Finding MCP Servers

- **Context7**: Library documentation - https://mcp.context7.com
- **MCP Server Directory**: https://modelcontextprotocol.io/servers
- **Build Your Own**: https://modelcontextprotocol.io/quickstart

#### Security Notes

- `mcp_config.json` is in `.gitignore` - never commit API keys
- Only connect to trusted MCP servers
- MCP servers receive conversation context - use appropriate data handling
- Review server permissions before enabling

#### Troubleshooting

**Bot not using MCP tools:**
- Verify `mcp_config.json` exists and is valid JSON
- Check bot logs for MCP initialization errors
- Ensure user has MCP enabled in settings
- Confirm model is GPT-5 or GPT-5 Mini

**MCP server connection errors:**
- Check `server_url` is correct and accessible
- Verify authorization credentials are valid
- Review server logs if you control the MCP server

### Configuration - Memory Cleanup
Thread cleanup runs on a schedule (cron format):
- `CLEANUP_SCHEDULE` - Cron expression (default: "0 0 * * *" for daily at midnight)
- `CLEANUP_MAX_AGE_HOURS` - Remove threads older than this (default: 24 hours)

### Running the bot
Run the py file for your chosen interface, e.g.:

- `python3 slackbot.py` - Run Slack bot
- `python3 main.py --platform slack` - Alternative with platform parameter
- Discord support temporarily unavailable in V2

### Running as a service/daemon
- You can run the script in the background with NOHUP on Linux so you can close the terminal and it will continue to run:
  - `nohup /path/to/venv/chatgpt-bots/bin/python3 slackbot.py &> /path/to/venv/environments/chatgpt-bots/slackbot.log &`
- Put it in your crontab to start on boot:
  - `@reboot cd /path/to/venv/chatgpt-bots && . bin/activate && /path/to/venv/chatgpt-bots/bin/python3 slackbot.py &`
- Use PM2 to manage the script (my pref):
  - `pm2 start /path/to/venv/chatgpt-bots/slackbot.py --name "SlackBot" --interpreter=/path/to/venv/chatgpt-bots/bin/python3 --output=/path/to/venv/chatgpt-bots/slackbot.log --error=/path/to/venv/chatgpt-bots/slackbot.err`
- You could also build a systemd service definition for it.

## Testing

Run the test suite:
```bash
make test           # Run unit tests with coverage
make test-all       # Run all tests including integration
make lint           # Run code quality checks
make format         # Auto-format code
```

## Performance

- Thread-safe with proper lock management
- Automatic token management with smart trimming
- SQLite WAL mode for concurrent database access
- Streaming responses with circuit breaker protection


