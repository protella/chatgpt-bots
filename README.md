# ChatGPT Slack Bot
Python-based ChatGPT Slack bot using OpenAI's Responses API

## Description
A production-ready Slack bot built with Python and OpenAI's Responses API (not Chat Completions). Features intelligent intent classification, image generation/editing, vision analysis, document processing, and user-specific settings with thread-level customization. The architecture is stateless with Slack as the source of truth, rebuilding context from platform history on demand.

## Recent Changes

For a detailed list of recent changes and improvements, please see the [CHANGELOG.md](CHANGELOG.md) file.

### ⚠️ Important: Timeout Configuration Update
**Breaking change for streaming responses:** The timeout behavior has been updated to improve reliability. If you previously had `API_TIMEOUT_STREAMING_CHUNK` set to a low value (e.g., 30 seconds), you must increase it to at least 270 seconds to avoid premature stream termination. Check the updated `.env.example` for recommended values and update your `.env` file accordingly. Low timeout values will cause responses to drop mid-stream.

### ⚠️ Important: Image Settings Update (v2.3.4)
**Breaking change for image generation:** The image model has been updated to `gpt-image-1.5` which uses different quality values. If you have `DEFAULT_IMAGE_QUALITY` set to `hd` or `standard` in your `.env`, you must update it:
```
DEFAULT_IMAGE_QUALITY=auto  # Valid values: auto, low, medium, high
DEFAULT_IMAGE_BACKGROUND=auto  # Valid values: auto, transparent, opaque
```
The old `DEFAULT_IMAGE_STYLE` setting has been removed (was DALL-E 3 only). Old quality values will cause API errors.

### ⚠️ Important: Upgrading to v2.5.0
This release introduces three things you need to know about:

1. **Image model defaults to `gpt-image-2`** (latest OpenAI image model). Update your `.env`:
   ```
   GPT_IMAGE_MODEL=gpt-image-2
   ```
   Existing users are auto-migrated to v2 on first startup. Users can pick `gpt-image-1` per-user in `/settings` if needed.

2. **Dependency layout changed to pip-tools**. Install command:
   ```bash
   make install   # NEW canonical command (uses --require-hashes against the lockfile)
   ```
   `pip install -r requirements.txt` still works but loses hash verification. See [Installing Dependencies](#installing-dependencies) for the new add/upgrade workflow.

3. **`PyPDF2` replaced with `pypdf`** — same `PdfReader` API, transparent migration. If you patched `document_handler.py` locally, the import is now `import pypdf` and the call is `pypdf.PdfReader(...)`.

Database schema changes (new `image_model` column + `settings_completed` backfill) run automatically on startup. Back up `data/slack.db` before deploying.

## Getting Started

### Requirements
- `Python 3.12+` 
- `SQLite 3.35+` for JSON support and WAL mode (usually included with Python)

### Model Support
**V2 Architecture**: Uses OpenAI's Responses API exclusively

**Supported Models:**
- **GPT-5.5** (`gpt-5.5`) - Reasoning model with 1.05M context window, 24-hour prompt caching, and web search
- **Image Generation**: `gpt-image-2`
- **Utility Model**: `gpt-5-mini` for intent classification and helper functions (not user-selectable)

The setup of a Slack App is out of scope of this README. There's plenty of documentation online detailing these processes.
  
### Slack quickstart guide: https://api.slack.com/start/quickstart

You can use the included `slack_app_manifest.example.yml` template (copy it to `slack_app_manifest.yml`, which is gitignored, and customize per environment) to quickly configure your Slack app with all required settings. Simply:
1. Create a new Slack app at https://api.slack.com/apps
2. Choose "From an app manifest"
3. Select your workspace
4. Paste the contents of your customized `slack_app_manifest.yml`
5. Review and create the app
6. **Enable Socket Mode** in your app settings (required - no webhook URLs needed)
7. Generate an App-Level Token with `connections:write` scope
8. Install to your workspace and copy both tokens to your `.env` file:
   - `SLACK_BOT_TOKEN` (starts with `xoxb-`)
   - `SLACK_APP_TOKEN` (starts with `xapp-`)
#### The Slack event subscriptions and scope are as follows:

| Event Name  	| Description                                                       	| Required Scope    	|
|-------------	|-------------------------------------------------------------------	|-------------------	|
| app_mention 	| Subscribe to only the message events that mention your app or bot 	| app_mentions:read 	|
| message.im  	| A message was posted in a direct message channel                  	| im:history        	|    
    
#### Slack Bot Token Scopes:
| Scope                	| Description | Usage |
|----------------------	|------------- |-------|
| app_mentions:read    	| Read messages that mention the bot | Required for @mentions |
| channels:history     	| View messages in public channels | Required for conversations_history/replies |
| channels:join        	| Join public channels | Allows bot to be invited to channels |
| chat:write           	| Send messages as the bot | Required for all message sending |
| chat:write.customize 	| Send messages with custom username/avatar | Reserved for future use |
| commands             	| Add and respond to slash commands | Required for /chatgpt-settings |
| files:read           	| Access files shared in channels | Required for downloading user uploads |
| files:write          	| Upload and modify files | Required for image generation |
| groups:history       	| View messages in private channels | Required for private channel history |
| im:history           	| View direct message history | Required for DM conversations |
| im:read              	| View direct messages | Required to access DMs |
| im:write             	| Send direct messages | Required to send DM responses |
| users:read           	| View people in workspace | Required for user info (display names, timezones) |
| users:read.email     	| View email addresses | Used for user preferences |

#### Slack Slash Commands:
Configure the following slash command in your Slack app:
- **Command**: `/chatgpt-settings` (production) or `/chatgpt-settings-dev` (development)
- **Request URL**: Not required when using Socket Mode
- **Short Description**: "Configure ChatGPT settings"
- **Usage Hint**: "Opens the settings modal"
- **Set in .env**: `SETTINGS_SLASH_COMMAND=/chatgpt-settings` (or `/chatgpt-settings-dev` for dev)

**Note:** Socket Mode handles events automatically without webhook URLs.

#### Slack App Shortcuts:
The bot includes a message shortcut for thread-specific settings:
- **Callback ID**: `configure_thread_settings` (production) or `configure_thread_settings_dev` (development)
- **Name**: "Configure Thread Settings" (or similar)
- **Description**: "Configure AI settings for this thread"
- **Where**: Messages -> Message shortcuts menu (three dots on any message)

#### Note on User Scopes:
The bot uses only **Bot Token** authentication. User scopes listed in the manifest are optional and not utilized by the current implementation. You can safely remove them from your app configuration if desired, or leave them for potential future features.

---


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

Dependencies are managed with [pip-tools](https://github.com/jazzband/pip-tools) in a two-file layout:
- `requirements.in` — human-edited source of truth (top-level deps)
- `requirements.txt` — autogenerated lockfile with exact pins + sha256 hashes for every dep including transitives. **Do not edit by hand.**

To install:
```bash
make install
# equivalent to: python3 -m pip install --require-hashes -r requirements.txt
```

The `--require-hashes` flag verifies every downloaded package against the locked hash — protects against supply-chain tampering and guarantees identical installs across machines.

**Adding or updating a dependency** (contributors only):
```bash
# 1. Edit requirements.in (add/remove/bump a line)
# 2. Regenerate the lockfile:
make lock
# 3. Commit both files together
git add requirements.in requirements.txt
```

To bump everything to the latest versions within the constraints in `requirements.in`:
```bash
make lock-upgrade
```

**Optional Dependencies:**
- _For OCR support on scanned PDFs, install `poppler-utils`: `apt-get install poppler-utils` (Linux) or `brew install poppler` (Mac)_
- _For better DOCX support, optionally install `pandoc`: `apt-get install pandoc` (Linux) or `brew install pandoc` (Mac)_

### Setup `.env` file
1. Copy the example configuration:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and configure the required values:
   - `OPENAI_KEY` - **Required**: Your OpenAI API key
   - `SLACK_BOT_TOKEN` - Required for Slack: Bot token (starts with `xoxb-`)
   - `SLACK_APP_TOKEN` - Required for Slack: App-level token (starts with `xapp-`)
   - Other settings are optional with sensible defaults

See [.env.example](.env.example) for all available configuration options and detailed descriptions.

### Key Configuration Options

- **Models**: GPT-5.5 is the supported primary model
- **User Settings**: Users can customize their experience via `/chatgpt-settings` command
- **Thread Settings**: Different settings per conversation thread
- **Web Search**: Available with all reasoning levels
- **Streaming**: Real-time response streaming with configurable update intervals
- **Token Management**: Automatic context window management with configurable buffer
- **Logging**: Comprehensive logging with rotation at 10MB, configurable levels per component

#### Token Buffer Configuration
The bot manages context window usage automatically using a buffer system:

- `TOKEN_BUFFER_PERCENTAGE` - Percentage of model's context limit to use (default: 0.875 = 87.5%)
  - Applies to the utility model window (gpt-5-mini, 400k); GPT-5.5 uses `GPT54_TOKEN_BUFFER_PERCENTAGE`
  - Lower values (e.g., 0.675 = 67.5%) provide more headroom for system prompts, tools, and reasoning
  - Higher values maximize context retention but may hit limits with complex tool use or reasoning

- `TOKEN_CLEANUP_THRESHOLD` - When to start trimming old messages (default: 0.8 = 80% of buffered limit)
- `TOKEN_TRIM_MESSAGE_COUNT` - Messages to remove per cleanup (default: 5)

**Trade-offs:**
- **Higher buffer** (0.875): More conversation history retained, better context continuity
- **Lower buffer** (0.675): More reliable with MCP tools, web search, and high reasoning efforts

**Recommendation:** Start with 0.875 and lower if you experience token limit errors with tools enabled.

### Features

#### Core Capabilities
- **Intelligent Intent Classification**: Automatically determines whether to generate images, analyze uploads, or provide text responses
- **Image Generation & Editing**: Create and modify images with natural language
- **Vision Analysis**: Analyze uploaded images and compare multiple images
- **Document Processing**: Extract and analyze text from PDFs, Office files, and code
- **Web Search**: Current information retrieval
- **Streaming Responses**: Real-time message updates as responses generate

#### User Experience
- **Settings Modal**: Interactive configuration UI with `/chatgpt-settings`
- **New User Welcome**: Automatic settings modal on first interaction with button-based access
- **Thread-Specific Settings**: Different configurations per conversation via message shortcuts
- **Custom Instructions**: Personalized response styles per user
- **Multi-User Context**: Maintains separate contexts in shared conversations
- **Persistent Settings**: User preferences saved to SQLite database
- **Smart Message Routing**: Ephemeral messages and DMs for settings, keeping channels clean

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

- **GPT-5 Model**: MCP tools require a GPT-5 series model (GPT-5.5)
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
      "headers": {
        "Authorization": "Bearer ${MY_DATABASE_TOKEN}"
      },
      "require_approval": "never",
      "enabled": true,
      "allowed_tools": ["query_customers", "get_orders"]
    }
  }
}
```

3. **Required Fields**:
   - `server_url`: HTTPS endpoint for the MCP server

4. **Optional Fields**:
   - `server_description`: Helps the AI understand when to use this server
   - `headers`: Authentication headers sent to the server. **Keep secrets in `.env`** — `${VAR_NAME}` placeholders are expanded from the environment at load time, and a server with unresolved placeholders is skipped with a warning
   - `enabled`: Set `false` to skip a server without deleting its config
   - `require_approval`: **IGNORED** - Always set to "never" internally. No approval UI is implemented yet, so other values would cause the bot to hang (a warning is logged if you request one). This field is preserved in config for future feature development.
   - `allowed_tools`: Whitelist specific tools (omit to allow all)

> **Security stance**: with approval forced to "never", the model can call any tool an MCP server exposes without user confirmation. Prefer read-only servers, and use `allowed_tools` allowlists to bound what each server can do.

5. **Restart the Bot**

The bot loads your configured MCP servers on startup, runs a background reachability probe (one log line per server), and logs each server's discovered tools as conversations exercise them.

#### User Configuration

Users can enable/disable MCP access via the settings modal:
1. Type `/chatgpt-settings` in Slack
2. Check/uncheck "MCP Servers" in the Features section
3. Note: MCP requires a GPT-5 series model (GPT-5.5)

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
- Confirm model is GPT-5.5

**MCP server connection errors:**
- Check `server_url` is correct and accessible
- Verify authorization credentials are valid
- Review server logs if you control the MCP server

### Configuration - Memory Cleanup
The bot automatically cleans up old thread data from memory to prevent resource buildup. Configure cleanup behavior:
- `CLEANUP_SCHEDULE` - Cron expression for cleanup schedule (default: `0 0 * * *` runs daily at midnight)
- `CLEANUP_MAX_AGE_HOURS` - Remove inactive threads older than this many hours (default: 24)

**Note:** Cleanup only affects in-memory thread state. Database records are preserved, and threads are rebuilt from platform history when needed.

### Running the bot

**First Run:**
The bot will automatically create necessary directories on first startup:
- `data/` - SQLite databases and backups
- `logs/` - Application logs with automatic rotation

**Start the bot:**
- `python3 slackbot.py` - Run Slack bot
- `python3 main.py --platform slack` - Alternative with platform parameter

The bot will connect via Socket Mode and start processing messages immediately.

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


