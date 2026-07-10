# ChatGPT Slack Bot
Python-based ChatGPT Slack bot using OpenAI's Responses API

## Description
A production-ready Slack bot built with Python and OpenAI's Responses API (not Chat Completions). Features intelligent intent classification, image generation/editing, vision analysis, document processing, user-specific settings with thread-level customization, and an optional channel-teammate mode where the bot participates thoughtfully in channels it's invited to. The architecture is stateless with Slack as the source of truth, rebuilding context from platform history on demand.

## Recent Changes

For a detailed list of recent changes and improvements, please see the [CHANGELOG.md](CHANGELOG.md) file.

### ⚠️ Important: Upgrading to v3.0.0

v3 is a major release: the **GPT-5.6 model family** (Sol/Terra/Luna) replaces the old
lineup, the bot can act as a **channel teammate** (off by default — no behavior change
until you enable it), and **conversation history now lives in Slack, not the database**.

The short version of the upgrade:
1. `make install` (dependencies)
2. Update `.env` — three changed values (`GPT_MODEL=gpt-5.6-sol`,
   `UTILITY_MODEL=gpt-5.6-luna`, `UTILITY_REASONING_EFFORT=none`), a few deletions,
   and a new optional "Channel participation & UX" section (all sane defaults)
3. Optionally move MCP API keys from `mcp_config.json` into `.env` (`${VAR}` placeholders)
4. Rebuild your Slack app manifest from `slack_app_manifest.example.yml` and reinstall
5. Start the bot — three automatic DB migrations run once, each taking a tagged
   backup into `data/backups/` first

The full step-by-step list (including exact `.env` keys, manifest deltas, and the
migration log lines to watch for) is in the
[CHANGELOG's Upgrade Instructions](CHANGELOG.md). Upgrading from before v2.5?
Read the older upgrade callouts in the CHANGELOG history first.

## Getting Started

### Requirements
- `Python 3.12+` 
- `SQLite 3.35+` for JSON support and WAL mode (usually included with Python)

### Model Support
**V3 Architecture**: Uses OpenAI's Responses API exclusively

**Supported Models** (all with a 1.05M context window and prompt caching):
- **GPT-5.6 Sol** (`gpt-5.6-sol`) - Flagship reasoning model, the default
- **GPT-5.6 Terra** (`gpt-5.6-terra`) - Balanced tier
- **GPT-5.6 Luna** (`gpt-5.6-luna`) - Fast/light tier; also runs the bot's internal
  utility functions (intent classification, summaries)
- **GPT-5.5** (`gpt-5.5`) - Previous flagship, still selectable
- **Image Generation**: `gpt-image-2`

Users pick their model and reasoning effort in `/chatgpt-settings`, and can override
both per channel and per thread. The `max` reasoning effort is available on the 5.6
family; the settings modal adapts the effort list to the selected model.

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
#### Slack events and scopes

The manifest template is the authoritative list — it carries everything the bot can
use. The highlights, grouped by what they power:

| Capability | Events | Scopes |
|---|---|---|
| Mentions & DMs (core) | `app_mention`, `message.im` | `app_mentions:read`, `im:history`, `im:read`, `im:write`, `chat:write` |
| Channel listening (optional, flag-gated) | `message.channels`, `message.groups`, `message.mpim` | `channels:history`, `groups:history`, `mpim:history`, `channels:read`, `groups:read`, `mpim:read` |
| Reactions (give + observe) | `reaction_added` | `reactions:write`, `reactions:read`, `emoji:read` |
| Agent/assistant surface | `app_home_opened`, `app_context_changed` (legacy `assistant_thread_*` kept during transition) | `assistant:write` |
| Workspace search tool | — | `search:read.public`, `search:read.private` (plus `.im`/`.mpim`/`.files`/`.users` if you widen the search surface) |
| Files, settings, misc | — | `files:read`, `files:write`, `commands`, `users:read`, `users:read.email`, `channels:join`, `chat:write.customize` |

Optional: subscribe `reaction_removed` if you want 👍/👎 reactions un-counted from
feedback when someone removes one.

Don't want a capability? Drop its scopes/events from your manifest — everything
channel-teammate-related is also feature-flagged in `.env` and off by default.

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

- **Models**: GPT-5.6 Sol is the default; Terra, Luna, and GPT-5.5 are selectable
- **User Settings**: Users can customize their experience via `/chatgpt-settings` command
- **Thread Settings**: Different settings per conversation thread
- **Channel participation**: `ENABLE_CHANNEL_LISTENING` (default **off**) is the master
  switch for teammate behavior; per-channel levels via the ⚙️ Configure button. See the
  "Channel participation & UX" section of `.env.example` for the full knob list.
- **Web Search**: Available with all reasoning levels
- **Streaming**: Real-time response streaming with configurable update intervals
  (native Slack streaming is built in behind `SLACK_NATIVE_STREAMING`, off by default)
- **Status messages**: while the bot works, the thread status bubble rotates through
  loading messages and per-stage progress texts — both customizable, see
  "Customizing status messages" below
- **Token Management**: Automatic context management with rolling thread compaction
- **Logging**: Comprehensive logging with rotation at 10MB, configurable levels per component

#### Token Buffer Configuration
The bot manages context window usage automatically using a buffer system:

- `TOKEN_BUFFER_PERCENTAGE` - Percentage of model's context limit to use (default: 0.875 = 87.5%)
  - Applies to the legacy utility model window (`GPT5_MAX_TOKENS`); the chat models use `GPT54_TOKEN_BUFFER_PERCENTAGE` (name kept for compatibility — it describes the 1.05M window)
  - Lower values (e.g., 0.675 = 67.5%) provide more headroom for system prompts, tools, and reasoning
  - Higher values maximize context retention but may hit limits with complex tool use or reasoning

- `TOKEN_CLEANUP_THRESHOLD` - When to start compacting the thread (default: 0.8 = 80% of buffered limit)
- `TOKEN_COMPACTION_TARGET` - How far a compaction pass shrinks the thread (default: 0.7 = down to 70% of the limit). Compacted spans roll into a summary that preserves file and image references — nothing is silently dropped.

**Trade-offs:**
- **Higher buffer** (0.875): More conversation history retained, better context continuity
- **Lower buffer** (0.675): More reliable with MCP tools, web search, and high reasoning efforts

**Recommendation:** Start with 0.875 and lower if you experience token limit errors with tools enabled.

#### Customizing status messages
While working, the bot shows Slack's native status bubble in the thread and rotates
through short "working…" messages. Everything is plain text (the status surface
doesn't render emoji or `:shortcodes:`) and safe to customize:

- **Loading messages** (shown while thinking): a bundled pool of 100 generic ones ships
  in `status_messages/loading_messages.generic.txt`. To brand them, copy that file,
  rewrite the lines (one message per line, `#` comments allowed), and point
  `STATUS_LOADING_MESSAGES_FILE` in `.env` at your copy. Keep company-specific files
  out of git if your repo is shared — add them to `.gitignore` like `.env`.
  For a short list without a file, set `STATUS_LOADING_MESSAGES=msg one…,msg two…`
  inline instead (it takes precedence over the file).
- **Pipeline stage messages** (shown during specific steps like generating an image or
  reading a document): `status_messages/pipeline_messages.txt` holds several phrasings
  per `[stage]` section and the bot picks one at random each time. Edit in place or
  point `PIPELINE_MESSAGES_FILE` at your own copy. `{file_name}`/`{count}` placeholders
  are filled in automatically where a stage uses them.

Misconfigured or missing files never break anything — the bot falls back to its
built-in texts.

### Features

#### Core Capabilities
- **Intelligent Intent Classification**: Automatically determines whether to generate images, analyze uploads, or provide text responses
- **Image Generation & Editing**: Create and modify images with natural language
- **Vision Analysis**: Analyze uploaded images and compare multiple images
- **Document Processing**: Uploads become concise summaries in the conversation; the bot re-reads the original from Slack on demand when asked for specifics. Document content is never stored and never touches disk — delete a file in Slack and it's genuinely gone from the bot's reach.
- **Web Search**: Current information retrieval
- **Streaming Responses**: Real-time message updates as responses generate
- **On-Demand Context**: The bot can fetch older history and search the workspace (permission-scoped) when a conversation references something it can't see

#### Channel Teammate (optional, off by default)
Flip `ENABLE_CHANNEL_LISTENING=true` and the bot behaves like a thoughtful colleague
in channels it's invited to:
- **Knows when to speak**: responds when it can genuinely help, reacts with an emoji when words would be noise, and stays out of human-to-human conversation — with a hard hourly cap on unprompted replies
- **Takes feedback**: "quiet down" earns a 🤐 and a 4-hour snooze (mentions still work); standing preferences like "stay out unless tagged" are remembered durably
- **Per-channel control for everyone**: any member can set the participation level (off / mentions-only / judicious / active), channel directives, and reply placement via the ⚙️ Configure button under bot responses
- **Per-channel memory**: durable facts (decisions, conventions, preferences) remembered and recalled in later conversations — managed by the bot's own judgment, viewable and correctable
- **No busy rejections**: messages that arrive mid-response are queued and answered together in one catch-up reply

#### User Experience
- **Settings Modal**: Interactive configuration UI with `/chatgpt-settings`
- **New User Welcome**: Automatic settings modal on first interaction with button-based access
- **Thread-Specific Settings**: Different configurations per conversation via message shortcuts
- **Custom Instructions**: Personalized response styles per user
- **Multi-User Context**: Maintains separate contexts in shared conversations
- **Feedback Buttons**: 👍/👎 under DM responses (once per thread; thumbs reactions on any bot message count too) — recorded locally for tuning
- **Personable Progress**: a single native status bubble with rotating, customizable loading messages and varied per-stage progress texts (see "Customizing status messages")
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

- **HTTP/SSE Transport**: Bot uses OpenAI's native MCP support (stdio not supported)
- All supported chat models can use MCP tools

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
- Check bot logs for MCP initialization errors and the startup health-probe lines
- Ensure user has MCP enabled in settings
- If a server was skipped at load, the log names the unresolved `${VAR}` — add it to `.env`

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


