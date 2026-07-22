# ChatGPT Slack Bot

A Slack bot built on OpenAI's **Responses API** (not Chat Completions). It chats, generates and edits
images, reads what you upload, searches the web, runs multi-source research jobs in the background —
and, optionally, behaves like a teammate in the channels it's invited to.

The architecture is stateless: **Slack is the source of truth.** Conversation context is rebuilt from
Slack history on demand, never mirrored into the database.

## Features

### Core
- **Conversation** — full thread context, rebuilt from Slack every turn (never truncated to "the last
  N messages"); long threads roll into compacted summaries rather than being silently dropped
- **Images** — generation and editing from natural language (`gpt-image-2`). Generation runs in the
  background, so the thread keeps moving; several can cook at once
- **Vision** — analyzes and compares uploaded images
- **Documents** — an upload becomes a short summary in the conversation, and the bot re-reads the
  original from Slack on demand when you ask for specifics. Scanned/image-only PDFs are OCR'd so they
  stay readable on later turns. Content is never stored and never touches disk — delete the file in
  Slack and it's genuinely gone from the bot's reach
- **Web search** — available at every reasoning level
- **Deep research** — for questions that deserve more than a quick lookup, the bot detaches a
  background job, posts a live status card while it works, and delivers a sourced report minutes
  later. You can keep chatting the whole time
- **On-demand context** — fetches older history, searches the workspace (permission-scoped), links to
  earlier messages ("link me to where we decided X"), and looks up channel info, pins, reactions, and
  people. All fetched live when needed, never stored
- **Reactions** — the bot can answer with an emoji when words would be noise, and reacts like a
  colleague (a laugh, a 👍 on good news)
- **Time awareness** — every message it reads is stamped with when it was said, so "last night" and
  "you asked an hour ago" mean something

### Channel teammate (optional — `ENABLE_CHANNEL_LISTENING`, default **off**)
Flip it on and the bot participates in channels it's invited to:
- **Knows when to speak** — replies when it can genuinely help, reacts when an emoji says it better,
  and stays out of human-to-human conversation. Pacing is the model's judgment, not a numeric quota
- **Takes feedback** — tell it to pipe down in a thread and it does, permanently, for that thread
  (mentions and name-summons still work) and remembers the preference channel-wide
- **Per-channel control for anyone** — participation level (off / mentions-only / judicious /
  active), channel directives, reply placement, and the channel's model/effort/verbosity, all set via
  the ⚙️ Configure button under any bot reply
- **Per-channel memory** — durable facts (decisions, conventions, preferences) recalled in later
  conversations; the bot manages them itself, and you can view and correct them
- **No busy rejections** — messages that arrive mid-response are queued and answered together

### User experience
- **Settings modal** (`/chatgpt-settings`) — model, reasoning effort, verbosity, image defaults,
  custom instructions; per-user, per-channel, and per-thread overrides
- **Feedback** — 👍/👎 under DM responses, and thumbs reactions on any bot message count too
- **Live progress** — a native status bubble with rotating, customizable "working…" messages, and a
  ticking checklist on image jobs
- **Multi-user aware** — keeps track of who said what in shared threads

## Recent Changes

See [CHANGELOG.md](CHANGELOG.md).

### ⚠️ Upgrading to v3.0.0

v3 is a major release: a new model lineup (**GPT-5.6 Sol/Terra/Luna**), an optional **channel
teammate** mode (off by default — no behavior change until you enable it), **background deep
research**, and **conversation history moved out of the database into Slack**.

The upgrade is: `make install` → install the system packages below → update a few `.env` values →
rebuild your Slack app manifest → start the bot (one-time DB migrations run automatically, taking
tagged backups first). The exact keys, manifest deltas, and migration log lines are in the
[CHANGELOG's Upgrade Instructions](CHANGELOG.md) — follow them in order.

## Getting Started

### Requirements
- **Python 3.12** (what the bot is developed and tested on)
- **SQLite 3.35+** — required by the v3 document-content migration (`ALTER TABLE … DROP COLUMN`).
  On older SQLite the migration logs a warning and leaves document text in the DB, which violates the
  project's no-content-at-rest rule. Check with `python3 -c "import sqlite3; print(sqlite3.sqlite_version)"`
- **System packages** (document handling; all three are optional and degrade gracefully, but the bot
  quietly loses capability without them):

  ```bash
  apt-get install poppler-utils tesseract-ocr pandoc     # Linux
  brew install poppler tesseract pandoc                  # macOS
  ```

  `poppler` renders PDF pages to images, `tesseract` OCRs scanned PDFs (`ENABLE_PDF_OCR`, on by
  default), `pandoc` is the fallback extractor for `.docx`. Missing any of them turns the affected
  documents into an honest "couldn't extract text" note rather than an error.

### Models
All chat models share a 1.05M-token context window and prompt caching. Users pick theirs in
`/chatgpt-settings`; a channel or a single thread can override it.

| Model | Role |
|---|---|
| `gpt-5.6-sol` | Flagship reasoning model — **the default** |
| `gpt-5.6-terra` | Balanced tier |
| `gpt-5.6-luna` | Fast tier; also runs the bot's internal utility calls (classification, summaries) |
| `gpt-5.5` | Previous flagship, still selectable |
| `gpt-image-2` | Image generation and editing |

Reasoning effort runs `none → low → medium → high → xhigh → max` on the 5.6 family (`max` is 5.6-only;
the settings modal adapts the list to the chosen model).

### Slack app setup
Create the app from a manifest: copy `slack_app_manifest.example.yml` to `slack_app_manifest.yml`
(gitignored — customize the name and slash command per environment), then
[api.slack.com/apps](https://api.slack.com/apps) → **Create New App** → **From an app manifest** →
paste it. Then:

1. Enable **Socket Mode** (required — no public webhook URLs needed)
2. Generate an App-Level Token with `connections:write`
3. Install to the workspace and copy `SLACK_BOT_TOKEN` (`xoxb-`) and `SLACK_APP_TOKEN` (`xapp-`) into `.env`

The manifest is the authoritative scope list. What each group buys you:

| Capability | Scopes / events |
|---|---|
| Mentions & DMs (core) | `app_mentions:read`, `im:history`, `im:read`, `im:write`, `chat:write` · events `app_mention`, `message.im` |
| Channel listening (optional) | `channels:history`, `groups:history`, `mpim:history`, `channels:read`, `groups:read`, `mpim:read` · events `message.channels`, `message.groups`, `message.mpim` |
| Reactions (give + observe) | `reactions:write`, `reactions:read` · events `reaction_added`, `reaction_removed` |
| Agent surface | `assistant:write` · events `app_home_opened`, `app_context_changed` (legacy `assistant_thread_*` kept during the transition) |
| Workspace search | `search:read.public`, `search:read.private` (add `.im`/`.mpim`/`.files`/`.users` to widen it) |
| People & context lookups | `users:read`, `users:read.email`, `pins:read` |
| Research byline | `chat:write.customize` — the "[research: …]" label on findings posts; without it the bot falls back to plain posts |
| Files, settings, misc | `files:read`, `files:write`, `commands`, `channels:join` |

The bot uses **bot-token auth only** — no user scopes. Don't want a capability? Drop its scopes and
events; everything channel-teammate-related is also feature-flagged in `.env` and off by default.

**Slash command:** `/chatgpt-settings` (set `SETTINGS_SLASH_COMMAND` to match; use a `-dev` suffix for
a dev install). No Request URL needed under Socket Mode.
**Message shortcut:** callback id `configure_thread_settings` — per-thread settings from any message's
⋯ menu.

### Install

```bash
git clone https://github.com/protella/chatgpt-bots
cd chatgpt-bots
python3 -m venv .venv && source .venv/bin/activate
make install                    # pip install --require-hashes -r requirements.txt
```

Dependencies use a [pip-tools](https://github.com/jazzband/pip-tools) two-file layout:
`requirements.in` is the human-edited source of truth; `requirements.txt` is the generated lockfile
with exact pins and sha256 hashes for every package including transitives — **never edit it by hand**.
`--require-hashes` verifies each download against the lock, which is what makes installs reproducible
and tamper-evident.

Changing a dependency: edit `requirements.in`, run `make lock`, commit both files.
Bumping everything within the existing constraints: `make lock-upgrade`.

### Configure

```bash
cp .env.example .env
```

Required: `OPENAI_KEY`, `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`. Everything else has a working default —
[.env.example](.env.example) documents every knob inline, grouped by audience (credentials and
branding first, tuning you shouldn't touch at the bottom).

Worth a decision on day one:

| Setting | Default | Why you'd change it |
|---|---|---|
| `ENABLE_CHANNEL_LISTENING` | `false` | The master switch for teammate behavior. Off = mentions + DMs only |
| `BOT_NAME_ALIASES` | `ChatGPT` | Names the bot answers to without an `@`. **Set this per environment** (e.g. `ChatGPT-Dev`) |
| `ENABLE_DEEP_RESEARCH` | `true` | Background research jobs (minutes of model time each) |
| `SLACK_NATIVE_STREAMING` | `false` | Slack's native streamed messages. Validate in your workspace before enabling |
| `STATUS_LOADING_MESSAGES_FILE` | unset | Point it at your own file to brand the "working…" messages |

#### Token budget
The bot manages the context window automatically. `GPT54_TOKEN_BUFFER_PERCENTAGE` (name kept for
compatibility; it describes the 1.05M window) sets how much of the window it will fill;
`TOKEN_CLEANUP_THRESHOLD` (0.8) decides when a thread gets compacted and `TOKEN_COMPACTION_TARGET`
(0.7) how far. Compaction rolls old spans into a summary that preserves file and image references —
nothing is dropped silently. Lower the buffer if you hit token-limit errors with heavy tool use.

#### Status messages
The "working…" bubble draws from `status_messages/loading_messages.generic.txt` (100 bundled lines).
To brand it, copy the file, rewrite the lines, and point `STATUS_LOADING_MESSAGES_FILE` at your copy —
plain text only, since Slack's status surface renders neither emoji nor `:shortcodes:`. Per-stage
texts ("reading a document", "generating an image") live in `status_messages/pipeline_messages.txt`.
A missing or broken file can never break the bot; it falls back to built-in text.

## MCP (Model Context Protocol)

> **Beta.** There is no approval UI, so `require_approval` is always forced to `"never"` internally —
> the model can call any tool an enabled server exposes without confirmation. Prefer read-only
> servers, and bound each one with an `allowed_tools` allowlist.

Copy `mcp_config.example.json` to `mcp_config.json` (gitignored) and list your servers:

```json
{
  "mcpServers": {
    "my_database": {
      "server_url": "https://api.example.com/mcp",
      "server_description": "Company database access",
      "headers": { "Authorization": "Bearer ${MY_DATABASE_TOKEN}" },
      "enabled": true,
      "allowed_tools": ["query_customers", "get_orders"]
    }
  }
}
```

Only `server_url` is required. **Keep secrets in `.env`** — `${VAR_NAME}` placeholders in `headers`
are expanded from the environment at load, and a server with unresolved placeholders is skipped with a
warning naming the variable. `"enabled": false` parks a server without deleting it.

HTTP/SSE transport only (OpenAI's native MCP support — stdio servers won't work). On startup the bot
probes each server and logs one reachable/unreachable line per server; users toggle MCP access in
`/chatgpt-settings`. If the bot isn't using a tool you expect, check that startup line first, then
that the user has MCP enabled.

## Running

```bash
python3 slackbot.py                 # or: python3 main.py --platform slack
```

It connects over Socket Mode and starts serving immediately. `data/` (SQLite + backups) and `logs/`
are created on first run; the database backs itself up nightly with 7-day retention. In-memory thread
state is pruned on `CLEANUP_SCHEDULE` (default: daily at midnight) — threads are rebuilt from Slack
when they're next touched, so nothing is lost.

For a long-running deployment, use whatever supervisor you already trust — `pm2`, `systemd`, or
`nohup … &` plus an `@reboot` crontab entry all work.

## Development

```bash
make test        # unit tests with coverage
make test-all    # + integration tests (real API keys)
make lint        # ruff + mypy
make format      # black + isort
```

## License

[MIT](LICENSE) © Peter Rotella
