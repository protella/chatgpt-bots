# ChatGPT Slack Bot

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Release](https://img.shields.io/github/v/release/protella/chatgpt-bots)](https://github.com/protella/chatgpt-bots/releases)

A Slack bot built on OpenAI's **Responses API** (not Chat Completions). It chats, generates and
edits images, reads documents, runs code in a sandbox, searches the web, runs multi-source
research jobs in the background - and, optionally, behaves like a teammate in the channels it's
invited to.

Slack is the source of truth for conversations: transcripts are never mirrored into the local
database. Context is rebuilt from Slack history on demand, and long threads roll into compacted
summaries instead of being silently truncated.

## Features

**Core**

- **Conversation** - talk to it in DMs, threads, or channels. It follows the whole
  conversation, keeps track of who said what, and knows what "last night" or "an hour ago"
  means
- **Images** - ask for a picture, or an edit to one, in plain language; it works in the
  background while the conversation keeps moving
- **Vision** - drop in a screenshot or photo and ask about it
- **Documents** - upload a PDF, spreadsheet, Word doc, or code file and ask questions about
  it. Scanned PDFs and Slack canvases included; file contents are read on demand and never
  stored
- **Code interpreter** - "chart this spreadsheet" runs real Python on your real data and
  hands the chart (or spreadsheet, deck, PDF) back into the thread - numbers are computed,
  never invented by an image model
- **Web search** - current information, with cited sources
- **Deep research** - a big question becomes a background job with a live progress card; a
  sourced report or a built deliverable arrives minutes later while you keep chatting
- **Slack search** - "what did we decide about X?" - it searches your workspace (in DMs) or
  the current channel (in channels) and answers with links to the messages
- **On-demand context** - it can pull up older history, link you to the message where
  something was decided, and look up people, pins, and channel info
- **Canvases** - it can create and edit the channel canvas, so living documents get updated
  in place instead of reposted

**Channel teammate (`ENABLE_CHANNEL_LISTENING`)** - invite it to a channel and it behaves
like a teammate, on by default:

- Answers when it can genuinely help, reacts when an emoji says it better (your workspace's
  custom emoji included), and stays out of conversations that aren't for it
- Takes feedback: tell it "be quieter in here" and that becomes the channel's standing rule
- Remembers each channel's durable facts - decisions, conventions, preferences - and you can
  review and edit them
- Quietly takes notes on links, files, and images shared in the channel, so "what did that
  chart say?" works days later (`ENABLE_AMBIENT_MEMORY`)
- Introduces itself once when added to a channel (`ENABLE_CHANNEL_JOIN_INTRO`)
- Anyone can tune it per channel - participation level (on / mentions-only / off), standing
  rules, reply placement, model - via the ⚙️ button under any of its replies
- For operators: every spoke/declined decision is logged with its reason
  (`logs/participation.jsonl`, `ENABLE_PARTICIPATION_TELEMETRY`)

**User experience**

- Settings modal (`/chatgpt-settings`): model, reasoning effort, verbosity, image defaults,
  custom instructions - per user, per channel, and per thread
- Live status bubble with customizable "working…" messages and ticking checklists on image
  jobs
- Optional 👍/👎 feedback buttons in DMs (`ENABLE_FEEDBACK_BUTTONS`, off by default); thumbs
  reactions on any bot message are always recorded as the same signal

## Known limitations

- **Context is bounded by Slack.** The bot rebuilds what Slack retains and returns - workspace
  retention policies and plan limits apply.
- **Sandbox containers idle out after ~20 minutes** (an OpenAI API limit); a revived thread
  quietly gets a fresh one.

## Requirements

- **Python 3.12** (what the bot is developed and tested on), plus `git` and `make`
- **An OpenAI API key** with access to the GPT-5.6 family and `gpt-image-2`
- **A Slack workspace** where you can create and install apps
- **A host that can run a persistent process** (Linux or macOS)
- **Optional system packages** for document handling:

  ```bash
  apt-get install poppler-utils tesseract-ocr pandoc     # Linux
  brew install poppler tesseract pandoc                  # macOS
  ```

  `poppler` + `tesseract` are required for scanned-PDF OCR (`ENABLE_PDF_OCR`, on by default);
  `pandoc` is only the last-resort `.docx` extractor (python-docx runs first). Missing
  packages degrade to an honest "text not extractable" note rather than an error - the bot
  works, but quietly loses that capability.
- **Upgrading from v2?** Your existing database needs SQLite 3.35+ for the v3 migrations -
  see [Upgrading to v3](#upgrading-to-v300).

## Installation

```bash
git clone https://github.com/protella/chatgpt-bots
cd chatgpt-bots
python3 -m venv .venv && source .venv/bin/activate
make install                    # pip install --require-hashes -r requirements.txt
cp .env.example .env
```

Then create the Slack app (next section), put its tokens plus your OpenAI key into `.env`,
review the feature flags, and run.

## Slack app setup

Create the app from a manifest: copy `slack_app_manifest.example.yml` to
`slack_app_manifest.yml` (gitignored - customize the app name and slash command per
environment), then [api.slack.com/apps](https://api.slack.com/apps) → **Create New App** →
**From an app manifest** → paste it. Then:

1. Enable **Socket Mode** (no public webhook URLs needed)
2. Generate an App-Level Token with `connections:write`
3. Install to the workspace and copy `SLACK_BOT_TOKEN` (`xoxb-`) and `SLACK_APP_TOKEN`
   (`xapp-`) into `.env`
4. **Turn on Agent mode** - App settings → **Agents & AI Apps**. The manifest's `agent_view`
   block sets it up, but confirm the toggle is on. It gives the bot the assistant split-view
   and is what makes Slack mint the per-message `action_token` behind *workspace* search in
   DMs. With it off, DM search degrades and the bot falls back to reading history directly;
   channel search is unaffected.

**The manifest is the authoritative scope and event inventory** - trim it there if you want
less. Roughly, by capability:

| Capability | Needs |
|---|---|
| Mentions & DMs (core) | `app_mentions:read`, `im:*`, `chat:write`, `files:*`, `commands` |
| Channel listening | `channels:history`/`groups:history`/`mpim:history` + matching `:read` scopes and `message.*` events |
| Reactions | `reactions:read`, `reactions:write`, their events, and `emoji:read` for custom emoji |
| Workspace search (DMs) | the six `search:read.*` scopes + Agent mode |
| Canvases | `canvases:read`, `canvases:write` |
| Channel intro | `member_joined_channel` event |
| Ambient memory cleanup | `file_deleted` event |
| Research byline | `chat:write.customize` (without it, findings post plainly) |
| People lookups | `users:read`, `users:read.email` |

A few manifest entries are forward-looking (`bookmarks:*`, `pins:write`, `users.profile:read`):
granted now so a future release doesn't force a re-consent, harmless to drop.

The bot uses **bot-token auth only** - no user scopes.

**Slash command:** `/chatgpt-settings` (set `SETTINGS_SLASH_COMMAND` to match; use a `-dev`
suffix for a dev install). **Message shortcut:** callback id `configure_thread_settings` -
per-thread settings from any message's ⋯ menu.

## Configuration

Required in `.env`: `OPENAI_API_KEY`, `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN` (the legacy name
`OPENAI_KEY` still works). Everything else has a working default, and
[.env.example](.env.example) documents every knob inline, grouped by audience.

> **⚠️ Know what the supplied config turns on.** Copied as-is, `.env.example` enables the
> full channel teammate (`ENABLE_CHANNEL_LISTENING=true` with channels defaulting to "on" -
> the master switch is off only when the key is absent), Slack-native streaming, ambient
> memory, the channel join intro, deep research (minutes of model time per job), the code
> interpreter, image tools, and MCP access for new users (`MCP_ENABLED_DEFAULT=true`).
> Any channel can be dialed down from the ⚙️ button (or by telling the bot), and
> `CHANNEL_RESPONSE_MODE=tag_only` makes mentions-only the global default instead. For a
> conservative mentions-and-DMs-only deployment, set:
>
> ```
> ENABLE_CHANNEL_LISTENING=false
> SLACK_NATIVE_STREAMING=false
> ENABLE_AMBIENT_MEMORY=false
> ENABLE_CHANNEL_JOIN_INTRO=false
> ENABLE_DEEP_RESEARCH=false
> ```

Also worth a decision on day one: `BOT_NAME_ALIASES` (names the bot answers to without an
`@` - set per environment so a dev bot doesn't answer to the prod bot's name),
`STATUS_LOADING_MESSAGES_FILE` (brand the "working…" messages; plain text, one per line),
and the status-emoji names near the top of `.env.example` - they must exist in your
workspace or the corresponding indicators silently fail.

### Models

All chat models share a 1.05M-token context window and prompt caching. Users pick theirs in
`/chatgpt-settings`; a channel or a single thread can override it.

| Model | Role |
|---|---|
| `gpt-5.6-sol` | Flagship reasoning model - **the default** |
| `gpt-5.6-terra` | Balanced tier |
| `gpt-5.6-luna` | Fast tier; also runs the bot's internal utility calls |
| `gpt-5.5` | Previous flagship, still selectable |
| `gpt-image-2` | Image generation and editing |

Reasoning effort runs `none → low → medium → high → xhigh → max` on the 5.6 family (`max` is
5.6-only; the settings modal adapts the list to the chosen model).

### Token budget

The bot manages the context window automatically: `TOKEN_CLEANUP_THRESHOLD` (0.5 as shipped)
decides when a thread is compacted and `TOKEN_COMPACTION_TARGET` (0.4) how far. Compaction
rolls old spans into a summary that preserves file and image references. Lower
`GPT54_TOKEN_BUFFER_PERCENTAGE` (the name is legacy; it sizes the 1.05M window) if you hit
token-limit errors with heavy tool use.

## Running

```bash
python3 slackbot.py                 # or: python3 main.py --platform slack
```

It connects over Socket Mode and serves immediately; `data/` (SQLite + backups) and `logs/`
are created on first run. To verify: watch `logs/app.log` for the startup lines (MCP probe
results included), DM the bot, @mention it in a channel it's been invited to, and open
`/chatgpt-settings`.

Cleanup and database backups run on `CLEANUP_SCHEDULE` - daily at midnight with the supplied
`.env` (backup retention 7 days); if the variable is unset the code falls back to weekly. Idle
in-memory thread state is pruned on the same schedule and rebuilt from Slack when a thread is
next touched, so nothing is lost.

For production, run it under a process supervisor (`systemd`, `pm2`) that starts it in the
repo directory with the virtualenv active and restarts on failure - a fatal startup error
exits non-zero on purpose.

A container works well too, and nothing about the bot resists it: it makes only outbound
connections (Socket Mode), so no ports are exposed and no reverse proxy is needed. Build the
image with the optional system packages (`poppler-utils`, `tesseract-ocr`, `pandoc`), mount
`data/` and `logs/` as volumes so the database and its backups outlive the container, and
pass the environment via `--env-file .env`. Keep `data/` on a real local volume - SQLite in
WAL mode should not live on a network filesystem.

## MCP (Model Context Protocol)

> **Beta.** There is no approval UI, so `require_approval` is always forced to `"never"`
> internally - the model can call any tool an enabled server exposes without confirmation.
> Prefer read-only servers, bound each one with an `allowed_tools` allowlist, and remember
> `MCP_ENABLED_DEFAULT=true` grants configured servers to new users by default.

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

Only `server_url` is required. **Keep secrets in `.env`** - `${VAR_NAME}` placeholders in
`headers` are expanded from the environment at load, and a server with unresolved placeholders
is skipped with a warning naming the variable. `"enabled": false` parks a server without
deleting it.

HTTP/SSE transport only (OpenAI's native MCP support - stdio servers won't work). On startup
the bot probes each server and logs one reachable/unreachable line; users toggle MCP access in
`/chatgpt-settings`. If a tool isn't being used, check that startup line first, then the
user's MCP toggle.

## Upgrading to v3.0.0

v3 is a major release: new model lineup, channel teammate mode, background research, and
conversation history moved out of the database into Slack. The short version:

1. **Back up `data/slack.db` yourself** before first boot - v3's migrations are one-way
   (the bot also takes tagged backups automatically).
2. Confirm SQLite ≥ 3.35: `python3 -c "import sqlite3; print(sqlite3.sqlite_version)"`
3. `make install`, then install the system packages above.
4. Update `.env` against the reorganized `.env.example` (changed defaults, deleted keys).
5. **Re-paste the manifest and reinstall the app** - new scopes need re-consent, and a
   missing one degrades its feature silently. Then confirm Agent mode is on (step 4 of
   [Slack app setup](#slack-app-setup)); an upgraded app that skips this keeps working but
   silently loses workspace search and the agent surface.
6. Start the bot and watch the migration log lines.

The exact keys, manifest deltas, and log lines are in [UPGRADING.md](UPGRADING.md) - follow
it in order.

## Development

```bash
make test               # unit tests with coverage
make test-fast          # all discovered tests, no coverage
make test-unit          # unit tests only
make test-all           # unit + integration (integration hits real APIs with .env keys)
make lint               # ruff + mypy (zero-error policy)
make format             # black + isort
make check              # lint + pii + test
make pii                # scan tracked files for internal identifiers
make install-hooks      # pre-commit hook: lint + pii
```

The lint/format tools themselves (`ruff`, `mypy`, `black`, `isort`) are not part of the
runtime lockfile - install them on your dev box; the Make targets skip tools that aren't
present.

Dependencies use a [pip-tools](https://github.com/jazzband/pip-tools) two-file layout:
`requirements.in` is the human-edited source of truth, `requirements.txt` is the generated
lockfile with hashes - never edit it by hand. Change a dep by editing `requirements.in`,
running `make lock`, and committing both; `make lock-upgrade` bumps everything within the
existing constraints.

## License

[MIT](LICENSE) © Peter Rotella
