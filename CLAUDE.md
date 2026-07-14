# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python Slack bot on OpenAI's **Responses API** (never Chat Completions). The architecture is
stateless: Slack is the source of truth, and context is rebuilt from platform history on demand.

## Commands

```bash
source venv/bin/activate
python3 -m pip install --require-hashes -r requirements.txt   # or: make install

python3 slackbot.py                 # run the bot (see "Live dev-bot testing" below)

make test                           # unit tests + coverage (default)
make test-fast                      # no coverage
make test-all                       # unit + integration (real API keys from .env)
python3 -m pytest tests/unit/test_config.py::TestBotConfig::test_default_initialization -v
python3 -m pytest -m critical       # markers: critical, smoke, unit, integration

make lint                           # ruff + mypy
make format                         # black + isort
```

**Dependencies are a pip-tools two-file layout.** `requirements.in` is the human-edited source of
truth; `requirements.txt` is a generated lockfile with exact pins + hashes — never hand-edit it.
Add a dep by editing `requirements.in`, running `make lock`, and committing both.

**Optional system binaries** (all degrade gracefully to an honest "text not extractable" note):
`poppler-utils` + `tesseract-ocr` for scanned-PDF OCR, `pandoc` as the `.docx` fallback extractor.

## Architecture

### Responses API
- `store=False`; full history passed in `input` every call, no `previous_response_id` chaining.
- System prompts ride as the `developer` role.

### Models
Selectable: **gpt-5.6-sol** (default), **gpt-5.6-terra**, **gpt-5.6-luna** (also `UTILITY_MODEL`),
and **gpt-5.5**. Everything else — GPT-4.x, gpt-5, nano/mini, gpt-5-chat, gpt-5.1–5.4 — is gone.

- **GPT-5.6 family** (1.05M context, Feb 2026 cutoff): hybrid — `temperature`/`top_p` are legal
  only when `reasoning_effort=none`, otherwise temperature is forced to 1.0. Effort ladder is
  `none/low/medium/high/xhigh/max`; **`minimal` is a hard 400**, so route every stored effort
  through `clamp_effort(model, effort)` in `config.py`. Implicit prompt caching: send
  `prompt_cache_key`, never `prompt_cache_retention` (deprecated on 5.6).
- **gpt-5.5**: same shape, but effort tops out at `xhigh` (no `max`), and it keeps explicit
  `prompt_cache_retention: 24h`.

### Threading & state
`ThreadStateManager` holds per-thread state in memory, keyed `channel_id:thread_ts`, with locks to
prevent concurrent processing of one thread. **All state is lost on restart and rebuilt from Slack
history** — never rely on in-memory state persisting. `AssetLedger` tracks generated images per
thread.

### Persistence (`data/slack.db`, WAL mode)
**Slack is the only transcript — conversation history is NEVER written to the DB** (the `messages`
mirror was removed in v3; see `Docs/CHANNEL_TEAMMATE_REDESIGN_PLAN.md` §5b). The DB holds only what
Slack doesn't: config (users/threads/channel_settings), channel memory, derived artifacts (image
analyses/prompts, document extractions), and thread compaction summaries. Backups to
`data/backups/`, 7-day retention.

### Message pipeline
`BaseClient.handle_event()` → `MessageProcessor.process_message()` → text handler → local-tool loop.
Image generation/editing, code interpreter, search, memory and research are all **tools the model
picks in context**, not routes. (A pre-flight intent classifier used to hard-route here; it now only
runs when `ENABLE_IMAGE_TOOLS` is off.)

### Config hierarchy
`.env` defaults (`BotConfig`) → thread-specific overrides (memory/DB). Utility functions must use
`UTILITY_REASONING_EFFORT`/`UTILITY_VERBOSITY`; analysis functions use `ANALYSIS_*` — not the
default reasoning/verbosity vars.

### Tool subsystems
Image tools, the code-interpreter container, file mounting, artifact publishing, deep-research
builds, and Slack canvases each carry non-obvious API constraints that have already cost bugs.
**Read `Docs/TOOL_SUBSYSTEMS.md` before touching any of them.** In brief:
- Containers are thread-scoped and persisted; the API caps idle life at 20 minutes.
- `generate_image` is detached (posts itself); `create_image_asset` and `edit_image` are
  synchronous and feed the sandbox. Image sources are named by opaque id — no "most recent" guess.
- **Charts are computed in the sandbox from real data, never drawn by an image model.**
- Publishing posts the deliverable, not its ingredients.
- **Background jobs produce; they don't deliver.** `start_background_job` (`research` / `build` /
  `research_and_build`) stages its files out of the container, then calls the model back to decide
  what to say and which artifacts to post. Delivery policy lives in the model, not in the tool.
  Its status card is a live todo list: the dispatching model writes the `plan`, the job revises it
  with `update_todos` — a **free** tool (`free_tools`), so card updates never spend the round
  budget the build needs for `mount_file` / `create_image_asset`.
- Slack canvases: we create the *channel* canvas only (the sole route to a pinned tab); its title
  is an undocumented create-time param that can never be changed after; creation is not idempotent.

## Pitfalls

1. **Never use Chat Completions.** Responses API exclusively.
2. **Full context in every API call** — never truncate to "last N messages".
3. **Thread keys contain colons** (`channel_id:thread_ts`) — mind the delimiter in DB work.
4. **Files and documents never persist.** The DB may hold metadata + summary + Slack CDN ref only,
   and file processing must never touch disk: download to memory, process via `BytesIO`, discard.
   Same for images — URLs and metadata, never base64.
5. **Never send our attachment dicts straight to the API.** They do double duty (API part *and* DB
   metadata), so `source`/`filename`/`url`/`file_id` ride along and earn a 400
   (`Unknown parameter: 'input[3].content[1].source'`). Use `utilities.api_part()`. Note `file_id`
   means an *OpenAI* file id; ours is Slack's, and passing it earns a second 400.
6. **A mock stream MUST terminate and MUST yield real strings.** `output_text += <MagicMock>` does
   not raise — it silently turns `output_text` into a mock, and a stale `side_effect` once left an
   async iterator that never ended, grew the suite to 30GB, and OOM-killed the box. Cap pytest
   memory (`ulimit -v`). Details in `Docs/TOOL_SUBSYSTEMS.md`.
7. **SQLite concurrency** — WAL mode, and be careful with `check_same_thread=False`.

## Live dev-bot testing (authorized)

Claude may test against real Slack with the DEV bot credentials (`.env` `SLACK_BOT_TOKEN`) —
messages, reactions, API calls — confined to **C04QDHE8W8M** (`#chatgpt-bot-test`) and DMs with the
dev bot, and may start/stop/restart the dev bot process as needed. **Prod remains hands-off.**
Post test messages as the user (`SLACK_TEST_USER_TOKEN`); the bot token cannot trigger the bot.

## Git workflow

**Do not decide to commit on your own — wait to be asked.** When asked:
1. Add a `## [X.Y.Z] - YYYY-MM-DD` section at the top of `CHANGELOG.md` (under `## [Unreleased]`),
   using `### Added` / `### Fixed` / `### Changed` / `### Removed` with an emoji in the header.
2. Commit with a short summary line + detail bullets (match `git log` style), then push.
3. Releases are a separate, explicit ask — not every commit needs one. Tags are `vX.Y.Z`; match the
   format of existing `gh release list` entries.

## House rules

- Prefer editing existing files over creating new ones; never create docs unless asked.
- Absolute paths for file operations. **Test files go in `tests/`, not the repo root.**
- Don't break working bot code — if a fix is needed outside the task, consult the user first.
- No timelines or work estimates.
