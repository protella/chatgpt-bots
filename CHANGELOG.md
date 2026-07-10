# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

> Shipping as **v3.0.0** — a major release. The three headlines: a new model lineup
> (GPT-5.6 Sol/Terra/Luna), the bot can now act like a real channel teammate
> (off by default), and conversation history now lives in Slack, not the database.
> Follow the Upgrade Instructions below in order; total hands-on time is a few minutes.

### 📦 Upgrade Instructions (start here)

**1. Update dependencies**
```bash
make install   # pip install --require-hashes -r requirements.txt (openai >= 2.45.0)
```

**2. Update your `.env`** — compare against the reorganized `.env.example`. Everything new has a sane default if omitted; the items that matter:

Changed values (update if you set them explicitly):
```
GPT_MODEL=gpt-5.6-sol            # was gpt-5.5
UTILITY_MODEL=gpt-5.6-luna       # was gpt-5-mini
UTILITY_REASONING_EFFORT=none    # was minimal ("minimal" is rejected by 5.6 models)
```

Delete (no longer used):
```
DISCORD_TOKEN / DISCORD_ALLOWED_CHANNEL_IDS / DISCORD_LOG_LEVEL
GPT4_MAX_TOKENS
THREAD_MAX_TOKEN_COUNT
```

New keys worth a decision (see `.env.example` for the full annotated list — there's a
whole new "Channel participation & UX" section at the bottom):
- `ENABLE_CHANNEL_LISTENING=false` — the master switch for teammate behavior in channels.
  **Off by default: the bot behaves exactly as before (mentions + DMs) until you flip it.**
- `BOT_NAME_ALIASES=ChatGPT` — set per environment (e.g. `ChatGPT-Dev` for a dev bot)
- `STATUS_LOADING_MESSAGES_FILE` — optional branded "working…" messages for the
  thread status indicator: point it at your own text file, one message per line,
  plain text (no emoji — the status surface doesn't render them). Unset = a bundled
  set of 100 generic ones (`status_messages/loading_messages.generic.txt`).
- `SLACK_NATIVE_STREAMING=false` — native streaming is built and tested but ships off;
  validate live in your workspace before enabling
- `ENABLE_FEEDBACK_BUTTONS=true` — 👍/👎 under DM/assistant responses

**3. Migrate `mcp_config.json` secrets (recommended, not breaking)** — literal keys still
work, but you can now keep them in `.env`:
```
"X-API-Key": "${YOUR_VAR}"     # in mcp_config.json (any var name you like)
YOUR_VAR=sk-...                # in .env
```
Also new: per-server `"enabled": false`, and auth uses the `headers` object shape
(see README — the previously documented `authorization` shape never worked).

**4. Rebuild your Slack app manifest and reinstall.** Copy
`slack_app_manifest.example.yml` over your environment copy (keep your names/commands)
and reinstall the app. New since v2.5: `agent_view` block, bot scopes
`search:read.public/private/im/mpim/files/users`, `reactions:read`, `emoji:read`,
`assistant:write`, and events `reaction_added`, `app_home_opened`,
`app_context_changed` (legacy `assistant_thread_*` events stay during the transition).
Optional: subscribe `reaction_removed` if you want thumb reactions un-counted when removed.

**5. First startup runs three automatic DB migrations** — each takes a tagged backup
into `data/backups/` first (your rollback path). Watch for these log lines:
- `DB: Mirror-drop migration complete — removed N cached message rows` (backup tagged
  `pre-v3-mirror-drop`) — the DB no longer stores conversation transcripts
- `DB: Doc-content-drop migration complete` (backup tagged `pre-v3-doc-content-drop`)
  — the DB no longer stores document content
- `DB: One-time GPT-5.6 migration — swapped N user(s) to gpt-5.6-sol with medium reasoning`
  — everyone moves to the new default; users can re-pick model/effort globally,
  per channel, and per thread afterward

### 🚀 Feature - GPT-5.6 model family (Sol / Terra / Luna)

#### Added
- **Model picker now offers four models**: GPT-5.6 Sol (flagship, new default),
  GPT-5.6 Terra (balanced), GPT-5.6 Luna (fast), and GPT-5.5
- **New `max` reasoning effort** on all GPT-5.6 models (the effort list in settings
  adapts to the selected model)
- **One-time migration**: all users move to `gpt-5.6-sol` with `medium` reasoning;
  a startup normalizer also clamps any stored model/effort a model no longer accepts,
  so stale settings can never cause API errors

#### Changed
- **Utility functions** (intent classification, summaries) now run on `gpt-5.6-luna`
  instead of `gpt-5-mini`
- Prompt caching on 5.6 models is automatic (no cache-retention parameter needed);
  GPT-5.5 keeps its 24-hour retention behavior
- **Context budgets audited against verified model specs**: all 5.6 models and
  GPT-5.5 use their full 1,050,000-token window (~920k usable after the reserve
  for output, tool results, and estimator error), and the bot now logs once per
  thread when a conversation crosses the 272k long-context billing tier (requests
  beyond it bill at 2× input / 1.5× output — informational only, nothing blocks)

#### Removed
- **All pre-5.5 model support**: GPT-4 series, `gpt-5`, `gpt-5-nano`, `gpt-5-chat-*`,
  and `gpt-5.1`–`gpt-5.4`, plus their dead API branches and one-off migration scripts.
  `gpt-5-mini` is no longer used anywhere.

### 🤝 Feature - The bot can be a channel teammate (off by default)

Everything here is inert until you flip `ENABLE_CHANNEL_LISTENING=true`; mentions and
DMs behave as before.

#### Added
- **Channel-wide listening with judgment**: a lightweight participation engine sees
  channel messages and decides — respond, react with an emoji, stay silent, or back
  off. Hard rails included: a per-channel hourly cap on unprompted replies
  (`MAX_UNPROMPTED_REPLIES_PER_HOUR`), rapid-fire debouncing, and "ignore" as the
  default verdict.
- **Per-channel participation levels** — off / mentions-only / judicious / active —
  set by anyone via the **⚙️ Configure** button under bot responses (plus channel
  directives and reply placement, as before)
- **Shared channel response settings**: the same ⚙️ modal now also sets the channel's
  model, reasoning effort, and verbosity — shared by everyone in that channel and
  editable by any member. Hierarchy: personal settings < channel settings < per-thread
  overrides; anything left on "each person's own setting" keeps using the asker's
  personal preferences. A "My personal settings" button inside the modal opens your
  own settings without a second button in chat.
- **"Quiet down" works like you'd hope**: telling the bot to pipe down gets a 🤐 and
  snoozes unprompted participation for 4 hours (mentions still answered). Standing
  feedback ("stay out of here unless tagged", "keep answers short in this channel")
  is remembered durably as a channel preference.
- **Per-channel memory, model-managed**: the bot decides what's durable
  (decisions, conventions, preferences) and remembers/updates/forgets it via its own
  tools; facts are recalled in future conversations in that channel
- **On-demand context tools**: the bot can fetch older thread/channel history and
  search the workspace (`assistant.search.context`) when a conversation references
  something it can't see — instead of guessing. It can also link directly to an
  earlier message (drop a clickable permalink for "where did we discuss X?"), read
  a message's current emoji reactions, list a channel's pins (needs the new
  `pins:read` scope from the updated manifest), and look up channel info and user
  profiles — all fetched live, nothing stored. Search is permission-gated in code
  (public/private channels only by default) and only possible while handling a real
  triggering message.
- **Emoji reactions** as a response type, both engine-chosen and model-invoked
  (allowlisted via `REACTION_EMOJIS`)

#### Changed
- **No more "busy" rejections anywhere**: messages arriving while the bot is working
  are queued and answered together in one coherent catch-up reply (DMs, threads, and
  channels). The old "I'm busy, try again" behavior is retired.
- **Replies thread by default** in channels; genuine top-level replies are reserved
  for answers the whole channel needs

### 🗄️ Changed - Slack is now the only transcript

- **The database no longer mirrors conversations.** Context is rebuilt from Slack
  history on demand; long threads are compacted into rolling summaries (file and
  image references preserved) instead of trimmed silently. What the DB still holds:
  settings, per-channel memory, derived artifacts (image analyses, document
  summaries), and thread summaries.
- **Token budgeting is usage-driven** (exact counts from API responses); the tiktoken
  dependency is gone
- One-time cleanup migration drops the old message mirror (tagged backup first — see
  Upgrade Instructions)

### 📄 Feature - Smarter, lighter document handling

#### Added
- **Documents no longer flood the conversation**: uploads inject a concise summary
  (spreadsheets show sheets/columns/sample rows); when you ask for specifics, the bot
  re-reads the original file on demand instead of guessing from the summary
- **PDFs are read natively by the model** (`ENABLE_NATIVE_FILE_INPUT`, on by default):
  tables, charts, and scanned pages are actually visible to it now
- **Privacy**: document content is never stored and never touches disk — the bot keeps
  only a summary and a reference to the file in Slack, and processes files in memory.
  Deleting a file from Slack removes its content from the bot's reach entirely.

### 👍 Feature - Response feedback

- **Feedback buttons** (👍/👎) under DM and assistant-surface responses
  (`ENABLE_FEEDBACK_BUTTONS`, on by default; channels stay clean)
- **Thumbs-up/down reactions on the bot's messages are recorded** as the same signal —
  passively, with zero model cost
- Feedback lands in a local table for future tuning; nothing leaves your workspace

### 🖥️ Changed - Slack agent surface & native streaming

- Migrated to Slack's current agent view (June 2026): greeting and suggested prompts
  now ride the new `app_home_opened` surface; legacy events remain subscribed during
  the transition
- **Native streaming** (`chat.startStream`/`appendStream`/`stopStream`) is fully wired
  behind `SLACK_NATIVE_STREAMING` — **ships off** pending live validation in your
  workspace; the classic edit-loop streaming remains the default and the automatic
  fallback
- The status indicator only appears once the bot has actually decided to respond
  (the new surface auto-opens threads on status, so no more speculative indicators)
- **One clean "working" indicator**: progress renders as Slack's single in-thread
  status bubble (the animated agent name) — no placeholder messages to edit/delete,
  no duplicate status line under the composer
- **Loading messages with personality**: while thinking, the bubble rotates through
  a random sample from a 100-message pool (bundled generic set included; brand it
  with `STATUS_LOADING_MESSAGES_FILE` — one message per line, plain text). Inline
  `STATUS_LOADING_MESSAGES` still works for short custom lists and wins when set.
- **Pipeline stage updates get variety too**: each stage (generating a response,
  creating/editing an image, reading a document, …) picks a random phrasing from
  `status_messages/pipeline_messages.txt` (`[stage]` sections; override the path
  with `PIPELINE_MESSAGES_FILE`). Missing files or stages fall back to the built-in
  texts — a broken file can never break the bot.
- **Quieter DM surface**: the greeting only posts in genuinely empty conversations,
  the feedback strip (👍/👎 + settings button) appears once per thread instead of
  under every reply, and the old "Quick Settings Access" notice is retired

### 🔌 Improvement - MCP hardening

#### Added
- **Secrets out of `mcp_config.json`**: header values support `${VAR_NAME}`
  placeholders expanded from `.env` at load; a server with unresolved variables is
  skipped with a warning naming them
- **Per-server `"enabled": false`** to turn off one server without deleting its config
- **Startup health probe**: one log line per MCP server (reachable/unreachable) plus
  its discovered tools

#### Fixed
- MCP failover survives multiple failing servers: exclusions accumulate across retries
  (previously two broken servers could retry each other forever), and failures are
  detected from structured error codes first with message-text matching as fallback
- A config requesting `require_approval` other than "never" now logs a clear warning
  instead of being silently ignored
- README documented an `authorization` config shape that never worked — corrected to
  the real `headers` shape

### ✨ Improvement - Prompts modernized for current models

- **Snappier, channel-appropriate replies**: brief and conversational at channel top
  level, fuller detail in threads; the old always-use-section-headers rule (which made
  every reply memo-shaped) is gone
- **Faster vision responses**: the extra "question enhancement" model call before every
  image analysis is off by default (`ENABLE_VISION_ENHANCEMENT=false`) — it added 1–2s
  latency; current models answer the question directly
- **More literal image edits**: edits state exactly what changes and preserve
  everything else; generation prompts preserve your explicit specifications verbatim
- **Lower cost per message**: the intent classifier prompt was trimmed ~60%, and
  multi-user threads no longer lose prompt-cache hits on every speaker change

### 🩹 Fixed - Error messages that respect the reader

- **No more raw error dumps in Slack**: the old `Error Code / Type / Details` code-block
  scaffold is retired; every user-facing error is now one friendly line with a clear
  next step, and technical details stay in the logs
- **Nothing fails silently anymore**: a file that couldn't be downloaded says so
  ("couldn't download report.pdf — try re-uploading") instead of being ignored;
  a failed catch-up on queued messages asks you to re-send; a Configure button that
  couldn't open the settings modal tells you
- Fixed an orphaned "Generating image…" indicator when image generation was blocked
  by moderation

### 🎨 Fixed - Image settings now reach the API on every path

- **Your quality setting applies to edits too**: image *edits* previously ignored the
  quality picker (only generations honored it); both paths now send it
- **Output format honored on generation**: `DEFAULT_IMAGE_FORMAT` (png/jpeg/webp) and
  compression were accepted but silently dropped on generation; they're now wired
  through just like on edits
- Verified against the live API: the shared size list (square/portrait/landscape/auto)
  is valid on both gpt-image-2 and gpt-image-1, so one setting covers both models

### 🧹 Changed - .env reorganized & stale settings retired

- **`.env.example` reordered by audience**: required credentials up top, branding/UX
  next, models & features in the middle, and "don't touch unless you know what you're
  doing" tuning at the bottom — same variables, no value changes
- **Dead entries removed**: `DALLE_MODEL`, `DEBUG_MODE`, `MAX_CONCURRENT_THREADS`,
  `MESSAGE_TIMEOUT` (nothing read them); missing live keys added
  (`TOKEN_COMPACTION_TARGET`, `UTILITY_MAX_TOKENS`, `DEFAULT_IMAGE_FORMAT`)
- **Settings modal remnants cleaned up**: leftover GPT-4/5.1/5.2-era form ids and the
  obsolete "web search disables Minimal reasoning" coupling are gone (verified live:
  web search works at reasoning `none` on 5.6); every comment now matches what the
  API actually accepts

### 🧹 Removed - Discord scaffolding & legacy code

- **Discord support removed**: the V2 Discord bot was never built (the launcher was a
  "Coming Soon" stub). The bot is Slack-only.
- **`legacy/` (V1 bots) deleted** — still available in early git history

### 🧪 Changed - Test suite restored

- The unit suite is fully green again (1,185 tests, 0 failures) after years of rot;
  `make test` now runs the entire suite instead of stopping at the first failure.
  Stale tests of removed behavior were deleted; tests of real behavior were repaired.

### ✅ Feature - Live progress checklists on image tasks

- Image generation and editing now show an accumulating checklist that ticks off each
  step in place ("✓ Enhanced prompt → ✓ Generated image → Uploading…") instead of a
  single status line that overwrites itself. On surfaces where only the composer status
  is available, it falls back to that automatically. Toggle with `ENABLE_PROGRESS_CHECKLIST`
  (default on).

### 🖼️ Feature - Image generation no longer freezes the conversation

- Creating a new image used to hold the thread while the model worked, so anything you
  said in the meantime had to wait. Image generation now runs in the background: the
  image posts automatically when it's ready and you can keep chatting the whole time.
  If you ask for another image while one is still cooking, the bot tells you it's still
  working rather than starting a second one. Image editing is unchanged this release.
  Toggle with `ENABLE_BACKGROUND_IMAGE_GEN` (default on); image jobs get their own
  longer time budget via `API_TIMEOUT_IMAGE` (default 300s).
- Fixed: the "✨ Enhanced Prompt" preview had stopped appearing on most surfaces (it was
  tied to a status message that newer Slack surfaces don't create) — it now posts as its
  own message so you can always see the prompt the image was built from.

### 🤐 Feature - The bot can now choose to stay quiet

- When the bot joins a channel conversation on its own (not @-mentioned or DMed), it can
  now decide that silence is the right move — the message wasn't for it, someone already
  answered, or a reaction says enough — instead of always producing a reply. These
  self-started turns also no longer stream partial text; they post once, complete, or not
  at all, so you never see a half-sentence that then vanishes. **Behavior change:**
  self-started (unprompted) channel replies no longer stream incrementally — they appear
  in one piece when finished. Directly addressed messages (mentions, DMs, 1:1 threads) are
  unchanged and still stream as before. Toggle with `ENABLE_NO_REPLY_TOOL` (default on).
- Fixed: a channel turn that ended in only an emoji reaction could still count against the
  bot's hourly self-started-reply budget — reactions and deliberate silence no longer burn
  that budget.

### 🧭 Feature - The bot knows why it woke up

- The model now receives a compact, internal "wake context" note alongside each channel
  message telling it why it's responding — an @-mention, its name coming up in passing, a
  direct-message, a 1:1 thread reply, an ambient judgment call (with the reason), or a
  batched catch-up — plus whether the sender started the thread or joined it, and whether
  they're a person or another bot. This sharpens the bot's read of who's talking to whom
  and when a reply is actually wanted. The note is internal context only (never posted,
  never stored). Toggle with `ENABLE_WAKE_ENVELOPE` (default on).

### 🩹 Fixed - Reliability hardening for the new channel/image features

- Live progress checklists now also appear when the bot **edits** an image (not just when
  it generates one), and when background image generation is turned off.
- A generated image now reliably lands in the bot's memory even if the thread was busy at
  the moment it finished uploading — so a follow-up "edit that" always finds it.
- If saving an image's details briefly fails after it was already posted, the bot no
  longer tells you the post failed — the image you can see is treated as posted.
- A reply that failed to send, ended in only an emoji reaction, or was deliberate silence
  no longer counts against the bot's hourly self-started-reply budget.
- Assorted internal cleanups: no lingering "Generating…" status when a background image is
  in flight, no duplicate "Enhanced Prompt" messages on some surfaces, and the bot
  remembers an "I'm still working on the last image" reply in-context right away.

## [2.5.1] - 2026-05-11

### 🚀 Feature - GPT-5.5 Support

#### Added
- **`gpt-5.5` added to the model picker** in `/settings` (top of dropdown, above GPT-5.4)
- **`gpt-5.5` is the new default model** for new users and all existing users
- **MODEL_KNOWLEDGE_CUTOFFS entry**: `gpt-5.5` → "August 31, 2025"
- **One-time DB migration**: existing users on any pre-5.5 model are auto-swapped to `gpt-5.5` on first startup. Gated by a `gpt55_migrated` sentinel column so users who later pick another model via `/settings` aren't reset on subsequent restarts.

#### Changed
- **`GPT_MODEL` env default**: `gpt-5.4` → `gpt-5.5`
- **API parameter handling**: `gpt-5.5` follows the same hybrid pattern as `gpt-5.4` — supports `temperature`/`top_p` when `reasoning_effort=none`, otherwise forces temp=1.0. Same prompt caching (`prompt_cache_retention: 24h`).
- **Token limits**: `gpt-5.5` reuses the existing 1.05M context window config (`GPT54_MAX_TOKENS`) since it has the same context size

#### Not supported
- `gpt-5.5-pro` (different pricing tier, deferred)
- `gpt-5.5-instant` (ChatGPT-only, not on the API)

#### Cost impact
GPT-5.5 input pricing is roughly 2× GPT-5.4 ($5 vs $2.50 per 1M tokens). Output is ~1.5× ($30 vs ~$20). Expect API spend per conversation to roughly double after this upgrade.

#### Upgrade Instructions
Update your `.env`:
```
GPT_MODEL=gpt-5.5
```
On first startup, watch the logs for the one-time swap:
```
DB: One-time migration — swapped N user(s) to gpt-5.5
```
Users can still pick `gpt-5.4` (or any older supported model) per-user in `/settings`.

## [2.5.0] - 2026-05-11

### 🚀 Feature - GPT Image 2 Support with Per-User Model Picker

#### Added
- **gpt-image-2 as default image model**: Latest OpenAI image generation model (released April 21, 2026) with agentic reasoning, near-perfect text rendering, and multilingual support
- **Image model picker in `/settings`**: Users can toggle between `gpt-image-2` (latest) and `gpt-image-1` (legacy) per-user
- **`image_model` column in `user_preferences`**: New schema column with automatic migration for existing databases
- **Model-aware parameter filtering**: Dropdown options dynamically filter based on selected image model — modal rebuilds when picker changes

#### Changed
- **`GPT_IMAGE_MODEL` env var default**: `gpt-image-1` → `gpt-image-2`
- **Backend parameter guards**: `gpt-image-2` doesn't support `background=transparent` (coerced to `auto` with warning) and ignores `input_fidelity` (auto-handled by model)
- **UI behavior on v2 selection**: "Transparent" background option hidden; "Image edit style" radio block hidden (model auto-handles fidelity)

#### Fixed
- **OpenAIClient wrapper signatures**: Added `model` kwarg to `generate_image()` and `edit_image()` wrapper methods to propagate user selection through to API calls

### 🐛 Bug Fix - "Please configure your settings" Warning Loop

#### Fixed
- **Thread-scope save now flags user as onboarded**: Previously, saving `/settings` with scope = "thread" updated only the thread config and never flipped `user_preferences.settings_completed`. Users who only ever saved in-thread kept seeing the "⚠️ Please configure your settings" reminder on every DM, forever. Now any save (thread or global) marks the user as onboarded.
- **One-time backfill on startup**: Any user whose `user_preferences` row was created more than 24 hours ago is auto-flagged `settings_completed=1`. Long-standing users who got stuck in the warning loop are unstuck immediately on the next deploy. New users (<24h) still see the welcome flow as intended.

### 🔒 Security - CodeQL High-Severity Fixes

#### Fixed
- **Slack URL hostname check** (`image_url_handler.py`): Replaced substring matching (`'slack.com/files/' in url`) with proper `urlparse()` hostname validation. Attacker URLs like `https://evil.com/slack.com/files/x` could previously have leaked the Slack auth token off-platform. (CodeQL #2)
- **API key leak in test output** (`tests/integration/test_intent_classification.py`): Removed `print(f"API Key: {key[:20]}...")` which exposed the first 20 chars (sk- prefix + 16 chars of secret) in test stdout. Replaced with set/MISSING boolean. (CodeQL #1)

### 🔧 Improvement - Dependency Hygiene + pip-tools Lockfile

#### Added
- **pip-tools two-file layout**: `requirements.in` (human-edited source of truth) + `requirements.txt` (autogenerated lockfile with exact pins + sha256 hashes for every dep including transitives)
- **`make install`** — uses `--require-hashes` against the lockfile for reproducible + supply-chain-safe installs
- **`make lock`** — regenerates `requirements.txt` from `requirements.in` (run after editing the manifest)
- **`make lock-upgrade`** — bumps all deps to latest within `requirements.in` constraints
- **Dependabot v2 config** (`.github/dependabot.yml`): weekly Monday scan, grouped minor+patch updates, `versioning-strategy: lockfile-only`
- **Dependabot security updates enabled** on the repo (auto-PRs for CVEs)

#### Changed
- **All unpinned deps now have floor versions** (`python-dotenv`, `openai`, `slack_bolt`, `Pillow`, `requests`, `croniter`, `pytz`, `discord`) — improves reproducibility and CVE audit clarity
- **`openai>=2.0.0`** floor pinned explicitly to lock in Responses-API-compatible SDK family
- **Bumped floors** to current-installed versions across the board (tiktoken, aiohttp, aiosqlite, pytest, pytest-asyncio, pytest-mock, pytest-env, coverage, pdfplumber, python-docx, pandas)
- **`PyPDF2` → `pypdf`** migration: PyPDF2 was deprecated in 2022 and replaced by `pypdf` (same `PdfReader` API). Updated imports in `document_handler.py` + tests. `extraction_method` field renamed `PyPDF2_fallback` → `pypdf_fallback`.

#### Removed
- **`asyncio` from requirements.in**: it's a stdlib module; the PyPI package is an abandoned 2015 backport that shadows stdlib

#### Developer Workflow
When adding/removing a dep:
1. Edit `requirements.in`
2. Run `make lock`
3. Commit both `requirements.in` and `requirements.txt` together

#### Upgrade Instructions

**1. Update `.env`:**
```
GPT_IMAGE_MODEL=gpt-image-2
```

**2. Backup the database before deploying** (schema migration auto-runs on startup):
```bash
cp data/slack.db data/slack.db.bak-$(date +%Y%m%d)
```

**3. Install dependencies via the new lockfile workflow:**
```bash
make install
```
Old `pip install -r requirements.txt` still works but loses hash verification.

**4. On first startup, watch the logs for these one-time migration entries:**
```
DB: Successfully added image_model column and migrated N existing user(s) to gpt-image-2
DB: Backfilled settings_completed=1 for M pre-existing user(s)
```
Subsequent startups skip both (gated by column-exists check and the `WHERE settings_completed=0` filter).

## [2.4.0] - 2026-03-06

### 🚀 Feature - GPT-5.4 Support with 1M Context Window

#### Added
- **GPT-5.4 as default model**: 1.05M token context window (~920k usable input)
- **Temperature/Top P for GPT-5.4**: Available when reasoning is set to None, dynamically shown/hidden in settings modal
- **Migration script**: `migrate_to_gpt54.py` to bump existing users (dry run by default)
- **Prompt caching**: Enabled for GPT-5.4 (24h retention)

#### Changed
- **Token limits fully model-aware**: Removed legacy flat `thread_max_token_count` usage; all paths now use `get_model_token_limit(model)`
- **API parameter handling**: GPT-5.4 with reasoning=none passes through temperature/top_p, otherwise forces temp=1.0

#### Fixed
- **Reasoning level compatibility**: Migration converts `minimal` (GPT-5/5-mini only) to `low` for GPT-5.4

#### Upgrade Instructions
Add these new environment variables to your `.env`:
```
GPT_MODEL = "gpt-5.4"
GPT54_MAX_TOKENS = "1050000"
GPT54_TOKEN_BUFFER_PERCENTAGE = "0.876"
```
Existing variables (`TOKEN_BUFFER_PERCENTAGE`, `TOKEN_CLEANUP_THRESHOLD`, etc.) do not need to change.

Run the migration to update existing user preferences:
```bash
python3 migrate_to_gpt54.py --db data/slack.db          # dry run
python3 migrate_to_gpt54.py --db data/slack.db --apply   # apply
```

## [2.3.6] - 2026-01-07

### 🐛 Bug Fix - MCP Error Handling & Retry UX

#### Fixed
- **MCP Graceful Fallback**: Improved error handling when MCP servers fail
  - Errors no longer shown directly to Slack users
  - Bot gracefully retries without the failing MCP server
  - Clear attribution shows which tools succeeded vs failed
- **Streaming on MCP Retry**: Fixed unnecessary fallback to non-streaming
  - Previously ANY retry fell back to non-streaming path
  - Now streaming continues when only MCP failed (streaming itself worked)
- **Retry Status Messages**: Non-streaming retries now show cycling progress updates
  - Added progress updater for retry scenarios
  - Uses proper emojis from config instead of hardcoded values

#### Changed
- **Status Message Emojis**: Now uses `circle_loader_emoji` from config for retry states
- **Tools Attribution**: Shows "(failed: server-name)" when MCP server couldn't be reached

## [2.3.5] - 2026-01-07

### 🐛 Bug Fix - MCP Authentication Headers

#### Fixed
- **MCP Headers Support**: Fixed authentication not being passed to MCP servers
  - Code was looking for `authorization` key but OpenAI expects `headers` object
  - Now correctly passes `headers` (including `Authorization: Bearer ...`) to OpenAI API
  - MCP servers requiring authentication will now work properly

#### Changed
- **MCP Example Config**: Updated `mcp_config.example.json` with correct format
  - Changed from incorrect `authorization` object to proper `headers` format
  - Simplified from ~225 lines to 39 lines with clear, copy-paste-ready examples
  - Shows four common patterns: public server, Bearer auth, custom header, tool whitelist

## [2.3.4] - 2025-12-16

### 🔧 Improvement - Image Quality Auto Option

#### Changed
- **Auto Quality Default**: Added 'auto' option for image quality and set as new default
  - Lets the model decide quality level based on prompt complexity
  - Available options now: auto, low, medium, high

## [2.3.3] - 2025-12-16

### 🚀 Feature - Image Quality & Background Settings

#### Added
- **Image Quality Setting**: User-configurable quality for image generation
  - Options: Auto, Low (faster/cheaper), Medium (balanced), High (best quality)
  - Exposed in `/chatgpt-settings` modal under Image Generation
- **Image Background Setting**: User-configurable background type
  - Options: Auto, Transparent, Opaque
  - Exposed in `/chatgpt-settings` modal under Image Generation
- **Database Migrations**: Automatic schema updates for existing users
  - New columns added with smart defaults on bot startup
  - No manual intervention required

#### Changed
- **Default Image Model**: Updated to `gpt-image-1.5` in `.env.example`
- **Documentation**: Updated README with GPT-5.2 model references

#### Removed
- **Deprecated Settings**: Removed `image_style` parameter (was DALL-E 3 only)

## [2.3.2] - 2025-12-15

### 🐛 Bug Fix - Streaming Blank Message & Pagination Orphan

#### Fixed
- **Vision Streaming Blank Updates**: Fixed race condition causing messages to briefly go blank during streaming
  - Root cause: `progress_task.cancel()` only requests cancellation, takes effect at next await point
  - Without awaiting, progress_task could overwrite streamed content with stale text
  - Now properly awaits cancellation before proceeding with streaming updates
- **Vision Pagination Orphan**: Fixed "Continued in next message..." appearing without Part 2
  - Vision handler had no overflow/pagination logic
  - Added full overflow handling matching text.py pattern with intelligent split points
- **Async Callback Support for Vision**: Added async callback support to vision API
  - Vision streaming callbacks can now properly await async operations
  - Matches pattern already used in responses.py for text streaming

#### Changed
- **Safety Margin Increase**: Increased overflow detection margin from 330 to 600 chars
  - Ensures overflow triggers before messaging layer's backup truncation at 3700 chars
  - Prevents orphaned "continued" messages from backup truncation

## [2.3.1] - 2025-12-15

### 🔧 Improvements - MCP Citation Stripping & Tool Attribution

#### Changed
- **MCP Citation Stripping**: Moved citation stripping from streaming layer to Slack messaging layer
  - Single point of control for all message types (streaming, non-streaming, updates)
  - Enhanced regex patterns to catch additional MCP citation formats
  - Properly handles tool-generated citations (`read_documentation`, `get_library`, etc.)
  - Web search citations preserved as clickable links
- **MCP Tool Attribution**: "Used Tools" footer now shows specific MCP server names
  - Format changed from `Used Tools: mcp` to `Used Tools: MCP (aws_knowledge, context7)`
  - Groups multiple MCP servers under single "MCP" label
  - Extracts server_label from `response.output_item.done` events for accurate attribution

#### Fixed
- **Citation Display**: Fixed MCP citations rendering as emoji + backend strings in Slack messages
- **Tool Attribution Accuracy**: Now correctly identifies which MCP servers were invoked during a response

## [2.3.0] - 2025-01-15

### 🚀 Feature - GPT-5.1 Model Support & Performance Optimizations

#### Added
- **GPT-5.1 Model Support**: Added GPT-5.1 as a new model option with enhanced capabilities
  - New "None" reasoning_effort option with adaptive reasoning
  - Automatic reasoning depth adjustment based on query complexity
  - 24-hour prompt caching for GPT-5.1 across all API calls (chat, vision, intent classification)
  - Web search now works with all reasoning levels including "none"
  - Separate settings UI for GPT-5.1 with dedicated reasoning options
  - Future-proof support for gpt-5.1-mini (not yet released)
- **Migration Script**: Created `scripts/migrate_users_to_gpt51.py` for automated user migration from GPT-5 to GPT-5.1
- **Configuration Updates**:
  - Added `gpt-5.1` to MODEL_KNOWLEDGE_CUTOFFS
  - Updated model dropdown in settings modal to include GPT-5.1 as top option
  - Added `_add_gpt51_settings()` method with new reasoning options
  - Changed default UTILITY_MODEL from gpt-4.1-mini to gpt-5-mini in .env.example

#### Changed
- **Reasoning Options**:
  - GPT-5.1 uses "none/low/medium/high" (replaces "minimal" with "none")
  - GPT-5 retains "minimal/low/medium/high" (backward compatible)
  - GPT-5.1 removes web_search + minimal reasoning constraint
- **API Integration**:
  - Added prompt caching (`prompt_cache_retention="24h"`) for GPT-5.1 in:
    - Main chat responses (streaming and non-streaming)
    - Vision analysis (streaming and non-streaming)
    - Intent classification (for future gpt-5.1-mini support)
  - Enhanced model detection logic in responses.py
  - Added `reasoning_level_gpt51` action handler for Slack modal interactions
- **System Prompt Optimization**: Moved date/time context to end of system prompt to maximize prompt caching effectiveness (90% cost savings on cached tokens)

#### Fixed
- **MCP Settings Preservation**: Fixed bug where MCP settings were lost when switching between GPT-4 and GPT-5 models
  - Validation no longer forces `enable_mcp=False` for GPT-4 users
  - Preserves user's MCP preference when switching back to GPT-5
  - Database now retains MCP setting even when using non-GPT-5 models

#### Notes
- GPT-5 model remains unchanged for backward compatibility
- Users can explicitly opt into GPT-5.1 via settings modal
- Run migration script manually to update existing GPT-5 users to GPT-5.1
- Reasoning effort preferences are model-specific and may need adjustment when switching models

## [2.2.3] - 2025-11-10

### 🐛 Bug Fix - MCP Settings Persistence

#### Fixed
- **MCP Toggle Persistence**: Fixed bug where MCP toggle changes in settings modal were not persisting to the database
  - Added `enable_mcp` to boolean fields list in `update_user_preferences()` (sync/async)
  - Added boolean conversion in `get_user_preferences()` (sync/async)
  - Added to thread config propagation in `get_or_create_thread_async()`
- MCP settings now correctly save and load across sessions for both global and thread-specific configurations

## [2.2.2] - 2025-11-07

### 🐛 Bug Fix - MCP Tool Attribution

#### Fixed
- **MCP Tool Attribution Accuracy**: Fixed bug where bot reported all available MCP servers in "Used Tools" footer instead of only servers actually invoked
  - Non-streaming: Detects tools via response.output inspection
  - Streaming: Detects tools via search_counts tracking
  - Both modes now show "Used Tools: mcp" only when MCP was actually invoked

#### Changed
- Simplified MCP attribution to show generic "mcp" label instead of individual server names
- Added `return_metadata` parameter to response API for tool usage tracking

## [2.2.1] - 2025-11-07

### 📝 Configuration & Documentation

#### Added
- **MCP Environment Variables**: Added MCP configuration to `.env.example`
  - `MCP_ENABLED_DEFAULT`: Enable MCP by default for new users
  - `MCP_CONFIG_PATH`: Path to MCP server configuration file
- MCP architecture documentation

## [2.2.0] - 2025-11-07

### 🎉 Major Feature - Model Context Protocol (MCP) Integration

#### Added
- **MCP Support (Beta)**: Full Model Context Protocol integration for GPT-5 models
  - Server configuration management via `mcp_config.json`
  - Database schema for caching MCP tool definitions
  - MCPManager handles server validation and tool discovery
  - Settings UI toggle for enabling/disabling MCP (GPT-5 only)
  - Dynamic MCP server inclusion in tools array
- **Citation & Attribution System**:
  - Strip MCP citations while preserving web_search citations (clickable links)
  - Unified tools attribution at end of responses
  - Clean API messages by removing attribution before OpenAI submission
- **Error Handling & Retry Logic**:
  - Graceful MCP connection failure handling with retry logic
  - Exclude failed MCP servers from retry attempts
  - User-friendly error messages for connection issues
  - Show failed servers in tools attribution
- **UI & Status Updates**:
  - MCP status messages during tool discovery and execution
  - Track MCP call counts with generic "MCP call #N" messages
  - Settings modal integration for GPT-5 models
  - Beta feature notice in documentation

#### Changed
- Updated README with MCP configuration instructions and Slack scope requirements
- Enhanced MCP config example with comprehensive documentation
- Added MCP metrics gathering for monitoring

## [2.1.5] - 2025-09-30

### 🐛 Bug Fix - Message Pagination

#### Fixed
- **Overflow Message Display**: Fixed continuation messages not appearing in thread when response exceeded Slack's message length limit
  - Changed thread_id parameter from thinking_id (status message timestamp) to message.thread_id (actual thread timestamp)
  - Continuation messages now properly appear in correct thread and trigger pagination if still too long
  - Full message content was always correctly stored in database - this was purely a display bug affecting Slack message delivery

## [2.1.4] - 2025-09-24

### 🎯 Configuration, Session Management & Licensing Update

#### Added
- **MIT License**: Added open source MIT license to the project
- **Database Directory Configuration**: New `DATABASE_DIR` environment variable for customizable database location
- **Modal Session Database Storage**: Settings modal sessions now stored in database instead of Slack metadata
- **Modal Session Cleanup**: Automatic cleanup of orphaned settings modal sessions during daily maintenance

#### Fixed
- **Hardcoded Timeouts Removed**: All text operations now respect configured `API_TIMEOUT_STREAMING_CHUNK` value instead of hardcoded 150s
- **Dead Code Cleanup**: Removed unused `text_high_reasoning` operation type that was never utilized
- **Slack Metadata Size Limits**: Resolved issues with oversized private_metadata by moving session data to database

#### Changed
- **Settings Modal Architecture**: Migrated from storing full session state in Slack's private_metadata to database-backed sessions with UUID references
- **Timeout Configuration**: Text operations (intent classification, prompt enhancement, normal text, text with tools) now use environment-configured timeouts
- **Database Path Flexibility**: Database and backup directories now use configurable path from `DATABASE_DIR` setting

## [2.1.3] - 2025-09-18

### 🐛 Settings & Configuration Fixes

#### Fixed
- **Default Values Correction**: Fixed incorrect default values for `reasoning_effort` and `verbosity` in user preferences
- **Settings Modal Defaults**: Ensured proper default values are applied when creating new user preferences

## [2.1.2] - 2025-09-17

### 🔧 Logging & Thread Safety Improvements

#### Fixed
- **Logger Thread Safety**: Updated logger implementation for async/thread safety paradigms after refactor
- **Log Rotation Issues**: Fixed problems with log file rotation under concurrent access
- **Import Errors**: Fixed missing imports in refactored modules

#### Changed
- **Message Processor Restoration**: Reverted accidental restoration of monolithic message processor, re-applied modular version

## [2.1.1] - 2025-09-16

### 🚀 Enhanced Streaming Reliability & UX Improvements

#### Fixed
- **User Context**: Fixed user timezone/context not being injected after async refactor
- **Settings Modal**: Fixed reasoning level being lost on mobile when toggling web search
- **Streaming Reliability**: Fixed text truncation when Slack API updates fail (17/18 success case)
- **Message Overflow**: Fixed transitions with proper continuation handling
- **Part Labels**: Fixed "Part X" labels disappearing during streaming updates
- **Loading Indicators**: Fixed enhanced prompt loading indicators not being removed properly

#### Changed
- **Timeout Adjustments**: Increased all text operation timeouts to 2.5 minutes minimum
- **Progress Feedback**: Added humorous progress messages after 30s and 60s+ for long operations
- **Image Analysis**: Added progress monitoring to image analysis operations
- **Timeout Handling**: Improved to only warn (never fail) on chunk timeouts

## [2.1.0] - 2025-09-16

### 🎉 Major Async/Await Refactor & Critical Stability Fixes

#### Changed
- **Async/Await Migration**: Migrated critical components to async/await pattern to fix concurrency issues
- **Thread Management**: Added AsyncThreadStateManager and AsyncThreadLockManager for proper synchronization
- **Database Operations**: Implemented async database methods running in parallel with sync versions

#### Fixed
- **Database Commits**: Fixed missing commits in async methods (save_thread_config_async, cache_message_async, etc.)
- **Settings Modal**: Fixed not preserving pending messages for new user flow
- **Web Search Persistence**: Fixed checkbox not persisting after save
- **Boolean Conversions**: Fixed issues in async database methods
- **Thread Config**: Fixed retrieval issues under concurrent load
- **Race Conditions**: Eliminated crashes under concurrent load

#### Added
- **Comprehensive Testing**: Expanded test coverage for async operations
- **Load Testing**: Verified stability under production workloads

## [2.0.4] - 2025-09-16

### 🐛 Critical Bug Fix - Bot Hanging Resolution

#### Fixed
- **Removed problematic `timeout_wrapper` that was causing zombie threads and bot hanging**
  - The wrapper was creating daemon threads that continued running after timeouts
  - These threads held HTTP connections, eventually exhausting the connection pool
  - Bot would become unresponsive after multiple timeouts, requiring manual restart
- Now using OpenAI SDK's native timeout handling via httpx
- Bot no longer hangs after consecutive timeout errors

#### Changed
- Improved timeout error messages to clearly indicate OpenAI as the source
  - "OpenAI Timeout" instead of generic "Taking Too Long"
  - "OpenAI's API is not responding" with specific timeout duration
  - All user-facing messages now explicitly mention OpenAI service issues
- Updated tests to remove references to deleted `timeout_wrapper`

#### Added
- Integration tests for intent classification model comparison
- Better timeout tracking and logging for diagnostics

## [2.0.3] - 2024-12-15

### 🔧 Code Quality & Reliability Improvements

#### Changed
- Refactored codebase to improve maintainability and reliability
- Cleaned up unused imports across all modules
- Fixed unused variables (`channel`, `truncated`, `content_preview`, `removed_msg`, etc.)
- Replaced bare except clauses with specific `Exception` handling
- Cleaned up f-string placeholders without variables
- Improved custom instructions handling in main prompt

#### Added
- Comprehensive timeout error handling test suite (`test_timeout_error_handling.py`)
- 586 new test cases covering various error scenarios
- Better error context and recovery strategies

#### Fixed
- All linting issues identified by ruff and pyright diagnostics
- Improved exception propagation throughout the codebase

## [2.0.2] - 2024-12-14

### 🐛 Bug Fixes

#### Fixed
- Prevented infinite retry loop on OpenAI timeout errors
- Reduced duplicate logging in error scenarios
- Improved timeout handling with proper circuit breaker implementation

## [2.0.1] - 2024-12-13

### ✨ Features & Documentation

#### Added
- Context-aware vision enhancement for better screenshot handling
- Slack app manifest file for easy app configuration
- Slack app commands documentation in README

#### Changed
- Made vision prompt enhancement more intelligent based on image context
- Improved handling of screenshot analysis

#### Developer
- Added debugging capabilities for Slack shortcut handlers (later reverted)

## [2.0.0] - 2024-09-12

### 🎉 Major Release - Complete V2 Rewrite

This release represents a complete rewrite of the ChatGPT Bots project, focusing on production stability, user experience, and advanced AI capabilities.

### ✨ New Features

#### Core Architecture
- **Responses API Migration**: Migrated from OpenAI's Chat Completions API to the new Responses API for advanced tool calling. The Chat Completions API is now deprecated.
- **Stateless Design**: Platform (Slack) as source of truth with dynamic context rebuilding
- **Abstract Base Client**: Modular architecture supporting multiple platforms
- **SQLite Persistence**: User preferences, thread settings, and message caching
- **Thread Management**: Concurrent request handling with proper locking mechanisms

#### User Experience
- **Interactive Settings Modal**: Configure preferences via `/chatgpt-settings` command
- **Thread-Specific Settings**: Different configurations per conversation
- **Custom Instructions**: Personalized AI behavior per user
- **Multi-User Context**: Proper handling of shared conversations with username tracking
- **Welcome Flow**: First-time user onboarding with guided setup

#### AI Capabilities
- **Intelligent Intent Classification**: Automatic detection of image/text/vision/edit requests
- **Image Generation & Editing**: Natural language image creation and modification
- **Vision Analysis**: Process uploaded images with detailed descriptions
- **Document Processing**: Extract and analyze PDFs, Office files, code files
- **Web Search Integration**: Current information retrieval (GPT-5 models)
- **Streaming Responses**: Real-time message updates with circuit breaker protection

#### Models & Configuration
- **Multi-Model Support**: GPT-5, GPT-5 Mini, GPT-4.1, GPT-4o
- **Dynamic Parameters**: Model-specific settings (reasoning_effort, verbosity for GPT-5)
- **Token Management**: Smart trimming with configurable thresholds
- **Utility Models**: Separate models for different tasks (analysis, utilities)

### 🔧 Technical Improvements

#### Performance
- Thread-safe operations with comprehensive locking
- SQLite WAL mode for concurrent database access
- Automatic message trimming at 80% token capacity
- Circuit breaker pattern for streaming failures

#### Testing
- 100+ unit tests with 80%+ coverage
- Integration tests for OpenAI API interactions
- Load testing verified with production workloads
- Comprehensive test fixtures and mocks

#### Developer Experience
- Makefile for common operations
- Structured logging with rotation
- Environment-based configuration
- Comprehensive error handling
- Type hints throughout codebase

### 📝 Configuration Changes

#### New Environment Variables
- `SETTINGS_SLASH_COMMAND`: Customizable settings command
- `DEFAULT_REASONING_EFFORT`: GPT-5 reasoning depth
- `DEFAULT_VERBOSITY`: Response detail level
- `UTILITY_REASONING_EFFORT`: For quick operations
- `ANALYSIS_REASONING_EFFORT`: For complex tasks
- `TOKEN_BUFFER_PERCENTAGE`: Dynamic token limits
- `ENABLE_WEB_SEARCH`: Web search capability
- `ENABLE_STREAMING`: Real-time responses
- Multiple streaming configuration options

#### New Slack Scopes
- `groups:history`: Private channel access
- `users:read`: Workspace member information
- `users:read.email`: Email address access

### 🐛 Bug Fixes
- Fixed race conditions in concurrent message processing
- Resolved settings persistence issues under load
- Fixed scope selection logic for new vs existing users
- Addressed oversized Slack message handling
- Fixed thread context mixing in shared conversations

### 📚 Documentation
- Comprehensive README with setup instructions
- Detailed CLAUDE.md for AI assistant guidance
- SQLite integration plan
- User settings modal design document
- Responses API implementation details
- Test documentation and templates

### ⚠️ Breaking Changes
- Discord support temporarily removed (V2 rewrite in progress)
- Changed from Chat Completions to Responses API
- New database schema
- Updated environment variable structure
- Modified logging configuration

### 🔄 Migration Guide

1. **Database Migration**: No migration path from V1 - fresh install
2. **Environment Variables**: Update `.env` using `.env.example` as template
3. **Slack App**: Add new required scopes in Slack App settings
4. **Model Selection**: Choose appropriate GPT model and defaults in configuration
5. **Custom Instructions**: Users should configure via `/chatgpt-settings`


### 🙏 Acknowledgments
Special thanks to all testers who participated in load testing and helped identify edge cases.

---

## Previous Versions

For changes prior to v2.0.0, please refer to git history.