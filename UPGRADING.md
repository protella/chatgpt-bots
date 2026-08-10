# Upgrading

## v2.x → v3.0.0

Follow the steps in order — steps 2 and 5 are the ones that bite.

> **If you have an existing clone:** `master` was rewritten during the v3 cycle, so a normal
> `git pull` will conflict. Re-clone, or `git fetch origin && git reset --hard origin/master`.
> Feature branches other than `master` were retired.

**0. Back up your database first.** v3's first boot runs irreversible migrations. The bot
now takes its own `pre-v3-upgrade` backup before touching anything, but a copy of
`data/slack.db` you made yourself is the one you can trust.

**1. Update dependencies**
```bash
make install   # pip install --require-hashes -r requirements.txt (openai >= 2.53.0)
```

**1b. Install the system packages** — new in v3, and easy to miss because everything
"works" without them:
```bash
apt-get install poppler-utils tesseract-ocr pandoc   # Linux
brew install poppler tesseract pandoc                # macOS
```
`tesseract` + `poppler` are what make scanned PDFs readable (`ENABLE_PDF_OCR`, **on by
default**); `pandoc` is the `.docx` fallback extractor. Without them those documents quietly
degrade to a "couldn't extract text" note instead of erroring — so a host built without them
silently loses the capability. (`python-magic` is *no longer* a dependency, so libmagic is no
longer needed.)

**1c. Check your SQLite** — `python3 -c "import sqlite3; print(sqlite3.sqlite_version)"` must
report **3.35 or newer**. The migration that removes stored document text uses
`ALTER TABLE … DROP COLUMN`, which doesn't exist below 3.35; on an older SQLite the bot logs a
warning, carries on, and *leaves document content in the database* — exactly what v3 promises
it won't do.

**2. Update your `.env`** — compare against the reorganized `.env.example`. Everything new
has a sane default if omitted; the items that matter:

Changed values (update if you set them explicitly):
```
OPENAI_API_KEY=sk-...            # standard name adopted; your existing OPENAI_KEY still works
GPT_MODEL=gpt-5.6-sol            # was gpt-5.5
UTILITY_MODEL=gpt-5.6-luna       # was gpt-5-mini
UTILITY_REASONING_EFFORT=low     # was minimal ("minimal" is rejected by 5.6 models); "low" on
                                 # Luna is adaptive — zero reasoning tokens on trivial verdicts
DEFAULT_REASONING_EFFORT=medium  # was low in the old example — medium is the intended default
TOKEN_CLEANUP_THRESHOLD=0.5      # was 0.9 — thread compaction now triggers at half the budget
TOKEN_COMPACTION_TARGET=0.4      # was 0.7 — must stay BELOW the threshold or compaction
                                 # never converges
```

Delete — nothing reads these anymore (the ones below the first block only ever existed in
v3 preview builds; if you never ran one, you won't have them):
```
DISCORD_TOKEN / DISCORD_ALLOWED_CHANNEL_IDS / DISCORD_LOG_LEVEL
GPT4_MAX_TOKENS
THREAD_MAX_TOKEN_COUNT
ENABLE_VISION_ENHANCEMENT

MAX_UNPROMPTED_REPLIES_PER_HOUR / PARTICIPATION_SNOOZE_HOURS
ENABLE_MULTIMODAL_GATE / GATE_VISION_*
ENABLE_MENTION_PLACEMENT_MODEL
ENABLE_BACKGROUND_IMAGE_GEN
SNOOZE_ACK_EMOJI / PARTICIPATION_CUSTOM_EMOJI_CAP / EMOJI_USAGE_FLUSH_SECONDS
PULSE_TAIL_TEXT_TRUNCATE / PARTICIPATION_ADDRESSEE_TAIL
CHANNEL_PULSE_ENVELOPE_MAX / CHANNEL_PULSE_SIZE
```

New keys worth a decision (see `.env.example` for the full annotated list — every knob is
documented inline there):
- `ENABLE_CHANNEL_LISTENING` — the master switch for teammate behavior in channels. **Code
  default `false`: an upgraded `.env` without the key behaves exactly as before (mentions +
  DMs).** The shipped `.env.example` sets it `true`, and with listening on, channels default
  to full participation — see the next key.
- `CHANNEL_RESPONSE_MODE=auto_respond` — what a channel does before anyone picks a level for
  it: `auto_respond` (full teammate — the default), `tag_only` (mentions only), or `off`.
  Any member can change a channel's level via the ⚙️ button, and "quiet down" feedback is
  remembered as that channel's standing policy.
- `BOT_NAME_ALIASES=ChatGPT` — names the bot answers to without an `@`. **Set this per
  environment** (e.g. `ChatGPT-Dev` on a dev install), or the dev bot will answer to the
  prod bot's name.
- `ENABLE_DEEP_RESEARCH=true` — **on by default, and it works in DMs too.** Each job is
  minutes of model time at `high` effort; turn it off if that's not a bill you want.
- `SLACK_NATIVE_STREAMING` — Slack's native streaming API (no "(edited)" marks on streamed
  replies). Code default `false`; the shipped `.env.example` sets it `true`. Recommended on
  once you've seen it work in your workspace — the classic edit-loop remains the fallback.
- `ENABLE_AMBIENT_MEMORY=true` — the bot quietly keeps notes on links, images, and files
  shared in channels it's in; per-channel opt-out in the ⚙️ Configure modal.
- `ENABLE_EDIT_TRIGGERED_REPLIES=false` — meaningful edits to recent messages get the same
  respond/ignore judgment as new messages (typo fixes stay silent).
- `PARTICIPATION_REASONING_EFFORT=medium` — how hard it thinks about whether a message is
  for it. Resolving who a message is addressed to is the hardest call it makes, and `low`
  gets it wrong measurably more often. Do not raise it to `high`: this is not monotonic,
  and `high` scored *worse* than `low` on the same replay (it reasons its way around the
  addressee rules).
- `ENABLE_LINK_PREVIEWS=false` — links in the bot's posts stay inline; set true for Slack's
  preview cards (a change from v2, where Slack unfurled them).
- `STATUS_LOADING_MESSAGES_FILE` — optional branded "working…" messages for the thread
  status indicator: point it at your own text file, one message per line, plain text (no
  emoji — the status surface doesn't render them). Unset = a bundled set of 100 generic
  ones (`status_messages/loading_messages.generic.txt`).
- `ENABLE_FEEDBACK_BUTTONS=false` — 👍/👎 under DM/assistant responses, **off by default**;
  thumbs reactions on bot messages are recorded passively either way.

**3. Migrate `mcp_config.json` secrets (recommended, not breaking)** — literal keys still
work, but you can now keep them in `.env`:
```
"X-API-Key": "${YOUR_VAR}"     # in mcp_config.json (any var name you like)
YOUR_VAR=sk-...                # in .env
```
Also new: per-server `"enabled": false`, and auth uses the `headers` object shape
(see README — the previously documented `authorization` shape never worked).

**4. Rebuild your Slack app manifest and reinstall.** Copy
`slack_app_manifest.example.yml` over your environment copy (keep your names/commands) and
reinstall the app — the new scopes need re-consent, and a missing one degrades a feature
silently. New since v2.5: the `agent_view` block; bot scopes
`search:read.public/private/im/mpim/files/users`, `reactions:read`, `reactions:write`,
`pins:read`, `users:read.email`, `chat:write.customize`, `assistant:write`, `emoji:read`,
`canvases:read`, `canvases:write`, and `users.profile:read` (custom profile fields +
pronouns; nothing consumes it yet — harmless to skip until a feature does); and events
`reaction_added`, `reaction_removed`, `app_home_opened`, `app_context_changed`,
`file_deleted` (ambient-memory cleanup), and `member_joined_channel` (the bot's one-time
intro when it's added to a channel — without it the intro never fires). **Remove the legacy
`assistant_thread_started` / `assistant_thread_context_changed` events** — Slack's manifest
validator now rejects them alongside `agent_view` (the code keeps no-op handlers, so nothing
breaks either way).

Two of those are easy to skip and annoying to debug: **`chat:write.customize`** is what puts
the "[research: …]" byline on findings posts (without it the bot silently falls back to plain
posts), and **`users:read.email`** is what lets it answer "what's her email?" instead of
shrugging.

**5. First startup migrates the database automatically.** It takes a `pre-v3-upgrade` backup
into `data/backups/` before touching anything, then two more tagged backups before each
destructive step. Watch for these lines, in order:
- `DB: Pre-v3 database detected — backup tagged pre-v3-upgrade before migrating` — **this is
  your rollback point.** Keep it.
- `DB: One-time GPT-5.6 migration — swapped N user(s) to gpt-5.6-sol with medium reasoning`
  — everyone moves to the new default (their old model/effort choice is not preserved; they
  can re-pick globally, per channel, and per thread afterward)
- `Created backup: …pre-v3-mirror-drop…` → `DB: Mirror-drop migration complete — removed N
  cached message row(s)` — the DB stops storing conversation transcripts
- `Created backup: …pre-v3-doc-content-drop…` → `DB: Doc-content-drop migration complete` —
  the DB stops storing document content
- Anything reading `DB: Migration step '<name>' FAILED` means that step did not complete.
  The remaining steps still run, and the bot will start — but don't leave it there.

Existing channel "ground rules" fold into the new standing channel policy at first start
(the bot refuses to start if that migration fails rather than silently dropping your rules),
and old participation levels map onto the new `on` / `mentions_only` / `off` scale.

From then on the database backs itself up nightly (7-day retention) as part of the scheduled
cleanup, which it never did before. The three `pre-v3-*` backups are **exempt from that
retention** — they're your rollback path, so nothing deletes them but you.
