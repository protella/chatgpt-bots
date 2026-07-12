# v3.0.0 Production Deployment Checklist

Everything that has to happen on **beastbox** beyond `git pull`. Written 2026-07-11 from a
read-only survey of the live prod box. Nothing here has been executed yet.

> **Living document.** v3 work is still landing (pinning/canvas support among it). Re-check the
> `.env` deltas (step 6) and manifest scopes (step 9) against `.env.example` and
> `slack_app_manifest.example.yml` immediately before running the upgrade — those two files are the
> source of truth, this is the operator's map.

## Prod as it stands today

| | |
|---|---|
| PM2 process | id **4**, name `SlackBot`, `fork` mode, **runs as root** (`/root/.pm2`), online 4d, 0 restarts |
| Path | `/home/blackhawk/environments/chatgpt-bots` (the venv **is** the project dir — `./bin/python3` runs `./slackbot.py`) |
| Owner | Files owned by **blackhawk**; PM2 owned by **root**. → run `git`/`pip` as blackhawk, `pm2` under `sudo` |
| Git | `master` @ `b8eb44b` = **v2.5.1**, with one local modification: `extract_metrics.py` (**blocks the pull** — see step 3) |
| Runtime | Python **3.12.3**, SQLite **3.45.1** (clears the 3.35 floor the doc-content migration needs) |
| Database | `data/slack.db` = **1.9 GB**; `data/backups/` = 2.7 GB; disk has **521 GB free** — space is not a constraint |
| System deps | `pdftoppm` ✅ · `pandoc` ✅ · **`tesseract` ❌ MISSING** |
| MCP servers | `context7`, `aws_knowledge`, `datassential-production-ai` (literal key in `headers`) |
| Status messages | **No `status_messages/` dir at all** — the branded file is gitignored and must be copied by hand |
| Log levels | `SLACK_/BOT_/UTILS_LOG_LEVEL = DEBUG` |

---

## 1. Pre-flight (do before touching anything)

- [ ] **Push the 86 local commits** and confirm `origin/master` has them (prod pulls from
      `https://github.com/protella/chatgpt-bots.git`).
**Feature posture for prod (decided):** everything **on**, *except* the feedback strip.
- `ENABLE_CHANNEL_LISTENING = "true"` — the bot participates in channels it's invited to.
  **This makes the `message.channels` / `message.groups` / `message.mpim` events and the
  `channels:history` / `groups:history` / `mpim:history` scopes REQUIRED** in the manifest (step 9),
  not optional. Per-channel behavior is still governed by `CHANNEL_RESPONSE_MODE` (default
  `tag_only`) and the ⚙️ Configure button, so channels start conservative.
- `ENABLE_DEEP_RESEARCH = "true"` — live in channels and DMs.
- `ENABLE_FEEDBACK_BUTTONS = "false"` — **off**: no 👍/👎 strip under DM/assistant replies.
  (Thumbs *reactions* on bot messages are still recorded passively — that's a separate,
  zero-cost path and needs no flag.)

## 2. Take a manual backup and stop the bot

```bash
ssh beastbox
cd /home/blackhawk/environments/chatgpt-bots
sudo pm2 stop 4                                  # SlackBot
cp data/slack.db ~/slack.db.pre-v3.$(date +%F)   # your own rollback copy — trust this one
```
The code now takes its own `pre-v3-upgrade` backup too, but take yours anyway.

## 3. Update the code (the pull will FAIL without this)

`extract_metrics.py` is **modified in prod** (122 added lines) and **deleted in v3** — git will
refuse to merge. The prod copy is the only place those edits exist, and the script cannot work
against v3 anyway (it queries the `messages` table, which the migration drops).

```bash
cp extract_metrics.py ~/extract_metrics.prod-final.py   # keep the prod edits somewhere
git checkout -- extract_metrics.py                      # discard so the pull can proceed
git pull origin master                                  # as blackhawk, NOT sudo
```
The pull also removes the v2 leftovers automatically (`discordbot.py`, `legacy/`,
`migrate_to_gpt54.py`, `markdown_to_mrkdwn/`). Untracked junk (`app_broken.log`,
`broken_error.log`, `mcp_config.json.bak`, `metrics_reports/`) stays — delete by hand if you want.

## 4. System package (OCR)

```bash
sudo apt-get install -y tesseract-ocr    # poppler-utils + pandoc already present
```
Without it, `ENABLE_PDF_OCR=true` degrades **silently** — scanned PDFs just come back as
"text not extractable". Verify: `tesseract --version`.

## 5. Python dependencies

```bash
./bin/pip install --require-hashes -r requirements.txt    # as blackhawk; installs into the in-place venv
```
New since v2.5.1: `pytesseract`, `pdf2image`, `aiofiles`, and an `openai >= 2.45` bump.
`python-magic` is gone (so libmagic is no longer needed).

## 6. `.env` — edit in place, never overwrite

**Do not copy `.env.example` over it.** Prod's `.env` holds the live secrets
(`OPENAI_KEY`, `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, `DATASSENTIAL_MCP_KEY` if you move it there).
Edit the keys below and leave every secret line untouched. Prod uses `KEY = "value"` spacing.

### 6a. Change — models and defaults, aligned to dev
```
GPT_MODEL = "gpt-5.6-sol"          # was gpt-5.5
UTILITY_MODEL = "gpt-5.6-luna"     # was gpt-5-mini  ← MUST change; gpt-5-mini is gone from v3
UTILITY_REASONING_EFFORT = "none"  # was low
UTILITY_VERBOSITY = "low"
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_VERBOSITY = "low"
ANALYSIS_REASONING_EFFORT = "medium"
ANALYSIS_VERBOSITY = "medium"
GPT_IMAGE_MODEL = "gpt-image-2"    # unchanged
```
Everything else model-shaped already matches dev (temperature 1.0, top_p 1.0, image size/quality/
format/fidelity, detail level, empty `WEB_SEARCH_MODEL`).

### 6b. Delete — dead in v3 (read by nothing)
```
DISCORD_TOKEN / DISCORD_ALLOWED_CHANNEL_IDS / DISCORD_LOG_LEVEL
THREAD_MAX_TOKEN_COUNT
REPORTPRO_SLASH_COMMAND
ELEVENLABS_KEY          # unused by this bot
OPENAI_KEY_PERSONAL     # unused by this bot
```

### 6c. Add — prod-specific, must be set explicitly
```
BOT_NAME_ALIASES = "ChatGPT"                                          # prod bot has NO "-dev"
STATUS_LOADING_MESSAGES_FILE = "status_messages/loading_messages.datassential.txt"
ENABLE_CHANNEL_LISTENING = "true"                                     # all features on in prod
ENABLE_DEEP_RESEARCH = "true"
ENABLE_FEEDBACK_BUTTONS = "false"                                     # the one feature we leave off
SLACK_NATIVE_STREAMING = "false"                                      # validate live before enabling
```
The other ~84 new keys can be omitted — each has a working default (and every remaining feature
defaults **on**, which is the posture we want), and `.env.example` documents each one inline if you
prefer to pin them explicitly.

### 6d. Consider — log levels
Prod runs `DEBUG` across the board. v3 is considerably chattier (channel pulse, participation
judgments, tool loop), and DEBUG logs message content. Recommend `INFO` for
`SLACK_LOG_LEVEL` / `BOT_LOG_LEVEL` / `UTILS_LOG_LEVEL`.

## 7. Status message files

`status_messages/generic` + `pipeline_messages.txt` arrive with the pull. The **branded file is
gitignored**, so copy it from dev — without it, prod silently falls back to the generic pool:

```bash
# from the dev box:
scp status_messages/loading_messages.datassential.txt \
    beastbox:/home/blackhawk/environments/chatgpt-bots/status_messages/
```
Then confirm `STATUS_LOADING_MESSAGES_FILE` (6c) points at it.

## 8. `mcp_config.json` — one description edit to carry over

The file is untracked on both boxes, so the pull won't touch it. Both have the same three servers
(`context7`, `aws_knowledge`, `datassential-production-ai`) at the same URLs. Two differences:

1. **The description edit** (this is the one that must go over). Dev's Datassential
   `server_description` gained **"and reports"**:
   - prod: `…restaurant industry data, consumer preferences…`
   - dev:  `…restaurant industry data and reports, consumer preferences…`
2. **The key**: prod stores the Datassential API key **literally** in the JSON; dev uses a
   `"${DATASSENTIAL_MCP_KEY}"` placeholder resolved from `.env`.

> ⚠️ **Do not just `scp` dev's file over.** It carries the `${DATASSENTIAL_MCP_KEY}` placeholder, and
> if prod's `.env` has no such variable the server is **skipped at load** with a warning — Datassential
> would silently go dark. Pick one:

**Option A — minimal (edit in place).** Change only the `server_description` string in prod's
`mcp_config.json`; leave the literal key alone. Nothing else changes.

**Option B — hardening (recommended).** Copy dev's file over **and** move the key out of the JSON:
add `DATASSENTIAL_MCP_KEY = "<the same key already in prod's mcp_config.json>"` to prod's `.env`
first, then copy. Verify at boot that the startup probe reports `datassential-production-ai` as
reachable — if the key didn't resolve, the log names the missing variable.

## 9. Slack app — rebuild the manifest and reinstall

> **The merged manifest is already on the box**: `slack_app_manifest.v3.yml` in the prod folder.
> Copy its contents into api.slack.com/apps → *ChatGPT Slackbot* → **App Manifest**, save, reinstall.
> Prod's previous manifest is untouched at `slack_app_manifest.yml` (your rollback reference).
> It keeps prod's identity (name, colour, `ChatGPT` display name, `/chatgpt-settings`,
> `configure_thread_settings`) and adds the v3 surface. It also **drops the four unused user
> scopes** — prod has no user token and no code reads one; paste the `user:` block back if you
> want them retained.

The prod Slack app is a **separate app** from dev. Its manifest needs the v3 scopes and events, and
**a missing scope degrades a feature silently rather than erroring**. The reference list below is
what that file contains.

Add to bot scopes:
```
assistant:write · chat:write.customize · pins:read · reactions:read · reactions:write
users:read.email · emoji:read · channels:read · groups:read · mpim:read
channels:history · groups:history · mpim:history      ← REQUIRED: channel listening is on
search:read.public · search:read.private · search:read.im · search:read.mpim
search:read.files · search:read.users
bookmarks:read · bookmarks:write · canvases:read · canvases:write · pins:write
```
The last line is **staged for the pinning/canvas work still in flight** — no v3 code calls those
APIs yet. They're included deliberately so this one reinstall covers that feature too, instead of
needing a second reinstall in a week.
Add to event subscriptions:
```
reaction_added · reaction_removed · app_home_opened · app_context_changed
message.channels · message.groups · message.mpim     ← REQUIRED: channel listening is on in prod
```
Also add the `agent_view` block (agent description + suggested prompts) from
`slack_app_manifest.example.yml`. **Reinstall the app to the workspace** so the new scopes take effect.

The two that fail quietly if forgotten: **`chat:write.customize`** (the "[research: …]" byline on
findings posts) and **`users:read.email`** (people lookups).

## 10. First boot — watch the migration

```bash
sudo pm2 restart 4 && sudo pm2 logs SlackBot --lines 100
```
The 1.9 GB database means this boot is **slow** (two backups + two `VACUUM`s). Expect these lines in
order:

```
DB: Pre-v3 database detected — backup tagged pre-v3-upgrade before migrating   ← your rollback point
Created backup: data/backups/slack_pre-v3-upgrade_<ts>.db
DB: One-time GPT-5.6 migration — swapped N user(s) to gpt-5.6-sol with medium reasoning
DB: Backfilled settings_completed=1 for N pre-existing user(s)
Created backup: data/backups/slack_pre-v3-mirror-drop_<ts>.db
DB: Mirror-drop migration complete — removed N cached message row(s), reclaimed N bytes
Created backup: data/backups/slack_pre-v3-doc-content-drop_<ts>.db
DB: Doc-content-drop migration complete — synthesized N summary(ies)
```
Any `DB: Migration step '<name>' FAILED` line means that step did not complete — the bot will still
start, but stop and investigate.

**Expect the DB to shrink a lot** (the message mirror and all document content are dropped).
Everyone's model/effort resets to `gpt-5.6-sol` / `medium` — that's intended, and users can re-pick.

## 11. Verify

- [ ] `sudo pm2 list` → SlackBot **online**, restarts 0
- [ ] DM the bot → it replies; `/chatgpt-settings` opens and shows the 5.6 models
- [ ] Upload a PDF → summary appears; ask a specific question → it re-reads the file
- [ ] Ask something research-worthy → status card appears, findings post lands with the byline
      (byline missing ⇒ `chat:write.customize` didn't take)
- [ ] MCP: startup log shows one reachable/unreachable line per server; a Datassential question works
- [ ] Status bubble shows the **Datassential** loading messages, not the generic ones
- [ ] **Channel listening**: in a channel the bot is in, @-mention it → it answers; post an unrelated
      human-to-human message → it stays out. (If it answers nothing at all, the `message.channels`
      event subscription didn't take.)
- [ ] **No 👍/👎 strip** under DM replies (feedback buttons are off in prod)
- [ ] Next day: `Scheduled database backup complete (7-day retention)` appears in the log

## 12. Rollback

```bash
sudo pm2 stop 4
cp ~/slack.db.pre-v3.<date> data/slack.db          # or data/backups/slack_pre-v3-upgrade_<ts>.db
git checkout b8eb44b                                # v2.5.1
./bin/pip install -r requirements.txt               # v2.5.1 lockfile
sudo pm2 restart 4
```
The three `pre-v3-*` backups are **exempt from the nightly 7-day retention** — nothing deletes them
but you. (Prod's existing `slack_manual_backup_*` files are also tagged, so they're preserved too;
only untagged nightly backups age out.)

---

## 13. GitHub release (after prod is verified)

**Conventions from history** — tag `vX.Y.Z`, release title `vX.Y.Z - Short Description`
(e.g. "v2.5.1 - GPT-5.5 Support"), body in `## 🚀 Feature - …` / `### Added|Changed|Fixed` sections
with an `## ⚠️ Upgrade Instructions` block. Keep it user-visible: no pricing sections, no
"not supported" lists, no internal refactor detail.

- [ ] Cut `## [Unreleased]` → `## [3.0.0] - YYYY-MM-DD` in CHANGELOG.md, commit
- [ ] `git tag v3.0.0 && git push origin v3.0.0`
- [ ] `gh release create v3.0.0 --title "v3.0.0 - Channel Teammate, Deep Research & GPT-5.6" --notes "…"`
- [ ] **Update the repo description** — the current one predates deep research:
      *"ChatGPT-powered Slack bot with image generation, vision analysis, document processing, and
      channel participation — built on OpenAI's Responses API."*
- [ ] **Fix the repo topics** — they still advertise Discord, which v3 removed:
      drop `discord`, `discord-bot`, `cli`; consider adding `mcp`, `gpt-5`, `deep-research`
