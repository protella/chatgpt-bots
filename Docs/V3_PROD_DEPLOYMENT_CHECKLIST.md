# v3.0.0 Production Deployment Checklist

Everything that has to happen on **beastbox** beyond `git pull`. Written 2026-07-11 from a
read-only survey of the live prod box. Nothing here has been executed yet.

> **Living document.** v3 work is still landing (pinning/bookmarks among it; canvas support has
> since merged, and so has the single-stream channel rebuild). Re-check the
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
New since v2.5.1: `pytesseract`, `pdf2image`, `aiofiles`, `striprtf` (F49 .rtf extraction),
`tiktoken` (the channel admission estimate counts real o200k tokens; without it the estimate
degrades to a byte ratio), and an `openai >= 2.45` bump. `python-magic` is gone (so libmagic is no
longer needed).

**Do not skip this install even if the code "looks" already pulled.** `beautifulsoup4` was
added to the lockfile in the pre-release hardening pass — it's imported for canvas parsing but
was previously undeclared, so a host that doesn't re-run the install will import-fail its
fallback and silently return raw HTML as canvas "markdown". The `--require-hashes` install
above pulls it (and `soupsieve`); confirm with `./bin/python -c "import bs4; print(bs4.__version__)"`.

## 6. `.env` — edit in place, never overwrite

**Do not copy `.env.example` over it.** Prod's `.env` holds the live secrets
(`OPENAI_KEY`, `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, `DATASSENTIAL_MCP_KEY` if you move it there).
Edit the keys below and leave every secret line untouched. Prod uses `KEY = "value"` spacing.

### 6a. Change — models and defaults, aligned to dev
```
GPT_MODEL = "gpt-5.6-sol"          # was gpt-5.5
UTILITY_MODEL = "gpt-5.6-luna"     # was gpt-5-mini  ← MUST change; gpt-5-mini is gone from v3
UTILITY_REASONING_EFFORT = "low"   # adaptive on Luna: 0 reasoning tokens on trivial verdicts, thinks only when needed (benchmarked 2026-07-16)
UTILITY_VERBOSITY = "low"
PARTICIPATION_REASONING_EFFORT = "medium"   # was "low" — see below
DEFAULT_REASONING_EFFORT = "low"
DEFAULT_VERBOSITY = "low"
ANALYSIS_REASONING_EFFORT = "medium"
ANALYSIS_VERBOSITY = "medium"
GPT_IMAGE_MODEL = "gpt-image-2"    # unchanged
DEFAULT_DETAIL_LEVEL = "auto"      # unchanged — and `auto` IS the max, see below
TOKEN_CLEANUP_THRESHOLD = "0.5"    # was 0.9 — owner 2026-08-03: start thread compaction at 50%; quality degrades long before the limit
TOKEN_COMPACTION_TARGET = "0.4"    # was 0.7 — must stay BELOW the threshold or compaction never converges
```

**`OPENAI_SERVICE_TIER` exists and ships `standard`.** `fast` buys up to 2.5x faster output on
`gpt-5.6-sol` at **double the token cost**, on the user-facing responder call only. Leave it
`standard` in prod unless the owner has decided to pay for it.

**Leave `DEFAULT_DETAIL_LEVEL` at `auto`.** On the 5.6 family `auto` (and an omitted detail) is
equivalent to `original`: the image is sent at its own dimensions with no resize. `high` and `low`
both resize under a finite limit, so pinning `high` here would *cap* large screenshots rather than
sharpen them. Set it to `high` only as a deliberate cost cap on very large images.

**`PARTICIPATION_REASONING_EFFORT` is `medium`, not `low`.** It is the one effort that does NOT
follow `UTILITY_REASONING_EFFORT`. The gate decides whether the bot speaks at all in a channel, and
at `low` it misresolves who a message is addressed to: measured on a replay of a real 2026-07-25
misfire, `low` got 46/66 scenarios right and `medium` 49/66, and `low` was the setting that let the
bot answer a message aimed at the humans in the room. Note the curve is NOT monotonic — `high`
scored *worse* than `low` on the headline case, because the extra reasoning talks itself past the
addressee rules. Do not "improve" this to `high`.

**`GATE_VISION_DETAIL` is gone — do not set it.** The gate no longer looks at images at all; it
asks one question, wake or don't. Image description is now the ambient-memory path (`ANALYSIS_*`
efforts above, `ENABLE_AMBIENT_IMAGE_MEMORY`), which runs off the debounce hot path, so the
cost/fidelity trade the old variable existed to make is no longer a knob anyone has to set.

Everything else model-shaped already matches dev (temperature 1.0, top_p 1.0, image size/quality/
format/fidelity, empty `WEB_SEARCH_MODEL`).

### 6b. Delete — dead in v3 (read by nothing)
```
DISCORD_TOKEN / DISCORD_ALLOWED_CHANNEL_IDS / DISCORD_LOG_LEVEL
THREAD_MAX_TOKEN_COUNT
REPORTPRO_SLASH_COMMAND
ELEVENLABS_KEY          # unused by this bot
OPENAI_KEY_PERSONAL     # unused by this bot
```
Also dead if prod ever picked them up from a dev `.env` — all four are now read by nothing:
```
ENABLE_BACKGROUND_IMAGE_GEN   # detached generation is the only path; there is no sync fallback to switch to
ENABLE_VISION_ENHANCEMENT     # the legacy rewrite hop is gone; vision models answer directly
SNOOZE_ACK_EMOJI              # nothing snoozes on a timer any more (F15)
GATE_VISION_DETAIL            # the gate does not look at images at all (see 6a)
```

### 6c. Add — prod-specific, must be set explicitly
```
BOT_NAME_ALIASES = "ChatGPT"                                          # prod bot has NO "-dev"
STATUS_LOADING_MESSAGES_FILE = "status_messages/loading_messages.datassential.txt"
ENABLE_CHANNEL_LISTENING = "true"                                     # all features on in prod
ENABLE_DEEP_RESEARCH = "true"
ENABLE_FEEDBACK_BUTTONS = "false"                                     # the one feature we leave off
SLACK_NATIVE_STREAMING = "true"                                       # validated live in dev since 2026-07-09; kills the "(edited)" markers on streamed replies
ENABLE_EDIT_TRIGGERED_REPLIES = "true"                                # meaningful message edits go through the participation judgment; code default is false
```
> **The single-stream rebuild has merged** (`Docs/SINGLE_STREAM_SPEC.md`). The channel pulse ring
> and its envelope are **retired**, so `ENABLE_CHANNEL_PULSE`, `CHANNEL_PULSE_SIZE`,
> `CHANNEL_PULSE_ENVELOPE_MAX`, `PULSE_TEXT_TRUNCATE` and `PULSE_THREAD_TAILS_MAX` are read by
> nothing — delete them if prod ever had them set. The per-thread **actor tail survives** under
> `PARTICIPATION_THREAD_TAIL` (default 15), which is no longer a prompt input: it is only how the
> bot spots a second agent in a thread. `ENABLE_MENTION_PLACEMENT_MODEL` is likewise gone.
>
> The stream/inventory/history keys it introduced (`COVERAGE_BOOTSTRAP_DAYS`,
> `HISTORY_PAGE_SIZE`/`_CEILING`, `REPLY_FETCH_CONCURRENCY`, `FETCH_RETRY_*`,
> `COVERAGE_SWEEP_CONCURRENCY`, `INDEX_DRAIN_TIMEOUT_SECONDS`, `INGRESS_DRAIN_TIMEOUT_SECONDS`)
> **all have working defaults** — nothing here is required, and there is no longer any boot-time
> rule tying two of them together.
>
> **Background compaction was never deployed and has been removed.** `SNAPSHOT_RETAIN_*`,
> `COMPACTION_TRIGGER_RATIO`/`_TARGET_RATIO` and `ROOT_ANCHOR_TEXT_MAX` are read by nothing —
> delete them if prod ever had them set. Deeper history is reached with `search_slack` and the
> history-fetch tools instead. The first boot after this upgrade drops the compaction tables in
> one recorded migration; nothing dropped is a transcript, and no manual step is needed.
>
> **`DEV_TURN_BARRIERS` must stay EMPTY (unset) in prod.** It is a dev-only test seam — the two
> seams are `post_admission` and `post_partial_post` — and unset it is a hard no-op that touches
> nothing on the turn path. Same for `DEV_TREAT_BOT_IDS_AS_HUMAN`.
The other new keys can be omitted — each has a working default (and every remaining feature
defaults **on**, which is the posture we want), and `.env.example` documents each one inline if you
prefer to pin them explicitly. Ambient memory (F51: quiet notes on links/images/files for later
recall) is among the defaults-on set — its ~24 `AMBIENT_*`/`LINK_FETCH_*` knobs and the new
`DOCUMENT_RETENTION_DAYS`/`EDIT_REPLY_WINDOW_MINUTES` all have working defaults. So do the two new
image-timing knobs (F53) — both govern the same Slack fact, that a file's share record only lands a
few seconds after upload:
- `IMAGE_INDICATOR_HOLD_SECONDS` (default `12.0`) — how long the "Uploading…" indicator waits for
  that share record (which is what makes the image *visible*) before clearing anyway. Measured
  upload→visible is 2.9s for a small image to 5.4s for a real generation, and it scales with image
  size. **Visible** bound: too high shows a stale spinner, too low re-opens the dead-air gap. The
  bot logs the real figure per image, so raise this if you ever see `Image not visible … —
  completing the indicator anyway` in prod logs.
- `IMAGE_SHARE_TS_TIMEOUT_SECONDS` (default `15.0`) — how long the tool-provenance resolver polls
  for that same record so an image message can carry its "made by `generate_image`" note.
  **Invisible** bound (nobody's watching), so it's deliberately longer; on timeout the image just
  carries no provenance. Only read when `ENABLE_TOOL_PROVENANCE` is on.

### 6d. Consider — log levels
Prod runs `DEBUG` across the board. v3 is considerably chattier (channel stream builds, participation
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
On that last line, `canvases:read` / `canvases:write` are now **live** — the canvas tools shipped
(`ENABLE_CANVAS_TOOLS`, default on), and without both scopes they fail quietly. `pins:read` is used
by the history tool. `bookmarks:read` / `bookmarks:write` / `pins:write` are still **staged for the
pinning work in flight** — no code calls them yet. They're included deliberately so this one
reinstall covers that feature too, instead of needing a second reinstall in a week.
Add to event subscriptions:
```
reaction_added · reaction_removed · app_home_opened · app_context_changed
message.channels · message.groups · message.mpim     ← REQUIRED: channel listening is on in prod
file_deleted                                          ← F51 ambient-memory purge on file deletion
```
Also add the `agent_view` block (agent description + suggested prompts) from
`slack_app_manifest.example.yml` — and **REMOVE the legacy `assistant_thread_started` /
`assistant_thread_context_changed` events if present**: Slack's manifest validator rejects them
alongside `agent_view` (2026-07-16; the code handlers remain as a no-op safety net).
**Reinstall the app to the workspace** so the new scopes take effect.

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
Then, only if there is anything to clean up, the participation-redesign steps (per-thread mutes were
removed entirely, so their storage goes with them):

```
DB: Cleared legacy muted_threads JSON on N channel(s)
DB: Removed N stale participation-engine memory fact(s)
```
Both are conditional — silence just means prod had none. The `channel_thread_mutes` table is
dropped in the same pass with no line of its own.

Any `DB: Migration step '<name>' FAILED` line means that step did not complete — the bot will still
start, but stop and investigate. Three migrations are deliberately **fatal** rather than logged —
the compaction-schema drop, the `channel_coverage` inventory rename and the channel-document
uniqueness dedup halt startup rather than serve traffic on a half-migrated schema.

The compaction drop and the rename each run **once** and record a `bot_meta` marker; on a database
that never had the tables, or that has already been renamed, both are no-ops. Expect:

```
DB: renamed channel_coverage coverage_* columns to inventory_*
```

New tables create themselves silently on first boot — no migration line, nothing to do:
`ambient_artifacts`, `thread_summary_addenda`, and the single-stream set (`channel_thread_activity`,
`channel_coverage`, `outbound_receipts`, `pending_share_receipts`).

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
