# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

> Shipping as **v3.0.0** — a major release. The headlines: a new model lineup
> (GPT-5.6 Sol/Terra/Luna), an opt-in **channel teammate** mode,
> **deep research and document builds that run in the background**, a code sandbox that
> computes real charts from real data, and conversation history moved out of the
> database — Slack is now the only transcript.
> **Upgrading from v2? Follow [UPGRADING.md](UPGRADING.md) in order** — it has the `.env`
> deltas, the manifest rebuild, and the migration log lines to watch for. (Existing clones:
> `master` was rewritten during the v3 cycle, so re-clone or
> `git fetch origin && git reset --hard origin/master`.)

### 📦 Upgrade Instructions

Moved to [UPGRADING.md](UPGRADING.md).

### 🚀 Added - GPT-5.6 model family (Sol / Terra / Luna)

- **The model picker offers four models**: GPT-5.6 Sol (flagship, the new default),
  GPT-5.6 Terra (balanced), GPT-5.6 Luna (fast — it also runs the bot's internal utility
  calls, replacing `gpt-5-mini`), and GPT-5.5.
- **New `max` reasoning effort** on the 5.6 family; the effort list in settings adapts to
  the selected model.
- **One-time migration**: all users move to `gpt-5.6-sol` with `medium` reasoning; a startup
  normalizer also clamps any stored model/effort a model no longer accepts, so stale
  settings can never cause API errors.
- **Prompt caching is automatic** on 5.6 models (no cache-retention parameter); GPT-5.5
  keeps its 24-hour retention behavior.
- **Context budgets audited against verified model specs**: all four models use their full
  1,050,000-token window (~920k usable after the reserve for output, tool results, and
  estimator error), and the bot logs once per thread when a conversation crosses the 272k
  long-context billing tier (2× input / 1.5× output past it — informational only).
- **Optional fast mode for Sol** (`OPENAI_SERVICE_TIER=fast`, ships `standard`): up to 2.5×
  faster output at double the token price. Applies only to the user-facing reply — background
  jobs, research, and utility calls never pay for it — and the bot logs whether each reply
  actually got the fast pool or was silently downgraded.

### 🗄️ Changed - Slack is now the only transcript

- **The database no longer mirrors conversations.** Context is rebuilt from Slack history on
  demand; long threads are compacted into rolling summaries (file and image references
  preserved) instead of trimmed silently. What the DB still holds: settings, per-channel
  memory, derived artifacts (image analyses, document summaries), and thread summaries.
- **Full context in every call, never "the last N messages."** Channel answers are built
  from the room as a single time-ordered stream — the most recent ~50 conversations with
  their replies in place (`CHANNEL_WINDOW_TARGET` / `CHANNEL_WINDOW_CEILING`) plus the
  complete thread being answered, however far back it goes. Messages inside the window
  arrive whole, and history tools return oldest-first and say so. Busy channels that used to
  fail outright with "Couldn't Load Conversation History" just work now — a turn fetches the
  newest conversations and stops, whether the channel holds a hundred messages or a million.
- **Token budgeting is usage-driven** (exact counts from API responses; tiktoken remains
  only for the channel-admission estimate, which degrades to a byte ratio without it).
- **People have names in rebuilt conversations** — no more `U01AB2CD3EF:` in transcripts
  after a restart; participants resolve to names on every path (history, search, live).
- **A deleted thread can't wedge a channel**: a thread Slack says no longer exists is
  skipped and its stale record cleaned up (including "tombstone" edits, which Slack sends
  when a root message with replies is deleted). Real fetch errors still stop the turn loudly.

### 🤝 Added - Channel teammate mode (opt-in)

Everything here is inert unless `ENABLE_CHANNEL_LISTENING=true` (code default `false`, so an
upgraded `.env` without the key behaves as before; the shipped `.env.example` sets it `true`).
With listening on, channels default to full participation (`CHANNEL_RESPONSE_MODE=auto_respond`);
any member can set a channel to mentions-only or off via the ⚙️ button.

#### Judgment — when it speaks, reacts, or stays out

- **A lightweight wake-up check watches the channel** and answers exactly one question —
  "should the bot look at this?" The full model then decides *how* to engage with all the
  context in front of it: reply, react, both, or stay silent. Silence is the default.
- **The judgment is staged**: whose message is this, is the exchange still open to me, can I
  actually supply what's being asked — settled in that order, and the chosen action is
  checked against those answers, so it can't reply to a message it concluded belongs to
  someone else. On a replay of 43 real channel situations, unwanted replies dropped from 27
  to 3 with no loss on messages it should answer.
- **Ownership beats helpfulness.** A message aimed at the room's humans or at another
  assistant by name gets nothing, however much it had to offer. An explicit @-mention of
  someone else outranks ground rules and standing instructions to be proactive. A bare
  follow-up after you turn to someone else stays theirs until you name the bot again.
- **It only speaks when it can supply the kind of answer asked for.** Questions asking for
  teammates' firsthand experience or human authority stay with the humans; "I searched and
  found nothing" is not an answer; opinions it has no standing to hold (what a product is
  like to use, whether one tool beats another) are not its to endorse or dispute — in words
  or by reaction. But being addressed by name always gets an honest reply.
- **It reads the room's moments**: it joins a welcome or a send-off the way a teammate would
  (usually one fitting reaction or a short warm line), banters back only when the banter is
  aimed at it, lets you have the last word, and never jokes at a coworker's expense.
- **A correction that lands is finished** — no agreeing with the criticism, no apology tour.
  Told to be quieter, it records that as the channel's standing preference and adjusts.
- **It doesn't borrow answers from adjacent messages.** Two messages count as the same
  conversation only when something actually says so — never because they landed near each
  other. A claim stays exactly as strong as its source ("might be the cache" never becomes
  "the cache was the culprit"), and what it read is treated as evidence, not the whole truth
  ("I don't see it in what I have" instead of "nobody discussed that").
- **No message in a burst is dropped**: rapid-fire messages are judged and answered as one
  conversation, each thread debounced on its own, and a queued @-mention can't be discarded
  by a verdict on surrounding chatter.
- Internal context that sharpens these calls: a wake note saying why it woke
  (`ENABLE_WAKE_ENVELOPE`), the thread's recent back-and-forth
  (`PARTICIPATION_THREAD_TAIL`), thread markers on messages that carry discussions, who's
  in the room and recently active, and a living per-channel gist (what the channel is for,
  who works on what, threads still in the air) built strictly from that channel.

#### Reactions

- **An emoji can be the whole reply** — a 👍 on "cover for me, brb", a 🎉 on a ship — and it
  can react *and* reply in one beat when each carries something the other can't. Reactions
  are held to a lower bar than words but the same ownership rules.
- **It picks emoji that fit the moment** rather than reflexively 👍 — including **your
  workspace's custom emoji**, found by meaning ("a deploy that went badly" can surface your
  `:dumpster-fire:`). Which custom names it knows follows what the workspace actually uses;
  the moment decides the pick. Needs the `emoji:read` scope; without it standard emoji work.
- **Reactions carry meaning**: agreement-by-emoji counts as endorsement and follows the same
  standing rules as words. It sees its own reactions in the record ("(you)"), so "why did
  you react with 🎉?" gets an answer instead of a denial. Up to `REACTION_MAX_PER_MESSAGE`
  (default 4) when asked for several; `REACTION_EMOJIS` restricts the palette if set.
- **👀 means "I'm on this."** The ack reaction lands when genuinely slow work starts (a
  build, a background job, reading your file) and comes off if the work comes to nothing.
  Quick answers get no eye — the answer is the acknowledgment. (`ENABLE_ACK_REACTION`,
  `ACK_REACTION_EMOJI`.)

#### Where the answer lands

- **Placement is part of answering**: thread vs. channel top level is the answering model's
  own call, message by message, with balanced calls defaulting to the thread. Channels where
  top-level replies are switched off are unaffected.
- **It can answer in the thread that's owed the answer** — an open question there, a promise
  made there, an earlier answer of its own that's now wrong — posting directly into that
  thread with at most a brief acknowledgment where it was standing. Same channel only, with
  receipts kept across restarts. (`ENABLE_POST_TO_THREAD_TOOL`.)
- **It can edit its own messages, never silently.** Reserved for a detail that would keep
  misleading people; the default correction is still a new message. Every edit posts an
  announcement in the edited message's thread first — a silent edit is structurally
  impossible — and only its own finalized replies from this turn are editable.
- **A finished answer survives being raced.** If new messages land between composing and
  posting, the bot re-reads and decides: post as written, revise, or drop because the room
  no longer needs it. An overtaken half-composed reply is dropped before it reaches Slack;
  unprompted channel replies post only when complete and still current. Every
  reconsideration and drop is recorded in the participation ledger.

#### Settings, policy, and memory

- **Per-channel participation levels — `on` / `mentions_only` / `off`** — set by anyone via
  the ⚙️ Configure button under bot replies, alongside reply placement and the channel's
  model, effort, and verbosity (hierarchy: personal < channel < per-thread; "each person's
  own setting" keeps the asker's preferences). `off` really means off — no replies even to
  @-mentions in that channel.
- **One standing channel policy** — the channel's ground rules as one authoritative text,
  editable from the modal or by telling the bot directly, and read by *both* the wake-up
  check and the reply, so the two can't diverge. Conditional grants ("jump in when it's
  clearly about the bots") are remembered with the condition intact instead of flipping a
  blanket setting; explicit instructions ("only reply when I tag you") change the setting.
  Any human whose message reached the bot can adjust settings — bots and apps can't, and a
  passing name-drop or quoted text can never flip anything.
- **Per-channel memory, model-managed and yours to edit**: durable facts (decisions,
  conventions, preferences) remembered, recalled, and updated by the bot's own tools, with a
  plain text box in the modal to review or correct them.
- **It knows its own settings**: asked "what's your participation setting here?", it reads
  the actual values instead of guessing from chat history.
- **It introduces itself when added to a channel** — once, publicly: a short hello, then a
  threaded read on what the channel is about composed from the channel's own messages, with
  a Configure button. Fires only for its own join, never in DMs, exactly once per channel.
  Needs the `member_joined_channel` event.
- **Newcomers aren't stopped at the door**: the first @-mention just gets answered; the
  settings invitation arrives quietly in a DM, once ever, clearly optional. First contact in
  a DM keeps the full walk-through.

#### Ambient memory (`ENABLE_AMBIENT_MEMORY`, on by default)

- **Links, images, and files shared in channels get quietly summarized in the background** —
  even when the bot doesn't reply — so a later "what did that chart say?" has real context.
  Only the summary and a pointer to the original are kept; content never persists. Notes age
  out (30-day default), deletions and edits propagate (`file_deleted` event), and a
  per-channel toggle sits in the ⚙️ modal.
- **Links are actually opened** by a hardened fetcher (private/internal addresses refused,
  size- and time-capped, redirects re-checked hop by hop); when a site blocks bots, Slack's
  link preview fills in. A `fetch_url` tool does the same on demand.

#### Boundaries

- **It only reads conversations you and it are both in**, declines the rest with uniform
  wording, and won't repeat a private channel's or DM's content into a shared room even when
  it could — it offers to continue in a DM instead. Search follows the same rule: full power
  in a DM; in a channel, hits from private conversations are left out.
- **`search_slack` is split by surface**: in a DM it searches the workspace through Slack's
  own index (permission-scoped, via the agent surface's `action_token`); in a channel it
  runs a keyword scan of *that channel* — history plus the replies inside its threads —
  which works on every turn, reports how much of the channel it managed to read, and reaches
  exactly what everyone in the room can already see.

#### Operator visibility

- **A participation ledger** (`logs/participation.jsonl`, `ENABLE_PARTICIPATION_TELEMETRY`)
  records every unprompted-message decision — woke or declined, replied/reacted/stayed
  silent, and why (silence carries one of eight explicit reasons) — plus stream builds,
  outbound receipts, reply reconsiderations, and message edits, with a checker
  (`tools/participation_ledger_check.py`) that validates a ledger end-to-end. A provider
  outage during the wake-up check is recorded as a failure, never scored as chosen
  restraint. Coalesced bursts link to the turn that covered them.

### 🔬 Added - Deep research and background jobs

Some questions deserve more than a fifteen-second answer. The bot recognizes those and
detaches the work — in DMs and channels alike.

- **It detaches the job and keeps talking.** Research and document builds run as background
  jobs; the thread stays usable the whole time, and a short "On it — working on this in the
  background." message lands before the status card so the card never just materializes.
- **The status card is a live to-do list**: the plan it wrote when it set off, steps ticking
  from ◦ to ✓, the current one called out, and a running tally (*23 web searches · 2
  MCP calls*). It revises the list as it learns. The card closes with a ✅ and what
  it delivered, or an honest ❌ and the reason — and a terminal card stops animating.
- **It comes back with the thing, not just the findings**: a deck, spreadsheet, or PDF built
  from what it researched, charts computed from real data, delivered without the scratch
  files. Findings arrive under their own byline — "ChatGPT [research: …]" — closing with
  what it used (*deep research · 4m 56s · effort high · tools: web_search*).
- **A running job can take mid-run updates** ("drop that section") via
  `update_background_job` — they appear on the card, prompt a todo revision, and updates
  that arrive too late are surfaced in the delivery reply as not-applied rather than
  silently claimed.
- **Work can be cancelled mid-run** via `cancel_background_job` — a job or an in-flight
  image generation, by id, with the reason on the card. A job already posting its results
  refuses the cancel; cancelling mid-build releases its container.
- **Corrections edit, not rewrite**: a job revising a file it already produced starts from
  that file's current content (fetched from Slack, extracted in memory, never stored) and
  makes point edits instead of rebuilding from scratch. Jobs are held to their sources — a
  claim the material doesn't establish is left out or marked unverified.
- **It knows what it has running**, so a passing remark can't spawn a duplicate build; a
  second job takes an actual ask, and two jobs can never write the same filename. Build-only
  jobs pass their worker's notes to the reply, so it can say what the build showed.
- **Nothing fails silently**: API errors, timeouts, empty results, and failed posts each
  surface as one honest line. Two jobs per thread at once.

Flags: `ENABLE_DEEP_RESEARCH` (default **on**), `DEEP_RESEARCH_REASONING_EFFORT` (high),
`DEEP_RESEARCH_TIMEOUT` (600s), `DEEP_RESEARCH_MAX_PER_THREAD` (2), `ENABLE_RESEARCH_LABEL`
(on — the byline needs `chat:write.customize`; without it the bot posts plainly).

### 📊 Added - It can write and run code, and hand you the file

- **Charts are computed, never drawn.** "Chart this" goes to a Python sandbox working on
  your actual data — previously it could be routed to the image model, which drew a
  plausible chart with invented numbers. Anything the code writes (`.png`, `.xlsx`,
  `.docx`, `.pptx`, `.csv`, `.pdf`) uploads back into the thread; executables, archives,
  and macro-enabled Office files are never handed back.
- **The scratch space survives the turn**: each thread gets its own sandbox, so "now add a
  trendline" reuses what was computed. It goes cold after ~20 minutes idle (an API limit);
  a revived thread quietly gets a fresh one.
- **Long builds don't freeze the reply**: work that takes minutes goes to a background job
  with a live status card; quick sandbox work stays inline. Running code is not an outside
  source, so it doesn't appear in the "Tools Used" footer.

Flags: `ENABLE_CODE_INTERPRETER` (on), `ARTIFACT_MAX_FILES` (4), `ARTIFACT_MAX_MB` (25),
`ARTIFACT_ALLOWED_EXTENSIONS`, `CODE_INTERPRETER_CONTAINER_TTL_MINUTES` (20 — the API
maximum), `CODE_INTERPRETER_CONTAINER_REUSE_MINUTES` (15).

### 🎨 Changed - Images and code are the same conversation now

- **The pre-flight image/text router is gone.** Generating, editing, and running code are
  tools the model picks while thinking — so one turn can generate a cover image, compute a
  chart from your numbers, and assemble both into a `.pptx`. (`ENABLE_IMAGE_TOOLS`, on;
  off restores the old classifier.)
- **Image generation runs in the background**: the image posts itself when ready and the
  thread keeps moving; several can cook at once (`MAX_CONCURRENT_IMAGE_GENERATIONS`,
  default 5 per thread; `API_TIMEOUT_IMAGE`, 300s). Edits wait their turn, and an
  acknowledgment mid-generation ("thanks!") is no longer misread as another image request.
- **It edits the image you meant** — picked by name from the thread's actual images, asking
  when genuinely ambiguous — and **it looks at what it produced**, so it can flag a result
  that drifted off the brief instead of presenting it as a success. It can also re-open an
  older image when a question turns on fine detail, and in DMs it finds images from earlier
  top-level messages.
- **Images are read at full resolution** — screenshots of tables, logs, and serial numbers
  stop coming back subtly wrong. Formats the API won't take (BMP, TIFF, ICO, anything
  Pillow decodes) convert to PNG in memory; truly corrupt files get an honest note.
- **Your image settings reach the API on every path** (quality on edits, format and
  compression on generation), the "Enhanced Prompt" wall of text is gone, and image jobs
  show a live ticking checklist (`ENABLE_PROGRESS_CHECKLIST`) with an "Uploading…" stage
  that waits for the image to actually land in Slack instead of guessing.

### 📄 Changed - Smarter, lighter document handling

- **Uploads become a concise summary in the conversation** (spreadsheets show
  sheets/columns/sample rows); when you ask for specifics the bot re-reads the original from
  Slack on demand. PDFs are read natively by the model (`ENABLE_NATIVE_FILE_INPUT`, on) —
  tables, charts, and scanned pages are actually visible to it.
- **Content is never stored and never touches disk** — the DB keeps only a summary and a
  Slack reference; files process in memory. Deleting a file from Slack removes it from the
  bot's reach entirely. Old summaries slim down after `DOCUMENT_RETENTION_DAYS` (90) instead
  of vanishing.
- **Scanned PDFs stay readable after the first turn**: image-only pages are OCR'd on demand
  (`ENABLE_PDF_OCR`, on; `OCR_MAX_PAGES`, 20 — past the cap it says so loudly). Needs the
  `tesseract-ocr` + `poppler-utils` system packages; without them it degrades to an honest
  "text not extractable" note.
- **~100 file extensions extract inline** — code and config files, `.rtf`, `.eml`, Jupyter
  notebooks, tab/pipe-separated data, and more. Secrets are deliberately refused: `.env`,
  `.pem`, `.key` and friends are never ingested, even mislabeled.
- **Slack canvases are readable documents** — converted to markdown on arrival, via
  `read_document`, in cold rebuilds, and in ambient memory (previously every canvas read as
  "deleted/unavailable" because Slack serves them as HTML, which looked like a login page).
- **Files are readable across the channel**, not just the thread they landed in — and
  findable, because filenames ride the channel history the bot sees. Same channel only;
  DMs stay private to the DM.
- **Slack-native tables, forwarded posts, link unfurl cards, and webhook attachments**
  render into context on every path — previously all silently dropped.
- **Failed attachments are news for the reply, not a warning card**: the reply itself says
  which file failed, why, and what to do. The static card remains only where no reply can
  carry the news. A message permalink pasted in chat is recognized as a conversation
  reference, not downloaded as a file.

### 🧾 Added - Canvases, for work that outlives the thread

- The bot can create the channel canvas (the one that gets a pinned tab), read, edit, and
  list canvases — including yours — amending in place instead of posting another copy. It
  names the canvas at creation, because Slack has no rename. (`ENABLE_CANVAS_TOOLS`, on;
  needs `canvases:read` / `canvases:write`.)

### 🧠 Added - Time, provenance, and tool memory

- **Every message it reads is stamped with when it was said**, in the sender's timezone —
  "last night" and "you asked an hour ago" mean something, and it can tell a stale thread
  from a live one (`ENABLE_MESSAGE_TIMESTAMPS`).
- **It remembers which tools it used** — each reply (and each image it posts) records the
  tools it ran, reinjected as a compact note when the thread is reread, so "did you actually
  look that up?" gets a real answer. Only tool names and neutral hints are kept, never
  results or your content (`ENABLE_TOOL_PROVENANCE`).
- **It stops retracting what it looked up**: results from data servers are remembered
  alongside the reply, and it's forbidden from taking back a fact it already gave because a
  fresh search missed. Overlong tool results are summarized preserving every URL, title,
  date, figure, and ID verbatim — not chopped at a character count.

### ⚡ Changed - Replies start seconds sooner, everywhere

Three dead waits were cut from every turn: the code sandbox is created lazily instead of up
front (the 1–15s pause in front of DM replies is gone), channel replies carry their own
placement instead of spending a model round on it, and the wake-up gate no longer sleeps a
flat 3 seconds on every ambient message. Measured on the dev bot: DMs 4–5s (was 4–18s),
mentions 7–10s (was 8–20s), ambient replies ~12s (was 15–22s).

### 💬 Changed - Slack surfaces, streaming, and delivery

- **Migrated to Slack's current agent view** (June 2026): greeting and suggested prompts on
  the `app_home_opened` surface; the assistant split-view and the per-message `action_token`
  behind DM workspace search come with it (see README — easy to lose on upgrade).
- **Native streaming** (`chat.startStream`/`appendStream`/`stopStream`) fully wired behind
  `SLACK_NATIVE_STREAMING` (code default `false`; the shipped `.env.example` turns it on) —
  classic edit-loop streaming remains the automatic fallback either way. Streamed replies
  credit their tools the same as non-streamed ones.
- **One clean "working" indicator**: Slack's native status bubble, rotating through a
  100-message pool (brand it with `STATUS_LOADING_MESSAGES_FILE`; per-stage texts in
  `status_messages/pipeline_messages.txt`; a missing or broken file can never break the
  bot). Nothing shows until the bot has committed to replying — no "Thinking…" flash on
  messages it declines, and the public context-usage banner is gone.
- **Replies come out once, and clean**: no "(edited)" stamps on channel-level replies, no
  double answers after a retry (it continues the reply you're already reading), no glued
  sentences around tool calls, and a staged reply keeps its opening words with streaming
  off. Long threaded replies no longer collapse behind "Show more" (the ⚙️ footer rides
  short replies; long ones post as plain text with the footer separate).
- **No busy rejections anywhere**: messages arriving mid-response are queued and answered
  together — DMs, threads, and channels. The "I'm busy, try again" behavior is retired.
- **Sturdier delivery**: links stay inline (`ENABLE_LINK_PREVIEWS=false` default), split
  replies retry a failed middle part and say so loudly if it stays lost, the same answer
  can't post twice after a transport hiccup, and a reply that never sent isn't remembered
  as said.
- **Feedback**: 👍/👎 under DM/assistant responses (`ENABLE_FEEDBACK_BUTTONS`, off by default; the strip
  appears once per thread), and thumbs reactions on any bot message count as the same
  signal. Feedback lands in a local table; nothing leaves your workspace.
- **A socket-liveness watchdog** logs a clear "socket presumed dead — restart likely
  required" if the Slack connection ever goes half-open (`SOCKET_LIVENESS_TIMEOUT`, 600s;
  0 disables), and a fatal startup error exits non-zero so supervisors restart it.

### 🩹 Fixed - Error messages that respect the reader

- The `Error Code / Type / Details` scaffold is retired; every user-facing error is one
  friendly line with a next step, and technical details stay in the logs. Nothing fails
  silently: an undownloadable file says so, a failed catch-up asks you to re-send, a broken
  Configure button tells you.

### 🔌 Changed - MCP hardening

- **Secrets out of `mcp_config.json`**: `${VAR_NAME}` placeholders in `headers` expand from
  `.env` at load; a server with unresolved variables is skipped with a warning naming them.
  Per-server `"enabled": false` parks a server without deleting it.
- **Startup health probe**: one reachable/unreachable line per server plus its discovered
  tools.
- **Failover survives multiple failing servers** (exclusions accumulate across retries),
  failures are detected from structured error codes first, and a config requesting
  `require_approval` other than "never" logs a clear warning instead of being silently
  ignored. The README's old `authorization` config shape never worked — corrected to
  `headers`.

### ✨ Changed - Prompts modernized for current models

- Brief and conversational at channel top level, fuller detail in threads; the
  always-use-section-headers rule is gone. Image edits state exactly what changes and
  preserve everything else; generation prompts keep your explicit specifications verbatim.
  Multi-user threads no longer lose prompt-cache hits on every speaker change.

### 🛡️ Fixed - Pre-release hardening (full adversarial review)

A ground-up review of the whole codebase before v3 goes live — all fixes to unreleased v3
behavior, condensed: the channel-settings modal can't be bricked by a stored value or an
overlong description; bold/links/parentheses format correctly on every path; background
jobs can't silently drop a declared deliverable and slow multi-file builds keep what they
staged; thread history survives compaction with its images and documents; image-URL
fetching validates against internal addresses and caps downloads (SSRF/memory-exhaustion);
streaming handles the `incomplete`/`failed` terminal states; concurrent tool calls can't
exceed image caps or double-create a channel canvas; nightly backup/cleanup doesn't block
the event loop; and `beautifulsoup4` — imported but never declared — is in the lockfile.

Upgrade safety got the same pass: the v3 migrations back up **before** any write, a failed
migration step fails loudly by name instead of silently skipping the rest, nightly backups
actually run (the docs had promised them for years), retention never deletes the tagged
migration backups, and `.gitignore` now covers every `.env*` variant (with `.env.example`
explicitly re-included) so nothing env-shaped can slip into a commit.

### 🧹 Removed

- **Discord support** — the V2 Discord bot was never built (the launcher was a "Coming
  Soon" stub). The bot is Slack-only.
- **All pre-5.5 model support**: GPT-4 series, `gpt-5`, `gpt-5-nano`, `gpt-5-chat-*`,
  `gpt-5.1`–`5.4`, and `gpt-5-mini`, plus their dead API branches and one-off migration
  scripts.
- **`legacy/` (V1 bots)** and **`extract_metrics.py`** (it read the dropped `messages`
  table) — both still in git history.
- **`python-magic`** — nothing imported it, and it pulled in a `libmagic` system
  requirement for no reason.
- **Dead `.env` entries** (`DALLE_MODEL`, `DEBUG_MODE`, `MAX_CONCURRENT_THREADS`,
  `MESSAGE_TIMEOUT`) and stale settings-modal remnants from the GPT-4/5.x era;
  `.env.example` is reordered by audience — credentials up top, tuning at the bottom.

### 🧪 Changed - Test suite restored

- The unit suite is fully green again (1,185 tests, 0 failures) after years of rot;
  `make test` runs the entire suite instead of stopping at the first failure. Stale tests
  of removed behavior were deleted; tests of real behavior were repaired.

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