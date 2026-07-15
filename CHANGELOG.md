# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

> Shipping as **v3.0.0** — a major release. The headlines: a new model lineup
> (GPT-5.6 Sol/Terra/Luna), the bot can now act like a real channel teammate
> (off by default), it can run **deep research jobs in the background**, and conversation
> history now lives in Slack, not the database.
> Follow the Upgrade Instructions below in order — steps 2 and 5 are the ones that bite.

### 📦 Upgrade Instructions (start here)

**0. Back up your database first.** v3's first boot runs irreversible migrations. The bot
now takes its own `pre-v3-upgrade` backup before touching anything, but a copy of
`data/slack.db` you made yourself is the one you can trust.

**1. Update dependencies**
```bash
make install   # pip install --require-hashes -r requirements.txt (openai >= 2.45.0)
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
MAX_UNPROMPTED_REPLIES_PER_HOUR   # the hourly cap is gone — pacing is the model's judgment
PARTICIPATION_SNOOZE_HOURS        # "butt out" now mutes the thread durably, no timer
```

New keys worth a decision (see `.env.example` for the full annotated list — every knob is
documented inline there):
- `ENABLE_CHANNEL_LISTENING=false` — the master switch for teammate behavior in channels.
  **Off by default: the bot behaves exactly as before (mentions + DMs) until you flip it.**
- `BOT_NAME_ALIASES=ChatGPT` — names the bot answers to without an `@`. **Set this per
  environment** (e.g. `ChatGPT-Dev` on a dev install), or the dev bot will answer to the
  prod bot's name.
- `ENABLE_DEEP_RESEARCH=true` — **on by default, and it works in DMs too.** Each job is
  minutes of model time at `high` effort; turn it off if that's not a bill you want.
- `SLACK_NATIVE_STREAMING=false` — native streaming is built and tested but ships off;
  validate live in your workspace before enabling
- `ENABLE_LINK_PREVIEWS=false` — links in the bot's posts stay inline; set true for Slack's
  preview cards (this is a change from v2 behavior, where Slack unfurled them)
- `STATUS_LOADING_MESSAGES_FILE` — optional branded "working…" messages for the
  thread status indicator: point it at your own text file, one message per line,
  plain text (no emoji — the status surface doesn't render them). Unset = a bundled
  set of 100 generic ones (`status_messages/loading_messages.generic.txt`).
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
`slack_app_manifest.example.yml` over your environment copy (keep your names/commands) and
reinstall the app — the new scopes need re-consent, and a missing one degrades a feature
silently. New since v2.5: the `agent_view` block; bot scopes
`search:read.public/private/im/mpim/files/users`, `reactions:read`, `reactions:write`,
`pins:read`, `users:read.email`, `chat:write.customize`, `assistant:write`, `emoji:read`; and
events `reaction_added`, `reaction_removed`, `app_home_opened`, `app_context_changed` (the
legacy `assistant_thread_*` events stay subscribed during the transition).

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

From then on the database backs itself up nightly (7-day retention) as part of the scheduled
cleanup, which it never did before. The three `pre-v3-*` backups are **exempt from that
retention** — they're your rollback path, so nothing deletes them but you.

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
  off. Pacing is the model's own judgment (it sees how often it has spoken up
  recently), backed by rapid-fire debouncing and "ignore" as the default verdict.
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
  permanently mutes unprompted participation *in that thread* — no timer to wait out,
  and mentions and name-summons still answer. It also writes a dated note to channel
  memory, so the bar goes up channel-wide. Standing feedback ("stay out of here unless
  tagged", "keep answers short in this channel") is remembered durably as a channel
  preference.
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
- **Emoji reactions** as a response type, both engine-chosen and model-invoked. The bot
  picks from the full standard emoji set by default; set `REACTION_EMOJIS` to a list if
  you want to constrain it to brand-approved reactions.

- **It knows what it can actually do**: an open question to the room ("anyone know what
  our data says about X?") gets an answer when the bot's own tools or MCP servers can
  answer it. Previously it stayed quiet because the part of it that decides whether to
  speak had no idea what the rest of it was capable of. Nothing is hardcoded — the list
  follows whatever servers and tools you've configured.
- **Files dropped in a channel reach it**: a photo, PDF, or spreadsheet posted with a
  question ("what do we think — good enough to send?") now gets a reply. Slack delivers
  uploads as a special kind of message that the bot was discarding before it ever got to
  the "should I answer this?" decision, so *every* channel file question had been invisible.
- **It knows who's in the room**: the bot sees roughly who's around and recently active,
  which sharpens its read of who "you" refers to. It can also look someone up by name or
  @mention (title, timezone, whether they're a bot) and list the channel's members when
  that's what you're asking about.
- **A fast follow-up gets one answer, not two**: post a question and then a "second
  thought — also…" a moment later and the bot answers both in a single reply. Two people
  asking different things at the same moment still get their own answers (previously the
  first one could be silently dropped).
- **It can hold a real conversation**: the bot can now take part in a genuine multi-person
  back-and-forth without needing to be re-named in every message, and it banters back
  briefly when the room (or a jab aimed at it) invites it.

#### Changed
- **No more "busy" rejections anywhere**: messages arriving while the bot is working
  are queued and answered together in one coherent catch-up reply (DMs, threads, and
  channels). The old "I'm busy, try again" behavior is retired.
- **Replies lean toward threads** in channels — long answers, likely back-and-forths, and
  busy rooms go in a thread; a short answer the whole channel needs can still land at
  channel level.

#### Fixed
- **"Off" now really means off.** Setting a channel's participation level to *off* still
  answered @-mentions there — the modal promised "never respond in this channel" and the
  bot replied anyway. Off is now fully silent in that channel (DMs are unaffected), and
  genuinely different from "mentions only".
- **A message addressed to someone else is never hijacked.** "@Claude, I heard you can…"
  in a channel could be answered *by this bot*, cheerfully explaining another assistant's
  internals as if they were its own. Unresolved @-mentions were being stripped out of the
  text entirely, destroying the very signal that said who was being spoken to. Mentions are
  now preserved, and an explicit @-mention of someone else is the strongest possible signal
  that a message isn't for the bot — it outranks channel ground rules and standing
  instructions to be proactive.
- **Questions in other threads stop vanishing.** Rapid-fire chatter in one thread could
  silently cancel the pending evaluation of an unrelated message elsewhere in the channel,
  so it was never judged and never answered. Each conversation is now debounced on its own.

### 🔬 Feature - Deep research, in the background

Some questions deserve more than a fifteen-second answer and one web search. The bot can now
recognize those and go do the work properly — in DMs and channels alike, in the default config.

- **It detaches the job and keeps talking.** Ask something that genuinely needs digging
  ("what happened to egg prices this year, and what's the H2 outlook?") and the bot spins the
  research off into a background job, then posts a sourced report back into the thread minutes
  later. The conversation stays usable the whole time — you can ask other things, and it answers.
- **It comes back with the thing, not just the findings.** A job can build what it researched
  into a real deliverable — a deck, a spreadsheet, a PDF — with any charts computed from the
  actual data rather than drawn. It decides what's worth handing over, so you get the report, or
  the file, or both, and never the pile of scratch files it made along the way.
- **You can watch it work, and it's a real to-do list.** A single live card sits in the thread
  while the job runs: the plan it wrote when it set off, each step ticking from ◦ to ✓, and the
  one it's working on right now called out. It revises the list as it learns — the plan at minute
  one often isn't the plan it finishes with — alongside a running count of what it's been doing:
  *todos as of 7:36 PM · 23 web searches · 2 datassential calls*. The card closes with a ✅ and
  what it delivered, or an honest ❌ and the reason.
- **The findings arrive under their own byline** — "ChatGPT [research: 2026 US egg outlook]" —
  so a long report is clearly the research job talking, not the bot interrupting the chat. It
  closes with what it used: *deep research · 4m 56s · effort high · tools: web_search*.
- **Nothing fails silently.** An API error, a timeout, an empty result, or a failed post each
  surface as one honest line in the thread. Two jobs per thread run at once; ask for a third and
  it says so.

Flags: `ENABLE_DEEP_RESEARCH` (default **on**), `DEEP_RESEARCH_REASONING_EFFORT` (high),
`DEEP_RESEARCH_TIMEOUT` (600s), `DEEP_RESEARCH_MAX_PER_THREAD` (2), and `ENABLE_RESEARCH_LABEL`
(on — the byline needs the `chat:write.customize` scope; without it the bot posts plainly rather
than failing).

### 🖼️ Fixed - It looks at your image before reacting to it

Drop a picture in a channel with no caption and the bot used to react to it *blind* — the quick
"should I chime in here?" check only ever saw the filename, never the image. So a meme or a
screenshot got an emoji chosen from thin air, sometimes plainly the wrong one. That check now
actually sees the picture, so when it reacts — or decides the image is worth a real answer — it's
responding to what's in it, not guessing from the words around it.

And when it *does* study an uploaded image, that finally works at all: image analysis had been
silently failing on every upload, so the bot quietly lost track of what a picture showed later in
the conversation. It reads them correctly now.

Kept deliberately cheap: a couple of images at most, small, at low resolution, on a short
deadline — and if it can't see one, it's told so plainly instead of inventing what's in it.
Flag: `ENABLE_MULTIMODAL_GATE` (default **on**).

### ✂️ Fixed - Replies come out once, and clean

- **No more "(edited)" on channel replies.** When the bot answered at the top of a channel
  rather than in a thread, it posted a stub and rewrote it as the words arrived — and Slack
  stamped the result "(edited)" every time, as if it had gone back and changed its answer. Those
  replies now appear once, whole, with no marker. (Inside a thread it still types the answer out
  live, which Slack never marks edited.)
- **No more double answers.** If a tool the bot was using hiccuped and it had to retry, it could
  post the entire answer a second time and leave the half-finished first copy sitting above it.
  A retry now continues the reply you're already reading instead of starting a new one.

### ⏳ Fixed - You can watch it work while it edits an image

Ask the bot to edit an image and it would say something like "On it, fixing that now —" and then
freeze mid-sentence for the whole minute the edit took, snapping to the full reply only once the
new picture was ready. Now whatever it says *before* it starts editing reaches you right away, so
you're not staring at half a sentence wondering if it hung. And when it picks the reply back up
afterward, the two halves no longer collide into one jammed-together word.

### 🔁 Fixed - It no longer builds the same thing twice

Ask for a deck, say something in the thread while it's working, and the bot could quietly
start building the deck *again* — two status cards, two files, one request.

The cause was a blind spot: the bot could see when it was already generating an *image*, but
not when it was already running a *background job*. So a passing remark in the thread ("never
tried this, not sure how it'll turn out") was enough to wake it, and with no idea a build was
already under way, it started a second one. It now knows what it has running, and says so to
itself before deciding anything. A second job needs you to actually ask for separate work, and
two jobs can never write the same filename.

### 🤫 Fixed - It stops thinking out loud

Three things the bot used to say that it had no business saying.

- **No more "Thinking…" flash on messages it doesn't answer.** Listening in a channel, the bot
  decides twice whether to speak: a quick judgment call, and then the real one once it has
  actually read the room. The "Thinking…" indicator went up between the two — so a message it
  ultimately had nothing to add to still got a spinner that appeared and vanished. Now nothing
  shows until it has committed to replying. If it decides to stay out of it, you never see it
  consider the question at all.
- **👀 means "I'm on this", and it means it.** The eye used to be a *guess*, dropped before the
  bot had done anything, on a hunch that real work was coming — so it landed on passing
  comments and then nothing followed. It's now a claim on work: it appears when the bot
  genuinely starts doing something slow (a search, a build, reading your file), and if that
  work comes to nothing, **it takes the eye back off**. A quick answer gets no eye at all — the
  answer is the acknowledgment.
- **The context-usage box is gone.** It printed a public banner of token counts and "tips for
  optimal performance" into the thread, where the whole channel could see it, about
  housekeeping you never asked about and can't act on. Conversation compaction is the bot's
  own business and it now keeps it to itself.

### 🧾 Feature - Canvases, for work that outlives the thread

Some things shouldn't be a chat message. A running spec, a launch checklist, a summary that keeps
getting amended — in a thread it's buried within the hour, and as a posted file it forks into
`_final_v3` by Thursday. The bot can now put that kind of work in a Slack canvas and go back and
edit it in place.

- **It makes the canvas that gets a tab** at the top of the channel, so there's somewhere to find
  it later rather than a link you have to dig for.
- **It can read, edit and list canvases** — including ones you made. Ask it to add a section to
  the spec and it amends the canvas instead of posting another copy of it.
- **It names the canvas when it creates it**, because Slack has no rename — an untitled canvas
  stays untitled forever.

Flag: `ENABLE_CANVAS_TOOLS` (default **on**; needs the `canvases:read` / `canvases:write` scopes).

### 🎨 Changed - Images and code are the same conversation now

The bot used to decide, before it had really read your message, whether the turn was "an image
request" or "a chat request" — one guess, no take-backs. That guess is gone. Making an image,
editing one, and running code are now just things it can *do*, chosen while it's thinking, the
same way it decides to search the web.

- **It can make a picture and compute a chart in the same breath.** Ask for a deck with a cover
  image and a chart of your real numbers, and you get a single `.pptx` with both in it — the
  image generated, the chart computed from your actual data, the whole thing assembled and
  handed back. Before, a turn could do one or the other, never both.
- **"Chart this" stopped being an image request.** The old router treated "visualize" as a cue
  to *draw*, so it would hand your spreadsheet to an image model, which produced a
  handsome-looking chart with invented numbers and invented category names. Charts are computed
  now, always.
- **It edits the image you meant.** It picks from the actual images in the thread by name rather
  than guessing "probably the last one" — and if your request is genuinely ambiguous it asks
  instead of quietly editing the wrong picture.
- **It respects your image settings, and knows when not to.** Your saved model, size, quality
  and background still apply by default. The model can now deviate when the job calls for it
  (a wide image for a title slide) — except the image *model* itself, which is yours and is not
  up for negotiation.
- **The "Enhanced Prompt" wall of text is gone.** It still rewrites your prompt into something
  the image model can work with — that just isn't your business any more, the same way the code
  it runs isn't. You get an image, not a lecture about how it got there.

Flag: `ENABLE_IMAGE_TOOLS` (default **on**). Off restores the old classifier and its routing.

### 📊 Feature - It can write and run code, and hand you the file

Ask for a chart, a cleaned-up spreadsheet, a summary of the numbers in a CSV you dropped in the
thread — the bot now writes Python, runs it in a sandbox, and uploads whatever it produced back
into the thread as a real file.

- **The numbers are real.** Charts are computed from your actual data, not drawn. Previously
  "chart this" could be mistaken for an image request, and the image model would draw a
  plausible-looking chart with invented numbers and invented labels. That is fixed: charting data
  goes to the sandbox, always.
- **The file comes back.** Anything the code writes — `.png`, `.xlsx`, `.docx`, `.pptx`, `.csv`,
  `.pdf` — is uploaded into the thread. The sandbox ships with pandas, matplotlib, openpyxl,
  python-docx, python-pptx, LibreOffice, ffmpeg and more. Executables, archives and
  macro-enabled Office files are never handed back.
- **The scratch space survives the turn.** Each thread (channel or DM) gets its own sandbox, so a
  follow-up like "now add a trendline" reuses what was already computed instead of starting over.
  It goes cold after ~20 minutes idle — an API limit — and a revived thread quietly gets a fresh
  one.
- **Internal steps stay internal.** The "Tools Used" footer is there to tell you where outside
  facts came from (web search, Datassential). Running code isn't an outside source, so it no
  longer shows up there.

Flags: `ENABLE_CODE_INTERPRETER` (default **on**), `ARTIFACT_MAX_FILES` (4), `ARTIFACT_MAX_MB`
(25), `ARTIFACT_ALLOWED_EXTENSIONS`, `CODE_INTERPRETER_CONTAINER_TTL_MINUTES` (20 — the API
maximum), `CODE_INTERPRETER_CONTAINER_REUSE_MINUTES` (15).

### 💬 Feature - Reactions that read like a colleague's

- **"I'm on it."** When a request implies real work — files to read, data to look up, several
  steps to run — the bot drops a 👀 on your message immediately and then goes and does it, so
  you're not left wondering whether it heard you. No timers and no extra model calls; it's a
  judgment the bot already makes. Set the emoji with `ACK_REACTION_EMOJI` (default `eyes`) or
  turn it off with `ENABLE_ACK_REACTION`.
- **Sometimes the reaction *is* the reply.** "Please cover for me while I'm out, brb" now gets a
  single 👍 instead of a paragraph. When one emoji fully carries the answer — an acknowledgment,
  an FYI, agreement that needs no elaboration — the bot prefers it to writing, and it won't
  restate in words what someone else already said with a reaction.
- **It reacts like a person would**: joining a laugh, thumbs-upping good news or a fix it helped
  with. Others having already reacted makes it *more* likely to join in, not less. Most messages
  still get nothing, and it stays away from anything heated or personal.
- **The emoji palette is now open by default.** The bot picks whatever emoji actually fits, from
  the full standard set. `REACTION_EMOJIS` is still there if you want to hold it to a
  brand-approved list — set it and the restriction is enforced everywhere.
- **Ask for several reactions and you get several.** The bot was hard-limited to one emoji per
  message on several layers, so a request for three would get one — and it would sometimes follow
  up by claiming it was "showing restraint". Up to `REACTION_MAX_PER_MESSAGE` (default 4) now.
- **It remembers the reactions it placed**, so asking "why did you react with 🎉?" no longer gets
  a confused denial or an answer about someone else's reaction.

### 🕰️ Feature - The bot can reason about time and remember what it found

- **Every message it reads is stamped with when it was said**, in the sender's own timezone. So
  "last night", "before the meeting", and "you asked me this an hour ago" now mean something, and
  it can tell a stale thread from a live one. Toggle with `ENABLE_MESSAGE_TIMESTAMPS`.
- **It stops losing — and retracting — what it looked up.** It would cite a real report with a
  link, then on "can you send me that link?" find the link gone from its memory, re-run the
  lookup, miss, and *retract its own correct answer*. Results from your data servers are now
  remembered alongside the reply, and it's explicitly forbidden from taking back a fact it
  already gave you just because a fresh search didn't turn it up again.
- **Long results get summarized, not guillotined.** An overlong tool result used to be chopped at
  a character count, which could amputate the very link or figure that made it worth keeping. It's
  now summarized once — preserving every URL, title, date, figure, and ID verbatim — and falls
  back to plain truncation if anything goes wrong.

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
- **Files are readable across the channel, not just in the thread they landed in**: a CSV
  shared in one thread can be read from another thread in the same channel, and the bot says
  where it came from ("shared in another conversation in this channel"). Same channel only —
  never across channels, and DMs stay private to the DM.
- **…and it can actually find them.** Filenames now show up in what the bot sees of channel
  activity and history, so "the vendor contract PDF from the review thread" is enough to go on.
  A search inside a document that matches nothing no longer dead-ends — it comes back with the
  content and a way to navigate it, so one look always yields something answerable.
- **Scanned PDFs stay readable after the first turn.** An image-only or scanned PDF was legible
  on the turn you uploaded it and effectively lost afterward. Its pages are now OCR'd on demand,
  so "what's the vendor code in that contract?" still works days later. Requires the
  `tesseract-ocr` and `poppler-utils` system packages (see the Upgrade Instructions); without
  them it degrades to an honest "scanned document, text not extractable" note rather than
  failing. Gated by `ENABLE_PDF_OCR` (default on) and bounded by `OCR_MAX_PAGES` (20) — past
  the cap it says loudly that it truncated, and never pretends otherwise.

### 🧵 Feature - The bot can answer in a different thread

- "Go back and answer that question in the other thread" now works: the bot acknowledges briefly
  where you are, and posts the real answer where it belongs. Same channel only, never
  cross-channel, and it refuses to post into a thread someone has told it to stay out of.
  Toggle with `ENABLE_POST_TO_THREAD_TOOL` (default on).

### 🔗 Changed - Quieter, sturdier message delivery

- **Links no longer explode into preview cards.** The bot's posts keep links inline, which also
  stops Slack's link unfurler from stamping an "(edited)" badge on them. Set
  `ENABLE_LINK_PREVIEWS=true` to get the preview cards back.
- **No more "Continued in next message…" trailers** on long split replies — the next message
  already says "…continued", so the seam was being announced twice.
- **A long reply can't silently lose its middle.** If part of a split message fails to post, the
  bot retries it once and, if that fails too, says so loudly ("⚠️ This message was cut off…")
  instead of leaving a hole you'd never notice.

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
- **`extract_metrics.py` deleted** — the usage-report script read the `messages` table, which
  no longer exists, so it could not run against a v3 database. Still in git history if it's
  worth rewriting against the new schema.
- **`python-magic` dropped from the dependencies** — nothing imported it, and it pulled in a
  `libmagic` system requirement for no reason.

### 🧪 Changed - Test suite restored

- The unit suite is fully green again (1,185 tests, 0 failures) after years of rot;
  `make test` now runs the entire suite instead of stopping at the first failure.
  Stale tests of removed behavior were deleted; tests of real behavior were repaired.

### ✅ Feature - Live progress checklists on image tasks

- Image generation and editing now show an accumulating checklist that ticks off each
  step in place ("✓ Enhanced prompt → ✓ Generated image → Uploading…") instead of a
  single status line that overwrites itself. Where Slack has a native status surface the
  checklist lives there and nowhere else — Slack already shows it both in-thread and under
  the composer, so posting a third copy as a message was noise. Surfaces without a native
  status still get a real checklist message. Toggle the whole thing with
  `ENABLE_PROGRESS_CHECKLIST` (default on); set `PROGRESS_CHECKLIST_PREFER_MESSAGE=true` if
  you want the extra visible thread message on top.

### 🖼️ Feature - Image generation no longer freezes the conversation

- Creating a new image used to hold the thread while the model worked, so anything you
  said in the meantime had to wait. Image generation now runs in the background: the
  image posts automatically when it's ready and you can keep chatting the whole time.
  Ask for a second image while the first is still cooking and it simply runs too — up to
  `MAX_CONCURRENT_IMAGE_GENERATIONS` (default 5) per thread. Edits still wait their turn
  (you can't edit an image that doesn't exist yet). Toggle with `ENABLE_BACKGROUND_IMAGE_GEN`
  (default on); image jobs get their own longer time budget via `API_TIMEOUT_IMAGE`
  (default 300s).
- An acknowledgment while an image is generating ("ok", "thanks", "nice") is no longer
  misread as a request for another picture — a follow-up only counts as an image request
  when it actually adds or changes something visual.
- Fixed: the "✨ Enhanced Prompt" preview had stopped appearing on most surfaces (it was
  tied to a status message that newer Slack surfaces don't create) — it now posts as its
  own message so you can always see the prompt the image was built from.

### 🤐 Feature - The bot can now choose to stay quiet

- When the bot joins a channel conversation on its own (not @-mentioned or DMed), it can
  now decide that silence is the right move — the message wasn't for it, someone already
  answered, or a reaction says enough — instead of always producing a reply. Self-started
  replies stream live just like every other reply; the "should I stay silent?" decision is
  made before any text appears, and once the bot has begun a visible reply it always
  finishes it rather than vanishing mid-sentence. Toggle with `ENABLE_NO_REPLY_TOOL`
  (default on).
- It also applies to threads the bot is already part of: in a 1:1 thread, a message clearly
  aimed at someone else ("claude, what do you think?") no longer earns a reply about not
  replying. And the bot never posts a placeholder announcing that it's staying quiet.

### 🧭 Feature - The bot knows why it woke up

- The model now receives a compact, internal "wake context" note alongside each channel
  message telling it why it's responding — an @-mention, its name coming up in passing, a
  direct-message, a 1:1 thread reply, an ambient judgment call (with the reason), or a
  batched catch-up — plus whether the sender started the thread or joined it, and whether
  they're a person or another bot. This sharpens the bot's read of who's talking to whom
  and when a reply is actually wanted. The note is internal context only (never posted,
  never stored). Toggle with `ENABLE_WAKE_ENVELOPE` (default on).

### 🧵 Fixed - The bot reads the room before deciding to jump in

- When the bot is deciding whether an unaddressed channel message is meant for it, it now
  sees the thread's recent back-and-forth — so an unnamed follow-up like "are you not able
  to see that?" is correctly read as continuing whoever the sender was already talking to,
  instead of the bot assuming "you" means itself and barging in. This closes a live case
  where it answered a question aimed at another participant. The thread context is internal
  only (never posted, never stored) and costs no extra API calls. Tune how much it sees with
  `PARTICIPATION_THREAD_TAIL` (default 15 messages; 0 turns it off).
- The bot's awareness of channel activity is now more reliable: it takes in other apps'
  messages and its own posted replies (not just people's), so its sense of who-said-what to
  whom is complete.

### 🧠 Feature - The bot now remembers which tools it used

- When you ask the bot how it arrived at something ("did you actually look that up, or
  guess?"), it now knows — each reply quietly records the tools it ran that turn (a
  history lookup, a web search, a reaction) and reinjects that as a compact note the next
  time it reads the thread. Previously those actions left no trace in the rebuilt
  conversation, so the bot would confidently make up a wrong answer about its own past
  behavior and then contradict itself when corrected. Only tool names and a short hint of
  their arguments are kept — never their results or your content — and old records age out
  on their own. Turn it off with `ENABLE_TOOL_PROVENANCE=false`.

### ⚙️ Fixed - The settings button is back on the message

- The "⚙️ <model>" Configure button now rides the reply message itself on non-streamed
  replies too (fallback and config-off paths), instead of arriving as a separate little
  message underneath. Streamed replies already did this; now every reply is consistent.

### 🔌 Feature - A clear alarm if the Slack connection silently dies

- Very rarely a Slack socket connection can go "half-open" — the process looks healthy but
  quietly stops receiving any messages until it's restarted. The bot now watches for this
  and, if it ever happens, logs a clear error ("socket presumed dead — restart likely
  required") so the cause is obvious instead of a mystery. It only reports the problem (no
  automatic reconnection); tune or disable the watchdog with `SOCKET_LIVENESS_TIMEOUT`
  (default 600 seconds; 0 disables).

### 🩹 Fixed - Reliability hardening for the new channel/image features

- **Streaming into Slack's native message surface works again.** Slack now requires both a
  workspace id and the asking user's id to open a streamed reply in a channel, and every
  attempt was missing them — so native streaming failed on every turn and quietly fell
  back to the classic edit loop. The bot now supplies both; when either is unavailable it
  still falls back cleanly instead of erroring.
- The bot now reliably remembers its **own** streamed replies and which tools it used on
  them, so a moment later it can refer back to what it just said and did instead of
  denying it. It also now treats its own recorded "used tools" note as the authoritative
  record of what it did, so it no longer second-guesses or denies a tool it actually ran.
  (Previously these were filed under a placeholder that no longer existed, so they
  silently vanished; and even when present, the bot sometimes contradicted them.)
- **Replies with the settings footer show the full answer again.** On some non-streamed
  replies the message could render as *only* the ⚙️ settings button, hiding the actual
  answer — the reply text now always rides the message, with the button beneath it.
- The bot no longer keeps snippets of what you asked (search terms, prompts, links) in
  its internal "which tools did I use" memory — it records only the tool names and neutral
  details like result counts, never the content of your request.
- After a hiccup mid-reply, the bot finishes the reply instead of occasionally going quiet
  and leaving a half-written message stranded.
- Longer replies that get split into parts still get their settings footer, and a reply
  that never actually sent is no longer remembered as if it had been said.
- In busy channels, a delayed duplicate of an old message can no longer resurface as if
  it were new, and the bot's channel-awareness memory no longer grows without bound.
- When the bot decides to stay silent and just add reactions, it can no longer slip one
  extra reaction past its own per-turn limit or double-fire the silence.
- Emoji reactions placed at the same moment on the same message no longer occasionally
  step on each other under heavy concurrency.
- Live progress checklists now also appear when the bot **edits** an image (not just when
  it generates one), and when background image generation is turned off.
- A generated image now reliably lands in the bot's memory even if the thread was busy at
  the moment it finished uploading — so a follow-up "edit that" always finds it.
- If saving an image's details briefly fails after it was already posted, the bot no
  longer tells you the post failed — the image you can see is treated as posted.
- Assorted internal cleanups: no lingering "Generating…" status when a background image is
  in flight, no duplicate "Enhanced Prompt" messages on some surfaces, and the bot
  remembers an "I'm still working on the last image" reply in-context right away.

### 🛡️ Fixed - Upgrade safety (for the operator, not the user)

- **The v3 migrations now take a backup before they change anything.** The one-time move of
  every user to GPT-5.6 ran *before* the first tagged backup was written, so neither backup
  could restore the model and effort people had actually chosen. A `pre-v3-upgrade` backup is
  now taken at the top of the run, before any write.
- **A failed migration can no longer silently skip the rest.** Every step shared one
  error handler that swallowed the exception and abandoned the remaining steps, leaving the
  bot serving traffic on a half-migrated schema with one quiet line in the log. Each step now
  fails on its own, loudly and by name, and the others still run.
- **The database actually backs itself up now.** The docs have long promised nightly backups
  with 7-day retention; in practice `backup_database()` was only ever called by the migrations,
  so after the upgrade no backup was ever taken again. It now runs as part of the nightly
  cleanup — and backup retention no longer deletes the tagged migration backups, which it
  would have started doing (7 days after the upgrade, to the day) the moment backups became
  a nightly event.
- **`reaction_removed` is subscribed in the example manifest.** The bot has always handled the
  event, but the manifest never asked for it — so reaction counts could go up and never down.
- Fixed two bugs hiding behind duplicate function definitions: threads created through the async
  path skipped their activity-touch and user-config copy, and the OpenAI client leaked its HTTP
  session on shutdown.

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