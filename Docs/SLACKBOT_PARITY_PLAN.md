# Slackbot Parity Plan — Human-like Channel Participation

Goal: make the ChatGPT Slackbot behave like Anthropic's Claude Slackbot — participating in
channels like a human (listening beyond @-mentions, deciding when to speak, reacting with
emoji, posting top-level *and* in threads), with per-channel config, on-demand context, and
per-channel memory. Affects **public/private channel behavior**, not DMs.

This is a single-push effort (one or two sessions). Phases are ordering/dependency groupings,
not time estimates. A **dev bot** is available in this environment — every phase is validated
live against it, since the unit suite is unreliable (see Testing Strategy).

---

## Testing strategy (read first)

The existing unit suite is **substantially outdated** — independent of any of this work:

- Baseline on the *pre-upgrade* dependencies: **104 failed, 620 passed, 24 skipped, 118 errors**.
- `pytest.ini` sets `--maxfail=1`, and the *first* collected test (`test_get_or_create_thread_creates_new`)
  asserts a `created` key the method never returns — so `make test` always dies on test #1 and
  the rot below it has been invisible.
- Most errors are `coroutine ... was never awaited` — sync-style tests left behind by the
  v2.1.0 async/await refactor.

Therefore verification for this plan relies on:
1. **Targeted tests** we add/fix for the specific code we touch (not the whole legacy suite).
2. **Live dev-bot testing** in a scratch channel for each phase.
3. **Import/smoke checks** for anything dependency-sensitive.

Fixing the whole legacy suite is out of scope here; flagged as optional follow-up.

---

## Phase 0 — Environment & foundations

### 0.1 Dependency upgrade ✅ DONE
- `make lock-upgrade` regenerated `requirements.txt`; all bumps are minor/patch (no major jumps):
  openai 2.36→2.44, slack-sdk 3.41→**3.42**, aiohttp 3.13→3.14, cryptography 48→49, pandas 3.0.2→3.0.3,
  pytest 9.0→9.1, numpy 2.4→2.5, pypdf 6.11→6.14, tiktoken 0.12→0.13, etc.
- Verified: all 16 core modules import cleanly; full suite delta = **0 real regressions**
  (one apparent flip, `test_logger_format`, passes in isolation — order-dependent flakiness).
- **slack-sdk 3.42.0 natively exposes everything we need**: `chat_startStream` /
  `chat_appendStream` / `chat_stopStream`, the `slack_sdk.web.chat_stream.ChatStream` helper,
  `assistant_threads_setStatus` / `setTitle` / `setSuggestedPrompts`, and `reactions_add`.

### 0.2 Slack app manifest — events, scopes, AI-app features  ⚠️ MANUAL (user reinstalls app)
See the **Scope/Permission Changes** summary at the end. Adds channel/reaction message events,
`reactions:write`/`reactions:read`, `assistant:write`, and enables the AI/agent app surface.

### 0.3 Bot self-identity  ✅ DONE
- On startup, call `auth_test` and store `bot_user_id` + `bot_id` on the client.
- Implemented in `slack_client/base.py` + `utilities.py` (`_ensure_self_identity`, `is_own_message`, `classify_sender`).
- Files: `slack_client/base.py` (init), available to event handlers + history rebuild.
- Required by Phases 1, 2, 5 (self vs other-bot detection, self-mention detection, loop prevention).

---

## Phase 1 — Sender classification & multi-bot history fix  🐞 ✅ DONE

Active correctness bug: `thread_management.py:509-510` maps **every** bot message to
`role="assistant"`, so another bot's messages (e.g. Claude) are replayed to OpenAI as *our own*
prior turns — corrupting context in any shared thread.

- **1.1** Sender enum `human | self | other_bot`, derived from `bot_id` + `app_id` present and
  `bot_id != own bot_id` (not `subtype == "bot_message"`, which misses app-posted messages).
- **1.2** History rebuild: `self` → `assistant`; `other_bot` → `user`, name-prefixed like humans
  (`thread_management.py:516-539`); resolve the other bot's app/display name.
  Files: `slack_client/messaging.py:346,370`, `message_processor/thread_management.py:509`.
- **1.3** Stop blanket-dropping bot messages: `registration.py:15` (`and not event.get("bot_id")`).
- **Test**: dev bot + a second bot in a thread; confirm the other bot's lines arrive as named
  `user` turns, not `assistant`.

---

## Phase 2 — @mention tagging fix  🐞 ✅ DONE (human tagging)

> **Phase 2.5 ✅ RESOLVED:** verified `get_thread_history` already captures a bot's real `user_id` and
> `build_roster_text` already includes it — other bots ARE taggable when they post with a `user` field
> (test added). Inherent Slack limit: a bot posting with only a `Bxxx` bot_id (no `user`) isn't `<@>`-taggable.

Inbound mentions are stripped and discarded (`formatting/text.py:9`); outbound has no
name→`<@id>` step, so the bot emits raw display names (the `<@Peter Rotella>` bug).

- **2.1** Inbound: resolve `<@U…>` → display name and keep it (so the model knows who was
  addressed); flag whether *self* was mentioned. Replace the blind strip.
- **2.2** Outbound: post-process the model's text to convert intended mentions
  (`@Name` / known names) → valid `<@U…>` using the user/bot directory (`user_cache`,
  `users:read`). Cache name→id lookups.
- **Test**: ask the bot to "tag Peter and @Claude" — both render as real, clickable mentions.

---

## Phase 3 — Status & streaming refactor (off `chat.update`)  ✅ DONE (status + capability; native-stream flip pending live test)

> **State:** `setStatus` branded-emoji indicator wired in (fires via `send_thinking_indicator`); self-prefix
> output guard live; `NativeStreamSession` built + tested but **gated OFF** (`config.slack_native_streaming`)
> — the legacy `chat.update` path is still live. **Remaining (needs live dev-bot validation):** wire
> `NativeStreamSession` into `_handle_streaming_text_response`/`vision.py`, observe streamed UX + long-message
> splitting, then flip the flag ON. Deliberately not rewired blind (fragile overflow/split/circuit-breaker logic).

Today: post a "Thinking…" message (`messaging.py:262`) then edit it via `chat.update`
(`messaging.py:287`, `update_message_streaming:428`). `chat.update` is **Tier-3 rate-limited**
(~50/min, org-wide bucket) and tightened for non-Marketplace apps in 2025 — it will not sustain
frequent updates once we're channel-wide. Replace with the native AI-app APIs.

- **3.1** Streaming: replace the `chat.update` edit loop with `chat.startStream` /
  `chat.appendStream` / `chat.stopStream` via `slack_sdk` `ChatStream`. Note: `blocks` only
  allowed on `stopStream`, not start/append.
  Files: `slack_client/messaging.py:428`, `message_processor/handlers/{text,vision}.py`.
- **3.2** Status: use `assistant.threads.setStatus` for "thinking/working" indicators instead of
  a posted+edited message. Provide a rotating `loading_messages` array built from
  **company-branded emoji** + short phrases (e.g. `:datassential: crunching the data…`).
  Status renders as `<AppName> <status>` and auto-clears on reply.
- **3.3** Retire the manual `send_thinking_indicator` post+edit path (keep a minimal fallback for
  contexts where the assistant surface isn't active).
- **3.4** (folded in) Output hygiene now that bot turns are in history: reinforce the
  "don't echo the `Name:` prefix" system-prompt guard, and defensively strip a leading
  self-name prefix from outbound text so the model never replies as `ChatGPT: …`.
- **Caveats to validate on dev bot**: (a) does the workspace render *custom* branded emoji in
  status / loading_messages; (b) `setStatus` is tied to the assistant/agent thread surface —
  confirm where it renders in plain channels vs. relying on stream text there; (c) known edge
  cases: status can persist after reply when tools run post-response, or clear too early.

---

## Phase 4 — Emoji reactions as a response  ✅ DONE (mechanism; trigger in Phase 5)

> `react()` via `reactions_add` (treats `already_reacted` as success), `Response.reaction()` helper, and a
> `"reaction"` branch in `handle_response`. Vetted emoji set in config. The "when to react" decision is wired in Phase 5's classifier.

Claude can reply with *only* an emoji reaction. Add reaction as a first-class response type.

- **4.1** `react(channel, ts, emoji)` capability on the client (`reactions_add`).
- **4.2** Let the response path choose: **react-only**, **react + text**, or **text**. The
  classifier (Phase 5) and/or the model decides; constrain to a vetted emoji set.
- **Test**: dev bot reacts 👍 to an acknowledgeable message without posting text.

---

## Phase 5 — Channel listening + classifier wake-gate  ⭐ ✅ DONE (gated OFF by default)

> **Out-of-the-box: no behavior change** (`enable_channel_listening=False`). When enabled,
> default `tag_only` responds only when clearly addressed (name/alias whole-word, or 1:1 thread reply);
> `<@bot>` deduped against `app_mention`; onboarding bypassed in channels; own-message loop guard first.
> Operator opts into unprompted participation via `CHANNEL_RESPONSE_MODE=auto_respond` (wake classifier).
> **Live-validation needed:** flip `ENABLE_CHANNEL_LISTENING=true` on dev bot; tune `WAKE_CLASSIFIER_SYSTEM_PROMPT`.

Today the bot only receives `app_mention` + `message.im`, so it *cannot* see normal channel
messages. Decouple **trigger** (cheap classifier) from **context** (built only on wake).

- **5.1** Handle `message.channels` / `message.groups` / `message.mpim`. **Short-circuit
  self/own-bot messages immediately** (loop + cost guard) using Phase 0.3 identity.
- **5.2** Lightweight classifier (small/fast model) on each top-level message →
  `respond | react | ignore`, **before** any history fetch. Threshold = "does this look aimed at
  me" (name, reply-to-our-message, or clear intent — not literal `<@>`).
- **5.3** Apply the same judgment to thread replies, so the bot can stay silent in a thread it's
  in when a message isn't for it.
  Files: `slack_client/event_handlers/registration.py`, `message_events.py`,
  new classifier prompt in `prompts.py`.
- **Test**: in dev channel, "Claude, you there?" (name, no @) wakes it; unrelated chatter does
  not; a thread aside meant for someone else is ignored.

---

## Phase 6 — Reply placement & channel envelope  ✅ DONE (reply-in-thread; window deferred)

> Channel replies default to reply-in-thread (`thread_id = thread_ts or ts`); bare top-level keys as a
> length-1 thread reusing existing per-thread assembly. `channel_context_window=0` — the bounded
> recent-channel window is a documented follow-up (enable + size via config when wanted).

- **6.1** Default to **threaded replies** even in auto-respond mode (preserves the entire
  existing per-thread context model); allow top-level in-channel replies per channel config.
- **6.2** For a bare top-level wake (no thread), assemble a **bounded recent-channel window** as
  the envelope (last N msgs / few minutes) — never the whole channel scrollback.
  Files: `message_processor/thread_management.py`, `slack_client/messaging.py`.

Note: a top-level channel message already *is* a thread (`thread_ts == ts`); the existing
`channel_id:thread_ts` key needs no change to represent it.

---

## Phase 7 — Per-channel config & response modes  ✅ DONE

> `channel_settings` table (`response_mode`/`directives`/`reply_in_channel`) + sync/async get/set;
> `_get_channel_response_mode` is DB-backed (per-channel override → global fallback); directives flow to
> BOTH the wake classifier (gating) and the response system prompt (ground rules once awake). **Also fixed
> the bot-onboarding-nag bug** — `other_bot` senders bypass the settings/welcome flow (humans unchanged).
> **Global defaults live in `.env`** (see the "Channel participation & UX" section of `.env.example`).
> **Per-channel overrides are edited in Slack via a response footer** (Claude-style): every channel response
> carries a small context line (model) + a **⚙️ Configure** button (`open_channel_settings` action) that opens a
> modal — response mode (incl. "inherit"), ground-rule directives, reply-placement. **Any channel member** may
> open and save (no admin gating). "Inherit" clears the DB override (NULL) so the global default applies again.
> The DB (`channel_settings`) is storage only. **No slash command and no manifest change / reinstall** —
> interactivity is already enabled. Footer is `ENABLE_RESPONSE_FOOTER` (default on), posted as a separate trailing
> message (never touches the split/streaming path), channels-only, text-responses-only. No row = unchanged global
> defaults. **Follow-up:** `reply_in_channel=True` top-level posting is plumbed but not yet wired (modal control labeled "not yet active").

- **7.1** `channel_settings` table: `response_mode` (`tag_only` | `auto_respond` | `off`) +
  freeform `directives` ("only jump in on deploy failures", "stay quiet unless tagged").
  Files: `database.py` (schema + migration), `config.py`.
- **7.2** Surface mode + directives to the classifier and the system prompt. **Bypass the
  per-user settings-completion gate for bot senders and channel context** (a bot can't click the
  settings modal — `message_events.py:91-261`). Configure via DB flag now; slash command later.
- **Test**: flip a dev channel to `auto_respond` with a directive; confirm behavior changes.

---

## Phase 8 — On-demand fetch tools  ✅ DONE (executor + privacy built; model-wiring deferred)

> **Architecture finding:** no local function-call loop exists — Responses API uses only server-side tools
> (web_search + MCP). So `slack_client/history_tool.py` (executor, schemas, dispatcher, hard limit) and the
> **tool-layer privacy gate** (`conversations_info` → allow public or bot-member-private; refuse others with
> no read) are built + tested (16 tests), gated by `enable_history_tools`. **Deferred (needs live validation):**
> wiring a function-call loop into `create_response_with_tools` + streaming paths so the model can call it —
> same risky rewire Phase 3 declined; or expose via a small MCP server (server-side, no loop). **Search excluded**
> (legacy `search:read` deprecated/user-token, not granted; history-fetch is the reliable path).

Replaces front-loading with deliberate pulls (Claude's "fetch the slice I need").

- **8.1** Tools: fetch thread/channel history (bounded), Slack message search. Register through
  `mcp_manager.py` / the tool layer so both providers can call them.
- **8.2** **Privacy enforced at the tool layer** (not the prompt): scope reads/search to
  **public channels + channels the bot is a member of**; refuse private channels it isn't in.
- **Test**: ask "what did we decide in #x last week" → bot fetches only that slice; private
  non-member channel access is refused.

---

## Phase 9 — Per-channel memory  ✅ DONE (inject + post-response extraction)

> `channel_memory` table (rows, scoped) + sync/async CRUD; **read** = concise "CHANNEL MEMORY" block injected
> into the system prompt on each response; **write** = a post-response utility-model extraction step
> (`extract_memory`, hooked at `_async_post_response_cleanup`) that decides none/add/update, dedupes, caps at
> `memory_max_rows`, writes `channel` scope only, and never touches the reply path on failure. No function-call
> loop needed (matches "decide to append each turn if needed"). **Privacy** enforced in the query:
> `WHERE (scope='channel' AND channel_id=?) OR scope='workspace'`. Flag `enable_channel_memory` (default ON);
> no rows = unchanged prompt. **Live-validation:** extraction quality/cadence tuning; vision-only turns don't extract.

- **9.1** `channel_memory` table: discrete rows `(id, channel_id, scope, content, author,
  created_ts, updated_ts)` — **not** an append-only blob. `scope` = channel-private or
  workspace-shared (read-mostly).
- **9.2** Model-invoked tools `remember` / `update_memory` / `forget`, gated on judgment
  (write a durable *fact*, not transcript). Inject the channel's rows into context on wake
  (cheap — small per channel).
- **9.3** Partition by scope: a private channel's memory never bleeds into shared memory or
  other channels.
  Files: `database.py`, tool layer, `prompts.py`.
- **Test**: tell the bot a durable fact; confirm a row is written and recalled next turn;
  confirm private-channel facts don't surface elsewhere.

---

## Remaining last-mile work (needs live dev-bot validation)

All 9 substantive phases are implemented (Phases 0.3–9). Full unit suite is at the exact pre-work baseline
(105 failed / 118 errors — pre-existing rot) + 145 new passing tests = **zero net regressions**. Nothing committed.
Deferred "needs eyes on the bot" items, each behind a flag / safe default:

1. **Branded emoji** (you): replace `config.status_loading_messages` placeholders with real Datassential custom-emoji names.
2. **Native streaming wiring**: wire `NativeStreamSession` into `_handle_streaming_text_response` + `vision.py`, then flip `slack_native_streaming=True` (capability built/tested, default OFF; legacy `chat.update` live).
3. **Function-call loop**: needed to make the Phase-8 history-fetch tool model-reachable (or expose via a small MCP server). Executor + privacy gate are built/tested.
4. **`reply_in_channel` posting**: plumbed (DB + metadata) but send still threads; wire top-level posting.
5. **Recent-channel-window envelope**: `channel_context_window=0`; implement bounded window if wanted.
6. **Classifier + memory-extraction tuning**: `WAKE_CLASSIFIER_SYSTEM_PROMPT` and `MEMORY_EXTRACTION_SYSTEM_PROMPT` are only unit-tested with mocked output — observe/tune live.
7. **`setStatus` rendering**: confirm where it shows (assistant threads) vs. cleanly no-ops (plain channels).

To exercise channel participation on the dev bot: set `ENABLE_CHANNEL_LISTENING=true` (default mode `tag_only`;
`CHANNEL_RESPONSE_MODE=auto_respond` for unprompted). Out of the box everything stays gated OFF / conservative.

## Phase 10 — Context-loading optimization (later, optional)  ⏳ FUTURE (after #2/#3 proven live)

Once Phase 8 tools are proven, relax the front-load-everything model: load a recent window by
default and pull more on demand. Reword the CLAUDE.md "ALWAYS include full context" rule then —
**not before**. Today's `_smart_trim_with_summarization` already bounds active threads and works.

---

## Dependency / sequencing graph

```
0.1✅ ──► 0.2(manual) ──► 0.3 ──► 5 ──► 6
                  │               │
                  ├──► 3          └──► 7 ──► 9
                  └──► 4
1 ───────(independent, do first; bug fix)
2 ───────(independent, do first; bug fix)
8 ───────(independent; needed before 10)
```

Recommended order: **1 + 2** (bug fixes, no reinstall) → **0.2/0.3** (manifest + self-id) →
**3 + 4** (status/streaming + reactions) → **5 + 6** (keystone) → **7** → **8** → **9** → 10 later.

---

## Slack scope / permission changes (summary)

**Bot event subscriptions** — add:
- `message.channels` (public channel messages) — *required for channel listening*
- `message.groups` (private channels), `message.mpim` (group DMs)
- `reaction_added` (optional — only if the bot should observe reactions)
- `assistant_thread_started`, `assistant_thread_context_changed` (if using the assistant surface for status)

**Bot OAuth scopes** — add:
- `reactions:write` — react to messages (Phase 4)
- `reactions:read` — observe reactions (only if subscribing to `reaction_added`)
- `assistant:write` — `assistant.threads.setStatus` and the agent surface (Phase 3)
- `channels:history` ✅ already present · `groups:history` ✅ already present
- `search:read` — only if Slack message search is used (Phase 8); confirm availability for bot tokens
- already present and reused: `chat:write`, `chat:write.customize`, `users:read`, `commands`, `files:read/write`

**App config**:
- Enable the **AI app / agent** features (assistant surface) for `setStatus` + streaming UX.
- Reinstall the app to the workspace after manifest changes (done manually — prod is manual).

**Operational note**: subscribing to all channel messages massively increases event volume —
the Phase 5 classifier must be cheap and self/own-bot messages short-circuited before any model
call or history fetch.
