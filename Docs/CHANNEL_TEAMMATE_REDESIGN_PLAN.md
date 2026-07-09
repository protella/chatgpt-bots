# Channel Teammate Redesign Plan

**Goal:** the bot behaves like a human teammate in channels — it sees every message, decides on
its own whether to reply, react, or stay silent (in the main channel or threads), manages its own
context (pulling older Slack history/search on demand), takes "butt out" feedback gracefully,
threads its replies appropriately, and reacts tastefully. DMs keep today's experience, plus
reactions and the streaming/status upgrades. Nothing here changes prod until flags are flipped.

**Builds on** the parity work committed at `57abc54` (see `Docs/SLACKBOT_PARITY_PLAN.md`).
This plan REPLACES the parity plan's "remaining last-mile work" list — every deferred item is
absorbed into a phase below.

**Platform facts this design relies on** (verified against current Slack docs, June 2026):
- `agent_view` is the current agent surface; legacy `assistant_view` is deprecated (sunset
  pending). Setup event is `app_home_opened` (check `tab == "messages"`), not
  `assistant_thread_started`. Requires slack-sdk ≥ 3.43 / bolt ≥ 1.29 (we're on both).
- `assistant.search.context` works with a **bot token** + granular scopes
  (`search:read.public`, `.private`, `.im`, `.mpim`) but requires an **`action_token`** minted
  by a triggering `message`/`app_mention` event — a bot cannot search except in response to a
  user interaction. This is our anti-exfiltration backstop.
- `chat.startStream/appendStream/stopStream` (Oct 2025) is the native streaming path;
  `assistant.threads.setStatus` now **auto-opens the thread** it's called on.
- Rate limits: internal custom apps are exempt from the 2025 non-Marketplace caps —
  `conversations.history/replies` stay Tier 3 (~50 rps), `reactions.add` Tier 3,
  `assistant.search.context` 10–400 rps team-size-dependent. Event delivery cap 30k/60s/workspace.

---

## 1. What we keep / extend / replace

| Component (post-57abc54) | Verdict | Why |
|---|---|---|
| Thread-as-context-unit + `ThreadStateManager` + rebuild-from-Slack | **KEEP** | Threads remain the unit of *conversation*; the redesign adds channel-level *awareness* around them, not instead of them. |
| Sender classification, @mention tagging, roster injection, footer skip | **KEEP** | Live fixes, already correct. |
| `_handle_channel_message` heuristics (name alias, 1:1 thread ownership, mention dedup) | **KEEP** as prefilters | They become the cheap short-circuit layer in front of the decision engine. |
| Wake classifier (`classify_wake`, one-word respond/react/ignore, message-text-only) | **REPLACE** → ParticipationEngine (§4) | It's blind: no channel context, no memory, no participation history, no emoji choice, no placement. |
| Post-response memory extraction (`extract_memory` + `_async_extract_channel_memory`) | **REPLACE** → model-invoked memory tools (§3), extraction kept as flag-gated fallback for one release | Judgment belongs to the model in-flight ("the bot decides to append on its own each turn"); a second guessing pass is weaker and costs an extra call every exchange. |
| `history_tool.py` executor + privacy gate | **KEEP**, finally wired | It was built for the function-call loop; the loop now exists (§2). |
| `channel_settings` / `channel_memory` tables, channel modal, Configure footer | **EXTEND** | Add participation level + snooze columns; modal gains a participation control. Thread-level settings modal untouched. |
| `assistant_events.py` (assistant_thread_started greeting) | **MIGRATE** to `app_home_opened` (§6), old handler kept during transition |
| `NativeStreamSession` | **KEEP + WIRE** into the streaming handler (§6) — it was built and tested but never connected. |
| chat.update streaming edit-loop | **DEMOTE** to fallback behind native streaming. |
| DM pipeline (`_handle_slack_message`, onboarding, per-user prefs) | **KEEP untouched** | Explicit requirement. |

---

## 2. The keystone: local function-call loop

Today `openai_client` only passes server-side tools (`web_search`, MCP). The model literally
cannot call our code. Everything else in this redesign (search, history, memory-on-demand,
model-chosen reactions) hangs off fixing that.

### New files
- `openai_client/api/tool_loop.py` — the loop
- `tool_registry.py` (root, platform-agnostic) — registry + `ToolContext`

### Registry
```python
@dataclass
class ToolContext:            # built per-request by the processor
    channel_id: str
    thread_ts: str
    trigger_ts: str           # ts of the message we're answering
    action_token: str | None  # from the triggering Slack event (search API)
    client: Any               # platform client (SlackBot)
    db: Any

class ToolRegistry:
    def register(self, schema: dict, executor: Callable[[ToolContext, dict], Awaitable[dict]])
    def schemas(self, thread_config) -> list[dict]      # only enabled tools
    async def dispatch(self, ctx, name, arguments) -> dict
```
Executors return JSON-serializable dicts; results are truncated to
`TOOL_RESULT_MAX_CHARS` (default 20 000) before being fed back.

### Loop semantics (Responses API)
1. Build `input` as today; `tools = server_tools + registry.schemas(...)`.
2. `responses.create(...)`. Collect `function_call` output items
   (`item.type == "function_call"` → `name`, `call_id`, `arguments`).
3. If none → done (normal text path). If present → run them **in parallel**
   (`asyncio.gather`, per-call timeout `TOOL_CALL_TIMEOUT` default 20s; a timeout returns
   `{"ok": false, "error": "timeout"}` to the model, never raises).
4. Append each `function_call` item + a `{"type": "function_call_output", "call_id", "output"}`
   item to `input`; go to 2.
5. Hard caps: `MAX_TOOL_ROUNDS` (default 4) and `MAX_TOOL_CALLS_PER_TURN` (default 8). On cap,
   re-call with `tool_choice: "none"` so the model must answer with what it has.

**Streaming composition:** tool rounds run with `stream=True` exactly like today's MCP path —
the deltas of intermediate rounds are *not* sent to Slack; instead `tool_callback` drives the
status line ("🔎 searching Slack…"), and only the final round's text streams to the message.
This drops into `_handle_streaming_text_response` where `_build_tools_array` is called today;
the non-streaming `_handle_text_response` gets the same loop without callbacks.

**Function-call args in streams:** function_call arguments arrive via
`response.function_call_arguments.delta/.done` events; we only need the completed items, which
are available on `response.output_item.done` — no delta parsing required.

### Launch tool set
| Tool | Executor | Notes |
|---|---|---|
| `fetch_channel_history`, `fetch_thread_messages` | exists (`history_tool.py`) | wire as-is; privacy gate already at tool layer |
| `search_slack` | new (§3a) | `assistant.search.context`; needs action_token |
| `remember_fact` / `update_fact` / `forget_fact` | new (§3b) | channel memory writes |
| `react_to_message` | new (§3c) | emoji side-effect on the triggering (or referenced) message |

`_build_tools_array` grows a `registry` parameter and appends `registry.schemas(thread_config)`.
Tools ship for **all surfaces** (DMs get search/history/memory-read too — a "bonus" the user
asked for), with memory writes restricted to channel scope.

---

## 3. New local tools

### 3a. `search_slack` — `slack_client/search_tool.py`
- Schema: `{query: string, scope?: "channel"|"workspace", limit?: int}`.
- **action_token plumbing:** Slack includes an `action_token` on message/app_mention event
  payloads for AI apps. Capture it in `_event_to_message` →
  `message.metadata["action_token"]` → `ToolContext.action_token`. Executor calls
  `assistant.search.context(query=..., action_token=ctx.action_token, channel_types=...,
  context_channel_id=...)`. If the token is absent/expired → `{"ok": false, "error":
  "search_unavailable"}` (model falls back to history tools).
- **Scope gate in code, not just manifest:** `SLACK_SEARCH_CHANNEL_TYPES` env (default
  `public_channel`) maps to the API's channel-type filter. Even though the manifest grants the
  granular scopes, the executor only ever requests the configured types — prompt-injected
  "search his DMs" physically can't widen it.
- Results normalized to `{channel, ts, user, text, permalink}` and capped.

### 3b. Memory tools — executors in `message_processor/memory_tools.py`
- `remember_fact {content}`, `update_fact {id, content}`, `forget_fact {id}` — thin wrappers
  over the existing `channel_memory` CRUD, channel-scope only, row cap enforced exactly as the
  extractor does today. Current memory (numbered) is already injected into the system prompt, so
  the model can reference ids.
- System-prompt addendum tells the model: *you may save durable channel facts; bias strongly
  against saving; update/supersede rather than duplicate* (reuse the judgment text from
  `MEMORY_EXTRACTION_SYSTEM_PROMPT`).
- `ENABLE_MEMORY_EXTRACTION_FALLBACK` (default **false** once tools land) keeps the old
  post-response extractor available for one release in case tool-driven writes under-perform.

### 3c. `react_to_message` — executor on `SlackMessagingMixin`
- Schema: `{emoji: string, ts?: string}` (default `ts` = the triggering message). Emoji must be
  in `REACTION_EMOJIS` allowlist (env) or the call is refused — keeps reactions on-brand and
  prevents anything embarrassing.
- **Two reaction paths, both wanted:**
  - *reaction-instead-of-reply* — decided by the ParticipationEngine (fast path, no main-model
    call), as today.
  - *reaction-alongside-reply* — the responding model itself reacts mid-turn (e.g. ✅ on the
    request it just fulfilled). That's this tool.
- Dedup guard: at most one bot reaction per message ts (in-memory set), never react to own msgs.

---

## 4. ParticipationEngine (decision engine v2)

Replaces `classify_wake`. New file `message_processor/participation.py` (platform-agnostic core)
+ prompt in `prompts.py` (`PARTICIPATION_SYSTEM_PROMPT`).

### Pipeline per channel message (cheap → expensive)
```
event → prefilters (no LLM, µs) → debounce (0.8s quiet window) → engine (1 utility call) → act
```

**Prefilters (short-circuit, in `_handle_channel_message`):** own message; subtype; mode=off;
snoozed (§4c); explicit @mention/alias/1:1-thread → *respond directly, skip the engine* (current
behavior, kept); mode=tag_only and not addressed → ignore (current behavior — tag_only never
pays for a classifier call).

**Debounce (new, `ChannelPulse.debounce()`):** in auto_respond mode, rapid-fire messages
(< `WAKE_DEBOUNCE_SECONDS`, default 0.8s, per channel) collapse into one engine call over the
batch — a person typing four short lines shouldn't cost four calls or produce a reply to line 2.

### Engine inputs (one utility-model call, JSON out)
- the message (or debounced batch), with sender name + `is_thread_reply`
- **channel window**: last `CHANNEL_PULSE_SIZE` (default 30) channel messages from ChannelPulse
  (§5), rendered as `[12:03] Peter: …` one-liners — this is what makes it a *participation*
  decision, not a message classification
- channel memory facts + operator directives (existing injection)
- **participation stats**: bot replies in this channel in the last hour / last 20 messages
  (computed from ChannelPulse) — the self-throttle signal
- channel mode + participation level (§4c)

### Output (strict JSON, falls back to `ignore` on any parse/API failure — unchanged principle)
```json
{"action": "respond" | "react" | "ignore" | "backoff",
 "emoji": "thumbsup",            // when action=react; must be in allowlist
 "placement": "thread" | "channel",  // when action=respond
 "reason": "…"}                   // logged, never posted
```

### 4a. Placement policy
- Default `thread` (reply to the message's thread; a top-level message keys as its own
  length-1 thread — unchanged model).
- `channel` placement (a genuine top-level reply, not threaded) is allowed only when the
  channel setting `reply_in_channel` is on AND the engine chooses it (e.g. answering a
  question asked at channel level where a threaded reply would hide the answer). This wires the
  parity plan's dangling `reply_in_channel` flag: `main.handle_message` passes
  `thread_id=None`-equivalent (post without `thread_ts`) when
  `response.metadata["placement"] == "channel"`.
- Images: always threaded (existing behavior, hardcoded).

### 4b. Self-throttling
Prompt rule + hard rail. Prompt: "you are one voice among teammates; if you've spoken recently
and add only marginal value, stay silent." Hard rail in code:
`MAX_UNPROMPTED_REPLIES_PER_HOUR` (default 6, per channel, unprompted = not addressed) — when
exceeded, engine verdicts of `respond` are downgraded to `react`/`ignore`. Addressed messages
are never throttled.

### 4c. "Butt out" loop
- Engine gets a 4th verdict: **`backoff`** — the message is social feedback aimed at the bot
  ("chill", "let the humans talk", "stop replying to everything").
- On `backoff`: (1) react with 🤐/👍 (ack without more words), (2) set
  `channel_settings.snooze_until = now + BACKOFF_SNOOZE_HOURS` (default 4h), (3) write a
  channel memory fact ("On <date> the channel asked for less unprompted participation") so the
  behavior persists past the snooze, (4) log.
- While snoozed: prefilter drops all *unprompted* engagement; @mentions/aliases/1:1 threads
  still work (told to be quiet ≠ deaf).
- New `channel_settings` columns: `participation_level` TEXT NULL
  (`quiet`/`normal`/`chatty` — fed to the engine prompt), `snooze_until` TEXT NULL. Channel
  modal gains a Participation select (inherit/quiet/normal/chatty); Configure-footer stays the
  entry point.

### Cost & latency at scale
- gpt-5-mini, ~1–2k input tokens/call with the 30-line window: a busy 300-msg/day channel in
  auto_respond ≈ 300 calls ≈ **~$0.15/day**; tag_only channels cost zero. Prefilters + debounce
  cut the real number well below message count.
- Latency: +0.5–1.5s before a reply starts in auto_respond; imperceptible next to generation.
  Socket-mode acks are immediate (Bolt acks the envelope before our handler runs), so
  classifier latency can't trigger Slack redelivery.

---

## 5. ChannelPulse — ambient channel context

New file `slack_client/channel_pulse.py`.

- **Store:** in-memory per-channel ring buffer, `deque(maxlen=CHANNEL_PULSE_SIZE)` of
  `{ts, thread_ts, sender_name, sender_type, text[:300]}`. Fed by **every** channel message
  event (in `_handle_channel_message`, before any filtering except subtype) — even messages the
  bot ignores update its awareness. Not persisted: on cold start it lazily backfills with ONE
  `conversations.history(limit=CHANNEL_PULSE_SIZE)` call the first time a channel's window is
  requested (Tier 3, we're exempt from the non-Marketplace caps). DB persistence is deliberate
  non-goal — this is a peripheral-vision cache, not a source of truth.
- **Consumers:**
  1. ParticipationEngine input (§4).
  2. **Response-context envelope**: when the bot responds to a *top-level* channel message
     (thread of length 1), a "Recent channel activity (context only — reply to the last
     message)" block renders the window into the system prompt. This resolves the parity plan's
     `channel_context_window=0` follow-up; the env var now controls how many window lines the
     *responder* sees (default 15; 0 = off), while threads keep their full existing context.
     For deeper digging the model uses the history/search tools instead of a bigger envelope.
- Token cost: 30 one-liners ≈ 600–900 tokens — negligible against the 920k budget.
- The buffer also feeds participation stats (bot message counts) with zero extra API calls.

---

## 5b. Slack-native context: retire the message mirror (user decision 2026-07-09)

**Finding (code trace):** the `messages` table is not a backup — it's a cache-first PRIMARY.
`get_or_create_thread_async` loads DB rows into thread state (`thread_manager.py:861-873`) and
`_get_or_rebuild_thread_state` only rebuilds from Slack when that cache is empty
(`thread_management.py:528`). Consequence: messages edited/deleted in Slack after caching stay
in context forever (staleness bug). The rate-limit rationale is dead — internal custom apps
kept Tier 3 (~50 req/min, 1000 msgs/call) after the May-2025 changes.

**New precedence: Slack is the only transcript.** Every context build fetches
`conversations.replies` fresh (bounded window). No OpenAI-side history either
(`store=False` stays; no `previous_response_id` chaining — user decision).

**The DB keeps only what Slack doesn't have:**
- `user_preferences`, `threads.config_json`, `channel_settings`, `channel_memory`,
  `modal_sessions` — load-bearing config/memory, untouched.
- `images` — derived hidden context (vision analyses, generation prompts for edit flows).
  Stays; re-injected on rebuild exactly as today (`_inject_image_analyses`).
- `documents` — extraction cache (avoid re-download/re-parse per rebuild). Stays, keyed to the
  Slack file id so a missing file can expire the row. The never-read `documents.summary`
  column is dropped.
- **NEW `thread_summaries`** — the compaction store. When a long thread exceeds the token
  budget, `_smart_trim_with_summarization` writes: `thread_key`, `summary_text`,
  `boundary_ts` (everything ≤ this ts is covered by the summary), `refs_json` (structured
  refs to files/images/links inside the summarized span so edit/vision flows still resolve
  them), `updated_ts`. Rebuild = stored summary head + `conversations.replies(oldest=boundary_ts)`
  tail + injected hidden context. Re-summarization extends the same row (rolling).
- Long CHANNELS need no equivalent: ChannelPulse is a bounded window by design, durable facts
  belong in `channel_memory`, and the tool loop lets the model fetch/search older history
  on demand.

**Deletions:** `cache_message(_async)` call sites, `get_cached_messages(_async)` reads, the
DB-first branch in thread rebuild, message-cache rewrite in trim/summarize. `threads` table
stays (config only).

**Cleanup migration (one-time, on startup):** take a forced pre-migration backup into
`data/backups/` (tagged `pre-v3-mirror-drop`), then `DROP TABLE IF EXISTS messages`, drop the
dead `documents.summary` column, `VACUUM` to reclaim space, and log rows removed + bytes
reclaimed. Idempotent (guarded on table existence). Rollback = restore the tagged backup;
in-flight conversations are unaffected since context now always comes from Slack.

## 6. agent_view migration + native streaming

- **Events:** register `app_home_opened` (filter `tab == "messages"`) → existing greeting +
  suggested-prompts handler; register `app_context_changed` → existing debug logger. KEEP the
  `assistant_thread_started/context_changed` registrations during transition (whichever fires,
  fires once; the greeting handler is idempotent-enough via best-effort semantics). Remove
  legacy handlers one release after the manifest flips.
- **Manifest:** `features.agent_view` with `agent_description` + manifest-level
  `suggested_prompts` (they can stay dynamic via setSuggestedPrompts too; manifest ones are the
  zero-code default). Note Slack's UI rewrites this block when toggles are flipped — treat the
  Slack console as source of truth and re-export.
- **setStatus gotcha:** setStatus now auto-opens the thread. Consequence: never call it
  speculatively on channel messages the engine might ignore — only after a `respond` verdict,
  and only on assistant/DM surfaces (current `set_assistant_status` call sites comply; add a
  guard comment + test).
- **Native streaming (absorbs parity last-mile item #2):** in
  `_handle_streaming_text_response` and the vision streaming path, wrap the update sink:
  ```
  if client.supports_native_streaming(): session = client.begin_native_stream(...)
      → session.start() on first chunk, session.update(cumulative) per tick,
        session.finish(final) at end; on any inert-fallback → legacy update_message_streaming
  ```
  `NativeStreamSession` already handles cumulative→delta bridging and self-inerting fallback;
  the existing `RateLimitManager` circuit breaker stays wrapped around whichever sink is live.
  Flip `SLACK_NATIVE_STREAMING=true` in dev after live validation; the "Thinking…" placeholder
  message is skipped when native streaming starts (startStream creates the message).
- **feedback_buttons / context_actions:** adopt LATER (optional Phase H) — thumbs up/down on AI
  responses is genuinely useful signal, but it needs a feedback sink (DB table + log) to be
  worth the pixels. Not load-bearing for teammate behavior.

## 7. DMs
Unchanged pipeline, three additive gains: the model's `react_to_message` tool works in DMs;
native streaming + setStatus upgrades apply; search/history/memory-read tools are available.
Channel machinery is structurally incapable of touching DMs: ChannelPulse, ParticipationEngine,
and footer all key off `channel_type != im` / `channel_id.startswith("D")` guards that already
exist. A regression test asserts a DM message never enters the participation path.

---

## 8. Config surface (env; sane defaults; nothing global moves to DB)

| Env var | Default | Meaning |
|---|---|---|
| `ENABLE_TOOL_LOOP` | `true` | master switch for the local function-call loop |
| `MAX_TOOL_ROUNDS` / `MAX_TOOL_CALLS_PER_TURN` | `4` / `8` | runaway caps |
| `TOOL_CALL_TIMEOUT` | `20` | seconds per executor |
| `TOOL_RESULT_MAX_CHARS` | `20000` | result truncation |
| `ENABLE_SLACK_SEARCH_TOOL` | `true` | search_slack registration |
| `SLACK_SEARCH_CHANNEL_TYPES` | `public_channel` | hard scope gate for search executor |
| `ENABLE_MEMORY_TOOLS` | `true` | remember/update/forget tools |
| `ENABLE_MEMORY_EXTRACTION_FALLBACK` | `false` | legacy post-response extractor |
| `ENABLE_REACT_TOOL` | `true` | model-invoked reactions (allowlist still `REACTION_EMOJIS`) |
| `CHANNEL_PULSE_SIZE` | `30` | ring-buffer depth per channel |
| `CHANNEL_CONTEXT_WINDOW` | `15` | window lines shown to the responder (existing var, now live) |
| `WAKE_DEBOUNCE_SECONDS` | `0.8` | rapid-fire collapse |
| `MAX_UNPROMPTED_REPLIES_PER_HOUR` | `6` | hard self-throttle |
| `BACKOFF_SNOOZE_HOURS` | `4` | butt-out snooze duration |
| Existing: `ENABLE_CHANNEL_LISTENING` (still master, still default off), `CHANNEL_RESPONSE_MODE`, `BOT_NAME_ALIASES`, `REACTION_EMOJIS`, `SLACK_NATIVE_STREAMING`, memory/history flags | | unchanged |

**Per-channel (modal, DB):** response mode (existing), directives (existing), reply placement
(existing), **participation level (new)**. **Per-thread (existing modal):** unchanged.
`snooze_until` is engine-written state, shown in the modal ("Snoozed until 6pm — clear?") but
set by social feedback, not a form field.

## 9. Manifest deltas (sample manifest; env-specific copy is gitignored)

Scopes to ADD: `search:read.public` (+ `search:read.private` only if/when
`SLACK_SEARCH_CHANNEL_TYPES` is widened — ship the manifest matching the code gate),
`reactions:read` (already present), `assistant:write` (already present).
Events to ADD: `app_home_opened`, `app_context_changed`, `reaction_added` (Phase H feedback
loop; harmless to subscribe early). KEEP `assistant_thread_started/_context_changed` until the
migration release, then drop. Everything else (message.channels/groups/im/mpim, histories,
reactions:write) landed in the parity manifest.

---

## 10. Rollout phases

Each phase: independently shippable, flag-gated, unit tests + a live dev-bot checklist, rollback
= flip the flag. Order matters — everything leans on Phase A.

**A. Function-call loop + history tools live** — `tool_loop.py`, `tool_registry.py`, registry
into `_build_tools_array`, wire `history_tool.py`. Files: `openai_client/api/tool_loop.py`,
`tool_registry.py`, `message_processor/handlers/text.py`, `openai_client/base.py`, `config.py`.
Tests: loop round-trip w/ mocked API (single, parallel, cap-hit, timeout, malformed args);
history tool dispatch. Live: DM "fetch the last 10 messages from #chatgpt-bot-test and
summarize". *Replaces parity last-mile #3.*

**B. search_slack** — action_token capture in `_event_to_message`, executor, scope gate,
manifest scope. Tests: token plumbing, scope-gate refusal, missing-token fallback. Live: "search
slack for the parity plan discussion".

**C. Memory tools** — 3 executors, prompt addendum, extraction fallback flag default-off.
Tests: cap enforcement, scope restriction, forget. Live: "remember that demos are on Fridays" +
implicit save; verify recall in a new thread. *Retires the extractor (kept behind flag).*

**D. react tool** — executor + allowlist + dedup. Tests: allowlist refusal, default-ts, dedup.
Live: ask for something and watch for ✅ alongside the reply; DM reactions.

**S. Slack-native context (drop the message mirror)** — see §5b. Always rebuild from
`conversations.replies`; add `thread_summaries` (rolling summary + boundary_ts + refs_json);
delete cache-first reads + cache writes; keep `images`/`documents` as derived caches; drop
`documents.summary`. Files: `thread_manager.py`, `message_processor/thread_management.py`,
`database.py`, `message_processor/handlers/vision.py` (unchanged injection path verified).
Tests: rebuild-always-fetches, summary head + tail composition, refs preserved through
compaction, edited-message freshness (the old staleness bug as a regression test). Live: long
thread past the token budget → responses keep file/image awareness; edit a message mid-thread
→ bot sees the edit. *Fixes the staleness bug; depends on Phase A (tool loop shrinks upfront
context needs).*

**E. ChannelPulse + response envelope** — buffer, backfill, envelope injection, participation
stats. Files: `slack_client/channel_pulse.py`, `message_events.py`, `base.py` (processor).
Tests: ring behavior, backfill-once, envelope rendering, DM exclusion. Live: two unrelated
top-level questions; second answer shouldn't bleed the first's thread but engine should show
awareness.

**F. ParticipationEngine** — new prompt + JSON verdict, debounce, placement, throttle, backoff +
snooze + modal participation control; delete `classify_wake` call sites (keep fn one release).
Files: `message_processor/participation.py`, `prompts.py`, `main.py` (gate swap),
`message_events.py`, `database.py` (2 columns), `settings_modal.py`, `settings.py`. Tests:
verdict parsing/fallback, throttle math, snooze lifecycle, backoff writes, placement wiring,
tag_only-never-calls-engine. Live: auto_respond in test channel — the full etiquette script
(answerable question → respond; human-to-human → silent; "thanks" → react; "chatgpt butt out" →
🤐 + snooze; @mention while snoozed → answers). **This phase rewrites
`test_wake_classifier.py` / parts of `test_channel_listening.py`.**

**G. agent_view migration + native streaming flip** — events, manifest, setStatus guard, stream
sink wiring, live validation, then `SLACK_NATIVE_STREAMING=true` in dev. *Absorbs parity
last-mile #2/#7 and the branded-emoji fill-in (#1, user action).* Tests: sink selection +
fallback, app_home_opened tab filter. Live: split-view greeting, streamed reply rendering,
status with branded emoji.

**H. (Optional) feedback_buttons + reaction_added ingestion** — thumbs signal → DB; engine may
read per-channel feedback ratio later.

Suggested releases: this ships as **v3.0.0** (major — model lineup reduction + context
paradigm change, per user). A–D + S = "the bot can act, statelessly"; E–F = "the bot has
judgment"; G = "native surface"; H opportunistic. Bundle per user preference at release time.

## 11. Risks & open questions

- **Prompt injection → tool abuse.** Search/history are read-only and scope-gated in code;
  memory writes are capped + channel-scoped; reactions allowlisted. Residual: a hostile channel
  message could coax a *wrong but in-scope* search into context. Accepted for an internal
  workspace; revisit if search scopes widen.
- **Memory poisoning / visibility.** Anyone in the channel can influence memory. Mitigation:
  facts are injected with provenance ("channel memory, may be stale"), modal shows a read-only
  list with per-row forget (small Phase F add-on if desired).
- **Engine misjudgment.** Verdict quality is prompt-tuning work (parity last-mile #6 absorbed
  into Phase F live testing). Fail-safe stays "ignore"; worst failure mode is silence, never spam
  — plus the hard throttle caps spam even if the prompt regresses.
- **action_token semantics.** Exact TTL/single-use behavior isn't fully documented; Phase B
  must verify live whether a token survives multi-round loops (fallback path already designed).
- **chat.startStream rate tier** undocumented — circuit breaker + legacy fallback already wrap it.
- **Debounce vs. Slack event ordering** — events per channel arrive ordered over one socket, but
  a reconnect can replay; dedup by ts in ChannelPulse handles it.
- **Memory (RAM) growth**: ChannelPulse is `O(channels × 30 × 300 chars)` — trivial.
- **Cost ceiling**: auto_respond is per-channel opt-in; a runaway channel costs ~$0.15/day in
  classifier calls (§4) before anyone notices; hard throttle caps the expensive main-model calls.
