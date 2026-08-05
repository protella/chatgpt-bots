# Single-Stream Channel Context — Implementation Spec

Status: **PLANNED, codex signed off — awaiting owner "go". No implementation has begun.**
Planning: 6 codex rounds (session `019fabf3-ef7a-7631-89e4-28a6dd0cc0ca`), 2026-07-28/29, informed
by direct interrogation of a live Claude Tag instance and its public docs. Codex sign-off: round 6.

## 0. Goal

Imitate Claude Tag's room model on our stateless stack: every channel turn — top-level or inside
any thread — is written against ONE ts-ordered view of the whole channel, so cross-thread
awareness and loop-closing emerge naturally. We adopt the *stream*, not the *statefulness*:
Slack already is the single stream; today's per-thread partition is a choice made at
context-rebuild time, and this spec changes that choice. No sessions, no in-memory transcript
authority, no spawned thread-sessions.

**Decisions already made by the owner (not up for relitigation):**
1. The binary wake gate stays as-is (generous, cheap pre-filter). In this design a gate sleep
   only defers content — the stream carries it into the next turn — it never hides it.
2. Stateless stays. Slack + DB remain the only durable truth; restart loses nothing.
3. Per-thread locks / parallel turns stay. We do NOT adopt Claude Tag's serial channel loop.
4. Cost is not a constraint. Optimize the cache within the best architecture; never shrink the
   design to save tokens.

## 1. Definitions

- **Stream**: the canonical rendered channel transcript — every admitted message in the channel
  (top-level and all threads), globally sorted by Slack `ts`, thread membership as labels.
- **H (high-watermark)**: the admission watermark — the newest admitted event ts at turn
  admission (thread-lock acquire + queue drain); never refreshed mid-turn. Fetched messages with
  `ts > H` are excluded from the render.
- **Pinned tuple**: everything rendering is a pure function of:
  `(team_id, channel_id, snapshot_id, H, normalized-fetch snapshot, sidecar versions, name-map
  snapshot, channel-config/tool-schema version, model/request-capability profile, serializer
  version, outbound-receipt exclusion set)`. Same tuple ⇒ byte-identical stream from any origin
  thread. The **normalized-fetch snapshot** (the deduped, normalized Slack messages actually
  fetched) is itself a pinned input: an inline Slack mutation (edit/reaction) between two fetches
  produces two different snapshots, so cross-origin byte-equality is defined over a shared
  snapshot, never across a mutation.
- **H and edits**: H pins from the **admission watermark** — the newest admitted event ts at
  lock-acquire + queue-drain — not from the trigger message's own ts. (An edit-triggered turn
  carries the ORIGINAL message ts; pinning H from it would exclude everything posted since.) The
  trigger/target message ts is tracked separately from H.
- **Pinned invariant — own output is never a trigger**: no message authored by our bot id ever
  starts a turn, is admitted to a gate cohort, or feeds the activity index as a wake source.
  Index ingestion runs before participation/listening filters but AFTER the own-message check.
- **Origin thread**: the thread whose lock the turn holds and where its reply lands by default.
- **Foreign target**: a message/thread other than the turn's trigger that the model chooses to
  act on (answer, react, post into).

## 2. The canonical stream (serializer contract)

- Roles: our finalized posts render as `assistant`; every other sender renders as `user`.
  Nothing else ever occupies the assistant role.
- Order: global numeric `ts` (Slack creation order). This is NOT event-delivery order; a delayed
  event can insert before already-rendered bytes — that is an accepted prefix invalidation, not
  a bug.
- Labels: every message carries a deterministic label — sender (stable Slack id + name from the
  pinned name map), sender type, message `ts`, and for thread replies the **root `ts` as the
  thread identity** (root-text snippets are display sugar only; identity is never prose).
- Mutations stay INLINE: edits, deletes, reactions, tool-provenance markers, completed analyses,
  and profile renames rewrite the affected bytes and intentionally invalidate the cached prefix
  from that point. No synthetic append events. Correctness first; cache is an optimization.
- Chrome exclusion: status cards, placeholders, progress/streaming intermediates and the ⚙️
  settings footer never enter the stream (structural provenance via outbound receipts, §5;
  the legacy text-heuristic filter is retained only for pre-epoch history).
- Attachments: canonical rendering uses ONE uniform fidelity for every thread — text markers
  (`[+1 file: report.pdf]`-style) with stable ids. Raw `input_image`/`input_file` parts never
  enter the stream (CDN expiry, size, irrelevance to other threads). High fidelity for the
  current turn rides the post-breakpoint supplement (§3).
- Foreign-thread artifacts: documents remain channel-actionable (existing `read_document`
  behavior). Images and containers are **awareness-only this phase**, marked as such in the
  marker text. Catalog widening is a named follow-up.
- Subtype admission, `thread_broadcast` dedup by `(team, channel, ts)`, escaping, and the label
  grammar are pinned by a `serializer_version` that participates in the cache tuple and in
  compaction snapshot identity.
- An end-of-stream marker (stable `input_text` block) closes the stream so the explicit cache
  breakpoint always has a deterministic, supported block — including for empty streams.

## 3. Request layout & caching

One canonical Responses-API layout for channel turns (both streaming paths must converge —
today the plain path prepends a developer item while the tool path uses top-level
`instructions`; that split ends):

1. **Invariant base instructions** — top-level `instructions`. Bot-version- and channel-stable.
2. **Tools** — part of the effective prefix. Tool sets and schemas must be a function of
   `(channel, channel config, bot version)` only (§3a).
3. **Canonical stream** — compaction summary as the FIRST stream item (`user` evidence), then
   canonical messages in original roles, then the end-of-stream marker.
4. **Explicit cache breakpoint** on the end-of-stream marker (GPT-5.6 only; 5.5 keeps implicit
   caching + `prompt_cache_retention: 24h`; the wrapper grows `prompt_cache_options` support).
5. **Post-breakpoint evidence** (`user` role, in this order): current-thread pre-boundary
   rehydration (labeled as exact expansion of summarized material); **current-thread**
   high-fidelity supplement — the trigger's raw image/file parts keyed to source ts, with the
   current thread's image/file/document CATALOGS supplying the remaining current-thread
   actionability; channel/workspace memory; channel topic/purpose, roster, requester profile
   fields, canvas/image/file catalogs; **requester custom instructions** (demoted from system
   prompt: "style, never policy" framing — user authority, not developer).
6. **Final developer suffix**: reserved standing channel policy (developer-voiced steering),
   structural settings (participation level, placement), trigger/destination/thread coordinates
   and trusted ids, time (minute precision), restraint + terminal-action contract paragraphs,
   capability state.

- `prompt_cache_key` = `(team_id, channel_id)` namespace. Provider guidance ~15 req/min per key:
  overflow means misses, never errors.
- Channel steering keeps ONE pinned structured snapshot per turn exposing `developer_policy`
  (step 6) and `user_facts` (step 5) separately — no more single flattened string. Gate and
  responder read the same snapshot version.
- Sidecar pinning is atomic: one DB read transaction collects versions + rows; retries reuse the
  same pins, never rebuild against newer data.

### 3a. Tool-schema stability

All per-turn/per-thread data leaves the schemas; executors keep (or gain) hard authorization:
- `react_to_message`: static description; emoji catalog/usage evidence moves to step 5; executor
  validation retained.
- `no_response_needed`: statically exposed on channel turns; the EXECUTOR enforces
  silence-capability (fails closed on owed-words turns); the tool loop honors terminal silence
  only after an authorized success.
- `set_reply_destination`: static schema; `TurnRuntime.select_destination` rejects illegal calls.
- `search_slack`: static schema; action token lives only in `ToolContext`; honest runtime failure.
- Canvas tools: static schemas; catalog to step 5; `delete_canvas` fails closed when sender
  classification is absent.
- Image tools: stable superset schema; saved defaults + model-legal options to the suffix;
  validation through `resolve_settings` + pinned allowlist. Catalog-empty no longer removes tools.
- `mount_file`: file catalog to step 5 evidence; runtime authorization retained.
- **Documented cache-fork exceptions (the only four)**: code-interpreter container id (inherently
  thread-scoped), MCP-failure exclusion retry, timeout tool-drop fallback, model fallback.

### 3b. Channel capability profile — ⚠️ OWNER DECISION, flagged

Channel turns resolve capability-affecting settings (model, web search, MCP set, image model,
tool exposure) from a **channel-owned profile** (channel settings → global defaults), never from
the requester. Requester cosmetic prefs ride the suffix. DMs keep per-user settings verbatim.
**OWNER-APPROVED 2026-07-29, conditional on the channel settings modal being updated in the same
work**: the modal (settings_modal.py) grows the channel capability profile controls (model, web,
MCP, image model) so the channel-owned values are visible and editable where the other channel
settings live — approval assumed this is included; it is in scope, not a follow-up.

## 4. History fetch & coverage (the honest horizon)

- **Live index**: DB-backed thread-activity index `(root_ts → last_observed_reply_ts,
  advisory_reply_count)` fed from Socket Mode BEFORE participation/listening/subtype filters but
  after the own-message trigger check (the old pulse ingestion hook is conditional on channel
  listening and is NOT an adequate feed). `advisory_reply_count` is persisted but advisory-only;
  the monotonic ts hint may cause an extra fetch but must never suppress one. Edits, reply
  deletions, and tombstoned roots update the index explicitly.
- **Coverage**: per-channel persisted `coverage_start_ts` + bootstrap status. A background
  bootstrap sweep builds the retained-root inventory and extends coverage backward. The stream
  framing DECLARES the horizon. Fail closed if coverage does not reach the compaction boundary.
  `is_limited=true` (Slack retention) updates the declared coverage state.
- **Cold rebuild**: page `conversations.history` back to the boundary (`oldest=boundary,
  inclusive=false`; `latest=H, inclusive=true`); parents' `latest_reply`/`reply_count` are
  positive hints only; fetch `conversations.replies` for every root the pages or the index says
  has post-boundary replies — **every replies fetch is likewise capped at `latest=H,
  inclusive=true`**. Pre-boundary roots with newer replies come from the index —
  `history(oldest=boundary)` can never surface them (the parent keeps its original ts).
- Pagination: follow cursors + `has_more`; any incomplete page ⇒ `HistoryFetchError`, turn fails
  closed — a partial stream is never returned as complete. Pages ≤200; bounded reply-fetch
  concurrency; Retry-After honored.
- Normalization: dedupe, then sort with the shared numeric-ts comparator (history arrives
  newest-first, replies oldest-first).
- The in-memory stream cache object is **discardable memoization**, never authority. Authority is
  Slack + the pinned DB snapshot/sidecars. ThreadState is demoted to locks/queues/jobs/
  containers/delivery state.

## 5. Outbound receipts (own-output finalization)

- Durable DB table: `(team, channel, message_ts, turn_id, state ∈ in_flight|finalized|chrome)`.
  Derived state — DB-eligible; Slack still holds the content. **`turn_id` encodes the owning
  session** (`{session_id}:{sequence}`), so dead-session reconciliation is "finalize every
  `in_flight` row whose session_id is not the current session". The grandfathering **feature
  epoch ts** is written once to a DB meta row at migration time and read from there — never
  hardcoded.
- **Inclusion rule** (closes the registration race): an own-bot message enters the stream ONLY
  with a `finalized` receipt. Legacy own-messages before the feature epoch ts are grandfathered.
  `chrome` rows are permanently excluded (status cards, placeholders, footer).
- Finalization happens after the LAST context-relevant Slack update (native streaming creates the
  message at `chat.startStream` — the stale-send lease committing is NOT finalization). Rolled
  multi-part replies and cross-thread posts finalize as a unit when the producing turn's
  conversational output is final.
- Startup reconciliation: rows owned by dead sessions finalize on boot (the partial in Slack IS
  the final content once the producer is gone).

## 6. Duplication policy & idempotency

**OWNER RULING 2026-07-29: no answer-claim protocol.** The originally specced channel-scoped
claim system (`claim_answer_target` tool, HELD/COMMITTED/ABANDONED states, claim epochs, buffered
prose, `claim_lost` terminal) is REMOVED. It guarded against our own parallel turns both
answering the same third message under full visibility; the owner accepts that residual —
duplicates are tolerable in dev (and multi-bot duplicates are explicitly fine: take the best of
both). If live testing shows the bot double-answering, a minimal guard may be revisited as a
follow-up; until then, `turn_outcome` telemetry (§10) records every cross-thread post so
duplication is observable, never silent.

What remains (and is the owner's intended mechanism — matching the Claude Tag behavior observed
live: react, notice the chain advanced, give up, un-react):
- `stale_send_guard` unchanged — a turn whose conversation advanced never creates its answer
  surface; the newer message drives the successor turn.
- The model may simply decide, from the current stream, that its contribution is no longer
  needed and end with `no_response_needed`.
- The 👀 work reaction keeps its current placement logic, and the existing settle behavior
  retracts it when the turn ends silent or suppressed — the visible "backed off" signal.
- Cross-thread posts use `post_to_thread` as it exists today; no declaration step, no binding.
- Background jobs: existing per-thread in-memory reservation stays AS-IS (the channel-wide DB
  upgrade is dropped with the claims).
- Canvas creation: existing process-lock + live list-before-create recheck; preserve the recheck
  on retries/ambiguous failures.
- Images: in-turn retry idempotency key `(turn_id, tool_call_id)` so a timeout-retried round
  never re-fires an already-launched generation (§12 decision 2). Full semantic image
  idempotency stays a follow-up.

## 7. Channel compaction snapshots

- New immutable snapshot store: immutable snapshot rows keyed by opaque `snapshot_id`, PLUS a
  separate **active-pointer row** per `(team, channel, serializer_version)` holding the current
  `snapshot_id`. Publication is a compare-and-swap on the pointer (`UPDATE … WHERE snapshot_id =
  :expected_previous`); a lost CAS discards the loser's snapshot. ONE channel-keyed coordinator
  (never per-thread cleanup tasks) triggers compaction.
- **Genesis**: with no snapshot yet, boundary = `coverage_start_ts` and the pointer holds a null
  sentinel; the render carries no summary block. Always require selected boundary ≤ H.
- Boundary is global channel ts, never per-thread. The summary is the FIRST canonical stream
  item, `user`-role evidence with explicit framing — never developer authority (unlike today's
  thread heads; DM thread heads keep their current shape).
- Summaries EXCLUDE volatile inputs (reactions, display names) so their mutation cannot
  invalidate a snapshot. They PRESERVE thread topology, open questions, addressee state,
  artifact references, and a **root-anchor map** (bounded root text/author per straddling
  thread) so a thread whose root predates the boundary keeps a deterministic referent.
- Invalidation: an edit/delete at-or-before the boundary invalidates the active generation;
  recompaction is lazy (next turn). If Slack retention no longer exposes the source span,
  degrade to retained-evidence-with-stale-marker — an honest note, never a silent lie and never
  a permanent fail-closed.
- Old snapshots stay readable while any turn/retry/detached job pins them; sweep on
  no-live-reference + retention.
- Late ambient artifacts: post-boundary completions are inline sidecar mutations on their source
  message. A late artifact whose source is at/before the boundary becomes a versioned user-role
  pre-boundary-artifact evidence block AFTER the breakpoint (keyed to source ts + snapshot id) —
  the current concatenate-onto-summary-head addenda mechanism does not port to channels
  (`thread_summary_addenda` remains untouched for DMs).

## 8. Retire / keep / adapt inventory

| Surface | Disposition |
|---|---|
| ChannelPulse content ring, envelope, exclusion, labels, backfill, artifact patching | **Retire** |
| ChannelPulse actor tail + `thread_has_other_bot` | **Extract** into an actor-only component with its own ingestion lifecycle (gate + 1:1 continuation block depend on it; gate is unchanged) |
| Channel narrative block in responder context | **Retire** (second lossy account of the stream) |
| Channel summary service | **Keep narrowly** for channel-join intro only |
| Channel people line / recent speakers | **Adapt**: derivable from stream; membership count stays suffix data |
| Thread participant roster / taggable speakers | **Adapt**: stable ids in canonical labels; bounded taggable roster in step 5 |
| Channel topic/purpose/settings/canvases in system prompt | **Move**: split `_build_channel_info` — user-controlled text to step 5, structural settings to step 6 |
| Channel steering | **Keep**, structured snapshot split (§3) |
| Gate source assembly, debounce cohort, gate prompt | **Keep unchanged** |
| `_merge_gate_cohort` for channel responder | **Retire** (stream already contains those messages); keep attachment-supplement duty |
| `CHANNEL_ACTIVITY`/`THREAD_ACTIVITY` restraint suffixes | **Adapt** (§9) |
| `no_response_needed` | **Keep** — more load-bearing than ever |
| ThreadState per-thread message storage + token accounting | **Demote**: locks/queues/jobs/containers/delivery only; token accounting covers instructions+tools+stream+supplement+suffix+output reserve |
| AssetLedger, CI containers | **Keep thread-scoped** |
| History tools (`fetch_channel_history`, `fetch_thread_messages`) | **Keep** (pre-coverage detail, other channels, explicit expansion) |
| Memory-extraction fallback | **Adapt**: bind to the trigger's actual exchange, never "latest assistant item in the stream" |
| CV7 telemetry | **Bump to CV8** (§10) |
| DMs — everything | **Unchanged verbatim** (branch before the new builder and tool materialization; no global conversion of catalog factories or terminal-tool exposure) |

## 9. Prompt & restraint changes

Restraint moves almost entirely to prompting — Claude Tag's own assessment: nothing structural
stops it replying to anything visible; what works is explicit **cost asymmetry** plus per-channel
calibration in memory.
- The retired ring framing's job transfers to the new suffix contract: the stream is the room,
  not an invitation — "don't jump into someone else's exchange" survives, now against full
  visibility (the F47 scar tissue must survive the rewrite; the scenario suite proves it).
- "Latest message" in both restraint suffixes must mean **the identified trigger in the
  identified thread**, never "globally last in the channel". Sticky hand-off stays thread-scoped.
- New: cross-thread conduct paragraph — closing a loop you were part of is legitimate (post
  once, in that thread); continuing strangers' exchanges is not. Includes the
  let-the-exchange-end guidance (react or stay silent on landed closers) — the previously tabled
  "last word" item lands here.

## 10. Telemetry — contract CV8 (CV9 and CV10 addenda below)

(Authoritative event list; §16 describes usage in the battery.)
- `visible_action` stays **gate-attempt-only** with its one-terminal-per-attempt invariant and
  its existing kind vocabulary — historical participation denominators stay valid.
- New **all-turn population**: `turn_start` / `turn_outcome` keyed by `turn_id` for EVERY channel
  responder turn (mentions and direct continuations included); `attempt_id` joins gated turns.
  `turn_outcome` records destination(s), including every cross-thread post target — this is how
  duplicate answers stay observable with no claim protocol (§6).
- New `stream_render` event: `turn_id`, channel, origin thread, trigger ts, `snapshot_id`/
  generation/boundary, H, `coverage_start_ts`, serializer + sidecar/name-snapshot versions,
  stream byte count, message count, SHA-256 over the exact canonical pre-breakpoint bytes.
- New `outbound_receipt` and `compaction_snapshot` (`read|publish|invalidate|stale_retained`)
  events; per-response model/fork-reason + cached-input token count.
- Ledger version bump `v7 → v8`; the analyzer and live-battery checks update together.

### CV9 addendum — stale reconsideration (2026-08-04)

`CONTRACT_VERSION` 8 → 9 (`participation_telemetry.py`; checker
`tools/participation_ledger_check.py`). Channel-turn population only — DMs stay excluded, and
`GATE_CONTRACT` does not move. **Null encoding:** unavailable optional fields are OMITTED,
matching the emitter's drop-None behavior; no JSON nulls anywhere in the grammar, and the
checker treats an absent optional field as "unavailable".

- **`stale_send`** gains one field, `turn_id` (both emitters), joining the row to the turn
  population; its cardinality generalizes from one-per-suppression to **one per suppression
  EVENT** — the initiating refusal, each per-pass re-race inside the reconsideration runner,
  and a post-run once-gate suppression each emit exactly one row. Single-owner rule: the runner
  emits for every suppression it handles and marks the exception `telemetry_recorded`; an
  unmarked suppression is emitted by the terminal catch. The checker tolerates DM `stale_send`
  rows that carry a `turn_id` with no channel `turn_start` to join. Everything else on the row
  (including the `scope[0]`-only scope field) is unchanged.
- **New `reconsider_start`** — one per reconsideration pass, emitted via the decision wrapper's
  `on_attempt_open` callback. Keys: `turn_id` (mandatory; the primary turn-population join),
  `channel_id`, `trigger_ts`, `attempt_id` (optional — ungated channel turns have none),
  `pass` (int, from 1), `scope` (the full three-part suppressing scope as a JSON list),
  `observed_latest_ts`, `model_attempt_seq` (optional — omitted when the attempt sink failed
  to open).
- **New `reconsider_outcome`** — at most one per runner invocation; exactly one on every
  non-cancelled path. Keys: `turn_id` (mandatory), `channel_id`, `trigger_ts`, `attempt_id`
  (optional), `outcome` ∈ {`posted_asis`, `posted_revised`, `skipped`, `fuse_dropped`,
  `error_dropped`, `cancelled`}, `passes` (int — the number of `reconsider_start` events the
  invocation emitted; a fuse drop records 5, a failure or cancellation the passes started by
  then), `forced` (bool, posted outcomes only), `error` (`error_dropped` only, one of the
  EIGHT §4f subtypes: `context_rebuild` / `model_failure` / `admission_overflow` /
  `delivery_failed` / `epoch_invalidated` / `guard_rearm_failed` / `request_build` /
  `delivery_exception`). A posted outcome asserts PHYSICAL Slack
  acceptance of the first surface, not finalized turn accounting.
- **`turn_outcome`** may carry a nested `reconsider` object —
  `{outcome, passes[, forced][, error]}`, `ReconsiderFacts.as_payload()` verbatim, with
  inapplicable keys omitted so no nested null survives — absent when no reconsideration ran.
  Its `destinations` contract gains one ruled exception: the zero-chunk truncation notice
  (Slack-accepted, turn-owned, never registered as a destination) is legitimately absent.
- **`model_response`** gains the `stale_reconsideration` fork reason: each reconsideration
  pass is a new `ModelAttempt` of the same turn.
- **Checker invariants**, all joined on `turn_id`: pass numbers contiguous from 1; a turn's
  `reconsider_start` count ≤ its `stale_send` count; ≤ 1 `reconsider_outcome` per turn; an
  outcome's `passes` EQUALS the number of `reconsider_start` rows joined to its turn — the
  field is the started-pass count, so a disagreement means one of the two is counting
  something else. The
  checker does NOT cross-join posted outcomes to `turn_outcome` kind or to F7 — that
  correspondence is unit/integration-mandated instead.

### CV10 addendum — edit_own_message (2026-08-04)

`CONTRACT_VERSION` 9 → 10 (`participation_telemetry.py`; checker
`tools/participation_ledger_check.py`; spec `Docs/specs/EDIT_OWN_MESSAGE.md` §7). Every CV9
field and invariant above is preserved unchanged; v9 rows in a mixed file are skipped by the
checker's existing older-contract rule, never graded by v10 rules. The null encoding is the
CV9 rule, restated because the new grammar depends on it: unavailable optional values are
OMITTED, never null.

- **`turn_outcome`** gains **`edits`** — ALWAYS present as a list in CV10, written even when
  empty (a turn that edited nothing and a turn whose edit records were lost are not the same
  fact). One entry per `TurnRuntime.edits` EditRecord, exact payload:
  `{"channel_id", "target_ts"[, "announcement_ts"],
  "state": "announcement_only"|"committed"[, "error"]}`. The entry grammar is CLOSED (those
  five keys and nothing else) and null-free: `announcement_ts` (the disclosure post's ts) and
  `error` (the exact post-announcement failure code) are the only optionals, omitted when
  unavailable. `state=announcement_only` means the disclosure landed and the `chat.update` did
  not; `committed` means both did. `reconsider` and `edits` may COEXIST on one row — a turn
  can be reconsidered and edit an earlier message.
- **Destination kinds** gain **`correction_announcement`**
  (`DEST_KIND_CORRECTION_ANNOUNCEMENT` in `turn_runtime.py`): the executor-synthesized
  disclosure post of an `edit_own_message` transaction, recorded as a committed destination of
  the turn that posted it.
- **Checker** (`tools/participation_ledger_check.py`): `edits` is mandatory on v10
  `turn_outcome` rows and must be a list; each entry is graded against the closed key grammar,
  the two-value `state` vocabulary, and the no-explicit-null rule. **Join invariant** (checked
  on the row, since both sides live on it): every entry's `announcement_ts`, when present,
  joins a COMMITTED `correction_announcement` destination in that turn's `destinations` — the
  disclosure is a real post the room saw, so an unjoined ts means one of the two records is
  wrong about what was delivered.

## 11. Config & pinned rules

**Pinned correctness rules (not config):** no "last N messages" stream — context is exactly
summary-through-boundary + every canonical event `(boundary, H]`; H pins at admission; token
accounting covers the full request; partial fetch ⇒ fail closed; gate cohort semantics unchanged;
attachment omissions leave explicit canonical markers; receipt state machine as specced;
serializer grammar versioned.

**Config with documented starting defaults** (all env-tunable; values below are the shipped
defaults, not laws): compaction trigger/target 0.80/0.70 of the model window (invariant
`target < trigger` with worst-case suffix+output headroom); reply-fetch concurrency 4; fetch
retry budget 3 attempts with Retry-After honored, 60s total per turn; history page size 200,
page-count safety ceiling 50/turn; bootstrap sweep depth: to Slack retention or 90 days,
whichever first; root-anchor text bound 240 chars; snapshot retention: last 3 generations or
7 days, whichever more; raw-attachment cap 10 images/turn (existing), catalog caps unchanged
from current config; summary output budget: ANALYSIS_* settings, ~2000-token target; gate
debounce 3s and drain batch unchanged; cache-metrics logging on.

## 12. Owner decisions (resolved 2026-07-29 unless noted)

1. **§3b channel capability profile** — APPROVED, conditional on settings-modal updates being in
   scope (they are; P4).
2. **§6 image idempotency** — decision pending owner's read of the explanation; default if
   unanswered: in-turn retry idempotency key `(turn_id, tool_call_id)` folded into P3 (small),
   full semantic image idempotency stays a follow-up.
3. **Phasing** — no cutover machinery (owner: pure dev, build direct); order as §13.
4. **Second test identity** — use Claude Tag as the third party (§16); no new token.

## 13. Phasing (proposed)

**OWNER RULING 2026-07-29: no cutover machinery.** This is pure dev — nothing is live, nothing
is lost if a build breaks. No feature flag, no dark-shipping, no parallel old/new path: build the
new architecture directly and let the dev bot run it as it lands. (Codex's duplicate-answer-race concern is
formally accepted as residual — §6.)

- **P1 — Foundations**: outbound receipts + reconciliation; thread-activity index + coverage
  bootstrap; snapshot store + coordinator; request-layout unification (instructions vs
  developer-item split ends); tool-schema staticization + executor authorization;
  `prompt_cache_options`/breakpoint support in the wrapper.
- **P2 — The stream**: serializer + fetch/coverage builder; channel turns switch to
  stream+supplement+suffix; ChannelPulse retired (actor tail extracted); prompt rework (§9); CV8.
- **P3 — Cross-thread action**: prompt work for cross-thread conduct (§9), `post_to_thread`
  exercised under full visibility, image in-turn retry key (§6). (Claims removed by owner — §6.)
- **P4 — Hardening**: compaction live; settings-modal capability profile (§3b); contamination
  hygiene tooling; full live battery.
  Foreign image/container catalogs are a **post-project follow-up** (§17). Each phase ends with
  codex diff review + capped unit suite + targeted live checks.

**Implementer style note (owner):** Opus 5 implementation agents must keep code comments TRIMMED —
comment only what genuinely needs explaining, at minimum length. No paragraph-length narration of
mechanics; this spec and the tests carry the rationale.

## 14. Unit-test plan

Policy: tests that encode the old architecture are **deleted and rewritten**, never contorted.

**Delete (and rewrite where noted):** `test_channel_pulse.py`; `test_channel_summary.py`;
`test_stateless_context.py` (surviving DM contract → new `test_dm_stateless_context.py`);
`test_taggable_speakers.py` (→ pinned roster/name-snapshot evidence tests);
`test_late_artifact_addenda.py` (DM addenda → fresh DM-only file; channel pre-boundary artifacts
→ new suite); `test_f23_post_to_thread.py` (→ foreign-target protocol shape);
`test_no_reply_tool.py` (schema-exposure assertions obsolete; terminal-loop cases →
`test_terminal_actions.py`); `test_thread_manager_advanced.py`.

**Adapt (highlights):** `test_thread_tail_context.py` → rename `test_actor_tail.py`, actor-only;
`test_wake_envelope.py` → metadata now asserted in the post-breakpoint developer suffix;
`test_channel_context.py` → info-cache kept, prompt-section tests become request-layout placement
tests; `test_message_timestamps.py` → channel determinism moves into the serializer suite;
gate suites keep gate behavior, add H/admission timing + CV8; `test_thread_manager.py` → strip
transcript/token/compaction authority; steering/memory/settings/prompt suites → structured
policy-as-developer / facts-as-user placement instead of flattened-bytes equality;
`test_sender_classification.py` → one canonical raw-Slack normalizer covering roots, replies,
edits, broadcasts; receipts/ack/reconcile suites → receipt transitions;
`test_stale_send_guard.py` unchanged in scope;
`test_retired_machinery.py` → invert the "pulse must remain" assertions, keep rich-gate
tripwires. Fixture-only cleanups: `test_footer_nonstreamed.py`, `test_image_transcode.py`,
`test_mute_enforcement_rewire.py`, `test_deep_research.py`, `test_research_build_phase.py`,
`test_stream_terminal_states.py`.

**Keep:** `test_wake_classifier.py` (binary-gate contract untouched); pure Slack rendering,
documents, images, MCP, unrelated DB suites; DM-only suites.

**New mandatory suites:**
1. `test_channel_stream_serializer.py` — **crown jewel**: same pinned tuple from two origin
   threads ⇒ byte-identical canonical stream + hash; ts ordering, labels, role mapping,
   broadcast dedupe, H exclusion, finalized-receipt inclusion, horizon framing, markers.
2. `test_channel_history_discovery.py` — mocked cursor pagination, root inventory, activity
   index, old-roots-with-new-replies, boundary/H inclusivity, fail-closed partial fetches.
3. `test_channel_coverage_bootstrap.py` — monotonic coverage, concurrent sweeps, restart,
   honest-horizon framing.
4. `test_channel_compaction_snapshots.py` — opaque ids, active-generation CAS, competing
   coordinators, pinned readers, root-anchor maps, invalidation, stale-retained, lease-aware GC.
5. `test_outbound_receipts.py` — register/finalize race, finalized-only inclusion, multipart,
   dead-session reconciliation, legacy epoch, chrome exclusion, DB-failure behavior.
6. `test_channel_request_layout.py` — exact role/order contract incl. breakpoint, evidence,
   suffix, custom-instruction demotion, DM shape unchanged.
7. `test_channel_cache_layout.py` — static schemas/profile, channel cache key, suffix variance
   without stream variance, exactly the four fork exceptions.
8. `test_channel_high_watermark.py` — admission-after-drain, no mid-turn refresh,
   arrival-exclusion/next-turn-inclusion.
10. `test_channel_roster_evidence.py`, `test_channel_preboundary_artifacts.py`,
    `test_foreign_target_tools.py`, `test_dm_stateless_context.py`.

Equality assertions compare the serializer's defined canonical bytes — never `repr`, token ids,
or the full request (which is intentionally origin-specific after the breakpoint).

## 15. Integration scenario harness

The old suite is dead (`participation_eval.py` deleted in 1ca7cb2; `participation_scenarios.py`
speaks the retired rich-gate API and isn't pytest-discovered). Replace with two tiers:

- **Tier 1 — wake**: drive the real `classify_wake` with production `SourceMessage` rendering
  and a pinned steering snapshot. Labels `must_wake` / `must_sleep` / `either`. False sleeps
  score heavier (the gate is generous by design): any `must_wake` miss is a hard fail;
  false-wake soft threshold ≤10%. No stream, no tools, no destination decisions.
- **Tier 2 — responder**: force admission, invoke the production request assembler with a
  synthetic pinned canonical stream, real tool schemas, in-memory Slack/effect sink. Grade
  observable outcomes only: `no_response_needed` silence / reaction-only / in-thread reply /
  channel reply / cross-thread post / detached effect / contract violation.
  Addressee hand-off, third-party-praise, rebuff, and continuation-bait cases live HERE (their
  prose tails become real timestamped, thread-labeled stream messages). Hosted web/MCP/CI
  disabled unless the scenario tests one. 3 trials per scenario; hard cases pass 3/3, soft 2/3;
  any hard-case regression against the recorded baseline blocks. **The per-scenario
  expected-outcome table ships in the scenario file and gets owner review** (§12).
- Optional tier 3 end-to-end (gate + responder) exists for smoke only and never grades
  restraint — a gate sleep would mask a responder-prompt failure.

## 16. Live-test battery & harness changes

**Telemetry prerequisites (CV8, ruled):** `visible_action` stays gate-attempt-only (historical
denominators preserved); NEW all-turn population `turn_start`/`turn_outcome` keyed by `turn_id`
for every channel responder turn (`attempt_id` joins gated ones). New events: `stream_render`
(snapshot id/generation/boundary, H, coverage start, serializer + sidecar/name versions, byte
count, message count, SHA-256 **over the exact canonical pre-breakpoint stream bytes**),
`outbound_receipt` transitions, `compaction_snapshot` (read|publish|invalidate|stale_retained),
and per-response model/fork-reason + cached-input token count (the cache PROOF — the hash proves
byte identity, usage proves reuse). The dev fence's `test_epoch_id` is stamped on every event
during a battery.

**Dev barriers (ruled):** a dev-only `dev_barriers` module, hard no-op unless
`DEV_TURN_BARRIERS` is set (empty in prod; unit test asserts no-op). Named barriers at exactly
three seams: post-admission (H/snapshot pinned), post-partial-Slack-post pre-finalization,
pre-resume-after-compaction. Harness controls via scratch-dir flag files.
Fixed sleeps and "long prompts" are forbidden as race inducers.

**Battery** (each case: setup → pass condition → ledger/Slack evidence):
1. Cross-thread awareness — nonce posted top-level after thread B starts; ask in B; reply
   carries nonce with `stream_render.H ≥ fact_ts` and NO history/search tool call.
2. Cross-thread action — target T in thread C, ask from A; the answer lands under C (not
   pasted into A); `turn_outcome` records the cross-thread destination.
3. Parallel-turn duplication probe — two overlapping turns that can both see unanswered T:
   record what happens. Duplicates are ACCEPTED (§6), so this case measures, never fails —
   its output is the data for revisiting a guard later.
4. Stream currency — barrier after admission, post nonce M, resume: turn excludes M
   (`H < M.ts`), next turn includes it.
5. In-flight exclusion — receipt-state evidence, not model self-report: B renders while A's
   receipt is in_flight (excluded); C after finalization includes A.
6. Compaction pinning — A pinned on S1, S2 publishes mid-flight, A logs S1, next turn S2.
7. Edit/delete invalidation — nonce edit/delete changes hash + explicit snapshot
   invalidate/republish or stale_retained for the pre-boundary case.
8. Full-visibility restraint — seeded foreign exchange + bait: responder ends in declared
   silence, zero Slack surface.
9. F54-era regressions — directed banter answered short, foreign banter untouched, thanks →
   reaction-only, value floor holds. Outcome-shaped assertions, never exact prose.
10. Crown-jewel live equality — **render-only probe** (ruled): dev-only command rebuilds the
    stream for two origin threads under one pinned tuple and compares hashes. No admission
    barrier; two NATURAL turns differ in H by construction and must not be compared.

**Contamination hygiene (ruled):** no artificial compaction reset (it launders contamination
into a summary). For the fixed channel `C0BKX77NU66`: a dev-only **context epoch fence** —
persisted `{channel, epoch_id, start_ts}` separate from production compaction; exclusive
battery lease + drain before fencing; in-epoch mode omits everything before `start_ts`
including replies to pre-fence roots; the deliberate horizon is declared in framing; channel
memory/steering/capability profile swapped to known fixtures; memory extraction + cross-epoch
job persistence disabled; crash-safe restore. A pre-provisioned fresh-channel pool is the
better future mechanism once authorization allows it.

**Harness accuracy fixes:**
- One canonical raw-Slack normalizer for ALL ingest paths (history roots, replies incl.
  separately-returned root, live events, `message_changed` both payloads, broadcasts pre-dedupe,
  snapshot anchors, receipt filtering, artifact attribution). All behavior branches on
  `sender_type`; raw `bot_id`/`app_id` is provenance only; own-bot check precedes the dev
  allowlist. This CLOSES the documented dev-only mention-cleaning gap
  (`is_bot = bool(bot_id)` bypassing the carve-out in rebuilt history).
- Second party for addressee scenarios (**owner-ruled**: no second user token exists): use the
  **Claude Tag bot in the test channel** as the third party. For scenarios that intend a bot
  actor, use it as-is; for scenarios that need a second HUMAN, add its bot_id to the
  `DEV_TREAT_BOT_IDS_AS_HUMAN` allowlist for the battery so it classifies as human end-to-end
  (dev-only; restore after). The harness prompts it in-thread to play the needed part.
- Seed posts assert both raw and normalized shape; all waits poll by ts/nonce + telemetry with
  deadlines (never fixed sleeps); results joined to one `session_start` build; battery waits for
  `coverage_start_ts` to reach the boundary before starting; search-token scenarios verify the
  token exists rather than assuming.

## 17. Explicitly out of scope

- Any change to the gate model or its prompt (owner decision #1, standing).
- DM behavior in any form.
- Conversational/steerable background jobs (separate feature; discussed, not specced here).
- Foreign-thread image/container actionability (post-project follow-up).
- Image-idempotency upgrade (post-project follow-up; codex dissent on record, §12).
- Prod deployment; v3 release mechanics.
