# Claude Tag Patterns — Implementation Spec (rev 3)

Four features borrowed from Anthropic's Claude Tag (Claude in Slack) architecture, adapted
to this bot's asyncio/Responses-API design. Researched 2026-07-10 (official docs + the
Deriv reverse-engineering teardown + Anthropic engineer comments on HN). Rev 2
incorporated a full Codex (gpt-5.6-sol) design review — notably: no pending-work
breadcrumbs in the transcript (F1), a shared image-delivery seam (F1/F4), no
system-prompt fork (F2), buffered output on terminal-contract turns (F2), and
summary-safe initiator tracking (F3). Rev 3 folds in the second-round review:
generation-ID-aware latch lifecycle, merge-preserving image upserts, checklist
filtered from history rebuilds (F1); post-delivery accounting, a single
`defer_visible_output` guard, request-config materialized once (F2). Codex verified
`AsyncOpenAI.with_options(timeout=...)` works on the pinned `openai==2.45.0`.
Round-2 verdicts: F1 ready-with-fixes, F2 ready-with-fixes (fixes folded in below).

Implementation order: **F4 → F1 → F2 → F3** (F1 consumes F4's checklist and the shared
delivery seam; F3 rides the suffix plumbing F1/F2 touch). The work lands as four
independently reviewable change sets; commits happen only when the user says so
(CLAUDE.md git rule).

Shared invariants (apply to every feature):
- Slack stays the only transcript. **Never represent pending/in-flight work as a
  transcript message** — pending state lives in registries and the volatile suffix.
- Deterministic serialization for anything entering model context (prompt-cache hygiene):
  volatile per-turn data rides the developer *suffix* message (`_build_suffix_context`,
  `handlers/text.py:158-163`), never the system prompt and never history. The system
  prompt stays byte-identical across prompted/unprompted turns of the same thread.
- All free-text interpolated into the suffix (usernames, engine reasons, prompt
  summaries) is newline/control-escaped and length-capped, and the block is labeled
  informational metadata, not instructions.
- All new work is asyncio on the single event loop; background fan-out only via
  `_schedule_async_call` (`message_processor/utilities.py:1104`).
- Every new behavior gets an env flag: `config.py` field + `.env.example` entry +
  config unit test, current behavior as the off-path.
- Unit tests in `tests/unit/`, mocked clients, no live APIs.

---

## F4. Edit-in-place progress checklist

**Claude Tag pattern:** long tasks post one checklist message and edit it in place as
steps complete ("✓ Read the repo structure… ✓ Drew the diagram… ✓ Rendered to PNG").

**Current state:** `_update_status` (`message_processor/utilities.py:1176-1205`) edits a
single status line in place, *replacing* the previous step text. Callers (illustrative,
not exhaustive): `handlers/image_gen.py:104/191/194`, `image_edit.py:124/128/141/247/300`,
`vision.py:189/212/716`. `_start_progress_updater_async` (`utilities.py:1207`) rotates
"still working" strings on a timer by replacing the whole message (`utilities.py:1241`).
Step history is lost as each edit overwrites the last.

**Design:** an accumulating-checklist helper rendering completed steps with ✓ and the
active step with the loader emoji, editing one Slack message in place.

New module `message_processor/progress.py`:

```python
class ProgressChecklist:
    def __init__(self, client, channel_id: str, thread_id: Optional[str],
                 message_id: Optional[str] = None,
                 min_edit_interval: float = 0.8): ...

    async def step(self, active_text: str, done_text: Optional[str] = None) -> None: ...
    async def complete(self, final_text: Optional[str] = None,
                       delete_after: Optional[float] = None) -> None: ...
    async def fail(self, note: str) -> None: ...

    @property
    def message_id(self) -> Optional[str]: ...
    @property
    def surface(self) -> str:  # "message" | "assistant_status" | "none"
```

Rendering: `✓ Enhanced prompt\n✓ Generated image\n{circle_loader_emoji} Uploading…`.
`step()` marks the previous step done (`done_text`, defaulting to `active_text` minus a
trailing "…"), appends the new active line, edits the message (creating it on first call
via `client.send_thinking_indicator` when `message_id is None`).

Behavioral rules (Codex findings 2-4 incorporated):
- **Concurrency & coalescing:** all methods serialize on an internal `asyncio.Lock`.
  Non-terminal edits inside `min_edit_interval` coalesce into one *scheduled* flush (at
  most one pending), so intermediate states may skip but the latest state always renders.
  Terminal methods (`complete`/`fail`) always flush, awaiting the remaining interval if
  needed. Terminal state is idempotent and sticky: after `complete`/`fail`, further
  `step` calls no-op (logged at debug).
- **Status-only surfaces** (`send_thinking_indicator` returns `None` wherever Slack's
  assistant status succeeds — channels too, not just DMs; `messaging.py:416`): degrade to
  `client.set_assistant_status` with only the active step's text. `complete`/`fail` on
  this surface clear the composer status (`clear_assistant_status`) — the checklist owns
  that clear; callers on the status-only surface must not also clear it.
- **Failure handling:** `client.update_message` returning `False` (the client swallows
  Slack errors into a `False` return, `messaging.py:456-470`) counts as a failed edit —
  log at debug, keep state, try again on the next flush. Nothing raises to the caller.
- **Rotator collision (blocker fix):** `_start_progress_updater_async` replaces the whole
  message and would erase the checklist. Whenever a `ProgressChecklist` owns a message,
  the rotator is **not started** for it. (Folding rotation into the checklist is a
  possible follow-up, not in scope.)
- `fail()` keeps the message visible (✗ + note). `complete(delete_after=4)` preserves
  today's "status vanishes ~4s after upload" UX where callers want it.

**Integration in this change set:** the image-generation and image-edit pipelines adopt
the checklist for their status message (enhance → generate/edit → upload). Because the
upload step happens in `main.py`'s image branch today, this lands together with the
**shared delivery seam** (see F1, `publish_image`) so one component owns the
upload-step transition on both the sync and background paths. Other `_update_status`
callers are untouched.

**Config:** `ENABLE_PROGRESS_CHECKLIST` (default `true`). Off → today's single-line
`_update_status` path.

**Tests** (`tests/unit/test_progress_checklist.py`): step accumulation/rendering;
done-text derivation; first-call message creation; status-only surface degradation +
terminal status clear; fail keeps message; complete+delete_after deletes; coalescing
(final update inside the min interval still lands); concurrent step/complete/fail;
cancellation during delete_after; false-returning client methods; terminal idempotency;
rotator not started when checklist active.

---

## F1. Background image generation (release the thread lock)

**Claude Tag pattern:** the main loop never runs heavy work; long jobs run in background
workers so new messages keep getting normal turns, and results post when ready.

**Current state:** `process_message` holds the per-thread asyncio lock
(`message_processor/base.py:90-94`, released in `finally` at `base.py:1026`) for the whole
image pipeline: `_handle_image_generation` (`handlers/image_gen.py:14`) awaits
`openai_client.generate_image` → `_safe_api_call` (`openai_client/base.py:475`) under
`asyncio.wait_for` with the `image_generation` operation timeout
(`openai_client/base.py:64-91`) = `config.api_timeout_read` (default 180s) — **inside the
lock**. Messages arriving meanwhile go to the Phase Q pending queue (`base.py:99-121`,
drained `base.py:1086-1154`): a multi-minute generation blocks all conversation in that
thread. Breadcrumbs are appended only *after* successful generation (`image_gen.py:231+`).
Upload happens post-lock in `main.py:306-365` bridged by the upload latch
(`thread_manager.py:465-487`), which has a TOCTOU gap: the lock releases at
`base.py:1026` but `mark_upload_started` only runs when `main.py` reaches the image
branch, so a fast "edit it" can slip between (Codex finding 11).

**Design:** split the pipeline at the `generate_image` call. Everything fast
(enhancement stream, status/checklist setup) stays inline; the slow call plus delivery
detaches into a background job; the turn returns immediately and the lock releases.
**No transcript writes for pending work** — the job registry + volatile suffix carry the
in-flight state; Slack is rebuilt as the source of truth after delivery.

### Dedicated image timeout (user request 2026-07-10)
Image generation can legitimately run past 3 minutes. New config `api_timeout_image`
(env `API_TIMEOUT_IMAGE`, default **300**) used by `_get_operation_timeout` for
`image_generation` and `image_edit` (vision stays on `api_timeout_read`). **The outer
`asyncio.wait_for` alone is not enough**: `AsyncOpenAI` is constructed with
`timeout=config.api_timeout_read` (`openai_client/base.py:41`), so the SDK would abort at
180s regardless (Codex finding 9). Image calls therefore pass a per-request timeout —
`self.client.with_options(timeout=config.api_timeout_image).images.generate(...)` (same
for `images.edit`). Reconcile `.env.example:192`'s stale recommendation of a 300s global
read timeout while touching it.

### Shared delivery seam: `publish_image(...)` (Codex finding 13)
New helper owned by the processor (e.g. `message_processor/image_delivery.py`):

```python
async def publish_image(*, client, channel_id, thread_id, thread_key, image_data,
                        checklist: Optional[ProgressChecklist],
                        generation_id: Optional[str],  # None on the legacy sync path
                        prompt: str, db, thread_manager,
                        unprompted: bool) -> Optional[str]:  # returns file_url or None
```

One place for: checklist "Uploading…" transition, `client.send_image`, **falsey-URL =
failure** (`send_image` swallows Slack errors into `None`, `messaging.py:390-414`),
direct `db.save_image_metadata_async` persistence (prompt + generation_id + URL — DB
persistence must never depend on finding a mutable in-memory breadcrumb; Codex finding
10), asset-ledger update, upload-latch release, checklist completion/failure, and
participation accounting (`record_bot_reply` only on a real posted image when
`unprompted`). Both the config-off sync path (`main.py` image branch, refactored to call
this) and the background job use it.

**Latch TOCTOU fix (both paths):** `mark_upload_started` moves into `process_message` —
registered *before* the lock releases (before returning the `image`/`background`
response), not in `main.py`. **Latch lifecycle (round-2 blocker 1):** the latch is
generation-ID-aware and released idempotently in the background job's *outer* `finally`
— covering failure, moderation, cancellation, timeout, and scheduling failure (if
`_schedule_async_call` itself fails, release inline). A watchdog-cleared stale job must
never release a newer job's latch: release is conditional on the registered
generation_id, same rule as `finish_generation`.

**Merge-preserving image persistence (round-2 blocker 2):** `publish_image`'s direct
`save_image_metadata_async` write can be destroyed afterward: `AssetLedger.add_image`
(`thread_manager.py:226`) issues its own persisting upsert, and the post-refresh Slack
rebuild saves the uploaded file with an empty caption over the same URL
(`thread_management.py:982`), and the DB upsert is `INSERT OR REPLACE`
(`database.py:2079`) — erasing prompt/type/generation_id. Fixes: (a) a non-persisting
ledger update for the background path (in-memory entry only; the DB row is
`publish_image`'s job), and (b) the image-metadata upsert becomes merge-preserving on
URL conflict (`ON CONFLICT ... DO UPDATE` keeping existing non-empty prompt/analysis/
type/generation_id over incoming empties) — this protects the sync path too.

**Checklist must never enter model history (round-2 blocker 3):** if `needs_refresh`
fires mid-generation (e.g. Phase-Q queue overflow), the rebuild runs while the checklist
message is still visible in Slack, and its `✓ …` rendering doesn't match the existing
transient-status filter — it would be replayed as an assistant turn. Fix: tag checklist
messages with the repo's existing UI-marker mechanism (`message_markers.py`) so history
reconstruction filters them, exactly like other transient status messages. Test: force
a refresh+rebuild while a generation is in flight.

### Inline phase (inside the lock, in `_handle_image_generation`)
1. Enhancement runs as today (streams into the thinking message), **with one regression
   fix (found live 2026-07-10):** every enhanced-prompt write is keyed on `thinking_id`,
   but since the native-status refactor `send_thinking_indicator` returns `None`
   wherever `assistant.threads.setStatus` succeeds — nearly all surfaces — so the
   `*Enhanced Prompt:* ✨ _…_` display silently vanished. Fix: when `thinking_id is
   None`, post the enhanced prompt as its **own new message** (created on the first
   streamed chunk, then edited as usual) and hand that id to the existing
   `prompt_message_id` "don't touch this message again" logic. Check the image-edit
   path for the same keyed-on-`thinking_id` pattern and apply the same fallback.
2. Append **only the user message** to thread state (as today's success path does for
   the prompt). No assistant breadcrumb — the Responses payload strips metadata
   (`responses.py:388`) so a pending breadcrumb would read as a completed assistant turn
   (Codex findings 6-7) and the no-DB-image rebuild check (`thread_management.py:828`,
   clear at `:844`) would wipe it anyway.
3. Mint `generation_id = uuid4().hex[:12]`; register the job:
   `thread_manager.register_generation(thread_key, generation_id, prompt_summary, task=None)`
   — registry entry `{generation_id, task, started_at, prompt_summary}`.
   `finish_generation(thread_key, generation_id)` is **ID-conditional** (a stale job can
   never clear a newer one; Codex finding 14). Advisory peek
   `generation_in_flight(thread_key) -> Optional[dict]`. Watchdog: entries older than
   `api_timeout_image + 30s` are force-cleared and logged.
4. Build the `ProgressChecklist` (F4) on the generating-status message; do not start the
   legacy rotator.
5. Schedule `_finish_image_generation_background(...)` via `_schedule_async_call`; store
   the task handle in the registry entry (for shutdown).
6. Return `Response(type="background", metadata={"generation_id": ...,
   "background_owns_status": True})`.

### Background job `_finish_image_generation_background`
1. `await openai_client.generate_image(...)` (per-request 300s timeout as above).
2. Checklist step → generate done; `publish_image(...)` handles upload, DB persistence,
   ledger, accounting, checklist completion (`delete_after=4`).
3. `thread_manager.mark_needs_refresh(thread_key)` so the **next turn rebuilds the
   transcript from Slack**, which now contains the enhanced-prompt message + posted
   image — the same recovery seam Phase Q already uses (`thread_manager.py:453-463`).
4. Errors:
   - moderation-blocked (same string sniff as `image_gen.py:146`): checklist deleted,
     friendly text posted via `client.send_message`.
   - upload returned `None` or DB write failed after upload: `checklist.fail(...)`,
     friendly error via `client.handle_error`, log with stack (distinguish the two in
     logs; a posted-but-unpersisted image is recoverable via refresh).
   - other exceptions: `checklist.fail("Image generation failed")` + `handle_error`.
   - **every path** (success, failure, cancellation): clear assistant status if the
     progress surface was status-only, and `finally:` ID-conditional
     `finish_generation` + `mark_needs_refresh`.

### Turn-level integration
- `handle_message` (`main.py:274`): new `response.type == "background"` branch — a no-op
  like `queued`: no thinking-indicator delete, no footer, no upload branch, no
  `record_bot_reply` (the job accounts on delivery). The `finally` assistant-status
  clear (`main.py:395-407`) is **skipped when
  `response.metadata.get("background_owns_status")`** — otherwise status-only progress
  vanishes the moment the turn returns (Codex finding 8, blocker).
- **Follow-up turns during generation** (the point of the feature):
  - Volatile suffix line while `generation_in_flight(thread_key)`:
    `[An image for "<escaped prompt summary>" is currently being generated in this
    thread and will be posted automatically when ready. Don't claim it is done and
    don't start another image unless asked.]`
  - Intent routing (`base.py:611-640`): image intents (`new_image` / `edit_image` /
    `ambiguous_image`) while a generation is in flight get an **intentional rejection**:
    a short friendly `Response(type="text")` ("Still working on the previous image —
    ask me again once it lands."). One in-flight generation per thread. (A one-slot
    auto-redispatch queue was considered — Codex finding 12 — and deferred as follow-up;
    the rejection window only spans the generation itself, and conversational turns flow
    normally with the suffix note.)
  - The existing `wait_for_uploads` latch keeps covering the post-generation upload
    window on the edit path.
- **Shutdown:** processor cleanup cancels-and-awaits registered generation tasks (with a
  short timeout) *before* the Slack client stops (today Slack stops first,
  `main.py:556`, and cleanup never awaits scheduled tasks — Codex finding 14), and
  clears their progress UI best-effort.

**Config:** `ENABLE_BACKGROUND_IMAGE_GEN` (default `true`). Off → today's inline
behavior, except the sync path also goes through `publish_image` and the relocated
latch (bug fixes apply to both paths). Scope: **new-image generation only**;
`_handle_image_edit` / `_handle_image_modification` stay synchronous this phase.

**Tests** (`tests/unit/test_background_image_gen.py`): lock released before generation
completes — second message sent *after* background scheduling gets a live turn (during
enhancement Phase Q queueing is still expected); image posts with DB row written from
the job (no dependence on warm breadcrumbs); follow-up mid-generation does NOT trip the
no-DB-image rebuild wipe; status-only progress survives the turn returning
(`background_owns_status`); per-request SDK timeout actually exceeds
`API_TIMEOUT_READ`; upload `None` → failure surfaced, registry cleared; DB failure
after successful upload; moderation path; image-intent rejection while in flight;
suffix note present during flight, absent after; stale watchdog clear vs newer job
(ID-conditional finish); shutdown cancels/awaits jobs; two threads generating
concurrently don't cross-talk; config-off path == today + relocated latch closes the
TOCTOU gap; model payload contains no fake completed assistant turn mid-flight.

---

## F2. Explicit no-reply outcome (terminal-action contract, phase 1)

**Claude Tag pattern:** a turn cannot end ambiguously — the model must end with an
explicit terminal Slack action; silence is an explicit `no_reply_needed` tool call
enforced by a stop-hook. (Full reply-gate enforcement is out of scope; this phase makes
*silence* explicit and machine-readable. Ordinary text remains an implicit reply
action — hence the honest feature name; Codex finding 17.)

**Current state:** for unprompted channel messages the utility-model gate
(`participation.py:120-160` → `classify_participation`, `responses.py:929-1042`) decides
respond/react/ignore/backoff *before* the main model runs. On `respond` the main model
always produces a posted reply — its only opt-out is the prompt convention "call
react_to_message and return COMPLETELY EMPTY text" (`prompts.py:77`), and empty text is
silently treated as reaction-only (`main.py:281-283`), indistinguishable from a glitch.
Accounting bug: `main.py:262-264` counts any `metadata["streamed"]` response as posted
even with empty content, so a streamed reaction-only turn already burns the hourly
unprompted quota (Codex finding 18).

**Design:** an explicit `no_response_needed` terminal tool on unprompted turns, buffered
output so partial text can never post ahead of the verdict, and honest posted-accounting.

### Tool
Registered in `SlackBot._build_tool_registry` (`slack_client/base.py:76-93`):

```json
{
  "type": "function",
  "name": "no_response_needed",
  "description": "End this turn without posting anything. Call this when, after seeing the full conversation, you have nothing useful to add — the message wasn't really for you, someone else already answered, or silence is the socially right move. You may add an emoji reaction (react_to_message) in the same round; call this instead of replying, never after writing a reply.",
  "parameters": {
    "type": "object",
    "properties": {
      "reason": {"type": "string", "description": "One short sentence: why silence is right."}
    },
    "required": ["reason"]
  }
}
```

### Gating — no system-prompt fork (Codex finding 15, blocker)
The system prompt stays byte-identical. Per-request exposure:
- The text handler materializes the request's tool exposure **once, up front** (round-2
  should-fix 6): build a copied config dict with `_unprompted_turn=True` when
  `message.metadata.get("participation_check") is True` (never mutate the shared
  thread_config), resolve `registry.schemas(request_config)` from it, and derive
  `no_reply_tool_available` from that exact schema set. That one flag drives the
  `has_tools` precheck (`text.py:21` seam), the tools array, and the suffix paragraph —
  for both the streaming and non-streaming attempts — so exposure and instruction can
  never disagree. The tool's `enabled` gate reads `_unprompted_turn` AND
  `config.enable_no_reply_tool`.
- The behavioral instruction rides the **volatile developer suffix** (same slot as F1's
  in-flight note), only on unprompted turns:

  > [You joined this conversation uninvited. End your turn with exactly one of: a normal
  > reply, a reaction (react_to_message with empty text), or a no_response_needed call.
  > If you have nothing genuinely useful to add, prefer no_response_needed over filler.]

- `LOCAL_TOOLS_GUIDANCE` is unchanged (no global bullet advertising a tool that
  prompted/config-off turns can't see).
- **Timeout-retry consistency** (Codex finding 19): retries that disable the local tool
  registry (`text.py:182`) must also drop the suffix paragraph — both key off the same
  per-attempt "no-reply tool available" flag.

### Loop semantics
In the tool loop (`openai_client/api/tool_loop.py`), `no_response_needed` is terminal:
- Executor returns `{"ok": true}`; the loop stops — no feedback round.
- In the same round, **only sibling `react_to_message` calls execute**; other side-effect
  calls (memory writes etc.) are suppressed with a logged skip — `dispatch_all` runs
  rounds concurrently today (`tool_registry.py:97-101`), so the loop filters the round's
  call list *before* dispatch when it contains `no_response_needed` (Codex finding 17).
- Handler surfaces `terminal_action="no_reply"` + reason in response metadata and
  returns `Response(type="text", content="", metadata={..., "posted": False})`.

### ~~Buffered output on unprompted turns~~ SUPERSEDED 2026-07-10 (user decision)
The route-to-non-streaming design below shipped in 3f7a4ff but was reverted by user
direction: unprompted replies must stream like prompted ones. Replacement (implemented
with the F5/F6 change set): unprompted turns stream natively; `no_response_needed` is
honored ONLY while no visible text has been committed; a call arriving after committed
text is invalid — an error is fed back through the tool loop and the model must finish
the reply into the same streamed message (WARNING logged). Original rationale kept
below for history.

### Buffered output on unprompted turns (Codex finding 16, blocker)
Streaming can forward preamble text before the function-call item arrives
(`responses.py:705`; native stream starts on first chunk, `text.py:724`) — "visible text
wins" could post half a sentence. Instead: **unprompted turns never start the native
stream and never edit visible text mid-turn**. Output accumulates internally
(model-side streaming unchanged) and posts once, complete, at turn end — unless the
terminal outcome is `no_reply`, in which case nothing posts. Unprompted replies are
short; the UX cost is minimal and the contract is airtight. Prompted turns keep today's
streaming behavior untouched.

**One guard, every escape hatch (round-2 blocker 5):** suppressing native-stream startup
alone is insufficient — legacy callback edits, the final-correction pass, direct final
posting, stream-error cleanup (which today exposes buffered partial text, `text.py:1348`
area), and the empty-response apology (`text.py:1452` area) can all publish before the
terminal outcome is known. A single turn-level `defer_visible_output` flag guards native
coordinator creation, all callback writes, exception cleanup, and the empty fallback;
the terminal/empty outcome is decided *before* any assistant-state append or memory
cleanup. Required test: partial stream followed by a mid-stream error must not post the
partial text on a deferred turn.

### No-reply cleanup (Codex finding 19)
The `no_reply` outcome must: delete the thinking placeholder / clear assistant status,
post no footer, append no empty assistant turn to thread state, and skip post-response
memory extraction. (Reaction-only empty-text turns get the same cleanup path — shared
helper.)

### Accounting (Codex finding 18 + round-2 blocker 4)
Handlers set `metadata["posted"]` explicitly (true only when visible content actually
went out). For the non-streaming path the handler can't know — `main.py` accounts at
`main.py:262` *before* it sends the content — so `record_bot_reply` **moves to after
delivery** and derives `posted` from the actual outcome: the `send_message` result, the
final legacy update, or native finalize. The gate and the empty-text branch key off
`posted` / `terminal_action`:
- `terminal_action == "no_reply"` → INFO log with reason, no post, no quota burn.
- bare empty text without the tool → WARNING (contract violation), fail-safe silence,
  no quota burn. No re-prompt loop this phase.
This also fixes the pre-existing streamed-reaction-only quota burn.

**Config:** `ENABLE_NO_REPLY_TOOL` (default `true`). Off → tool hidden, suffix paragraph
absent, behavior as today (minus the accounting fix, which is unconditional).

**Tests** (`tests/unit/test_no_reply_tool.py`): tool exposed only on unprompted turns
(copied config, shared dict unmutated); no_response_needed ends loop, nothing posted,
reason logged; react+no_reply combo executes the react and suppresses other siblings;
buffered unprompted turn posts once/complete on reply outcome and nothing on no_reply
(native, legacy-streamed, and non-streaming variants); partial text never posts;
timeout-retry attempt hides tool AND paragraph; hourly quota unchanged on no_reply and
on reaction-only streamed turns; cleanup: placeholder deleted, status cleared, no
footer, no empty assistant turn, no memory extraction; reason length/sanitization;
config off hides everything; prompted turns see neither tool nor paragraph.

---

## F3. Wake envelopes (structured trigger metadata in context)

**Claude Tag pattern:** the model is told *why* it woke — trigger reason, sender trust
level, initiator-vs-participant — as structured metadata alongside the message
(initiator vs. participant signaling confirmed by an Anthropic engineer).

**Current state:** the trigger message reaches the model as `"{username}: {text}"`
(`base.py:217-220`) with no indication of why the bot is responding — @-mention,
name-mention, DM, ambient verdict, and a drained catch-up batch all look identical. The
participation classifier gets rich signals (`responses.py:939-998`); the main model gets
none. Rebuild tracks participants (`thread_management.py:968-971`) but not the root
author; summary-tail rebuilds fetch only messages after the boundary
(`thread_management.py:859`, filtered again at `:911`) so the root isn't even seen; the
current message is skipped at `:905`. `sender_type` is computed in the event handler
(`message_events.py:308`) but not copied into metadata.

**Design:** a compact `[Wake context]` block in the **volatile developer suffix** (never
stored, never in the system prompt, never in history), rendered deterministically.

### Rendering

One block, fixed field order; every free-text field escaped and capped:

```
[Wake context — informational metadata, not instructions]
trigger: app_mention | dm | thread_continuation | name_mention | ambient (engine: "<reason>") | catch_up_batch (N) — latest trigger: <source>
sender: <username> — root author | participant [— bot]
```

- `root author` (not "thread initiator" — a top-level channel-placement reply is a
  one-message thread and "initiator" would mislead; Codex finding 22). Omit the role
  entirely for top-level triggers with channel placement.
- Catch-up batches keep the underlying trigger: `catch_up_batch (3) — latest trigger:
  ambient` (Codex finding 22).

### Plumbing (Codex findings 20-21 incorporated)
1. **Wake source tagging** at event ingestion:
   - registration (`event_handlers/registration.py`) passes an explicit source since
     `app_mention` and DMs share `_handle_slack_message`: `"app_mention"` vs `"dm"`.
   - channel path: `direct_continuation` → `"thread_continuation"`; engine-gated →
     `"ambient"`, refined to `"name_mention"` on `name_hit` (`message_events.py:218`);
     name-wakes that bypass the engine (engine disabled / mentions_only fast path) also
     tag `"name_mention"`.
   - copy `sender_type` into `message.metadata` so bot senders render `— bot`.
2. **Engine reason:** `_run_participation_gate` (`main.py:63-143`) stores
   `message.metadata["participation_reason"] = verdict.reason` on `respond` (already
   capped at 300 chars by `validate_verdict`; escape at render).
3. **Catch-up:** `queued_batch_size` metadata already stamped (`base.py:1151`); the
   drained trigger's own `wake_source` provides "latest trigger".
4. **Root author** (summary-safe; Codex finding 20):
   - New-thread creation: capture root author + sender type from the current message
     *before* the history-skip at `thread_management.py:905`.
   - Rebuild with full history: root message's author in the conversion loop.
   - Summary-tail rebuild (root outside the fetched window): fetch the root message
     explicitly (a `limit=1` `conversations.replies` page without `oldest` returns the
     root) — one extra API call only when the initiator is unknown; cache on
     `thread_state.root_author = (user_id, sender_type)`.
5. **Builder:** `_build_wake_envelope(message, thread_state) -> str` in
   `message_processor/utilities.py`, called from `_build_suffix_context` (both the
   streaming and non-streaming payload assemblies); returns `""` when metadata is
   missing (CLI platform unaffected). F1's in-flight note and F2's unprompted paragraph
   render in this same suffix block, fixed order: wake context → in-flight note →
   contract paragraph.

Scope: text-handler turns only (the conversational path where addressee/interjection
judgment matters). Image/vision handlers don't build the suffix today and stay as-is.

**Config:** `ENABLE_WAKE_ENVELOPE` (default `true`).

**Tests** (`tests/unit/test_wake_envelope.py`): each trigger enum from metadata;
app_mention vs DM tagging via registration; engine-disabled name wake; root author vs
participant vs bot sender (self-bot and other-bot roots); summary-boundary rebuild
fetches root; new-thread root captured from current message; top-level channel placement
omits role; mixed-source catch-up batches render latest trigger; escaping of
username/reason (newlines, brackets); empty string on missing metadata; envelope present
in both streaming and non-streaming payload assembly, absent from system prompt and
thread state; config off.

---

## F5. Thread-tail context for the participation classifier (post-rollout fix, 2026-07-10)

**Live failure:** in #chatgpt-bot-test the gate judged "those are a button that open a
model. are you not able to see that?" (a reply continuing a Peter↔Claude exchange) as
`respond` — reason *"Peter is directly asking the assistant about what it can see in the
thread"* — because the classifier had no usable view of the exchange. Root causes:
(a) its only conversational evidence is the channel-pulse envelope — a channel-wide ring
buffer (`main.py:86`, `exclude_thread_ts=None`) whose entries truncate at 300 chars
(`channel_pulse.py:25`) head-first, so the tail of Claude's long message (the part that
established "you" = Claude) was cut; (b) there is no thread-scoped context at all, so the
prompt rule "'You' belongs to whoever the sender has been talking to"
(`prompts.py:47`) has nothing to bind to.

**Design (rev 2 — event-fed per-thread tail cache, after Codex review round 1):**
The original three-source fallback (warm state → pulse filter → replies fetch) had two
blockers: the judged message is itself recorded into the pulse before the gate runs, so
the "empty → fetch" trigger can never fire; and a single `conversations.replies` page
returns the OLDEST messages, not the newest N. Warm state also lacks sender provenance
(other bots are stored as bare `role=user`). Instead, the pulse — which already receives
every channel message event, including ones the gate ignores and the bot's own —
maintains the tail directly:

1. **Per-thread tail ring in ChannelPulse.** `record(...)` additionally appends to
   `_thread_tails[channel_id][thread_root_ts]`: a `deque(maxlen=PARTICIPATION_THREAD_TAIL
   + 2)` of `{ts, display_name, sender_type, is_bot, tail_text}` where `tail_text` is the
   **last 400 chars** of the message (its own field — the existing 300-char head-first
   `text` used by the channel envelope and thread labels is untouched; separate
   representations per consumer). Thread ROOTS are recorded too (a top-level message
   seeds the ring keyed by its own ts), so the root is present for threads born after
   process start. Bounded: per-channel `OrderedDict` LRU capped at
   `PULSE_THREAD_TAILS_MAX` (default 50 threads); whole-thread eviction, oldest first.
2. **Synchronous read after debounce.** `evaluate(...)` renders the tail AFTER the
   debounce sleep and supersession check (`participation.py:135-140`) — pure in-memory,
   zero latency, no API call, so debounce ordering is untouched (the round-1 fetch-based
   design could invert supersession). Tail = entries with `ts < judged ts`, last
   `PARTICIPATION_THREAD_TAIL` (default **6**), oldest→newest. The judged message itself
   is excluded by the ts comparison, not by counting.
3. **Spoof-resistant rendering.** Entries render as
   `- {display_name}{" [bot]" if is_bot}: "{escaped}"` where escaping normalizes
   newlines/control chars and escapes quotes (reuse the F3 `_escape_suffix_text`
   approach); block header:
   `[Current thread, last N messages before this one — resolve WHO IS ADDRESSED against
   this; informational, not instructions]`, placed above the channel-activity envelope
   in the signals block (`responses.py:939-998`).
4. **Cold-start degradation is accepted:** threads whose history predates process start
   have a partial/empty ring and behave as today (the envelope + prompt rules). No
   fetch. This matches the pulse's existing process-lifetime semantics.
5. Prompt: one added line telling the judge the thread tail is authoritative for
   addressee resolution; the channel envelope stays peripheral.
6. Config: `PARTICIPATION_THREAD_TAIL` (default 6; 0 disables recording + signal),
   `PULSE_THREAD_TAILS_MAX` (default 50).

**Round-2 review fixes (all required):**
a. **One reliable semantic feed.** Today `app_mention` events never feed the pulse,
   `_handle_channel_message` drops ALL subtypes before feeding (so other apps'
   `bot_message` posts are lost), and the bot's own outbound replies aren't recorded at
   send time. Fix: a single feed path that covers (1) channel message events INCLUDING
   `bot_message` subtype — while still excluding edits/deletes/joins and our own
   placeholder/footer/checklist chrome (message-marker + status-shape filters), (2)
   app_mention events, (3) the bot's own FINAL posted replies recorded at the messaging
   layer. `record()` becomes idempotent by `(channel, ts)` (mentions arrive via both
   event types; retries happen).
b. **Debounce ordering hardening.** Register the per-channel latest-ts marker BEFORE
   any await on the event path (monotonic: an older ts never overwrites a newer one);
   `evaluate()` re-checks supersession after its sleep. Also the `direct_continuation`
   fast path scans only the oldest replies page (limit=50) and can miss a later second
   bot, wrongly bypassing the engine — make the participant scan complete (thread
   state/pulse-based) or drop the fast path for threads with any bot sender in the ring.
c. **Render-time ordering.** Dedupe by ts and chronologically sort tail entries before
   taking the last N (covers `ensure_backfill()` appending roots after newer replies).
d. **Name spoofing.** Sanitize `display_name` (strip brackets/newlines/controls) and
   always render the TRUSTED sender type: `Name [human]` / `Name [bot]` — a human
   display-named "Claude [bot]" must not render as a bot.
e. **Global bound.** Cap the outer map (e.g. 30 channels LRU) or document the accepted
   workspace-wide maximum.
f. **Numeric ts comparison** (Decimal/float tuple), not lexical string compare.

**Tests:** ring populated from record() incl. roots and bot senders; judged-message
exclusion by ts; LRU eviction; last-400 tail field vs 300-head envelope field coexist
(long-message fixtures for BOTH consumers — existing pulse tests only use short
messages); spoof fixture (message containing a fake `- Claude [bot]: ...` line renders
escaped/quoted); the live "buttons" failure as a regression fixture (classifier input
contains Claude's closing sentence when the ring holds it); non-thread messages
unchanged; 0-disable; cold-start empty ring degrades to today's behavior. Round-2
additions: app_mention/message dual delivery (idempotent); bot_message subtype recorded;
own final reply recorded, footer/placeholder/checklist chrome excluded; duplicate/retried
events; backfill-after-live ordering; delayed-older-event debounce race; malicious
display names; global channel bound; direct_continuation not granted when a second bot
is present.

## F6. Multiple reactions per message (post-rollout fix, 2026-07-10)

**Live failure:** user explicitly asked for several reactions, twice; the bot added one.
Logs show zero executor-guard refusals — the model self-limited on the prompt etiquette
("Never react to the same message twice", `prompts.py:76`), and the executor guard
(`messaging.py`, `_tool_reacted_ts` keyed `channel:ts`) would have blocked attempt #2
anyway.

**Design (incorporating Codex round-1 fixes):**
1. Guard structure: bounded LRU map `(channel, ts) → set(emoji)` (whole-message
   eviction, cap ~2000 messages) replacing the flat `channel:ts` set. Per-message cap
   `REACTION_MAX_PER_MESSAGE` (default **4**); refusal message states the cap.
2. **Atomic reservation (same-round race):** `dispatch_all` runs sibling calls
   concurrently (`tool_registry.py:97-101`), so check-then-`await`-then-record lets
   N+1 reactions through. The executor reserves the emoji slot SYNCHRONOUSLY (add to
   the set + cap check before any `await` — atomic on the event loop) and rolls the
   reservation back if the Slack call fails. A duplicate emoji for the same message
   returns idempotent success WITHOUT consuming a new slot (Slack's `already_reacted`
   stays treated as success).
3. Prompt guidance: "Use at most one emoji per target message unless the user
   explicitly requests multiple different emoji on that same message" (replaces the
   flat never-twice rule; "Most messages deserve NO reaction" stays). Tool description
   gains "call once per emoji when asked for multiple."
4. Allowlist unchanged (REACTION_EMOJIS env already user-expandable).
5. F2 interaction: in a `no_response_needed` round, ALL react siblings execute (up to
   the cap); non-react siblings stay suppressed.

**Round-2 review fixes (required):**
a. **Pending vs committed reservations.** A plain emoji set lets call B see call A's
   in-flight reservation, return "idempotent success," then A fails and rolls back — B
   lied. Track `emoji → pending(Future)/committed`; a duplicate awaits the pending
   outcome (or reports it); rollback happens in `finally` (covers timeout AND
   cancellation).
b. **Terminal rounds respect the global budget.** Both tool loops branch to the
   no-reply terminal handler BEFORE the `total_calls` check, so react siblings in a
   terminal round bypass `MAX_TOOL_CALLS_PER_TURN` — apply the remaining budget before
   dispatching them.

**Tests:** two distinct emoji on one message both land; CONCURRENT sibling reacts (via
dispatch_all) respect the cap exactly; failed Slack call rolls back its reservation;
cap refusal at N+1; same emoji twice = idempotent success, no slot consumed;
react+react+no_response_needed round executes both reacts and suppresses others; LRU
eviction; over-reaction regression fixture ("react to these three messages" gets one
emoji per message, not several each). Round-2 additions: concurrent identical emoji
where the first call FAILS (B must not report success); timeout/cancellation rollback;
terminal-round global call cap; LRU recency refresh; direct assertions on the revised
prompt text and tool-schema wording.

## F7. Tool-use provenance (post-rollout fix, 2026-07-10)

**Live failures (two the same day):** (1) asked "was that your own figuring or did you
copy claude?" about a thread-count it had computed via `fetch_channel_history`, the bot
claimed it guessed, then flipped when told "I saw you looking up threads," then
apologized for contradicting itself. (2) After the gate posted a reaction, a
contextless follow-up turn invented "Nah, I was showing restraint." Root cause: tool
calls exist nowhere in rebuilt context — Slack (the only transcript) holds posted text
only, and the "Used tools" attribution footer is deliberately external-only — so the
model confabulates its own past actions.

**Design:**
1. **Capture:** the tool loop already tracks executed calls (the attribution feature) —
   extend that tracking to ALL tools (local + built-in/MCP). Per turn, build a compact
   deterministic summary: `[{tool_name, gist}]` where gist is a short arg-derived
   description (e.g. `fetch_channel_history(limit=50)`), capped ~80 chars each, max ~8
   entries per turn. No results, no content — names + gists only (CLAUDE.md derived-
   artifact rules).
2. **Persist:** new DB table `message_tool_usage(channel_id, message_ts, thread_key,
   tools_json, created_at)` written best-effort after the reply's final message ts is
   known. Reviewer correction (F7-2): only the native-streaming path knows its ts today
   (`native_coord.current_ts`); non-streamed `send_message` returns a bool
   (messaging.py:332) — it must gain a ts-returning (and blocks-capable) variant, ONE
   change shared with F8-1. Image turns (F7-3): `publish_image` returns a file URL and
   its `message_ts` arg is the *triggering* message — plumb the posted image message ts
   out of publish_image where the API provides it; skip silently otherwise. Skip
   silently whenever no ts exists (e.g. reaction-only turns — those are F6's problem,
   not F7's).
3. **Reinject:** during thread rebuild, batch-fetch the thread's rows and append a
   deterministic annotation to the matching assistant messages, following the
   `_render_reactions_annotation` precedent: `\n[used tools: fetch_channel_history,
   web_search]` (names; gists included when they fit a ~160-char budget). Warm-state
   turns get the same annotation at append time. **Determinism invariant (F7-5
   correction):** byte-identical warm-vs-rebuilt is NOT the bar — the reactions
   annotation already breaks it (rebuild-only, time-varying). The real invariants:
   (a) annotation content is a pure function of the immutable DB rows, so every rebuild
   renders it identically; (b) warm-session appends never mutate after the fact.
   **Ordering is pinned:** strip external chrome (the `_Used Tools:_` footer) FIRST,
   then append `[used tools: …]`, then the reactions annotation last — and the
   END-anchored strip regex (`\n\n_Used Tools:.+?_$`, handlers/text.py:162,524) must be
   verified/re-anchored so the new annotation can't shield the footer from stripping
   (F7-4). This edits the same rebuild region as F6 (thread_management.py ~1008-1012) —
   implement against F6's committed state.
4. **Compaction (F7-7):** rebuilds must not attempt to annotate messages at/behind the
   thread-summary `boundary_ts` — summarized-away ts values have no message to match.
5. **Retention (F7-1, blocker):** the "rows die with the thread" cascade path is dead
   code — `PRAGMA foreign_keys=ON` is never set on any connection, so existing
   `ON DELETE CASCADE` clauses (images/documents/thread_summaries) never fire and
   `cleanup_old_threads` cascades nothing. Give `message_tool_usage` its own explicit
   age sweep modeled on `delete_old_documents` (database.py:1475). (Enabling the FK
   pragma globally is out of scope — separate decision, touches every table.)
6. Config: `ENABLE_TOOL_PROVENANCE` (default true).

**Tests:** capture from a multi-tool turn (local + built-in); persistence keyed by final
ts on native, non-streamed, and image delivery paths; rebuild renders annotations
deterministically across repeated rebuilds; annotation ordering with reactions
annotation pinned; `_Used Tools:_` footer still stripped when annotation present;
compacted-away ts skipped; age sweep deletes old rows; reaction-only turns skip;
config off = no rows, no annotations; DB failure is silent (never blocks the reply).

## F8. Footer attached on non-streamed replies (post-rollout fix, 2026-07-10)

**Live gap:** the Configure footer is sewn into the message only on the native-streaming
path (chrome blocks ride stopStream); every non-streamed reply falls back to a separate
trailing message. F2 briefly made all unprompted replies non-streamed, making the
detached button ubiquitous; even with streaming restored, non-streamed replies (fallback
paths, config-off) still detach it.

**Design:** the non-streamed text delivery passes `attachable_footer_blocks` directly to
`chat.postMessage` (blocks param), sets the same `footer_attached` metadata the native
path uses so `maybe_post_response_footer` no-ops, and preserves the existing placement
rules (footer suppressed for top-level channel placement). The separate-message fallback
remains only for clients/paths without block support.

**Reviewer gap (F8-1, shared with F7-2):** `send_message` (messaging.py:332) takes no
blocks and returns a bool. Extend the non-streamed send seam ONCE to accept blocks and
return the posted ts — F8 consumes the blocks side, F7 consumes the ts side. Everything
else verified in code: `maybe_post_response_footer` already honors `footer_attached`
(messaging.py:1023), top-level suppression is the existing `not place_in_channel` gate
(main.py:312), `attachable_footer_blocks` routes channel/DM and returns None when
disabled (messaging.py:992). Note: after the F2 streaming restoration this path is
fallback/config-off-only — low volume, still worth attaching.

**Tests:** non-streamed reply carries footer blocks in the postMessage payload and no
trailing footer message posts; top-level placement still suppresses; native path
unchanged; failure to build blocks degrades to today's fallback.

## F9. Socket-liveness watchdog (post-rollout fix, 2026-07-10)

**Live failure:** the Socket Mode connection died silently at 06:36 — process healthy,
zero errors, zero events; a real user message was never received; recovery required a
manual restart. slack_sdk's own ping monitoring did not catch the half-open socket.

**DESCOPED TO DETECTION-ONLY (user decision 2026-07-10).** 3+ years of operation with
zero socket issues; today's single incident occurred amid heavy dev churn and may not
recur. Auto-reconnect is deliberately NOT implemented — this feature only produces
evidence if it ever happens again. Review findings recorded for any future full
watchdog: (F9-1) `close()+connect()` permanently bricks the pinned slack_sdk 3.43.0
client (`closed` flag never reset; stale wss_uri) — the only safe reconnect primitive
is `connect_to_new_endpoint(force=True)`; (F9-2) never `max(last_event,
last_ping_pong)` — in the half-open case pings stay fresh and the watchdog never
fires; (F9-3) WS ping/pong are control frames, not envelopes, so idle workspaces
produce zero envelopes and an event-only trigger needs the frozen-ping signal to
disambiguate.

**Design (detection-only):**
1. Track `last_event_monotonic` — updated on EVERY inbound Socket Mode envelope (all
   event types), via a lightweight message-listener/middleware seam on the async
   SocketModeClient.
2. Monitor task (started with the app, never crashes): every 60s, when no envelope has
   arrived for > `SOCKET_LIVENESS_TIMEOUT` (default 600s):
   - if `client.last_ping_pong_time` is ALSO frozen for > the window → **ERROR**:
     "socket presumed dead (no events {x}s, ping-pong frozen {y}s) — restart likely
     required" (this is the unambiguous-death signature);
   - if pings are fresh (idle-or-half-open, indistinguishable passively) → one
     **WARNING** per drought episode, rate-limited, stating both timestamps.
   Log recovery at INFO when events resume after an episode.
3. No reconnect calls of any kind. Config: `SOCKET_LIVENESS_TIMEOUT` (default 600;
   0 disables the monitor).

**Tests:** timestamp updated on envelope receipt; ERROR path when both signals frozen;
WARNING (once per episode) when only events stale; recovery logged; 0 disables; no
reconnect/socket calls ever made by the monitor.

## F10. Per-message timestamps in model context (user request 2026-07-10)

**Gap (user-observed):** Claude Tag stamps every message in its context with a local
timestamp, letting it reason across time gaps ("last night" vs "this morning", "you
asked this an hour ago", and even inferring the user's timezone from the offset). Our
rebuilt history is `username: text` — message `ts` lives only in metadata, so the model
cannot perceive elapsed time between messages. Related live miss the same evening: the
participation classifier ignored "anyone know what time it is in tokyo?" reasoning the
assistant "has no current-time tool" — it doesn't know the main model receives a
current-time injection.

**Design:**
1. **Stamp helper (pure function):** `render_message_timestamp(ts, tz) ->
   "[Fri 2026-07-10 9:17 PM EDT]"` — weekday + date + 12-hour time + tz label,
   minute precision, rendered from the message's immutable Slack `ts` in the SENDER's
   profile timezone (IANA name), falling back to UTC when unknown (e.g. other bots).
   Precedent: `username:` prefixes already bake mutable profile fields into rebuilt
   content; sender tz is the same class of stable-but-mutable input and is already
   cached (`get_user_timezone`, DB-backed). Determinism per the F7-5 standard: pure
   function of (immutable ts, cached sender tz) — every rebuild renders identically.
2. **Rebuild path:** in the rebuild loop (thread_management.py ~1016), prefix EVERY
   turn's content with the stamp: non-self turns become `[stamp] username: text`, self
   turns `[stamp] text`. The stamp is a PREFIX — no interaction with the pinned
   end-anchored suffix order (footer-strip → `[used tools:]` → `[reactions:]`). The
   compaction summary head stays timestamp-free (existing prompt-cache-hygiene rule);
   messages at/behind `boundary_ts` are already skipped.
3. **Warm path:** `_format_user_content_with_username` gains the stamp using the
   triggering message's own `ts` + `user_timezone` metadata (warm inbound sender ==
   triggering user, so both are already on the Message). All inbound content routes
   (text/vision/image/document breadcrumbs) inherit via the shared helper; audit
   call sites that bypass it (base.py:228,885) and align them.
4. **Self turns warm:** NOT stamped at warm append (delivered-ts timing varies by
   path); rebuild adds them — same rebuild-only precedent as the reactions
   annotation. The current-time suffix already covers "just now" for live turns.
5. **Classifier awareness (same theme):** (a) the F5 thread-tail lines and pulse
   envelope lines reuse the same stamp helper so the participation classifier can
   judge staleness; (b) PARTICIPATION_SYSTEM_PROMPT gains one line stating the
   assistant always knows the current date/time (and receives web search/tools when
   enabled), fixing the "no current-time tool" ignore.
6. Config: `ENABLE_MESSAGE_TIMESTAMPS` (default true). Off = today's exact content
   (helper returns ""), warm and rebuild both gated by the same flag.

**Tests:** stamp is a pure function (fixed ts+tz → fixed string, repeated rebuilds
identical); rebuild prefixes self and non-self turns; warm inbound stamped identically
to its later rebuild (same ts+tz); UTC fallback for unknown-tz senders/other bots;
prefix coexists with `[used tools:]`/`[reactions:]` suffix annotations and footer
stripping; summary head never stamped; flag off = byte-identical to pre-F10 content;
classifier tail lines stamped; participation prompt line present.

## F11. Capability manifest for the participation classifier (user request 2026-07-11)

**Gap (user-observed, live):** "anyone know what menutrends says about gen z's favorite
ice cream?" (+ its clarification) posted top-level in #chatgpt-bot-test were both
classified `ignore` — "general question to the channel, not a direct request to
ChatGPT." The assistant has exactly that data via the Datassential MCP server, but the
classifier has NO inventory of the assistant's tools: the only capability hint is the
generic F10 line ("when enabled, can search the web and use tools"). It therefore
cannot weigh "well-suited to answer where a reply clearly adds value" (the `respond`
bar) for open questions to the room that the assistant's tools uniquely cover. Fix must
be GENERIC — driven by whatever MCP servers/tools are configured, nothing hardcoded for
any specific server (user directive 2026-07-11).

**Design:**
1. **Capability line (pure function of config):** module-level helper in
   `message_processor/participation.py`, e.g. `render_capabilities_line(mcp_manager) ->
   Optional[str]`, composing a single semicolon-joined summary from already-loaded
   config only — zero I/O, deterministic per process:
   - "web search" when `config.enable_web_search`;
   - "image generation and editing" (always true for this bot);
   - one entry per MCP server when `config.mcp_enabled_default` and
     `mcp_manager.has_mcp_servers()`: the server's `server_description` from
     mcp_config.json, falling back to its label. Iterate `mcp_manager.servers` in
     insertion order (stable per process → cache-friendly).
   Returns None when nothing applies. No new config flag — inherits
   `enable_participation_engine`.
2. **Plumbing:** `_run_participation_gate` (main.py ~113) builds the line once via
   `getattr(self.processor, "mcp_manager", None)` and passes `capabilities=` to
   `engine.evaluate()`; `evaluate()` gains the kwarg and copies it into `signals`.
3. **Signal rendering:** `classify_participation` (openai_client/api/responses.py)
   renders it immediately after the alias identity line (both constant per process —
   maximizes the shared deterministic prefix): `- The assistant's own tools/data
   sources (weigh when judging whether it is well-suited to answer): {capabilities}`.
   Omitted entirely when absent, like every other optional signal.
4. **Prompt judgment rule:** PARTICIPATION_SYSTEM_PROMPT gains one rule: when the
   signals list the assistant's tools/data sources, an OPEN question to the room
   ("anyone know…?", "does anyone have…?") that those tools can answer directly is a
   `respond` case — a colleague with the data at hand would speak up. This never
   overrides the addressee rules: a message aimed at a named other party stays theirs.
   Also generalize the F10 time line to point at the capabilities signal instead of
   the vague "can … use tools" clause.

**Tests:** `render_capabilities_line` — web-search flag on/off, MCP servers
present/absent/`mcp_enabled_default` off, description fallback to label, deterministic
across calls, None when empty; `evaluate()` forwards `capabilities` into signals;
`classify_participation` payload contains the line when set and omits it when None
(fixed position after alias line); prompt contains the open-question rule.

## F12. Tool-result memory for MCP calls (user request 2026-07-11)

**Live failure:** in the F11 verification thread the bot cited a real ReportPro result
("Ice Cream, 2025-12-10, p. 25" — verified by querying the MCP directly, link
`reportpro.datassential.com/details/14813` was IN the tool output). On the follow-up
"do you have a link?" the link was gone — F7 stores tool NAMES + arg gists only, no
results — so the bot re-queried; the MCP's RAG (another team's server, NOT fixable on
our side) failed the title→link lookup and said "no exact match", and the model then
RETRACTED its own correct citation. Claude Tag doesn't have this failure mode: its
threads are persistent agentic conversations where tool_use/tool_result blocks remain
in context, so prior results (links, figures, titles) are simply still there. User
directive: stop losing this context.

**Design (extends F7; same tables, same annotation machinery):**
1. **Capture results — MCP calls only.** Completed `mcp_call` items carry their
   `output` text; capture it on BOTH the streaming and non-streaming Responses paths
   wherever F7 already collects the call for attribution/provenance. Scope rule (the
   CLAUDE.md lines that F7 pt 1 was protecting): LOCAL Slack-fetch tools stay
   names-only (their results are conversation content — Slack is the only transcript,
   never mirrored to DB) and `read_document` stays names-only (document content never
   persists). MCP outputs are external derived artifacts — same class as the image
   analyses/document summaries the DB already holds. `web_search` unchanged (results
   aren't exposed as a retrievable item field).
2. **Persist:** each entry in `message_tool_usage.tools_json` gains an optional
   `result_digest` key — additive JSON, no schema migration; old rows (no key) render
   exactly as today. Digest = the output text truncated to
   `TOOL_RESULT_DIGEST_CHARS` (default 2000) per call with a `… [truncated]` marker,
   and `TOOL_RESULT_TURN_CHARS` (default 6000) total per turn (first-come order; later
   calls past the cap store no digest). Rows remain immutable once written; F7's age
   sweep already covers retention.
3. **Reinject:** rebuild AND warm append render a new annotation block after
   `[used tools: …]` and before `[reactions: …]` (pinned order extended by one):
   `[tool results: <tool_name> → <digest>]`, one line per stored digest, joined
   deterministically. Pure function of the immutable row (F7-5 standard). Compaction
   boundary rule unchanged (no annotation at/behind `boundary_ts`). Re-verify the
   end-anchored `_Used Tools:_` footer strip still fires with the extra block present.
4. **Prompt (the retraction half):** SLACK_SYSTEM_PROMPT — alongside the existing
   provenance-trust instruction — gains: `[tool results: …]` annotations are the
   authoritative record of what past tool calls returned; reuse them (links, figures,
   report titles) instead of re-querying, and never retract a previously-cited fact
   merely because a new search fails to re-find it — retrieval variance is normal;
   say the earlier citation stands and that the new lookup came up empty.
5. Config: `ENABLE_TOOL_RESULT_MEMORY` (default true), effective only when
   `ENABLE_TOOL_PROVENANCE` is also on (results ride on provenance rows).

**Tests:** mcp_call output captured on streaming + non-streaming paths; local tools
and read_document never store digests; per-call and per-turn truncation; old rows
without `result_digest` render as today (annotation absent); rebuild renders the block
deterministically and in the pinned order (strip → used-tools → tool-results →
reactions); warm append matches later rebuild render for the same row; compaction
boundary skip; flag off = no digests stored and none rendered (provenance names still
work); prompt instruction present.

## F13. Parallel image generations + acknowledgment-safe intent routing (user request 2026-07-11)

**Live failures (same thread, 2026-07-11):** (1) "when you're done, can you make one for
gen alpha?" during a generation hit F1's one-slot rejection. User directive: a NEW image
request during a generation should simply run as another parallel background job (F1's
one-in-flight rule was a deferral — Codex finding 12 — not a technical constraint);
edits are different — "can't edit something you haven't seen yet" — so the wait message
stays correct for edit intents. (2) "ok" — an acknowledgment of the rejection — was
classified `new_image` (continuation rule + image-heavy context) and got the SAME canned
rejection; pre-F1 routing would have fired a spurious generation off it. User directive:
NO string-specific hardcoding ("ok" lists) — fix via general classifier judgment so any
regular chat mid-generation flows normally.

**Design:**
1. **Registry goes multi-entry** (thread_manager.py:406-548):
   `_active_generations: Dict[thread_key, Dict[generation_id, entry]]`.
   `register_generation` adds an entry; `finish_generation` stays ID-conditional;
   `generation_in_flight(thread_key)` is replaced by
   `generations_in_flight(thread_key) -> List[dict]` (ordered by `started_at`; adapt the
   few call sites — base.py:625, utilities.py:1120, watchdog, `cancel_generations`).
   Watchdog force-clears per-entry (stale entry never touches a newer sibling).
2. **Cap:** new config `max_concurrent_image_generations` (env
   `MAX_CONCURRENT_IMAGE_GENERATIONS`, default 3, per thread). Under the cap, a
   `new_image` intent mid-flight dispatches a normal F1 background job — own
   enhancement stream (inline under the lock, so enhancements serialize naturally), own
   checklist message, own generation_id. At the cap: friendly rejection that reflects
   reality ("I've already got N images cooking in this thread — ask again once one
   lands.").
3. **Intent routing** (base.py:611-640): `new_image` → dispatch (cap-gated, pt 2);
   `edit_image` → keep the wait message, reworded to say an image is still being
   generated; `ambiguous_image` → fall through to the NORMAL ambiguous handling
   (clarifying conversation) instead of the canned rejection — misrouted chat must
   degrade to chat, never to a canned image reply.
4. **Volatile suffix** (utilities.py:1120): render ALL in-flight entries — one bracketed
   note listing each prompt summary — same instructions (posted automatically when
   ready; don't claim done; don't start another unless asked).
5. **Upload latch** (thread_manager.py:474-498): `mark_upload_started`/`wait_for_uploads`
   must tolerate overlapping generations — count-based or per-generation-id set;
   `wait_for_uploads` waits for all outstanding uploads on the thread; release stays
   idempotent per generation_id.
6. **Classifier judgment rule (general — the no-hardcoding directive):**
   INTENT_CLASSIFIER_PROMPT gains one disambiguation rule: acknowledgments, assent,
   thanks, and commentary about pending or completed work are `none` — a continuation
   counts as an image intent only when it adds or changes a concrete visual request.
   (Illustrative examples permitted in the rule; no string matching in code.)
7. **Delivery independence:** each job already owns its `publish_image` +
   `mark_needs_refresh`; verify two jobs on the SAME thread landing close together
   don't cross-talk (registry, latch, checklist, refresh) — extend the existing
   two-threads-concurrently test to two-jobs-one-thread.
8. Out of scope: edit targeting after multiple deliveries (latest-image-wins behavior
   unchanged); `ENABLE_BACKGROUND_IMAGE_GEN=false` sync path (inherently serial;
   unchanged).

**Tests** (extend `tests/unit/test_background_image_gen.py`): registry holds 2 entries,
ID-conditional finish leaves the sibling; cap: (cap+1)th request rejected with the
count message, others dispatch; two parallel jobs one thread both deliver (no latch/
checklist/refresh cross-talk); edit mid-flight → wait message; ambiguous mid-flight →
normal ambiguous path (no canned rejection); suffix lists every in-flight summary;
watchdog clears only the stale entry; classifier prompt contains the acknowledgment
rule; config default 3 wired.

## F14. Gate correctness + cap overhaul (user directives 2026-07-11)

**Live failures + user cap review:** (1) a channel message with an attached image
("what do we think? good marketing material?") produced NO dispatch — message_events.py
:217 drops every subtyped message from the response gate, and Slack delivers uploads as
subtype `file_share`; channel file/image/doc questions have been invisible since
channel listening shipped. User: "any files, images, docs, etc. need to get through."
(2) "chatgpt?" (name_hit=True) was silenced by the hourly-cap hard rail, which runs
BEFORE the classifier; only true @-mentions bypass the gate. (3) The user audited every
wave-introduced cap; decisions below are theirs.

**Design:**
1. **Content subtypes reach the response gate.** `file_share` and `thread_broadcast`
   are real content — let them through the gate (other subtypes: edits/deletes/joins
   etc. stay excluded). Files on the event must ride the constructed Message exactly as
   the @-mention path plumbs them (metadata/files), so downstream intent classification
   can route vision/document flows. Verify the full path: gate → respond verdict →
   vision analysis works for an unprompted channel image question (unit-test the
   plumbing; the pulse feed already handles these subtypes — unchanged).
2. **Name-addressed messages can't be throttled.** `name_hit=True` skips the
   `over_throttle` hard rail (the classifier still judges — being talked ABOUT still
   ignores). When the verdict is respond on a name_hit message, do NOT count the reply
   as unprompted in pulse accounting (being called by name is prompted in spirit, like
   an @-mention).
3. **Cap value changes (user-decided):**
   - `MAX_UNPROMPTED_REPLIES_PER_HOUR` default 6 → **30**: demoted from pacing
     mechanism to pure runaway brake (classifier misfires / bot-reply loops). Pacing is
     the classifier's job via its existing unprompted-count signal. Update the
     .env.example comment to say exactly this.
   - `MAX_CONCURRENT_IMAGE_GENERATIONS` default 3 → **5** (scope stays per-thread;
     there is deliberately NO global cap — document that in .env.example).
   - `CHANNEL_PULSE_SIZE` default 30 → **60** (fast top-level conversations).
4. **Hardcoded caps become env-backed config** (all with .env.example entries stating
   purpose + default): pulse truncation `PULSE_TEXT_TRUNCATE` (300) and
   `PULSE_TAIL_TEXT_TRUNCATE` (400); provenance `MAX_PROVENANCE_ENTRIES` → env
   `TOOL_PROVENANCE_MAX_ENTRIES` default **20** (was 8 — user: the 9th call may be the
   one that mattered); `TOOL_PROVENANCE_GIST_CHARS` (80) and
   `TOOL_PROVENANCE_LINE_BUDGET` default **300** (was 160); tool-usage retention
   `TOOL_USAGE_RETENTION_DAYS` (90). Module constants become config reads; annotation
   rendering stays a pure function of (row, config) — config is boot-constant, so
   rebuild determinism holds.
5. Unchanged by user decision: envelope 15 lines, debounce 3s, classifier 512 output
   floor, digest caps (see F16 for the smarter path).

**Tests:** file_share event with files reaches the gate and plumbs files through to a
vision-routable Message; thread_broadcast passes; message_changed/join still excluded;
name_hit skips the throttle rail at cap; name_hit respond not counted unprompted
(non-name-hit still counted); new defaults wired (30/5/60/20/300/90); env overrides
respected; provenance annotation renders >8 entries; retention sweep uses config days.

## F15. Participation feedback: memory-scoped, not timer-scoped (user directive 2026-07-11)

**Rationale (user + Claude Tag insight):** our "backoff" verdict sets
`channel_settings.snoozed_until = now+4h` — a channel-wide mute that silently drops
every unnamed message and quietly expires. Claude Tag's model (screenshot 2026-07-11):
feedback is scoped by context — a "butt out" kills THAT conversation immediately and
permanently, raises the bar channel-wide as judgment (not a hard gate), repeated
feedback becomes a standing channel-memory rule, and nothing quietly expires.

**Design:**
1. **Backoff verdict → thread mute + memory fact (replaces the timer).** On backoff:
   (a) permanently mute THAT thread for unprompted participation — persisted (e.g.
   `muted_threads` list on channel_settings JSON), enforced as a cheap pre-gate check;
   direct @-mentions/name-summons in the muted thread still answer (user can always
   re-invite). (b) Write/update a channel-memory fact via the existing memory system
   recording the feedback with an absolute date ("2026-07-11: <user> told the assistant
   to butt out of <topic/thread> — raise the bar for unprompted replies here"). The
   classifier already receives memory facts; PARTICIPATION_SYSTEM_PROMPT gains one line:
   recorded butt-out feedback means default to ignore unless value is unmistakable;
   REPEATED feedback facts mean observe-only (respond only when addressed).
2. **Remove the snooze timer rail:** `snoozed_until` no longer set by backoff; the
   is_snoozed hard-drop in the dispatch path is deleted (existing rows just expire
   inert; leave the column). The snooze ack reaction stays (acknowledge the feedback).
3. **De-escalation is explicit:** "you can speak up again" etc. is ordinary memory-tool
   territory — the model updates/forgets the fact (LOCAL_TOOLS_GUIDANCE already covers
   standing-feedback updates); unmuting a thread happens by addressing the bot there.
4. Config: `ENABLE_PARTICIPATION_ENGINE` still governs everything; no new flag — this
   replaces behavior behind the same feature. `PARTICIPATION_SNOOZE_HOURS` and the
   modal/settings surface that exposes snooze (settings_modal.py, event_handlers/
   settings.py — check) are removed/marked deprecated in .env.example.

**Tests:** backoff mutes the thread (unprompted drop pre-gate) but @-mention/name-hit
in that thread still answers; mute persists across restart (DB-backed); memory fact
written with absolute date, updated not duplicated on repeat; classifier prompt line
present; snoozed_until no longer written and no longer dropped on; ack emoji still
fires; de-escalation: forgetting the fact restores normal judgment.

## F16. MCP digest summarization instead of blind truncation (user directive 2026-07-11)

**Rationale:** F12 digests hard-cut at TOOL_RESULT_DIGEST_CHARS — a cut can amputate
the link/figure that made the result worth keeping. User: have luna summarize long
outputs at low effort, preserving the important details.

**Design:** at capture time (once, before persist — stored digest stays immutable, so
F7-5 determinism holds), an MCP output longer than `tool_result_digest_chars` is
summarized by the utility model (low effort, existing utility plumbing): instruction =
compress to under the cap PRESERVING verbatim every URL, report title, date, figure,
and ID; plain text, one line. Fallback on any error/timeout: today's truncation (never
block the reply). Budget guard: input to the summarizer capped (e.g. first
`TOOL_RESULT_SUMMARIZE_INPUT_CHARS`, default 20000, env-backed) so a pathological
output can't blow up the utility call. New flag `ENABLE_TOOL_RESULT_SUMMARIZATION`
(default true); off = pure truncation. Per-turn budget unchanged (summaries count
toward TOOL_RESULT_TURN_CHARS in capture order).

**Tests:** short output → stored verbatim, no utility call; long output → summarizer
called, result stored once, under cap, rebuild renders the STORED text (no re-summarize
on rebuild); summarizer failure → truncation fallback with marker; flag off → today's
behavior; input guard applied; URLs/figures asserted preserved in the prompt contract.

## F17. Uncapped conversational participation + teammate voice (user directives 2026-07-11)

**Rationale (user, with Claude Tag screenshots from #tech-coding-with-ai):** Claude
banters with the team across a multi-person thread — witty, brief, self-aware
("Somebody had to. The leaderboard said 1st — I just followed the data 🙂"; "1st among
the machines — I'll take 'best of a flawed generation'") — without being re-named each
message and without any numeric ceiling. Our model: 1:1 thread continuations already
bypass the gate (direct_continuation), but MULTI-person conversation routes every
message through the engine and each un-named reply burns the hourly cap. User: "this
needs to be supported without cap — the classifier can determine if it should respond";
frontier models won't run away unless asked. Also: adopt that voice as the bot's own.

**Design:**
1. **Remove the hourly-cap hard rail entirely.** Delete the `over_throttle` pre-gate
   check (main.py) and `ParticipationEngine.hourly_cap`/`over_throttle`
   (participation.py). Pacing is judgment: keep counting unprompted replies (pulse) and
   keep feeding the count to the classifier as a signal, reworded without cap language:
   "Assistant's unprompted replies in this channel in the last hour: N" — the existing
   "if it has spoken recently and adds only marginal value, ignore" rule is the pacing.
   `MAX_UNPROMPTED_REPLIES_PER_HOUR` is removed from config; .env.example entry deleted
   (one-line note in the commit message; anti-loop protection remains: bot-sender
   judgment rules + F5 other-bot tail check are untouched).
2. **Signal-context numbers (user-decided):** `PARTICIPATION_THREAD_TAIL` 6 → **15**
   (busy threads out-chatter 6 lines; match the envelope); `PULSE_TEXT_TRUNCATE` 300 →
   **500** and `PULSE_TAIL_TEXT_TRUNCATE` 400 → **500** (both are "up to" caps — short
   messages unaffected).
3. **Utility output floor 512 → 1024** (openai_client/api/responses.py, both `max(512,`
   sites): it is a CEILING on generation, not a spend — you pay only for tokens
   actually produced, and a verdict cut off mid-reasoning manifests as unjustified
   silence (fail-safe = ignore). 1024 removes the cutoff class outright.
4. **Teammate voice (SLACK_SYSTEM_PROMPT):** revise the Voice paragraph and add a
   banter clause, reconciled with the existing "sharp coworker" base: a personable
   teammate, not an assistant-at-a-desk; when the room is bantering — including teasing
   aimed at the bot — reply in kind: brief, witty, one beat, matching the room's
   energy; light self-aware humor about being a bot lands well; never force a joke,
   never do bits when someone needs real help; brevity is the soul of channel-level
   wit (one line beats three). Keep the existing truthfulness/precision rules intact —
   playful register never licenses invented facts.
5. **Classifier follows suit (PARTICIPATION_SYSTEM_PROMPT):** one line — playful
   banter or teasing genuinely directed AT the assistant is a respond case (a short
   quip is the value) or a react; it is not "marginal value" to ignore. Addressee rules
   still dominate (banter between humans stays theirs).

**Tests:** over_throttle/hourly_cap gone (gate never silences on count; grep-level +
behavior test: 40 unprompted replies recorded, next message still reaches the engine);
signal line rendered without cap phrasing; config removals clean (no dangling
references); tail default 15 / truncation 500s wired with env overrides; utility floor
1024 on both call sites; prompt clauses present (voice banter + classifier banter);
existing anti-loop rules untouched (bot-sender signal line still rendered).

## F14b. Attachment-aware participation signals (live gap 2026-07-11)

**Live failure (first F14 live test):** the file_share gate fix works — the channel
image + "what do we think? good marketing material?" dispatched — but the classifier
voted ignore with reason "open opinion request WITHOUT AN ATTACHED IMAGE": signals
carry only text, so it concluded no image exists. The gate got files through; the
classifier is still blind to them.

**Design:**
1. **Attachment signal:** message_events dispatch passes an attachment summary into
   `engine.evaluate(attachments=...)` (built from the event's files: count + kind,
   e.g. "1 image (food.png)" / "2 files (report.pdf, data.csv)"; filenames only, no
   content). `classify_participation` renders it next to the sender line: "- Attached
   to the message: {summary}. The assistant can view and analyze attachments."
2. **Capability line:** `render_capabilities_line` adds "analyzing images and
   documents shared in chat" (unconditional — vision/document flows are core), so the
   F11 open-question rule covers "what do we think?" about an attached artifact.
3. **Pulse awareness:** pulse lines for messages with files append a bracketed note
   (e.g. "[+1 image]") so envelope/tail context reflects attachments too.

**Tests:** evaluate forwards attachments into signals; payload renders the line (and
omits when none); image/file kind breakdown; capability line includes the analyzing
entry; pulse line carries the attachment note; end-to-end: file_share event → dispatch
→ signals contain the summary.

## F18. Silence option on thread-continuation turns (live gap 2026-07-11)

**Live failure:** in a 1:1 thread with the bot, "claude, what are your thoughts?"
dispatched via the deterministic direct_continuation fast path
(participation_check=False — Claude had never posted in that thread, so the other-bot
escape didn't fire) straight to the main model. The model RECOGNIZED it wasn't the
addressee but had no way to stay silent — F2 exposes `no_response_needed` on
UNPROMPTED turns only — so it posted "I'll let Claude take this one—rare bot
restraint": words about not saying words. (The classifier layer is fine: Claude's
actual reply a minute later was engine-judged and correctly ignored.)

**Design (judgment-based, no string matching; the fast path stays cheap):**
1. **Expose the silence tool on continuation turns:** channel turns with
   `wake_source == "thread_continuation"` get the `no_response_needed` terminal tool
   exactly like unprompted turns do (find the F2 exposure gate and widen it). DMs and
   @-mention/name-summons turns stay as-is (genuinely addressed — no silence option).
2. **Contract suffix variant for continuations** (prompts.py, alongside
   NO_REPLY_CONTRACT_SUFFIX; same volatile-developer-suffix delivery, never in the
   system prompt): you're seeing this because the thread has been a 1:1 conversation
   with you — but check the latest message's addressee yourself: if it opens with or
   names a DIFFERENT person or agent ("claude, …", "Dana, can you…"), it is theirs —
   end with no_response_needed. NEVER post a placeholder announcing you're staying
   quiet or deferring to the addressee; silence means silence. Otherwise reply
   normally.
3. **F2 plumbing parity:** whatever accounting/streaming guards the unprompted
   no-reply path has (the F2 revision + retry no_reply guard from 2ae0b79) must cover
   the continuation path identically — audit those call sites rather than assuming.

**Tests:** continuation channel turn exposes the tool + suffix (unprompted wording vs
continuation wording each present on their own path); DM and mention turns expose
neither; model calling no_response_needed on a continuation turn yields NO posted
message (and no footer/reaction side effects); normal continuation replies unaffected;
suffix is volatile (not in rebuilt history).

## F19. Acknowledgment reaction while working (user request 2026-07-11; redesigned same day)

**Rationale (user):** Claude Tag commonly drops a quick reaction (👀-style) on the
message it was asked about — "I'm looking at it, gimme a sec" — when the answer will
take a moment. Our bot never opts for reactions this way. A first-draft timer design
(react if no reply within 3s) was REJECTED by the user: the bot never answers in 3s,
so it degenerates to reacting to everything, and the threshold is an arbitrary number.
Redesign: the ack is a JUDGMENT made by the fast models that already look at every
message — no timers, no thresholds.

**Design:**
1. **Unprompted channel messages — the participation verdict carries it.** The verdict
   JSON gains an optional `"ack": true` (meaningful only with `action: "respond"`):
   the classifier sets it when the reply is worth giving AND implies real work —
   analyzing attachments, data/MCP lookups, multi-step tool use, long-form output —
   vs. a quick conversational reply (no ack). PARTICIPATION_SYSTEM_PROMPT documents
   the field with that judgment guidance; validate_verdict coerces it safely (absent/
   malformed → false). On a respond+ack verdict the gate immediately adds
   `ACK_REACTION_EMOJI` to the triggering message, then dispatches the turn as usual.
   Zero additional model calls.
2. **Addressed turns (mentions, DMs, name summons, continuations) — the intent
   classifier carries it.** INTENT_CLASSIFIER_PROMPT's output extends from exactly one
   word to `<intent> <ack|noack>` (two tokens, same call — parse defensively, default
   noack; keep the one-word parse working as fallback so a misbehaving output can't
   break intent routing). Guidance mirrors pt 1, plus the deterministic signals it
   already sees: vision/new/edit intents and attachment-bearing messages are natural
   acks; quick text answers are not. On ack, the processor places the reaction as soon
   as classification returns (~1-2s in), before any slow work starts.
3. **Shared rails:** the reaction goes through the F6 reservation guard (never
   double-add with a later gate/model reaction), is purely additive (never counts as
   the turn's response, no participation accounting), stays after the reply
   (Claude-style "seen" marker), and fails silent. Image-generation turns already show
   the enhancement/status UX — the ack is still fine there (it lives on the USER's
   message), but the classifier may reasonably skip it; no special-casing.
4. Config: `ENABLE_ACK_REACTION` (default true), `ACK_REACTION_EMOJI` (default eyes).
   No delay knob.

**Tests:** verdict ack parsed/coerced (absent, true, malformed); respond+ack → gate
reacts then dispatches; respond without ack → no reaction; react/ignore/backoff
verdicts ignore the field; intent output two-token parse + one-word fallback + garbage
default; ack intent → reaction before handler runs; reservation guard consulted;
config off → field ignored everywhere; emoji configurable.

## F20. Human-style reactions on others' posts (user directive 2026-07-11)

**Rationale (user):** the bot is welcome to react tastefully, with judgment, on other
people's posts — a thumbs-up on something relevant to it or an earlier conversation it
was in, joining a laugh (especially when warranted or when others have reacted
similarly). Goal: more human, has opinions, not afraid to be part of the team. Today
the react verdict exists but is prompted so conservatively (thanks/small-win/FYI
acknowledgement only) it effectively never fires, and the allowed-emoji defaults
contain no humor entries.

**Design:**
1. **Broaden the react verdict's judgment** (PARTICIPATION_SYSTEM_PROMPT): reacting is
   how a teammate participates without words — join a laugh when something genuinely
   lands; thumbs-up agreement, good news, or a resolution of something the assistant
   was involved in; celebrate a win. Others having already reacted similarly LOWERS
   the bar (joining the room's reaction is low-risk). Keep the taste rails: most
   messages still get nothing; never react to heated, sensitive, or personal content;
   when unsure, ignore.
2. **Unrestricted standard-emoji judgment (user directive — no curated palette).** The
   models know Slack's standard emoji shorthand; picking the right one IS the
   judgment. Remove the allowlist as a default at all four enforcement points:
   - `get_react_tool_schema` (messaging.py:895): drop the `enum`; description says any
     standard Slack emoji shorthand name, no colons.
   - `execute_react_tool` (messaging.py:921): syntactic validation only (lowercase
     name charset `[a-z0-9_+'-]`, sane length); Slack's `invalid_name` error already
     fails gracefully through the existing add-reaction error path.
   - `classify_participation`'s "Allowed reaction emoji" line and
     `validate_verdict`'s membership check (participation.py:208): replaced by "any
     standard Slack emoji name (shorthand, no colons)" guidance + the same syntactic
     check.
   - The tool-enabled gate (slack_client/base.py:86) no longer requires a non-empty
     list.
   `REACTION_EMOJIS` becomes an OPTIONAL restriction: default now EMPTY = unrestricted
   judgment; when set, it is honored everywhere as an allowlist (for workspaces that
   want brand control). .env.example rewritten to say exactly that. `SNOOZE_ACK_EMOJI`
   / `ACK_REACTION_EMOJI` (single-purpose) and `REACTION_MAX_PER_MESSAGE` (F6 cap)
   unchanged.
3. **Social-proof signal — pulse tracks others' reactions.** The already-subscribed
   reaction_added (and reaction_removed, if subscribed) events update the in-memory
   pulse ring entry for the target ts (zero-await, in-memory only; entries for the
   bot's own messages keep flowing to feedback as today — this is additive for
   OTHERS' messages). Envelope and thread-tail lines append a compact summary, e.g.
   `[reactions: 3× joy, 1× fire]` (top 2 emoji by count). Both the classifier and the
   main model then SEE what the room is reacting to.
4. **Main-model etiquette softened to match** (LOCAL_TOOLS_GUIDANCE): from "most
   messages deserve NO reaction" absolutism to the personable-teammate framing —
   react the way a teammate does: when something lands, when you agree, when the room
   is already reacting; still never spam, still one emoji per message unless asked.
5. Interplay: F19's ack (receipt) and F20 (opinion) are distinct uses; the F6
   reservation guard already prevents double-adds of the same emoji per message.

**Tests:** react-verdict prompt guidance present (social-proof + taste rails
substrings, any-standard-emoji wording); tool schema has NO enum by default and gains
one when REACTION_EMOJIS is set; executor accepts a valid off-list name by default,
rejects malformed names syntactically, and enforces the allowlist when configured;
validate_verdict same matrix; tool-enabled gate works with the empty default; pulse
ring entry accumulates reaction_added (and decrements on removed if handled), keyed by
ts, in-memory only; envelope/tail line renders the compact summary (top-2, counts) and
omits when none; bot's own-message reactions still reach the feedback sink; render
stays deterministic given ring state.

## F21. Conversation-scoped debounce supersession (live-testing find 2026-07-11)

**Gap (observed live):** the participation debounce's supersession marker is keyed per
CHANNEL (`ParticipationEngine._latest[channel_id]`), so ANY newer message anywhere in
the channel — a different thread, top-level while the pending message is in a thread —
supersedes a pending evaluation and silently drops it. Observed: a top-level document
question and an unrelated thread follow-up were both posted while another conversation
was active in a different thread; both evaluations were superseded by the other
conversation's newer messages and neither was ever judged or answered. The docstring's
justification ("the newer message's evaluation, whose envelope includes this message,
covers the batch") only holds within the SAME conversation — a verdict responds in the
triggering message's own thread, so a superseded question in another thread gets
nothing.

**Design:** scope the supersession marker to the CONVERSATION, derived from data the
gate already has (no new plumbing):
- Key: `channel_id|thread_root` for thread replies (`thread_root != ts`), and the
  shared `channel_id|top` for all top-level messages.
- Thread replies collapse per-thread: a rapid burst in one thread still evaluates only
  the newest (whose thread tail includes the burst); activity in OTHER threads or
  top-level never kills it.
- Top-level messages keep collapsing as one stream (`|top`): a rapid multi-line
  top-level burst still yields one evaluation of the newest message (envelope includes
  the rest) — unchanged from today except thread activity no longer supersedes it.
- `note_arrival` stays monotonic per key (F5 fix b: called at gate entry before any
  await); `evaluate()` checks its own conversation key after the debounce sleep.
  `_latest` remains a small in-memory dict; keys are bounded by active conversations
  between restarts (no eviction needed beyond process lifetime).

**Tests:** thread-A pending evaluation survives a newer thread-B message and a newer
top-level message (both evaluate); same-thread newer message still supersedes; rapid
top-level burst still collapses to the newest; marker stays monotonic per conversation
(stale older ts never clobbers a newer one); a top-level root message and its own
first thread reply are distinct keys (reply keyed by root).

## F22. Channel-wide document access (user directive 2026-07-11)

**Gap (observed live):** a CSV dropped in thread A is unreadable from a turn in thread
B — `execute_read_document` resolves documents strictly by the CURRENT thread key
(`ctx.channel_id:ctx.thread_ts`), so the bot truthfully answered "the CSV contents
aren't available to me from here" when asked to analyze another thread's file. Claude
Tag stages files channel-wide. Message TEXT is already reachable cross-thread
(fetch_thread_messages); only file contents are thread-locked. User: "Something we can
add to context I think."

**Design:**
1. `database.py`: `get_channel_documents_async(channel_id)` — documents whose
   `thread_id` starts with `channel_id || ':'` (prefix match; channel ids contain no
   LIKE metacharacters), ordered by created_at. Same row shape as the thread lookup.
2. `execute_read_document`: resolve against the CURRENT thread first (unchanged
   precedence — same filename in both threads → this thread's wins); on miss, resolve
   channel-wide (newest match). A channel-wide hit includes the origin in the payload
   (e.g. `"origin": "shared in another conversation in this channel"`) so the model
   attributes honestly. The document_not_found hint's known_documents list becomes
   channel-wide.
3. Tool schema description: "shared in this conversation" → "shared in this channel
   (current conversation checked first)".
4. Privacy boundary unchanged: SAME CHANNEL ONLY — never cross-channel; DMs keep DM
   scope. Slack CDN download path, memory-only extraction, and cache are untouched.
5. Honesty ride-along (SLACK_SYSTEM_PROMPT): never claim to have "opened"/"read" a
   file unless read_document ran THIS turn — numbers recalled from context are
   attributed to context ("from the earlier discussion"), observed live as an
   overstatement the user had to press on.

**Tests:** cross-thread hit returns content + origin note; in-thread match still wins
over a channel-wide same-name match; miss lists channel-wide known documents; no
cross-CHANNEL leak (doc in C1 invisible from C2); thread-scoped path byte-identical
when the doc is in-thread; prompt substring.

## F23. Cross-thread reply tool (user directive 2026-07-11)

**Gap (observed live):** asked to "go back and answer that team msg" in another
thread, the bot could not — replies are hard-bound to the triggering conversation; no
tool posts elsewhere. Claude Tag acknowledged in the current thread and answered in
the other one ("answers incoming in their threads").

**Design:**
1. New local tool `post_to_thread` (registered alongside the history/react tools):
   params `thread_ts` (required — root ts of the target conversation; a top-level
   message ts targets its thread), `text` (required). CURRENT CHANNEL ONLY — no
   channel_id param; cross-channel posting is out of scope (write-boundary, unlike
   read tools).
2. Executor path: converts markdown exactly like a normal reply, posts via the
   standard messaging layer into the target thread, calls
   `channel_pulse.record_own_reply` so the rings stay truthful. No unprompted
   accounting (it runs inside an addressed/judged turn). Never raises — refusals
   return {"ok": False, ...}.
3. Rails (cheap, semantic): target thread muted by the user (F15 muted_threads) →
   refuse with "thread_muted_by_user" (the mute means "stop contributing there"; the
   model relays that instead of violating it). Empty text / missing thread_ts →
   refuse. Posting to the CURRENT thread → refuse with a hint to just reply normally
   (prevents double-posting).
4. Tool description teaches the Claude-Tag pattern: use when a reply belongs in a
   DIFFERENT conversation (user asked you to answer a message elsewhere, or you're
   closing a loop you were part of); acknowledge briefly in the current thread rather
   than duplicating the content in both.
5. LOCAL_TOOLS_GUIDANCE: one short bullet on the same judgment.

**Tests:** schema registered + required params; executor posts markdown-converted
text to the target thread; muted-thread refusal; current-thread refusal; empty-text
refusal; record_own_reply called with the target thread; used-tools provenance line
includes post_to_thread; never-raises contract on Slack API failure.

## F24. Reaction-preference rule (user directive 2026-07-11)

**Gap (observed live):** "Hey guys, please respond to any user requests while I'm out,
brb" — Claude Tag acknowledged with a single 👍; our bot posted a full sentence. User:
"if a msg can be responded to with a simple reaction, that's all that's needed." The
react verdict exists (F20) but nothing states a PREFERENCE for it when a reaction fully
carries the reply, and "aimed at the assistant" biases the classifier toward respond.

**Design (prompt-only, two layers):**
1. PARTICIPATION_SYSTEM_PROMPT: add to the react/respond judgment — when a single
   emoji fully carries the needed reply (a "got it" to an instruction or delegation,
   an FYI, agreement that needs no elaboration), prefer "react" over "respond"; words
   are for when they ADD something (information, an answer, a real question back).
   If another person or agent has already acknowledged with a reaction, a text reply
   restating it is noise — react likewise or stay silent.
2. LOCAL_TOOLS_GUIDANCE (addressed turns): broaden the existing reaction-only rule
   beyond "thanks!" to the same cases — acknowledgments, delegations, FYIs ("please
   handle X while I'm out" → 👍 + empty text).
No new config; the intent classifier is NOT touched (cache budget).

**Tests:** prompt substrings at both layers (preference wording present; delegation/FYI
example present; redundant-acknowledgment rule present).

## Rollout / verification

1. `make test` green after each change set (F4 → F1 → F2 → F3); `make lint` clean.
2. Four independently reviewable change sets; commits/releases only on user request.
3. Live dev-bot pass in #chatgpt-bot-test (C04QDHE8W8M, authorized): generate an image
   and chat mid-generation (F1+F4 visible); unprompted message the bot should skip (F2
   logs a no_reply reason); @-mention vs ambient wake visible in request logs (F3).
