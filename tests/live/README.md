# The live battery — run book

Thirteen rows against real Slack with the **dev** bot (`SHALLOW_STREAM_RESPEC` §7, §9). Prod is
hands-off; the battery runs in **`C0BKX77NU66`** (`#chatgpt-bot-test`) and nowhere else — the
runner **refuses any other `--channel`**, and there is no override flag.

```bash
source .venv/bin/activate
python3 -m tests.live.run_battery                       # all thirteen rows
python3 -m tests.live.run_battery --rows value-floor-holds,thanks-response-choice
python3 -m tests.live.run_battery --out /tmp/battery.json
```

The exit code is the verdict: **nonzero unless every executed row is a clean `pass`.**

---

## The two rules that shape everything here

**1. Nothing is deleted.** Every message the battery posts stays in the channel, and so does every
reply the bot makes to it. The owner watches these runs live and reads the room afterwards, so a
harness that tidied up behind itself would be erasing what they are reading — and the deleting
version also took each failed assertion's evidence with it before anyone could look. Rows still
record every ts they seeded and every ts they observed; that list is in the report, and it is how
you find a run's messages without searching for text.

The one thing a row puts back is **durable bot state** — today only row 8's `channel_window_anchor`
— because that is configuration the next turn reads rather than conversation. A restore that
cannot be applied downgrades the row to `unrestored`.

**2. Nothing the harness posts looks like a test.** No token-shaped nonce, no `ALL-CAPS` marker
word, no sentence about probes or batteries. Every seed reads as a message a coworker would type.
Where a row needs a fact the bot cannot already know, the fact is a natural one: a supplier that
does not exist (`Kestwood Freight`), a quantity nobody has quoted (`8,472 crates`), a quoted price,
a weekday. They are minted from the row's nonce in `battery_harness` — `vendor_name`, `quantity`,
`money`, `chatter_lines` — so a run is reproducible from the nonce in its report while the room
never sees a token.

Correlation does not depend on any of that text: a row grades the turn its trigger started, found
by `turn_id` and read through that turn's own `outbound_receipts` rows. Grading compares
**digit-normalized** for numbers (`847,800` satisfies a seeded `847800` — a verbatim compare once
failed a correct answer over the comma) and case-insensitively, on word boundaries, for names.

**A BURIED FACT IS A PAIR: a supplier AND a figure, seeded in one sentence, both asked for by the
trigger.** A name alone was never sufficient evidence that information flowed — `vendor_name` has
~9,000 outputs, the bulk chatter mints from the same space, and nothing is ever deleted, so given
enough runs one half turns up in the window by coincidence and the row credits a hit it never had.
The pair is ~10⁸, so a false pass needs two independent collisions on the same run. The chatter
guard still excludes both halves for the running row; it is noise control, not the proof.

## What a row may assert, and what it may only record

**Machinery is assertable; the model's free choices are not** (owner ruling, 2026-08-03). The
stream, the window, receipts, the anchor and the tuned wake/silence gate are built and tuned here,
so a row may hold them to a contract. How the responder expresses itself once it is awake — words
or an emoji, long or short — is judgement it is supposed to have, and a harness that graded it
would be failing the bot for using it.

| Row | Class | What it grades |
|---|---|---|
| `cross-thread-awareness` | machinery | the fact reached the window; no search tool was needed |
| `verification-rule` | machinery | a search ran and the buried **pair** came back — supplier and figure |
| `cross-thread-action` | machinery | the answer landed under C and nowhere else. What the bot did with the open question first is **recorded, not graded** |
| `search-to-action` | machinery | search → a post under the target root carrying the seeded **pair** |
| `full-origin-fidelity` | machinery | the whole origin thread rendered (`origin_count`), and the root's **pair** came back |
| `stream-currency` | machinery | `H` is pinned at admission and the next turn moves it |
| `in-flight-exclusion` | machinery | receipt state across two renders |
| `re-anchor-observable` | machinery | the build reselected and the floor moved |
| `foreign-exchange-bait` | machinery (gate) | our bot stays out of two other parties' exchange |
| `directed-banter-answered` | **model choice** | only that it **responded** — message or reaction. The form is recorded |
| `thanks-response-choice` | **model choice** | **nothing. It always passes** and records what the bot chose |
| `value-floor-holds` | machinery (gate) | an undirected aside gets no message (silence or emoji, its pick) |
| `render-equality-probe` | machinery | one periphery, two origins, identical prefix |

An observation-only row declares itself in the registry (`observation_only=True`) and carries an
empty `assertions` tuple. Everywhere else an empty one is refused, because a row that grades
nothing by accident reports `pass` for having done nothing.

## Before you start the bot

**`DEV_TREAT_BOT_IDS_AS_HUMAN` must contain BOTH our app's USER-TOKEN bot record and Claude Tag's**,
set in the bot's `.env` **BEFORE the bot process starts** — changing it requires a bot restart. The
harness is a separate process and the bot reads that list once at import, so nothing here can set
it for you.

The user-token record is **not** what `auth.test` returns. A Slack app owns two bot records: the
one a user-token `chat.postMessage` carries (our `app_id`, no `user_id`) and the one the bot token
carries (our `app_id`, with `user_id`). Every seed here is posted with the user token, so it is the
first that needs the carve-out; the bot-token id is inert in that list because `is_own_message`
matches it first. Configuring the "obvious" id voided a whole pass on 2026-08-01 — 442 gate events
classified the operator as `other_bot`, silently changing what the gate saw in every row.

The preflight therefore partitions the allowlist **by `app_id` via `bots.info`**, and aborts unless:

| Violation | Why it aborts |
|---|---|
| no user-token record for our app | every seed classifies as a bot, and every row's gate input is wrong |
| zero foreign entries | Claude Tag is not listed, and row 9a's second human does not exist |
| two or more foreign entries | the battery would be guessing which third party it grades |

**DO NOT GATE A PASS ON `⚡️ Bolt app is running!`.** That line can arrive **minutes** after the bot
is already serving — measured at 4+ minutes on the 2026-08-03 18:30 boot against ~2 on a healthy
one — so a `grep || exit` after a 30-second sleep aborts against a working bot. Check the process
is up, identity resolved and no ERROR/CRITICAL, then prove liveness with the probe below: a log
line never showed that events are being received, and the probe does.

The preflight proves the **file** says so, not that the live process loaded it. Prove the process
before every armed pass:

```bash
python3 -m tests.live.classify_probe      # 0 = human, 1 = classified as a bot, 2 = never judged
```

It posts one ordinary low-value aside as the operator and reads that message's own `gate_start`
out of `logs/participation.jsonl`; `sender_type` must be `human`. The remark is worth nothing and
addressed to nobody, so the gate declines it and it costs a classification rather than a turn. Its
message stays in the channel like everything else.

**The channel must not be fenced.** The preflight reads the durable `epoch_fence_lease` row and
**fails closed**: the only ways past are a recognised `released` row, or a recognised busy row
whose expiry parses and has passed. An unparseable expiry or a state this harness does not
recognise refuses, because damaged evidence is not proof the channel is free. Every row here is
written to run unfenced — under a fence the settings, memory, steering and window-anchor reads it
grades are served from an in-memory overlay instead of SQLite. An `invalidated` lease is cleared
only by a human who has looked at why the last battery died.

## Barriers: one seam per pass, in the process environment

**`post_admission` fires on EVERY turn that builds a stream**, so a seam armed for the whole run
adds the full `DEV_TURN_BARRIERS_TIMEOUT` to every waking turn — measured at 130.8s round trip
against a 180s reply deadline. Gate-declined messages never reach the seam, so bulk chatter is
unaffected, but every row's trigger is. `DEV_TURN_BARRIERS` is read at boot, so the split is a
restart per seam:

1. seams **unset** → every row except `stream-currency` and `in-flight-exclusion`
2. restart with `DEV_TURN_BARRIERS=post_admission` → `--rows stream-currency`
3. restart with `DEV_TURN_BARRIERS=post_partial_post` → `--rows in-flight-exclusion`

**One seam at a time, never both**: the seams are process-global, so with `post_admission` armed,
row 7's ordinary B and C turns would freeze at a seam row 7 never releases.

**Both barrier variables live in the LAUNCH ENVIRONMENT, never in `.env`.** In `.env` they survive
every future bot start, and a casually started bot then stalls 120s on every waking turn. Pass them
on the command that starts the bot, with `DEV_TURN_BARRIERS_DIR` pointing at a directory the
harness can also write to (`data/barriers`).

**A turn held at a seam owns a message nobody can modify.** Measured live: while a turn sits at
`post_partial_post`, its reply refuses `chat.update` with **`streaming_state_conflict`** and
`chat.delete` with **`cant_delete_message`** on *both* tokens — it is mid-stream. Releasing the
barrier lets the turn finish. Rows 6 and 7 release in a `finally` for exactly that reason: a row
that dies holding a barrier leaves the bot paused until its own timeout.

## Tokens the harness reads

| Variable | Used for |
|---|---|
| `SLACK_TEST_USER_TOKEN` | **posts every seed, as a human.** The bot token cannot trigger the bot |
| `SLACK_BOT_TOKEN` | reads history, replies, reactions and identities |

Missing either is a **named startup failure** before anything is posted, not a row failure fifty
seconds later. No second user token exists or is needed: the second human is Claude Tag.

The harness also reads the bot's own `config` for `database_dir` (receipts, recorded tool
provenance, the window anchor) and `log_directory` (the participation ledger, `participation.jsonl`
plus rotations `.1`…`.5` — a long run can rotate mid-battery).

## Small-window battery mode — run it this way

**Owner ruling, 2026-08-03: *"these tests with 100s of msgs are way too much."*** The bulk seeding
existed for one reason — to push a seeded fact below the rendered window floor — and the floor is
env-tunable, so the battery shrinks the window instead of flooding the channel.

```bash
# the BOT, launched with the small window in its PROCESS ENVIRONMENT (never .env)
CHANNEL_WINDOW_TARGET=8 CHANNEL_WINDOW_CEILING=12 python3 slackbot.py

# the HARNESS, with the SAME two variables, because every count is computed from them
CHANNEL_WINDOW_TARGET=8 CHANNEL_WINDOW_CEILING=12 python3 -m tests.live.run_battery --rows …
```

| Row | at the shipped 50/100 | at 8/12 |
|---|---|---|
| `verification-rule` | 1 fact + 101 chatter + 1 trigger = **103** | 1 + 13 + 1 = **15** |
| `search-to-action` | 1 root + 1 fact + 101 chatter + 1 trigger = **104** | **16** |
| `full-origin-fidelity` | 1 root + 120 replies + 1 trigger = **122** | 1 + 24 + 1 = **26** |
| `re-anchor-observable` | 101 roots + 1 trigger = **102** | **14** |
| **total for those four** | **431** | **71** |

**Not one assertion changes.** Same production code path, same below-the-floor semantics; the
counts were always `CEILING + 1` computed from the resolved config, and row 5's depth is now
derived too (`origin_reply_count()` = 2 × ceiling, capped at the historical 120, so the shipped
window still seeds the 120 it always did).

**BOTH PROCESSES NEED THE SAME TWO VARIABLES.** The harness cannot read the bot's environment, so
a mismatch is not detectable up front — it shows up as row 8's `root_count` assertion failing
against a target it did not expect. Every row's `evidence.window` records what the harness
resolved, and the runner prints it at startup: **a pass at 8/12 is a pass at 8/12** and must never
be read as a pass at 50/100.

### Clean up the anchor when you put the window back

**A compact pass leaves the test channel anchored shallow, and restarting the bot does not undo
it.** Rows 2, 4 and 5 advance `channel_window_anchor` the way any ordinary turn does — only row 8
registers a restore, because only row 8 sets out to move the floor — so a build during a compact
pass writes a floor chosen against a 12-root ceiling. **The floor never moves backward.** Measured
on 2026-08-03: after restoring the shipped window and restarting, the next turn still rendered
`root_count = 11` against a floor from the middle of the compact run.

Other channels are untouched — the anchor is keyed `(team_id, channel_id)` — but the test channel
renders a shallow window until it accumulates roots again. Two honest options:

```sql
-- the remedy: one row, so the next build reselects cold at the shipped window
DELETE FROM channel_window_anchor WHERE channel_id = 'C0BKX77NU66';
```

or accept the shallow floor and let it self-heal as the channel fills. Deleting is derived
internal state, not room content, so it sits inside the harness's remit rather than under the
no-deletion ruling, which is about messages. **Verify either way with one natural turn**: after
the delete, the 2026-08-03 check rendered `root_count = 50` with `reselected: true` and a floor
older than the compact run's. Back the table up first — the battery's own restores do.

## What a run costs in wall time

Seeding is paced at one message per second, which is Slack's documented `chat.postMessage`
guidance. **This is expected and is not a hang.**

| Row | Seeds | ~seeding |
|---|---|---|
| `verification-rule` | 1 decision + `CEILING + 1` chatter lines + 1 trigger | ~101s |
| `search-to-action` | 1 root + 1 fact reply + `CEILING + 1` chatter + 1 trigger | ~101s |
| `full-origin-fidelity` | 1 root + 120 replies + 1 trigger | ~120s |
| `re-anchor-observable` | `CEILING + 1` roots + 1 mentioned trigger | ~101s |

**ROWS 2 AND 4 CURRENTLY FAIL, AND THE CAUSE IS OPEN.** Both bury a fact and ask a question only
that fact answers; both get an honest "I couldn't find any Slack record of that" from a bot whose
log shows `search_slack` running and returning `ok`. Measured 2026-08-03: adding a **782-second**
settle before asking changed nothing, so the age of the fact is not the variable. Neither of our
tokens holds `search:read` (`search.messages` answers `missing_scope`), so the workspace index
cannot be queried to see whether the message is there at all. Row 4's own answer is the sharpest
clue on record — *"the cert-renewal details appear to be in a channel unavailable from this
thread"* — which points at what the search path returns for this channel rather than at the
harness. Unresolved; do not tune the rows around it.

The counts are **formulas computed from the resolved config**, never the literal 101 — raising
`CHANNEL_WINDOW_CEILING` changes the seeding, the totals and the assertions together. The chatter
lines are deliberately low-value statements, never questions and never mentions, so the gate
declines them; and a line is re-minted if it happens to state a number the row grades on, which
would otherwise put the graded fact above the window floor.

## Reading a row's result

| Status | Means |
|---|---|
| `pass` | every assertion held |
| `unrestored` | every assertion held, but durable state the row changed could not be put back. Row 8 is the only row that can report it: its restore is compare-and-restore, so if a legitimate turn advanced the anchor after us we leave it alone and say so |
| | An observation-only row has no assertions to hold, so it reaches `pass` unless its restore fails or the harness breaks — that is its contract, not a gap |
| `fail` | an assertion did not hold |
| `error` | the harness broke — a poller raise, a correlation failure, or a row whose premise never held |
| `skipped` | **only** via an explicit `--rows`. A row never skips itself |

Answers are read through each turn's own `outbound_receipts` rows, never "a bot message that
arrived after my trigger": in a shared channel a time window matches another conversation's reply.
The receipts are read **after the turn's `turn_outcome` lands** — the bot's own completion fence —
because "nothing is in flight at this instant" is also true of a chrome-only snapshot and of a
split reply's first part.

**The outcome is a necessary fence, not a sufficient one**, and a row can fail on that. Two
production paths emit `turn_outcome` without settled receipts: a turn that could not revoke its own
effects after a failed flight drain deliberately does not settle, and `settle_ledger` hands its work
to the drain worker when its ten-second timeout fires. If receipts are still `in_flight` when the
row's bound expires the harness **raises** — grading half a reply would score the bot on words it
had not finished. Both halves of that wait share **one declared deadline**, never two in sequence.

**And the surface list has a bound the harness cannot close.** The receipt drain worker retries
every two seconds indefinitely, so no finite number of identical polls proves the queue has
drained: a receipt landing after the stability window is absent from the row's `observed_ts` and
its text was never read. Every row that reads a turn this way carries that limitation verbatim in
its `observations`, so a green row never implies a completeness guarantee it cannot give.

## The report

One JSON array, one object per row, at `--out`. Each object carries `row`, `status`, `nonce`,
`started_at`, `finished_at`, `seeded_ts`, `observed_ts`, `external_ts`, `evidence`, `assertions`,
`observations`, `cleanup` (`restored` / `restore_failures`) and `notes`.

`evidence.observed_text` holds the exact reply text every assertion graded, keyed by turn. That is
what makes a failure diagnosable: the 2026-08-01 report said only that a reply "lacked the seeded
decision", which left no way to tell a bot that had found the fact and summarised the number away
from one that never found it.

`external_ts` is another app's messages — Claude Tag's replies in row 9a. They are listed so a
reader can tell them from ours, and they change no status.

**`observations` are not assertions.** They carry what a row saw in a direction it cannot soundly
grade — row 1's tool reading when no provenance row appeared, over a store that writes no row at
all for a zero-tool turn and therefore cannot distinguish "no tools" from "no write". **Nothing in
`observations` can change a row's status.** The same read in the other direction *is* an assertion:
a provenance row that **names** a search or history tool fails row 1, because a written name is
authoritative.

## Where the network-free tests live

`tests/unit/test_battery_harness.py`, deliberately — `make test` runs `pytest tests/unit` and
nothing else, so a harness test anywhere else would sit outside the capped gate and rot unnoticed.
`tests/live/` holds only code that talks to Slack. Never run pytest uncapped here; see the
`run-tests` skill.
