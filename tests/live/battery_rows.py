"""The THIRTEEN live battery rows (SHALLOW_STREAM_RESPEC §9's registry).

One function per row, each declaring its own setup, assertions and what durable state it puts
back. Every exchange below is ADAPTED from the ones the shipped P2 and P3 batteries actually ran.

**EVERY MESSAGE THIS FILE POSTS READS AS A COWORKER TALKING** (owner ruling, 2026-08-02). No
token-shaped nonce, no ALL-CAPS marker word, no sentence about tests, probes or batteries. The
owner watches these runs happen in the room, so a channel full of `searchlag1785728257` is a
channel nobody can read. Where a row needs a fact the bot could not possibly know already, the
fact is a NATURAL one — a supplier that does not exist, a quantity nobody has quoted, a date —
minted from the row's nonce by `battery_harness`, which is why the nonce itself never appears in
anything posted.

**CORRELATION IS STRUCTURAL, NOT TEXTUAL.** A row grades the turn it triggered, resolved through
`turn_id` and that turn's own `outbound_receipts` rows (`observe_turn_output`) — never "a bot
message that arrived after my trigger", which in a shared channel matches another conversation's
reply. That is what makes unmarked text safe to grade: nothing is found by searching for a token.

**NUMBERS ARE COMPARED DIGIT-NORMALIZED** (`states_number`). The 2026-08-02 run seeded `847800`
crates, the bot answered "847,800 crates.", and a verbatim compare graded a correct answer as a
failure. Punctuation is the writer's choice; the number is the fact. Names are compared
case-insensitively (`states_phrase`).

**NOTHING IS DELETED.** Rows leave their messages in the channel. The only thing a row puts back
is durable bot state — row 8's window anchor — because that is configuration the next turn reads.

**EVERY ROW IS IN ONE OF TWO CLASSES, AND THE CLASS DECIDES WHAT MAY BE ASSERTED** (owner ruling,
2026-08-03):

* **MACHINERY** — the stream, the window, receipts, the anchor, and the tuned wake/silence gate.
  These are built and tuned by us, so a row may assert them outright. Rows 1-8, 9a and 9d.
* **MODEL CHOICE** — how the responder expresses itself once it is awake: words or an emoji, long
  or short, or nothing at all. **A row may RECORD these and must never GRADE them.** Rows 9b and
  9c, the second of which asserts nothing at all and always passes.

The line runs between "did the machinery wake and put something in the room" (assertable) and
"what did it say" (observable). Row 9b used to assert `turn_outcome.kind == "reply"`, so a
perfectly good 🌭 on a joke would have failed it; row 9c used to demand an emoji and no words.
Both were the harness deciding on the model's behalf, which is what the ruling removes.

**EVERY ROW HERE RUNS UNFENCED, and the preflight ENFORCES it** (`assert_channel_unfenced`).
Under an epoch fence, channel settings, memory, steering and the window anchor are served from an
in-memory overlay rather than from SQLite — so a row that reads the anchor (8), reads recorded
tool provenance (1, 2, 4) or grades receipt membership (7) would be measuring the overlay's
baseline instead of the world the bot actually answered from.

**RUN THE BATTERY IN SMALL-WINDOW MODE** (owner ruling, 2026-08-03: *"these tests with 100s of
msgs are way too much"*). The bulk seeding exists for exactly one purpose — to push a fact below
the rendered window floor — and the floor is env-tunable, so the fix is to shrink the window
rather than flood the room. Launch the BOT with `CHANNEL_WINDOW_TARGET=8` and
`CHANNEL_WINDOW_CEILING=12` in its process environment (never `.env`, same discipline as the
barrier seams) and give the harness the SAME two variables, because every count here is computed
from them at runtime. Rows 2, 4 and 8 then seed ~13 messages instead of ~101 and row 5 seeds ~24
replies instead of 120 — one-eighth the traffic, the same production path, the same
below-the-floor semantics, and not one assertion changed.

**THE REPORT RECORDS THE WINDOW IT RAN AGAINST** (`evidence.window`). A pass at 8/12 is a pass at
8/12; it must never be read as a pass at the shipped 50/100.
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from config import BotConfig
from message_processor.dev_barriers import POST_ADMISSION, POST_PARTIAL_POST

from tests.live.battery_harness import (REPLY_DEADLINE_SECONDS, REPLY_POLL_SECONDS,
                                        SLOW_TURN_DEADLINE_SECONDS,
                                        THIRD_PARTY_REPLY_DEADLINE_SECONDS, HarnessError,
                                        HarnessPreflightError, Observed, PartyIdentity,
                                        ProvenanceRead, Restore, RowContext, await_tools_used_for,
                                        await_thread_visible, bot_identity, bot_team_id,
                                        chatter_lines, clients,
                                        find_turn_id, harness_user_display_names,
                                        harness_user_identity,
                                        is_search_or_history_name, money, numeric_token,
                                        origin_ack_violation,
                                        project_phrase,
                                        observe_turn_output, pace, post_seed,
                                        quantity, read_receipt_state, read_window_anchor,
                                        freeing_other_turns, record_observed, release_barrier,
                                        seed_messages,
                                        states_number, states_phrase, ts_ge, ts_lt, vendor_name,
                                        weekday, await_in_flight_surface, await_trigger_verdict,
                                        barrier_operation, wait_barrier_reached,
                                        wait_bot_reaction, wait_bot_reply, wait_for_telemetry)

# ROW 2 NO LONGER SCANS FOR DENIAL PHRASES (codex, 2026-08-03). The list was context-blind: it
# read the whole answer for fragments like "no mention of", so a correct reply — "No mention of a
# delivery date, but the accepted quote was $41,770" — failed for a sentence that was true and
# helpful. The false negative it was written against is already covered without prose scanning: an
# answer that carries neither the seeded decision NOR a recorded search or history tool fails on
# both remaining assertions, which is exactly the "denied it without looking" shape.

# Row 9a's bait, VARIED BETWEEN RUNS. Claude Tag's replies are another app's messages that stay in
# the channel, so identical bait every run would leave a pile that reads as one long repeated
# conversation to any model that later renders the window. The nonce picks the variant; the words
# themselves are ordinary.
_BAIT_VARIANTS = (
    "what do you reckon — worth calling the vendor?",
    "any idea whether that's worth chasing with the supplier?",
    "would you bother raising a ticket for it?",
    "does that sound like something to escalate?",
)

# Row 9d's aside: low value, addressed to nobody, and varied so a run does not repeat the last
# run's words verbatim.
_ASIDE_VARIANTS = (
    "ha, classic", "ha, amazing", "that's about right", "of course it did", "every single time",
)


@dataclass(frozen=True)
class BatteryRow:
    """One registry entry: its name, the SHAPE of the trigger it posts, and what it grades.

    `trigger_template` is documentation — the sentence a reader should expect to find in the
    channel, with the run's minted facts shown as slots. `assertions` names the graded claims up
    front so the registry can be checked for completeness without running anything: a row in §9's
    table with no implementation, or an implementation that grades nothing, is a row that would
    report `pass` for having done nothing.
    """

    name: str
    trigger_template: str
    assertions: Tuple[str, ...]
    run: Callable[[RowContext], Awaitable[None]]
    slow: bool = False   # seeds in bulk; expect minutes, not seconds
    # OBSERVATION-ONLY: the row records what the bot chose and grades nothing, so an empty
    # `assertions` is its contract rather than an unfinished implementation. The registry check
    # refuses an empty one otherwise, which is what stops a row passing for having done nothing.
    observation_only: bool = False


def window_ceiling() -> int:
    """Read at RUNTIME from the resolved config, never hardcoded as 101.

    `CEILING + 1` guaranteed-eligible roots is the smallest number that provably pushes an older
    root out of the window. Raising the ceiling must change the seeding, the totals and the
    assertions together.
    """
    return BotConfig().channel_window_ceiling


def window_target() -> int:
    return BotConfig().channel_window_target


# The deepest origin thread the battery has ever seeded, and the value row 5 used when it was
# written. It is a CEILING now rather than the count: small-window mode scales the thread down
# with the window, and this stops the derivation from growing the thread past what the row has
# historically proven when someone runs at a large window.
ORIGIN_REPLY_CAP = 120


def origin_reply_count() -> int:
    """How many replies row 5 seeds under its origin root — DERIVED, never a literal 120.

    The row's claim is that the WHOLE origin thread renders, `origin_count` exact. That needs a
    thread comfortably deeper than the window, not a specific depth: multi-page reply pagination
    is pinned in the unit tests against fixtures, so the live row does not have to re-prove
    page-walking with a hundred real messages. Two per window ceiling gives ~24 replies at the
    battery's small-window setting and leaves the historical 120 untouched at the shipped one.
    """
    return max(4, min(ORIGIN_REPLY_CAP, 2 * window_ceiling()))


def bait_for(nonce: str) -> str:
    """One of `_BAIT_VARIANTS`, chosen by the nonce so two runs rarely post the same words."""
    return _BAIT_VARIANTS[sum(ord(c) for c in nonce) % len(_BAIT_VARIANTS)]


def aside_for(nonce: str) -> str:
    """Row 9d's throwaway line. Same reason as the bait: vary it, and keep it worth nothing."""
    return _ASIDE_VARIANTS[sum(ord(c) for c in nonce) % len(_ASIDE_VARIANTS)]


async def _observe(ctx: RowContext, observations: Sequence[Observed]) -> None:
    """Route what a wait SAW into the report's buckets, by author."""
    await record_observed(ctx, observations, ours=await bot_identity(),
                          operator=await harness_user_identity())


async def _turn_output(ctx: RowContext, trigger_ts: str, *,
                       deadline: float = REPLY_DEADLINE_SECONDS) -> Tuple[str, List[Observed]]:
    """The turn our trigger started, and everything it posted — correlated, never guessed.

    Returns `(turn_id, prose)`. Every surface the turn owns, chrome included, is recorded in the
    report as a side effect; only the prose comes back, because "did the bot answer" is a question
    about words and a thinking indicator is not an answer.
    """
    turn_id = await find_turn_id(ctx.channel, trigger_ts, deadline=deadline)
    ctx.evidence["turn_id"] = turn_id
    prose = await observe_turn_output(ctx.team_id, ctx.channel, turn_id,
                                      deadline=deadline, ctx=ctx)
    return turn_id, prose


async def _posted_by(ctx: RowContext, trigger_ts: str, verdict) -> List[Observed]:
    """What this trigger actually PUT IN THE ROOM, whichever shape the ledger recorded.

    A declined trigger opened no turn, so it owns no receipts and therefore posted nothing —
    that is a fact about the store, not an assumption. A woken one is read through its receipts
    exactly as every other row reads an answer.
    """
    if not verdict.woke or not verdict.turn_id:
        return []
    return await observe_turn_output(ctx.team_id, ctx.channel, verdict.turn_id, ctx=ctx)


async def _require_turn_output(ctx: RowContext, trigger_ts: str, *, what: str,
                               deadline: float = REPLY_DEADLINE_SECONDS) -> List[Observed]:
    """A SETUP step the row's premise depends on. Its absence is an `error`, not a `fail`.

    A row that never got the exchange it grades has not measured restraint or anything else, and
    reporting that as a failed assertion would be scoring the bot on a conversation that never
    happened.
    """
    _, prose = await _turn_output(ctx, trigger_ts, deadline=deadline)
    if not prose:
        raise HarnessError(f"{ctx.row}: {what} never arrived within {deadline}s; the row's "
                           f"premise did not hold, so nothing was measured")
    return prose


# The settings-button chrome Slack flattens into a fetched reply's text (":gear: gpt-5.6-sol
# button"). It is the BOT'S surface, not the model's words, and it broke the first live ack
# grading twice over: "5.6" is a digit and "button" is not a receipt. The pattern matches the
# TRANSPORTED STRUCTURE — gear, a MODEL label (gpt- followed by a version digit, the shape every
# selectable model id has), the word button, at end of text — not any gear-adjacent sentence, so
# ":gear: settings button" and ":gear: gpt-settings button" both survive (codex ack #3/#4).
# Content predicates are exact-match, so removing chrome can only remove noise.
_REPLY_CHROME = re.compile(r"\s*:gear:\s*gpt-\d[\w.\-]*\s+button\s*$")


def _text_of(observations: Sequence[Observed]) -> str:
    return "\n".join(_REPLY_CHROME.sub("", o.text) for o in observations)


def _destinations(outcome: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = outcome.get("destinations") or []
    return [d for d in raw if isinstance(d, dict)]


def _posts_under(outcome: Dict[str, Any], root_ts: str) -> List[Dict[str, Any]]:
    return [d for d in _destinations(outcome)
            if d.get("kind") == "post_to_thread" and d.get("thread_root_ts") == root_ts]


# ------------------------------------------------------- row 4's premise: subject and guard
#
# THE ROW CONTAMINATED ITSELF ACROSS RUNS (2026-08-04). The channel keeps everything — nothing is
# ever deleted — so a question worded the same way on every run is answerable from the LAST run's
# material: the trigger that supplied the previous figure is still in history, and so is whatever
# this row's own turn stored with remember_fact. On the run that caught it, the seed-time turn
# answered trial 2's question with trial 1's cap, which closed the obligation before the graded
# trigger ever ran; the row went on measuring a thread that no longer owed anything.
#
# Two independent repairs, because either alone leaves a hole. The SUBJECT varies per run, so a
# stored answer to last run's question is an answer to a different question. And the guard reads
# the seed-time reply for ANY cap-shaped figure rather than only this run's, so a stale answer
# that slips through — from residue nobody cleared, or from a source we have not thought of —
# stops the row as an invalid premise instead of being graded as a model failure.
#
# The row's own remember_fact residue is EXPECTED, not a defect: the runner clears it after the
# pass. The guard exists for the residue that survives anyway.
ROW4_SUBJECTS: Tuple[Tuple[str, str], ...] = (
    ("renewals cap", "cert renewal"),
    ("freight surcharge ceiling", "pallet shipment"),
    ("storage rate", "overflow warehouse"),
    ("inspection fee", "line-3 recertification"),
    ("courier rate", "sample courier run"),
    ("permit fee", "loading dock permit"),
    ("calibration fee", "scale recalibration"),
    ("disposal rate", "solvent disposal"),
)


def row4_subject(nonce: str) -> Tuple[str, str]:
    """The (subject, work) pair this run's buried ask is about — deterministic from the nonce.

    Deterministic for the same reason every other seeded word is: the report's nonce has to
    reproduce the exact sentences the run posted, because the channel is the only record. The
    PAIR moves together so the whole scenario is consistent — a run asking about the storage rate
    asks it about the overflow warehouse, in both the buried question and the news that settles it.
    """
    index = int(numeric_token(f"{nonce}|row4-subject", digits=6)) % len(ROW4_SUBJECTS)
    return ROW4_SUBJECTS[index]


# A written price: "$79,822", "$9,400", "$41770.00". Deliberately narrow — a bare "$0" or a year
# is not a cap-shaped figure, and widening this to every number would trip on the run's own
# timestamps and counts.
_MONEY_SHAPED = re.compile(r"\$\s?\d{1,3}(?:,\d{3})+(?:\.\d{2})?|\$\s?\d{4,}(?:\.\d{2})?")


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text or "")


def stale_cap_figure(reply_text: str, *, quoted: str) -> Optional[str]:
    """The figure in a seed-time reply that means the obligation was already closed, or None.

    ANY cap-shaped money figure counts, not just this run's cap — that narrower check is exactly
    what let the contamination through, since the figure that closed the question came from the
    PREVIOUS run and matched nothing this run had minted.

    THE ONE FIGURE THAT IS ALLOWED IS THE QUOTE, and it has to be: the quote is stated in the
    question itself, so a reply that repeats it ("I don't have the cap — they quoted $9,400") has
    echoed the ask rather than answered it. Failing that would make the honest seed-time reply an
    invalid premise, which is the opposite of what this guard is for.
    """
    wanted = _digits(quoted)
    for match in _MONEY_SHAPED.finditer(reply_text or ""):
        if _digits(match.group(0)) != wanted:
            return match.group(0).strip()
    return None


# --------------------------------------------------------------------------- row 4's grader

def origin_surface_ok(*, outcome_kind: str, spoke_in_origin: bool, origin_text: str,
                      fragments: Sequence[str] = (),
                      addressees: Sequence[str] = (),
                      addressee_ids: Sequence[str] = ()) -> bool:
    """The origin-surface half of the cross-thread contract, ack-shaped (owner ruling 2026-08-03).

    A silent origin still passes as before (`detached` or `reaction_only`, no origin destination).
    What is NEW is the third legal ending: a brief non-reporting acknowledgment to the person who
    handed over the piece — judged by `origin_ack_violation`, the same predicate the scenario
    oracle uses, so live and offline cannot drift. A turn whose telemetry says words landed in the
    origin but whose words the harness could not read back FAILS CLOSED: unreadable words cannot
    be certified as an ack.
    """
    words = (origin_text or "").strip()
    if (spoke_in_origin or outcome_kind == "reply") and not words:
        return False
    if words:
        return (origin_ack_violation(words, fragments=fragments, addressees=addressees,
                                     addressee_ids=addressee_ids) is None
                and outcome_kind in ("detached", "reaction_only", "reply"))
    return outcome_kind in ("detached", "reaction_only")


def grade_search_to_action(*, fact: str, cap: str, target_root: str,
                           origin_root: str, outcome_kind: str,
                           destinations: Sequence[Dict[str, Any]],
                           provenance: ProvenanceRead,
                           posted_text: str, origin_text: str = "",
                           origin_addressees: Sequence[str] = (),
                           origin_addressee_ids: Sequence[str] = ()) -> List[Tuple[str, bool]]:
    """Row 4's predicate, as a PURE function over evidence the harness can see.

    IT IS THE CROSS-THREAD ORACLE, NOT A ROUTING PREFERENCE (codex r2 #1). The first version of
    this counted destinations that had ALREADY been filtered to `{C}`, so "exactly one" was a
    claim about that thread alone: a turn could post the answer under `{C}`, post it again in a
    sibling thread, and narrate the whole thing in the origin, and still score four for four. The
    counting is global now, and the two clauses the scenario oracle makes that this one used to
    skip — one post in the WHOLE turn, and nothing said where the trigger was — are graded here.

    ONE CLAUSE OF THE SCENARIO ORACLE IS NOT AVAILABLE LIVE, and this is the honest statement of
    it rather than silent parity: REFUSED post attempts are invisible to the battery. The scenario
    harness sees every `post_to_thread` call and its result because it owns the executor;
    everything the live battery can read about a turn is what the room and the telemetry hold, and
    a refused call creates no destination record and no message to hang provenance on. So a turn
    that aimed at a second thread, was refused by the allowlist, and then posted correctly reads
    here as a clean pass. Fan-out that LANDED is caught; fan-out that was blocked is not.

    THE PREDICATE IS INFERENTIAL, AND IT PROVES MORE THAN A DIRECT ONE COULD. The harness cannot
    read a `search_slack` RESULT — local tool bookkeeping deliberately keeps name, ok and an
    argument-derived gist and no results at all — so "a hit whose ts was the target" is not
    observable. What IS observable is the tool NAME, the destinations, and the seeded words in the
    answer, and their CONJUNCTION proves the whole path:

    1. a search or history tool ran;
    2. exactly one post landed in the whole turn and it landed, committed, under `{C}` — inspected
       on `destinations[]`, since a post under a sibling thread is a different claim about where
       information went. Nothing the bot was told puts it there: `{C}` is where the question was
       asked and is still owed;
    3. the origin heard at most a brief non-reporting ACK — three legal surfaces, not one (codex
       r3 #1, then the owner's 2026-08-03 ruling). `detached` is the turn's own word for "a
       producer owned the surface and the Response was empty by design"; an empty response that
       calls both `post_to_thread` and `react_to_message` takes the reaction-only branch
       (message_processor/handlers/text.py) and is labelled `reaction_only` before the committed
       detached surface is ever consulted (main.py); and a turn that spoke in the origin reports
       `reply` — which now passes ONLY when its words survive `origin_ack_violation` (the shared
       predicate: full-match receipt grammar, no digits, no reporting stems, no caller-named
       answer fragments). The scenario oracle applies the same
       predicate, so the two surfaces cannot drift. Telemetry that says words landed but words the
       harness could not read back fail closed;
    4. the post carries the cap the trigger supplied (codex r2 #2). The people in `{C}` never saw
       the trigger, so the number they were waiting for is the answer, and it is the one thing in
       the post that can only have come from this turn.

    **THE LANDING IS THE READ-PROOF; THE PAIR CLAUSES ARE GONE** (first live run, 2026-08-03).
    The redesigned row's first live sample posted "Finance has now set the annual courier rate at
    $90,742" into the buried thread — a complete, natural answer — and failed two clauses that
    demanded it restate the asker's own supplier and quote back at them. Those clauses were
    verify-7's pair proof, carried over from the OLD row where the buried sentence was the
    payload the answer had to transport; in the redesign the payload flows the other way, the
    pair already stands in the target thread, and restating it is exactly the "repeats their own
    question back to them" shape this docstring warns about. The proof of reading is now the
    LANDING: the buried root is below the rendered floor and outside the trusted-root allowlist
    until a search or history tool returns it THIS turn, so a committed post under it (clause 3)
    plus the recorded tool name (clause 1) proves the turn went and found the thread — an exact
    root ts, with none of the collision space that made the pair necessary as text evidence. The
    supplier still guards the ORIGIN (an ack that names it is the answer leaking back) and the
    cap is digit-normalized via `states_number` — punctuation is the writer's.
    """
    tool_seen = provenance.row_present and any(is_search_or_history_name(name)
                                               for name in provenance.names)
    posts = [d for d in destinations if d.get("kind") == "post_to_thread"]
    only_post = posts[0] if len(posts) == 1 else None
    # An UNCOMMITTED destination is a delivery that started and did not finish, which is not the
    # post this row is about — so the target clause requires the state as well as the root.
    on_target = bool(only_post
                     and only_post.get("thread_root_ts") == target_root
                     and only_post.get("state") == "committed")
    spoke_in_origin = [d for d in destinations if d.get("thread_root_ts") == origin_root]
    return [
        ("a search or history tool name is recorded for the destination post", tool_seen),
        ("exactly one post_to_thread landed in the whole turn", len(posts) == 1),
        ("that post landed, committed, under the target root", on_target),
        ("the origin heard nothing, or only a brief non-reporting acknowledgment",
         # The supplier is the one non-numeric half of the seeded answer the origin never saw —
         # an ack that names it is the answer leaking back (codex, ack round #1).
         origin_surface_ok(outcome_kind=outcome_kind, spoke_in_origin=bool(spoke_in_origin),
                           origin_text=origin_text, fragments=(fact,),
                           addressees=origin_addressees,
                           addressee_ids=origin_addressee_ids)),
        ("the posted answer carries the cap the trigger supplied",
         states_number(posted_text, cap)),
    ]


def verification_decision(nonce: str) -> str:
    """Row 2's seeded FIGURE — run-unique, absent from the trigger, and written as money.

    A permanent literal like "Q3" is guessable and also sits in the channel's history from every
    previous run, so a tool call followed by a lucky guess — or a stale answer — would pass the row
    without the target ever being read. A quoted price is what a coworker actually writes down,
    and it carries the entropy half of the pair: the trigger names only the WORK, so neither this
    figure nor the supplier beside it can be echoed out of the question.
    """
    return money(nonce, "quote")


def grade_verification(*, decision: str, supplier: str, reply_text: str,
                       provenance: ProvenanceRead) -> List[Tuple[str, bool]]:
    """Row 2's predicate, pure so the contract can be exercised without Slack.

    "A tool ran and the bot said something" is not the contract. The seeded PAIR — which supplier
    and what figure — is what proves the bot went and looked; absence from the view is never
    evidence of absence from the channel, so a reply that has neither half nor a recorded lookup
    fails on every count.

    **BOTH HALVES, BECAUSE ONE IS NOT ENOUGH EVIDENCE** (codex, verify-7). Neither the supplier
    nor the figure appears in the trigger — the question names the WORK ("the loading dock
    resurfacing"), so the answer has to bring both from the message below the floor. A single
    half could be a coincidence out of an accumulating channel; the pair is ~10^8.

    THE PRICE IS COMPARED DIGIT-NORMALIZED. `$41,770`, `41770` and `41,770.00` are the same
    decision; only the digits are the fact — and `41,770.50` is a different one.

    NO PROSE IS SCANNED. The retired denial-marker list read the answer for fragments anywhere in
    it, which failed "No mention of a delivery date, but the accepted quote was $41,770" — a
    correct answer, punished for a sentence about something else.
    """
    return [
        ("a search or history tool was recorded for the reply",
         provenance.row_present and any(is_search_or_history_name(x) for x in provenance.names)),
        ("the reply names the supplier we chose", states_phrase(reply_text, supplier)),
        ("the reply carries this run's seeded figure", states_number(reply_text, decision)),
    ]


async def search_coverage_probe(ctx: RowContext, query: str, *,
                                trigger_ts: str) -> Dict[str, Any]:
    """Run the REAL channel search executor over this channel and record what it could reach.

    WHY A PROBE AND NOT AN ASSERTION ON THE TURN. The harness cannot read the model's own
    `search_slack` result — tool provenance deliberately keeps the tool NAME, `ok`, and an
    argument-derived gist with the query itself redacted, and no results at all — so from the
    reply alone "the model never searched" and "search ran and could not reach the target" are
    the same observation. This runs the same executor the turn had, with the row's own seeded
    keywords, and records the coverage block and the roots it retained beside the provenance the
    row already reads. Between the two, the failure modes separate.

    IT GRADES NOTHING and it never raises: an evidence probe that could fail a row would be
    grading by the back door. Read-only, on the bot token, and it posts nothing.
    """
    from types import SimpleNamespace

    from slack_client.history_tool import SlackHistoryToolMixin
    from slack_client.search_tool import SlackSearchToolMixin
    from tool_registry import ToolContext

    class _Probe(SlackSearchToolMixin, SlackHistoryToolMixin):
        """The two mixins as production composes them, over the harness's own bot client."""

        def __init__(self, client, team: str, bot_user: str) -> None:
            self.app = SimpleNamespace(client=client)
            self.self_team_id = team
            self.bot_user_id = bot_user

        def classify_sender(self, msg: Dict[str, Any]) -> str:
            if msg.get("user") == self.bot_user_id:
                return "self"
            return "other_bot" if msg.get("bot_id") else "human"

        async def resolve_usernames(self, ids, api_client=None):
            return {}   # the probe records ids; who they are is not what it measures

        def log_info(self, *a, **k): pass
        def log_debug(self, *a, **k): pass
        def log_warning(self, *a, **k): pass
        def log_error(self, *a, **k): pass

    try:
        bot = await bot_identity()
        operator = await harness_user_identity()
        probe = _Probe(clients().bot, await bot_team_id(), bot.user_id)
        tool_ctx = ToolContext(
            channel_id=ctx.channel, thread_ts=None, trigger_ts=trigger_ts,
            user_id=operator.user_id, is_dm=False, requester_is_human=True,
            # TRUE, not a shortcut: this operator posted the trigger into this channel a moment
            # ago, which is exactly what the attestation asserts.
            origin_membership_attested=True)
        payload = await probe.execute_search_tool(tool_ctx, {"query": query, "limit": 20})
    except Exception as e:  # noqa: BLE001 — evidence, never a verdict
        return {"query": query, "error": f"{type(e).__name__}: {e}"}
    results = [r for r in (payload.get("results") or ()) if isinstance(r, dict)]
    return {
        "query": query,
        "ok": payload.get("ok"),
        "error": payload.get("error"),
        "count": payload.get("count"),
        "coverage": payload.get("coverage"),
        # The ROOT each retained result would make actionable — a top-level hit stands for
        # itself. This is what says whether the scan reached the seeded thread at all.
        "roots": sorted({str(r.get("thread_ts") or r.get("ts")) for r in results}),
    }


def anchor_restore_for(render: Dict[str, Any],
                       floor_before: Optional[Tuple[str, int]]) -> Optional[Restore]:
    """Row 8's restore decision, pure — `None` when this build did not move the stored row.

    `reselected` says this build CHOSE a different floor; `anchor_advanced` says its write is the
    one that LANDED. Two builders can both legitimately reselect and only one wins the CAS, so
    registering a restore on `reselected` would let the loser undo the WINNER's advance on its
    way out — the battery corrupting live selection state while tidying up after itself.
    """
    floor_after = str(render.get("periphery_floor_ts") or "")
    if not floor_after or render.get("anchor_advanced") is not True:
        return None
    return Restore(kind="window_anchor",
                   key=f"{floor_after}|{int(render.get('selection_version') or 0)}",
                   prior=floor_before, existed=floor_before is not None)


# ---------------------------------------------------------------------------------- the rows

async def row_cross_thread_awareness(ctx: RowContext) -> None:
    """1. THE STRICT FORM: the fact is a REPLY INSIDE thread A, never top-level.

    The placement is load-bearing. A top-level fact is found by the history walk alone, while a
    reply under another root exercises the activity-index discovery path that free-riding replies
    depend on — which is the path this row exists to measure.
    """
    n = ctx.nonce
    supplier = vendor_name(n)
    crates = quantity(n, "crates", digits=4)   # run-unique: a stale answer must not satisfy this
    channel = ctx.channel
    root_a = await post_seed(channel, f"starting the {supplier} capacity review this week", ctx=ctx)
    fact_ts = await post_seed(channel, f"their yard came back to us — crate count is {crates}",
                              thread_ts=root_a, ctx=ctx)
    root_b = await post_seed(channel, "deli run in twenty minutes if anyone wants anything",
                             ctx=ctx)
    trigger_ts = await post_seed(channel,
                                 f"sorry, what was the crate count on the {supplier} review?",
                                 thread_ts=root_b, ctx=ctx)

    turn_id, replies = await _turn_output(ctx, trigger_ts)
    if not replies:
        raise HarnessError(f"{ctx.row}: the bot never answered in thread B")
    render = await wait_for_telemetry(turn_id, "stream_render")
    ctx.evidence.update({"stream_render": render, "fact_ts": fact_ts, "crates": crates,
                         "supplier": supplier})

    # DIGIT-NORMALIZED. "847,800 crates." is the seeded 847800, and the 2026-08-02 run failed a
    # correct answer over the comma.
    ctx.assert_that("the reply carries this run's crate count",
                    states_number(_text_of(replies), crates))
    horizon = str(render.get("H") or "")
    # NECESSARY evidence that the fact predates admission — NOT proof that it rendered. Compared
    # NUMERICALLY: the production contract rejects string ordering for Slack timestamps.
    ctx.assert_that("H is at or past the fact's ts", bool(horizon) and ts_ge(horizon, fact_ts))

    provenance = await await_tools_used_for(channel, replies[0].ts)
    looked = [name for name in provenance.names if is_search_or_history_name(name)]
    # ASYMMETRIC, and the asymmetry is the point. The store is authoritative WHEN PRESENT, so a
    # recorded search name is positive evidence and FAILS the row. Absence is not gradeable: "no
    # row" is equally what a lost, disabled or still-in-flight write looks like, and grading it as
    # a pass would make a green row indistinguishable from a broken provenance pipeline.
    if provenance.row_present:
        ctx.assert_that("no history or search tool was recorded for the reply", not looked)
    ctx.observe("tools recorded for the reply", list(provenance.names), provenance.row_present)


async def row_verification_rule(ctx: RowContext) -> None:
    """2. A fact below the floor, asked about directly. A DENIAL FAILS — and so does a CONFABULATION.

    Absence from the view is never evidence of absence from the channel; that is the verification
    rule. But "a tool ran and the bot said something" is not the contract either. The row seeds a
    RUN-UNIQUE supplier AND figure in one sentence below the floor, names NEITHER in the trigger —
    the question asks about the WORK — and requires the answer to bring both back. A permanent
    literal would be guessable and would sit in the channel from every previous run; a single
    unique half can still collide once the channel has accumulated enough of them; the pair is
    ~10^8 and cannot.

    SMALL-WINDOW MODE APPLIES HERE: the filler count is `CEILING + 1`, computed at runtime,
    so a bot launched with `CHANNEL_WINDOW_CEILING=12` costs this row ~13 seeded messages
    instead of ~101 — the fact still lands below the floor, which is all the bulk was for.

    IT NO LONGER SCANS THE PROSE FOR DENIALS. That check failed correct answers that happened to
    say "no mention of" about something else; the three assertions that remain — a recorded
    lookup, the supplier, the figure — catch the shape it was written for, a reply that denies
    without having gone to look.
    """
    n = ctx.nonce
    channel = ctx.channel
    supplier = vendor_name(n)
    decision = verification_decision(n)
    work = project_phrase(n)
    # BOTH GRADED HALVES IN ONE SENTENCE, and the WORK is the handle the question will use — so
    # the trigger can ask about this decision without naming either half of what it grades.
    await post_seed(channel,
                    f"we went with the {decision} quote from {supplier} for {work} in the end — "
                    f"the other bids come back next quarter", ctx=ctx)
    # The chatter must state neither the price nor the supplier: either above the floor would put
    # the graded fact in the rendered window, and the row would pass without the bot searching.
    # SEEDED FROM `ctx.nonce`, so the report's nonce reproduces the bulk too — a fresh
    # `mint_nonce()` here was a random seed nobody recorded.
    await seed_messages(channel,
                        chatter_lines(f"{n}|chatter", window_ceiling() + 1,
                                      avoid=[decision], avoid_names=[supplier]), ctx=ctx)

    bot = await bot_identity()
    trigger_ts = await post_seed(
        channel,
        f"<@{bot.user_id}> {work} — which supplier did we go with in the end, and what was the "
        f"figure?", ctx=ctx)
    # MENTIONED FOR A DETERMINISTIC WAKE, and no longer for search itself. It used to be here
    # because an unmentioned turn carries no action_token and `search_slack` was hidden without
    # one; the channel backend is a bot-token scan now and is available on every turn, mentioned
    # or not. What the mention still buys is that the row's own trigger reliably opens a turn, so
    # a failure reads as "it did not go and look" rather than "the gate declined to answer".
    _, replies = await _turn_output(ctx, trigger_ts, deadline=SLOW_TURN_DEADLINE_SECONDS)
    ctx.assert_that("the bot answered the mention", bool(replies))
    if not replies:
        return

    provenance = await await_tools_used_for(channel, replies[0].ts,
                                            required_name=is_search_or_history_name,
                                            deadline=SLOW_TURN_DEADLINE_SECONDS)
    ctx.evidence.update({"tools": list(provenance.names), "decision": decision,
                         "supplier": supplier, "work": work})
    for name, ok in grade_verification(decision=decision, supplier=supplier,
                                       reply_text=_text_of(replies), provenance=provenance):
        ctx.assert_that(name, ok)
    # RECORDED AFTER THE GRADING, and graded by nothing: whether the channel scan can reach the
    # seeded pair at all is the fact that tells a failure here apart from a model that never
    # looked. The seeded supplier and figure are the distinctive words the scan is built for.
    ctx.evidence["search_probe"] = await search_coverage_probe(
        ctx, f"{supplier} {decision}", trigger_ts=trigger_ts)


async def row_cross_thread_action(ctx: RowContext) -> None:
    """3. The answer lands under C, not pasted into A; A gets at most a brief ack.

    THE FYI STEP IS GRADED, NOT ASSUMED. The 2026-08-02 run reported `error` here because the
    gate declined the fyi and no `turn_start` was ever written — a correlation failure standing
    in for what is really a finding about the gate. So the trigger is read through
    `await_trigger_verdict`, which sees the declined shape too, and "it opened a turn at all" is an
    assertion the row can fail rather than an error it dies on.

    AND THE POST HAS TO CARRY THE ANSWER. Destination, absence-from-A and `kind="detached"` are
    all satisfied by a post under C that says "I don't know" — which is the row passing while the
    information it exists to track never moved. The seeded capacity is minted per run and the
    detached post must state it. That is information FLOW, not expression: the row's whole subject
    is whether the number reached the thread that asked for it.

    WHAT THE BOT DOES WITH `{C}` IS RECORDED, NEVER GRADED (owner ruling, 2026-08-03). The row
    used to assert that the bot answered the open pallet question, and it died there twice —
    reporting `fail` on the premise and never exercising the cross-thread half at all. The
    assertion was also asking for something the tuned prompts still refuse: `{C}` is deliberately
    unanswerable when it is asked (the capacity does not exist in this channel until the fyi two
    steps below), and the open-question exception preserves silence exactly where the only honest
    contribution is "I don't know". So the choice — words, a reaction, or nothing — goes in the
    report as an observation and the row CONTINUES regardless. The open question is the stake in
    `{C}` whether or not the bot spoke into it, which is all the cross-thread half needs.

    The hard grading of the open-question floor lives in the scenario corpus instead
    (`open-question-answerable`), where the seeded question IS answerable at the moment it is
    asked, so a bot that stays silent there is failing a rule rather than obeying one.
    """
    n = ctx.nonce
    channel = ctx.channel
    supplier = vendor_name(n)
    capacity = quantity(n, "pallet-capacity", digits=2)
    root_c = await post_seed(channel, "does anyone remember how many crates fit on one pallet?",
                             ctx=ctx)
    c_verdict = await await_trigger_verdict(channel, root_c)
    c_replies = await _posted_by(ctx, root_c, c_verdict)
    # The form comes off the turn's own outcome, not a reaction poll: `reaction_only` is already
    # a `turn_outcome` kind, so nothing here has to spend a reply deadline watching for an emoji.
    chosen = "words" if c_replies else (
        "reaction" if c_verdict.kind == "reaction_only" else "silence")
    ctx.evidence["premise_verdict"] = {"kind": c_verdict.kind, "source": c_verdict.source,
                                       "woke": c_verdict.woke, "choice": chosen}
    ctx.observe("what the bot did with the open pallet question",
                {"choice": chosen, "kind": c_verdict.kind, "woke": c_verdict.woke,
                 "messages": len(c_replies)}, bool(c_verdict.woke))

    # `{C}` is never named in the trigger — the model has to find it.
    trigger_ts = await post_seed(
        channel,
        f"just heard back from {supplier} — their pallets hold {capacity} crates each", ctx=ctx)
    verdict = await await_trigger_verdict(channel, trigger_ts,
                                          deadline=SLOW_TURN_DEADLINE_SECONDS)
    ctx.evidence["verdict"] = {"kind": verdict.kind, "source": verdict.source,
                               "woke": verdict.woke}
    ctx.assert_that("the fyi opened a turn", verdict.woke and bool(verdict.turn_id))
    if not verdict.woke or not verdict.turn_id:
        return

    posted = await observe_turn_output(ctx.team_id, channel, verdict.turn_id,
                                       deadline=SLOW_TURN_DEADLINE_SECONDS, ctx=ctx)
    outcome = await wait_for_telemetry(verdict.turn_id, "turn_outcome",
                                       deadline=SLOW_TURN_DEADLINE_SECONDS)
    ctx.evidence.update({"turn_outcome": outcome, "capacity": capacity, "supplier": supplier})

    ctx.assert_that("exactly one post_to_thread landed under C",
                    len(_posts_under(outcome, root_c)) == 1)
    ctx.assert_that("the post under C states the capacity we just supplied",
                    states_number(_text_of([o for o in posted if o.thread_ts == root_c]),
                                  capacity))
    spoke_a = bool([d for d in _destinations(outcome)
                    if d.get("thread_root_ts") == trigger_ts])
    ctx.assert_that("A heard nothing, or only a brief non-reporting acknowledgment",
                    origin_surface_ok(outcome_kind=str(outcome.get("kind") or ""),
                                      spoke_in_origin=spoke_a,
                                      origin_text=_text_of(
                                          [o for o in posted if o.thread_ts != root_c]),
                                      addressees=await harness_user_display_names(),
                                      addressee_ids=(
                                          (await harness_user_identity()).user_id,)))


async def row_search_to_action(ctx: RowContext) -> None:
    """4. An OBLIGATION is buried below the floor, and the trigger makes it answerable.

    REDESIGNED 2026-08-04 (owner ruling). The trigger used to end with an explicit instruction to
    put the answer in the other thread — a magic phrase, and the exact thing the ruling threw out:
    "the bot should intelligently decide if it needs to go back... no special wording". Deleting
    that clause alone would not have been enough either, because what was left was a question asked
    here, which can perfectly reasonably be answered here. So the row's PREMISE moved instead of
    its wording: what is buried is no longer a fact the bot is sent to fetch, it is a question the
    bot was asked and has never answered.
    Nothing in the trigger says where anything goes; the reason the answer belongs under `{C}` is
    that `{C}` is where it was asked and where it is still owed.

    The obligation is genuinely OPEN when it is seeded, not merely unanswered: the buried question
    cannot be answered at the time it is asked (this run's figure does not exist in this channel
    yet), which is the same construction row 3 uses. So whatever the bot does with the mention when
    it lands — say it cannot reach the number, react, say nothing — the thing owed is still owed,
    and the trigger is what makes it payable. That choice is RECORDED, never graded, for the same
    reason it is in row 3: silence where the only honest answer is "I can't say yet" is a rule
    being obeyed, not a rule being broken.

    AND THE QUESTION ASKS FOR THE NUMBER, NOT FOR A YES (codex r2 #3). It used to ask whether the
    quote was inside the cap, which is answerable by guessing: a seed-time "yes, that's within it"
    would close the obligation before the graded trigger ever ran, and the row would go on
    measuring a thread that no longer owed anything. What it asks for now is the number itself — a
    run-unique figure that exists nowhere in the workspace when the question is posted — so no
    guess can discharge it.

    THE PREVIOUS RUN COULD STILL ANSWER IT, THOUGH, and did (2026-08-04): the channel keeps
    everything, the question read the same every run, and the seed-time turn answered it with the
    cap from the run before. So the SUBJECT is drawn per run (`row4_subject`) and the guard reads
    the seed-time reply for ANY cap-shaped figure (`stale_cap_figure`), not merely this run's. A
    premise that quietly stopped holding ERRORS the row; it must never be reported as a model
    failure.

    The obligation CANNOT live in a reply-less root: §2g derives a hit's `thread_ts` from a thread
    reply's permalink, and a hit on a bare top-level message carries none and enrolls nothing.

    SMALL-WINDOW MODE APPLIES HERE: the filler count is `CEILING + 1`, computed at runtime,
    so a bot launched with `CHANNEL_WINDOW_CEILING=12` costs this row ~13 seeded messages
    instead of ~101 — the obligation still lands below the floor, which is all the bulk was for.
    """
    n = ctx.nonce
    channel = ctx.channel
    supplier = vendor_name(n, "issuer")
    quoted = money(n, "reissue")
    # THE CAP MUST NOT EQUAL THE QUOTE (codex r3 #2). Both are independent five-digit `money`
    # draws, so they can collide — and on a colliding run the cap would already be sitting in the
    # buried question, letting a post that merely repeats the supplier and the quote satisfy all
    # three content clauses without ever using what the trigger supplied. Reminting by salt keeps
    # the row REPLAYABLE (same nonce, same figures) where rejecting the nonce would not.
    cap = money(n, "renewals-cap")
    attempt = 1
    while cap == quoted:
        attempt += 1
        cap = money(n, f"renewals-cap-{attempt}")
    subject, work = row4_subject(n)
    bot = await bot_identity()
    root_c = await post_seed(channel, "weekly infra sync", ctx=ctx)
    # THE PAIR, IN ONE SENTENCE, INSIDE THE QUESTION ITSELF. It is what makes the buried thread
    # FINDABLE — the trigger names the subject and the work, and the search that proves the row
    # has to have words to land on — and what arms the premise guard. The post is no longer
    # graded on reproducing it (the landing is the read-proof; see the grader). What the ask
    # WANTS is the cap: a figure that does not exist yet.
    asked_ts = await post_seed(
        channel,
        f"<@{bot.user_id}> the {work} is stuck with {supplier} — they've quoted {quoted} for it. "
        f"what's our {subject} for the year? nobody's given us the number",
        thread_ts=root_c, ctx=ctx)
    # THE PREMISE, READ RATHER THAN ASSUMED. The mention wakes a turn; what that turn chooses is an
    # observation, and settling it here also keeps the seed-time turn from still being in flight
    # when the trigger arrives.
    asked_verdict = await await_trigger_verdict(channel, asked_ts)
    asked_replies = await _posted_by(ctx, asked_ts, asked_verdict)
    ctx.evidence["premise_verdict"] = {"kind": asked_verdict.kind, "source": asked_verdict.source,
                                       "woke": asked_verdict.woke,
                                       "messages": len(asked_replies)}
    # AN INVALID SETUP, NOT A FAILED ROW. Nothing in the workspace holds this run's figure when
    # the question is posted, so ANY price in the answer means the obligation was discharged from
    # somewhere it should not have been — the previous run's trigger still in history, residue
    # nobody cleared, a colliding figure — and everything after this point would be grading a
    # thread that no longer owes anything.
    stale = stale_cap_figure(_text_of(asked_replies), quoted=quoted)
    if stale:
        raise HarnessError(f"{ctx.row}: the seed-time answer already states {stale} as the "
                           f"{subject} (this run's is {cap}), so the obligation was closed before "
                           f"the trigger ran and nothing downstream measures what this row is "
                           f"about")
    ctx.observe("what the bot did with the unanswerable question",
                {"kind": asked_verdict.kind, "woke": asked_verdict.woke,
                 "messages": len(asked_replies)}, bool(asked_verdict.woke))
    # BOTH SEEDED HALVES ARE EXCLUDED FROM THE CHATTER. A bulk line stating the quote would arm
    # the premise guard against the run's own noise, and one minting the supplier (codex generated
    # a real collision) would hand the origin-fragment guard a false leak — the pair is no longer
    # graded in the post, but it still guards the premise and the origin.
    await seed_messages(channel,
                        chatter_lines(f"{n}|chatter", window_ceiling() + 1,
                                      avoid=[quoted], avoid_names=[supplier]), ctx=ctx)

    # THE MISSING PIECE, HANDED OVER AS NEWS. It names the subject — without which nothing would
    # make searching this channel the obvious move — and it names NOTHING about placement: no
    # thread, no "over there", not even that a question is outstanding. Where the answer goes is
    # left entirely to the bot, which is the whole of what this row now measures.
    trigger_ts = await post_seed(
        channel,
        f"<@{bot.user_id}> finance settled the {subject} — {cap} a year, effective now. that's "
        f"the number the {work} was waiting on", ctx=ctx)

    turn_id, prose = await _turn_output(ctx, trigger_ts, deadline=SLOW_TURN_DEADLINE_SECONDS)
    outcome = await wait_for_telemetry(turn_id, "turn_outcome",
                                       deadline=SLOW_TURN_DEADLINE_SECONDS)
    ctx.evidence["turn_outcome"] = outcome
    posts = [o for o in prose if o.thread_ts == root_c]

    provenance = ProvenanceRead(row_present=False, names=())
    if posts:
        # Against the DESTINATION post's ts — W3 extended the provenance writer's reach to the
        # cross-thread post precisely so this is readable.
        provenance = await await_tools_used_for(channel, posts[0].ts,
                                                required_name=is_search_or_history_name,
                                                deadline=SLOW_TURN_DEADLINE_SECONDS)
    ctx.evidence.update({"tools": list(provenance.names), "supplier": supplier,
                         "quoted": quoted, "cap": cap, "asked_ts": asked_ts,
                         # The run's own words, so a failed assertion can be read back against
                         # the messages this run actually posted rather than a remembered shape.
                         "subject": subject, "work": work})

    for name, ok in grade_search_to_action(fact=supplier, cap=cap,
                                           target_root=root_c, origin_root=trigger_ts,
                                           outcome_kind=str(outcome.get("kind") or ""),
                                           destinations=_destinations(outcome),
                                           provenance=provenance,
                                           posted_text=_text_of(posts),
                                           # Everything the turn said ANYWHERE but the target —
                                           # a threaded ack and a top-level one are the same
                                           # surface to the ack predicate.
                                           origin_text=_text_of(
                                               [o for o in prose if o.thread_ts != root_c]),
                                           # The person the bot may thank BY NAME or mention is
                                           # the real operator whose token posted the news.
                                           origin_addressees=await harness_user_display_names(),
                                           origin_addressee_ids=(
                                               (await harness_user_identity()).user_id,)):
        ctx.assert_that(name, ok)
    # OBSERVATION ONLY. `roots` says whether the scan reached `{C}` — the buried reply's root and
    # the thread the answer was supposed to land in — so "search did not cover the target" is
    # readable in the report instead of being inferred from a missing figure.
    ctx.evidence["search_probe"] = await search_coverage_probe(
        ctx, f"{supplier} {quoted}", trigger_ts=trigger_ts)
    ctx.evidence["target_root"] = root_c


async def row_full_origin_fidelity(ctx: RowContext) -> None:
    """5. A DEEP origin thread whose OLDEST message carries the fact.

    THE DEPTH IS DERIVED (`origin_reply_count`), not a literal: in small-window mode this is ~24
    replies rather than 120, which still buries the root far below the window while costing the
    room a fifth of the messages. What the row proves is that the origin renders WHOLE, and that
    is `origin_count` being exactly right — a property the thread's absolute depth does not change.

    The count assertion compares like with like: `origin_count` is RENDERED ELIGIBLE origin
    messages, so the thread is seeded with HUMAN messages only and eligible and raw coincide —
    root + the seeded replies + the trigger reply, all eligible at admission.

    THE FACT IS A PAIR HERE TOO — the supplier whose count it was and the opening tally, both in
    the root, 121 messages above the question. The trigger asks for both, so grading both is
    grading whether the oldest message survived the render rather than grading how the bot writes.
    """
    n = ctx.nonce
    channel = ctx.channel
    supplier = vendor_name(n, "count")
    tally = quantity(n, "tally", digits=5)   # run-unique, same reason as row 1's crate count
    root_a = await post_seed(
        channel, f"kicking off the {supplier} inventory count — opening tally is {tally} cases",
        ctx=ctx)
    replies = chatter_lines(f"{n}|chatter", origin_reply_count(), avoid=[tally],
                            avoid_names=[supplier])
    await seed_messages(channel, replies, thread_ts=root_a, ctx=ctx)
    # THE QUESTION ASKS FOR BOTH HALVES, so the row can grade both without grading expression:
    # reproducing what was asked for is answering, not a style the harness happens to prefer.
    trigger_ts = await post_seed(channel,
                                 "what was the opening tally on this again, and whose count "
                                 "was it?", thread_ts=root_a, ctx=ctx)

    turn_id, answers = await _turn_output(ctx, trigger_ts, deadline=SLOW_TURN_DEADLINE_SECONDS)
    if not answers:
        raise HarnessError(f"{ctx.row}: no reply landed in the deep origin thread")
    render = await wait_for_telemetry(turn_id, "stream_render")
    ctx.evidence.update({"stream_render": render, "tally": tally, "supplier": supplier})

    ctx.assert_that("the reply carries this run's opening tally",
                    states_number(_text_of(answers), tally))
    ctx.assert_that("the reply names whose count it was",
                    states_phrase(_text_of(answers), supplier))
    expected = 1 + len(replies) + 1
    ctx.assert_that(f"origin_count is {expected}", render.get("origin_count") == expected)


async def row_stream_currency(ctx: RowContext) -> None:
    """6. Freeze after admission, post M, resume: the frozen turn cannot see M.

    `H` is pinned at admission and never refreshed, so a message that arrives afterwards belongs
    to the NEXT turn. Only a barrier can observe that — once the reply has landed, the stream it
    rendered is no longer inspectable from outside.

    M IS AN ORDINARY MESSAGE, and the thing the frozen reply must not contain is the supplier it
    names. An ALL-CAPS marker word was the old mechanism; a company that does not exist does the
    same job while reading as news somebody actually posted.
    """
    n = ctx.nonce
    channel = ctx.channel
    late = vendor_name(n, "late")
    root_a = await post_seed(channel, "quick sanity check on something", ctx=ctx)
    trigger_ts = await post_seed(channel,
                                 "what's the most recent message you can see in this channel?",
                                 thread_ts=root_a, ctx=ctx)

    turn_id = await find_turn_id(channel, trigger_ts)
    ctx.evidence["turn_id"] = turn_id
    admission = barrier_operation(POST_ADMISSION, turn_id=turn_id)
    await wait_barrier_reached(POST_ADMISSION, admission)
    try:
        m_ts = await post_seed(channel,
                               f"heads up — {late} just moved our delivery to {weekday(n)}",
                               ctx=ctx)
    finally:
        # RELEASE IN `finally`. Anything going wrong between reaching the barrier and releasing it
        # leaves the BOT paused until its own timeout — a failed row that also wedges every row
        # queued behind it.
        release_barrier(POST_ADMISSION, admission)

    frozen = await observe_turn_output(ctx.team_id, channel, turn_id, ctx=ctx)
    if not frozen:
        raise HarnessError(f"{ctx.row}: the frozen turn never produced a reply")
    render = await wait_for_telemetry(turn_id, "stream_render")
    ctx.evidence.update({"stream_render": render, "M_ts": m_ts, "late_supplier": late})
    ctx.assert_that("the frozen turn's H predates M", ts_lt(str(render.get("H") or ""), m_ts))
    ctx.assert_that("the frozen turn's reply does not mention M",
                    not states_phrase(_text_of(frozen), late))

    follow_ts = await post_seed(channel, "anything new since?", thread_ts=root_a, ctx=ctx)
    follow_turn, follow = await _turn_output(ctx, follow_ts)
    if not follow:
        raise HarnessError(f"{ctx.row}: the follow-up turn never produced a reply")
    follow_render = await wait_for_telemetry(follow_turn, "stream_render")
    ctx.evidence["follow_up_stream_render"] = follow_render
    ctx.assert_that("the next turn's H includes M",
                    ts_ge(str(follow_render.get("H") or ""), m_ts))


async def row_in_flight_exclusion(ctx: RowContext) -> None:
    """7. Receipt-STATE evidence, plus aggregates — and the gap between those is recorded.

    B renders while A's receipt is `in_flight`; C renders after A finalizes. The row proves A's
    STATE TRANSITIONS precisely: A's own surface ts is read from `outbound_receipts` while A is
    still frozen at the barrier (so the read brackets B rather than being taken afterwards), and
    again after A settles and before C is posted — so C can never be admitted while A is still in
    flight, which would have made its inclusion prove nothing.

    **WHAT THIS ROW DOES NOT PROVE, STATED PLAINLY.** It does not prove that B EXCLUDED A or that
    C INCLUDED A. Those two conclusions rest on `receipts_excluded_count`, `receipts_included_count`
    and an opaque room-wide `receipts_membership_hash`, none of which names a message. A
    serializer bug that included A in BOTH renders, while any unrelated receipt changed state
    between them, satisfies every assertion here.

    **AND WHY NO STRONGER EVIDENCE IS TAKEN.** `ChannelStream` holds the full `receipts_included`
    and `receipts_excluded` tuples, but `stream_render_fields()` deliberately emits only the two
    counts and a `sha256("included:<csv>;excluded:<csv>")`. Recovering A's membership from that
    hash means reconstructing the COMPLETE included and excluded sets, which means reproducing
    production's eligibility, floor, dedupe and RULING-1 duplication rules inside the harness — a
    reimplementation whose disagreement with production would mis-grade silently in both
    directions. Bracketing `byte_count` fails the same way, and additionally cannot account for
    the bot's own model-authored reply lengths. Emitting the membership lists would be a new
    telemetry field, which this contract forbids. So the limitation is RECORDED as an
    observation rather than papered over with a proxy that only looks stronger.
    """
    channel = ctx.channel
    bot = await bot_identity()
    root_a = await post_seed(channel, "pallet standards for the new hires", ctx=ctx)
    root_b = await post_seed(channel, "printer situation", ctx=ctx)

    a_ts = await post_seed(
        channel,
        f"<@{bot.user_id}> could you write up a proper explainer on our pallet standards for the "
        f"new hires — stacking limits, labelling, the lot?", thread_ts=root_a, ctx=ctx)
    turn_a = await find_turn_id(channel, a_ts, deadline=SLOW_TURN_DEADLINE_SECONDS)
    ctx.evidence["turn_a"] = turn_a
    # THE PARTIAL-POST SEAM KEYS ON THE SURFACE'S message_ts, NOT THE TURN ID — its call site
    # passes no turn_id, so `dev_barriers.operation_id` falls through to the message. A live run
    # proved it: the bot wrote `post_partial_post.<its reply ts>.0.waiting`. Releasing on a turn
    # id would name a path no waiter is watching, and the turn would sit there until its own
    # timeout holding a message NO token can delete. The receipt row is written BEFORE the seam
    # announces, so the ts is available in time to build the path from it.
    a_surface_ts = await await_in_flight_surface(ctx.team_id, channel, turn_a)
    partial = barrier_operation(POST_PARTIAL_POST, message_ts=a_surface_ts)
    if a_surface_ts not in ctx.observed_ts:
        ctx.observed_ts.append(a_surface_ts)
    # THE SEAM IS PROCESS-GLOBAL, so B and C freeze on it too unless somebody frees them. Two
    # live runs proved it: the row reported `error: turn C never replied` with its first two
    # assertions already green, and left a frozen streaming message behind. `freeing_other_turns`
    # releases every key but A's for as long as this row runs.
    async with freeing_other_turns(POST_PARTIAL_POST, partial) as freed_elsewhere:
        await _run_in_flight_body(ctx, root_a, root_b, a_ts, turn_a, a_surface_ts, partial, bot)
    ctx.evidence["barriers_freed_for_other_turns"] = list(freed_elsewhere)


async def _run_in_flight_body(ctx: RowContext, root_a: str, root_b: str, a_ts: str, turn_a: str,
                              a_surface_ts: str, partial: str, bot) -> None:
    """Row 7's measurement, with A's barrier held and every other turn's freed around it."""
    channel = ctx.channel
    try:
        await wait_barrier_reached(POST_PARTIAL_POST, partial)

        # MENTIONED, both of them. B and C are INSTRUMENTS — they exist to produce a stream
        # render on either side of A's state transition, and they grade nothing about
        # participation. Live, an undirected "and now?" drew a silent turn and the row died
        # with two of its six assertions already green. A mention makes the instrument fire
        # without touching any claim the row makes.
        b_ts = await post_seed(channel, f"<@{bot.user_id}> any news on that printer?",
                               thread_ts=root_b, ctx=ctx)
        turn_b = await find_turn_id(channel, b_ts)
        render_b = await wait_for_telemetry(turn_b, "stream_render")

        # A-SPECIFIC, read while A is STILL frozen: A's own surface was in_flight across B's
        # admission and render, so the receipt B excluded is A's and not some other turn's.
        state_at_b = await read_receipt_state(ctx.team_id, channel, a_surface_ts)
        ctx.assert_that("A's own surface was in_flight when B rendered", state_at_b == "in_flight")
    finally:
        release_barrier(POST_PARTIAL_POST, partial)

    # A must SETTLE before C is posted, or C could render while A is still in flight and its
    # inclusion would prove nothing about finalization.
    await _turn_output(ctx, a_ts, deadline=SLOW_TURN_DEADLINE_SECONDS)
    state_after = await read_receipt_state(ctx.team_id, channel, a_surface_ts)
    ctx.assert_that("A's own surface finalized before C", state_after == "finalized")

    c_ts = await post_seed(channel, f"<@{bot.user_id}> and now? anything changed?",
                           thread_ts=root_b, ctx=ctx)
    turn_c, c_prose = await _turn_output(ctx, c_ts)
    if not c_prose:
        raise HarnessError(f"{ctx.row}: turn C never replied")
    render_c = await wait_for_telemetry(turn_c, "stream_render")
    ctx.evidence.update({"stream_render": render_b, "stream_render_c": render_c,
                         "A_surface_ts": a_surface_ts})

    ctx.assert_that("A predates B's horizon", ts_lt(a_surface_ts, str(render_b.get("H") or "")))
    ctx.assert_that("B excluded an in-flight receipt",
                    int(render_b.get("receipts_excluded_count") or 0) >= 1)
    ctx.assert_that("C includes more receipts than B did",
                    int(render_c.get("receipts_included_count") or 0)
                    > int(render_b.get("receipts_included_count") or 0))
    ctx.assert_that("the two membership hashes differ",
                    render_b.get("receipts_membership_hash")
                    != render_c.get("receipts_membership_hash"))
    # THE HONEST CAVEAT, ON THE ROW'S OWN REPORT. An observation can never change a status; it is
    # here so a reader of this row's green result knows exactly which claim it did not make.
    ctx.observe("A-specific membership not directly evidenced; state transitions + aggregates only",
                {"excluded_b": render_b.get("receipts_excluded_count"),
                 "included_b": render_b.get("receipts_included_count"),
                 "included_c": render_c.get("receipts_included_count")},
                False)


async def row_re_anchor_observable(ctx: RowContext) -> None:
    """8. Drive the channel past the ceiling in ROOTS and watch a build reselect.

    GRADED ON `reselected`, not on `anchor_advanced` and not on "exactly one turn": two concurrent
    builders can both legitimately reselect.

    SMALL-WINDOW MODE APPLIES HERE: the filler count is `CEILING + 1`, computed at runtime,
    so a bot launched with `CHANNEL_WINDOW_CEILING=12` costs this row ~13 seeded messages
    instead of ~101 — the fact still lands below the floor, which is all the bulk was for.

    THIS IS THE ONLY ROW THAT CHANGES DURABLE SELECTION STATE, and the only reason this file still
    has a restore at all. The restore is COMPARE-AND-RESTORE: if a legitimate turn advanced the
    anchor after us, we do NOT overwrite it — the restore fails, and the row lands on `unrestored`
    saying so.
    """
    channel = ctx.channel
    floor_before = await read_window_anchor(ctx.team_id, channel)

    await seed_messages(channel,
                        chatter_lines(f"{ctx.nonce}|chatter", window_ceiling() + 1), ctx=ctx)
    bot = await bot_identity()
    trigger_ts = await post_seed(channel, f"<@{bot.user_id}> you still with us?", ctx=ctx)

    turn_id, replies = await _turn_output(ctx, trigger_ts, deadline=SLOW_TURN_DEADLINE_SECONDS)
    render = await wait_for_telemetry(turn_id, "stream_render")
    ctx.evidence.update({"stream_render": render,
                         "floor_before": list(floor_before) if floor_before else None})

    floor_after = str(render.get("periphery_floor_ts") or "")
    restore = anchor_restore_for(render, floor_before)
    if restore is not None:
        ctx.restores.append(restore)

    ctx.assert_that("the build reselected", render.get("reselected") is True)
    ctx.assert_that("the floor moved forward",
                    bool(floor_after) and (floor_before is None
                                           or ts_lt(floor_before[0], floor_after)))
    ctx.assert_that("root_count returned to the target",
                    render.get("root_count") == window_target())
    # Provider-dependent, and the standing ruling is that render-only hash probes are the equality
    # proof. Recorded, never a pass condition.
    ctx.observe("cached_input_tokens", render.get("cached_input_tokens"),
                "cached_input_tokens" in render)
    ctx.observe("the bot answered the re-anchor trigger", len(replies), bool(replies))


async def row_foreign_exchange_bait(ctx: RowContext) -> None:
    """9a. MACHINERY ROW. Two apparent humans talk. Our bot stays out of it.

    ASSERTABLE, and it stays assertable under the 2026-08-03 ruling: staying out of a conversation
    between two other parties is the tuned gate deciding not to wake, not the responder choosing
    how to express itself. Nothing here grades what our bot would have SAID.

    THE SECOND HUMAN IS CLAUDE TAG, via the preconfigured `DEV_TREAT_BOT_IDS_AS_HUMAN` carve-out —
    a recorded owner decision, not a new choice, and no second user token exists. The identity is
    the one the RUN's preflight already verified; resolving it again would spend a second
    `bots.info` re-deriving an answer we already hold.

    THE GRADED TRIGGER IS CLAUDE TAG'S REPLY: the last inbound event, and the moment restraint is
    actually being tested. The row grades the FIRST reply that lands, then waits for QUIESCENCE —
    two consecutive EMPTY POLLS, not a fixed sleep — and requires EVERY observed Claude reply's
    turn to be silent. Grading only the first would let a second exchange turn draw our bot in
    unnoticed; sleeping ten seconds and reading once establishes nothing about the interval it
    slept through, and §9 forbids fixed sleeps as race inducers.

    Claude Tag's replies stay in the channel like everything else here, listed in the report under
    `external_ts` so a reader can tell them from ours.
    """
    n = ctx.nonce
    channel = ctx.channel
    claude = ctx.claude
    if claude is None:
        raise HarnessPreflightError(
            f"{ctx.row}: no verified Claude Tag identity was handed to the row; the run's "
            f"preflight is what establishes it.")

    root_a = await post_seed(channel, "did the espresso machine ever get fixed?", ctx=ctx)
    bait_ts = await post_seed(channel, f"<@{claude.user_id}> {bait_for(n)}",
                              thread_ts=root_a, ctx=ctx)

    # THE THIRD-PARTY BOUND, not ours. Claude Tag answers on its own schedule and the 180s we
    # allow our own bot is inside its measured spread — the 2026-08-01 pass reported `error` here
    # for a premise that was merely slow. See THIRD_PARTY_REPLY_DEADLINE_SECONDS for the numbers.
    first = await wait_bot_reply(channel, root_a, bait_ts, author=claude,
                                 deadline=THIRD_PARTY_REPLY_DEADLINE_SECONDS)
    if not first:
        # STILL AN ERROR, never a pass. Our bot may well have been silent throughout, but silence
        # in a room where nobody spoke is not restraint and grading it as such would turn the
        # battery's one third-party row green for having measured nothing. The observation records
        # WHY, so a reader can tell this from a row that failed on our own behaviour.
        ctx.observe("Claude Tag did not reply within the third-party bound; known-flaky premise "
                    "(measured first replies 31.0/82.8/100.5/477.6s, and Tag has itself reported "
                    "dropping an @-mention in this channel)",
                    {"waited_seconds": THIRD_PARTY_REPLY_DEADLINE_SECONDS,
                     "bait_ts": bait_ts, "claude_user_id": claude.user_id}, False)
        raise HarnessPreflightError(
            f"{ctx.row}: Claude Tag never replied within "
            f"{THIRD_PARTY_REPLY_DEADLINE_SECONDS}s. Grading our bot's silence against a "
            f"conversation that never happened would measure nothing.")
    claude_replies = await _until_quiet(channel, root_a, first, claude)
    await _observe(ctx, claude_replies)

    kinds: List[str] = []
    for reply in claude_replies:
        # EITHER SHAPE. If our gate declines Claude's reply there is no turn and no
        # `turn_outcome` — only `visible_action(kind=silence)` — and that is the very restraint
        # this row measures. Reading only the woken shape would report `error` for a pass.
        kinds.append((await await_trigger_verdict(channel, reply.ts)).kind)
    ctx.evidence.update({"claude_reply_ts": [r.ts for r in claude_replies],
                         "turn_outcome_kinds": kinds})

    ctx.assert_that("every Claude reply's turn was silent",
                    bool(kinds) and all(kind == "silence" for kind in kinds))

    ours = await wait_bot_reply(channel, root_a, root_a, deadline=0)
    ctx.assert_that("our bot posted nothing in the exchange", not ours)

    # EVERY message in the exchange, not only the bait: a reaction on the root or on one of
    # Claude's replies is our bot inserting itself just as surely as one on the bait.
    reacted: Dict[str, List[str]] = {}
    for ts in [root_a, bait_ts, *[r.ts for r in claude_replies]]:
        marks = await wait_bot_reaction(channel, ts, deadline=0)
        if marks:
            reacted[ts] = list(marks)
    ctx.evidence["reactions"] = reacted
    ctx.assert_that("our bot reacted to nothing in the exchange", not reacted)


async def _until_quiet(channel: str, root_ts: str, seen: List[Observed],
                       author: PartyIdentity, *, quiet_polls: int = 2,
                       poll: float = REPLY_POLL_SECONDS) -> List[Observed]:
    """Collect replies until TWO CONSECUTIVE polls come back empty.

    Two poll intervals is the smallest quiescence window a single tick's timing luck cannot
    satisfy — and it has to be two consecutive EMPTY READS, not one sleep of that length followed
    by one read, which proves nothing about the interval it slept through.
    """
    collected = list(seen)
    latest = collected[-1].ts
    empties = 0
    while empties < quiet_polls:
        await pace(poll)
        batch = await wait_bot_reply(channel, root_ts, latest, author=author, deadline=0)
        if batch:
            collected.extend(batch)
            latest = batch[-1].ts
            empties = 0
        else:
            empties += 1
    return collected


async def row_directed_banter_answered(ctx: RowContext) -> None:
    """9b. MODEL-CHOICE ROW. Directly addressed, the bot RESPONDS — the form is its own business.

    **WHAT IS GRADED IS THE MACHINERY, NOT THE EXPRESSION** (owner ruling, 2026-08-03). "The bot
    was addressed and it did something visible" is a property of the wake path, which is tuned and
    testable. WHETHER it answers in words or with an emoji, and how long the answer is, is the
    responder model choosing — and a harness that pinned that would be failing the bot for
    exercising judgement it is supposed to have. An earlier version asserted `kind == "reply"`,
    which is exactly that mistake: a perfectly good 🌭 on the question would have failed the row.

    So a message OR a reaction satisfies it, and which one arrived is recorded rather than graded.
    """
    channel = ctx.channel
    bot = await bot_identity()
    trigger_ts = await post_seed(
        channel, f"<@{bot.user_id}> settle a bet for us — is a hot dog a sandwich?", ctx=ctx)

    verdict = await await_trigger_verdict(channel, trigger_ts)
    replies = await _posted_by(ctx, trigger_ts, verdict)
    # Only WAIT for a reaction when no words landed. A reply is already a response, and spending
    # the full deadline watching for an emoji the bot had no reason to add would add three minutes
    # to every green run.
    reactions = await wait_bot_reaction(channel, trigger_ts,
                                        deadline=0 if replies else REPLY_DEADLINE_SECONDS)
    ctx.evidence.update({"verdict": {"kind": verdict.kind, "source": verdict.source,
                                     "woke": verdict.woke},
                         "reactions": list(reactions)})

    ctx.assert_that("the bot responded when directly addressed",
                    bool(replies) or bool(reactions))
    ctx.observe("the form the bot chose",
                {"kind": verdict.kind, "messages": len(replies), "reactions": list(reactions)},
                bool(replies) or bool(reactions))


async def row_thanks_response_choice(ctx: RowContext) -> None:
    """9c. OBSERVATION-ONLY ROW: it records what the bot did about a "thanks" and grades nothing.

    **IT ALWAYS PASSES, DELIBERATELY** (owner ruling, 2026-08-03). Answering a thank-you with an
    emoji, with a short line, or with nothing at all are three reasonable choices, and which one
    the responder makes is not the harness's business. The row exists to put the choice in the
    report — with the words the bot actually used — so a human reading the run can see how the bot
    behaves in the room without a grader deciding on their behalf.

    It used to assert the emoji: "a reaction landed", "no new bot message followed",
    "the turn reports reaction_only". That is the hardcoded behavioural expectation the ruling
    removes.

    **THE SETUP IS ALSO UNGRADED.** Whether the bot answers the arithmetic is another free choice,
    so it is read through the both-shapes verdict reader and recorded, never required. The one
    thing that still raises is a trigger the bot never JUDGED at all — no `turn_outcome` and no
    `visible_action` — because that is the machinery failing rather than the model choosing.
    """
    channel = ctx.channel
    bot = await bot_identity()
    root_a = await post_seed(channel, f"<@{bot.user_id}> what's 34 × 18?", ctx=ctx)
    sum_verdict = await await_trigger_verdict(channel, root_a)
    sum_replies = await _posted_by(ctx, root_a, sum_verdict)
    ctx.observe("what the bot did with the arithmetic",
                {"kind": sum_verdict.kind, "messages": len(sum_replies)}, bool(sum_replies))

    thanks_ts = await post_seed(channel, "thanks!", thread_ts=root_a, ctx=ctx)
    verdict = await await_trigger_verdict(channel, thanks_ts)
    followups = await _posted_by(ctx, thanks_ts, verdict)
    # The reaction wait is the long one ONLY when no words landed — same reason as 9b.
    reactions = await wait_bot_reaction(channel, thanks_ts,
                                        deadline=0 if followups else REPLY_DEADLINE_SECONDS)

    chosen = "reaction" if reactions and not followups else (
        "message" if followups and not reactions else (
            "message and reaction" if followups else "silence"))
    ctx.evidence.update({"verdict": {"kind": verdict.kind, "source": verdict.source,
                                     "woke": verdict.woke},
                         "reactions": list(reactions), "thanks_ts": thanks_ts,
                         "choice": chosen})
    # The TEXT rides in `evidence.observed_text` already (`observe_turn_output` copies it there),
    # so the report shows both what the bot chose and, when it chose words, which words.
    ctx.observe("what the bot chose in answer to thanks",
                {"choice": chosen, "kind": verdict.kind, "woke": verdict.woke,
                 "reactions": list(reactions), "messages": len(followups)}, True)


async def row_value_floor_holds(ctx: RowContext) -> None:
    """9d. MACHINERY ROW. A low-value aside addressed to nobody draws silence or an emoji.

    ASSERTABLE for the same reason as 9a: the value floor is tuned gate machinery. The row allows
    either silence or a reaction precisely because choosing BETWEEN those two is the model's, and
    it grades only that no message was posted.
    """
    channel = ctx.channel
    trigger_ts = await post_seed(channel, aside_for(ctx.nonce), ctx=ctx)

    # THE DECLINED SHAPE IS THE EXPECTED ONE HERE. An aside addressed to nobody is judged by the
    # gate and never opens a turn, so there is no `turn_outcome` to read — only
    # `visible_action(kind=silence)`. A live probe confirmed exactly that sequence.
    verdict = await await_trigger_verdict(channel, trigger_ts)
    replies = await _posted_by(ctx, trigger_ts, verdict)
    ctx.evidence["verdict"] = {"kind": verdict.kind, "source": verdict.source,
                               "woke": verdict.woke}
    ctx.assert_that("the turn stayed at or below a reaction",
                    verdict.kind in ("silence", "reaction_only"))
    ctx.assert_that("no message was posted", not replies)


async def row_render_equality_probe(ctx: RowContext) -> None:
    """10. ONE shared periphery pin, TWO origin pins — the equality proof.

    Two separate invocations cannot prove the same thing: their `H` values differ by construction,
    so their peripheries differ and any hash comparison between them is meaningless. Two natural
    turns are the same trap. This is why the probe takes both origins in ONE process.

    THE PROBE WAITS FOR ITS OWN THREADS TO EXIST. On 2026-08-02 it launched immediately after
    seeding and died on `OriginFetchError: origin thread … came back empty for a reply-triggered
    turn` — `conversations.replies` had not yet returned the reply we had just posted. The wait
    reads through the SAME token the probe uses, so "visible to us" means visible to it.
    """
    n = ctx.nonce
    channel = ctx.channel
    root_a = await post_seed(channel,
                             f"has anyone tried the {vendor_name(n, 'oat')} oat milk in the "
                             f"kitchen?", ctx=ctx)
    reply_a = await post_seed(channel, "yeah, it's fine actually — better than the last one",
                              thread_ts=root_a, ctx=ctx)
    root_b = await post_seed(channel, "garage is closed for resurfacing tomorrow", ctx=ctx)
    reply_b = await post_seed(channel, "good to know, I'll park on the street",
                              thread_ts=root_b, ctx=ctx)

    # BY TS, not by count: another conversation's reply in the same thread satisfies a count while
    # the reply we just posted is still invisible to the probe.
    for root, reply in ((root_a, reply_a), (root_b, reply_b)):
        await await_thread_visible(channel, root, expected_ts=(root, reply))

    with tempfile.TemporaryDirectory() as workdir:
        out = Path(workdir) / "probe.json"
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "tools.stream_probe", "--channel", channel,
            "--origin", root_a, "--origin", root_b, "--out", str(out),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stderr = b""
        try:
            # BOUNDED, and reaped in `finally`. A wedged probe would otherwise block the battery
            # for as long as the operator's patience lasts, and a cancellation here would leave a
            # live subprocess behind holding its own Slack reads open.
            _, stderr = await asyncio.wait_for(process.communicate(),
                                               timeout=SLOW_TURN_DEADLINE_SECONDS)
        except asyncio.TimeoutError as e:
            raise HarnessError(f"{ctx.row}: stream_probe did not finish within "
                               f"{SLOW_TURN_DEADLINE_SECONDS}s") from e
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        if process.returncode != 0 or not out.exists():
            raise HarnessError(
                f"{ctx.row}: stream_probe exited {process.returncode}: "
                f"{stderr.decode('utf-8', 'replace')[-500:]}")
        report = json.loads(out.read_text(encoding="utf-8"))

    ctx.evidence["probe"] = report
    ctx.assert_that("the two origins share their pre-breakpoint bytes",
                    report.get("prefix_identical") is True)
    ctx.assert_that("the two origins' unions differ", report.get("unions_differ") is True)


# ------------------------------------------------------------------------------- the registry

REGISTRY: Tuple[BatteryRow, ...] = (
    BatteryRow(
        name="cross-thread-awareness",
        trigger_template="sorry, what was the crate count on the {supplier} review?",
        assertions=("the reply carries this run's crate count", "H is at or past the fact's ts",
                    "no history or search tool was recorded for the reply"),
        run=row_cross_thread_awareness),
    BatteryRow(
        name="verification-rule",
        trigger_template=("<@{bot_user_id}> {work} — which supplier did we go with in the end, "
                          "and what was the figure?"),
        assertions=("the bot answered the mention",
                    "a search or history tool was recorded for the reply",
                    "the reply names the supplier we chose",
                    "the reply carries this run's seeded figure"),
        run=row_verification_rule, slow=True),
    BatteryRow(
        name="cross-thread-action",
        trigger_template=("just heard back from {supplier} — their pallets hold {capacity} "
                          "crates each"),
        assertions=("the fyi opened a turn",
                    "exactly one post_to_thread landed under C",
                    "the post under C states the capacity we just supplied",
                    "A heard nothing, or only a brief non-reporting acknowledgment"),
        run=row_cross_thread_action),
    BatteryRow(
        name="search-to-action",
        trigger_template=("<@{bot_user_id}> finance settled the {subject} — {cap} a year, "
                          "effective now. that's the number the {work} was waiting on"),
        assertions=("a search or history tool name is recorded for the destination post",
                    "exactly one post_to_thread landed in the whole turn",
                    "that post landed, committed, under the target root",
                    "the origin heard nothing, or only a brief non-reporting acknowledgment",
                    "the posted answer carries the cap the trigger supplied"),
        run=row_search_to_action, slow=True),
    BatteryRow(
        name="full-origin-fidelity",
        trigger_template=("what was the opening tally on this again, and whose count was it?"),
        assertions=("the reply carries this run's opening tally",
                    "the reply names whose count it was",
                    "origin_count is root + seeded replies + the trigger"),
        run=row_full_origin_fidelity, slow=True),
    BatteryRow(
        name="stream-currency",
        trigger_template="what's the most recent message you can see in this channel?",
        assertions=("the frozen turn's H predates M", "the frozen turn's reply does not mention M",
                    "the next turn's H includes M"),
        run=row_stream_currency),
    BatteryRow(
        name="in-flight-exclusion",
        trigger_template=("<@{bot_user_id}> could you write up a proper explainer on our pallet "
                          "standards for the new hires?"),
        assertions=("A's own surface was in_flight when B rendered",
                    "A's own surface finalized before C",
                    "A predates B's horizon", "B excluded an in-flight receipt",
                    "C includes more receipts than B did", "the two membership hashes differ"),
        run=row_in_flight_exclusion),
    BatteryRow(
        name="re-anchor-observable",
        trigger_template="<@{bot_user_id}> you still with us?",
        assertions=("the build reselected", "the floor moved forward",
                    "root_count returned to the target"),
        run=row_re_anchor_observable, slow=True),
    BatteryRow(
        name="foreign-exchange-bait",
        trigger_template="<@{claude_user_id}> {bait}",
        assertions=("every Claude reply's turn was silent",
                    "our bot posted nothing in the exchange",
                    "our bot reacted to nothing in the exchange"),
        run=row_foreign_exchange_bait),
    BatteryRow(
        name="directed-banter-answered",
        trigger_template="<@{bot_user_id}> settle a bet for us — is a hot dog a sandwich?",
        assertions=("the bot responded when directly addressed",),
        run=row_directed_banter_answered),
    BatteryRow(
        name="thanks-response-choice",
        trigger_template="thanks!",
        assertions=(),
        run=row_thanks_response_choice, observation_only=True),
    BatteryRow(
        name="value-floor-holds",
        trigger_template="{aside}",
        assertions=("the turn stayed at or below a reaction", "no message was posted"),
        run=row_value_floor_holds),
    BatteryRow(
        name="render-equality-probe",
        trigger_template=("python3 -m tools.stream_probe --channel {channel} "
                          "--origin {A} --origin {B}"),
        assertions=("the two origins share their pre-breakpoint bytes",
                    "the two origins' unions differ"),
        run=row_render_equality_probe),
)

ROW_NAMES: Tuple[str, ...] = tuple(row.name for row in REGISTRY)


def row_by_name(name: str) -> Optional[BatteryRow]:
    for row in REGISTRY:
        if row.name == name:
            return row
    return None
