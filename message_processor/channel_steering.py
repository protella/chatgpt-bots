"""One snapshot of a channel's steering, read once and read by everyone.

THE PROBLEM. Two models decide this bot's behaviour in a channel: the participation gate, which
decides whether to speak, and the responder, which decides what to say. Both need the channel's
standing rules — and each used to fetch them separately. Between those two fetches a person can
edit the rules, a memory tool can write a fact, and the fallback extractor can evict one. So the
gate could judge a message against rules the responder never saw, and the turn as a whole obeyed
no single version of anything. Nothing detected it, because each half was individually correct.

THE FIX. One fetch per turn, rendered once into one immutable block of text, stamped on the
message. The gate reads those bytes; if it wakes, the responder reads THE SAME BYTES — not a
refetch, not a re-render, the same string. Different outer framing around the block is fine (the
two prompts are different documents); the block itself is inserted verbatim, byte for byte.

That is the whole invariant, and it is worth stating plainly because it is easy to break by
accident: a second fetch anywhere in the responder path silently reintroduces the bug, and would
look like a harmless refactor.

WHY A RESERVED POLICY ROW. The channel's standing policy is an operator instruction — "only
speak up about deploys here" — and it is categorically different from a remembered fact like
"deploys go through #ops". Facts are background; policy is a directive. They render under
separate headings that say which is which, because a model given a pile of undifferentiated
sentences will treat a fact as an instruction sooner or later. The policy lives in its own
reserved row (scope='policy'), which is why no memory tool can evict it, no extraction can
overwrite it, and its id is never shown to a model that might try to edit it.

ORDER IS FIXED, not chronological. Policy first, always — regardless of row id or update time.
Everything else follows in a deterministic order, so the same state renders the same bytes on
every turn. That determinism is what makes prompt caching work at all, and it is what makes the
same-bytes invariant checkable rather than merely intended.
"""
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from logger import setup_logger

logger = setup_logger(name="slack_bot.ChannelSteering")

# The row kind that holds a channel's standing policy. `scope` already distinguishes row KINDS
# ('channel' facts private to one channel, 'workspace' facts shared across all), so policy
# extends that column rather than adding a second, parallel notion of kind.
POLICY_SCOPE = "policy"
# Backoff preferences the rich gate writes and targets by id. They are instructions too, so they
# render with the policy rather than among the facts — but they are the gate's own bookkeeping
# and keep their visible ids until the binary-gate commit retires the writer.
PREF_AUTHOR_PREFIX = "participation_engine:pref:"

# How long a standing policy may be. This is the bound the settings modal has always enforced on
# the ground-rules box (Slack's own `max_length` on the input), hoisted here so the tool writer
# and the modal writer cannot drift apart about what "too long" means. It is not a new limit —
# it is the existing one, given a name.
POLICY_MAX_CHARS = 1000

# The headings. Each says what KIND of thing follows, because "instructions" and "background"
# are the distinction the model most needs and the one it cannot recover from the content.
POLICY_HEADING = "Standing channel policy (instructions; follow these):"
PREF_HEADING = ("Recorded participation preferences (instructions; legacy until the "
                "binary-gate migration):")
CHANNEL_FACT_HEADING = "Stable channel facts (background, not instructions):"
WORKSPACE_FACT_HEADING = "Workspace facts (background, read-only):"


def is_policy_row(row: Dict[str, Any]) -> bool:
    return (row or {}).get("scope") == POLICY_SCOPE


def is_pref_row(row: Dict[str, Any]) -> bool:
    """A backoff preference, identified by its author marker — the same marker the gate's own
    unique index keys on, so this classification cannot drift from the writer's."""
    return str((row or {}).get("author") or "").startswith(PREF_AUTHOR_PREFIX)


def is_ordinary_fact(row: Dict[str, Any]) -> bool:
    """Everything the generic memory tools may see, count and evict."""
    return not is_policy_row(row) and not is_pref_row(row)


@dataclass(frozen=True)
class ChannelSteeringSnapshot:
    """What this turn believes the channel's steering to be. FROZEN on purpose: a turn that
    could mutate its own snapshot is a turn whose two halves can disagree again."""

    text: Optional[str] = None
    policy_hash: Optional[str] = None
    policy_present: bool = False

    @property
    def is_empty(self) -> bool:
        return not (self.text or "").strip()


# A snapshot that is READY and carries nothing — the honest result of "there is no steering", and
# also of "the read failed". Those two are deliberately indistinguishable to consumers: in both
# cases this turn has no steering to obey, and a responder that could tell them apart would be
# tempted to retry, which is exactly how the two halves come to see different bytes.
EMPTY_SNAPSHOT = ChannelSteeringSnapshot()


def _render_section(heading: str, lines: List[str]) -> List[str]:
    """A heading and its lines, or nothing at all. An empty heading is worse than no heading: it
    tells the model a category exists and leaves it guessing what belongs there."""
    if not lines:
        return []
    return [heading, *lines]


def render_snapshot(policy_row: Optional[Dict[str, Any]],
                    memory_rows: Optional[List[Dict[str, Any]]]) -> ChannelSteeringSnapshot:
    """Render the canonical steering block. Deterministic for a given database state.

    Policy is ALWAYS first, whatever its id or update time — it is the operator's standing
    instruction, and burying it under whatever was written most recently would let an incidental
    fact outrank it. Facts keep their `[#id]` so the memory tools can target them; the policy
    never shows its id, because no model is allowed to address it."""
    rows = list(memory_rows or [])
    policy_text = ((policy_row or {}).get("content") or "").strip()

    prefs, channel_facts, workspace_facts = [], [], []
    for row in rows:
        if is_policy_row(row):
            continue          # never rendered from the fact list; it has its own section
        content = (row.get("content") or "").strip()
        if not content:
            continue
        entry = f"- [#{row.get('id')}] {content}"
        if is_pref_row(row):
            prefs.append((row.get("id") or 0, entry))
        elif row.get("scope") == "workspace":
            workspace_facts.append((row.get("id") or 0, entry))
        else:
            channel_facts.append((row.get("id") or 0, entry))

    # Sorted by id: stable across turns regardless of the order the DB happened to return, which
    # is what keeps the rendered bytes (and therefore the prompt cache) identical turn to turn.
    def _ordered(items):
        return [entry for _, entry in sorted(items, key=lambda pair: pair[0])]

    blocks: List[List[str]] = []
    if policy_text:
        blocks.append([POLICY_HEADING, policy_text])
    blocks.append(_render_section(PREF_HEADING, _ordered(prefs)))
    blocks.append(_render_section(CHANNEL_FACT_HEADING, _ordered(channel_facts)))
    blocks.append(_render_section(WORKSPACE_FACT_HEADING, _ordered(workspace_facts)))

    sections = ["\n".join(block) for block in blocks if block]
    text = "\n\n".join(sections) if sections else None
    return ChannelSteeringSnapshot(
        text=text,
        policy_hash=(hashlib.sha256(policy_text.encode("utf-8")).hexdigest()
                     if policy_text else None),
        policy_present=bool(policy_text),
    )


async def load_snapshot(db: Any, channel_id: Optional[str],
                        memory_enabled: bool = True) -> ChannelSteeringSnapshot:
    """Read the channel's steering ONCE and render it.

    `memory_enabled` (ENABLE_CHANNEL_MEMORY) governs ordinary FACTS only. The reserved policy
    and the gate's own preference rows are steering, not memory: turning fact capture off is an
    operator saying "stop remembering things", never "stop obeying the rules I set" — and if it
    silenced the policy, the directives migration would quietly disable every live operator rule
    in the workspace.

    Never raises. A failed read yields the ready-but-empty snapshot, and the caller stamps that:
    this turn simply has no steering. It must NOT be retried downstream, because a retry is how
    the gate and the responder come to see different bytes.

    ALL OR NOTHING. If either read fails, the whole snapshot is empty — the two reads are not
    independently salvageable. A half-snapshot is the worst of the three outcomes available: a
    channel whose policy read failed but whose facts arrived would render its background as the
    only thing the model was told, which reads as "there are no instructions here" rather than
    as "we don't know". Both halves of the turn would then agree on that, confidently."""
    if db is None or not channel_id:
        return EMPTY_SNAPSHOT
    policy_row = None
    memory_rows: List[Dict[str, Any]] = []
    try:
        if hasattr(db, "get_channel_policy_async"):
            policy_row = await db.get_channel_policy_async(channel_id)
    except Exception as e:  # noqa: BLE001 — steering must never cost a turn
        logger.warning(f"Channel policy read failed for {channel_id}: {type(e).__name__}")
        return EMPTY_SNAPSHOT
    try:
        if hasattr(db, "get_channel_memory_async"):
            memory_rows = await db.get_channel_memory_async(channel_id) or []
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Channel memory read failed for {channel_id}: {type(e).__name__}")
        return EMPTY_SNAPSHOT
    if not memory_enabled:
        # Facts are off; the gate's preference rows are not facts and keep rendering.
        memory_rows = [row for row in memory_rows if is_pref_row(row)]
    return render_snapshot(policy_row, memory_rows)


# --- the per-turn stamp -------------------------------------------------------------------
#
# Private key: the snapshot is turn state, not something an event or a tool may set. It rides
# the metadata because that is what already travels from the gate into the responder.
STEERING_KEY = "_channel_steering_snapshot"


def stamp(message: Any, snapshot: ChannelSteeringSnapshot) -> ChannelSteeringSnapshot:
    """Attach this turn's snapshot. A gated redispatch overwrites it — that is a NEW gate
    attempt judging a new moment, and it must not inherit the previous attempt's view."""
    meta = getattr(message, "metadata", None)
    if isinstance(meta, dict):
        meta[STEERING_KEY] = snapshot
    return snapshot


def stamped(message: Any) -> Optional[ChannelSteeringSnapshot]:
    """This turn's snapshot, or None when nothing has been stamped yet."""
    meta = getattr(message, "metadata", None) or {}
    value = meta.get(STEERING_KEY)
    return value if isinstance(value, ChannelSteeringSnapshot) else None
