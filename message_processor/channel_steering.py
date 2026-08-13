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

WHY THE SNAPSHOT IS SPLIT (P2, spec §3). One read still, one version still — but the two halves
are now addressable separately, because they belong at different places in a channel request and
under different authority. `developer_policy` is the operator's directive and rides the final
developer suffix; `user_facts` is remembered background and rides the post-breakpoint user
evidence, where untrusted-derived content belongs. `gate_text` is the flattened block the wake
classifier reads, and `.text` is the same bytes under its old name for the DM path and every
intermediate caller. Both are DERIVED from the halves, so the split cannot drift from what the
gate saw: tests/fixtures/channel_steering_golden.json pins those bytes against the pre-split
renderer.
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
# Backoff preference rows written by the RICH gate, which no longer exists. Nothing writes them,
# and the startup migration moves every surviving row into the reserved policy row and deletes it
# — a process that fails that migration refuses to start, so no live run can see one.
#
# The marker still earns its keep as a CLASSIFIER. `is_ordinary_fact` is what keeps a row of this
# kind out of the fact cap, out of the settings textarea, out of the memory tools and out of the
# fallback extractor; deleting it would mean a row that "cannot exist" would, if one ever did,
# arrive as an ordinary fact with an editable [#id] pointing at an operator instruction. A guard
# against an impossible state is cheap; discovering you needed it is not. The RENDERING is gone,
# though — a heading for a section that can never have content is a promise to the model about a
# kind of instruction this build cannot produce.
PREF_AUTHOR_PREFIX = "participation_engine:pref:"

# How long a standing policy may be. This is the bound the settings modal has always enforced on
# the ground-rules box (Slack's own `max_length` on the input), hoisted here so the tool writer
# and the modal writer cannot drift apart about what "too long" means. It is not a new limit —
# it is the existing one, given a name.
POLICY_MAX_CHARS = 1000

# The headings. Each says what KIND of thing follows, because "instructions" and "background"
# are the distinction the model most needs and the one it cannot recover from the content.
POLICY_HEADING = "Standing channel policy (instructions; follow these):"
CHANNEL_FACT_HEADING = "Stable channel facts (background, not instructions):"
WORKSPACE_FACT_HEADING = "Workspace facts (background, read-only):"
# Personal facts about the person on the other side of a DM. Its own heading because it is its
# own KIND: a channel fact is true of a room, this is true of one person, and only a DM turn is
# ever allowed to see it (owner ruling: DM facts stay out of channel turns, always).
USER_FACT_HEADING = "What you remember about this person (background, not instructions):"


def is_policy_row(row: Dict[str, Any]) -> bool:
    return (row or {}).get("scope") == POLICY_SCOPE


def is_pref_row(row: Dict[str, Any]) -> bool:
    """A backoff preference, identified by its author marker — the same marker the gate's own
    unique index keys on, so this classification cannot drift from the writer's."""
    return str((row or {}).get("author") or "").startswith(PREF_AUTHOR_PREFIX)


def is_ordinary_fact(row: Dict[str, Any]) -> bool:
    """Everything the generic memory tools may see, count and evict."""
    return not is_policy_row(row) and not is_pref_row(row)


# Bumped when the RENDERED bytes of either half change. It travels with the snapshot so a
# consumer (and the request-cache tuple) can say which grammar it obeyed.
STEERING_SNAPSHOT_VERSION = 1


@dataclass(frozen=True)
class ChannelSteeringSnapshot:
    """What this turn believes the channel's steering to be. FROZEN on purpose: a turn that
    could mutate its own snapshot is a turn whose two halves can disagree again.

    Stored as the two halves; every flattened form is derived (see the module docstring)."""

    developer_policy: Optional[str] = None
    user_facts: Optional[str] = None
    version: int = STEERING_SNAPSHOT_VERSION
    policy_hash: Optional[str] = None
    policy_present: bool = False

    @property
    def gate_text(self) -> Optional[str]:
        """The flattened block the wake classifier reads — policy first, then facts."""
        sections = [s for s in (self.developer_policy, self.user_facts) if s]
        return "\n\n".join(sections) if sections else None

    @property
    def text(self) -> Optional[str]:
        """The pre-split name for the same bytes. DMs and the intermediate callers that thread
        one string through the responder keep working unchanged."""
        return self.gate_text

    @property
    def is_empty(self) -> bool:
        return not (self.gate_text or "").strip()


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
                    memory_rows: Optional[List[Dict[str, Any]]],
                    user_rows: Optional[List[Dict[str, Any]]] = None) -> ChannelSteeringSnapshot:
    """Render the canonical steering block. Deterministic for a given database state.

    Policy is ALWAYS first, whatever its id or update time — it is the operator's standing
    instruction, and burying it under whatever was written most recently would let an incidental
    fact outrank it. Facts keep their `[#id]` so the memory tools can target them; the policy
    never shows its id, because no model is allowed to address it.

    `user_rows` (toolbelt T2) are the requester's PERSONAL facts and are passed only on a DM
    turn — `load_snapshot` is what refuses to fetch them anywhere else. They render last, under
    their own heading, in the same `- [#id]` form and id-sorted for the same prompt-cache reason.
    Their ids live in `user_memory`'s id space, and only a DM ever renders them, so the DM tools'
    `[#id]` and this block cannot disagree about which store an id names."""
    rows = list(memory_rows or [])
    policy_text = ((policy_row or {}).get("content") or "").strip()

    channel_facts, workspace_facts = [], []
    for row in rows:
        if is_policy_row(row):
            continue          # never rendered from the fact list; it has its own section
        content = (row.get("content") or "").strip()
        if not content:
            continue
        entry = f"- [#{row.get('id')}] {content}"
        if is_pref_row(row):
            # Cannot happen after a successful start (see PREF_AUTHOR_PREFIX). If one somehow
            # does, it stays out of the prompt rather than being rendered as a fact the model
            # could edit — inert, not half-present.
            continue
        elif row.get("scope") == "workspace":
            workspace_facts.append((row.get("id") or 0, entry))
        else:
            channel_facts.append((row.get("id") or 0, entry))

    # Sorted by id: stable across turns regardless of the order the DB happened to return, which
    # is what keeps the rendered bytes (and therefore the prompt cache) identical turn to turn.
    def _ordered(items):
        return [entry for _, entry in sorted(items, key=lambda pair: pair[0])]

    personal_facts = []
    for row in (user_rows or []):
        content = (row.get("content") or "").strip()
        if content:
            personal_facts.append((row.get("id") or 0, f"- [#{row.get('id')}] {content}"))

    fact_blocks = [
        _render_section(CHANNEL_FACT_HEADING, _ordered(channel_facts)),
        _render_section(WORKSPACE_FACT_HEADING, _ordered(workspace_facts)),
        _render_section(USER_FACT_HEADING, _ordered(personal_facts)),
    ]
    fact_sections = ["\n".join(block) for block in fact_blocks if block]
    return ChannelSteeringSnapshot(
        developer_policy=("\n".join([POLICY_HEADING, policy_text]) if policy_text else None),
        user_facts=("\n\n".join(fact_sections) if fact_sections else None),
        policy_hash=(hashlib.sha256(policy_text.encode("utf-8")).hexdigest()
                     if policy_text else None),
        policy_present=bool(policy_text),
    )


def _is_dm_surface(channel_id: Optional[str]) -> bool:
    """The one discriminator the whole build already uses, imported locally to avoid the cycle
    (slack_client imports message_processor). A failure to classify answers False: personal facts
    not shown is a degraded turn, personal facts shown in a channel is a leak."""
    try:
        from slack_client.utilities import is_dm_conversation
        return is_dm_conversation(channel_id)
    except Exception:  # noqa: BLE001
        return False


async def load_snapshot(db: Any, channel_id: Optional[str],
                        memory_enabled: bool = True,
                        user_id: Optional[str] = None,
                        user_memory_enabled: bool = True) -> ChannelSteeringSnapshot:
    """Read the channel's steering ONCE and render it.

    `user_id` + `user_memory_enabled` (ENABLE_USER_MEMORY) add the requester's PERSONAL facts,
    and only ever on a DM: the DM check below is unconditional, so a caller that passes a
    `user_id` on a channel turn still gets no user facts. That is the owner's ruling made
    structural rather than a rule every call site has to remember — personal memory is written
    in a DM and read in a DM, and a channel turn must never surface it.

    `memory_enabled` (ENABLE_CHANNEL_MEMORY) governs ordinary FACTS only. The reserved policy is
    steering, not memory: turning fact capture off is an operator saying "stop remembering
    things", never "stop obeying the rules I set" — and if it silenced the policy, the directives
    migration would quietly disable every live operator rule in the workspace.

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
        # Facts are off. Nothing else in this list is steering any more — the policy comes from
        # its own row — so there is nothing left to render from it.
        memory_rows = []
    user_rows: List[Dict[str, Any]] = []
    if user_id and user_memory_enabled and _is_dm_surface(channel_id):
        try:
            if hasattr(db, "get_user_memory_async"):
                user_rows = await db.get_user_memory_async(user_id) or []
        except Exception as e:  # noqa: BLE001 — same all-or-nothing contract as the two above
            logger.warning(f"User memory read failed for {user_id}: {type(e).__name__}")
            return EMPTY_SNAPSHOT
    return render_snapshot(policy_row, memory_rows, user_rows)


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
